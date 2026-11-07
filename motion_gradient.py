"""
Gradient-based inter-slice motion correction of cine MRI: the motion-correction library. It is not run on its own;
the entry points are the data loaders, which read a subject, hand it over and write the result in their own format:
  moco_nifti.py   UK Biobank NIfTI (SAX stack and/or slices + LAX views, nnU-Net segmentations)
  moco_dicom.py   DICOM series (any number of stacks / orientations)

Input (from a loader, see SliceImage / Subject): a list of 2D+time cine slices, each (X, Y, T) with its own 4x4
affine (voxel (i, j, 0) -> world mm, one world frame per subject), image (stack) name and index in its stack, slice
thickness and an optional (X, Y, T) segmentation, its view and modality. Slices of view "SAX" are short-axis slices
(SAX regularizers, QC and reports). Every slice is a plane with its own pose; no regular stack geometry is assumed. Segmentations are optional:
a non-SAX image's segmentation defines its dilated heart mask (the line points scored); without one its lines run
over the whole overlap with the other image, or with --center-mask-radius-mm over a disc around the centre of its stack
(dilated and annealed the same way). Segmentations are also used for the reported Dice / Hausdorff.
Output: per slice its corrected affine (corrected_affines), written back by the loader; metrics and figures below.
run() is the shared run loop (subject scanning, background loading/writing, CSVs, figures).

Aortic slices (UKBB ao.nii.gz, is_aorta) are LAX views in every respect when present (--use-ao): their lines with the
SAX slices are SAX-LAX lines, those with the other LAX views LAX-LAX lines, and they count in the LAX reference frame
and statistics. Only the LV long axis (heart_axis) leaves them out. Having no segmentation, their lines are masked by
the partner: with a LAX view by its dilated heart mask (as any LAX-LAX line), with a SAX slice by the SAX-SAX cylinder
(points within --sax-sax-radius-mm of the LV long axis; see build_lines).

Pairs are decided by orientation alone, whatever the views and modalities. Intersection pairs (candidate_pairs): every
two slices whose acquired normals differ by at least --min-pair-angle-deg, thinned between multi-slice stacks to
partners about a slice thickness apart (--pair-spacing-mm). Parallel pairs (attach_sax_pairs): two slices closer in
angle than that and within --parallel-max-dist-mm along the normal -- adjacent slices of a stack, and between images
each slice and its nearest parallel slice of every other image -- get the neighbouring-slice crossing terms
(--sax-sax-weight, --sax-gap-reg; see SaxNeighbourLines); those whose slabs overlap (near-coplanar, e.g. one view in two
modalities) are also compared over their shared area with --coplanar-weight (CoplanarPairs). Lines and parallel pairs
within one modality use --metric, between modalities --cross-mod-metric. The LV long axis (rotation
pivots, SAX-SAX terms) is the crossing line of the two most perpendicular single-slice LAX views, or without such views
the axis of the SAX stack (else of the first image found).

For every pair of non-parallel planes (each SAX slice vs each LAX view, and LAX vs LAX) the 3D
intersection line is computed from the slice affines. Intensity profiles are sampled along the
line from both images; each profile is averaged across a band whose width equals the *other*
image's slice thickness, approximating that image's slice-selection point-spread function.
Profiles are compared with normalized cross-correlation (NCC; or gradient NCC, local NCC, normalized
gradient agreement, normalized MI, or L1 / L2 on percentile-normalized intensities, --metric), restricted to a dilated heart
mask derived from the LAX nnU-Net segmentations (or, with --outside-mask-weight, with the points
outside the mask down-weighted) and averaged over several cardiac phases.

A 6-DOF rigid pose per plane (rotation vector about a pivot on the LV long axis + 3D translation of that pivot,
--rotation-pivot; so rotation does not move the heart and the two are decoupled) is
optimized jointly for all planes with Adam (or another --optimizer), starting from the acquired geometry. Intersection
lines are recomputed from the current poses every step, so through-plane shifts and rotations
are observable. A weak gauge penalty on the mean pose removes the global rigid-body freedom.
Optional SAX stack regularizers act on each pair of adjacent slices: --tilt-hinge-reg penalizes
their relative tilt above --tilt-hinge-deg, --spacing-reg the deviation of their distance along
the normal from --sax-spacing-mm (signed, so slices cannot swap order), --rel-shift-reg their
relative in-plane offset and --rel-rot-reg their relative in-plane rotation; all are zero for a
stack that moves as one rigid body. Two more terms keep neighbouring slices of any stack (SAX or
e.g. the oblique DICOM stacks) from crossing inside the heart (see SaxNeighbourLines):
--sax-gap-reg, a barrier on the gap between them on a disk around the LV long axis (other stacks:
their own axis), and --sax-sax-weight, a dissimilarity cost along their crossing line where they
do cross.
Optionally (--blur-sigma) the optimization is coarse-to-fine: the images are blurred in-plane at
the start and the blur decays linearly to zero over the first --blur-frac of the steps; the best
pose is only tracked on the remaining, unblurred steps (--blur-frac 0: constant blur, tracked
throughout). Optionally (--pair-keep < 1) each step
drops a random subset of the intersection lines from the objective, which makes the optimization
stochastic; the best pose is then tracked on the full objective evaluated every --eval-every steps.

Any image may be missing (see pose_model). A LAX view takes part only with a segmentation containing the heart (it
defines the mask). The full 6-DOF pose needs SAX slices and >= 2 such LAX views; with SAX slices and a single LAX
view, or LAX views only, every plane is crossed along a single direction and its tilt is unconstrained, so only the
translation and the rotation about the plane's own normal are optimized (summary column pose_model: inplane). Without
any crossing pair (SAX only, a single LAX view, ...) nothing is optimized (status not_optimized). Every image present
is written to the output either way, unaligned ones with their acquired affine.

Optionally (--edge-weight) an edge term is blended into every line's similarity late in the run: the agreement of
the signed derivatives of both profiles along the line (--edge-metric), which favours narrow, high-contrast edges
over long stretches of slight variation (see LineSampler.similarity_lines).

With --logging, segmentation consistency (Dice between the two views' labels along the same lines) is
reported before and after alignment as an independent check; it is not optimized. Also over the long-axis
segmentations only: per view the Dice over its LAX-LAX lines (lax_dice), and on the LAX-LAX lines along the LV long
axis how far apart the two views' apices are and how well their apical myocardium overlaps (LineSampler.apex_lines).

Outputs go to <out-root>/<run-name>/ (default run name: the start time, then the settings given on the command line,
so a rerun never writes into an earlier run). Always written:
  <subject>/
    the motion-corrected images, written by the loader in its own format and layout (see moco_nifti.py,
                              moco_dicom.py; not with --save-images false).
Always:
  overall_metrics/
    run_config.json           every setting of the run
    sax_relative_motion.csv   estimated SAX motion relative to the other slices (see sax_relative_motion): one
                              'slice' row per slice against the stack consensus, one 'pair' row per neighbouring pair
Only with --logging (otherwise nothing else is computed or written, which is the fastest mode):
  overall_metrics/
    motion_summary.csv       one row per subject (append-only, latest row per subject wins)
    motion_summary_slices.csv one row per SAX slice / LAX view
    motion_summary_pairs.csv  one row per intersection-line pair (NCC, LNCC, NMI, edge, Dice, Hausdorff and the
                              apex metrics before/after; with
                              --sax-sax-weight also SAX-SAX rows: NCC -- between modalities --cross-mod-metric --,
                              counting points and axis distance at the end)
    motion_summary_regions.csv one row per subject and combination of slice types (LAX-LAX, each pair of long-axis
                              views, each long-axis view x SAX region; region_rows): NCC, LNCC and the LV-only Dice
                              before/after
    motion_*.png              dataset-level figures (written at the end of a run)
  <subject>/
    metrics/
      summary.csv, slices.csv, pairs.csv, regions.csv  this subject's rows
      history.csv             pose snapshots every 2% of the run (step 0 = acquired geometry, and the
                              saved "best" pose with is_best=1): per plane the full 3x4 affine (a00..a23, same
                              convention as the saved images), the pose and NCC/LNCC/NMI/Dice/Hausdorff, the
                              LAX-LAX-only Dice and the apex distance
      history_summary.csv     the same snapshots, one row per step with the objective and subject-level metrics
      lines.csv               every intersection line entering the loss (acquired geometry): the two planes, the
                              anchor point x0 and unit direction d (world, mm) and the parameter ranges along the
                              line that are inside both images (t_img_*) and actually sampled (t_used_*: inside the
                              mask -- for SAX-aortic lines the cylinder of --sax-sax-radius-mm around the LV long axis)
      sax_lines.csv           the crossing line of every pair of neighbouring slices of a stack (kind SAX-SAX for the
                              SAX slices, STACK-STACK for any other stack, image_a / image_b its image) at every history snapshot
                              (recorded whatever --sax-sax-weight is): these lines exist only when the poses tilt the
                              slices against each other, so they are stored per step (same columns as history.csv's
                              step/fraction/is_best), in the lines.csv convention (x0 + t d, t_img_* inside both images,
                              t_used_* also within --sax-sax-radius-mm of the long axis) and in history.csv's world
                              frame (relative to the mean LAX pose); crossing = 0 / NaN geometry where the slices are
                              parallel; supervised = 1 where the line entered the loss (--sax-sax-weight > 0 and points
                              with weight); QC per pair and snapshot: n_points (sum of the point weights), ncc and
                              dissim (1 - NCC) along the line, gap_min_mm (smallest gap between the two slices on the
                              --sax-gap-radius-mm disk, < 0 = they cross inside it), rel_tilt_deg, crossing_in_heart.
                              The same QC at the final pose: the SAX-SAX rows of pairs.csv (gap_min_mm, rel_tilt_deg)
                              and the summary columns sax_cross_pairs (pairs crossing inside the disk), sax_gap_min_mm,
                              sax_rel_tilt_max_deg and sax_cross_dissim_max (worst 1 - NCC of a crossing pair within a
                              modality).
      long_axis.csv           the LV long axis the SAX-SAX lines are (or would be) supervised around, at every
                              history snapshot in the frame of history.csv / sax_lines.csv: a point (p_*), the unit
                              direction (d_*) and --sax-sax-radius-mm (the axis is fixed in the optimizer's frame, which
                              history.csv shifts by the mean LAX pose, so it moves from snapshot to snapshot here)
      mask_contours.csv       outer contour of each LAX view's dilated heart mask, in voxel coordinates (i, j)
      motion_subject.png      motion histograms, per-pair similarity histograms, per-slice translations
      motion_pairs.png        per-pair NCC / NMI / Dice / Hausdorff, split SAX-LAX and LAX-LAX
      mask_overlay.png        dilated heart mask and the line points entering the loss

dx_mm / dy_mm are translations along the image's own row (i) and column (j) axes, dz_mm along
its normal, that must be applied to the image content to bring it into agreement with the other
views (translations are expressed relative to the mean LAX pose). rot_inplane_deg is the
rotation about the slice normal (well constrained by three crossing lines); rot_tilt_deg the
rotation about in-plane axes (weakly constrained, treat with caution).
"""
import argparse
import contextlib
import csv
import math
import re
import sys
import time
import traceback
from collections import deque
import dataclasses
import multiprocessing
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

# Aortic images (UKBB ao.nii.gz: one axial slice through the ascending aorta, no segmentation). They are LAX views
# (--use-ao): paired with the SAX slices (SAX-LAX) and the other LAX views (LAX-LAX), and part of the LAX reference frame
# and statistics. Their lines are masked by the partner: a LAX view's dilated heart mask, or for a SAX slice the SAX-SAX
# cylinder around the LV long axis (--sax-sax-radius-mm; build_lines). They never define the LV long axis (heart_axis):
# an axial slice crosses the 2Ch view at a larger angle than the 4Ch view does, so it would win long_axis_line.
AORTA_NAME = re.compile(r"^ao($|[_\-. ])|aort", re.I)


def is_aorta(image: str) -> bool:
    """Whether an image name denotes an aortic slice (e.g. 'ao', 'ao_1', 'aorta')."""
    return bool(AORTA_NAME.search(image))


def lax_reference_rows(planes: list, n_pairs: list[int]) -> list[int]:
    """Rows of the planes whose mean pose is the output frame (relative_translation): the aligned non-SAX planes
    (aortic slices included)."""
    return [i for i, p in enumerate(planes) if p.view != "SAX" and n_pairs[i] > 0]


SUMMARY_COLUMNS = [
    "subject_id", "status", "pose_model", "n_lax_views", "sax_n_slices", "sax_n_eval",
    "sax_shift_mean_mm", "sax_shift_rms_mm", "sax_shift_max_mm", "sax_shift_jitter_mm",
    "sax_dz_rms_mm", "sax_rot_inplane_rms_deg", "sax_tilt_rms_deg",
    "lax_shift_mean_mm", "lax_shift_max_mm",
    "ncc_zero_mean", "ncc_mean", "ncc_gain_mean",
    "lncc_zero_mean", "lncc_mean", "lncc_gain_mean",
    "nmi_zero_mean", "nmi_mean", "nmi_gain_mean",
    "edge_zero_mean", "edge_mean",
    "seg_dice_zero_mean", "seg_dice_mean",
    "hd_zero_mean_mm", "hd_mean_mm", "hd_zero_max_mm", "hd_max_mm",
    "lax_dice_zero_mean", "lax_dice_mean",
    "apex_zero_mean_mm", "apex_mean_mm", "apex_zero_max_mm", "apex_max_mm",
    "apex_dice_zero_mean", "apex_dice_mean",
    "sax_cross_pairs", "sax_gap_min_mm", "sax_rel_tilt_max_deg", "sax_cross_dissim_max",
    "coplanar_pairs", "coplanar_sim_zero", "coplanar_sim",
    "n_line_pairs", "steps", "seconds", "timestamp",
]
DETAIL_COLUMNS = [
    "subject_id", "image", "view", "modality", "slice", "sax_region", "n_pairs", "n_points",
    "dx_mm", "dy_mm", "dz_mm", "shift_inplane_mm", "shift_mm", "rot_inplane_deg", "rot_tilt_deg",
    "ncc_zero", "ncc", "lncc_zero", "lncc", "nmi_zero", "nmi", "seg_dice_zero", "seg_dice", "hd_zero_mm", "hd_mm",
    "lax_dice_zero", "lax_dice", "apex_zero_mm", "apex_mm", "apex_dice_zero", "apex_dice",
]
PAIR_COLUMNS = [
    "subject_id", "image_a", "slice_a", "image_b", "slice_b", "view_a", "view_b", "kind", "n_points",
    "ncc_zero", "ncc", "lncc_zero", "lncc", "nmi_zero", "nmi", "edge_zero", "edge",
    "seg_dice_zero", "seg_dice", "hd_zero_mm", "hd_mm",
    "apex_zero_mm", "apex_mm", "apex_endo_zero_mm", "apex_endo_mm", "apex_dice_zero", "apex_dice",
    "axis_dist_mm", "gap_min_mm", "rel_tilt_deg",
]
# LAX-LAX apex metrics (LineSampler.apex_lines): on the lines between two long-axis views that run along the LV long
# axis, where each view's segmentation ends toward the apex, and how well the apical myocardium overlaps
APEX_LV_LABEL, APEX_MYO_LABEL = 1, 2     # LV blood pool, LV myocardium (the UKBB LAX labels; atria 4 / 5, see apex_direction)
APEX_MIN_COS = 0.5        # an apex line runs within 60 deg of the base -> apex direction (the 2Ch / 3Ch / 4Ch lines)
APEX_WINDOW_MM = 30.0     # apex_dice: the stretch of the line within this distance of the more apical of the two apices
# Metrics per combination of slice types (region_rows): the LV labels, the same in every UKBB view (the others are not:
# label 3 is the LA in the 2Ch view but the RV in 3Ch / 4Ch / SAX; 5 is the RA in 4Ch, probably the aorta in 3Ch),
# and the SAX regions along the LV long axis (assign_sax_regions), from base to apex
LV_LABELS = (APEX_LV_LABEL, APEX_MYO_LABEL)
SAX_REGIONS = ("pre_basal", "basal", "mid", "apical", "post_apical")
REGION_COLUMNS = ["subject_id", "combination", "image_a", "image_b", "sax_region", "n_lines", "n_points",
                  "ncc_zero", "ncc", "lncc_zero", "lncc", "dice_zero", "dice"]


@dataclass
class Plane:
    key: str
    image: str
    slice_idx: int
    img: torch.Tensor                # (F, 1, H, W)
    seg: torch.Tensor | None         # (F, 1, H, W) integer labels as float
    mask: torch.Tensor | None        # (1, 1, H, W) dilated heart mask
    inv_rot: torch.Tensor            # (3, 3) world -> voxel
    inv_trans: torch.Tensor          # (3,)
    origin: np.ndarray
    center: np.ndarray
    u_hat: np.ndarray
    v_hat: np.ndarray
    normal: np.ndarray
    thickness: float
    spacing: tuple = (1.0, 1.0)      # in-plane voxel size (mm) along image axes 0 and 1
    affine: np.ndarray | None = None  # acquired 4x4 voxel -> world affine of this slice
    pivot: np.ndarray | None = None   # point (world, in the plane) the pose rotates about; None = the image centre
    stack_rank: int = 0              # position of the plane's image in the order the images were found
    fvalid: torch.Tensor | None = None   # (F,) bool: the frames the slice has (None: all); a single-frame slice
                                         # (end-diastole) only frame 0, its image copied into the others
    view: str = ""                   # the slice's view (SliceImage.view): "SAX", "LAX_4Ch", "ao", ...
    modality: str = ""               # the acquisition it belongs to (SliceImage.modality)
    apex_dir: np.ndarray | None = None   # the subject's base -> apex unit vector (world; prepare_subject), for the
                                         # LAX-LAX apex metrics; None if unknown
    region: str = ""                 # a SAX slice's region along the LV long axis (assign_sax_regions; SAX_REGIONS),
                                     # "" for other planes or when unknown

    @property
    def rot_centre(self) -> np.ndarray:
        """The point the plane's pose rotation is about (see set_rotation_pivots)."""
        return self.center if self.pivot is None else self.pivot

    @property
    def H(self):
        return self.img.shape[2]

    @property
    def W(self):
        return self.img.shape[3]


@dataclass
class Line:
    t: torch.Tensor                  # (L,) signed distance along the line from its anchor point
    w: torch.Tensor                  # (L,) 1 where the point is valid (inside both images and the mask)
    n_points: float
    w_loss: torch.Tensor | None = None   # (L,) loss weights if they differ from w (--outside-mask-weight > 0)
    x0: np.ndarray | None = None     # anchor point (world, mm) at the acquired geometry
    d: np.ndarray | None = None      # unit direction (world)


# --------------------------------------------------------------------------------------
# Input: what a data loader hands over
# --------------------------------------------------------------------------------------

@dataclass
class SliceImage:
    """One acquired 2D cine slice, as handed over by a data loader (moco_nifti.py, moco_dicom.py).

    data[i, j, t] is the intensity of pixel (i, j) in frame t, located at world point affine @ [i, j, 0, 1] (mm, one
    world frame for all slices of a subject; the loaders use RAS). Slices of one image (stack) share `image` and are
    numbered by `index` along the stack; `image` identifies the stack (unique within the subject). `view` says what
    the slice shows -- "SAX" marks short-axis slices, which get the SAX-specific regularizers, QC and reports; aortic
    slices are recognized by it (is_aorta) -- and `modality` which acquisition it belongs to (e.g. its folder: cine,
    t1, lge, ...): intersection lines and parallel pairs between slices of different modalities are scored with
    --cross-mod-metric instead of --metric. All slices of one non-SAX image must have the same in-plane size.

    A slice may have a single frame (T = 1), taken as end-diastole (frame 0): it is compared with the other slices at
    that frame only (prepare_subject, LineSampler.FW). Its segmentation may also have a single frame (end-diastole)
    next to a cine of T frames."""
    image: str                      # image (stack) the slice belongs to, e.g. "SAX", "LAX_4Ch", a series name
    index: int                      # position in its stack (0, 1, ...)
    data: np.ndarray                # (X, Y, T) float32 full cine, T >= 1 (T = 1: end-diastole only)
    affine: np.ndarray              # (4, 4) voxel (i, j, 0) -> world mm
    thickness: float                # slice thickness (mm), for the slice-profile PSF
    seg: np.ndarray | None = None   # (X, Y, T) or (X, Y, 1) (end-diastole) integer labels (optional; heart mask and
                                    # Dice / Hausdorff)
    source: object = None           # loader-specific handle for writing the slice back (never read here)
    view: str = ""                  # "SAX", "LAX_2Ch", "LAX_3Ch", "LAX_4Ch", "ao", any other name (default: the image)
    modality: str = ""              # the acquisition (one per modality folder; "" when there is one)

    def __post_init__(self):
        if not self.view:
            self.view = self.image

    @property
    def key(self) -> str:
        return f"{self.image}:{self.index}"


@dataclass
class Subject:
    subject_id: str
    slices: list                    # list[SliceImage], images in the order they were found
    status: str = "ok"              # "ok", or e.g. "no_images" (nothing to align)


def phase_frames(n_frames_available: int, n_frames: int) -> list[int]:
    """The frames of an n_frames_available-frame cine at n_frames evenly spaced phases of the cardiac cycle: frame k
    -> floor(k * n_frames_available / n_frames), in integer arithmetic (the float k / n * N lands just below an
    integer for some k, e.g. k = 29 of 50, which repeated a frame and dropped one)."""
    return sorted({k * n_frames_available // n_frames for k in range(n_frames)})


def stack_center(affines: np.ndarray, shape: tuple) -> np.ndarray:
    """World centre (homogeneous, (4,)) of a stack: the mean of its slices' image centres, acquired geometry."""
    mid = np.array([(shape[0] - 1) / 2.0, (shape[1] - 1) / 2.0, 0.0, 1.0])
    return np.mean([A @ mid for A in affines], 0)


def center_disc_dist(affines: np.ndarray, shape: tuple, radius_mm: float) -> np.ndarray:
    """(X, Y, Z) distance (mm) of every pixel of a stack's slices to a disc of radius_mm centred where the stack's
    centre (stack_center) projects onto each slice; 0 inside the disc. The default mask of an image without a
    segmentation (--center-mask-radius-mm), dilated and annealed like a heart mask (dist <= dilation)."""
    X, Y = shape
    c = stack_center(affines, shape)
    ii, jj = np.meshgrid(np.arange(X), np.arange(Y), indexing="ij")
    out = np.empty((X, Y, len(affines)), np.float32)
    for z, A in enumerate(affines):
        ci, cj = (np.linalg.inv(A) @ c)[:2]
        su, sv = np.linalg.norm(A[:3, 0]), np.linalg.norm(A[:3, 1])
        out[:, :, z] = np.maximum(np.hypot((ii - ci) * su, (jj - cj) * sv) - radius_mm, 0.0)
    return out


def apex_direction(sax: list[tuple[np.ndarray, tuple, np.ndarray | None]],
                   lax: list[tuple[np.ndarray, np.ndarray | None]] = ()) -> np.ndarray | None:
    """Unit vector (world) from the base of the SAX stack to its apex, or None with fewer than two slices.
    sax: per SAX slice in stack order (affine, in-plane shape (X, Y), segmentation or None); lax: per LAX view
    (affine, segmentation (X, Y, ...) or None). The stack axis is oriented by anatomy: in the LAX segmentations the atria
    (labels 4, 5) lie at the base and the LV blood pool (label 1) toward the apex, so the apex is on the side of the LV
    centroid relative to the atria centroid (summed over the views). Without both in any view: the UKBB convention,
    slice 0 basal (the anatomy agreed with it on 22 of 22 test subjects; the SAX segmented area does not tell, as the
    basal-most slices lie above the ventricles and are empty)."""
    if len(sax) < 2:
        return None
    centres = [(A @ np.array([(sh[0] - 1) / 2.0, (sh[1] - 1) / 2.0, 0.0, 1.0]))[:3] for A, sh, _ in sax]
    d = centres[-1] - centres[0]
    if np.linalg.norm(d) < 1e-6:
        return None
    d = d / np.linalg.norm(d)
    vote = 0.0
    for v in lv_atria_vectors(lax):
        vote += float(v @ d)
    return -d if vote < 0 else d


def lv_atria_vectors(lax: list[tuple[np.ndarray, np.ndarray | None]]) -> list[np.ndarray]:
    """Per LAX view (affine, segmentation (X, Y, ...) or None) whose segmentation has both: the LV blood pool (label 1)
    centroid minus the atria (labels 4, 5) centroid (world mm), a vector from the base toward the apex."""
    out = []
    for A, g in lax:
        if g is None:
            continue
        g = np.asarray(g)
        g = g.reshape(g.shape[0], g.shape[1], -1).max(-1)
        cs = []
        for m in (g == 1, np.isin(g, (4, 5))):
            ij = np.argwhere(m)
            cs.append(None if len(ij) < 20 else (np.asarray(A, float) @ np.r_[ij.mean(0), 0.0, 1.0])[:3])
        if cs[0] is not None and cs[1] is not None:
            out.append(cs[0] - cs[1])
    return out


def subject_apex_direction(sax: list[tuple[np.ndarray, tuple, np.ndarray | None]],
                           lax: list[tuple[np.ndarray, np.ndarray | None]] = ()) -> np.ndarray | None:
    """The base -> apex unit vector (world) for the LAX-LAX apex metrics: the SAX stack axis oriented by anatomy
    (apex_direction), or without a SAX stack the mean LV-minus-atria vector of the LAX segmentations (lv_atria_vectors;
    within a few tens of degrees of the long axis, enough to tell which end of a long-axis line is apical). None if
    neither is available."""
    d = apex_direction(sax, lax)
    if d is not None:
        return d
    vecs = lv_atria_vectors(lax)
    v = np.sum(vecs, 0) if vecs else np.zeros(3)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-6 else None


def _shift2d(a: np.ndarray, di: int, dj: int) -> np.ndarray:
    """a shifted by (di, dj) pixels, zero-filled."""
    out = np.zeros_like(a)
    X, Y = a.shape
    if abs(di) >= X or abs(dj) >= Y:
        return out
    out[max(di, 0):X + min(di, 0), max(dj, 0):Y + min(dj, 0)] = a[max(-di, 0):X - max(di, 0), max(-dj, 0):Y - max(dj, 0)]
    return out


def apical_sweep(heart: np.ndarray, affine: np.ndarray, apex: np.ndarray | None, extend_mm: float) -> np.ndarray:
    """The heart region (X, Y) of a long-axis slice swept toward the apex by extend_mm: the union of its copies
    shifted 0 ... extend_mm along the in-plane projection of `apex` (the base-to-apex direction), at most one pixel
    apart. The dilated mask measured from it (--mask-dilation-mm, annealed) then reaches extend_mm further past the
    apex, whatever the segmentation does there (apical segmentations are often incomplete). Unchanged when the slice
    plane is far from containing the long axis (projection < half), or without a direction."""
    if apex is None or extend_mm <= 0 or not heart.any():
        return heart
    M = np.asarray(affine, float)[:3, :3]
    n = np.cross(M[:, 0], M[:, 1])
    n /= np.linalg.norm(n)
    u = apex - (apex @ n) * n
    if np.linalg.norm(u) < 0.5:
        return heart
    u /= np.linalg.norm(u)
    v = np.array([u @ M[:, 0] / (M[:, 0] @ M[:, 0]), u @ M[:, 1] / (M[:, 1] @ M[:, 1])])   # pixels per mm along u
    steps = max(1, int(math.ceil(extend_mm * np.abs(v).max())))
    out = heart.copy()
    for k in range(1, steps + 1):
        di, dj = np.rint(k / steps * extend_mm * v).astype(int)
        out |= _shift2d(heart, int(di), int(dj))
    return out


def prepare_subject(subject: Subject, args) -> dict:
    """The internal form of a subject: --n-frames evenly spaced frames per slice (the same number for every slice;
    at most the fewest frames of a multi-frame slice). A single-frame slice is end-diastole, the first of these phases:
    its frame is copied into every phase (the images of a subject share one volume) but it is valid at the first only
    (fvalid, (Z, F) per image), and it is compared with the other slices there only. Segmentations follow their slice's
    phases; a single-frame one (end-diastole) gives the first phase, -1 (no labels) the others.
    SAX slices one by one, every other image as a stack (X, Y, Z, F) with per-slice affines (Z, 4, 4), and, for images
    with a segmentation, the dilated heart mask (--mask-dilation-mm) and the distance to the heart (mm, for mask
    annealing) per slice; with --apex-extend-mm the heart region is first swept toward the apex (apical_sweep). A non-SAX image whose segmentation contains no heart is left out of the alignment (it is
    still written, unaligned, by the loader); one without a segmentation takes part unmasked, or with
    --center-mask-radius-mm > 0 masked by a disc around its stack's centre (center_disc_dist)."""
    out = dict(subject_id=subject.subject_id, status=subject.status, sax=[], lax={}, order=[])
    if not subject.slices:
        out["status"] = "no_images"
        return out
    multi = [s.data.shape[-1] for s in subject.slices if s.data.shape[-1] > 1]
    n_f = min([args.n_frames] + multi) if multi else 1

    def frames(a):
        """A slice's cine at the n_f phases; a single frame copied into all of them."""
        if a.shape[-1] == 1:
            return np.repeat(a, n_f, -1)
        return a[..., phase_frames(a.shape[-1], n_f)]

    def valid(s):
        """(n_f,) the phases slice s has: all, or (single frame: end-diastole) the first."""
        v = np.ones(n_f, bool)
        if s.data.shape[-1] == 1:
            v[1:] = False
        return v

    def seg_frames(s):
        """Slice s's segmentation at its phases; a single-frame one (or that of a single-frame slice) at the first,
        -1 (no labels) at the others."""
        g = s.seg
        if g.shape[-1] == s.data.shape[-1] > 1:
            return frames(g)
        out = np.full(g.shape[:2] + (n_f,), -1, np.int16)
        out[..., 0] = g[..., 0]
        return out

    for s in subject.slices:
        if s.image not in out["order"]:
            out["order"].append(s.image)
    # the SAX slices, one entry each: every SAX image (one per modality), in the order found, by index
    for s in sorted((s for s in subject.slices if s.view == "SAX"), key=lambda s: (out["order"].index(s.image), s.index)):
        out["sax"].append(dict(index=s.index, arr=frames(s.data)[:, :, None].astype(np.float32), affine=s.affine,
                               seg=None if s.seg is None else seg_frames(s)[:, :, None], thickness=s.thickness,
                               fvalid=valid(s)[None], image=s.image, view=s.view, modality=s.modality))
    extend = getattr(args, "apex_extend_mm", 0.0)
    sax_images = [e["image"] for e in out["sax"]]
    main_sax = max(dict.fromkeys(sax_images), key=sax_images.count) if sax_images else None   # the largest SAX stack
    sax_axis = [(e["affine"], e["arr"].shape[:2], e["seg"]) for e in out["sax"] if e["image"] == main_sax]
    lax_segs = [(s.affine, s.seg) for s in subject.slices if s.view != "SAX"]
    apex = apex_direction(sax_axis, lax_segs) if extend > 0 else None      # the apical sweep (needs a SAX stack)
    out["apex_dir"] = subject_apex_direction(sax_axis, lax_segs)          # the LAX-LAX apex metrics
    for image in out["order"]:
        sl = sorted((s for s in subject.slices if s.image == image), key=lambda s: s.index)
        if sl[0].view == "SAX":
            continue
        if is_aorta(sl[0].view) and not getattr(args, "use_ao", True):
            continue                                       # --use-ao false: written unaligned by the loader
        if len({s.data.shape[:2] for s in sl}) > 1:
            raise ValueError(f"{subject.subject_id}: the slices of image {image} differ in size")
        arr = np.stack([frames(s.data) for s in sl], 2).astype(np.float32)          # (X, Y, Z, F)
        entry = dict(arr=arr, affine=np.stack([s.affine for s in sl]), mask=None, seg=None,
                     thickness=sl[0].thickness, dist=None, fvalid=np.stack([valid(s) for s in sl]),
                     view=sl[0].view, modality=sl[0].modality)
        if any(s.seg is not None for s in sl):
            seg = np.stack([seg_frames(s) if s.seg is not None else np.zeros(s.data.shape[:2] + (n_f,), np.int16)
                            for s in sl], 2)
            entry["seg"] = seg
            heart = np.stack([(s.seg > 0).any(-1) if s.seg is not None else np.zeros(s.data.shape[:2], bool)
                              for s in sl], 2)                                       # (X, Y, Z), union over all frames
            if not heart.any():
                continue                                                             # segmented, but no heart
            spacing = tuple(float(np.linalg.norm(sl[0].affine[:3, k])) for k in (0, 1))
            dist = np.full(heart.shape, np.inf, np.float32)                          # inf: slice without heart
            for z in range(len(sl)):
                if heart[:, :, z].any():
                    h = apical_sweep(heart[:, :, z], sl[z].affine, apex, extend)
                    dist[:, :, z] = distance_transform_edt(~h, sampling=spacing)
            entry.update(dist=dist, mask=dist <= args.mask_dilation_mm)
        elif getattr(args, "center_mask_radius_mm", 0.0) > 0:                       # no segmentation: default disc
            dist = center_disc_dist(entry["affine"], arr.shape[:2], args.center_mask_radius_mm)
            entry.update(dist=dist, mask=dist <= args.mask_dilation_mm)
        out["lax"][image] = entry
    if not out["sax"] and not out["lax"]:
        out["status"] = "no_images"
    return out


# --------------------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------------------

def normalize_intensity(arr: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.0, ref: np.ndarray | None = None) -> np.ndarray:
    """Rescale an image so the lo/hi percentiles of `ref` (default: the image itself, all slices and frames together)
    map to 0/1, giving the images a common intensity scale for the L2 metric. Returns a new float32 array."""
    lo, hi = np.percentile(arr if ref is None else ref, [lo_pct, hi_pct])
    return ((arr - lo) / max(float(hi - lo), 1e-6)).astype(np.float32)


def build_planes(data: dict, args, device, with_seg: bool = True) -> tuple[list[Plane], list[Plane]]:
    """(SAX planes, LAX planes) of a loaded subject, one plane per slice file. For --metric l2 the intensities are
    normalized, all SAX slices with one common scale. with_seg=False leaves the segmentations off the GPU (they are
    only needed for the Dice / Hausdorff metrics; the LAX heart masks are built from them at load time)."""
    l2 = getattr(args, "metric", "ncc") in ("l1", "l2")              # intensity-difference metrics need a common scale
    sax_ref = np.concatenate([e["arr"].ravel() for e in data["sax"]]) if l2 and data["sax"] else None
    order = data.get("order", [])
    rank = lambda name: order.index(name) if name in order else len(order)  # noqa: E731
    sax_planes = []
    for e in data["sax"]:
        arr = normalize_intensity(e["arr"], ref=sax_ref) if l2 else e["arr"]
        image = e.get("image", "SAX")
        sax_planes += make_planes(image, arr, e["affine"], e.get("thickness", args.sax_thickness), None,
                                  e["seg"] if with_seg else None,
                                  device, first_index=e["index"], stack_rank=rank(image), fvalid=e.get("fvalid"),
                                  view="SAX", modality=e.get("modality", ""))
    lax_planes = []
    for view, d in data["lax"].items():
        arr = normalize_intensity(d["arr"]) if l2 else d["arr"]
        lax_planes += make_planes(view, arr, d["affine"], d["thickness"], d["mask"], d["seg"] if with_seg else None, device,
                                  stack_rank=rank(view), fvalid=d.get("fvalid"), view=d.get("view", view),
                                  modality=d.get("modality", ""))
    # programmatic callers without the option (older runs' configs) keep the image-centre rotation they were made with
    set_rotation_pivots(sax_planes, lax_planes, getattr(args, "rotation_pivot", "image-centre"))
    for p in sax_planes + lax_planes:
        p.apex_dir = data.get("apex_dir")
    assign_sax_regions(data, sax_planes, lax_planes)
    return sax_planes, lax_planes


def assign_sax_regions(data: dict, sax_planes: list[Plane], lax_planes: list[Plane]) -> None:
    """Set Plane.region of every SAX slice (SAX_REGIONS) from the long-axis segmentations, which are more reliable than
    the SAX ones at the base and the apex; at the acquired geometry and at end-diastole (frame 0: the base moves ~1 cm
    toward the apex in systole), so a slice keeps its region whatever the alignment does.

    Along the LV long axis (heart_axis, oriented base -> apex by data['apex_dir']), every segmented long-axis view (not
    aortic) gives the LV's extent: its most basal and most apical LV pixel (LV_LABELS); the LV spans the median of the
    views' ends. A SAX slice is placed where the axis crosses it: before the LV's basal end pre_basal (the long-axis
    views show atria there, no LV), past its apical end post_apical (whatever its own segmentation shows: SAX slices
    beyond the apex often carry spurious labels), and in between basal / mid / apical by thirds of the LV's length.
    At the base the slice's own segmentation counts too: a slice in the basal third is basal only once it shows LV
    itself (at least 20 LV pixels at end-diastole), else pre_basal -- the slice through the tilted mitral annulus,
    where the long-axis views show LV and atria and the SAX segmentation nothing. Left "" without a SAX stack, a base
    -> apex direction or a segmented long-axis view."""
    for p in sax_planes:
        p.region = ""
    sax_seg = {(e.get("image", "SAX"), e["index"]): e.get("seg") for e in data.get("sax", [])}   # (X, Y, 1, F)
    apex_dir = data.get("apex_dir")
    axis = heart_axis(sax_planes, lax_planes) if sax_planes and apex_dir is not None else None
    if axis is None:
        return
    L0, ell = np.asarray(axis[0], float), np.asarray(axis[1], float) / np.linalg.norm(axis[1])
    if float(ell @ apex_dir) < 0:
        ell = -ell
    ends = []                                                # per long-axis view: (basal end, apical end) along ell
    for image, d in data.get("lax", {}).items():
        seg = d.get("seg")
        if seg is None or is_aorta(d.get("view", image)):
            continue
        pos = []
        for z in range(seg.shape[2]):
            ij = np.argwhere(np.isin(seg[:, :, z, 0], LV_LABELS))
            if len(ij) < 20:
                continue
            A = np.asarray(d["affine"][z], float)
            pos.append((ij @ A[:3, :2].T + A[:3, 3] - L0) @ ell)
        if pos:
            pos = np.concatenate(pos)
            ends.append((float(pos.min()), float(pos.max())))
    if not ends:
        return
    base, apex = float(np.median([e[0] for e in ends])), float(np.median([e[1] for e in ends]))
    if apex - base < 20.0:                                   # (no plausible LV length)
        return
    for p in sax_planes:
        cos = float(ell @ p.normal)
        s = float((p.center - L0) @ p.normal) / cos if abs(cos) > 0.2 else float((p.center - L0) @ ell)
        f = (s - base) / (apex - base)
        p.region = SAX_REGIONS[0 if f < 0 else 4 if f > 1 else 1 if f < 1 / 3 else 2 if f < 2 / 3 else 3]
        if p.region == "basal":
            g = sax_seg.get((p.image, p.slice_idx))
            if g is not None and int(np.isin(g[:, :, 0, 0], LV_LABELS).sum()) < 20:
                p.region = "pre_basal"


def set_rotation_pivots(sax_planes: list[Plane], lax_planes: list[Plane], mode: str) -> None:
    """Choose the point each plane's pose rotates about (Plane.pivot; the translation t then moves that point).

    image-centre: the centre of the plane's pixel grid. The heart is typically 20-60 mm away from it, so a rotation
      also displaces the heart by ~angle x that distance, which the translation has to cancel: rotation and
      translation are entangled.
    long-axis: a point on the LV long axis (see heart_axis: the crossing line of the two most perpendicular LAX views,
      acquired geometry, or without two such views the axis through the centre of the first stack): for a SAX slice
      where the axis pierces it, for any other plane the heart centre (the mean of the SAX pivots, i.e. the axis at
      mid-stack) projected onto the plane. Rotations then turn the plane about the heart, leaving it in place, and t is
      the heart's displacement.
    """
    for p in sax_planes + lax_planes:
        p.pivot = None
    if mode != "long-axis":
        return
    axis = heart_axis(sax_planes, lax_planes)
    if axis is None:
        return
    L0, ell = axis[0], axis[1] / np.linalg.norm(axis[1])
    for p in sax_planes:
        cos = float(ell @ p.normal)
        if abs(cos) > 0.2:                                   # the axis pierces the slice
            p.pivot = L0 + float((p.center - L0) @ p.normal) / cos * ell
        else:                                                # (nearly in-plane: closest axis point, projected)
            q = L0 + float((p.center - L0) @ ell) * ell
            p.pivot = q - float((q - p.center) @ p.normal) * p.normal
    if sax_planes:
        heart = np.mean([p.pivot for p in sax_planes], 0)
    else:                                                    # no SAX slices: the axis point closest to the views' centres
        c = np.mean([p.center for p in lax_planes], 0)
        heart = L0 + float((c - L0) @ ell) * ell
    for p in lax_planes:
        p.pivot = heart - float((heart - p.center) @ p.normal) * p.normal


def make_planes(image: str, arr: np.ndarray, affine: np.ndarray, thickness: float,
                mask_arr: np.ndarray | None, seg_arr: np.ndarray | None, device, first_index: int = 0,
                stack_rank: int = 0, fvalid: np.ndarray | None = None, view: str | None = None,
                modality: str = "") -> list[Plane]:
    """Planes of a (X, Y, Z, F) image; plane z gets slice index first_index + z and the affine of slice z: affine is
    either one per slice (Z, 4, 4) or the stack's (4, 4), whose origin is then moved to slice z. SAX slices come one at
    a time (Z = 1, first_index = the slice's index); other images may be stacks (first_index 0). fvalid (Z, F) bool:
    the frames each slice has (prepare_subject; None: all). view (default: the image name) and modality: see
    SliceImage."""
    X, Y, Z, _ = arr.shape
    planes = []
    for z in range(Z):
        if affine.ndim == 3:
            A = affine[z].copy()
        else:
            A = affine.copy()
            A[:3, 3] = (affine @ np.array([0.0, 0.0, z, 1.0]))[:3]
        inv = torch.linalg.inv(torch.from_numpy(A)).numpy()
        u, v = A[:3, 0], A[:3, 1]
        n = np.cross(u, v)
        n /= np.linalg.norm(n)
        img = torch.from_numpy(np.ascontiguousarray(arr[:, :, z].transpose(2, 0, 1)))[:, None].to(device)
        mask = seg = None
        if mask_arr is not None:
            mask = torch.from_numpy(mask_arr[:, :, z].astype(np.float32))[None, None].to(device)
        if seg_arr is not None:
            seg = torch.from_numpy(np.ascontiguousarray(seg_arr[:, :, z].transpose(2, 0, 1)).astype(np.float32))[:, None].to(device)
        planes.append(Plane(
            key=f"{image}:{first_index + z}", image=image, slice_idx=first_index + z, img=img, seg=seg, mask=mask,
            inv_rot=torch.tensor(inv[:3, :3], dtype=torch.float32, device=device),
            inv_trans=torch.tensor(inv[:3, 3], dtype=torch.float32, device=device),
            origin=A[:3, 3].copy(),
            center=(A @ np.array([(X - 1) / 2, (Y - 1) / 2, 0.0, 1.0]))[:3],
            u_hat=u / np.linalg.norm(u), v_hat=v / np.linalg.norm(v), normal=n, thickness=thickness,
            spacing=(float(np.linalg.norm(u)), float(np.linalg.norm(v))), affine=A.copy(), stack_rank=stack_rank,
            fvalid=None if fvalid is None else torch.from_numpy(np.asarray(fvalid[z], bool)).to(device),
            view=image if view is None else view, modality=modality,
        ))
    return planes


def voxel_grid(plane: Plane, pts: torch.Tensor) -> torch.Tensor:
    """World points (..., 3) -> grid_sample coordinates (..., 2) in [-1, 1]."""
    vox = pts @ plane.inv_rot.T + plane.inv_trans
    gx = 2 * vox[..., 1] / (plane.W - 1) - 1
    gy = 2 * vox[..., 0] / (plane.H - 1) - 1
    return torch.stack([gx, gy], -1)


def sample_image(img: torch.Tensor, grid: torch.Tensor, mode: str) -> torch.Tensor:
    """img (N, 1, H, W), grid (M, 2) -> (N, M)."""
    g = grid.view(1, 1, -1, 2).expand(img.shape[0], 1, -1, 2)
    return F.grid_sample(img, g, mode=mode, padding_mode="zeros", align_corners=True)[:, 0, 0, :]


def inside(plane: Plane, pts: torch.Tensor) -> torch.Tensor:
    vox = pts @ plane.inv_rot.T + plane.inv_trans
    return (vox[:, 0] >= 0) & (vox[:, 0] <= plane.H - 1) & (vox[:, 1] >= 0) & (vox[:, 1] <= plane.W - 1)


def intersection_line(a: Plane, b: Plane, args, device,
                      cylinder: tuple[np.ndarray, np.ndarray, float] | None = None) -> Line | None:
    """Valid parameter range and mask weights of the a/b intersection line at the acquired geometry.

    The mask is the union of the planes' masks (a LAX view's dilated heart mask); when neither plane has one, the
    optional cylinder (point, unit direction, radius mm) is the mask instead: the points within radius of that axis
    (build_lines passes the SAX-SAX cylinder around the LV long axis for the SAX-aortic pairs).
    Whether a line is kept (--min-line-mm) and its n_points only count the points inside the mask. With
    --outside-mask-weight > 0 the line is extended over its whole part inside both images, and the points outside
    the mask get that weight in the loss (w_loss); w stays the binary in-mask indicator used by the metrics.
    """
    d = np.cross(a.normal, b.normal)
    nd = np.linalg.norm(d)
    if nd < 1e-3:
        return None
    d /= nd
    M = np.stack([a.normal, b.normal, d])
    rhs = np.array([a.normal @ a.origin, b.normal @ b.origin, d @ b.center])
    x0 = torch.linalg.solve(torch.from_numpy(M), torch.from_numpy(rhs)).numpy()

    t = np.arange(-args.line_extent_mm, args.line_extent_mm + 1e-6, args.line_step_mm)
    pts = torch.tensor(x0[None] + t[:, None] * d[None], dtype=torch.float32, device=device)

    in_img = inside(a, pts) & inside(b, pts)
    valid = in_img.clone()
    masked_planes = [p for p in (a, b) if p.mask is not None]
    if masked_planes:
        in_mask = torch.zeros_like(valid)
        for p in masked_planes:
            in_mask |= sample_image(p.mask, voxel_grid(p, pts), "nearest")[0] > 0.5
        valid &= in_mask
    elif cylinder is not None:
        c0, ell, radius = cylinder
        rel = x0[None] + t[:, None] * d[None] - np.asarray(c0)[None]
        perp = rel - (rel @ ell)[:, None] * np.asarray(ell)[None]
        valid &= torch.from_numpy((perp * perp).sum(1) <= radius ** 2).to(device)

    n_points = int(valid.sum().item())
    if n_points < args.min_line_mm / args.line_step_mm:
        return None
    w_out = getattr(args, "outside_mask_weight", 0.0)
    soft = w_out > 0 and bool((in_img & ~valid).any())
    idx = (in_img if soft else valid).nonzero().squeeze(1)
    lo, hi = int(idx[0]), int(idx[-1]) + 1
    w_loss = None
    if soft:
        w_loss = torch.where(valid, 1.0, torch.where(in_img, w_out, 0.0))[lo:hi].float()
    return Line(t=torch.tensor(t[lo:hi], dtype=torch.float32, device=device), w=valid[lo:hi].float(), n_points=float(n_points),
                w_loss=w_loss, x0=x0, d=d)


# --------------------------------------------------------------------------------------
# Pose-dependent line sampling
# --------------------------------------------------------------------------------------

def skew(v: torch.Tensor) -> torch.Tensor:
    """(N, 3) rotation vectors -> (N, 3, 3) skew-symmetric matrices."""
    z = torch.zeros_like(v[:, 0])
    return torch.stack([
        torch.stack([z, -v[:, 2], v[:, 1]], -1),
        torch.stack([v[:, 2], z, -v[:, 0]], -1),
        torch.stack([-v[:, 1], v[:, 0], z], -1),
    ], -2)


def rodrigues(v: torch.Tensor) -> torch.Tensor:
    """(N, 3) rotation vectors (radians) -> (N, 3, 3) rotation matrices, closed form (= matrix_exp(skew(v))).

    Uses the Taylor expansion near zero so both value and gradient are exact at v = 0.
    """
    t2 = (v * v).sum(-1)
    small = t2 < 1e-8
    th = torch.sqrt(t2.clamp_min(1e-12))
    a = torch.where(small, 1.0 - t2 / 6.0, torch.sin(th) / th)
    b = torch.where(small, 0.5 - t2 / 24.0, (1.0 - torch.cos(th)) / t2.clamp_min(1e-12))
    k = skew(v)
    eye = torch.eye(3, dtype=v.dtype, device=v.device)
    return eye + a[:, None, None] * k + b[:, None, None] * (k @ k)


def solve3_rows(r0: torch.Tensor, r1: torch.Tensor, r2: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Solve [r0; r1; r2] x = rhs for a batch of 3x3 systems given by their rows (M, 3) each; closed form, no sync."""
    c12, c20, c01 = torch.cross(r1, r2, dim=1), torch.cross(r2, r0, dim=1), torch.cross(r0, r1, dim=1)
    det = (r0 * c12).sum(1, keepdim=True)
    return (rhs[:, 0:1] * c12 + rhs[:, 1:2] * c20 + rhs[:, 2:3] * c01) / det


def rotation_matrices(theta_deg: np.ndarray) -> np.ndarray:
    """(N, 3) rotation vectors in degrees -> (N, 3, 3) rotation matrices."""
    th = torch.tensor(np.asarray(theta_deg, dtype=np.float64) * (math.pi / 180.0))
    return rodrigues(th).numpy()


def frame_mean(x: torch.Tensor, fw: torch.Tensor | None = None) -> torch.Tensor:
    """Per-line values per frame (M, F) -> their mean over the frames (M,); with fw (M, F) the weighted mean over the
    frames both slices of a line have (LineSampler.FW: a single-frame slice only at end-diastole)."""
    if fw is None:
        return x.mean(1)
    return (x * fw).sum(1) / fw.sum(1).clamp_min(1e-6)


def ncc(a: torch.Tensor, b: torch.Tensor, w: torch.Tensor, use_gradient: bool) -> torch.Tensor:
    """Weighted NCC along the last axis; a, b, w broadcast against each other."""
    if use_gradient:
        a = a[..., 1:] - a[..., :-1]
        b = b[..., 1:] - b[..., :-1]
        w = w[..., 1:] * w[..., :-1]
    ws = w.sum(-1, keepdim=True).clamp_min(1.0)
    am = (a * w).sum(-1, keepdim=True) / ws
    bm = (b * w).sum(-1, keepdim=True) / ws
    ac, bc = a - am, b - bm
    num = (w * ac * bc).sum(-1)
    den = torch.sqrt(((w * ac * ac).sum(-1) * (w * bc * bc).sum(-1)).clamp_min(1e-12)) + 1e-6   # finite gradient at 0
    return num / den


def soft_nmi(a: torch.Tensor, b: torch.Tensor, w: torch.Tensor, bins: int = 24, sigma_bins: float = 1.0) -> torch.Tensor:
    """Differentiable normalized mutual information 2 I(A;B) / (H(A) + H(B)) per row, in [0, 1].

    a, b, w: (M, S). Each row is z-scored with its own weighted statistics (MI is invariant to monotonic
    intensity maps, so only robustness matters here), clamped to +-3 SD and mapped to [0, 1]. Samples are
    soft-assigned to `bins` bins with a Gaussian Parzen window of `sigma_bins` bin widths; the joint
    histogram is accumulated with the sample weights w (mask / padding).
    """
    eps = 1e-10
    ws = w.sum(-1, keepdim=True).clamp_min(1.0)

    def to_unit(x):
        m = (x * w).sum(-1, keepdim=True) / ws
        sd = torch.sqrt((w * (x - m) ** 2).sum(-1, keepdim=True) / ws + 1e-6)
        return ((x - m) / sd).clamp(-3.0, 3.0) / 6.0 + 0.5

    centers = (torch.arange(bins, device=a.device, dtype=a.dtype) + 0.5) / bins
    sig = sigma_bins / bins

    def parzen(x):
        k = torch.exp(-0.5 * ((x[..., None] - centers) / sig) ** 2)            # (M, S, bins)
        return k / k.sum(-1, keepdim=True).clamp_min(eps)

    pa, pb = parzen(to_unit(a)), parzen(to_unit(b))
    joint = torch.bmm((pa * w[..., None]).transpose(1, 2), pb) / ws[..., None]  # (M, bins, bins), no (M, S, b, b)
    pa_m, pb_m = joint.sum(2), joint.sum(1)
    h_ab = -(joint * torch.log(joint + eps)).sum((1, 2))
    h_a = -(pa_m * torch.log(pa_m + eps)).sum(1)
    h_b = -(pb_m * torch.log(pb_m + eps)).sum(1)
    mi = h_a + h_b - h_ab
    return 2.0 * mi / (h_a + h_b).clamp_min(eps)


def l1_similarity(a: torch.Tensor, b: torch.Tensor, w: torch.Tensor, fw: torch.Tensor | None = None) -> torch.Tensor:
    """--metric l1: -mean |a - b| of profiles a, b (M, F, L) with point weights w (M, L), mean over frames (fw:
    frame_mean) -> (M,). Only meaningful on images normalized to a common intensity scale (build_planes does this)."""
    wf = w[:, None, :]
    return -frame_mean((wf * (a - b).abs()).sum(-1) / wf.sum(-1).clamp_min(1.0), fw)


def ngf_similarity(a: torch.Tensor, b: torch.Tensor, w: torch.Tensor, grad_eps: float,
                   fw: torch.Tensor | None = None) -> torch.Tensor:
    """--metric ngf (normalized gradient agreement) of profiles a, b (M, F, L) with point weights w (M, L) -> (M,).

    The central-difference gradients g along the line are normalized per point, g / sqrt(g^2 + eps^2), with eps =
    grad_eps x the standard deviation of g over the weighted points (per profile and frame, so each image's contrast
    does not matter): edges well above eps -> +-1, flat stretches -> 0. Score sum(na nb) / sum((na^2 + nb^2) / 2) in
    [-1, 1] (interior points, weighted where all three samples are): 1 = the same edges at the same places with the
    same sign, 0 for edges in one image only, -1 for opposite edges. Mean over frames (fw: frame_mean)."""
    wg = (w[:, :-2] * w[:, 1:-1] * w[:, 2:])[:, None, :]
    ws = wg.sum(-1, keepdim=True).clamp_min(1e-6)

    def normalize(x):
        g = x[..., 2:] - x[..., :-2]
        eps = grad_eps * ((wg * g * g).sum(-1, keepdim=True) / ws + 1e-12).sqrt() + 1e-6   # finite gradient at 0
        return g / (g * g + eps * eps).sqrt()

    na, nb = normalize(a), normalize(b)
    num = (wg * na * nb).sum(-1)
    den = (wg * 0.5 * (na * na + nb * nb)).sum(-1).clamp_min(1e-6)
    return frame_mean(num / den, fw)


def lncc_similarity(a: torch.Tensor, b: torch.Tensor, w: torch.Tensor, k: int, min_std: float,
                    fw: torch.Tensor | None = None) -> torch.Tensor:
    """--metric lncc (local NCC) of profiles a, b (M, F, L) with point weights w (M, L) -> (M,).

    The profiles are z-scored over their weighted points, then NCC is taken in windows of k samples sliding by one
    sample and averaged over the windows, each weighted by its weighted point count (0 below half a window) x a
    confidence that fades out windows where the profiles hardly vary: min(1, (sd_a sd_b)^(1/2) / min_std), sd in units
    of the profile's standard deviation over the whole line. Every stretch has to match on its own; one strong edge
    cannot dominate. Mean over frames (fw: frame_mean)."""
    k = min(k, a.shape[-1])
    wf = w[:, None, :]
    ws = wf.sum(-1, keepdim=True).clamp_min(1e-6)

    def zscore(x):
        mu = (wf * x).sum(-1, keepdim=True) / ws
        return (x - mu) / (((wf * (x - mu) ** 2).sum(-1, keepdim=True) / ws + 1e-12).sqrt() + 1e-6)

    def window_sum(x):
        c = F.pad(torch.cumsum(x, -1), (1, 0))
        return c[..., k:] - c[..., :-k]

    za, zb = zscore(a), zscore(b)
    n = window_sum(wf.expand_as(za))
    nn_ = n.clamp_min(1e-6)
    ma, mb = window_sum(wf * za) / nn_, window_sum(wf * zb) / nn_
    va = (window_sum(wf * za * za) / nn_ - ma * ma).clamp_min(0.0)
    vb = (window_sum(wf * zb * zb) / nn_ - mb * mb).clamp_min(0.0)
    cov = window_sum(wf * za * zb) / nn_ - ma * mb
    loc = cov / (va * vb + 1e-6).sqrt()
    conf = ((va * vb + 1e-12).pow(0.25) / min_std).clamp(max=1.0)                 # finite gradient at 0
    wt = n * conf * (n >= 0.5 * k).float()
    return frame_mean((loc * wt).sum(-1) / wt.sum(-1).clamp_min(1e-6), fw)


def metric_rows(metric: str, a: torch.Tensor, b: torch.Tensor, w: torch.Tensor, args, fw: torch.Tensor | None = None,
                step_mm: float | None = None) -> torch.Tensor:
    """The similarity `metric` (a --metric choice) of profile pairs a, b (M, F, L) with point weights w (M, L), over
    the frames (fw: frame_mean) -> (M,); higher is better. step_mm: the spacing of the profile points (--lncc window;
    default --line-step-mm)."""
    if metric == "nmi":                                    # samples pooled over the frames
        ww = w[:, None, :].expand_as(a)
        if fw is not None:
            ww = ww * fw[:, :, None]
        m = a.shape[0]
        return soft_nmi(a.reshape(m, -1), b.reshape(m, -1), ww.reshape(m, -1), args.nmi_bins, args.nmi_sigma)
    if metric == "l2":
        wf = w[:, None, :]
        return -frame_mean((wf * (a - b) ** 2).sum(-1) / wf.sum(-1).clamp_min(1.0), fw)
    if metric in ("ncc", "ncc_grad"):
        return frame_mean(ncc(a, b, w[:, None, :], metric == "ncc_grad"), fw)
    if metric == "l1":
        return l1_similarity(a, b, w, fw)
    if metric == "ngf":
        return ngf_similarity(a, b, w, args.grad_eps, fw)
    step = args.line_step_mm if step_mm is None else step_mm
    return lncc_similarity(a, b, w, max(3, int(round(args.lncc_window_mm / step))), args.lncc_min_std, fw)


def cross_mod_metric(args) -> str:
    """The similarity used between slices of different modalities (--cross-mod-metric; --metric if unset)."""
    return getattr(args, "cross_mod_metric", None) or args.metric


def apex_line_sign(a: Plane, b: Plane) -> float:
    """Whether the a/b intersection line gets the LAX-LAX apex metrics (LineSampler.apex_lines), and which way it runs:
    +1 / -1 if both are long-axis views (neither SAX nor aortic), the subject's base -> apex direction is known
    (Plane.apex_dir) and the line runs within 60 deg of it (APEX_MIN_COS) -- the 2Ch / 3Ch / 4Ch lines, which contain
    the LV long axis -- with its direction (cross(n_a, n_b), along which the line positions T grow) pointing toward
    (+1) or away from (-1) the apex; 0 otherwise. From the acquired geometry (the poses rotate the line by a few
    degrees at most)."""
    ap = getattr(a, "apex_dir", None)
    if ap is None or "SAX" in (a.view, b.view) or is_aorta(a.view) or is_aorta(b.view):
        return 0.0
    d = np.cross(a.normal, b.normal)
    n = float(np.linalg.norm(d))
    if n < 1e-3:
        return 0.0
    c = float(d @ ap) / n
    return math.copysign(1.0, c) if abs(c) >= APEX_MIN_COS else 0.0


class ImageVolume:
    """All planes' images in one zero-padded volume (1, F, D, Hmax, Wmax), plane p at depth p, stored once per subject
    in the sampling precision (--sample-dtype) and shared by its samplers.

    Any set of lines is sampled with one 3D grid_sample call (line-major: the cost scales with the lines sampled, e.g.
    a --pair-keep subset). Trilinear sampling at plane p's depth is the bilinear sample of plane p: in float32 the
    normalized depth 2p / (D - 1) - 1 round-trips to p within ~1e-5 (a negligible weight on the neighbour); in float16
    / bfloat16 D is padded to 2^k + 1, where depth and un-normalization are exact. Pixels beyond a plane's own size are
    zeros, as outside the image with padding_mode="zeros". In half precision, images brighter than 32768 are scaled by
    an exact power of two (the metrics are invariant to the intensity scale; float16 holds integers to 2048 exactly,
    blurred values to ~0.05%).

    raw holds the originals; vol the images sampled -- the same tensor until an in-plane blur is first set, then a
    blurred copy (allocated before any CUDA graph is captured, as the blur is set before the first step). The planes'
    img / seg become views into raw / seg, so the images are held once. seg (segmentations, nearest sampling; -1
    inside a plane without one) exists only when a plane has one.
    """

    def __init__(self, planes: list[Plane], device, dtype=torch.float32):
        N, self.nf = len(planes), planes[0].img.shape[0]
        self.dtype = dtype
        self.Hmax, self.Wmax = max(p.H for p in planes), max(p.W for p in planes)
        self.D = max(2, N) if dtype == torch.float32 else 2 ** max(1, math.ceil(math.log2(max(2, N - 1)))) + 1
        self.depth = torch.tensor([2.0 * p / (self.D - 1) - 1.0 for p in range(N)], dtype=torch.float32, device=device)
        shape = (1, self.nf, self.D, self.Hmax, self.Wmax)
        peak = max(float(p.img.abs().max()) for p in planes)
        self.scale = 1.0 if dtype == torch.float32 or peak <= 32768 else 2.0 ** -math.ceil(math.log2(peak / 32768))
        self.raw = torch.zeros(shape, dtype=dtype, device=device)
        self.seg = torch.zeros(shape, dtype=dtype, device=device) if any(p.seg is not None for p in planes) else None
        self.groups = []                                 # planes of equal size, blurred together
        by_size = {}
        for p, plane in enumerate(planes):
            by_size.setdefault((plane.H, plane.W), []).append(p)
        for (H, W), members in by_size.items():
            self.groups.append(dict(H=H, W=W, pidx=torch.tensor(members, device=device),
                                    spacing_list=[list(planes[p].spacing) for p in members]))
        for p, pl in enumerate(planes):
            H, W = pl.H, pl.W
            self.raw[0, :, p, :H, :W] = pl.img[:, 0] * self.scale
            pl.img = self.raw[0, :, p, :H, :W][:, None]              # (F, 1, H, W) view: the plane's own copy is freed
            if self.seg is not None:
                self.seg[0, :, p, :H, :W] = pl.seg[:, 0] if pl.seg is not None else -1.0
                if pl.seg is not None:
                    pl.seg = self.seg[0, :, p, :H, :W][:, None]
        self.vol = self.raw
        self.cur_sigma = 0.0

    @torch.no_grad()
    def set_blur(self, sigma_mm: float) -> None:
        """Write in-plane Gaussian-blurred versions of the originals (sigma in mm; 0 = the originals) into vol.

        Updated in place (CUDA graphs keep pointing at it); nothing is done if sigma is unchanged. Blurred in float32,
        stored in the volume's precision."""
        if sigma_mm == self.cur_sigma:
            return
        self.cur_sigma = sigma_mm
        if sigma_mm > 1e-6 and self.vol is self.raw:
            self.vol = self.raw.clone()
        if self.vol is self.raw:
            return
        if sigma_mm <= 1e-6:
            self.vol.copy_(self.raw)
            return
        for gr in self.groups:
            raw = self.raw[0, :, gr["pidx"], :gr["H"], :gr["W"]].transpose(0, 1).float()     # (G, F, H, W)
            self.vol[0, :, gr["pidx"], :gr["H"], :gr["W"]] = self._blur(raw, gr["spacing_list"], sigma_mm).transpose(0, 1).to(self.dtype)

    @staticmethod
    def _blur(raw: torch.Tensor, spacing_list: list, sigma_mm: float) -> torch.Tensor:
        """(G, F, H, W) images blurred in-plane with sigma (mm), per plane in its own pixel spacing."""
        G, Fn, H, W = raw.shape
        sig = [[max(sigma_mm / s, 1e-3) for s in row] for row in spacing_list]       # voxels, per plane/axis
        r = int(math.ceil(3.0 * max(max(row) for row in sig)))
        x = torch.arange(-r, r + 1, device=raw.device, dtype=raw.dtype)
        sig_t = torch.tensor(sig, device=raw.device, dtype=raw.dtype)            # (G, 2)
        k = torch.exp(-0.5 * (x[None, None, :] / sig_t[:, :, None]) ** 2)      # (G, 2, 2r+1)
        k = (k / k.sum(-1, keepdim=True)).repeat_interleave(Fn, dim=0)          # (G*F, 2, 2r+1)
        out = F.pad(raw.reshape(1, G * Fn, H, W), (r, r, r, r), mode="replicate")
        out = F.conv2d(out, k[:, 0][:, None, :, None], groups=G * Fn)          # along image axis 0 (H)
        out = F.conv2d(out, k[:, 1][:, None, None, :], groups=G * Fn)          # along image axis 1 (W)
        return out.reshape(G, Fn, H, W)

    def frame_subset(self, stride: int) -> "ImageVolume":
        """Every `stride`-th frame of the images (frames 0, stride, 2 stride, ...), as a volume of its own with its own
        blur (for the coarse early steps, --coarse-frame-stride); images only (no segmentations). Made once per stride
        and shared by the samplers of this volume."""
        cache = self.__dict__.setdefault("_subsets", {})
        if stride not in cache:
            import copy
            sub = copy.copy(self)
            sub.__dict__.pop("_subsets", None)
            sub.raw = self.raw[:, ::stride].contiguous()
            sub.vol, sub.seg, sub.nf, sub.cur_sigma = sub.raw, None, sub.raw.shape[1], 0.0
            cache[stride] = sub
        return cache[stride]

    def sample(self, grid: torch.Tensor, kind: str = "img", mode: str = "bilinear") -> torch.Tensor:
        """grid_sample of vol (kind 'img') or seg at grid (1, Do, Ho, Wo, 3) (float32; cast to the volume's precision)
        -> (F, Do, Ho, Wo), in the volume's precision."""
        src = self.vol if kind == "img" else self.seg
        return F.grid_sample(src, grid.to(self.dtype), mode=mode, padding_mode="zeros", align_corners=True)[0]


class LineSampler:
    """Samples both images (or segmentations) of every intersection line for arbitrary rigid poses.

    Lines are recomputed from the posed planes, so through-plane shifts and rotations are
    observable. Planes of equal image size are batched into a single grid_sample call.
    """

    def __init__(self, planes: list[Plane], lines: list[tuple[int, int, Line]], n_band: int, device,
                 psf_model: str = "pulse", psf_tbw: float = 2.5, psf_scale_max: float = 1.0, sample_dtype: str = "float32",
                 images: "ImageVolume | None" = None):
        self.planes, self.device = planes, device
        # image precision (--sample-dtype): images stored and sampled (with the sample coordinates) in half precision;
        # the poses, the line geometry and the metrics stay float32
        self.lp = None if sample_dtype == "float32" else getattr(torch, sample_dtype)
        self.N, self.M = len(planes), len(lines)
        self.nf = planes[0].img.shape[0]
        self.psf_model, self.psf_tbw = psf_model, psf_tbw
        self.sax_pairs = None                    # SaxNeighbourLines term (attach_sax_pairs), None when off
        # through-plane slice profile, as a function of offset x in units of the nominal thickness T (= the FWHM)
        if psf_model == "gaussian":
            self.psf_sigma = 1.0 / 2.3548                                       # FWHM = T
            self.psf_support = 2.0 * self.psf_sigma                             # +-2 sigma
        elif psf_model == "rect":
            self.psf_sigma = 0.0
            self.psf_support = 0.5
        else:   # "pulse": rect of width T convolved with a Gaussian edge; 10-90% transition width ~ T / TBW
            self.psf_sigma = 1.0 / psf_tbw / 2.5631                             # 10-90% width of a Gaussian = 2.563 sigma
            self.psf_support = 0.5 + 2.5 * self.psf_sigma
        # PSF band samples: fixed offsets (units of the other slice's thickness) covering the widest profile used
        # (psf_scale_max, for PSF annealing) at the same density as n_band samples cover the nominal profile; the
        # weights are recomputed from the current scale (self.psf_scale, a device scalar so CUDA graphs see updates).
        smax = max(1.0, psf_scale_max)
        self.K = max(1, int(math.ceil(max(1, n_band) * smax))) if n_band > 1 else 1
        self.u_img = (torch.linspace(-self.psf_support * smax, self.psf_support * smax, self.K, device=device)
                      if self.K > 1 else torch.zeros(1, device=device))
        self.u_seg, self.w_seg = torch.zeros(1, device=device), torch.ones(1, device=device)
        self.psf_scale = torch.ones((), device=device)
        # weight of the late edge term in the optimized similarity (--edge-weight; set every step by align, a device
        # scalar so CUDA graphs see updates; 0 = the --metric alone)
        self.edge_alpha = torch.zeros((), device=device)

        Lmax = max(l.t.numel() for _, _, l in lines)
        self.Lmax = Lmax
        self.T = torch.zeros(self.M, Lmax, device=device)
        self.W = torch.zeros(self.M, Lmax, device=device)                       # 1 inside the mask (all reported metrics)
        self.W_loss = torch.zeros(self.M, Lmax, device=device)                  # weights of the optimized similarity
        for m, (_, _, l) in enumerate(lines):
            n = l.t.numel()
            self.T[m, :n] = l.t
            self.T[m, n:] = l.t[-1]
            self.W[m, :n] = l.w
            self.W_loss[m, :n] = l.w if l.w_loss is None else l.w_loss
        self.IA = torch.tensor([ia for ia, _, _ in lines], device=device)
        self.IB = torch.tensor([ib for _, ib, _ in lines], device=device)
        self.npts = torch.tensor([l.n_points for _, _, l in lines], device=device)
        self.lax_lax = torch.tensor([planes[ia].view != "SAX" and planes[ib].view != "SAX" for ia, ib, _ in lines], device=device)
        self.apex_sign = torch.tensor([apex_line_sign(planes[ia], planes[ib]) for ia, ib, _ in lines], device=device)
        # lines between slices of different modalities: scored with --cross-mod-metric (similarity_lines)
        self.cross_mod = torch.tensor([planes[ia].modality != planes[ib].modality for ia, ib, _ in lines], device=device)
        self.any_cross = bool(self.cross_mod.any())
        la_idx = [torch.nonzero(self.IA == p).squeeze(1) for p in range(self.N)]
        lb_idx = [torch.nonzero(self.IB == p).squeeze(1) for p in range(self.N)]
        self.plane_lines = [torch.cat([la_idx[p], lb_idx[p]]) for p in range(self.N)]

        f32 = dict(dtype=torch.float32, device=device)
        self.O = torch.tensor(np.stack([p.origin for p in planes]), **f32)
        self.C = torch.tensor(np.stack([p.rot_centre for p in planes]), **f32)     # rotation centres (in-plane)
        self.Ctr = torch.tensor(np.stack([p.center for p in planes]), **f32)       # image centres (line anchors)
        self.Nrm = torch.tensor(np.stack([p.normal for p in planes]), **f32)
        self.thick = torch.tensor([p.thickness for p in planes], **f32)
        self.inv_rot = torch.stack([p.inv_rot for p in planes])
        self.inv_trans = torch.stack([p.inv_trans for p in planes])

        # the planes' images (shared between the samplers of one subject, e.g. the mask-annealing stages)
        self.img = images if images is not None else ImageVolume(planes, device, self.lp or torch.float32)
        self.eval_chunk = 1024                           # lines per chunk when sampling all lines without gradients
        self.point_stride = 1                            # the line points are every point_stride-th of --line-step-mm

        # the frames of every plane (N, F) and of every line (FW, (M, F): those both its slices have; a single-frame
        # slice only end-diastole). None when every plane has every frame: plain means, as before
        self.frame_valid = torch.stack([p.fvalid.float() if p.fvalid is not None
                                        else torch.ones(self.nf, device=device) for p in planes])
        self.FW = None if bool(self.frame_valid.all()) else self.frame_valid[self.IA] * self.frame_valid[self.IB]

        # which labels each plane's segmentation contains anywhere, per frame: (N, F, max_label + 1)
        max_label = max([int(p.seg.max()) for p in planes if p.seg is not None] + [0])
        labs = torch.arange(max_label + 1, device=device, dtype=torch.float32)
        self.label_present = torch.stack([
            (p.seg[:, 0, :, :, None] == labs).flatten(1, 2).any(1) if p.seg is not None
            else torch.zeros(self.nf, max_label + 1, dtype=torch.bool, device=device) for p in planes])

    def counted_labels(self, a: torch.Tensor, b: torch.Tensor, labels: list[int], labels_sax: list[int]):
        """For each label: per (line, frame) whether it is counted, and the two on-line masks.

        A label is counted on a line/frame when it is allowed for the pair type, both planes contain it somewhere in
        their segmentation, and at least one of the two line profiles hits it.
        """
        valid_pt = (self.W > 0)[:, None, :]
        pres_a, pres_b = self.label_present[self.IA], self.label_present[self.IB]              # (M, F, K)
        for lab in sorted(set(labels) | set(labels_sax)):
            if lab >= pres_a.shape[-1]:
                continue
            allowed = torch.where(self.lax_lax, lab in labels, lab in labels_sax)[:, None]       # (M, 1)
            ma, mb = (a == lab) & valid_pt, (b == lab) & valid_pt                                # (M, F, L)
            counted = allowed & pres_a[..., lab] & pres_b[..., lab] & (ma.any(-1) | mb.any(-1))  # (M, F)
            yield lab, counted, ma, mb

    def coarse(self, frame_stride: int, point_stride: int) -> "LineSampler":
        """A cheaper version of this sampler for the early steps (--coarse-frac): the same lines and poses, but only
        every frame_stride-th cine frame (ImageVolume.frame_subset) and every point_stride-th point of each line. The
        cost of a step (sampling and its gradient) scales with frames x points; the objective is the same similarity on
        a subsample, so the early, large moves -- which the blur and the dilated masks make smooth anyway -- are about
        the same. Shares everything else (geometry, PSF scale) with this sampler."""
        import copy
        c = copy.copy(self)
        if point_stride > 1:
            c.T, c.W = self.T[:, ::point_stride].contiguous(), self.W[:, ::point_stride].contiguous()
            c.W_loss = self.W_loss[:, ::point_stride].contiguous()
            c.Lmax, c.point_stride = c.T.shape[1], self.point_stride * point_stride
        if frame_stride > 1:
            c.img = self.img.frame_subset(frame_stride)
            c.nf = c.img.nf
            c.label_present = self.label_present[:, ::frame_stride]
            c.frame_valid = self.frame_valid[:, ::frame_stride]               # (frame 0 is always kept)
            c.FW = None if self.FW is None else self.FW[:, ::frame_stride].contiguous()
        return c

    def fw(self, sel: torch.Tensor | None = None) -> torch.Tensor | None:
        """The frame weights (frame_mean) of all lines, or of the lines sel; None when every line has every frame."""
        return None if self.FW is None else (self.FW if sel is None else self.FW[sel])

    def set_blur(self, sigma_mm: float) -> None:
        """In-plane Gaussian blur (sigma in mm) of the images that are sampled (see ImageVolume.set_blur)."""
        self.img.set_blur(sigma_mm)

    def set_psf_scale(self, scale: float) -> None:
        """Width of the PSF band relative to the nominal slice profile (1 = physical)."""
        self.psf_scale.fill_(scale)

    def set_edge_weight(self, alpha: float) -> None:
        """Weight of the edge term (--edge-metric) in the optimized similarity of every line (see similarity_lines)."""
        self.edge_alpha.fill_(alpha)

    def band_weights(self) -> torch.Tensor:
        """PSF weights of the band samples at the current scale (sum to 1)."""
        if self.K == 1:
            return self.w_seg
        w = self.slice_profile(self.u_img / self.psf_scale)
        return w / w.sum()

    def slice_profile(self, x: torch.Tensor) -> torch.Tensor:
        """Relative excitation at through-plane offset x (units of the nominal thickness)."""
        if self.psf_model == "gaussian":
            return torch.exp(-0.5 * (x / self.psf_sigma) ** 2)
        if self.psf_model == "rect":
            return (x.abs() <= 0.5).float()
        s = self.psf_sigma * math.sqrt(2.0)
        return 0.5 * (torch.erf((x + 0.5) / s) - torch.erf((x - 0.5) / s))

    def profiles(self, theta_deg: torch.Tensor, t: torch.Tensor, kind: str = "img",
                 sel: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Profiles of both planes of every line (or of the lines `sel`, an index tensor) at the given poses
        -> (M or len(sel), F, Lmax) x 2. Only the requested lines are sampled; without gradients all lines are
        sampled in chunks of self.eval_chunk lines."""
        if sel is None and not torch.is_grad_enabled() and self.M > self.eval_chunk:
            parts = [self.profiles(theta_deg, t, kind, torch.arange(s, min(s + self.eval_chunk, self.M), device=self.device))
                     for s in range(0, self.M, self.eval_chunk)]
            return torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts])
        band = kind == "img"
        mode = "bilinear" if band else "nearest"
        # PSF: samples are placed across the other slice's profile (through-plane offsets u, in units of that
        # slice's thickness T) and weighted by the slice profile at the current PSF scale.
        K = self.K if band else 1
        u = self.u_img if band else self.u_seg
        wts = self.band_weights() if band else self.w_seg

        R = rodrigues(theta_deg * (math.pi / 180.0))                           # (N, 3, 3)
        Op = (R @ (self.O - self.C)[..., None])[..., 0] + self.C + t
        Np = (R @ self.Nrm[..., None])[..., 0]
        Cp = self.C + t
        # the lines' sample positions T are measured from the point closest to plane B's *image centre* (as in
        # intersection_line, where the masks were made), so anchor on the posed image centre, not the rotation pivot
        Gp = (R @ (self.Ctr - self.C)[..., None])[..., 0] + Cp

        IA, IB, T = (self.IA, self.IB, self.T) if sel is None else (self.IA[sel], self.IB[sel], self.T[sel])
        na, nb = Np[IA], Np[IB]
        d = torch.cross(na, nb, dim=1)
        d = d / d.norm(dim=1, keepdim=True)
        rhs = torch.stack([(na * Op[IA]).sum(1), (nb * Op[IB]).sum(1), (d * Gp[IB]).sum(1)], 1)
        x0 = solve3_rows(na, nb, d, rhs)
        pts = x0[:, None, :] + T[:, :, None] * d[:, None, :]                   # (M, Lmax, 3)

        # band directions: the other plane's normal projected into this plane. Moving a distance x along this
        # in-plane direction moves sin(theta) * x along the other plane's normal (theta = crossing angle), so a
        # through-plane offset of x*T in the other slice is x*T/sin(theta) within this plane.
        ma = nb - (nb * na).sum(1, keepdim=True) * na
        sin_ab = ma.norm(dim=1, keepdim=True).clamp_min(1e-3)                  # |n_b x n_a| = sin(theta)
        ma = ma / sin_ab
        mb = na - (na * nb).sum(1, keepdim=True) * nb
        mb = mb / mb.norm(dim=1, keepdim=True).clamp_min(1e-3)
        sig_a = self.thick[IB] / sin_ab[:, 0]                                 # mm per unit of u inside A (B's slab)
        sig_b = self.thick[IA] / sin_ab[:, 0]                                 # and inside B (A's slab)

        # both sides of every line in one batch: side A samples plane IA across B's slab (direction ma, width sig_a),
        # side B plane IB across A's slab
        plane = torch.cat([IA, IB])                                            # (2M,)
        mdir = torch.cat([ma, mb])
        sig = torch.cat([sig_a, sig_b])
        P = pts.repeat(2, 1, 1)[:, :, None, :] + u[None, None, :, None] * sig[:, None, None, None] * mdir[:, None, None, :]
        # world -> plane-local (undo the pose) -> voxel, folded into one affine map per sampled plane
        A = torch.einsum("pij,pkj->pik", self.inv_rot[plane], R[plane])             # inv_rot @ R^T
        bvec = (torch.einsum("pij,pj->pi", self.inv_rot[plane], self.C[plane]) + self.inv_trans[plane]
                - torch.einsum("pij,pj->pi", A, Cp[plane]))
        vox = torch.einsum("plkj,pij->plki", P, A) + bvec[:, None, None, :]   # (2M, Lmax, K, 3)
        iv = self.img
        z = iv.depth[plane][:, None, None].expand_as(vox[..., 0])
        grid = torch.stack([2 * vox[..., 1] / (iv.Wmax - 1) - 1, 2 * vox[..., 0] / (iv.Hmax - 1) - 1, z], -1)
        out = iv.sample(grid[None], kind, mode)                                # (F, 2M, Lmax, K)
        out = (out * wts.to(out.dtype)).sum(-1).float().transpose(0, 1)       # (2M, F, Lmax)
        n = IA.shape[0]
        return out[:n], out[n:]

    def ncc_lines(self, theta_deg, t, use_gradient: bool, prof=None, W=None, fw=None) -> torch.Tensor:
        """NCC per line, mean over its frames (fw: the lines' frame weights, see fw()) -> (M,)."""
        a, b = prof if prof is not None else self.profiles(theta_deg, t, "img")
        W = self.W if W is None else W
        return frame_mean(ncc(a, b, W[:, None, :], use_gradient), fw)        # (M,)

    def nmi_lines(self, theta_deg, t, bins: int, sigma_bins: float, prof=None, W=None, fw=None) -> torch.Tensor:
        """Normalized mutual information per line, samples pooled over its frames (fw) -> (M,)."""
        a, b = prof if prof is not None else self.profiles(theta_deg, t, "img")
        w = (self.W if W is None else W)[:, None, :].expand_as(a)
        if fw is not None:
            w = w * fw[:, :, None]
        m = a.shape[0]
        return soft_nmi(a.reshape(m, -1), b.reshape(m, -1), w.reshape(m, -1), bins, sigma_bins)

    def l2_lines(self, theta_deg, t, prof=None, W=None, fw=None) -> torch.Tensor:
        """Negative weighted mean squared intensity difference per line, mean over its frames (fw) -> (M,) (higher =
        better). Only meaningful on images normalized to a common intensity scale (see normalize_intensity)."""
        a, b = prof if prof is not None else self.profiles(theta_deg, t, "img")
        w = (self.W if W is None else W)[:, None, :]
        return -frame_mean((w * (a - b) ** 2).sum(-1) / w.sum(-1).clamp_min(1.0), fw)

    def similarity_lines(self, theta_deg, t, args, prof=None, W=None, sel=None) -> torch.Tensor:
        """The optimized similarity per line (--metric), with the loss weights W_loss (points outside the mask
        down-weighted by --outside-mask-weight) or the given W (e.g. self.W: the mask points only) -> (M,).
        sel: only the lines with these indices are sampled and scored -> (len(sel),).
        With --edge-weight > 0 the similarity of lines within a modality is (1 - a) x --metric + a x --edge-metric, a =
        edge_alpha (ramped up late in the run by align); lines between modalities keep --cross-mod-metric."""
        W = self.W_loss if W is None else W
        fw = self.fw(sel)                                                  # the frames both slices of a line have
        if sel is not None:
            W = W[sel]
            prof = self.profiles(theta_deg, t, "img", sel) if prof is None else prof
        a, b = prof if prof is not None else self.profiles(theta_deg, t, "img")
        step = args.line_step_mm * self.point_stride                      # (coarse samplers: sparser points)
        out = metric_rows(args.metric, a, b, W, args, fw, step)
        if getattr(args, "edge_weight", 0.0) > 0:                        # late edge term: blended in by edge_alpha
            edge = metric_rows(args.edge_metric, a, b, W, args, fw, step)
            out = out + self.edge_alpha * (edge - out)
        cross = cross_mod_metric(args)
        if self.any_cross and cross != args.metric:                       # lines between modalities: --cross-mod-metric
            cm = self.cross_mod if sel is None else self.cross_mod[sel]
            out = torch.where(cm, metric_rows(cross, a, b, W, args, fw, step), out)
        return out

    def dice_lines(self, theta_deg, t, labels: list[int], labels_sax: list[int], prof=None) -> torch.Tensor:
        """Mean Dice over the counted labels (see counted_labels) along each line -> (M,).

        A counted label that the two profiles place at non-overlapping positions -- including the case where only
        one profile hits it -- scores 0. Per frame: mean over counted labels; then mean over frames with at least
        one counted label. NaN if no label is counted on any frame. `labels` applies to LAX-LAX lines,
        `labels_sax` to SAX-LAX lines (the SAX model lacks some classes).
        """
        a, b = prof if prof is not None else self.profiles(theta_deg, t, "seg")
        total = torch.zeros(a.shape[:2], device=self.device)
        count = torch.zeros(a.shape[:2], device=self.device)
        for lab, counted, ma, mb in self.counted_labels(a, b, labels, labels_sax):
            inter = (ma & mb).sum(-1).float()
            tot = (ma.sum(-1) + mb.sum(-1)).float()
            dice = 2 * inter / tot.clamp_min(1)
            total += torch.where(counted, dice, torch.zeros_like(dice))
            count += counted.float()
        valid = count > 0
        per_frame = total / count.clamp_min(1)
        out = (per_frame * valid).sum(1) / valid.sum(1).clamp_min(1)
        return torch.where(valid.any(1), out, torch.full_like(out, float("nan")))

    def hd_lines(self, theta_deg, t, labels: list[int], labels_sax: list[int], miss_mm: float = 100.0, prof=None) -> torch.Tensor:
        """1D Hausdorff distance (mm) between the two views' labels along each line -> (M,).

        Counted labels as in dice_lines. If only one profile hits a counted label the distance is undefined
        (infinite); it is reported as `miss_mm`. Per frame: the worst (largest) value over the counted labels;
        then the mean over frames with at least one counted label. NaN if no label is counted on any frame.
        """
        a, b = prof if prof is not None else self.profiles(theta_deg, t, "seg")
        if getattr(self, "_dist", None) is None:                                 # (M, 1, L, L) mm along the line
            self._dist = (self.T[:, :, None] - self.T[:, None, :]).abs()[:, None]
        dist = self._dist
        inf = torch.tensor(float("inf"), device=self.device)
        worst = torch.full(a.shape[:2], -1.0, device=self.device)               # (M, F)
        for lab, counted, ma, mb in self.counted_labels(a, b, labels, labels_sax):
            both = ma.any(-1) & mb.any(-1)
            d_ab = torch.where(mb[:, :, None, :], dist, inf).amin(-1)            # each A point to nearest B point
            d_ba = torch.where(ma[:, :, :, None], dist, inf).amin(-2)            # each B point to nearest A point
            h = torch.maximum(torch.where(ma, d_ab, -inf).amax(-1), torch.where(mb, d_ba, -inf).amax(-1))
            h = torch.where(both, h, torch.full_like(h, miss_mm))
            worst = torch.where(counted, torch.maximum(worst, h), worst)
        ok = worst >= 0
        out = (worst.clamp_min(0) * ok).sum(1) / ok.sum(1).clamp_min(1)
        return torch.where(ok.any(1), out, torch.full_like(out, float("nan")))

    def apex_lines(self, theta_deg, t, miss_mm: float = 100.0, prof=None) -> dict:
        """The LAX-LAX apex metrics of every line -> {name: (M,)}; NaN on the lines that are not apex lines
        (apex_line_sign 0: SAX-LAX lines, aortic slices, LAX-LAX lines across the long axis) and where nothing is
        counted. Along an apex line, a view's apex is its most apical line point (in the mask) that its segmentation
        labels as LV:
          apex       LV myocardium or blood pool (APEX_MYO_LABEL / APEX_LV_LABEL): the epicardial apex
          apex_endo  LV blood pool only: the endocardial apex
        Each: |apex of view a - apex of view b| along the line (mm), per frame; a frame counts when both
        segmentations contain the labels in it and at least one profile hits them (miss_mm when only one does, as for
        the Hausdorff distance); mean over the counted frames.
          apex_dice  Dice of the two views' LV myocardium over the line points within APEX_WINDOW_MM of the more
                     apical of the two (epicardial) apices -- the apical cap; a frame counts when both segmentations
                     contain myocardium in it and either profile has some in the window; mean over the counted frames.
        """
        a, b = prof if prof is not None else self.profiles(theta_deg, t, "seg")
        line_ok = (self.apex_sign != 0)[:, None]                               # (M, 1)
        valid_pt = (self.W > 0)[:, None, :]                                    # (M, 1, L)
        pos = (self.apex_sign[:, None] * self.T)[:, None, :]                   # (M, 1, L) mm toward the apex
        pres_a, pres_b = self.label_present[self.IA], self.label_present[self.IB]      # (M, F, K)
        neg_inf = torch.tensor(-float("inf"), device=self.device)

        def present(pres, labs):                       # (M, F): the segmentation contains any of labs in that frame
            labs = [lab for lab in labs if lab < pres.shape[-1]]
            if not labs:
                return torch.zeros(pres.shape[:2], dtype=torch.bool, device=self.device)
            return pres[..., labs].any(-1)

        def tip(seg, labs):                            # (M, F) x 2: the most apical hit, and whether there is one
            hit = torch.zeros_like(seg, dtype=torch.bool)
            for lab in labs:
                hit |= seg == lab
            hit &= valid_pt
            return torch.where(hit, pos, neg_inf).amax(-1), hit.any(-1)

        def counted_mean(v, counted):                  # (M, F) -> (M,): mean over the counted frames, NaN if none
            n = counted.sum(1)
            out = torch.where(counted, v, torch.zeros_like(v)).sum(1) / n.clamp_min(1)
            return torch.where(n > 0, out, torch.full_like(out, float("nan")))

        out, tips = {}, {}
        for name, labs in (("apex", (APEX_LV_LABEL, APEX_MYO_LABEL)), ("apex_endo", (APEX_LV_LABEL,))):
            (ta, ha), (tb, hb) = tip(a, labs), tip(b, labs)
            tips[name] = (ta, tb)
            counted = line_ok & present(pres_a, labs) & present(pres_b, labs) & (ha | hb)
            dist = torch.where(ha & hb, (ta - tb).abs(), torch.full_like(ta, miss_mm))
            out[name] = counted_mean(dist, counted)
        top = torch.maximum(*tips["apex"])                                     # (M, F); -inf where neither hits
        win = valid_pt & (pos >= (top - APEX_WINDOW_MM)[..., None])           # (M, F, L)
        ma, mb = (a == APEX_MYO_LABEL) & win, (b == APEX_MYO_LABEL) & win
        counted = (line_ok & present(pres_a, (APEX_MYO_LABEL,)) & present(pres_b, (APEX_MYO_LABEL,))
                   & (ma.any(-1) | mb.any(-1)))
        dice = 2 * (ma & mb).sum(-1).float() / (ma.sum(-1) + mb.sum(-1)).clamp_min(1).float()
        out["apex_dice"] = counted_mean(dice, counted)
        return out

    def per_plane(self, values: torch.Tensor) -> list[float]:
        """Line values (M,) -> npts-weighted mean per plane, ignoring NaN (one host transfer). Summed on the host in
        float64: a GPU index_add rounds differently from run to run (the reported metrics would vary in the last
        digits)."""
        ok = torch.isfinite(values)
        v = (torch.where(ok, values, torch.zeros_like(values)) * self.npts).double().cpu()
        w = (self.npts * ok).double().cpu()
        if getattr(self, "_host_pairs", None) is None:                  # (the plane indices, kept on the host)
            self._host_pairs = (self.IA.cpu(), self.IB.cpu())
        ia, ib = self._host_pairs
        num = torch.bincount(ia, v, self.N) + torch.bincount(ib, v, self.N)
        den = torch.bincount(ia, w, self.N) + torch.bincount(ib, w, self.N)
        return torch.where(den > 0, num / den.clamp_min(1e-12), torch.full_like(num, float("nan"))).tolist()

    def plane_values(self, lm: dict) -> dict:
        """Per-plane values (lists of N) of the per-line metrics lm (evaluate): the npts-weighted mean over each
        plane's lines (per_plane), for the Hausdorff distance the worst line (per_plane_max); lax_dice: the Dice over
        the plane's LAX-LAX lines only (the long-axis segmentations are more reliable than the basal / apical SAX ones;
        NaN for a SAX slice); apex / apex_dice: over the plane's apex lines (apex_lines; NaN elsewhere)."""
        out = {k: self.per_plane(lm[k]) for k in ("ncc", "lncc", "nmi", "dice", "apex", "apex_dice")}
        out["hd"] = self.per_plane_max(lm["hd"])
        out["lax_dice"] = self.per_plane(torch.where(self.lax_lax, lm["dice"], torch.full_like(lm["dice"], float("nan"))))
        return out

    def per_plane_max(self, values: torch.Tensor) -> list[float]:
        """Line values (M,) -> worst (largest) value per plane, ignoring NaN (one host transfer)."""
        v = torch.where(torch.isfinite(values), values, torch.full_like(values, -float("inf")))
        out = torch.full((self.N,), -float("inf"), device=self.device)
        out = out.scatter_reduce(0, self.IA, v, "amax").scatter_reduce(0, self.IB, v, "amax")
        return torch.where(torch.isfinite(out), out, torch.full_like(out, float("nan"))).tolist()

    @torch.no_grad()
    def evaluate(self, theta_deg, t, args, labels, labels_sax, has_seg: bool) -> dict:
        """All reported per-line metrics at the given poses, as tensors of shape (M,).

        The image and segmentation profiles are sampled once each and shared by the metrics.
        """
        nan = torch.full((self.M,), float("nan"), device=self.device)
        img = self.profiles(theta_deg, t, "img")
        seg = self.profiles(theta_deg, t, "seg") if has_seg else None
        k_lncc = max(3, int(round(getattr(args, "lncc_window_mm", 20.0) / args.line_step_mm)))
        fw = self.fw()
        miss = getattr(args, "hd_miss_mm", 100.0)
        out = dict(ncc=self.ncc_lines(theta_deg, t, args.metric == "ncc_grad", img, fw=fw),
                   lncc=lncc_similarity(img[0], img[1], self.W, k_lncc, getattr(args, "lncc_min_std", 0.25), fw),
                   nmi=self.nmi_lines(theta_deg, t, args.nmi_bins, args.nmi_sigma, img, fw=fw),
                   edge=metric_rows(getattr(args, "edge_metric", "ncc_grad"), img[0], img[1], self.W, args, fw),
                   dice=self.dice_lines(theta_deg, t, labels, labels_sax, seg) if has_seg else nan,
                   # LV labels only on every line (their IDs agree between all views; region_rows)
                   dice_lv=self.dice_lines(theta_deg, t, list(LV_LABELS), list(LV_LABELS), seg) if has_seg else nan,
                   hd=self.hd_lines(theta_deg, t, labels, labels_sax, miss, seg) if has_seg else nan)
        out.update(self.apex_lines(theta_deg, t, miss, seg) if has_seg
                   else dict(apex=nan, apex_endo=nan, apex_dice=nan))
        return {k: v.float() for k, v in out.items()}


def long_axis_line(lax_planes: list[Plane]) -> tuple[np.ndarray, np.ndarray] | None:
    """(point, unit direction) of the LV long axis: the intersection line of the two LAX views that cross at the
    largest angle, at the acquired geometry. None with fewer than two non-parallel views."""
    best = None
    for a, b in combinations(lax_planes, 2):
        d = np.cross(a.normal, b.normal)
        nd = float(np.linalg.norm(d))
        if nd > 1e-3 and (best is None or nd > best[0]):
            best = (nd, a, b, d / nd)
    if best is None:
        return None
    _, a, b, d = best
    M = np.stack([a.normal, b.normal, d])
    rhs = np.array([a.normal @ a.origin, b.normal @ b.origin, d @ a.center])
    return torch.linalg.solve(torch.from_numpy(M), torch.from_numpy(rhs)).numpy(), d


def stack_frame(planes: list[Plane]) -> tuple[np.ndarray, np.ndarray]:
    """(centre, unit normal) of a set of (near-)parallel planes: the mean of their centres and of their normals."""
    n0 = planes[0].normal
    n = np.mean([p.normal if p.normal @ n0 >= 0 else -p.normal for p in planes], 0)
    return np.mean([p.center for p in planes], 0), n / np.linalg.norm(n)


def heart_axis(sax_planes: list[Plane], lax_planes: list[Plane]) -> tuple[np.ndarray, np.ndarray] | None:
    """(point, unit direction) of the LV long axis used for the rotation pivots and the SAX-SAX terms.

    With two non-parallel single-slice long-axis views (the UKBB 2Ch / 3Ch / 4Ch views; images of one slice that are
    not SAX): their crossing line (long_axis_line). Otherwise, as no long axis can be derived, the line through the
    centre of the first stack along its normal: the SAX image if there is one (its normal approximates the long axis),
    else the image found first (the loader's order). None without planes. Aortic slices (is_aorta) are left out: they do
    not contain the LV long axis."""
    lax_planes = [p for p in lax_planes if not is_aorta(p.view)]
    count = {}
    for p in lax_planes:
        count[p.image] = count.get(p.image, 0) + 1
    axis = long_axis_line([p for p in lax_planes if count[p.image] == 1])
    if axis is not None or not (sax_planes or lax_planes):
        return axis
    first = sax_planes or min(([p for p in lax_planes if p.image == im] for im in count),
                              key=lambda ps: ps[0].stack_rank)
    return stack_frame(first)


def stack_spacing(planes: list[Plane]) -> float:
    """Median distance (mm) between neighbouring slices of a stack along its normal (0 for a single slice)."""
    if len(planes) < 2:
        return 0.0
    _, n = stack_frame(planes)
    pos = np.sort([p.center @ n for p in planes])
    return float(np.median(np.diff(pos)))


def candidate_pairs(planes: list[Plane], args) -> list[tuple[Plane, Plane]]:
    """The plane pairs whose intersection lines enter the loss (acquired geometry), in plane order (SAX first).

    A pair is a candidate when the planes cross at an angle of at least --min-pair-angle-deg (slices of one stack are
    parallel and never pair; nearly parallel slices give ill-conditioned lines). Between two stacks of several slices
    the pairs are thinned (--pair-spacing-mm, default each stack's slice thickness): densely spaced, thick slices
    overlap, so the lines of neighbouring slices carry nearly the same information. With k = max over both stacks of
    round(pair spacing / slice spacing), slices of rank i and j along their stacks pair when (i + j) % k == 0: every
    slice keeps partners k slices apart in the other stack, at a different phase than its neighbours, so every slice
    still has lines. Single slices (e.g. the UKBB LAX views) are never thinned. Whether a candidate's line lies inside
    both images (and the masks) is decided by intersection_line. Aortic slices (is_aorta) pair like any LAX view (their
    masking: see build_lines)."""
    cos_max = math.cos(math.radians(getattr(args, "min_pair_angle_deg", 20.0)))
    spacing_opt = getattr(args, "pair_spacing_mm", -1.0)
    by_image = {}
    for p in planes:
        by_image.setdefault(p.image, []).append(p)
    rank, k_of = {}, {}
    for image, ps in by_image.items():
        _, n = stack_frame(ps)
        for r, p in enumerate(sorted(ps, key=lambda p: float(p.center @ n))):
            rank[p.key] = r
        spacing = stack_spacing(ps)
        target = ps[0].thickness if spacing_opt < 0 else spacing_opt
        k_of[image] = max(1, int(round(target / spacing))) if spacing > 1e-6 and target > 0 else 1
    out = []
    for i, a in enumerate(planes):
        for b in planes[i + 1:]:
            if abs(float(a.normal @ b.normal)) > cos_max:
                continue
            if len(by_image[a.image]) > 1 and len(by_image[b.image]) > 1 and a.image != b.image:
                k = max(k_of[a.image], k_of[b.image])
                if (rank[a.key] + rank[b.key]) % k:
                    continue
            out.append((a, b))
    return out


def build_lines(planes: list[Plane], args, device) -> list[tuple[int, int, Line]]:
    """(index a, index b, Line) of every candidate pair (candidate_pairs) whose line lies inside both images and the
    masks over at least --min-line-mm (intersection_line), at the planes' current masks.

    An aortic slice (is_aorta) has no segmentation: its lines with a LAX view are masked by that view's dilated heart
    mask (like every LAX-LAX line); its lines with a SAX slice, where neither plane has a mask, by the cylinder of the
    SAX-SAX terms instead: the points within --sax-sax-radius-mm of the LV long axis (heart_axis, acquired geometry),
    i.e. a disk on the SAX slice around the heart rather than the whole image overlap."""
    index = {p.key: i for i, p in enumerate(planes)}
    cylinder = None
    if any(is_aorta(p.view) for p in planes):
        axis = heart_axis([p for p in planes if p.view == "SAX"], [p for p in planes if p.view != "SAX"])
        if axis is not None:
            cylinder = (axis[0], axis[1] / np.linalg.norm(axis[1]), float(getattr(args, "sax_sax_radius_mm", 120.0)))
    lines = []
    with torch.no_grad():
        for a, b in candidate_pairs(planes, args):
            sax_ao = (a.view == "SAX") != (b.view == "SAX") and (is_aorta(a.view) or is_aorta(b.view))
            line = intersection_line(a, b, args, device, cylinder if sax_ao else None)
            if line is not None:
                lines.append((index[a.key], index[b.key], line))
    return lines


class SaxNeighbourLines:
    """Neighbouring slices of a stack (adjacent along the stack normal, parallel as acquired and at most
    --parallel-max-dist-mm apart; UKBB SAX: z and z + 1; every stack of several slices, see attach_sax_pairs): their
    crossing line and the gap between them. Below, "the long axis" is each pair's own axis: the LV long axis for SAX
    slices, the stack's axis (through its centre along its normal) for any other stack.

    As acquired the SAX slices are parallel and ~10 mm apart, and real breath-hold tilts are a few degrees, so two
    neighbours crossing inside the heart means a pose is wrong. Two loss terms act on this (both off at weight 0):

    * --sax-gap-reg (barrier): points on a disk of --sax-gap-radius-mm around the LV long axis on each slice; their
      signed distance to the neighbouring slice along the acquired stacking direction (the gap, ~10 mm as acquired)
      is penalized smoothly once it drops below --sax-gap-margin-mm: mean of softplus(margin - gap)^2 (mm^2).
      Ordinary small tilts cost nothing; a slice approaching or crossing its neighbour inside the heart is pushed
      back. Purely geometric, no image sampling.
    * --sax-sax-weight (dissimilarity): where the poses do make two neighbours cross, the images are compared along
      the crossing line with NCC and the cost w * sum_points weight * (1 - NCC) / (number of SAX-LAX/LAX-LAX line
      points) is subtracted from the objective. A crossing therefore never pays: a matching one costs little, a
      mismatched one a lot. The points' weights fall off with the distance from the long axis: a Gaussian with
      sigma --sax-sax-radius-mm (--sax-sax-falloff gaussian; it gives a gradient that moves a mismatched crossing
      out of the heart) or a hard cut at that radius (threshold); only points inside both images count.

    The crossing line is recomputed from the posed planes every step and sampled at --line-step-mm over
    +-line-extent-mm around its closest point to the LV long axis (the intersection line of the two LAX views that
    cross at the largest angle, acquired geometry). The images are sampled on the line itself (no PSF band: at the
    shallow angles neighbouring slices cross at, the band would span tens of mm) from the LineSampler's buffers, so
    the in-plane blur annealing applies. With --logging both are always recorded (gap, relative tilt, crossing line
    and its NCC), whatever the weights.
    """

    def __init__(self, sampler: LineSampler, planes: list[Plane], pairs: list[tuple[int, int]],
                 axes: list[tuple[np.ndarray, np.ndarray]], args, device, axis: tuple[np.ndarray, np.ndarray] | None = None):
        """pairs: (plane index a, plane index b) of neighbouring slices; axes: per pair the (point, direction) of the
        axis its weights fall off from and its barrier disk is centred on (the LV long axis for SAX slices, the stack's
        own axis otherwise, see attach_sax_pairs); axis: the LV long axis recorded in long_axis.csv (default: the first
        pair's)."""
        self.sampler, self.device = sampler, device
        # loss weights; 0 = only recorded (with logging). Without tilts (--dof 3, 4) the slices stay parallel and cannot cross,
        # but through-plane shifts can still close the gap, so the barrier applies then as well
        self.weight = float(getattr(args, "sax_sax_weight", 0.0)) if args.dof == 6 else 0.0
        self.falloff = getattr(args, "sax_sax_falloff", "gaussian")
        self.radius = float(args.sax_sax_radius_mm)
        self.gap_weight = float(getattr(args, "sax_gap_reg", 0.0))
        self.gap_margin = float(getattr(args, "sax_gap_margin_mm", 4.0))
        self.gap_radius = float(getattr(args, "sax_gap_radius_mm", 50.0))
        self.pairs = list(pairs)
        self.P = len(self.pairs)
        f32 = dict(dtype=torch.float32, device=device)
        self.IA = torch.tensor([ia for ia, _ in self.pairs], device=device)
        self.IB = torch.tensor([ib for _, ib in self.pairs], device=device)
        self.T = torch.arange(-args.line_extent_mm, args.line_extent_mm + 1e-6, args.line_step_mm, **f32)
        self.T_np = self.T.cpu().numpy()
        axis = axes[0] if axis is None else axis
        self.L0 = torch.tensor(axis[0], **f32)                                       # recorded long axis (logging)
        self.ell = torch.tensor(axis[1] / np.linalg.norm(axis[1]), **f32)
        self.L0p = torch.tensor(np.stack([a[0] for a in axes]), **f32)               # per pair (P, 3)
        self.ellp = torch.tensor(np.stack([a[1] / np.linalg.norm(a[1]) for a in axes]), **f32)
        self.args = args
        self.images_a = [planes[ia].image for ia, _ in self.pairs]
        self.images_b = [planes[ib].image for _, ib in self.pairs]
        self.kinds = [neighbour_kind(planes[ia].view, planes[ib].view) for ia, ib in self.pairs]
        # pairs between modalities: compared with --cross-mod-metric instead of NCC
        self.cross = torch.tensor([planes[ia].modality != planes[ib].modality for ia, ib in self.pairs], device=device)
        self.any_cross = bool(self.cross.any())
        self.size = [torch.tensor([[planes[i].H - 1, planes[i].W - 1] for i in idx], **f32)
                     for idx in (self.IA.tolist(), self.IB.tolist())]
        # barrier: a 5 mm grid of in-plane offsets on a disk of gap_radius, the planes' in-plane axes, and the acquired
        # stacking direction of each pair (+1 when slice b lies along +normal of slice a)
        g = np.arange(-self.gap_radius, self.gap_radius + 1e-6, 5.0)
        gx, gy = np.meshgrid(g, g, indexing="ij")
        inside_disk = gx ** 2 + gy ** 2 <= self.gap_radius ** 2
        self.disk = torch.tensor(np.stack([gx[inside_disk], gy[inside_disk]], 1), **f32)          # (D, 2)
        self.U = torch.tensor(np.stack([p.u_hat for p in planes]), **f32)
        self.V = torch.tensor(np.stack([p.v_hat for p in planes]), **f32)
        C0, N0 = sampler.C, sampler.Nrm
        step = ((C0[self.IB] - C0[self.IA]) * N0[self.IA]).sum(1)
        self.stack_sign = torch.where(step >= 0, 1.0, -1.0)
        # per pair, the barrier margin is at most half the acquired gap, so densely spaced slices (e.g. 1-3 mm) are not
        # penalized as acquired (UKBB, ~10 mm apart: --sax-gap-margin-mm itself)
        self.margins = torch.minimum(torch.full_like(step, self.gap_margin), 0.5 * step.abs())[:, None, None]
        # the frames both slices of a pair have (a single-frame slice: end-diastole only); None: all frames
        fv = sampler.frame_valid
        self.FW = None if sampler.FW is None else fv[self.IA] * fv[self.IB]

    def _sample(self, side: int, vox: torch.Tensor) -> torch.Tensor:
        """Bilinear samples (P, F, L) of each pair's side-`side` image at voxel coordinates vox (P, L, 3), read from
        the sampler's image volume (exact plane depths, see LineSampler)."""
        iv = self.sampler.img
        idx = self.IA if side == 0 else self.IB
        z = iv.depth[idx][:, None].expand_as(vox[..., 0])
        # clamped just outside the image: points outside it have weight 0 anyway (sampled as 0 either way), but the
        # crossing line of nearly parallel neighbours lies extremely far away, and its coordinates would overflow a
        # half-precision grid to inf -- whose gradient (inf x 0 weight) is NaN
        xy = torch.stack([2 * vox[..., 1] / (iv.Wmax - 1) - 1, 2 * vox[..., 0] / (iv.Hmax - 1) - 1], -1).clamp(-1.1, 1.1)
        grid = torch.cat([xy, z[..., None]], -1)
        return iv.sample(grid[None, :, :, None, :])[:, :, :, 0].float().transpose(0, 1)       # (P, F, L)

    def profiles(self, theta_deg: torch.Tensor, t: torch.Tensor, g: dict | None = None):
        """At the given poses: both images' profiles (P, F, L) x 2, the point weights (P, L), which pairs cross at
        all (P,) and the distance (mm) of each pair's line from the long axis (P,). g: geometry() at these poses."""
        g = self.geometry(theta_deg, t) if g is None else g
        prof = [self._sample(side, g["vox"][side]) for side in (0, 1)]
        return prof[0], prof[1], g["w"], g["active"], g["axis_dist"]

    def geometry(self, theta_deg: torch.Tensor, t: torch.Tensor) -> dict:
        """The crossing line of every pair at the given poses: anchor x0 (P, 3; the line's point closest to the long
        axis) and unit direction d (P, 3), so the samples are x0 + T d; which pairs cross at all (active, P), which
        samples are inside both images (in_img, P x L) and within --sax-sax-radius-mm of the axis (in_radius), the
        samples' loss weights (w, P x L: inside both images x the falloff with the axis distance), the axis distance
        (P,) and the samples' voxel coordinates in both images (vox, 2 x (P, L, 3))."""
        smp = self.sampler
        IA, IB = self.IA, self.IB
        R = rodrigues(theta_deg * (math.pi / 180.0))
        Cp = smp.C + t
        Op = (R @ (smp.O - smp.C)[..., None])[..., 0] + Cp
        Np = (R @ smp.Nrm[..., None])[..., 0]
        na, nb = Np[IA], Np[IB]
        d = torch.cross(na, nb, dim=1)
        nd = torch.sqrt((d * d).sum(1, keepdim=True).clamp_min(1e-12))              # sin(crossing angle)
        active = nd[:, 0] > 1e-3
        dh = d / nd.clamp_min(1e-3)
        # a point on the intersection line (3x3 solve with det = |na x nb| = nd, kept finite for parallel slices,
        # whose points are masked out below), then the point on that line closest to the long axis
        L0, ell = self.L0p, self.ellp                                                 # each pair's axis (P, 3)
        c12, c20, c01 = torch.cross(nb, dh, dim=1), torch.cross(dh, na, dim=1), torch.cross(na, nb, dim=1)
        rhs = torch.stack([(na * Op[IA]).sum(1), (nb * Op[IB]).sum(1), (dh * L0).sum(1)], 1)
        p = (rhs[:, 0:1] * c12 + rhs[:, 1:2] * c20 + rhs[:, 2:3] * c01) / nd.clamp_min(1e-3)
        w0 = p - L0
        b = (dh * ell).sum(1)
        dd, e = (dh * w0).sum(1), (ell * w0).sum(1)
        denom = (1.0 - b * b).clamp_min(1e-3)
        s, tau = (b * e - dd) / denom, (e - b * dd) / denom
        x0 = p + s[:, None] * dh
        gap = w0 + s[:, None] * dh - tau[:, None] * ell
        axis_dist = torch.sqrt((gap * gap).sum(1).clamp_min(1e-12))
        pts = x0[:, None, :] + self.T[None, :, None] * dh[:, None, :]                  # (P, L, 3)
        rel = pts - L0[:, None, :]
        perp = rel - (rel * ell[:, None, :]).sum(-1, keepdim=True) * ell[:, None, :]
        r2 = (perp * perp).sum(-1)
        in_radius = r2 <= self.radius ** 2
        falloff = torch.exp(-0.5 * r2 / self.radius ** 2) if self.falloff == "gaussian" else in_radius.float()
        vox_both, inside = [], []
        for side, idx in enumerate((IA, IB)):
            A = torch.einsum("pij,pkj->pik", smp.inv_rot[idx], R[idx])                # inv_rot @ R^T
            bvec = (torch.einsum("pij,pj->pi", smp.inv_rot[idx], smp.C[idx]) + smp.inv_trans[idx]
                    - torch.einsum("pij,pj->pi", A, Cp[idx]))
            vox = torch.einsum("plj,pij->pli", pts, A) + bvec[:, None, :]             # (P, L, 3)
            hw = self.size[side][:, None, :]
            inside.append((vox[..., 0] >= 0) & (vox[..., 0] <= hw[..., 0]) & (vox[..., 1] >= 0) & (vox[..., 1] <= hw[..., 1]))
            vox_both.append(vox)
        in_img = inside[0] & inside[1] & active[:, None]
        return dict(x0=x0, d=dh, active=active, in_img=in_img, in_radius=in_img & in_radius, w=in_img.float() * falloff,
                    axis_dist=axis_dist, vox=vox_both)

    def gaps(self, theta_deg: torch.Tensor, t: torch.Tensor) -> dict:
        """Per pair at the given poses: the gap (mm, P x 2 x D) between the neighbours at the disk points on each slice
        (signed along the acquired stacking direction: ~ +spacing as acquired, < 0 where the slices have crossed) and
        their relative tilt (deg, P; angle between the posed normals)."""
        smp = self.sampler
        IA, IB = self.IA, self.IB
        R = rodrigues(theta_deg * (math.pi / 180.0))
        Cp = smp.C + t
        Np = (R @ smp.Nrm[..., None])[..., 0]
        Up = (R @ self.U[..., None])[..., 0]
        Vp = (R @ self.V[..., None])[..., 0]
        out = []
        # points on slice a measured against slice b: gap = -s (x - Cb).nb; on slice b against a: gap = s (x - Ca).na
        L0, ell = self.L0p, self.ellp                                                 # each pair's axis (P, 3)
        for own, other, side_sign in ((IA, IB, -1.0), (IB, IA, 1.0)):
            n = Np[own]
            cos = (ell * n).sum(1)
            cos = torch.where(cos >= 0, 1.0, -1.0) * cos.abs().clamp_min(0.1)
            centre = L0 + (((Cp[own] - L0) * n).sum(1) / cos)[:, None] * ell              # the axis x this slice
            pts = (centre[:, None, :] + self.disk[None, :, 0:1] * Up[own][:, None, :]
                   + self.disk[None, :, 1:2] * Vp[own][:, None, :])                          # (P, D, 3)
            dist = ((pts - Cp[other][:, None, :]) * Np[other][:, None, :]).sum(-1)
            out.append(side_sign * self.stack_sign[:, None] * dist)
        chord2 = ((Np[IA] - Np[IB]) ** 2).sum(1).clamp_min(1e-12)       # |na - nb| = 2 sin(angle / 2)
        tilt = 2.0 * torch.asin((chord2.sqrt() / 2.0).clamp(max=1.0)) * (180.0 / math.pi)
        return dict(gap=torch.stack(out, 1), rel_tilt=tilt)

    def barrier(self, theta_deg: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """--sax-gap-reg term (mm^2): mean over pairs, both slices and the disk points of softplus(margin - gap)^2."""
        gap = self.gaps(theta_deg, t)["gap"]
        return F.softplus(self.margins - gap).pow(2).mean()

    def dissimilarity(self, theta_deg: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """--sax-sax-weight term: sum over pairs of (sum of point weights) x (1 - NCC), unnormalized."""
        c, n, _ = self.scores(theta_deg, t)
        return (n * (1.0 - c)).sum()

    @torch.no_grad()
    def segment_rows(self, theta_deg: torch.Tensor, t: torch.Tensor, shift: np.ndarray) -> list[dict]:
        """One row per pair describing its crossing line at the given poses, in the lines.csv convention: points
        x0 + t d (world mm, minus `shift` -- the translation convention of history.csv, relative to the mean LAX pose);
        t_img_lo/hi bound the part inside both images, t_used_lo/hi its part within --sax-sax-radius-mm of the long
        axis (the part the dissimilarity term weighs fully; with the Gaussian falloff points beyond it weigh less);
        NaN where there is none (the slices do not cross, or not inside the images). Also the QC values: point weight
        sum, NCC and dissimilarity (1 - NCC) along the line, the smallest gap between the slices on the barrier disk
        and their relative tilt; crossing_in_heart = 1 where that gap is negative (the slices cross inside the disk)."""
        return self._rows(self._packed(theta_deg, t), shift)

    @torch.no_grad()
    def segment_rows_many(self, poses: list[tuple[torch.Tensor, torch.Tensor]], shifts: list[np.ndarray]) -> list[list[dict]]:
        """segment_rows of several poses (theta_deg, t) at once: the S poses of the N planes are evaluated as one set
        of S x N planes (see _stacked), in one pass over the GPU and one transfer (~20x faster than one call per
        pose). Equal to segment_rows up to float32 rounding (the batched kernels round differently): relative ~1e-5,
        more in absolute terms where nearly parallel neighbours cross far away (an ill-conditioned crossing point)."""
        if not poses:
            return []
        S, P = len(poses), self.P
        packed = self._stacked(S)._packed(torch.cat([th for th, _ in poses]), torch.cat([tt for _, tt in poses]))
        return [self._rows(packed[k * P:(k + 1) * P], shifts[k]) for k in range(S)]

    def _stacked(self, S: int) -> "SaxNeighbourLines":
        """A view of these pairs repeated S times over S copies of the planes (pair k of copy s joins planes
        IA[k] + s N and IB[k] + s N): evaluates S poses at once. Shares the images (and their blur)."""
        cache = self.__dict__.setdefault("_stacked_cache", {})
        if S not in cache:
            import copy
            from types import SimpleNamespace
            smp, iv = self.sampler, self.sampler.img
            N = smp.C.shape[0]
            off = (torch.arange(S, device=self.device) * N)[:, None]
            rp = lambda x: x.repeat(S, *([1] * (x.dim() - 1)))  # noqa: E731
            st = copy.copy(self)
            st.__dict__.pop("_stacked_cache", None)
            st.IA, st.IB = (self.IA[None] + off).reshape(-1), (self.IB[None] + off).reshape(-1)
            st.L0p, st.ellp, st.stack_sign, st.margins = rp(self.L0p), rp(self.ellp), rp(self.stack_sign), rp(self.margins)
            st.size = [rp(x) for x in self.size]
            st.U, st.V = rp(self.U), rp(self.V)
            st.FW = None if self.FW is None else rp(self.FW)
            st.pairs, st.images_a, st.images_b, st.kinds = self.pairs * S, self.images_a * S, self.images_b * S, self.kinds * S
            st.P, st.cross = self.P * S, rp(self.cross)
            st.sampler = SimpleNamespace(C=rp(smp.C), O=rp(smp.O), Nrm=rp(smp.Nrm), inv_rot=rp(smp.inv_rot),
                                         inv_trans=rp(smp.inv_trans),
                                         img=SimpleNamespace(depth=rp(iv.depth), Wmax=iv.Wmax, Hmax=iv.Hmax, sample=iv.sample))
            cache[S] = st
        return cache[S]

    def _packed(self, theta_deg: torch.Tensor, t: torch.Tensor) -> np.ndarray:
        """segment_rows' values of every pair, packed into one host array (P, 12 + 2 L): one transfer, one GPU sync."""
        g = self.geometry(theta_deg, t)
        c, n, _ = self.scores(theta_deg, t, g)
        gp = self.gaps(theta_deg, t)
        return torch.cat([gp["gap"].flatten(1).min(1).values[:, None], gp["rel_tilt"][:, None], g["x0"], g["d"],
                          g["active"].float()[:, None], g["axis_dist"][:, None], c[:, None], n[:, None],
                          g["in_img"].float(), g["in_radius"].float()], 1).cpu().numpy()

    def _rows(self, packed: np.ndarray, shift: np.ndarray) -> list[dict]:
        """segment_rows from _packed values."""
        L = self.T.numel()
        gap_min, tilt, x0, d = packed[:, 0], packed[:, 1], packed[:, 2:5].astype(np.float64) - shift, packed[:, 5:8]
        active, dist, c, n = packed[:, 8] > 0.5, packed[:, 9], packed[:, 10], packed[:, 11]
        in_img, used = packed[:, 12:12 + L] > 0.5, packed[:, 12 + L:12 + 2 * L] > 0.5
        T = self.T_np
        nan = float("nan")
        rows = []
        for k, (ia, ib) in enumerate(self.pairs):
            ti, tu = T[in_img[k]], T[used[k]]
            counts = n[k] > 1e-3
            row = dict(image_a=self.images_a[k], slice_a=self.sampler.planes[ia].slice_idx,
                       image_b=self.images_b[k], slice_b=self.sampler.planes[ib].slice_idx,
                       kind=self.kinds[k],
                       crossing=int(active[k]), n_points=float(n[k]), supervised=int(self.weight > 0 and counts),
                       axis_dist_mm=float(dist[k]) if active[k] else nan,
                       ncc=float(c[k]) if counts else nan, dissim=float(1.0 - c[k]) if counts else nan,
                       gap_min_mm=float(gap_min[k]), rel_tilt_deg=float(tilt[k]), crossing_in_heart=int(gap_min[k] < 0),
                       x0_x=nan, x0_y=nan, x0_z=nan, d_x=nan, d_y=nan, d_z=nan,
                       t_used_lo=float(tu.min()) if tu.size else nan, t_used_hi=float(tu.max()) if tu.size else nan,
                       t_img_lo=float(ti.min()) if ti.size else nan, t_img_hi=float(ti.max()) if ti.size else nan)
            if active[k]:
                row.update(x0_x=float(x0[k, 0]), x0_y=float(x0[k, 1]), x0_z=float(x0[k, 2]),
                           d_x=float(d[k, 0]), d_y=float(d[k, 1]), d_z=float(d[k, 2]))
            rows.append(row)
        return rows

    def scores(self, theta_deg: torch.Tensor, t: torch.Tensor, g: dict | None = None):
        """Per pair: NCC along its line with the point weights (between modalities: --cross-mod-metric; 0 when it does
        not cross), the sum of the point weights and the axis distance (mm). g: geometry() at these poses, if already
        computed."""
        a, b, w, active, dist = self.profiles(theta_deg, t, g)
        # NCC is invariant to the scale of the weights, but ncc() floors their sum at 1 (meant for 0/1 masks), so
        # soft weights are rescaled to a maximum of 1 per line first
        wn = w / w.amax(-1, keepdim=True).clamp_min(1e-6)
        c = frame_mean(ncc(a, b, wn[:, None, :], False), self.FW)                    # over the pair's frames
        cross = cross_mod_metric(self.args)
        if self.any_cross and cross != "ncc":                     # pairs between modalities: --cross-mod-metric
            c = torch.where(self.cross, metric_rows(cross, a, b, wn, self.args, self.FW), c)
        c = c * active.float()
        return c, w.sum(-1), dist


def new_sampler(planes: list[Plane], lines: list[tuple[int, int, Line]], args, device,
                images: ImageVolume | None = None) -> LineSampler:
    """A LineSampler with the run's PSF settings and image precision (--sample-dtype); `images`: the ImageVolume of
    another sampler of the same planes, to share (default: a new one)."""
    return LineSampler(planes, lines, args.n_band, device, args.psf_model, args.psf_tbw, getattr(args, "psf_scale_start", 1.0),
                       sample_dtype=getattr(args, "sample_dtype", "float32"), images=images)


def neighbour_kind(view_a: str, view_b: str | None = None) -> str:
    """pairs.csv / sax_lines.csv kind of a pair of parallel slices of these views (see attach_sax_pairs)."""
    return "SAX-SAX" if view_a == "SAX" and (view_b or view_a) == "SAX" else "STACK-STACK"


NEIGHBOUR_KINDS = ("SAX-SAX", "STACK-STACK")     # recorded neighbour rows, left out of the line-pair statistics


def attach_sax_pairs(sampler: LineSampler, planes: list[Plane], sax_planes: list[Plane], lax_planes: list[Plane],
                     args, device) -> None:
    """Give a sampler its parallel-pair terms. Two slices are parallel when they are closer in angle than
    --min-pair-angle-deg (so never an intersection pair) and at most --parallel-max-dist-mm apart along the normal,
    whatever their views and modalities: in a stack its adjacent slices (directly above and below each other, so a gap
    in the stack is not bridged), and between two images each slice and the nearest such slice of the other image whose
    image overlaps it in-plane (e.g. the same view in two modalities, or two SAX stacks).

    All of them get the neighbouring-slice terms (SaxNeighbourLines; --sax-sax-weight / --sax-gap-reg > 0, with logging
    also at weight 0, so that sax_lines.csv, the neighbour rows of pairs.csv and the summary's QC columns are always
    recorded): pairs of SAX slices are weighted around the LV long axis (heart_axis), a stack's other pairs around its
    own axis (through its centre along its normal), other pairs around the axis through their centres. Those whose
    slabs overlap (closer than half their summed thicknesses: near-coplanar) are also compared over their shared area
    with --coplanar-weight > 0 (CoplanarPairs)."""
    sampler.sax_pairs = None
    sampler.coplanar = None
    supervise = ((getattr(args, "sax_sax_weight", 0.0) > 0 and args.dof == 6) or getattr(args, "sax_gap_reg", 0.0) > 0)
    coplanar = getattr(args, "coplanar_weight", 0.0) > 0
    if not (supervise or coplanar or getattr(args, "logging", True)):
        return
    lv_axis = heart_axis(sax_planes, lax_planes)
    index = {p.key: i for i, p in enumerate(planes)}
    cos_min = math.cos(math.radians(getattr(args, "min_pair_angle_deg", 20.0)))
    max_dist = getattr(args, "parallel_max_dist_mm", 12.0)

    def parallel(a, b):
        return abs(float(a.normal @ b.normal)) > cos_min

    def offset(a, b):                                      # distance along the normal (mm)
        return abs(float((b.center - a.center) @ a.normal))

    def half_extent(p):
        return 0.5 * math.hypot(p.H * p.spacing[0], p.W * p.spacing[1])

    def overlap(a, b):                                     # the two images share part of their field of view
        d = b.center - a.center
        return float(np.linalg.norm(d - (d @ a.normal) * a.normal)) < half_extent(a) + half_extent(b)

    stacks = {}
    for p in sax_planes + lax_planes:
        stacks.setdefault(p.image, []).append(p)
    found = {}                                             # (plane index a, plane index b) -> axis
    for image, ps in stacks.items():                       # within a stack: its adjacent slices
        if len(ps) < 2:
            continue
        centre, n = stack_frame(ps)
        axis = lv_axis if ps[0].view == "SAX" and lv_axis is not None else (centre, n)
        ordered = sorted(ps, key=lambda p: float(p.center @ n))
        for a, b in zip(ordered, ordered[1:]):
            if parallel(a, b) and offset(a, b) <= max_dist:
                a, b = sorted((a, b), key=lambda p: p.slice_idx)          # rows as slice z, z + 1
                found[(index[a.key], index[b.key])] = axis
    for a in sax_planes + lax_planes:                      # between images: each slice's nearest parallel slice
        for image, ps in stacks.items():
            if image == a.image:
                continue
            near = [b for b in ps if parallel(a, b) and offset(a, b) <= max_dist and overlap(a, b)]
            if not near:
                continue
            b = min(near, key=lambda b: offset(a, b))
            key = tuple(sorted((index[a.key], index[b.key])))
            if key not in found:
                pa, pb = planes[key[0]], planes[key[1]]
                found[key] = (lv_axis if pa.view == "SAX" and pb.view == "SAX" and lv_axis is not None
                              else ((pa.center + pb.center) / 2.0, pa.normal))
    pairs = list(found)
    if pairs and (supervise or getattr(args, "logging", True)):
        sampler.sax_pairs = SaxNeighbourLines(sampler, planes, pairs, [found[k] for k in pairs], args, device,
                                              axis=lv_axis)
    if coplanar:
        near = [(ia, ib) for ia, ib in pairs
                if offset(planes[ia], planes[ib]) < 0.5 * (planes[ia].thickness + planes[ib].thickness)]
        if near:
            sampler.coplanar = CoplanarPairs(sampler, planes, near, lv_axis, args, device)


class CoplanarPairs:
    """--coplanar-weight: near-coplanar slices -- parallel (attach_sax_pairs) and so close along the normal that their
    slabs overlap (less than half their summed thicknesses apart; e.g. the same view in two modalities, or densely
    spaced slices of a stack) -- image the same tissue, so their images are compared over the area they share.

    A grid of points on slice a (every --coplanar-step-mm, on a's pixel grid at the acquired geometry), inside the heart
    region (the union of the two slices' dilated heart masks; without masks the points within --sax-sax-radius-mm of
    the LV long axis; without that everywhere) and inside slice b as acquired (cropped to their bounding box). At the
    current poses each point is sampled in slice a and, at its projection onto slice b, in slice b, weighted by b's
    slice profile at the point's distance from b's plane and 0 outside b. The grid's rows are scored like intersection
    lines -- --metric within a modality, --cross-mod-metric between modalities, over the frames both slices have -- and
    a pair's score is the mean over its rows weighted by their point weights. The term --coplanar-weight x the mean
    score over the pairs is added to the objective."""

    def __init__(self, sampler: LineSampler, planes: list[Plane], pairs: list[tuple[int, int]],
                 axis: tuple[np.ndarray, np.ndarray] | None, args, device):
        self.sampler, self.device, self.args = sampler, device, args
        self.weight = float(getattr(args, "coplanar_weight", 0.0))
        self.step = float(getattr(args, "coplanar_step_mm", 2.0))
        radius = float(getattr(args, "sax_sax_radius_mm", 120.0))
        grids = []                                         # per pair: (voxel (R, L, 2) in a, world (R, L, 3), weight (R, L))
        for ia, ib in pairs:
            a, b = planes[ia], planes[ib]
            ii = np.arange(0.0, a.H - 1 + 1e-6, max(self.step / a.spacing[0], 1e-3))
            jj = np.arange(0.0, a.W - 1 + 1e-6, max(self.step / a.spacing[1], 1e-3))
            gi, gj = np.meshgrid(ii, jj, indexing="ij")
            X = (a.affine @ np.stack([gi, gj, np.zeros_like(gi), np.ones_like(gi)], 0).reshape(4, -1))[:3].T
            pts = torch.tensor(X, dtype=torch.float32, device=device)
            with torch.no_grad():
                keep = inside(b, pts).cpu().numpy()
                if a.mask is not None or b.mask is not None:
                    region = np.zeros(len(X), bool)
                    for p in (a, b):
                        if p.mask is not None:
                            region |= (sample_image(p.mask, voxel_grid(p, pts), "nearest")[0] > 0.5).cpu().numpy()
                elif axis is not None:
                    d = axis[1] / np.linalg.norm(axis[1])
                    rel = X - axis[0]
                    region = np.linalg.norm(rel - (rel @ d)[:, None] * d, axis=1) <= radius
                else:
                    region = np.ones(len(X), bool)
            w = (keep & region).reshape(gi.shape)
            if not w.any():
                continue
            r, c = np.nonzero(w)
            r0, r1, c0, c1 = r.min(), r.max() + 1, c.min(), c.max() + 1
            grids.append((ia, ib, np.stack([gi, gj], -1)[r0:r1, c0:c1], X.reshape(*gi.shape, 3)[r0:r1, c0:c1],
                          w[r0:r1, c0:c1].astype(np.float32)))
        self.pairs = [(ia, ib) for ia, ib, *_ in grids]
        self.P = len(self.pairs)
        if not self.P:
            return
        L = max(g[2].shape[1] for g in grids)

        def pad(x):                                        # rows padded to L points (weight 0) by repeating the last
            return np.concatenate([x, np.repeat(x[:, -1:], L - x.shape[1], 1)], 1) if x.shape[1] < L else x

        f32 = dict(dtype=torch.float32, device=device)
        self.vox_a = torch.tensor(np.concatenate([pad(g[2]) for g in grids]), **f32)            # (R, L, 2)
        self.X = torch.tensor(np.concatenate([pad(g[3]) for g in grids]), **f32)                # (R, L, 3)
        self.w0 = torch.tensor(np.concatenate([np.pad(g[4], ((0, 0), (0, L - g[4].shape[1]))) for g in grids]), **f32)
        rows = [g[2].shape[0] for g in grids]
        self.ra = torch.tensor(np.repeat([g[0] for g in grids], rows), device=device)
        self.rb = torch.tensor(np.repeat([g[1] for g in grids], rows), device=device)
        self.row_sum = BlockSum(rows, device)               # per-pair sums over its rows (deterministic)
        self.size_b = torch.tensor([[planes[i].H - 1, planes[i].W - 1] for i in self.rb.tolist()], **f32)
        self.thick_b = torch.tensor([planes[i].thickness for i in self.rb.tolist()], **f32)
        self.cross = torch.tensor(np.repeat([planes[ia].modality != planes[ib].modality for ia, ib in self.pairs], rows),
                                  device=device)
        self.any_cross = bool(self.cross.any())
        fv = sampler.frame_valid
        self.FW = None if sampler.FW is None else fv[self.ra] * fv[self.rb]

    def _sample(self, idx: torch.Tensor, ij: torch.Tensor) -> torch.Tensor:
        """Bilinear samples (R, F, L) of the planes idx (R,) at voxel coordinates ij (R, L, 2) (see SaxNeighbourLines)."""
        iv = self.sampler.img
        z = iv.depth[idx][:, None].expand_as(ij[..., 0])
        xy = torch.stack([2 * ij[..., 1] / (iv.Wmax - 1) - 1, 2 * ij[..., 0] / (iv.Hmax - 1) - 1], -1).clamp(-1.1, 1.1)
        grid = torch.cat([xy, z[..., None]], -1)
        return iv.sample(grid[None, :, :, None, :])[:, :, :, 0].float().transpose(0, 1)

    def pair_scores(self, theta_deg: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Per pair at the given poses: its score (row-weighted mean similarity) and its total point weight (P,)."""
        smp = self.sampler
        R = rodrigues(theta_deg * (math.pi / 180.0))
        ra, rb = self.ra, self.rb
        Ca, Cb = smp.C[ra], smp.C[rb]
        # slice a's grid points at its pose, then slice b's pose undone and b's world -> voxel map applied
        p = torch.einsum("rij,rlj->rli", R[ra], self.X - Ca[:, None]) + (Ca + t[ra])[:, None]
        q = torch.einsum("rji,rlj->rli", R[rb], p - (Cb + t[rb])[:, None]) + Cb[:, None]
        vox = torch.einsum("rij,rlj->rli", smp.inv_rot[rb], q) + smp.inv_trans[rb][:, None]
        nb = (R[rb] @ smp.Nrm[rb][..., None])[..., 0]
        dist = ((p - (Cb + t[rb])[:, None]) * nb[:, None]).sum(-1)                     # mm from b's plane
        prof = smp.slice_profile(dist / self.thick_b[:, None])
        prof = prof / smp.slice_profile(torch.zeros((), device=self.device))
        hw = self.size_b[:, None, :]
        in_b = (vox[..., 0] >= 0) & (vox[..., 0] <= hw[..., 0]) & (vox[..., 1] >= 0) & (vox[..., 1] <= hw[..., 1])
        w = self.w0 * prof * in_b.float()
        a = self._sample(ra, self.vox_a)
        b = self._sample(rb, vox[..., :2])
        sim = metric_rows(self.args.metric, a, b, w, self.args, self.FW, self.step)
        cross = cross_mod_metric(self.args)
        if self.any_cross and cross != self.args.metric:
            sim = torch.where(self.cross, metric_rows(cross, a, b, w, self.args, self.FW, self.step), sim)
        rw = w.sum(-1)
        wsum = self.row_sum(rw)
        return self.row_sum(sim * rw) / wsum.clamp_min(1e-6), wsum

    def score(self, theta_deg: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """The mean score over the pairs that overlap at the given poses."""
        sc, wsum = self.pair_scores(theta_deg, t)
        ok = (wsum > 1e-6).float()
        return (sc * ok).sum() / ok.sum().clamp_min(1.0)


def coplanar_term(smp: LineSampler, theta_deg, t):
    """The --coplanar-weight term added to the objective (CoplanarPairs); 0 when off."""
    cp = getattr(smp, "coplanar", None)
    if cp is None or cp.weight <= 0 or not cp.P:
        return 0.0
    return cp.weight * cp.score(theta_deg, t)


def similarity_objective(smp: LineSampler, theta_deg, t, args, sel=None, sim=None) -> torch.Tensor:
    """The optimized similarity: the point-weighted mean over the SAX-LAX / LAX-LAX lines (npts weights), or over the
    lines `sel` only (an index tensor; only those are sampled). `sim`: the per-line similarities of smp if already
    computed (all lines)."""
    c = smp.similarity_lines(theta_deg, t, args, sel=sel) if sim is None else sim
    w = smp.npts if sel is None else smp.npts[sel]
    return (c * w).sum() / w.sum()


def sax_pair_penalty(smp: LineSampler, theta_deg, t):
    """The neighbouring-SAX-slice loss terms (see SaxNeighbourLines): --sax-sax-weight x the crossing dissimilarity
    per SAX-LAX/LAX-LAX line point, plus --sax-gap-reg x the gap barrier (mm^2). 0 when both are off."""
    sp = smp.sax_pairs
    pen = 0.0
    if sp is not None and sp.weight > 0:
        pen = pen + sp.weight * sp.dissimilarity(theta_deg, t) / smp.npts.sum()
    if sp is not None and sp.gap_weight > 0:
        pen = pen + sp.gap_weight * sp.barrier(theta_deg, t)
    return pen


class GraphSGD(torch.optim.Optimizer):
    """SGD with optional (Nesterov) momentum, written with tensor ops only so that a step can be captured in a CUDA
    graph while the learning rates (0-dim device tensors) are updated in place between replays. torch.optim.SGD reads
    a tensor learning rate back to the host, which is not allowed during capture."""

    def __init__(self, params, momentum: float = 0.0, nesterov: bool = False):
        super().__init__(params, dict(lr=1.0, momentum=momentum, nesterov=nesterov))
        for g in self.param_groups:
            for p in g["params"]:
                self.state[p]["momentum_buffer"] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure=None):
        for g in self.param_groups:
            mu = g["momentum"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                d = p.grad
                if mu > 0:
                    buf = self.state[p]["momentum_buffer"]
                    buf.mul_(mu).add_(d)
                    d = d + mu * buf if g["nesterov"] else buf
                p.sub_(d * g["lr"])


GRAD_SCALE = 2.0 ** 10   # loss scaling of the backward pass (align.body): keeps half-precision sampling gradients exact
                         # (2^8 already matches float32 on a DICOM subject; larger scales risk float16 overflow)
OPTIMIZERS = ("adam", "nadam", "radam", "adamax", "rmsprop", "adagrad", "sgd")
# default learning rates (translation mm, rotation deg) per optimizer when --lr / --rot-lr are not given. The adaptive
# methods take steps of about lr per iteration whatever the gradient scale; Adagrad's steps shrink like 1/sqrt(step),
# so it needs a larger lr; plain SGD steps are lr * gradient, and the NCC objective's gradients are small (see --lr).
LR_END_FRACTION = 0.5                    # default end learning rate of the linear schedule, as a fraction of the start
DRAW_AHEAD = 64                          # batched line dropping: steps whose kept lines are drawn at once
DEFAULT_LR = {"adam": (0.2, 0.04), "nadam": (0.2, 0.04), "radam": (0.2, 0.04), "adamax": (0.2, 0.04),
              "rmsprop": (0.2, 0.04), "adagrad": (1.0, 0.2), "sgd": (80.0, 20.0)}


def make_optimizer(param_groups: list[dict], args, device):
    """The optimizer chosen by --optimizer; every one of them can be captured in a CUDA graph with tensor learning
    rates (checked: graph replay and eager steps give identical results)."""
    name = getattr(args, "optimizer", "adam")
    if name == "sgd":
        return GraphSGD(param_groups, momentum=getattr(args, "momentum", 0.9), nesterov=getattr(args, "nesterov", False))
    if name == "adagrad":
        return torch.optim.Adagrad(param_groups)
    cls = {"adam": torch.optim.Adam, "nadam": torch.optim.NAdam, "radam": torch.optim.RAdam, "adamax": torch.optim.Adamax,
           "rmsprop": torch.optim.RMSprop}[name]
    return cls(param_groups, capturable=device.type == "cuda")


def stack_terms(theta_deg, t, C0, N0, nb_a, nb_b, target, which, tilt_deg: float = 5.0) -> dict:
    """Unweighted SAX stack regularizer terms over adjacent slice pairs (nb_a[k], nb_b[k]), each a mean over pairs.

    They split the relative pose of two neighbours into complementary parts, and all are zero when the stack moves as
    one rigid body (at the target spacing). (A tilt about a slice's own centre also moves its neighbour's centre
    slightly off its normal axis, so rel_shift sees a small part of it: ~0.03 mm^2 per 3 deg at 10 mm spacing.)
      tilt_hinge relative tilt above tilt_deg: relu(angle between the posed normals - tilt_deg)^2 (deg^2); zero for
                 the few-degree tilts of real breath-hold motion, rising for outlier orientations
      rel_rot    relative in-plane rotation: the component along the acquired normal of the relative rotation
                 R_a^T R_b (~ deg^2)
      spacing    (distance of the posed centres along the mean posed normal - target)^2 (mm^2, signed target)
      rel_shift  in-plane offset: the posed offset a -> b seen from each slice's own frame, minus the acquired offset,
                 its part perpendicular to the acquired normal, mean of both frames (mm^2)
    `which` selects the terms to compute. C0, N0: acquired plane centres and normals (N, 3).
    """
    deg2 = (180.0 / math.pi) ** 2
    R = rodrigues(theta_deg * (math.pi / 180.0))
    Np = (R @ N0[..., None])[..., 0]
    na, nb = Np[nb_a], Np[nb_b]
    n0 = N0[nb_a]
    d = C0[nb_b] + t[nb_b] - C0[nb_a] - t[nb_a]                             # posed centre offset a -> b
    out = {}
    if "tilt_hinge" in which:                # |na - nb| = 2 sin(angle / 2); clamped so the gradient stays finite at 0
        chord2 = (na - nb).pow(2).sum(1).clamp_min(1e-12)
        angle = 2.0 * torch.asin((chord2.sqrt() / 2.0).clamp(max=1.0)) * (180.0 / math.pi)
        out["tilt_hinge"] = torch.relu(angle - tilt_deg).pow(2).mean()
    if "spacing" in which:
        n = na + nb
        n = n / n.norm(dim=1, keepdim=True)
        out["spacing"] = ((d * n).sum(1) - target).pow(2).mean()
    if "rel_shift" in which:
        d0 = C0[nb_b] - C0[nb_a]
        err = 0.0
        for Ri in (R[nb_a], R[nb_b]):
            e = (Ri.transpose(1, 2) @ d[..., None])[..., 0] - d0
            e = e - (e * n0).sum(1, keepdim=True) * n0
            err = err + 0.5 * e.pow(2).sum(1)
        out["rel_shift"] = err.mean()
    if "rel_rot" in which:                   # R_a^T R_b = rotation by phi about k; w = sin(phi) k
        Rr = R[nb_a].transpose(1, 2) @ R[nb_b]
        w = 0.5 * torch.stack([Rr[:, 2, 1] - Rr[:, 1, 2], Rr[:, 0, 2] - Rr[:, 2, 0], Rr[:, 1, 0] - Rr[:, 0, 1]], 1)
        out["rel_rot"] = deg2 * (w * n0).sum(1).pow(2).mean()
    return out


def pose_model(n_sax: int, n_lax: int, dof: int, n_lax_stacks: int = 0) -> str:
    """The pose model a subject's planes support, given its number of SAX slices, of usable non-SAX images and of
    those that are stacks (more than one slice).

    6dof: >= 3 images (SAX counts as one) of which at least one is a stack (SAX slices, or e.g. an oblique cine stack):
      every plane is crossed by planes of two other orientations, so its tilt is constrained. inplane: two images, or
      single-slice LAX views only; each plane is crossed along a single direction (or by too few planes), so tilts are
      unconstrained and rotations can run into degenerate solutions -- only translation and rotation about the plane's
      own normal are optimized. none: fewer than two images; nothing is optimized. --dof 4 forces inplane (e.g. to
      leave the weakly constrained tilts out even where 6dof is possible); --dof 3 turns 6dof / inplane into
      translation.
    """
    n_images = (1 if n_sax else 0) + n_lax
    if n_images < 2:
        return "none"
    if dof == 3:
        return "translation"
    if dof == 4:
        return "inplane"
    return "6dof" if n_images >= 3 and (n_sax > 0 or n_lax_stacks > 0) else "inplane"


def step_progress_bar(args):
    """--step-progress: a tqdm bar over one subject's optimizer steps, below run()'s bar over the subjects and cleared
    when done; otherwise (or without tqdm) a context yielding None."""
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None
    if tqdm is None or not getattr(args, "step_progress", False):
        return contextlib.nullcontext(None)
    return tqdm(total=args.steps, desc="steps", unit="step", leave=False, position=1, dynamic_ncols=True)


class BlockSum:
    """Sums of values laid out block after block (sizes[g] values for block g, e.g. a batch's subjects): the values are
    gathered into a zero-padded (G, largest block) array and summed along it. No atomic additions, so the sums are
    bit-reproducible on the GPU (index_add is not: its float rounding varies from run to run, and an optimization
    amplifies that). Values in the padding never reach a sum (where, not a multiplication by 0, so a non-finite value
    of one block cannot spread to the others)."""

    def __init__(self, sizes: list[int], device):
        G, L = len(sizes), max([1] + list(sizes))
        idx = torch.zeros(G, L, dtype=torch.long)
        mask = torch.zeros(G, L, dtype=torch.bool)
        o = 0
        for g, n in enumerate(sizes):
            idx[g, :n] = torch.arange(o, o + n)
            mask[g, :n] = True
            o += n
        self.idx, self.mask = idx.to(device), mask.to(device)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """x (K,) or (K, d), K = sum(sizes) -> (G,) or (G, d)."""
        v = x[self.idx]
        m = self.mask if x.dim() == 1 else self.mask[..., None]
        return torch.where(m, v, torch.zeros((), dtype=v.dtype, device=v.device)).sum(1)


def align(final_sampler: LineSampler, args, stages: list[tuple[int, LineSampler]] | None = None,
          pose_model: str = "6dof", groups: list[int] | None = None, models: list[str] | None = None,
          anneal_ends: list[int] | None = None) -> dict:
    """Jointly optimize a rigid pose per plane from the acquired geometry: translation plus a 3D rotation (pose_model
    6dof), plus a rotation about the plane's own normal only (inplane), or translation only (translation).

    `stages` optionally lists (start_step, sampler) for the annealed mask geometry used early in the run, ending
    with (anneal_end, final_sampler); `final_sampler` -- the nominal geometry -- is used from anneal_end on (or always,
    if no stages are given). The best pose is only tracked once every annealed quantity (geometry, blur, PSF width) is
    at its nominal value, on the full (all lines) objective.

    Several subjects at once (align_batch): `groups` gives every plane's subject (0 .. G-1; the samplers hold the
    subjects' planes and lines one subject after the other), `models` each subject's pose model and `anneal_ends` the
    step its own mask annealing ends (0: none). Nothing couples the subjects: the objective is the sum of the subjects'
    own objectives (each a mean over its own lines, with its own gauge, priors and stack terms), so every pose gets the
    gradient it would get alone; steps with a non-finite gradient are skipped, the best pose is tracked and the lines
    are dropped (--pair-keep) per subject: each keeps its own share of its own lines, drawn for the whole batch at once. In-plane subjects use
    the first component of their rotation parameter as the angle about the plane normal (the others get no gradient),
    which updates exactly as a solo in-plane run. Returns best_step and n_skipped per subject (lists) in that case.

    Each optimization step (objective, backward, best-pose bookkeeping, optimizer update) is captured once per geometry
    stage as a CUDA graph and replayed, which removes the per-step Python/kernel-launch overhead; all per-step inputs
    (learning rates, blur, PSF scale, line-dropping mask, step counter) live in device buffers that are updated in
    place before each replay. --no-cuda-graph runs the identical step eagerly.
    """
    device, N = final_sampler.device, final_sampler.N
    if device.type == "cuda":                    # CUDA graphs allocate from a private pool: hand cached blocks back
        torch.cuda.empty_cache()
    stages = sorted(stages or [(0, final_sampler)], key=lambda s: s[0])
    anneal_end = stages[-1][0]
    # coarse early steps (--coarse-frac): the first steps sample every --coarse-frame-stride-th frame and every
    # --coarse-point-stride-th line point (LineSampler.coarse), a fraction of a full step's cost
    coarse_steps = int(round(getattr(args, "coarse_frac", 0.0) * args.steps))
    fs, ps = max(1, getattr(args, "coarse_frame_stride", 1)), max(1, getattr(args, "coarse_point_stride", 1))
    coarse = ({id(smp): smp.coarse(fs, ps) for _, smp in stages}
              if coarse_steps > 0 and (fs > 1 or ps > 1) else {})
    if not coarse:
        coarse_steps = 0
    all_samplers = [smp for _, smp in stages] + list(coarse.values())

    def sampler_at(step):
        active = stages[0][1]
        for start, smp in stages:
            if step >= start:
                active = smp
        return coarse.get(id(active), active) if step < coarse_steps else active

    f32 = dict(dtype=torch.float32, device=device)
    batched = groups is not None
    G = max(groups) + 1 if batched else 1
    models = list(models) if batched else [pose_model]
    single = G == 1
    pg = torch.tensor(groups if batched else [0] * N, dtype=torch.long, device=device)      # plane -> subject
    n_per = torch.bincount(pg, minlength=G).float()                                         # planes per subject
    if batched and any(b < a for a, b in zip(groups, groups[1:])):
        raise ValueError("align: the planes of a batch must be ordered by subject")
    plane_sum = BlockSum(n_per.long().tolist(), device)                # per-subject sums over its planes
    rot_ok = args.dof in (4, 6) and args.rot_lr > 0
    full_g = [rot_ok and m == "6dof" for m in models]
    inp_g = [rot_ok and m == "inplane" for m in models]
    rotate = any(full_g) or any(inp_g)
    all_full, all_inp = all(full_g), all(inp_g)
    gl = groups if batched else [0] * N
    full = torch.tensor([[float(full_g[g])] for g in gl], **f32)
    inp = torch.tensor([[float(inp_g[g])] for g in gl], **f32)
    # rotation parameter: the rotation vector (6dof), or in its first component the angle about the normal (inplane)
    theta_deg = torch.zeros(N, 3, **f32).requires_grad_(rotate)
    t = torch.zeros(N, 3, **f32).requires_grad_(True)

    def pose_theta():
        """Rotation vectors (N, 3, deg) of the current parameters."""
        if not rotate or all_full:
            return theta_deg
        if all_inp:
            return theta_deg[:, :1] * final_sampler.Nrm
        return full * theta_deg + inp * (theta_deg[:, :1] * final_sampler.Nrm)

    def gmean(x):
        """Per-plane values (N,) -> their mean over each subject's planes (G,)."""
        if single:
            return x.mean()[None]
        return plane_sum(x) / n_per

    def gmean3(v):
        """Per-plane vectors (N, 3) -> their mean over each subject's planes (G, 3)."""
        if single:
            return v.mean(0)[None]
        return plane_sum(v) / n_per[:, None]

    # SAX stack regularizers over adjacent slices (a, b = slice z, z + 1) of each subject; the plane normals and
    # centres are the same in every stage sampler
    tilt_prior = getattr(args, "tilt_prior", 0.0) if rotate else 0.0
    shift_prior = getattr(args, "shift_prior", 0.0)
    shift_free = getattr(args, "shift_free_mm", 20.0)
    rot_prior = getattr(args, "rot_prior", 0.0) if rotate else 0.0
    rot_free = getattr(args, "rot_free_deg", 10.0)
    w_tilt_hinge = getattr(args, "tilt_hinge_reg", 0.0) if rotate else 0.0
    tilt_hinge_deg = getattr(args, "tilt_hinge_deg", 5.0)
    w_spacing = getattr(args, "spacing_reg", 0.0)
    w_rel_shift = getattr(args, "rel_shift_reg", 0.0)
    w_rel_rot = getattr(args, "rel_rot_reg", 0.0) if rotate else 0.0
    weights = dict(tilt_hinge=w_tilt_hinge, spacing=w_spacing, rel_shift=w_rel_shift, rel_rot=w_rel_rot)
    adj = []                                                 # per subject: (nb_a, nb_b, target) or None
    C0, N0 = final_sampler.C, final_sampler.Nrm
    for g in range(G):
        adjacent = []                                        # within each SAX image (stack) of the subject
        for im in dict.fromkeys(p.image for i, p in enumerate(final_sampler.planes) if p.view == "SAX" and gl[i] == g):
            sax = sorted((p.slice_idx, i) for i, p in enumerate(final_sampler.planes) if p.image == im and gl[i] == g)
            adjacent += [(i, j) for (za, i), (zb, j) in zip(sax, sax[1:]) if zb == za + 1]
        if not adjacent:
            adj.append(None)
            continue
        nb_a = torch.tensor([i for i, _ in adjacent], device=device)
        nb_b = torch.tensor([j for _, j in adjacent], device=device)
        step0 = ((C0[nb_b] - C0[nb_a]) * N0[nb_a]).sum(1)                   # acquired signed spacing along the normal
        spacing = getattr(args, "sax_spacing_mm", 10.0)
        adj.append((nb_a, nb_b, torch.sign(step0) * (spacing if spacing > 0 else step0.abs())))
    stack_reg = any(a is not None for a in adj) and (w_tilt_hinge > 0 or w_spacing > 0 or w_rel_shift > 0 or w_rel_rot > 0)

    def stack_penalty(theta_deg, t):
        """The weighted stack terms of every subject (G,), each a mean over its own adjacent pairs."""
        vals = []
        for a in adj:
            if a is None:
                vals.append(torch.zeros((), **f32))
                continue
            terms = stack_terms(theta_deg, t, C0, N0, a[0], a[1], a[2], {k for k, w in weights.items() if w > 0},
                                tilt_hinge_deg)
            vals.append(sum(weights[k] * v for k, v in terms.items()))
        return torch.stack(vals)

    line_group = {}                                          # per sampler: the subject of every line (M,)
    line_sums = {}                                           # per sampler: BlockSum over its lines / its kept lines

    def score(smp, theta_deg, t, sel):
        """The similarity of every subject (G,): the point-weighted mean over its own (kept) lines."""
        if single:
            return similarity_objective(smp, theta_deg, t, args, sel)[None]
        c = smp.similarity_lines(theta_deg, t, args, sel=sel)
        w = smp.npts if sel is None else smp.npts[sel]
        bsum = line_sums[id(smp)][0 if sel is None else 1]   # the lines (kept lines) are grouped by subject
        return bsum(c * w) / bsum(w).clamp_min(1e-12)

    def objective(theta_deg, t, sel=None, smp=final_sampler):
        """The objective of every subject (G,); the optimizer maximizes their sum."""
        out = score(smp, theta_deg, t, sel)
        gauge = args.gauge * (gmean3(t).pow(2).sum(1) + gmean3(theta_deg).pow(2).sum(1))
        dz = (t * smp.Nrm).sum(1)
        prior = args.prior * gmean(theta_deg.pow(2).sum(1) + dz.pow(2))
        if tilt_prior > 0:                     # tilt above --tilt-hinge-deg (deg^2); in-plane rotation is free
            tilt = theta_deg - (theta_deg * smp.Nrm).sum(1, keepdim=True) * smp.Nrm
            tilt_deg = tilt.pow(2).sum(1).clamp_min(1e-12).sqrt()          # = rot_tilt_deg; finite gradient at 0
            prior = prior + tilt_prior * gmean(torch.relu(tilt_deg - tilt_hinge_deg).pow(2))
        if shift_prior > 0:                    # each plane's shift from its acquired pose beyond --shift-free-mm (mm^2)
            shift = t.pow(2).sum(1).clamp_min(1e-12).sqrt()                 # finite gradient at 0
            prior = prior + shift_prior * gmean(torch.relu(shift - shift_free).pow(2))
        if rot_prior > 0:                      # each plane's rotation angle beyond --rot-free-deg (deg^2)
            angle = theta_deg.pow(2).sum(1).clamp_min(1e-12).sqrt()
            prior = prior + rot_prior * gmean(torch.relu(angle - rot_free).pow(2))
        out = out - gauge - prior - (stack_penalty(theta_deg, t) if stack_reg else 0.0)
        if single:                             # (batches are only formed without these terms, see batchable)
            out = out - sax_pair_penalty(smp, theta_deg, t) + coplanar_term(smp, theta_deg, t)
        return out

    # stochastic line-pair dropping: each step samples only a random subset of round(--pair-keep x M) of the
    # intersection lines (at least one) -- per subject, from its own lines and its own generator -- so memory and time
    # per step scale with the kept lines; a fixed subset size keeps the shapes static for the CUDA graphs
    keep_frac = getattr(args, "pair_keep", 1.0)
    stochastic = keep_frac < 1.0
    gens = [torch.Generator(device=device)]                 # one for the run (or batch), as before
    gens[0].manual_seed(getattr(args, "seed", 0))
    draw_maps = {}                                           # per sampler: (subject of every line, kept positions)
    draw_cache = {}                                          # per sampler: [draws of the next steps, next one]
    eval_every = max(1, getattr(args, "eval_every", 10))
    blocks = {}                                              # per sampler: (offset, lines, kept) of every subject

    for smp in all_samplers:
        lg = pg[smp.IA]
        line_group[id(smp)] = lg
        counts = torch.bincount(lg, minlength=G).tolist()
        offs = [0]
        for c in counts[:-1]:
            offs.append(offs[-1] + c)
        if not single and not bool((lg[1:] >= lg[:-1]).all()):
            raise ValueError("align: the lines of a batch must be ordered by subject")
        blocks[id(smp)] = [(o, m, max(1, min(m, int(round(keep_frac * m))))) for o, m in zip(offs, counts)]
        line_sums[id(smp)] = (BlockSum(counts, device), BlockSum([k for _, _, k in blocks[id(smp)]], device))
        draw_maps[id(smp)] = (lg.double(), torch.cat([torch.arange(o, o + k, device=device)
                                                     for o, _, k in blocks[id(smp)]]))

    def n_keep(smp):
        if single:
            return max(1, min(smp.M, int(round(keep_frac * smp.M))))
        return sum(k for _, _, k in blocks[id(smp)])

    def draw_keep(smp):
        if single:
            return torch.randperm(smp.M, generator=gens[0], device=device)[:n_keep(smp)]
        # every subject's own lines in random order with one draw for the whole batch: sorting subject + U(0, 0.5)
        # keeps each subject's lines in its own block (the lines are ordered by subject), shuffled within it; the
        # first kept-count of each block are taken (a randperm per subject and step cost ~0.4 s per subject). The
        # 0.5 keeps rounding from carrying a key into the next block; float64 keys make ties rare and the stable sort
        # orders the remaining ones the same way every run (an unstable one made batched runs irreproducible). The
        # draws of DRAW_AHEAD steps are made at once (one sort per DRAW_AHEAD steps instead of one per step).
        cache = draw_cache.get(id(smp))
        if cache is None or cache[1] == len(cache[0]):
            lg_f, keep_pos = draw_maps[id(smp)]
            key = lg_f + 0.5 * torch.rand(DRAW_AHEAD, smp.M, generator=gens[0], device=device, dtype=torch.float64)
            cache = draw_cache[id(smp)] = [torch.argsort(key, dim=1, stable=True)[:, keep_pos], 0]
        cache[1] += 1
        return cache[0][cache[1] - 1]

    # learning rates as device tensors (updated in place each step; captured graphs read them)
    lr_t = torch.tensor(float(args.lr), **f32)
    lr_r = torch.tensor(float(args.rot_lr), **f32)
    param_groups = [{"params": [t], "lr": lr_t}] + ([{"params": [theta_deg], "lr": lr_r}] if rotate else [])
    opt = make_optimizer(param_groups, args, device)

    def lr_at(step, start, end):
        if args.lr_schedule == "constant" or start <= 0:
            return start
        return start + (end - start) * min(1.0, step / max(1, args.steps - 1))        # linear start -> end

    blur_sigma = getattr(args, "blur_sigma", 0.0)
    blur_end = min(getattr(args, "blur_sigma_end", 0.0), blur_sigma)          # the blur decays to this, then stays
    blur_const = blur_sigma > 0 and getattr(args, "blur_frac", 0.6) <= 0      # --blur-frac 0: no decay
    blur_steps = int(round(getattr(args, "blur_frac", 0.6) * args.steps)) if blur_sigma > 0 and not blur_const else 0
    blur_quantum = 0.2                                     # mm; the blur is only recomputed when sigma moves this much
    psf_s0 = getattr(args, "psf_scale_start", 1.0)
    psf_steps = int(round(getattr(args, "psf_anneal_frac", 0.6) * args.steps)) if psf_s0 > 1.0 else 0
    # late edge term (--edge-weight): its weight in the per-line similarity ramps linearly from 0 at --edge-start-frac
    # to --edge-weight over --edge-ramp-frac of the run (LineSampler.similarity_lines), then stays
    edge_w = getattr(args, "edge_weight", 0.0)
    edge_start = int(round(getattr(args, "edge_start_frac", 0.8) * args.steps))
    edge_ramp = int(round(getattr(args, "edge_ramp_frac", 0.1) * args.steps))
    edge_steps = edge_start + edge_ramp if edge_w > 0 else 0             # the edge weight is final from here on

    def edge_at(step):
        if edge_w <= 0 or step < edge_start:
            return 0.0
        return edge_w * min(1.0, (step - edge_start + 1) / edge_ramp) if edge_ramp > 0 else edge_w

    # from here on a subject's objective is the nominal one (its own mask annealing may end earlier, or not exist)
    ends = list(anneal_ends) if batched and anneal_ends is not None else [anneal_end] * G
    track_from = [max(e, blur_steps, psf_steps, coarse_steps, edge_steps) for e in ends]

    # device-side bookkeeping: best pose so far, per subject (no host sync per step)
    best_obj = torch.full((G,), -float("inf"), **f32)
    best_th, best_t = torch.zeros(N, 3, **f32), t.detach().clone()
    best_step = torch.zeros(G, **f32)
    step_buf, track_buf = torch.zeros((), **f32), torch.zeros(G, **f32)
    flags = {}                                             # cached (G,) 0/1 tensors for track_buf

    def flag(values):
        key = tuple(bool(v) for v in values)
        if key not in flags:
            flags[key] = torch.tensor([float(v) for v in key], **f32)
        return flags[key]

    sel_bufs = {id(smp): torch.arange(n_keep(smp), device=device) for smp in all_samplers} if stochastic else {}
    for smp in all_samplers:                             # the full-objective evaluations stay within a step's memory
        smp.eval_chunk = min(smp.eval_chunk, n_keep(smp)) if stochastic else smp.eval_chunk

    def update_best(o):
        better = (o > best_obj) & (track_buf > 0)                          # (G,)
        bp = better[pg][:, None]                                           # (N, 1)
        best_obj.copy_(torch.where(better, o, best_obj))
        best_th.copy_(torch.where(bp, pose_theta().detach(), best_th))
        best_t.copy_(torch.where(bp, t.detach(), best_t))
        best_step.copy_(torch.where(better, step_buf, best_step))

    params_all = [p for g in param_groups for p in g["params"]]
    n_skipped = torch.zeros(G, **f32)                    # steps skipped for a non-finite gradient, per subject
    # step cap (--max-step-mm / --max-step-deg): after each optimizer step, a plane whose translation moved more than
    # the cap (Euclidean, mm), or whose rotation vector changed more than the cap (deg; for --pose inplane its angle),
    # is moved back along its step to the cap. One large gradient (a degenerate line, a mask-annealing switch) can then
    # move a plane only a few mm per step instead of throwing it out of the image overlap, where its lines go flat and
    # nothing brings it back. Tensor ops on buffers allocated here, so it runs inside the captured CUDA graph.
    caps = []
    if getattr(args, "max_step_mm", 0.0) > 0:
        caps.append((t, torch.zeros_like(t), float(args.max_step_mm)))
    if rotate and getattr(args, "max_step_deg", 0.0) > 0:
        caps.append((theta_deg, torch.zeros_like(theta_deg), float(args.max_step_deg)))

    def cap_steps():
        with torch.no_grad():
            for p, prev, limit in caps:
                d = p - prev
                n = d.pow(2).sum(1, keepdim=True).sqrt()
                p.copy_(prev + d * (limit / n.clamp_min(1e-12)).clamp(max=1.0))

    # hard limits (--max-shift-mm / --max-rot-deg): a plane's translation (Euclidean, mm) and rotation angle (deg) from
    # its acquired pose never exceed them -- after each step a plane beyond is projected back onto the limit (same
    # direction), so no slice can leave the cluster whatever the gradients do. Graph-safe like the step cap.
    limits = []
    if getattr(args, "max_shift_mm", 0.0) > 0:
        limits.append((t, float(args.max_shift_mm)))
    if rotate and getattr(args, "max_rot_deg", 0.0) > 0:
        limits.append((theta_deg, float(args.max_rot_deg)))

    def apply_limits():
        with torch.no_grad():
            for p, limit in limits:
                n = p.pow(2).sum(1, keepdim=True).sqrt()
                p.mul_((limit / n.clamp_min(1e-12)).clamp(max=1.0))

    def body(smp):
        opt.zero_grad(set_to_none=False)
        obj = objective(pose_theta(), t, sel_bufs.get(id(smp)), smp)       # (G,)
        # loss scaling: with --sample-dtype float16 / bfloat16 the backward pass of the image sampling runs in half
        # precision, where each sample's share of the gradient (~1e-5: the objective averages thousands of line points)
        # underflows -- the gradient came out ~100x too small and pointing nowhere (cosine 0.01 with the float32
        # gradient on a DICOM subject), so the poses random-walked and the metrics fell. Scaled by GRAD_SCALE (a power
        # of two, exact) it matches float32 (cosine 0.997); the scale is divided out before the update.
        (-obj.sum() * GRAD_SCALE).backward()
        with torch.no_grad():
            # a step whose gradient is not finite (a half-precision overflow, or a degenerate line) is skipped -- per
            # subject: its gradient is zeroed (as AMP's GradScaler does) instead of turning every pose into NaN for the
            # rest of the run; counted in n_skipped (no host sync inside the CUDA graph)
            grads = [p.grad for p in params_all if p.grad is not None]
            if single:
                finite = (torch.stack([torch.isfinite(g).all() for g in grads]).all() & torch.isfinite(obj).all()).reshape(1)
                ok = finite.reshape(1, 1)
            else:
                bad = torch.zeros(N, dtype=torch.bool, device=device)
                for g in grads:
                    bad = bad | ~torch.isfinite(g).all(1)
                finite = (plane_sum(bad.float()) == 0) & torch.isfinite(obj)
                ok = finite[pg][:, None]
            for g in grads:
                g.copy_(torch.where(ok, torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0) / GRAD_SCALE,
                                    torch.zeros_like(g)))
            n_skipped.add_((~finite).float())
        update_best(torch.where(finite, obj.detach(), best_obj))   # pose *before* this update, as before
        if caps:
            with torch.no_grad():
                for p, prev, _ in caps:
                    prev.copy_(p)
        opt.step()
        if caps:
            cap_steps()
        if limits:
            apply_limits()

    use_graph = device.type == "cuda" and not getattr(args, "no_cuda_graph", False)
    graphs = {}

    def capture(smp):
        """Capture body(smp) as a CUDA graph. Warm-up iterations run for real, so all state is restored after."""
        params = [p for g in param_groups for p in g["params"]]
        saved_p = [p.detach().clone() for p in params]
        saved_state = {i: {k: v.clone() for k, v in opt.state[p].items()} for i, p in enumerate(params) if p in opt.state}
        saved_book = [x.clone() for x in (best_obj, best_th, best_t, best_step, n_skipped)]
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                body(smp)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            body(smp)
        with torch.no_grad():
            for p, v in zip(params, saved_p):
                p.copy_(v)
            for i, p in enumerate(params):
                for k, v in opt.state[p].items():
                    if i in saved_state:
                        v.copy_(saved_state[i][k])
                    else:
                        v.zero_()
            for x, v in zip((best_obj, best_th, best_t, best_step, n_skipped), saved_book):
                x.copy_(v)
        return graph

    def run_step(smp):
        nonlocal use_graph
        if use_graph:
            graph = graphs.get(id(smp))
            if graph is None:
                # The mask-annealing stages run in order and are never revisited: free the previous stage's graph and
                # its private memory pool (the sampling / gradient buffers of a step, GBs for many slices) before
                # capturing the next. Kept, they pile up over the stages until the GPU memory overflows, and on
                # Windows the driver then spills to system RAM, making every step seconds slow.
                graphs.clear()
                torch.cuda.empty_cache()
                try:
                    graph = graphs[id(smp)] = capture(smp)
                except Exception as e:  # noqa: BLE001
                    print(f"CUDA graph capture failed ({type(e).__name__}: {e}); continuing without graphs")
                    use_graph = False
            if use_graph:
                graph.replay()
                return
        body(smp)

    every = getattr(args, "history_every", 0.0)
    snap_steps = sorted({int(round(k * every * args.steps)) for k in range(int(round(1 / every)) + 1)}) if every > 0 else []
    snapshots = []
    edge_cur = None                                      # the edge weight the samplers hold (set on the first step)
    with torch.enable_grad(), step_progress_bar(args) as bar:
        for step in range(args.steps):
            if step in snap_steps:
                snapshots.append((step, pose_theta().detach().clone(), t.detach().clone()))
            smp = sampler_at(step)
            sigma = (blur_sigma if blur_const
                     else blur_end + (blur_sigma - blur_end) * (1.0 - step / blur_steps) if step < blur_steps
                     else blur_end)
            smp.set_blur(math.ceil(sigma / blur_quantum) * blur_quantum if sigma > 0 else 0.0)
            smp.set_psf_scale(psf_s0 + (1.0 - psf_s0) * step / psf_steps if step < psf_steps else 1.0)
            if edge_at(step) != edge_cur:            # on every sampler: the best pose is scored on final_sampler
                edge_cur = edge_at(step)
                for s in all_samplers:
                    s.set_edge_weight(edge_cur)
            lr_t.fill_(lr_at(step, args.lr, LR_END_FRACTION * args.lr if args.lr_end is None else args.lr_end))
            lr_r.fill_(lr_at(step, args.rot_lr,
                             LR_END_FRACTION * args.rot_lr if args.rot_lr_end is None else args.rot_lr_end))
            nominal = [step >= tf for tf in track_from]
            step_buf.fill_(float(step))
            if stochastic:
                sel_bufs[id(smp)].copy_(draw_keep(smp))
                # the dropped-line objective is not comparable between steps: track the best periodically on the
                # full objective (eagerly, before the update), and never inside the graph
                if any(nominal) and (step % eval_every == 0 or step == args.steps - 1):
                    track_buf.copy_(flag(nominal))
                    with torch.no_grad():
                        update_best(objective(pose_theta(), t))
                track_buf.fill_(0.0)
            else:
                track_buf.copy_(flag(nominal))
            run_step(smp)
            if bar is not None:
                bar.update()
    if args.steps in snap_steps:
        snapshots.append((args.steps, pose_theta().detach().clone(), t.detach().clone()))
    with torch.no_grad():
        # the final parameters are also a candidate (scored with the blur / PSF of the last step, so that a constant
        # --blur-frac 0 blur is comparable to the tracked steps); they are the only one when an annealing schedule
        # (e.g. --psf-anneal-frac > 1) outlasts the run, as nothing was tracked then
        final = [(stochastic and tf <= args.steps) or tf >= args.steps for tf in track_from]
        if any(final):
            step_buf.fill_(float(args.steps))
            track_buf.copy_(flag(final))
            update_best(objective(pose_theta(), t))
    for smp in all_samplers:
        smp.set_blur(0.0)
        smp.set_psf_scale(1.0)
        smp.set_edge_weight(0.0)                 # the reported similarity is --metric alone (the edge term: 'edge')
    skipped = [int(x) for x in n_skipped.tolist()]
    steps_found = [int(x) for x in best_step.tolist()]
    if not batched:
        if skipped[0]:
            print(f"\n[align] {skipped[0]} of {args.steps} steps skipped (non-finite gradient)", flush=True)
        return dict(theta_deg=best_th.clone(), t=best_t.clone(), best_step=steps_found[0], snapshots=snapshots)
    return dict(theta_deg=best_th.clone(), t=best_t.clone(), best_step=steps_found, snapshots=snapshots,
                n_skipped=skipped)


# --------------------------------------------------------------------------------------
# Per-subject pipeline
# --------------------------------------------------------------------------------------

def relative_translation(t: torch.Tensor, lax_rows: list[int]) -> np.ndarray:
    """Translations expressed relative to the mean LAX pose (the convention used for all outputs)."""
    t = t.detach().cpu().numpy().astype(np.float64)
    return t - t[lax_rows].mean(0) if lax_rows else t


def base_affine(data: dict, plane: Plane) -> np.ndarray:
    """The acquired 4x4 affine of one plane (its own slice file's affine)."""
    return plane.affine.copy()


AFFINE_COLUMNS = [f"a{r}{c}" for r in range(3) for c in range(4)]
HISTORY_COLUMNS = (["subject_id", "step", "fraction", "is_best", "image", "slice"] + AFFINE_COLUMNS
                   + ["dx_mm", "dy_mm", "dz_mm", "rot_x_deg", "rot_y_deg", "rot_z_deg", "rot_inplane_deg", "rot_tilt_deg",
                      "ncc", "lncc", "nmi", "seg_dice", "hd_mm", "lax_dice", "apex_mm"])
HISTORY_STEP_COLUMNS = ["subject_id", "step", "fraction", "is_best", "objective", "ncc_mean", "lncc_mean", "nmi_mean",
                        "seg_dice_mean",
                        "hd_mean_mm", "hd_max_mm", "lax_dice_mean", "apex_mean_mm",
                        "sax_shift_rms_mm", "sax_shift_jitter_mm", "sax_dz_rms_mm", "sax_tilt_rms_deg"]
SAX_LINE_COLUMNS = ["subject_id", "step", "fraction", "is_best", "image_a", "slice_a", "image_b", "slice_b", "kind",
                    "crossing", "n_points", "supervised", "axis_dist_mm", "ncc", "dissim", "gap_min_mm", "rel_tilt_deg",
                    "crossing_in_heart", "x0_x", "x0_y", "x0_z", "d_x", "d_y", "d_z",
                    "t_used_lo", "t_used_hi", "t_img_lo", "t_img_hi"]
LONG_AXIS_COLUMNS = ["subject_id", "step", "fraction", "is_best", "p_x", "p_y", "p_z", "d_x", "d_y", "d_z", "radius_mm"]


def build_history(sid, data, planes, n_pairs, sampler, snapshots, best_step, lax_rows, args, labels, labels_sax, has_seg,
                  line_metrics: list | None = None) -> dict:
    """Affines, poses and metrics of every pose snapshot. Snapshot step 'best' is the pose that is saved/reported,
    re-labelled with the step it was found at; step 0 is the acquired geometry (identity pose). With --sax-sax-weight
    also the neighbouring-SAX-slice crossing lines of every snapshot ('sax_lines', see SaxNeighbourLines.segment_rows).
    `line_metrics`: the per-line metrics of every snapshot if already evaluated (align_batch evaluates a whole batch's
    snapshots at once), else sampler.evaluate."""
    nan = float("nan")
    plane_rows, step_rows, sax_line_rows, axis_rows = [], [], [], []
    if sampler.sax_pairs is not None:                 # every snapshot's neighbouring-slice rows at once
        shifts = [tt.detach().cpu().numpy().astype(np.float64)[lax_rows].mean(0) if lax_rows else np.zeros(3)
                  for _, _, tt in snapshots]
        seg_rows = sampler.sax_pairs.segment_rows_many([(th, tt) for _, th, tt in snapshots], shifts)
    for k_snap, (step, th, tt) in enumerate(snapshots):
        is_best = step == "best"
        step_no = best_step if is_best else step
        if sampler.sax_pairs is not None:
            shift = shifts[k_snap]
            for r in seg_rows[k_snap]:
                sax_line_rows.append(dict(subject_id=sid, step=step_no, fraction=step_no / args.steps, is_best=int(is_best),
                                          **r))
            sp = sampler.sax_pairs
            L0 = sp.L0.cpu().numpy().astype(np.float64) - shift      # same frame as the sax_lines rows
            ell = sp.ell.cpu().numpy().astype(np.float64)
            axis_rows.append(dict(subject_id=sid, step=step_no, fraction=step_no / args.steps, is_best=int(is_best),
                                  p_x=float(L0[0]), p_y=float(L0[1]), p_z=float(L0[2]),
                                  d_x=float(ell[0]), d_y=float(ell[1]), d_z=float(ell[2]), radius_mm=sp.radius))
        lm = (line_metrics[len(step_rows)] if line_metrics is not None
              else sampler.evaluate(th, tt, args, labels, labels_sax, has_seg))
        with torch.no_grad():
            # the optimized similarity on the mask points
            sim = (sampler.similarity_lines(th, tt, args, W=sampler.W)
                   if sampler.any_cross and cross_mod_metric(args) != args.metric      # (two metrics: see there)
                   else lm["nmi"] if args.metric == "nmi" else lm["ncc"] if args.metric == "ncc"
                   else lm["lncc"] if args.metric == "lncc" else sampler.similarity_lines(th, tt, args, W=sampler.W))
            objective = float(similarity_objective(sampler, th, tt, args, sim=sim))
        per = sampler.plane_values(lm)
        theta = th.detach().cpu().numpy().astype(np.float64)
        t_rel = relative_translation(tt, lax_rows)
        Rs = rotation_matrices(theta)
        sax_vec, sax_dz, sax_tilt = [], [], []
        for i, p in enumerate(planes):
            A = corrected_affine(base_affine(data, p), p, Rs[i], t_rel[i])
            dx, dy, dz = float(t_rel[i] @ p.u_hat), float(t_rel[i] @ p.v_hat), float(t_rel[i] @ p.normal)
            inplane = float(theta[i] @ p.normal)
            tilt = float(np.linalg.norm(theta[i] - inplane * p.normal))
            if p.view == "SAX" and n_pairs[i] > 0:
                sax_vec.append((dx, dy)), sax_dz.append(dz), sax_tilt.append(tilt)
            plane_rows.append(dict(subject_id=sid, step=step_no, fraction=step_no / args.steps, is_best=int(is_best),
                                   image=p.image, slice=p.slice_idx,
                                   **{c: float(A[int(c[1]), int(c[2])]) for c in AFFINE_COLUMNS},
                                   dx_mm=dx, dy_mm=dy, dz_mm=dz, rot_x_deg=float(theta[i, 0]), rot_y_deg=float(theta[i, 1]),
                                   rot_z_deg=float(theta[i, 2]), rot_inplane_deg=inplane, rot_tilt_deg=tilt,
                                   ncc=per["ncc"][i] if n_pairs[i] else nan, lncc=per["lncc"][i] if n_pairs[i] else nan,
                                   nmi=per["nmi"][i] if n_pairs[i] else nan,
                                   seg_dice=per["dice"][i] if n_pairs[i] else nan, hd_mm=per["hd"][i] if n_pairs[i] else nan,
                                   lax_dice=per["lax_dice"][i] if n_pairs[i] else nan,
                                   apex_mm=per["apex"][i] if n_pairs[i] else nan))
        vec = np.array(sax_vec).reshape(-1, 2)
        ok = lambda x: np.asarray(x, dtype=float)[np.isfinite(np.asarray(x, dtype=float))]  # noqa: E731
        hd_lines = ok(lm["hd"].cpu().numpy())
        lax_dice_lines = ok(lm["dice"][sampler.lax_lax].cpu().numpy())
        apex_lines = ok(lm["apex"].cpu().numpy())
        step_rows.append(dict(
            subject_id=sid, step=step_no, fraction=step_no / args.steps, is_best=int(is_best), objective=objective,
            ncc_mean=float(np.nanmean(per["ncc"])), lncc_mean=float(np.nanmean(per["lncc"])),
            nmi_mean=float(np.nanmean(per["nmi"])),
            seg_dice_mean=float(np.nanmean(per["dice"])) if np.isfinite(per["dice"]).any() else nan,
            hd_mean_mm=float(hd_lines.mean()) if hd_lines.size else nan, hd_max_mm=float(hd_lines.max()) if hd_lines.size else nan,
            lax_dice_mean=float(lax_dice_lines.mean()) if lax_dice_lines.size else nan,
            apex_mean_mm=float(apex_lines.mean()) if apex_lines.size else nan,
            sax_shift_rms_mm=float(np.sqrt((vec ** 2).sum(1).mean())) if len(vec) else nan,
            sax_shift_jitter_mm=float(np.sqrt(((vec - vec.mean(0)) ** 2).sum(1).mean())) if len(vec) else nan,
            sax_dz_rms_mm=float(np.sqrt(np.mean(np.square(sax_dz)))) if sax_dz else nan,
            sax_tilt_rms_deg=float(np.sqrt(np.mean(np.square(sax_tilt)))) if sax_tilt else nan))
    return dict(planes=plane_rows, steps=step_rows, sax_lines=sax_line_rows, long_axis=axis_rows)


def annealed_stage_lines(data, planes, lax_planes, args, device) -> tuple[list, int] | None:
    """The intersection lines of every mask-annealing stage: ([(start_step, lines), ...], anneal_end), or None when
    mask annealing is off. The mask dilation goes linearly from --mask-dilation-start-mm to --mask-dilation-mm over the
    first --mask-anneal-frac of the run, in --anneal-stages discrete steps; the lines are rebuilt at each step (a stage
    without any line is left out; the stage before it then lasts longer)."""
    dil0 = getattr(args, "mask_dilation_start_mm", None)
    n_stages = max(1, getattr(args, "anneal_stages", 6))
    anneal_end = int(round(getattr(args, "mask_anneal_frac", 0.8) * args.steps)) if dil0 is not None else 0
    if anneal_end <= 0:
        return None
    starts = sorted({int(round(k / n_stages * anneal_end)) for k in range(n_stages)})
    dists = {v: torch.from_numpy(d["dist"]).to(device) for v, d in data["lax"].items() if d.get("dist") is not None}
    if not dists:                                        # no masked image (segmentation or center disc): nothing to anneal
        return None
    nominal_masks = {p.key: p.mask for p in lax_planes}
    out = []
    try:
        for k, start in enumerate(starts):
            dil = dil0 + (args.mask_dilation_mm - dil0) * k / n_stages
            for p in lax_planes:
                if p.image in dists:                             # (images without a mask stay unmasked)
                    p.mask = (dists[p.image][:, :, p.slice_idx] <= dil).float()[None, None]
            lines = build_lines(planes, args, device)
            if lines:
                out.append((start, lines))
    finally:
        for p in lax_planes:
            p.mask = nominal_masks[p.key]
    return out, anneal_end


def annealed_stages(data, planes, sax_planes, lax_planes, final_sampler, args, device, stage_lines=None) -> list | None:
    """(start_step, LineSampler) stages for mask-dilation annealing, ending with the nominal sampler (see
    annealed_stage_lines; `stage_lines`: its result if already computed). Returns None when mask annealing is off."""
    sl = annealed_stage_lines(data, planes, lax_planes, args, device) if stage_lines is None else stage_lines
    if sl is None:
        return None
    stages = []
    for start, lines in sl[0]:
        smp = new_sampler(planes, lines, args, device, images=final_sampler.img)
        attach_sax_pairs(smp, planes, sax_planes, lax_planes, args, device)
        stages.append((start, smp))
    return stages + [(sl[1], final_sampler)]


def lines_to(lines: list, device) -> list:
    """(index a, index b, Line) triples with the Line tensors moved to `device`."""
    def move(x):
        return None if x is None else x.to(device)
    return [(ia, ib, dataclasses.replace(line, t=move(line.t), w=move(line.w), w_loss=move(line.w_loss)))
            for ia, ib, line in lines]


def prepare_lines(data: dict, args, on_lines=None) -> dict:
    """The intersection lines of a prepared subject (prepare_subject), built on the CPU: the nominal ones (build_lines)
    and those of every mask-annealing stage (annealed_stage_lines). They depend only on the acquired geometry and the
    masks, and come out identical to those built on the GPU (checked on 30 subjects) -- but ~3x faster, as they are
    many tiny tensor operations; run() builds them in its worker processes (load_job), next to prepare_subject.
    `on_lines(planes, lines)`: called with the nominal lines (as by prepare_alignment), e.g. to write them."""
    cpu = torch.device("cpu")
    sax_planes, lax_planes = build_planes(data, args, cpu, with_seg=on_lines is not None)  # (the overlay shows the segs)
    planes = sax_planes + lax_planes
    lines = build_lines(planes, args, cpu)
    if on_lines is not None and lines:
        on_lines(planes, lines)
    return dict(lines=lines, stage_lines=annealed_stage_lines(data, planes, lax_planes, args, cpu) if lines else None)


def prepare_alignment(subject: Subject, args, device, on_lines=None, data: dict | None = None,
                      pre: dict | None = None) -> dict:
    """Everything one subject's alignment needs (process_subject's first half): its planes, intersection lines and
    pose model, the sampler of its nominal geometry and the lines of its mask-annealing stages. `data`: the subject's
    prepare_subject() if already computed (run() does it in its worker processes, see load_job); `pre`: its
    prepare_lines() if already computed (idem)."""
    data = prepare_subject(subject, args) if data is None else data
    log = getattr(args, "logging", True)
    sax_planes, lax_planes = build_planes(data, args, device, with_seg=log)
    planes = sax_planes + lax_planes
    lines = build_lines(planes, args, device) if pre is None else lines_to(pre["lines"], device)
    n_pairs = [0] * len(planes)
    for ia, ib, _ in lines:
        n_pairs[ia] += 1
        n_pairs[ib] += 1
    if on_lines is not None:
        on_lines(planes, lines)
    per_image = {}
    for p in lax_planes:
        per_image[p.image] = per_image.get(p.image, 0) + 1
    model = (pose_model(len(sax_planes), len(per_image), args.dof, sum(n > 1 for n in per_image.values()))
             if lines else "none")
    ctx = dict(data=data, sid=data["subject_id"], log=log, sax_planes=sax_planes, lax_planes=lax_planes, planes=planes,
               lines=lines, n_pairs=n_pairs, model=model, sampler=None, stage_lines=None)
    if lines:
        sampler = new_sampler(planes, lines, args, device)
        attach_sax_pairs(sampler, planes, sax_planes, lax_planes, args, device)
        ctx["sampler"] = sampler
        if pre is None:
            ctx["stage_lines"] = annealed_stage_lines(data, planes, lax_planes, args, device)
        elif pre["stage_lines"] is not None:
            ctx["stage_lines"] = ([(start, lines_to(ls, device)) for start, ls in pre["stage_lines"][0]],
                                  pre["stage_lines"][1])
    return ctx


def align_one(ctx: dict, args, device) -> dict:
    """align() of one prepared subject (prepare_alignment)."""
    stages = annealed_stages(ctx["data"], ctx["planes"], ctx["sax_planes"], ctx["lax_planes"], ctx["sampler"], args,
                             device, stage_lines=ctx["stage_lines"])
    return align(ctx["sampler"], args, stages, pose_model=ctx["model"])


def label_settings(args, planes) -> tuple[list[int], list[int], bool]:
    """(labels compared on LAX-LAX lines, on SAX-LAX lines, whether any plane has a segmentation) of the Dice /
    Hausdorff metrics (loader options; no labels = no label metrics)."""
    labels = [int(x) for x in getattr(args, "seg_labels", "").split(",") if x.strip()]
    labels_sax = [int(x) for x in getattr(args, "sax_seg_labels", "").split(",") if x.strip()]
    return labels, labels_sax, any(p.seg is not None for p in planes)


def batchable(args) -> bool:
    """Whether subjects can share one optimization (align_batch): not with the parallel-pair terms (--sax-sax-weight,
    --sax-gap-reg, --coplanar-weight), which are normalized over a whole sampler; those runs align one at a time."""
    return not ((getattr(args, "sax_sax_weight", 0.0) > 0 and args.dof == 6) or getattr(args, "sax_gap_reg", 0.0) > 0
                or getattr(args, "coplanar_weight", 0.0) > 0)


def align_batch(ctxs: list[dict], args, device) -> list[dict]:
    """Align several prepared subjects (prepare_alignment) in one optimization: their planes and lines are put one
    subject after the other into shared samplers (one per mask-annealing stage), so every step samples all of them in
    one go, and align() keeps every term per subject (see there). Returns one align() result per subject, as a solo
    run would. The stages start where any subject's start; a subject uses its own latest stage started by then (its
    nominal lines, if it has no mask annealing), exactly as in a solo run."""
    planes, groups, offsets = [], [], []
    for g, c in enumerate(ctxs):
        offsets.append(len(planes))
        planes += c["planes"]
        groups += [g] * len(c["planes"])

    def merged(lines_of):
        out = []
        for g, c in enumerate(ctxs):
            off = offsets[g]
            out += [(ia + off, ib + off, line) for ia, ib, line in lines_of(c)]
        return out

    final = new_sampler(planes, merged(lambda c: c["lines"]), args, device)
    ends = [c["stage_lines"][1] if c["stage_lines"] else 0 for c in ctxs]
    starts = sorted({s for c in ctxs if c["stage_lines"] for s, _ in c["stage_lines"][0]})

    def lines_at(c, s):
        if not c["stage_lines"]:
            return c["lines"]
        own = c["stage_lines"][0]
        active = own[0][1]
        for start, lines in own:
            if s >= start:
                active = lines
        return active

    stages = [(s, new_sampler(planes, merged(lambda c, s=s: lines_at(c, s)), args, device, images=final.img))
              for s in starts]
    if stages:
        stages.append((max(ends), final))
    res = align(final, args, stages or None, groups=groups, models=[c["model"] for c in ctxs], anneal_ends=ends)
    # with logging, the reported metrics of every subject (acquired pose, saved pose, every history snapshot) are
    # evaluated for the whole batch at once -- they are per line, so the shared sampler gives each subject's own
    # values -- instead of one evaluation per subject and snapshot; on first use in finish_alignment (batch_metrics)
    shared = dict(final=final, planes=planes, res=res, metrics=None) if getattr(args, "logging", True) else None
    out, m0 = [], 0
    for g, c in enumerate(ctxs):
        lo, hi = offsets[g], offsets[g] + len(c["planes"])
        m1 = m0 + len(c["lines"])
        if res["n_skipped"][g]:
            print(f"\n[align] {c['sid']}: {res['n_skipped'][g]} of {args.steps} steps skipped (non-finite gradient)",
                  flush=True)
        r = dict(theta_deg=res["theta_deg"][lo:hi].clone(), t=res["t"][lo:hi].clone(), best_step=res["best_step"][g],
                 snapshots=[(step, th[lo:hi], tt[lo:hi]) for step, th, tt in res["snapshots"]])
        if shared is not None:                       # this subject's lines: m0 .. m1 of the shared sampler
            r["batch_metrics"] = (shared, m0, m1)
        out.append(r)
        m0 = m1
    return out


def batch_metrics(result: dict, args) -> dict | None:
    """The per-line metrics (acquired pose, saved pose, every snapshot) of a subject aligned in a batch (align_batch):
    its share of the whole batch's evaluation, which is done on the first call for any of its subjects. None for a
    subject aligned alone."""
    bm = result.get("batch_metrics")
    if bm is None:
        return None
    shared, m0, m1 = bm
    if shared["metrics"] is None:
        final, planes, res = shared["final"], shared["planes"], shared["res"]
        labels, labels_sax, has_seg = label_settings(args, planes)
        zeros = torch.zeros(len(planes), 3, device=final.device)
        poses = [(zeros, zeros), (res["theta_deg"], res["t"])] + [(th, tt) for _, th, tt in res["snapshots"]]
        shared["metrics"] = [final.evaluate(th, tt, args, labels, labels_sax, has_seg) for th, tt in poses]
    mine = [{k: v[m0:m1] for k, v in lm.items()} for lm in shared["metrics"]]
    return dict(zero=mine[0], final=mine[1], snapshots=mine[2:])


def finish_alignment(ctx: dict, result: dict | None, args, device) -> tuple:
    """process_subject's second half: metrics, history and rows of a prepared subject (prepare_alignment) aligned to
    `result` (align(); None when it has no lines) -> (summary row, per-plane rows, per-line rows, poses, history,
    per-combination rows (region_rows))."""
    data, sid, log = ctx["data"], ctx["sid"], ctx["log"]
    planes, lines, n_pairs, model = ctx["planes"], ctx["lines"], ctx["n_pairs"], ctx["model"]
    sax_planes, lax_planes = ctx["sax_planes"], ctx["lax_planes"]
    nan = float("nan")
    N = len(planes)
    if not log:                                      # alignment only
        theta, t_rel = np.zeros((N, 3)), np.zeros((N, 3))
        if lines:
            theta = result["theta_deg"].cpu().numpy().astype(np.float64)
            t_rel = relative_translation(result["t"], lax_reference_rows(planes, n_pairs))
        Rs = rotation_matrices(theta)
        details = [dict(subject_id=sid, image=p.image, view=p.view, modality=p.modality, slice=p.slice_idx,
                        n_pairs=n_pairs[i]) for i, p in enumerate(planes)]
        return None, details, [], [(p, Rs[i], t_rel[i]) for i, p in enumerate(planes)], None, []

    metric_names = ("ncc", "lncc", "nmi", "dice", "hd", "lax_dice", "apex", "apex_dice")     # (LineSampler.plane_values)
    plane_m = {f"{k}_{s}": [nan] * N for k in metric_names for s in ("zero", "final")}
    pairs = []
    sax_qc = dict(sax_cross_pairs=nan, sax_gap_min_mm=nan, sax_rel_tilt_max_deg=nan, sax_cross_dissim_max=nan)
    cop_qc = dict(coplanar_pairs=0, coplanar_sim_zero=nan, coplanar_sim=nan)
    history = dict(planes=[], steps=[])
    regions = []
    n_points = [0.0] * N
    t_rel = np.zeros((N, 3))
    theta = np.zeros((N, 3))
    if lines:
        sampler = ctx["sampler"]
        n_points = [float(sampler.npts[idx].sum()) for idx in sampler.plane_lines]
        labels, labels_sax, has_seg = label_settings(args, planes)
        zeros = torch.zeros(N, 3, device=device)
        lmet = batch_metrics(result, args)           # evaluated for the whole batch (align_batch), else here
        line_zero = lmet["zero"] if lmet else sampler.evaluate(zeros, zeros, args, labels, labels_sax, has_seg)
        line_final = (lmet["final"] if lmet
                      else sampler.evaluate(result["theta_deg"], result["t"], args, labels, labels_sax, has_seg))
        for stage, lm in (("zero", line_zero), ("final", line_final)):
            pv = sampler.plane_values(lm)
            for k in metric_names:
                plane_m[f"{k}_{stage}"] = pv[k]
        lz = {k: v.cpu().numpy() for k, v in line_zero.items()}
        lf = {k: v.cpu().numpy() for k, v in line_final.items()}
        regions = region_rows(sid, planes, lines, lz, lf)
        for m, (ia, ib, line) in enumerate(lines):
            pa, pb = planes[ia], planes[ib]
            pairs.append(dict(subject_id=sid, image_a=pa.image, slice_a=pa.slice_idx, image_b=pb.image, slice_b=pb.slice_idx,
                              view_a=pa.view, view_b=pb.view,
                              kind="LAX-LAX" if pa.view != "SAX" and pb.view != "SAX" else "SAX-LAX", n_points=int(line.n_points),
                              ncc_zero=float(lz["ncc"][m]), ncc=float(lf["ncc"][m]), lncc_zero=float(lz["lncc"][m]),
                              lncc=float(lf["lncc"][m]), nmi_zero=float(lz["nmi"][m]), nmi=float(lf["nmi"][m]),
                              edge_zero=float(lz["edge"][m]), edge=float(lf["edge"][m]),
                              seg_dice_zero=float(lz["dice"][m]), seg_dice=float(lf["dice"][m]),
                              hd_zero_mm=float(lz["hd"][m]), hd_mm=float(lf["hd"][m]),
                              apex_zero_mm=float(lz["apex"][m]), apex_mm=float(lf["apex"][m]),
                              apex_endo_zero_mm=float(lz["apex_endo"][m]), apex_endo_mm=float(lf["apex_endo"][m]),
                              apex_dice_zero=float(lz["apex_dice"][m]), apex_dice=float(lf["apex_dice"][m]),
                              axis_dist_mm=nan, gap_min_mm=nan, rel_tilt_deg=nan))
        if sampler.sax_pairs is not None:                  # neighbouring SAX slices: crossing line, gap, relative tilt
            sp = sampler.sax_pairs
            with torch.no_grad():
                cz, nz, _ = sp.scores(zeros, zeros)
                cf, nf, df = sp.scores(result["theta_deg"], result["t"])
                gp = sp.gaps(result["theta_deg"], result["t"])
            cz, nz, cf, nf, df = (x.cpu().numpy() for x in (cz, nz, cf, nf, df))
            gap_min = gp["gap"].flatten(1).min(1).values.cpu().numpy()
            tilt = gp["rel_tilt"].cpu().numpy()
            for k, (ia, ib) in enumerate(sp.pairs):
                pairs.append(dict(subject_id=sid, image_a=sp.images_a[k], slice_a=planes[ia].slice_idx, image_b=sp.images_b[k],
                                  slice_b=planes[ib].slice_idx, view_a=planes[ia].view, view_b=planes[ib].view,
                                  kind=sp.kinds[k], n_points=round(float(nf[k]), 2), ncc_zero=float(cz[k]) if nz[k] > 1e-3 else nan,
                                  ncc=float(cf[k]) if nf[k] > 1e-3 else nan, nmi_zero=nan, nmi=nan, seg_dice_zero=nan, seg_dice=nan,
                                  hd_zero_mm=nan, hd_mm=nan, axis_dist_mm=float(df[k]),
                                  gap_min_mm=float(gap_min[k]), rel_tilt_deg=float(tilt[k]),
                                  **dict.fromkeys(("lncc_zero", "lncc", "edge_zero", "edge", "apex_zero_mm", "apex_mm",
                                                   "apex_endo_zero_mm", "apex_endo_mm", "apex_dice_zero", "apex_dice"), nan)))
            crossing = gap_min < 0
            dissim = 1.0 - cf[crossing & (nf > 1e-3) & ~sp.cross.cpu().numpy()]     # NCC: within a modality
            sax_qc = dict(sax_cross_pairs=int(crossing.sum()), sax_gap_min_mm=float(gap_min.min()) if gap_min.size else nan,
                          sax_rel_tilt_max_deg=float(tilt.max()) if tilt.size else nan,
                          sax_cross_dissim_max=float(dissim.max()) if dissim.size else nan)
        cp = getattr(sampler, "coplanar", None)
        if cp is not None and cp.P:                         # near-coplanar pairs (--coplanar-weight): mean score
            with torch.no_grad():
                cop_qc = dict(coplanar_pairs=cp.P, coplanar_sim_zero=float(cp.score(zeros, zeros)),
                              coplanar_sim=float(cp.score(result["theta_deg"], result["t"])))
        theta = result["theta_deg"].cpu().numpy().astype(np.float64)
        lax_rows = lax_reference_rows(planes, n_pairs)
        t_rel = relative_translation(result["t"], lax_rows)
        if result["snapshots"]:
            snaps = result["snapshots"] + [("best", result["theta_deg"], result["t"])]
            history = build_history(sid, data, planes, n_pairs, sampler, snaps, result["best_step"], lax_rows,
                                    args, labels, labels_sax, has_seg,
                                    line_metrics=lmet["snapshots"] + [lmet["final"]] if lmet else None)
    Rs = rotation_matrices(theta)
    poses = [(p, Rs[i], t_rel[i]) for i, p in enumerate(planes)]

    details = []
    for i, p in enumerate(planes):
        row = dict(subject_id=sid, image=p.image, view=p.view, modality=p.modality, slice=p.slice_idx,
                   sax_region=p.region, n_pairs=n_pairs[i], n_points=int(n_points[i]),
                   dx_mm=nan, dy_mm=nan, dz_mm=nan, shift_inplane_mm=nan, shift_mm=nan, rot_inplane_deg=nan,
                   rot_tilt_deg=nan, ncc_zero=nan, ncc=nan, lncc_zero=nan, lncc=nan, nmi_zero=nan, nmi=nan,
                   seg_dice_zero=nan, seg_dice=nan, hd_zero_mm=nan, hd_mm=nan, lax_dice_zero=nan, lax_dice=nan,
                   apex_zero_mm=nan, apex_mm=nan, apex_dice_zero=nan, apex_dice=nan)
        if n_pairs[i] > 0:
            tv, th = t_rel[i], theta[i]
            dx, dy, dz = float(tv @ p.u_hat), float(tv @ p.v_hat), float(tv @ p.normal)
            inplane = float(th @ p.normal)
            row.update(dx_mm=dx, dy_mm=dy, dz_mm=dz, shift_inplane_mm=math.hypot(dx, dy), shift_mm=float(np.linalg.norm(tv)),
                       rot_inplane_deg=inplane, rot_tilt_deg=float(np.linalg.norm(th - inplane * p.normal)),
                       ncc_zero=plane_m["ncc_zero"][i], ncc=plane_m["ncc_final"][i],
                       lncc_zero=plane_m["lncc_zero"][i], lncc=plane_m["lncc_final"][i],
                       nmi_zero=plane_m["nmi_zero"][i], nmi=plane_m["nmi_final"][i],
                       seg_dice_zero=plane_m["dice_zero"][i], seg_dice=plane_m["dice_final"][i],
                       hd_zero_mm=plane_m["hd_zero"][i], hd_mm=plane_m["hd_final"][i],
                       lax_dice_zero=plane_m["lax_dice_zero"][i], lax_dice=plane_m["lax_dice_final"][i],
                       apex_zero_mm=plane_m["apex_zero"][i], apex_mm=plane_m["apex_final"][i],
                       apex_dice_zero=plane_m["apex_dice_zero"][i], apex_dice=plane_m["apex_dice_final"][i])
        details.append(row)

    def arr(rows, fn):
        return np.array([fn(d) for d in rows], dtype=float)

    def stat(fn, x):
        x = x[np.isfinite(x)] if x.size else x
        return float(fn(x)) if x.size else nan

    def rms(x):
        return np.sqrt(np.mean(x ** 2))

    sax_rows = [d for d in details if d["view"] == "SAX" and d["n_pairs"] > 0]
    lax_rows = [d for d in details if d["view"] != "SAX" and d["n_pairs"] > 0]
    ok_rows = [d for d in details if d["n_pairs"] > 0]
    vec = np.array([[d["dx_mm"], d["dy_mm"]] for d in sax_rows]).reshape(-1, 2)
    mag = np.hypot(vec[:, 0], vec[:, 1])
    jit = np.hypot(*(vec - vec.mean(0)).T) if len(vec) else np.array([])
    nz, nf_ = arr(ok_rows, lambda d: d["ncc_zero"]), arr(ok_rows, lambda d: d["ncc"])
    lz_, lf_ = arr(ok_rows, lambda d: d["lncc_zero"]), arr(ok_rows, lambda d: d["lncc"])
    mz, mf = arr(ok_rows, lambda d: d["nmi_zero"]), arr(ok_rows, lambda d: d["nmi"])
    line_rows = [d for d in pairs if d["kind"] not in NEIGHBOUR_KINDS]       # the SAX-LAX / LAX-LAX lines
    laxlax = [d for d in line_rows if d["kind"] == "LAX-LAX"]

    summary = dict(
        subject_id=sid, status="ok" if lines else "not_optimized", pose_model=model,
        n_lax_views=len(lax_planes), sax_n_slices=len(sax_planes), sax_n_eval=len(sax_rows),
        sax_shift_mean_mm=stat(np.mean, mag), sax_shift_rms_mm=stat(rms, mag), sax_shift_max_mm=stat(np.max, mag),
        sax_shift_jitter_mm=stat(rms, jit),
        sax_dz_rms_mm=stat(rms, arr(sax_rows, lambda d: d["dz_mm"])),
        sax_rot_inplane_rms_deg=stat(rms, arr(sax_rows, lambda d: d["rot_inplane_deg"])),
        sax_tilt_rms_deg=stat(rms, arr(sax_rows, lambda d: d["rot_tilt_deg"])),
        lax_shift_mean_mm=stat(np.mean, arr(lax_rows, lambda d: d["shift_inplane_mm"])),
        lax_shift_max_mm=stat(np.max, arr(lax_rows, lambda d: d["shift_inplane_mm"])),
        ncc_zero_mean=stat(np.mean, nz), ncc_mean=stat(np.mean, nf_), ncc_gain_mean=stat(np.mean, nf_ - nz),
        lncc_zero_mean=stat(np.mean, lz_), lncc_mean=stat(np.mean, lf_), lncc_gain_mean=stat(np.mean, lf_ - lz_),
        nmi_zero_mean=stat(np.mean, mz), nmi_mean=stat(np.mean, mf), nmi_gain_mean=stat(np.mean, mf - mz),
        edge_zero_mean=stat(np.mean, arr(line_rows, lambda d: d["edge_zero"])),          # (mean over the lines)
        edge_mean=stat(np.mean, arr(line_rows, lambda d: d["edge"])),
        seg_dice_zero_mean=stat(np.mean, arr(ok_rows, lambda d: d["seg_dice_zero"])),
        seg_dice_mean=stat(np.mean, arr(ok_rows, lambda d: d["seg_dice"])),
        hd_zero_mean_mm=stat(np.mean, arr(pairs, lambda d: d["hd_zero_mm"])), hd_mean_mm=stat(np.mean, arr(pairs, lambda d: d["hd_mm"])),
        hd_zero_max_mm=stat(np.max, arr(pairs, lambda d: d["hd_zero_mm"])), hd_max_mm=stat(np.max, arr(pairs, lambda d: d["hd_mm"])),
        # the long-axis segmentations only: Dice over the LAX-LAX lines, apex metrics over the apex lines
        lax_dice_zero_mean=stat(np.mean, arr(laxlax, lambda d: d["seg_dice_zero"])),
        lax_dice_mean=stat(np.mean, arr(laxlax, lambda d: d["seg_dice"])),
        apex_zero_mean_mm=stat(np.mean, arr(line_rows, lambda d: d["apex_zero_mm"])),
        apex_mean_mm=stat(np.mean, arr(line_rows, lambda d: d["apex_mm"])),
        apex_zero_max_mm=stat(np.max, arr(line_rows, lambda d: d["apex_zero_mm"])),
        apex_max_mm=stat(np.max, arr(line_rows, lambda d: d["apex_mm"])),
        apex_dice_zero_mean=stat(np.mean, arr(line_rows, lambda d: d["apex_dice_zero"])),
        apex_dice_mean=stat(np.mean, arr(line_rows, lambda d: d["apex_dice"])),
        **sax_qc, **cop_qc, n_line_pairs=len(line_rows),
        steps=args.steps if lines else 0,
    )
    return summary, details, pairs, poses, history, regions


def region_rows(sid: str, planes: list[Plane], lines: list, lz: dict, lf: dict) -> list[dict]:
    """The metrics of every combination of slice types (REGION_COLUMNS rows): NCC, LNCC and the LV-only Dice
    (evaluate's dice_lv: labels LV_LABELS on both sides) at the acquired pose (lz) and the saved pose (lf), each the
    point-weighted mean over the combination's lines (n_points weights; lines whose value is NaN are left out).
    Combinations: all LAX-LAX lines together ('LAX-LAX'), every pair of long-axis images (e.g. 'LAX_2Ch-LAX_3Ch',
    names in sorted order), and every long-axis image against the SAX slices of each region (assign_sax_regions; e.g.
    'LAX_2Ch-SAX_basal'). Lines with an aortic slice are left out, as are SAX slices without a region."""
    groups = {}
    for m, (ia, ib, _) in enumerate(lines):
        pa, pb = planes[ia], planes[ib]
        if is_aorta(pa.view) or is_aorta(pb.view):
            continue
        if pa.view != "SAX" and pb.view != "SAX":
            a, b = sorted((pa.image, pb.image))
            groups.setdefault(("LAX-LAX", "LAX", "LAX", ""), []).append(m)
            groups.setdefault((f"{a}-{b}", a, b, ""), []).append(m)
        elif (pa.view == "SAX") != (pb.view == "SAX"):
            sax, lax = (pa, pb) if pa.view == "SAX" else (pb, pa)
            if sax.region:
                groups.setdefault((f"{lax.image}-{sax.image}_{sax.region}", lax.image, sax.image, sax.region),
                                  []).append(m)

    def order(item):                                     # LAX-LAX, the long-axis pairs, then each view base -> apex
        (combo, a, b, region), _ = item
        return (0,) if combo == "LAX-LAX" else (1, combo) if not region else (2, a, b, SAX_REGIONS.index(region))

    rows = []
    for (combo, a, b, region), idx in sorted(groups.items(), key=order):
        w = np.array([lines[m][2].n_points for m in idx], dtype=float)

        def wmean(values):
            x = np.asarray(values, dtype=float)[idx]
            ok = np.isfinite(x) & (w > 0)
            return float((x[ok] * w[ok]).sum() / w[ok].sum()) if ok.any() else float("nan")

        rows.append(dict(subject_id=sid, combination=combo, image_a=a, image_b=b, sax_region=region, n_lines=len(idx),
                         n_points=int(w.sum()), ncc_zero=wmean(lz["ncc"]), ncc=wmean(lf["ncc"]),
                         lncc_zero=wmean(lz["lncc"]), lncc=wmean(lf["lncc"]),
                         dice_zero=wmean(lz["dice_lv"]), dice=wmean(lf["dice_lv"])))
    return rows


def process_subject(subject: Subject, args, device, on_lines=None) -> tuple:
    """Align one subject (its slices as handed over by a loader, see SliceImage). Returns the summary row, the
    per-plane rows, the per-line-pair rows, the per-plane poses (plane, R, t; see corrected_affines), the
    pose/metric history (dict with 'planes' and 'steps' row lists, empty unless args.history_every > 0, i.e. with --logging)
    and the per-combination rows (region_rows; empty without --logging).

    `on_lines(planes, lines)` is called once the intersection lines are known, before alignment.
    Without args.logging only the poses are computed: no metrics are evaluated, the segmentations stay off the GPU and
    the summary / per-pair rows and the history are None / empty (the per-plane rows only hold n_pairs).
    """
    ctx = prepare_alignment(subject, args, device, on_lines)
    result = align_one(ctx, args, device) if ctx["lines"] else None
    return finish_alignment(ctx, result, args, device)


def process_subjects(subjects: list[Subject], args, device, on_lines: list | None = None,
                     datas: list | None = None, pres: list | None = None) -> list:
    """process_subject for several subjects, aligned together in one optimization (align_batch, --batch-size): per
    subject its result tuple, or the exception it raised. Subjects without lines are not optimized (as alone); a
    subject failing to prepare or finish fails alone; if the shared optimization fails (e.g. out of GPU memory), its
    subjects are aligned one at a time instead. Without batchable(args), every subject is aligned alone. `datas`: each
    subject's prepare_subject() and `pres` its prepare_lines() if already computed (run(): in the worker processes)."""
    ctxs, results = align_subjects(subjects, args, device, on_lines, datas, pres)
    return finish_subjects(ctxs, results, args, device)


def align_subjects(subjects: list[Subject], args, device, on_lines: list | None = None, datas: list | None = None,
                   pres: list | None = None, stats: dict | None = None) -> tuple[list, dict]:
    """process_subjects' first half: every subject prepared (its prepare_alignment() context, or the exception it
    raised) and aligned ({id(context): its align() result, or the exception}). A batch that runs out of GPU memory is
    split in halves (stats["oom"] counts it); one failing otherwise is aligned one subject at a time."""
    ctxs = []
    for i, s in enumerate(subjects):
        try:
            ctxs.append(prepare_alignment(s, args, device, on_lines[i] if on_lines else None,
                                          datas[i] if datas else None, pres[i] if pres else None))
        except Exception as e:  # noqa: BLE001
            ctxs.append(e)
    live = [c for c in ctxs if not isinstance(c, Exception) and c["lines"]]
    results = {}

    def together(group):
        if len(group) < 2:
            return
        try:
            results.update({id(c): r for c, r in zip(group, align_batch(group, args, device))})
        except torch.cuda.OutOfMemoryError:
            if stats is not None:
                stats["oom"] = stats.get("oom", 0) + 1
            print(f"\n[batch] {len(group)} subjects ran out of GPU memory together; splitting the batch", flush=True)
            torch.cuda.empty_cache()
            together(group[:len(group) // 2])
            together(group[len(group) // 2:])
        except Exception as e:  # noqa: BLE001
            print(f"\n[batch] aligning {len(group)} subjects together failed ({type(e).__name__}: {e}); "
                  f"aligning them one at a time", flush=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if batchable(args):
        # the subjects of a batch share one image volume, so one frame count (fewer phases for a subject whose cines
        # have fewer frames, one for a subject of single-frame slices only): one batch per count
        by_frames = {}
        for c in live:
            by_frames.setdefault(c["planes"][0].img.shape[0], []).append(c)
        for group in by_frames.values():
            together(group)
    for c in live:                                   # alone: not batchable, a single one, or its batch failed
        if id(c) not in results:
            try:
                results[id(c)] = align_one(c, args, device)
            except Exception as e:  # noqa: BLE001
                results[id(c)] = e
    return ctxs, results


def finish_subjects(ctxs: list, results: dict, args, device) -> list:
    """process_subjects' second half: per subject (align_subjects) its result tuple (finish_alignment), or the
    exception it raised."""
    out = []
    for c in ctxs:
        if isinstance(c, Exception):
            out.append(c)
            continue
        try:
            r = results.get(id(c)) if c["lines"] else None
            if isinstance(r, Exception):
                raise r
            out.append(finish_alignment(c, r, args, device))
        except Exception as e:  # noqa: BLE001
            out.append(e)
    return out


RELMOTION_COLUMNS = ["subject_id", "image", "kind", "slice_a", "slice_b", "n_pairs", "dx_mm", "dy_mm", "dz_mm",
                     "shift_inplane_mm", "shift_mm", "rot_inplane_deg", "rot_tilt_deg", "rot_deg"]


def sax_relative_motion(sid: str, poses: list, n_pairs: list[int]) -> list[dict]:
    """Estimated SAX motion relative to the other slices (independent of any motion the whole stack shares, e.g. a
    misplaced LAX reference). Pose i maps x -> R_i (x - c_i) + c_i + t_i (c_i: the slice's rotation centre).

    kind 'slice' (slice_a): the slice against the stack consensus -- at the slice's centre, the median over all
      evaluated slices j of the displacement pose j would give it; rotation against the median rotation.
    kind 'pair' (slice_a, slice_b = a + 1): slice b's pose relative to slice a's, T_a^-1 T_b, at b's centre.
    Translations are split in the frame of the slice concerned (dx, dy in-plane along its rows / columns, dz along
    its normal); rotations into in-plane (about the normal, signed) and tilt. Slices without intersection lines
    (pose not estimated) are left out. Per SAX image (stack; one per modality).
    """
    rows = []
    for image in dict.fromkeys(p.image for p, _, _ in poses if p.view == "SAX"):
        rows += _stack_relative_motion(sid, image, [(p, R, t, n_pairs[i]) for i, (p, R, t) in enumerate(poses)
                                                    if p.image == image])
    return rows


def _stack_relative_motion(sid: str, image: str, slices: list) -> list[dict]:
    """sax_relative_motion of one stack: slices = [(plane, R, t, n_pairs)]."""
    from scipy.spatial.transform import Rotation
    sax = sorted((p.slice_idx, p, np.asarray(R, float), np.asarray(t, float), npairs)
                 for p, R, t, npairs in slices if npairs > 0)
    rows = []
    if not sax:
        return rows

    def row(kind, a, b, p, npairs, d, R):
        rv = np.degrees(Rotation.from_matrix(R).as_rotvec())
        dx, dy, dz = float(d @ p.u_hat), float(d @ p.v_hat), float(d @ p.normal)
        inplane = float(rv @ p.normal)
        return dict(subject_id=sid, image=image, kind=kind, slice_a=a, slice_b=b, n_pairs=npairs, dx_mm=dx, dy_mm=dy, dz_mm=dz,
                    shift_inplane_mm=math.hypot(dx, dy), shift_mm=float(np.linalg.norm(d)), rot_inplane_deg=inplane,
                    rot_tilt_deg=float(np.linalg.norm(rv - inplane * p.normal)), rot_deg=float(np.linalg.norm(rv)))

    C = [p.rot_centre.astype(float) for _, p, _, _, _ in sax]
    rotvecs = Rotation.from_matrix(np.stack([R for _, _, R, _, _ in sax])).as_rotvec()
    R_med = Rotation.from_rotvec(np.median(rotvecs, 0)).as_matrix()
    for i, (z, p, R, t, npairs) in enumerate(sax):
        pred = np.median([Rj @ (C[i] - C[j]) + C[j] + tj - C[i] for j, (_, _, Rj, tj, _) in enumerate(sax)], 0)
        rows.append(row("slice", z, float("nan"), p, npairs, t - pred, R_med.T @ R))
    for (za, pa, Ra, ta, _), (zb, pb, Rb, tb, nb), ca, cb in zip(sax, sax[1:], C, C[1:]):
        if zb != za + 1:
            continue
        moved = Ra.T @ (Rb @ (cb - cb) + cb + tb - ca - ta) + ca          # T_a^-1 T_b applied to b's centre
        rows.append(row("pair", za, zb, pb, nb, moved - cb, Ra.T @ Rb))
    return rows


def empty_summary(sid: str, status: str) -> dict:
    row = {c: float("nan") for c in SUMMARY_COLUMNS}
    row.update(subject_id=sid, status=status, pose_model="none", n_lax_views=0, sax_n_slices=0, sax_n_eval=0,
               n_line_pairs=0, steps=0)
    return row


# --------------------------------------------------------------------------------------
# Corrected geometry
# --------------------------------------------------------------------------------------

def corrected_affine(affine: np.ndarray, plane: Plane, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Affine of the plane's content after applying the rigid pose (rotation about its rot_centre, then t)."""
    A = np.eye(4)
    A[:3, :3] = R @ affine[:3, :3]
    c = plane.rot_centre
    A[:3, 3] = R @ (plane.origin - c) + c + t
    return A


def corrected_affines(subject: Subject, poses: list[tuple[Plane, np.ndarray, np.ndarray]]) -> list[np.ndarray]:
    """The motion-corrected 4x4 affine of every slice of the subject (in subject.slices order): the acquired affine
    with the slice's pose applied, or unchanged for a slice without a pose (not aligned)."""
    pose_of = {plane.key: (plane, R, t) for plane, R, t in poses}
    return [corrected_affine(s.affine, *pose_of[s.key]) if s.key in pose_of else s.affine.copy() for s in subject.slices]



# --------------------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------------------

PLOT_STYLE = dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7", blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", yellow="#eda100",
                  blue_ramp=("#cde2fb", "#0d366b"))
LABEL_RAW = "as acquired"
LABEL_ALIGNED = "after alignment"
BASAL_SLICES = 3
APICAL_SLICES = 3
SUBSETS = {
    "LAX_2Ch": ("LAX 2Ch view", lambda d: d["view"] == "LAX_2Ch"),
    "LAX_3Ch": ("LAX 3Ch view", lambda d: d["view"] == "LAX_3Ch"),
    "LAX_4Ch": ("LAX 4Ch view", lambda d: d["view"] == "LAX_4Ch"),
    "SAX_basal": (f"basal SAX slices (index < {BASAL_SLICES})", lambda d: (d["view"] == "SAX") & (d["region"] == "basal")),
    "SAX_mid": ("mid-ventricular SAX slices", lambda d: (d["view"] == "SAX") & (d["region"] == "mid")),
    "SAX_apical": (f"apical SAX slices (last {APICAL_SLICES})", lambda d: (d["view"] == "SAX") & (d["region"] == "apical")),
}


def legacy_nmi(obj):
    """Runs from before the mi -> nmi rename: mi* columns of a DataFrame / mi* keys and --metric mi of a config dict."""
    old = lambda k: k == "mi" or k.startswith("mi_")
    if isinstance(obj, pd.DataFrame):
        return obj.rename(columns=lambda c: "n" + c if old(c) else c)
    obj = {("n" + k if old(k) else k): v for k, v in obj.items()}
    if obj.get("metric") == "mi":
        obj["metric"] = "nmi"
    return obj


def load_results(summary_path: Path, detail_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    summ = legacy_nmi(pd.read_csv(summary_path, dtype={"subject_id": str})).drop_duplicates("subject_id", keep="last")
    summ = summ[summ["status"] == "ok"].reset_index(drop=True)
    det = legacy_nmi(pd.read_csv(detail_path, dtype={"subject_id": str}))
    det = det.drop_duplicates(["subject_id", "image", "slice"], keep="last")
    if "view" not in det:                                  # runs from before views: the image name was the view
        det["view"] = det["image"]
    for col in ("nmi_zero", "nmi", "hd_zero_mm", "hd_mm"):  # runs from before these metrics were added
        if col not in det:
            det[col] = np.nan
    det = det[det["subject_id"].isin(summ["subject_id"])].reset_index(drop=True)
    n_slices = det[det["view"] == "SAX"].groupby("subject_id")["slice"].max() + 1
    det["n_slices"] = det["subject_id"].map(n_slices)
    det["region"] = np.where(det["slice"] < BASAL_SLICES, "basal",
                             np.where(det["slice"] >= det["n_slices"] - APICAL_SLICES, "apical", "mid"))
    det = det[det["n_pairs"] > 0].reset_index(drop=True)
    return summ, det


def load_pairs(pairs_path: Path, det: pd.DataFrame) -> pd.DataFrame | None:
    """Per-line-pair rows (SAX-LAX and LAX-LAX; the recorded SAX-SAX rows are left out) of the subjects in `det`,
    with the SAX slice's region attached (None if absent)."""
    if not pairs_path.exists():
        return None
    pairs = legacy_nmi(pd.read_csv(pairs_path, dtype={"subject_id": str}))
    pairs = pairs[~pairs["kind"].isin(NEIGHBOUR_KINDS)]
    pairs = pairs.drop_duplicates(["subject_id", "image_a", "slice_a", "image_b", "slice_b"], keep="last")
    pairs = pairs[pairs["subject_id"].isin(det["subject_id"])].reset_index(drop=True)
    for side in ("a", "b"):                                # runs from before views: the image name was the view
        if f"view_{side}" not in pairs:
            pairs[f"view_{side}"] = pairs[f"image_{side}"]
    regions = det[det["view"] == "SAX"].drop_duplicates(["subject_id", "slice"]).set_index(["subject_id", "slice"])["region"]
    key = pd.MultiIndex.from_arrays([pairs["subject_id"], pairs["slice_a"]])
    pairs["region"] = np.where(pairs["view_a"] == "SAX", regions.reindex(key).to_numpy(), "LAX")
    return pairs


def select_pairs(pairs: pd.DataFrame, subset: str) -> pd.DataFrame:
    """Pairs involving the planes of a SUBSETS entry (a LAX view, or a SAX region)."""
    if subset.startswith("LAX"):
        return pairs[(pairs["view_a"] == subset) | (pairs["view_b"] == subset)]
    return pairs[(pairs["view_a"] == "SAX") & (pairs["region"] == subset.split("_", 1)[1])]


def plot_pairs_figure(pairs: pd.DataFrame, label: str, out_path: Path, st=PLOT_STYLE) -> None:
    """One value per intersection-line pair: NCC / NMI / Dice / Hausdorff before and after, per pair type."""
    from matplotlib.figure import Figure

    kinds = [k for k in ("SAX-LAX", "LAX-LAX") if (pairs["kind"] == k).any()]
    if not kinds:
        return
    fig = Figure(figsize=(18, 4.2 * len(kinds) + 0.8), layout="constrained", facecolor=st["surface"])
    fig.get_layout_engine().set(rect=(0, 0, 1, 1 - 0.5 / (4.2 * len(kinds) + 0.8)))
    axes = np.atleast_2d(fig.subplots(len(kinds), 4))
    panels = [("ncc_zero", "ncc", "NCC", "NCC"), ("nmi_zero", "nmi", "Normalized MI", "normalized MI"),
              ("seg_dice_zero", "seg_dice", "Segmentation Dice", "Dice"),
              ("hd_zero_mm", "hd_mm", "Segmentation Hausdorff (mm)", "Hausdorff distance (mm)")]
    for r, kind in enumerate(kinds):
        sub = pairs[pairs["kind"] == kind]
        for c, (c0, c1, title, xl) in enumerate(panels):
            _hist_series(axes[r, c], [(sub[c0], LABEL_RAW, st["aqua"]), (sub[c1], LABEL_ALIGNED, st["orange"])],
                         f"{title}, {kind} ({len(sub)} pairs)", xlabel=xl)
    _legend(fig, axes[0, 0].get_legend_handles_labels())
    fig.suptitle(f"Per intersection-line pair, {label}  (n = {pairs['subject_id'].nunique()} subjects, {len(pairs)} pairs)",
                 color=st["ink"], fontsize=12, x=0.02, ha="left")
    fig.savefig(out_path, dpi=150)


def per_subject_metrics(det: pd.DataFrame, reference: pd.DataFrame | None = None) -> pd.DataFrame:
    """Subject-level metrics recomputed from a subset of per-slice rows.

    Jitter is the RMS deviation of the slice shifts from their mean. When the subset holds a
    single plane per subject, `reference` rows (e.g. all LAX views) define the mean instead.
    """

    def rms(x):
        x = x[np.isfinite(x)]
        return float(np.sqrt(np.mean(x ** 2))) if x.size else np.nan

    def nanstat(fn, x):
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        return float(fn(x)) if x.size else np.nan

    def jitter(vec, ref_vec=None):
        vec = vec[np.isfinite(vec).all(1)]
        if len(vec) >= 2:
            return rms(np.hypot(*(vec - vec.mean(0)).T))
        if len(vec) == 1 and ref_vec is not None:
            ref_vec = ref_vec[np.isfinite(ref_vec).all(1)]
            if len(ref_vec) >= 2:
                return float(np.hypot(*(vec[0] - ref_vec.mean(0))))
        return np.nan

    ref_groups = dict(tuple(reference.groupby("subject_id"))) if reference is not None else {}
    rows = {}
    for sid, g in det.groupby("subject_id"):
        v = g[["dx_mm", "dy_mm"]].values
        ref = ref_groups.get(sid)
        mag = np.hypot(v[:, 0], v[:, 1])
        rows[sid] = dict(
            n=len(g), shift_rms=rms(mag), shift_max=nanstat(np.max, mag), shift_mean=nanstat(np.mean, mag),
            shift_jitter=jitter(v, ref[["dx_mm", "dy_mm"]].values if ref is not None else None),
            dz_rms=rms(g["dz_mm"].values), rot_inplane_rms=rms(g["rot_inplane_deg"].values), tilt_rms=rms(g["rot_tilt_deg"].values),
            ncc_zero=nanstat(np.mean, g["ncc_zero"]), ncc=nanstat(np.mean, g["ncc"]),
            nmi_zero=nanstat(np.mean, g["nmi_zero"]), nmi=nanstat(np.mean, g["nmi"]),
            seg_dice_zero=nanstat(np.mean, g["seg_dice_zero"]), seg_dice=nanstat(np.mean, g["seg_dice"]),
            hd_zero=nanstat(np.max, g["hd_zero_mm"]), hd=nanstat(np.max, g["hd_mm"]),
        )
    m = pd.DataFrame.from_dict(rows, orient="index")
    m["ncc_gain"] = m["ncc"] - m["ncc_zero"]
    m["nmi_gain"] = m["nmi"] - m["nmi_zero"]
    return m


def _style_axis(ax, st=PLOT_STYLE):
    from matplotlib.ticker import MaxNLocator
    ax.set_facecolor(st["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(st["axis"])
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=st["muted"], labelsize=8, length=3, width=0.8)
    ax.yaxis.grid(True, color=st["grid"], linewidth=0.6)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(st["ink2"])
    ax.yaxis.label.set_color(st["ink2"])
    ax.title.set_color(st["ink"])


def _clean(values) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    return v[np.isfinite(v)]


HIST_BINS = 25          # bins per histogram panel


def _hist_series(ax, series: list[tuple], title: str, xlabel: str = "", n_bins: int | None = None, st=PLOT_STYLE) -> bool:
    """Grouped-bar histogram: within each bin the series are drawn as narrow side-by-side bars."""
    series = [(_clean(v), label, color) for v, label, color in series]
    series = [s for s in series if s[0].size]
    if series:
        allv = np.concatenate([s[0] for s in series])
        lo, hi = float(allv.min()), float(allv.max())
        if hi - lo < 1e-9:
            lo, hi = lo - 0.5, hi + 0.5
        edges = np.linspace(lo, hi, (n_bins or HIST_BINS) + 1)
        width = edges[1] - edges[0]
        k = len(series)
        bw = 0.8 * width / k
        for i, (v, label, color) in enumerate(series):
            counts, _ = np.histogram(v, bins=edges)
            x = edges[:-1] + 0.1 * width + bw * (i + 0.5)
            ax.bar(x, counts, width=bw, color=color, linewidth=0, label=label)
            med = float(np.median(v))
            ax.axvline(med, color=color, linestyle=(0, (3, 2)), linewidth=1.2)
            ax.annotate(f"median {med:.2f}", xy=(med, 1), xycoords=("data", "axes fraction"),
                        xytext=(4, -4 - 11 * i), textcoords="offset points", ha="left", va="top",
                        fontsize=8, color=st["ink2"])
        ax.set_xlim(edges[0], edges[-1])
    else:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, color=st["muted"], fontsize=9)
    ax.set_title(title, fontsize=10, loc="left")
    ax.set_ylabel("count")
    if xlabel:
        ax.set_xlabel(xlabel)
    _style_axis(ax, st)
    return bool(series)


def _legend(fig, handles_labels, st=PLOT_STYLE):
    handles, labels = handles_labels
    if len(labels) > 1:
        leg = fig.legend(handles, labels, loc="upper right", ncol=len(labels), frameon=False, fontsize=9,
                         bbox_to_anchor=(0.99, 0.995))
        for text in leg.get_texts():
            text.set_color(st["ink2"])


def _inset_legend(ax, st=PLOT_STYLE):
    leg = ax.legend(frameon=False, fontsize=8, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2,
                    borderaxespad=0, handlelength=1.2)
    for text in leg.get_texts():
        text.set_color(st["ink2"])


def plot_metrics_figure(metrics: pd.DataFrame, label: str, out_path: Path, lax_metrics: pd.DataFrame | None = None,
                        jitter_title: str = "Inter-slice jitter (mm)", st=PLOT_STYLE):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 4, figsize=(18, 10.5), layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.95))
    one = lambda col, frame=metrics: [(frame[col], LABEL_ALIGNED, st["orange"])]  # noqa: E731
    two = lambda col0, col1: [(metrics[col0], LABEL_RAW, st["aqua"]), (metrics[col1], LABEL_ALIGNED, st["orange"])]  # noqa: E731
    _hist_series(axes[0, 0], one("shift_rms"), "In-plane shift RMS (mm)")
    _hist_series(axes[0, 1], one("shift_jitter"), jitter_title)
    _hist_series(axes[0, 2], one("shift_max"), "Max slice shift (mm)")
    if lax_metrics is not None:
        _hist_series(axes[0, 3], one("shift_mean", lax_metrics), "LAX mean in-plane shift (mm)")
    else:
        _hist_series(axes[0, 3], one("shift_mean"), "Mean in-plane shift (mm)")
    _hist_series(axes[1, 0], two("ncc_zero", "ncc"), "Mean NCC at slice intersections")
    _hist_series(axes[1, 1], one("ncc_gain"), "NCC gain over as-acquired")
    _hist_series(axes[1, 2], two("nmi_zero", "nmi"), "Mean normalized MI at slice intersections")
    _hist_series(axes[1, 3], one("nmi_gain"), "Normalized MI gain over as-acquired")
    _hist_series(axes[2, 0], one("dz_rms"), "Through-plane shift RMS (mm)")
    if _hist_series(axes[2, 1], [(metrics["rot_inplane_rms"], "in-plane", st["orange"]),
                                 (metrics["tilt_rms"], "tilt (weak)", st["yellow"])], "Rotation RMS (deg)"):
        _inset_legend(axes[2, 1])
    _hist_series(axes[2, 2], two("seg_dice_zero", "seg_dice"), "Segmentation Dice at slice intersections")
    _hist_series(axes[2, 3], two("hd_zero", "hd"), "Worst segmentation Hausdorff (mm)")
    _legend(fig, axes[1, 0].get_legend_handles_labels())
    fig.suptitle(f"Subject-level misalignment metrics (one value per subject), {label}  (n = {len(metrics)} subjects)",
                 color=st["ink"], fontsize=12, x=0.02, ha="left")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_translation_figure(det: pd.DataFrame, label: str, out_path: Path, per_image: bool, st=PLOT_STYLE):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    fig = plt.figure(figsize=(17, 8.5), layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.94))
    gs = fig.add_gridspec(2, 4, height_ratios=(1, 1.3))
    if per_image:
        names = list(dict.fromkeys(det["image"]))         # SAX first, then the others in the order found
        sax_names = set(det.loc[det["view"] == "SAX", "image"])
        images = [im for im in names if im in sax_names] + [im for im in names if im not in sax_names]
        for i, image in enumerate(images[:4]):
            sub = det[det["image"] == image]
            _hist_series(fig.add_subplot(gs[0, i]), [(sub["shift_inplane_mm"], LABEL_ALIGNED, st["orange"])],
                         f"{image} in-plane shift (mm), {len(sub)} slices", xlabel="translation distance (mm)")
    else:
        _hist_series(fig.add_subplot(gs[0, 0]), [(det["shift_inplane_mm"], LABEL_ALIGNED, st["orange"])],
                     f"In-plane shift (mm), {len(det)} slices", xlabel="translation distance (mm)")
        _hist_series(fig.add_subplot(gs[0, 1]), [(det["dz_mm"], LABEL_ALIGNED, st["orange"])],
                     "Through-plane shift (mm)", xlabel="signed translation (mm)")
        ax = fig.add_subplot(gs[0, 2])
        if _hist_series(ax, [(det["rot_inplane_deg"], "in-plane", st["orange"]), (det["rot_tilt_deg"], "tilt (weak)", st["yellow"])],
                        "Rotation (deg)", xlabel="degrees"):
            _inset_legend(ax)
        _hist_series(fig.add_subplot(gs[0, 3]), [(det["seg_dice_zero"], LABEL_RAW, st["aqua"]), (det["seg_dice"], LABEL_ALIGNED, st["orange"])],
                     "Segmentation Dice along intersections", xlabel="Dice")

    has_sax = bool((det["view"] == "SAX").any())
    hex_rows = det[det["view"] == "SAX"] if has_sax else det
    hex_name = "SAX" if has_sax else "LAX"
    ax = fig.add_subplot(gs[1, 0] if has_sax else gs[1, :])
    ok = np.isfinite(hex_rows["dx_mm"]) & np.isfinite(hex_rows["dy_mm"])
    lim = 1.0
    if ok.any():
        vals = np.abs(hex_rows.loc[ok, ["dx_mm", "dy_mm"]].values).ravel()
        lim = max(float(np.percentile(vals, 99.5)), 1.0)
        hb = ax.hexbin(hex_rows.loc[ok, "dx_mm"], hex_rows.loc[ok, "dy_mm"], gridsize=35, extent=(-lim, lim, -lim, lim),
                       cmap=LinearSegmentedColormap.from_list("blue_ramp", st["blue_ramp"]), mincnt=1,
                       linewidths=0.2, edgecolors=st["surface"])
        cb = fig.colorbar(hb, cax=ax.inset_axes([1.04, 0.05, 0.04, 0.9]))
        cb.set_label("slices", color=st["ink2"])
        cb.ax.tick_params(colors=st["muted"], labelsize=8)
        cb.outline.set_edgecolor(st["axis"])
    ax.axhline(0, color=st["axis"], linewidth=0.8)
    ax.axvline(0, color=st["axis"], linewidth=0.8)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_box_aspect(1)
    ax.set_anchor("W")
    ax.set_xlabel("dx (mm, along image rows)")
    ax.set_ylabel("dy (mm, along image columns)")
    ax.set_title(f"{hex_name} in-plane correction vectors", fontsize=10, loc="left")
    _style_axis(ax, st)
    ax.yaxis.grid(False)

    sax = det[det["view"] == "SAX"]
    if len(sax):
        ax = fig.add_subplot(gs[1, 1:])
        q = sax.groupby("slice")["shift_inplane_mm"].quantile([0.25, 0.5, 0.75]).unstack()
        ax.fill_between(q.index, q[0.25], q[0.75], color=st["orange"], alpha=0.15, linewidth=0)
        ax.plot(q.index, q[0.5], color=st["orange"], linewidth=2, marker="o", markersize=4, label=LABEL_ALIGNED)
        for z, c in sax.groupby("slice").size().items():
            ax.annotate(str(c), xy=(z, 0), xycoords=("data", "axes fraction"), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom", fontsize=7, color=st["muted"])
        ax.set_xticks(sorted(sax["slice"].unique()))
        ax.set_xlabel("SAX slice index (count of slices shown at baseline)")
        ax.set_ylabel("in-plane translation distance (mm)")
        ax.set_title("SAX shift by slice index (median, IQR band)", fontsize=10, loc="left")
        _style_axis(ax, st)
    _legend(fig, fig.axes[0].get_legend_handles_labels() if per_image else fig.axes[3].get_legend_handles_labels())
    fig.suptitle(f"Estimated breath-hold motion, {label}  (n = {det['subject_id'].nunique()} subjects)",
                 color=st["ink"], fontsize=12, x=0.02, ha="left")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_results(summary_path: Path, detail_path: Path, out_dir: Path, pairs_path: Path | None = None) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    st = PLOT_STYLE
    _, det = load_results(summary_path, detail_path)
    pairs = load_pairs(pairs_path, det) if pairs_path is not None else None
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "sans-serif", "figure.facecolor": st["surface"],
                         "savefig.facecolor": st["surface"]})
    written = []
    sax, lax = det[det["view"] == "SAX"], det[det["view"] != "SAX"]
    p = out_dir / "motion_metrics_hist.png"
    if len(sax):
        plot_metrics_figure(per_subject_metrics(sax), "all SAX slices", p,
                            lax_metrics=per_subject_metrics(lax) if len(lax) else None)
    else:                                            # no SAX image (e.g. oblique stacks only): all slices together
        plot_metrics_figure(per_subject_metrics(det), "all slices", p)
    written.append(p)
    p = out_dir / "motion_translations.png"
    plot_translation_figure(det, "all slices", p, per_image=True)
    written.append(p)
    if pairs is not None and len(pairs):
        p = out_dir / "motion_pairs_hist.png"
        plot_pairs_figure(pairs, "all pairs", p)
        written.append(p)
    for name, (label, select) in SUBSETS.items():
        sub = det[select(det)]
        if sub.empty:
            continue
        if pairs is not None:
            sub_pairs = select_pairs(pairs, name)
            if len(sub_pairs):
                p = out_dir / f"motion_pairs_hist_{name}.png"
                plot_pairs_figure(sub_pairs, f"pairs involving {label}", p)
                written.append(p)
        p = out_dir / f"motion_metrics_hist_{name}.png"
        if name.startswith("LAX"):
            plot_metrics_figure(per_subject_metrics(sub, reference=lax), label, p, jitter_title="Deviation from mean LAX shift (mm)")
        else:
            plot_metrics_figure(per_subject_metrics(sub), label, p)
        written.append(p)
        p = out_dir / f"motion_translations_{name}.png"
        plot_translation_figure(sub, label, p, per_image=False)
        written.append(p)
    return written


# --------------------------------------------------------------------------------------
# Subject-level outputs (thread-safe: matplotlib Figure objects, no pyplot)
# --------------------------------------------------------------------------------------

def mask_plot_data(planes: list[Plane], lines: list[tuple[int, int, Line]], extent_mm: float = 250.0,
                   step_mm: float = 1.0) -> dict:
    """CPU copies of what the mask overlay needs: LAX image/seg/mask and, per line in voxel coords, the full
    +-extent candidate segment, the part inside both images and the points used in the loss."""
    views = {}
    by_image = {}
    for i, p in enumerate(planes):
        if p.view != "SAX" and p.mask is not None:
            by_image.setdefault(p.image, []).append(i)
    shown = {ids[len(ids) // 2] for ids in by_image.values()}          # one panel per image: a stack's middle slice
    for i, p in enumerate(planes):
        if i not in shown:
            continue
        name = p.image if len(by_image[p.image]) == 1 else f"{p.image} slice {p.slice_idx}"
        views[i] = dict(name=name, img=p.img[0, 0].float().cpu().numpy(),
                        seg=(p.seg[:, 0].cpu().numpy() > 0).any(0) if p.seg is not None else None,
                        mask=p.mask[0, 0].cpu().numpy() > 0.5, pts=[], soft=[], full=[], inimg=[])
    for ia, ib, line in lines:
        if line.x0 is None:
            continue
        w = line.w.cpu().numpy() > 0
        world = line.x0[None] + line.t.cpu().numpy()[w, None] * line.d[None]
        soft = (line.w_loss.cpu().numpy() > 0) & ~w if line.w_loss is not None else np.zeros_like(w)
        world_soft = line.x0[None] + line.t.cpu().numpy()[soft, None] * line.d[None]
        # full candidate line (+-extent) and the part of it inside both images
        t_all = np.arange(-extent_mm, extent_mm + 1e-6, step_mm)
        world_all = line.x0[None] + t_all[:, None] * line.d[None]
        pts_all = torch.tensor(world_all, dtype=torch.float32, device=planes[ia].img.device)
        inb = (inside(planes[ia], pts_all) & inside(planes[ib], pts_all)).cpu().numpy()
        kind = "SAX-LAX" if "SAX" in (planes[ia].view, planes[ib].view) else "LAX-LAX"
        for i in (ia, ib):
            if i in views:
                p = planes[i]
                to_vox = lambda xyz: (xyz @ p.inv_rot.cpu().numpy().T.astype(np.float64) + p.inv_trans.cpu().numpy())[:, :2]  # noqa: E731
                views[i]["pts"].append((to_vox(world), kind))
                if soft.any():
                    views[i]["soft"].append((to_vox(world_soft), kind))
                views[i]["full"].append((to_vox(world_all[[0, -1]]), kind))
                if inb.any():
                    idx = np.flatnonzero(inb)
                    views[i]["inimg"].append((to_vox(world_all[[idx[0], idx[-1]]]), kind))
    return dict(views=list(views.values()))


LINE_COLUMNS = ["image_a", "slice_a", "image_b", "slice_b", "kind", "n_points",
                "x0_x", "x0_y", "x0_z", "d_x", "d_y", "d_z", "t_used_lo", "t_used_hi", "t_img_lo", "t_img_hi"]
MASK_CONTOUR_COLUMNS = ["image", "slice", "contour", "i", "j"]


def save_line_geometry(planes: list[Plane], lines: list[tuple[int, int, Line]], metrics_dir: Path,
                       extent_mm: float, step_mm: float) -> None:
    """lines.csv: the intersection lines (acquired geometry) with the parameter range inside both images and
    the range actually sampled; mask_contours.csv: 0.5-level contours of every dilated LAX mask (voxel i, j)."""
    from contourpy import contour_generator

    rows = []
    t_all = np.arange(-extent_mm, extent_mm + 1e-6, step_mm)
    for ia, ib, line in lines:
        if line.x0 is None:
            continue
        a, b = planes[ia], planes[ib]
        t_used = line.t.cpu().numpy()[line.w.cpu().numpy() > 0]
        pts_all = torch.tensor(line.x0[None] + t_all[:, None] * line.d[None], dtype=torch.float32, device=a.img.device)
        inb = np.flatnonzero((inside(a, pts_all) & inside(b, pts_all)).cpu().numpy())
        nan = float("nan")
        rows.append(dict(image_a=a.image, slice_a=a.slice_idx, image_b=b.image, slice_b=b.slice_idx,
                         kind="SAX-LAX" if "SAX" in (a.view, b.view) else "LAX-LAX", n_points=int(line.n_points),
                         x0_x=float(line.x0[0]), x0_y=float(line.x0[1]), x0_z=float(line.x0[2]),
                         d_x=float(line.d[0]), d_y=float(line.d[1]), d_z=float(line.d[2]),
                         t_used_lo=float(t_used.min()) if t_used.size else nan,
                         t_used_hi=float(t_used.max()) if t_used.size else nan,
                         t_img_lo=float(t_all[inb[0]]) if inb.size else nan,
                         t_img_hi=float(t_all[inb[-1]]) if inb.size else nan))
    pd.DataFrame(rows, columns=LINE_COLUMNS).to_csv(metrics_dir / "lines.csv", index=False, float_format="%.4f")

    crow = []
    for p in planes:
        if p.mask is None:
            continue
        mask = p.mask[0, 0].cpu().numpy() > 0.5                     # (H, W) = voxel axes (i, j)
        if not mask.any():
            continue
        # contourpy indexes z[row, col] = z[i, j] and returns (x, y) = (j, i)
        for c, seg in enumerate(contour_generator(z=mask.astype(float)).lines(0.5)):
            for x, y in seg:
                crow.append(dict(image=p.image, slice=p.slice_idx, contour=c, i=float(y), j=float(x)))
    pd.DataFrame(crow, columns=MASK_CONTOUR_COLUMNS).to_csv(metrics_dir / "mask_contours.csv", index=False,
                                                           float_format="%.3f")


def save_mask_plot(viz: dict, subject_id: str, dilation_mm: float, path: Path, extent_mm: float = 250.0,
                   st=PLOT_STYLE) -> None:
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    def matplotlib_rect(xy, w, h, color):
        return Rectangle(xy, w, h, fill=False, edgecolor=color, lw=0.8)

    views = viz["views"]
    if not views:
        return
    fig = Figure(figsize=(6 * len(views), 6.4), layout="constrained", facecolor=st["surface"])
    axes = np.atleast_1d(fig.subplots(1, len(views)))
    colors = {"SAX-LAX": st["blue"], "LAX-LAX": st["orange"]}
    for ax, v in zip(axes, views):
        H, W = v["img"].shape
        ax.set_facecolor("#000000")
        ax.imshow(v["img"].T, cmap="gray", origin="lower", extent=(-0.5, H - 0.5, -0.5, W - 0.5))
        ax.add_patch(matplotlib_rect((-0.5, -0.5), H, W, st["ink2"]))
        if v["seg"] is not None and v["seg"].any():
            ax.contour(v["seg"].T.astype(float), levels=[0.5], colors=st["aqua"], linewidths=1.5, origin="lower")
        ax.contour(v["mask"].T.astype(float), levels=[0.5], colors=st["yellow"], linewidths=1.5, linestyles="--", origin="lower")
        for vox, kind in v["full"]:
            ax.plot(vox[:, 0], vox[:, 1], color=colors[kind], lw=0.6, ls=(0, (4, 3)), alpha=0.45)
        for vox, kind in v["inimg"]:
            ax.plot(vox[:, 0], vox[:, 1], color=colors[kind], lw=0.8, alpha=0.8)
        for vox, kind in v["soft"]:
            ax.plot(vox[:, 0], vox[:, 1], ".", ms=1.6, color=st["axis"])
        for vox, kind in v["pts"]:
            ax.plot(vox[:, 0], vox[:, 1], ".", ms=1.6, color=colors[kind])
        ends = np.concatenate([vox for vox, _ in v["full"]] + [np.array([[0, 0], [H, W]])])
        pad = 5
        ax.set_xlim(ends[:, 0].min() - pad, ends[:, 0].max() + pad)
        ax.set_ylim(ends[:, 1].min() - pad, ends[:, 1].max() + pad)
        ax.set_aspect("equal")
        ax.set_title(v["name"], loc="left", fontsize=11, color=st["ink"])
        ax.set_xticks([])
        ax.set_yticks([])
    handles = [Line2D([], [], color=st["aqua"], lw=1.5, label="heart segmentation (union over frames)"),
               Line2D([], [], color=st["yellow"], lw=1.5, ls="--", label=f"dilated mask (+{dilation_mm:g} mm)"),
               Line2D([], [], color=st["blue"], lw=0.6, ls=(0, (4, 3)), label=f"candidate line (+-{extent_mm:g} mm)"),
               Line2D([], [], color=st["blue"], lw=0.8, label="part inside both images"),
               Line2D([], [], color=st["blue"], marker=".", ls="", label="points used: SAX-LAX"),
               Line2D([], [], color=st["orange"], marker=".", ls="", label="points used: LAX-LAX")]
    if any(v["soft"] for v in views):
        handles.append(Line2D([], [], color=st["axis"], marker=".", ls="", label="points used, down-weighted (outside the mask)"))
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False, fontsize=9)
    fig.suptitle(f"Subject {subject_id}: intersection-line points entering the NCC loss (frame 0, acquired geometry)",
                 x=0.01, ha="left", color=st["ink"])
    fig.savefig(path, dpi=130)


def save_subject_metrics(summary: dict, details: list[dict], pairs: list[dict], metrics_dir: Path, plots: bool,
                         history: dict | None = None, st=PLOT_STYLE, regions: list[dict] | None = None) -> None:
    """Per-subject CSVs (summary row, per-plane rows, per-line-pair rows, per-combination rows (region_rows),
    pose/metric history) and figures."""
    metrics_dir.mkdir(parents=True, exist_ok=True)
    if history and history["planes"]:
        pd.DataFrame([fmt_row(r) for r in history["planes"]], columns=HISTORY_COLUMNS).to_csv(metrics_dir / "history.csv", index=False)
        pd.DataFrame([fmt_row(r) for r in history["steps"]], columns=HISTORY_STEP_COLUMNS).to_csv(
            metrics_dir / "history_summary.csv", index=False)
        if history.get("sax_lines"):
            pd.DataFrame([fmt_row(r) for r in history["sax_lines"]], columns=SAX_LINE_COLUMNS).to_csv(
                metrics_dir / "sax_lines.csv", index=False)
        if history.get("long_axis"):
            pd.DataFrame([fmt_row(r) for r in history["long_axis"]], columns=LONG_AXIS_COLUMNS).to_csv(
                metrics_dir / "long_axis.csv", index=False)
    pd.DataFrame([fmt_row(summary)], columns=SUMMARY_COLUMNS).to_csv(metrics_dir / "summary.csv", index=False)
    pd.DataFrame([fmt_row(r) for r in details], columns=DETAIL_COLUMNS).to_csv(metrics_dir / "slices.csv", index=False)
    pd.DataFrame([fmt_row(r) for r in pairs], columns=PAIR_COLUMNS).to_csv(metrics_dir / "pairs.csv", index=False)
    if regions is not None:
        pd.DataFrame([fmt_row(r) for r in regions], columns=REGION_COLUMNS).to_csv(metrics_dir / "regions.csv", index=False)
    det = pd.DataFrame(details, columns=DETAIL_COLUMNS)
    det = det[det["n_pairs"] > 0]
    pdf = pd.DataFrame(pairs, columns=PAIR_COLUMNS)
    pdf = pdf[~pdf["kind"].isin(NEIGHBOUR_KINDS)]                 # the figures cover the SAX-LAX / LAX-LAX lines
    if not plots or det.empty:
        return
    if len(pdf):
        plot_pairs_figure(pdf, f"subject {summary['subject_id']}", metrics_dir / "motion_pairs.png")
    from matplotlib.figure import Figure
    from matplotlib.ticker import AutoLocator

    sax, lax = det[det["view"] == "SAX"], det[det["view"] != "SAX"]
    sid = summary["subject_id"]
    fig = Figure(figsize=(17, 12), layout="constrained", facecolor=st["surface"])
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.95))
    gs = fig.add_gridspec(3, 4)
    one = lambda frame, col: [(frame[col], LABEL_ALIGNED, st["orange"])]  # noqa: E731
    two = lambda col0, col1: [(det[col0], LABEL_RAW, st["aqua"]), (det[col1], LABEL_ALIGNED, st["orange"])]  # noqa: E731
    _hist_series(fig.add_subplot(gs[0, 0]), one(sax, "shift_inplane_mm"), f"SAX in-plane shift (mm), {len(sax)} slices",
                 xlabel="translation distance (mm)")
    _hist_series(fig.add_subplot(gs[0, 1]), one(sax, "dz_mm"), "SAX through-plane shift (mm)", xlabel="signed translation (mm)")
    ax = fig.add_subplot(gs[0, 2])
    if _hist_series(ax, [(det["rot_inplane_deg"], "in-plane", st["orange"]), (det["rot_tilt_deg"], "tilt (weak)", st["yellow"])],
                    "Rotation (deg)", xlabel="degrees, all planes"):
        _inset_legend(ax)
    _hist_series(fig.add_subplot(gs[0, 3]), one(lax, "shift_inplane_mm"), f"LAX in-plane shift (mm), {len(lax)} views",
                 xlabel="translation distance (mm)")
    two = lambda col0, col1: [(pdf[col0], LABEL_RAW, st["aqua"]), (pdf[col1], LABEL_ALIGNED, st["orange"])]  # noqa: E731
    _hist_series(fig.add_subplot(gs[1, 0]), two("ncc_zero", "ncc"), f"NCC per line pair ({len(pdf)} pairs)", xlabel="NCC")
    _hist_series(fig.add_subplot(gs[1, 1]), two("nmi_zero", "nmi"), "Normalized MI per line pair", xlabel="normalized MI")
    _hist_series(fig.add_subplot(gs[1, 2]), two("seg_dice_zero", "seg_dice"), "Segmentation Dice per line pair", xlabel="Dice")
    _hist_series(fig.add_subplot(gs[1, 3]), two("hd_zero_mm", "hd_mm"), "Segmentation Hausdorff per line pair (mm)",
                 xlabel="Hausdorff distance (mm)")

    ax = fig.add_subplot(gs[2, :])
    if len(sax):
        x = sax["slice"].to_numpy()
        bw = 0.27
        for k, (col, name, color) in enumerate([("dx_mm", "dx (rows)", st["blue"]), ("dy_mm", "dy (columns)", st["aqua"]),
                                                ("dz_mm", "dz (through-plane)", st["orange"])]):
            ax.bar(x + (k - 1) * bw, sax[col].to_numpy(), width=bw, color=color, linewidth=0, label=name)
        ax.axhline(0, color=st["axis"], linewidth=0.8)
        ax.set_xticks(sorted(x))
        ax.set_xlabel("SAX slice index")
        ax.set_ylabel("translation (mm)")
        ax.set_title("SAX translation per slice (relative to mean LAX pose)", fontsize=10, loc="left")
        _style_axis(ax, st)
        ax.yaxis.set_major_locator(AutoLocator())
        _inset_legend(ax)
    _legend(fig, fig.axes[4].get_legend_handles_labels())
    fig.suptitle(f"Subject {sid}: NCC {summary['ncc_zero_mean']:.3f} -> {summary['ncc_mean']:.3f}, "
                 f"NMI {summary['nmi_zero_mean']:.3f} -> {summary['nmi_mean']:.3f}, "
                 f"Dice {summary['seg_dice_zero_mean']:.3f} -> {summary['seg_dice_mean']:.3f}, "
                 f"worst HD {summary['hd_zero_max_mm']:.1f} -> {summary['hd_max_mm']:.1f} mm, "
                 f"SAX jitter {summary['sax_shift_jitter_mm']:.2f} mm", color=st["ink"], fontsize=12, x=0.02, ha="left")
    fig.savefig(metrics_dir / "motion_subject.png", dpi=130)


# --------------------------------------------------------------------------------------
# CSV / driver
# --------------------------------------------------------------------------------------

def open_csv(path: Path, columns: list[str]):
    new = not path.exists() or path.stat().st_size == 0
    if not new:
        with open(path, newline="") as f:
            header = next(csv.reader(f), [])
        if header != columns:
            sys.exit(f"{path} has a different column layout than this version of the script; "
                     f"use a new --run-name or move the old file aside.")
    f = open(path, "a", newline="")
    writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
    if new:
        writer.writeheader()
    return f, writer


def fmt_row(row: dict) -> dict:
    out = {}
    for k, v in row.items():
        if isinstance(v, float):
            out[k] = "nan" if math.isnan(v) else f"{v:.4f}"
        else:
            out[k] = v
    return out


# arguments that only affect input/output/bookkeeping, not the result, and are therefore left out of run names
NAME_EXCLUDED = {"root", "out_root", "run_name", "logging", "history_every", "no_cuda_graph", "max_subjects", "skip_processed",
                 "device", "workers", "step_progress"}
HISTORY_EVERY = 0.02    # with --logging: pose/metric snapshot every this fraction of the run (plus step 0 and the best pose)
def _worker_init():
    """Worker processes of run(): one thread each (the run has as many processes as --workers), no GPU."""
    torch.set_num_threads(1)


def load_job(load, is_complete, subj_dir: Path, out_dir: Path, args):
    """Worker process: SKIPPED_JOB when --skip-processed finds the subject complete, else (subject, data, pre) pickled
    to bytes (unpack_loaded) -- the subject without its pixels (the main process only needs its slices' geometry and
    identity; the images are written by save_job, which reads them again), its prepare_subject() (the frames used, masks and distance maps)
    and its prepare_lines() (intersection lines of every mask stage), None when it has nothing to align. The CPU
    work of loading and preparing runs here, in parallel."""
    if args.skip_processed and is_complete(subj_dir, out_dir):
        return SKIPPED_JOB
    subject = load(subj_dir, args)
    data = prepare_subject(subject, args) if subject.slices and subject.status == "ok" else None
    on_lines = None
    if getattr(args, "logging", True):              # the nominal lines (lines.csv) and their mask overlay, written here
        def on_lines(planes, lines):
            metrics_dir = out_dir / "metrics"
            metrics_dir.mkdir(parents=True, exist_ok=True)
            save_line_geometry(planes, lines, metrics_dir, args.line_extent_mm, args.line_step_mm)
            save_mask_plot(mask_plot_data(planes, lines, args.line_extent_mm, args.line_step_mm), subj_dir.name,
                           args.mask_dilation_mm, metrics_dir / "mask_overlay.png", args.line_extent_mm)
    pre = prepare_lines(data, args, on_lines) if data is not None and (data["sax"] or data["lax"]) else None
    light = Subject(subject.subject_id, [dataclasses.replace(s, data=None, seg=None) for s in subject.slices],
                    subject.status)
    # sent back as one byte string (plain pickle: tensors by value). Returned as objects, the result would go through
    # torch's multiprocessing pickler, which on Linux shares every CPU tensor through a file descriptor of its own --
    # hundreds per subject (its lines), so a few queued subjects exceed the open-file limit (OSError: [Errno 24] Too
    # many open files) and every load fell back to the main process
    return pickle.dumps((light, data, pre), protocol=pickle.HIGHEST_PROTOCOL)


def unpack_loaded(res):
    """load_job's result as (subject, data, pre), or SKIPPED_JOB."""
    return pickle.loads(res) if isinstance(res, bytes) else res


def save_job(load, save, subj_dir: Path, affines: list, out_dir: Path, args) -> None:
    """Worker process: write the subject with the given affines (save), reading its images from subj_dir again (a
    worker's CPU time instead of moving the full images between processes)."""
    save(load(subj_dir, args), affines, out_dir, args)


AUTO_BATCH_PROBE = 4     # --batch-size 0: subjects in the first batch, which measures the GPU memory one takes
AUTO_BATCH_FILL = 0.8    # ... the share of the GPU memory free at the start that a batch may fill
AUTO_BATCH_MAX = 64      # ... at most this many subjects per batch
AUTO_BATCH_GAIN = 0.1    # ... the batch size doubles while the alignment time per subject drops by more than this
CUDA_LOST_EXIT = 75      # run()'s exit code when the GPU context is lost; submit_alignment*.sh then start it again


def cuda_lost(device) -> bool:
    """Whether this process's CUDA context is unusable. After some CUDA errors (cudaErrorUnknown, an illegal memory
    access, a driver fault) every later GPU call fails the same way; an out-of-memory error leaves it usable."""
    if device.type != "cuda":
        return False
    try:
        torch.ones(1, device=device).add_(1).cpu()
        torch.cuda.synchronize(device)
        return False
    except Exception:  # noqa: BLE001
        return True
RESCAN_SECONDS = 30.0   # how often --root is rescanned for new subjects while others are still queued
SKIPPED_JOB = "__skipped__"   # load_job(): the subject was found complete (--skip-processed)


def write_run_config(out_dir: Path, args) -> None:
    """Record every resolved setting (defaults included) in run_config.json. If the run directory already holds a
    config with different result-affecting settings, warn and keep both (the new one gets a timestamp suffix)."""
    import json
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in sorted(vars(args).items())}
    config["command_line"] = " ".join(sys.argv)
    path = out_dir / "run_config.json"
    if path.exists():
        old = json.loads(path.read_text())
        changed = sorted(k for k in config if k not in NAME_EXCLUDED | {"command_line"} and old.get(k) != config[k])
        if not changed:
            return
        print(f"WARNING: {path} was written with different settings ({', '.join(changed)}); "
              f"results in this directory now mix configurations.")
        path = out_dir / f"run_config_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(config, indent=2))


def default_run_name(ap: argparse.ArgumentParser, args, argv: list[str]) -> str:
    """<YYYYMMDD-HHMMSS>_steps<N>-lr<x>, followed by every result-affecting option given on the command line (in --help
    order). The start-time prefix keeps reruns with the same settings from writing into an earlier run's directory."""
    def fmt(v):
        if isinstance(v, float):
            return f"{v:g}"
        return str(v).replace(",", "+").replace("/", "_").replace("\\", "_").replace(" ", "")

    given = {tok.split("=", 1)[0] for tok in argv if tok.startswith("--")}
    parts = [f"steps{args.steps}", f"lr{fmt(args.lr)}"]
    for action in ap._actions:
        if action.dest in NAME_EXCLUDED | {"steps", "lr", "help"} or not given.intersection(action.option_strings):
            continue
        name = short_option_name(action.option_strings[-1])
        value = getattr(args, action.dest)
        parts.append(name if action.nargs == 0 else f"{name}{fmt(value)}")
    name = "-".join(parts)
    if len(name) > MAX_RUN_NAME:                      # keep paths under the Windows 260-character limit
        import hashlib
        digest = hashlib.sha1(name.encode()).hexdigest()[:8]
        name = name[:MAX_RUN_NAME - 9].rstrip("-") + "-" + digest
    return f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{name}"


MAX_RUN_NAME = 120
NAME_ABBREVIATIONS = {"schedule": "sched", "dilation": "dil", "start": "st", "sigma": "sig", "rotation": "rot",
                      "metric": "", "mask": "mask", "scale": "sc", "stages": "stg", "frac": "fr", "extent": "ext"}


def short_option_name(option: str) -> str:
    """'--mask-dilation-start-mm' -> 'maskdilst': drop unit suffixes, abbreviate words, join without dashes."""
    words = [w for w in option.lstrip("-").split("-") if w not in ("mm", "deg")]
    return "".join(NAME_ABBREVIATIONS.get(w, w) for w in words)


def str2bool(s: str) -> bool:
    """argparse type for boolean options: true/false, yes/no, 1/0, on/off (type=bool would make 'false' True)."""
    v = s.strip().lower()
    if v in ("1", "true", "t", "yes", "y", "on"):
        return True
    if v in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected true or false, got {s!r}")


def build_parser(description: str) -> argparse.ArgumentParser:
    """Every option of the motion correction with its general default: paths, run options, the algorithm (optimizer,
    similarity, regularizers, ...), metrics and hardware. An entry script (moco_nifti.py, moco_dicom.py) adds the
    options of its data format and may repeat any option here with its own default and help -- the parser resolves
    conflicts, so the entry script's definition replaces this one (e.g. moco_dicom.py: --blur-frac 0.1) -- then calls
    run()."""
    ap = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 conflict_handler="resolve")
    # ###################    PATHS    ####################################################
    g = ap.add_argument_group("paths")
    g.add_argument("--root", type=Path, default=None,
                   help="Directory containing one folder per subject (the images anywhere below each). Default: %(default)s")
    g.add_argument("--out-root", type=Path, default=None,
                   help="Parent of the run directory. Default: %(default)s (None: <root>_aligned next to --root)")
    g.add_argument("--run-name", type=str, default=None,
                   help="Run directory name, used as given (pass an existing run's name to add to or resume it, e.g. with "
                        "--skip-processed). Default: the start time followed by the settings given on the command line, "
                        "so every run gets a new directory")

    # ###################    RUN PARAMS    ###############################################
    ap.add_argument("--max-subjects", type=int, default=-1,
                    help="Max number of subjects to process (-1 = all). Default: %(default)s")
    ap.add_argument("--start-index", type=int, default=0,
                    help="Process only the subject folders from this index on, in the list of subject folders under "
                         "--root sorted by name (0-based; with --end-index, to submit a cohort in chunks, e.g. one job "
                         "per 0-10000, 10000-20000, ...). Default: %(default)s")
    ap.add_argument("--end-index", type=int, default=-1,
                    help="Process only the subject folders before this index (exclusive) in the sorted list (-1 = to the "
                         "end). Default: %(default)s")
    ap.add_argument("--skip-processed", action="store_true",
                    help="Skip subjects whose aligned images are already complete in the run directory (to resume a run, "
                         "pass its directory name with --run-name; a default name is always new)")
    ap.add_argument("--logging", type=str2bool, default=True, metavar="BOOL",
                    help="true (default): write everything besides the aligned images: the pose/metric history during "
                         f"optimization (every {HISTORY_EVERY:g} of the run), per-subject CSVs and figures (metrics/), "
                         "cohort-wide CSVs and figures (overall_metrics/) and run_config.json. false: only the "
                         "motion-corrected images are written and no metrics are computed (fastest). "
                         "overall_metrics/sax_relative_motion.csv and run_config.json are written either way.")
    ap.add_argument("--uint8-percentile", type=float, default=99.9,
                    help="Clip percentile for --image-dtype uint8: per subject, intensities above this percentile of all "
                         "its slices and frames are clipped (99.9 clips 0.1%% of the voxels). Default: %(default)s")
    ap.add_argument("--n-frames", type=int, default=10,
                    help="Number of cardiac phases (evenly spaced) used for the alignment (at most the fewest frames of a "
                         "multi-frame slice). A slice with a single frame is taken as end-diastole, the first phase: "
                         "its intersections are compared at that phase only. Default: %(default)s")
    ap.add_argument("--save-images", type=str2bool, default=True, metavar="BOOL",
                    help="false: do not write the motion-corrected images (e.g. to only estimate the motion of a cohort "
                         "with --logging false: then just sax_relative_motion.csv is written)")
    ap.add_argument("--step-progress", action="store_true",
                    help="Show a second progress bar over the optimizer steps of the current subject, below the one over "
                         "the subjects (needs tqdm)")

    # ###################    OPTIMIZER PARAMS    #########################################
    ap.add_argument("--steps", type=int, default=1000, help="Optimizer steps")
    ap.add_argument("--optimizer", choices=OPTIMIZERS, default="sgd",
                    help="sgd: plain gradient steps (see --momentum, --nesterov); its steps scale with the gradient, "
                         "so it needs much larger learning rates (see --lr). SGD lead consistently higher performance. "
                         "adam: the default. nadam: Adam with Nesterov momentum (looks ahead, often a little faster). "
                         "radam: Adam with the variance rectified during the first steps (no warm-up needed, steadier start). "
                         "adamax: Adam with an infinity-norm denominator (steps bounded by lr, robust to gradient spikes). "
                         "rmsprop: per-parameter step scaling without momentum. "
                         "adagrad: step sizes shrink like 1/sqrt(step), a built-in annealing. "
                         "See default: %(default)s for per-optimizer default learning rates.")
    ap.add_argument("--lr", type=float, default=None,
                    help="Learning rate for translations. For the adaptive optimizers roughly the step in mm per "
                         "iteration; for sgd the step is lr x gradient, and the NCC objective's gradients are ~1e-3 per "
                         "mm. Default: per optimizer (adam/nadam/radam/adamax/rmsprop 0.2, adagrad 1.0, sgd 80)")
    ap.add_argument("--rot-lr", type=float, default=None,
                    help="Learning rate for rotations (degrees; see --lr). Default: per optimizer "
                         "(adam/nadam/radam/adamax/rmsprop 0.04, adagrad 0.2, sgd 20)")
    ap.add_argument("--momentum", type=float, default=0.0, help="Momentum for --optimizer sgd (0 = plain SGD). Default: %(default)s")
    ap.add_argument("--nesterov", action="store_true", help="Nesterov momentum for --optimizer sgd")
    ap.add_argument("--lr-schedule", choices=["linear", "constant"], default="linear",
                    help="linear: decay each learning rate linearly from its start value to its end value over the run "
                         "(the line-dropping noise of --pair-keep averages out, so the poses settle where the best pose "
                         "is tracked); constant: no decay. Default: %(default)s")
    ap.add_argument("--lr-end", type=float, default=None,
                    help="Final translation learning rate for the linear schedule. Default: 50%% of --lr")
    ap.add_argument("--rot-lr-end", type=float, default=None,
                    help="Final rotation learning rate for the linear schedule. Default: 50%% of --rot-lr")
    ap.add_argument("--pair-keep", type=float, default=0.8,
                    help="Stochastic optimization: fraction of intersection lines used at each step (1 = deterministic). "
                         "Each step draws a new random subset of round(fraction x lines) lines and samples only those, "
                         "so GPU memory and time per step scale with this fraction (the full objective for tracking "
                         "the best pose is evaluated without gradients, in chunks)")
    ap.add_argument("--eval-every", type=int, default=10, help="With --pair-keep < 1: evaluate the full objective every N steps to track the best pose")
    ap.add_argument("--seed", type=int, default=0, help="Seed for the line dropping")

    # ###################    IMAGES    ##################################################
    ap.add_argument("--use-ao", type=str2bool, default=True, metavar="BOOL",
                    help="Use aortic slices when present (UKBB ao.nii.gz: one axial slice through the ascending aorta, "
                         "no segmentation) as LAX views: SAX-LAX lines with the SAX slices, LAX-LAX lines with the "
                         "other LAX views, part of the LAX reference frame and statistics; never used for the LV long "
                         "axis. Its LAX-LAX lines are masked by the other view's dilated heart mask, its SAX-LAX lines "
                         "by the SAX-SAX cylinder (within --sax-sax-radius-mm of the LV long axis). false: left out of the alignment (written with its acquired geometry). Subjects "
                         "without one are unaffected. Default: %(default)s")

    # ###################    POSE MODEL PARAMS    ########################################
    ap.add_argument("--dof", type=int, choices=[3, 4, 6], default=6,
                    help="Degrees of freedom per slice. 3: translation only. 4: translation + rotation about the slice's "
                         "own normal (in-plane rotation; no tilts). 6: rigid (translation + 3D rotation) where the "
                         "geometry constrains the tilts (>= 3 images, one of them a stack), otherwise automatically "
                         "reduced to 4 (see pose_model; summary column pose_model). Default: %(default)s")
    ap.add_argument("--rotation-pivot", choices=["long-axis", "image-centre"], default="long-axis",
                    help="Point each plane's pose rotates about. long-axis: on the LV long axis (the crossing line of the "
                         "two most perpendicular LAX views) -- for a SAX slice where the axis pierces it, for a LAX view "
                         "the heart centre projected onto it -- so rotations turn the plane about the heart and the "
                         "translation is the heart's displacement (decoupled). image-centre: the centre of the pixel grid "
                         "(the earlier behaviour; a rotation then also moves the heart, entangling it with the "
                         "translation). The reported dx/dy/dz are the displacement of this point. Default: %(default)s")

    # ###################    SIMILARITY METRIC PARAMS    #################################
    ap.add_argument("--metric", choices=["ncc", "ncc_grad", "lncc", "ngf", "nmi", "l1", "l2"], default="lncc",
                    help="Optimized similarity along each line. ncc: normalized cross-correlation (one high-contrast edge can "
                         "carry most of the score). ncc_grad: NCC of the profiles' 1D gradients. lncc: local NCC in windows "
                         "of --lncc-window-mm sliding along the line, averaged (every stretch has to match on its own; "
                         "flat windows fade out, --lncc-min-std). ngf: normalized gradient agreement -- per-point "
                         "normalized gradients, so every edge counts about equally whatever its contrast (--grad-eps), "
                         "in [-1, 1]. nmi: normalized mutual information. l1 / l2: negative mean absolute / squared "
                         "intensity difference, every image first rescaled so its 1st/99th percentiles map to 0/1 (the "
                         "SAX slices with one common scale). Used between slices of the same modality (SliceImage.modality); "
                         "see --cross-mod-metric. Default: %(default)s")
    ap.add_argument("--cross-mod-metric", choices=["ncc", "ncc_grad", "lncc", "ngf", "nmi", "l1", "l2"], default="nmi",
                    help="The similarity between slices of different modalities (e.g. cine and LGE, whose contrasts "
                         "differ): their intersection lines and their parallel pairs (the crossing term of "
                         "--sax-sax-weight and --coplanar-weight). Choices as --metric. Default: %(default)s")
    ap.add_argument("--lncc-window-mm", type=float, default=20.0, help="--metric lncc: window length (mm). Default: %(default)s")
    ap.add_argument("--lncc-min-std", type=float, default=0.1,
                    help="--metric lncc: windows whose profiles vary less than this (geometric mean of both standard "
                         "deviations, in units of the profile's standard deviation over the line) are faded out "
                         "(weight sd / this). Default: %(default)s")
    ap.add_argument("--grad-eps", type=float, default=0.5,
                    help="--metric ngf: edge threshold, in standard deviations of the profile's gradient along the line; "
                         "gradients well above it count as full edges, those below it fade out. Default: %(default)s")
    ap.add_argument("--edge-weight", type=float, default=0.0,
                    help="Late edge term (0 = off): toward the end of the run, the similarity of each line within a "
                         "modality becomes (1 - w) x --metric + w x --edge-metric, w ramping from 0 to this value (0-1) "
                         "over --edge-ramp-frac of the run from --edge-start-frac. The edge metrics compare the signed "
                         "derivatives of both profiles along their shared line (the same 3D directional derivative in "
                         "both views), so narrow, high-contrast edges such as the myocardial borders count more than "
                         "long stretches with slight variation, which --metric lncc weighs about equally. Their capture "
                         "range is about the blur width, hence late. Lines between modalities keep --cross-mod-metric. "
                         "The best pose is tracked once the ramp is over. Default: %(default)s")
    ap.add_argument("--edge-metric", choices=["ncc_grad", "ngf"], default="ncc_grad",
                    help="The edge term of --edge-weight (also reported as 'edge' with --logging). ncc_grad: NCC of the "
                         "profiles' 1D gradients along the line, weighted by edge strength (strong edges dominate). ngf: "
                         "normalized gradient agreement (every edge above --grad-eps counts about equally). "
                         "Default: %(default)s")
    ap.add_argument("--edge-start-frac", type=float, default=0.8,
                    help="--edge-weight: fraction of the run at which the edge term starts ramping up (by default when "
                         "the blur and the mask dilation have reached their final values). Default: %(default)s")
    ap.add_argument("--edge-ramp-frac", type=float, default=0.1,
                    help="--edge-weight: fraction of the run over which its weight ramps up linearly (0 = at once). "
                         "Default: %(default)s")
    ap.add_argument("--nmi-bins", type=int, default=24, help="Histogram bins for mutual information")
    ap.add_argument("--nmi-sigma", type=float, default=1.0, help="Parzen window width for mutual information, in bins")

    # ###################    COARSE-TO-FINE (BLUR) PARAMS    #############################
    ap.add_argument("--blur-sigma", type=float, default=5.0,
                    help="Coarse-to-fine: initial in-plane Gaussian sigma (mm) of the images, decayed linearly to --blur-sigma-end "
                         "over --blur-frac of the run (0 = off). Default: %(default)s")
    ap.add_argument("--blur-frac", type=float, default=0.8,
                    help="Fraction of the steps over which the blur decays to --blur-sigma-end (> 1: it never gets "
                         "there); 0 = no decay, the blur stays at --blur-sigma for the whole run")
    ap.add_argument("--blur-sigma-end", type=float, default=2.0,
                    help="Blur sigma (mm) the coarse-to-fine blur decays to (linearly from --blur-sigma over "
                         "--blur-frac of the run) and then keeps until the end (0: unblurred at the end). Default: %(default)s")

    # ###################    INTERSECTION LINE AND MASK PARAMS    ########################
    ap.add_argument("--line-step-mm", type=float, default=1.8,
                    help="Spacing (mm) between samples along each intersection line. Default: %(default)s")
    ap.add_argument("--line-extent-mm", type=float, default=160.0,
                    help="Half-length (mm) of each intersection line: samples run from -extent to +extent around a "
                         "point near the plane centre, then are clipped to the images and the mask. Default: %(default)s")
    ap.add_argument("--min-line-mm", type=float, default=40.0,
                    help="An intersection line is dropped if fewer than this many mm of it lie inside both images and "
                         "the mask. Default: %(default)s")
    ap.add_argument("--min-pair-angle-deg", type=float, default=20.0,
                    help="Two slices form an intersection pair only if their acquired normals differ by at least this "
                         "angle: slices of one stack (parallel) never pair, and nearly parallel slices, whose crossing "
                         "line is ill-conditioned, do not either (UKBB: SAX-LAX ~90 deg, LAX-LAX ~60 deg). "
                         "Default: %(default)s")
    ap.add_argument("--pair-spacing-mm", type=float, default=-1.0,
                    help="Thinning of the pairs between two multi-slice stacks (see candidate_pairs): each slice pairs "
                         "with slices about this far apart in the other stack, phase-shifted between neighbours so every "
                         "slice keeps lines. -1 (default) = each stack's slice thickness (thick, densely spaced slices "
                         "overlap, so their lines are nearly redundant); 0 = keep every pair. Single slices (e.g. the "
                         "UKBB LAX views) are never thinned, and with the UKBB 10 mm SAX spacing nothing is thinned.")
    ap.add_argument("--mask-dilation-mm", type=float, default=20.0,
                    help="How far (mm) the LAX heart segmentation is grown to form the mask; only intersection-line points "
                         "inside this mask enter the loss (see --outside-mask-weight). This is also the final value when "
                         "mask annealing is on. Default: %(default)s")
    ap.add_argument("--outside-mask-weight", type=float, default=0.0,
                    help="Loss weight of intersection-line points outside the dilated mask (points inside it weigh 1). "
                         "0 = they are excluded (default); e.g. 0.1 = the lines run through the whole image overlap and "
                         "the points outside the mask count 10x less. Which lines are kept (--min-line-mm), the line "
                         "weights and all reported metrics still use the mask points only. Default: %(default)s")
    ap.add_argument("--mask-dilation-start-mm", type=float, default=80,
                    help="Mask annealing (off unless given). The dilation (mm) used at the start of the run; it then "
                         "shrinks linearly to --mask-dilation-mm over the first --mask-anneal-frac of the run. "
                         "Example: '--mask-dilation-start-mm 40 --mask-dilation-mm 25' starts with a 40 mm mask and "
                         "ends at 25 mm.")
    ap.add_argument("--mask-anneal-frac", type=float, default=0.8,
                    help="Fraction of the run (0-1) over which the mask dilation goes from --mask-dilation-start-mm to "
                         "--mask-dilation-mm. Default: %(default)s")
    ap.add_argument("--anneal-stages", type=int, default=6,
                    help="Mask annealing does not change every step: the dilation advances this many times along its "
                         "linear ramp, and the lines are rebuilt each time. Once it has finished, the final dilation is "
                         "used for the rest of the run and the best pose is chosen only from that part. "
                         "Default: %(default)s")

    # ###################    SLICE PROFILE (PSF) PARAMS    ###############################
    ap.add_argument("--n-band", type=int, default=8,
                    help="PSF approximation: when sampling one image along an intersection line, this many points are "
                         "taken across the other slice's through-plane profile (see --psf-model) and combined with the "
                         "profile as weights; the band is widened by 1/sin of the crossing angle. 1 = sample the line "
                         "only. Default: %(default)s")
    ap.add_argument("--psf-model", choices=["pulse", "gaussian", "rect"], default="pulse",
                    help="Shape of a slice's through-plane excitation profile, with FWHM = the nominal slice thickness. "
                         "'pulse': the profile of a slice-selective RF pulse of finite time-bandwidth product "
                         "(--psf-tbw) -- a rectangle of the nominal thickness with smoothed edges, sampled out to where "
                         "the profile vanishes; 'gaussian': a Gaussian (sampled to +-2 sigma); 'rect': an ideal box. "
                         "Default: %(default)s")
    ap.add_argument("--psf-tbw", type=float, default=2.5,
                    help="Time-bandwidth product of the slice-selective RF pulse for --psf-model pulse. It sets the "
                         "edge sharpness: the 10-90%% transition width is about thickness / TBW. Short bSSFP pulses "
                         "(TR ~2.7 ms as in UK Biobank) have TBW of roughly 2-3; the exact vendor pulse is not "
                         "published. Default: %(default)s")
    ap.add_argument("--psf-scale-start", type=float, default=3.0,
                    help="PSF annealing (off at 1). The PSF band (through-plane slice profile, see --psf-model) starts "
                         "this many times wider than physical and shrinks linearly to the physical width over the first "
                         "--psf-anneal-frac of the run. Unlike --blur-sigma (isotropic in-plane blur of both images), "
                         "this smooths each image only perpendicular to each intersection line. More band samples are "
                         "used so the nominal density of --n-band is kept. Default: %(default)s")
    ap.add_argument("--psf-anneal-frac", type=float, default=0.9,
                    help="Fraction of the run over which --psf-scale-start decays to 1. Default: %(default)s")
    ap.add_argument("--no-psf", action="store_true",
                    help="Disable the PSF model: each image is sampled (bilinearly) only at the intersection-line points "
                         "themselves, i.e. its raw intensities, with no band across the other slice's profile. Overrides "
                         "--n-band (set to 1) and --psf-scale-start (set to 1, no PSF annealing); --psf-model and "
                         "--psf-tbw have no effect. In-plane --blur-sigma annealing still applies if set.")
    ap.add_argument("--coarse-frac", type=float, default=0.5,
                    help="Fraction of the run (from the start) sampled coarsely: only every --coarse-frame-stride-th "
                         "cine frame and every --coarse-point-stride-th line point, which makes those steps several "
                         "times cheaper. The early steps make the large moves, on blurred images and dilated masks; the "
                         "best pose is only tracked after it. 0 = off. Default: %(default)s")
    ap.add_argument("--coarse-frame-stride", type=int, default=4,
                    help="--coarse-frac: use every n-th cine frame (UKBB: 50 -> 13 frames at 4). Default: %(default)s")
    ap.add_argument("--coarse-point-stride", type=int, default=2,
                    help="--coarse-frac: use every n-th intersection-line point (--line-step-mm x n apart). "
                         "Default: %(default)s")

    # ###################    POSE REGULARIZATION PARAMS    ###############################
    ap.add_argument("--gauge", type=float, default=1e-2, help="Penalty weight on the mean pose (removes global drift)")
    ap.add_argument("--prior", type=float, default=0.0, help="Penalty weight on rotation (deg^2) and through-plane shift (mm^2)")
    ap.add_argument("--tilt-prior", type=float, default=0.0,
                    help="Penalty weight on each plane's own tilt (rotation about its in-plane axes, as rot_tilt_deg in the "
                         "outputs) above --tilt-hinge-deg: relu(tilt - threshold)^2 (deg^2), mean over planes. In-plane "
                         "rotation and translations are not penalized; tilts up to the threshold cost nothing (0 = off)")
    ap.add_argument("--shift-prior", type=float, default=0.01,
                    help="Penalty weight on each plane's shift from its acquired pose beyond --shift-free-mm: "
                         "relu(|t| - free)^2 (mm^2, |t| the Euclidean norm of the plane's translation), mean over planes. "
                         "Shifts within the free zone cost nothing, so ordinary breath-hold corrections are not biased "
                         "toward zero; a plane drifting out of the cluster is pulled back (its lines go flat once they "
                         "leave the image overlap, so the image term alone cannot). With SGD a plane's step from it is "
                         "lr x 2 w / n_planes x the excess, so keep w well below n_planes / lr (e.g. 0.01 at lr 80) "
                         "(0 = off). Default: %(default)s")
    ap.add_argument("--shift-free-mm", type=float, default=20.0,
                    help="Shift (mm) from the acquired pose allowed without cost by --shift-prior. Default: %(default)s")
    ap.add_argument("--rot-prior", type=float, default=0.01,
                    help="Penalty weight on each plane's rotation angle from its acquired pose beyond --rot-free-deg: "
                         "relu(angle - free)^2 (deg^2), mean over planes; in-plane rotation and tilt alike (see "
                         "--tilt-prior for tilt only) (0 = off). Default: %(default)s")
    ap.add_argument("--rot-free-deg", type=float, default=10.0,
                    help="Rotation (deg) from the acquired pose allowed without cost by --rot-prior. Default: %(default)s")
    ap.add_argument("--max-step-mm", type=float, default=2.0,
                    help="Largest translation of a plane per optimizer step (mm, Euclidean): a longer step is shortened "
                         "to it, keeping its direction. Keeps one large gradient from throwing a plane out of the image "
                         "overlap (e.g. 2; 0 = off). Default: %(default)s")
    ap.add_argument("--max-step-deg", type=float, default=1.0,
                    help="Largest rotation of a plane per optimizer step (deg; change of its rotation vector) (e.g. 1; "
                         "0 = off). Default: %(default)s")
    ap.add_argument("--apex-extend-mm", type=float, default=20.0,
                    help="Extend each segmented LAX view's heart mask toward the apex by this much (mm) before it is "
                         "dilated (--mask-dilation-mm, annealed): the heart region is swept along the in-plane direction "
                         "of the LV long axis (base to apex: the SAX stack, oriented by the LAX segmentations: atria at "
                         "the base, LV toward the apex). Keeps the apical SAX slices' lines supervised when the mask shrinks, also where the "
                         "apical segmentation is incomplete (0 = off; subjects without SAX slices are "
                         "unaffected). Default: %(default)s")
    ap.add_argument("--max-shift-mm", type=float, default=50.0,
                    help="Hard limit on each plane's translation from its acquired pose (mm, Euclidean): after every "
                         "step a plane beyond it is projected back onto it. Unlike --shift-prior it cannot be overcome by "
                         "the image term (e.g. 50: far beyond real breath-hold motion, UKBB 99.9th percentile ~23 mm; "
                         "0 = off). Default: %(default)s")
    ap.add_argument("--max-rot-deg", type=float, default=20.0,
                    help="Hard limit on each plane's rotation angle from its acquired pose (deg) (e.g. 20; UKBB 99.9th "
                         "percentile ~8 deg; 0 = off). Default: %(default)s")
    ap.add_argument("--tilt-hinge-reg", type=float, default=0.0,
                    help="Penalty weight on the relative tilt of adjacent SAX slices above --tilt-hinge-deg: "
                         "relu(angle between their normals - threshold)^2 (deg^2), mean over adjacent pairs. Ordinary "
                         "breath-hold tilts cost nothing; outlier orientations are pulled back (0 = off; e.g. 1e-3 "
                         "costs as much as 0.025 NCC at 5 deg above the threshold). Default: %(default)s")
    ap.add_argument("--tilt-hinge-deg", type=float, default=5.0,
                    help="Tilt (deg) allowed without cost: the relative tilt of adjacent SAX slices for --tilt-hinge-reg, "
                         "and each plane's own tilt for --tilt-prior. Default: %(default)s")

    # ###################    SAX STACK REGULARIZATION PARAMS    ##########################
    ap.add_argument("--spacing-reg", type=float, default=0.0,
                    help="Penalty weight on (distance - --sax-spacing-mm)^2 (mm^2) of adjacent SAX slices, the distance "
                         "between their centres measured along their mean normal, in the acquired stacking direction; "
                         "mean over adjacent pairs (0 = off)")
    ap.add_argument("--sax-spacing-mm", type=float, default=10.0,
                    help="Target distance of adjacent SAX slices for --spacing-reg (UKBB: 10 mm; 0 = each pair's acquired spacing)")
    ap.add_argument("--rel-shift-reg", type=float, default=0.0,
                    help="Penalty weight on the in-plane offset of adjacent SAX slices relative to each other (mm^2): the "
                         "posed centre of one slice, seen from its neighbour's own frame, should sit where it was acquired "
                         "(directly below/above it), so a shifted slice pulls its neighbours along. Only the in-plane part; "
                         "the part along the normal is --spacing-reg. Mean over adjacent pairs (0 = off)")
    ap.add_argument("--rel-rot-reg", type=float, default=0.0,
                    help="Penalty weight on the relative in-plane rotation (about the normal) of adjacent SAX slices "
                         "(~deg^2), so a rotated slice pulls its neighbours' rotation along. The relative tilt is "
                         "--tilt-hinge-reg. Mean over adjacent pairs (0 = off)")

    # ###################    NEIGHBOURING-SLICE INTERACTION PARAMS    ###################
    # (named after the SAX stack they were made for; they act on the neighbouring slices of every stack: the SAX slices
    # and every other image of several slices, e.g. the oblique DICOM stacks -- see attach_sax_pairs)
    ap.add_argument("--parallel-max-dist-mm", type=float, default=12.0,
                    help="Parallel pairs for the terms below (and their QC logging): slices parallel as acquired "
                         "(closer than --min-pair-angle-deg) and at most this far apart along the normal, whatever their "
                         "views and modalities: in every stack (SAX, or any image of several slices) each slice and the "
                         "next one along the stack normal (so a gap in the stack is not bridged; UKBB, 10 mm spacing: z "
                         "and z + 1), and between two images each slice and the nearest parallel slice of the other "
                         "image that overlaps it in-plane (e.g. the same view in two modalities). Default: %(default)s")
    ap.add_argument("--sax-gap-reg", type=float, default=0.0,
                    help="Barrier keeping neighbouring slices of a stack from approaching or crossing each other inside "
                         "the heart: on a disk of --sax-gap-radius-mm around an axis on each slice (SAX: the LV long "
                         "axis, the crossing line of the two most perpendicular LAX views, else the SAX stack's axis; "
                         "other stacks: their own axis, through the stack centre along its normal), the gap to the "
                         "neighbouring slice along the acquired stacking direction (~10 mm as acquired for UKBB SAX) is "
                         "penalized below the margin (--sax-gap-margin-mm, at most half the pair's acquired gap): mean "
                         "softplus(margin - gap)^2 (mm^2). Zero for ordinary poses, smooth, no image sampling (0 = off; "
                         "e.g. 1e-2 costs 0.16 at a 0 mm gap). Default: %(default)s")
    ap.add_argument("--sax-gap-margin-mm", type=float, default=4.0,
                    help="Smallest gap (mm) between neighbouring slices allowed without cost by --sax-gap-reg (for "
                         "each pair at most half its acquired gap). Default: %(default)s")
    ap.add_argument("--sax-gap-radius-mm", type=float, default=120.0,
                    help="Radius (mm) of the disk around the axis on which --sax-gap-reg measures the gap (5 mm grid). "
                         "Default: %(default)s")
    ap.add_argument("--coplanar-weight", type=float, default=0.0,
                    help="Supervise near-coplanar slices: parallel pairs (closer in angle than --min-pair-angle-deg, within "
                         "--parallel-max-dist-mm) whose slabs overlap (less than half their summed thicknesses apart), "
                         "e.g. the same view in two modalities, of any view. Their images are compared over the area they "
                         "share (a grid every --coplanar-step-mm inside the heart region; --metric within a modality, "
                         "--cross-mod-metric between modalities), and this weight x the mean score is added to the "
                         "objective (see CoplanarPairs). 0 = off: they are only treated as parallel pairs. Default: "
                         "%(default)s")
    ap.add_argument("--coplanar-step-mm", type=float, default=2.0,
                    help="--coplanar-weight: spacing (mm) of the grid of compared points. Default: %(default)s")
    ap.add_argument("--sax-sax-weight", type=float, default=0.0,
                    help="Dissimilarity cost of parallel pairs (--parallel-max-dist-mm: the slice above / below in a "
                         "stack, the nearest parallel slice of another image) crossing each other: where the poses make "
                         "them cross, their images are compared along the crossing line with NCC (between modalities: "
                         "--cross-mod-metric) and weight x sum(point weights x (1 - NCC)) / (number of intersection-line points) is "
                         "subtracted from the objective, so a crossing never pays and a mismatched one costs most. The "
                         "point weights fall off with the distance from the axis (SAX: LV long axis; other stacks: their "
                         "own axis; --sax-sax-falloff, --sax-sax-radius-mm); only points inside both images count. The "
                         "line is recomputed from the poses every step. Slices that stay parallel do not cross and get "
                         "no cost; neighbours d mm apart cross d / tan(relative tilt) from the tilt pivot, so with densely "
                         "spaced slices (e.g. 2 mm apart, 240 mm field of view) a relative tilt of ~1 deg already brings "
                         "the crossing into the image. Needs rotations (--dof 6; "
                         "0 = off). Default: %(default)s")
    ap.add_argument("--sax-sax-falloff", choices=["gaussian", "threshold"], default="gaussian",
                    help="Point weights of --sax-sax-weight along the crossing line. gaussian: exp(-r^2 / (2 sigma^2)) of the "
                         "distance r from the axis, sigma = --sax-sax-radius-mm; smooth, so a mismatched crossing is "
                         "pushed out of the heart. threshold: 1 within --sax-sax-radius-mm, 0 beyond (no such push). "
                         "Default: %(default)s")
    ap.add_argument("--sax-sax-radius-mm", type=float, default=120.0,
                    help="--sax-sax-falloff "
                         "gaussian: its sigma (mm); "
                         "threshold: the radius (mm) around the axis. Also "
                         "the extent of the 'used' part of the lines in sax_lines.csv, and the radius of the cylinder "
                         "around the LV long axis that masks the lines of an aortic slice with the SAX slices (always "
                         "a hard cut, like a heart mask). Default: %(default)s")

    # ###################    SEGMENTATION METRICS (REPORTING ONLY, WITH --logging)    ####
    g = ap.add_argument_group("segmentation metrics (reporting only, with --logging; images with segmentations)")
    g.add_argument("--seg-labels", type=str, default="1,2,3,4,5",
                   help="Labels compared (Dice / Hausdorff) along intersections of two non-SAX images. Default: %(default)s")
    g.add_argument("--sax-seg-labels", type=str, default="1,2,3",
                   help="Labels compared along SAX-LAX intersections (the UKBB SAX model has fewer classes). "
                        "Default: %(default)s")
    g.add_argument("--hd-miss-mm", type=float, default=100.0,
                   help="Hausdorff value (mm) when both planes contain a label but only one intersection profile hits "
                        "it. Default: %(default)s")

    # ###################    HARDWARE PARAMS    ##########################################
    g = ap.add_argument_group("hardware")
    g.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                   help="Device for the alignment. Default: %(default)s")
    g.add_argument("--workers", type=int, default=4,
                   help="Worker processes for the CPU work around the alignment: loading and preparing subjects "
                        "(masks, distance maps), writing the aligned images and the per-subject metrics and figures. "
                        "Up to about the number of CPU cores (each holds a few subjects in memory). Default: %(default)s")
    g.add_argument("--batch-size", type=int, default=0,
                   help="Subjects aligned together in one optimization (align_batch): their planes and lines share the "
                        "samplers, so each step samples all of them at once, while every term stays per subject (the "
                        "result of each is that of a solo run, up to floating-point rounding). Raises GPU utilization "
                        "and throughput; GPU memory grows about linearly with it. Not with --sax-sax-weight / "
                        "--sax-gap-reg (then 1). 0 = automatic: the first batch (4 subjects) measures the GPU memory a "
                        "subject takes, which bounds the batch size (80%% of the GPU memory free at the start, at most "
                        "64); from there it doubles while that makes the alignment per subject >10%% faster, then "
                        "stays (a GPU gains nothing from batches beyond the size that saturates it). A batch that "
                        "still runs out of memory is split in halves, and later batches shrink. On the CPU, or with "
                        "the terms above, 0 means 1. Default: %(default)s")
    g.add_argument("--no-cuda-graph", action="store_true",
                   help="Run each optimization step eagerly instead of replaying a captured CUDA graph (same result, "
                        "slower; for debugging)")
    g.add_argument("--sample-dtype", choices=["float32", "float16", "bfloat16"], default="float32",
                   help="Precision of the images held on the GPU and of their sampling along the intersection lines (the "
                        "bulk of the memory for many slices): float16 halves the image memory and the sampled tensors; it "
                        "holds integer intensities up to 2048 exactly (brighter images are scaled by a power of two; the "
                        "metrics are scale-invariant) and resolves sample positions to ~0.05 pixel on a 200-pixel image. "
                        "bfloat16 only to ~0.4 pixel and ~0.4%% in intensity (coarse for sub-mm motion). Poses, line "
                        "geometry and metrics stay float32; the saved images are unaffected (see --image-dtype). "
                        "Default: %(default)s")
    return ap



def run(ap: argparse.ArgumentParser, args, load, save, is_complete) -> tuple[Path, list[str]]:
    """Align every subject folder under args.root, writing to args.out_root / (args.run_name or a generated name)
    (see the module docstring); args come from build_parser (out_root None: <root>_aligned next to args.root).
    Returns the run directory and the subject folders found under args.root (for an entry point's own outputs on
    top; with --skip-processed also the skipped ones).

    The data loader supplies:
      load(subj_dir, args) -> Subject              read one subject folder (called in background threads)
      save(subject, affines, out_dir, args)        write the subject's slices with the given 4x4 affines (one per
                                                   slice, in subject.slices order) into out_dir, in its own format
      is_complete(subj_dir, out_dir) -> bool       whether out_dir already holds every output (--skip-processed)
    """
    if args.no_psf:                          # one band sample at offset 0 with weight 1 = the raw line intensities
        args.n_band, args.psf_scale_start = 1, 1.0
    if not 0.0 <= args.edge_weight <= 1.0:
        sys.exit(f"--edge-weight must lie in [0, 1] (got {args.edge_weight})")
    if args.edge_weight > 0 and (args.edge_ramp_frac < 0 or args.edge_start_frac + args.edge_ramp_frac > 1.0 + 1e-9):
        sys.exit("--edge-start-frac + --edge-ramp-frac must not exceed 1 (the edge term must reach its weight within "
                 "the run, or no pose is tracked after it)")
    lr0, rot_lr0 = DEFAULT_LR[args.optimizer]
    args.lr = lr0 if args.lr is None else args.lr
    args.rot_lr = rot_lr0 if args.rot_lr is None else args.rot_lr
    args.lr_end = LR_END_FRACTION * args.lr if getattr(args, "lr_end", None) is None else args.lr_end
    args.rot_lr_end = LR_END_FRACTION * args.rot_lr if getattr(args, "rot_lr_end", None) is None else args.rot_lr_end
    args.history_every = HISTORY_EVERY if args.logging else 0.0     # pose snapshots during optimization
    if args.root is None:
        sys.exit("No input directory: pass --root")
    if args.out_root is None:
        args.out_root = args.root.parent / f"{args.root.name}_aligned"

    run_dir = args.out_root / (args.run_name or default_run_name(ap, args, sys.argv[1:]))
    overall_dir = run_dir / "overall_metrics"
    summary_path = overall_dir / "motion_summary.csv"
    detail_path = overall_dir / "motion_summary_slices.csv"
    pairs_path = overall_dir / "motion_summary_pairs.csv"
    regions_path = overall_dir / "motion_summary_regions.csv"

    device = torch.device(args.device)
    seen: set[str] = set()
    budget = args.max_subjects                            # subjects still allowed by --max-subjects (-1 = no limit)
    n_skipped = 0

    start, end = max(0, getattr(args, "start_index", 0)), getattr(args, "end_index", -1)

    def scan() -> list[Path]:
        """Subject folders under --root not listed before, within --start-index / --end-index of all subject folders
        sorted by name. Hidden folders are ignored (a tool copying subjects in can write them as .<subject> and rename
        them when complete, so a subject is only picked up once fully written). Only lists folders: whether a subject
        is already complete (--skip-processed) is checked when it comes up, in the loader threads (load_or_skip), so a
        large --root starts aligning at once instead of checking every subject first, without any output (hours for a
        full UK Biobank folder on a network filesystem)."""
        folders = sorted(d for d in args.root.iterdir() if d.is_dir() and not d.name.startswith("."))
        chunk = folders[start:len(folders) if end < 0 else end]
        new = [d for d in chunk if d.name not in seen]
        if not seen and chunk and (start > 0 or end >= 0):     # first scan of a chunk: report it
            print(f"Subjects {start} to {start + len(chunk) - 1} of {len(folders)} (sorted by name): "
                  f"{chunk[0].name} ... {chunk[-1].name}", flush=True)
        seen.update(d.name for d in new)
        return new

    print(f"Scanning {args.root} for subject folders ...", flush=True)
    subjects = scan()
    print(f"Found {len(subjects)} subject folder(s); aligning on {device}, outputs in {run_dir}"
          + (" (--skip-processed: subjects already complete are skipped as they come up)" if args.skip_processed else "")
          + ("" if args.logging else " (aligned images only; --logging for metrics, CSVs and figures)"), flush=True)
    if not subjects:
        return run_dir, []
    run_dir.mkdir(parents=True, exist_ok=True)
    overall_dir.mkdir(parents=True, exist_ok=True)
    write_run_config(overall_dir, args)
    relmotion_f, relmotion_writer = open_csv(overall_dir / "sax_relative_motion.csv", RELMOTION_COLUMNS)

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda *a, **k: None  # noqa: E731

    writers = None
    if args.logging:
        writers = [open_csv(summary_path, SUMMARY_COLUMNS), open_csv(detail_path, DETAIL_COLUMNS), open_csv(pairs_path, PAIR_COLUMNS),
                   open_csv(regions_path, REGION_COLUMNS)]
    save_jobs = deque()                                   # (subject, description, future) for background writes
    failed, unaligned = [], []
    try:
        # worker processes (not threads: loading, preparing, gzip-writing and plotting are CPU work that threads
        # would serialize on the interpreter lock), started fresh ("spawn": never forked from a process using CUDA)
        n_workers = max(1, min(args.workers, 61) if os.name == "nt" else args.workers)
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=_worker_init) as ex:
            it = iter(subjects)
            queue = deque()
            capped = args.max_subjects >= 0                   # total: the folders found, or --max-subjects
            pbar = tqdm(total=min(len(subjects), args.max_subjects) if capped else len(subjects), unit="subj")
            last_scan = time.time()

            # --batch-size 0: the first batch (AUTO_BATCH_PROBE subjects) measures the GPU memory a subject takes,
            # which bounds the batch size (AUTO_BATCH_FILL of the memory free now); from there the batch size doubles
            # as long as that speeds up the alignment per subject (by AUTO_BATCH_GAIN), else it goes back and stays.
            # Beyond the size that saturates the GPU, a larger batch only costs memory and a longer last batch.
            auto_batch = getattr(args, "batch_size", 1) == 0 and device.type == "cuda" and batchable(args)
            batch_size = AUTO_BATCH_PROBE if auto_batch else max(1, getattr(args, "batch_size", 1))
            free0 = torch.cuda.mem_get_info(device)[0] if auto_batch else 0
            probing, tuning, mem_max, tune_best = auto_batch, auto_batch, AUTO_BATCH_MAX, None
            align_stats = {}                                  # OOM splits (align_subjects)

            def top_up():
                """Keep 2 x --workers (at least 2 x --batch-size) subjects loading ahead. When the known subjects run out, rescan --root for
                subjects written since (e.g. by a concurrently running download): at most every
                RESCAN_SECONDS while others are still queued, always once the queue is empty."""
                nonlocal it, last_scan, budget
                while len(queue) < max(2 * args.workers, 2 * batch_size):
                    if budget == 0:                       # --max-subjects reached (skipped ones are given back)
                        return
                    nxt = next(it, None)
                    if nxt is None:
                        if queue and time.time() - last_scan < RESCAN_SECONDS:
                            return
                        more = scan()
                        last_scan = time.time()
                        if not more:
                            return
                        print(f"\nFound {len(more)} new subject(s) in {args.root}", flush=True)
                        if hasattr(pbar, "total"):
                            pbar.total += len(more)
                            pbar.refresh()
                        it = iter(more)
                        continue
                    if budget > 0:
                        budget -= 1
                    queue.append((nxt, ex.submit(load_job, load, is_complete, nxt, run_dir / nxt.name, args)))

            def loaded(subj, fut):
                """The load job's result, or its exception. A job that failed is loaded once more here, in this
                process: a worker can fail to send back a subject it loaded fine (Windows pipes refuse large writes
                under memory pressure: WinError 1450), while a real load error fails again here as well."""
                try:
                    return unpack_loaded(fut.result())
                except Exception as err:  # noqa: BLE001
                    print(f"\nLoading {subj.name} in a worker failed ({type(err).__name__}: {err}); retrying here",
                          flush=True)
                    try:
                        return unpack_loaded(load_job(load, is_complete, subj, run_dir / subj.name, args))
                    except Exception as err2:  # noqa: BLE001
                        return err2

            def stop_if_cuda_lost(errors):
                """After a CUDA error that leaves the GPU context unusable, every later subject would fail the same
                way within milliseconds (a full-cohort run marked thousands as failed): stop instead, once the images
                being written are done, with exit code CUDA_LOST_EXIT -- submit_alignment*.sh then start the run
                again in a fresh process, which resumes where it stopped (--skip-processed)."""
                if not errors or not cuda_lost(device):
                    return
                err = errors[0]
                print(f"\nThe CUDA context is lost ({type(err).__name__}: {str(err).splitlines()[0] if str(err) else ''}); "
                      f"stopping with exit code {CUDA_LOST_EXIT} so that the run can be restarted", flush=True)
                for _, _, job in save_jobs:
                    try:
                        job.result()
                    except Exception:  # noqa: BLE001
                        pass
                ex.shutdown(wait=True, cancel_futures=True)
                raise SystemExit(CUDA_LOST_EXIT)

            def complete(entries, todo, outs, seconds):
                """Write the results of a finished batch (finish_subjects), which took `seconds` in all."""
                stop_if_cuda_lost([o for o in outs if isinstance(o, Exception)])
                for e, o in zip(todo, outs):
                    e["out"] = o
                write_entries(entries, seconds / len(entries))   # the batch's time, shared by its subjects

            def write_entries(entries, seconds):
                """Write the rows, outputs and progress of a batch's subjects (complete)."""
                for e in entries:
                    subj, subject, status, metrics_dir = e["subj"], e["subject"], e["status"], e["metrics_dir"]
                    summary, details, pairs, history, regions = None, [], [], None, []
                    try:
                        poses = []
                        if status == "ok":
                            if isinstance(e["out"], Exception):
                                raise e["out"]
                            summary, details, pairs, poses, history, regions = e["out"]
                            if not any(d["n_pairs"] > 0 for d in details):
                                status = "not_optimized"                 # no crossing planes: written unaligned
                            for row in sax_relative_motion(subj.name, poses, [d["n_pairs"] for d in details]):
                                relmotion_writer.writerow(fmt_row(row))
                            relmotion_f.flush()
                        if args.save_images and subject is not None:   # whatever could be loaded, aligned or not
                            out_dir = run_dir / subj.name
                            out_dir.mkdir(parents=True, exist_ok=True)
                            save_jobs.append((subj.name, "aligned images",
                                              ex.submit(save_job, load, save, subj, corrected_affines(subject, poses),
                                                        out_dir, args)))
                    except Exception as err:  # noqa: BLE001
                        traceback.print_exception(err)
                        status = f"error: {type(err).__name__}: {err}"[:200]
                    if status in ("not_optimized", "no_images"):
                        unaligned.append((subj.name, status))
                    elif status != "ok":
                        failed.append((subj.name, status))
                    if args.logging:
                        summary = summary if summary is not None else empty_summary(subj.name, status)
                        summary["seconds"] = round(seconds, 2)
                        summary["timestamp"] = datetime.now().isoformat(timespec="seconds")
                        if details:
                            save_jobs.append((subj.name, "subject metrics", ex.submit(save_subject_metrics, dict(summary),
                                                                                     details, pairs, metrics_dir, True,
                                                                                     history, regions=regions)))
                        (sf, swriter), (df_, dwriter), (pf, pwriter), (rf, rwriter) = writers
                        swriter.writerow(fmt_row(summary))
                        for row in details:
                            dwriter.writerow(fmt_row(row))
                        for row in pairs:
                            pwriter.writerow(fmt_row(row))
                        for row in regions:
                            rwriter.writerow(fmt_row(row))
                        for f, _ in writers:
                            f.flush()
                    while len(save_jobs) > 3 * args.workers:
                        name, what, job = save_jobs.popleft()
                        try:
                            job.result()
                        except Exception:  # noqa: BLE001
                            print(f"Writing {what} failed for {name}:")
                            traceback.print_exc()
                    if hasattr(pbar, "set_postfix"):
                        post = dict(subj=subj.name, status=status[:12], **(dict(skipped=n_skipped) if n_skipped else {}))
                        if summary is not None:
                            post.update(ncc=f"{summary['ncc_mean']:.2f}", jitter=f"{summary['sax_shift_jitter_mm']:.1f}")
                        pbar.set_postfix(**post)
                        pbar.update(1)

            top_up()
            while queue:
                # the next --batch-size loaded subjects (those found complete are skipped as they come up)
                batch = []
                while queue and len(batch) < batch_size:
                    subj, fut = queue.popleft()
                    res = loaded(subj, fut)
                    skip = isinstance(res, str) and res == SKIPPED_JOB
                    if skip:                                      # already complete (--skip-processed)
                        n_skipped += 1
                        if budget >= 0:
                            budget += 1
                        if hasattr(pbar, "total"):
                            if not capped:                    # (capped: another folder takes its place)
                                pbar.total -= 1
                            pbar.set_postfix(skipped=n_skipped)
                        top_up()
                        continue
                    batch.append((subj, res))
                    top_up()
                if not batch:
                    continue
                t0 = time.time()

                def lines_callback(sid, out):
                    def on_lines(planes, lines):
                        if not lines:
                            return
                        out.mkdir(parents=True, exist_ok=True)
                        save_line_geometry(planes, lines, out, args.line_extent_mm, args.line_step_mm)
                        viz = mask_plot_data(planes, lines, args.line_extent_mm, args.line_step_mm)
                        save_jobs.append((sid, "mask plot", ex.submit(save_mask_plot, viz, sid, args.mask_dilation_mm,
                                                                      out / "mask_overlay.png", args.line_extent_mm)))
                    return on_lines

                entries = []
                for subj, res in batch:
                    e = dict(subj=subj, metrics_dir=run_dir / subj.name / "metrics", subject=None, data=None,
                             pre=None, status="ok", out=None)
                    try:
                        if isinstance(res, Exception):
                            raise res
                        e["subject"], e["data"], e["pre"] = res
                        e["status"] = e["subject"].status if e["subject"].slices else "no_images"
                    except Exception as err:  # noqa: BLE001
                        traceback.print_exception(err)
                        e["status"] = f"error: {type(err).__name__}: {err}"[:200]
                    entries.append(e)
                todo = [e for e in entries if e["status"] == "ok"]
                probe = probing and len(todo) > 0
                if probe:
                    torch.cuda.synchronize(device)
                    base = torch.cuda.memory_allocated(device)
                    torch.cuda.reset_peak_memory_stats(device)
                n_oom = align_stats.get("oom", 0)
                ctxs, results = ([], {})
                if todo:                                  # aligned together (--batch-size; alone when 1)
                    ctxs, results = align_subjects([e["subject"] for e in todo], args, device,
                                                   [lines_callback(e["subj"].name, e["metrics_dir"])
                                                    if args.logging and e["pre"] is None else None  # (else by load_job)
                                                    for e in todo], [e["data"] for e in todo], [e["pre"] for e in todo],
                                                   align_stats)
                t_align = time.time() - t0
                stop_if_cuda_lost([r for r in list(ctxs) + list(results.values()) if isinstance(r, Exception)])
                n_live = sum(1 for c in ctxs if not isinstance(c, Exception) and c["lines"])
                outs = finish_subjects(ctxs, results, args, device) if todo else []
                if probe and n_live:                      # --batch-size 0: the memory bound, from this batch
                    probing = False
                    per = max(0, torch.cuda.max_memory_allocated(device) - base) / n_live
                    mem_max = int(max(1, min(AUTO_BATCH_MAX, AUTO_BATCH_FILL * free0 // max(per, 1))))
                    print(f"\n[batch] --batch-size 0: a subject takes ~{per / 2 ** 20:.0f} MiB of GPU memory "
                          f"({free0 / 2 ** 30:.1f} GiB were free): at most {mem_max} per batch", flush=True)
                complete(entries, todo, outs, time.time() - t0)
                del ctxs, results
                if tuning and not probing and n_live >= max(2, int(0.75 * batch_size)):
                    per_t = t_align / n_live                     # alignment seconds per subject at this size
                    if tune_best is None or per_t < tune_best[1] * (1.0 - AUTO_BATCH_GAIN):
                        tune_best = (batch_size, per_t)
                        nxt = min(2 * batch_size, mem_max)
                        tuning = nxt > batch_size
                        batch_size = nxt
                    else:
                        batch_size, tuning = tune_best[0], False
                    print(f"\n[batch] --batch-size 0: {per_t:.2f} s per subject in this batch; "
                          f"{'trying' if tuning else 'settled on'} {batch_size} subjects per batch", flush=True)
                if auto_batch and align_stats.get("oom", 0) > n_oom and batch_size > 1:
                    batch_size = max(1, batch_size * 3 // 4)       # a batch had to be split: smaller from now on
                    mem_max, tuning = batch_size, False
                    print(f"\n[batch] --batch-size 0: down to {batch_size} subjects per batch", flush=True)
                if not queue:                              # subjects written while these were aligned
                    top_up()
            for name, what, job in save_jobs:
                try:
                    job.result()
                except Exception:  # noqa: BLE001
                    print(f"Writing {what} failed for {name}:")
                    traceback.print_exc()
    finally:
        for f, _ in writers or []:
            f.close()
        relmotion_f.close()
    if n_skipped:
        print(f"\nSkipped {n_skipped} subject(s) whose aligned images were already complete (--skip-processed)")
    if unaligned:
        print(f"\n{len(unaligned)} subject(s) without crossing planes, written unaligned: "
              + ", ".join(f"{s} ({st})" for s, st in unaligned))
    if failed:
        print(f"\n{len(failed)} subject(s) failed: " + ", ".join(f"{s} ({st})" for s, st in failed))
    if args.logging:
        print(f"\nSummary written to {summary_path}\nPer-slice results written to {detail_path}\nPer-pair results written to {pairs_path}"
              f"\nPer-combination results written to {regions_path}")
        try:
            n = len(plot_results(summary_path, detail_path, overall_dir, pairs_path))
            print(f"{n} overall figures written to {overall_dir}")
        except Exception:  # noqa: BLE001
            print("Overall plotting failed:")
            traceback.print_exc()
    if args.save_images:
        print(f"Aligned images under {run_dir}\\<subject>\\" + (" (and metrics\\)" if args.logging else ""))
    print(f"SAX relative motion: {overall_dir / 'sax_relative_motion.csv'}")
    return run_dir, sorted(seen)


if __name__ == "__main__":
    sys.exit("motion_gradient.py is the motion-correction library; run it through a data entry point:\n"
             "  python moco_nifti.py ...   (UKBB NIfTI: SAX stack + LAX views)\n"
             "  python moco_dicom.py ...   (DICOM series)")
