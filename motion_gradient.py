"""
Gradient-based inter-slice motion correction of cine MRI: the motion-correction library. It is not run on its own;
the entry points are the data loaders, which read a subject, hand it over and write the result in their own format:
  moco_nifti.py   UK Biobank NIfTI (SAX stack and/or slices + LAX views, nnU-Net segmentations)
  moco_dicom.py   DICOM series (any number of stacks / orientations)

Input (from a loader, see SliceImage / Subject): a list of 2D+time cine slices, each (X, Y, T) with its own 4x4
affine (voxel (i, j, 0) -> world mm, one world frame per subject), image (stack) name and index in its stack, slice
thickness and an optional (X, Y, T) segmentation. Slices named "SAX" are short-axis slices (SAX regularizers, QC and
reports). Every slice is a plane with its own pose; no regular stack geometry is assumed. Segmentations are optional:
a non-SAX image's segmentation defines its dilated heart mask (the line points scored); without one its lines run
over the whole overlap with the other image. They are also used for the reported Dice / Hausdorff.
Output: per slice its corrected affine (corrected_affines), written back by the loader; metrics and figures below.
run() is the shared run loop (subject scanning, background loading/writing, CSVs, figures).

Intersection pairs (candidate_pairs): every two slices whose acquired normals differ by at least --min-pair-angle-deg,
thinned between multi-slice stacks to partners about a slice thickness apart (--pair-spacing-mm). Nearly parallel SAX
slices within --parallel-max-dist-mm are neighbours for the SAX-SAX crossing terms instead. The LV long axis (rotation
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
stack that moves as one rigid body. Two more terms keep neighbours from crossing inside the heart
(see SaxNeighbourLines): --sax-gap-reg, a barrier on the gap between them on a disk around the LV
long axis, and --sax-sax-weight, a dissimilarity cost along their crossing line where they do
cross.
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

With --logging, segmentation consistency (Dice between the two views' labels along the same lines) is
reported before and after alignment as an independent check; it is not optimized.

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
    motion_summary_pairs.csv  one row per intersection-line pair (NCC, NMI, Dice, Hausdorff before/after; with
                              --sax-sax-weight also SAX-SAX rows: NCC, counting points and axis distance at the end)
    motion_*.png              dataset-level figures (written at the end of a run)
  <subject>/
    metrics/
      summary.csv, slices.csv, pairs.csv  this subject's rows
      history.csv             pose snapshots every 2% of the run (step 0 = acquired geometry, and the
                              saved "best" pose with is_best=1): per plane the full 3x4 affine (a00..a23, same
                              convention as the saved images), the pose and NCC/LNCC/NMI/Dice/Hausdorff
      history_summary.csv     the same snapshots, one row per step with the objective and subject-level metrics
      lines.csv               every intersection line entering the loss (acquired geometry): the two planes, the
                              anchor point x0 and unit direction d (world, mm) and the parameter ranges along the
                              line that are inside both images (t_img_*) and actually sampled (t_used_*)
      sax_lines.csv           the crossing line of every pair of neighbouring SAX slices at every history snapshot
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
                              sax_rel_tilt_max_deg and sax_cross_dissim_max (worst dissimilarity of a crossing pair).
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
import csv
import math
import sys
import time
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt


SUMMARY_COLUMNS = [
    "subject_id", "status", "pose_model", "n_lax_views", "sax_n_slices", "sax_n_eval",
    "sax_shift_mean_mm", "sax_shift_rms_mm", "sax_shift_max_mm", "sax_shift_jitter_mm",
    "sax_dz_rms_mm", "sax_rot_inplane_rms_deg", "sax_tilt_rms_deg",
    "lax_shift_mean_mm", "lax_shift_max_mm",
    "ncc_zero_mean", "ncc_mean", "ncc_gain_mean",
    "nmi_zero_mean", "nmi_mean", "nmi_gain_mean",
    "seg_dice_zero_mean", "seg_dice_mean",
    "hd_zero_mean_mm", "hd_mean_mm", "hd_zero_max_mm", "hd_max_mm",
    "sax_cross_pairs", "sax_gap_min_mm", "sax_rel_tilt_max_deg", "sax_cross_dissim_max",
    "n_line_pairs", "steps", "seconds", "timestamp",
]
DETAIL_COLUMNS = [
    "subject_id", "image", "slice", "n_pairs", "n_points",
    "dx_mm", "dy_mm", "dz_mm", "shift_inplane_mm", "shift_mm", "rot_inplane_deg", "rot_tilt_deg",
    "ncc_zero", "ncc", "nmi_zero", "nmi", "seg_dice_zero", "seg_dice", "hd_zero_mm", "hd_mm",
]
PAIR_COLUMNS = [
    "subject_id", "image_a", "slice_a", "image_b", "slice_b", "kind", "n_points",
    "ncc_zero", "ncc", "nmi_zero", "nmi", "seg_dice_zero", "seg_dice", "hd_zero_mm", "hd_mm", "axis_dist_mm",
    "gap_min_mm", "rel_tilt_deg",
]


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
    numbered by `index` along the stack; the image name "SAX" marks short-axis slices, which get the SAX-specific
    regularizers, QC and reports. All slices of one non-SAX image must have the same in-plane size."""
    image: str                      # image (stack) the slice belongs to, e.g. "SAX", "LAX_4Ch", a series name
    index: int                      # position in its stack (0, 1, ...)
    data: np.ndarray                # (X, Y, T) float32 full cine
    affine: np.ndarray              # (4, 4) voxel (i, j, 0) -> world mm
    thickness: float                # slice thickness (mm), for the slice-profile PSF
    seg: np.ndarray | None = None   # (X, Y, T) integer labels (optional; heart mask and Dice / Hausdorff)
    source: object = None           # loader-specific handle for writing the slice back (never read here)

    @property
    def key(self) -> str:
        return f"{self.image}:{self.index}"


@dataclass
class Subject:
    subject_id: str
    slices: list                    # list[SliceImage], images in the order they were found
    status: str = "ok"              # "ok", or e.g. "no_images" (nothing to align)


def phase_frames(n_frames_available: int, n_frames: int) -> list[int]:
    return sorted({int(math.floor(k / n_frames * n_frames_available)) for k in range(n_frames)})


def prepare_subject(subject: Subject, args) -> dict:
    """The internal form of a subject: --n-frames evenly spaced frames per slice (the same number for every slice),
    SAX slices one by one, every other image as a stack (X, Y, Z, F) with per-slice affines (Z, 4, 4), and, for images
    with a segmentation, the dilated heart mask (--mask-dilation-mm) and the distance to the heart (mm, for mask
    annealing) per slice. A non-SAX image whose segmentation contains no heart is left out of the alignment (it is
    still written, unaligned, by the loader); one without a segmentation takes part unmasked."""
    out = dict(subject_id=subject.subject_id, status=subject.status, sax=[], lax={}, order=[])
    if not subject.slices:
        out["status"] = "no_images"
        return out
    n_f = min([args.n_frames] + [s.data.shape[-1] for s in subject.slices])

    def frames(a):
        return a[..., phase_frames(a.shape[-1], n_f)]

    for s in subject.slices:
        if s.image not in out["order"]:
            out["order"].append(s.image)
    for s in sorted((s for s in subject.slices if s.image == "SAX"), key=lambda s: s.index):
        out["sax"].append(dict(index=s.index, arr=frames(s.data)[:, :, None].astype(np.float32), affine=s.affine,
                               seg=None if s.seg is None else frames(s.seg)[:, :, None], thickness=s.thickness))
    for image in out["order"]:
        if image == "SAX":
            continue
        sl = sorted((s for s in subject.slices if s.image == image), key=lambda s: s.index)
        if len({s.data.shape[:2] for s in sl}) > 1:
            raise ValueError(f"{subject.subject_id}: the slices of image {image} differ in size")
        arr = np.stack([frames(s.data) for s in sl], 2).astype(np.float32)          # (X, Y, Z, F)
        entry = dict(arr=arr, affine=np.stack([s.affine for s in sl]), mask=None, seg=None,
                     thickness=sl[0].thickness, dist=None)
        if any(s.seg is not None for s in sl):
            seg = np.stack([frames(s.seg) if s.seg is not None else np.zeros_like(frames(s.data), np.int16)
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
                    dist[:, :, z] = distance_transform_edt(~heart[:, :, z], sampling=spacing)
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
        sax_planes += make_planes("SAX", arr, e["affine"], e.get("thickness", args.sax_thickness), None,
                                  e["seg"] if with_seg else None,
                                  device, first_index=e["index"], stack_rank=rank("SAX"))
    lax_planes = []
    for view, d in data["lax"].items():
        arr = normalize_intensity(d["arr"]) if l2 else d["arr"]
        lax_planes += make_planes(view, arr, d["affine"], d["thickness"], d["mask"], d["seg"] if with_seg else None, device,
                                  stack_rank=rank(view))
    # programmatic callers without the option (older runs' configs) keep the image-centre rotation they were made with
    set_rotation_pivots(sax_planes, lax_planes, getattr(args, "rotation_pivot", "image-centre"))
    return sax_planes, lax_planes


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
                stack_rank: int = 0) -> list[Plane]:
    """Planes of a (X, Y, Z, F) image; plane z gets slice index first_index + z and the affine of slice z: affine is
    either one per slice (Z, 4, 4) or the stack's (4, 4), whose origin is then moved to slice z. SAX slices come one at
    a time (Z = 1, first_index = the slice's index); other images may be stacks (first_index 0)."""
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


def intersection_line(a: Plane, b: Plane, args, device) -> Line | None:
    """Valid parameter range and mask weights of the a/b intersection line at the acquired geometry.

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


def l1_similarity(a: torch.Tensor, b: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """--metric l1: -mean |a - b| of profiles a, b (M, F, L) with point weights w (M, L), mean over frames -> (M,).
    Only meaningful on images normalized to a common intensity scale (build_planes does this for l1)."""
    wf = w[:, None, :]
    return -((wf * (a - b).abs()).sum(-1) / wf.sum(-1).clamp_min(1.0)).mean(1)


def ngf_similarity(a: torch.Tensor, b: torch.Tensor, w: torch.Tensor, grad_eps: float) -> torch.Tensor:
    """--metric ngf (normalized gradient agreement) of profiles a, b (M, F, L) with point weights w (M, L) -> (M,).

    The central-difference gradients g along the line are normalized per point, g / sqrt(g^2 + eps^2), with eps =
    grad_eps x the standard deviation of g over the weighted points (per profile and frame, so each image's contrast
    does not matter): edges well above eps -> +-1, flat stretches -> 0. Score sum(na nb) / sum((na^2 + nb^2) / 2) in
    [-1, 1] (interior points, weighted where all three samples are): 1 = the same edges at the same places with the
    same sign, 0 for edges in one image only, -1 for opposite edges. Mean over frames."""
    wg = (w[:, :-2] * w[:, 1:-1] * w[:, 2:])[:, None, :]
    ws = wg.sum(-1, keepdim=True).clamp_min(1e-6)

    def normalize(x):
        g = x[..., 2:] - x[..., :-2]
        eps = grad_eps * ((wg * g * g).sum(-1, keepdim=True) / ws + 1e-12).sqrt() + 1e-6   # finite gradient at 0
        return g / (g * g + eps * eps).sqrt()

    na, nb = normalize(a), normalize(b)
    num = (wg * na * nb).sum(-1)
    den = (wg * 0.5 * (na * na + nb * nb)).sum(-1).clamp_min(1e-6)
    return (num / den).mean(1)


def lncc_similarity(a: torch.Tensor, b: torch.Tensor, w: torch.Tensor, k: int, min_std: float) -> torch.Tensor:
    """--metric lncc (local NCC) of profiles a, b (M, F, L) with point weights w (M, L) -> (M,).

    The profiles are z-scored over their weighted points, then NCC is taken in windows of k samples sliding by one
    sample and averaged over the windows, each weighted by its weighted point count (0 below half a window) x a
    confidence that fades out windows where the profiles hardly vary: min(1, (sd_a sd_b)^(1/2) / min_std), sd in units
    of the profile's standard deviation over the whole line. Every stretch has to match on its own; one strong edge
    cannot dominate. Mean over frames."""
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
    return ((loc * wt).sum(-1) / wt.sum(-1).clamp_min(1e-6)).mean(1)


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
        self.lax_lax = torch.tensor([planes[ia].image != "SAX" and planes[ib].image != "SAX" for ia, ib, _ in lines], device=device)
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

    def set_blur(self, sigma_mm: float) -> None:
        """In-plane Gaussian blur (sigma in mm) of the images that are sampled (see ImageVolume.set_blur)."""
        self.img.set_blur(sigma_mm)

    def set_psf_scale(self, scale: float) -> None:
        """Width of the PSF band relative to the nominal slice profile (1 = physical)."""
        self.psf_scale.fill_(scale)

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

    def ncc_lines(self, theta_deg, t, use_gradient: bool, prof=None, W=None) -> torch.Tensor:
        a, b = prof if prof is not None else self.profiles(theta_deg, t, "img")
        W = self.W if W is None else W
        return ncc(a, b, W[:, None, :], use_gradient).mean(1)                  # (M,)

    def nmi_lines(self, theta_deg, t, bins: int, sigma_bins: float, prof=None, W=None) -> torch.Tensor:
        """Normalized mutual information per line, samples pooled over frames -> (M,)."""
        a, b = prof if prof is not None else self.profiles(theta_deg, t, "img")
        w = (self.W if W is None else W)[:, None, :].expand_as(a)
        m = a.shape[0]
        return soft_nmi(a.reshape(m, -1), b.reshape(m, -1), w.reshape(m, -1), bins, sigma_bins)

    def l2_lines(self, theta_deg, t, prof=None, W=None) -> torch.Tensor:
        """Negative weighted mean squared intensity difference per line, mean over frames -> (M,) (higher = better).
        Only meaningful on images normalized to a common intensity scale (see normalize_intensity)."""
        a, b = prof if prof is not None else self.profiles(theta_deg, t, "img")
        w = (self.W if W is None else W)[:, None, :]
        return -((w * (a - b) ** 2).sum(-1) / w.sum(-1).clamp_min(1.0)).mean(1)

    def similarity_lines(self, theta_deg, t, args, prof=None, W=None, sel=None) -> torch.Tensor:
        """The optimized similarity per line (--metric), with the loss weights W_loss (points outside the mask
        down-weighted by --outside-mask-weight) or the given W (e.g. self.W: the mask points only) -> (M,).
        sel: only the lines with these indices are sampled and scored -> (len(sel),)."""
        W = self.W_loss if W is None else W
        if sel is not None:
            W = W[sel]
            prof = self.profiles(theta_deg, t, "img", sel) if prof is None else prof
        if args.metric == "nmi":
            return self.nmi_lines(theta_deg, t, args.nmi_bins, args.nmi_sigma, prof, W)
        if args.metric == "l2":
            return self.l2_lines(theta_deg, t, prof, W)
        if args.metric in ("ncc", "ncc_grad"):
            return self.ncc_lines(theta_deg, t, args.metric == "ncc_grad", prof, W)
        a, b = prof if prof is not None else self.profiles(theta_deg, t, "img")
        if args.metric == "l1":
            return l1_similarity(a, b, W)
        if args.metric == "ngf":
            return ngf_similarity(a, b, W, args.grad_eps)
        return lncc_similarity(a, b, W, max(3, int(round(args.lncc_window_mm / args.line_step_mm))), args.lncc_min_std)

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

    def per_plane(self, values: torch.Tensor) -> list[float]:
        """Line values (M,) -> npts-weighted mean per plane, ignoring NaN (one host transfer)."""
        ok = torch.isfinite(values)
        v = torch.where(ok, values, torch.zeros_like(values)) * self.npts
        w = self.npts * ok
        num = torch.zeros(self.N, device=self.device).index_add_(0, self.IA, v).index_add_(0, self.IB, v)
        den = torch.zeros(self.N, device=self.device).index_add_(0, self.IA, w).index_add_(0, self.IB, w)
        return torch.where(den > 0, num / den.clamp_min(1e-12), torch.full_like(num, float("nan"))).tolist()

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
        out = dict(ncc=self.ncc_lines(theta_deg, t, args.metric == "ncc_grad", img),
                   lncc=lncc_similarity(img[0], img[1], self.W, k_lncc, getattr(args, "lncc_min_std", 0.25)),
                   nmi=self.nmi_lines(theta_deg, t, args.nmi_bins, args.nmi_sigma, img),
                   dice=self.dice_lines(theta_deg, t, labels, labels_sax, seg) if has_seg else nan,
                   hd=self.hd_lines(theta_deg, t, labels, labels_sax, getattr(args, "hd_miss_mm", 100.0), seg) if has_seg else nan)
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
    else the image found first (the loader's order). None without planes."""
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
    both images (and the masks) is decided by intersection_line."""
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
    masks over at least --min-line-mm (intersection_line), at the planes' current masks."""
    index = {p.key: i for i, p in enumerate(planes)}
    lines = []
    with torch.no_grad():
        for a, b in candidate_pairs(planes, args):
            line = intersection_line(a, b, args, device)
            if line is not None:
                lines.append((index[a.key], index[b.key], line))
    return lines


class SaxNeighbourLines:
    """Neighbouring SAX slices (parallel as acquired and at most --parallel-max-dist-mm apart; UKBB: z and z + 1):
    their crossing line and the gap between them.

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
                 axis: tuple[np.ndarray, np.ndarray], args, device):
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
        self.L0 = torch.tensor(axis[0], **f32)
        self.ell = torch.tensor(axis[1] / np.linalg.norm(axis[1]), **f32)
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

    def _sample(self, side: int, vox: torch.Tensor) -> torch.Tensor:
        """Bilinear samples (P, F, L) of each pair's side-`side` image at voxel coordinates vox (P, L, 3), read from
        the sampler's image volume (exact plane depths, see LineSampler)."""
        iv = self.sampler.img
        idx = self.IA if side == 0 else self.IB
        z = iv.depth[idx][:, None].expand_as(vox[..., 0])
        grid = torch.stack([2 * vox[..., 1] / (iv.Wmax - 1) - 1, 2 * vox[..., 0] / (iv.Hmax - 1) - 1, z], -1)
        return iv.sample(grid[None, :, :, None, :])[:, :, :, 0].float().transpose(0, 1)       # (P, F, L)

    def profiles(self, theta_deg: torch.Tensor, t: torch.Tensor):
        """At the given poses: both images' profiles (P, F, L) x 2, the point weights (P, L), which pairs cross at
        all (P,) and the distance (mm) of each pair's line from the long axis (P,)."""
        g = self.geometry(theta_deg, t)
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
        c12, c20, c01 = torch.cross(nb, dh, dim=1), torch.cross(dh, na, dim=1), torch.cross(na, nb, dim=1)
        rhs = torch.stack([(na * Op[IA]).sum(1), (nb * Op[IB]).sum(1), (dh * self.L0).sum(1)], 1)
        p = (rhs[:, 0:1] * c12 + rhs[:, 1:2] * c20 + rhs[:, 2:3] * c01) / nd.clamp_min(1e-3)
        w0 = p - self.L0
        b = (dh * self.ell).sum(1)
        dd, e = (dh * w0).sum(1), (self.ell * w0).sum(1)
        denom = (1.0 - b * b).clamp_min(1e-3)
        s, tau = (b * e - dd) / denom, (e - b * dd) / denom
        x0 = p + s[:, None] * dh
        gap = w0 + s[:, None] * dh - tau[:, None] * self.ell
        axis_dist = torch.sqrt((gap * gap).sum(1).clamp_min(1e-12))
        pts = x0[:, None, :] + self.T[None, :, None] * dh[:, None, :]                  # (P, L, 3)
        rel = pts - self.L0
        perp = rel - (rel * self.ell).sum(-1, keepdim=True) * self.ell
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
        for own, other, side_sign in ((IA, IB, -1.0), (IB, IA, 1.0)):
            n = Np[own]
            cos = (self.ell * n).sum(1)
            cos = torch.where(cos >= 0, 1.0, -1.0) * cos.abs().clamp_min(0.1)
            centre = self.L0 + (((Cp[own] - self.L0) * n).sum(1) / cos)[:, None] * self.ell   # long axis x this slice
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
        g = self.geometry(theta_deg, t)
        c, n, _ = self.scores(theta_deg, t)
        gp = self.gaps(theta_deg, t)
        gap_min = gp["gap"].flatten(1).min(1).values.cpu().numpy()
        tilt = gp["rel_tilt"].cpu().numpy()
        T = self.T.cpu().numpy()
        x0 = g["x0"].cpu().numpy().astype(np.float64) - shift
        d, active = g["d"].cpu().numpy(), g["active"].cpu().numpy()
        in_img, used = g["in_img"].cpu().numpy(), g["in_radius"].cpu().numpy()
        dist, c, n = g["axis_dist"].cpu().numpy(), c.cpu().numpy(), n.cpu().numpy()
        nan = float("nan")
        rows = []
        for k, (ia, ib) in enumerate(self.pairs):
            ti, tu = T[in_img[k]], T[used[k]]
            counts = n[k] > 1e-3
            row = dict(slice_a=self.sampler.planes[ia].slice_idx, slice_b=self.sampler.planes[ib].slice_idx,
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

    def scores(self, theta_deg: torch.Tensor, t: torch.Tensor):
        """Per pair: NCC along its line with the point weights (0 when it does not cross), the sum of the point weights
        and the axis distance (mm)."""
        a, b, w, active, dist = self.profiles(theta_deg, t)
        # NCC is invariant to the scale of the weights, but ncc() floors their sum at 1 (meant for 0/1 masks), so
        # soft weights are rescaled to a maximum of 1 per line first
        wn = w / w.amax(-1, keepdim=True).clamp_min(1e-6)
        c = ncc(a, b, wn[:, None, :], False).mean(1) * active.float()
        return c, w.sum(-1), dist


def new_sampler(planes: list[Plane], lines: list[tuple[int, int, Line]], args, device,
                images: ImageVolume | None = None) -> LineSampler:
    """A LineSampler with the run's PSF settings and image precision (--sample-dtype); `images`: the ImageVolume of
    another sampler of the same planes, to share (default: a new one)."""
    return LineSampler(planes, lines, args.n_band, device, args.psf_model, args.psf_tbw, getattr(args, "psf_scale_start", 1.0),
                       sample_dtype=getattr(args, "sample_dtype", "float32"), images=images)


def attach_sax_pairs(sampler: LineSampler, planes: list[Plane], sax_planes: list[Plane], lax_planes: list[Plane],
                     args, device) -> None:
    """Give a sampler the neighbouring-SAX-slice terms (SaxNeighbourLines; needs >= 2 LAX views for the long axis).
    They enter the loss with --sax-sax-weight / --sax-gap-reg > 0; with logging they are built at weight 0 as well,
    so that sax_lines.csv, the SAX-SAX rows of pairs.csv and the summary's QC columns are always recorded."""
    sampler.sax_pairs = None
    supervise = ((getattr(args, "sax_sax_weight", 0.0) > 0 and args.dof == 6) or getattr(args, "sax_gap_reg", 0.0) > 0)
    if not (supervise or getattr(args, "logging", True)):
        return
    axis = heart_axis(sax_planes, lax_planes)
    if axis is None:
        return
    # neighbours: (nearly) parallel SAX slices -- closer in angle than --min-pair-angle-deg, so not an intersection
    # pair -- whose centres are at most --parallel-max-dist-mm apart along the normal (UKBB, 10 mm spacing: z, z + 1)
    index = {p.key: i for i, p in enumerate(planes)}
    cos_min = math.cos(math.radians(getattr(args, "min_pair_angle_deg", 20.0)))
    max_dist = getattr(args, "parallel_max_dist_mm", 12.0)
    ordered = sorted(sax_planes, key=lambda p: p.slice_idx)
    pairs = [(index[a.key], index[b.key]) for i, a in enumerate(ordered) for b in ordered[i + 1:]
             if abs(float(a.normal @ b.normal)) > cos_min and abs(float((b.center - a.center) @ a.normal)) <= max_dist]
    if pairs:
        sampler.sax_pairs = SaxNeighbourLines(sampler, planes, pairs, axis, args, device)


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


OPTIMIZERS = ("adam", "nadam", "radam", "adamax", "rmsprop", "adagrad", "sgd")
# default learning rates (translation mm, rotation deg) per optimizer when --lr / --rot-lr are not given. The adaptive
# methods take steps of about lr per iteration whatever the gradient scale; Adagrad's steps shrink like 1/sqrt(step),
# so it needs a larger lr; plain SGD steps are lr * gradient, and the NCC objective's gradients are small (see --lr).
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


def align(final_sampler: LineSampler, args, stages: list[tuple[int, LineSampler]] | None = None,
          pose_model: str = "6dof") -> dict:
    """Jointly optimize a rigid pose per plane from the acquired geometry: translation plus a 3D rotation (pose_model
    6dof), plus a rotation about the plane's own normal only (inplane), or translation only (translation).

    `stages` optionally lists (start_step, sampler) for the annealed mask geometry used early in the run, ending
    with (anneal_end, final_sampler); `final_sampler` -- the nominal geometry -- is used from anneal_end on (or always,
    if no stages are given). The best pose is only tracked once every annealed quantity (geometry, blur, PSF width) is
    at its nominal value, on the full (all lines) objective.

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

    def sampler_at(step):
        active = stages[0][1]
        for start, smp in stages:
            if step >= start:
                active = smp
        return active

    f32 = dict(dtype=torch.float32, device=device)
    rotate = args.dof in (4, 6) and args.rot_lr > 0 and pose_model in ("6dof", "inplane")
    inplane = rotate and pose_model == "inplane"
    theta_deg = torch.zeros(N, 3, **f32).requires_grad_(rotate and not inplane)
    phi = torch.zeros(N, 1, **f32).requires_grad_(inplane)          # inplane: angle (deg) about each plane's normal
    t = torch.zeros(N, 3, **f32).requires_grad_(True)

    def pose_theta():
        """Rotation vectors (N, 3, deg) of the current parameters."""
        return phi * final_sampler.Nrm if inplane else theta_deg

    # SAX stack regularizers over adjacent slices (a, b = slice z, z + 1); the plane normals and centres are the same
    # in every stage sampler
    tilt_prior = getattr(args, "tilt_prior", 0.0) if rotate else 0.0
    w_tilt_hinge = getattr(args, "tilt_hinge_reg", 0.0) if rotate else 0.0
    tilt_hinge_deg = getattr(args, "tilt_hinge_deg", 5.0)
    w_spacing = getattr(args, "spacing_reg", 0.0)
    w_rel_shift = getattr(args, "rel_shift_reg", 0.0)
    w_rel_rot = getattr(args, "rel_rot_reg", 0.0) if rotate else 0.0
    sax = sorted((p.slice_idx, i) for i, p in enumerate(final_sampler.planes) if p.image == "SAX")
    adjacent = [(i, j) for (za, i), (zb, j) in zip(sax, sax[1:]) if zb == za + 1]
    stack_reg = bool(adjacent) and (w_tilt_hinge > 0 or w_spacing > 0 or w_rel_shift > 0 or w_rel_rot > 0)
    if stack_reg:
        nb_a = torch.tensor([i for i, _ in adjacent], device=device)
        nb_b = torch.tensor([j for _, j in adjacent], device=device)
        C0, N0 = final_sampler.C, final_sampler.Nrm
        step0 = ((C0[nb_b] - C0[nb_a]) * N0[nb_a]).sum(1)                   # acquired signed spacing along the normal
        spacing = getattr(args, "sax_spacing_mm", 10.0)
        target = torch.sign(step0) * (spacing if spacing > 0 else step0.abs())

    weights = dict(tilt_hinge=w_tilt_hinge, spacing=w_spacing, rel_shift=w_rel_shift, rel_rot=w_rel_rot)

    def stack_penalty(theta_deg, t):
        terms = stack_terms(theta_deg, t, C0, N0, nb_a, nb_b, target, {k for k, w in weights.items() if w > 0},
                            tilt_hinge_deg)
        return sum(weights[k] * v for k, v in terms.items())

    def objective(theta_deg, t, sel=None, smp=final_sampler):
        score = similarity_objective(smp, theta_deg, t, args, sel)
        gauge = args.gauge * (t.mean(0).pow(2).sum() + theta_deg.mean(0).pow(2).sum())
        dz = (t * smp.Nrm).sum(1)
        prior = args.prior * (theta_deg.pow(2).sum(1) + dz.pow(2)).mean()
        if tilt_prior > 0:                     # tilt above --tilt-hinge-deg (deg^2); in-plane rotation is free
            tilt = theta_deg - (theta_deg * smp.Nrm).sum(1, keepdim=True) * smp.Nrm
            tilt_deg = tilt.pow(2).sum(1).clamp_min(1e-12).sqrt()          # = rot_tilt_deg; finite gradient at 0
            prior = prior + tilt_prior * torch.relu(tilt_deg - tilt_hinge_deg).pow(2).mean()
        return score - gauge - prior - (stack_penalty(theta_deg, t) if stack_reg else 0.0) - sax_pair_penalty(smp, theta_deg, t)

    # stochastic line-pair dropping: each step samples only a random subset of round(--pair-keep x M) of the
    # intersection lines (at least one), so memory and time per step scale with the kept lines; a fixed subset size
    # keeps the shapes static for the CUDA graphs
    keep_frac = getattr(args, "pair_keep", 1.0)
    stochastic = keep_frac < 1.0
    gen = torch.Generator(device=device)
    gen.manual_seed(getattr(args, "seed", 0))
    eval_every = max(1, getattr(args, "eval_every", 10))

    def n_keep(smp):
        return max(1, min(smp.M, int(round(keep_frac * smp.M))))

    def draw_keep(smp):
        return torch.randperm(smp.M, generator=gen, device=device)[:n_keep(smp)]

    # learning rates as device tensors (updated in place each step; captured graphs read them)
    lr_t = torch.tensor(float(args.lr), **f32)
    lr_r = torch.tensor(float(args.rot_lr), **f32)
    param_groups = [{"params": [t], "lr": lr_t}] + ([{"params": [phi if inplane else theta_deg], "lr": lr_r}] if rotate else [])
    opt = make_optimizer(param_groups, args, device)

    def lr_at(step, start, end):
        if args.lr_schedule == "constant" or start <= 0:
            return start
        return start + (end - start) * min(1.0, step / max(1, args.steps - 1))        # linear start -> end

    blur_sigma = getattr(args, "blur_sigma", 0.0)
    blur_const = blur_sigma > 0 and getattr(args, "blur_frac", 0.6) <= 0      # --blur-frac 0: no decay
    blur_steps = int(round(getattr(args, "blur_frac", 0.6) * args.steps)) if blur_sigma > 0 and not blur_const else 0
    blur_quantum = 0.2                                     # mm; the blur is only recomputed when sigma moves this much
    psf_s0 = getattr(args, "psf_scale_start", 1.0)
    psf_steps = int(round(getattr(args, "psf_anneal_frac", 0.6) * args.steps)) if psf_s0 > 1.0 else 0
    track_from = max(anneal_end, blur_steps, psf_steps)    # from here on the objective is the nominal one

    # device-side bookkeeping: best pose so far (no host sync per step)
    best_obj = torch.tensor(-float("inf"), **f32)
    best_th, best_t = theta_deg.detach().clone(), t.detach().clone()
    best_step = torch.zeros((), **f32)
    step_buf, track_buf = torch.zeros((), **f32), torch.zeros((), **f32)
    sel_bufs = {id(smp): torch.arange(n_keep(smp), device=device) for _, smp in stages} if stochastic else {}
    for _, smp in stages:                                # the full-objective evaluations stay within a step's memory
        smp.eval_chunk = min(smp.eval_chunk, n_keep(smp)) if stochastic else smp.eval_chunk

    def update_best(o):
        better = (o > best_obj) & (track_buf > 0)
        best_obj.copy_(torch.where(better, o, best_obj))
        best_th.copy_(torch.where(better, pose_theta().detach(), best_th))
        best_t.copy_(torch.where(better, t.detach(), best_t))
        best_step.copy_(torch.where(better, step_buf, best_step))

    def body(smp):
        opt.zero_grad(set_to_none=False)
        obj = objective(pose_theta(), t, sel_bufs.get(id(smp)), smp)
        (-obj).backward()
        update_best(obj.detach())                          # pose *before* this update, as before
        opt.step()

    use_graph = device.type == "cuda" and not getattr(args, "no_cuda_graph", False)
    graphs = {}

    def capture(smp):
        """Capture body(smp) as a CUDA graph. Warm-up iterations run for real, so all state is restored after."""
        params = [p for g in param_groups for p in g["params"]]
        saved_p = [p.detach().clone() for p in params]
        saved_state = {i: {k: v.clone() for k, v in opt.state[p].items()} for i, p in enumerate(params) if p in opt.state}
        saved_book = [x.clone() for x in (best_obj, best_th, best_t, best_step)]
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
            for x, v in zip((best_obj, best_th, best_t, best_step), saved_book):
                x.copy_(v)
        return graph

    def run_step(smp):
        nonlocal use_graph
        if use_graph:
            graph = graphs.get(id(smp))
            if graph is None:
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
    with torch.enable_grad():
        for step in range(args.steps):
            if step in snap_steps:
                snapshots.append((step, pose_theta().detach().clone(), t.detach().clone()))
            smp = sampler_at(step)
            sigma = blur_sigma if blur_const else blur_sigma * max(0.0, 1.0 - step / blur_steps) if step < blur_steps else 0.0
            smp.set_blur(math.ceil(sigma / blur_quantum) * blur_quantum if sigma > 0 else 0.0)
            smp.set_psf_scale(psf_s0 + (1.0 - psf_s0) * step / psf_steps if step < psf_steps else 1.0)
            lr_t.fill_(lr_at(step, args.lr, args.lr_end))
            lr_r.fill_(lr_at(step, args.rot_lr, args.rot_lr_end))
            nominal = step >= track_from
            step_buf.fill_(float(step))
            if stochastic:
                sel_bufs[id(smp)].copy_(draw_keep(smp))
                # the dropped-line objective is not comparable between steps: track the best periodically on the
                # full objective (eagerly, before the update), and never inside the graph
                if nominal and (step % eval_every == 0 or step == args.steps - 1):
                    track_buf.fill_(1.0)
                    with torch.no_grad():
                        update_best(objective(pose_theta(), t))
                track_buf.fill_(0.0)
            else:
                track_buf.fill_(1.0 if nominal else 0.0)
            run_step(smp)
    if args.steps in snap_steps:
        snapshots.append((args.steps, pose_theta().detach().clone(), t.detach().clone()))
    with torch.no_grad():
        # the final parameters are also a candidate (scored with the blur / PSF of the last step, so that a constant
        # --blur-frac 0 blur is comparable to the tracked steps); they are the only one when an annealing schedule
        # (e.g. --psf-anneal-frac > 1) outlasts the run, as nothing was tracked then
        if (stochastic and track_from <= args.steps) or track_from >= args.steps:
            step_buf.fill_(float(args.steps))
            track_buf.fill_(1.0)
            update_best(objective(pose_theta(), t))
    for _, smp in stages:
        smp.set_blur(0.0)
        smp.set_psf_scale(1.0)
    return dict(theta_deg=best_th.clone(), t=best_t.clone(), best_step=int(best_step.item()), snapshots=snapshots)


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
                      "ncc", "lncc", "nmi", "seg_dice", "hd_mm"])
HISTORY_STEP_COLUMNS = ["subject_id", "step", "fraction", "is_best", "objective", "ncc_mean", "lncc_mean", "nmi_mean",
                        "seg_dice_mean",
                        "hd_mean_mm", "hd_max_mm", "sax_shift_rms_mm", "sax_shift_jitter_mm", "sax_dz_rms_mm", "sax_tilt_rms_deg"]
SAX_LINE_COLUMNS = ["subject_id", "step", "fraction", "is_best", "image_a", "slice_a", "image_b", "slice_b", "kind",
                    "crossing", "n_points", "supervised", "axis_dist_mm", "ncc", "dissim", "gap_min_mm", "rel_tilt_deg",
                    "crossing_in_heart", "x0_x", "x0_y", "x0_z", "d_x", "d_y", "d_z",
                    "t_used_lo", "t_used_hi", "t_img_lo", "t_img_hi"]
LONG_AXIS_COLUMNS = ["subject_id", "step", "fraction", "is_best", "p_x", "p_y", "p_z", "d_x", "d_y", "d_z", "radius_mm"]


def build_history(sid, data, planes, n_pairs, sampler, snapshots, best_step, lax_rows, args, labels, labels_sax, has_seg) -> dict:
    """Affines, poses and metrics of every pose snapshot. Snapshot step 'best' is the pose that is saved/reported,
    re-labelled with the step it was found at; step 0 is the acquired geometry (identity pose). With --sax-sax-weight
    also the neighbouring-SAX-slice crossing lines of every snapshot ('sax_lines', see SaxNeighbourLines.segment_rows)."""
    nan = float("nan")
    plane_rows, step_rows, sax_line_rows, axis_rows = [], [], [], []
    for step, th, tt in snapshots:
        is_best = step == "best"
        step_no = best_step if is_best else step
        if sampler.sax_pairs is not None:
            shift = tt.detach().cpu().numpy().astype(np.float64)[lax_rows].mean(0) if lax_rows else np.zeros(3)
            for r in sampler.sax_pairs.segment_rows(th, tt, shift):
                sax_line_rows.append(dict(subject_id=sid, step=step_no, fraction=step_no / args.steps, is_best=int(is_best),
                                          image_a="SAX", image_b="SAX", kind="SAX-SAX", **r))
            sp = sampler.sax_pairs
            L0 = sp.L0.cpu().numpy().astype(np.float64) - shift      # same frame as the sax_lines rows
            ell = sp.ell.cpu().numpy().astype(np.float64)
            axis_rows.append(dict(subject_id=sid, step=step_no, fraction=step_no / args.steps, is_best=int(is_best),
                                  p_x=float(L0[0]), p_y=float(L0[1]), p_z=float(L0[2]),
                                  d_x=float(ell[0]), d_y=float(ell[1]), d_z=float(ell[2]), radius_mm=sp.radius))
        lm = sampler.evaluate(th, tt, args, labels, labels_sax, has_seg)
        with torch.no_grad():
            # the optimized similarity on the mask points
            sim = (lm["nmi"] if args.metric == "nmi" else lm["ncc"] if args.metric == "ncc"
                   else lm["lncc"] if args.metric == "lncc" else sampler.similarity_lines(th, tt, args, W=sampler.W))
            objective = float(similarity_objective(sampler, th, tt, args, sim=sim))
        per = {k: (sampler.per_plane_max if k == "hd" else sampler.per_plane)(lm[k])
               for k in ("ncc", "lncc", "nmi", "dice", "hd")}
        theta = th.detach().cpu().numpy().astype(np.float64)
        t_rel = relative_translation(tt, lax_rows)
        Rs = rotation_matrices(theta)
        sax_vec, sax_dz, sax_tilt = [], [], []
        for i, p in enumerate(planes):
            A = corrected_affine(base_affine(data, p), p, Rs[i], t_rel[i])
            dx, dy, dz = float(t_rel[i] @ p.u_hat), float(t_rel[i] @ p.v_hat), float(t_rel[i] @ p.normal)
            inplane = float(theta[i] @ p.normal)
            tilt = float(np.linalg.norm(theta[i] - inplane * p.normal))
            if p.image == "SAX" and n_pairs[i] > 0:
                sax_vec.append((dx, dy)), sax_dz.append(dz), sax_tilt.append(tilt)
            plane_rows.append(dict(subject_id=sid, step=step_no, fraction=step_no / args.steps, is_best=int(is_best),
                                   image=p.image, slice=p.slice_idx,
                                   **{c: float(A[int(c[1]), int(c[2])]) for c in AFFINE_COLUMNS},
                                   dx_mm=dx, dy_mm=dy, dz_mm=dz, rot_x_deg=float(theta[i, 0]), rot_y_deg=float(theta[i, 1]),
                                   rot_z_deg=float(theta[i, 2]), rot_inplane_deg=inplane, rot_tilt_deg=tilt,
                                   ncc=per["ncc"][i] if n_pairs[i] else nan, lncc=per["lncc"][i] if n_pairs[i] else nan,
                                   nmi=per["nmi"][i] if n_pairs[i] else nan,
                                   seg_dice=per["dice"][i] if n_pairs[i] else nan, hd_mm=per["hd"][i] if n_pairs[i] else nan))
        vec = np.array(sax_vec).reshape(-1, 2)
        ok = lambda x: np.asarray(x, dtype=float)[np.isfinite(np.asarray(x, dtype=float))]  # noqa: E731
        hd_lines = ok(lm["hd"].cpu().numpy())
        step_rows.append(dict(
            subject_id=sid, step=step_no, fraction=step_no / args.steps, is_best=int(is_best), objective=objective,
            ncc_mean=float(np.nanmean(per["ncc"])), lncc_mean=float(np.nanmean(per["lncc"])),
            nmi_mean=float(np.nanmean(per["nmi"])),
            seg_dice_mean=float(np.nanmean(per["dice"])) if np.isfinite(per["dice"]).any() else nan,
            hd_mean_mm=float(hd_lines.mean()) if hd_lines.size else nan, hd_max_mm=float(hd_lines.max()) if hd_lines.size else nan,
            sax_shift_rms_mm=float(np.sqrt((vec ** 2).sum(1).mean())) if len(vec) else nan,
            sax_shift_jitter_mm=float(np.sqrt(((vec - vec.mean(0)) ** 2).sum(1).mean())) if len(vec) else nan,
            sax_dz_rms_mm=float(np.sqrt(np.mean(np.square(sax_dz)))) if sax_dz else nan,
            sax_tilt_rms_deg=float(np.sqrt(np.mean(np.square(sax_tilt)))) if sax_tilt else nan))
    return dict(planes=plane_rows, steps=step_rows, sax_lines=sax_line_rows, long_axis=axis_rows)


def annealed_stages(data, planes, sax_planes, lax_planes, final_sampler, args, device) -> list | None:
    """(start_step, LineSampler) stages for mask-dilation annealing, ending with the nominal sampler.

    The mask dilation goes linearly from --mask-dilation-start-mm to --mask-dilation-mm over the first
    --mask-anneal-frac of the run, in --anneal-stages discrete steps; the lines are rebuilt at each step.
    Returns None when mask annealing is off.
    """
    dil0 = getattr(args, "mask_dilation_start_mm", None)
    n_stages = max(1, getattr(args, "anneal_stages", 6))
    anneal_end = int(round(getattr(args, "mask_anneal_frac", 0.8) * args.steps)) if dil0 is not None else 0
    if anneal_end <= 0:
        return None
    starts = sorted({int(round(k / n_stages * anneal_end)) for k in range(n_stages)})
    dists = {v: torch.from_numpy(d["dist"]).to(device) for v, d in data["lax"].items() if d.get("dist") is not None}
    if not dists:                                        # no segmented image: no masks to anneal
        return None
    nominal_masks = {p.key: p.mask for p in lax_planes}
    stages = []
    try:
        for k, start in enumerate(starts):
            dil = dil0 + (args.mask_dilation_mm - dil0) * k / n_stages
            for p in lax_planes:
                if p.image in dists:                             # (unsegmented images stay unmasked)
                    p.mask = (dists[p.image][:, :, p.slice_idx] <= dil).float()[None, None]
            lines = build_lines(planes, args, device)
            if lines:
                smp = new_sampler(planes, lines, args, device, images=final_sampler.img)
                attach_sax_pairs(smp, planes, sax_planes, lax_planes, args, device)
                stages.append((start, smp))
    finally:
        for p in lax_planes:
            p.mask = nominal_masks[p.key]
    return stages + [(anneal_end, final_sampler)]


def process_subject(subject: Subject, args, device, on_lines=None) -> tuple:
    """Align one subject (its slices as handed over by a loader, see SliceImage). Returns the summary row, the
    per-plane rows, the per-line-pair rows, the per-plane poses (plane, R, t; see corrected_affines) and the
    pose/metric history (dict with 'planes' and 'steps' row lists, empty unless args.history_every > 0, i.e. with --logging).

    `on_lines(planes, lines)` is called once the intersection lines are known, before alignment.
    Without args.logging only the poses are computed: no metrics are evaluated, the segmentations stay off the GPU and
    the summary / per-pair rows and the history are None / empty (the per-plane rows only hold n_pairs).
    """
    data = prepare_subject(subject, args)
    sid = data["subject_id"]
    nan = float("nan")
    log = getattr(args, "logging", True)
    sax_planes, lax_planes = build_planes(data, args, device, with_seg=log)
    planes = sax_planes + lax_planes
    index = {p.key: i for i, p in enumerate(planes)}

    lines, n_pairs = build_lines(planes, args, device), [0] * len(planes)
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

    N = len(planes)
    if not log:                                      # alignment only
        theta, t_rel = np.zeros((N, 3)), np.zeros((N, 3))
        if lines:
            sampler = new_sampler(planes, lines, args, device)
            attach_sax_pairs(sampler, planes, sax_planes, lax_planes, args, device)
            result = align(sampler, args, annealed_stages(data, planes, sax_planes, lax_planes, sampler, args, device),
                           pose_model=model)
            theta = result["theta_deg"].cpu().numpy().astype(np.float64)
            t_rel = relative_translation(result["t"], [i for i, p in enumerate(planes) if p.image != "SAX" and n_pairs[i] > 0])
        Rs = rotation_matrices(theta)
        details = [dict(subject_id=sid, image=p.image, slice=p.slice_idx, n_pairs=n_pairs[i]) for i, p in enumerate(planes)]
        return None, details, [], [(p, Rs[i], t_rel[i]) for i, p in enumerate(planes)], None

    metric_names = ("ncc", "nmi", "dice", "hd")
    plane_m = {f"{k}_{s}": [nan] * N for k in metric_names for s in ("zero", "final")}
    pairs = []
    sax_qc = dict(sax_cross_pairs=nan, sax_gap_min_mm=nan, sax_rel_tilt_max_deg=nan, sax_cross_dissim_max=nan)
    history = dict(planes=[], steps=[])
    n_points = [0.0] * N
    t_rel = np.zeros((N, 3))
    theta = np.zeros((N, 3))
    if lines:
        sampler = new_sampler(planes, lines, args, device)
        attach_sax_pairs(sampler, planes, sax_planes, lax_planes, args, device)
        n_points = [float(sampler.npts[idx].sum()) for idx in sampler.plane_lines]
        # segmentation labels compared by the Dice / Hausdorff metrics (loader options; none = no label metrics)
        labels = [int(x) for x in getattr(args, "seg_labels", "").split(",") if x.strip()]
        labels_sax = [int(x) for x in getattr(args, "sax_seg_labels", "").split(",") if x.strip()]
        has_seg = any(p.seg is not None for p in planes)
        zeros = torch.zeros(N, 3, device=device)
        line_zero = sampler.evaluate(zeros, zeros, args, labels, labels_sax, has_seg)
        result = align(sampler, args, annealed_stages(data, planes, sax_planes, lax_planes, sampler, args, device),
                       pose_model=model)
        line_final = sampler.evaluate(result["theta_deg"], result["t"], args, labels, labels_sax, has_seg)
        for stage, lm in (("zero", line_zero), ("final", line_final)):
            for k in metric_names:
                agg = sampler.per_plane_max if k == "hd" else sampler.per_plane      # Hausdorff: worst line per plane
                plane_m[f"{k}_{stage}"] = agg(lm[k])
        lz = {k: v.cpu().numpy() for k, v in line_zero.items()}
        lf = {k: v.cpu().numpy() for k, v in line_final.items()}
        for m, (ia, ib, line) in enumerate(lines):
            pa, pb = planes[ia], planes[ib]
            pairs.append(dict(subject_id=sid, image_a=pa.image, slice_a=pa.slice_idx, image_b=pb.image, slice_b=pb.slice_idx,
                              kind="LAX-LAX" if pa.image != "SAX" and pb.image != "SAX" else "SAX-LAX", n_points=int(line.n_points),
                              ncc_zero=float(lz["ncc"][m]), ncc=float(lf["ncc"][m]), nmi_zero=float(lz["nmi"][m]), nmi=float(lf["nmi"][m]),
                              seg_dice_zero=float(lz["dice"][m]), seg_dice=float(lf["dice"][m]),
                              hd_zero_mm=float(lz["hd"][m]), hd_mm=float(lf["hd"][m]), axis_dist_mm=nan,
                              gap_min_mm=nan, rel_tilt_deg=nan))
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
                pairs.append(dict(subject_id=sid, image_a="SAX", slice_a=planes[ia].slice_idx, image_b="SAX", slice_b=planes[ib].slice_idx,
                                  kind="SAX-SAX", n_points=round(float(nf[k]), 2), ncc_zero=float(cz[k]) if nz[k] > 1e-3 else nan,
                                  ncc=float(cf[k]) if nf[k] > 1e-3 else nan, nmi_zero=nan, nmi=nan, seg_dice_zero=nan, seg_dice=nan,
                                  hd_zero_mm=nan, hd_mm=nan, axis_dist_mm=float(df[k]),
                                  gap_min_mm=float(gap_min[k]), rel_tilt_deg=float(tilt[k])))
            crossing = gap_min < 0
            dissim = 1.0 - cf[crossing & (nf > 1e-3)]
            sax_qc = dict(sax_cross_pairs=int(crossing.sum()), sax_gap_min_mm=float(gap_min.min()) if gap_min.size else nan,
                          sax_rel_tilt_max_deg=float(tilt.max()) if tilt.size else nan,
                          sax_cross_dissim_max=float(dissim.max()) if dissim.size else nan)
        theta = result["theta_deg"].cpu().numpy().astype(np.float64)
        lax_rows = [i for i, p in enumerate(planes) if p.image != "SAX" and n_pairs[i] > 0]
        t_rel = relative_translation(result["t"], lax_rows)
        if result["snapshots"]:
            snaps = result["snapshots"] + [("best", result["theta_deg"], result["t"])]
            history = build_history(sid, data, planes, n_pairs, sampler, snaps, result["best_step"], lax_rows,
                                    args, labels, labels_sax, has_seg)
    Rs = rotation_matrices(theta)
    poses = [(p, Rs[i], t_rel[i]) for i, p in enumerate(planes)]

    details = []
    for i, p in enumerate(planes):
        row = dict(subject_id=sid, image=p.image, slice=p.slice_idx, n_pairs=n_pairs[i], n_points=int(n_points[i]),
                   dx_mm=nan, dy_mm=nan, dz_mm=nan, shift_inplane_mm=nan, shift_mm=nan, rot_inplane_deg=nan,
                   rot_tilt_deg=nan, ncc_zero=nan, ncc=nan, nmi_zero=nan, nmi=nan, seg_dice_zero=nan, seg_dice=nan,
                   hd_zero_mm=nan, hd_mm=nan)
        if n_pairs[i] > 0:
            tv, th = t_rel[i], theta[i]
            dx, dy, dz = float(tv @ p.u_hat), float(tv @ p.v_hat), float(tv @ p.normal)
            inplane = float(th @ p.normal)
            row.update(dx_mm=dx, dy_mm=dy, dz_mm=dz, shift_inplane_mm=math.hypot(dx, dy), shift_mm=float(np.linalg.norm(tv)),
                       rot_inplane_deg=inplane, rot_tilt_deg=float(np.linalg.norm(th - inplane * p.normal)),
                       ncc_zero=plane_m["ncc_zero"][i], ncc=plane_m["ncc_final"][i],
                       nmi_zero=plane_m["nmi_zero"][i], nmi=plane_m["nmi_final"][i],
                       seg_dice_zero=plane_m["dice_zero"][i], seg_dice=plane_m["dice_final"][i],
                       hd_zero_mm=plane_m["hd_zero"][i], hd_mm=plane_m["hd_final"][i])
        details.append(row)

    def arr(rows, fn):
        return np.array([fn(d) for d in rows], dtype=float)

    def stat(fn, x):
        x = x[np.isfinite(x)] if x.size else x
        return float(fn(x)) if x.size else nan

    def rms(x):
        return np.sqrt(np.mean(x ** 2))

    sax_rows = [d for d in details if d["image"] == "SAX" and d["n_pairs"] > 0]
    lax_rows = [d for d in details if d["image"] != "SAX" and d["n_pairs"] > 0]
    ok_rows = [d for d in details if d["n_pairs"] > 0]
    vec = np.array([[d["dx_mm"], d["dy_mm"]] for d in sax_rows]).reshape(-1, 2)
    mag = np.hypot(vec[:, 0], vec[:, 1])
    jit = np.hypot(*(vec - vec.mean(0)).T) if len(vec) else np.array([])
    nz, nf_ = arr(ok_rows, lambda d: d["ncc_zero"]), arr(ok_rows, lambda d: d["ncc"])
    mz, mf = arr(ok_rows, lambda d: d["nmi_zero"]), arr(ok_rows, lambda d: d["nmi"])

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
        nmi_zero_mean=stat(np.mean, mz), nmi_mean=stat(np.mean, mf), nmi_gain_mean=stat(np.mean, mf - mz),
        seg_dice_zero_mean=stat(np.mean, arr(ok_rows, lambda d: d["seg_dice_zero"])),
        seg_dice_mean=stat(np.mean, arr(ok_rows, lambda d: d["seg_dice"])),
        hd_zero_mean_mm=stat(np.mean, arr(pairs, lambda d: d["hd_zero_mm"])), hd_mean_mm=stat(np.mean, arr(pairs, lambda d: d["hd_mm"])),
        hd_zero_max_mm=stat(np.max, arr(pairs, lambda d: d["hd_zero_mm"])), hd_max_mm=stat(np.max, arr(pairs, lambda d: d["hd_mm"])),
        **sax_qc, n_line_pairs=sum(d["kind"] != "SAX-SAX" for d in pairs), steps=args.steps if lines else 0,
    )
    return summary, details, pairs, poses, history


RELMOTION_COLUMNS = ["subject_id", "kind", "slice_a", "slice_b", "n_pairs", "dx_mm", "dy_mm", "dz_mm",
                     "shift_inplane_mm", "shift_mm", "rot_inplane_deg", "rot_tilt_deg", "rot_deg"]


def sax_relative_motion(sid: str, poses: list, n_pairs: list[int]) -> list[dict]:
    """Estimated SAX motion relative to the other slices (independent of any motion the whole stack shares, e.g. a
    misplaced LAX reference). Pose i maps x -> R_i (x - c_i) + c_i + t_i (c_i: the slice's rotation centre).

    kind 'slice' (slice_a): the slice against the stack consensus -- at the slice's centre, the median over all
      evaluated slices j of the displacement pose j would give it; rotation against the median rotation.
    kind 'pair' (slice_a, slice_b = a + 1): slice b's pose relative to slice a's, T_a^-1 T_b, at b's centre.
    Translations are split in the frame of the slice concerned (dx, dy in-plane along its rows / columns, dz along
    its normal); rotations into in-plane (about the normal, signed) and tilt. Slices without intersection lines
    (pose not estimated) are left out.
    """
    from scipy.spatial.transform import Rotation
    sax = sorted((p.slice_idx, p, np.asarray(R, float), np.asarray(t, float), n_pairs[i])
                 for i, (p, R, t) in enumerate(poses) if p.image == "SAX" and n_pairs[i] > 0)
    rows = []
    if not sax:
        return rows

    def row(kind, a, b, p, npairs, d, R):
        rv = np.degrees(Rotation.from_matrix(R).as_rotvec())
        dx, dy, dz = float(d @ p.u_hat), float(d @ p.v_hat), float(d @ p.normal)
        inplane = float(rv @ p.normal)
        return dict(subject_id=sid, kind=kind, slice_a=a, slice_b=b, n_pairs=npairs, dx_mm=dx, dy_mm=dy, dz_mm=dz,
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
    "LAX_2Ch": ("LAX 2Ch view", lambda d: d["image"] == "LAX_2Ch"),
    "LAX_3Ch": ("LAX 3Ch view", lambda d: d["image"] == "LAX_3Ch"),
    "LAX_4Ch": ("LAX 4Ch view", lambda d: d["image"] == "LAX_4Ch"),
    "SAX_basal": (f"basal SAX slices (index < {BASAL_SLICES})", lambda d: (d["image"] == "SAX") & (d["region"] == "basal")),
    "SAX_mid": ("mid-ventricular SAX slices", lambda d: (d["image"] == "SAX") & (d["region"] == "mid")),
    "SAX_apical": (f"apical SAX slices (last {APICAL_SLICES})", lambda d: (d["image"] == "SAX") & (d["region"] == "apical")),
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
    for col in ("nmi_zero", "nmi", "hd_zero_mm", "hd_mm"):  # runs from before these metrics were added
        if col not in det:
            det[col] = np.nan
    det = det[det["subject_id"].isin(summ["subject_id"])].reset_index(drop=True)
    n_slices = det[det["image"] == "SAX"].groupby("subject_id")["slice"].max() + 1
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
    pairs = pairs[pairs["kind"] != "SAX-SAX"]
    pairs = pairs.drop_duplicates(["subject_id", "image_a", "slice_a", "image_b", "slice_b"], keep="last")
    pairs = pairs[pairs["subject_id"].isin(det["subject_id"])].reset_index(drop=True)
    regions = det[det["image"] == "SAX"].set_index(["subject_id", "slice"])["region"]
    key = pd.MultiIndex.from_arrays([pairs["subject_id"], pairs["slice_a"]])
    pairs["region"] = np.where(pairs["image_a"] == "SAX", regions.reindex(key).to_numpy(), "LAX")
    return pairs


def select_pairs(pairs: pd.DataFrame, subset: str) -> pd.DataFrame:
    """Pairs involving the planes of a SUBSETS entry (a LAX view, or a SAX region)."""
    if subset.startswith("LAX"):
        return pairs[(pairs["image_a"] == subset) | (pairs["image_b"] == subset)]
    return pairs[(pairs["image_a"] == "SAX") & (pairs["region"] == subset.split("_", 1)[1])]


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
        images = (["SAX"] if "SAX" in names else []) + [im for im in names if im != "SAX"]
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

    has_sax = bool((det["image"] == "SAX").any())
    hex_rows = det[det["image"] == "SAX"] if has_sax else det
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

    sax = det[det["image"] == "SAX"]
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
    sax, lax = det[det["image"] == "SAX"], det[det["image"] != "SAX"]
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
        if p.image != "SAX" and p.mask is not None:
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
        kind = "SAX-LAX" if "SAX" in (planes[ia].image, planes[ib].image) else "LAX-LAX"
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
                         kind="SAX-LAX" if "SAX" in (a.image, b.image) else "LAX-LAX", n_points=int(line.n_points),
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
                         history: dict | None = None, st=PLOT_STYLE) -> None:
    """Per-subject CSVs (summary row, per-plane rows, per-line-pair rows, pose/metric history) and figures."""
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
    det = pd.DataFrame(details, columns=DETAIL_COLUMNS)
    det = det[det["n_pairs"] > 0]
    pdf = pd.DataFrame(pairs, columns=PAIR_COLUMNS)
    pdf = pdf[pdf["kind"] != "SAX-SAX"]                 # the figures cover the SAX-LAX / LAX-LAX lines
    if not plots or det.empty:
        return
    if len(pdf):
        plot_pairs_figure(pdf, f"subject {summary['subject_id']}", metrics_dir / "motion_pairs.png")
    from matplotlib.figure import Figure
    from matplotlib.ticker import AutoLocator

    sax, lax = det[det["image"] == "SAX"], det[det["image"] != "SAX"]
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
                 "device", "workers"}
HISTORY_EVERY = 0.02    # with --logging: pose/metric snapshot every this fraction of the run (plus step 0 and the best pose)
RESCAN_SECONDS = 30.0   # how often --root is rescanned for new subjects while others are still queued


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
    """The algorithm and run-loop options of the motion correction (optimizer, similarity, regularizers, metrics,
    logging). A data loader's entry script adds the rest -- the paths (--root, --out-root, --run-name), the run
    options --max-subjects, --skip-processed and --logging, its input/output formats (slice thickness, --image-dtype) and
    the hardware options (--device, --workers, --no-cuda-graph, --sample-dtype) -- and calls run()."""
    ap = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    # ###################    RUN PARAMS    ###############################################
    ap.add_argument("--n-frames", type=int, default=10,
                    help="Number of cardiac phases (evenly spaced) used for the alignment. Default: %(default)s")
    ap.add_argument("--save-images", type=str2bool, default=True, metavar="BOOL",
                    help="false: do not write the motion-corrected images (e.g. to only estimate the motion of a cohort "
                         "with --logging false: then just sax_relative_motion.csv is written)")

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
    ap.add_argument("--lr-schedule", choices=["linear", "constant"], default="constant",
                    help="linear: decay each learning rate linearly from its start value to its end value; constant: no decay")
    ap.add_argument("--lr-end", type=float, default=0.0, help="Final translation learning rate for the linear schedule")
    ap.add_argument("--rot-lr-end", type=float, default=0.0, help="Final rotation learning rate for the linear schedule")
    ap.add_argument("--pair-keep", type=float, default=0.8,
                    help="Stochastic optimization: fraction of intersection lines used at each step (1 = deterministic). "
                         "Each step draws a new random subset of round(fraction x lines) lines and samples only those, "
                         "so GPU memory and time per step scale with this fraction (the full objective for tracking "
                         "the best pose is evaluated without gradients, in chunks)")
    ap.add_argument("--eval-every", type=int, default=10, help="With --pair-keep < 1: evaluate the full objective every N steps to track the best pose")
    ap.add_argument("--seed", type=int, default=0, help="Seed for the line dropping")

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
                         "SAX slices with one common scale). Default: %(default)s")
    ap.add_argument("--lncc-window-mm", type=float, default=20.0, help="--metric lncc: window length (mm). Default: %(default)s")
    ap.add_argument("--lncc-min-std", type=float, default=0.1,
                    help="--metric lncc: windows whose profiles vary less than this (geometric mean of both standard "
                         "deviations, in units of the profile's standard deviation over the line) are faded out "
                         "(weight sd / this). Default: %(default)s")
    ap.add_argument("--grad-eps", type=float, default=0.5,
                    help="--metric ngf: edge threshold, in standard deviations of the profile's gradient along the line; "
                         "gradients well above it count as full edges, those below it fade out. Default: %(default)s")
    ap.add_argument("--nmi-bins", type=int, default=24, help="Histogram bins for mutual information")
    ap.add_argument("--nmi-sigma", type=float, default=1.0, help="Parzen window width for mutual information, in bins")

    # ###################    COARSE-TO-FINE (BLUR) PARAMS    #############################
    ap.add_argument("--blur-sigma", type=float, default=3.0,
                    help="Coarse-to-fine: initial in-plane Gaussian sigma (mm) of the images, decayed linearly to 0 (0 = off)")
    ap.add_argument("--blur-frac", type=float, default=0.9,
                    help="Fraction of the steps over which the blur decays to 0 (> 1: it never reaches 0); 0 = no decay, "
                         "the blur stays at --blur-sigma for the whole run")

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

    # ###################    POSE REGULARIZATION PARAMS    ###############################
    ap.add_argument("--gauge", type=float, default=1e-2, help="Penalty weight on the mean pose (removes global drift)")
    ap.add_argument("--prior", type=float, default=0.0, help="Penalty weight on rotation (deg^2) and through-plane shift (mm^2)")
    ap.add_argument("--tilt-prior", type=float, default=0.0,
                    help="Penalty weight on each plane's own tilt (rotation about its in-plane axes, as rot_tilt_deg in the "
                         "outputs) above --tilt-hinge-deg: relu(tilt - threshold)^2 (deg^2), mean over planes. In-plane "
                         "rotation and translations are not penalized; tilts up to the threshold cost nothing (0 = off)")
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

    # ###################    SAX-SAX INTERACTION PARAMS    ###############################
    ap.add_argument("--parallel-max-dist-mm", type=float, default=12.0,
                    help="Neighbouring SAX slices for the SAX-SAX terms below (and their QC logging): slices that are "
                         "parallel as acquired (closer than --min-pair-angle-deg) and at most this far apart along the "
                         "normal. Slices further apart are never checked for crossing. UKBB (10 mm spacing): each slice "
                         "and the next; a stack with 1-3 mm spacing: all slices within this distance. Default: %(default)s")
    ap.add_argument("--sax-gap-reg", type=float, default=0.0,
                    help="Barrier keeping neighbouring SAX slices from approaching or crossing each other inside the heart: "
                         "on a disk of --sax-gap-radius-mm around the LV long axis (the crossing line of the two most "
                         "perpendicular LAX views, else the SAX stack's axis) on each slice, the gap to the neighbouring "
                         "slice along the acquired stacking direction (~10 mm as acquired) is penalized below the margin "
                         "(--sax-gap-margin-mm, at most half the pair's acquired gap): mean softplus(margin - gap)^2 "
                         "(mm^2). Zero for ordinary poses, smooth, no image sampling (0 = off; e.g. 1e-2 costs 0.16 at a "
                         "0 mm gap). Default: %(default)s")
    ap.add_argument("--sax-gap-margin-mm", type=float, default=4.0,
                    help="Smallest gap (mm) between neighbouring SAX slices allowed without cost by --sax-gap-reg (for "
                         "each pair at most half its acquired gap). Default: %(default)s")
    ap.add_argument("--sax-gap-radius-mm", type=float, default=120.0,
                    help="Radius (mm) of the disk around the long axis on which --sax-gap-reg measures the gap (5 mm grid). "
                         "Default: %(default)s")
    ap.add_argument("--sax-sax-weight", type=float, default=0.0,
                    help="Dissimilarity cost of neighbouring SAX slices (z and z + 1) crossing each other: where the poses "
                         "make them cross, their images are compared along the crossing line with NCC and "
                         "weight x sum(point weights x (1 - NCC)) / (number of SAX-LAX/LAX-LAX line points) is subtracted "
                         "from the objective, so a crossing never pays and a mismatched one costs most. The point weights "
                         "fall off with the distance from the LV long axis (--sax-sax-falloff, --sax-sax-radius-mm); only "
                         "points inside both images count. The line is recomputed from the poses every step. Needs "
                         "rotations (--dof 6) and two LAX views (0 = off). Default: %(default)s")
    ap.add_argument("--sax-sax-falloff", choices=["gaussian", "threshold"], default="gaussian",
                    help="Point weights of --sax-sax-weight along the crossing line. gaussian: exp(-r^2 / (2 sigma^2)) of the "
                         "distance r from the long axis, sigma = --sax-sax-radius-mm; smooth, so a mismatched crossing is "
                         "pushed out of the heart. threshold: 1 within --sax-sax-radius-mm, 0 beyond (no such push). "
                         "Default: %(default)s")
    ap.add_argument("--sax-sax-radius-mm", type=float, default=120.0,
                    help="--sax-sax-falloff "
                         "gaussian: its sigma (mm); "
                         "threshold: the radius (mm) around the long axis. Also "
                         "the extent of the 'used' part of the lines in sax_lines.csv. Default: %(default)s")


    return ap



def run(ap: argparse.ArgumentParser, args, load, save, is_complete) -> None:
    """Align every subject folder under args.root, writing to args.out_root / (args.run_name or a generated name)
    (see the module docstring). args.root / out_root / run_name are the entry script's path options.

    The data loader supplies:
      load(subj_dir, args) -> Subject              read one subject folder (called in background threads)
      save(subject, affines, out_dir, args)        write the subject's slices with the given 4x4 affines (one per
                                                   slice, in subject.slices order) into out_dir, in its own format
      is_complete(subj_dir, out_dir) -> bool       whether out_dir already holds every output (--skip-processed)
    """
    if args.no_psf:                          # one band sample at offset 0 with weight 1 = the raw line intensities
        args.n_band, args.psf_scale_start = 1, 1.0
    lr0, rot_lr0 = DEFAULT_LR[args.optimizer]
    args.lr = lr0 if args.lr is None else args.lr
    args.rot_lr = rot_lr0 if args.rot_lr is None else args.rot_lr
    args.history_every = HISTORY_EVERY if args.logging else 0.0     # pose snapshots during optimization

    run_dir = args.out_root / (args.run_name or default_run_name(ap, args, sys.argv[1:]))
    overall_dir = run_dir / "overall_metrics"
    summary_path = overall_dir / "motion_summary.csv"
    detail_path = overall_dir / "motion_summary_slices.csv"
    pairs_path = overall_dir / "motion_summary_pairs.csv"

    device = torch.device(args.device)
    seen: set[str] = set()
    budget = args.max_subjects                            # subjects still allowed by --max-subjects (-1 = no limit)

    def scan() -> list[Path]:
        """Subject folders under --root not listed before, without those already complete (--skip-processed), within
        --max-subjects. Hidden folders are ignored (a tool copying subjects in can write them as .<subject> and rename
        them when complete, so a subject is only picked up once fully written)."""
        nonlocal budget
        new = sorted(d for d in args.root.iterdir() if d.is_dir() and not d.name.startswith(".") and d.name not in seen)
        seen.update(d.name for d in new)
        if args.skip_processed:
            n0 = len(new)
            new = [s for s in new if not is_complete(s, run_dir / s.name)]
            if n0 > len(new):
                print(f"Skipping {n0 - len(new)} subjects whose aligned images are complete")
        if budget >= 0:
            new, budget = new[:budget], budget - min(budget, len(new))
        return new

    subjects = scan()
    print(f"Aligning {len(subjects)} subjects on {device}; outputs in {run_dir}"
          + ("" if args.logging else " (aligned images only; --logging for metrics, CSVs and figures)"))
    if not subjects:
        return
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
        writers = [open_csv(summary_path, SUMMARY_COLUMNS), open_csv(detail_path, DETAIL_COLUMNS), open_csv(pairs_path, PAIR_COLUMNS)]
    save_jobs = deque()                                   # (subject, description, future) for background writes
    failed, unaligned = [], []
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            it = iter(subjects)
            queue = deque()
            pbar = tqdm(total=len(subjects), unit="subj")
            last_scan = time.time()

            def top_up():
                """Keep 2 x --workers subjects loading ahead. When the known subjects run out, rescan --root for
                subjects written since (e.g. by a concurrently running download): at most every
                RESCAN_SECONDS while others are still queued, always once the queue is empty."""
                nonlocal it, last_scan
                while len(queue) < 2 * args.workers:
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
                    queue.append((nxt, ex.submit(load, nxt, args)))

            top_up()
            while queue:
                subj, fut = queue.popleft()
                top_up()
                t0 = time.time()
                metrics_dir = run_dir / subj.name / "metrics"
                summary, details, pairs, history = None, [], [], None

                def on_lines(planes, lines, sid=subj.name, out=metrics_dir):
                    if not lines:
                        return
                    out.mkdir(parents=True, exist_ok=True)
                    save_line_geometry(planes, lines, out, args.line_extent_mm, args.line_step_mm)
                    viz = mask_plot_data(planes, lines, args.line_extent_mm, args.line_step_mm)
                    save_jobs.append((sid, "mask plot", ex.submit(save_mask_plot, viz, sid, args.mask_dilation_mm,
                                                                  out / "mask_overlay.png", args.line_extent_mm)))

                status = "ok"
                try:
                    subject = fut.result()
                    status = subject.status if subject.slices else "no_images"
                    poses = []
                    if status == "ok":
                        summary, details, pairs, poses, history = process_subject(
                            subject, args, device, on_lines=on_lines if args.logging else None)
                        if not any(d["n_pairs"] > 0 for d in details):
                            status = "not_optimized"                 # no crossing planes: written unaligned
                        for row in sax_relative_motion(subj.name, poses, [d["n_pairs"] for d in details]):
                            relmotion_writer.writerow(fmt_row(row))
                        relmotion_f.flush()
                    if args.save_images:          # whatever could be loaded, aligned or not (folder created regardless)
                        out_dir = run_dir / subj.name
                        out_dir.mkdir(parents=True, exist_ok=True)
                        save_jobs.append((subj.name, "aligned images",
                                          ex.submit(save, subject, corrected_affines(subject, poses), out_dir, args)))
                except Exception as e:  # noqa: BLE001
                    traceback.print_exc()
                    status = f"error: {type(e).__name__}: {e}"[:200]
                if status in ("not_optimized", "no_images"):
                    unaligned.append((subj.name, status))
                elif status != "ok":
                    failed.append((subj.name, status))
                if args.logging:
                    summary = summary if summary is not None else empty_summary(subj.name, status)
                    summary["seconds"] = round(time.time() - t0, 2)
                    summary["timestamp"] = datetime.now().isoformat(timespec="seconds")
                    if details:
                        save_jobs.append((subj.name, "subject metrics", ex.submit(save_subject_metrics, dict(summary), details,
                                                                                 pairs, metrics_dir, True, history)))
                    (sf, swriter), (df_, dwriter), (pf, pwriter) = writers
                    swriter.writerow(fmt_row(summary))
                    for row in details:
                        dwriter.writerow(fmt_row(row))
                    for row in pairs:
                        pwriter.writerow(fmt_row(row))
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
                    post = dict(subj=subj.name, status=status[:12])
                    if summary is not None:
                        post.update(ncc=f"{summary['ncc_mean']:.2f}", jitter=f"{summary['sax_shift_jitter_mm']:.1f}")
                    pbar.set_postfix(**post)
                    pbar.update(1)
                if not queue:                              # subjects written while this one was aligned
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
    if unaligned:
        print(f"\n{len(unaligned)} subject(s) without crossing planes, written unaligned: "
              + ", ".join(f"{s} ({st})" for s, st in unaligned))
    if failed:
        print(f"\n{len(failed)} subject(s) failed: " + ", ".join(f"{s} ({st})" for s, st in failed))
    if args.logging:
        print(f"\nSummary written to {summary_path}\nPer-slice results written to {detail_path}\nPer-pair results written to {pairs_path}")
        try:
            n = len(plot_results(summary_path, detail_path, overall_dir, pairs_path))
            print(f"{n} overall figures written to {overall_dir}")
        except Exception:  # noqa: BLE001
            print("Overall plotting failed:")
            traceback.print_exc()
    if args.save_images:
        print(f"Aligned images under {run_dir}\\<subject>\\" + (" (and metrics\\)" if args.logging else ""))
    print(f"SAX relative motion: {overall_dir / 'sax_relative_motion.csv'}")


if __name__ == "__main__":
    sys.exit("motion_gradient.py is the motion-correction library; run it through a data entry point:\n"
             "  python moco_nifti.py ...   (UKBB NIfTI: SAX stack + LAX views)\n"
             "  python moco_dicom.py ...   (DICOM series)")
