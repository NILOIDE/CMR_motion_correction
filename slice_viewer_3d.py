"""
3D slice-alignment viewer for UK Biobank cardiac cine MRI motion-correction runs.

Every 2D slice (each short-axis slice of the SA stack and the three long-axis
views) is placed in scanner/world space and rendered as a flat quad carrying the
image intensities, so that the slices intersect where they should.  Segmentations
(nnU-Net) can be overlaid; label 0 is transparent.

The slice *geometry* comes from the optimisation history written by the alignment
script (<run>/<subject>/metrics/history.csv): one affine per plane for step 0
(the acquired geometry), every 2% of the run, and the saved
"best" pose.  A slider scrubs through these snapshots; the header shows the
optimisation step the displayed affines come from, plus the subject-level
metrics of that step from history_summary.csv when present.

The pixel data are read from the run's NIfTIs in <run>/<subject>/ -- with several modalities (acquisitions, see
moco_nifti.py) one folder per modality, <run>/<subject>/<modality>/ (older runs: <run>/<subject>/Instance_<n>/):
    <SA>_slice00.nii.gz, <SA>_slice01...   one file per SA slice
    <LA>_2Ch.nii.gz / <LA>_3Ch / <LA>_4Ch  any other image as <name>.nii.gz (<name>_sliceNN.nii.gz: a stack)
    nnUNet_segs/<same name>_nnUNetSeg.nii.gz
(a single <SA>.nii.gz volume is also accepted; <SA> / <LA> are the alignment script's
on-disk spellings, see ON_DISK). With several modalities the images are named <modality>/<image>, as in the run's
CSVs; the panel lists the slices of one modality at a time (see "Modality tabs" below). A single-frame image
(end-diastole) is shown at every cardiac phase.

Any of the views may be missing. Slices that were not part of the alignment (no rows in
history.csv, e.g. a LA view without a segmentation) are shown at their saved pose for every
snapshot and marked "unaligned" in the slice list. The header notes a restricted pose model
(translation + in-plane rotation only) from metrics/summary.csv; a subject that was not optimized
(no crossing views) is shown at its acquired pose without metrics. Subject folders without
images are skipped.

Usage
-----
    python slice_viewer_3d.py                       # most recently created run, its first subject
    python slice_viewer_3d.py <subject-id>               # a specific subject
    python slice_viewer_3d.py <subject-id> --run steps600-lr0.05-lin-blur4x0.6
    python slice_viewer_3d.py <subject-id> --run-dir D:\\some\\run\\dir
    python slice_viewer_3d.py <subject-id> --start 0     # open at the acquired geometry
    python slice_viewer_3d.py <subject-id> --frame 10 --no-seg
    python slice_viewer_3d.py --remote user@host:/path/to/run_dir --subjects worst_cases/slice_shift_max.txt
                                                    # a run on another machine (ssh, key authentication): each
                                                    # subject is downloaded once into --cache-dir; [ / ] step
                                                    # through the listed subjects in the file's order

Interaction
-----------
    Slider "optimization history" (bottom)   scrub through the pose snapshots
    Slider "cardiac phase" (below it)         time frame of the cine
    Checkboxes (left)   segmentation, outlines, all SA, all LA, every individual slice
    Modality tabs (left, above All SA / All LA; subjects with several modalities): one tab per modality,
                        clicked or stepped through with v / V, picks the modality whose slices the panel lists
                        (All SA / All LA, the digit keys and , / . act on it; picking a slice of another
                        modality to profile switches to its tab). The checkbox left of each tab shows / hides
                        all slices of that modality (key h: the active tab's). Charts name a slice of another
                        modality than the profiled one with its <modality>/ prefix
    Radio buttons       which per-plane metric of history.csv colours the slice outlines and is
                        shown in the intersection profile charts: NCC, LNCC, NMI, Dice or Hausdorff
                        (red = bad, green = good, grey = no value; LNCC needs runs logged after it
                        was added); each row also shows the subject-level value at the shown step
    0 ... 9             toggle the first ten SA slices (in list order)
    Sliders "colour min" / "colour max" (above the colour bar)
                        clamp the colour range: values below the min are full red / full
                        green (whichever is the "bad" end), values above the max the other
                        extreme.  Defaults: the subject's min and max over the history.
    Up / Down           next / previous history snapshot
    Home / b / e, End   jump to step 0 / the best pose / the final snapshot
    Left / Right        previous / next cardiac frame (wraps around: the last frame is
                        followed by the first)
    s / o / a / l       toggle segmentation / outlines / all SA / all LA
    m / i               toggle the dilated-mask contours (metrics/mask_contours.csv) / the
                        intersection sampling lines (metrics/lines.csv); both follow the planes
                        through the history (each line is the intersection of its two planes
                        at the shown step, i.e. the line the loss samples at that step)
                        Sampling lines (also SA-SA) are burned into each slice they belong to:
                        every slice shows its own copy, visible from both sides, hidden with it.
                        SA-LA lines light blue; LA-LA lines on a LA slice in the colour of the
                        view they cross (2Ch amber, 3Ch white, 4Ch pink, Ao aquamarine); lines
                        that coincide on a slice (e.g. 2Ch and 3Ch on the 4Ch view, both near the
                        long axis) as interleaved dashes of their colours
    k                   toggle the SA-SA sampling lines of neighbouring SA slices
                        (runs with the SA-SA term; violet): stored per
                        snapshot, so a pair only shows a line at steps where its slices cross
                        (thin: inside both images, thick: entering the loss, i.e. also within
                        the radius of the long axis)
    u                   toggle the long-axis radius on the SA slices (metrics/long_axis.csv): the
                        ellipse where the cylinder of the SA-SA radius around the LV long axis
                        cuts each slice -- SA-SA points count only inside it
    Buttons < / > next to "subject" (top right), keys ] and [
                        (also PageDown / PageUp): load the next / previous subject of the run
                        (alphabetical) into the same window; toggles, metric and frame are kept
    Buttons < / > next to "run" (top right, above), keys } and { (> = older)
                        step through the runs of the same directory (--align-root, or the
                        parent of --run-dir), newest first; the current subject is kept when
                        the other run has it, else the same position in its subject list. Same
                        subject: its pixels are not read again (only the NIfTI headers, for the
                        run's saved affines, and its history), the slices and the view (camera,
                        shown slices, frame, profiled slice) stay; only overlays and UI are rebuilt
    p / , / .           intersection intensity profiles (right): toggle / previous / next slice;
                        hovering a chart marks that position on the line in 3D (red tick in both
                        slices) and shows both grey values in the chart title;
                        the small radio next to each slice row picks it. An SA slice shows its
                        SA-LA lines (2Ch, 3Ch, 4Ch and, when present, the aortic slice Ao) and its
                        SA-SA crossing with the next slice; an LA view (Ao included) first its
                        crossings with the other LA views, then with every SA slice that crosses
                        it: its loss line, or for pairs without one (e.g. Ao x 4Ch) the planes'
                        crossing at the shown step, "(not in loss)", with no thick part. The
                        intensity and the metric charts each sit in a framed rectangle on the
                        right showing 4 charts; with more intersections, the mouse wheel over
                        either rectangle scrolls both together, smoothly (half a chart per notch;
                        scroll bar on the right edge; the camera does not zoom there). Curves: displayed grey value of both slices along the line
                        at the shown step and frame (thin: inside both images, thick: in the
                        loss; both follow the annealed mask dilation of the shown step, as do
                        the mask contours and the yellow bar); y toggles raw intensities / the loss view: the raw image blurred
                        and averaged over the PSF band across the other slice's slab with the blur
                        and PSF scale of the shown step (from the run's run_config.json; SA-SA
                        lines: blur only); yellow bar along the bottom: inside the LA dilated mask (either
                        LA slice of the line, at its current pose; SA-Ao lines, the aortic slice having no
                        segmentation: inside the SA-SA cylinder around the shown step's long axis, the
                        mask the alignment uses for them); NCC over the loss part,
                        current frame only
    n                   toggle the metric charts below the profile charts: one per intersection, the
                        selected metric per point along the same line (NCC: z_a z_b, mean = NCC;
                        LNCC: local NCC per window; Dice: 1 same label / 0 one only; Hausdorff:
                        distance to the nearest same label in the other profile, max = HD; empty
                        for NMI), drawn as the profiles: thin over the whole intersection (the
                        metric without the mask; NCC z-scored with the loss part's mean / std),
                        thick over the loss part, yellow bar: inside the mask; hovering shows the
                        value at that position. Each chart keeps one
                        y scale: the curve's min / max over all history snapshots (at the frame shown
                        when it was built, widened if another frame goes beyond), so the metric can
                        be watched rising or falling while scrubbing the history
    Manual pose (block bottom left): [-] / [+] shift the profile-selected slice by 1 mm along
                        its own u / v / normal axes or rotate it by 1 deg about them (about the slice
                        centre), on top of the shown step's pose; lines, contours and profiles follow
                        (outline colours stay the logged metrics). "reset all slices" clears every
                        offset; offsets are also cleared when switching subject / run
    r                   reset camera
    q                   quit
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pyvista as pv
from pyvista.plotting.renderer import Renderer as PvRenderer
from matplotlib.path import Path as PolyPath
from contourpy import contour_generator
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates
from vtkmodules.vtkRenderingCore import vtkActor2D, vtkCoordinate, vtkPolyDataMapper2D

ALIGN_ROOT = Path(r"C:\Users\Nil\Documents\data\ukbb_alignment")
REMOTE_CACHE = Path.home() / ".cache" / "cmr_slice_viewer"   # downloaded subjects of remote runs (--remote)
# Views (image_view): SA, LA_2Ch, LA_3Ch, LA_4Ch, ao -- the UKBB aortic slice (ao.nii.gz, a single axial slice through
# the ascending aorta, without segmentation), shown and handled like a LA view, as is any other image


def label(name: str) -> str:
    """Display form of a view / slice / line-kind name: SA -> SA, LA -> LA (the data keep SA / LA), ao -> Ao (also
    after a modality prefix, <modality>/ao)."""
    head, sep, base = name.rpartition("/")
    return head + sep + ("Ao" if base == "ao" else base)


VIEW_RANK = {"SA": 0, "LA_2Ch": 1, "LA_3Ch": 2, "LA_4Ch": 3, "ao": 5}      # order of a modality's images (others: 4)


def image_view(image: str) -> str:
    """The view of an image name (viewer spelling, without its <modality>/ prefix), classified as the alignment script
    does (moco_nifti.image_kind): SA, LA_2Ch / LA_3Ch / LA_4Ch, else the name itself (e.g. ao; handled like a LA view)."""
    base = image.rpartition("/")[2]
    m = re.search(r"([234])[_\- ]?ch(?![a-z])", base, re.I)
    if m:
        return f"LA_{m.group(1)}Ch"
    if re.search(r"(^|[^a-z])(sa|short[_\- ]?axis)([^a-z]|$)", base, re.I):
        return "SA"
    return base

SEG_SUBDIR = "nnUNet_segs"
SEG_SUFFIX = "_nnUNetSeg.nii.gz"
NII_SUFFIX = ".nii.gz"
# The alignment script writes view names, CSV values / columns and one file name with an extra "X"
# (e.g. the short-axis stack). The viewer uses SA / LA throughout and converts only here, on reading.
ON_DISK = {"SA": "SA" + "X", "LA": "LA" + "X"}


def from_disk(text: str) -> str:
    """On-disk spelling -> the viewer's SA / LA names (views, line kinds, column and file names)."""
    for code, disk in ON_DISK.items():
        text = text.replace(disk, code).replace(disk.lower(), code.lower())
    return text


def to_disk(text: str) -> str:
    """The viewer's SA / LA names -> on-disk spelling (file names to look up)."""
    for code, disk in ON_DISK.items():
        text = text.replace(code, disk).replace(code.lower(), disk.lower())
    return text


def read_csv(path: Path) -> pd.DataFrame:
    """pd.read_csv with view names, line kinds and column names converted to SA / LA."""
    df = pd.read_csv(path)
    df.columns = [from_disk(c) for c in df.columns]
    df = df.rename(columns=lambda c: "n" + c if c == "mi" or c.startswith("mi_") else c)   # runs before mi -> nmi
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):     # text columns (object or string dtype)
            df[c] = df[c].map(lambda v: from_disk(v) if isinstance(v, str) else v)
    return df
SLICE_RE = re.compile(r"_slice(\d+)" + re.escape(NII_SUFFIX) + r"$")
METRICS_SUBDIR = "metrics"
HISTORY_CSV = "history.csv"
HISTORY_SUMMARY_CSV = "history_summary.csv"
AFFINE_COLUMNS = [f"a{r}{c}" for r in range(3) for c in range(4)]

# Colour per segmentation label (label 0 = background, never drawn).
SEG_COLORS = {
    1: (220, 40, 40),     # red
    2: (40, 200, 40),     # green
    3: (60, 90, 255),     # blue
    4: (240, 220, 40),    # yellow
    5: (230, 60, 230),    # magenta
    6: (40, 220, 220),    # cyan
    7: (255, 140, 0),     # orange
}
# Outline colour = one per-plane misalignment metric of history.csv at the shown snapshot.
#   label          shown in the UI
#   col            column of history.csv
#   higher_better  True: high = green, low = red; False (distances): the reverse
#   fmt            value format in the slice list
#   bounds         absolute limits of the colour-range sliders
#   slider_fmt     label format of those sliders
#   summary        subject-level column(s) of history_summary.csv shown next to the radio button
OUTLINE_METRICS = {
    "ncc": dict(label="NCC", col="ncc", higher_better=True, fmt="{:.3f}",
                bounds=(-1.0, 1.0), slider_fmt="%.2f", summary=("ncc_mean",)),
    "lncc": dict(label="LNCC", col="lncc", higher_better=True, fmt="{:.3f}",
                 bounds=(-1.0, 1.0), slider_fmt="%.2f", summary=("lncc_mean",)),
    "nmi": dict(label="NMI", col="nmi", higher_better=True, fmt="{:.3f}",
                bounds=(0.0, 1.0), slider_fmt="%.2f", summary=("nmi_mean",)),
    "dice": dict(label="Dice", col="seg_dice", higher_better=True, fmt="{:.3f}",
                 bounds=(0.0, 1.0), slider_fmt="%.2f", summary=("seg_dice_mean",)),
    "hd": dict(label="Hausdorff", col="hd_mm", higher_better=False, fmt="{:.1f} mm",
               bounds=(0.0, 50.0), slider_fmt="%.1f mm", summary=("hd_mean_mm", "hd_max_mm")),
}
RANGE_MIN_GAP = 1e-3   # colour min is kept at least this far below colour max

# Colour bar block (normalised viewport coordinates): the bar strip with the
# "colour max" slider right above it and the "colour min" slider right below it.
BAR_X0, BAR_X1 = 0.76, 0.90          # horizontal extent of strip and sliders
BAR_Y0, BAR_H = 0.13, 0.022          # strip bottom and height
SLIDER_HI_Y = BAR_Y0 + BAR_H + 0.011  # max slider
SLIDER_LO_Y = BAR_Y0 - 0.011          # min slider
RANGE_TEXT_X = BAR_X1 + 0.008        # small value texts to the right of the sliders
OUTLINE_CMAP = "RdYlGn"          # low = red ... high = green; reversed when lower is better
OUTLINE_NAN_COLOR = "#808080"    # plane without a value (no intersection lines / no segmentation)
OUTLINE_LINE_WIDTH = 3

# Overlays written by the alignment script next to the history
LINES_CSV = "lines.csv"
SA_LINES_CSV = to_disk("sa_lines.csv")      # per-snapshot SA-SA crossing lines
SA_LINE_COLOR = (206, 147, 216)     # violet, distinct from the SA-LA / LA-LA colours
SA_LINE_IMG_WIDTH = 2               # SA-SA lines are drawn a bit wider than the SA-LA / LA-LA ones:
SA_LINE_USED_WIDTH = 6              # they are short and cross the stack at shallow angles
LONG_AXIS_CSV = "long_axis.csv"      # per-snapshot LV long axis + radius of the SA-SA supervision
RADIUS_COLOR = (206, 147, 216)       # same family as the SA-SA lines it bounds
RADIUS_WIDTH = 2
RADIUS_SAMPLES = 96                  # points on each slice's radius ellipse
MASK_CONTOURS_CSV = "mask_contours.csv"
MASK_CONTOUR_COLOR = "#ffeb3b"
MASK_CONTOUR_WIDTH = 2
LINE_COLORS = {"SA-LA": (79, 195, 247), "LA-LA": (255, 183, 77)}   # SA-LA / LA-LA, as in mask_overlay.png
# LA-LA lines burned into a LA slice take the colour of the view they cross (the 2Ch / 3Ch / 4Ch views all contain the
# LV long axis, so on one view the lines with the other two nearly coincide). Lines that coincide on a slice are drawn
# as interleaved dashes, one colour each, so every one of them stays visible.
LA_PARTNER_COLORS = {"LA_2Ch": (255, 183, 77), "LA_3Ch": (240, 240, 240), "LA_4Ch": (255, 64, 129),
                     "ao": (100, 255, 218)}       # amber, white, pink, aquamarine
COINCIDE_DEG, COINCIDE_MM = 6.0, 4.0     # two lines on one slice coincide: directions within / midpoint distance below
DASH_MM = 8.0                            # dash length of coinciding lines
LINE_USED_WIDTH = 4     # part of the line that is sampled in the loss
LINE_IMG_WIDTH = 1      # part inside both images (thin, faint)

# Intersection intensity profiles: small charts on the right, one per intersection of the
# slice picked with the radio buttons next to the slice list (SA slice: its SA-LA lines with
# 2Ch / 3Ch / 4Ch / Ao and its SA-SA crossing with the next slice; LA view, Ao included: its
# crossings with the other LA views, then with every SA slice); below them a second set with
# the selected metric per point along the same lines. Each set lives in its own framed
# rectangle in the right column (intensity on top, metric below; one rectangle takes the whole
# column when the other set is hidden) showing PROFILE_VISIBLE charts; with more intersections
# the mouse wheel over either rectangle scrolls both together, smoothly (SCROLL_STEP charts per
# notch, eased over a few frames; a scroll bar on the right edge shows the position). The charts
# of each rectangle are drawn by their own overlay renderer whose viewport is the rectangle, so
# charts scrolled half out of it are clipped at its edges. The column runs from PROFILE_Y0 (above the colour
# bar and its max slider) to PROFILE_TOP (below the run / subject navigation, top right).
PROFILE_VISIBLE = 4                         # charts shown per rectangle; the others scroll
PROFILE_X, PROFILE_W = 0.735, 0.25
PROFILE_Y0, PROFILE_TOP = 0.19, 0.935       # column of the charts (normalised window height)
PROFILE_GAP = 0.006                         # vertical gap between two charts
RECT_GAP = 0.014                            # gap between the intensity and the metric rectangle
RECT_PAD = 0.004                            # frame padding around the charts
SCROLL_X0, SCROLL_W = PROFILE_X + PROFILE_W + 0.004, 0.005   # scroll bar, right of the charts
RECT_X0, RECT_X1 = PROFILE_X - RECT_PAD, SCROLL_X0 + SCROLL_W + 0.003
RECT_COLOR = (0.35, 0.35, 0.35)             # rectangle frame
SCROLL_TRACK_COLOR = (0.20, 0.20, 0.20)
SCROLL_THUMB_COLOR = (0.70, 0.70, 0.70)
SCROLL_STEP = 0.5                           # charts scrolled per mouse-wheel notch
SCROLL_EASE = 0.45                          # fraction of the remaining distance covered per animation frame
SCROLL_TIMER_MS = 15                        # animation frame interval (also the background work's timer, _tick)
IDLE_BUDGET_S = 0.012                       # background work per timer tick (metric-chart ranges, colour prefetch) ...
IDLE_BUDGET_SCROLL_S = 0.004                # ... and while the charts scroll
COLOR_CACHE_MB = 400                        # slice colours kept per (slice, frame, display settings, blur level)
CHART_CACHE_MB = 200                        # chart data kept per (step, frame, intersection) of the profiled slice
TICK_LABEL_PX = 12                          # min vertical distance (px) between two y tick labels
CHART_FRAME_PX = 34                         # chart height (px) taken by its title and x axis, not the plot area
NAV_RIGHT_W = 380                          # width (px) of the run / subject navigation block, top right
MODALITY_COLOR = "#ce93d8"                 # the active modality tab
TAB_X1 = 284                               # modality tabs: right end of the strip (px; left of the orientation axes)
TABS_PER_ROW = 2                           # more modalities wrap onto further rows of tabs
TAB_CHAR_PX = 7.5                          # approx. width of a tab label character (font size 7); longer names
                                           # are shortened to fit their tab
TAB_COLORS = {True: ("#4a3a55", MODALITY_COLOR, MODALITY_COLOR),   # active tab: fill, frame, label
              False: ("#2a2a2a", "#555555", "#9e9e9e")}
HEADER_TOP_DY = 14                         # header (title, run name) top edge, px below the window top
HIST_SLIDER_Y, FRAME_SLIDER_Y = 0.03, 0.075  # bottom sliders (normalised height of their tubes)
SLIDER_TITLE_GAP = 14                      # px between a bottom slider's title and its tube
PROFILE_COLORS = ("#e0e0e0", "#ffb74d")    # picked slice, the other slice of the line
PROFILE_STEP_MM = 0.5
BLUR_QUANTUM_MM = 0.2                     # motion_gradient.align applies the blur sigma rounded up to this
BLUR_TEXT_X = 218                         # blur value right of the "Image blurring  [g]" label
PROFILE_RADIO_X = 230                     # radio column right of the slice rows
KNOB_LABEL_DY = 9                          # px from a bottom slider's tube to its value label
PROFILE_MASK_Y = 10                        # height (grey units) of the in-mask bar along the chart bottom
PROFILE_GRID_GREY = 0.22                   # horizontal grid lines (dim)
PROFILE_LOSS_COLOR = "#4dd0e1"             # per-point metric curve (metric charts)
HOVER_COLOR = "#ff1744"                    # hover cursor in the charts and its tick marks in 3D
HOVER_TICK_MM = 12.0                       # length of the tick across the line, in each slice
HOVER_WIDTH = 4

# Manual pose controls for the profile-selected slice (applied on top of the shown step's pose)
MOVE_PARAMS = [("shift u", "mm"), ("shift v", "mm"), ("shift n", "mm"),
               ("rot u", "deg"), ("rot v", "deg"), ("rot n", "deg")]
MOVE_STEP = (1.0, 1.0)                     # mm per click, degrees per click
MOVE_X, MOVE_TOP = 12, 92                  # block position (pixels from the bottom left)
MOVE_COL_W = 185                          # two columns: shifts left, rotations right

# Panel layout (pixels from the bottom-left corner of the window)
PANEL_X_BOX = 12
PANEL_X_TXT = 42
PANEL_X_VAL = 146                         # slice rows: column of the outline-metric values
ROW_PITCH = 24


# ----------------------------------------------------------------------------
# Image loading
# ----------------------------------------------------------------------------
def _as_4d(arr: np.ndarray) -> np.ndarray:
    """Return array as (x, y, z, t): a 3D array is one slice (x, y, t), a 2D one one slice with one frame (x, y)."""
    if arr.ndim == 2:
        return arr[:, :, None, None]
    if arr.ndim == 3:
        return arr[:, :, None, :]
    if arr.ndim == 4:
        return arr
    raise ValueError(f"Unexpected array dimensionality {arr.ndim}")


def _n_frames_on_disk(img) -> int:
    """Number of cine frames of a NIfTI image, from the header: its last axis for (x, y, z, t) and (x, y, t), 1 for a
    2D image (x, y)."""
    return int(img.shape[-1]) if len(img.shape) >= 3 else 1


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
            vote += float((cs[0] - cs[1]) @ d)
    return -d if vote < 0 else d


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


def phase_frames(n_available: int, n: int) -> list[int]:
    """The frames of an n_available-frame cine at the phases of an n-frame one: frame k of n -> floor(k / n *
    n_available), as motion_gradient.phase_frames. Every cine spans one cardiac cycle, so this pairs the same
    cardiac phase across cines of different frame counts (UKBB: the aortic cine has 100 frames, the others 50)."""
    return sorted({k * n_available // n for k in range(n)})        # integer arithmetic: exact for every k


def _read_frames(img, frames: list[int], dtype=np.float32) -> np.ndarray:
    """The given frames of a NIfTI image as (x, y, z, t). Evenly strided frames (e.g. every other frame of the
    aortic cine) are read with a strided slice of the array proxy, so the frames left out are never kept in
    memory; other selections read the whole image first."""
    n_disk = _n_frames_on_disk(img)
    if frames == list(range(n_disk)):
        return _as_4d(np.asarray(img.dataobj, dtype=dtype))
    if len(img.shape) >= 3:                                         # the time axis is the last one
        step = frames[1] - frames[0] if len(frames) > 1 else n_disk
        if frames == list(range(frames[0], n_disk, step))[:len(frames)] and frames[0] + step * len(frames) >= n_disk:
            return _as_4d(np.asarray(img.dataobj[..., frames[0]::step], dtype=dtype))
    return _as_4d(np.asarray(img.dataobj, dtype=dtype))[..., frames]


def _slice_index(path: Path) -> int:
    m = SLICE_RE.search(path.name)
    return int(m.group(1)) if m else 0


def find_images(subject_dir: Path) -> list[dict]:
    """
    The images of a subject folder as the alignment script writes them (moco_nifti.save): every folder holding NIfTI
    files is one modality (segmentation folders, metrics/ and hidden folders aside); in it <name>.nii.gz is an image (a
    single slice or a stack) and <name>_slice<nn>.nii.gz are the slices of the image <name>. With several modalities an
    image is named <modality>/<name> (the folder relative to subject_dir), as in the run's CSVs, else <name>; names in
    the viewer's spelling (from_disk).

    Returns [{"image", "view" (image_view), "modality" ("" for subject_dir itself), "mode": "volume" | "slices",
    "paths"}]: the modalities in folder order, each with its SA image(s) first, then the LA views, other images and
    the aortic slice (VIEW_RANK).
    """
    groups = []
    for folder, dirs, _files in os.walk(subject_dir):
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d != METRICS_SUBDIR and "seg" not in d.lower())
        files = sorted((q for q in Path(folder).iterdir() if q.is_file() and q.name.endswith(NII_SUFFIX)
                        and not q.name.startswith(".")), key=lambda q: q.name)
        if files:
            rel = Path(folder).relative_to(subject_dir).as_posix()
            groups.append(("" if rel == "." else rel, files))
    multi = len(groups) > 1
    out = []
    for modality, files in groups:
        stems = {}
        for q in files:
            m = SLICE_RE.search(q.name)
            stems.setdefault(q.name[:m.start()] if m else q.name[:-len(NII_SUFFIX)], []).append(q)
        images = []
        for stem, paths in stems.items():
            vol = [q for q in paths if not SLICE_RE.search(q.name)]      # <stem>.nii.gz wins over <stem>_sliceNN
            mode, paths = ("volume", vol[:1]) if vol else ("slices", sorted(paths, key=_slice_index))
            image = from_disk(f"{modality}/{stem}" if multi else stem)
            images.append(dict(image=image, view=image_view(image), modality=modality, mode=mode, paths=paths))
        out += sorted(images, key=lambda d: (VIEW_RANK.get(d["view"], 4), d["image"]))
    return out


def seg_path_for(img_path: Path) -> Path:
    """nnUNet_segs/<image file name without .nii.gz>_nnUNetSeg.nii.gz next to the image."""
    return img_path.parent / SEG_SUBDIR / f"{img_path.name[: -len(NII_SUFFIX)]}{SEG_SUFFIX}"


def _load_seg(seg_path: Path, img_shape: tuple, img_affine: np.ndarray, label: str, frames: list[int] | None = None):
    """The segmentation of an image, at the image's selected `frames` (default: all), or None."""
    if not seg_path.exists():
        return None
    seg_img = nib.load(str(seg_path))
    # Some seg files are stored as scaled int16 (scl_slope/inter); nibabel then
    # returns floats that may sit just below the integer label, so round
    # instead of truncating.
    sel = list(range(_n_frames_on_disk(seg_img))) if frames is None else frames
    if sel and max(sel) >= _n_frames_on_disk(seg_img):
        print(f"[warn] {label}: seg has {_n_frames_on_disk(seg_img)} frames, fewer than its image; "
              f"ignoring segmentation", file=sys.stderr)
        return None
    seg = np.rint(_read_frames(seg_img, sel, np.float32)).astype(np.int16)
    if seg.shape != img_shape:
        print(f"[warn] {label}: seg shape {seg.shape} != image shape {img_shape}; "
              f"ignoring segmentation", file=sys.stderr)
        return None
    if not np.allclose(seg_img.affine, img_affine, atol=1e-3):
        print(f"[warn] {label}: seg affine differs from image affine; using image affine",
              file=sys.stderr)
    return seg


def load_images(subject_dir: Path) -> dict:
    """
    Load every image of a subject folder (find_images).

    Returns {image: {"mode": "volume"|"slices", "view", "modality", "slices": [slice, ...]}} where each
    slice is {"grey": (nx, ny, t) uint8, "seg": (nx, ny, t) int16 | None, with t the smallest frame count of
    the subject's cines and the other cines read at those phases (phase_frames; only those frames are kept) -- a
    single-frame image (end-diastole) is repeated over them --, "affine": 4x4 with the origin at this slice (k = 0),
    "index": slice index within the image (0 for a single slice), "name": str (the image, or image[NN] for a slice
    of a stack), "file": its path relative to subject_dir}.
    """
    if not subject_dir.is_dir():
        raise FileNotFoundError(subject_dir)

    found = find_images(subject_dir)
    # All cines are shown at the phases of the one with the fewest frames (UKBB: 50; the aortic cine's 100 frames
    # are read every other one -- frame k of the viewer is the same cardiac phase in every slice, see phase_frames);
    # single-frame images do not count
    counts = [_n_frames_on_disk(nib.load(str(q))) for f in found for q in f["paths"]]
    cines = [c for c in counts if c > 1]
    n_common = min(cines) if cines else 1
    data = {}
    for f in found:
        image, mode = f["image"], f["mode"]
        # Collect raw slices first so intensities can be normalised over the
        # whole image (all slices and frames) for consistent grey levels.
        raw_slices = []
        for q in f["paths"]:
            img = nib.load(str(q))
            n_disk = _n_frames_on_disk(img)
            frames = phase_frames(n_disk, n_common)
            arr = _read_frames(img, frames)
            affine = np.asarray(img.affine, dtype=np.float64)
            seg = _load_seg(seg_path_for(q), arr.shape, affine, q.name, frames)
            if arr.shape[3] < n_common:                    # one frame (end-diastole): shown at every phase
                arr = np.repeat(arr, n_common, axis=3)
                seg = None if seg is None else np.repeat(seg, n_common, axis=3)
            zooms = img.header.get_zooms()
            thick_hdr = float(zooms[2]) if len(img.shape) >= 4 else float("nan")   # (X, Y, T): zooms[2] is time
            nz = arr.shape[2]
            if mode == "slices" and nz != 1:
                print(f"[warn] {q.name}: expected a single slice, found {nz}; using all",
                      file=sys.stderr)
            for k in range(nz):
                idx = _slice_index(q) if mode == "slices" else k
                if mode == "slices" and nz > 1:
                    idx = idx * nz + k
                # Affine with the origin moved to this slice so that k = 0 addresses it,
                # the same convention the history CSV uses for every plane.
                A0 = affine.copy()
                A0[:3, 3] += k * affine[:3, 2]
                raw_slices.append({
                    "raw": arr[:, :, k, :],
                    "seg": None if seg is None else seg[:, :, k, :],
                    "affine": A0,
                    "index": idx,
                    "file": q.relative_to(subject_dir).as_posix(),
                    "thick_hdr": thick_hdr,
                    "n_frames_disk": n_disk,
                })
        raw_slices.sort(key=lambda r: r["index"])

        lo, hi = np.percentile(np.concatenate([r["raw"].ravel() for r in raw_slices]), [1.0, 99.0])
        if hi <= lo:
            hi = lo + 1.0
        # the raw intensities kept as int16 (half of float32): MR images are floats but mostly whole numbers, which
        # are kept exactly (rounded, clipped to the int16 range); an image with fractional values (e.g. normalised to
        # 0..1) is scaled to fill the int16 range instead (raw = stored * raw_scale)
        scale = 1.0
        if any(np.any(np.abs(r["raw"] - np.rint(r["raw"])) > 1e-3) for r in raw_slices):
            amax = max(float(np.abs(r["raw"]).max()) for r in raw_slices)
            scale = amax / 32767.0 if amax > 0 else 1.0
        for r in raw_slices:
            r["raw"] = np.clip(np.rint(r["raw"] / scale), -32768, 32767).astype(np.int16)
        slices = []
        for r in raw_slices:
            grey = np.clip((r["raw"] - lo) / (hi - lo), 0.0, 1.0)
            name = image if len(raw_slices) == 1 else f"{image}[{r['index']:02d}]"
            slices.append({
                "grey": (grey * 255.0 + 0.5).astype(np.uint8),
                "seg": r["seg"],
                "affine": r["affine"],
                "index": r["index"],
                "name": name,
                "file": r["file"],
                "raw": r["raw"],                  # NIfTI intensities (what the loss compares): int16 ...
                "raw_scale": scale,               # ... times this (1 for whole-numbered images)
                "lo": float(lo), "hi": float(hi),  # display scaling of the image (grey = (raw - lo) / (hi - lo))
                "thick_hdr": r["thick_hdr"],
                "n_frames_disk": r["n_frames_disk"],   # frames in the file (the cine keeps n_common of them)
            })
        data[image] = {"mode": mode, "view": f["view"], "modality": f["modality"], "slices": slices}

    if not data:
        raise RuntimeError(f"No images found in {subject_dir}")
    return data


def image_signature(subject_dir: Path) -> dict:
    """What a run's images of one subject look like without reading their pixels: per image file (image, file path)
    its shape, data type and intensity scaling, whether it has a segmentation, and its affine (NIfTI headers only).
    Two runs of a subject with equal signatures (but for the affines) hold the same pixels: the alignment writes the
    input images with corrected affines (see reuse_images)."""
    sig = {}
    for f in find_images(subject_dir):
        for q in f["paths"]:
            img = nib.load(str(q))
            slope, inter = img.dataobj.slope, img.dataobj.inter
            sig[(f["image"], q.relative_to(subject_dir).as_posix())] = dict(
                mode=f["mode"], shape=tuple(img.shape), dtype=str(img.get_data_dtype()),
                scale=(float(slope), float(inter)), seg=seg_path_for(q).exists(),
                affine=np.asarray(img.affine, dtype=np.float64))
    return sig


def reuse_images(cached: dict, cached_sig: dict, subject_dir: Path) -> tuple[dict, dict] | None:
    """Another run's images of the cached subject without reading pixels: (data as load_images, signature), sharing
    the cached slices' arrays, with the affines of subject_dir's headers (each run saves its own corrected pose).
    None if any file differs in name, shape, data type, scaling or segmentation (e.g. a run saved with --image-dtype
    uint8): then everything has to be read."""
    sig = image_signature(subject_dir)
    strip = lambda d: {k: (v["mode"], v["shape"], v["dtype"], v["scale"], v["seg"]) for k, v in d.items()}  # noqa: E731
    if strip(sig) != strip(cached_sig):
        return None
    data = {}
    for image, v in cached.items():
        slices = []
        for sl in v["slices"]:
            hdr = sig[(image, sl["file"])]
            nz = hdr["shape"][2] if len(hdr["shape"]) > 3 else 1
            k = int(sl["index"]) if v["mode"] == "volume" else (int(sl["index"]) % nz if nz > 1 else 0)
            A0 = hdr["affine"].copy()                    # origin moved to this slice, as load_images
            A0[:3, 3] += k * hdr["affine"][:3, 2]
            slices.append(dict(sl, affine=A0))
        data[image] = dict(v, slices=slices)
    return data, sig


def describe_images(subject_dir: Path, data: dict) -> None:
    print(f"[images] {subject_dir}")
    width = max(8, max(len(label(image)) for image in data))
    for image, v in data.items():
        labs = sorted({int(x) for sl in v["slices"] if sl["seg"] is not None for x in np.unique(sl["seg"])})
        shape = v["slices"][0]["grey"].shape
        n_disk = v["slices"][0].get("n_frames_disk", shape[2])
        frames = (f"{shape[2]} frames" if n_disk == shape[2] else
                  "1 frame (end-diastole, shown at every phase)" if n_disk == 1 else
                  f"{shape[2]} frames (of {n_disk}, at the same phases)")
        print(f"  {label(image):{width}s} {v['mode']:7s} {len(v['slices']):2d} slice(s) of {shape[0]}x{shape[1]}, "
              f"{frames}, seg labels={','.join(map(str, labs)) or 'none'}")


# ----------------------------------------------------------------------------
# History loading
# ----------------------------------------------------------------------------
def load_history(metrics_dir: Path) -> dict:
    """
    Parse history.csv (+ history_summary.csv if present).

    Returns {"snapshots": [snap, ...], "n_steps": int, "best_index": int | None}
    with snapshots ordered by step (the "best" entry sorts after a regular
    snapshot of the same step).  Each snap is
        {"step": int, "fraction": float, "is_best": bool,
         "affines": {(image, slice): 4x4}, "metrics": {(image, slice): {ncc, nmi, seg_dice, hd_mm}},
         "summary": {column: value} | None}
    """
    path = metrics_dir / HISTORY_CSV
    if not path.exists():
        raise FileNotFoundError(f"{path} not found (run the alignment with --logging)")
    df = read_csv(path)
    missing = [c for c in ["step", "fraction", "is_best", "image", "slice", *AFFINE_COLUMNS] if c not in df.columns]
    if missing:
        raise ValueError(f"{path} lacks columns {missing}")

    summary = None
    spath = metrics_dir / HISTORY_SUMMARY_CSV
    if spath.exists():
        sdf = read_csv(spath)
        summary = {(int(r.step), bool(int(r.is_best))): r._asdict() for r in sdf.itertuples(index=False)}

    snapshots = []
    for (step, is_best), g in df.groupby(["step", "is_best"], sort=False):
        step, is_best = int(step), bool(int(is_best))
        affines, metrics = {}, {}
        for r in g.itertuples(index=False):
            A = np.eye(4)
            A[:3, :] = np.array([getattr(r, c) for c in AFFINE_COLUMNS], dtype=np.float64).reshape(3, 4)
            key = (str(r.image), int(r.slice))
            affines[key] = A
            metrics[key] = {m: float(getattr(r, m)) for m in ("ncc", "lncc", "nmi", "seg_dice", "hd_mm")
                            if m in df.columns}
        snapshots.append({
            "step": step,
            "fraction": float(g["fraction"].iloc[0]),
            "is_best": is_best,
            "affines": affines,
            "metrics": metrics,
            "summary": summary.get((step, is_best)) if summary else None,
        })
    snapshots.sort(key=lambda s: (s["step"], s["is_best"]))

    # Total number of optimisation steps: step / fraction of any row with fraction > 0.
    n_steps = 0
    for s in snapshots:
        if s["fraction"] > 0:
            n_steps = int(round(s["step"] / s["fraction"]))
            break
    if n_steps == 0:
        n_steps = max(s["step"] for s in snapshots)

    # Value range of every per-plane metric over the whole history, so that outline
    # colours are comparable between snapshots.
    metric_range = {}
    for spec in OUTLINE_METRICS.values():
        col = spec["col"]
        vals = np.array([m[col] for snap in snapshots for m in snap["metrics"].values() if col in m], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            lo, hi = float(vals.min()), float(vals.max())
            if hi - lo < 1e-9:
                lo, hi = lo - 0.5, hi + 0.5
            metric_range[col] = (lo, hi)

    best = [i for i, s in enumerate(snapshots) if s["is_best"]]
    return {"snapshots": snapshots, "n_steps": n_steps, "best_index": best[0] if best else None,
            "metric_range": metric_range, "available": True}


def describe_history(metrics_dir: Path, hist: dict) -> None:
    print(f"[history] {metrics_dir / HISTORY_CSV}: {len(hist['snapshots'])} snapshots, "
          f"{hist['n_steps']} optimisation steps")
    for i, s in enumerate(hist["snapshots"]):
        tag = "  <- best" if s["is_best"] else ""
        planes = ", ".join(sorted({k[0] for k in s["affines"]}))
        print(f"  [{i:2d}] step {s['step']:5d} ({100 * s['fraction']:5.1f}%)  {len(s['affines']):2d} planes "
              f"({planes}){tag}")


def load_overlays(metrics_dir: Path) -> dict:
    """
    Optional geometry written by the alignment script.

    Returns {"lines": DataFrame | None, "contours": {(image, slice): [(N, 3) voxel ijk, ...]},
             "sa_lines": {(step, is_best): {"img": (K, 2, 3), "used": (K, 2, 3)}} | None}.
    lines.csv rows: planes a/b, kind, anchor x0 and unit direction d (world, acquired
    geometry), t_used_lo/hi (sampled range) and t_img_lo/hi (inside both images).
    sa_lines.csv: the same per pair of parallel slices (neighbouring SA slices; with several modalities also the
    nearest parallel slice of another image) and history snapshot, already at that snapshot's poses (the slices only
    cross once tilted); kept here as segment end points, only the parts that exist (inside both images / entering the
    loss), each with its pair of slice keys ((image, slice), (image, slice)); "sa_pairs": every pair of the file.
    """
    out = {"lines": None, "contours": {}, "sa_lines": None, "long_axis": None}
    ap = metrics_dir / LONG_AXIS_CSV
    if ap.exists():
        df = read_csv(ap)
        out["long_axis"] = {
            (int(r.step), bool(int(r.is_best))): (np.array([r.p_x, r.p_y, r.p_z], float),
                                                  np.array([r.d_x, r.d_y, r.d_z], float), float(r.radius_mm))
            for r in df.itertuples(index=False)}
    sp = metrics_dir / SA_LINES_CSV
    if sp.exists():
        df = read_csv(sp)
        x0 = df[["x0_x", "x0_y", "x0_z"]].to_numpy(float)
        d = df[["d_x", "d_y", "d_z"]].to_numpy(float)
        segs = {}
        for name in ("img", "used"):
            lo, hi = df[f"t_{name}_lo"].to_numpy(float), df[f"t_{name}_hi"].to_numpy(float)
            ok = np.isfinite(x0).all(1) & np.isfinite(d).all(1) & np.isfinite(lo) & np.isfinite(hi)
            segs[name] = (ok, np.stack([x0 + lo[:, None] * d, x0 + hi[:, None] * d], axis=1))
        img_a = df["image_a"].astype(str).tolist() if "image_a" in df else ["SA"] * len(df)   # (older runs: SA only)
        img_b = df["image_b"].astype(str).tolist() if "image_b" in df else ["SA"] * len(df)
        pair = list(zip(zip(img_a, df["slice_a"].astype(int).tolist()), zip(img_b, df["slice_b"].astype(int).tolist())))
        out["sa_pairs"] = list(dict.fromkeys(pair))
        out["sa_lines"] = {}
        keys = list(zip(df["step"].astype(int), df["is_best"].astype(int).astype(bool)))
        for key in dict.fromkeys(keys):
            rows = np.array([k == key for k in keys])
            entry = {}
            for name, (ok, e) in segs.items():
                entry[name] = e[rows & ok]
                entry[name + "_pairs"] = [pair[j] for j in np.flatnonzero(rows & ok)]   # slice keys of each segment
            out["sa_lines"][key] = entry
    lp = metrics_dir / LINES_CSV
    if lp.exists():
        df = read_csv(lp)
        ok = np.isfinite(df["t_used_lo"]) & np.isfinite(df["t_used_hi"])
        out["lines"] = df[ok].reset_index(drop=True)
    cp = metrics_dir / MASK_CONTOURS_CSV
    if cp.exists():
        df = read_csv(cp)
        for (image, sl, _c), g in df.groupby(["image", "slice", "contour"], sort=False):
            ijk = np.column_stack([g["i"].to_numpy(float), g["j"].to_numpy(float), np.zeros(len(g))])
            out["contours"].setdefault((str(image), int(sl)), []).append(ijk)
    return out


def describe_overlays(metrics_dir: Path, ov: dict) -> None:
    n_lines = 0 if ov["lines"] is None else len(ov["lines"])
    n_contours = sum(len(c) for c in ov["contours"].values())
    print(f"[overlays] {metrics_dir / LINES_CSV}: {n_lines} intersection lines;  "
          f"{metrics_dir / MASK_CONTOURS_CSV}: {n_contours} mask contour(s) on "
          f"{', '.join(k[0] for k in ov['contours']) or 'no plane'}")
    if ov.get("long_axis"):
        radius = next(iter(ov["long_axis"].values()))[2]
        print(f"[overlays] {metrics_dir / LONG_AXIS_CSV}: long axis at {len(ov['long_axis'])} snapshots, "
              f"SA-SA radius {radius:g} mm")
    elif ov.get("sa_lines") is not None:
        print(f"[overlays] {LONG_AXIS_CSV} missing: re-run the alignment to get the long-axis radius overlay",
              file=sys.stderr)
    if ov.get("sa_lines") is not None:
        n_used = max((len(v["used"]) for v in ov["sa_lines"].values()), default=0)
        print(f"[overlays] {metrics_dir / SA_LINES_CSV}: {len(ov['sa_lines'])} snapshots, "
              f"up to {n_used} SA-SA line(s) entering the loss")
    if ov["lines"] is None:
        print("[overlays] lines.csv missing: re-run the alignment to get the sampling-line overlay", file=sys.stderr)
    if not ov["contours"]:
        print("[overlays] mask_contours.csv missing: re-run the alignment to get the mask overlay", file=sys.stderr)


def _created(p: Path) -> float:
    """Creation time of a path (st_birthtime where available; st_ctime is creation time on Windows)."""
    st = p.stat()
    return getattr(st, "st_birthtime", st.st_ctime)


def _is_subject_dir(p: Path) -> bool:
    """A viewable subject of a run: a numeric folder with a history (metrics/history.csv, written with
    --logging) or at least the aligned images (<subject>/*.nii.gz, or in a folder below it: one per modality, or
    <subject>/Instance_<n>/ in older runs)."""
    if not (p.is_dir() and p.name.isdigit()):
        return False
    if (p / METRICS_SUBDIR / HISTORY_CSV).exists():
        return True
    return any(p.glob(f"*{NII_SUFFIX}")) or any(p.glob(f"*/*{NII_SUFFIX}"))


def runs_by_creation(align_root: Path, subject: str | None = None) -> list[Path]:
    """Run directories under align_root with at least one viewable subject (`subject` if given), newest first."""
    if not align_root.is_dir():
        return []

    def viewable(p: Path) -> bool:
        if subject is not None:
            return _is_subject_dir(p / subject)
        return any(_is_subject_dir(q) for q in p.iterdir())

    runs = [p for p in align_root.iterdir() if p.is_dir() and viewable(p)]
    return sorted(runs, key=_created, reverse=True)


def find_run_dir(align_root: Path, subject: str | None, run: str | None) -> Path | None:
    """Resolve the run directory: --run under align_root, else the most recently *created* run
    holding a history (for the given subject when provided)."""
    if run is not None:
        cand = align_root / run
        if not cand.is_dir():
            raise FileNotFoundError(cand)
        return cand
    runs = runs_by_creation(align_root, subject)
    return runs[0] if runs else None


class RemoteRun:
    """A run directory on another machine, `user@host:/path/to/run_dir`, read over ssh (key authentication; ssh runs
    with BatchMode, so it fails instead of prompting). Each subject folder (images, segmentations, metrics/) is
    downloaded once, as one tar stream, into a local mirror of the run, <cache>/<host>/<run name>/, and from there
    loaded like a local run; a subject is only complete once its marker file is written, so an interrupted download
    is fetched again. The next subject of the list is prefetched in the background while one is viewed."""

    MARKER = ".downloaded"

    def __init__(self, spec: str, cache_root: Path):
        if ":" not in spec:
            raise ValueError(f"--remote expects user@host:/path/to/run_dir, got {spec!r}")
        self.target, path = spec.split(":", 1)
        self.path = path.rstrip("/")
        host = re.sub(r"[^\w.-]", "_", self.target.split("@")[-1])
        self.local = Path(cache_root) / host / Path(self.path).name
        self.local.mkdir(parents=True, exist_ok=True)
        for stale in self.local.glob(".partial-*"):          # downloads interrupted by closing the viewer
            shutil.rmtree(stale, ignore_errors=True)
        self._lock = threading.Lock()
        self._busy: dict[str, threading.Event] = {}         # subject -> set when its download has finished

    def _ssh(self, command: str, **kwargs) -> subprocess.Popen:
        return subprocess.Popen(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", self.target, command],
                                **kwargs)

    def run(self, command: str, timeout: float = 300) -> str:
        """Output of a shell command on the remote machine (raises on failure)."""
        p = self._ssh(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out, err = p.communicate(timeout=timeout)
        if p.returncode:
            raise RuntimeError(f"ssh {self.target} {command!r} failed ({p.returncode}): {err.strip()}")
        return out

    def _download(self, rel: str) -> None:
        """Stream <remote run>/<rel> (a file or folder) into the mirror as a tar archive, via a temporary folder."""
        tmp = self.local / f".partial-{rel.replace('/', '_')}"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        p = self._ssh(f"tar -C {shlex.quote(self.path)} -cf - {shlex.quote(rel)}",
                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            with tarfile.open(fileobj=p.stdout, mode="r|") as tar:
                tar.extractall(tmp, filter="data")
        finally:
            err = p.stderr.read().decode(errors="replace")
            p.wait()
        if p.returncode:
            shutil.rmtree(tmp, ignore_errors=True)
            raise RuntimeError(f"downloading {self.target}:{self.path}/{rel} failed: {err.strip()}")
        dest = self.local / rel
        if dest.exists():
            shutil.rmtree(dest) if dest.is_dir() else dest.unlink()
        dest.parent.mkdir(parents=True, exist_ok=True)
        (tmp / rel).replace(dest)
        shutil.rmtree(tmp, ignore_errors=True)

    def fetch_subject(self, subject: str) -> Path:
        """The local copy of a subject folder, downloaded first if needed (waits for a running prefetch of it)."""
        marker = self.local / subject / self.MARKER
        while True:
            if marker.exists():
                return self.local / subject
            with self._lock:
                event = self._busy.get(subject)
                if event is None:
                    event = self._busy[subject] = threading.Event()
                    mine = True
                else:
                    mine = False
            if not mine:
                event.wait()
                continue
            try:
                print(f"[remote] downloading {subject} from {self.target}:{self.path} ...", flush=True)
                self._download(subject)
                marker.write_text("")
            finally:
                with self._lock:
                    self._busy.pop(subject, None)
                event.set()

    def prefetch(self, subject: str) -> None:
        """Download a subject in the background (errors are only reported)."""
        def work():
            try:
                self.fetch_subject(subject)
            except Exception as exc:                          # noqa: BLE001
                print(f"[remote] prefetch of {subject} failed: {exc}", file=sys.stderr)
        if not (self.local / subject / self.MARKER).exists():
            threading.Thread(target=work, daemon=True).start()

    def fetch_run_config(self) -> None:
        """overall_metrics/run_config.json (the run's settings: mask annealing, line sampling, PSF), if present."""
        try:
            self._download("overall_metrics/run_config.json")
        except RuntimeError as exc:
            print(f"[remote] no run_config.json: {exc}", file=sys.stderr)

    def list_subjects(self) -> list[str]:
        """The run's subjects with images, from overall_metrics/motion_summary.csv (one row per subject; listing the
        run directory itself is slow on a large cohort), in numerical order."""
        out = self.run(f"tail -n +2 {shlex.quote(self.path + '/overall_metrics/motion_summary.csv')} | cut -d, -f1,2")
        status = {}
        for line in out.splitlines():
            sid, _, st = line.partition(",")
            if sid.strip():
                status[sid.strip()] = st.strip()               # append-only: the latest row of a subject wins
        return sorted((sid for sid, st in status.items() if st != "no_images"), key=lambda x: (len(x), x))


SUBJECT_FOCUS: dict[str, str] = {}      # subject -> slice to profile first, from a --subjects list (read_subject_list)


def focus_slice(image: str, index: int) -> str:
    """The viewer's name of slice `index` of an image as written in the alignment CSVs: SAX 5 -> SA[05], LAX_4Ch 0 ->
    LA_4Ch (a single slice keeps the image name), ao 0 -> ao, cine/SAX 5 -> cine/SA[05]."""
    image = from_disk(image)
    return f"{image}[{index:02d}]" if image_view(image) == "SA" else image


def read_subject_list(items: list[str]) -> list[str]:
    """Subject ids from --subjects: each item is a subject id or a text file with one subject per line (its first
    whitespace / comma separated token; lines starting with '#' and non-numeric tokens, e.g. a header, are skipped).
    Order kept, duplicates dropped. Lines of a worst_cases.py list (subject, value, image, slice, ...) also give the
    slice where the subject is worst: it is the one profiled first (SUBJECT_FOCUS)."""
    out = []
    for item in items:
        p = Path(item)
        if p.is_file():
            for line in p.read_text().splitlines():
                toks = line.split()
                if not toks or toks[0].startswith("#"):
                    continue
                tok = re.split(r"[\s,]+", line.strip(), maxsplit=1)[0]
                if not tok.isdigit():
                    continue
                out.append(tok)
                if len(toks) >= 4 and toks[3].isdigit() and tok not in SUBJECT_FOCUS:
                    SUBJECT_FOCUS[tok] = focus_slice(toks[2], int(toks[3]))
        else:
            out += [t for t in re.split(r"[\s,]+", item) if t]
    return list(dict.fromkeys(out))


# ----------------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------------
def voxel_to_world(affine: np.ndarray, ijk: np.ndarray) -> np.ndarray:
    """ijk: (N, 3) voxel coordinates -> (N, 3) world coordinates (mm)."""
    hom = np.column_stack([ijk, np.ones(len(ijk))])
    return (affine @ hom.T)[:3].T


def corner_ijk(nx: int, ny: int) -> np.ndarray:
    """Voxel coordinates of the 4 outer corners (voxel edges) of a slice at k = 0."""
    return np.array([
        [-0.5, -0.5, 0.0],
        [nx - 0.5, -0.5, 0.0],
        [nx - 0.5, ny - 0.5, 0.0],
        [-0.5, ny - 0.5, 0.0],
    ], dtype=np.float64)


QUAD_TCOORDS = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])   # texture coordinates of corner_ijk


def slice_quad(affine: np.ndarray, nx: int, ny: int) -> tuple[pv.PolyData, pv.ImageData, pv.Texture]:
    """A slice (k = 0 of `affine`) as one textured quad: (quad, its RGB image (nx, ny), the texture). The image's
    point (i, j) is pixel (i, j) (point index i + j nx, the order of cell_colors); the texture is not interpolated,
    so every pixel shows as a square, and the quad spans the voxel edges."""
    quad = pv.PolyData(voxel_to_world(affine, corner_ijk(nx, ny)), faces=np.array([4, 0, 1, 2, 3]))
    quad.active_texture_coordinates = QUAD_TCOORDS
    img = pv.ImageData(dimensions=(nx, ny, 1))
    img.point_data["rgb"] = np.zeros((nx * ny, 3), np.uint8)
    tex = pv.Texture(img)
    tex.interpolate = False
    tex.mipmap = False
    tex.repeat = False
    return quad, img, tex


def tab_texture(width: int, height: int, fill: str, frame: str) -> pv.ImageData:
    """Rectangular button image (a modality tab): `fill` inside a one-pixel `frame`."""
    arr = np.empty((height, width, 3), dtype=np.uint8)
    arr[:] = pv.Color(frame).int_rgb
    arr[1:-1, 1:-1] = pv.Color(fill).int_rgb
    img = pv.ImageData(dimensions=(width, height, 1))
    img.point_data["texture"] = arr.reshape(width * height, 3)
    return img


def checkbox_texture(rgb01, size: int = 20, border: int = 1, bg: str = "#303030") -> pv.ImageData:
    """Square checkbox button image: `rgb01` fill inside a background-coloured border."""
    arr = np.empty((size, size, 3), dtype=np.uint8)
    arr[:] = pv.Color(bg).int_rgb
    arr[border:size - border, border:size - border] = np.clip(np.asarray(rgb01) * 255 + 0.5, 0, 255).astype(np.uint8)
    img = pv.ImageData(dimensions=(size, size, 1))
    img.point_data["texture"] = arr.reshape(size * size, 3)
    return img


BURN_OFFSET_MM = 0.25    # burned-in lines: one trace this far in front of and one behind the slice


def np_ncc(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    """Weighted NCC of two profiles (motion_gradient.ncc)."""
    ws = max(w.sum(), 1.0)
    ac, bc = a - (w * a).sum() / ws, b - (w * b).sum() / ws
    return float((w * ac * bc).sum() / (np.sqrt(max((w * ac * ac).sum() * (w * bc * bc).sum(), 1e-12)) + 1e-6))


def np_lncc(a: np.ndarray, b: np.ndarray, w: np.ndarray, k: int, min_std: float, return_windows: bool = False):
    """Local NCC of two profiles (motion_gradient.lncc_similarity, one frame); with return_windows also the
    per-window local NCC and window weights (window i covers samples i .. i + k - 1)."""
    k = min(k, len(a))
    ws = max(w.sum(), 1e-6)

    def z(x):
        mu = (w * x).sum() / ws
        return (x - mu) / (np.sqrt((w * (x - mu) ** 2).sum() / ws + 1e-12) + 1e-6)

    def wsum(x):
        c = np.concatenate([[0.0], np.cumsum(x)])
        return c[k:] - c[:-k]

    za, zb = z(a), z(b)
    n = wsum(w)
    nn = np.maximum(n, 1e-6)
    ma, mb = wsum(w * za) / nn, wsum(w * zb) / nn
    va = np.maximum(wsum(w * za * za) / nn - ma * ma, 0.0)
    vb = np.maximum(wsum(w * zb * zb) / nn - mb * mb, 0.0)
    loc = (wsum(w * za * zb) / nn - ma * mb) / np.sqrt(va * vb + 1e-6)
    conf = np.minimum((va * vb + 1e-12) ** 0.25 / min_std, 1.0)
    wt = n * conf * (n >= 0.5 * k)
    value = float((loc * wt).sum() / max(wt.sum(), 1e-6))
    return (value, loc, wt) if return_windows else value


def np_nmi(a: np.ndarray, b: np.ndarray, w: np.ndarray, bins: int = 24, sigma_bins: float = 1.0) -> float:
    """Soft normalized mutual information 2 I / (H_a + H_b) (motion_gradient.soft_nmi, one frame)."""
    eps = 1e-10
    ws = max(w.sum(), 1.0)

    def unit(x):
        m = (x * w).sum() / ws
        sd = np.sqrt((w * (x - m) ** 2).sum() / ws + 1e-6)
        return np.clip((x - m) / sd, -3.0, 3.0) / 6.0 + 0.5

    centres = (np.arange(bins) + 0.5) / bins
    sig = sigma_bins / bins

    def parzen(x):
        kk = np.exp(-0.5 * ((x[:, None] - centres) / sig) ** 2)
        return kk / np.maximum(kk.sum(1, keepdims=True), eps)

    pa, pb = parzen(unit(a)), parzen(unit(b))
    joint = np.einsum("sk,sl,s->kl", pa, pb, w) / ws
    pam, pbm = joint.sum(1), joint.sum(0)
    h_ab = -(joint * np.log(joint + eps)).sum()
    h_a, h_b = -(pam * np.log(pam + eps)).sum(), -(pbm * np.log(pbm + eps)).sum()
    return float(2.0 * (h_a + h_b - h_ab) / max(h_a + h_b, eps))


def rotvec_matrix(rv: np.ndarray) -> np.ndarray:
    """Rotation matrix of a rotation vector (radians)."""
    th = float(np.linalg.norm(rv))
    if th < 1e-12:
        return np.eye(3)
    k = rv / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def apply_offset(A: np.ndarray, offset: np.ndarray, centre_ijk: np.ndarray) -> np.ndarray:
    """Slice affine moved by a manual offset (shift u, v, n in mm; rotation about u, v, n in degrees),
    all along the slice's own axes at pose A; rotations about the slice centre."""
    u, v = A[:3, 0] / np.linalg.norm(A[:3, 0]), A[:3, 1] / np.linalg.norm(A[:3, 1])
    n = np.cross(u, v)
    c = (A @ np.append(centre_ijk, 1.0))[:3]
    R = rotvec_matrix(np.deg2rad(offset[3] * u + offset[4] * v + offset[5] * n))
    out = A.copy()
    out[:3, :3] = R @ A[:3, :3]
    out[:3, 3] = R @ (A[:3, 3] - c) + c + offset[0] * u + offset[1] * v + offset[2] * n
    return out


def crossing_segment(Aa, shape_a, Ab, shape_b):
    """Intersection line of two slices clipped to the part inside both images:
    (x0, unit d, t_lo, t_hi) with points x0 + t d, or None (parallel / no overlap)."""
    na = np.cross(Aa[:3, 0], Aa[:3, 1]); na /= np.linalg.norm(na)
    nb = np.cross(Ab[:3, 0], Ab[:3, 1]); nb /= np.linalg.norm(nb)
    d = np.cross(na, nb)
    nd = np.linalg.norm(d)
    if nd < 1e-6:
        return None
    d /= nd
    x0 = np.linalg.solve(np.stack([na, nb, d]), np.array([na @ Aa[:3, 3], nb @ Ab[:3, 3], d @ Ab[:3, 3]]))
    lo, hi = -np.inf, np.inf
    for A, shape in ((Aa, shape_a), (Ab, shape_b)):
        Ai = np.linalg.inv(A)
        p0, dv = (Ai @ np.append(x0, 1.0))[:2], (Ai[:3, :3] @ d)[:2]     # voxel (i, j) = p0 + t dv
        for ax in range(2):
            bmax = shape[ax] - 1
            if abs(dv[ax]) < 1e-12:
                if not 0 <= p0[ax] <= bmax:
                    return None
                continue
            ta, tb = (0 - p0[ax]) / dv[ax], (bmax - p0[ax]) / dv[ax]
            lo, hi = max(lo, min(ta, tb)), min(hi, max(ta, tb))
    return (x0, d, lo, hi) if hi > lo else None


def burn_points(seg: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Segments (K, 2, 3) projected exactly into the slice plane of `affine`, then duplicated
    +-BURN_OFFSET_MM along its normal so they show from both sides -> (4K, 3) points
    (first all front segments, then all back ones; see burn_cells)."""
    n = np.cross(affine[:3, 0], affine[:3, 1])
    n /= np.linalg.norm(n)
    p = seg - ((seg - affine[:3, 3]) @ n)[..., None] * n
    return np.concatenate([p + BURN_OFFSET_MM * n, p - BURN_OFFSET_MM * n]).reshape(-1, 3)


def burn_cells(k: int) -> np.ndarray:
    """Line cells for the 2k two-point segments of burn_points."""
    return np.column_stack([np.full(2 * k, 2), np.arange(0, 4 * k, 2), np.arange(1, 4 * k, 2)]).ravel()


def cell_colors(grey2d: np.ndarray, seg2d: np.ndarray | None, show_seg: bool,
                alpha: float) -> np.ndarray:
    """(nx*ny, 3) uint8 RGB per pixel (i fastest): grey image, optionally blended with seg."""
    g = grey2d.ravel(order="F").astype(np.float32)
    rgb = np.repeat(g[:, None], 3, axis=1)
    if show_seg and seg2d is not None:
        s = seg2d.ravel(order="F")
        for lab in np.unique(s):
            if lab == 0:
                continue
            col = SEG_COLORS.get(int(lab), SEG_COLORS[1 + (int(lab) - 1) % len(SEG_COLORS)])
            m = s == lab
            rgb[m] = (1.0 - alpha) * rgb[m] + alpha * np.asarray(col, dtype=np.float32)
    return np.clip(rgb + 0.5, 0, 255).astype(np.uint8)


class FrameCache:
    """A cache capped at cap_mb of entries keyed by tuples that hold a cine frame (at position `fpos`). Reading an
    entry marks it used; when something shown needs room, the least recently used entries go. The background
    prefetch (put / make_room with prefetch=True) never pushes out what is near: it only replaces entries of the frame
    furthest ahead of the shown one (frames wrap around), and only for an entry of a nearer frame -- so it fills the
    cache from the shown frame on and follows when another frame is shown, and stops when the cache is full of
    nearer frames."""

    def __init__(self, cap_mb: float, fpos: int):
        self.cap, self.fpos = cap_mb * 2 ** 20, fpos
        self.clear()

    def clear(self):
        self.data, self.size, self.by_frame, self.total = {}, {}, {}, 0

    def __contains__(self, key) -> bool:
        return key in self.data

    def __len__(self) -> int:
        return len(self.data)

    def get(self, key):
        value = self.data.pop(key, None)
        if value is not None:
            self.data[key] = value                       # (now the most recently used)
        return value

    def _drop(self, key):
        del self.data[key]
        self.total -= self.size.pop(key)
        bucket = self.by_frame[key[self.fpos]]
        del bucket[key]
        if not bucket:
            del self.by_frame[key[self.fpos]]

    def make_room(self, key, nbytes: int, shown_frame: int, n_frames: int) -> bool:
        """For the prefetch: room for an entry of nbytes, replacing entries of frames further ahead than key's."""
        ahead = lambda f: (f - shown_frame) % max(1, n_frames)  # noqa: E731
        want = ahead(key[self.fpos])
        while self.total + nbytes > self.cap:
            far = max(self.by_frame, key=ahead, default=None)
            if far is None or ahead(far) <= want:
                return False
            self._drop(next(iter(self.by_frame[far])))
        return True

    def put(self, key, value, nbytes: int, prefetch: bool = False, shown_frame: int = 0, n_frames: int = 1) -> bool:
        """Store an entry (False: the prefetch found no room)."""
        if key in self.data:
            self._drop(key)
        if prefetch and not self.make_room(key, nbytes, shown_frame, n_frames):
            return False
        self.data[key], self.size[key] = value, nbytes
        self.total += nbytes
        self.by_frame.setdefault(key[self.fpos], {})[key] = None
        while not prefetch and self.total > self.cap and len(self.data) > 1:     # (least recently used first)
            self._drop(next(iter(self.data)))
        return True


# ----------------------------------------------------------------------------
# Viewer
# ----------------------------------------------------------------------------
class SliceViewer:
    """
    runs:         ordered run directories (newest first); subjects_for(run) -> ordered
                  subject ids of that run; loader(run, subject) -> dict with "data"
                  (load_images), "history" (load_history), "overlays" (load_overlays) and
                  "title".  Runs and subjects are loaded on demand into the same window;
                  the user's toggles, metric and frame persist across them.
    run_index:    initial run;  subject: initial subject id (or None -> subject_index)
    """

    def __init__(self, runs: list, subjects_for, loader, run_index: int = 0,
                 subject: str | None = None, subject_index: int = 0, frame: int = 0,
                 show_seg: bool = True, seg_alpha: float = 0.5, start: int | None = None,
                 metric: str = "ncc", show_mask: bool = False, show_lines: bool = False,
                 show_sa_lines: bool = False, show_radius: bool = False, show_profiles: bool = True,
                 show_psf: bool = True, show_blur: bool = False, show_metric_profiles: bool = True,
                 window_size=(1500, 1000), verbose: bool = True):
        if not runs:
            raise ValueError("no runs to show")
        self.runs = list(runs)
        self.subjects_for = subjects_for
        self.loader = loader
        self.run_index = int(np.clip(run_index, 0, len(self.runs) - 1))
        self.subjects = list(subjects_for(self.runs[self.run_index]))
        if not self.subjects:
            raise ValueError(f"no subjects in {self.runs[self.run_index]}")
        if subject is not None and subject in self.subjects:
            subject_index = self.subjects.index(subject)
        self.subject_index = int(np.clip(subject_index, 0, len(self.subjects) - 1))
        # user state that survives a subject change
        self.metric = metric if metric in OUTLINE_METRICS else "ncc"
        self.frame = int(frame)
        self.show_seg = show_seg
        self.show_outlines = True
        self.show_mask = show_mask
        self.show_lines = show_lines
        self.show_sa_lines = show_sa_lines
        self.show_radius = show_radius
        self.show_profiles = show_profiles
        self.show_metric_profiles = show_metric_profiles    # per-point metric charts below the intensity ones
        self.show_psf = show_psf                # profiles as the loss sees them (blur + PSF band) instead of raw
        self.show_blur = show_blur              # slice images blurred as at the shown optimisation step
        self._colors_sigma = 0.0                # blur (mm) the slice colours were last computed with
        self.profile_key: str | None = None     # slice whose intersections are plotted (kept across subjects)
        self.offsets: dict[tuple, np.ndarray] = {}   # manual pose offsets per slice key (reset per subject)
        self.modalities: list[str] = []              # the subject's modalities (folder names; [""] for one)
        self._multi = False                          # several modalities: the panel lists one at a time
        self.listed_modality: str | None = None      # the modality whose slices the panel lists (kept across subjects)
        self._color_cache = FrameCache(COLOR_CACHE_MB, 1)   # slice colours per (pixels, frame, settings, blur level)
        self._color_settings = None                  # (segmentation shown, its opacity) of the cached colours
        self._chart_cache = FrameCache(CHART_CACHE_MB, 2)   # chart data per (chart state, step, frame, intersection)
        self._bg_full: set = set()                   # parts of the background stream whose cache is full
        self._chart_state = None                     # the chart state (_range_key) the cache holds
        self._timer = None                           # repeating timer of the scroll animation and background work
        self._timer_wanted = False                   # work queued before the window's event loop runs (_ensure_timer)
        self.seg_alpha = seg_alpha
        self.verbose = verbose
        self._first_start = start           # only the first subject honours --start

        self.pl = pv.Plotter(window_size=window_size, title="slice viewer")
        self.pl.set_background("#202020")
        # One lookup table shared by all outline mappers: switching the metric only
        # has to update this table (colour map + range) and the scalar bar follows.
        self._lut = pv.LookupTable(cmap=OUTLINE_CMAP, scalar_range=(0.0, 1.0),
                                   nan_color=OUTLINE_NAN_COLOR)
        self._scene_actors: list = []        # everything rebuilt per subject
        self._built = False
        self._reset_scene_state()
        self._build_persistent_ui()
        self.load_index(self.subject_index)

    # --- per-subject lifecycle ---------------------------------------------------
    def _reset_scene_state(self):
        self._blur_text = None                       # text next to "Image blurring" (built by _build_ui with a config)
        self.entries: list[dict] = []
        self._entry_by_key: dict[tuple, dict] = {}
        self._mask_items: list[dict] = []
        self._line_items: dict | None = None
        self._sa_line_items: dict | None = None
        self._radius_items: list[dict] = []
        self._widgets: dict[str, object] = {}
        self._radios: dict[str, object] = {}
        self._profile_radios: dict[str, object] = {}
        self._row_text: dict[str, object] = {}
        self._row_value: dict[str, object] = {}      # slice rows: outline-metric value column
        self._tabs: dict[str, dict] = {}             # modality tabs: widget, label, textures (_build_tabs)
        self._tex_cache: dict[str, tuple] = {}
        self._range_sliders: dict[str, object] = {}
        self._range_text: dict[str, object] = {}
        self._frame_slider = None
        self._hist_slider = None
        self._knob = {}
        self._scalar_bar = None
        self._profile_scroll = 0.0                   # scroll position (charts, fractional) of the chart rectangles
        self._scroll_target = 0.0                    # where the scroll animation is heading
        self._n_profile_items = 0
        self._items_cache: dict = {}                 # chart intersections per (snapshot, profiled slice, offsets)
        self._range_cache = None                     # metric-chart y ranges (_metric_ranges) ...
        self._rjob = None                            # ... completed in the background (_range_step)
        self._bg = None                              # the background stream (_bg_step) and the state it was made for
        self._bg_sig = None
        self._pcache = None                          # intersections and chart data of the current state (see
                                                     # _update_profiles / _render_profiles)

    def _text(self, *args, **kwargs):
        actor = self.pl.add_text(*args, **kwargs)
        self._scene_actors.append(actor)
        return actor

    def _mesh(self, *args, **kwargs):
        actor = self.pl.add_mesh(*args, **kwargs)
        self._scene_actors.append(actor)
        return actor

    def _teardown(self, keep_slices: bool = False):
        """Remove every actor and widget of the current subject (the navigation UI stays). keep_slices: the slices'
        actors (image quads and outlines) and their entries stay, for another run of the same subject (load_index)."""
        pl = self.pl
        wh = getattr(pl, "widgets", pl)          # pyvista >= 0.45 keeps the widget lists here
        for w in list(self._widgets.values()) + list(self._radios.values()) + list(self._profile_radios.values()) + \
                list(self._range_sliders.values()) + [self._frame_slider, self._hist_slider]:
            if w is None:
                continue
            w.Off()
            # switching a widget off leaves its representation (the drawn button / slider) in the renderer
            pl.renderer.RemoveViewProp(w.GetRepresentation())
            for lst in (getattr(wh, "button_widgets", []), getattr(wh, "slider_widgets", [])):
                if w in lst:
                    lst.remove(w)
        rd = getattr(wh, "radio_button_widget_dict", None)
        if isinstance(rd, dict):
            rd.pop("outline_metric", None)
            rd.pop("profile", None)
        rt = getattr(wh, "radio_button_title_dict", None)
        if isinstance(rt, dict):
            for t in rt.pop("outline_metric", None) or []:
                pl.remove_actor(t, render=False)
        if "outline colour" in pl.scalar_bars:
            pl.remove_scalar_bar("outline colour", render=False)
        keep = {id(e[k]) for e in self.entries for k in ("actor", "outline")} if keep_slices else set()
        for a in self._scene_actors:
            if id(a) not in keep:
                pl.remove_actor(a, render=False)
        kept = ([a for a in self._scene_actors if id(a) in keep], self.entries, self._entry_by_key, self._tex_cache)
        self._scene_actors = []
        self._reset_scene_state()
        if keep_slices:
            self._scene_actors, self.entries, self._entry_by_key, self._tex_cache = kept

    def _same_slices(self, data: dict) -> bool:
        """Whether `data` (load_images) holds exactly the shown slices -- the same subject from another run, its
        pixels reused (load_subject_run's cache) -- so their actors can be kept (load_index)."""
        new = [(image, s) for image, v in data.items() for s in v["slices"]]
        return len(new) == len(self.entries) and all(
            e["image"] == image and e["name"] == s["name"] and e["slice"]["grey"] is s["grey"]
            for e, (image, s) in zip(self.entries, new))

    def load_index(self, index: int, run_index: int | None = None, subjects: list[str] | None = None) -> bool:
        """Show subject number `index` (clipped) of run `run_index` (default: the current run),
        keeping the user's settings.  Returns False, leaving the view unchanged, if loading fails
        (the very first load re-raises)."""
        run_index = self.run_index if run_index is None else run_index
        run = self.runs[run_index]
        subjects = self.subjects if subjects is None else subjects
        index = int(np.clip(index, 0, len(subjects) - 1))
        subject = subjects[index]
        self._log(f"loading run {Path(run).name} ({run_index + 1}/{len(self.runs)}), "
                  f"subject {subject} ({index + 1}/{len(subjects)}) ...")
        try:
            loaded = self.loader(run, subject)
        except Exception as exc:  # noqa: BLE001
            if not self._built:
                raise
            print(f"[warn] could not load {Path(run).name} / {subject}: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            return False
        # another run of the shown subject (same pixels): keep the slices' actors and the view (camera, visibility,
        # frame, profiled slice); only the run's data, overlays and UI are rebuilt -- much faster than a full load
        same = self._built and self._same_slices(loaded["data"])
        if self._built:
            self._teardown(keep_slices=same)
        self.offsets = {}
        self._hover = None
        self._blur_cache = {}
        self.run_index = run_index
        self.subjects = list(subjects)
        self.subject_index = index
        self.data = loaded["data"]
        self.history = loaded["history"]
        self.overlays = loaded.get("overlays") or {"lines": None, "contours": {}}
        self.overlays.setdefault("sa_lines", None)
        self.overlays.setdefault("long_axis", None)
        self.title = loaded.get("title", subject)
        self.config = loaded.get("config")
        self.run_name = loaded.get("run_name", Path(run).name)
        self.snapshots = self.history["snapshots"]
        self.n_steps = self.history["n_steps"]
        self.metric_range = self.history.get("metric_range", {})
        self.has_metrics = bool(self.history.get("available", True))   # False: no history.csv
        self.not_optimized = bool(loaded.get("not_optimized", False))   # no crossing views: acquired pose
        # Colour range per metric: the subject's min/max over its history (clipped to
        # the metric's absolute bounds), adjustable with the sliders.
        self.color_range: dict[str, list[float]] = {}
        for key, spec in OUTLINE_METRICS.items():
            b_lo, b_hi = spec["bounds"]
            lo, hi = self.metric_range.get(spec["col"], (b_lo, b_hi))
            lo, hi = float(np.clip(lo, b_lo, b_hi)), float(np.clip(hi, b_lo, b_hi))
            if hi - lo < RANGE_MIN_GAP:
                lo, hi = max(b_lo, hi - RANGE_MIN_GAP), min(b_hi, lo + RANGE_MIN_GAP)
            self.color_range[key] = [lo, hi]
        self.n_frames = min(s["grey"].shape[2] for v in self.data.values() for s in v["slices"])
        self.frame = int(np.clip(self.frame, 0, self.n_frames - 1))
        start = self._first_start if not self._built else None
        self._first_start = None
        if start is None:
            start = self.history["best_index"] if self.history["best_index"] is not None else len(self.snapshots) - 1
        self.snap_index = int(np.clip(start, 0, len(self.snapshots) - 1))

        if same:                                         # the kept entries: this run's slice dicts (its affines)
            new = [s for v in self.data.values() for s in v["slices"]]
            for e, s in zip(self.entries, new):
                e["slice"] = s
                e["in_history"] = any(e["key"] in snap["affines"] for snap in self.snapshots)
        else:
            self._build_slices()
        self._build_overlays()
        names = [e["name"] for e in self.entries]
        focus = loaded.get("profile")                    # e.g. a --subjects list's worst slice of this subject
        if focus in names:
            if focus != self.profile_key:
                self._profile_scroll = self._scroll_target = 0.0
            self.profile_key = focus
        if self.profile_key not in names:              # the middle slice of the first SA stack
            sa = [e for e in self.entries if e["view"] == "SA"]
            sa = [e["name"] for e in sa if e["image"] == sa[0]["image"]] if sa else []
            self.profile_key = sa[len(sa) // 2] if sa else (names[0] if names else None)
        self.modalities = list(dict.fromkeys(e["modality"] for e in self.entries))
        self._multi = len(self.modalities) > 1
        prof = self._entry_by_name().get(self.profile_key)
        if self.listed_modality not in self.modalities or (prof is not None and focus == self.profile_key):
            self.listed_modality = prof["modality"] if prof is not None else (self.modalities or [None])[0]
        self._build_ui()
        self._configure_metric()
        self._apply_snapshot()
        self._apply_visibility()
        if not same:
            self._set_initial_camera()
        self._update_nav_text()
        self._update_move_text()
        self._built = True
        self.pl.render()
        return True

    def next_subject(self, step: int = 1):
        target = self.subject_index + step
        if 0 <= target < len(self.subjects):
            self.load_index(target)
        else:
            self._log("no subject in that direction")

    def next_run(self, step: int = 1):
        """Move to the next / previous run (list order, newest first), skipping runs without
        subjects or that fail to load.  The current subject is kept if the run has it,
        otherwise the same position in the run's subject list is used."""
        current = self.subjects[self.subject_index]
        target = self.run_index + step
        while 0 <= target < len(self.runs):
            subjects = list(self.subjects_for(self.runs[target]))
            if subjects:
                index = subjects.index(current) if current in subjects else self.subject_index
                if self.load_index(index, run_index=target, subjects=subjects):
                    return
            target += step
        self._log("no run in that direction")

    # --- persistent navigation UI ----------------------------------------------------
    def _build_persistent_ui(self):
        pl = self.pl
        w, h = pl.window_size
        y = h - 40

        def push(key, glyph, cb, x, y):
            holder = {}

            def _cb(_state):
                holder["w"].GetRepresentation().SetState(0)    # a push button, not a toggle
                cb()
            holder["w"] = pl.add_checkbox_button_widget(
                _cb, value=False, position=(x, y), size=18, border_size=1,
                color_on="#ffa726", color_off="#4a4a4a", background_color="#303030")
            pl.add_text(glyph, position=(x + 5, y + 1), font_size=9, color="white")
            self._nav_widgets[key] = holder["w"]

        # Navigation in the top-right corner: one row per run / subject,  label  [<] [>]  keys
        self._nav_widgets: dict[str, object] = {}
        bx = w - NAV_RIGHT_W
        for row, (attr, colour, prev_key, next_key, prev_cb, next_cb, keys) in enumerate((
                ("_run_text", "#80cbc4", "prev_run", "next_run",
                 lambda: self.next_run(-1), lambda: self.next_run(+1), "{ }  newest first"),
                ("_nav_text", "#ffa726", "prev", "next",
                 lambda: self.next_subject(-1), lambda: self.next_subject(+1), "[ ]"))):
            ry = h - 30 - row * ROW_PITCH
            setattr(self, attr, pl.add_text("", position=(bx, ry + 2), font_size=9, color=colour))
            push(prev_key, "<", prev_cb, w - 154, ry)
            push(next_key, ">", next_cb, w - 130, ry)
            pl.add_text(keys, position=(w - 104, ry + 3), font_size=7, color="#888888")
        self._panel_top = y                           # per-subject rows start at the top

        # Manual pose block, bottom left (acts on the profile-selected slice): two columns
        # (shifts | rotations) of  name [-] [+] value, "reset all slices" next to the title
        self._move_text = pl.add_text("", position=(MOVE_X, MOVE_TOP + 2), font_size=9, color="#ffa726")
        self._move_values = []
        for k, (name, unit) in enumerate(MOVE_PARAMS):
            mx = MOVE_X + MOVE_COL_W * (k // 3)
            my = MOVE_TOP - 24 * (k % 3 + 1)
            pl.add_text(name, position=(mx, my + 3), font_size=8, color="#bbbbbb")
            for j, (sign, glyph) in enumerate(((-1, "-"), (+1, "+"))):
                bx = mx + 50 + 24 * j
                holder = {}

                def _cb(_state, h=holder, k=k, sign=sign):
                    h["w"].GetRepresentation().SetState(0)
                    self.move_selected(k, sign)
                holder["w"] = pl.add_checkbox_button_widget(
                    _cb, value=False, position=(bx, my), size=18, border_size=1,
                    color_on="#ffa726", color_off="#4a4a4a", background_color="#303030")
                pl.add_text(glyph, position=(bx + 5, my + 1), font_size=9, color="white")
                self._nav_widgets[f"move{k}{glyph}"] = holder["w"]
            self._move_values.append(pl.add_text("", position=(mx + 102, my + 3), font_size=8,
                                                 color="white"))
        my = MOVE_TOP
        momentary_x = MOVE_X + 266                   # right of "Manual correction LA_4Ch" (the longest)
        holder = {}

        def _reset(_state, h=holder):
            h["w"].GetRepresentation().SetState(0)
            self.reset_offsets()
        holder["w"] = pl.add_checkbox_button_widget(
            _reset, value=False, position=(momentary_x, my), size=18, border_size=1,
            color_on="#ffa726", color_off="#4a4a4a", background_color="#303030")
        self._reset_text = pl.add_text("reset all slices", position=(momentary_x + 26, my + 2), font_size=8,
                                       color="white")
        self._nav_widgets["move_reset"] = holder["w"]

        # intersection charts (persistent; filled per subject / step / frame): intensity profiles on
        # top, below them one chart per intersection with the selected metric per point
        def new_chart():                                     # placed by _layout_charts when shown
            ch = pv.Chart2D(size=(PROFILE_W, 0.08), loc=(PROFILE_X, PROFILE_Y0),
                            x_label="", y_label="", grid=False)
            ch.background_color = (0.10, 0.10, 0.10, 0.85)
            ch.border_color = (0.35, 0.35, 0.35)
            ch.legend_visible = False
            tp = ch.GetTitleProperties()
            tp.SetFontSize(11); tp.SetColor(0.85, 0.85, 0.85); tp.BoldOff()
            for ax in (ch.x_axis, ch.y_axis):
                ax.tick_label_size = 9
                ax.GetLabelProperties().SetColor(0.65, 0.65, 0.65)
                ax.pen.color = (0.4, 0.4, 0.4)
            for i in range(4):
                ch.GetAxis(i).GetGridPen().SetColorF(*(3 * (PROFILE_GRID_GREY,)))
            ch.visible = False
            return ch

        # one overlay renderer per chart rectangle (layer 1: transparent, not interactive -- the camera keeps
        # following the main renderer); its viewport is the rectangle, which clips the charts scrolled past its edges
        pl.ren_win.SetNumberOfLayers(max(2, pl.ren_win.GetNumberOfLayers()))
        self._chart_rens = {}
        for name in ("profile", "metric"):
            ren = PvRenderer(pl, border=False)
            ren.SetLayer(1)
            ren.InteractiveOff()
            ren.SetViewport(0.0, 0.0, 0.01, 0.01)
            pl.ren_win.AddRenderer(ren)
            self._chart_rens[name] = ren

        x0 = np.array([0.0, 1.0])
        self._profile_charts, self._metric_charts = [], []
        for k in range(PROFILE_VISIBLE + 1):              # + 1: while scrolling, a chart is partly in view at each end
            ch = new_chart()
            ch.y_axis.tick_locations = (0, 128, 255)
            ch.y_axis.tick_labels = ("0", "128", "255")
            plots = {}
            for side, col in zip(("a", "b"), PROFILE_COLORS):
                plots[side + "_img"] = ch.line(x0, x0 * 0, color=col, width=1.0)
                plots[side + "_used"] = ch.line(x0, x0 * 0, color=col, width=2.5)
            plots["mask"] = ch.line(x0, x0 * np.nan, color=MASK_CONTOUR_COLOR, width=2.0)
            plots["cursor"] = ch.line(np.array([0.0, 0.0]), np.array([0.0, 255.0]), color=HOVER_COLOR, width=1.5)
            plots["cursor"].visible = False
            ch.y_range = [0, 255]
            self._chart_rens["profile"].add_chart(ch)
            self._profile_charts.append({"chart": ch, "plots": plots, "item": None, "bound": None})

            ch = new_chart()
            # as the profile charts: thin over the whole intersection, thick where it enters the loss, mask bar
            plots = {"zero": ch.line(x0, x0 * 0, color=(0.4, 0.4, 0.4), width=1.0),
                     "full": ch.line(x0, x0 * np.nan, color=PROFILE_LOSS_COLOR, width=1.0),
                     "curve": ch.line(x0, x0 * np.nan, color=PROFILE_LOSS_COLOR, width=2.5),
                     "mask": ch.line(x0, x0 * np.nan, color=MASK_CONTOUR_COLOR, width=2.0),
                     "cursor": ch.line(np.array([0.0, 0.0]), np.array([0.0, 1.0]), color=HOVER_COLOR, width=1.5)}
            plots["cursor"].visible = False
            self._chart_rens["metric"].add_chart(ch)
            self._metric_charts.append({"chart": ch, "plots": plots, "item": None})
        pl.iren.interactor.AddObserver("TimerEvent", lambda *_: self._tick())
        # background work queued before the event loop runs starts with the first render (_ensure_timer)
        pl.add_on_render_callback(lambda *_: self._ensure_timer() if self._timer_wanted else None, render_event=False)

        # the two chart rectangles: frame, scroll track and thumb (2D actors, normalised viewport coordinates)
        self._rects = {}
        for name in ("profile", "metric"):
            parts = {}
            for part, color, filled in (("frame", RECT_COLOR, False), ("track", SCROLL_TRACK_COLOR, True),
                                        ("thumb", SCROLL_THUMB_COLOR, True)):
                poly = (pv.PolyData(np.zeros((4, 3)), faces=np.array([4, 0, 1, 2, 3])) if filled
                        else pv.PolyData(np.zeros((4, 3)), lines=np.array([5, 0, 1, 2, 3, 0])))
                mapper = vtkPolyDataMapper2D()
                mapper.SetInputData(poly)
                coord = vtkCoordinate()
                coord.SetCoordinateSystemToNormalizedViewport()
                mapper.SetTransformCoordinate(coord)
                actor = vtkActor2D()
                actor.SetMapper(mapper)
                actor.GetProperty().SetColor(*color)
                actor.GetProperty().SetLineWidth(1.0)
                actor.SetVisibility(False)
                pl.renderer.AddViewProp(actor)
                parts[part] = (poly, actor)
            self._rects[name] = {"parts": parts, "box": None}
        # mouse wheel over a rectangle: scroll the charts (and keep the camera from zooming; the interactor style
        # listens at priority 0, this observer runs first and aborts the event)
        iren = pl.iren.interactor
        for evt, step in (("MouseWheelForwardEvent", -1), ("MouseWheelBackwardEvent", +1)):
            holder = {}

            def _wheel(obj, _evt, step=step, holder=holder):
                if self._on_wheel(step):
                    obj.GetCommand(holder["tag"]).SetAbortFlag(1)
            holder["tag"] = iren.AddObserver(evt, _wheel, 10.0)

        # hover: a tick across the intersection line at the hovered position, burned into both slices
        self._hover = None                                   # (chart slot, t along the line) or None
        self._hover_mesh = pv.PolyData(np.zeros((8, 3)), lines=burn_cells(2))
        self._hover_actor = pl.add_mesh(self._hover_mesh, color=HOVER_COLOR, line_width=HOVER_WIDTH,
                                        lighting=False, name="profile_hover")
        self._hover_actor.SetVisibility(False)
        pl.iren.add_observer("MouseMoveEvent", self._on_mouse_move)

        # orientation axes top left, right of the toggle panel
        pl.add_axes(xlabel="X (mm)", ylabel="Y (mm)", zlabel="Z (mm)", viewport=(0.19, 0.80, 0.33, 1.0))

        # Keyboard (bound once; the callbacks look up the current subject's state)
        pl.add_key_event("bracketright", lambda: self.next_subject(+1))
        pl.add_key_event("bracketleft", lambda: self.next_subject(-1))
        pl.add_key_event("Next", lambda: self.next_subject(+1))
        pl.add_key_event("Prior", lambda: self.next_subject(-1))
        pl.add_key_event("braceright", lambda: self.next_run(+1))
        pl.add_key_event("braceleft", lambda: self.next_run(-1))
        pl.add_key_event("s", lambda: self._toggle_checkbox("seg"))
        pl.add_key_event("o", lambda: self._toggle_checkbox("outlines"))
        pl.add_key_event("m", lambda: self._toggle_checkbox("mask"))
        pl.add_key_event("i", lambda: self._toggle_checkbox("lines"))
        pl.add_key_event("k", lambda: self._toggle_checkbox("sa_lines"))
        pl.add_key_event("u", lambda: self._toggle_checkbox("radius"))
        pl.add_key_event("p", lambda: self._toggle_checkbox("profiles"))
        pl.add_key_event("n", lambda: self._toggle_checkbox("metric_profiles"))
        pl.add_key_event("y", lambda: self._toggle_checkbox("psf"))
        pl.add_key_event("g", lambda: self._toggle_checkbox("blur"))
        pl.add_key_event("period", lambda: self.step_profile(+1))
        pl.add_key_event("comma", lambda: self.step_profile(-1))
        pl.add_key_event("a", lambda: self._toggle_checkbox("all_sa"))
        pl.add_key_event("l", lambda: self._toggle_checkbox("all_la"))
        pl.add_key_event("h", lambda: self._toggle_checkbox(f"mod_shown:{self.listed_modality}"))   # (several modalities)
        # the cine is periodic: stepping past either end wraps around
        pl.add_key_event("Left", lambda: self.set_frame((self.frame - 1) % self.n_frames))
        pl.add_key_event("Right", lambda: self.set_frame((self.frame + 1) % self.n_frames))
        for d in range(10):                     # digit d toggles the d-th SA slice (in list order)
            pl.add_key_event(str(d), lambda d=d: self._toggle_sa_slice(d))
        # pyvista binds Up/Down to camera zoom and b to point picking by default, and
        # add_key_event appends rather than replaces: drop those defaults first
        for k in ("Up", "Down", "b", "v"):
            pl.iren._key_press_event_callbacks.pop(k, None)
        pl.add_key_event("v", lambda: self.step_modality(+1))
        pl.add_key_event("V", lambda: self.step_modality(-1))
        pl.add_key_event("Up", lambda: self.set_snapshot(self.snap_index + 1))
        pl.add_key_event("Down", lambda: self.set_snapshot(self.snap_index - 1))
        pl.add_key_event("Home", lambda: self.set_snapshot(0))
        pl.add_key_event("e", lambda: self.set_snapshot(len(self.snapshots) - 1))
        pl.add_key_event("End", lambda: self.set_snapshot(len(self.snapshots) - 1))
        pl.add_key_event("b", lambda: self.set_snapshot(self.history["best_index"])
                         if self.history["best_index"] is not None else None)

    def _listed_entries(self) -> list[dict]:
        """The slices the panel lists: those of the listed modality (all of them with a single modality)."""
        return [e for e in self.entries if e["modality"] == self.listed_modality] if self._multi else self.entries

    def _sa_entries(self) -> list[dict]:
        return [e for e in self._listed_entries() if e["view"] == "SA"]

    def _short_name(self, x: dict, ref: dict | None = None) -> str:
        """Display name of slice x: without its <modality>/ prefix when it is of the modality of `ref` (default: the
        listed one); the prefix is only there with several modalities."""
        mod = ref["modality"] if ref is not None else self.listed_modality
        if self._multi and x["modality"] == mod and x["name"].startswith(x["modality"] + "/"):
            return label(x["name"][len(x["modality"]) + 1:])
        return label(x["name"])

    def _key_view(self, key: tuple) -> str:
        """The view of a slice key (image, slice), also for an image the subject does not show."""
        e = self._entry_by_key.get(key)
        return e["view"] if e is not None else image_view(key[0])

    def _toggle_sa_slice(self, d: int):
        sa = self._sa_entries()
        if d < len(sa):
            self._toggle_checkbox(sa[d]["name"])

    def _row_label(self, e: dict) -> str:
        """Slice-row label: the slice name and, for the first ten SA slices, its digit key."""
        sa = [x["name"] for x in self._sa_entries()]
        d = sa.index(e["name"]) if e["name"] in sa else None
        return self._short_name(e) + (f"  [{d}]" if d is not None and d < 10 else "")

    def _update_nav_text(self):
        self._nav_text.SetInput(f"subject {self.subject_index + 1}/{len(self.subjects)}  "
                                f"{self.subjects[self.subject_index]}")
        self._run_text.SetInput(f"run {self.run_index + 1}/{len(self.runs)}")     # the name is in the header

    # --- scene -----------------------------------------------------------
    def _build_slices(self):
        self._color_cache.clear()                        # (new pixels)
        self._color_settings = (bool(self.show_seg), float(self.seg_alpha))
        named = {k[0] for snap in self.snapshots for k in snap["affines"]}
        self._history_matches = not named or bool(named & set(self.data))   # (else load_subject_run has warned)
        for image, v in self.data.items():
            for s in v["slices"]:
                nx, ny, _ = s["grey"].shape
                key = (image, int(s["index"]))
                in_history = any(key in snap["affines"] for snap in self.snapshots)
                if not in_history and self._history_matches:   # e.g. a LA view without a segmentation: unaligned
                    print(f"[info] {s['name']}: not part of the alignment (no rows in history.csv); shown at its "
                          f"saved (acquired) pose for every snapshot", file=sys.stderr)
                # one textured quad (the image as a texture, recoloured per frame: _recolor_entry), built at the
                # slice's saved pose and moved to the pose of the shown step by its actor's matrix (_apply_snapshot)
                A0 = np.asarray(s["affine"], dtype=np.float64)
                quad, tex_img, tex = slice_quad(A0, nx, ny)
                actor = self._mesh(quad, texture=tex, color="white", lighting=False, show_scalar_bar=False,
                                   name=s["name"])
                outline = pv.lines_from_points(voxel_to_world(A0, corner_ijk(nx, ny)), close=True)
                outline.point_data["metric"] = np.full(outline.n_points, np.nan)
                outline_actor = self._mesh(
                    outline, scalars="metric", cmap=OUTLINE_CMAP, clim=(0.0, 1.0),
                    nan_color=OUTLINE_NAN_COLOR, show_scalar_bar=False,
                    line_width=OUTLINE_LINE_WIDTH, name=s["name"] + "_outline",
                )
                outline_actor.mapper.lookup_table = self._lut
                outline_actor.mapper.SetUseLookupTableScalarRange(True)
                self.entries.append({
                    "name": s["name"], "view": v["view"], "image": image, "modality": v["modality"],
                    "key": key, "slice": s,
                    "tex_img": tex_img, "texture": tex, "mesh_inv": np.linalg.inv(A0), "rgb_key": None,
                    "outline_mesh": outline,
                    "actor": actor, "outline": outline_actor, "visible": True,
                    "in_history": in_history,
                })
                self._entry_by_key[key] = self.entries[-1]
                self._recolor_entry(self.entries[-1])

    # --- overlays: dilated-mask contours and intersection sampling lines -----------------
    def _base_affine(self, key: tuple) -> np.ndarray | None:
        """Affine of a plane at the acquired geometry (the first snapshot, step 0)."""
        return self.snapshots[0]["affines"].get(key)

    def _plane_transform(self, key: tuple, snap: dict) -> np.ndarray:
        """World -> world map taking a plane from the acquired geometry to its pose at `snap`."""
        A0, A = self._base_affine(key), snap["affines"].get(key)
        if A0 is None or A is None:
            return np.eye(4)
        return A @ np.linalg.inv(A0)

    def _build_overlays(self):
        if self.snapshots[0]["step"] != 0 and (self.overlays["contours"] or self.overlays["lines"] is not None):
            print("[warn] the history has no step-0 snapshot; overlays are placed relative to the first "
                  "snapshot instead of the acquired geometry", file=sys.stderr)
        # Dilated-mask contours: polylines in voxel coordinates of their LA plane. With the run's config and the
        # LA segmentations the mask is rebuilt per step (annealed dilation, motion_gradient.annealed_stages);
        # otherwise the nominal contours of mask_contours.csv are used.
        self._dil_ijk_cache = {}
        keys = set(self.overlays["contours"])
        if self._dilation(0) is not None:
            keys |= {e["key"] for e in self.entries if e["view"] != "SA" and self._heart_dist(e) is not None}
        for key in sorted(keys):
            mesh = pv.PolyData(np.zeros((2, 3)), lines=np.array([2, 0, 1]))
            actor = self._mesh(mesh, color=MASK_CONTOUR_COLOR, line_width=MASK_CONTOUR_WIDTH,
                               lighting=False, name=f"mask:{key[0]}:{key[1]}")
            self._mask_items.append({"key": key, "ijk": None, "dil": "unset", "mesh": mesh, "actor": actor})

        # Long-axis radius: where the cylinder of the SA-SA radius around the long axis cuts each SA slice
        if self.overlays.get("long_axis"):
            n = RADIUS_SAMPLES
            ring = np.concatenate([[n + 1], np.arange(n), [0]])
            for e in self.entries:
                if e["view"] != "SA":
                    continue
                mesh = pv.PolyData(np.zeros((n, 3)), lines=ring)
                actor = self._mesh(mesh, color=RADIUS_COLOR, line_width=RADIUS_WIDTH, lighting=False,
                                   name=f"radius:{e['name']}")
                self._radius_items.append({"entry": e, "mesh": mesh, "actor": actor, "ok": False})

        # SA-SA sampling lines: stored per snapshot; burned into both slices of each pair (neighbouring SA slices, and
        # parallel slices of other images, e.g. of another modality)
        if self.overlays.get("sa_lines"):
            in_pairs = {k for pair in self.overlays.get("sa_pairs") or [] for k in pair}
            self._sa_line_items = {"per_slice": [{"entry": e, "burned": {}} for e in self.entries
                                                 if e["view"] == "SA" or e["key"] in in_pairs]}
            self._add_burned_actors(self._sa_line_items, "sa_lines", (("img", SA_LINE_IMG_WIDTH, 0.6),
                                                                      ("used", SA_LINE_USED_WIDTH, 1.0)))

        # Intersection lines: one segment per line, recomputed at every snapshot as the
        # intersection of the two planes at their current poses -- the line the loss samples
        # (motion_gradient.LineSampler.profiles: direction n_a x n_b, anchored at the moved
        # centre of plane b, same parameter range t as at the acquired geometry).
        df = self.overlays["lines"]
        if df is None or not len(df):
            return
        x0 = df[["x0_x", "x0_y", "x0_z"]].to_numpy(float)
        d = df[["d_x", "d_y", "d_z"]].to_numpy(float)
        t_used = df[["t_used_lo", "t_used_hi"]].to_numpy(float)
        t_img = df[["t_img_lo", "t_img_hi"]].to_numpy(float)
        bad = ~np.isfinite(t_img).all(1)
        t_img[bad] = t_used[bad]
        keys_a = list(zip(df["image_a"].astype(str).tolist(), df["slice_a"].astype(int).tolist()))
        keys_b = list(zip(df["image_b"].astype(str).tolist(), df["slice_b"].astype(int).tolist()))
        # centre of plane b in its voxel coordinates ((X-1)/2, (Y-1)/2, 0), as in make_planes
        centre_b = []
        for kb in keys_b:
            e = self._entry_by_key.get(kb)
            nx, ny = (e["slice"]["grey"].shape[:2] if e is not None else (np.nan, np.nan))
            centre_b.append([(nx - 1) / 2.0, (ny - 1) / 2.0, 0.0])
        kinds = list(df["kind"])
        virtual = [False] * len(df)
        # pairs missing from lines.csv that enter the loss at an earlier, wider mask dilation (see _virtual_lines)
        for v in self._virtual_lines({(a, b) for a, b in zip(keys_a, keys_b)}):
            keys_a.append(v["ka"]); keys_b.append(v["kb"]); kinds.append(v["kind"]); virtual.append(True)
            x0, d = np.vstack([x0, v["x0"]]), np.vstack([d, v["d"]])
            t_used = np.vstack([t_used, [np.nan, np.nan]]); t_img = np.vstack([t_img, v["t_img"]])
            centre_b.append(v["centre_b"])
        rgb = np.array([LINE_COLORS.get(k, (255, 255, 255)) for k in kinds], dtype=np.uint8)
        n = len(keys_a)
        # the slices of the lines (_line_geometry poses each once) and every slice's lines (_profile_items)
        ukeys = list(dict.fromkeys(keys_a + keys_b))
        pos = {k: u for u, k in enumerate(ukeys)}
        by_key = {}
        for i, (ka, kb) in enumerate(zip(keys_a, keys_b)):
            by_key.setdefault(ka, []).append(i)
            by_key.setdefault(kb, []).append(i)
        self._line_items = {"keys_a": keys_a, "keys_b": keys_b, "x0": x0, "d": d,
                            "t_used_nominal": t_used.copy(), "t_used_cache": {},
                            "t_img": t_img, "centre_b": np.array(centre_b, dtype=float).reshape(-1, 3), "n": n,
                            "virtual": np.array(virtual, bool), "per_slice": [],
                            "ukeys": ukeys, "ia": np.array([pos[k] for k in keys_a], int),
                            "ib": np.array([pos[k] for k in keys_b], int),
                            "by_key": {k: np.array(v, int) for k, v in by_key.items()}}
        # every slice gets its own burned-in copy of the lines it is part of
        for e in self.entries:
            idx = self._line_items["by_key"].get(e["key"])
            if idx is None or not idx.size:
                continue
            # LA-LA lines on this slice: the colour of the view they cross (LA_PARTNER_COLORS)
            col = rgb[idx].copy()
            la_la = np.array([self._key_view(keys_a[i]) != "SA" and self._key_view(keys_b[i]) != "SA" for i in idx])
            for j, i in enumerate(idx):
                if la_la[j]:
                    other = keys_b[i] if keys_a[i] == e["key"] else keys_a[i]
                    col[j] = LA_PARTNER_COLORS.get(self._key_view(other), col[j])
            self._line_items["per_slice"].append({"entry": e, "idx": idx, "rgb": col, "la_la": la_la, "burned": {}})
        self._add_burned_actors(self._line_items, "lines", (("img", LINE_IMG_WIDTH, 0.45),
                                                            ("used", LINE_USED_WIDTH, 1.0)), rgb=True)

    def _add_burned_actors(self, items: dict, tag: str, kinds: tuple, rgb: bool = False):
        """One actor per kind of burned-in line ("img", "used") holding the lines of every shown slice (_merge_burned):
        two actors instead of two per slice."""
        items["meshes"], items["actors"], items["count"] = {}, {}, {}
        for name, width, opacity in kinds:
            mesh = pv.PolyData(np.zeros((2, 3)), lines=np.array([2, 0, 1]))
            if rgb:
                mesh.cell_data["rgb"] = np.zeros((1, 3), np.uint8)
                actor = self._mesh(mesh, scalars="rgb", rgb=True, line_width=width, opacity=opacity, lighting=False,
                                   show_scalar_bar=False, name=f"{tag}_{name}")
            else:
                actor = self._mesh(mesh, color=SA_LINE_COLOR, line_width=width, opacity=opacity, lighting=False,
                                   name=f"{tag}_{name}")
            actor.SetVisibility(False)
            items["meshes"][name], items["actors"][name], items["count"][name] = mesh, actor, 0

    def _merge_burned(self, items: dict | None, shown: bool):
        """Put the burned-in segments of every shown slice (item["burned"][kind] = (points (4K, 3), colours (2K, 3) or
        None), from burn_points) into the kind's one mesh, and show it when there is something to show."""
        if items is None or "meshes" not in items:
            return
        for name, mesh in items["meshes"].items():
            parts = [it["burned"][name] for it in items["per_slice"]
                     if it["entry"]["visible"] and name in it["burned"] and len(it["burned"][name][0])]
            k = sum(len(pts) for pts, _ in parts) // 4
            items["count"][name] = k if shown else 0
            if not shown:
                items["actors"][name].SetVisibility(False)
                continue
            if k:
                new = pv.PolyData(np.concatenate([pts for pts, _ in parts]), lines=burn_cells(k))
                if "rgb" in mesh.cell_data:
                    new.cell_data["rgb"] = np.concatenate([cols for _, cols in parts])
            else:                                        # placeholder (hidden)
                new = pv.PolyData(np.zeros((2, 3)), lines=np.array([2, 0, 1]))
                if "rgb" in mesh.cell_data:
                    new.cell_data["rgb"] = np.zeros((1, 3), np.uint8)
            mesh.copy_from(new)
            items["actors"][name].SetVisibility(bool(k))

    def _apply_overlays(self):
        """Move contours and lines to the planes' poses at the current snapshot -- only the overlays that are shown;
        the others are brought up to date when switched on."""
        snap = self.snapshots[self.snap_index]
        self._apply_masks(snap)
        self._apply_sa_lines(snap)
        self._apply_radius(snap)
        self._apply_lines(snap)

    def _apply_masks(self, snap: dict):
        if not self.show_mask:
            return
        dil = self._dilation(snap["step"])
        for m in self._mask_items:
            if m["dil"] != dil:                          # (re)build the contour for this step's dilation
                ijk = self._mask_contour_ijk(m["key"], dil)
                m["dil"], m["ijk"] = dil, ijk
                if ijk is None:
                    m["mesh"].copy_from(pv.PolyData(np.zeros((2, 3)), lines=np.array([2, 0, 1])))
                else:
                    m["mesh"].copy_from(pv.PolyData(np.zeros((len(ijk[0]), 3)), lines=ijk[1]))
            A = self._pose(m["key"], snap)
            if A is not None and m["ijk"] is not None:
                m["mesh"].points = voxel_to_world(A, m["ijk"][0])

    def _apply_lines(self, snap: dict):
        """The burned-in intersection lines at `snap` (when shown): each slice's own copy of its lines (_dashed), all
        in two meshes (_merge_burned)."""
        li = self._line_items
        if li is None or not self.show_lines:
            return
        x0, d, t_used = self._line_geometry(snap)
        t_cur = {"used": t_used, "img": li["t_img"]}
        off = li.get("virtual", np.zeros(li["n"], bool)) & ~np.isfinite(t_used).all(1)  # virtual, not in loss
        for name in ("used", "img"):
            t = t_cur[name]                                     # (N, 2)
            t = np.where(np.isfinite(t), t, li["t_img"][:, :1])  # no loss points: zero-length segment
            t = np.where(off[:, None], li["t_img"][:, :1], t)
            seg = np.stack([x0 + t[:, :1] * d, x0 + t[:, 1:] * d], axis=1)   # (N, 2, 3)
            for item in li["per_slice"]:
                A = self._affine_for(item["entry"], snap)
                segs, cols = self._dashed(item, seg[item["idx"]], A)
                if not len(segs):                                # placeholder: a zero-length segment on the slice
                    first = seg[item["idx"][0], 0]
                    segs, cols = np.stack([first, first])[None], item["rgb"][:1]
                item["burned"][name] = (burn_points(segs, A), np.concatenate([cols, cols]))
        self._merge_burned(li, True)

    def _dashed(self, item: dict, seg: np.ndarray, A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The segments (K, 2, 3) and colours (K, 3) to burn into one slice: its lines as they are, except LA-LA
        lines that coincide on this slice (directions within COINCIDE_DEG, midpoints within COINCIDE_MM of the other
        line, in the slice plane), which are cut into DASH_MM dashes along a common axis and interleaved: dash k
        belongs to the k mod m-th of the m coinciding lines, so each keeps its own colour where they overlap."""
        n = np.cross(A[:3, 0], A[:3, 1]); n /= np.linalg.norm(n)
        p = seg - ((seg - A[:3, 3]) @ n)[..., None] * n                     # in the slice plane
        rows = np.flatnonzero(item["la_la"])
        groups = []
        for i in rows:
            di = p[i, 1] - p[i, 0]
            li_ = np.linalg.norm(di)
            for g in groups:
                j = g[0]
                dj = p[j, 1] - p[j, 0]
                lj = np.linalg.norm(dj)
                if li_ < 1e-6 or lj < 1e-6:
                    continue
                cos = abs(di @ dj) / (li_ * lj)
                mid = p[i].mean(0) - p[j, 0]
                off = np.linalg.norm(mid - (mid @ dj) / lj ** 2 * dj)
                if cos >= np.cos(np.radians(COINCIDE_DEG)) and off <= COINCIDE_MM:
                    g.append(i)
                    break
            else:
                groups.append([i])
        dashed = {i for g in groups if len(g) > 1 for i in g}
        segs = [seg[i] for i in range(len(seg)) if i not in dashed]
        cols = [item["rgb"][i] for i in range(len(seg)) if i not in dashed]
        for g in groups:
            if len(g) < 2:
                continue
            u = p[g[0], 1] - p[g[0], 0]
            u /= np.linalg.norm(u)
            for r, i in enumerate(g):
                s0, s1 = p[i] @ u                                          # positions along the common axis
                if abs(s1 - s0) < 1e-9:
                    continue
                lo, hi = min(s0, s1), max(s0, s1)
                for kk in range(int(np.floor(lo / DASH_MM)), int(np.floor(hi / DASH_MM)) + 1):
                    if kk % len(g) != r:
                        continue
                    a_, b_ = max(lo, kk * DASH_MM), min(hi, (kk + 1) * DASH_MM)
                    if b_ <= a_:
                        continue
                    fa, fb = (a_ - s0) / (s1 - s0), (b_ - s0) / (s1 - s0)
                    segs.append(np.stack([seg[i, 0] + fa * (seg[i, 1] - seg[i, 0]),
                                          seg[i, 0] + fb * (seg[i, 1] - seg[i, 0])]))
                    cols.append(item["rgb"][i])
        if not segs:
            return np.zeros((0, 2, 3)), np.zeros((0, 3), np.uint8)
        return np.array(segs), np.array(cols, dtype=np.uint8)

    def _line_geometry(self, snap: dict, idx: np.ndarray | None = None):
        """(x0, d, t_used) of the SA-LA / LA-LA lines `idx` (default: all) at `snap` (manual offsets included): the
        intersection of its two posed planes, anchored at the moved centre of plane b, and the loss range of the
        snapshot's mask dilation. All lines at once (batched cross products and 3x3 solves); a line whose planes are
        (nearly) parallel at this pose, or a plane without a pose, keeps its acquired geometry."""
        li = self._line_items
        sel = np.arange(li["n"]) if idx is None else np.asarray(idx, dtype=int)
        x0, d = li["x0"][sel].copy(), li["d"][sel].copy()      # acquired geometry (fallback)
        if len(sel):
            P = np.full((len(li["ukeys"]), 4, 4), np.nan)       # every slice of the lines posed once
            for u, key in enumerate(li["ukeys"]):
                A = self._pose(key, snap)
                if A is not None:
                    P[u] = A
            Aa, Ab, cb_ijk = P[li["ia"][sel]], P[li["ib"][sel]], li["centre_b"][sel]
            with np.errstate(invalid="ignore", divide="ignore"):
                na = np.cross(Aa[:, :3, 0], Aa[:, :3, 1])
                na /= np.linalg.norm(na, axis=1, keepdims=True)
                nb = np.cross(Ab[:, :3, 0], Ab[:, :3, 1])
                nb /= np.linalg.norm(nb, axis=1, keepdims=True)
                di = np.cross(na, nb)
                nd = np.linalg.norm(di, axis=1)
                ok = np.isfinite(nd) & (nd >= 1e-6) & np.isfinite(cb_ijk).all(1)
            if ok.any():
                di = di[ok] / nd[ok, None]
                cb = np.einsum("nij,nj->ni", Ab[ok, :3, :3], cb_ijk[ok]) + Ab[ok, :3, 3]
                rhs = np.stack([(na[ok] * Aa[ok, :3, 3]).sum(1), (nb[ok] * Ab[ok, :3, 3]).sum(1), (di * cb).sum(1)], 1)
                x0[ok] = np.linalg.solve(np.stack([na[ok], nb[ok], di], axis=1), rhs[..., None])[..., 0]
                d[ok] = di
        t_used = self._stage_t_used(self._dilation(snap["step"]))
        return x0, d, (t_used if idx is None else t_used[sel])

    def _apply_sa_lines(self, snap: dict):
        """Burn the stored SA-SA segments of this snapshot into both slices of their pair
        (none where the slices do not cross); only when shown."""
        si = self._sa_line_items
        if si is None or not self.show_sa_lines:
            return
        segs = self._sa_segments(snap)
        for item in si["per_slice"]:
            key = item["entry"]["key"]
            A = self._affine_for(item["entry"], snap)
            for name in ("img", "used"):
                e = segs.get(name, np.zeros((0, 2, 3)))
                pairs = segs.get(name + "_pairs", [])
                mine = e[np.array([key in pr for pr in pairs], bool)] if len(e) else e
                item["burned"][name] = (burn_points(mine, A), None) if len(mine) else (np.zeros((0, 3)), None)
        self._merge_burned(si, True)

    def _parallel_pairs(self) -> list[tuple]:
        """Pairs of parallel slices (slice keys): every pair of sa_lines.csv, and the neighbouring slices of each SA
        stack."""
        pairs = list(self.overlays.get("sa_pairs") or [])
        stacks = {}
        for e in self.entries:
            if e["view"] == "SA":
                stacks.setdefault(e["image"], []).append(e["key"])
        for keys in stacks.values():
            keys = sorted(keys, key=lambda k: k[1])
            pairs += [(a, b) for a, b in zip(keys, keys[1:]) if b[1] == a[1] + 1]
        return list(dict.fromkeys(pairs))

    def _sa_segments(self, snap: dict) -> dict:
        """Parallel-pair (SA-SA) segments of this snapshot: as stored, except pairs with a manually moved slice, whose
        crossing is recomputed (inside both images; the loss part within the long-axis radius)."""
        segs = (self.overlays.get("sa_lines") or {}).get((snap["step"], bool(snap["is_best"])), {})
        moved = {k for k, off in self.offsets.items() if np.any(off)}
        if not moved:
            return segs
        axis = (self.overlays.get("long_axis") or {}).get((snap["step"], bool(snap["is_best"])))
        out = {}
        for name in ("img", "used"):
            e, pr = segs.get(name, np.zeros((0, 2, 3))), segs.get(name + "_pairs", [])
            keep = [j for j, pair in enumerate(pr) if not set(pair) & moved]
            out[name], out[name + "_pairs"] = [e[j] for j in keep], [pr[j] for j in keep]
        for ka, kb in self._parallel_pairs():
            ea, eb = self._entry_by_key.get(ka), self._entry_by_key.get(kb)
            if ea is None or eb is None or not {ka, kb} & moved:
                continue
            cs = crossing_segment(self._pose(ea["key"], snap), ea["slice"]["grey"].shape,
                                  self._pose(eb["key"], snap), eb["slice"]["grey"].shape)
            if cs is None:
                continue
            x0, d, lo, hi = cs
            out["img"].append(np.stack([x0 + lo * d, x0 + hi * d])); out["img_pairs"].append((ka, kb))
            if axis is not None:                         # within the radius: |perp(x0 + t d - p)| <= r
                p0, ell, r = axis
                ell = ell / np.linalg.norm(ell)
                w0 = x0 - p0
                a_, b_ = d - (d @ ell) * ell, w0 - (w0 @ ell) * ell
                A2, B2, C2 = a_ @ a_, 2 * a_ @ b_, b_ @ b_ - r * r
                disc = B2 * B2 - 4 * A2 * C2
                if A2 > 1e-12 and disc > 0:
                    t1, t2 = (-B2 - np.sqrt(disc)) / (2 * A2), (-B2 + np.sqrt(disc)) / (2 * A2)
                    u1, u2 = max(lo, t1), min(hi, t2)
                    if u2 > u1:
                        out["used"].append(np.stack([x0 + u1 * d, x0 + u2 * d])); out["used_pairs"].append((ka, kb))
        for name in ("img", "used"):
            out[name] = np.array(out[name]).reshape(-1, 2, 3)
        return out

    def move_selected(self, param: int, sign: int):
        """Nudge the profile-selected slice by one step of parameter `param` (see MOVE_PARAMS)."""
        e = self._entry_by_name().get(self.profile_key)
        if e is None:
            return
        off = self.offsets.setdefault(e["key"], np.zeros(6))
        off[param] += sign * MOVE_STEP[0 if param < 3 else 1]
        self._apply_snapshot()
        self._update_move_text()
        self.pl.render()

    def reset_offsets(self):
        self.offsets = {}
        self._log("manual poses reset")
        self._apply_snapshot()
        self._update_move_text()
        self.pl.render()

    def _update_move_text(self):
        if not hasattr(self, "_move_text") or not self.entries:
            return
        e = self._entry_by_name().get(self.profile_key)
        off = self.offsets.get(e["key"], np.zeros(6)) if e is not None else np.zeros(6)
        n_moved = sum(bool(np.any(o)) for o in self.offsets.values())
        self._move_text.SetInput(f"Manual correction {self._short_name(e, e) if e else '-'}")
        self._reset_text.SetInput("reset all slices" + (f"  ({n_moved} moved)" if n_moved else ""))
        for k, (name, unit) in enumerate(MOVE_PARAMS):
            self._move_values[k].SetInput(f"{off[k]:+.0f} {unit}")

    def _apply_radius(self, snap: dict):
        """Intersect the radius cylinder around this snapshot's long axis with every SA slice at its pose:
        each cylinder generator p + r (cos f e1 + sin f e2) + s ell meets the slice plane at one point."""
        if not self._radius_items or not self.show_radius:
            return
        axis = self.overlays["long_axis"].get((snap["step"], bool(snap["is_best"])))
        if axis is not None:
            p0, ell, r = axis
            ell = ell / np.linalg.norm(ell)
            e1 = np.cross(ell, [1.0, 0.0, 0.0] if abs(ell[0]) < 0.9 else [0.0, 1.0, 0.0])
            e1 /= np.linalg.norm(e1)
            e2 = np.cross(ell, e1)
            phi = np.linspace(0.0, 2 * np.pi, RADIUS_SAMPLES, endpoint=False)
            gen = p0 + r * (np.cos(phi)[:, None] * e1 + np.sin(phi)[:, None] * e2)     # (n, 3)
        for item in self._radius_items:
            item["ok"] = False
            if axis is None:
                continue
            A = self._affine_for(item["entry"], snap)
            nrm = np.cross(A[:3, 0], A[:3, 1]); nrm /= np.linalg.norm(nrm)
            denom = nrm @ ell
            if abs(denom) < 1e-3:
                continue                                   # axis (nearly) in the slice plane: no ellipse
            sgen = ((A[:3, 3] - gen) @ nrm) / denom
            item["mesh"].points = gen + sgen[:, None] * ell
            item["ok"] = True
        self._apply_radius_visibility()

    def _apply_radius_visibility(self):
        for item in self._radius_items:
            item["actor"].SetVisibility(self.show_radius and item["ok"] and item["entry"]["visible"])

    def _apply_sa_line_visibility(self):
        self._merge_burned(self._sa_line_items, self.show_sa_lines)

    def _apply_line_visibility(self):
        self._merge_burned(self._line_items, self.show_lines)

    # --- intersection intensity profiles ---------------------------------------
    def set_profile(self, name: str):
        e = self._entry_by_name().get(name)
        if e is None:
            return
        if self._multi and e["modality"] != self.listed_modality:     # list the modality of the profiled slice
            self.set_listed_modality(e["modality"])
        if name != self.profile_key:
            self._profile_scroll = self._scroll_target = 0.0
        self.profile_key = name
        for n, w in self._profile_radios.items():
            w.GetRepresentation().SetState(int(n == name))
        self._log(f"profiles / move: {label(name)}")
        self._update_move_text()
        self._update_profiles()
        self.pl.render()

    def _menu_entries(self, modality: str | None = None) -> list[dict]:
        """The slices of a modality (default: all) in the order of the left-hand menu: the aortic slice, then the LA
        views (VIEW_RANK order), then the SA slices (any other view after the LA views, before SA)."""
        rank = {"ao": 0, "SA": 2}
        rows = [e for e in self.entries if modality is None or e["modality"] == modality]
        return sorted(rows, key=lambda e: rank.get(e["view"], 1))      # stable: VIEW_RANK / slice order within

    def step_profile(self, step: int):
        names = [e["name"] for e in self._menu_entries(self.listed_modality if self._multi else None)]
        # , / . walk the menu (of the listed modality) top-down
        if names:
            i = names.index(self.profile_key) if self.profile_key in names else 0
            self.set_profile(names[(i + step) % len(names)])

    def _entry_by_name(self) -> dict:
        return {e["name"]: e for e in self.entries}

    def _sample(self, e: dict, snap: dict, pts: np.ndarray) -> np.ndarray:
        """Displayed grey values (0-255) of slice `e` at world points pts (L, 3), current frame;
        nan outside the image."""
        A = self._affine_for(e, snap)
        vox = (np.linalg.inv(A) @ np.column_stack([pts, np.ones(len(pts))]).T)[:3]
        grey = e["slice"]["grey"][:, :, self.frame].astype(np.float32)
        return map_coordinates(grey, vox[:2], order=1, mode="constant", cval=np.nan)

    # --- dilated mask (annealed over the run) ------------------------------------------
    def _dilation(self, step: int) -> float | None:
        """Mask dilation (mm) of the loss at `step`: linear from --mask-dilation-start-mm to --mask-dilation-mm over
        the first --mask-anneal-frac of the run in --anneal-stages discrete stages (motion_gradient.annealed_stages),
        nominal afterwards. None without the run's config."""
        cfg = getattr(self, "config", None)
        if not cfg or cfg.get("mask_dilation_mm") is None:
            return None
        nominal = float(cfg["mask_dilation_mm"])
        d0 = cfg.get("mask_dilation_start_mm")
        steps = max(1, int(cfg.get("steps", 1)))
        anneal_end = int(round(float(cfg.get("mask_anneal_frac", 0.8)) * steps)) if d0 is not None else 0
        if anneal_end <= 0 or step >= anneal_end:
            return nominal
        n_st = max(1, int(cfg.get("anneal_stages", 6)))
        starts = sorted({int(round(k / n_st * anneal_end)) for k in range(n_st)})
        k = max(i for i, st in enumerate(starts) if step >= st)
        return float(d0) + (nominal - float(d0)) * k / n_st

    def _heart_dist(self, e: dict) -> np.ndarray | None:
        """Distance (mm) of every voxel of LA slice e to the heart (its segmentation, union over frames), as the
        alignment script computes it -- with the run's --apex-extend-mm the heart region first swept toward the apex
        (apical_sweep, at the acquired geometry); None without a segmentation."""
        extend = float((getattr(self, "config", None) or {}).get("apex_extend_mm") or 0.0)
        if e.get("heart_dist_extend") != extend:
            seg = e["slice"]["seg"]
            if seg is None or not (seg > 0).any():
                e["heart_dist"] = None
            else:
                A = e["slice"]["affine"]
                sp = (np.linalg.norm(A[:3, 0]), np.linalg.norm(A[:3, 1]))
                heart = (seg > 0).any(-1)
                if extend > 0:
                    snap0 = self.snapshots[0]["affines"]
                    sa = [x for x in self.entries if x["view"] == "SA"]
                    if sa:                               # the largest SA stack (as motion_gradient.prepare_subject)
                        n_sa = {}
                        for x in sa:
                            n_sa[x["image"]] = n_sa.get(x["image"], 0) + 1
                        big = max(n_sa, key=n_sa.get)
                        sa = sorted((x for x in sa if x["image"] == big), key=lambda x: x["key"][1])
                    apex = apex_direction([(snap0.get(x["key"], x["slice"]["affine"]), x["slice"]["grey"].shape[:2],
                                            x["slice"]["seg"]) for x in sa],
                                          [(snap0.get(x["key"], x["slice"]["affine"]), x["slice"]["seg"])
                                           for x in self.entries if x["view"] != "SA"])
                    heart = apical_sweep(heart, snap0.get(e["key"], A), apex, extend)
                e["heart_dist"] = distance_transform_edt(~heart, sampling=sp)
            e["heart_dist_extend"] = extend
        return e["heart_dist"]

    def _mask_contour_ijk(self, key: tuple, dil: float | None):
        """(points (N, 3) voxel ijk, line cells) of slice `key`'s dilated-mask outline at dilation `dil`."""
        cache_key = (key, dil)
        if cache_key in self._dil_ijk_cache:
            return self._dil_ijk_cache[cache_key]
        e = self._entry_by_key.get(key)
        dist = self._heart_dist(e) if (e is not None and dil is not None) else None
        if dist is not None:
            contours = [np.column_stack([c[:, 1], c[:, 0], np.zeros(len(c))])           # contourpy (x, y) = (j, i)
                        for c in contour_generator(z=(dist <= dil).astype(float)).lines(0.5) if len(c) > 1]
        else:
            contours = self.overlays["contours"].get(key, [])
        out = None
        if contours:
            pts, cells, off = [], [], 0
            for c in contours:
                pts.append(c)
                cells.append(np.concatenate([[len(c)], np.arange(off, off + len(c))]))
                off += len(c)
            out = (np.vstack(pts), np.concatenate(cells))
        self._dil_ijk_cache[cache_key] = out
        return out

    def _virtual_lines(self, existing: set) -> list[dict]:
        """Candidate pairs missing from lines.csv. lines.csv holds the lines of the nominal mask dilation, but with
        mask annealing (--mask-dilation-start-mm) the optimizer rebuilds its lines at every stage, and at the wider
        early dilations more pairs have --min-line-mm inside the mask (e.g. Ao x 4Ch, basal / apical SA x LA). Each
        such pair is rebuilt here as motion_gradient.intersection_line does at the acquired geometry (direction
        n_a x n_b, anchored at the centre of plane b, the --line-extent-mm / --line-step-mm grid, the part inside
        both images); whether and where it enters the loss at a step is decided by _stage_t_used. Pairs: two images
        crossing at --min-pair-angle-deg or more, one of them a LA view with a segmentation (a mask that anneals),
        not two stacks of several slices (thinned by candidate_pairs). None without mask annealing."""
        cfg = getattr(self, "config", None) or {}
        if cfg.get("mask_dilation_start_mm") is None or cfg.get("mask_dilation_mm") is None:
            return []
        snap0 = self.snapshots[0]
        ext, step = float(cfg.get("line_extent_mm", 150.0)), float(cfg.get("line_step_mm", 0.5))
        cos_max = math.cos(math.radians(float(cfg.get("min_pair_angle_deg", 20.0))))
        T = np.arange(-ext, ext + 1e-6, step)
        n_slices = {}
        for e in self.entries:
            n_slices[e["image"]] = n_slices.get(e["image"], 0) + 1

        def frame(e):
            A = snap0["affines"].get(e["key"], e["slice"]["affine"])
            nrm = np.cross(A[:3, 0], A[:3, 1])
            return A, nrm / np.linalg.norm(nrm)

        def inside(e, A, pts):
            nx, ny = e["slice"]["grey"].shape[:2]
            vox = (np.linalg.inv(A) @ np.column_stack([pts, np.ones(len(pts))]).T)[:2]
            return (vox[0] >= 0) & (vox[0] <= nx - 1) & (vox[1] >= 0) & (vox[1] <= ny - 1)

        out = []
        for i, a in enumerate(self.entries):
            for b in self.entries[i + 1:]:
                if a["image"] == b["image"] or (n_slices[a["image"]] > 1 and n_slices[b["image"]] > 1):
                    continue
                if (a["key"], b["key"]) in existing or (b["key"], a["key"]) in existing:
                    continue
                if not any(x["view"] != "SA" and self._heart_dist(x) is not None for x in (a, b)):
                    continue
                (Aa, na), (Ab, nb) = frame(a), frame(b)
                if abs(float(na @ nb)) > cos_max:
                    continue
                d = np.cross(na, nb)
                d /= np.linalg.norm(d)
                nx, ny = b["slice"]["grey"].shape[:2]
                cb_ijk = [(nx - 1) / 2.0, (ny - 1) / 2.0, 0.0]
                cb = (Ab @ np.append(cb_ijk, 1.0))[:3]
                x0 = np.linalg.solve(np.stack([na, nb, d]), np.array([na @ Aa[:3, 3], nb @ Ab[:3, 3], d @ cb]))
                pts = x0 + T[:, None] * d
                ok = inside(a, Aa, pts) & inside(b, Ab, pts)
                if not ok.any():
                    continue
                out.append(dict(ka=a["key"], kb=b["key"], x0=x0, d=d, t_img=[T[ok].min(), T[ok].max()],
                                centre_b=cb_ijk, kind="SA-LA" if "SA" in (a["view"], b["view"]) else "LA-LA"))
        return out

    def _stage_t_used(self, dil: float | None) -> np.ndarray:
        """Loss range (t_lo, t_hi) of every SA-LA / LA-LA line at dilation `dil`: as intersection_line, the first
        and last point (--line-step-mm) at the acquired geometry inside both images and inside the dilated mask of
        either LA plane. The logged (nominal) ranges at the nominal dilation or without the segmentations, and
        for lines without any LA mask (SA-Ao: masked by the fixed SA-SA cylinder, which does not anneal). Virtual
        lines (_virtual_lines, not in lines.csv) enter only where --min-line-mm of them is inside the mask, as
        build_lines keeps them; elsewhere (NaN, NaN): not in the loss at that dilation."""
        li = self._line_items
        cfg = getattr(self, "config", None) or {}
        if dil is None or cfg.get("mask_dilation_mm") is None or abs(dil - float(cfg["mask_dilation_mm"])) < 1e-9:
            return li["t_used_nominal"]
        if dil in li["t_used_cache"]:
            return li["t_used_cache"][dil]
        snap0 = self.snapshots[0]
        step_mm = float(cfg.get("line_step_mm", 0.5))
        ext = float(cfg.get("line_extent_mm", 150.0))
        grid = np.arange(-ext, ext + 1e-6, step_mm)
        min_pts = float(cfg.get("min_line_mm", 40.0)) / step_mm
        virtual = li.get("virtual", np.zeros(li["n"], bool))
        out = li["t_used_nominal"].copy()
        for i in range(li["n"]):
            lo, hi = li["t_img"][i]
            if not (np.isfinite(lo) and np.isfinite(hi)):
                continue
            t = (grid[(grid >= lo - 1e-6) & (grid <= hi + 1e-6)] if virtual[i]
                 else np.arange(lo, hi + 1e-6, step_mm))
            pts = li["x0"][i] + t[:, None] * li["d"][i]
            inside = np.zeros(len(t), bool)
            have = False
            for key in (li["keys_a"][i], li["keys_b"][i]):
                e = self._entry_by_key.get(key)
                if e is None or e["view"] == "SA":
                    continue
                m = self._in_dist_mask(e, snap0["affines"].get(key, e["slice"]["affine"]), pts, dil)
                if m is not None:
                    inside |= m
                    have = True
            if have:
                enough = inside.sum() >= min_pts if virtual[i] else inside.any()   # (virtual: kept as build_lines)
                out[i] = (t[inside].min(), t[inside].max()) if enough else (np.nan, np.nan)
        li["t_used_cache"][dil] = out
        return out

    def _in_dist_mask(self, e: dict, A: np.ndarray, pts: np.ndarray, dil: float) -> np.ndarray | None:
        """Nearest-voxel lookup (as grid_sample 'nearest') of e's dilated mask (dist <= dil) at world points pts."""
        dist = self._heart_dist(e)
        if dist is None:
            return None
        vox = (np.linalg.inv(A) @ np.column_stack([pts, np.ones(len(pts))]).T)[:2]
        i, j = np.rint(vox[0]).astype(int), np.rint(vox[1]).astype(int)
        ok = (i >= 0) & (i < dist.shape[0]) & (j >= 0) & (j < dist.shape[1])
        out = np.zeros(len(pts), bool)
        out[ok] = dist[i[ok], j[ok]] <= dil
        return out

    def _mask_paths(self) -> dict:
        """Dilated-mask contours per LA slice key as polygons in its voxel (i, j) plane (cached per subject)."""
        cache = getattr(self, "_mask_path_cache", None)
        if cache is None or cache[0] is not self.overlays:
            paths = {key: [PolyPath(c[:, :2]) for c in cs if len(c) > 2]
                     for key, cs in (self.overlays.get("contours") or {}).items()}
            self._mask_path_cache = cache = (self.overlays, paths)
        return cache[1]

    def _in_mask(self, e: dict, snap: dict, pts: np.ndarray) -> np.ndarray | None:
        """Which world points lie inside slice e's dilated mask at its current pose (even-odd over its
        contours); None if the slice has no mask."""
        dil = self._dilation(snap["step"])
        if dil is not None and e["view"] != "SA" and self._heart_dist(e) is not None:
            return self._in_dist_mask(e, self._affine_for(e, snap), pts, dil)
        paths = self._mask_paths().get(e["key"])
        if not paths:
            return None
        A = self._affine_for(e, snap)
        vox = (np.linalg.inv(A) @ np.column_stack([pts, np.ones(len(pts))]).T)[:2].T
        inside = np.zeros(len(pts), bool)
        for path in paths:
            inside ^= path.contains_points(vox)
        return inside

    def _in_cylinder(self, e: dict, o: dict, snap: dict, pts: np.ndarray) -> np.ndarray | None:
        """For a line between an SA slice and the aortic slice (no segmentation, so no LA mask): which world points
        lie inside the SA-SA cylinder (radius around the long axis of this snapshot, long_axis.csv), the mask
        motion_gradient.build_lines gives these lines; None for any other line or without long_axis.csv."""
        if {e["view"], o["view"]} != {"SA", "ao"} or not self.overlays.get("long_axis"):
            return None
        axis = self.overlays["long_axis"].get((snap["step"], bool(snap["is_best"])))
        if axis is None:
            return None
        p0, ell, r = axis
        ell = ell / np.linalg.norm(ell)
        rel = pts - p0
        perp = rel - (rel @ ell)[:, None] * ell
        return (perp * perp).sum(1) <= r * r

    # --- profiles as the loss sees them -----------------------------------------------
    def _loss_state(self, step: int) -> tuple[float, float]:
        """(PSF scale, in-plane blur sigma in mm) the optimizer used at `step` (motion_gradient.align)."""
        cfg = self.config or {}
        steps = max(1, int(cfg.get("steps", 1)))
        psf_s0 = 1.0 if cfg.get("no_psf") else float(cfg.get("psf_scale_start", 1.0) or 1.0)
        psf_steps = int(round(float(cfg.get("psf_anneal_frac", 0.6)) * steps)) if psf_s0 > 1.0 else 0
        scale = psf_s0 + (1.0 - psf_s0) * step / psf_steps if step < psf_steps else 1.0
        # as motion_gradient.align: from --blur-sigma linearly down to --blur-sigma-end over --blur-frac of the run, then
        # held; --blur-frac 0: --blur-sigma throughout; applied rounded up to 0.2 mm (its blur_quantum)
        b0 = float(cfg.get("blur_sigma", 0.0) or 0.0)
        b_end = min(float(cfg.get("blur_sigma_end", 0.0) or 0.0), b0)
        b_frac = float(cfg.get("blur_frac", 0.6))
        if b0 > 0 and b_frac <= 0:                       # --blur-frac 0: constant blur for the whole run
            sigma = b0
        else:
            b_steps = int(round(b_frac * steps)) if b0 > 0 else 0
            sigma = b_end + (b0 - b_end) * (1.0 - step / b_steps) if step < b_steps else b_end
        sigma = math.ceil(sigma / BLUR_QUANTUM_MM) * BLUR_QUANTUM_MM if sigma > 0 else 0.0
        return scale, sigma

    def _band(self, scale: float) -> tuple[np.ndarray, np.ndarray]:
        """PSF band offsets u (units of the other slice's thickness) and weights at PSF scale `scale`
        (motion_gradient.LineSampler: slice_profile / band_weights)."""
        cfg = self.config or {}
        n_band = 1 if cfg.get("no_psf") else int(cfg.get("n_band", 1))
        model, tbw = cfg.get("psf_model", "pulse"), float(cfg.get("psf_tbw", 2.5))
        smax = max(1.0, 1.0 if cfg.get("no_psf") else float(cfg.get("psf_scale_start", 1.0) or 1.0))
        if model == "gaussian":
            sig, support = 1.0 / 2.3548, 2.0 / 2.3548
        elif model == "rect":
            sig, support = 0.0, 0.5
        else:
            sig = 1.0 / tbw / 2.5631
            support = 0.5 + 2.5 * sig
        K = max(1, int(math.ceil(max(1, n_band) * smax))) if n_band > 1 else 1
        if K == 1:
            return np.zeros(1), np.ones(1)
        u = np.linspace(-support * smax, support * smax, K)
        x = u / scale
        if model == "gaussian":
            w = np.exp(-0.5 * (x / sig) ** 2)
        elif model == "rect":
            w = (np.abs(x) <= 0.5).astype(float)
        else:
            from scipy.special import erf
            sq = sig * math.sqrt(2.0)
            w = 0.5 * (erf((x + 0.5) / sq) - erf((x - 0.5) / sq))
        return u, w / w.sum()

    def _thickness(self, e: dict) -> float:
        cfg = self.config or {}
        if e["view"] == "SA":
            return float(cfg.get("sax_thickness", 8.0))
        t = cfg.get("lax_thickness")
        return float(t) if t else float(e["slice"]["thick_hdr"])

    def _blurred(self, e: dict, sigma: float, frame: int | None = None, keep: bool = True) -> np.ndarray:
        """Raw intensities of a frame (default: the shown one), in-plane Gaussian blurred with sigma mm (as
        LineSampler.set_blur); kept for reuse (the last 64) unless keep is False."""
        frame = self.frame if frame is None else frame
        key = (e["name"], frame, round(sigma, 4))
        out = self._blur_cache.get(key)
        if out is None:
            raw = e["slice"]["raw"][:, :, frame].astype(np.float32) * np.float32(e["slice"].get("raw_scale", 1.0))
            if sigma > 1e-6:
                A = e["slice"]["affine"]
                sp = (np.linalg.norm(A[:3, 0]), np.linalg.norm(A[:3, 1]))
                raw = gaussian_filter(raw, (max(sigma / sp[0], 1e-3), max(sigma / sp[1], 1e-3)), mode="nearest",
                                      truncate=3.0)
            out = raw
            if keep:
                if len(self._blur_cache) > 64:
                    self._blur_cache.clear()
                self._blur_cache[key] = out
        return out

    def _loss_profile(self, e: dict, other: dict, snap: dict, pts: np.ndarray, sigma: float, scale: float,
                      band: bool) -> np.ndarray:
        """Slice e's profile along the line as the loss samples it: blurred raw image, averaged over the PSF band
        across `other`'s slab (offsets along the in-plane direction of e that crosses other, u T / sin(theta)),
        mapped to the view's display scale (0-255, unclipped)."""
        A = self._affine_for(e, snap)
        img = self._blurred(e, sigma)
        Ai = np.linalg.inv(A)
        u, w = self._band(scale) if band else (np.zeros(1), np.ones(1))
        if len(u) > 1:
            Ao = self._affine_for(other, snap)
            n_e = np.cross(A[:3, 0], A[:3, 1]); n_e /= np.linalg.norm(n_e)
            n_o = np.cross(Ao[:3, 0], Ao[:3, 1]); n_o /= np.linalg.norm(n_o)
            m = n_o - (n_o @ n_e) * n_e
            sin = max(np.linalg.norm(m), 1e-3)
            m /= sin
            step = self._thickness(other) / sin
        else:
            m, step = np.zeros(3), 0.0
        # every band offset in one interpolation call (zeros padding, as grid_sample), then the weighted sum
        P = pts[None] + (np.asarray(u, float) * step)[:, None, None] * m                 # (K, L, 3)
        vox = np.einsum("ij,klj->ikl", Ai[:2, :3], P) + Ai[:2, 3][:, None, None]        # (2, K, L)
        vals = map_coordinates(img, vox.reshape(2, -1), order=1, mode="constant", cval=0.0).reshape(len(u), len(pts))
        prof = np.zeros(len(pts))
        for vk, wk in zip(vals, w):
            prof += wk * vk
        inside = np.isfinite(self._sample(e, snap, pts))
        lo, hi = e["slice"]["lo"], e["slice"]["hi"]
        return np.where(inside, (prof - lo) / (hi - lo) * 255.0, np.nan)

    def _profile_values(self, e: dict, it: dict, snap: dict, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
        if not (self.show_psf and self.config is not None):
            return self._sample(e, snap, pts), self._sample(it["other"], snap, pts), ""
        scale, sigma = self._loss_state(snap["step"])
        band = not it.get("sa_sa", False)                 # SA-SA lines are sampled on the line, no band
        ya = self._loss_profile(e, it["other"], snap, pts, sigma, scale, band)
        yb = self._loss_profile(it["other"], e, snap, pts, sigma, scale, band)
        return ya, yb, (f"  psf\u00d7{scale:.1f}" if band else "") + f"  blur {sigma:.1f}"

    def _on_blur_toggle(self, value: bool):
        self.show_blur = bool(value)
        self._refresh_colors()

    def _on_psf_toggle(self, value: bool):
        self.show_psf = bool(value)
        self._update_profiles()
        self.pl.render()

    def _profile_items(self, snap: dict) -> list[dict]:
        """Every intersection of the picked slice, in chart order: (other slice, start, unit dir, t_img, t_used,
        in_loss). SA slice: its SA-LA lines (2Ch, 3Ch, 4Ch, Ao) and its SA-SA crossing. LA view (Ao included): first
        its crossings with the other LA views, then with every SA slice that crosses it inside both images (by slice
        index) -- the loss line where the pair has one (lines.csv), else the planes' crossing at the shown poses
        (in_loss False: no loss part; e.g. Ao x 4Ch, too far from the heart masks or too shallow for the loss). The
        chart rectangles show PROFILE_VISIBLE of them at a time (scrolled with the mouse wheel)."""
        e = self._entry_by_name().get(self.profile_key)
        if e is None:
            return []
        # cached: the same per frame (only the profiles' grey values change), recomputed per step / offsets
        ckey = (id(snap), self.profile_key, self._offsets_sig())
        cached = self._items_cache.get(ckey)
        if cached is not None:
            return cached
        if len(self._items_cache) > 4 * max(1, len(self.snapshots)):
            self._items_cache.clear()
        out, with_sa = [], []
        li = self._line_items
        idx = li["by_key"].get(e["key"]) if li is not None else None
        if idx is not None and len(idx):
            x0, d, t_used = self._line_geometry(snap, idx)                  # (only this slice's lines)
            for j, i in enumerate(idx):
                ka, kb = li["keys_a"][i], li["keys_b"][i]
                other = kb if ka == e["key"] else ka
                oe = self._entry_by_key.get(other)
                if oe is None:
                    continue
                item = dict(other=oe, x0=x0[j], d=d[j], t_img=li["t_img"][i], t_used=t_used[j],
                            in_loss=bool(np.isfinite(t_used[j]).all()))     # at this step's mask dilation
                (with_sa if e["view"] != "SA" and self._key_view(other) == "SA" else out).append(item)
        parallel = {kb if ka == e["key"] else ka for ka, kb in self.overlays.get("sa_pairs") or [] if e["key"] in (ka, kb)}
        if e["view"] != "SA":
            have = {r["other"]["key"] for r in out + with_sa} | parallel      # (parallel pairs: charted below)
            Ae = self._pose(e["key"], snap)
            for oe in self.entries:                       # pairs without a loss line: the geometric crossing
                if oe["key"] == e["key"] or oe["key"] in have or Ae is None:
                    continue
                Ao = self._pose(oe["key"], snap)
                cs = None if Ao is None else crossing_segment(Ae, e["slice"]["grey"].shape, Ao,
                                                             oe["slice"]["grey"].shape)
                if cs is None and oe["view"] == "SA":
                    continue                              # SA slices not crossing this view inside both images
                item = dict(other=oe, x0=None, in_loss=False)
                if cs is not None:
                    item.update(x0=cs[0], d=cs[1], t_img=np.array([cs[2], cs[3]]), t_used=np.array([np.nan, np.nan]))
                (with_sa if oe["view"] == "SA" else out).append(item)
            # (with several modalities: the profiled slice's own modality first)
            out.sort(key=lambda r: (r["other"]["modality"] != e["modality"], VIEW_RANK.get(r["other"]["view"], 4),
                                    r["other"]["image"], r["other"]["key"][1]))
            out += sorted(with_sa, key=lambda r: (r["other"]["modality"] != e["modality"], r["other"]["image"],
                                                  r["other"]["key"][1]))
        else:
            out.sort(key=lambda r: (r["other"]["modality"] != e["modality"], VIEW_RANK.get(r["other"]["view"], 4),
                                    r["other"]["image"]))
        # parallel slices: an SA slice's next neighbour in its stack (else the previous one), and the parallel
        # slices of other images it is paired with (sa_lines.csv; e.g. the same view in another modality)
        partners = []
        if e["view"] == "SA":
            sl = e["key"][1]
            names = {x["key"][1]: x for x in self.entries if x["image"] == e["image"]}
            nb = sl + 1 if sl + 1 in names else sl - 1
            if nb in names:
                partners.append(names[nb])
        for key in sorted(parallel):
            o = self._entry_by_key.get(key)
            if o is not None and o["image"] != e["image"] and all(o is not x for x in partners):
                partners.append(o)
        segs = self._sa_segments(snap) if partners else {}
        for o in partners:
            item = dict(other=o, x0=None, sa_sa=True)
            pair = {e["key"], o["key"]}
            hit = [j for j, pr in enumerate(segs.get("img_pairs", [])) if set(pr) == pair]
            if hit:
                a, b = segs["img"][hit[0]]
                length = float(np.linalg.norm(b - a))
                if length > 0:
                    item.update(x0=a, d=(b - a) / length, t_img=np.array([0.0, length]),
                                t_used=np.array([np.nan, np.nan]))
                    uh = [j for j, pr in enumerate(segs.get("used_pairs", [])) if set(pr) == pair]
                    if uh:
                        item["t_used"] = np.sort((segs["used"][uh[0]] - a) @ item["d"])
            out.append(item)
        self._items_cache[ckey] = out
        return out

    def _offsets_sig(self) -> tuple:
        """The manual pose offsets as a hashable value (cache keys)."""
        return tuple(sorted((k, tuple(np.round(v, 6))) for k, v in self.offsets.items() if np.any(v)))

    def _profile_metric(self, e: dict, it: dict, snap: dict, t: np.ndarray, ya: np.ndarray, yb: np.ndarray,
                        used: np.ndarray):
        """The selected metric of one chart and its per-point curves: (label, value or None, curve t, loss curve or
        None, full curve or None), resampled on the --line-step-mm grid, current frame only. The value and the loss
        curve are computed like the loss / logged metrics on the line's loss points (NaN elsewhere); the full curve
        is the same metric over the whole intersection inside both images (the metric without the mask), equal to
        the loss curve on the loss points. Curves (NaN where undefined):
          NCC   z_a(t) z_b(t) of the profiles z-scored with the loss points' mean / std -- their mean over the loss
                points is the NCC (no loss points: z-scored over the whole intersection)
          LNCC  the local NCC of the window centred at t -- their weighted mean is the LNCC
          Dice  1 where both slices show the same counted label, 0 where only one does
          HD    distance (mm) to the nearest point of the same label in the other profile -- the line's
                Hausdorff distance is the maximum of the curve
          NMI   none"""
        spec = OUTLINE_METRICS[self.metric]
        cfg = getattr(self, "config", None) or {}
        step = float(cfg.get("line_step_mm", PROFILE_STEP_MM))
        tg = np.arange(np.ceil(t[0] / step) * step, t[-1] + 1e-6, step)
        none = (spec["label"], None, tg, None, None)
        if len(tg) < 3:
            return none
        interp = lambda y: np.interp(tg, t, np.where(np.isfinite(y), y, np.nan))   # noqa: E731
        a, b = interp(ya), interp(yb)
        v = np.isfinite(a) & np.isfinite(b)                 # inside both images
        w = (np.interp(tg, t, used.astype(float)) > 0.5) & v  # loss points
        if v.sum() < 3:
            return none
        has_loss = w.sum() >= 3
        a, b = np.where(v, a, 0.0), np.where(v, b, 0.0)
        wf, vf = w.astype(float), v.astype(float)

        def merged(loss, full):                             # the full curve takes the loss values on the loss points
            if full is None:
                return loss
            return full if loss is None else np.where(w & np.isfinite(loss), loss, full)

        if self.metric == "ncc":
            sf = wf if has_loss else vf                       # z-scored with the loss points' statistics
            ss = sf.sum()
            za = (a - (sf * a).sum() / ss); za = za / max(np.sqrt((sf * za * za).sum() / ss), 1e-9)
            zb = (b - (sf * b).sum() / ss); zb = zb / max(np.sqrt((sf * zb * zb).sum() / ss), 1e-9)
            full = np.where(v, za * zb, np.nan)
            if not has_loss:
                return spec["label"], None, tg, None, full
            return spec["label"], np_ncc(a, b, wf), tg, np.where(w, full, np.nan), full
        if self.metric == "lncc":
            k = max(3, int(round(float(cfg.get("lncc_window_mm", 20.0)) / step)))
            min_std = float(cfg.get("lncc_min_std", 0.25))
            centre = np.arange(len(tg) - k + 1) + (k - 1) // 2

            def local(weights):
                value, loc, wt = np_lncc(a, b, weights, k, min_std, return_windows=True)
                curve = np.full(len(tg), np.nan)
                curve[centre[:len(loc)]] = np.where(wt > 0, loc, np.nan)
                return value, curve

            _, full = local(vf)
            if not has_loss:
                return spec["label"], None, tg, None, full
            value, curve = local(wf)
            return spec["label"], value, tg, curve, merged(curve, full)
        if self.metric == "nmi":
            if not has_loss:
                return none
            return (spec["label"], np_nmi(a, b, wf, int(cfg.get("nmi_bins", 24)), float(cfg.get("nmi_sigma", 1.0))),
                    tg, None, None)
        if self.metric in ("dice", "hd"):
            if it.get("sa_sa"):
                return none                                 # not evaluated on SA-SA lines
            pts = it["x0"] + tg[:, None] * it["d"]
            la_la = e["view"] != "SA" and it["other"]["view"] != "SA"
            labs = cfg.get("seg_labels", "1,2,3,4,5") if la_la else cfg.get("sax_seg_labels", "1,2,3")
            labs = [int(x) for x in str(labs).split(",") if x.strip()]
            if self.metric == "dice":
                line = lambda valid: self._line_dice(e, it["other"], snap, pts, valid, labs)   # noqa: E731
            else:
                line = lambda valid: self._line_hd(e, it["other"], snap, pts, valid, labs, tg,   # noqa: E731
                                                   float(cfg.get("hd_miss_mm", 100.0)))
            _, full = line(v)
            if not has_loss:
                return spec["label"], None, tg, None, full
            value, curve = line(w)
            return spec["label"], value, tg, curve, merged(curve, full)
        return none

    def _seg_on(self, e: dict, snap: dict, pts: np.ndarray) -> np.ndarray | None:
        seg = e["slice"]["seg"]
        if seg is None:
            return None
        A = self._affine_for(e, snap)
        vox = (np.linalg.inv(A) @ np.column_stack([pts, np.ones(len(pts))]).T)[:2]
        i, j = np.rint(vox[0]).astype(int), np.rint(vox[1]).astype(int)
        ok = (i >= 0) & (i < seg.shape[0]) & (j >= 0) & (j < seg.shape[1])
        out = np.full(len(pts), -1, dtype=int)
        out[ok] = seg[i[ok], j[ok], self.frame]
        return out

    def _counted_labels(self, e: dict, o: dict, snap: dict, pts: np.ndarray, valid: np.ndarray, labels: list[int]):
        """(seg profile a, seg profile b, counted labels) as motion_gradient.LineSampler.counted_labels, one frame:
        a label counts if both slices contain it somewhere and at least one profile hits it on a valid point."""
        sa, sb = self._seg_on(e, snap, pts), self._seg_on(o, snap, pts)
        if sa is None or sb is None:
            return None, None, []
        fa, fb = e["slice"]["seg"][:, :, self.frame], o["slice"]["seg"][:, :, self.frame]
        counted = [lab for lab in labels if (fa == lab).any() and (fb == lab).any()
                   and (((sa == lab) & valid).any() or ((sb == lab) & valid).any())]
        return sa, sb, counted

    def _line_dice(self, e: dict, o: dict, snap: dict, pts: np.ndarray, valid: np.ndarray, labels: list[int]):
        """Mean Dice over the counted labels along the line (LineSampler.dice_lines, one frame) and the per-point
        agreement curve: 1 where both profiles show the same counted label, 0 where only one shows it."""
        sa, sb, counted = self._counted_labels(e, o, snap, pts, valid, labels)
        if not counted:
            return None, None
        dice = []
        for lab in counted:
            ma, mb = (sa == lab) & valid, (sb == lab) & valid
            dice.append(2.0 * (ma & mb).sum() / max(1, ma.sum() + mb.sum()))
        hit = valid & (np.isin(sa, counted) | np.isin(sb, counted))
        return float(np.mean(dice)), np.where(hit, (sa == sb).astype(float), np.nan)

    def _line_hd(self, e: dict, o: dict, snap: dict, pts: np.ndarray, valid: np.ndarray, labels: list[int],
                 t: np.ndarray, miss_mm: float):
        """1D Hausdorff distance along the line (LineSampler.hd_lines, one frame) and its per-point curve: for every
        labelled point the distance to the nearest point of the same label in the other profile (miss_mm if the
        other profile lacks it); the line value is the curve's maximum."""
        sa, sb, counted = self._counted_labels(e, o, snap, pts, valid, labels)
        if not counted:
            return None, None
        curve = np.full(len(t), np.nan)
        for lab in counted:
            ma, mb = (sa == lab) & valid, (sb == lab) & valid
            for mine, theirs in ((ma, mb), (mb, ma)):
                if not mine.any():
                    continue
                if theirs.any():
                    d = np.abs(t[mine][:, None] - t[theirs][None, :]).min(1)
                else:
                    d = np.full(int(mine.sum()), miss_mm)
                idx = np.flatnonzero(mine)
                curve[idx] = np.fmax(curve[idx], d)
        return float(np.nanmax(curve)) if np.isfinite(curve).any() else None, curve

    def _chart_data(self, snap: dict, items: list[dict] | None = None) -> list[dict]:
        """Per intersection at `snap` (current frame; `items`, default all of _profile_items): the item and,
        where the slices cross, its profiles (t, ya, yb, used, mode_txt) and the selected metric (mname, mval, ct,
        curve)."""
        e = self._entry_by_name().get(self.profile_key)
        out = []
        for it in (self._profile_items(snap) if items is None else items):
            c = dict(it=it)
            if it["x0"] is not None:
                t = np.arange(it["t_img"][0], it["t_img"][1] + 1e-6, PROFILE_STEP_MM)
                pts = it["x0"] + t[:, None] * it["d"]
                ya, yb, mode_txt = self._profile_values(e, it, snap, pts)
                used = (t >= it["t_used"][0]) & (t <= it["t_used"][1]) if np.isfinite(it["t_used"]).all() \
                    else np.zeros(len(t), bool)
                mname, mval, ct, curve, full = self._profile_metric(e, it, snap, t, ya, yb, used)
                c.update(t=t, pts=pts, ya=ya, yb=yb, used=used, mode_txt=mode_txt, mname=mname, mval=mval,
                         ct=ct, curve=curve, full=full)
            out.append(c)
        return out

    @staticmethod
    def _range_ident(it: dict) -> tuple:
        return it["other"]["key"], bool(it.get("sa_sa"))

    def _range_key(self) -> tuple:
        return (self.run_index, self.subject_index, self.profile_key, self.metric,
                self.show_psf and self.config is not None, self._offsets_sig())

    def _line_in_mask(self, e: dict, other: dict, snap: dict, pts: np.ndarray) -> np.ndarray:
        """Which points of an intersection line are inside either slice's dilated mask (or, SA x Ao, the SA-SA
        cylinder) at `snap`: the yellow bar of its charts."""
        masks = [m for m in (self._in_mask(e, snap, pts), self._in_mask(other, snap, pts),
                             self._in_cylinder(e, other, snap, pts)) if m is not None]
        return np.logical_or.reduce(masks) if masks else np.zeros(len(pts), bool)

    def _chart_entry(self, si: int, it: dict, frame: int | None = None, prefetch: bool = False) -> dict | None:
        """The chart data (as _chart_data) of intersection `it` (of snapshot si's _profile_items) at snapshot si and
        `frame` (default: the shown one), through the chart cache: per chart state (_range_key: slice, metric,
        profile mode, offsets), snapshot, frame and intersection, its curves as float32 plus the in-mask bar (the
        line and its sample points are rebuilt from `it`; least recently used dropped first). prefetch: for the cache
        only, None if it is full (CHART_CACHE_MB; prefetching does not push anything out)."""
        frame = self.frame if frame is None else frame
        rk = self._range_key()
        if rk != self._chart_state:                    # another slice, metric, profile mode or offsets: start over
            self._chart_cache.clear()
            self._chart_state = rk
        key = (rk, si, frame, self._range_ident(it))
        hit = self._chart_cache.get(key)
        if hit is None:
            n_pts = int((it["t_img"][1] - it["t_img"][0]) / PROFILE_STEP_MM) + 1 if it["x0"] is not None else 0
            if prefetch and not self._chart_cache.make_room(key, 21 * n_pts + 512, self.frame, self.n_frames):
                return None
            snap = self.snapshots[si]
            shown, self.frame = self.frame, frame             # (the chart data are of self.frame)
            try:
                c = self._chart_data(snap, [it])[0]
            finally:
                self.frame = shown
            if "ya" in c:
                f32 = lambda a: None if a is None else np.asarray(a, np.float32)  # noqa: E731
                e = self._entry_by_name().get(self.profile_key)
                hit = dict(ya=f32(c["ya"]), yb=f32(c["yb"]), curve=f32(c["curve"]), full=f32(c["full"]),
                           ct=f32(c["ct"]), mname=c["mname"], mval=c["mval"], mode_txt=c["mode_txt"],
                           in_mask=self._line_in_mask(e, it["other"], snap, c["pts"]))
            else:
                hit = {}                                       # no crossing
            nbytes = sum(a.nbytes for a in hit.values() if isinstance(a, np.ndarray)) + 512
            self._chart_cache.put(key, hit, nbytes, prefetch, self.frame, self.n_frames)
        c = dict(it=it)
        if "ya" in hit:
            t = np.arange(it["t_img"][0], it["t_img"][1] + 1e-6, PROFILE_STEP_MM)
            used = (t >= it["t_used"][0]) & (t <= it["t_used"][1]) if np.isfinite(it["t_used"]).all() \
                else np.zeros(len(t), bool)
            c.update(t=t, pts=it["x0"] + t[:, None] * it["d"], used=used,
                     **{k: hit[k] for k in ("ya", "yb", "curve", "full", "ct", "mname", "mval", "mode_txt", "in_mask")})
        return c

    def _metric_ranges(self, data: list[dict]) -> list:
        """Per shown chart: (min, max) of the full metric curve over every history snapshot, so the metric
        charts keep one y scale while scrubbing the history; None where an intersection never has a curve.
        One pass over the history costs ~0.2 s per intersection, so it runs in the background, in timer ticks
        (_range_step; the shown charts first): until an intersection's pass is done its range is that of the curves
        seen so far, and its chart is redrawn once the pass completes. Ranges are kept per slice, metric, profile
        mode and manual offsets, computed at the frame shown at the time, and widened when another frame's curve
        (`data`, the shown ones) leaves them. Without an event loop (off screen) the pass runs right away."""
        key = self._range_key()
        if self._range_cache is None or self._range_cache[0] != key:
            self._range_cache = (key, {}, set())
            self._rjob = None
        _, ranges, complete = self._range_cache
        missing = {self._range_ident(c["it"]) for c in data} - complete
        if missing and self._offscreen():
            for si in range(len(self.snapshots)):
                self._range_accumulate(missing, si)
            complete |= missing
        elif missing:
            self._start_range_job()
        out = []
        for c in data:
            r = ranges.setdefault(self._range_ident(c["it"]), [np.inf, -np.inf])
            cv = c.get("full")
            if cv is not None and np.isfinite(cv).any():
                r[0], r[1] = min(r[0], float(np.nanmin(cv))), max(r[1], float(np.nanmax(cv)))
            out.append((float(r[0]), float(r[1])) if np.isfinite(r[0]) else None)
        return out

    def _range_accumulate(self, idents: set, si: int):
        """Widen the ranges of the given intersections by their full metric curves at snapshot si."""
        ranges = self._range_cache[1]
        its = [it for it in self._profile_items(self.snapshots[si]) if self._range_ident(it) in idents]
        for c in (self._chart_entry(si, it) for it in its):            # (cached: the history at the shown frame)
            cv = c.get("full")
            if cv is not None and np.isfinite(cv).any():
                r = ranges.setdefault(self._range_ident(c["it"]), [np.inf, -np.inf])
                r[0], r[1] = min(r[0], float(np.nanmin(cv))), max(r[1], float(np.nanmax(cv)))

    def _start_range_job(self):
        """Queue the history pass of every intersection of the profiled slice whose range is not complete."""
        key = self._range_key()
        if self._rjob is not None and self._rjob["key"] == key:
            self._ensure_timer()
            return
        complete = self._range_cache[2]
        items = (self._pcache or {}).get("items") or []
        todo = {}
        for it in items:
            idn = self._range_ident(it)
            if idn not in complete and idn not in todo:
                todo[idn] = list(range(len(self.snapshots)))[::-1]
        self._rjob = {"key": key, "todo": todo}
        self._ensure_timer()

    def _range_step(self, shown_only: bool) -> tuple[bool, bool]:
        """One background unit of the range pass: one intersection at one snapshot (an intersection of a shown chart
        first, or with shown_only False any other). When an intersection completes, its shown charts are redrawn
        with the full range. (did work, charts changed)"""
        job = self._rjob
        if job is None or self._range_cache is None or job["key"] != self._range_key() or not job["todo"]:
            return False, False
        shown = {self._range_ident(c["it"]) for i, c in (self._pcache or {}).get("data", {}).items()
                 if any(slot["bound"] == i for slot in self._profile_charts)}
        idn = next((k for k in job["todo"] if (k in shown) == shown_only), None)
        if idn is None:
            return False, False
        self._range_accumulate({idn}, job["todo"][idn].pop())
        if job["todo"][idn]:
            return True, False
        del job["todo"][idn]
        _, ranges, complete = self._range_cache
        complete.add(idn)
        changed = False
        r = ranges.get(idn)
        new = (float(r[0]), float(r[1])) if r is not None and np.isfinite(r[0]) else None
        for i, c in (self._pcache or {}).get("data", {}).items():
            if self._range_ident(c["it"]) == idn and self.show_metric_profiles and c.get("range") != new:
                c["range"] = new
                for slot in self._profile_charts:
                    if slot["bound"] == i:
                        slot["bound"] = None             # refilled with the full range (_render_profiles)
                        changed = True
        return True, changed

    # --- background work and the scroll animation (one repeating timer) -------------------
    def _offscreen(self) -> bool:
        return bool(pv.OFF_SCREEN or self.pl.off_screen)

    def _ensure_timer(self):
        """Run the timer (_tick) while there is background work or a scroll animation; before the window's event
        loop runs, the first render starts it."""
        if self._offscreen() or self._timer is not None:
            return
        iren = self.pl.iren.interactor if self.pl.iren is not None else None
        if iren is None or not iren.GetInitialized():
            self._timer_wanted = True
            return
        self._timer = iren.CreateRepeatingTimer(SCROLL_TIMER_MS) or None      # (0: not created; tried again later)
        self._timer_wanted = self._timer is None

    def _idle_work(self, t_end: float) -> tuple[bool, bool]:
        """Background work until t_end, in priority order: the ranges of the shown metric charts (their chart data
        at every step of the shown frame), the prefetch stream (_bg_stream: per frame, from the shown one on, the
        slice colours and the shown charts' data at every step), the ranges of the other charts. (more left, charts
        changed)"""
        changed = False
        while time.perf_counter() < t_end:
            for step in (lambda: self._range_step(True), self._bg_step, lambda: self._range_step(False)):
                did, ch = step()
                if did:
                    changed |= ch
                    break
            else:
                return False, changed
        return True, changed

    def _tick(self):
        """Timer tick: one frame of the scroll animation (SCROLL_EASE of the way to the target, ease-out), then
        background work for the rest of the tick's budget; the timer stops when neither is left."""
        if self._timer is None:
            return
        scrolling = self._profile_scroll != self._scroll_target
        if scrolling:
            d = self._scroll_target - self._profile_scroll
            if abs(d) < 0.01:
                self._profile_scroll = self._scroll_target
            else:
                self._profile_scroll += d * SCROLL_EASE if abs(d) * SCROLL_EASE > 0.01 else np.sign(d) * 0.01
            self._hover = None
        more, changed = self._idle_work(time.perf_counter() + (IDLE_BUDGET_SCROLL_S if scrolling else IDLE_BUDGET_S))
        if scrolling or changed:
            self._render_profiles()
            self.pl.render()
        if not more and self._profile_scroll == self._scroll_target:
            self.pl.iren.interactor.DestroyTimer(self._timer)
            self._timer = None

    def _update_profiles(self):
        """Recompute the intersection charts for the current state (slice, step, frame, toggles, offsets...): drop
        the cached intersections and chart data, then draw (_render_profiles)."""
        self._pcache = None
        for slot in getattr(self, "_profile_charts", []):
            slot["bound"] = None
        self._render_profiles()
        self._ensure_timer()                             # (prefetch for the new state)

    def _render_profiles(self):
        """Draw the charts at the current scroll position: the intersections from base = floor(scroll) on fill the
        PROFILE_VISIBLE + 1 chart slots (a slot is only refilled when its intersection changes; chart data are
        cached until _update_profiles), which are then placed at their fractional positions (_position_charts)."""
        charts = getattr(self, "_profile_charts", [])
        if not charts or not self.entries:
            return
        snap = self.snapshots[self.snap_index]
        if self._pcache is None:
            items = self._profile_items(snap) if self.show_profiles or self.show_metric_profiles else []
            self._pcache = {"items": items, "data": {}}
        items, cache = self._pcache["items"], self._pcache["data"]
        n = self._n_profile_items = len(items)
        smax = float(max(0, n - PROFILE_VISIBLE))
        self._profile_scroll = float(np.clip(self._profile_scroll, 0.0, smax))
        self._scroll_target = float(np.clip(self._scroll_target, 0.0, smax))
        self._layout_charts(n)
        base = int(np.floor(self._profile_scroll + 1e-9))
        for k, (slot, mslot) in enumerate(zip(charts, self._metric_charts)):
            i = base + k
            if i >= n:
                slot["item"] = mslot["item"] = slot["bound"] = None
                slot["chart"].visible = mslot["chart"].visible = False
                continue
            if slot["bound"] == i:
                continue
            if i not in cache:
                c = self._chart_entry(self.snap_index, items[i])
                c["range"] = self._metric_ranges([c])[0] if self.show_metric_profiles else None
                cache[i] = c
            self._fill_slot(k, cache[i], snap)
            slot["bound"] = i
        self._position_charts()
        self._draw_hover()

    def _fill_slot(self, k: int, c: dict, snap: dict):
        """Draw intersection `c` (see _chart_data) into chart slot k of both sets."""
        slot, mslot = self._profile_charts[k], self._metric_charts[k]
        ch, pl_ = slot["chart"], slot["plots"]
        e = self._entry_by_name().get(self.profile_key)
        slot["item"] = mslot["item"] = None
        it = c["it"]
        ch.visible, mslot["chart"].visible = self.show_profiles, self.show_metric_profiles
        names = f"{self._short_name(e, e)} \u00d7 {self._short_name(it['other'], e)}" + \
            ("  (not in loss)" if it.get("in_loss") is False else "")
        if it["x0"] is None:
            ch.title = f"{names}: no crossing"
            for p_ in pl_.values():
                p_.update(np.array([0.0, 1.0]), np.array([np.nan, np.nan]))
            self._fill_metric_chart(mslot, f"{names}: no crossing", None, None, None, None, None, None)
            return
        t, ya, yb, used, mname, mval = c["t"], c["ya"], c["yb"], c["used"], c["mname"], c["mval"]
        in_mask = c.get("in_mask")
        if in_mask is None:
            in_mask = self._line_in_mask(e, it["other"], snap, c["pts"])
        ch.title = (f"{names}   {mname} {mval:.2f}{c['mode_txt']}" if mval is not None
                    else f"{names}   {mname} -{c['mode_txt']}")
        for side, y in (("a", ya), ("b", yb)):
            pl_[side + "_img"].update(t, y)
            pl_[side + "_used"].update(t, np.where(used, y, np.nan))
        pl_["mask"].update(t, np.where(in_mask, float(PROFILE_MASK_Y), np.nan))
        xr = [float(t[0]), float(t[-1]) if len(t) > 1 else float(t[0]) + 1.0]
        ch.x_range = xr
        ch.y_range = [0, 255]
        item = dict(e=e, other=it["other"], x0=it["x0"], d=it["d"], t=t, ya=ya, yb=yb, names=names,
                    title=ch.title, ct=c["ct"], curve=c["full"], mname=mname)
        slot["item"] = mslot["item"] = item
        item["mtitle"], item["mrange"] = self._fill_metric_chart(
            mslot, f"{names}   {mname} per point", c["ct"], c["curve"], c["full"], xr, c.get("range"),
            (t, in_mask))

    def _rect_boxes(self) -> dict:
        """(x0, y0, x1, y1) of the shown chart rectangles (normalised): intensity on top, metric below, splitting
        the right column (PROFILE_Y0 .. PROFILE_TOP); a single shown set takes the whole column."""
        sets = [n for n, on in (("profile", self.show_profiles), ("metric", self.show_metric_profiles)) if on]
        if not sets:
            return {}
        h = (PROFILE_TOP - PROFILE_Y0 - (len(sets) - 1) * RECT_GAP) / len(sets)
        out, top = {}, PROFILE_TOP
        for n in sets:
            out[n] = (RECT_X0, top - h, RECT_X1, top)
            top -= h + RECT_GAP
        return out

    def _layout_charts(self, n_items: int):
        """Frame each shown chart rectangle, set its chart renderer's viewport to the rectangle's inside and, with
        more than PROFILE_VISIBLE intersections, draw its scroll bar (thumb: the shown window within the list, at
        the current fractional scroll position)."""
        boxes = self._rect_boxes()
        V = PROFILE_VISIBLE
        h = self.pl.window_size[1]

        def quad(part, x0, y0, x1, y1):
            poly, actor = part
            poly.points = np.array([[x0, y0, 0.0], [x1, y0, 0.0], [x1, y1, 0.0], [x0, y1, 0.0]])
            poly.Modified()
            actor.SetVisibility(True)

        for name in ("profile", "metric"):
            r = self._rects[name]
            for _, actor in r["parts"].values():
                actor.SetVisibility(False)
            r["box"] = box = boxes.get(name) if n_items else None
            ren = self._chart_rens[name]
            if box is None:
                ren.SetViewport(0.0, 0.0, 0.001, 0.001)
                continue
            x0, y0, x1, y1 = box
            ren.SetViewport(PROFILE_X, y0 + RECT_PAD, PROFILE_X + PROFILE_W, y1 - RECT_PAD)
            pitch = (y1 - y0 - 2 * RECT_PAD + PROFILE_GAP) / V
            self._chart_px_h = (pitch - PROFILE_GAP) * h
            quad(r["parts"]["frame"], x0, y0, x1, y1)
            if n_items > V:
                ty0, ty1 = y0 + RECT_PAD, y1 - RECT_PAD
                quad(r["parts"]["track"], SCROLL_X0, ty0, SCROLL_X0 + SCROLL_W, ty1)
                top = ty1 - self._profile_scroll / n_items * (ty1 - ty0)
                quad(r["parts"]["thumb"], SCROLL_X0, top - V / n_items * (ty1 - ty0), SCROLL_X0 + SCROLL_W, top)

    def _position_charts(self):
        """Place every filled chart slot at its fractional position in its renderer: intersection i sits (i -
        scroll) chart pitches below the top, so the slots slide smoothly and the renderer viewport clips the ones
        partly out of view. The pixel geometry is set directly (pyvista's loc only takes positions inside the
        renderer); size stays in sync with it so pyvista does not re-place the chart on render."""
        V = PROFILE_VISIBLE
        gap = PROFILE_GAP * self.pl.window_size[1]
        base = int(np.floor(self._profile_scroll + 1e-9))
        for name, slots in (("profile", self._profile_charts), ("metric", self._metric_charts)):
            if self._rects[name]["box"] is None:
                continue
            rw, rh = self._chart_rens[name].GetSize()
            if rw < 2 or rh < 2:
                continue
            pitch = (rh + gap) / V
            size_h = min(1.0, (pitch - gap) / rh)
            shown = self.show_profiles if name == "profile" else self.show_metric_profiles
            for k, slot in enumerate(slots):
                ch = slot["chart"]
                if self._profile_charts[k]["bound"] is None:
                    continue
                bottom = rh - (base + k - self._profile_scroll) * pitch - (pitch - gap)
                # a chart wholly outside the rectangle (the spare slot, between scroll steps) is not drawn
                ch.visible = shown and bottom < rh and bottom + size_h * rh > 0
                ch.size = (1.0, size_h)
                ch._loc = (0.0, float(np.clip(bottom / rh, 0.0, 1.0)))
                ch._geometry = (0, int(round(bottom)), int(rw), int(size_h * rh))

    def _on_wheel(self, step: int) -> bool:
        """Mouse wheel: over a chart rectangle, scroll both rectangles by one chart (`step` -1 up, +1 down) and
        return True (the event is consumed: no camera zoom); elsewhere False."""
        x, y = self.pl.iren.interactor.GetEventPosition()
        w, h = self.pl.window_size
        for r in getattr(self, "_rects", {}).values():
            box = r["box"]
            if box is not None and box[0] * w <= x <= box[2] * w and box[1] * h <= y <= box[3] * h:
                self.scroll_profiles(step)
                return True
        return False

    def scroll_profiles(self, step: float):
        """Scroll the intersection charts by `step` wheel notches (SCROLL_STEP charts each), clamped to the list;
        the move is animated (_scroll_tick) -- off screen it jumps."""
        smax = float(max(0, self._n_profile_items - PROFILE_VISIBLE))
        target = float(np.clip(self._scroll_target + step * SCROLL_STEP, 0.0, smax))
        if target == self._scroll_target and target == self._profile_scroll:
            return
        self._scroll_target = target
        if self._offscreen():
            self.set_scroll(target)
            return
        self._ensure_timer()

    def set_scroll(self, pos: float):
        """Show the charts at scroll position `pos` (charts from the top, fractional) without animating."""
        smax = float(max(0, self._n_profile_items - PROFILE_VISIBLE))
        self._profile_scroll = self._scroll_target = float(np.clip(pos, 0.0, smax))
        self._hover = None
        self._render_profiles()
        self.pl.render()

    def _fill_metric_chart(self, mslot: dict, title: str, t, curve, full, xr, rng, mask):
        """Draw the per-point metric curves of one intersection as its intensity profiles: thin over the whole
        intersection (`full`), thick on the loss points (`curve`), and the mask range (`mask`: (t, in_mask)) as a bar
        along the bottom; on the fixed y range `rng` (the full curve's min / max over the history, padded; empty
        chart when there is no curve, e.g. NMI). Returns the chart title and its y range."""
        ch, p = mslot["chart"], mslot["plots"]
        nan2 = np.array([np.nan, np.nan])
        if full is None or t is None or not np.isfinite(full).any():
            for key in ("full", "curve", "zero", "mask"):
                p[key].update(np.array([0.0, 1.0]), nan2)
            if xr is not None:
                title = title.replace(" per point", ": no per-point values")
                ch.x_range = xr
            ch.y_range = [0, 1]
            ch.y_axis.tick_locations = (0.0, 1.0)
            ch.y_axis.tick_labels = ("", "")
            ch.title = title
            return title, (0.0, 1.0)
        dmin, dmax = rng if rng is not None else (float(np.nanmin(full)), float(np.nanmax(full)))
        span = dmax - dmin
        pad = max(0.05 * span, 0.05)
        lo, hi = dmin - pad, dmax + pad
        dec = 2 if span < 2 else 1 if span < 20 else 0          # tick rounding stays inside the padding
        # the tick labels need ~TICK_LABEL_PX between them: as a fraction of the data span on this chart's plot area
        plot_px = max(1.0, getattr(self, "_chart_px_h", 80) - CHART_FRAME_PX)
        sep = max(0.2, TICK_LABEL_PX / plot_px * (hi - lo) / max(span, 1e-12))
        if dmin < 0 < dmax:                                     # 0, plus the extremes not crowding it
            ticks = tuple(v for v in (dmin, 0.0, dmax) if v == 0.0 or abs(v) > sep * span)
        else:
            ticks = (dmin, (dmin + dmax) / 2, dmax) if sep <= 0.5 else (dmin, dmax)
        ticks = tuple(dict.fromkeys(round(v, dec) for v in ticks))
        p["full"].update(t, full)
        p["curve"].update(t, curve if curve is not None else np.full(len(t), np.nan))
        p["zero"].update(np.array(xr, dtype=float), np.zeros(2) if lo < 0 < hi else nan2)
        mt, in_mask = mask if mask is not None else (None, None)
        if mt is not None and in_mask is not None and in_mask.any():   # same height share as in the profile charts
            p["mask"].update(mt, np.where(in_mask, lo + PROFILE_MASK_Y / 255.0 * (hi - lo), np.nan))
        else:
            p["mask"].update(np.array([0.0, 1.0]), nan2)
        ch.x_range = xr
        ch.y_range = [lo, hi]
        ch.y_axis.tick_locations = ticks
        ch.y_axis.tick_labels = tuple(f"{v:.{dec}f}" for v in ticks)
        ch.title = title
        return title, (lo, hi)

    # --- hover on the profile charts -> tick marks on the line in 3D ------------------
    def _on_mouse_move(self, *_):
        x, y = self.pl.iren.interactor.GetEventPosition()
        w, h = self.pl.window_size
        hover = None
        for k, slots in enumerate(zip(getattr(self, "_profile_charts", []), getattr(self, "_metric_charts", []))):
            for name, slot in zip(("profile", "metric"), slots):
                ch, it = slot["chart"], slot["item"]
                if it is None or not ch.visible:
                    continue
                # the chart in window pixels: its renderer's viewport offset plus its geometry, clipped to the viewport
                vp = self._chart_rens[name].GetViewport()
                ox, oy = vp[0] * w, vp[1] * h
                cx, cy, cw, chh = ch._geometry
                if not (ox + cx <= x <= ox + cx + cw and max(oy + cy, vp[1] * h) <= y <= min(oy + cy + chh, vp[3] * h)):
                    continue
                p1, p2 = ch.x_axis.GetPoint1(), ch.x_axis.GetPoint2()
                if p2[0] - p1[0] >= 1 and ox + p1[0] <= x <= ox + p2[0]:
                    lo, hi = ch.x_range
                    hover = (k, lo + (x - ox - p1[0]) / (p2[0] - p1[0]) * (hi - lo))
                break
            if hover is not None:
                break
        if hover is None and self._hover is None:
            return                                            # nothing changed: no render
        self._hover = hover
        self._draw_hover()
        self.pl.render()

    def _draw_hover(self):
        charts = getattr(self, "_profile_charts", [])
        mcharts = getattr(self, "_metric_charts", [])
        for slots, key in ((charts, "title"), (mcharts, "mtitle")):
            for slot in slots:
                slot["plots"]["cursor"].visible = False
                if slot["item"] is not None:
                    slot["chart"].title = slot["item"][key]
        if self._hover is None or not self.entries:
            self._hover_actor.SetVisibility(False)
            return
        k, t = self._hover
        it = charts[k]["item"] if k < len(charts) else None
        if it is None:
            self._hover_actor.SetVisibility(False)
            return
        snap = self.snapshots[self.snap_index]
        P = it["x0"] + t * it["d"]
        pts = []
        for e in (it["e"], it["other"]):
            A = self._affine_for(e, snap)
            n = np.cross(A[:3, 0], A[:3, 1]); n /= np.linalg.norm(n)
            across = np.cross(n, it["d"]); across /= max(np.linalg.norm(across), 1e-9)
            tick = np.stack([P - 0.5 * HOVER_TICK_MM * across, P + 0.5 * HOVER_TICK_MM * across])[None]
            pts.append(burn_points(tick, A))
        self._hover_mesh.points = np.concatenate(pts)
        self._hover_actor.SetVisibility(True)
        cur = charts[k]["plots"]["cursor"]
        cur.update(np.array([t, t]), np.array([0.0, 255.0]))
        cur.visible = True
        i = int(np.clip(np.rint((t - it["t"][0]) / PROFILE_STEP_MM), 0, len(it["t"]) - 1))
        fmt = lambda v: "-" if not np.isfinite(v) else f"{v:.0f}"     # noqa: E731
        charts[k]["chart"].title = f"{it['names']}   t {t:+.0f} mm: {fmt(it['ya'][i])} / {fmt(it['yb'][i])}"
        # metric chart: cursor and the curve value at t
        mcur = mcharts[k]["plots"]["cursor"]
        mcur.update(np.array([t, t]), np.array(it["mrange"], dtype=float))
        mcur.visible = True
        c = np.nan
        if it.get("curve") is not None and len(it["ct"]):
            c = it["curve"][int(np.argmin(np.abs(it["ct"] - t)))]
        mcharts[k]["chart"].title = (f"{it['names']}   t {t:+.0f} mm: {it['mname']} "
                                     + (f"{c:.2f}" if np.isfinite(c) else "-"))

    def _on_profiles_toggle(self, value: bool):
        self.show_profiles = bool(value)
        self._update_profiles()
        self.pl.render()

    def _on_metric_profiles_toggle(self, value: bool):
        self.show_metric_profiles = bool(value)
        self._update_profiles()
        self.pl.render()

    def _affine_for(self, e: dict, snap: dict) -> np.ndarray:
        return self._pose(e["key"], snap)

    def _pose(self, key: tuple, snap: dict) -> np.ndarray | None:
        """Pose of a slice at `snap` (its NIfTI affine if the history lacks it), with the manual offset."""
        A = snap["affines"].get(key)
        e = self._entry_by_key.get(key)
        if A is None:
            if e is None:
                return None
            A = e["slice"]["affine"]
        off = self.offsets.get(key)
        if off is not None and e is not None and np.any(off):
            nx, ny = e["slice"]["grey"].shape[:2]
            A = apply_offset(A, off, np.array([(nx - 1) / 2.0, (ny - 1) / 2.0, 0.0]))
        return A

    def _apply_snapshot(self):
        snap = self.snapshots[self.snap_index]
        if abs(self._display_sigma() - self._colors_sigma) > 1e-6:
            self._recolor()
        for e in self.entries:                           # the slice and its outline: moved, not rebuilt
            M = self._affine_for(e, snap) @ e["mesh_inv"]
            e["actor"].user_matrix = M
            e["outline"].user_matrix = M
        self._apply_overlays()
        self._apply_metric_values()
        self._update_knob_labels()
        self._update_header()
        self._update_blur_text()
        self._update_profiles()

    def _update_blur_text(self):
        """The in-plane blur sigma (mm) the optimizer used at the shown step (--blur-sigma annealed over --blur-frac
        of the run, see _loss_state), next to the 'Image blurring' toggle; 0 mm once it no longer blurs."""
        txt = getattr(self, "_blur_text", None)
        if txt is None or not self.snapshots:
            return
        _, sigma = self._loss_state(self.snapshots[self.snap_index]["step"])
        txt.SetInput(f"{sigma:.1f} mm" if sigma > 0 else "0 mm")       # (the panel font has no sigma glyph)

    # --- outline colour by metric --------------------------------------------
    def _metric_value(self, e: dict, snap: dict) -> float:
        """Per-plane value of the active metric at `snap` (nan when unavailable)."""
        col = OUTLINE_METRICS[self.metric]["col"]
        return float(snap["metrics"].get(e["key"], {}).get(col, np.nan))

    def _apply_metric_values(self):
        """Write the active metric of the current snapshot into every outline."""
        snap = self.snapshots[self.snap_index]
        fmt = OUTLINE_METRICS[self.metric]["fmt"]
        for e in self.entries:
            val = self._metric_value(e, snap)
            e["outline_mesh"].point_data["metric"][:] = val
            txt = self._row_value.get(e["name"])
            if txt is not None:
                txt.SetInput(fmt.format(val) if np.isfinite(val) else "-" if e["in_history"] else "unaligned")
        header = getattr(self, "_value_header", None)
        if header is not None:
            header.SetInput(OUTLINE_METRICS[self.metric]["label"] if self.has_metrics else "")
        self._update_toggle_colors()

    def _configure_metric(self):
        """Point the shared lookup table, scalar bar, range sliders and key text at the
        active metric and its current colour range."""
        spec = OUTLINE_METRICS[self.metric]
        lo, hi = self.color_range[self.metric]
        self._lut.cmap = OUTLINE_CMAP if spec["higher_better"] else OUTLINE_CMAP + "_r"
        self._lut.scalar_range = (lo, hi)      # values outside are clamped to the end colours
        self._lut.nan_color = OUTLINE_NAN_COLOR
        for e in self.entries:
            e["outline"].mapper.scalar_range = (lo, hi)
        self._sync_range_sliders()
        self._update_toggle_colors()
        if not self.has_metrics:
            self.outline_text.SetInput(
                "subject not optimized (no crossing views):\n"
                "  no metric colours; the slices show the acquired pose" if self.not_optimized else
                "no history for this subject (aligned without --logging):\n"
                "  no metric colours; the slices show the saved (final) pose")
            return
        self.outline_text.SetInput("")

    # --- colour-range sliders ----------------------------------------------
    def _add_range_sliders(self):
        """Colour-range sliders hugging the colour bar: max above the strip, min below.
        They carry no built-in label; the values are small texts on the right."""
        spec = OUTLINE_METRICS[self.metric]
        b_lo, b_hi = spec["bounds"]
        lo, hi = self.color_range[self.metric]
        w, h = self.pl.window_size
        common = dict(style="modern", color="#bbbbbb", slider_width=0.008, tube_width=0.002,
                      interaction_event="always")
        for which, value, y in (("hi", hi, SLIDER_HI_Y), ("lo", lo, SLIDER_LO_Y)):
            widget = self.pl.add_slider_widget(
                lambda v, k=which: self._on_range_slider(k, v), rng=[b_lo, b_hi], value=value,
                title="", pointa=(BAR_X0, y), pointb=(BAR_X1, y), **common)
            rep = widget.GetRepresentation()
            rep.ShowSliderLabelOff()
            rep.SetTitleHeight(0.0)
            self._range_sliders[which] = widget
            self._range_text[which] = self._text(
                "", position=(RANGE_TEXT_X * w, (y - 0.006) * h), font_size=6, color="#bbbbbb")
        self._sync_range_sliders()

    def _sync_range_sliders(self):
        """Reset both sliders to the active metric's bounds, format and current values."""
        if not self._range_sliders:
            return
        spec = OUTLINE_METRICS[self.metric]
        b_lo, b_hi = spec["bounds"]
        lo, hi = self.color_range[self.metric]
        for which, value in (("lo", lo), ("hi", hi)):
            if which not in self._range_sliders:   # pyvista fires the callback once on creation
                continue
            rep = self._range_sliders[which].GetRepresentation()
            rep.SetMinimumValue(b_lo)
            rep.SetMaximumValue(b_hi)
            rep.SetValue(value)
            txt = self._range_text.get(which)
            if txt is not None:
                txt.SetInput(f"{'max' if which == 'hi' else 'min'}  {spec['fmt'].format(value)}")

    def _on_range_slider(self, which: str, value: float):
        lo, hi = self.color_range[self.metric]
        if which == "lo":
            lo = min(float(value), hi - RANGE_MIN_GAP)
        else:
            hi = max(float(value), lo + RANGE_MIN_GAP)
        self.set_color_range(lo, hi)

    def set_color_range(self, lo: float, hi: float):
        """Set the colour range of the active metric (clamped to its bounds)."""
        b_lo, b_hi = OUTLINE_METRICS[self.metric]["bounds"]
        lo = float(np.clip(lo, b_lo, b_hi - RANGE_MIN_GAP))
        hi = float(np.clip(hi, lo + RANGE_MIN_GAP, b_hi))
        if [lo, hi] == self.color_range[self.metric]:
            self._sync_range_sliders()
            return
        self.color_range[self.metric] = [lo, hi]
        self._configure_metric()
        self.pl.render()

    def set_metric(self, key: str):
        if not self.has_metrics:
            # no history: the metric buttons are inert (undo the radio group's own switch)
            for k, widget in self._radios.items():
                widget.GetRepresentation().SetState(int(k == self.metric))
            self._log("subject not optimized: no metrics" if self.not_optimized
                      else "no history for this subject: metric colours unavailable")
            self.pl.render()
            return
        if key not in OUTLINE_METRICS or key == self.metric:
            return
        if OUTLINE_METRICS[key]["col"] not in self.metric_range:
            self._log(f"metric {OUTLINE_METRICS[key]['label']}: no values in history.csv")
        self.metric = key
        for k, widget in self._radios.items():
            widget.GetRepresentation().SetState(int(k == key))
        self._configure_metric()
        self._apply_metric_values()
        self._update_profiles()
        self._log(f"outline colour: {OUTLINE_METRICS[key]['label']}")
        self.pl.render()

    def _set_initial_camera(self):
        """Oblique view roughly in the SA plane so the stack is seen from the
        side and the long-axis planes are not hidden behind it."""
        snap = self.snapshots[self.snap_index]
        sa = [e for e in self.entries if e["view"] == "SA"]
        sa = [e for e in sa if e["image"] == sa[0]["image"]] or self.entries      # the first SA stack
        A = self._affine_for(sa[len(sa) // 2], snap)
        u, v, n = (A[:3, c] / np.linalg.norm(A[:3, c]) for c in range(3))
        self.pl.view_vector(0.45 * n + 1.0 * u + 0.8 * v, viewup=-n)
        self.pl.reset_camera()

    def _display_sigma(self) -> float:
        """In-plane blur (mm) of the slice images: the optimizer's blur at the shown step with the blur toggle on."""
        if not (self.show_blur and getattr(self, "config", None) is not None and getattr(self, "snapshots", None)):
            return 0.0
        return self._loss_state(self.snapshots[self.snap_index]["step"])[1]

    def _colors(self, s: dict, frame: int | None = None, sigma: float | None = None) -> np.ndarray:
        """RGB per pixel (cell_colors) of slice dict s at `frame` (default: the shown one), blurred with sigma mm
        (default: as at the shown step with the blur toggle on, else none)."""
        frame = self.frame if frame is None else frame
        sigma = self._display_sigma() if sigma is None else sigma
        if sigma > 1e-6:
            raw = self._blurred({"name": s["name"], "slice": s}, sigma, frame, keep=False)
            grey = (np.clip((raw - s["lo"]) / (s["hi"] - s["lo"]), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        else:
            grey = s["grey"][:, :, frame]
        seg = None if s["seg"] is None else s["seg"][:, :, frame]
        return cell_colors(grey, seg, self.show_seg, self.seg_alpha)

    def _color_key(self, s: dict, frame: int | None = None, sigma: float | None = None) -> tuple:
        frame = self.frame if frame is None else frame
        sigma = self._display_sigma() if sigma is None else sigma
        return id(s["grey"]), frame, bool(self.show_seg), float(self.seg_alpha), round(sigma, 4)

    def _slice_rgb(self, s: dict, frame: int | None = None, sigma: float | None = None,
                   prefetch: bool = False) -> np.ndarray | None:
        """_colors through the colour cache (FrameCache, COLOR_CACHE_MB): stepping back to a frame is free.
        prefetch: for the cache only; None when it has no room (see FrameCache)."""
        key = self._color_key(s, frame, sigma)
        rgb = self._color_cache.get(key)
        if rgb is not None:
            return rgb
        if prefetch and not self._color_cache.make_room(key, s["grey"][:, :, 0].size * 3, self.frame, self.n_frames):
            return None
        rgb = self._colors(s, frame, sigma)
        self._color_cache.put(key, rgb, rgb.nbytes, prefetch, self.frame, self.n_frames)
        return rgb

    def _recolor_entry(self, e: dict):
        """Bring a slice's texture to the shown frame and display settings (nothing to do if it is)."""
        key = self._color_key(e["slice"])
        if e["rgb_key"] == key:
            return
        e["tex_img"].point_data["rgb"][:] = self._slice_rgb(e["slice"])
        e["tex_img"].Modified()
        e["rgb_key"] = key

    def _recolor(self):
        """Recolour the shown slices (hidden ones when they are shown again: _apply_visibility); the colours of the
        coming frames follow in the background (_bg_step). Colours of other display settings (segmentation shown or
        not, its opacity) are dropped."""
        settings = (bool(self.show_seg), float(self.seg_alpha))
        if settings != self._color_settings:
            self._color_cache.clear()
            self._color_settings = settings
        self._colors_sigma = self._display_sigma()
        for e in self.entries:
            if e["visible"]:
                self._recolor_entry(e)
        self._ensure_timer()

    def _bg_stream(self, idents: list):
        """The background prefetch, frame by frame from the shown one on (the shown frame, the next, ...): the shown
        slices' colours at every blur level of the history (one level, none, with the blur toggle off; levels of the
        steps nearest the shown one first), then the chart data of the shown charts (`idents`) at every step (nearest
        first). Units: ("rgb", entry, frame, sigma) or ("chart", snapshot index, intersection, frame)."""
        steps = sorted(range(len(self.snapshots)), key=lambda si: abs(si - self.snap_index))
        if self.show_blur and self.config is not None:
            levels = list(dict.fromkeys(round(self._loss_state(self.snapshots[si]["step"])[1], 4) for si in steps))
        else:
            levels = [0.0]
        shown = [e for e in self.entries if e["visible"]]
        for df in range(self.n_frames):
            f = (self.frame + df) % self.n_frames
            for sg in levels:
                for e in shown:
                    yield "rgb", e, f, sg
            for si in steps:
                for idn in idents:
                    yield "chart", si, idn, f

    def _bg_step(self) -> tuple[bool, bool]:
        """One unit of the background prefetch (_bg_stream), remade whenever the state it depends on changes (step,
        frame, shown slices and charts, display settings). Units already cached are skipped; a part whose cache is
        full stops. (did work, charts changed)"""
        charts = self.show_profiles or self.show_metric_profiles
        idents = [self._range_ident(c["it"]) for i, c in (self._pcache or {}).get("data", {}).items()
                  if charts and any(slot["bound"] == i for slot in self._profile_charts)]
        sig = (self._range_key(), self.frame, self.snap_index, tuple(idents), bool(self.show_seg),
               float(self.seg_alpha), bool(self.show_blur), tuple(e["visible"] for e in self.entries), id(self.entries))
        if sig != self._bg_sig:
            self._bg, self._bg_sig, self._bg_full = self._bg_stream(idents), sig, set()
        if self._bg is None:
            return False, False
        rk = self._range_key()
        if rk != self._chart_state:                    # (a full cache of another chart state would stop the charts)
            self._chart_cache.clear()
            self._chart_state = rk
        skipped = 0
        for unit in self._bg:
            skipped += 1
            if skipped > 500:                          # (only cached units so far: back to the timer's time check,
                self._bg = itertools.chain([unit], self._bg)          # this unit first next time)
                return True, False
            if unit[0] in self._bg_full:               # (its cache is full of nearer frames: the rest is further)
                continue
            if unit[0] == "rgb":
                _, e, f, sg = unit
                if self._color_key(e["slice"], f, sg) in self._color_cache:
                    continue
                if self._slice_rgb(e["slice"], f, sg, prefetch=True) is None:
                    self._bg_full.add("rgb")
                    continue
                return True, False
            _, si, idn, f = unit
            if (rk, si, f, idn) in self._chart_cache:
                continue
            it = next((x for x in self._profile_items(self.snapshots[si]) if self._range_ident(x) == idn), None)
            if it is None:
                continue
            if self._chart_entry(si, it, f, prefetch=True) is None:
                self._bg_full.add("chart")
                continue
            return True, False
        self._bg = None                                        # done, until the state changes
        return False, False

    def _slider_title(self, title: str, x0: float, y: float):
        """Title of a bottom slider, right-aligned just left of its tube (x0, y normalised)."""
        w, h = self.pl.window_size
        txt = self._text(title, position=(x0 * w - SLIDER_TITLE_GAP, y * h), font_size=8, color="white")
        tp = txt.GetTextProperty()
        tp.SetJustificationToRight()
        tp.SetVerticalJustificationToCentered()
        return txt

    def _header_text(self, y: float, font_size: int, color: str):
        """Header line centred at the top middle of the window, its top edge at pixel row y."""
        txt = self._text("", position=(self.pl.window_size[0] / 2, y), font_size=font_size, color=color)
        tp = txt.GetTextProperty()
        tp.SetJustificationToCentered()
        tp.SetVerticalJustificationToTop()
        return txt

    def _knob_text(self, y: float):
        txt = self._text("", position=(0, y), font_size=10, color="white")
        txt.GetTextProperty().SetJustificationToCentered()
        txt.GetTextProperty().BoldOn()
        return txt

    def _update_knob_labels(self):
        """Value labels just above the history / cardiac-phase knobs (orange on the best pose)."""
        knob = getattr(self, "_knob", None)
        if not knob:
            return
        w, h = self.pl.window_size
        snap = self.snapshots[self.snap_index]
        for name, value, colour in (("hist", snap["step"], "#ffa726" if snap["is_best"] else "white"),
                                    ("frame", self.frame, "white")):
            k = knob.get(name)
            if k is None:
                continue
            # the knob centre travels from (end cap + knob length) inside one end of the line to the same inside
            # the other end (vtkSliderRepresentation2D)
            slider = self._hist_slider if name == "hist" else self._frame_slider
            rep = slider.GetRepresentation()
            inset = rep.GetEndCapLength() + rep.GetSliderLength()
            frac = inset + (1.0 - 2.0 * inset) * min(1.0, value / k["vmax"])
            x = (k["x0"] + (k["x1"] - k["x0"]) * frac) * w
            k["text"].SetPosition(x, k["y"] * h + KNOB_LABEL_DY)
            k["text"].SetInput(f"{int(value)}")
            k["text"].GetTextProperty().SetColor(*pv.Color(colour).float_rgb)

    def _refresh_colors(self):
        self._recolor()
        self._update_knob_labels()
        self._update_header()
        self._update_profiles()
        self.pl.render()

    def _apply_visibility(self):
        self._ensure_timer()                             # (prefetch for the shown slices)
        for e in self.entries:
            if e["visible"]:
                self._recolor_entry(e)                   # (hidden slices are not recoloured)
            e["actor"].SetVisibility(e["visible"])
            e["outline"].SetVisibility(e["visible"] and self.show_outlines)
        for m in self._mask_items:                 # a contour follows its plane's visibility
            e = self._entry_by_key.get(m["key"])
            m["actor"].SetVisibility(self.show_mask and (e is None or e["visible"]))
        self._merge_burned(self._line_items, self.show_lines)   # burned-in lines: only the shown slices'
        self._merge_burned(self._sa_line_items, self.show_sa_lines)
        self._apply_radius_visibility()

    def _log(self, msg: str):
        if self.verbose:
            print(f"[viewer] {msg}", flush=True)

    # --- checkbox helpers --------------------------------------------------
    def _add_checkbox(self, label: str, cb, value: bool, key: str, y: int):
        def _cb(state):
            self._log(f"{label}: {'on' if state else 'off'}")
            cb(bool(state))

        widget = self.pl.add_checkbox_button_widget(
            _cb, value=value, position=(PANEL_X_BOX, y), size=20, border_size=1,
            color_on="#4caf50", color_off="#555555", background_color="#303030",
        )
        self._row_text[key] = self._text(label, position=(PANEL_X_TXT, y + 2), font_size=9,
                                               color="white")
        self._widgets[key] = widget

    def _add_radio(self, label: str, key: str, y: int):
        widget = self.pl.add_radio_button_widget(
            lambda: self.set_metric(key), "outline_metric", value=(key == self.metric),
            position=(PANEL_X_BOX, y), size=20, border_size=4,
            color_on="#ffa726", color_off="#555555", background_color="#303030",
        )
        self._radio_text[key] = self._text(label, position=(PANEL_X_TXT, y + 2), font_size=9, color="white")
        self._radios[key] = widget

    def _paint_checkbox(self, key: str, value: float):
        """Fill the 'on' image of a checkbox with the metric colour of `value` (nan -> grey)."""
        widget = self._widgets.get(key)
        if widget is None:
            return
        if np.isfinite(value):
            rgb = np.asarray(self._lut.map_value(float(value)), dtype=float)[:3]
            rgb = rgb / 255.0 if rgb.max() > 1.0 else rgb
        else:
            rgb = np.asarray(pv.Color(OUTLINE_NAN_COLOR).float_rgb)
        ck = tuple(np.round(rgb, 3))
        if self._tex_cache.get(key, (None,))[0] == ck:
            return
        tex = checkbox_texture(rgb)
        self._tex_cache[key] = (ck, tex)         # keep a reference to the image
        rep = widget.GetRepresentation()
        rep.SetButtonTexture(1, tex)
        rep.Modified()

    def _update_toggle_colors(self):
        """Colour every slice toggle like its outline; the SA / LA group toggles use the mean value."""
        if not self.has_metrics:
            return                                   # keep the plain green toggles
        snap = self.snapshots[self.snap_index]
        vals = {e["name"]: self._metric_value(e, snap) for e in self.entries}
        for name, v in vals.items():
            self._paint_checkbox(name, v)
        for key, prefix in (("all_sa", "SA"), ("all_la", "LA")):
            group = np.array([vals[e["name"]] for e in self._listed_entries() if e["view"].startswith(prefix)],
                             dtype=float)
            group = group[np.isfinite(group)]
            self._paint_checkbox(key, float(group.mean()) if group.size else float("nan"))

    def _set_checkbox_state(self, key: str, state: bool):
        """Update a checkbox's appearance without firing its callback."""
        widget = self._widgets.get(key)
        if widget is not None:
            widget.GetRepresentation().SetState(int(state))

    def _toggle_checkbox(self, key: str):
        """Emulate a click on a checkbox: flip its state and fire its callback."""
        widget = self._widgets.get(key)
        if widget is None:
            return
        rep = widget.GetRepresentation()
        rep.SetState(int(not bool(rep.GetState())))
        # pyvista registers the checkbox callback on StateChangedEvent
        widget.InvokeEvent("StateChangedEvent")

    # --- ui ----------------------------------------------------------------
    def _build_ui(self):
        pl = self.pl
        w, h = pl.window_size
        y = self._panel_top

        def add_row(label, cb, value, key):
            nonlocal y
            self._add_checkbox(label, cb, value, key, y)
            y -= ROW_PITCH

        add_row("Segmentation  [s]", self._on_seg_toggle, self.show_seg, "seg")
        add_row("Outlines  [o]", self._on_outline_toggle, self.show_outlines, "outlines")
        if self._mask_items:
            add_row("Dilated mask contour  [m]", self._on_mask_toggle, self.show_mask, "mask")
        if self._line_items is not None:
            add_row("Sampling lines  [i]", self._on_lines_toggle, self.show_lines, "lines")
        if self._sa_line_items is not None:
            add_row("SA-SA sampling lines  [k]", self._on_sa_lines_toggle, self.show_sa_lines, "sa_lines")
        if self._radius_items:
            add_row("LA radius on SA  [u]", self._on_radius_toggle, self.show_radius, "radius")
        add_row("Intersection profiles  [p]", self._on_profiles_toggle, self.show_profiles, "profiles")
        add_row("Metric profiles  [n]", self._on_metric_profiles_toggle, self.show_metric_profiles, "metric_profiles")
        if self.config is not None:
            add_row("Apply PSF to profile  [y]", self._on_psf_toggle, self.show_psf, "psf")
            # the optimizer's in-plane blur at the shown step, right of the toggle (updated per step: _update_blur_text)
            self._blur_text = self._text("", position=(BLUR_TEXT_X, y + 4), font_size=8, color="#bbbbbb")
            add_row("Image blurring  [g]", self._on_blur_toggle, self.show_blur, "blur")
        y -= 8
        if self._multi:                                  # modality tabs: the rows below are the active tab's
            y = self._build_tabs(y)
        # every checkbox shows the actual state: on a run change the slices are kept with their visibility (load_index)
        # (with several modalities the group rows and the slice rows are those of the listed modality)
        listed = self._listed_entries()
        if any(e["view"] == "SA" for e in self.entries):
            add_row("All SA  [a]", lambda v: self._set_group("SA", v),
                    all(e["visible"] for e in listed if e["view"] == "SA"), "all_sa")
        if any(e["view"] != "SA" for e in self.entries):     # (LA views and the aortic slice)
            add_row("All LA  [l]", lambda v: self._set_group("LA", v),
                    all(e["visible"] for e in listed if e["view"] != "SA"), "all_la")
        y -= 8
        # column headers above the slice rows: the outline metric (value column) and "profile"
        self._text("profile", position=(PROFILE_RADIO_X + 7, y + ROW_PITCH - 4), font_size=7,
                   color="#bbbbbb").GetTextProperty().SetJustificationToCentered()
        self._value_header = self._text("", position=(PANEL_X_VAL, y + ROW_PITCH - 4), font_size=7,
                                        color="#bbbbbb")
        # one block of rows per modality, all at the same place: only the listed one is shown (_apply_listing)
        y_list = y_end = y
        for mod in (self.modalities if self._multi else [None]):
            y = y_list
            for e in self._menu_entries(mod):
                self._profile_radios[e["name"]] = pl.add_radio_button_widget(
                    lambda n=e["name"]: self.set_profile(n), "profile", value=(e["name"] == self.profile_key),
                    position=(PROFILE_RADIO_X, y + 3), size=14, border_size=3,
                    color_on="#ffa726", color_off="#555555", background_color="#303030")
                self._row_value[e["name"]] = self._text("", position=(PANEL_X_VAL, y + 2), font_size=9, color="white")
                add_row(self._row_label(e), self._make_slice_cb(e), e["visible"], e["name"])
            y_end = min(y_end, y)
        y = y_end
        if self._multi:
            self._apply_listing()

        # Outline-colour metric (radio group: exactly one active); each row also shows the
        # subject-level value of its metric at the shown step (filled by _update_header)
        y -= 14
        self._text("Outline colour by", position=(PANEL_X_BOX, y + 2), font_size=9, color="#bbbbbb")
        y -= ROW_PITCH
        self._radio_text = {}
        for key, spec in OUTLINE_METRICS.items():
            self._add_radio(spec["label"], key, y)
            y -= ROW_PITCH

        # Header texts first: slider callbacks may fire during widget creation
        # (top middle, centred, each anchored at its top edge)
        self.title_text = self._header_text(h - HEADER_TOP_DY, 11, "white")
        self.run_name_text = self._header_text(h - HEADER_TOP_DY - 30, 7, "#80cbc4")
        self.outline_text = self._header_text(h - HEADER_TOP_DY - 60, 8, "#bbbbbb")

        # History slider (bottom, right-aligned): its value is the optimisation step, snapped to
        # the nearest snapshot; the knob label shows that step.  Cardiac-phase slider above it.
        # Their titles are drawn left of the tubes (see _slider_title).
        hx0, hx1, hy, fy = 0.42, 0.95, HIST_SLIDER_Y, FRAME_SLIDER_Y
        self._snap_steps = np.array([snap["step"] for snap in self.snapshots], dtype=float)
        step_max = max(float(self._snap_steps.max()), 1.0)   # a single snapshot still needs a non-empty range
        self._hist_slider = pl.add_slider_widget(
            self._on_hist_slider, rng=[0.0, step_max], value=self._snap_steps[self.snap_index],
            title="", pointa=(hx0, hy), pointb=(hx1, hy),
            fmt="%.0f", style="modern", color="white", tube_width=0.006, slider_width=0.016,
            interaction_event="always",   # update while dragging, not only on release
        )
        self._hist_slider.GetRepresentation().ShowSliderLabelOff()
        self._slider_title("optimization history" if self.has_metrics
                           else "acquired pose (not optimized)" if self.not_optimized else "saved pose (no history)", hx0, hy)
        self._knob = {"hist": dict(x0=hx0, x1=hx1, y=hy, vmax=step_max,
                                   text=self._knob_text(hy * h + KNOB_LABEL_DY))}
        # with a single snapshot the knob simply snaps back (see _on_hist_slider)
        self._text(f"{int(self._snap_steps.max())}", position=(hx1 * w + 30, hy * h), font_size=9,
                   color="#bbbbbb").GetTextProperty().SetVerticalJustificationToCentered()
        if self.history["best_index"] is not None:
            # marker right below the slider at the best pose's position
            best_step = self._snap_steps[self.history["best_index"]]
            bx = hx0 + (hx1 - hx0) * best_step / step_max
            self._text(f"best {int(best_step)}", position=(bx * w - 22, max(2, hy * h - 24)), font_size=7,
                        color="#ffa726")

        if self.n_frames > 1:
            self._frame_slider = pl.add_slider_widget(
                self._on_frame_slider, rng=[0, self.n_frames - 1], value=self.frame,
                title="", pointa=(hx0, fy), pointb=(hx1, fy),
                fmt="%.0f", style="modern", color="white", tube_width=0.006, slider_width=0.016,
                interaction_event="always",
            )
            self._frame_slider.GetRepresentation().ShowSliderLabelOff()
            self._slider_title("cardiac phase", hx0, fy)
            self._knob["frame"] = dict(x0=hx0, x1=hx1, y=fy, vmax=max(1, self.n_frames - 1),
                                       text=self._knob_text(fy * h + KNOB_LABEL_DY))
        self._update_knob_labels()

        # Colour bar (the key text above is filled in by _configure_metric); none without metrics
        if self.entries and self.has_metrics:
            pl.add_scalar_bar(
                title="outline colour", mapper=self.entries[0]["outline"].mapper, n_labels=0,
                vertical=False, position_x=BAR_X0, position_y=BAR_Y0, width=BAR_X1 - BAR_X0,
                height=BAR_H, color="white",
            )
            sb = pl.scalar_bars["outline colour"]
            sb.SetLookupTable(self._lut)
            sb.SetTitle("")                 # the metric name is in the key text and radio group
            sb.SetNumberOfLabels(0)
            if hasattr(sb, "SetDrawTickLabels"):
                sb.SetDrawTickLabels(False)
            sb.SetBarRatio(1.0)             # the strip fills the whole box
            sb.SetTextPad(0)
            self._scalar_bar = sb
            self._add_range_sliders()


    # --- modalities ------------------------------------------------------
    def _build_tabs(self, y: int) -> int:
        """Modality tabs at row y (subjects with several modalities): per modality a checkbox showing / hiding all of
        its slices (key h: the active tab's) and a tab, a button whose click lists that modality's slices in the rows
        below (keys v / V: next / previous tab). TABS_PER_ROW tabs per row, sharing the panel width; labels shortened
        to fit. Returns the y of the next row."""
        pl = self.pl
        per_row = min(len(self.modalities), TABS_PER_ROW)
        slot = (TAB_X1 - PANEL_X_BOX) // per_row
        for i, mod in enumerate(self.modalities):
            r, c = divmod(i, per_row)
            x, ty = PANEL_X_BOX + c * slot, y - r * ROW_PITCH
            mine = [e for e in self.entries if e["modality"] == mod]
            self._widgets[f"mod_shown:{mod}"] = pl.add_checkbox_button_widget(
                lambda state, m=mod: self._set_modality_visible(bool(state), m),
                value=all(e["visible"] for e in mine), position=(x, ty + 1), size=18, border_size=1,
                color_on="#4caf50", color_off="#555555", background_color="#303030")
            tx0, tx1 = x + 21, x + slot - 4
            tab = pl.add_checkbox_button_widget(lambda _state, m=mod: self._on_tab(m), value=(mod == self.listed_modality),
                                                position=(tx0, ty), size=20)
            textures = {on: tab_texture(tx1 - tx0, 20, fill, frame) for on, (fill, frame, _) in TAB_COLORS.items()}
            tab_rep = tab.GetRepresentation()
            tab_rep.SetButtonTexture(0, textures[False])
            tab_rep.SetButtonTexture(1, textures[True])
            tab_rep.PlaceWidget([tx0, tx1, ty, ty + 20, 0.0, 0.0])      # a rectangle instead of the square checkbox
            chars = max(3, int((tx1 - tx0 - 10) // TAB_CHAR_PX))
            name = mod if len(mod) <= chars else mod[:chars - 1] + "\u2026"
            text = self._text(name, position=(tx0 + 6, ty + 4), font_size=7, color=TAB_COLORS[False][2])
            self._widgets[f"tab:{mod}"] = tab
            self._tabs[mod] = {"widget": tab, "text": text, "textures": textures}   # (textures: kept referenced)
        self._sync_tabs()
        return y - -(-len(self.modalities) // per_row) * ROW_PITCH

    def _sync_tabs(self):
        """The active tab (the listed modality) highlighted, the others not."""
        for mod, t in self._tabs.items():
            on = mod == self.listed_modality
            t["widget"].GetRepresentation().SetState(int(on))
            t["text"].GetTextProperty().SetColor(pv.Color(TAB_COLORS[on][2]).float_rgb)

    def _on_tab(self, modality: str):
        self.set_listed_modality(modality)
        self._sync_tabs()                     # (a click on the active tab would otherwise switch its highlight off)
        self.pl.render()

    def _apply_listing(self):
        """Show the slice rows of the listed modality, hide the others' (their widgets also stop reacting)."""
        if not self._multi:
            return
        for e in self.entries:
            show = e["modality"] == self.listed_modality
            for wdg in (self._widgets.get(e["name"]), self._profile_radios.get(e["name"])):
                if wdg is None:
                    continue
                if wdg.GetInteractor() is not None:
                    wdg.SetEnabled(int(show))
                wdg.GetRepresentation().SetVisibility(int(show))
            for txt in (self._row_text.get(e["name"]), self._row_value.get(e["name"])):
                if txt is not None:
                    txt.SetVisibility(show)

    def set_listed_modality(self, modality: str):
        if not self._multi or modality == self.listed_modality or modality not in self.modalities:
            return
        self.listed_modality = modality
        for e in self.entries:                           # row labels: the digit keys follow the listed SA slices
            txt = self._row_text.get(e["name"])
            if txt is not None:
                txt.SetInput(self._row_label(e))
        self._apply_listing()
        self._sync_group_boxes()
        self._update_toggle_colors()
        self._sync_tabs()
        self._log(f"listed modality: {modality}")
        self.pl.render()

    def step_modality(self, step: int):
        if self._multi:
            i = self.modalities.index(self.listed_modality)
            self.set_listed_modality(self.modalities[(i + step) % len(self.modalities)])

    def _set_modality_visible(self, value: bool, modality: str | None = None):
        """Show / hide every slice of a modality (default: the listed one)."""
        modality = self.listed_modality if modality is None else modality
        for e in self.entries:
            if e["modality"] != modality:
                continue
            e["visible"] = bool(value)
            self._set_checkbox_state(e["name"], e["visible"])
        self._sync_group_boxes()
        self._apply_visibility()
        self.pl.render()

    # --- callbacks -------------------------------------------------------
    def _sync_group_boxes(self):
        """'All SA' / 'All LA' (of the listed modality) and each modality tab's checkbox show ticked exactly when
        every slice of their group is shown."""
        listed = self._listed_entries()
        self._set_checkbox_state("all_sa", all(e["visible"] for e in listed if e["view"] == "SA"))
        self._set_checkbox_state("all_la", all(e["visible"] for e in listed if e["view"] != "SA"))
        for mod in self._tabs:
            self._set_checkbox_state(f"mod_shown:{mod}", all(e["visible"] for e in self.entries if e["modality"] == mod))

    def _make_slice_cb(self, e: dict):
        def cb(value: bool):
            e["visible"] = bool(value)
            self._set_checkbox_state(e["name"], e["visible"])
            self._sync_group_boxes()
            self._apply_visibility()
            self.pl.render()
        return cb

    def _set_group(self, prefix: str, value: bool):
        for e in self._listed_entries():
            # "LA": every non-SA view, the aortic slice included (it is handled like a LA view)
            if e["view"].startswith(prefix) or (prefix == "LA" and e["view"] != "SA"):
                e["visible"] = bool(value)
                self._set_checkbox_state(e["name"], e["visible"])
        self._sync_group_boxes()
        self._apply_visibility()
        self.pl.render()

    def _on_seg_toggle(self, value: bool):
        self.show_seg = bool(value)
        self._refresh_colors()

    def _on_outline_toggle(self, value: bool):
        self.show_outlines = bool(value)
        self._apply_visibility()
        self.pl.render()

    def _on_mask_toggle(self, value: bool):
        self.show_mask = bool(value)
        self._apply_masks(self.snapshots[self.snap_index])        # (not kept up to date while hidden)
        self._apply_visibility()
        self.pl.render()

    def _on_radius_toggle(self, value: bool):
        self.show_radius = bool(value)
        self._apply_radius(self.snapshots[self.snap_index])
        self._apply_radius_visibility()
        self.pl.render()

    def _on_sa_lines_toggle(self, value: bool):
        self.show_sa_lines = bool(value)
        self._apply_sa_lines(self.snapshots[self.snap_index])
        self._apply_sa_line_visibility()
        self.pl.render()

    def _on_lines_toggle(self, value: bool):
        self.show_lines = bool(value)
        self._apply_lines(self.snapshots[self.snap_index])
        self._apply_visibility()
        self.pl.render()

    def _on_frame_slider(self, value: float):
        self.set_frame(int(round(value)), from_slider=True)
        self._snap_slider(self._frame_slider, self.frame)

    def set_frame(self, frame: int, from_slider: bool = False):
        frame = int(np.clip(frame, 0, self.n_frames - 1))
        if frame == self.frame:
            return
        self.frame = frame
        if self._frame_slider is not None and not from_slider:
            self._frame_slider.GetRepresentation().SetValue(frame)
        self._refresh_colors()

    def _on_hist_slider(self, value: float):
        if abs(self._snap_steps[self.snap_index] - float(value)) < 1e-6:
            return      # still on the current snapshot (also keeps "best" when a regular snapshot shares its step)
        index = int(np.argmin(np.abs(self._snap_steps - float(value))))
        if index == self.snap_index:            # e.g. a single snapshot: the knob does not move
            self._snap_slider(self._hist_slider, self._snap_steps[self.snap_index])
            return
        self.set_snapshot(index, from_slider=True)
        self._snap_slider(self._hist_slider, self._snap_steps[self.snap_index])

    @staticmethod
    def _snap_slider(slider, value: float):
        """Move the knob onto the value it was resolved to."""
        if slider is None:
            return
        rep = slider.GetRepresentation()
        if abs(rep.GetValue() - value) > 1e-6:
            rep.SetValue(value)

    def set_snapshot(self, index: int, from_slider: bool = False):
        index = int(np.clip(index, 0, len(self.snapshots) - 1))
        if index == self.snap_index:
            return
        self.snap_index = index
        if self._hist_slider is not None and not from_slider:
            self._hist_slider.GetRepresentation().SetValue(self._snap_steps[index])
        self._apply_snapshot()
        snap = self.snapshots[index]
        self._log(f"snapshot {index}: step {snap['step']}/{self.n_steps}"
                  f"{' (best)' if snap['is_best'] else ''}")
        self.pl.render()

    # --- header ------------------------------------------------------------
    def _update_header(self):
        snap = self.snapshots[self.snap_index]
        sig = self._display_sigma()
        dil = self._dilation(self.snapshots[self.snap_index]["step"]) if getattr(self, "snapshots", None) else None
        extra = (f"   mask +{round(dil, 1):g} mm" if dil is not None else "") + (f"   images blurred {sig:.1f} mm" if sig > 1e-6 else "")
        self.title_text.SetInput(self.title + extra)
        # wrap long run names at '-' so they stay inside the window (two lines at most)
        lines, cur = [], "run "
        for part in self.run_name.split("-"):
            piece = part if cur in ("run ",) else "-" + part
            if len(cur) + len(piece) > 84 and cur.strip() != "run":
                lines.append(cur)
                cur = "    " + part
            else:
                cur += piece
        lines.append(cur)
        if len(lines) > 2:
            lines = [lines[0], (lines[1] + "-" + "-".join(l.strip() for l in lines[2:]))[:84] + "…"]
        self.run_name_text.SetInput("\n".join(lines))
        if self.pl.render_window is not None:
            self.pl.render_window.SetWindowName(f"{self.title}   run {self.run_name}")

        # subject-level value of each metric next to its radio button (Hausdorff: mean and max)
        sm = snap["summary"] or {}
        for key, spec in OUTLINE_METRICS.items():
            if not self.has_metrics:
                value = "(not optimized)" if self.not_optimized else "(no history)"
            elif spec["col"] not in self.metric_range:
                value = "(no values)"
            else:
                vals = [sm.get(c) for c in spec["summary"]]
                vals = [spec["fmt"].format(v) if v is not None and np.isfinite(v) else "-" for v in vals]
                value = vals[0] + (f"  (max {vals[1]})" if len(vals) > 1 else "")
            self._radio_text[key].SetInput(f"{spec['label']}   {value}")

    # --- run -------------------------------------------------------------
    def show(self, **kwargs):
        self.pl.show(**kwargs)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def _subjects_in_run(run_dir: Path) -> list[str]:
    """Viewable subjects of a run (with or without a history), alphabetical."""
    return sorted(p.name for p in run_dir.iterdir() if _is_subject_dir(p))


# pose models of motion_gradient.pose_model that are not the full 6-DOF pose, as shown in the header
POSE_NOTES = {"inplane": "translation + in-plane rotation only",
              "translation": "translation only",
              "none": "not optimized: acquired pose"}


def load_alignment_status(metrics_dir: Path) -> tuple[str | None, str | None]:
    """(status, pose_model) of the subject from metrics/summary.csv; None for what is not recorded (runs without
    --logging, or from before the pose_model column)."""
    path = metrics_dir / "summary.csv"
    df = read_csv(path) if path.exists() and path.stat().st_size else None
    if df is None or df.empty:
        return None, None
    row = df.iloc[-1]
    get = lambda c: row[c] if c in row.index and isinstance(row[c], str) else None  # noqa: E731
    return get("status"), get("pose_model")


def history_from_images(data: dict) -> dict:
    """Stand-in history for a subject aligned without --logging: one snapshot holding the saved
    (final) pose from the NIfTI affines and no metrics."""
    affines = {(view, int(s["index"])): s["affine"] for view, v in data.items() for s in v["slices"]}
    snap = {"step": 0, "fraction": 0.0, "is_best": False, "affines": affines, "metrics": {}, "summary": None}
    return {"snapshots": [snap], "n_steps": 0, "best_index": None, "metric_range": {}, "available": False}


def run_config_for(run_dir: Path, metrics_dir: Path, history: dict) -> dict | None:
    """The settings the subject was aligned with. A run directory aligned into again with other settings keeps one
    config per start (run_config.json, then run_config_<YYYYMMDD-HHMMSS>.json, see motion_gradient.write_run_config);
    the subject's is the newest one started before it was aligned (the timestamp in its metrics/summary.csv, else its
    history's modification time). Everything the viewer rebuilds per step (blur, PSF scale, mask annealing stage)
    depends on it, e.g. --steps."""
    om = run_dir / "overall_metrics"
    starts = []
    for path in om.glob("run_config*.json"):
        m = re.search(r"run_config_(\d{8}-\d{6})\.json$", path.name)
        try:
            t = datetime.strptime(m.group(1), "%Y%m%d-%H%M%S").timestamp() if m else path.stat().st_mtime
        except (OSError, ValueError):
            continue
        starts.append((t, path))
    if not starts:
        print(f"[warn] {om / 'run_config.json'} missing: profiles can only show raw intensities", file=sys.stderr)
        return None
    starts.sort()
    aligned = None
    try:
        row = pd.read_csv(metrics_dir / "summary.csv").iloc[-1]
        aligned = datetime.fromisoformat(str(row["timestamp"])).timestamp()
    except Exception:                                    # noqa: BLE001 (no summary / timestamp)
        hist = metrics_dir / HISTORY_CSV
        aligned = hist.stat().st_mtime if hist.exists() else None
    path = starts[-1][1] if aligned is None else next((p for t, p in reversed(starts) if t <= aligned + 1), starts[0][1])
    config = json.loads(path.read_text())
    n = history.get("n_steps")
    if len(starts) > 1:
        print(f"[config] {path.name} (one of {len(starts)} configs of this run directory)")
    if n and config.get("steps") and int(config["steps"]) != int(n):
        print(f"[warn] {path.name}: steps={config['steps']} but the history runs to step {n}: per-step "
              f"reconstructions (blur, PSF, mask annealing) may be off", file=sys.stderr)
    return config


def load_subject_run(run_dir: Path, subject: str, instance: int | None = None, cache: dict | None = None) -> dict:
    """Everything the viewer needs for one subject of a run. The images are read from <run>/<subject>/ and its
    modality folders (find_images; older runs: <run>/<subject>/Instance_<n>/, a single modality), only from
    <run>/<subject>/Instance_<instance>/ with `instance`. `cache` (a dict kept by the caller) holds the last subject's
    images: the same subject from another run then reads only the NIfTI headers (reuse_images)."""
    subject_dir = run_dir / subject
    image_dir = subject_dir / f"Instance_{instance}" if instance is not None else subject_dir
    metrics_dir = subject_dir / METRICS_SUBDIR

    print(f"Subject {subject}" + (f", instance {instance}" if instance is not None else ""))
    reused = None
    if cache is not None and cache.get("subject") == (subject, instance):
        reused = reuse_images(cache["data"], cache["sig"], image_dir)
    if reused is not None:
        data = reused[0]
        print(f"[images] {image_dir}: pixels reused from the previous run (headers read)")
    else:
        data = load_images(image_dir)
        describe_images(image_dir, data)
        if cache is not None:                            # only the latest subject is kept
            cache.clear()
            cache.update(subject=(subject, instance), data=data, sig=image_signature(image_dir))
    status, model = load_alignment_status(metrics_dir)
    not_optimized = status == "not_optimized" or model == "none"
    if (metrics_dir / HISTORY_CSV).exists():
        history = load_history(metrics_dir)
        describe_history(metrics_dir, history)
        named = {k[0] for snap in history["snapshots"] for k in snap["affines"]}
        if named and not named & set(data):
            print(f"[warn] none of the images of {metrics_dir / HISTORY_CSV} ({', '.join(sorted(named))}) is one of "
                  f"the subject's ({', '.join(data)}): its history is from another layout (e.g. a run of a single "
                  f"modality); every slice is shown at its saved pose", file=sys.stderr)
    else:
        history = history_from_images(data)
        why = ("not optimized (no crossing views): showing the acquired pose" if not_optimized
               else "aligned without --logging: showing the saved pose")
        print(f"[history] none for {subject} ({why}), no metrics")
    if model in POSE_NOTES:
        print(f"[pose] {POSE_NOTES[model]}")
    overlays = load_overlays(metrics_dir)
    if metrics_dir.is_dir() and not not_optimized:          # a subject that was not optimized has no lines / masks
        describe_overlays(metrics_dir, overlays)

    # Sanity check: the saved NIfTI affines should equal the history's best pose.
    best = history["snapshots"][history["best_index"]] if history["best_index"] is not None else None
    if best is not None:
        for view, v in data.items():
            for s in v["slices"]:
                A = best["affines"].get((view, int(s["index"])))
                if A is not None and not np.allclose(A, s["affine"], atol=5e-3):
                    print(f"[warn] {s['file']}: NIfTI affine differs from the history's best pose "
                          f"(max |diff| {np.abs(A - s['affine']).max():.3f})", file=sys.stderr)
    config = run_config_for(run_dir, metrics_dir, history)
    title = f"UKBB {subject}" + (f" / Instance_{instance}" if instance is not None else "")
    n_mod = len({v["modality"] for v in data.values()})
    if n_mod > 1:
        title += f"   ({n_mod} modalities)"
    if model in POSE_NOTES:
        title += f"   [{POSE_NOTES[model]}]"
    return {"data": data, "history": history, "overlays": overlays, "config": config, "title": title,
            "run_name": run_dir.name, "not_optimized": not_optimized}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("subject", nargs="?", help="subject id (default: first subject of the run)")
    ap.add_argument("--subject-index", type=int, default=0,
                    help="alphabetical index of the subject to display if subject id is not specified (default: 0)")
    ap.add_argument("--align-root", type=Path, default=ALIGN_ROOT,
                    help="directory containing motion-correction run directories (default: %(default)s)")
    ap.add_argument("--run", default=None,
                    help="name of the run directory under --align-root "
                         "(default: the most recently created run containing the subject)")
    ap.add_argument("--run-dir", type=Path, default=None,
                    help="explicit run directory containing <subject>/ (images + metrics/) "
                         "(overrides --align-root/--run)")
    ap.add_argument("--remote", default=None, metavar="USER@HOST:/RUN_DIR",
                    help="a run directory on another machine, read over ssh (key authentication): every subject is "
                         "downloaded once into --cache-dir when shown (the next one in the background). Subjects: "
                         "--subjects, else every subject of its overall_metrics/motion_summary.csv")
    ap.add_argument("--cache-dir", type=Path, default=REMOTE_CACHE,
                    help="where --remote subjects are kept (<cache>/<host>/<run name>/<subject>/; delete freely). "
                         "Default: %(default)s")
    ap.add_argument("--subjects", nargs="+", default=None, metavar="ID_OR_FILE",
                    help="the subjects to step through with [ / ], in this order: subject ids and/or text files with "
                         "one subject per line (first token; e.g. a list written by worst_cases.py, whose slice column "
                         "also picks the slice profiled first). Default: all subjects of the run")
    ap.add_argument("--instance", type=int, default=None,
                    help="show only the images of <subject>/Instance_<n>/ (default: every image folder of the "
                         "subject, one per modality)")
    ap.add_argument("--start", default="best",
                    help="initial snapshot: 'best', 'end', '0', or a snapshot index (default: %(default)s)")
    ap.add_argument("--frame", type=int, default=0, help="initial cardiac phase")
    ap.add_argument("--metric", choices=list(OUTLINE_METRICS), default="ncc",
                    help="metric that colours the slice outlines at start (default: %(default)s)")
    ap.add_argument("--mask", action="store_true", help="start with the dilated-mask contours shown")
    ap.add_argument("--lines", action="store_true", help="start with the intersection sampling lines shown")
    ap.add_argument("--sa-lines", action="store_true",
                    help="start with the SA-SA sampling lines (metrics/sa_lines.csv) shown")
    ap.add_argument("--no-profiles", action="store_true", help="start with the intersection profile charts hidden")
    ap.add_argument("--no-metric-profiles", action="store_true",
                    help="start with the per-point metric charts (below the profile charts) hidden")
    ap.add_argument("--blur", action="store_true",
                    help="start with the slice images blurred as at the shown optimisation step")
    ap.add_argument("--raw-profiles", action="store_true",
                    help="start with the profiles showing raw intensities instead of the loss view (blur + PSF band)")
    ap.add_argument("--radius", action="store_true",
                    help="start with the long-axis radius of the SA-SA supervision (metrics/long_axis.csv) shown")
    ap.add_argument("--no-seg", action="store_true", help="start with segmentation overlay off")
    ap.add_argument("--seg-alpha", type=float, default=0.5, help="segmentation overlay opacity (0-1)")
    ap.add_argument("--screenshot", type=Path, default=None,
                    help="render off-screen to this file instead of opening a window")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    listed = read_subject_list(args.subjects) if args.subjects else None
    if listed is not None and not listed:
        sys.exit("--subjects: no subject ids found")
    if args.remote:
        return main_remote(args, listed)

    run_dir = args.run_dir or find_run_dir(args.align_root, args.subject, args.run)
    if run_dir is None:
        sys.exit(f"No run with aligned subjects found under {args.align_root}"
                 + (f" for subject {args.subject}" if args.subject else ""))
    run_dir = Path(run_dir)

    subjects_for = _subjects_in_run
    if listed is not None:                               # --subjects: those of the list the run has, in list order
        subjects_for = lambda run: [s for s in listed if _is_subject_dir(Path(run) / s)]   # noqa: E731
    subjects = subjects_for(run_dir)
    if not subjects:
        sys.exit(f"No aligned subject in {run_dir}" + (" from --subjects" if listed is not None else ""))
    if args.subject is not None:
        if args.subject not in subjects:
            if (run_dir / args.subject).is_dir():
                sys.exit(f"Subject {args.subject} has no images in {run_dir} (it had no input images to write)")
            sys.exit(f"Subject {args.subject} has no aligned images or history in {run_dir}")
    elif not 0 <= args.subject_index < len(subjects):
        sys.exit(f"Subject index {args.subject_index} out of bounds (found {len(subjects)} subjects).")

    # Runs to navigate: every run with aligned subjects next to the chosen one (its parent
    # directory, i.e. --align-root unless --run-dir points elsewhere), newest first.
    runs = runs_by_creation(run_dir.parent)
    resolved = [r.resolve() for r in runs]
    if run_dir.resolve() not in resolved:
        runs.insert(0, run_dir)
        resolved.insert(0, run_dir.resolve())
    run_index = resolved.index(run_dir.resolve())
    print(f"{len(runs)} run(s) in {run_dir.parent}; showing run {run_index + 1} "
          f"({run_dir.name}): {len(subjects)} subject(s)")

    image_cache = {}                                     # the last subject's images, reused across runs

    def loader(run: Path, subject: str) -> dict:
        return dict(load_subject_run(run, subject, args.instance, image_cache), profile=SUBJECT_FOCUS.get(subject))

    start = None
    if args.start == "end":
        start = 10 ** 9                      # clipped to the last snapshot
    elif args.start != "best":
        start = int(args.start)

    if args.screenshot is not None:
        pv.OFF_SCREEN = True

    viewer = SliceViewer(
        runs, subjects_for, loader, run_index=run_index,
        subject=args.subject, subject_index=args.subject_index,
        frame=args.frame, show_seg=not args.no_seg, seg_alpha=args.seg_alpha, start=start,
        metric=args.metric, show_mask=args.mask, show_lines=args.lines, show_sa_lines=args.sa_lines,
        show_radius=args.radius, show_profiles=not args.no_profiles,
        show_metric_profiles=not args.no_metric_profiles,
        show_psf=not args.raw_profiles, show_blur=args.blur,
    )
    if args.screenshot is not None:
        viewer.show(screenshot=str(args.screenshot), auto_close=True)
        print(f"Saved {args.screenshot}")
    else:
        viewer.show()


def main_remote(args, listed: list[str] | None):
    """--remote: view a run on another machine through a local cache (RemoteRun). One run only (no { } run
    navigation); [ / ] step through --subjects (or all of the run's subjects), downloading each when shown."""
    remote = RemoteRun(args.remote, args.cache_dir)
    print(f"Remote run {remote.target}:{remote.path}, cached in {remote.local}")
    remote.fetch_run_config()
    subjects = listed if listed is not None else remote.list_subjects()
    if not subjects:
        sys.exit("No subjects to show")
    if args.subject is not None and args.subject not in subjects:
        subjects = [args.subject] + subjects
    elif args.subject is None and not 0 <= args.subject_index < len(subjects):
        sys.exit(f"Subject index {args.subject_index} out of bounds ({len(subjects)} subjects).")
    print(f"{len(subjects)} subject(s) to step through")

    image_cache = {}

    def loader(run: Path, subject: str) -> dict:
        try:
            remote.fetch_subject(subject)
        except RuntimeError as exc:
            sys.exit(str(exc))
        i = subjects.index(subject) if subject in subjects else -1
        for nxt in subjects[i + 1:i + 2] + subjects[max(i - 1, 0):i]:   # the next (and previous) in the background
            remote.prefetch(nxt)
        return dict(load_subject_run(run, subject, args.instance, image_cache), profile=SUBJECT_FOCUS.get(subject))

    start = None
    if args.start == "end":
        start = 10 ** 9
    elif args.start != "best":
        start = int(args.start)
    if args.screenshot is not None:
        pv.OFF_SCREEN = True
    viewer = SliceViewer(
        [remote.local], lambda run: subjects, loader, run_index=0,
        subject=args.subject, subject_index=args.subject_index,
        frame=args.frame, show_seg=not args.no_seg, seg_alpha=args.seg_alpha, start=start,
        metric=args.metric, show_mask=args.mask, show_lines=args.lines, show_sa_lines=args.sa_lines,
        show_radius=args.radius, show_profiles=not args.no_profiles,
        show_metric_profiles=not args.no_metric_profiles,
        show_psf=not args.raw_profiles, show_blur=args.blur,
    )
    if args.screenshot is not None:
        viewer.show(screenshot=str(args.screenshot), auto_close=True)
        print(f"Saved {args.screenshot}")
    else:
        viewer.show()


if __name__ == "__main__":
    main()
