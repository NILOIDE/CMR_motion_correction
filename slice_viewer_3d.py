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

The pixel data are read from the run's NIfTIs in <run>/<subject>/ (older runs: <run>/<subject>/Instance_<n>/):
    <SA>_slice00.nii.gz, <SA>_slice01...   one file per SA slice
    <LA>_2Ch.nii.gz / <LA>_3Ch / <LA>_4Ch
    nnUNet_segs/<same name>_nnUNetSeg.nii.gz
(a single <SA>.nii.gz volume is also accepted; <SA> / <LA> are the alignment script's
on-disk spellings, see ON_DISK).

Any of the views may be missing. Slices that were not part of the alignment (no rows in
history.csv, e.g. a LA view without a segmentation) are shown at their saved pose for every
snapshot and marked "unaligned" in the slice list. The header notes a restricted pose model
(translation + in-plane rotation only) from metrics/summary.csv; a subject that was not optimized
(no crossing views) is shown at its acquired pose without metrics. Subject folders without
images are skipped.

Usage
-----
    python slice_viewer_3d.py                       # most recently created run, its first subject
    python slice_viewer_3d.py 1000071               # a specific subject
    python slice_viewer_3d.py 1000071 --run steps600-lr0.05-lin-blur4x0.6
    python slice_viewer_3d.py 1000071 --run-dir D:\\some\\run\\dir
    python slice_viewer_3d.py 1000071 --start 0     # open at the acquired geometry
    python slice_viewer_3d.py 1000071 --frame 10 --no-seg

Interaction
-----------
    Slider "optimization history" (bottom)   scrub through the pose snapshots
    Slider "cardiac phase" (below it)         time frame of the cine
    Checkboxes (left)   segmentation, outlines, all SA, all LA, every individual slice
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
                        every slice shows its own copy, visible from both sides, hidden with it
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
                        3 SA-LA lines and its SA-SA crossing with the next slice, an LA view its
                        2 LA-LA lines. Curves: displayed grey value of both slices along the line
                        at the shown step and frame (thin: inside both images, thick: in the
                        loss; both follow the annealed mask dilation of the shown step, as do
                        the mask contours and the yellow bar); y toggles raw intensities / the loss view: the raw image blurred
                        and averaged over the PSF band across the other slice's slab with the blur
                        and PSF scale of the shown step (from the run's run_config.json; SA-SA
                        lines: blur only); yellow bar along the bottom: inside the LA dilated mask (either
                        LA slice of the line, at its current pose); NCC over the loss part,
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
import json
import math
import re
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pyvista as pv
from matplotlib.path import Path as PolyPath
from contourpy import contour_generator
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates

ALIGN_ROOT = Path(r"C:\Users\Nil\Documents\data\ukbb_alignment")
VIEWS = ["SA", "LA_2Ch", "LA_3Ch", "LA_4Ch"]      # data keys: file names and CSV values


def label(name: str) -> str:
    """Display form of a view / slice / line-kind name: SA -> SA, LA -> LA (the data keep SA / LA)."""
    return name.replace("SA", "SA").replace("LA", "LA")

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
LINE_USED_WIDTH = 4     # part of the line that is sampled in the loss
LINE_IMG_WIDTH = 1      # part inside both images (thin, faint)

# Intersection intensity profiles: small charts on the right, one per intersection of the
# slice picked with the radio buttons next to the slice list (SA slice: its 3 SA-LA lines and
# its SA-SA crossing with the next slice; LA view: its 2 LA-LA lines); below them a second set
# with the selected metric per point along the same lines
PROFILE_SLOTS = 4
PROFILE_X, PROFILE_W = 0.735, 0.25
PROFILE_Y0, PROFILE_H, PROFILE_PITCH = 0.19, 0.084, 0.09     # bottom row; 2 x PROFILE_SLOTS rows
NAV_RIGHT_W = 380                          # width (px) of the run / subject navigation block, top right
HEADER_TOP_DY = 14                         # header (title, run name) top edge, px below the window top
HIST_SLIDER_Y, FRAME_SLIDER_Y = 0.03, 0.075  # bottom sliders (normalised height of their tubes)
SLIDER_TITLE_GAP = 14                      # px between a bottom slider's title and its tube
PROFILE_COLORS = ("#e0e0e0", "#ffb74d")    # picked slice, the other slice of the line
PROFILE_STEP_MM = 0.5
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
    """Return array as (x, y, z, t)."""
    if arr.ndim == 2:
        return arr[:, :, None, None]
    if arr.ndim == 3:
        return arr[:, :, :, None]
    if arr.ndim == 4:
        return arr
    raise ValueError(f"Unexpected array dimensionality {arr.ndim}")


def _slice_index(path: Path) -> int:
    m = SLICE_RE.search(path.name)
    return int(m.group(1)) if m else 0


def find_view_files(base: Path, view: str) -> tuple[str | None, list[Path]]:
    """
    Locate the image file(s) of a view.

    Returns ("volume", [path]) when <view>.nii.gz exists, otherwise
    ("slices", [path_00, path_01, ...]) for per-slice files <view>_sliceNN.nii.gz
    sorted by slice index, or (None, []) when nothing is found.
    """
    vol = base / f"{to_disk(view)}{NII_SUFFIX}"
    if vol.exists():
        return "volume", [vol]
    files = [p for p in base.glob(f"{to_disk(view)}_slice*{NII_SUFFIX}") if SLICE_RE.search(p.name)]
    if files:
        return "slices", sorted(files, key=_slice_index)
    return None, []


def seg_path_for(base: Path, img_path: Path) -> Path:
    """nnUNet_segs/<image name without .nii.gz>_nnUNetSeg.nii.gz"""
    stem = img_path.name[: -len(NII_SUFFIX)]
    return base / SEG_SUBDIR / f"{stem}{SEG_SUFFIX}"


def _load_seg(seg_path: Path, img_shape: tuple, img_affine: np.ndarray, label: str):
    if not seg_path.exists():
        return None
    seg_img = nib.load(str(seg_path))
    # Some seg files are stored as scaled int16 (scl_slope/inter); nibabel then
    # returns floats that may sit just below the integer label, so round
    # instead of truncating.
    seg = _as_4d(np.rint(np.asarray(seg_img.dataobj)).astype(np.int16))
    if seg.shape != img_shape:
        print(f"[warn] {label}: seg shape {seg.shape} != image shape {img_shape}; "
              f"ignoring segmentation", file=sys.stderr)
        return None
    if not np.allclose(seg_img.affine, img_affine, atol=1e-3):
        print(f"[warn] {label}: seg affine differs from image affine; using image affine",
              file=sys.stderr)
    return seg


def load_images(instance_dir: Path) -> dict:
    """
    Load all views found in an Instance_<n> directory.

    Returns {view: {"mode": "volume"|"slices", "slices": [slice, ...]}} where each
    slice is {"grey": (nx, ny, t) uint8, "seg": (nx, ny, t) int16 | None,
    "affine": 4x4 with the origin at this slice (k = 0), "index": slice index
    within the view (0 for LA), "name": str, "file": str}.
    """
    if not instance_dir.is_dir():
        raise FileNotFoundError(instance_dir)

    data = {}
    for view in VIEWS:
        mode, paths = find_view_files(instance_dir, view)
        if mode is None:
            print(f"[warn] {view}: no {view}{NII_SUFFIX} or {view}_sliceNN{NII_SUFFIX} "
                  f"in {instance_dir}; skipping view", file=sys.stderr)
            continue

        # Collect raw slices first so intensities can be normalised over the
        # whole view (all slices and frames) for consistent grey levels.
        raw_slices = []
        for p in paths:
            img = nib.load(str(p))
            arr = _as_4d(np.asarray(img.dataobj, dtype=np.float32))
            affine = np.asarray(img.affine, dtype=np.float64)
            seg = _load_seg(seg_path_for(instance_dir, p), arr.shape, affine, p.name)
            zooms = img.header.get_zooms()
            thick_hdr = float(zooms[2]) if len(zooms) > 2 else float("nan")
            nz = arr.shape[2]
            if mode == "slices" and nz != 1:
                print(f"[warn] {p.name}: expected a single slice, found {nz}; using all",
                      file=sys.stderr)
            for k in range(nz):
                idx = _slice_index(p) if mode == "slices" else k
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
                    "file": p.name,
                    "thick_hdr": thick_hdr,
                })
        raw_slices.sort(key=lambda s: s["index"])

        lo, hi = np.percentile(np.concatenate([s["raw"].ravel() for s in raw_slices]), [1.0, 99.0])
        if hi <= lo:
            hi = lo + 1.0
        slices = []
        for s in raw_slices:
            grey = np.clip((s["raw"] - lo) / (hi - lo), 0.0, 1.0)
            name = view if len(raw_slices) == 1 else f"{view}[{s['index']:02d}]"
            slices.append({
                "grey": (grey * 255.0 + 0.5).astype(np.uint8),
                "seg": s["seg"],
                "affine": s["affine"],
                "index": s["index"],
                "name": name,
                "file": s["file"],
                "raw": s["raw"],                  # NIfTI intensities: what the loss compares
                "lo": float(lo), "hi": float(hi),  # display scaling of the view (grey = (raw - lo) / (hi - lo))
                "thick_hdr": s["thick_hdr"],
            })
        data[view] = {"mode": mode, "slices": slices}

    if not data:
        raise RuntimeError(f"No views found in {instance_dir}")
    return data


def image_signature(instance_dir: Path) -> dict:
    """What a run's images of one subject look like without reading their pixels: per image file (view, file name)
    its shape, data type and intensity scaling, whether it has a segmentation, and its affine (NIfTI headers only).
    Two runs of a subject with equal signatures (but for the affines) hold the same pixels: the alignment writes the
    input images with corrected affines (see reuse_images)."""
    sig = {}
    for view in VIEWS:
        mode, paths = find_view_files(instance_dir, view)
        for p in paths:
            img = nib.load(str(p))
            slope, inter = img.dataobj.slope, img.dataobj.inter
            sig[(view, p.name)] = dict(mode=mode, shape=tuple(img.shape), dtype=str(img.get_data_dtype()),
                                       scale=(float(slope), float(inter)),
                                       seg=seg_path_for(instance_dir, p).exists(),
                                       affine=np.asarray(img.affine, dtype=np.float64))
    return sig


def reuse_images(cached: dict, cached_sig: dict, instance_dir: Path) -> tuple[dict, dict] | None:
    """Another run's images of the cached subject without reading pixels: (data as load_images, signature), sharing
    the cached slices' arrays, with the affines of instance_dir's headers (each run saves its own corrected pose).
    None if any file differs in name, shape, data type, scaling or segmentation (e.g. a run saved with --image-dtype
    uint8): then everything has to be read."""
    sig = image_signature(instance_dir)
    strip = lambda d: {k: (v["mode"], v["shape"], v["dtype"], v["scale"], v["seg"]) for k, v in d.items()}  # noqa: E731
    if strip(sig) != strip(cached_sig):
        return None
    data = {}
    for view, v in cached.items():
        slices = []
        for s in v["slices"]:
            hdr = sig[(view, s["file"])]
            nz = hdr["shape"][2] if len(hdr["shape"]) > 2 else 1
            k = int(s["index"]) if v["mode"] == "volume" else (int(s["index"]) % nz if nz > 1 else 0)
            A0 = hdr["affine"].copy()                    # origin moved to this slice, as load_images
            A0[:3, 3] += k * hdr["affine"][:3, 2]
            slices.append(dict(s, affine=A0))
        data[view] = {"mode": v["mode"], "slices": slices}
    return data, sig


def describe_images(instance_dir: Path, data: dict) -> None:
    print(f"[images] {instance_dir}")
    for view, v in data.items():
        labs = sorted({int(x) for s in v["slices"] if s["seg"] is not None for x in np.unique(s["seg"])})
        shape = v["slices"][0]["grey"].shape
        print(f"  {label(view):8s} {v['mode']:7s} {len(v['slices']):2d} slice(s) of {shape[0]}x{shape[1]}, "
              f"{shape[2]} frames, seg labels={','.join(map(str, labs)) or 'none'}")


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
    sa_lines.csv: the same per neighbouring-SA-slice pair and history snapshot, already
    at that snapshot's poses (the slices only cross once tilted); kept here as segment
    end points, only the parts that exist (inside both images / entering the loss).
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
        pair = df[["slice_a", "slice_b"]].to_numpy(int)
        out["sa_lines"] = {}
        keys = list(zip(df["step"].astype(int), df["is_best"].astype(int).astype(bool)))
        for key in dict.fromkeys(keys):
            rows = np.array([k == key for k in keys])
            entry = {}
            for name, (ok, e) in segs.items():
                entry[name] = e[rows & ok]
                entry[name + "_pairs"] = pair[rows & ok]          # (K, 2) SA slice indices of each segment
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
    --logging) or at least the aligned images (<subject>/*.nii.gz, or <subject>/Instance_<n>/ in older runs)."""
    if not (p.is_dir() and p.name.isdigit()):
        return False
    if (p / METRICS_SUBDIR / HISTORY_CSV).exists():
        return True
    return any(p.glob(f"*{NII_SUFFIX}")) or any(q.is_dir() for q in p.glob("Instance_*"))


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


def grid_ijk(nx: int, ny: int) -> np.ndarray:
    """Voxel-edge grid points (nx+1)*(ny+1) at k = 0, ordered i-fastest."""
    ii, jj = np.meshgrid(np.arange(nx + 1) - 0.5, np.arange(ny + 1) - 0.5, indexing="ij")
    return np.column_stack([ii.ravel(order="F"), jj.ravel(order="F"), np.zeros(ii.size)])


def slice_faces(nx: int, ny: int) -> np.ndarray:
    """
    Quad faces for a (nx+1)x(ny+1) point grid, one cell per voxel.

    Cell c = i + j*nx corresponds to voxel (i, j), so cell scalars can be
    assigned with ``img2d.ravel(order="F")``.
    """
    i, j = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    i = i.ravel(order="F")
    j = j.ravel(order="F")
    w = nx + 1
    p0 = i + j * w
    p1 = (i + 1) + j * w
    p2 = (i + 1) + (j + 1) * w
    p3 = i + (j + 1) * w
    return np.column_stack([np.full(i.size, 4), p0, p1, p2, p3]).ravel()


def slice_mesh(affine: np.ndarray, nx: int, ny: int) -> pv.PolyData:
    """Flat quad mesh of a slice (k = 0 of `affine`) with one cell per voxel."""
    return pv.PolyData(voxel_to_world(affine, grid_ijk(nx, ny)), slice_faces(nx, ny))


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
    """(nx*ny, 3) uint8 RGB per cell: grey image, optionally blended with seg."""
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
        self._tex_cache: dict[str, tuple] = {}
        self._range_sliders: dict[str, object] = {}
        self._range_text: dict[str, object] = {}
        self._frame_slider = None
        self._hist_slider = None
        self._knob = {}
        self._scalar_bar = None

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
        new = [(view, s) for view in VIEWS if view in data for s in data[view]["slices"]]
        return len(new) == len(self.entries) and all(
            e["view"] == view and e["name"] == s["name"] and e["slice"]["grey"] is s["grey"]
            for e, (view, s) in zip(self.entries, new))

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
            new = [s for view in VIEWS if view in self.data for s in self.data[view]["slices"]]
            for e, s in zip(self.entries, new):
                e["slice"] = s
                e["in_history"] = any(e["key"] in snap["affines"] for snap in self.snapshots)
        else:
            self._build_slices()
        self._build_overlays()
        names = [e["name"] for e in self.entries]
        if self.profile_key not in names:
            sa = [n for n, e in zip(names, self.entries) if e["view"] == "SA"]
            self.profile_key = sa[len(sa) // 2] if sa else (names[0] if names else None)
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
        momentary_x = MOVE_X + 232                   # right of "Manual correction SA(xx)"
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
        def new_chart(row):
            ch = pv.Chart2D(size=(PROFILE_W, PROFILE_H), loc=(PROFILE_X, PROFILE_Y0 + row * PROFILE_PITCH),
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

        x0 = np.array([0.0, 1.0])
        self._profile_charts, self._metric_charts = [], []
        for k in range(PROFILE_SLOTS):
            ch = new_chart(2 * PROFILE_SLOTS - 1 - k)
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
            pl.add_chart(ch)
            self._profile_charts.append({"chart": ch, "plots": plots, "item": None})

            ch = new_chart(PROFILE_SLOTS - 1 - k)
            # as the profile charts: thin over the whole intersection, thick where it enters the loss, mask bar
            plots = {"zero": ch.line(x0, x0 * 0, color=(0.4, 0.4, 0.4), width=1.0),
                     "full": ch.line(x0, x0 * np.nan, color=PROFILE_LOSS_COLOR, width=1.0),
                     "curve": ch.line(x0, x0 * np.nan, color=PROFILE_LOSS_COLOR, width=2.5),
                     "mask": ch.line(x0, x0 * np.nan, color=MASK_CONTOUR_COLOR, width=2.0),
                     "cursor": ch.line(np.array([0.0, 0.0]), np.array([0.0, 1.0]), color=HOVER_COLOR, width=1.5)}
            plots["cursor"].visible = False
            pl.add_chart(ch)
            self._metric_charts.append({"chart": ch, "plots": plots, "item": None})

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
        # the cine is periodic: stepping past either end wraps around
        pl.add_key_event("Left", lambda: self.set_frame((self.frame - 1) % self.n_frames))
        pl.add_key_event("Right", lambda: self.set_frame((self.frame + 1) % self.n_frames))
        for d in range(10):                     # digit d toggles the d-th SA slice (in list order)
            pl.add_key_event(str(d), lambda d=d: self._toggle_sa_slice(d))
        # pyvista binds Up/Down to camera zoom and b to point picking by default, and
        # add_key_event appends rather than replaces: drop those defaults first
        for k in ("Up", "Down", "b"):
            pl.iren._key_press_event_callbacks.pop(k, None)
        pl.add_key_event("Up", lambda: self.set_snapshot(self.snap_index + 1))
        pl.add_key_event("Down", lambda: self.set_snapshot(self.snap_index - 1))
        pl.add_key_event("Home", lambda: self.set_snapshot(0))
        pl.add_key_event("e", lambda: self.set_snapshot(len(self.snapshots) - 1))
        pl.add_key_event("End", lambda: self.set_snapshot(len(self.snapshots) - 1))
        pl.add_key_event("b", lambda: self.set_snapshot(self.history["best_index"])
                         if self.history["best_index"] is not None else None)

    def _sa_entries(self) -> list[dict]:
        return [e for e in self.entries if e["view"].startswith("SA")]

    def _toggle_sa_slice(self, d: int):
        sa = self._sa_entries()
        if d < len(sa):
            self._toggle_checkbox(sa[d]["name"])

    def _row_label(self, e: dict) -> str:
        """Slice-row label: the slice name and, for the first ten SA slices, its digit key."""
        sa = [x["name"] for x in self._sa_entries()]
        d = sa.index(e["name"]) if e["name"] in sa else None
        return label(e["name"]) + (f"  [{d}]" if d is not None and d < 10 else "")

    def _update_nav_text(self):
        self._nav_text.SetInput(f"subject {self.subject_index + 1}/{len(self.subjects)}  "
                                f"{self.subjects[self.subject_index]}")
        self._run_text.SetInput(f"run {self.run_index + 1}/{len(self.runs)}")     # the name is in the header

    # --- scene -----------------------------------------------------------
    def _build_slices(self):
        for view in VIEWS:
            if view not in self.data:
                continue
            for s in self.data[view]["slices"]:
                nx, ny, _ = s["grey"].shape
                key = (view, int(s["index"]))
                in_history = any(key in snap["affines"] for snap in self.snapshots)
                if not in_history:          # e.g. a LA view without a segmentation: written unaligned
                    print(f"[info] {s['name']}: not part of the alignment (no rows in history.csv); shown at its "
                          f"saved (acquired) pose for every snapshot", file=sys.stderr)
                mesh = slice_mesh(s["affine"], nx, ny)
                mesh.cell_data["rgb"] = self._colors(s)
                actor = self._mesh(
                    mesh, scalars="rgb", rgb=True, lighting=False,
                    show_scalar_bar=False, name=s["name"],
                )
                outline = pv.lines_from_points(voxel_to_world(s["affine"], corner_ijk(nx, ny)), close=True)
                outline.point_data["metric"] = np.full(outline.n_points, np.nan)
                outline_actor = self._mesh(
                    outline, scalars="metric", cmap=OUTLINE_CMAP, clim=(0.0, 1.0),
                    nan_color=OUTLINE_NAN_COLOR, show_scalar_bar=False,
                    line_width=OUTLINE_LINE_WIDTH, name=s["name"] + "_outline",
                )
                outline_actor.mapper.lookup_table = self._lut
                outline_actor.mapper.SetUseLookupTableScalarRange(True)
                self.entries.append({
                    "name": s["name"], "view": view, "key": key, "slice": s,
                    "grid": grid_ijk(nx, ny), "corners": corner_ijk(nx, ny),
                    "mesh": mesh, "outline_mesh": outline,
                    "actor": actor, "outline": outline_actor, "visible": True,
                    "in_history": in_history,
                })
                self._entry_by_key[key] = self.entries[-1]

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

        # SA-SA sampling lines: stored per snapshot; burned into both SA slices of each pair
        if self.overlays.get("sa_lines"):
            self._sa_line_items = {"per_slice": []}
            for e in self.entries:
                if e["view"] != "SA":
                    continue
                item = {"entry": e}
                for name, width, opacity in (("img", SA_LINE_IMG_WIDTH, 0.6), ("used", SA_LINE_USED_WIDTH, 1.0)):
                    mesh = pv.PolyData(np.zeros((2, 3)), lines=np.array([2, 0, 1]))
                    item[name + "_mesh"] = mesh
                    item[name + "_actor"] = self._mesh(mesh, color=SA_LINE_COLOR, line_width=width, opacity=opacity,
                                                       lighting=False, name=f"sa_lines_{name}:{e['name']}")
                    item[name + "_count"] = 0
                self._sa_line_items["per_slice"].append(item)

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
        keys_a = list(zip(df["image_a"].astype(str), df["slice_a"].astype(int)))
        keys_b = list(zip(df["image_b"].astype(str), df["slice_b"].astype(int)))
        # centre of plane b in its voxel coordinates ((X-1)/2, (Y-1)/2, 0), as in make_planes
        centre_b = []
        for kb in keys_b:
            e = self._entry_by_key.get(kb)
            nx, ny = (e["slice"]["grey"].shape[:2] if e is not None else (np.nan, np.nan))
            centre_b.append([(nx - 1) / 2.0, (ny - 1) / 2.0, 0.0])
        rgb = np.array([LINE_COLORS.get(k, (255, 255, 255)) for k in df["kind"]], dtype=np.uint8)
        n = len(df)
        self._line_items = {"keys_a": keys_a, "keys_b": keys_b, "x0": x0, "d": d,
                            "t_used": t_used, "t_used_nominal": t_used.copy(), "t_used_cache": {},
                            "t_img": t_img, "centre_b": np.array(centre_b), "n": n,
                            "per_slice": []}
        # every slice gets its own burned-in copy of the lines it is part of
        for e in self.entries:
            idx = np.array([i for i in range(n) if e["key"] in (keys_a[i], keys_b[i])], dtype=int)
            if not idx.size:
                continue
            k = len(idx)
            item = {"entry": e, "idx": idx}
            for name, width, opacity in (("img", LINE_IMG_WIDTH, 0.45), ("used", LINE_USED_WIDTH, 1.0)):
                mesh = pv.PolyData(np.zeros((4 * k, 3)), lines=burn_cells(k))   # line cells only, no vertices
                mesh.cell_data["rgb"] = np.concatenate([rgb[idx], rgb[idx]])
                item[name + "_mesh"] = mesh
                item[name + "_actor"] = self._mesh(mesh, scalars="rgb", rgb=True, line_width=width, opacity=opacity,
                                                   lighting=False, show_scalar_bar=False,
                                                   name=f"lines_{name}:{e['name']}")
            self._line_items["per_slice"].append(item)

    def _apply_overlays(self):
        """Move contours and lines to the planes' poses at the current snapshot."""
        snap = self.snapshots[self.snap_index]
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
        self._apply_sa_lines(snap)
        self._apply_radius(snap)
        li = self._line_items
        if li is None:
            return
        x0, d, li["t_used"] = self._line_geometry(snap)
        li["cur_x0"], li["cur_d"] = x0, d
        for name in ("used", "img"):
            t = li["t_" + name]                                 # (N, 2)
            t = np.where(np.isfinite(t), t, li["t_img"][:, :1])  # no loss points: zero-length segment
            seg = np.stack([x0 + t[:, :1] * d, x0 + t[:, 1:] * d], axis=1)   # (N, 2, 3)
            for item in li["per_slice"]:
                A = self._affine_for(item["entry"], snap)
                item[name + "_mesh"].points = burn_points(seg[item["idx"]], A)

    def _line_geometry(self, snap: dict):
        """(x0, d, t_used) of every SA-LA / LA-LA line at `snap` (manual offsets included): the intersection
        of its two posed planes, anchored at the moved centre of plane b, and the loss range of the
        snapshot's mask dilation."""
        li = self._line_items
        x0, d = li["x0"].copy(), li["d"].copy()                # acquired geometry (fallback)
        for i in range(li["n"]):
            Aa = self._pose(li["keys_a"][i], snap)
            Ab = self._pose(li["keys_b"][i], snap)
            if Aa is None or Ab is None or not np.isfinite(li["centre_b"][i]).all():
                continue
            na = np.cross(Aa[:3, 0], Aa[:3, 1]); na /= np.linalg.norm(na)
            nb = np.cross(Ab[:3, 0], Ab[:3, 1]); nb /= np.linalg.norm(nb)
            di = np.cross(na, nb)
            nd = np.linalg.norm(di)
            if nd < 1e-6:
                continue                                        # planes (nearly) parallel at this pose
            di /= nd
            cb = (Ab @ np.append(li["centre_b"][i], 1.0))[:3]
            rhs = np.array([na @ Aa[:3, 3], nb @ Ab[:3, 3], di @ cb])
            x0[i] = np.linalg.solve(np.stack([na, nb, di]), rhs)
            d[i] = di
        return x0, d, self._stage_t_used(self._dilation(snap["step"]))

    def _apply_sa_lines(self, snap: dict):
        """Burn the stored SA-SA segments of this snapshot into both slices of their pair
        (none where the slices do not cross)."""
        si = self._sa_line_items
        if si is None:
            return
        segs = self._sa_segments(snap)
        for item in si["per_slice"]:
            sl = item["entry"]["key"][1]
            A = self._affine_for(item["entry"], snap)
            for name in ("img", "used"):
                e = segs.get(name, np.zeros((0, 2, 3)))
                pairs = segs.get(name + "_pairs", np.zeros((0, 2), int))
                mine = e[(pairs == sl).any(1)] if len(e) else e
                k = len(mine)
                item[name + "_count"] = k
                new = (pv.PolyData(burn_points(mine, A), lines=burn_cells(k)) if k
                       else pv.PolyData(np.zeros((2, 3)), lines=np.array([2, 0, 1])))   # placeholder, hidden
                item[name + "_mesh"].copy_from(new)
        self._apply_sa_line_visibility()

    def _sa_segments(self, snap: dict) -> dict:
        """SA-SA segments of this snapshot: as stored, except pairs with a manually moved slice, whose
        crossing is recomputed (inside both images; the loss part within the long-axis radius)."""
        segs = (self.overlays.get("sa_lines") or {}).get((snap["step"], bool(snap["is_best"])), {})
        moved = {k[1] for k, off in self.offsets.items() if k[0] == "SA" and np.any(off)}
        if not moved:
            return segs
        sa = {e["key"][1]: e for e in self.entries if e["view"] == "SA"}
        axis = (self.overlays.get("long_axis") or {}).get((snap["step"], bool(snap["is_best"])))
        out = {}
        for name in ("img", "used"):
            e, pr = segs.get(name, np.zeros((0, 2, 3))), segs.get(name + "_pairs", np.zeros((0, 2), int))
            keep = ~np.isin(pr, list(moved)).any(1) if len(pr) else np.zeros(0, bool)
            out[name], out[name + "_pairs"] = list(e[keep]), list(pr[keep])
        for z in sorted(sa):
            if z + 1 not in sa or not ({z, z + 1} & moved):
                continue
            ea, eb = sa[z], sa[z + 1]
            cs = crossing_segment(self._pose(ea["key"], snap), ea["slice"]["grey"].shape,
                                  self._pose(eb["key"], snap), eb["slice"]["grey"].shape)
            if cs is None:
                continue
            x0, d, lo, hi = cs
            out["img"].append(np.stack([x0 + lo * d, x0 + hi * d])); out["img_pairs"].append([z, z + 1])
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
                        out["used"].append(np.stack([x0 + u1 * d, x0 + u2 * d])); out["used_pairs"].append([z, z + 1])
        for name in ("img", "used"):
            out[name] = np.array(out[name]).reshape(-1, 2, 3)
            out[name + "_pairs"] = np.array(out[name + "_pairs"], dtype=int).reshape(-1, 2)
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
        self._move_text.SetInput(f"Manual correction {label(self.profile_key) if e else '-'}")
        self._reset_text.SetInput("reset all slices" + (f"  ({n_moved} moved)" if n_moved else ""))
        for k, (name, unit) in enumerate(MOVE_PARAMS):
            self._move_values[k].SetInput(f"{off[k]:+.0f} {unit}")

    def _apply_radius(self, snap: dict):
        """Intersect the radius cylinder around this snapshot's long axis with every SA slice at its pose:
        each cylinder generator p + r (cos f e1 + sin f e2) + s ell meets the slice plane at one point."""
        if not self._radius_items:
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
        si = self._sa_line_items
        if si is None:
            return
        for item in si["per_slice"]:
            for name in ("img", "used"):
                item[name + "_actor"].SetVisibility(self.show_sa_lines and item["entry"]["visible"]
                                                    and item[name + "_count"] > 0)

    def _apply_line_visibility(self):
        if self._line_items is None:
            return
        for item in self._line_items["per_slice"]:
            for name in ("img", "used"):
                item[name + "_actor"].SetVisibility(self.show_lines and item["entry"]["visible"])

    # --- intersection intensity profiles ---------------------------------------
    def set_profile(self, name: str):
        if name not in self._entry_by_name():
            return
        self.profile_key = name
        for n, w in self._profile_radios.items():
            w.GetRepresentation().SetState(int(n == name))
        self._log(f"profiles / move: {label(name)}")
        self._update_move_text()
        self._update_profiles()
        self.pl.render()

    def step_profile(self, step: int):
        names = [e["name"] for e in self.entries]
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
        alignment script computes it; None without a segmentation."""
        if "heart_dist" not in e:
            seg = e["slice"]["seg"]
            if seg is None or not (seg > 0).any():
                e["heart_dist"] = None
            else:
                A = e["slice"]["affine"]
                sp = (np.linalg.norm(A[:3, 0]), np.linalg.norm(A[:3, 1]))
                e["heart_dist"] = distance_transform_edt(~(seg > 0).any(-1), sampling=sp)
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

    def _stage_t_used(self, dil: float | None) -> np.ndarray:
        """Loss range (t_lo, t_hi) of every SA-LA / LA-LA line at dilation `dil`: as intersection_line, the first
        and last point (--line-step-mm) at the acquired geometry inside both images and inside the dilated mask of
        either LA plane. The logged (nominal) ranges at the nominal dilation or without the segmentations."""
        li = self._line_items
        cfg = getattr(self, "config", None) or {}
        if dil is None or cfg.get("mask_dilation_mm") is None or abs(dil - float(cfg["mask_dilation_mm"])) < 1e-9:
            return li["t_used_nominal"]
        if dil in li["t_used_cache"]:
            return li["t_used_cache"][dil]
        snap0 = self.snapshots[0]
        step_mm = float(cfg.get("line_step_mm", 0.5))
        out = li["t_used_nominal"].copy()
        for i in range(li["n"]):
            lo, hi = li["t_img"][i]
            if not (np.isfinite(lo) and np.isfinite(hi)):
                continue
            t = np.arange(lo, hi + 1e-6, step_mm)
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
                out[i] = (t[inside].min(), t[inside].max()) if inside.any() else (np.nan, np.nan)
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

    # --- profiles as the loss sees them -----------------------------------------------
    def _loss_state(self, step: int) -> tuple[float, float]:
        """(PSF scale, in-plane blur sigma in mm) the optimizer used at `step` (motion_gradient.align)."""
        cfg = self.config or {}
        steps = max(1, int(cfg.get("steps", 1)))
        psf_s0 = 1.0 if cfg.get("no_psf") else float(cfg.get("psf_scale_start", 1.0) or 1.0)
        psf_steps = int(round(float(cfg.get("psf_anneal_frac", 0.6)) * steps)) if psf_s0 > 1.0 else 0
        scale = psf_s0 + (1.0 - psf_s0) * step / psf_steps if step < psf_steps else 1.0
        b0 = float(cfg.get("blur_sigma", 0.0) or 0.0)
        b_frac = float(cfg.get("blur_frac", 0.6))
        if b0 > 0 and b_frac <= 0:                       # --blur-frac 0: constant blur for the whole run
            sigma = b0
        else:
            b_steps = int(round(b_frac * steps)) if b0 > 0 else 0
            sigma = b0 * max(0.0, 1.0 - step / b_steps) if step < b_steps else 0.0
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

    def _blurred(self, e: dict, sigma: float) -> np.ndarray:
        """Raw intensities of the current frame, in-plane Gaussian blurred with sigma mm (as LineSampler.set_blur)."""
        key = (e["name"], self.frame, round(sigma, 4))
        out = self._blur_cache.get(key)
        if out is None:
            raw = e["slice"]["raw"][:, :, self.frame].astype(np.float32)
            if sigma > 1e-6:
                A = e["slice"]["affine"]
                sp = (np.linalg.norm(A[:3, 0]), np.linalg.norm(A[:3, 1]))
                raw = gaussian_filter(raw, (max(sigma / sp[0], 1e-3), max(sigma / sp[1], 1e-3)), mode="nearest",
                                      truncate=3.0)
            if len(self._blur_cache) > 64:
                self._blur_cache.clear()
            self._blur_cache[key] = out = raw
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
        prof = np.zeros(len(pts))
        for uk, wk in zip(u, w):
            P = pts + uk * step * m
            vox = (Ai @ np.column_stack([P, np.ones(len(P))]).T)[:2]
            prof += wk * map_coordinates(img, vox, order=1, mode="constant", cval=0.0)   # zeros padding, as grid_sample
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
        """The intersections of the picked slice: (other slice name, start, unit dir, t_img, t_used)."""
        e = self._entry_by_name().get(self.profile_key)
        if e is None:
            return []
        out = []
        li = self._line_items
        if li is not None:
            x0, d, t_used = self._line_geometry(snap)
            for i in range(li["n"]):
                ka, kb = li["keys_a"][i], li["keys_b"][i]
                if e["key"] not in (ka, kb):
                    continue
                other = kb if ka == e["key"] else ka
                if e["view"] != "SA" and other[0] == "SA":
                    continue                              # LA views: only their LA-LA pairs
                oe = self._entry_by_key.get(other)
                if oe is None:
                    continue
                out.append(dict(other=oe, x0=x0[i], d=d[i], t_img=li["t_img"][i], t_used=t_used[i]))
            out.sort(key=lambda r: VIEWS.index(r["other"]["view"]))
        if e["view"] == "SA":
            sl = e["key"][1]
            names = {x["key"][1]: x for x in self.entries if x["view"] == "SA"}
            nb = sl + 1 if sl + 1 in names else sl - 1
            if nb in names:
                item = dict(other=names[nb], x0=None, sa_sa=True)
                segs = self._sa_segments(snap)
                pairs = segs.get("img_pairs", np.zeros((0, 2), int))
                hit = np.flatnonzero(((pairs == sl).any(1)) & ((pairs == nb).any(1))) if len(pairs) else []
                if len(hit):
                    a, b = segs["img"][hit[0]]
                    length = float(np.linalg.norm(b - a))
                    if length > 0:
                        item.update(x0=a, d=(b - a) / length, t_img=np.array([0.0, length]),
                                    t_used=np.array([np.nan, np.nan]))
                        upairs = segs.get("used_pairs", np.zeros((0, 2), int))
                        uh = np.flatnonzero(((upairs == sl).any(1)) & ((upairs == nb).any(1))) if len(upairs) else []
                        if len(uh):
                            item["t_used"] = np.sort((segs["used"][uh[0]] - a) @ item["d"])
                out.append(item)
        return out[:PROFILE_SLOTS]

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

    def _chart_data(self, snap: dict) -> list[dict]:
        """Per chart slot at `snap` (current frame): the intersection item and, where the slices cross,
        its profiles (t, ya, yb, used, mode_txt) and the selected metric (mname, mval, ct, curve)."""
        e = self._entry_by_name().get(self.profile_key)
        out = []
        for it in self._profile_items(snap):
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

    def _metric_ranges(self, data: list[dict]) -> list:
        """Per chart slot: (min, max) of the full metric curve over every history snapshot, so the metric
        charts keep one y scale while scrubbing the history; None where a slot never has a curve.
        Computed at the frame shown when it is (re)built -- slice, metric, profile mode or manual
        offsets changed (~0.6 s for 50 snapshots) -- and only widened afterwards when another frame's
        curve (`data`, the shown one) leaves it, so stepping through the cine stays fast."""
        key = (self.run_index, self.subject_index, self.profile_key, self.metric,
               self.show_psf and self.config is not None,
               tuple(sorted((k, tuple(np.round(v, 6))) for k, v in self.offsets.items())))
        cache = getattr(self, "_range_cache", None)
        if cache is None or cache[0] != key:
            lo = np.full(PROFILE_SLOTS, np.inf)
            hi = np.full(PROFILE_SLOTS, -np.inf)
            for snap in self.snapshots:
                for k, c in enumerate(self._chart_data(snap)):
                    cv = c.get("full")
                    if cv is not None and np.isfinite(cv).any():
                        lo[k] = min(lo[k], float(np.nanmin(cv)))
                        hi[k] = max(hi[k], float(np.nanmax(cv)))
            self._range_cache = cache = (key, lo, hi)
        _, lo, hi = cache
        for k, c in enumerate(data[:PROFILE_SLOTS]):
            cv = c.get("full")
            if cv is not None and np.isfinite(cv).any():
                lo[k] = min(lo[k], float(np.nanmin(cv)))
                hi[k] = max(hi[k], float(np.nanmax(cv)))
        return [(float(l), float(h)) if np.isfinite(l) else None for l, h in zip(lo, hi)]

    def _update_profiles(self):
        charts = getattr(self, "_profile_charts", [])
        if not charts or not self.entries:
            return
        snap = self.snapshots[self.snap_index]
        data = self._chart_data(snap) if self.show_profiles or self.show_metric_profiles else []
        ranges = self._metric_ranges(data) if data and self.show_metric_profiles else [None] * PROFILE_SLOTS
        e = self._entry_by_name().get(self.profile_key)
        for k, (slot, mslot) in enumerate(zip(charts, self._metric_charts)):
            ch, pl_ = slot["chart"], slot["plots"]
            slot["item"] = mslot["item"] = None
            if k >= len(data):
                ch.visible = mslot["chart"].visible = False
                continue
            c = data[k]
            it = c["it"]
            ch.visible, mslot["chart"].visible = self.show_profiles, self.show_metric_profiles
            names = f"{label(e['name'])} \u00d7 {label(it['other']['name'])}"
            if it["x0"] is None:
                ch.title = f"{names}: no crossing"
                for p_ in pl_.values():
                    p_.update(np.array([0.0, 1.0]), np.array([np.nan, np.nan]))
                self._fill_metric_chart(mslot, f"{names}: no crossing", None, None, None, None, None, None)
                continue
            t, ya, yb, used, mname, mval = c["t"], c["ya"], c["yb"], c["used"], c["mname"], c["mval"]
            ch.title = (f"{names}   {mname} {mval:.2f}{c['mode_txt']}" if mval is not None
                        else f"{names}   {mname} -{c['mode_txt']}")
            for side, y in (("a", ya), ("b", yb)):
                pl_[side + "_img"].update(t, y)
                pl_[side + "_used"].update(t, np.where(used, y, np.nan))
            masks = [m for m in (self._in_mask(e, snap, c["pts"]), self._in_mask(it["other"], snap, c["pts"]))
                     if m is not None]
            in_mask = np.logical_or.reduce(masks) if masks else np.zeros(len(t), bool)
            pl_["mask"].update(t, np.where(in_mask, float(PROFILE_MASK_Y), np.nan))
            xr = [float(t[0]), float(t[-1]) if len(t) > 1 else float(t[0]) + 1.0]
            ch.x_range = xr
            ch.y_range = [0, 255]
            item = dict(e=e, other=it["other"], x0=it["x0"], d=it["d"], t=t, ya=ya, yb=yb, names=names,
                        title=ch.title, ct=c["ct"], curve=c["full"], mname=mname)
            slot["item"] = mslot["item"] = item
            item["mtitle"], item["mrange"] = self._fill_metric_chart(
                mslot, f"{names}   {mname} per point", c["ct"], c["curve"], c["full"], xr, ranges[k],
                (t, in_mask))
        self._draw_hover()

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
        if dmin < 0 < dmax:                                     # 0, plus the extremes not crowding it
            ticks = tuple(v for v in (dmin, 0.0, dmax) if v == 0.0 or abs(v) > 0.2 * span)
        else:
            ticks = (dmin, (dmin + dmax) / 2, dmax)
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
            for slot in slots:
                ch, it = slot["chart"], slot["item"]
                if it is None or not ch.visible:
                    continue
                (cx, cy), (cw, chh) = ch.loc, ch.size
                if not (cx * w <= x <= (cx + cw) * w and cy * h <= y <= (cy + chh) * h):
                    continue
                p1, p2 = ch.x_axis.GetPoint1(), ch.x_axis.GetPoint2()
                if p2[0] - p1[0] >= 1 and p1[0] <= x <= p2[0]:
                    lo, hi = ch.x_range
                    hover = (k, lo + (x - p1[0]) / (p2[0] - p1[0]) * (hi - lo))
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
        for e in self.entries:
            A = self._affine_for(e, snap)
            e["mesh"].points = voxel_to_world(A, e["grid"])
            e["outline_mesh"].points = voxel_to_world(A, e["corners"])
        self._apply_overlays()
        self._apply_metric_values()
        self._update_knob_labels()
        self._update_header()
        self._update_profiles()

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
        sa = [e for e in self.entries if e["view"] == "SA"] or self.entries
        A = self._affine_for(sa[len(sa) // 2], snap)
        u, v, n = (A[:3, c] / np.linalg.norm(A[:3, c]) for c in range(3))
        self.pl.view_vector(0.45 * n + 1.0 * u + 0.8 * v, viewup=-n)
        self.pl.reset_camera()

    def _display_sigma(self) -> float:
        """In-plane blur (mm) of the slice images: the optimizer's blur at the shown step with the blur toggle on."""
        if not (self.show_blur and getattr(self, "config", None) is not None and getattr(self, "snapshots", None)):
            return 0.0
        return self._loss_state(self.snapshots[self.snap_index]["step"])[1]

    def _colors(self, s: dict) -> np.ndarray:
        sigma = self._display_sigma()
        if sigma > 1e-6:
            raw = self._blurred({"name": s["name"], "slice": s}, sigma)
            grey = (np.clip((raw - s["lo"]) / (s["hi"] - s["lo"]), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        else:
            grey = s["grey"][:, :, self.frame]
        seg = None if s["seg"] is None else s["seg"][:, :, self.frame]
        return cell_colors(grey, seg, self.show_seg, self.seg_alpha)

    def _recolor(self):
        self._colors_sigma = self._display_sigma()
        for e in self.entries:
            e["mesh"].cell_data["rgb"] = self._colors(e["slice"])

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
        for e in self.entries:
            e["actor"].SetVisibility(e["visible"])
            e["outline"].SetVisibility(e["visible"] and self.show_outlines)
        for m in self._mask_items:                 # a contour follows its plane's visibility
            e = self._entry_by_key.get(m["key"])
            m["actor"].SetVisibility(self.show_mask and (e is None or e["visible"]))
        self._apply_line_visibility()
        self._apply_sa_line_visibility()
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
            group = np.array([vals[e["name"]] for e in self.entries if e["view"].startswith(prefix)], dtype=float)
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
        add_row("Outlines  [o]", self._on_outline_toggle, True, "outlines")
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
            add_row("Image blurring  [g]", self._on_blur_toggle, self.show_blur, "blur")
        y -= 8
        if "SA" in self.data:
            add_row("All SA  [a]", lambda v: self._set_group("SA", v), True, "all_sa")
        if any(v.startswith("LA") for v in self.data):
            add_row("All LA  [l]", lambda v: self._set_group("LA", v), True, "all_la")
        y -= 8
        # column headers above the slice rows: the outline metric (value column) and "profile"
        self._text("profile", position=(PROFILE_RADIO_X + 7, y + ROW_PITCH - 4), font_size=7,
                   color="#bbbbbb").GetTextProperty().SetJustificationToCentered()
        self._value_header = self._text("", position=(PANEL_X_VAL, y + ROW_PITCH - 4), font_size=7,
                                        color="#bbbbbb")
        for e in self.entries:
            self._profile_radios[e["name"]] = pl.add_radio_button_widget(
                lambda n=e["name"]: self.set_profile(n), "profile", value=(e["name"] == self.profile_key),
                position=(PROFILE_RADIO_X, y + 3), size=14, border_size=3,
                color_on="#ffa726", color_off="#555555", background_color="#303030")
            self._row_value[e["name"]] = self._text("", position=(PANEL_X_VAL, y + 2), font_size=9, color="white")
            add_row(self._row_label(e), self._make_slice_cb(e), True, e["name"])

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


    # --- callbacks -------------------------------------------------------
    def _make_slice_cb(self, e: dict):
        def cb(value: bool):
            e["visible"] = bool(value)
            self._set_checkbox_state(e["name"], e["visible"])
            self._apply_visibility()
            self.pl.render()
        return cb

    def _set_group(self, prefix: str, value: bool):
        for e in self.entries:
            if e["view"].startswith(prefix):
                e["visible"] = bool(value)
                self._set_checkbox_state(e["name"], e["visible"])
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
        self._apply_visibility()
        self.pl.render()

    def _on_radius_toggle(self, value: bool):
        self.show_radius = bool(value)
        self._apply_radius_visibility()
        self.pl.render()

    def _on_sa_lines_toggle(self, value: bool):
        self.show_sa_lines = bool(value)
        self._apply_sa_line_visibility()
        self.pl.render()

    def _on_lines_toggle(self, value: bool):
        self.show_lines = bool(value)
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


def _first_instance(subject_dir: Path) -> int | None:
    """The first Instance_<n> folder of a subject (runs made before the single-slice layout), None if there is none."""
    for p in sorted(subject_dir.glob("Instance_*")):
        try:
            return int(p.name.split("_")[1])
        except (IndexError, ValueError):
            continue
    return None


def load_subject_run(run_dir: Path, subject: str, instance: int | None = None, cache: dict | None = None) -> dict:
    """Everything the viewer needs for one subject of a run. The images are read from <run>/<subject>/ (current
    layout) or, for older runs, from <run>/<subject>/Instance_<n>/. `cache` (a dict kept by the caller) holds the
    last subject's images: the same subject from another run then reads only the NIfTI headers (reuse_images)."""
    subject_dir = run_dir / subject
    if instance is None:
        instance = _first_instance(subject_dir)
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
    cfg_path = run_dir / "overall_metrics" / "run_config.json"
    config = json.loads(cfg_path.read_text()) if cfg_path.exists() else None
    if config is None:
        print(f"[warn] {cfg_path} missing: profiles can only show raw intensities", file=sys.stderr)
    title = f"UKBB {subject}" + (f" / Instance_{instance}" if instance is not None else "")
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
    ap.add_argument("--instance", type=int, default=None,
                    help="older runs only: imaging instance folder (default: the first Instance_<n> found, else none)")
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

    run_dir = args.run_dir or find_run_dir(args.align_root, args.subject, args.run)
    if run_dir is None:
        sys.exit(f"No run with aligned subjects found under {args.align_root}"
                 + (f" for subject {args.subject}" if args.subject else ""))
    run_dir = Path(run_dir)

    subjects = _subjects_in_run(run_dir)
    if not subjects:
        sys.exit(f"No aligned subject in {run_dir}")
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
        return load_subject_run(run, subject, args.instance, image_cache)

    start = None
    if args.start == "end":
        start = 10 ** 9                      # clipped to the last snapshot
    elif args.start != "best":
        start = int(args.start)

    if args.screenshot is not None:
        pv.OFF_SCREEN = True

    viewer = SliceViewer(
        runs, _subjects_in_run, loader, run_index=run_index,
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
