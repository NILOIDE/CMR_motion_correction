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
    Radio buttons       which per-plane metric of history.csv colours the slice outlines:
                        NCC, NMI, Dice or Hausdorff (red = bad, green = good, grey = no value)
    1 / 2 / 3 / 4       select NCC / NMI / Dice / Hausdorff for the outline colour
    Sliders "colour min" / "colour max" (above the colour bar)
                        clamp the colour range: values below the min are full red / full
                        green (whichever is the "bad" end), values above the max the other
                        extreme.  Defaults: the subject's min and max over the history.
    Up / Down           next / previous history snapshot
    0 / b / e           jump to step 0 / the best pose / the final snapshot
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
    Buttons "next subject" / "previous subject" (bottom left), keys ] and [
                        (also PageDown / PageUp): load the next / previous subject of the run
                        (alphabetical) into the same window; toggles, metric and frame are kept
    Buttons "next run (older)" / "previous run (newer)" (above them), keys } and {
                        step through the runs of the same directory (--align-root, or the
                        parent of --run-dir), newest first; the current subject is kept when
                        the other run has it, else the same position in its subject list
    p / , / .           intersection intensity profiles (right): toggle / previous / next slice;
                        hovering a chart marks that position on the line in 3D (red tick in both
                        slices) and shows both grey values in the chart title;
                        the small radio next to each slice row picks it. An SA slice shows its
                        3 SA-LA lines and its SA-SA crossing with the next slice, an LA view its
                        2 LA-LA lines. Curves: displayed grey value of both slices along the line
                        at the shown step and frame (thin: inside both images, thick: in the
                        loss); yellow bar along the bottom: inside the LA dilated mask (either
                        LA slice of the line, at its current pose); NCC over the loss part,
                        current frame only
    Manual pose (block left of the stack): [-] / [+] shift the profile-selected slice by 1 mm along
                        its own u / v / normal axes or rotate it by 1 deg about them (about the slice
                        centre), on top of the shown step's pose; lines, contours and profiles follow
                        (outline colours stay the logged metrics). "reset all slices" clears every
                        offset; offsets are also cleared when switching subject / run
    r                   reset camera
    q                   quit
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pyvista as pv
from matplotlib.path import Path as PolyPath
from scipy.ndimage import map_coordinates

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
OUTLINE_METRICS = {
    "ncc": dict(label="NCC", col="ncc", higher_better=True, fmt="{:.3f}",
                bounds=(-1.0, 1.0), slider_fmt="%.2f"),
    "nmi": dict(label="NMI", col="nmi", higher_better=True, fmt="{:.3f}",
                bounds=(0.0, 1.0), slider_fmt="%.2f"),
    "dice": dict(label="Dice", col="seg_dice", higher_better=True, fmt="{:.3f}",
                 bounds=(0.0, 1.0), slider_fmt="%.2f"),
    "hd": dict(label="Hausdorff", col="hd_mm", higher_better=False, fmt="{:.1f} mm",
               bounds=(0.0, 50.0), slider_fmt="%.1f mm"),
}
RANGE_MIN_GAP = 1e-3   # colour min is kept at least this far below colour max

# Colour bar block (normalised viewport coordinates): the bar strip with the
# "colour max" slider right above it and the "colour min" slider right below it.
BAR_X0, BAR_X1 = 0.70, 0.90          # horizontal extent of strip and sliders
BAR_Y0, BAR_H = 0.31, 0.035          # strip bottom and height
SLIDER_HI_Y = BAR_Y0 + BAR_H + 0.018  # max slider
SLIDER_LO_Y = BAR_Y0 - 0.018          # min slider
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
# its SA-SA crossing with the next slice; LA view: its 2 LA-LA lines)
PROFILE_SLOTS = 4
PROFILE_X, PROFILE_W = 0.735, 0.25
PROFILE_Y0, PROFILE_H, PROFILE_PITCH = 0.405, 0.083, 0.088
PROFILE_COLORS = ("#e0e0e0", "#ffb74d")    # picked slice, the other slice of the line
PROFILE_STEP_MM = 0.5
PROFILE_RADIO_X = 215                      # radio column right of the slice rows
PROFILE_MASK_Y = 10                        # height (grey units) of the in-mask bar along the chart bottom
HOVER_COLOR = "#ff1744"                    # hover cursor in the charts and its tick marks in 3D
HOVER_TICK_MM = 12.0                       # length of the tick across the line, in each slice
HOVER_WIDTH = 4

# Manual pose controls for the profile-selected slice (applied on top of the shown step's pose)
MOVE_PARAMS = [("shift u", "mm"), ("shift v", "mm"), ("shift n", "mm"),
               ("rot u", "deg"), ("rot v", "deg"), ("rot n", "deg")]
MOVE_STEP = (1.0, 1.0)                     # mm per click, degrees per click
MOVE_X, MOVE_TOP = 272, 430                # block position (pixels from the bottom left)

# Panel layout (pixels from the bottom-left corner of the window)
PANEL_X_BOX = 12
PANEL_X_TXT = 42
ROW_PITCH = 26


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
            })
        data[view] = {"mode": mode, "slices": slices}

    if not data:
        raise RuntimeError(f"No views found in {instance_dir}")
    return data


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
            metrics[key] = {m: float(getattr(r, m)) for m in ("ncc", "nmi", "seg_dice", "hd_mm") if m in df.columns}
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
        self._tex_cache: dict[str, tuple] = {}
        self._range_sliders: dict[str, object] = {}
        self._range_text: dict[str, object] = {}
        self._frame_slider = None
        self._hist_slider = None
        self._scalar_bar = None

    def _text(self, *args, **kwargs):
        actor = self.pl.add_text(*args, **kwargs)
        self._scene_actors.append(actor)
        return actor

    def _mesh(self, *args, **kwargs):
        actor = self.pl.add_mesh(*args, **kwargs)
        self._scene_actors.append(actor)
        return actor

    def _teardown(self):
        """Remove every actor and widget of the current subject (the navigation UI stays)."""
        pl = self.pl
        wh = getattr(pl, "widgets", pl)          # pyvista >= 0.45 keeps the widget lists here
        for w in list(self._widgets.values()) + list(self._radios.values()) + list(self._profile_radios.values()) + \
                list(self._range_sliders.values()) + [self._frame_slider, self._hist_slider]:
            if w is None:
                continue
            w.Off()
            for lst in (getattr(wh, "button_widgets", []), getattr(wh, "slider_widgets", [])):
                if w in lst:
                    lst.remove(w)
        rd = getattr(wh, "radio_button_widget_dict", None)
        if isinstance(rd, dict):
            rd.pop("outline_metric", None)
            rd.pop("profile", None)
        rt = getattr(wh, "radio_button_title_dict", None)
        if isinstance(rt, dict):
            rt.pop("outline_metric", None)
        if "outline colour" in pl.scalar_bars:
            pl.remove_scalar_bar("outline colour", render=False)
        for a in self._scene_actors:
            pl.remove_actor(a, render=False)
        self._scene_actors = []
        self._reset_scene_state()

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
        if self._built:
            self._teardown()
        self.offsets = {}
        self._hover = None
        self.run_index = run_index
        self.subjects = list(subjects)
        self.subject_index = index
        self.data = loaded["data"]
        self.history = loaded["history"]
        self.overlays = loaded.get("overlays") or {"lines": None, "contours": {}}
        self.overlays.setdefault("sa_lines", None)
        self.overlays.setdefault("long_axis", None)
        self.title = loaded.get("title", subject)
        self.run_name = loaded.get("run_name", Path(run).name)
        self.snapshots = self.history["snapshots"]
        self.n_steps = self.history["n_steps"]
        self.metric_range = self.history.get("metric_range", {})
        self.has_metrics = bool(self.history.get("available", True))   # False: aligned without --logging
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

        def momentary(label, key, cb, y):
            holder = {}

            def _cb(_state):
                holder["w"].GetRepresentation().SetState(0)    # a push button, not a toggle
                cb()
            holder["w"] = pl.add_checkbox_button_widget(
                _cb, value=False, position=(PANEL_X_BOX, y), size=20, border_size=1,
                color_on="#ffa726", color_off="#4a4a4a", background_color="#303030")
            pl.add_text(label, position=(PANEL_X_TXT, y + 2), font_size=9, color="white")
            self._nav_widgets[key] = holder["w"]

        # Navigation in the bottom-left corner: run block above the subject block,
        # each a label followed by next / previous push buttons
        self._nav_widgets: dict[str, object] = {}
        ny = 16 + 2 * ROW_PITCH
        self._nav_text = pl.add_text("", position=(PANEL_X_BOX, ny + 2), font_size=9, color="#ffa726")
        momentary("next subject  ]", "next", lambda: self.next_subject(+1), ny - ROW_PITCH)
        momentary("previous subject  [", "prev", lambda: self.next_subject(-1), ny - 2 * ROW_PITCH)
        ry = ny + 3 * ROW_PITCH + 6
        self._run_text = pl.add_text("", position=(PANEL_X_BOX, ry + 2), font_size=9, color="#80cbc4")
        momentary("next run (older)  }", "next_run", lambda: self.next_run(+1), ry - ROW_PITCH)
        momentary("previous run (newer)  {", "prev_run", lambda: self.next_run(-1), ry - 2 * ROW_PITCH)
        self._panel_top = y                           # per-subject rows start at the top

        # Manual pose block (acts on the profile-selected slice): per parameter  [-] [+] value
        self._move_text = pl.add_text("", position=(MOVE_X, MOVE_TOP + 2), font_size=9, color="#ffa726")
        self._move_values = []
        my = MOVE_TOP
        for k, (name, unit) in enumerate(MOVE_PARAMS):
            my -= 24
            pl.add_text(name, position=(MOVE_X, my + 3), font_size=8, color="#bbbbbb")
            for j, (sign, glyph) in enumerate(((-1, "-"), (+1, "+"))):
                bx = MOVE_X + 56 + 24 * j
                holder = {}

                def _cb(_state, h=holder, k=k, sign=sign):
                    h["w"].GetRepresentation().SetState(0)
                    self.move_selected(k, sign)
                holder["w"] = pl.add_checkbox_button_widget(
                    _cb, value=False, position=(bx, my), size=18, border_size=1,
                    color_on="#ffa726", color_off="#4a4a4a", background_color="#303030")
                pl.add_text(glyph, position=(bx + 5, my + 1), font_size=9, color="white")
                self._nav_widgets[f"move{k}{glyph}"] = holder["w"]
            self._move_values.append(pl.add_text("", position=(MOVE_X + 108, my + 3), font_size=8,
                                                 color="white"))
        my -= 28
        momentary_x = MOVE_X
        holder = {}

        def _reset(_state, h=holder):
            h["w"].GetRepresentation().SetState(0)
            self.reset_offsets()
        holder["w"] = pl.add_checkbox_button_widget(
            _reset, value=False, position=(momentary_x, my), size=18, border_size=1,
            color_on="#ffa726", color_off="#4a4a4a", background_color="#303030")
        pl.add_text("reset all slices", position=(momentary_x + 26, my + 2), font_size=8, color="white")
        self._nav_widgets["move_reset"] = holder["w"]

        # intersection intensity charts (persistent; filled per subject / step / frame)
        self._profile_charts = []
        for k in range(PROFILE_SLOTS):
            ch = pv.Chart2D(size=(PROFILE_W, PROFILE_H),
                            loc=(PROFILE_X, PROFILE_Y0 + (PROFILE_SLOTS - 1 - k) * PROFILE_PITCH),
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
            ch.y_axis.tick_locations = (0, 128, 255)
            ch.y_axis.tick_labels = ("0", "128", "255")
            x0 = np.array([0.0, 1.0])
            plots = {}
            for side, col in zip(("a", "b"), PROFILE_COLORS):
                plots[side + "_img"] = ch.line(x0, x0 * 0, color=col, width=1.0)
                plots[side + "_used"] = ch.line(x0, x0 * 0, color=col, width=2.5)
            plots["mask"] = ch.line(x0, x0 * np.nan, color=MASK_CONTOUR_COLOR, width=2.0)
            plots["cursor"] = ch.line(np.array([0.0, 0.0]), np.array([0.0, 255.0]), color=HOVER_COLOR, width=1.5)
            plots["cursor"].visible = False
            ch.y_range = [0, 255]
            ch.visible = False
            pl.add_chart(ch)
            self._profile_charts.append({"chart": ch, "plots": plots, "item": None})

        # hover: a tick across the intersection line at the hovered position, burned into both slices
        self._hover = None                                   # (chart slot, t along the line) or None
        self._hover_mesh = pv.PolyData(np.zeros((8, 3)), lines=burn_cells(2))
        self._hover_actor = pl.add_mesh(self._hover_mesh, color=HOVER_COLOR, line_width=HOVER_WIDTH,
                                        lighting=False, name="profile_hover")
        self._hover_actor.SetVisibility(False)
        pl.iren.add_observer("MouseMoveEvent", self._on_mouse_move)

        # orientation axes to the right of the navigation block, left of the sliders
        pl.add_axes(xlabel="X (mm)", ylabel="Y (mm)", zlabel="Z (mm)", viewport=(0.15, 0.0, 0.29, 0.2))

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
        pl.add_key_event("period", lambda: self.step_profile(+1))
        pl.add_key_event("comma", lambda: self.step_profile(-1))
        pl.add_key_event("a", lambda: self._toggle_checkbox("all_sa"))
        pl.add_key_event("l", lambda: self._toggle_checkbox("all_la"))
        # the cine is periodic: stepping past either end wraps around
        pl.add_key_event("Left", lambda: self.set_frame((self.frame - 1) % self.n_frames))
        pl.add_key_event("Right", lambda: self.set_frame((self.frame + 1) % self.n_frames))
        for i, key in enumerate(OUTLINE_METRICS, start=1):
            pl.add_key_event(str(i), lambda k=key: self.set_metric(k))
        # pyvista binds Up/Down to camera zoom and b to point picking by default, and
        # add_key_event appends rather than replaces: drop those defaults first
        for k in ("Up", "Down", "b"):
            pl.iren._key_press_event_callbacks.pop(k, None)
        pl.add_key_event("Up", lambda: self.set_snapshot(self.snap_index + 1))
        pl.add_key_event("Down", lambda: self.set_snapshot(self.snap_index - 1))
        pl.add_key_event("0", lambda: self.set_snapshot(0))
        pl.add_key_event("e", lambda: self.set_snapshot(len(self.snapshots) - 1))
        pl.add_key_event("b", lambda: self.set_snapshot(self.history["best_index"])
                         if self.history["best_index"] is not None else None)

    def _update_nav_text(self):
        self._nav_text.SetInput(f"subject {self.subject_index + 1} / {len(self.subjects)}:  "
                                f"{self.subjects[self.subject_index]}")
        name = Path(self.runs[self.run_index]).name      # full name is in the header
        short = name if len(name) <= 24 else name[:23] + "…"
        self._run_text.SetInput(f"run {self.run_index + 1} / {len(self.runs)}:  {short}")

    # --- scene -----------------------------------------------------------
    def _build_slices(self):
        for view in VIEWS:
            if view not in self.data:
                continue
            for s in self.data[view]["slices"]:
                nx, ny, _ = s["grey"].shape
                key = (view, int(s["index"]))
                in_history = any(key in snap["affines"] for snap in self.snapshots)
                if not in_history:
                    print(f"[warn] {s['name']}: no rows in history.csv; its NIfTI affine is used "
                          f"for every snapshot", file=sys.stderr)
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
        # Dilated-mask contours: polylines in voxel coordinates of their LA plane.
        for key, contours in self.overlays["contours"].items():
            pts, cells, off = [], [], 0
            for c in contours:
                n = len(c)
                pts.append(c)
                cells.append(np.concatenate([[n], np.arange(off, off + n)]))
                off += n
            ijk = np.vstack(pts)
            mesh = pv.PolyData(np.zeros_like(ijk), lines=np.concatenate(cells))
            actor = self._mesh(mesh, color=MASK_CONTOUR_COLOR, line_width=MASK_CONTOUR_WIDTH,
                                     lighting=False, name=f"mask:{key[0]}:{key[1]}")
            self._mask_items.append({"key": key, "ijk": ijk, "mesh": mesh, "actor": actor})

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
                            "t_used": t_used, "t_img": t_img, "centre_b": np.array(centre_b), "n": n,
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
        for m in self._mask_items:
            A = self._pose(m["key"], snap)
            if A is not None:
                m["mesh"].points = voxel_to_world(A, m["ijk"])
        self._apply_sa_lines(snap)
        self._apply_radius(snap)
        li = self._line_items
        if li is None:
            return
        n = li["n"]
        x0, d = li["x0"].copy(), li["d"].copy()                # acquired geometry (fallback)
        for i in range(n):
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
        li["cur_x0"], li["cur_d"] = x0, d
        for name in ("used", "img"):
            t = li["t_" + name]                                 # (N, 2)
            seg = np.stack([x0 + t[:, :1] * d, x0 + t[:, 1:] * d], axis=1)   # (N, 2, 3)
            for item in li["per_slice"]:
                A = self._affine_for(item["entry"], snap)
                item[name + "_mesh"].points = burn_points(seg[item["idx"]], A)

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
        self._move_text.SetInput(f"move {label(self.profile_key) if e else '-'}"
                                 + (f"   ({n_moved} moved)" if n_moved else ""))
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
        paths = self._mask_paths().get(e["key"])
        if not paths:
            return None
        A = self._affine_for(e, snap)
        vox = (np.linalg.inv(A) @ np.column_stack([pts, np.ones(len(pts))]).T)[:2].T
        inside = np.zeros(len(pts), bool)
        for path in paths:
            inside ^= path.contains_points(vox)
        return inside

    def _profile_items(self, snap: dict) -> list[dict]:
        """The intersections of the picked slice: (other slice name, start, unit dir, t_img, t_used)."""
        e = self._entry_by_name().get(self.profile_key)
        if e is None:
            return []
        out = []
        li = self._line_items
        if li is not None and "cur_x0" in li:
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
                out.append(dict(other=oe, x0=li["cur_x0"][i], d=li["cur_d"][i],
                                t_img=li["t_img"][i], t_used=li["t_used"][i]))
            out.sort(key=lambda r: VIEWS.index(r["other"]["view"]))
        if e["view"] == "SA":
            sl = e["key"][1]
            names = {x["key"][1]: x for x in self.entries if x["view"] == "SA"}
            nb = sl + 1 if sl + 1 in names else sl - 1
            if nb in names:
                item = dict(other=names[nb], x0=None)
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

    def _update_profiles(self):
        charts = getattr(self, "_profile_charts", [])
        if not charts or not self.entries:
            return
        snap = self.snapshots[self.snap_index]
        items = self._profile_items(snap) if self.show_profiles else []
        e = self._entry_by_name().get(self.profile_key)
        for k, slot in enumerate(charts):
            ch, pl_ = slot["chart"], slot["plots"]
            slot["item"] = None
            if k >= len(items):
                ch.visible = False
                continue
            it = items[k]
            ch.visible = True
            names = f"{label(e['name'])} \u00d7 {label(it['other']['name'])}"
            if it["x0"] is None:
                ch.title = f"{names}: no crossing"
                for p_ in pl_.values():
                    p_.update(np.array([0.0, 1.0]), np.array([np.nan, np.nan]))
                continue
            t = np.arange(it["t_img"][0], it["t_img"][1] + 1e-6, PROFILE_STEP_MM)
            pts = it["x0"] + t[:, None] * it["d"]
            ya, yb = self._sample(e, snap, pts), self._sample(it["other"], snap, pts)
            used = (t >= it["t_used"][0]) & (t <= it["t_used"][1]) if np.isfinite(it["t_used"]).all() \
                else np.zeros(len(t), bool)
            ok = used & np.isfinite(ya) & np.isfinite(yb)
            if ok.sum() > 2 and ya[ok].std() > 0 and yb[ok].std() > 0:
                ncc = float(np.corrcoef(ya[ok], yb[ok])[0, 1])
                ch.title = f"{names}   NCC {ncc:.2f}"
            else:
                ch.title = f"{names}   (no loss points)"
            for side, y in (("a", ya), ("b", yb)):
                pl_[side + "_img"].update(t, y)
                pl_[side + "_used"].update(t, np.where(used, y, np.nan))
            masks = [m for m in (self._in_mask(e, snap, pts), self._in_mask(it["other"], snap, pts)) if m is not None]
            in_mask = np.logical_or.reduce(masks) if masks else np.zeros(len(t), bool)
            pl_["mask"].update(t, np.where(in_mask, float(PROFILE_MASK_Y), np.nan))
            ch.x_range = [float(t[0]), float(t[-1]) if len(t) > 1 else float(t[0]) + 1.0]
            ch.y_range = [0, 255]
            slot["item"] = dict(e=e, other=it["other"], x0=it["x0"], d=it["d"], t=t, ya=ya, yb=yb, names=names,
                                title=ch.title)
        self._draw_hover()

    # --- hover on the profile charts -> tick marks on the line in 3D ------------------
    def _on_mouse_move(self, *_):
        x, y = self.pl.iren.interactor.GetEventPosition()
        w, h = self.pl.window_size
        hover = None
        for k, slot in enumerate(getattr(self, "_profile_charts", [])):
            ch, it = slot["chart"], slot["item"]
            if it is None or not ch.visible:
                continue
            (cx, cy), (cw, chh) = ch.loc, ch.size
            if not (cx * w <= x <= (cx + cw) * w and cy * h <= y <= (cy + chh) * h):
                continue
            p1, p2 = ch.x_axis.GetPoint1(), ch.x_axis.GetPoint2()
            if p2[0] - p1[0] < 1 or not p1[0] <= x <= p2[0]:
                break
            lo, hi = ch.x_range
            hover = (k, lo + (x - p1[0]) / (p2[0] - p1[0]) * (hi - lo))
            break
        if hover is None and self._hover is None:
            return                                            # nothing changed: no render
        self._hover = hover
        self._draw_hover()
        self.pl.render()

    def _draw_hover(self):
        charts = getattr(self, "_profile_charts", [])
        for slot in charts:
            slot["plots"]["cursor"].visible = False
            if slot["item"] is not None:
                slot["chart"].title = slot["item"]["title"]
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

    def _on_profiles_toggle(self, value: bool):
        self.show_profiles = bool(value)
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
        for e in self.entries:
            A = self._affine_for(e, snap)
            e["mesh"].points = voxel_to_world(A, e["grid"])
            e["outline_mesh"].points = voxel_to_world(A, e["corners"])
        self._apply_overlays()
        self._apply_metric_values()
        if self._hist_slider is not None:
            c = pv.Color("#ffa726" if snap["is_best"] else "white").float_rgb
            self._hist_slider.GetRepresentation().GetLabelProperty().SetColor(*c)
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
            txt = self._row_text.get(e["name"])
            if txt is not None:
                shown = fmt.format(val) if np.isfinite(val) else "-"
                txt.SetInput(f"{label(e['name'])}   {shown}")
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
                "no history for this subject (aligned without --logging):\n"
                "  no metric colours; the slices show the saved (final) pose")
            return
        fmt = spec["fmt"]
        good, bad = (hi, lo) if spec["higher_better"] else (lo, hi)
        sub = self.metric_range.get(spec["col"])
        sub_txt = (f"subject range {fmt.format(sub[0])} .. {fmt.format(sub[1])}" if sub
                   else "no values for this subject")
        self.outline_text.SetInput(
            f"outline colour = {spec['label']} of each plane at the shown step\n"
            f"  full green from {fmt.format(good)}   full red from {fmt.format(bad)}   grey = no value\n"
            f"  {sub_txt}; adjust with the colour min / max sliders")

    # --- colour-range sliders ----------------------------------------------
    def _add_range_sliders(self):
        """Colour-range sliders hugging the colour bar: max above the strip, min below.
        They carry no built-in label; the values are small texts on the right."""
        spec = OUTLINE_METRICS[self.metric]
        b_lo, b_hi = spec["bounds"]
        lo, hi = self.color_range[self.metric]
        w, h = self.pl.window_size
        common = dict(style="modern", color="#bbbbbb", slider_width=0.012, tube_width=0.003,
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
                "", position=(RANGE_TEXT_X * w, (y - 0.008) * h), font_size=7, color="#bbbbbb")
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
            self._log("no history for this subject: metric colours unavailable")
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

    def _colors(self, s: dict) -> np.ndarray:
        grey = s["grey"][:, :, self.frame]
        seg = None if s["seg"] is None else s["seg"][:, :, self.frame]
        return cell_colors(grey, seg, self.show_seg, self.seg_alpha)

    def _refresh_colors(self):
        for e in self.entries:
            e["mesh"].cell_data["rgb"] = self._colors(e["slice"])
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
        self._text(label, position=(PANEL_X_TXT, y + 2), font_size=9, color="white")
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
        y -= 8
        if "SA" in self.data:
            add_row("All SA  [a]", lambda v: self._set_group("SA", v), True, "all_sa")
        if any(v.startswith("LA") for v in self.data):
            add_row("All LA  [l]", lambda v: self._set_group("LA", v), True, "all_la")
        y -= 8
        self._text("profile", position=(PROFILE_RADIO_X - 14, y + ROW_PITCH - 4), font_size=7, color="#bbbbbb")
        for e in self.entries:
            self._profile_radios[e["name"]] = pl.add_radio_button_widget(
                lambda n=e["name"]: self.set_profile(n), "profile", value=(e["name"] == self.profile_key),
                position=(PROFILE_RADIO_X, y + 3), size=14, border_size=3,
                color_on="#ffa726", color_off="#555555", background_color="#303030")
            add_row(label(e["name"]), self._make_slice_cb(e), True, e["name"])

        # Outline-colour metric (radio group: exactly one active)
        y -= 14
        self._text("Outline colour by", position=(PANEL_X_BOX, y + 2), font_size=9, color="#bbbbbb")
        y -= ROW_PITCH
        for i, (key, spec) in enumerate(OUTLINE_METRICS.items(), start=1):
            suffix = ("  (no history)" if not self.has_metrics
                      else "" if spec["col"] in self.metric_range else "  (no values)")
            self._add_radio(f"{spec['label']}  [{i}]{suffix}", key, y)
            y -= ROW_PITCH

        # Header texts first: slider callbacks may fire during widget creation
        self.title_text = self._text("", position=(w - 640, h - 40), font_size=11, color="white")
        self.run_name_text = self._text("", position=(w - 640, h - 80), font_size=7, color="#80cbc4")
        self.metrics_text = self._text("", position=(w - 640, h - 144), font_size=9, color="#bbbbbb")
        self.outline_text = self._text("", position=(w - 640, h - 212), font_size=8, color="#bbbbbb")

        # History slider (bottom): its value is the optimisation step, snapped to the
        # nearest snapshot; the knob label shows that step.  Cardiac-phase slider above it.
        hx0, hx1, hy = 0.30, 0.95, 0.07
        self._snap_steps = np.array([snap["step"] for snap in self.snapshots], dtype=float)
        step_max = max(float(self._snap_steps.max()), 1.0)   # a single snapshot still needs a non-empty range
        self._hist_slider = pl.add_slider_widget(
            self._on_hist_slider, rng=[0.0, step_max], value=self._snap_steps[self.snap_index],
            title="optimization history" if self.has_metrics else "saved pose (no history)",
            pointa=(hx0, hy), pointb=(hx1, hy),
            fmt="%.0f", style="modern", color="white", title_height=0.02,
            interaction_event="always",   # update while dragging, not only on release
        )
        # with a single snapshot the knob simply snaps back (see _on_hist_slider)
        self._text(f"{int(self._snap_steps.max())}", position=(hx1 * w + 12, hy * h - 4), font_size=9,
                   color="#bbbbbb")
        if self.history["best_index"] is not None:
            # marker below the slider (under its title row) at the best pose's position
            best_step = self._snap_steps[self.history["best_index"]]
            bx = hx0 + (hx1 - hx0) * best_step / step_max
            self._text(f"best {int(best_step)}", position=(bx * w - 22, hy * h - 62), font_size=7,
                        color="#ffa726")

        if self.n_frames > 1:
            self._frame_slider = pl.add_slider_widget(
                self._on_frame_slider, rng=[0, self.n_frames - 1], value=self.frame,
                title="cardiac phase", pointa=(hx0, 0.20), pointb=(hx1, 0.20),
                fmt="%.0f", style="modern", color="white", title_height=0.02,
                interaction_event="always",
            )

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
        self.title_text.SetInput(self.title)
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

        sm = snap["summary"]
        if sm:
            def f(key, fmt="{:.3f}"):
                val = sm.get(key)
                return fmt.format(val) if val is not None and np.isfinite(val) else "-"
            lines = [
                f"objective {f('objective')}   NCC {f('ncc_mean')}   NMI {f('nmi_mean')}   Dice {f('seg_dice_mean')}",
                f"Hausdorff mean {f('hd_mean_mm', '{:.2f}')} mm   max {f('hd_max_mm', '{:.2f}')} mm",
                f"SA shift rms {f('sa_shift_rms_mm', '{:.2f}')} mm   jitter {f('sa_shift_jitter_mm', '{:.2f}')} mm   "
                f"dz rms {f('sa_dz_rms_mm', '{:.2f}')} mm   tilt rms {f('sa_tilt_rms_deg', '{:.2f}')} deg",
            ]
        elif not self.has_metrics:
            lines = ["(no history: align with motion_gradient.py --logging for metrics",
                     " and the optimisation history)"]
        else:
            lines = ["(no history_summary.csv for this step)"]
        self.metrics_text.SetInput("\n".join(lines))

    # --- run -------------------------------------------------------------
    def show(self, **kwargs):
        self.pl.show(**kwargs)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def _subjects_in_run(run_dir: Path) -> list[str]:
    """Viewable subjects of a run (with or without a history), alphabetical."""
    return sorted(p.name for p in run_dir.iterdir() if _is_subject_dir(p))


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


def load_subject_run(run_dir: Path, subject: str, instance: int | None = None) -> dict:
    """Everything the viewer needs for one subject of a run. The images are read from <run>/<subject>/ (current
    layout) or, for older runs, from <run>/<subject>/Instance_<n>/."""
    subject_dir = run_dir / subject
    if instance is None:
        instance = _first_instance(subject_dir)
    image_dir = subject_dir / f"Instance_{instance}" if instance is not None else subject_dir
    metrics_dir = subject_dir / METRICS_SUBDIR

    print(f"Subject {subject}" + (f", instance {instance}" if instance is not None else ""))
    data = load_images(image_dir)
    describe_images(image_dir, data)
    if (metrics_dir / HISTORY_CSV).exists():
        history = load_history(metrics_dir)
        describe_history(metrics_dir, history)
    else:
        history = history_from_images(data)
        print(f"[history] none for {subject} (aligned without --logging): showing the saved pose, no metrics")
    overlays = load_overlays(metrics_dir)
    if metrics_dir.is_dir():
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
    return {"data": data, "history": history, "overlays": overlays,
            "title": f"UKBB {subject}" + (f" / Instance_{instance}" if instance is not None else ""), "run_name": run_dir.name}


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

    def loader(run: Path, subject: str) -> dict:
        return load_subject_run(run, subject, args.instance)

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
    )
    if args.screenshot is not None:
        viewer.show(screenshot=str(args.screenshot), auto_close=True)
        print(f"Saved {args.screenshot}")
    else:
        viewer.show()


if __name__ == "__main__":
    main()
