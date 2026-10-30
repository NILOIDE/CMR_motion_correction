"""
Motion correction of cine CMR stored as DICOM (entry point; the motion correction is motion_gradient.py).

Input (--root): one folder per subject; every .dcm / .ima file below it is read (hidden files and folders skipped).
  - An image is one series (the files sharing a SeriesInstanceUID), named after the deepest folder holding all its
    files (e.g. <subject>/<series>/phase_<nn>/*.dcm -> <series>; files directly in the subject folder:
    SeriesDescription_SeriesNumber).
  - Its slices are the distinct ImagePositionPatient / ImageOrientationPatient, ordered along the slice normal; its
    cardiac frames are the files at one slice, ordered by TriggerTime, then InstanceNumber, then path. The folder layout
    of the files does not matter.
  - Slice thickness: SliceThickness (--thickness if the header has none). RescaleSlope / Intercept are applied.
  - Complex data: REAL / IMAGINARY image pairs (ComplexImageComponent, or ImageType R / I) are combined into the
    magnitude; PHASE images are ignored; if a series has MAGNITUDE images, only those are used.
  - No segmentations: the lines are masked by a disc around the centre of each stack (--center-mask-radius-mm,
    dilated and annealed like a heart mask; 0 = unmasked, lines over the whole image overlap).
Geometry is converted from DICOM's LPS to RAS; arrays use the NIfTI convention arr[x, y] = pixel (row y, column x).
Each slice is handed to motion_gradient.py as a 2D+time cine (motion_gradient.SliceImage) with its own affine.

Output (<out-root>/<run-name>/<subject>/): every input file rewritten at the same relative path with the corrected
ImagePositionPatient / ImageOrientationPatient / SliceLocation, new SOP / series UIDs (derived from the originals) and
SeriesDescription + ' aligned'; pixel data unchanged (magnitudes are written for real/imaginary input;
--image-dtype uint8: 8-bit with RescaleSlope / Intercept). Plus the
metrics of motion_gradient.py (--logging), and per subject two line plots of the metrics over the optimization
(plot_metric_trends, in <subject>/metrics/): metric_trends.png (the optimized objective, then NCC / LNCC / NMI per slice:
mean, min / max and inter-quartile range over all slices, the slope over the last 25% of the steps, the saved best pose)
and metric_trends_stacks.png (the same per stack).

Usage: python moco_dicom.py [--root DIR] [--out-root DIR] [--run-name NAME] [--thickness MM]
                            [--image-dtype original|uint8] [motion correction options, see --help]
Needs pydicom (pip install pydicom).
"""
import json
import os
import re
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

import motion_gradient as mg

DCM_SUFFIXES = (".dcm", ".ima")
LPS = np.diag([-1.0, -1.0, 1.0, 1.0])                  # RAS <-> LPS (its own inverse)
SAME_POS_MM = 1e-3                                       # files closer than this (and same orientation) = one slice
SAME_IOP = 1e-5


@dataclass(frozen=True)
class DicomSeries:
    name: str                                           # image name: the series' folder (see dicom_series)
    files: tuple                                        # its .dcm files (Paths)


@dataclass
class DicomSlice:
    """Where one slice came from: per frame, its source file(s) (two for a real/imaginary pair)."""
    files: list
    subj_dir: Path


def long_path(p: Path) -> str:
    """p as a string usable past Windows' 260-character MAX_PATH limit (the output path repeats the long series folder
    name in the file names, so it easily exceeds it): the extended-length form \\\\?\\C:\\... (\\\\?\\UNC\\... for network
    paths). Unchanged on other systems."""
    s = os.path.abspath(p)
    if os.name != "nt" or s.startswith("\\\\?\\"):
        return s
    return "\\\\?\\UNC\\" + s[2:] if s.startswith("\\\\") else "\\\\?\\" + s


def dicom_files(subj_dir: Path) -> list[Path]:
    return sorted(p for p in subj_dir.rglob("*") if p.is_file() and p.name.lower().endswith(DCM_SUFFIXES)
                  and not any(r.startswith(".") for r in p.relative_to(subj_dir).parts))


def dicom_component(ds) -> str:
    """'M' (magnitude), 'P' (phase), 'R' (real) or 'I' (imaginary) of a DICOM image; 'M' if not stated."""
    c = str(ds.get("ComplexImageComponent", "") or "").upper()
    if c in ("MAGNITUDE", "PHASE", "REAL", "IMAGINARY"):
        return c[0]
    it = [str(x).upper() for x in (ds.get("ImageType") or [])]
    return it[2] if len(it) > 2 and it[2] in ("M", "P", "R", "I") else "M"


def dicom_series(subj_dir: Path) -> list[DicomSeries]:
    """The subject's series (headers only), nearest to subj_dir first."""
    groups, meta = {}, {}
    for f in dicom_files(subj_dir):
        ds = pydicom.dcmread(str(f), stop_before_pixels=True,
                             specific_tags=["SeriesInstanceUID", "SeriesDescription", "SeriesNumber"])
        uid = str(ds.get("SeriesInstanceUID", f.parent))
        groups.setdefault(uid, []).append(f)
        meta.setdefault(uid, ds)
    out = []
    for uid, fs in groups.items():
        common = Path(os.path.commonpath([str(f.parent) for f in fs]))
        if common == subj_dir:
            ds = meta[uid]
            name = re.sub(r"[^A-Za-z0-9_\-]+", "_", f"{ds.get('SeriesDescription', 'series')}_{ds.get('SeriesNumber', '')}").strip("_")
        else:
            name = common.name
        out.append(DicomSeries(name=name, files=tuple(sorted(fs))))
    names = [s.name for s in out]
    out = [DicomSeries(f"{s.name}_{i}" if names.count(s.name) > 1 else s.name, s.files) for i, s in enumerate(out)]
    return sorted(out, key=lambda s: (len(s.files[0].relative_to(subj_dir).parts), s.name.lower()))


def dicom_layout(series: DicomSeries) -> dict:
    """Geometry of a series from its headers: {affines (Z, 4, 4) RAS, one per slice with the origin at its first pixel,
    shape (X, Y), files [z][t] -> tuple of paths (one magnitude file, or a real/imaginary pair), thickness
    (SliceThickness, None if absent)}. Slices are the files sharing position and orientation (so slices that are not
    parallel, e.g. of a previously aligned series, stay apart), ordered along the first file's normal."""
    hdrs = [(f, pydicom.dcmread(str(f), stop_before_pixels=True)) for f in series.files]
    comps = {f: dicom_component(d) for f, d in hdrs}
    use = "M" if "M" in comps.values() else "RI"
    hdrs = [(f, d) for f, d in hdrs if comps[f] in use]
    d0 = hdrs[0][1]
    dr, dc = (float(x) for x in d0.PixelSpacing)         # (between rows, between columns)
    by_slice = {}
    for f, d in hdrs:
        ipp = np.array(d.ImagePositionPatient, float)
        iop = np.array(d.ImageOrientationPatient, float)
        key = tuple(np.round(ipp / SAME_POS_MM).astype(int)) + tuple(np.round(iop / SAME_IOP).astype(int))
        by_slice.setdefault(key, dict(ipp=ipp, iop=iop, items=[]))["items"].append(
            (float(d.get("TriggerTime", 0) or 0), int(d.get("InstanceNumber", 0) or 0), str(f), comps[f], f))
    n0 = np.cross(*np.split(np.array(d0.ImageOrientationPatient, float), 2))
    affines, files = [], []
    for s in sorted(by_slice.values(), key=lambda s: float(s["ipp"] @ n0)):
        row_dir, col_dir = s["iop"][:3], s["iop"][3:]    # DICOM: along a row (columns increase), along a column
        A = np.eye(4)
        A[:3, 0], A[:3, 1], A[:3, 2] = row_dir * dc, col_dir * dr, np.cross(row_dir, col_dir)
        A[:3, 3] = s["ipp"]
        affines.append(LPS @ A)
        items = sorted(s["items"])
        if use == "M":
            files.append([(it[4],) for it in items])
        else:                                            # pair the k-th real with the k-th imaginary image
            re_, im_ = [it[4] for it in items if it[3] == "R"], [it[4] for it in items if it[3] == "I"]
            files.append(list(zip(re_, im_)))
    nt = min(len(f) for f in files)
    if any(len(f) != nt for f in files):
        print(f"WARNING {series.name}: slices have different frame counts, keeping the first {nt}")
    affines = np.stack(affines)
    if len(affines) > 1:                                 # slice axis column: each slice's normal x the mean spacing
        spacing = abs(float(np.diff(affines[:, :3, 3] @ (LPS[:3, :3] @ n0)).mean()))
        affines[:, :3, 2] *= spacing if spacing > 1e-6 else 1.0
    th = d0.get("SliceThickness")
    return dict(affines=affines, shape=(int(d0.Columns), int(d0.Rows)), files=[f[:nt] for f in files],
                thickness=float(th) if th not in (None, "") else None)


def read_pixels(path: Path) -> np.ndarray:
    """One DICOM image as float32 (x = column, y = row), RescaleSlope / Intercept applied."""
    d = pydicom.dcmread(str(path))
    a = d.pixel_array.astype(np.float32) * float(d.get("RescaleSlope", 1) or 1) + float(d.get("RescaleIntercept", 0) or 0)
    return a.T


def load_subject(subj_dir: Path, args) -> mg.Subject:
    """One subject folder -> motion_gradient.Subject: one SliceImage (full (X, Y, T) cine, own affine) per slice of
    every series; `source` is the slice's DicomSlice (its files, for save())."""
    slices = []
    for series in dicom_series(subj_dir):
        lay = dicom_layout(series)
        X, Y = lay["shape"]
        thickness = lay["thickness"] or args.thickness
        if thickness is None:
            raise ValueError(f"{series.name}: no SliceThickness in the header; pass --thickness")
        for z, frames in enumerate(lay["files"]):
            data = np.zeros((X, Y, len(frames)), np.float32)
            for t, fs in enumerate(frames):
                data[:, :, t] = read_pixels(fs[0]) if len(fs) == 1 else np.hypot(read_pixels(fs[0]), read_pixels(fs[1]))
            slices.append(mg.SliceImage(image=series.name, index=z, data=data, affine=lay["affines"][z],
                                        thickness=thickness, source=DicomSlice(files=frames, subj_dir=subj_dir)))
    return mg.Subject(subject_id=subj_dir.name, slices=slices, status="ok" if slices else "no_images")


def write_slice(src: DicomSlice, data: np.ndarray, affine: np.ndarray, out_subj: Path,
                window: tuple[float, float] | None = None) -> None:
    """Write one slice's cine as DICOM, file for file at the input's relative paths under out_subj, with the geometry
    of `affine` (RAS): ImagePositionPatient / ImageOrientationPatient / SliceLocation, new SOP and series UIDs
    (derived from the originals), SeriesDescription + ' aligned'.
    Pixel data: copied unchanged (window None), except for a real/imaginary pair, whose magnitude (data[:, :, t]) is
    written in the source's integer type. With window (lo, clip) (--image-dtype uint8): data[:, :, t] clipped to
    [lo, clip] and scaled to 8-bit [0, 255], with RescaleSlope = (clip - lo) / 255 and RescaleIntercept = lo so that
    readers recover the original intensity scale."""
    A = LPS @ affine
    u, v = A[:3, 0] / np.linalg.norm(A[:3, 0]), A[:3, 1] / np.linalg.norm(A[:3, 1])
    n = np.cross(u, v)
    for t, fs in enumerate(src.files):
        ds = pydicom.dcmread(str(fs[0]))
        ds.ImagePositionPatient = [f"{x:.6f}" for x in A[:3, 3]]
        ds.ImageOrientationPatient = [f"{x:.8f}" for x in np.concatenate([u, v])]
        if "SliceLocation" in ds:
            ds.SliceLocation = f"{float(A[:3, 3] @ n):.6f}"
        ds.SeriesInstanceUID = generate_uid(entropy_srcs=[str(ds.SeriesInstanceUID), "aligned"])
        ds.SOPInstanceUID = generate_uid(entropy_srcs=[str(ds.SOPInstanceUID), "aligned"])
        ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        ds.SeriesDescription = f"{ds.get('SeriesDescription', '')} aligned".strip()
        if window is not None or len(fs) > 1:            # pixel data rewritten (uncompressed)
            img = data[:, :, t].T                        # back to (rows, cols), original intensity units
            for k in ("RescaleSlope", "RescaleIntercept"):
                if k in ds:
                    del ds[k]
            if window is not None:
                lo, clip = window
                step = max(clip - lo, 1e-6) / 255.0
                px = np.clip(np.rint((img - lo) / step), 0, 255).astype(np.uint8)
                ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 8, 8, 7, 0
                ds.RescaleSlope, ds.RescaleIntercept = f"{step:.8g}", f"{lo:.8g}"
            else:
                signed = int(ds.get("PixelRepresentation", 0))
                px = np.rint(img).astype(np.int16 if signed else np.uint16)
                ds.BitsAllocated, ds.BitsStored, ds.HighBit = 16, 16, 15
            if len(fs) > 1:
                ds.ComplexImageComponent = "MAGNITUDE"
            ds.PixelData = px.tobytes()
            ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        path = out_subj / Path(fs[0]).relative_to(src.subj_dir)
        os.makedirs(long_path(path.parent), exist_ok=True)
        ds.save_as(long_path(path), enforce_file_format=True)


def save(subject: mg.Subject, affines: list[np.ndarray], out: Path, args) -> None:
    window = None
    if args.image_dtype == "uint8" and subject.slices:
        # one window per subject: from its minimum (signed data can be negative) to --uint8-percentile of all its
        # slices and frames
        values = np.concatenate([s.data.ravel() for s in subject.slices])
        window = (min(0.0, float(values.min())), float(np.percentile(values, args.uint8_percentile)))
    for s, A in zip(subject.slices, affines):
        write_slice(s.source, s.data, A, out, window)


def is_complete(subj_in: Path, subj_out: Path) -> bool:
    """True when every (used) input file has its output file at the same relative path."""
    if not os.path.isdir(long_path(subj_out)):            # not started: no need to read the input headers
        return False
    names =[fs[0].relative_to(subj_in) for series in dicom_series(subj_in) for z in dicom_layout(series)["files"]
             for fs in z]
    return bool(names) and all(os.path.exists(long_path(subj_out / n)) for n in names)


# --------------------------------------------------------------------------------------
# Metric trends over the optimization (--logging; from each subject's metrics/history.csv)
# --------------------------------------------------------------------------------------
TREND_METRICS = (("ncc", "NCC"), ("lncc", "LNCC"), ("nmi", "NMI"))
TREND_LATE_FRAC = 0.25                     # the "late slope": linear fit over this last fraction of the steps
TREND_STYLE = dict(mg.PLOT_STYLE, line="#2a78d6", range="#86b6ef", fit="#52514e")   # one hue: mean dark, range light


def short_stack_names(names: list[str]) -> list[str]:
    """Stack labels: the '_'-separated tokens that differ between the series names (often long and alike)."""
    tokens = [n.split("_") for n in names]
    common = set.intersection(*(set(t) for t in tokens)) if len(tokens) > 1 else set()
    return [("_".join(x for x in t if x not in common) or n)[:28] for n, t in zip(names, tokens)]


def trend_stats(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Per optimization step (snapshots in order, the re-labelled best-pose copy left out): mean, min, max and
    quartiles of `col` over the slices that have a value."""
    d = df[(df["is_best"] == 0) & np.isfinite(df[col])]
    g = d.groupby("step")[col]
    return pd.DataFrame(dict(mean=g.mean(), lo=g.min(), hi=g.max(), q1=g.quantile(0.25), q3=g.quantile(0.75)))


def trend_slope(steps: np.ndarray, y: np.ndarray, frac: float = TREND_LATE_FRAC):
    """Least-squares line through the last `frac` of the steps: (slope per 100 steps, fitted x, fitted y), or None
    with fewer than 3 points. Near 0 at the end of a run: converged; still clearly positive: more steps would help."""
    ok = np.isfinite(y)
    steps, y = steps[ok], y[ok]
    if len(steps) < 3:
        return None
    late = steps >= steps.max() - frac * (steps.max() - steps.min())
    if late.sum() < 3:
        late = np.arange(len(steps)) >= len(steps) - 3
    a, b = np.polyfit(steps[late], y[late], 1)
    xs = np.array([steps[late].min(), steps[late].max()])
    return 100.0 * a, xs, a * xs + b


def _trend_panel(ax, s: pd.DataFrame, best_step, label: str, title: str, st=TREND_STYLE, with_range: bool = True,
                 fontsize: int = 8) -> None:
    """One metric over the steps: the mean (2 px line), its range over the slices (min / max, 1 px lines in a lighter
    step of the hue) and inter-quartile wash, the late-slope fit and the best pose; the title, and below it (above the
    plot, clear of the data) first -> last mean, change and late slope (significant digits: tiny changes stay
    readable)."""
    from matplotlib.ticker import AutoLocator
    x = s.index.to_numpy(float)
    if with_range:
        ax.fill_between(x, s["q1"], s["q3"], color=st["line"], alpha=0.10, linewidth=0, label="inter-quartile range")
        ax.plot(x, s["lo"], color=st["range"], linewidth=1.0, label="min / max over slices")
        ax.plot(x, s["hi"], color=st["range"], linewidth=1.0)
    ax.plot(x, s["mean"], color=st["line"], linewidth=2.0, solid_capstyle="round", label="mean")
    fit = trend_slope(x, s["mean"].to_numpy(float))
    note = ""
    if fit is not None:
        slope, fx, fy = fit
        ax.plot(fx, fy, color=st["fit"], linewidth=1.0, label=f"linear fit, last {TREND_LATE_FRAC:.0%} of steps")
        note = f",  late slope {slope:+.2g} / 100 steps"
    if best_step is not None:
        ax.axvline(best_step, color=st["muted"], linewidth=0.8, label="best pose (saved)")
    first, last = s["mean"].iloc[0], s["mean"].iloc[-1]
    ax.set_title(title, fontsize=10, loc="left", pad=16)
    ax.text(0.0, 1.015, f"{label} {first:.4f} -> {last:.4f} ({last - first:+.2g}){note}", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=fontsize, color=st["ink2"])
    mg._style_axis(ax, st)
    ax.yaxis.set_major_locator(AutoLocator())               # (_style_axis sets integer ticks)
    ax.margins(x=0)


def plot_metric_trends(metrics_dir: Path, st=TREND_STYLE) -> list[Path]:
    """Line plots of the metrics over the optimization, from a subject's history.csv (and history_summary.csv):
      metric_trends.png         all slices: the optimized objective, then NCC / LNCC / NMI per slice -- mean, min / max
                                and inter-quartile range over the slices, the late slope and the best pose
      metric_trends_stacks.png  the same per stack (rows: metrics, columns: stacks, one y scale per row)
    Returns the files written (none without a history)."""
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    hist_path = metrics_dir / "history.csv"
    if not hist_path.exists():
        return []
    df = pd.read_csv(hist_path, dtype={"subject_id": str})
    metrics = [(c, lab) for c, lab in TREND_METRICS if c in df and np.isfinite(df[c]).any()]
    if not metrics or df["step"].nunique() < 2:
        return []
    sid = str(df["subject_id"].iloc[0])
    best = df.loc[df["is_best"] == 1, "step"]
    best_step = float(best.iloc[0]) if len(best) else None
    cfg_path = metrics_dir.parent.parent / "overall_metrics" / "run_config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    handles = [Line2D([], [], color=st["line"], linewidth=2, label="mean over slices"),
               Line2D([], [], color=st["range"], linewidth=1, label="min / max over slices"),
               Patch(facecolor=st["line"], alpha=0.10, label="inter-quartile range"),
               Line2D([], [], color=st["fit"], linewidth=1, label=f"linear fit, last {TREND_LATE_FRAC:.0%} of steps"),
               Line2D([], [], color=st["muted"], linewidth=0.8, label="best pose (saved)")]

    def legend(fig):
        leg = fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False, fontsize=9)
        for t in leg.get_texts():
            t.set_color(st["ink2"])

    out = []
    # all slices: objective + per-slice metrics
    summ_path = metrics_dir / "history_summary.csv"
    summ = pd.read_csv(summ_path) if summ_path.exists() else None
    panels = len(metrics) + (summ is not None)
    fig = Figure(figsize=(4.6 * panels, 4.4), layout="constrained", facecolor=st["surface"])
    axes = fig.subplots(1, panels, sharex=True, squeeze=False)[0]
    k = 0
    if summ is not None:
        s = summ[summ["is_best"] == 0].set_index("step")[["objective"]].rename(columns={"objective": "mean"})
        name = str(cfg.get("metric", "")).upper()
        _trend_panel(axes[0], s, best_step, "objective",
                     f"Optimized objective{f' ({name}, all lines)' if name else ''}", st, with_range=False)
        axes[0].set_xlabel("optimization step")
        k = 1
    n_slices = df[(df["is_best"] == 0) & np.isfinite(df[metrics[0][0]])].groupby("step").size().max()
    for ax, (col, lab) in zip(axes[k:], metrics):
        _trend_panel(ax, trend_stats(df, col), best_step, lab, f"{lab} per slice", st)
        ax.set_xlabel("optimization step")
    legend(fig)
    fig.suptitle(f"Subject {sid}: metric trends over the optimization ({n_slices} slices, all stacks)",
                 x=0.01, ha="left", color=st["ink"])
    path = metrics_dir / "metric_trends.png"
    fig.savefig(path, dpi=130)
    out.append(path)

    # per stack: rows = metrics, columns = stacks, one y scale per row
    stacks = list(dict.fromkeys(df["image"]))
    labels = short_stack_names(stacks)
    fig = Figure(figsize=(max(4.2 * len(stacks), 8), 3.0 * len(metrics) + 0.8), layout="constrained",
                 facecolor=st["surface"])
    axes = fig.subplots(len(metrics), len(stacks), sharex=True, sharey="row", squeeze=False)
    for j, (stack, lab_s) in enumerate(zip(stacks, labels)):
        d = df[df["image"] == stack]
        for i, (col, lab) in enumerate(metrics):
            ax = axes[i, j]
            s = trend_stats(d, col)
            if s.empty:
                ax.set_axis_off()
                continue
            _trend_panel(ax, s, best_step, lab, f"{lab_s} ({d['slice'].nunique()} slices)" if i == 0 else "", st,
                         fontsize=7)
            if j == 0:
                ax.set_ylabel(f"{lab} per slice")
            if i == len(metrics) - 1:
                ax.set_xlabel("optimization step")
    legend(fig)
    fig.suptitle(f"Subject {sid}: metric trends per stack", x=0.01, ha="left", color=st["ink"])
    path = metrics_dir / "metric_trends_stacks.png"
    fig.savefig(path, dpi=130)
    out.append(path)
    return out


OPTIMIZERS = ("adam", "nadam", "radam", "adamax", "rmsprop", "adagrad", "sgd")
# default learning rates (translation mm, rotation deg) per optimizer when --lr / --rot-lr are not given. The adaptive
# methods take steps of about lr per iteration whatever the gradient scale; Adagrad's steps shrink like 1/sqrt(step),
# so it needs a larger lr; plain SGD steps are lr * gradient, and the NCC objective's gradients are small (see --lr).
DEFAULT_LR = {"adam": (0.01, 0.004), "nadam": (0.2, 0.04), "radam": (0.2, 0.04), "adamax": (0.2, 0.04),
              "rmsprop": (0.2, 0.04), "adagrad": (1.0, 0.2), "sgd": (80.0, 20.0)}

def main():
    # Every option of motion_gradient.build_parser is available; the ones below are repeated here for visibility and
    # with defaults for DICOM data (a repeated option replaces the general one, see build_parser).
    ap = mg.build_parser(__doc__)
    g = ap.add_argument_group("paths")
    g.add_argument("--root", type=Path, default=Path(r"C:\Users\Nil\Documents\data\Wenqi_data"),
                   help="Directory containing one folder per subject (DICOM files anywhere below each). "
                        "Default: %(default)s")
    g.add_argument("--out-root", type=Path, default=None,
                   help="Parent of the run directory. Default: <root>_aligned next to --root")
    g.add_argument("--run-name", type=str, default=None,
                   help="Run directory name, used as given (pass an existing run's name to add to or resume it, e.g. with "
                        "--skip-processed). Default: the start time followed by the settings given on the command line, "
                        "so every run gets a new directory")
    g = ap.add_argument_group("run")
    g.add_argument("--max-subjects", type=int, default=-1,
                   help="Max number of subjects to process (-1 = all). Default: %(default)s")
    g.add_argument("--start-index", type=int, default=0,
                   help="First subject folder to process, as an index into the subject folders under --root sorted by "
                        "name (0-based; with --end-index, to submit the cohort in chunks). Default: %(default)s")
    g.add_argument("--end-index", type=int, default=-1,
                   help="Process only subject folders before this index (exclusive; -1 = to the end). "
                        "Default: %(default)s")
    g.add_argument("--skip-processed", action="store_true",
                   help="Skip subjects whose aligned images are already complete in the run directory (to resume a run, "
                        "pass its directory name with --run-name; a default name is always new)")
    g.add_argument("--logging", type=mg.str2bool, default=True, metavar="BOOL",
                   help="true (default): write everything besides the aligned images: the pose/metric history during "
                        f"optimization (every {mg.HISTORY_EVERY:g} of the run), per-subject CSVs and figures (metrics/), "
                        "cohort-wide CSVs and figures (overall_metrics/) and run_config.json. false: only the "
                        "motion-corrected images are written and no metrics are computed (fastest). "
                        "overall_metrics/sax_relative_motion.csv and run_config.json are written either way.")
    g = ap.add_argument_group("motion correction (DICOM defaults; every other option: motion_gradient.py, listed above)")
    g.add_argument("--pair-keep", type=float, default=0.1,
                    help="Stochastic optimization: fraction of intersection lines used at each step (1 = deterministic). "
                         "Each step draws a new random subset of round(fraction x lines) lines and samples only those, "
                         "so GPU memory and time per step scale with this fraction (the full objective for tracking "
                         "the best pose is evaluated without gradients, in chunks)")
    g.add_argument("--center-mask-radius-mm", type=float, default=55.0,
                   help="DICOM data has no segmentations, so by default the lines are masked by a disc of this radius "
                        "(mm) around the centre of each stack (where it projects onto each slice), grown by "
                        "--mask-dilation-mm and annealed from --mask-dilation-start-mm like a heart mask (final radius = "
                        "this + --mask-dilation-mm). 0 = no mask: lines over the whole image overlap. Default: %(default)s")
    ap.add_argument("--mask-dilation-start-mm", type=float, default=70,
                    help="Mask annealing (off unless given). The dilation (mm) used at the start of the run; it then "
                         "shrinks linearly to --mask-dilation-mm over the first --mask-anneal-frac of the run. "
                         "Example: '--mask-dilation-start-mm 40 --mask-dilation-mm 25' starts with a 40 mm mask and "
                         "ends at 25 mm.")
    g = ap.add_argument_group("neighbouring slices of a stack (the slice above / below; motion_gradient.SaxNeighbourLines)")
    g.add_argument("--sax-sax-weight", type=float, default=0.0,
                   help="Similarity supervision of neighbouring slices of the same stack: where the poses make a slice "
                        "and the one above / below it cross, their images are compared along the crossing line with NCC "
                        "and weight x sum(point weights x (1 - NCC)) / (number of intersection-line points) is "
                        "subtracted from the objective (a crossing never pays, a mismatched one costs most). Neighbours "
                        "d mm apart cross d / tan(relative tilt) from the tilt pivot: 2 mm apart, a relative tilt of "
                        "~1 deg brings the crossing into a 240 mm field of view, ~1.6 deg within 70 mm of the centre. "
                        "Parallel neighbours get no cost. Needs --dof 6 (0 = off). Default: %(default)s")
    g.add_argument("--sax-sax-falloff", choices=["gaussian", "threshold"], default="gaussian",
                   help="Point weights of --sax-sax-weight along the crossing line, by the distance r from the stack's "
                        "axis (through its centre along its normal). gaussian: exp(-r^2 / (2 sigma^2)), sigma = "
                        "--sax-sax-radius-mm. threshold: 1 within --sax-sax-radius-mm, 0 beyond. Default: %(default)s")
    g.add_argument("--sax-sax-radius-mm", type=float, default=70.0,
                   help="--sax-sax-falloff gaussian: its sigma (mm); threshold: the radius (mm) around the stack's axis "
                        "(here as --center-mask-radius-mm). Default: %(default)s")
    g.add_argument("--sax-gap-reg", type=float, default=0.0,
                   help="Barrier keeping neighbouring slices of a stack from approaching / crossing each other: on a disk "
                        "of --sax-gap-radius-mm around the stack's axis, their gap along the stacking direction is "
                        "penalized below --sax-gap-margin-mm (at most half the acquired gap: 1 mm for slices 2 mm "
                        "apart): mean softplus(margin - gap)^2 (mm^2). No image sampling (0 = off). Default: %(default)s")
    g.add_argument("--sax-gap-margin-mm", type=float, default=4.0,
                   help="Smallest gap (mm) allowed without cost by --sax-gap-reg (for each pair at most half its "
                        "acquired gap). Default: %(default)s")
    g.add_argument("--sax-gap-radius-mm", type=float, default=70.0,
                   help="Radius (mm) of the disk around the stack's axis on which --sax-gap-reg measures the gap. "
                        "Default: %(default)s")
    g.add_argument("--parallel-max-dist-mm", type=float, default=12.0,
                   help="Neighbours are each slice and the next one along its stack's normal, if at most this far "
                        "apart (a gap in a stack is not bridged). Default: %(default)s")
    g = ap.add_argument_group("DICOM data")
    g.add_argument("--thickness", type=float, default=None,
                   help="Slice thickness (mm) for series whose headers have no SliceThickness (else an error)")
    ap.add_argument("--optimizer", choices=OPTIMIZERS, default="sgd",
                    help="sgd: plain gradient steps (see --momentum, --nesterov); its steps scale with the gradient, "
                         "so it needs much larger learning rates (see --lr). SGD lead consistently higher performance. "
                         "adam: the default. nadam: Adam with Nesterov momentum (looks ahead, often a little faster). "
                         "radam: Adam with the variance rectified during the first steps (no warm-up needed, steadier start). "
                         "adamax: Adam with an infinity-norm denominator (steps bounded by lr, robust to gradient spikes). "
                         "rmsprop: per-parameter step scaling without momentum. "
                         "adagrad: step sizes shrink like 1/sqrt(step), a built-in annealing. "
                         "See default: %(default)s for per-optimizer default learning rates.")
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
    ap.add_argument("--lncc-window-mm", type=float, default=10.0, help="--metric lncc: window length (mm). Default: %(default)s")
    ap.add_argument("--lncc-min-std", type=float, default=0.1,
                    help="--metric lncc: windows whose profiles vary less than this (geometric mean of both standard "
                         "deviations, in units of the profile's standard deviation over the line) are faded out "
                         "(weight sd / this). Default: %(default)s")
    # ###################    HARDWARE PARAMS    ##########################################
    g.add_argument("--image-dtype", choices=["original", "uint8"], default="original",
                   help="Pixel data of the saved images. original: the input's pixel data unchanged (only the geometry "
                        "tags change). uint8: per subject, intensities from its minimum up to --uint8-percentile of all "
                        "its slices and frames (clipped above) are scaled to 8-bit [0, 255], written uncompressed with "
                        "RescaleSlope / RescaleIntercept (readers recover the original intensity scale). "
                        "Default: %(default)s")
    g.add_argument("--uint8-percentile", type=float, default=99.9,
                   help="Clip percentile for --image-dtype uint8 (99.9 clips 0.1%% of the voxels). Default: %(default)s")
    g = ap.add_argument_group("hardware")
    g.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                   help="Device for the alignment. Default: %(default)s")
    g.add_argument("--workers", type=int, default=4,
                   help="Background threads for loading and writing image files. Default: %(default)s")
    g.add_argument("--no-cuda-graph", action="store_true",
                   help="Run each optimization step eagerly instead of replaying a captured CUDA graph (same result, "
                        "slower; for debugging)")
    g.add_argument("--sample-dtype", choices=["float32", "float16", "bfloat16"], default="float16",
                   help="Precision of the images held on the GPU and of their sampling along the intersection lines (the "
                        "bulk of the memory for many slices): float16 halves the image memory and the sampled tensors; it "
                        "holds integer intensities up to 2048 exactly (brighter images are scaled by a power of two; the "
                        "metrics are scale-invariant) and resolves sample positions to ~0.05 pixel on a 200-pixel image. "
                        "bfloat16 only to ~0.4 pixel and ~0.4%% in intensity (coarse for sub-mm motion). Poses, line "
                        "geometry and metrics stay float32; the saved images are unaffected (see --image-dtype). "
                        "Default: %(default)s")
    args = ap.parse_args()
    run_dir, subjects = mg.run(ap, args, load_subject, save, is_complete)
    if args.logging:                                     # metric trends over the optimization, per subject
        n = 0
        for sid in subjects:
            try:
                n += len(plot_metric_trends(run_dir / sid / "metrics"))
            except Exception:  # noqa: BLE001
                print(f"Metric trend plots failed for {sid}:")
                traceback.print_exc()
        if n:
            print(f"{n} metric trend figure(s) written (<subject>\\metrics\\metric_trends*.png)")


if __name__ == "__main__":
    main()
