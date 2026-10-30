"""
3D slice-alignment viewer for moco_dicom.py runs with many slices (e.g. several stacks of 40-60 slices each).

The UK Biobank viewer (slice_viewer_3d.py) shows every slice and gives each its own checkbox, which does not scale past
a few dozen slices. Here the slices are organised by stack (DICOM series) and only a *subset* is drawn at a time: per
stack its current slice and +-w neighbours, which its ghost button turns into faint outlines in the stack's colour.

Data
----
Geometry comes from the run's history (<run>/<subject>/metrics/history.csv, written with --logging): one affine per
slice for step 0 (acquired), every 2% of the run and the best pose. The pixel data are read from the *input* DICOMs
(run_config.json's root, or --root), so slice indices match the history exactly. Without a history the aligned DICOMs
in <run>/<subject>/ are shown at their saved pose. Every slice is drawn as one textured quad (cheap to move while
scrubbing the history, even for hundreds of slices).

Which slices are shown
----------------------
Every stack shows its current slice and +-w neighbouring slices (w = 0 by default). Each stack's slice slider scrolls
only that stack; on the selected stack (profile radio button, , / .) its slice also becomes the selected one. Double
clicking a slice selects it and sets every other stack's current slice to the slice passing closest to its centre
(matched on the acquired geometry) -- the local neighbourhood the selected slice is checked against. The selected
Every shown slice has corner marks in its stack's colour around its outline-metric border; the selected slice's border
and corner marks are drawn thicker. The
"worst slice" button (d) moves every stack to its worst slice by the outline metric at the shown step (QC: where does
the alignment fail?) and selects the selected stack's one, the stacks keeping those slices. It only jumps there: the
slice sliders keep scrolling through the slices in stack order from those positions.
Each stack (block at the bottom left) has a row with its checkbox (show / hide), profile button, mean
outline metric and acquisition geometry (pixel spacing, slice thickness, slice step); below it its slice (with
"slice i/N": current slice index / number of slices) and
neighbours (+-w) sliders, and the correction of its shown slices at the shown step relative to the acquired pose: for
one slice its shift along the stack's u / v / normal axes and its in-plane rotation and tilt, for several their mean
shift, mean shift length and mean rotation angle. The header shows the largest slice motion from the acquired pose at
the shown step: if it is a fraction of a pixel, scrubbing the history visibly changes nothing.

Usage
-----
    python slice_viewer_dicom.py                          # newest run under --align-root, its first subject
    python slice_viewer_dicom.py vol1302 --run 20261029-170903_steps1000-lr80
    python slice_viewer_dicom.py --run-dir D:\\runs\\my_run --neighbours 2
    python slice_viewer_dicom.py --root D:\\dicom_input   # input DICOMs moved since the run

Interaction
-----------
    d                   go to the worst slice of each stack (also the button above the stack block)
    + -                 neighbours (+-w) of the selected slice's stack
    g                   ghosts on / off for the selected slice's stack (also the "ghost" buttons, right of each
                        stack's neighbours slider): its +-w neighbours are drawn as a thin border in the stack's colour
                        (no image) instead of as images; its current slice stays an image
    1 ... 9             profiles of that stack (also its round button, next to its checkbox): its current slice
                        becomes the selected one, every stack keeping its slice
    , / .               previous / next slice of the selected stack (as its slider: the other stacks keep theirs)
    double left click   select the slice under the mouse (the other stacks move to the slices crossing it)
    v                   show / hide the selected slice's stack (also the coloured checkboxes, left)
    m                   next outline metric (also the radio buttons "Outline colour by", left, each with the
                        subject-level value at the shown step): NCC, LNCC, NMI or motion (displacement of the slice
                        centre from the acquired pose, mm); red = bad, green = good, grey = no value
    o / i / p / n       slice outlines / intersection lines on the slices (the
                        profiled crossings on the selected slice, in the other stack's colour, and the run's
                        sampling lines between shown slices, lines.csv, at the shown step; drawn into both slices) /
                        intersection profiles / per-point metric charts of the selected slice
    j                   neighbour lines (white): each shown slice's crossing with the slice before and after it in
                        its stack at the shown step (the lines of --sax-sax-weight, motion_gradient.SaxNeighbourLines;
                        as slice_viewer_3d's SA-SA lines), thin inside both images, thick within --sax-sax-radius-mm
                        of the stack axis; parallel neighbours do not cross and show no line
    u / y               the loss's view of the images at the shown step (as slice_viewer_3d; from run_config.json):
                        u: image blurring -- the slice images and the intensity / metric profiles blurred in-plane as
                        by the coarse-to-fine schedule at that step (--blur-sigma -> --blur-sigma-end over
                        --blur-frac, in 0.2 mm steps; the header shows it); y: PSF in profiles -- each profile
                        averaged over the PSF band across the other slice's slab (--n-band samples, --psf-model, PSF
                        scale annealed from --psf-scale-start), as the loss samples it (not on the neighbour lines,
                        which the loss samples on the line). The chart titles note the blur / PSF scale in effect
    k                   the run's disc mask (--center-mask-radius-mm, yellow) on the shown slices: centred where
                        the stack centre projects onto each slice, radius = --center-mask-radius-mm + the mask
                        dilation of the shown step (annealed like the alignment; radius in the header)
    Up / Down           next / previous history snapshot;  Home / End / b: acquired / final / best pose
    Left / Right        previous / next cardiac frame
    z                   look at the selected slice face-on;  r: reset the camera
    ] / [ (PageDown / PageUp), buttons < / > next to "subject" (top right)
                        next / previous subject of the run (alphabetical)
    } / {, buttons < / > next to "run" (top right, above)
                        next / previous run of the same directory (newest first; > = older); the current
                        subject is kept when the other run has it, else the same position in its subject list.
                        Same subject: its pixels are not read again (only the run's history), the slices stay in
                        place and so does the view (camera, slice positions, +-w, ghosts, selection, frame): <1 s
                        instead of reading every DICOM file again
    h                   print this help;  q: quit
Profiles (right, as slice_viewer_3d): one per other stack, the selected slice against that stack's current slice (if
it crosses) or else its shown slice crossing the selected one closest to its centre (the
crossings are drawn on the selected slice in the other stack's colour), then two for the selected slice's neighbours
in its own stack, before and after it (their crossing exists only once the poses tilt them against each other: "no
crossing (parallel)" until then; thick / yellow bar: within --sax-sax-radius-mm of the stack axis, the part
--sax-sax-weight supervises; the title gives that weight, or "unsupervised"). Top block: grey values of both slices along
their intersection at the shown step and frame; thin curves: inside both images; thick: entering the loss, i.e. inside
either slice's disc mask at the shown step (yellow bar along the bottom); the outline metric over the thick part in the
title (NCC for motion). Bottom block: that metric per point along the same line (NCC: z_a z_b, mean = NCC; LNCC: local
NCC per window; none for NMI), drawn as the profiles: thin over the whole intersection (the metric without the mask; NCC
z-scored with the loss part's mean / std), thick over the loss part, yellow bar: inside the disc mask; each chart on one
y scale over the whole history so it can be watched rising or falling while scrubbing. Hovering a chart marks that position on the intersection in 3D (red tick across the line in both
slices) and shows the grey values / the metric value there in the chart titles.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyvista as pv
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.spatial.transform import Rotation

import moco_dicom as md
import motion_gradient as mg
import slice_viewer_3d as sv

ALIGN_ROOT = Path(r"C:\Users\Nil\Documents\data\Wenqi_data_aligned")
METRICS_SUBDIR, HISTORY_CSV = "metrics", "history.csv"
NEIGHBOURS_MAX = 10                          # +-w: up to this many neighbouring slices shown on each side
METRICS = {k: sv.OUTLINE_METRICS[k] for k in ("ncc", "lncc", "nmi")}
METRICS["motion"] = dict(label="motion", col="motion", higher_better=False, fmt="{:.2f} mm",
                         bounds=(0.0, 50.0), slider_fmt="%.1f mm", summary=())
# stack colours (checkboxes, ghosts, corner marks, partner lines): the first four well apart from each other (blue,
# magenta, saturated orange, dark red: distinct from the metric map's light yellow / green and its grey), the rest
# blues / violets clear of the cyan sampling lines and the yellow disc mask
STACK_COLORS = ["#2979ff", "#e040fb", "#ff6d00", "#b71c1c", "#40c4ff", "#b388ff", "#7c4dff", "#f48fb1",
                "#82b1ff", "#a1887f"]
OUTLINE_WIDTH, SEL_OUTLINE_WIDTH = 3, 5      # metric outline of the slices / of the selected slice
# every shown slice: an L-shaped mark at each corner in its stack colour, CORNER_FRAC of each edge long, 1 px wider than
# the metric outline below it (and drawn in front of it, see _front) so the two do not flicker
CORNER_FRAC = 0.08
CORNER_WIDTH, SEL_CORNER_WIDTH = OUTLINE_WIDTH + 1, SEL_OUTLINE_WIDTH + 1
CORNER_OFFSET_UNITS = -20                    # depth offset of the corner marks towards the camera (VTK offset units)
TCOORDS = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
PROFILE_SLOTS = 4                            # at most this many other-stack partners are profiled
NEIGHBOUR_SLOTS = 2                          # plus the selected slice's neighbours in its own stack: before, after
PROFILE_STEP_MM = 0.5
CHART_X, CHART_W = 0.745, 0.25              # chart column (right): one profile chart per slot (partners, then the two
CHART_Y0, CHART_TOP = 0.23, 0.93            # neighbours) above as many metric charts, from CHART_Y0 (above the colour
CHART_GAP = 0.004                           # bar) up to CHART_TOP (below the navigation block); gap between charts
# neighbour crossings (the slice above / below in the same stack, motion_gradient.SaxNeighbourLines): burnt into the
# slices, thin over the part inside both images, thick within --sax-sax-radius-mm of the stack axis (the part
# --sax-sax-weight supervises); white, apart from the stack colours, the cyan sampling lines and the yellow mask
NB_LINE_COLOR = "#ffffff"
NB_LINE_IMG_WIDTH, NB_LINE_USED_WIDTH = 2, 5
NAV_RIGHT_W = 380                          # width (px) of the run / subject navigation block, top right
# outline colour bar (bottom right, just above the frame slider) as slice_viewer_3d: a narrow strip with the
# "colour max" slider right above it and the "colour min" slider right below it, their values to the right
BAR_X0, BAR_X1 = 0.795, 0.935            # horizontal extent of strip and sliders (room for "max 50.00 mm")
BAR_Y0, BAR_H = 0.163, 0.022             # strip bottom and height
SLIDER_HI_Y = BAR_Y0 + BAR_H + 0.011      # max slider
SLIDER_LO_Y = BAR_Y0 - 0.011              # min slider
RANGE_TEXT_X = BAR_X1 + 0.006             # small value texts to the right of the sliders
RANGE_MIN_GAP = 1e-3                      # colour min is kept at least this far below colour max
MASK_CIRCLE_SAMPLES = 120                   # points on each slice's disc-mask circle
BLUR_QUANTUM_MM = 0.2                        # the optimizer's in-plane blur is rounded up to this (motion_gradient.align)
BLUR_CACHE_MAX = 256                         # blurred frame images kept (per slice, frame and blur)
LINE_COLOR = "#4dd0e1"
MAX_LINES = 3000                             # sampling-line overlay: at most this many segments
PANEL_X, TEXT_X, ROW = 12, 40, 24


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def short_labels(names: list[str]) -> list[str]:
    """Stack labels: the '_'-separated tokens that differ between the stack names (series names are often long and
    share most of their text), e.g. ...124139_..._slice_60_fov... -> '124139_60'."""
    tokens = [n.split("_") for n in names]
    common = set(tokens[0]).intersection(*tokens[1:]) if len(names) > 1 else set()
    out = []
    for n, t in zip(names, tokens):
        diff = [x for x in t if x not in common]
        out.append(("_".join(diff) or n)[:28])
    return out


def read_stacks(subj_dir: Path, workers: int = 8) -> list[dict]:
    """Every DICOM series below subj_dir as a stack of slices (moco_dicom's series / slice / frame logic, so names and
    slice indices match the run's history): [{name, thickness (mm, None if the header has none),
    slices: [{index, grey (X, Y, T) uint8, affine, shape}]}].
    Intensities are mapped to 0..255 per stack (1st / 99th percentile of all its slices and frames)."""
    stacks = []
    with ThreadPoolExecutor(workers) as ex:
        for series in md.dicom_series(subj_dir):
            lay = md.dicom_layout(series)
            X, Y = lay["shape"]

            def frame(fs):
                return md.read_pixels(fs[0]) if len(fs) == 1 else np.hypot(md.read_pixels(fs[0]), md.read_pixels(fs[1]))

            raw = [np.stack(list(ex.map(frame, frames)), axis=2) for frames in lay["files"]]       # (X, Y, T) each
            lo, hi = np.percentile(np.concatenate([r[::2, ::2].ravel() for r in raw]), [1.0, 99.0])
            hi = hi if hi > lo else lo + 1.0
            slices = [dict(index=z, grey=np.clip((r - lo) / (hi - lo) * 255.0 + 0.5, 0, 255).astype(np.uint8),
                           affine=lay["affines"][z], shape=(X, Y)) for z, r in enumerate(raw)]
            stacks.append(dict(name=series.name, slices=slices, thickness=lay["thickness"]))
            print(f"  {series.name}: {len(slices)} slices x {slices[0]['grey'].shape[2]} frames ({X}x{Y})", flush=True)
    return stacks


def history_from_stacks(stacks: list[dict]) -> dict:
    affines = {(sv.from_disk(st["name"]), s["index"]): s["affine"] for st in stacks for s in st["slices"]}
    snap = {"step": 0, "fraction": 0.0, "is_best": False, "affines": affines, "metrics": {}, "summary": None}
    return {"snapshots": [snap], "n_steps": 0, "best_index": None, "metric_range": {}, "available": False}


def reuse_stacks(cached: list[dict], src: Path, geometry_only: bool) -> list[dict] | None:
    """The cached stacks of a subject for another run: fresh stack dicts (the viewer adds its state to them) sharing the
    cached slices' pixels. geometry_only False: everything as cached (same input DICOMs: same pixels and acquired
    affines; the poses come from the run's history). True: the affines re-read from src's headers (no pixel data; for
    a run without history, whose saved output carries the corrected geometry); None if src's series / slices do not
    match the cached ones (then everything is read again)."""
    if not geometry_only:
        return [dict(name=st["name"], slices=st["slices"], thickness=st["thickness"]) for st in cached]
    by_name = {st["name"]: st for st in cached}
    out = []
    for series in md.dicom_series(src):
        st = by_name.get(series.name)
        lay = md.dicom_layout(series)
        if st is None or len(lay["affines"]) != len(st["slices"]) or tuple(lay["shape"]) != tuple(st["slices"][0]["shape"]):
            return None
        out.append(dict(name=st["name"], thickness=lay["thickness"],
                        slices=[dict(sl, affine=lay["affines"][z]) for z, sl in enumerate(st["slices"])]))
    return out if len(out) == len(cached) else None


def load_subject(run_dir: Path, subject: str, root: Path | None = None, workers: int = 8,
                 cache: dict | None = None) -> dict:
    """Stacks (pixels), history and lines.csv of one subject of a moco_dicom.py run. `cache` (a dict kept by the
    caller) holds the last subject's stacks: loading the same subject from another run then reads no pixel data -- from
    the same input DICOMs nothing at all but the run's history, otherwise only the headers (reuse_stacks)."""
    subj_out = run_dir / subject
    metrics = subj_out / METRICS_SUBDIR
    has_history = (metrics / HISTORY_CSV).exists()
    cfg_path = run_dir / "overall_metrics" / "run_config.json"
    config = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    src = subj_out
    if has_history:
        in_root = root or (Path(config["root"]) if config.get("root") else None)
        if in_root is not None and (in_root / subject).is_dir():
            src = in_root / subject
        else:
            print(f"[warn] input DICOMs of {subject} not found ({in_root}); reading the aligned output instead "
                  f"(pass --root): slice order may differ from the history", file=sys.stderr)
    stacks = None
    if cache is not None and cache.get("subject") == subject:
        same_src = Path(cache["src"]).resolve() == src.resolve()
        stacks = reuse_stacks(cache["stacks"], src, geometry_only=not same_src)
        if stacks is not None:
            print(f"Subject {subject}: pixels reused from the previous run"
                  + ("" if same_src else f" (geometry read from {src})"), flush=True)
    if stacks is None:
        print(f"Subject {subject}: reading {src} ...", flush=True)
        stacks = read_stacks(src, workers)
        if not stacks:
            raise RuntimeError(f"no DICOM series in {src}")
        if cache is not None:                            # only the latest subject is kept (~100 MB for 200 slices)
            cache.clear()
            cache.update(subject=subject, src=src, stacks=stacks)
    if has_history:
        history = sv.load_history(metrics)
        keys = {(sv.from_disk(st["name"]), s["index"]) for st in stacks for s in st["slices"]}
        n_hist = len(keys & set(history["snapshots"][0]["affines"]))
        print(f"[history] {len(history['snapshots'])} snapshots, {history['n_steps']} steps; "
              f"{n_hist}/{len(keys)} slices found in it")
        if n_hist == 0:
            print("[warn] no slice of the history matches the loaded DICOMs: showing the loaded geometry",
                  file=sys.stderr)
    else:
        history = history_from_stacks(stacks)
        print(f"[history] none for {subject} (run without --logging): showing the saved pose, no metrics")
    overlays = sv.load_overlays(metrics) if metrics.is_dir() else {"lines": None}
    return dict(stacks=stacks, history=history, lines=overlays.get("lines"), run_name=run_dir.name, subject=subject,
                config=config)


def is_subject_dir(p: Path) -> bool:
    if not p.is_dir() or p.name.startswith(".") or p.name == "overall_metrics":
        return False
    return (p / METRICS_SUBDIR / HISTORY_CSV).exists() or next(p.rglob("*.dcm"), None) is not None


def subjects_in(run_dir: Path) -> list[str]:
    return sorted(p.name for p in run_dir.iterdir() if is_subject_dir(p))


def runs_in(align_root: Path) -> list[Path]:
    """Run directories under align_root with at least one viewable subject, newest first (as slice_viewer_3d)."""
    runs = [p for p in align_root.iterdir() if p.is_dir() and any(is_subject_dir(q) for q in p.iterdir())]
    return sorted(runs, key=sv._created, reverse=True)


def find_run(align_root: Path, subject: str | None) -> Path | None:
    runs = [p for p in align_root.iterdir() if p.is_dir()
            and (is_subject_dir(p / subject) if subject else any(is_subject_dir(q) for q in p.iterdir()))]
    return max(runs, key=sv._created) if runs else None


# ----------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------
def frame_of(A: np.ndarray) -> np.ndarray:
    """Orthonormal (u, v, n) of a slice affine, as columns."""
    u, v = A[:3, 0] / np.linalg.norm(A[:3, 0]), A[:3, 1] / np.linalg.norm(A[:3, 1])
    return np.column_stack([u, v, np.cross(u, v)])


def rotation_deg(A: np.ndarray, B: np.ndarray) -> float:
    R = frame_of(A) @ frame_of(B).T
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))


def stack_geometry(st: dict) -> str:
    """Pixel spacing, slice thickness and slice step of a stack (acquired geometry), e.g. '1.5 x 1.5 mm px, 6 mm thick,
    2 mm apart'."""
    A = [s["affine"] for s in st["slices"]]
    du, dv = np.linalg.norm(A[0][:3, 0]), np.linalg.norm(A[0][:3, 1])
    text = f"{du:.3g} x {dv:.3g} mm px"
    if st.get("thickness"):
        text += f", {st['thickness']:.3g} mm thick"
    if len(A) > 1:
        n = frame_of(A[0])[:, 2]
        step = float(np.abs(np.diff([a[:3, 3] @ n for a in A])).mean())
        text += f", {step:.3g} mm apart"
    return text


# ----------------------------------------------------------------------------
# Viewer
# ----------------------------------------------------------------------------
class DicomSliceViewer:
    def __init__(self, run_dir: Path, subjects: list[str], subject: str, load,
                 neighbours: int = 0, metric: str = "ncc", start: int | None = None, frame: int = 0,
                 window_size=(1600, 1000), runs: list[Path] | None = None):
        self.run_dir, self.subjects, self.load = run_dir, subjects, load
        self.subject_index = subjects.index(subject)
        # runs to step through with the < / > buttons (newest first); the current one is always among them
        self.runs = list(runs) if runs else [run_dir]
        if not any(Path(r).resolve() == Path(run_dir).resolve() for r in self.runs):
            self.runs.insert(0, run_dir)
        self.run_index = next(i for i, r in enumerate(self.runs) if Path(r).resolve() == Path(run_dir).resolve())
        self._neighbours0 = int(np.clip(neighbours, 0, NEIGHBOURS_MAX))   # initial +-w of every stack
        self.metric = metric if metric in METRICS else "ncc"
        self.frame = frame
        self.show_outlines, self.show_lines, self.show_profiles = True, True, True
        self.show_mask = True
        self.show_metric_profiles = True              # per-point metric charts below the intensity profiles
        self.show_nb_lines = True                     # crossings with the neighbouring slices of the same stack
        # the loss's view of the images at the shown step (as slice_viewer_3d): blur = the in-plane Gaussian blur of
        # the coarse-to-fine schedule (slice images and profiles); PSF = profiles averaged over the other slice's slab
        self.show_blur = False
        self.show_psf = True
        self._blur_cache: dict = {}
        self._start = start
        self.W, self.H = window_size
        self.pl = pv.Plotter(window_size=window_size, title="DICOM slice viewer")
        self.pl.set_background("#202020")
        self._lut = pv.LookupTable(cmap=sv.OUTLINE_CMAP, scalar_range=(0.0, 1.0), nan_color=sv.OUTLINE_NAN_COLOR)
        self._bind_keys()
        self.pl.track_click_position(self._on_pick, side="left", double=True)
        self.pl.iren.add_observer("MouseMoveEvent", self._on_mouse_move)
        self._built = False
        self._ui_ready = False
        self._load_subject(self.subject_index)

    # --- subject lifecycle -----------------------------------------------------------
    def _load_subject(self, index: int) -> bool:
        """Load subject `index` of self.subjects (of self.run_dir) into the window. False (view unchanged) if it fails
        (the very first load re-raises)."""
        index = int(np.clip(index, 0, len(self.subjects) - 1))
        try:
            loaded = self.load(self.run_dir, self.subjects[index])
        except Exception as exc:  # noqa: BLE001
            if not self._built:
                raise
            print(f"[warn] could not load {Path(self.run_dir).name} / {self.subjects[index]}: "
                  f"{type(exc).__name__}: {exc}", file=sys.stderr)
            return False
        self.subject_index = index
        if self._built and self._same_slices(loaded):
            self._switch_run(loaded)
            return True
        if self._built:
            self._teardown()
        self._set_run_data(loaded)
        self._blur_cache = {}                            # (keyed by slice: new slices, new cache)
        self.stacks = loaded["stacks"]
        self.labels = short_labels([st["name"] for st in self.stacks])
        self.entries = []
        for s, st in enumerate(self.stacks):
            st["color"] = STACK_COLORS[s % len(STACK_COLORS)]
            st["geometry"] = stack_geometry(st)
            st["on"] = True
            st["ghost"] = False                              # its +-w neighbours drawn as ghosts (border only)
            st["pos"] = len(st["slices"]) // 2               # the stack's current slice (centre of its +-w window)
            st["w"] = self._neighbours0                      # neighbouring slices shown on each side
            st["entries"] = []
            # the run's default disc mask (--center-mask-radius-mm) is centred where the stack centre projects onto
            # each slice, in the slice's own pixel grid (it moves with the slice), as motion_gradient.center_disc_dist
            affines0 = np.stack([sl["affine"] for sl in st["slices"]])
            centre = mg.stack_center(affines0, st["slices"][0]["shape"])
            # the stack's axis (through its centre along its mean normal), as motion_gradient.attach_sax_pairs: the
            # neighbour crossings are supervised within --sax-sax-radius-mm of it
            normals = np.stack([frame_of(A)[:, 2] for A in affines0])
            normals *= np.where(normals @ normals[0] >= 0, 1.0, -1.0)[:, None]
            n = normals.mean(0)
            st["axis"] = (np.asarray(centre, float)[:3], n / np.linalg.norm(n))
            for p, sl in enumerate(st["slices"]):
                e = dict(stack=s, pos=p, index=sl["index"], key=(sv.from_disk(st["name"]), sl["index"]),
                         grey=sl["grey"], affine0=sl["affine"], shape=sl["shape"], tex_key=None,
                         thickness=float(st.get("thickness") or self._config_thickness(loaded)),
                         mask_c=(np.linalg.inv(sl["affine"]) @ centre)[:2],
                         mask_sp=(float(np.linalg.norm(sl["affine"][:3, 0])), float(np.linalg.norm(sl["affine"][:3, 1]))))
                st["entries"].append(e)
                self.entries.append(e)
        self.n_frames = min(e["grey"].shape[2] for e in self.entries)
        self.frame = int(np.clip(self.frame, 0, self.n_frames - 1))
        mid = self.stacks[0]["entries"]
        self.sel = mid[len(mid) // 2]
        self._synced_sel = None                      # selection the stacks' "around" positions were last set from
        self._lines_pairs = self._line_pairs(loaded.get("lines"))
        self._build_scene()
        self._ui_ready = False
        self._build_ui()
        self._ui_ready = True
        self._configure_metric()
        self._apply_snapshot()
        self._set_camera()
        self._built = True
        self.pl.render()
        return True

    def _set_run_data(self, loaded: dict):
        """Everything of a load that belongs to the run rather than the slices: history (poses, metrics), config, the
        initial snapshot and the caches derived from them."""
        self.data = loaded
        self.history = loaded["history"]
        self.snapshots = self.history["snapshots"]
        self.has_metrics = bool(self.history.get("available", True))
        self.config = loaded.get("config") or {}
        self.mask_r0 = float(self.config.get("center_mask_radius_mm") or 0.0)   # 0: the run used no default mask
        start = self._start
        self._start = None
        if start is None:
            start = self.history["best_index"] if self.history["best_index"] is not None else len(self.snapshots) - 1
        self.snap_index = int(np.clip(start, 0, len(self.snapshots) - 1))
        self._motion_range = None
        self.color_range = {}                        # metric -> outline colour [min, max] (see _color_range)
        self._max_motion = {}
        self._range_cache = (None,)                  # metric-chart y ranges (_metric_ranges)
        self._hover = None

    def _same_slices(self, loaded: dict) -> bool:
        """Whether `loaded` has exactly the shown slices (same subject from another run: same stacks, slice counts,
        pixels and acquired affines), so the scene can be kept and only the run's data swapped (_switch_run)."""
        if loaded["subject"] != self.data["subject"] or len(loaded["stacks"]) != len(self.stacks):
            return False
        for new, old in zip(loaded["stacks"], self.stacks):
            if new["name"] != old["name"] or len(new["slices"]) != len(old["slices"]):
                return False
            for a, b in zip(new["slices"], old["slices"]):
                if a["grey"] is not b["grey"] or not np.allclose(a["affine"], b["affine"]):
                    return False
        return True

    def _switch_run(self, loaded: dict):
        """Another run of the shown subject: keep the slices' actors, the view (camera, stack positions, +-w, ghosts,
        selection, frame) and swap the run's data -- poses, metrics, sampling lines, disc mask, config -- then rebuild
        only the UI (its sliders, charts and texts depend on the run). Much faster than a full load (no scene rebuild of
        hundreds of slices)."""
        self._teardown(keep_scene=True)
        self._set_run_data(loaded)
        self._lines_pairs = self._line_pairs(loaded.get("lines"))
        self._ui_ready = False
        self._build_ui()
        self._ui_ready = True
        self._configure_metric()
        self._apply_snapshot()
        self.pl.render()

    def _teardown(self, keep_scene: bool = False):
        """Remove the UI (widgets, charts, colour bar, texts) and, unless keep_scene, every actor of the scene."""
        pl = self.pl
        # switching a widget off leaves its representation (the drawn button / slider) in the renderer: remove those too
        # (a full teardown's clear_actors() would, but a kept scene must not lose its actors)
        reps = [w.GetRepresentation() for w in list(pl.button_widgets) + list(pl.slider_widgets)]
        pl.clear_button_widgets()
        pl.clear_slider_widgets()
        for r in reps:
            pl.renderer.RemoveViewProp(r)
        wh = getattr(pl, "widgets", pl)                  # pyvista keeps the radio-button groups by name: switch ours
        for attr in ("radio_button_widget_dict", "radio_button_title_dict"):  # off (else they stay active, invisible)
            groups = getattr(wh, attr, None)            # and forget them
            if isinstance(groups, dict):
                for name in ("outline_metric", "profile_stack"):
                    for w in groups.pop(name, None) or []:
                        if hasattr(w, "Off"):
                            w.Off()
                            pl.renderer.RemoveViewProp(w.GetRepresentation())
                        else:                            # (titles: text actors)
                            pl.remove_actor(w, render=False)
        # remove the charts *and* their shared context actor: pyvista adds that actor only when the first chart of a
        # renderer is added, so after clear_actors() (below) re-added charts would otherwise never be drawn again
        charts = getattr(pl.renderer, "_charts", None)
        if charts is not None:
            charts.deep_clean()
        for name in list(pl.scalar_bars.keys()) if hasattr(pl, "scalar_bars") else []:
            pl.remove_scalar_bar(name)
        if keep_scene:                                   # the UI's texts only (the scene's actors stay)
            for t in getattr(self, "_ui_texts", []):
                pl.remove_actor(t, render=False)
        else:
            pl.clear_actors()
        self._ui_texts = []

    # --- scene -----------------------------------------------------------------------
    def _pose(self, e: dict, snap: dict | None = None) -> np.ndarray:
        snap = self.snapshots[self.snap_index] if snap is None else snap
        return snap["affines"].get(e["key"], e["affine0"])

    @staticmethod
    def _centre(e: dict, A: np.ndarray) -> np.ndarray:
        nx, ny = e["shape"]
        return (A @ np.array([(nx - 1) / 2.0, (ny - 1) / 2.0, 0.0, 1.0]))[:3]

    def _plane_distance(self, e: dict, p: np.ndarray) -> float:
        """Distance (mm) of point p from the plane of slice e at the shown step."""
        A = self._pose(e)
        return abs(float(frame_of(A)[:, 2] @ (p - A[:3, 3])))

    def _build_scene(self):
        pl = self.pl
        for e in self.entries:
            nx, ny = e["shape"]
            corners = sv.voxel_to_world(self._pose(e), sv.corner_ijk(nx, ny))
            quad = pv.PolyData(corners.copy(), faces=np.array([4, 0, 1, 2, 3]))
            quad.active_texture_coordinates = TCOORDS.copy()
            img = pv.ImageData(dimensions=(nx, ny, 1))
            img.point_data["grey"] = e["grey"][:, :, self.frame].ravel(order="F")
            tex = pv.Texture(img)
            tex.interpolate = False
            e.update(quad=quad, img=img, tex_key=(self.frame, 0.0))
            e["actor"] = pl.add_mesh(quad, texture=tex, lighting=False, render=False, pickable=True)
            outline = pv.lines_from_points(corners, close=True)
            outline.point_data["metric"] = np.full(outline.n_points, np.nan)
            e["outline_mesh"] = outline
            e["outline"] = pl.add_mesh(outline, scalars="metric", cmap=sv.OUTLINE_CMAP, nan_color=sv.OUTLINE_NAN_COLOR,
                                       clim=(0.0, 1.0), show_scalar_bar=False, line_width=OUTLINE_WIDTH, render=False,
                                       pickable=False)
            e["outline"].mapper.lookup_table = self._lut
            e["outline"].mapper.SetUseLookupTableScalarRange(True)
            ghost = pv.lines_from_points(corners, close=True)
            e["ghost_mesh"] = ghost
            e["ghost"] = pl.add_mesh(ghost, color=self.stacks[e["stack"]]["color"], line_width=1, opacity=0.3,
                                     render=False, pickable=False)
            # numpy views of the VTK arrays, written in place when the pose changes (pyvista's setters are slow
            # for hundreds of meshes)
            e["point_views"] = [(m.points, m.GetPoints()) for m in (quad, outline, ghost)]
            e["metric_view"] = (outline.point_data["metric"], outline.GetPointData().GetArray("metric"))
        # selected slice: its metric outline and its corner marks drawn thicker (the other shown slices' corner marks:
        # one mesh, _update_corners)
        self._sel_mesh = pv.PolyData(np.zeros((16, 3)), lines=self._corner_cells(1))
        self._sel_actor = pl.add_mesh(self._sel_mesh, color="white", line_width=SEL_CORNER_WIDTH, lighting=False,
                                      render=False, pickable=False)
        self._front(self._sel_actor)
        self._sel_outlined = None                            # entry whose outline is drawn thick
        # hovering a profile chart: a tick across the intersection line at that position, burnt into both slices
        self._hover = None                                   # (chart slot, t along the line) or None
        self._hover_mesh = pv.PolyData(np.zeros((8, 3)), lines=sv.burn_cells(2))
        self._hover_actor = pl.add_mesh(self._hover_mesh, color=sv.HOVER_COLOR, line_width=sv.HOVER_WIDTH,
                                        lighting=False, render=False, pickable=False)
        self._hover_actor.SetVisibility(False)

    def _line_pairs(self, df) -> list[tuple[dict, dict]]:
        if df is None or not len(df):
            return []
        by_key = {e["key"]: e for e in self.entries}
        pairs = []
        for a, sa, b, sb in zip(df["image_a"], df["slice_a"], df["image_b"], df["slice_b"]):
            ea, eb = by_key.get((str(a), int(sa))), by_key.get((str(b), int(sb)))
            if ea is not None and eb is not None:
                pairs.append((ea, eb))
        return pairs

    # --- which slices are shown ---------------------------------------------------------
    def _sel_shown(self) -> bool:
        return id(self.sel) in getattr(self, "_vis", ())

    def _visible(self) -> tuple[set[int], set[int]]:
        """(ids of the shown entries, ids of the ghosts): in every shown stack its current slice (see _sync_positions)
        and its +-w neighbours, which are ghosts instead (a thin border in the stack colour, no image) when the stack's
        ghost button is on."""
        shown, ghosts = set(), set()
        for st in self.stacks:
            if not st["on"]:
                continue
            for e in st["entries"]:
                k = abs(e["pos"] - st["pos"])
                if k == 0 or (k <= st["w"] and not st["ghost"]):
                    shown.add(id(e))
                elif k <= st["w"]:
                    ghosts.add(id(e))
        return shown, ghosts

    def _stacks_from_selected(self) -> list[int]:
        """Stack indices starting with the selected slice's stack, then the following ones (cyclically)."""
        s0, n = self.sel["stack"], len(self.stacks)
        return [(s0 + i) % n for i in range(n)]

    def _sync_positions(self):
        """After the selection changed: the selected stack's position is the selected slice, every other stack's the
        slice whose plane passes closest to the selected slice's centre (each stack's slider scrolls it afterwards).
        Matched on the acquired geometry, where a stack's slices are parallel and ordered: the corrected poses move
        slices by more than the slice spacing, so the nearest one there would flip between neighbours while scrolling."""
        if self.sel is self._synced_sel:
            return
        self._synced_sel = self.sel
        c = self._centre(self.sel, self.sel["affine0"])
        for s, st in enumerate(self.stacks):
            if s == self.sel["stack"]:
                st["pos"] = self.sel["pos"]
            else:
                dist = [abs(float(frame_of(e["affine0"])[:, 2] @ (c - e["affine0"][:3, 3]))) for e in st["entries"]]
                st["pos"] = int(np.argmin(dist))
            if self._ui_ready:
                self._pos_sliders[s].GetRepresentation().SetValue(st["pos"])

    def _apply_visibility(self):
        if not self.stacks[self.sel["stack"]]["on"]:          # its stack was switched off: select the current slice
            nxt = next((s for s in self._stacks_from_selected() if self.stacks[s]["on"]), None)
            if nxt is not None:                               # of the next shown stack, every stack keeping its slice
                st = self.stacks[nxt]
                self.sel = self._synced_sel = st["entries"][st["pos"]]
        self._sync_positions()
        vis, ghosts = self._visible()
        self._vis = vis
        self._update_selected()
        for e in self.entries:
            shown = id(e) in vis
            if shown and e["tex_key"] != (self.frame, self._display_sigma()):
                self._set_texture(e)
            e["actor"].SetVisibility(shown)
            e["outline"].SetVisibility(shown and self.show_outlines)
            e["ghost"].SetVisibility(id(e) in ghosts)
        self._update_lines()
        self._update_nb_lines()
        self._update_mask()
        self._update_profiles()
        self._update_texts()

    # --- default disc mask (--center-mask-radius-mm) ---------------------------------------
    def _dilation(self, step: int) -> float | None:
        """Mask dilation (mm) of the loss at `step`, as motion_gradient.annealed_stages: linear from
        --mask-dilation-start-mm to --mask-dilation-mm over the first --mask-anneal-frac of the run in --anneal-stages
        discrete stages, nominal afterwards. None without the run's config."""
        cfg = self.config
        if cfg.get("mask_dilation_mm") is None:
            return None
        nominal, d0 = float(cfg["mask_dilation_mm"]), cfg.get("mask_dilation_start_mm")
        steps = max(1, int(cfg.get("steps", 1)))
        anneal_end = int(round(float(cfg.get("mask_anneal_frac", 0.8)) * steps)) if d0 is not None else 0
        if anneal_end <= 0 or step >= anneal_end:
            return nominal
        n_st = max(1, int(cfg.get("anneal_stages", 6)))
        starts = sorted({int(round(k / n_st * anneal_end)) for k in range(n_st)})
        k = max(i for i, s in enumerate(starts) if step >= s)
        return float(d0) + (nominal - float(d0)) * k / n_st

    def _mask_radius(self, snap: dict | None = None) -> float | None:
        """Radius (mm) of the disc mask at the shown step (or `snap`'s): --center-mask-radius-mm + the annealed
        dilation; None if the run used no disc mask."""
        snap = self.snapshots[self.snap_index] if snap is None else snap
        dil = self._dilation(snap["step"])
        return None if self.mask_r0 <= 0 or dil is None else self.mask_r0 + dil

    def _in_mask(self, e: dict, pts: np.ndarray, radius: float, snap: dict | None = None) -> np.ndarray:
        """Whether world points pts (N, 3) lie inside slice e's disc mask at its shown pose (or `snap`'s)."""
        Ai = np.linalg.inv(self._pose(e, snap))
        vox = pts @ Ai[:3, :3].T + Ai[:3, 3]
        (ci, cj), (su, sv_) = e["mask_c"], e["mask_sp"]
        return np.hypot((vox[:, 0] - ci) * su, (vox[:, 1] - cj) * sv_) <= radius

    def _update_mask(self):
        """The disc mask of every shown slice at the shown step, burnt into the slice (visible from both sides)."""
        radius = self._mask_radius()
        if not self.show_mask or radius is None:
            self.pl.remove_actor("mask_circles", render=False)
            return
        th = np.linspace(0.0, 2.0 * np.pi, MASK_CIRCLE_SAMPLES + 1)
        blocks = []
        for e in self.entries:
            if id(e) not in self._vis:
                continue
            nx, ny = e["shape"]
            (ci, cj), (su, sv_) = e["mask_c"], e["mask_sp"]
            i, j = ci + radius / su * np.cos(th), cj + radius / sv_ * np.sin(th)
            inside = (i >= -0.5) & (i <= nx - 0.5) & (j >= -0.5) & (j <= ny - 0.5)
            keep = inside[:-1] & inside[1:]                  # only the part of the circle inside the image
            if not keep.any():
                continue
            A = self._pose(e)
            w = sv.voxel_to_world(A, np.column_stack([i, j, np.zeros_like(i)]))
            blocks.append(sv.burn_points(np.stack([w[:-1][keep], w[1:][keep]], axis=1), A))
        if not blocks:
            self.pl.remove_actor("mask_circles", render=False)
            return
        pts = np.concatenate(blocks)
        mesh = pv.PolyData(pts, lines=sv.burn_cells(len(pts) // 4))
        self.pl.add_mesh(mesh, color=sv.MASK_CONTOUR_COLOR, line_width=sv.MASK_CONTOUR_WIDTH, name="mask_circles",
                         render=False, pickable=False)

    def _set_texture(self, e: dict):
        """Slice e's image at the shown frame -- blurred as the loss saw it at the shown step with the blur toggle on."""
        sigma = self._display_sigma()
        grey = (e["grey"][:, :, self.frame] if sigma <= 0
                else np.clip(self._frame_image(e, sigma) + 0.5, 0, 255).astype(np.uint8))
        e["img"].point_data["grey"][:] = grey.ravel(order="F")
        e["img"].Modified()
        e["tex_key"] = (self.frame, sigma)

    # --- the loss's view: in-plane blur and PSF band at a step (as motion_gradient.align / LineSampler) -----------
    @staticmethod
    def _config_thickness(loaded: dict) -> float:
        """Slice thickness (mm) for stacks whose headers have none: the run's --thickness."""
        return float((loaded.get("config") or {}).get("thickness") or 1.0)

    def _loss_state(self, step: int) -> tuple[float, float]:
        """(PSF scale, in-plane blur sigma in mm) the optimizer used at `step`: the blur decays linearly from
        --blur-sigma to --blur-sigma-end over --blur-frac of the steps (constant with --blur-frac 0), rounded up to
        BLUR_QUANTUM_MM; the PSF scale from --psf-scale-start to 1 over --psf-anneal-frac."""
        cfg = self.config
        steps = max(1, int(cfg.get("steps", 1) or 1))
        psf_s0 = 1.0 if cfg.get("no_psf") else float(cfg.get("psf_scale_start", 1.0) or 1.0)
        psf_steps = int(round(float(cfg.get("psf_anneal_frac", 0.6)) * steps)) if psf_s0 > 1.0 else 0
        scale = psf_s0 + (1.0 - psf_s0) * step / psf_steps if step < psf_steps else 1.0
        b0 = float(cfg.get("blur_sigma", 0.0) or 0.0)
        b_end = min(float(cfg.get("blur_sigma_end", 0.0) or 0.0), b0)
        b_frac = float(cfg.get("blur_frac", 0.6))
        if b0 > 0 and b_frac <= 0:
            sigma = b0
        else:
            b_steps = int(round(b_frac * steps)) if b0 > 0 else 0
            sigma = b_end + (b0 - b_end) * (1.0 - step / b_steps) if step < b_steps else b_end
        sigma = math.ceil(sigma / BLUR_QUANTUM_MM - 1e-9) * BLUR_QUANTUM_MM if sigma > 0 else 0.0
        return scale, round(sigma, 4)

    def _display_sigma(self, snap: dict | None = None) -> float:
        """In-plane blur (mm) of the images (slices and profiles): the optimizer's at the shown step (or `snap`'s),
        with the blur toggle on; else 0."""
        if not (self.show_blur and self.config and getattr(self, "snapshots", None)):
            return 0.0
        snap = self.snapshots[self.snap_index] if snap is None else snap
        return self._loss_state(snap["step"])[1]

    def _band(self, scale: float) -> tuple[np.ndarray, np.ndarray]:
        """PSF band offsets u (units of the other slice's thickness) and weights at PSF scale `scale`
        (LineSampler.u_img / band_weights / slice_profile)."""
        cfg = self.config
        n_band = 1 if cfg.get("no_psf") else int(cfg.get("n_band", 1) or 1)
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

    def _frame_image(self, e: dict, sigma: float) -> np.ndarray:
        """Slice e's grey image (0-255, float) at the shown frame, in-plane Gaussian blurred with sigma mm in its own
        pixel spacing, edges replicated (ImageVolume._blur); cached."""
        key = (id(e), self.frame, sigma)
        img = self._blur_cache.get(key)
        if img is None:
            img = e["grey"][:, :, self.frame].astype(np.float32)
            if sigma > 1e-6:
                su, sv_ = e["mask_sp"]                   # pixel spacing (mm) along i, j
                img = gaussian_filter(img, (max(sigma / su, 1e-3), max(sigma / sv_, 1e-3)), mode="nearest",
                                      truncate=3.0)
            if len(self._blur_cache) >= BLUR_CACHE_MAX:
                self._blur_cache.clear()
            self._blur_cache[key] = img
        return img

    def _loss_profile(self, e: dict, other: dict, pts: np.ndarray, snap: dict, band: bool) -> np.ndarray:
        """Slice e's grey values along an intersection with `other` as the loss sees them at `snap`: its image blurred
        as at that step (blur toggle) and averaged over the PSF band across other's slab (PSF toggle, `band`): offsets
        along the in-plane direction of e that crosses other, u T_other / sin(crossing angle), weighted by the slice
        profile at that step's PSF scale (LineSampler.profiles). NaN outside e's image."""
        A = self._pose(e, snap)
        Ai = np.linalg.inv(A)
        img = self._frame_image(e, self._display_sigma(snap))
        scale = self._loss_state(snap["step"])[0] if self.config else 1.0
        u, w = self._band(scale) if (band and self.show_psf and self.config) else (np.zeros(1), np.ones(1))
        if len(u) > 1:
            n_e, n_o = frame_of(A)[:, 2], frame_of(self._pose(other, snap))[:, 2]
            m = n_o - (n_o @ n_e) * n_e
            sin = max(float(np.linalg.norm(m)), 1e-3)
            m, step = m / sin, other["thickness"] / sin
        else:
            m, step = np.zeros(3), 0.0
        prof = np.zeros(len(pts))
        for uk, wk in zip(u, w):
            P = pts + uk * step * m
            vox = (Ai[:3, :3] @ P.T) + Ai[:3, 3:4]
            prof += wk * map_coordinates(img, vox[:2], order=1, mode="constant", cval=0.0)   # zero padding (grid_sample)
        vox = (Ai[:3, :3] @ pts.T) + Ai[:3, 3:4]
        nx, ny = e["shape"]
        inside = (vox[0] >= 0) & (vox[0] <= nx - 1) & (vox[1] >= 0) & (vox[1] <= ny - 1)
        return np.where(inside, prof, np.nan)

    def _loss_view_text(self, snap: dict, band: bool) -> str:
        """Chart-title note of the loss view in effect: PSF scale (band lines) and blur."""
        if not self.config:
            return ""
        scale, sigma = self._loss_state(snap["step"])
        parts = []
        if band and self.show_psf and len(self._band(scale)[0]) > 1:
            parts.append(f"psf x{scale:.1f}")
        if self.show_blur:
            parts.append(f"blur {sigma:.1f}")
        return ("   " + "  ".join(parts)) if parts else ""

    # --- snapshot / frame ----------------------------------------------------------------
    def _apply_snapshot(self):
        for e in self.entries:
            nx, ny = e["shape"]
            corners = sv.voxel_to_world(self._pose(e), sv.corner_ijk(nx, ny))
            for view, vtk_points in e["point_views"]:
                view[:] = corners
                vtk_points.Modified()
            view, vtk_array = e["metric_view"]
            view[:] = self._metric_value(e)
            vtk_array.Modified()
        self._apply_visibility()

    @staticmethod
    def _corner_cells(n: int) -> np.ndarray:
        """Line cells of n slices' corner marks: 8 two-point segments per slice."""
        return np.column_stack([np.full(8 * n, 2), np.arange(0, 16 * n, 2), np.arange(1, 16 * n, 2)]).ravel()

    def _corner_marks(self, e: dict) -> np.ndarray:
        """(16, 3) segment end points of slice e's corner marks at the shown step: at each corner two segments along
        its adjacent edges, CORNER_FRAC of their length."""
        nx, ny = e["shape"]
        c = sv.voxel_to_world(self._pose(e), sv.corner_ijk(nx, ny))          # (4, 3), in order around the slice
        return np.array([p for i in range(4) for j in ((i + 1) % 4, (i - 1) % 4)
                         for p in (c[i], c[i] + CORNER_FRAC * (c[j] - c[i]))])

    @staticmethod
    def _front(actor):
        """Draw a line actor in front of coincident lines (the metric outlines) instead of z-fighting with them."""
        m = actor.GetMapper()
        m.SetResolveCoincidentTopologyToPolygonOffset()
        m.SetRelativeCoincidentTopologyLineOffsetParameters(0, CORNER_OFFSET_UNITS)

    def _update_corners(self):
        """Corner marks of every shown slice but the selected one, in their stack colours (one mesh)."""
        ents = [e for e in self.entries if id(e) in self._vis and e is not self.sel]
        if not ents:
            self.pl.remove_actor("slice_corners", render=False)
            return
        mesh = pv.PolyData(np.concatenate([self._corner_marks(e) for e in ents]), lines=self._corner_cells(len(ents)))
        rgb = np.array([pv.Color(st["color"]).int_rgb for st in self.stacks], np.uint8)
        mesh.cell_data["rgb"] = np.repeat(rgb[[e["stack"] for e in ents]], 8, axis=0)
        actor = self.pl.add_mesh(mesh, scalars="rgb", rgb=True, line_width=CORNER_WIDTH, lighting=False,
                                 name="slice_corners", render=False, pickable=False)
        self._front(actor)

    def _update_selected(self):
        """Corner marks (stack colour) and metric outline of the selected slice, both thicker than the others'."""
        e = self.sel
        self._sel_mesh.points = self._corner_marks(e)
        self._sel_actor.prop.color = self.stacks[e["stack"]]["color"]
        self._sel_actor.SetVisibility(self._sel_shown())
        self._update_corners()
        if self._sel_outlined is not e:
            if self._sel_outlined is not None:
                self._sel_outlined["outline"].prop.line_width = OUTLINE_WIDTH
            e["outline"].prop.line_width = SEL_OUTLINE_WIDTH
            self._sel_outlined = e

    def _max_motion_mm(self) -> float:
        """Largest displacement of a slice centre from its acquired pose at the shown step (cached per snapshot)."""
        if self.snap_index not in self._max_motion:
            snap, snap0 = self.snapshots[self.snap_index], self.snapshots[0]
            self._max_motion[self.snap_index] = max(
                float(np.linalg.norm(self._centre(e, self._pose(e, snap)) - self._centre(e, self._pose(e, snap0))))
                for e in self.entries)
        return self._max_motion[self.snap_index]

    def set_snapshot(self, index: int, from_slider: bool = False):
        index = int(np.clip(index, 0, len(self.snapshots) - 1))
        if index == self.snap_index:
            return
        self.snap_index = index
        if not from_slider and self._hist_slider is not None:
            self._hist_slider.GetRepresentation().SetValue(self.snapshots[index]["step"])
        self._apply_snapshot()
        self.pl.render()

    def _on_history_slider(self, value: float):
        """The history slider runs over optimizer steps: go to the snapshot nearest to the dragged step and snap the
        knob onto it (a regular snapshot before the best one recorded at the same step)."""
        if not self._ui_ready:                           # pyvista calls slider callbacks once while creating them
            return
        steps = np.array([s["step"] for s in self.snapshots])
        index = int(np.argmin(np.abs(steps - value)))
        self._hist_slider.GetRepresentation().SetValue(steps[index])
        self.set_snapshot(index, from_slider=True)

    def set_frame(self, frame: int, from_slider: bool = False):
        frame = int(frame) % self.n_frames
        if frame == self.frame:
            return
        self.frame = frame
        if not from_slider and self._frame_slider is not None:
            self._frame_slider.GetRepresentation().SetValue(frame + 1)
        for e in self.entries:
            if id(e) in self._vis:
                self._set_texture(e)
        self._update_profiles()
        self._update_texts()
        self.pl.render()

    # --- metric ---------------------------------------------------------------------------
    def _metric_value(self, e: dict, snap: dict | None = None) -> float:
        snap = self.snapshots[self.snap_index] if snap is None else snap
        if self.metric == "motion":
            A0 = self.snapshots[0]["affines"].get(e["key"], e["affine0"])
            return float(np.linalg.norm(self._centre(e, self._pose(e, snap)) - self._centre(e, A0)))
        return float(snap["metrics"].get(e["key"], {}).get(METRICS[self.metric]["col"], np.nan))

    def _metric_range(self) -> tuple[float, float]:
        if self.metric == "motion":
            if self._motion_range is None:
                hi = max(self._metric_value(e, s) for s in self.snapshots for e in self.entries)
                self._motion_range = (0.0, max(hi, 0.1))
            return self._motion_range
        lo, hi = self.history.get("metric_range", {}).get(METRICS[self.metric]["col"], METRICS[self.metric]["bounds"])
        return (lo, hi) if hi - lo > 1e-6 else (lo - 0.5, hi + 0.5)

    def _color_range(self) -> list[float]:
        """Outline colour range of the active metric: initially its range over the history (clipped to the metric's
        bounds), then whatever the colour min / max sliders set; kept per metric while the subject is shown."""
        if self.metric not in self.color_range:
            b_lo, b_hi = METRICS[self.metric]["bounds"]
            lo, hi = self._metric_range()
            lo, hi = float(np.clip(lo, b_lo, b_hi)), float(np.clip(hi, b_lo, b_hi))
            if hi - lo < RANGE_MIN_GAP:
                lo, hi = max(b_lo, hi - RANGE_MIN_GAP), min(b_hi, lo + RANGE_MIN_GAP)
            self.color_range[self.metric] = [lo, hi]
        return self.color_range[self.metric]

    def _configure_metric(self):
        spec = METRICS[self.metric]
        lo, hi = self._color_range()
        self._lut.cmap = sv.OUTLINE_CMAP if spec["higher_better"] else sv.OUTLINE_CMAP + "_r"
        self._lut.scalar_range = (lo, hi)                # values outside are clamped to the end colours
        for e in self.entries:
            e["outline"].mapper.scalar_range = (lo, hi)
        for k, w in self._metric_radios.items():
            w.GetRepresentation().SetState(int(k == self.metric))
        if self._bar_key is not None:
            self._bar_key.SetInput(f"outline colour: {spec['label']}")
        self._sync_range_sliders()

    # --- colour-range sliders (as slice_viewer_3d) -------------------------------------------------------
    def _add_range_sliders(self):
        """The colour max / min sliders right above / below the colour bar. They carry no built-in label; their
        values are small texts on the right."""
        b_lo, b_hi = METRICS[self.metric]["bounds"]
        lo, hi = self._color_range()
        for which, value, y in (("hi", hi, SLIDER_HI_Y), ("lo", lo, SLIDER_LO_Y)):
            widget = self.pl.add_slider_widget(lambda v, k=which: self._on_range_slider(k, v), rng=[b_lo, b_hi],
                                               value=value, title="", pointa=(BAR_X0, y), pointb=(BAR_X1, y),
                                               style="modern", color="#bbbbbb", slider_width=0.008, tube_width=0.002,
                                               interaction_event="always")
            rep = widget.GetRepresentation()
            rep.ShowSliderLabelOff()
            rep.SetTitleHeight(0.0)
            self._range_sliders[which] = widget
            self._range_text[which] = self._text("", int(RANGE_TEXT_X * self.W), int((y - 0.006) * self.H), 6,
                                                 "#bbbbbb")

    def _sync_range_sliders(self):
        """Reset both sliders to the active metric's bounds and current colour range."""
        if not self._range_sliders:
            return
        spec = METRICS[self.metric]
        b_lo, b_hi = spec["bounds"]
        lo, hi = self._color_range()
        for which, value in (("lo", lo), ("hi", hi)):
            rep = self._range_sliders[which].GetRepresentation()
            rep.SetMinimumValue(b_lo)
            rep.SetMaximumValue(b_hi)
            rep.SetValue(value)
            self._range_text[which].SetInput(f"{'max' if which == 'hi' else 'min'}  {spec['fmt'].format(value)}")

    def _on_range_slider(self, which: str, value: float):
        if not self._ui_ready:                           # (slider callbacks fire once on creation)
            return
        lo, hi = self._color_range()
        if which == "lo":
            lo = min(float(value), hi - RANGE_MIN_GAP)
        else:
            hi = max(float(value), lo + RANGE_MIN_GAP)
        self.set_color_range(lo, hi)

    def set_color_range(self, lo: float, hi: float):
        b_lo, b_hi = METRICS[self.metric]["bounds"]
        lo = float(np.clip(lo, b_lo, b_hi - RANGE_MIN_GAP))
        hi = float(np.clip(hi, lo + RANGE_MIN_GAP, b_hi))
        if [lo, hi] == self._color_range():
            self._sync_range_sliders()
            return
        self.color_range[self.metric] = [lo, hi]
        self._configure_metric()
        self.pl.render()

    def set_metric(self, key: str):
        if not self._ui_ready:                           # (radio callbacks may fire while the widgets are created)
            return
        self.metric = key
        self._configure_metric()
        self._apply_snapshot()
        self.pl.render()

    def cycle_metric(self):
        keys = list(METRICS)
        self.set_metric(keys[(keys.index(self.metric) + 1) % len(keys)])

    # --- overlays ---------------------------------------------------------------------------
    def _update_lines(self):
        if not self.show_lines or not self._lines_pairs:
            self.pl.remove_actor("sampling_lines", render=False)
            return
        # each line lies exactly where its two opaque slices meet, so (as in slice_viewer_3d) it is burnt into both:
        # drawn just in front of and behind each slice, visible from either side
        blocks = []
        for ea, eb in self._lines_pairs:
            if id(ea) not in self._vis or id(eb) not in self._vis:
                continue
            Aa, Ab = self._pose(ea), self._pose(eb)
            c = sv.crossing_segment(Aa, ea["shape"], Ab, eb["shape"])
            if c is not None:
                x0, d, lo, hi = c
                seg = np.array([[x0 + lo * d, x0 + hi * d]])
                blocks += [sv.burn_points(seg, Aa), sv.burn_points(seg, Ab)]
            if len(blocks) >= 2 * MAX_LINES:
                break
        if not blocks:
            self.pl.remove_actor("sampling_lines", render=False)
            return
        pts = np.concatenate(blocks)
        mesh = pv.PolyData(pts, lines=sv.burn_cells(len(pts) // 4))
        self.pl.add_mesh(mesh, color=LINE_COLOR, line_width=2, name="sampling_lines", render=False, pickable=False)

    def _update_nb_lines(self):
        """Neighbour crossings (as slice_viewer_3d's SA-SA lines, for every stack): each shown slice's crossing with
        the slice before and after it in its stack at the shown step, burnt into the shown slice; thin over the part
        inside both images, thick within --sax-sax-radius-mm of the stack axis (the part --sax-sax-weight weighs
        fully). Parallel neighbours do not cross: nothing is drawn."""
        blocks = {"img": [], "used": []}
        if self.show_nb_lines:
            for e in self.entries:
                if id(e) not in self._vis:
                    continue
                A = self._pose(e)
                for nb in self._neighbours(e):
                    if nb is None:
                        continue
                    c = sv.crossing_segment(A, e["shape"], self._pose(nb), nb["shape"])
                    if c is None:
                        continue
                    x0, d, lo, hi = c
                    blocks["img"].append(sv.burn_points(np.array([[x0 + lo * d, x0 + hi * d]]), A))
                    r = self._radius_range(e, x0, d, lo, hi)
                    if r is not None:
                        blocks["used"].append(sv.burn_points(np.array([[x0 + r[0] * d, x0 + r[1] * d]]), A))
        for name, width, opacity in (("img", NB_LINE_IMG_WIDTH, 0.6), ("used", NB_LINE_USED_WIDTH, 1.0)):
            if not blocks[name]:
                self.pl.remove_actor(f"nb_lines_{name}", render=False)
                continue
            pts = np.concatenate(blocks[name])
            self.pl.add_mesh(pv.PolyData(pts, lines=sv.burn_cells(len(pts) // 4)), color=NB_LINE_COLOR,
                             line_width=width, opacity=opacity, lighting=False, name=f"nb_lines_{name}", render=False,
                             pickable=False)

    def _partners(self) -> list[dict]:
        """The slices the selected one is profiled against, one per other shown stack: the stack's current slice (the
        centre of its window) if it crosses the selected slice, else its shown slice crossing the selected one
        closest to its centre."""
        a = self.sel
        Aa = self._pose(a)
        c = self._centre(a, Aa)
        out = []
        for st in self.stacks:
            if not st["on"] or st is self.stacks[a["stack"]]:
                continue
            centre = st["entries"][st["pos"]]
            if (id(centre) in self._vis
                    and sv.crossing_segment(Aa, a["shape"], self._pose(centre), centre["shape"]) is not None):
                out.append(centre)
                continue
            best = None
            for e in st["entries"]:
                if id(e) not in self._vis:
                    continue
                seg = sv.crossing_segment(Aa, a["shape"], self._pose(e), e["shape"])
                if seg is None:
                    continue
                x0, d, _lo, _hi = seg
                dist = float(np.linalg.norm((c - x0) - float((c - x0) @ d) * d))
                if best is None or dist < best[0]:
                    best = (dist, e)
            if best is not None:
                out.append(best[1])
        return out[:PROFILE_SLOTS]

    def _neighbours(self, e: dict) -> list[dict | None]:
        """The slices before and after e in its stack (None at either end)."""
        ents = self.stacks[e["stack"]]["entries"]
        p = e["pos"]
        return [ents[p - 1] if p > 0 else None, ents[p + 1] if p + 1 < len(ents) else None]

    def _nb_radius(self) -> float:
        """--sax-sax-radius-mm of the run: the neighbour crossings' supervised part lies within it of the stack axis."""
        return float(self.config.get("sax_sax_radius_mm") or 120.0)

    def _in_radius(self, e: dict, pts: np.ndarray) -> np.ndarray:
        """Whether world points pts (N, 3) lie within --sax-sax-radius-mm of slice e's stack axis."""
        c, n = self.stacks[e["stack"]]["axis"]
        rel = pts - c
        perp = rel - (rel @ n)[:, None] * n
        return np.einsum("ij,ij->i", perp, perp) <= self._nb_radius() ** 2

    def _radius_range(self, e: dict, x0, d, lo: float, hi: float) -> tuple[float, float] | None:
        """The part [u1, u2] of the segment x0 + t d, t in [lo, hi], within --sax-sax-radius-mm of e's stack axis."""
        c, n = self.stacks[e["stack"]]["axis"]
        a, b = d - (d @ n) * n, (x0 - c) - ((x0 - c) @ n) * n
        A2, B2, C2 = a @ a, 2 * a @ b, b @ b - self._nb_radius() ** 2
        disc = B2 * B2 - 4 * A2 * C2
        if A2 < 1e-12:                                   # line parallel to the axis: inside throughout or not at all
            return (lo, hi) if C2 <= 0 else None
        if disc <= 0:
            return None
        u1 = max(lo, (-B2 - np.sqrt(disc)) / (2 * A2))
        u2 = min(hi, (-B2 + np.sqrt(disc)) / (2 * A2))
        return (u1, u2) if u2 > u1 else None

    def _line_data(self, other: dict, snap: dict, neighbour: bool = False) -> dict | None:
        """The selected slice x `other` at `snap` (shown frame): their crossing, both profiles, the loss points (inside
        both images and either slice's disc mask; for a neighbour of the same stack, `neighbour`, within
        --sax-sax-radius-mm of the stack axis instead, the part --sax-sax-weight weighs fully) and the outline metric's
        value and per-point curve; None if the two slices do not cross."""
        a = self.sel
        seg = sv.crossing_segment(self._pose(a, snap), a["shape"], self._pose(other, snap), other["shape"])
        if seg is None:
            return None
        x0, d, lo, hi = seg
        t = np.arange(lo, hi + 1e-9, PROFILE_STEP_MM)
        pts = x0[None] + t[:, None] * d[None]
        # as the loss sees them (blur / PSF toggles); neighbour crossings are sampled on the line, without a PSF band
        # (motion_gradient.SaxNeighbourLines)
        ya = self._loss_profile(a, other, pts, snap, band=not neighbour)
        yb = self._loss_profile(other, a, pts, snap, band=not neighbour)
        radius = None if neighbour else self._mask_radius(snap)
        if neighbour:
            in_mask = self._in_radius(a, pts)
        else:
            in_mask = (self._in_mask(a, pts, radius, snap) | self._in_mask(other, pts, radius, snap)
                       if radius is not None else np.ones(len(t), bool))
        used = np.isfinite(ya) & np.isfinite(yb) & in_mask
        mname, mval, ct, curve, full = self._profile_metric(t, ya, yb, used)
        return dict(other=other, x0=x0, d=d, lo=lo, hi=hi, t=t, pts=pts, ya=ya, yb=yb, in_mask=in_mask, used=used,
                    masked=neighbour or radius is not None, neighbour=neighbour, mname=mname, mval=mval, ct=ct,
                    curve=curve, full=full, view_txt=self._loss_view_text(snap, band=not neighbour))

    def _profile_metric(self, t: np.ndarray, ya: np.ndarray, yb: np.ndarray, used: np.ndarray):
        """The outline metric of one intersection and its per-point curves, as slice_viewer_3d: (label, value or None,
        curve t, loss curve or None, full curve or None), resampled to the run's --line-step-mm, shown frame only. The
        value and the loss curve are computed on the loss points (NaN elsewhere); the full curve is the same metric
        over the whole intersection inside both images (the metric without the mask), equal to the loss curve on the
        loss points.
          NCC   z_a(t) z_b(t) of the profiles z-scored with the loss points' mean / std -- their mean over the loss
                points is the NCC (no loss points: z-scored over the whole intersection)
          LNCC  the local NCC of the window centred at t -- their weighted mean is the LNCC
          NMI   value only, no per-point curve;  motion: per slice, not per line -> the NCC is shown instead"""
        metric = self.metric if self.metric in ("ncc", "lncc", "nmi") else "ncc"
        label = METRICS[metric]["label"]
        step = float(self.config.get("line_step_mm", PROFILE_STEP_MM))
        tg = np.arange(np.ceil(t[0] / step) * step, t[-1] + 1e-6, step)
        none = (label, None, tg, None, None)
        if len(tg) < 3:
            return none
        interp = lambda y: np.interp(tg, t, np.where(np.isfinite(y), y, np.nan))   # noqa: E731
        a, b = interp(ya), interp(yb)
        v = np.isfinite(a) & np.isfinite(b)                       # inside both images
        w = (np.interp(tg, t, used.astype(float)) > 0.5) & v        # loss points
        if v.sum() < 3:
            return none
        has_loss = w.sum() >= 3
        a, b = np.where(v, a, 0.0), np.where(v, b, 0.0)
        wf, vf = w.astype(float), v.astype(float)
        if metric == "ncc":
            sf = wf if has_loss else vf                             # z-scored with the loss points' statistics
            ss = sf.sum()
            za = a - (sf * a).sum() / ss
            za = za / max(np.sqrt((sf * za * za).sum() / ss), 1e-9)
            zb = b - (sf * b).sum() / ss
            zb = zb / max(np.sqrt((sf * zb * zb).sum() / ss), 1e-9)
            full = np.where(v, za * zb, np.nan)
            if not has_loss:
                return label, None, tg, None, full
            return label, sv.np_ncc(a, b, wf), tg, np.where(w, full, np.nan), full
        if metric == "lncc":
            k = max(3, int(round(float(self.config.get("lncc_window_mm", 20.0)) / step)))
            min_std = float(self.config.get("lncc_min_std", 0.25))
            centre = np.arange(len(tg) - k + 1) + (k - 1) // 2

            def local(weights):
                value, loc, wt = sv.np_lncc(a, b, weights, k, min_std, return_windows=True)
                curve = np.full(len(tg), np.nan)
                curve[centre[:len(loc)]] = np.where(wt > 0, loc, np.nan)
                return value, curve

            _, full = local(vf)
            if not has_loss:
                return label, None, tg, None, full
            value, curve = local(wf)
            return label, value, tg, curve, np.where(w & np.isfinite(curve), curve, full)
        if not has_loss:
            return none
        return label, sv.np_nmi(a, b, wf, int(self.config.get("nmi_bins", 24)),
                                float(self.config.get("nmi_sigma", 1.0))), tg, None, None

    def _metric_ranges(self, specs: list, data: list[dict | None]) -> list:
        """Per chart slot: (min, max) of the full metric curve over every history snapshot (at the frame shown when
        computed), so the metric charts keep one y scale while scrubbing the history (as slice_viewer_3d); widened when
        the shown curve leaves it. Recomputed when the selection, its partners / neighbours or the metric change.
        specs: per slot (other slice, neighbour?, side) or None."""
        key = (self.subject_index, id(self.sel),
               tuple(None if s is None or s[0] is None else (id(s[0]), s[1]) for s in specs), self.metric,
               self.show_blur, self.show_psf)
        if getattr(self, "_range_cache", (None,))[0] != key:
            lo, hi = np.full(len(specs), np.inf), np.full(len(specs), -np.inf)
            for snap in self.snapshots:
                for k, s in enumerate(specs):
                    if s is None or s[0] is None:
                        continue
                    c = self._line_data(s[0], snap, s[1])
                    cv = None if c is None else c["full"]
                    if cv is not None and np.isfinite(cv).any():
                        lo[k], hi[k] = min(lo[k], float(np.nanmin(cv))), max(hi[k], float(np.nanmax(cv)))
            self._range_cache = (key, lo, hi)
        _, lo, hi = self._range_cache
        for k, c in enumerate(data):
            cv = None if c is None else c["full"]
            if cv is not None and np.isfinite(cv).any():
                lo[k], hi[k] = min(lo[k], float(np.nanmin(cv))), max(hi[k], float(np.nanmax(cv)))
        return [(float(l), float(h)) if np.isfinite(l) else None for l, h in zip(lo, hi)]

    def _update_profiles(self):
        """Intensity profile charts (top block, right) and, below them, the per-point metric charts of the selected
        slice against each partner (_partners), at the shown step and frame."""
        charts, mcharts = getattr(self, "_charts", []), getattr(self, "_mcharts", [])
        for slot in charts + mcharts:
            slot["item"] = None
        if not (self.show_profiles or self.show_metric_profiles) or not self._sel_shown():
            for slot in charts + mcharts:
                slot["chart"].visible = False
            self.pl.remove_actor("partner_lines", render=False)
            self._draw_hover()
            return
        snap = self.snapshots[self.snap_index]
        # slots: the other stacks' partners (unused slots hidden), then the selected slice's neighbours before / after
        # in its own stack (always shown; "no crossing" while they are parallel)
        partners = [c for c in (self._line_data(e, snap) for e in self._partners()) if c is not None]
        specs, data = [], []
        for k in range(self._n_partner_slots):
            c = partners[k] if k < len(partners) else None
            specs.append(None if c is None else (c["other"], False, None))
            data.append(c)
        for side, nb in zip(("before", "after"), self._neighbours(self.sel)):
            specs.append((nb, True, side))
            data.append(None if nb is None else self._line_data(nb, snap, neighbour=True))
        ranges = self._metric_ranges(specs, data) if self.show_metric_profiles else [None] * len(specs)
        lab = f"{self.labels[self.sel['stack']]} [{self.sel['index']}]"
        nan2, x01 = np.array([np.nan, np.nan]), np.array([0.0, 1.0])
        segs, colors = [], []
        for k, (ch, mch) in enumerate(zip(charts, mcharts)):
            spec = specs[k] if k < len(specs) else None
            if spec is None:
                ch["chart"].visible = mch["chart"].visible = False
                continue
            c = data[k]
            if c is None:                                # a neighbour that does not cross (or none at this end)
                nb, _, side = spec
                why = f"no slice {side}" if nb is None else "no crossing (parallel)"
                title = f"{lab} x {'-' if nb is None else '[%d]' % nb['index']} ({side}): {why}"
                for key, pl_ in ch["plots"].items():
                    if key != "cursor":
                        pl_.update(x01, nan2)
                ch["chart"].x_range = [0.0, 1.0]
                ch["chart"].title = title
                self._fill_metric_chart(mch, title, None, None, None, [0.0, 1.0], None, None)
                ch["chart"].visible, mch["chart"].visible = self.show_profiles, self.show_metric_profiles
                continue
            e, t, ya, yb, used = c["other"], c["t"], c["ya"], c["yb"], c["used"]
            p = ch["plots"]
            for side, y in (("a", ya), ("b", yb)):
                p[side + "_img"].update(t, y)
                p[side + "_used"].update(t, np.where(used, y, np.nan))
            p["mask"].update(t, np.where(c["in_mask"], float(sv.PROFILE_MASK_Y), np.nan) if c["masked"]
                             else np.full(len(t), np.nan))
            color = self.stacks[e["stack"]]["color"]
            p["b_img"].color = p["b_used"].color = color
            xr = [float(t[0]), float(t[-1]) if len(t) > 1 else float(t[0]) + 1]
            ch["chart"].x_range = xr
            value = f"{c['mval']:.2f}" if c["mval"] is not None else "-"
            if c["neighbour"]:                           # thick / yellow: within --sax-sax-radius-mm of the stack axis
                w = float(self.config.get("sax_sax_weight") or 0.0)
                names = f"{lab} x [{e['index']}] ({spec[2]})"
                ch["chart"].title = (f"{names}   {c['mname']} {value}   r < {self._nb_radius():.0f} mm, "
                                     + (f"w {w:g}" if w > 0 else "unsupervised") + c["view_txt"])
            else:
                names = f"{lab} x {self.labels[e['stack']]} [{e['index']}]"
                ch["chart"].title = f"{names}   {c['mname']} {value}{' (mask)' if c['masked'] else ''}{c['view_txt']}"
            ch["chart"].visible = self.show_profiles
            item = dict(other=e, x0=c["x0"], d=c["d"], t=t, ya=ya, yb=yb, names=names, title=ch["chart"].title,
                        ct=c["ct"], curve=c["full"], mname=c["mname"])
            ch["item"] = mch["item"] = item
            item["mtitle"], item["mrange"] = self._fill_metric_chart(
                mch, f"{names}   {c['mname']} per point", c["ct"], c["curve"], c["full"], xr, ranges[k],
                (t, c["in_mask"]) if c["masked"] else None)
            mch["chart"].visible = self.show_metric_profiles
            if not c["neighbour"]:                       # (the neighbour crossings: _update_nb_lines)
                segs.append([c["x0"] + c["lo"] * c["d"], c["x0"] + c["hi"] * c["d"]])
                colors.append(pv.Color(self.stacks[e["stack"]]["color"]).int_rgb)
        if segs and self.show_lines:                     # the profiled crossings drawn on the selected slice
            seg = np.asarray(segs)
            mesh = pv.PolyData(sv.burn_points(seg, self._pose(self.sel)), lines=sv.burn_cells(len(segs)))
            mesh.cell_data["rgb"] = np.asarray(colors + colors, np.uint8)
            self.pl.add_mesh(mesh, scalars="rgb", rgb=True, line_width=3, name="partner_lines", render=False,
                             pickable=False)
        else:
            self.pl.remove_actor("partner_lines", render=False)
        self._draw_hover()

    def _fill_metric_chart(self, mslot: dict, title: str, t, curve, full, xr, rng, mask):
        """Draw the per-point metric curves of one intersection as its intensity profiles (as slice_viewer_3d): thin
        over the whole intersection (`full`), thick on the loss points (`curve`), and the disc mask range (`mask`:
        (t, in_mask), None without a mask) as a bar along the bottom; on the fixed y range `rng` (the full curve's
        min / max over the history, padded; empty chart when there is no curve, e.g. NMI). Returns the chart title and
        its y range."""
        ch, p = mslot["chart"], mslot["plots"]
        nan2 = np.array([np.nan, np.nan])
        if full is None or t is None or not np.isfinite(full).any():
            for key in ("full", "curve", "zero", "mask"):
                p[key].update(np.array([0.0, 1.0]), nan2)
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
        dec = 2 if span < 2 else 1 if span < 20 else 0           # tick rounding stays inside the padding
        if dmin < 0 < dmax:                                      # 0, plus the extremes not crowding it
            ticks = tuple(v for v in (dmin, 0.0, dmax) if v == 0.0 or abs(v) > 0.2 * span)
        else:
            ticks = (dmin, (dmin + dmax) / 2, dmax)
        ticks = tuple(dict.fromkeys(round(v, dec) for v in ticks))
        p["full"].update(t, full)
        p["curve"].update(t, curve if curve is not None else np.full(len(t), np.nan))
        p["zero"].update(np.array(xr, dtype=float), np.zeros(2) if lo < 0 < hi else nan2)
        if mask is not None and mask[1].any():                    # same height share as in the profile charts
            p["mask"].update(mask[0], np.where(mask[1], lo + sv.PROFILE_MASK_Y / 255.0 * (hi - lo), np.nan))
        else:
            p["mask"].update(np.array([0.0, 1.0]), nan2)
        ch.x_range = xr
        ch.y_range = [lo, hi]
        ch.y_axis.tick_locations = ticks
        ch.y_axis.tick_labels = tuple(f"{v:.{dec}f}" for v in ticks)
        ch.title = title
        return title, (lo, hi)

    # --- hover on the profile charts -> tick marks on the intersection in 3D (as slice_viewer_3d) -------------------
    def _on_mouse_move(self, *_):
        """Mouse over a profile or metric chart: remember the position along its intersection line (None elsewhere)."""
        if not getattr(self, "_built", False):
            return
        x, y = self.pl.iren.interactor.GetEventPosition()
        w, h = self.pl.window_size
        hover = None
        for k, slots in enumerate(zip(getattr(self, "_charts", []), getattr(self, "_mcharts", []))):
            for slot in slots:
                ch = slot["chart"]
                if slot["item"] is None or not ch.visible:
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
        """Cursor line in both charts of the hovered intersection, both grey values / the metric value at that position
        in their titles, and a tick across the intersection there in the selected slice and in the other slice."""
        charts, mcharts = getattr(self, "_charts", []), getattr(self, "_mcharts", [])
        for slots, key in ((charts, "title"), (mcharts, "mtitle")):
            for slot in slots:
                slot["plots"]["cursor"].visible = False
                if slot["item"] is not None:
                    slot["chart"].title = slot["item"][key]
        it = charts[self._hover[0]]["item"] if self._hover is not None and self._hover[0] < len(charts) else None
        if it is None:
            self._hover_actor.SetVisibility(False)
            return
        k, t = self._hover
        P = it["x0"] + t * it["d"]
        pts = []
        for e in (self.sel, it["other"]):
            A = self._pose(e)
            across = np.cross(frame_of(A)[:, 2], it["d"])
            across /= max(np.linalg.norm(across), 1e-9)
            tick = np.stack([P - 0.5 * sv.HOVER_TICK_MM * across, P + 0.5 * sv.HOVER_TICK_MM * across])[None]
            pts.append(sv.burn_points(tick, A))
        self._hover_mesh.points = np.concatenate(pts)
        self._hover_actor.SetVisibility(True)
        cur = charts[k]["plots"]["cursor"]
        cur.update(np.array([t, t]), np.array([0.0, 255.0]))
        cur.visible = True
        i = int(np.clip(np.rint((t - it["t"][0]) / PROFILE_STEP_MM), 0, len(it["t"]) - 1))
        fmt = lambda v: "-" if not np.isfinite(v) else f"{v:.0f}"     # noqa: E731
        charts[k]["chart"].title = f"{it['names']}   t {t:+.0f} mm: {fmt(it['ya'][i])} / {fmt(it['yb'][i])}"
        mcur = mcharts[k]["plots"]["cursor"]                    # metric chart: cursor and the curve value at t
        mcur.update(np.array([t, t]), np.array(it["mrange"], dtype=float))
        mcur.visible = True
        c = np.nan
        if it.get("curve") is not None and len(it["ct"]):
            c = it["curve"][int(np.argmin(np.abs(it["ct"] - t)))]
        mcharts[k]["chart"].title = (f"{it['names']}   t {t:+.0f} mm: {it['mname']} "
                                     + (f"{c:.2f}" if np.isfinite(c) else "-"))

    # --- UI ------------------------------------------------------------------------------------
    def _text(self, text, x, y, size=9, color="white", top=False):
        t = self.pl.add_text(text, position=(x, y), font_size=size, color=color)
        if top:
            t.GetTextProperty().SetVerticalJustificationToTop()
        if not hasattr(self, "_ui_texts"):
            self._ui_texts = []
        self._ui_texts.append(t)                         # removed by _teardown (also when the scene is kept)
        return t

    def _slider(self, callback, rng, value, x0: float, x1: float, y_px: float, fmt: str, label: str | None = None,
                value_label: bool = True):
        """A thin slider at pixel height y_px (normalized x0..x1); its value shows above the knob in white (unless
        value_label is False), its label (if any) below."""
        w = self.pl.add_slider_widget(callback, rng=rng, value=value, title="", pointa=(x0, y_px / self.H),
                                      pointb=(x1, y_px / self.H), style="modern", fmt=fmt, interaction_event="always",
                                      tube_width=0.004, slider_width=0.012)
        rep = w.GetRepresentation()
        rep.SetLabelHeight(0.016)
        rep.SetShowSliderLabel(int(value_label))
        prop = rep.GetLabelProperty()
        prop.SetColor(1.0, 1.0, 1.0)
        prop.ShadowOff()
        prop.BoldOff()
        if label is not None:
            self._text(label, int(x0 * self.W), int(y_px - 24), 8, "#bbbbbb")
        return w

    def _motion(self, e: dict) -> tuple[np.ndarray, float, float, float]:
        """Correction of slice e at the shown step relative to its acquired pose: the shift of its centre along its
        acquired (u, v, n) axes (mm), and its rotation: in-plane (about n, signed), tilt and total angle (deg)."""
        A, A0 = self._pose(e), self.snapshots[0]["affines"].get(e["key"], e["affine0"])
        F0 = frame_of(A0)
        shift = F0.T @ (self._centre(e, A) - self._centre(e, A0))
        rv = np.degrees(Rotation.from_matrix(frame_of(A) @ F0.T).as_rotvec())
        inplane = float(rv @ F0[:, 2])
        return shift, inplane, float(np.linalg.norm(rv - inplane * F0[:, 2])), float(np.linalg.norm(rv))

    def _stack_motion_text(self, st: dict) -> str:
        """The correction of the stack's shown slices: one slice -> its shift and rotation; several -> their mean
        shift (along the stack's axes, plus the mean shift length) and mean rotation angle."""
        def f(x, sign=True):                       # rounded first, so that -0.001 shows as +0.00, not -0.00
            return f"{round(float(x), 2) + 0.0:{'+' if sign else ''}.2f}"

        shown = [e for e in st["entries"] if id(e) in self._vis]
        if not shown:
            return "no slice shown"
        if len(shown) == 1:
            (du, dv, dn), inplane, tilt, _ = self._motion(shown[0])
            return (f"slice {shown[0]['index']}: shift u {f(du)}  v {f(dv)}  n {f(dn)} mm   "
                    f"rot in-plane {f(inplane)}, tilt {f(tilt, False)} deg")
        m = [self._motion(e) for e in shown]
        shifts = np.array([x[0] for x in m])
        mu, mv, mn = shifts.mean(0)
        return (f"{len(shown)} shown: mean shift u {f(mu)}  v {f(mv)}  n {f(mn)} mm, "
                f"mean length {f(np.linalg.norm(shifts, axis=1).mean(), False)} mm   "
                f"mean rot {f(np.mean([x[3] for x in m]), False)} deg")

    def _checkbox(self, callback, value: bool, x: int, y: int, color_on: str):
        return self.pl.add_checkbox_button_widget(callback, value=value, position=(x, y), size=18, border_size=2,
                                                  color_on=color_on, color_off="#555555", background_color="#303030")

    def _build_ui(self):
        pl, W, H = self.pl, self.W, self.H
        self._header = self._text("", PANEL_X, H - 10, 10, top=True)
        self._build_nav()
        y = H - 92
        self._text("Show", PANEL_X, y + 4, 9, "#bbbbbb")
        toggles = (("outlines  [o]", "show_outlines"),
                   ("lines  [i]", "show_lines"), ("profiles  [p]", "show_profiles"),
                   ("metric profiles  [n]", "show_metric_profiles"), ("neighbour lines  [j]", "show_nb_lines"),
                   ("image blurring  [u]", "show_blur"), ("PSF in profiles  [y]", "show_psf"))
        if self.mask_r0 > 0:
            toggles += (("disc mask  [k]", "show_mask"),)
        for k, (label, attr) in enumerate(toggles):
            x, yy = PANEL_X + (k % 2) * 205, y - ROW * (1 + k // 2)
            setattr(self, f"_btn_{attr}", self._checkbox(lambda v, a=attr: self.toggle(a, v), getattr(self, attr),
                                                         x, yy, "#66bb6a"))
            self._text(label, x + TEXT_X - PANEL_X, yy + 1)
        y -= ROW * (1 + (len(toggles) + 1) // 2) + 12
        # outline-colour metric (radio group, as slice_viewer_3d): each row also shows the subject-level value of its
        # metric at the shown step (filled by _update_texts)
        self._text("Outline colour by  [m]", PANEL_X, y + 4, 9, "#bbbbbb")
        self._metric_radios, self._metric_text = {}, {}
        for key in METRICS:
            y -= ROW
            self._metric_radios[key] = pl.add_radio_button_widget(
                lambda key=key: self.set_metric(key), "outline_metric", value=(key == self.metric),
                position=(PANEL_X, y), size=18, border_size=3, color_on="#ffa726", color_off="#555555",
                background_color="#303030")
            self._metric_text[key] = self._text("", TEXT_X, y + 1)
        self._info = self._text("", PANEL_X, y - 14, 9, top=True)
        self._build_stack_block()
        # step and frame sliders (bottom right, from the middle to the right edge), colour bar above them
        n = len(self.snapshots)
        self._hist_slider = None
        steps = [s["step"] for s in self.snapshots]
        if n > 1 and max(steps) > min(steps):
            self._hist_slider = self._slider(self._on_history_slider, [min(steps), max(steps)],
                                             steps[self.snap_index], 0.52, 0.97, 45, "%.0f",
                                             "optimization step  [Up / Down, Home / End / b: best]")
        self._frame_slider = None
        if self.n_frames > 1:
            self._frame_slider = self._slider(lambda v: self.set_frame(int(round(v)) - 1, from_slider=True),
                                              [1, self.n_frames], self.frame + 1, 0.52, 0.97, 110, "%.0f",
                                              "cardiac frame  [Left / Right]")
        # profile charts (right), styled and laid out as slice_viewer_3d: the intensity profiles in the top block, the
        # per-point metric of the same intersections in the block below. Profiles: per slice a thin curve where the
        # line is inside both images and a thick one where it enters the loss (inside the mask); yellow bar: in mask.
        # Slots per block: one per other stack (at most PROFILE_SLOTS), then the selected slice's two neighbours
        self._n_partner_slots = max(0, min(PROFILE_SLOTS, len(self.stacks) - 1))
        n_slots = self._n_partner_slots + NEIGHBOUR_SLOTS
        pitch = (CHART_TOP - CHART_Y0) / (2 * n_slots)

        def new_chart(row):
            ch = pv.Chart2D(size=(CHART_W, pitch - CHART_GAP), loc=(CHART_X, CHART_Y0 + row * pitch), x_label="",
                            y_label="", grid=False)
            ch.background_color = (0.10, 0.10, 0.10, 0.85)
            ch.border_color = (0.35, 0.35, 0.35)
            ch.legend_visible = False
            tp = ch.GetTitleProperties()
            tp.SetFontSize(11)
            tp.SetColor(0.85, 0.85, 0.85)
            tp.BoldOff()
            for ax in (ch.x_axis, ch.y_axis):
                ax.tick_label_size = 9
                ax.GetLabelProperties().SetColor(0.65, 0.65, 0.65)
                ax.pen.color = (0.4, 0.4, 0.4)
            for i in range(4):
                ch.GetAxis(i).GetGridPen().SetColorF(*(3 * (sv.PROFILE_GRID_GREY,)))
            ch.visible = False
            return ch

        self._charts, self._mcharts = [], []
        x0 = np.array([0.0, 1.0])
        for k in range(n_slots):
            ch = new_chart(2 * n_slots - 1 - k)
            ch.y_axis.tick_locations = (0, 128, 255)
            ch.y_axis.tick_labels = ("0", "128", "255")
            ch.y_range = [0, 255]
            plots = {}
            for side, col in zip(("a", "b"), sv.PROFILE_COLORS):
                plots[side + "_img"] = ch.line(x0, x0 * 0, color=col, width=1.0)
                plots[side + "_used"] = ch.line(x0, x0 * 0, color=col, width=2.5)
            plots["mask"] = ch.line(x0, x0 * np.nan, color=sv.MASK_CONTOUR_COLOR, width=2.0)
            plots["cursor"] = ch.line(np.array([0.0, 0.0]), np.array([0.0, 255.0]), color=sv.HOVER_COLOR, width=1.5)
            plots["cursor"].visible = False
            pl.add_chart(ch)
            self._charts.append(dict(chart=ch, plots=plots, item=None))

            mch = new_chart(n_slots - 1 - k)
            # as the profile charts: thin over the whole intersection, thick where it enters the loss, mask bar
            mplots = {"zero": mch.line(x0, x0 * 0, color=(0.4, 0.4, 0.4), width=1.0),
                      "full": mch.line(x0, x0 * np.nan, color=sv.PROFILE_LOSS_COLOR, width=1.0),
                      "curve": mch.line(x0, x0 * np.nan, color=sv.PROFILE_LOSS_COLOR, width=2.5),
                      "mask": mch.line(x0, x0 * np.nan, color=sv.MASK_CONTOUR_COLOR, width=2.0),
                      "cursor": mch.line(np.array([0.0, 0.0]), np.array([0.0, 1.0]), color=sv.HOVER_COLOR, width=1.5)}
            mplots["cursor"].visible = False
            pl.add_chart(mch)
            self._mcharts.append(dict(chart=mch, plots=mplots, item=None))
        # outline colour bar (as slice_viewer_3d): a narrow strip without labels between its colour max / min sliders,
        # the metric's name above it and the slider values to the right
        self._bar_key = None
        self._range_sliders, self._range_text = {}, {}
        if self.entries:
            pl.add_scalar_bar(title="outline colour", mapper=self.entries[0]["outline"].mapper, n_labels=0,
                              vertical=False, position_x=BAR_X0, position_y=BAR_Y0, width=BAR_X1 - BAR_X0,
                              height=BAR_H, color="white")
            sb = pl.scalar_bars["outline colour"]
            sb.SetLookupTable(self._lut)
            sb.SetTitle("")
            sb.SetNumberOfLabels(0)
            if hasattr(sb, "SetDrawTickLabels"):
                sb.SetDrawTickLabels(False)
            sb.SetBarRatio(1.0)
            sb.SetTextPad(0)
            self._bar_key = self._text("", int(BAR_X0 * W), int((SLIDER_HI_Y + 0.008) * H), 8, "#bbbbbb")
            self._add_range_sliders()

    def _build_stack_block(self):
        """Bottom left: the worst-slice button, then per stack (top to bottom) a row with its checkbox, profile radio
        button, name, mean outline metric and geometry; below it its slice and neighbours (+-w) sliders; below those the correction of
        its shown slices."""
        W = self.W
        n = len(self.stacks)
        y0, stack_h = 14, 70
        sx0, sx1, ax0, ax1 = 0.03, 0.17, 0.235, 0.375        # slice slider (left), neighbours slider (right)
        top = y0 + n * stack_h
        holder = {}

        def _worst(_state):
            holder["w"].GetRepresentation().SetState(0)          # a push button, not a toggle
            self.go_to_worst()
        holder["w"] = self._checkbox(_worst, False, PANEL_X, top + 50, "#ffa726")
        self._text("go to the worst slice of each stack  [d]", TEXT_X, top + 51)
        self._text("Stacks:", PANEL_X, top + 22, 9, "#bbbbbb")
        self._text("slice  [, .]", int(sx0 * W), top + 4, 8, "#bbbbbb")
        self._text("neighbours +-w  [+ -]", int(ax0 * W), top + 4, 8, "#bbbbbb")
        gx = int(ax1 * W) + 44                                # ghost buttons: right of the neighbours value
        self._text("ghost  [g]", gx, top + 4, 8, "#bbbbbb")
        self._ghost_buttons = []
        self._stack_text, self._stack_buttons, self._amount_sliders, self._pos_sliders = [], [], [], []
        self._amount_value, self._pos_value, self._motion_text, self._profile_radios = [], [], [], []
        for s, st in enumerate(self.stacks):
            base = y0 + (n - 1 - s) * stack_h
            self._stack_buttons.append(self._checkbox(lambda v, s=s: self.set_stack(s, v), st["on"], PANEL_X, base + 46,
                                                      st["color"]))
            # which stack's selected slice the profile charts show (one radio group over the stacks)
            self._profile_radios.append(self.pl.add_radio_button_widget(
                lambda s=s: self.set_profile_stack(s), "profile_stack", value=(s == self.sel["stack"]),
                position=(PANEL_X + 26, base + 48), size=14, border_size=3, color_on="#ffa726", color_off="#555555",
                background_color="#303030"))
            self._stack_text.append(self._text("", TEXT_X + 22, base + 47))
            self._pos_sliders.append(self._slider(
                lambda v, s=s: self.set_position(s, int(round(v)), from_slider=True), [0, len(st["entries"]) - 1],
                st["pos"], sx0, sx1, base + 32, "%.0f", value_label=False))
            self._amount_sliders.append(self._slider(
                lambda v, s=s: self.set_amount(s, int(round(v)), from_slider=True), [0, NEIGHBOURS_MAX],
                st["w"], ax0, ax1, base + 32, "%.0f", value_label=False))
            self._pos_value.append(self._text("", int(sx1 * W) + 6, base + 25, 8, "#dddddd"))
            self._amount_value.append(self._text("", int(ax1 * W) + 6, base + 25, 8, "#dddddd"))
            # this stack's +-w neighbours as ghosts (border only) instead of images
            self._ghost_buttons.append(self.pl.add_checkbox_button_widget(
                lambda v, s=s: self.set_ghost(s, v), value=st["ghost"], position=(gx, base + 24), size=14,
                border_size=2, color_on=st["color"], color_off="#555555", background_color="#303030"))
            self._motion_text.append(self._text("", int(sx0 * W), base + 4, 8, "#9ecfff"))

    def _update_texts(self):
        snap = self.snapshots[self.snap_index]
        sm = snap.get("summary") or {}
        summ = "   ".join(f"{METRICS[k]['label']} {sm[c]:.3f}" for k in ("ncc", "lncc", "nmi")
                          for c in METRICS[k]["summary"][:1] if c in sm and np.isfinite(sm[c]))
        status = (f"max motion {self._max_motion_mm():.2f} mm" + (" (best)" if snap["is_best"] else "")
                  if self.has_metrics else "saved pose (no history)")
        radius = self._mask_radius()
        if radius is not None:
            status += f"   disc r {radius:.0f} mm"
        if self.config:                                  # the loss's blur / PSF scale at this step
            scale, sigma = self._loss_state(snap["step"])
            status += f"   blur {sigma:.1f} mm" + (f", psf x{scale:.1f}" if scale != 1.0 else "")
        self._header.SetInput(f"{self.data['subject']}   run {self.data['run_name']}   [h: help]\n"
                              f"{summ}{'   ' if summ else ''}{status}")
        # subject-level value of each metric next to its radio button (motion: the largest slice motion)
        for key, spec in METRICS.items():
            if not self.has_metrics:
                value = "(no history)"
            elif key == "motion":
                value = f"max {self._max_motion_mm():.2f} mm"
            else:
                v = sm.get(spec["summary"][0]) if spec["summary"] else None
                value = spec["fmt"].format(v) if v is not None and np.isfinite(v) else "-"
            self._metric_text[key].SetInput(f"{spec['label']}   {value}")
        spec = METRICS[self.metric]
        for s, st in enumerate(self.stacks):
            vals = np.array([self._metric_value(e) for e in st["entries"]])
            mean = f"{spec['label']} {np.nanmean(vals):.3f}" if np.isfinite(vals).any() else ""
            self._stack_text[s].SetInput(f"{s + 1}  {self.labels[s]}   {mean}     {st['geometry']}")
            self._profile_radios[s].GetRepresentation().SetState(int(s == self.sel["stack"]))
            self._pos_value[s].SetInput(f"slice {st['entries'][st['pos']]['index']}/{len(st['entries'])}")
            self._amount_value[s].SetInput(f"+-{st['w']}")
            self._motion_text[s].SetInput(self._stack_motion_text(st) if self.has_metrics else "")
        e = self.sel
        if not self._sel_shown():
            self._info.SetInput("no slice shown")
            return
        A, A0 = self._pose(e), self.snapshots[0]["affines"].get(e["key"], e["affine0"])
        vals = []
        for k in ("ncc", "lncc", "nmi"):
            v = snap["metrics"].get(e["key"], {}).get(METRICS[k]["col"], np.nan)
            v0 = self.snapshots[0]["metrics"].get(e["key"], {}).get(METRICS[k]["col"], np.nan)
            if np.isfinite(v):
                vals.append(f"{METRICS[k]['label']} {v:.3f} (acq. {v0:.3f})")
        shift = float(np.linalg.norm(self._centre(e, A) - self._centre(e, A0)))
        in_hist = e["key"] in snap["affines"]
        self._info.SetInput(
            f"stack {e['stack'] + 1} ({self.labels[e['stack']]})  slice {e['index']}  "
            f"({e['pos'] + 1}/{len(self.stacks[e['stack']]['entries'])})\n"
            + ("\n".join(vals) + "\n" if vals else "")
            + (f"moved {shift:.2f} mm, {rotation_deg(A, A0):.2f} deg from acquired" if in_hist
               else "not in the history (shown at its loaded pose)"))

    # --- actions ----------------------------------------------------------------------------------
    def go_to_worst(self):
        """Move every stack to its worst slice by the outline metric at the shown step (lowest NCC / LNCC / NMI,
        largest motion) and select the selected stack's one; the stacks keep these slices (no re-matching), the
        sliders and +-w work on from there."""
        if not self._ui_ready:
            return
        better_high = METRICS[self.metric]["higher_better"]
        found = False
        for s, st in enumerate(self.stacks):
            vals = np.array([self._metric_value(e) for e in st["entries"]], float)
            if not np.isfinite(vals).any():
                continue
            st["pos"] = int(np.nanargmin(vals) if better_high else np.nanargmax(vals))
            self._pos_sliders[s].GetRepresentation().SetValue(st["pos"])
            found = True
        if not found:
            print(f"[worst] no {METRICS[self.metric]['label']} values at this step")
            return
        st = self.stacks[self.sel["stack"]]
        self.sel = self._synced_sel = st["entries"][st["pos"]]
        self._apply_visibility()
        self.pl.render()

    def set_amount(self, s: int, value: int, from_slider: bool = False):
        """Neighbouring slices (+-w) shown around stack s's current slice."""
        value = int(np.clip(value, 0, NEIGHBOURS_MAX))
        st = self.stacks[s]
        if value == st["w"] or not self._ui_ready:
            return
        st["w"] = value
        if not from_slider:
            self._amount_sliders[s].GetRepresentation().SetValue(value)
        self._apply_visibility()
        self.pl.render()

    def set_position(self, s: int, pos: int, from_slider: bool = False):
        """Scroll stack s to slice position pos (its +-w window moves with it); only this stack moves. For the selected
        slice's stack (profile radio button) that slice becomes the selected one, the other stacks keep their slices."""
        if not self._ui_ready:                           # pyvista calls slider callbacks once while creating them
            return
        st = self.stacks[s]
        pos = int(np.clip(pos, 0, len(st["entries"]) - 1))
        if pos == st["pos"]:
            return
        st["pos"] = pos
        if s == self.sel["stack"]:
            self.sel = self._synced_sel = st["entries"][pos]     # (no re-matching of the other stacks)
        if not from_slider:
            self._pos_sliders[s].GetRepresentation().SetValue(pos)
        self._apply_visibility()
        self.pl.render()

    def set_ghost(self, s: int, on: bool | None = None):
        """Stack s's +-w neighbours drawn as ghosts (a thin border in the stack colour, no image) or as images; None
        toggles. Its current slice stays an image."""
        st = self.stacks[s]
        st["ghost"] = (not st["ghost"]) if on is None else bool(on)
        self._ghost_buttons[s].GetRepresentation().SetState(int(st["ghost"]))
        if self._ui_ready:
            self._apply_visibility()
            self.pl.render()

    def set_stack(self, s: int, on: bool):
        self.stacks[s]["on"] = bool(on)
        self._stack_buttons[s].GetRepresentation().SetState(int(on))
        self._apply_visibility()                          # moves the selection off a hidden stack
        self.pl.render()

    def toggle(self, attr: str, value: bool | None = None):
        value = (not getattr(self, attr)) if value is None else bool(value)
        setattr(self, attr, value)
        button = getattr(self, f"_btn_{attr}", None)       # (no disc-mask button for runs without one)
        if button is not None:
            button.GetRepresentation().SetState(int(value))
        self._apply_visibility()
        self.pl.render()

    def select(self, e: dict):
        self.sel = e
        self._apply_visibility()
        self.pl.render()

    def set_profile_stack(self, s: int):
        """Show the profiles of stack s's current slice (its round button, keys 1-9): it becomes the selected slice,
        and every stack keeps its slice (the charts then show the intersections of the stacks' current slices)."""
        if not self._ui_ready:
            return
        if s >= len(self.stacks) or not self.stacks[s]["on"]:
            self._update_texts()                          # (resets the radio buttons)
            self.pl.render()
            return
        st = self.stacks[s]
        self.sel = self._synced_sel = st["entries"][st["pos"]]       # no re-sync: the other stacks stay where they are
        self._apply_visibility()
        self.pl.render()

    def step_slice(self, direction: int):
        """Previous / next slice of the selected stack, as its slider (the other stacks keep their slices)."""
        self.set_position(self.sel["stack"], self.sel["pos"] + direction)

    def _on_pick(self, pos):
        """Double click: the shown slice whose plane passes through the clicked point (inside its image)."""
        p = np.asarray(pos, float)
        best = None
        for e in self.entries:
            if id(e) not in self._vis:
                continue
            Ai = np.linalg.inv(self._pose(e))
            i, j, _ = Ai[:3, :3] @ p + Ai[:3, 3]
            nx, ny = e["shape"]
            if not (-0.5 <= i <= nx - 0.5 and -0.5 <= j <= ny - 0.5):
                continue
            dist = self._plane_distance(e, p)
            if best is None or dist < best[0]:
                best = (dist, e)
        if best is not None and best[0] < 2.0:
            self.select(best[1])

    def focus_selected(self):
        A = self._pose(self.sel)
        F = frame_of(A)
        c = self._centre(self.sel, A)
        self.pl.camera.focal_point = c
        self.pl.camera.position = c + F[:, 2] * 400.0
        self.pl.camera.up = -F[:, 1]
        self.pl.reset_camera(bounds=self._sel_mesh.bounds)          # keeps the direction, fits the slice
        self.pl.render()

    def _set_camera(self):
        self.pl.reset_camera()
        self.pl.camera.zoom(1.3)

    def next_subject(self, step: int):
        target = self.subject_index + step
        if 0 <= target < len(self.subjects):
            self._load_subject(target)
        else:
            print("[nav] no subject in that direction")

    def next_run(self, step: int):
        """Next / previous run (newest first), skipping runs without subjects or that fail to load. The current
        subject is kept if the run has it, else the same position in the run's subject list."""
        current = self.subjects[self.subject_index]
        target = self.run_index + step
        while 0 <= target < len(self.runs):
            subjects = subjects_in(Path(self.runs[target]))
            if subjects:
                old = (self.run_dir, self.subjects, self.run_index)
                self.run_dir, self.subjects, self.run_index = Path(self.runs[target]), subjects, target
                index = subjects.index(current) if current in subjects else min(self.subject_index, len(subjects) - 1)
                if self._load_subject(index):
                    return
                self.run_dir, self.subjects, self.run_index = old       # failed: stay on the current run
            target += step
        print("[nav] no run in that direction")

    def _build_nav(self):
        """Top right, as slice_viewer_3d: a row per run / subject with  label  [<] [>]  keys."""
        def push(glyph, cb, x, y):
            holder = {}

            def _cb(_state):
                holder["w"].GetRepresentation().SetState(0)          # a push button, not a toggle
                cb()
            holder["w"] = self.pl.add_checkbox_button_widget(_cb, value=False, position=(x, y), size=18, border_size=1,
                                                             color_on="#ffa726", color_off="#4a4a4a",
                                                             background_color="#303030")
            self._text(glyph, x + 5, y + 1, 9)

        W, H = self.W, self.H
        rows = (("_run_text", "#80cbc4", lambda: self.next_run(-1), lambda: self.next_run(+1), "{ }  newest first"),
                ("_nav_text", "#ffa726", lambda: self.next_subject(-1), lambda: self.next_subject(+1), "[ ]"))
        for row, (attr, colour, prev_cb, next_cb, keys) in enumerate(rows):
            ry = H - 30 - row * ROW
            setattr(self, attr, self._text("", W - NAV_RIGHT_W, ry + 2, 9, colour))
            push("<", prev_cb, W - 154, ry)
            push(">", next_cb, W - 130, ry)
            self._text(keys, W - 104, ry + 3, 7, "#888888")
        self._run_text.SetInput(f"run {self.run_index + 1}/{len(self.runs)}")
        self._nav_text.SetInput(f"subject {self.subject_index + 1}/{len(self.subjects)}  {self.subjects[self.subject_index]}")

    def _bind_keys(self):
        pl = self.pl
        def bump(step):                          # neighbours (+-w) of the selected slice's stack
            s = self.sel["stack"]
            self.set_amount(s, self.stacks[s]["w"] + step)

        bindings = {
            "d": self.go_to_worst, "plus": lambda: bump(1), "equal": lambda: bump(1), "KP_Add": lambda: bump(1),
            "minus": lambda: bump(-1), "KP_Subtract": lambda: bump(-1),
            "comma": lambda: self.step_slice(-1), "period": lambda: self.step_slice(1),
            "v": lambda: self.set_stack(self.sel["stack"], not self.stacks[self.sel["stack"]]["on"]),
            "m": self.cycle_metric,
            "o": lambda: self.toggle("show_outlines"), "g": lambda: self.set_ghost(self.sel["stack"]),
            "i": lambda: self.toggle("show_lines"), "p": lambda: self.toggle("show_profiles"),
            "k": lambda: self.toggle("show_mask"), "n": lambda: self.toggle("show_metric_profiles"),
            "j": lambda: self.toggle("show_nb_lines"),
            "u": lambda: self.toggle("show_blur"), "y": lambda: self.toggle("show_psf"),
            "Up": lambda: self.set_snapshot(self.snap_index + 1), "Down": lambda: self.set_snapshot(self.snap_index - 1),
            "Home": lambda: self.set_snapshot(0), "End": lambda: self.set_snapshot(len(self.snapshots) - 1),
            "b": lambda: self.set_snapshot(self.history["best_index"]) if self.history["best_index"] is not None else None,
            "Left": lambda: self.set_frame(self.frame - 1), "Right": lambda: self.set_frame(self.frame + 1),
            "z": self.focus_selected, "Next": lambda: self.next_subject(1), "Prior": lambda: self.next_subject(-1),
            "bracketright": lambda: self.next_subject(1), "bracketleft": lambda: self.next_subject(-1),
            "braceright": lambda: self.next_run(1), "braceleft": lambda: self.next_run(-1),
            "h": lambda: print(__doc__),
        }
        for d in range(1, 10):
            bindings[str(d)] = lambda d=d: self.set_profile_stack(d - 1)
        for key, fn in bindings.items():
            pl.iren._key_press_event_callbacks.pop(key, None)       # drop pyvista's defaults (camera views, picking)
            pl.add_key_event(key, fn)

    def show(self, **kwargs):
        self.pl.show(**kwargs)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("subject", nargs="?", help="subject folder name (default: the run's first subject)")
    ap.add_argument("--align-root", type=Path, default=ALIGN_ROOT,
                    help="directory holding moco_dicom.py run directories (default: %(default)s)")
    ap.add_argument("--run", default=None, help="run directory name under --align-root (default: the newest)")
    ap.add_argument("--run-dir", type=Path, default=None, help="explicit run directory (overrides --align-root / --run)")
    ap.add_argument("--root", type=Path, default=None,
                    help="input DICOM root (default: the run's run_config.json 'root')")
    ap.add_argument("--neighbours", type=int, default=0,
                    help="neighbouring slices (+-w) shown around each stack's current slice at start (default: %(default)s)")
    ap.add_argument("--metric", choices=list(METRICS), default="ncc", help="initial outline metric")
    ap.add_argument("--start", default="best", help="initial snapshot: best, end, 0 or an index (default: %(default)s)")
    ap.add_argument("--frame", type=int, default=0, help="initial cardiac frame (0-based)")
    ap.add_argument("--workers", type=int, default=8, help="threads for reading DICOM files")
    ap.add_argument("--screenshot", type=Path, default=None, help="render off-screen to this file and exit")
    args = ap.parse_args(argv)

    run_dir = args.run_dir or (args.align_root / args.run if args.run else find_run(args.align_root, args.subject))
    if run_dir is None or not Path(run_dir).is_dir():
        sys.exit(f"No run found under {args.align_root}" + (f" for subject {args.subject}" if args.subject else ""))
    run_dir = Path(run_dir)
    subjects = subjects_in(run_dir)
    if not subjects:
        sys.exit(f"No subject with DICOMs or a history in {run_dir}")
    subject = args.subject or subjects[0]
    if subject not in subjects:
        sys.exit(f"Subject {subject} not in {run_dir} (found: {', '.join(subjects)})")
    start = None if args.start == "best" else 10 ** 9 if args.start == "end" else int(args.start)
    if args.screenshot is not None:
        pv.OFF_SCREEN = True
    print(f"Run {run_dir} ({len(subjects)} subject(s))")
    runs = runs_in(run_dir.parent)                       # the runs next to this one, for the run < / > buttons
    pixel_cache = {}                                     # the last subject's stacks, reused across runs
    viewer = DicomSliceViewer(run_dir, subjects, subject,
                              lambda r, s: load_subject(r, s, args.root, args.workers, pixel_cache),
                              neighbours=args.neighbours, metric=args.metric, start=start, frame=args.frame,
                              runs=runs)
    if args.screenshot is not None:
        viewer.show(screenshot=str(args.screenshot), auto_close=True)
        print(f"Saved {args.screenshot}")
    else:
        viewer.show()


if __name__ == "__main__":
    main()
