"""Render the optimization of one subject in the 3D viewer (slice_viewer_3d.py) as a GIF for the README.

The subject's ID is never shown: the viewer's header reads "UK Biobank subject" and the subject navigation text is
blank; rendering stops if an ID-like number is on screen. Only one SA slice (--slice), LA_3Ch and LA_4Ch are shown, the
camera is tilted up and zoomed in.

Two sources:
  1. an alignment run (moco_nifti.py with --logging true, the default). --export writes the replay data of the
     subject to a folder (default media/optimization_data/): the pose history and the overlays of its metrics/
     folder without the subject_id column, the run's settings without paths and subject selection, and the GIF
     settings (gif.json). It holds no subject ID and no image data.
         python moco_nifti.py --max-subjects 1 --run-name readme_gif
         python media/make_readme_gif.py --run readme_gif --slice "SA[05]" --label "default settings" --export
  2. that replay data and the subject's original images (the UK Biobank folder; no run directory needed).
     --images is the subject's image folder, or any folder above it (e.g. the whole data directory): the subject is
     then found by its image geometry (the SAX affine at step 0). Images whose geometry differs from the history
     are refused.
         python media/make_readme_gif.py --data media/optimization_data --images <ukbb_dir>

Timing: --opt-frames frames cover step 0 .. the last step, spaced by how much the slices move (equal motion per
frame, with a floor so static stretches still advance): the fast early optimization plays slowly, the static tail is
mostly skipped. The cardiac phase advances by --phase-step per frame; after the last step the final pose is held for
--hold-seconds while the cardiac phase keeps advancing.

Colours: one palette for the whole GIF, no dithering, so colours do not flicker between frames: a grey ramp, a ramp
per segmentation label shown, the rest from the other colours of the frames (build_palette).
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pyvista as pv
from PIL import Image
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from slice_viewer_3d import (ALIGN_ROOT, METRICS_SUBDIR, SEG_COLORS, SliceViewer, _subjects_in_run,  # noqa: E402
                             corner_ijk, find_run_dir, from_disk, load_history, load_images, load_overlays,
                             load_subject_run, voxel_to_world)

LA_SHOWN = ("LA_3Ch", "LA_4Ch")
ELEVATION_DEG = 45.0
FRAME_MS = 80
MOTION_FLOOR = 0.15        # share of the mean per-snapshot motion granted to every snapshot interval
HEADER = "UK Biobank subject"
DATA_DIR = REPO / "media" / "optimization_data"
DATA_FILES = ("history.csv", "history_summary.csv", "lines.csv", "long_axis.csv", "mask_contours.csv",
              "sax_lines.csv")                     # from <run>/<subject>/metrics/, without the subject_id column
CONFIG_DROP = ("root", "out_root", "run_name", "command_line", "start_index", "end_index", "max_subjects",
               "skip_processed", "device", "workers")    # paths, and what selects subjects (their folder order)
GIF_DEFAULTS = {"slice": "SA[08]", "opt_frames": 77, "phase_step": 2, "hold_seconds": 2.0, "width": 1000,
                "zoom": 1.5, "shift": 0.0, "label": None}
ID_LIKE = re.compile(r"(?<![0-9.])[1-6][0-9]{6}(?![0-9])")       # a UK Biobank participant ID (7 digits)


class AnonymousViewer(SliceViewer):
    """The viewer without the subject ID: blank subject navigation text (the header is set by the loaders)."""

    def _update_nav_text(self):
        self._nav_text.SetInput("")
        self._run_text.SetInput("")


# --- sources ---------------------------------------------------------------------------------------------------------

def load_from_run(run: Path, subject: str, label: str | None) -> dict:
    loaded = load_subject_run(run, subject)
    loaded["title"] = loaded["title"].replace(f"UKBB {subject}", HEADER)
    if label:
        loaded["run_name"] = label
    return loaded


def check_geometry(data: dict, history: dict) -> int:
    """Number of slices whose NIfTI affine equals the history's step-0 pose; exits if any differs or none match."""
    snap0 = history["snapshots"][0]
    same = differ = 0
    for view, v in data.items():
        for s in v["slices"]:
            A = snap0["affines"].get((view, int(s["index"])))
            if A is not None:
                ok = np.allclose(A, s["affine"], atol=1e-3)
                same, differ = same + ok, differ + (not ok)
    if differ or not same:
        sys.exit(f"these images are not the ones of the replay data ({same} slices match its step-0 poses, "
                 f"{differ} differ)")
    return same


def find_image_dir(path: Path, history: dict) -> Path:
    """The folder under `path` (itself, or up to three levels down) holding the SAX image of the replay data,
    recognised by its affine (the SAX pose at step 0)."""
    ref = history["snapshots"][0]["affines"].get((from_disk("SAX"), 0))        # the viewer's name of the stack
    if ref is None:
        sys.exit("the replay data has no SAX pose to find the subject by: pass its image folder with --images")
    for pattern in ("SAX.nii.gz", "*/SAX.nii.gz", "*/*/SAX.nii.gz", "*/*/*/SAX.nii.gz"):
        for f in sorted(path.glob(pattern)):
            try:
                if np.allclose(nib.load(f).affine, ref, atol=1e-3):
                    return f.parent
            except Exception:                            # noqa: BLE001 (unreadable file: skip)
                continue
    sys.exit(f"no SAX.nii.gz under {path} has the geometry of the replay data")


def load_from_data(data_dir: Path, image_dir: Path, label: str | None) -> dict:
    history = load_history(data_dir)
    data = load_images(image_dir)
    n = check_geometry(data, history)
    print(f"images: {n} slices match the replay data's step-0 poses")
    return {"data": data, "history": history, "overlays": load_overlays(data_dir),
            "config": json.loads((data_dir / "run_config.json").read_text()), "title": HEADER,
            "run_name": label or data_dir.name, "not_optimized": False}


def export(run: Path, subject: str, out_dir: Path, gif: dict) -> None:
    """The replay data of `subject`: its metrics CSVs without the subject_id column (values unchanged), the run's
    settings without CONFIG_DROP, and the GIF settings. Exits (writing nothing) if the subject ID would remain."""
    metrics = run / subject / METRICS_SUBDIR
    files = {}
    for name in DATA_FILES:
        path = metrics / name
        if not path.exists():
            continue
        with path.open(newline="") as fh:
            rows = list(csv.reader(fh))
        drop = [i for i, c in enumerate(rows[0]) if c == "subject_id"]
        lines = [",".join(c for i, c in enumerate(r) if i not in drop) for r in rows]
        files[name] = "\n".join(lines) + "\n"
    config = json.loads((run / "overall_metrics" / "run_config.json").read_text())
    files["run_config.json"] = json.dumps({k: v for k, v in config.items() if k not in CONFIG_DROP}, indent=2,
                                          sort_keys=True) + "\n"
    files["gif.json"] = json.dumps(gif, indent=2) + "\n"
    leaked = [n for n, text in files.items() if subject in text]
    if leaked:
        sys.exit(f"the subject ID would remain in {leaked}; nothing exported")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (out_dir / name).write_text(text, newline="\n")
    print(f"replay data -> {out_dir} ({', '.join(files)}; "
          f"{sum(len(t) for t in files.values()) / 1e3:.0f} kB)")


# --- rendering -------------------------------------------------------------------------------------------------------

def pan(cam, frac: float, aspect: float) -> None:
    """Move the scene left on screen by `frac` of the window width (right for frac < 0), e.g. to centre the slices
    between the panels at the sides."""
    pos, fp, up = (np.asarray(v, float) for v in (cam.position, cam.focal_point, cam.up))
    dist = np.linalg.norm(fp - pos)
    right = np.cross((fp - pos) / dist, up)
    right /= np.linalg.norm(right)
    half_h = cam.parallel_scale if cam.parallel_projection else dist * np.tan(np.radians(cam.view_angle) / 2)
    d = frac * 2 * half_h * aspect * right
    cam.position, cam.focal_point = tuple(pos + d), tuple(fp + d)


def build_palette(frames: list, alpha: float, n_grey: int = 64, n_label: int = 20, covered: float = 10.0) -> np.ndarray:
    """One 256-colour palette for all frames (no dithering, so colours do not flicker between frames): a grey ramp,
    a ramp per segmentation label in the frames (its colour blended over the grey range, as cell_colors draws it), and
    the rest median-cut from the pixels those ramps do not cover (outlines, charts, text, colour bar). A palette
    median-cut from the whole frames spends nearly all entries on the grey images, leaving a few per label: holes and
    banding in the segmentations."""
    sample = np.concatenate([np.asarray(f).reshape(-1, 3)[::5] for f in frames[::3]]).astype(np.float32)
    g = np.linspace(0.0, 255.0, n_grey)
    ramps = [np.stack([g, g, g], axis=1)]
    fine = np.linspace(0.0, 255.0, 256)[:, None]
    for c in SEG_COLORS.values():
        c = np.asarray(c, np.float32)[None]
        dist, _ = cKDTree((1 - alpha) * fine + alpha * c).query(sample, distance_upper_bound=6.0)
        if np.isfinite(dist).sum() > 10:                 # the label is in the frames (even a small region)
            ramps.append((1 - alpha) * np.linspace(0.0, 255.0, n_label)[:, None] + alpha * c)
    fixed = np.concatenate(ramps)
    dist, _ = cKDTree(fixed).query(sample)
    rest = sample[dist > covered]
    n_rest = 256 - len(fixed)
    if len(rest) and n_rest > 0:
        q = Image.fromarray(rest.astype(np.uint8)[None]).quantize(colors=n_rest, method=Image.Quantize.MEDIANCUT)
        extra = np.array(q.getpalette()[:3 * n_rest], np.float32).reshape(-1, 3)
        fixed = np.concatenate([fixed, extra])
    return np.clip(np.round(fixed), 0, 255).astype(np.uint8)


def to_palette(frame: Image.Image, palette: np.ndarray) -> Image.Image:
    """The frame in `palette`, every pixel its nearest palette colour (exact, no dithering)."""
    a = np.asarray(frame, np.uint8)
    packed = (a[..., 0].astype(np.int32) << 16) | (a[..., 1].astype(np.int32) << 8) | a[..., 2]
    uniq, inv = np.unique(packed.ravel(), return_inverse=True)
    cols = np.stack([(uniq >> 16) & 255, (uniq >> 8) & 255, uniq & 255], axis=1).astype(np.float32)
    _, idx = cKDTree(palette.astype(np.float32)).query(cols)
    im = Image.fromarray(idx[inv].reshape(a.shape[:2]).astype(np.uint8), mode="P")
    im.putpalette(np.concatenate([palette.ravel(), np.zeros(768 - palette.size, np.uint8)]).tolist())
    return im


def shown_texts(pl) -> list[str]:
    """Every string drawn by a text actor of the plotter."""
    out = []
    for ren in pl.renderers:
        for a in ren.GetActors2D():
            if hasattr(a, "GetInput") and a.GetInput():
                out.append(a.GetInput())
            elif hasattr(a, "GetText"):
                out += [a.GetText(i) for i in range(8) if a.GetText(i)]
    return out


def render(viewer: SliceViewer, gif: dict, out: Path, preview: Path | None, forbidden: set[str]) -> None:
    names = [e["name"] for e in viewer.entries]
    show = (gif["slice"],) + LA_SHOWN
    missing = [n for n in show if n not in names]
    if missing:
        sys.exit(f"not in this subject: {missing}; slices: {names}")
    regular = [i for i, s in enumerate(viewer.snapshots) if not s["is_best"]]     # step 0 .. last step, in order

    def corners(i):                                  # world corners of every slice at snapshot i
        snap = viewer.snapshots[i]
        return np.concatenate([voxel_to_world(viewer._affine_for(e, snap), corner_ijk(*e["slice"]["grey"].shape[:2]))
                               for e in viewer.entries])

    for e in viewer.entries:                         # only the chosen slices; the SA slice's profiles in the charts
        e["visible"] = e["name"] in show
        viewer._set_checkbox_state(e["name"], e["visible"])
    viewer._apply_visibility()
    viewer.set_profile(gif["slice"])

    # The camera: the viewer's oblique view direction, framed on the corners of every slice at step 0. (Not on the
    # actors' bounds: VTK's bounds of a moved actor are its mesh's box transformed, inflated by the rotation, which
    # differs between a run's images (saved at the aligned pose) and the original ones.)
    pl = viewer.pl
    pl.show(auto_close=False, interactive=False)
    viewer._set_initial_camera()
    lo, hi = corners(regular[0]).min(axis=0), corners(regular[0]).max(axis=0)
    pl.renderer.ResetCamera(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])
    cam = pl.camera
    cam.Elevation(ELEVATION_DEG)
    cam.OrthogonalizeViewUp()
    cam.zoom(gif["zoom"])
    pan(cam, gif["shift"], pl.window_size[0] / pl.window_size[1])

    # frame schedule: equal slice motion per frame, then the final pose held while the heart beats
    C = [corners(i) for i in regular]
    motion = np.array([np.linalg.norm(C[k + 1] - C[k], axis=1).mean() for k in range(len(C) - 1)])
    cum = np.concatenate([[0.0], np.cumsum(motion + MOTION_FLOOR * motion.mean())])
    targets = np.linspace(0.0, cum[-1], max(2, gif["opt_frames"]))
    picks = [regular[min(int(np.searchsorted(cum, x, side="right")) - 1, len(regular) - 1)] for x in targets]
    picks[-1] = regular[-1]
    schedule = picks + [regular[-1]] * int(round(gif["hold_seconds"] * 1000 / FRAME_MS))

    def frame(n, i):
        viewer.set_snapshot(i)
        viewer.set_frame((n * gif["phase_step"]) % viewer.n_frames)
        pl.render()
        leaked = [t for t in shown_texts(pl) if ID_LIKE.search(t) or any(f in t for f in forbidden)]
        if leaked:
            sys.exit(f"an ID-like number is on screen ({len(leaked)} text(s)); nothing written")
        im = Image.fromarray(pl.screenshot(return_img=True))
        if gif["width"] and im.width > gif["width"]:
            im = im.resize((gif["width"], round(im.height * gif["width"] / im.width)), Image.LANCZOS)
        return im.convert("RGB")

    if preview is not None:                          # step 0 above the last step
        first, last = frame(0, schedule[0]), frame(len(picks) - 1, regular[-1])
        both = Image.new("RGB", (first.width, first.height * 2))
        both.paste(first, (0, 0))
        both.paste(last, (0, first.height))
        preview.parent.mkdir(parents=True, exist_ok=True)
        both.save(preview)
        print(f"preview (step 0, last step) -> {preview}")
        return

    rgb = []
    for n, i in enumerate(schedule):
        rgb.append(frame(n, i))
        print(f"  frame {n + 1}/{len(schedule)}: step {viewer.snapshots[i]['step']}", end="\r")

    palette = build_palette(rgb, viewer.seg_alpha)
    frames = [to_palette(f, palette) for f in rgb]

    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=FRAME_MS, loop=0, optimize=True)
    print(f"\n{len(frames)} frames, {len(frames) * FRAME_MS / 1000:.1f} s -> {out} ({out.stat().st_size / 1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_argument_group("source 1: an alignment run")
    src.add_argument("--run", default=None, help="run directory name under --align-root (default: the newest run, "
                                                 "unless --data is given)")
    src.add_argument("--align-root", type=Path, default=ALIGN_ROOT, help="Default: %(default)s")
    src.add_argument("--subject-index", type=int, default=0, help="subject of the run (alphabetical order)")
    src.add_argument("--export", nargs="?", const=DATA_DIR, type=Path, default=None, metavar="DIR",
                     help=f"also write the replay data of the subject (default DIR: {DATA_DIR.relative_to(REPO)})")
    src = ap.add_argument_group("source 2: exported replay data and the subject's original images")
    src.add_argument("--data", type=Path, default=None, help=f"replay data folder, e.g. {DATA_DIR.relative_to(REPO)}")
    src.add_argument("--images", type=Path, default=None,
                     help="the subject's image folder, or a folder above it to search (e.g. the UK Biobank data dir)")
    g = ap.add_argument_group("GIF (with --data: the exported gif.json, unless given)")
    g.add_argument("--out", type=Path, default=REPO / "media" / "optimization.gif", help="Default: %(default)s")
    g.add_argument("--slice", default=None, help=f"SA slice shown with the two LA views (and in the charts). "
                                                 f"Default: {GIF_DEFAULTS['slice']}")
    g.add_argument("--opt-frames", type=int, default=None,
                   help=f"GIF frames for step 0 .. last step. Default: {GIF_DEFAULTS['opt_frames']}")
    g.add_argument("--phase-step", type=int, default=None,
                   help=f"cardiac phases advanced per GIF frame (0 = fixed phase). Default: {GIF_DEFAULTS['phase_step']}")
    g.add_argument("--hold-seconds", type=float, default=None,
                   help=f"seconds the final pose is held at the end while the cardiac phase keeps advancing "
                        f"(2 s = one heartbeat of 50 phases at --phase-step 2). Default: {GIF_DEFAULTS['hold_seconds']}")
    g.add_argument("--width", type=int, default=None,
                   help=f"GIF width in pixels (frames are downscaled). Default: {GIF_DEFAULTS['width']}")
    g.add_argument("--zoom", type=float, default=None, help=f"camera zoom. Default: {GIF_DEFAULTS['zoom']}")
    g.add_argument("--shift", type=float, default=None,
                   help=f"move the slices left by this fraction of the window width (negative: right). "
                        f"Default: {GIF_DEFAULTS['shift']}")
    g.add_argument("--label", default=None, help="shown after 'run' in the header. Default: the run's name")
    g.add_argument("--preview", type=Path, default=None, metavar="PNG",
                   help="only write the step-0 and the last-step frame (one above the other) to this PNG")
    args = ap.parse_args()

    gif = dict(GIF_DEFAULTS)
    if args.data is not None:
        gif.update(json.loads((args.data / "gif.json").read_text()))
    gif.update({k: getattr(args, k) for k in GIF_DEFAULTS if getattr(args, k) is not None})

    pv.OFF_SCREEN = True
    if args.data is not None:
        if args.images is None:
            sys.exit("--data needs --images (the subject's original images, or a folder to search)")
        history = load_history(args.data)
        image_dir = args.images if any(args.images.glob("*.nii.gz")) else find_image_dir(args.images, history)
        print(f"replay {args.data}, images {image_dir}")
        forbidden = {p for p in image_dir.parts if p.isdigit()}
        viewer = AnonymousViewer([args.data], lambda r: ["subject"],
                                 lambda r, s: load_from_data(args.data, image_dir, gif["label"]), verbose=False,
                                 start=0)
    else:
        run = find_run_dir(args.align_root, None, args.run)
        if run is None:
            sys.exit(f"no run {args.run or ''} under {args.align_root}")
        subjects = _subjects_in_run(run)
        subject = subjects[args.subject_index]
        print(f"run {run.name}, subject {args.subject_index + 1}/{len(subjects)}")
        if args.export is not None:
            export(run, subject, args.export, gif)
        forbidden = {subject}
        viewer = AnonymousViewer([run], _subjects_in_run, lambda r, s: load_from_run(r, s, gif["label"]),
                                 subject_index=args.subject_index, verbose=False, start=0)
    render(viewer, gif, args.out, args.preview, forbidden)


if __name__ == "__main__":
    main()
