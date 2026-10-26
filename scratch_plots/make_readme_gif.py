"""Render the optimisation of the viewer's default subject as a GIF for the README.

Same run / subject as `python slice_viewer_3d.py` (newest run, first subject); only one SA slice
(--slice), LA_3Ch and LA_4Ch are shown, the camera is tilted up and zoomed in.

Timing: --opt-frames frames cover step 0 .. the last step, spaced by how much the slices move (equal
motion per frame, with a floor so static stretches still advance): the fast early optimisation plays
slowly, the static tail is mostly skipped. The cardiac phase advances by --phase-step per frame; at the
end the final pose beats for --hold-cycles heartbeats.

Colours: one palette for the whole GIF (built from sample frames plus swatches of every segmentation
colour blended over the grey range), no dithering, so colours do not flicker between frames.

    python scratch_plots/make_readme_gif.py [--slice "SA[09]"] [--preview] [--out media/optimization.gif]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pyvista as pv
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from slice_viewer_3d import (ALIGN_ROOT, SEG_COLORS, SliceViewer, _subjects_in_run, corner_ijk,  # noqa: E402
                             find_run_dir, load_subject_run, voxel_to_world)

LA_SHOWN = ("LA_3Ch", "LA_4Ch")
ELEVATION_DEG, ZOOM = 45.0, 1.5
FRAME_MS = 80
MOTION_FLOOR = 0.15        # share of the mean per-snapshot motion granted to every snapshot interval

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, default=REPO / "media" / "optimization.gif")
ap.add_argument("--slice", default="SA[08]", help="SA slice shown with the two LA views (and in the charts)")
ap.add_argument("--opt-frames", type=int, default=77, help="GIF frames for step 0 .. last step")
ap.add_argument("--phase-step", type=int, default=2, help="cardiac phases advanced per GIF frame (0 = fixed phase)")
ap.add_argument("--hold-cycles", type=float, default=0.5, help="heartbeats played on the final pose at the end")
ap.add_argument("--width", type=int, default=1000, help="GIF width in pixels (frames are downscaled)")
ap.add_argument("--preview", action="store_true", help="only write the step-0 frame as a PNG next to --out")
args = ap.parse_args()

pv.OFF_SCREEN = True
run = find_run_dir(ALIGN_ROOT, None, None)
subjects = _subjects_in_run(run)
viewer = SliceViewer([run], _subjects_in_run, lambda r, s: load_subject_run(r, s), verbose=False, start=0)
print(f"run {run.name}, subject {subjects[0]}")

# only the chosen slices; the SA slice's intersection profiles in the charts
show = (args.slice,) + LA_SHOWN
for e in viewer.entries:
    e["visible"] = e["name"] in show
    viewer._set_checkbox_state(e["name"], e["visible"])
viewer._apply_visibility()
viewer.set_profile(args.slice)

pl = viewer.pl
pl.show(auto_close=False, interactive=False)
cam = pl.camera
cam.Elevation(ELEVATION_DEG)
cam.OrthogonalizeViewUp()
cam.zoom(ZOOM)

# --- frame schedule: equal slice motion per frame -------------------------------------------
regular = [i for i, s in enumerate(viewer.snapshots) if not s["is_best"]]     # step 0 .. last step, in order


def corners(i):
    snap = viewer.snapshots[i]
    return np.concatenate([voxel_to_world(viewer._affine_for(e, snap), corner_ijk(*e["slice"]["grey"].shape[:2]))
                           for e in viewer.entries])


C = [corners(i) for i in regular]
motion = np.array([np.linalg.norm(C[k + 1] - C[k], axis=1).mean() for k in range(len(C) - 1)])
weight = motion + MOTION_FLOOR * motion.mean()
cum = np.concatenate([[0.0], np.cumsum(weight)])
targets = np.linspace(0.0, cum[-1], max(2, args.opt_frames))
picks = [regular[min(int(np.searchsorted(cum, x, side="right")) - 1, len(regular) - 1)] for x in targets]
picks[-1] = regular[-1]
n_hold = int(round(args.hold_cycles * viewer.n_frames / max(1, args.phase_step))) if args.phase_step else 0
schedule = picks + [regular[-1]] * n_hold

# --- render ----------------------------------------------------------------------------------
args.out.parent.mkdir(parents=True, exist_ok=True)
rgb = []
for n, i in enumerate(schedule):
    viewer.set_snapshot(i)
    viewer.set_frame((n * args.phase_step) % viewer.n_frames)
    pl.render()
    img = pl.screenshot(return_img=True)
    if args.preview:
        Image.fromarray(img).save(args.out.with_suffix(".png"))
        print(f"preview -> {args.out.with_suffix('.png')}")
        sys.exit()
    im = Image.fromarray(img)
    if args.width and im.width > args.width:
        im = im.resize((args.width, round(im.height * args.width / im.width)), Image.LANCZOS)
    rgb.append(im.convert("RGB"))
    print(f"  frame {n + 1}/{len(schedule)}: step {viewer.snapshots[i]['step']}", end="\r")

# --- one palette for all frames (sample frames + segmentation swatches), no dithering -----------
grey = np.linspace(0, 255, 64)
alpha = viewer.seg_alpha
swatch = np.stack([(1 - alpha) * grey[:, None] + alpha * np.asarray(c, float)[None] for c in SEG_COLORS.values()])
swatch = np.repeat(np.repeat(swatch.astype(np.uint8), 8, axis=0), 8, axis=1)           # (labels*8, 64*8, 3)
sample = rgb[:: max(1, len(rgb) // 12)]
W = rgb[0].width
canvas = Image.new("RGB", (W, sum(f.height for f in sample) + swatch.shape[0] * 4))
y = 0
for f in sample:
    canvas.paste(f, (0, y)); y += f.height
sw = Image.fromarray(swatch).resize((W, swatch.shape[0] * 4), Image.NEAREST)
canvas.paste(sw, (0, y))
palette = canvas.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
frames = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in rgb]

frames[0].save(args.out, save_all=True, append_images=frames[1:], duration=FRAME_MS, loop=0, optimize=True)
print(f"\n{len(frames)} frames, {len(frames) * FRAME_MS / 1000:.1f} s -> {args.out} "
      f"({args.out.stat().st_size / 1e6:.1f} MB)")
