"""Check metrics/sax_lines.csv: run 1000071 with a forced tilt between SAX slices 5 and 6 so that their crossing line
exists, then verify the stored line against the stored history affines of the same snapshot (independent geometry).

The optimization itself is short (it only has to produce snapshots); the tilt is imposed through a monkeypatched
align() start, so the run uses the real pipeline code for building, storing and writing the rows."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, r"C:\Users\Nil\Documents\git\UKBB_implicit_baselines")
import motion_gradient as mg  # noqa: E402

OUT = Path(r"C:\Users\Nil\Documents\git\UKBB_implicit_baselines\scratch_plots\sax_lines_check")

# start the optimization from a pose where SAX slices 5 and 6 are tilted +-12 deg about their row axis
orig_align = mg.align


def tilted_align(final_sampler, args, stages=None):
    planes = final_sampler.planes
    idx = {p.key: i for i, p in enumerate(planes)}
    th0 = torch.zeros(len(planes), 3)
    for key, sgn in (("SAX:5", 1.0), ("SAX:6", -1.0)):
        th0[idx[key]] = torch.tensor(sgn * 12.0 * planes[idx[key]].u_hat, dtype=torch.float32)
    real_zeros = torch.zeros

    def zeros(*shape, **kw):                    # the first (N, 3) zeros in align() is theta_deg
        z = real_zeros(*shape, **kw)
        if not zeros.done and tuple(shape) == (len(planes), 3):
            zeros.done = True
            z = th0.to(z.device, z.dtype)
        return z
    zeros.done = False
    torch.zeros = zeros
    try:
        return orig_align(final_sampler, args, stages)
    finally:
        torch.zeros = real_zeros


mg.align = tilted_align
sys.argv = ["motion_gradient.py", "--out-root", str(OUT.parent), "--run-name", OUT.name, "--max-subjects", "1",
            "--steps", "50", "--optimizer", "sgd", "--lr", "0.001", "--rot-lr", "0.001", "--sax-sax-weight", "1",
            "--sax-sax-radius-mm", "40", "--blur-sigma", "0", "--psf-scale-start", "1"]
mg.main()

m = OUT / "1000071" / "metrics"
sl = pd.read_csv(m / "sax_lines.csv")
h = pd.read_csv(m / "history.csv")
print(f"sax_lines.csv: {len(sl)} rows, {sl.step.nunique()} snapshots x {len(sl) // sl.step.nunique()} pairs; "
      f"crossing rows {int(sl.crossing.sum())}, rows with used points {int((sl.n_points > 0).sum())}")
row = sl[(sl.slice_a == 5) & (sl.slice_b == 6) & (sl.step == 0) & (sl.is_best == 0)].iloc[0]
print(row[["crossing", "n_points", "axis_dist_mm", "ncc", "t_used_lo", "t_used_hi", "t_img_lo", "t_img_hi"]].to_dict())


def affine(image, z, step):
    r = h[(h.image == image) & (h["slice"] == z) & (h.step == step) & (h.is_best == 0)].iloc[0]
    return np.vstack([r[mg.AFFINE_COLUMNS].to_numpy(float).reshape(3, 4), [0, 0, 0, 1]])


A5, A6 = affine("SAX", 5, 0), affine("SAX", 6, 0)
x0 = row[["x0_x", "x0_y", "x0_z"]].to_numpy(float)
d = row[["d_x", "d_y", "d_z"]].to_numpy(float)
for name, A in (("slice 5", A5), ("slice 6", A6)):
    n = np.cross(A[:3, 0], A[:3, 1]); n /= np.linalg.norm(n)
    off = [abs((x0 + tt * d - A[:3, 3]) @ n) for tt in (row.t_img_lo, row.t_used_lo, row.t_used_hi, row.t_img_hi)]
    inv = np.linalg.inv(A)
    vox = [inv @ np.append(x0 + tt * d, 1) for tt in (row.t_img_lo, row.t_img_hi)]
    print(f"  {name}: max distance of the stored line from the stored posed plane {max(off):.2e} mm; "
          f"t_img ends in voxels (i, j): {np.round(vox[0][:2], 1)} / {np.round(vox[1][:2], 1)}")
print(f"  ordering t_img_lo <= t_used_lo <= t_used_hi <= t_img_hi: "
      f"{row.t_img_lo <= row.t_used_lo <= row.t_used_hi <= row.t_img_hi}; used length {row.t_used_hi - row.t_used_lo:.0f} mm "
      f"(radius 40 mm -> at most {2 * 40:.0f} mm / sin(angle to the axis))")
