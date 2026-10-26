"""Curvature of the objective along every pose DOF of every plane at a run's solution (central differences).

Usage: python investigate_curvature.py RUN_DIR SUBJECT
Prints, per plane, -d2f/dq2 for in-plane shifts (per mm^2), through-plane shift dz (per mm^2), in-plane rotation and
the two tilt axes (per deg^2; a 1 deg tilt of a LAX view moves the lines in a SAX slice 50 mm from its centre by
0.9 mm, so deg and mm are comparable), plus the mean over SAX and LAX planes. Small = weakly determined.
"""
import json
import sys
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, r"C:\Users\Nil\Documents\git\UKBB_implicit_baselines")
import motion_gradient as mg  # noqa: E402


def main():
    run, sid = Path(sys.argv[1]), sys.argv[2]
    cfg = json.loads((run / "overall_metrics" / "run_config.json").read_text())
    args = SimpleNamespace(**cfg)
    args.no_save_images = True
    device = torch.device("cuda")
    data = mg.load_subject(Path(cfg["root"]) / sid, args)
    sax_arr, sax_aff = data["sax"]
    sax_planes = mg.make_planes("SAX", sax_arr, sax_aff, args.sax_thickness, None, data["sax_seg"], device)
    lax_planes = []
    for view, d in data["lax"].items():
        lax_planes += mg.make_planes(view, d["arr"], d["affine"], d["thickness"], d["mask"], d["seg"], device)
    planes = sax_planes + lax_planes
    index = {p.key: i for i, p in enumerate(planes)}
    lines = []
    with torch.no_grad():
        for pa, pb in [(s, l) for s in sax_planes for l in lax_planes] + list(combinations(lax_planes, 2)):
            ln = mg.intersection_line(pa, pb, args, device)
            if ln is not None:
                lines.append((index[pa.key], index[pb.key], ln))
    smp = mg.LineSampler(planes, lines, args.n_band, device, args.psf_model, args.psf_tbw)
    h = pd.read_csv(run / sid / "metrics" / "history.csv")
    h = h[h.is_best == 1].drop_duplicates(["image", "slice"])
    N = len(planes)
    theta = torch.zeros(N, 3, device=device)
    t = torch.zeros(N, 3, device=device)
    for _, r in h.iterrows():
        i = index[f"{r.image}:{int(r.slice)}"]
        p = planes[i]
        theta[i] = torch.tensor([r.rot_x_deg, r.rot_y_deg, r.rot_z_deg], device=device)
        t[i] = torch.tensor(r.dx_mm * p.u_hat + r.dy_mm * p.v_hat + r.dz_mm * p.normal, dtype=torch.float32, device=device)

    def f(th, tt):
        with torch.no_grad():
            return float((smp.similarity_lines(th, tt, args) * smp.npts).sum() / smp.npts.sum())

    f0 = f(theta, t)
    rows = []
    for i, p in enumerate(planes):
        if len(smp.plane_lines[i]) == 0:
            continue
        axes = {"shift_u": ("t", p.u_hat, 0.5), "shift_v": ("t", p.v_hat, 0.5), "dz": ("t", p.normal, 0.5),
                "rot_inplane": ("th", p.normal, 0.5), "tilt_u": ("th", p.u_hat, 0.5), "tilt_v": ("th", p.v_hat, 0.5)}
        row = dict(plane=p.key)
        for name, (kind, ax, hh) in axes.items():
            vec = torch.tensor(ax, dtype=torch.float32, device=device)
            vals = []
            for sgn in (+1, -1):
                th, tt = theta.clone(), t.clone()
                if kind == "t":
                    tt[i] += sgn * hh * vec
                else:
                    th[i] += sgn * hh * vec
                vals.append(f(th, tt))
            row[name] = -(vals[0] - 2 * f0 + vals[1]) / hh ** 2       # positive = a maximum along this axis
        rows.append(row)
    df = pd.DataFrame(rows).set_index("plane")
    pd.set_option("display.width", 200)
    print(f"{sid} ({run.name}): -d2f/dq2 at the solution (per mm^2 / per deg^2), objective {f0:.4f}")
    print(df.round(5).to_string())
    print("\nmean over SAX planes:")
    print(df[df.index.str.startswith("SAX")].mean().round(5).to_string())
    print("mean over LAX planes:")
    print(df[~df.index.str.startswith("SAX")].mean().round(5).to_string())


if __name__ == "__main__":
    main()
