"""Objective landscape of one SAX slice: all other planes held at a run's final poses, the slice's pose scanned.

Usage: python investigate_apex.py RUN_DIR SUBJECT SLICE [--blur 0] [--psf 1]
Writes scratch_plots/line_check/<subject>_slice<k>_landscape[_blur..].png: NCC (the optimized similarity over the
slice's lines) and Dice over in-plane shift (dx, dy) at the solution's dz, and over (dy, dz) at the solution's dx.
Marks: acquired pose (0, 0), the run's solution, the scan maximum. Also prints a finite-difference gradient check
and the objective's sensitivity to 1e-3 mm perturbations (float32 noise floor).
"""
import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure

sys.path.insert(0, r"C:\Users\Nil\Documents\git\UKBB_implicit_baselines")
import motion_gradient as mg  # noqa: E402

OUT = Path(r"C:\Users\Nil\Documents\git\UKBB_implicit_baselines\scratch_plots\line_check")
ST = mg.PLOT_STYLE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run"), ap.add_argument("subject"), ap.add_argument("slice", type=int)
    ap.add_argument("--blur", type=float, default=0.0), ap.add_argument("--psf", type=float, default=1.0)
    ap.add_argument("--range", type=float, default=25.0), ap.add_argument("--step", type=float, default=1.0)
    a = ap.parse_args()
    run = Path(a.run)
    cfg = json.loads((run / "overall_metrics" / "run_config.json").read_text())
    args = SimpleNamespace(**cfg)
    args.no_save_images = True
    device = torch.device("cuda")
    data = mg.load_subject(Path(cfg["root"]) / a.subject, args)
    sax_arr, sax_aff = data["sax"]
    sax_planes = mg.make_planes("SAX", sax_arr, sax_aff, args.sax_thickness, None, data["sax_seg"], device)
    lax_planes = []
    for view, d in data["lax"].items():
        lax_planes += mg.make_planes(view, d["arr"], d["affine"], d["thickness"], d["mask"], d["seg"], device)
    planes = sax_planes + lax_planes
    index = {p.key: i for i, p in enumerate(planes)}
    lines = []
    with torch.no_grad():
        from itertools import combinations
        for pa, pb in [(s, l) for s in sax_planes for l in lax_planes] + list(combinations(lax_planes, 2)):
            ln = mg.intersection_line(pa, pb, args, device)
            if ln is not None:
                lines.append((index[pa.key], index[pb.key], ln))
    smp = mg.LineSampler(planes, lines, args.n_band, device, args.psf_model, args.psf_tbw, max(a.psf, 1.0))
    smp.set_blur(a.blur)
    smp.set_psf_scale(a.psf)
    labels = [int(x) for x in args.seg_labels.split(",")]
    labels_sax = [int(x) for x in args.sax_seg_labels.split(",")]

    # the run's final poses (translations relative to the mean LAX pose: a global shift, irrelevant to the objective)
    h = pd.read_csv(run / a.subject / "metrics" / "history.csv")
    h = h[h.is_best == 1].drop_duplicates(["image", "slice"])
    N = len(planes)
    theta = torch.zeros(N, 3, device=device)
    t = torch.zeros(N, 3, device=device)
    for _, r in h.iterrows():
        i = index[f"{r.image}:{int(r.slice)}"]
        p = planes[i]
        theta[i] = torch.tensor([r.rot_x_deg, r.rot_y_deg, r.rot_z_deg], device=device)
        t[i] = torch.tensor(r.dx_mm * p.u_hat + r.dy_mm * p.v_hat + r.dz_mm * p.normal, dtype=torch.float32, device=device)
    k = index[f"SAX:{a.slice}"]
    p = planes[k]
    mine = smp.plane_lines[k]
    u, v, n = [torch.tensor(x, dtype=torch.float32, device=device) for x in (p.u_hat, p.v_hat, p.normal)]
    sol = (float(t[k] @ u), float(t[k] @ v), float(t[k] @ n))
    print(f"slice {a.slice}: {len(mine)} lines, solution dx {sol[0]:.2f} dy {sol[1]:.2f} dz {sol[2]:.2f} mm, "
          f"theta {theta[k].tolist()}")

    def score(tk):
        tt = t.clone()
        tt[k] = tk
        with torch.no_grad():
            img = smp.profiles(theta, tt, "img")
            c = smp.ncc_lines(theta, tt, False, img, smp.W_loss)[mine]
            seg = smp.profiles(theta, tt, "seg")
            d = smp.dice_lines(theta, tt, labels, labels_sax, seg)[mine]
        w = smp.npts[mine]
        return float((c * w).sum() / w.sum()), float(np.nanmean(d.cpu().numpy()))

    # --- float32 noise floor and gradient check at the solution
    base = score(t[k])[0]
    eps = 1e-3
    noise = max(abs(score(t[k] + eps * d)[0] - base) for d in (u, v, n))
    print(f"objective at the solution {base:.6f}; change for 1e-3 mm shifts <= {noise:.2e}")
    tt = t.clone().requires_grad_(True)
    th = theta.clone().requires_grad_(True)
    obj = (smp.similarity_lines(th, tt, args) * smp.npts).sum() / smp.npts.sum()
    obj.backward()
    g_auto = tt.grad[k]
    for name, d in (("u", u), ("v", v), ("n", n)):
        hh = 0.05
        tp, tm = t.clone(), t.clone()
        tp[k] += hh * d
        tm[k] -= hh * d
        with torch.no_grad():
            fp = float((smp.similarity_lines(theta, tp, args) * smp.npts).sum() / smp.npts.sum())
            fm = float((smp.similarity_lines(theta, tm, args) * smp.npts).sum() / smp.npts.sum())
        print(f"  d/d{name}: autograd {float(g_auto @ d):+.5f}  finite diff (h=0.05 mm) {(fp - fm) / (2 * hh):+.5f}")

    # --- landscapes
    rng = np.arange(-a.range, a.range + 1e-6, a.step)
    def scan(fix, axes_):
        ncc = np.zeros((len(rng), len(rng)))
        dice = np.zeros_like(ncc)
        for i, x in enumerate(rng):
            for j, y in enumerate(rng):
                tk = fix + x * axes_[0] + y * axes_[1]
                ncc[i, j], dice[i, j] = score(tk)
        return ncc, dice
    t_sol = t[k].clone()
    lands = [("in-plane (dx, dy) at the solution's dz", (u, v), ("dx (mm)", "dy (mm)"), (sol[0], sol[1])),
             ("through-plane (dy, dz) at the solution's dx", (v, n), ("dy (mm)", "dz (mm)"), (sol[1], sol[2]))]
    fig = Figure(figsize=(13, 11), layout="constrained", facecolor=ST["surface"])
    axes = fig.subplots(2, 2)
    for r, (title, ax_vecs, labs, sol2) in enumerate(lands):
        # scan around the acquired pose, with the other two pose components at the solution
        fix = t_sol - sol2[0] * ax_vecs[0] - sol2[1] * ax_vecs[1]
        ncc, dice = scan(fix, ax_vecs)
        for c, (val, name) in enumerate(((ncc, "NCC over the slice's lines"), (dice, "Dice over the slice's lines"))):
            ax = axes[r, c]
            im = ax.imshow(val.T, origin="lower", extent=(rng[0], rng[-1], rng[0], rng[-1]), cmap="viridis", aspect="equal")
            fig.colorbar(im, ax=ax, shrink=0.8)
            ax.contour(rng, rng, val.T, levels=8, colors="white", linewidths=0.4, alpha=0.5)
            ax.plot(0, 0, "s", color="white", ms=8, mfc="none", mew=1.5, label="acquired")
            ax.plot(sol2[0], sol2[1], "o", color=ST["orange"], ms=8, mfc="none", mew=2, label="run solution")
            ii, jj = np.unravel_index(np.nanargmax(val), val.shape)
            ax.plot(rng[ii], rng[jj], "x", color=ST["red"] if "red" in ST else "#e34948", ms=9, mew=2, label="scan max")
            ax.set_xlabel(labs[0]), ax.set_ylabel(labs[1])
            ax.set_title(f"{name}: {title}\nmax {np.nanmax(val):.3f} at ({rng[ii]:g}, {rng[jj]:g}); "
                         f"at solution {val[np.argmin(abs(rng - sol2[0])), np.argmin(abs(rng - sol2[1]))]:.3f}", fontsize=9, loc="left")
            if r == 0 and c == 0:
                ax.legend(loc="upper right", fontsize=8)
    tag = f"_{run.name}" + (f"_blur{a.blur:g}_psf{a.psf:g}" if a.blur or a.psf != 1 else "")
    fig.suptitle(f"{a.subject} SAX slice {a.slice} ({run.name}): other planes at the run's solution", x=0.01, ha="left", color=ST["ink"])
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{a.subject}_slice{a.slice}_landscape{tag}.png", dpi=110)
    print("written", OUT / f"{a.subject}_slice{a.slice}_landscape{tag}.png")


if __name__ == "__main__":
    main()
