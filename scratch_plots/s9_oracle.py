"""Diagnostic ceiling: the best Dice one SAX slice can reach when ONLY its 6-DOF pose is searched for Dice directly
(all other planes held at a run's solution). Not a training method -- it tells how much an NCC-trained run leaves.

Usage: python s9_oracle.py RUN_DIR [SUBJECT] [SLICE]
Prints the run's slice Dice/NCC, the Dice-optimal pose (random search + local refinement) with its NCC, and the
NCC-optimal pose found the same way with its Dice.
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
    run = Path(sys.argv[1])
    sid = sys.argv[2] if len(sys.argv) > 2 else "1000071"
    k_slice = int(sys.argv[3]) if len(sys.argv) > 3 else 9
    cfg = json.loads((run / "overall_metrics" / "run_config.json").read_text())
    args = SimpleNamespace(**cfg)
    args.no_save_images = True
    args.outside_mask_weight = 0.0
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
    labels = [int(x) for x in args.seg_labels.split(",")]
    labels_sax = [int(x) for x in args.sax_seg_labels.split(",")]
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
    k = index[f"SAX:{k_slice}"]
    p = planes[k]
    mine = smp.plane_lines[k]
    if len(mine) == 0:
        print("slice has no lines in this configuration")
        return
    basis = torch.tensor(np.stack([p.u_hat, p.v_hat, p.normal]), dtype=torch.float32, device=device)   # rows u, v, n

    def evaluate(q):
        """q = (dx, dy, dz, rot_u, rot_v, rot_n) in the slice frame (mm, deg) -> (ncc, dice) of the slice's lines."""
        th, tt = theta.clone(), t.clone()
        tt[k] = torch.tensor(q[:3], dtype=torch.float32, device=device) @ basis
        th[k] = torch.tensor(q[3:], dtype=torch.float32, device=device) @ basis
        with torch.no_grad():
            c = smp.ncc_lines(th, tt, False)[mine]
            dsc = smp.dice_lines(th, tt, labels, labels_sax)[mine]
        w = smp.npts[mine]
        return float((c * w).sum() / w.sum()), float(np.nanmean(dsc.cpu().numpy()))

    q0 = np.concatenate([(t[k] @ basis.T).cpu().numpy(), (theta[k] @ basis.T).cpu().numpy()])
    print(f"run solution  q = {np.round(q0, 2)}  ncc {evaluate(q0)[0]:.3f}  dice {evaluate(q0)[1]:.3f}")
    rng = np.random.default_rng(0)
    for target, idx in (("dice", 1), ("ncc", 0)):
        best_q, best = q0.copy(), evaluate(q0)
        scale = np.array([20, 20, 20, 10, 10, 10.0])
        for it in range(2500):
            q = (rng.uniform(-1, 1, 6) * scale + (q0 if it < 1500 else best_q)) if it < 2000 else best_q + rng.normal(0, 1, 6) * np.array([1, 1, 1, .5, .5, .5])
            if it >= 1500 and it < 2000:
                q = best_q + rng.uniform(-1, 1, 6) * scale / 4
            s = evaluate(q)
            if s[idx] > best[idx]:
                best_q, best = q, s
        print(f"{target}-optimal q = {np.round(best_q, 2)}  ncc {best[0]:.3f}  dice {best[1]:.3f}")


if __name__ == "__main__":
    main()
