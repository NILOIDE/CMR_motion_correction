"""Headroom of the worst planes: for each of the worst-10% planes of a run (fixed evaluator lines), search ONLY that
plane's 6-DOF pose for Dice directly (all other planes at the run's solution) and report the best Dice reached.
Diagnostic, not a training method: it separates planes that alignment could still fix from planes whose Dice is
capped by the segmentations themselves.

Usage: python night_headroom.py RUN_NAME [worst_fraction]
Writes <OUT>/headroom_<run>.csv.
"""
import sys
import time

import numpy as np
import pandas as pd
import torch

import night_gs as ng


def main():
    run = sys.argv[1]
    frac = float(sys.argv[2]) if len(sys.argv) > 2 else 0.10
    ev_all = pd.read_csv(ng.OUT / "eval_slices.csv", dtype={"subject_id": str})
    ev = ev_all[ev_all.run == run].dropna(subset=["dice"]).sort_values("dice")
    worst = ev.head(max(1, int(np.ceil(frac * len(ev)))))
    E = ng.Evaluator()
    dev = E.device
    rng = np.random.default_rng(0)
    rows = []
    for w in worst.itertuples():
        t0 = time.time()
        s = E.subject(w.subject_id)
        # the run's solution for every plane (same reconstruction as Evaluator.evaluate_run)
        h = pd.read_csv(ng.OUT / run / w.subject_id / "metrics" / "history.csv")
        h = h[h.is_best == 1] if (h.is_best == 1).any() else h[h.step == h.step.max()]
        h = h.drop_duplicates(["image", "slice"])
        N = len(s["planes"])
        theta = torch.zeros(N, 3, device=dev)
        t = torch.zeros(N, 3, device=dev)
        for r in h.itertuples():
            i = s["idx"].get(f"{r.image}:{int(r.slice)}")
            if i is None:
                continue
            p = s["planes"][i]
            theta[i] = torch.tensor([r.rot_x_deg, r.rot_y_deg, r.rot_z_deg], device=dev)
            t[i] = torch.tensor(r.dx_mm * p.u_hat + r.dy_mm * p.v_hat + r.dz_mm * p.normal, dtype=torch.float32, device=dev)
        k = s["idx"][f"{w.image}:{int(w.slice)}"]
        p = s["planes"][k]
        smp = s["smp"]
        mine = smp.plane_lines[k]
        basis = torch.tensor(np.stack([p.u_hat, p.v_hat, p.normal]), dtype=torch.float32, device=dev)

        def dice_at(q):
            """(Dice of the plane's lines, coverage = number of counted (label, frame, line) comparisons)."""
            th, tt = theta.clone(), t.clone()
            tt[k] = torch.tensor(q[:3], dtype=torch.float32, device=dev) @ basis
            th[k] = torch.tensor(q[3:], dtype=torch.float32, device=dev) @ basis
            with torch.no_grad():
                prof = smp.profiles(th, tt, "seg")
                d = smp.dice_lines(th, tt, E.labels, E.labels_sax, prof)[mine].cpu().numpy()
                cov = sum(int(c[mine].sum()) for _, c, _, _ in smp.counted_labels(*prof, E.labels, E.labels_sax))
            w_ = smp.npts[mine].cpu().numpy()
            ok = np.isfinite(d)
            return (float((d[ok] * w_[ok]).sum() / w_[ok].sum()) if ok.any() else np.nan), cov

        q0 = np.concatenate([(t[k] @ basis.T).cpu().numpy(), (theta[k] @ basis.T).cpu().numpy()])
        d0, cov0 = dice_at(q0)
        best_q, best = q0.copy(), d0 if np.isfinite(d0) else -1.0
        # guards against degenerate optima: moves within +-LIM of the run's pose, and no fewer label comparisons
        lim = np.array([12, 12, 12, 8, 8, 8.0])
        for it in range(1500):
            if it < 900:
                q = q0 + rng.uniform(-1, 1, 6) * lim
            elif it < 1300:
                q = best_q + rng.uniform(-1, 1, 6) * lim / 4
            else:
                q = best_q + rng.normal(0, 1, 6) * np.array([1, 1, 1, .5, .5, .5])
            q = np.clip(q, q0 - lim, q0 + lim)
            d, cov = dice_at(q)
            if np.isfinite(d) and d > best and cov >= cov0:
                best_q, best = q, d
        shift = best_q[:3] - q0[:3]
        rows.append(dict(subject_id=w.subject_id, image=w.image, slice=int(w.slice), dice_acquired=w.dice0, dice_run=w.dice,
                         dice_run_lines=d0, dice_ceiling=best, move_inplane_mm=float(np.hypot(*shift[:2])),
                         move_dz_mm=float(shift[2]), ncc_run=w.ncc))
        print(f"{w.subject_id} {w.image}:{int(w.slice)}  run {w.dice:.3f}  ceiling {best:.3f}  ({time.time() - t0:.0f} s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(ng.OUT / f"headroom_guarded_{run}.csv", index=False, float_format="%.4f")
    print(f"\nworst {len(df)} planes: run mean Dice {df.dice_run.mean():.3f}, ceiling mean {df.dice_ceiling.mean():.3f}; "
          f"planes with ceiling < 0.3: {(df.dice_ceiling < 0.3).sum()}, < 0.5: {(df.dice_ceiling < 0.5).sum()}")


if __name__ == "__main__":
    main()
