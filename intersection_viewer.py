"""
2D intersection viewer: what the alignment score actually compares along each intersection line.

For one subject of a motion_bruteforce.py / motion_gradient.py run and one pose snapshot of its history.csv (default:
the saved best pose), every intersection line of the chosen plane(s) is recomputed exactly as the brute-force score
does (motion_bruteforce.GridSearcher: same geometry, PSF band, scored region, metric) and drawn as one figure row:

  image A | image B   the frame shown, the line over its extent inside both images (thin blue), the scored points
                      (thick, coloured by their contribution: green = agreeing, red = disagreeing), the LAX mask
                      contour (yellow, dashed)
  profiles            both PSF-band profiles of that frame along the line -- z-scored over the scored points for the
                      correlation metrics, the normalized intensities for l1 / l2; faint = not scored
  contribution        per point / window, what the metric averages: z_a * z_b (ncc, nmi, ncc_grad: NCC is their mean),
                      the local NCC of each window at its centre with its weight (lncc), the product of the normalized
                      gradients (ngf), -|a - b| (l1), -(a - b)^2 (l2);
                      per frame (thin) and averaged over frames (thick)

Each row's title gives the line's score (= motion_bruteforce's) and weight; the header the plane score (the weighted
mean over its lines), as a check that the viewer reproduces the optimized metric. --metric / --score-region override
the run's settings, to see how another metric would judge the same poses.

Usage:
  python intersection_viewer.py --run <run dir> --subject 1000071 --plane SAX:9 [--step best|0|1..] [--frame 0]
                                [--metric lncc] [--score-region centred]
  --plane may be given several times (default: every SAX slice); one PNG per plane in <run>/<subject>/intersections/.
"""
import argparse
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import motion_bruteforce as bf
import moco_nifti as mn
import motion_gradient as mg

DEFAULTS = dict(fp16=False, eval_batch=1024, top_k=8, tie_tol=1e-4, sax_overlap_weight=0.0, sax_overlap_step_mm=6.0,
                sax_overlap_radius_mm=0.0, outside_mask_weight=0.0, score_region="mask", score_half_length_mm=60.0,
                lncc_window_mm=20.0, lncc_min_std=0.25, grad_eps=0.5)


def load_run(run: Path, subject: str, device, overrides: dict) -> tuple:
    """Args of the run (+ overrides), the subject's planes, their pair weights and PSF band, and the history table."""
    cfg = mg.legacy_nmi(json.loads((run / "overall_metrics" / "run_config.json").read_text()))
    args = Namespace(**{k: v for k, v in cfg.items() if k != "command_line"})
    for k, v in DEFAULTS.items():                      # motion_gradient runs / older runs lack some settings
        if not hasattr(args, k):
            setattr(args, k, v)
    for k, v in overrides.items():
        if v is not None:
            setattr(args, k, v)
    if getattr(args, "no_psf", False):
        args.n_band = 1
    data = mg.prepare_subject(mn.load_subject(Path(args.root) / subject, args), args)
    sax, lax = mg.build_planes(data, args, device, with_seg=False)
    planes = sax + lax
    lines = mg.build_lines(planes, args, device)
    smp = mg.LineSampler(planes, lines, args.n_band, device, args.psf_model, args.psf_tbw, 1.0)
    weights = {frozenset((ia, ib)): float(n) for ia, ib, n in zip(smp.IA.tolist(), smp.IB.tolist(), smp.npts.tolist())}
    hist = pd.read_csv(run / subject / "metrics" / "history.csv", dtype={"subject_id": str})
    return args, planes, weights, (smp.u_img, smp.band_weights()), hist


def set_poses(gs: bf.GridSearcher, planes, hist: pd.DataFrame, step: str) -> str:
    """Poses of a history snapshot ('best' or a step number); translations relative to the mean LAX pose (a global
    shift, which leaves every line unchanged)."""
    rows = hist[hist["is_best"] == 1] if step == "best" else hist[(hist["step"] == int(step)) & (hist["is_best"] == 0)]
    if rows.empty:
        raise SystemExit(f"no history rows for step {step}")
    idx = {p.key: i for i, p in enumerate(planes)}
    for _, r in rows.iterrows():
        i = idx.get(f"{r['image']}:{int(r['slice'])}")
        if i is None:
            continue
        p = planes[i]
        gs.theta[i] = torch.tensor([r["rot_x_deg"], r["rot_y_deg"], r["rot_z_deg"]], device=gs.device)
        gs.t[i] = torch.tensor(r["dx_mm"] * p.u_hat + r["dy_mm"] * p.v_hat + r["dz_mm"] * p.normal,
                               device=gs.device, dtype=torch.float32)
    return f"step {int(rows['step'].iloc[0])}" + (" (best)" if step == "best" else "")


@torch.no_grad()
def line_detail(gs: bf.GridSearcher, m: int, r: int) -> dict:
    """Everything along the m-r line at the current poses, from the same code as GridSearcher.line_score."""
    pm, pr = gs.current(m, 1), gs.current(r, 1)
    g = gs.line_geometry(m, pm, r, pr)
    a, b = gs.band_profiles(m, pm, r, pr, g)
    w = gs.point_weights(g)
    out = dict(t=gs.T.cpu().numpy(), in_img=g["in_img"][0].cpu().numpy(), valid=g["valid"][0].cpu().numpy(),
               w=w[0].cpu().numpy(), a=a[0].cpu().numpy(), b=b[0].cpu().numpy(), vox_a=g["vox_m"][0].cpu().numpy(),
               vox_b=g["vox_r"][0].cpu().numpy(), score=float(gs.line_score(m, pm, r, pr)[0]),
               n_valid=int(g["valid"].sum()))
    if gs.args.metric == "lncc":
        loc, wt = gs.local_ncc(a, b, w)
        out.update(loc=loc[0].cpu().numpy(), loc_w=wt[0].cpu().numpy())
    if gs.args.metric == "ngf":
        na, nb, wg = gs.normalized_gradients(a, b, w)
        out.update(na=na[0].cpu().numpy(), nb=nb[0].cpu().numpy(), wg=wg[0, 0].cpu().numpy())
    return out


def zscore(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    m = (x * w).sum(-1, keepdims=True) / max(w.sum(), 1e-6)
    sd = np.sqrt(((x - m) ** 2 * w).sum(-1, keepdims=True) / max(w.sum(), 1e-6)) + 1e-9
    return (x - m) / sd


def contribution(dt: dict, metric: str, step_mm: float) -> tuple:
    """(positions, per-frame values (F, n), weights (F, n) or None, label) of what the metric averages."""
    t, w = dt["t"], dt["w"]
    if metric == "lncc":
        k = dt["a"].shape[-1] - dt["loc"].shape[-1] + 1
        return t[:len(t) - k + 1] + (k - 1) / 2 * step_mm, dt["loc"], dt["loc_w"], "local NCC per window (at its centre)"
    if metric == "ngf":
        return t[1:-1], dt["na"] * dt["nb"], None, "na * nb (normalized gradients; score = sum / sum((na^2 + nb^2) / 2))"
    if metric == "l1":
        return t, -np.abs(dt["a"] - dt["b"]), None, "-|a - b|  (score = weighted mean)"
    if metric == "l2":
        return t, -(dt["a"] - dt["b"]) ** 2, None, "-(a - b)^2  (score = weighted mean)"
    return t, zscore(dt["a"], w) * zscore(dt["b"], w), None, "z_a * z_b  (NCC = weighted mean)"


def plot_plane(gs, planes, weights, key: str, frame: int, title: str, out: Path, refs_all: bool) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    args = gs.args
    metric = args.metric
    idx = {p.key: i for i, p in enumerate(planes)}
    m = idx[key]
    refs = [r for r in range(len(planes)) if r != m and frozenset((m, r)) in weights
            and (refs_all or planes[r].image != "SAX")]
    if not refs:
        print(f"{key}: no intersection lines")
        return
    details = [(r, line_detail(gs, m, r)) for r in refs]
    wsum = sum(weights[frozenset((m, r))] for r in refs)
    plane_score = sum(weights[frozenset((m, r))] * d["score"] for r, d in details) / wsum
    absolute = metric in ("l1", "l2")

    fig, axes = plt.subplots(len(refs), 4, figsize=(22, 5.2 * len(refs)), squeeze=False,
                             gridspec_kw=dict(width_ratios=(1, 1, 1.6, 1.6)))
    for row, (r, dt) in zip(axes, details):
        t, v, w = dt["t"], dt["valid"], dt["w"]
        pos, vals, vw, ylabel = contribution(dt, metric, args.line_step_mm)
        mean_vals = vals.mean(0)
        # colour of each scored point: the contribution there (for lncc: of the window centred on it)
        point_c = np.interp(t, pos, mean_vals) if metric in ("lncc", "ngf") else mean_vals
        lim = 1.0 if metric in ("lncc", "ngf") else (0.5 if absolute else 2.0)
        for ax, i, vox in ((row[0], m, dt["vox_a"]), (row[1], r, dt["vox_b"])):
            p = planes[i]
            img = gs.img[i][0, frame].float().cpu().numpy()
            ax.imshow(img.T, cmap="gray", origin="lower")
            if gs.mask[i] is not None:
                ax.contour(gs.mask[i][0, 0].float().cpu().numpy().T, levels=[0.5], colors="#eda100",
                           linewidths=1.0, linestyles="--")
            ax.plot(vox[dt["in_img"], 0], vox[dt["in_img"], 1], color="#2a78d6", lw=0.8, alpha=0.7)
            keep = v[:-1] & v[1:]
            lc = LineCollection(np.stack([vox[:-1], vox[1:]], 1)[keep], cmap="RdYlGn",
                                norm=plt.Normalize(-lim, 0 if absolute else lim), linewidths=3.5)
            lc.set_array(point_c[:-1][keep])
            ax.add_collection(lc)
            if v.any():
                ax.plot(*vox[np.flatnonzero(v)[0]], "o", color="white", ms=4)       # start of the scored part (t low)
            ax.set_title(f"{p.key}  (frame {frame})", fontsize=10, loc="left")
            ax.set_xticks([]), ax.set_yticks([])
        row[0].set_title(f"{planes[m].key} x {planes[r].key}: {metric} {dt['score']:.3f}, {dt['n_valid']} scored points, "
                         f"line weight {weights[frozenset((m, r))]:.0f}\n{planes[m].key} (frame {frame})",
                         fontsize=10, loc="left")

        ax = row[2]
        pa, pb = (dt["a"], dt["b"]) if absolute else (zscore(dt["a"], w), zscore(dt["b"], w))
        for prof, color, name in ((pa, "#2a78d6", planes[m].key), (pb, "#eb6834", planes[r].key)):
            ax.plot(t, np.where(v, prof[frame], np.nan), color=color, lw=1.6, label=name)
            ax.plot(t, np.where(dt["in_img"] & ~v, prof[frame], np.nan), color=color, lw=0.8, alpha=0.3)
        ax.axhline(0, color="#c3c2b7", lw=0.8)
        ax.set_xlabel("position along the line (mm, 0 = point nearest both pivots)")
        ax.set_ylabel("normalized intensity" if absolute else "z-scored intensity (over the scored points)")
        ax.legend(fontsize=8, loc="upper right")
        ax.set_title(f"profiles, frame {frame} (faint = inside both images, not scored; --score-region {args.score_region})",
                     fontsize=9, loc="left")

        ax = row[3]
        shown = (np.interp(pos, t, v.astype(float)) > 0.5) if metric != "lncc" else (vw.max(0) > 0)
        for f in range(vals.shape[0]):
            ax.plot(pos, np.where(shown, vals[f], np.nan), color="#898781", lw=0.6, alpha=0.6)
        mv = np.where(shown, mean_vals, np.nan)
        ax.plot(pos, mv, color="#0b0b0b", lw=1.6, label="mean over frames")
        ax.fill_between(pos, 0, mv, where=np.nan_to_num(mv) < 0, color="#d03b3b", alpha=0.35)
        ax.fill_between(pos, 0, mv, where=np.nan_to_num(mv) > 0, color="#1baf7a", alpha=0.25)
        if vw is not None:
            ax2 = ax.twinx()
            ax2.plot(pos, vw.mean(0), color="#2a78d6", lw=0.8, alpha=0.6)
            ax2.set_ylabel("window weight", color="#2a78d6")
        ax.axhline(0, color="#c3c2b7", lw=0.8)
        ax.set_xlabel("position along the line (mm)")
        ax.set_ylabel(ylabel)
        bad = np.nan_to_num(mv) < 0
        ax.set_title(f"contribution: {int(bad.sum())} of {int(shown.sum())} {'windows' if metric == 'lncc' else 'points'} "
                     f"negative" if not absolute else "contribution per scored point", fontsize=9, loc="left")
    fig.suptitle(f"{key} at {title}: plane score ({metric}, weighted mean over its lines) = {plane_score:.3f}",
                 x=0.01, ha="left", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=90)
    plt.close(fig)
    print(f"{key}: {metric} score {plane_score:.3f} -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="Run directory (holds overall_metrics/run_config.json)")
    ap.add_argument("--subject", type=str, required=True)
    ap.add_argument("--plane", action="append", default=None, help="Plane key, e.g. SAX:9 or LAX_2Ch:0 (repeatable; "
                                                                    "default: every SAX slice)")
    ap.add_argument("--step", type=str, default="best", help="History snapshot: 'best' or a step number (0 = acquired)")
    ap.add_argument("--frame", type=int, default=0, help="Cardiac frame shown (index into the run's --n-frames)")
    ap.add_argument("--all-lines", action="store_true", help="Also show SAX-SAX lines (only where they exist)")
    ap.add_argument("--metric", choices=["ncc", "ncc_grad", "lncc", "ngf", "l1", "nmi", "l2"], default=None,
                    help="Judge with this metric instead of the run's")
    ap.add_argument("--score-region", choices=["mask", "full", "centred"], default=None, help="Override the run's")
    ap.add_argument("--score-half-length-mm", type=float, default=None, help="Override the run's")
    ap.add_argument("--lncc-window-mm", type=float, default=None, help="Override the run's")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    device = torch.device(a.device)
    overrides = dict(metric=a.metric, score_region=a.score_region, score_half_length_mm=a.score_half_length_mm,
                     lncc_window_mm=a.lncc_window_mm)
    args, planes, weights, band, hist = load_run(a.run, a.subject, device, overrides)
    gs = bf.GridSearcher(planes, weights, band, args, device)
    title = set_poses(gs, planes, hist, a.step)
    keys = a.plane or [p.key for p in planes if p.image == "SAX"]
    tag = f"{args.metric}_{args.score_region}"
    for key in keys:
        out = a.run / a.subject / "intersections" / f"{key.replace(':', '_')}_{a.step}_f{a.frame}_{tag}.png"
        plot_plane(gs, planes, weights, key, a.frame, title, out, a.all_lines)


if __name__ == "__main__":
    main()
