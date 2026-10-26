"""Why are SAX slices 9 / 10 of 1000071 off after motion_bruteforce? Segmentations are used only as a diagnostic.

A. SAX segmentation content per slice (LV blood pool / myocardium / RV areas).
B. Where each SAX plane sits relative to the LV apex seen in the LAX segmentations (acquired and final poses).
C. For slices 8-10 (others at the final poses): the NCC score and the LV segmentation agreement (Dice of labels 1, 2
   along the lines) on the coarse-to-fine grid of the search; NCC-best vs Dice-best poses.
"""
import json
import math
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import motion_bruteforce as bf  # noqa: E402
import motion_gradient as mg  # noqa: E402

RUN = Path(r"C:\Users\Nil\Documents\data\ukbb_alignment_bf\20261026-033840_bf")
SID = "1000071"
LV_LABELS = (1, 2)
dev = torch.device("cuda")

cfg = json.loads((RUN / "overall_metrics" / "run_config.json").read_text())
args = Namespace(**{k: v for k, v in cfg.items() if k != "command_line"})
ONLY_C = len(sys.argv) > 1                       # usage: bf_apex_diag.py [mask dilation mm]: part C only, other mask
if ONLY_C:
    args.mask_dilation_mm = float(sys.argv[1])
    args.min_line_mm = 20.0 if args.mask_dilation_mm < 50 else args.min_line_mm   # tighter masks shorten apical lines
    if len(sys.argv) > 3:                        # search box override: t range (mm), rot range (deg)
        args.t_range_mm, args.rot_range_deg = float(sys.argv[2]), float(sys.argv[3])
    print(f"mask dilation {args.mask_dilation_mm} mm, min line {args.min_line_mm} mm, "
          f"box +-{args.t_range_mm} mm / +-{args.rot_range_deg} deg")
data = mg.load_subject(Path(args.root) / SID, args)
sax_planes, lax_planes = mg.build_planes(data, args, dev, with_seg=True)
planes = sax_planes + lax_planes
idx = {p.key: i for i, p in enumerate(planes)}

# ---- A: SAX segmentation content ---------------------------------------------------------------------------------
print("A. SAX segmentation areas (mm^2, max over frames)")
for p in ([] if ONLY_C else sax_planes):
    px = p.spacing[0] * p.spacing[1]
    areas = {lab: float((p.seg == lab).flatten(1).sum(1).max()) * px for lab in (1, 2, 3)}
    print(f"  slice {p.slice_idx:2d}: LV {areas[1]:7.0f}  myo {areas[2]:7.0f}  RV {areas[3]:7.0f}")

# ---- poses: acquired and final (history is_best rows; translations relative to the mean LAX pose, which is a global
# shift and does not change any line) --------------------------------------------------------------------------------
hist = pd.read_csv(RUN / SID / "metrics" / "history.csv", dtype={"subject_id": str})
best = hist[hist["is_best"] == 1]
theta_f = torch.zeros(len(planes), 3, device=dev)
t_f = torch.zeros(len(planes), 3, device=dev)
for _, r in best.iterrows():
    i = idx[f"{r['image']}:{int(r['slice'])}"]
    p = planes[i]
    theta_f[i] = torch.tensor([r["rot_x_deg"], r["rot_y_deg"], r["rot_z_deg"]], device=dev)
    t_f[i] = torch.tensor(r["dx_mm"] * p.u_hat + r["dy_mm"] * p.v_hat + r["dz_mm"] * p.normal, device=dev, dtype=torch.float32)

# ---- B: apex position from the LAX segmentations -----------------------------------------------------------------
axis = mg.long_axis_line(lax_planes)
L0, ell = axis[0], axis[1] / np.linalg.norm(axis[1])
apex_side = np.sign(ell @ (sax_planes[-1].center - sax_planes[0].center))   # +ell points from base to apex
ell = ell * apex_side
print("\nB. LV extent along the long axis (mm from L0, + towards the apex) in each LAX view, and SAX slice positions")


def world_pts(p, mask_ij, theta, t):
    """World coordinates of voxels (i, j) of plane p at pose (theta, t)."""
    A = p.affine
    X = mask_ij @ A[:3, :2].T + A[:3, 3]
    R = mg.rotation_matrices(theta[None])[0]
    c = p.rot_centre
    return (X - c) @ R.T + c + t


for label, (th, tt) in (() if ONLY_C else (("acquired", (torch.zeros_like(theta_f), torch.zeros_like(t_f))),
                                            ("final", (theta_f, t_f)))):
    th, tt = th.cpu().numpy().astype(float), tt.cpu().numpy().astype(float)
    print(f"  [{label}]")
    for p in lax_planes:
        i = idx[p.key]
        lv = (p.seg[:, 0] == 2).any(0).cpu().numpy()                      # myocardium, any frame
        ij = np.argwhere(lv).astype(float)
        s = (world_pts(p, ij, th[i], tt[i]) - L0) @ ell
        print(f"    {p.image}: myocardium from {s.min():7.1f} to {s.max():7.1f} mm (apex tip at {s.max():.1f})")
    for p in sax_planes:
        i = idx[p.key]
        c = world_pts(p, np.array([[(p.H - 1) / 2, (p.W - 1) / 2]]), th[i], tt[i])[0]
        # where the posed plane crosses the long axis
        n = mg.rotation_matrices(th[i][None])[0] @ p.normal
        s_axis = float(((c - L0) @ n) / (ell @ n))
        print(f"    SAX {p.slice_idx:2d}: crosses the long axis at {s_axis:7.1f} mm")


# ---- C: NCC vs segmentation agreement on the search grid ----------------------------------------------------------
class DiagSearcher(bf.GridSearcher):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.seg = [p.seg.transpose(0, 1).contiguous() if p.seg is not None else None for p in self.planes]

    def line_dice(self, m, pm, r, pr):
        """Mean over frames and LV labels of the Dice of the two segmentations along the m-r line (valid points)."""
        na, nb = pm.Np, pr.Np
        d = torch.cross(na, nb, dim=1)
        d = d / d.norm(dim=1, keepdim=True).clamp_min(1e-3)
        anchor = 0.5 * (pm.Cp + pr.Cp)
        rhs = torch.stack([(na * pm.Op).sum(1), (nb * pr.Op).sum(1), (d * anchor).sum(1)], 1)
        pts = mg.solve3_rows(na, nb, d, rhs)[:, None, :] + self.T[None, :, None] * d[:, None, :]
        va, vb = self.to_voxel(m, pm, pts), self.to_voxel(r, pr, pts)
        ok = self.inside(m, va) & self.inside(r, vb)
        sa = self.sample(self.seg[m], va, "nearest")
        sb = self.sample(self.seg[r], vb, "nearest")
        dice, cnt = 0.0, 0.0
        for lab in LV_LABELS:
            a, b = (sa == lab) & ok[:, None], (sb == lab) & ok[:, None]
            tot = (a.sum(-1) + b.sum(-1)).float()
            has = tot > 0
            dice = dice + torch.where(has, 2 * (a & b).sum(-1).float() / tot.clamp_min(1), torch.zeros_like(tot))
            cnt = cnt + has.float()
        return (dice / cnt.clamp_min(1)).mean(1), (cnt > 0).float().mean(1)

    @torch.no_grad()
    def landscape(self, m, refs, spec):
        """NCC score and LV Dice of every candidate of the coarse-to-fine schedule (survivors chosen by NCC)."""
        frame = self.frames[m]
        th0, t0 = self.theta[m].clone(), self.t[m].clone()
        rows, survivors = [], None
        for li, level in enumerate(spec.levels()):
            offs = level.offsets(dev) if survivors is None else torch.unique(spec.clip(
                (survivors[:, None, :] + level.offsets(dev)[None]).reshape(-1, 6)), dim=0)
            sc, dc, hc = [], [], []
            for o in offs.split(self.batch):
                th, tt = th0 + o[:, 3:] @ frame, t0 + o[:, :3] @ frame
                sc.append(self.score(m, refs, [], th, tt))
                pm = self.posed(m, th, tt)
                ds = [self.line_dice(m, pm, r, self.current(r, th.shape[0])) for r, _ in refs]
                dc.append(sum(x for x, _ in ds) / len(ds))
                hc.append(sum(h for _, h in ds) / len(ds))
            s, dsc, hit = torch.cat(sc), torch.cat(dc), torch.cat(hc)
            rows.append((offs.cpu(), s.cpu(), dsc.cpu(), hit.cpu()))
            survivors = self.distinct_top(offs, torch.nan_to_num(s, nan=-1e9), args.top_k, level)
        return rows


final_sampler_band = None
lines_ = []
for a in sax_planes:
    for b in lax_planes:
        line = mg.intersection_line(a, b, args, dev)
        if line is not None:
            lines_.append((idx[a.key], idx[b.key], line))
for a, b in [(lax_planes[i], lax_planes[j]) for i in range(len(lax_planes)) for j in range(i + 1, len(lax_planes))]:
    line = mg.intersection_line(a, b, args, dev)
    if line is not None:
        lines_.append((idx[a.key], idx[b.key], line))
smp = mg.LineSampler(planes, lines_, args.n_band, dev, args.psf_model, args.psf_tbw, 1.0)
pair_weight = {frozenset((ia, ib)): float(n) for ia, ib, n in zip(smp.IA.tolist(), smp.IB.tolist(), smp.npts.tolist())}
gs = DiagSearcher(planes, pair_weight, (smp.u_img, smp.band_weights()), args, dev)
gs.theta.copy_(theta_f)
gs.t.copy_(t_f)
views = [idx[p.key] for p in lax_planes]
spec = bf.GridSpec(args.t_range_mm, args.t_step_mm, args.t_range_mm, args.rot_range_deg, args.rot_step_deg,
                   args.coarse_t_step_mm, args.coarse_rot_step_deg, 0.0)

print("\nC. NCC score vs LV segmentation Dice along the 3 LAX lines (other planes at the final poses)")
out = {}
for z in (8, 9, 10):
    m = idx[f"SAX:{z}"]
    refs = [(r, pair_weight[frozenset((m, r))]) for r in views if frozenset((m, r)) in pair_weight]
    if not refs:
        print(f"  slice {z}: no line long enough")
        continue
    # search from the acquired pose of this slice (as the search did)
    gs.theta[m] = 0
    gs.t[m] = 0
    rows = gs.landscape(m, refs, spec)
    offs = torch.cat([r[0] for r in rows])
    s = torch.cat([r[1] for r in rows])
    dsc = torch.cat([r[2] for r in rows])
    hit = torch.cat([r[3] for r in rows])
    out[z] = (offs, s, dsc, hit)
    zero = int((offs.abs().sum(1) == 0).nonzero()[0])
    kb = int(torch.nan_to_num(s, nan=-9).argmax())
    kd = int(torch.nan_to_num(dsc, nan=-9).argmax())
    fmt = lambda k: (f"NCC {s[k]:.3f}  Dice {dsc[k]:.3f}  labels hit on {hit[k]:.2f} of line-frames  "  # noqa: E731
                     f"offset {np.round(offs[k].numpy(), 1).tolist()}")
    print(f"  slice {z}: acquired : {fmt(zero)}")
    print(f"            NCC-best : {fmt(kb)}")
    print(f"            Dice-best: {fmt(kd)}")
    good = dsc >= dsc[kd] - 0.05
    print(f"            best NCC among poses within 0.05 of the best Dice: {s[good].max():.3f}; "
          f"candidates with NCC >= NCC-best - 0.02: {(s >= s[kb] - 0.02).sum().item()}, "
          f"their Dice range {dsc[s >= s[kb] - 0.02].min():.2f}-{dsc[s >= s[kb] - 0.02].max():.2f}")
    gs.theta[m] = theta_f[m]
    gs.t[m] = t_f[m]
torch.save(out, Path(__file__).with_name("bf_apex_diag.pt"))
