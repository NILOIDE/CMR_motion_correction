"""Checks of motion_gradient.SaxNeighbourLines on subject 1000071: geometry of the crossing line, gating, gradients."""
import math
import sys
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, r"C:\Users\Nil\Documents\git\UKBB_implicit_baselines")
import motion_gradient as mg  # noqa: E402

dev = torch.device("cuda")
args = SimpleNamespace(n_frames=3, mask_dilation_mm=20.0, lax_thickness=6.0, sax_thickness=8.0, line_extent_mm=150.0,
                       line_step_mm=1.0, min_line_mm=40.0, outside_mask_weight=0.0, metric="ncc", n_band=8, psf_model="gaussian",
                       psf_tbw=2.5, psf_scale_start=1.0, sax_sax_weight=1.0, sax_sax_radius_mm=40.0, dof=6, mi_bins=24, mi_sigma=1.0)
data = mg.load_subject(Path(r"C:\Users\Nil\Documents\data\ukbb_slices\1000071"), args)
sax, lax = mg.build_planes(data, args, dev)
planes = sax + lax
idx = {p.key: i for i, p in enumerate(planes)}
lines = [(idx[a.key], idx[b.key], ln) for a, b in [(s, l) for s in sax for l in lax] + list(combinations(lax, 2))
         if (ln := mg.intersection_line(a, b, args, dev)) is not None]
smp = mg.LineSampler(planes, lines, 8, dev, "gaussian")
mg.attach_sax_pairs(smp, planes, sax, lax, args, dev)
sp = smp.sax_pairs
print(f"{sp.P} neighbouring pairs; long axis point {np.round(sp.L0.cpu().numpy(), 1)} dir {np.round(sp.ell.cpu().numpy(), 3)}")
N = len(planes)

# 1. identity pose: parallel slices -> nothing counts, no NaN anywhere (values or gradients)
th = torch.zeros(N, 3, device=dev, requires_grad=True)
t = torch.zeros(N, 3, device=dev, requires_grad=True)
c, n, dist = sp.scores(th, t)
obj = mg.similarity_objective(smp, th, t, args)
obj.backward()
print(f"identity: active pairs {int((n > 0).sum())}, scores {c.abs().max().item():.3g}, objective {obj.item():.4f}, "
      f"grad finite: {bool(torch.isfinite(th.grad).all() and torch.isfinite(t.grad).all())}")

# 2. tilt slices 5 and 6 against each other by +-10 deg about their row axis -> they cross 10 mm / (2 tan 10 deg) from
#    the centres; the crossing line must lie in both posed planes and the axis distance must match an independent one
i5, i6 = idx["SAX:5"], idx["SAX:6"]
p5, p6 = planes[i5], planes[i6]
th = torch.zeros(N, 3, device=dev)
th[i5] = torch.tensor(10.0 * p5.u_hat, dtype=torch.float32, device=dev)
th[i6] = torch.tensor(-10.0 * p6.u_hat, dtype=torch.float32, device=dev)
t = torch.zeros(N, 3, device=dev)
with torch.no_grad():
    a, b, w, active, dist = sp.profiles(th, t)
k = [j for j, (ia, ib) in enumerate(sp.pairs) if (ia, ib) == (i5, i6)][0]
R = mg.rotation_matrices(th.cpu().numpy())
n5, n6 = R[i5] @ p5.normal, R[i6] @ p6.normal
o5, o6 = R[i5] @ (p5.origin - p5.center) + p5.center, R[i6] @ (p6.origin - p6.center) + p6.center
d = np.cross(n5, n6); d /= np.linalg.norm(d)
x0 = sp.T.numel() // 2
pts = (torch.zeros(1) if False else None)
# recompute the sample points as profiles() does, to test them against the planes
with torch.no_grad():
    Rt = mg.rodrigues(th * (math.pi / 180.0))
    Cp = smp.C + t
    Np = (Rt @ smp.Nrm[..., None])[..., 0]
print(f"pair (5,6): active {bool(active[k])}, counting points {int(w[k].sum())}, axis distance {dist[k].item():.2f} mm, "
      f"others active: {int(active.sum()) - int(active[k])}")
# independent axis distance: closest approach of the crossing line and the long axis (numpy, float64)
M = np.stack([n5, n6, d]); rhs = np.array([n5 @ o5, n6 @ o6, d @ sp.L0.cpu().numpy().astype(np.float64)])
p0 = np.linalg.solve(M, rhs)
L0, ell = sp.L0.cpu().numpy().astype(np.float64), sp.ell.cpu().numpy().astype(np.float64)
w0 = p0 - L0; bb = d @ ell; dd, ee = d @ w0, ell @ w0
s_ = (bb * ee - dd) / (1 - bb * bb); tau = (ee - bb * dd) / (1 - bb * bb)
print(f"  independent axis distance {np.linalg.norm(w0 + s_ * d - tau * ell):.2f} mm; "
      f"expected crossing offset from the slice centres {10 / (2 * math.tan(math.radians(10))):.1f} mm, "
      f"centre-to-line distance {np.linalg.norm(np.cross(p5.center - (p0 + s_ * d), d)):.1f} mm")
# counting points lie within the radius of the axis and inside both images; off-count points are outside one of them
pts_k = (p0 + s_ * d)[None] + sp.T.cpu().numpy()[:, None] * d[None]
rel = pts_k - L0; perp = rel - (rel @ ell)[:, None] * ell; r = np.linalg.norm(perp, axis=1)
wk = w[k].cpu().numpy() > 0
print(f"  points within radius: {int((r <= 40).sum())}, counting: {int(wk.sum())}, "
      f"max axis distance of a counting point {r[wk].max():.1f} mm, on plane 5/6: "
      f"{np.abs((pts_k[wk] - o5) @ n5).max():.1e} / {np.abs((pts_k[wk] - o6) @ n6).max():.1e} mm")

# 3. gradient of the SA-SA objective wrt slice 5's rotation: autograd vs central differences
def f(th_):
    c, n, _ = sp.scores(th_, t)
    return (c * n).sum() / n.sum().clamp_min(1)
thg = th.clone().requires_grad_(True)
f(thg).backward()
for j, name in enumerate("xyz"):
    h = 0.05
    tp, tm = th.clone(), th.clone()
    tp[i5, j] += h; tm[i5, j] -= h
    with torch.no_grad():
        fd = (f(tp) - f(tm)).item() / (2 * h)
    print(f"  d/dtheta_{name}(slice 5): autograd {thg.grad[i5, j].item():+.5f}  finite diff {fd:+.5f}")
