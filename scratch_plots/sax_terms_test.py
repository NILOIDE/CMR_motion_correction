"""Checks of the new neighbouring-SAX terms on subject 1000071: gap geometry, barrier, tilt hinge, dissimilarity and
their gradients (autograd vs central differences), and that a mismatched crossing is pushed away."""
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


def setup(falloff="gaussian"):
    args = SimpleNamespace(n_frames=3, mask_dilation_mm=20.0, lax_thickness=6.0, sax_thickness=8.0, line_extent_mm=150.0,
                           line_step_mm=1.0, min_line_mm=40.0, outside_mask_weight=0.0, metric="ncc", n_band=8,
                           psf_model="gaussian", psf_tbw=2.5, psf_scale_start=1.0, sax_sax_weight=1.0, sax_sax_falloff=falloff,
                           sax_sax_radius_mm=40.0, sax_gap_reg=1.0, sax_gap_margin_mm=4.0, sax_gap_radius_mm=50.0, dof=6,
                           mi_bins=24, mi_sigma=1.0, logging=True)
    data = mg.load_subject(Path(r"C:\Users\Nil\Documents\data\ukbb_slices\1000071"), args)
    sax, lax = mg.build_planes(data, args, dev)
    planes = sax + lax
    idx = {p.key: i for i, p in enumerate(planes)}
    lines = [(idx[a.key], idx[b.key], ln) for a, b in [(s, l) for s in sax for l in lax] + list(combinations(lax, 2))
             if (ln := mg.intersection_line(a, b, args, dev)) is not None]
    smp = mg.LineSampler(planes, lines, 8, dev, "gaussian")
    mg.attach_sax_pairs(smp, planes, sax, lax, args, dev)
    return smp, planes, idx


smp, planes, idx = setup()
sp = smp.sax_pairs
N = len(planes)
zeros = torch.zeros(N, 3, device=dev)
g0 = sp.gaps(zeros, zeros)
print(f"{sp.P} pairs, {sp.disk.shape[0]} disk points per slice; acquired gap min/max {g0['gap'].min():.2f}/{g0['gap'].max():.2f} mm, "
      f"rel tilt max {g0['rel_tilt'].max():.4f} deg; barrier {sp.barrier(zeros, zeros).item():.2e} mm^2")

# translate slice 6 by -8 mm towards slice 5 along the normal: gap 5-6 -> 2 mm, gap 6-7 -> 18 mm
i5, i6 = idx["SAX:5"], idx["SAX:6"]
k56 = [k for k, (a, b) in enumerate(sp.pairs) if (a, b) == (i5, i6)][0]
k67 = [k for k, (a, b) in enumerate(sp.pairs) if a == i6][0]
step_dir = float(sp.stack_sign[k56].item())
t = torch.zeros(N, 3, device=dev)
t[i6] = torch.tensor(-8.0 * step_dir * planes[i6].normal, dtype=torch.float32, device=dev)
g = sp.gaps(zeros, t)
print(f"slice 6 moved 8 mm towards 5: gap(5,6) {g['gap'][k56].min():.2f}..{g['gap'][k56].max():.2f}, "
      f"gap(6,7) {g['gap'][k67].min():.2f}..{g['gap'][k67].max():.2f} mm; barrier {sp.barrier(zeros, t).item():.4f}")

# tilt 5 and 6 by +-12 deg about their row axis -> they cross near the axis
th = torch.zeros(N, 3, device=dev)
th[i5] = torch.tensor(12.0 * planes[i5].u_hat, dtype=torch.float32, device=dev)
th[i6] = torch.tensor(-12.0 * planes[i6].u_hat, dtype=torch.float32, device=dev)
g = sp.gaps(th, zeros)
c, n, dist = sp.scores(th, zeros)
print(f"5,6 tilted +-12 deg: rel tilt {g['rel_tilt'][k56].item():.2f} deg (expected 24), gap(5,6) min {g['gap'][k56].min():.2f} mm, "
      f"line weight {n[k56].item():.1f}, NCC {c[k56].item():.3f}, axis dist {dist[k56].item():.1f} mm")

# tilt hinge: angle and penalty
C0, N0 = smp.C, smp.Nrm
sax_i = sorted((p.slice_idx, i) for i, p in enumerate(planes) if p.image == "SAX")
nb_a = torch.tensor([i for (_, i), _ in zip(sax_i, sax_i[1:])], device=dev)
nb_b = torch.tensor([j for _, (_, j) in zip(sax_i, sax_i[1:])], device=dev)
step0 = ((C0[nb_b] - C0[nb_a]) * N0[nb_a]).sum(1)
terms = mg.stack_terms(th, zeros, C0, N0, nb_a, nb_b, step0, {"tilt_hinge"}, 5.0)
print(f"tilt hinge at +-12 deg (5 deg free): {terms['tilt_hinge'].item():.2f} deg^2 (expected 2 pairs x (24-5)^2 + 1 pair x (12-5)^2 "
      f"over {len(nb_a)} pairs = {(2 * 19 ** 2 + 49) / len(nb_a):.2f}); at 0: "
      f"{mg.stack_terms(zeros, zeros, C0, N0, nb_a, nb_b, step0, {'tilt_hinge'}, 5.0)['tilt_hinge'].item():.2f}")

# gradients: autograd vs central differences, at the tilted pose, for barrier, dissimilarity and tilt hinge
def check(name, f, base_th):
    thg = base_th.clone().requires_grad_(True)
    f(thg).backward()
    worst = 0.0
    for j in range(3):
        h = 0.02
        tp, tm = base_th.clone(), base_th.clone()
        tp[i5, j] += h
        tm[i5, j] -= h
        with torch.no_grad():
            fd = (f(tp) - f(tm)).item() / (2 * h)
        worst = max(worst, abs(fd - thg.grad[i5, j].item()) / max(abs(fd), 1e-3))
    print(f"  {name}: grad finite {bool(torch.isfinite(thg.grad).all())}, worst relative autograd/finite-diff mismatch {worst:.3f}")


check("barrier", lambda x: sp.barrier(x, zeros), th)
check("dissimilarity", lambda x: sp.dissimilarity(x, zeros), th)
check("tilt hinge", lambda x: mg.stack_terms(x, zeros, C0, N0, nb_a, nb_b, step0, {"tilt_hinge"}, 5.0)["tilt_hinge"], th)
check("all at the acquired pose", lambda x: sp.barrier(x, zeros) + sp.dissimilarity(x, zeros)
      + mg.stack_terms(x, zeros, C0, N0, nb_a, nb_b, step0, {"tilt_hinge"}, 5.0)["tilt_hinge"], zeros)

# a mismatched crossing is pushed out: shift slice 6 in-plane by 15 mm (images no longer agree along the line), then a
# few gradient steps on the dissimilarity alone (Gaussian falloff) should move the line away from the axis
th2 = th.clone()
t2 = torch.zeros(N, 3, device=dev)
t2[i6] = torch.tensor(15.0 * planes[i6].v_hat, dtype=torch.float32, device=dev)
x = th2.clone().requires_grad_(True)
opt = torch.optim.SGD([x], lr=20.0)
d0 = sp.scores(x, t2)[2][k56].item()
c0, n0, _ = sp.scores(x.detach(), t2)
for _ in range(60):
    opt.zero_grad()
    sp.dissimilarity(x, t2).backward()
    x.grad[[i for i in range(N) if i not in (i5, i6)]] = 0
    opt.step()
c1, n1, d1 = sp.scores(x.detach(), t2)
print(f"mismatched crossing, 60 SGD steps on the dissimilarity: axis distance {d0:.1f} -> {d1[k56].item():.1f} mm, "
      f"line weight {n0[k56].item():.1f} -> {n1[k56].item():.1f}, cost {(n0[k56] * (1 - c0[k56])).item():.2f} -> "
      f"{(n1[k56] * (1 - c1[k56])).item():.2f}, rel tilt {sp.gaps(x.detach(), t2)['rel_tilt'][k56].item():.1f} deg")
