"""Checks of --rotation-pivot on subject 1000071 (image similarity only, no segmentation metrics):
1. the pivots lie in their planes and SAX pivots on the long axis; distance of the image centres from the heart
2. consistency: for random poses, the sampler's world->voxel map equals the inverse of corrected_affine (saved NIfTIs)
3. entanglement: Hessian of the NCC objective w.r.t. one plane's (rotation, translation) at the acquired pose;
   the normalized rotation-translation coupling |H_rt| / sqrt(H_rr H_tt) for both pivot choices
"""
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
SUBJ = sys.argv[1] if len(sys.argv) > 1 else "1000071"


def setup(pivot):
    args = SimpleNamespace(n_frames=10, mask_dilation_mm=20.0, lax_thickness=6.0, sax_thickness=8.0, line_extent_mm=150.0,
                           line_step_mm=1.0, min_line_mm=40.0, outside_mask_weight=0.0, metric="ncc", n_band=8,
                           psf_model="gaussian", psf_tbw=2.5, rotation_pivot=pivot, mi_bins=24, mi_sigma=1.0)
    data = mg.load_subject(Path(r"C:\Users\Nil\Documents\data\ukbb_slices") / SUBJ, args)
    sax, lax = mg.build_planes(data, args, dev)
    planes = sax + lax
    idx = {p.key: i for i, p in enumerate(planes)}
    lines = [(idx[a.key], idx[b.key], ln) for a, b in [(s, l) for s in sax for l in lax] + list(combinations(lax, 2))
             if (ln := mg.intersection_line(a, b, args, dev)) is not None]
    smp = mg.LineSampler(planes, lines, args.n_band, dev, args.psf_model, args.psf_tbw)
    return args, data, sax, lax, planes, smp


args, data, sax, lax, planes, smp = setup("long-axis")
L0, d = mg.long_axis_line(lax)
d = d / np.linalg.norm(d)
in_plane = max(abs(float((p.pivot - p.center) @ p.normal)) for p in planes)
on_axis = max(float(np.linalg.norm(np.cross(p.pivot - L0, d))) for p in sax)
offs = [float(np.linalg.norm(p.pivot - p.center)) for p in planes]
print(f"1. pivots: max distance from their plane {in_plane:.1e} mm, SAX pivots from the axis {on_axis:.1e} mm; "
      f"image centre -> pivot distance {min(offs):.0f}..{max(offs):.0f} mm (median {np.median(offs):.0f})")

# 2. consistency with the saved affine
torch.manual_seed(0)
N = len(planes)
theta = torch.randn(N, 3, device=dev) * 4
t = torch.randn(N, 3, device=dev) * 5
R = mg.rodrigues(theta * (math.pi / 180))
Cp = smp.C + t
worst = 0.0
for i, p in enumerate(planes):
    A = torch.einsum("ij,kj->ik", smp.inv_rot[i], R[i])
    b = smp.inv_rot[i] @ smp.C[i] + smp.inv_trans[i] - A @ Cp[i]
    Aff = mg.corrected_affine(mg.base_affine(data, p), p, mg.rotation_matrices(theta[i:i + 1].cpu().numpy())[0],
                              t[i].cpu().numpy().astype(np.float64))
    x = np.random.default_rng(i).normal(0, 80, (20, 3)) + p.center
    v_s = (torch.tensor(x, dtype=torch.float32, device=dev) @ A.T + b).cpu().numpy()
    v_a = (np.c_[x, np.ones(len(x))] @ np.linalg.inv(Aff).T)[:, :3]
    worst = max(worst, float(np.abs(v_s - v_a).max()))
print(f"2. sampler map vs corrected_affine (saved NIfTIs): max voxel difference {worst:.1e}")


# 3'. geometric rotation-translation coupling. A pose moves a content point x of plane i to R (x - c) + c + t, so near
# the identity dx = w x (x - c) + t. Over the plane's in-mask line points (the points that drive the loss), the
# canonical correlation between the rotation and translation columns of this Jacobian measures how far a rotation can
# be imitated by a translation (Gauss-Newton view): 0 when the pivot c is at the points' centroid, -> 1 far away.
def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def geometric_coupling(pivot):
    args, data, sax, lax, planes, smp = setup(pivot)
    pts = {i: [] for i in range(len(planes))}
    T, W = smp.T.cpu().numpy(), smp.W.cpu().numpy()
    for m, (ia, ib) in enumerate(zip(smp.IA.tolist(), smp.IB.tolist())):
        pass
    lines = smp._lines if hasattr(smp, "_lines") else None
    out = {"SAX": [], "LAX": []}
    for i, p in enumerate(planes):
        ms = smp.plane_lines[i].tolist()
        if not ms:
            continue
        xs = np.concatenate([LINES[m][0][None] + T[m, W[m] > 0][:, None] * LINES[m][1][None] for m in ms])
        c = p.rot_centre
        J = np.concatenate([np.concatenate([-skew(x - c), np.eye(3)], 1) for x in xs])      # (3K, 6): d x / d (w, t)
        G = J.T @ J
        def inv_sqrt(M):
            w, V = np.linalg.eigh(M)
            return V @ np.diag(np.clip(w, 1e-12, None) ** -0.5) @ V.T
        rho = np.linalg.svd(inv_sqrt(G[:3, :3]) @ G[:3, 3:] @ inv_sqrt(G[3:, 3:]), compute_uv=False)
        out["SAX" if p.image == "SAX" else "LAX"].append(float(rho[0]))
    return {k: (np.median(v), np.max(v)) for k, v in out.items() if v}


_, _, sax_, lax_, planes_, smp_ = setup("image-centre")
idx_ = {p.key: i for i, p in enumerate(planes_)}
LINES = [(ln.x0, ln.d) for a, b in [(s, l) for s in sax_ for l in lax_] + list(combinations(lax_, 2))
         if (ln := mg.intersection_line(a, b, setup.__defaults__ or SimpleNamespace(line_extent_mm=150.0, line_step_mm=1.0,
                                                                                   min_line_mm=40.0, outside_mask_weight=0.0),
                                        dev)) is not None]
for pivot in ("image-centre", "long-axis"):
    c = geometric_coupling(pivot)
    print(f"3. {pivot:12s}: geometric rotation-translation coupling (0 = independent, 1 = interchangeable): "
          + ", ".join(f"{k} median {m:.2f} max {x:.2f}" for k, (m, x) in c.items()))
sys.exit(0)


# 3. rotation-translation coupling of the NCC objective, per plane, at the acquired pose
def coupling(pivot):
    args, data, sax, lax, planes, smp = setup(pivot)
    N = len(planes)

    def f(x):                                     # x = (N, 6): theta (deg), t (mm)
        return (smp.similarity_lines(x[:, :3], x[:, 3:], args) * smp.npts).sum() / smp.npts.sum()

    out = {"SAX": [], "LAX": []}
    for i, p in enumerate(planes):
        if not len(smp.plane_lines[i]):
            continue
        def g(v, i=i):
            x = torch.zeros(N, 6, device=dev)
            x = x.index_put((torch.tensor([i], device=dev),), v[None])
            return f(x)
        def grad(v):                              # grid_sample has no double backward: Hessian = d(gradient) by FD
            v = v.clone().requires_grad_(True)
            g(v).backward()
            return v.grad.detach()
        h = 0.25                                  # deg / mm
        H = np.zeros((6, 6))
        for j in range(6):
            e = torch.zeros(6, device=dev)
            e[j] = h
            H[:, j] = ((grad(e) - grad(-e)) / (2 * h)).cpu().numpy()
        H = 0.5 * (H + H.T)
        Hrr, Htt, Hrt = H[:3, :3], H[3:, 3:], H[:3, 3:]
        # coupling in the whitened space: largest singular value of Hrr^-1/2 Hrt Htt^-1/2 (0 = decoupled, 1 = degenerate)
        def inv_sqrt(M):
            w, V = np.linalg.eigh(-M)                    # the objective is maximized: -H is ~positive definite
            w = np.clip(w, 1e-9, None)
            return V @ np.diag(w ** -0.5) @ V.T
        C = inv_sqrt(Hrr) @ (-Hrt) @ inv_sqrt(Htt)
        out["SAX" if p.image == "SAX" else "LAX"].append(float(np.linalg.svd(C, compute_uv=False)[0]))
    return {k: (np.median(v), np.max(v)) for k, v in out.items() if v}


for pivot in ("image-centre", "long-axis"):
    c = coupling(pivot)
    print(f"3. {pivot:12s}: rotation-translation coupling (0 = independent, 1 = fully entangled): "
          + ", ".join(f"{k} median {m:.2f} max {x:.2f}" for k, (m, x) in c.items()))
