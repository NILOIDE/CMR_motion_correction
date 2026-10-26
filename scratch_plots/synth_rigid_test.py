"""Perturb plane headers with known rigid transforms and measure how well each stage recovers the pose."""
import sys
import time
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, r"C:\Users\Nil\Documents\git\UKBB_implicit_baselines")
import motion_misalignment as mm

kw = dict(a.split("=") for a in sys.argv[1:])
args = mm.argparse.Namespace(
    instance="Instance_2", n_frames=5, mask_dilation_mm=15.0, lax_thickness=None, sax_thickness=8.0,
    search_mm=15.0, coarse_step_mm=1.0, fine_step_mm=0.25, rounds=int(kw.get("rounds", 2)), min_line_mm=20.0, line_step_mm=1.0,
    line_extent_mm=250.0, n_band=5, metric="ncc",
    refine_steps=int(kw.get("steps", 200)), refine_lr=float(kw.get("lr", 0.05)),
    refine_rot_lr=float(kw.get("rot_lr", 0.01)), refine_dof=int(kw.get("dof", 6)),
    refine_gauge=float(kw.get("gauge", 1e-2)), refine_prior=float(kw.get("prior", 0.0)), seg_labels="1,2,3",
)
ROT_DEG = float(kw.get("rot", 3.0))
TRANS_MM = float(kw.get("trans", 5.0))
SUBJECT = kw.get("subject", "1000109")
dev = torch.device("cuda")


def rotmat(theta_deg):
    th = torch.tensor(theta_deg * np.pi / 180.0, dtype=torch.float64)
    return torch.linalg.matrix_exp(mm.skew(th[None]))[0].numpy()


def perturb(plane, Rm, tv):
    """Change the header as if the slice were planned at a different pose; content unchanged."""
    C = plane.center.copy()
    plane.origin = Rm @ (plane.origin - C) + C + tv
    plane.center = C + tv
    plane.u_hat, plane.v_hat, plane.normal = Rm @ plane.u_hat, Rm @ plane.v_hat, Rm @ plane.normal
    inv_rot = plane.inv_rot.cpu().numpy().astype(np.float64) @ Rm.T
    plane.inv_rot = torch.tensor(inv_rot, dtype=torch.float32, device=plane.inv_rot.device)
    plane.inv_trans = torch.tensor(-inv_rot @ plane.origin, dtype=torch.float32, device=plane.inv_rot.device)


def final_pose(plane, theta_deg, t):
    Rm = rotmat(np.asarray(theta_deg, dtype=np.float64))
    return plane.center + t, Rm @ plane.normal, Rm @ plane.u_hat


def run(perturbations):
    orig_make = mm.make_planes
    captured = {}

    def patched(image, a, affine, thickness, mask_arr, seg_arr, device):
        planes = orig_make(image, a, affine, thickness, mask_arr, seg_arr, device)
        for p in planes:
            if p.key in perturbations:
                perturb(p, *perturbations[p.key])
            captured[p.key] = p
        return planes

    mm.make_planes = patched
    data = mm.load_subject(Path(rf"D:\ukbb\{SUBJECT}"), args)
    summary, details = mm.process_subject(data, args, dev)
    mm.make_planes = orig_make
    poses = {}
    for d in details:
        key = f"{d['image']}:{d['slice']}"
        p = captured[key]
        if d["n_pairs"] == 0:
            continue
        grid = final_pose(p, np.zeros(3), p.shift)
        theta = np.zeros(3)
        if np.isfinite(d["rot_inplane_ref_deg"]):
            tv = d["dx_ref_mm"] * p.u_hat + d["dy_ref_mm"] * p.v_hat + d["dz_ref_mm"] * p.normal
            # rotation vector is not recoverable from the magnitude alone; recompute from refine output
            theta = REFINED_THETA[key]
        else:
            tv = p.shift
        poses[key] = dict(grid=grid, ref=final_pose(p, theta, tv), raw=final_pose(p, np.zeros(3), np.zeros(3)))
    return poses, summary


# capture per-plane rotation vectors from refine_rigid
REFINED_THETA = {}
_orig_refine = mm.refine_rigid


def _refine_capture(sampler, a):
    out = _orig_refine(sampler, a)
    for p, th in zip(sampler.planes, out["theta_deg"].cpu().numpy()):
        REFINED_THETA[p.key] = th
    return out


mm.refine_rigid = _refine_capture

t0 = time.time()
base, base_summary = run({})
rng = np.random.default_rng(0)
targets = ["SAX:3", "SAX:5", "SAX:7", "LAX_4Ch:0"]
pert = {}
for k in targets:
    th = rng.normal(size=3)
    th = th / np.linalg.norm(th) * ROT_DEG
    tv = rng.normal(size=3)
    tv = tv / np.linalg.norm(tv) * TRANS_MM
    pert[k] = (rotmat(th), tv)
REFINED_THETA.clear()
test, test_summary = run(pert)
print(f"config {kw}  ({time.time() - t0:.1f} s for 2 runs)  perturbed {targets}: |rot|={ROT_DEG} deg, |trans|={TRANS_MM} mm")


def ang(a, b):
    return float(np.degrees(np.arccos(np.clip(a @ b, -1, 1))))


def kabsch(P, Q):
    """Rigid (R, t) minimizing |R P + t - Q| over point rows."""
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, qc - R @ pc


def report(stage):
    # remove the global rigid mode: fit test -> base on the unperturbed planes (centres + normal/u tips)
    keys = [k for k in base if k in test]
    ref = [k for k in keys if k not in pert]
    P = np.concatenate([[test[k][stage][0], test[k][stage][0] + 50 * test[k][stage][1], test[k][stage][0] + 50 * test[k][stage][2]] for k in ref])
    Q = np.concatenate([[base[k][stage][0], base[k][stage][0] + 50 * base[k][stage][1], base[k][stage][0] + 50 * base[k][stage][2]] for k in ref])
    Rg, tg = kabsch(P, Q)
    rows = []
    for k in keys:
        c, n, u = test[k][stage]
        dt = np.linalg.norm(Rg @ c + tg - base[k][stage][0])
        dn = ang(Rg @ n, base[k][stage][1])
        du = ang(Rg @ u, base[k][stage][2])
        rows.append((k, dt, dn, du))
    pert_rows = [r for r in rows if r[0] in pert]
    other_rows = [r for r in rows if r[0] not in pert]
    def fmt(rs):
        return "trans %.2f/%.2f mm  tilt %.2f/%.2f deg  in-plane rot %.2f/%.2f deg (mean/max)" % (
            np.mean([r[1] for r in rs]), np.max([r[1] for r in rs]), np.mean([r[2] for r in rs]), np.max([r[2] for r in rs]),
            np.mean([r[3] for r in rs]), np.max([r[3] for r in rs]))
    print(f"[{stage:4s}] perturbed planes residual: {fmt(pert_rows)}")
    print(f"[{stage:4s}] other planes residual:     {fmt(other_rows)}")
    for r in pert_rows:
        print("        %-10s trans %.2f mm  tilt %.2f deg  in-plane %.2f deg" % r)


report("raw")
report("grid")
report("ref")
print("NCC: baseline best=%.3f ref=%.3f | perturbed best=%.3f ref=%.3f" % (
    base_summary["ncc_best_mean"], base_summary["ncc_ref_mean"], test_summary["ncc_best_mean"], test_summary["ncc_ref_mean"]))
