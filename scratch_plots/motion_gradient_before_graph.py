"""
Gradient-based inter-slice motion estimation for UKBB cine MRI (SAX stack + LAX views).

For every pair of non-parallel planes (each SAX slice vs each LAX view, and LAX vs LAX) the 3D
intersection line is computed from the NIfTI affines. Intensity profiles are sampled along the
line from both images; each profile is averaged across a band whose width equals the *other*
image's slice thickness, approximating that image's slice-selection point-spread function.
Profiles are compared with normalized cross-correlation (NCC), restricted to a dilated heart
mask derived from the LAX nnU-Net segmentations and averaged over several cardiac phases.

A 6-DOF rigid pose per plane (rotation vector + 3D translation about the plane centre) is
optimized jointly for all planes with Adam, starting from the acquired geometry. Intersection
lines are recomputed from the current poses every step, so through-plane shifts and rotations
are observable. A weak gauge penalty on the mean pose removes the global rigid-body freedom.
Optionally (--blur-sigma) the optimization is coarse-to-fine: the images are blurred in-plane at
the start and the blur decays linearly to zero over the first --blur-frac of the steps; the best
pose is only tracked on the remaining, unblurred steps. Optionally (--pair-keep < 1) each step
drops a random subset of the intersection lines from the objective, which makes the optimization
stochastic; the best pose is then tracked on the full objective evaluated every --eval-every steps.

Segmentation consistency (Dice between the two views' labels along the same lines) is reported
before and after alignment as an independent check; it is not optimized.

Outputs go to <out-root>/<run-name>/ (run name built from the optimization settings):
  overall_metrics/
    motion_summary.csv        one row per subject (append-only, latest row per subject wins)
    motion_summary_slices.csv one row per SAX slice / LAX view
    motion_summary_pairs.csv  one row per intersection-line pair (NCC, NMI, Dice, Hausdorff before/after)
    motion_*.png              dataset-level figures (written at the end of a run, or with --plot)
  <subject>/
    <instance>/               motion-corrected NIfTIs (--no-save-images to skip): the full cine as
                              uint16 with the corrected affine. The SAX stack is written one file per
                              slice (SAX_slice<zz>.nii.gz) because every slice gets its own pose; LAX
                              views keep their names. Segmentations are written alongside.
    metrics/
      summary.csv, slices.csv, pairs.csv  this subject's rows
      history.csv             pose snapshots every --history-every of the run (step 0 = acquired geometry, and the
                              saved "best" pose with is_best=1): per plane the full 3x4 affine (a00..a23, same
                              convention as the saved NIfTIs), the pose and NCC/NMI/Dice/Hausdorff
      history_summary.csv     the same snapshots, one row per step with the objective and subject-level metrics
      lines.csv               every intersection line entering the loss (acquired geometry): the two planes, the
                              anchor point x0 and unit direction d (world, mm) and the parameter ranges along the
                              line that are inside both images (t_img_*) and actually sampled (t_used_*)
      mask_contours.csv       outer contour of each LAX view's dilated heart mask, in voxel coordinates (i, j)
      motion_subject.png      motion histograms, per-pair similarity histograms, per-slice translations
      motion_pairs.png        per-pair NCC / NMI / Dice / Hausdorff, split SAX-LAX and LAX-LAX
                              (both figures: --no-subject-plots)
      mask_overlay.png        dilated heart mask and the line points entering the loss (--no-mask-plots)

dx_mm / dy_mm are translations along the image's own row (i) and column (j) axes, dz_mm along
its normal, that must be applied to the image content to bring it into agreement with the other
views (translations are expressed relative to the mean LAX pose). rot_inplane_deg is the
rotation about the slice normal (well constrained by three crossing lines); rot_tilt_deg the
rotation about in-plane axes (weakly constrained, treat with caution).
"""
import argparse
import csv
import math
import sys
import time
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations, islice
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

LAX_VIEWS = ("LAX_2Ch", "LAX_3Ch", "LAX_4Ch")
SEG_DIR = "nnUNet_segs"

SUMMARY_COLUMNS = [
    "subject_id", "status", "n_lax_views", "sax_n_slices", "sax_n_eval",
    "sax_shift_mean_mm", "sax_shift_rms_mm", "sax_shift_max_mm", "sax_shift_jitter_mm",
    "sax_dz_rms_mm", "sax_rot_inplane_rms_deg", "sax_tilt_rms_deg",
    "lax_shift_mean_mm", "lax_shift_max_mm",
    "ncc_zero_mean", "ncc_mean", "ncc_gain_mean",
    "mi_zero_mean", "mi_mean", "mi_gain_mean",
    "seg_dice_zero_mean", "seg_dice_mean",
    "hd_zero_mean_mm", "hd_mean_mm", "hd_zero_max_mm", "hd_max_mm",
    "n_line_pairs", "steps", "seconds", "timestamp",
]
DETAIL_COLUMNS = [
    "subject_id", "image", "slice", "n_pairs", "n_points",
    "dx_mm", "dy_mm", "dz_mm", "shift_inplane_mm", "shift_mm", "rot_inplane_deg", "rot_tilt_deg",
    "ncc_zero", "ncc", "mi_zero", "mi", "seg_dice_zero", "seg_dice", "hd_zero_mm", "hd_mm",
]
PAIR_COLUMNS = [
    "subject_id", "image_a", "slice_a", "image_b", "slice_b", "kind", "n_points",
    "ncc_zero", "ncc", "mi_zero", "mi", "seg_dice_zero", "seg_dice", "hd_zero_mm", "hd_mm",
]


@dataclass
class Plane:
    key: str
    image: str
    slice_idx: int
    img: torch.Tensor                # (F, 1, H, W)
    seg: torch.Tensor | None         # (F, 1, H, W) integer labels as float
    mask: torch.Tensor | None        # (1, 1, H, W) dilated heart mask
    inv_rot: torch.Tensor            # (3, 3) world -> voxel
    inv_trans: torch.Tensor          # (3,)
    origin: np.ndarray
    center: np.ndarray
    u_hat: np.ndarray
    v_hat: np.ndarray
    normal: np.ndarray
    thickness: float
    spacing: tuple = (1.0, 1.0)      # in-plane voxel size (mm) along image axes 0 and 1

    @property
    def H(self):
        return self.img.shape[2]

    @property
    def W(self):
        return self.img.shape[3]


@dataclass
class Line:
    t: torch.Tensor                  # (L,) signed distance along the line from its anchor point
    w: torch.Tensor                  # (L,) 1 where the point is valid
    n_points: float
    x0: np.ndarray | None = None     # anchor point (world, mm) at the acquired geometry
    d: np.ndarray | None = None      # unit direction (world)


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------

def phase_frames(n_frames_available: int, n_frames: int) -> list[int]:
    return sorted({int(math.floor(k / n_frames * n_frames_available)) for k in range(n_frames)})


def load_nifti(path: Path, dtype=np.float32) -> tuple[np.ndarray, np.ndarray, nib.Nifti1Header]:
    """Full 4D array (X, Y, Z, T), affine and header. Integer dtypes are rounded, not truncated."""
    img = nib.load(str(path))
    arr = np.asarray(img.dataobj, dtype=np.float32)
    if np.issubdtype(dtype, np.integer):
        arr = np.rint(arr).astype(dtype)                                   # scaled label files store 1 as 0.9999999
    if arr.ndim == 3:
        arr = arr[..., None]
    return arr, img.affine.astype(np.float64), img.header.copy()


def load_subject(subj_dir: Path, args) -> dict:
    out = {"subject_id": subj_dir.name, "status": "ok", "sax": None, "sax_seg": None, "lax": {}, "full": {}, "full_seg": {}}
    inst = subj_dir / args.instance
    sax_path = inst / "SAX.nii.gz"
    if not sax_path.exists():
        out["status"] = "missing_SAX"
        return out
    keep = not args.no_save_images

    arr, aff, header = load_nifti(sax_path)
    frames = phase_frames(arr.shape[3], args.n_frames)
    out["sax"] = (arr[..., frames], aff)
    if keep:
        out["full"]["SAX"] = (arr, aff, header)
    sax_seg_path = inst / SEG_DIR / "SAX_nnUNetSeg.nii.gz"
    if sax_seg_path.exists():
        seg, _, seg_header = load_nifti(sax_seg_path, dtype=np.int16)
        if seg.shape == arr.shape:
            out["sax_seg"] = seg[..., frames]
            if keep:
                out["full_seg"]["SAX"] = (seg, seg_header)

    for view in LAX_VIEWS:
        img_path = inst / f"{view}.nii.gz"
        seg_path = inst / SEG_DIR / f"{view}_nnUNetSeg.nii.gz"
        if not (img_path.exists() and seg_path.exists()):
            continue
        a, affine, header = load_nifti(img_path)
        seg, _, seg_header = load_nifti(seg_path, dtype=np.int16)
        if view == "LAX_2Ch":
            seg[seg == 3] = 4                                   # 2Ch has no class 3; its class 3 is class 4 elsewhere
        frames = phase_frames(a.shape[3], args.n_frames)
        zooms = header.get_zooms()
        seg_sub = seg[..., frames] if seg.shape == a.shape else None
        heart = (seg > 0).any(-1)[:, :, 0]
        if not heart.any():
            continue
        dist = distance_transform_edt(~heart, sampling=zooms[:2])
        mask = (dist <= args.mask_dilation_mm)[:, :, None]
        thickness = args.lax_thickness if args.lax_thickness else float(zooms[2])
        out["lax"][view] = dict(arr=a[..., frames], affine=affine, mask=mask, seg=seg_sub, thickness=thickness,
                                dist=dist.astype(np.float32))      # distance to the heart (mm), for mask annealing
        if keep:
            out["full"][view] = (a, affine, header)
            if seg_sub is not None:
                out["full_seg"][view] = (seg, seg_header)

    if not out["lax"]:
        out["status"] = "no_lax_views"
    return out


# --------------------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------------------

def make_planes(image: str, arr: np.ndarray, affine: np.ndarray, thickness: float,
                mask_arr: np.ndarray | None, seg_arr: np.ndarray | None, device) -> list[Plane]:
    X, Y, Z, _ = arr.shape
    planes = []
    for z in range(Z):
        A = affine.copy()
        A[:3, 3] = (affine @ np.array([0.0, 0.0, z, 1.0]))[:3]
        inv = torch.linalg.inv(torch.from_numpy(A)).numpy()
        u, v = A[:3, 0], A[:3, 1]
        n = np.cross(u, v)
        n /= np.linalg.norm(n)
        img = torch.from_numpy(np.ascontiguousarray(arr[:, :, z].transpose(2, 0, 1)))[:, None].to(device)
        mask = seg = None
        if mask_arr is not None:
            mask = torch.from_numpy(mask_arr[:, :, z].astype(np.float32))[None, None].to(device)
        if seg_arr is not None:
            seg = torch.from_numpy(np.ascontiguousarray(seg_arr[:, :, z].transpose(2, 0, 1)).astype(np.float32))[:, None].to(device)
        planes.append(Plane(
            key=f"{image}:{z}", image=image, slice_idx=z, img=img, seg=seg, mask=mask,
            inv_rot=torch.tensor(inv[:3, :3], dtype=torch.float32, device=device),
            inv_trans=torch.tensor(inv[:3, 3], dtype=torch.float32, device=device),
            origin=A[:3, 3].copy(),
            center=(A @ np.array([(X - 1) / 2, (Y - 1) / 2, 0.0, 1.0]))[:3],
            u_hat=u / np.linalg.norm(u), v_hat=v / np.linalg.norm(v), normal=n, thickness=thickness,
            spacing=(float(np.linalg.norm(u)), float(np.linalg.norm(v))),
        ))
    return planes


def voxel_grid(plane: Plane, pts: torch.Tensor) -> torch.Tensor:
    """World points (..., 3) -> grid_sample coordinates (..., 2) in [-1, 1]."""
    vox = pts @ plane.inv_rot.T + plane.inv_trans
    gx = 2 * vox[..., 1] / (plane.W - 1) - 1
    gy = 2 * vox[..., 0] / (plane.H - 1) - 1
    return torch.stack([gx, gy], -1)


def sample_image(img: torch.Tensor, grid: torch.Tensor, mode: str) -> torch.Tensor:
    """img (N, 1, H, W), grid (M, 2) -> (N, M)."""
    g = grid.view(1, 1, -1, 2).expand(img.shape[0], 1, -1, 2)
    return F.grid_sample(img, g, mode=mode, padding_mode="zeros", align_corners=True)[:, 0, 0, :]


def inside(plane: Plane, pts: torch.Tensor) -> torch.Tensor:
    vox = pts @ plane.inv_rot.T + plane.inv_trans
    return (vox[:, 0] >= 0) & (vox[:, 0] <= plane.H - 1) & (vox[:, 1] >= 0) & (vox[:, 1] <= plane.W - 1)


def intersection_line(a: Plane, b: Plane, args, device) -> Line | None:
    """Valid parameter range and mask weights of the a/b intersection line at the acquired geometry."""
    d = np.cross(a.normal, b.normal)
    nd = np.linalg.norm(d)
    if nd < 1e-3:
        return None
    d /= nd
    M = np.stack([a.normal, b.normal, d])
    rhs = np.array([a.normal @ a.origin, b.normal @ b.origin, d @ b.center])
    x0 = torch.linalg.solve(torch.from_numpy(M), torch.from_numpy(rhs)).numpy()

    t = np.arange(-args.line_extent_mm, args.line_extent_mm + 1e-6, args.line_step_mm)
    pts = torch.tensor(x0[None] + t[:, None] * d[None], dtype=torch.float32, device=device)

    valid = inside(a, pts) & inside(b, pts)
    masked_planes = [p for p in (a, b) if p.mask is not None]
    if masked_planes:
        in_mask = torch.zeros_like(valid)
        for p in masked_planes:
            in_mask |= sample_image(p.mask, voxel_grid(p, pts), "nearest")[0] > 0.5
        valid &= in_mask

    n_points = int(valid.sum().item())
    if n_points < args.min_line_mm / args.line_step_mm:
        return None
    idx = valid.nonzero().squeeze(1)
    lo, hi = int(idx[0]), int(idx[-1]) + 1
    return Line(t=torch.tensor(t[lo:hi], dtype=torch.float32, device=device), w=valid[lo:hi].float(), n_points=float(n_points),
                x0=x0, d=d)


# --------------------------------------------------------------------------------------
# Pose-dependent line sampling
# --------------------------------------------------------------------------------------

def skew(v: torch.Tensor) -> torch.Tensor:
    """(N, 3) rotation vectors -> (N, 3, 3) skew-symmetric matrices."""
    z = torch.zeros_like(v[:, 0])
    return torch.stack([
        torch.stack([z, -v[:, 2], v[:, 1]], -1),
        torch.stack([v[:, 2], z, -v[:, 0]], -1),
        torch.stack([-v[:, 1], v[:, 0], z], -1),
    ], -2)


def rodrigues(v: torch.Tensor) -> torch.Tensor:
    """(N, 3) rotation vectors (radians) -> (N, 3, 3) rotation matrices, closed form (= matrix_exp(skew(v))).

    Uses the Taylor expansion near zero so both value and gradient are exact at v = 0.
    """
    t2 = (v * v).sum(-1)
    small = t2 < 1e-8
    th = torch.sqrt(t2.clamp_min(1e-12))
    a = torch.where(small, 1.0 - t2 / 6.0, torch.sin(th) / th)
    b = torch.where(small, 0.5 - t2 / 24.0, (1.0 - torch.cos(th)) / t2.clamp_min(1e-12))
    k = skew(v)
    eye = torch.eye(3, dtype=v.dtype, device=v.device)
    return eye + a[:, None, None] * k + b[:, None, None] * (k @ k)


def solve3_rows(r0: torch.Tensor, r1: torch.Tensor, r2: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Solve [r0; r1; r2] x = rhs for a batch of 3x3 systems given by their rows (M, 3) each; closed form, no sync."""
    c12, c20, c01 = torch.cross(r1, r2, dim=1), torch.cross(r2, r0, dim=1), torch.cross(r0, r1, dim=1)
    det = (r0 * c12).sum(1, keepdim=True)
    return (rhs[:, 0:1] * c12 + rhs[:, 1:2] * c20 + rhs[:, 2:3] * c01) / det


def rotation_matrices(theta_deg: np.ndarray) -> np.ndarray:
    """(N, 3) rotation vectors in degrees -> (N, 3, 3) rotation matrices."""
    th = torch.tensor(np.asarray(theta_deg, dtype=np.float64) * (math.pi / 180.0))
    return rodrigues(th).numpy()


def ncc(a: torch.Tensor, b: torch.Tensor, w: torch.Tensor, use_gradient: bool) -> torch.Tensor:
    """Weighted NCC along the last axis; a, b, w broadcast against each other."""
    if use_gradient:
        a = a[..., 1:] - a[..., :-1]
        b = b[..., 1:] - b[..., :-1]
        w = w[..., 1:] * w[..., :-1]
    ws = w.sum(-1, keepdim=True).clamp_min(1.0)
    am = (a * w).sum(-1, keepdim=True) / ws
    bm = (b * w).sum(-1, keepdim=True) / ws
    ac, bc = a - am, b - bm
    num = (w * ac * bc).sum(-1)
    den = torch.sqrt((w * ac * ac).sum(-1) * (w * bc * bc).sum(-1)) + 1e-6
    return num / den


def soft_nmi(a: torch.Tensor, b: torch.Tensor, w: torch.Tensor, bins: int = 24, sigma_bins: float = 1.0) -> torch.Tensor:
    """Differentiable normalized mutual information 2 I(A;B) / (H(A) + H(B)) per row, in [0, 1].

    a, b, w: (M, S). Each row is z-scored with its own weighted statistics (MI is invariant to monotonic
    intensity maps, so only robustness matters here), clamped to +-3 SD and mapped to [0, 1]. Samples are
    soft-assigned to `bins` bins with a Gaussian Parzen window of `sigma_bins` bin widths; the joint
    histogram is accumulated with the sample weights w (mask / padding).
    """
    eps = 1e-10
    ws = w.sum(-1, keepdim=True).clamp_min(1.0)

    def to_unit(x):
        m = (x * w).sum(-1, keepdim=True) / ws
        sd = torch.sqrt((w * (x - m) ** 2).sum(-1, keepdim=True) / ws + 1e-6)
        return ((x - m) / sd).clamp(-3.0, 3.0) / 6.0 + 0.5

    centers = (torch.arange(bins, device=a.device, dtype=a.dtype) + 0.5) / bins
    sig = sigma_bins / bins

    def parzen(x):
        k = torch.exp(-0.5 * ((x[..., None] - centers) / sig) ** 2)            # (M, S, bins)
        return k / k.sum(-1, keepdim=True).clamp_min(eps)

    pa, pb = parzen(to_unit(a)), parzen(to_unit(b))
    joint = torch.einsum("msk,msl,ms->mkl", pa, pb, w) / ws[..., None]         # (M, bins, bins)
    pa_m, pb_m = joint.sum(2), joint.sum(1)
    h_ab = -(joint * torch.log(joint + eps)).sum((1, 2))
    h_a = -(pa_m * torch.log(pa_m + eps)).sum(1)
    h_b = -(pb_m * torch.log(pb_m + eps)).sum(1)
    mi = h_a + h_b - h_ab
    return 2.0 * mi / (h_a + h_b).clamp_min(eps)


class LineSampler:
    """Samples both images (or segmentations) of every intersection line for arbitrary rigid poses.

    Lines are recomputed from the posed planes, so through-plane shifts and rotations are
    observable. Planes of equal image size are batched into a single grid_sample call.
    """

    def __init__(self, planes: list[Plane], lines: list[tuple[int, int, Line]], n_band: int, device,
                 psf_model: str = "pulse", psf_tbw: float = 2.5, psf_scale_max: float = 1.0):
        self.planes, self.device = planes, device
        self.N, self.M = len(planes), len(lines)
        self.nf = planes[0].img.shape[0]
        self.psf_model, self.psf_tbw = psf_model, psf_tbw
        # through-plane slice profile, as a function of offset x in units of the nominal thickness T (= the FWHM)
        if psf_model == "gaussian":
            self.psf_sigma = 1.0 / 2.3548                                       # FWHM = T
            self.psf_support = 2.0 * self.psf_sigma                             # +-2 sigma
        elif psf_model == "rect":
            self.psf_sigma = 0.0
            self.psf_support = 0.5
        else:   # "pulse": rect of width T convolved with a Gaussian edge; 10-90% transition width ~ T / TBW
            self.psf_sigma = 1.0 / psf_tbw / 2.5631                             # 10-90% width of a Gaussian = 2.563 sigma
            self.psf_support = 0.5 + 2.5 * self.psf_sigma
        # PSF band samples: fixed offsets (units of the other slice's thickness) covering the widest profile used
        # (psf_scale_max, for PSF annealing) at the same density as n_band samples cover the nominal profile; the
        # weights are recomputed from the current scale (self.psf_scale, a device scalar so CUDA graphs see updates).
        smax = max(1.0, psf_scale_max)
        self.K = max(1, int(math.ceil(max(1, n_band) * smax))) if n_band > 1 else 1
        self.u_img = (torch.linspace(-self.psf_support * smax, self.psf_support * smax, self.K, device=device)
                      if self.K > 1 else torch.zeros(1, device=device))
        self.u_seg, self.w_seg = torch.zeros(1, device=device), torch.ones(1, device=device)
        self.psf_scale = torch.ones((), device=device)
        self.cur_sigma = 0.0

        Lmax = max(l.t.numel() for _, _, l in lines)
        self.Lmax = Lmax
        self.T = torch.zeros(self.M, Lmax, device=device)
        self.W = torch.zeros(self.M, Lmax, device=device)
        for m, (_, _, l) in enumerate(lines):
            n = l.t.numel()
            self.T[m, :n] = l.t
            self.T[m, n:] = l.t[-1]
            self.W[m, :n] = l.w
        self.IA = torch.tensor([ia for ia, _, _ in lines], device=device)
        self.IB = torch.tensor([ib for _, ib, _ in lines], device=device)
        self.npts = torch.tensor([l.n_points for _, _, l in lines], device=device)
        self.lax_lax = torch.tensor([planes[ia].image != "SAX" and planes[ib].image != "SAX" for ia, ib, _ in lines], device=device)
        la_idx = [torch.nonzero(self.IA == p).squeeze(1) for p in range(self.N)]
        lb_idx = [torch.nonzero(self.IB == p).squeeze(1) for p in range(self.N)]
        self.plane_lines = [torch.cat([la_idx[p], lb_idx[p]]) for p in range(self.N)]

        f32 = dict(dtype=torch.float32, device=device)
        self.O = torch.tensor(np.stack([p.origin for p in planes]), **f32)
        self.C = torch.tensor(np.stack([p.center for p in planes]), **f32)
        self.Nrm = torch.tensor(np.stack([p.normal for p in planes]), **f32)
        self.thick = torch.tensor([p.thickness for p in planes], **f32)
        self.inv_rot = torch.stack([p.inv_rot for p in planes])
        self.inv_trans = torch.stack([p.inv_trans for p in planes])

        self.groups = []
        by_size = {}
        for p, plane in enumerate(planes):
            by_size.setdefault((plane.H, plane.W), []).append(p)
        for (H, W), members in by_size.items():
            entries = [[(int(m), 0) for m in la_idx[p]] + [(int(m), 1) for m in lb_idx[p]] for p in members]
            nmax = max(len(e) for e in entries)
            if nmax == 0:
                continue
            line_ids = torch.zeros(len(members), nmax, dtype=torch.long, device=device)
            side = torch.zeros(len(members), nmax, dtype=torch.long, device=device)
            e_g, e_slot, e_m, e_side = [], [], [], []
            for g, e in enumerate(entries):
                for s, (m, sd) in enumerate(e):
                    line_ids[g, s], side[g, s] = m, sd
                    e_g.append(g), e_slot.append(s), e_m.append(m), e_side.append(sd)
            imgs = torch.cat([planes[p].img.transpose(0, 1) for p in members])
            segs = torch.cat([planes[p].seg.transpose(0, 1) if planes[p].seg is not None
                              else torch.full_like(planes[p].img.transpose(0, 1), -1.0) for p in members])
            self.groups.append(dict(
                H=H, W=W, nmax=nmax, pidx=torch.tensor(members, device=device), img=imgs.clone(), img_raw=imgs, seg=segs,
                spacing=torch.tensor([planes[p].spacing for p in members], dtype=torch.float32, device=device),
                spacing_list=[list(planes[p].spacing) for p in members],
                line_ids=line_ids, side=side,
                e_g=torch.tensor(e_g, device=device), e_slot=torch.tensor(e_slot, device=device),
                e_m=torch.tensor(e_m, device=device), e_side=torch.tensor(e_side, device=device),
            ))

        # static scatter indices from the per-group profiles back to (line, side) -- precomputed so that profiles()
        # does no boolean indexing (which needs a host sync and cannot be captured in a CUDA graph)
        em = torch.cat([gr["e_m"] for gr in self.groups])
        es = torch.cat([gr["e_side"] for gr in self.groups])
        self.pos_a, self.pos_b = torch.nonzero(es == 0).squeeze(1), torch.nonzero(es == 1).squeeze(1)
        self.m_a, self.m_b = em[self.pos_a], em[self.pos_b]

        # which labels each plane's segmentation contains anywhere, per frame: (N, F, max_label + 1)
        max_label = max([int(p.seg.max()) for p in planes if p.seg is not None] + [0])
        labs = torch.arange(max_label + 1, device=device, dtype=torch.float32)
        self.label_present = torch.stack([
            (p.seg[:, 0, :, :, None] == labs).flatten(1, 2).any(1) if p.seg is not None
            else torch.zeros(self.nf, max_label + 1, dtype=torch.bool, device=device) for p in planes])

    def counted_labels(self, a: torch.Tensor, b: torch.Tensor, labels: list[int], labels_sax: list[int]):
        """For each label: per (line, frame) whether it is counted, and the two on-line masks.

        A label is counted on a line/frame when it is allowed for the pair type, both planes contain it somewhere in
        their segmentation, and at least one of the two line profiles hits it.
        """
        valid_pt = (self.W > 0)[:, None, :]
        pres_a, pres_b = self.label_present[self.IA], self.label_present[self.IB]              # (M, F, K)
        for lab in sorted(set(labels) | set(labels_sax)):
            if lab >= pres_a.shape[-1]:
                continue
            allowed = torch.where(self.lax_lax, lab in labels, lab in labels_sax)[:, None]       # (M, 1)
            ma, mb = (a == lab) & valid_pt, (b == lab) & valid_pt                                # (M, F, L)
            counted = allowed & pres_a[..., lab] & pres_b[..., lab] & (ma.any(-1) | mb.any(-1))  # (M, F)
            yield lab, counted, ma, mb

    @torch.no_grad()
    def set_blur(self, sigma_mm: float) -> None:
        """Write in-plane Gaussian-blurred versions of the originals (sigma in mm) into the persistent image buffers.

        The buffers are updated in place (CUDA graphs keep pointing at them); nothing is done if sigma is unchanged.
        """
        if sigma_mm == self.cur_sigma:
            return
        self.cur_sigma = sigma_mm
        for gr in self.groups:
            if sigma_mm <= 1e-6:
                gr["img"].copy_(gr["img_raw"])
                continue
            raw = gr["img_raw"]                                                    # (G, F, H, W)
            G, Fn, H, W = raw.shape
            sig = [[max(sigma_mm / s, 1e-3) for s in row] for row in gr["spacing_list"]]   # voxels, per plane/axis
            r = int(math.ceil(3.0 * max(max(row) for row in sig)))
            x = torch.arange(-r, r + 1, device=raw.device, dtype=raw.dtype)
            sig_t = torch.tensor(sig, device=raw.device, dtype=raw.dtype)            # (G, 2)
            k = torch.exp(-0.5 * (x[None, None, :] / sig_t[:, :, None]) ** 2)      # (G, 2, 2r+1)
            k = (k / k.sum(-1, keepdim=True)).repeat_interleave(Fn, dim=0)          # (G*F, 2, 2r+1)
            out = F.pad(raw.reshape(1, G * Fn, H, W), (r, r, r, r), mode="replicate")
            out = F.conv2d(out, k[:, 0][:, None, :, None], groups=G * Fn)          # along image axis 0 (H)
            out = F.conv2d(out, k[:, 1][:, None, None, :], groups=G * Fn)          # along image axis 1 (W)
            gr["img"].copy_(out.reshape(G, Fn, H, W))

    def set_psf_scale(self, scale: float) -> None:
        """Width of the PSF band relative to the nominal slice profile (1 = physical)."""
        self.psf_scale.fill_(scale)

    def band_weights(self) -> torch.Tensor:
        """PSF weights of the band samples at the current scale (sum to 1)."""
        if self.K == 1:
            return self.w_seg
        w = self.slice_profile(self.u_img / self.psf_scale)
        return w / w.sum()

    def slice_profile(self, x: torch.Tensor) -> torch.Tensor:
        """Relative excitation at through-plane offset x (units of the nominal thickness)."""
        if self.psf_model == "gaussian":
            return torch.exp(-0.5 * (x / self.psf_sigma) ** 2)
        if self.psf_model == "rect":
            return (x.abs() <= 0.5).float()
        s = self.psf_sigma * math.sqrt(2.0)
        return 0.5 * (torch.erf((x + 0.5) / s) - torch.erf((x - 0.5) / s))

    def profiles(self, theta_deg: torch.Tensor, t: torch.Tensor, kind: str = "img") -> tuple[torch.Tensor, torch.Tensor]:
        """Profiles of both planes of every line at the given poses -> (M, F, Lmax) x 2."""
        band = kind == "img"
        mode = "bilinear" if band else "nearest"
        # PSF: samples are placed across the other slice's profile (through-plane offsets u, in units of that
        # slice's thickness T) and weighted by the slice profile at the current PSF scale.
        K = self.K if band else 1
        u = self.u_img if band else self.u_seg
        wts = self.band_weights() if band else self.w_seg

        R = rodrigues(theta_deg * (math.pi / 180.0))                           # (N, 3, 3)
        Op = (R @ (self.O - self.C)[..., None])[..., 0] + self.C + t
        Np = (R @ self.Nrm[..., None])[..., 0]
        Cp = self.C + t

        IA, IB = self.IA, self.IB
        na, nb = Np[IA], Np[IB]
        d = torch.cross(na, nb, dim=1)
        d = d / d.norm(dim=1, keepdim=True)
        rhs = torch.stack([(na * Op[IA]).sum(1), (nb * Op[IB]).sum(1), (d * Cp[IB]).sum(1)], 1)
        x0 = solve3_rows(na, nb, d, rhs)
        pts = x0[:, None, :] + self.T[:, :, None] * d[:, None, :]              # (M, Lmax, 3)

        # band directions: the other plane's normal projected into this plane. Moving a distance x along this
        # in-plane direction moves sin(theta) * x along the other plane's normal (theta = crossing angle), so a
        # through-plane offset of x*T in the other slice is x*T/sin(theta) within this plane.
        ma = nb - (nb * na).sum(1, keepdim=True) * na
        sin_ab = ma.norm(dim=1, keepdim=True).clamp_min(1e-3)                  # |n_b x n_a| = sin(theta)
        ma = ma / sin_ab
        mb = na - (na * nb).sum(1, keepdim=True) * nb
        mb = mb / mb.norm(dim=1, keepdim=True).clamp_min(1e-3)
        sig_a = self.thick[IB] / sin_ab[:, 0]                                 # mm per unit of u inside A (B's slab)
        sig_b = self.thick[IA] / sin_ab[:, 0]                                 # and inside B (A's slab)

        prof_parts = []
        for gr in self.groups:
            li, sd, pidx = gr["line_ids"], gr["side"], gr["pidx"]
            mdir = torch.where(sd[..., None] == 0, ma[li], mb[li])
            sig = torch.where(sd == 0, sig_a[li], sig_b[li])
            P = pts[li][:, :, :, None, :] + u[None, None, None, :, None] * sig[:, :, None, None, None] * mdir[:, :, None, None, :]
            G = P.shape[0]
            P = P.reshape(G, -1, 3)
            # world -> plane-local (undo the pose) -> voxel, folded into one affine map per plane
            A = torch.einsum("gij,gkj->gik", self.inv_rot[pidx], R[pidx])            # inv_rot @ R^T
            bvec = (torch.einsum("gij,gj->gi", self.inv_rot[pidx], self.C[pidx]) + self.inv_trans[pidx]
                    - torch.einsum("gij,gj->gi", A, Cp[pidx]))
            vox = torch.einsum("gnj,gij->gni", P, A) + bvec[:, None, :]
            grid = torch.stack([2 * vox[..., 1] / (gr["W"] - 1) - 1, 2 * vox[..., 0] / (gr["H"] - 1) - 1], -1)
            out = F.grid_sample(gr[kind], grid[:, None], mode=mode, padding_mode="zeros", align_corners=True)
            out = (out[:, :, 0].view(G, self.nf, gr["nmax"], self.Lmax, K) * wts).sum(-1)
            prof_parts.append(out[gr["e_g"], :, gr["e_slot"]])
        prof = torch.cat(prof_parts)
        profA = torch.zeros(self.M, self.nf, self.Lmax, device=self.device).index_put((self.m_a,), prof[self.pos_a])
        profB = torch.zeros(self.M, self.nf, self.Lmax, device=self.device).index_put((self.m_b,), prof[self.pos_b])
        return profA, profB

    def ncc_lines(self, theta_deg, t, use_gradient: bool) -> torch.Tensor:
        a, b = self.profiles(theta_deg, t, "img")
        return ncc(a, b, self.W[:, None, :], use_gradient).mean(1)             # (M,)

    def mi_lines(self, theta_deg, t, bins: int, sigma_bins: float) -> torch.Tensor:
        """Normalized mutual information per line, samples pooled over frames -> (M,)."""
        a, b = self.profiles(theta_deg, t, "img")
        w = self.W[:, None, :].expand_as(a)
        return soft_nmi(a.reshape(self.M, -1), b.reshape(self.M, -1), w.reshape(self.M, -1), bins, sigma_bins)

    def similarity_lines(self, theta_deg, t, args) -> torch.Tensor:
        """The optimized similarity per line (NCC, gradient NCC or normalized MI)."""
        if args.metric == "mi":
            return self.mi_lines(theta_deg, t, args.mi_bins, args.mi_sigma)
        return self.ncc_lines(theta_deg, t, args.metric == "ncc_grad")

    def dice_lines(self, theta_deg, t, labels: list[int], labels_sax: list[int]) -> torch.Tensor:
        """Mean Dice over the counted labels (see counted_labels) along each line -> (M,).

        A counted label that the two profiles place at non-overlapping positions -- including the case where only
        one profile hits it -- scores 0. Per frame: mean over counted labels; then mean over frames with at least
        one counted label. NaN if no label is counted on any frame. `labels` applies to LAX-LAX lines,
        `labels_sax` to SAX-LAX lines (the SAX model lacks some classes).
        """
        a, b = self.profiles(theta_deg, t, "seg")
        total = torch.zeros(a.shape[:2], device=self.device)
        count = torch.zeros(a.shape[:2], device=self.device)
        for lab, counted, ma, mb in self.counted_labels(a, b, labels, labels_sax):
            inter = (ma & mb).sum(-1).float()
            tot = (ma.sum(-1) + mb.sum(-1)).float()
            dice = 2 * inter / tot.clamp_min(1)
            total += torch.where(counted, dice, torch.zeros_like(dice))
            count += counted.float()
        valid = count > 0
        per_frame = total / count.clamp_min(1)
        out = (per_frame * valid).sum(1) / valid.sum(1).clamp_min(1)
        return torch.where(valid.any(1), out, torch.full_like(out, float("nan")))

    def hd_lines(self, theta_deg, t, labels: list[int], labels_sax: list[int], miss_mm: float = 100.0) -> torch.Tensor:
        """1D Hausdorff distance (mm) between the two views' labels along each line -> (M,).

        Counted labels as in dice_lines. If only one profile hits a counted label the distance is undefined
        (infinite); it is reported as `miss_mm`. Per frame: the worst (largest) value over the counted labels;
        then the mean over frames with at least one counted label. NaN if no label is counted on any frame.
        """
        a, b = self.profiles(theta_deg, t, "seg")
        dist = (self.T[:, :, None] - self.T[:, None, :]).abs()[:, None]          # (M, 1, L, L) mm along the line
        inf = torch.tensor(float("inf"), device=self.device)
        worst = torch.full(a.shape[:2], -1.0, device=self.device)               # (M, F)
        for lab, counted, ma, mb in self.counted_labels(a, b, labels, labels_sax):
            if not counted.any():
                continue
            both = ma.any(-1) & mb.any(-1)
            d_ab = torch.where(mb[:, :, None, :], dist, inf).amin(-1)            # each A point to nearest B point
            d_ba = torch.where(ma[:, :, :, None], dist, inf).amin(-2)            # each B point to nearest A point
            h = torch.maximum(torch.where(ma, d_ab, -inf).amax(-1), torch.where(mb, d_ba, -inf).amax(-1))
            h = torch.where(both, h, torch.full_like(h, miss_mm))
            worst = torch.where(counted, torch.maximum(worst, h), worst)
        ok = worst >= 0
        out = (worst.clamp_min(0) * ok).sum(1) / ok.sum(1).clamp_min(1)
        return torch.where(ok.any(1), out, torch.full_like(out, float("nan")))

    def per_plane(self, values: torch.Tensor) -> list[float]:
        """Line values (M,) -> npts-weighted mean per plane, ignoring NaN."""
        out = []
        for idx in self.plane_lines:
            v, w = values[idx], self.npts[idx]
            ok = torch.isfinite(v)
            out.append(float((v[ok] * w[ok]).sum() / w[ok].sum()) if ok.any() else float("nan"))
        return out

    def per_plane_max(self, values: torch.Tensor) -> list[float]:
        """Line values (M,) -> worst (largest) value per plane, ignoring NaN."""
        out = []
        for idx in self.plane_lines:
            v = values[idx]
            v = v[torch.isfinite(v)]
            out.append(float(v.max()) if v.numel() else float("nan"))
        return out

    @torch.no_grad()
    def evaluate(self, theta_deg, t, args, labels, labels_sax, has_seg: bool) -> dict:
        """All reported per-line metrics at the given poses, as numpy arrays of shape (M,)."""
        nan = torch.full((self.M,), float("nan"), device=self.device)
        out = dict(ncc=self.ncc_lines(theta_deg, t, args.metric == "ncc_grad"),
                   mi=self.mi_lines(theta_deg, t, args.mi_bins, args.mi_sigma),
                   dice=self.dice_lines(theta_deg, t, labels, labels_sax) if has_seg else nan,
                   hd=self.hd_lines(theta_deg, t, labels, labels_sax, getattr(args, "hd_miss_mm", 100.0)) if has_seg else nan)
        return {k: v.float() for k, v in out.items()}


def align(final_sampler: LineSampler, args, stages: list[tuple[int, LineSampler]] | None = None) -> dict:
    """Jointly optimize a 6-DOF rigid pose per plane from the acquired geometry.

    `stages` optionally lists (start_step, sampler) for the annealed line/mask geometry used early in the run, ending
    with (anneal_end, final_sampler); `final_sampler` -- the nominal geometry -- is used from anneal_end on (or always,
    if no stages are given). The best pose is only tracked on the nominal geometry.
    """
    device, N = final_sampler.device, final_sampler.N
    stages = sorted(stages or [(0, final_sampler)], key=lambda s: s[0])
    anneal_end = stages[-1][0]

    def sampler_at(step):
        active = stages[0][1]
        for start, smp in stages:
            if step >= start:
                active = smp
        return active

    f32 = dict(dtype=torch.float32, device=device)
    theta_deg = torch.zeros(N, 3, **f32).requires_grad_(True)
    t = torch.zeros(N, 3, **f32).requires_grad_(True)

    def objective(theta_deg, t, keep=None, smp=final_sampler):
        c = smp.similarity_lines(theta_deg, t, args)
        w = smp.npts if keep is None else smp.npts * keep
        score = (c * w).sum() / w.sum()
        gauge = args.gauge * (t.mean(0).pow(2).sum() + theta_deg.mean(0).pow(2).sum())
        dz = (t * smp.Nrm).sum(1)
        prior = args.prior * (theta_deg.pow(2).sum(1) + dz.pow(2)).mean()
        return score - gauge - prior

    # stochastic line-pair dropping: each step keeps a random subset of the intersection lines
    keep_frac = getattr(args, "pair_keep", 1.0)
    stochastic = keep_frac < 1.0
    gen = torch.Generator(device=device)
    gen.manual_seed(getattr(args, "seed", 0))
    eval_every = max(1, getattr(args, "eval_every", 10))

    def draw_keep(smp):
        keep = (torch.rand(smp.M, generator=gen, device=device) < keep_frac).float()
        if keep.sum() == 0:
            keep[torch.randint(smp.M, (1,), generator=gen, device=device)] = 1.0
        return keep

    param_groups = [{"params": [t], "lr": args.lr}]
    if args.dof == 6 and args.rot_lr > 0:
        param_groups.append({"params": [theta_deg], "lr": args.rot_lr})
    opt = torch.optim.Adam(param_groups)
    ends = [args.lr_end] + ([args.rot_lr_end] if len(param_groups) > 1 else [])

    def decay(end, start):
        if args.lr_schedule == "constant" or start <= 0:
            return lambda step: 1.0
        return lambda step: 1.0 + (end / start - 1.0) * min(1.0, step / max(1, args.steps - 1))   # linear start -> end

    sched = torch.optim.lr_scheduler.LambdaLR(opt, [decay(e, g["lr"]) for e, g in zip(ends, param_groups)])
    best_obj, best, best_step = -float("inf"), (theta_deg.detach().clone(), t.detach().clone()), 0
    blur_sigma = getattr(args, "blur_sigma", 0.0)
    blur_steps = int(round(getattr(args, "blur_frac", 0.6) * args.steps)) if blur_sigma > 0 else 0
    # pose snapshots every history_every of the run; step k is the pose *before* the k-th update, step=steps the final one
    every = getattr(args, "history_every", 0.0)
    snap_steps = sorted({int(round(k * every * args.steps)) for k in range(int(round(1 / every)) + 1)}) if every > 0 else []
    snapshots = []
    with torch.enable_grad():
        for step in range(args.steps):
            if step in snap_steps:
                snapshots.append((step, theta_deg.detach().clone(), t.detach().clone()))
            sigma = blur_sigma * max(0.0, 1.0 - step / blur_steps) if step < blur_steps else 0.0
            smp = sampler_at(step)
            smp.set_blur(sigma)
            opt.zero_grad()
            obj = objective(theta_deg, t, draw_keep(smp) if stochastic else None, smp)
            # objectives at different blur levels / line subsets / line geometries are not comparable: track the best
            # on the full, unblurred, nominal-geometry objective (every step when deterministic, else periodically)
            if sigma == 0.0 and step >= anneal_end:
                if not stochastic:
                    full = float(obj.detach())
                elif step % eval_every == 0 or step == args.steps - 1:
                    with torch.no_grad():
                        full = float(objective(theta_deg, t))
                else:
                    full = -float("inf")
                if full > best_obj:
                    best_obj, best, best_step = full, (theta_deg.detach().clone(), t.detach().clone()), step
            (-obj).backward()
            opt.step()
            sched.step()
    for _, smp in stages:
        smp.set_blur(0.0)
    if args.steps in snap_steps:
        snapshots.append((args.steps, theta_deg.detach().clone(), t.detach().clone()))
    with torch.no_grad():
        if stochastic:                                     # the final parameters are also a candidate
            full = float(objective(theta_deg, t))
            if full > best_obj:
                best, best_step = (theta_deg.detach().clone(), t.detach().clone()), args.steps
    return dict(theta_deg=best[0], t=best[1], best_step=best_step, snapshots=snapshots)


# --------------------------------------------------------------------------------------
# Per-subject pipeline
# --------------------------------------------------------------------------------------

def relative_translation(t: torch.Tensor, lax_rows: list[int]) -> np.ndarray:
    """Translations expressed relative to the mean LAX pose (the convention used for all outputs)."""
    t = t.detach().cpu().numpy().astype(np.float64)
    return t - t[lax_rows].mean(0) if lax_rows else t


def base_affine(data: dict, plane: Plane) -> np.ndarray:
    """The acquired 4x4 affine of one plane (for SAX: the stack affine with the origin at that slice)."""
    if plane.image == "SAX":
        A = data["sax"][1].copy()
        A[:3, 3] = plane.origin
        return A
    return data["lax"][plane.image]["affine"].copy()


AFFINE_COLUMNS = [f"a{r}{c}" for r in range(3) for c in range(4)]
HISTORY_COLUMNS = (["subject_id", "step", "fraction", "is_best", "image", "slice"] + AFFINE_COLUMNS
                   + ["dx_mm", "dy_mm", "dz_mm", "rot_x_deg", "rot_y_deg", "rot_z_deg", "rot_inplane_deg", "rot_tilt_deg",
                      "ncc", "mi", "seg_dice", "hd_mm"])
HISTORY_STEP_COLUMNS = ["subject_id", "step", "fraction", "is_best", "objective", "ncc_mean", "mi_mean", "seg_dice_mean",
                        "hd_mean_mm", "hd_max_mm", "sax_shift_rms_mm", "sax_shift_jitter_mm", "sax_dz_rms_mm", "sax_tilt_rms_deg"]


def build_history(sid, data, planes, n_pairs, sampler, snapshots, best_step, lax_rows, args, labels, labels_sax, has_seg) -> dict:
    """Affines, poses and metrics of every pose snapshot. Snapshot step 'best' is the pose that is saved/reported,
    re-labelled with the step it was found at; step 0 is the acquired geometry (identity pose)."""
    nan = float("nan")
    plane_rows, step_rows = [], []
    for step, th, tt in snapshots:
        is_best = step == "best"
        step_no = best_step if is_best else step
        lm = sampler.evaluate(th, tt, args, labels, labels_sax, has_seg)
        with torch.no_grad():
            sim = sampler.similarity_lines(th, tt, args)
            objective = float((sim * sampler.npts).sum() / sampler.npts.sum())
        per = {k: (sampler.per_plane_max if k == "hd" else sampler.per_plane)(lm[k]) for k in ("ncc", "mi", "dice", "hd")}
        theta = th.detach().cpu().numpy().astype(np.float64)
        t_rel = relative_translation(tt, lax_rows)
        Rs = rotation_matrices(theta)
        sax_vec, sax_dz, sax_tilt = [], [], []
        for i, p in enumerate(planes):
            A = corrected_affine(base_affine(data, p), p, Rs[i], t_rel[i])
            dx, dy, dz = float(t_rel[i] @ p.u_hat), float(t_rel[i] @ p.v_hat), float(t_rel[i] @ p.normal)
            inplane = float(theta[i] @ p.normal)
            tilt = float(np.linalg.norm(theta[i] - inplane * p.normal))
            if p.image == "SAX" and n_pairs[i] > 0:
                sax_vec.append((dx, dy)), sax_dz.append(dz), sax_tilt.append(tilt)
            plane_rows.append(dict(subject_id=sid, step=step_no, fraction=step_no / args.steps, is_best=int(is_best),
                                   image=p.image, slice=p.slice_idx,
                                   **{c: float(A[int(c[1]), int(c[2])]) for c in AFFINE_COLUMNS},
                                   dx_mm=dx, dy_mm=dy, dz_mm=dz, rot_x_deg=float(theta[i, 0]), rot_y_deg=float(theta[i, 1]),
                                   rot_z_deg=float(theta[i, 2]), rot_inplane_deg=inplane, rot_tilt_deg=tilt,
                                   ncc=per["ncc"][i] if n_pairs[i] else nan, mi=per["mi"][i] if n_pairs[i] else nan,
                                   seg_dice=per["dice"][i] if n_pairs[i] else nan, hd_mm=per["hd"][i] if n_pairs[i] else nan))
        vec = np.array(sax_vec).reshape(-1, 2)
        ok = lambda x: np.asarray(x, dtype=float)[np.isfinite(np.asarray(x, dtype=float))]  # noqa: E731
        hd_lines = ok(lm["hd"].cpu().numpy())
        step_rows.append(dict(
            subject_id=sid, step=step_no, fraction=step_no / args.steps, is_best=int(is_best), objective=objective,
            ncc_mean=float(np.nanmean(per["ncc"])), mi_mean=float(np.nanmean(per["mi"])),
            seg_dice_mean=float(np.nanmean(per["dice"])) if np.isfinite(per["dice"]).any() else nan,
            hd_mean_mm=float(hd_lines.mean()) if hd_lines.size else nan, hd_max_mm=float(hd_lines.max()) if hd_lines.size else nan,
            sax_shift_rms_mm=float(np.sqrt((vec ** 2).sum(1).mean())) if len(vec) else nan,
            sax_shift_jitter_mm=float(np.sqrt(((vec - vec.mean(0)) ** 2).sum(1).mean())) if len(vec) else nan,
            sax_dz_rms_mm=float(np.sqrt(np.mean(np.square(sax_dz)))) if sax_dz else nan,
            sax_tilt_rms_deg=float(np.sqrt(np.mean(np.square(sax_tilt)))) if sax_tilt else nan))
    return dict(planes=plane_rows, steps=step_rows)


def annealed_stages(data, planes, sax_planes, lax_planes, final_sampler, args, device) -> list | None:
    """(start_step, LineSampler) stages for line-geometry / mask-dilation annealing, ending with the nominal sampler.

    The line scale s goes linearly from --line-scale-start to 1 over the first --line-anneal-frac of the run (line
    extent and line step are both multiplied by s, so the line shrinks while the sampling gets denser at a constant
    number of samples); the mask dilation goes linearly from --mask-dilation-start-mm to --mask-dilation-mm over the
    first --mask-anneal-frac. Each schedule advances in --anneal-stages discrete steps; the lines are rebuilt whenever
    either schedule advances. Returns None when neither is enabled.
    """
    scale0 = getattr(args, "line_scale_start", 1.0)
    dil0 = getattr(args, "mask_dilation_start_mm", None)
    n_stages = max(1, getattr(args, "anneal_stages", 6))
    line_end = int(round(getattr(args, "line_anneal_frac", 0.8) * args.steps)) if scale0 != 1.0 else 0
    mask_end = int(round(getattr(args, "mask_anneal_frac", 0.8) * args.steps)) if dil0 is not None else 0
    anneal_end = max(line_end, mask_end)
    if anneal_end <= 0:
        return None

    def stage_frac(step, end):          # piecewise-constant position on a schedule ending at `end` (1 = finished)
        return 1.0 if end <= 0 or step >= end else math.floor(step / end * n_stages) / n_stages

    starts = sorted({int(round(k / n_stages * e)) for e in (line_end, mask_end) if e > 0 for k in range(n_stages + 1)}
                    - {anneal_end})               # includes the end of the shorter schedule; anneal_end = final sampler
    index = {p.key: i for i, p in enumerate(planes)}
    dists = {v: torch.from_numpy(d["dist"]).to(device) for v, d in data["lax"].items()}
    nominal_masks = {p.key: p.mask for p in lax_planes}
    stages = []
    try:
        for start in starts:
            s_args = argparse.Namespace(**vars(args))
            scale = scale0 + (1.0 - scale0) * stage_frac(start, line_end)
            s_args.line_extent_mm = args.line_extent_mm * scale
            s_args.line_step_mm = args.line_step_mm * scale
            if dil0 is not None:
                dil = dil0 + (args.mask_dilation_mm - dil0) * stage_frac(start, mask_end)
                for p in lax_planes:
                    p.mask = (dists[p.image] <= dil).float()[None, None]
            lines = []
            with torch.no_grad():
                for a, b in [(s, l) for s in sax_planes for l in lax_planes] + list(combinations(lax_planes, 2)):
                    line = intersection_line(a, b, s_args, device)
                    if line is not None:
                        lines.append((index[a.key], index[b.key], line))
            if lines:
                stages.append((start, LineSampler(planes, lines, args.n_band, device, args.psf_model, args.psf_tbw)))
    finally:
        for p in lax_planes:
            p.mask = nominal_masks[p.key]
    return stages + [(anneal_end, final_sampler)]


def process_subject(data: dict, args, device, on_lines=None) -> tuple:
    """Returns the summary row, the per-plane rows, the per-line-pair rows, the per-plane poses (plane, R, t) and the
    pose/metric history (dict with 'planes' and 'steps' row lists, empty unless --history-every > 0).

    `on_lines(planes, lines)` is called once the intersection lines are known, before alignment.
    """
    sid = data["subject_id"]
    nan = float("nan")
    sax_arr, sax_aff = data["sax"]
    sax_planes = make_planes("SAX", sax_arr, sax_aff, args.sax_thickness, None, data["sax_seg"], device)
    lax_planes = []
    for view, d in data["lax"].items():
        lax_planes += make_planes(view, d["arr"], d["affine"], d["thickness"], d["mask"], d["seg"], device)
    planes = sax_planes + lax_planes
    index = {p.key: i for i, p in enumerate(planes)}

    lines, n_pairs = [], [0] * len(planes)
    with torch.no_grad():
        for a, b in [(s, l) for s in sax_planes for l in lax_planes] + list(combinations(lax_planes, 2)):
            line = intersection_line(a, b, args, device)
            if line is not None:
                lines.append((index[a.key], index[b.key], line))
                n_pairs[index[a.key]] += 1
                n_pairs[index[b.key]] += 1
    if on_lines is not None:
        on_lines(planes, lines)

    N = len(planes)
    metric_names = ("ncc", "mi", "dice", "hd")
    plane_m = {f"{k}_{s}": [nan] * N for k in metric_names for s in ("zero", "final")}
    pairs = []
    history = dict(planes=[], steps=[])
    n_points = [0.0] * N
    t_rel = np.zeros((N, 3))
    theta = np.zeros((N, 3))
    if lines:
        sampler = LineSampler(planes, lines, args.n_band, device, args.psf_model, args.psf_tbw)
        n_points = [float(sampler.npts[idx].sum()) for idx in sampler.plane_lines]
        labels = [int(x) for x in args.seg_labels.split(",") if x.strip()]
        labels_sax = [int(x) for x in args.sax_seg_labels.split(",") if x.strip()]
        has_seg = any(p.seg is not None for p in planes)
        zeros = torch.zeros(N, 3, device=device)
        line_zero = sampler.evaluate(zeros, zeros, args, labels, labels_sax, has_seg)
        result = align(sampler, args, annealed_stages(data, planes, sax_planes, lax_planes, sampler, args, device))
        line_final = sampler.evaluate(result["theta_deg"], result["t"], args, labels, labels_sax, has_seg)
        for stage, lm in (("zero", line_zero), ("final", line_final)):
            for k in metric_names:
                agg = sampler.per_plane_max if k == "hd" else sampler.per_plane      # Hausdorff: worst line per plane
                plane_m[f"{k}_{stage}"] = agg(lm[k])
        lz = {k: v.cpu().numpy() for k, v in line_zero.items()}
        lf = {k: v.cpu().numpy() for k, v in line_final.items()}
        for m, (ia, ib, line) in enumerate(lines):
            pa, pb = planes[ia], planes[ib]
            pairs.append(dict(subject_id=sid, image_a=pa.image, slice_a=pa.slice_idx, image_b=pb.image, slice_b=pb.slice_idx,
                              kind="LAX-LAX" if pa.image != "SAX" and pb.image != "SAX" else "SAX-LAX", n_points=int(line.n_points),
                              ncc_zero=float(lz["ncc"][m]), ncc=float(lf["ncc"][m]), mi_zero=float(lz["mi"][m]), mi=float(lf["mi"][m]),
                              seg_dice_zero=float(lz["dice"][m]), seg_dice=float(lf["dice"][m]),
                              hd_zero_mm=float(lz["hd"][m]), hd_mm=float(lf["hd"][m])))
        theta = result["theta_deg"].cpu().numpy().astype(np.float64)
        lax_rows = [i for i, p in enumerate(planes) if p.image != "SAX" and n_pairs[i] > 0]
        t_rel = relative_translation(result["t"], lax_rows)
        if result["snapshots"]:
            snaps = result["snapshots"] + [("best", result["theta_deg"], result["t"])]
            history = build_history(sid, data, planes, n_pairs, sampler, snaps, result["best_step"], lax_rows,
                                    args, labels, labels_sax, has_seg)
    Rs = rotation_matrices(theta)
    poses = [(p, Rs[i], t_rel[i]) for i, p in enumerate(planes)]

    details = []
    for i, p in enumerate(planes):
        row = dict(subject_id=sid, image=p.image, slice=p.slice_idx, n_pairs=n_pairs[i], n_points=int(n_points[i]),
                   dx_mm=nan, dy_mm=nan, dz_mm=nan, shift_inplane_mm=nan, shift_mm=nan, rot_inplane_deg=nan,
                   rot_tilt_deg=nan, ncc_zero=nan, ncc=nan, mi_zero=nan, mi=nan, seg_dice_zero=nan, seg_dice=nan,
                   hd_zero_mm=nan, hd_mm=nan)
        if n_pairs[i] > 0:
            tv, th = t_rel[i], theta[i]
            dx, dy, dz = float(tv @ p.u_hat), float(tv @ p.v_hat), float(tv @ p.normal)
            inplane = float(th @ p.normal)
            row.update(dx_mm=dx, dy_mm=dy, dz_mm=dz, shift_inplane_mm=math.hypot(dx, dy), shift_mm=float(np.linalg.norm(tv)),
                       rot_inplane_deg=inplane, rot_tilt_deg=float(np.linalg.norm(th - inplane * p.normal)),
                       ncc_zero=plane_m["ncc_zero"][i], ncc=plane_m["ncc_final"][i],
                       mi_zero=plane_m["mi_zero"][i], mi=plane_m["mi_final"][i],
                       seg_dice_zero=plane_m["dice_zero"][i], seg_dice=plane_m["dice_final"][i],
                       hd_zero_mm=plane_m["hd_zero"][i], hd_mm=plane_m["hd_final"][i])
        details.append(row)

    def arr(rows, fn):
        return np.array([fn(d) for d in rows], dtype=float)

    def stat(fn, x):
        x = x[np.isfinite(x)] if x.size else x
        return float(fn(x)) if x.size else nan

    def rms(x):
        return np.sqrt(np.mean(x ** 2))

    sax_rows = [d for d in details if d["image"] == "SAX" and d["n_pairs"] > 0]
    lax_rows = [d for d in details if d["image"] != "SAX" and d["n_pairs"] > 0]
    ok_rows = [d for d in details if d["n_pairs"] > 0]
    vec = np.array([[d["dx_mm"], d["dy_mm"]] for d in sax_rows]).reshape(-1, 2)
    mag = np.hypot(vec[:, 0], vec[:, 1])
    jit = np.hypot(*(vec - vec.mean(0)).T) if len(vec) else np.array([])
    nz, nf_ = arr(ok_rows, lambda d: d["ncc_zero"]), arr(ok_rows, lambda d: d["ncc"])
    mz, mf = arr(ok_rows, lambda d: d["mi_zero"]), arr(ok_rows, lambda d: d["mi"])

    summary = dict(
        subject_id=sid, status="ok", n_lax_views=len(lax_planes), sax_n_slices=len(sax_planes), sax_n_eval=len(sax_rows),
        sax_shift_mean_mm=stat(np.mean, mag), sax_shift_rms_mm=stat(rms, mag), sax_shift_max_mm=stat(np.max, mag),
        sax_shift_jitter_mm=stat(rms, jit),
        sax_dz_rms_mm=stat(rms, arr(sax_rows, lambda d: d["dz_mm"])),
        sax_rot_inplane_rms_deg=stat(rms, arr(sax_rows, lambda d: d["rot_inplane_deg"])),
        sax_tilt_rms_deg=stat(rms, arr(sax_rows, lambda d: d["rot_tilt_deg"])),
        lax_shift_mean_mm=stat(np.mean, arr(lax_rows, lambda d: d["shift_inplane_mm"])),
        lax_shift_max_mm=stat(np.max, arr(lax_rows, lambda d: d["shift_inplane_mm"])),
        ncc_zero_mean=stat(np.mean, nz), ncc_mean=stat(np.mean, nf_), ncc_gain_mean=stat(np.mean, nf_ - nz),
        mi_zero_mean=stat(np.mean, mz), mi_mean=stat(np.mean, mf), mi_gain_mean=stat(np.mean, mf - mz),
        seg_dice_zero_mean=stat(np.mean, arr(ok_rows, lambda d: d["seg_dice_zero"])),
        seg_dice_mean=stat(np.mean, arr(ok_rows, lambda d: d["seg_dice"])),
        hd_zero_mean_mm=stat(np.mean, arr(pairs, lambda d: d["hd_zero_mm"])), hd_mean_mm=stat(np.mean, arr(pairs, lambda d: d["hd_mm"])),
        hd_zero_max_mm=stat(np.max, arr(pairs, lambda d: d["hd_zero_mm"])), hd_max_mm=stat(np.max, arr(pairs, lambda d: d["hd_mm"])),
        n_line_pairs=len(pairs), steps=args.steps if lines else 0,
    )
    return summary, details, pairs, poses, history


def empty_summary(sid: str, status: str) -> dict:
    row = {c: float("nan") for c in SUMMARY_COLUMNS}
    row.update(subject_id=sid, status=status, n_lax_views=0, sax_n_slices=0, sax_n_eval=0, n_line_pairs=0, steps=0)
    return row


# --------------------------------------------------------------------------------------
# Saving aligned images
# --------------------------------------------------------------------------------------

def corrected_affine(affine: np.ndarray, plane: Plane, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Affine of the plane's content after applying the rigid pose (rotation about the plane centre)."""
    A = np.eye(4)
    A[:3, :3] = R @ affine[:3, :3]
    A[:3, 3] = R @ (plane.origin - plane.center) + plane.center + t
    return A


def write_nifti(vol: np.ndarray, affine: np.ndarray, header: nib.Nifti1Header, path: Path, dtype) -> None:
    if np.issubdtype(dtype, np.unsignedinteger):
        vol = np.clip(np.rint(vol), 0, np.iinfo(dtype).max)
    img = nib.Nifti1Image(np.ascontiguousarray(vol).astype(dtype), affine, header=header)
    img.set_data_dtype(dtype)
    img.header.set_slope_inter(None, None)
    img.set_qform(affine, code=1)
    img.set_sform(affine, code=1)
    nib.save(img, str(path))


def save_aligned(data: dict, poses: list[tuple[Plane, np.ndarray, np.ndarray]], run_dir: Path, args) -> None:
    inst = run_dir / data["subject_id"] / args.instance
    (inst / SEG_DIR).mkdir(parents=True, exist_ok=True)
    for plane, R, t in poses:
        name, z = plane.image, plane.slice_idx
        if name not in data["full"]:
            continue
        arr, affine, header = data["full"][name]
        A = corrected_affine(affine, plane, R, t)
        stem = f"SAX_slice{z:02d}" if name == "SAX" else name
        vol = arr[:, :, z:z + 1] if name == "SAX" else arr
        write_nifti(vol, A, header, inst / f"{stem}.nii.gz", np.uint16)
        if name in data["full_seg"]:
            seg, seg_header = data["full_seg"][name]
            seg_vol = seg[:, :, z:z + 1] if name == "SAX" else seg
            write_nifti(seg_vol, A, seg_header, inst / SEG_DIR / f"{stem}_nnUNetSeg.nii.gz", np.uint8)


# --------------------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------------------

PLOT_STYLE = dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7", blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", yellow="#eda100",
                  blue_ramp=("#cde2fb", "#0d366b"))
LABEL_RAW = "as acquired"
LABEL_ALIGNED = "after alignment"
BASAL_SLICES = 3
APICAL_SLICES = 3
SUBSETS = {
    "LAX_2Ch": ("LAX 2Ch view", lambda d: d["image"] == "LAX_2Ch"),
    "LAX_3Ch": ("LAX 3Ch view", lambda d: d["image"] == "LAX_3Ch"),
    "LAX_4Ch": ("LAX 4Ch view", lambda d: d["image"] == "LAX_4Ch"),
    "SAX_basal": (f"basal SAX slices (index < {BASAL_SLICES})", lambda d: (d["image"] == "SAX") & (d["region"] == "basal")),
    "SAX_mid": ("mid-ventricular SAX slices", lambda d: (d["image"] == "SAX") & (d["region"] == "mid")),
    "SAX_apical": (f"apical SAX slices (last {APICAL_SLICES})", lambda d: (d["image"] == "SAX") & (d["region"] == "apical")),
}


def load_results(summary_path: Path, detail_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    summ = pd.read_csv(summary_path, dtype={"subject_id": str}).drop_duplicates("subject_id", keep="last")
    summ = summ[summ["status"] == "ok"].reset_index(drop=True)
    det = pd.read_csv(detail_path, dtype={"subject_id": str})
    det = det.drop_duplicates(["subject_id", "image", "slice"], keep="last")
    for col in ("mi_zero", "mi", "hd_zero_mm", "hd_mm"):  # runs from before these metrics were added
        if col not in det:
            det[col] = np.nan
    det = det[det["subject_id"].isin(summ["subject_id"])].reset_index(drop=True)
    n_slices = det[det["image"] == "SAX"].groupby("subject_id")["slice"].max() + 1
    det["n_slices"] = det["subject_id"].map(n_slices)
    det["region"] = np.where(det["slice"] < BASAL_SLICES, "basal",
                             np.where(det["slice"] >= det["n_slices"] - APICAL_SLICES, "apical", "mid"))
    det = det[det["n_pairs"] > 0].reset_index(drop=True)
    return summ, det


def load_pairs(pairs_path: Path, det: pd.DataFrame) -> pd.DataFrame | None:
    """Per-line-pair rows of the subjects in `det`, with the SAX slice's region attached (None if absent)."""
    if not pairs_path.exists():
        return None
    pairs = pd.read_csv(pairs_path, dtype={"subject_id": str})
    pairs = pairs.drop_duplicates(["subject_id", "image_a", "slice_a", "image_b", "slice_b"], keep="last")
    pairs = pairs[pairs["subject_id"].isin(det["subject_id"])].reset_index(drop=True)
    regions = det[det["image"] == "SAX"].set_index(["subject_id", "slice"])["region"]
    key = pd.MultiIndex.from_arrays([pairs["subject_id"], pairs["slice_a"]])
    pairs["region"] = np.where(pairs["image_a"] == "SAX", regions.reindex(key).to_numpy(), "LAX")
    return pairs


def select_pairs(pairs: pd.DataFrame, subset: str) -> pd.DataFrame:
    """Pairs involving the planes of a SUBSETS entry (a LAX view, or a SAX region)."""
    if subset.startswith("LAX"):
        return pairs[(pairs["image_a"] == subset) | (pairs["image_b"] == subset)]
    return pairs[(pairs["image_a"] == "SAX") & (pairs["region"] == subset.split("_", 1)[1])]


def plot_pairs_figure(pairs: pd.DataFrame, label: str, out_path: Path, st=PLOT_STYLE) -> None:
    """One value per intersection-line pair: NCC / NMI / Dice / Hausdorff before and after, per pair type."""
    from matplotlib.figure import Figure

    kinds = [k for k in ("SAX-LAX", "LAX-LAX") if (pairs["kind"] == k).any()]
    if not kinds:
        return
    fig = Figure(figsize=(18, 4.2 * len(kinds) + 0.8), layout="constrained", facecolor=st["surface"])
    fig.get_layout_engine().set(rect=(0, 0, 1, 1 - 0.5 / (4.2 * len(kinds) + 0.8)))
    axes = np.atleast_2d(fig.subplots(len(kinds), 4))
    panels = [("ncc_zero", "ncc", "NCC", "NCC"), ("mi_zero", "mi", "Normalized MI", "normalized MI"),
              ("seg_dice_zero", "seg_dice", "Segmentation Dice", "Dice"),
              ("hd_zero_mm", "hd_mm", "Segmentation Hausdorff (mm)", "Hausdorff distance (mm)")]
    for r, kind in enumerate(kinds):
        sub = pairs[pairs["kind"] == kind]
        for c, (c0, c1, title, xl) in enumerate(panels):
            _hist_series(axes[r, c], [(sub[c0], LABEL_RAW, st["aqua"]), (sub[c1], LABEL_ALIGNED, st["orange"])],
                         f"{title}, {kind} ({len(sub)} pairs)", xlabel=xl)
    _legend(fig, axes[0, 0].get_legend_handles_labels())
    fig.suptitle(f"Per intersection-line pair, {label}  (n = {pairs['subject_id'].nunique()} subjects, {len(pairs)} pairs)",
                 color=st["ink"], fontsize=12, x=0.02, ha="left")
    fig.savefig(out_path, dpi=150)


def per_subject_metrics(det: pd.DataFrame, reference: pd.DataFrame | None = None) -> pd.DataFrame:
    """Subject-level metrics recomputed from a subset of per-slice rows.

    Jitter is the RMS deviation of the slice shifts from their mean. When the subset holds a
    single plane per subject, `reference` rows (e.g. all LAX views) define the mean instead.
    """

    def rms(x):
        x = x[np.isfinite(x)]
        return float(np.sqrt(np.mean(x ** 2))) if x.size else np.nan

    def nanstat(fn, x):
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        return float(fn(x)) if x.size else np.nan

    def jitter(vec, ref_vec=None):
        vec = vec[np.isfinite(vec).all(1)]
        if len(vec) >= 2:
            return rms(np.hypot(*(vec - vec.mean(0)).T))
        if len(vec) == 1 and ref_vec is not None:
            ref_vec = ref_vec[np.isfinite(ref_vec).all(1)]
            if len(ref_vec) >= 2:
                return float(np.hypot(*(vec[0] - ref_vec.mean(0))))
        return np.nan

    ref_groups = dict(tuple(reference.groupby("subject_id"))) if reference is not None else {}
    rows = {}
    for sid, g in det.groupby("subject_id"):
        v = g[["dx_mm", "dy_mm"]].values
        ref = ref_groups.get(sid)
        mag = np.hypot(v[:, 0], v[:, 1])
        rows[sid] = dict(
            n=len(g), shift_rms=rms(mag), shift_max=nanstat(np.max, mag), shift_mean=nanstat(np.mean, mag),
            shift_jitter=jitter(v, ref[["dx_mm", "dy_mm"]].values if ref is not None else None),
            dz_rms=rms(g["dz_mm"].values), rot_inplane_rms=rms(g["rot_inplane_deg"].values), tilt_rms=rms(g["rot_tilt_deg"].values),
            ncc_zero=nanstat(np.mean, g["ncc_zero"]), ncc=nanstat(np.mean, g["ncc"]),
            mi_zero=nanstat(np.mean, g["mi_zero"]), mi=nanstat(np.mean, g["mi"]),
            seg_dice_zero=nanstat(np.mean, g["seg_dice_zero"]), seg_dice=nanstat(np.mean, g["seg_dice"]),
            hd_zero=nanstat(np.max, g["hd_zero_mm"]), hd=nanstat(np.max, g["hd_mm"]),
        )
    m = pd.DataFrame.from_dict(rows, orient="index")
    m["ncc_gain"] = m["ncc"] - m["ncc_zero"]
    m["mi_gain"] = m["mi"] - m["mi_zero"]
    return m


def _style_axis(ax, st=PLOT_STYLE):
    from matplotlib.ticker import MaxNLocator
    ax.set_facecolor(st["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(st["axis"])
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=st["muted"], labelsize=8, length=3, width=0.8)
    ax.yaxis.grid(True, color=st["grid"], linewidth=0.6)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(st["ink2"])
    ax.yaxis.label.set_color(st["ink2"])
    ax.title.set_color(st["ink"])


def _clean(values) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    return v[np.isfinite(v)]


HIST_BINS = 50          # bins per histogram panel (--hist-bins)


def _hist_series(ax, series: list[tuple], title: str, xlabel: str = "", n_bins: int | None = None, st=PLOT_STYLE) -> bool:
    """Grouped-bar histogram: within each bin the series are drawn as narrow side-by-side bars."""
    series = [(_clean(v), label, color) for v, label, color in series]
    series = [s for s in series if s[0].size]
    if series:
        allv = np.concatenate([s[0] for s in series])
        lo, hi = float(allv.min()), float(allv.max())
        if hi - lo < 1e-9:
            lo, hi = lo - 0.5, hi + 0.5
        edges = np.linspace(lo, hi, (n_bins or HIST_BINS) + 1)
        width = edges[1] - edges[0]
        k = len(series)
        bw = 0.8 * width / k
        for i, (v, label, color) in enumerate(series):
            counts, _ = np.histogram(v, bins=edges)
            x = edges[:-1] + 0.1 * width + bw * (i + 0.5)
            ax.bar(x, counts, width=bw, color=color, linewidth=0, label=label)
            med = float(np.median(v))
            ax.axvline(med, color=color, linestyle=(0, (3, 2)), linewidth=1.2)
            ax.annotate(f"median {med:.2f}", xy=(med, 1), xycoords=("data", "axes fraction"),
                        xytext=(4, -4 - 11 * i), textcoords="offset points", ha="left", va="top",
                        fontsize=8, color=st["ink2"])
        ax.set_xlim(edges[0], edges[-1])
    else:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, color=st["muted"], fontsize=9)
    ax.set_title(title, fontsize=10, loc="left")
    ax.set_ylabel("count")
    if xlabel:
        ax.set_xlabel(xlabel)
    _style_axis(ax, st)
    return bool(series)


def _legend(fig, handles_labels, st=PLOT_STYLE):
    handles, labels = handles_labels
    if len(labels) > 1:
        leg = fig.legend(handles, labels, loc="upper right", ncol=len(labels), frameon=False, fontsize=9,
                         bbox_to_anchor=(0.99, 0.995))
        for text in leg.get_texts():
            text.set_color(st["ink2"])


def _inset_legend(ax, st=PLOT_STYLE):
    leg = ax.legend(frameon=False, fontsize=8, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2,
                    borderaxespad=0, handlelength=1.2)
    for text in leg.get_texts():
        text.set_color(st["ink2"])


def plot_metrics_figure(metrics: pd.DataFrame, label: str, out_path: Path, lax_metrics: pd.DataFrame | None = None,
                        jitter_title: str = "Inter-slice jitter (mm)", st=PLOT_STYLE):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 4, figsize=(18, 10.5), layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.95))
    one = lambda col, frame=metrics: [(frame[col], LABEL_ALIGNED, st["orange"])]  # noqa: E731
    two = lambda col0, col1: [(metrics[col0], LABEL_RAW, st["aqua"]), (metrics[col1], LABEL_ALIGNED, st["orange"])]  # noqa: E731
    _hist_series(axes[0, 0], one("shift_rms"), "In-plane shift RMS (mm)")
    _hist_series(axes[0, 1], one("shift_jitter"), jitter_title)
    _hist_series(axes[0, 2], one("shift_max"), "Max slice shift (mm)")
    if lax_metrics is not None:
        _hist_series(axes[0, 3], one("shift_mean", lax_metrics), "LAX mean in-plane shift (mm)")
    else:
        _hist_series(axes[0, 3], one("shift_mean"), "Mean in-plane shift (mm)")
    _hist_series(axes[1, 0], two("ncc_zero", "ncc"), "Mean NCC at slice intersections")
    _hist_series(axes[1, 1], one("ncc_gain"), "NCC gain over as-acquired")
    _hist_series(axes[1, 2], two("mi_zero", "mi"), "Mean normalized MI at slice intersections")
    _hist_series(axes[1, 3], one("mi_gain"), "Normalized MI gain over as-acquired")
    _hist_series(axes[2, 0], one("dz_rms"), "Through-plane shift RMS (mm)")
    if _hist_series(axes[2, 1], [(metrics["rot_inplane_rms"], "in-plane", st["orange"]),
                                 (metrics["tilt_rms"], "tilt (weak)", st["yellow"])], "Rotation RMS (deg)"):
        _inset_legend(axes[2, 1])
    _hist_series(axes[2, 2], two("seg_dice_zero", "seg_dice"), "Segmentation Dice at slice intersections")
    _hist_series(axes[2, 3], two("hd_zero", "hd"), "Worst segmentation Hausdorff (mm)")
    _legend(fig, axes[1, 0].get_legend_handles_labels())
    fig.suptitle(f"Subject-level misalignment metrics (one value per subject), {label}  (n = {len(metrics)} subjects)",
                 color=st["ink"], fontsize=12, x=0.02, ha="left")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_translation_figure(det: pd.DataFrame, label: str, out_path: Path, per_image: bool, st=PLOT_STYLE):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    fig = plt.figure(figsize=(17, 8.5), layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.94))
    gs = fig.add_gridspec(2, 4, height_ratios=(1, 1.3))
    if per_image:
        images = [im for im in ("SAX",) + LAX_VIEWS if (det["image"] == im).any()]
        for i, image in enumerate(images[:4]):
            sub = det[det["image"] == image]
            _hist_series(fig.add_subplot(gs[0, i]), [(sub["shift_inplane_mm"], LABEL_ALIGNED, st["orange"])],
                         f"{image} in-plane shift (mm), {len(sub)} slices", xlabel="translation distance (mm)")
    else:
        _hist_series(fig.add_subplot(gs[0, 0]), [(det["shift_inplane_mm"], LABEL_ALIGNED, st["orange"])],
                     f"In-plane shift (mm), {len(det)} slices", xlabel="translation distance (mm)")
        _hist_series(fig.add_subplot(gs[0, 1]), [(det["dz_mm"], LABEL_ALIGNED, st["orange"])],
                     "Through-plane shift (mm)", xlabel="signed translation (mm)")
        ax = fig.add_subplot(gs[0, 2])
        if _hist_series(ax, [(det["rot_inplane_deg"], "in-plane", st["orange"]), (det["rot_tilt_deg"], "tilt (weak)", st["yellow"])],
                        "Rotation (deg)", xlabel="degrees"):
            _inset_legend(ax)
        _hist_series(fig.add_subplot(gs[0, 3]), [(det["seg_dice_zero"], LABEL_RAW, st["aqua"]), (det["seg_dice"], LABEL_ALIGNED, st["orange"])],
                     "Segmentation Dice along intersections", xlabel="Dice")

    has_sax = bool((det["image"] == "SAX").any())
    hex_rows = det[det["image"] == "SAX"] if has_sax else det
    hex_name = "SAX" if has_sax else "LAX"
    ax = fig.add_subplot(gs[1, 0] if has_sax else gs[1, :])
    ok = np.isfinite(hex_rows["dx_mm"]) & np.isfinite(hex_rows["dy_mm"])
    lim = 1.0
    if ok.any():
        vals = np.abs(hex_rows.loc[ok, ["dx_mm", "dy_mm"]].values).ravel()
        lim = max(float(np.percentile(vals, 99.5)), 1.0)
        hb = ax.hexbin(hex_rows.loc[ok, "dx_mm"], hex_rows.loc[ok, "dy_mm"], gridsize=35, extent=(-lim, lim, -lim, lim),
                       cmap=LinearSegmentedColormap.from_list("blue_ramp", st["blue_ramp"]), mincnt=1,
                       linewidths=0.2, edgecolors=st["surface"])
        cb = fig.colorbar(hb, cax=ax.inset_axes([1.04, 0.05, 0.04, 0.9]))
        cb.set_label("slices", color=st["ink2"])
        cb.ax.tick_params(colors=st["muted"], labelsize=8)
        cb.outline.set_edgecolor(st["axis"])
    ax.axhline(0, color=st["axis"], linewidth=0.8)
    ax.axvline(0, color=st["axis"], linewidth=0.8)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_box_aspect(1)
    ax.set_anchor("W")
    ax.set_xlabel("dx (mm, along image rows)")
    ax.set_ylabel("dy (mm, along image columns)")
    ax.set_title(f"{hex_name} in-plane correction vectors", fontsize=10, loc="left")
    _style_axis(ax, st)
    ax.yaxis.grid(False)

    sax = det[det["image"] == "SAX"]
    if len(sax):
        ax = fig.add_subplot(gs[1, 1:])
        q = sax.groupby("slice")["shift_inplane_mm"].quantile([0.25, 0.5, 0.75]).unstack()
        ax.fill_between(q.index, q[0.25], q[0.75], color=st["orange"], alpha=0.15, linewidth=0)
        ax.plot(q.index, q[0.5], color=st["orange"], linewidth=2, marker="o", markersize=4, label=LABEL_ALIGNED)
        for z, c in sax.groupby("slice").size().items():
            ax.annotate(str(c), xy=(z, 0), xycoords=("data", "axes fraction"), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom", fontsize=7, color=st["muted"])
        ax.set_xticks(sorted(sax["slice"].unique()))
        ax.set_xlabel("SAX slice index (count of slices shown at baseline)")
        ax.set_ylabel("in-plane translation distance (mm)")
        ax.set_title("SAX shift by slice index (median, IQR band)", fontsize=10, loc="left")
        _style_axis(ax, st)
    _legend(fig, fig.axes[0].get_legend_handles_labels() if per_image else fig.axes[3].get_legend_handles_labels())
    fig.suptitle(f"Estimated breath-hold motion, {label}  (n = {det['subject_id'].nunique()} subjects)",
                 color=st["ink"], fontsize=12, x=0.02, ha="left")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_results(summary_path: Path, detail_path: Path, out_dir: Path, pairs_path: Path | None = None) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    st = PLOT_STYLE
    _, det = load_results(summary_path, detail_path)
    pairs = load_pairs(pairs_path, det) if pairs_path is not None else None
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "sans-serif", "figure.facecolor": st["surface"],
                         "savefig.facecolor": st["surface"]})
    written = []
    sax, lax = det[det["image"] == "SAX"], det[det["image"] != "SAX"]
    p = out_dir / "motion_metrics_hist.png"
    plot_metrics_figure(per_subject_metrics(sax), "all SAX slices", p,
                        lax_metrics=per_subject_metrics(lax) if len(lax) else None)
    written.append(p)
    p = out_dir / "motion_translations.png"
    plot_translation_figure(det, "all slices", p, per_image=True)
    written.append(p)
    if pairs is not None and len(pairs):
        p = out_dir / "motion_pairs_hist.png"
        plot_pairs_figure(pairs, "all pairs", p)
        written.append(p)
    for name, (label, select) in SUBSETS.items():
        sub = det[select(det)]
        if sub.empty:
            continue
        if pairs is not None:
            sub_pairs = select_pairs(pairs, name)
            if len(sub_pairs):
                p = out_dir / f"motion_pairs_hist_{name}.png"
                plot_pairs_figure(sub_pairs, f"pairs involving {label}", p)
                written.append(p)
        p = out_dir / f"motion_metrics_hist_{name}.png"
        if name.startswith("LAX"):
            plot_metrics_figure(per_subject_metrics(sub, reference=lax), label, p, jitter_title="Deviation from mean LAX shift (mm)")
        else:
            plot_metrics_figure(per_subject_metrics(sub), label, p)
        written.append(p)
        p = out_dir / f"motion_translations_{name}.png"
        plot_translation_figure(sub, label, p, per_image=False)
        written.append(p)
    return written


# --------------------------------------------------------------------------------------
# Subject-level outputs (thread-safe: matplotlib Figure objects, no pyplot)
# --------------------------------------------------------------------------------------

def mask_plot_data(planes: list[Plane], lines: list[tuple[int, int, Line]], extent_mm: float = 250.0,
                   step_mm: float = 1.0) -> dict:
    """CPU copies of what the mask overlay needs: LAX image/seg/mask and, per line in voxel coords, the full
    +-extent candidate segment, the part inside both images and the points used in the loss."""
    views = {}
    for i, p in enumerate(planes):
        if p.image == "SAX" or p.mask is None:
            continue
        views[i] = dict(name=p.image, img=p.img[0, 0].cpu().numpy(),
                        seg=(p.seg[:, 0].cpu().numpy() > 0).any(0) if p.seg is not None else None,
                        mask=p.mask[0, 0].cpu().numpy() > 0.5, pts=[], full=[], inimg=[])
    for ia, ib, line in lines:
        if line.x0 is None:
            continue
        w = line.w.cpu().numpy() > 0
        world = line.x0[None] + line.t.cpu().numpy()[w, None] * line.d[None]
        # full candidate line (+-extent) and the part of it inside both images
        t_all = np.arange(-extent_mm, extent_mm + 1e-6, step_mm)
        world_all = line.x0[None] + t_all[:, None] * line.d[None]
        pts_all = torch.tensor(world_all, dtype=torch.float32, device=planes[ia].img.device)
        inb = (inside(planes[ia], pts_all) & inside(planes[ib], pts_all)).cpu().numpy()
        kind = "SAX-LAX" if "SAX" in (planes[ia].image, planes[ib].image) else "LAX-LAX"
        for i in (ia, ib):
            if i in views:
                p = planes[i]
                to_vox = lambda xyz: (xyz @ p.inv_rot.cpu().numpy().T.astype(np.float64) + p.inv_trans.cpu().numpy())[:, :2]  # noqa: E731
                views[i]["pts"].append((to_vox(world), kind))
                views[i]["full"].append((to_vox(world_all[[0, -1]]), kind))
                if inb.any():
                    idx = np.flatnonzero(inb)
                    views[i]["inimg"].append((to_vox(world_all[[idx[0], idx[-1]]]), kind))
    return dict(views=list(views.values()))


LINE_COLUMNS = ["image_a", "slice_a", "image_b", "slice_b", "kind", "n_points",
                "x0_x", "x0_y", "x0_z", "d_x", "d_y", "d_z", "t_used_lo", "t_used_hi", "t_img_lo", "t_img_hi"]
MASK_CONTOUR_COLUMNS = ["image", "slice", "contour", "i", "j"]


def save_line_geometry(planes: list[Plane], lines: list[tuple[int, int, Line]], metrics_dir: Path,
                       extent_mm: float, step_mm: float) -> None:
    """lines.csv: the intersection lines (acquired geometry) with the parameter range inside both images and
    the range actually sampled; mask_contours.csv: 0.5-level contours of every dilated LAX mask (voxel i, j)."""
    from contourpy import contour_generator

    rows = []
    t_all = np.arange(-extent_mm, extent_mm + 1e-6, step_mm)
    for ia, ib, line in lines:
        if line.x0 is None:
            continue
        a, b = planes[ia], planes[ib]
        t_used = line.t.cpu().numpy()[line.w.cpu().numpy() > 0]
        pts_all = torch.tensor(line.x0[None] + t_all[:, None] * line.d[None], dtype=torch.float32, device=a.img.device)
        inb = np.flatnonzero((inside(a, pts_all) & inside(b, pts_all)).cpu().numpy())
        nan = float("nan")
        rows.append(dict(image_a=a.image, slice_a=a.slice_idx, image_b=b.image, slice_b=b.slice_idx,
                         kind="SAX-LAX" if "SAX" in (a.image, b.image) else "LAX-LAX", n_points=int(line.n_points),
                         x0_x=float(line.x0[0]), x0_y=float(line.x0[1]), x0_z=float(line.x0[2]),
                         d_x=float(line.d[0]), d_y=float(line.d[1]), d_z=float(line.d[2]),
                         t_used_lo=float(t_used.min()) if t_used.size else nan,
                         t_used_hi=float(t_used.max()) if t_used.size else nan,
                         t_img_lo=float(t_all[inb[0]]) if inb.size else nan,
                         t_img_hi=float(t_all[inb[-1]]) if inb.size else nan))
    pd.DataFrame(rows, columns=LINE_COLUMNS).to_csv(metrics_dir / "lines.csv", index=False, float_format="%.4f")

    crow = []
    for p in planes:
        if p.mask is None:
            continue
        mask = p.mask[0, 0].cpu().numpy() > 0.5                     # (H, W) = voxel axes (i, j)
        if not mask.any():
            continue
        # contourpy indexes z[row, col] = z[i, j] and returns (x, y) = (j, i)
        for c, seg in enumerate(contour_generator(z=mask.astype(float)).lines(0.5)):
            for x, y in seg:
                crow.append(dict(image=p.image, slice=p.slice_idx, contour=c, i=float(y), j=float(x)))
    pd.DataFrame(crow, columns=MASK_CONTOUR_COLUMNS).to_csv(metrics_dir / "mask_contours.csv", index=False,
                                                           float_format="%.3f")


def save_mask_plot(viz: dict, subject_id: str, dilation_mm: float, path: Path, extent_mm: float = 250.0,
                   st=PLOT_STYLE) -> None:
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    def matplotlib_rect(xy, w, h, color):
        return Rectangle(xy, w, h, fill=False, edgecolor=color, lw=0.8)

    views = viz["views"]
    if not views:
        return
    fig = Figure(figsize=(6 * len(views), 6.4), layout="constrained", facecolor=st["surface"])
    axes = np.atleast_1d(fig.subplots(1, len(views)))
    colors = {"SAX-LAX": st["blue"], "LAX-LAX": st["orange"]}
    for ax, v in zip(axes, views):
        H, W = v["img"].shape
        ax.set_facecolor("#000000")
        ax.imshow(v["img"].T, cmap="gray", origin="lower", extent=(-0.5, H - 0.5, -0.5, W - 0.5))
        ax.add_patch(matplotlib_rect((-0.5, -0.5), H, W, st["ink2"]))
        if v["seg"] is not None and v["seg"].any():
            ax.contour(v["seg"].T.astype(float), levels=[0.5], colors=st["aqua"], linewidths=1.5, origin="lower")
        ax.contour(v["mask"].T.astype(float), levels=[0.5], colors=st["yellow"], linewidths=1.5, linestyles="--", origin="lower")
        for vox, kind in v["full"]:
            ax.plot(vox[:, 0], vox[:, 1], color=colors[kind], lw=0.6, ls=(0, (4, 3)), alpha=0.45)
        for vox, kind in v["inimg"]:
            ax.plot(vox[:, 0], vox[:, 1], color=colors[kind], lw=0.8, alpha=0.8)
        for vox, kind in v["pts"]:
            ax.plot(vox[:, 0], vox[:, 1], ".", ms=1.6, color=colors[kind])
        ends = np.concatenate([vox for vox, _ in v["full"]] + [np.array([[0, 0], [H, W]])])
        pad = 5
        ax.set_xlim(ends[:, 0].min() - pad, ends[:, 0].max() + pad)
        ax.set_ylim(ends[:, 1].min() - pad, ends[:, 1].max() + pad)
        ax.set_aspect("equal")
        ax.set_title(v["name"], loc="left", fontsize=11, color=st["ink"])
        ax.set_xticks([])
        ax.set_yticks([])
    handles = [Line2D([], [], color=st["aqua"], lw=1.5, label="heart segmentation (union over frames)"),
               Line2D([], [], color=st["yellow"], lw=1.5, ls="--", label=f"dilated mask (+{dilation_mm:g} mm)"),
               Line2D([], [], color=st["blue"], lw=0.6, ls=(0, (4, 3)), label=f"candidate line (+-{extent_mm:g} mm)"),
               Line2D([], [], color=st["blue"], lw=0.8, label="part inside both images"),
               Line2D([], [], color=st["blue"], marker=".", ls="", label="points used: SAX-LAX"),
               Line2D([], [], color=st["orange"], marker=".", ls="", label="points used: LAX-LAX")]
    fig.legend(handles=handles, loc="outside lower center", ncol=6, frameon=False, fontsize=9)
    fig.suptitle(f"Subject {subject_id}: intersection-line points entering the NCC loss (frame 0, acquired geometry)",
                 x=0.01, ha="left", color=st["ink"])
    fig.savefig(path, dpi=130)


def save_subject_metrics(summary: dict, details: list[dict], pairs: list[dict], metrics_dir: Path, plots: bool,
                         history: dict | None = None, st=PLOT_STYLE) -> None:
    """Per-subject CSVs (summary row, per-plane rows, per-line-pair rows, pose/metric history) and figures."""
    metrics_dir.mkdir(parents=True, exist_ok=True)
    if history and history["planes"]:
        pd.DataFrame([fmt_row(r) for r in history["planes"]], columns=HISTORY_COLUMNS).to_csv(metrics_dir / "history.csv", index=False)
        pd.DataFrame([fmt_row(r) for r in history["steps"]], columns=HISTORY_STEP_COLUMNS).to_csv(
            metrics_dir / "history_summary.csv", index=False)
    pd.DataFrame([fmt_row(summary)], columns=SUMMARY_COLUMNS).to_csv(metrics_dir / "summary.csv", index=False)
    pd.DataFrame([fmt_row(r) for r in details], columns=DETAIL_COLUMNS).to_csv(metrics_dir / "slices.csv", index=False)
    pd.DataFrame([fmt_row(r) for r in pairs], columns=PAIR_COLUMNS).to_csv(metrics_dir / "pairs.csv", index=False)
    det = pd.DataFrame(details, columns=DETAIL_COLUMNS)
    det = det[det["n_pairs"] > 0]
    pdf = pd.DataFrame(pairs, columns=PAIR_COLUMNS)
    if not plots or det.empty:
        return
    if len(pdf):
        plot_pairs_figure(pdf, f"subject {summary['subject_id']}", metrics_dir / "motion_pairs.png")
    from matplotlib.figure import Figure
    from matplotlib.ticker import AutoLocator

    sax, lax = det[det["image"] == "SAX"], det[det["image"] != "SAX"]
    sid = summary["subject_id"]
    fig = Figure(figsize=(17, 12), layout="constrained", facecolor=st["surface"])
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.95))
    gs = fig.add_gridspec(3, 4)
    one = lambda frame, col: [(frame[col], LABEL_ALIGNED, st["orange"])]  # noqa: E731
    two = lambda col0, col1: [(det[col0], LABEL_RAW, st["aqua"]), (det[col1], LABEL_ALIGNED, st["orange"])]  # noqa: E731
    _hist_series(fig.add_subplot(gs[0, 0]), one(sax, "shift_inplane_mm"), f"SAX in-plane shift (mm), {len(sax)} slices",
                 xlabel="translation distance (mm)")
    _hist_series(fig.add_subplot(gs[0, 1]), one(sax, "dz_mm"), "SAX through-plane shift (mm)", xlabel="signed translation (mm)")
    ax = fig.add_subplot(gs[0, 2])
    if _hist_series(ax, [(det["rot_inplane_deg"], "in-plane", st["orange"]), (det["rot_tilt_deg"], "tilt (weak)", st["yellow"])],
                    "Rotation (deg)", xlabel="degrees, all planes"):
        _inset_legend(ax)
    _hist_series(fig.add_subplot(gs[0, 3]), one(lax, "shift_inplane_mm"), f"LAX in-plane shift (mm), {len(lax)} views",
                 xlabel="translation distance (mm)")
    two = lambda col0, col1: [(pdf[col0], LABEL_RAW, st["aqua"]), (pdf[col1], LABEL_ALIGNED, st["orange"])]  # noqa: E731
    _hist_series(fig.add_subplot(gs[1, 0]), two("ncc_zero", "ncc"), f"NCC per line pair ({len(pdf)} pairs)", xlabel="NCC")
    _hist_series(fig.add_subplot(gs[1, 1]), two("mi_zero", "mi"), "Normalized MI per line pair", xlabel="normalized MI")
    _hist_series(fig.add_subplot(gs[1, 2]), two("seg_dice_zero", "seg_dice"), "Segmentation Dice per line pair", xlabel="Dice")
    _hist_series(fig.add_subplot(gs[1, 3]), two("hd_zero_mm", "hd_mm"), "Segmentation Hausdorff per line pair (mm)",
                 xlabel="Hausdorff distance (mm)")

    ax = fig.add_subplot(gs[2, :])
    if len(sax):
        x = sax["slice"].to_numpy()
        bw = 0.27
        for k, (col, name, color) in enumerate([("dx_mm", "dx (rows)", st["blue"]), ("dy_mm", "dy (columns)", st["aqua"]),
                                                ("dz_mm", "dz (through-plane)", st["orange"])]):
            ax.bar(x + (k - 1) * bw, sax[col].to_numpy(), width=bw, color=color, linewidth=0, label=name)
        ax.axhline(0, color=st["axis"], linewidth=0.8)
        ax.set_xticks(sorted(x))
        ax.set_xlabel("SAX slice index")
        ax.set_ylabel("translation (mm)")
        ax.set_title("SAX translation per slice (relative to mean LAX pose)", fontsize=10, loc="left")
        _style_axis(ax, st)
        ax.yaxis.set_major_locator(AutoLocator())
        _inset_legend(ax)
    _legend(fig, fig.axes[4].get_legend_handles_labels())
    fig.suptitle(f"Subject {sid}: NCC {summary['ncc_zero_mean']:.3f} -> {summary['ncc_mean']:.3f}, "
                 f"NMI {summary['mi_zero_mean']:.3f} -> {summary['mi_mean']:.3f}, "
                 f"Dice {summary['seg_dice_zero_mean']:.3f} -> {summary['seg_dice_mean']:.3f}, "
                 f"worst HD {summary['hd_zero_max_mm']:.1f} -> {summary['hd_max_mm']:.1f} mm, "
                 f"SAX jitter {summary['sax_shift_jitter_mm']:.2f} mm", color=st["ink"], fontsize=12, x=0.02, ha="left")
    fig.savefig(metrics_dir / "motion_subject.png", dpi=130)


# --------------------------------------------------------------------------------------
# CSV / driver
# --------------------------------------------------------------------------------------

def open_csv(path: Path, columns: list[str]):
    new = not path.exists() or path.stat().st_size == 0
    if not new:
        with open(path, newline="") as f:
            header = next(csv.reader(f), [])
        if header != columns:
            sys.exit(f"{path} has a different column layout than this version of the script; "
                     f"point --csv at a new file or move the old one aside.")
    f = open(path, "a", newline="")
    writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
    if new:
        writer.writeheader()
    return f, writer


def fmt_row(row: dict) -> dict:
    out = {}
    for k, v in row.items():
        if isinstance(v, float):
            out[k] = "nan" if math.isnan(v) else f"{v:.4f}"
        else:
            out[k] = v
    return out


# arguments that only affect input/output/bookkeeping, not the result, and are therefore left out of run names
NAME_EXCLUDED = {"root", "out_root", "run_name", "csv", "no_save_images", "no_mask_plots", "no_subject_plots",
                 "history_every", "max_subjects", "skip_processed", "device", "workers", "hist_bins", "plot", "plot_dir"}


def write_run_config(out_dir: Path, args) -> None:
    """Record every resolved setting (defaults included) in run_config.json. If the run directory already holds a
    config with different result-affecting settings, warn and keep both (the new one gets a timestamp suffix)."""
    import json
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in sorted(vars(args).items())}
    config["command_line"] = " ".join(sys.argv)
    path = out_dir / "run_config.json"
    if path.exists():
        old = json.loads(path.read_text())
        changed = sorted(k for k in config if k not in NAME_EXCLUDED | {"command_line"} and old.get(k) != config[k])
        if not changed:
            return
        print(f"WARNING: {path} was written with different settings ({', '.join(changed)}); "
              f"results in this directory now mix configurations.")
        path = out_dir / f"run_config_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(config, indent=2))


def default_run_name(ap: argparse.ArgumentParser, args, argv: list[str]) -> str:
    """steps<N>-lr<x>, followed by every result-affecting option given on the command line (in --help order)."""
    def fmt(v):
        if isinstance(v, float):
            return f"{v:g}"
        return str(v).replace(",", "+").replace("/", "_").replace("\\", "_").replace(" ", "")

    given = {tok.split("=", 1)[0] for tok in argv if tok.startswith("--")}
    parts = [f"steps{args.steps}", f"lr{fmt(args.lr)}"]
    for action in ap._actions:
        if action.dest in NAME_EXCLUDED | {"steps", "lr", "help"} or not given.intersection(action.option_strings):
            continue
        name = short_option_name(action.option_strings[-1])
        value = getattr(args, action.dest)
        parts.append(name if action.nargs == 0 else f"{name}{fmt(value)}")
    name = "-".join(parts)
    if len(name) > MAX_RUN_NAME:                      # keep paths under the Windows 260-character limit
        import hashlib
        digest = hashlib.sha1(name.encode()).hexdigest()[:8]
        name = name[:MAX_RUN_NAME - 9].rstrip("-") + "-" + digest
    return name


MAX_RUN_NAME = 120
NAME_ABBREVIATIONS = {"schedule": "sched", "dilation": "dil", "start": "st", "sigma": "sig", "rotation": "rot",
                      "metric": "", "mask": "mask", "scale": "sc", "stages": "stg", "frac": "fr", "extent": "ext"}


def short_option_name(option: str) -> str:
    """'--mask-dilation-start-mm' -> 'maskdilst': drop unit suffixes, abbreviate words, join without dashes."""
    words = [w for w in option.lstrip("-").split("-") if w not in ("mm", "deg")]
    return "".join(NAME_ABBREVIATIONS.get(w, w) for w in words)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path(r"C:\Users\Nil\Documents\data\ukbb"), help="Directory containing one folder per subject")
    ap.add_argument("--out-root", type=Path, default=Path(r"C:\Users\Nil\Documents\data\ukbb_alignment"), help="Parent of the run directory")
    ap.add_argument("--run-name", type=str, default=None, help="Run directory name (default: built from steps / learning rates / schedule / blur)")
    ap.add_argument("--csv", type=Path, default=None, help="Overall summary CSV (default: <run dir>/overall_metrics/motion_summary.csv)")
    ap.add_argument("--no-save-images", action="store_true", help="Do not write the motion-corrected NIfTI files")
    ap.add_argument("--no-mask-plots", action="store_true", help="Do not write the per-subject mask / line overlay figure")
    ap.add_argument("--no-subject-plots", action="store_true", help="Do not write the per-subject metrics figure (CSVs are still written)")
    ap.add_argument("--history-every", type=float, default=0.02,
                    help="Store poses, affines and metrics every this fraction of the run, plus step 0 and the best pose "
                         "(per-subject history.csv / history_summary.csv; 0 = off)")
    ap.add_argument("--max-subjects", type=int, default=3, help="Max number of subjects to evaluate (-1 = all)")
    ap.add_argument("--skip-processed", action="store_true", help="Skip subject IDs already present in the CSV")
    ap.add_argument("--instance", type=str, default="Instance_2", help="Instance subfolder inside each subject")
    ap.add_argument("--n-frames", type=int, default=5, help="Number of cardiac phases (evenly spaced) to use")
    ap.add_argument("--steps", type=int, default=600, help="Adam steps")
    ap.add_argument("--lr", type=float, default=0.05, help="Adam learning rate for translations (mm per step)")
    ap.add_argument("--rot-lr", type=float, default=0.01, help="Adam learning rate for rotations (degrees per step)")
    ap.add_argument("--lr-schedule", choices=["linear", "constant"], default="linear",
                    help="linear: decay each learning rate linearly from its start value to its end value; constant: no decay")
    ap.add_argument("--lr-end", type=float, default=0.0, help="Final translation learning rate for the linear schedule")
    ap.add_argument("--rot-lr-end", type=float, default=0.0, help="Final rotation learning rate for the linear schedule")
    ap.add_argument("--blur-sigma", type=float, default=6.0,
                    help="Coarse-to-fine: initial in-plane Gaussian sigma (mm) of the images, decayed linearly to 0 (0 = off)")
    ap.add_argument("--blur-frac", type=float, default=0.6, help="Fraction of the steps over which the blur decays to 0")
    ap.add_argument("--pair-keep", type=float, default=1.0,
                    help="Stochastic optimization: fraction of intersection lines kept at each step (1 = deterministic)")
    ap.add_argument("--eval-every", type=int, default=10, help="With --pair-keep < 1: evaluate the full objective every N steps to track the best pose")
    ap.add_argument("--seed", type=int, default=0, help="Seed for the line dropping")
    ap.add_argument("--dof", type=int, choices=[3, 6], default=6, help="3 = translation only, 6 = rigid (translation + rotation)")
    ap.add_argument("--gauge", type=float, default=1e-2, help="Penalty weight on the mean pose (removes global drift)")
    ap.add_argument("--prior", type=float, default=0.0, help="Penalty weight on rotation (deg^2) and through-plane shift (mm^2)")
    ap.add_argument("--seg-labels", type=str, default="1,2,3,4,5", help="Labels compared along LAX-LAX intersections")
    ap.add_argument("--sax-seg-labels", type=str, default="1,2,3", help="Labels compared along SAX-LAX intersections")
    ap.add_argument("--hd-miss-mm", type=float, default=100.0,
                    help="Hausdorff value (mm) when both planes contain a label but only one intersection profile hits it")
    ap.add_argument("--mask-dilation-mm", type=float, default=25.0,
                    help="How far (mm) the LAX heart segmentation is grown to form the mask; only intersection-line points "
                         "inside this mask enter the loss. This is also the final value when mask annealing is on. "
                         "Default: %(default)s")
    ap.add_argument("--mask-dilation-start-mm", type=float, default=None,
                    help="Mask annealing (off unless given). The dilation (mm) used at the start of the run; it then "
                         "shrinks linearly to --mask-dilation-mm over the first --mask-anneal-frac of the run. "
                         "Example: '--mask-dilation-start-mm 40 --mask-dilation-mm 25' starts with a 40 mm mask and "
                         "ends at 25 mm.")
    ap.add_argument("--mask-anneal-frac", type=float, default=0.8,
                    help="Fraction of the run (0-1) over which the mask dilation goes from --mask-dilation-start-mm to "
                         "--mask-dilation-mm. Default: %(default)s")
    ap.add_argument("--line-scale-start", type=float, default=1.0,
                    help="Line annealing (off at 1). A multiplier applied at the start of the run to BOTH "
                         "--line-extent-mm and --line-step-mm; it shrinks linearly back to 1 over the first "
                         "--line-anneal-frac of the run. Because both are scaled together, the number of samples per line "
                         "stays the same: the line starts long and coarsely sampled and ends at its normal length with "
                         "normal spacing. Example: '--line-scale-start 3' with extent 200 and step 1 starts at +-600 mm "
                         "with 3 mm spacing and ends at +-200 mm with 1 mm spacing. Note that the mask still clips the "
                         "line, so a longer line only matters where the mask reaches. Default: %(default)s")
    ap.add_argument("--line-anneal-frac", type=float, default=0.8,
                    help="Fraction of the run (0-1) over which the line scale goes from --line-scale-start to 1. "
                         "Default: %(default)s")
    ap.add_argument("--anneal-stages", type=int, default=6,
                    help="The annealing does not change every step: each schedule (mask, line) advances this many times "
                         "along its linear ramp, and the lines are rebuilt whenever either advances. Once both have "
                         "finished, the final values are used for the rest of the run and the best pose is chosen only "
                         "from that part. Default: %(default)s")
    ap.add_argument("--line-step-mm", type=float, default=1.0,
                    help="Spacing (mm) between samples along each intersection line (final value when "
                         "--line-scale-start is used). Default: %(default)s")
    ap.add_argument("--line-extent-mm", type=float, default=100.0,
                    help="Half-length (mm) of each intersection line: samples run from -extent to +extent around a "
                         "point near the plane centre, then are clipped to the images and the mask (final value when "
                         "--line-scale-start is used). Default: %(default)s")
    ap.add_argument("--min-line-mm", type=float, default=80.0,
                    help="An intersection line is dropped if fewer than this many mm of it lie inside both images and "
                         "the mask. Default: %(default)s")
    ap.add_argument("--sax-thickness", type=float, default=8.0,
                    help="SAX slice thickness in mm. The NIfTI header only stores the slice spacing (10 mm = 8 mm slice "
                         "+ 2 mm gap in the UK Biobank protocol). Default: %(default)s")
    ap.add_argument("--lax-thickness", type=float, default=6.0,
                    help="LAX slice thickness in mm (UK Biobank protocol: 6 mm; the header's 7.5 mm is the spacing). "
                         "Default: %(default)s")
    ap.add_argument("--n-band", type=int, default=8,
                    help="PSF approximation: when sampling one image along an intersection line, this many points are "
                         "taken across the other slice's through-plane profile (see --psf-model) and combined with the "
                         "profile as weights; the band is widened by 1/sin of the crossing angle. 1 = sample the line "
                         "only. Default: %(default)s")
    ap.add_argument("--psf-model", choices=["pulse", "gaussian", "rect"], default="pulse",
                    help="Shape of a slice's through-plane excitation profile, with FWHM = the nominal slice thickness. "
                         "'pulse': the profile of a slice-selective RF pulse of finite time-bandwidth product "
                         "(--psf-tbw) -- a rectangle of the nominal thickness with smoothed edges, sampled out to where "
                         "the profile vanishes; 'gaussian': a Gaussian (sampled to +-2 sigma); 'rect': an ideal box. "
                         "Default: %(default)s")
    ap.add_argument("--psf-tbw", type=float, default=2.5,
                    help="Time-bandwidth product of the slice-selective RF pulse for --psf-model pulse. It sets the "
                         "edge sharpness: the 10-90%% transition width is about thickness / TBW. Short bSSFP pulses "
                         "(TR ~2.7 ms as in UK Biobank) have TBW of roughly 2-3; the exact vendor pulse is not "
                         "published. Default: %(default)s")
    ap.add_argument("--metric", choices=["ncc", "ncc_grad", "mi"], default="ncc",
                    help="Optimized similarity: NCC on intensities, NCC on their 1D gradient, or normalized mutual information")
    ap.add_argument("--mi-bins", type=int, default=24, help="Histogram bins for mutual information")
    ap.add_argument("--mi-sigma", type=float, default=1.0, help="Parzen window width for mutual information, in bins")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--workers", type=int, default=4, help="Background threads for loading and writing NIfTI files")
    ap.add_argument("--hist-bins", type=int, default=25, help="Number of bins in every histogram panel")
    ap.add_argument("--plot", action="store_true", help="Only plot distributions from the existing CSVs, then exit")
    ap.add_argument("--plot-dir", type=Path, default=None, help="Where to write the overall figures (default: <run dir>/overall_metrics)")
    args = ap.parse_args()
    global HIST_BINS
    HIST_BINS = args.hist_bins

    run_dir = args.out_root / (args.run_name or default_run_name(ap, args, sys.argv[1:]))
    overall_dir = run_dir / "overall_metrics"
    summary_path = args.csv or (overall_dir / "motion_summary.csv")
    detail_path = summary_path.with_name(summary_path.stem + "_slices.csv")
    pairs_path = summary_path.with_name(summary_path.stem + "_pairs.csv")
    plot_dir = args.plot_dir or summary_path.parent
    if args.plot:
        for p in plot_results(summary_path, detail_path, plot_dir, pairs_path):
            print(f"Wrote {p}")
        return

    device = torch.device(args.device)
    subjects = sorted(d for d in args.root.iterdir() if d.is_dir())
    if args.skip_processed and summary_path.exists() and summary_path.stat().st_size > 0:
        done = set(pd.read_csv(summary_path, usecols=["subject_id"], dtype=str)["subject_id"])
        subjects = [s for s in subjects if s.name not in done]
        print(f"Skipping {len(done)} already processed subjects")
    if args.max_subjects >= 0:
        subjects = subjects[:args.max_subjects]
    print(f"Evaluating {len(subjects)} subjects on {device}; outputs in {run_dir}")
    if not subjects:
        return
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    write_run_config(summary_path.parent, args)

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **k: x  # noqa: E731

    sf, swriter = open_csv(summary_path, SUMMARY_COLUMNS)
    df_, dwriter = open_csv(detail_path, DETAIL_COLUMNS)
    pf, pwriter = open_csv(pairs_path, PAIR_COLUMNS)
    save_jobs = deque()                                   # (subject, description, future) for background writes
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            it = iter(subjects)
            queue = deque((s, ex.submit(load_subject, s, args)) for s in islice(it, 2 * args.workers))
            pbar = tqdm(total=len(subjects), unit="subj")
            while queue:
                subj, fut = queue.popleft()
                nxt = next(it, None)
                if nxt is not None:
                    queue.append((nxt, ex.submit(load_subject, nxt, args)))
                t0 = time.time()
                metrics_dir = run_dir / subj.name / "metrics"
                details, pairs, history = [], [], None

                def on_lines(planes, lines, sid=subj.name, out=metrics_dir):
                    if not lines:
                        return
                    out.mkdir(parents=True, exist_ok=True)
                    save_line_geometry(planes, lines, out, args.line_extent_mm, args.line_step_mm)
                    if args.no_mask_plots:
                        return
                    viz = mask_plot_data(planes, lines, args.line_extent_mm, args.line_step_mm)
                    save_jobs.append((sid, "mask plot", ex.submit(save_mask_plot, viz, sid, args.mask_dilation_mm,
                                                                  out / "mask_overlay.png", args.line_extent_mm)))

                try:
                    data = fut.result()
                    if data["status"] != "ok":
                        summary = empty_summary(subj.name, data["status"])
                    else:
                        summary, details, pairs, poses, history = process_subject(data, args, device, on_lines=on_lines)
                        if not args.no_save_images:
                            save_jobs.append((subj.name, "aligned NIfTIs", ex.submit(save_aligned, data, poses, run_dir, args)))
                except Exception as e:  # noqa: BLE001
                    traceback.print_exc()
                    summary = empty_summary(subj.name, f"error: {type(e).__name__}: {e}"[:200])
                summary["seconds"] = round(time.time() - t0, 2)
                summary["timestamp"] = datetime.now().isoformat(timespec="seconds")
                if details:
                    save_jobs.append((subj.name, "subject metrics", ex.submit(save_subject_metrics, dict(summary), details,
                                                                             pairs, metrics_dir, not args.no_subject_plots,
                                                                             history)))
                while len(save_jobs) > 3 * args.workers:
                    name, what, job = save_jobs.popleft()
                    try:
                        job.result()
                    except Exception:  # noqa: BLE001
                        print(f"Writing {what} failed for {name}:")
                        traceback.print_exc()
                swriter.writerow(fmt_row(summary))
                for row in details:
                    dwriter.writerow(fmt_row(row))
                for row in pairs:
                    pwriter.writerow(fmt_row(row))
                sf.flush()
                df_.flush()
                pf.flush()
                if hasattr(pbar, "set_postfix"):
                    pbar.set_postfix(subj=subj.name, status=summary["status"][:12],
                                     ncc=f"{summary['ncc_mean']:.2f}", jitter=f"{summary['sax_shift_jitter_mm']:.1f}")
                    pbar.update(1)
            for name, what, job in save_jobs:
                try:
                    job.result()
                except Exception:  # noqa: BLE001
                    print(f"Writing {what} failed for {name}:")
                    traceback.print_exc()
    finally:
        sf.close()
        df_.close()
        pf.close()
    print(f"\nSummary written to {summary_path}\nPer-slice results written to {detail_path}\nPer-pair results written to {pairs_path}")
    try:
        n = len(plot_results(summary_path, detail_path, plot_dir, pairs_path))
        print(f"{n} overall figures written to {plot_dir}")
    except Exception:  # noqa: BLE001
        print("Overall plotting failed:")
        traceback.print_exc()
    print(f"Per-subject outputs under {run_dir}\\<subject>\\ (metrics\\, and {args.instance}\\ unless --no-save-images)")


if __name__ == "__main__":
    main()
