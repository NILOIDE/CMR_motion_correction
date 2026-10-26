"""
Brute-force (grid-search) inter-slice motion estimation for UKBB cine MRI (SAX slices + LAX views).

Same input layout, similarity ingredients (intersection lines, PSF band, dilated LAX heart mask, --metric) and outputs
as motion_gradient.py, whose loading, metric and writing code is reused. Instead of optimizing all poses jointly, the
planes are placed one at a time, each by an exhaustive batched grid search over its own pose with the planes placed
before it held fixed:

  0. LAX_4Ch is fixed: it defines the frame (no gauge term needed).
  1. LAX_3Ch, translations only, against 4Ch. Their single line runs near the long axis, so rotating 3Ch about it
     would not change the line: rotations are not observable from this line and are left at 0 here.
  2. The mid-ventricular SAX slice (the one with the most in-mask line points, or --mid-slice) against 4Ch and 3Ch.
  3. LAX_2Ch against 4Ch, 3Ch and the mid slice.
  4. Every SAX slice (the mid slice again, now with all three views) against the LAX views, outward from the mid
     slice; with --sax-overlap-weight also against its already placed SAX neighbours (see below).
  5. (--lax-refine) LAX_3Ch, then LAX_2Ch: a narrow grid (--refine-*) against all other planes.
Missing views are skipped (the first present of 4Ch, 3Ch, 2Ch is the fixed reference).

Grid: offsets of the plane's pose in its own frame -- translations along its row axis, column axis and normal
(+-t-range-mm / +-dz-range-mm) and, with --dof 6, rotations about those three axes (+-rot-range-deg) -- centred on its
current pose. Rotations are about the --rotation-pivot point, so with the default long-axis pivot they turn the plane
about the heart and do not move it.

Coarse-to-fine: the whole box is first searched at --coarse-t-step-mm / --coarse-rot-step-deg; the --top-k best
candidates that are not grid neighbours of a better one (non-maximum suppression, so they are distinct basins) are
kept, and each is re-searched on a local grid of +-1 previous step at half the step; this repeats until the steps
reach --t-step-mm / --rot-step-deg. With --coarse-blur-mm the images are blurred in-plane at the first level (sigma
halving per level, none at the last), so the coarse grid does not step over narrow peaks. The final choice is made at
the last level, unblurred, among its candidates and the current pose (so a search never ends worse than it started).
Coarse steps equal to the final ones = one exhaustive grid.

Cost per candidate: each line is sampled with the PSF band only over the stretch where at least one candidate of the
batch has valid points (computed first, from one cheap mask lookup per point); points outside it weigh 0 in every
similarity, so this changes nothing but the time. --n-frames, --n-band and --fp16 (half-precision image sampling)
trade accuracy for speed.

Score of a candidate pose: for every line between the moving plane and a fixed plane, the similarity (--metric) of the
two PSF-band profiles over the scored points, averaged over the cardiac frames; the lines are weighted by their
in-mask point count at the acquired geometry, which stays fixed. The scored points are recomputed per candidate
(--score-region): mask = inside both images and inside the dilated heart mask of a LAX plane involved (which moves
with its plane); full = inside both images; centred = inside both images within --score-half-length-mm of the point
nearest both pivots (on the long axis, i.e. the heart), the same stretch for every candidate. A line with fewer than
--min-line-mm of scored points gets the floor score (0 for the correlations / MI, -1 for L1 / L2), so moving a line
off the scored region is never rewarded.

Metrics (--metric):
  ncc       normalized cross-correlation over the scored points. = 1 - mean squared difference of the z-scored
            profiles / 2, z-scored over the scored stretch: a single high-contrast edge can carry most of the variance
            and hence the score, and mismatches in flatter parts then cost little.
  ncc_grad  NCC of the profiles' 1D gradients (like ncc, dominated by the strongest edges).
  ngf       normalized gradient agreement: the gradients along the line are normalized per point, g / sqrt(g^2 +
            eps^2) (eps = --grad-eps x the gradient's standard deviation over the scored stretch), so every edge counts
            about equally whatever its contrast, and the score is sum(na nb) / sum((na^2 + nb^2) / 2) in [-1, 1]:
            1 = the same edges at the same places with the same sign, 0 for edges present in only one image, -1 for
            opposite edges (dark-to-bright in one, bright-to-dark in the other). Insensitive to absolute intensities.
  lncc      local NCC: NCC in windows of --lncc-window-mm sliding along the line, averaged over the windows (weighted
            by their scored points; windows where the profiles hardly vary fade out, --lncc-min-std). Every stretch has
            to match on its own: a mismatched stretch costs its share of the windows, one edge cannot dominate.
  l1        -mean |a - b| of the intensities, every image rescaled so its 1st / 99th percentiles map to 0 / 1 (all SAX
            slices with one scale). Black against white costs the most, in proportion to its length.
  l2        as l1 with squared differences (normalized in motion_gradient.build_planes).
  nmi        normalized mutual information (samples pooled over frames). Among the candidates within --tie-tol of the best score the one closest to the current pose wins
(1 deg counting as 0.52 mm, the arc at 30 mm from the pivot): flat directions (dz of mid-ventricular slices) do not
wander.

SAX-SAX overlap (--sax-overlap-weight > 0, step 4): two SAX slabs (--sax-thickness) share tissue where their planes
come closer than one slab thickness -- where tilted slices cross (typically away from the heart) or where through-plane
shifts close the gap. On a --sax-overlap-step-mm grid of the moving slice (optionally within --sax-overlap-radius-mm of
its pivot) each point gets the overlap fraction of the two slabs, max(0, 1 - |distance to the neighbour plane| /
thickness), the neighbour is sampled at the point's projection onto its plane, and the cost
weight x sum(overlap) x (1 - weighted NCC) / (sum of the line weights) is subtracted from the score: overlap never pays,
a mismatched one costs most. 0 for the acquired stack (10 mm spacing, 8 mm slices).

Outputs: as motion_gradient.py (aligned NIfTIs; with --logging the per-subject / overall CSVs and figures; history.csv
holds one snapshot per stage, step k = after stage k), plus with --logging <subject>/metrics/search.csv and
overall_metrics/motion_search.csv: one row per plane search (stage, references, candidates, score before/after, the
chosen offset in the plane's frame, seconds).
"""
import argparse
import functools
import math
import sys
import time
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from itertools import islice
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import motion_gradient as mg

VIEW_ORDER = ("LAX_4Ch", "LAX_3Ch", "LAX_2Ch")      # fixed reference, stage 1, stage 3
N_STAGES = 5
DEG_TO_MM = 30.0 * math.pi / 180.0                  # tie-break: 1 deg = the arc it moves a point 30 mm from the pivot
SCORE_FLOOR = {"ncc": 0.0, "ncc_grad": 0.0, "lncc": 0.0, "ngf": 0.0, "nmi": 0.0, "l1": -1.0, "l2": -1.0}   # too few points
OFFSET_COLUMNS = ["d_row_mm", "d_col_mm", "d_normal_mm", "rot_row_deg", "rot_col_deg", "rot_normal_deg"]
SEARCH_COLUMNS = (["subject_id", "stage", "image", "slice", "references", "n_candidates", "n_levels", "score_start", "score"]
                  + OFFSET_COLUMNS + ["seconds"])


@dataclass
class Level:
    """One coarse-to-fine level: a regular grid of pose offsets in a plane's own frame (row axis, column axis, normal),
    +-t_range (in-plane) / +-dz_range (normal) at t_step mm and +-rot_range about each axis at rot_step deg (rot_range
    0 = translations only), sampled on images blurred by blur_mm."""
    t_step: float
    rot_step: float
    t_range: float
    dz_range: float
    rot_range: float
    blur_mm: float = 0.0

    def offsets(self, device) -> torch.Tensor:
        """(P, 6) offsets [row, col, normal (mm), rot_row, rot_col, rot_normal (deg)]; always contains 0."""
        def axis(rng, step):
            n = int(math.floor(rng / step + 1e-9)) if rng > 0 and step > 0 else 0
            return torch.arange(-n, n + 1, device=device, dtype=torch.float32) * step

        tt, tz, rr = axis(self.t_range, self.t_step), axis(self.dz_range, self.t_step), axis(self.rot_range, self.rot_step)
        g = torch.meshgrid(tt, tt, tz, rr, rr, rr, indexing="ij")
        return torch.stack([x.reshape(-1) for x in g], 1)


@dataclass
class GridSpec:
    """The search box around a plane's current pose (+-t_range in-plane, +-dz_range along the normal, +-rot_range about
    each axis; rot_range 0 = translations only), the final steps t_step / rot_step and the coarse-to-fine schedule:
    the first level covers the box at coarse_t_step / coarse_rot_step (None = the final steps, i.e. a single
    exhaustive level) on images blurred by blur_mm; every further level halves the steps (not below the final ones) and
    searches +-1 previous step around each survivor."""
    t_range: float
    t_step: float
    dz_range: float
    rot_range: float = 0.0
    rot_step: float = 1.0
    coarse_t_step: float | None = None
    coarse_rot_step: float | None = None
    blur_mm: float = 0.0

    def levels(self) -> list[Level]:
        ts = max(self.t_step, self.coarse_t_step or self.t_step)
        rs = max(self.rot_step, self.coarse_rot_step or self.rot_step)
        out = [Level(ts, rs, self.t_range, self.dz_range, self.rot_range)]
        while ts > self.t_step * (1 + 1e-6) or (self.rot_range > 0 and rs > self.rot_step * (1 + 1e-6)):
            nts, nrs = max(self.t_step, ts / 2), max(self.rot_step, rs / 2)
            out.append(Level(nts, nrs, ts, ts, rs if self.rot_range > 0 else 0.0))
            ts, rs = nts, nrs
        for k, lv in enumerate(out[:-1]):
            lv.blur_mm = self.blur_mm / 2 ** k
        return out

    def clip(self, offs: torch.Tensor) -> torch.Tensor:
        """The offsets (P, 6) inside the search box."""
        eps = 1e-4
        keep = ((offs[:, :2].abs() <= self.t_range + eps).all(1) & (offs[:, 2].abs() <= self.dz_range + eps)
                & (offs[:, 3:].abs() <= self.rot_range + eps).all(1))
        return offs[keep]


@dataclass
class Posed:
    """A batch of P poses of one plane: rotation R (P, 3, 3), posed pivot Cp, normal Np and origin Op (P, 3)."""
    R: torch.Tensor
    Cp: torch.Tensor
    Np: torch.Tensor
    Op: torch.Tensor

    def expand(self, P: int) -> "Posed":
        return Posed(self.R.expand(P, 3, 3), self.Cp.expand(P, 3), self.Np.expand(P, 3), self.Op.expand(P, 3))


class GridSearcher:
    """Current poses of all planes and the batched score of candidate poses of one plane against fixed others.

    Poses are motion_gradient's: rotation vector theta (deg, world) about the plane's rot_centre and translation t (mm,
    world) of that point. `pair_weight` maps frozenset({i, j}) of two planes with an intersection line (acquired
    geometry) to the line's weight (its in-mask point count); `band` = (offsets u, weights) of the PSF band.
    """

    def __init__(self, planes: list[mg.Plane], pair_weight: dict, band: tuple[torch.Tensor, torch.Tensor], args, device):
        f32 = dict(dtype=torch.float32, device=device)
        self.planes, self.args, self.device = planes, args, device
        self.N = len(planes)
        self.pair_weight = pair_weight
        self.O = torch.tensor(np.stack([p.origin for p in planes]), **f32)
        self.C = torch.tensor(np.stack([p.rot_centre for p in planes]), **f32)
        self.Nrm = torch.tensor(np.stack([p.normal for p in planes]), **f32)
        self.frames = torch.tensor(np.stack([np.stack([p.u_hat, p.v_hat, p.normal]) for p in planes]), **f32)  # (N, 3, 3) rows
        self.thick = [float(p.thickness) for p in planes]
        self.dtype = torch.float16 if args.fp16 else torch.float32                  # of the sampled image buffers
        self.raw = [p.img.transpose(0, 1).contiguous() for p in planes]                                 # (1, F, H, W)
        if args.metric == "l1":                       # common intensity scale (--metric l2 is normalized in build_planes)
            groups = ([[i for i, p in enumerate(planes) if p.image == "SAX"]]
                      + [[i] for i, p in enumerate(planes) if p.image != "SAX"])
            for g in filter(None, groups):            # all SAX slices with one scale, each LAX view with its own
                lo, hi = np.percentile(torch.cat([self.raw[i].flatten() for i in g]).cpu().numpy(), [1.0, 99.0])
                for i in g:
                    self.raw[i] = (self.raw[i] - float(lo)) / max(float(hi - lo), 1e-6)
        self._blurred = {0.0: [x.to(self.dtype) for x in self.raw]}
        self.blur_mm = 0.0
        self.img = self._blurred[0.0]
        self.mask = [p.mask.to(self.dtype) if p.mask is not None else None for p in planes]           # (1, 1, H, W)
        self.theta = torch.zeros(self.N, 3, **f32)
        self.t = torch.zeros(self.N, 3, **f32)
        self.T = torch.arange(-args.line_extent_mm, args.line_extent_mm + 1e-6, args.line_step_mm, **f32)
        self.u_band, self.w_band = band
        self.min_points = args.min_line_mm / args.line_step_mm
        self.floor = SCORE_FLOOR[args.metric]
        self.batch = max(1, args.eval_batch)
        self._overlap_grids = {}

    # ---- geometry ----------------------------------------------------------------------------------------------

    def posed(self, i: int, theta: torch.Tensor, t: torch.Tensor) -> Posed:
        """Plane i at the poses theta, t (P, 3)."""
        R = mg.rodrigues(theta * (math.pi / 180.0))
        Cp = self.C[i] + t
        return Posed(R, Cp, R @ self.Nrm[i], R @ (self.O[i] - self.C[i]) + Cp)

    def current(self, i: int, P: int) -> Posed:
        """Plane i at its current pose, broadcast to a batch of P."""
        return self.posed(i, self.theta[i:i + 1], self.t[i:i + 1]).expand(P)

    def to_voxel(self, i: int, pose: Posed, X: torch.Tensor) -> torch.Tensor:
        """World points X (P, n, 3) -> voxel coordinates (P, n, 2) of plane i at the poses (undo the pose, then the
        acquired world -> voxel map)."""
        local = torch.einsum("pnj,pjk->pnk", X - pose.Cp[:, None, :], pose.R) + self.C[i]          # R^T (x - Cp) + C
        p = self.planes[i]
        return (local @ p.inv_rot.T + p.inv_trans)[..., :2]

    def inside(self, i: int, vox: torch.Tensor) -> torch.Tensor:
        p = self.planes[i]
        return (vox[..., 0] >= 0) & (vox[..., 0] <= p.H - 1) & (vox[..., 1] >= 0) & (vox[..., 1] <= p.W - 1)

    @staticmethod
    def sample(src: torch.Tensor, vox: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
        """src (1, C, H, W) at voxel coordinates vox (P, n, 2) -> (P, C, n), float32 (sampled in src's dtype)."""
        H, W = src.shape[-2:]
        grid = torch.stack([2 * vox[..., 1] / (W - 1) - 1, 2 * vox[..., 0] / (H - 1) - 1], -1)[None].to(src.dtype)
        return F.grid_sample(src, grid, mode=mode, padding_mode="zeros", align_corners=True)[0].transpose(0, 1).float()

    def set_blur(self, sigma_mm: float) -> None:
        """Sample in-plane Gaussian-blurred images (sigma in mm; 0 = the originals) from now on; cached per sigma."""
        if sigma_mm not in self._blurred:
            out = []
            for p, img in zip(self.planes, self.raw):
                x = img
                for axis, spacing in enumerate(p.spacing):                       # image axis 0 (H), then 1 (W)
                    s = max(sigma_mm / spacing, 1e-3)
                    r = int(math.ceil(3.0 * s))
                    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=img.device, dtype=img.dtype) / s) ** 2)
                    k = (k / k.sum()).repeat(img.shape[1], 1)[:, None]                    # (F, 1, 2r + 1)
                    pad = (0, 0, r, r) if axis == 0 else (r, r, 0, 0)
                    weight = k[..., None] if axis == 0 else k[:, :, None, :]
                    x = F.conv2d(F.pad(x, pad, mode="replicate"), weight, groups=img.shape[1])
                out.append(x.to(self.dtype))
            self._blurred[sigma_mm] = out
        self.blur_mm, self.img = sigma_mm, self._blurred[sigma_mm]

    # ---- score terms -------------------------------------------------------------------------------------------

    def similarity(self, a: torch.Tensor, b: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """--metric of profiles a, b (P, F, L) with point weights w (P, L) -> (P,), frames averaged (MI: pooled)."""
        args = self.args
        if args.metric in ("ncc", "ncc_grad"):
            return mg.ncc(a, b, w[:, None, :], args.metric == "ncc_grad").mean(1)
        if args.metric == "lncc":
            loc, wt = self.local_ncc(a, b, w)
            return ((loc * wt).sum(-1) / wt.sum(-1).clamp_min(1e-6)).mean(1)
        if args.metric == "l1":
            return -((w[:, None, :] * (a - b).abs()).sum(-1) / w.sum(-1, keepdim=True).clamp_min(1.0)).mean(1)
        if args.metric == "ngf":
            na, nb, wg = self.normalized_gradients(a, b, w)
            num = (wg * na * nb).sum(-1)
            den = (wg * 0.5 * (na * na + nb * nb)).sum(-1).clamp_min(1e-6)
            return (num / den).mean(1)
        if args.metric == "nmi":
            P = a.shape[0]
            return mg.soft_nmi(a.reshape(P, -1), b.reshape(P, -1), w[:, None, :].expand_as(a).reshape(P, -1),
                               args.nmi_bins, args.nmi_sigma)
        return -((w[:, None, :] * (a - b) ** 2).sum(-1) / w.sum(-1, keepdim=True).clamp_min(1.0)).mean(1)

    def normalized_gradients(self, a: torch.Tensor, b: torch.Tensor, w: torch.Tensor) -> tuple:
        """Normalized 1D gradients of profiles a, b (P, F, L) along the line, at the interior points (P, F, L - 2):
        g / sqrt(g^2 + eps^2), with g the central difference and eps = --grad-eps x the standard deviation of g over
        the scored points (per profile and frame, so the contrast of each image does not matter): edges well above
        eps -> +-1, flat stretches -> 0. Also the point weights (P, 1, L - 2): scored where all three samples are."""
        wg = (w[:, :-2] * w[:, 1:-1] * w[:, 2:])[:, None, :]
        ws = wg.sum(-1, keepdim=True).clamp_min(1e-6)

        def normalize(x):
            g = x[..., 2:] - x[..., :-2]
            eps = self.args.grad_eps * ((wg * g * g).sum(-1, keepdim=True) / ws).sqrt() + 1e-6
            return g / (g * g + eps * eps).sqrt()

        return normalize(a), normalize(b), wg

    def local_ncc(self, a: torch.Tensor, b: torch.Tensor, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Windowed NCC of profiles a, b (P, F, L) with point weights w (P, L): per window of --lncc-window-mm
        (sliding by one sample) the NCC (P, F, L - k + 1) and its weight: the number of weighted points (0 below half a
        window) x a confidence that fades out windows where the profiles hardly vary -- min(1, (sd_a sd_b)^(1/2) /
        --lncc-min-std), sd in units of the profile's standard deviation over the whole scored stretch."""
        k = max(3, int(round(self.args.lncc_window_mm / self.args.line_step_mm)))
        k = min(k, a.shape[-1])
        wf = w[:, None, :]
        ws = wf.sum(-1, keepdim=True).clamp_min(1e-6)

        def zscore(x):
            mu = (wf * x).sum(-1, keepdim=True) / ws
            return (x - mu) / (((wf * (x - mu) ** 2).sum(-1, keepdim=True) / ws).sqrt() + 1e-6)

        def window_sum(x):
            c = F.pad(torch.cumsum(x, -1), (1, 0))
            return c[..., k:] - c[..., :-k]

        za, zb = zscore(a), zscore(b)
        n = window_sum(wf.expand_as(za))
        nn_ = n.clamp_min(1e-6)
        ma, mb = window_sum(wf * za) / nn_, window_sum(wf * zb) / nn_
        va = (window_sum(wf * za * za) / nn_ - ma * ma).clamp_min(0.0)
        vb = (window_sum(wf * zb * zb) / nn_ - mb * mb).clamp_min(0.0)
        cov = window_sum(wf * za * zb) / nn_ - ma * mb
        loc = cov / (va * vb + 1e-6).sqrt()
        conf = ((va * vb).pow(0.25) / self.args.lncc_min_std).clamp(max=1.0)
        return loc, n * conf * (n >= 0.5 * k).float()

    def line_geometry(self, m: int, pm: Posed, r: int, pr: Posed) -> dict:
        """The m-r intersection line at the poses: sample points pts (P, L, 3) at offsets self.T from the point nearest
        both pivots, their voxel coordinates in both planes, which are inside both images (in_img) and which are scored
        (valid, P x L; see --score-region), and the crossing geometry the PSF band needs."""
        na, nb = pm.Np, pr.Np
        d = torch.cross(na, nb, dim=1)
        sin = d.norm(dim=1, keepdim=True).clamp_min(1e-3)
        d = d / sin
        anchor = 0.5 * (pm.Cp + pr.Cp)                       # samples run +-extent around the point nearest the pivots
        rhs = torch.stack([(na * pm.Op).sum(1), (nb * pr.Op).sum(1), (d * anchor).sum(1)], 1)
        pts = mg.solve3_rows(na, nb, d, rhs)[:, None, :] + self.T[None, :, None] * d[:, None, :]       # (P, L, 3)
        vox = {m: self.to_voxel(m, pm, pts), r: self.to_voxel(r, pr, pts)}
        in_img = self.inside(m, vox[m]) & self.inside(r, vox[r])
        region = self.args.score_region
        if region == "full":
            valid = in_img
        elif region == "centred":
            valid = in_img & (self.T.abs() <= self.args.score_half_length_mm)[None, :]
        else:
            in_mask = None
            for i in (m, r):
                if self.mask[i] is not None:
                    hit = self.sample(self.mask[i], vox[i], "nearest")[:, 0] > 0.5
                    in_mask = hit if in_mask is None else in_mask | hit
            valid = in_img if in_mask is None else in_img & in_mask
        return dict(pts=pts, vox_m=vox[m], vox_r=vox[r], in_img=in_img, valid=valid, sin=sin, na=na, nb=nb)

    def point_weights(self, g: dict) -> torch.Tensor:
        """Weights (P, L) of the line points in the score: 1 where scored; --outside-mask-weight for the other points
        inside both images (mask region only)."""
        w_out = self.args.outside_mask_weight if self.args.score_region == "mask" else 0.0
        valid, in_img = g["valid"], g["in_img"]
        return valid.float() if w_out <= 0 else torch.where(valid, 1.0, torch.where(in_img, w_out, 0.0))

    def band_profiles(self, m: int, pm: Posed, r: int, pr: Posed, g: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """PSF-band profiles (P, F, L) x 2 of both images at the line points g['pts']: each image is sampled across the
        other slice's profile, along the other normal projected into its plane."""
        pts, sin, na, nb = g["pts"], g["sin"], g["na"], g["nb"]
        K = self.u_band.numel()
        profiles = []
        for i, pose, n_own, n_other, width in ((m, pm, na, nb, self.thick[r]), (r, pr, nb, na, self.thick[m])):
            X = pts
            if K > 1:
                mdir = n_other - (n_other * n_own).sum(1, keepdim=True) * n_own
                mdir = mdir / mdir.norm(dim=1, keepdim=True).clamp_min(1e-6)
                step = (width / sin)[:, None, :, None] * self.u_band[None, None, :, None]                # (P, 1, K, 1)
                X = (pts[:, :, None, :] + step * mdir[:, None, None, :]).reshape(pts.shape[0], -1, 3)
            s = self.sample(self.img[i], self.to_voxel(i, pose, X))                                     # (P, F, L K)
            profiles.append((s.reshape(s.shape[0], s.shape[1], -1, K) * self.w_band).sum(-1))
        return profiles[0], profiles[1]

    def line_score(self, m: int, pm: Posed, r: int, pr: Posed) -> torch.Tensor:
        """Similarity (P,) along the m-r intersection line, the floor where it has too few scored points."""
        g = self.line_geometry(m, pm, r, pr)
        n_valid = g["valid"].sum(1)
        w = self.point_weights(g)
        # only the stretch with weight for some candidate of the batch is sampled (weight 0 elsewhere: same result)
        used = (w > 0).any(0).nonzero()
        if used.numel() == 0:
            return torch.full((g["pts"].shape[0],), self.floor, device=self.device)
        lo, hi = int(used[0]), int(used[-1]) + 1
        g = dict(g, pts=g["pts"][:, lo:hi])
        a, b = self.band_profiles(m, pm, r, pr, g)
        sim = self.similarity(a, b, w[:, lo:hi])
        return torch.where(n_valid >= self.min_points, sim, torch.full_like(sim, self.floor))

    def overlap_grid(self, m: int) -> tuple[torch.Tensor, torch.Tensor]:
        """World points (G, 3) of a --sax-overlap-step-mm grid on plane m (acquired geometry; within
        --sax-overlap-radius-mm of its pivot if > 0) and the (currently blurred) image there (1, F, G)."""
        if (m, self.blur_mm) not in self._overlap_grids:
            p, step = self.planes[m], self.args.sax_overlap_step_mm
            ii = torch.arange(0, p.H, max(1, round(step / p.spacing[0])), device=self.device, dtype=torch.float32)
            jj = torch.arange(0, p.W, max(1, round(step / p.spacing[1])), device=self.device, dtype=torch.float32)
            vox = torch.stack(torch.meshgrid(ii, jj, indexing="ij"), -1).reshape(-1, 2)
            A = torch.tensor(p.affine, dtype=torch.float32, device=self.device)
            X = vox @ A[:3, :2].T + A[:3, 3]
            if self.args.sax_overlap_radius_mm > 0:
                keep = (X - self.C[m]).norm(dim=1) <= self.args.sax_overlap_radius_mm
                X, vox = X[keep], vox[keep]
            self._overlap_grids[m, self.blur_mm] = (X, self.sample(self.img[m], vox[None]))
        return self._overlap_grids[m, self.blur_mm]

    def overlap_cost(self, m: int, pm: Posed, j: int, pj: Posed) -> torch.Tensor:
        """sum(slab overlap) x (1 - weighted NCC) (P,) of plane m at the poses against plane j (see the docstring)."""
        X, a = self.overlap_grid(m)
        Xp = torch.einsum("pij,gj->pgi", pm.R, X - self.C[m]) + pm.Cp[:, None, :]                    # (P, G, 3)
        dist = ((Xp - pj.Op[:, None, :]) * pj.Np[:, None, :]).sum(-1)                                # (P, G)
        overlap = (1.0 - dist.abs() / (0.5 * (self.thick[m] + self.thick[j]))).clamp_min(0.0)
        vox = self.to_voxel(j, pj, Xp - dist[..., None] * pj.Np[:, None, :])
        overlap = overlap * self.inside(j, vox).float()
        b = self.sample(self.img[j], vox)                                                           # (P, F, G)
        # ncc() floors the weight sum at 1 (meant for 0/1 masks): rescale the soft weights to a maximum of 1
        wn = overlap / overlap.amax(-1, keepdim=True).clamp_min(1e-6)
        c = mg.ncc(a, b, wn[:, None, :], False).mean(1)
        return overlap.sum(-1) * (1.0 - c)

    def score(self, m: int, refs: list[tuple[int, float]], neighbours: list[int], theta: torch.Tensor,
              t: torch.Tensor) -> torch.Tensor:
        """Score (P,) of plane m at the candidate poses theta, t (P, 3), the others at their current poses: the
        weighted mean line similarity against refs [(plane, line weight)], minus the SAX overlap cost against
        `neighbours`."""
        P = theta.shape[0]
        pm = self.posed(m, theta, t)
        wsum = sum(w for _, w in refs)
        score = sum(w * self.line_score(m, pm, r, self.current(r, P)) for r, w in refs) / wsum
        lam = self.args.sax_overlap_weight
        for j in neighbours:
            score = score - lam * self.overlap_cost(m, pm, j, self.current(j, P)) / wsum
        return score

    # ---- search ------------------------------------------------------------------------------------------------

    @staticmethod
    def distinct_top(offs: torch.Tensor, scores: torch.Tensor, k: int, level: Level) -> torch.Tensor:
        """The (up to) k best offsets, skipping any within one grid step (on every axis) of a better kept one."""
        n = min(scores.numel(), 64 * k)
        top = torch.topk(scores, n)
        cand = offs[top.indices][torch.isfinite(top.values)]
        unit = torch.tensor([level.t_step] * 3 + [level.rot_step or 1.0] * 3, device=offs.device)
        q = (cand / unit).cpu().numpy()
        kept = []
        for i in range(len(q)):
            if all(np.abs(q[i] - q[j]).max() > 1.0 + 1e-6 for j in kept):
                kept.append(i)
                if len(kept) == k:
                    break
        return cand[kept]

    @torch.no_grad()
    def search(self, m: int, refs: list[tuple[int, float]], spec: GridSpec, neighbours: list[int] = ()) -> dict:
        """Coarse-to-fine search of plane m's pose in the box of `spec` around its current pose (see GridSpec); m is
        moved to the best candidate of the last level (ties within --tie-tol broken towards the smallest move), which
        always includes the current pose. Returns the search record."""
        frame = self.frames[m]
        th0, t0 = self.theta[m].clone(), self.t[m].clone()
        levels = spec.levels()
        n_total = 0
        survivors = None
        for li, level in enumerate(levels):
            if survivors is None:
                offs = level.offsets(self.device)
            else:                                    # +-1 previous step around every survivor, inside the box
                offs = spec.clip((survivors[:, None, :] + level.offsets(self.device)[None]).reshape(-1, 6))
                offs = torch.unique(torch.round(offs * 1e4) / 1e4, dim=0)
            last = li == len(levels) - 1
            if last:                                 # the current pose competes in the final choice (row 0)
                offs = torch.cat([torch.zeros(1, 6, device=self.device), offs])
            self.set_blur(level.blur_mm)
            scores = torch.cat([self.score(m, refs, list(neighbours), th0 + o[:, 3:] @ frame, t0 + o[:, :3] @ frame)
                                for o in offs.split(self.batch)])
            scores = torch.nan_to_num(scores, nan=-float("inf"))
            n_total += offs.shape[0]
            if not last:
                survivors = self.distinct_top(offs, scores, self.args.top_k, level)
        self.set_blur(0.0)
        near = scores >= scores.max() - self.args.tie_tol
        move = offs[:, :3].pow(2).sum(1) + (DEG_TO_MM * offs[:, 3:]).pow(2).sum(1)
        k = int(torch.where(near, move, torch.full_like(move, float("inf"))).argmin())
        self.theta[m] = th0 + offs[k, 3:] @ frame
        self.t[m] = t0 + offs[k, :3] @ frame
        return dict(n_candidates=n_total, n_levels=len(levels), score_start=float(scores[0]), score=float(scores[k]),
                    offset=offs[k].tolist())


def brute_force_align(final_sampler: mg.LineSampler, args, stages=None, log: list | None = None) -> dict:
    """Stands in for motion_gradient.align (same signature and result; `stages`, the mask annealing, is unused): the
    staged grid search of the module docstring. `log` collects one SEARCH_COLUMNS row per plane search."""
    planes, device = final_sampler.planes, final_sampler.device
    pair_weight = {frozenset((ia, ib)): float(n) for ia, ib, n in
                   zip(final_sampler.IA.tolist(), final_sampler.IB.tolist(), final_sampler.npts.tolist())}
    gs = GridSearcher(planes, pair_weight, (final_sampler.u_img, final_sampler.band_weights()), args, device)

    view_idx = {p.image: i for i, p in enumerate(planes) if p.image != "SAX"}
    views = [view_idx[v] for v in VIEW_ORDER if v in view_idx]
    sax = sorted((p.slice_idx, i) for i, p in enumerate(planes) if p.image == "SAX")
    dz = args.dz_range_mm if args.dz_range_mm is not None else args.t_range_mm
    rot = args.rot_range_deg if args.dof == 6 else 0.0
    coarse = dict(coarse_t_step=args.coarse_t_step_mm, coarse_rot_step=args.coarse_rot_step_deg, blur_mm=args.coarse_blur_mm)
    full = GridSpec(args.t_range_mm, args.t_step_mm, dz, rot, args.rot_step_deg, **coarse)
    translations = GridSpec(args.t_range_mm, args.t_step_mm, dz, **coarse)
    refine = GridSpec(args.refine_t_range_mm, args.refine_t_step_mm, args.refine_t_range_mm,
                      args.refine_rot_range_deg if args.dof == 6 else 0.0, args.refine_rot_step_deg)   # one exhaustive level

    def run(stage: int, m: int | None, others: list, spec: GridSpec, neighbours=()):
        if m is None:
            return
        refs = [(r, pair_weight[frozenset((m, r))]) for r in others if r is not None and frozenset((m, r)) in pair_weight]
        if not refs:
            return
        t0 = time.time()
        rec = gs.search(m, refs, spec, neighbours)
        p = planes[m]
        if log is not None:
            log.append(dict(stage=stage, image=p.image, slice=p.slice_idx, n_candidates=rec["n_candidates"], n_levels=rec["n_levels"],
                            references=" ".join(planes[r].key for r, _ in refs) + "".join(
                                f" overlap:{planes[j].key}" for j in neighbours),
                            score_start=rec["score_start"], score=rec["score"],
                            **dict(zip(OFFSET_COLUMNS, rec["offset"])), seconds=round(time.time() - t0, 3)))

    snapshots = [(0, gs.theta.clone(), gs.t.clone())]
    ref = views[0] if views else None
    lax_a = views[1] if len(views) > 1 else None
    lax_b = views[2] if len(views) > 2 else None
    mid = None
    if sax:
        if args.mid_slice >= 0 and any(z == args.mid_slice for z, _ in sax):
            mid = next(i for z, i in sax if z == args.mid_slice)
        else:                                       # the slice with the most in-mask points on its LAX lines
            mid = max((i for _, i in sax), key=lambda i: sum(pair_weight.get(frozenset((i, v)), 0.0) for v in views))

    run(1, lax_a, [ref], translations)
    snapshots.append((1, gs.theta.clone(), gs.t.clone()))
    run(2, mid, [ref, lax_a], full)
    snapshots.append((2, gs.theta.clone(), gs.t.clone()))
    run(3, lax_b, [ref, lax_a, mid], full)
    snapshots.append((3, gs.theta.clone(), gs.t.clone()))
    if mid is not None:
        z_mid = planes[mid].slice_idx
        by_slice = dict(sax)
        placed = set()
        for z, i in sorted(sax, key=lambda zi: (abs(zi[0] - z_mid), zi[0])):          # outward from the mid slice
            neighbours = ([by_slice[n] for n in (z - 1, z + 1) if n in by_slice and by_slice[n] in placed]
                          if args.sax_overlap_weight > 0 else [])
            run(4, i, views, full, neighbours)
            placed.add(i)
    snapshots.append((4, gs.theta.clone(), gs.t.clone()))
    if args.lax_refine:
        for v in (lax_a, lax_b):
            run(5, v, [i for i in range(len(planes)) if i != v], refine)
    snapshots.append((5, gs.theta.clone(), gs.t.clone()))
    return dict(theta_deg=gs.theta.clone(), t=gs.t.clone(), best_step=N_STAGES, snapshots=snapshots)


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------

def default_run_name(ap: argparse.ArgumentParser, argv: list[str], args) -> str:
    """<YYYYMMDD-HHMMSS>_bf, followed by every result-affecting option given on the command line (in --help order)."""
    given = {tok.split("=", 1)[0] for tok in argv if tok.startswith("--")}
    parts = ["bf"]
    for action in ap._actions:
        if action.dest in mg.NAME_EXCLUDED | {"help", "eval_batch"} or not given.intersection(action.option_strings):
            continue
        value = getattr(args, action.dest)
        value = f"{value:g}" if isinstance(value, float) else str(value).replace(",", "+").replace(" ", "")
        parts.append(mg.short_option_name(action.option_strings[-1]) + ("" if action.nargs == 0 else value))
    return f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{'-'.join(parts)[:mg.MAX_RUN_NAME]}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path(r"C:\Users\Nil\Documents\data\ukbb_slices"),
                    help="Directory containing one folder per subject with single-slice NIfTIs (see prepare_ukbb_slices.py)")
    ap.add_argument("--out-root", type=Path, default=Path(r"C:\Users\Nil\Documents\data\ukbb_alignment_bf"), help="Parent of the run directory")
    ap.add_argument("--run-name", type=str, default=None,
                    help="Run directory name (default: the start time followed by the settings given on the command line)")
    ap.add_argument("--logging", type=mg.str2bool, default=True, metavar="BOOL",
                    help="true (default): also write metrics, CSVs, figures and run_config.json; false: aligned NIfTIs only")
    ap.add_argument("--max-subjects", type=int, default=1, help="Max number of subjects to evaluate (-1 = all)")
    ap.add_argument("--skip-processed", action="store_true",
                    help="Skip subjects whose aligned NIfTIs are already complete in the run directory")
    ap.add_argument("--n-frames", type=int, default=5, help="Number of cardiac phases (evenly spaced) to use. Default: %(default)s")
    ap.add_argument("--dof", type=int, choices=[3, 6], default=6,
                    help="3 = translations only, 6 = translations and rotations (stage 1 is always translations only)")
    ap.add_argument("--rotation-pivot", choices=["long-axis", "image-centre"], default="long-axis",
                    help="Point each plane rotates about (see motion_gradient.py). Default: %(default)s")
    # grid
    ap.add_argument("--t-range-mm", type=float, default=25.0, help="In-plane translation range +- (mm). Default: %(default)s")
    ap.add_argument("--dz-range-mm", type=float, default=None,
                    help="Through-plane translation range +- (mm). Default: --t-range-mm")
    ap.add_argument("--t-step-mm", type=float, default=2.0, help="Final translation grid step (mm). Default: %(default)s")
    ap.add_argument("--rot-range-deg", type=float, default=8.0,
                    help="Rotation range +- (deg) about each of the plane's three axes (--dof 6). Default: %(default)s")
    ap.add_argument("--rot-step-deg", type=float, default=1.0, help="Final rotation grid step (deg). Default: %(default)s")
    ap.add_argument("--coarse-t-step-mm", type=float, default=4.0,
                    help="Translation step of the first, box-wide coarse-to-fine level (mm); the step halves per level "
                         "down to --t-step-mm (equal to it: no coarse-to-fine). Default: %(default)s")
    ap.add_argument("--coarse-rot-step-deg", type=float, default=4.0,
                    help="Rotation step of the first level (deg), halving down to --rot-step-deg. Default: %(default)s")
    ap.add_argument("--top-k", type=int, default=8,
                    help="Distinct candidates (not grid neighbours of a better one) kept per level and refined at the "
                         "next. Default: %(default)s")
    ap.add_argument("--coarse-blur-mm", type=float, default=10.0,
                    help="In-plane Gaussian blur sigma (mm) of the images at the first level, halving per level, none at "
                         "the last (0 = off). Default: %(default)s")
    ap.add_argument("--fp16", type=mg.str2bool, default=False, metavar="BOOL",
                    help="Sample the images in half precision (faster; the sample positions are then rounded to ~0.1 "
                         "voxel). Default: %(default)s")
    ap.add_argument("--lax-refine", type=mg.str2bool, default=True, metavar="BOOL",
                    help="Stage 5: re-search 3Ch and 2Ch on a narrow grid against all other planes. Default: %(default)s")
    ap.add_argument("--refine-t-range-mm", type=float, default=5.0, help="Stage 5 translation range +- (mm). Default: %(default)s")
    ap.add_argument("--refine-t-step-mm", type=float, default=1.0, help="Stage 5 translation step (mm). Default: %(default)s")
    ap.add_argument("--refine-rot-range-deg", type=float, default=2.0, help="Stage 5 rotation range +- (deg). Default: %(default)s")
    ap.add_argument("--refine-rot-step-deg", type=float, default=1.0, help="Stage 5 rotation step (deg). Default: %(default)s")
    ap.add_argument("--tie-tol", type=float, default=1e-4,
                    help="Candidates within this of the best score count as tied; the smallest move among them wins. "
                         "Default: %(default)s")
    ap.add_argument("--mid-slice", type=int, default=-1,
                    help="SAX slice index placed in stage 2 (-1 = the one with the most in-mask line points)")
    ap.add_argument("--eval-batch", type=int, default=1024,
                    help="Candidate poses per batched GPU pass (speed / memory only). Default: %(default)s")
    # SAX-SAX overlap
    ap.add_argument("--sax-overlap-weight", type=float, default=0.0,
                    help="Weight of the SAX-SAX slab-overlap dissimilarity in stage 4 (0 = off). Default: %(default)s")
    ap.add_argument("--sax-overlap-step-mm", type=float, default=6.0,
                    help="Spacing (mm) of the grid of points on which the overlap is evaluated. Default: %(default)s")
    ap.add_argument("--sax-overlap-radius-mm", type=float, default=0.0,
                    help="Only grid points within this distance (mm) of the slice's pivot (0 = the whole image). "
                         "Default: %(default)s")
    # similarity (as in motion_gradient.py)
    ap.add_argument("--metric", choices=["ncc", "ncc_grad", "lncc", "ngf", "l1", "nmi", "l2"], default="lncc",
                    help="Similarity along each line (see 'Metrics' above). Default: %(default)s")
    ap.add_argument("--grad-eps", type=float, default=0.5,
                    help="--metric ngf: edge threshold, in standard deviations of the profile's gradient over the scored "
                         "stretch; gradients well above it count as full edges, those below it fade out. Default: %(default)s")
    ap.add_argument("--lncc-window-mm", type=float, default=20.0, help="--metric lncc: window length (mm). Default: %(default)s")
    ap.add_argument("--lncc-min-std", type=float, default=0.25,
                    help="--metric lncc: windows whose profiles vary less than this (geometric mean of both standard "
                         "deviations, in units of the profile's standard deviation over the scored stretch) are faded "
                         "out (weight sd / this). Default: %(default)s")
    ap.add_argument("--score-region", choices=["mask", "full", "centred"], default="full",
                    help="Which line points are scored: mask = inside both images and the dilated LAX heart mask; full "
                         "= everywhere inside both images; centred = inside both images within --score-half-length-mm "
                         "of the point nearest both pivots (the heart). Default: %(default)s")
    ap.add_argument("--score-half-length-mm", type=float, default=60.0,
                    help="--score-region centred: half-length (mm) of the scored stretch. Default: %(default)s")
    ap.add_argument("--nmi-bins", type=int, default=24, help="Histogram bins for mutual information")
    ap.add_argument("--nmi-sigma", type=float, default=1.0, help="Parzen window width for mutual information, in bins")
    ap.add_argument("--mask-dilation-mm", type=float, default=100.0,
                    help="How far (mm) the LAX heart segmentation is grown to form the mask. Default: %(default)s")
    ap.add_argument("--outside-mask-weight", type=float, default=0.0,
                    help="Weight of line points outside the mask (0 = excluded). Default: %(default)s")
    ap.add_argument("--line-step-mm", type=float, default=2.0, help="Sample spacing along the lines (mm). Default: %(default)s")
    ap.add_argument("--line-extent-mm", type=float, default=150.0, help="Half-length of the lines (mm). Default: %(default)s")
    ap.add_argument("--min-line-mm", type=float, default=40.0,
                    help="A line needs this many mm inside both images and the mask (acquired geometry: to be used at "
                         "all; per candidate: to score more than the floor). Default: %(default)s")
    ap.add_argument("--sax-thickness", type=float, default=8.0, help="SAX slice thickness (mm). Default: %(default)s")
    ap.add_argument("--lax-thickness", type=float, default=6.0, help="LAX slice thickness (mm). Default: %(default)s")
    ap.add_argument("--n-band", type=int, default=4,
                    help="PSF band samples across the other slice's profile (1 = the line only). Default: %(default)s")
    ap.add_argument("--psf-model", choices=["pulse", "gaussian", "rect"], default="gaussian",
                    help="Through-plane slice profile (see motion_gradient.py). Default: %(default)s")
    ap.add_argument("--psf-tbw", type=float, default=2.5, help="Time-bandwidth product for --psf-model pulse")
    ap.add_argument("--no-psf", action="store_true", help="Sample the line only (sets --n-band 1)")
    # metrics only (not optimized)
    ap.add_argument("--seg-labels", type=str, default="1,2,3,4,5", help="Labels compared along LAX-LAX intersections (metrics)")
    ap.add_argument("--sax-seg-labels", type=str, default="1,2,3", help="Labels compared along SAX-LAX intersections (metrics)")
    ap.add_argument("--hd-miss-mm", type=float, default=100.0, help="Hausdorff value when only one profile hits a label (metrics)")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--workers", type=int, default=4, help="Background threads for loading and writing NIfTI files")
    args = ap.parse_args()
    if args.no_psf:
        args.n_band = 1
    # settings motion_gradient's shared code reads; nothing here optimizes them
    args.steps = N_STAGES                        # history.csv: step k = the poses after stage k
    args.psf_scale_start = 1.0
    args.sax_sax_weight, args.sax_sax_radius_mm = 0.0, 120.0     # SaxNeighbourLines QC (sax_lines.csv) only

    run_dir = args.out_root / (args.run_name or default_run_name(ap, sys.argv[1:], args))
    overall_dir = run_dir / "overall_metrics"
    summary_path = overall_dir / "motion_summary.csv"
    detail_path = overall_dir / "motion_summary_slices.csv"
    pairs_path = overall_dir / "motion_summary_pairs.csv"
    search_path = overall_dir / "motion_search.csv"

    device = torch.device(args.device)
    subjects = sorted(d for d in args.root.iterdir() if d.is_dir())
    if args.skip_processed:
        n0 = len(subjects)
        subjects = [s for s in subjects if not mg.aligned_complete(s, run_dir / s.name)]
        print(f"Skipping {n0 - len(subjects)} subjects whose aligned NIfTIs are complete")
    if args.max_subjects >= 0:
        subjects = subjects[:args.max_subjects]
    print(f"Aligning {len(subjects)} subjects on {device}; outputs in {run_dir}")
    if not subjects:
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.logging:
        overall_dir.mkdir(parents=True, exist_ok=True)
        mg.write_run_config(overall_dir, args)

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **k: x  # noqa: E731

    writers = None
    if args.logging:
        writers = [mg.open_csv(summary_path, mg.SUMMARY_COLUMNS), mg.open_csv(detail_path, mg.DETAIL_COLUMNS),
                   mg.open_csv(pairs_path, mg.PAIR_COLUMNS), mg.open_csv(search_path, SEARCH_COLUMNS)]
    save_jobs = deque()
    failed = []
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            it = iter(subjects)
            queue = deque((s, ex.submit(mg.load_subject, s, args)) for s in islice(it, 2 * args.workers))
            pbar = tqdm(total=len(subjects), unit="subj")
            while queue:
                subj, fut = queue.popleft()
                nxt = next(it, None)
                if nxt is not None:
                    queue.append((nxt, ex.submit(mg.load_subject, nxt, args)))
                t0 = time.time()
                metrics_dir = run_dir / subj.name / "metrics"
                summary, details, pairs, history, search_rows = None, [], [], None, []

                def on_lines(planes, lines, sid=subj.name, out=metrics_dir):
                    if not lines:
                        return
                    out.mkdir(parents=True, exist_ok=True)
                    mg.save_line_geometry(planes, lines, out, args.line_extent_mm, args.line_step_mm)
                    viz = mg.mask_plot_data(planes, lines, args.line_extent_mm, args.line_step_mm)
                    save_jobs.append((sid, "mask plot", ex.submit(mg.save_mask_plot, viz, sid, args.mask_dilation_mm,
                                                                  out / "mask_overlay.png", args.line_extent_mm)))

                status = "ok"
                try:
                    data = fut.result()
                    status = data["status"]
                    if status == "ok":
                        # process_subject builds the planes and lines, calls align() and computes the metrics
                        mg.align = functools.partial(brute_force_align, log=search_rows)
                        summary, details, pairs, poses, history = mg.process_subject(
                            data, args, device, on_lines=on_lines if args.logging else None)
                        save_jobs.append((subj.name, "aligned NIfTIs", ex.submit(mg.save_aligned, data, poses, run_dir, args)))
                except Exception as e:  # noqa: BLE001
                    traceback.print_exc()
                    status = f"error: {type(e).__name__}: {e}"[:200]
                if status != "ok":
                    failed.append((subj.name, status))
                search_s = sum(r["seconds"] for r in search_rows)
                if args.logging:
                    summary = summary if summary is not None else mg.empty_summary(subj.name, status)
                    summary["seconds"] = round(time.time() - t0, 2)
                    summary["timestamp"] = datetime.now().isoformat(timespec="seconds")
                    for r in search_rows:
                        r["subject_id"] = subj.name
                    if details:
                        save_jobs.append((subj.name, "subject metrics", ex.submit(mg.save_subject_metrics, dict(summary), details,
                                                                                 pairs, metrics_dir, True, history)))
                    if search_rows:
                        metrics_dir.mkdir(parents=True, exist_ok=True)
                        pd.DataFrame([mg.fmt_row(r) for r in search_rows], columns=SEARCH_COLUMNS).to_csv(
                            metrics_dir / "search.csv", index=False)
                    rows = ([summary], details, pairs, search_rows)
                    for (f, writer), rs in zip(writers, rows):
                        for row in rs:
                            writer.writerow(mg.fmt_row(row))
                        f.flush()
                while len(save_jobs) > 3 * args.workers:
                    name, what, job = save_jobs.popleft()
                    try:
                        job.result()
                    except Exception:  # noqa: BLE001
                        print(f"Writing {what} failed for {name}:")
                        traceback.print_exc()
                if hasattr(pbar, "set_postfix"):
                    post = dict(subj=subj.name, status=status[:12], search=f"{search_s:.1f}s")
                    if summary is not None:
                        post.update(ncc=f"{summary['ncc_mean']:.3f}")
                    pbar.set_postfix(**post)
                    pbar.update(1)
            for name, what, job in save_jobs:
                try:
                    job.result()
                except Exception:  # noqa: BLE001
                    print(f"Writing {what} failed for {name}:")
                    traceback.print_exc()
    finally:
        for f, _ in writers or []:
            f.close()
    if failed:
        print(f"\n{len(failed)} subject(s) not aligned: " + ", ".join(f"{s} ({st})" for s, st in failed))
    if args.logging:
        print(f"\nSummary written to {summary_path}\nPlane searches written to {search_path}")
        try:
            n = len(mg.plot_results(summary_path, detail_path, overall_dir, pairs_path))
            print(f"{n} overall figures written to {overall_dir}")
        except Exception:  # noqa: BLE001
            print("Overall plotting failed:")
            traceback.print_exc()
    print(f"Aligned NIfTIs under {run_dir}\\<subject>\\")


if __name__ == "__main__":
    main()
