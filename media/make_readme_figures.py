"""
Figures of the README section "How an alignment run proceeds", all from one UK Biobank subject and made with the
alignment code itself (motion_gradient.py): the masks come from prepare_subject, the intensity profiles from
LineSampler.profiles (what the loss compares), and the metrics from aligning the subject with the default settings.

    blur.png      in-plane blur: an image at sigma 0 / 1 / 3 mm and its profile along one intersection line
    mask.png      mask dilation stages, the apex extension and the supervised line points per SAX slice
    psf.png       the slice profile band: its weights, its footprint on the image, and the resulting profiles
    metrics.png   the subject aligned with the defaults: the schedules and the metrics over the optimization steps

The subject folder needs a SAX stack and a LAX_4Ch view with their nnU-Net segmentations (the UKBB layout read by
moco_nifti.py). For metrics.png the subject is aligned first (moco_nifti.py on a copy of the folder, in a temporary
directory unless --work-dir is given; ~30 s on a GPU), or an existing run is used (--run-dir).

Usage (from anywhere; figures go to the folder of this script unless --out is given):
    python media/make_readme_figures.py <ukbb_dir>/<subject>
    python media/make_readme_figures.py <ukbb_dir>/<subject> --figures metrics --run-dir <alignment_dir>/<run>
"""
import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
import numpy as np                                  # noqa: E402
import pandas as pd                                 # noqa: E402
import torch                                        # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import moco_nifti as mn                             # noqa: E402
import motion_gradient as mg                        # noqa: E402

# reference palette (categorical slots 1-3, blue ordinal ramp), text and chrome tokens
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RAMP = ["#86b6ef", "#5598e7", "#256abf", "#104281"]          # ordinal: light (first) -> dark (last)
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e6e3", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "normal", "axes.titlelocation": "left",
    "axes.edgecolor": "#b8b7b2", "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "lines.linewidth": 2.0,
    "legend.frameon": False, "legend.fontsize": 8.5,
})
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def default_args(root: Path, **over):
    """moco_nifti.py's arguments with their defaults (as a run would resolve them), then `over`."""
    cap = {}
    run, argv = mg.run, sys.argv
    mg.run = lambda ap, a, *rest: cap.update(a=a)
    sys.argv = ["moco_nifti.py", "--root", str(root)]
    try:
        mn.main()
    finally:
        mg.run, sys.argv = run, argv
    a = cap["a"]
    lr0, rot0 = mg.DEFAULT_LR[a.optimizer]
    a.lr = lr0 if a.lr is None else a.lr
    a.rot_lr = rot0 if a.rot_lr is None else a.rot_lr
    a.lr_end = mg.LR_END_FRACTION * a.lr if a.lr_end is None else a.lr_end
    a.rot_lr_end = mg.LR_END_FRACTION * a.rot_lr if a.rot_lr_end is None else a.rot_lr_end
    a.device = DEVICE
    for k, v in over.items():
        setattr(a, k, v)
    return a


class Scene:
    """One subject prepared with given settings: planes, intersection lines, sampler; helpers for the LAX_4Ch view."""

    def __init__(self, subject, args, view="LAX_4Ch"):
        self.args = args
        self.data = mg.prepare_subject(subject, args)
        sax, lax = mg.build_planes(self.data, args, DEVICE)
        self.planes = sax + lax
        self.lines = mg.build_lines(self.planes, args, DEVICE)
        self.view = view
        e = self.data["lax"][view]
        self.img = e["arr"][:, :, 0, 0]
        self.A = e["affine"][0]
        self.inv = np.linalg.inv(self.A)
        self.spacing = [float(np.linalg.norm(self.A[:3, k])) for k in (0, 1)]
        self.dist = e["dist"][:, :, 0]
        self.heart = (e["seg"][:, :, 0, :] > 0).any(-1)
        self.sax = sorted((i for i, p in enumerate(self.planes) if p.image == "SAX"), key=lambda i: self.planes[i].slice_idx)

    def line_with(self, sax_index: int):
        """(line index, Line, side of the view) of the line between SAX slice `sax_index` and the view."""
        for m, (ia, ib, line) in enumerate(self.lines):
            pa, pb = self.planes[ia], self.planes[ib]
            if {pa.image, pb.image} == {"SAX", self.view} and sax_index in (pa.slice_idx, pb.slice_idx):
                if (pa if pa.image == "SAX" else pb).slice_idx == sax_index:
                    return m, line, (1 if pb.image == self.view else 0)
        raise LookupError(f"no line between SAX {sax_index} and {self.view}")

    def to_vox(self, pts):
        pts = np.atleast_2d(pts)
        return (self.inv @ np.c_[pts, np.ones(len(pts))].T)[:2].T

    def blurred(self, sigma_mm: float) -> np.ndarray:
        """The view's image blurred in-plane as the optimizer does (ImageVolume._blur)."""
        if sigma_mm <= 0:
            return self.img
        x = torch.tensor(self.img, dtype=torch.float32)[None, None]
        return mg.ImageVolume._blur(x, [self.spacing], sigma_mm)[0, 0].numpy()

    def profile(self, sampler, m: int, side: int, sigma: float, scale: float):
        """(t mm, intensity) of the view along line m at the acquired poses, frame 0, as the loss samples it."""
        sampler.set_blur(sigma)
        sampler.set_psf_scale(scale)
        zero = torch.zeros(sampler.N, 3, device=DEVICE)
        with torch.no_grad():
            a, b = sampler.profiles(zero, zero, "img", torch.tensor([m], device=DEVICE))
        n = self.lines[m][2].t.numel()
        return sampler.T[m, :n].cpu().numpy(), (a if side == 0 else b)[0, 0, :n].cpu().numpy()

    def crop(self, ax, dist_mm: float, margin: int = 6):
        """Zoom an image axis to the region within dist_mm of the heart."""
        ii, jj = np.nonzero(self.dist <= dist_mm)
        ax.set_xlim(max(ii.min() - margin, 0), min(ii.max() + margin, self.img.shape[0] - 1))
        ax.set_ylim(max(jj.min() - margin, 0), min(jj.max() + margin, self.img.shape[1] - 1))


def show(ax, img, title):
    ax.imshow(img.T, cmap="gray", origin="lower", interpolation="bilinear",
              vmin=np.percentile(img, 1), vmax=np.percentile(img, 99.5))
    ax.set_title(title)
    ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


def legend_below(ax, handles, ncol=1):
    """A legend under an image axis (on an image it would cover what it explains)."""
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.02), ncol=ncol, fontsize=8,
              handlelength=1.2 if ncol > 2 else 1.6, columnspacing=0.8, handletextpad=0.4, borderaxespad=0.0)


def headroom(ax, frac=0.22):
    """Extra space above the data for a legend inside the plot."""
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + frac * (hi - lo))


def line_handle(color, label, lw=2.0, **kw):
    return plt.Line2D([], [], color=color, lw=lw, label=label, **kw)


# ----------------------------------------------------------------------------------------------------------------------
def fig_blur(sc: Scene, out: Path):
    b0, b1 = float(sc.args.blur_sigma), min(float(sc.args.blur_sigma_end), float(sc.args.blur_sigma))  # the defaults
    sigmas = [(0.0, "original", BLUE), (b1, f"σ = {b1:g} mm (end of the run)", ORANGE),
              (b0, f"σ = {b0:g} mm (start of the run)", AQUA)]
    sampler = mg.new_sampler(sc.planes, sc.lines, sc.args, DEVICE)
    sax = sc.planes[sc.sax[len(sc.sax) // 2]].slice_idx
    m, line, side = sc.line_with(sax)
    pts = line.x0[None] + line.t.cpu().numpy()[:, None] * line.d[None]
    vox = sc.to_vox(pts)

    fig = plt.figure(figsize=(10, 6.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1], hspace=0.28, wspace=0.05)
    for k, (s, name, col) in enumerate(sigmas):
        ax = fig.add_subplot(gs[0, k])
        show(ax, sc.blurred(s), name)
        ax.plot(vox[:, 0], vox[:, 1], color=col, lw=2)
        sc.crop(ax, 45)
    ax = fig.add_subplot(gs[1, :])
    for s, name, col in sigmas:
        t, y = sc.profile(sampler, m, side, s, 1.0)
        ax.plot(t, y, color=col, label=name)
    ax.set_xlabel("position along the intersection line (mm)")
    ax.set_ylabel("intensity")
    ax.set_title(f"{sc.view} intensity along its intersection line with SAX slice {sax} (drawn above), "
                 "as the loss samples it", loc="left")
    headroom(ax)
    ax.legend(loc="upper left", ncol=3)
    fig.suptitle("Coarse-to-fine: the images are blurred in-plane, less and less over the run", x=0.07, ha="left",
                 fontsize=11)
    fig.savefig(out / "blur.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def supervised_points(sc: Scene) -> list[float]:
    """Line points (inside both images and the mask) of every SAX slice's lines with the LAX views, base -> apex."""
    pts = {i: 0.0 for i in sc.sax}
    for ia, ib, line in sc.lines:
        if sc.planes[ia].image == "SAX" and sc.planes[ib].image != "SAX":
            pts[ia] += line.n_points
        elif sc.planes[ib].image == "SAX" and sc.planes[ia].image != "SAX":
            pts[ib] += line.n_points
    return [pts[i] for i in sc.sax]


def fig_mask(subject, root: Path, out: Path):
    ext = Scene(subject, default_args(root))
    plain = Scene(subject, default_args(root, apex_extend_mm=0.0))
    a = ext.args
    fig = plt.figure(figsize=(11, 4.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.15], wspace=0.28)

    # (a) annealing: the mask at a few dilations
    ax = fig.add_subplot(gs[0, 0])
    show(ax, ext.img, f"Annealing: {a.mask_dilation_start_mm:g} mm \u2192 {a.mask_dilation_mm:g} mm")
    ax.contour(ext.heart.T.astype(float), [0.5], colors="white", linewidths=1.2)
    levels = [80.0, 60.0, 40.0, 20.0]
    for lev, col in zip(levels, RAMP):
        ax.contour(ext.dist.T, [lev], colors=[col], linewidths=2)
    ext.crop(ax, 85, margin=2)
    legend_below(ax, [line_handle("#9a9993", "heart", lw=1.2)]
                 + [line_handle(c, f"{lev:g}" + (" mm" if lev == levels[-1] else "")) for lev, c in zip(levels, RAMP)],
                 ncol=5)

    # (b) the apex extension at the final dilation
    ax = fig.add_subplot(gs[0, 1])
    show(ax, ext.img, f"Final {a.mask_dilation_mm:g} mm mask, apex extension")
    ax.contour(plain.dist.T, [a.mask_dilation_mm], colors=[ORANGE], linewidths=2)
    ax.contour(ext.dist.T, [a.mask_dilation_mm], colors=[AQUA], linewidths=2)
    sl = [s for s in subject.slices]
    apex = mg.apex_direction([(s.affine, s.data.shape[:2], s.seg) for s in sorted(
        (s for s in sl if s.image == "SAX"), key=lambda s: s.index)], [(s.affine, s.seg) for s in sl if s.image != "SAX"])
    c = np.argwhere(ext.heart).mean(0)
    v = ext.inv[:3, :3] @ apex
    v = v[:2] / np.linalg.norm(v[:2]) * (35.0 / ext.spacing[0])
    ax.annotate("", xy=(c[0] + v[0], c[1] + v[1]), xytext=(c[0], c[1]),
                arrowprops=dict(arrowstyle="-|>", color="white", lw=1.6))
    ax.text(c[0] + v[0], c[1] + v[1], "  to apex", color="white", fontsize=8, va="center")
    ext.crop(ax, 45)
    legend_below(ax, [line_handle(ORANGE, "without extension"),
                      line_handle(AQUA, f"extended {a.apex_extend_mm:g} mm toward the apex")])

    # (c) supervised points per SAX slice
    ax = fig.add_subplot(gs[0, 2])
    p0, p1 = supervised_points(plain), supervised_points(ext)
    x = np.arange(len(p0))
    ax.plot(x, p0, "o-", color=ORANGE, ms=5, label="without extension")
    ax.plot(x, p1, "s-", color=AQUA, ms=5, label="with extension")
    ax.set_xticks(x, [str(i) for i in x])
    ax.set_xlabel("SAX slice (0 = base \u2192 apex)")
    ax.set_ylabel("supervised line points")
    ax.set_title("Supervision per SAX slice at the final mask")
    ax.legend(loc="lower left")
    fig.suptitle("Heart mask: annealed dilation and the apex extension", x=0.07, ha="left", fontsize=11)
    fig.savefig(out / "mask.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def fig_psf(sc: Scene, out: Path):
    a = sc.args
    sampler = mg.new_sampler(sc.planes, sc.lines, a, DEVICE)
    a1 = default_args(Path(a.root), n_band=1)
    no_band = mg.new_sampler(sc.planes, sc.lines, a1, DEVICE)
    sax_i = sc.sax[len(sc.sax) // 2]
    sax = sc.planes[sax_i].slice_idx
    m, line, side = sc.line_with(sax)
    T = sc.planes[sax_i].thickness
    na, nb = sc.planes[sax_i].normal, np.cross(sc.A[:3, 0], sc.A[:3, 1])
    nb = nb / np.linalg.norm(nb)
    mb = na - (na @ nb) * nb
    sin = float(np.linalg.norm(nb - (nb @ na) * na))
    mb = mb / np.linalg.norm(mb)
    pts = line.x0[None] + line.t.cpu().numpy()[:, None] * line.d[None]

    fig = plt.figure(figsize=(11, 4.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.25], wspace=0.25)
    # (a) the band weights across the SAX slab
    ax = fig.add_subplot(gs[0, 0])
    u = np.linspace(-3.5, 3.5, 600)
    for scale, col, name in ((1.0, BLUE, "physical (end of the run)"),
                             (a.psf_scale_start, AQUA, f"{a.psf_scale_start:g}\u00d7 wider (start)")):
        w = sampler.slice_profile(torch.tensor(u / scale, dtype=torch.float32)).numpy()
        ax.plot(u * T, w / w.max(), color=col, label=name)
    ax.axvspan(-T / 2, T / 2, color=GRID, zorder=0)
    ax.text(0, 1.06, f"{T:g} mm slice", ha="center", fontsize=8, color=INK2)
    ax.set_xlabel("offset across the SAX slice (mm)")
    ax.set_ylabel("weight (max = 1)")
    ax.set_title("Slice profile used as the band weights")
    ax.set_ylim(-0.03, 1.32)
    ax.legend(loc="upper left")

    # (b) where the band samples the LAX view
    ax = fig.add_subplot(gs[0, 1])
    show(ax, sc.img, f"Band on the {sc.view} view")
    for scale, col in ((a.psf_scale_start, AQUA), (1.0, BLUE)):
        h = 0.5 * T / sin * scale                        # half the profile's FWHM, measured within the view
        vv = sc.to_vox(np.r_[pts + h * mb, (pts - h * mb)[::-1]])
        if scale > 1:
            ax.fill(vv[:, 0], vv[:, 1], color=col, alpha=0.45, lw=0)
        else:
            for edge in (pts + h * mb, pts - h * mb):
                ev = sc.to_vox(edge)
                ax.plot(ev[:, 0], ev[:, 1], color=col, lw=1.6)
    vox = sc.to_vox(pts)
    ax.plot(vox[:, 0], vox[:, 1], color="white", lw=1)
    sc.crop(ax, 40)
    legend_below(ax, [line_handle("white", f"line with SAX slice {sax}", lw=1, mec=INK2),
                      line_handle(BLUE, "physical band (FWHM)", lw=1.6),
                      plt.Rectangle((0, 0), 1, 1, color=AQUA, alpha=0.45, label=f"{a.psf_scale_start:g}× band")])
    ax.get_legend().get_lines()[0].set_color(INK2)

    # (c) the resulting profiles
    ax = fig.add_subplot(gs[0, 2])
    t, y = sc.profile(no_band, m, side, 0.0, 1.0)
    ax.plot(t, y, color=ORANGE, label="the line only (no band)")
    t, y = sc.profile(sampler, m, side, 0.0, 1.0)
    ax.plot(t, y, color=BLUE, label="physical band")
    t, y = sc.profile(sampler, m, side, 0.0, a.psf_scale_start)
    ax.plot(t, y, color=AQUA, label=f"{a.psf_scale_start:g}\u00d7 band")
    ax.set_xlabel("position along the intersection line (mm)")
    ax.set_ylabel("intensity")
    ax.set_title(f"{sc.view} profile the loss compares (no blur)")
    headroom(ax, 0.25)
    ax.legend(loc="upper left", ncol=1)
    fig.suptitle("Slice profile (PSF): each image is averaged across the other slice's thickness", x=0.07, ha="left",
                 fontsize=11)
    fig.savefig(out / "psf.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def schedule(cfg: dict, s: np.ndarray) -> dict:
    """Learning-rate fraction, mask dilation, blur sigma and PSF scale at steps s (as motion_gradient.align)."""
    steps = int(cfg["steps"])
    lr = 1.0 + (cfg["lr_end"] / cfg["lr"] - 1.0) * np.minimum(1.0, s / max(1, steps - 1)) \
        if cfg.get("lr_schedule", "constant") == "linear" else np.ones_like(s, float)
    d0, d1 = float(cfg["mask_dilation_start_mm"]), float(cfg["mask_dilation_mm"])
    a_end = int(round(cfg["mask_anneal_frac"] * steps))
    n_st = int(cfg["anneal_stages"])
    starts = sorted({int(round(k / n_st * a_end)) for k in range(n_st)})
    dil = np.array([d1 if x >= a_end else d0 + (d1 - d0) * max(i for i, st in enumerate(starts) if x >= st) / n_st
                    for x in s])
    b0, b1 = float(cfg["blur_sigma"]), min(float(cfg["blur_sigma_end"]), float(cfg["blur_sigma"]))
    b_steps = int(round(cfg["blur_frac"] * steps))
    sig = np.where(s < b_steps, b1 + (b0 - b1) * (1 - s / b_steps), b1)
    sig = np.ceil(sig / 0.2) * 0.2
    p0 = float(cfg["psf_scale_start"])
    p_steps = int(round(cfg["psf_anneal_frac"] * steps))
    psf = np.where(s < p_steps, p0 + (1 - p0) * s / p_steps, 1.0)
    return dict(lr=lr, dil=dil, sig=sig, psf=psf, track_from=max(a_end, b_steps, p_steps), steps=steps)


def fig_metrics(run_dir: Path, subject: str, out: Path):
    cfg = json.loads((run_dir / "overall_metrics" / "run_config.json").read_text())
    h = pd.read_csv(run_dir / subject / "metrics" / "history_summary.csv")
    best = h[h.is_best == 1].iloc[0]
    h = h[h.is_best == 0].sort_values("step")
    s = np.arange(int(cfg["steps"]) + 1)
    sch = schedule(cfg, s)
    left = [("lr", "learning rate\n(fraction of start)", "{:.2f}"), ("dil", "mask dilation (mm)", "{:g}"),
            ("sig", "blur \u03c3 (mm)", "{:.1f}"), ("psf", "PSF width\n(\u00d7 physical)", "{:.1f}")]
    right = [("lncc_mean", "LNCC (optimized)", True), ("ncc_mean", "NCC", True),
             ("seg_dice_mean", "Dice", True), ("hd_mean_mm", "Hausdorff (mm)", False)]
    fig, axes = plt.subplots(4, 2, figsize=(11, 7.6), sharex=True, gridspec_kw=dict(hspace=0.22, wspace=0.22))
    for ax in axes.ravel():
        ax.axvspan(sch["track_from"], sch["steps"], color=GRID, zorder=0, lw=0)
    for ax, (key, name, fmt) in zip(axes[:, 0], left):
        ax.plot(s, sch[key], color=BLUE, drawstyle="steps-post" if key in ("dil", "sig") else "default")
        ax.set_ylabel(name)
    for ax, (key, name, up) in zip(axes[:, 1], right):
        ax.axhline(h[key].iloc[0], color="#b8b7b2", lw=1)
        ax.text(sch["steps"] * 0.01, h[key].iloc[0], "acquired", color=INK2, fontsize=8,
                va="bottom" if up else "top")
        ax.plot(h.step, h[key], color=ORANGE)
        ax.plot([best.step], [best[key]], "o", color=ORANGE, ms=8, mec=SURFACE, mew=2)
        ax.set_ylabel(name + ("  \u2191" if up else "  \u2193"))
    axes[0, 1].annotate("saved pose", (best.step, best["lncc_mean"]), xytext=(-62, -14), textcoords="offset points",
                        fontsize=8, color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    axes[0, 0].text((sch["track_from"] + sch["steps"]) / 2, 0.98, "best pose\ntracked", ha="center", va="top",
                    fontsize=8, color=INK2, transform=axes[0, 0].get_xaxis_transform())
    axes[0, 0].set_title("Schedules (what the optimizer does)")
    axes[0, 1].set_title(f"Metrics of subject {subject} (\u2191 / \u2193: higher / lower is better)")
    for ax in axes[-1]:
        ax.set_xlabel("optimizer step")
    fig.suptitle("One alignment run with the default settings", x=0.07, ha="left", fontsize=11)
    fig.savefig(out / "metrics.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def align_subject(subject_dir: Path, work: Path) -> Path:
    """Align the subject with moco_nifti.py's defaults (with --logging) on a copy of its folder in `work`; returns the
    run directory. The copy keeps moco_nifti.py from scanning the subject's (possibly large) parent folder."""
    root = work / "input"
    copy = root / subject_dir.name
    if not copy.exists():
        shutil.copytree(subject_dir, copy)
    run_dir = work / "readme_figures_run"
    print(f"aligning {subject_dir.name} (moco_nifti.py defaults) into {run_dir} ...", flush=True)
    subprocess.run([sys.executable, str(REPO / "moco_nifti.py"), "--root", str(root), "--out-root", str(work),
                    "--run-name", run_dir.name, "--logging", "true"], cwd=str(REPO), check=True)
    return run_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("subject_dir", type=Path,
                    help="a UK Biobank subject folder (SAX stack and LAX views with nnU-Net segmentations)")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent,
                    help="output folder of the figures (default: the folder of this script)")
    ap.add_argument("--figures", nargs="+", choices=["blur", "psf", "mask", "metrics"],
                    default=["blur", "psf", "mask", "metrics"], help="which figures to make (default: all)")
    ap.add_argument("--run-dir", type=Path, default=None,
                    help="metrics.png from this existing moco_nifti.py run of the subject (made with --logging) "
                         "instead of aligning it now")
    ap.add_argument("--work-dir", type=Path, default=None,
                    help="keep the alignment of the subject here (default: a temporary folder, deleted afterwards)")
    a = ap.parse_args()
    subject_dir = a.subject_dir.resolve()
    if not subject_dir.is_dir():
        sys.exit(f"no such subject folder: {subject_dir}")
    root, sid = subject_dir.parent, subject_dir.name
    a.out.mkdir(parents=True, exist_ok=True)

    if {"blur", "psf", "mask"} & set(a.figures):
        args = default_args(root)
        subject = mn.load_subject(subject_dir, args)
        if not any(s.image == "SAX" for s in subject.slices) or not any(s.image == "LAX_4Ch" for s in subject.slices):
            sys.exit(f"{sid}: the figures need a SAX stack and a LAX_4Ch view")
        # the profile figures use the widest mask (start of the run), so the lines show more than the heart
        wide = Scene(subject, default_args(root, mask_dilation_mm=float(args.mask_dilation_start_mm)))
        if "blur" in a.figures:
            fig_blur(wide, a.out)
        if "psf" in a.figures:
            fig_psf(wide, a.out)
        if "mask" in a.figures:
            fig_mask(subject, root, a.out)

    if "metrics" in a.figures:
        if a.run_dir is not None:
            fig_metrics(a.run_dir, sid, a.out)
        elif a.work_dir is not None:
            a.work_dir.mkdir(parents=True, exist_ok=True)
            fig_metrics(align_subject(subject_dir, a.work_dir.resolve()), sid, a.out)
        else:
            with tempfile.TemporaryDirectory(prefix="readme_figures_") as tmp:
                fig_metrics(align_subject(subject_dir, Path(tmp)), sid, a.out)
    print("figures written to", a.out)


if __name__ == "__main__":
    main()
