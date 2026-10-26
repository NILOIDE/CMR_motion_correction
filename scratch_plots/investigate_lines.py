"""Independent checks of the intersection-line geometry and the profile sampling of motion_gradient.py.

1. For every LAX view: where each SAX slice (and each other LAX view) crosses it, computed by brute force (every LAX
   voxel is mapped into the other image's voxel space; the slice's trace is the contour where the through-plane
   voxel coordinate equals that slice index), overlaid with the lines used by the script. Distances are reported.
2. The profiles sampled by LineSampler at the acquired geometry (rect PSF, 1 sample) against a straight
   scipy.ndimage.map_coordinates reslice of the same points.
Usage: python investigate_lines.py [subject ...]
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from matplotlib.figure import Figure
from scipy.ndimage import map_coordinates

sys.path.insert(0, r"C:\Users\Nil\Documents\git\UKBB_implicit_baselines")
import motion_gradient as mg  # noqa: E402

ROOT = Path(r"C:\Users\Nil\Documents\git\UKBB_implicit_baselines\scratch_plots\gs_root5")
OUT = Path(r"C:\Users\Nil\Documents\git\UKBB_implicit_baselines\scratch_plots\line_check")
ST = mg.PLOT_STYLE


def args_for(sid):
    return SimpleNamespace(instance="Instance_2", n_frames=1, no_save_images=True, mask_dilation_mm=25.0, lax_thickness=6.0,
                           sax_thickness=8.0, line_extent_mm=150.0, line_step_mm=1.0, min_line_mm=1.0, outside_mask_weight=0.0,
                           metric="ncc", mi_bins=24, mi_sigma=1.0)


def brute_trace(lax_img: np.ndarray, lax_aff: np.ndarray, other_aff: np.ndarray, k: float) -> np.ndarray:
    """(H, W) field: the other image's through-plane voxel coordinate at every LAX voxel, minus k (zero on the trace)."""
    H, W = lax_img.shape
    ii, jj = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    P = np.stack([ii.ravel(), jj.ravel(), np.zeros(ii.size), np.ones(ii.size)], 1) @ lax_aff.T
    q = P @ np.linalg.inv(other_aff).T
    return (q[:, 2] - k).reshape(H, W)


def main():
    from contourpy import contour_generator
    OUT.mkdir(exist_ok=True)
    sids = sys.argv[1:] or sorted(p.name for p in ROOT.iterdir() if p.is_dir())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for sid in sids:
        args = args_for(sid)
        data = mg.load_subject(ROOT / sid, args)
        if data["status"] != "ok":
            print(sid, data["status"])
            continue
        sax_arr, sax_aff = data["sax"]
        sax_planes = mg.make_planes("SAX", sax_arr, sax_aff, args.sax_thickness, None, data["sax_seg"], device)
        lax_planes = []
        for view, d in data["lax"].items():
            lax_planes += mg.make_planes(view, d["arr"], d["affine"], d["thickness"], d["mask"], d["seg"], device)
        planes = sax_planes + lax_planes
        index = {p.key: i for i, p in enumerate(planes)}
        lines = []
        with torch.no_grad():
            for a in sax_planes:
                for b in lax_planes:
                    ln = mg.intersection_line(a, b, args, device)
                    if ln is not None:
                        lines.append((index[a.key], index[b.key], ln))
            for i in range(len(lax_planes)):
                for j in range(i + 1, len(lax_planes)):
                    ln = mg.intersection_line(lax_planes[i], lax_planes[j], args, device)
                    if ln is not None:
                        lines.append((index[lax_planes[i].key], index[lax_planes[j].key], ln))

        # --- 1. geometry: brute-force traces vs the lines
        worst = 0.0
        fig = Figure(figsize=(6.5 * len(lax_planes), 6.8), layout="constrained", facecolor=ST["surface"])
        axes = np.atleast_1d(fig.subplots(1, len(lax_planes)))
        for ax, lp in zip(axes, lax_planes):
            img = data["lax"][lp.image]["arr"][:, :, 0, 0]
            A_l = data["lax"][lp.image]["affine"]
            H, W = img.shape
            ax.imshow(img.T, cmap="gray", origin="lower", extent=(-0.5, H - 0.5, -0.5, W - 0.5))
            others = [(p.image, p.slice_idx, sax_aff, p.slice_idx) for p in sax_planes] + \
                     [(p.image, 0, data["lax"][p.image]["affine"], 0) for p in lax_planes if p is not lp]
            for name, z, aff, k in others:
                field = brute_trace(img, A_l, aff, k)
                for seg in contour_generator(z=field).lines(0.0):     # (x, y) = (j, i)
                    ax.plot(seg[:, 1], seg[:, 0], color=ST["yellow"] if name == "SAX" else ST["aqua"], lw=2.2, alpha=0.6)
                # the script's line for this pair, in LAX voxel coordinates
                for ia, ib, ln in lines:
                    pa, pb = planes[ia], planes[ib]
                    if {pa.key, pb.key} != {lp.key, f"{name}:{z}"}:
                        continue
                    w = ln.w.cpu().numpy() > 0
                    world = ln.x0[None] + ln.t.cpu().numpy()[:, None] * ln.d[None]
                    vox = world @ lp.inv_rot.cpu().numpy().T.astype(np.float64) + lp.inv_trans.cpu().numpy()
                    ax.plot(vox[:, 0], vox[:, 1], color=ST["blue"] if name == "SAX" else ST["orange"], lw=0.7)
                    ax.plot(vox[w, 0], vox[w, 1], ".", ms=2, color=ST["blue"] if name == "SAX" else ST["orange"])
                    # distance of the line's points from the brute-force trace: |field| / |grad field| at the points
                    fi, fj = np.gradient(field)
                    val = map_coordinates(field, [vox[:, 0], vox[:, 1]], order=1, mode="nearest")
                    gi = map_coordinates(fi, [vox[:, 0], vox[:, 1]], order=1, mode="nearest")
                    gj = map_coordinates(fj, [vox[:, 0], vox[:, 1]], order=1, mode="nearest")
                    inb = (vox[:, 0] >= 0) & (vox[:, 0] <= H - 1) & (vox[:, 1] >= 0) & (vox[:, 1] <= W - 1)
                    dist = np.abs(val[inb]) / np.hypot(gi[inb], gj[inb]).clip(1e-6)       # in voxels of the LAX image
                    if dist.size:
                        worst = max(worst, float(dist.max()) * lp.spacing[0])
            ax.set_title(f"{lp.image}: brute-force traces (thick) vs script lines (thin)", fontsize=10, loc="left")
            ax.set_xticks([]), ax.set_yticks([])
            ax.set_aspect("equal")
        fig.suptitle(f"Subject {sid}: SAX traces yellow / blue, LAX-LAX traces aqua / orange; worst distance {worst:.3f} mm",
                     x=0.01, ha="left", color=ST["ink"])
        fig.savefig(OUT / f"{sid}_traces.png", dpi=110)
        print(f"{sid}: {len(lines)} lines, worst line-to-trace distance {worst:.4f} mm")

        # --- 2. profiles vs an independent reslice (identity pose, rect PSF with one sample, so no band averaging)
        smp = mg.LineSampler(planes, lines, 1, device, "rect")
        zeros = torch.zeros(len(planes), 3, device=device)
        with torch.no_grad():
            pa, pb = smp.profiles(zeros, zeros, "img")
        pa, pb = pa[:, 0].cpu().numpy(), pb[:, 0].cpu().numpy()
        err = []
        for m, (ia, ib, ln) in enumerate(lines):
            world = ln.x0[None] + ln.t.cpu().numpy()[:, None] * ln.d[None]
            n = len(ln.t)
            for side, p in ((pa[m, :n], planes[ia]), (pb[m, :n], planes[ib])):
                vox = world @ p.inv_rot.cpu().numpy().T.astype(np.float64) + p.inv_trans.cpu().numpy()
                im = p.img[0, 0].cpu().numpy()
                ref = map_coordinates(im, [vox[:, 0], vox[:, 1]], order=1, mode="constant", cval=0.0)
                w = ln.w.cpu().numpy() > 0
                err.append(np.abs(side[w] - ref[w]).max() / max(1.0, np.abs(ref[w]).max()))
        print(f"{sid}: profiles vs map_coordinates, worst relative error {max(err):.2e}")


if __name__ == "__main__":
    main()
