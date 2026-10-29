"""
Motion correction of UK Biobank cine CMR stored as NIfTI (entry point; the motion correction is motion_gradient.py).

Input (--root): one folder per subject, any structure below it (see discover_subject), e.g. the download as is:
  <subject>/Instance_2/SAX.nii.gz, LAX_2Ch.nii.gz, LAX_3Ch.nii.gz, LAX_4Ch.nii.gz, nnUNet_segs/<stem>_nnUNetSeg.nii.gz
Files are classified by name: short-axis images (SAX / SA / short axis in the name) as a stack (X, Y, Z, T) and/or
single-slice files (X, Y, 1, T), all forming the image 'SAX'; every other NIfTI is a long-axis image (2Ch / 3Ch / 4Ch
-> LAX_2Ch / LAX_3Ch / LAX_4Ch, other names kept), a single slice or a stack; segmentations ('seg' / 'label' in the
file or a folder name) are matched to their image by name, SAX slices by position. Segmentations are optional: a LAX
image's segmentation defines its dilated heart mask; without one its lines run over the whole image overlap.
NIfTI headers store the slice spacing, not the thickness: --sax-thickness / --lax-thickness. Complex NIfTIs are
reduced to their magnitude.

Each slice is handed to motion_gradient.py as a 2D+time cine (motion_gradient.SliceImage) with its own affine.

Output (<out-root>/<run-name>/<subject>/), every slice's full cine with its corrected affine:
  SAX_slice<nn>.nii.gz, <image>.nii.gz (single-slice LAX views) or <image>_slice<nn>.nii.gz (LAX stacks),
  nnUNet_segs/<stem>_nnUNetSeg.nii.gz (segmentations), plus the metrics of motion_gradient.py (--logging).

Usage: python moco_nifti.py [--root DIR] [--out-root DIR] [--run-name NAME] [--sax-thickness MM] [--lax-thickness MM]
                            [--image-dtype uint16|uint8] [motion correction options, see --help]
"""
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

import motion_gradient as mg

NII_SUFFIXES = (".nii.gz", ".nii")
SEG_DIR = "nnUNet_segs"
SEG_WORD = re.compile(r"seg|label", re.I)                                   # marks a segmentation (file or folder name)
SEG_SUFFIX = re.compile(r"[_\-. ]*(nnunet)?[_\-. ]*(seg(mentation)?s?|labels?)$", re.I)
SAX_NAME = re.compile(r"(^|[^a-z])(sax|sa|short[_\- ]?axis)([^a-z]|$)", re.I)
LAX_CH_NAME = re.compile(r"([234])[_\- ]?ch(?![a-z])", re.I)
SAME_SLICE_MM = 0.5                         # SAX slices whose origins are closer than this are the same slice


def load_nifti(path: Path, dtype=np.float32) -> tuple[np.ndarray, np.ndarray, nib.Nifti1Header]:
    """Full 4D array (X, Y, Z, T), affine and header. Integer dtypes are rounded, not truncated. Complex data
    (complex64 / complex128) is reduced to its magnitude."""
    img = nib.load(str(path))
    raw = np.asarray(img.dataobj)
    arr = np.abs(raw).astype(np.float32) if np.iscomplexobj(raw) else raw.astype(np.float32)
    if np.issubdtype(dtype, np.integer):
        arr = np.rint(arr).astype(dtype)                                   # scaled label files store 1 as 0.9999999
    if arr.ndim == 3:
        arr = arr[..., None]
    return arr, img.affine.astype(np.float64), img.header.copy()


def nii_stem(p: Path) -> str:
    return next(p.name[:-len(s)] for s in NII_SUFFIXES if p.name.lower().endswith(s))


def image_kind(stem: str) -> str:
    """The image name of a file name stem without any segmentation suffix: 'SAX' (all short-axis files form one SAX
    image), 'LAX_2Ch' / 'LAX_3Ch' / 'LAX_4Ch', 'LAX_<stem>' for another long-axis name, otherwise the stem itself."""
    m = LAX_CH_NAME.search(stem)
    if m:
        return f"LAX_{m.group(1)}Ch"
    if SAX_NAME.search(stem):
        return "SAX"
    if re.search(r"lax|long[_\- ]?axis", stem, re.I):
        return stem if stem.upper().startswith("LAX") else f"LAX_{stem}"
    return stem.replace(":", "_")


def discover_subject(subj_dir: Path) -> dict:
    """Find a subject's NIfTI images and segmentations anywhere below subj_dir, whatever the folder structure (e.g. the
    download's <subject>/Instance_2/SAX.nii.gz + nnUNet_segs/, or single-slice files SAX_slice<nn>.nii.gz). A NIfTI is
    a segmentation if its name or a folder below subj_dir contains 'seg' or 'label'; the image it belongs to follows
    from the name without that suffix (..._nnUNetSeg, _seg, _label, ...). Hidden files and folders are skipped; if a
    LAX name occurs twice, the file nearest to subj_dir wins. Only headers are read.
    Returns {sax, sax_seg: [paths], lax, lax_seg: {name: path}, order: [image names, nearest first]}."""
    files = [p for p in subj_dir.rglob("*") if p.is_file() and p.name.lower().endswith(NII_SUFFIXES)
             and not any(r.startswith(".") for r in p.relative_to(subj_dir).parts)]
    files.sort(key=lambda p: (len(p.relative_to(subj_dir).parts), str(p).lower()))
    out = dict(sax=[], sax_seg=[], lax={}, lax_seg={}, order=[])
    for p in files:
        stem = nii_stem(p)
        is_seg = bool(SEG_WORD.search(stem)) or any(SEG_WORD.search(d) for d in p.relative_to(subj_dir).parts[:-1])
        kind = image_kind(SEG_SUFFIX.sub("", stem) if is_seg else stem)
        if kind == "SAX":
            out["sax_seg" if is_seg else "sax"].append(p)
        else:
            out["lax_seg" if is_seg else "lax"].setdefault(kind, p)
        if not is_seg and kind not in out["order"]:
            out["order"].append(kind)
    return out


def slice_affines(path: Path) -> tuple[np.ndarray, tuple]:
    """Per-slice affines (Z, 4, 4) -- the file's affine with the origin moved to slice z -- and the in-plane shape."""
    img = nib.load(str(path))
    A, shp = img.affine.astype(np.float64), img.shape
    Z = shp[2] if len(shp) >= 3 else 1
    out = np.repeat(A[None], Z, 0)
    out[:, :3, 3] = (A @ np.stack([np.zeros(Z), np.zeros(Z), np.arange(Z), np.ones(Z)]))[:3].T
    return out, tuple(shp[:2])


def sax_slice_list(paths: list[Path]) -> list[dict]:
    """Every SAX slice in the given files (stacks are split), ordered along the stacking direction and indexed 0, 1,
    ...: {index, path, z (slice in the file), affine, shape (X, Y)}. The order is by position along the mean slice axis
    (affine column 2), which reproduces a stack's z order. Slices found twice (same origin and size, e.g. a stack and
    its split slices) are kept once."""
    slices = []
    for p in paths:
        affines, shape = slice_affines(p)
        slices += [dict(path=p, z=z, affine=A, shape=shape) for z, A in enumerate(affines)]
    if not slices:
        return []
    axes = [s["affine"][:3, 2] / max(np.linalg.norm(s["affine"][:3, 2]), 1e-9) for s in slices]
    axis = np.mean([a if a @ axes[0] >= 0 else -a for a in axes], 0)
    slices.sort(key=lambda s: float(s["affine"][:3, 3] @ axis))
    kept = []
    for s in slices:
        if not any(np.linalg.norm(s["affine"][:3, 3] - k["affine"][:3, 3]) < SAME_SLICE_MM and s["shape"] == k["shape"]
                   for k in kept):
            kept.append(s)
    for i, s in enumerate(kept):
        s["index"] = i
    return kept


def output_stem(image: str, z: int, n: int) -> str:
    """Output file stem of slice z of an image with n slices: SAX_slice<nn> for SAX, the image name for a single slice,
    <image>_slice<nn> for the slices of a LAX stack."""
    if image == "SAX":
        return f"SAX_slice{z:02d}"
    return image if n == 1 else f"{image}_slice{z:02d}"


def load_subject(subj_dir: Path, args) -> mg.Subject:
    """One subject folder -> motion_gradient.Subject: one SliceImage (full (X, Y, T) cine, own affine) per SAX slice
    and per slice of every LAX image, with its segmentation where present. Each slice's `source` holds what save()
    needs: its output file stem and the NIfTI headers."""
    found = discover_subject(subj_dir)
    cache = {}

    def load(path, dtype=np.float32):                     # each file read once (a stack serves all its slices)
        if (path, dtype) not in cache:
            cache[(path, dtype)] = load_nifti(path, dtype=dtype)
        return cache[(path, dtype)]

    per_image = {}
    seg_slices = sax_slice_list(found["sax_seg"])
    sax = sax_slice_list(found["sax"])
    for s in sax:
        arr, _, header = load(s["path"])
        seg = seg_header = None
        match = next((g for g in seg_slices if g["shape"] == s["shape"]
                      and np.linalg.norm(g["affine"][:3, 3] - s["affine"][:3, 3]) < SAME_SLICE_MM), None)
        if match is not None:
            seg4, _, seg_header = load(match["path"], np.int16)
            if seg4.shape[3] == arr.shape[3]:
                seg = seg4[:, :, match["z"]]
        per_image.setdefault("SAX", []).append(mg.SliceImage(
            image="SAX", index=s["index"], data=arr[:, :, s["z"]], affine=s["affine"], thickness=args.sax_thickness,
            seg=seg, source=dict(stem=output_stem("SAX", s["index"], len(sax)), header=header, seg_header=seg_header)))

    for view, path in found["lax"].items():
        arr, _, header = load(path)
        affines, _ = slice_affines(path)
        seg = seg_header = None
        if view in found["lax_seg"]:
            seg4, _, seg_header = load(found["lax_seg"][view], np.int16)
            if seg4.shape == arr.shape:
                seg = seg4.copy()
                if view == "LAX_2Ch":
                    seg[seg == 3] = 4                            # 2Ch has no class 3; its class 3 is class 4 elsewhere
        thickness = args.lax_thickness if args.lax_thickness else float(np.linalg.norm(affines[0][:3, 2]))
        Z = arr.shape[2]
        per_image[view] = [mg.SliceImage(
            image=view, index=z, data=arr[:, :, z], affine=affines[z], thickness=thickness,
            seg=None if seg is None else seg[:, :, z],
            source=dict(stem=output_stem(view, z, Z), header=header, seg_header=seg_header)) for z in range(Z)]

    slices = [s for image in found["order"] for s in per_image.get(image, [])]
    return mg.Subject(subject_id=subj_dir.name, slices=slices, status="ok" if slices else "no_images")


def write_nifti(vol: np.ndarray, affine: np.ndarray, header: nib.Nifti1Header, path: Path, dtype,
                descrip: str | None = None) -> None:
    if np.issubdtype(dtype, np.unsignedinteger):
        vol = np.clip(np.rint(vol), 0, np.iinfo(dtype).max)
    img = nib.Nifti1Image(np.ascontiguousarray(vol).astype(dtype), affine, header=header)
    img.set_data_dtype(dtype)
    img.header.set_slope_inter(None, None)
    if descrip is not None:
        img.header["descrip"] = descrip.encode()[:79]
    img.set_qform(affine, code=1)
    img.set_sform(affine, code=1)
    nib.save(img, str(path))


def uint8_clip_value(path) -> float | None:
    """Clip value of an image saved with --image-dtype uint8 (original intensity = value * clip / 255), else None."""
    d = nib.load(str(path)).header["descrip"].item().decode(errors="ignore")
    m = re.search(r"uint8 clip=([0-9.eE+-]+)", d)
    return float(m.group(1)) if m else None


def save(subject: mg.Subject, affines: list[np.ndarray], out: Path, args) -> None:
    """Write every slice's full cine (X, Y, 1, T) with the given affine to out/<stem>.nii.gz, its segmentation to
    out/nnUNet_segs/<stem>_nnUNetSeg.nii.gz (uint8 labels)."""
    (out / SEG_DIR).mkdir(parents=True, exist_ok=True)
    dtype, scale, descrip = np.uint16, 1.0, None
    if args.image_dtype == "uint8" and subject.slices:
        # one scale per subject (SAX and LAX share it): clip at --uint8-percentile of all its slices and frames
        clip = float(np.percentile(np.concatenate([s.data.ravel() for s in subject.slices]), args.uint8_percentile))
        dtype, scale = np.uint8, 255.0 / max(clip, 1e-6)
        # header descrip: original intensity = value * clip / 255 (below the clip); read with uint8_clip_value()
        descrip = f"uint8 clip={clip:.6g} pct={args.uint8_percentile:g} orig=v*clip/255"
    for s, A in zip(subject.slices, affines):
        vol = s.data[:, :, None, :]
        write_nifti(vol if dtype is np.uint16 else vol.astype(np.float32) * scale, A, s.source["header"],
                    out / f"{s.source['stem']}.nii.gz", dtype, descrip)   # uint8: clipped to [0, 255] in write_nifti
        if s.seg is not None:
            write_nifti(s.seg[:, :, None, :], A, s.source["seg_header"],
                        out / SEG_DIR / f"{s.source['stem']}_nnUNetSeg.nii.gz", np.uint8)


def is_complete(subj_in: Path, subj_out: Path) -> bool:
    """True when every image slice found in the input has its output file in subj_out (SAX_slice<nn>.nii.gz,
    <image>.nii.gz, <image>_slice<nn>.nii.gz)."""
    found = discover_subject(subj_in)
    names = [output_stem("SAX", s["index"], 0) for s in sax_slice_list(found["sax"])]
    for image, path in found["lax"].items():
        n = len(slice_affines(path)[0])
        names += [output_stem(image, z, n) for z in range(n)]
    return bool(names) and all((subj_out / f"{n}.nii.gz").exists() for n in names)


def main():
    ap = mg.build_parser(__doc__)
    g = ap.add_argument_group("paths")
    g.add_argument("--root", type=Path, default=Path(r"C:\Users\Nil\Documents\data\ukbb"),
                   help="Directory containing one folder per subject, any structure below it (see discover_subject). "
                        "Default: %(default)s")
    g.add_argument("--out-root", type=Path, default=Path(r"C:\Users\Nil\Documents\data\ukbb_alignment"),
                   help="Parent of the run directory. Default: %(default)s")
    g.add_argument("--run-name", type=str, default=None,
                   help="Run directory name, used as given (pass an existing run's name to add to or resume it, e.g. with "
                        "--skip-processed). Default: the start time followed by the settings given on the command line, "
                        "so every run gets a new directory")
    g = ap.add_argument_group("run")
    g.add_argument("--max-subjects", type=int, default=-1,
                   help="Max number of subjects to process (-1 = all). Default: %(default)s")
    g.add_argument("--skip-processed", action="store_true",
                   help="Skip subjects whose aligned images are already complete in the run directory (to resume a run, "
                        "pass its directory name with --run-name; a default name is always new)")
    g.add_argument("--logging", type=mg.str2bool, default=True, metavar="BOOL",
                   help="true (default): write everything besides the aligned images: the pose/metric history during "
                        f"optimization (every {mg.HISTORY_EVERY:g} of the run), per-subject CSVs and figures (metrics/), "
                        "cohort-wide CSVs and figures (overall_metrics/) and run_config.json. false: only the "
                        "motion-corrected images are written and no metrics are computed (fastest). "
                        "overall_metrics/sax_relative_motion.csv and run_config.json are written either way.")
    g = ap.add_argument_group("NIfTI data")
    g.add_argument("--sax-thickness", type=float, default=8.0,
                   help="SAX slice thickness in mm. The NIfTI header only stores the slice spacing (10 mm = 8 mm slice "
                        "+ 2 mm gap in the UK Biobank protocol). Default: %(default)s")
    g.add_argument("--lax-thickness", type=float, default=6.0,
                   help="Slice thickness in mm of every LAX image (UK Biobank protocol: 6 mm; the header's 7.5 mm is the "
                        "spacing). 0 = the header's slice spacing. Default: %(default)s")
    g.add_argument("--image-dtype", choices=["uint16", "uint8"], default="uint16",
                   help="Data type of the saved images. uint16: the acquired intensities unchanged. uint8: ~30%% less disk "
                        "space per subject (the NIfTIs are gzip-compressed and the segmentations are unchanged); per "
                        "subject, intensities are clipped at --uint8-percentile of all its slices and frames and scaled "
                        "to [0, 255] (one scale per subject; the clip value is stored in the header's descrip, see "
                        "uint8_clip_value). Segmentations are always uint8 labels")
    g.add_argument("--uint8-percentile", type=float, default=99.9,
                   help="Clip percentile for --image-dtype uint8. On the UKBB cohort the max is ~1.25x the 99.9th "
                        "percentile (a few bright pixels), so 99.9 clips 0.1%% of the voxels")
    g = ap.add_argument_group("segmentation metrics (reporting only, with --logging)")
    g.add_argument("--seg-labels", type=str, default="1,2,3,4,5",
                   help="nnU-Net labels compared (Dice / Hausdorff) along LAX-LAX intersections. Default: %(default)s")
    g.add_argument("--sax-seg-labels", type=str, default="1,2,3",
                   help="nnU-Net labels compared along SAX-LAX intersections (the SAX model has fewer classes). "
                        "Default: %(default)s")
    g.add_argument("--hd-miss-mm", type=float, default=100.0,
                   help="Hausdorff value (mm) when both planes contain a label but only one intersection profile hits "
                        "it. Default: %(default)s")
    g = ap.add_argument_group("hardware")
    g.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                   help="Device for the alignment. Default: %(default)s")
    g.add_argument("--workers", type=int, default=4,
                   help="Background threads for loading and writing image files. Default: %(default)s")
    g.add_argument("--no-cuda-graph", action="store_true",
                   help="Run each optimization step eagerly instead of replaying a captured CUDA graph (same result, "
                        "slower; for debugging)")
    g.add_argument("--sample-dtype", choices=["float32", "float16", "bfloat16"], default="float32",
                   help="Precision of the images held on the GPU and of their sampling along the intersection lines (the "
                        "bulk of the memory for many slices): float16 halves the image memory and the sampled tensors; it "
                        "holds integer intensities up to 2048 exactly (brighter images are scaled by a power of two; the "
                        "metrics are scale-invariant) and resolves sample positions to ~0.05 pixel on a 200-pixel image. "
                        "bfloat16 only to ~0.4 pixel and ~0.4%% in intensity (coarse for sub-mm motion). Poses, line "
                        "geometry and metrics stay float32; the saved images are unaffected (see --image-dtype). "
                        "Default: %(default)s")
    args = ap.parse_args()
    mg.run(ap, args, load_subject, save, is_complete)


if __name__ == "__main__":
    main()
