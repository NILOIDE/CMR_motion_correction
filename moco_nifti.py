"""
Motion correction of UK Biobank cine CMR stored as NIfTI (entry point; the motion correction is motion_gradient.py).

Input (--root): one folder per subject. Below it, every folder that directly contains NIfTI images is one modality
(an acquisition: cine, t1, lge, ...; the UKBB download's Instance_2/ is one), see discover_subject:
  <subject>/<modality>/<images>.nii.gz            slices and stacks
  <subject>/<modality>/<seg folder>/<segs>.nii.gz  optional; the folder name must contain 'seg'
e.g. the UKBB download as is: <subject>/Instance_2/SAX.nii.gz, LAX_2Ch.nii.gz, ..., nnUNet_segs/LAX_2Ch_nnUNetSeg.nii.gz.
A modality folder may hold no other subfolder than one segmentation folder (otherwise the subject fails with an
error). Within a modality, short-axis images (SAX / SA / short axis in the name) form its SAX stack (a stack and/or
single-slice files); every other file is an image of its own (a single slice or a stack), named after the file, its
view from the name (2Ch / 3Ch / 4Ch -> LAX_2Ch / LAX_3Ch / LAX_4Ch; ao; else the name). Several modalities may each
have their own SAX stack and views; slices are paired by orientation whatever their views (motion_gradient.py), and
compared with --metric within a modality, --cross-mod-metric between modalities. A segmentation belongs to the image
whose file name is the longest prefix of its own (a warning when several match), the SAX slices to theirs by
position; segmentations matching no image (e.g. the UKBB seg_sa.nii.gz) or not stored as integers are ignored.
Segmentations are optional: a LAX image's segmentation defines its dilated heart mask; without one its lines run over
the whole image overlap.
NIfTI headers store the slice spacing, not the thickness: --sax-thickness / --lax-thickness. Complex NIfTIs are
reduced to their magnitude.

Axes: a 4D file is (X, Y, Z, T); a 3D file is one slice (X, Y, T); a 2D file one slice with one frame (X, Y). A slice
with a single frame is taken as end-diastole (frame 0); motion_gradient.py compares it with the others at that frame
only. A segmentation may have a single frame (end-diastole) next to an image with several; other frame counts that
differ from the image's leave the segmentation out.

Each slice is handed to motion_gradient.py as a 2D+time cine (motion_gradient.SliceImage) with its own affine.

Output (<out-root>/<run-name>/<subject>/; with several modalities in <modality>/ subfolders), every slice's full cine
with its corrected affine:
  SAX_slice<nn>.nii.gz, <image>.nii.gz (single-slice images) or <image>_slice<nn>.nii.gz (other stacks),
  nnUNet_segs/<stem>_nnUNetSeg.nii.gz (segmentations), plus the metrics of motion_gradient.py (--logging).

Usage: python moco_nifti.py [--root DIR] [--out-root DIR] [--run-name NAME] [--sax-thickness MM] [--lax-thickness MM]
                            [--image-dtype uint16|uint8] [motion correction options, see --help]
"""
import os
import re
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

import motion_gradient as mg

NII_SUFFIXES = (".nii.gz", ".nii")
SEG_DIR = "nnUNet_segs"                     # the segmentation folder of the output
SEG_FOLDER = "seg"                          # a modality folder's segmentation folder has this in its name
SAX_NAME = re.compile(r"(^|[^a-z])(sax|sa|short[_\- ]?axis)([^a-z]|$)", re.I)
LAX_CH_NAME = re.compile(r"([234])[_\- ]?ch(?![a-z])", re.I)
SAME_SLICE_MM = 0.5                         # SAX slices whose origins are closer than this are the same slice


def load_nifti(path: Path, dtype=np.float32) -> tuple[np.ndarray, np.ndarray, nib.Nifti1Header]:
    """Full 4D array (X, Y, Z, T), affine and header. A 4D file is (X, Y, Z, T); a 3D file is one slice (X, Y, T) ->
    (X, Y, 1, T); a 2D file one slice with one frame (X, Y) -> (X, Y, 1, 1) (a single frame is taken as end-diastole).
    Trailing singleton axes beyond the 4th are dropped. Integer dtypes are rounded, not truncated. Complex data
    (complex64 / complex128) is reduced to its magnitude."""
    img = nib.load(str(path))
    raw = np.asarray(img.dataobj)
    arr = np.abs(raw).astype(np.float32) if np.iscomplexobj(raw) else raw.astype(np.float32)
    if np.issubdtype(dtype, np.integer):
        arr = np.rint(arr).astype(dtype)                                   # scaled label files store 1 as 0.9999999
    if arr.ndim > 4:
        if any(n != 1 for n in arr.shape[4:]):
            raise ValueError(f"{path}: {arr.ndim}D image {arr.shape}; expected (X, Y, Z, T) at most")
        arr = arr.reshape(arr.shape[:4])
    if arr.ndim == 3:                                      # one slice (X, Y, T) -> (X, Y, 1, T)
        arr = arr[:, :, None, :]
    elif arr.ndim == 2:                                    # one slice, one frame (X, Y) -> (X, Y, 1, 1)
        arr = arr[:, :, None, None]
    return arr, img.affine.astype(np.float64), img.header.copy()


def seg_frames_match(seg_frames: int, image_frames: int) -> bool:
    """Whether a segmentation with seg_frames frames can go with an image with image_frames: the same count, or a
    single frame (end-diastole, frame 0)."""
    return seg_frames in (1, image_frames)


def nii_stem(p: Path) -> str:
    return next(p.name[:-len(s)] for s in NII_SUFFIXES if p.name.lower().endswith(s))


def image_kind(stem: str) -> str:
    """The view of an image file name stem: 'SAX' (short-axis files form the modality's SAX stack), 'LAX_2Ch' /
    'LAX_3Ch' / 'LAX_4Ch', 'LAX_<stem>' for another long-axis name, otherwise the stem itself (e.g. 'ao')."""
    m = LAX_CH_NAME.search(stem)
    if m:
        return f"LAX_{m.group(1)}Ch"
    if SAX_NAME.search(stem):
        return "SAX"
    if re.search(r"lax|long[_\- ]?axis", stem, re.I):
        return stem if stem.upper().startswith("LAX") else f"LAX_{stem}"
    return stem.replace(":", "_")


def _warn(msg: str) -> None:
    print(f"WARNING {msg}", file=sys.stderr, flush=True)


def _niftis(folder: Path) -> list[Path]:
    return sorted((p for p in folder.iterdir() if p.is_file() and not p.name.startswith(".")
                   and p.name.lower().endswith(NII_SUFFIXES)), key=lambda p: p.name.lower())


def is_label_file(path: Path) -> bool:
    """Whether a NIfTI stores integers (a segmentation must; header only)."""
    return bool(np.issubdtype(nib.load(str(path)).get_data_dtype(), np.integer))


def discover_subject(subj_dir: Path) -> list[dict]:
    """A subject's modalities: every folder below subj_dir (or subj_dir itself) that directly contains NIfTI files and
    is not a segmentation folder, in folder order. In each:
      - every NIfTI is an image: short-axis ones (image_kind 'SAX') form its SAX stack, every other one is an image of
        its own named after its file (a warning for a file with 'seg' in its name: a segmentation outside its folder);
      - its only subfolder is its segmentation folder, whose name contains 'seg' (ValueError for any other subfolder
        and for a second one): each segmentation belongs to the image whose file name is the longest prefix of its own
        (case-insensitive; a warning when several are); segmentations matching no image or not stored as integers are
        ignored (the latter with a warning).
    Hidden files and folders are skipped; only headers are read. Returns [{name (the folder relative to subj_dir; ""
    for subj_dir itself), dir, sax, sax_seg: [paths], lax, lax_seg: {image stem: path}, views: {image stem: view},
    order: [image names: 'SAX' and the stems, in file order]}]."""
    out = []
    for folder, dirs, _ in os.walk(subj_dir):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        folder = Path(folder)
        if folder != subj_dir and SEG_FOLDER in folder.name.lower():
            dirs[:] = []                                   # a segmentation folder (read with its modality)
            continue
        files = _niftis(folder)
        if not files:
            continue
        rel = folder.relative_to(subj_dir).as_posix()
        name = "" if rel == "." else rel
        subs = [folder / d for d in dirs]
        bad = [d.name for d in subs if SEG_FOLDER not in d.name.lower()]
        if bad:
            raise ValueError(f"{subj_dir.name}: modality folder '{name or '.'}' contains folder(s) {bad} that are not a "
                             f"segmentation folder (its name must contain '{SEG_FOLDER}')")
        if len(subs) > 1:
            raise ValueError(f"{subj_dir.name}: modality folder '{name or '.'}' has {len(subs)} segmentation folders "
                             f"({[d.name for d in subs]}); one is allowed")
        dirs[:] = []                                       # (nothing below a modality folder is another modality)
        m = dict(name=name, dir=folder, sax=[], sax_seg=[], lax={}, lax_seg={}, views={}, order=[])
        stems = {}
        for f in files:
            stem = nii_stem(f)
            if SEG_FOLDER in stem.lower():
                _warn(f"{subj_dir.name}: {f.name} in '{name or '.'}' is named like a segmentation but is not in a "
                      f"segmentation folder (a subfolder with '{SEG_FOLDER}' in its name); loaded as an image")
            view = image_kind(stem)
            stems[stem] = view
            if view == "SAX":
                m["sax"].append(f)
                if "SAX" not in m["order"]:
                    m["order"].append("SAX")
            else:
                m["lax"][stem], m["views"][stem] = f, view
                m["order"].append(stem)
        for f in (_niftis(subs[0]) if subs else []):
            seg_stem = nii_stem(f).lower()
            match = sorted((st for st in stems if seg_stem.startswith(st.lower())), key=len, reverse=True)
            if not match:                                  # (e.g. the UKBB seg_sa.nii.gz)
                continue
            image = lambda st: "SAX" if stems[st] == "SAX" else st     # noqa: E731 (the SAX files are one image)
            if len({image(st) for st in match}) > 1:
                _warn(f"{subj_dir.name}: segmentation {f.name} matches images {match}; taken as {match[0]}'s")
            if not is_label_file(f):
                _warn(f"{subj_dir.name}: segmentation {f.name} is not stored as integers; ignored")
                continue
            if stems[match[0]] == "SAX":
                m["sax_seg"].append(f)
            elif match[0] in m["lax_seg"]:
                _warn(f"{subj_dir.name}: two segmentations for {match[0]}; {f.name} ignored")
            else:
                m["lax_seg"][match[0]] = f
        out.append(m)
    return out


def slice_affines(path: Path) -> tuple[np.ndarray, tuple]:
    """Per-slice affines (Z, 4, 4) -- the file's affine with the origin moved to slice z -- and the in-plane shape.
    Z is the 3rd axis of a 4D file (X, Y, Z, T); a 2D / 3D file ((X, Y), (X, Y, T)) is one slice."""
    img = nib.load(str(path))
    A, shp = img.affine.astype(np.float64), img.shape
    Z = shp[2] if len(shp) >= 4 else 1
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
    and per slice of every other image of every modality (discover_subject), with its segmentation where present, its
    view and its modality. With several modalities the images are named <modality>/<image>. Each slice's `source`
    holds what save() needs: its output folder and file stem and the NIfTI headers. ValueError for a folder structure
    that does not fit (see discover_subject)."""
    modalities = discover_subject(subj_dir)
    multi = len(modalities) > 1
    cache = {}

    def load(path, dtype=np.float32):                     # each file read once (a stack serves all its slices)
        if (path, dtype) not in cache:
            cache[(path, dtype)] = load_nifti(path, dtype=dtype)
        return cache[(path, dtype)]

    slices = []
    for m in modalities:
        prefix = f"{m['name']}/" if multi else ""
        sub = m["name"] if multi else ""
        per_image = {}
        seg_slices = sax_slice_list(m["sax_seg"])
        sax = sax_slice_list(m["sax"])
        for s in sax:
            arr, _, header = load(s["path"])
            seg = seg_header = None
            for g in seg_slices:                         # the first segmentation at this slice whose frames fit
                if g["shape"] != s["shape"] or np.linalg.norm(g["affine"][:3, 3] - s["affine"][:3, 3]) >= SAME_SLICE_MM:
                    continue
                seg4, _, g_header = load(g["path"], np.int16)
                if seg_frames_match(seg4.shape[3], arr.shape[3]):
                    seg, seg_header = seg4[:, :, g["z"]], g_header
                    break
            per_image.setdefault("SAX", []).append(mg.SliceImage(
                image=prefix + "SAX", index=s["index"], data=arr[:, :, s["z"]], affine=s["affine"],
                thickness=args.sax_thickness, seg=seg, view="SAX", modality=m["name"],
                source=dict(subdir=sub, stem=output_stem("SAX", s["index"], len(sax)), header=header,
                            seg_header=seg_header)))
        for stem, path in m["lax"].items():
            view = m["views"][stem]
            arr, _, header = load(path)
            affines, _ = slice_affines(path)
            seg = seg_header = None
            if stem in m["lax_seg"]:
                seg4, _, seg_header = load(m["lax_seg"][stem], np.int16)
                if seg4.shape[:3] == arr.shape[:3] and seg_frames_match(seg4.shape[3], arr.shape[3]):
                    seg = seg4.copy()
                    if view == "LAX_2Ch":
                        seg[seg == 3] = 4                        # 2Ch has no class 3; its class 3 is class 4 elsewhere
            thickness = args.lax_thickness if args.lax_thickness else float(np.linalg.norm(affines[0][:3, 2]))
            Z = arr.shape[2]
            per_image[stem] = [mg.SliceImage(
                image=prefix + stem, index=z, data=arr[:, :, z], affine=affines[z], thickness=thickness,
                seg=None if seg is None else seg[:, :, z], view=view, modality=m["name"],
                source=dict(subdir=sub, stem=output_stem(stem, z, Z), header=header, seg_header=seg_header))
                for z in range(Z)]
        slices += [s for image in m["order"] for s in per_image.get(image, [])]
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
    """Write every slice's full cine (X, Y, 1, T) with the given affine to out/<modality>/<stem>.nii.gz (out/<stem>.nii.gz
    with one modality), its segmentation to .../nnUNet_segs/<stem>_nnUNetSeg.nii.gz (uint8 labels)."""
    for sub in {s.source["subdir"] for s in subject.slices}:
        (out / sub / SEG_DIR).mkdir(parents=True, exist_ok=True)
    dtype, scale, descrip = np.uint16, 1.0, None
    if args.image_dtype == "uint8" and subject.slices:
        # one scale per subject (SAX and LAX share it): clip at --uint8-percentile of all its slices and frames
        clip = float(np.percentile(np.concatenate([s.data.ravel() for s in subject.slices]), args.uint8_percentile))
        dtype, scale = np.uint8, 255.0 / max(clip, 1e-6)
        # header descrip: original intensity = value * clip / 255 (below the clip); read with uint8_clip_value()
        descrip = f"uint8 clip={clip:.6g} pct={args.uint8_percentile:g} orig=v*clip/255"
    for s, A in zip(subject.slices, affines):
        vol = s.data[:, :, None, :]
        d = out / s.source["subdir"]
        write_nifti(vol if dtype is np.uint16 else vol.astype(np.float32) * scale, A, s.source["header"],
                    d / f"{s.source['stem']}.nii.gz", dtype, descrip)     # uint8: clipped to [0, 255] in write_nifti
        if s.seg is not None:
            write_nifti(s.seg[:, :, None, :], A, s.source["seg_header"],
                        d / SEG_DIR / f"{s.source['stem']}_nnUNetSeg.nii.gz", np.uint8)


def is_complete(subj_in: Path, subj_out: Path) -> bool:
    """True when every image slice found in the input has its output file in subj_out (SAX_slice<nn>.nii.gz,
    <image>.nii.gz, <image>_slice<nn>.nii.gz; in <modality>/ with several modalities)."""
    if not subj_out.is_dir():                            # not started: no need to read the input (one stat, not a
        return False                                     # folder walk + header reads on a network filesystem)
    modalities = discover_subject(subj_in)
    names = []
    for m in modalities:
        sub = m["name"] if len(modalities) > 1 else ""
        names += [(sub, output_stem("SAX", s["index"], 0)) for s in sax_slice_list(m["sax"])]
        for stem, path in m["lax"].items():
            n = len(slice_affines(path)[0])
            names += [(sub, output_stem(stem, z, n)) for z in range(n)]
    return bool(names) and all((subj_out / sub / f"{n}.nii.gz").exists() for sub, n in names)


def main():
    # Every option of motion_gradient.build_parser is available; the ones below are repeated here for visibility and
    # with defaults for UK Biobank NIfTI data (a repeated option replaces the general one, see build_parser).
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
    g.add_argument("--start-index", type=int, default=0,
                   help="First subject folder to process, as an index into the subject folders under --root sorted by "
                        "name (0-based; with --end-index, to submit the cohort in chunks). Default: %(default)s")
    g.add_argument("--end-index", type=int, default=-1,
                   help="Process only subject folders before this index (exclusive; -1 = to the end). "
                        "Default: %(default)s")
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
                   help="Worker processes for loading and preparing subjects and writing their images, metrics and "
                        "figures (CPU work, in parallel); up to about the number of CPU cores. Default: %(default)s")
    g.add_argument("--no-cuda-graph", action="store_true",
                   help="Run each optimization step eagerly instead of replaying a captured CUDA graph (same result, "
                        "slower; for debugging)")
    g.add_argument("--sample-dtype", choices=["float32", "float16", "bfloat16"], default="float16",
                   help="Precision of the images held on the GPU and of their sampling along the intersection lines (the "
                        "bulk of the memory for many slices): float16 halves the image memory and the sampled tensors; it "
                        "holds integer intensities up to 2048 exactly (brighter images are scaled by a power of two; the "
                        "metrics are scale-invariant) and resolves sample positions to ~0.05 pixel on a 200-pixel image. "
                        "bfloat16 only to ~0.4 pixel and ~0.4%% in intensity (coarse for sub-mm motion). Poses, line "
                        "geometry and metrics stay float32; the saved images are unaffected (see --image-dtype). "
                        "Default: %(default)s")
    mg.run(ap, ap.parse_args(), load_subject, save, is_complete)


if __name__ == "__main__":
    main()
