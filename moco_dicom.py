"""
Motion correction of cine CMR stored as DICOM (entry point; the motion correction is motion_gradient.py).

Input (--root): one folder per subject; every .dcm / .ima file below it is read (hidden files and folders skipped).
  - An image is one series (the files sharing a SeriesInstanceUID), named after the deepest folder holding all its
    files (e.g. <subject>/<series>/phase_<nn>/*.dcm -> <series>; files directly in the subject folder:
    SeriesDescription_SeriesNumber).
  - Its slices are the distinct ImagePositionPatient / ImageOrientationPatient, ordered along the slice normal; its
    cardiac frames are the files at one slice, ordered by TriggerTime, then InstanceNumber, then path. The folder layout
    of the files does not matter.
  - Slice thickness: SliceThickness (--thickness if the header has none). RescaleSlope / Intercept are applied.
  - Complex data: REAL / IMAGINARY image pairs (ComplexImageComponent, or ImageType R / I) are combined into the
    magnitude; PHASE images are ignored; if a series has MAGNITUDE images, only those are used.
  - No segmentations: every image takes part unmasked (lines over the whole image overlap).
Geometry is converted from DICOM's LPS to RAS; arrays use the NIfTI convention arr[x, y] = pixel (row y, column x).
Each slice is handed to motion_gradient.py as a 2D+time cine (motion_gradient.SliceImage) with its own affine.

Output (<out-root>/<run-name>/<subject>/): every input file rewritten at the same relative path with the corrected
ImagePositionPatient / ImageOrientationPatient / SliceLocation, new SOP / series UIDs (derived from the originals) and
SeriesDescription + ' aligned'; pixel data unchanged (magnitudes are written for real/imaginary input;
--image-dtype uint8: 8-bit with RescaleSlope / Intercept). Plus the
metrics of motion_gradient.py (--logging).

Usage: python moco_dicom.py --root DIR [--out-root DIR] [--run-name NAME] [--thickness MM]
                            [--image-dtype original|uint8] [motion correction options, see --help]
Needs pydicom (pip install pydicom).
"""
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import pydicom
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

import motion_gradient as mg

DCM_SUFFIXES = (".dcm", ".ima")
LPS = np.diag([-1.0, -1.0, 1.0, 1.0])                  # RAS <-> LPS (its own inverse)
SAME_POS_MM = 1e-3                                       # files closer than this (and same orientation) = one slice
SAME_IOP = 1e-5


@dataclass(frozen=True)
class DicomSeries:
    name: str                                           # image name: the series' folder (see dicom_series)
    files: tuple                                        # its .dcm files (Paths)


@dataclass
class DicomSlice:
    """Where one slice came from: per frame, its source file(s) (two for a real/imaginary pair)."""
    files: list
    subj_dir: Path


def dicom_files(subj_dir: Path) -> list[Path]:
    return sorted(p for p in subj_dir.rglob("*") if p.is_file() and p.name.lower().endswith(DCM_SUFFIXES)
                  and not any(r.startswith(".") for r in p.relative_to(subj_dir).parts))


def dicom_component(ds) -> str:
    """'M' (magnitude), 'P' (phase), 'R' (real) or 'I' (imaginary) of a DICOM image; 'M' if not stated."""
    c = str(ds.get("ComplexImageComponent", "") or "").upper()
    if c in ("MAGNITUDE", "PHASE", "REAL", "IMAGINARY"):
        return c[0]
    it = [str(x).upper() for x in (ds.get("ImageType") or [])]
    return it[2] if len(it) > 2 and it[2] in ("M", "P", "R", "I") else "M"


def dicom_series(subj_dir: Path) -> list[DicomSeries]:
    """The subject's series (headers only), nearest to subj_dir first."""
    groups, meta = {}, {}
    for f in dicom_files(subj_dir):
        ds = pydicom.dcmread(str(f), stop_before_pixels=True,
                             specific_tags=["SeriesInstanceUID", "SeriesDescription", "SeriesNumber"])
        uid = str(ds.get("SeriesInstanceUID", f.parent))
        groups.setdefault(uid, []).append(f)
        meta.setdefault(uid, ds)
    out = []
    for uid, fs in groups.items():
        common = Path(os.path.commonpath([str(f.parent) for f in fs]))
        if common == subj_dir:
            ds = meta[uid]
            name = re.sub(r"[^A-Za-z0-9_\-]+", "_", f"{ds.get('SeriesDescription', 'series')}_{ds.get('SeriesNumber', '')}").strip("_")
        else:
            name = common.name
        out.append(DicomSeries(name=name, files=tuple(sorted(fs))))
    names = [s.name for s in out]
    out = [DicomSeries(f"{s.name}_{i}" if names.count(s.name) > 1 else s.name, s.files) for i, s in enumerate(out)]
    return sorted(out, key=lambda s: (len(s.files[0].relative_to(subj_dir).parts), s.name.lower()))


def dicom_layout(series: DicomSeries) -> dict:
    """Geometry of a series from its headers: {affines (Z, 4, 4) RAS, one per slice with the origin at its first pixel,
    shape (X, Y), files [z][t] -> tuple of paths (one magnitude file, or a real/imaginary pair), thickness
    (SliceThickness, None if absent)}. Slices are the files sharing position and orientation (so slices that are not
    parallel, e.g. of a previously aligned series, stay apart), ordered along the first file's normal."""
    hdrs = [(f, pydicom.dcmread(str(f), stop_before_pixels=True)) for f in series.files]
    comps = {f: dicom_component(d) for f, d in hdrs}
    use = "M" if "M" in comps.values() else "RI"
    hdrs = [(f, d) for f, d in hdrs if comps[f] in use]
    d0 = hdrs[0][1]
    dr, dc = (float(x) for x in d0.PixelSpacing)         # (between rows, between columns)
    by_slice = {}
    for f, d in hdrs:
        ipp = np.array(d.ImagePositionPatient, float)
        iop = np.array(d.ImageOrientationPatient, float)
        key = tuple(np.round(ipp / SAME_POS_MM).astype(int)) + tuple(np.round(iop / SAME_IOP).astype(int))
        by_slice.setdefault(key, dict(ipp=ipp, iop=iop, items=[]))["items"].append(
            (float(d.get("TriggerTime", 0) or 0), int(d.get("InstanceNumber", 0) or 0), str(f), comps[f], f))
    n0 = np.cross(*np.split(np.array(d0.ImageOrientationPatient, float), 2))
    affines, files = [], []
    for s in sorted(by_slice.values(), key=lambda s: float(s["ipp"] @ n0)):
        row_dir, col_dir = s["iop"][:3], s["iop"][3:]    # DICOM: along a row (columns increase), along a column
        A = np.eye(4)
        A[:3, 0], A[:3, 1], A[:3, 2] = row_dir * dc, col_dir * dr, np.cross(row_dir, col_dir)
        A[:3, 3] = s["ipp"]
        affines.append(LPS @ A)
        items = sorted(s["items"])
        if use == "M":
            files.append([(it[4],) for it in items])
        else:                                            # pair the k-th real with the k-th imaginary image
            re_, im_ = [it[4] for it in items if it[3] == "R"], [it[4] for it in items if it[3] == "I"]
            files.append(list(zip(re_, im_)))
    nt = min(len(f) for f in files)
    if any(len(f) != nt for f in files):
        print(f"WARNING {series.name}: slices have different frame counts, keeping the first {nt}")
    affines = np.stack(affines)
    if len(affines) > 1:                                 # slice axis column: each slice's normal x the mean spacing
        spacing = abs(float(np.diff(affines[:, :3, 3] @ (LPS[:3, :3] @ n0)).mean()))
        affines[:, :3, 2] *= spacing if spacing > 1e-6 else 1.0
    th = d0.get("SliceThickness")
    return dict(affines=affines, shape=(int(d0.Columns), int(d0.Rows)), files=[f[:nt] for f in files],
                thickness=float(th) if th not in (None, "") else None)


def read_pixels(path: Path) -> np.ndarray:
    """One DICOM image as float32 (x = column, y = row), RescaleSlope / Intercept applied."""
    d = pydicom.dcmread(str(path))
    a = d.pixel_array.astype(np.float32) * float(d.get("RescaleSlope", 1) or 1) + float(d.get("RescaleIntercept", 0) or 0)
    return a.T


def load_subject(subj_dir: Path, args) -> mg.Subject:
    """One subject folder -> motion_gradient.Subject: one SliceImage (full (X, Y, T) cine, own affine) per slice of
    every series; `source` is the slice's DicomSlice (its files, for save())."""
    slices = []
    for series in dicom_series(subj_dir):
        lay = dicom_layout(series)
        X, Y = lay["shape"]
        thickness = lay["thickness"] or args.thickness
        if thickness is None:
            raise ValueError(f"{series.name}: no SliceThickness in the header; pass --thickness")
        for z, frames in enumerate(lay["files"]):
            data = np.zeros((X, Y, len(frames)), np.float32)
            for t, fs in enumerate(frames):
                data[:, :, t] = read_pixels(fs[0]) if len(fs) == 1 else np.hypot(read_pixels(fs[0]), read_pixels(fs[1]))
            slices.append(mg.SliceImage(image=series.name, index=z, data=data, affine=lay["affines"][z],
                                        thickness=thickness, source=DicomSlice(files=frames, subj_dir=subj_dir)))
    return mg.Subject(subject_id=subj_dir.name, slices=slices, status="ok" if slices else "no_images")


def write_slice(src: DicomSlice, data: np.ndarray, affine: np.ndarray, out_subj: Path,
                window: tuple[float, float] | None = None) -> None:
    """Write one slice's cine as DICOM, file for file at the input's relative paths under out_subj, with the geometry
    of `affine` (RAS): ImagePositionPatient / ImageOrientationPatient / SliceLocation, new SOP and series UIDs
    (derived from the originals), SeriesDescription + ' aligned'.
    Pixel data: copied unchanged (window None), except for a real/imaginary pair, whose magnitude (data[:, :, t]) is
    written in the source's integer type. With window (lo, clip) (--image-dtype uint8): data[:, :, t] clipped to
    [lo, clip] and scaled to 8-bit [0, 255], with RescaleSlope = (clip - lo) / 255 and RescaleIntercept = lo so that
    readers recover the original intensity scale."""
    A = LPS @ affine
    u, v = A[:3, 0] / np.linalg.norm(A[:3, 0]), A[:3, 1] / np.linalg.norm(A[:3, 1])
    n = np.cross(u, v)
    for t, fs in enumerate(src.files):
        ds = pydicom.dcmread(str(fs[0]))
        ds.ImagePositionPatient = [f"{x:.6f}" for x in A[:3, 3]]
        ds.ImageOrientationPatient = [f"{x:.8f}" for x in np.concatenate([u, v])]
        if "SliceLocation" in ds:
            ds.SliceLocation = f"{float(A[:3, 3] @ n):.6f}"
        ds.SeriesInstanceUID = generate_uid(entropy_srcs=[str(ds.SeriesInstanceUID), "aligned"])
        ds.SOPInstanceUID = generate_uid(entropy_srcs=[str(ds.SOPInstanceUID), "aligned"])
        ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        ds.SeriesDescription = f"{ds.get('SeriesDescription', '')} aligned".strip()
        if window is not None or len(fs) > 1:            # pixel data rewritten (uncompressed)
            img = data[:, :, t].T                        # back to (rows, cols), original intensity units
            for k in ("RescaleSlope", "RescaleIntercept"):
                if k in ds:
                    del ds[k]
            if window is not None:
                lo, clip = window
                step = max(clip - lo, 1e-6) / 255.0
                px = np.clip(np.rint((img - lo) / step), 0, 255).astype(np.uint8)
                ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 8, 8, 7, 0
                ds.RescaleSlope, ds.RescaleIntercept = f"{step:.8g}", f"{lo:.8g}"
            else:
                signed = int(ds.get("PixelRepresentation", 0))
                px = np.rint(img).astype(np.int16 if signed else np.uint16)
                ds.BitsAllocated, ds.BitsStored, ds.HighBit = 16, 16, 15
            if len(fs) > 1:
                ds.ComplexImageComponent = "MAGNITUDE"
            ds.PixelData = px.tobytes()
            ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        path = out_subj / Path(fs[0]).relative_to(src.subj_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        ds.save_as(str(path), enforce_file_format=True)


def save(subject: mg.Subject, affines: list[np.ndarray], out: Path, args) -> None:
    window = None
    if args.image_dtype == "uint8" and subject.slices:
        # one window per subject: from its minimum (signed data can be negative) to --uint8-percentile of all its
        # slices and frames
        values = np.concatenate([s.data.ravel() for s in subject.slices])
        window = (min(0.0, float(values.min())), float(np.percentile(values, args.uint8_percentile)))
    for s, A in zip(subject.slices, affines):
        write_slice(s.source, s.data, A, out, window)


def is_complete(subj_in: Path, subj_out: Path) -> bool:
    """True when every (used) input file has its output file at the same relative path."""
    names = [fs[0].relative_to(subj_in) for series in dicom_series(subj_in) for z in dicom_layout(series)["files"]
             for fs in z]
    return bool(names) and all((subj_out / n).exists() for n in names)


def main():
    ap = mg.build_parser(__doc__)
    g = ap.add_argument_group("paths")
    g.add_argument("--root", type=Path, required=True,
                   help="Directory containing one folder per subject (DICOM files anywhere below each)")
    g.add_argument("--out-root", type=Path, default=None,
                   help="Parent of the run directory. Default: <root>_aligned next to --root")
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
    g = ap.add_argument_group("DICOM data")
    g.add_argument("--thickness", type=float, default=None,
                   help="Slice thickness (mm) for series whose headers have no SliceThickness (else an error)")
    g.add_argument("--image-dtype", choices=["original", "uint8"], default="original",
                   help="Pixel data of the saved images. original: the input's pixel data unchanged (only the geometry "
                        "tags change). uint8: per subject, intensities from its minimum up to --uint8-percentile of all "
                        "its slices and frames (clipped above) are scaled to 8-bit [0, 255], written uncompressed with "
                        "RescaleSlope / RescaleIntercept (readers recover the original intensity scale). "
                        "Default: %(default)s")
    g.add_argument("--uint8-percentile", type=float, default=99.9,
                   help="Clip percentile for --image-dtype uint8 (99.9 clips 0.1%% of the voxels). Default: %(default)s")
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
    if args.out_root is None:
        args.out_root = args.root.parent / f"{args.root.name}_aligned"
    mg.run(ap, args, load_subject, save, is_complete)


if __name__ == "__main__":
    main()
