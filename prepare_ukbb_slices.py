"""
Convert the downloaded UK Biobank layout into the single-slice layout used by motion_gradient.py.

Source (--src, as downloaded):
  <src>/<subject>/[Instance_2/]SAX.nii.gz                        SAX stack (X, Y, Z, T)
  <src>/<subject>/[Instance_2/]LAX_2Ch.nii.gz, LAX_3Ch, LAX_4Ch  (X, Y, 1, T)
  <src>/<subject>/[Instance_2/]nnUNet_segs/<stem>_nnUNetSeg.nii.gz
Wrapper folders such as Instance_2 are found automatically: starting at the subject folder, a folder that holds no
NIfTI files and exactly one subfolder (other than nnUNet_segs) is descended into. They are dropped in the output.

Destination (--dst), no wrapper folder:
  <dst>/<subject>/SAX_slice00.nii.gz, SAX_slice01.nii.gz, ...   one file per SAX slice, (X, Y, 1, T)
  <dst>/<subject>/LAX_2Ch.nii.gz, LAX_3Ch.nii.gz, LAX_4Ch.nii.gz copied unchanged
  <dst>/<subject>/nnUNet_segs/<same stem>_nnUNetSeg.nii.gz         integer labels
  <dst>/prepare_log.csv                                          one row per subject (what was written)

Each SAX slice keeps the stack's voxel data, data type and header, with its own affine: the stack affine with the
origin moved to that slice (A[:3, 3] = A_stack @ [0, 0, z, 1]), written to the sform (and qform if the source has
one) with the source's codes. Image values are copied exactly. Segmentations (SAX slices and LAX views) are written
as plain int16 labels without scaling: the source label files are int16 with a slope/intercept (labels stored as
e.g. 0.9999999), which are rounded to the exact integer labels.

Missing inputs (the SAX stack, any LAX view, any segmentation) are not an error: the subject folder is always created
and whatever is present is written (the SAX stack split if present). The log's `missing` column lists the absent
images; status is ok (all present), partial (some missing), no_images, skipped_existing or error: <message>.

Subjects whose destination already holds every expected file are skipped unless --overwrite.

Each subject is written into a hidden folder <dst>/.<subject> and renamed to <dst>/<subject> once complete, so
motion_gradient.py can run on --dst at the same time: it ignores hidden folders and picks up new subjects as they
appear.

Usage: python prepare_ukbb_slices.py [--src DIR] [--dst DIR] [--max-subjects N] [--workers N] [--overwrite]
"""
import argparse
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

LAX_VIEWS = ("LAX_2Ch", "LAX_3Ch", "LAX_4Ch")
SEG_DIR = "nnUNet_segs"
TMP_PREFIX = "."                 # subjects are written as <dst>/.<subject> and renamed to <dst>/<subject> when complete


def is_scaled(img) -> bool:
    """nibabel moves scl_slope / scl_inter from the header to the data object on load, so check it there."""
    return (float(img.dataobj.slope), float(img.dataobj.inter)) != (1.0, 0.0)


def read_data(img, is_seg: bool, name: str, warnings: list) -> np.ndarray:
    """Segmentations: the integer labels (scaled values rounded; the source label files are int16 with a slope/
    intercept that stores 1 as 0.9999999). Images: the stored values exactly, or the scaled float32 values if the
    file has a non-trivial scaling (none in the current data)."""
    if is_seg:
        return np.rint(np.asanyarray(img.dataobj)).astype(np.int16)
    if is_scaled(img):
        warnings.append(f"{name}: scaled image (slope {float(img.dataobj.slope)}, inter {float(img.dataobj.inter)}); float32 written")
        return np.asanyarray(img.dataobj).astype(np.float32)
    return np.asanyarray(img.dataobj.get_unscaled())                 # the stored values, exactly


def save_like(data: np.ndarray, affine: np.ndarray, src_img, path: Path) -> None:
    """Save with the source header (zooms, units, descriptions), the given affine in the source's sform/qform codes,
    the data's own dtype and no scaling."""
    hdr = src_img.header
    q_code, s_code = int(hdr["qform_code"]), int(hdr["sform_code"]) or 1
    out = nib.Nifti1Image(np.ascontiguousarray(data), affine, header=hdr.copy())
    out.set_data_dtype(data.dtype)
    out.header.set_slope_inter(None, None)
    out.set_sform(affine, code=s_code)
    out.set_qform(affine if q_code else None, code=q_code)
    nib.save(out, str(path))


def split_stack(src: Path, dst_dir: Path, is_seg: bool) -> list[str]:
    """Write one (X, Y, 1, T) file per slice of a SAX stack; returns warnings."""
    img = nib.load(str(src))
    warnings = []
    data = read_data(img, is_seg, src.name, warnings)
    if data.ndim == 3:
        data = data[..., None]
    A_stack = img.affine
    for z in range(data.shape[2]):
        A = A_stack.copy()
        A[:3, 3] = (A_stack @ np.array([0.0, 0.0, z, 1.0]))[:3]
        name = f"SAX_slice{z:02d}_nnUNetSeg.nii.gz" if is_seg else f"SAX_slice{z:02d}.nii.gz"
        save_like(data[:, :, z:z + 1], A, img, dst_dir / name)
    return warnings


def copy_lax(src: Path, dst: Path, is_seg: bool) -> list[str]:
    """LAX images are copied byte for byte; LAX segmentations are rewritten as integer labels (see read_data)."""
    if not is_seg:
        shutil.copy2(src, dst)
        return []
    img = nib.load(str(src))
    warnings = []
    save_like(read_data(img, True, src.name, warnings), img.affine, img, dst)
    return warnings


def image_dir(subj: Path) -> Path:
    """The folder of a subject that holds its NIfTIs: descend through redundant wrapper folders (e.g. Instance_2),
    i.e. folders without NIfTI files that contain exactly one subfolder besides nnUNet_segs."""
    d = subj
    while not any(d.glob("*.nii.gz")):
        subdirs = [p for p in d.iterdir() if p.is_dir() and p.name != SEG_DIR]
        if len(subdirs) != 1:
            break
        d = subdirs[0]
    return d


def expected_outputs(src_inst: Path, dst: Path) -> list[Path]:
    files = []
    sax = src_inst / "SAX.nii.gz"
    if sax.exists():
        nz = nib.load(str(sax)).shape[2]
        files += [dst / f"SAX_slice{z:02d}.nii.gz" for z in range(nz)]
        if (src_inst / SEG_DIR / "SAX_nnUNetSeg.nii.gz").exists():
            files += [dst / SEG_DIR / f"SAX_slice{z:02d}_nnUNetSeg.nii.gz" for z in range(nz)]
    for v in LAX_VIEWS:
        if (src_inst / f"{v}.nii.gz").exists():
            files.append(dst / f"{v}.nii.gz")
        if (src_inst / SEG_DIR / f"{v}_nnUNetSeg.nii.gz").exists():
            files.append(dst / SEG_DIR / f"{v}_nnUNetSeg.nii.gz")
    return files


def prepare_subject(subj: Path, args) -> dict:
    t0 = time.time()
    sid = subj.name
    row = dict(subject_id=sid, status="ok", missing="", n_sax_slices=0, sax_seg=False, lax_views="", lax_segs="",
               warnings="", seconds=0.0)
    try:
        return _prepare_subject(subj, args, row, t0)
    except Exception as e:  # noqa: BLE001  one unreadable subject must not stop the others
        row.update(status=f"error: {type(e).__name__}: {e}"[:200], seconds=round(time.time() - t0, 1))
        return row


def _prepare_subject(subj: Path, args, row: dict, t0: float) -> dict:
    src_inst = image_dir(subj)
    dst = args.dst / subj.name
    missing = [name for name, f in [("SAX", "SAX.nii.gz")] + [(v, f"{v}.nii.gz") for v in LAX_VIEWS]
               if not (src_inst / f).exists()]
    row["missing"] = "+".join(missing)
    expected = expected_outputs(src_inst, dst)
    if expected and not args.overwrite and all(p.exists() for p in expected):
        row["status"] = "skipped_existing"
    else:
        # written into a hidden temporary folder that is renamed into place once complete, so that a concurrently
        # running motion_gradient.py (which ignores hidden folders) never picks up a half-written subject
        tmp = args.dst / f"{TMP_PREFIX}{subj.name}"
        if tmp.exists():
            shutil.rmtree(tmp)                           # left over from an interrupted run
        (tmp / SEG_DIR).mkdir(parents=True)              # the subject folder is created whatever is missing
        warnings = []
        if (src_inst / "SAX.nii.gz").exists():
            warnings += split_stack(src_inst / "SAX.nii.gz", tmp, is_seg=False)
            seg = src_inst / SEG_DIR / "SAX_nnUNetSeg.nii.gz"
            if seg.exists():
                warnings += split_stack(seg, tmp / SEG_DIR, is_seg=True)
        for v in LAX_VIEWS:
            for src, out, is_seg in ((src_inst / f"{v}.nii.gz", tmp / f"{v}.nii.gz", False),
                                     (src_inst / SEG_DIR / f"{v}_nnUNetSeg.nii.gz", tmp / SEG_DIR / f"{v}_nnUNetSeg.nii.gz", True)):
                if src.exists():
                    warnings += copy_lax(src, out, is_seg)
        if dst.exists():                                 # incomplete earlier output, or --overwrite
            shutil.rmtree(dst)
        tmp.rename(dst)
        row["warnings"] = "; ".join(warnings)
        if not expected:
            row["status"] = "no_images"
        elif missing:
            row["status"] = "partial"                    # whatever is present is still written
    row["n_sax_slices"] = sum(1 for p in expected if p.parent == dst and p.name.startswith("SAX_slice"))
    row["sax_seg"] = any(p.parent.name == SEG_DIR and p.name.startswith("SAX_slice") for p in expected)
    row["lax_views"] = "+".join(v for v in LAX_VIEWS if (dst / f"{v}.nii.gz") in expected)
    row["lax_segs"] = "+".join(v for v in LAX_VIEWS if (dst / SEG_DIR / f"{v}_nnUNetSeg.nii.gz") in expected)
    row["seconds"] = round(time.time() - t0, 1)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=Path(r"C:\Users\Nil\Documents\data\ukbb"), help="downloaded data, one folder per subject")
    ap.add_argument("--dst", type=Path, default=Path(r"C:\Users\Nil\Documents\data\ukbb_slices"), help="output root")
    ap.add_argument("--max-subjects", type=int, default=-1, help="process only the first N subjects (-1 = all)")
    ap.add_argument("--overwrite", action="store_true", help="rewrite subjects whose outputs already exist")
    ap.add_argument("--workers", type=int, default=4, help="subjects processed in parallel")
    args = ap.parse_args()

    subjects = sorted(p for p in args.src.iterdir() if p.is_dir())
    if args.max_subjects >= 0:
        subjects = subjects[:args.max_subjects]
    args.dst.mkdir(parents=True, exist_ok=True)
    print(f"{len(subjects)} subject folders in {args.src} -> {args.dst}")
    rows = []
    with ThreadPoolExecutor(args.workers) as ex:
        for row in ex.map(lambda s: prepare_subject(s, args), subjects):
            rows.append(row)
            note = f"  WARN {row['warnings']}" if row["warnings"] else ""
            note += f"  missing {row['missing']}" if row["missing"] else ""
            print(f"  {row['subject_id']}: {row['status']}, {row['n_sax_slices']} SAX slices, LAX {row['lax_views'] or '-'}"
                  f" ({row['seconds']} s){note}", flush=True)
    log = pd.DataFrame(rows)
    log_path = args.dst / "prepare_log.csv"
    if log_path.exists():                              # keep rows of subjects not in this call
        old = pd.read_csv(log_path, dtype={"subject_id": str})
        log = pd.concat([old[~old.subject_id.isin(log.subject_id)], log]).sort_values("subject_id")
    log.to_csv(log_path, index=False)
    n = lambda status: int((log.status == status).sum())  # noqa: E731
    n_err = int(log.status.str.startswith("error").sum())
    print(f"\n{n('ok')} written complete, {n('partial')} written with missing views, {n('skipped_existing')} already "
          f"present, {n('no_images')} without images, {n_err} failed. Log: {log_path}")


if __name__ == "__main__":
    main()
