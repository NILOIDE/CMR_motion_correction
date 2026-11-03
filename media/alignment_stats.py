"""
The cohort statistics of an alignment run (moco_nifti.py with --logging): its metrics brought into one table per
subject and one per slice / view, and judged on the SAX slices that contain the heart. Ported from the survey of
UKBB_implicit_baselines (data/analysis/alignment.py); self-contained here: everything comes from the run and the
images it aligned.

The run directory holds, in overall_metrics/,
  motion_summary.csv          per subject: status, pose model, NCC / NMI before and after, motion statistics
  motion_summary_slices.csv   per SAX slice / long-axis view: lines, pose correction, NCC / NMI / Dice before and after
  sax_relative_motion.csv     per SAX slice its pose against the rest of the stack (the median of what the other
                              slices' poses predict for it, the "stack consensus"), per neighbouring pair the
                              relative pose
  run_config.json             the settings
(append-only while the run goes on: a subject's latest row / block counts).

Kept slices. The survey judges the slices its preprocessing keeps; here a subject's kept range is the SAX slices
from the most basal to the most apical one whose segmentation (the one the run used: seg_sa where present, else the
nnU-Net SAX segmentation) contains LV blood pool at end-diastole (frame 0). The apical end is the one with the
smaller LV area. Slices outside the range lie beyond the base / beyond the apex; the kept range is split into thirds
for the LV regions (basal / mid / apical). Alternatively (survey_kept) the kept ranges, the verdict on the subjects
and the LV regions come from an existing survey of UKBB_implicit_baselines (its survey.csv and slices.csv), which
reproduces the survey's own figures without reading any segmentation. Only the per-slice Dice of the run's segmentation measures (SAX labels
against the long-axis views' along the intersection lines) is kept, and judged only on mid-ventricular slices:
apical and basal labels disagree across views by themselves (partial volume, outflow tract).

Signs and frames as in the run: dx / dy along the image's own rows / columns, dz along its normal, translations
relative to the mean pose of the long-axis views (which therefore move little); rel_* are frame-free (a slice against
its own stack). Tilt (rotation about in-plane axes) is weakly constrained by three crossing lines: reported, not judged.
"""
import json
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))           # the repository root (moco_nifti)

SOURCE_FILES = ("motion_summary.csv", "motion_summary_slices.csv", "sax_relative_motion.csv", "run_config.json")
SERIES_OF = {"LAX_2Ch": "lax2", "LAX_3Ch": "lax3", "LAX_4Ch": "lax4", "ao": "ao"}     # run image -> series kind
SUBJECT_COLS = {"status": "status", "pose_model": "pose_model", "n_lax_views": "lax_views", "sax_n_eval": "sax_slices",
                "n_line_pairs": "lines", "ncc_zero_mean": "ncc_before", "ncc_mean": "ncc_after",
                "ncc_gain_mean": "ncc_gain", "nmi_gain_mean": "nmi_gain", "sax_shift_max_mm": "sax_shift_max_mm",
                "sax_shift_rms_mm": "sax_shift_rms_mm", "sax_dz_rms_mm": "sax_dz_rms_mm",
                "sax_rot_inplane_rms_deg": "sax_rot_rms_deg", "sax_tilt_rms_deg": "sax_tilt_rms_deg",
                "lax_shift_max_mm": "lax_shift_max_mm", "timestamp": "timestamp"}
PLANE_COLS = {"n_pairs": "n_lines", "n_points": "n_points", "dx_mm": "dx_mm", "dy_mm": "dy_mm", "dz_mm": "dz_mm",
              "shift_inplane_mm": "shift_inplane_mm", "shift_mm": "shift_mm", "rot_inplane_deg": "rot_inplane_deg",
              "rot_tilt_deg": "rot_tilt_deg", "ncc_zero": "ncc_before", "ncc": "ncc_after", "nmi_zero": "nmi_before",
              "nmi": "nmi_after", "seg_dice_zero": "dice_before", "seg_dice": "dice_after"}
REL_COLS = {"dx_mm": "rel_dx_mm", "dy_mm": "rel_dy_mm", "dz_mm": "rel_dz_mm", "shift_inplane_mm": "rel_shift_mm",
            "shift_mm": "rel_shift3d_mm", "rot_inplane_deg": "rel_rot_inplane_deg", "rot_deg": "rel_rot_deg",
            "rot_tilt_deg": "rel_tilt_deg"}
ADJ_COLS = {"shift_inplane_mm": "adj_shift_mm", "dz_mm": "adj_dz_mm", "rot_deg": "adj_rot_deg"}
FULL_MODEL = "6dof"                                 # the run's pose model with every degree of freedom
MOTION_MAX_MM = 10.0                                # a kept slice further than this from its stack: the subject moved
ALIGN_NCC_MIN = 0.5                                 # a kept slice matching its lines worse after the alignment: not trusted
LV_LABEL = 1                                        # LV blood pool in seg_sa and the nnU-Net SAX segmentation


# ---------------------------------------------------------------------------------------------------------- source

def remote_parts(src: str) -> tuple[str, str] | None:
    """(host, path) of 'user@host:/path' or 'host:/path' (not a Windows drive 'C:/...' or an existing local path)."""
    m = re.fullmatch(r"((?:[^@/\\:\s]+@)?[^@/\\:\s]{2,}):(.+)", str(src).strip())
    return (m.group(1), m.group(2)) if m and not Path(str(src)).exists() else None


def _metrics_dir(path: str) -> str:
    p = str(path).rstrip("/\\")
    return p if p.endswith("overall_metrics") else p + "/overall_metrics"


def run_dir(src: str) -> str:
    """The run directory of a source (a run directory or its overall_metrics/), as given (local or remote)."""
    p = str(src).rstrip("/\\")
    return p[: -len("overall_metrics")].rstrip("/\\") if p.endswith("overall_metrics") else p


def _scp(host: str, remote: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["scp", "-q", "-o", "BatchMode=yes", f"{host}:{remote}", str(dest)], capture_output=True, text=True)
    return r.returncode == 0 and dest.exists()


def fetch_source(src: str, copy_dir: Path) -> Path:
    """The run's overall_metrics/ as a local directory: the source itself, or for a remote source the files of
    SOURCE_FILES copied into copy_dir (scp; motion_summary_pairs.csv is not needed)."""
    rp = remote_parts(src)
    if rp is None:
        d = Path(_metrics_dir(src))
        if not d.is_dir():
            raise SystemExit(f"alignment: no directory {d}")
        return d
    host, path = rp
    got = [f for f in SOURCE_FILES if _scp(host, f"{_metrics_dir(path)}/{f}", copy_dir / f)]
    if not got:
        raise SystemExit(f"alignment: nothing copied from {host}:{_metrics_dir(path)} (ssh key / path?)")
    print(f"alignment: copied {', '.join(got)} from {host}:{_metrics_dir(path)}")
    return copy_dir


def latest_blocks(df):
    """Rows of an append-only per-subject file of the run: each subject's latest contiguous block of rows (a subject
    aligned again appends a new block)."""
    if df.empty:
        return df
    blk = (df.subject != df.subject.shift()).cumsum()
    return df[blk == blk.groupby(df.subject).transform("max")]


def _read(path: Path):
    import pandas as pd
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=["subject"])
    df = pd.read_csv(path, dtype={"subject_id": str}, on_bad_lines="skip", low_memory=False)   # a run still appends
    return df.rename(columns={"subject_id": "subject"})


def import_run(d: Path) -> tuple:
    """(per-subject table, per-plane table, run_config) from a run's overall_metrics/ directory."""
    import pandas as pd
    summ = _read(d / "motion_summary.csv")
    sl = latest_blocks(_read(d / "motion_summary_slices.csv"))
    rel = latest_blocks(_read(d / "sax_relative_motion.csv"))
    cfg = json.loads((d / "run_config.json").read_text()) if (d / "run_config.json").exists() else {}
    if len(summ):
        summ = summ.drop_duplicates("subject", keep="last")
        subj = summ[["subject"] + [c for c in SUBJECT_COLS if c in summ]].rename(columns=SUBJECT_COLS)
    else:                                   # a run without --logging: only the relative motion (aligned subjects)
        subj = pd.DataFrame(dict(subject=sorted(rel.subject.unique()), status="ok"))
    parts = []
    if len(sl):
        p = sl[["subject", "image", "slice"] + [c for c in PLANE_COLS if c in sl]].rename(columns=PLANE_COLS)
        p["series"] = np.where(p.image == "SAX", "sax" + p.slice.astype(int).astype(str), p.image.map(SERIES_OF).fillna(p.image))
        parts.append(p)
    planes = parts[0] if parts else pd.DataFrame(columns=["subject", "series", "image", "slice"])
    if len(rel):
        rel = rel.assign(slice_a=pd.to_numeric(rel.slice_a, errors="coerce"),
                         slice_b=pd.to_numeric(rel.slice_b, errors="coerce"))
        r = rel[rel.kind == "slice"]
        r = r[["subject", "slice_a"] + list(REL_COLS)].rename(columns=dict(REL_COLS, slice_a="slice"))
        a = rel[rel.kind == "pair"]
        a = a[["subject", "slice_b"] + list(ADJ_COLS)].rename(columns=dict(ADJ_COLS, slice_b="slice"))
        r = r.merge(a, on=["subject", "slice"], how="outer")
        r["slice"] = r.slice.astype(int)
        r["series"] = "sax" + r.slice.astype(str)
        r["image"] = "SAX"
        planes = planes.merge(r, on=["subject", "series", "image", "slice"], how="outer") if len(planes) else r
    planes = planes[planes.subject.isin(subj.subject)].sort_values(["subject", "image", "slice"])
    return subj.sort_values("subject"), planes, cfg


# ------------------------------------------------------------------------------------------------ kept SAX slices

def lv_pixels(subj_dir: Path) -> list[tuple[int, int]]:
    """[(SAX slice index, LV blood-pool pixels at end-diastole)] of one subject, with the run's slice indices and
    segmentation choice (moco_nifti.load_subject: seg_sa where it covers a slice, else the nnU-Net one). Reads only
    the headers and frame 0 of the segmentations; slices without a segmentation are left out."""
    import nibabel as nib
    import moco_nifti as mn
    found = mn.discover_subject(subj_dir)
    preferred = [p for p in found["sax_seg"] if mn.PREFERRED_SAX_SEG.match(mn.nii_stem(p))]
    seg_slices = mn.sax_slice_list(preferred) + mn.sax_slice_list([p for p in found["sax_seg"] if p not in preferred])
    frames = {}
    out = []
    for s in mn.sax_slice_list(found["sax"]):
        g = next((g for g in seg_slices if g["shape"] == s["shape"]
                  and np.linalg.norm(g["affine"][:3, 3] - s["affine"][:3, 3]) < mn.SAME_SLICE_MM), None)
        if g is None:
            continue
        if g["path"] not in frames:
            obj = nib.load(str(g["path"])).dataobj
            frames[g["path"]] = np.rint(np.asarray(obj[..., 0] if len(obj.shape) == 4 else obj)).astype(np.int16)
        seg = frames[g["path"]]
        seg = seg[:, :, g["z"]] if seg.ndim == 3 else seg
        out.append((s["index"], int((seg == LV_LABEL).sum())))
    return out


def _lv_job(args):
    sid, d = args
    try:
        return sid, lv_pixels(Path(d)), None
    except Exception as e:  # noqa: BLE001
        return sid, [], f"{type(e).__name__}: {e}"


def lv_table(subjects, root: Path, cache: Path, workers: int = 8):
    """LV pixels per SAX slice (subject, slice, lv_px) of the given subjects under root, cached in `cache` (only the
    subjects not in it yet are read)."""
    import pandas as pd
    have = pd.read_csv(cache, dtype={"subject": str}) if cache.exists() else pd.DataFrame(columns=["subject", "slice", "lv_px"])
    todo = [s for s in subjects if s not in set(have.subject) and (Path(root) / s).is_dir()]
    if todo:
        print(f"alignment: reading the SAX segmentations of {len(todo)} subjects ({workers} processes) ...", flush=True)
        rows, errors = [], 0
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for i, (sid, px, err) in enumerate(ex.map(_lv_job, [(s, str(Path(root) / s)) for s in todo], chunksize=16)):
                rows += [dict(subject=sid, slice=k, lv_px=n) for k, n in px] or [dict(subject=sid, slice=-1, lv_px=0)]
                errors += err is not None
                if (i + 1) % 2000 == 0:
                    print(f"  {i + 1} / {len(todo)}", flush=True)
        if errors:
            print(f"alignment: {errors} subjects without a readable SAX segmentation", flush=True)
        have = pd.concat([have, pd.DataFrame(rows)], ignore_index=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        have.to_csv(cache, index=False)
    return have[have.subject.isin(set(subjects)) & (have.slice >= 0)]


def kept_ranges(lv):
    """Per subject (index) the kept SAX range from the LV pixels per slice: base / apex (slice indices; the apex the
    end whose LV-containing slices have the smaller mean LV area) and dir (+1: indices grow toward the apex)."""
    import pandas as pd
    rows = {}
    for sid, g in lv[lv.lv_px > 0].groupby("subject"):
        k = np.sort(g.slice.to_numpy(int))
        area = g.set_index("slice").lv_px
        third = max(1, len(k) // 3)
        up = area.loc[k[-third:]].mean() <= area.loc[k[:third]].mean()          # the LV narrows toward the apex
        rows[sid] = dict(base=k[0] if up else k[-1], apex=k[-1] if up else k[0], dir=1 if up else -1)
    return pd.DataFrame.from_dict(rows, orient="index", columns=["base", "apex", "dir"])


def survey_kept(survey_dir: Path, variant: str = "nn"):
    """(kept ranges, verdict, regions) from a survey of UKBB_implicit_baselines (data/analysis/survey.py output):
    per subject the base / apex of the coverage test its default verdict uses (LAX-based with a usable long-axis
    view, else SAX-only; NaN where that end is not covered; dir +1, the survey's slices count from the base), the
    verdict {variant}_ok (True: the preprocessing keeps the subject), and per SAX slice its LV region (slices.csv
    lax_region: beyond base / basal / mid / apical / beyond apex). variant: nn (Ugurlu et al.) or sa (Bai et al.)."""
    import pandas as pd
    v = variant
    cols = ["subject", "lax_view", f"{v}_ok", f"{v}_lax_base", f"{v}_lax_apex", f"{v}_sax_base", f"{v}_sax_apex"]
    s = pd.read_csv(Path(survey_dir) / "survey.csv", usecols=cols, dtype={"subject": str}, low_memory=False)
    s = s.drop_duplicates("subject", keep="last").set_index("subject")
    has_view = s.lax_view.notna()
    kr = pd.DataFrame(dict(base=pd.to_numeric(s[f"{v}_lax_base"].where(has_view, s[f"{v}_sax_base"]), errors="coerce"),
                           apex=pd.to_numeric(s[f"{v}_lax_apex"].where(has_view, s[f"{v}_sax_apex"]), errors="coerce"),
                           dir=1.0), index=s.index)
    verdict = s[f"{v}_ok"].astype(str).str.lower() == "true"
    sl = pd.read_csv(Path(survey_dir) / "slices.csv", usecols=["subject", "variant", "slice", "lax_region"],
                     dtype={"subject": str}, low_memory=False)
    regions = sl[sl.variant == v][["subject", "slice", "lax_region"]].rename(columns={"lax_region": "region"})
    regions = regions.dropna(subset=["slice"]).astype({"slice": int}).drop_duplicates(["subject", "slice"])
    return kr, verdict, regions


# -------------------------------------------------------------------------------------------------------- assemble

def sax_planes(series, kr=None):
    """The SAX rows of the series table, with (given the kept ranges kr) their position against the kept range:
    pos = kept / beyond base / beyond apex (NaN without a range), from_apex = slices past the apical end (> 0: beyond
    it), region = beyond base / basal / mid / apical / beyond apex (the kept range in thirds, unless the rows already
    carry a region)."""
    p = series[(series.kind == "sax") & series.slice.notna()].copy()
    p["slice"] = p.slice.astype(int)
    if kr is not None:
        p = p.join(kr, on="subject")
        span = ((p.apex - p.base) * p.dir).clip(lower=1)
        from_base = (p.slice - p.base) * p.dir
        p["from_apex"] = (p.slice - p.apex) * p.dir
        p["pos"] = np.where(p.base.isna() | p.apex.isna(), None,
                            np.where(from_base < 0, "beyond base", np.where(p.from_apex > 0, "beyond apex", "kept")))
        if "region" not in p:
            frac = from_base / span
            p["region"] = np.where(p.pos != "kept", p.pos,
                                   np.where(frac < 1 / 3, "basal", np.where(frac < 2 / 3, "mid", "apical")))
    return p


def _extremes(p):
    """Per subject over SAX plane rows p: the largest in-plane / through-plane / rotational motion against the
    stack (and the slice), the largest relative shift of neighbours, the lowest NCC after the alignment (and the
    slice), the slices without an intersection line."""
    import pandas as pd
    out = pd.DataFrame(index=p.subject.unique())
    out.index.name = "subject"
    out["slices_no_lines"] = (p.align_n_lines.fillna(0) <= 0).groupby(p.subject).sum() if "align_n_lines" in p else np.nan

    def pick(col, name, fn="idxmax", key=None, with_slice=True):
        if col not in p:
            return
        q = p.dropna(subset=[col])
        if q.empty:
            return
        k = q[col].abs() if key == "abs" else q[col]
        idx = getattr(k.groupby(q.subject), fn)()
        out[name] = pd.Series(k.loc[idx.values].values, index=idx.index)
        if with_slice:
            out[name.replace("_max_mm", "_slice").replace("_min", "_min_slice")] = pd.Series(
                q.slice.loc[idx.values].values, index=idx.index)

    pick("align_rel_shift_mm", "rel_shift_max_mm")
    pick("align_rel_dz_mm", "rel_dz_max_mm", key="abs", with_slice=False)
    pick("align_rel_rot_deg", "rel_rot_max_deg", with_slice=False)
    pick("align_adj_shift_mm", "adj_shift_max_mm", with_slice=False)
    pick("align_ncc_after", "slice_ncc_min", fn="idxmin")
    return out


def _problems(r) -> str | None:
    """Why the alignment of a subject's kept slices is not trustworthy (None: it is): slices without an intersection
    line (no pose), a slice matching its lines worse than ALIGN_NCC_MIN after the alignment."""
    out = []
    if r.get("slices_no_lines", 0) > 0:
        out.append(f"{int(r['slices_no_lines'])} kept slice(s) without an intersection line")
    if r.get("slice_ncc_min", 1) < ALIGN_NCC_MIN:
        out.append(f"slice {int(r['slice_ncc_min_slice'])} NCC {r['slice_ncc_min']:.2f} after the alignment "
                   f"(< {ALIGN_NCC_MIN:g})")
    return "; ".join(out) or None


def assemble(subj, planes, kr, verdict=None, regions=None):
    """(df, series): per subject (index) the run's values (align_*), whether it has an aortic slice, its kept range,
    whether it is kept (the verdict, else: it has a kept range), and over its kept slices the motion (motion_mm, motion_slice, motion_ok), the alignment quality (align_ok,
    align_problem), the extremes (kept_align_*) and the mid-ventricular Dice; per slice / view the run's values
    (align_*) with its kind (sax, lax2, lax3, lax4, ao), slice and (SAX) LV region (`regions`: subject, slice,
    region; else the kept range in thirds)."""
    import pandas as pd
    al = subj.drop_duplicates("subject", keep="last").set_index("subject")
    df = al.add_prefix("align_")
    df["has_ao"] = df.index.isin(planes.subject[planes.image == "ao"])
    df = df.join(kr)
    df["kept"] = (verdict.reindex(df.index).fillna(False).astype(bool) if verdict is not None
                  else df.base.notna() & df.apex.notna())
    cols = [c for c in planes if c not in ("subject", "series", "image", "slice")]
    series = planes.rename(columns={c: f"align_{c}" for c in cols})
    is_sax = series.image == "SAX"
    series["kind"] = np.where(is_sax, "sax", series.image.map(SERIES_OF).fillna(series.image))
    status, model = df.align_status, df.align_pose_model
    full = (status == "ok") & (model == FULL_MODEL)
    allx = _extremes(sax_planes(series)).drop(columns=["slices_no_lines"]).add_prefix("align_")
    p = sax_planes(series, kr)
    if regions is not None:
        p = p.drop(columns="region").merge(regions, on=["subject", "slice"], how="left")
    series = series.merge(p[["subject", "slice", "region"]].assign(kind="sax"), on=["subject", "kind", "slice"],
                          how="left")
    mid = p[(p.region == "mid") & (p.pos == "kept")].groupby("subject")
    k = p[p.pos == "kept"].copy()
    k.loc[k.slice == k.base, [c for c in ("align_adj_shift_mm", "align_adj_dz_mm", "align_adj_rot_deg") if c in k]] = np.nan
    ex = _extremes(k).reindex(df.index)
    prob = ex.apply(_problems, axis=1) if len(ex) else pd.Series(np.nan, index=df.index)
    ranged = ex.rel_shift_max_mm.notna() if "rel_shift_max_mm" in ex else pd.Series(False, index=df.index)
    est = full & ranged
    ok = pd.Series(np.nan, index=df.index, dtype=object)
    text = pd.Series(np.nan, index=df.index, dtype=object)
    ok[est], text[est] = prob[est].isna(), prob[est]
    text[full & ~ranged] = "no kept slices (no LV in the SAX segmentation)"
    bad = status != "ok"
    ok[bad], text[bad] = False, "not aligned: " + status[bad].astype(str)
    part = (status == "ok") & ~(model == FULL_MODEL)
    ok[part], text[part] = False, "pose model " + model[part].astype(str) + " (one crossing direction)"
    motion = ex.get("rel_shift_max_mm", pd.Series(np.nan, index=df.index)).where(est)
    moved = pd.Series(np.nan, index=df.index, dtype=object)
    moved[est] = motion[est] <= MOTION_MAX_MM
    df = pd.concat([df, allx.reindex(df.index),
                    ex.drop(columns=["slices_no_lines", "rel_shift_max_mm", "rel_shift_slice"], errors="ignore")
                    .add_prefix("kept_align_"),
                    pd.DataFrame({"mid_dice_before": mid.align_dice_before.mean(),
                                  "mid_dice_after": mid.align_dice_after.mean(),
                                  "mid_dice_min": mid.align_dice_after.min()}).reindex(df.index)], axis=1)
    df = df.assign(motion_mm=motion, motion_slice=ex.get("rel_shift_slice", motion * np.nan).where(est),
                   motion_ok=moved, align_ok=ok, align_problem=text)
    return df.copy(), series.sort_values(["subject", "series"])


# ------------------------------------------------------------------------------------------------- poses of a subject

def history(run: str, sid: str, copy_dir: Path):
    """The run's pose history of one subject (<run>/<subject>/metrics/history.csv; copied with scp into copy_dir for
    a remote run), or None."""
    import pandas as pd
    rp = remote_parts(run)
    if rp is None:
        f = Path(run_dir(run)) / sid / "metrics" / "history.csv"
    else:
        f = copy_dir / sid / "history.csv"
        if not f.exists() and not _scp(rp[0], f"{run_dir(rp[1])}/{sid}/metrics/history.csv", f):
            return None
    return pd.read_csv(f) if f.exists() else None


def poses(hist) -> dict:
    """{'acquired' | 'aligned': {(image, slice): 4 x 4 voxel -> world}} from a history table: step 0 (the acquired
    geometry) and the saved pose (is_best; the corrected affines the run writes)."""
    out = {}
    for name, rows in (("acquired", hist[hist.step == hist.step.min()]), ("aligned", hist[hist.is_best == 1])):
        m = {}
        for _, r in rows.iterrows():
            A = np.eye(4)
            A[:3] = [[r[f"a{i}{j}"] for j in range(4)] for i in range(3)]
            m[(r.image, int(r.slice))] = A
        out[name] = m
    return out
