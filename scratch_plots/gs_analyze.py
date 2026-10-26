"""Summarize grid-search runs in ukbb_alignment_anneal_gs. Usage: python gs_analyze.py [name_prefix ...]"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(r"C:\Users\Nil\Documents\data\ukbb_alignment_anneal_gs")
REF = Path(r"C:\Users\Nil\Documents\data\ukbb_alignment\steps600-lr0.05-lin-blur4x0.6\overall_metrics\motion_summary_slices.csv")

ref = pd.read_csv(REF, dtype={"subject_id": str})
ref = ref[(ref.subject_id == "1000071") & (ref.image == "SAX")].drop_duplicates("slice", keep="last").set_index("slice")
REF_APEX = ref.loc[[8, 9], ["dx_mm", "dy_mm", "dz_mm"]].to_numpy()


def converged_fraction(h: pd.DataFrame, col: str, tol: float) -> float:
    """Earliest fraction of the run after which `col` stays within tol of its final value (snapshots only)."""
    h = h[h.is_best == 0].sort_values("step")
    if h.empty or not np.isfinite(h[col]).any():
        return np.nan
    final = h[col].iloc[-1]
    ok = (h[col] - final).abs() <= tol
    bad = np.flatnonzero(~ok.to_numpy())
    return float(h["fraction"].iloc[bad[-1] + 1]) if len(bad) and bad[-1] + 1 < len(h) else (0.0 if not len(bad) else 1.0)


def summarize(run: Path) -> dict | None:
    f = run / "overall_metrics" / "motion_summary.csv"
    if not f.exists():
        return None
    s = pd.read_csv(f, dtype={"subject_id": str}).drop_duplicates("subject_id", keep="last")
    s = s[s.status == "ok"]
    d = pd.read_csv(run / "overall_metrics" / "motion_summary_slices.csv", dtype={"subject_id": str})
    cfg = json.loads((run / "overall_metrics" / "run_config.json").read_text())
    apex = d[(d.subject_id == "1000071") & (d.image == "SAX")].set_index("slice")
    apex_dist = np.linalg.norm(apex.loc[[8, 9], ["dx_mm", "dy_mm", "dz_mm"]].to_numpy() - REF_APEX, axis=1)
    conv_obj, conv_dice = [], []
    for sid in s.subject_id:
        hf = run / sid / "metrics" / "history_summary.csv"
        if hf.exists():
            h = pd.read_csv(hf)
            conv_obj.append(converged_fraction(h, "objective", 0.002))
            conv_dice.append(converged_fraction(h, "seg_dice_mean", 0.005))
    sax = d[d.image == "SAX"]
    return dict(
        run=run.name, steps=cfg["steps"], n=len(s),
        dice=s.seg_dice_mean.mean(), dice_gain=(s.seg_dice_mean - s.seg_dice_zero_mean).mean(),
        ncc=s.ncc_mean.mean(), nmi=s.mi_mean.mean(), hd_mean=s.hd_mean_mm.mean(), hd_max=s.hd_max_mm.mean(),
        jitter=s.sax_shift_jitter_mm.mean(), dz=s.sax_dz_rms_mm.mean(), tilt=s.sax_tilt_rms_deg.mean(),
        sax_optimized=(sax.n_pairs > 0).mean(),
        apex8=apex_dist[0], apex9=apex_dist[1],
        conv_obj=np.nanmean(conv_obj) if conv_obj else np.nan, conv_dice=np.nanmean(conv_dice) if conv_dice else np.nan,
        sec=s.seconds.mean(),
    )


def main():
    prefixes = sys.argv[1:] or [""]
    rows = [r for run in sorted(OUT.iterdir()) if run.is_dir() and any(run.name.startswith(p) for p in prefixes)
            if (r := summarize(run))]
    df = pd.DataFrame(rows).sort_values("dice", ascending=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 300)
    print(df.round(3).to_string(index=False))
    df.to_csv(OUT / f"gs_summary_{'_'.join(p.rstrip('_') for p in prefixes) or 'all'}.csv", index=False)
    return df


if __name__ == "__main__":
    main()
