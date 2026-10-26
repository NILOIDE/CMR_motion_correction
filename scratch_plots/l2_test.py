"""Recommended settings with --metric l2 vs ncc on the 5 grid-search subjects.
Usage: python l2_test.py run | measure"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PY = r"C:\Users\Nil\.conda\envs\ukbb\python.exe"
REPO = Path(r"C:\Users\Nil\Documents\git\UKBB_implicit_baselines")
ROOT = REPO / "scratch_plots" / "gs_root5"
OUT = Path(r"C:\Users\Nil\Documents\data\ukbb_alignment_l2")
REF = Path(r"C:\Users\Nil\Documents\data\ukbb_alignment_anneal_gs\p2_b6psf3_s200_lin20")
SKIP = {"root", "out_root", "run_name", "csv", "plot", "plot_dir", "command_line", "line_scale_start", "line_anneal_frac",
        "no_save_images", "no_mask_plots", "no_subject_plots", "no_overall_plots", "no_cuda_graph", "skip_processed"}
RUNS = {
    "ncc": {},
    "l2": {"metric": "l2"},
    "l2_noanneal": {"metric": "l2", "blur-sigma": 0, "psf-scale-start": 1.0},
    "l2_s600": {"metric": "l2", "steps": 600},
    "l2_eager": {"metric": "l2", "no-cuda-graph": True, "max-subjects": 1},
}


def run():
    cfg = json.loads((REF / "overall_metrics" / "run_config.json").read_text())
    base = {k: v for k, v in cfg.items() if k not in SKIP and v is not None and v is not False}
    for name, extra in RUNS.items():
        if (OUT / name / "overall_metrics" / "motion_summary.csv").exists():
            continue
        opts = {k.replace("_", "-"): v for k, v in base.items()} | extra
        cmd = [PY, str(REPO / "motion_gradient.py"), "--root", str(ROOT), "--out-root", str(OUT), "--run-name", name,
               "--no-save-images", "--no-mask-plots", "--no-subject-plots", "--no-overall-plots"]
        for k, v in opts.items():
            cmd += [f"--{k}"] if v is True else [f"--{k}", str(v)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        print(name, "exit", r.returncode, r.stderr[-800:] if r.returncode else "", flush=True)


def measure():
    rows = []
    for name in RUNS:
        if name.endswith("eager"):
            continue
        s = pd.read_csv(OUT / name / "overall_metrics" / "motion_summary.csv", dtype={"subject_id": str})
        d = pd.read_csv(OUT / name / "overall_metrics" / "motion_summary_slices.csv", dtype={"subject_id": str})
        d = d[d.n_pairs > 0]
        dice = d.seg_dice.dropna().sort_values()
        apex = d[(d.subject_id == "1000071") & (d.image == "SAX") & (d.slice == 9)]
        hs = pd.concat(pd.read_csv(f) for f in (OUT / name).glob("*/metrics/history_summary.csv"))
        obj = hs[hs.is_best == 1].objective.mean()
        obj0 = hs[hs.step == 0].objective.mean()
        rows.append(dict(run=name, dice=s.seg_dice_mean.mean(), dice_worst10=dice.iloc[:max(1, len(dice) // 10)].mean(),
                         hd=s.hd_mean_mm.mean(), ncc=s.ncc_mean.mean(), nmi=s.mi_mean.mean(), objective_start=obj0,
                         objective_best=obj, sax_tilt_rms=s.sax_tilt_rms_deg.mean(),
                         apex71_dice=float(apex.seg_dice.iloc[0]) if len(apex) else np.nan, sec=s.seconds.mean()))
    pd.set_option("display.width", 220)
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    a = pd.read_csv(OUT / "l2" / "overall_metrics" / "motion_summary_slices.csv", dtype={"subject_id": str})
    b = pd.read_csv(OUT / "l2_eager" / "overall_metrics" / "motion_summary_slices.csv", dtype={"subject_id": str})
    a = a[a.subject_id.isin(b.subject_id)].reset_index(drop=True)
    num = [c for c in a.columns if a[c].dtype.kind == "f"]
    print("l2 graph vs eager max abs diff:", np.nanmax(np.abs(a[num].values - b[num].values)))
    c = pd.read_csv(OUT / "ncc" / "overall_metrics" / "motion_summary_slices.csv", dtype={"subject_id": str})
    r = pd.read_csv(REF / "overall_metrics" / "motion_summary_slices.csv", dtype={"subject_id": str})
    print("ncc vs earlier identical run max abs diff:", np.nanmax(np.abs(c[num].values - r[num].values)))


if __name__ == "__main__":
    {"run": run, "measure": measure}[sys.argv[1]]()
