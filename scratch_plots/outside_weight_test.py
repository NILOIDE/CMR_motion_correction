"""Recommended settings with different --outside-mask-weight values on the 5 grid-search subjects.
Usage: python outside_weight_test.py run | measure"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PY = r"C:\Users\Nil\.conda\envs\ukbb\python.exe"
REPO = Path(r"C:\Users\Nil\Documents\git\UKBB_implicit_baselines")
ROOT = REPO / "scratch_plots" / "gs_root5"
OUT = Path(r"C:\Users\Nil\Documents\data\ukbb_alignment_outside_weight")
REF = Path(r"C:\Users\Nil\Documents\data\ukbb_alignment_anneal_gs\p2_b6psf3_s200_lin20")
SKIP = {"root", "out_root", "run_name", "csv", "plot", "plot_dir", "command_line", "line_scale_start", "line_anneal_frac",
        "no_save_images", "no_mask_plots", "no_subject_plots", "no_overall_plots", "no_cuda_graph", "skip_processed"}
RUNS = {f"ow{w:g}": {"outside-mask-weight": w} for w in (0.0, 0.05, 0.1, 0.25, 0.5, 1.0)}
RUNS["ow0.1_eager"] = {"outside-mask-weight": 0.1, "no-cuda-graph": None, "max-subjects": 1}


def run():
    cfg = json.loads((REF / "overall_metrics" / "run_config.json").read_text())
    base = {k: v for k, v in cfg.items() if k not in SKIP and v is not None and v is not False}
    for name, extra in RUNS.items():
        opts = {k.replace("_", "-"): v for k, v in base.items()} | extra
        cmd = [PY, str(REPO / "motion_gradient.py"), "--root", str(ROOT), "--out-root", str(OUT), "--run-name", name,
               "--no-save-images", "--no-subject-plots", "--no-overall-plots"]
        if name != "ow0.1":
            cmd.append("--no-mask-plots")
        for k, v in opts.items():
            cmd += [f"--{k}"] if v is True or v is None else [f"--{k}", str(v)]
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
        rows.append(dict(run=name, dice=s.seg_dice_mean.mean(), dice_worst10=dice.iloc[:max(1, len(dice) // 10)].mean(),
                         hd=s.hd_mean_mm.mean(), ncc=s.ncc_mean.mean(), tilt_rms=s.sax_tilt_rms_deg.mean(),
                         apex_1000071_dice=float(apex.seg_dice.iloc[0]) if len(apex) else np.nan, sec=s.seconds.mean()))
    pd.set_option("display.width", 200)
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    for a_name, b_name, label in (("ow0", None, "ow0 vs earlier run with the same settings"), ("ow0.1", "ow0.1_eager", "graph vs eager (w 0.1)")):
        a = pd.read_csv(OUT / a_name / "overall_metrics" / "motion_summary_slices.csv", dtype={"subject_id": str})
        b = pd.read_csv((OUT / b_name if b_name else REF) / "overall_metrics" / "motion_summary_slices.csv", dtype={"subject_id": str})
        a = a[a.subject_id.isin(b.subject_id)].reset_index(drop=True)
        num = [c for c in a.columns if a[c].dtype.kind == "f"]
        print(label, "max abs diff:", np.nanmax(np.abs(a[num].values - b[num].values)))


if __name__ == "__main__":
    {"run": run, "measure": measure}[sys.argv[1]]()
