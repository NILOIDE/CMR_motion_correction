"""Smoke test of --optimizer on subject 1000071 with the script's defaults (one seed each): every optimizer runs with
CUDA graphs, graph == eager for sgd and adagrad, and a rough look at the result. Not a tuned comparison."""
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

PY = r"C:\Users\Nil\.conda\envs\ukbb\python.exe"
REPO = Path(r"C:\Users\Nil\Documents\git\UKBB_implicit_baselines")
ROOT = REPO / "scratch_plots" / "grid_root"
OUT = REPO / "scratch_plots" / "optimizer_smoke"
RUNS = {
    "adam": [], "nadam": ["--optimizer", "nadam"], "radam": ["--optimizer", "radam"], "adamax": ["--optimizer", "adamax"],
    "rmsprop": ["--optimizer", "rmsprop"], "adagrad": ["--optimizer", "adagrad"],
    "sgd": ["--optimizer", "sgd"], "sgd_nesterov": ["--optimizer", "sgd", "--nesterov"],
    "sgd_lr50": ["--optimizer", "sgd", "--lr", "50", "--rot-lr", "20"], "sgd_lr10": ["--optimizer", "sgd", "--lr", "10", "--rot-lr", "5"],
    "sgd_eager": ["--optimizer", "sgd", "--no-cuda-graph"], "adagrad_eager": ["--optimizer", "adagrad", "--no-cuda-graph"],
}


def slices(name):
    return pd.read_csv(OUT / name / "overall_metrics" / "motion_summary_slices.csv", dtype={"subject_id": str})


rows = []
for name, extra in RUNS.items():
    cmd = [PY, str(REPO / "motion_gradient.py"), "--root", str(ROOT), "--out-root", str(OUT), "--run-name", name,
           "--no-save-images", "--no-mask-plots", "--no-subject-plots", "--no-overall-plots", "--history-every", "0"] + extra
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(name, "FAILED", r.stderr[-1000:])
        continue
    fell_back = "continuing without graphs" in (r.stdout + r.stderr)
    s = pd.read_csv(OUT / name / "overall_metrics" / "motion_summary.csv").iloc[-1]
    d = slices(name)
    s9 = d[(d.image == "SAX") & (d.slice == 9)].iloc[0]
    rows.append(dict(run=name, sec=round(time.time() - t0, 1), graph_fallback=fell_back, ncc=s.ncc_mean, dice=s.seg_dice_mean,
                     dice9=s9.seg_dice, dz9=s9.dz_mm, sax_tilt_rms=s.sax_tilt_rms_deg))
pd.set_option("display.width", 200)
print(pd.DataFrame(rows).round(3).to_string(index=False))
for a, b in (("sgd", "sgd_eager"), ("adagrad", "adagrad_eager")):
    x, y = slices(a), slices(b)
    num = [c for c in x.columns if x[c].dtype.kind == "f"]
    print(f"{a}: graph vs eager max abs diff {np.nanmax(np.abs(x[num].values - y[num].values))}")
