"""Runs the recommended settings with SAX stack regularizers and measures adjacent-slice angles / spacings.
Usage: python stack_reg_test.py run   |   python stack_reg_test.py measure"""
import json
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

PY = r"C:\Users\Nil\.conda\envs\ukbb\python.exe"
REPO = Path(r"C:\Users\Nil\Documents\git\UKBB_implicit_baselines")
ROOT = REPO / "scratch_plots" / "gs_root5"
OUT = Path(r"C:\Users\Nil\Documents\data\ukbb_alignment_stackreg")
REF = Path(r"C:\Users\Nil\Documents\data\ukbb_alignment_anneal_gs\p2_b6psf3_s200_lin20")
SKIP = {"root", "out_root", "run_name", "csv", "plot", "plot_dir", "command_line", "line_scale_start", "line_anneal_frac",
        "no_save_images", "no_mask_plots", "no_subject_plots", "no_overall_plots", "no_cuda_graph", "skip_processed"}
RUNS = {
    "base": {},
    "n1e-3": {"normal-reg": 1e-3},
    "n1e-2": {"normal-reg": 1e-2},
    "s1e-3": {"spacing-reg": 1e-3},
    "s1e-2": {"spacing-reg": 1e-2},
    "n1e-3_s1e-3": {"normal-reg": 1e-3, "spacing-reg": 1e-3},
    "n1e-2_s1e-2": {"normal-reg": 1e-2, "spacing-reg": 1e-2},
}


def run():
    cfg = json.loads((REF / "overall_metrics" / "run_config.json").read_text())
    base = []
    for k, v in cfg.items():
        if k in SKIP or v is None or v is False:
            continue
        base += [f"--{k.replace('_', '-')}"] if v is True else [f"--{k.replace('_', '-')}", str(v)]
    for name, extra in RUNS.items():
        cmd = [PY, str(REPO / "motion_gradient.py"), "--root", str(ROOT), "--out-root", str(OUT), "--run-name", name,
               "--no-save-images", "--no-mask-plots", "--no-subject-plots", "--no-overall-plots"] + base
        for k, v in extra.items():
            cmd += [f"--{k}", str(v)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        print(name, "exit", r.returncode, r.stderr[-800:] if r.returncode else "", flush=True)


def stack_geometry(run_dir: Path) -> pd.DataFrame:
    rows = []
    for hist in sorted(run_dir.glob("*/metrics/history.csv")):
        sid = hist.parent.parent.name
        X, Y = nib.load(str(ROOT / sid / "Instance_2" / "SAX.nii.gz")).shape[:2]
        d = pd.read_csv(hist)
        d = d[d["image"] == "SAX"]
        for step_kind, g in (("acquired", d[d["step"] == 0]), ("aligned", d[d["is_best"] == 1])):
            g = g.drop_duplicates("slice").sort_values("slice")
            A = g[[f"a{r}{c}" for r in range(3) for c in range(4)]].to_numpy(float).reshape(-1, 3, 4)
            n = np.cross(A[:, :, 0], A[:, :, 1])
            n /= np.linalg.norm(n, axis=1, keepdims=True)
            c = A[:, :, :3] @ np.array([(X - 1) / 2, (Y - 1) / 2, 0.0]) + A[:, :, 3]
            for k in range(len(A) - 1):
                m = n[k] + n[k + 1]
                m /= np.linalg.norm(m)
                ang = np.degrees(np.arccos(np.clip(n[k] @ n[k + 1], -1, 1)))
                rows.append(dict(run=run_dir.name, subject_id=sid, kind=step_kind, slice=k, angle_deg=ang,
                                 spacing_mm=abs((c[k + 1] - c[k]) @ m)))
    return pd.DataFrame(rows)


def measure():
    geo = pd.concat([stack_geometry(OUT / name) for name in RUNS if (OUT / name).exists()])
    g = geo[geo.kind == "aligned"].groupby("run").agg(
        angle_mean=("angle_deg", "mean"), angle_max=("angle_deg", "max"),
        spacing_err_rms=("spacing_mm", lambda s: float(np.sqrt(((s - 10) ** 2).mean()))),
        spacing_min=("spacing_mm", "min"), spacing_max=("spacing_mm", "max"))
    acq = geo[(geo.kind == "acquired") & (geo.run == "base")]
    print(f"acquired: angle max {acq.angle_deg.max():.3f} deg, spacing {acq.spacing_mm.min():.2f}-{acq.spacing_mm.max():.2f} mm")
    rows = []
    for name in RUNS:
        s = pd.read_csv(OUT / name / "overall_metrics" / "motion_summary.csv", dtype={"subject_id": str})
        d = pd.read_csv(OUT / name / "overall_metrics" / "motion_summary_slices.csv", dtype={"subject_id": str})
        d = d[d.n_pairs > 0]
        dice = d.seg_dice.dropna().sort_values()
        rows.append(dict(run=name, dice=s.seg_dice_mean.mean(), dice_worst10=dice.iloc[:max(1, len(dice) // 10)].mean(),
                         ncc=s.ncc_mean.mean(), tilt_rms=s.sax_tilt_rms_deg.mean(), dz_rms=s.sax_dz_rms_mm.mean(), sec=s.seconds.mean()))
    out = pd.DataFrame(rows).set_index("run").join(g)
    pd.set_option("display.width", 200)
    print(out.round(4).to_string())


if __name__ == "__main__":
    {"run": run, "measure": measure}[sys.argv[1]]()
