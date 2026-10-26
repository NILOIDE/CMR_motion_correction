"""Parameter search for the penultimate SAX slice (slice 9) of subject 1000071. Training stays on NCC; Dice is only
evaluated. Every configuration runs with several seeds (line dropping makes the outcome bimodal).

Usage: python s9_search.py run <stage> | report [stage ...]
Results: <OUT>/s9_results.csv (one row per run: slice-9 Dice/NCC/pose, slice-8 Dice, subject mean Dice/NCC).
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PY = r"C:\Users\Nil\.conda\envs\ukbb\python.exe"
REPO = Path(r"C:\Users\Nil\Documents\git\UKBB_implicit_baselines")
ROOT = REPO / "scratch_plots" / "grid_root"            # 1000071 only
OUT = Path(r"C:\Users\Nil\Documents\data\ukbb_alignment_s9search")
SID, SLICE = "1000071", 9
SEEDS = (0, 1, 2)

# every run pins all optimization settings
BASE = {
    "steps": 200, "lr": 0.2, "lr-end": 0.0, "rot-lr": 0.04, "rot-lr-end": 0.0, "lr-schedule": "linear",
    "blur-sigma": 6.0, "blur-frac": 0.6, "psf-scale-start": 3.0, "psf-anneal-frac": 0.6, "pair-keep": 0.8, "eval-every": 10,
    "dof": 6, "gauge": 0.01, "prior": 0.0, "tilt-prior": 0.0, "normal-reg": 0.0, "spacing-reg": 0.0, "sax-spacing-mm": 10.0,
    "rel-shift-reg": 0.0, "rel-rot-reg": 0.0, "mask-dilation-mm": 20.0, "outside-mask-weight": 0.0,
    "mask-anneal-frac": 0.8, "anneal-stages": 6, "line-step-mm": 1.0, "line-extent-mm": 150.0, "min-line-mm": 40.0,
    "sax-thickness": 8.0, "lax-thickness": 6.0, "n-band": 8, "psf-model": "gaussian", "psf-tbw": 2.5,
    "metric": "ncc", "n-frames": 5, "history-every": 0,
}

STAGES = {
    "s1": {   # one factor at a time around BASE
        "base": {},
        **{f"spacing{w:g}": {"spacing-reg": w} for w in (1e-3, 3e-3, 1e-2, 3e-2, 1e-1)},
        **{f"relshift{w:g}": {"rel-shift-reg": w} for w in (1e-3, 1e-2, 3e-2)},
        **{f"relrot{w:g}": {"rel-rot-reg": w} for w in (1e-3, 1e-2)},
        **{f"normal{w:g}": {"normal-reg": w} for w in (1e-3, 1e-2)},
        **{f"tilt{w:g}": {"tilt-prior": w} for w in (1e-3, 1e-2)},
        **{f"prior{w:g}": {"prior": w} for w in (1e-4, 1e-3)},
        "stack_all1e-3": {"normal-reg": 1e-3, "spacing-reg": 1e-3, "rel-shift-reg": 1e-3, "rel-rot-reg": 1e-3},
        "stack_all1e-2": {"normal-reg": 1e-2, "spacing-reg": 1e-2, "rel-shift-reg": 1e-2, "rel-rot-reg": 1e-2},
        "nopsf": {"no-psf": True},
        "psf1": {"psf-scale-start": 1.0},
        "psf5": {"psf-scale-start": 5.0},
        "psf_pulse": {"psf-model": "pulse"},
        "nband16": {"n-band": 16},
        **{f"blur{b:g}": {"blur-sigma": b} for b in (0.0, 12.0, 18.0)},
        **{f"mask{m:g}": {"mask-dilation-mm": m} for m in (10.0, 15.0, 30.0, 40.0)},
        "maskanneal100": {"mask-dilation-start-mm": 100.0},
        "outside0.05": {"outside-mask-weight": 0.05},
        **{f"minline{m:g}": {"min-line-mm": m} for m in (20.0, 80.0)},
        **{f"frames{f}": {"n-frames": f} for f in (1, 3, 10)},
        "ncc_grad": {"metric": "ncc_grad"},
        "mi": {"metric": "mi"},
        "steps400": {"steps": 400},
        "lr0.1": {"lr": 0.1, "rot-lr": 0.02},
        "rotlr0.08": {"rot-lr": 0.08},
        "rotlr0.01": {"rot-lr": 0.01},
        "dof3": {"dof": 3},
        "keep1": {"pair-keep": 1.0},
        "keep0.6": {"pair-keep": 0.6},
        "saxthick10": {"sax-thickness": 10.0},
        "laxthick8": {"lax-thickness": 8.0},
    },
    "s2": {   # combinations of the stage-1 winners
        **{f"sp{sp:g}_b{b:g}_pr{pr:g}_rr{rr:g}": {"spacing-reg": sp, "blur-sigma": b, "prior": pr, "rel-rot-reg": rr}
           for sp in (3e-3, 1e-2, 2e-2) for b in (6.0, 12.0, 18.0) for pr in (0.0, 1e-3) for rr in (0.0, 1e-2)},
        # PSF vs no PSF, with the in-plane blur annealing kept (no-psf alone also turns PSF annealing off)
        "nopsf_b18": {"no-psf": True, "blur-sigma": 18.0},
        "nopsf_b18_sp1e-2": {"no-psf": True, "blur-sigma": 18.0, "spacing-reg": 1e-2},
        "nopsf_b12_sp1e-2": {"no-psf": True, "blur-sigma": 12.0, "spacing-reg": 1e-2},
        "psf1_b18_sp1e-2": {"psf-scale-start": 1.0, "blur-sigma": 18.0, "spacing-reg": 1e-2},
        "pulse_b6_sp1e-2": {"psf-model": "pulse", "spacing-reg": 1e-2},
        "sp1e-2_normal1e-2": {"spacing-reg": 1e-2, "normal-reg": 1e-2},
        "sp1e-2_mask30": {"spacing-reg": 1e-2, "mask-dilation-mm": 30.0},
        "sp1e-2_frames10": {"spacing-reg": 1e-2, "n-frames": 10},
    },
}


def run_one(name: str, opts: dict, seed: int) -> dict | None:
    run_name = f"{name}_seed{seed}"
    out = OUT / run_name
    if not (out / SID / "metrics" / "slices.csv").exists():
        cmd = [PY, str(REPO / "motion_gradient.py"), "--root", str(ROOT), "--out-root", str(OUT), "--run-name", run_name,
               "--no-save-images", "--no-mask-plots", "--no-subject-plots", "--no-overall-plots", "--seed", str(seed)]
        for k, v in opts.items():
            if v is True:
                cmd.append(f"--{k}")
            elif v is not None and v is not False:
                cmd += [f"--{k}", str(v)]
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  {run_name}: FAILED\n{r.stderr[-1200:]}", flush=True)
            return None
        print(f"  {run_name}: {time.time() - t0:.0f} s", flush=True)
    d = pd.read_csv(out / SID / "metrics" / "slices.csv")
    s = pd.read_csv(out / SID / "metrics" / "summary.csv").iloc[0]
    sax = d[d.image == "SAX"].set_index("slice")
    r9 = sax.loc[SLICE]
    return dict(config=name, seed=seed, dice9=r9.seg_dice, ncc9=r9.ncc, dx9=r9.dx_mm, dy9=r9.dy_mm, dz9=r9.dz_mm,
                tilt9=r9.rot_tilt_deg, dice8=sax.loc[SLICE - 1].seg_dice, dice_mean=s.seg_dice_mean, ncc_mean=s.ncc_mean,
                n10=int(sax.loc[10].n_pairs) if 10 in sax.index else -1, seconds=s.seconds)


def run(stage: str):
    OUT.mkdir(parents=True, exist_ok=True)
    res_path = OUT / "s9_results.csv"
    rows = []
    for name, extra in STAGES[stage].items():
        opts = {**BASE, **extra}
        for seed in SEEDS:
            r = run_one(f"{stage}_{name}", opts, seed)
            if r:
                rows.append({**r, "stage": stage, "settings": json.dumps(extra)})
        old = pd.read_csv(res_path) if res_path.exists() else pd.DataFrame()
        new = pd.DataFrame(rows)
        allr = pd.concat([old, new]).drop_duplicates(["config", "seed"], keep="last")
        allr.to_csv(res_path, index=False, float_format="%.4f")


def report(stages=None):
    r = pd.read_csv(OUT / "s9_results.csv")
    if stages:
        r = r[r.stage.isin(stages)]
    g = r.groupby("config").agg(n=("seed", "count"), dice9_mean=("dice9", "mean"), dice9_min=("dice9", "min"),
                                dice9_max=("dice9", "max"), ok=("dice9", lambda x: int((x > 0.5).sum())),
                                ncc9=("ncc9", "mean"), dz9=("dz9", "mean"), tilt9=("tilt9", "mean"), dice8=("dice8", "mean"),
                                dice_mean=("dice_mean", "mean"), ncc_mean=("ncc_mean", "mean"))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 300)
    print(g.sort_values("dice9_mean", ascending=False).round(3).to_string())
    return g


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2])
    else:
        report(sys.argv[2:] or None)
