"""Resumable grid search driver for motion_gradient.py.

Usage: python grid_search.py <runs.json>
runs.json: list of {"name": ..., "args": {option: value, ...}}; every run is launched with the BASE settings below,
overridden by its own args, so that each run pins every optimization setting explicitly.
A run is skipped when its summary already holds all subjects.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

PY = r"C:\Users\Nil\.conda\envs\ukbb\python.exe"
SCRIPT = r"C:\Users\Nil\Documents\git\UKBB_implicit_baselines\motion_gradient.py"
ROOT = r"C:\Users\Nil\Documents\git\UKBB_implicit_baselines\scratch_plots\gs_root5"
OUT = Path(r"C:\Users\Nil\Documents\data\ukbb_alignment_anneal_gs")
N_SUBJECTS = 5

BASE = {
    "steps": 600, "lr": 0.05, "rot-lr": 0.01, "lr-schedule": "constant", "lr-end": 0.0, "rot-lr-end": 0.0,
    "blur-sigma": 0.0, "blur-frac": 0.6, "pair-keep": 0.8, "eval-every": 10, "seed": 0, "dof": 6, "gauge": 0.01,
    "prior": 0.0, "mask-dilation-mm": 25.0, "mask-anneal-frac": 0.8, "anneal-stages": 6, "line-step-mm": 1.0, "line-extent-mm": 100.0, "min-line-mm": 80.0,
    "n-band": 8, "psf-model": "pulse", "psf-tbw": 2.5, "psf-scale-start": 1.0, "psf-anneal-frac": 0.6,
    "n-frames": 5, "metric": "ncc", "sax-thickness": 8.0, "lax-thickness": 6.0, "mask-dilation-start-mm": None,
    "history-every": 0.02,
}
FLAGS = ["--max-subjects", "-1", "--no-save-images", "--no-mask-plots", "--no-subject-plots", "--no-overall-plots"]


def done(name: str) -> bool:
    f = OUT / name / "overall_metrics" / "motion_summary.csv"
    return f.exists() and sum(1 for _ in open(f)) - 1 >= N_SUBJECTS


def main():
    runs = json.loads(Path(sys.argv[1]).read_text())
    log = open(OUT / "grid_search_log.txt", "a")
    for i, run in enumerate(runs):
        name = run["name"]
        if done(name):
            print(f"[{i + 1}/{len(runs)}] {name}: already done", flush=True)
            continue
        cfg = {**BASE, **run["args"]}
        cmd = [PY, SCRIPT, "--root", ROOT, "--out-root", str(OUT), "--run-name", name] + FLAGS
        for k, v in cfg.items():
            if v is not None:
                cmd += [f"--{k}", str(v)]
        t0 = time.time()
        res = subprocess.run(cmd, capture_output=True, text=True)
        status = "ok" if res.returncode == 0 and done(name) else f"FAILED ({res.returncode})"
        msg = f"[{i + 1}/{len(runs)}] {name}: {status} in {time.time() - t0:.0f} s"
        print(msg, flush=True)
        log.write(msg + "\n")
        if status != "ok":
            log.write(res.stdout[-3000:] + "\n" + res.stderr[-3000:] + "\n")
        log.flush()


if __name__ == "__main__":
    main()
