"""Overnight grid search of motion_gradient.py on all subjects, focused on the worst-aligned slices.

Usage:
  python night_gs.py run <stage> [--parallel N]   run every config of a stage (resumable; a run is done when its
                                                   DONE marker exists)
  python night_gs.py eval                          re-score every finished run with the fixed evaluator (resumable)
  python night_gs.py report [stage ...]            ranking table

Every run = base_config.json (the script defaults of 2026-10-25 + the user's SGD settings) overridden by the config.
Runs are made with --logging (history.csv holds the final poses). The evaluator re-scores those poses on ONE fixed
set of intersection lines for all runs (mask 20 mm, extent 150 mm, step 1 mm, min-line 20 mm, 10 frames, physical
PSF, no blur), so runs that change the mask or line length are still compared on the same lines.
Per run it reports mean Dice and the worst-case statistics over planes (SAX slices and LAX views with a Dice value):
mean of the worst 10% / 5%, number of planes with Dice < 0.5 and < 0.3, worst-10% Hausdorff.
"""
import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

PY = r"C:\Users\Nil\.conda\envs\ukbb\python.exe"
REPO = Path(r"C:\Users\Nil\Documents\git\UKBB_implicit_baselines")
ROOT = Path(r"C:\Users\Nil\Documents\data\ukbb_slices")             # single-slice layout (prepare_ukbb_slices.py)
OUT = Path(r"C:\Users\Nil\Documents\data\ukbb_alignment_night_gs")
BASE = {k: v for k, v in json.loads((OUT / "base_config.json").read_text()).items() if k != "instance"}  # no instance level
EVAL_CFG = dict(instance="Instance_2", n_frames=10, no_save_images=True, mask_dilation_mm=20.0, lax_thickness=6.0,
                sax_thickness=8.0, line_extent_mm=150.0, line_step_mm=1.0, min_line_mm=20.0, outside_mask_weight=0.0,
                metric="ncc", n_band=8, psf_model="gaussian", psf_tbw=2.5, seg_labels="1,2,3,4,5", sax_seg_labels="1,2,3",
                hd_miss_mm=100.0, mi_bins=24, mi_sigma=1.0)

sgd = lambda lr, rot, mom=0.0, **kw: {"optimizer": "sgd", "lr": lr, "rot-lr": rot, "momentum": mom, **kw}  # noqa: E731
opt = lambda name, lr, rot, **kw: {"optimizer": name, "lr": lr, "rot-lr": rot, **kw}  # noqa: E731

STAGES = {
    # A: optimizers and learning rates (everything else = base)
    "A": {
        "base": {},
        "sgd_lr10": sgd(10, 5), "sgd_lr50": sgd(50, 25), "sgd_lr100": sgd(100, 50),
        "sgd_lr25_rot6": sgd(25, 6), "sgd_lr25_rot25": sgd(25, 25),
        "sgdm_lr2.5": sgd(2.5, 1.2, 0.9), "sgdm_lr5": sgd(5, 2.5, 0.9), "sgdm_lr10": sgd(10, 5, 0.9),
        "sgdn_lr5": sgd(5, 2.5, 0.9, nesterov=True),
        "adam_lr0.05": opt("adam", 0.05, 0.01), "adam_lr0.1": opt("adam", 0.1, 0.02), "adam_lr0.2": opt("adam", 0.2, 0.04),
        "adam_lr0.2_lin": opt("adam", 0.2, 0.04, **{"lr-schedule": "linear"}),
        "radam_lr0.1": opt("radam", 0.1, 0.02), "radam_lr0.2": opt("radam", 0.2, 0.04),
        "adamax_lr0.1": opt("adamax", 0.1, 0.02), "adamax_lr0.2": opt("adamax", 0.2, 0.04),
        "nadam_lr0.2": opt("nadam", 0.2, 0.04),
        "adagrad_lr1": opt("adagrad", 1.0, 0.2), "adagrad_lr2": opt("adagrad", 2.0, 0.4),
        "rmsprop_lr0.1": opt("rmsprop", 0.1, 0.02),
        "sgd_lin_lr50": sgd(50, 25, **{"lr-schedule": "linear"}),
        "sgd_steps600": sgd(25, 12, steps=600), "sgd_steps2000": sgd(25, 12, steps=2000),
    },
}


def winner_stages() -> dict:
    """Stages B (regularization) and C (blur, PSF, line length, mask, sampling) around each stage-A winner listed in
    winners.json ({tag: overrides}); D (combinations + seeds) from d_configs.json ({name: overrides})."""
    stages = {"B": {}, "C": {}, "D": {}, "E": {}, "F": {}}
    wpath = OUT / "winners.json"
    if wpath.exists():
        for tag, w in json.loads(wpath.read_text()).items():
            B = {"seed1": {"seed": 1}, "seed2": {"seed": 2},
                 **{f"spacing{v:g}": {"spacing-reg": v} for v in (0.0, 3e-3, 3e-2)},
                 **{f"prior{v:g}": {"prior": v} for v in (1e-4, 1e-3)},
                 **{f"tilt{v:g}": {"tilt-prior": v} for v in (1e-3, 1e-2)},
                 **{f"normal{v:g}": {"normal-reg": v} for v in (1e-3, 1e-2)},
                 **{f"relrot{v:g}": {"rel-rot-reg": v} for v in (1e-3, 1e-2)}}
            C = {**{f"blur{v:g}": {"blur-sigma": v} for v in (0.0, 3.0, 12.0, 18.0)},
                 "blurfrac0.6": {"blur-frac": 0.6},
                 **{f"psf{v:g}": {"psf-scale-start": v} for v in (1.0, 5.0)},
                 "nopsf": {"no-psf": True},
                 **{f"extent{v:g}": {"line-extent-mm": v} for v in (40.0, 60.0, 100.0)},
                 **{f"mask{v:g}": {"mask-dilation-mm": v} for v in (15.0, 30.0, 40.0)},
                 "outside0.05": {"outside-mask-weight": 0.05},
                 **{f"minline{v:g}": {"min-line-mm": v} for v in (20.0, 60.0)},
                 "step2": {"line-step-mm": 2.0},
                 **{f"frames{v}": {"n-frames": v} for v in (5, 20)},
                 **{f"keep{v:g}": {"pair-keep": v} for v in (1.0, 0.6)}}
            for name, ov in B.items():
                stages["B"][f"{tag}-{name}"] = {**w, **ov}
            for name, ov in C.items():
                stages["C"][f"{tag}-{name}"] = {**w, **ov}
    for st in ("D", "E", "F"):
        path = OUT / f"{st.lower()}_configs.json"
        if path.exists():
            stages[st] = json.loads(path.read_text())
    return stages


def done(name: str) -> bool:
    return (OUT / name / "DONE").exists()


def launch(name: str, overrides: dict) -> str:
    if done(name):
        return f"{name}: already done"
    opts = {**BASE, **overrides}
    cmd = [PY, str(REPO / "motion_gradient.py"), "--root", str(ROOT), "--out-root", str(OUT), "--run-name", name,
           "--max-subjects", "-1", "--logging"]       # --logging: history.csv (final poses) for the evaluator
    for k, v in opts.items():
        if v is True:
            cmd.append(f"--{k}")
        elif v is not None and v is not False:
            cmd += [f"--{k}", str(v)]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        (OUT / f"FAILED_{name}.log").write_text(r.stdout[-5000:] + "\n" + r.stderr[-5000:])
        return f"{name}: FAILED ({r.returncode})"
    (OUT / name / "settings.json").write_text(json.dumps(overrides))
    (OUT / name / "DONE").write_text(f"{time.time() - t0:.0f}")
    return f"{name}: {time.time() - t0:.0f} s"


def run(stage: str, parallel: int):
    stages = {**STAGES, **{k: v for k, v in winner_stages().items() if v}}
    jobs = [(f"{stage}_{name}", ov) for name, ov in stages[stage].items()]
    log = open(OUT / "night_gs_log.txt", "a")
    with ThreadPoolExecutor(parallel) as ex:
        for msg in ex.map(lambda j: launch(*j), jobs):
            line = f"[{time.strftime('%H:%M:%S')}] {msg}"
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()


# --------------------------------------------------------------------------------------
# Fixed evaluator
# --------------------------------------------------------------------------------------

class Evaluator:
    def __init__(self):
        import torch
        sys.path.insert(0, str(REPO))
        import motion_gradient as mg
        self.torch, self.mg = torch, mg
        self.device = torch.device("cuda")
        self.args = SimpleNamespace(**EVAL_CFG)
        self.labels = [int(x) for x in self.args.seg_labels.split(",")]
        self.labels_sax = [int(x) for x in self.args.sax_seg_labels.split(",")]
        self.subjects = {}

    def subject(self, sid: str):
        if sid in self.subjects:
            return self.subjects[sid]
        mg, args, dev = self.mg, self.args, self.device
        data = mg.load_subject(ROOT / sid, args)
        if data["status"] != "ok":
            self.subjects[sid] = None
            return None
        sax, lax = mg.build_planes(data, args, dev)
        planes = sax + lax
        idx = {p.key: i for i, p in enumerate(planes)}
        with self.torch.no_grad():
            lines = [(idx[a.key], idx[b.key], ln) for a, b in [(s, l) for s in sax for l in lax] + list(combinations(lax, 2))
                     if (ln := mg.intersection_line(a, b, args, dev)) is not None]
        smp = mg.LineSampler(planes, lines, args.n_band, dev, args.psf_model, args.psf_tbw)
        has_seg = any(p.seg is not None for p in planes)
        zero = self.torch.zeros(len(planes), 3, device=dev)
        s = dict(planes=planes, idx=idx, smp=smp, has_seg=has_seg, n_lines=[len(x) for x in smp.plane_lines])
        s["acquired"] = self.per_plane(s, zero, zero)
        self.subjects[sid] = s
        return s

    def per_plane(self, s, theta, t):
        smp = s["smp"]
        with self.torch.no_grad():
            lm = smp.evaluate(theta, t, self.args, self.labels, self.labels_sax, s["has_seg"])
        return dict(dice=smp.per_plane(lm["dice"]), hd=smp.per_plane_max(lm["hd"]), ncc=smp.per_plane(lm["ncc"]))

    def evaluate_run(self, run_dir: Path) -> pd.DataFrame:
        torch = self.torch
        rows = []
        for hist in sorted(run_dir.glob("*/metrics/history.csv")):
            sid = hist.parent.parent.name
            s = self.subject(sid)
            if s is None:
                continue
            h = pd.read_csv(hist)
            h = h[h.is_best == 1] if (h.is_best == 1).any() else h[h.step == h.step.max()]
            h = h.drop_duplicates(["image", "slice"])
            N = len(s["planes"])
            theta = torch.zeros(N, 3, device=self.device)
            t = torch.zeros(N, 3, device=self.device)
            for r in h.itertuples():
                i = s["idx"].get(f"{r.image}:{int(r.slice)}")
                if i is None:
                    continue
                p = s["planes"][i]
                theta[i] = torch.tensor([r.rot_x_deg, r.rot_y_deg, r.rot_z_deg], device=self.device)
                t[i] = torch.tensor(r.dx_mm * p.u_hat + r.dy_mm * p.v_hat + r.dz_mm * p.normal, dtype=torch.float32, device=self.device)
            m = self.per_plane(s, theta, t)
            for i, p in enumerate(s["planes"]):
                if s["n_lines"][i] == 0:
                    continue
                rows.append(dict(run=run_dir.name, subject_id=sid, image=p.image, slice=p.slice_idx,
                                 dice=m["dice"][i], hd=m["hd"][i], ncc=m["ncc"][i],
                                 dice0=s["acquired"]["dice"][i], hd0=s["acquired"]["hd"][i], ncc0=s["acquired"]["ncc"][i]))
        return pd.DataFrame(rows)


def evaluate_all():
    path = OUT / "eval_slices.csv"
    old = pd.read_csv(path, dtype={"subject_id": str}) if path.exists() else pd.DataFrame(columns=["run"])
    todo = [p for p in sorted(OUT.iterdir()) if p.is_dir() and done(p.name) and p.name not in set(old.run)]
    if not todo:
        print("nothing to evaluate")
        return
    ev = Evaluator()
    new = []
    for p in todo:
        t0 = time.time()
        new.append(ev.evaluate_run(p))
        print(f"evaluated {p.name} in {time.time() - t0:.0f} s", flush=True)
        pd.concat([old] + new).to_csv(path, index=False, float_format="%.4f")


def summarize(ev: pd.DataFrame) -> pd.DataFrame:
    def stats(g):
        d = g.dice.dropna().sort_values().to_numpy()
        sax = g[g.image == "SAX"].dice.dropna().sort_values().to_numpy()
        hd_all = g.hd.dropna()
        miss = hd_all >= EVAL_CFG["hd_miss_mm"] - 1e-6               # a label hit by one line only (scored 100 mm)
        hd = hd_all[~miss].sort_values(ascending=False).to_numpy()
        k10, k5 = max(1, int(np.ceil(0.10 * len(d)))), max(1, int(np.ceil(0.05 * len(d))))
        return pd.Series(dict(n=len(d), dice=d.mean(), worst10=d[:k10].mean(), worst5=d[:k5].mean(),
                              sax_worst10=sax[:max(1, int(np.ceil(0.1 * len(sax))))].mean(),
                              n_below05=int((d < 0.5).sum()), n_below03=int((d < 0.3).sum()),
                              hd=hd_all.mean(), n_hd_miss=int(miss.sum()),
                              hd_worst10=hd[:max(1, int(np.ceil(0.1 * len(hd))))].mean(), ncc=g.ncc.mean()))
    return ev.groupby("run").apply(stats, include_groups=False)


def report_seeds(stage: str = "D"):
    """Mean +- sd over seeds of every stage-D config (runs named <stage>_<config>_s<seed>)."""
    ev = pd.read_csv(OUT / "eval_slices.csv", dtype={"subject_id": str})
    ev = ev[ev.run.str.startswith(f"{stage}_")]
    s = summarize(ev)
    s["config"] = s.index.str.replace(rf"^{stage}_", "", regex=True).str.replace(r"_s\d+$", "", regex=True)
    cols = ["dice", "worst10", "worst5", "sax_worst10", "n_below05", "n_below03", "n_hd_miss", "hd_worst10", "ncc"]
    g = s.groupby("config")[cols].agg(["mean", "std"])
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    g.insert(0, "seeds", s.groupby("config").size())
    pd.set_option("display.width", 300)
    print(g.sort_values("worst10_mean", ascending=False).round(3).to_string())
    g.to_csv(OUT / f"report_{stage}_seeds.csv", float_format="%.4f")
    return g


def report(stages):
    ev = pd.read_csv(OUT / "eval_slices.csv", dtype={"subject_id": str})
    if stages:
        ev = ev[ev.run.str.split("_").str[0].isin(stages)]
    s = summarize(ev)
    acq = summarize(ev.assign(dice=ev.dice0, hd=ev.hd0, ncc=ev.ncc0, run="as_acquired").drop_duplicates(["subject_id", "image", "slice"]))
    s = pd.concat([acq, s])
    secs = {p.name: float((p / "DONE").read_text()) for p in OUT.iterdir() if (p / "DONE").exists()}
    s["sec"] = [secs.get(r, np.nan) for r in s.index]
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    print(s.sort_values("worst10", ascending=False).round(3).to_string())
    s.to_csv(OUT / f"report_{'_'.join(stages) or 'all'}.csv", float_format="%.4f")
    return s


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "eval", "report", "seeds"])
    ap.add_argument("stages", nargs="*")
    ap.add_argument("--parallel", type=int, default=1)
    a = ap.parse_args()
    if a.cmd == "run":
        for st in a.stages:
            run(st, a.parallel)
    elif a.cmd == "eval":
        evaluate_all()
    elif a.cmd == "seeds":
        report_seeds(*(a.stages or ["D"]))
    else:
        report(a.stages)
