"""Compare grid-initialized vs gradient-only alignment on the local subjects."""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

sys.path.insert(0, r"C:\Users\Nil\Documents\git\UKBB_implicit_baselines")
import motion_misalignment as mm

ROOT = Path(r"C:\Users\Nil\Documents\git\UKBB_implicit_baselines\scratch_plots")
RUNS = {
    "grid + 200 steps": ROOT / "mm_full.csv",
    "gradient-only 200": ROOT / "ablation" / "grad200.csv",
    "gradient-only 600": ROOT / "ablation" / "grad600.csv",
    "gradient-only 600, lr 0.1": ROOT / "ablation" / "grad600_lr01.csv",
}
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)

summ, det = {}, {}
for name, path in RUNS.items():
    s, d = mm.load_results(path, path.with_name(path.stem + "_slices.csv"))
    summ[name], det[name] = s.set_index("subject_id"), d.set_index(["subject_id", "image", "slice"])
ref_name = "grid + 200 steps"
common = det[ref_name].index
for name in RUNS:
    common = common.intersection(det[name].index)
for name in RUNS:
    det[name] = det[name].loc[common]
base = det[ref_name]

print("=== Subject-level means (n = %d subjects, %d slices/views) ===" % (len(summ[ref_name]), len(common)))
rows = []
for name in RUNS:
    s, d = summ[name], det[name]
    sax = d[d.index.get_level_values("image") == "SAX"]
    rows.append(dict(
        method=name,
        ncc_acquired=s["ncc_zero_mean"].mean(), ncc_final=s["ncc_ref_mean"].mean(),
        dice_acquired=s["seg_dice_zero_mean"].mean(), dice_final=s["seg_dice_ref_mean"].mean(),
        sax_inplane_rms=s["sax_shift_rms_ref_mm"].mean(), sax_jitter=s["sax_shift_jitter_ref_mm"].mean(),
        sax_dz_rms=s["sax_dz_rms_ref_mm"].mean(), sax_tilt_rms=s["sax_tilt_rms_ref_deg"].mean(),
        slice_ncc_final=d["ncc_ref"].mean(), slice_ncc_p10=d["ncc_ref"].quantile(0.1),
        sec_per_subject=s["seconds"].mean(),
    ))
print(pd.DataFrame(rows).set_index("method").round(3).to_string())

print("\n=== Per-slice comparison against grid + 200 steps ===")
b_vec = base[["dx_ref_mm", "dy_ref_mm"]].values
for name in RUNS:
    if name == ref_name:
        continue
    d = det[name]
    dncc = d["ncc_ref"] - base["ncc_ref"]
    ddice = d["seg_dice_ref"] - base["seg_dice_ref"]
    dist = np.hypot(*(d[["dx_ref_mm", "dy_ref_mm"]].values - b_vec).T)
    print(f"{name:28s} dNCC mean {dncc.mean():+.3f}  worse(<-0.03) {100*(dncc < -0.03).mean():4.1f}%  better(>+0.03) {100*(dncc > 0.03).mean():4.1f}%  "
          f"| dDice mean {ddice.mean():+.3f} | solution differs by >3 mm in {100*(dist > 3).mean():4.1f}% of slices, >6 mm in {100*(dist > 6).mean():4.1f}%")

print("\n=== Dependence on the estimated misalignment (grid + refine in-plane shift, SAX slices only) ===")
sax_mask = base.index.get_level_values("image") == "SAX"
init = np.hypot(base["dx_ref_mm"], base["dy_ref_mm"])[sax_mask]
bins = [0, 2, 4, 7, 10, 100]
labels = ["0-2 mm", "2-4 mm", "4-7 mm", "7-10 mm", ">10 mm"]
cat = pd.cut(init, bins=bins, labels=labels)
table = pd.DataFrame({"n": cat.value_counts().reindex(labels)})
table["ncc acquired"] = base["ncc_zero"][sax_mask].groupby(cat, observed=False).mean()
for name in RUNS:
    d = det[name]
    table[f"ncc {name}"] = d["ncc_ref"][sax_mask].groupby(cat, observed=False).mean()
for name in RUNS:
    if name == ref_name:
        continue
    dist = pd.Series(np.hypot(*(det[name][["dx_ref_mm", "dy_ref_mm"]].values - b_vec).T), index=base.index)[sax_mask]
    table[f"|solution diff| {name}"] = dist.groupby(cat, observed=False).median()
print(table.round(3).to_string())

print("\n=== Subject ranking agreement with grid + 200 steps (Spearman rho) ===")
for name in RUNS:
    if name == ref_name:
        continue
    a, b = summ[ref_name], summ[name].reindex(summ[ref_name].index)
    print(f"{name:28s} jitter rho = {spearmanr(a['sax_shift_jitter_ref_mm'], b['sax_shift_jitter_ref_mm']).statistic:.2f}   "
          f"in-plane RMS rho = {spearmanr(a['sax_shift_rms_ref_mm'], b['sax_shift_rms_ref_mm']).statistic:.2f}   "
          f"final NCC rho = {spearmanr(a['ncc_ref_mean'], b['ncc_ref_mean']).statistic:.2f}")

# ---- figure -------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
st = mm.PLOT_STYLE
plt.rcParams.update({"font.family": "sans-serif", "figure.facecolor": st["surface"], "savefig.facecolor": st["surface"]})
colors = {ref_name: st["blue"], "gradient-only 200": st["orange"], "gradient-only 600": st["yellow"], "gradient-only 600, lr 0.1": st["violet"] if "violet" in st else "#6250d6"}
fig, axes = plt.subplots(1, 3, figsize=(17, 5), layout="constrained")
fig.get_layout_engine().set(rect=(0, 0, 1, 0.92))

ax = axes[0]
series = [(base["ncc_zero"], "as acquired", st["aqua"])] + [(det[n]["ncc_ref"], n, colors[n]) for n in RUNS]
mm._hist_series(ax, series, "Final NCC per slice / view", xlabel="NCC")

ax = axes[1]
x = np.arange(len(labels))
ax.plot(x, table["ncc acquired"], color=st["aqua"], marker="o", linewidth=2, label="as acquired")
for n in RUNS:
    ax.plot(x, table[f"ncc {n}"], color=colors[n], marker="o", linewidth=2, label=n)
ax.set_xticks(x, labels)
ax.set_xlabel("estimated in-plane misalignment of the SAX slice (grid + refine)")
ax.set_ylabel("mean final NCC")
ax.set_title("Final NCC vs initial misalignment (SAX slices)", fontsize=10, loc="left")
mm._style_axis(ax)
ax.yaxis.set_major_locator(matplotlib.ticker.AutoLocator())
for z, c in zip(x, table["n"]):
    ax.annotate(f"n={c}", xy=(z, 0), xycoords=("data", "axes fraction"), xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=7, color=st["muted"])

ax = axes[2]
for n in RUNS:
    if n == ref_name:
        continue
    dist = pd.Series(np.hypot(*(det[n][["dx_ref_mm", "dy_ref_mm"]].values - b_vec).T), index=base.index)[sax_mask]
    q = dist.groupby(cat, observed=False).quantile([0.25, 0.5, 0.75]).unstack()
    ax.fill_between(x, q[0.25], q[0.75], color=colors[n], alpha=0.15, linewidth=0)
    ax.plot(x, q[0.5], color=colors[n], marker="o", linewidth=2, label=n)
ax.set_xticks(x, labels)
ax.set_xlabel("estimated in-plane misalignment of the SAX slice (grid + refine)")
ax.set_ylabel("distance to grid + refine solution (mm)")
ax.set_title("How far the gradient-only solution lands from the grid-initialized one", fontsize=10, loc="left")
mm._style_axis(ax)
ax.yaxis.set_major_locator(matplotlib.ticker.AutoLocator())
handles, labs = axes[1].get_legend_handles_labels()
leg = fig.legend(handles, labs, loc="upper right", ncol=5, frameon=False, fontsize=9, bbox_to_anchor=(0.99, 0.995))
for t in leg.get_texts():
    t.set_color(st["ink2"])
fig.suptitle(f"Grid-search initialization vs gradient-only alignment  (n = {len(summ[ref_name])} subjects)", color=st["ink"], fontsize=12, x=0.02, ha="left")
out = ROOT / "ablation" / "grid_vs_gradient.png"
fig.savefig(out, dpi=150)
print("\nWrote", out)
