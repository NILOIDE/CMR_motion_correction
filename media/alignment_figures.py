"""
Cohort figures of an alignment run (moco_nifti.py with --logging), for the README. Ported from the survey of
UKBB_implicit_baselines (data/analysis/alignment_figures.py, with the chart helpers of survey_figures.py and tools/qc.py);
the data side is alignment_stats.py.
  alignment.png            the overview: who is aligned and how, NCC per subject before / after, the lowest NCC of
                           a kept slice after the alignment (alignment quality)
  alignment_translation.png  every SAX slice's translation against the rest of its stack -- in-plane, through-plane,
  alignment_rotation.png     total -- and rotation -- in-plane, total --, one row per measure: per slice by position
                           (kept, beyond the base / apex), by slice position from the apical end of the kept range, per
                           kept subject the largest over its kept slices and over all slices
  alignment_dice.png       the run's Dice (SAX labels against the long-axis views' labels along the intersection lines)
                           before / after: per kept slice, by LV region, by slice position, mid-ventricular per subject
  alignment_example_<n>.png  one figure per example subject: 1 the largest kept-slice motion, 2 EXAMPLE_SUBJECT (a
                           mid-ventricular slice that moved; else a typical subject), 3 the lowest kept-slice NCC.
                           Per slice the motion against the stack, the NCC and the Dice (y ranges shared by the
                           figures); the SAX slices' intersection lines on one long-axis view (4Ch, else 2Ch / 3Ch),
                           acquired and aligned; the SAX stack resliced onto that plane with the acquired and the
                           aligned poses (the images from --root, the poses from the run's metrics/history.csv)

Usage: the kept slices from the SAX segmentations under --root (run it where the data is), or from an existing
survey of UKBB_implicit_baselines (--survey: its survey.csv and slices.csv); the examples need the images of the
example subjects under --root. The run may be remote (user@host:/path; its metrics and histories are copied).
    python media/alignment_figures.py <alignment_dir>/<run> --root <ukbb_dir> --workers 16
    python media/alignment_figures.py user@host:/path/<run> --survey <survey_dir> --root <ukbb_dir> \
        --examples auto
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from alignment_stats import (ALIGN_NCC_MIN, FULL_MODEL, MOTION_MAX_MM, assemble, fetch_source, history,  # noqa: E402
                             import_run, kept_ranges, lv_table, poses, sax_planes, survey_kept)

# ------------------------------------------------------------------------------- chart style (survey_figures.py)

L_SURFACE, L_INK, L_INK2, L_GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e5e4df"
EXTRA = ("#1baf7a", "#eda100", "#e87ba4", "#6250d6", "#008300")        # series without a fixed colour
INK, INK2, SURFACE = "#f0efec", "#c3c2b7", "#1a1a19"                    # the dark surface of the image panels
LABEL_COLORS = ("#3987e5", "#d95926", "#199e70")                        # LV blood pool, LV myocardium, RV blood pool
ACQ, ALI = "#eda100", "#5aa9e6"             # acquired / aligned on the dark surface
ACQ_L, ALI_L = "#eda100", "#2a78d6"         # the same on the light surface of the cohort charts
KEPT_NAME = "kept (LV in the SAX segmentation)"       # (set by main: the survey's verdict with --survey)
KEPT_SLICES = "the slices with LV in the SAX segmentation"                 # (--survey: the survey's kept slices)
REGION_LABEL = "region (the kept range in thirds)"                         # (--survey: from the long-axis view)
NAMED_COLORS = {"acquired poses": ACQ_L, "aligned": ALI_L, "all slices": EXTRA[3], "in the alignment run": "#8a8984",
                KEPT_NAME: EXTRA[0]}
VIEW_ORDER = (("4ch", "LAX_4Ch"), ("2ch", "LAX_2Ch"), ("3ch", "LAX_3Ch"))   # the long-axis view of the examples
EXAMPLE_SUBJECTS = ("<subject-id>", "<subject-id>", "<subject-id>")   # the README's examples (default --examples), in the order
                                                        # of EXAMPLE_ROLES / EXAMPLE_TITLES
EXAMPLE_SUBJECT = EXAMPLE_SUBJECTS[1]       # --examples auto, second example: a large motion of a mid-ventricular
                                            # slice; the most typical kept subject when it is not available
EXAMPLE_WHY = "a mid-ventricular slice moved"
EXAMPLE_ROLES = ("largest kept-slice motion", EXAMPLE_WHY, "lowest kept-slice NCC")   # --examples, in this order
EXAMPLE_TITLES = ("Subject with large apical misalignment",                     # the titles of the --examples figures
                  "Subject with large 4CH misalignment (all intersections with SA slices improve in alignment)",
                  "Subject with good initial alignment (motion correction hardly changes parameters)")
POSITIONS = ("kept", "beyond base", "beyond apex")
REGIONS = ("beyond base", "basal", "mid", "apical", "beyond apex")
# the motion of a SAX slice against the rest of its stack: column, name, unit, signed (shown as magnitude)
TRANSLATION = (("align_rel_shift_mm", "in-plane translation", "mm", False),
               ("align_rel_dz_mm", "through-plane translation", "mm", True),
               ("align_rel_shift3d_mm", "total translation", "mm", False))
ROTATION = (("align_rel_rot_inplane_deg", "in-plane rotation", "deg", True),
            ("align_rel_rot_deg", "total rotation", "deg", False))


def _color(name: str, i: int) -> str:
    return NAMED_COLORS.get(name) or EXTRA[i % len(EXTRA)]


def capitalize_titles(fig) -> None:
    """Start the figure title and every axes title with a capital letter."""
    def cap(text: str) -> str:
        i = next((k for k, ch in enumerate(text) if not ch.isspace()), None)
        return text if i is None else text[:i] + text[i].upper() + text[i + 1:]
    if getattr(fig, "_suptitle", None) is not None:
        fig._suptitle.set_text(cap(fig._suptitle.get_text()))
    for a in fig.axes:
        for t in (a.title, a._left_title, a._right_title):
            if t.get_text():
                t.set_text(cap(t.get_text()))


def _ax(a, title: str, xlabel: str = "", ylabel: str = "subjects") -> None:
    a.set_facecolor(L_SURFACE)
    a.set_title(title, color=L_INK, fontsize=9, loc="left")
    a.set_xlabel(xlabel, fontsize=8, color=L_INK2)
    a.set_ylabel(ylabel, fontsize=8, color=L_INK2)
    a.tick_params(colors=L_INK2, labelsize=7)
    a.grid(axis="y", color=L_GRID, lw=0.6)
    a.set_axisbelow(True)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        a.spines[sp].set_color(L_INK2)


def _edge_labels(edges) -> list[str]:
    """Category labels of bins: integers for unit bins centred on integers ([k - 0.5, k + 0.5)), ranges otherwise."""
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if float(lo + 0.5).is_integer() and float(hi + 0.5).is_integer():
            a, b = int(lo + 0.5), int(hi - 0.5)
            out.append(str(a) if a == b else f"{a}-{b}" if np.isfinite(hi) else f">= {a}")
        else:
            out.append(f"{(lo + hi) / 2:g}" if np.isfinite(hi) else f">= {lo:g}")      # bin centre
    return out


def _label_small(a, bars, log: bool, frac: float = 0.06, everything: bool = False) -> None:
    """Counts above the bars too small to read (all bars with `everything`)."""
    ymax = max((b.get_height() for b in bars), default=0)
    for b in bars:
        h = b.get_height()
        if h <= 0:
            continue
        if everything or (not log and h < frac * ymax) or (log and h < 3):
            a.text(b.get_x() + b.get_width() / 2, max(h, 0), f"{h:.0f}" if float(h).is_integer() else f"{h:.1f}",
                   ha="center", va="bottom", fontsize=5.5, color=L_INK2, rotation=90 if b.get_width() < 0.12 else 0)


def _cat(a, categories, values: dict, title, xlabel="", ylabel="subjects", total_width=0.8, log=False,
         label_all=False, tick_every=1, rotate=0, legend_counts=True) -> list:
    """Categorical bar chart: one narrow bar per series side by side at every category (no overlap). values: name ->
    counts per category. Bars too small to read get their count printed (all of them with label_all)."""
    x = np.arange(len(categories))
    k = max(len(values), 1)
    w = total_width / k
    all_bars = []
    for i, (name, v) in enumerate(values.items()):
        bars = a.bar(x + (i - (k - 1) / 2) * w, v, w * 0.92, color=_color(name, i),
                     label=(f"{name} ({int(np.sum(v))})" if legend_counts else name) if k > 1 else None)
        all_bars.append(bars)
    if log:
        a.set_yscale("log")
    flat = [b for bars in all_bars for b in bars]
    _label_small(a, flat, log, everything=label_all)
    a.set_xticks(x[::tick_every], [str(c) for c in categories][::tick_every], fontsize=6.5, rotation=rotate,
                 ha="right" if rotate else "center")
    _ax(a, title, xlabel, ylabel)
    if k > 1:
        a.legend(frameon=False, fontsize=7)
    return flat


def _hist2(a, data: dict, edges, title, xlabel, ylabel="subjects", log=False, labels=None, tick_every=None,
           label_all=False, tight=False) -> np.ndarray:
    """Distribution of one or more series over bins, drawn categorically: each bin is a category on the x axis and
    every series has its own narrow bar there, side by side (no overlapping outlines). Returns the bin edges."""
    series = {}
    for name, v in data.items():
        if v is None:
            continue
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        if len(v):
            series[name] = v
    edges = np.asarray(edges, float)
    counts = {}
    for name, v in series.items():
        clipped = np.clip(v, edges[0], np.nextafter(edges[-1], -np.inf)) if np.isfinite(edges[-1]) else np.maximum(v, edges[0])
        top = max(float(clipped.max()), float(edges[np.isfinite(edges)].max())) + 1
        counts[name] = np.histogram(clipped, np.where(np.isfinite(edges), edges, top))[0]
    cats = labels or _edge_labels(edges)
    if tight and counts:                                    # drop empty bins at both ends, keep one bin of margin
        tot = np.sum(list(counts.values()), axis=0)
        nz = np.flatnonzero(tot)
        if len(nz):
            lo, hi = max(nz[0] - 1, 0), min(nz[-1] + 2, len(tot))
            counts = {k: v[lo:hi] for k, v in counts.items()}
            cats = cats[lo:hi]
            edges = edges[lo:hi + 1]
    if tick_every is None:
        tick_every = max(1, -(-len(cats) // 12))                   # at most ~12 tick labels
    _cat(a, cats, counts, title, xlabel, ylabel, log=log, tick_every=tick_every, label_all=label_all,
         rotate=45 if len(cats) > 8 and tick_every == 1 else 0)
    if len(series) == 1:
        a.legend([f"{n} ({len(v)})" for n, v in series.items()], frameon=False, fontsize=7)
    return edges


def _vline(a, edges, value, text: str) -> None:
    """A dashed line at the bin boundary `value` of a categorical histogram (_hist2), with a label."""
    i = int(np.argmin(np.abs(np.asarray(edges) - value)))
    a.axvline(i - 0.5, color=L_INK, ls="--", lw=1)
    a.text(i - 0.4, 0.03, text, fontsize=7, color=L_INK2, transform=a.get_xaxis_transform())


def _bars(a, labels, values: dict, title, ylabel="subjects", width=0.8, rotate=0) -> None:
    """Categorical bars with every count printed (few categories)."""
    _cat(a, labels, values, title, "", ylabel, total_width=width, label_all=True, rotate=rotate)


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    capitalize_titles(fig)
    fig.savefig(path, dpi=110, facecolor=L_SURFACE)
    print(f"  {path}")


def mask_edges(mask: np.ndarray) -> np.ndarray:
    """The boundary of a 2D mask along the pixel edges: (N, 2, 2) segments of corner coordinates (i, j)."""
    p = np.pad(np.asarray(mask, bool), 1)
    ii, jj = np.nonzero(p[1:, 1:-1] != p[:-1, 1:-1])
    v = np.stack([np.stack([ii - 0.5, jj - 0.5], 1), np.stack([ii - 0.5, jj + 0.5], 1)], 1)
    ii, jj = np.nonzero(p[1:-1, 1:] != p[1:-1, :-1])
    h = np.stack([np.stack([ii - 0.5, jj - 0.5], 1), np.stack([ii + 0.5, jj - 0.5], 1)], 1)
    return np.concatenate([v, h], 0)


def label_outlines(ax, seg: np.ndarray, labels: tuple, lw: float = 1.4, **kw) -> None:
    """Pixel-edge outlines of the three labels (myocardium first: its inner outline is drawn over by the LV's)."""
    from matplotlib.collections import LineCollection
    for k in (1, 0, 2):
        segs = mask_edges(seg == labels[k])
        if len(segs):
            ax.add_collection(LineCollection(segs, colors=LABEL_COLORS[k], linewidths=lw, capstyle="projecting", **kw))


# ------------------------------------------------------------------------------------------------ cohort figures

def _count(n: int) -> str:
    """A slice / subject count for a tick label: 950, 23.2k."""
    return str(n) if n < 1000 else f"{n / 1000:.1f}k"


def _medians(a, frame, key, cats, ticks, cols: dict, title, xlabel, ylabel, cat_color=None, lo=.25, hi=.95) -> None:
    """Medians per category of frame[key] (cats, labelled `ticks` and the row count): one narrow bar per column (cols:
    name -> (column, colour)) side by side, whiskers p{lo}-p{hi}; with a single column cat_color(category) colours
    the bars instead."""
    k = len(cols)
    w = 0.8 / k
    labels = []
    for xi, c in enumerate(cats):
        g = frame[frame[key] == c]
        labels.append(f"{ticks[xi]}\n{_count(len(g))}")
        for i, (col, colour) in enumerate(cols.values()):
            x = g[col].dropna()
            if not len(x):
                continue
            xo, m = xi + (i - (k - 1) / 2) * w, x.median()
            a.bar(xo, m, w * 0.92, color=cat_color(c) if k == 1 and cat_color else colour)
            a.errorbar(xo, m, yerr=[[m - x.quantile(lo)], [x.quantile(hi) - m]], fmt="none", ecolor=L_INK2,
                       elinewidth=0.8, capsize=2)
    a.set_xticks(range(len(cats)), labels, fontsize=6.5)
    _ax(a, title, xlabel, ylabel)


def _by_offset(a, p, cols: dict, title: str, ylabel: str, lo=.25, hi=.95) -> None:
    """_medians by slice position from the apical end of the kept range (-9 .. +3); a single column coloured kept /
    beyond the apex."""
    from matplotlib.patches import Patch
    q = p[p.from_apex.notna()]
    offs = list(range(max(int(q.from_apex.min()), -9), min(int(q.from_apex.max()), 3) + 1)) if len(q) else []
    _medians(a, q, "from_apex", offs, [f"+{o}" if o > 0 else str(o) for o in offs], cols, title,
             "slices from the apical end of the kept range (0 = the apical kept slice; slices below)", ylabel,
             cat_color=lambda o: EXTRA[0] if o <= 0 else EXTRA[2], lo=lo, hi=hi)
    handles = ([Patch(color=EXTRA[0], label="kept slices"), Patch(color=EXTRA[2], label="beyond the apex")]
               if len(cols) == 1 else [Patch(color=c, label=n) for n, (_, c) in cols.items()])
    a.legend(handles=handles, frameon=False, fontsize=7, loc="upper left")


def fig_alignment(df, series, out: Path) -> None:
    """alignment.png, alignment_translation.png, alignment_rotation.png, alignment_dice.png (module docstring)."""
    from matplotlib.figure import Figure
    st, model = df.align_status.astype(str), df.align_pose_model.astype(str)
    full_m = (st == "ok") & (model == FULL_MODEL)
    full = df[full_m]
    kept = full[full.kept]
    # overview
    has_ao = df.has_ao
    groups = [full_m & has_ao, full_m & ~has_ao, (st == "ok") & (model != FULL_MODEL),
              st.isin(["not_optimized", "no_images"]), st.str.startswith("error")]
    cats = ["6-DOF,\nwith aorta", "6-DOF,\nno aortic image", "in-plane only\n(no SAX stack)",
            "not optimised\n(no long-axis view)", "error"]
    fig = Figure(figsize=(18, 4.9), facecolor=L_SURFACE)
    axs = fig.subplots(1, 3)
    _bars(axs[0], cats, {"in the alignment run": [int(g.sum()) for g in groups],
                         KEPT_NAME: [int((g & df.kept).sum()) for g in groups]},
          f"pose model ({len(df)} subjects in the run)")
    edges = [-1.0] + list(np.round(np.arange(0.35, 1.0001, 0.05), 2))       # below the first edge: clipped into it
    _hist2(axs[1], {"acquired poses": full.align_ncc_before, "aligned": full.align_ncc_after}, edges,
           f"NCC per subject (median {full.align_ncc_before.median():.2f} -> {full.align_ncc_after.median():.2f})",
           "mean NCC over the intersection lines of the slices and views (bins of 0.05)",
           labels=["< 0.35"] + [f"{a:.2f}" for a in edges[1:-1]], tight=True)
    nmi = int((full.align_nmi_gain > 0).sum())
    axs[1].text(0.02, 0.97, f"NMI (not optimised) rises in {nmi} of {len(full)} subjects", transform=axs[1].transAxes,
                fontsize=7, color=L_INK2, va="top")
    ok, prob = kept.align_ok, kept.align_problem.fillna("").astype(str)
    n_ncc = int(prob.str.contains("NCC").sum())
    edges = [-1.0] + list(np.round(np.arange(0.2, 1.0001, 0.05), 2))
    e = _hist2(axs[2], {"kept slices": kept.kept_align_slice_ncc_min, "all slices": kept.align_slice_ncc_min}, edges,
               f"lowest NCC of a kept slice after alignment ({n_ncc} of {len(kept)} below {ALIGN_NCC_MIN:g})",
               "NCC of the slice, mean over its lines (bins of 0.05)",
               labels=["< 0.2"] + [f"{a:.2f}" for a in edges[1:-1]], tight=True, log=True)
    _vline(axs[2], e, ALIGN_NCC_MIN, f" limit {ALIGN_NCC_MIN:g}")
    npass = int((ok == True).sum())                                             # noqa: E712
    axs[2].text(0.02, 0.86, f"kept and aligned: {len(kept)}\nalignment trustworthy: {npass} "
                f"({100 * npass / max(len(kept), 1):.0f} %)", transform=axs[2].transAxes, fontsize=7,
                color=L_INK2, va="top", ha="left")
    fig.suptitle(f"Slice alignment of {len(full)} subjects", color=L_INK, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, out / "alignment.png")
    # motion against the stack, and Dice
    p = _positions(df, series[series.subject.isin(full.index)])
    for rows, name, what in ((TRANSLATION, "alignment_translation.png", "translation"),
                             (ROTATION, "alignment_rotation.png", "rotation")):
        _fig_motion(p, kept.index, rows, out / name,
                    f"SAX slice {what} against the rest of its stack ({len(full)} subjects; kept: {KEPT_SLICES})")
    _fig_dice(p, kept, out / "alignment_dice.png")


def _positions(df, series):
    """SAX rows of the series table with alignment values and their position against the kept range."""
    s = series[(series.kind == "sax") & series.slice.notna()]
    s = s[s.get("align_rel_shift_mm").notna() | s.get("align_ncc_after").notna()]
    return sax_planes(s, df[["base", "apex", "dir"]].dropna().astype(int))


def _fig_motion(p, kept_ids, rows: tuple, path: Path, title: str) -> None:
    """One row per measure (rows: column, name, unit, signed): per SAX slice by position, by slice position from
    the apical end of the kept range, per kept subject the largest over its kept slices and over all slices; the
    motion limit on the in-plane translation."""
    from matplotlib.figure import Figure
    fig = Figure(figsize=(18, 4.5 * len(rows) + 0.6), facecolor=L_SURFACE)
    axs = fig.subplots(len(rows), 3, squeeze=False)
    for (col, name, unit, signed), (a0, a1, a2) in zip(rows, axs):
        if col not in p:
            continue
        x = p.assign(value=p[col].abs() if signed else p[col])
        step, top = (1.0, 20) if unit == "mm" else (0.5, 12)
        edges = list(np.arange(0, top, step)) + [np.inf]
        lim = MOTION_MAX_MM if col == "align_rel_shift_mm" else None
        mag = "|" if signed else ""
        e = _hist2(a0, {pos: x[x.pos == pos].value for pos in POSITIONS}, edges, f"per SAX slice: {name}",
                   f"{mag}{name}{mag} ({unit}, bins of {step:g})", "slices", log=True)
        if lim:
            _vline(a0, e, lim, f" limit {lim:g}")
        _by_offset(a1, x, {name: ("value", None)}, f"{name} by slice position (median, p25-p95)", unit)
        top_k = x[x.pos == "kept"].groupby("subject").value.max().reindex(kept_ids)
        top_all = x.groupby("subject").value.max().reindex(kept_ids)
        over = f" ({int((top_k > lim).sum())} of {int(top_k.notna().sum())} beyond {lim:g} mm)" if lim else ""
        e = _hist2(a2, {"kept slices": top_k, "all slices": top_all}, edges, f"per kept subject: the largest {name}{over}",
                   f"{unit} (bins of {step:g})", log=True)
        if lim:
            _vline(a2, e, lim, f" limit {lim:g}")
            m = top_k.dropna()
            if len(m):
                lines = [f"{s} %: above {m.quantile(1 - s / 100):.1f} mm" for s in (5, 10, 20)]
                a2.text(0.98, 0.6, "to leave out the most moving\n" + "\n".join(lines), transform=a2.transAxes,
                        fontsize=7, color=L_INK2, va="top", ha="right")
    fig.suptitle(title, color=L_INK, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.45 / (4.5 * len(rows) + 0.6)))
    _save(fig, path)


def _fig_dice(p, kept, path: Path) -> None:
    """alignment_dice.png: the run's Dice before / after per kept SAX slice, by LV region of the slice (the kept
    range in thirds), by slice position from the apical end of the kept range, and per kept subject over its kept
    mid-ventricular slices."""
    from matplotlib.figure import Figure
    from matplotlib.patches import Patch
    if "align_dice_after" not in p:
        return
    cols = {"acquired poses": ("align_dice_before", ACQ_L), "aligned": ("align_dice_after", ALI_L)}
    fig = Figure(figsize=(16, 9.2), facecolor=L_SURFACE)
    axs = fig.subplots(2, 2)
    k = p[(p.pos == "kept") & p.subject.isin(kept.index)]
    edges = [-1.0] + list(np.round(np.arange(0.3, 1.0001, 0.05), 2))
    _hist2(axs[0, 0], {n: k[c] for n, (c, _) in cols.items()}, edges, "per kept SAX slice of the kept subjects",
           "Dice (bins of 0.05)", "slices", labels=["< 0.3"] + [f"{a:.2f}" for a in edges[1:-1]], tight=True)
    legend = [Patch(color=c, label=n) for n, (_, c) in cols.items()]
    q = p[p.subject.isin(kept.index)]
    _medians(axs[0, 1], q, "region", list(REGIONS), list(REGIONS), cols,
             "per SAX slice of the kept subjects, by LV region (median, p25-p75)", REGION_LABEL,
             "Dice", hi=.75)
    axs[0, 1].legend(handles=legend, frameon=False, fontsize=7, loc="upper right")
    _by_offset(axs[1, 0], q, cols, "by slice position (median, p25-p75)", "Dice", hi=.75)
    _hist2(axs[1, 1], {"acquired poses": kept.mid_dice_before, "aligned": kept.mid_dice_after},
           edges, "per kept subject: mean over its kept mid-ventricular slices", "Dice (bins of 0.05)",
           labels=["< 0.3"] + [f"{a:.2f}" for a in edges[1:-1]], tight=True)
    fig.suptitle(f"Dice of the SAX labels against the long-axis views' along the intersection lines ({len(kept)} kept "
                 f"subjects)", color=L_INK, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, path)


# ------------------------------------------------------------------------------------------------------- examples

def reslice(stack: np.ndarray, affines: list, lax_aff: np.ndarray, shape: tuple, frame: int = 0,
            reach: float = 0.75) -> np.ndarray:
    """The SAX stack (X, Y, Z, T) seen in a long-axis plane: every pixel (i, j) of the plane (lax_aff, shape (X, Y))
    takes the nearest SAX slice within `reach` slice steps (affines: per slice voxel -> world, the third column the
    step; 0.75: no gaps where aligned slices drift apart), sampled bilinearly at `frame`; NaN where none is near."""
    from scipy.ndimage import map_coordinates
    X, Y = shape
    ii, jj = np.meshgrid(np.arange(X), np.arange(Y), indexing="ij")
    P = lax_aff @ np.stack([ii.ravel(), jj.ravel(), np.zeros(ii.size), np.ones(ii.size)])
    out, best = np.full(ii.size, np.nan, np.float32), np.full(ii.size, np.inf)
    Xs, Ys = stack.shape[:2]
    for k, A in enumerate(affines):
        if A is None:
            continue
        q = np.linalg.inv(A) @ P
        w = np.abs(q[2])
        m = (w <= reach) & (w < best) & (q[0] >= -0.5) & (q[0] <= Xs - 0.5) & (q[1] >= -0.5) & (q[1] <= Ys - 0.5)
        if m.any():
            out[m] = map_coordinates(stack[:, :, k, frame].astype(np.float32), q[:2, m], order=1, mode="nearest")
            best[m] = w[m]
    return out.reshape(X, Y)


def plane_line(A_sax: np.ndarray, A_lax: np.ndarray, sax_shape: tuple, box: tuple) -> np.ndarray | None:
    """Where a SAX slice's plane cuts a long-axis view, in the view's pixels (i, j): the part inside `box`
    (i0, i1, j0, j1) and inside the SAX image; None if they do not meet there."""
    n = np.cross(A_sax[:3, 0], A_sax[:3, 1])
    n /= np.linalg.norm(n)
    a, b = n @ A_lax[:3, 0], n @ A_lax[:3, 1]
    h = float(np.hypot(a, b))
    if h < 1e-6:
        return None
    c = n @ (A_sax[:3, 3] - A_lax[:3, 3])
    p0, d = c * np.array([a, b]) / h ** 2, np.array([-b, a]) / h
    pts = p0 + np.arange(-800, 800, 0.5)[:, None] * d
    i0, i1, j0, j1 = box
    m = (pts[:, 0] >= i0) & (pts[:, 0] <= i1) & (pts[:, 1] >= j0) & (pts[:, 1] <= j1)
    W = A_lax @ np.stack([pts[:, 0], pts[:, 1], np.zeros(len(pts)), np.ones(len(pts))])
    q = np.linalg.inv(A_sax) @ W
    m &= (q[0] >= -0.5) & (q[0] <= sax_shape[0] - 0.5) & (q[1] >= -0.5) & (q[1] <= sax_shape[1] - 0.5)
    return pts[m] if m.sum() > 1 else None


def _candidates(df, root: Path) -> list[tuple[str, list[str]]]:
    """Kept, fully aligned subjects with images under root, ranked per role, in the order of the examples: the
    largest kept-slice motion, EXAMPLE_SUBJECT (else the typical one: largest kept-slice motion nearest the median),
    the lowest kept-slice NCC."""
    c = df[df.kept & (df.align_pose_model == FULL_MODEL) & df.motion_mm.notna()]
    c = c[[(Path(root) / s).exists() for s in c.index]]
    if c.empty:
        return []
    sh, nc = c.motion_mm, c.kept_align_slice_ncc_min
    typical = list((sh - sh.median()).abs().sort_values().index)
    example = (EXAMPLE_WHY, [EXAMPLE_SUBJECT] + typical) if EXAMPLE_SUBJECT in c.index else ("typical", typical)
    return [("largest kept-slice motion", list(sh.sort_values(ascending=False).index)),
            example,
            ("lowest kept-slice NCC", list(nc.sort_values().index))]


def _values(rows: list, cols: tuple) -> np.ndarray:
    """All finite values of some series columns over the examples' slice tables."""
    v = [s[c].to_numpy(float) for s in rows for c in cols if c in s]
    v = np.concatenate(v) if v else np.zeros(0)
    return v[np.isfinite(v)]


def _chart_limits(tables: list) -> dict:
    """y ranges shared by every example: distance from the stack 0 .. the largest (or the limit), NCC and Dice from
    the lowest value (in steps of 0.05, at most 0.4; NCC down to its limit) to 1."""
    dist = _values(tables, ("align_rel_shift_mm",))
    lo = lambda v, floor=1.0: min(np.floor((min(v.min(), floor) - 0.02) * 20) / 20, 0.4) if len(v) else 0.4  # noqa: E731
    return dict(dist=(0, 1.08 * max(dist.max() if len(dist) else 0, MOTION_MAX_MM)),
                ncc=(lo(_values(tables, ("align_ncc_before", "align_ncc_after")), ALIGN_NCC_MIN), 1.0),
                dice=(lo(_values(tables, ("align_dice_before", "align_dice_after"))), 1.0))


def _slice_chart(axs, s, lo_k: int, hi_k: int, title: str, lims: dict) -> None:
    """Per SAX slice, three panels on a shared slice axis: the distance from the stack (bars; kept slices in colour,
    the motion limit dashed), the NCC and the Dice along its intersection lines with the acquired and the aligned
    poses (NCC limit dashed); the kept range (lo_k .. hi_k) shaded; y ranges `lims` (the same in every example)."""
    a_dist, a_ncc, a_dice = axs
    s = s.sort_values("slice")
    k = s.slice.to_numpy(int)
    kept = (k >= lo_k) & (k <= hi_k)
    for a in axs:
        a.set_facecolor(SURFACE)
        a.tick_params(colors=INK2, labelsize=6)
        for sp in a.spines.values():
            sp.set_color(INK2)
        a.set_xlim(k.min() - 0.6, k.max() + 0.6)
        a.set_xticks([])
    a_dist.bar(k, s.align_rel_shift_mm, 0.7, color=np.where(kept, ALI, "#6b6a65"))
    a_dist.axhline(MOTION_MAX_MM, color=INK, ls="--", lw=0.8)
    a_dist.set_ylim(*lims["dist"])
    a_dist.set_ylabel("from the\nstack (mm)", fontsize=6.5, color=INK2)
    a_dist.set_title(title, color=INK, fontsize=8, loc="left")
    for a, name, limit in ((a_ncc, "ncc", ALIGN_NCC_MIN), (a_dice, "dice", None)):
        a.axvspan(lo_k - 0.5, hi_k + 0.5, color=INK2, alpha=0.12, lw=0)
        if f"align_{name}_after" in s:
            a.plot(k, s[f"align_{name}_before"], "o", ms=3.2, color=ACQ, label="acquired")
            a.plot(k, s[f"align_{name}_after"], "o", ms=3.2, color=ALI, label="aligned")
        if limit is not None:
            a.axhline(limit, color=INK, ls="--", lw=0.8)
        a.set_ylim(*lims[name])
        a.set_ylabel("NCC" if name == "ncc" else "Dice", fontsize=6.5, color=INK2)
    a_dice.set_xticks(k[:: max(1, len(k) // 7)])
    a_dice.set_xlabel("slice (shaded: kept)", fontsize=6.5, color=INK2)


def _view_panels(axs, view: str, key: str, lax, seg, P: dict, stack: np.ndarray, lo_k: int, hi_k: int, cmap,
                 svmax: float) -> None:
    """One long-axis view of an example: the view with the SAX slices' intersection lines (acquired dashed, aligned
    solid; kept slices bold), the SAX stack resliced onto its plane with the acquired and with the aligned poses,
    the view's own label outlines on both (for orientation only)."""
    for a in axs:
        a.set_facecolor(SURFACE)
        a.set_xticks([])
        a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(False)
    if lax is None or (key, 0) not in P["aligned"]:
        axs[0].set_title(f"{view}: missing", color=INK2, fontsize=8)
        return
    X, Y = lax.shape
    box = (0, X - 1, 0, Y - 1)
    ij = np.argwhere(np.isin(seg, (1, 2, 3))) if seg is not None else np.zeros((0, 2))
    if len(ij):
        mg = 14
        box = (max(ij[:, 0].min() - mg, 0), min(ij[:, 0].max() + mg, X - 1),
               max(ij[:, 1].min() - mg, 0), min(ij[:, 1].max() + mg, Y - 1))
    axs[0].imshow(lax.T, cmap="gray", origin="lower", vmin=0, vmax=float(np.percentile(lax, 99.5)) or 1.0)
    Z = stack.shape[2]
    for state, colour, ls, ax in (("acquired", ACQ, "--", axs[1]), ("aligned", ALI, "-", axs[2])):
        A_lax = P[state][(key, 0)]
        for k in range(Z):
            A = P[state].get(("SAX", k))
            pts = None if A is None else plane_line(A, A_lax, stack.shape[:2], box)
            if pts is None:
                continue
            inside = lo_k <= k <= hi_k
            axs[0].plot(pts[[0, -1], 0], pts[[0, -1], 1], color=colour, ls=ls, lw=1.1 if inside else 0.6,
                        alpha=1.0 if inside else 0.55)
            if state == "aligned":
                end = pts[np.argmax(pts[:, 0])]
                axs[0].text(end[0] + 0.8, end[1], str(k), color=INK2, fontsize=5.5, va="center")
        im = reslice(stack, [P[state].get(("SAX", k)) for k in range(Z)], A_lax, (X, Y))
        ax.imshow(np.ma.masked_invalid(im).T, cmap=cmap, origin="lower", vmin=0, vmax=svmax)
        if seg is not None:
            label_outlines(ax, seg, (1, 2, 3), lw=0.7, alpha=0.8)
        ax.set_title(f"SAX stack on the {view} plane: {state}", color=INK, fontsize=8)
    for a in axs:
        a.set_xlim(box[0] - 0.5, box[1] + 0.5)
        a.set_ylim(box[2] - 0.5, box[3] + 0.5)
    axs[0].set_title(f"{view}: where the SAX slices cut it", color=INK, fontsize=8)


def _load_images(subj_dir: Path):
    """(SAX stack (X, Y, Z, T) in the run's slice order, {image: (frame-0 image, frame-0 segmentation or None)} of
    the single-slice views) with moco_nifti's loader, as the run read them."""
    import moco_nifti as mn
    subject = mn.load_subject(subj_dir, SimpleNamespace(sax_thickness=8.0, lax_thickness=6.0))
    sax = sorted((s for s in subject.slices if s.image == "SAX"), key=lambda s: s.index)
    stack = np.stack([s.data for s in sax], 2) if sax and len({s.data.shape for s in sax}) == 1 else None
    views = {s.image: (s.data[:, :, 0], None if s.seg is None else s.seg[:, :, 0].astype(np.uint8))
             for s in subject.slices if s.image != "SAX" and s.index == 0}
    return stack, views


def fig_alignment_examples(df, series, run: str, root: Path, copy_dir: Path, out: Path, fixed=None,
                           titles=EXAMPLE_TITLES) -> list:
    """alignment_example_<n>.png, one figure per example subject (see the module docstring; `fixed`: the subjects to
    show instead, in the order of EXAMPLE_ROLES, with `titles`). The subject IDs are not shown. Returns the examples
    [(subject, why)]."""
    import matplotlib
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    rows = []
    for why, ranked in ([(r, [s]) for r, s in zip(EXAMPLE_ROLES, fixed)] if fixed else _candidates(df, root)):
        for sid in [s for s in ranked if s not in [r[0] for r in rows]][:5]:   # the next ones without a history
            h = history(run, sid, copy_dir)
            if h is not None:
                rows.append((sid, "typical" if why == EXAMPLE_WHY and sid != EXAMPLE_SUBJECT and not fixed else why,
                             poses(h)))
                break
    if not rows:
        print("alignment_examples: no kept subject with images and pose history")
        return []
    tables = [series[(series.subject == sid) & (series.kind == "sax") & series.slice.notna()] for sid, _, _ in rows]
    lims = _chart_limits(tables)
    cmap = matplotlib.colormaps["gray"].copy()
    cmap.set_bad(SURFACE)
    for old in out.glob("alignment_example_*.png"):
        old.unlink(missing_ok=True)                       # no figure of an earlier pick lingers
    H = 4.9
    for n, ((sid, why, P), s) in enumerate(zip(rows, tables), 1):
        stack, views = _load_images(Path(root) / sid)
        if stack is None:
            continue
        fig = Figure(figsize=(13.5, H), facecolor=SURFACE)
        gs = fig.add_gridspec(1, 4, width_ratios=[1.25, 1, 1, 1], wspace=0.06)
        x = df.loc[sid]
        lo_k, hi_k = sorted((int(x.base), int(x.apex)))
        sub = gs[0, 0].subgridspec(3, 1, hspace=0.1)
        _slice_chart([fig.add_subplot(sub[i]) for i in range(3)], s, lo_k, hi_k,
                     f"kept slices {lo_k}-{hi_k}: furthest {x.motion_mm:.1f} mm, "
                     f"lowest NCC {x.kept_align_slice_ncc_min:.2f}", lims)
        svmax = float(np.percentile(stack[..., 0], 99.5)) or 1.0
        view, key = next(((w, k) for w, k in VIEW_ORDER if k in views and (k, 0) in P["aligned"]), VIEW_ORDER[0])
        lax, seg = views.get(key, (None, None))
        _view_panels([fig.add_subplot(gs[0, 1 + i]) for i in range(3)], view, key, lax, seg, P, stack, lo_k, hi_k,
                     cmap, svmax)
        title = titles[n - 1] if fixed and n <= len(titles) else f"Slice alignment example {n} ({why})"
        fig.suptitle(title, color=INK, fontsize=10)
        fig.legend(handles=[Line2D([], [], color=ACQ, ls="--", marker="o", ms=4, label="acquired poses"),
                            Line2D([], [], color=ALI, ls="-", marker="o", ms=4, label="aligned poses")],
                   loc="upper center", bbox_to_anchor=(0.5, 1 - 0.3 / H), ncol=2, frameon=False, fontsize=8,
                   labelcolor=INK)
        fig.subplots_adjust(left=0.05, right=0.995, bottom=0.09, top=1 - 0.95 / H)
        capitalize_titles(fig)
        fig.savefig(out / f"alignment_example_{n}.png", dpi=90, facecolor=SURFACE)
        print(f"  {out / f'alignment_example_{n}.png'} ({sid}: {why})")
    return [(sid, why) for sid, why, _ in rows]


# --------------------------------------------------------------------------------------------- numbers for the text

def numbers(df, series) -> dict:
    """The values the README quotes, over the same subjects and slices as the figures."""
    st, model = df.align_status.astype(str), df.align_pose_model.astype(str)
    full_m = (st == "ok") & (model == FULL_MODEL)
    full, kept = df[full_m], df[full_m & df.kept]
    p = _positions(df, series[series.subject.isin(full.index)])
    med = lambda s: round(float(s.median()), 2) if len(s.dropna()) else None  # noqa: E731
    mk = kept.motion_mm.dropna()
    out = dict(
        subjects_in_run=len(df), six_dof=int(full_m.sum()), six_dof_with_ao=int((full_m & df.has_ao).sum()),
        inplane_only=int(((st == "ok") & (model != FULL_MODEL)).sum()),
        not_optimized=int(st.isin(["not_optimized", "no_images"]).sum()), errors=int(st.str.startswith("error").sum()),
        kept_subjects=len(kept),
        ncc_before_median=med(full.align_ncc_before), ncc_after_median=med(full.align_ncc_after),
        nmi_rises=int((full.align_nmi_gain > 0).sum()),
        kept_below_ncc_min=int(kept.align_problem.fillna("").astype(str).str.contains("NCC").sum()),
        trustworthy=int((kept.align_ok == True).sum()),                               # noqa: E712
        motion_limits_mm={f"{s}%": round(float(mk.quantile(1 - s / 100)), 1) for s in (5, 10, 20)} if len(mk) else {},
        beyond_motion_max=int((mk > MOTION_MAX_MM).sum()),
        mid_dice_before_median=med(kept.mid_dice_before), mid_dice_after_median=med(kept.mid_dice_after))
    for col, name, _, signed in TRANSLATION + ROTATION:
        if col in p:
            v = p[col].abs() if signed else p[col]
            out[f"{name} median by position"] = {pos: med(v[p.pos == pos]) for pos in POSITIONS}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="the run directory (or its overall_metrics/), local or user@host:/path")
    ap.add_argument("--root", type=Path, required=True,
                    help="the images the run aligned (one folder per subject): the SAX segmentations give the kept "
                         "slices (without --survey), the images the examples")
    ap.add_argument("--survey", type=Path, default=None,
                    help="a survey output directory of UKBB_implicit_baselines (survey.csv, slices.csv): the kept "
                         "slices, the verdict and the LV regions from there instead of the segmentations")
    ap.add_argument("--examples", default=",".join(EXAMPLE_SUBJECTS),
                    help="comma-separated subjects for the example figures, in the order: " + ", ".join(EXAMPLE_ROLES)
                         + "; 'auto': chosen by those roles. Default: the README's examples, %(default)s")
    ap.add_argument("--titles", default=None,
                    help="'|'-separated titles of the --examples figures. Default: " + " | ".join(EXAMPLE_TITLES))
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent, help="Default: %(default)s")
    ap.add_argument("--cache", type=Path, default=Path("alignment_cache"),
                    help="where the LV pixels per slice (and copies of a remote run) are kept. Default: %(default)s")
    ap.add_argument("--workers", type=int, default=8, help="processes reading the segmentations. Default: %(default)s")
    ap.add_argument("--no-examples", action="store_true")
    args = ap.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    src = fetch_source(args.run, args.cache / "source")
    subj, planes, _ = import_run(src)
    print(f"alignment: {len(subj)} subjects ({int((subj.status == 'ok').sum())} aligned), {len(planes)} slices / views")
    global KEPT_NAME, KEPT_SLICES, REGION_LABEL
    if args.survey is not None:
        kr, verdict, regions = survey_kept(args.survey)
        df, series = assemble(subj, planes, kr, verdict, regions)
        KEPT_NAME = "kept (the survey's verdict)"
        KEPT_SLICES = "the slices the survey's preprocessing keeps"
        REGION_LABEL = "region (from the long-axis view)"
        NAMED_COLORS[KEPT_NAME] = EXTRA[0]
    else:
        lv = lv_table(list(subj.subject), args.root, args.cache / "lv_pixels.csv", args.workers)
        df, series = assemble(subj, planes, kept_ranges(lv))
    print(f"alignment: kept range for {int(df.kept.sum())} subjects; figures:")
    fig_alignment(df, series, args.out)
    examples = [] if args.no_examples else fig_alignment_examples(
        df, series, args.run, args.root, args.cache / "history", args.out,
        [s.strip() for s in args.examples.split(",")] if args.examples != "auto" else None,
        [t.strip() for t in args.titles.split("|")] if args.titles else EXAMPLE_TITLES)
    nums = numbers(df, series)
    nums["examples"] = examples
    (args.cache / "alignment_numbers.json").write_text(json.dumps(nums, indent=1))
    print(json.dumps(nums, indent=1))


if __name__ == "__main__":
    main()
