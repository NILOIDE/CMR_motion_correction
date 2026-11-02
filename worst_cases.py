"""
Rank the subjects of a moco_nifti.py / motion_gradient.py run by how badly they did, per metric, and write one subject
list per metric for slice_viewer_3d.py (--subjects <list>; [ / ] step through it, worst first).

Reads the run's cohort CSVs (overall_metrics/motion_summary.csv, motion_summary_slices.csv, motion_summary_pairs.csv,
sax_relative_motion.csv). They are append-only: a subject aligned again keeps its latest row (summary) or its latest
block of rows (the per-slice / per-pair / relative-motion files). Standard library only, so with --remote the script
runs on the machine holding the run (piped to `ssh <host> python3 -`; nothing but the ranked lists is transferred).

Each list <out-dir>/<metric>.txt: '#' header lines (metric, direction, filters), then one subject per line, worst
first: subject id, value, and where it occurs (slice / view / intersection). Run without --metrics for all of them.

Usage:
    python worst_cases.py --remote stol@131.159.110.9:/vol/miltank/projects/ukbb/data/cardiac/slice_alignment/aligned_subjects
    python worst_cases.py --run-dir C:\\Users\\Nil\\Documents\\data\\ukbb_alignment\\<run> --top 100
    python worst_cases.py --remote ... --metrics slice_shift_max slice_dice_min --min-points 30
"""
import argparse
import csv
import json
import math
import shlex
import subprocess
import sys
from pathlib import Path

# metric: (source file, description, worst = "high" | "low" | "list"); the values themselves are computed in rank()
METRICS = {
    # --- motion (per slice / view; motion_summary_slices.csv: pose relative to the mean LAX pose)
    "slice_shift_max":       ("slices", "largest 3D shift of a single slice or view (mm)", "high"),
    "slice_inplane_max":     ("slices", "largest in-plane shift of a single slice or view (mm)", "high"),
    "slice_dz_max":          ("slices", "largest through-plane shift |dz| of a single slice or view (mm)", "high"),
    "slice_tilt_max":        ("slices", "largest tilt (out-of-plane rotation) of a single slice or view (deg)", "high"),
    "slice_rot_inplane_max": ("slices", "largest |in-plane rotation| of a single slice or view (deg)", "high"),
    # --- SAX motion relative to the rest of the stack (sax_relative_motion.csv)
    "sax_rel_shift_max":     ("relmotion", "largest shift of one SAX slice relative to the rest of the stack (mm)", "high"),
    "sax_rel_rot_max":       ("relmotion", "largest rotation of one SAX slice relative to the rest of the stack (deg)", "high"),
    "sax_adjacent_shift_max": ("relmotion", "largest relative shift between two adjacent SAX slices (mm)", "high"),
    # --- subject-level motion (motion_summary.csv)
    "sax_jitter":            ("summary", "SAX slice-to-slice shift jitter (mm)", "high"),
    "sax_shift_max":         ("summary", "largest SAX in-plane shift (mm)", "high"),
    "lax_shift_max":         ("summary", "largest LAX view shift (mm)", "high"),
    "sax_tilt_rms":          ("summary", "rms SAX tilt (deg)", "high"),
    "sax_gap_min":           ("summary", "smallest gap between neighbouring SAX slices near the LV axis (mm; < 0: "
                                         "they cross)", "low"),
    # --- agreement after alignment (lower = worse)
    "slice_dice_min":        ("slices", "worst single-slice / view Dice after alignment", "low"),
    "slice_dice_drop":       ("slices", "largest Dice decrease of a single slice / view (after - before)", "low"),
    "slice_ncc_min":         ("slices", "worst single-slice / view NCC after alignment", "low"),
    "slice_ncc_drop":        ("slices", "largest NCC decrease of a single slice / view (after - before)", "low"),
    "slice_hd_max":          ("slices", "worst single-slice / view Hausdorff distance after alignment (mm)", "high"),
    "pair_dice_min":         ("pairs", "worst Dice of a single intersection line after alignment", "low"),
    "pair_ncc_min":          ("pairs", "worst NCC of a single intersection line after alignment", "low"),
    "dice_final_low":        ("summary", "lowest subject mean Dice after alignment", "low"),
    "dice_drop":             ("summary", "largest subject mean Dice decrease (after - before)", "low"),
    "ncc_final_low":         ("summary", "lowest subject mean NCC after alignment", "low"),
    "ncc_gain_negative":     ("summary", "subject mean NCC made worst by the alignment (gain)", "low"),
    "hd_max":                ("summary", "largest subject Hausdorff distance after alignment (mm)", "high"),
    # --- not (fully) aligned
    "not_optimized":         ("summary", "not optimized (no crossing views): value = number of LAX views", "list"),
    "inplane_only":          ("summary", "in-plane pose model only (a single LAX view, or no SAX stack): value = LAX views",
                              "list"),
}
FILES = {"summary": "motion_summary.csv", "slices": "motion_summary_slices.csv",
         "pairs": "motion_summary_pairs.csv", "relmotion": "sax_relative_motion.csv"}


def num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return math.nan
    return v


def latest_blocks(path: Path):
    """Rows of an append-only per-subject CSV, keeping each subject's latest contiguous block of rows (a subject
    aligned again appends a new block). Yields (subject_id, [rows]) once the file is read."""
    blocks, last = {}, None
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            sid = row["subject_id"]
            if sid != last:
                blocks[sid] = []                              # a new block of this subject replaces the old one
                last = sid
            blocks[sid].append(row)
    return blocks


def where(row, pair=False):
    if pair:
        return f"{row['image_a']} {row['slice_a']} x {row['image_b']} {row['slice_b']}"
    return f"{row['image']} {row['slice']}"


def rank(run_dir: Path, metrics: list[str], top: int, min_points: int) -> dict:
    """{metric: {"desc", "worst", "n", "rows": [[subject, value, where], ...] worst first, at most `top`}}."""
    om = run_dir / "overall_metrics"
    need = {METRICS[m][0] for m in metrics}
    per = {m: [] for m in metrics}                            # (value, subject, where)

    if "summary" in need:
        latest = {}
        with open(om / FILES["summary"], newline="") as f:
            for row in csv.DictReader(f):
                latest[row["subject_id"]] = row
        for sid, r in latest.items():
            st, model = r["status"], r["pose_model"]
            vals = {
                "sax_jitter": num(r["sax_shift_jitter_mm"]), "sax_shift_max": num(r["sax_shift_max_mm"]),
                "lax_shift_max": num(r["lax_shift_max_mm"]), "sax_tilt_rms": num(r["sax_tilt_rms_deg"]),
                "sax_gap_min": num(r["sax_gap_min_mm"]),
                "dice_final_low": num(r["seg_dice_mean"]),
                "dice_drop": num(r["seg_dice_mean"]) - num(r["seg_dice_zero_mean"]),
                "ncc_final_low": num(r["ncc_mean"]), "ncc_gain_negative": num(r["ncc_gain_mean"]),
                "hd_max": num(r["hd_max_mm"]),
            }
            for m, v in vals.items():
                if m in per and st == "ok":
                    per[m].append((v, sid, f"{model}, {r['sax_n_slices']} SAX slices, {r['n_lax_views']} LAX views"))
            if "not_optimized" in per and st == "not_optimized":
                per["not_optimized"].append((num(r["n_lax_views"]), sid, f"{r['sax_n_slices']} SAX slices"))
            if "inplane_only" in per and st == "ok" and model == "inplane":
                per["inplane_only"].append((num(r["n_lax_views"]), sid, f"{r['sax_n_slices']} SAX slices"))

    if "slices" in need:
        for sid, rows in latest_blocks(om / FILES["slices"]).items():
            best = {}
            for r in rows:
                if num(r["n_pairs"]) <= 0:
                    continue                                    # not aligned (no lines)
                enough = num(r["n_points"]) >= min_points
                cand = {
                    "slice_shift_max": num(r["shift_mm"]), "slice_inplane_max": num(r["shift_inplane_mm"]),
                    "slice_dz_max": abs(num(r["dz_mm"])), "slice_tilt_max": num(r["rot_tilt_deg"]),
                    "slice_rot_inplane_max": abs(num(r["rot_inplane_deg"])),
                    "slice_hd_max": num(r["hd_mm"]),
                }
                if enough:
                    cand.update(slice_dice_min=num(r["seg_dice"]),
                                slice_dice_drop=num(r["seg_dice"]) - num(r["seg_dice_zero"]),
                                slice_ncc_min=num(r["ncc"]), slice_ncc_drop=num(r["ncc"]) - num(r["ncc_zero"]))
                npts = num(r["n_points"])
                for m, v in cand.items():
                    if m not in per or math.isnan(v):
                        continue
                    worse = (m not in best or ((v, npts) > best[m][:2] if METRICS[m][2] == "high"
                                               else (v, -npts) < (best[m][0], -best[m][1])))
                    if worse:
                        best[m] = (v, npts, where(r))
            for m, (v, npts, w) in best.items():
                per[m].append((v, sid, f"{w} ({npts:.0f} pts)", npts))

    if "pairs" in need:
        for sid, rows in latest_blocks(om / FILES["pairs"]).items():
            best = {}
            for r in rows:
                if r["kind"] not in ("SAX-LAX", "LAX-LAX") or num(r["n_points"]) < min_points:
                    continue
                npts = num(r["n_points"])
                for m, v in (("pair_dice_min", num(r["seg_dice"])), ("pair_ncc_min", num(r["ncc"]))):
                    if m in per and not math.isnan(v) and (m not in best or (v, -npts) < (best[m][0], -best[m][1])):
                        best[m] = (v, npts, where(r, pair=True))
            for m, (v, npts, w) in best.items():
                per[m].append((v, sid, f"{w} ({npts:.0f} pts)", npts))

    if "relmotion" in need:
        for sid, rows in latest_blocks(om / FILES["relmotion"]).items():
            best = {}
            for r in rows:
                if r["kind"] == "slice":
                    cand = (("sax_rel_shift_max", num(r["shift_mm"]), f"SAX {r['slice_a']}"),
                            ("sax_rel_rot_max", num(r["rot_deg"]), f"SAX {r['slice_a']}"))
                else:
                    cand = (("sax_adjacent_shift_max", num(r["shift_mm"]), f"SAX {r['slice_a']}-{r['slice_b']}"),)
                for m, v, w in cand:
                    if m in per and not math.isnan(v) and (m not in best or v > best[m][0]):
                        best[m] = (v, w)
            for m, (v, w) in best.items():
                per[m].append((v, sid, w))

    out = {}
    for m in metrics:
        # ties (e.g. many slices with Dice 0) are broken by the number of line points: more points, more certain
        rows = [x if len(x) == 4 else (*x, 0.0) for x in per[m] if not math.isnan(x[0])]
        worst = METRICS[m][2]
        if worst == "high":
            rows.sort(key=lambda x: (-x[0], -x[3]))
        elif worst == "low":
            rows.sort(key=lambda x: (x[0], -x[3]))
        else:
            rows.sort(key=lambda x: int(x[1]))
        out[m] = dict(desc=METRICS[m][1], worst=worst, n=len(rows),
                      rows=[[sid, round(v, 4), w] for v, sid, w, _ in rows[:top]])
    return out


def write_lists(result: dict, out_dir: Path, source: str, min_points: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for m, r in result.items():
        order = {"high": "highest first", "low": "lowest first", "list": "by subject id"}[r["worst"]]
        lines = [f"# {m}: {r['desc']}",
                 f"# {order}; top {len(r['rows'])} of {r['n']} subjects; run {source}",
                 f"# per-slice / per-line agreement metrics only count those with >= {min_points} line points",
                 "# subject  value  where"]
        lines += [f"{sid}  {v:g}  {w}" for sid, v, w in r["rows"]]
        (out_dir / f"{m}.txt").write_text("\n".join(lines) + "\n")
    print(f"{len(result)} list(s) written to {out_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--run-dir", type=Path, help="local run directory (with overall_metrics/)")
    src.add_argument("--remote", metavar="USER@HOST:/RUN_DIR",
                     help="run directory on another machine: this script runs there (ssh, python3) and only the lists "
                          "come back")
    ap.add_argument("--metrics", nargs="+", choices=list(METRICS), default=list(METRICS),
                    help="metrics to rank (default: all)")
    ap.add_argument("--top", type=int, default=50, help="subjects per list (default: %(default)s)")
    ap.add_argument("--min-points", type=int, default=20,
                    help="per-slice / per-line Dice and NCC only where the line points number at least this many "
                         "(few points give noisy values). Default: %(default)s")
    ap.add_argument("--out-dir", type=Path, default=Path("worst_cases"), help="output directory (default: %(default)s)")
    ap.add_argument("--json", action="store_true", help=argparse.SUPPRESS)   # remote side: print the result as JSON
    args = ap.parse_args()

    if args.remote:
        target, path = args.remote.split(":", 1)
        cmd = ["python3", "-", "--run-dir", path, "--top", str(args.top), "--min-points", str(args.min_points),
               "--json", "--metrics", *args.metrics]
        print(f"Ranking on {target} (reads the run's cohort CSVs there; ~1 min for 50k subjects) ...", flush=True)
        p = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", target,
                            " ".join(shlex.quote(c) for c in cmd)],
                           input=Path(__file__).read_text(), capture_output=True, text=True)
        if p.returncode:
            sys.exit(f"remote ranking failed:\n{p.stderr}")
        result = json.loads(p.stdout)
        write_lists(result, args.out_dir, args.remote, args.min_points)
    else:
        result = rank(args.run_dir, args.metrics, args.top, args.min_points)
        if args.json:
            print(json.dumps(result))
            return
        write_lists(result, args.out_dir, str(args.run_dir), args.min_points)
    for m, r in result.items():
        head = ", ".join(f"{sid} ({v:g})" for sid, v, _ in r["rows"][:3])
        print(f"  {m:24s} {r['n']:6d} subjects; worst: {head}")


if __name__ == "__main__":
    main()
