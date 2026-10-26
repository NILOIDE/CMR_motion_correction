"""Decompose refined rotations into in-plane vs tilt, and compare 6-DOF against translation-only."""
import sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, r"C:\Users\Nil\Documents\git\UKBB_implicit_baselines")
import motion_misalignment as mm

kw = dict(a.split("=") for a in sys.argv[1:])
base = dict(instance="Instance_2", n_frames=5, mask_dilation_mm=15.0, lax_thickness=None, sax_thickness=8.0,
            search_mm=15.0, coarse_step_mm=1.0, fine_step_mm=0.25, rounds=2, min_line_mm=20.0, line_step_mm=1.0,
            line_extent_mm=250.0, n_band=5, metric="ncc", refine_steps=200, refine_lr=0.05, refine_rot_lr=0.05,
            refine_dof=6, refine_gauge=1e-2, refine_prior=0.0)
dev = torch.device("cuda")

captured = {}
_orig = mm.refine_rigid


def cap(planes, lines, a, device):
    out = _orig(planes, lines, a, device)
    captured.update({p.key: (p, th) for p, th in zip(planes, out["theta_deg"])})
    return out


mm.refine_rigid = cap

for subject in kw.get("subjects", "1000456,1000109,1000071").split(","):
    print(f"===== {subject}")
    for dof in (3, 6):
        args = mm.argparse.Namespace(**{**base, "refine_dof": dof})
        data = mm.load_subject(Path(rf"D:\ukbb\{subject}"), args)
        captured.clear()
        summary, details = mm.process_subject(data, args, dev)
        print(f"dof={dof}: ncc zero={summary['ncc_zero_mean']:.3f} grid={summary['ncc_best_mean']:.3f} "
              f"ref={summary['ncc_ref_mean']:.3f} | sax rms grid={summary['sax_shift_rms_mm']:.2f} ref={summary['sax_shift_rms_ref_mm']:.2f} "
              f"jitter grid={summary['sax_shift_jitter_mm']:.2f} ref={summary['sax_shift_jitter_ref_mm']:.2f} dz={summary['sax_dz_rms_ref_mm']:.2f}")
        if dof == 6:
            for d in details:
                if d["n_pairs"] == 0:
                    continue
                p, th = captured[f"{d['image']}:{d['slice']}"]
                inplane = float(th @ p.normal)
                tilt = float(np.linalg.norm(th - inplane * p.normal))
                print(f"   {d['image']:8s} {d['slice']:2d}  in-plane {inplane:6.2f} deg  tilt {tilt:5.2f} deg  "
                      f"| dxy grid=({d['dx_mm']:6.2f},{d['dy_mm']:6.2f}) ref=({d['dx_ref_mm']:6.2f},{d['dy_ref_mm']:6.2f}) dz={d['dz_ref_mm']:6.2f} "
                      f"| ncc grid={d['ncc_best']:.3f} ref={d['ncc_ref']:.3f}")
