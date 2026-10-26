"""Reload the aligned NIfTIs of one subject and recompute the intersection NCC at identity pose.

If the saved affines are right, the NCC recomputed from the saved files should match the `ncc`
column of the run (up to uint16 rounding and the mask lines being re-derived at the new geometry).
"""
import sys
import numpy as np
import nibabel as nib
import torch
from pathlib import Path
import pandas as pd

sys.path.insert(0, r"C:\Users\Nil\Documents\git\UKBB_implicit_baselines")
import motion_gradient as mg

run_dir = Path(sys.argv[1])
subject = sys.argv[2]
args = mg.argparse.Namespace(instance="Instance_2", n_frames=5, mask_dilation_mm=15.0, lax_thickness=None, sax_thickness=8.0,
                             min_line_mm=20.0, line_step_mm=1.0, line_extent_mm=250.0, n_band=5, metric="ncc", seg_labels="1,2,3",
                             no_save_images=True)
dev = torch.device("cuda")
inst = run_dir / subject / "Instance_2"

# original geometry for the dilated masks (same as the run); aligned geometry from the saved files
orig = mg.load_subject(Path(r"C:\Users\Nil\Documents\data\ukbb") / subject, args)
planes = []
for f in sorted(inst.glob("SAX_slice*.nii.gz")):
    img = nib.load(str(f))
    arr = np.asarray(img.dataobj, dtype=np.float32)
    assert img.get_data_dtype() == np.uint16, img.get_data_dtype()
    z = int(f.stem.split("slice")[1].split(".")[0])
    frames = mg.phase_frames(arr.shape[3], 5)
    p = mg.make_planes("SAX", arr[..., frames], img.affine, 8.0, None, None, dev)[0]
    p.slice_idx, p.key = z, f"SAX:{z}"
    planes.append(p)
for view in mg.LAX_VIEWS:
    f = inst / f"{view}.nii.gz"
    if not f.exists():
        continue
    img = nib.load(str(f))
    arr = np.asarray(img.dataobj, dtype=np.float32)
    frames = mg.phase_frames(arr.shape[3], 5)
    d = orig["lax"][view]
    planes.append(mg.make_planes(view, arr[..., frames], img.affine, d["thickness"], d["mask"], None, dev)[0])
sax = [p for p in planes if p.image == "SAX"]
lax = [p for p in planes if p.image != "SAX"]
index = {p.key: i for i, p in enumerate(planes)}
lines = []
from itertools import combinations
with torch.no_grad():
    for a, b in [(s, l) for s in sax for l in lax] + list(combinations(lax, 2)):
        line = mg.intersection_line(a, b, args, dev)
        if line is not None:
            lines.append((index[a.key], index[b.key], line))
    sampler = mg.LineSampler(planes, lines, 5, dev)
    zeros = torch.zeros(len(planes), 3, device=dev)
    ncc_saved = sampler.per_plane(sampler.ncc_lines(zeros, zeros, False))

det = pd.read_csv(run_dir / "motion_summary_slices.csv", dtype={"subject_id": str})
det = det[det.subject_id == subject].set_index(["image", "slice"])
print(f"{'plane':10s} {'ncc acquired':>13s} {'ncc run':>8s} {'ncc from saved files':>21s}")
for p, v in zip(planes, ncc_saved):
    r = det.loc[(p.image, p.slice_idx)]
    print(f"{p.key:10s} {r['ncc_zero']:13.3f} {r['ncc']:8.3f} {v:21.3f}")
