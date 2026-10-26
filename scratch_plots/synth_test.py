import sys
import numpy as np
import torch
from pathlib import Path
from scipy.ndimage import rotate

sys.path.insert(0, r"C:\Users\Nil\Documents\git\UKBB_implicit_baselines")
import motion_misalignment as mm

kw = dict(a.split("=") for a in sys.argv[1:])
args = mm.argparse.Namespace(
    instance="Instance_2", n_frames=5, mask_dilation_mm=15.0, lax_thickness=None, sax_thickness=8.0,
    search_mm=15.0, coarse_step_mm=1.0, fine_step_mm=0.25, rounds=2, min_line_mm=20.0, line_step_mm=1.0,
    line_extent_mm=250.0, n_band=5, metric="ncc",
    refine_steps=int(kw.get("steps", 200)), refine_lr=float(kw.get("lr", 0.05)),
    refine_rot_lr=float(kw.get("rot_lr", 0.01)), refine_dof=int(kw.get("dof", 6)),
    refine_gauge=float(kw.get("gauge", 1e-2)), refine_prior=float(kw.get("prior", 0.0)),
)
dev = torch.device("cuda")
data = mm.load_subject(Path(r"D:\ukbb\1000109"), args)
arr, aff = data["sax"]
pix = np.linalg.norm(aff[:3, 0]), np.linalg.norm(aff[:3, 1])
arr[:, :, 4] = np.roll(np.roll(arr[:, :, 4], 3, axis=0), -2, axis=1)
for f in range(arr.shape[3]):
    arr[:, :, 6, f] = rotate(arr[:, :, 6, f], 5.0, reshape=False, order=1)
DZ = 4.0
orig_make = mm.make_planes


def patched(image, a, affine, thickness, mask_arr, device):
    planes = orig_make(image, a, affine, thickness, mask_arr, device)
    if image == "SAX":
        p = planes[5]
        p.origin = p.origin - DZ * p.normal
        p.center = p.center - DZ * p.normal
        p.inv_trans = p.inv_trans + DZ * (p.inv_rot @ torch.tensor(p.normal, dtype=torch.float32, device=device))
    return planes


mm.make_planes = patched
import time
t0 = time.time()
summary, details = mm.process_subject(data, args, dev)
print(f"config {kw}  ({time.time() - t0:.1f} s)")
print("expected slice4: dx=%.2f dy=%.2f(+0.75 native) | slice5: dz=+%.1f | slice6: rot=5 deg; others ~0" % (-3 * pix[0], 2 * pix[1], DZ))
for d in details:
    if d["image"] == "SAX" and 2 <= d["slice"] <= 8:
        print("slice %d  grid: dx=%6.2f dy=%6.2f | ref: dx=%6.2f dy=%6.2f dz=%6.2f rot=%5.2f | ncc best=%.3f ref=%.3f"
              % (d["slice"], d["dx_mm"], d["dy_mm"], d["dx_ref_mm"], d["dy_ref_mm"], d["dz_ref_mm"], d["rot_ref_deg"], d["ncc_best"], d["ncc_ref"]))
for d in details:
    if d["image"] != "SAX":
        print("%s ref: dx=%.2f dy=%.2f dz=%.2f rot=%.2f" % (d["image"], d["dx_ref_mm"], d["dy_ref_mm"], d["dz_ref_mm"], d["rot_ref_deg"]))
print("summary: ncc best=%.3f ref=%.3f  rot_rms=%.2f dz_rms=%.2f" % (summary["ncc_best_mean"], summary["ncc_ref_mean"], summary["sax_rot_rms_ref_deg"], summary["sax_dz_rms_ref_mm"]))
