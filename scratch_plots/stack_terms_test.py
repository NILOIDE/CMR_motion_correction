"""Checks of motion_gradient.stack_terms on a real SAX stack: zero for rigid motion of the whole stack, and which terms
respond to which single-slice perturbation."""
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, r"C:\Users\Nil\Documents\git\UKBB_implicit_baselines")
import motion_gradient as mg  # noqa: E402

ROOT = Path(r"C:\Users\Nil\Documents\git\UKBB_implicit_baselines\scratch_plots\gs_root5")
torch.set_default_dtype(torch.float64)
args = SimpleNamespace(instance="Instance_2", n_frames=1, no_save_images=True, mask_dilation_mm=20.0, lax_thickness=6.0)
data = mg.load_subject(ROOT / "1000071", args)
planes = mg.make_planes("SAX", *data["sax"], 8.0, None, None, "cpu")
N = len(planes)
C0 = torch.tensor(np.stack([p.center for p in planes]))
N0 = torch.tensor(np.stack([p.normal for p in planes]))
nb_a, nb_b = torch.arange(N - 1), torch.arange(1, N)
step0 = ((C0[nb_b] - C0[nb_a]) * N0[nb_a]).sum(1)
target = torch.sign(step0) * step0.abs()                       # acquired spacing
ALL = {"normal", "rel_rot", "spacing", "rel_shift"}
u, v = torch.tensor(planes[0].u_hat), torch.tensor(planes[0].v_hat)


def show(name, theta, t):
    terms = mg.stack_terms(theta, t, C0, N0, nb_a, nb_b, target, ALL)
    print(f"{name:52s} " + "  ".join(f"{k} {float(x):9.4f}" for k, x in terms.items()))


zero = torch.zeros(N, 3)
show("identity", zero, zero)

# whole stack: rotation R (8 deg about an oblique axis) about a point O, then a translation
axis = torch.tensor([0.3, -0.5, 0.8])
axis = axis / axis.norm()
rv = axis * math.radians(8.0)
R = mg.rodrigues(rv[None])[0]
O = C0.mean(0) + torch.tensor([20.0, -10.0, 5.0])
tau = torch.tensor([4.0, -7.0, 3.0])
theta = torch.rad2deg(rv)[None].repeat(N, 1)
t = (C0 - O) @ R.T + O + tau - C0
show("whole stack rigid (8 deg + shift)", theta, t)

k = 5
t1 = zero.clone(); t1[k] = 3.0 * u
show(f"slice {k} shifted 3 mm in-plane", zero, t1)
t2 = zero.clone(); t2[k] = 2.0 * N0[k]
show(f"slice {k} shifted 2 mm through-plane", zero, t2)
th3 = zero.clone(); th3[k] = 5.0 * N0[k]
show(f"slice {k} rotated 5 deg in-plane", th3, zero)
th4 = zero.clone(); th4[k] = 3.0 * u
show(f"slice {k} tilted 3 deg", th4, zero)
# expected: in-plane shift s -> rel_shift = 2 pairs * s^2 / (N - 1); in-plane rotation r -> rel_rot = 2 r^2 / (N - 1)
print(f"expected rel_shift for 3 mm: {2 * 9 / (N - 1):.4f};  rel_rot for 5 deg: {2 * math.radians(5) ** 2 * (180 / math.pi) ** 2 * (math.sin(math.radians(5)) / math.radians(5)) ** 2 / (N - 1):.4f}")

# gradient check (float64)
th = (torch.randn(N, 3) * 2).requires_grad_(True)
tt = (torch.randn(N, 3) * 3).requires_grad_(True)
ok = torch.autograd.gradcheck(lambda a, b: sum(mg.stack_terms(a, b, C0, N0, nb_a, nb_b, target, ALL).values()), (th, tt))
print("gradcheck:", ok)
