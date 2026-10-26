"""Profile one optimization step of motion_gradient on a single subject."""
import sys
import time
from itertools import combinations
from pathlib import Path

import torch

sys.path.insert(0, r"C:\Users\Nil\Documents\git\UKBB_implicit_baselines")
import motion_gradient as mg

subject = sys.argv[1] if len(sys.argv) > 1 else "1000071"
dev = torch.device("cuda")
args = mg.argparse.Namespace(instance="Instance_2", n_frames=5, mask_dilation_mm=25.0, lax_thickness=6.0, sax_thickness=8.0,
                             min_line_mm=80.0, line_step_mm=1.0, line_extent_mm=100.0, no_save_images=True, metric="ncc",
                             mi_bins=24, mi_sigma=1.0, n_band=8, psf_model="pulse", psf_tbw=2.5)
data = mg.load_subject(Path(r"C:\Users\Nil\Documents\data\ukbb") / subject, args)
planes = mg.make_planes("SAX", data["sax"][0], data["sax"][1], 8.0, None, data["sax_seg"], dev)
for v, d in data["lax"].items():
    planes += mg.make_planes(v, d["arr"], d["affine"], d["thickness"], d["mask"], d["seg"], dev)
idx = {p.key: i for i, p in enumerate(planes)}
sax = [p for p in planes if p.image == "SAX"]
lax = [p for p in planes if p.image != "SAX"]
lines = [(idx[a.key], idx[b.key], l) for a, b in [(s, x) for s in sax for x in lax] + list(combinations(lax, 2))
         if (l := mg.intersection_line(a, b, args, dev)) is not None]
S = mg.LineSampler(planes, lines, args.n_band, dev, args.psf_model, args.psf_tbw)
print(f"planes {S.N}, lines {S.M}, Lmax {S.Lmax}, frames {S.nf}, K {S.K}, groups "
      + ", ".join(f"[G={len(g['pidx'])} nmax={g['nmax']} {g['H']}x{g['W']}]" for g in S.groups))
used = float(S.W.sum())
padded = sum(len(g["pidx"]) * g["nmax"] * S.Lmax for g in S.groups)
print(f"line points used {used:.0f}, sampled (padded) {padded}, efficiency {used / padded * 100:.0f}%")

th = torch.zeros(S.N, 3, device=dev, requires_grad=True)
t = torch.zeros(S.N, 3, device=dev, requires_grad=True)
opt = torch.optim.Adam([t, th], lr=0.05)


def step(blur=None):
    if blur is not None:
        S.set_blur(blur)
    opt.zero_grad()
    c = S.ncc_lines(th, t, False)
    obj = (c * S.npts).sum() / S.npts.sum()
    (-obj).backward()
    opt.step()
    return obj


def timeit(fn, n=100):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


print(f"full step (no blur):           {timeit(step):.2f} ms")
print(f"full step (blur recomputed):   {timeit(lambda: step(6.0)):.2f} ms")
S.set_blur(0.0)
print(f"set_blur alone:                {timeit(lambda: S.set_blur(6.0)):.2f} ms")
S.set_blur(0.0)
print(f"forward profiles only:         {timeit(lambda: S.profiles(th, t, 'img')):.2f} ms")
with torch.no_grad():
    print(f"forward profiles (no grad):    {timeit(lambda: S.profiles(th, t, 'img')):.2f} ms")
print(f"float(obj) sync per step adds: {timeit(lambda: float(step())) - timeit(step):.2f} ms")

from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(20):
        step()
    torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=12))
ka = prof.key_averages()
print("kernel launches per step:", sum(e.count for e in ka if e.device_type.name == "CUDA") / 20)
