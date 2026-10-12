#!/usr/bin/env python3
"""t512: realistic bound for starting pfwc N+1 per core as each core finishes mat+blend N.

Uses the t506 capture (prep506.py output). For each steady period v -> v+1 (same chunk, v not
first of chunk) it compares the measured pfwc N+1 end with modelled ends:
  percore : pfwc_c(N+1) starts at blend_end_c(N) + gap  (fused tail / per-core go)
  perfect : pfwc N+1 work spread evenly over all cores' blend tail idle (lower bound on end)
Then k2 N+1 starts at max over cores, so the saving is old pfwc end - new pfwc end.
Usage: bound512.py CSV [--mhz 1350] [--gap-us 2]
"""
import argparse, os, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "b2b-timeline-p150-223"))
import ana506 as A  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("csv"); ap.add_argument("--mhz", type=float, default=1350.0)
ap.add_argument("--gap-us", type=float, default=2.0)
a = ap.parse_args()
A.CYC_MS = a.mhz * 1000.0
iv, td = A.load_dev(a.csv)
FWI, Z, D, NV, cores = A.assign(iv, td)
VPC = A.VIEWS_PER_CHUNK
gap = a.gap_us * a.mhz
def CL(v, p):
    d = defaultdict(lambda: [1 << 62, 0])
    for c, r, s, e in FWI[(v, p)]:
        d[c][0] = min(d[c][0], s); d[c][1] = max(d[c][1], e)
    return d
rows = []
for v in range(NV - 1):
    if v // VPC != (v + 1) // VPC or v % VPC == 0:
        continue
    mb, pf = CL(v, 3), CL(v + 1, 0)
    cs = [c for c in cores if c in mb and c in pf]
    old_end = max(pf[c][1] for c in cs)
    bend = {c: mb[c][1] for c in cs}
    plen = {c: pf[c][1] - pf[c][0] for c in cs}
    pc_end = max(bend[c] + gap + plen[c] for c in cs)
    # same per-core lengths, but slowest pfwc cores assigned to earliest-finishing blend cores
    sb = sorted(bend.values()); sl = sorted(plen.values(), reverse=True)
    best_perm = max(b + gap + l for b, l in zip(sb, sl))
    # perfectly divisible pfwc work spread into tails
    T0 = max(bend.values()); work = sum(plen.values()); tail = sum(T0 - b for b in bend.values())
    perf_end = T0 + gap + max(0.0, (work - tail)) / len(cs)
    corr = np.corrcoef([bend[c] for c in cs], [plen[c] for c in cs])[0, 1]
    rows.append((v, A.ms(old_end - pc_end), A.ms(old_end - best_perm), A.ms(old_end - perf_end), corr,
                 A.ms(max(plen.values()) - min(plen.values())), A.ms(T0 - min(bend.values()))))
r = np.array([x[1:] for x in rows])
print(f"# bound512 {a.csv} periods={len(rows)} gap={a.gap_us}us mhz={a.mhz}")
print("view  save_percore  save_bestperm  save_perfect  corr(blend_end,pfwc_len)  pfwc_spread  blend_spread")
for x in rows:
    print(f"{x[0]:4d}  {x[1]:12.3f}  {x[2]:13.3f}  {x[3]:12.3f}  {x[4]:23.2f}  {x[5]:11.3f}  {x[6]:12.3f}")
m = r.mean(0)
print(f"mean  {m[0]:12.3f}  {m[1]:13.3f}  {m[2]:12.3f}  {m[3]:23.2f}  {m[4]:11.3f}  {m[5]:12.3f}")

# pfwc per-core balance: is the per-core length a stable property of the core?
vs = [v for v in range(NV) if v % VPC != 0]
L = np.array([[A.ms((lambda d: d[c][1] - d[c][0])(CL(v, 0))) for c in cores] for v in vs])
win = L.max(1)
print(f"\n## pfwc per-core length, {len(vs)} views x {len(cores)} cores (ms)")
print(f"mean core {L.mean():.3f}  mean max {win.mean():.3f}  max-mean (end spread) {(win - L.mean(1)).mean():.3f}")
rel = L / L.mean(1, keepdims=True)
fc = rel.mean(0)
resid = rel - fc
print(f"relative per-core speed: core-mean factor range {fc.min():.3f}..{fc.max():.3f}, "
      f"var explained by core {1 - resid.var() / (rel - 1).var():.2f}")
loo = []
for i in range(len(vs)):
    f = np.delete(rel, i, 0).mean(0)          # leave-one-out static weights
    loo.append(win[i] - (L[i] / f * (L[i].sum() / (L[i] / f).sum())).max())
print(f"static per-core reweight (leave-one-out weights, same total work): save {np.mean(loo):.3f} ms/view "
      f"(min {np.min(loo):.3f}, max {np.max(loo):.3f})")
slow = np.argsort(-fc)[:8]
print("slowest cores (factor):", ", ".join(f"{cores[i]}:{fc[i]:.3f}" for i in slow))
