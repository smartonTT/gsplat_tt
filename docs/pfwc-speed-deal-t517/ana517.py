#!/usr/bin/env python3
"""t517: per-core pfwc speed factors from the t506 capture, mapped to logical core ids.

Usage: ana517.py PREP_CSV WEIGHTS -o factors.txt
Prints whether the per-core factor follows the core's strided tile set (t439 hero weights)
or the core's place on the chip, and writes one factor per logical core (c = y*12 + x).
"""
import argparse, os, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "b2b-timeline-p150-223"))
import ana506 as A  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("csv"); ap.add_argument("weights"); ap.add_argument("-o", required=True)
a = ap.parse_args()
A.CYC_MS = 1350.0 * 1000.0
iv, td = A.load_dev(a.csv)
FWI, Z, D, NV, cores = A.assign(iv, td)
VPC = A.VIEWS_PER_CHUNK
def CL(v, p):
    d = defaultdict(lambda: [1 << 62, 0])
    for c, r, s, e in FWI[(v, p)]:
        d[c][0] = min(d[c][0], s); d[c][1] = max(d[c][1], e)
    return d
vs = [v for v in range(NV) if v % VPC != 0]
L = np.array([[A.ms((lambda d: d[c][1] - d[c][0])(CL(v, 0))) for c in cores] for v in vs])
rel = L / L.mean(1, keepdims=True)
fc = rel.mean(0)
xs = sorted({c[0] for c in cores}); ys = sorted({c[1] for c in cores})
print("phys x:", xs, "phys y:", ys)
GX = len(xs)
logical = [ys.index(c[1]) * GX + xs.index(c[0]) for c in cores]
f = np.ones(len(cores))
for i, lc in enumerate(logical):
    f[lc] = fc[i]
w = np.array([float(t) for t in open(a.weights).read().split() if not t.startswith("#")] if False else
             [float(l) for l in open(a.weights) if l.strip() and not l.startswith("#")])
T, C = len(w), len(cores)
load = np.array([w[c::C].sum() for c in range(C)])
load /= load.mean()
print(f"T={T} C={C}; factor range {f.min():.3f}..{f.max():.3f}")
print(f"corr(core factor, strided tile-set weight) = {np.corrcoef(f, load)[0, 1]:.2f}; "
      f"tile-set weight range {load.min():.3f}..{load.max():.3f}")
# residual after the tile-set model: the core's own speed
k = np.polyfit(load, f, 1); res = f / np.polyval(k, load)
print(f"fit factor = {k[0]:.3f}*load + {k[1]:.3f}; residual 'core speed' range {res.min():.3f}..{res.max():.3f}, "
      f"std {res.std():.3f} (factor std {f.std():.3f})")
G = f.reshape(len(ys), GX)
print("factor by logical (row y, col x):")
for y in range(len(ys)):
    print(f" y{y:2d} " + " ".join(f"{G[y, x]:.3f}" for x in range(GX)))
print("row means:", " ".join(f"{v:.3f}" for v in G.mean(1)))
print("col means:", " ".join(f"{v:.3f}" for v in G.mean(0)))
with open(a.o, "w") as fo:
    fo.write("# t517 pfwc per-core length factor (t506 capture, best-iter-223, 20 views), logical c = y*%d + x\n" % GX)
    for v in f:
        fo.write(f"{v:.4f}\n")
