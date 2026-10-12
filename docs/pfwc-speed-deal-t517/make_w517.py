#!/usr/bin/env python3
"""t517: per-tile weights for GSPLAT_TT_PFWC_DEAL=range.

Tile t (legacy core c = t mod C) gets f_c * w_t / mean(w over core c's tiles): the measured
per-core pfwc length factor (core-factor-t506.txt, 20 views) sets each legacy core's total,
the t439 hero CPU model (visible count per tile) spreads it over the core's tiles.
Usage: make_w517.py FACTORS HERO_W -o OUT
"""
import argparse
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("factors"); ap.add_argument("hero_w"); ap.add_argument("-o", required=True)
a = ap.parse_args()
rd = lambda p: np.array([float(l) for l in open(p) if l.strip() and not l.startswith("#")])
f, w = rd(a.factors), rd(a.hero_w)
C, T = len(f), len(w)
out = np.empty(T)
for c in range(C):
    out[c::C] = f[c] * w[c::C] / w[c::C].mean()
with open(a.o, "w") as fo:
    fo.write(f"# t517 range-deal tile weights: core factor x hero model share, {T} tiles, {C} cores\n")
    fo.writelines(f"{x:.5f}\n" for x in out)
# predicted max/mean of per-core load before (strided) and after (equal-weight contiguous runs)
base, rem = divmod(T, C)
seq = np.concatenate([out[c::C] for c in range(C)])
pre = np.concatenate([[0], np.cumsum(seq)])
cuts = [0] + [int(np.argmin(np.abs(pre - pre[-1] * c / C))) for c in range(1, C)] + [T]
runs = np.array([seq[cuts[c]:cuts[c + 1]].sum() for c in range(C)])
lens = np.diff(cuts)
print(f"strided max/mean {f.max() / f.mean():.3f}; range max/mean {runs.max() / runs.mean():.3f}; "
      f"run length {lens.min()}..{lens.max()} (strided {base}..{base + (rem > 0)})")
