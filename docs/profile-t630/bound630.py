#!/usr/bin/env python3
"""t630: per-view mat+blend window vs the window if each core's mat phase overlapped its blend
(max over cores of blend zone + drain), and per-view mat-phase spread. Mat+blend groups whose view has
no k2 (chunk-final) are skipped. Usage: ANA_PROGS=mat+blend,sort,pfwc bound630.py STITCHED.csv.gz"""
import os, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ana506 as A
C = 1350e3
iv, td = A.load_dev(sys.argv[1])
FWI, Z, D, nv, cores = A.assign(iv, td)
rows = []
for v in range(nv):
    fw = FWI[(v, 0)]
    z = {}
    for (vv, pp, n), L in Z.items():
        if vv == v and pp == 0:
            for c, r, s, e in L:
                q = z.setdefault((c, r, n), [s, e]); q[0] = min(q[0], s); q[1] = max(q[1], e)
    if not any(n == "k2_pairs" for (_, _, n) in z):
        continue
    w0 = min(s for _, _, s, _ in fw); w1 = max(e for _, _, _, e in fw)
    end = defaultdict(int)
    for c, r, s, e in fw:
        end[c] = max(end[c], e)
    mat, bl, tail = [], [], []
    for c in end:
        b = z.get((c, "TRISC_1", "tile_blend_sfpu")); mz = z.get((c, "TRISC_1", "mat_cull_mask"))
        if not (b and mz):
            continue
        mat.append((mz[1] - mz[0]) / C); bl.append((b[1] - b[0]) / C); tail.append((end[c] - b[1]) / C)
    mat, bl, tail = map(np.array, (mat, bl, tail))
    rows.append(((w1 - w0) / C, mat.mean(), mat.max(), bl.mean(), bl.max(), (bl + tail).max(), tail.max(),
                 (mat + bl + tail).max()))
R = np.array(rows)
names = ["window", "mat mean", "mat max", "blend mean", "blend max", "max(blend+drain)", "drain max", "max(mat+blend+drain)"]
print(f"views {len(R)} (steady mat+blend groups with folded k2)\n")
print("| quantity | median | min | max |\n|---|---|---|---|")
for i, n in enumerate(names):
    print(f"| {n} | {np.median(R[:, i]):.3f} | {R[:, i].min():.3f} | {R[:, i].max():.3f} |")
print(f"\nfit mat max vs blend max over views: slope {np.polyfit(R[:, 4], R[:, 2], 1)[0]:.3f}")
print(f"overlap bound (window - max(blend+drain)) median {np.median(R[:, 0] - R[:, 5]):.3f} ms")
