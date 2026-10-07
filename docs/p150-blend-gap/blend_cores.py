#!/usr/bin/env python3
"""t366: per-core mat/blend spans from a Tracy csvexport -u dump (device zones; thread = core/RISC).
Per frame: window (first start -> last end), per-core span mean/min/max, spread (max-min end).
Usage: blend_cores.py <tracy-u.csv[.gz]> [zone ...]"""
import csv, gzip, sys
from collections import defaultdict
import numpy as np

path = sys.argv[1]
zones = sys.argv[2:] or ["mat_cull_mask", "sort_subchunk_mat", "tile_blend_load", "tile_blend_sfpu"]
op = gzip.open if path.endswith(".gz") else open
rows = defaultdict(list)
with op(path, "rt") as f:
    for r in csv.DictReader(f):
        if r["name"] in zones:
            s = int(r["ns_since_start"]); rows[r["name"]].append((s, s + int(r["exec_time_ns"]), r["thread"]))
# frames from the blend sfpu zone starts (gap > 3 ms = new frame)
ref = sorted(s for s, _, _ in rows["tile_blend_sfpu"])
cuts = [ref[0]] + [b for a, b in zip(ref, ref[1:]) if b - a > 3e6]
def frame_of(t):
    i = np.searchsorted(cuts, t + 2.5e6, side="right") - 1  # mat starts ~2 ms before blend
    return int(i)
print(f"{path}: frames={len(cuts)} (stats over frames 1..)")
print(f"{'zone':20s} {'window':>7s} {'core_mean':>9s} {'core_min':>8s} {'core_max':>8s} {'end_spread':>10s} {'start_spread':>12s} ncores")
for z in zones:
    per = defaultdict(lambda: defaultdict(list))
    for s, e, th in rows[z]:
        per[frame_of(s)][th].append((s, e))
    W, M, MN, MX, ES, SS, NC = [], [], [], [], [], [], []
    for fr, cores in per.items():
        if fr < 1: continue
        spans = []; ends = []; starts = []
        for th, iv in cores.items():
            s0 = min(a for a, _ in iv); e0 = max(b for _, b in iv)
            spans.append(sum(b - a for a, b in iv)); ends.append(e0); starts.append(s0)
        W.append(max(ends) - min(starts)); M.append(np.mean(spans)); MN.append(min(spans)); MX.append(max(spans))
        ES.append(max(ends) - min(ends)); SS.append(max(starts) - min(starts)); NC.append(len(cores))
    f = lambda v: np.mean(v) / 1e6
    print(f"{z:20s} {f(W):7.3f} {f(M):9.3f} {f(MN):8.3f} {f(MX):8.3f} {f(ES):10.3f} {f(SS):12.3f} {int(np.mean(NC))}")
