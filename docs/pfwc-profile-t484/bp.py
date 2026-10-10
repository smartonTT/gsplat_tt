#!/usr/bin/env python3
"""t484: writer back-pressure on TRISC. The first output CB reserve of a chunk sits in step 4
(means, T1). Excess over the core-launch minimum of step 4 = time TRISC waited on a writer.
Models the pfwc window if no core waited on its writers. ana.py's loader. bp.py out/dev-s.csv.gz"""
import sys
from collections import defaultdict
import numpy as np
import ana

raw, ts = ana.load(sys.argv[1])
runs = set(ana.launches(raw))
t1 = {}
for (core, risc, z), lst in ts.items():
    if risc == "TRISC_1" and z == "pfwc_pc":
        for r, d in enumerate(lst):
            if r in runs:
                t1[(core, r)] = (d.get(1, 0), d.get(7, 0))  # wall, s4 means
base = np.percentile([m for _, m in t1.values()], 5)
by_run = defaultdict(list)
for (c, r), (w, m) in t1.items():
    by_run[r].append((w, max(0.0, m - base)))
ex_all = np.array([e for v in by_run.values() for _, e in v])
print(f"step-4 floor (p5) {ana.ms(base):.3f} ms; back-pressure mean {ana.ms(ex_all.mean()):.3f} ms, "
      f"p90 {ana.ms(np.percentile(ex_all, 90)):.3f}, max {ana.ms(ex_all.max()):.3f}; "
      f"core-launches with >0.05 ms: {100 * (ex_all > 0.05 * ana.C * 1000).mean():.1f}%")
wmax = np.mean([ana.ms(max(w for w, _ in v)) for v in by_run.values()])
wnobp = np.mean([ana.ms(max(w - e for w, e in v)) for v in by_run.values()])
wmean = np.mean([ana.ms(np.mean([w for w, _ in v])) for v in by_run.values()])
print(f"TRISC_1 wall: slowest core {wmax:.3f} ms, slowest core minus its back-pressure {wnobp:.3f} ms "
      f"(-{wmax - wnobp:.3f}), mean core {wmean:.3f} ms")
