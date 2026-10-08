#!/usr/bin/env python3
"""t437: per-RISC pfwc breakdown per core per view from #427 device CSVs (CPU only).
Usage: ana437.py ARM_PREFIX GRID_X   e.g. ../eth-tracy-t427/raw/dev-E12 12
For each core and view: pfwc-program BRISC-KERNEL / NCRISC-KERNEL / TRISC0..2 'pfwc' zones,
all relative to the program's first kernel start on any core (launch skew included).
Writes ARM.npz-style CSV (per core x view x riscs) to stdout summary + core_view.csv."""
import csv, sys, os
from collections import defaultdict
import numpy as np
pre, gx = sys.argv[1], int(sys.argv[2])
C = 1350.0  # cycles per us
RISCS = ["BRISC", "NCRISC", "TRISC_0", "TRISC_1", "TRISC_2"]
rows = []  # (chunk, core, risc, name, kind, t)
for ci, ch in enumerate(("c0", "c10", "c20")):
    with open(f"{pre}-{ch}.csv") as f:
        next(f); next(f)
        for p in csv.reader(f):
            if len(p) < 12: continue
            nm, ty = p[10].strip(), p[11].strip()
            if ty not in ("ZONE_START", "ZONE_END"): continue
            if nm not in ("pfwc", "BRISC-KERNEL", "NCRISC-KERNEL", "TRISC-KERNEL"): continue
            rows.append((ci, (int(p[1]), int(p[2])), p[3].strip(), nm, ty, int(p[5])))
# zones per (chunk, core, risc, name) in time order
z = defaultdict(list); st = {}
for ci, core, r, nm, ty, t in sorted(rows, key=lambda x: x[5]):
    k = (ci, core, r, nm)
    if ty == "ZONE_START": st[k] = t
    elif k in st: z[k].append((st.pop(k), t))
cores = sorted({k[1] for k in z})
# pfwc instances: TRISC_1 'pfwc' zones; per chunk, index = view (first = warmup, dropped later)
out = []  # chunk, view, core, then per risc (start,end)
for ci in range(3):
    pf = {c: z[(ci, c, "TRISC_1", "pfwc")] for c in cores}
    nv = min(len(v) for v in pf.values())
    for v in range(nv):
        for c in cores:
            s1, e1 = pf[c][v]
            rec = {"TRISC_1": (s1, e1)}
            for r in ("TRISC_0", "TRISC_2"):
                rec[r] = z[(ci, c, r, "pfwc")][v]
            for r in ("BRISC", "NCRISC"):
                # kernel zone that contains the TRISC_1 pfwc zone
                cand = [iv for iv in z[(ci, c, r, r + "-KERNEL")] if iv[0] <= s1 + 20000 and iv[1] >= e1 - 20000]
                rec[r] = min(cand, key=lambda iv: abs(iv[0] - s1)) if cand else (np.nan, np.nan)
            out.append((ci, v, c, rec))
# write core_view.csv relative to per-instance program start (min start over cores/riscs)
inst = defaultdict(list)
for ci, v, c, rec in out: inst[(ci, v)].append((c, rec))
od = os.path.dirname(os.path.abspath(__file__))
tag = os.path.basename(pre)
with open(os.path.join(od, f"core_view-{tag}.csv"), "w") as f:
    w = csv.writer(f)
    w.writerow(["chunk", "view", "x", "y", "c"] + [f"{r}_{s}" for r in RISCS for s in ("s", "e")])
    for (ci, v), lst in sorted(inst.items()):
        if v == 0: continue  # warmup launch of each chunk
        t0 = min(min(rec[r][0] for r in RISCS) for c, rec in lst)
        for c, rec in lst:
            lx = (c[0] - 1) if c[0] <= 7 else (c[0] - 5)  # x 1..6 -> 0..5, 11..16 -> 6..11 (cols 7,10 harvested)
            ci_ = (c[1] - 2) * gx + lx
            w.writerow([ci, v, c[0], c[1], ci_] + [f"{(rec[r][k] - t0) / C:.2f}" for r in RISCS for k in (0, 1)])
print("wrote", os.path.join(od, f"core_view-{tag}.csv"), "cores", len(cores))
