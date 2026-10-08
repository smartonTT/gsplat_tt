#!/usr/bin/env python3
"""t439: per-program windows of the pfwc and blend programs from one Tracy device CSV per arm
(10 views + warmup, as bench439.sh), CPU only. Usage: ana439.py dev-S.csv.gz dev-L.csv.gz
Per view: window = first kernel start on any core -> last BRISC/NCRISC/TRISC end on any core,
for the kernels that contain each core's TRISC_1 <zone> (as ana437.py). Warmup instance dropped."""
import csv, gzip, sys
from collections import defaultdict
import numpy as np
C = 1350.0  # cycles per us

def load(path):
    z, st = defaultdict(list), {}
    rows = []
    with gzip.open(path, "rt") as f:
        next(f); next(f)
        for p in csv.reader(f):
            if len(p) < 12 or p[11].strip() not in ("ZONE_START", "ZONE_END"): continue
            nm = p[10].strip()
            if nm not in ("pfwc", "tile_blend_sfpu", "BRISC-KERNEL", "NCRISC-KERNEL", "TRISC-KERNEL"): continue
            rows.append(((int(p[1]), int(p[2])), p[3].strip(), nm, p[11].strip(), int(p[5])))
    for core, r, nm, ty, t in sorted(rows, key=lambda x: x[4]):
        k = (core, r, nm)
        if ty == "ZONE_START": st[k] = t
        elif k in st: z[k].append((st.pop(k), t))
    return z

def windows(z, zone):
    cores = sorted({k[0] for k in z if k[2] == zone and k[1] == "TRISC_1"})
    nv = min(len(z[(c, "TRISC_1", zone)]) for c in cores)
    res = []  # per view: (window ms, mean core end ms, max core end ms)
    for v in range(1, nv):
        st, en = [], []
        for c in cores:
            s1, e1 = z[(c, "TRISC_1", zone)][v]
            ss, ee = [s1], [e1]
            for r in ("BRISC", "NCRISC", "TRISC_0", "TRISC_2"):
                kz = r + "-KERNEL" if r in ("BRISC", "NCRISC") else zone
                if kz == zone:
                    iv = z[(c, r, zone)][v] if len(z[(c, r, zone)]) > v else None
                else:
                    cand = [iv for iv in z[(c, r, kz)] if iv[0] <= s1 + 20000 and iv[1] >= e1 - 20000]
                    iv = min(cand, key=lambda iv: abs(iv[0] - s1)) if cand else None
                if iv: ss.append(iv[0]); ee.append(iv[1])
            st.append(min(ss)); en.append(max(ee))
        t0 = min(st)
        ends = (np.array(en) - t0) / C / 1000
        res.append((ends.max(), ends.mean()))
    return np.array(res), len(cores)

out = {}
for path in sys.argv[1:]:
    z = load(path)
    for zone in ("pfwc", "tile_blend_sfpu"):
        r, nc = windows(z, zone)
        out[(path, zone)] = r
        print(f"{path.split('/')[-1]:14s} {zone:16s} cores={nc} views={len(r)} window(max-core end) "
              f"mean={r[:,0].mean():.3f} ms  mean-core end={r[:,1].mean():.3f}  max/mean={r[:,0].mean()/r[:,1].mean():.3f}")
if len(sys.argv) == 3:
    a, b = sys.argv[1:]
    for zone in ("pfwc", "tile_blend_sfpu"):
        d = out[(b, zone)][:, 0] - out[(a, zone)][:, 0]
        print(f"delta {zone:16s} (2nd - 1st) per view: mean {d.mean():+.3f} ms, min {d.min():+.3f}, max {d.max():+.3f}")
