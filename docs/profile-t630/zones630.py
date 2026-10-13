#!/usr/bin/env python3
"""t630: per-program zone timeline (median over views): per zone and RISC, the median over cores of
start offset from the program window start, duration (first START -> last END on that core), and the
same on the critical (last-ending) core. Usage: ANA_PROGS=mat+blend,sort,pfwc zones630.py STITCHED.csv.gz"""
import os, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ana506 as A
C = 1350e3
iv, td = A.load_dev(sys.argv[1])
FWI, Z, D, nv, cores = A.assign(iv, td)
for p, pn in enumerate(A.PROGS):
    acc = defaultdict(lambda: [[], [], [], []])
    for v in range(nv):
        fw = FWI[(v, p)]
        if not fw:
            continue
        w0 = min(s for _, _, s, _ in fw); w1 = max(e for _, _, _, e in fw)
        cend = defaultdict(int)
        for c, r, s, e in fw:
            cend[c] = max(cend[c], e)
        crit = max(cend, key=cend.get)
        per = defaultdict(lambda: [1 << 62, 0])
        for (vv, pp, z), L in Z.items():
            if vv == v and pp == p:
                for c, r, s, e in L:
                    q = per[(z, r, c)]; q[0] = min(q[0], s); q[1] = max(q[1], e)
        for (z, r, c), (s, e) in per.items():
            a = acc[(z, r)]
            a[0].append((s - w0) / C); a[1].append((e - s) / C)
            if c == crit:
                a[2].append((s - w0) / C); a[3].append((e - s) / C)
    print(f"\n### {pn}\n\n| zone | RISC | start (all) | dur (all) | start (crit) | dur (crit) |\n|---|---|---|---|---|---|")
    md = lambda x: f"{np.median(x):.3f}" if x else "-"
    for (z, r), a in sorted(acc.items(), key=lambda kv: np.median(kv[1][0])):
        if z.endswith("-KERNEL"):
            continue
        print(f"| {z} | {r} | {md(a[0])} | {md(a[1])} | {md(a[2])} | {md(a[3])} |")
