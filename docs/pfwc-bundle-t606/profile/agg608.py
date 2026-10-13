#!/usr/bin/env python3
"""t608: 30-view pfwc window per arm from the pfwc608.py summaries.
Chunk 0 (views 0:5) is out/prof-<arm>.txt, chunks 5..25 are out30/prof-<arm>-<a>.txt.
Prints per chunk the MEAN window and slowest-core busy per RISC, then the mean over chunks and the
gain vs base, next to the t606/t599 b2b gains.  Usage: agg608.py [profile dir]
"""
import os
import re
import sys

D = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
ARMS = ["base", "rec", "vis", "new"]
CH = [0, 5, 10, 15, 20, 25]
B2B = {"rec": "0.071 (t599)", "vis": "0.022 (t606 C, 1 round)", "new": "0.070 (t606 r1-r4)"}


def read(arm, c):
    p = os.path.join(D, "out", f"prof-{arm}.txt") if c == 0 else os.path.join(D, "out30", f"prof-{arm}-{c}.txt")
    if not os.path.exists(p):
        return None
    t = open(p).read()
    m = re.search(r"^MEAN window ([\d.]+) ms; setter per view: (.*)$", t, re.M)
    if not m:
        return None
    b = dict(re.findall(r"(TRISC_\d|NCRISC|BRISC)=([\d.]+)", re.search(r"^slowest-core busy.*$", t, re.M).group(0)))
    return float(m.group(1)), {k: float(v) for k, v in b.items()}, m.group(2)


res = {a: {c: read(a, c) for c in CH} for a in ARMS}
print("pfwc window (ms, mean of 5 views per chunk) | slowest core TRISC_1 / NCRISC / BRISC busy | setter")
for c in CH:
    row = [f"{c:2d}:{c + 5:<2d}"]
    for a in ARMS:
        r = res[a][c]
        row.append(f"{a} {r[0]:.3f} ({r[1]['TRISC_1']:.3f}/{r[1]['NCRISC']:.3f}/{r[1]['BRISC']:.3f})" if r else f"{a} -")
    print("  ".join(row))
print()
for a in ARMS:
    cs = [c for c in CH if res[a][c] and res["base"][c]]
    w = sum(res[a][c][0] for c in cs) / len(cs)
    wb = sum(res["base"][c][0] for c in cs) / len(cs)
    busy = {k: sum(res[a][c][1][k] for c in cs) / len(cs) for k in ("TRISC_0", "TRISC_1", "TRISC_2", "NCRISC", "BRISC")}
    sb = " ".join(f"{k}={v:.3f}" for k, v in busy.items())
    print(f"{a:4s} views={5 * len(cs):2d} window {w:.3f} gain vs base {wb - w:+.3f}  b2b gain {B2B.get(a, '-')}  | {sb}")
for a in ARMS:
    print(a, "setters:", "; ".join(f"{c}:{res[a][c][2]}" for c in CH if res[a][c]))
