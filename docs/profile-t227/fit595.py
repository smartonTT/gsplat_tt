#!/usr/bin/env python3
"""t595: t224-style linear fits of per-view program windows (ana-P.txt) against M (visible) and
P (pairs) from VIEW_COUNTS (run-counts.log). Usage: fit595.py out/ana-P.txt out/run-counts.log"""
import re, sys
import numpy as np
ana, cnt = sys.argv[1:3]
C = {}
for l in open(cnt):
    m = re.search(r"VIEW_COUNTS i=(\d+) .*N=(\d+) M=(\d+) P=(\d+)", l)
    if m:
        C[int(m[1])] = (int(m[2]) / 1e6, int(m[3]) / 1e6, int(m[4]) / 1e6)
W = {}
for l in open(ana):
    m = re.match(r"\s*(\d+)\s+([\d.]+)/\s*([\d.]+)\s+([\d.]+)/\s*([\d.]+)\s+([\d.]+)/\s*([\d.]+)\s+([\d.]+)/\s*([\d.]+)\s+([\d.]+)", l)
    if m and int(m[1]) < 30 and int(m[1]) not in W:
        g = [float(x) for x in m.groups()[1:]]
        W[int(m[1])] = dict(pfwc=g[1], k2=g[3], sort=g[5], mb=g[7])
vs = [v for v in sorted(set(W) & set(C)) if v % 5]  # t224: first view of each chunk dropped
M = np.array([C[v][1] for v in vs]); P = np.array([C[v][2] for v in vs])
print(f"views {len(vs)}  M mean {M.mean():.3f}  P mean {P.mean():.3f} (millions)")
print("| program | fit (ms, M and P in millions) | r | fixed part of the mean |\n|---|---|---|---|")
tot = np.zeros(len(vs))
for k, x, xn in (("pfwc", M, "M"), ("k2", P, "P"), ("sort", P, "P"), ("mb", P, "P")):
    y = np.array([W[v][k] for v in vs]); tot += y
    b, a = np.polyfit(x, y, 1); r = np.corrcoef(x, y)[0, 1]
    print(f"| {k if k != 'mb' else 'mat+blend'} | {a:.3f} + {b:.3f} x {xn} | {r:.3f} | {100 * a / y.mean():.0f}% |")
b, a = np.polyfit(P, tot, 1); r = np.corrcoef(P, tot)[0, 1]
print(f"| sum of windows | {a:.3f} + {b:.3f} x P | {r:.3f} | {100 * a / tot.mean():.0f}% |")
