#!/usr/bin/env python3
"""t183 cross-check on exact per-tile (rec, live, cycles) from the t147 dprint (older tip,
PRECULL=1). Compares claim orders under 'early' (today) and 'late' (reserve slot, then claim)."""
import re, sys, os, statistics as S
from collections import defaultdict
sys.path.insert(0, os.path.dirname(__file__))
from sim import replay, stats, rank_interleave, build_lpt
MHZ = 1350.0; FIT = 8192
per_core = defaultdict(lambda: [[]])
for line in open(sys.argv[1], errors='replace'):
    m = re.search(r'^\d+:(\d+)-(\d+):TR1:\s*TC(END)?\s+(\d+)(?:\s+(\d+)\s+(\d+))?', line)
    if not m: continue
    c = (int(m.group(1)), int(m.group(2)))
    if m.group(3): per_core[c].append([])
    else: per_core[c][-1].append((int(m.group(4)), int(m.group(5)), int(m.group(6)) / MHZ))
n = len(per_core); nl = min(len(v) - 1 for v in per_core.values())
print(f'cores={n} launches={nl}')
for L in range(nl):
    T = [t for v in per_core.values() for t in v[L]]
    rec = [t[0] for t in T]; us = [t[2] for t in T]
    fit = [87 + 0.027 * t[0] + 0.197 * t[1] for t in T]
    sub = [[u * min(FIT, r - k) / r for k in range(0, r, FIT)] if r else [u] for r, u in zip(rec, us)]
    orders = {'lpt_rec(today)': rank_interleave(build_lpt(rec, n)),
              'rec_desc': sorted(range(len(T)), key=lambda i: -rec[i]),
              'fit_desc(live)': sorted(range(len(T)), key=lambda i: -fit[i]),
              'oracle_us': sorted(range(len(T)), key=lambda i: -us[i])}
    orders['lpt_rec(today)'] += [i for i in range(len(T)) if rec[i] == 0]
    print(f'launch {L}: tiles={len(T)} sum_us={sum(us):.0f} mean/core={sum(us)/n:.1f} max_tile={max(us):.0f}')
    for pol in ('early', 'late'):
        for k, o in orders.items():
            mx, mn, tl = stats(replay(sub, o, n, pol)[0])
            print(f'  {pol:5s} {k:16s} max_end {mx:8.1f} tail {tl:6.1f}')
