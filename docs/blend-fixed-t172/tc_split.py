#!/usr/bin/env python3
"""t172: per-tile blend fixed-cost split from GSPLAT_TT_MB_TILECYC=1 DPRINT.

Line per tile: "TC rec live cyc disp ntrb trb stage tail init emit wall" (MATH wall
cycles, 1350 MHz); "TCEND n drop" per core and launch.
  tc_split.py tc.dprint [--cores 110]
Prints per-launch mean-per-core ms of each part, the fixed-cost share per tile,
and least-squares fits of the loop time with and without the dispatch count.
"""
import re, sys
from collections import defaultdict
import numpy as np

MHZ = 1350.0
F = 'rec live cyc disp ntrb trb stage tail init emit wall'.split()
per_core = defaultdict(lambda: [[]])
drops = 0
for line in open(sys.argv[1], errors='replace'):
    m = re.search(r'^\d+:(\d+)-(\d+):TR1:\s*TC(END)?\s+([\d ]+)', line)
    if not m:
        continue
    core = (int(m.group(1)), int(m.group(2)))
    v = [int(x) for x in m.group(4).split()]
    if m.group(3):
        per_core[core].append([])
        drops += v[1] if len(v) > 1 else 0
    elif len(v) == len(F):
        per_core[core][-1].append(v)
nl = min(len(v) - 1 for v in per_core.values())
print(f'cores={len(per_core)} launches={nl} dropped_tiles={drops}')
for L in range(nl):
    rows = np.array([t for v in per_core.values() for t in v[L]], float)
    d = dict(zip(F, rows.T))
    nc = len(per_core)
    ms = lambda x: x.sum() / MHZ / 1e3 / nc
    loop_rec = d['cyc'] - d['trb'] - d['stage']
    fixed = d['stage'] + d['tail'] + d['init'] + d['emit']
    other = d['wall'] - d['cyc'] - d['tail'] - d['init'] - d['emit']
    print(f'\n== launch {L}: tiles {len(rows)}, records {int(d["rec"].sum())}, '
          f'live {int(d["live"].sum())}, disp {int(d["disp"].sum())}, readbacks {int(d["ntrb"].sum())}')
    print('mean per core ms: ' + ' '.join(f'{k}={ms(x):.3f}' for k, x in [
        ('wall', d['wall']), ('loop(records)', loop_rec), ('trb', d['trb']), ('stage', d['stage']),
        ('tail', d['tail']), ('init', d['init']), ('emit', d['emit']), ('other(waits)', other)]))
    nt = len(rows)
    print('per tile us: ' + ' '.join(f'{k}={x.sum()/MHZ/nt:.2f}' for k, x in [
        ('stage', d['stage']), ('tail', d['tail']), ('init', d['init']), ('emit', d['emit']),
        ('other', other), ('trb_each', d['trb'] / np.maximum(d['ntrb'], 1))]))
    print(f'fixed (stage+tail+init+emit) per tile {fixed.sum()/MHZ/nt:.2f} us, '
          f'{ms(fixed):.3f} ms/core; + waits {ms(fixed + other):.3f}')
    us = d['cyc'] / MHZ
    for name, X in [('cyc ~ 1 + rec + live', np.c_[np.ones(nt), d['rec'], d['live']]),
                    ('cyc ~ 1 + rec + live + disp + ntrb', np.c_[np.ones(nt), d['rec'], d['live'], d['disp'], d['ntrb']]),
                    ('cyc-trb-stage ~ 1 + rec + live + disp', np.c_[np.ones(nt), d['rec'], d['live'], d['disp']])]:
        y = us if 'trb' not in name else loop_rec / MHZ
        c, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = np.sqrt(np.mean((X @ c - y) ** 2))
        print(f'fit {name}: ' + ' '.join(f'{x:.4f}' for x in c) + f'  rms {r:.1f} us')
