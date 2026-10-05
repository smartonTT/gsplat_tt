#!/usr/bin/env python3
"""t197: per-launch critical-core view of the pfwc counters (pc_split.py input).
For each launch: slowest core per RISC, and on all cores the writer busy time
(wall - wait) vs the math thread's (TRISC_1) wall, to see which side limits.
  core_crit.py profile_log_device.csv[.gz]
"""
import gzip, sys
from collections import defaultdict
import numpy as np
MS = 1350.0e3
op = gzip.open if sys.argv[1].endswith('.gz') else open
seq = defaultdict(list)
with op(sys.argv[1], 'rt') as f:
    f.readline(); f.readline()
    for line in f:
        p = line.rstrip('\n').split(',')
        if len(p) <= 10 or not p[10].startswith('pfwc_p'):
            continue
        v = int(p[6]); i, val = v >> 32, v & 0xffffffff
        k = (p[1] + '-' + p[2], p[3])
        if i == 0 or not seq[k]:
            seq[k].append({})
        seq[k][-1][i] = val
cores = sorted({k[0] for k in seq})
nl = min(len(v) for v in seq.values())
for L in range(nl):
    g = lambda c, r, i: seq[(c, r)][L][i] / MS
    wall = {r: np.array([g(c, r, 1) for c in cores]) for r in ('BRISC', 'NCRISC', 'TRISC_0', 'TRISC_1', 'TRISC_2')}
    wbusy = np.array([g(c, 'BRISC', 1) - g(c, 'BRISC', 2) for c in cores])  # writer wall - wait on compute
    vis = np.array([seq[(c, 'BRISC')][L][8] for c in cores]); prs = np.array([seq[(c, 'BRISC')][L][9] for c in cores])
    j = int(np.argmax(wall['BRISC']))
    # compute "own" time: TRISC_1 wall minus its means+depth steps above the median (output back-pressure)
    print(f'launch {L}: wall max/mean BR {wall["BRISC"].max():.3f}/{wall["BRISC"].mean():.3f} '
          f'TR1 {wall["TRISC_1"].max():.3f}/{wall["TRISC_1"].mean():.3f}; writer busy max/mean/min '
          f'{wbusy.max():.3f}/{wbusy.mean():.3f}/{wbusy.min():.3f}; cores with writer busy > 0.95*wall: '
          f'{int((wbusy > 0.95 * wall["BRISC"]).sum())}')
    print(f'   slowest core {cores[j]}: wall {wall["BRISC"][j]:.3f} writer busy {wbusy[j]:.3f} vis {vis[j]} pairs {prs[j]} '
          f'(mean vis {vis.mean():.0f} pairs {prs.mean():.0f}); corr(wall, pairs) {np.corrcoef(wall["BRISC"], prs)[0,1]:.2f}')
    o = np.argsort(-wall['BRISC'])[:5]
    print('   top5 walls ' + ' '.join(f'{cores[i]}:{wall["BRISC"][i]:.3f}/busy{wbusy[i]:.3f}/pr{prs[i]}' for i in o))
