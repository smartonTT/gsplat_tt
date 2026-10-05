#!/usr/bin/env python3
"""t222: how much of the pfwc makespan is feed (reader/writer) rather than TRISC compute,
on the t221 default (WRITER_SPLIT + COVCAM_SFPU), from the STEPCYC=1 Tracy CSV.

Per (core, launch): TRISC busy = wall - input wait (max over TRISC0/1/2); feed wall =
max(BRISC, NCRISC) pfwc_ws wall. Feed slack on a core = feed wall - TRISC busy (start
wait + writer tail). Makespan slack per launch = max_core(feed wall, TRISC wall) -
max_core(TRISC busy): what an ideal (zero-latency) feed could still save.
  feed_slack.py prof-both-dev.csv.gz
"""
import gzip, sys
from collections import defaultdict
import numpy as np

MHZ = 1350.0
T_COL, DATA_COL, ZONE_COL = 5, 6, 10
WS = 'n wall wait cls pfx rec opn tail rd'.split()
PC = 'n wall init wait'.split()
seq = defaultdict(list)
with gzip.open(sys.argv[1], 'rt') as f:
    f.readline(); f.readline()
    for line in f:
        p = line.rstrip('\n').split(',')
        if len(p) <= ZONE_COL or p[ZONE_COL] not in ('pfwc_pc', 'pfwc_ws'):
            continue
        v = int(p[DATA_COL]); i, val = v >> 32, v & 0xffffffff
        if p[ZONE_COL] == 'pfwc_ws':
            i &= 15
        k = (p[1] + '-' + p[2], p[3], p[ZONE_COL])
        if i == 0 or not seq[k]:
            seq[k].append({})
        seq[k][-1][i] = val
ms = lambda c: c / MHZ / 1e3
cores = sorted({k[0] for k in seq})
nl = min(len(v) for v in seq.values())
rows = []  # launch, core, trisc_wall, trisc_busy, trisc_wait, feed_wall, b_wall, n_wall, b_tail, n_tail
for c in cores:
    for L in range(nl):
        tw = [seq[(c, f'TRISC_{t}', 'pfwc_pc')][L] for t in range(3)]
        b = seq[(c, 'BRISC', 'pfwc_ws')][L]; n = seq[(c, 'NCRISC', 'pfwc_ws')][L]
        twall = max(ms(t[1]) for t in tw)
        tbusy = max(ms(t[1] - t[3]) for t in tw)
        twait = max(ms(t[3]) for t in tw)
        rows.append((L, c, twall, tbusy, twait, max(ms(b[1]), ms(n[1])), ms(b[1]), ms(n[1]),
                     ms(b[7]), ms(n[7]), ms(b[6]), ms(n[6])))
a = np.array([r[2:] for r in rows]); Ls = np.array([r[0] for r in rows])
print(f'cores={len(cores)} launches={nl} (launch 0 = JIT warm-up of view 0)')
print('per (core, launch) mean ms: TRISC wall %.3f busy %.3f input-wait %.3f | feed wall %.3f '
      '(BRISC %.3f NCRISC %.3f) | feed - TRISC busy %.3f | tail B %.3f N %.3f | opn B %.3f N %.3f'
      % tuple([a[:, 0].mean(), a[:, 1].mean(), a[:, 2].mean(), a[:, 3].mean(), a[:, 4].mean(),
               a[:, 5].mean(), (a[:, 3] - a[:, 1]).mean(), a[:, 6].mean(), a[:, 7].mean(),
               a[:, 8].mean(), a[:, 9].mean()]))
print('\nlaunch | makespan(max feed/TRISC wall) | max TRISC busy | feed slack | crit core feed-TRISCbusy | crit core')
tot = []
for L in range(nl):
    m = Ls == L
    sub = a[m]; cs = [r[1] for r in rows if r[0] == L]
    span = np.maximum(sub[:, 3], sub[:, 0]).max()
    j = int(np.maximum(sub[:, 3], sub[:, 0]).argmax())
    floor = sub[:, 1].max()
    tot.append(span - floor)
    print(f'  {L} | {span:.3f} | {floor:.3f} | {span - floor:.3f} | {sub[j, 3] - sub[j, 1]:.3f} | {cs[j]}')
print(f'mean feed slack on the makespan (launches 1..): {np.mean(tot[1:]):.3f} ms/view, max {max(tot):.3f}')
d = a[:, 3] - a[:, 1]
print('per-core feed - TRISC busy: p50 %.3f p90 %.3f p99 %.3f max %.3f ms'
      % tuple(np.percentile(d, [50, 90, 99, 100])))
