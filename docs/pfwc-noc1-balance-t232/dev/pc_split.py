#!/usr/bin/env python3
"""t232 (copy of t221 + the writer's fl field; from t197 pc_split.py, + pfwc_ws of the t207 split writer): pfwc per-step / per-RISC cycle split from a Tracy device CSV
(capture with GSPLAT_TT_PFWC_STEPCYC=1|2, see remote_prof.sh).

Each instrumented RISC records timestamped data at kernel end, value =
(index << 32) | wall cycles (1350 MHz) summed over the core's chunks:
  pfwc_pc  TRISC0/1/2  n wall init s0..s12 [copy mul add acq pack]
           steps: 0 input wait, 1 transform, 2 recip, 3 depth, 4 means, 5 cov_cam,
           6 a, 7 b, 8 c, 9 conic, 10 radii x, 11 radii y, 12 vis+pops
           (last 5 with STEPCYC=2: cov_cam split by call type)
  pfwc_pr  NCRISC      n wall reserve barrier
  pfwc_pw  BRISC       n wall wait bar cls rec tail iss m pr
  pfwc_ws  BRISC/NCRISC (GSPLAT_TT_PFWC_WRITER_SPLIT=1), index = role*16 + i:
           n wall wait cls pfx rec opn tail rd
Prints, per RISC, the mean over (core, launch) and the max-wall core, in ms per launch
(one launch per view).
  pc_split.py profile_log_device.csv
"""
import sys
from collections import defaultdict
import numpy as np

MHZ = 1350.0
T_COL, DATA_COL, ZONE_COL = 5, 6, 10
STEPS = 'wait xform recip depth means cov_cam a b c conic radx rady vis+pop'.split()
FMT = {'pfwc_pc': ['n', 'wall', 'init'] + STEPS + 'copy mul add acq pack'.split(),
       'pfwc_pr': 'n wall reserve barrier'.split(),
       'pfwc_pw': 'n wall wait bar cls rec tail iss m pr'.split(),
       'pfwc_ws': 'n wall wait cls pfx rec opn tail rd fl'.split()}
NOCYC = ('n', 'm', 'pr')
seq = defaultdict(list)  # (core, risc, zone) -> list of launches (dict idx -> value)
with open(sys.argv[1]) as f:
    f.readline(); f.readline()
    for line in f:
        p = line.rstrip('\n').split(',')
        if len(p) <= ZONE_COL or p[ZONE_COL] not in FMT:
            continue
        v = int(p[DATA_COL]); i, val = v >> 32, v & 0xffffffff
        if p[ZONE_COL] == 'pfwc_ws':
            i &= 15
        k = (p[1] + '-' + p[2], p[3], p[ZONE_COL])
        if i == 0 or not seq[k]:
            seq[k].append({})
        seq[k][-1][i] = val
if not seq:
    sys.exit('no pfwc_p* markers: capture with GSPLAT_TT_PFWC_STEPCYC=1|2')
ms = lambda c: c / MHZ / 1e3
nl = min(len(v) for v in seq.values())
print(f'cores={len({k[0] for k in seq})} launches/core={nl}..{max(len(v) for v in seq.values())} series={len(seq)}')
for risc, zone in sorted({(k[1], k[2]) for k in seq}):
    f = FMT[zone]
    recs = [r for k, v in seq.items() if k[1] == risc and k[2] == zone for r in v]
    w = max(len(r) for r in recs)
    rows = np.array([[r.get(i, 0) for i in range(w)] for r in recs if len(r) == w], float)
    cyc = [i for i in range(w) if f[i] not in NOCYC]
    print(f'\n{risc} {zone}: {len(rows)} (core, launch) records')
    print('  mean ms : ' + ', '.join(f'{f[i]} {ms(rows[:, i].mean()):.3f}' for i in cyc))
    j = rows[:, 1].argmax()
    print('  max core: ' + ', '.join(f'{f[i]} {ms(rows[j, i]):.3f}' for i in cyc) + f'  (n={int(rows[j, 0])})')
    print(f'  wall max/mean {ms(rows[:, 1].max()):.3f}/{ms(rows[:, 1].mean()):.3f}  chunks/launch mean {rows[:, 0].mean():.1f}'
          + (f'  vis/launch {rows[:, f.index("m")].sum() / nl:.0f} pairs/launch {rows[:, f.index("pr")].sum() / nl:.0f}'
             if 'm' in f else ''))
