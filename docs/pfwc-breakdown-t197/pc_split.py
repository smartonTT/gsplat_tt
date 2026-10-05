#!/usr/bin/env python3
"""t197: pfwc per-step / per-RISC cycle split from GSPLAT_TT_PFWC_STEPCYC DPRINT.

Lines (wall cycles, 1350 MHz), one per core, RISC and launch:
  TRISC0/1/2  "PC n wall init s0..s12"  steps: 0 input wait, 1 transform, 2 recip,
              3 depth, 4 means, 5 cov_cam, 6 a, 7 b, 8 c, 9 conic, 10 radii x,
              11 radii y, 12 vis+pops
              "PO copy mul add acquire pack" (STEPCYC=2, cov_cam only)
  NCRISC      "PR n wall reserve barrier"
  BRISC       "PW n wall wait bar cls rec tail iss m pr"
  pc_split.py pcN.dprint
Prints per launch: mean and max over cores, in ms.
"""
import re, sys
from collections import defaultdict
import numpy as np

MHZ = 1350.0
STEPS = 'wait xform recip depth means cov_cam a b c conic radx rady vis+pop'.split()
FMT = {'PC': ['n', 'wall', 'init'] + STEPS, 'PO': 'copy mul add acq pack'.split(),
       'PR': 'n wall reserve barrier'.split(),
       'PW': 'n wall wait bar cls rec tail iss m pr'.split()}
seq = defaultdict(list)  # (core, risc, tag) -> [values per launch]
for line in open(sys.argv[1], errors='replace'):
    m = re.search(r'^\d+:(\d+)-(\d+):(\w+):\s*(PC|PO|PR|PW)\s+([\d ]+)', line)
    if not m:
        continue
    v = [int(x) for x in m.group(5).split()]
    if len(v) == len(FMT[m.group(4)]):
        seq[(m.group(1) + '-' + m.group(2), m.group(3), m.group(4))].append(v)
nl = min(len(v) for v in seq.values())
cores = {k[0] for k in seq}
print(f'cores={len(cores)} launches={nl} series={len(seq)}')
ms = lambda c: c / MHZ / 1e3
for L in range(nl):
    print(f'\n== launch {L}')
    for risc, tag in [('TR0', 'PC'), ('TR1', 'PC'), ('TR2', 'PC'), ('TR0', 'PO'), ('TR1', 'PO'),
                      ('TR2', 'PO'), ('NC', 'PR'), ('BR', 'PW')]:
        rows = np.array([v[L] for k, v in seq.items() if k[1] == risc and k[2] == tag], float)
        if not len(rows):
            continue
        f = FMT[tag]
        cyc = [i for i, n in enumerate(f) if n not in ('n', 'm', 'pr')]
        mean = ', '.join(f'{f[i]} {ms(rows[:, i].mean()):.3f}' for i in cyc)
        print(f'{risc} {tag} mean: {mean}')
        if 'wall' in f:
            w = f.index('wall'); j = rows[:, w].argmax()
            crit = ', '.join(f'{f[i]} {ms(rows[j, i]):.3f}' for i in cyc)
            print(f'{risc} {tag} max-wall core: {crit}  (n={int(rows[j, 0])})')
            print(f'{risc} {tag} wall max/mean {ms(rows[:, w].max()):.3f}/{ms(rows[:, w].mean()):.3f}'
                  f'  chunks total {int(rows[:, 0].sum())}'
                  + (f'  vis {int(rows[:, f.index("m")].sum())} rec {int(rows[:, f.index("pr")].sum())}'
                     if 'm' in f else ''))
