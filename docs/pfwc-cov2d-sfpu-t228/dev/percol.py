#!/usr/bin/env python3
"""t228: pfwc wall per grid column and the slowest cores, from Tracy STEPCYC=1 captures
(pc_split.py's input, gzipped). Core wall = the slowest RISC of the core, mean over launches.
   percol.py prof-k0-dev.csv.gz prof-k1-dev.csv.gz"""
import gzip
import sys
from collections import defaultdict

import numpy as np

MHZ = 1350.0
FMT = {'pfwc_pc': 'n wall init wait xform recip depth means cov_cam a b c conic radx rady vis+pop'.split(),
       'pfwc_ws': 'n wall wait cls pfx rec opn tail rd'.split()}
ms = lambda c: c / MHZ / 1e3


def load(path):
    seq = defaultdict(list)  # (x, y, risc) -> launches
    with gzip.open(path, 'rt') as f:
        f.readline(); f.readline()
        for line in f:
            p = line.rstrip('\n').split(',')
            if len(p) <= 10 or p[10] not in FMT:
                continue
            v = int(p[6]); i, val = v >> 32, v & 0xffffffff
            if p[10] == 'pfwc_ws':
                i &= 15
            k = (int(p[1]), int(p[2]), p[3])
            if i == 0 or not seq[k]:
                seq[k].append({})
            seq[k][-1][i] = val
    per = defaultdict(dict)  # (x, y) -> risc -> field -> mean cycles
    for (x, y, r), L in seq.items():
        f = FMT['pfwc_ws' if r in ('BRISC', 'NCRISC') else 'pfwc_pc']
        a = np.array([[d.get(i, 0) for i in range(len(f))] for d in L], float)
        per[(x, y)][r] = dict(zip(f, a.mean(0)))
    return per


for path in sys.argv[1:]:
    per = load(path)
    wall = {c: max(v[r]['wall'] for r in v) for c, v in per.items()}
    w = np.array(list(wall.values()))
    print(f'== {path}: {len(per)} cores, core wall p50 {ms(np.percentile(w, 50)):.3f} '
          f'p90 {ms(np.percentile(w, 90)):.3f} max {ms(w.max()):.3f} ms')
    print('  x   wall mean   max | BRISC rec | NCRISC rec  rd')
    for x in sorted({c[0] for c in per}):
        cs = [c for c in per if c[0] == x]
        f = lambda r, k: ms(np.mean([per[c][r][k] for c in cs]))
        print(f'  {x:2d}  {ms(np.mean([wall[c] for c in cs])):.3f}  {ms(max(wall[c] for c in cs)):.3f} |'
              f'   {f("BRISC", "rec"):.3f}   |   {f("NCRISC", "rec"):.3f}   {f("NCRISC", "rd"):.3f}')
    print('  slowest cores: ' + ', '.join(f'{c} {ms(wall[c]):.3f}'
                                          for c in sorted(wall, key=wall.get, reverse=True)[:8]))
