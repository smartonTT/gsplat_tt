#!/usr/bin/env python3
"""t232 (from t228 percol.py): pfwc wall per physical grid column and the slowest cores, from
Tracy STEPCYC=1 captures (pc_split.py's input, gzipped). Core wall = the slowest RISC of the
core, mean over launches. rd = time in the reader's polls (on NCRISC, or BRISC on the
GSPLAT_TT_PFWC_RD_BRISC columns).
   percol.py prof-p0-dev.csv.gz prof-pf-dev.csv.gz ..."""
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
    print('  x | wall mean   max | BRISC wall  rec   rd | NCRISC wall  rec   rd | TRISC wall')
    for x in sorted({c[0] for c in per}):
        cs = [c for c in per if c[0] == x]
        f = lambda r, k: ms(np.mean([per[c][r][k] for c in cs if r in per[c]] or [0]))
        tw = ms(np.mean([max(per[c][r]['wall'] for r in per[c] if r.startswith('TRISC')) for c in cs]))
        print(f' {x:2d} | {ms(np.mean([wall[c] for c in cs])):.3f}  {ms(max(wall[c] for c in cs)):.3f} |'
              f'  {f("BRISC", "wall"):.3f}  {f("BRISC", "rec"):.3f} {f("BRISC", "rd"):.3f} |'
              f'   {f("NCRISC", "wall"):.3f}  {f("NCRISC", "rec"):.3f} {f("NCRISC", "rd"):.3f} |  {tw:.3f}')
    print('  slowest cores: ' + ', '.join(f'{c} {ms(wall[c]):.3f}'
                                          for c in sorted(wall, key=wall.get, reverse=True)[:8]))
