#!/usr/bin/env python3
"""t232: per-core grids (rows y, physical columns x) of pfwc STEPCYC fields from Tracy captures."""
import gzip, sys
from collections import defaultdict
import numpy as np
MHZ=1350.0
FMT = {'pfwc_pc': 'n wall init wait xform recip depth means cov_cam a b c conic radx rady vis+pop'.split(),
       'pfwc_ws': 'n wall wait cls pfx rec opn tail rd fl'.split()}
def load(path):
    seq = defaultdict(list)
    with gzip.open(path, 'rt') as f:
        f.readline(); f.readline()
        for line in f:
            p = line.rstrip('\n').split(',')
            if len(p) <= 10 or p[10] not in FMT: continue
            v = int(p[6]); i, val = v >> 32, v & 0xffffffff
            if p[10] == 'pfwc_ws': i &= 15
            k = (int(p[1]), int(p[2]), p[3])
            if i == 0 or not seq[k]: seq[k].append({})
            seq[k][-1][i] = val
    per = defaultdict(dict)
    for (x, y, r), L in seq.items():
        f = FMT['pfwc_ws' if r in ('BRISC', 'NCRISC') else 'pfwc_pc']
        a = np.array([[d.get(i, 0) for i in range(len(f))] for d in L], float)
        per[(x, y)][r] = dict(zip(f, a.mean(0)))
    return per
for path in sys.argv[1:]:
    per = load(path)
    xs = sorted({c[0] for c in per}); ys = sorted({c[1] for c in per})
    for title, fn in [('core wall', lambda v: max(v[r]['wall'] for r in v)),
                      ('NCRISC rec', lambda v: v['NCRISC']['rec']),
                      ('BRISC rec', lambda v: v['BRISC']['rec']),
                      ('NCRISC wait', lambda v: v['NCRISC']['wait']),
                      ('BRISC fl', lambda v: v['BRISC'].get('fl', 0)),
                      ('NCRISC fl', lambda v: v['NCRISC'].get('fl', 0))]:
        print(f'== {path.split("/")[-1]} {title} (ms), rows y, cols x')
        print('  y\\x ' + ' '.join(f'{x:6d}' for x in xs))
        for y in ys:
            print(f'  {y:3d} ' + ' '.join(f'{fn(per[(x,y)])/MHZ/1e3:6.3f}' if (x,y) in per else '     -' for x in xs))
