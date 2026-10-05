#!/usr/bin/env python3
"""t232: per-core STEPCYC fields (ms per launch, mean over launches) of the slowest and median
pfwc cores, every RISC. cores.py prof-X-dev.csv.gz [n_slow]"""
import sys
import numpy as np
sys.argv, path = sys.argv[:1], sys.argv[1]
exec(open(__file__.replace('cores.py', 'percol.py')).read().split('\nfor path in')[0])
per = load(path)
wall = {c: max(v[r]['wall'] for r in v) for c, v in per.items()}
order = sorted(wall, key=wall.get, reverse=True)
pick = order[:6] + order[len(order) // 2 - 2:len(order) // 2 + 2]
for c in pick:
    print(f'{c} core wall {ms(wall[c]):.3f}')
    for r in sorted(per[c]):
        f = FMT['pfwc_ws' if r in ('BRISC', 'NCRISC') else 'pfwc_pc']
        print(f'   {r:7s} ' + ' '.join(f'{k}={ms(per[c][r][k]):.3f}' if k not in ('n', 'm') else f'{k}={per[c][r][k]:.0f}'
                                    for k in f))
