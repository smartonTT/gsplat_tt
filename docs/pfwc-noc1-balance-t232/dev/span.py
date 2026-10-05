#!/usr/bin/env python3
"""Per-launch pfwc kernel span from STEPCYC records: end = timestamp of word 0, start = end - wall."""
import gzip, sys
from collections import defaultdict
import numpy as np
MHZ = 1350.0
for path in sys.argv[1:]:
    recs = defaultdict(list)  # (x,y,risc) -> list of (t_end, wall)
    cur = {}
    with gzip.open(path, 'rt') as f:
        f.readline(); f.readline()
        for line in f:
            p = line.split(',')
            if len(p) <= 10 or p[10] not in ('pfwc_ws', 'pfwc_pc'):
                continue
            v = int(p[6]); i = (v >> 32) & 15; val = v & 0xffffffff
            k = (int(p[1]), int(p[2]), p[3])
            if i == 0:
                cur[k] = [int(p[5]), None]
                recs[k].append(cur[k])
            elif i == 1:
                cur[k][1] = val
    nl = min(len(L) for L in recs.values())
    spans, maxw, p50 = [], [], []
    for l in range(nl):
        ends = [L[l][0] for L in recs.values()]
        starts = [L[l][0] - L[l][1] for L in recs.values()]
        walls = [L[l][1] for L in recs.values()]
        spans.append((max(ends) - min(starts)) / MHZ / 1e3)
        maxw.append(max(walls) / MHZ / 1e3)
        p50.append(np.median(walls) / MHZ / 1e3)
    print(f"{path.split('/')[-1]}: launches {nl}  span " + ' '.join(f'{s:.3f}' for s in spans) +
          f"  | mean span {np.mean(spans):.3f} max-wall {np.mean(maxw):.3f} med-wall {np.mean(p50):.3f}")
