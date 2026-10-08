#!/usr/bin/env python3
"""t384: per-(core, RISC) zone durations from a Tracy device CSV (mean over frames, ms).
  zones.py dev.csv.gz zone [zone ...]
Prints, per zone and RISC: movers, mean / min / max duration, and the mean start / end
relative to that frame's earliest start of the same zone (ms)."""
import gzip, sys
from collections import defaultdict
import statistics as st
MHZ = 1350.0
want = set(sys.argv[2:])
op = gzip.open if sys.argv[1].endswith('.gz') else open
starts = defaultdict(list)  # (zone, core, risc) -> [t]
ends = defaultdict(list)
with op(sys.argv[1], 'rt') as f:
    f.readline(); f.readline()
    for line in f:
        p = line.split(',')
        if len(p) < 12 or p[10] not in want:
            continue
        k = (p[10], p[1] + '-' + p[2], p[3])
        (starts if p[11] == 'ZONE_START' else ends)[k].append(int(p[5]))
for z in sys.argv[2:]:
    for risc in sorted({k[2] for k in starts if k[0] == z}):
        keys = [k for k in starts if k[0] == z and k[2] == risc]
        n = min(min(len(starts[k]), len(ends[k])) for k in keys)
        durs = {k: [(ends[k][i] - starts[k][i]) / MHZ / 1e3 for i in range(n)] for k in keys}
        # frame-relative start/end per mover
        rel_s, rel_e = defaultdict(list), defaultdict(list)
        for i in range(n):
            t0 = min(starts[k][i] for k in keys)
            for k in keys:
                rel_s[k].append((starts[k][i] - t0) / MHZ / 1e3)
                rel_e[k].append((ends[k][i] - t0) / MHZ / 1e3)
        md = [st.mean(v) for v in durs.values()]
        me = [st.mean(v) for v in rel_e.values()]
        ms_ = [st.mean(v) for v in rel_s.values()]
        print(f'{z:16s} {risc:7s} movers={len(keys)} frames={n} dur mean {st.mean(md):.3f} '
              f'min {min(md):.3f} max {max(md):.3f} | start mean {st.mean(ms_):.3f} max {max(ms_):.3f} '
              f'| end mean {st.mean(me):.3f} max {max(me):.3f}')
