#!/usr/bin/env python3
"""t427: per-core pfwc zone length (max over RISCs), per view (nth pfwc zone on a core = view n).
Usage: pfwc_cores.py STITCHED.csv  -> mean/max over views per core, and the slowest-core list."""
import csv, sys
from collections import defaultdict
import numpy as np
st, d = {}, defaultdict(lambda: defaultdict(list))
with open(sys.argv[1]) as f:
    next(f); next(f)
    for p in csv.reader(f):
        if len(p) < 12 or p[10].strip() != "pfwc":
            continue
        k = ((int(p[1]), int(p[2])), p[3].strip())
        if p[11].strip() == "ZONE_START":
            st[k] = int(p[5])
        elif k in st:
            d[k[0]][k[1]].append(int(p[5]) - st.pop(k))
C = 1350e3
cores = sorted(d)
nv = min(len(v) for c in cores for v in d[c].values())
m = np.array([[max(d[c][r][i] for r in d[c]) for i in range(nv)] for c in cores]) / C  # core x view
print(f"cores={len(cores)} views={nv}")
print(f"per view: mean-core {m.mean(0).mean():.3f} max-core {m.max(0).mean():.3f} (max/mean {m.max(0).mean()/m.mean(0).mean():.3f})")
slow = np.argmax(m, 0)
cnt = defaultdict(int)
for s in slow: cnt[cores[s]] += 1
print("slowest core per view (count):", sorted(cnt.items(), key=lambda x: -x[1])[:6])
cm = m.mean(1)
o = np.argsort(-cm)
print("top-8 cores by mean pfwc ms:", [(cores[i], round(cm[i], 3)) for i in o[:8]])
print("bottom-4:", [(cores[i], round(cm[i], 3)) for i in o[-4:]])
for x in sorted({c[0] for c in cores}):
    v = [cm[i] for i, c in enumerate(cores) if c[0] == x]
    print(f"  col x={x:2d} n={len(v):2d} mean {np.mean(v):.3f} max {np.max(v):.3f}")
