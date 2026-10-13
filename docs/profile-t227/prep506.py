#!/usr/bin/env python3
"""t506: join per-chunk Tracy device CSVs into one CSV of the measured b2b views only.
Each chunk process renders 11 views (warm hero + 5-view check pass + 5-view measured pass) =
44 BRISC-FW programs per core. Per core, every row before the end of the core's
(44 - 4*K)-th BRISC-FW is dropped (K = --keep, default 5), so only the last K views remain
(with xview, pfwc N+1 starts on a core only after blend N ends there). An END left without
its START is ignored by the readers. Chunks get a time shift so they never overlap.
Usage: prep506.py -o OUT.csv.gz [--keep 5] CHUNK.csv.gz ...
"""
import argparse, csv, gzip, os
from collections import defaultdict
NP = len(os.environ.get("ANA_PROGS", "pfwc,k2,sort,mat+blend").split(","))  # t595

ap = argparse.ArgumentParser()
ap.add_argument("-o", required=True)
ap.add_argument("--keep", type=int, default=5)
ap.add_argument("chunks", nargs="+")
a = ap.parse_args()
out = gzip.open(a.o, "wt", newline="")
w = csv.writer(out)
base = 0
for i, path in enumerate(a.chunks):
    with gzip.open(path, "rt") as f:
        h1, h2 = next(f), next(f)
        rows = [p for p in csv.reader(f) if len(p) >= 12]
    if i == 0:
        out.write(h1); out.write(h2)
    fw = defaultdict(list)          # core -> BRISC-FW (start, end)
    st = {}
    for p in rows:
        if p[3].strip() == "BRISC" and p[10].strip() == "BRISC-FW":
            c, t = (p[1], p[2]), int(p[5])
            if p[11].strip() == "ZONE_START":
                st[c] = t
            elif c in st:
                fw[c].append((st.pop(c), t))
    cut = {}
    for c, L in fw.items():
        L.sort(); n = len(L)
        assert n % NP == 0 and n >= NP * a.keep, (path, c, n)
        cut[c] = L[n - NP * a.keep - 1][1] if n > NP * a.keep else -1
    kept = [p for p in rows if (p[1], p[2]) in cut and int(p[5]) > cut[(p[1], p[2])]]
    t0 = min(int(p[5]) for p in kept)
    sh = base - t0 + (1350 * 1000 * 1000 if i else 0)
    for p in kept:
        p[5] = str(int(p[5]) + sh)
        w.writerow(p)
    base = max(int(p[5]) for p in kept)
    print(f"{path}: cores {len(cut)} programs/core {n} kept rows {len(kept)}")
out.close()
