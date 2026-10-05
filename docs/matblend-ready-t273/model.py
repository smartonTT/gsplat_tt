#!/usr/bin/env python3
"""t273 kill gate: model of the mat->blend barrier removal (per-tile ready flags).

Inputs (all measured, none estimated except the cost fits named below):
  - per-tile record counts, 30 bicycle views: docs/mat-split-sort-model/out/dump.txt.gz
    (GSPLAT_TT_MAT_DUMP, line 1 = warm-up, md5 r82new - same tile lists as today)
  - per-tile blend MATH cycles, views 0/1: docs/fuse-matblend-t147/out/t147-tc.dprint
    (fit blend_us = a + b*rec, then scaled per view to today's measured blend length)
  - mat item costs: t144/t147 mover fits (whole tile NCRISC 5.5+0.197n, BRISC 16.1+0.187n,
    n>6144 -282+0.2226n; over-cap item 0.1083 n + 0.1267 l_sub), mat schedule = replay of
    build_mat_worklist (onelaunch, OL_MAT_SELECT off, 2 movers, m0_cap 6144)
  - today's timeline (t267 Tracy, 29 views): mat_end mean 2.512 / max 3.001,
    blend length mean 3.732.
Model: mat item end = cumulative cost on its mover slot (slot order = host order),
scaled per view so the busiest slot ends at the measured per-view mat max; tile ready =
end of its last item + TRISC cull lag. Blend: tiles in claim order (count desc), each
goes to the earliest-free core. barrier: all cores free at max mat end. flags: core free
at its own mat end, tile start >= ready (strict order: a core that claims an unready tile
waits for it).
"""
import argparse, gzip, re, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--dump", default="docs/mat-split-sort-model/out/dump.txt.gz")
ap.add_argument("--tc", default="docs/fuse-matblend-t147/out/t147-tc.dprint")
ap.add_argument("--cores", type=int, default=110)
ap.add_argument("--mat_max", type=float, default=3.001)   # ms, t267 per-view mean of max
ap.add_argument("--blend_len", type=float, default=3.732)  # ms, t267 per-core mean
ap.add_argument("--lag", type=float, default=0.02)        # ms TRISC cull after last mover write
ap.add_argument("--order", default="desc", choices=["desc", "ready"])
ap.add_argument("--cal_mean", type=float, default=0.0)  # >0: compress mat ends so mean == this (t267: 2.512)
ap.add_argument("-v", action="store_true")
a = ap.parse_args()

BF, OVCAP, M0CAP = 8192, 16384, 6144

# ---- blend per-tile cost fit from t147 ----
rec, cyc = [], []
for ln in open(a.tc, errors="replace"):
    m = re.search(r"TC\s+(\d+)\s+(\d+)\s+(\d+)\s*$", ln)
    if m:
        rec.append(int(m.group(1))); cyc.append(int(m.group(3)) / 1350.0)
rec, us = np.array(rec, float), np.array(cyc, float)
A = np.vstack([np.ones_like(rec), rec, np.maximum(rec - 8192, 0)]).T
coef, *_ = np.linalg.lstsq(A, us, rcond=None)
def blend_us(n):
    return coef[0] + coef[1] * n + coef[2] * max(n - 8192, 0)

def mat_items(counts):
    items = []
    for t, c in counts.items():
        if c <= OVCAP:
            if c <= M0CAP:
                items.append((t, c, False, None))
            else:
                items.append((t, c, True, None))
            continue
        nsc = (c + BF - 1) // BF
        for sc in range(nsc):
            l = min(BF, c - sc * BF)
            items.append((t, c + l, True, (c, l)))
    items.sort(key=lambda x: -x[1])
    return items

def item_us(it, mover):
    t, cost, big, ov = it
    if ov is not None:
        n, l = ov
        return 0.1083 * n + 0.1267 * l
    n = cost
    if n > M0CAP:
        return -282 + 0.2226 * n
    return 5.5 + 0.197 * n if mover == 0 else 16.1 + 0.187 * n

def model(counts):
    C = a.cores; slots = 2 * C
    load = [0] * slots; per = [[] for _ in range(slots)]
    for it in mat_items(counts):
        step = 2 if it[2] else 1
        c = 0
        for k in range(step, slots, step):
            if load[k] < load[c]: c = k
        per[c].append(it); load[c] += it[1]
    ready = {}; slot_end = np.zeros(slots)
    raw = []
    for s in range(slots):
        tt = 0.0
        for it in per[s]:
            tt += item_us(it, s & 1)
            raw.append((it[0], s, tt))
        slot_end[s] = tt
    sc = a.mat_max * 1000.0 / slot_end.max()
    for t, s, tt in raw:
        ready[t] = max(ready.get(t, 0.0), tt * sc / 1000.0 + a.lag)
    core_end = np.maximum(slot_end[0::2], slot_end[1::2]) * sc / 1000.0 + a.lag
    if a.cal_mean > 0:
        mx = core_end.max(); k = (mx - a.cal_mean) / (mx - core_end.mean())
        core_end = mx - (mx - core_end) * k
        ready = {t: mx - max(mx - r, 0.0) * k for t, r in ready.items()}
    tiles = sorted(counts, key=lambda t: (-counts[t], -t))
    b = np.array([blend_us(counts[t]) for t in tiles])
    b *= a.blend_len * C / b.sum()    # today's per-core mean blend length
    if a.order == "ready":
        idx = sorted(range(len(tiles)), key=lambda i: (ready[tiles[i]] > core_end.mean(), i))
        tiles = [tiles[i] for i in idx]; b = b[idx]
    def sim(free, gate):
        free = list(free); end = 0.0; wait = 0.0
        for t, d in zip(tiles, b):
            c = int(np.argmin(free))
            s = max(free[c], ready[t]) if gate else free[c]
            wait += s - free[c]
            free[c] = s + d; end = max(end, free[c])
        return end, wait
    bar, _ = sim([core_end.max()] * C, False)
    flg, w = sim(core_end, True)
    nog, _ = sim(core_end, False)
    return dict(mat_mean=core_end.mean(), mat_max=core_end.max(), barrier=bar, flags=flg,
                saving=bar - flg, ungated=bar - nog, ready_wait=w / C,
                big_ready=max(ready[t] for t in tiles[:5]), bmax=b.max())

lines = gzip.open(a.dump, "rt").read().splitlines()[1:]
rows = []
for v, ln in enumerate(lines):
    counts = {int(k): int(c) for k, c in (p.split(":") for p in ln.split()[3:])}
    r = model(counts); rows.append(r)
    if a.v:
        print(v, " ".join(f"{k}={x:.3f}" for k, x in r.items()))
print(f"blend fit us = {coef[0]:.1f} + {coef[1]:.4f} n + {coef[2]:.4f} max(n-8192,0)  (t147 views 0/1)")
print(f"views={len(rows)} order={a.order} lag={a.lag}")
for k in rows[0]:
    print(f"  {k:12s} {np.mean([r[k] for r in rows]):.3f}  (min {min(r[k] for r in rows):.3f} max {max(r[k] for r in rows):.3f})")
