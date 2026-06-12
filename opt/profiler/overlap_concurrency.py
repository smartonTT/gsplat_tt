#!/usr/bin/env python3
"""iter-140 Stage-0 decisive concurrency probe.

Reads a tt-metal profile_log_device.csv captured with the gated NCRISC<->SFPU
overlap probe ON (GSPLAT_TT_OVERLAP_PROBE=1). For the busiest reader core (max
tile_l1_cull_rd time) it measures, in WALL-CLOCK on that SAME core, how much the
NCRISC store window (ovlp_store) temporally intersects the SFPU compute zones
(tile_mb_mask). Same-core timestamps share the core clock, so a non-trivial
intersection is direct evidence the two engines run CONCURRENTLY (overlap), not
serialized.

Verdict logic:
  concurrent ~= min(SFPU_busy, store_window)  -> OVERLAP (PROCEED)
  concurrent ~= 0                              -> SERIAL  (FREEZE)

Usage: overlap_concurrency.py <csv> [n_views]
"""
import sys
from collections import defaultdict

FREQ_MHZ = 1350.0
CYC_MS = FREQ_MHZ * 1000.0
def ms(c): return c / CYC_MS

path = sys.argv[1]
nv = int(sys.argv[2]) if len(sys.argv) > 2 else 30

# Collect zone intervals per (cx,cy) core, splitting NCRISC store zones from
# SFPU (TRISC) compute zones. We use the global capture timeline (all views
# concatenated); intersections are computed on the merged interval sets so
# per-view boundaries don't matter for the aggregate concurrency fraction.
stack = defaultdict(list)                       # (cx,cy,risc,zone) -> [start...]
reader_cyc = defaultdict(int)                   # (cx,cy) -> tile_l1_cull_rd cyc
store_iv = defaultdict(list)                    # (cx,cy) -> [(s,e)] ovlp_store
sfpu_iv  = defaultdict(list)                    # (cx,cy) -> [(s,e)] tile_mb_mask
store_cyc = defaultdict(int)
sfpu_cyc  = defaultdict(int)

with open(path) as f:
    f.readline(); f.readline()
    for line in f:
        p = line.rstrip("\n").split(",")
        if len(p) < 12:
            continue
        cx, cy, risc = p[1], p[2], p[3]
        try:
            t = int(p[5])
        except ValueError:
            continue
        zone, typ = p[10], p[11]
        key = (cx, cy, risc, zone)
        if typ == "ZONE_START":
            stack[key].append(t)
        elif typ == "ZONE_END":
            st = stack[key]
            if not st:
                continue
            s = st.pop(); d = t - s
            if d < 0:
                continue
            core = (cx, cy)
            if zone == "tile_l1_cull_rd":
                reader_cyc[core] += d
            elif zone == "ovlp_store":
                store_iv[core].append((s, t)); store_cyc[core] += d
            elif zone == "tile_mb_mask":
                sfpu_iv[core].append((s, t)); sfpu_cyc[core] += d

def merge(ivs):
    if not ivs:
        return []
    ivs = sorted(ivs)
    out = [list(ivs[0])]
    for s, e in ivs[1:]:
        if s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out

def span(m):
    return sum(e - s for s, e in m)

def intersect(a, b):
    i = j = 0; tot = 0
    while i < len(a) and j < len(b):
        lo = max(a[i][0], b[j][0]); hi = min(a[i][1], b[j][1])
        if hi > lo:
            tot += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return tot

# busiest reader core
busiest = max(reader_cyc, key=reader_cyc.get)
print(f"n_views={nv}")
print(f"busiest reader core = {busiest}  reader(tile_l1_cull_rd) = {ms(reader_cyc[busiest])/nv:.2f} ms/view")

for core in [busiest]:
    st = merge(store_iv[core]); sf = merge(sfpu_iv[core])
    st_span = span(st); sf_span = span(sf)
    inter = intersect(st, sf)
    store_busy = store_cyc[core]; sfpu_busy = sfpu_cyc[core]
    print(f"\ncore {core}:")
    print(f"  NCRISC ovlp_store busy           = {ms(store_busy)/nv:8.2f} ms/view  (window span {ms(st_span)/nv:.2f})")
    print(f"  SFPU tile_mb_mask zone(busy+wait) = {ms(sfpu_busy)/nv:8.2f} ms/view  (window span {ms(sf_span)/nv:.2f})")
    print(f"  WALL-CLOCK intersection(store, SFPU-zone) = {ms(inter)/nv:8.2f} ms/view")
    if sf_span:
        print(f"  -> {100.0*inter/sf_span:5.1f}% of the SFPU-zone wall overlaps NCRISC stores")
    if st_span:
        print(f"  -> {100.0*inter/st_span:5.1f}% of the NCRISC store window overlaps SFPU-zone")
