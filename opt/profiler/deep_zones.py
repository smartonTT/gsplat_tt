#!/usr/bin/env python3
"""Deep per-zone / per-RISC breakdown from a tt-metal profile_log_device.csv.

Reports, for the whole capture (all views) and per-view:
  * every zone: summed busy cycles across cores, per-view makespan (busiest core),
    the RISC(s) that emit it, and pairs/view
  * per-RISC-TYPE occupancy: busy cycles vs the core's wall span (busy/wall =
    how compute-bound that RISC is; low => stalled on NoC/CB, high => ALU/issue bound)
  * a coarse stage grouping by zone-name substring

Usage: deep_zones.py <csv> [n_views]
"""
import sys
from collections import defaultdict

FREQ_MHZ = 1350.0
CYC_MS = FREQ_MHZ * 1000.0

path = sys.argv[1]
nv = int(sys.argv[2]) if len(sys.argv) > 2 else 30

stack = defaultdict(list)
zone_cyc = defaultdict(int)          # zone -> summed busy cyc (all cores)
zone_cnt = defaultdict(int)
zone_core = defaultdict(lambda: defaultdict(int))  # zone -> (cx,cy,risc) -> cyc
zone_risc = defaultdict(set)
risc_busy = defaultdict(int)         # risc-type -> summed busy cyc
core_first = {}                      # (cx,cy,risc) -> min start
core_last = {}                       # (cx,cy,risc) -> max end
risc_types = set()

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
        zone = p[10]; typ = p[11]
        risc_types.add(risc)
        ck = (cx, cy, risc)
        key = (cx, cy, risc, zone)
        if typ == "ZONE_START":
            stack[key].append(t)
            if ck not in core_first or t < core_first[ck]:
                core_first[ck] = t
        elif typ == "ZONE_END":
            st = stack[key]
            if not st:
                continue
            d = t - st.pop()
            if d < 0:
                continue
            zone_cyc[zone] += d
            zone_cnt[zone] += 1
            zone_core[zone][ck] += d
            zone_risc[zone].add(risc)
            risc_busy[risc] += d
            if ck not in core_last or t > core_last[ck]:
                core_last[ck] = t

def ms(c): return c / CYC_MS

print(f"n_views={nv}  RISC types={sorted(risc_types)}")
print()
print("=== ZONES by per-view MAKESPAN (busiest core / nviews) ===")
print(f"{'zone':38} {'tot_ms':>9} {'pv_tot':>8} {'pv_make':>8} {'pairs/v':>8}  riscs")
rows = []
for z, c in zone_cyc.items():
    make = max(zone_core[z].values())
    rows.append((make, z, c))
for make, z, c in sorted(rows, reverse=True)[:30]:
    print(f"{z[:38]:38} {ms(c):9.1f} {ms(c)/nv:8.2f} {ms(make)/nv:8.2f} "
          f"{zone_cnt[z]/nv:8.1f}  {','.join(sorted(zone_risc[z]))}")

print()
print("=== per-RISC-TYPE occupancy (busy vs wall span) ===")
# wall span per risc-type = max over its cores of (last-first); busy = summed zone cyc
risc_wall = defaultdict(int)
for (cx, cy, risc), first in core_first.items():
    last = core_last.get((cx, cy, risc), first)
    span = last - first
    if span > risc_wall[risc]:
        risc_wall[risc] = span
print(f"{'risc':10} {'busy_pv_ms':>11} {'wallspan_pv_ms':>15} {'occupancy':>10}")
for r in sorted(risc_types):
    busy = risc_busy[r]
    wall = risc_wall[r]
    occ = (busy / wall) if wall else 0  # busy is summed across that risc's cores; wall is one core span
    # normalize busy to per-core average for occupancy: divide by #cores of that risc
    ncore = len({(cx, cy) for (cx, cy, rr) in core_first if rr == r})
    busy_per_core = busy / ncore if ncore else 0
    occ = (busy_per_core / wall) if wall else 0
    print(f"{r:10} {ms(busy)/nv:11.2f} {ms(wall)/nv:15.2f} {occ*100:9.1f}%  (cores={ncore})")

print()
print("=== coarse STAGE grouping (by zone substring) ===")
stage_keys = {
    "proj": ["proj", "project"],
    "ta/tile": ["tile_assign", "ta_", "scatter", "bucket"],
    "sort": ["sort", "radix", "bin", "subchunk", "materialize", "mat_"],
    "cull": ["cull", "mb_mask", "mb_cov", "preconic"],
    "blend": ["blend", "alpha"],
}
stage_make = defaultdict(int)
stage_tot = defaultdict(int)
assigned = {}
for z in zone_cyc:
    zl = z.lower()
    for stg, subs in stage_keys.items():
        if any(s in zl for s in subs):
            assigned[z] = stg
            break
    else:
        assigned[z] = "other"
for z, c in zone_cyc.items():
    stage_tot[assigned[z]] += c
    stage_make[assigned[z]] += max(zone_core[z].values())
print(f"{'stage':10} {'pv_make_sum_ms':>15} {'pv_tot_ms':>11}")
for stg in ["proj", "ta/tile", "sort", "cull", "blend", "other"]:
    print(f"{stg:10} {ms(stage_make[stg])/nv:15.2f} {ms(stage_tot[stg])/nv:11.2f}")
