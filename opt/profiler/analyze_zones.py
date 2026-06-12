#!/usr/bin/env python3
"""Aggregate render_clean DEVICE zones from a tt-metal profile_log_device.csv.

Pairs ZONE_START/ZONE_END per (core, RISC, zone) with a stack (handles nesting/
recursion), sums durations, and reports per-zone:
  - total_ms  : summed device-busy time across ALL cores + ALL views
  - per_view  : total_ms / n_views
  - max_core  : max over cores of that core's summed zone time (parallel-makespan
                proxy: all cores run concurrently, so the busiest core bounds the
                stage wall time)
  - per_view_makespan : max_core / n_views
  - riscs     : which RISC(s) emit the zone

Usage: analyze_zones.py <profile_log_device.csv> [n_views]
"""
import sys
from collections import defaultdict

CHIP_FREQ_MHZ = 1350.0          # from CSV header line 1
CYC_PER_MS = CHIP_FREQ_MHZ * 1000.0  # cycles per ms


def main():
    path = sys.argv[1]
    n_views = int(sys.argv[2]) if len(sys.argv) > 2 else 30

    # stack[(cx,cy,risc,zone)] = [start_cycle, ...]
    stack = defaultdict(list)
    total_cyc = defaultdict(int)        # zone -> summed cycles (all cores)
    count = defaultdict(int)            # zone -> #pairs
    per_core_cyc = defaultdict(lambda: defaultdict(int))  # zone -> (core,risc) -> cyc
    riscs = defaultdict(set)            # zone -> {risc}

    with open(path) as f:
        f.readline()  # ARCH header
        f.readline()  # column header
        for line in f:
            parts = line.rstrip("\n").split(",")
            if len(parts) < 12:
                continue
            cx, cy, risc = parts[1], parts[2], parts[3]
            try:
                t = int(parts[5])
            except ValueError:
                continue
            zone = parts[10]
            typ = parts[11]
            key = (cx, cy, risc, zone)
            if typ == "ZONE_START":
                stack[key].append(t)
            elif typ == "ZONE_END":
                st = stack[key]
                if not st:
                    continue
                t0 = st.pop()
                dur = t - t0
                if dur < 0:
                    continue
                total_cyc[zone] += dur
                count[zone] += 1
                per_core_cyc[zone][(cx, cy, risc)] += dur
                riscs[zone].add(risc)

    rows = []
    for zone, tc in total_cyc.items():
        maxcore = max(per_core_cyc[zone].values()) if per_core_cyc[zone] else 0
        rows.append((
            zone,
            tc / CYC_PER_MS,
            tc / CYC_PER_MS / n_views,
            maxcore / CYC_PER_MS,
            maxcore / CYC_PER_MS / n_views,
            count[zone],
            ",".join(sorted(riscs[zone])),
        ))
    rows.sort(key=lambda r: -r[1])

    print(f"n_views={n_views}  freq={CHIP_FREQ_MHZ}MHz")
    print(f"{'zone':<22} {'total_ms':>11} {'per_view':>9} "
          f"{'maxcore_ms':>11} {'pv_makespan':>11} {'pairs':>8}  riscs")
    print("-" * 96)
    for (zone, tot, pv, mc, pvm, n, rs) in rows:
        print(f"{zone:<22} {tot:>11.1f} {pv:>9.3f} {mc:>11.1f} {pvm:>11.3f} {n:>8}  {rs}")


if __name__ == "__main__":
    main()
