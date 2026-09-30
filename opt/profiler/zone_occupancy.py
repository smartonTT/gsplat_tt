#!/usr/bin/env python3
"""Per-zone device time and per-core occupancy from a (stitched) profile_log_device.csv.

Frames are segmented with stitch_device_csv.segment_frames (warmup must already
be dropped, or pass --skip-first). For every zone, per view:
  sum_ms     : busy time summed over all cores
  cores      : distinct (core, RISC) that emit the zone
  mean_ms    : mean busy time per emitting core
  max_ms     : busiest-core time (per-view makespan proxy; averaged over views)
  balance    : mean_ms / max_ms (1.0 = perfectly balanced across cores)
  occ_%      : max_ms / device frame span (share of the frame the busiest core
               spends in this zone)
Device frame span = last - first device timestamp of the frame.

Usage: zone_occupancy.py <csv> [--skip-first] [--top N]
"""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stitch_device_csv import read_csv, segment_frames  # noqa: E402

CYC_MS = 1350.0 * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--skip-first", action="store_true", help="drop frame 0 (warmup)")
    ap.add_argument("--top", type=int, default=40)
    args = ap.parse_args()

    _, lines, ts = read_csv(args.csv)
    frame, n_frames, ncores = segment_frames(lines, ts)
    first = 1 if args.skip_first else 0
    nv = n_frames - first

    stack = defaultdict(list)
    # busy[zone][(frame, core)] = cycles
    busy = defaultdict(lambda: defaultdict(int))
    riscs = defaultdict(set)
    for ln, t, fr in zip(lines, ts, frame):
        if fr < first:
            continue
        p = ln.split(",", 12)
        key = (p[1], p[2], p[3], p[ZONE])
        if p[TYPE] == "ZONE_START":
            stack[key].append(t)
        elif p[TYPE] == "ZONE_END" and stack[key]:
            d = int(t) - stack[key].pop()
            if d >= 0:
                busy[p[ZONE]][(fr, (p[1], p[2], p[3]))] += d
                riscs[p[ZONE]].add(p[3])

    span = np.zeros(n_frames)
    for f in range(first, n_frames):
        sel = ts[frame == f]
        span[f] = (sel.max() - sel.min()) / CYC_MS
    mean_span = span[first:].mean()

    rows = []
    for zone, d in busy.items():
        per_frame = defaultdict(list)
        for (fr, core), c in d.items():
            per_frame[fr].append(c / CYC_MS)
        cores = len({core for (_, core) in d})
        sum_ms = sum(sum(v) for v in per_frame.values()) / nv
        max_ms = sum(max(v) for v in per_frame.values()) / nv
        mean_ms = sum(sum(v) / len(v) for v in per_frame.values()) / nv
        occ = np.mean([max(per_frame[f]) / span[f] for f in per_frame]) * 100
        rows.append((zone, sum_ms, cores, mean_ms, max_ms,
                     mean_ms / max_ms if max_ms else 0, occ, ",".join(sorted(riscs[zone]))))
    rows.sort(key=lambda r: -r[1])

    print(f"views={nv} cores/frame(anchor)={ncores} device_frame_span_ms "
          f"mean={mean_span:.2f} min={span[first:].min():.2f} max={span[first:].max():.2f}")
    print(f"{'zone':<20} {'sum_ms/v':>9} {'cores':>5} {'mean_ms':>8} {'max_ms':>8} "
          f"{'balance':>7} {'occ_%':>6}  riscs")
    print("-" * 80)
    for z, s, c, mn, mx, b, o, r in rows[:args.top]:
        print(f"{z:<20} {s:>9.2f} {c:>5} {mn:>8.3f} {mx:>8.3f} {b:>7.2f} {o:>6.1f}  {r}")


ZONE, TYPE = 10, 11

if __name__ == "__main__":
    main()
