#!/usr/bin/env python3
"""Per-frame wall window of named zone groups (first start .. last end over all cores).

A program's window is where its zones run, so for chained programs that
program_gaps.py merges into one busy segment (no idle between them), this splits
the segment per program. Also prints each zone's mean per-core busy time.

Usage: zone_windows.py <csv> [--skip-first] [--group name=zoneA+zoneB ...]
Default groups: the final chained segment (sort_subchunk_mat, SFPU cull, blend).
"""
import argparse
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stitch_device_csv import read_csv, segment_frames  # noqa: E402

CYC_MS = 1350.0 * 1000.0
ZONE, TYPE = 10, 11
DEFAULT_GROUPS = [
    "subchunk_mat=sort_subchunk_mat",
    "cull=tile_mb_mask+tile_l1_cull_rd",
    "blend=tile_blend_sfpu+tile_blend_load+wr_pack",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--skip-first", action="store_true")
    ap.add_argument("--group", action="append", default=None)
    args = ap.parse_args()
    groups = []
    for g in (args.group or DEFAULT_GROUPS):
        name, zs = g.split("=", 1)
        groups.append((name, set(zs.split("+"))))

    _, lines, ts = read_csv(args.csv)
    frame, n_frames, _ = segment_frames(lines, ts)
    first = 1 if args.skip_first else 0
    stack = {}
    win = defaultdict(lambda: [None, None])    # (frame, group) -> [start, end]
    busy = defaultdict(int)                    # zone -> summed cycles
    cores = defaultdict(set)                   # zone -> {(x, y, risc)}
    for ln, t, fr in zip(lines, ts, frame):
        if fr < first:
            continue
        p = ln.split(",", 12)
        z = p[ZONE]
        key = (p[1], p[2], p[3], z)
        if p[TYPE] == "ZONE_START":
            stack[key] = int(t)
        elif p[TYPE] == "ZONE_END" and key in stack:
            t0, t1 = stack.pop(key), int(t)
            busy[z] += t1 - t0
            cores[z].add((p[1], p[2], p[3]))
            for name, zs in groups:
                if z in zs:
                    w = win[(fr, name)]
                    w[0] = t0 if w[0] is None else min(w[0], t0)
                    w[1] = t1 if w[1] is None else max(w[1], t1)
    nf = n_frames - first
    print(f"views={nf}")
    print(f"{'group':<14} {'window_ms':>9} {'gap_before':>10}")
    prev_end = None
    for name, _ in groups:
        ws = [win[(f, name)] for f in range(first, n_frames) if win[(f, name)][0] is not None]
        if not ws:
            print(f"{name:<14} {'-':>9}")
            continue
        w = sum(b - a for a, b in ws) / len(ws) / CYC_MS
        gap = ""
        if prev_end is not None:
            gs = [win[(f, name)][0] - prev_end[f] for f in range(first, n_frames)
                  if f in prev_end and win[(f, name)][0] is not None]
            gap = f"{sum(gs) / len(gs) / CYC_MS:10.3f}" if gs else ""
        prev_end = {f: win[(f, name)][1] for f in range(first, n_frames)
                    if win[(f, name)][1] is not None}
        print(f"{name:<14} {w:9.3f} {gap:>10}")
    print(f"\n{'zone':<20} {'mean_per_core_ms/view':>22} {'cores':>6}")
    for z in sorted(busy, key=lambda k: -busy[k]):
        if z.endswith(("-FW", "-KERNEL")):
            continue
        print(f"{z:<20} {busy[z] / len(cores[z]) / nf / CYC_MS:22.3f} {len(cores[z]):6}")


if __name__ == "__main__":
    main()
