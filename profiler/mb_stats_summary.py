#!/usr/bin/env python3
"""Summarize GSPLAT_TT_MB_STATS DPRINT output (sub-tile waste counters).

Each blend-compute core prints one line per launch at kernel end:
  MBSTATS rec rec_live mb_kept mb_sat mb_disp pairops mb_useful px_live
Sums all lines, divides by the number of blend launches (--launches, e.g.
warmup + hero + 30 views) and prints per-frame ratios.

  python3 profiler/mb_stats_summary.py dprint.log --launches 31 [--blend-ms 63.6]
"""
import argparse
import re

FIELDS = ["rec", "rec_live", "mb_kept", "mb_sat", "mb_disp", "pairops", "mb_useful", "px_live"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--launches", type=int, required=True)
    ap.add_argument("--entries", type=float, default=None,
                    help="mean num_entries (tile_assign pairs) per frame, if known")
    args = ap.parse_args()
    tot = dict.fromkeys(FIELDS, 0)
    n = 0
    pat = re.compile(r"MBSTATS((?:\s+\d+){8})")
    with open(args.log, errors="replace") as f:
        for line in f:
            m = pat.search(line)
            if not m:
                continue
            for k, v in zip(FIELDS, m.group(1).split()):
                tot[k] += int(v)
            n += 1
    L = args.launches
    pf = {k: v / L for k, v in tot.items()}
    print(f"lines={n} launches={L} lines/launch={n / L:.1f}")
    for k in FIELDS:
        print(f"  {k:10s} {pf[k]:16.0f} /frame")
    r = pf["rec"] or 1
    print("ratios:")
    if args.entries:
        print(f"  rec / tile_assign entries       {pf['rec'] / args.entries:8.4f}")
    print(f"  rec_live / rec (pair survives)  {pf['rec_live'] / r:8.4f}")
    print(f"  mb_kept / (32*rec)              {pf['mb_kept'] / (32 * r):8.4f}")
    print(f"  mb_disp / (32*rec)              {pf['mb_disp'] / (32 * r):8.4f}")
    print(f"  mb_useful / mb_disp             {pf['mb_useful'] / max(pf['mb_disp'], 1):8.4f}")
    print(f"  px_live / (32*mb_disp)          {pf['px_live'] / max(32 * pf['mb_disp'], 1):8.4f}")
    print(f"  px_live / (1024*rec)            {pf['px_live'] / (1024 * r):8.4f}")
    print(f"  mb_disp / pairops (ILP pairing) {pf['mb_disp'] / max(pf['pairops'], 1):8.4f}")
    print(f"  mean mb_disp per live rec       {pf['mb_disp'] / max(pf['rec_live'], 1):8.2f}")


if __name__ == "__main__":
    main()
