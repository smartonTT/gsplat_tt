#!/usr/bin/env python3
"""Summarize GSPLAT_TT_MB_STATS DPRINT output (sub-tile waste counters).

Each blend-compute core prints one line per launch at kernel end:
  MBSTATS rec rec_live mb_kept mb_sat mb_disp pairops mb_useful px_live
          [rec_disp mb_dsat mb_waste pair_waste rec_waste]   (task #148)
Sums all lines, divides by the number of blend launches (--launches, e.g.
warmup + hero + 30 views) and prints per-frame ratios.

  python3 profiler/mb_stats_summary.py dprint.log --launches 31 [--blend-ms 63.6]
"""
import argparse
import re

FIELDS = ["rec", "rec_live", "mb_kept", "mb_sat", "mb_disp", "pairops", "mb_useful", "px_live",
          "rec_disp", "mb_dsat", "mb_waste", "pair_waste", "rec_waste"]
CLK_HZ = 1.35e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--launches", type=int, required=True)
    ap.add_argument("--entries", type=float, default=None,
                    help="mean num_entries (tile_assign pairs) per frame, if known")
    ap.add_argument("--cyc-mb", type=float, default=65.0, help="cycles per microblock dispatch")
    ap.add_argument("--cyc-rec", type=float, default=44.0, help="loop cycles per record")
    args = ap.parse_args()
    tot = dict.fromkeys(FIELDS, 0)
    n = 0
    pat = re.compile(r"MBSTATS((?:\s+\d+){8,13})")
    per_core = []
    with open(args.log, errors="replace") as f:
        for line in f:
            m = pat.search(line)
            if not m:
                continue
            vals = dict(zip(FIELDS, (int(v) for v in m.group(1).split())))
            for k, v in vals.items():
                tot[k] += v
            if vals.get("rec", 0):
                per_core.append(vals)
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
    if not tot["rec_disp"]:
        return
    d = max(pf["mb_disp"], 1)
    print("waste (task #148):")
    print(f"  dead records (no dispatch) / rec      {1 - pf['rec_disp'] / r:8.4f}")
    print(f"  ... of which mask 0 after cull        {1 - pf['rec_live'] / r:8.4f}")
    print(f"  dispatches into saturated mb / disp   {pf['mb_dsat'] / d:8.4f}")
    print(f"  dispatches, no live px / disp         {1 - pf['mb_useful'] / d:8.4f}")
    print(f"  wasted dispatches (union) / disp      {pf['mb_waste'] / d:8.4f}")
    print(f"  pair-ops fully wasted / pairops       {pf['pair_waste'] / max(pf['pairops'], 1):8.4f}")
    print(f"  dispatching recs fully wasted / disp  {pf['rec_waste'] / max(pf['rec_disp'], 1):8.4f}")
    # ms per frame: total cycles / (cores * clk); cores = active lines per launch.
    cores = max(len(per_core) / L, 1)
    def ms(cnt, cyc):
        return cnt * cyc / (cores * CLK_HZ) * 1e3
    dead = pf["rec"] - pf["rec_disp"]
    rows = [
        ("dead records x cyc_rec", ms(dead, args.cyc_rec)),
        ("no-live-px dispatches x cyc_mb", ms(pf["mb_disp"] - pf["mb_useful"], args.cyc_mb)),
        ("saturated dispatches x cyc_mb", ms(pf["mb_dsat"], args.cyc_mb)),
        ("union wasted dispatches x cyc_mb", ms(pf["mb_waste"], args.cyc_mb)),
    ]
    print(f"ms/frame (mean core, {cores:.1f} active cores/launch, {CLK_HZ / 1e9:.2f} GHz):")
    for name, v in rows:
        print(f"  {name:34s} {v:7.3f}")
    print(f"  {'TOTAL (dead rec + union disp)':34s} {rows[0][1] + rows[3][1]:7.3f}")


if __name__ == "__main__":
    main()
