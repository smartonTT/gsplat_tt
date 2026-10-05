#!/usr/bin/env python3
"""Per-core timeline of the merged mat+blend program (program 3): mat end vs blend
start per core, to tell whether blend waits on a global barrier after mat and how
much of the mat imbalance reaches the program window.

Usage: matblend_cores.py <csv> [--skip-first]
"""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stitch_device_csv import read_csv, segment_frames  # noqa: E402

CYC_MS = 1350.0 * 1000.0
ZONE, TYPE = 10, 11
Z = ("mat_cull_mask", "tile_blend_sfpu", "sort_subchunk_mat", "tile_blend_load")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--skip-first", action="store_true")
    a = ap.parse_args()
    _, lines, ts = read_csv(a.csv)
    frame, n_frames, _ = segment_frames(lines, ts)
    first = 1 if a.skip_first else 0
    stack = {}
    iv = defaultdict(list)  # (frame, core, risc, zone) -> [(s, e)]
    for ln, t, fr in zip(lines, ts, frame):
        if fr < first:
            continue
        p = ln.split(",", 12)
        z = p[ZONE]
        if z not in Z:
            continue
        r = p[3]
        if z in ("mat_cull_mask", "tile_blend_sfpu") and r != "TRISC_1":
            continue
        key = (fr, (p[1], p[2]), r, z)
        if p[TYPE] == "ZONE_START":
            stack[key] = int(t)
        elif p[TYPE] == "ZONE_END" and key in stack:
            iv[key].append((stack.pop(key), int(t)))
    rows = defaultdict(list)
    for f in range(first, n_frames):
        cores = {k[1] for k in iv if k[0] == f}
        per = {}
        for c in cores:
            g = lambda r, z: iv.get((f, c, r, z), [])
            m, b = g("TRISC_1", "mat_cull_mask"), g("TRISC_1", "tile_blend_sfpu")
            sm, bl = g("NCRISC", "sort_subchunk_mat"), g("NCRISC", "tile_blend_load")
            if not (m and b and sm and bl):
                continue
            per[c] = (min(x[0] for x in m), max(x[1] for x in m), min(x[0] for x in b),
                      max(x[1] for x in b), max(x[1] for x in sm), min(x[0] for x in bl))
        if not per:
            continue
        t0 = min(v[0] for v in per.values())
        v = np.array(list(per.values()), dtype=np.float64) - t0
        v /= CYC_MS
        mat_end, b_s, b_e = v[:, 1], v[:, 2], v[:, 3]
        rows["cores"].append(len(per))
        rows["mat_end_max"].append(mat_end.max())
        rows["mat_end_mean"].append(mat_end.mean())
        rows["blend_start_min"].append(b_s.min())
        rows["blend_start_mean"].append(b_s.mean())
        rows["blend_start_max"].append(b_s.max())
        rows["wait_mat_to_blend_mean"].append((b_s - mat_end).mean())
        rows["blend_len_mean"].append((b_e - b_s).mean())
        rows["blend_len_max"].append((b_e - b_s).max())
        rows["blend_end_max"].append(b_e.max())
        rows["blend_end_mean"].append(b_e.mean())
        rows["tail_idle_mean"].append((b_e.max() - b_e).mean())
        rows["ncrisc_mat_end_max"].append(v[:, 4].max())
        rows["blend_load_start_min"].append(v[:, 5].min())
        rows["corr(mat_end, blend_end)"].append(np.corrcoef(mat_end, b_e)[0, 1])
    print(f"views={len(rows['cores'])} (ms from first mat start on any core, mean over views)")
    for k, x in rows.items():
        print(f"{k:28s} {np.mean(x):8.3f}")


if __name__ == "__main__":
    main()
