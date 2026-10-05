#!/usr/bin/env python3
"""Per-program device launches and the all-core idle gaps between them.

zone_occupancy.py only sees named kernel zones, so a program without a
DeviceZoneScoped zone (the tile_assign scans, sort's count pass, ...) looks
like idle time. This script uses the firmware zones instead: every program
launch opens one *-FW zone on each RISC of each core it runs on. Per frame it
merges all FW intervals (any core, any RISC) into busy segments; a gap between
segments is time no core on the device was running a program. Each segment is
labelled with the kernel zones that start inside it, or with its core count
when it has none.

Usage: program_gaps.py <csv> [--skip-first] [--min-gap-us 20]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--skip-first", action="store_true", help="drop frame 0 (warmup)")
    ap.add_argument("--min-gap-us", type=float, default=20.0,
                    help="merge FW intervals separated by less than this")
    args = ap.parse_args()

    _, lines, ts = read_csv(args.csv)
    frame, n_frames, _ = segment_frames(lines, ts)
    first = 1 if args.skip_first else 0
    min_gap = args.min_gap_us * 1e-3 * CYC_MS

    stack = {}
    fw = defaultdict(list)       # frame -> [(start, end, core)]
    kz = defaultdict(list)       # frame -> [(start, zone)]
    for ln, t, fr in zip(lines, ts, frame):
        if fr < first:
            continue
        p = ln.split(",", 12)
        z = p[ZONE]
        key = (p[1], p[2], p[3], z)
        if p[TYPE] == "ZONE_START":
            stack[key] = int(t)
            if not z.endswith(("-FW", "-KERNEL")):
                kz[fr].append((int(t), z))
        elif p[TYPE] == "ZONE_END" and key in stack:
            t0 = stack.pop(key)
            if z.endswith("-FW"):
                fw[fr].append((t0, int(t), (p[1], p[2])))

    # Per frame: merged segments, then label them.
    per_frame = []
    for f in range(first, n_frames):
        iv = sorted(fw[f])
        segs = []  # [start, end, cores]
        for a, b, c in iv:
            if segs and a <= segs[-1][1] + min_gap:
                segs[-1][1] = max(segs[-1][1], b)
                segs[-1][2].add(c)
            else:
                segs.append([a, b, {c}])
        f0 = segs[0][0]
        zs = sorted(kz[f])
        out = []
        for a, b, cores in segs:
            names = []
            for t, z in zs:
                if a <= t <= b and z not in names:
                    names.append(z)
            out.append(((a - f0) / CYC_MS, (b - f0) / CYC_MS, len(cores),
                        "+".join(names) if names else f"<unnamed {len(cores)} cores>"))
        per_frame.append(out)

    # Align frames by segment index (same program sequence every view).
    nseg = {len(x) for x in per_frame}
    print(f"views={len(per_frame)} segments/view={sorted(nseg)}")
    n = min(nseg)
    labels = [per_frame[0][i][3] for i in range(n)]
    if len(nseg) != 1:
        print("WARNING: segment count differs across views; aligning by index")
    print(f"{'#':>2} {'start':>8} {'end':>8} {'busy':>7} {'gap_before':>10} {'cores':>5}  program")
    tot_gap = 0.0
    for i in range(n):
        s = np.mean([x[i][0] for x in per_frame])
        e = np.mean([x[i][1] for x in per_frame])
        c = int(np.median([x[i][2] for x in per_frame]))
        g = np.mean([x[i][0] - x[i - 1][1] for x in per_frame]) if i else 0.0
        tot_gap += g
        print(f"{i:>2} {s:>8.3f} {e:>8.3f} {e - s:>7.3f} {g:>10.3f} {c:>5}  {labels[i]}")
    print(f"total all-core idle between programs: {tot_gap:.3f} ms/view")


if __name__ == "__main__":
    main()
