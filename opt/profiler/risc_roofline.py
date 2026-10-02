#!/usr/bin/env python3
"""Per-program, per-RISC kernel busy time and the overlap bound (task #115).

Programs are found as in program_gaps.py: per frame, all *-FW intervals (any
core, any RISC) merged into busy segments. Inside each program every core's
*-KERNEL zone gives each RISC's busy time. Per program it prints the window and,
per RISC, the mean core's kernel time, the busiest core's time (makespan) and
the busiest core's share of the window. Named kernel zones that start in the
program get the same mean/max columns.

Per frame it also sums each core's kernel time per RISC over all programs. The
busiest core's sum is what that RISC needs if every stage overlapped perfectly
(a lower bound for a fused or persistent pipeline); the mean core's sum is the
bound with perfect load balance on top.

Usage: risc_roofline.py <csv> [--skip-first] [--min-gap-us 1]
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
RISCS = ["BRISC", "NCRISC", "TRISC_0", "TRISC_1", "TRISC_2"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--skip-first", action="store_true", help="drop frame 0 (warmup)")
    ap.add_argument("--min-gap-us", type=float, default=1.0)
    args = ap.parse_args()

    _, lines, ts = read_csv(args.csv)
    frame, n_frames, _ = segment_frames(lines, ts)
    first = 1 if args.skip_first else 0
    min_gap = args.min_gap_us * 1e-3 * CYC_MS

    stack = {}
    fw = defaultdict(list)    # frame -> [(t0, t1, core)]
    kern = defaultdict(list)  # frame -> [(t0, t1, core, risc)]
    named = defaultdict(list)  # frame -> [(t0, t1, core, risc, zone)]
    for ln, t, fr in zip(lines, ts, frame):
        if fr < first:
            continue
        p = ln.split(",", 12)
        z = p[ZONE]
        key = (p[1], p[2], p[3], z)
        if p[TYPE] == "ZONE_START":
            stack[key] = int(t)
        elif p[TYPE] == "ZONE_END" and key in stack:
            t0 = stack.pop(key)
            core = (p[1], p[2])
            if z.endswith("-FW"):
                fw[fr].append((t0, int(t), core))
            elif z.endswith("-KERNEL"):
                kern[fr].append((t0, int(t), core, p[3]))
            else:
                named[fr].append((t0, int(t), core, p[3], z))

    progs = []       # per frame: list of dicts
    frame_tot = []   # per frame: {risc: (mean, max)} of per-core sums
    spans = []
    for f in range(first, n_frames):
        segs = []
        for a, b, c in sorted(fw[f]):
            if segs and a <= segs[-1][1] + min_gap:
                segs[-1][1] = max(segs[-1][1], b)
            else:
                segs.append([a, b])
        if not segs:
            continue
        f0 = segs[0][0]
        spans.append((segs[-1][1] - f0) / CYC_MS)
        starts = np.array([s[0] for s in segs], dtype=np.int64)

        def seg_of(t):
            return int(np.searchsorted(starts, t, side="right")) - 1

        busy = [defaultdict(float) for _ in segs]       # (core, risc) -> ms
        zb = [defaultdict(float) for _ in segs]         # (zone, core, risc) -> ms
        tot = defaultdict(float)                        # (core, risc) -> ms
        for a, b, core, risc in kern[f]:
            d = (b - a) / CYC_MS
            busy[seg_of(a)][(core, risc)] += d
            tot[(core, risc)] += d
        for a, b, core, risc, z in named[f]:
            zb[seg_of(a)][(z, core, risc)] += (b - a) / CYC_MS
        out = []
        for i, (a, b) in enumerate(segs):
            r = {"start": (a - f0) / CYC_MS, "end": (b - f0) / CYC_MS,
                 "gap": (a - segs[i - 1][1]) / CYC_MS if i else 0.0}
            for risc in RISCS:
                v = [ms for (c, rr), ms in busy[i].items() if rr == risc]
                r[risc] = (float(np.mean(v)), float(np.max(v)), len(v)) if v else (0.0, 0.0, 0)
            zs = defaultdict(list)
            for (z, c, rr), ms in zb[i].items():
                zs[(z, rr)].append(ms)
            r["zones"] = {k: (float(np.mean(v)), float(np.max(v)), len(v)) for k, v in zs.items()}
            out.append(r)
        progs.append(out)
        ft = {}
        for risc in RISCS:
            v = [ms for (c, rr), ms in tot.items() if rr == risc]
            ft[risc] = (float(np.mean(v)), float(np.max(v))) if v else (0.0, 0.0)
        frame_tot.append(ft)

    nseg = {len(x) for x in progs}
    n = min(nseg)
    print(f"views={len(progs)} programs/view={sorted(nseg)} "
          f"device_span_ms mean={np.mean(spans):.3f} min={np.min(spans):.3f} max={np.max(spans):.3f}")
    if len(nseg) != 1:
        print("WARNING: program count differs across views; aligning by index")
    print("\nPer program (mean over views). Per RISC: mean core kernel ms / busiest core ms "
          "[busiest / window].")
    print(f"{'#':>2} {'start':>7} {'window':>7} {'gap':>6}  " +
          "  ".join(f"{r:>19}" for r in RISCS))
    sum_win = sum_gap = 0.0
    for i in range(n):
        st = np.mean([p[i]["start"] for p in progs])
        win = np.mean([p[i]["end"] - p[i]["start"] for p in progs])
        gap = np.mean([p[i]["gap"] for p in progs])
        sum_win += win
        sum_gap += gap
        cells = []
        for risc in RISCS:
            mn = np.mean([p[i][risc][0] for p in progs])
            mx = np.mean([p[i][risc][1] for p in progs])
            cells.append(f"{mn:6.3f}/{mx:6.3f}[{(mx / win if win else 0):4.2f}]" if mx else f"{'-':>19}")
        print(f"{i:>2} {st:7.3f} {win:7.3f} {gap:6.3f}  " + "  ".join(cells))
        zkeys = sorted({k for p in progs for k in p[i]["zones"]})
        for z, rr in zkeys:
            v = [p[i]["zones"].get((z, rr)) for p in progs]
            v = [x for x in v if x]
            mn = np.mean([x[0] for x in v])
            mx = np.mean([x[1] for x in v])
            nc = int(np.median([x[2] for x in v]))
            print(f"{'':>26}zone {z:<24} {rr:<8} mean {mn:6.3f} max {mx:6.3f} "
                  f"imb {mx - mn:6.3f} cores {nc}")
    print(f"\nsum of program windows {sum_win:.3f} ms + idle between programs {sum_gap:.3f} ms")
    print("\nPer frame, per-core kernel time summed over all programs (mean over views):")
    print(f"{'RISC':<8} {'mean core':>10} {'busiest core':>13}")
    for risc in RISCS:
        mn = np.mean([ft[risc][0] for ft in frame_tot])
        mx = np.mean([ft[risc][1] for ft in frame_tot])
        print(f"{risc:<8} {mn:10.3f} {mx:13.3f}")
    bound = max(np.mean([ft[r][1] for ft in frame_tot]) for r in RISCS)
    print(f"overlap bound (busiest RISC, busiest core): {bound:.3f} ms vs device span "
          f"{np.mean(spans):.3f} ms")


if __name__ == "__main__":
    main()
