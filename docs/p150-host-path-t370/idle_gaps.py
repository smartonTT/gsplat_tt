#!/usr/bin/env python3
"""t370: device-idle gaps inside each frame of a Tracy `csvexport -u` dump (device zones on the
host clock). Device busy = union of all *-FW zones over all cores. A frame runs from its first
pfwc start to the end of its last tile_blend_sfpu; frames 1..N-1 (frame 0 warm-up dropped).
Each gap >= MIN_US is listed with the device zone before/after it and the host zones overlapping it.

Usage: idle_gaps.py <host_u.csv[.gz]> [min_us=10]
"""
import csv, gzip, sys
from collections import defaultdict

FW = {"BRISC-FW", "NCRISC-FW", "TRISC-FW"}
DEVZ = {"pfwc", "k2_pairs", "k2_rows", "sort_ol_prefix", "sort_ol_barrier", "sort_ol_fill",
        "sort_ol_fillb", "sort_ol_emit", "sort_ol_town", "sort_subchunk_mat", "mat_cull_mask",
        "tile_blend_load", "tile_blend_sfpu"}
HOSTSKIP = {"PollLoop", "Idle", "readDeviceMarkerData", "rd_l1_bulk"}


def main():
    path = sys.argv[1]; min_ns = float(sys.argv[2] if len(sys.argv) > 2 else 10) * 1e3
    op = gzip.open if path.endswith(".gz") else open
    fw, dev, host = [], [], []
    with op(path, "rt") as f:
        for r in csv.reader(f):
            if len(r) < 8 or r[0] == "name":
                continue
            n, t0, d = r[0], int(r[5]), int(r[6])
            if n in FW: fw.append((t0, t0 + d))
            elif n in DEVZ: dev.append((t0, t0 + d, n))
            elif n not in HOSTSKIP and not n.endswith("-KERNEL"): host.append((t0, t0 + d, n))
    fw.sort(); dev.sort(); host.sort()
    pf = [z for z in dev if z[2] == "pfwc"]
    starts = sorted({z[0] for z in pf})
    # frame starts: pfwc starts more than 1 ms after the previous one
    fs = [starts[0]] + [b for a, b in zip(starts, starts[1:]) if b - a > 1e6]
    busy = []
    for a, b in fw:
        if busy and a <= busy[-1][1]: busy[-1][1] = max(busy[-1][1], b)
        else: busy.append([a, b])
    agg = defaultdict(lambda: [0, 0.0]); tot_idle = []; spans = []
    for i in range(1, len(fs)):
        f0 = fs[i]; f1 = fs[i + 1] if i + 1 < len(fs) else float("inf")
        bl = [z for z in dev if f0 <= z[0] < f1 and z[2] == "tile_blend_sfpu"]
        if not bl: continue
        fe = max(z[1] for z in bl); spans.append(fe - f0)
        bb = [x for x in busy if x[1] > f0 and x[0] < fe]
        idle = 0
        for (a0, a1), (b0, b1) in zip(bb, bb[1:]):
            g = b0 - a1
            if g < min_ns: continue
            idle += g
            before = max((z for z in dev if z[1] <= a1 + 1000), key=lambda z: z[1], default=(0, 0, "?"))[2]
            after = min((z for z in dev if z[0] >= b0 - 1000), key=lambda z: z[0], default=(0, 0, "?"))[2]
            hz = sorted({z[2] for z in host if z[0] < b0 and z[1] > a1})
            key = f"{before} -> {after}"
            agg[key][0] += 1; agg[key][1] += g
            if i <= 2: print(f"frame {i} gap {g/1e3:8.1f} us  {key:40s} host: {', '.join(hz)[:120]}")
        tot_idle.append(idle)
    n = len(spans)
    print(f"\nframes {n}: device span pfwc start -> blend end mean {sum(spans)/n/1e6:.3f} ms, "
          f"idle inside it {sum(tot_idle)/n/1e6:.3f} ms/frame")
    for k, (c, s) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
        print(f"  {k:44s} n={c:3d} mean/frame {s/n/1e3:8.1f} us  mean/gap {s/c/1e3:8.1f} us")


if __name__ == "__main__":
    main()
