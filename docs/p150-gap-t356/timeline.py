#!/usr/bin/env python3
"""t356: per-frame host+device timeline from a Tracy `csvexport -u` dump (device zones are
already on the host clock there). Frames are split at the end of host_finish_blend. Times are
ms relative to the frame's first device pfwc start (t=0), averaged over frames 1..N-1.

Usage: timeline.py <host_u.csv[.gz]> [label]
"""
import csv, gzip, sys
from collections import defaultdict

DEV = ["pfwc", "k2_pairs", "k2_rows", "sort_ol_prefix", "sort_ol_barrier", "sort_ol_fill",
       "sort_ol_emit", "sort_ol_town", "sort_subchunk_mat", "mat_cull_mask", "tile_blend_load",
       "tile_blend_sfpu", "BRISC-FW", "NCRISC-FW"]
HOST = ["host_cq1_proj_m", "host_cq1_k2_rows", "host_finish_cq1_bridge", "host_blend_setup",
        "host_finish_blend"]


def main():
    path = sys.argv[1]
    op = gzip.open if path.endswith(".gz") else open
    rows = []
    with op(path, "rt") as f:
        for r in csv.reader(f):
            if len(r) < 8 or r[0] == "name":
                continue
            n = r[0]
            if n in DEV or n in HOST or n == "EnqueueProgram":
                t0 = int(r[5]); rows.append((t0, t0 + int(r[6]), n))
    rows.sort()
    # Host zones: frame = up to the end of host_finish_blend; anchor = the frame's first
    # EnqueueProgram. Device zones (own clock, possibly offset from the host's in the dump):
    # frame = cluster of pfwc starts; anchor = first pfwc start.
    ends = sorted(e for s, e, n in rows if n == "host_finish_blend")
    ps = sorted(s for s, e, n in rows if n == "pfwc")
    dstarts = [ps[0]] + [b for a, b in zip(ps, ps[1:]) if b - a > 1_000_000]
    frames = defaultdict(list)
    import bisect
    for s, e, n in rows:
        if n in DEV:
            frames[("d", bisect.bisect_right(dstarts, s) - 1)].append((s, e, n))
        else:
            frames[("h", bisect.bisect_left(ends, e))].append((s, e, n))
    acc = defaultdict(lambda: [0.0, 0.0, 0])   # name -> [sum start, sum end, n]
    enq = defaultdict(lambda: [0.0, 0])
    nf = 0
    for kind, fi in sorted(frames):
        if fi <= 0 or (kind == "h" and fi >= len(ends)) or (kind == "d" and fi >= len(dstarts)):
            continue
        fr = frames[(kind, fi)]
        p = [s for s, e, n in fr if n == ("pfwc" if kind == "d" else "EnqueueProgram")]
        if not p:
            continue
        t0 = min(p); nf += kind == "d"
        first = {}
        for s, e, n in fr:
            if n == "EnqueueProgram":
                continue
            a = first.setdefault(n, [s, e])
            a[0] = min(a[0], s); a[1] = max(a[1], e)
        for n, (s, e) in first.items():
            acc[n][0] += (s - t0) / 1e6; acc[n][1] += (e - t0) / 1e6; acc[n][2] += 1
        if kind == "d":
            continue
        for k, (s, e, n) in enumerate(sorted(x for x in fr if x[2] == "EnqueueProgram")):
            enq[k][0] += (s - t0) / 1e6; enq[k][1] += 1
    print(f"{sys.argv[2] if len(sys.argv) > 2 else path}: frames={nf}  (device ms from first pfwc start; host ms from first EnqueueProgram)")
    print(f"{'zone':<24}{'start':>9}{'end':>9}{'window':>9}{'n':>4}")
    for n in sorted(acc, key=lambda k: acc[k][0] / acc[k][2]):
        s, e, c = acc[n]
        print(f"{n:<24}{s / c:9.3f}{e / c:9.3f}{(e - s) / c:9.3f}{c:4}")
    print("EnqueueProgram k: mean start  " +
          "  ".join(f"{k}:{v[0] / v[1]:.3f}" for k, v in sorted(enq.items()) if v[1] >= nf // 2))


if __name__ == "__main__":
    main()
