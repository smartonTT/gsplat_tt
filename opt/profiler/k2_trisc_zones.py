#!/usr/bin/env python3
"""K2 TRISC job timeline from zones (task #291, GSPLAT_TT_K2_TRISC=1 capture).

Needs no K2_PROF: uses the movers' "k2_pairs" zone (job prep + own part + DONE
waits + write-back) and the TRISCs' "k2_trisc" zone (one per job with pages).
TRISC 0 runs BRISC's job 0, TRISC 2 NCRISC's job 0, TRISC 1 BRISC's then
NCRISC's job 1. Per core and view, ms (means over views and cores):
  prep   = job zone start - mover k2_pairs start (window reads, post, GO seen)
  run    = job zone length
  slack  = mover k2_pairs end - job end (write-back plus whatever the mover
           still did after the job ended; small = the mover waited for it)
Usage: k2_trisc_zones.py <dev30.csv> [n_views=30]
"""
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stitch_device_csv import read_csv, segment_frames  # noqa: E402

CYC_MS = 1350.0 * 1000.0
ZONE, TYPE = 10, 11


def main():
    path = sys.argv[1]
    _, lines, ts = read_csv(path)
    frame, n_frames, _ = segment_frames(lines, ts)
    iv = defaultdict(list)  # (frame, core, risc, zone) -> [(s, e)]
    stack = {}
    for ln, t, fr in zip(lines, ts, frame):
        if fr < 0:
            continue
        p = ln.split(",", 12)
        z = p[ZONE]
        if z not in ("k2_pairs", "k2_trisc"):
            continue
        key = (fr, (p[1], p[2]), p[3], z)
        if p[TYPE] == "ZONE_START":
            stack[key] = int(t)
        elif p[TYPE] == "ZONE_END" and key in stack:
            iv[key].append((stack.pop(key), int(t)))
    acc = defaultdict(list)
    cores = {(k[0], k[1]) for k in iv if k[3] == "k2_pairs"}
    for fr, c in cores:
        mv = {r: iv.get((fr, c, r, "k2_pairs"), []) for r in ("BRISC", "NCRISC")}
        if not mv["BRISC"] or not mv["NCRISC"]:
            continue
        (sb, eb), (sn, en) = mv["BRISC"][0], mv["NCRISC"][0]
        acc["mover B pairs"].append(eb - sb)
        acc["mover N pairs"].append(en - sn)
        acc["core K2 pairs span"].append(max(eb, en) - min(sb, sn))
        jobs = []
        t0 = iv.get((fr, c, "TRISC_0", "k2_trisc"), [])
        t2 = iv.get((fr, c, "TRISC_2", "k2_trisc"), [])
        t1 = sorted(iv.get((fr, c, "TRISC_1", "k2_trisc"), []))
        if t0:
            jobs.append(("B job0 (T0)", t0[0], sb, eb))
        if t2:
            jobs.append(("N job0 (T2)", t2[0], sn, en))
        if len(t1) == 2:
            jobs.append(("B job1 (T1 1st)", t1[0], sb, eb))
            jobs.append(("N job1 (T1 2nd)", t1[1], sn, en))
        for name, (js, je), ms_, me in jobs:
            acc[name + " prep"].append(js - ms_)
            acc[name + " run"].append(je - js)
            acc[name + " slack"].append(me - je)
        # which finished last on the core: a mover's own work or a TRISC job
        lastj = max((j[1][1] for j in jobs), default=0)
        acc["core: last job end -> last mover end"].append(max(eb, en) - lastj)
    print(f"== K2 TRISC job timeline ({len(cores)} core-views, ms per core per view)")
    for k in sorted(acc):
        v = acc[k]
        print(f"{k:<44} mean {sum(v) / len(v) / CYC_MS:7.4f}  max {max(v) / CYC_MS:7.4f}  n {len(v)}")
    # Per view: the K2 critical path is the slowest core.
    span = defaultdict(int)
    for fr, c in cores:
        mv = [iv.get((fr, c, r, "k2_pairs"), []) for r in ("BRISC", "NCRISC")]
        if all(mv):
            span[fr] = max(span[fr], max(m[0][1] for m in mv) - min(m[0][0] for m in mv))
    if span:
        print(f"slowest core K2 pairs span per view: mean {sum(span.values()) / len(span) / CYC_MS:.4f} ms")


if __name__ == "__main__":
    main()
