#!/usr/bin/env python3
"""t553 (B4' step 3b, step 1): BRISC/NCRISC (mover) idle in fused mat+blend vs the NT=1 sort's mover work.

Inputs: per-chunk device CSVs (profile_log_device.csv, frame 0 = warmup, dropped).
  --def: default config + GSPLAT_TT_MV_PROF=1 ("mv_pc" per BRISC/NCRISC, kernels/dataflow/mv_prof.h)
  --nt1: #550's NT=1 captures (GSPLAT_TT_OL_TOWN_NT=1 GSPLAT_TT_OL_EMIT_PROF=1): sort_ol_* zones, ep_* counters
Per (view, core, RISC), ms at 1.35 GHz:
  blend mover: tot, pre (mat phase), busy = tot - pre - waits, wcb (idle on a CB: compute or the other
  mover), wnoc (small NoC round trips), wbulk (slab reads), wrdy (mat ready poll), l1k/l4k/l16k (wcb in
  long waits), tail = last mover end on the chip - this mover's end (from the mv_pc marker time).
  sort mover (NT=1): gross = sum of sort_ol_{count,fill,fillb,prefix,emit} zones on that RISC (barrier
  excluded); net = gross - ep_wfl - ep_proc (emit waits for the town's slots / final batch).
Verdict: per (view, core, RISC) late = max(0, sort_net - hide), hide = wcb (+ tail); extension per view =
max over (core, RISC). Movers have room when the sort's mover work fits their idle on (nearly) every core.
Usage: ana_mv.py --def a.csv[.gz] ... --nt1 b.csv[.gz] ... [--out dir]
"""
import argparse
import gzip
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../opt/profiler"))
from stitch_device_csv import segment_frames  # noqa: E402

MHZ = 1350.0
DATA, ZONE, TYPE, T_COL = 6, 10, 11, 5
MV = "tot pre wcb wnoc wbulk wrdy l1k l4k l16k nw".split()
SORTZ = ("sort_ol_count", "sort_ol_fill", "sort_ol_fillb", "sort_ol_prefix", "sort_ol_emit")
MOVERS = ("BRISC", "NCRISC")


def read(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        f.readline()
        f.readline()
        lines = [ln for ln in f if ln.strip()]
    ts = np.array([int(ln.split(",", 6)[T_COL]) for ln in lines], dtype=np.int64)
    return lines, ts


def parse(paths):
    """-> list of views: dict(mv={(core, risc): {k: v, 'end': t}}, zd={(core, risc, zone): cycles},
    ep={(core, risc): {name: v}}, iv={(core, risc, zone): [(s, e)]})."""
    views = []
    for path in paths:
        lines, ts = read(path)
        frame, nf, _ = segment_frames(lines, ts)
        fv = [dict(mv=defaultdict(dict), zd=defaultdict(int), ep=defaultdict(dict), iv=defaultdict(list))
              for _ in range(nf)]
        stack = {}
        for ln, t, fr in zip(lines, ts, frame):
            if fr < 1:
                continue
            p = ln.split(",", 12)
            z, r, c, ty = p[ZONE], p[3], (int(p[1]), int(p[2])), p[TYPE]
            v = fv[fr]
            if not ty.startswith("ZONE"):
                if z == "mv_pc":
                    x = int(p[DATA])
                    d = v["mv"][(c, r)]
                    d[MV[x >> 32]] = x & 0xFFFFFFFF
                    d.setdefault("end", int(t))
                elif z.startswith("ep_"):
                    v["ep"][(c, r)][z[3:]] = v["ep"][(c, r)].get(z[3:], 0) + int(p[DATA])
                continue
            key = (fr, c, r, z)
            if ty == "ZONE_START":
                stack[key] = int(t)
            elif ty == "ZONE_END" and key in stack:
                s0 = stack.pop(key)
                v["zd"][(c, r, z)] += int(t) - s0
                v["iv"][(c, r, z)].append((s0, int(t)))
        views += fv[1:]
    return views


def ms(x):
    return x / MHZ / 1000.0


def stats(a):
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return "n=0"
    return (f"n={a.size:5d} min {a.min():.3f} p10 {np.percentile(a, 10):.3f} med {np.median(a):.3f} "
            f"mean {a.mean():.3f} p90 {np.percentile(a, 90):.3f} max {a.max():.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--def", dest="d", nargs="+", required=True)
    ap.add_argument("--nt1", nargs="+", default=[])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dv = parse(a.d)
    nv = parse(a.nt1) if a.nt1 else []
    out = []
    pr = lambda s="": (print(s), out.append(s))
    pr(f"default+MV_PROF arm: {len(dv)} views; nt1 arm: {len(nv)} views (warmup frames dropped)")

    rows = []
    for vi, v in enumerate(dv):
        ends = {k: d["end"] for k, d in v["mv"].items() if "tot" in d}
        if not ends:
            continue
        pend = max(ends.values())
        for (c, r), d in v["mv"].items():
            if len(d) < len(MV):
                continue
            w = sum(d[k] for k in ("wcb", "wnoc", "wbulk", "wrdy"))
            rows.append(dict(view=vi, core=c, risc=r, tot=ms(d["tot"]), pre=ms(d["pre"]),
                             blend=ms(d["tot"] - d["pre"]), busy=ms(d["tot"] - d["pre"] - w),
                             **{k: ms(d[k]) for k in ("wcb", "wnoc", "wbulk", "wrdy", "l1k", "l4k", "l16k")},
                             nw=d["nw"], tail=ms(pend - d["end"])))
    if not rows:
        sys.exit("no mv_pc markers: capture with GSPLAT_TT_MV_PROF=1 under the device profiler")
    for r_ in MOVERS:
        rr = [x for x in rows if x["risc"] == r_]
        pr(f"\n{r_} in fused mat+blend: {len(rr)} (view, core) launches, {len({x['core'] for x in rr})} cores. "
           "ms per core per view:")
        for k in ("tot", "pre", "blend", "busy", "wcb", "wnoc", "wbulk", "wrdy", "l1k", "l4k", "l16k", "tail"):
            pr(f"  {k:6s} {stats([x[k] for x in rr])}")
        pr(f"  nw     {stats([x['nw'] for x in rr])}  (count)")

    srows = []
    for vi, v in enumerate(nv):
        keys = {(c, r) for (c, r, z) in v["zd"] if r in MOVERS and z in SORTZ}
        for c, r in keys:
            g = sum(v["zd"].get((c, r, z), 0) for z in SORTZ)
            ep = v["ep"].get((c, r), {})
            net = g - ep.get("wfl", 0) - ep.get("proc", 0)
            srows.append(dict(view=vi, core=c, risc=r, gross=ms(g), net=ms(net),
                              emit=ms(v["zd"].get((c, r, "sort_ol_emit"), 0)),
                              fill=ms(v["zd"].get((c, r, "sort_ol_fill"), 0) + v["zd"].get((c, r, "sort_ol_fillb"), 0)),
                              prefix=ms(v["zd"].get((c, r, "sort_ol_prefix"), 0)),
                              count=ms(v["zd"].get((c, r, "sort_ol_count"), 0)),
                              barrier=ms(v["zd"].get((c, r, "sort_ol_barrier"), 0)),
                              wfl=ms(ep.get("wfl", 0)), proc=ms(ep.get("proc", 0)), rdw=ms(ep.get("rdw", 0)),
                              wbar=ms(ep.get("wbar", 0))))
    if srows:
        for r_ in MOVERS:
            ss = [x for x in srows if x["risc"] == r_]
            pr(f"\n{r_} NT=1 sort mover work: {len(ss)} (view, core), ms per core per view:")
            for k in ("gross", "net", "count", "fill", "prefix", "emit", "barrier", "wfl", "proc", "rdw", "wbar"):
                pr(f"  {k:7s} {stats([x[k] for x in ss])}")
        blend_by = {(x["view"], x["core"], x["risc"]): x for x in rows}
        pr("\nprojection: sort(N+1) mover work inside mat+blend(N) mover idle, per (view, core, RISC):")
        for g in ("wcb", "l1k", "l4k", "l16k", "wcb+wnoc", "none"):
            ext, hid, late_n = defaultdict(float), [], 0
            for x in srows:
                b = blend_by.get((x["view"], x["core"], x["risc"]))
                if g == "none" or b is None:
                    hide = 0.0
                elif g == "wcb+wnoc":
                    hide = b["wcb"] + b["wnoc"] + b["tail"]
                else:
                    hide = b[g] + b["tail"]
                late = max(0.0, x["net"] - hide)
                late_n += late > 0
                ext[x["view"]] = max(ext[x["view"]], late)
                hid.append(min(x["net"], hide) / x["net"] if x["net"] > 0 else 1.0)
            pr(f"  hide = {g:8s}+tail: extension ms/view {stats(list(ext.values()))}; "
               f"hidden fraction median {np.median(hid):.2f}, p10 {np.percentile(hid, 10):.2f}; "
               f"late (view, core, RISC) {late_n}/{len(srows)}")
        # per-core medians (over views), the tightest cores
        pr("\nper (core, RISC) medians over views: sort net vs blend wcb+tail (ms), 8 tightest:")
        pc = defaultdict(lambda: ([], []))
        for x in srows:
            b = blend_by.get((x["view"], x["core"], x["risc"]))
            if b:
                pc[(x["core"], x["risc"])][0].append(x["net"])
                pc[(x["core"], x["risc"])][1].append(b["wcb"] + b["tail"])
        lst = sorted(((np.median(s) - np.median(h), k, np.median(s), np.median(h)) for k, (s, h) in pc.items()),
                     reverse=True)
        for d, k, s, h in lst[:8]:
            pr(f"  core {k[0]} {k[1]:6s} sort {s:.3f} idle {h:.3f} margin {-d:+.3f}")
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, "ana553.txt"), "w") as f:
            f.write("\n".join(out) + "\n")
        for name, rr in (("mv-per-core.csv", rows), ("sort-mover-nt1-per-core.csv", srows)):
            if rr:
                with open(os.path.join(a.out, name), "w") as f:
                    ks = list(rr[0])
                    f.write(",".join(ks) + "\n")
                    for x in rr:
                        f.write(",".join(str(x[k]).replace(",", ";") for k in ks) + "\n")


if __name__ == "__main__":
    main()
