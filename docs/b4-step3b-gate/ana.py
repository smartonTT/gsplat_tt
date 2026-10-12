#!/usr/bin/env python3
"""t550 (B4' step 3b gate): TRISC 2 idle in fused mat+blend vs NT=1 town sort work, per core.

Inputs: per-chunk Tracy device CSVs (profile_log_device.csv, frame 0 = warmup, dropped).
  --def: default config + GSPLAT_TT_T2_PROF=1 (blend "t2_pc" counters; mat "fz_p_*" counters)
  --nt1: GSPLAT_TT_OL_TOWN_NT=1 GSPLAT_TT_OL_EMIT_PROF=1 (+T2_PROF): town "town_pc" on TRISC 2
Per (view, core), ms at 1.35 GHz:
  blend: tot, busy (tot - pre - waits), waits by kind, idle in waits >= 1K/4K/16K/64K cycles,
  tail = program blend end (last core) - this core's blend end (TRISC 2 idle until the program ends)
  mat:   PACK mailbox + tile_regs_wait + patch wait (fz_p_mbw + regw + patchw)
  sort:  NT=1 town TRISC 2 busy = proc - wfl - wq (net work), and tot - go - rdy (incl. in-batch waits)
Projection (per view): with sort(N+1) on TRISC 2 inside mat+blend(N), core c ends late by
  max(0, sort_c - hide_c), hide_c = TRISC 2 idle at granularity G + tail_c; extension = max over cores
  of that lateness (lateness only on the critical path: tail already counts the slack).
  Saving bound = removed sort program span (sort zones, first start -> last end, default arm) - extension.
Usage: ana.py --def a.csv[.gz] ... --nt1 b.csv[.gz] ... [--out dir]
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
T2 = "tot pre wcnt wbulk wrba wmath wres l1k l4k l16k l64k nw nt nsc nrb".split()
TOWN = "tot go proc wfl wq rdy rec run nb ng me".split()


def read(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        f.readline()
        f.readline()
        lines = [ln for ln in f if ln.strip()]
    ts = np.array([int(ln.split(",", 6)[T_COL]) for ln in lines], dtype=np.int64)
    return lines, ts


def parse(paths):
    """-> list of views; each: dict(iv, t2, fz, town) keyed by core."""
    views = []
    for path in paths:
        lines, ts = read(path)
        frame, nf, _ = segment_frames(lines, ts)
        fv = [dict(iv=defaultdict(list), t2=defaultdict(dict), fz=defaultdict(dict),
                   town=defaultdict(dict)) for _ in range(nf)]
        stack = {}
        for ln, t, fr in zip(lines, ts, frame):
            if fr < 1:
                continue
            p = ln.split(",", 12)
            z, r, c, ty = p[ZONE], p[3], (int(p[1]), int(p[2])), p[TYPE]
            v = fv[fr]
            if not ty.startswith("ZONE"):
                if z == "t2_pc" and r == "TRISC_2":
                    x = int(p[DATA]); v["t2"][c][T2[x >> 32]] = x & 0xFFFFFFFF
                elif z == "town_pc" and r == "TRISC_2":
                    x = int(p[DATA]); v["town"][c][TOWN[x >> 32]] = x & 0xFFFFFFFF
                elif z.startswith("fz_p_"):
                    v["fz"][c][z[5:]] = v["fz"][c].get(z[5:], 0) + int(p[DATA])
                continue
            key = (c, r, z)
            if ty == "ZONE_START":
                stack[(fr,) + key] = int(t)
            elif ty == "ZONE_END" and (fr,) + key in stack:
                v["iv"][key].append((stack.pop((fr,) + key), int(t)))
        views += fv[1:]
    return views


def ms(x):
    return x / MHZ / 1000.0


def stats(a):
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return "n=0"
    return (f"n={a.size:4d} min {a.min():.3f} p10 {np.percentile(a, 10):.3f} med {np.median(a):.3f} "
            f"mean {a.mean():.3f} p90 {np.percentile(a, 90):.3f} max {a.max():.3f}")


def spans(v, pred):
    s = [x for (c, r, z), l in v["iv"].items() if pred(z, r) for x, _ in l]
    e = [y for (c, r, z), l in v["iv"].items() if pred(z, r) for _, y in l]
    return (min(s), max(e)) if s else None


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
    pr(f"default arm: {len(dv)} views; nt1 arm: {len(nv)} views (warmup frames dropped)")

    # Zone spans (program-level, first start -> last end over cores), median over views.
    names = sorted({z for v in dv for (_, _, z) in v["iv"]})
    pr("\nzone spans, default arm (ms, median over views; first start -> last end over cores/RISCs):")
    for z in names:
        sp = [spans(v, lambda zz, r, z=z: zz == z) for v in dv]
        sp = [ms(e - s) for x in sp if x for s, e in [x]]
        if sp:
            pr(f"  {z:24s} {np.median(sp):7.3f}")

    sp = [spans(v, lambda z, r: z.startswith("sort_ol")) for v in dv]
    sp = [ms(e - s) for x in sp if x for s, e in [x]]
    pr(f"  {'sort program (sort_ol_*)':24s} {np.median(sp) if sp else float('nan'):7.3f}   {stats(sp)}")

    # Per (view, core) TRISC 2 metrics in the default arm.
    rows = []
    for vi, v in enumerate(dv):
        ends = {c: max(e for _, e in l) for (c, r, z), l in v["iv"].items() if z == "tile_blend_sfpu" and r == "TRISC_2"}
        if not ends:
            continue
        pend = max(ends.values())
        for c, t in v["t2"].items():
            if "tot" not in t:
                continue
            w = sum(t[k] for k in ("wcnt", "wbulk", "wrba", "wmath", "wres"))
            fz = v["fz"].get(c, {})
            matw = sum(fz.get(k, 0) for k in ("mbw", "regw", "patchw"))
            rows.append(dict(view=vi, core=c, tot=ms(t["tot"]), pre=ms(t["pre"]), wait=ms(w),
                             busy=ms(t["tot"] - t["pre"] - w), **{k: ms(t[k]) for k in T2[2:11] if k != "nw"},
                             nw=t["nw"], nt=t["nt"], nsc=t["nsc"], nrb=t["nrb"],
                             tail=ms(pend - ends.get(c, pend)), mat_tot=ms(fz.get("tot", 0)), mat_wait=ms(matw)))
    if not rows:
        sys.exit("no t2_pc markers: capture with GSPLAT_TT_T2_PROF=1 under the device profiler")
    pr(f"\nTRISC 2 in fused mat+blend, default arm: {len(rows)} (view, core) launches, "
       f"{len({r['core'] for r in rows})} cores. ms per core per view:")
    for k in ("tot", "busy", "pre", "wait", "wcnt", "wbulk", "wrba", "wmath", "wres", "l1k", "l4k", "l16k", "l64k",
              "tail", "mat_tot", "mat_wait"):
        pr(f"  {k:9s} {stats([r[k] for r in rows])}")
    for k in ("nw", "nt", "nsc", "nrb"):
        pr(f"  {k:9s} {stats([r[k] for r in rows])}  (counts)")
    pr("  per-core (median over views) blend idle = wait + tail:")
    pc = defaultdict(list)
    for r in rows:
        pc[r["core"]].append(r["wait"] + r["tail"])
    pr(f"    {stats([np.median(x) for x in pc.values()])}")

    # NT=1 town per (view, core).
    srows = []
    for vi, v in enumerate(nv):
        for c, t in v["town"].items():
            if len(t) < len(TOWN):
                continue
            srows.append(dict(view=vi, core=c, net=ms(t["proc"] - t["wfl"] - t["wq"]),
                              busy=ms(t["tot"] - t["go"] - t["rdy"]), tot=ms(t["tot"]), rec=t["rec"], nb=t["nb"]))
    if srows:
        pr(f"\nNT=1 town (TRISC 2 only) per core per view, nt1 arm: {len(srows)} launches, "
           f"{len({r['core'] for r in srows})} cores:")
        for k in ("net", "busy", "tot"):
            pr(f"  {k:9s} {stats([r[k] for r in srows])}")
        pr(f"  rec       {stats([r['rec'] for r in srows])}  (records)")
        pr(f"  per-batch net us: {stats([1000 * r['net'] / r['nb'] for r in srows if r['nb']])}")
        bc, sc = {r["core"] for r in rows}, {r["core"] for r in srows}
        pr(f"  cores: blend {len(bc)}, sort {len(sc)}, both {len(bc & sc)}, sort-only {len(sc - bc)}")
        sv = [spans(v, lambda z, r: z == "sort_ol_town") for v in nv]
        sv = [ms(e - s) for x in sv if x for s, e in [x]]
        pr(f"  sort_ol_town span NT=1 (ms): {stats(sv)}")

        # Projection, view by view (views match by index: same 30-view order in both arms).
        sort_by = defaultdict(dict)
        for r in srows:
            sort_by[r["view"]][r["core"]] = r["net"]
        blend_by = defaultdict(dict)
        for r in rows:
            blend_by[r["view"]][r["core"]] = r
        pr("\nprojection: extension of mat+blend(N) when core c also runs its NT=1 town work (net):")
        for g in ("wait", "l1k", "l4k", "l16k", "l64k", "none"):
            ext = []
            hid = []
            for vi in sorted(set(sort_by) & set(blend_by)):
                late = []
                for c, s in sort_by[vi].items():
                    b = blend_by[vi].get(c)
                    hide = (0.0 if g == "none" else b[g] + b["tail"]) if b else 0.0
                    late.append(max(0.0, s - hide))
                    hid.append(min(s, hide) / s if s > 0 else 1.0)
                ext.append(max(late) if late else 0.0)
            pr(f"  hide = {g:5s}+tail: extension ms/view {stats(ext)}; hidden fraction per core median {np.median(hid):.2f}")
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, "ana550.txt"), "w") as f:
            f.write("\n".join(out) + "\n")
        with open(os.path.join(a.out, "t2-per-core.csv"), "w") as f:
            ks = list(rows[0])
            f.write(",".join(ks) + "\n")
            for r in rows:
                f.write(",".join(str(r[k]).replace(",", ";") for k in ks) + "\n")
        if srows:
            with open(os.path.join(a.out, "town-nt1-per-core.csv"), "w") as f:
                ks = list(srows[0])
                f.write(",".join(ks) + "\n")
                for r in srows:
                    f.write(",".join(str(r[k]).replace(",", ";") for k in ks) + "\n")


if __name__ == "__main__":
    main()
