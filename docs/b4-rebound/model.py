#!/usr/bin/env python3
"""t539: per-core bound for B4 (k2 + sort of view N+1 hidden inside mat+blend of view N).

Input: the t528 device-profiler chunks (best-iter-224, bh-33 p100a, 110 cores, 1350 MHz),
docs/profile-t224/out/dev-P{0,5,10,15,20,25}.csv.gz on ttp/t528-profile-t224 (377b0ab7).
Each chunk holds 11 views x 4 programs per core; the last 5 views are kept (as prep506).
Programs per view, in order: pfwc, k2, sort, mat+blend (BRISC-FW windows, index mod 4).

For every kept view and core it measures (ms):
  - program windows (pfwc, k2, sort, mb) and the period;
  - mover (BRISC, NCRISC) busy in k2 and sort (named zones), TRISC busy in k2_trisc / sort_ol_town;
  - inside mat+blend: when each mover's mat body (sort_subchunk_mat) ends, the NCRISC blend
    loader zone length, and the core's mb end.
B4 model: period* = pfwc window + max_c (mb_c + overflow_c) + 2 boundary gaps, where the
movers of core c must absorb k2+sort(N+1) work W_c in their free time F_c after their mat body:
  F_c = (mb_end - BRISC mat end) + share * (NCRISC loader zone)   [share = 0, 0.5, 0.8]
  W_c = k2+sort work moved to movers only (TRISC parts re-costed on a mover, factor r).
Two W variants: 'xfer' = mover busy + r * TRISC busy (k2_trisc, town), r = 1.2 (t200: 216 vs 179
cycles/record); 'calib' = mover-only emit calibrated from t196 (2.16 ms/mover at 11.9k
records/mover, scaled by today's per-core emit time / mean emit time).
A balanced variant also lets the host deal the N+1 k2/sort ranges so that sum W fills sum F
(work-claiming), giving the chip-level bound.
Usage: model.py dev-P0.csv.gz ... > model_out.txt
"""
import csv, gzip, sys
from collections import defaultdict
import statistics as st

MHZ = 1350.0
KEEP = 5
GAP_KEPT = 0.013 * 2   # all-core gaps left (mb->pfwc, pfwc->mb*); #528 measured 0.013 each
R_TRISC = 1.2          # mover cycles per TRISC cycle for the same per-record work (t200)
T196_EMIT_PER_MOVER = 2.16 * (2.72e6 / 220) / 11888  # mover-only emit, ms per mover at today's P


def ms(c):
    return c / (MHZ * 1000.0)


def load(path):
    zones = defaultdict(list)   # (core, risc, name) -> [(s, e)]
    open_ = {}
    with gzip.open(path, "rt") as f:
        next(f); next(f)
        for p in csv.reader(f):
            if len(p) < 12:
                continue
            core = (int(p[1]), int(p[2])); risc = p[3].strip(); name = p[10].strip(); typ = p[11].strip()
            t = int(p[5])
            k = (core, risc, name)
            if typ == "ZONE_START":
                open_[k] = t
            elif typ == "ZONE_END" and k in open_:
                zones[k].append((open_.pop(k), t))
    return zones


def busy_in(iv, s, e):
    return sum(max(0, min(b, e) - max(a, s)) for a, b in iv)


def last_end_in(iv, s, e):
    x = [b for a, b in iv if a >= s and b <= e]
    return max(x) if x else None


def analyse(paths):
    views = []
    for path in paths:
        z = load(path)
        cores = sorted({c for (c, r, n) in z if r == "BRISC" and n == "BRISC-FW"})
        per_view = defaultdict(dict)
        for c in cores:
            fw = sorted(z[(c, "BRISC", "BRISC-FW")])
            n = len(fw)
            assert n % 4 == 0, (path, c, n)
            # program window per core = union over RISCs of FW zones overlapping the BRISC-FW window
            allfw = []
            for r in ("BRISC", "NCRISC", "TRISC_0", "TRISC_1", "TRISC_2"):
                nm = "TRISC-FW" if r.startswith("TRISC") else f"{r}-FW"
                allfw += z.get((c, r, nm), [])
            for v in range(KEEP):
                base = n - 4 * KEEP + 4 * v
                w = []
                for k in range(4):
                    s, e = fw[base + k]
                    ov = [(a, b) for a, b in allfw if b > s and a < e]
                    w.append((min(a for a, b in ov), max(b for a, b in ov)))
                (ps, pe), (ks, ke), (ss, se), (ms_, me) = w
                g = lambda r, nm, s, e: busy_in(z.get((c, r, nm), []), s, e)
                d = {}
                d["pfwc"] = (ps, pe); d["k2"] = (ks, ke); d["sort"] = (ss, se); d["mb"] = (ms_, me)
                mov = {}
                for r in ("BRISC", "NCRISC"):
                    k2b = sum(g(r, nm, ks, ke) for nm in ("k2_pairs", "k2_rows"))
                    sob = sum(g(r, nm, ss, se) for nm in ("sort_ol_fill", "sort_ol_fillb", "sort_ol_prefix", "sort_ol_barrier"))
                    emb = g(r, "sort_ol_emit", ss, se)
                    mat_end = last_end_in(z.get((c, r, "sort_subchunk_mat"), []), ms_, me)
                    mov[r] = dict(k2=ms(k2b), sortx=ms(sob), emit=ms(emb),
                                  mat_end=ms((mat_end or ms_) - ms_))
                d["mov"] = mov
                d["k2_trisc"] = ms(sum(g(r, "k2_trisc", ks, ke) for r in ("TRISC_0", "TRISC_1", "TRISC_2")))
                d["town"] = ms(sum(g(r, "sort_ol_town", ss, se) for r in ("TRISC_0", "TRISC_1", "TRISC_2")))
                t2m = last_end_in(z.get((c, "TRISC_2", "mat_cull_mask"), []), ms_, me)
                d["t2_mat_end"] = ms((t2m or ms_) - ms_)
                d["loader"] = ms(g("NCRISC", "tile_blend_load", ms_, me))
                d["rd_bulk"] = ms(g("NCRISC", "rd_l1_bulk", ms_, me))
                d["mb_len"] = ms(me - ms_)
                per_view[v][c] = d
        for v in range(KEEP):
            views.append((path, v, per_view[v]))
    return views


def model(views):
    out = []
    rows = []
    for path, v, cd in views:
        cs = list(cd)
        P = lambda k: (min(cd[c][k][0] for c in cs), max(cd[c][k][1] for c in cs))
        win = {k: ms(P(k)[1] - P(k)[0]) for k in ("pfwc", "k2", "sort", "mb")}
        period = ms(P("mb")[1] - P("pfwc")[0])
        mean_emit = st.mean(cd[c]["mov"]["BRISC"]["emit"] + cd[c]["mov"]["NCRISC"]["emit"] for c in cs) / 2
        rec = dict(view=f"{path.split('/')[-1]}:{v}", period=period, **{f"w_{k}": x for k, x in win.items()})
        mbstart = P("mb")[0]
        for share in (0.0, 0.5, 0.8):
            for wv in ("xfer", "calib", "t2"):
                worst = 0.0; sumW = 0.0; sumF = 0.0; mb_max = 0.0
                for c in cs:
                    d = cd[c]; m = d["mov"]
                    base = sum(m[r]["k2"] + m[r]["sortx"] for r in m) + R_TRISC * d["k2_trisc"]
                    if wv == "t2":
                        # TRISC2 (pack thread; in decode-ahead blend it only joins the T readbacks)
                        # takes the TRISC parts at their own rate (r = 1), BRISC the mover parts.
                        W = sum(m[r]["k2"] + m[r]["sortx"] + m[r]["emit"] for r in m) + d["k2_trisc"] + d["town"]
                    elif wv == "xfer":
                        W = base + sum(m[r]["emit"] for r in m) + R_TRISC * d["town"]
                    else:
                        emit_c = (m["BRISC"]["emit"] + m["NCRISC"]["emit"]) / 2
                        W = base + 2 * T196_EMIT_PER_MOVER * emit_c / mean_emit
                    mb_off = ms(d["mb"][0]) - ms(mbstart)
                    F = (d["mb_len"] - m["BRISC"]["mat_end"]) + share * max(0.0, d["loader"] - d["rd_bulk"])
                    if wv == "t2":
                        F += d["mb_len"] - d["t2_mat_end"] - 0.05  # 0.05: ~60 readback/tile-end syncs

                    over = max(0.0, W - F)
                    worst = max(worst, mb_off + d["mb_len"] + over)
                    mb_max = max(mb_max, mb_off + d["mb_len"])
                    sumW += W; sumF += F
                new = win["pfwc"] + worst + GAP_KEPT
                rec[f"save_{wv}_s{share}"] = period - new
                # chip-level (work-claiming) bound: overflow spread evenly over cores
                bal = win["pfwc"] + mb_max + max(0.0, (sumW - sumF) / len(cs)) + GAP_KEPT
                rec[f"bal_{wv}_s{share}"] = period - bal
                rec[f"W_{wv}"] = sumW / len(cs)
                rec[f"F_{wv}_s{share}"] = sumF / len(cs)
        rec["brisc_mat_end"] = st.mean(cd[c]["mov"]["BRISC"]["mat_end"] for c in cs)
        rec["ncrisc_mat_end"] = st.mean(cd[c]["mov"]["NCRISC"]["mat_end"] for c in cs)
        rec["t2_mat_end"] = st.mean(cd[c]["t2_mat_end"] for c in cs)
        rec["loader"] = st.mean(cd[c]["loader"] for c in cs)
        rec["rd_bulk"] = st.mean(cd[c]["rd_bulk"] for c in cs)
        rec["mb_len"] = st.mean(cd[c]["mb_len"] for c in cs)
        rec["k2_mov"] = st.mean(sum(cd[c]["mov"][r]["k2"] for r in ("BRISC", "NCRISC")) for c in cs)
        rec["sort_mov"] = st.mean(sum(cd[c]["mov"][r]["sortx"] + cd[c]["mov"][r]["emit"] for r in ("BRISC", "NCRISC")) for c in cs)
        rec["k2_trisc"] = st.mean(cd[c]["k2_trisc"] for c in cs)
        rec["town"] = st.mean(cd[c]["town"] for c in cs)
        rows.append(rec)
    return rows


def main():
    views = analyse(sys.argv[1:])
    rows = model(views)
    # skip the first kept view of each chunk only if it is a chunk boundary artefact: keep all
    keys = list(rows[0].keys())[1:]
    print(f"# t539 B4 model over {len(rows)} views (t528 chunks, last {KEEP} views each); ms/view")
    print(f"# R_TRISC={R_TRISC}  mover-only emit (t196 calib) = {T196_EMIT_PER_MOVER:.3f} ms/mover at mean load")
    print("view " + " ".join(keys))
    for r in rows:
        print(r["view"], " ".join(f"{r[k]:.3f}" for k in keys))
    print("median", " ".join(f"{st.median(r[k] for r in rows):.3f}" for k in keys))
    print("min   ", " ".join(f"{min(r[k] for r in rows):.3f}" for k in keys))
    print("max   ", " ".join(f"{max(r[k] for r in rows):.3f}" for k in keys))


if __name__ == "__main__":
    main()
