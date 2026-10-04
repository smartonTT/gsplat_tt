#!/usr/bin/env python3
"""Per-core view of the one-launch sort emit (task #174).

Needs a capture with GSPLAT_TT_OL_EMIT_PROF=1 (see emit_parts.py). Prints, per
physical core (x, y), BRISC and NCRISC ms/view of: the sort_ol_emit zone, ep_brec
(brec read issue), ep_proc (process loop) and the records per view; then per NoC
row y the mean / max of each column, and the per-view emit window (last mover end
minus first mover start, averaged over views) with the share of views each mover
ends last.

Usage: emit_cores.py <dev30.csv> [n_views=30] [--weights]
--weights also prints a sort_mover_weights.h table: per physical core, BRISC and
NCRISC records per ms (1/cost) scaled so the mean mover is 1000, and the emit time
the measured costs predict for a split in proportion to them.
"""
import sys
from collections import defaultdict

CYC_PER_MS = 1350.0 * 1000.0
T_COL, DATA_COL, ZONE_COL, TYPE_COL = 5, 6, 10, 11


def main():
    path = sys.argv[1]
    args = [a for a in sys.argv[2:] if not a.startswith("--")]
    nv = int(args[0]) if args else 30
    spans = defaultdict(list)  # mover -> [(start, end)] per launch
    starts = {}
    val = defaultdict(lambda: defaultdict(int))
    with open(path) as f:
        f.readline()
        f.readline()
        for line in f:
            p = line.rstrip("\n").split(",")
            if len(p) < 12:
                continue
            mover = (int(p[1]), int(p[2]), p[3])
            z, typ = p[ZONE_COL], p[TYPE_COL]
            if z == "sort_ol_emit":
                if typ == "ZONE_START":
                    starts[mover] = int(p[T_COL])
                elif typ == "ZONE_END" and mover in starts:
                    spans[mover].append((starts.pop(mover), int(p[T_COL])))
            elif z in ("ep_brec", "ep_proc", "ep_nrec", "ep_wiss", "ep_npf"):
                try:
                    val[mover][z] += int(p[DATA_COL])
                except ValueError:
                    pass
    movers = sorted(spans)

    def ms(c):
        return c / CYC_PER_MS / nv

    emit = {m: sum(e - s for s, e in spans[m]) for m in movers}
    print(f"{'x':>3}{'y':>3} | {'emitB':>7}{'emitN':>7} | {'brecB':>7}{'brecN':>7} | "
          f"{'procB':>7}{'procN':>7} | {'recB':>7}{'recN':>7}")
    cores = sorted({(m[0], m[1]) for m in movers}, key=lambda c: (c[1], c[0]))
    rows = defaultdict(list)
    for c in cores:
        b, n = c + ("BRISC",), c + ("NCRISC",)
        r = [ms(emit.get(b, 0)), ms(emit.get(n, 0)), ms(val[b]["ep_brec"]), ms(val[n]["ep_brec"]),
             ms(val[b]["ep_proc"]), ms(val[n]["ep_proc"]), val[b]["ep_nrec"] / nv, val[n]["ep_nrec"] / nv]
        rows[c[1]].append(r)
        print(f"{c[0]:>3}{c[1]:>3} | {r[0]:7.3f}{r[1]:7.3f} | {r[2]:7.3f}{r[3]:7.3f} | "
              f"{r[4]:7.3f}{r[5]:7.3f} | {r[6]:7.0f}{r[7]:7.0f}")
    print("\nper row y: mean (max) of emitB emitN brecB brecN, mean recB recN")
    for y in sorted(rows):
        rr = rows[y]
        cols = list(zip(*rr))
        s = " ".join(f"{sum(c) / len(c):6.3f}({max(c):6.3f})" for c in cols[:4])
        print(f"y={y:>2} n={len(rr):>2} {s}  rec {sum(cols[6]) / len(rr):6.0f} {sum(cols[7]) / len(rr):6.0f}")
    # per-view window: launches are aligned by index across movers
    nl = min(len(spans[m]) for m in movers)
    win, last = 0, defaultdict(int)
    for i in range(nl):
        s0 = min(spans[m][i][0] for m in movers)
        e_m = max(movers, key=lambda m: spans[m][i][1])
        win += spans[e_m][i][1] - s0
        last[e_m] += 1
    print(f"\nemit window (first start -> last end) mean over {nl} launches: {win / nl / CYC_PER_MS:.3f} ms")
    for m, k in sorted(last.items(), key=lambda kv: -kv[1])[:8]:
        print(f"  last to end: {m} {k}x")
    if "--weights" in sys.argv:
        speed = {m: val[m]["ep_nrec"] / max(emit[m], 1) for m in movers}  # records per cycle
        mean = sum(speed.values()) / len(speed)
        tot = sum(val[m]["ep_nrec"] for m in movers)
        print(f"\npredicted emit (pages in proportion to speed): "
              f"{ms(tot / sum(speed.values())):.3f} ms/view; max now {ms(max(emit.values())):.3f}")
        print("// {x, y, BRISC, NCRISC} relative speed (records per cycle, mean mover = 1000)")
        for c in cores:
            b, n = c + ("BRISC",), c + ("NCRISC",)
            print(f"    {{{c[0]}, {c[1]}, {round(1000 * speed[b] / mean)}, {round(1000 * speed[n] / mean)}}},")


if __name__ == "__main__":
    main()
