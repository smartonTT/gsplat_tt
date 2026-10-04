#!/usr/bin/env python3
"""Split the one-launch sort emit (zone sort_ol_emit) into its parts (task #154).

Needs a capture with GSPLAT_TT_OL_EMIT_PROF=1: each emit mover (BRISC, NCRISC of
every sort core) then records "ep_*" timestamped-data markers at its end, value =
cycles (or counts: ep_nrec, ep_npf, ep_nb) summed over that launch.

Usage: emit_parts.py <dev30.csv> [n_views=30]
Prints ms per view for: the mean mover, the mean BRISC / NCRISC mover, and the
busiest mover (largest summed sort_ol_emit zone), plus counts per view.
"""
import sys
from collections import defaultdict

CYC_PER_MS = 1350.0 * 1000.0
T_COL, DATA_COL, ZONE_COL, TYPE_COL = 5, 6, 10, 11
PARTS = ["ep_pro", "ep_rdw", "ep_brec", "ep_pairs", "ep_proc", "ep_wfl", "ep_wiss", "ep_drain", "ep_wbar"]
COUNTS = ["ep_nrec", "ep_npf", "ep_nb", "ep_ncold", "ep_nrun"]  # ncold, nrun: fast loop only (t166)
LABEL = {
    "ep_pro": "prologue (ring starts, first pair/brec reads)",
    "ep_rdw": "read barrier wait in loop (brec k, pairs k+1)",
    "ep_brec": "issue_brec: keep/gid scan + brec read issue",
    "ep_pairs": "issue_pairs: pair page read issue",
    "ep_proc": "process_batch total",
    "ep_wfl": "  of which writes_flushed wait (run start)",
    "ep_wiss": "  of which run write issue (flush_run)",
    "ep_drain": "tail drain (partial runs)",
    "ep_wbar": "final write barrier",
}


def main():
    path = sys.argv[1]
    nv = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    emit = defaultdict(int)  # (cx, cy, risc) -> summed sort_ol_emit cycles
    starts = {}
    val = defaultdict(lambda: defaultdict(int))  # mover -> part -> summed value
    with open(path) as f:
        f.readline()
        f.readline()
        for line in f:
            p = line.rstrip("\n").split(",")
            if len(p) < 12:
                continue
            mover = (p[1], p[2], p[3])
            z, typ = p[ZONE_COL], p[TYPE_COL]
            if z == "sort_ol_emit":
                if typ == "ZONE_START":
                    starts[mover] = int(p[T_COL])
                elif typ == "ZONE_END" and mover in starts:
                    emit[mover] += int(p[T_COL]) - starts.pop(mover)
            elif z.startswith("ep_"):
                try:
                    val[mover][z] += int(p[DATA_COL])
                except ValueError:
                    pass
    movers = [m for m in emit if m in val]
    if not movers:
        sys.exit("no ep_* markers: capture with GSPLAT_TT_OL_EMIT_PROF=1")

    def ms(c):
        return c / CYC_PER_MS / nv

    groups = {"mean mover": movers,
              "mean BRISC": [m for m in movers if m[2] == "BRISC"],
              "mean NCRISC": [m for m in movers if m[2] == "NCRISC"]}
    busiest = max(movers, key=lambda m: emit[m])
    print(f"n_views={nv} movers={len(movers)} busiest={busiest} (emit zone {ms(emit[busiest]):.3f} ms/view)")
    cols = list(groups) + ["busiest"]
    print(f"{'part (ms/view)':<50}" + "".join(f"{c:>13}" for c in cols))

    def row(name, f):
        vals = [sum(f(m) for m in g) / max(len(g), 1) for g in groups.values()] + [f(busiest)]
        print(f"{name:<50}" + "".join(f"{v:>13.3f}" for v in vals))

    row("sort_ol_emit zone", lambda m: ms(emit[m]))
    for k in PARTS:
        row(f"{k} {LABEL[k]}", lambda m, k=k: ms(val[m][k]))
    row("pack/key build = proc - wfl - wiss",
        lambda m: ms(val[m]["ep_proc"] - val[m]["ep_wfl"] - val[m]["ep_wiss"]))
    row("unattributed = zone - sum(top-level parts)",
        lambda m: ms(emit[m] - sum(val[m][k] for k in PARTS if k not in ("ep_wfl", "ep_wiss"))))
    for k in COUNTS:
        vals = [sum(val[m][k] for m in g) / max(len(g), 1) / nv for g in groups.values()] + [val[busiest][k] / nv]
        print(f"{k + ' per view':<50}" + "".join(f"{v:>13.0f}" for v in vals))
    vals = [sum(emit[m] for m in g) / max(sum(val[m]["ep_nrec"] for m in g), 1) for g in groups.values()]
    vals.append(emit[busiest] / max(val[busiest]["ep_nrec"], 1))
    print(f"{'emit cycles per record':<50}" + "".join(f"{v:>13.1f}" for v in vals))


if __name__ == "__main__":
    main()
