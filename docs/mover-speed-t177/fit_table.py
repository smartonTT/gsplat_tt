#!/usr/bin/env python3
"""t177: refit kMoverSpeedP150 from two emit_cores.py per-core tables.

Each mover's emit time is modelled as t = a + b * rec (fixed + per-record cost),
fitted through the two captures (e.g. t166-p2f even split, t174-p2w old table).
The new split gives every mover the record count that makes all t equal for the
same total (secant step); movers whose fit is unusable (b <= 0 or the two record
counts too close) (|d rec| <= 1000, slope outside 0.1-0.35 ms per 1k records, or a < 0)
fall back to the proportional model t = rec / speed.
Usage: fit_table.py <older.txt> <newer.txt> [--prop]
Prints the predicted window and the new table (mean mover = 1000).

emit_cores.py prints Tracy core x, but the host matches kMoverSpeedP150 against
translated NOC x (worker_core_from_logical_core). On yyzo-bh-07 Tracy x 11..15 is
translated x 7, 10, 11, 12, 13 (found from t174-p2w / t177-v1 record counts: Tracy
x 13-15 took the table rows of x 11-13, Tracy x 11-12 matched no row). The table
is printed in translated x.
"""
import sys

TRACY_TO_NOC_X = {11: 7, 12: 10, 13: 11, 14: 12, 15: 13}


def load(path):
    d = {}
    for line in open(path):
        p = line.split()
        if len(p) == 14 and p[2] == "|" and p[0].isdigit():
            x, y = int(p[0]), int(p[1])
            d[(x, y, "B")] = (float(p[3]), float(p[12]))
            d[(x, y, "N")] = (float(p[4]), float(p[13]))
    return d


def main():
    old, new = load(sys.argv[1]), load(sys.argv[2])
    prop = "--prop" in sys.argv
    R = sum(r for _, r in new.values())
    model = {}
    nfb = 0
    for m, (t2, r2) in new.items():
        t1, r1 = old[m]
        b = (t2 - t1) / (r2 - r1) if abs(r2 - r1) > 1000 else 0
        if prop or not 0.1e-3 <= b <= 0.35e-3 or t2 - b * r2 < 0:
            model[m] = (0.0, t2 / r2)
            nfb += not prop
        else:
            model[m] = (t2 - b * r2, b)

    def recs(T):
        return {m: max(0.0, (T - a) / b) for m, (a, b) in model.items()}

    lo, hi = 0.0, 10.0
    for _ in range(100):
        T = (lo + hi) / 2
        lo, hi = (T, hi) if sum(recs(T).values()) < R else (lo, T)
    rc = recs(T)
    mean = R / len(rc)
    print(f"fallback movers {nfb}; max t now {max(t for t, _ in new.values()):.3f}; predicted {T:.3f} ms")
    print("// {x, y, BRISC, NCRISC} relative records share (mean mover = 1000)")
    for (x, y) in sorted({(m[0], m[1]) for m in rc}, key=lambda c: (c[1], c[0])):
        nx = TRACY_TO_NOC_X.get(x, x)
        print(f"    {{{nx}, {y}, {round(1000 * rc[(x, y, 'B')] / mean)}, {round(1000 * rc[(x, y, 'N')] / mean)}}},")


if __name__ == "__main__":
    main()
