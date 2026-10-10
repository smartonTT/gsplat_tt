#!/usr/bin/env python3
"""t464: device busy/idle timeline per view from a Tracy device CSV (CPU only).
Usage: devtimeline.py dev.csv.gz [-v]
The capture carries no run host IDs, so launches are found by time: the union over all
cores of the *-KERNEL zones gives device-busy intervals; idle = gaps between them. A view
starts at the earliest 'pfwc' zone start of a cluster. Gaps > 5 ms (host PNG save in
latency+dump mode) are reported separately and excluded from in-view idle."""
import csv, gzip, sys
C = 1350.0  # cycles per us (aiclk 1350)

def load(path):
    iv, pfwc, sub = [], [], []
    open_ = {}
    with gzip.open(path, "rt") as f:
        next(f); next(f)
        for p in csv.reader(f):
            if len(p) < 12:
                continue
            ty = p[11].strip()
            if ty not in ("ZONE_START", "ZONE_END"):
                continue
            nm, t = p[10].strip(), int(p[5])
            k = (p[1], p[2], p[3], nm)
            if ty == "ZONE_START":
                open_[k] = t
                continue
            s = open_.pop(k, None)
            if s is None:
                continue
            if nm.endswith("-KERNEL"):
                iv.append((s, t))
            elif nm == "pfwc":
                pfwc.append(s)
            if not nm.endswith("-FW") and not nm.endswith("-KERNEL"):
                sub.append((s, t, nm))
    return iv, pfwc, sub

def merge(iv, slack=0):
    iv.sort(); out = []
    for s, e in iv:
        if out and s <= out[-1][1] + slack:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out

def main(path):
    iv, pfwc, sub = load(path)
    busy = merge(iv)
    pfwc.sort(); starts = []
    for t in pfwc:                      # cluster pfwc zone starts into views
        if not starts or t - starts[-1] > 3 * 1350 * 1000:
            starts.append(t)
    starts.append(float("inf"))
    ms = lambda c: c / C / 1000
    rows = []
    for vi in range(len(starts) - 1):
        a, b = starts[vi], starts[vi + 1]
        seg = [x for x in busy if a - 1350 * 50 <= x[0] < b]
        if not seg:
            continue
        on = sum(e - s for s, e in seg); idle = 0; big = 0; gaps = []
        for (s0, e0), (s1, e1) in zip(seg, seg[1:]):
            g = s1 - e0
            if ms(g) > 5:
                big += g
            else:
                idle += g; gaps.append(ms(g))
        span = seg[-1][1] - seg[0][0] - big
        # which sub-zones start right after each gap (first program after a bubble)
        rows.append((vi, ms(span), ms(on), ms(idle), len(gaps), sorted(gaps, reverse=True)[:6], ms(big)))
    for vi, span, on, idle, n, top, big in rows:
        print(f"view {vi:2d}: device span {span:6.3f} busy {on:6.3f} in-view idle {idle:6.3f} "
              f"({n} gaps, top {', '.join(f'{g:.3f}' for g in top)}) host gap {big:6.1f}")
    r = rows[1:]
    if r:
        f = lambda i: sum(x[i] for x in r) / len(r)
        print(f"mean views 1..{len(rows) - 1}: span {f(1):.3f} busy {f(2):.3f} idle {f(3):.3f} ms")
    if "-v" in sys.argv and len(rows) > 2:   # program sequence of view 2 with bubbles
        a, b = starts[2], starts[3]
        seg = [x for x in busy if a - 1350 * 50 <= x[0] < b]
        for s, e in seg:
            names = sorted({nm for s2, e2, nm in sub if s <= s2 < e})
            print(f"  +{ms(s - seg[0][0]):7.3f} dur {ms(e - s):6.3f} {','.join(names)[:110]}")

if __name__ == "__main__":
    main(sys.argv[1])

def occupancy(path, view=2):
    """Per program of one view: core-occupancy = sum over cores of that core's kernel
    span / (cores used x program window). 1 - occupancy = idle core-time a concurrent
    program on the same cores could use (the blend tail, pfwc imbalance)."""
    percore = []
    open_ = {}
    with gzip.open(path, "rt") as f:
        next(f); next(f)
        for p in csv.reader(f):
            if len(p) < 12 or not p[10].strip().endswith("-KERNEL"):
                continue
            k = (p[1], p[2], p[3]); t = int(p[5])
            if p[11].strip() == "ZONE_START":
                open_[k] = t
            elif k in open_:
                percore.append(((p[1], p[2]), open_.pop(k), t))
    return percore
