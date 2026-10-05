#!/usr/bin/env python3
# t196: per-mover view of the one-launch sort emit from a default-chain Tracy CSV
# (sort_ol_emit zones only, no EMIT_PROF counters needed). Per mover: emit duration,
# start offset (vs the view's first emit start), pairs from the host's speed-
# proportional page ranges (#174 table, looked up by translated x like the host),
# and pairs/us. Then per NoC row/column means, per-view makespan vs median, and
# which movers end last.
#   movers.py <sort_ol rows csv(.gz)> <capture.log> [--per-view]
import collections, gzip, re, statistics as st, sys

CLK = 1350.0  # cycles/us
PX = [1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15]   # Tracy (physical) x of logical columns 0..10
TX = [1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13]    # translated x (worker_core_from_logical_core)
GX, GY = 11, 10


def table(path):
    t = {}
    for m in re.finditer(r"\{(\d+), (\d+), (\d+), (\d+)\}", open(path).read()):
        x, y, b, n = map(int, m.groups())
        t[(x, y)] = (b, n)
    return t


def bounds(P, speed):
    pages = (P + 15) // 16
    tot = sum(speed)
    b, acc = [0], 0
    for s in speed:
        acc += s
        b.append(pages * acc // tot)
    b[-1] = pages
    return b


def main():
    src, log = sys.argv[1], sys.argv[2]
    tab = table(sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3].startswith("--")
                else "render/host/sort_mover_speed.h")
    op = gzip.open if src.endswith(".gz") else open
    iv = collections.defaultdict(list)
    open_ = {}
    with op(src, "rt") as f:
        for line in f:
            c = line.split(",")
            if len(c) < 12 or c[10].strip() != "sort_ol_emit":
                continue
            key = (int(c[1]), int(c[2]), c[3].strip())
            t = int(c[5])
            if c[11].strip() == "ZONE_START":
                open_[key] = t
            elif key in open_:
                iv[key].append((open_.pop(key), t))
    Ps = [int(m) for m in re.findall(r"stage=ONELAUNCH P=(\d+)", open(log).read())]
    nl = min(len(v) for v in iv.values())
    Ps = Ps[-nl:]
    # speed per mover in core order (BRISC, NCRISC), as the host does it
    speed = []
    for c in range(GX * GY):
        lx, ly = c % GX, c // GX
        b, n = tab.get((TX[lx], ly + 2), (1000, 1000))
        speed += [b, n]
    mov = {}
    for c in range(GX * GY):
        lx, ly = c % GX, c // GX
        mov[(PX[lx], ly + 2, "BRISC")] = 2 * c
        mov[(PX[lx], ly + 2, "NCRISC")] = 2 * c + 1
    assert set(mov) == set(iv), (sorted(set(iv) - set(mov))[:5], sorted(set(mov) - set(iv))[:5])
    dur = collections.defaultdict(list)
    off = collections.defaultdict(list)
    rate = collections.defaultdict(list)
    pairs = collections.defaultdict(list)
    mk, med, mx, lastc = [], [], [], collections.Counter()
    skip = 1  # first launch: cold
    for i in range(skip, nl):
        b = bounds(Ps[i], speed)
        s0 = min(iv[k][i][0] for k in iv)
        e1 = max(iv[k][i][1] for k in iv)
        mk.append((e1 - s0) / CLK)
        ds = []
        for k in iv:
            s, e = iv[k][i]
            m = mov[k]
            pr = min(16 * b[m + 1], Ps[i]) - min(16 * b[m], Ps[i])
            d = (e - s) / CLK
            dur[k].append(d); off[k].append((s - s0) / CLK); pairs[k].append(pr)
            rate[k].append(pr / d)
            ds.append(d)
        med.append(st.median(ds)); mx.append(max(ds))
        for k in sorted(iv, key=lambda k: -iv[k][i][1])[:5]:
            lastc[k] += 1
    n = nl - skip
    print(f"launches {n} (skip {skip}); P mean {st.mean(Ps[skip:]):.0f}")
    print(f"makespan us mean {st.mean(mk):.1f}; per-view median dur {st.mean(med):.1f}; "
          f"per-view max dur {st.mean(mx):.1f} (max {max(mx):.1f}); makespan-median {st.mean(mk) - st.mean(med):.1f}")
    allr = st.mean([st.mean(v) for v in rate.values()])
    print(f"\nmean pairs/us over movers {allr:.2f}\n")
    print(" x  y risc   | dur us  sd   | start off | pairs  | pairs/us rel | top5-last")
    rows = []
    for k in sorted(iv, key=lambda k: (k[1], k[0], k[2])):
        r = (k, st.mean(dur[k]), st.pstdev(dur[k]), st.mean(off[k]), st.mean(pairs[k]), st.mean(rate[k]))
        rows.append(r)
        if "--per-mover" in sys.argv:
            print(f"{k[0]:2d} {k[1]:2d} {k[2]:6s} | {r[1]:7.1f} {r[2]:5.1f} | {r[3]:8.1f} | {r[4]:6.0f} | "
                  f"{r[5]:6.2f} {r[5] / allr:5.3f} | {lastc[k]}")
    for risc in ("BRISC", "NCRISC"):
        print(f"\n{risc}: mean dur us by row y (cols x) — then pairs/us rel")
        hdr = "   y\\x " + " ".join(f"{x:6d}" for x in PX)
        print(hdr)
        for y in range(2, 12):
            print(f"  {y:3d}  " + " ".join(f"{st.mean(dur[(x, y, risc)]):6.0f}" for x in PX))
        print(hdr)
        for y in range(2, 12):
            print(f"  {y:3d}  " + " ".join(f"{st.mean(rate[(x, y, risc)]) / allr:6.3f}" for x in PX))
    print("\nmost often among the 5 last to end:")
    for k, v in lastc.most_common(12):
        print(f"  {k} {v}/{n}")
    if "--per-view" in sys.argv:
        for j, i in enumerate(range(skip, nl)):
            print(f"view {i:2d} P {Ps[i]} makespan {mk[j]:7.1f} median {med[j]:7.1f} max {mx[j]:7.1f}")


if __name__ == "__main__":
    main()
