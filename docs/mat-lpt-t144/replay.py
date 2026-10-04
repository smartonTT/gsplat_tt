#!/usr/bin/env python3
"""Task #144: replay the materialize LPT on the measured items of a capture
(items.csv from mat_fit.py --items). Each item's measured time is taken as its
true cost; makespan = busiest mover's summed item time per frame. Policies:
legacy (record counts), calibrated (a + b * n per kind / RISC, the model in
render/host/sort_mover_split.h) and oracle (LPT on the measured times).
  replay.py items.csv [model "aWs_N,bWs_N,aWs_B,bWs_B,aWl,bWl,bBc,bBl"]"""
import csv, sys
from collections import defaultdict

M0_CAP, FIT, SLOTS = 6144, 4096, 220
R = list(csv.DictReader(open(sys.argv[1])))
model = [float(v) for v in (sys.argv[2] if len(sys.argv) > 2 else
         "5.5,0.197,16.1,0.187,-282,0.2226,0.1083,0.1267").split(',')]


def lsub(it):
    sc = it['w'] & 255
    return min(FIT, max(0, it['cnt'] - sc * FIT))


def cal(it, risc):
    aN, bN, aB, bB, aL, bL, bc, bl = model
    if it['k'] == 1:
        return bc * it['cnt'] + bl * lsub(it)
    if it['cnt'] > M0_CAP:
        return aL + bL * it['cnt']
    return (aN + bN * it['cnt']) if risc == 0 else (aB + bB * it['cnt'])


def legacy(it, risc):
    return it['cnt'] + (lsub(it) if it['k'] == 1 else 0)


def oracle(it, risc):
    return it['us_n'] if risc == 0 else it['us_b']


def refine(slots_items, load, cost, big, iters=200):
    """Busiest-slot move / swap local search on the model costs."""
    nit = 0
    stuck = set(k for k in range(SLOTS) if len(slots_items[k]) < 2)
    for _ in range(iters):
        cand = [k for k in range(SLOTS) if k not in stuck]
        if not cand:
            break
        b = max(cand, key=lambda k: load[k])
        floor = max((load[k] for k in stuck), default=0.0)
        if load[b] <= floor:
            break
        Lb = load[b]; best = None
        for i, it in enumerate(slots_items[b]):
            ci = cost(it, b & 1)
            for k in range(SLOTS):
                if k == b or (big(it) and (k & 1)):
                    continue
                ck = cost(it, k & 1)
                m = max(Lb - ci, load[k] + ck)
                if m < Lb - 1e-9 and (best is None or m < best[0]):
                    best = (m, i, k, None)
                for j, jt in enumerate(slots_items[k]):
                    if big(jt) and (b & 1):
                        continue
                    m = max(Lb - ci + cost(jt, b & 1), load[k] - cost(jt, k & 1) + ck)
                    if m < Lb - 1e-9 and (best is None or m < best[0]):
                        best = (m, i, k, j)
        if best is None:
            stuck.add(b)
            continue
        _, i, k, j = best
        it = slots_items[b].pop(i)
        load[b] -= cost(it, b & 1); load[k] += cost(it, k & 1)
        if j is not None:
            jt = slots_items[k].pop(j)
            load[k] -= cost(jt, k & 1); load[b] += cost(jt, b & 1)
            slots_items[b].append(jt)
        slots_items[k].append(it)
        stuck.discard(k); stuck.discard(b)
        if len(slots_items[b]) < 2: stuck.add(b)
        nit += 1
    return nit


REFINE = 0


def lpt(items, cost):
    big = lambda it: it['k'] == 1 or it['cnt'] > M0_CAP
    order = sorted(items, key=lambda it: -cost(it, 0))  # stable like std::sort? close enough
    load = [0.0] * SLOTS; true = [0.0] * SLOTS; si = [[] for _ in range(SLOTS)]
    for it in order:
        step = 2 if big(it) else 1
        best, c = None, 0
        for k in range(0, SLOTS, step):
            f = load[k] + cost(it, k & 1)
            if best is None or f < best:
                best, c = f, k
        load[c] = best
        si[c].append(it)
    print("  lpt model_max", round(max(load)), file=sys.stderr)
    if REFINE:
        n = refine(si, load, cost, big); print("  refine", n, "model_max", round(max(load)), file=sys.stderr)
    for c in range(SLOTS):
        true[c] = sum(it['us_n'] if (c & 1) == 0 else it['us_b'] for it in si[c])
    return max(true), sum(true) / SLOTS


frames = defaultdict(list)
for r in R:
    it = dict(f=int(r['f']), tile=int(r['tile']), w=int(r['w']), cnt=int(r['cnt']),
              k=int(r['k']), risc=int(r['risc']), us=float(r['us']))
    frames[it['f']].append(it)
# measured time per RISC: the item's own; for the other RISC use the fitted
# ratio of BRISC/NCRISC whole-small costs (items > M0_CAP never run on BRISC).
aN, bN, aB, bB = model[:4]
for its in frames.values():
    for it in its:
        n = it['cnt']
        rN, rB = aN + bN * n, aB + bB * n
        if it['risc'] == 0:
            it['us_n'] = it['us']; it['us_b'] = it['us'] * rB / rN
        else:
            it['us_b'] = it['us']; it['us_n'] = it['us'] * rN / rB
REFINE = int(sys.argv[3]) if len(sys.argv) > 3 else 0
print("frame  legacy   calib  oracle   mean  max_item  (us, busiest mover)")
tot = defaultdict(float)
for f in sorted(frames):
    its = frames[f]
    l, m = lpt(its, legacy); c, _ = lpt(its, cal); o, _ = lpt(its, oracle)
    mi = max(it['us'] for it in its)
    tot['l'] += l; tot['c'] += c; tot['o'] += o; tot['m'] += m
    print(f"{f:5d} {l:7.0f} {c:7.0f} {o:7.0f} {m:6.0f} {mi:8.0f}")
n = len(frames)
print(f"mean  {tot['l']/n:7.0f} {tot['c']/n:7.0f} {tot['o']/n:7.0f} {tot['m']/n:6.0f}")
# lower bound per frame: max(max item, NCRISC-only work / 110, all work / 220)
print("frame  LB  (max_item, nc_only/110, total/220)")
for f in sorted(frames):
    its = frames[f]
    nco = sum(it['us_n'] for it in its if it['k'] == 1 or it['cnt'] > M0_CAP) / (SLOTS // 2)
    allw = sum(min(it['us_n'], it['us_b']) for it in its) / SLOTS
    mi = max(it['us'] for it in its)
    print(f"{f:5d} {max(mi, nco, allw):6.0f} ({mi:.0f}, {nco:.0f}, {allw:.0f})")
