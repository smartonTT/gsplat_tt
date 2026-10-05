#!/usr/bin/env python3
"""Task #147: model of materialize fused into blend, per core, on measured tiles.

Input: a DPRINT file from GSPLAT_TT_MB_TILECYC=1 ("TC rec live cyc" per output
tile, "TCEND n" per core and launch). Each launch's tiles = one frame.

Per tile: n records, measured blend MATH time b (cycles / 1350 MHz).
Mover cost (task #144 fits, us, include the cull lockstep and the DRAM emit):
  whole n <= 6144: NCRISC 5.5 + 0.197 n, BRISC 16.1 + 0.187 n
  whole n >  6144: NCRISC only, -282 + 0.2226 n
  n > 16384 (one slab cannot sort it): NCRISC, sort 0.1083 n once + 0.1267/rec
A whole tile's mover time is split into a sort phase S = total - P*n and a
per-subchunk phase of P us/rec (permute + coefficient fill/patch). The SFPU
cull (CULL us/rec) is TRISC work done in lockstep with the mover at the end
of each subchunk. One slab per mover: the next permute (and the next tile's
sort, which uses the slab as scratch) starts only after the TRISC has blended
the slab. NCRISC claims the big list (n > 6144, descending), then the small
list; BRISC claims the small list (descending). TRISC serves ready entries in
arrival order.
  sim.py t147-tc.dprint [--cull 0.04] [--perm 0.10] [--fit 8192] [--launch -1]
"""
import argparse, heapq, re
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument('dprint')
ap.add_argument('--cull', type=float, default=0.04)    # us/rec TRISC band cull
ap.add_argument('--perm', type=float, default=0.10)    # us/rec mover per-subchunk part
ap.add_argument('--fit', type=int, default=8192)       # slab records (NCRISC)
ap.add_argument('--fit_b', type=int, default=6144)     # BRISC whole-tile cap
ap.add_argument('--switch', type=float, default=1.5)   # us TRISC per entry (DEST init/save)
ap.add_argument('--cores', type=int, default=110)
ap.add_argument('--launch', type=int, default=-1)
ap.add_argument('--big_free', type=int, default=0,
                help='1 = tiles > 16384 records arrive pre-materialized (upper bound)')
ap.add_argument('--dbuf', type=int, default=0,
                help='1 = two slabs per mover: the next permute overlaps the blend')
ap.add_argument('--first_sc', type=int, default=0,     # >0: big tiles' first slab size
                help='records in a big tile first subchunk (select path), 0 = fit')
a = ap.parse_args()
MHZ = 1350.0

# ---- parse: per core, list of launches, each a list of (rec, live, us) ----
per_core = defaultdict(lambda: [[]])
for line in open(a.dprint, errors='replace'):
    m = re.search(r'^\d+:(\d+)-(\d+):TR1:\s*TC(END)?\s+(\d+)(?:\s+(\d+)\s+(\d+))?', line)
    if not m:
        continue
    core = (int(m.group(1)), int(m.group(2)))
    if m.group(3):
        per_core[core].append([])
    else:
        per_core[core][-1].append((int(m.group(4)), int(m.group(5)), int(m.group(6)) / MHZ))
nl = min(len(v) - 1 for v in per_core.values())
L = a.launch if a.launch >= 0 else nl - 1
tiles = [t for v in per_core.values() for t in v[L]]
print(f'cores={len(per_core)} launches={nl} using={L} tiles={len(tiles)} '
      f'records={sum(t[0] for t in tiles)} blend_sum_ms={sum(t[2] for t in tiles)/1e3:.1f}')
big = sorted([t for t in tiles if t[0] > 16384], key=lambda t: -t[0])
print('n>16384:', [(t[0], round(t[2])) for t in big])
mid = [t for t in tiles if 8192 < t[0] <= 16384]
print(f'8192<n<=16384: {len(mid)} tiles; blend us/rec all={sum(t[2] for t in tiles)/max(1,sum(t[0] for t in tiles)):.3f} '
      f'big={sum(t[2] for t in big)/max(1,sum(t[0] for t in big)):.3f}')


def mover_cost(n, ncrisc):
    if n <= a.fit_b:
        return (5.5 + 0.197 * n) if ncrisc else (16.1 + 0.187 * n)
    if n <= 16384:
        return -282 + 0.2226 * n
    return None


def subchunks(n, bigsel):
    if n == 0:
        return [0]
    out, s = [], 0
    if bigsel and a.first_sc:
        out.append(min(n, a.first_sc)); s = out[0]
    while s < n:
        out.append(min(a.fit, n - s)); s += out[-1]
    return out


def plan(t, ncrisc):
    """(sort_us, [(l_sub, mover_us, blend_us)]) for one tile on one mover."""
    n, live, b = t
    if n > 16384 and a.big_free:
        sort, per = 0.0, 0.0
    elif n > 16384:
        sort = 0.1083 * n
        per = 0.1267
    else:
        tot = mover_cost(n, ncrisc)
        per = a.perm
        sort = max(0.0, tot - per * n)
    scs = subchunks(n, n > 16384)
    return sort, [(l, per * l, b * (l / n) if n else 0.0) for l in scs]


def fused(tiles):
    bigq = sorted([t for t in tiles if t[0] > a.fit_b], key=lambda t: -t[0])
    smallq = sorted([t for t in tiles if t[0] <= a.fit_b], key=lambda t: -t[0])
    bi = si = 0
    # per core state
    trisc_free = [0.0] * a.cores
    ready = [[] for _ in range(a.cores)]  # heap of (ready_time, seq, mover, entry)
    ev = []  # (time, seq, kind, core, mover)
    seq = 0
    state = {}  # (core, mover) -> dict(tile plan, idx, t)
    mk = 0.0

    def claim(c, m, now):
        nonlocal bi, si, seq
        t = None
        if m == 0 and bi < len(bigq):
            t = bigq[bi]; bi += 1
        elif si < len(smallq):
            t = smallq[si]; si += 1
        if t is None:
            return False
        sort, scs = plan(t, m == 0)
        state[(c, m)] = {'scs': scs, 'i': 0}
        # sort, then permute the first subchunk
        seq += 1
        heapq.heappush(ev, (now + sort + scs[0][1], seq, 'slab', c, m))
        return True

    for c in range(a.cores):
        for m in (0, 1):
            claim(c, m, 0.0)
    while ev:
        now, _, kind, c, m = heapq.heappop(ev)
        st = state[(c, m)]
        l, mv, bl = st['scs'][st['i']]
        # TRISC: cull (lockstep) + blend of this slab, after earlier entries
        start = max(now, trisc_free[c])
        done = start + a.cull * l + bl + a.switch
        trisc_free[c] = done
        mk = max(mk, done)
        st['i'] += 1
        seq += 1
        if st['i'] < len(st['scs']):
            # next permute starts when the slab is released (dbuf: at once)
            t0 = now if a.dbuf else done
            heapq.heappush(ev, (t0 + st['scs'][st['i']][1], seq, 'slab', c, m))
        else:
            claim(c, m, done)
    return mk


def today(tiles, mat_ms, gap_ms):
    """Separate blend after materialize: greedy dynamic claims, descending."""
    h = [(0.0, c) for c in range(a.cores)]
    heapq.heapify(h)
    for t in sorted(tiles, key=lambda t: -t[2]):
        f, c = heapq.heappop(h)
        heapq.heappush(h, (f + t[2] + 2.0, c))
    return mat_ms + gap_ms + max(f for f, _ in h) / 1e3


fm = fused(tiles) / 1e3
print(f'blend-only makespan (greedy) = {today(tiles, 0, 0):.3f} ms')
print(f'FUSED model makespan = {fm:.3f} ms  (cull={a.cull} perm={a.perm} fit={a.fit} '
      f'first_sc={a.first_sc} switch={a.switch})')


def fit_cost(tiles):
    """Least squares blend_us ~ a + b*rec + c*live (numpy-free normal equations)."""
    X = [(1.0, t[0], t[1]) for t in tiles if t[0] > 0]
    y = [t[2] for t in tiles if t[0] > 0]
    import itertools
    A = [[sum(x[i] * x[j] for x in X) for j in range(3)] for i in range(3)]
    B = [sum(x[i] * v for x, v in zip(X, y)) for i in range(3)]
    # Gaussian elimination
    for i in range(3):
        p = A[i][i]
        for j in range(i + 1, 3):
            f = A[j][i] / p
            for k in range(3):
                A[j][k] -= f * A[i][k]
            B[j] -= f * B[i]
    c = [0.0] * 3
    for i in (2, 1, 0):
        c[i] = (B[i] - sum(A[i][k] * c[k] for k in range(i + 1, 3))) / A[i][i]
    res = [v - (c[0] + c[1] * x[1] + c[2] * x[2]) for x, v in zip(X, y)]
    rms = (sum(r * r for r in res) / len(res)) ** 0.5
    return c, rms


def static_lpt(tiles, order):
    """Host LPT of whole tiles on predicted TRISC cost (oracle = measured), then the
    per-core fused pipeline with the given per-core tile order; both movers feed
    alternately, big tiles (> fit_b) on NCRISC."""
    cost = lambda t: a.cull * t[0] + t[2] + a.switch * max(1, -(-t[0] // a.fit))
    h = [(0.0, c) for c in range(a.cores)]
    heapq.heapify(h)
    lists = [[] for _ in range(a.cores)]
    for t in sorted(tiles, key=lambda t: -cost(t)):
        f, c = heapq.heappop(h)
        lists[c].append(t)
        heapq.heappush(h, (f + cost(t), c))
    mk = 0.0
    for c in range(a.cores):
        ts = sorted(lists[c], key=lambda t: t[0])
        if order == 'small_first_then_desc' and ts:
            ts = ts[:1] + sorted(ts[1:], key=lambda t: -t[0])
        # movers: NCRISC takes tiles > fit_b, the rest alternate by availability
        mfree = [0.0, 0.0]
        tfree = 0.0
        pend = []  # (ready, seq, l, blend)
        for t in ts:
            m = 0 if t[0] > a.fit_b else (0 if mfree[0] < mfree[1] else 1)
            sort, scs = plan(t, m == 0)
            now = mfree[m] + sort
            for l, mv, bl in scs:
                now += mv
                start = max(now, tfree)
                tfree = start + a.cull * l + bl + a.switch
                now = tfree if not a.dbuf else now
            mfree[m] = now if a.dbuf else tfree
        mk = max(mk, tfree)
    return mk


(cf, rms) = fit_cost(tiles)
print(f'blend cost fit: {cf[0]:.1f} + {cf[1]:.4f}*rec + {cf[2]:.4f}*live us (rms {rms:.0f} us)')
for o in ('asc', 'small_first_then_desc'):
    print(f'STATIC LPT (oracle TRISC cost) order={o}: {static_lpt(tiles, o)/1e3:.3f} ms')
print(f'TRISC mean busy (cull+blend) = {(a.cull*sum(t[0] for t in tiles)+sum(t[2] for t in tiles))/a.cores/1e3:.3f} ms')


def today_mat(tiles):
    """Today's separate materialize: LPT of t144-priced items over 2 movers/core."""
    items = []
    for t in tiles:
        n = t[0]
        if n > 16384:
            for s in range(0, n, a.fit):
                items.append((0.1083 * n + 0.1267 * min(a.fit, n - s), True))
        elif n > a.fit_b:
            items.append((-282 + 0.2226 * n, True))
        elif n > 0:
            items.append((None, False, n))
    load = [0.0] * (2 * a.cores)
    for it in sorted(items, key=lambda it: -(it[0] if it[0] is not None else 0.19 * it[2])):
        if it[1]:
            k = min(range(0, 2 * a.cores, 2), key=lambda k: load[k])
            load[k] += it[0]
        else:
            n = it[2]
            k = min(range(2 * a.cores),
                    key=lambda k: load[k] + ((5.5 + 0.197 * n) if k % 2 == 0 else (16.1 + 0.187 * n)))
            load[k] += (5.5 + 0.197 * n) if k % 2 == 0 else (16.1 + 0.187 * n)
    global mat_mean
    mat_mean = sum(load) / len(load)
    return max(load)


tm = today_mat(tiles) / 1e3
print(f'today mat: busiest mover {tm:.3f} ms, mean mover {mat_mean/1e3:.3f} ms')
print(f'TODAY model: mat {tm:.3f} + gap 0.33 + blend {today(tiles,0,0):.3f} = {tm + 0.33 + today(tiles,0,0):.3f} ms')
