#!/usr/bin/env python3
"""Task #175: schedule model of one-launch materialize, no device.

Compares three ways to handle big tiles (> 16384 records):
  off    : every subchunk item re-reads keys and re-sorts the whole tile (pre-t168)
  t168   : one sort item publishes ids, one gather item per subchunk waits on it
  split  : two half-sort items (records [0,N/2) and [N/2,N)) publish keys+ids,
           gather items wait for both halves and merge their own rank range
  free   : upper bound, big-tile sort chain costs nothing (sort items = 0 us)

Tiles: per-tile record counts of views 0 and 1 from docs/fuse-matblend-t147/out/t147-tc.dprint.
Host assignment = build_mat_worklist (render/host/sort_mover_split.h @ a71c61e): LPT on host cost,
big items NCRISC only (even slots), each mover runs sort items first, gathers last. Device
execution = each mover runs its list in order, a gather starts after max(own mover free,
its sort item(s) done).

Costs (us). Big-tile terms from the t168 Tracy zones (yyzo-bh-07 p100a, 30 views, MATCULL_PROF=1,
docs/mat-shared-t168/out on branch ttp/t168-...), per record of the tile N (avg big N 19238):
  keys 0.294 ms/item -> 0.01528 N ; sort 22.5 ms/view saved by 14.1 fewer sorts -> 1.595 ms -> 0.0829 N
  pub 0.156 ms/item -> 0.0081 per published word ; gather 0.284 ms/item -> 0.01476 N
  cull+write of own records 0.059 us/rec (590.5 ms/view mover total minus non-nested zones, 8 ms/view
  over ~135k big-tile records) ; ids read 1.3 us/item
Whole-tile items: t144 fits (NCRISC 5.5+0.197n, BRISC 16.1+0.187n, 6144<n<=16384: -282+0.2226n).
Merge (split only): --merge us per merged record (default 0.015 = ~20 cycles, about one radix
scatter pass: the radix sort costs 0.0829 us/rec = 112 cycles over 3-4 passes + histogram),
plus --merge_fix us per gather item (read both halves' keys, merge-path search).
"""
import argparse, re
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument('--dprint', default='docs/fuse-matblend-t147/out/t147-tc.dprint')
ap.add_argument('--merge', type=float, default=0.015)
ap.add_argument('--merge_fix', type=float, default=10.0)
ap.add_argument('--sort', type=float, default=0.0829)
ap.add_argument('--cores', type=int, default=110)
ap.add_argument('--sched', default='host', choices=['host', 'dep'],
                help='host = build_mat_worklist LPT + kind order; dep = oracle list scheduler on true '
                     'costs that only starts a gather once its sort is done (a bound, not host code)')
ap.add_argument('--mid', type=int, default=0,
                help='split mode: also split whole tiles with mid < n <= 16384 (0 = off)')
a = ap.parse_args()

FIT, OV, M0 = 8192, 16384, 6144
BRISC_HALF_CAP = 2 * M0 + 256  # BRISC CB_BSORT holds (2*6144+256) u32 = 12544 keys
K_KEYS, K_PUB, K_GATH, K_OWN, IDS = 0.01528, 0.0081, 0.01476, 0.059, 1.3


def load_views(path):
    per_core = defaultdict(lambda: [[]])
    for line in open(path, errors='replace'):
        m = re.search(r'^\d+:(\d+)-(\d+):TR1:\s*TC(END)?\s+(\d+)(?:\s+(\d+)\s+(\d+))?', line)
        if not m:
            continue
        core = (int(m.group(1)), int(m.group(2)))
        if m.group(3):
            per_core[core].append([])
        else:
            per_core[core][-1].append(int(m.group(4)))
    nl = min(len(v) - 1 for v in per_core.values())
    return [[n for v in per_core.values() for n in v[L]] for L in range(1, nl)]


def whole_cost(n, ncrisc):
    if n <= M0:
        return (5.5 + 0.197 * n) if ncrisc else (16.1 + 0.187 * n)
    return -282 + 0.2226 * n  # NCRISC only


def items_for(counts, mode):
    """[(host_cost, kind, tile, data)]; kind: 'w' whole, 'o' off subchunk, 's' sort, 'h' half, 'g' gather."""
    it = []
    for t, n in enumerate(counts):
        if n == 0:
            continue
        if n <= OV and not (mode == 'split' and a.mid and n > a.mid):
            it.append((n, 'w', t, n))
            continue
        nsc = -(-n // FIT)
        if mode == 'off':
            for sc in range(nsc):
                l = min(FIT, n - sc * FIT)
                it.append((n + l, 'o', t, (n, l)))
            continue
        if mode == 'split' and (n + 1) // 2 <= BRISC_HALF_CAP:
            h0 = n // 2
            for h, nh in ((0, h0), (1, n - h0)):
                it.append((nh * 61 // 100, 'h', t, (n, nh, h)))
        else:
            it.append((n * 61 // 100, 's', t, (n,)))
        for sc in range(nsc):
            l = min(FIT, n - sc * FIT)
            it.append(((n * 6 + l * 64) // 100, 'g', t, (n, l)))
    return it


def true_cost(kind, data, ncrisc, mode):
    if kind == 'w':
        return whole_cost(data, ncrisc)
    if kind == 'o':
        n, l = data
        return (K_KEYS + a.sort) * n + K_GATH * n + K_OWN * l + 0.002 * l
    if kind == 's':
        return 0.0 if mode == 'free' else (K_KEYS + a.sort + K_PUB) * data[0]
    if kind == 'h':
        n, nh, h = data
        return (K_KEYS + a.sort + 2 * K_PUB) * nh + 5.0  # keys + ids published
    if kind == 'g':
        n, l = data
        c = K_GATH * n + K_OWN * l + IDS
        if mode == 'split':
            c += a.merge * l + a.merge_fix
        return c


def dep_lists(items, S, mode, pair):
    """Greedy list scheduling on true costs: the earliest-free mover takes the first sort/half
    item, else the largest whole item or gather whose sort(s) are already done; with nothing
    ready it waits for the next sort completion. Big items stay on NCRISC (BRISC may take a half
    when pair=False). pair=True: the second half of a tile goes to the partner mover at once."""
    import heapq
    sorts = [x for x in items if x[1] in ('s', 'h')]
    rest = sorted([x for x in items if x[1] not in ('s', 'h')],
                  key=lambda x: -true_cost(x[1], x[3], True, mode))
    need = defaultdict(int)
    for x in sorts:
        need[x[2]] += 1
    done_t = {}
    part = defaultdict(float)
    free = [0.0] * S
    lists = [[] for _ in range(S)]
    pending = {}
    while sorts or rest:
        k = min(range(S), key=lambda k: (free[k], k))
        now = free[k]
        nc = k % 2 == 0
        pick = None
        if k in pending:
            pick = pending.pop(k)
        else:
            for x in sorts:
                if nc or (x[1] == 'h' and x[3][1] <= BRISC_HALF_CAP):
                    pick = x
                    break
            if pick is not None:
                sorts.remove(pick)
                if pair and pick[1] == 'h':
                    mate = next((y for y in sorts if y[2] == pick[2]), None)
                    if mate is not None:
                        sorts.remove(mate)
                        pending[k + 1 if nc else k - 1] = mate
        if pick is None:
            for x in rest:
                big = x[1] in ('o', 'g') or (x[1] == 'w' and x[3] > M0)
                if big and not nc:
                    continue
                if x[1] == 'g' and done_t.get(x[2], float('inf')) > now + 1e-9:
                    continue
                pick = x
                break
            if pick is not None:
                rest.remove(pick)
        if pick is None:
            cand = [t for t in done_t.values() if t > now + 1e-9] + \
                   [free[j] for j in range(S) if free[j] > now + 1e-9]
            free[k] = min(cand) if cand else now + 1.0
            continue
        lists[k].append((pick[1], pick[2], pick[3]))
        free[k] = now + true_cost(pick[1], pick[3], nc, mode)
        if pick[1] in ('s', 'h'):
            part[pick[2]] = max(part[pick[2]], free[k])
            need[pick[2]] -= 1
            if need[pick[2]] == 0:
                done_t[pick[2]] = part[pick[2]]
    return lists


def schedule(counts, mode, pair=False):
    """pair=True (split): both halves of a tile go to one core's NCRISC + BRISC."""
    hm = 'shared' if mode in ('t168', 'free') else mode
    items = items_for(counts, 'split' if mode == 'split' else ('off' if mode == 'off' else 't168'))
    items.sort(key=lambda x: -x[0])  # std::sort desc (ties unspecified; stable here)
    S = 2 * a.cores
    load = [0] * S
    lists = [[] for _ in range(S)]
    pend_half = {}
    for hc, kind, t, d in items:
        big = kind in ('o', 's', 'g') or (kind == 'w' and d > M0)
        if kind == 'h':
            if pair:
                if t not in pend_half:
                    c = min(range(0, S, 2), key=lambda k: load[k] + load[k + 1])
                    pend_half[t] = c + 1
                else:
                    c = pend_half[t]
            else:
                c = min(range(S), key=lambda k: load[k])
        else:
            step = 2 if big else 1
            c = min(range(0, S, step), key=lambda k: load[k])
        lists[c].append((kind, t, d))
        load[c] += hc
    rank = {'s': 0, 'h': 0, 'g': 2}
    for v in lists:
        v.sort(key=lambda w: rank.get(w[0], 1))
    if a.sched == 'dep':
        lists = dep_lists(items, S, mode, pair)
    # device execution; iterate until gather start times settle (sorts never wait)
    done_sort = {}
    fin = [0.0] * S
    for _ in range(4):
        new_done = defaultdict(float)
        for k in range(S):
            now = 0.0
            for kind, t, d in lists[k]:
                if kind == 'g':
                    now = max(now, done_sort.get(t, 0.0))
                now += true_cost(kind, d, k % 2 == 0, mode)
                if kind in ('s', 'h'):
                    new_done[t] = max(new_done[t], now)
            fin[k] = now
        if dict(new_done) == done_sort:
            break
        done_sort = dict(new_done)
    busy = [sum(true_cost(kind, d, k % 2 == 0, mode) for kind, t, d in lists[k]) for k in range(S)]
    kmax = max(range(S), key=lambda k: fin[k])
    return {'window': max(fin) / 1e3, 'mean': sum(busy) / S / 1e3,
            'mean_nc': sum(busy[0::2]) / a.cores / 1e3, 'mean_br': sum(busy[1::2]) / a.cores / 1e3,
            'busiest': (kmax // 2, 'NCRISC' if kmax % 2 == 0 else 'BRISC'),
            'idle_on_busiest': (fin[kmax] - busy[kmax]) / 1e3,
            'busiest_list': [(w[0], counts[w[1]]) for w in lists[kmax]],
            'chain': max(done_sort.values()) / 1e3 if done_sort else 0.0}


views = load_views(a.dprint)
print(f'merge={a.merge} us/rec merge_fix={a.merge_fix} us sort={a.sort} us/rec')
res = {}
print(f'mid-tile split threshold: {a.mid or "off"}')
for vi, counts in enumerate(views):
    big = sorted([n for n in counts if n > OV], reverse=True)
    print(f'\nview {vi}: records={sum(counts)} big tiles={big}')
    print(f'  {"schedule":<12} {"window":>7} {"mean":>6} {"nc":>6} {"br":>6} {"lastsort":>8}  busiest (core, mover, idle ms, top items)')
    for mode, pair in (('off', False), ('t168', False), ('split', True), ('split', False), ('free', False)):
        r = schedule(counts, mode, pair)
        name = mode + ('-pair' if pair else ('-any' if mode == 'split' else ''))
        res[(vi, name)] = r
        top = [x for x in r['busiest_list'] if x[0] != 'w'][:4] or r['busiest_list'][:2]
        print(f'  {name:<12} {r["window"]:7.3f} {r["mean"]:6.3f} {r["mean_nc"]:6.3f} {r["mean_br"]:6.3f} '
              f'{r["chain"]:8.3f}  {r["busiest"]} idle={r["idle_on_busiest"]:.3f} n_items={len(r["busiest_list"])} {top}')
print('\nDeltas vs t168 (ms, window):')
for vi in range(len(views)):
    base = res[(vi, 't168')]['window']
    print(f'  view {vi}: ' + '  '.join(f'{k[1]} {res[k]["window"] - base:+.3f}'
                                      for k in res if k[0] == vi and k[1] != 't168'))
