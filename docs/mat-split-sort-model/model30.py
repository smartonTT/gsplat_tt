#!/usr/bin/env python3
"""Task #176: 30-view model of the mid-tile split with a dependency-aware host worklist (no device).

Extends model.py (task #175). Same cost terms; new here:
  * per-tile record counts for all 30 bicycle views (GSPLAT_TT_MAT_DUMP lines, out/dump.txt.gz)
  * tip = today's build_mat_worklist (one-launch, select off, record-count LPT, no t144/t168):
      whole item per tile n <= 16384 (host cost n, NCRISC only if n > 6144),
      big tile n > 16384: one item per 8192-record subchunk, each re-reads + re-sorts the tile
      (host cost n + l), NCRISC only. Lists run in LPT order, nothing waits.
  * split = tiles with n > --mid become two half-sorts (keys + ids published) plus one
      merge-gather item per subchunk (NCRISC only; reads both halves, merges its own rank range).
      --big also splits n > 16384 tiles the same way (otherwise they stay as in tip).
  * dep host scheduler: an event-driven list scheduler the host can run. It plans on ESTIMATED
      costs (model terms), half-sorts first (longest tile chain first), then the largest ready
      item; a gather becomes ready when its halves' estimated finish has passed; with nothing
      ready the slot idles to the next estimated event. The device runs each slot's list in order
      on TRUE costs = estimate x (1 + bias + N(0, sigma)); a gather spins until both halves are done.
      sigma per kind from t144's fit residuals (whole <= 6144: 4 %, whole > 6144: 5 %, big
      subchunk: 12 %); half-sort and merge-gather were never measured: 12 % plus --bias.
Output: per-view traced window (busiest slot finish) for tip and split, the saving, and the
untraced saving = traced x --ratio (0.5, measured in t168).
"""
import argparse, gzip, random, re, statistics as st
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument('--dump', default='docs/mat-split-sort-model/out/dump.txt.gz')
ap.add_argument('--dprint', default='', help='use t147 TC dprint (views 0,1) instead of the dump')
ap.add_argument('--merges', default='0.010,0.020,0.030')
ap.add_argument('--merge_fix', type=float, default=10.0)
ap.add_argument('--sort', type=float, default=0.0829)
ap.add_argument('--cores', type=int, default=110)
ap.add_argument('--mid', type=int, default=8192)
ap.add_argument('--big', type=int, default=1, help='1: also split n > 16384 tiles')
ap.add_argument('--bias', type=float, default=0.0, help='true/est - 1 for half-sort + merge-gather')
ap.add_argument('--seeds', type=int, default=8)
ap.add_argument('--sigma', type=float, default=1.0, help='scale of the per-item noise (0 = true = estimate)')
ap.add_argument('--ratio', type=float, default=0.5)
ap.add_argument('--ofit', default='model', choices=['model', 't144'],
                help='true cost of a tip big-tile subchunk item: model.py terms or the t144 fit')
ap.add_argument('--per_view', type=int, default=1)
a = ap.parse_args()

FIT, OV, M0 = 8192, 16384, 6144
BRISC_HALF_CAP = 2 * M0 + 256
K_KEYS, K_PUB, K_GATH, K_OWN, IDS = 0.01528, 0.0081, 0.01476, 0.059, 1.3
SIG = {'w': 0.04, 'W': 0.05, 'o': 0.12, 'h': 0.12, 'g': 0.12}
S = 2 * a.cores


def load_dump(path):
    op = gzip.open if path.endswith('.gz') else open
    views = []
    for line in op(path, 'rt'):
        if not line.startswith('MATCOUNTS') or ' ol=1 ' not in line:
            continue
        views.append([int(tok.split(':')[1]) for tok in line.split()[3:]])
    return views


def load_dprint(path):
    per_core = defaultdict(lambda: [[]])
    for line in open(path, errors='replace'):
        m = re.search(r'^\d+:(\d+)-(\d+):TR1:\s*TC(END)?\s+(\d+)', line)
        if not m:
            continue
        core = (int(m.group(1)), int(m.group(2)))
        if m.group(3):
            per_core[core].append([])
        else:
            per_core[core][-1].append(int(m.group(4)))
    nl = min(len(v) - 1 for v in per_core.values())
    return [[n for v in per_core.values() for n in v[L]] for L in range(1, nl)]


def whole_cost(n, nc):
    if n <= M0:
        return (5.5 + 0.197 * n) if nc else (16.1 + 0.187 * n)
    return -282 + 0.2226 * n


def est_cost(kind, d, nc, merge):
    """Model cost in us (also the host's estimate)."""
    if kind == 'w':
        return whole_cost(d, nc)
    if kind == 'o':
        n, l = d
        if a.ofit == 't144':
            return 0.1083 * n + 0.1267 * l
        return (K_KEYS + a.sort + K_GATH) * n + (K_OWN + 0.002) * l
    if kind == 'h':
        n, nh = d
        return (K_KEYS + a.sort + 2 * K_PUB) * nh + 5.0
    if kind == 'g':
        n, l = d
        return K_GATH * n + K_OWN * l + IDS + merge * l + a.merge_fix


def items_for(counts, split):
    """[(id, kind, tile, data, big)]; big = NCRISC only."""
    it = []
    for t, n in enumerate(counts):
        do_split = split and n > a.mid and (n <= OV or a.big)
        if not do_split:
            if n <= OV:
                it.append(['w', t, n, n > M0])
            else:
                for sc in range(-(-n // FIT)):
                    it.append(['o', t, (n, min(FIT, n - sc * FIT)), True])
            continue
        h0 = n // 2
        for nh in (h0, n - h0):
            it.append(['h', t, (n, nh), nh > BRISC_HALF_CAP])
        for sc in range(-(-n // FIT)):
            it.append(['g', t, (n, min(FIT, n - sc * FIT)), True])
    return [(i, k, t, d, b) for i, (k, t, d, b) in enumerate(it)]


def noise_key(kind, d):
    return 'W' if kind == 'w' and d > M0 else kind


def true_costs(items, merge, rng):
    """per item: (cost on NCRISC, cost on BRISC); one draw per item (the same work on either RISC)."""
    out = {}
    for i, kind, t, d, big in items:
        e = 1.0 + rng.gauss(0.0, SIG[noise_key(kind, d)] * a.sigma)
        if kind in ('h', 'g'):
            e += a.bias
        e = max(e, 0.3)
        out[i] = (est_cost(kind, d, True, merge) * e, est_cost(kind, d, False, merge) * e)
    return out


def tip_lists(items):
    """build_mat_worklist today: sort by host cost desc, least-loaded eligible slot (ties lowest)."""
    def hc(x):
        kind, d = x[1], x[3]
        return d if kind == 'w' else d[0] + d[1]
    order = sorted(items, key=lambda x: -hc(x))  # std::sort: ties unspecified; stable here
    load = [0] * S
    lists = [[] for _ in range(S)]
    for x in order:
        step = 2 if x[4] else 1
        c = min(range(0, S, step), key=lambda k: load[k])
        lists[c].append(x)
        load[c] += hc(x)
    return lists


def dep_lists(items, merge):
    """Host dependency-aware list scheduler on estimated costs."""
    est = {x[0]: (est_cost(x[1], x[3], True, merge), est_cost(x[1], x[3], False, merge)) for x in items}
    chain = defaultdict(float)
    for x in items:
        if x[1] in ('h', 'g'):
            chain[x[2]] = max(chain[x[2]], est[x[0]][0])
    halves = sorted([x for x in items if x[1] == 'h'],
                    key=lambda x: (-(chain[x[2]] + est[x[0]][0]), x[0]))
    rest = sorted([x for x in items if x[1] != 'h'], key=lambda x: (-est[x[0]][0], x[0]))
    need = defaultdict(int)
    for x in halves:
        need[x[2]] += 1
    ready_at = {}
    part = defaultdict(float)
    free = [0.0] * S
    lists = [[] for _ in range(S)]
    import heapq
    heap = [(0.0, k) for k in range(S)]
    left = len(halves) + len(rest)
    while left:
        now, k = heapq.heappop(heap)
        nc = k % 2 == 0
        pick = None
        for x in halves:
            if nc or not x[4]:
                pick = x
                break
        if pick is not None:
            halves.remove(pick)
        else:
            for x in rest:
                if x[4] and not nc:
                    continue
                if x[1] == 'g' and ready_at.get(x[2], float('inf')) > now + 1e-9:
                    continue
                pick = x
                break
            if pick is not None:
                rest.remove(pick)
        if pick is None:
            nxt = [t for t in ready_at.values() if t > now + 1e-9] + [h[0] for h in heap if h[0] > now + 1e-9]
            if not nxt:  # only BRISC-ineligible work left
                continue
            heapq.heappush(heap, (min(nxt), k))
            continue
        left -= 1
        lists[k].append(pick)
        end = now + est[pick[0]][0 if nc else 1]
        if pick[1] == 'h':
            part[pick[2]] = max(part[pick[2]], end)
            need[pick[2]] -= 1
            if need[pick[2]] == 0:
                ready_at[pick[2]] = part[pick[2]]
        heapq.heappush(heap, (end, k))
    return lists


def run_device(lists, tc):
    """Each slot runs its list in order; a gather waits for its tile's halves. Returns window, mean busy."""
    nh = defaultdict(int)
    for v in lists:
        for x in v:
            if x[1] == 'h':
                nh[x[2]] += 1
    done_h = defaultdict(list)
    ptr = [0] * S
    now = [0.0] * S
    busy = [0.0] * S
    progress = True
    while progress:
        progress = False
        for k in range(S):
            v = lists[k]
            while ptr[k] < len(v):
                x = v[ptr[k]]
                if x[1] == 'g':
                    if len(done_h[x[2]]) < nh[x[2]]:
                        break
                    now[k] = max(now[k], max(done_h[x[2]]))
                c = tc[x[0]][0 if k % 2 == 0 else 1]
                now[k] += c
                busy[k] += c
                if x[1] == 'h':
                    done_h[x[2]].append(now[k])
                ptr[k] += 1
                progress = True
    assert all(ptr[k] == len(lists[k]) for k in range(S)), 'deadlock'
    return max(now) / 1e3, sum(busy) / S / 1e3


views = load_dprint(a.dprint) if a.dprint else load_dump(a.dump)
merges = [float(m) for m in a.merges.split(',')]
if a.sigma == 0:
    a.seeds = 1
print(f'views={len(views)} mid={a.mid} big_split={a.big} bias={a.bias:+.2f} seeds={a.seeds} sigma={a.sigma} '
      f'ofit={a.ofit} merge_fix={a.merge_fix} ratio={a.ratio}')
nmid = [sum(1 for n in c if a.mid < n <= OV) for c in views]
nbig = [sum(1 for n in c if n > OV) for c in views]
print(f'tiles/view: mid ({a.mid},16384] mean {st.mean(nmid):.1f} (min {min(nmid)} max {max(nmid)}), '
      f'big >16384 mean {st.mean(nbig):.1f} (max {max(nbig)}), records/view mean '
      f'{st.mean(sum(c) for c in views):.0f}')
summary = {}
for merge in merges:
    rows = []
    for vi, counts in enumerate(views):
        tip_it = items_for(counts, False)
        sp_it = items_for(counts, True)
        tl = tip_lists(tip_it)
        dl = dep_lists(sp_it, merge)
        tw, sw, tm, sm = [], [], [], []
        for s in range(a.seeds):
            rng = random.Random(1000 * vi + s)
            w, m = run_device(tl, true_costs(tip_it, merge, rng))
            tw.append(w); tm.append(m)
            rng = random.Random(1000 * vi + s)
            w, m = run_device(dl, true_costs(sp_it, merge, rng))
            sw.append(w); sm.append(m)
        rows.append((vi, st.mean(tw), st.mean(sw), st.mean(tm), st.mean(sm), max(counts)))
    summary[merge] = rows
    sav = [r[1] - r[2] for r in rows]
    print(f'\n## merge {merge:.3f} us/rec')
    if a.per_view:
        print(f'  {"view":>4} {"maxtile":>7} {"tip_win":>7} {"split_win":>9} {"tip_mean":>8} '
              f'{"split_mean":>10} {"save_traced":>11} {"save_untraced":>13}')
        for (vi, t, s_, tm, sm, mx), d in zip(rows, sav):
            print(f'  {vi:4d} {mx:7d} {t:7.3f} {s_:9.3f} {tm:8.3f} {sm:10.3f} {d:+11.3f} {d * a.ratio:+13.3f}')
    print(f'  MEAN tip window {st.mean(r[1] for r in rows):.3f}  split window {st.mean(r[2] for r in rows):.3f}  '
          f'tip mean load {st.mean(r[3] for r in rows):.3f}  split mean load {st.mean(r[4] for r in rows):.3f} ms traced')
    print(f'  MEAN tip (window - mean load) {st.mean(r[1] - r[3] for r in rows):.3f} ms traced '
          f'(measured t170 tracy-on: mat window 3.376 - mean mover zone 2.758 = 0.618, incl. launch skew)')
    print(f'  MEAN saving traced {st.mean(sav):+.3f} ms/view, untraced {st.mean(sav) * a.ratio:+.3f} ms/view '
          f'(min {min(sav) * a.ratio:+.3f}, max {max(sav) * a.ratio:+.3f}; '
          f'{sum(1 for d in sav if d < 0)} of {len(sav)} views slower)')
