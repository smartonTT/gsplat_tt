#!/usr/bin/env python3
"""t183: replay on the t170-on capture (30 views). Per-subchunk TRISC cost from the 2-slot
ring: subchunk j finishes when zone j+2's slot reserve returns (~ e_{j+2} - read). The last
two subchunks per core share tr_end - fin(N-2), split by LAST_SPLIT. Zones with a gap
< GROUP_GAP us after the previous one are later subchunks of the same tile."""
import pickle, sys, statistics as S, argparse
sys.path.insert(0, __import__('os').path.dirname(__file__))
from sim import replay, stats
import random
ap = argparse.ArgumentParser(); ap.add_argument('pkl')
ap.add_argument('--last_split', type=float, default=0.5)
ap.add_argument('--group_gap', type=float, default=1.5)
ap.add_argument('--read', type=float, default=20.0)
ap.add_argument('--split_k', default='0,8,16,32')
ap.add_argument('--half', default='0.5,0.6')
ap.add_argument('--fixed', type=float, default=87.0)
ap.add_argument('--jitter', type=int, default=0, help='average N replays with start/cost jitter')
a = ap.parse_args()
views = pickle.load(open(a.pkl, 'rb'))

def extract(d):
    cores = sorted(d); tiles = []; first_s = []; start = []; meas_seq = []
    for ci, c in enumerate(cores):
        b = d[c]['bulk']; end = d[c]['tr'][1]; N = len(b)
        fin = [None] * N
        for j in range(N - 2): fin[j] = b[j + 2][1] - a.read
        f2 = fin[N - 3] if N >= 3 else b[0][1]
        X = end - f2
        cost = [None] * N
        for j in range(N - 2):
            st = max(fin[j - 1] if j else 0.0, b[j][1]); cost[j] = fin[j] - st
        if N >= 2:
            st = max(f2, b[N - 2][1]); X = end - st
            cost[N - 2] = X * a.last_split; cost[N - 1] = X * (1 - a.last_split)
        else:
            cost[0] = end - b[0][1]
        start.append(b[0][0] - 3.5)
        seq = []
        for j in range(N):
            if j and b[j][0] - b[j - 1][1] < a.group_gap:
                tiles[-1].append(cost[j])
            else:
                tiles.append([cost[j]]); first_s.append(b[j][0]); seq.append(len(tiles) - 1)
        meas_seq.append(seq)
    order = sorted(range(len(tiles)), key=lambda t: first_s[t])
    meas_end = [d[c]['tr'][1] for c in cores]
    return tiles, order, start, meas_end, meas_seq

def split_tiles(tiles, order, k, h, fixed):
    """Split the k most expensive tiles into two 32x16 halves (cost fixed + (c-fixed)*h each);
    halves are claimed back to back at the parent's position."""
    tot = {t: sum(tiles[t]) for t in range(len(tiles))}
    top = set(sorted(tot, key=lambda t: -tot[t])[:k])
    nt = []; no = []; m = {}
    for t in range(len(tiles)):
        if t in top:
            hc = [fixed / len(tiles[t]) + (x - fixed / len(tiles[t])) * h for x in tiles[t]]
            m[t] = [len(nt), len(nt) + 1]; nt += [hc, list(hc)]
        else:
            m[t] = [len(nt)]; nt.append(tiles[t])
    for t in order: no += m[t]
    return nt, no

def rstats(tiles, order, n, pol, start):
    if not a.jitter:
        return stats(replay(tiles, order, n, pol, start, a.read)[0])
    rng = random.Random(7); acc = [0.0, 0.0, 0.0]
    for _ in range(a.jitter):
        st = [x + rng.uniform(-10, 10) for x in start]
        tj = [[x * rng.uniform(0.97, 1.03) for x in t] for t in tiles]
        acc = [p + q for p, q in zip(acc, stats(replay(tj, order, n, pol, st, a.read)[0]))]
    return tuple(x / a.jitter for x in acc)

rows = []
ks = [int(x) for x in a.split_k.split(',')]; hs = [float(x) for x in a.half.split(',')]
for vi, d in enumerate(views):
    tiles, order, start, meas_end, meas_seq = extract(d)
    n = len(start)
    oracle = sorted(range(len(tiles)), key=lambda t: -sum(tiles[t]))
    r = {'meas': stats(meas_end)}
    e, seq = replay(tiles, order, n, 'early', start, a.read); r['early'] = rstats(tiles, order, n, 'early', start)
    r['match'] = sum(1 for c in range(n) if seq[c] == meas_seq[c]) / n
    r['late'] = rstats(tiles, order, n, 'late', start)
    r['early_oracle'] = rstats(tiles, oracle, n, 'early', start)
    r['late_oracle'] = rstats(tiles, oracle, n, 'late', start)
    for k in ks[1:]:
        for h in hs:
            for pol in ('early', 'late'):
                for oname, o in (('ord', order), ('oracle', oracle)):
                    nt, no = split_tiles(tiles, o, k, h, a.fixed)
                    if oname == 'oracle':
                        no = sorted(range(len(nt)), key=lambda t: -sum(nt[t]))
                    r['split%d_h%.1f_%s_%s' % (k, h, pol, oname)] = rstats(nt, no, n, pol, start)
    r['floor_mean'] = (r['meas'][1],)
    rows.append(r)
keys = list(rows[0].keys())
print('n_views %d  read %.0f  last_split %.2f  group_gap %.1f' % (len(rows), a.read, a.last_split, a.group_gap))
print('%-30s %9s %9s %9s %10s' % ('variant', 'max_end', 'mean_end', 'tail', 'd_vs_early'))
base = S.mean(r['early'][0] for r in rows)
for k in keys:
    if k in ('match', 'floor_mean'): continue
    mx = S.mean(r[k][0] for r in rows); mn = S.mean(r[k][1] for r in rows); tl = S.mean(r[k][2] for r in rows)
    print('%-30s %9.1f %9.1f %9.1f %10.1f' % (k, mx, mn, tl, mx - base))
print('replay(early) per-core sequence == measured: %.3f of cores' % S.mean(r['match'] for r in rows))
print('per-view max_end measured vs replay(early) abs err: mean %.1f us' %
      S.mean(abs(r['meas'][0] - r['early'][0]) for r in rows))
