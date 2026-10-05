#!/usr/bin/env python3
"""t202: TOWN TRISC counters from a Tracy device CSV (capture with
GSPLAT_TT_OL_EMIT_TOWN=1 GSPLAT_TT_OL_EMIT_PROF=1, see remote_tracy.sh).

Each TRISC of every sort core records "town_pc" timestamped data at its end, value =
(index << 32) | total over that launch (sort_ol_town_compute.cpp TP_*):
  0 tot   wall cycles, kernel start to end      6 rec  records (cursor RMW) done
  1 go    waiting for the first GO               7 run  run words pushed
  2 proc  in process() (all of it)               8 nb   batches done
  3 wfl   of which waiting for fl[t] (run free)  9 ng   blendrec page loads
  4 wq    of which waiting for queue space      10 me   TRISC index
  5 rdy   waiting for a published batch
Gate (task #202): cycles per owned record = (proc - wfl - wq) / rec, over all TRISCs.
  town_parts.py <dev30.csv> [n_views=10]
"""
import sys
from collections import defaultdict

MHZ = 1350.0
DATA_COL, ZONE_COL = 6, 10
NAMES = 'tot go proc wfl wq rdy rec run nb ng me'.split()
CYC = ('tot', 'go', 'proc', 'wfl', 'wq', 'rdy')


def main():
    path = sys.argv[1]
    nv = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    seq = defaultdict(list)  # (core, risc) -> launches (dict name -> value)
    with open(path) as f:
        f.readline()
        f.readline()
        for line in f:
            p = line.rstrip('\n').split(',')
            if len(p) <= ZONE_COL or p[ZONE_COL] != 'town_pc':
                continue
            v = int(p[DATA_COL])
            i, val = v >> 32, v & 0xffffffff
            k = (p[1] + '-' + p[2], p[3])
            if i == 0 or not seq[k]:
                seq[k].append({})
            if i < len(NAMES):
                seq[k][-1][NAMES[i]] = val
    if not seq:
        sys.exit('no town_pc markers: capture with GSPLAT_TT_OL_EMIT_TOWN=1 GSPLAT_TT_OL_EMIT_PROF=1')
    by_risc = defaultdict(list)
    for (core, risc), ls in seq.items():
        for d in ls:
            if len(d) == len(NAMES):
                by_risc[risc].append(d)
    print(f'town_pc: {len(seq)} (core, TRISC) streams, {sum(len(v) for v in by_risc.values())} launches, {nv} views')
    print(f'{"":8s} ' + ' '.join(f'{n:>8s}' for n in NAMES[:10]) + '   (cycle columns: ms per launch, mean over cores)')
    tot = defaultdict(float)
    for risc in sorted(by_risc):
        ls = by_risc[risc]
        n = len(ls)
        row = []
        for nm in NAMES[:10]:
            s = sum(d[nm] for d in ls)
            tot[nm] += s
            row.append(f'{s / n / MHZ / 1000:8.3f}' if nm in CYC else f'{s / n:8.0f}')
        print(f'{risc:8s} ' + ' '.join(row))
        net = sum(d['proc'] - d['wfl'] - d['wq'] for d in ls)
        rec = sum(d['rec'] for d in ls)
        print(f'{"":8s} cycles/record: net {net / max(rec, 1):.1f}, with waits {sum(d["proc"] for d in ls) / max(rec, 1):.1f}')
    net = tot['proc'] - tot['wfl'] - tot['wq']
    print(f'ALL      cycles/owned record (gate <= 270): {net / max(tot["rec"], 1):.1f}  '
          f'(with waits {tot["proc"] / max(tot["rec"], 1):.1f}; wfl {tot["wfl"] / max(tot["rec"], 1):.1f}, '
          f'wq {tot["wq"] / max(tot["rec"], 1):.1f})')
    # Per launch: the slowest TRISC on any core (busy = tot - go - rdy, the TRISC side critical path).
    per = defaultdict(lambda: defaultdict(int))  # launch idx -> core -> max busy
    for (core, risc), ls in seq.items():
        for j, d in enumerate(ls):
            if len(d) == len(NAMES):
                b = d['tot'] - d['go'] - d['rdy']
                per[j][core] = max(per[j][core], b)
    if per:
        mx = [max(c.values()) for c in per.values()]
        mean = [sum(c.values()) / len(c) for c in per.values()]
        print(f'busy (tot - go - rdy) per launch: max core {sum(mx) / len(mx) / MHZ / 1000:.3f} ms, '
              f'mean core {sum(mean) / len(mean) / MHZ / 1000:.3f} ms')


if __name__ == '__main__':
    main()
