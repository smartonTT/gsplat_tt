#!/usr/bin/env python3
"""Task #385: can the sort emit write records straight into the owner core's L1?

Model only (no device). Inputs: the 30-view per-tile record counts dumped by
build_mat_worklist (docs/mat-split-sort-model/out/dump.txt.gz, first line is
warmup), the one-launch LPT replicated from render/host/sort_mover_split.h, and
traced zone times from docs/fill-zones-t319.md (p100a, iter 207).
All outputs are MODELED numbers.
"""
import gzip, statistics as st, sys

DUMP = 'docs/mat-split-sort-model/out/dump.txt.gz'
REC_B = 32                      # bytes per sort record
CORES = 110                     # 11 x 10 render grid (both boards)
MOVERS = 2                      # slot 2c = NCRISC, 2c+1 = BRISC
BUCKET_FIT = 8192               # render_config::kBucketFit
OV_CAP = 16384                  # render_config::kOverflowL1Cap
M0_CAP = 6144                   # kMatMover0Cap
L1_USABLE = 1.42e6              # usable CB bytes per core (device_state worker_l1, +32 KB kcfg)
MAT_WS = 0.42e6 + 0.44e6        # mat working set: NCRISC (CB5+CB6+cull) + BRISC (CB20/21/22)
SORT_PROG = (1.06e6, 1.16e6)    # sort program CBs (2 movers x 521,280 / 567,680 + shared)
# traced per-view mover bucket-read time (fill-zones-t319, p100a, ms)
RD_NC, RD_BR = 0.102, 0.049
UNTRACED = 0.5                  # untraced/traced zone ratio (t168)


def load():
    views = []
    for line in gzip.open(DUMP, 'rt'):
        if line.startswith('MATCOUNTS') and ' ol=1 ' in line:
            c = [0] * 1024
            for tok in line.split()[3:]:
                t, n = tok.split(':'); c[int(t)] = int(n)
            views.append(c)
    return views[1:]


def lpt(counts):
    """Replicates build_mat_worklist(onelaunch=True, ol_select=False)."""
    items = []
    for t, cnt in enumerate(counts):
        if cnt == 0:
            continue
        if cnt <= OV_CAP:
            items.append((cnt, t, 0, cnt > M0_CAP))
            continue
        nsc = (cnt + BUCKET_FIT - 1) // BUCKET_FIT
        for sc in range(nsc):
            l_sub = min(BUCKET_FIT, cnt - sc * BUCKET_FIT)
            items.append((cnt + l_sub, t, sc, True))
    items.sort(key=lambda x: -x[0])          # std::sort desc (ties: model only)
    slots = CORES * MOVERS
    load = [0] * slots
    owner = {}                                # whole tile -> slot
    big_slot = {}
    for cost, t, sc, big in items:
        step = 2 if big else 1
        c = 0
        for k in range(step, slots, step):
            if load[k] < load[c]:
                c = k
        load[c] += cost
        if counts[t] <= OV_CAP:
            owner[t] = c
        else:
            big_slot.setdefault(t, []).append(c)
    return owner, big_slot


def main():
    views = load()
    out = []
    p = out.append
    p(f'views={len(views)} (MODELED, 32 B/record, {CORES} cores x {MOVERS} movers)')
    hdr = ('view recs  maxtile >16k  core_max_KB core_mean_KB  big_KB_maxcore '
           'res256%  res512%')
    p(hdr)
    agg = {256: [], 512: []}
    fit_full = 0
    core_maxes = []
    for vi, c in enumerate(views):
        owner, big_slot = lpt(c)
        per_core = [0] * CORES          # all records the core's mat reads
        whole = [[] for _ in range(CORES)]
        nc_whole = [0] * CORES
        for t, s in owner.items():
            per_core[s // 2] += c[t] * REC_B
            whole[s // 2].append(c[t] * REC_B)
            if s % 2 == 0:
                nc_whole[s // 2] += c[t] * REC_B
        big_b = [0] * CORES
        for t, sl in big_slot.items():
            # every subchunk item reads the whole tile (cnt + l_sub cost)
            for s in sl:
                big_b[s // 2] += c[t] * REC_B
        for i in range(CORES):
            per_core[i] += big_b[i]
        cmax = max(per_core)
        core_maxes.append(cmax)
        if cmax + MAT_WS <= L1_USABLE:
            fit_full += 1
        tot_whole = sum(sum(w) for w in whole)
        res = {}
        for R in (256, 512):
            cap = R * 1024
            kept = 0
            for w in whole:
                room = cap
                for b in sorted(w, reverse=True):   # greedy, biggest tiles first
                    if b <= room:
                        room -= b; kept += b
            res[R] = kept / tot_whole
            agg[R].append(res[R])
        p(f'{vi:4d} {sum(c):8d} {max(c):6d} {sum(1 for x in c if x > OV_CAP):4d} '
          f'{cmax/1024:10.0f} {st.mean(per_core)/1024:11.0f} {max(big_b)/1024:12.0f} '
          f'{100*res[256]:7.1f} {100*res[512]:7.1f}')
    p('')
    p(f'full residency fits (core_max + mat working set {MAT_WS/1e3:.0f} KB <= '
      f'{L1_USABLE/1e3:.0f} KB): {fit_full}/{len(views)} views; '
      f'core_max range {min(core_maxes)/1024:.0f}-{max(core_maxes)/1024:.0f} KB')
    p(f'sort program + R must fit in {L1_USABLE/1e3:.0f} KB: free beside sort program = '
      f'{(L1_USABLE-SORT_PROG[1])/1e3:.0f}-{(L1_USABLE-SORT_PROG[0])/1e3:.0f} KB '
      f'-> R=512 KB needs the sort program to shrink by '
      f'{(512*1024-(L1_USABLE-SORT_PROG[0]))/1e3:.0f}-{(512*1024-(L1_USABLE-SORT_PROG[1]))/1e3:.0f} KB')
    for R in (256, 512):
        p(f'R={R} KB: resident share of whole-tile bytes mean {100*st.mean(agg[R]):.1f}% '
          f'min {100*min(agg[R]):.1f}% max {100*max(agg[R]):.1f}%')
    f = st.mean(agg[512])
    fn = f  # R is shared by both movers of a core
    sv_tr = max(RD_NC * fn, RD_BR * f)
    p('')
    p('mat-side saving (MODELED): upper bound = traced mover bucket-read time x resident share')
    p(f'  traced (p100a zones): NCRISC {RD_NC} x {fn:.2f} = {RD_NC*fn:.3f} ms, '
      f'BRISC {RD_BR} x {f:.2f} = {RD_BR*f:.3f} ms -> per-view <= {sv_tr:.3f} ms')
    p(f'  untraced (x{UNTRACED}): <= {sv_tr*UNTRACED:.3f} ms')
    p('  bh-30 vs p100a: same (R256 reads measured equal on both boards, p150-emit-throughput)')
    emit_demand = 770e3 / (1.33e-3 * 1.35e9)   # bytes per core per tick over the emit window
    p(f'emit write demand ~{emit_demand:.2f} B/tick/core vs DRAM-write W256@513 2.08-2.73 '
      f'B/tick: write channel is not the bound (emit is TRISC pack-bound, t215) '
      f'-> emit saving 0-0.1 ms (MODELED)')
    p(f'total MODELED saving per view: {sv_tr*UNTRACED:.2f}-{sv_tr+0.1:.2f} ms (gate 0.3 ms)')
    print('\n'.join(out))


if __name__ == '__main__':
    main()
