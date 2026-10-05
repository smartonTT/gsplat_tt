#!/usr/bin/env python3
"""t189: blend per-dispatch cost model. Inputs: t172 view-0 counts and fit, t148 pair ratio,
body instruction counts from count_bodies.py (out/body_counts.txt). No device data of its own."""
MHZ, CORES = 1350.0, 110
rec, live, mb = 2_895_701, 2_345_549, 6_584_240          # t172 tc_split.txt launch 0 (default code)
fit_rec, fit_live, fit_mb = 0.0161, 0.0801, 0.0610         # us each, t172 fit (rms 14.7 us/tile)
loop_ms = 5.683                                            # t172 loop(records) ms per core
mb_per_call = 1.275                                        # t148 mb_disp / pairops (blend loop unchanged since)
single, pair = 60, 109                                     # issued SFPU ops (pair = 104 + 5 replayed)
nop_s, nop_p = 5, 8
walk = 13                                                  # ctz andi srl andi sh1add sh2add lw jalr | ret | sll andn bnez
rec_head, stage = 11, 92                                   # issued insns: dead-record loop head, live staging (q)

calls = 1 / mb_per_call                                    # body calls per microblock
f_pair = 2 * (1 - calls)                                   # share of microblocks run in pair bodies
f_single = 1 - f_pair
per_core = lambda n: n / CORES
ms_per_cyc = lambda n: per_core(n) / MHZ / 1e3             # ms/view for 1 cycle per item
cyc_mb = fit_mb * MHZ
sfpu = f_single * (single - nop_s) + f_pair * (pair - nop_p) / 2
nops = f_single * nop_s + f_pair * nop_p / 2
walk_mb = calls * walk
resid = cyc_mb - sfpu - nops - walk_mb
k = ms_per_cyc(mb)
print(f'per core: {per_core(mb):.0f} microblocks, {per_core(mb*calls):.0f} body calls, '
      f'{per_core(live):.0f} live recs, {per_core(rec):.0f} recs')
print(f'1 cycle per microblock = {k:.4f} ms/view; per call = {ms_per_cyc(mb*calls):.4f}; '
      f'per live rec = {ms_per_cyc(live):.4f}')
print(f'calls/mb {calls:.3f}; microblocks in pair bodies {f_pair:.1%}, in single bodies {f_single:.1%}')
print(f'\nper microblock: fit {cyc_mb:.1f} cyc = {cyc_mb*k:.2f} ms/view')
for n, c in [('SFPU work (loads, math, stores)', sfpu), ('sfpnop', nops),
             ('walk RISC insns', walk_mb), ('unattributed stall', resid)]:
    print(f'  {n:34s} {c:5.1f} cyc  {c*k:.2f} ms/view')
print(f'  non-SFPU time per body call: {(cyc_mb - sfpu - nops)/calls:.1f} cyc for {walk} insns')
issued = per_core(rec)*rec_head + per_core(live)*stage + per_core(mb)*(sfpu+nops+walk_mb)
tot = loop_ms * MHZ * 1e3
print(f'\nloop: {tot/1e6:.2f} Mcyc/core measured, {issued/1e6:.2f} Mcyc issued, '
      f'stall {(tot-issued)/1e6:.2f} Mcyc = {(tot-issued)/MHZ/1e3:.2f} ms ({1-issued/tot:.0%})')
kc = ms_per_cyc(mb * calls)
print('\nlevers (ms/view, upper bounds unless noted):')
L = [('fill every sfpnop', nops * k),
     ('drop duplicate k99/floor loads in pair bodies', f_pair / 2 * 2 * k),
     ('tail-chained walk, low (2 insns + 2 transfers x1 cyc, no load hide)', 4 * kc),
     ('tail-chained walk, mid (2 + 2x3 + load-use hidden 1)', 9 * kc),
     ('tail-chained walk, high (2 + 2x4 + load-use hidden 5)', 15 * kc),
     ('every single body turned into a pair (impossible: needs 496 bodies)',
      f_single * (single - pair / 2) * k + f_single / 2 * (cyc_mb - sfpu - nops) / calls * k),
     ('all walk RISC + all unattributed stall gone (ceiling)', (walk_mb + resid) * k)]
for n, v in L:
    print(f'  {n:72s} {v:.2f}')
