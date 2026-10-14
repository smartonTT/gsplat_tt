# #782 / #787: BWD_SKIPDEAD device A/B (bicycle hero, p150 bh-49)

Arms: base BWD_SKIPDEAD=0, cand BWD_SKIPDEAD=1, both BWD_PRESTAGE=2, BWD_STATS=1, own
TT_METAL_CACHE per arm. Driver docs/t782/ab.sh, summary docs/t782/summarize.py.

## Run 1 (98c4239d, 2026-10-14 15:05-15:31 PDT): the skip never fired

| | base | cand |
|---|---|---|
| hero bench T1 ms (4 rounds ABBA, median of round medians) | 82.41 | 82.96 |
| chain T1 ms med (2 rounds) | 81.41 / 81.42 | 81.70 / 81.76 |
| chain ms med (T1+T2) | 179.75 / 179.79 | 180.09 / 180.11 |
| ST_NV vector bodies | 30,459,432 | 30,459,432 |
| setup cyc/pair | 178.0 | 191.8 |
| per-core imbalance (max/mean tot) | 1.655 | 1.652 |

grads.f32 md5 4ea5c698... and slots9 md5 5857966c... equal on both arms; T0 oracle passes on
sh0/sh3/s1/s2 (both arms) and hero (cand, rel_err 3.27e-3).

ST_NV counts the mask after the dead-vector drop, so equal counts mean the device dropped no
vector, and cand only paid the PACK scan (+0.55 ms). Cause: PACK read CB_LIVE and CB_TL through
`fifo_rd_ptr`, which the firmware sets only on UNPACK (PACK sees 0). PACK scanned L1 from address
0 and published its mask at L1 0/4, never in the mailbox MATH reads. Fix in 0e9335da: UNPACK sends
both addresses to PACK by mailbox (pattern of k2_trisc_compute.cpp); source test
`test_bwd_compute_cb_read_ptr_only_on_unpack` fails on 98c4239d.

## Bodies: plan's ~53M vs 30.5M

runs/2188/t1_dead.json (Mac model, same hero view): 30,459,434 visited vector bodies, matching the
device ST_NV (30,459,432) within 2. Only 1.15M of them have a pixel that is still live and
active. The ~53M figure in the plan does not match any count here; 30.5M is the measured number.
Per-core load is uneven: slowest core (107) runs 492k vectors vs 277k mean (1.78x), and its
time is 1.65x the mean, so T1 is set by the slowest cores.

## Run 2 (0e9335da, 2026-10-14, done 16:18 PDT): keep

| | base | cand |
|---|---|---|
| hero bench T1 ms (4 rounds ABBA, round medians) | 82.40 / 82.42 / 82.42 / 82.41 | 33.23 / 33.39 / 33.23 / 33.19 |
| hero bench T1 ms (median of round medians) | **82.42** | **33.23** (-59.7%) |
| chain T1 ms med (train_step, 2 rounds) | 81.42 / 81.41 | **23.68 / 23.74** |
| chain T2 ms med | 98.45 / 98.45 | 98.39 / 98.45 |
| chain ms med (T1+T2, per hero view) | 179.78 / 179.77 | **122.08 / 122.05** (-57.7 ms, -32%) |
| ST_NV vector bodies | 30,459,432 | 1,561,327 |
| body cyc/pair | 1333.4 | 84.2 |
| setup cyc/pair (incl. PACK dead scan) | 178.0 | 654.3 |
| per-core imbalance (max/mean tot) | 1.655 | 1.275 |

Bit-identical: grads.f32 4ea5c6982acc673a2aa0a047f26c55a7, params_out.f32 27e0856d..., m_out
76076652..., v_out 35601514..., slots9 5857966c97c208449b441243c831c0aa on all four chain runs
(both arms, both rounds), same as run 1. T0 oracle passes on sh0/sh3/s1/s2 (both arms, equal
rel_err) and hero (cand, rel_err 3.27e-3). Small cases: sh0 0.066/0.062, sh3 0.066/0.064, s1
0.435/0.437, s2 0.784/0.785 ms (base/cand; few dead vectors there, the scan roughly breaks even).

JIT: `opt/ab_jit_check.py --kernel blend_bwd_compute` JIT-HASH OK (base hashes 1041766662...,
1160240405..., cand 1841648399..., 3585794814...); per-arm ELF md5s in out2/remote/jit-elf-md5.txt.
The base round medians spread 0.02 ms, so the 49 ms delta is far outside noise.

ST_NV 1.56M vs the Mac model's 1.19M active vectors (runs/2188/t1_dead.json): the device keeps a
vector when any of its pixels is still live in the tile-level test, slightly coarser than the
model's per-pixel active count. The PACK scan moved cost into setup (+476 cyc/pair) but removed
95% of the bodies. T1 is now set by setup and by uneven live-vector counts per core (nv max/mean
4.1x, time max/mean 1.28x).

Verdict: keep. T1 81.4 -> 23.7 ms in the trainer chain, chain 179.8 -> 122.1 ms per hero view,
lossless.

Outputs: out-r1/ and out2/ (logs, timing, stats, md5s, summary.json; slots.f32 dumps dropped).
