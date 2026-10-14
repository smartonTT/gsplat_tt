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

## Run 2 (0e9335da)

Pending.
