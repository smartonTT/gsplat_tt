# Sort emit straight into the owner core's L1 — model (#385)

**Verdict: shelve.** Modeled saving is 0.04-0.17 ms/view on both bh-30 and the
p100a, under the 0.3 ms gate. No view fits fully in L1. A bounded spill
works, but it needs a smaller sort program and a device-side tile-to-core
assignment, and it still saves too little.

All numbers below are **modeled**, not measured. No device was used.
`docs/p150-tracy-210` has not landed, so sizes come from the 30-view count dump
and existing traces. Run `python3 docs/emit-to-l1-model/model.py` (output in `out.txt`).

## Idea

Today the one-launch sort emit writes each tile's 32 B records into a DRAM
tile bucket (page `tile*513 + k` on the p150, `tile*512 + k` on the p100a).
The fused mat+blend program then reads the bucket back (`read_bucket`,
2 KB pages). The idea is to scatter the records straight into the L1 of the
core that will materialize and blend the tile, skipping the DRAM round trip.

## Inputs

| input | value | source |
|---|---|---|
| per-tile record counts, 30 bicycle views | 2.05M-3.19M records/view, max tile 17.8k-23.0k | `docs/mat-split-sort-model/out/dump.txt.gz` (first line = warmup, skipped) |
| tile-to-core map | replica of `build_mat_worklist` (one-launch, 220 slots, LPT) | `render/host/sort_mover_split.h` |
| usable L1 for CBs | ~1.42 MB/core | `device_state.cpp` worker_l1, +32 KB kcfg; fused CBs fit within ~2.3 KB (#302) |
| mat working set | ~420 KB NCRISC (CB5, CB6, cull) + ~440 KB BRISC (CB20-22), CB4 512 KB bucket | `add_mat_cbs_and_defines`, `sort_device.cpp:519` |
| sort program CBs | 1.06-1.16 MB (2 movers: 266 KB ring + 196 KB window each) | `build_program_sort_onelaunch`, `sort_device.cpp:948` |
| mover bucket read, traced | NCRISC 0.102 ms, BRISC 0.049 ms per view | `docs/fill-zones-t319.md` (p100a, iter 207) |
| emit | TRISC pack-bound, 178.8 cycles/record; movers have ~0.35 ms slack | `docs/emit-town-next-t215/` |
| DRAM write, stride 513 | W256 2.08 / 2.73 B/tick (NoC0/NoC1); reads equal on p150 and p100a | `docs/p150-emit-throughput/` |

## 1. Does it fit? No. A bounded spill does, at a cost.

- Per-core records under the current LPT are **882-1128 KB** on the busiest
  core and 615-956 KB on average. Full residency fits in **0 of 30 views**.
  Even the smallest view's records plus the mat working set (~860 KB) exceed
  1.42 MB.
- Data only survives from the sort program to mat+blend in an L1 region that
  neither program uses for CBs. The sort program already takes 1.06-1.16 MB,
  which leaves 260-360 KB.
- **Spill rule that stays bounded:**
  - a fixed region R of 512 KB per core replaces CB4 in mat+blend
  - tiles over 16384 records (2-5 per view) always stay in DRAM
  - each core fills R greedily with its whole tiles, biggest first; the rest
    goes to the DRAM bucket as today
  - mat processes resident tiles in place first, then streams spilled tiles
    into R
  - result: 57-87 % of whole-tile bytes resident (mean 69 %); at R = 256 KB,
    28-45 % (mean 35 %)
- **What it costs:**
  - The sort program must shrink by 164-264 KB (smaller emit window or rings),
    and that would likely slow the emit.
  - The emit must know each tile's owner core. Today the LPT runs on the host
    after the counts come back (`sort_device.cpp:2623`). It would have to run
    on device in the prefix stage and be replayed exactly on the host, or be
    replaced by a static map that hurts the 98.5 % mat+blend balance.

## 2. NoC cost: no gain on the write side

A remote L1 write costs the same NoC issue as a DRAM write of the same size.
The emit asks for ~0.43 B/tick per core (770 KB over the 1.33 ms window)
against 2.08-2.73 B/tick of DRAM-write capacity at stride 513. Writes are not
the bound: the emit is bound by the TRISC pack (t215). Moving the target to L1
saves **0-0.1 ms** of the emit at most (modeled). The stride-513 fix (#365)
already took the bank-aliasing penalty off the p150.

## 3. Modeled saving

| part | bh-30 (p150) | p100a |
|---|---:|---:|
| mat bucket read avoided: 0.102 ms traced × 69 % resident | ≤ 0.070 | ≤ 0.070 |
| same, untraced (×0.5, t168 ratio) | ≤ 0.035 | ≤ 0.035 |
| emit write side | 0-0.1 | 0-0.1 |
| **total (modeled)** | **0.04-0.17** | **0.04-0.17** |

Reads run at the same speed on both boards, so both boards get the same
saving. This is an upper bound. The critical mat cores (row 2 in t319) are
set by big-tile items, and those stay in DRAM under the spill rule, so the
critical-path gain may be close to zero. The bound also ignores the slower
emit caused by the smaller sort program.

## 4. Is md5 safe? Yes, if built this way

Each record must keep the slot index it has in the DRAM bucket today
(`base[tile][mover] + cursor`). mat's depth radix sort is stable, so ties
keep the same order and the image is byte-identical. Writing in arrival
order instead would break md5.

## Decision

Below the 0.3 ms gate. Implementation is not recommended. The lever is added
to the shelved table in `docs/conclusion.md`. Revisit only if a future trace
shows the mat bucket read at ≥ 0.5 ms per view on the critical core, or if
the sort program's L1 use drops by ~250 KB for another reason.
