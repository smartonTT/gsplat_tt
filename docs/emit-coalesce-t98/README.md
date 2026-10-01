# Task #98: coalesce sort_bucket_emit record writes — gate check (no fix)

Board **yyzo-bh-07 (Blackhole p100a)**, bicycle 30 views 1024x1024, untraced
`render/run.py --no-ref`. Base = 08f9200 (tip of `smarton/tt-project-opt`);
tree `/localdev/smarton/gstt2-t98` at 850fa8d / 7d261ca (08f9200 + default-off
ablations). All four base runs: 30/30 views md5-identical to `md5-r82new.txt`,
hero_vs_ref 100 dB, bin_emit 8.821-8.825 ms. Logs: `abl-r1..3.log`.

## Verdict

**Gate fails: coalescing the 32 B record writes is worth at most 1.18 ms/view**
(spec gate: 3 ms). Removing every record write outright only takes bin_emit from
8.82 to 7.64 ms. So no coalescing was implemented.

## Tool

`GSPLAT_TT_EMIT_ABLATE=<mask>` (profiling only; any bit set makes the output
wrong; unset builds the unchanged kernel) removes parts of the emit in
`sort_bin.cpp`:

| bit | removes |
|---|---|
| 1 | 32 B record writes (`flush_recs`) |
| 2 | 16 B packed op/color writes to blendrec (`flush_packoc`) |
| 4 | `pack_invariants` + `pack_rec` compute |
| 8 | blendrec prefetch reads (one 64 B read per gaussian, one barrier per pair page) |
| 16 | depth page reads |
| 32 | the whole per-pair loop (pair pages and blendrec prefetch still read) |

`remote_job.sh <round> base a1 a2 ...` runs them (aN = mask N).

## Results (SORT_STAGES bin_emit, ms/view; host timer, ~0.8 ms above the Tracy zone)

| run | removed | bin_emit | vs base |
|---|---|---:|---:|
| base (x4) | - | **8.82** | |
| a1 | record writes | 7.64 | **-1.18** |
| a2 | packoc writes | 8.14 | -0.68 |
| a3 | both writes | 7.21 | -1.61 |
| a8 | blendrec reads | 7.72 | -1.11 |
| a16 | depth reads | 8.55 | -0.28 |
| a4 (x2) | pack compute | 9.22, 9.21 | **+0.40** |
| a12 | pack + blendrec reads | 6.90 | -1.93 |
| a11 | writes + blendrec reads | 6.30 | -2.52 |
| a27 | writes + blendrec + depth reads (pack kept) | 6.14 | -2.69 |
| a31 | a27 + pack compute | 2.58 | -6.25 |
| a32 | per-pair loop | 4.19 | -4.64 |
| a40 | per-pair loop + blendrec reads (pair page reads only) | 1.96 | -6.86 |

Ablations are deterministic to ~0.01 ms (base 8.821-8.825 over 3 rounds, a4
9.222 / 9.213).

## What it means

1. **The emit is bound by its DRAM traffic, not its compute.** Without the extra
   traffic (a27) the pack compute costs 3.56 ms (a27 - a31; the #27 sum zones
   measured 3.4 ms). With the traffic it costs nothing: removing it (a4) makes
   the program 0.4 ms *slower*. Likely mechanism (not traced): writes are
   posted, so the movers pack while earlier batches drain, and the barriered
   reads wait behind that traffic. Without the compute the same traffic just
   arrives in denser bursts.
2. Traffic savings add up (writes 1.61 + blendrec 1.11 + depth 0.28 = 3.0 vs
   2.69 together) until they reach the **6.1 ms compute floor** (a27: pair-page
   reads + bookkeeping + pack). **Cutting traffic alone tops out at ~2.7 ms**,
   still under the gate.
3. A >= 3 ms win needs traffic and compute cut **together**. The joint upper
   bound is a31: 8.82 -> 2.58 ms (-6.25 ms/view). What is left there is
   pair-page reads plus fixed cost (a40, 1.96 ms) and 0.6 ms of per-pair
   bookkeeping.

## Candidate levers for a combined emit rewrite (upper bounds from the table)

| lever | bound | note |
|---|---:|---|
| coalesce record writes (per-tile L1 staging runs) | 1.18 | this task's idea |
| drop the packoc writes: have the producer publish packed op/color | 0.68 | also removes most of `pack_invariants` (#27: 1.8 ms of compute) |
| fewer blendrec read transactions (larger blendrec pages, batch over the core's contiguous gaussian range) | 1.11 | a core's gids are non-decreasing |
| batch the pair-page reads (one barrier per page today; count pass already batches 32) | part of 1.96 | |
| `pack_rec` (2 x sub_int + 8 stores per pair) | ~1.6 | #27 sum zone; only shows once traffic is gone |

Each one alone is under 3 ms; the traffic levers saturate at the 6.1 ms compute
floor unless the pack compute drops too. Gate each step with the ablation
switch.

Commits: 850fa8d (ablation switch + job script), 7d261ca (bit 32), this README.
