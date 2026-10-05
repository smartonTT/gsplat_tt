# Blend stage: what bounds it (task #145, 2026-10-04)

Tip analysed: 5178c67 (one-launch sort v2 default), 24.54 ms/view, board
yyzo-bh-07 (Blackhole p100a). No new device runs: every number below comes
from earlier measured profiles on that board, cited per row. Savings are
**estimates (upper bounds)** unless marked measured.

## Bottleneck

Blend is bound by **SFPU instruction issue on TRISC1 (the math thread)**.
Nothing else is close.

| Quantity | Value | Source |
|---|---|---|
| Blend program window | 7.53 ms/view | docs/lever-a-t121/out/t121-nosel-zones.txt (17.287 -> 24.818 ms) |
| TRISC `tile_blend_sfpu` busy, mean per core | 6.96 ms (92% of window) | same, 25270 ms / 3630 inst. over 11 frames |
| TRISC busy, max core | 7.12 ms | same |
| Tile load imbalance (max - mean) | ~0.16 ms | derived |
| Ramp + tail (window - mean busy) | ~0.57 ms | derived |
| NCRISC reader `tile_blend_load` busy | 6.63 ms, of which ~97% waiting for a free slot | t121 zones; docs/blend-floor-t83 (rd wait 6.48 of 6.67 ms) |
| Compute waiting for data | 0.00 ms | docs/blend-floor-t83 (cmp_bulk_wait) |
| BRISC writer | waits ~6.4 ms; pack 0.61 ms | docs/blend-floor-t83 |
| FPU use | 0% (all math on SFPU) | kernel source; FPU quadratic form rejected in docs/blend-fpu-qf-t111 |
| TRISC0 / TRISC2 | idle apart from the per-tile pack | kernel source |

Where the 6.96 ms of TRISC1 time goes (ablations in docs/blend-floor-t80,
kernel unchanged since apart from off-by-default knobs):

| Part | ms/view | Note |
|---|---|---|
| SFPU bodies (pair 130 insns, single 72) | ~4.0 | ~65 SFPU insns per blended 4x8 microblock |
| Per-record decode + coefficient staging into DEST slot 6 | ~1.76 | ~97 cycles per live record |
| Record loop, mask walk, T readbacks | ~1.0 | ~44 cycles per record |
| Fixed (ramps, acquire, emit) | ~0.7 | |

Cross-check: ~336M blended lanes x ~65 insns / (110 cores x 1.35 GHz x 32
lanes) = ~4.6 ms, close to the 4.0 ms body share. Bodies are issue-bound,
not stalled on latency: padding with NOPs cost +0.89 ms for ~0.86 M extra
issue slots per core (docs/blend-sfpu-mix-t78), so about 1 cycle per SFPU
instruction.

What does **not** bound blend:

- **Data movement / L1 reuse.** Each tile's record slab is read from DRAM
  once into L1 (CB_BUCKET_BULK) and fully hidden behind compute; the
  reader is idle 97% of the time. Output is one u8 pack per tile.
- **Load balance.** Max/mean TRISC = 1.02.
- **T-saturation readback.** `BLEND_T_PERIOD=0` vs default: 10.63 vs
  10.69 ms (docs/blend-fpu-qf-t111), so today it saves nothing.

Work shape (hero view, docs/reprofile-t115, docs/subtile-waste-2026-09-30):
3.37M records, 19.7% dead (mask 0, cost loop time only); live records
cover 3.06 of 32 microblocks; 1.39 microblocks per dispatch; 12.6% of
blended microblocks contributed nothing even at the old floor.

## Levers (max 3)

### Lever 1 - fuse materialize into blend per core (deep; est. upper bound ~3.1 ms)

Today materialize (3.77 ms window, + 0.33 ms host gap) and blend
(7.53 ms) run as two programs. Materialize is mover-bound (BRISC/NCRISC
2.93 ms mean, max ~3.6 ms) with the SFPU cull (~1.2 ms TRISC,
docs/matcull-t86/t90). Blend is TRISC-bound with the movers idle. One
fused per-core program lets the movers materialize tile N+1 into L1 while
TRISC culls and blends tile N.

- Serial today: 3.77 + 0.33 + 7.53 = 11.6 ms.
- Fused critical path: TRISC = cull 1.2 + blend 6.96 = ~8.2 ms; movers =
  ~2.9 + ~0.9 (blend reader) = ~3.8 ms, hidden.
- **Upper bound ~3.1-3.4 ms/view (estimate)**; realistic 2-2.5 ms after
  pipeline fill, tail and big-tile handling. This is the R7 "tile-owner
  L1 pipeline" from docs/OPTIMIZATION-PLAN.md, scoped to the last two
  stages only.
- Meets the 3 ms bar for a deep task.

Measurements it needs before/while building:
1. L1 budget of the fused kernel: mat staging + two blend slabs + cull
   mask (one-launch sort already uses 1.03 of 1.46 MB, but in a different
   program, so it does not count here).
2. Per-tile TRISC cost model (cull + blend, ~records x 97 + live
   microblocks x 65 cycles) to drive LPT; today's LPT is keyed on mover
   cost.
3. Path for tiles larger than one slab (max tile 25,699 records = 822 KB).
4. Gate: md5-identical over 30 views; interleaved A/B rounds on
   yyzo-bh-07; Tracy confirms movers fully hidden.

### Lever 2 - TRISC1 instruction diet, bit-identical (medium; est. upper bound ~3.2 ms, realistic 1.2-1.8 ms)

Since the SFPU costs about 1 cycle per instruction, every removed
instruction pays off directly.

- Hoist the 22 constant SFPLOADIs per pair body and drop the dead clamp.
  **Upper bound ≤0.9 ms (measured** as the NOP-pad cost of the same
  count).
- Cheaper staging: have materialize emit the record already in the form
  DEST slot 6 wants (pre-encoded words), so TRISC1 skips the decode.
  **Upper bound ≤1.76 ms (ablation a3)**, realistic about half.
- Move the T-saturation max reduce from the scalar 1024-iteration loop
  onto the SFPU and shorten the period so it actually skips work. Upper
  bound ~0.5 ms readback cost plus an unmeasured skip gain.
- Compounds with lever 1: after fusion, TRISC1 is the whole critical
  path of the last two stages.

Measurements it needs:
1. Per-part TRISC1 split on the current tip with `GSPLAT_TT_BLEND_PROF=1`
   (body / staging / loop / readback) to refresh the t80 numbers.
2. Microbenchmark whether SFPLOADMACRO or co-issue can hide the const
   loads (unverified; no LLK headers checked locally).
3. Count of dispatches after a microblock saturated (needs lever 3's
   counter) to size the T-reduce gain.
4. Gate: md5-identical over 30 views.

### Lever 3 - measure, then cut wasted work (small task; deep only if counters show ≥3 ms)

Unknown how much body work is wasted at the current 1/255 floor.

- Extend `GSPLAT_TT_MB_STATS=1`: dead microblocks at 1/255, dispatches
  into already-saturated microblocks, dead-record share at the tip.
- Dead records (19.7%) cost only loop time: ≤~0.2 ms (estimate).
- A cheaper exp is worth ≤1.5 ms (estimate, docs/blend-sfpu-mix-t78) but
  is not bit-identical; it needs a golden re-freeze decision and a PSNR
  gate (≥ hero_vs_ref threshold), not md5.

Measurements it needs: the counters above on the 30-view bicycle set;
turn them into ms using ~65 cycles per microblock dispatch.

## Ranking

1. Lever 1 (only lever above the 3 ms deep-task bar on its own).
2. Lever 2 (const hoist first: cheap, measured ceiling, bit-identical).
3. Lever 3 counters (cheap, decide later work).
