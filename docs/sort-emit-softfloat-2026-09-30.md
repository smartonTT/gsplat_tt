# sort_bucket_emit soft-float audit (task #27, 2026-09-30)

Board: **yyzo-bh-07, Blackhole p100a** (not a p150). Scene: bicycle, 30 views, 1024x1024.
Base: `smarton/tt-project-opt` @ cc9399c (T-B, T-C and task #30 landed). The first A/B and
the sum-zone capture were run on 7df44de; the A/B was repeated after rebasing onto cc9399c.

## Finding: the soft-float was already gone from the hot path

`docs/hw-ceilings.md` (task #21) estimated that ~40 % of the emit's cycles per pair were
soft-float in `pack_rec` / `pack_invariants`. That estimate was made against the pre-T-B
code. T-B(3) (b13b2cc, iter-146) had already replaced both expressions with the
integer, bit-exact helpers in `render/kernels/dataflow/sort_bin_fp32.h`:

- `to_unorm` (4 per gaussian) -> `sort_bin_fp32::unorm16`, float path only for NaN;
- `mean - (float)(txi*32)` (2 per pair) -> `sort_bin_fp32::sub_int`, float path only for
  |mean| < 2^-17 or >= 2^24.

Disassembly of the compiled `sort_bin` ELFs (NCRISC and BRISC, JIT cache of the T-C
build, `riscv-tt-elf-objdump`; ISA rv32im + Zba/Zbb, no F) shows exactly 24 soft-float
call sites: 4 x (`__lesf2`, `__gesf2`, `__mulsf3`, `__addsf3`, `__fixunssfsi`) for the 4
`to_unorm` NaN fallbacks and 2 x (`__floatunsisf`, `__subsf3`) for the 2 mean fallbacks.
None of them is reached for finite on-screen data. So the "float math" share of the
emit is already ~0.

## What this task changed

1. **32-bit fast paths** in `sort_bin_fp32.h`. T-B(3)'s helpers did their rounding on
   `uint64_t` (rv32: every shift/compare is 2-4 instructions). Now:
   - `unorm16`: only the 24x16-bit product is 64-bit (mul + mulhu); rounding runs on
     32 bits (low byte folded into a sticky bit), `clz` (Zbb) replaces the bit-length loop.
   - `sub_int`: `k == 0` returns `a`; for `|a| >= 4` and tile origins `k <= 1016` (all of
     a 1024 px frame) the difference fits int32 and rounds on 32 bits; the 64-bit path
     stays for the rest.
   Bit-exactness: `tests/unit/test_sort_bin_fp32.cpp` (now also full 2^32 sweeps at
   k = 1016, the last 32-bit k, and k = 1024, the first past it): unorm16 4,278,190,082
   inputs, sub_int 4,447,490,326 inputs, **0 mismatches**.
2. **Accumulating sub-zones** `emit_pack_invariants` (`DeviceZoneScopedSumN1`) and
   `emit_pack_rec` (`DeviceZoneScopedSumN2`) around the two pack calls. They compile to
   nothing unless `TT_METAL_DEVICE_PROFILER=1` and `TT_METAL_PROFILER_SUM=1`; then each
   RISC reports one `ZONE_TOTAL` row per launch. Analysis script:
   `opt/profiler/emit_sumzones.py <profile_log_device.csv>`.

## Measured

### Pack share of the emit (Tracy, 10 views, TT_METAL_PROFILER_SUM=1)

Per RISC per launch, mean over 110 cores x 2 movers x 10 views (warmup dropped):

| zone | ms | share of the 7.99 ms emit zone | empty-zone calibration |
|---|---:|---:|---:|
| `emit_pack_invariants` (4 x unorm16 + 9 L1 loads, per gaussian) | 1.81 | **22.7 %** | 0.052 ms (0.7 %) |
| `emit_pack_rec` (2 x sub_int + 8 stores, per pair) | 1.59 | **19.8 %** | 0.025 ms (0.3 %) |

The calibration run moved each zone onto an empty `asm volatile("")` at the same call
site, so the timer itself adds < 1 %. At K = 1.71 pairs per gaussian and ~14 k pairs per
mover (~3.08 M pairs / 220 movers), that is roughly 300 cycles per gaussian and 150 cycles
per pair, all integer work plus L1 loads/stores (no soft-float calls on this path).

### A/B (interleaved, 3 rounds, --no-ref, bicycle 30 views)

| variant | avg_frame_ms (r1, r2, r3) | mean | stage_sort_bin_emit ms | mean | stage_sort ms |
|---|---|---:|---|---:|---:|
| base cc9399c | 121.698, 122.181, 122.072 | 121.98 | 8.993, 8.999, 8.999 | 8.997 | 18.781 |
| t27 on cc9399c | 121.555, 121.590, 121.560 | 121.57 | 8.852, 8.848, 8.849 | **8.850** | 18.602 |
| base 7df44de | 145.571, 145.342, 145.185 | 145.37 | 8.995, 9.002, 8.994 | 8.997 | 18.752 |
| t27 on 7df44de | 145.169, 145.211, 145.782 | 145.39 | 8.845, 8.842, 8.835 | 8.841 | 18.635 |

Emit -0.15 ms/view (-1.6 %), ranges disjoint on both bases. The frame moves -0.42 ms on
cc9399c and +0.02 ms on 7df44de, both inside the ~0.3-0.5 ms round-to-round spread, so
only the ~0.15 ms emit/sort share is attributable.

All 30 dumped views byte-identical to the base build; hero md5 `e3fefb11...`,
hero_vs_ref 100.00 dB.

## Read

1. The lever this task targeted had already landed in T-B(3). The ~40 % "float math"
   share in `docs/hw-ceilings.md` described the pre-T-B code; today it is ~0.
2. Rounding on 32 bits instead of 64 bits saves 0.16 ms/view of emit, below the 1 %
   frame threshold. It is kept because it is bit-exact, exhaustively tested and free.
3. The two pack calls still take ~42 % of the emit on each mover (~3.4 ms of ~8 ms).
   That is integer work, captured-variable loads and the 8 record stores, not float.
   Halving it would save at most ~1.7 ms/view (~1.2 % of the frame). Cheapest routes if
   the emit becomes the long pole again: have gather write the packed op/color words so
   `pack_invariants` only copies them (it already publishes them for materialize), and
   keep `pack_rec`'s inputs in registers (the lambdas capture ~12 variables by reference).
