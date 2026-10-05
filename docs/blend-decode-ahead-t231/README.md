# t231: TRISC0 decode-ahead staging for blend (GSPLAT_TT_BLEND_DECODE_AHEAD)

Design from docs/blend-trisc1-t230 section 5 (S2a, S2). Kernel:
`render/kernels/compute/alpha_blend_compute_mb.cpp` (decode-ahead block after
`blend_t_readback`, `blend_da_walk`). Host: `render/host/blend_device.cpp` (CB 10 ring,
`BLEND_DECODE_AHEAD` define).

## 1. What it does

t230 found that blend is limited by TRISC1 (MATH) instruction issue, not the SFPU.
Per live record TRISC1 spent about 77 RISC cycles staging the 7 coefficients
(14 SFPLOADI immediates built with shifts, masks and adds) and about 27 cycles
scanning records. TRISC0 (UNPACK) sits idle during the blend loop.

- **Knob 1 (S2a).** TRISC0 walks the same records. For each live record it writes the
  14 SFPLOADI words, already encoded, plus the record's microblock mask to an L1 ring
  (CB 10, 64 slots of 64 B, 4160 B). TRISC1 still scans and does the T readbacks; for
  each live record it takes the next slot, loads the 14 words and pushes them with the
  same SFPU ops in the same order as `BLEND_RAW_STAGE` (t146, md5-identical).
- **Knob 2 (S2).** The ring holds only live records plus markers: a readback marker at
  the same record position as the `g_seen % 512` trigger, and an end-of-subchunk
  marker. TRISC1 reads only the ring: no scan, no T reduce.
- TRISC0 runs its own copy of the T reduce at each readback (same function, same
  packed T tile, fenced), so it masks records exactly as TRISC1 did.
- Counters: produced count (TRISC0) and consumed count (TRISC1) in a 64 B ring header,
  volatile, with `fence` before each read of the other side's counter. TRISC0 fences
  the slot stores before it publishes. The counts run over the whole launch.
- Deadlock: TRISC0 publishes every slot (and the S2 readback marker) before it blocks
  in a readback, and its ring-full wait only needs TRISC1 to consume. The existing
  MATH-to-UNPACK end-of-subchunk ack is kept, so TRISC0 never runs into the next
  subchunk's slab early.
- Mailbox order is unchanged: TRISC0 mailboxes the ring address to TRISC1 once, before
  the first `get_tile_address`.
- TRISC2 (PACK) only takes part in the readbacks, so it steps from one to the next.

Compile-only check (`build.sh`, on the host, no device): all nine TRISC0/1/2 ELFs build
clean at knobs 0, 1 and 2. TRISC1 text: 12924 B (0), 12984 B (1), 11948 B (2).
TRISC1 consume path at knob 2 (disassembly): ring check, 14 `lw` interleaved with
14 `sw` to the instruction buffer and the 17 fixed SFPU ops, two counter stores, then
the unchanged jump dispatch. The TRISC0 producer is about 55 instructions per live
record plus a `fence`.

## 2. Device gate (yyzo-bh-07 p100a, bicycle, 30 views, 1024x1024, untraced)

Board: yyzo-bh-07, a Blackhole p100a, not a p150. One sync and build of 2f926c6
(knobs are JIT defines). Smoke at knobs 1 and 2, then 3 rotated rounds. Driver
`drive.sh`, logs `out/run-r*-*.log`, chain log `out/t231-gate.log`.

| arm | r1 | r2 | r3 | mean ms/view | FPS | paired vs def | blend stage |
|---|---:|---:|---:|---:|---:|---:|---:|
| def (knob 0) | 12.698 | 12.617 | 12.660 | 12.658 | 79.0 | | 8.145 |
| DA1 = S2a | 12.382 | 12.257 | 12.266 | 12.302 | 81.3 | **-0.357** | 7.771 |
| DA2 = S2 | 11.965 | 11.949 | 11.939 | 11.951 | 83.7 | **-0.707** | 7.486 |
| RAW_STAGE=1 (calibration) | 12.413 | 12.466 | 12.478 | 12.452 | 80.3 | -0.206 | 7.947 |

- md5: all 14 arms (smoke and gate) 30/30 views identical to 46a725ab.
- Kill gate (S2a at least 0.15): passed, -0.357. Gate (at least 0.3): S2 passes at -0.707.
- The model overestimated: S2a modeled +0.66 to +0.90, measured 0.357; S2 modeled
  +0.91 to +1.20, measured 0.707. The RAW_STAGE calibration came out near its model
  (+0.27 modeled, 0.206 measured). So the staging ALU cut is about right, but the 14
  L1 loads and the ring handshake cost TRISC1 more than modeled. S2 cuts about
  45 cycles per live record (0.707 / 0.0158), leaving about 255 against the
  210-cycle SFPU-only bound from t230.

## 3. Default flip and verify (096d892)

`GSPLAT_TT_BLEND_DECODE_AHEAD` now defaults to 2 (=1 S2a, =0 off). The kernel drops
to 0 on knob mixes it does not stage (COEF_DEST, SFPU_UNORM, JUMP_WALK, ABL,
FPU_QF_ABL off their defaults, or MB_STATS), so those experiments still build
(checked with `EXTRA="#define GSPLAT_TT_MB_STATS 1" build.sh`).

Verify (`verify.sh`, two swapped rounds, logs `out/run-rv*-*.log`):

| arm | rv1 | rv2 | mean ms/view | FPS | blend stage |
|---|---:|---:|---:|---:|---:|
| default (knob 2) | 11.931 | 11.946 | **11.939** | **83.8** | 7.464 / 7.481 |
| knob 0 | 12.612 | 12.639 | 12.626 | 79.2 | 8.164 / 8.159 |

Paired -0.681 / -0.693, mean **-0.687 ms/view**. md5 46a725ab on all 4 arms.

Device hero (`opt/metal-screenshots/ttw-198/hero.png`, the rv1 default run):
bit-identical to `tests/fixtures/hero/hero_golden_8bit.png` (max diff 0; 41.16 dB
against `benchmarks/reference_v2/hero.png`, the same as the golden). Visual check:
no tile seams, blocky or empty tiles, stripes or color shifts. The 10x diff against
reference_v2 shows only edge detail (spokes, frame, foliage), with no tile grid.

## 4. Tracy of the default (096d892, `GSPLAT_TT_KCFG_EXTRA_KB=32`)

`remote_tracy.sh`, 30 views, output `out/tracy-{zones,gaps,roofline}.txt`, capture
`opt/profiler/ttw-198/render.tracy`.

| | t230 (06d4fa7) | t231 default | change |
|---|---:|---:|---:|
| device span ms/view | ~12.31 | 11.663 | -0.65 |
| mat + blend program window | 7.562 | 6.898 | -0.66 |
| tile_blend_sfpu makespan | 4.417 | 3.760 | -0.66 |
| tile_blend_load (NCRISC) makespan | 4.341 | 3.684 | -0.66 |
| pfwc / K2 / sort_ol windows | 2.330 / 0.984 / 1.436 | 2.330 / 0.981 / 1.430 | 0 |
| TRISC kernel ms per frame, busiest core | 10.85 | 10.19 | -0.66 |

The whole saving is in the blend phase. Other programs are unchanged.

## 5. Next

- Blend staging still costs TRISC1 about 31 pushes and 14 loads per live record.
  The next blend lever is to move the staging off TRISC1 completely: for example,
  TRISC0 unpacks the record's coefficients straight into the DEST slots, so TRISC1
  only runs the bodies. t230's bound for "all scan and staging work gone" is
  +0.92 to +1.45 ms/view at the old base. S2 now gets 0.707 of that. Model it
  before building.
- The program windows are now pfwc 2.33, K2 0.98, sort_ol 1.43 and mat + blend
  6.90 ms/view (mat_cull_mask 2.84 inside it).
