# B4' step 1: fused kcfg size on device, blend alias fifo cut (task #544)

Measured on bh-33 (p150, measurement reservation), 2026-10-12 13:14-13:26 PDT, commit 5c34168c
(base best-iter-225 b612d4ed). Driver: `drive.sh` -> `remote_kcfg.sh` (kcfg/ELF probes) and
`remote_ab.sh` (b2b A/B), all under `ttp lock p100`. Raw logs: `out/`. Run 1 (52c6e47d) failed to
link trisc1 in the cut arm (`out-run1-linkfail/`); fixed in 5c34168c (math TRISC has no CB interface,
so the shrink is a no-op there; host test `tests/unit/test_blend_alias_fifo_math.cpp`).

## Verdict: GO for B4' sort integration (single stream, ring R=8)

| item | value |
|---|---:|
| fused mat+blend program kcfg size | **88,592 B** |
| today's kcfg ring (69 KB + 32 KB extra) | 103,424 B (14,832 B spare) |
| sort BRISC code (sort_bin_onelaunch, largest variant) | 22,408 B |
| town TRISC2 code (NT=3 today; NT=1 is no larger) | 4,312 B |
| fused mat+blend + sort, estimated | about 115,300-115,800 B |
| kcfg growth needed over today | **about +12.4 KB** (KCFG_EXTRA_KB 32 -> 45) |
| L1 made usable by the alias cut (per core) | **569,344 B** (CB5 123,904 + CB6 251,904 + CB20 193,536) |
| total dead L1 during blend (with CB21, CB22, cull) | 853,696 B |
| left after the +13 KB kcfg | about 840,384 B |
| one sort stream at R=8, window 1024 | 592,640 B -> **fits, about 247 KB spare** |
| b2b ms/view, cut on vs off (median of 4 alternating passes) | **8.990 vs 8.996** (no regression) |
| md5 over 30 views, cut on | 906e0435, 30/30 |
| hero PSNR vs benchmarks/reference_v2/hero.png | 42.51 dB, golden match, max LSB 0 |

The kcfg need (+12.4 KB) is well inside the step-0 budget (+32 KB), so ring R=4 is not needed.

## 1. kcfg size of the fused program

`GSPLAT_TT_PFWC_WRITER_SPLIT=0` makes pfwc small, so the first program that overflows a small ring is
fused mat+blend. tt-metal's state.offset check (program.cpp) reported:

- KCFG_EXTRA_KB=14 (84,992 B ring): `Program size (88592) too large`
- KCFG_EXTRA_KB=8 (78,848 B) and 0 (70,656 B): same 88,592 B
- KCFG_EXTRA_KB=20 (91,136 B): runs

88,592 B matches the fused program: its ELF text sums to 87,324 B (below) plus 1,268 B of RTAs,
CB config and semaphores. Default settings (KCFG_EXTRA_KB=32) and x26 (97,280 B) both run.

## 2. Per-RISC ELF text sizes (tt-metal-cache, `out/elfsize.txt`)

| kernel | RISC | bytes |
|---|---|---:|
| matblend_ncrisc | NCRISC | 28,540 |
| matblend_brisc | BRISC | 19,352 |
| matblend_compute | TRISC0 / 1 / 2 | 8,112 / 28,536 / 2,784 |
| sort_bin_onelaunch | BRISC (2 variants) | 22,252 / 22,408 |
| sort_bin_onelaunch | NCRISC (2 variants) | 22,268 / 22,424 |
| sort_ol_town_compute (NT=3) | TRISC0 / 1 / 2 | 4,336 / 4,328 / 4,312 |

Fused estimate = 88,592 + 22,408 (sort movers on BRISC) + 4,312 (town on TRISC2) + a few hundred B
of extra RTAs. This is an upper bound on code: the sort and blend BRISC code share one ELF, so common
runtime code is counted twice here.

Cost: static CBs start right after the kcfg ring, so +13 KB of kcfg moves every CB up by 13 KB. The
static layout has 2,304 B headroom, so the static CB total must drop by about 11 KB (CB22 is dead
during blend; shrink it if mat can spare it, else take it out of the sort window). That comes out of
the 247 KB spare.

## 3. Blend alias fifo cut

`GSPLAT_TT_BLEND_ALIAS_CUT` (default on, `render/kernels/blend_alias_fifo.h`): after mat ends, the
reader/writer RISCs and the unpack/pack TRISCs set `fifo_size`/`fifo_limit` of CB_OUT (CB6),
MB_COUNTS (CB5) and IMG_U8 (CB20) to the ring sizes blend actually uses (12,288 / 8,192 / 3,072 B),
so those rings no longer wrap through the mat region. The math TRISC is skipped (no CB interface).

- Output unchanged: md5 906e0435 on 30/30 views, hero golden match.
- b2b (render/run.py --back-to-back, 30 views, 20 passes each, alternating order):

| pass | off | on |
|---|---:|---:|
| r0 | 8.996 | 8.993 |
| r1 | 8.999 | 8.988 |
| r2 | 8.996 | 8.989 |
| r3 | 8.996 | 8.992 |
| median | 8.996 | 8.990 |

The cut is not a speedup by itself (-0.006 ms is noise); it is what makes the 569 KB of alias tails
usable for the sort arena.

- Hero (`opt/metal-screenshots/ttw-t544/hero.png`, device render at 5c34168c defaults) and the x10
  diff were checked by eye: no tile seams or block artifacts; the diff shows only the usual thin
  edge residue (spokes, foliage) against the reference.

## 4. Next (B4' steps 3-5)

1. NT=1 town variant on TRISC2, arena offsets as RTAs.
2. Fuse k2(N+1), then sort(N+1) into mat+blend(N): movers on BRISC, town on TRISC2, KCFG_EXTRA_KB 45,
   static CB total cut by about 11 KB.
3. Gate: b2b ms/view drops by >= 0.4 ms; hero + diff checked.
