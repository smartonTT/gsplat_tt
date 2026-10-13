# t573: B4' sort(N+1)-in-blend(N) crash: root cause and fix

## Arms (19c39d5e, bh-33, GSPLAT_TT_OL_TOWN_NT=1 GSPLAT_TT_OL_SOLO=1 GSPLAT_TT_SORT_IN_BLEND=1)

| arm | what runs | result |
|---|---|---|
| 1 | global sort only | pass |
| 5 | arm-5 writes | hang (wild BRISC PCs at view 2) |
| 6 | arm-5 writes at CB21 base | pass |
| 7 | arm-5 writes at CB5 base + 8192 | pass |
| 8 | stores compiled in, skipped at runtime | **hang** |
| 9 | no K2 (B4A fold body skipped) | pass |

Arm 8 hangs without executing a single SIB store, so the stores are not the cause. Arm 9 passes,
so it needs the folded K2. (t571's arms 6/7 "hang" result was invalid: the host accepted only 1-5.)

## Root cause: BRISC local-memory stack overflow

Blackhole BRISC kernel .data/.bss and the stack share 8 KB local memory (0xFFB00000-0xFFB02000);
the stack grows down to `__stack_base`. The folded K2 (`k2k::kernel_main`, tile_assign_scatter_seg.cpp)
has a 4528 B frame (`cnt[1024]` plus 2 KB); the deepest chain from `_start` is 4688 B.
SIB adds `kSibIds` (56 B .data) and `olk::g_ol_arena[32]` (128 B .bss), moving `__stack_base` from
0xffb00d40 to 0xffb00e00. `opt/brisc_stack_check.py` (static: frame sizes over the direct call graph)
on the JIT-cached matblend BRISC ELFs:

| build | __stack_base | stack avail | deepest chain | margin | device |
|---|---|---|---|---|---|
| passing arms / production | 0xffb00d40 | 4800 | 4688 | +112 | pass |
| hanging arms (5, 8, production SIB) | 0xffb00e00 | 4608 | 4688 | **-80** | hang |

K2's frame runs over `g_b4a_fn`/`g_ol_arena` (function pointers and CB addresses), and the next
indirect call jumps to a wild PC. Pass/fail tracks the margin exactly, independent of the store
addresses (arms 6/7) and of whether the stores run (arm 8).

## Fix (8231f5db)

- SIB CB pointers come straight from the runtime args (`sib_cbp(id)`, matblend_brisc.cpp), via
  `OL_ARENA_CBP` in sort_bin_onelaunch.cpp: no `kSibIds`, no `g_ol_arena` (-184 B).
- pfwc_fuse.h k2_range/k2_range_speed: `muldiv_lo32` (32x32/32, shift-subtract) instead of
  64-bit `/`, which pulled libgcc `__udivdi3` and its 256 B `__clz_tab` into .data (-256 B,
  about -956 B text). Host unit test (tests/unit/test_pfwc_fuse.cpp) checks it against uint64 division
  on edge values and 200k random triples.
- Expected margin after the fix: about +360 B for SIB, about +370 B for production B4A
  (measured on the new ELFs: out-fix-check/stack.jsonl).

Upstream/project note: nothing checks BRISC stack depth at build time; a 4.5 KB frame on an 8 KB
local memory is one table away from a silent crash. `opt/brisc_stack_check.py <elf>` exits 1 when
margin < --reserve (default 128 B).

## Device verification of the fix (bh-33, 8231f5db, 2026-10-12 20:06-20:27 PDT)

- Stack margin on the new matblend BRISC ELFs: +368 B (avail 5056 B, depth 4688 B); old cached
  ELFs still show +112 / -80 (out-fix-check/stack.jsonl).
- drive.sh check: SIB on (`GSPLAT_TT_OL_TOWN_NT=1 GSPLAT_TT_OL_SOLO=1 GSPLAT_TT_SORT_IN_BLEND=1`)
  `MD5_GOLDEN_OK 906e0435 30/30`; SIB off also 30/30. The crash is gone.
- Hero (SIB on): opt/metal-screenshots/ttw-t555/hero.png, PSNR 42.51 dB vs
  benchmarks/reference_v2/hero.png, golden match, max LSB 0. Looked at hero and diff: no tile seams.
- b2b, render/run.py --back-to-back, 30 views, 20 passes, median ms/view, alternating order:

| round | base (SIB off) | SIB on |
|---|---|---|
| r0 | 8.672 | 11.380 |
| r1 | 8.661 | 11.505 |
| r2 | 8.670 | 11.536 |
| r3 | 8.669 | 11.506 |
| check run | 8.698 | 11.479 |
| **median** | **8.670** | **11.505** |

SIB on is +2.84 ms/view slower (blend stage 7.18 -> 9.88 ms; sort 0.93 -> 1.01). raw_md5 5a438f5a
on every run. Base at 8231f5db (8.67) is within run-to-run spread of best-iter-226 (8.615 on bh-33
at another time); not A/B'd against the best-iter-226 binary in this run.

Verdict: **SHELVE** sort-in-blend (gain must be >= 0.1 ms/view; it is a 2.8 ms loss). Not
measured why; likely causes are the sort sharing blend's BRISC (blend's reader waits while it
sorts) and the single mover (`OL_TOWN_NT=1`, movers=1 vs 2 in the base). The stack fix itself
(no SIB tables, no libgcc `__udivdi3`) is worth keeping: it gives the default path +256 B of
BRISC stack headroom.
