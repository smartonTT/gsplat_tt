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
