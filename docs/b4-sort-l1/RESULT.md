# B4 step 0: can sort(N+1) fit inside fused mat+blend(N)? (task #543)

CPU-only study, 2026-10-12. Base: 9ceb255f (B4 re-bound, docs/b4-rebound/RESULT.md).
All byte counts are from the host CB code (render/host/sort_device.cpp,
render/host/blend_device.cpp, render/host/matblend_fuse.h) at default env
settings. The kernel-size numbers are the one part that is not measured offline (see section 4).

## Verdict

**Conditional GO for the B4' sort part, single stream, with a phased L1 layout.**

- Free L1 per core during blend(N), once the aliased blend rings are cut down to what they actually use: **about 853,696 B (834 KB)**.
- Sort working set of one stream (all mover work on BRISC, town emit on TRISC2), same settings as today (ring R=8, window 1024): **592,640 B (579 KB)**. It fits, with about 261 KB to spare.
- Running both movers' sets, as the sort does today (1,160,320 B), does **not** fit. That is a NO-GO.
- Leaving today's aliasing unchanged frees only about 279,552 B. That is also a NO-GO.
- Byte budget for step 1: sort arena ≤ 600 KB, kcfg growth ≤ +32 KB, both paid for out of the 834 KB.
- Expected gain: **0.6–1.0 ms/view realistic** (bound 2.11). This is the B4' range from #539 minus a small NoC cost. Staging does not need to shrink at R=8, so the NoC cost is 0–0.1 ms.
- Conditions:
  1. shrink the aliased blend rings;
  2. add an NT=1 town variant, because only TRISC2 is free;
  3. measure the fused program's kcfg size on device before building anything else.

## 1. Per-core L1 map of fused mat+blend (static CBs, creation order)

CB space at +32 KB kcfg is about 1,429,056 B: 1536 KB minus MEM_MAP_END minus the 103,424 B kcfg ring. This matches t302, which overflowed by 5,888 B at +40 KB.

| CB | bytes | role after mat ends (during blend) |
|---|---:|---|
| 0, 1 | 64 + 64 | dead |
| 2 | 2,048 | **live**: blend SCR_ATTR alias (2 KB) |
| 3 | 32 | dead |
| 4 | 524,288 | **live**: blend BULK ring, 2 × 256 KB slots, fully used |
| 10, 26 | 64 + 64 | dead |
| 5 | 132,096 | alias MB_COUNTS: uses 8,192 → **123,904 dead** |
| 6 | 264,192 | alias CB_OUT: uses 12,288 (6 × 2 KB) → **251,904 dead** |
| 16, 17 | 64 + 64 | dead |
| 18 | 2,048 | dead |
| 19 | 32 | dead |
| 20 | 196,608 | alias IMG_U8: uses 3,072 → **193,536 dead** |
| 21 | 50,176 | **dead** |
| 22 | 196,608 | **dead** |
| 8, 9, 24, 25 (cull) | 32,768 | **dead** |
| 7, 23 | 128 | live flags (counted as live) |
| blend non-aliased (XRAMP, YRAMP, T_RB, SCR_IDS, SCR_MASK, CORE_TILES, TILE_Q, DA_RING) | 25,344 | live |
| **total static** | **1,426,752** | headroom about 2,304 B |

Live during blend: 575,360 B. **Dead: 853,696 B.**

### What needs to change

Each index of a multi-index CB config gets the config's *total* size as its fifo size (circular_buffer_init.h; #289). So the aliased blend rings (CB_OUT on 6, MB_COUNTS on 5, IMG_U8 on 20) wrap through the whole mat region today. The "dead" remainders of CB5, CB6 and CB20 only become usable if blend's fifo for those ids is cut to its real size. Two ways to do that:

- **(a) Device-side override:** after mat ends, set `fifo_limit`/`fifo_size` on all three RISCs for the 3 alias ids. This costs no L1, but it has to be done identically on every RISC that touches each ring.
- **(b) Separate CBs:** give those three rings their own CBs (23.5 KB). Today's static layout has only 2.3 KB of headroom, so these bytes have to come out of CB22 or CB21 (both dead during blend, but live during mat). That means a small mat resize.

Without either change, only CB21 + CB22 + cull + small are free: about 279,552 B. That is less than half of one sort stream.

Contiguity:

- CB20+3,072 → CB21 → CB22 → cull is one span of **473,088 B**.
- CB6 rest (251,904) and CB5 rest (123,904) are the other two large blocks.

The sort onelaunch kernels use their CBs only as raw scratch: there are no push, wait, reserve or pop calls, only `get_write_ptr`. So B4' can address an arena by byte offset, passed as RTAs computed from mat's CB addresses. It needs no new CB ids. If explicit CBs are preferred, there are 25 free ids: 11–15, 27–31 and 49–63.

## 2. Sort working set and how far it can shrink

Per mover today (`build_program_sort_onelaunch`, pb=8, ring=8, depth=8, win=1024, town on):

| CB | bytes | what |
|---|---:|---|
| 0–2 | 4,608 | staging 3 × 1,536 |
| 3 | 64 | ctl |
| 4, 5 | 16,384 | bin rows |
| 6 | 4,096 | |
| 7 | 65,536 | blendrec, 4 slots × 256 pages × 64 B |
| 8 | 512 | |
| 12 | 196,608 | window, 1024 × 3 × 64 |
| 13 | 266,240 | ring, 1024 × (8 × 32 + 4) |
| 14 | 13,632 | town (NT=3, SLOTS=4, LIST_MAX=128, QCAP=256) |
| **per mover** | **567,680** | |
| shared 10 + 11 | 24,960 | BIN_ROW 8 KB + (2 × cores + 2) × 64 |

Today's total for both movers is 1,160,320 B, which does not fit.

B4' single stream, placed in the dead regions:

- fixed part 104,832 + window 196,608 + ring 266,240 + shared 24,960 = **592,640 B**;
- placement: ring + window (462,848) go in the 473,088 contiguous span, and the remaining 129,792 go in CB6's dead tail;
- k2(N+1) scratch is small (raw ring a few KB per mover; CB_JOB = 2 × JOB_BYTES + 64). k2 runs before the sort, so it time-shares the same arena.

Shrink options, if the arena must get smaller (for example, to fund more kcfg):

| change | saves | NoC / time cost |
|---|---:|---|
| ring R=8 → 4 | 131,072 | twice as many ring flushes, with 128 B emit writes instead of 256 B: about +0.05–0.15 ms |
| ring R=8 → 2 | 196,608 | 64 B writes, NoC-transaction bound: about +0.2–0.3 ms |
| window 1024 → 512 pages | 98,304 | more DRAM re-reads of the key window: about +0.05–0.1 ms |
| blendrec 4 → 2 slots | 32,768 | less overlap of blendrec emits: small |
| town NT=3 → 1 | about 9,000 | required anyway (only TRISC2 is free) |

None of these is needed at R=8: the full-size stream fits with 261 KB to spare. The NoC deductions above are estimates scaled from #539's 0–0.3 ms range, not measurements.

## 3. Kernel binary and kcfg headroom

- On Blackhole, kernel binaries run from the kcfg ring at MEM_MAP_END. There is no per-RISC binary limit: MEM_MAX_KERNEL_SIZE is 1497 KB for every RISC. The limit is the whole program (all binaries, RTAs, CB config and semaphores) against the kcfg size. That is 69 KB default + GSPLAT_TT_KCFG_EXTRA_KB, so 103,424 B today; the check is at program.cpp:2280.
- Static CBs start right after the kcfg ring, so **1 KB more kcfg costs 1 KB of CB space**. In today's static layout that leaves only 2.3 KB. In the phased layout the 261 KB spare can pay for it, but only if the static CB total itself shrinks. The simplest way: shrink CB22 (dead during blend, live during mat) by the kcfg growth, if mat can spare it, or run the arena off a smaller sort window.
- The 108,384 > 103,424 B overflow (run 1454 / t528) is the **profiler build plus MATCULL_PROF**. The plain fused program fits, but its size is not known offline.
- Adding the sort BRISC code (sort_bin_onelaunch.cpp, 1,257 lines + algo/town headers) and the town TRISC2 code (about 300 lines) to the fused program: **estimated +10 to +25 KB of kcfg**, not measured. So the plan is +16 to +32 KB more kcfg, paid out of the spare.
- Profiled builds will overflow even more. Profiling B4' needs MATCULL_PROF off, or a profiler-only kcfg bump paid for by a smaller sort ring (R=4).
- Per-RISC local memory (BRISC 8 KB, TRISC 4 KB of stack and globals) must also hold the sort's globals. Check this in the same build.

## 4. Step-1 checklist (the measurement first)

1. **Measure:** JIT-build today's fused mat+blend on the box and record:
   - the program's kcfg offset (log `state.offset` at program.cpp:2280 or read it from the error at a low KCFG_EXTRA_KB);
   - the per-RISC ELF text sizes in tt-metal-cache.

   Do the same for the sort onelaunch program. Unfused size + sort size gives the real kcfg need. If it is above +32 KB more, shrink the ring to R=4 to pay for it.
2. **Shrink the blend alias fifos** (option a or b in section 1). Check that md5 906e0435 is unchanged across 30/30 views and that there are no seams.
3. **NT=1 town variant** (sort_ol_town.h / sort_ol_town_compute.cpp), with the arena offsets as RTAs.
4. **Fuse k2(N+1) and then sort(N+1)** into mat+blend(N): BRISC runs the movers, TRISC2 runs the town between blend's TRISC2 work (this is the B4a then B4b staging from #539).
5. **Gate:** b2b ms/view over 30 bicycle views must drop by ≥ 0.4 ms. The hero screenshot and diff must be checked as usual.
