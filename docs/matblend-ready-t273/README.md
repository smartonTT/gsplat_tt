# t273: remove the mat -> blend program barrier (L1 from #267)

Status: step 1 (kill gate) passed on the model. Step 2 (fused program, device A/B) not built yet.

## Problem

t267 (iter-199, traced): mat ends at 2.512 ms mean / 3.001 ms max per core, but every core
starts blend at 3.001 ms, because blend is a separate program and tt-metal runs programs on one
CQ back to back. Mean idle wait 0.489 ms/view.

## Step 1: model (`model.py`, outputs `model-cal.txt`, `model-raw.txt`)

Per view (30 bicycle views, counts dump from t267):
- mat items replayed with `build_mat_worklist` (LPT over 2 slots per core, t144 mover fits),
  scaled so the busiest slot ends at the measured 3.001 ms;
- blend per-tile cost from the t147 fit `87.0 + 0.1841 n - 0.1161 max(n-8192,0)` us, scaled
  so the per-core mean blend span is the measured 3.732 ms;
- blend claims tiles in the existing global descending-count order; with flags a core that
  claims a not-yet-materialized tile waits for it.

| variant (traced ms/view) | barrier | ready flags | saving |
|---|---:|---:|---:|
| raw mover fits | 6.807 (measured 6.815) | 6.173 | 0.634 (0.228-0.934) |
| mat ends compressed to measured mean 2.512 | 6.807 | 6.357 | **0.450** (0.396-0.491) |

Calibrated ready-wait inside blend is 0.066 ms. Mover lag +-: 0.43-0.52. Re-ordering blend
claims by ready time breaks blend balance (worse), so keep descending-count order.
Gate (>= 0.25 ms) passes. Caveat: t176 saw untraced gains of ~0.5x traced for mat-side
changes, so the untraced gain may be ~0.2-0.45 ms; the device A/B decides.

## Co-residence (L1 / kcfg)

Ready flags only help if mat and blend kernels are in ONE program (one kernel per RISC that
runs mat then blend), because a core runs one program at a time.

L1 CBs today (BH worker L1 ~1.5 MB minus kcfg):
- mat: NCRISC CBs 0-6 (CB_BUCKET 512 KB, CB_BSORT ~132 KB, CB_SLAB 256 KB), BRISC 16-22
  (192 + ~50 + 192 KB), cull fp32 tiles 8/9/24/25 -> ~1.37 MB.
- blend: CB_BUCKET 256 KB, CB_BUCKET_BULK 256 KB, CB_BMASK ~33 KB, ramps 32 KB, out/u8/
  scratch/ring small -> ~0.56 MB.

Sum ~1.93 MB does not fit. It fits only if blend's big CBs alias mat's (they are never live at
the same time on a core once that core's mat slots are done):
- blend CB_BUCKET (256 KB) -> inside mat CB 4 (512 KB)
- blend CB_BUCKET_BULK (256 KB) -> mat CB 6 (256 KB)
- blend CB_BMASK (~33 KB) -> mat CB 21 (~50 KB)
- small blend CBs (~70 KB) allocated fresh; mat ~1.37 MB + 70 KB is at the edge, so the
  ramps/out may also need to alias (mat CB 20/22, 192 KB each).
Aliasing by multi-index `CircularBufferConfig` on one allocation needs matching page sizes
per index; blend and mat use different page sizes, so a manual globally-allocated L1 buffer
(or one CB with several index/page-size entries) is needed. Unverified on device.
kcfg: the fused binaries roughly add blend's kernel text to mat's; must stay under the
kcfg sizing in `device_state.cpp` / `kcfg_size.h` (pfwc is the largest program at ~96 KB, so
it likely fits, unverified).

## Step 2 design (not built)

Flag `GSPLAT_TT_MATBLEND_FUSE=1`, default off; only when onelaunch && !OL_MAT_SELECT,
else the current two-program path.
1. Wrapper kernels per RISC: `matblend_ncrisc.cpp` (mat mover 1 then blend reader),
   `matblend_brisc.cpp` (mat mover 0 then blend writer), `matblend_compute.cpp` (mat cull
   then blend compute). Each includes both bodies in separate namespaces (shared headers
   pre-included once).
2. Blend CB ids shifted by `BLEND_CB_BASE` (e.g. +32; BH has 64 CB ids); runtime-arg and
   TensorAccessorArgs bases shifted by `BLEND_RTA_BASE`/`BLEND_CTA_BASE` defines (namespace-
   local `get_arg_val` shim).
3. Ready flags: DRAM buffer, 64 B slot per (tile, sub-item) at index `t*8 + sc`, holding a
   per-launch epoch (no clearing needed). The mat mover writes the epoch after
   `noc_async_write_barrier()` of that item's payload. Blend reader, after claiming tile t,
   polls `nitems = cnt > 16384 ? ceil(cnt/8192) : 1` flags until all equal the epoch.
4. Local ordering: blend NCRISC must not overwrite aliased CBs until the same core's BRISC mat
   slot is done (L1 semaphore BRISC -> NCRISC), and the TRISCs move from cull to blend only
   after mat compute drains.
5. A/B: paired, swapped untraced rounds, 30 views, md5 46a725ab on all views; keep only if
   >= 0.3 ms/view faster.

Files to touch: `render/host/sort_device.cpp` (build_program_subchunk, enqueue ~741-784),
`render/host/blend_device.cpp` (build_program_and_workload_mb, rt args), the three kernels
above plus `reader_alpha_blend_mb_devcull.cpp` (claim ~187-292) and
`sort_subchunk_materialize.cpp` (flag write), `tests/syntax_stub/check.sh` (new wrappers).
Size: ~3.1k lines of kernel code across three RISCs get wrapped; expect several device
debug rounds.

## t289: device build and A/B (2026-10-05, yyzo-bh-07 p100a)

- Compile fixes: `mat_defines` shadowed the out-parameter in `add_mat_cbs_and_defines`
  (syntax stub); the fused program's static CBs were 21248 B over the 1572864 B L1
  (`TT_THROW ... grow to 1594112 B`). Fixed by giving four blend CBs the storage of mat CBs
  that are idle by the time the blend side first touches them: CB_SCR_ATTR on mat CB 2,
  CB_MB_COUNTS on CB 5, CB_OUT on CB 6 (rounded up to 3-tile groups), CB_IMG_U8 on BRISC
  CB 20. Saves ~23.5 KB. No kcfg change was needed.
- md5: FUSE=1 gives 46a725ab on 30/30 views (rounds r2, r3, r4); FUSE=0 also 46a725ab.
- Untraced A/B (`render/run.py` avg_frame_ms, 30 views, same tree e3b4346):

  | round / order | base | fuse |
  |---|---|---|
  | r3: base fuse base fuse | 11.6, 11.9 | 11.1, 11.1 |
  | r4: fuse base fuse base | 11.8, 11.7 | 11.1, 11.2 |
  | mean | 11.75 | 11.13 |

  Gain ~0.62 ms/view, above the 0.3 ms keep gate. Logs: `t289/out/`.

## Step 3 (t293): Tracy check, default flip, iter 206

- Default flip: `GSPLAT_TT_MATBLEND_FUSE` unset = on, `=0` = off (`render/host/matblend_fuse.h`),
  on top of iter 205 (PFWC_RECIP_NEWTON on).
- Tracy (`t289/out/tracy-t293-ns-cores.txt`, 29 views, `opt/profiler/matblend_cores.py`):
  mat->blend wait 0.000 ms (was 0.489 in #267); blend_end_max 6.261 ms vs 6.815 measured /
  6.895 in #267 for the separate programs (-0.55 to -0.63 ms traced). Model predicted 6.173.
  Each core starts blend right after its own mat (blend_start = mat_end, 2.243..2.987 ms).
- Tracy caveat: under the profiler the auto kcfg grows by 32 KB and the fused program's CBs
  then overflow L1 by 5888 B (`tracy-t293-fuse-capture.log`). `GSPLAT_TT_KCFG_EXTRA_KB=26`
  makes pfwc's program too large for kcfg (97536 > 97280 B, `tracy-t293-k26-capture.log`).
  The capture that worked used `GSPLAT_TT_PFWC_WRITER_SPLIT=0` (pfwc only; mat/blend code
  unchanged). Default (untraced) runs are not affected.
- md5 with Newton on: FUSE on and FUSE=0 both 906e0435 on 30/30 views (round r6); the device
  screenshot hero is bit-identical to iter 205 (md5 86524912, golden max diff 0).
- Untraced r6 (order nofuse, base): fuse on 11.129, off 11.653 ms/view (-0.52). Screenshot run
  at defaults: 11.105. Round r5 failed at device init (FW init timeout on core 11-9 right after
  the Tracy captures); r6 ran clean on the same reservation.
