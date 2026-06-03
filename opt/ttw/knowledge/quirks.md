# gsplat project quirks (render-specific)

**This is the gsplat (`gsplat_tt` / gstt2) project quirk pack.** It is read by the
tt-workflows loop at loop start **in addition to** the framework-generic
`tt-workflows/knowledge/quirks.md` (wired via `ttw.toml [project] knowledge_dir`).
The framework file holds codebase-agnostic Tensix/NoC/watcher quirks; THIS file
holds gsplat render-pipeline-specific traps. (The generic TT kernel quirks that
the gsplat work surfaced — the 70656 B per-core kernel-config-buffer ceiling, the
direct-L1-read-on-MATH consumer→producer handshake, and the `DeviceZoneScopedN`
compile-time-string-literal rule — were promoted into the framework `quirks.md`
because they apply to any TT kernel project.)

## Render pipeline (stage order)

```
project -> pfwc -> gather -> tile_assign -> sort -> cull -> blend
```

- Per-stage device timing comes out as `TTW_TIMING proj/ta/sort/cull/blend=<ms>`.
- The **ideal path** the loop gates/optimizes = TILE_BUCKET + L1-resident, host
  out of loop: `GSPLAT_TT_TILE_BUCKET=1 GSPLAT_TT_BUCKET_FIT=8192
  GSPLAT_TT_FUSED_TILE=0 GSPLAT_TT_L1_RECORD=1 GSPLAT_TT_SFPU_CULL=1` plus the
  device/resident project/gather/tile_assign/sort/blend flags. NOT FUSED_TILE,
  NOT soft-float `MB_DEVCULL`.

## Stage-timer lumping (sort= lies on the L1_RECORD path)

On the L1_RECORD / sort→blend-pipe path the SUMMARY `sort=` host-timer LUMPS
cull+blend into it and reports `blend=0.0`. The true DRAM radix is only ~7.5 ms
(+ ~1.3 ms publish), not the ~232 ms the timer shows. Before framing a milestone
around a huge stage timer: check whether a sibling stage reads `0.0`, and
re-measure with the pipe/fusion OFF (device zones). The real ms/view cost lives in
`blend = cull (~80 SFPU) + blend (~95)`.

## SFPU is the wall (cull + blend share SFPU cores)

- The ~80 ms `CULL_SPLIT` is irreducible DEVICE SFPU compute (Mahalanobis keep-test
  over ~3.37M candidates), not a host/dispatch bubble (only ~6 ms of it is the
  random gather).
- cull-SFPU (~80) and blend-SFPU (~93) run on the **same** SFPU cores ⇒ ~173 ms
  serial SFPU on the critical path **even on separate command queues** (multi-CQ
  buys nothing when the bottleneck resource is shared). Fusing cull+blend into one
  program is blocked by the iter-26 DEST hazard (and overflows the 70656 B
  kernel-config buffer — see framework quirks).
- The real lever is SHRINKING the SFPU candidate count (cheaper keep-test / tighter
  pre-cull prefilter), not "overlapping" or "hiding" the pass.

## Bucket blend (TILE_BUCKET) gotchas

- The bucket reader reads the keep-mask from `recp[10]` (RMW'd in the sort stage)
  OR from the shared `cull_masks` DRAM buffer the single `CULL_SPLIT` fills. Do NOT
  run `BUCKET_MASK` (a second `BUCKET_CULL` SFPU pass) on top of `CULL_SPLIT` — it
  is a duplicated ~74 ms cull. Read the shared `cull_masks` instead.
- Re-folding the soft-float cull INLINE into the L1-resident blend reader explodes
  blend (~417 ms): production's inline cull is free ONLY because it hides in the
  random-gather latency shadow; with L1-resident records the mover isn't stalled so
  the cull is fully-exposed serial work. A pre-computed SFPU `cull_masks` is
  ESSENTIAL once the gather shadow is gone.
- The bucket compute kernel (`alpha_blend_compute_mb.cpp`) reads coeff rows from L1
  directly on MATH ⇒ needs the explicit MATH→UNPACK handshake +
  `invalidate_l1_cache()` (`GSPLAT_TT_BUCKET_CB_FENCE=1`); the CB semaphore alone
  (UNPACK-only) does not protect it. See framework quirks "direct-L1-read on MATH".
- A clean "small Lb passes / large Lb fails" FIT split (`BUCKET_FIT` sweep) points
  at a fast-producer timing race, NOT the dense-record assembly — instrument the
  producer/consumer handshake, don't rewrite the scatter.

## Per-frame host bridges to delete (host-free hot path)

The render hot path is host-free only when these are gone:
- per-frame H2D of CONSTANT data (the `ta_no_cull` all-ones keep mask = 13.5
  MB/frame of 1s) — hoist the fill to the alloc/grow path.
- mid-chain sizing D2H + `Finish()` drains between project count→scan→scatter —
  pre-size outputs to the safe ceiling `padded_n` (M ≤ N) so the scatter dispatches
  without the host knowing M first.
- the `render_full_py` host bridges: tile_assign prefix-sum + K3 `m2_thresh` +
  compaction, sort CPU fallback vectors, blend `build_tile_assignment` + per-tile
  count scan. The host-free path `std::abort()`s on a device-stage failure (no host
  fallback is taken), so post-chain host reads can assume device success and shrink
  to a single 1-page control read.

## Load balance (already shipped — don't rebuild)

- cull/blend already use textbook LPT (`sort_device.cpp::build_lpt`,
  `blend_device.cpp::compute_lpt_assignment`), within ~1.3% of the candidate-count
  optimum. No load-balance win to ship there.
- `project` gather scatter: spatially-clustered tile→core split had a 2.12× WRITE
  skew — fixed by STRIDED assignment (`GSPLAT_TT_PROJ_BALANCE`, default-on).
- `tile_assign` K2 scatter splits the already-compacted uniform pair domain ⇒
  already balanced (1.0004× write); a strided split is a wash/loss.

## Profiler / Tracy (gsplat)

- gsplat NEVER closes the device, so a plain Tracy `capture-release` gets only
  host/JIT-warmup zones; DEVICE zones stream ONLY via `python -m tracy
  --dump-device-data-mid-run` (`opt/profiler/capture_tracy.sh`).
- The Tracy deliverable is ALWAYS the FULL 30-view render (gate `verify_cmd` stays
  `--views 1`). Confirm coverage via `profile_log_device.csv` row count (~30× the
  ~65.6k 1-view baseline). A 1-view / warmup-only `.tracy` is NOT satisfied.
- `MB_TIMING` device-zone sums are the trustworthy per-stage signal.
