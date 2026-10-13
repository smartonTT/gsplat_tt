# t654: absolute means step 2 — shelved before the device build

Task #654 was to build the real absolute-means path (emit writes absolute fp32 means, cull and
blend use them directly) and keep it at >= 0.15 ms/view b2b. The reopen came from #647
(docs/absmeans-t647/RESULT.md on ttp/t647-cull-gate-absmeans): 0.13 (emit, #634) + 0.064 / 2
(cull) = 0.162 ms/view. Checked 2026-10-13 PDT on best-iter-228 (b722408a): the 0.064 ms cull
term is not real, so the bound is 0.13 ms/view, below the bar. No device build was made.

## Evidence

1. **The gated cull kernel does not run in the default path.** #647 gated
   `dm_fp32::add_sub_roundtrip` in `microblock_cull_compute.cpp` (lines 420-421). That kernel is
   built only by
   - `sort_device.cpp` `build_program_bucket_cull`, called only if `bucket_mask_enabled()`, which
     is `return false;` (sort_device.cpp:128), and
   - `blend_device.cpp` `cull::init_device_context` (line 1265), which has no caller. The blend
     only creates `tile_l1_cull` contexts (lines 1700, 1731), and only when
     `!sort_matcull_fused()`. Fused mat-cull is on by default (`GSPLAT_TT_FUSE_MATCULL`, run log
     `MATBLEND_PROGRAM fz=1`).

   The default cull is the band cull in `mat_cull_compute.cpp` (inside `matblend_compute`). It
   reads `F_MX`/`F_MY` straight from the record (microblock_band_cull_compute.cpp:160-163) with no
   round trip, so absolute means save nothing there.

2. **The two #647 arms ran the same binaries.** The JIT caches of both arms on bh-33
   (`/localdev/smarton/.cache/ttmc-t647abl`, `ttmc-t647raw`) hold the same 21 kernels with the
   same hashes, for example `matblend_compute/3680402692383663153` and
   `sort_ol_town_compute/3787070471743617640`. Neither cache has any cull kernel. So
   `CULL_ABS_RAW` changed no compiled code, and the 0.100 / 0.041 / 0.047 / 0.080 ms "deltas"
   (median 0.064) are an A/A comparison.

## Consequences

- Upper bound for real absolute means = the #634 emit saving, 0.13 ms/view b2b. That is below the
  0.15 bar before counting the costs the real path adds:
  - the tile origin in the mat-cull job page plus one SFPU subtract per batch in `band_derive`;
  - a per-tile ramp offset in the blend.
- The #647 A/A gave a 0.04-0.10 ms/view spread, with the same sign in all 4 rotated rounds. Treat
  b2b deltas under ~0.1 ms/view from 4 rounds as noise. That includes part of #634's 0.13: its
  arm 1 also handed wrong (absolute-as-local) means to the band cull, which changes the cull
  work.
- Decision: **shelved** (lever: absolute means / drop the emit tile-origin subtract). Reopen only
  with an emit-side change worth > 0.15 ms/view on its own. Any reopen gate must first confirm
  that the kernel it edits shows up in the run's JIT cache.
