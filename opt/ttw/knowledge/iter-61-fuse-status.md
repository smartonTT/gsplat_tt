# Iter 61 (ttw-061) — fuse tile_l1_cull into blend

**Status:** implemented locally + on `yyzo-bh-03`; **gate not verified** (device wedge after kill mid-hang; ARC timeout on retry).

## What changed

1. **`MB_FUSE_TILE_L1_CULL`** on blend reader + compute (replaces standalone `tile_l1_cull` 3-kernel program).
2. **Reader:** one `rd_l1_bulk` path per subchunk (in-budget tiles use `CB_BUCKET_BULK` + radix reorder); no DRAM `cull_masks` load.
3. **Compute:** `tile_l1_cull_sfpu.hpp` + `fuse_l1_cull_into_bmask()` runs `tile_mb_mask` on same L1 residency, packs masks into `CB_BMASK_BULK`, then `cp_l1_blend`. Stashes R,G,B,T across cull when `MB_FLAG_CONTINUE`.
4. **Host:** `blend_mb_devcull_resident` no longer enqueues `tile_l1_cull::process_frame`.

## Device runs

| Try | Result |
|-----|--------|
| 1–2 | Kernel compile: `get_read_ptr` on compute; `cull_dispatch` inside `#ifdef TRISC_MATH` only |
| 3 | JIT OK, sort ~49ms, **hang >30min** in blend (killed) — likely `logf` on non-MATH TRISC (fixed in try 4+) |
| 4–5 | ARC / device init timeout after kill (needs unwedge) |

## Next steps

1. Recover `yyzo-bh-03` (reboot / `recover.sh` / clear stale device).
2. Re-run `render/run.py --iter-dir ttw-061` via `devrun.sh`; confirm no hang and `TTW_METRIC hero_vs_ref >= 63.6`, `ms_view < 296`.
3. If hang persists: Tracy zones `tile_mb_mask` vs `rd_l1_bull`; check CB_BLEND_STASH / multi-subchunk `continue_blend`.

**Blocked:** mat-before-blend / unified payload reader (step C) — not touched.
