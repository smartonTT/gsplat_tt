# L1 subchunk sprint (post-sort cull/blend in pure L1)

**Active loop.** Supervisor dispatches Composer 2.5 workers; all implementation in
`render/` only. Do not touch `src/` unless fixing shared headers consumed by render.

## Goal (ordered)

1. **Keep DRAM sort as-is** — do not move full-tile radix into L1 for fat tiles.
2. **After sort**, split tiles with `count > kBucketFit` (8192) into **subchunks**:
   contiguous depth-ordered slices of `<= 8192` records each (same tile, same sort order).
3. **Cull + blend purely from L1** per subchunk:
   - Reader: **one NOC transfer** loads the whole subchunk payload into L1 (phase A).
   - Compute reads L1 directly — **no per-candidate CB row stream** (phase A).
   - Process subchunks **in order** for a tile (blend state carries across subchunks).
4. **Phase B:** double-buffer reader (prefetch subchunk N+1 while cull/blend N).
5. **Phase C:** microblock-major **transmittance saturation early-out** (mb done mask).

## Architecture anchor

- Record today: **one** 32 B splat in the **low 32 B** of a 64 B DRAM page (upper 32 B wasted; sub-64 B pages unreliable on BH).
- Target (iter 50): **two** 32 B splats packed per 64 B page → halves bulk-load bytes and ~doubles effective splats per L1 footprint at the same `kBucketFit` slot count.
- `kBucketFit = 8192` (logical splat slots per tile/subchunk; with 2-pack, 8192 slots = 4096 pages).
- Overflow path: per-subchunk bulk L1 load in `reader_alpha_blend_mb_devcull.cpp` (iter 49); delete per-row CB stream once stable.
- In-budget tiles (`count <= 8192`): may stay on current L1 bucket path until unified with 2-pack layout.

## Iteration slices (strict order)

| Iter | Deliverable | Gate |
|------|-------------|------|
| 48 | Post-sort **subchunk table** + blend/cull **dispatch loop** over (tile, subchunk); `[SUBCHUNK]` stats; correctness first (may still use old reader internally for one subchunk) | `hero_vs_ref >= 63.6` (target 63.85) |
| 49 | Reader **single-buffer bulk L1 load** of subchunk payload; compute consumes L1 (no per-row emit) | same + `[L1LOAD]` zone timing |
| 50 | **2×32 B records per 64 B page** — pack/unpack in scatter (`sort_bin` / `buf_l1_recs`), bulk reader, and compute; stop using high 32 B as padding. Re-validate PSNR bit-identical to iter 49. Report `[PACK2]` bytes_loaded vs old. | `hero_vs_ref >= 63.6` (target 63.85) |
| 51 | **Double-buffer** overlap (prefetch subchunk N+1 while cull/blend N) | same + Tracy shows overlap |
| 52 | **MB saturation early-out** (gaussian-major useless; must be **microblock-major** resident T) | same; perf may improve |

## Anti-flounder rules (workers)

- One milestone per iter. **Stop when gate passes** or 45 min with no progress — report blocker with file:line evidence.
- **devrun.sh only** for device. rsync `render/` before build.
- Rebuild: `cmake --build render/build-tt -j 16` (configure once if missing).
- Verify: `python3 render/run.py --iter-dir ttw-NNN` (30 views, level-orbit cameras).
- **Build delta required** — `render/kernels` hash must move.
- **No CPU fallback** — hard-fail on overflow without subchunk plan.
- Keep renderer clean — no debug `#ifdef`; archive removed paths to `opt/render-alternate-paths.md`.

## Measurement

Report every iter: `hero_vs_ref`, `avg_frame_ms` (30 views), subchunk count, max subchunks/tile, % candidates in overflow tiles before/after. Iter 50+: `[PACK2]` pages_per_subchunk vs splat_count.

**Do not start iter 50 until iter 49 is kept** — 2-pack changes the record layout; prove the bulk-L1 path first at 63.85 dB.
