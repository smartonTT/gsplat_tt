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

- Record: 32 B in 64 B page (L1_RECORD path). `kBucketFit = 8192`.
- Overflow today: per-candidate DRAM gather in `reader_alpha_blend_mb_devcull.cpp` — delete this path for subchunked tiles once L1 path works.
- In-budget tiles (`count <= 8192`): may stay on current L1 bucket path until unified.

## Iteration slices (strict order)

| Iter | Deliverable | Gate |
|------|-------------|------|
| 48 | Post-sort **subchunk table** + blend/cull **dispatch loop** over (tile, subchunk); `[SUBCHUNK]` stats; correctness first (may still use old reader internally for one subchunk) | `hero_vs_ref >= 63.6` (target 63.85) |
| 49 | Reader **single-buffer bulk L1 load** of subchunk payload; compute consumes L1 (no per-row emit) | same + `[L1LOAD]` zone timing |
| 50 | **Double-buffer** overlap | same + Tracy shows overlap |
| 51 | **MB saturation early-out** (gaussian-major useless; must be **microblock-major** resident T) | same; perf may improve |

## Anti-flounder rules (workers)

- One milestone per iter. **Stop when gate passes** or 45 min with no progress — report blocker with file:line evidence.
- **devrun.sh only** for device. rsync `render/` before build.
- Rebuild: `cmake --build render/build-tt -j 16` (configure once if missing).
- Verify: `python3 render/run.py --iter-dir ttw-NNN` (30 views, level-orbit cameras).
- **Build delta required** — `render/kernels` hash must move.
- **No CPU fallback** — hard-fail on overflow without subchunk plan.
- Keep renderer clean — no debug `#ifdef`; archive removed paths to `opt/render-alternate-paths.md`.

## Measurement

Report every iter: `hero_vs_ref`, `avg_frame_ms` (30 views), subchunk count, max subchunks/tile, % candidates in overflow tiles before/after.
