# t479 working notes (for the next run)

State at hand-off 1:
- Branch ttp/t479-... fast-forwarded to f04fd67c (t470 results), plus this folder.
- CPU model `imbalance.py` (hero, orb_09, orb_19; 120 pfwc cores):
  per-core visible max/mean ply 1.24/1.21/1.27 vs morton 1.15/1.17/1.27; pairs max/mean
  ply 1.22/1.22/1.25 vs morton 1.21/1.21/1.22. So Morton gids do NOT worsen pfwc per-core balance.
  Big screen tiles (top 10% by pairs) draw pairs from 120 pfwc segments in ply order (top
  segment 1.2%) vs ~80 segments in Morton order (top segment ~8%, max 15%).
- Ties: bicycle.ply has only 4 duplicate-mean pairs, so the tie-order md5 change (254 px on hero)
  comes from equal fp32 depth keys of distinct gaussians. A reorder trick cannot restore today's
  tie order; it needs a per-record original rank and a tie fix-up after the materialize's stable
  32-bit depth radix (pfwc writer + onelaunch emit + materialize). Today's tie order is
  (((i >> 10) % C), i) for PLY index i, C = pfwc cores (30 bits). Deferred until part 2 shows the
  cull can beat off; ties cannot explain +0.46 ms blend (same gaussian set per tile).
- Profile: `drive479.sh HEAD` (detached t479p, run dir 1288) = Tracy device capture off vs ro on
  the p100 yyzo-bh-04, views 0:10; CSVs to out/dev-{off,ro}-c0.csv.gz. Analyze with
  `python3 opt/profiler/analyze_zones.py <csv> 10` (total and max_core per zone), compare arms.

State at hand-off 2 (run 1294):
- Tracy off vs ro (p100 yyzo-bh-04, views 0:10, out/zones-{off,ro}.txt), max-core ms/view:
  sort_ol_town 1.32 -> 2.01, sort_ol_emit (movers) 1.42 -> 2.07, pfwc 2.09 -> 2.38;
  tile_blend_sfpu 4.29 -> 4.29, tile_blend_load 4.26 -> 4.27 (blend kernels unchanged: the host
  "blend" stage growth is the sort tail inside it). mat_cull_mask, sort_subchunk_mat unchanged.
- Emit counters (GSPLAT_TT_OL_EMIT_PROF=1, out/town-{offp,rop}.txt), per TRISC launch:
  wfl (wait for the tile's previous run to leave L1) 0.06 -> 0.47 ms, rdy 0.006 -> 0.21 ms,
  net 179 cycles/record unchanged, same records. Cause: Morton gids send bursts of records to
  the same tile, so every 8th record waits for the mover's flush of the tile's single run buffer.
- Fix (889d76db+): GSPLAT_TT_OL_RING_DEPTH=D, ring of D records per tile, a run starts once the
  run D records back has left (fl[t] + D - R >= c). D=16 with R=8 needs +256 KB per mover (L1 has
  567 KB/mover used today, no room); R=4 D=8 keeps the L1 size, at 2x run words.
- Unit test tests/unit/test_sort_ol_town.cpp now covers (R,D) = (8,8),(4,8),(8,16),(2,8) and bursty
  tiles; the Mac linker is broken (SDK tbd), so bench479.sh compiles and runs it on the box, plus a
  too-lax-wait mutant that must fail.
- Running: drive479 t479b (run dir 1294): Tracy counters rop48, then b2b off/cull/cull48/off48.
