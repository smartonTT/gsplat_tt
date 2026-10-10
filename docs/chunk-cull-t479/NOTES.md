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
