# Shelved levers: triage for a back-to-back re-check (task #468)

No device was used. Sources: each lever's task result and doc, `docs/b2b-gap/README.md` (#464),
#275 (b2b mode), #388/#393 (xview), `git merge-tree` of each lever branch against
`origin/smarton/tt-project-opt` at c7232e14.

## The key fact: when could the PNG save hide pfwc?

Only since xview became the default (#393, iter 211). Before that, `render()` enqueued pfwc itself
and ended with a blocking D2H, so pfwc was inside every latency window. #275 measured b2b and
latency at that time: 11.618 vs 11.639 ms/view, equal within 0.05 ms. #388 shows the change: with
xview, gather_wait dropped from 2.94 to 1.10 ms because pfwc moved into the gap between views.

So a latency verdict is blind to pfwc only if it was measured after #393 **and** the lever changes
pfwc time. In the list, only #408, #419, #423 and #428 come after #393. All four used CPU models or
Tracy device time, not the latency number, and only #423 touches pfwc (it used Tracy pfwc time,
which b2b sees in full). **No lever in the list was misjudged because the PNG save hid pfwc.**
(#433 and #439 were misjudged that way; they are being re-checked elsewhere.)

What did change is the bar: the old gate was 0.3 ms/view on 11-18 ms frames. The bar is now
0.15 ms/view on a ~9.3 ms b2b frame. Levers that measured 0.15-0.3 ms and missed only the old gate
are the real candidates.

## Worth a b2b device re-measure

### 1. #274 K2 on the idle TRISCs (`GSPLAT_TT_K2_TRISC=1`): expected 0.15-0.22 ms/view

- (a) Mode: latency before xview (= b2b, per #275), p100a, 3 swapped rounds: 11.71 -> 11.46 ms/view,
  **-0.245 ms/view**, md5 46a725ab on all 6 runs. Shelved only because it missed the old 0.3 gate.
- (b) Stage: K2 pairs/rows, inside the frame on the serial device chain (after gather, before
  sort_ol). Not pfwc, not a host stage. In b2b it is fully on the critical path (#464: device ~95%
  busy, K2 window 0.746 ms traced at 0.951 occupancy on bh-30).
- (c) Code: branch `ttp/t274-l2-use-k2-s-idle-triscs-or-fold-k2-into-` (a9dd06e), flag default off.
  `git merge-tree` onto the opt tip is **clean**. The K2 kernel files it changes
  (`tile_assign_scatter_seg.cpp`, `pfwc_fuse.h`) are unchanged on the tip since its base. Only
  `tile_assign_device.cpp` moved (#355 rows view, pair_guard, cap_grid), and it merges without
  conflict. Not built here (no device or remote build in this task).
  #291 (8b19709) on top adds split knobs `GSPLAT_TT_K2_TJ0/_TJ1/_FCHUNK` and a chunked fill. It
  gained nothing over #274, and its Tracy capture hung once, so use #274's head.
- (d) Upside: #274 cut ~0.245 off ~0.85 ms of traced K2 on the p100a. Today's K2 is about the same
  size (0.746 ms traced on bh-30 12x10), so expect 0.15-0.22 ms/view. It adds to the pfwc chunk
  cull: off-screen Gaussians make no K2 pairs, so the cull does not shrink K2.

A/B (measurement box under the existing reservation; never ird reserve/release; never disable
host-key checking):

```bash
# prep, on the Mac, own worktree
git -C ~/dev/gstt2 worktree add -b ttp/t<N>-k2trisc-b2b <wt> origin/smarton/tt-project-opt
cd <wt> && git merge --no-ff origin/ttp/t274-l2-use-k2-s-idle-triscs-or-fold-k2-into-
"$TTP_PROJECT/harness/bin/ssh-preflight" <host>
ttp lock p100 -- opt/sync_remote.sh <host> /localdev/smarton/t<N> HEAD
# on <host>, same build, alternate arm order over 3 rounds (0 1 / 1 0 / 0 1), env as in docs/b2b-gap/probe464.sh
GSPLAT_TT_K2_TRISC=$arm .venv/bin/python3 render/run.py --no-ref --back-to-back --iter-dir t<N>-k2t$arm
# one md5 pass per arm: add --dump-views t<N>-k2t$arm-dump, then opt/md5_golden.py on the md5 list
```

Gate: paired b2b ms/frame mean <= -0.15, md5 golden 30/30, device hero + diff checked by eye.

## Stay shelved

None of these was hidden by the PNG save. Each was measured before xview, or is in-frame mat, blend
or sort work, or is a device-time model.

| Lever | Mode it was judged in | Why it stays shelved |
|---|---|---|
| #176 mat mid-tile split | model + untraced latency, pre-xview | Mat only, capped at 0.26-0.30 on an old mat; #418/#425 since cut big-tile sort; #428 says stop mat. |
| #172 blend per-tile fixed cost | Tracy + model, pre-xview | Blend only; the cuttable part is ~0.1 ms (T read-back), which the saturation early-out needs. |
| #185 mask-0 record drop | model, pre-xview | Blend/mat only; net -0.13..+0.16 ms and not md5-exact. |
| #187 v2 mover table + t164 fold | latency, pre-xview (-0.196 on p100a) | Sort emit only; the table was replaced (#358: the p150 table loses), and the emit was rewritten (#418/#425). The branch conflicts in `sort_bin_onelaunch.cpp`. Scaled to today's 1.29 ms emit, the fold is ~0.06. |
| #191 tail-chained mask walk | latency, pre-xview (-0.163 blend, knob 1) | Blend only; TRISC1 was since rebuilt (#219 sched, #231 decode-ahead, #280 fused mat+blend). The branch conflicts in `alpha_blend_compute_mb.cpp`. Scaled to the 5.2 ms blend: ~0.11. |
| #215 emit TRISC cursors | model | Sort emit only; 0.10-0.20 is an upper bound, and the similar #165 split measured a loss. |
| #252 U2 blend coeffs | latency, pre-xview | +3.63 ms slower. |
| #165 emit pack on idle TRISCs | latency, pre-xview | +0.056 ms (slower). |
| #168 shared big-tile sort | latency, pre-xview | -0.038; superseded by #425's big-tile sort cut. |
| #171 host residue / d2h overlap | host profile, pre-xview | d2h is gone (pinned output, #393). The host bridge is hidden behind device work in b2b (#464: host has >=1 ms slack; emit->mat bridge idle 0.008 ms, #412). |
| #148 blend-waste work-cut | counters + model | Blend only; 0.19 ms upper bound, refined to 0.04-0.08 by #185. |
| #298 fold K2 into sort_ol | latency, pre-xview | +0.19 ms slower (the rectangle walk lands on the critical sort/mat movers). It merges cleanly, but there is no reason to retry. |
| #358 p150 emit mover table | latency on bh-30, pre-xview | +0.69 ms slower; per-core speed depends on the split. |
| #384 K2 pairs from pfwc writers | model from Tracy | Net ~-0.05. The model already counted writer growth on the critical path. b2b makes that cost fully visible, so it can only look worse, not better. |
| #362 K2 rows early on CQ1 | latency on bh-30 + p100a, pre-xview | 0.00 / +0.07. Sort shrank, but blend wait grew by the same amount: the device is the bottleneck, and in b2b even more so. |
| #385 sort emit into owner L1 | model | 0.04-0.17, does not fit in L1, needs device LPT. |
| #371 DRAM bank aliasing (pfwc writer seg_base) | model | The only pfwc item: <= 0.12 even if pfwc were DRAM-bound. #423/t232 put pfwc as TRISC-bound (TRISC 1.72-1.78 vs writers 1.25 ms traced), so ~0. Revisit only if the #433 pfwc-only cull re-check shows pfwc is writer-bound. |
| #408 microblock shape | CPU model | 4x8 is already the minimum-dispatch shape; the others are not bit-exact. |
| #419 blend interleave | model from Tracy | <= 0.115 after the mover cuts; L1 is aliased. |
| #428 mat stream | Tracy replay | 2-deep queue -0.040; worst-core excess barely reaches the frame. |
| #423 presort / publish | model + Tracy | Presort adds >= 1-3 ms to the serial chain. Device publish: the bridge idle is 0.008 ms. Its pfwc items (X fusion ~0.11, precull ~0.09-0.13) were judged on Tracy pfwc time, so b2b sees them in full, but they stay under 0.15 alone. They are per-Gaussian costs that shrink with the chunk cull (~0.07 after a 39% cull), so they do not add to it. |
| #155 early sort_ol | — | Not shelved: it landed as #198 (`GSPLAT_TT_SORT_OL_EARLY` default 1, 64d951e7). |

## Notes

- Flags present on the tip: only `SORT_OL_EARLY` (landed), `MB_STATS` and `MB_TILECYC`
  (instrumentation). All the other lever flags live only on their task branches.
- Clean `git merge-tree` onto the tip: #274, #291, #298, #252. Conflicts: #191, #187, #165, #168,
  #362, #358.
