# t173: re-check of the t164 emit fold after PRECULL=2 — no separate A/B, deferred to #177

No device run. Tip read: f39a023 (smarton/tt-project-opt, after #174 and #170).

## Did issue_brec become a larger share of emit?
- The emit loop is unchanged since t164 apart from #170 (keep plane is all ones, read from the window).
- Scan cost is unchanged: ep_brec on uncongested movers is 0.46-0.54 ms (t174-p2w,
  docs/mover-speed-t174/out/emit_cores-t174-p2w.txt) vs 0.507 mean at t160.
- On the movers that set the emit window it grew: BRISC on rows y=2,3 spend 1.0-1.3 ms in issue_brec
  out of 2.5-2.9 ms (t160 busiest: 0.63 of 2.76). That extra time is NoC0 read-issue stall, not scan.
  The v2 fold issues those reads from inside the pack loop, so pack work overlaps the stall. That is
  why the fold is worth more under PRECULL=2 than under PRECULL=1.

## Why no A/B here
#177 (mover-speed refit, running) already carries the t164 fold (a56e86b, rebased onto t170) and
gated it with its v2 table on c1d83a9 (pre-#170), yyzo-bh-07 p100a, 3 rounds, 30 views:
- fold on top of v2: -0.155 / -0.284 / -0.173 ms/view (mean -0.204)
- v2 + fold vs #174 tip: -0.482 / -0.593 / -0.458 (mean -0.511)
- all 9 runs md5-identical to the 46a725ab set.
A fold-only A/B here would use the #174 table, which #177 found keyed by the wrong core x and is
replacing. It would also take p100 time and race #177's push of the same kernel change. Landing goes
through #177. It must re-gate a56e86b on the post-#170 tip, because #170 changed the window fill
and the keep plane.

## Why EMIT_PROF=1 + fold hangs (t164 v2 capture, #177 t177-cand), not verified
The likely cause is a stack overrun. The fast loop keeps `uint32_t cur_lm[OL_RING_TILES]` (1024 words,
4 KB) on the stack, inside the 8 KB RISC local memory. EMIT_PROF adds about 14 accumulators and
timestamps that stay live across the loop, and the v2 fold adds about 7 more (s_g, s_dst, sj, n_sc,
skp, sgp, s_dst0). The spills grow the frame past the stack region, into the globals in local memory,
including the device profiler's own state. Untraced builds and the v1 fold (fewer live values)
capture fine, which fits this. The hang is after JIT warmup because that is the first real launch.
Check: compile with -fstack-usage, or read the frame size in the BRISC/NCRISC ELF disassembly,
for EMIT_PROF=1 with fold on and off. Fix if confirmed: move cur_lm to L1 when OL_EMIT_PROF=1, or
cut the EP_* counters for the fold arm.
