# t177: emit mover speed refit + t164 issue_brec fold — gate

Board: yyzo-bh-07 (Blackhole p100a). Bicycle, 30 views 1024x1024, untraced, 3 interleaved rounds.

## Finding: the #174 table was keyed by the wrong core x
`emit_cores.py` prints Tracy core x (1..6, 11..15); the host matches the table against
`worker_core_from_logical_core` (translated x 1..7, 10..13). Record counts in t174-p2w and
t177-v1 show Tracy x 13..15 took the rows of x 11..13 and Tracy x 11..12 matched no row (always
the default share). So the #174 table balanced only columns 1..6 as intended. `fit_table.py` now
prints translated x; `test_mover_speed_keys` rejects rows outside the translated grid (the old
table fails it with 20 rows at x 14/15).

## Iterations
| table | fit | emit window (Tracy) | paired vs tip, ms/view | md5 |
|---|---|---|---|---|
| #174 (tip) | t166-p2f, Tracy x | 3.011 (t174-p2w) | — | 46a725ab |
| v1 | secant t166-p2f + t174-p2w, Tracy x | 3.091 | +0.104 / +0.040 / +0.150 (worse) | 46a725ab |
| v2 | t174-p2w + t177-v1, translated x | 2.789 | −0.164 / −0.403 / −0.153 (file override) | 46a725ab |

## Combined gate (c1d83a9: v2 built in + t164 fold, one build, three arms)
tip = fold off + #174 table (tables/v0.txt via GSPLAT_TT_OL_MOVER_SPEED_FILE); v2 = fold off; cand = default.

| arm | frame ms/view (r1/r2/r3) | mean | project | sort | blend |
|---|---|---|---|---|---|
| tip | 18.388 / 18.248 / 18.237 | 18.291 | 4.592 | 4.507 | 8.869 |
| v2 | 18.061 / 17.939 / 17.952 | 17.984 | 4.596 | 4.276 | 8.874 |
| cand | 17.906 / 17.655 / 17.779 | 17.780 | 4.592 | 3.990 | 8.896 |

Paired vs tip: v2 −0.327/−0.309/−0.285 (mean −0.307); cand −0.482/−0.593/−0.458 (mean −0.511, −2.8%).
Fold on top of v2: −0.155/−0.284/−0.173. All 9 runs md5-identical to the 46a725ab set (30 views).
Unit test: `test_sort_onelaunch_v2` all ok on the remote host.

Open issue: fold + GSPLAT_TT_OL_EMIT_PROF=1 hangs on device (Tracy t177-cand stuck after the first
one-launch; untraced runs are fine). No Tracy of the combined build yet.
Logs: out/gate-v1.txt, out/gate-v2.txt, out/gate-cand.txt.

## After the rebase onto t170 (K2 diet + count fold, iter 186)
Rebased as a56e86b. The combined gate stopped at the smoke run: cand (fold on) hung on the first
view (`timeout` rc=124; log ends after `[SORT] ONELAUNCH k2_fold=1`). With the earlier
EMIT_PROF hang, the fold is timing-sensitive and not safe to land. a6da4e4 makes it opt-in
(GSPLAT_TT_OL_EMIT_FOLD=1, default off) and drive3.sh gates the v2 table alone against the tip
(#174 table via file), two arms, 3 rounds, 30 views. Gate for v2 alone: >= 0.3 ms/view.

## drive3 smoke hang (run 493) and drive4
drive3 on a6da4e4 stopped at its smoke too: v2 alone (fold off) hung on view 0 (rc=124, log ends after
`[SORT] ONELAUNCH k2_fold=1`). #181's Tracy run on the same board hung just before (16:46-16:54, no CSV),
so the board state is suspect; but the v2 table also reaches the K2 count-fold split (shared
mover_speed_table()), which the pre-rebase gate never ran with. drive4.sh smokes tip, v2 and v2 with
GSPLAT_TT_K2_FOLD=0 (2 views each, every arm runs), then the drive3 gate if tip and v2 pass.

## drive4 result and drive5 (table only)

drive4 (e4562c0, yyzo-bh-07 p100a, 17:15-17:30): all three 2-view smokes hung
(rc=124): tip (v0 file), v2, v2 with GSPLAT_TT_K2_FOLD=0. The hang is not the v2
table. The t164 fold is dropped (kernel back to smarton/tt-project-opt, no
GSPLAT_TT_OL_EMIT_FOLD), so the branch now differs from the tip only in the host
speed table. drive5.sh smokes the real tip in its own dir (board check), then the
branch's v0 and v2 arms, and runs the drive3 gate if all pass.

## drive5 gate result (2026-10-04, yyzo-bh-07 p100a, ccb455b, untraced, 3x30 views interleaved)

Board check: real tip f4d91df smoke passed (rc=0), branch tip and v2 smokes passed. The drive4 hangs came from the t164 fold kernel edit, not the board or the table.

| round | tip (v0 table) ms/view | v2 ms/view | v2 - tip |
|---|---|---|---|
| r1 | 17.265 | 17.053 | -0.212 |
| r2 | 17.299 | 17.107 | -0.192 |
| r3 | 17.309 | 17.113 | -0.196 |
| mean | | | **-0.200** |

bin_emit drops ~3.67 -> ~3.43 ms (-0.24). md5 46a725ab on all 6 runs, ALL_VIEWS_IDENTICAL.

Decision: SHELVED. -0.200 ms/view is below the 0.3 gate. The 0.2 allowance needed the t164 issue_brec fold, which hangs with the K2 count fold after the rebase onto #170, so no combined arm exists. The v2 table stays on this branch (ttp/t177-...) for a later combined landing if a working emit fold or another emit lever comes along.
