# t164: fold the issue_brec keep/gid scan into the previous batch's emit loop — NOT LANDED

Board: yyzo-bh-07 p100a (IRD p150 reservation). Base: iter-184 (220ba33), bicycle, 30 views, untraced.
Kill switch in the code: `GSPLAT_TT_OL_EMIT_FOLD=0` (fold is on by default on this branch only).

## Idea
In `render/kernels/dataflow/sort_bin_onelaunch.cpp` (fast emit path), `issue_brec` scanned the
keep/gid planes of batch k+1 before issuing its brec reads (0.51 ms/view of the 2.61 ms emit zone,
docs/emit-diet-t160). The fold does that scan inside `process_batch` of batch k, so `issue_brec`
only issues reads. Same reads, same order: output is bit-identical.

- v1 (1795ba2): 2-element fold per pack step. Disassembly: ~14 scan-setup instructions re-run per pack element.
- v2 (e77c98c): one keep/gid page scanned per 8 pack elements.

## Results (avg_frame_ms, paired, fold vs FOLD=0 in the same round, arm order swapped each round)

| round | rev | fold | FOLD=0 | delta frame | delta bin_emit |
|---|---|---|---|---|---|
| r1 | v1 | 18.463 | 18.731 | -0.268 | -0.174 |
| r2 | v1 | 18.420 | 18.547 | -0.127 | -0.151 |
| r3 | v2 | 18.553 | 18.585 | -0.032 | -0.121 |
| r4 | v2 | 18.358 | 18.628 | -0.270 | -0.151 |
| r5 | v2 | 18.589 | 18.635 | -0.046 | -0.117 |
| r6 | v2 | 18.426 | 18.559 | -0.133 | -0.139 |

- v2 mean: -0.12 ms/view frame, -0.13 ms bin_emit. v1 mean: -0.20 frame, -0.16 bin_emit (2 rounds).
- All 6 rounds: -0.15 ms/view frame (0.8%). The bin_emit stage, which has less noise, says the same: ~0.13-0.16 ms.
- Arm order matters: the arm that runs second gains ~0.1 ms (r4/r6 vs r3/r5), so only the mean over swapped rounds counts.
- md5 vs md5-r82new.txt: ALL_VIEWS_IDENTICAL in all 12 runs (both arms, all rounds).

## Tracy
- v1 (out/v1): EMIT_PROF=1 capture valid. issue_brec 0.51 -> 0.002 ms, but process_batch absorbs it.
- v2: the Tracy capture timed out (the 450 s `timeout` killed the render right after JIT warmup, see
  out/v2/capture.log), so post-processing re-read v1's CSV. The v2 emit tables were stale and are not kept.
  Untraced v2 runs were fine. A hang specific to EMIT_PROF=1 + v2 was not investigated.

## Decision
Below the task's 0.2 ms paired gate and under the charter's ~1% bar. Not landed. The code stays on
branch `ttp/t164-sort-ol-emit-fold-issue-brec-keep-gid-sc` (local) for reuse. The scan work mostly moves
into the pack loop rather than disappearing: the emit zone is bound by pack/key build on BRISC+NCRISC,
not by read issue. A bigger emit gain needs the pack/key work itself to shrink (e.g. fold the count into
K2 / pair diet, docs/rerank-18ms.md) or move off the data movers.

Files: drive.sh (sync, swapped rounds, Tracy), remote_time.sh, remote_tracy.sh, out/v1, out/v2.
