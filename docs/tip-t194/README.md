# Task #194: tip measurement after #181 + #188, with a fresh Tracy capture

Board: yyzo-bh-07 (Blackhole p100a, not a p150). Bicycle, 1024x1024, 30 views.
Code: 4d591f7 (tip of smarton/tt-project-opt, includes #181 bulk fill and #188 blend
late claim), run as 24556e8 (same code plus these drivers). Default env.

## Untraced tip (3 rounds, warmup excluded)

| round | view_total | avg_frame_ms | project | sort | blend | d2h |
|---|---|---|---|---|---|---|
| r1 | 16.236 | 16.269 | 4.092 | 3.389 | 8.547 | 0.198 |
| r2 | 16.271 | 16.300 | 4.089 | 3.415 | 8.553 | 0.204 |
| r3 | 16.227 | 16.260 | 4.094 | 3.373 | 8.540 | 0.210 |
| mean | **16.245 ms/view (61.6 FPS)** | 16.276 | 4.092 | 3.392 | 8.547 | 0.204 |

All 3 rounds are md5-identical to md5-r82new.txt (46a725ab set; checked by `drive.sh`
with `diff` on the board, the chain stops on a mismatch).

The tip is better than the expected ~16.6: #181 was measured on a base without #188
(17.001 -> 16.659). Its -0.342 and #188's -0.381 add up: 17.001 - 0.342 - 0.381 = 16.28,
so 16.245 is what the two separate gains predict.
Files: out/run-r*-base.log, out/md5-r*-base.txt.

## Tracy, 30 views (default chain, emit part counters off)

Program busy time, ms/view (`out/tracy-gaps.txt`), against #170's base arm (t170):

| | pfwc | K2 | gap | sort_ol | gap | mat | blend | idle | traced end |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| t170 base | 2.981 | 0.985 | 0.422 | 3.547 | 0.305 | 3.008 | 5.868 | 0.740 | 17.129 |
| **t194 tip** | **2.981** | **0.985** | **0.419** | **2.975** | **0.260** | **3.008** | **5.540** | **0.691** | **16.179** |

sort_ol inside its 2.975 ms (zone makespans, `out/tracy-fill.txt`, views 1-29):

| prefix | barrier | fill (NCRISC) | fillb (BRISC) | emit |
|---:|---:|---:|---:|---:|
| 0.085 | 0.149 | 0.135 | 0.148 | 2.862 |

Fill: 142 GB/s (8 B/pair). Emit per-mover duration: median 2.53 ms, max 3.62 ms.
Blend: tile_blend_sfpu makespan 5.37, tile_blend_load 5.29. Mat: mat_cull_mask 2.84,
sort_subchunk_mat 2.78. pfwc compute 2.95.

Share of the 15.49 ms device-busy time: blend 36%, mat 19%, pfwc 19%, sort_ol 19%,
K2 6%. All-core idle between programs: 0.69 ms/view (4.3% of the traced view).

## Untraced host bridges (TTW_TIMING, r1)

- K2 -> sort_ol: `project_gather_wait` 3.922 (blocking projM read, ends with K2),
  `gather_result` 0.103, `sort_pread` 0.019, `sort_other` 0.061, then rt-args + enqueue.
- sort_ol -> mat: `sort_bin_layout` 0.049 + `sort_publish_host` 0.165 + `sort_mat` 0.049
  = 0.263 ms, matching the traced 0.260 gap.

## Top remaining levers (by size; measured costs only, no gain estimates)

1. **Blend, 5.54 ms.** Biggest program. #148: waste is small (~0.43 ms). #172: MATH time is
   98% the records loop (64% microblock dispatches at ~82 cycles each, 30% live-record
   staging at ~108 cycles each). A lever must cut the per-dispatch or per-record cycle
   cost (e.g. FPU for the alpha/weight math, larger blocks per dispatch), not count work.
2. **Mat, 3.01 ms.** Unchanged since t170. Mid-tile split is shelved (#176). Open: fold
   the cull mask into the sort_ol emit or the blend load, to drop a full pass over P pairs.
3. **pfwc, 2.98 ms.** Unchanged since #122. Its compute zone covers 2.95 ms on all 110 cores;
   needs a fresh breakdown (SFPU vs unpack/pack vs reader) before picking a lever.
4. **sort_ol emit, 2.86 ms makespan.** Mover imbalance: median 2.53 vs max 3.62 ms per mover.
   A better range split across movers could cut toward the median.
5. **Host bridges, 0.69 ms/view traced (K2 -> sort_ol 0.42, sort_ol -> mat 0.26).**
   #184 shelved the mat bridge alone (0.25-0.30). Hiding both with a second command
   queue is one change (the K2 rows and P are known after the existing projM read) and
   now the largest pure idle block; worth a model that splits the 0.42 traced gap untraced.

Reproduce: `docs/tip-t194/drive.sh` (Mac; each device step under `ttp lock p100`).
Tracy: yyzo-bh-07:/localdev/smarton/gstt2-t194/opt/profiler/t194-tip/render.tracy.
