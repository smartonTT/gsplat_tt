# Task #199: tip measurement after #196 (bulk blendrec reads), with a fresh Tracy capture

Board: yyzo-bh-07 (Blackhole p100a, not a p150). Bicycle, 1024x1024, 30 views.
Code: 7a23beb (#196 bulk blendrec reads on top of the t194 tip), run as 3855aa2 = d78a457
(7a23beb plus #197 pfwc counters, off by default) plus these drivers. Default env.

Note: while this ran, #198 (2-CQ bridge hiding, iter 192) landed on top and is now the tip
(15.221 ms/view). This measurement is the 7a23beb reference point and matches #198's
"off" arm (15.639-15.680 ms/view untraced, all-core idle 0.723 ms/view traced).

## Untraced tip (3 rounds, warmup excluded)

| round | view_total | avg_frame_ms | project | sort | blend | d2h |
|---|---|---|---|---|---|---|
| r1 | 15.766 | 15.823 | 4.145 | 2.828 | 8.539 | 0.238 |
| r2 | 15.743 | 15.774 | 4.099 | 2.814 | 8.544 | 0.275 |
| r3 | 15.728 | 15.761 | 4.095 | 2.759 | 8.544 | 0.319 |
| mean | **15.746 ms/view (63.5 FPS)** | 15.786 | 4.113 | 2.800 | 8.542 | 0.277 |

Stdev 0.019 ms/view. All 3 rounds are md5-identical to md5-r82new.txt (46a725ab set;
checked by `remote_time.sh` with `diff` on the board, the chain stops on a mismatch).
Matches #196's measured 15.740. Against t194 (16.245): -0.499 ms/view, sort 3.392 -> 2.800.
Files: out/run-r*-base.log, out/md5-r*-base.txt.

## Tracy, 30 views (default chain, emit part counters off)

Program busy time, ms/view (`out/tracy-gaps.txt`):

| | pfwc | K2 | gap | sort_ol | gap | mat | blend | idle | traced end |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| t194 tip | 2.981 | 0.985 | 0.419 | 2.975 | 0.260 | 3.008 | 5.540 | 0.691 | 16.179 |
| **t199 (7a23beb)** | **2.981** | **0.986** | **0.452** | **2.395** | **0.259** | **3.009** | **5.538** | **0.723** | **15.632** |

Only sort_ol moved (-0.580 ms/view busy). The K2 -> sort_ol gap grew 0.033 ms.

sort_ol inside its 2.395 ms (zone makespans, `out/tracy-fill.txt`, views 1-29), us:

| prefix | barrier | fill (NCRISC) | fillb (BRISC) | emit |
|---:|---:|---:|---:|---:|
| 84.5 | 148.8 | 134.5 | 148.6 | 2286.2 |

Fill: 142 GB/s (8 B/pair). Emit per-mover duration: median 2.13 ms, max 2.74 ms
(t194: 2.53 / 3.62). Blend: tile_blend_sfpu makespan 5.38, tile_blend_load 5.28.
Mat: mat_cull_mask 2.84, sort_subchunk_mat 2.78. pfwc compute 2.95.

Share of the 14.91 ms device-busy time: blend 37%, mat 20%, pfwc 20%, sort_ol 16%,
K2 7%. All-core idle between programs: 0.72 ms/view (4.6% of the traced view).

## Untraced host bridges (TTW_TIMING, r1)

- K2 -> sort_ol: `project_gather_wait` 3.918 (blocking projM read, ends with K2),
  `gather_result` 0.105, `sort_pread` 0.018, `sort_other` 0.074, then rt-args + enqueue.
- sort_ol -> mat: `sort_bin_layout` 0.051 + `sort_publish_host` 0.170 + `sort_mat` 0.057
  = 0.278 ms, close to the traced 0.259 gap.

Both bridges are what #198 hides with its second command queue (traced idle 0.723 -> 0.029).

## Top remaining levers after #198 (by size; measured costs only)

1. **Blend, 5.54 ms.** Unchanged. MATH-bound records loop (#172).
2. **Mat, 3.01 ms.** Unchanged since t170.
3. **pfwc, 2.98 ms.** Unchanged; #197 has the breakdown.
4. **sort_ol emit, 2.29 ms makespan.** Mover spread now median 2.13 vs max 2.74 ms.

Reproduce: `docs/tip-t199/drive.sh` (Mac; each device step under `ttp lock p100`).
Tracy: `opt/profiler/ttw-193/render.tracy`
(copy of yyzo-bh-07:/localdev/smarton/gstt2-t199/opt/profiler/t199-tip/render.tracy).
