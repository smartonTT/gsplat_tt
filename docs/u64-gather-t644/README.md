# t644: device probe of the pfwc u64 unit gather (U3 precondition)

Branch `ttp/t644-u64-gather-probe`, code at 29a00c41 (on best-iter-227). Box bh-33 (p150, measurement
reservation), every sync+build+run under `ttp lock p100`. Drive: `drive.sh` / `remote_ab.sh`,
3 rounds per arm (orders forward / reversed / forward), 30 bicycle views, b2b = `render/run.py
--back-to-back` (20 passes, FOLD=1 config), lat = latency mode with `GSPLAT_TT_XVIEW_OVERLAP=0`
so the project stage is pfwc serialised. Raw outputs in `out/` (run 3); run-1/run-2 (broken
writer, see below) in `out-run1/`, `out-run2/`.

## Arms
- **base**: best-iter-227 as is (no chunk cull).
- **c1k**: existing chunk cull, whole-tile (1k) chunks.
- **u64id**: u64 cull (64-gaussian spatial units, t637 model), lists carried as unit ids.
- **u64**: u64 cull, 16 surviving units per pfwc tile read as 16 x 256 B per plane (the U3 gather).
Sort and emit order unchanged in all arms.

## Results (ms/view, 3 rounds)

| arm   | b2b ms/view (r1/r2/r3)   | lat avg ms | serialised project (pfwc) ms | host unit-list ms | pfwc device ms (project - host list) |
|-------|--------------------------|-----------:|-----------------------------:|------------------:|-------------------------------------:|
| base  | 8.455 / 8.446 / 8.455    | 9.08 | 2.37 | 0.00 | 2.37 |
| c1k   | 9.060 / 9.060 / 9.060    | 9.89 | 2.52 | 0.18 | 2.34 |
| u64id | 11.064 / 11.157 / 11.203 | 12.69 | 5.30 | 2.19 | 3.11 |
| u64   | 10.258 / 10.372 / 10.381 | 11.61 | 4.47 | 2.06 | 2.41 |

Round spread is under 0.15 ms in every arm; all differences above are 10x or more larger.
View 1 counts: 43138 surviving 64-units, 2698 u64 chunks of 16 units vs 3619 id chunks vs 3704
c1k tiles (u64 reads about 27% fewer chunks than c1k).

## Verdict: gate fails, U3 (u64g1) shelved

- The pfwc device saving from reading only surviving 64-units is nil: pfwc device time is 2.41 ms
  with the u64 gather vs 2.37 ms base and 2.34 ms c1k. Reading 27% fewer chunks does not shorten
  pfwc; the 16 x 256 B scattered reads cost about what the saved reads did.
- Host unit-list cost is 2.06 ms/view in this probe build (single-threaded list build; t637 assumed
  0.19 ms). Even taking the t637 figure and zero device cost, there is no device saving to net it against.
- b2b: u64 is +1.92 ms/view vs base and +1.31 vs c1k. Blend also got slower with any chunk cull
  (base 5.60, c1k 6.29, u64 6.01, u64id 6.40 ms serialised), which the t637 model did not include.
- Gate was >= 0.35 ms/view b2b saving net of host cost. Measured: about -0.04 ms pfwc device
  (a loss) before host cost. U3 is shelved with the t637 model (net 0.0-0.3, central 0.15 ms) as
  the upper bound, now tightened to <= 0 by this probe for the pfwc gather part.

## Correctness note
md5 per view (`out/md5-r0-dump-*.txt`): every chunk-cull arm differs from base on all 30 views,
including the existing c1k path, so the chunk cull is not bit-exact against base at best-iter-227.
u64id matches c1k on 3 of 30 views only; u64 differs from both. No PSNR or image check was done:
this is a timing probe, no iteration is recorded, and the timing verdict does not depend on it.
Runs 1-2 failed because the BRISC writer (role 0, even chunks) lacked the PFWC_UNIT_LIST /
PFWC_TLIST_PAGE defines and read unit lists as tile ids (fixed in 46a6a67a).
