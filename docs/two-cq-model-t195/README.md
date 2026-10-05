# t195: model of hiding both host bridges (K2 -> sort_ol, sort_ol -> mat) with a second CQ

No device run. Code read at b92a28f (tip of smarton/tt-project-opt). Data:
`docs/tip-t194/out/run-r{1,2,3}-base.log` (untraced TTW_TIMING, 3 rounds),
`docs/tip-t194/out/tracy-gaps.txt` (traced gaps), and #155's A/B logs on commit d9842af
(`git show d9842af:docs/t155/out/run-r{1,2}-{base,off}.log`; not on this branch).

**Answer: modeled saving 0.43-0.52 ms/view untraced (central ~0.47). After the discount
from #155, the only measured precedent, ~0.26-0.52 (central ~0.35-0.40). The combined
untraced bridge is ~0.50 ms, which passes the #155/#171 rule (>= 0.3). Verdict: BUILD, as one
change with two kill switches and a paired A/B. Land only if the paired gain is >= 0.3 ms/view.**

## 1. K2 -> sort_ol bridge, split untraced (tip, 3-round mean)

Traced gap: 0.419 ms. What happens on the host between K2's end and sort_ol's start:

| part | ms/view | source |
|---|---:|---|
| projM read returns after K2 ends | ~0.01-0.02 | inferred, not timed separately |
| `gather_result`: `proj.depths.assign(M, 0)`, a new zeroed M-float vector | **0.100** (0.098-0.103) | PROJECT_STAGES |
| tile_assign host + project_other | 0.006 | TILE_ASSIGN_STAGES |
| `sort_pread` (64 B P read, blocking) | 0.022 (0.019-0.026) | SORT_STAGES |
| `sort_other`, the part before the enqueue (buffer checks, `noc_xy`, speed split, fold check) | 0.02-0.064 | SORT resid 0.064. Existing spans do not split pre-enqueue from post-mat time |
| sort_ol rt-args (110 cores x 2 kernels) + enqueue + launch | ~0.05 | `bin_emit` 3.036 - traced sort_ol busy 2.975 = 0.061, minus ~0.01 for the totals read to return |
| **total** | **~0.21-0.26 (central ~0.23)** | traced 0.419 |

Cross-check with #155 (older base, before the fold): when sort_ol was enqueued right behind K2,
`bin_emit` fell by 0.160 / 0.121 ms. That puts its bridge at ~0.16-0.19 ms. The tip has more
host code before the enqueue (#170 fold check, #174 speed split): `sort_other` 0.064 vs
0.036-0.043. So ~0.2-0.25 at the tip is consistent.

`gather_result` is pure waste. The vector only carries M: `sort_and_bin_tt` ignores the
pointer (`(void)depths`, `render/host/sort_device.cpp:3201`), and `render.cpp:255` only
uses `.size()`.

## 2. sort_ol -> mat bridge (tip, 3-round mean)

`bin_layout` 0.052 + `publish_host` 0.170 + `sort_mat` 0.049 = 0.271. Add ~0.01-0.02 for the
totals read to return and for mat dispatch: **~0.28-0.29 untraced** (traced 0.260). #184
modeled hiding it on its own at 0.25-0.29.

**Total exposed: ~0.49-0.55 ms/view untraced.** Both gaps are inside one view's `render()`.
`render/run.py` times each view's `render()` latency, so hiding them shortens the metric
directly. d2h is different (#171): it has nothing to overlap with.

## 3. Design (one change)

CQ0 (in order): pfwc -> K2 -> sort_ol -> mat -> blend. CQ1: reads and uploads only.

1. Enqueue sort_ol on CQ0 right after K2, with no host wait in between (#155's early
   enqueue, redone on top of the fold). The kernel reads P / P_pad from `ta_pairs_P` on
   device. It takes its mover ranges from the same formula K2 used
   (`pfwc_fuse::k2_range_speed`), or from a small page that K2 writes. That makes `fold == 1`
   true by construction. The static fold inputs (cores, row_pages, tiles) are known on the
   host before the enqueue.
2. Record event E after K2 on CQ0. On CQ1: wait for E, then do a blocking read of the projM
   page (M, P, overflow word) and the K2 count rows (0.9 MB, ~0.07 ms). Sum the totals on the
   host (~0.05; #184 §1).
3. On the host: `bin_layout` + `publish_host` (0.22). Do the uploads on CQ1, then
   `Finish(CQ1)`. Then set the mat rt-args and enqueue mat and blend on CQ0. Finally, do the
   blocking image read.
4. Drop the M-float `depths` vector (carry M as a scalar) and the separate `pread`
   (P arrives in step 2).

After K2 ends, the host work is ~0.45 ms (rows read, sum, bridge, mat enqueue), against
sort_ol's ~2.9 ms. That leaves ~2.4 ms of slack, so the host can also wait out sort_ol's
prefix phase before the CQ1 read. The prefix is contention-sensitive (t170).
API check (vendored tt-metal): `MeshCommandQueue::enqueue_record_event` /
`enqueue_wait_for_event` (`mesh_command_queue.hpp:140-146`) and `distributed::EventSynchronize`
are present. The second CQ is the `num_command_queues` argument of `create_unit_mesh`
(`render/host/device_state.cpp:93`). Non-blocking `EnqueueReadMeshBuffer` TT_FATALs (#155),
so use blocking reads on CQ1 only.

## 4. Estimate (untraced, ms/view)

| item | ms |
|---|---:|
| K2 bridge hidden (exposed 0.21-0.26, minus the 0.006-0.011 back-to-back launch left over) | 0.20-0.25 |
| mat bridge hidden (#184, updated to tip numbers) | 0.25-0.28 |
| CQ1 reads contending with the emit (0.9 MB vs >100 MB of emit traffic) | 0 to -0.03 |
| **modeled** | **0.43-0.52 (central ~0.47)** |

**Discount from #155.** #155 closed the traced K2 gap (0.507 -> 0.011) but reported only
"0.03 ms/view paired". Its logs show that number is pulled down by noise outside the bridge.
In round 2 the early arm had a d2h outlier (0.326 vs 0.195 ms) and blend moved by ±0.04. The
two stages the change touches, project + sort, fell by 0.105 and 0.089 ms in the two rounds.
That is ~0.1 ms realized out of a ~0.17-0.19 ms bridge, about 55%. Applying 55-100%:
**0.26-0.52, central ~0.35-0.40 ms/view (2.2-2.5%).**

## 5. Verdict: build

- The combined untraced bridge (~0.50) passes the >= 0.3 rule. The central estimate clears
  the 0.3 gate, even after the #155 discount. Neither half clears it alone (#155, #184).
- Risks:
  - Exact range parity between the K2 and the early sort_ol, so the fold cannot be undone
    after the enqueue.
  - The second CQ might change dispatch timing. Check that pfwc->K2 and mat->blend stay at
    ~0.006 in Tracy.
  - The board's tt-metal may differ from the vendored copy.
  - The pair-overflow word is now checked after sort_ol is queued. It must still hard-fail
    before mat.
- Measurement:
  - Kill switches `GSPLAT_TT_SORT_OL_EARLY` and `GSPLAT_TT_MAT_CQ1`.
  - 3 arms (off / early only / both), >= 3 paired rounds.
  - Report view_total and also project + sort host stages, which are not affected by d2h
    and blend noise.
  - md5 must match `md5-r82new.txt`.
  - Land if the paired view_total gain is >= 0.3.
- The cheapest piece stands alone if the rest fails: dropping the `depths` vector removes
  ~0.10 ms from the critical path on one CQ, with no device change.
