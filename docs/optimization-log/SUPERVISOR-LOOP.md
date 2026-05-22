# Supervisor Optimization Loop — TT Alpha-Blend Kernel

> Living document. Every hour the supervisor re-reads this file, edits as
> needed, captures new ideas, and continues optimizing until the stopping
> criteria are met. **Do not rewrite from scratch — append, refine, mark
> obsolete.**

## Status snapshot

- **Hardware:** `yyzo-bh-14`, P300 reservation, **1 of 2 chips healthy**
  (`p300(1|2)`). Mesh descriptor pinned to `p100_mesh_graph_descriptor`.
- **Working tree (Mac):** `gsplat_tt @ e0a3640` — Phase 1 (static/dyn DRAM
  split + SCN1/FRM2 IPC + page-aligned grow). Mac drives via `devsync`.
- **Viewer policy:** always up at `http://localhost:8080` (Mac tunnel).
  Stop *only* for benchmarking; restart immediately after.
- **Background P300 poller:** `~/dev/tmp/p300_poll.sh` looking for
  `p300(2|2)`; will let us upgrade to a full-capacity chip mid-loop.

## Baseline (commit e0a3640, 2026-05-22T04:55Z, stitch_doll, 341,426 G)

Kernel-only median is `sub.blend.daemon_rt.device_kernel` — that's what
"theoretical peak" applies to.

| Resolution | visible | entries | kernel ms | daemon_rt ms | blend ms | total ms | kernel-FPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 480x640    | 256,558 |   832,049 |  58.5 |  61.5 |  82.2 | 153.3 | 17.10 |
| 1024x1024  | 280,007 | 1,591,353 | 106.5 | 126.8 | 168.5 | 293.8 |  9.39 |

Stage breakdown @ 1024x1024 (median ms):

| Stage | ms | Owner | Note |
| --- | --- | --- | --- |
| project        | 16.5 | CPU (Python) | per-Gaussian projection |
| tile_assign    | 13.4 | CPU         | LPT load balance + tile stamping |
| sort           | 93.3 | CPU         | **largest non-kernel cost** |
| blend.prep     | 19.2 | CPU         | 5-col dyn pack encode + bf16 tilize |
| blend.load_npy |  4.6 | CPU         | should be ~0 with shm-ring IPC |
| blend.save_npy | 17.9 | CPU         | image readback + reshape |
| blend.daemon_rt − kernel | 20.3 | TT host | dispatch + DRAM xfer |
| **kernel**     | **106.5** | **TT device** | **main optimization target** |
| **total**      | **293.8** | end-to-end | |

## Theoretical peak (1-chip Blackhole P300)

The kernel processes **1,591,353 (Gaussian, tile) entries** at 1024x1024
across **1024 tiles of 32x32** distributed over **80 Tensix cores**.

**Compute lower bound** (no memory stalls, perfect issue):

- Per (Gaussian, pixel) inner step: 5×fp16 covariance eval + 1 exp + 4 mul +
  3 mac for RGBA = **~20 effective FMAs** (with early termination,
  alpha-saturated pixels skip).
- Effective inner steps after early-term: assume on average **40 entries
  effectively touched per pixel** (down from ~1500 due to T<ε cutoff).
- Total: 1024×1024 px × 40 entries × 20 FMAs = **0.84 GFMAs**.
- 80 cores × ~5 FP16 FMAs/cycle × 1 GHz × 50% utilization = **0.2 TFLOPS
  sustained**. → 0.84 GFMAs / 0.2e12 = **~4 ms**.

**Memory lower bound** (DRAM-only):

- Static colors+opacity: 341,426 × 16 B = 5.5 MB read once per scene.
- Dynamic packs: 280,007 visible × 20 B = 5.6 MB per frame.
- sorted_gids: 1,591,353 × 4 B = 6.4 MB per frame.
- offsets: 1024 × 4 B = 4 KB.
- PX/PY: 2 × 1024 × 2048 B = 4 MB.
- Output: 1024×1024×3×2 = 6 MB write.
- Total per frame: ~22 MB. Blackhole DRAM ~1.2 TB/s → **0.02 ms**.

**Realistic engineering target** (accounting for: imperfect early-term,
half-utilized FPU, kernel dispatch, NoC moves, host overhead): **~15 ms
device_kernel @ 1024x1024**.

**Stopping criteria** (any *one* of these):
- Median 1024x1024 stitch_doll `device_kernel` ≤ **15 ms** (≈7× speedup).
- 3 consecutive iterations no-win (≤2% improvement) — declare convergence.
- Manual call from user.

(The 4 ms compute-only bound is acknowledged as physically below what the
host stack can hide; we won't chase below ~10 ms without a full host
rewrite.)

## Workflow — how iterations are run

Goals: every change is *measured*, *visually verified*, and *cheaply
revertible*. We use the `optimization-loop` skill philosophy: small
diffs, best-of-N parallel attempts when an idea has multiple shapes,
ALWAYS bench against the canonical baseline.

### Per-iteration recipe

1. **Pick the next phase** from the queue at the bottom of this doc.
2. **Branch:** create `opt/NNN-short-name` off current `smarton/opt-stable`.
3. **Implement** — keep the diff small. Write a one-line summary in the
   commit message. Do not touch UI / camera / viewer code in perf iters.
4. **Devsync to box** (Mac → `yyzo-bh-14`).
5. **Stop viewer**, run benchmark, **restart viewer** (in script form so
   we don't forget):
   ```bash
   ssh yyzo-bh-14 bash /tmp/run_baseline.sh   # both resolutions
   ```
6. **Visual gate:** the saved 1024x1024 PNG must look like the reference
   (`benchmarks/reference/stitch_hero_1024.png`) within image_diff
   tolerance. `scripts/image_diff.py` produces an amplified diff.
7. **Record** kernel/total ms in the table at the bottom of this file
   under "Iteration history".
8. **Decision:**
   - Improvement on `device_kernel` ≥ 2% AND visual gate passes →
     fast-forward `smarton/opt-stable` to this commit, write
     `docs/optimization-log/NNN-name.md`, increment.
   - Improvement < 2% or visual regression → discard branch, mark in the
     history as `KEEP=NO`, move on.
9. **If 3 in a row are no-wins**, fall back to the "ideas backlog" and
   pick the most speculative item to try.

### Best-of-N

For phases where multiple shapes are plausible (e.g. CB depths, tile
sizes, scratch placement), launch multiple subagents in parallel via the
`Task` tool with `subagent_type=best-of-n-runner`. Each gets its own
worktree+branch. We bench all variants serially (one device), pick the
fastest *that passes the visual gate*, discard the rest.

### Recovery rules (don't get wedged)

- Never `ird reboot --force`. Use `pkill -TERM` first, then SIGKILL only
  after a 10s grace.
- If the daemon segfaults → kill, restart viewer, re-bench. *Never*
  attempt to rescue a half-dead daemon's stdin/stdout.
- If ARC firmware wedges (TT_FATAL: not found) → `tt-smi --warm-reset`
  on the box; if that fails, wait for the cluster scheduler to recycle.
- Watcher: enable only when debugging an actual hang
  (`TT_METAL_WATCHER=5 TT_METAL_LOGGER_LEVEL=info`); the watcher itself
  costs ~5 ms per iteration.

## Optimization queue (in priority order)

Ordered roughly by predicted benefit / effort. Update freely as data
comes in.

### Tier-1 (kernel) — biggest expected gains

1. **Block-wide early termination + bit-pattern compare** *(was Phase 4)*.
   When all 1024 lanes of a tile have alpha ≥ 0.999, exit the per-Gaussian
   loop. Currently each lane masks itself but still pulls every CB. Should
   cut kernel ms 30-50% on dense tiles.
2. **Dst-resident R/G/B/T accumulators** *(was Phase 5)*. Keep the running
   color/transmittance in dst register file across the inner loop instead
   of round-tripping to CB_*_STATE every iter. Best-of-N: try 4-channel,
   3+1, and 7-register variants.
3. **Reader async coalescing + true PX/PY double-buffer + drop CB_SAT_MASK**
   *(was Phase 3)*. NoC reads are issued one entry at a time; batch them
   so the compute pipeline isn't waiting on DRAM. Drop CB_SAT_MASK
   (unused after the early-term refactor).
4. **16x16 tiles, conditional on Phase 4 saturation signal**
   *(was Phase 6)*. Smaller tiles give better load-balancing on dense
   regions; doubling tile count (4096 instead of 1024) should keep all
   80 cores fed. Speculative — could help or hurt depending on overhead.

### Tier-2 (kernel host)

5. **Persistent kernel + mailbox dispatch + async host prep overlap**
   *(was Phase 2)*. Currently every frame re-issues the workload via
   `EnqueueMeshWorkload`; a persistent kernel reading commands from an L1
   mailbox skips the dispatch overhead (~5 ms of the 20 ms host gap).
   Lets host prep overlap with device kernel.

### Tier-1 (CPU) — sort is 93 ms, that's table stakes

6. **Replace `numpy.argsort` for tile-IDs with bucket/radix sort.**
   `argsort(tile_ids)` is the dominant cost in `pipeline.sort` at 1.6 M
   entries. Tile IDs fit in 11 bits (<2048) → 1-pass counting sort, ~3
   ms instead of 93 ms. **Highest single-line ROI on the board.**
7. **Preallocated prep buffers** — every frame allocates a fresh np
   buffer for `dyn_packs`, `sorted_gids`, `px`, `py`. Reuse caches.
8. **Tighter `tile_assign`** — vectorize the per-Gaussian tile box
   iteration; currently a Python double-loop where it could be one
   `numpy.repeat`/`numpy.cumsum` block.
9. **Two-pass project** — first pass to compute the cull mask, second
   only on visible. Saves ~30% of `project` time.
10. **shm-ring IPC for image readback** — `save_npy` is 17.9 ms because
    we serialize a 6 MB image through a pipe. Move to shared-memory
    ring buffer.

### Tier-3 (algorithmic)

11. **Spatial index on Gaussians** — k-d tree or BVH built once, traversed
    per frame to skip Gaussians not in any tile.
12. **Frame-coherency cache** — most tiles change very little between
    frames; cache `dyn_packs` for tiles whose visible-Gaussian set hasn't
    changed.
13. **Adaptive precision** — far Gaussians at fp8/int8, near at bf16.
14. **Tile-major static-data layout** — pack
    static_colors_opacity per-tile so each tile streams a contiguous DRAM
    region rather than gathering through `sorted_gids[i]`.

### Tier-4 (stretch / risky)

15. **`project_gaussians` on device** *(was Phase 9)*. Eliminates the
    16.5 ms project + ~5 ms transfer-back. Big lift; only after the
    kernel is in shape.
16. **Re-bench all kept iters at 320x640 + 1024 + 2048**, write
    SUMMARY.md.

## Iteration history (newest at top)

| # | Branch / SHA | What | kernel ms (1024) | total ms (1024) | prep ms (1024) | sort ms (1024) | KEEP | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 029 | (in-tree, KEEP) | host pipelining: split `blend()` into `submit_frame()` (prep + shm-write + FRM2) and `recv_frame()` (OK11 + image read); pipelined bench loop runs frame N+1's pre-blend (project + tile_assign + sort) DURING frame N's daemon kernel. Capped Python MKL/OMP/OpenBLAS to 4 threads in the bench env (with default 12 threads, project's pytorch math contends with tt-metal's dispatch worker threads on the 6-core Ryzen and goes from 17 ms → 147 ms). | 106.7 | **157.0** | 17.9 | 59.1 | YES | total **−91.4 ms (−36.8%)** at 1024×1024, FPS 4.00 → **6.37**. At 480×640: total **133.2 → 78.4 ms (−41%)**, FPS 7.39 → **12.75**. By far the biggest single-iter win in the log. Daemon-kernel and visible/entries counts identical to baseline (no correctness drift). The win comes from overlap: pre-blend (~103 ms) now runs in parallel with daemon (~132 ms), so per-frame interval = max(pre-blend + submit, daemon + recv) ≈ daemon+submit. **Critical learning:** 1) thread-pool contention is a first-class concern when CPU and accelerator are on the same box — capping MKL solved an 8× project slowdown; 2) the strict request/response IPC means we can't pipeline submits, but we CAN overlap the entire pre-blend phase with the in-flight kernel — this is the only "free" parallelism available without daemon protocol changes. **Followups:** (a) lift prep+save_npy out of submit_frame and into pre-blend so they too overlap with daemon (iter 030; would save another ~20 ms by removing the post-recv work from the critical path); (b) consider explicit CPU pinning (`taskset` python to 0-3, daemon to 4-11) for cleaner isolation than the OMP cap. |
| 028 | (in-tree, reverted) | fuse Stage D1 (`alpha·T = contrib`) into Stage D2 producer using 4 dst slots: 4×`mul_tiles(CB_ALPHA, CB_T_STATE)` + 3×`mul_unary` in one acquire-block, packing dst[0..2] → CB_T_R/G/B and dst[3] → CB_CONTRIB | 106.5 | 251.8 | 17.8 | 56.5 | NO | kernel **−0.27 ms (−0.25%)**, total **+3.4 ms** (in noise band). Watcher probe ran clean (frame 0 produced visible=256558, entries=832049 — kernel correct, no L1 fault, no hang). The "−1 acquire block per Gaussian × 20k g/core" theoretical saving turned out to be ~13 cycles/block, not 100-150 as I estimated. **Lesson:** acquire/commit/wait/release on tt-metal compute is mostly hardware-managed and is NOT a real synchronization cost; per-Gaussian cycle budget is dominated by SFPU compute (especially `exp_tile<approx=true>` ≈ 70% of the budget) and other tile ops, not synchronizer overhead. Future kernel iters MUST reduce SFPU tile ops or skip work entirely (early term, more aggressive culling) — restructuring blocks alone won't move the needle. |
| 027 | (in-tree, reverted) | replace 3× `np.divide(c[i,j], det)` with one `np.reciprocal(det)` + 3× `np.multiply` for covariance inversion | 106.2 | 249.9 | 17.8 | 56.4 | NO | `prep` did improve as expected (-0.18 ms at 1024, -0.25 ms at 480×640 — 1 div is ~5× slower than mul on M=280k floats so the math checks out), but **total moved +1.5 ms** which is within the 3 ms run-to-run noise band. No measurable headline win, and the change isn't bit-exact (`1/det` then `*` vs single `/` differ in last-place rounding). Revert. **Lesson:** sub-millisecond `prep` micro-optimizations are below our detection floor at 1024×1024 — stop chasing CPU-prep cycles, the kernel (106 ms) and sort (56 ms) are where the budget is. |
| 026 | (in-tree, KEEP) | fold the `2×` from the Gaussian power expression `power = a·dx² + 2b·dx·dy + c·dy²` into the M-sized `cov_inv_b` precompute (one `*= -2.0` on ~280k floats) instead of doing `attr[:, 3] = 2.0 * cov_inv_b[gids]` on ~1.6M floats per frame | 106.8 | **248.4** | 18.0 | 55.7 | YES | total **−5.3 ms (−2.1%)**, save_npy 5.4→2.5 ms (−2.9, surprisingly large; removing the 6.4 MB temp from `attr[:, 3] = 2.0 * gather` allocation also frees pipeline timing downstream). Bit-exact: `×(−1.0)` then `×2.0` and `×(−2.0)` are both exact in fp32 (powers of 2 only change the exponent). PSNR identical. **Lesson:** look for per-pair scalars that can be folded into per-Gaussian precompute — the 5.7× expansion factor at stitch_doll scale makes that fold worth a measurable chunk. |
| 025 | (in-tree, reverted) | numpy reimpl of `get_tile_assignments` (replace `torch.repeat_interleave` chain with `np.repeat`, modulo→subtract, single `inv_ts` multiply) | 106.5 | 258.5 | 17.8 | 56.0 | NO | tile_assign **17.4 → 23.8 ms (+6.4 ms / +37%)**. Bit-exact (PASS on 5 cases up to M=280k). The torch `repeat_interleave` path on int32 is faster than `np.repeat` after `.astype(np.int64)` — the conversion copies dominate any savings from removing modulo. **Lesson:** torch CPU ops with int32 + MKL backend are already well-tuned; don't assume numpy is faster for tight vectorized loops. Future tile_assign wins must be algorithmic (e.g. AABB from cov diagonal instead of lambda_max circle to reduce P), not implementation. |
| 024 | `7a6ca88` + `9f3b3b2` | POSIX SHM IPC (`shm_in` + `shm_out`) replacing stdin/stdout pipe for FRM2 + image readback (gated by `GSPLAT_TT_USE_SHM=1`, falls back to pipe) | 106.6 | **253.7** | 18.3 | 56.6 | YES | save_npy **17.6 → 5.4 ms (−12.2)**, load_npy **4.7 → 0.78 ms (−3.9)**, blend **165 → 156 ms (−9.3)**, total **−5.3 ms (−2.0%)**. Daemon-side memcpy absorbs some of the saved Python pipe-write time, so headline `total` win is smaller than the IPC-stage win. Subagent transport errored on result delivery but committed work; cherry-pick recovered. **Followups:** (a) fix `BufferError`/`resource_tracker` SHM-leak warning at process exit (numpy views outlive `close()`); (b) consider double-buffering `shm_in` so daemon read can overlap next-frame Python copyto. |
| 019 | `144ca57` | preallocated per-pipeline scratch + lazy bf16→fp32 in `save_npy` | 107.1 | 259.2 | **18.5** | 56.5 | YES | prep -0.8 ms (-4.2%). PSNR 168 dB (bit-identical). Total within noise but no regression. Locks in shm-IPC scaffolding for future iter. |
| 017-C | `6a44a8b` (subagent), reverted | counting + per-tile depth quicksort (claimed 2.1×) | 106.6 | 284.8 | 19.6 | **84.8** | NO | **Sort got SLOWER** (+28 ms). Subagent's microbench was on synthetic input; real distribution at stitch_doll has many small buckets where the per-bucket overhead dominates. PSNR 168 dB (correct, just slow). |
| 022 | (in-tree, reverted) | project two-pass: depth/opacity cull before Jacobian + cov_2d | 106.6 | 259.9 | 19.3 | 56.8 | NO | `project` 16.9 → 18.3 ms. Slice overhead at 82% survival rate exceeded the saved Jacobian work. Visually bit-exact (PSNR 168 dB). |
| 020 | `b0daced` (subagent), reverted | reader NoC pipeline: 8B offsets coalesce + PX/PY overlap + static prefetch | — | — | — | — | NO | **Kernel hangs.** No frame produced; chip needed full `tt-smi -r all` to recover. Subagent diff has an NCRISC ordering bug. Re-attempt with watcher (`TT_METAL_WATCHER=5`) before relanding. |
| 018 | `3a8d815` (subagent), reverted | block-wide early term via `reduce<MAX,SCALAR>` over CB_T_STATE | — | — | — | — | NO | **Kernel hangs.** Probably the new CBs (CB_CONST_ONE, CB_T_MAX) interact badly with reduce_init/reduce_uninit + the existing Stage F binary_op_init. Re-attempt with watcher and a unit-test that does the reduce in isolation. |
| 017 | `01b321c` (subagent commit `a29890a` rebased onto baseline) | int64 composite sort key | 106.6 | **258.3** | 19.3 | **56.2** | YES | sort -39%, total -12% (-35 ms). PSNR 44.0 dB. Confirmed via clean-baseline rebench post-commit. Kernel unchanged. |
| 0   | `e0a3640` | baseline (Phase 1, end of first pass) | 106.5 | 293.8 | 19.3 | 93.3 | YES | Reference. (Pre-supervisor-loop sort path used `torch.argsort` on a float key.) |

(append iterations here)

### Active baseline for new iterations: post-iter-029 (`smarton/opt-stable`, with `GSPLAT_TT_USE_SHM=1` and `OMP_NUM_THREADS=4`/`MKL_NUM_THREADS=4`/`OPENBLAS_NUM_THREADS=4` set in bench env)
  Kernel **106.7 ms**, total **157.0 ms**, prep **17.9 ms**, sort **59.1 ms**, project **26.1 ms**, save_npy **3.2 ms**, load_npy **0.84 ms**, daemon_rt **31.1 ms** at 1024×1024 stitch_doll (post iter 029, host-pipelined).
  At 480×640: total **78.4 ms**, kernel 58.5 ms, project 25.7 ms, sort 29.8 ms, prep 9.3 ms, daemon_rt 1.9 ms.
  Theoretical lower bound at this kernel time = max(pre-blend, daemon_kernel) + small_recv_overhead ≈ max(103, 107) + 30 = 137 ms (we're 20 ms above that, of which ~17 ms is prep on the post-recv critical path — addressable in iter 030 by lifting prep into pre-blend).

### Stability + variance notes
* `save_npy` exhibits run-to-run variance of ~3 ms at 1024×1024 (5.4 / 7.4 / 2.5 across iter 024 / 025 / 026 benches that touched only Python CPU code unrelated to save_npy). Likely cause: pipeline timing depends on whether the daemon read of `shm_in` overlaps the prep of the next frame. **Until daemon-read overlap is implemented (followup to iter 024), require any save_npy or daemon_rt claim to be cross-validated with at least 2 bench runs.**
* `device_kernel` is rock-stable (106.5–107.0 ms across all iters) — kernel-only claims do not need re-runs.

## Hard-won lessons (post-018/020 hang)

- **Stale daemons hold the device lock silently.** `pkill -TERM` is not enough.
  Use `sudo pkill -KILL -9 -f metal_example` and **verify** with `pgrep -af`.
  If a kernel is mid-flight when its host gets killed, RISC firmware can wedge
  and require `sudo /opt/venv/bin/tt-smi -r all` (deeper than `-r 0`).
- **Sudo is needed for both rebuilds and resets** (`build/` is owned by root,
  `tt-smi -r` needs PCI access). Saved in `tmp/run_baseline.sh`.
- **Bench harness must check + cleanup before every run.** Implemented in the
  hardened `tmp/run_baseline.sh`.
- **A failing iteration can be REALLY costly**: hang → wedge → reset → ~3-5 min
  recovery. Future kernel-touching iters MUST run with `TT_METAL_WATCHER=5`
  on the FIRST validation bench so we catch the L1 fault directly instead of
  a silent hang.
- **NEVER re-dispatch a subagent without a forensic check first.** A "PING
  timed out" / "WritableIterable closed" / RPC drop is a *delivery-channel*
  failure, NOT a work-result failure. Subagents commit on the worktree
  branch *before* the parent receives the response. **Mandatory checklist
  before any re-dispatch:**
  1. `git -C <worktree> log --oneline -5` — is there a new commit on the
     iteration branch since the worktree was created?
  2. If yes → that's the answer; cherry-pick / bench / decide. **Do not
     re-dispatch.**
  3. If no → grep the subagent transcript at
     `~/.cursor/projects/.../agent-transcripts/<parent>/subagents/<id>.jsonl`
     for `commit -m`, `Write` calls, microbench numbers — if substantive
     work was performed, salvage from the worktree, not a redo.
  4. Only re-dispatch when the worktree is genuinely empty AND the
     transcript shows the agent never reached commit. Cost of a wrong
     redo: 5-10 min wasted compute + 1-2 ms of supervisor credibility.
- **One-line forensic command** to run for every "errored" subagent before
  even thinking about a redo:
  `git -C <worktree> log --oneline <base>..HEAD && wc -l <transcript>.jsonl`

## Ideas backlog (un-prioritized)

- Try `math_fidelity = LoFi`/`HiFi2` for the inner FPU ops.
- Replace `exp_tile<approx=true>` with a polynomial approximation in
  the early-term-guarded path.
- Use 16-bit tile IDs in DRAM to halve `sorted_gids` traffic.
- Investigate `noc_async_read_tile` vs current `noc_async_read` for
  packet alignment wins.
- Profile with `TT_METAL_DEVICE_PROFILER=1` (one frame) to find the
  longest-stall stage of the kernel.
- Try `fp32_dest_acc_en=false` for the inner loop (if HiFi3 with bf16
  DM is enough precision for stitch_doll).
- Investigate per-core 2D work mapping (8×10 or 10×8) vs flat 80-core
  list to better match Blackhole's NoC topology.

## Hourly cadence checklist

Set a calendar reminder; on each hour:

1. Re-read the **status snapshot** above. Update commit / chips healthy
   if changed.
2. Re-read **iteration history** since the last review.
3. If the last 3 iterations are all no-wins, switch to the **Ideas
   backlog** for the next pick.
4. Add at least one new idea to the backlog from what you've learned.
5. If the optimization queue is empty, mark the loop as **converged**
   and inform the user — even if we haven't hit 15 ms.
