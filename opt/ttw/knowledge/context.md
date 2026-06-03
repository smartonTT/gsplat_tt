# gsplat project knowledge pack — context / north star

**This is the gsplat (`gsplat_tt` / gstt2) project knowledge pack. It is read by
the tt-workflows loop at loop start IN ADDITION to the framework-generic
`tt-workflows/knowledge/`** (wired via `ttw.toml [project] knowledge_dir` /
`context_doc`). The framework docs are codebase-agnostic; this pack defines what
"the gate", "the timing metric", "the bench set", and "the pipeline" concretely
mean for gsplat. When the generic skills say "the project defines X", X is
defined here.

This file is the **canonical** statement of the gsplat mission/gate/metric. The
`[loop]` comment block in `ttw.toml` ("STAGE 2 MISSION / north star / regression
policy") is a convenience copy; if they ever diverge, this file wins.

## Mission / north star (ordered)

Implement `opt/host-free-l1-render-plan.md`: a host-free, work-queue,
L1-resident render loop.

1. **ZERO host in the render hot path** (`render_full_py`). No host CPU compute,
   no `Finish()` stage-locks, no per-frame H2D/D2H round-trips between project
   and blend. Device kernels merely *existing* is NOT the goal — the host must be
   out of the per-frame loop. (Final image D2H once per view is OK; everything
   between project and blend must be device-resident.)
2. **~1 ms/frame** (`goal_value = 1.0`, `goal_metric = ms_view`).
3. The **quality gate holds on every kept iteration** (below).

Rank every candidate idea by **"does this remove host CPU work or a stage-lock?"**
Pick only from: (1) move emulated soft-float math onto SFPU/FPU; (2) delete
`Finish()` stage-locks via persistent/fused kernels; (3) delete host compute
(prefix-sums, per-gaussian precompute, SoA repack) by doing it on-device; (4)
delete D2H/H2D round-trips. Host-side speedups, bit-exact micro-reorders, and
spin tuning are NOT progress.

## Quality gate (the project's gate metric + threshold)

- **Metric:** `hero_vs_ref` — PSNR (dB) of the rendered HERO view vs the saved
  reference image.
- **Threshold (keep-gate):** `hero_vs_ref >= 63.6 dB` (`ttw.toml [gate]
  metric_threshold = 63.6`, `metric_lower = 0`).
- **Anchor:** the known-good ideal path scores **63.85 dB**; a correct,
  bit-identical change re-verifies at exactly 63.85. Treat a drop below 63.85 as
  a regression to explain, and below 63.6 as a hard keep-gate failure.
- The HERO view is for the **screenshot + the PSNR/quality gate ONLY** — never
  quote hero render time as the frame time (see timing metric).

**Stage-2 regression policy:** perf regressions ARE acceptable when an iteration
moves the architecture toward the plan (on-device record format, buckets,
persistent kernels, phase barriers) — later milestones offset the cost. The ONLY
hard keep-gate is the quality gate (`hero_vs_ref >= 63.6`). Do NOT reject a
milestone for being slower; reject only if it breaks quality or is
architecturally unsound. `stuck_after` is relaxed (8) so a temporary perf
plateau/regression does not thrash into `tt-debug` — use supervisor judgment.

## Timing metric (the project's representative AVERAGE timing)

The canonical perf number is the **AVERAGE FRAME TIME over ALL 30 bench views** —
the 30 views in `cameras_v2.json` order. Render every one of the 30 views,
**exclude the warmup view**, and report `avg_frame_ms` (+ p50/min/max if cheap)
as the headline.

REJECT these as misleading, never quote them as the frame time:
- a **single hero frame** — one (possibly unrepresentative) view; it is for the
  screenshot + PSNR only;
- the amortized **`@8v` / `ms/view`** figure — 8-view amortized ms/view hides
  per-frame host bubbles that overlap across views (e.g. a PR once claimed
  "~170 ms/view @8v" while the real single-frame latency was ~300 ms).

The 30-view bench exists specifically to measure average frame time. User
directive: *"the whole reason I gave you a 30-view test is to measure average
frame time. Start reporting average frame time from now on, always rendering all
30 test frames. Hero is for screenshot, not timing."* `render/run.py` renders the
30-view average.

## Render pipeline stages

The host-free render chain, in order:

```
project -> pfwc -> gather -> tile_assign -> sort -> cull -> blend
```

Per-stage device timing is emitted as `TTW_TIMING <stage>=<ms>` for
`proj/ta/sort/cull/blend` (the loop records these). Key stage facts (see the
plans under `opt/` for depth):
- **project** (`proj`): means_cam + pfwc (matmul-shaped, uniform/cheap) + the
  `gather_visible_scatter` compaction (the dominant, WRITE-skew-prone sub-kernel).
- **tile_assign** (`ta`): K1 bbox + K2 scatter (already balanced) + K4 keep-mask.
- **sort**: depth radix; on the L1_RECORD path the host `sort=` timer can LUMP
  cull+blend (see lessons.md) — the true DRAM radix is small.
- **cull**: SFPU microblock Mahalanobis keep-test — irreducible SFPU compute,
  shares SFPU cores with blend.
- **blend**: the dominant cost; microblock-major blend + transmittance early-out
  is the lever (currently gaussian-major with no early-out).

## Bench / inputs

- **Bench set:** 30 views, `cameras_v2.json` order (the bicycle scene). Resolution
  and inputs are locked across iterations (kernel ms scales super-linearly with
  H×W).
- **Hero view:** the screenshot + PSNR view only.

## Build artifact / device layout

- pybind artifact Python imports on-device:
  `backends/cpu_cpp/_gsplat_cpu.cpython-310-x86_64-linux-gnu.so` (`gsplat_tt`
  static-links into it) — fingerprinted by `ttw.toml [build] bin_artifact`.
- on-device JIT'd kernel source tree: `src/gsplat_tt/kernels` (`kernel_src_dir`) —
  also fingerprinted, because kernel `.cpp` edits leave the `.so` byte-identical.
- device entrypoint: `scripts/a003_verify.py` (enforces the devrun lock; refuses
  to open the device unless `TTW_DEVRUN=1`).
- the gsplat render hot path `std::abort()`s on a device-stage failure — there is
  no host fallback, so post-chain host reads can assume device success.

## Milestones (M0..M6)

The Stage-2 milestone plan lives in **`opt/host-free-l1-render-plan.md`**
(milestones M0..M6, each behind its own validation gate). Drive them in order.
Related deep-dive plans: `opt/plan-high-utilization-pipeline.md`,
`opt/blend-data-movement-plan.md`, `opt/host-free-fusion.md`.

## gsplat env flags (the render configuration)

The verify/Tracy config is a fixed set of `GSPLAT_TT_*` env flags (see `ttw.toml
[run] verify_cmd`): the device project/gather/tile_assign/sort/blend resident
path (`GSPLAT_TT_DEVICE_*`, `GSPLAT_TT_RESIDENT_*`), `GSPLAT_TT_SFPU_CULL=1`,
`GSPLAT_TT_TILE_BUCKET=1` + `GSPLAT_TT_BUCKET_FIT=8192`, `GSPLAT_TT_FUSED_TILE=0`,
`GSPLAT_TT_L1_RECORD=1`, plus the device-scan flags. The IDEAL path (the one the
loop gates/optimizes) is TILE_BUCKET + L1-resident, host out of loop — NOT
FUSED_TILE and NOT soft-float `MB_DEVCULL` cull.

## Tracy deliverable (gsplat-specific capture)

The per-iteration Tracy trace is ALWAYS the FULL **30-view** render (the fast
gate `verify_cmd` stays `--views 1`). Capture via
`opt/profiler/capture_tracy.sh <iter-dir>` (`ttw.toml [profile]
tracy_capture_cmd`), which runs `python -m tracy --dump-device-data-mid-run` —
the only path that streams DEVICE zones, because gsplat never closes the device
(a plain `capture-release` gets host/JIT-warmup zones only). Confirm 30-view
coverage via the dumped `profile_log_device.csv` row count (~30× the ~65.6k
1-view baseline; e.g. ttw-043 = 1,016,801 rows / 16.3 MB / 1.58M zones). A 1-view
or warmup-only `.tracy` is NOT a satisfied gate.

## Report assets / git policy (gstt2-specific)

The report's linked assets (`opt/REPORT.html`, the ledger `opt/ttw/iters.jsonl`,
every iter's `opt/metal-screenshots/ttw-*/*.png`, `opt/profiler/ttw-*/render.tracy`)
must be TRACKED so the report stays valid forever. NEVER blanket-ignore `opt/`
(it is ~2.7G) — `gstt2/.gitignore` ignores all of `opt/` then re-includes exactly
the report deliverables; `gstt2/.gitattributes` routes `opt/**/*.png`,
`opt/**/*.tracy`, `*.ply` (and `tests/fixtures/hero/*.npz`) through git-LFS. Junk
(loose `*.json`, logs, `*.npy`, stale pre-ttw `metal-iter-*`/`amendment-*` dirs)
stays ignored and belongs in `tmp/`. See lessons.md for the full story.

## Remotes

`origin` = smartonTT/gsplat_tt (push target), `upstream` = Kovelja009 (optional
fetch). If a clone has them swapped: `git remote rename origin upstream && git
remote rename smarton origin`.

## Active sprint: L1 subchunk cull/blend

Read **`opt/ttw/knowledge/l1-subchunk-sprint.md`** — post-sort subchunk payloads,
pure-L1 cull/blend (single-buffer then double-buffer), then mb saturation EO.
Implement only in `render/`. Loop uses `render/run.py` + `render/kernels` build-ID.

## Clean renderer policy (`render/`)

The readable TT renderer lives in `render/` (`render_clean`). Keep it clean through
every optimization iteration:

- **One live path only** — no debug `#ifdef`s, no alternate algorithms in-tree.
  Removed alternates go to `opt/render-alternate-paths.md` with a git ref to recover.
- **No CPU fallback** in the hot path — unsupported inputs hard-fail.
- **Kernels stay one file per kernel** (tt-metal JIT path requirement); host may
  stay one `*_device.cpp` per stage until a safe consolidation pass lands.
- **Gate still holds:** `hero_vs_ref >= 63.6 dB` on the hero view after every change.

## Subagent model (free tier)

When out of paid credits, dispatch Task subagents with **`model="composer-2.5"`**
(non-fast). **`composer-2.5-fast` is not allowed on the free tier.** Do not use
`model="auto"` when you intend the free Composer worker.
