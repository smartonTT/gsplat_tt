# gsplat_tt 1 ms alpha-blend kernel — autonomous optimization design

**Date:** 2026-05-25
**Branch target:** `smarton/opt-v2` (new, off `origin/main`)
**Hardware:** Tenstorrent Blackhole P300 at yyzo-bh-14, single chip via `MESH_DEVICE=P100`
**Workload:** 3D Gaussian Splatting training-pattern rendering at 1024×1024
**Target:** kernel-only median ≤ 1.0 ms with no visible artifacts vs the iter-0 frozen reference

---

## 0. Verbatim instructions (preserved from kickoff)

These were the user's original instructions. Some have been refined during brainstorming; refinements are noted inline in [brackets].

- Optimize alpha blend kernels to finish in ~1ms per 1024x1024 frame to approach GPU CUDA performance
- Only kernel time matters (data movement and compute) — NOT CPU sort/binning time [refined: binning/sort optimizations are in scope as long as they don't introduce visible artifacts; the *target metric* is still kernel ms]
- May modify CPU code or port to metal if it speeds up alpha blend kernels
- Shift work from SFPU to FPU by refactoring math
- Study `/Users/smarton/Downloads/gs-fpu.md` proposal
- Propose best FPU/tile/face usage for user approval before starting [this spec is that proposal]
- Deep research loop into tt-metal and LLK best practices
- Run an infinite optimization loop as supervisor (Opus 4.7) of Sonnet subagents
- Check every 3 minutes that subagents are on track, valid screenshots, no artifacts
- Have a validator subagent ensuring results are valid
- Maintain `docs/optimization-log/REPORT.html` with running graph per experiment showing PSNR and timings (alpha blend kernel time most important)
- Start from main, wipe all products from previous agent attempts (Cursor + claude attempts failed)
- Pull from backed up branches ONLY: report.html format, screenshot views, 1024 resolution, viewer improvements
- Run iter 0; user confirms validity before proceeding [refined 2026-05-25: fully autonomous — no user gates anywhere in the loop; iter-0 reference is auto-frozen and the loop continues. User audits asynchronously via STATUS.md / BACKBURNER.md / REPORT.html]
- Iteration work on bh-14 (yyzo-bh-14), permanent viewer/no iteration on bh-30
- Keep viewer on localhost 8080 on bh-14 (dev), stable on bh-30 port 8081
- User audits ideas in plan before starting [this spec is the audit gate]
- One commit per successful experiment

**Refined constraints from brainstorming (2026-05-25):**
- **No visible artifacts is the primary gate, not a hard PSNR floor.** Validator must visually inspect every render + diff×10 image for tile seams, banded patterns, missing-splat holes, color clipping, ringing, NaN sentinels, geometry shift. PSNR is a numeric proxy used per optimization class (>40 dB for kernel-algebra, >35 dB for binning/sort/host-prep — revised 2026-05-25, see `2026-05-25-fpu-heavy-architecture.md`). The prior run failed because PSNR-only acceptance let tile artifacts through.
- **No blocking on the user, ever.** Every situation has a forward path. Reasoning lands in STATUS.md / BACKBURNER.md / REPORT.html for audit at the user's pace.
- **Agent never modifies `benchmarks/reference/*.png`.** Protected by a PreToolUse hook in `.claude/settings.local.json`.
- **Workload is training-pattern.** Benchmark cycles through 3 views to emulate per-frame view changes; view-invariant data is cached, view-dependent data rebuilt every frame.

---

## 1. Repository setup & iter-0 baseline

**Branch:**
- New branch `smarton/opt-v2` off `origin/main`.
- One "infra carry-over" commit cherry-picking ONLY these files from `smarton/opt-stable`:
  - Viewer: `gsplat/viewer.py`, `gsplat/camera_controls.py`, `gsplat/letterbox.py`, `gsplat/nerfview_viewer.py`, `gsplat/viser_patches.py`
  - Scripts: `scripts/render_fixed.py`, `scripts/derive_camera.py`, `scripts/derive_all_views.py`, `scripts/regen_tt_references.sh`, `scripts/rerun_all_keeps.sh`, `scripts/start_stable_viewer.sh`
  - Report generator: `docs/optimization-log/build_report.py`, `docs/optimization-log/report.css`
  - 1024-res wiring: `gsplat/__main__.py` `--force-square 1024` flag, `gsplat/rasterization.py` 1024 diff
- Wipe before commit: `docs/optimization-log/screenshots/`, all `iter-XXX-*.md`, `REPORT.md`, `REPORT.html`, `SUMMARY.md`, `SUPERVISOR-LOOP.md`.

**iter-0 baseline:**
- Commit message: `iter 0 BASELINE: main-kernel render at 1024², frozen reference`
- Render `scenes/stitch_doll.ply` at 1024×1024 with unmodified main-branch alpha-blend kernel from 3 fixed cameras (hero/side/top).
- Save: `benchmarks/reference/stitch_{hero,side,top}.png` — frozen, never modified by any agent. (1024×1024 is implicit; agent only operates at 1024.)
- Capture training-pattern timing baseline (see §3 measurement protocol). Expected ~106 ms.
- iter-0 PSNR vs reference = ∞ by construction.

**No user audit gate.** The supervisor proceeds straight from iter-0 to iter-1. If reference renders fail sanity (NaN, all-black, dim mismatch), supervisor retries 3× with diagnostics, then logs to STATUS.md and proceeds with whatever exists.

---

## 2. Kernel architecture (committed track)

Two layered changes, evaluated independently then composed. Each must hold the primary gate (no visible artifacts) and the class PSNR floor (>40 dB for kernel-algebra, revised 2026-05-25).

### Layer A — Dst-resident state

- Today's kernel does `tile_regs_acquire/release` per-Gaussian for R/G/B/T accumulators, spilling state back through CBs each iteration.
- Refactor: acquire Dst once per output tile, keep `R, G, B, T` live in `Dst[0..3]` across the entire Gaussian loop for that tile, release once at writer handoff.
- Tile state never round-trips through CBs → eliminates ~4 spill/reload pairs per Gaussian.
- Strict-equivalence: purely a storage relocation. fp32 Dst accumulation already matches fp32 acc, so numeric result is bit-identical in principle. Validator must still confirm.
- Risk: register pressure. P300 has 16 fp32 Dst tiles when `fp32_dest_acc_en=true`. R/G/B/T take 4; per-Gaussian temps must fit in remaining 12 across pipeline stages. If spilling, revert.

### Layer B — Basis-form Q with tile-local centered coords

- Today: `Q = a·dx² + 2b·dx·dy + c·dy²` recomputed per-Gaussian with global `dx = x - μ_x`, `dy = y - μ_y`.
- Refactor: expand to `Q = A·x² + B·xy + C·y² + D·x + E·y + F` where (A..F) are precomputed per-Gaussian on host, and `x²/xy/y²/x/y/1` are six **constant basis tiles** generated once per program launch using **tile-local centered coordinates** (`x_local, y_local ∈ [-15.5, +15.5]`).
- Why tile-local: iter-057's prior basis-form attempt used global pixel coords and PSNR collapsed to 14 dB from catastrophic cancellation in fp32. Tile-local coords are bounded → fp32 has plenty of headroom.
- Per-Gaussian inner loop becomes 6 `mul_tiles` with `acc_to_dest` plus `exp_tile` + alpha clamp. Heavy ops shift SFPU→FPU.
- If basis-form drifts below 100 dB after tile-local fix, it goes on BACKBURNER — never the kernel.

### Composition rule

- iter-1: Layer A only.
- iter-2: Layer B only (on top of main, not A).
- iter-3: A+B combined, if both pass.
- Each iter is one commit. Failed PSNR/visuals → revert, append to BACKBURNER with measured (ms, PSNR) pair.

### Out of committed kernel track (BACKBURNER-only)

- 4-face sub-tiling with per-face culling (alters what's rendered).
- bf16 storage swaps (measurable drift below 100 dB).
- SFPU→FPU substitutions that change op ordering of non-associative fp32 reductions.

### Parallel exploration track (Approach B+, BACKBURNER-only)

A separate subagent explores face-packing with per-face cull, producing measurements but no commits. Output appended to `BACKBURNER.md` with: experiment name, kernel ms, PSNR, why it fell below 100, suggested follow-up.

---

## 3. Host architecture

### Workload model

3D Gaussian Splatting training: scene loaded once, new camera every frame for thousands of frames.

- **View-invariant** (precomputed once at scene load, reused all frames): canonical per-Gaussian μ_world, Σ_world, opacity, R/G/B; derived constants like 2σ_max; tile-local basis tiles.
- **View-dependent** (rebuilt every frame): projected μ_screen, Σ_screen; per-Gaussian basis coefficients (A..F encode projected Σ); depth values; tile bbox; tile assignment; per-tile depth-sorted lists; the 10-fp32 per-Gaussian payload.

### Measurement protocol

- Sequence: `[hero, side, top] × 11 cycles`. 1 warmup cycle (3 frames) discarded, 10 measured (30 frames).
- Every frame fully reconstitutes view-dependent state — no same-view-repeat caching that would hide cost.
- Kernel ms = sum of alpha-blend compute kernel wall time across cores via tt-metal device profiler.
- Reported: median (primary), p99, per-view medians (hero/side/top).
- `kernel_ms_median` is the gating metric vs the 1 ms target.

### Per-Gaussian payload (Layer B)

- Today: 9 fp32 scalars per Gaussian (μ_x, μ_y, a, 2b, c, opacity, R, G, B).
- After Layer B: 10 fp32 scalars (A, B, C, D, E, F, opacity, R, G, B). A..F derived on host from (μ, Σ⁻¹).
- Reader signature: 9→10 fp32. CB indices shift; CT-args bump → JIT cache wipe required.

### Per-output-tile basis constants

- Generated once per program launch by a tiny init compute kernel filling 6 CBs with tile-local `x², xy, y², x, y, 1`. 6 × 1KB = 6KB CB footprint.
- Same for every tile, every view, every scene — pure compile-time invariant.

### Binning (in scope, gated on visual check)

- **Tighter tile bbox** from per-Gaussian (μ, Σ): use actual covariance for 3σ ellipse bbox vs conservative axis-aligned.
- **Skip-tile early-out:** zero-Gaussian tiles skip compute kernel launch entirely.
- **View-invariant `2σ_max`** precomputed once at scene load.

### Sort (in scope, gated on visual check)

- **Radix sort** per-tile lists by depth (faster than std::sort; technically reorders non-commutative fp accumulation but visually indistinguishable for matching keys).
- **Parallel sort across tiles** using available CPU cores.
- Note: per-frame depth sort because views change every frame. Sorted-list caching across frames is dropped (training pattern).

### Host/device overlap

- While device runs frame N's compute, host prepares frame N+1's sort/bin/payload. Pipeline depth ≥ 2 at the host/device boundary. If serialized, host prep cost shows up in steady-state cadence even with low kernel ms.

### 1024×1024 wiring

- `gsplat/__main__.py --force-square 1024` flag from opt-stable carry-over.

---

## 4. Fully autonomous supervisor / worker / validator loop

### Roles

1. **Supervisor (Opus 4.7, this Claude):** runs the outer loop, dispatches work, never edits experiment code directly, makes KEEP/REJECT decisions only after validator signs off, maintains BACKBURNER and REPORT.
2. **Worker (Sonnet 4.6, dispatched per iter):** given one hypothesis, edits gsplat_tt on Mac, devsync→bh-14, builds, runs 11-cycle benchmark, returns artifacts. Does not commit.
3. **Validator (Sonnet 4.6, fresh context per iter):** given only artifacts + §5 checklist, no worker reasoning. Returns KEEP/REJECT/NEEDS_REVIEW with cited check failures.

### Inner experiment loop (one iter)

1. Supervisor picks next experiment from queue (Layer A → Layer B → A+B → tighter-bbox → skip-tile → parallel-sort → host-device-overlap → ...).
2. Supervisor writes `docs/optimization-log/iter-NNN-<slug>.md` with hypothesis, expected ms, class tag, rollback plan.
3. Supervisor dispatches Worker. Worker returns binary path, screenshots, metrics, manifest of edited files.
4. Supervisor dispatches Validator with artifacts + §5 rules. Validator returns JSON verdict cold.
5. Supervisor reconciles per §4 decision matrix.
6. Return to step 1.

### Outer health check (every ~3 min via ScheduleWakeup)

- `tt-smi -s` exit 0 on bh-14.
- Worker process alive AND advancing (log mtime within 60s).
- Viewer 8080 responsive.
- Watcher waypoints fresh (<30s).
- On wedge: SIGTERM → 10s → ird reboot autonomously per `feedback-autonomous-loop` (Weka care).
- bh-30 untouched except for KEEP-promotion via `promote_to_stable.sh`.

### No-block decision matrix (universal rule: never wait for user)

| Situation | Supervisor action |
|---|---|
| iter-0 reference broken (NaN/black/dim) | 3 retries → log STATUS → proceed with what exists |
| Validator KEEP + faster | commit on opt-v2, advance |
| Validator KEEP + not faster | log "valid but not faster", no commit, advance |
| Validator REJECT | wipe worker state, append BACKBURNER, advance |
| Validator NEEDS_REVIEW | treat as REJECT for commit, append BACKBURNER with reasoning, advance |
| 3 consecutive REJECTs on same sub-track | abandon sub-track, log STALLED, jump to next queue item |
| Structured artifact detected (tile grid, bands) | auto root-cause iter; revert last KEEP that introduced regression |
| Build failure | retry 2× with JIT wipe → REJECT, advance |
| Worker stalls (>60s no progress) | SIGTERM → retry once → REJECT, advance |
| Device wedge | SIGTERM → 10s → ird reboot |
| 3 reboots in a row don't recover | switch to host-side investigation iters; health-check still polls |
| Ambiguous fork | pick one, log reasoning to STATUS, run it |
| Queue + BACKBURNER exhausted | meta-iter: mine candidates, propose new items |

### Anti-drift rules

- **One commit per successful experiment.** No bundled commits.
- **`benchmarks/reference/*.png` is write-protected** (PreToolUse hook in `.claude/settings.local.json`).
- **No experiments on bh-30.** Worker dispatch prompt forbids it explicitly.
- **Validator runs in fresh context every iter.**
- **Structured artifacts trigger root-cause iter** before resuming queue (don't pile experiments on broken base).
- **BACKBURNER is append-only and never merged by the agent.** User decides what graduates.

### Hard floor (only thing that protects user)

Never do anything that would lose Weka filesystem data or require full re-provisioning. Everything else is recoverable.

---

## 5. Validator rejection criteria

The validator runs in a fresh subagent context with no knowledge of what the worker tried.

### Inputs (and only these)

- 3 renders (hero/side/top, 1024×1024).
- 3 references (frozen iter-0 PNGs).
- 3 diff×10 images.
- `metrics.json`: `{kernel_ms_median, kernel_ms_p99, per_view_median, psnr_per_view, prev_best_kernel_ms}`.
- Optimization class tag.
- This §5 checklist.

### Visual checks (primary gate — REJECT short-circuits)

Any ✗ on a structural check → REJECT regardless of numbers.

1. **Tile grid seams?** Diff×10 shows lines at 32px or 16px spacing → REJECT.
2. **Tile-shaped uniform-fill blocks?** Constant-color 32×32 / 16×16 squares where reference has detail → REJECT.
3. **Missing-splat black holes?** Dark regions >~16px → REJECT. Sub-16px speckle OK.
4. **Color channel clipping bands?** Saturated stripes not in reference → REJECT.
5. **Ringing / halos** at high-contrast edges (structured ringing, not edge noise) → REJECT.
6. **NaN/Inf signatures:** spatially-correlated patches of pure black / white / magenta → REJECT.
7. **Geometry shift:** diff×10 shows object silhouette → REJECT.
8. **Diff×10 structure:** uniform speckle = ✓; structured patterns (grids, edges, bands) = ✗ REJECT.

### Numeric checks (after visual gate passes)

**Revised 2026-05-25** (user override, see `2026-05-25-fpu-heavy-architecture.md`): the
previous 100 dB "kernel-algebra" floor was overconservative — it rejected basis-form
iter-2 (47 dB / visually clean) for sub-LSB rounding noise that has no perceptual
effect. The new floor is **perceptual**: 40 dB minimum + visual checks.

| Class | PSNR floor | Below floor |
|---|---|---|
| kernel-algebra (Layer A/B) | >40 dB any view | NEEDS_REVIEW |
| precompute (view-invariant host work) | >40 dB | NEEDS_REVIEW |
| dispatch (skip-tile, launch order, overlap) | >40 dB | NEEDS_REVIEW |
| binning (tighter bbox, etc.) | >35 dB | KEEP if visuals pass; below 35 → NEEDS_REVIEW |
| sort (reorder accumulation) | >35 dB | KEEP if visuals pass; below 35 → NEEDS_REVIEW |
| host-prep (other view-dependent host work) | >35 dB | KEEP if visuals pass; below 35 → NEEDS_REVIEW |

### Per-view consistency

- Max per-view PSNR delta >20 dB → NEEDS_REVIEW.
- Max per-view kernel-ms ratio >2× → NEEDS_REVIEW.

### Timing

- `kernel_ms_median ≤ prev_best × 1.02` → progress/break-even.
- `kernel_ms_p99 > kernel_ms_median × 3` → NEEDS_REVIEW (suspicious tail).

### Output schema

```json
{
  "verdict": "KEEP" | "REJECT" | "NEEDS_REVIEW",
  "visual_checks": [
    {"name": "tile_grid_seams", "result": "pass" | "fail", "evidence": "..."},
    ... (all 8)
  ],
  "psnr_check": {"floor": 40.0, "actual": {"hero": ..., "side": ..., "top": ...}, "pass": true|false},
  "per_view_consistency": {"max_psnr_delta_db": ..., "max_ms_ratio": ..., "pass": true|false},
  "timing": {"median_ms": ..., "p99_ms": ..., "vs_prev_best_pct": ..., "pass": true|false},
  "reasoning": "one paragraph citing specific check failures if any"
}
```

Validator must cite specific failures. "Looks okay" is malformed and re-dispatched once; still malformed → REJECT.

---

## 6. REPORT.html structure

Regenerated by `build_report.py` after every iter from `iters.jsonl` (single source of truth).

### Top of page

- Title bar: branch, current HEAD, iters run, current best ms, target (1 ms), iter-0 baseline ms.
- Live status banner: running / paused / device-wedge.
- Links to STATUS.md, BACKBURNER.md, raw iter logs.

### Running graphs (above per-iter cards)

- `graph-kernel-ms.png` — x=iter, y=kernel ms median+p99, dashed lines at iter-0 baseline and target, points colored by verdict.
- `graph-kernel-ms-per-view.png` — hero/side/top medians on shared x-axis.
- `graph-psnr.png` — y=min PSNR across views; dashed at 40 dB and 35 dB floors.
- `graph-class-progress.png` — stacked bar: iters per class × verdict.

### Per-iter cards (newest first)

- Header: iter-NNN-slug · class · verdict badge · timestamp · commit hash · md-log link.
- Metrics: kernel ms (median/p99/Δ%), per-view medians, PSNR per-view (min highlighted), class floor.
- Thumbnails (3×3 grid): render / reference / diff×10 per view.
- Validator JSON rendered as 8 ✓/✗ visual checks + reasoning paragraph.
- Hypothesis & rationale from supervisor's iter-md.
- If REJECT/NEEDS_REVIEW: link to BACKBURNER entry.

### BACKBURNER section (collapsible)

- All REJECT + NEEDS_REVIEW + stalled sub-tracks.
- Entry: iter, slug, class, reason, metrics, thumbnails (hero + diff×10), promote-priority badge.
- ⭐ "high-promotion-priority" when NEEDS_REVIEW achieves ms < current best.

### Footer

- Link to `git log smarton/opt-v2` summary.
- Link to BACKBURNER.md, STATUS.md.
- `build_report.py` last-run timestamp.

### Update cadence

- Runs at end of every iter (including REJECT — user must be able to audit validator).
- JSONL append atomic (write-temp + rename).
- Graphs regenerated from JSONL each run.

### File layout

```
docs/optimization-log/
  REPORT.html
  report.css
  build_report.py
  iters.jsonl
  STATUS.md
  BACKBURNER.md
  iter-000-baseline.md
  iter-NNN-<slug>.md
  screenshots/iter-NNN/
    {hero,side,top}.png
    {hero,side,top}_diff10.png
    build.log run.log timing.jsonl
    metrics.json validator.json decision.json
  graphs/
    graph-{kernel-ms,kernel-ms-per-view,psnr,class-progress}.png
benchmarks/reference/
  stitch_{hero,side,top}.png   ← frozen at iter-0 (1024×1024 implicit)
```

---

## 7. Validation harness — scripts & infrastructure

### `scripts/run_iter.sh <iter_num> <slug> <class>`

Single-command worker entry point.

1. `git status --porcelain` must show clean working tree on opt-v2 (else refuse).
2. `devsync is-finished yyzo-bh-14` (gate build).
3. JIT cache wipe if any `backends/tt/.../alpha_blend_*.{cpp,hpp}` or CT-args header changed since `.opt-v2-last-build-commit` (gitignored sentinel).
4. `ssh yyzo-bh-14 "cd /proj_sw/user_dev/smarton/gsplat_tt && sudo ninja -C backends/tt/tt-metal/build metal_example_gaussian_splatting"`, timeout 180s, log to `iter-NNN/build.log`. Non-zero exit → `BUILD_FAIL` sentinel, exit 2.
5. `ssh yyzo-bh-14 "MESH_DEVICE=P100 python scripts/render_image.py --backend tt --size 1024 --warmup-cycles 1 --measure-cycles 10 --views hero,side,top --out-dir <iter_dir>"`.
6. SCP renders back to `docs/optimization-log/screenshots/iter-NNN/{hero,side,top}.png`.
7. `python scripts/compute_metrics.py iter-NNN` locally on Mac.
8. Append line to `iters.jsonl`.
9. `python docs/optimization-log/build_report.py`.
10. Exit 0, stdout = metrics.json path.

### `scripts/render_image.py` (extended)

- Args: `--backend tt --size 1024 --warmup-cycles 1 --measure-cycles 10 --views hero,side,top --out-dir <iter_dir>`.
- Loads scene once; holds view-invariant data across cycles.
- Cycles `[hero, side, top] × 11`; first cycle = warmup, rest measured.
- Per measured frame: kernel-only wall time via device profiler → `iter-NNN/timing.jsonl`.
- Final renders saved at end.
- On device error: `DEVICE_FAIL` sentinel + stderr, exit 3.

### `scripts/compute_metrics.py iter-NNN`

- Load 3 renders + 3 frozen references.
- PSNR per view in fp64 (so >100 dB representable).
- Generate `iter-NNN/{view}_diff10.png` (10× amplified absolute diff, clipped to [0,255]).
- Read timing.jsonl; compute median, p99, per-view medians.
- Write `iter-NNN/metrics.json`.

### `scripts/dispatch_validator.sh iter-NNN`

- Bundle: 3 renders + 3 references + 3 diff×10s + metrics.json + class + §5 text.
- Invoke Sonnet 4.6 subagent fresh context with strict prompt: artifacts as file paths, §5 rules, required JSON schema. **No worker hypothesis in prompt.**
- Capture → `iter-NNN/validator.json`.
- Schema-validate; malformed → re-dispatch once with stern prompt; still malformed → `MALFORMED_VALIDATOR` + REJECT.

### `scripts/decide_and_log.py iter-NNN`

- Read validator.json + metrics.json.
- Apply §4 reconciliation matrix → `iter-NNN/decision.json`.
- If commit: `git add` only worker-manifest files, then commit `iter NNN <slug>: <class> kernel=X.XX ms PSNR_min=YY.Y`.
- Update iters.jsonl (temp+rename) with verdict + commit hash.
- If REJECT/NEEDS_REVIEW: append BACKBURNER.md entry with thumbnail relative links.
- Update STATUS.md (recent decisions, current best, current sub-track, ESCALATIONS section if applicable).
- Re-run build_report.py.

### `scripts/health_check.sh`

Returns one of `OK` / `STALLED` / `DEVICE_HUNG` / `BUILD_STUCK`.

- `ssh yyzo-bh-14 'tt-smi -s'` exit 0.
- `ssh yyzo-bh-14 'pgrep -af metal_example_gaussian_splatting | head -3'` → mtime of log advancing within 60s.
- `curl -s -o /dev/null -w "%{http_code}" http://localhost:8080` returns 200.
- Watcher waypoint freshness (<30s) via `cat /tmp/watcher_waypoints.log`.

### `scripts/promote_to_stable.sh <commit-sha>`

Only path that touches bh-30. Manual or supervisor-promoted after sustained progress.

- Copy built binary from bh-14 to bh-30 `stable_viewer/`.
- SIGTERM → 10s → `start_stable_viewer.sh`.

### Anti-tampering hook (`.claude/settings.local.json` at gsplat_tt root, created at iter-0)

```json
{
  "permissions": { "defaultMode": "bypassPermissions" },
  "hooks": {
    "PreToolUse": [{
      "matcher": "Write|Edit",
      "hooks": [{
        "type": "command",
        "command": "jq -r '.tool_input.file_path' | grep -q '^.*benchmarks/reference/.*\\.png$' && echo '{\"decision\":\"block\",\"reason\":\"benchmarks/reference/*.png are frozen at iter-0; do not modify\"}' || true"
      }]
    }]
  }
}
```

### Per-iter artifact inventory

```
iter-NNN/
  build.log run.log timing.jsonl
  {hero,side,top}.png
  {hero,side,top}_diff10.png
  metrics.json validator.json decision.json
  BUILD_FAIL | DEVICE_FAIL | MALFORMED_VALIDATOR  (sentinel if applicable)
```

---

## 8. Stopping criteria & escalation

### Hard stops (loop exits)

1. **Target hit:** `kernel_ms_median ≤ 1.0 ms` AND validator KEEP AND held across 3 consecutive iters without regression. Loop writes final STATUS summarizing path + promotion candidates, terminates.
2. **User interrupts the supervisor.** Standard exit.
3. **Catastrophic environmental failure** that risks losing Weka data (the only hard floor from `feedback-autonomous-loop`).

### Soft re-strategy points (meta-iter, loop continues)

| Trigger | Action |
|---|---|
| Priority queue exhausted | Mine BACKBURNER for NEEDS_REVIEW with promising ms; propose variants. |
| 5 consecutive iters no ms improvement | Step back; switch to highest-expected-value remaining item not in current track. |
| >2× speedup since last meta-iter | Re-evaluate queue (items irrelevant at 16 ms may matter at 4 ms). |
| Plateau across 3 different sub-tracks | 1 ms may be unreachable under no-visible-artifact gate; shift to Pareto-curve exploration mode. |

### Graceful idle (queue + BACKBURNER mining exhausted)

Loop does not terminate; shifts to investigation mode:
- Re-render BACKBURNER items against current best base.
- Generate Pareto curve plot (kernel ms vs min PSNR) in REPORT.
- Run host-side analysis iters: kernel disassembly review, llk best-practices cross-check, propose new sub-tracks.
- Pick most promising candidate, run it.
- Continues indefinitely. Token usage logged per iter for cost audit.

### ESCALATIONS — no-block visibility mechanism

Append-only section at top of STATUS.md:

```
## ESCALATIONS (read these first)
- [2026-05-NN HH:MM] HIGH: 3 sub-tracks stalled — target may be unreachable under no-artifact gate. See BACKBURNER for ms-vs-PSNR tradeoffs.
- [2026-05-NN HH:MM] MED: NEEDS_REVIEW iter-047 achieved 1.8 ms PSNR 88/96/95 — visual check passed. High-promotion candidate.
- [2026-05-NN HH:MM] LOW: ird reboot count = 4 in last 12h, all recovered, trending up.
```

Old escalations stay with timestamps (audit trail); new go at top.

### Per-iter token budget

Supervisor logs per-iter token usage to iters.jsonl. Soft warning at >1M/iter; ESCALATIONS entry at >5M/iter (usually a subagent went off-rails). Loop is not killed.

### Cumulative time visibility

STATUS.md header always shows: loop start, current time, total iters, current best ms, target, "estimated iters to target at current rate" (linear extrapolation).

### User interaction model (final)

- Loop never asks, never waits, never terminates unless target is hit or environment is unrecoverable.
- User reads REPORT.html and STATUS.md at their leisure. ESCALATIONS at the top tells them what's important in 30 seconds.
- User may interrupt and redirect at any time; loop respects that and restarts cleanly.

---

## Open risks & known constraints

1. **1 ms target may not be reachable under no-visible-artifact gate.** Theoretical analysis: 1 ms ≈ 70 cycles/entry; `exp_tile` alone is ~80 cycles. Plan B is the Pareto-curve exploration mode, producing a curated BACKBURNER for user audit rather than a single KEEP.
2. **Dst-resident state register pressure** may force partial spills. Validator's per-view consistency check should catch any tile-boundary failures from this.
3. **Basis-form tile-local coords** assumes the algebra is bit-equivalent within fp32 in `[-15.5, +15.5]`. Numerical verification at iter-1 (or wherever Layer B first ships).
4. **Worker subagent context bloat** — each iter spawns a fresh worker, but a worker that reads too many files balloons tokens. ESCALATIONS surfaces this at >5M tokens/iter.
5. **Device wedge fallback to host-side work** must actually have host-side work to do — host-side investigation iters are designed for this but produce slower forward progress.
