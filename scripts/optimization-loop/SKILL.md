---
name: optimization-loop
description: Drive the gsplat_tt alpha-blend kernel optimization loop end-to-end. Pick the next idea from the current profile, dispatch a worker subagent on Composer 2.5 to implement it, gate the change with permissive visual regression checks, decide keep/flag-for-review/hard-reject, and log every iteration. Use when the supervisor is starting, advancing, or recovering the optimization loop.
---

# optimization-loop

The supervisor's playbook for advancing the alpha-blend kernel optimization one iteration at a time. References [tt-profile](../tt-profile/SKILL.md) for kernel measurement and [image-diff](../image-diff/SKILL.md) for visual regression gating.

## Architecture (one-line)

Supervisor (this chat, **Opus 4.7**) reads the current kernel profile, picks the next idea, dispatches a worker `Task` subagent on **Composer 2.5**, the worker edits + builds + benches + image-diffs, then the supervisor decides keep/flag/reject and writes the iter log.

## Candidate idea pool

Initial priorities (the supervisor re-ranks every iteration based on what the latest profile shows is dominant):

1. **Reader-signaled block-wide early termination** — when all 1024 pixels of a tile saturate (T < 1e-4), the reader stops streaming. Foundational because without it the binning log over-counts work.
2. **Per-Gaussian bbox / per-pixel coverage skip** — most Gaussians cover < 20% of their assigned 32×32 tile, but the kernel computes Q over all 1024 pixels.
3. **`addcmul_tile` fusion for R/G/B** — the kernel does 6 separate acquire blocks for the three color channels; the SFPU has a fused `a + b*c` op.
4. **Dst-resident R/G/B/T accumulators** — Spec Rev 2 decided to spill to L1 between Gaussians; the FPU optimization notes argue this should be revisited.
5. **16×16 binning** — half the pixels per tile, four tiles per hw-tile (parallel pipeline of 4 work units).
6. **Fused quadratic form via FPU** — `a·dx² + 2b·dx·dy + c·dy²` is three muls + two adds; the FPU can possibly run a smaller matmul-shaped op.
7. **Hand-written LLK if Dst-resident state is still bottlenecked.**

Add or drop items as the loop learns. The supervisor's first move every iteration is to **re-rank**, not to follow this list mechanically.

## Per-iteration recipe

The supervisor performs steps 1, 2, 5, 6, 7 directly. Step 3 (dispatch) launches a worker that performs step 4 autonomously.

### 1. Profile current best

Invoke the [tt-profile](../tt-profile/SKILL.md) skill on stitch_hero. Capture:

- kernel ms (median over 10 frames)
- per-stage cycles (only if a previous iter already added zone markers and they're still in)
- which stage looks dominant by inspection of the kernel + the cycle counts

### 2. Pick next idea

Read the candidate pool above and the running optimization log. The supervisor's selection rule:

- Prefer the idea whose **mechanism** targets the current dominant stage.
- Avoid an idea already hard-rejected against this bottleneck (mark it exhausted-against-this-bottleneck in the log; may revisit when the bottleneck shifts).
- Among feasible ideas, prefer the one with the **clearest expected win** the worker can implement in a single iteration.

### 3. Dispatch worker

Launch a `Task` subagent with the following invariants in the prompt:

- `subagent_type: generalPurpose`
- `model: "composer-2.5-fast"` (default; escalation per fallback table below)
- Prompt **must** include:
  - The exact idea to implement (e.g. "add reader-signaled block-wide early termination using a per-core semaphore")
  - Files to touch (typically `kernels/dataflow/reader_alpha_blend.cpp`, `kernels/compute/alpha_blend_compute.cpp`, `alpha_blend.cpp`, and any Python CPU side that has to pack new data)
  - **Permissive gate verbatim** (see "Gating policy" below)
  - "Reference skills: tt-profile, image-diff"
  - "On finish, write `docs/optimization-log/NNN-<short-name>.md` per the template in this skill"

### 4. Worker autonomous run (composer 2.5)

The worker does these steps without supervisor intervention. All work happens on the **bh-30** (Blackhole P150) host; the worker SSHes in for build/test/profile steps.

1. Per-iter branch: `git checkout -b opt/NNN-<name>` from the previous kept commit (locally; devsync mirrors to bh-30).
2. Edit kernel + Python files (local).
3. On bh-30: `cd /proj_sw/user_dev/smarton/gsplat_tt && sudo ninja -C backends/tt/tt-metal/build metal_example_gaussian_splatting` (rebuild host binary if any host-side change).
4. On bh-30: `pkill -f metal_example_gaussian_splatting || true` and `pkill -f 'gsplat scenes' || true` to clear any stale daemon / viewer holding the device.
5. On bh-30: `python scripts/render_fixed.py stitch hero --warmup 3 --frames 10 --out /tmp/it.png` → grab kernel ms.
6. Locally or on bh-30: `python scripts/image_diff.py benchmarks/reference/stitch_hero.png /tmp/it.png --amplified-diff /tmp/it_diff.png` → gate.
7. If iter % 5 == 0 → also run `bench_panel.py --view hero` (all 4 scenes — bicycle only if its .ply has been pushed to bh-30).
8. Write the iter markdown.
9. Commit on the per-iter branch.

### 5. Supervisor decision

Read the worker's iter markdown + the `image_diff.py` exit code (0/1/2):

- **0 — clean keep**: merge the branch into `smarton/optimization` (fast-forward; no merge commit). Update `current best` pointer.
- **2 — kept with `NEEDS_REVIEW`**: merge anyway. The flag is logged in the iter markdown for later human audit.
- **1 — hard reject**: delete the branch. Mark the idea exhausted-against-this-bottleneck. Log why.

Also reject if perf got materially worse (≥ 15% slower on stitch with no compensating gain elsewhere) **and** no metric-improvement justification.

### 6. Re-profile

Run [tt-profile](../tt-profile/SKILL.md) again on the new kept state. If the bottleneck shifted, the candidate pool re-ranks. If not, refine the same idea with a variant on next iter.

### 7. Drift response (cumulative SSIM vs original baseline)

If cumulative SSIM vs `benchmarks/reference/stitch_hero.png` drops below 0.85:

1. **Locate the biggest drift contributor**: re-render the fixed image at each kept iter's commit (`git checkout opt/NNN-... && python scripts/render_fixed.py stitch hero --out /tmp/checkpoint.png`), compute per-iter SSIM delta. Identify the top 1-2 iterations responsible.
2. **Pick a response**:
   - Precision tradeoff (e.g. fp16, approx exp): try a tighter variant of that change. If perf gain mostly holds at higher precision, swap in.
   - Math reorder / algorithmic: leave as-is; accept drift; continue.
   - Clearly wrong on inspection: revert that single iter and re-run the chain from there.
3. **Worst case**: if SSIM stays below 0.85, declare a **new drift baseline** at the current kept iter (copy current PNGs into `benchmarks/reference/`), log `DRIFT_BUDGET_EXHAUSTED` in `SUMMARY.md`, and continue. Allowed at most twice total. Third trigger → stop optimizing, write final report.

## Gating policy (paste this verbatim into worker prompts)

```
Visual regression gate (permissive):

1. Run `python scripts/image_diff.py benchmarks/reference/stitch_hero.png <new>.png --amplified-diff <new>_diff.png`.
2. Read the exit code:
   - 0 (clean keep): proceed.
   - 2 (NEEDS_REVIEW): proceed, but record the `NEEDS_REVIEW` tag and the metric drop in the iter markdown.
   - 1 (hard reject): the change is broken (NaN/Inf, SSIM<0.75, PSNR<20 dB, or mean-abs-diff > 25 LSB). Revert.
3. Additionally: if kernel ms regressed ≥ 15% vs previous kept on stitch with no compensating gain elsewhere, treat as hard reject.

You may NOT tighten the gate. Err strongly toward keeping. The supervisor audits NEEDS_REVIEW flags later.
```

## Worker model fallback ladder (supervisor decides)

| Trigger | Re-dispatch on |
|---|---|
| Composer 2.5 worker can't resolve a build error after one self-correction | GPT-5.3 Codex |
| Same idea regression-gate-rejected twice OR worker shows TT-Metal-subtlety confusion (CB deadlock, race) | Opus 4.7 |
| Worker hangs within 2 min of its 15-min timeout | Opus 4.7 (one retry) |

Cap: 2 re-dispatches per idea. Second failure marks the idea exhausted. Log every dispatch's model + outcome.

## Iteration log template

`docs/optimization-log/NNN-<short-name>.md`:

```markdown
# Iter NNN — <short-name>

- **Idea**: <one-line description>
- **Hypothesis**: <expected impact, target stage>
- **Branch**: `opt/NNN-<short-name>`
- **Worker model**: composer-2.5-fast | gpt-5.3-codex | claude-opus-4-7-thinking-xhigh
- **Decision**: clean-keep | needs-review | hard-reject

## Code diff
See `git log --stat opt/NNN-<short-name> ^smarton/optimization`.

## Bench

| Scene | view | prev kernel ms | this kernel ms | Δ% | PSNR vs prev | PSNR vs baseline | SSIM vs prev | SSIM vs baseline |
|---|---|---|---|---|---|---|---|---|
| stitch | hero | … | … | … | … | … | … | … |
| (iter % 5) luigi/strawberry/bicycle hero …

## Screenshots
- `screenshots/NNN_stitch_hero_after.png`
- `screenshots/NNN_stitch_hero_diff.png` (amplified 10x)

## Notes
<what the supervisor wants to remember for next iter — what shifted in the profile, what to try next, anything anomalous>

## NEEDS_REVIEW (only if decision = needs-review)
<exact metric drops + why we kept anyway>
```

## SUMMARY.md (produced at loop end)

- Cumulative kernel ms / fps curve (one row per kept iter)
- Final ms / fps vs theoretical peak from `binning_log.py`
- Index of all NEEDS_REVIEW iters with one-line summaries (so the user can audit)
- Index of hard-rejected ideas with reasons (for future thesis writeup)
