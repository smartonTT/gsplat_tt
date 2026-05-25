# iter-000 — BASELINE

**Date:** 2026-05-25
**Branch:** smarton/opt-v2
**Commit:** (this commit)
**Class:** baseline (not part of optimization queue)
**Hypothesis:** N/A — capture frozen reference for the rest of the run.

## Setup

- Hardware: yyzo-bh-14, MESH_DEVICE=P100 (single chip P300)
- Scene: stitch_doll.ply
- Resolution: 1024×1024
- Kernel: unmodified main-branch alpha_blend_compute.cpp
- Measurement: 1 warmup cycle + 10 measured cycles of [hero, side, top]

## Results

- kernel ms median: 99.95
- kernel ms p99: 105.40
- per-view: hero 99.95 / side 90.57 / top 105.28

## What this commit produces

- `benchmarks/reference/stitch_{hero,side,top}.png` — the frozen reference for all subsequent iters (1024×1024 implicit)
- `docs/optimization-log/screenshots/iter-000-baseline/` — captured renders + timing

## Frozen reference rule

`benchmarks/reference/*.png` is write-protected by a PreToolUse hook in
`.claude/settings.local.json`. No agent (supervisor, worker, validator) modifies
these files. The user explicitly promotes a new reference only if they choose to,
outside the autonomous loop.
