# Supervisor Status

## ESCALATIONS (read these first)

_None._

## Current State

- Last updated: 2026-05-26 03:30 UTC
- Last iter: iter-005-verify-revert-baseline → KEEP (no commit; baseline reproduced after revert)
- Current best kernel ms (committed): 99.95 ms (iter-000-baseline, after iter-003 revert)
- Last iter kernel ms median: 100.34
- Target: 1.0 ms

## Recent correction

- **iter-003-m1-basis-form-fpu-q** was committed (3d5b162) with KEEP verdict 2026-05-26 02:18 UTC but produced a visible tile-grid seam artifact at the bottom of Stitch's ears in hero.png. The validator subagent had flagged REJECT on diff10 cross-hatch; the supervisor override to KEEP was wrong. Reverted in 044f398.
- **iter-004-fuse-inner-acquires** built on M1 inherits the same artifact and never reached a verdict. Reverted in the same M1 rollback.
- Root cause: basis-form Q(x,y) = A·x² + B·xy + C·y² + D·x + E·y + F has catastrophic cancellation (6 fp32 terms of magnitude ~10 summed to a small Q ~0.01). Cancellation loses ~17 binary bits of precision; the residual is then quantized by bf16 packing of R/G/B/T state across Gaussians, and adjacent tiles accumulate divergent error patterns at tile boundaries → cross-hatch.
- Implications for QUEUE: M1, M2, M3, M4 (iters 3-6 in QUEUE.json) all build on basis-form and need rethinking. The FPU-heavy roadmap as written assumed basis-form was numerically safe; it isn't on this kernel at this Gaussian-count.
