# Final report — host-free / L1-resident renderer optimization

**Status: FROZEN at the 173.1 ms/view plateau, bit-identical (`hero_vs_ref = 100.00`).**
Branch `smarton/stage2-hostfree-l1`. Date 2026-06-11. Ledger: `opt/ttw/iters.jsonl`; live
report: `opt/REPORT.html`; roadmap + refuted-premises ledger: `opt/sort-l1-resident-plan.md`.

## Outcome

The renderer reached a measured, evidence-confirmed performance plateau. The descent and the
proof that it is a true architectural floor (not a lack of effort) are below.

- **Frame:** recovered baseline ~195.6 ms/view (iter-127) → **173.3 ms/view** (iter-141), all
  **bit-identical** to the committed 8-bit golden `tests/fixtures/hero/hero_golden_8bit.png`.
  (Exploration briefly peaked at ~236 ms during the Metal-Trace detour, iter-125, which was a
  self-inflicted regression later recovered — see refutation #1, so the honest "win" is ~195.6 → 173.3.)
- **Quality gate rebaselined** (iter-132) to 8-bit PSNR vs a committed golden (bit-identical = 100.00 dB).
- **Critical path:** BRISC-FW ~159 ms / NCRISC-FW ~125 ms (both saturated and flat); TRISC/SFPU
  ~52 ms is OFF the critical path; cull/blend DRAM readers are ~98% SFPU-backpressured.

## What landed (all bit-identical unless noted)

| Lever | iter | effect |
|---|---|---|
| `sort_bucket_emit` invariant-pack hoist | 128 | −25% (40.9→30.6 ms); refuted the scatter-write premise |
| `sort_subchunk_materialize` LPT load-balance | 130 | overflow-gather + imbalance fix (26.2→19.6 ms) |
| Stage-2b op/color UNORM16 pre-pack at birth + off-pole relocation | 131/132/138 | de-dup 4×fp32→UNORM16/gaussian |
| NCRISC soft-float strength reduction `x/2^k → x*(1/2^k)` | 134/135 | bit-identical, ~2.2 ms |
| `ta_gauss_aabb` multi-buffered reads | 137 | hide NoC read latency |
| Blend microblock ILP (Track-1b, C=2) | 118 | −13% blend SFPU (off-path) |
| On-device sort bin-layout | 121/122 | kept, gated-off (trace prereq) |

## What was refuted, with on-device evidence (the value of the loop)

1. **Metal Trace / host-free dispatch** (121–127): removes <0.5 ms — the ~92 ms is on-device
   firmware + NCRISC dataflow, not host dispatch. `host_free_mp` cost ~31 ms/frame (not neutral).
2. **Program fusion** (133): reclaimable launch firmware ≈ 0.11 ms; the 73 ms "BRISC non-kernel" is
   BRISC idle-**waiting** in barriers on NCRISC, not removable overhead.
3. **~32 ms inter-frame host gap** (136): a Tracy observer artifact; real gap ~4.86 ms.
4. **Record-layout dedup** (137): cost-shuffle trap (net +5..+25 ms).
5. **Stage-3 L1-resident handoff as a frame lever** (137): the DRAM round-trip it deletes is ~0.57 ms
   and ~98% SFPU-masked.
6. **Blend transmittance early-out** (139): already shipped (iter-107, lossy, baked into golden);
   measured residual overshoot only 7.82% → perfect-early-out ceiling ~2.3 ms (unphysical); realizable
   version lossy + hang-prone (iter-52 hung the device). Also fixed the stale "fan-out K≈1" premise
   (iter-117 measured median K=4).
7. **Persistent-kernel / cross-stage-pipeline rewrite** (140–141):
   - The overlap **mechanism is real** (140): NCRISC↔SFPU run concurrently on the same cores under one
     in-order CQ, fused makespan = max not sum, zero hangs (the old iter-55-59 hangs were the
     2nd-CQ / intermediate-Finish class, now avoided).
   - But the **cheap version banks ~0** (141): cull is already a single overlapped program; the in-budget
     emit is ~1.7 ms (NCRISC ≪ SFPU, nothing to hide); the deleted DRAM read is already backpressured.
   - And the **heavy version is structurally impossible** (141 feasibility): the ~32 ms `sort_bucket_emit`
     prize is a GLOBAL gaussian→tile transpose gated by a prefix-sum barrier → **0 ms pipelinable**
     per-tile against cull/blend. The only per-tile heavy stage (materialize ~12 ms) is already shadowed.
     L1 (exactly 1.5 MB) fits the fused double-buffered kernel only on a ~4% knife-edge in-budget and
     **blows the budget on the heavy/overflow tiles** the lever targets. Net delta after the ~45-54 ms
     prerequisite regression hole: **~+33-42 ms WORSE**.

## Why 173.1 ms is the architectural floor

The poles (BRISC-FW/NCRISC-FW) are saturated; SFPU is off-path; every incremental kernel lever is at a
structural floor, refuted, or backpressure-masked. The one structural lever that could move the poles —
overlapping the heavy NCRISC stages under the SFPU shadow via a cross-stage per-tile pipeline — is
blocked by the global emit prefix-sum barrier (the gaussian→tile transpose is not separable per-tile)
and by L1 capacity. Going below 173 ms requires a **fundamentally different algorithm** (e.g. a
tile-major sort that avoids the global transpose, or an SFPU-saturation-bound blend rewrite), not a
re-scheduling of the current one.

## Open / not-pursued (deliberately deferred levers)

- Tile-major sort to make emit per-tile-pipelinable — refuted as a cost-shuffle at the current record
  layout (iter-137); would need a different on-device data structure to avoid the +5-25 ms gather.
- Aggressive (lossy, refreeze) blend early-out — bounded ~3-7 ms, hang-prone; not worth it vs the gate.
- Adaptive radix bucket count in `sort_tile_depth` — ~2-3 ms, algorithm change, risk > reward.

These remain documented in `opt/sort-l1-resident-plan.md` for future re-litigation if hardware,
thermal, or the blend algorithm changes (per the standing "re-test refuted premises periodically" rule).
