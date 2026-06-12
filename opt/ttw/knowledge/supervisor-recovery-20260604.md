# Supervisor recovery (2026-06-04)

## Report
- `opt/build_report.py` + `opt/validate_report.py` were deleted in iter-73 auto-commit `5129b5e`; restored from parent.
- Recovered `opt/metal-iters.jsonl` (74 rows) + `opt/iters.jsonl` (58 rows); merged `opt/ttw/iters.jsonl` to 81 rows from git `4e08337` + hand rows (87 reject).
- `REPORT.html` regen: **156 ledger rows** (was 22 when only ttw slice existed).
- Validate: hero check only for kept rows with `screenshot` paths on disk.

## Repo / device
- `backends/` restored from `origin/main` (missing on flattened `stage2-hostfree-l1`).
- Production default: `use_payload=false` in blend reader (iter-85 mat PACK2 fix kept in materialize).
- **Cumulative perf keep:** iter 74+76 (~290.75 ms/view); iter 85 payload path correct but slower.
- Device iter 86–87: broken rsync / no `.git` → gate ~47 dB until full clone.

## Next
1. Full `git clone` on bh-07 at `smarton/stage2-hostfree-l1` after this commit pushed.
2. Gate verify → must see 63.63 dB, ~290 ms/view.
3. Resume unified L1: C1b payload optional; C2 in-budget reader only with 3-try rule.
