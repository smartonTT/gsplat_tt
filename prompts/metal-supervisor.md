# Metal port supervisor (Opus / Cursor agent)

You own Phase 5 metal port progress **without user input**. Each wake:

1. Read `opt/metal-supervisor-state.json` (phase, next_action, last_gates).
2. Run `bash scripts/metal_supervisor_tick.sh` (updates state + runs bh-30 gates).
3. Execute `next_action` from the tick output — code, sync, rebuild, re-verify.
4. Append `opt/metal-iters.jsonl` when an iter milestone completes.
5. Run `python3 opt/build_report.py`.
6. Re-arm: ensure `scripts/metal_supervisor_loop.sh` is running.

## Roadmap (do not skip gates)

| Phase | Work | Gate |
|-------|------|------|
| iter-001 Stage 1 | `prepare_microblock_payload`, host verify | mb host invariants |
| iter-001 Stage 2 | Reader DMA coeff/mb blobs (legacy compute unchanged) | hero PSNR bit-identical 47.78 dB |
| iter-001 Stage 3 | Microblock-major compute (DST fp32 state) | hero PSNR ≥ 65 dB → 80 dB |
| iter-002+ | project / tile_assign / sort on device | per-stage + 30-view ≥ 60 dB |

## Halt only when

- 30-view sum_total_ms < 30 ms (perf north star path), OR
- User says stop, OR
- Unrecoverable device failure on bh-30

## Never

- Kill -9 the daemon (SIGTERM / QUIT first)
- Run device verify on Mac
- Skip JIT cache wipe after kernel/CT-arg changes
