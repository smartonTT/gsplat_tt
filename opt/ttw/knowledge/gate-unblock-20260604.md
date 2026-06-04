# Gate unblock (2026-06-04)

## Done
- Stopped old loopd; armed fresh `loopd.sh run` (20m heartbeat).
- `render/run.py`: prod/render JIT cache split; anchor `benchmarks/anchor/hero_cpu_cpp_mb.npy` (from `ttw-047/ref.png`, mean~0.33); fail-fast ref validation; a003 subprocess (segfaults on bh-07 — falls back to anchor).
- Sync includes `render/`, `benchmarks/anchor/`, `ttw.toml` (`cd /localdev/smarton/gstt2`).
- Build: cpp#215 bin=b398993350d0061c.

## Blocker (2026-06-04 worker, bh-07)
- **Render PSNR**: `render_clean` vs anchor ref ~46–57 dB (cpp#217 bin=b398993). Anchor valid (mean~0.33, std~0.22) but not the in-process CPU-fallback oracle used at 63.85.
- **Live ref**: `a003_verify` OK standalone (~5s); **SIGSEGV** when spawned from `run.py` (gate path). Ref-after-render subprocess **PCIe lock** hang. Ref-only child → **~3 dB** (wrong oracle). In-process ref after render → **SIGSEGV** (JIT/linker) on bh-07.
- **Historical 63.85**: cpp#157 / 9f1d618, single `tt-metal-cache`, in-process `cpu_cpp_mb` after 30 views (device held → CPU fallback).

## Next (worker)
1. Force-rebuild `render_clean.so` and bisect `render/` commits after 9f1d618 (blend/L1/subchunk) until hero_vs_ref ≥63.6 with **in-process** ref + `silence_os_fd_output` (63.85 path).
2. Fix `a003` subprocess SIGSEGV separately (env/parent pollution?) for live ref; do not use anchor as primary.
3. Do **not** start unified L1 step C until gate green.
