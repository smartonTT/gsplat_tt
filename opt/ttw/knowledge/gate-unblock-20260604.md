# Gate unblock (2026-06-04)

## Gate — GREEN (iter 62)
- `hero_vs_ref=63.63` dB, `ms_view≈296`, cpp#219 `8833462`, bin=`793507cbb5f51704`.
- **Fix:** `render/run.py` spawns `cpu_cpp_mb` ref in a **subprocess before** `render_clean` (exclusive device); `TT_METAL_CACHE` split prod vs render; rebuild stale `_gsplat_cpu.so` on bh-07. Fail-fast if ref mean ∉ [0.05, 0.95].
- **Do not** run in-process ref **after** render_clean on bh-07 (dual MetalContext / bad ref).

## Prior investigation (resolved)
- False ~46 dB was harness/oracle ordering, not render/kernel regression at `c946556` (iter 60 still ~63.84 dB on device).
- Anchor npy valid but wrong path for gate; in-process CPU-fallback oracle matches historical 63.85.

## Next (L1 plan)
1. Iter 63: verify steps A+B (materialize + device subchunk dir) with gate ≥63.6.
2. Step C (unified payload blend reader) only after A/B sign-off; 3-try rule per tt-loop.
