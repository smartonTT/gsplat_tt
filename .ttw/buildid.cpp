id=152
sha=bb52a1d
ts=2026-06-02T19:04:51-0700
desc=iter 46: Part A: M2 §6 per-microblock transmittance early-out — replace iter-44's tile-level break with a true per-microblock done-mask (spilled-T scan groups px by SFPU vector V=(r&~1)|(c&1); clear keep-bit M when all 32 px of mb M have T<eps; subsequent gaussians AND ~done into their coverage mask). PROVEN CORRECT: 63.85-63.86 dB across EO_BLK 64..768. Gated-off (GSPLAT_TT_BLEND_EARLYOUT): perf-neutral because the resident blend is READER-GATHER-bound, not SFPU-compute-bound (smaller blocks monotonically worse: 768->184.6 256->186.8 128->188.7 64->193.5ms) — the compute early-out skips SFPU math but the reader still gathers every candidate. Next lever: reader-side tile-saturation tail-skip.
bin=12bb207ef9ce5a5e
