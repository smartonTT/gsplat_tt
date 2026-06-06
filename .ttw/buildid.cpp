id=80
sha=04c8992
ts=2026-06-06T12:05:24-0700
desc=iter 111: A1: hoist conic (cov->A,B,C) into pfwc; delete per-microblock det/recip/A/B/C recompute in blend; rewrite cull in conic space; invert conic->cov at the cpu_cpp_mb device->host readback boundary so the reference's CPU cull/blend stay raw-cov. Attacks SFPU wall (tile_blend_sfpu ~291k ms).
bin=665234a6034eca2b
