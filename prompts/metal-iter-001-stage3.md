# metal-iter-001 Stage 3 — microblock-major compute

Prerequisite: Stage 2 done (hero 47.783 dB with GSPLAT_METAL_MB=1).

## Sub-steps (land one at a time; gate each on bh-30 hero PSNR)

| Step | Work | Gate |
|------|------|------|
| 3a | Reader: push mb_header (4 pages/tile) to shadow CB | Blocked — header path hung bh-30; stream L' up to 51k/tile makes shadow CB infeasible |
| 3b | Compute reads mb_header/mb_stream from DRAM; microblock outer loop | Next after bh-30 SSH restored |
| 3c | Per-microblock fp32 R/G/B/T in Dst (MB_TO_DST_ADDR); drop full-tile state CBs for active path | PSNR ≥ 65 dB |
| 3d | Basis-form coeff (A..F) + SFPU replay inner loop per design §6 | PSNR ≥ 80 dB |

## Files

- `kernels/dataflow/reader_alpha_blend.cpp`
- `kernels/compute/alpha_blend_compute.cpp`
- `alpha_blend_host.h`, `alpha_blend.cpp`

## Verify

```bash
ssh bh-30 "cd /proj_sw/user_dev/smarton/gstt2 && rm -rf /proj_sw/user_dev/smarton/.cache/tt-metal-cache/* && source .venv/bin/activate && GSPLAT_METAL_MB=1 python3 scripts/verify_blend_metal.py"
```
