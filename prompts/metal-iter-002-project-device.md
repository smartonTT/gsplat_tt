# Metal iter-002 — port `project` to device

Port cov3d + cov_cam + cov2d on SFPU (single core first). CPU golden: `tests/fixtures/hero/project_*.npz`.

Gate: project fixture max_abs ≤ 1e-3; 30-view PSNR unchanged vs numpy reference.

Driver: `REMOTE_HOST=bh-30 scripts/run_iter_metal.sh 2 project-device port`
