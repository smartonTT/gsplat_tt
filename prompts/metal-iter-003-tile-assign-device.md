# Metal iter-003 — port `tile_assign` + per-pair cull to device

Gate: tile_assign fixture exact match; 30-view PSNR unchanged.

Driver: `REMOTE_HOST=bh-30 scripts/run_iter_metal.sh 3 tile-assign-device port`
