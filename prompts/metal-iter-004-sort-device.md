# Metal iter-004 — port `sort` to device

Per-tile radix or NoC scan. Gate: sort depth-sequence match vs CPU; PSNR unchanged.

Driver: `REMOTE_HOST=bh-30 scripts/run_iter_metal.sh 4 sort-device port`
