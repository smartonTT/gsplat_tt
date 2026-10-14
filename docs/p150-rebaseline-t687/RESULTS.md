# p150 re-baseline of best-iter-229 (#687, 2026-10-14)

Measurement box: **bh-49 board 0, Blackhole p150a** (tt-smi: Board Type Blackhole, Series p150a),
IRD job 134115 (cluster tt_bgd) via yyz-ird, extended to 14 h at 02:47 UTC, ends 2026-10-14 13:48 PDT (20:48 UTC).
Code: best-iter-229 (b9171bf9) unchanged, #667 camera files copied next to it. One build (bin 7f442ac3).
Every sequence ran under one `ttp lock p100` (`drive.sh`). Dashboard row: iter 230 (rebaseline, no tag).

| metric | bh-49 p150a | bh-33 p100 (same code) |
|---|---|---|
| orbit b2b ms/view (30 views, 1024x1024; headline) | **7.339** (136.3 FPS), rounds 7.339 / 7.336 / 7.339 / 7.340 | 8.063 |
| train NN b2b ms/view (169 views, 1237x822) | **7.917** (126.3 FPS), rounds 7.916 / 7.918 / 7.918 / 7.916 | 8.565 |
| latency ms/view (no --dump-views; secondary) | 7.307 (runs 7.304 / 7.307 / 7.307) | 7.99 |
| raw md5 orbit / NN | c194c715 / 21ed4759 | 1d372b94 / ea105357 |
| hero PSNR vs benchmarks/reference_v2/hero.png | 42.85 dB | 42.85 dB |
| golden (p100 hero_golden_8bit) | no match, max 13 LSB | match |

Hero (`opt/metal-screenshots/ttw-230/`) checked by eye: no tile seams or blocky tiles; the x10 diff
shows only thin edges, same as on the p100. JIT kernel hashes for later A/B checks on this box:
`jit-hashes-p150-229.txt` (listing in `evidence/jit-eth12.txt`; on the p150a the eth-dispatch overlay
keeps the JIT cache in `<tree>/tmp/ttm-eth12-cache`).

Target this box: `opt/ird/renew.sh measure --keep-old` (extends bh-49; `--keep-old` while bh-33 is
still held by #653), then under one lock
`ttp lock p100 -- opt/sync_remote.sh bh-49 /localdev/$USER/gstt2 <rev>` + the run with `MESH_DEVICE=P150`
(bh-49 has no rsync: sync_remote falls back to ssh+tar, one-way, changed files by md5).
Driver used here: `env H=bh-49 docs/p150-rebaseline-t687/drive.sh <rev>` (`SKIP_SHOT=1`, `NN_ONLY=1`).
