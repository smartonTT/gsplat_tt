# t649: L-A (#639) + L-A' (#643) combined, defaults on — RESULT

Tree 2e817955 (0c20a479 + merge of t643 5ca20987), bh-33 p100 (measurement box),
render/run.py --back-to-back defaults (1 check, 1 warm-up, 20 measured passes), 30 bicycle views.
Arms by env on one build: base (LA=0, SIF=0), la (LA=1, SIF=0), sif (LA=0, SIF=1), both (defaults).

| arm  | run medians (ms/view)            | median | FPS   |
|------|----------------------------------|--------|-------|
| base | 8.452, 8.447, 8.455, 8.471       | 8.454  | 118.3 |
| la   | 8.236, 8.228, 8.230              | 8.230  | 121.5 |
| sif  | 8.325, 8.262, 8.263              | 8.263  | 121.0 |
| both | 8.232, 8.233, 8.230, 8.233       | 8.233  | 121.5 |

- both vs base: -0.221 ms/view (-2.6%). both = la within noise: L-A already enqueues the next
  view's sort, so L-A' only acts when L-A does not (23 LA lines, 0 SIF lines in the both arms).
  Landing both defaults on: same speed as L-A alone, L-A' covers the views L-A skips.
- raw_md5 1d372b94 in every arm, identical across passes; sweep md5 077caaba 30/30 (default config).
- Latency (no --dump-views, secondary): base 8.527, both 8.182 ms/view (one run each).
- Device hero (opt/metal-screenshots/ttw-228): PSNR 42.85 dB vs benchmarks/reference_v2/hero.png,
  golden match; hero and diff x10 inspected, no tile seams or blocky tiles.
