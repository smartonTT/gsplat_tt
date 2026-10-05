# Viewer vs bench gap (task #257)

Box: bh-35 p100a (viewer reservation), tt-metal 437bc366. Bench = `render/run.py` sweep, 30 views.
Hero = first view of `benchmarks/cameras_v2.json` (it costs ~0.5 ms more than the 30-view mean).

| stage | before ms | after ms |
|---|---|---|
| bench best-iter-199, bh-35 (30-view mean) | 12.23 | 12.19 (t257 tree, md5 46a725ab both) |
| bench best-iter-199, yyzo-bh-07 (box/pin delta) | 11.63 | 11.63 (+0.6 ms on bh-35, not viewer code) |
| pipeline.render, hero, bench floor 1/255 | 12.69 | 12.69 |
| viewer render at slider floor 1/16384 (old default) | 15.57 | n/a (viewer now uses 1/255) |
| render_fn callback overhead | +0.24 | +0.24 |
| nerfview settrace hook | +0.21 | 0 (FastRenderer) |
| JPEG encode (PIL q40 / cv2 q40) | 4.97 serial | 10.3 on sender thread, overlapped |
| render slowed by concurrent encode (GIL) | n/a | PIL +1.65 / cv2 +0.20 |
| viewer frame time (throughput) | 18.5 (serial render+encode), ~60 FPS cap from 16 ms tick | 13.62 (73 FPS) |
| render start -> frame sent (latency) | 18.5 | 23.6 (cv2 encode is slower but off the GIL) |
| viewer selftest hero (on-screen "device render") | 15.48 | 12.69 |
| load_ply + first frame on bh-35 | minutes (THP compaction) | seconds |

Root causes: (1) the viewer used the slider contrib floor 1/16384, the bench uses 1/255 (+2.9 ms);
(2) nerfview's settrace hook and serial JPEG encode, and the 16 ms burst tick (~60 FPS cap);
(3) PIL's chunked JPEG encode on the sender thread ping-pongs the GIL with the render (+1.65 ms);
cv2's encode is one GIL-free call; (4) numpy madvises large arrays for huge pages and bh-35's
memory is fragmented (compact_fail 2.8M), so page faults ran direct compaction: a 144 MB alloc
took >100 s (0.3 s with `NUMPY_MADVISE_HUGEPAGE=0`). That made the viewer and benches take
minutes to start. Remaining gap: render in the live loop 13.14 vs bench hero 12.69 (+0.45 ms:
callback +0.24, encode overlap +0.20). bh-35 itself is +0.6 ms vs yyzo-bh-07.

Correctness: viewer-path hero (render_fn) md5 c93c22df == bench hero_clean.png; equals the golden
`hero_golden_8bit` (badge). Device screenshot `hero.png` (t257 tree on bh-35), `hero_diff10.png`
vs `benchmarks/reference_v2/hero.png`: PSNR 41.16 dB. Visual check: no tile seams, blocky or empty
tiles; diff shows only edge detail (spokes, foliage) and a faint low-frequency halo upper left,
the known contrib-floor divergence (#251). render.cpp change (GIL release) is bit-exact.

Tools: `opt/viewer/viewer_probe.py` (stage timer, `--switch-ms`, sleep-send control, per-run
pipeline totals).
