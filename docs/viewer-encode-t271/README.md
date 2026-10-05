# Viewer JPEG encode (task #271)

Box: bh-35 p100a (viewer reservation), tt-metal 437bc366, tree bc4a879 (opt head + t257 + t271).
Frame 1024x1024 bicycle hero, q40 (the "move" quality). `opt/viewer/viewer_probe.py --n 30
--jpeg viser,cv2,simplejpeg` (device) and `opt/viewer/encode_bench.py` (CPU only).

| sender encoder | encode q40 ms | render in live loop ms | render -> sent ms | frame ms (throughput) |
|---|---|---|---|---|
| old nerfview loop (deployed before: settrace, serial viser encode) | 11.69 | 13.12 | 23.81 | 23.81 |
| FastRenderer + viser encode (t257) | 11.57 | 13.06 | 23.46 | 13.40 |
| FastRenderer + cv2, cvtColor swap | 3.13 | 13.09 | 16.45 | 13.19 |
| **FastRenderer + simplejpeg (new default)** | **3.07** | **13.12** | **16.23** | **13.26** |
| control: 5 ms sleep instead of encode | - | 12.95 | 18.15 | 13.13 |

render_fn alone is 12.89 ms (pipeline.render hero 12.66, 30-view mean 12.13), so the render in the
loop is +0.23 ms with simplejpeg (goal <= 0.3). Render-to-sent latency drops 23.5 -> 16.2 ms.
Frame time is set by the render, not the encode, so it barely moves (13.40 -> 13.26 ms, 75 FPS).

Why viser's encode is slow: it swaps RGB->BGR with a numpy gather (`image[:, :, [2, 1, 0]]`),
~7 ms of the 10.3 ms. cv2's own `cvtColor` swap plus `imencode` is 3.1 ms. simplejpeg encodes RGB
directly (bundled libjpeg-turbo, 4:2:0, fastdct) and releases the GIL: a busy Python thread keeps 99%
of its speed during the encode (cv2 92-95%, viser 97%). JPEG size is the same for all (105 KiB q40).

PyTurboJPEG 2.5.0 is not usable on bh-35: it needs the system libturbojpeg.so, which the box does not
have (only libjpeg.so.8), and installing it needs root. simplejpeg wraps the same library, so nothing
is lost. The encoder is picked by `GSPLAT_VIEWER_JPEG` (auto | simplejpeg | turbojpeg | cv2 | viser);
auto takes the first one that imports. setup_box.sh installs simplejpeg 1.9.0.

No-interrupt patch: `viser_patches._patch_nerfview_no_interrupt` set `_may_interrupt_render = False`
and then called nerfview's submit, which set it back to True, so it never did anything.
FastRenderer renders without nerfview's settrace hook, so the interrupt cannot fire. The patch is
deleted. `tests/unit/test_viewer_fast_renderer.py` checks that a move submitted mid-render does not
abort it, that each encoder round-trips with the right channel order, and that submit is unwrapped.

Live viewer: redeployed on bh-35 port 8080 at bc4a879 (HTTP 200, encoder simplejpeg). SELFTEST hero
12.67 ms. Device screenshot `hero.png`, md5 c93c22df (= bench hero_clean, as in t257), PSNR 41.16 dB
vs `benchmarks/reference_v2/hero.png`; `hero_diff10.png` is |diff| x10. Visual check: no tile seams,
blocky or empty tiles; the diff shows only edge detail (spokes, foliage) and the known faint halo upper
left. This build is also the first device run of the t259 kcfg auto-size without an override: it built
and rendered the same md5.
