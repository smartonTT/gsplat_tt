"""Stage timings of the live viewer path vs the bench path, on the viewer box.

Run with the viewer stopped (it holds the device). Builds the viewer exactly as
viewer_clean.py does (on a private port), then times at the bicycle hero view:

  render_bench   pipeline.render at the bench's contrib floor (1/255), hero and
                 the 30 bench views (the bench's ms/view metric)
  render_slider  pipeline.render at the old slider floor (1/16384)
  render_fn      the viewer's render callback (render + letterbox + stats)
  render_fn_tr   the same under nerfview's per-line sys.settrace hook (old loop)
  jpeg_q40/q70   viser's JPEG encode of the frame
  loop_old       old nerfview loop per frame: lock + settrace + render_fn, then encode
  loop_new       FastRenderer: render_fn, encode on the sender thread (overlapped)

and prints md5s of the hero frame from pipeline.render and from render_fn.
  .venv/bin/python opt/viewer/viewer_probe.py [--n 30] [--bench-hero tmp/x/hero_clean.png]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import threading
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "opt" / "viewer"))
os.environ.setdefault(
    "TT_METAL_CACHE", f"/localdev/{os.environ.get('USER', 'smarton')}/.cache/tt-metal-cache-viewer")


def md5(img: np.ndarray) -> str:
    return hashlib.md5(np.ascontiguousarray(img).tobytes()).hexdigest()[:8]


def timed(fn, n: int, warm: int = 3) -> list[float]:
    for _ in range(warm):
        fn()
    out = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t) * 1000.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--port", type=int, default=8097)
    ap.add_argument("--bench-hero", default=None, help="bench hero_clean.png to md5-compare")
    args = ap.parse_args()

    from viewer_clean import _load_run_py
    run = _load_run_py()
    import backends
    backends.REGISTRY["tt_clean"] = run.CleanBackend
    import torch
    import nerfview
    from nerfview._renderer import RenderTask, set_trace_context
    from viser._scene_api import _encode_image_binary
    from gsplat.loading_gaussians import load_ply
    from gsplat.nerfview_viewer import FastRenderer
    from gsplat.utils import c2w_to_w2c
    from gsplat.viewer import GaussianViewer

    cam = json.loads((REPO / "benchmarks" / "cameras_v2.json").read_text())["bicycle"]
    W, H = cam["image_size"]
    floor = float(cam["contrib_floor"])
    gauss = load_ply(str(REPO / cam["ply"]))
    viewer = GaussianViewer(gauss, host="127.0.0.1", port=args.port, backend="tt_clean",
                            render_width=W, render_height=H, scene_path=str(REPO / cam["ply"]),
                            contrib_floor=floor)
    pipe = viewer.pipeline
    K = run.build_intrinsics(W, H, float(cam["fov_deg"]))
    hero_c2w = np.asarray(cam["views"][cam["order"][0]]["c2w"], dtype=np.float32)
    extr = c2w_to_w2c(torch.from_numpy(hero_c2w))
    rows: list[tuple[str, list[float]]] = []

    def set_floor(f: float):
        pipe.contrib_floor = f
        pipe._sync_render_settings_to_backend()

    set_floor(floor)
    print(f"[probe] pipeline floor={pipe.contrib_floor:.6g} tr={pipe.transmittance_threshold:.6g} "
          f"min_op={pipe.min_opacity:.6g} max_r={pipe.max_radius} k_cap={pipe.k_cap}", flush=True)
    rows.append(("render_bench hero", timed(lambda: pipe.render(gauss, extr, K, H, W), args.n)))
    hero_img = pipe.render(gauss, extr, K, H, W).image
    views = []
    for name in cam["order"]:
        e = c2w_to_w2c(torch.from_numpy(np.asarray(cam["views"][name]["c2w"], dtype=np.float32)))
        t = time.perf_counter()
        pipe.render(gauss, e, K, H, W)
        views.append((time.perf_counter() - t) * 1000.0)
    rows.append(("render_bench 30 views", views))
    set_floor(1.0 / 16384.0)
    rows.append(("render_slider hero (1/16384)", timed(lambda: pipe.render(gauss, extr, K, H, W), args.n)))
    set_floor(floor)

    # Viewer callback at the hero camera (nerfview CameraState, OpenCV c2w).
    cs = nerfview.CameraState(fov=np.deg2rad(float(cam["fov_deg"])), aspect=W / H,
                              c2w=hero_c2w.astype(np.float64))
    rts = viewer.viewer.render_tab_state
    fn_img = viewer._render_fn(cs, rts)
    rows.append(("render_fn", timed(lambda: viewer._render_fn(cs, rts), args.n)))

    def tracer(frame, event, arg):  # same cost shape as nerfview's _may_interrupt_trace
        return tracer

    def fn_traced():
        with viewer.viewer.lock, set_trace_context(tracer):
            return viewer._render_fn(cs, rts)
    rows.append(("render_fn under settrace", timed(fn_traced, args.n)))
    rows.append(("jpeg_q40", timed(lambda: _encode_image_binary(fn_img, "jpeg", jpeg_quality=40), args.n)))
    rows.append(("jpeg_q70", timed(lambda: _encode_image_binary(fn_img, "jpeg", jpeg_quality=70), args.n)))

    def loop_old():
        img = fn_traced()
        _encode_image_binary(img, "jpeg", jpeg_quality=40)
    rows.append(("loop_old frame (serial)", timed(loop_old, args.n)))

    # FastRenderer with a client stub whose send is viser's real JPEG encode.
    class _Scene:
        def set_background_image(self, img, format, jpeg_quality, depth=None):
            _encode_image_binary(img, format, jpeg_quality=jpeg_quality)

    class _Client:
        scene = _Scene()
        client_id = -1

    sent: list[tuple[float, float, float]] = []
    done = threading.Event()

    def on_sent(t0, t1, t2):
        sent.append((t0, t1, t2))
        if len(sent) >= args.n + 3:
            done.set()
    fr = FastRenderer(viewer=viewer.viewer, client=_Client(), lock=viewer.viewer.lock,
                      on_frame_sent=on_sent)
    t_start = time.perf_counter()
    for _ in range(args.n + 3):
        fr.render_once(RenderTask("move", cs))
    done.wait(5.0)
    t_end = time.perf_counter()
    fr.running = False
    e2e = [(c - a) * 1000.0 for a, _, c in sent[3:]]
    rows.append(("loop_new end-to-end (render->sent)", e2e))
    print(f"[probe] loop_new: {len(sent)} sent of {args.n + 3}, "
          f"{(t_end - t_start) * 1000.0 / max(len(sent), 1):.2f} ms/frame throughput", flush=True)

    print("PROBE stage | median ms | min | max | n")
    for name, v in rows:
        print(f"PROBE {name} | {statistics.median(v):.2f} | {min(v):.2f} | {max(v):.2f} | {len(v)}")
    print(f"PROBE md5 hero pipeline.render={md5(hero_img)} render_fn={md5(fn_img)} "
          f"shape={fn_img.shape} dtype={fn_img.dtype}", flush=True)
    if args.bench_hero:
        from PIL import Image
        b = np.asarray(Image.open(args.bench_hero).convert("RGB"))
        print(f"PROBE md5 bench hero_clean.png={md5(b)} "
              f"identical={bool(np.array_equal(b, fn_img))}", flush=True)
    Image_out = REPO.parent / "hero_viewer_t257.png"
    from PIL import Image
    Image.fromarray(fn_img).save(Image_out)
    print(f"PROBE saved {Image_out}", flush=True)
    pipe.close()
    os._exit(0)


if __name__ == "__main__":
    main()
