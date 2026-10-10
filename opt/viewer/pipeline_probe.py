"""Viewer frame pipelining (task #457): the 30 bench views through FastRenderer,
non-pipelined vs pipelined (next-pose xview hint), on the viewer box.

Run with the live viewer stopped (it holds the device). Builds the viewer as
viewer_clean.py does on a private port, then for each mode renders the bicycle
bench views in bench order through FastRenderer.render_once("move") with a client
whose camera walks the bench poses, so the pipelined mode latches and hints
exactly the next bench view. Prints per mode: ms/view (render loop wall / views),
xview hits/misses (delta of the C++ counters) and the list md5 computed like
render/run.py's raw_md5 (39d84b28 on a p150 with eth 12x10 dispatch). Saves the
pipelined hero frame as hero.png, a x4 diff and the PSNR vs the reference render.
  venv/bin/python opt/viewer/pipeline_probe.py --out DIR [--passes 4] [--port 8097]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")  # see viewer_probe.py

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "opt" / "viewer"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--passes", type=int, default=4, help="passes per mode; pass 0 is the check pass")
    ap.add_argument("--port", type=int, default=8097)
    ap.add_argument("--reference", default=str(REPO / "benchmarks" / "reference_v2" / "hero.png"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # Same dispatch as the live viewer (viewer_clean.py): eth 12x10 on a p150.
    sys.path.insert(0, str(REPO / "render"))
    from eth_default import setup as eth_setup
    eth_setup(os.environ, REPO, log=lambda m: print(m, flush=True))
    from viewer_clean import _load_run_py
    run = _load_run_py()
    import backends
    backends.REGISTRY["tt_clean"] = run.CleanBackend
    import nerfview
    from nerfview._renderer import RenderTask
    from PIL import Image
    from gsplat.loading_gaussians import load_ply
    from gsplat.nerfview_viewer import FastRenderer
    from gsplat.viewer import GaussianViewer

    cam = json.loads((REPO / "benchmarks" / "cameras_v2.json").read_text())["bicycle"]
    W, H = cam["image_size"]
    gauss = load_ply(str(REPO / cam["ply"]))
    gv = GaussianViewer(gauss, host="127.0.0.1", port=args.port, backend="tt_clean",
                        render_width=W, render_height=H, scene_path=str(REPO / cam["ply"]),
                        contrib_floor=float(cam["contrib_floor"]))
    pipe = gv.pipeline
    pipe.contrib_floor = float(cam["contrib_floor"])
    pipe._sync_render_settings_to_backend()
    clean = pipe.backend._clean
    order = cam["order"]
    states = [nerfview.CameraState(fov=np.deg2rad(float(cam["fov_deg"])), aspect=W / H,
                                   c2w=np.asarray(cam["views"][n]["c2w"], dtype=np.float64))
              for n in order]
    v = gv.viewer
    frames: list[np.ndarray] = []
    orig_fn = v.render_fn

    def rec_fn(cs, ts):
        img = orig_fn(cs, ts)
        frames.append(np.ascontiguousarray(img))
        return img
    v.render_fn = rec_fn
    walk = {"i": 0}

    def get_cs(client):  # the client camera: the bench view after the one just started
        walk["i"] += 1
        return states[walk["i"] % len(states)]
    v.get_camera_state = get_cs

    class _Ws:
        def queue_message(self, msg):
            pass

    class _Scene:
        _websock_interface = _Ws()

        def set_background_image(self, *a, **k):
            pass

    class _Client:
        scene = _Scene()
        client_id = -1

    def xv():
        st = clean.stage_timings()
        return int(st.get("xview_hits", 0)), int(st.get("xview_misses", 0))

    results = {}
    shots = {}
    K_bench = run.build_intrinsics(W, H, float(cam["fov_deg"]))
    w2cs = [_bench_pose(run, cs, W, H)[0] for cs in states]
    K_view = _bench_pose(run, states[0], W, H)[1]
    print(f"PIPE K_view==K_bench {bool(np.array_equal(np.asarray(K_view), np.asarray(K_bench)))}",
          flush=True)

    def report(mode, ms, digests, h, m, **extra):
        raw = hashlib.md5("".join(digests[0]).encode()).hexdigest()[:8]
        timed = ms[1:] or ms
        results[mode] = dict(ms_view=statistics.mean(timed), passes_ms=[round(x, 3) for x in ms],
                             raw_md5=raw, identical_across_passes=all(d == digests[0] for d in digests),
                             xview_hits=h, xview_misses=m, **extra)
        print(f"PIPE mode={mode} ms_view={results[mode]['ms_view']:.3f} "
              f"fps={1000.0 / results[mode]['ms_view']:.1f} passes_ms={results[mode]['passes_ms']} "
              f"raw_md5={raw} identical_across_passes={results[mode]['identical_across_passes']} "
              f"xview_hits={h} xview_misses={m}", flush=True)

    def viewer_loop(mode, pipelined):
        """FastRenderer.render_once("move") over the bench views (the live path)."""
        os.environ["GSPLAT_VIEWER_PIPELINE"] = "1" if pipelined else "0"
        fr = FastRenderer(viewer=v, client=_Client(), lock=v.lock)
        for _ in range(3):  # warm (JIT, caches), unhinted
            gv.pipeline.render(gauss, w2cs[0], K_bench, H, W)
        h0, m0 = xv()
        ms, digests = [], []
        for p in range(args.passes):
            frames.clear()
            walk["i"] = 0
            fr._latched = None
            t0 = time.perf_counter()
            for cs in states:
                fr.render_once(RenderTask("move", cs))
            ms.append((time.perf_counter() - t0) * 1000.0 / len(states))
            digests.append([hashlib.md5(f.tobytes()).hexdigest() for f in frames])
            if p == 0:
                shots[mode] = frames[0].copy()
        fr.running = False
        h1, m1 = xv()
        report(mode, ms, digests, h1 - h0, m1 - m0, shape=list(frames[0].shape))

    def bench_loop(mode, pipeline, hinted):
        """render/run.py's b2b loop: pipeline.render + _to_image only, bench K."""
        h0, m0 = xv()
        ms, digests = [], []
        for p in range(args.passes):
            imgs = []
            t0 = time.perf_counter()
            for i, w2c in enumerate(w2cs):
                if hinted:
                    run._set_next_extr(pipeline, w2cs[(i + 1) % len(w2cs)])
                imgs.append(run._to_image(pipeline.render(gauss, w2c, K_bench, H, W)))
            ms.append((time.perf_counter() - t0) * 1000.0 / len(w2cs))
            digests.append([hashlib.md5(run._to_u8(im).tobytes()).hexdigest() for im in imgs])
        pipeline.backend.next_extrinsics = None
        h1, m1 = xv()
        report(mode, ms, digests, h1 - h0, m1 - m0)

    def settings(pl):
        return {k: getattr(pl, k, None) for k in ("contrib_floor", "transmittance_threshold",
                                                   "min_opacity", "max_radius", "k_cap")}

    # 1) The viewer's own render settings (slider defaults).
    print(f"PIPE settings viewer {settings(pipe)}", flush=True)
    viewer_loop("viewer_off", False)
    viewer_loop("viewer_on", True)
    bench_loop("viewer_settings_bench_off", pipe, False)
    bench_loop("viewer_settings_bench_on", pipe, True)
    # 2) The bench's settings (render/run.py: Pipeline(CleanBackend, 32, contrib_floor)) on
    # the same backend, swapped into the viewer: the pipelined viewer path vs the golden.
    from gsplat.pipeline import Pipeline
    bp = Pipeline(pipe.backend, tile_size=32, contrib_floor=float(cam["contrib_floor"]))
    print(f"PIPE settings bench {settings(bp)}", flush=True)
    gv.pipeline = bp
    bench_loop("bench_off", bp, False)
    bench_loop("bench_on", bp, True)
    viewer_loop("bench_settings_viewer_off", False)
    viewer_loop("bench_settings_viewer_on", True)
    hero = shots["bench_settings_viewer_on"]
    pipe = bp

    ref = np.asarray(Image.open(args.reference).convert("RGB"))
    Image.fromarray(hero).save(args.out / "hero.png")
    d = np.abs(hero.astype(np.int16) - ref.astype(np.int16)).astype(np.uint8)
    Image.fromarray(np.clip(d.astype(np.int32) * 4, 0, 255).astype(np.uint8)).save(args.out / "diff.png")
    mse = float(np.mean((hero.astype(np.float64) - ref.astype(np.float64)) ** 2))
    psnr = 10.0 * np.log10(255.0 ** 2 / max(mse, 1e-12))
    results["hero"] = dict(psnr_vs=str(Path(args.reference).relative_to(REPO)), psnr_db=round(psnr, 3),
                           max_abs_diff=int(d.max()), md5=hashlib.md5(hero.tobytes()).hexdigest()[:8])
    print(f"PIPE hero psnr={psnr:.3f} dB vs {results['hero']['psnr_vs']} max_abs={int(d.max())} "
          f"saved {args.out}/hero.png diff.png", flush=True)
    (args.out / "pipeline_probe.json").write_text(json.dumps(results, indent=1))
    pipe.close()
    os._exit(0)


def _bench_pose(run, cs, W, H):
    """(w2c, K) for a CameraState built as the bench builds them."""
    import torch
    from gsplat.utils import c2w_to_w2c
    w2c = c2w_to_w2c(torch.from_numpy(np.asarray(cs.c2w, dtype=np.float32)))
    return w2c, torch.tensor(cs.get_K((W, H)), dtype=torch.float32)


if __name__ == "__main__":
    sys.exit(main())
