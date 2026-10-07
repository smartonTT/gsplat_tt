"""Replay one viewer camera pose through render_clean (task #357).

    TTW_ALLOW_DIRECT=1 .venv/bin/python opt/viewer/replay_pose.py POSE.json [--timeout 30] [-n 3]

POSE.json is a pose the render watchdog wrote (viewer_hang_pose_*.json) or a
viewer_poses.jsonl file (its last line): w2c and K exactly as the viewer passed
them, the resolution and the backend's cull settings. Renders it n times
through the same Pipeline + CleanBackend as the viewer, under the same
RenderWatchdog: a render over --timeout writes its stacks to replay_hang.log
next to POSE.json and exits 86 (the hang reproduced). Otherwise it prints ms
and the frame md5 and saves the frame next to POSE.json. Opens the TT device,
so run it under the device lock (devrun / ttp lock), like render/run.py.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pose", type=Path)
    ap.add_argument("--ply", default=str(REPO / "scenes" / "bicycle.ply"))
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("-n", type=int, default=3)
    args = ap.parse_args()
    if os.environ.get("TTW_DEVRUN") != "1" and os.environ.get("TTW_ALLOW_DIRECT") != "1":
        print("[replay] REFUSING to open the TT device without TTW_DEVRUN or TTW_ALLOW_DIRECT=1",
              file=sys.stderr)
        return 3
    text = args.pose.read_text().strip()
    pose = json.loads(text.splitlines()[-1] if args.pose.suffix == ".jsonl" else text)

    import numpy as np
    import torch
    from PIL import Image

    spec = importlib.util.spec_from_file_location("render_run", REPO / "render" / "run.py")
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    from gsplat.loading_gaussians import load_ply
    from gsplat.pipeline import Pipeline
    from gsplat.render_watchdog import RenderWatchdog

    backend = run.CleanBackend(k_cap=3.0)
    for k, v in (pose.get("settings") or {}).items():
        if v is not None and hasattr(backend, k):
            setattr(backend, k, v)
    pipe = Pipeline(backend, tile_size=32, k_cap=3.0)
    out = args.pose.parent
    # The first render (device open, JIT) gets 600 s more than the limit.
    wd = RenderWatchdog(backend, out / "replay_hang.log", out / "replay_poses.jsonl",
                        limit_s=args.timeout, first_limit_s=600.0 + args.timeout).start()
    render = wd.wrap(pipe.render)
    gauss = load_ply(args.ply)
    extr = torch.tensor(pose["w2c"], dtype=torch.float32)
    K = torch.tensor(pose["K"], dtype=torch.float32)
    W, H = int(pose["width"]), int(pose["height"])
    print(f"[replay] {args.pose} {W}x{H} settings={pose.get('settings')}", flush=True)
    try:
        for i in range(args.n):
            t = time.perf_counter()
            img = np.ascontiguousarray(render(gauss, extr, K, H, W).image)
            ms = (time.perf_counter() - t) * 1000.0
            print(f"[replay] render {i}: {ms:.2f} ms md5={hashlib.md5(img.tobytes()).hexdigest()[:8]}",
                  flush=True)
        if img.dtype != np.uint8:
            img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
        png = args.pose.with_suffix(".replay.png")
        Image.fromarray(img).save(png)
        print(f"[replay] saved {png}: no hang in {args.n} renders", flush=True)
    finally:
        pipe.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
