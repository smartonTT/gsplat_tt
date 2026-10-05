"""JPEG encode backends of the viewer's sender thread, timed off-device (task #271).

For each backend in gsplat.nerfview_viewer.JPEG_ENCODERS: median encode ms of an RGB uint8
frame at q40/q70, the JPEG size, and how much a pure-Python thread slows while the encode
runs on another thread (GIL held by the encode shows up as lost Python progress).
  .venv/bin/python opt/viewer/encode_bench.py [image.png] [--n 50]
"""
from __future__ import annotations

import argparse
import statistics
import sys
import threading
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def python_progress(stop: threading.Event, out: list[int]) -> None:
    n = 0
    while not stop.is_set():
        n += 1
    out.append(n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?", default=str(REPO / "benchmarks" / "reference_v2" / "hero.png"))
    ap.add_argument("--n", type=int, default=50)
    args = ap.parse_args()
    from PIL import Image
    from gsplat.nerfview_viewer import JPEG_ENCODERS, make_jpeg_encoder

    img = np.ascontiguousarray(np.asarray(Image.open(args.image).convert("RGB")))
    print(f"[encode_bench] {args.image} {img.shape}")
    sys.setswitchinterval(0.005)
    # Python progress per second with no encode running.
    stop, cnt = threading.Event(), []
    th = threading.Thread(target=python_progress, args=(stop, cnt)); th.start()
    time.sleep(1.0); stop.set(); th.join()
    base_rate = cnt[0] / 1.0
    for name in JPEG_ENCODERS:
        try:
            enc = make_jpeg_encoder(name)
        except ImportError as e:
            print(f"ENC {name}: unavailable ({e})")
            continue
        for q in (40, 70):
            for _ in range(3):
                enc(img, q)
            ts = []
            for _ in range(args.n):
                t = time.perf_counter(); data = enc(img, q); ts.append((time.perf_counter() - t) * 1e3)
            # Same encodes with a Python thread competing for the GIL.
            stop, cnt = threading.Event(), []
            th = threading.Thread(target=python_progress, args=(stop, cnt)); th.start()
            t0 = time.perf_counter(); tc = []
            for _ in range(args.n):
                t = time.perf_counter(); enc(img, q); tc.append((time.perf_counter() - t) * 1e3)
            wall = time.perf_counter() - t0
            stop.set(); th.join()
            py_share = cnt[0] / wall / base_rate
            print(f"ENC {name} q{q}: median {statistics.median(ts):.2f} ms (min {min(ts):.2f}) "
                  f"| with busy py thread {statistics.median(tc):.2f} ms, py thread keeps "
                  f"{py_share * 100:.0f}% of its speed | {len(data) / 1024:.0f} KiB")


if __name__ == "__main__":
    main()
