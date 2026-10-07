#!/usr/bin/env python3
"""t366: host-only check of the d2h destination cost: copy a 3 MB u8 image (1024x1024x3) from a
persistent source into (a) a fresh numpy array each time (as render.cpp's per-view py::array)
or (b) one reused array. Reports median ms over N reps. No device use."""
import time, numpy as np
N = 200; shape = (1024, 1024, 3)
src = np.random.randint(0, 255, shape, np.uint8)
def run(fresh):
    keep = np.empty(shape, np.uint8); ts = []
    for _ in range(N):
        t = time.perf_counter()
        dst = np.empty(shape, np.uint8) if fresh else keep
        np.copyto(dst, src)
        ts.append((time.perf_counter() - t) * 1e3)
        if fresh: keep2 = dst  # one live image at a time, like the bench
    return np.median(ts)
for k in range(2):
    print(f"fresh={run(True):.3f} ms  reused={run(False):.3f} ms")
