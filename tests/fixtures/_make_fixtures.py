"""One-shot script to generate synthetic test fixtures."""
from pathlib import Path
import json
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

def make_pair(seed, name, identical=True):
    rng = np.random.default_rng(seed)
    ref = (rng.random((1024, 1024, 3)) * 255).astype(np.uint8)
    Image.fromarray(ref).save(ROOT / "reference" / f"stitch_{name}.png")
    cand = ref.copy() if identical else (ref + rng.integers(0, 2, size=ref.shape, dtype=np.int16)).clip(0, 255).astype(np.uint8)
    Image.fromarray(cand).save(ROOT / "iter-test-good" / f"{name}.png")

for n, s in [("hero", 1), ("side", 2), ("top", 3)]:
    make_pair(s, n, identical=True)  # identical PNGs → PSNR = inf (> 80 dB threshold)

# 10 cycles × 3 views = 30 rows
timing = []
views = ["hero", "side", "top"]
for cycle in range(10):
    for v in views:
        timing.append({"cycle": cycle, "view": v, "kernel_ms": 16.5 + (cycle % 3) * 0.1})
(ROOT / "iter-test-good" / "timing.jsonl").write_text(
    "\n".join(json.dumps(t) for t in timing)
)
