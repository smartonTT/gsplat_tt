"""pipeline_probe.golden_list_md5 equals opt/md5_golden.py's list md5 of `md5sum *` over the saved PNGs."""
import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_golden_list_md5_matches_md5sum_of_dumped_pngs(tmp_path):
    probe = _load("pipeline_probe", REPO / "opt" / "viewer" / "pipeline_probe.py")
    golden = _load("md5_golden", REPO / "opt" / "md5_golden.py")
    rng = np.random.default_rng(0)
    order = ["b", "a", "c"]  # name order differs from view order: sorting is by file name
    frames = [rng.integers(0, 256, (6, 8, 3), dtype=np.uint8) for _ in order]
    for i, (n, f) in enumerate(zip(order, frames)):
        Image.fromarray(f).save(tmp_path / f"view{i:02d}_{n}.png")
    lines = "".join(f"{hashlib.md5(p.read_bytes()).hexdigest()}  {p.name}\n"
                    for p in sorted(tmp_path.iterdir()))
    want = golden.list_md5(lines.encode())
    assert probe.golden_list_md5(frames, order) == want
    # and it is not the raw_md5 of the same frames: two different hashes, same pixels
    raw = hashlib.md5("".join(hashlib.md5(f.tobytes()).hexdigest() for f in frames).encode()).hexdigest()[:8]
    assert raw != want
