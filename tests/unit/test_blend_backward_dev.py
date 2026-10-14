"""T1 device decomposition (#685): the numpy model of the device kernel (same prep files,
vector masks and lane order) must match the reference slots on the T0 oracle dumps."""
import os
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from gsplat import blend_backward_dev as dev  # noqa: E402

DUMPS = sorted(str(p) for p in (ROOT / "benchmarks" / "train_oracle").glob("t0_sh*_64x64_n50.npz"))


@pytest.mark.parametrize("path", DUMPS, ids=[os.path.basename(p) for p in DUMPS])
def test_emulator_matches_ref(path, tmp_path):
    d, cfg = dev.load_dump(path)
    p = dev.prep(d["means2d"], d["conic"], d["opacity"], d["colors"], d["order"], d["valid"],
                 d["image"], d["dL_dimage"], cfg)
    dev.write(p, str(tmp_path))
    assert np.fromfile(tmp_path / "recs.u32", np.uint32).size == 16 * p["n_slots"]
    rep = dev.compare_slots(dev.emulate(p), dev.ref_slots(d, cfg))
    assert rep["rel_err"] < 1e-4, rep


def test_vector_lanes_cover_tile():
    assert sorted(dev.vector_lanes().ravel().tolist()) == list(range(1024))
