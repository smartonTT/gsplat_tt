"""T0 forward harness (#716): the numpy model of the BLEND_TRAIN device forward (same prep
files, UNORM16 opacity/colours, fp32 math) must match the oracle image and final T."""
import os
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from gsplat import blend_forward_dev as fwd  # noqa: E402
from gsplat.blend_backward_dev import cfg_from_dump, tilize  # noqa: E402

DUMPS = sorted(str(p) for p in (ROOT / "benchmarks" / "train_oracle").glob("t0_sh*_64x64_n50.npz"))


@pytest.mark.parametrize("path", DUMPS, ids=[os.path.basename(p) for p in DUMPS])
def test_model_matches_oracle(path, tmp_path):
    d = np.load(path, allow_pickle=False)
    cfg = cfg_from_dump(d)
    H, W = d["image"].shape[:2]
    p = fwd.prep(d["means2d"], d["conic"], d["opacity"], d["colors"], d["order"], d["valid"], H, W)
    fwd.write(p, str(tmp_path))
    n = int(np.asarray(d["valid"]).sum())
    assert len(p["tiles"]) == 4 and (p["tiles"][:, 2] == n).all()
    assert np.fromfile(tmp_path / "recs.u32", np.uint32).size == 8 * len(p["recs"])
    img, fT, _ = fwd.to_image(fwd.emulate(p, cfg), p["tiles"], H, W, cfg.background)
    rep = fwd.compare(d, img, fT, 1e-4)
    assert rep["pass"], rep


def test_untilize_inverts_tilize():
    t = np.arange(1024, dtype=np.float32).reshape(32, 32)
    assert (fwd.untilize(tilize(t)) == t).all()


def test_unorm16_roundtrip():
    v = np.linspace(0.0, 1.0, 1001)
    assert np.abs(fwd.unorm16_decode(fwd.unorm16(v)) - v).max() <= 2.0 ** -16
