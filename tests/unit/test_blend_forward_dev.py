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
# #763 (3DGS colour max(SH+0.5, 0), no clamp at 1) left 2 sh3 colours above 1 (max 1.084); the
# UNORM16 record colour clips them at 1, so the device forward cannot match that oracle image.
COLOUR_GT1 = pytest.mark.xfail(strict=True, reason="UNORM16 record colour clips the #763 colours > 1")
ORACLE_DUMPS = [pytest.param(p, marks=COLOUR_GT1) if "sh3" in p else p for p in DUMPS]


@pytest.mark.parametrize("path", ORACLE_DUMPS, ids=[os.path.basename(p) for p in DUMPS])
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


def _crop(path, scale=1.0):
    d = np.load(path, allow_pickle=False)
    H, W = d["image"].shape[:2]
    con = np.asarray(d["conic"], np.float32) * np.float32(scale)
    return d, H, W, (d["means2d"], con, d["opacity"], d["colors"], d["order"], d["valid"])


@pytest.mark.parametrize("scale", [1.0, 4.0, 64.0])
@pytest.mark.parametrize("path", DUMPS, ids=[os.path.basename(p) for p in DUMPS])
def test_prep_binned_is_prep_cut_to_rects(path, scale):
    """#781: prep_binned == prep() with each tile's list cut to the Gaussians whose 3-sigma
    rectangle covers it (same order, same record bits, page-aligned tiles). Larger conic scales
    shrink the rectangles so that some tiles drop Gaussians."""
    from gsplat.blend_backward_dev import tile_rects_vec
    d, H, W, a = _crop(path, scale)
    full, binned = fwd.prep(*a, H, W), fwd.prep_binned(*a, H, W)
    sel = [int(g) for g in np.asarray(a[4]) if bool(a[5][g])]
    rects = tile_rects_vec(a[0], a[5], a[1], H, W)
    tiles_x = (W + 31) // 32
    assert (binned["tiles"][:, 0] == full["tiles"][:, 0]).all()
    assert (binned["ramps"] == full["ramps"]).all()
    assert (binned["tiles"][:, 3:] == 0).all()
    page, dropped = 0, 0
    for (tid, fp, fn), (_, bp, bn) in zip(full["tiles"][:, :3], binned["tiles"][:, :3]):
        ty, tx = divmod(int(tid), tiles_x)
        keep = np.array([rects[g, 0] <= tx <= rects[g, 1] and rects[g, 2] <= ty <= rects[g, 3] for g in sel])
        want = full["recs"][2 * fp:2 * fp + fn][keep]
        assert bp == page and bn == len(want)
        assert (binned["recs"][2 * bp:2 * bp + bn] == want).all()
        assert (binned["recs"][2 * bp + bn:2 * (bp + (bn + 1) // 2)] == 0).all()   # pad record
        page += (bn + 1) // 2
        dropped += fn - bn
    assert len(binned["recs"]) == 2 * page
    if scale == 64.0:
        assert dropped > 0


@pytest.mark.parametrize("binned", [False, True])
@pytest.mark.parametrize("path", DUMPS, ids=[os.path.basename(p) for p in DUMPS])
def test_emulate_fast_equals_emulate(path, binned):
    d, H, W, a = _crop(path, 4.0)
    cfg = cfg_from_dump(d)
    p = (fwd.prep_binned if binned else fwd.prep)(*a, H, W)
    ref, got = fwd.emulate(p, cfg), fwd.emulate_fast(p, cfg)
    assert np.abs(got - ref).max() <= 1e-6


@pytest.mark.parametrize("path", ORACLE_DUMPS, ids=[os.path.basename(p) for p in DUMPS])
def test_binned_model_matches_forward_fast(path):
    """The binned device model against blend_backward_dev.forward_fast (the hero dump's image):
    same tile rule, UNORM16 inputs and tile-local power form, so within the 1e-4 forward bar."""
    from gsplat.blend_backward_dev import forward_fast
    d, H, W, a = _crop(path)
    cfg = cfg_from_dump(d)
    p = fwd.prep_binned(*a, H, W)
    img, _, _ = fwd.to_image(fwd.emulate_fast(p, cfg), p["tiles"], H, W, cfg.background)
    ref, _ = forward_fast(*a, H, W, cfg)
    assert np.abs(img - ref).max() <= 1e-4
