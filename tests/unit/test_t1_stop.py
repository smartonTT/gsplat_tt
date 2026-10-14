"""#802: T1 per-tile stop at the last contributor (gsplat.t1_stop) is lossless and keeps the
run-2188 hero count (2,209,295 of 4,478,225 pairs, pre-jclamp)."""
import importlib.util
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from gsplat import blend_backward_dev as dev  # noqa: E402
from gsplat import t1_balance as bal  # noqa: E402
from gsplat import t1_balance_dev as bd  # noqa: E402
from gsplat import t1_stop as ts  # noqa: E402
from gsplat.blend_backward_ref import BlendCfg  # noqa: E402

f32 = np.float32


def test_hero_fixture_kept():
    z = np.load(ROOT / "tests/fixtures/t1_dead_live.npz")
    counts = z["tile_counts"].astype(np.int64)
    live = np.unpackbits(z["live_bits"])[:int(z["n_pairs"])]
    tile_of = np.repeat(np.arange(len(counts)), counts)
    s = ts.stops_from_live(tile_of, live, len(counts))
    assert int(counts.sum()) == 4_478_225 and int(s.sum()) == 2_209_295
    assert (s <= counts).all()


def test_dead_values_match_pack_v2():
    rng = np.random.default_rng(0)
    fr = rng.standard_normal((16, 3)).astype(f32)
    fr[:4] = [[1, 1, 1], [-1, -1, -1], [1, -1, 1], [-1, 1, -1]]
    recs = np.zeros((16, 16), np.uint32)
    recs[:, :3] = fr.view(np.uint32)
    _, want = dev.pack_v2(np.zeros((16, 9, 4, 8), f32), fr[:, 0], fr[:, 1], fr[:, 2])
    got = ts.dead_values(recs)
    assert np.array_equal(got[:, :9].view(np.uint32), want.view(np.uint32))
    assert not got[:, 9:].any()


@pytest.fixture(scope="module")
def dense():
    spec = importlib.util.spec_from_file_location("synth", ROOT / "docs/t685/synth.py")
    sy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sy)
    rng = np.random.default_rng(1)
    H, W, N, cfg = 32, 64, 300, BlendCfg()
    m2 = np.stack([rng.uniform(-8, W + 8, N), rng.uniform(-8, H + 8, N)], 1).astype(f32)
    s = np.exp(rng.uniform(np.log(4.0), np.log(16.0), (N, 2)))
    th = rng.uniform(0, np.pi, N)
    c, sn = np.cos(th), np.sin(th)
    a = c * c * s[:, 0] ** 2 + sn * sn * s[:, 1] ** 2 + .3
    b = sn * sn * s[:, 0] ** 2 + c * c * s[:, 1] ** 2 + .3
    o = c * sn * (s[:, 0] ** 2 - s[:, 1] ** 2)
    det = a * b - o * o
    con = np.stack([b / det, -o / det, a / det], 1).astype(f32)
    op = rng.uniform(0.5, 0.99, N).astype(f32)
    col = rng.uniform(0, 1, (N, 3)).astype(f32)
    order, valid = rng.permutation(N).astype(np.int64), np.ones(N, bool)
    img = sy.forward(m2, con, op, col, order, valid, H, W, cfg)
    g = (rng.standard_normal((H, W, 3)) * 1e-3).astype(f32)
    p = dev.prep(m2, con, op, col, order, valid, img, g, cfg)
    return p, dev.emulate(p)


@pytest.mark.parametrize("margin", [False, True])
def test_truncate_lossless(dense, margin):
    p, full = dense
    ps = ts.pixel_stops(p, margin)
    q = ts.truncate(p, ts.stops_from_pixel(ps))
    assert q["stop_kept"] < q["stop_total"] and len(q["dead_slots"]) == q["stop_total"] - q["stop_kept"]
    got = ts.fill_dead(dev.emulate(q), q)
    assert np.array_equal(got, full)                      # numerically identical everywhere
    kept = np.setdiff1d(np.arange(p["n_slots"]), q["dead_slots"])
    assert np.array_equal(got[kept].view(np.uint32), full[kept].view(np.uint32))
    if margin:
        assert (ps >= ts.pixel_stops(p, False)).all()    # margin never stops earlier
    else:                                                 # strict: each tile's last kept pair is used
        t = np.asarray(q["tiles"], np.int64)
        last = p["recs"][t[:, 1] + t[:, 2] - 1, 13]
        assert (np.abs(full[last]).sum(1) > 0).all()


def test_lpt_plan_on_truncated(dense):
    p, full = dense
    q = ts.truncate(p, ts.tile_stops(p))
    pl = bal.plan(q["tiles"], q["recs"][:, 12], 4, False, 8)
    r = bal.apply_plan(q, pl)
    assert int(np.asarray(r["tiles"], np.int64)[:, 2].sum()) == q["stop_kept"]
    assert np.array_equal(ts.fill_dead(dev.emulate(r), r), full)


def test_cli_and_plan_keep_dead(dense, tmp_path):
    p, _ = dense
    dev.write(p, str(tmp_path / "prep"))
    assert ts.main([str(tmp_path / "prep"), str(tmp_path / "cut")]) == 0
    assert ts.main([str(tmp_path / "cut"), str(tmp_path / "x")]) == 2       # refuses a truncated dir
    q = ts.truncate(p, ts.tile_stops(p))
    got = bd.read_prep(str(tmp_path / "cut"))
    assert np.array_equal(got["dead_slots"], q["dead_slots"])
    assert np.array_equal(got["dead_vals"].view(np.uint32), q["dead_vals"].view(np.uint32))
    assert bd.main(["plan", str(tmp_path / "cut"), str(tmp_path / "pl"), "--cores", "4"]) == 0
    assert (tmp_path / "pl/dead.u32").read_bytes() == (tmp_path / "cut/dead.u32").read_bytes()
    assert (tmp_path / "pl/dead.f32").read_bytes() == (tmp_path / "cut/dead.f32").read_bytes()


def test_env_flag(monkeypatch):
    monkeypatch.delenv(ts.ENV, raising=False)
    assert not ts.enabled()
    monkeypatch.setenv(ts.ENV, "0")
    assert not ts.enabled()
    monkeypatch.setenv(ts.ENV, "1")
    assert ts.enabled()
