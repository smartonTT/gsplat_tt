"""#750 step 2: host glue of the dump-free chain (forward image -> loss inputs -> T1 dump)."""
from pathlib import Path

import numpy as np

from gsplat import train_loop_dev

ROOT = Path(__file__).resolve().parents[2]
DUMP = ROOT / "benchmarks/train_oracle/t0_sh0_64x64_n50.npz"


def test_chain_glue_round_trip(tmp_path):
    d = np.load(DUMP)
    H, W = int(d["H"]), int(d["W"])
    img = np.asarray(d["image"], np.float32) + np.float32(1e-3)
    img.tofile(tmp_path / "img.f32")
    rep = train_loop_dev.loss_prep(str(DUMP), str(tmp_path / "img.f32"), str(tmp_path / "loss"))
    assert abs(rep["image_max_abs_vs_oracle"] - 1e-3) < 1e-5
    c = np.fromfile(tmp_path / "loss" / "consts.f32", np.float32)
    assert c.size == 2 * 3 * (-(-H // 32) * 32) * (-(-W // 32) * 32) and np.isfinite(c).all()
    assert np.array_equal(np.fromfile(tmp_path / "loss" / "target.f32", np.float32).reshape(H, W, 3),
                          np.asarray(d["target"], np.float32))
    (tmp_path / "loss" / "out").mkdir()
    g = np.asarray(d["dL_dimage"], np.float32) * np.float32(2)
    g.tofile(tmp_path / "loss" / "out" / "grad.f32")
    (tmp_path / "loss" / "out" / "loss.txt").write_text("0.5 0.1\n")
    rep = train_loop_dev.chain_dump(str(DUMP), str(tmp_path / "loss"), str(tmp_path / "c.npz"))
    assert rep["loss"] == 0.5 and rep["grad_finite"] and abs(rep["grad_l2rel_vs_oracle"] - 1.0) < 1e-5
    c = np.load(tmp_path / "c.npz")
    # The chain marker switches train_step_dev check to the chain T0 gate (fp32 loss gradient).
    assert set(c.files) == set(d.files) | {"chain_dl_dimage_max_rel"}
    assert abs(float(c["chain_dl_dimage_max_rel"]) - 1.0) < 1e-5
    assert abs(rep["grad_max_rel_vs_oracle"] - 1.0) < 1e-5
    assert np.array_equal(c["image"], img) and np.array_equal(c["dL_dimage"], g)
    assert np.array_equal(c["grad_means"], d["grad_means"])


def test_dssim_l1_f64_matches_torch_oracle():
    # The hero loss oracle (#795 c) is independent of the device form; on the crop dumps it equals
    # the torch autograd oracle (gsplat/train_oracle.py loss_fn) to fp64 rounding.
    for name in ("sh0", "sh3"):
        d = np.load(ROOT / f"benchmarks/train_oracle/t0_{name}_64x64_n50.npz")
        loss, g = train_loop_dev.dssim_l1_f64(d["image"], d["target"])
        ref = np.asarray(d["dL_dimage"])
        assert abs(loss - float(d["loss"])) < 1e-12
        assert np.abs(g - ref).max() < 1e-12 * np.abs(ref).max()


def test_chain_dump_without_oracle_loss_uses_host_oracle(tmp_path):
    d = np.load(DUMP)
    H, W = int(d["H"]), int(d["W"])
    keys = {k: d[k] for k in d.files if k != "loss"}
    keys["dL_dimage"] = np.ones_like(d["dL_dimage"])  # like the hero dump: not this loss's gradient
    np.savez(tmp_path / "hero.npz", **keys)
    img = np.asarray(d["image"], np.float32) + np.float32(1e-3)
    img.tofile(tmp_path / "img.f32")
    train_loop_dev.loss_prep(str(tmp_path / "hero.npz"), str(tmp_path / "img.f32"), str(tmp_path / "loss"))
    loss, g = train_loop_dev.dssim_l1_f64(img, d["target"])
    (tmp_path / "loss" / "out").mkdir()
    g.astype(np.float32).tofile(tmp_path / "loss" / "out" / "grad.f32")
    (tmp_path / "loss" / "out" / "loss.txt").write_text(f"{loss:.9g} 0\n")
    rep = train_loop_dev.chain_dump(str(tmp_path / "hero.npz"), str(tmp_path / "loss"), str(tmp_path / "c.npz"))
    assert rep["oracle"] == "host_f64_dssim_l1" and rep["loss_abs_vs_oracle"] < 1e-8
    assert rep["grad_max_rel_vs_oracle"] < 1e-6 and rep["grad_l2rel_vs_oracle"] < 1e-6


def test_hero_fwd_tiles_splits_by_subchunk_path():
    import importlib.util
    spec = importlib.util.spec_from_file_location("hft", ROOT / "docs/train-step/hero_fwd_tiles.py")
    hft = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hft)
    H, W = 70, 100  # 3 x 4 tiles, the last row and column partial
    model = np.random.default_rng(0).random((H, W, 3))
    img = model.astype(np.float32)
    img[64 + 3, 96 + 1, 2] += 0.01  # tile 11 (tx 3, ty 2), a CONTINUE tile
    counts = np.full(12, 100, np.uint32)
    counts[11], counts[0] = 20000, 9000
    r = hft.tile_report(img, model, counts)
    assert not r["pass"] and r["bad_tiles"] == 1 and r["worst"][0]["tile"] == 11
    assert r["gt_8192"] == {"tiles": 2, "bad": 1, "max_abs": r["max_abs"]} and r["le_8192"]["bad"] == 0
    assert r["bad_by_subchunks"]["3"] == [1, 1] and r["bad_by_subchunks"]["2"] == [0, 1]
