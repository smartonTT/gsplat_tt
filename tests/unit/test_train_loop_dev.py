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
    assert set(c.files) == set(d.files)
    assert np.array_equal(c["image"], img) and np.array_equal(c["dL_dimage"], g)
    assert np.array_equal(c["grad_means"], d["grad_means"])
