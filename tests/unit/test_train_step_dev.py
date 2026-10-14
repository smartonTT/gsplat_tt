"""#750: train_step_dev prep on hero-size dumps (fast reference slots) and hero.py's T2 inputs."""
import importlib.util
import json
from pathlib import Path

import numpy as np

from gsplat import blend_backward_dev as t1
from gsplat import t2_param_backward as t2
from gsplat import train_step_dev

ROOT = Path(__file__).resolve().parents[2]
DUMP = ROOT / "benchmarks/train_oracle/t0_sh0_64x64_n50.npz"


def test_prep_fast_ref_slots_equal_blend_backward(tmp_path):
    info = train_step_dev.prep(str(DUMP), tmp_path)
    d, cfg = t1.load_dump(str(DUMP))
    r = np.load(tmp_path / "ref_slots.npz")
    ref = t1.ref_slots(d, cfg)
    assert r["slots"].shape == ref.shape and info["n_slots"] == ref.shape[0]
    assert t1.rel_error(r["slots"].astype(np.float64), ref.astype(np.float64)) < 1e-5
    assert r["tie"].dtype == bool and r["tie"].shape == (ref.shape[0],)
    t2in = np.load(tmp_path / "t2" / "ref_in.npz")
    assert np.array_equal(t2in["slots"], r["slots"])


def test_hero_t2_inputs_keys_for_prep_dump():
    spec = importlib.util.spec_from_file_location("hero", ROOT / "docs/t685/hero.py")
    hero = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hero)
    names = (["x", "y", "z"] + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)]
             + ["opacity"] + [f"f_dc_{i}" for i in range(3)])
    col = {n: i for i, n in enumerate(names)}
    rows = np.random.default_rng(0).normal(size=(7, len(names))).astype(np.float32)
    c2w = np.eye(4)
    c2w[:3, 3] = [1, 2, 3]
    out = hero.t2_inputs(rows, col, c2w, {"fx": 100, "fy": 110, "cx": 50, "cy": 40})
    for k in t2.PARAM_NAMES:
        assert out[f"param_{k}"].shape[0] == 7
        assert not out[f"adam_bf16_m_in_{k}"].any() and not out[f"adam_bf16_v_in_{k}"].any()
    assert out["param_sh"].shape == (7, 1, 3)
    assert np.allclose(out["param_sh"][:, 0], rows[:, 11:14])
    assert np.allclose(out["param_opacity_logits"], rows[:, 10])
    assert np.allclose(out["w2c"][:3, 3], [-1, -2, -3]) and out["K"][1, 2] == 40
    assert set(json.loads(out["adam_lr_json"])) >= {"means", "log_scales", "quats", "opacity_logits", "sh_dc"}
    assert int(out["adam_step_in"]) == 0
