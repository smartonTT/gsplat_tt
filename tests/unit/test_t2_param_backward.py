"""T2 gate (task #689): segment reduce + projection/SH0 backward + Adam vs the T0 oracle.

PLAN-single-chip.md T2 acceptance: param grads within rel 1e-2 of T0, Adam (params,
updates, bf16 and fp32 moments) within rel 1e-3, on the committed SH0 dump and on a fresh
second oracle scene (another seed). Also the rank-major reduce order, the dense flag and
the pfwc epilogue layout.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch")
from gsplat import t2_param_backward as t2  # noqa: E402
from gsplat import train_oracle as o  # noqa: E402

DUMP = ROOT / "benchmarks" / "train_oracle" / "t0_sh0_64x64_n50.npz"


def _assert_pass(rep):
    assert rep["visible_match"], rep
    assert rep["worst_grad"] <= 1e-2, rep["grads"]
    assert rep["worst_adam"] <= 1e-3, (rep["adam_bf16"], rep["adam_fp32"])
    assert rep["worst_e2e_param"] <= 1e-3, rep["e2e_bf16_param"]
    assert rep["pass"]


def test_dump_gate():
    _assert_pass(t2.check(np.load(DUMP)))


def test_fresh_case_gate():
    d = o.build_case(0, n=50, height=64, width=64, seed=7)  # ~16 s (torch oracle)
    rep = t2.check(d)
    assert rep["max_count"] >= 2  # Gaussians spanning several tiles: a real segmented reduce
    _assert_pass(rep)


def test_rejects_sh3():
    d = dict(np.load(DUMP))
    d["sh_degree"] = np.int32(3)
    with pytest.raises(ValueError):
        t2.check(d)


def test_segment_reduce_matches_t1_sum_order():
    bw = t2.oracle_slots(np.load(DUMP))
    g = t2.segment_reduce(bw["slots"], bw["P"], bw["counts"])
    ref = np.concatenate([bw["dL_dmeans2d"], bw["dL_dconic"], bw["dL_dopacity"][:, None],
                          bw["dL_dcolors"]], 1)
    assert np.array_equal(g, ref)  # bit-exact: same fp32 order (j ascending)


def _step(d, dense):
    params = {k: np.asarray(d[f"param_{k}"], np.float32) for k in t2.PARAM_NAMES}
    bw = t2.oracle_slots(d)
    m = {k: d[f"adam_bf16_m_in_{k}"] for k in t2.PARAM_NAMES}
    v = {k: d[f"adam_bf16_v_in_{k}"] for k in t2.PARAM_NAMES}
    out = t2.t2_step(params, bw["slots"], bw["P"], bw["counts"], d["w2c"], d["K"],
                     int(d["H"]), int(d["W"]), m, v, int(d["adam_step_in"]),
                     json.loads(str(d["adam_lr_json"])), dense=dense)
    return params, m, out


def test_sparse_leaves_invisible_rows_dense_moves_them():
    d = np.load(DUMP)
    params, m, sp = _step(d, dense=False)
    _, _, de = _step(d, dense=True)
    inv = ~sp["visible"]
    assert inv.any()
    for k in t2.PARAM_NAMES:
        assert np.array_equal(sp["params"][k][inv], params[k][inv])
        assert np.array_equal(sp["m"][k][inv], np.asarray(m[k], np.float32)[inv])
        assert np.array_equal(de["params"][k][~inv], sp["params"][k][~inv])
    # dense: zero-grad rows still decay their moments
    assert not np.array_equal(de["m"]["means"][inv], np.asarray(m["means"], np.float32)[inv])


def test_bf16_round_is_rne():
    x = np.array([1.0, 1.00390625, 1.01171875, -2.5e-3, 3.0e38], np.float32)
    ref = torch.from_numpy(x).to(torch.bfloat16).float().numpy()
    assert np.array_equal(t2.bf16_round(x), ref)


def test_pfwc_epilogue_layout():
    d = np.load(DUMP)
    _, _, out = _step(d, dense=False)
    pf = out["pfwc"]
    n = int(d["N"])
    assert set(pf) == set(t2.PFWC_STREAMS)
    assert all(a.shape == (t2.ELEMS_PER_TILE,) and a.dtype == np.float32 for a in pf.values())
    p = out["params"]
    assert np.array_equal(pf["mx"][:n], p["means"][:, 0])
    assert not pf["mx"][n:].any()
    # cov3d unique vs the oracle's covariance of the updated params
    q = torch.from_numpy(p["quats"]).double()
    Rq = o.quat_to_rotation_matrix(q / q.norm(dim=-1, keepdim=True))
    M = Rq * torch.exp(torch.from_numpy(p["log_scales"]).double()).unsqueeze(1)
    cov = (M @ M.transpose(1, 2)).numpy()
    for k, (i, j) in zip(("c00", "c01", "c02", "c11", "c12", "c22"),
                         ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))):
        assert t2.rel_error(pf[k][:n], cov[:, i, j]) <= 1e-4
    raw = o.SH_C0 * p["sh"][:, 0].astype(np.float64) + 0.5
    assert t2.rel_error(pf["col_r"][:n], np.clip(raw[:, 0], 0, 1)) <= 1e-5


def test_degenerate_cov2d_row_without_slots_has_zero_grads():
    # #750 hero: rows with det(cov2d) 0 or overflowing a/c and no slots gave NaN grads (0 * inf).
    f = np.float32
    K = np.array([[1100, 0, 600], [0, 1100, 400], [0, 0, 1]], f)
    p = {"means": np.array([[0.05, 0.02, 0.25], [0.0, 0.0, 4.0]], f),
         "log_scales": np.array([[20, 20, 20], [-3, -3, -3]], f),
         "quats": np.array([[1, 0, 0, 0], [1, 0, 0, 0]], f),
         "opacity_logits": np.zeros(2, f), "sh": np.zeros((2, 1, 3), f)}
    g2d = np.zeros((2, 9), f)
    g2d[1] = 1e-3
    with np.errstate(all="ignore"):
        grads, _ = t2.projection_backward_sh0(p, g2d, np.eye(4, dtype=f), K, 800, 1200)
    for k, a in grads.items():
        assert np.array_equal(a[0], np.zeros_like(a[0])), k
        assert np.isfinite(a[1]).all(), k
    assert np.abs(grads["means"][1]).max() > 0
