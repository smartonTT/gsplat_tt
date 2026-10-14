"""t2_dev layout round trip: emulate the device's page outputs from t2_step and check()."""
import json

import numpy as np
import pytest

from gsplat import t2_dev as td
from gsplat import t2_param_backward as t2


def _emulate(indir, outdir):
    meta = json.loads((indir / "meta.json").read_text())
    N, ng = meta["n"], meta["n_groups"]
    ref = td.reference(indir)
    # Inputs as the device reads them, back from the pages.
    pin = td.from_pages(np.fromfile(indir / "params.f32", np.float32).reshape(ng, 1024), N, 14)
    r = np.load(indir / "ref_in.npz")
    assert np.array_equal(pin, td.param_cols({k: r[f"p_{k}"] for k in t2.PARAM_NAMES}))
    outdir.mkdir(parents=True, exist_ok=True)
    td.to_pages(td.param_cols(ref["params"]), ng).astype(np.float32).tofile(outdir / "params_out.f32")
    for w in ("m", "v"):
        td.to_pages(td.bf16_bits(td.param_cols(ref[w])), ng).tofile(outdir / f"{w}_out.u16")
    td.to_pages(td.param_cols(ref["grads"]), ng).astype(np.float32).tofile(outdir / "grads.f32")
    pf = np.zeros((13, ng * 64), np.float32)
    for i, s in enumerate(t2.PFWC_STREAMS):
        pf[i, :N] = ref["pfwc"][s][:N]
    pf.tofile(outdir / "pfwc.f32")


def _emulate_grads(indir, outdir, cols):
    """The device's outputs when its grads are `cols` (14 param columns): Adam and pfwc follow
    its own grads."""
    meta = json.loads((indir / "meta.json").read_text())
    N, ng = meta["n"], meta["n_groups"]
    ref = td.reference(indir)
    ar = td.adam_ref(indir, td.from_cols(cols), ref["visible"],
                     np.asarray(ref["grads"]["sh"])[:, 1:])
    td.to_pages(cols, ng).astype(np.float32).tofile(outdir / "grads.f32")
    td.to_pages(td.param_cols(ar["params"]), ng).astype(np.float32).tofile(outdir / "params_out.f32")
    for w in ("m", "v"):
        td.to_pages(td.bf16_bits(td.param_cols(ar[w])), ng).tofile(outdir / f"{w}_out.u16")
    pf = np.zeros((13, ng * 64), np.float32)
    for i, s in enumerate(t2.PFWC_STREAMS):
        pf[i, :N] = ar["pfwc"][s][:N]
    pf.tofile(outdir / "pfwc.f32")


@pytest.mark.parametrize("dump", ["t0_sh0_64x64_n50.npz", "t0_sh3_64x64_n50.npz"])
def test_dump_roundtrip(tmp_path, dump):
    # sh3: the reference carries 16 SH bands, the device only SH0; check() compares band 0.
    td.prep_dump(f"benchmarks/train_oracle/{dump}", tmp_path / "in")
    _emulate(tmp_path / "in", tmp_path / "out")
    rep = td.check(tmp_path / "in", tmp_path / "out")
    assert rep["pass"] and rep["worst_adam"] == 0.0 and rep["m_bits_equal"]


def test_synth_meta_and_vis(tmp_path):
    meta = td.prep_synth(tmp_path, 2000, 3.0)
    assert meta["n_groups"] % td.GPB == 0
    words = np.fromfile(tmp_path / "vis.u32", np.uint32)
    bits = ((words[:, None] >> np.arange(32)) & 1).astype(bool).ravel()[:2000]
    s = td.synth(2000, 3.0)
    vis = t2.forward_lanes(s["params"], s["w2c"], s["K"], s["H"], s["W"])["visible"]
    assert np.array_equal(bits, vis)
    mp = np.fromfile(tmp_path / "meta.u32", np.uint32).reshape(-1, 128)
    assert mp[0, 64:].sum() == s["counts"][:64].sum() and mp[1, 0] == s["P"][64]


# Two rows of the #750 hero view (bicycle hero camera): row 0 is #953138, a Gaussian just past
# the near plane (z 0.2015) centred ~195,000 px off screen, whose dL/dconic (~1e8) cancels to
# grads ~1e-7 of its terms, so fp32 misses fp64 by ~1e5 relative; row 1 (#640834) is benign.
HERO_W2C = [[0.6356840663395591, 0.1589352827320079, -0.7554107119341765, 0.0027671256602394287],
            [-0.03804422066319288, 0.9838365806164537, 0.17498062725643193, -1.102439129716195],
            [0.7710112872559866, -0.08249338484885188, 0.6314558071472983, 3.8377925032777758],
            [0.0, 0.0, 0.0, 1.0]]
HERO_K = [[1162.8465213241284, 0.0, 618.5], [0.0, 1157.5291863156094, 410.99999999999994],
          [0.0, 0.0, 1.0]]
HERO_ROWS = {
    "means": [[-23.77781105041504, -21.425249099731445, 20.475194931030273],
              [-3.6363065242767334, -6.114758014678955, -2.1075148582458496]],
    "log_scales": [[-2.398000955581665, -3.972705125808716, -4.724484443664551],
                   [-0.5200343132019043, -2.3588762283325195, -0.6450792551040649]],
    "quats": [[1.0368211269378662, 0.1601843237876892, -0.3653588891029358, 0.0006356520461849868],
              [0.8616774082183838, 0.32618674635887146, -0.040493592619895935, -0.16532443463802338]],
    "opacity_logits": [7.530975818634033, 5.720972537994385],
    "sh": [[[1.994084358215332, 1.9625657796859741, 2.056088447570801]],
           [[1.7820647954940796, 1.7748156785964966, 1.7688229084014893]]]}
HERO_G2D = [[-1.3742380815529032e-06, 3.020265921804821e-06, -144248880.0, -149485200.0,
             -38727888.0, 0.007512468844652176, 0.0039655668660998344, 0.004284481517970562,
             0.0037393090315163136],
            [-5.9459043910692344e-08, 2.780670804725105e-08, -132886.609375, -1109902.625,
             -2318514.25, 0.0027354091871529818, 0.0009180696797557175, 0.0020090085454285145,
             0.005766487680375576]]


def test_fp32_ill_rows_flags_near_plane_off_screen_row(tmp_path):
    f = np.float32
    p = {k: np.asarray(v, f) for k, v in HERO_ROWS.items()}
    np.savez(tmp_path / "ref_in.npz", w2c=np.asarray(HERO_W2C, f), K=np.asarray(HERO_K, f),
             H=822, W=1237, **{f"p_{k}": v for k, v in p.items()})
    g2d = np.asarray(HERO_G2D, f)
    grads, _ = t2.projection_backward_sh0(p, g2d, np.asarray(HERO_W2C, f), np.asarray(HERO_K, f),
                                          822, 1237)
    ref = {"g2d": g2d, "grads": {**grads, "sh": grads["sh"][:, :1, :]}}
    ill, x = td.fp32_ill_rows(tmp_path, ref, 1e-2)
    assert ill.tolist() == [True, False]
    assert x["e32"]["means"][0] > 1e3 and x["e32"]["means"][1] <= 1e-2


# Hero row 1611162 (#770): fp32 misses fp64 by 0.37 on its means grad, but only relative to its
# own size (~2e-8); next to an ill row's ~1e5 grads a whole-tensor floor hid it.
MILD_ROW = {"means": [3.6506292819976807, 3.953773260116577, -9.661470413208008],
            "log_scales": [-5.190515995025635, -4.503379821777344, -2.525245428085327],
            "quats": [1.0947757959365845, 0.0070793358609080315, 0.4106677174568176,
                      0.01848435215651989],
            "opacity_logits": 2.1491668224334717,
            "sh": [[-0.5874078273773193, -0.5266046524047852, -1.0195577144622803]]}
MILD_G2D = [-1.5921415661562799e-12, 1.65134052265703e-11, -0.8908247947692871,
            -0.15829350054264069, -0.007031942717730999, 7.259725043340381e-10,
            7.955573977369568e-10, 9.360906494393362e-10, 9.672886935874203e-10]


def test_fp32_ill_rows_floor_from_kept_rows(tmp_path):
    f = np.float32
    p = {k: np.asarray([HERO_ROWS[k][0], MILD_ROW[k]], f) for k in HERO_ROWS}
    np.savez(tmp_path / "ref_in.npz", w2c=np.asarray(HERO_W2C, f), K=np.asarray(HERO_K, f),
             H=822, W=1237, **{f"p_{k}": v for k, v in p.items()})
    g2d = np.asarray([HERO_G2D[0], MILD_G2D], f)
    grads, _ = t2.projection_backward_sh0(p, g2d, np.asarray(HERO_W2C, f), np.asarray(HERO_K, f),
                                          822, 1237)
    ref = {"g2d": g2d, "grads": {**grads, "sh": grads["sh"][:, :1, :]}}
    ill, x = td.fp32_ill_rows(tmp_path, ref, 1e-3)
    assert td.row_err(ref["grads"]["means"], x["g64"]["means"])[1] < 1e-3  # whole-tensor floor
    assert ill.tolist() == [True, True]
    assert x["e32"]["means"][1] > 1e-2


def test_check_gates_well_rows_and_reports_ill_rows(tmp_path, monkeypatch):
    # A device that misses the reference only on fp32-ill rows passes (rows reported); one that
    # misses it on a well-conditioned row fails.
    td.prep_dump("benchmarks/train_oracle/t0_sh0_64x64_n50.npz", tmp_path / "in")
    _emulate(tmp_path / "in", tmp_path / "out")
    meta = json.loads((tmp_path / "in" / "meta.json").read_text())
    N, ng = meta["n"], meta["n_groups"]
    gp = tmp_path / "out" / "grads.f32"
    cols = td.from_pages(np.fromfile(gp, np.float32).reshape(ng, 1024), N, 14)
    real = td.fp32_ill_rows
    monkeypatch.setattr(td, "fp32_ill_rows", lambda i, r, b: (
        (lambda ill, x: (np.arange(N) == 3, x))(*real(i, r, b))))
    bad = cols.copy()
    bad[3, 0:3] *= -50.0
    _emulate_grads(tmp_path / "in", tmp_path / "out", bad)
    rep = td.check(tmp_path / "in", tmp_path / "out")
    assert rep["pass"] and rep["fp32_ill_rows"] == 1 and rep["ill"]["means"]["dev_vs_f64"] > 1
    assert rep["grads_all_rows"]["means"] > 1
    assert (tmp_path / "out" / "t2_rows.npz").exists()
    bad = cols.copy()
    bad[4, 0:3] *= -50.0
    _emulate_grads(tmp_path / "in", tmp_path / "out", bad)
    assert not td.check(tmp_path / "in", tmp_path / "out")["pass"]


def test_check_gates_adam_on_device_grads(tmp_path):
    # #770 hero: grads within 1e-3 of the reference flip bf16 moment roundings (m off by one
    # bf16 ulp, ~4e-3 relative); Adam is gated given the device's grads, so that passes. A
    # wrong Adam (moments off for the device's own grads) still fails.
    td.prep_dump("benchmarks/train_oracle/t0_sh0_64x64_n50.npz", tmp_path / "in")
    _emulate(tmp_path / "in", tmp_path / "out")
    meta = json.loads((tmp_path / "in" / "meta.json").read_text())
    N, ng = meta["n"], meta["n_groups"]
    cols = td.from_pages(np.fromfile(tmp_path / "out" / "grads.f32", np.float32)
                         .reshape(ng, 1024), N, 14)
    near = cols * np.float32(1 + 2e-3)
    _emulate_grads(tmp_path / "in", tmp_path / "out", near)
    rep = td.check(tmp_path / "in", tmp_path / "out")
    assert rep["pass"] and rep["worst_adam"] == 0.0 and rep["m_bits_equal"]
    assert max(v for k, v in rep["adam_e2e"].items() if k.startswith("m_")) > 1e-3
    mp = tmp_path / "out" / "m_out.u16"
    mb = np.fromfile(mp, np.uint16).reshape(ng, 1024)
    mb[0, 5] ^= 1 << 4
    mb.tofile(mp)
    assert not td.check(tmp_path / "in", tmp_path / "out")["pass"]
