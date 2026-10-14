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
