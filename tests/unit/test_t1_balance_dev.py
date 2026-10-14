"""#785: the planned prep dirs the device bench runs (gsplat.t1_balance_dev plan/merge) are the
#775 plan (apply_plan + write_plan) read back from and written to prep-dir files."""
import pathlib
import subprocess
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from gsplat import blend_backward_dev as dev  # noqa: E402
from gsplat import t1_balance as bal  # noqa: E402
from gsplat import t1_balance_dev as bd  # noqa: E402

FILES = ("tiles.u32", "recs.u32", "planes.f32", "ramps.f32", "cores.u32", "merge.i64")


def _synth(tmp_path):
    out = tmp_path / "s.npz"
    subprocess.run([sys.executable, str(ROOT / "docs/t685/synth.py"), str(out), "64", "96", "120", "3"],
                   check=True, env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"})
    d, cfg = dev.load_dump(str(out))
    p = dev.prep(d["means2d"], d["conic"], d["opacity"], d["colors"], d["order"], d["valid"],
                 d["image"], d["dL_dimage"], cfg)
    dev.write(p, str(tmp_path / "prep"))
    return p


@pytest.mark.parametrize("split", [False, True])
def test_plan_cli_matches_write_plan(split, tmp_path):
    p = _synth(tmp_path)
    n = 8
    args = ["plan", str(tmp_path / "prep"), str(tmp_path / "cli"), "--cores", str(n)]
    assert bd.main(args + (["--split", "--kmax", "8"] if split else [])) == 0
    pl = bal.plan(p["tiles"], p["recs"][:, 12], n, split, 8)
    q = bal.apply_plan(p, pl)
    bal.write_plan(q, str(tmp_path / "ref"))
    for f in FILES:
        assert (tmp_path / "cli" / f).read_bytes() == (tmp_path / "ref" / f).read_bytes(), f
    got = bd.read_prep(str(tmp_path / "cli"))
    assert got["n_slots"] == q["n_slots"] and got["meta"]["n_slots_base"] == p["n_slots"]
    assert got["meta"]["n_tiles"] == len(q["tiles"]) and got["meta"]["n_recs"] == len(q["recs"])
    assert np.fromfile(tmp_path / "cli" / "cores.u32", np.uint32)[-1] == len(q["tiles"])
    if split:
        assert (pl["ks"] > 1).any() and q["n_slots"] > p["n_slots"]
    # merge: the device's slots.f32 (16 floats per slot) back to the base slot count
    slots = np.zeros((q["n_slots"], 16), np.float32)
    slots[:, :9] = dev.emulate(q)
    slots.tofile(tmp_path / "slots.f32")
    assert bd.main(["merge", str(tmp_path / "cli"), str(tmp_path / "slots.f32"), str(tmp_path / "m.f32")]) == 0
    merged = np.fromfile(tmp_path / "m.f32", np.float32).reshape(-1, 16)
    want = bal.merge_partials(slots, p["n_slots"], q["merge_dst"], q["merge_rank"])
    assert merged.shape == (p["n_slots"], 16) and np.array_equal(merged, want)
    if not split:   # LPT alone: the device model's slots are bit-identical to the unplanned prep
        assert np.array_equal(merged[:, :9].view(np.uint32), dev.emulate(p).view(np.uint32))
