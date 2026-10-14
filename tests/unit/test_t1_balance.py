"""#775 T1 load balance: LPT row order and the microblock split with partial slots merged in T2
must keep the device model's slots (bit for bit where one part sees a pair, fixed-order sum
within the T0 oracle tolerance otherwise)."""
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from gsplat import blend_backward_dev as dev  # noqa: E402
from gsplat import t1_balance as bal  # noqa: E402

DUMPS = sorted(str(p) for p in (ROOT / "benchmarks" / "train_oracle").glob("t0_sh*_64x64_n50.npz"))


def _load(which, tmp_path):
    if which == "synth":
        out = tmp_path / "s.npz"
        subprocess.run([sys.executable, str(ROOT / "docs/t685/synth.py"), str(out), "64", "96", "120", "3"],
                       check=True, env={**os.environ, "PYTHONPATH": str(ROOT)})
        path = str(out)
    else:
        path = next(p for p in DUMPS if which in os.path.basename(p))
    d, cfg = dev.load_dump(path)
    p = dev.prep(d["means2d"], d["conic"], d["opacity"], d["colors"], d["order"], d["valid"],
                 d["image"], d["dL_dimage"], cfg)
    return d, cfg, p


def _run(p, pl):
    q = bal.apply_plan(p, pl)
    return q, bal.merge_partials(dev.emulate(q), q["n_slots_base"], q["merge_dst"], q["merge_rank"])


def test_popcount_and_microblocks():
    x = np.array([0, 1, 0xFFFFFFFF, 0x80000001, 0x0F0F0F0F], np.uint32)
    assert bal.popcount32(x).tolist() == [0, 1, 32, 2, 16]
    for k in (1, 2, 4, 8, 16, 32):
        mb = bal.microblocks(k)
        assert len(mb) == k and sum(mb) == 0xFFFFFFFF and all(a & b == 0 for a in mb for b in mb if a != b)
    # k=4: one 16x16 face per part (vectors 8f..8f+7 are face f)
    lanes = dev.vector_lanes()
    for f, vm in enumerate(bal.microblocks(4)):
        V = [v for v in range(32) if vm >> v & 1]
        assert set((lanes[V] // 256).ravel().tolist()) == {f}


def test_contiguous_split_is_the_bench_rule():
    pairs = np.array([5, 1, 9, 2, 2, 7, 3, 1, 1, 8])
    beg = bal.contiguous_split(pairs, 4)
    assert beg[0] == 0 and beg[-1] == len(pairs) and np.all(np.diff(beg) >= 0)
    w, acc, c, want = pairs + 1, 0, 0, [0]           # blend_bwd_bench.cpp, transcribed
    for t in range(len(pairs)):
        if c + 1 >= 4:
            break
        acc += w[t]
        if acc * 4 >= w.sum() * (c + 1):
            c += 1
            want.append(t + 1)
    assert beg.tolist() == want + [len(pairs)] * (5 - len(want))


def test_lpt_bound_and_determinism():
    rng = np.random.default_rng(3)
    for n, m in ((50, 7), (1014, 110), (13, 20)):
        w = rng.pareto(1.5, n) + 0.1
        core = bal.lpt(w, m)
        assert np.array_equal(core, bal.lpt(w, m))
        load = np.bincount(core, w, m)
        assert load.max() <= 4 / 3 * max(w.sum() / m, w.max()) + 1e-9


@pytest.mark.parametrize("which", ["sh0", "sh3", "synth"])
def test_lpt_order_is_bit_identical(which, tmp_path):
    _, _, p = _load(which, tmp_path)
    base = dev.emulate(p)
    for n_cores in (1, 3, 64):
        pl = bal.plan(p["tiles"], p["recs"][:, 12], n_cores)
        q, got = _run(p, pl)
        assert q["n_slots"] == p["n_slots"] and q["merge_dst"].size == 0
        assert np.array_equal(got.view(np.uint32), base.view(np.uint32))
        assert np.array_equal(np.sort(q["tiles"][:, 0]), np.sort(p["tiles"][:, 0]))
        assert np.diff(pl["beg"]).sum() == len(q["tiles"])


@pytest.mark.parametrize("which", ["sh0", "sh3", "synth"])
@pytest.mark.parametrize("k", [2, 4, 8])
def test_split_merge(which, k, tmp_path):
    d, cfg, p = _load(which, tmp_path)
    base = dev.emulate(p)
    masks = p["recs"][:, 12]
    ks = np.full(len(p["tiles"]), k)
    sp = bal.split_rows(p["tiles"], masks, ks)
    # every pair keeps its whole mask across its parts and has exactly one real-slot part
    allp = np.concatenate(sp["pairs"])
    union = np.zeros(len(masks), np.uint32)
    np.bitwise_or.at(union, allp, np.concatenate(sp["masks"]))
    assert np.array_equal(union, masks)
    assert np.array_equal(np.bincount(allp[np.concatenate(sp["rank"]) == 0], minlength=len(masks)),
                          np.ones(len(masks), np.int64))
    n_parts = np.bincount(allp, minlength=len(masks))
    assert n_parts.max() > 1, "the split must produce multi-part pairs to test the merge"

    pl = bal.plan(p["tiles"], masks, 3)
    pl.update(split=sp, perm=np.arange(len(sp["src"]))[::-1].copy())     # any row order
    q, got = _run(p, pl)
    assert q["n_slots"] == p["n_slots"] + int((n_parts - 1).sum())

    # 1. fixed-order merge, independently: partials of each slot by ascending part index
    ext = dev.emulate(q)
    slot_of = q["recs"][:, 13].astype(np.int64)
    want = ext[:p["n_slots"]].copy()
    part_of_rec = np.repeat(sp["part"][pl["perm"]], [len(sp["pairs"][i]) for i in pl["perm"]])
    real = np.full(len(q["recs"]), -1, np.int64)
    extra = slot_of >= p["n_slots"]
    real[extra] = q["merge_dst"][slot_of[extra] - p["n_slots"]]
    for s in np.unique(real[extra]):
        for r in sorted(np.flatnonzero(real == s), key=lambda r: part_of_rec[r]):
            want[s] = want[s] + ext[slot_of[r]]
    assert np.array_equal(got.view(np.uint32), want.view(np.uint32))

    # 2. pairs seen by one part are bit-identical to the unsplit run
    one = p["recs"][n_parts == 1, 13].astype(np.int64)          # pair index -> slot
    assert np.array_equal(got[one].view(np.uint32), base[one].view(np.uint32))

    # 3. row order does not change the merged result (merge order is by part, not by core)
    pl2 = dict(pl, perm=np.arange(len(sp["src"])))
    assert np.array_equal(_run(p, pl2)[1].view(np.uint32), got.view(np.uint32))

    # 4. T0 oracle, same bar as the unsplit device model (test_blend_backward_dev)
    bar = 1e-2 if which == "synth" else 1e-4
    assert dev.compare_slots(got, dev.ref_slots(d, cfg))["rel_err"] < bar
    assert np.abs(got - base).max() <= 1e-6 * np.abs(base).max()       # rounding of the merge only


def test_plan_split_balances_a_heavy_tile(tmp_path):
    """One tile carrying most bodies: LPT alone is stuck on it, the split spreads it."""
    _, _, p = _load("synth", tmp_path)
    masks = p["recs"][:, 12]
    n = 8
    whole = bal.plan(p["tiles"], masks, n)
    pl = bal.plan(p["tiles"], masks, n, split=True)
    assert (pl["ks"] > 1).any()
    assert pl["core_cycles"].max() < whole["core_cycles"].max()
    assert pl["core_bodies"].sum() == whole["core_bodies"].sum()     # a split adds no bodies
    q = bal.apply_plan(p, pl)
    out = tmp_path / "prep"
    bal.write_plan(q, str(out))
    assert np.fromfile(out / "cores.u32", np.uint32).tolist() == pl["beg"].tolist()
    assert np.fromfile(out / "merge.i64", np.int64).size == 2 * pl["n_extra"]
