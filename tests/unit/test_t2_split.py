"""T2 heavy-row split (#774): bit-equal to the rank-plane reference, plan covers every group
once, plane balance on the hero view's counts, and the T0 oracle still passes with it."""
from __future__ import annotations

import os
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gsplat import t2_param_backward as t2  # noqa: E402
from gsplat import t2_split as sp  # noqa: E402

HERO_COUNTS = ROOT / "tests" / "fixtures" / "hero" / "t2_counts.npz"
DUMPS = [ROOT / "benchmarks" / "train_oracle" / "t0_sh0_64x64_n50.npz"]


def _bits(a):
    return np.ascontiguousarray(a, np.float32).view(np.uint32)


def _case(counts, seed=0):
    """Slots that make the sum order visible: wide magnitudes, cancellation, signed zeros."""
    rng = np.random.default_rng(seed)
    counts = np.asarray(counts, np.int64)
    P = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
    ns = int(counts.sum())
    s = (rng.normal(size=(ns, 9)) * 10.0 ** rng.integers(-6, 6, (ns, 9))).astype(np.float32)
    s[rng.random((ns, 9)) < 0.05] = -0.0
    s[rng.random((ns, 9)) < 0.02] = 0.0
    return s, P, counts


def _heavy_tail_counts(n, seed=0):
    rng = np.random.default_rng(seed)
    c = rng.poisson(1.5, n)
    h = rng.choice(n, max(1, n // 200), replace=False)
    c[h] = rng.integers(17, 400, h.size)
    c[rng.choice(n, 3, replace=False)] = [1014, 17, 16]
    return c


@pytest.mark.parametrize("n,kcap,cores", [(5000, 16, 110), (4999, 4, 7), (777, 1, 3),
                                          (64, 16, 1), (130, 0, 200)])
def test_split_bit_equal_synthetic(n, kcap, cores):
    s, P, c = _case(_heavy_tail_counts(n, n), seed=n)
    p = sp.plan(c, kcap=kcap, n_cores=cores)
    ref = t2.segment_reduce(s, P, c)
    got = sp.segment_reduce_split(s, P, c, p)
    assert np.array_equal(_bits(ref), _bits(got))
    assert p.heavy_rows.size == int((c > kcap).sum())


def test_bit_check_catches_other_sum_order():
    """The data above does detect a changed summation order (heavy rows summed j descending)."""
    s, P, c = _case(_heavy_tail_counts(2000, 1), seed=1)
    ref = t2.segment_reduce(s, P, c)
    other = ref.copy()
    for r in sp.heavy_order(c):
        acc = np.zeros(9, np.float32)
        for j in range(c[r] - 1, -1, -1):
            acc += s[P[r] + j]
        other[r] = acc
    assert not np.array_equal(_bits(ref), _bits(other))


def test_split_bit_equal_on_hero_counts():
    """The hero view's own slot counts with random slots: every 64-row group holding a heavy
    row (k > 16, up to 1014) plus every 16th other group, in row order (the full 1.81M rows
    take minutes; T2_HERO_REF_IN runs the real dump)."""
    c = np.load(HERO_COUNTS)["counts"].astype(np.int64)
    ng = -(-c.size // 64)
    gk = np.concatenate([c, np.zeros(ng * 64 - c.size, np.int64)]).reshape(ng, 64).max(1)
    keep = (gk > 16) | (np.arange(ng) % 16 == 0)
    rows = (np.flatnonzero(keep)[:, None] * 64 + np.arange(64)).ravel()
    s, P, c = _case(c[rows[rows < c.size]], seed=7)
    assert int(c.max()) == 1014 and int((c > 16).sum()) == 2975
    p = sp.plan(c)
    assert np.array_equal(_bits(t2.segment_reduce(s, P, c)),
                          _bits(sp.segment_reduce_split(s, P, c, p)))


@pytest.mark.skipif(not os.environ.get("T2_HERO_REF_IN"),
                    reason="set T2_HERO_REF_IN to a t2_dev prep-dump ref_in.npz (hero: 494 MB)")
def test_split_bit_equal_on_hero_dump():
    r = np.load(os.environ["T2_HERO_REF_IN"])
    s, P, c = r["slots"], r["P"], r["counts"]
    assert np.array_equal(_bits(t2.segment_reduce(s, P, c)),
                          _bits(sp.segment_reduce_split(s, P, c)))


def _hero_plan(**kw):
    d = np.load(HERO_COUNTS)
    vis = np.unpackbits(d["visible_bits"])[:int(d["n"])].astype(bool)
    return d["counts"].astype(np.int64), vis, sp.plan(d["counts"], vis, **kw)


def test_hero_plane_balance():
    c, vis, p = _hero_plan()
    sp.check_plan(p)
    m = p.summary()
    assert m["n_heavy_rows"] == int((c > 16).sum()) == 2975
    assert (p.group_planes <= 16).all()
    # Light + heavy planes are about a fifth of today's padded planes.
    base = sp.contiguous_baseline(c, vis)
    assert m["total_planes"] * 4 < base["total_planes"]
    # The busiest core is within one light group (16 planes) of a perfect split, and far
    # under today's busiest core.
    lower = max(m["total_planes"] / p.n_cores, m["max_heavy_group_planes"])
    assert m["core_max"] <= lower + 16
    assert m["mean_over_max"] >= 0.98
    assert m["core_max"] * 10 < base["core_max"]


def test_plan_covers_every_group_once_and_ranges_are_contiguous():
    c = _heavy_tail_counts(10_000, 3)
    vis = np.random.default_rng(3).random(c.size) < 0.7
    for ov in (0.0, 8.0):
        p = sp.plan(c, vis, n_cores=13, group_overhead=ov)
        sp.check_plan(p)
        assert (np.diff(p.core_bounds) >= 0).all()
        cs = np.concatenate([[0.0], np.cumsum(p.group_planes + ov * p.active)])
        light = cs[p.core_bounds[1:]] - cs[p.core_bounds[:-1]]
        heavy = np.array([p.hgroup_planes[h].sum() for h in p.core_hgroups])
        assert np.allclose(p.core_planes, light + heavy)
        assert np.isclose(p.core_planes.sum(), cs[-1] + p.hgroup_planes.sum())


def test_heavy_order_and_group_planes_deterministic():
    c = np.array([3, 20, 0, 20, 100, 17, 5])
    assert sp.heavy_order(c, 16).tolist() == [4, 1, 3, 5]
    p = sp.plan(c, kcap=16, n_cores=2, group=2)
    assert p.hgroup_planes.tolist() == [100, 20]  # first lane of each 2-lane heavy group
    # groups (3,20->0) (0,20->0) (100->0,17->0) (5): max light count, at least 1 when active
    assert p.group_planes.tolist() == [3, 1, 1, 5]


@pytest.mark.parametrize("dump", DUMPS, ids=lambda d: d.name)
def test_t0_oracle_passes_with_split(dump, monkeypatch):
    """The whole T2 step on the T0 oracle dump with the split reduce (kcap 1, so most rows take
    the heavy path) passes the same bars, and its slot sums are bit-equal."""
    d = np.load(dump)
    bw = t2.oracle_slots(d)
    assert int(bw["counts"].max()) > 1
    ref = t2.segment_reduce(bw["slots"], bw["P"], bw["counts"])
    got = sp.segment_reduce_split(bw["slots"], bw["P"], bw["counts"], kcap=1)
    assert np.array_equal(_bits(ref), _bits(got))
    monkeypatch.setattr(t2, "segment_reduce",
                        lambda s, P, c: sp.segment_reduce_split(s, P, c, kcap=1))
    rep = t2.check(d)
    assert rep["pass"], rep


# ---- #784: the on-device split (train_step T2_SPLIT=1), mirrored by device_plan / device_split_sums

def _dev_inputs(counts, active=None):
    c = np.asarray(counts, np.int64)
    ng = -(-c.size // 64)
    cp = np.concatenate([c, np.zeros(ng * 64 - c.size, np.int64)])
    P = np.concatenate([[0], np.cumsum(cp)[:-1]])
    meta = np.concatenate([P.reshape(ng, 64), cp.reshape(ng, 64)], 1).astype(np.uint32)
    act = np.ones(ng, bool) if active is None else np.asarray(active, bool)
    vis = np.zeros((-(-ng // 16), 32), np.uint32)
    g = np.flatnonzero(act)
    vis[g // 16, 2 * (g % 16)] = 1
    return meta, vis, act


def _assert_bits_equal_but_zero_sign(ref, got):
    """Bit-equal, except that the device copies plane 0 (as today's T2 does): a row whose sum
    is a lone -0.0 slot keeps -0.0 where the reference's +0.0 + -0.0 gives +0.0."""
    rb, gb = _bits(ref), _bits(got)
    diff = rb != gb
    assert ((ref[diff] == 0) & (got[diff] == 0)).all()
    return int(diff.sum())


@pytest.mark.parametrize("n,kcap,hl,cores", [(5000, 16, 16, 110), (4999, 8, 16, 7), (777, 1, 8, 3),
                                             (3000, 16, 64, 5), (130, 0, 4, 200)])
def test_device_split_sums_bit_equal(n, kcap, hl, cores):
    s, P, c = _case(_heavy_tail_counts(n, n), seed=n)
    act = np.random.default_rng(n).random(-(-n // 64)) < 0.8
    meta, vis, act = _dev_inputs(c, act)
    got = sp.device_split_sums(s, meta, vis, kcap=kcap, hl=hl, n_cores=cores)[:n]
    ref = t2.segment_reduce(s, P, c)
    live = np.repeat(act, 64)[:n]
    _assert_bits_equal_but_zero_sign(ref[live], got[live])
    assert not got[~live].any()
    # without -0.0 slots the sums are bit-equal outright
    s2 = np.where(s == 0, np.float32(0), s)
    got2 = sp.device_split_sums(s2, meta, vis, kcap=kcap, hl=hl, n_cores=cores)[:n]
    assert np.array_equal(_bits(t2.segment_reduce(s2, P, c)[live]), _bits(got2[live]))


def test_device_split_catches_wrong_heavy_merge():
    """The bit test sees a heavy sum taken from the wrong record (h off by one)."""
    s, P, c = _case(_heavy_tail_counts(3000, 5), seed=5)
    meta, vis, _ = _dev_inputs(c)
    p = sp.device_plan(meta, vis)
    h = p["heavy"]
    bad = meta.copy()
    bad[h[0] // 64, 64 + h[0] % 64], bad[h[1] // 64, 64 + h[1] % 64] = c[h[1]], c[h[0]]
    ok = sp.device_split_sums(s, meta, vis)[:c.size]
    assert not np.array_equal(_bits(ok), _bits(sp.device_split_sums(s, bad, vis)[:c.size]))


def test_device_plan_layout_small():
    c = np.zeros(128, np.int64)
    c[[3, 70, 71, 5]] = [20, 100, 17, 2]
    meta, vis, _ = _dev_inputs(c)
    p = sp.device_plan(meta, vis, kcap=16, hl=2, ovh=2, n_cores=2)
    assert p["heavy"].tolist() == [70, 3, 71]
    assert p["htab"][:, 64:66].tolist() == [[100, 20], [17, 0]]
    assert p["htab"][0, :2].tolist() == [int(meta[1, 6]), int(meta[0, 3])]
    assert int(p["metab"][1, 64 + 6]) == sp.HFLAG | 0 and int(p["metab"][0, 64 + 3]) == sp.HFLAG | 1
    assert p["core_h"] == [[0], [1]] and p["a_max_planes"] == 100
    assert p["bounds"].tolist() == [0, 1, 2]  # weights 2+2, 1+2: one group per core


def test_device_plan_hero_counts():
    """Hero view counts (all groups active): phase A is bounded by the 1014-slot row, phase B
    by about planes/110."""
    c = np.load(HERO_COUNTS)["counts"].astype(np.int64)
    meta, vis, _ = _dev_inputs(c)
    p = sp.device_plan(meta, vis, kcap=16, hl=16, ovh=2, n_cores=110)
    assert p["heavy"].size == 2975 and p["a_max_planes"] == 1014
    ng = meta.shape[0]
    assert p["b_max_weight"] <= (p["light_planes"] + 2 * ng) / 110 + 18
    assert (np.diff(p["bounds"]) >= 0).all() and p["bounds"][-1] == ng


def test_train_step_defaults_split_on_kcap8():
    """#792: the trainer runs T2 as the heavy-row split by default, at kcap 8."""
    src = (ROOT / "render" / "train" / "train_step.cpp").read_text()
    assert 'envu("T2_SPLIT", 1)' in src
    assert 'envu("T2_KCAP", 8)' in src


def test_split_reader_feeds_sh_aux():
    """#798: SH mode keeps the split on: the split reader feeds the t2sh aux page per group (the
    compute pops CB_AUX after the planes in every T2_SH group), from accessor 8 / runtime arg 9."""
    src = (ROOT / "render" / "train" / "kernels" / "t2g_reader.cpp").read_text()
    split = src.split("#elif defined(T2_SPLIT)", 1)[1].split("\n#else", 1)[0]
    assert "TensorAccessorArgs<hsa.next_compile_time_args_offset()>" in split
    assert "get_arg_val<uint32_t>(9)" in split
    assert split.count("cb_reserve_back(CB_AUX, 1)") == 1 and split.count("cb_push_back(CB_AUX, 1)") == 1
    host = (ROOT / "render" / "train" / "train_step.cpp").read_text()
    assert "6 + int(split) + int(sh)" in host and host.count("if (sh) ra.push_back(uint32_t(b_x->address()));") == 2
