"""Model test for task #59: the band-extent microblock cull.

The SFPU cull used to evaluate a box-constrained Mahalanobis minimum per
(gaussian, microblock): one gaussian per SFPU vector, 32 microblocks in the
lanes, ~140 SFPU + ~100 scalar instructions per pair. The band form puts one
gaussian in each lane and walks the 8 microblock rows: per row it clamps the
ellipse's right/left-most point into the row's pixel-centre span, which gives
the ellipse's exact x-extent in that row, and keeps the 4 column blocks that
extent meets (compared on squares, no sqrt). Geometrically that is the same
"ellipse meets the pixel-centre box" test as boxmin_m2, so the mask must

1. cover every microblock with a live pixel (alpha >= floor at a pixel centre),
   even when the SFPU log reads ln low and the blend exp reads alpha high, and
2. match the box-min mask except for pairs sitting on the thr boundary.

band_keep_f32 is the fp32 model of microblock_band_cull_compute.cpp.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

_CHK = Path(__file__).resolve().parents[2] / "opt" / "golden-check" / "cull_mask_check.py"
_spec = importlib.util.spec_from_file_location("cull_mask_check", _CHK)
cmc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cmc)

FLOOR = 1.0 / 255.0
EXP_REL_ERR = 1.73e-3   # blend exp_21f fit, max relative over-read of alpha
LOG_UNDER = 0.0035      # cull SFPU log, max underestimate of ln
M = np.arange(32)
PC_OX = ((M & 3) * 8 + 0.5).astype(float)
PC_OY = ((M >> 2) * 4 + 0.5).astype(float)


def _conic(sig_major, sig_minor, theta):
    c, s = np.cos(theta), np.sin(theta)
    r = np.array([[c, -s], [s, c]])
    ci = np.linalg.inv(r @ np.diag([sig_major**2, sig_minor**2]) @ r.T)
    return ci[0, 0], ci[0, 1], ci[1, 1]


def _random_pairs(n, seed):
    """Slab-record fields (A,B,C, tile-local mean, UNORM16 q) for n random splats."""
    rng = np.random.default_rng(seed)
    ci = np.array([_conic(10 ** rng.uniform(-1, 2), 10 ** rng.uniform(-1, 2),
                          rng.uniform(0, np.pi)) for _ in range(n)])
    ca, cb, cc = ci[:, 0], ci[:, 1], ci[:, 2]
    # Record words carry the pre-folded conic A=-ca/2, B=-cb, C=-cc/2 (fp32).
    A = (-0.5 * ca).astype(np.float32)
    B = (-cb).astype(np.float32)
    C = (-0.5 * cc).astype(np.float32)
    mx = rng.uniform(-40, 72, n).astype(np.float32)
    my = rng.uniform(-40, 72, n).astype(np.float32)
    q = np.round(10 ** rng.uniform(-2.5, 0, n) * 65535).astype(np.int64)
    return A, B, C, mx, my, q


def _exact(A, B, C, mx, my, q):
    ca = -2.0 * A.astype(float)
    cb = -B.astype(float)
    cc = -2.0 * C.astype(float)
    return ca, cb, cc, mx.astype(float), my.astype(float), q / 65535.0


def test_band_mask_covers_every_live_pixel():
    A, B, C, mx, my, q = _random_pairs(3000, seed=5)
    keep = cmc.band_keep_f32(A, B, C, mx, my, q, FLOOR, log_under=LOG_UNDER)
    ca, cb, cc, mxd, myd, op = _exact(A, B, C, mx, my, q)
    yy, xx = np.mgrid[0:32, 0:32] + 0.5
    mb = ((yy // 4) * 4 + (xx // 8)).astype(int).ravel()
    for i in range(len(A)):
        dx, dy = xx.ravel() - mxd[i], yy.ravel() - myd[i]
        a = op[i] * np.exp(-0.5 * (ca[i] * dx * dx + 2 * cb[i] * dx * dy + cc[i] * dy * dy))
        live = np.zeros(32, bool)
        np.logical_or.at(live, mb, a * (1 + EXP_REL_ERR) >= FLOOR)
        assert not (live & ~keep[i]).any(), i


def test_band_mask_matches_boxmin_mask():
    A, B, C, mx, my, q = _random_pairs(3000, seed=6)
    keep = cmc.band_keep_f32(A, B, C, mx, my, q, FLOOR)
    ca, cb, cc, mxd, myd, op = _exact(A, B, C, mx, my, q)
    thr = 2 * np.log(op / FLOOR) + cmc.THR_MARGIN
    ulo = PC_OX[None, :] - mxd[:, None]
    vlo = PC_OY[None, :] - myd[:, None]
    m2 = cmc.boxmin_m2(ca[:, None], cb[:, None], cc[:, None], ulo, ulo + cmc.BOX_W,
                       vlo, vlo + cmc.BOX_H)
    # Same geometric test: any disagreement must sit on the thr boundary.
    eps = 1e-4 * np.maximum(1.0, np.abs(thr))[:, None]
    assert not (keep & ~(m2 <= thr[:, None] + eps)).any()
    assert not ((m2 <= thr[:, None] - eps) & ~keep).any()
    assert keep.sum() > 0.05 * keep.size  # the sample actually exercises the cull


def test_band_mask_drops_invisible_splats():
    A, B, C, mx, my, q = _random_pairs(200, seed=7)
    q = np.zeros_like(q)  # opacity 0: thr = -inf, nothing may be kept
    assert not cmc.band_keep_f32(A, B, C, mx, my, q, FLOOR).any()
    q = np.full_like(q, 2)  # op*255 < 1: below the floor everywhere
    assert not cmc.band_keep_f32(A, B, C, mx, my, q, FLOOR, margin=0.0).any()


def test_coeff_tile_and_mask_layout_match_sfpu_words():
    """Pure-index mirror of reader fill_coeff_tile and writer mask decode.

    SFPU layout: vector V lane l lives at tile word 64*(V>>1) + (V&1) + 2*l.
    Record i is group g=i>>5, lane l=i&31; its coeff field f (A,B,C,opq,mx,my)
    goes to vector V=6g+f, and its mask halves come back in vectors 2g (bits
    0-15) and 2g+1 (bits 16-31) of the keep tile.
    """
    def sfpu_word(v, lane):
        return 64 * (v >> 1) + (v & 1) + 2 * lane

    # reader_tile_l1_cull.cpp fill_coeff_tile: dst = 192*(i>>5) + 2*(i&31),
    # fields at dst+{0,1,64,65,128,129}.
    offs = (0, 1, 64, 65, 128, 129)
    seen = set()
    for i in range(128):  # COEFF_BATCH
        dst = 192 * (i >> 5) + 2 * (i & 31)
        for f, off in enumerate(offs):
            w = dst + off
            assert w == sfpu_word(6 * (i >> 5) + f, i & 31), (i, f)
            seen.add(w)
    assert len(seen) == 128 * 6 and max(seen) < 1024  # no overlap, fits a tile

    # writer_tile_l1_mask.cpp: o = 64*(i>>5) + 2*(i&31); lo = t[o], hi = t[o+1].
    rng = np.random.default_rng(3)
    masks = rng.integers(0, 2**32, 128, dtype=np.uint64)
    tile = np.zeros(1024, np.uint64)
    for i in range(128):  # compute side: fp32 2^23 + 16-bit half per vector
        tile[sfpu_word(2 * (i >> 5), i & 31)] = 0x4B000000 | (masks[i] & 0xFFFF)
        tile[sfpu_word(2 * (i >> 5) + 1, i & 31)] = 0x4B000000 | (masks[i] >> 16)
    for i in range(128):
        o = 64 * (i >> 5) + 2 * (i & 31)
        lo, hi = int(tile[o]), int(tile[o + 1])
        assert ((lo & 0xFFFF) | ((hi << 16) & 0xFFFFFFFF)) == int(masks[i]), i


# det = ci_a*ci_c - B^2 is formed in fp32 in the kernel. For a needle splat
# (sigma_minor^2 ~ 0.3, the anti-alias floor) at an oblique angle, ci_a*ci_c
# and B^2 are both ~1/0.3^2 while det ~ 1/(0.3*sigma_major^2): a small
# difference of O(1) terms, so its relative error grows ~ sigma_major^2 * 2^-24
# / 0.3. At large sigma_major that eats the ellipse extent and tip microblocks
# get culled (model: clean to sigma_major 500 px, misses from ~600 px when the
# splat is not radius-capped). The default max_radius = min(H, W)/2 caps each
# splat's projected x/y radius (sqrt(thr * cov_xx), sqrt(thr * cov_yy)), which
# drops the long needles before the cull ever sees them: with the bicycle cap
# the model shows no false culls at 300-700 px, and at >= 1000 px no needle
# that reaches the tile survives the cap at all. Uncapped (max_radius<0)
# needles of sigma_major >= 1000 px are the known limit: expected fail.
# Kernels are unchanged.
BICYCLE_HW = (822, 1237)  # 4x-downsampled bicycle; cap = min(H, W)/2


def _needle_tip_false_culls(sig_major, max_radius):
    """Needle splats with the tile at the ellipse tip (tt-project run 174 model).

    Returns (live microblocks, falsely culled microblocks) over splats whose
    projected radius passes max_radius (None = no cap).
    """
    rng = np.random.default_rng(int(sig_major))
    n = 1000
    th = rng.uniform(0, np.pi, n)
    smin = np.sqrt(0.3 + rng.uniform(0, 0.2, n))
    cth, sth = np.cos(th), np.sin(th)
    q = np.round(10 ** rng.uniform(np.log10(1.2 / 255), 0, n) * 65535).astype(np.int64)
    thr = 2 * np.log((q / 65535.0) / FLOOR)
    cov_xx = sig_major**2 * cth**2 + smin**2 * sth**2
    cov_yy = sig_major**2 * sth**2 + smin**2 * cth**2
    ok = np.ones(n, bool) if max_radius is None else (
        np.maximum(np.sqrt(thr * cov_xx), np.sqrt(thr * cov_yy)) <= max_radius)
    l1, l2 = 1 / smin**2, 1 / sig_major**2
    ca = l1 * sth**2 + l2 * cth**2
    cc = l1 * cth**2 + l2 * sth**2
    cb = (l2 - l1) * cth * sth
    A = (-0.5 * ca).astype(np.float32)
    B = (-cb).astype(np.float32)
    C = (-0.5 * cc).astype(np.float32)
    dist = np.sqrt(thr) * sig_major * rng.uniform(0.9, 1.02, n)
    mx = (16 - dist * cth).astype(np.float32)
    my = (16 - dist * sth).astype(np.float32)
    keep = cmc.band_keep_f32(A, B, C, mx, my, q, FLOOR)
    ca, cb, cc, mxd, myd, op = _exact(A, B, C, mx, my, q)
    yy, xx = np.mgrid[0:32, 0:32] + 0.5
    mb = ((yy // 4) * 4 + (xx // 8)).astype(int).ravel()
    n_live = n_bad = 0
    for i in np.nonzero(ok)[0]:
        dx, dy = xx.ravel() - mxd[i], yy.ravel() - myd[i]
        a = op[i] * np.exp(-0.5 * (ca[i] * dx * dx + 2 * cb[i] * dx * dy + cc[i] * dy * dy))
        live = np.zeros(32, bool)
        np.logical_or.at(live, mb, a >= FLOOR)
        n_live += live.sum()
        n_bad += (live & ~keep[i]).sum()
    return n_live, n_bad


@pytest.mark.parametrize("sig_major", [300.0, 500.0, 600.0, 700.0, 1000.0, 2000.0])
def test_needle_tip_no_false_cull_default_cap(sig_major):
    n_live, n_bad = _needle_tip_false_culls(sig_major, min(BICYCLE_HW) / 2)
    assert n_bad == 0
    if sig_major < 1000:
        assert n_live > 0  # the tip really reaches the tile
    else:
        assert n_live == 0  # cap removes every tip-reaching needle


@pytest.mark.parametrize("sig_major", [
    300.0, 500.0,
    *(pytest.param(s, marks=pytest.mark.xfail(
        reason="fp32 det cancellation limit (det=ci_a*ci_c-B^2)", strict=True))
      for s in (1000.0, 2000.0)),
])
def test_needle_tip_no_false_cull_uncapped(sig_major):
    n_live, n_bad = _needle_tip_false_culls(sig_major, None)
    assert n_live > 0
    assert n_bad == 0
