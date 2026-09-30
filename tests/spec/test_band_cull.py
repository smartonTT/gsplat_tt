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

band_keep_f32 is the fp32 model of microblock_cull_compute.cpp.
"""
import importlib.util
from pathlib import Path

import numpy as np

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
