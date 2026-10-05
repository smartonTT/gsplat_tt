"""Model test for task #41: the microblock keep test is exact, and a per-pixel
contribution floor is what makes contrib_floor 1/255 seam-free.

The device blend evaluates a gaussian on every pixel of each 8x4 microblock its
mask keeps; the mask keeps a block when ANY point of the block reaches the floor
(box-min Mahalanobis). Without a per-pixel floor, the sub-floor tail of a faint
gaussian is therefore kept in some blocks and dropped in the next, which is the
seam task #28 saw on wheel spokes. With alpha < floor -> 0 per pixel (the GPU
3DGS rule, BLEND_PIXEL_FLOOR), the result no longer depends on the mask.
"""
import importlib.util
from pathlib import Path

import numpy as np

_CHK = Path(__file__).resolve().parents[2] / "opt" / "golden-check" / "cull_mask_check.py"
_spec = importlib.util.spec_from_file_location("cull_mask_check", _CHK)
cmc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cmc)

FLOOR = 1.0 / 255.0
OX = np.array([(m & 3) * 8 for m in range(32)], float)
OY = np.array([(m >> 2) * 4 for m in range(32)], float)


def _conic(sig_major, sig_minor, theta):
    c, s = np.cos(theta), np.sin(theta)
    r = np.array([[c, -s], [s, c]])
    ci = np.linalg.inv(r @ np.diag([sig_major**2, sig_minor**2]) @ r.T)
    return ci[0, 0], ci[0, 1], ci[1, 1]


def test_boxmin_is_exact_vs_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(300):
        ca, cb, cc = _conic(10 ** rng.uniform(-1.3, 2), 10 ** rng.uniform(-1.3, 2),
                            rng.uniform(0, np.pi))
        ulo, vlo = rng.uniform(-20, 20, 2)
        got = cmc.boxmin_m2(ca, cb, cc, ulo, ulo + 8, vlo, vlo + 4)
        uu, vv = np.meshgrid(np.linspace(ulo, ulo + 8, 161), np.linspace(vlo, vlo + 4, 81))
        brute = (ca * uu * uu + 2 * cb * uu * vv + cc * vv * vv).min()
        assert got <= brute + 1e-9 * max(1.0, brute)


def _render(gs, pixel_floor):
    """Front-to-back blend of one 32x32 tile, device mask rule, optional pixel floor."""
    yy, xx = np.mgrid[0:32, 0:32] + 0.5
    mb = ((yy // 4) * 4 + (xx // 8)).astype(int)
    rgb, t = np.zeros((32, 32)), np.ones((32, 32))
    for (mx, my, op, ca, cb, cc, col) in gs:
        m2 = cmc.boxmin_m2(ca, cb, cc, OX - mx, OX - mx + 8, OY - my, OY - my + 4)
        keep = (op * np.exp(-0.5 * m2) >= FLOOR)[mb]
        dx, dy = xx - mx, yy - my
        a = np.minimum(op * np.exp(-0.5 * (ca * dx * dx + 2 * cb * dx * dy + cc * dy * dy)), 0.99)
        a = a * keep
        if pixel_floor:
            a = np.where(a >= FLOOR, a, 0.0)
        rgb += a * t * col
        t *= 1 - a
    return rgb * 255


def _gpu_rule(gs):
    yy, xx = np.mgrid[0:32, 0:32] + 0.5
    rgb, t = np.zeros((32, 32)), np.ones((32, 32))
    for (mx, my, op, ca, cb, cc, col) in gs:
        dx, dy = xx - mx, yy - my
        a = np.minimum(op * np.exp(-0.5 * (ca * dx * dx + 2 * cb * dx * dy + cc * dy * dy)), 0.99)
        a = np.where(a >= FLOOR, a, 0.0)
        rgb += a * t * col
        t *= 1 - a
    return rgb * 255


def _haze_scene():
    # Many faint, bright, wide gaussians (alpha ~0.004-0.006 at their centre,
    # i.e. mostly just below/above the floor) over a thin opaque spoke.
    rng = np.random.default_rng(1)
    gs = []
    for _ in range(400):
        ca, cb, cc = _conic(rng.uniform(3, 8), rng.uniform(3, 8), rng.uniform(0, np.pi))
        gs.append((rng.uniform(0, 32), rng.uniform(0, 32), rng.uniform(0.004, 0.006),
                   ca, cb, cc, 1.0))
    ca, cb, cc = _conic(40.0, 0.4, np.pi / 3)
    gs.append((16.0, 16.0, 0.95, ca, cb, cc, 0.8))
    return gs


def test_mask_only_blend_deviates_from_gpu_rule():
    gs = _haze_scene()
    assert np.abs(_render(gs, pixel_floor=False) - _gpu_rule(gs)).max() > 4.0


def test_pixel_floor_blend_matches_gpu_rule():
    gs = _haze_scene()
    assert np.abs(_render(gs, pixel_floor=True) - _gpu_rule(gs)).max() < 1e-6


# Task #44: with the per-pixel floor the mask only has to cover blocks where a
# pixel CENTRE reaches the floor, so the device box is the pixel-centre box
# [x0+0.5, x0+7.5] x [y0+0.5, y0+3.5] instead of [x0, x0+8] x [y0, y0+4].
PC_OX, PC_OY = OX + 0.5, OY + 0.5


def _pc_keep(mx, my, op, ca, cb, cc):
    m2 = cmc.boxmin_m2(ca, cb, cc, PC_OX - mx, PC_OX - mx + cmc.BOX_W,
                       PC_OY - my, PC_OY - my + cmc.BOX_H)
    return op * np.exp(-0.5 * m2) >= FLOOR


def test_checker_models_pixel_centre_box():
    assert (cmc.BOX_W, cmc.BOX_H) == (7.0, 3.0)


def test_pixel_centre_mask_covers_every_live_pixel():
    rng = np.random.default_rng(2)
    yy, xx = np.mgrid[0:32, 0:32] + 0.5
    mb = ((yy // 4) * 4 + (xx // 8)).astype(int)
    tighter = 0
    for _ in range(2000):
        ca, cb, cc = _conic(10 ** rng.uniform(-1, 1.5), 10 ** rng.uniform(-1, 1.5),
                            rng.uniform(0, np.pi))
        mx, my = rng.uniform(-16, 48, 2)
        op = 10 ** rng.uniform(-2.5, 0)
        dx, dy = xx - mx, yy - my
        a = op * np.exp(-0.5 * (ca * dx * dx + 2 * cb * dx * dy + cc * dy * dy))
        live = np.zeros(32, bool)
        np.logical_or.at(live, mb.ravel(), (a >= FLOOR).ravel())
        keep = _pc_keep(mx, my, op, ca, cb, cc)
        assert not (live & ~keep).any()
        old = op * np.exp(-0.5 * cmc.boxmin_m2(ca, cb, cc, OX - mx, OX - mx + 8,
                                              OY - my, OY - my + 4)) >= FLOOR
        assert not (keep & ~old).any()
        tighter += int((old & ~keep).sum())
    assert tighter > 0


def test_pixel_centre_mask_blend_matches_gpu_rule():
    gs = _haze_scene()
    yy, xx = np.mgrid[0:32, 0:32] + 0.5
    mb = ((yy // 4) * 4 + (xx // 8)).astype(int)
    rgb, t = np.zeros((32, 32)), np.ones((32, 32))
    for (mx, my, op, ca, cb, cc, col) in gs:
        keep = _pc_keep(mx, my, op, ca, cb, cc)[mb]
        dx, dy = xx - mx, yy - my
        a = np.minimum(op * np.exp(-0.5 * (ca * dx * dx + 2 * cb * dx * dy + cc * dy * dy)), 0.99)
        a = np.where(keep & (a >= FLOOR), a, 0.0)
        rgb += a * t * col
        t *= 1 - a
    assert np.abs(rgb * 255 - _gpu_rule(gs)).max() < 1e-6


EXP_REL_ERR = 1.73e-3   # blend exp_21f fit, max relative over-read of alpha
LOG_UNDER = 0.0035      # cull SFPU log, max underestimate of ln


def test_thr_margin_covers_blend_alpha_rounding():
    # fp32 dest, no bf16 rounding. The blend's exp_21f alpha can read up to
    # EXP_REL_ERR high (~0.0035 in m2) and the cull's SFPU log can read ln up to
    # LOG_UNDER low (0.007 in thr): ~0.011 worst case. Right at a pixel centre
    # there is no box slack, so THR_MARGIN must absorb both.
    worst = 2 * np.log1p(EXP_REL_ERR) + 2 * LOG_UNDER
    assert 0.010 < worst < 0.012
    assert cmc.THR_MARGIN >= worst
    rng = np.random.default_rng(3)
    yy, xx = np.mgrid[0:32, 0:32] + 0.5
    mb = ((yy // 4) * 4 + (xx // 8)).astype(int)
    for _ in range(2000):
        ca, cb, cc = _conic(10 ** rng.uniform(-1, 1.5), 10 ** rng.uniform(-1, 1.5),
                            rng.uniform(0, np.pi))
        mx, my = rng.uniform(-16, 48, 2)
        op = 10 ** rng.uniform(-2.5, 0)
        dx, dy = xx - mx, yy - my
        a = op * np.exp(-0.5 * (ca * dx * dx + 2 * cb * dx * dy + cc * dy * dy))
        live = np.zeros(32, bool)
        np.logical_or.at(live, mb.ravel(), (a * (1 + EXP_REL_ERR) >= FLOOR).ravel())
        m2 = cmc.boxmin_m2(ca, cb, cc, PC_OX - mx, PC_OX - mx + cmc.BOX_W,
                           PC_OY - my, PC_OY - my + cmc.BOX_H)
        keep = m2 <= 2 * (np.log(op / FLOOR) - LOG_UNDER) + cmc.THR_MARGIN
        assert not (live & ~keep).any()
