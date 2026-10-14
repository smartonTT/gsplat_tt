"""Task #671: CleanBackend.bump_scene_version (render/run.py) with a fake
render_clean, no device. The host scene caches (numpy views, cov3d, prune) are
keyed by pointer and shape, so an in-place parameter update keeps hitting them;
after bump_scene_version the next render sees the new values and render_clean's
scene version is bumped.

    python3 tests/unit/test_scene_version_t671.py      (or pytest)
"""
import pathlib
import sys
import types

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from test_run_back_to_back import _load_run  # noqa: E402


class _FakeClean:
    def __init__(self):
        self.calls = []
        self.version = 1

    def render_view(self, means, cov3d, opac, colors, *a, **kw):
        self.calls.append((means.copy(), cov3d.copy(), opac.copy()))
        return np.zeros((2, 2, 3), np.uint8), {}

    def bump_scene_version(self):
        self.version += 1
        return self.version


class _FakeMod:
    def compute_cov3d(self, scales, rotations):
        return np.repeat(scales[:, :1], 9, axis=1).astype(np.float32)


def _backend(run):
    b = object.__new__(run.CleanBackend)
    b._clean = _FakeClean()
    b._mod = _FakeMod()
    for k, v in dict(_mb_contrib_floor=0.0, contrib_floor_override=None, min_opacity=0.0,
                     cull_disabled=False, transmittance_threshold=0.0, max_radius=0,
                     k_cap=3.0, use_isoellipse=False, next_extrinsics=None,
                     _prune=False, _pruned=None, _gauss_np_cache_key=None,
                     _gauss_np_cache=None, _cov3d_cache_key=None,
                     _cov3d_cache_buf=None).items():
        setattr(b, k, v)
    return b


def test_in_place_update_needs_bump():
    run = _load_run()
    torch = run.torch
    n = 4
    g = types.SimpleNamespace(
        means=torch.zeros(n, 3), scales=torch.ones(n, 3), rotations=torch.zeros(n, 4),
        opacities=torch.full((n,), 0.5), colors=torch.zeros(n, 3))
    b = _backend(run)
    render = lambda: b.render_fused(g, torch.eye(4), torch.eye(3), 2, 2)  # noqa: E731

    render()
    with torch.no_grad():  # an optimizer step: same tensors, new values
        g.means += 1.0
        g.scales *= 3.0
        g.opacities.fill_(0.9)
    render()
    # Without a bump the host caches still hand render_view the old scene
    # (this is the training bug the version fixes).
    m0, c0, o0 = b._clean.calls[0]
    m1, c1, o1 = b._clean.calls[1]
    assert np.array_equal(c1, c0)

    assert b.bump_scene_version() == 2 and b._clean.version == 2
    render()
    m2, c2, o2 = b._clean.calls[2]
    assert np.allclose(m2, 1.0) and np.allclose(o2, 0.9) and np.allclose(c2, 3.0)
    # And the bump is not a one-shot recompute: the next view hits again.
    key = b._cov3d_cache_key
    render()
    assert b._cov3d_cache_key == key and np.allclose(b._clean.calls[3][1], 3.0)


def test_bump_drops_prune_cache():
    run = _load_run()
    b = _backend(run)
    b._pruned = ("src", "arrs")
    b.bump_scene_version()
    assert b._pruned is None


if __name__ == "__main__":
    test_in_place_update_needs_bump()
    test_bump_drops_prune_cache()
    print("ok")
