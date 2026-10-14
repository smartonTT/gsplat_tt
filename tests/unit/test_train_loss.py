"""Task #690 (PLAN-single-chip.md T7): the device-loss model in gsplat/train_loss.py
against the T0 oracle (gsplat/train_oracle.py, torch autograd fp64).

    python3 -m pytest -q tests/unit/test_train_loss.py
"""
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
torch = pytest.importorskip("torch")
from gsplat import train_loss as T  # noqa: E402
from gsplat import train_oracle as O  # noqa: E402

DUMPS = sorted((ROOT / "benchmarks" / "train_oracle").glob("t0_*.npz"))


def test_band_tiles_equal_filter_matrix():
    """The three constant band tiles rebuild the oracle's banded G (both passes)."""
    rng = np.random.default_rng(0)
    z = rng.random((2, 96, 128)).astype(np.float32)
    Gh = O.filter_matrix(96).numpy()
    Gw = O.filter_matrix(128).numpy()
    with np.errstate(all="ignore"):  # macOS Accelerate matmul raises spurious FP flags
        ref = Gh @ z.astype(np.float64) @ Gw.T
    np.testing.assert_allclose(T.filt_tiles(z), ref, rtol=0, atol=2e-6)


@pytest.mark.parametrize("dump", DUMPS, ids=lambda p: p.stem)
@pytest.mark.parametrize("scheme", ["fp32", "split3"])
def test_loss_and_grad_match_oracle_dumps(dump, scheme):
    d = np.load(dump)
    x, y = d["image"].astype(np.float32), d["target"].astype(np.float32)
    loss, _, _, g = T.loss_and_grad_device(x, y, scheme)
    assert abs(loss - float(d["loss"])) < 1e-6
    ref = d["dL_dimage"]
    assert np.abs(g - ref).max() / np.abs(ref).max() < 1e-5


def _pair(h, w, seed=0):
    """A rough natural-image pair: smooth field plus flat patches (the cancellation case)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w] / max(h, w)
    base = 0.5 + 0.3 * np.sin(9 * xx[..., None] + np.arange(3)) * np.cos(7 * yy[..., None])
    base[: h // 3, : w // 3] = 0.7  # flat region: E_xx - mu_x^2 cancels
    y = np.clip(base + rng.normal(0, 0.05, base.shape), 0, 1)
    x = np.clip(y + rng.normal(0, 0.02, base.shape), 0, 1)
    return x, y


def _oracle(x, y):
    xt = torch.tensor(x, requires_grad=True)
    loss = O.loss_fn(xt, torch.tensor(y))
    loss.backward()
    return loss.item(), xt.grad.numpy()


def test_odd_size_padding_matches_oracle():
    """Image sizes that are not multiples of 32 (bicycle is 1237x822): padding masked."""
    x, y = _pair(70, 101)
    lref, gref = _oracle(x, y)
    loss, _, _, g = T.loss_and_grad_device(x.astype(np.float32), y.astype(np.float32), "split3")
    assert abs(loss - lref) < 1e-6
    assert np.linalg.norm(g - gref) / np.linalg.norm(gref) < 1e-4


def test_single_tf32_pass_is_not_enough():
    """Why the kernel splits operands: one TF32 matmul per band tile breaks the variance
    terms in flat regions (rel L2 error far above the 1e-2 oracle bar)."""
    x, y = _pair(96, 128)
    _, gref = _oracle(x, y)
    _, _, _, g = T.loss_and_grad_device(x.astype(np.float32), y.astype(np.float32), "tf32")
    _, _, _, g3 = T.loss_and_grad_device(x.astype(np.float32), y.astype(np.float32), "split3")
    assert np.linalg.norm(g3 - gref) / np.linalg.norm(gref) < 1e-4
    assert np.linalg.norm(g - gref) / np.linalg.norm(gref) > 1e-2


def test_l1_step():
    d = np.load(DUMPS[0])
    x, y = d["image"], d["target"]
    xt = torch.tensor(x, requires_grad=True)
    l1 = 0.8 * (xt - torch.tensor(y)).abs().mean()
    l1.backward()
    loss, g = T.loss_and_grad_l1(x.astype(np.float32), y.astype(np.float32))
    assert abs(loss - l1.item()) < 1e-7
    np.testing.assert_allclose(g, xt.grad.numpy(), rtol=1e-6, atol=0)
