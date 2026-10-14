"""#750 step 2: host glue between the device stages of one training step, dump-free.

  python3 -m gsplat.train_loop_dev loss-prep <dump> <image.f32> <loss_dir>
  python3 -m gsplat.train_loop_dev chain-dump <dump> <loss_dir> <out.npz>

The chain: blend_fwd_bench (#716, train-mode fp32 forward; `blend_forward_dev check --image-out`
writes its image) -> loss_bench LOSS_MODE=ssim (#736/#743, fused L1 + D-SSIM loss and dL/dimage)
-> train_step (#748, T1 blend backward -> T2 gather + Adam).

loss-prep: the device forward image (H, W, 3) and the dump's target -> loss_bench inputs
(image.f32, target.f32, hw.txt, consts.f32). consts.f32 holds the target's filtered maps, a
per-view constant computed once per dataset, not per step.
chain-dump: the dump with image and dL_dimage replaced by the device forward image and the device
loss gradient, the input of `train_step_dev prep`. The T0 oracle keys (autograd grads, bf16 Adam)
stay, so `train_step_dev check` gates the whole device chain against the oracle.
A dump without the oracle's loss (the #781 hero dump: its dL/dimage is not a D-SSIM+L1 gradient)
is checked against dssim_l1_f64 of the device image instead (#795 step c, the hero loss oracle).
Each prints one JSON line with its host ms; chain-dump also compares the device loss and dL/dimage
with the oracle's.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

f32 = np.float32


def target_consts(y: np.ndarray) -> np.ndarray:
    """loss_bench consts.f32 for target y (opt/t736/ssim_check.consts)."""
    root = Path(__file__).resolve().parent.parent
    for p in ("t690", "t714", "t720", "t736"):
        if str(root / "opt" / p) not in sys.path:
            sys.path.insert(0, str(root / "opt" / p))
    from ssim_check import consts
    return consts(np.asarray(y, f32))


def _gauss_f64(z: np.ndarray, win: int = 11, sigma: float = 1.5) -> np.ndarray:
    """3DGS's 11x11 Gaussian window on the last two axes, zero padding ('same' size), fp64. The
    kernel is symmetric, so this is also its own adjoint."""
    r = win // 2
    g = np.exp(-((np.arange(win) - r) ** 2) / (2.0 * sigma * sigma))
    g /= g.sum()
    H, W = z.shape[-2:]
    zp = np.pad(np.asarray(z, np.float64), [(0, 0)] * (z.ndim - 2) + [(r, r), (r, r)])
    v = sum(g[k] * zp[..., k:k + H, :] for k in range(win))
    return sum(g[k] * v[..., k:k + W] for k in range(win))


def dssim_l1_f64(image: np.ndarray, target: np.ndarray, lambda_dssim: float = 0.2):
    """Host oracle of the 3DGS loss (1 - l) L1 + l (1 - SSIM) and its dL/dimage, fp64, written
    directly from the definition (gsplat/train_oracle.py loss_fn: per-channel 11x11 Gaussian
    window, sigma 1.5, zero padding, mean over all H W 3 values; torch's sign(0) = 0 for L1),
    independent of the device's tile-matmul form (gsplat/train_loss.py). (H, W, 3) in and out."""
    x = np.moveaxis(np.asarray(image, np.float64), -1, 0)
    y = np.moveaxis(np.asarray(target, np.float64), -1, 0)
    c1, c2, n = 0.01 ** 2, 0.03 ** 2, float(x.size)
    mx, my = _gauss_f64(x), _gauss_f64(y)
    exx, eyy, exy = _gauss_f64(x * x), _gauss_f64(y * y), _gauss_f64(x * y)
    A, B = 2 * mx * my + c1, 2 * (exy - mx * my) + c2
    C, D = mx * mx + my * my + c1, (exx - mx * mx) + (eyy - my * my) + c2
    S = A * B / (C * D)
    w = -lambda_dssim / n
    d_mu = w * S * (2 * my / A - 2 * my / B - 2 * mx / C + 2 * mx / D)
    d_xx, d_xy = -w * S / D, 2 * w * S / B
    g = _gauss_f64(d_mu) + 2 * x * _gauss_f64(d_xx) + y * _gauss_f64(d_xy)
    g += (1.0 - lambda_dssim) / n * np.sign(x - y)
    l1 = float(np.abs(x - y).mean())
    loss = (1.0 - lambda_dssim) * l1 + lambda_dssim * (1.0 - float(S.mean()))
    return loss, np.moveaxis(g, 0, -1)


def loss_prep(dump: str, image: str, out: str) -> dict:
    t = time.perf_counter()
    d = np.load(dump)
    H, W = int(d["H"]), int(d["W"])
    x = np.fromfile(image, f32).reshape(H, W, 3)
    o = Path(out)
    o.mkdir(parents=True, exist_ok=True)
    (o / "hw.txt").write_text(f"{H} {W}\n")
    x.tofile(o / "image.f32")
    y = np.ascontiguousarray(d["target"], f32)
    y.tofile(o / "target.f32")
    t1 = time.perf_counter()
    target_consts(y).tofile(o / "consts.f32")
    t2 = time.perf_counter()
    return {"H": H, "W": W, "image_max_abs_vs_oracle": float(np.abs(x - d["image"]).max()),
            "host_ms": round((t1 - t) * 1e3, 3), "consts_ms_per_dataset": round((t2 - t1) * 1e3, 3)}


def chain_dump(dump: str, loss_dir: str, out: str) -> dict:
    t = time.perf_counter()
    d = np.load(dump)
    H, W = int(d["H"]), int(d["W"])
    ld = Path(loss_dir)
    x = np.fromfile(ld / "image.f32", f32).reshape(H, W, 3)
    g = np.fromfile(ld / "out" / "grad.f32", f32).reshape(H, W, 3)
    loss = float((ld / "out" / "loss.txt").read_text().split()[0])
    keys = {k: d[k] for k in d.files}
    if "loss" in keys:
        ref, ref_loss, rep = np.asarray(keys["dL_dimage"], np.float64), float(keys["loss"]), {"oracle": "dump"}
    else:  # no oracle loss in the dump (hero): the fp64 host loss of the device image
        ref_loss, ref = dssim_l1_f64(x, keys["target"])
        rep = {"oracle": "host_f64_dssim_l1"}
    rep.update({"loss": loss, "oracle_loss": ref_loss, "loss_abs_vs_oracle": abs(loss - ref_loss),
                "grad_l2rel_vs_oracle": float(np.linalg.norm(g - ref) / max(np.linalg.norm(ref), 1e-30)),
                "grad_finite": bool(np.isfinite(g).all())})
    rep["grad_max_rel_vs_oracle"] = float(np.abs(g - ref).max() / max(np.abs(ref).max(), 1e-30))
    keys["image"], keys["dL_dimage"] = x, g
    keys["chain_dl_dimage_max_rel"] = np.float64(rep["grad_max_rel_vs_oracle"])
    np.savez(out, **keys)
    rep["host_ms"] = round((time.perf_counter() - t) * 1e3, 3)
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("loss-prep")
    a.add_argument("dump")
    a.add_argument("image")
    a.add_argument("out")
    b = sp.add_parser("chain-dump")
    b.add_argument("dump")
    b.add_argument("loss_dir")
    b.add_argument("out")
    a = ap.parse_args(argv)
    if a.cmd == "loss-prep":
        print(json.dumps(loss_prep(a.dump, a.image, a.out)))
    else:
        print(json.dumps(chain_dump(a.dump, a.loss_dir, a.out)))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    sys.exit(main())
