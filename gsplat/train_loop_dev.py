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
    ref = np.asarray(keys["dL_dimage"], np.float64)
    rep = {"loss": loss, "oracle_loss": float(keys["loss"]) if "loss" in keys else None,
           "grad_l2rel_vs_oracle": float(np.linalg.norm(g - ref) / max(np.linalg.norm(ref), 1e-30)),
           "grad_finite": bool(np.isfinite(g).all())}
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
