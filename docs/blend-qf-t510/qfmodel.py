"""t510 stage 0: CPU precision model of an FPU quadratic-form blend alpha.

The device blend computes, per (gaussian, microblock), on the SFPU in fp32:
    dx = x - mx ; dy = y - my ; power = A dx^2 + B dx dy + C dy^2
An FPU path would instead compute power = f . c as a matmul, with per-pixel
features f = [u^2, uv, v^2, u, v, 1] (u, v = pixel centre minus an origin) and
per-gaussian coefficients c = [A, B, C, D, E, F] (D, E, F expanded around the
same origin in fp32), operands rounded to the FPU format (tf32: 10 mantissa
bits, bf16: 7), products exact, fp32 accumulation. "hilo" splits each
coefficient into hi + lo parts of the same format (twice the K).

Everything else (projection, tile lists, depth order, exp, alpha clamp 0.99,
alpha floor 1/255, T stop 1e-4) is identical across arms, so the per-view PSNR
of an arm against "base" isolates the quadratic-form error.

Usage: python3 qfmodel.py --views all|hero|N --out <dir> [--variants a,b,..]
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(os.environ.get("QF_ROOT", "/Users/smarton/dev/gsplat_tt"))  # gsplat package + scenes (read only)
WT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _py39_shim():
    """The gsplat package uses PEP 604 annotations; this Mac's torch python is
    3.9, so load its modules with postponed annotations."""
    import importlib.abc
    import importlib.util

    class L(importlib.abc.SourceLoader):
        def __init__(self, path):
            self.path = path

        def get_filename(self, name):
            return self.path

        def get_data(self, path):
            return b"from __future__ import annotations\n" + Path(path).read_bytes()

    class F(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if not name.startswith("gsplat."):
                return None
            fp = ROOT / (name.replace(".", "/") + ".py")
            if not fp.exists():
                return None
            src = fp.read_text()
            if src.lstrip().startswith(("from __future__", '"""')) and "from __future__" in src[:2000]:
                return None
            return importlib.util.spec_from_file_location(name, str(fp), loader=L(str(fp)))

    sys.meta_path.insert(0, F())


_py39_shim()

from gsplat.backend import Backend  # noqa: E402
from gsplat.data_structures import Gaussians  # noqa: E402
from gsplat.pipeline import Pipeline  # noqa: E402
from gsplat.utils import c2w_to_w2c  # noqa: E402
from PIL import Image  # noqa: E402

torch.set_num_threads(12)


def rnd(x: torch.Tensor, mbits: int) -> torch.Tensor:
    """Round fp32 to `mbits` explicit mantissa bits, nearest-even."""
    if mbits >= 23:
        return x
    i = x.contiguous().view(torch.int32)
    drop = 23 - mbits
    half = (1 << (drop - 1)) - 1
    lsb = (i >> drop) & 1
    i = (i + half + lsb) & ~((1 << drop) - 1)
    return i.view(torch.float32)


FMT = {"tf32": 10, "bf16": 7, "fp32": 23}


def load_ply(path):
    """Numpy PLY reader (no plyfile here); same activations as gsplat.loading_gaussians."""
    with open(path, "rb") as fh:
        names, n = [], 0
        while True:
            ln = fh.readline().decode().strip()
            if ln.startswith("element vertex"):
                n = int(ln.split()[-1])
            elif ln.startswith("property float"):
                names.append(ln.split()[-1])
            elif ln == "end_header":
                break
        a = np.fromfile(fh, dtype=np.float32, count=n * len(names)).reshape(n, len(names))
    col = {k: i for i, k in enumerate(names)}

    def g(*ks):
        return torch.from_numpy(np.ascontiguousarray(a[:, [col[k] for k in ks]]))

    q = g("rot_0", "rot_1", "rot_2", "rot_3")
    return Gaussians(
        means=g("x", "y", "z"),
        scales=torch.exp(g("scale_0", "scale_1", "scale_2")),
        rotations=q / q.norm(dim=-1, keepdim=True),
        colors=torch.clamp(0.5 + 0.28209479177387814 * g("f_dc_0", "f_dc_1", "f_dc_2"), 0.0, 1.0),
        opacities=torch.sigmoid(g("opacity")[:, 0]),
    )


class Grab(Backend):
    def blend(self, *a):
        self.args = a
        return np.zeros((1, 1, 3), np.float32), {}


def conic(covs):
    a, b, c = covs[:, 0, 0], covs[:, 0, 1], covs[:, 1, 1]
    det = a * c - b * b
    inv = 1.0 / det
    return -0.5 * c * inv, b * inv, -0.5 * a * inv  # A, B, C (-0.5 folded)


def power_base(u, v, mu, mv, A, B, C):
    dx = u[None, :] - mu[:, None]
    dy = v[None, :] - mv[:, None]
    p = A[:, None] * (dx * dx)
    p = p + B[:, None] * (dx * dy)
    return p + C[:, None] * (dy * dy)


def power_qf(u, v, mu, mv, A, B, C, fmt, origin, hilo, mb_origin=False):
    """u, v: (N,) tile-local pixel centres; mu, mv: (P,) tile-local means.
    Coefficients are rounded per gaussian (per microblock with mb_origin), features
    per pixel; products of two rounded operands are exact in fp32, sums are fp32."""
    m = FMT[fmt]
    if mb_origin:  # per-microblock (4 rows x 8 cols) centre
        ou = (torch.floor(u / 8.0) * 8.0 + 4.0)
        ov = (torch.floor(v / 4.0) * 4.0 + 2.0)
        key = torch.floor(v / 4.0) * 4.0 + torch.floor(u / 8.0)  # 0..31
        mbi = key.long()
        ou_k = torch.zeros(32).index_put_((mbi,), ou)[None, :]
        ov_k = torch.zeros(32).index_put_((mbi,), ov)[None, :]
    else:
        ou = torch.full_like(u, float(origin))
        ov = ou
        mbi = torch.zeros_like(u, dtype=torch.long)
        ou_k = torch.full((1, 1), float(origin))
        ov_k = ou_k
    uu = (u - ou)[None, :]  # exact (half-integers)
    vv = (v - ov)[None, :]
    mu_ = mu[:, None] - ou_k  # (P, K) fp32, K = 1 or 32 origins
    mv_ = mv[:, None] - ov_k
    A_, B_, C_ = A[:, None], B[:, None], C[:, None]
    D = -(2.0 * A_ * mu_ + B_ * mv_)
    E = -(B_ * mu_ + 2.0 * C_ * mv_)
    F = A_ * (mu_ * mu_) + B_ * (mu_ * mv_) + C_ * (mv_ * mv_)
    feats = [rnd(f.contiguous(), m) for f in (uu * uu, uu * vv, vv * vv, uu, vv, torch.ones_like(uu))]
    coefs = [A_.expand_as(D), B_.expand_as(D), C_.expand_as(D), D, E, F]
    p = None
    for f, c in zip(feats, coefs):
        hi = rnd(c.contiguous(), m)
        parts = [hi] + ([rnd((c - hi).contiguous(), m)] if hilo else [])
        for q in parts:
            t = f * q[:, mbi]
            p = t if p is None else p + t
    return p


_CTX = None


def _tiles(ts):
    """Blend a list of tiles from the fork-inherited _CTX; returns [(t, (1024, 3) array)]."""
    tx, tr, sids, means2d, A, B, C, opac, colors, u, v, pfn = _CTX
    outl = []
    for t in ts:
        s, e = int(tr[t, 0]), int(tr[t, 1])
        if e <= s:
            continue
        g = sids[s:e].long()
        mu = means2d[g, 0] - (t % tx) * 32.0
        mv = means2d[g, 1] - (t // tx) * 32.0
        p = pfn(u, v, mu, mv, A[g], B[g], C[g])
        w = torch.exp(torch.clamp(p, max=0.0))
        al = torch.clamp(opac[g][:, None] * w, max=0.99)
        al = torch.where(al < 1.0 / 255.0, torch.zeros_like(al), al)
        tafter = torch.cumprod(1.0 - al, dim=0)
        tbefore = torch.cat([torch.ones(1, al.shape[1]), tafter[:-1]], 0)
        wgt = al * tbefore * (tafter >= 1e-4)
        outl.append((t, (wgt.T @ colors[g]).numpy()))
    return outl


def _init_worker():
    torch.set_num_threads(1)


def blend(args, H, W, pfn, procs=1):
    """Per-tile Python loop over small tensors is GIL/serial-bound, so tiles go to `procs`
    forked workers (one BLAS/intra-op thread each)."""
    global _CTX
    means2d, covs2d, colors, opac, sids, tile_ranges = args[:6]
    A, B, C = conic(covs2d.float())
    tx = (W + 31) // 32
    ty = (H + 31) // 32
    r = torch.arange(32, dtype=torch.float32)
    u = (r[None, :].expand(32, 32) + 0.5).reshape(-1)  # col
    v = (r[:, None].expand(32, 32) + 0.5).reshape(-1)  # row
    tr = tile_ranges.reshape(-1, 2) if tile_ranges.dim() > 1 else tile_ranges
    _CTX = (tx, tr, sids, means2d.float(), A, B, C, opac.float().reshape(-1), colors.float(), u, v, pfn)
    chunks = [list(range(i, min(i + 8, tx * ty))) for i in range(0, tx * ty, 8)]
    if procs > 1:
        with mp.get_context("fork").Pool(procs, initializer=_init_worker) as pool:
            parts = pool.map(_tiles, chunks, chunksize=1)
    else:
        parts = [_tiles(c) for c in chunks]
    img = np.zeros((ty * 32, tx * 32, 3), np.float32)
    for part in parts:
        for t, col in part:
            ty_, tx_ = t // tx, t % tx
            img[ty_ * 32:(ty_ + 1) * 32, tx_ * 32:(tx_ + 1) * 32] = col.reshape(32, 32, 3)
    return np.clip(img[:H, :W], 0, 1)


def u8(img):
    return (np.clip(img, 0, 1) * 255.0).astype(np.uint8)


def psnr8(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)) / 255.0 ** 2
    return 100.0 if mse <= 0 else min(100.0, -10 * math.log10(mse))


VARIANTS = {
    "base": None,
    "tf32_corner": dict(fmt="tf32", origin=0.0, hilo=False),
    "tf32_center": dict(fmt="tf32", origin=16.0, hilo=False),
    "tf32_center_hilo": dict(fmt="tf32", origin=16.0, hilo=True),
    "tf32_mb": dict(fmt="tf32", origin=0.0, hilo=False, mb_origin=True),
    "bf16_center_hilo": dict(fmt="bf16", origin=16.0, hilo=True),
    "bf16_mb_hilo": dict(fmt="bf16", origin=0.0, hilo=True, mb_origin=True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", default="hero")
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--procs", type=int, default=1, help="forked blend workers (1 thread each)")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cam = json.loads((WT / "benchmarks/cameras_v2.json").read_text())["bicycle"]
    W, H = cam["image_size"]
    order = cam["order"]
    if args.views == "hero":
        order = order[:1]
    elif ":" in args.views:  # a:b slice of the 31-view order (hero first)
        a, b = (int(x) for x in args.views.split(":"))
        order = order[a:b]
    elif args.views != "all":
        order = order[: int(args.views)]
    f = 0.5 * max(W, H) / math.tan(0.5 * math.radians(cam["fov_deg"]))
    K = torch.tensor([[f, 0, W * 0.5], [0, f, H * 0.5], [0, 0, 1]], dtype=torch.float32)
    t0 = time.time()
    gauss = load_ply(str(ROOT / cam["ply"]))
    print(f"loaded ply {time.time() - t0:.1f}s", flush=True)
    ref = np.asarray(Image.open(WT / "benchmarks/reference_v2/hero.png").convert("RGB"))
    names = args.variants.split(",")
    rows = []
    for vi, name in enumerate(order):
        c2w = np.asarray(cam["views"][name]["c2w"], dtype=np.float32)
        be = Grab()
        pipe = Pipeline(be, tile_size=32, contrib_floor=cam["contrib_floor"])
        t1 = time.time()
        res = pipe.render(gauss, c2w_to_w2c(torch.from_numpy(c2w)), K, H, W)
        print(f"{name}: pipeline {time.time() - t1:.1f}s {res.timings}", flush=True)
        imgs = {}
        for vn in names:
            spec = VARIANTS[vn]
            if spec is None:
                pfn = power_base
            else:
                pfn = (lambda s: lambda u, v, mu, mv, A, B, C: power_qf(u, v, mu, mv, A, B, C, **s))(spec)
            t1 = time.time()
            imgs[vn] = u8(blend(be.args, H, W, pfn, args.procs))
            row = dict(view=name, variant=vn, s=round(time.time() - t1, 1),
                       pairs=int(be.args[4].numel()))
            if vn != "base" and "base" in imgs:
                row["psnr_vs_base"] = round(psnr8(imgs[vn], imgs["base"]), 3)
                row["maxdiff"] = int(np.abs(imgs[vn].astype(int) - imgs["base"]).max())
            if name == "hero":
                row["psnr_vs_ref"] = round(psnr8(imgs[vn], ref), 3)
                pass
            Image.fromarray(imgs[vn]).save(out / f"{name}_{vn}.png")
            print(json.dumps(row), flush=True)
            rows.append(row)
    (out / f"rows-{args.views.replace(':', '_')}-{args.variants.replace(',', '_')}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)  # skip interpreter teardown: it hung after a forked pool on bh-33
