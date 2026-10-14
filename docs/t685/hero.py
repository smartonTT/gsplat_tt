"""#707: bicycle hero training-view dump for the T1 device blend backward.

    PYTHONPATH=. python3 docs/t685/hero.py tmp/t707/hero.npz [--meta docs/t685/hero_meta.json]

scenes/bicycle.ply, camera _DSC8679 from benchmarks/cameras_test.json at 1237x822. Projection
as gsplat/train_oracle.render (near 0.2, lowpass 0.3, SH degree 3 + 0.5 clamped, depth order);
Gaussians with sigmoid(opacity) < 1/255 are dropped first, as the renderer does (lossless:
they never pass alpha_min). The image is an fp32 forward with the device rules
(blend_backward_dev.forward_fast); dL/dimage is the mean-L1 gradient against
scenes/mipnerf360/bicycle/images_4/_DSC8679.JPG. The npz goes under tmp/ (not committed);
only the small metadata JSON is committed. Never commit the images or the dataset files.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

from gsplat.blend_backward_dev import forward_fast
from gsplat.blend_backward_ref import BlendCfg
from gsplat.train_oracle import SH_C0, SH_C1, SH_C2, SH_C3

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCENES = os.environ.get("GSTT_SCENES", os.path.join(ROOT, "scenes"))


def scenes_dir() -> str:
    """scenes/ lives in the main checkout; a worktree has none."""
    if os.path.isdir(SCENES):
        return SCENES
    p = os.path.abspath(os.path.join(ROOT, "..", "..", "..", "scenes"))   # tt-project/worktrees/<t>
    if os.path.isdir(p):
        return p
    sys.exit(f"no scenes dir (set GSTT_SCENES): {SCENES}")


def load_ply(path: str) -> tuple[np.ndarray, dict]:
    """All-float binary ply as one (n, n_props) float32 array (one sequential read) + column map."""
    with open(path, "rb") as f:
        names, n = [], 0
        while True:
            ln = f.readline().decode().strip()
            if ln.startswith("element vertex"):
                n = int(ln.split()[-1])
            elif ln.startswith("property float"):
                names.append(ln.split()[-1])
            elif ln == "end_header":
                off = f.tell()
                break
    a = np.fromfile(path, "<f4", count=n * len(names), offset=off).reshape(n, len(names))
    return a, {k: i for i, k in enumerate(names)}


def sh_eval(sh, d):
    x, y, z = d[:, 0:1], d[:, 1:2], d[:, 2:3]
    xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
    return (SH_C0 * sh[:, 0] - SH_C1 * y * sh[:, 1] + SH_C1 * z * sh[:, 2] - SH_C1 * x * sh[:, 3]
            + SH_C2[0] * xy * sh[:, 4] + SH_C2[1] * yz * sh[:, 5] + SH_C2[2] * (2 * zz - xx - yy) * sh[:, 6]
            + SH_C2[3] * xz * sh[:, 7] + SH_C2[4] * (xx - yy) * sh[:, 8]
            + SH_C3[0] * y * (3 * xx - yy) * sh[:, 9] + SH_C3[1] * xy * z * sh[:, 10]
            + SH_C3[2] * y * (4 * zz - xx - yy) * sh[:, 11] + SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[:, 12]
            + SH_C3[4] * x * (4 * zz - xx - yy) * sh[:, 13] + SH_C3[5] * z * (xx - yy) * sh[:, 14]
            + SH_C3[6] * x * (xx - 3 * yy) * sh[:, 15])


def project(rows, col, c2w, intr, H, W, near=0.2, lowpass=0.3):
    """float64 projection of the ply rows (column map col); returns the dump arrays."""
    def v(name):
        return rows[:, col[name]]

    w2c = np.linalg.inv(np.asarray(c2w, np.float64))
    R, t = w2c[:3, :3], w2c[:3, 3]
    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
    mean = np.stack([v("x"), v("y"), v("z")], 1).astype(np.float64)
    pc = mean @ R.T + t
    tx, ty, tz = pc.T
    valid = tz > near
    tzs = np.where(valid, tz, 1.0)
    m2 = np.stack([fx * tx / tzs + cx, fy * ty / tzs + cy], 1)
    q = np.stack([v(f"rot_{i}") for i in range(4)], 1).astype(np.float64)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    Rq = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                   2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                   2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1).reshape(-1, 3, 3)
    s = np.exp(np.stack([v(f"scale_{i}") for i in range(3)], 1).astype(np.float64))
    M = R @ (Rq * s[:, None, :])                       # camera-frame R Rq S
    z0 = np.zeros_like(tz)
    J = np.stack([np.stack([fx / tzs, z0, -fx * tx / tzs ** 2], -1),
                  np.stack([z0, fy / tzs, -fy * ty / tzs ** 2], -1)], 1)
    JM = J @ M
    cov = JM @ JM.transpose(0, 2, 1)
    a, b, c = cov[:, 0, 0] + lowpass, cov[:, 0, 1], cov[:, 1, 1] + lowpass
    det = a * c - b * b
    conic = np.stack([c / det, -b / det, a / det], 1)
    rx, ry = np.ceil(3 * np.sqrt(np.maximum(a, 0))), np.ceil(3 * np.sqrt(np.maximum(c, 0)))
    vis = valid & (m2[:, 0] + rx > 0) & (m2[:, 0] - rx < W) & (m2[:, 1] + ry > 0) & (m2[:, 1] - ry < H)
    cam = -(R.T @ t)
    d = mean - cam
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    dc, rest = col["f_dc_0"], col["f_rest_0"]
    sh = np.concatenate([rows[:, dc:dc + 3][:, None, :],
                         rows[:, rest:rest + 45].reshape(-1, 3, 15).transpose(0, 2, 1)], 1).astype(np.float64)
    col = np.clip(sh_eval(sh, d) + 0.5, 0.0, 1.0)
    return m2, conic, col, tz, vis


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--view", default="_DSC8679")
    ap.add_argument("--meta", default=None, help="write derived metadata JSON here")
    a = ap.parse_args(argv)
    t0 = time.time()
    cams = json.load(open(os.path.join(ROOT, "benchmarks/cameras_test.json")))["bicycle"]
    W, H = cams["image_size"]
    sc = scenes_dir()
    ply, cm = load_ply(os.path.join(sc, "bicycle.ply"))
    op_all = 1.0 / (1.0 + np.exp(-ply[:, cm["opacity"]].astype(np.float64)))
    keep = np.flatnonzero(op_all >= 1.0 / 255.0)
    m2, conic, col, tz, vis = project(ply[keep], cm, cams["views"][a.view]["c2w"], cams["intrinsics"], H, W)
    sel = np.flatnonzero(vis)                              # in front, 3-sigma box overlaps the image
    m2, conic, col, tz = m2[sel], conic[sel], col[sel], tz[sel]
    op = op_all[keep][sel]
    order = np.argsort(tz, kind="stable").astype(np.int64)
    valid = np.ones(len(sel), bool)
    f = np.float32
    m2, conic, op, col = m2.astype(f), conic.astype(f), op.astype(f), col.astype(f)
    cfg = BlendCfg()
    t1 = time.time()
    img, st = forward_fast(m2, conic, op, col, order, valid, H, W, cfg)
    t2 = time.time()
    from PIL import Image
    jpg = os.path.join(sc, "mipnerf360/bicycle/images_4", a.view + ".JPG")
    gt = np.asarray(Image.open(jpg).convert("RGB"), f) / f(255)
    assert gt.shape == img.shape, (gt.shape, img.shape)
    mse = float(np.mean((np.clip(img, 0, 1).astype(np.float64) - gt) ** 2))
    psnr = 10 * np.log10(1.0 / mse)
    dimg = (np.sign(img - gt) / f(3 * H * W)).astype(f)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, H=H, W=W, N=len(sel), means2d=m2, conic=conic, opacity=op, colors=col, order=order,
             valid=valid, image=img, dL_dimage=dimg, target_md5=hashlib.md5(open(jpg, "rb").read()).hexdigest(),
             cfg_json=json.dumps({"alpha_max": cfg.alpha_max, "alpha_min": cfg.alpha_min, "t_min": cfg.t_min,
                                  "background": list(cfg.background)}))
    meta = {"task": 707, "scene": "scenes/bicycle.ply", "view": a.view, "H": H, "W": W,
            "n_ply": int(len(ply)), "n_opacity_ge_1_255": int(len(keep)), "n_visible": int(len(sel)),
            **st, "l1_loss": float(np.mean(np.abs(img - gt))), "psnr_vs_jpg_db": round(psnr, 3),
            "target": f"scenes/mipnerf360/bicycle/images_4/{a.view}.JPG",
            "target_md5": hashlib.md5(open(jpg, "rb").read()).hexdigest(),
            "image_md5_f32": hashlib.md5(img.tobytes()).hexdigest(),
            "forward": "blend_backward_dev.forward_fast (fp32, device rules, numpy exp)",
            "build_s": round(time.time() - t0, 1), "forward_s": round(t2 - t1, 1)}
    print(json.dumps(meta))
    if a.meta:
        json.dump(meta, open(a.meta, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
