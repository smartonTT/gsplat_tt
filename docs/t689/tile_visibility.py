"""T2 sizing (task #689): how many SoA blocks does a sparse (visible-only) Adam pass touch?

Per training view of bicycle (benchmarks/cameras_train_nn.json, 1237x822), the T0 oracle's
sparse-Adam visibility (z > 0.2 and the 3-sigma box overlaps the image), then the share of
Gaussian blocks (B = 32, 1024 rows) holding at least one visible row, in file order and
after a one-time Morton (Z-order) permutation of the means (an exact relabelling).
CPU only, every --stride-th view.  python3 docs/t689/tile_visibility.py [--stride 4]
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from gsplat.utils import c2w_to_w2c, quat_to_rotation_matrix  # noqa: E402


def read_ply(path):
    """Minimal binary_little_endian float PLY reader (no plyfile dependency)."""
    with open(path, "rb") as f:
        n, names = 0, []
        while (line := f.readline().strip()) != b"end_header":
            w = line.split()
            if w[:2] == [b"element", b"vertex"]:
                n = int(w[2])
            elif w[0] == b"property":
                assert w[1] == b"float", w
                names.append(w[2].decode())
        a = np.frombuffer(f.read(n * 4 * len(names)), "<f4").reshape(n, len(names))
    col = {k: a[:, i] for i, k in enumerate(names)}
    st = lambda *ks: torch.from_numpy(np.stack([col[k] for k in ks], 1).copy())  # noqa: E731
    q = st("rot_0", "rot_1", "rot_2", "rot_3")
    return st("x", "y", "z"), torch.exp(st("scale_0", "scale_1", "scale_2")), q / q.norm(dim=1, keepdim=True)


def morton(means, bits=21):
    lo, hi = np.percentile(means, 1, 0), np.percentile(means, 99, 0)
    q = np.clip((means - lo) / (hi - lo), 0, 1) * ((1 << bits) - 1)
    q = q.astype(np.uint64)
    code = np.zeros(len(q), np.uint64)
    for b in range(bits):
        for a in range(3):
            code |= ((q[:, a] >> np.uint64(b)) & np.uint64(1)) << np.uint64(3 * b + a)
    return np.argsort(code, kind="stable")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    cams = json.load(open(ROOT / "benchmarks/cameras_train_nn.json"))["bicycle"]
    W, H = cams["image_size"]
    it = cams["intrinsics"]
    ply = ROOT / "scenes/bicycle.ply"
    means, scales, quats = read_ply(ply if ply.exists() else ROOT.parents[2] / "scenes/bicycle.ply")
    Rq = quat_to_rotation_matrix(quats)
    M = Rq * scales.unsqueeze(1)
    cov3 = M @ M.transpose(1, 2)
    N = means.shape[0]
    perm = torch.from_numpy(morton(means.numpy()))
    rows = []
    for name in cams["order"][::a.stride]:
        w2c = c2w_to_w2c(torch.tensor(cams["views"][name]["c2w"], dtype=torch.float32))
        R, t = w2c[:3, :3], w2c[:3, 3]
        pc = means @ R.T + t
        z = pc[:, 2]
        valid = z > 0.2
        zs = torch.where(valid, z, torch.ones_like(z))
        mx = it["fx"] * pc[:, 0] / zs + it["cx"]
        my = it["fy"] * pc[:, 1] / zs + it["cy"]
        J = torch.zeros(N, 2, 3)
        J[:, 0, 0] = it["fx"] / zs
        J[:, 0, 2] = -it["fx"] * pc[:, 0] / zs ** 2
        J[:, 1, 1] = it["fy"] / zs
        J[:, 1, 2] = -it["fy"] * pc[:, 1] / zs ** 2
        T = J @ R
        c2 = T @ cov3 @ T.transpose(1, 2)
        rx = torch.ceil(3 * torch.sqrt((c2[:, 0, 0] + 0.3).clamp(min=0)))
        ry = torch.ceil(3 * torch.sqrt((c2[:, 1, 1] + 0.3).clamp(min=0)))
        vis = valid & (mx + rx > 0) & (mx - rx < W) & (my + ry > 0) & (my - ry < H)
        r = {"view": name, "visible": float(vis.float().mean())}
        for tag, v in (("file", vis), ("morton", vis[perm])):
            for B in (32, 1024):
                pad = (-N) % B
                vb = torch.cat([v, torch.zeros(pad, dtype=torch.bool)]).view(-1, B).any(1)
                r[f"{tag}_B{B}"] = float(vb.float().mean())
        rows.append(r)
        print(json.dumps(r), flush=True)
    keys = [k for k in rows[0] if k != "view"]
    print("MEAN", json.dumps({k: round(float(np.mean([r[k] for r in rows])), 4) for k in keys}),
          "N", N, "views", len(rows))


if __name__ == "__main__":
    main()
