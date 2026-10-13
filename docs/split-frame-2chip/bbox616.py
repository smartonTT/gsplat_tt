"""Task #616: per-view bbox statistics for row/column split lines (fast, no pair loop).

For each of the 30 bicycle views: the joint histogram of each visible gaussian's tile-row range
(y0, y1) and tile-column range (x0, x1), and the per-tile bbox pair count (every tile of the
AABB, what K2 emits before mat's early cull). From these, M on each side of any split line
and bbox P per tile follow exactly. Same projection as tiles616.py.

Usage: python3 bbox616.py [out.npz]
"""
import json, math, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tiles616 import REPO, TS, FLOOR, load_bicycle  # noqa: E402


def view(means, cov3, op, c2w, W, H, fov_deg, chunk=1_000_000):
    c2w = np.asarray(c2w, np.float64)
    Rw = c2w[:3, :3].T
    tw = -Rw @ c2w[:3, 3]
    f = 0.5 * max(W, H) / math.tan(0.5 * math.radians(fov_deg))
    cx, cy = W * 0.5, H * 0.5
    nx, ny = (W + TS - 1) // TS, (H + TS - 1) // TS
    hy = np.zeros((ny, ny), np.int64)
    hx = np.zeros((nx, nx), np.int64)
    diff = np.zeros((ny + 1, nx + 1), np.int64)
    max_r = min(H, W) / 2
    with np.errstate(all="ignore"):
        for s in range(0, len(means), chunk):
            mc = means[s:s + chunk].astype(np.float64) @ Rw.T + tw
            o = op[s:s + chunk].astype(np.float64)
            keep = (mc[:, 2] > 0.2) & (o >= FLOOR)
            mc, o, cv = mc[keep], o[keep], cov3[s:s + chunk][keep].astype(np.float64)
            tx, ty, tz = mc.T
            u = f * tx / tz + cx
            v = f * ty / tz + cy
            J = np.zeros((len(tz), 2, 3))
            J[:, 0, 0] = f / tz; J[:, 0, 2] = -f * tx / tz ** 2
            J[:, 1, 1] = f / tz; J[:, 1, 2] = -f * ty / tz ** 2
            T = J @ Rw[None]
            c2 = T @ cv @ T.transpose(0, 2, 1)
            a, c = c2[:, 0, 0] + 0.3, c2[:, 1, 1] + 0.3
            t = 2.0 * np.log(o / FLOOR)
            rx = np.minimum(np.sqrt(t * a), max_r)
            ry = np.minimum(np.sqrt(t * c), max_r)
            vis = (u + rx > 0) & (u - rx < W) & (v + ry > 0) & (v - ry < H)
            u, v, rx, ry = u[vis], v[vis], rx[vis], ry[vis]
            x0 = np.clip(np.floor((u - rx) / TS), 0, nx - 1).astype(np.int64)
            x1 = np.clip(np.floor((u + rx) / TS), 0, nx - 1).astype(np.int64)
            y0 = np.clip(np.floor((v - ry) / TS), 0, ny - 1).astype(np.int64)
            y1 = np.clip(np.floor((v + ry) / TS), 0, ny - 1).astype(np.int64)
            np.add.at(hy, (y0, y1), 1)
            np.add.at(hx, (x0, x1), 1)
            np.add.at(diff, (y0, x0), 1)
            np.add.at(diff, (y0, x1 + 1), -1)
            np.add.at(diff, (y1 + 1, x0), -1)
            np.add.at(diff, (y1 + 1, x1 + 1), 1)
    bbox_tile = diff.cumsum(0).cumsum(1)[:ny, :nx].reshape(-1)
    return hy, hx, bbox_tile


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("bbox616.npz")
    cam = json.loads((REPO / "benchmarks/cameras_v2.json").read_text())["bicycle"]
    W, H = cam["image_size"]
    ply = REPO / cam["ply"]
    if not ply.exists():
        ply = Path("/Users/smarton/dev/gsplat_tt/scenes/bicycle.ply")
    means, cov3, op = load_bicycle(ply)
    HY, HX, BT = [], [], []
    for name in cam["order"]:
        t0 = time.time()
        hy, hx, bt = view(means, cov3, op, cam["views"][name]["c2w"], W, H, cam["fov_deg"])
        HY.append(hy); HX.append(hx); BT.append(bt)
        print(name, int(hy.sum()), int(bt.sum()), f"{time.time() - t0:.1f}s", flush=True)
    np.savez_compressed(out, order=np.array(cam["order"]), hy=np.stack(HY), hx=np.stack(HX),
                        bbox_tile=np.stack(BT))


if __name__ == "__main__":
    main()
