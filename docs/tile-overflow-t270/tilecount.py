"""Task #270: host-side (numpy) per-tile record-count model for bicycle.

Models the device's kept (gaussian, tile) pairs: a pair is kept when the
Gaussian's level set o*exp(-d^2/2) >= floor reaches the tile's pixel-centre
rectangle (pfwc precull mode 2 + the per-pair contrib cull), after the near
plane (z > 0.2), min_opacity (1/255) and blur (+0.3) of the reference
projection. Radii are capped at max_radius = min(H, W)/2 like the device.

Close = dolly in / zoom; far = dolly out / wide FOV (the overflow cases).
Usage: python tilecount.py [--floors 255,1024,16384] [--set bench|close|far|all]
Prints, per view and floor, max/p99/p50 per-tile counts and the number of
tiles over the 32768-record bucket cap.
"""
import argparse, json, math, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
CAP = 32768
TS = 32


def load_bicycle(path):
    with open(path, "rb") as f:
        hdr = b""
        while not hdr.endswith(b"end_header\n"):
            hdr += f.readline()
        props = [l.split()[-1].decode() for l in hdr.splitlines() if l.startswith(b"property")]
        n = int([l for l in hdr.splitlines() if l.startswith(b"element vertex")][0].split()[-1])
        off = len(hdr)
    raw = np.memmap(path, dtype=np.float32, mode="r", offset=off, shape=(n, len(props)))
    col = {p: i for i, p in enumerate(props)}
    g = lambda *ks: np.stack([np.asarray(raw[:, col[k]]) for k in ks], -1)
    means = g("x", "y", "z").astype(np.float32)
    scales = np.exp(g("scale_0", "scale_1", "scale_2")).astype(np.float32)
    q = g("rot_0", "rot_1", "rot_2", "rot_3").astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    op = (1.0 / (1.0 + np.exp(-np.asarray(raw[:, col["opacity"]])))).astype(np.float32)
    w, x, y, z = q.T
    R = np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(-1, 3, 3)
    M = R * scales[:, None, :]
    cov3 = M @ M.transpose(0, 2, 1)
    return means, cov3.astype(np.float32), op


def tile_counts(means, cov3, op, c2w, W, H, fov_deg, floor, chunk=1_000_000):
    c2w = np.asarray(c2w, np.float64)
    Rw = c2w[:3, :3].T
    tw = -Rw @ c2w[:3, 3]
    f = 0.5 * max(W, H) / math.tan(0.5 * math.radians(fov_deg))
    cx, cy = W * 0.5, H * 0.5
    tiles_x, tiles_y = (W + TS - 1) // TS, (H + TS - 1) // TS
    hist = np.zeros(tiles_x * tiles_y, np.int64)
    max_r = min(H, W) / 2
    for s in range(0, len(means), chunk):
        mc = means[s:s + chunk].astype(np.float64) @ Rw.T + tw
        o = op[s:s + chunk].astype(np.float64)
        keep = (mc[:, 2] > 0.2) & (o >= 1.0 / 255.0) & (o > floor)
        mc, o, cv = mc[keep], o[keep], cov3[s:s + chunk][keep].astype(np.float64)
        tx, ty, tz = mc.T
        u = f * tx / tz + cx
        v = f * ty / tz + cy
        J = np.zeros((len(tz), 2, 3))
        J[:, 0, 0] = f / tz; J[:, 0, 2] = -f * tx / tz ** 2
        J[:, 1, 1] = f / tz; J[:, 1, 2] = -f * ty / tz ** 2
        T = J @ Rw[None]
        c2 = T @ cv @ T.transpose(0, 2, 1)
        a, b, c = c2[:, 0, 0] + 0.3, c2[:, 0, 1], c2[:, 1, 1] + 0.3
        t = 2.0 * np.log(o / floor)
        rx = np.minimum(np.sqrt(t * a), max_r)
        ry = np.minimum(np.sqrt(t * c), max_r)
        vis = (u + rx > 0) & (u - rx < W) & (v + ry > 0) & (v - ry < H)
        u, v, a, b, c, t, rx, ry = (q[vis] for q in (u, v, a, b, c, t, rx, ry))
        x0 = np.clip(np.floor((u - rx) / TS), 0, tiles_x - 1).astype(np.int64)
        x1 = np.clip(np.floor((u + rx) / TS), 0, tiles_x - 1).astype(np.int64)
        y0 = np.clip(np.floor((v - ry) / TS), 0, tiles_y - 1).astype(np.int64)
        y1 = np.clip(np.floor((v + ry) / TS), 0, tiles_y - 1).astype(np.int64)
        wd, ht = x1 - x0 + 1, y1 - y0 + 1
        nt = wd * ht
        # conic (inverse covariance) for the exact rectangle test
        det = np.maximum(a * c - b * b, 1e-6)
        A, B, C = c / det, -b / det, a / det
        for ps in range(0, len(nt), 200_000):
            sl = slice(ps, ps + 200_000)
            n = nt[sl]
            gi = np.repeat(np.arange(len(n)), n)
            offs = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
            w_ = wd[sl][gi]
            txi = x0[sl][gi] + offs % w_
            tyi = y0[sl][gi] + offs // w_
            # pixel-centre rect relative to the mean
            ulo = txi * TS + 0.5 - u[sl][gi]; uhi = ulo + TS - 1
            vlo = tyi * TS + 0.5 - v[sl][gi]; vhi = vlo + TS - 1
            Ag, Bg, Cg = A[sl][gi], B[sl][gi], C[sl][gi]
            q = lambda du, dv: Ag * du * du + 2 * Bg * du * dv + Cg * dv * dv
            inside = (ulo <= 0) & (uhi >= 0) & (vlo <= 0) & (vhi >= 0)
            m = np.full(len(gi), np.inf)
            for ue in (ulo, uhi):  # vertical edges: minimise over dv
                dv = np.clip(-Bg * ue / Cg, vlo, vhi)
                m = np.minimum(m, q(ue, dv))
            for ve in (vlo, vhi):  # horizontal edges
                du = np.clip(-Bg * ve / Ag, ulo, uhi)
                m = np.minimum(m, q(du, ve))
            m[inside] = 0.0
            kept = m <= t[sl][gi]
            hist += np.bincount((tyi * tiles_x + txi)[kept], minlength=len(hist))
    return hist


def dolly(c2w, dist):
    c = np.array(c2w, np.float64)
    c[:3, 3] += c[:3, 2] * dist  # OpenCV: +Z forward
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--floors", default="255,1024,16384")
    ap.add_argument("--set", default="all")
    args = ap.parse_args()
    cam = json.loads((REPO / "benchmarks/cameras_v2.json").read_text())["bicycle"]
    W, H = cam["image_size"]
    ply = REPO / cam["ply"]
    if not ply.exists():
        ply = Path("/Users/smarton/dev/gsplat_tt/scenes/bicycle.ply")
    means, cov3, op = load_bicycle(ply)
    views = []
    if args.set in ("bench", "all"):
        views += [(k, cam["views"][k]["c2w"], cam["fov_deg"]) for k in cam["order"]]
    if args.set in ("close", "all"):
        hero = cam["views"]["hero"]["c2w"]
        for d in (1.0, 2.0, 2.5, 3.0, 3.5):
            views.append((f"hero_dolly{d}", dolly(hero, d).tolist(), cam["fov_deg"]))
        for fov in (25.0, 12.0):
            views.append((f"hero_fov{fov:g}", hero, fov))
    if args.set in ("far", "all"):
        hero = cam["views"]["hero"]["c2w"]
        for d in (-1.0, -2.0, -4.0):
            views.append((f"hero_dolly{d}", dolly(hero, d).tolist(), cam["fov_deg"]))
        for fov in (70.0, 90.0):
            views.append((f"hero_fov{fov:g}", hero, fov))
    floors = [int(x) for x in args.floors.split(",")]
    print(f"view,floor_inv,max,p99,p50,mean,pairs,tiles_over_{CAP},argmax", flush=True)
    for name, c2w, fov in views:
        for fi in floors:
            h = tile_counts(means, cov3, op, c2w, W, H, fov, 1.0 / fi)
            print(f"{name},{fi},{h.max()},{int(np.percentile(h, 99))},{int(np.median(h))},"
                  f"{h.mean():.0f},{h.sum()},{int((h > CAP).sum())},{int(h.argmax())}",
                  flush=True)


if __name__ == "__main__":
    main()
