"""Task #616: per-tile pair counts and per-chip visible counts for a 2-chip screen split.

Host-only numpy model (t270's tilecount.py: the level-set o*exp(-d^2/2) >= 1/255 vs the
tile's pixel-centre rectangle), run on the 30 bicycle bench views. For each view it saves the
per-tile pair histogram (32x32 tiles) and, for each candidate partition of the tiles over 2
chips, the number of gaussians with >= 1 kept pair on each chip (M_c, what a pfwc that culls to
its chip's tiles would still write) and with a bbox touching each chip (M_c bbox).

Usage: python3 tiles616.py [out.npz]   (about 10-20 s per view on an M-series Mac)
"""
import json, math, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tile-overflow-t270"))
from tilecount import load_bicycle  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
TS, FLOOR = 32, 1.0 / 255.0


def partitions(tx, ty):
    """chip id (0/1) of tile (tx, ty) for each static partition."""
    return {
        "top_bottom": (ty >= 16),
        "left_right": (tx >= 16),
        "rows1": (ty & 1),
        "rows2": ((ty >> 1) & 1),
        "rows4": ((ty >> 2) & 1),
        "cols1": (tx & 1),
        "cols2": ((tx >> 1) & 1),
        "checker": ((tx + ty) & 1),
        "checker2": (((tx >> 1) + (ty >> 1)) & 1),
    }


def view(means, cov3, op, c2w, W, H, fov_deg, chunk=1_000_000):
    c2w = np.asarray(c2w, np.float64)
    Rw = c2w[:3, :3].T
    tw = -Rw @ c2w[:3, 3]
    f = 0.5 * max(W, H) / math.tan(0.5 * math.radians(fov_deg))
    cx, cy = W * 0.5, H * 0.5
    nx, ny = (W + TS - 1) // TS, (H + TS - 1) // TS
    hist = np.zeros(nx * ny, np.int64)
    max_r = min(H, W) / 2
    names = list(partitions(np.zeros(1, np.int64), np.zeros(1, np.int64)).keys())
    m_pair = {k: [0, 0] for k in names}
    m_bbox = {k: [0, 0] for k in names}
    m_vis = m_any = 0
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
        a, b, c = c2[:, 0, 0] + 0.3, c2[:, 0, 1], c2[:, 1, 1] + 0.3
        t = 2.0 * np.log(o / FLOOR)
        rx = np.minimum(np.sqrt(t * a), max_r)
        ry = np.minimum(np.sqrt(t * c), max_r)
        vis = (u + rx > 0) & (u - rx < W) & (v + ry > 0) & (v - ry < H)
        u, v, a, b, c, t, rx, ry = (q[vis] for q in (u, v, a, b, c, t, rx, ry))
        m_vis += len(u)
        x0 = np.clip(np.floor((u - rx) / TS), 0, nx - 1).astype(np.int64)
        x1 = np.clip(np.floor((u + rx) / TS), 0, nx - 1).astype(np.int64)
        y0 = np.clip(np.floor((v - ry) / TS), 0, ny - 1).astype(np.int64)
        y1 = np.clip(np.floor((v + ry) / TS), 0, ny - 1).astype(np.int64)
        wd, ht = x1 - x0 + 1, y1 - y0 + 1
        nt = wd * ht
        det = np.maximum(a * c - b * b, 1e-6)
        A, B, C = c / det, -b / det, a / det
        for ps in range(0, len(nt), 200_000):
            sl = slice(ps, ps + 200_000)
            n = nt[sl]
            ng = len(n)
            gi = np.repeat(np.arange(ng), n)
            offs = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
            w_ = wd[sl][gi]
            txi = x0[sl][gi] + offs % w_
            tyi = y0[sl][gi] + offs // w_
            ulo = txi * TS + 0.5 - u[sl][gi]; uhi = ulo + TS - 1
            vlo = tyi * TS + 0.5 - v[sl][gi]; vhi = vlo + TS - 1
            Ag, Bg, Cg = A[sl][gi], B[sl][gi], C[sl][gi]
            q = lambda du, dv: Ag * du * du + 2 * Bg * du * dv + Cg * dv * dv
            inside = (ulo <= 0) & (uhi >= 0) & (vlo <= 0) & (vhi >= 0)
            m = np.full(len(gi), np.inf)
            for ue in (ulo, uhi):
                dv = np.clip(-Bg * ue / Cg, vlo, vhi)
                m = np.minimum(m, q(ue, dv))
            for ve in (vlo, vhi):
                du = np.clip(-Bg * ve / Ag, ulo, uhi)
                m = np.minimum(m, q(du, ve))
            m[inside] = 0.0
            kept = m <= t[sl][gi]
            hist += np.bincount((tyi * nx + txi)[kept], minlength=len(hist))
            anyk = np.zeros(ng, bool); anyk[gi[kept]] = True
            m_any += int(anyk.sum())
            for k, chip in partitions(txi, tyi).items():
                for cc in (0, 1):
                    on = chip == cc
                    fl = np.zeros(ng, bool); fl[gi[kept & on]] = True
                    m_pair[k][cc] += int(fl.sum())
                    fb = np.zeros(ng, bool); fb[gi[on]] = True
                    m_bbox[k][cc] += int(fb.sum())
    return hist, m_vis, m_any, m_pair, m_bbox


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("tiles616.npz")
    cam = json.loads((REPO / "benchmarks/cameras_v2.json").read_text())["bicycle"]
    W, H = cam["image_size"]
    ply = REPO / cam["ply"]
    if not ply.exists():
        ply = Path("/Users/smarton/dev/gsplat_tt/scenes/bicycle.ply")
    means, cov3, op = load_bicycle(ply)
    print(f"N={len(means)} kept_by_opacity={(op >= FLOOR).sum()}", flush=True)
    res = {"order": np.array(cam["order"])}
    hists, rows = [], []
    for name in cam["order"]:
        t0 = time.time()
        h, mv, ma, mp, mb = view(means, cov3, op, cam["views"][name]["c2w"], W, H, cam["fov_deg"])
        hists.append(h)
        rows.append({"view": name, "M_vis": mv, "M_any": ma, "P": int(h.sum()),
                     "m_pair": mp, "m_bbox": mb})
        print(json.dumps(rows[-1]), f"{time.time() - t0:.1f}s", flush=True)
    res["hist"] = np.stack(hists)
    np.savez_compressed(out, **res)
    out.with_suffix(".json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
