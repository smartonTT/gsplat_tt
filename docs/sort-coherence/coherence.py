#!/usr/bin/env python3
"""t529: cross-view sort coherence study on the 30 bicycle views (CPU only).

Mirrors the CPU reference pipeline (gsplat/rasterization.py: EWA project, opacity-aware
3-sigma AABB, 32x32 tiles, per-pair contribution cull, per-tile depth sort) in numpy, then:

  coherence  consecutive views (b2b order hero, orb_00..orb_28, wrap): pairs kept, tiles with an
             identical list, inversions of the previous view's order on the common pairs.
  presort    per view: inversions / max displacement when each tile's list is in the order of a
             view-independent key d.x (d = camera forward is exact), for several choices of d.
  smallstep  viewer-like motion: views interpolated between orb_14 and orb_15 (fractions of the
             1.9 deg step), coherence of orb_14 -> each, plus the prev-frame forward presort.
  psnr       blend chosen orders with sc_native.c and compare with the exact order and with
             benchmarks/reference_v2/hero.png.

Usage: python coherence.py {coherence,presort,psnr,smallstep,all} [--out DIR]
Needs numpy, Pillow and sc_native.so (cc -O3 -shared -fPIC -o sc_native.so sc_native.c).
"""
import argparse
import ctypes
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
TILE = 32
NEAR = 0.2
C0 = 0.28209479177387814


def find_ply(cam_ply):
    for base in (REPO, Path.home() / "dev/gstt2", Path.home() / "dev/gsplat_tt"):
        p = base / cam_ply
        if p.exists():
            return p
    sys.exit(f"scene {cam_ply} not found")


def load_scene(path):
    with open(path, "rb") as f:
        hdr = b""
        while not hdr.endswith(b"end_header\n"):
            hdr += f.readline()
    lines = hdr.decode().splitlines()
    n = int([l for l in lines if l.startswith("element vertex")][0].split()[-1])
    props = [l.split()[-1] for l in lines if l.startswith("property")]
    cache = Path("/tmp/t529_scene.npz")
    if cache.exists():
        z = np.load(cache)
        return dict(xyz=z["xyz"], rgb=z["rgb"], op=z["op"], cov3=z["cov3"], n=n)
    raw = np.fromfile(path, dtype=np.float32, offset=len(hdr)).reshape(n, len(props))
    col = {p: i for i, p in enumerate(props)}
    g = lambda *names: np.ascontiguousarray(raw[:, [col[x] for x in names]])
    xyz = g("x", "y", "z")
    rgb = np.clip(0.5 + C0 * g("f_dc_0", "f_dc_1", "f_dc_2"), 0.0, 1.0).astype(np.float32)
    op = (1.0 / (1.0 + np.exp(-g("opacity")[:, 0]))).astype(np.float32)
    s = np.exp(g("scale_0", "scale_1", "scale_2"))
    q = g("rot_0", "rot_1", "rot_2", "rot_3")
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    R = np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1).reshape(-1, 3, 3)
    M = R * s[:, None, :]
    cov3 = np.einsum("nij,nkj->nik", M, M).astype(np.float32)  # scene-static (option c)
    np.savez(cache, xyz=xyz, rgb=rgb, op=op, cov3=cov3)
    return dict(xyz=xyz, rgb=rgb, op=op, cov3=cov3, n=n)


def cameras():
    cam = json.load(open(REPO / "benchmarks/cameras_v2.json"))["bicycle"]
    W, H = cam["image_size"]
    f = 0.5 * max(W, H) / math.tan(0.5 * math.radians(cam["fov_deg"]))
    views = []
    for name in cam["order"]:
        c2w = np.asarray(cam["views"][name]["c2w"], np.float64)
        Rc, tc = c2w[:3, :3], c2w[:3, 3]
        views.append(dict(name=name, R=Rc.T, t=-Rc.T @ tc, pos=tc))
    return cam, W, H, f, views


def project_and_bin(sc, v, W, H, f, floor):
    """Visible gaussians and their (tile, gaussian) pairs, sorted by (tile, depth, gid)."""
    R, t = v["R"].astype(np.float32), v["t"].astype(np.float32)
    mc = sc["xyz"] @ R.T + t
    z = mc[:, 2]
    vis = np.nonzero((z > NEAR) & (sc["op"] >= 1.0 / 255.0))[0]
    mc, z = mc[vis], z[vis]
    tx, ty = mc[:, 0], mc[:, 1]
    cx, cy = W * 0.5, H * 0.5
    mx, my = f * tx / z + cx, f * ty / z + cy
    J = np.zeros((len(vis), 2, 3), np.float32)
    J[:, 0, 0] = f / z
    J[:, 0, 2] = -f * tx / (z * z)
    J[:, 1, 1] = f / z
    J[:, 1, 2] = -f * ty / (z * z)
    T = J @ R  # (n,2,3)
    c2 = np.einsum("nij,njk,nlk->nil", T, sc["cov3"][vis], T)
    a, b, c = c2[:, 0, 0] + 0.3, c2[:, 0, 1], c2[:, 1, 1] + 0.3
    op = sc["op"][vis]
    k = np.minimum(np.sqrt(2.0 * np.log(np.maximum(op / floor, 1.0))), 3.0)
    rx, ry = np.ceil(k * np.sqrt(np.maximum(a, 0))), np.ceil(k * np.sqrt(np.maximum(c, 0)))
    ok = (mx + rx > 0) & (mx - rx < W) & (my + ry > 0) & (my - ry < H) & (rx > 0) & (ry > 0)
    ok &= (rx <= min(W, H) // 2) & (ry <= min(W, H) // 2)
    sel = np.nonzero(ok)[0]
    vis, mx, my, a, b, c, z, op, rx, ry = (x[sel] for x in (vis, mx, my, a, b, c, z, op, rx, ry))
    tiles_x, tiles_y = (W + TILE - 1) // TILE, (H + TILE - 1) // TILE
    x0 = np.clip((mx - rx) / TILE, 0, tiles_x - 1).astype(np.int32)
    x1 = np.clip((mx + rx) / TILE, 0, tiles_x - 1).astype(np.int32)
    y0 = np.clip((my - ry) / TILE, 0, tiles_y - 1).astype(np.int32)
    y1 = np.clip((my + ry) / TILE, 0, tiles_y - 1).astype(np.int32)
    wdt = x1 - x0 + 1
    cnt = wdt * (y1 - y0 + 1)
    gid = np.repeat(np.arange(len(vis), dtype=np.int32), cnt)
    start = np.repeat(np.cumsum(cnt) - cnt, cnt)
    o = np.arange(len(gid)) - start
    tile = ((y0[gid] + o // wdt[gid]) * tiles_x + x0[gid] + o % wdt[gid]).astype(np.int32)
    # Per-pair contribution cull (closest point of the tile to the centre).
    pa, pb, pc = a[gid], b[gid], c[gid]
    det = np.maximum(pa * pc - pb * pb, 1e-6)
    u_lo = (tile % tiles_x).astype(np.float32) * TILE - mx[gid]
    v_lo = (tile // tiles_x).astype(np.float32) * TILE - my[gid]
    u_hi, v_hi = u_lo + TILE, v_lo + TILE
    xin, yin = (u_lo <= 0) & (u_hi >= 0), (v_lo <= 0) & (v_hi >= 0)
    uf = np.where(u_lo > 0, u_lo, u_hi)
    vs = np.clip(pb * uf / np.maximum(pa, 1e-12), v_lo, v_hi)
    m_v = np.where(xin, np.inf, pc * uf * uf - 2 * pb * uf * vs + pa * vs * vs)
    vf = np.where(v_lo > 0, v_lo, v_hi)
    us = np.clip(pb * vf / np.maximum(pc, 1e-12), u_lo, u_hi)
    m_h = np.where(yin, np.inf, pc * us * us - 2 * pb * us * vf + pa * vf * vf)
    m2 = np.where(xin & yin, 0.0, np.minimum(m_v, m_h)) / det
    keep = op[gid] * np.exp(-0.5 * m2) >= floor
    gid, tile = gid[keep], tile[keep]
    order = np.lexsort((z[gid], tile))  # stable: equal depths keep gid order
    gid, tile = gid[order], tile[order]
    off = np.zeros(tiles_x * tiles_y + 1, np.int64)
    off[1:] = np.cumsum(np.bincount(tile, minlength=tiles_x * tiles_y))
    det = np.maximum(a * c - b * b, 1e-6)
    return dict(vis=vis, gid=gid, tile=tile, off=off, mx=mx, my=my, z=z, op=op,
                ca=c / det, cb=-b / det, cc=a / det, ntiles=tiles_x * tiles_y, tiles_x=tiles_x)


_lib = None


def lib():
    global _lib
    if _lib is None:
        so = HERE / "sc_native.so"
        _lib = ctypes.CDLL(str(so))
        P = ctypes.c_void_p
        _lib.tile_inversions.argtypes = [ctypes.c_int64, P, P, P, P]
        _lib.blend.argtypes = [ctypes.c_int64] * 4 + [P] * 10 + [ctypes.c_int]
    return _lib


def ptr(a):
    return a.ctypes.data_as(ctypes.c_void_p)


def tile_inv(off, keys):
    keys = np.ascontiguousarray(keys, np.float32)
    off = np.ascontiguousarray(off, np.int64)
    nt = len(off) - 1
    inv, md = np.zeros(nt, np.int64), np.zeros(nt, np.int64)
    lib().tile_inversions(nt, ptr(off), ptr(keys), ptr(inv), ptr(md))
    return inv, md


def inv_summary(off, inv, md):
    n = np.diff(off)
    pairs_total = n.sum()
    max_inv = (n * (n - 1) // 2).sum()
    return dict(inv_per_pair=float(inv.sum() / max(pairs_total, 1)),
                kendall_dist=float(inv.sum() / max(max_inv, 1)),
                tiles_sorted=float(((inv == 0) | (n < 2)).sum() / max((n > 0).sum(), 1)),
                maxdisp_p50=int(np.median(md[n > 0])) if (n > 0).any() else 0,
                maxdisp_p99=int(np.percentile(md[n > 0], 99)) if (n > 0).any() else 0,
                maxdisp_max=int(md.max()))


def blend_img(p, sc, order_gid, W, H, threads=12):
    img = np.zeros((H, W, 3), np.float32)
    rgb = np.ascontiguousarray(sc["rgb"][p["vis"]])
    arr = [np.ascontiguousarray(p[k], np.float32) for k in ("mx", "my", "ca", "cb", "cc", "op")]
    order_gid = np.ascontiguousarray(order_gid, np.int32)  # keep alive across the call
    lib().blend(p["ntiles"], p["tiles_x"], W, H, ptr(p["off"]), ptr(order_gid),
                *[ptr(x) for x in arr], ptr(rgb), ptr(img), threads)
    return img


def u8(img):
    return (np.clip(img, 0, 1) * 255.0).astype(np.uint8)


def psnr8(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2) / 255.0 ** 2
    return 100.0 if mse <= 0 else min(100.0, -10 * math.log10(mse))


def fib_dirs(k):
    i = np.arange(k) + 0.5
    phi = np.arccos(1 - 2 * i / k)
    th = math.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)


def rot_about(v, axis, deg):
    axis = axis / np.linalg.norm(axis)
    th = math.radians(deg)
    return v * math.cos(th) + np.cross(axis, v) * math.sin(th) + axis * (axis @ v) * (1 - math.cos(th))


def presort_keys(p, sc, d):
    """Key d.x for each pair, pairs in exact order (inversions = disorder of the d order)."""
    g = p["vis"][p["gid"]]
    return (sc["xyz"][g] @ d.astype(np.float32)).astype(np.float32)


def reorder_by_key(p, keys):
    """Each tile's list ordered by key (stable); returns gids (indices into p's visible set)."""
    o = np.lexsort((keys, p["tile"]))
    return p["gid"][o]


def run(mode, out):
    out.mkdir(parents=True, exist_ok=True)
    cam, W, H, f, views = cameras()
    floor = cam["contrib_floor"]
    t0 = time.time()
    sc = load_scene(find_ply(cam["ply"]))
    print(f"scene n={sc['n']} load {time.time() - t0:.1f}s", flush=True)
    fwd = np.array([v["R"][2] for v in views])  # camera forward (world), row 3 of w2c R
    steps = [math.degrees(math.acos(np.clip(fwd[i] @ fwd[(i + 1) % len(views)], -1, 1)))
             for i in range(len(views))]
    poss = [float(np.linalg.norm(views[i]["pos"] - views[(i + 1) % len(views)]["pos"]))
            for i in range(len(views))]
    res = dict(view_names=[v["name"] for v in views], fwd_step_deg=steps, pos_step=poss)
    print("forward-direction step between consecutive views (deg):",
          " ".join(f"{s:.1f}" for s in steps), flush=True)

    if mode == "smallstep":
        a, b = views[[v["name"] for v in views].index("orb_14")], views[[v["name"] for v in views].index("orb_15")]
        A = project_and_bin(sc, a, W, H, f, floor)
        rows = []
        for frac in (1 / 32, 1 / 8, 1 / 4, 1 / 2, 1.0):
            v = interp_view(a, b, frac)
            B = project_and_bin(sc, v, W, H, f, floor)
            c = compare(A, B, sc)
            inv, md = tile_inv(B["off"], presort_keys(B, sc, a["R"][2]))
            c.update(frac=frac, step_deg=frac * steps[[x["name"] for x in views].index("orb_14")],
                     prev_fwd_presort=inv_summary(B["off"], inv, md))
            rows.append(c)
            print(json.dumps(c), flush=True)
        json.dump(rows, open(out / "smallstep.json", "w"), indent=1)
        return

    if mode in ("coherence", "presort", "all"):
        prev = None
        coh, pre = [], []
        dirs = {k: fib_dirs(k) for k in (26, 128, 512, 2048)}
        for i, v in enumerate(views + [views[0]]):
            t1 = time.time()
            p = project_and_bin(sc, v, W, H, f, floor)
            P = len(p["gid"])
            if i < len(views):
                # Sanity: forward-direction key must give zero inversions (depth = fwd.x + t_z).
                r = dict(view=v["name"], visible=int(len(p["vis"])), pairs=int(P),
                         nonempty_tiles=int((np.diff(p["off"]) > 0).sum()),
                         max_tile=int(np.diff(p["off"]).max()))
                inv, md = tile_inv(p["off"], presort_keys(p, sc, v["R"][2]))
                r["exact_fwd_inv"] = int(inv.sum())
                zz = p["z"][p["gid"]]
                r["equal_depth_neighbours"] = int(((zz[1:] == zz[:-1]) & (p["tile"][1:] == p["tile"][:-1])).sum())
                cand = {}
                for kd, D in dirs.items():
                    j = int(np.argmax(D @ v["R"][2]))
                    ang = math.degrees(math.acos(np.clip(D[j] @ v["R"][2], -1, 1)))
                    inv, md = tile_inv(p["off"], presort_keys(p, sc, D[j]))
                    cand[f"fib{kd}"] = dict(angle_deg=ang, **inv_summary(p["off"], inv, md))
                axis = v["R"][0] + v["R"][1]
                for deg in (0.25, 1.0, 4.0, 16.0):
                    d = rot_about(v["R"][2], axis, deg)
                    inv, md = tile_inv(p["off"], presort_keys(p, sc, d))
                    cand[f"rot{deg}"] = dict(angle_deg=deg, **inv_summary(p["off"], inv, md))
                if i > 0:
                    d = views[i - 1]["R"][2]
                    inv, md = tile_inv(p["off"], presort_keys(p, sc, d))
                    cand["prev_view_fwd"] = dict(angle_deg=steps[i - 1], **inv_summary(p["off"], inv, md))
                # A fixed global axis: world up of the orbit (worst case for a single presort).
                r["presort"] = cand
                pre.append(r)
            if prev is not None:
                coh.append(dict(pair=f"{prev['name']}->{v['name']}", **compare(prev["p"], p, sc)))
                c = coh[-1]
                print(f"  {c['pair']}: kept {c['kept_frac_of_new']:.3f} tiles_identical "
                      f"{c['tiles_identical']:.3f} inv/pair {c['common_inv_per_pair']:.3f}", flush=True)
            prev = dict(name=v["name"], p=dict(vis=p["vis"], gid=p["gid"], tile=p["tile"], off=p["off"]))
            if i < len(views):
                c = pre[-1]["presort"]
                print(f"{v['name']}: pairs {P} vis {len(p['vis'])} fwd_inv {pre[-1]['exact_fwd_inv']} "
                      + " ".join(f"{k}:{x['angle_deg']:.1f}deg/{x['inv_per_pair']:.4f}" for k, x in c.items())
                      + f" [{time.time() - t1:.1f}s]", flush=True)
        res["coherence"], res["presort"] = coh, pre
        json.dump(res, open(out / "coherence.json", "w"), indent=1)

    if mode in ("psnr", "all"):
        from PIL import Image
        ref = np.asarray(Image.open(REPO / "benchmarks/reference_v2/hero.png").convert("RGB"))
        rows = []
        for vi in (0, 15):
            v = views[vi]
            p = project_and_bin(sc, v, W, H, f, floor)
            t1 = time.time()
            exact = u8(blend_img(p, sc, p["gid"], W, H))
            Image.fromarray(exact).save(out / f"{v['name']}_cpu_exact.png")
            row = dict(view=v["name"], blend_s=time.time() - t1)
            if vi == 0:
                row["exact_vs_ref"] = psnr8(exact, ref)
            cands = {f"rot{d}": rot_about(v["R"][2], v["R"][0] + v["R"][1], d)
                     for d in ((0.25, 1.0, 4.0, 16.0) if vi == 0 else (1.0, 4.0))}
            for kd in ((26, 128) if vi == 0 else ()):
                D = fib_dirs(kd)
                cands[f"fib{kd}"] = D[int(np.argmax(D @ v["R"][2]))]
            cands["prev_view_fwd"] = views[vi - 1]["R"][2]
            for name, d in cands.items():
                img = u8(blend_img(p, sc, reorder_by_key(p, presort_keys(p, sc, d)), W, H))
                row[name] = dict(vs_exact=psnr8(img, exact), identical=bool((img == exact).all()),
                                 diff_px=int((img != exact).any(-1).sum()))
                if vi == 0:
                    row[name]["vs_ref"] = psnr8(img, ref)
                    if name in ("rot4.0", "rot16.0", "fib26"):
                        Image.fromarray(img).save(out / f"hero_{name}.png")
                        Image.fromarray(np.clip(np.abs(img.astype(int) - exact.astype(int)) * 10, 0, 255)
                                        .astype(np.uint8)).save(out / f"hero_{name}_diff10.png")
            print(json.dumps(row), flush=True)
            rows.append(row)
        json.dump(rows, open(out / "psnr.json", "w"), indent=1)


def interp_view(a, b, frac):
    """Camera between views a and b: lerp position, slerp-free rotation via orthonormalised lerp."""
    pos = (1 - frac) * a["pos"] + frac * b["pos"]
    Rc = (1 - frac) * a["R"].T + frac * b["R"].T  # c2w rotation
    u, _, vt = np.linalg.svd(Rc)
    Rc = u @ vt
    return dict(name=f"interp{frac:g}", R=Rc.T, t=-Rc.T @ pos, pos=pos)


def compare(A, B, sc):
    """Coherence of view A -> view B (pairs keyed by scene gaussian id)."""
    N = np.int64(sc["n"])
    ka = A["tile"].astype(np.int64) * N + A["vis"][A["gid"]]
    kb = B["tile"].astype(np.int64) * N + B["vis"][B["gid"]]
    # Position of each A pair inside its tile's exact order.
    rank_a = np.arange(len(ka)) - A["off"][A["tile"]]
    sa = np.argsort(ka, kind="stable")
    pos = np.searchsorted(ka[sa], kb)
    pos = np.minimum(pos, len(ka) - 1)
    common = ka[sa][pos] == kb
    nt = len(B["off"]) - 1
    na, nb = np.diff(A["off"]), np.diff(B["off"])
    # Identical tile lists: same length and same gid at each position.
    same_len = na[B["tile"]] == nb[B["tile"]]
    rb = np.arange(len(kb)) - B["off"][B["tile"]]
    ia = np.minimum(A["off"][B["tile"]] + rb, len(ka) - 1)
    eq = same_len & (ka[ia] == kb)
    mism = np.bincount(B["tile"], weights=~eq, minlength=nt)
    nonempty = (nb > 0) | (na > 0)
    ident = ((mism == 0) & (na == nb) & nonempty).sum() / max(nonempty.sum(), 1)
    # Inversions of A's order among B's common pairs, walked in B's exact order.
    tc = B["tile"][common]
    off_c = np.zeros(nt + 1, np.int64)
    off_c[1:] = np.cumsum(np.bincount(tc, minlength=nt))
    keys = rank_a[sa][pos][common].astype(np.float32)
    inv, md = tile_inv(off_c, keys)
    s = inv_summary(off_c, inv, md)
    return dict(pairs_a=int(len(ka)), pairs_b=int(len(kb)), common=int(common.sum()),
                kept_frac_of_new=float(common.mean()), kept_frac_of_old=float(common.sum() / len(ka)),
                tiles_identical=float(ident), common_inv_per_pair=s["inv_per_pair"],
                common_kendall=s["kendall_dist"], common_tiles_sorted=s["tiles_sorted"],
                common_maxdisp_p99=s["maxdisp_p99"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["coherence", "presort", "psnr", "smallstep", "all"])
    ap.add_argument("--out", type=Path, default=HERE / "out")
    a = ap.parse_args()
    run(a.mode, a.out)
