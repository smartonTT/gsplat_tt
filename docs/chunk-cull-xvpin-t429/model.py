"""Task #429: CPU-only re-model of the #169 chunk frustum cull under xvpin (no device).

Per bicycle view (benchmarks/cameras_v2.json, 1024x1024, 30 views):
  (a) share of N that the conservative chunk test (skipshare_safe.py, K=1.5) removes, Morton
      order, chunk sizes 256/1024/4096; visible gaussians lost (must be 0);
  (b) kept 1024-chunks per pfwc core (strided deal over C=120 / 110 cores) and the visible
      count per core (writer record work) for the full scene vs the survivor list;
  (d) depth-key ties: visible gaussians with equal fp32 tz bits whose 32x32 (and 16x16) tile
      ranges overlap. For each tied pair the per-tile stable radix keeps emit order, which is the
      compact order of vis_tile::SeqMap: chunk c of the scene goes to core c % C, slot c // C,
      compact order = (core, slot, lane). We compare, per pair, the order under
        base120 / base110: ply order, all chunks, C = 120 / 110 (goldens 39d84b28 / 906e0435),
        gid: ascending original gid (an equal-key gid tie-break),
        mort120: Morton order, survivor chunks dealt strided over 120 cores (the #169 device path).
      A pair whose order differs between two arms flips blend order in a shared tile.
tz is computed in fp32 as ((r20 x + r21 y) + r22 z) + t2; the device's exact op order for the
camera z is not reproduced, so coincidental ties are statistically, not bit-exactly, modelled.
Exact duplicates (equal means) tie under any op order and are counted separately.
Usage (repo root): python3 docs/chunk-cull-xvpin-t429/model.py <bicycle.ply>
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "r16-record"))
from aniso_hero import load_ply  # noqa: E402
from skipshare import CAM, morton_order, visible  # noqa: E402
from skipshare_safe import culled, tables  # noqa: E402

K = 1.5
SIZES = (256, 1024, 4096)
CH = 1024  # pfwc work unit


def compact_rank(chunk_pos, lane, C):
    """Compact order key of (position in the dealt chunk list, lane) under the strided deal."""
    return ((chunk_pos % C) * (1 << 20) + chunk_pos // C) * CH + lane


def main():
    ply = sys.argv[1]
    cam = json.load(open(CAM))["bicycle"]
    W, H = cam["image_size"]
    floor = cam["contrib_floor"]
    f = 0.5 * max(W, H) / math.tan(0.5 * math.radians(cam["fov_deg"]))
    means, scales, q, op, _ = load_ply(ply)
    N = len(means)
    m64 = means.astype(np.float64)
    rho = 3.0 * np.sqrt((scales.astype(np.float64) ** 2).sum(1))
    order = morton_order(m64)               # morton position -> gid
    mpos = np.empty(N, np.int64); mpos[order] = np.arange(N)  # gid -> morton position
    tabs = {C: tables(m64, rho, order, C, K) for C in SIZES}
    nch = (N + CH - 1) // CH
    gid = np.arange(N, dtype=np.int64)
    base = {C: compact_rank(gid // CH, gid % CH, C) for C in (120, 110)}
    print(f"N {N}  chunks {nch}  per-core base ceil(nch/120)={-(-nch // 120)} /110={-(-nch // 110)}")
    hdr = ("view    vis  skip256 skip1024 skip4096 lost kept1024 cmax120 visimb_b visimb_c "
           "tie32 tie16 dup32 b120!=b110 gid!=b120 gid!=b110 m120!=b120 cull_ms")
    print(hdr)
    rows = []
    for v in cam["order"]:
        w2c = np.linalg.inv(np.array(cam["views"][v]["c2w"], np.float64))
        vis = visible(m64, scales, q, op, w2c, W, H, f, floor)   # by gid
        vis_m = vis[order]
        r = {"vis": vis.mean()}
        lost = 0
        for C in SIZES:
            t0 = time.perf_counter()
            cul = culled(*tabs[C], w2c, f, 0.5 * W, 0.5 * H, W, H, K)
            dt = time.perf_counter() - t0
            g = np.repeat(cul, C)[:N]
            r[f"skip{C}"] = g.mean()
            lost += int(vis_m[g].sum())
            if C == CH:
                kept = np.nonzero(~cul)[0]
                r["kept"] = len(kept); r["cull_ms"] = dt * 1e3
                # survivor position of each Morton chunk (-1 = culled)
                spos = np.full(len(cul), -1, np.int64); spos[kept] = np.arange(len(kept))
        r["lost"] = lost
        r["cmax120"] = -(-r["kept"] // 120)
        # visible per core (writer record work): base = ply chunks strided, cull = survivors strided
        vb = np.bincount((gid // CH) % 120, weights=vis, minlength=120)
        mc = mpos // CH
        sp = spos[mc]
        ok = sp >= 0
        vc = np.bincount(sp[ok] % 120, weights=vis[ok], minlength=120)
        r["visimb_b"] = vb.max() / vb.mean(); r["visimb_c"] = vc.max() / vc.mean()
        # ---- ties (visible only) ----
        vi = np.nonzero(vis)[0]
        m32 = means[vi].astype(np.float32)
        R = w2c[:3, :3].astype(np.float32); t = w2c[:3, 3].astype(np.float32)
        tz = ((m32[:, 0] * R[2, 0] + m32[:, 1] * R[2, 1]) + m32[:, 2] * R[2, 2]) + t[2]
        key = tz.view(np.uint32)
        # screen rect (float64, as skipshare.visible) for tile overlap
        pc = m64[vi] @ w2c[:3, :3].T + w2c[:3, 3]
        z = pc[:, 2]
        mx = f * pc[:, 0] / z + 0.5 * W; my = f * pc[:, 1] / z + 0.5 * H
        sm = scales[vi].astype(np.float64).max(1)
        # conservative 3-sigma half-extent in px (max scale, no 0.3 dilation term needed for ties)
        rr = np.ceil(3.0 * np.sqrt((f * sm / z) ** 2 + 0.3))
        srt = np.argsort(key, kind="stable")
        ks = key[srt]
        eq = np.nonzero(ks[1:] == ks[:-1])[0]
        # all pairs inside equal-key runs (runs are short)
        pa, pb = [], []
        if len(eq):
            idx = 0
            while idx < len(eq):
                s = eq[idx]; e = s + 1
                while idx + 1 < len(eq) and eq[idx + 1] == e:
                    idx += 1; e += 1
                members = srt[s:e + 1]
                for x in range(len(members)):
                    for y in range(x + 1, len(members)):
                        pa.append(members[x]); pb.append(members[y])
                idx += 1
        pa = np.array(pa, np.int64); pb = np.array(pb, np.int64)
        for T in (32, 16):
            if len(pa) == 0:
                r[f"tie{T}"] = 0; continue
            def rng(a):
                return (np.clip(np.floor((mx[a] - rr[a]) / T), 0, W // T - 1),
                        np.clip(np.floor((mx[a] + rr[a]) / T), 0, W // T - 1),
                        np.clip(np.floor((my[a] - rr[a]) / T), 0, H // T - 1),
                        np.clip(np.floor((my[a] + rr[a]) / T), 0, H // T - 1))
            ax0, ax1, ay0, ay1 = rng(pa); bx0, bx1, by0, by1 = rng(pb)
            ov = (ax0 <= bx1) & (bx0 <= ax1) & (ay0 <= by1) & (by0 <= ay1)
            r[f"tie{T}"] = int(ov.sum())
            if T == 32:
                ga, gb = vi[pa[ov]], vi[pb[ov]]
                r["dup32"] = int((means[ga] == means[gb]).all(1).sum())
                o = {k_: np.sign(b_[ga] - b_[gb]) for k_, b_ in base.items()}
                og = np.sign(ga - gb)
                cm = mpos[ga] // CH; cmb = mpos[gb] // CH
                ra = compact_rank(spos[cm], mpos[ga] % CH, 120)
                rb = compact_rank(spos[cmb], mpos[gb] % CH, 120)
                om = np.sign(ra - rb)
                r["b120!=b110"] = int((o[120] != o[110]).sum())
                r["gid!=b120"] = int((og != o[120]).sum())
                r["gid!=b110"] = int((og != o[110]).sum())
                r["m120!=b120"] = int((om != o[120]).sum())
                same_chunk = (ga // CH) == (gb // CH)
                r["same_chunk"] = int(same_chunk.sum())
        for k_ in ("dup32", "b120!=b110", "gid!=b120", "gid!=b110", "m120!=b120", "same_chunk"):
            r.setdefault(k_, 0)
        rows.append(r)
        print(f"{v:7s} {r['vis']:.3f} {r['skip256']:.3f} {r['skip1024']:.3f} {r['skip4096']:.3f} "
              f"{r['lost']:4d} {r['kept']:6d} {r['cmax120']:5d} {r['visimb_b']:.3f} {r['visimb_c']:.3f} "
              f"{r['tie32']:5d} {r['tie16']:5d} {r['dup32']:5d} {r['b120!=b110']:5d} {r['gid!=b120']:5d} "
              f"{r['gid!=b110']:5d} {r['m120!=b120']:5d} {r['cull_ms']:.2f}", flush=True)
    keys = [k_ for k_ in rows[0] if k_ != "lost"]
    mean = {k_: float(np.mean([r[k_] for r in rows])) for k_ in keys}
    mean["lost_total"] = int(sum(r["lost"] for r in rows))
    mean["views_b120!=b110"] = int(sum(r["b120!=b110"] > 0 for r in rows))
    mean["views_gid!=b120"] = int(sum(r["gid!=b120"] > 0 for r in rows))
    mean["views_gid!=b110"] = int(sum(r["gid!=b110"] > 0 for r in rows))
    mean["views_tie32"] = int(sum(r["tie32"] > 0 for r in rows))
    print("MEAN " + json.dumps({k_: round(x, 4) for k_, x in mean.items()}))


if __name__ == "__main__":
    main()
