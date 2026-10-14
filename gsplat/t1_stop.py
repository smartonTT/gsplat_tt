"""T1 per-tile stop at the last contributor (#802, docs/training-plan/t799/PLAN.md lever 2).

T1 runs a tile's pairs front to back and a pair adds to its slot only at pixels where it is live
(alpha >= alpha_min and T (1 - alpha) >= t_min, blend_backward_dev.emulate). Past the last pair
any pixel of the tile used (3DGS last_contributor, tile max), every pair's sums are exactly zero,
so its slot is the finish of zero sums. Stopping the tile there is lossless: kept pairs see the
same T and BH (both depend only on earlier pairs), dropped pairs get the same zero slot.

* pixel_stops(p): per pixel, 1 + the local index of the last pair that may be live there (0 =
  none), from a host replay of the forward T plane on the prep (bench: from dumps). With margin
  (default) alpha and T carry bounds wide enough for the device's own exp and fp32 rounding, so
  the stop is never earlier than the device's; margin 0 is the exact emulate() rule. The device
  trainer takes the same plane from the forward (#788 per-pixel stop): stops_from_pixel().
* truncate(p, stops): rec_count of every tile row cut to its stop (the reader, compute and writer
  take the count from the tile header, so the device runs each tile only up to it; rows with
  stop 0 are dropped). The dropped pairs' slots go to dead_slots / dead_vals: the bench writes
  them into the slot buffer before the runs (render/train/blend_bwd_bench.cpp). LPT plans built on
  the truncated prep (t1_balance.plan, t1_balance_dev plan) weight rows by truncated counts.

GSPLAT_TT_T1_STOP=1 (default 0 = off) makes `blend_backward_dev prep` truncate.
  python3 -m gsplat.t1_stop <prep_dir> <out_dir> [--margin 0]   (truncates a prep dir)
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys

import numpy as np

from gsplat.blend_backward_dev import BATCH, N_PLANES, vector_lanes
from gsplat.blend_backward_ref import BlendCfg, centred_power

f32 = np.float32
ENV = "GSPLAT_TT_T1_STOP"
ALPHA_MARGIN = 1e-4     # relative: device exp/power vs numpy (fp32-accurate exp, ~1e-6)
T_MARGIN = 2e-3         # relative: fp32 rounding of T over long tile lists (<= 15k pairs x 2^-24 x 2)


def enabled() -> bool:
    return os.environ.get(ENV, "0") not in ("", "0")


def stops_from_live(tile_of: np.ndarray, live: np.ndarray, n_tiles: int | None = None) -> np.ndarray:
    """Pairs kept per tile: 1 + local index of the tile's last pair with live > 0 (pairs in tile
    order, tile_of non-decreasing)."""
    t = np.asarray(tile_of, np.int64)
    n = int(t.max()) + 1 if n_tiles is None else n_tiles
    start = np.searchsorted(t, np.arange(n))
    pos = np.arange(t.size, dtype=np.int64) - start[t] + 1
    return np.maximum.reduceat(np.where(np.asarray(live) > 0, pos, 0), start) * (np.bincount(t, minlength=n) > 0) \
        if t.size else np.zeros(n, np.int64)


def stops_from_pixel(pixel_stop: np.ndarray) -> np.ndarray:
    """Tile stop = max over the tile's pixels of the per-pixel stop (n_tiles, ...) plane."""
    ps = np.asarray(pixel_stop)
    return ps.reshape(ps.shape[0], -1).max(1).astype(np.int64)


def pixel_stops(p: dict, margin: bool = True) -> np.ndarray:
    """(n_tiles, 32 vectors, 32 lanes) per-pixel stop of a prep dict, device rules (emulate()).
    Tiles are swept together by pair rank. With margin, per pixel T_hi bounds the device T given
    the pixel is still not done: a pair that surely passes alpha_min either is live (T drops) or
    ends the pixel, so on the not-done branch T_hi drops by it. A pair is counted unless T_hi and
    the alpha bounds rule out its being live, and the pixel is done once a pair surely stops it
    (after that the device adds nothing)."""
    cfg = p["cfg"]
    da, dt = (ALPHA_MARGIN, T_MARGIN) if margin else (0.0, 0.0)
    lanes = vector_lanes().reshape(32, 32)
    du, dv = p["ramps"][0][lanes] - f32(16), p["ramps"][1][lanes] - f32(16)
    tiles = np.asarray(p["tiles"], np.int64)
    nt = len(tiles)
    o = np.argsort(-tiles[:, 2], kind="stable")
    start, cnt = tiles[o, 1], tiles[o, 2]
    T0 = np.asarray(p["planes"], f32).reshape(nt, N_PLANES, 1024)[o, 0][:, lanes].astype(np.float64)
    Thi = T0.copy()
    done = np.zeros(T0.shape, bool)
    stop = np.zeros(T0.shape, np.int64)
    amax, amin, tmin = float(f32(cfg.alpha_max)), float(f32(cfg.alpha_min)), float(f32(cfg.t_min))
    bits = (np.uint32(1) << np.arange(32, dtype=np.uint32))[None, :, None]
    for k in range(int(cnt.max()) if nt else 0):
        n = int(np.searchsorted(-cnt, -k, side="left"))   # tiles with cnt > k
        rec = p["recs"][start[:n] + k]
        fr = rec[:, :12].view(f32)
        Af, Bf, Cf, op, P0, Lx, Ly = (fr[:, i][:, None, None] for i in (0, 1, 2, 5, 9, 10, 11))
        pw = centred_power(Af, Bf, Cf, P0, Lx, Ly, du, dv)
        al = (op * np.exp(np.minimum(pw, f32(0)))).astype(np.float64)
        alo, ahi = np.minimum(al * (1 - da), amax), np.minimum(al * (1 + da), amax)
        inm = (rec[:, 12][:, None, None] & bits) != 0
        th, dn = Thi[:n], done[:n]
        may_live = inm & ~dn & (ahi >= amin) & (th * (1 - alo) >= tmin * (1 - dt))
        dn |= inm & (alo >= amin) & (th * (1 - alo) < tmin * (1 - dt))
        Thi[:n] = np.where(inm & (alo >= amin), th * (1 - alo), th)
        stop[:n] = np.where(may_live, k + 1, stop[:n])
    out = np.empty_like(stop)
    out[o] = stop
    return out


def tile_stops(p: dict, margin: bool = True) -> np.ndarray:
    return stops_from_pixel(pixel_stops(p, margin))


def dead_values(recs: np.ndarray) -> np.ndarray:
    """(n, 16) slot pages the device writes for pairs with all-zero sums: pack_v2's finish of a
    zero ACC (dmx = -((Af*0 + Af*0) + Bf*0), dmy alike with Cf; -0.5*0 and -0 give -0)."""
    fr = np.asarray(recs)[:, :3].view(f32)
    z = f32(0)
    with np.errstate(all="ignore"):
        t = fr[:, 0] * z
        dmx = -((t + t) + fr[:, 1] * z)
        t = fr[:, 2] * z
        dmy = -((t + t) + fr[:, 1] * z)
    out = np.zeros((len(fr), 16), f32)
    out[:, 0], out[:, 1] = dmx, dmy
    out[:, 2:5] = -z
    return out


def truncate(p: dict, stops: np.ndarray) -> dict:
    """Prep dict with each tile row cut to stops[row] pairs (rows with 0 dropped). recs keep
    their place (rec_start unchanged); dead_slots / dead_vals hold the dropped pairs' slots."""
    tiles = np.asarray(p["tiles"], np.int64)
    s = np.minimum(np.asarray(stops, np.int64), tiles[:, 2])
    keep = s > 0
    t = tiles.copy()
    t[:, 2], t[:, 3] = s, -(-s // BATCH)
    dead = np.concatenate([np.arange(a + c, a + n) for a, c, n in zip(tiles[:, 1], s, tiles[:, 2])]
                          + [np.zeros(0, np.int64)]).astype(np.int64)
    q = dict(p)
    q.update(tiles=t[keep].astype(np.uint32), planes=np.asarray(p["planes"])[keep],
             dead_slots=p["recs"][dead, 13].astype(np.int64),
             dead_vals=dead_values(p["recs"][dead]),
             stop_kept=int(s.sum()), stop_total=int(tiles[:, 2].sum()))
    return q


def fill_dead(slots: np.ndarray, q: dict) -> np.ndarray:
    """Slots as the device leaves them: the dead slots' pages written (first slots.shape[1] words)."""
    out = np.array(slots, f32, copy=True)
    if len(q.get("dead_slots", ())):
        out[q["dead_slots"]] = q["dead_vals"][:, :out.shape[1]]
    return out


def write_dead(q: dict, out: str) -> None:
    """dead.u32 (slot index per dropped pair) and dead.f32 (its 16-float slot page)."""
    q["dead_slots"].astype(np.uint32).tofile(os.path.join(out, "dead.u32"))
    np.asarray(q["dead_vals"], f32).tofile(os.path.join(out, "dead.f32"))


def read_dead(d: str) -> dict:
    if not os.path.exists(os.path.join(d, "dead.u32")):
        return {}
    s = np.fromfile(os.path.join(d, "dead.u32"), np.uint32).astype(np.int64)
    return {"dead_slots": s, "dead_vals": np.fromfile(os.path.join(d, "dead.f32"), f32).reshape(len(s), 16)}


def cfg_from_meta(meta: dict) -> BlendCfg:
    fb = lambda b: struct.unpack("<f", struct.pack("<I", int(b)))[0]  # noqa: E731
    return BlendCfg(alpha_max=fb(meta["alpha_max_bits"]), alpha_min=fb(meta["alpha_min_bits"]),
                    t_min=fb(meta["t_min_bits"]))


def main(argv=None) -> int:
    from gsplat import t1_balance_dev as tbd
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("prep_dir")
    ap.add_argument("out")
    ap.add_argument("--margin", type=int, default=1, help="0: exact emulate() rule (no device margin)")
    a = ap.parse_args(argv)
    p = tbd.read_prep(a.prep_dir)
    if "cores.u32" in os.listdir(a.prep_dir) or read_dead(a.prep_dir):
        print("t1_stop: truncate an unplanned, untruncated prep dir (then plan it)", file=sys.stderr)
        return 2
    nt = len(p["tiles"])
    p.update(cfg=cfg_from_meta(p["meta"]), planes=p["planes"].reshape(nt, N_PLANES, 1024),
             ramps=p["ramps"].reshape(2, 1024))
    q = truncate(p, tile_stops(p, bool(a.margin)))
    meta = tbd.write_prep(q, a.out)
    rep = {"n_tiles": meta["n_tiles"], "pairs": q["stop_total"], "kept": q["stop_kept"],
           "dead": int(len(q["dead_slots"])), "margin": bool(a.margin)}
    print(json.dumps(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
