"""#784 box-side check of the T2 heavy-row split.

  python3 docs/t784/check_g2d.py g2d <in_dir> <out_dir> [kcap hl ovh cores]
      out_dir from a T2_SPLIT=1 T2_G2D=1 run: pfwc streams 0..8 hold g2d. Checks g2d against
      t2_param_backward.segment_reduce(slots, P, counts) bit for bit on the active groups' rows,
      the host plan (split_heavy.u32, split_bounds.u32) against t2_split.device_plan, and the
      hsum records against the heavy rows' reference sums.
  python3 docs/t784/check_g2d.py same <in_dir> <out_a> <out_b>
      bit compare of two arms' outputs (params_out, m_out, v_out, pfwc, grads).
Prints one JSON object; exit 0 = pass."""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from gsplat import t2_param_backward as t2  # noqa: E402
from gsplat import t2_split as sp  # noqa: E402


def _u32(p):
    return np.fromfile(p, np.uint32)


def g2d(in_dir, out_dir, kcap=16, hl=16, ovh=2, cores=110):
    i2, o = pathlib.Path(in_dir) / "t2", pathlib.Path(out_dir)
    meta = _u32(i2 / "meta.u32").reshape(-1, 128)
    vis = _u32(i2 / "vis.u32")
    ng = meta.shape[0]
    slots = np.fromfile(o / "slots.f32", np.float32).reshape(-1, 16)[:, :9]
    P, cnt = meta[:, :64].ravel().astype(np.int64), meta[:, 64:].ravel().astype(np.int64)
    p = sp.device_plan(meta, vis, kcap=kcap, hl=hl, ovh=ovh, n_cores=cores)
    act = np.repeat(p["active"], 64)
    ref = t2.segment_reduce(slots, P, np.where(act, cnt, 0))
    pf = np.fromfile(o / "pfwc.f32", np.float32)[:9 * ng * 64].reshape(9, ng * 64).T
    rb, db = ref.view(np.uint32)[act], pf.view(np.uint32)[act]
    diff = rb != db
    zsign = int((diff & (ref[act] == 0) & (pf[act] == 0)).sum())
    heavy = p["heavy"]
    hs = np.fromfile(o / "hsum.f32", np.float32).reshape(-1, 16)[:heavy.size, :9]
    plan_ok = (np.array_equal(_u32(o / "split_heavy.u32"), heavy.astype(np.uint32))
               and np.array_equal(_u32(o / "split_bounds.u32")[:cores + 1], p["bounds"].astype(np.uint32)))
    rep = dict(rows=int(act.sum()), words=int(diff.size), bit_mismatch=int(diff.sum()), zero_sign_only=zsign,
               max_abs_diff=float(np.abs(ref[act] - pf[act]).max(initial=0.0)),
               heavy_rows=int(heavy.size), hsum_bit_equal=bool(np.array_equal(hs.view(np.uint32), ref[heavy].view(np.uint32))),
               plan_matches_mirror=bool(plan_ok), a_max_planes=p["a_max_planes"], b_max_weight=p["b_max_weight"],
               heavy_planes=p["heavy_planes"], light_planes=p["light_planes"])
    rep["pass"] = bool(rep["bit_mismatch"] == 0 and rep["hsum_bit_equal"] and plan_ok)
    return rep


def same(in_dir, a, b):
    """Bit compare on the active groups (inactive groups' pfwc/grads pages are never written)."""
    i2, a, b = pathlib.Path(in_dir) / "t2", pathlib.Path(a), pathlib.Path(b)
    meta = _u32(i2 / "meta.u32").reshape(-1, 128)
    ng = meta.shape[0]
    act = sp.device_plan(meta, _u32(i2 / "vis.u32"), n_cores=1)["active"]
    rep = {}
    for f in ("params_out.f32", "m_out.u16", "v_out.u16", "pfwc.f32", "grads.f32"):
        if not ((a / f).exists() and (b / f).exists()):
            continue
        x, y = (np.fromfile(q / f, np.uint8) for q in (a, b))
        if f == "pfwc.f32":  # word (s * ng + g) * 64 + lane
            x, y = (z.view(np.uint32).reshape(-1, ng, 64)[:, act] for z in (x, y))
        else:  # one page per group
            x, y = (z.reshape(ng, -1)[act] for z in (x, y))
        d = int((x != y).sum())
        rep[f] = "bit-equal" if d == 0 else f"DIFF ({d} elements)"
    rep["pass"] = bool(rep) and all(v == "bit-equal" for v in rep.values())
    return rep


if __name__ == "__main__":
    mode, args = sys.argv[1], sys.argv[2:]
    r = g2d(args[0], args[1], *map(int, args[2:])) if mode == "g2d" else same(*args)
    print(json.dumps(r))
    sys.exit(0 if r["pass"] else 1)
