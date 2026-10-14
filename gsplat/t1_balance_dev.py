"""T1 load-balanced prep dirs for the device bench (#785, plan from gsplat/t1_balance.py, #775).

  python3 -m gsplat.t1_balance_dev plan <prep_dir> <out_dir> [--cores 110] [--split --kmax 2]
  python3 -m gsplat.t1_balance_dev merge <plan_dir> <slots.f32> <out.f32>

plan: reads a `blend_backward_dev prep` dir and writes the planned prep dir (rows in core order,
recs regrouped per row, cores.u32 with the per-core row boundaries, merge.i64). The bench
(render/train/blend_bwd_bench.cpp) takes cores.u32 in place of its contiguous split when present
(BWD_BALANCE=0 ignores it). Without --split the outputs are bit-identical to the unplanned run.
merge: T2 merge of a split plan's partial slots (merge_partials), giving the base-sized slots.f32
for `blend_backward_dev check`; a plan with no partial slots is copied unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from gsplat import t1_balance as bal
from gsplat.blend_backward_dev import N_PLANES, REC_WORDS
from gsplat.t1_stop import read_dead, write_dead


def read_prep(d: str) -> dict:
    """A prep dir as the dict apply_plan takes (tiles: first 4 words of each 16-word row)."""
    with open(os.path.join(d, "meta.json")) as f:
        meta = json.load(f)
    nt, nr = meta["n_tiles"], meta["n_recs"]
    return {"meta": meta, "n_slots": meta["n_slots"],
            "tiles": np.fromfile(os.path.join(d, "tiles.u32"), np.uint32).reshape(nt, 16)[:, :4],
            "recs": np.fromfile(os.path.join(d, "recs.u32"), np.uint32).reshape(nr, REC_WORDS),
            "planes": np.fromfile(os.path.join(d, "planes.f32"), np.float32).reshape(nt, N_PLANES * 1024),
            "ramps": np.fromfile(os.path.join(d, "ramps.f32"), np.float32), **read_dead(d)}


def write_prep(q: dict, out: str) -> dict:
    """blend_backward_dev.write() layout from the source meta, plus cores.u32 and merge.i64 (a
    plan) and dead.u32 / dead.f32 (a truncated prep, gsplat.t1_stop)."""
    os.makedirs(out, exist_ok=True)
    t = np.zeros((len(q["tiles"]), 16), np.uint32)
    t[:, :4] = q["tiles"]
    t.tofile(os.path.join(out, "tiles.u32"))
    q["recs"].astype(np.uint32).tofile(os.path.join(out, "recs.u32"))
    q["planes"].astype(np.float32).tofile(os.path.join(out, "planes.f32"))
    q["ramps"].astype(np.float32).tofile(os.path.join(out, "ramps.f32"))
    if "core_beg" in q:
        q["core_beg"].astype(np.uint32).tofile(os.path.join(out, "cores.u32"))
        np.stack([q["merge_dst"], q["merge_rank"]], 1).astype(np.int64).tofile(os.path.join(out, "merge.i64"))
    if "dead_slots" in q:   # #802 truncated prep: slots of the pairs past each tile's stop
        write_dead(q, out)
    meta = dict(q["meta"], n_tiles=len(q["tiles"]), n_recs=len(q["recs"]), n_slots=int(q["n_slots"]),
                n_slots_base=int(q.get("n_slots_base", q["n_slots"])),
                n_vectors=int(bal.popcount32(q["recs"][:, 12]).sum()))
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    return meta


def merge_file(plan_dir: str, slots: str, out: str) -> int:
    with open(os.path.join(plan_dir, "meta.json")) as f:
        meta = json.load(f)
    s = np.fromfile(slots, np.float32).reshape(-1, 16)
    m = np.fromfile(os.path.join(plan_dir, "merge.i64"), np.int64).reshape(-1, 2)
    n0 = meta.get("n_slots_base", meta["n_slots"])
    bal.merge_partials(s, n0, m[:, 0], m[:, 1]).tofile(out)
    return len(m)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("plan")
    a.add_argument("prep_dir")
    a.add_argument("out")
    a.add_argument("--cores", type=int, default=110)
    a.add_argument("--split", action="store_true")
    a.add_argument("--kmax", type=int, default=2)
    m = sub.add_parser("merge")
    m.add_argument("plan_dir")
    m.add_argument("slots")
    m.add_argument("out")
    args = ap.parse_args(argv)
    if args.cmd == "merge":
        print(json.dumps({"partials_merged": merge_file(args.plan_dir, args.slots, args.out)}))
        return 0
    p = read_prep(args.prep_dir)
    pl = bal.plan(p["tiles"], p["recs"][:, 12], args.cores, args.split, args.kmax)
    meta = write_prep(bal.apply_plan(p, pl), args.out)
    print(json.dumps({"n_tiles": meta["n_tiles"], "n_recs": meta["n_recs"], "n_slots": meta["n_slots"],
                      "split_tiles": int((pl["ks"] > 1).sum()), "partial_slots": pl["n_extra"],
                      **bal.summary(pl)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
