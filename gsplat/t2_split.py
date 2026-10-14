"""T2 heavy-row split: reference segment reduce + host-side core assignment (#774, L1 of
docs/training-plan/t773/PLAN.md, branch ttp/t773-plan-trainer-ms-view-breakdown).

The t697/t750 T2 gather (render/train/kernels/t2g_reader.cpp) runs, per 64-row group, one rank
plane per slot index up to the group's largest slot count kmax. On the hero view about 3,000 rows
with k > 16 (up to 1014) pad their groups and make most of the planes. This splits the work:

  light path  rows with counts <= kcap keep the rank planes of their own 64-row group (row
              layout and param pages unchanged); a heavy lane in the group reads as count 0, so a
              group runs max(light kmax, 1) <= kcap planes;
  heavy path  rows with counts > kcap, as a flat list sorted by count (descending, then row id),
              packed 64 per heavy group; lane i walks its own contiguous slot run P[r] .. P[r]+k-1,
              so a heavy group runs only its first lane's count in planes. Its sums replace the
              heavy lanes' (zero) light sums before the projection backward.

Sum order is unchanged: every row still sums its slots from +0.0, j ascending, one fp32 add per
slot (padding lanes add +0.0, which is exact: the accumulator is never -0.0). So the result is
bit-equal to t2_param_backward.segment_reduce. Nothing is kept between views: the plan is rebuilt
from each view's counts, so it is lossless by construction (msg #384).

Core assignment (plan()): heavy groups first, largest first onto the least-loaded core (LPT);
then light groups as one contiguous range per core (cores in order), with the smallest capacity
that fits them all (binary search), so a core's light work is (first group, last group) runtime
args and its DRAM reads stay contiguous. Cost unit: rank planes, plus an optional fixed cost per
active light group (projection backward + Adam; 0 = plane count only, the #773 model's unit).

  python3 -m gsplat.t2_split model <ref_in.npz> [--out plan.json]   hero T2 dump -> plane balance
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass

import numpy as np

f32 = np.float32
G = 64        # rows per group (t2_dev.G)
KCAP = 16     # light rows have counts <= KCAP
CORES = 110


@dataclass
class SplitPlan:
    n: int                     # rows
    kcap: int
    group: int                 # rows per group / lanes per heavy group
    n_cores: int
    heavy_rows: np.ndarray     # (nh,) row ids, counts descending then row ascending
    active: np.ndarray         # (ng,) bool: light group run on device (any visible or slotted row)
    group_planes: np.ndarray   # (ng,) planes per light group (0 when inactive)
    hgroup_planes: np.ndarray  # (nhg,) planes per heavy group (its first lane's count)
    core_bounds: np.ndarray    # (n_cores + 1,) light groups [b[c], b[c+1]) run on core c
    core_hgroups: list         # per core: heavy group ids (run before its light groups)
    core_planes: np.ndarray    # (n_cores,) modelled planes (+ group overhead) per core
    group_overhead: float = 0.0

    def summary(self) -> dict:
        cp = self.core_planes
        lp, hp = int(self.group_planes.sum()), int(self.hgroup_planes.sum())
        return {"n_rows": self.n, "kcap": self.kcap, "n_cores": self.n_cores,
                "n_groups": int(self.group_planes.size), "n_active_groups": int(self.active.sum()),
                "n_heavy_rows": int(self.heavy_rows.size),
                "n_heavy_groups": int(self.hgroup_planes.size),
                "light_planes": lp, "heavy_planes": hp, "total_planes": lp + hp,
                "max_heavy_group_planes": int(self.hgroup_planes.max(initial=0)),
                "group_overhead": self.group_overhead,
                "core_max": float(cp.max()), "core_mean": round(float(cp.mean()), 1),
                "core_min": float(cp.min()),
                "mean_over_max": round(float(cp.mean() / cp.max()), 4) if cp.max() else 1.0}


def heavy_order(counts: np.ndarray, kcap: int = KCAP) -> np.ndarray:
    """Row ids with counts > kcap, counts descending then row id ascending (deterministic)."""
    c = np.asarray(counts, np.int64)
    h = np.flatnonzero(c > kcap)
    return h[np.lexsort((h, -c[h]))]


def _contig_fill(w: np.ndarray, cap: np.ndarray) -> np.ndarray | None:
    """Greedy contiguous split of item weights w over cores with capacities cap (in order).
    Returns bounds (len(cap)+1,) or None when the items do not fit."""
    cs = np.concatenate([[0.0], np.cumsum(w, dtype=np.float64)])
    b = np.zeros(cap.size + 1, np.int64)
    for c in range(cap.size):
        # last end index e with cs[e] - cs[b[c]] <= cap[c]
        b[c + 1] = np.searchsorted(cs, cs[b[c]] + cap[c], side="right") - 1
        b[c + 1] = max(b[c + 1], b[c])
    return b if b[-1] == w.size else None


def plan(counts: np.ndarray, visible: np.ndarray | None = None, kcap: int = KCAP,
         n_cores: int = CORES, group: int = G, group_overhead: float = 0.0) -> SplitPlan:
    """Split rows into light groups and heavy groups and assign both to n_cores cores."""
    c = np.asarray(counts, np.int64)
    n = c.size
    ng = -(-n // group)
    heavy = heavy_order(c, kcap)
    cl = c.copy()
    cl[heavy] = 0
    pad = lambda a, v=0: np.concatenate([a, np.full(ng * group - n, v, a.dtype)])  # noqa: E731
    kmax = pad(cl).reshape(ng, group).max(1)
    vis = np.ones(n, bool) if visible is None else np.asarray(visible, bool)
    active = (pad(vis, False) | pad(c > 0, False)).reshape(ng, group).any(1)
    gp = np.where(active, np.maximum(kmax, 1), 0)
    hp = c[heavy[::group]] if heavy.size else np.zeros(0, np.int64)
    # Heavy groups: LPT (largest first, ties to the lower core id).
    load = np.zeros(n_cores, np.float64)
    core_h = [[] for _ in range(n_cores)]
    for j in np.argsort(-hp, kind="stable"):
        k = int(np.argmin(load))
        core_h[k].append(int(j))
        load[k] += hp[j]
    # Light groups: one contiguous range per core, minimal capacity by binary search.
    w = gp + group_overhead * active
    lo, hi = max(float(load.max()), float(w.max(initial=0))), float(load.max() + w.sum())
    best = _contig_fill(w, np.maximum(hi - load, 0))
    for _ in range(60):
        if hi - lo <= 0.5:
            break
        mid = (lo + hi) / 2
        b = _contig_fill(w, np.maximum(mid - load, 0))
        if b is None:
            lo = mid
        else:
            hi, best = mid, b
    cs = np.concatenate([[0.0], np.cumsum(w, dtype=np.float64)])
    cp = load + (cs[best[1:]] - cs[best[:-1]])
    return SplitPlan(n, kcap, group, n_cores, heavy, active, gp, hp, best,
                     [np.asarray(h, np.int64) for h in core_h], cp, group_overhead)


def check_plan(p: SplitPlan) -> None:
    """Every active light group and every heavy group runs exactly once."""
    b = p.core_bounds
    assert b[0] == 0 and b[-1] == p.group_planes.size and (np.diff(b) >= 0).all()
    hg = np.sort(np.concatenate(p.core_hgroups)) if p.core_hgroups else np.zeros(0, np.int64)
    assert np.array_equal(hg, np.arange(p.hgroup_planes.size))
    assert (p.group_planes <= max(p.kcap, 1)).all()  # an active group runs >= 1 plane


def _rank_planes(acc: np.ndarray, rows: np.ndarray, nplanes: np.ndarray, cnt: np.ndarray,
                 P: np.ndarray, slots: np.ndarray) -> None:
    """acc[rows] += plane k for k < nplanes (per row: its group's plane count); lanes with
    k >= cnt add +0.0, as the device's zero-filled planes do."""
    zero = np.zeros(slots.shape[1], f32)
    for k in range(int(nplanes.max(initial=0))):
        r = nplanes > k
        rr, cc, pp = rows[r], cnt[r], P[r]
        live = cc > k
        plane = np.where(live[:, None], slots[np.where(live, pp + k, 0)], zero)
        acc[rr] += plane


def segment_reduce_split(slots: np.ndarray, P: np.ndarray, counts: np.ndarray,
                         p: SplitPlan | None = None, kcap: int = KCAP) -> np.ndarray:
    """(N, 9) fp32 per-row slot sums, bit-equal to t2_param_backward.segment_reduce, computed
    core by core as the plan assigns: each core's heavy groups, then its light group range."""
    c = np.asarray(counts, np.int64)
    Pi = np.asarray(P, np.int64)
    slots = np.asarray(slots, f32)
    p = plan(c, kcap=kcap) if p is None else p
    check_plan(p)
    n, g = p.n, p.group
    acc = np.zeros((n, slots.shape[1]), f32)
    cl = c.copy()
    cl[p.heavy_rows] = 0
    hacc = np.zeros((p.heavy_rows.size, slots.shape[1]), f32)
    for core in range(p.n_cores):
        for j in p.core_hgroups[core]:
            lanes = np.arange(j * g, min((j + 1) * g, p.heavy_rows.size))
            rows = p.heavy_rows[lanes]
            sub = np.zeros((lanes.size, slots.shape[1]), f32)
            _rank_planes(sub, np.arange(lanes.size), np.full(lanes.size, p.hgroup_planes[j]),
                         c[rows], Pi[rows], slots)
            hacc[lanes] = sub
        g0, g1 = p.core_bounds[core], p.core_bounds[core + 1]
        rows = np.arange(g0 * g, min(g1 * g, n))
        if rows.size:
            _rank_planes(acc, rows, p.group_planes[rows // g], cl[rows], Pi[rows], slots)
    # Merge: a heavy lane's light sum is +0.0 + ... + 0.0 = +0.0, so adding is a plain write.
    acc[p.heavy_rows] += hacc
    return acc


HFLAG = 0x80000000


def device_plan(meta: np.ndarray, vis: np.ndarray, kcap: int = KCAP, hl: int = 16, ovh: int = 2,
                n_cores: int = CORES) -> dict:
    """#784: the device split plan exactly as render/train/train_step.cpp builds it (T2_SPLIT=1)
    from the T2 prep's meta.u32 (per group: P[64], counts[64]) and vis.u32 (16 groups x 2 words per
    128 B page).

    Phase A (program 1): heavy rows (active group, count > kcap), sorted by count descending then
    row ascending, packed hl per heavy group; htab holds one meta-layout page per heavy group; the
    groups go largest first onto the least-loaded core (ties: lowest core). Phase B (program 2):
    metab = meta with a heavy lane's count word set to HFLAG | h (its hsum record), and one
    contiguous group range per core with the smallest capacity that fits, weight = planes + ovh
    per active group (planes = max(light counts, 1))."""
    meta = np.asarray(meta, np.uint32).reshape(-1, 128)
    ng = meta.shape[0]
    vis = np.asarray(vis, np.uint32).reshape(-1, 32)
    g = np.arange(ng)
    active = (vis[g // 16, 2 * (g % 16)] | vis[g // 16, 2 * (g % 16) + 1]) != 0
    cnt = meta[:, 64:].astype(np.int64)
    cand = np.flatnonzero((active[:, None] & (cnt > kcap)).ravel())
    heavy = cand[np.argsort(-cnt.ravel()[cand], kind="stable")]
    nhg = -(-heavy.size // hl)
    htab = np.zeros((max(nhg, 1), 128), np.uint32)
    metab = meta.copy()
    h = np.arange(heavy.size)
    htab[h // hl, h % hl] = meta[heavy // 64, heavy % 64]
    htab[h // hl, 64 + h % hl] = cnt.ravel()[heavy]
    metab[heavy // 64, 64 + heavy % 64] = HFLAG | h.astype(np.uint32)
    core_h = [[] for _ in range(n_cores)]
    load = np.zeros(n_cores, np.int64)
    for j in range(nhg):
        c = int(np.argmin(load))
        core_h[c].append(j)
        load[c] += int(htab[j, 64])
    lc = np.where(metab[:, 64:] & HFLAG, 0, metab[:, 64:]).astype(np.int64)
    planes = np.maximum(lc.max(1), 1)
    w = np.where(active, planes + ovh, 0)

    def fill(cap):
        b, c, acc = [0], 0, 0
        for q in range(ng):
            if acc + w[q] > cap:
                c += 1
                if c >= n_cores:
                    return None
                b.append(q)
                acc = 0
            acc += int(w[q])
        return b + [ng] * (n_cores + 1 - len(b))

    lo = max(int(w.max(initial=0)), -(-int(w.sum()) // n_cores))
    hi = max(int(w.sum()), lo)
    while lo < hi:
        mid = (lo + hi) // 2
        if fill(mid) is None:
            lo = mid + 1
        else:
            hi = mid
    bounds = np.array(fill(lo), np.int64)
    return dict(heavy=heavy, htab=htab, metab=metab, core_h=core_h, bounds=bounds, active=active,
                a_max_planes=int(load.max(initial=0)), heavy_planes=int(htab[:nhg, 64].sum()),
                light_planes=int(planes[active].sum()), hl=hl,
                b_max_weight=int(max(w[bounds[i]:bounds[i + 1]].sum() for i in range(n_cores))))


def _lanes_sum(slots: np.ndarray, P: np.ndarray, cnt: np.ndarray, nk: int,
               first: np.ndarray | None = None) -> np.ndarray:
    """Device lane sums: plane 0 is copied (or `first`, a heavy lane's hsum record), planes
    1..nk-1 added in order; a lane past its count adds +0.0."""
    zero = np.zeros(slots.shape[1], f32)
    acc = np.where((cnt > 0)[:, None], slots[np.where(cnt > 0, P, 0)], zero).astype(f32)
    if first is not None:
        acc = np.where(first[0][:, None], first[1], acc).astype(f32)
    for k in range(1, nk):
        live = cnt > k
        acc += np.where(live[:, None], slots[np.where(live, P + k, 0)], zero)
    return acc


def device_split_sums(slots: np.ndarray, meta: np.ndarray, vis: np.ndarray, **kw) -> np.ndarray:
    """(ng*64, 9) fp32 g2d as the T2_SPLIT=1 device programs compute it (phase A into hsum, then
    phase B per active group); rows of inactive groups stay 0. Used by the unit tests and the
    box-side check of the T2_G2D=1 dump."""
    p = device_plan(meta, vis, **kw)
    slots = np.asarray(slots, f32)
    meta = np.asarray(meta, np.uint32).reshape(-1, 128)
    hl = p["hl"]
    hs = np.zeros((p["htab"].shape[0] * hl, slots.shape[1]), f32)
    for j in range(p["heavy"].size and p["htab"].shape[0]):
        t = p["htab"][j].astype(np.int64)
        hs[j * hl:(j + 1) * hl] = _lanes_sum(slots, t[:hl], t[64:64 + hl], int(t[64]))
    out = np.zeros((meta.shape[0] * 64, slots.shape[1]), f32)
    for g in np.flatnonzero(p["active"]):
        m = p["metab"][g].astype(np.int64)
        hv = (m[64:] & HFLAG) != 0
        lc = np.where(hv, 0, m[64:])
        nk = max(int(lc.max()), 1)
        first = (hv, hs[np.where(hv, m[64:] & ~HFLAG, 0)])
        out[g * 64:(g + 1) * 64] = _lanes_sum(slots, m[:64], np.where(hv, 1, lc), nk, first)
    return out


def contiguous_baseline(counts: np.ndarray, visible: np.ndarray | None = None,
                        n_cores: int = CORES, group: int = G) -> dict:
    """Today's T2 (no split): planes = max(kmax, 1) per active group, blocks of 16 groups
    round-robin over the cores (t2g_reader.cpp b0 / bstride)."""
    c = np.asarray(counts, np.int64)
    n = c.size
    ng = -(-n // group)
    pad = lambda a, v=0: np.concatenate([a, np.full(ng * group - n, v, a.dtype)])  # noqa: E731
    kmax = pad(c).reshape(ng, group).max(1)
    vis = np.ones(n, bool) if visible is None else np.asarray(visible, bool)
    active = pad(vis, False).reshape(ng, group).any(1)  # device skips groups by PFWC_VIS only
    gp = np.where(active, np.maximum(kmax, 1), 0)
    blk = np.concatenate([gp, np.zeros(-ng % 16, np.int64)]).reshape(-1, 16).sum(1)
    core = np.zeros(n_cores, np.int64)
    np.add.at(core, np.arange(blk.size) % n_cores, blk)
    return {"total_planes": int(gp.sum()), "core_max": int(core.max()),
            "core_mean": round(float(core.mean()), 1)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("model", help="plane balance on a T2 ref_in.npz (t2_dev prep-dump)")
    m.add_argument("ref_in")
    m.add_argument("--kcap", type=int, default=KCAP)
    m.add_argument("--cores", type=int, default=CORES)
    m.add_argument("--verify", action="store_true", help="also check bit-equal sums")
    m.add_argument("--out")
    a = ap.parse_args(argv)
    r = np.load(a.ref_in)
    counts, P = r["counts"], r["P"]
    vis = None
    if "p_means" in r.files:  # the device skips groups by PFWC_VIS (t2_dev.prep_dump)
        from gsplat import t2_param_backward as t2
        prm = {k: r[f"p_{k}"] for k in t2.PARAM_NAMES}
        vis = t2.forward_lanes(prm, r["w2c"], r["K"], int(r["H"]), int(r["W"]))["visible"]
    p = plan(counts, vis, a.kcap, a.cores)
    check_plan(p)
    out = {"source": a.ref_in, "split": p.summary(),
           "baseline_round_robin": contiguous_baseline(counts, vis, a.cores),
           "core_planes": [int(x) for x in p.core_planes]}
    if a.verify:
        from gsplat import t2_param_backward as t2
        ref = t2.segment_reduce(r["slots"], P, counts)
        got = segment_reduce_split(r["slots"], P, counts, p)
        out["bit_equal"] = bool(np.array_equal(ref.view(np.uint32), got.view(np.uint32)))
    print(json.dumps({k: v for k, v in out.items() if k != "core_planes"}, indent=1))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)
    return 0 if out.get("bit_equal", True) else 1


if __name__ == "__main__":
    sys.exit(main())
