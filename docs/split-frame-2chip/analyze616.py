"""Task #616: model single-frame latency and b2b on 2 chips for a screen split.

Reads tiles616.npz/.json (exact per-tile pairs, per-chip M for static partitions) and
bbox616.npz (row/column bbox histograms, bbox pairs per tile). Stage costs are the t595 fits
on best-iter-227 (docs/profile-t227/PROFILE.md on ttp/t595-profile-227):
  pfwc 1.063 + 0.274 M,  K2 (folded, net) 0.27/2.614 P,  sort 0.329 + 0.424 P,
  mat+blend 1.421 + 1.443 P,  gaps 0.03  (M, P in millions; P = bbox pairs K2 emits)
mat+blend on a chip = the 1-chip fit x (LPT makespan of that chip's tiles on 110 cores /
LPT makespan of all 1024 tiles), tile cost = a + pairs (a = fixed share per tile).
Prints per-partition means and writes model616.txt.
"""
import json, heapq, sys
from pathlib import Path
import numpy as np

D = Path(__file__).resolve().parent
CORES = 110
PF0, PF1, K2S, SO0, SO1, MB0, MB1, GAPS = 1.063, 0.274, 0.27 / 2.614, 0.329, 0.424, 1.421, 1.443, 0.03
MEAS_1CHIP = 8.451
A_TILE = MB0 / MB1 * 1e6 / 1024  # fixed share per tile in pair units (~962)


def lpt(costs, cores=CORES):
    c = np.sort(np.asarray(costs, float))[::-1]
    c = c[c > 0]
    h = [0.0] * cores
    for x in c:
        heapq.heapreplace(h, h[0] + x)
    return max(h), c.sum() / cores


def stage(M, P):
    return dict(pfwc=PF0 + PF1 * M, k2=K2S * P, sort=SO0 + SO1 * P, mb=MB0 + MB1 * P)


def main():
    rows = json.loads((D / "tiles616.json").read_text())
    T = np.load(D / "tiles616.npz")
    B = np.load(D / "bbox616.npz")
    ex, bb, hy, hx = T["hist"], B["bbox_tile"], B["hy"], B["hx"]
    nv = len(rows)
    ty, tx = np.divmod(np.arange(1024), 32)
    out = []
    pr = lambda s: (print(s), out.append(s))

    M = np.array([r["M_vis"] for r in rows]) / 1e6
    P = bb.sum(1) / 1e6
    # tile cost for mat+blend: mat (~29% of per-core time) sees bbox pairs, blend (~71%)
    # the pairs mat keeps (exact rectangle test), rescaled to the bbox total.
    w = 1.420 / (1.420 + 3.514)
    cost = w * bb + (1 - w) * ex * (bb.sum(1) / ex.sum(1))[:, None]

    one = [stage(M[v], P[v]) for v in range(nv)]
    tot1 = np.array([sum(s.values()) + GAPS for s in one])
    pr(f"views {nv}: M mean {M.mean():.3f} M, bbox P mean {P.mean():.3f} M, exact P mean "
       f"{ex.sum(1).mean() / 1e6:.3f} M (device t595: M 1.675, P 2.614; hero M 1.880 P 2.896)")
    pr(f"1-chip model mean {tot1.mean():.3f} ms (measured 8.451); scale {MEAS_1CHIP / tot1.mean():.3f}")
    scale = MEAS_1CHIP / tot1.mean()
    full = [lpt(cost[v] + A_TILE) for v in range(nv)]
    full0 = [lpt(cost[v]) for v in range(nv)]
    pr(f"LPT 1-chip mat+blend makespan/mean-core: {np.mean([a / b for a, b in full]):.3f}; "
       f"max tile / mean core: {np.mean([(cost[v] + A_TILE).max() / full[v][1] for v in range(nv)]):.3f}")

    def chip_times(v, mask, Mc, fixed_per_tile, pfwc_region):
        res = []
        for c in (0, 1):
            sel = mask == c
            Pc = bb[v][sel].sum() / 1e6
            st = stage(Mc[c] if pfwc_region else M[v], Pc)
            if fixed_per_tile is True:
                st["mb"] = (MB0 + MB1 * P[v]) * lpt(cost[v][sel] + A_TILE)[0] / full[v][0]
            elif fixed_per_tile:  # float: that many ms per view stay fixed, the rest per tile
                fv = float(fixed_per_tile)
                a = A_TILE * (MB0 - fv) / MB0
                st["mb"] = fv + (MB0 - fv + MB1 * P[v]) * lpt(cost[v][sel] + a)[0] / lpt(cost[v] + a)[0]
            else:
                st["mb"] = MB0 + MB1 * P[v] * lpt(cost[v][sel])[0] / full0[v][0]
            res.append(sum(st.values()) + GAPS)
        return res

    def split_row(v, r, axis):
        """mask and M per chip for a contiguous split at tile row/col r."""
        h = hy[v] if axis == 0 else hx[v]
        lo = h[:r, :].sum() / 1e6          # bbox starts before r -> touches side 0
        hi = h[:, r:].sum() / 1e6          # bbox ends at/after r -> touches side 1
        key = ty if axis == 0 else tx
        return (key >= r).astype(int), (lo, hi)

    def summarize(name, per_view, note=""):
        lat = np.array([max(t) for t in per_view]) * scale
        dec = max(sum(t[0] for t in per_view), sum(t[1] for t in per_view)) / nv * scale
        bal = np.mean([max(t) / (sum(t) / 2) for t in per_view])
        pr(f"{name:34s} latency {lat.mean():6.3f} (max {lat.max():6.3f})  b2b-decoupled {dec:6.3f}  "
           f"imbalance {bal:5.3f}  cut {100 * (1 - lat.mean() / MEAS_1CHIP):5.1f}% {note}".rstrip())
        return lat.mean()

    results = {}
    for fpt in (True, 0.3, False):
        lab = {True: "per tile (halves with the tiles)", False: "per view (does not halve)"}.get(fpt, f"{fpt} ms per view, rest per tile")
        pr(f"\n== mat+blend fixed part {lab}; "
           "device time only, no host join/sync")
        for axis, nm in ((0, "top/bottom"), (1, "left/right")):
            pv = []
            for v in range(nv):
                mask, Mc = split_row(v, 16, axis)
                pv.append(chip_times(v, mask, Mc, fpt, True))
            results[(fpt, nm, "static")] = summarize(f"{nm} static 16/16, pfwc culls", pv)
            best_r, pvo = [], []
            for v in range(nv):
                cand = []
                for r in range(4, 29):
                    mask, Mc = split_row(v, r, axis)
                    cand.append((max(chip_times(v, mask, Mc, fpt, True)), r))
                best_r.append(min(cand)[1])
                mask, Mc = split_row(v, best_r[-1], axis)
                pvo.append(chip_times(v, mask, Mc, fpt, True))
            results[(fpt, nm, "oracle")] = summarize(f"{nm} per-view best row (oracle)", pvo,
                                                     f"rows {min(best_r)}-{max(best_r)}")
            pvp = []
            for v in range(nv):
                r = best_r[v - 1] if v > 0 else 16  # previous view's best line (orbit order)
                mask, Mc = split_row(v, r, axis)
                pvp.append(chip_times(v, mask, Mc, fpt, True))
            summarize(f"{nm} previous view's best row", pvp)
        for k, fn in (("rows1", ty & 1), ("rows2", (ty >> 1) & 1), ("rows4", (ty >> 2) & 1),
                      ("cols1", tx & 1), ("checker", (tx + ty) & 1), ("checker2", ((tx >> 1) + (ty >> 1)) & 1)):
            pv = [chip_times(v, fn, [x / 1e6 for x in rows[v]["m_bbox"][k]], fpt, False) for v in range(nv)]
            summarize(f"{k} interleave, pfwc all M", pv)
            pv = [chip_times(v, fn, [x / 1e6 for x in rows[v]["m_bbox"][k]], fpt, True) for v in range(nv)]
            summarize(f"{k} interleave, pfwc culls to chip", pv)
    # per-stage breakdown, top/bottom static, per-tile fixed
    pr("\n== per-stage, chip max, left/right static 16/16 (fixed per tile), mean over views, x scale")
    acc = {k: 0.0 for k in ("pfwc", "k2", "sort", "mb")}
    for v in range(nv):
        mask, Mc = split_row(v, 16, 1)
        best = None
        for c in (0, 1):
            sel = mask == c
            Pc = bb[v][sel].sum() / 1e6
            st = stage(Mc[c], Pc)
            st["mb"] = (MB0 + MB1 * P[v]) * lpt(cost[v][sel] + A_TILE)[0] / full[v][0]
            if best is None or sum(st.values()) > sum(best.values()):
                best = st
        for k in acc:
            acc[k] += best[k] * scale / nv
    pr("  " + "  ".join(f"{k} {x:.3f}" for k, x in acc.items()) + f"  gaps {GAPS * scale:.3f}")
    s1 = {k: np.mean([o[k] for o in one]) * scale for k in acc}
    pr("  1-chip: " + "  ".join(f"{k} {x:.3f}" for k, x in s1.items()))
    (D / "model616.txt").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
