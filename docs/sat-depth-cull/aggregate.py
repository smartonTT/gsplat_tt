#!/usr/bin/env python3
"""Task #675: aggregate step0.py per-view JSON into the STEP0 tables.

Usage: aggregate.py OUTDIR [--per-pair-ms X]
Cost model (generous upper bound): a culled pair saves its whole per-pair share of
K2 emit + sort + mat + blend at best-iter-229 (stage split from training-view/ANALYSIS.md:
sort/K2 1.33 + mat 1.33 + blend 3.79 = 6.45 ms/view on the orbit) divided by the
orbit's mean pair count. A failed tile pays the pairs it blended before the cut again
(the rerun blends the full list, which the baseline pays anyway).
"""
import glob
import json
import sys
from collections import defaultdict

STAGE_MS = 1.33 + 1.33 + 3.79  # K2+sort, mat, blend at best-iter-229 (orbit 1024x1024)


def load(outdir, s):
    return [json.load(open(f)) for f in sorted(glob.glob(f"{outdir}/{s}_view*.json"))]


def main():
    outdir = sys.argv[1]
    orbit, train = load(outdir, "orbit"), load(outdir, "train")
    ref = orbit if orbit else train
    mean_pairs_orbit = sum(r["pairs"] for r in ref) / len(ref)
    c = STAGE_MS / mean_pairs_orbit  # ms per pair (upper bound)
    print(f"cost per pair (upper bound): {STAGE_MS} ms / {mean_pairs_orbit:,.0f} pairs = {c * 1e6:.3f} ns/pair")
    for name, rs in (("orbit", orbit), ("train", train)):
        if not rs:
            continue
        n = len(rs)
        P = sum(r["pairs"] for r in rs)
        beh = sum(r["pairs_behind"] for r in rs)
        lag = sum(r["pairs_behind_in_period_lag"] for r in rs)
        ts = sum(r["tiles_sat"] for r in rs)
        tn = sum(r["tiles_nonempty"] for r in rs)
        pu = sum(r["pix_unsat"] for r in rs) / sum(r["W"] * r["H"] for r in rs)
        steps = [r["step_deg"] for r in rs if "step_deg" in r]
        print(f"\n## {name}: {n} views, mean pairs {P / n:,.0f}, "
              f"median step {sorted(steps)[len(steps) // 2] if steps else 0:.2f} deg")
        print(f"tiles fully saturated: {ts / tn:.3%}; pixels never below eps: {pu:.3%}")
        print(f"oracle: pairs behind tile saturation {beh / P:.3%} = {beh / n * c:.4f} ms/view "
              f"(of which inside the 512-period live-mask lag {lag / P:.3%})")
        agg = defaultdict(lambda: defaultdict(int))
        k = 0
        for r in rs:
            if "pred" not in r:
                continue
            k += 1
            for key, v in r["pred"].items():
                for f, x in v.items():
                    agg[key][f] += x
        print("| margin, slack | tiles cut/view | fallback rate | captured pairs | wasted pairs | net ms/view |")
        print("|---|---|---|---|---|---|")
        Pk = sum(r["pairs"] for r in rs if "pred" in r)
        for key in sorted(agg):
            a = agg[key]
            fr = a["fail_tiles"] / a["cut_tiles"] if a["cut_tiles"] else 0
            net = (a["saved_pairs"] - a["wasted_pairs"]) / k * c
            first = rs[0]["pairs_behind"] * 0  # first view renders in full: no saving, no cost
            print(f"| {key} | {a['cut_tiles'] / k:.1f} | {fr:.1%} | {a['saved_pairs'] / Pk:.3%} "
                  f"(of behind {a['saved_pairs'] / max(1, sum(r['pairs_behind'] for r in rs if 'pred' in r)):.1%}) "
                  f"| {a['wasted_pairs'] / Pk:.3%} | {net + first:.4f} |")


if __name__ == "__main__":
    main()
