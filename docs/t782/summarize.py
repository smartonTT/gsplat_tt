#!/usr/bin/env python3
"""#787: summarize docs/t782/ab.sh outputs (docs/t782/out/remote): hero bench ms per arm and round,
MATH stats (ST_NV vectors, body cycles, per-core imbalance), chain timing and md5s, oracle checks."""
import glob
import json
import os
import statistics as st
import sys

import numpy as np

R = sys.argv[1] if len(sys.argv) > 1 else "docs/t782/out/remote"


def stats(p):
    d = json.load(open(p))
    c = np.array(d["cores"], dtype=np.int64)
    col = {k: c[:, i] for i, k in enumerate(d["cols"])}
    tot = col["tot"]
    return {
        "nv": int(col["nv"].sum()), "np": int(col["np"].sum()),
        "body_cyc_per_pair": float(col["body"].sum() / col["np"].sum()),
        "setup_cyc_per_pair": float(col["setup"].sum() / col["np"].sum()),
        "tot_max_ms": float(tot.max() / 1.35e6), "tot_mean_ms": float(tot.mean() / 1.35e6),
        "imbalance_max_over_mean": float(tot.max() / tot.mean()),
        "nv_max_over_mean": float(col["nv"].max() / col["nv"].mean()),
        "slowest_core": int(col["core"][tot.argmax()]),
        "slowest_body_cyc_per_pair": float(col["body"][tot.argmax()] / col["np"][tot.argmax()]),
        "slowest_nv": int(col["nv"][tot.argmax()]),
    }


out = {"hero": {}, "small": {}, "chain": {}, "checks": {}}
for arm in ("base", "cand"):
    meds = []
    for d in sorted(glob.glob(f"{R}/{arm}/hero-r*")):
        t = json.load(open(f"{d}/timing.json"))
        meds.append(t["device_ms_med"])
    s = [stats(p) for p in sorted(glob.glob(f"{R}/{arm}/hero-r*/stats.json"))]
    out["hero"][arm] = {"round_meds": meds, "median": st.median(meds) if meds else None,
                        "stats_r1": s[0] if s else None}
    for n in ("sh0", "sh3", "s1", "s2"):
        p = f"{R}/{arm}/{n}/timing.json"
        if os.path.exists(p):
            out["small"].setdefault(n, {})[arm] = json.load(open(p))["device_ms_med"]
    ch = []
    for d in sorted(glob.glob(f"{R}/chain-{arm}-r*")):
        p = f"{d}/run.log"
        step = [ln for ln in open(p) if ln.startswith("[STEP]")] if os.path.exists(p) else []
        md5 = open(f"{d}/md5.txt").read().split("\n") if os.path.exists(f"{d}/md5.txt") else []
        ch.append({"dir": os.path.basename(d), "step": json.loads(step[-1][7:]) if step else None,
                   "md5": [m.split()[0] if "slots9" not in m else m.split()[-1] for m in md5 if m.strip()]})
    out["chain"][arm] = ch
for p in sorted(glob.glob(f"{R}/check-*.json")):
    try:
        j = json.load(open(p))
        out["checks"][os.path.basename(p)] = {k: j.get(k) for k in ("rel_err", "pass", "n_skip", "rel_err_all")}
    except (ValueError, OSError) as e:
        out["checks"][os.path.basename(p)] = f"unreadable: {e}"
json.dump(out, sys.stdout, indent=1)
print()
