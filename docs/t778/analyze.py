"""t778: per-arm b2b ms/view (median of round medians, rounds 1-4) and host stage split from the
pulled run logs (docs/t778/out/run-r<round>-<arm>.log), plus the per-view stage table of pvnn.
Usage: python3 docs/t778/analyze.py [out_dir]"""
import glob, os, re, statistics as st, sys

O = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
ARMS = ["rnd", "unf", "s0", "tp0", "tr", "rndnn", "trnn"]
res = {}
for a in ARMS:
    meds, stages = [], {}
    for r in (1, 2, 3, 4):
        p = os.path.join(O, f"run-r{r}-{a}.log")
        if not os.path.exists(p):
            continue
        txt = open(p).read()
        m = re.search(r"^B2B .*ms_frame_median=([\d.]+)", txt, re.M)
        if m:
            meds.append(float(m.group(1)))
        for line in re.findall(r"^B2B_STAGES (.*)$", txt, re.M):
            for k, v in re.findall(r"(\w+)=([-\d.]+)", line):
                if k not in ("pass", "views"):
                    stages.setdefault(k, []).append(float(v))
    res[a] = (meds, {k: st.median(v) for k, v in stages.items()})
print("| arm | rounds (ms_frame_median) | median | delta vs previous rung |")
print("|---|---|---:|---:|")
prev = None
for a in ARMS:
    meds = res[a][0]
    med = st.median(meds) if meds else float("nan")
    d = "" if prev is None or a.endswith("nn") else f"{med - prev:+.3f}"
    print(f"| {a} | {' '.join(f'{x:.3f}' for x in meds)} | {med:.3f} | {d} |")
    if not a.endswith("nn"):
        prev = med
keys = sorted({k for a in ARMS for k in res[a][1]})
print("\nstage medians (ms/view, B2B_STAGES over all passes and rounds)")
print("| stage | " + " | ".join(ARMS) + " |")
print("|---|" + "---:|" * len(ARMS))
for k in keys:
    print(f"| {k} | " + " | ".join(f"{res[a][1].get(k, float('nan')):.3f}" for a in ARMS) + " |")
p = os.path.join(O, "run-r5-pvnn.log")
if os.path.exists(p):
    rows = re.findall(r"^VIEW_STAGES (.*)$", open(p).read(), re.M)
    print(f"\npvnn per-view (latency mode, train flag, {len(rows)} dataset views)")
    cols = {}
    for line in rows:
        for k, v in re.findall(r"(\w+)=([-\d.]+)", line):
            if k != "i":
                cols.setdefault(k, []).append(float(v))
    for k, v in cols.items():
        print(f"{k}: min {min(v):.2f} median {st.median(v):.2f} max {max(v):.2f} (n={len(v)})")
