# t198: per-arm view_total and host stages from out/run-r<round>-<arm>.log, paired
# per-round deltas vs base (off).   python3 docs/two-cq-t198/summarize.py [out_dir]
import glob, os, re, statistics as st, sys
O = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "out")
ARMS = ("base", "early", "both")
KEYS = {"STAGES": ("view_total", "project", "sort", "blend", "d2h"),
        "SORT_STAGES": ("pread", "bin_layout", "bin_emit", "publish_host", "mat", "resid"),
        "PROJECT_STAGES": ("gather_wait", "gather_result")}
runs = {}
for f in sorted(glob.glob(os.path.join(O, "run-r[0-9]*-*.log"))):
    m = re.match(r"run-r(\d+)-(\w+)\.log", os.path.basename(f))
    if not m or m.group(2) not in ARMS: continue
    v = {}
    for line in open(f, errors="replace"):
        tag = line.split(" ", 1)[0]
        if tag in KEYS:
            for k in KEYS[tag]:
                mm = re.search(rf"\b{k}=([+-]?[0-9.]+)", line)
                if mm: v[f"{tag}.{k}"] = float(mm.group(1))
        mm = re.search(r"avg_frame_ms=([0-9.]+)", line)
        if mm and line.startswith("SUMMARY"): v["fps"] = 1000.0 / float(mm.group(1))
    if "STAGES.view_total" in v: runs[(int(m.group(1)), m.group(2))] = v
rounds = sorted({r for r, _ in runs})
cols = [f"{t}.{k}" for t in KEYS for k in KEYS[t]]
print("arm    n  " + "  ".join(c.split(".", 1)[1][:12].rjust(12) for c in cols))
for a in ARMS:
    rs = [runs[(r, a)] for r in rounds if (r, a) in runs]
    if not rs: continue
    print(f"{a:6s} {len(rs)}  " + "  ".join(f"{st.mean(x[c] for x in rs if c in x):12.3f}"
                                         if any(c in x for x in rs) else " " * 12 for c in cols))
for a in ("early", "both"):
    d = [runs[(r, a)]["STAGES.view_total"] - runs[(r, "base")]["STAGES.view_total"]
         for r in rounds if (r, a) in runs and (r, "base") in runs]
    if d:
        print(f"paired {a}-base view_total ms/view: " + " ".join(f"{x:+.3f}" for x in d) +
              f"  mean {st.mean(d):+.3f}" + (f" sd {st.stdev(d):.3f}" if len(d) > 1 else ""))
d = [runs[(r, "both")]["STAGES.view_total"] - runs[(r, "early")]["STAGES.view_total"]
     for r in rounds if (r, "both") in runs and (r, "early") in runs]
if d: print("paired both-early view_total ms/view: " + " ".join(f"{x:+.3f}" for x in d) +
            f"  mean {st.mean(d):+.3f}")
