#!/usr/bin/env python3
"""t684: b2b ms/view per arm (base 32768 vs grow 65472) from docs/bucket-grow-t684/out.

  python3 docs/bucket-grow-t684/analyze.py [out_dir]

Per arm: the 4 scored rounds' median ms/view, their median and min..max, md5s, and
whether the run log shows the intended bucket (pre-grown line / stride line)."""
import glob, re, statistics as st, sys

out = sys.argv[1] if len(sys.argv) > 1 else "docs/bucket-grow-t684/out"
res = {"base": [], "grow": []}
for f in sorted(glob.glob(f"{out}/run-r[1-4]-*.log")):
    r, arm = re.search(r"run-r(\d)-(\w+)\.log", f).groups()
    txt = open(f).read()
    pre = "pre-grown to 65472" in txt
    stride = re.findall(r"bucket tile stride (\d+) pages", txt)
    over = txt.count("tile overflow")
    for line in txt.splitlines():
        if line.startswith("B2B "):
            kv = dict(x.split("=", 1) for x in line.split()[1:] if "=" in x)
            res[arm].append((int(r), float(kv["ms_frame_median"]), float(kv["ms_frame"]),
                             kv["raw_md5"], kv.get("identical_across_passes"), pre, stride, over))
for arm, rows in res.items():
    meds = [x[1] for x in rows]
    print(f"{arm}: rounds={[(x[0], x[1]) for x in rows]}")
    if meds:
        print(f"  b2b ms/view median-of-medians={st.median(meds):.3f} min..max={min(meds):.3f}..{max(meds):.3f} "
              f"md5={sorted({x[3] for x in rows})} identical={sorted({x[4] for x in rows})} "
              f"pregrown={sorted({x[5] for x in rows})} stride_pages={sorted({tuple(x[6]) for x in rows})} "
              f"overflows={sum(x[7] for x in rows)}")
if res["base"] and res["grow"]:
    d = st.median([x[1] for x in res["grow"]]) - st.median([x[1] for x in res["base"]])
    pr = {x[0]: x[1] for x in res["base"]}
    per = [round(x[1] - pr[x[0]], 3) for x in res["grow"] if x[0] in pr]
    print(f"delta grow-base = {d:+.3f} ms/view; per-round deltas {per}")
