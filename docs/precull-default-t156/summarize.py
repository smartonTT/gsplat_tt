#!/usr/bin/env python3
"""Summarize a t156 drive.sh log: per-arm mean ms/view and stages, paired deltas, md5 verdicts, PSNR range."""
import re, sys, statistics as st, collections
d = collections.defaultdict(dict); md5 = collections.defaultdict(list); ps = []; tag = None
for line in open(sys.argv[1]):
    m = re.match(r"=== t156r(\d+)-(\S+)", line)
    if m: tag = (int(m.group(1)), m.group(2)); continue
    m = re.match(r"STAGES n=30 .*project=([\d.]+).*sort=([\d.]+).*blend=([\d.]+).*view_total=([\d.]+)", line)
    if m and tag: d[tag[1]][tag[0]] = tuple(map(float, m.groups()))
    if tag and ("VIEWS DIFFER" in line or "ALL_VIEWS_IDENTICAL" in line) and tag[0] > 0:
        md5[tag[1]].append(line.split(" (")[0].strip())
    m = re.match(r"psnr r\d .*psnr=([\d.]+)", line)
    if m: ps.append(float(m.group(1)))
for a, rs in d.items():
    v = list(rs.values())
    print(f"{a:5s} rounds={len(v)} view={st.mean(x[3] for x in v):.3f}+-{st.stdev(x[3] for x in v):.3f} "
          f"project={st.mean(x[0] for x in v):.3f} sort={st.mean(x[1] for x in v):.3f} blend={st.mean(x[2] for x in v):.3f} "
          f"md5_vs_old_golden={sorted(set(md5[a]))}")
dl = [d["base"][r][3] - d["off"][r][3] for r in d["off"] if r in d["base"]]
print("paired base-off ms/view:", " ".join(f"{x:+.3f}" for x in dl), f"mean {st.mean(dl):+.3f}")
print(f"psnr base vs off: n={len(ps)} min={min(ps):.2f} max={max(ps):.2f}")
