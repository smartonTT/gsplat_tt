#!/usr/bin/env python3
"""Summarize t161 A/B logs (drive.sh r1..r3 + psnr): per-arm ms/view and host stages, paired deltas vs base, PSNR."""
import re, sys, statistics as st, collections
d = collections.defaultdict(dict); md5 = collections.defaultdict(set); ps = collections.defaultdict(list); mx = collections.defaultdict(int)
tag = None
for f in sys.argv[1:]:
    for line in open(f):
        if line.startswith("=== t161r"):  # t0 rounds (t161rt0-*) are not A/B rounds
            m = re.match(r"=== t161r(\d+)-(\S+)", line)
            tag = (int(m.group(1)), m.group(2)) if m and int(m.group(1)) > 0 else None; continue
        m = re.match(r"STAGES n=30 .*project=([\d.]+).*sort=([\d.]+).*blend=([\d.]+).*view_total=([\d.]+)", line)
        if m and tag: d[tag[1]][tag[0]] = tuple(map(float, m.groups()))
        if tag and ("VIEWS DIFFER" in line or "ALL_VIEWS_IDENTICAL" in line):
            md5[tag[1]].add(line.split(" (")[0].strip())
        m = re.match(r"psnr (\S+) r\d .*psnr=([\d.]+) .*max=(\d+)", line)
        if m: ps[m.group(1)].append(float(m.group(2))); mx[m.group(1)] = max(mx[m.group(1)], int(m.group(3)))
for a in ("off", "base", "pc"):
    v = list(d[a].values())
    print(f"{a:5s} rounds={len(v)} view={st.mean(x[3] for x in v):.3f}+-{st.stdev(x[3] for x in v):.3f} "
          f"({1000/st.mean(x[3] for x in v):.1f} FPS) project={st.mean(x[0] for x in v):.3f} sort={st.mean(x[1] for x in v):.3f} "
          f"blend={st.mean(x[2] for x in v):.3f} md5_vs_golden={sorted(md5[a])}")
for a in ("pc", "off"):
    dl = [d[a][r][3] - d["base"][r][3] for r in d["base"] if r in d[a]]
    print(f"paired {a}-base ms/view:", " ".join(f"{x:+.3f}" for x in dl), f"mean {st.mean(dl):+.3f}")
for a, p in ps.items():
    print(f"psnr {a} vs base: n={len(p)} min={min(p):.2f} max={max(p):.2f} max_abs_diff={mx[a]}")
