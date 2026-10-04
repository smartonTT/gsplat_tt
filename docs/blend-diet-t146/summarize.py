#!/usr/bin/env python3
"""Summarize a t146 drive.sh log: per-arm mean ms/view and blend ms, md5 verdicts."""
import re, sys, collections
arms = collections.defaultdict(list); bad = []
tag = None
for line in open(sys.argv[1]):
    m = re.match(r"=== t146r(\d+)-(\S+)", line)
    if m: tag = m.group(2); continue
    m = re.match(r"STAGES n=30 .*blend=([\d.]+).*view_total=([\d.]+)", line)
    if m and tag: arms[tag].append((float(m.group(2)), float(m.group(1))))
    if "VIEWS DIFFER" in line: bad.append(tag)
for a, v in arms.items():
    print(f"{a:5s} n={len(v)} view={sum(x for x,_ in v)/len(v):.3f} blend={sum(y for _,y in v)/len(v):.3f}  "
          + " ".join(f"{x:.2f}/{y:.2f}" for x, y in v))
print("md5 differ:", bad or "none")
