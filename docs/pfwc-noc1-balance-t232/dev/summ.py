#!/usr/bin/env python3
"""t232: paired untraced rounds (view_total, project ms/view): base (default) vs fix (P2 + the
BRISC input reader config) and, when run, rdb (the mask alone) or alt (P2 + another config);
verify rounds v1 v2: base (new default) vs off (the old default).
   summ.py <out dir>"""
import re
import sys

O = sys.argv[1] if len(sys.argv) > 1 else "out"


def stages(path):
    try:
        for ln in open(path):
            if ln.startswith("STAGES "):
                return {k: float(v) for k, v in re.findall(r"(\w+)=([-+0-9.]+)", ln)}
    except OSError:
        pass
    return None


for arm, rounds in (("fix", "1234"), ("rdb", "1234"), ("alt", "1234"), ("off", ("v1", "v2"))):
    rows = []
    for r in rounds:
        b, c = stages(f"{O}/run-r{r}-base.log"), stages(f"{O}/run-r{r}-{arm}.log")
        if b and c:
            rows.append((r, b, c))
    if not rows:
        continue
    print(f"| round | base view_total | {arm} view_total | delta | base project | {arm} project | delta |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for r, b, c in rows:
        print(f"| {r} | {b['view_total']:.3f} | {c['view_total']:.3f} | {c['view_total'] - b['view_total']:+.3f} "
              f"| {b['project']:.3f} | {c['project']:.3f} | {c['project'] - b['project']:+.3f} |")
    n = len(rows)
    mb = sum(b["view_total"] for _, b, _ in rows) / n
    mc = sum(c["view_total"] for _, _, c in rows) / n
    pb = sum(b["project"] for _, b, _ in rows) / n
    pc = sum(c["project"] for _, _, c in rows) / n
    print(f"| mean | {mb:.3f} | {mc:.3f} | **{mc - mb:+.3f}** | {pb:.3f} | {pc:.3f} | **{pc - pb:+.3f}** |")
    print(f"FPS base {1000 / mb:.1f} {arm} {1000 / mc:.1f}\n")
