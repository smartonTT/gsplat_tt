#!/usr/bin/env python3
"""t228: paired untraced rounds (view_total, project ms/view) and the TRISC step split.
   summ.py <out dir>"""
import os
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


rows = []
for r in ("1", "2", "3", "4"):
    b, c = stages(f"{O}/run-r{r}-base.log"), stages(f"{O}/run-r{r}-cov2d.log")
    if b and c:
        rows.append((r, b, c))
if rows:
    print("| round | base view_total | cov2d view_total | delta | base project | cov2d project | delta |")
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
    print(f"FPS base {1000 / mb:.1f} cov2d {1000 / mc:.1f}")

# TRISC step split: steps 6-11 (a, b, c, conic, radx, rady) in ms/view and cycles/chunk.
STEPS = ["a", "b", "c", "conic", "radx", "rady"]
for n in ("k0", "k1"):
    p = f"{O}/prof-{n}-pc_split.txt"
    if not os.path.exists(p):
        continue
    txt = open(p).read()
    for blk in re.findall(r"(TRISC_\d) pfwc_pc:.*?mean ms : (.*?)\n.*?chunks/launch mean ([0-9.]+)", txt, re.S):
        risc, mean, cpl = blk
        vals = dict(re.findall(r"([\w+]+) ([0-9.]+)", mean))
        s = sum(float(vals.get(k, 0)) for k in STEPS)
        cyc = s * 1.35e6 / float(cpl)
        print(f"{n} {risc}: wall {vals.get('wall')} steps6-11 {s:.3f} ms/view = {cyc / 1000:.1f}k cycles/chunk "
              f"({' '.join(k + ' ' + vals.get(k, '0') for k in STEPS)}) vis+pop {vals.get('vis+pop')}")
    for blk in re.findall(r"((?:BRISC|NCRISC) pfwc_ws:.*?mean ms : wall ([0-9.]+), wait ([0-9.]+))", txt, re.S):
        print(f"{n} {blk[0].split()[0]} writer: wall {blk[1]} wait {blk[2]}")
