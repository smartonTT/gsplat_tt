#!/usr/bin/env python3
"""t630 (U9): list backward-branch loops in the matblend TRISC1 disassembly with their SFPU/TT
instruction counts. Usage: loops630.py out/matblend_trisc1.dis.gz"""
import gzip, re, sys
from collections import Counter
ins = []  # (addr, mnemonic, operands)
for ln in gzip.open(sys.argv[1], "rt"):
    m = re.match(r"\s+([0-9a-f]+):\s+[0-9a-f]+\s+(\S+)\s*(.*)", ln)
    if m:
        ins.append((int(m.group(1), 16), m.group(2), m.group(3)))
idx = {a: i for i, (a, _, _) in enumerate(ins)}
loops = []
for i, (a, mn, op) in enumerate(ins):
    if mn.startswith("b") or mn in ("j",):
        t = re.search(r"\b([0-9a-f]{4,})\s*<", op)
        if t and int(t.group(1), 16) < a and int(t.group(1), 16) in idx:
            j = idx[int(t.group(1), 16)]
            body = ins[j:i + 1]
            c = Counter(x[1] for x in body)
            sf = sum(v for k, v in c.items() if k.startswith("sfp"))
            tt = sum(v for k, v in c.items() if k.startswith("tt"))
            loops.append((sf, len(body), tt, hex(ins[j][0]), hex(a), c))
loops.sort(key=lambda x: (-x[0], x[1]))
for sf, n, tt, s, e, c in loops[:15]:
    top = ", ".join(f"{k}:{v}" for k, v in c.most_common() if k.startswith(("sfp", "tt")))
    print(f"loop {s}-{e} insns {n} sfpu {sf} tt {tt} | {top}")
