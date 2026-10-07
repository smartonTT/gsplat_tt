#!/usr/bin/env python3
"""t365: per-row / per-column mean bytes per RISC tick from the [MBMAP] lines of mb.txt."""
import re, sys, statistics as st
def load(path):
    m = {}
    for l in open(path):
        if not l.startswith('[MBMAP]'): continue
        h = dict(re.findall(r'(\w+)=([\w+.]+)', l))
        cells = {int(x): float(v) for x, v in re.findall(r' (\d+):([\d.]+)', l)}
        m.setdefault((h['probe'], h['set'], h['risc']), {})[int(h['y'])] = cells
    return m
for path in sys.argv[1:]:
    m = load(path)
    print(f'### {path}')
    for (probe, s, risc), rows in sorted(m.items()):
        if s != risc: continue  # single-RISC sets: NoC0 = BRISC (b), NoC1 = NCRISC (n)
        ys = sorted(rows); xs = sorted({x for r in rows.values() for x in r})
        allv = [v for r in rows.values() for v in r.values()]
        rm = ' '.join(f'{st.mean(rows[y].values()):.2f}' for y in ys)
        cm = ' '.join(f'{st.mean(rows[y][x] for y in ys if x in rows[y]):.2f}' for x in xs)
        print(f'| {probe} | {"NoC0" if risc == "b" else "NoC1"} | {st.mean(allv):.2f} | {min(allv):.2f}-{max(allv):.2f} | y{ys[0]}..{ys[-1]}: {rm} | x{xs}: {cm} |'.replace(', ', ','))
