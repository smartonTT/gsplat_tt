#!/usr/bin/env python3
"""t356: re-derive the sort mover speed table for this board from an emit_cores.py run
(the record counts are 0 in iter-207 EMIT captures, so --weights cannot run). Pages are
split today in proportion to kMoverSpeedP150 (s_c), so a mover's measured emit time T_c
gives its true speed v_c ~ s_c / T_c. Prints the predicted emit time with pages split by v_c
(sum s / sum (s_c / T_c)) and the new table (mean = 1000).

Usage: reweight.py <emit_cores output> <sort_mover_speed.h>
"""
import re, sys

emit = {}
for ln in open(sys.argv[1]):
    m = re.match(r"\s*(\d+)\s+(\d+) \|\s+([\d.]+)\s+([\d.]+) \|", ln)
    if m:
        x, y = int(m[1]), int(m[2])
        emit[(x, y, 0)] = float(m[3]); emit[(x, y, 1)] = float(m[4])
src = open(sys.argv[2]).read()
tab = src[src.index("kMoverSpeedP150"):]
tab = tab[:tab.index("};")]
s = {}
for m in re.finditer(r"\{(\d+), (\d+), (\d+), (\d+)\}", tab):
    x, y = int(m[1]), int(m[2]); s[(x, y, 0)] = int(m[3]); s[(x, y, 1)] = int(m[4])
keys = [k for k in emit if k in s]
missing = [k for k in emit if k not in s]
tmax = max(emit[k] for k in keys); tmean = sum(emit[k] for k in keys) / len(keys)
pred = sum(s[k] for k in keys) / sum(s[k] / emit[k] for k in keys)
print(f"movers {len(keys)} (not in table: {len(missing)})  mean {tmean:.3f}  max {tmax:.3f}  "
      f"predicted balanced {pred:.3f} ms/view")
v = {k: s[k] / emit[k] for k in keys}
mv = sum(v.values()) / len(v)
print("// {x, y, BRISC, NCRISC} re-derived on p150 (bh-30) from the t356 EMIT capture")
for x, y in sorted({(k[0], k[1]) for k in keys}, key=lambda t: (t[1], t[0])):
    print(f"    {{{x}, {y}, {round(1000 * v[(x, y, 0)] / mv)}, {round(1000 * v[(x, y, 1)] / mv)}}},")
