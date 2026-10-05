#!/usr/bin/env python3
"""Task #226: no-device cycle model of pfwc TRISC cuts (MODELED, not measured).

Inputs are measured: T1 (math) step times from
docs/pfwc-writer-split-t207/dev-t221/out/prof-both-pc_split.txt and per-op costs
from docs/pfwc-breakdown-t197 (STEPCYC=2). Everything printed is a model.
"""
CLK = 1.35e9          # Hz
CHUNKS = 54.4         # chunks / core / view (t197)
CYC_PER_MS = CLK * 1e-3 / CHUNKS  # cycles per chunk for 1 ms/view of T1 time

COPY, UNARY, BINARY = 154, 160, 140   # t197 STEPCYC=2 (binary = mul/add mix)
UNIT = 230      # one section sync or one pack, fitted below
RADII_OPS = 1460  # relu + sqrt + ceil on one tile (fit from radx/rady residual)
CONIC = 900       # pfwc_conic_unroll pass (fit from the conic step residual)

def ms(c):
    return c / CYC_PER_MS

# Measured T1 step times (ms/view), default tip 21edcf0.
t1 = dict(xform=.206, recip=.046, depth=.017, means=.072, cov_cam=.256, a=.185,
          b=.187, c=.136, conic=.073, radx=.209, rady=.207, vis=.616)
print(f"T1 sum {sum(t1.values()):.3f} ms/view = {sum(t1.values())*CYC_PER_MS:.0f} cyc/chunk")

# --- fit check: steps a..rady (6 sections, 36 copies, 24 unary, 57 binary, 10 packs)
cur_cov = sum(t1[k] for k in "a b c conic radx rady".split()) * CYC_PER_MS
ops = 36 * COPY + 24 * UNARY + 57 * BINARY + CONIC + 2 * RADII_OPS
print(f"steps a..rady: {cur_cov:.0f} cyc, op model {ops:.0f}, residual "
      f"{cur_cov-ops:.0f} over 16 sec+pack units = {(cur_cov-ops)/16:.0f}/unit")

def pass_cyc(instr, cpi):
    return instr * cpi * 32   # 32 SFPU vectors per tile

# Calibrate SFPU cycles/instruction on the t206 cov_cam pass (same SFPMUL/SFPADD
# chain style): step 0.256 ms = 6 copies + 108 staging instr + 114 instr x 32
# vectors + 1 sec + 6 packs.
cc = t1["cov_cam"] * CYC_PER_MS
cpi_cc = (cc - 6 * COPY - 108 - 7 * UNIT) / (114 * 32)
print(f"cov_cam pass calibration: {cpi_cc:.2f} cycles per SFPU instruction")

res = {}
for name, cpi in (("optimistic", 1.0), ("central", 1.2), ("pessimistic", 2.0)):
    # P2: S_AC (8 copies, 52-instr pass, radii tile ops on 2 slots, 1 sec + 6 packs)
    s_ac = 8 * COPY + pass_cyc(52, cpi) + 2 * (RADII_OPS + UNARY) + 7 * UNIT
    # S_BC (9 copies, 35-instr pass, conic pass, 1 sec + 4 packs)
    s_bc = 9 * COPY + pass_cyc(35, cpi) + CONIC + 5 * UNIT
    p2 = cur_cov - (s_ac + s_bc)
    # X: steps 1-5 as one section (3 copies, staging, 37+17 instr passes, recip, 1 sec + 9 packs)
    cur_x = sum(t1[k] for k in "xform recip depth means".split()) * CYC_PER_MS
    new_x = 3 * COPY + 50 + pass_cyc(37 + 17, cpi) + 400 + 10 * UNIT
    x = cur_x - new_x
    res[name] = (p2, x)
    print(f"{name:11s} cpi {cpi}: P2 new {s_ac+s_bc:.0f} cyc, saves {p2:.0f} cyc = {ms(p2):.3f} ms/view;"
          f" X saves {x:.0f} cyc = {ms(x):.3f} ms/view")

# Radii-only reuse (R): conic section also does radii; per radius drop 5 copies,
# 4 unary, 10 binary, 1 section unit.
r = 2 * (5 * COPY + 4 * UNARY + 10 * BINARY + UNIT)
print(f"R (radii reuse only) saves {r:.0f} cyc = {ms(r):.3f} ms/view")

# Floor: TRISC wall vs writers (t221: BRISC 1.12, NCRISC 1.19 ms).
wall = 2.212
for name, (p2, x) in res.items():
    after = wall - ms(p2)
    print(f"{name:11s}: TRISC after P2 {after:.2f} ms, after P2+X {after-ms(x):.2f} ms; "
          f"floor max(TRISC, 1.19) = {max(after, 1.19):.2f}")
