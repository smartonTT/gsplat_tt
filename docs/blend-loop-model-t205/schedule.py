#!/usr/bin/env python3
"""t205: a stall-free schedule of the single blend body, checked. No device.

SCHED below is the compiled single body (blend_pair_body<1,1>, microblock 2)
reordered, with LREGs renamed, so that no op reads the result of the
MAD-class op right before it. A2 is the same schedule split for the replay
design: x/y loads, replay 1 (13 ops, D=0), pushed middle, colours, INCRWC to
D=2m, replay 2 (18 ops + SETRWC, addresses relative to D).

The check runs both bodies symbolically (every value is the expression that
made it, so equal expressions mean bit-identical results) and compares every
DEST word they write and the constant LREGs. It also counts MAD -> use stalls.
Run: python3 schedule.py > out/schedule.txt
"""
import re
import sys

import stalls

M = 2                                     # the compiled body below is microblock 2
X, Y, T, R, G, B = (256 + 2 * M, 320 + 2 * M, 192 + 2 * M, 2 * M, 64 + 2 * M, 128 + 2 * M)

FRONT = """sfpload L6,384,0,7
sfpload L5,386,0,7
sfpadd L2,L10,L2,L6,2
sfpadd L1,L10,L1,L5,2
sfpmul L5,L2,L2,L9,0
sfpload L4,388,0,7
sfpmul L4,L4,L5,L9,0
sfpmul L2,L2,L1,L9,0
sfpload L3,390,0,7
sfpmad L2,L3,L2,L4,0
sfpmul L1,L1,L1,L9,0
sfpload L0,392,0,7
sfpmad L0,L0,L1,L2,0"""

MIDDLE = """sfpnop
sfpswap L0,L9,1
sfpnop
sfpmul L0,L0,L12,L9,0
sfpload L4,406,0,7
sfpaddi L0,17150,0
sfpnop
sfpswap L9,L0,1
sfpnop
sfpexexp L1,L0,0
sfpexman L0,L0,0
sfpshft L0,L1,0x000,0
sfpexexp L1,L0,1
sfpexman L0,L0,1
sfpcast L0,L0,0
sfpmad L2,L0,L13,L14,0
sfpload L5,394,0,7
sfpmad L0,L0,L2,L4,0
sfpload L6,408,0,7
sfpsetexp L1,L0,0x000,0
sfpmul L0,L5,L1,L9,0
sfpload L7,396,0,7
sfpswap L0,L6,1
sfpnop
sfpload L3,398,0,7
sfpload L4,400,0,7
sfpload L5,402,0,7"""

# Replay 2 body; {t},{r},{g},{b} are the T/R/G/B addresses (absolute, or
# relative to D=2m in A2).
TAIL = """sfpmad L1,L7,L11,L0,0
sfpload L2,{t},0,7
sfpsetcc L1,0x000,0
sfpmov L0,L9,0
sfpencc 0x003,10
sfpmul L1,L0,L2,L9,0
sfpadd L0,L10,L10,L0,2
sfpload L6,{r},0,7
sfpmad L6,L1,L3,L6,0
sfpmul L0,L2,L0,L9,0
sfpstore L6,{r},0,7
sfpload L7,{g},0,7
sfpmad L7,L1,L4,L7,0
sfpload L6,{b},0,7
sfpstore L7,{g},0,7
sfpmad L6,L1,L5,L6,0
sfpstore L0,{t},0,7
sfpstore L6,{b},0,7"""

XY = f"sfpload L2,{X},0,7\nsfpload L1,{Y},0,7"
SCHED = "\n".join([XY, FRONT, MIDDLE, TAIL.format(t=T, r=R, g=G, b=B)])
A2 = "\n".join([XY, FRONT, MIDDLE, f"ttincrwc 0,{2 * M},0,0",
                TAIL.format(t=192, r=0, g=64, b=128), "ttsetrwc d0"])


def parse(text):
    out = []
    for line in text.strip().splitlines():
        op, _, a = line.strip().partition(" ")
        out.append((op, a.split(",") if a else []))
    return out


def run(ops):
    """Symbolic execution: returns (DEST writes, constant LREGs)."""
    L = {i: ("L", i) for i in range(16)}
    dest, cc, d = {}, None, 0

    def val(a):
        return dest.get(a, ("dest", a))

    def put(i, v):
        L[i] = ("sel", cc, v, L[i]) if cc else v

    for op, a in ops:
        r = [int(x[1:]) for x in a if re.fullmatch(r"L\d+", x)]
        imm = [x for x in a if not re.fullmatch(r"L\d+", x)]
        if op == "sfpload":
            put(r[0], val(int(imm[0]) + d))
        elif op == "sfpstore":
            ad = int(imm[0]) + d
            dest[ad] = ("sel", cc, L[r[0]], val(ad)) if cc else L[r[0]]
        elif op in ("sfpmad", "sfpmul", "sfpadd"):
            put(r[0], (op, imm[0], L[r[1]], L[r[2]], L[r[3]]))
        elif op == "sfpaddi":
            put(r[0], (op, imm[0], imm[1], L[r[0]]))
        elif op == "sfpswap":
            x, y = L[r[0]], L[r[1]]
            put(r[0], ("swap0", imm[0], x, y))
            put(r[1], ("swap1", imm[0], x, y))
        elif op in ("sfpexexp", "sfpexman", "sfpcast", "sfpmov"):
            put(r[0], (op, imm[0], L[r[1]]))
        elif op in ("sfpshft", "sfpsetexp"):
            put(r[0], (op, imm[0], imm[1], L[r[0]], L[r[1]]))
        elif op == "sfpsetcc":
            cc = ("cc", imm[0], imm[1], L[r[0]])
        elif op == "sfpencc":
            cc = None
        elif op == "ttincrwc":
            d += int(a[1])
        elif op == "ttsetrwc":
            d = 0
        elif op != "sfpnop":
            raise ValueError(op)
    return dest, {i: L[i] for i in range(9, 15)}, d


def mad_stalls(ops):
    sf = [o for o in ops if o[0].startswith("sfp")]
    return stalls.stalls(sf)


def main():
    orig = stalls.expand(stalls.bodies(stalls.DIS)[(1, 1)])
    ref = run(orig)
    print(f"compiled single body (microblock {M}): {len(orig)} ops, "
          f"{mad_stalls(orig)} MAD->use stalls")
    for name, text in (("F schedule", SCHED), ("A2 split", A2)):
        ops = parse(text)
        got = run(ops)
        n_sfpu = sum(o.startswith("sfp") for o, _ in ops)
        same = got[0] == ref[0] and got[1] == ref[1] and got[2] == 0
        print(f"{name}: {n_sfpu} SFPU ops, {mad_stalls(ops)} MAD->use stalls, "
              f"DEST writes {sorted(got[0])}, identical to compiled: {same}")
        if not same:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
