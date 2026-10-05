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
t219 adds PAIR, the hand schedule of the pair body (blend_pair_body<J,3>,
microblocks 2J and 2J+1): same 109 ops as the compiled pair, 0 MAD -> use
stalls instead of 22, LREGs L0-L7 only. main() checks every single (32) and
every pair (16) of the t189 disassembly against these listings, plus the
issue rules: an sfpnop after every sfpswap, no swap reading the result of the
MAD-class op right before it, no write to the destination of the MAD-class op
right before it, setcc/encc regions as compiled, the same ops (opcode,
immediates and modes) as the compiled body.
Run: python3 schedule.py > out/schedule.txt
"""
import collections
import re
import sys

import stalls

M = 2                                     # the compiled body below is microblock 2


def addrs(m):
    """x, y, T, R, G, B DEST addresses of microblock m (dst_reg index m)."""
    return 256 + 2 * m, 320 + 2 * m, 192 + 2 * m, 2 * m, 64 + 2 * m, 128 + 2 * m


X, Y, T, R, G, B = addrs(M)

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

# t219: the pair body, microblocks a = 2J and b = 2J+1. Chain b runs a few ops
# behind chain a, so every MAD-class result has one op between it and its
# first reader. Shared: MX MY A B C OP CR CG CB (one load each, as compiled);
# C0, K99 and FL are loaded once per chain, as compiled.
PAIR = """sfpload L0,384,0,7
sfpload L1,386,0,7
sfpload L2,{xa},0,7
sfpload L3,{ya},0,7
sfpadd L2,L10,L2,L0,2
sfpload L4,{xb},0,7
sfpadd L3,L10,L3,L1,2
sfpadd L4,L10,L4,L0,2
sfpload L0,{yb},0,7
sfpmul L5,L2,L2,L9,0
sfpadd L0,L10,L0,L1,2
sfpload L1,388,0,7
sfpmul L5,L1,L5,L9,0
sfpmul L6,L4,L4,L9,0
sfpmul L2,L2,L3,L9,0
sfpmul L6,L1,L6,L9,0
sfpload L1,390,0,7
sfpmad L2,L1,L2,L5,0
sfpmul L4,L4,L0,L9,0
sfpmul L3,L3,L3,L9,0
sfpmad L4,L1,L4,L6,0
sfpload L1,392,0,7
sfpmad L2,L1,L3,L2,0
sfpmul L0,L0,L0,L9,0
sfpswap L2,L9,1
sfpnop
sfpmad L0,L1,L0,L4,0
sfpmul L2,L2,L12,L9,0
sfpswap L0,L9,1
sfpnop
sfpaddi L2,17150,0
sfpmul L0,L0,L12,L9,0
sfpnop
sfpswap L9,L2,1
sfpnop
sfpaddi L0,17150,0
sfpexexp L1,L2,0
sfpexman L2,L2,0
sfpnop
sfpswap L9,L0,1
sfpnop
sfpshft L2,L1,0x000,0
sfpexexp L3,L0,0
sfpexexp L1,L2,1
sfpexman L0,L0,0
sfpexman L2,L2,1
sfpshft L0,L3,0x000,0
sfpcast L2,L2,0
sfpexexp L3,L0,1
sfpmad L4,L2,L13,L14,0
sfpexman L0,L0,1
sfpload L5,406,0,7
sfpmad L2,L2,L4,L5,0
sfpcast L0,L0,0
sfpsetexp L1,L2,0x000,0
sfpmad L4,L0,L13,L14,0
sfpload L5,406,0,7
sfpmad L0,L0,L4,L5,0
sfpload L4,394,0,7
sfpsetexp L3,L0,0x000,0
sfpmul L2,L4,L1,L9,0
sfpmul L0,L4,L3,L9,0
sfpload L1,408,0,7
sfpswap L2,L1,1
sfpnop
sfpload L3,408,0,7
sfpswap L0,L3,1
sfpnop
sfpload L1,396,0,7
sfpmad L1,L1,L11,L2,0
sfpload L3,396,0,7
sfpsetcc L1,0x000,0
sfpmov L2,L9,0
sfpencc 0x003,10
sfpmad L3,L3,L11,L0,0
sfpload L4,{ta},0,7
sfpsetcc L3,0x000,0
sfpmov L0,L9,0
sfpencc 0x003,10
sfpmul L1,L2,L4,L9,0
sfpload L5,{tb},0,7
sfpadd L2,L10,L10,L2,2
sfpmul L3,L0,L5,L9,0
sfpload L6,398,0,7
sfpadd L0,L10,L10,L0,2
sfpload L7,{ra},0,7
sfpmad L7,L1,L6,L7,0
sfpmul L4,L4,L2,L9,0
sfpstore L7,{ra},0,7
sfpload L7,{rb},0,7
sfpmad L7,L3,L6,L7,0
sfpstore L4,{ta},0,7
sfpstore L7,{rb},0,7
sfpmul L5,L5,L0,L9,0
sfpload L6,400,0,7
sfpstore L5,{tb},0,7
sfpload L7,{ga},0,7
sfpmad L7,L1,L6,L7,0
sfpload L5,{gb},0,7
sfpstore L7,{ga},0,7
sfpmad L5,L3,L6,L5,0
sfpload L6,402,0,7
sfpstore L5,{gb},0,7
sfpload L7,{ba},0,7
sfpmad L7,L1,L6,L7,0
sfpload L5,{bb},0,7
sfpmad L5,L3,L6,L5,0
sfpstore L7,{ba},0,7
sfpstore L5,{bb},0,7"""


def xy(m):
    x, y, *_ = addrs(m)
    return f"sfpload L2,{x},0,7\nsfpload L1,{y},0,7"


def single(m):
    """F single body of microblock m."""
    _, _, t, r, g, b = addrs(m)
    return "\n".join([xy(m), FRONT, MIDDLE, TAIL.format(t=t, r=r, g=g, b=b)])


def incrwc(n):
    """INCRWC D steps (even, <= 14: the field is 4 bits) that sum to n."""
    out = []
    while n > 0:
        out.append(min(n, 14))
        n -= out[-1]
    return out


def a2(m):
    """A2 single body of microblock m, replays expanded: x/y, replay 1 (FRONT),
    pushed MIDDLE, INCRWC to D=2m, replay 2 (TAIL relative to D + SETRWC)."""
    return "\n".join([xy(m), FRONT, MIDDLE] + [f"ttincrwc 0,{k},0,0" for k in incrwc(2 * m)]
                     + [TAIL.format(t=192, r=0, g=64, b=128), "ttsetrwc d0"])


def pair(j):
    """F pair body of microblocks 2j and 2j+1."""
    a, b = addrs(2 * j), addrs(2 * j + 1)
    return PAIR.format(xa=a[0], ya=a[1], ta=a[2], ra=a[3], ga=a[4], ba=a[5],
                       xb=b[0], yb=b[1], tb=b[2], rb=b[3], gb=b[4], bb=b[5])


SCHED = single(M)
A2 = a2(M)


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
        if i >= 8:                        # constant LREGs: writes are dropped (the
            return                        # bodies park dead swap outputs in L9)
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


def rules(ops, ref):
    """Issue-rule and op-set violations of ops against the compiled body ref."""
    bad = []
    for i, (op, a) in enumerate(ops):
        rd, wr = stalls.regs(op, a)
        if wr - set(range(8)) - ({9} if op == "sfpswap" else set()):
            bad.append(f"{i}: {op} writes a constant LREG")
        if rd - set(range(8)) - set(range(9, 15)):
            bad.append(f"{i}: {op} reads LREG {sorted(rd)}")
        prev = ops[i - 1] if i else ("", [])
        if prev[0] == "sfpswap" and op != "sfpnop":
            bad.append(f"{i}: no sfpnop after sfpswap")
        pw = stalls.regs(*prev)[1] if prev[0] in stalls.MAD else set()
        if op == "sfpswap" and pw & rd:
            bad.append(f"{i}: sfpswap reads the MAD result right before it")
        if pw & wr:
            bad.append(f"{i}: {op} overwrites the MAD result right before it")
        if op == "sfpsetcc" and [o for o, _ in ops[i + 1:i + 3]] != ["sfpmov", "sfpencc"]:
            bad.append(f"{i}: setcc region is not setcc, mov, encc")

    def opset(o):
        """Ops without their LREGs; DEST addresses made absolute (imm + D)."""
        c, d = collections.Counter(), 0
        for n, a in o:
            if n == "ttincrwc":
                d += int(a[1])
            elif n == "ttsetrwc":
                d = 0
            elif n.startswith("sfp"):
                imm = [x for x in a if not re.fullmatch(r"L\d+", x)]
                if n in ("sfpload", "sfpstore"):
                    imm[0] = str(int(imm[0]) + d)
                c[(n, tuple(imm))] += 1
        return c
    if ref is not None and opset(ops) != opset(ref):
        bad.append(f"ops differ: +{opset(ops) - opset(ref)} -{opset(ref) - opset(ops)}")
    return bad


def check(name, ops, ref):
    """(identical, stalls, n SFPU ops, rule violations) of ops against ref."""
    got, want = run(ops), run(ref)
    same = got[0] == want[0] and got[1] == want[1] and got[2] == 0
    # two singles in one body have 11 ops more than the compiled pair by design
    return same, mad_stalls(ops), sum(o.startswith("sfp") for o, _ in ops), \
        rules(ops, None if "singles" in name else ref)


def main():
    bs = {k: stalls.expand(v) for k, v in stalls.bodies(stalls.DIS).items()}
    orig = bs[(1, 1)]
    print(f"compiled single body (microblock {M}): {len(orig)} ops, "
          f"{mad_stalls(orig)} MAD->use stalls")
    ok = True
    for name, text in (("F schedule", SCHED), ("A2 split", A2)):
        same, st, n, bad = check(name, parse(text), orig)
        got = run(parse(text))
        print(f"{name}: {n} SFPU ops, {st} MAD->use stalls, "
              f"DEST writes {sorted(got[0])}, identical to compiled: {same}")
        ok &= same and not bad
    # t219: every body of the jump table. Singles: F and A2. Pairs: the F pair,
    # and the fallbacks two F singles / two A2 singles (the A2 pair body).
    kinds = collections.defaultdict(list)
    for (j, pm), ref in sorted(bs.items()):
        if pm == 3:
            cands = [("F pair", pair(j)),
                     ("2 F singles", single(2 * j) + "\n" + single(2 * j + 1)),
                     ("A2 pair (2 A2 singles)", a2(2 * j) + "\n" + a2(2 * j + 1))]
        else:
            m = 2 * j + (pm == 2)
            cands = [("F single", single(m)), ("A2 single", a2(m))]
        for name, text in cands:
            same, st, n, bad = check(name, parse(text), ref)
            kinds[name].append((same, st, n, mad_stalls(ref), len(ref), bad))
            if not same or bad:
                print(f"FAIL {name} J={j} PM={pm}: identical {same}, {bad[:4]}")
                ok = False
    for name, rs in kinds.items():
        print(f"{name}: {len(rs)} bodies, all identical to compiled: {all(r[0] for r in rs)}, "
              f"SFPU ops {sorted(set(r[2] for r in rs))} (compiled {sorted(set(r[4] for r in rs))}), "
              f"MAD->use stalls max {max(r[1] for r in rs)} (compiled {sorted(set(r[3] for r in rs))}), "
              f"rule violations {sum(len(r[5]) for r in rs)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
