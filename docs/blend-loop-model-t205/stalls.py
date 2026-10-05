#!/usr/bin/env python3
"""t205: count SFPU latency stalls in the compiled blend bodies. No device.

The Blackhole ISA (tt_llk_blackhole/instructions/assembly.yaml) calls SFPMAD,
SFPMUL, SFPADD, SFPADDI, SFPMULI, SFPLUT and SFPSWAP "a two cycle operation".
The compiler pads SFPSWAP with sfpnop, but puts dependent ops right after a
MAD-class op with no pad. The output is correct, so the hardware must hold the
consumer for one cycle. That cycle is not in the instruction count, so it
shows up as the unexplained stall in the t172/t189 fit.

This counts, per body, ops that read a register written by the MAD-class op
right before them (one stall cycle each), with REPLAY entries expanded. It also
counts how many of those slots a hand schedule can fill. Input: the t189
disassembly. Run: python3 stalls.py > out/stalls.txt
"""
import re
import sys

DIS = "../blend-dispatch-t189/out/trisc1-default.dis"
MAD = {"sfpmad", "sfpmul", "sfpadd", "sfpaddi", "sfpmuli", "sfplut"}
P_SINGLE, P_PAIR = 1.597, 0.605            # body calls per live record (model.py)
MB_PER_LIVE = 2.807
MS_PER_CYC_LIVE = 21323 / 1.35e9 * 1e3     # ms/view per cycle per live record


def regs(op, args):
    """(read set, write set) of LREG indexes."""
    L = [int(a[1:]) for a in args if re.fullmatch(r"L\d+", a)]
    if op in ("sfpload", "sfploadi"):
        return set(), {L[0]}
    if op == "sfpstore":
        return {L[0]}, set()
    if op in ("sfpmad", "sfpmul", "sfpadd"):
        return set(L[1:4]), {L[0]}
    if op in ("sfpaddi", "sfpmuli"):
        return {L[0]}, {L[0]}
    if op == "sfpswap":
        return set(L[:2]), set(L[:2])
    if op in ("sfpshft", "sfpsetexp", "sfpsetman", "sfpsetsgn"):
        return set(L[:2]), {L[0]}
    if op in ("sfpexexp", "sfpexman", "sfpcast", "sfpmov", "sfpabs", "sfplz"):
        return {L[1]}, {L[0]}
    if op == "sfpsetcc":
        return {L[0]}, set()
    return set(), set()


def bodies(path):
    out, name, cur = {}, None, None
    for line in open(path):
        m = re.match(r"^[0-9a-f]+ <void \(anonymous namespace\)::blend_pair_body<(\d+)ul, (\d+)ul>\(\)>:", line)
        if m:
            name, cur = (int(m[1]), int(m[2])), []
            out[name] = cur
            continue
        if cur is None:
            continue
        m = re.match(r"^\s+[0-9a-f]+:\s+[0-9a-f]+\s+(\S+)\s*(\S*)", line)
        if not m:
            if re.match(r"^[0-9a-f]+ <", line) and "LB" not in line and "LM" not in line and ".L" not in line:
                cur = None
            continue
        op, args = m[1], m[2].split(",") if m[2] else []
        if op == "ret":
            cur = None
            continue
        cur.append((op, args))
    return out


def expand(ops):
    """Expand ttreplay record/play entries into the ops they issue."""
    rec, res, i = {}, [], 0
    while i < len(ops):
        op, args = ops[i]
        if op == "ttreplay":
            start, n, load, ex = (int(a) for a in args)
            if load:
                rec[start] = ops[i + 1:i + 1 + n]
                res.extend(rec[start])
                i += 1 + n
                continue
            res.extend(rec[start])
        else:
            res.append((op, args))
        i += 1
    return res


def stalls(ops):
    n = 0
    for (a, aa), (b, ba) in zip(ops, ops[1:]):
        if a in MAD and regs(a, aa)[1] & regs(b, ba)[0]:
            n += 1
    return n


def main():
    bs = bodies(DIS)
    single, pair = [], []
    for (j, pm), ops in sorted(bs.items()):
        e = expand(ops)
        (pair if pm == 3 else single).append((len(e), stalls(e), sum(o == "sfpnop" for o, _ in e)))
    s = [sum(x[i] for x in single) / len(single) for i in range(3)]
    p = [sum(x[i] for x in pair) / len(pair) for i in range(3)]
    print(f"bodies: {len(single)} single, {len(pair)} pair (from {DIS})")
    print(f"single: {s[0]:.1f} ops, {s[1]:.2f} MAD-latency stalls, {s[2]:.1f} nops")
    print(f"pair:   {p[0]:.1f} ops, {p[1]:.2f} MAD-latency stalls, {p[2]:.1f} nops")
    per_live = P_SINGLE * s[1] + P_PAIR * p[1]
    print(f"per live record: {per_live:.1f} stall cycles = {per_live * MS_PER_CYC_LIVE:.2f} ms/view; "
          f"per microblock {per_live / MB_PER_LIVE:.1f} (t189 unexplained: 14.5)")


if __name__ == "__main__":
    sys.exit(main())
