#!/usr/bin/env python3
"""t584: check the BLEND_EXP1 (degree-1 minimax exp) source. No device.

1. Constants: the blend_stage_hoist EXP1 values equal the t580 model's exp1
   coefficients (docs/precision-model/model.py, E1['1'] at d285d66e) as float32,
   and the poly stays in [1, 2) on the whole 23-bit fraction (setexp needs it);
   max relative error vs 2^u is the minimax 3 %.
2. Bodies: with BLEND_SCHED 2 and BLEND_EXP1 1, every A2 single (after replay
   expansion) keeps the t205/t219 issue rules (an sfpnop after every sfpswap, no
   swap reading the MAD-class result right before it, no write to the
   destination of the MAD-class op right before it, no write to a constant
   LREG) and has the expected exp: one MAD x * L12 + bias (bias loaded from
   S_C0), no sfpaddi, one poly MAD x * L13 + L14, 3 ops fewer than EXP1 0.
   BLEND_SCHED 1 with EXP1 hits the #error.
Run: python3 check_exp1.py [path/to/tt-metal]
"""
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "blend-sched-t219"))
import check_cpp as cc  # noqa: E402  (chdirs to blend-loop-model-t205)
import stalls  # noqa: E402

KERNEL = os.path.join(HERE, "..", "..", "render", "kernels", "compute", "alpha_blend_compute_mb.cpp")
ERR = []
BS_S_C0 = None


def model_e1():
    q0, q1, k = 0.97030, 0.97000, 1.0307             # model.py E1['1']
    return np.float32(q0 * k), np.float32(q1 * k / 2 ** 23), np.float32(127.0 - np.log2(k))


def hoist_consts(src):
    m = re.search(r"inline void blend_stage_hoist\(\) \{(.*?)\n\}", src, re.S)
    txt = cc.preprocess(m[1], {"BLEND_EXP1": 1})
    v = dict(re.findall(r"(vConstFloatPrgm[12]|S_C0\]) = ([0-9.e+-]+)f;", txt))
    return np.float32(v["vConstFloatPrgm2"]), np.float32(v["vConstFloatPrgm1"]), np.float32(v["S_C0]"])


def check_consts(src):
    got, want = hoist_consts(src), model_e1()
    print(f"constants c0 c1 bias: kernel {[float(x) for x in got]} model {[float(x) for x in want]}")
    if any(g != w for g, w in zip(got, want)):
        ERR.append("EXP1 constants differ from the t580 model")
    c0, c1, bias = got
    f = np.arange(0, 1 << 23, 97, dtype=np.float32)
    p = f * c1 + c0
    if p.min() < 1 or p.max() >= 2:
        ERR.append(f"poly leaves [1, 2): {p.min()} .. {p.max()}")
    # exp(x) = 2^(x/ln2) = 2^(xl - bias) with xl = x/ln2 + bias -> rel err of k-scaled Q(u) vs 2^(u + log2 k)
    u = f / np.float32(1 << 23)
    rel = np.abs(p / np.exp2(u + (127 - bias)) - 1)
    print(f"poly range {p.min():.6f} .. {p.max():.6f}, max rel err vs 2^u {rel.max():.4f}")
    if rel.max() > 0.031:
        ERR.append(f"max rel err {rel.max()}")


def lreg_args(op, vals):
    names = cc.DISMAP[op][1]
    return [f"L{v}" if n.startswith("lreg") else str(v) for n, v in zip(names, vals)]


def bodies(src, ops, exp1, level=2):
    consts = {}
    for m in re.finditer(r"constexpr uint32_t ((?:DR|S)_\w+ = [^;]+);", src):
        for part in m[1].split(","):
            k, v = part.split("=")
            consts[k.strip()] = eval(v)
    a, b = src.index("#ifndef BLEND_SCHED"), src.index("using BlendBodyFn")
    env = {"BLEND_SCHED": level, "BLEND_CONST_HOIST": 1, "BLEND_PIXEL_FLOOR": 1, "BLEND_FPU_QF_ABL": 0,
           "BLEND_EXP1": exp1}
    n0 = len(cc.ERR)
    k = cc.Kernel(cc.preprocess(src[a:b], env), consts, ops)
    if level == 1:
        return cc.ERR[n0:]
    buf = [None] * 32
    cc.replay(k.call("blend_sched_record", []), buf)
    return {m: cc.replay(k.call("blend_pair_body", [m // 2, 1 + (m & 1)]), list(buf)) for m in range(32)}


def rules(seq):
    bad, sf = [], [(o, lreg_args(o, v)) for o, v in seq if o.startswith("sfp")]
    for i, (op, a) in enumerate(sf):
        rd, wr = stalls.regs(op, a)
        if wr - set(range(8)) - ({9} if op == "sfpswap" else set()):
            bad.append(f"{i}: {op} writes a constant LREG")
        prev = sf[i - 1] if i else ("", [])
        if prev[0] == "sfpswap" and op != "sfpnop":
            bad.append(f"{i}: no sfpnop after sfpswap")
        pw = stalls.regs(*prev)[1] if prev[0] in stalls.MAD else set()
        if op == "sfpswap" and pw & rd:
            bad.append(f"{i}: sfpswap reads the MAD result right before it")
        if pw & wr:
            bad.append(f"{i}: {op} overwrites the MAD result right before it")
    return bad, sf


def check_bodies(src, ops):
    old, new = bodies(src, ops, 0), bodies(src, ops, 1)
    for m in range(32):
        bad, sf = rules(new[m])
        ERR.extend(f"m{m} {b}" for b in bad)
        if len(old[m]) - len(new[m]) != 3:
            ERR.append(f"m{m}: {len(old[m])} -> {len(new[m])} ops, want 3 fewer")
        if any(o == "sfpaddi" for o, _ in sf):
            ERR.append(f"m{m}: sfpaddi left")
        mi = [i for i, (o, a) in enumerate(sf) if o == "sfpmad" and ("L12" in a[1:3] or "L13" in a[1:3])]
        mads = [sf[i][1] for i in mi]
        if len(mads) != 2 or mads[0][2] != "L12" or mads[1][2:4] != ["L13", "L14"]:
            ERR.append(f"m{m}: exp MADs {mads}")
            continue
        bias = mads[0][3]
        lds = [a for o, a in sf[:mi[0]] if o == "sfpload" and a[0] == bias]
        if not lds or int(lds[-1][1]) != BS_S_C0:
            ERR.append(f"m{m}: bias {bias} not loaded from S_C0 ({lds[-1:]})")
    print(f"A2 singles with EXP1: {32 - len({e.split()[0] for e in ERR if e.startswith('m')})}/32 pass; "
          f"ops {len(old[0])} -> {len(new[0])}")
    e1 = bodies(src, ops, 1, level=1)
    if not any("BLEND_EXP1" in e for e in e1):
        ERR.append("BLEND_SCHED 1 + EXP1 does not hit the #error")
    else:
        print("level 1 + EXP1: #error active as intended")


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/dev/tt-metal")
    src = open(KERNEL).read()
    check_consts(src)
    ops = cc.load_ops(os.path.join(root, cc.OPS_H))
    global BS_S_C0
    BS_S_C0 = 2 * (eval(re.search(r"constexpr uint32_t DR_S = ([^;]+);", src)[1])
                   + int(re.search(r"S_C0 = (\d+)", src)[1]))
    check_bodies(src, ops)
    for e in ERR:
        print("FAIL:", e)
    print("check_exp1:", "FAIL" if ERR else "OK")
    return 1 if ERR else 0


if __name__ == "__main__":
    sys.exit(main())
