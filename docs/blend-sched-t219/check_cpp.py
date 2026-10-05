#!/usr/bin/env python3
"""t219: check the BLEND_SCHED source against the t205 listings. No device.

1. Encoder: the TT_OP_* formulas of the Blackhole ckernel_ops.h, fed with the
   operands in disassembly order, give the instruction word of every op of the
   kinds used here in the t189 disassembly (stream word = rotl32(word, 2)).
2. The kernel's BLEND_SCHED code (alpha_blend_compute_mb.cpp), preprocessed
   per level and expanded for every blend_pair_body<J, PM>, issues exactly the
   schedule.py listings: level 1 single(m) / pair(j); level 2 a2(m) once the
   replays are expanded from the buffer that blend_sched_record() fills. Each
   op is compared as the word its TTI_ macro emits and as disassembly text, so
   the BS_ operand permutations are checked too.
Run from this directory: python3 check_cpp.py [path/to/ckernel_ops.h]
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "blend-loop-model-t205"))
os.chdir(os.path.join(HERE, "..", "blend-loop-model-t205"))
import schedule  # noqa: E402

KERNEL = os.path.join(HERE, "..", "..", "render", "kernels", "compute", "alpha_blend_compute_mb.cpp")
DIS = os.path.join(HERE, "..", "blend-dispatch-t189", "out", "trisc1-default.dis")
OPS_H = "tt_metal/tt-llk/tt_llk_blackhole/common/inc/ckernel_ops.h"
P_SETRWC = {"CLR_NONE": 0, "SET_D": 4}     # ckernel_instr_params.h

# disassembly mnemonic -> (TT_OP name, TT_OP parameters in disassembly operand order)
DISMAP = {
    "sfpload": ("SFPLOAD", ["lreg_ind", "dest_reg_addr", "instr_mod0", "sfpu_addr_mode"]),
    "sfpstore": ("SFPSTORE", ["lreg_ind", "dest_reg_addr", "instr_mod0", "sfpu_addr_mode"]),
    "sfpmad": ("SFPMAD", ["lreg_dest", "lreg_src_a", "lreg_src_b", "lreg_src_c", "instr_mod1"]),
    "sfpmul": ("SFPMUL", ["lreg_dest", "lreg_src_a", "lreg_src_b", "lreg_src_c", "instr_mod1"]),
    "sfpadd": ("SFPADD", ["lreg_dest", "lreg_src_a", "lreg_src_b", "lreg_src_c", "instr_mod1"]),
    "sfpaddi": ("SFPADDI", ["lreg_dest", "imm16_math", "instr_mod1"]),
    "sfpswap": ("SFPSWAP", ["lreg_dest", "lreg_src_c", "instr_mod1"]),
    "sfpexexp": ("SFPEXEXP", ["lreg_dest", "lreg_c", "instr_mod1"]),
    "sfpexman": ("SFPEXMAN", ["lreg_dest", "lreg_c", "instr_mod1"]),
    "sfpmov": ("SFPMOV", ["lreg_dest", "lreg_c", "instr_mod1"]),
    "sfpcast": ("SFPCAST", ["lreg_dest", "lreg_src_c", "instr_mod1"]),
    "sfpshft": ("SFPSHFT", ["lreg_dest", "lreg_c", "imm12_math", "instr_mod1"]),
    "sfpsetexp": ("SFPSETEXP", ["lreg_dest", "lreg_c", "imm12_math", "instr_mod1"]),
    "sfpsetcc": ("SFPSETCC", ["lreg_c", "imm12_math", "instr_mod1"]),
    "sfpencc": ("SFPENCC", ["imm12_math", "instr_mod1"]),
    "sfpnop": ("SFPNOP", []),
    "ttreplay": ("REPLAY", ["start_idx", "len", "execute_while_loading", "load_mode"]),
    "ttincrwc": ("INCRWC", ["rwc_cr", "rwc_d", "rwc_b", "rwc_a"]),
    "ttsetrwc": ("SETRWC", ["clear_ab_vld", "rwc_cr", "rwc_d", "rwc_b", "rwc_a", "BitMask"]),
}
TT2DIS = {v[0]: k for k, v in DISMAP.items()}
ERR = []


def fail(msg):
    ERR.append(msg)


def load_ops(path):
    text = re.sub(r"\\\n", " ", open(path).read())
    ops = {}
    for m in re.finditer(r"^#define TT_OP_(\w+)(?:\(([^)]*)\))?\s+TT_OP\((0x[0-9a-fA-F]+),\s*(.*)\)\s*$", text, re.M):
        params = [p.strip() for p in m[2].split(",")] if m[2] else []
        ops[m[1]] = (params, int(m[3], 16), m[4])
    return ops


def encode(ops, name, vals):
    params, opcode, expr = ops[name]
    return ((opcode << 24) + eval(expr, {}, dict(vals))) & 0xFFFFFFFF


def encode_dis(ops, mn, args):
    name, order = DISMAP[mn]
    vals = {p: 0 for p in ops[name][0]}
    vals.update(zip(order, args))
    for p, w in (("imm16_math", 0xFFFF), ("imm12_math", 0xFFF)):
        if p in vals:
            vals[p] &= w
    return encode(ops, name, vals)


def num(a):
    a = a.strip()
    return int(a[1:]) if re.fullmatch(r"L\d+", a) else int(a, 0)


def check_encoder(ops):
    seen = {}
    for line in open(DIS):
        m = re.match(r"^\s+[0-9a-f]+:\s+([0-9a-f]{8})\s+(\S+)\s*(\S*)", line)
        if not m or m[2] not in DISMAP:
            continue
        stream = int(m[1], 16)
        word = ((stream >> 2) | (stream << 30)) & 0xFFFFFFFF
        args = [num(a) for a in m[3].split(",")] if m[3] else []
        ok = encode_dis(ops, m[2], args) == word
        n, bad = seen.get(m[2], (0, 0))
        seen[m[2]] = (n + 1, bad + (not ok))
        if not ok:
            fail(f"encoder: {m[2]} {m[3]} -> {encode_dis(ops, m[2], args):08x}, dis word {word:08x}")
    return seen


def preprocess(src, env):
    """Active lines of src for the macro values in env (#if/#ifndef/#elif/#else/#endif)."""
    def ev(e):
        e = re.sub(r"defined\((\w+)\)", lambda m: "1" if m[1] in env else "0", e)
        e = e.replace("||", " or ").replace("&&", " and ")
        e = re.sub(r"!(?!=)", " not ", e)
        e = re.sub(r"\b[A-Za-z_]\w*\b", lambda m: m[0] if m[0] in ("or", "and", "not")
                   else str(env.get(m[0], 0)), e)
        return bool(eval(e))
    out, stack = [], []          # stack entries: [active, taken]
    for line in src.splitlines():
        s = line.strip()
        act = all(a for a, _ in stack)
        d = re.match(r"#\s*(ifndef|ifdef|if|elif|else|endif|define|error)\b\s*(.*)", s)
        if d:
            k, rest = d[1], re.sub(r"//.*", "", d[2]).strip()
            if k in ("if", "ifdef", "ifndef"):
                v = (rest.split()[0] not in env) if k == "ifndef" else \
                    (rest.split()[0] in env) if k == "ifdef" else (ev(rest) if act else False)
                stack.append([v, v])
            elif k == "elif":
                v = not stack[-1][1] and ev(rest)
                stack[-1] = [v, stack[-1][1] or v]
            elif k == "else":
                stack[-1] = [not stack[-1][1], True]
            elif k == "endif":
                stack.pop()
            elif act and k == "error":
                fail(f"#error active: {rest}")
            elif act and k == "define":
                out.append(s)
                m = re.match(r"(\w+)\s+(\S+)$", rest)
                if m and m[1] not in env:
                    env[m[1]] = int(m[2].rstrip("u"), 0)
            continue
        if act:
            out.append(line)
    return "\n".join(out)


def match_close(text, i, op, cl):
    depth = 0
    for k in range(i, len(text)):
        if text[k] == op:
            depth += 1
        elif text[k] == cl:
            depth -= 1
            if depth == 0:
                return k
    raise ValueError(f"unbalanced {op}{cl}")


def split_top(s):
    out, depth, cur = [], 0, ""
    for ch in s:
        depth += (ch == "(") - (ch == ")")
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


class Kernel:
    def __init__(self, text, consts, ops):
        self.ops, self.consts = ops, consts
        self.macros, self.funcs = {}, {}
        for m in re.finditer(r"^#define (BS_\w+)\(([^)]*)\)\s+(.*?)\s*(//.*)?$", text, re.M):
            self.macros[m[1]] = ([p.strip() for p in m[2].split(",")] if m[2].strip() else [], m[3])
        pat = r"(?:template <([^>]*)>\s*)?(?:\[\[gnu::always_inline\]\] inline|inline|__attribute__\(\(noinline\)\)) void (\w+)\(\) \{"
        for m in re.finditer(pat, text):
            end = match_close(text, m.end() - 1, "{", "}")
            tp = [p.split()[-1] for p in m[1].split(",")] if m[1] else []
            self.funcs[m[2]] = (tp, text[m.end():end])

    def expr(self, e, env):
        e = re.sub(r"p_setrwc::(\w+)", lambda m: str(P_SETRWC[m[1]]), e)
        while True:      # BS_V / BS_S function-like macros
            m = re.search(r"\b(BS_V|BS_S)\(", e)
            if not m:
                break
            end = match_close(e, m.end() - 1, "(", ")")
            params, body = self.macros[m[1]]
            vals = dict(zip(params, split_top(e[m.end():end])))
            body = re.sub(r"\b\w+\b", lambda t: f"({vals[t[0]]})" if t[0] in vals else t[0], body)
            e = e[:m.start()] + body + e[end + 1:]
        e = re.sub(r"\b(0x[0-9a-fA-F]+|\d+)u\b", r"\1", e)
        for _ in range(4):   # (a ? b : c) -> ((b) if (a) else (c))
            e = re.sub(r"\(([^()?:]+)\?([^()?:]+):([^()?:]+)\)", r"((\2) if (\1) else (\3))", e)
        return int(eval(e, {}, {**self.consts, **env}))

    def tti(self, name, vals):
        word = encode(self.ops, name, dict(zip(self.ops[name][0], vals)))
        return word

    def stmt(self, s, env, out):
        m = re.fullmatch(r"([\w:]+)\s*(<.*>)?\s*\((.*)\)", s, re.S)
        if not m:
            fail(f"cannot parse statement: {s!r}")
            return
        name, targs, args = m[1], m[2], split_top(m[3]) if m[3].strip() else []
        if name in self.macros and name not in ("BS_V", "BS_S"):
            params, body = self.macros[name]
            vals = {p: self.expr(a, env) for p, a in zip(params, args)}
            t = re.fullmatch(r"TTI_(\w+)(?:\((.*)\))?", body)
            tname = t[1]
            targ = [vals[a.strip()] if a.strip() in vals else int(a, 0)
                    for a in (t[2].split(",") if t[2] else [])]
            word = self.tti(tname, targ)
            mn = TT2DIS[tname]
            dis = tuple(vals[p] for p in params)
            if encode_dis(self.ops, mn, dis) != word:
                fail(f"{name}{tuple(dis)}: macro word {word:08x} != {mn} {dis} word")
            out.append((mn, dis))
        elif name.startswith("TTI_"):
            tname = name[4:]
            vals = [self.expr(a, env) for a in args]
            mn = TT2DIS[tname]
            out.append((mn, tuple(vals)))      # SETRWC/INCRWC: TT_OP order = disassembly order
        elif name == "lltt::replay":
            out.append(("ttreplay", (self.expr(args[0], env), self.expr(args[1], env), 0, 0)))
        elif name == "lltt::record":
            ex = {"<lltt::NoExec>": 0, "<lltt::Exec>": 1}[targs]
            out.append(("ttreplay", (self.expr(args[0], env), self.expr(args[1], env), ex, 1)))
        elif name in self.funcs:
            tv = [self.expr(a, env) for a in split_top(targs[1:-1])] if targs else []
            out.extend(self.call(name, tv))
        else:
            fail(f"unknown call {name}")

    def block(self, text, env, out):
        i = 0
        while i < len(text):
            m = re.compile(r"(?:\s+|//[^\n]*)*").match(text, i)
            i = m.end()
            if i >= len(text):
                break
            if text.startswith("if constexpr", i):
                p = text.index("(", i)
                pe = match_close(text, p, "(", ")")
                b = text.index("{", pe)
                be = match_close(text, b, "{", "}")
                cond = self.expr(text[p + 1:pe], env)
                if cond:
                    self.block(text[b + 1:be], env, out)
                i = be + 1
                m = re.compile(r"\s*else\s*").match(text, i)
                if m:
                    i = m.end()
                    if text[i] != "{":
                        fail("only if constexpr (c) {...} else {...} is supported")
                        return
                    b = i
                    be = match_close(text, b, "{", "}")
                    if not cond:
                        self.block(text[b + 1:be], env, out)
                    i = be + 1
                continue
            j = i
            depth = 0
            while text[j] != ";" or depth:
                depth += text[j] in "({"
                depth -= text[j] in ")}"
                j += 1
            self.stmt(text[i:j].strip(), env, out)
            i = j + 1

    def call(self, name, tv):
        tp, body = self.funcs[name]
        out = []
        self.block(body, dict(zip(tp, tv)), out)
        return out


def parse_listing(text):
    out = []
    for op, args in schedule.parse(text):
        if op == "ttsetrwc" and args == ["d0"]:
            out.append((op, (0, 0, 0, 0, 0, P_SETRWC["SET_D"])))
        else:
            out.append((op, tuple(num(a) for a in args)))
    return out


def replay(seq, buf):
    out, i = [], 0
    while i < len(seq):
        op, a = seq[i]
        if op == "ttreplay":
            s, n, ex, ld = a
            if ld:
                rec = seq[i + 1:i + 1 + n]
                if len(rec) != n or any(o == "ttreplay" for o, _ in rec):
                    fail(f"record {a}: {len(rec)} ops follow")
                buf[s:s + n] = rec
                out += rec if ex else []
                i += 1 + n
                continue
            if any(x is None for x in buf[s:s + n]):
                fail(f"replay {a} of empty slots")
            out += buf[s:s + n]
        else:
            out.append((op, a))
        i += 1
    return out


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/dev/tt-metal")
    ops = load_ops(os.path.join(root, OPS_H) if os.path.isdir(root) else root)
    seen = check_encoder(ops)
    print("encoder vs t189 words: " + ", ".join(f"{k} {n}" for k, (n, b) in sorted(seen.items())))
    src = open(KERNEL).read()
    consts = {}
    for m in re.finditer(r"constexpr uint32_t ((?:DR|S)_\w+ = [^;]+);", src):
        for part in m[1].split(","):
            k, v = part.split("=")
            consts[k.strip()] = eval(v)
    a = src.index("#ifndef BLEND_SCHED")
    b = src.index("using BlendBodyFn")
    for level in (0, 1, 2):
        env = {"BLEND_SCHED": level, "BLEND_CONST_HOIST": 1, "BLEND_PIXEL_FLOOR": 1,
               "BLEND_FPU_QF_ABL": 0}
        txt = preprocess(src[a:b], env)
        k = Kernel(txt, consts, ops)
        if level == 0:
            bad = "sched_" in txt or "BS_" in txt or "blend_pair_gaussian_math" not in txt
            print(f"level 0: BLEND_SCHED code compiled out, compiled bodies kept: {not bad}")
            if bad:
                fail("level 0 source differs")
            continue
        buf = [None] * 32
        if level == 2:
            issued = replay(k.call("blend_sched_record", []), buf)
            if issued or any(x is None for x in buf):
                fail(f"record: {len(issued)} ops issued, empty slots {[i for i, x in enumerate(buf) if x is None]}")
        n_ok, counts, rep = 0, set(), set()
        for j in range(16):
            for pm in (1, 2, 3):
                seq = k.call("blend_pair_body", [j, pm])
                ms = [m for m, bit in ((2 * j, 1), (2 * j + 1, 2)) if pm & bit]
                if level == 1:
                    want = parse_listing(schedule.pair(j) if pm == 3 else schedule.single(ms[0]))
                    got = seq
                    rep |= {o for o, _ in seq if o == "ttreplay"}
                else:
                    want = sum((parse_listing(schedule.a2(m)) for m in ms), [])
                    got = replay(seq, list(buf))
                    rep |= {a[3] for o, a in seq if o == "ttreplay"}
                counts.add((pm == 3, len(seq)))
                if got == want:
                    n_ok += 1
                else:
                    d = next(i for i, (x, y) in enumerate(zip(got + [None] * 200, want + [None] * 200)) if x != y)
                    fail(f"level {level} body <{j},{pm}> differs at op {d}: {got[d:d + 1]} vs {want[d:d + 1]}")
        cs = ", ".join(f"{'pair' if p else 'single'} {n}" for p, n in sorted(counts))
        if level == 1:
            print(f"level 1 (F): {n_ok}/48 bodies equal to schedule.py single(m)/pair(j); "
                  f"issued ops: {cs}; ttreplay in bodies: {bool(rep)}")
            if rep:
                fail("ttreplay in F bodies")
        else:
            print(f"level 2 (A2): record fills slots 0-31 ({sum(x is not None for x in buf)}) and issues nothing; "
                  f"{n_ok}/48 bodies equal to schedule.py a2(m) after replay; issued ops (instructions "
                  f"pushed by TRISC1): {cs}; record-mode replays in bodies: {1 in rep}")
            if 1 in rep:
                fail("record in A2 body")
    for e in ERR:
        print("FAIL:", e)
    print("check_cpp:", "FAIL" if ERR else "OK")
    return 1 if ERR else 0


if __name__ == "__main__":
    sys.exit(main())
