#!/usr/bin/env python3
"""t190: per-body instruction counts of the blend walk in TRISC1 objdumps.
  chain_bodies.py ../blend-dispatch-t189/out/trisc1-default.dis out/cw1/trisc1.dis out/cw2/trisc1.dis
(knob 0 is the t189 objdump: cc.sh's knob-0 ELF has the same addresses and encodings.)
For each body path (chain = ends in a sibling call, last = ends in ret) it
counts RISC instructions, issued SFPU ops (a ttreplay playback adds N ops, as
in t189's count_bodies.py) and taken jumps, and checks the walk: the table
`lw` comes before the first SFPU op, the chain path ends in `jr`, and no body
has a jalr. Every body path of a later file must issue the same SFPU / Tensix
ops (operands included, in order) as the same body of the first file (knob 0).
Prints one summary table per file and exits 1 on a failed check."""
import re
import sys
from collections import defaultdict

FUNC = re.compile(r'^([0-9a-f]+) <([^.].*)>:$')  # .L* labels stay in their function
INSN = re.compile(r'^\s+([0-9a-f]+):\s+[0-9a-f]+\s+(\S+)\s*([^#]*)')
BODY = re.compile(r'blend_(chain|pair)_body(?:ILm(\d+)ELm(\d+)E|<(\d+)ul, (\d+)ul>)')  # mangled or -C
BRANCH = {'beq', 'bne', 'blt', 'bge', 'bltu', 'bgeu', 'beqz', 'bnez', 'blez', 'bgez', 'bltz', 'bgtz'}


def parse(path):
    funcs, cur = {}, None
    for line in open(path):
        m = FUNC.match(line)
        if m:
            cur = m.group(2)
            funcs[cur] = []
            continue
        m = INSN.match(line)
        if m and cur is not None:
            funcs[cur].append((int(m.group(1), 16), m.group(2), m.group(3).strip()))
    return funcs


def issued(op, args):
    if op == 'ttreplay':
        a = [int(x) for x in args.split(',')]
        return a[1] if a[3] == 0 else 0  # a recording replay executes inline
    return 1 if op.startswith('sfp') else 0


def path_stats(insns):
    risc = sfpu = 0
    first_sfp = first_lw = None
    for i, (_, op, args) in enumerate(insns):
        n = issued(op, args)
        if op.startswith('sfp') or op == 'ttreplay':
            sfpu += n
            if first_sfp is None:
                first_sfp = i
        else:
            risc += 1
            if op == 'lw' and first_lw is None:
                first_lw = i
    return risc, sfpu, first_sfp, first_lw


def tensix(insns):
    return [(op, args) for _, op, args in insns if op.startswith('sfp') or op.startswith('tt')]


def analyse(path, ref=None):
    funcs = parse(path)
    rows, bad, ops_of = defaultdict(list), [], {}
    for name, insns in funcs.items():
        m = BODY.search(name)
        if not m:
            continue
        kind = m.group(1)
        j, pm = (int(x) for x in (m.group(2, 3) if m.group(2) else m.group(4, 5)))
        shape = 'pair' if pm == 3 else 'single'
        ops = [op for _, op, _ in insns]
        if 'jalr' in ops:
            bad.append(f'{name}: jalr in a body')
        addr = {a: i for i, (a, _, _) in enumerate(insns)}
        paths = [insns]
        if 'jr' in ops:  # chain body: chain path = start .. jr
            k = ops.index('jr')
            chain = insns[:k + 1]
            paths = [chain] + ([insns[k + 1:]] if 'ret' in ops else [])
            risc, sfpu, fs, fl = path_stats(chain)
            if fl is None or fs is None or fl > fs:
                bad.append(f'{name}: table lw not before the first SFPU op')
            branches = [i for i, (_, op, _) in enumerate(chain) if op in BRANCH]
            gap = k - fl - 1 if fl is not None else -1  # instructions that hide the load
            rows[('chain', shape, 'J<15')].append((risc, sfpu, 1, gap))
            if branches:  # knob 1: the branch to the last path (not taken on the chain path)
                b = chain[branches[0]]
                tgt = int(b[2].split(',')[-1].split()[0], 16)
                last = insns[addr[tgt]:]
                last = last[:[op for _, op, _ in last].index('ret') + 1]
                pre = chain[:branches[0] + 1]
                r1, s1, _, _ = path_stats(pre + last)
                rows[('last', shape, 'J<15')].append((r1, s1, 2, -1))  # taken branch + ret
            else:  # knob 2: the last body runs the chain path, then the end stub's ret
                rows[('last', shape, 'J<15')].append((risc + 1, sfpu, 2, -1))
        else:  # ends in ret: the jump-walk body (knob 0) or pair 15 of the chain
            k = ops.index('ret')
            risc, sfpu, _, _ = path_stats(insns[:k + 1])
            tag = 'J=15' if kind == 'chain' else 'any J'
            rows[('ret', shape, tag)].append((risc, sfpu, 1, -1))
        ops_of[(j, pm)] = tensix(paths[0])
        if ref is not None and any(tensix(q) != ref.get((j, pm)) for q in paths):
            bad.append(f'{name}: SFPU ops differ from the first file')
    return funcs, rows, bad, ops_of


def main():
    rc, ref = 0, None
    for path in sys.argv[1:]:
        funcs, rows, bad, ops_of = analyse(path, ref)
        if ref is None:
            ref = ops_of
        n = sum(len(v) for k, v in rows.items() if k[0] != 'last')
        print(f'== {path}: {n} body paths')
        print(f'{"path":6} {"shape":6} {"pairs":6} {"bodies":>6} {"RISC":>9} {"SFPU issued":>12} '
              f'{"taken jumps":>11} {"insns between lw and jr":>24}')
        for key in sorted(rows):
            v = rows[key]
            rs = sorted({r for r, _, _, _ in v})
            ss = sorted({s for _, s, _, _ in v})
            tj = sorted({t for _, _, t, _ in v})
            gap = sorted({g for _, _, _, g in v if g >= 0})
            gap = [min(gap), max(gap)] if len(gap) > 2 else gap
            fmt = lambda xs: '/'.join(map(str, xs)) if xs else '-'
            print(f'{key[0]:6} {key[1]:6} {key[2]:6} {len(v):6} {fmt(rs):>9} {fmt(ss):>12} '
                  f'{fmt(tj):>11} {fmt(gap) if gap else "-":>24}')
        for b in bad:
            print('CHECK FAILED:', b)
            rc = 1
        if ref is not ops_of:
            print('SFPU ops of every body path same as the first file:', not bad)
        end = [k for k in funcs if 'blend_chain_end' in k]
        if end:
            print('end stub:', ' ; '.join(f'{op} {a}'.strip() for _, op, a in funcs[end[0]]))
    sys.exit(rc)


if __name__ == '__main__':
    main()
