#!/usr/bin/env python3
"""t219: check the built TRISC1 ELFs (objdump -d -C) of BLEND_SCHED 0, 1 and 2.

- level 0: every blend_pair_body equals the t189 default build (same flags).
- level 1 (F): every body is exactly its schedule.py listing (single 60 ops,
  pair 109), straight line, no ttreplay.
- level 2 (A2): the record site holds front (slots 0-12) and tail + SETRWC
  (13-31); every body, with its replays expanded from those slots, is exactly
  schedule.py a2(m); no body records.
- every ttreplay outside the bodies, per level (where the replay buffer is used).
Run from this directory: python3 check_elf.py sched0.dis sched1.dis sched2.dis
"""
import collections
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import check_cpp  # noqa: E402  (chdirs to the t205 directory, imports schedule)

schedule = check_cpp.schedule
T189 = os.path.join(HERE, "..", "blend-dispatch-t189", "out", "trisc1-default.dis")
BODY = re.compile(r"blend_pair_body<(\d+)ul, (\d+)ul>\(\)$")


def functions(path):
    """name -> [(addr, op, args)] in address order."""
    out, cur = collections.OrderedDict(), None
    for line in open(path):
        m = re.match(r"^([0-9a-f]+) <(.+)>:$", line)
        if m:
            name = m[2]
            if not re.match(r"\.L|.*\.L\w*$|L[BM]\w*$", name):
                cur = out.setdefault(name, [])
            continue
        m = re.match(r"^\s+([0-9a-f]+):\s+[0-9a-f]+\s+(\S+)\s*(\S*)", line)
        if m and cur is not None:
            cur.append((int(m[1], 16), m[2], m[3]))
    return out


def body_ops(fn):
    ops = []
    for _, op, args in fn:
        if op == "ret":
            return ops
        ops.append((op, args))
    return None


def norm(ops):
    return [(op, tuple(check_cpp.num(a) for a in args.split(",")) if args else ()) for op, args in ops]


def bodies(fs):
    return {(int(m[1]), int(m[2])): body_ops(f) for n, f in fs.items() if (m := BODY.search(n))}


def replays(fs):
    """[(function, addr, (start, len, exec, load))] outside the bodies."""
    return [(n, a, tuple(int(x) for x in args.split(","))) for n, f in fs.items() if not BODY.search(n)
            for a, op, args in f if op == "ttreplay"]


def main():
    paths = sys.argv[1:4]
    if len(paths) != 3:
        print(__doc__)
        return 2
    fails = []
    ref = bodies(functions(T189))
    for level, path in enumerate(paths):
        fs = functions(path)
        bs = bodies(fs)
        print(f"== level {level}: {os.path.basename(path)}, {len(bs)} bodies")
        if sorted(bs) != sorted(ref):
            fails.append(f"level {level}: body set {sorted(bs)} differs from t189")
            continue
        if any(v is None for v in bs.values()):
            fails.append(f"level {level}: a body has no ret")
            continue
        if level == 0:
            same = sum(bs[k] == ref[k] for k in ref)
            print(f"bodies equal to t189 default build: {same}/{len(ref)}")
            if same != len(ref):
                fails.append(f"level 0: {len(ref) - same} bodies differ from t189")
        buf = [None] * 32
        if level == 2:
            sites = []
            for n, f in fs.items():
                if BODY.search(n):
                    continue
                for i, (a, op, args) in enumerate(f):
                    if op == "ttreplay" and args == "0,13,0,1":
                        rec = [(o, x) for _, o, x in f[i:i + 34]]
                        sites.append((n, a, rec))
            print(f"record sites (ttreplay 0,13,0,1): {[(n, hex(a)) for n, a, _ in sites]}")
            if len(sites) != 1:
                fails.append(f"level 2: {len(sites)} record sites")
            else:
                rec = norm(sites[0][2])
                check_cpp.replay(rec, buf)
                front = check_cpp.parse_listing(schedule.FRONT)
                tail = check_cpp.parse_listing(schedule.TAIL.format(t=192, r=0, g=64, b=128) + "\nttsetrwc d0")
                ok = rec[14] == ("ttreplay", (13, 19, 0, 1)) and buf[:13] == front and buf[13:] == tail
                print(f"record: slots 0-12 = FRONT, 13-31 = TAIL(T,R,G,B of microblock 0) + SETRWC D=0: {ok}")
                if not ok:
                    fails.append("level 2: record contents differ")
        n_ok, counts, rep = 0, collections.Counter(), collections.Counter()
        for (j, pm), ops in sorted(bs.items()):
            got = norm(ops)
            ms = [m for m, bit in ((2 * j, 1), (2 * j + 1, 2)) if pm & bit]
            rep.update(a[3] for o, a in got if o == "ttreplay")
            counts[(pm == 3, len(got))] += 1
            if level == 0:
                continue
            if level == 1:
                want = check_cpp.parse_listing(schedule.pair(j) if pm == 3 else schedule.single(ms[0]))
            else:
                want = sum((check_cpp.parse_listing(schedule.a2(m)) for m in ms), [])
                got = check_cpp.replay(got, list(buf))
            if got == want:
                n_ok += 1
            else:
                d = next(i for i, (x, y) in enumerate(zip(got + [None] * 200, want + [None] * 200)) if x != y)
                fails.append(f"level {level} body <{j},{pm}> op {d}: {got[d:d + 1]} vs {want[d:d + 1]}")
        cs = ", ".join(f"{'pair' if p else 'single'} {n} ops x{c}" for (p, n), c in sorted(counts.items()))
        print(f"body sizes (instructions before ret): {cs}")
        print(f"ttreplay in bodies: {sum(rep.values())} (record mode {rep[1]})")
        if level:
            print(f"bodies equal to schedule.py {'single/pair' if level == 1 else 'a2 after replay'}: {n_ok}/48")
            if n_ok != 48:
                fails.append(f"level {level}: {48 - n_ok} bodies differ")
        if level == 1 and sum(rep.values()):
            fails.append("level 1: ttreplay in F bodies")
        if level == 2 and rep[1]:
            fails.append("level 2: a body records")
        for n, a, args in replays(fs):
            print(f"ttreplay outside bodies: {n} @{a:#x} {','.join(map(str, args))}")
    for f in fails:
        print("FAIL:", f)
    print("check_elf:", "FAIL" if fails else "OK")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
