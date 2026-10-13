#!/usr/bin/env python3
"""Static stack margin of a RISC-V kernel ELF (task #573).

A Blackhole BRISC kernel's .data, .bss and stack share 8 KB of local memory
(stack from the top down to __stack_base). Nothing checks that the deepest call
chain fits: in t571/t573 the fused mat+blend BRISC with sort-in-blend had 184 B
more .data/.bss, K2's 4.5 KB frame then ran into g_b4a_fn and the kernel jumped
to wild PCs. This sums `addi sp,sp,-N` per function over the direct call graph
from _start and compares the deepest chain with top - __stack_base.

  brisc_stack_check.py <elf> [--tools <dir with riscv-tt-elf-objdump/nm>]
                             [--top 0xffb02000] [--reserve 128]
  brisc_stack_check.py --scan <JIT cache dir> [--since <epoch s>] [...]

--scan checks every brisc.elf under the dir modified at or after --since (task
#574: render/run.py runs it on the kernels each run compiles and fails the run
when one is short).

Indirect calls (jalr) are not followed: --reserve covers them and the firmware
frame above the kernel. Prints one JSON line; exit 1 when margin < reserve.
"""
import argparse
import json
import os
import re
import subprocess
import sys

FN_RE = re.compile(r"^([0-9a-f]+) <([^>]+)>:$")
FRAME_RE = re.compile(r"\baddi\s+sp,\s*sp,\s*-(\d+)")
CALL_RE = re.compile(r"\s(?:jal|j|tail|call)\s+(?:ra,\s*)?[0-9a-f]+ <([^>+]+)(?:\+0x[0-9a-f]+)?>")


def parse_objdump(text):
    """{function: (frame bytes, set of direct callees)}; local .L labels stay in their function."""
    frames, calls, fn = {}, {}, None
    for line in text.splitlines():
        m = FN_RE.match(line.strip())
        if m:
            if not m.group(2).startswith("."):
                fn = m.group(2)
                frames.setdefault(fn, 0)
                calls.setdefault(fn, set())
            continue
        if fn is None:
            continue
        f = FRAME_RE.search(line)
        if f:
            frames[fn] += int(f.group(1))
        c = CALL_RE.search(line)
        if c and c.group(1) != fn and not c.group(1).startswith("."):
            calls[fn].add(c.group(1))
    return {k: (frames[k], calls[k]) for k in frames}


def max_depth(graph, root="_start"):
    """Deepest frame sum from root and its chain; recursion counted once."""
    memo = {}

    def go(f, stack):
        if f in memo:
            return memo[f]
        if f in stack or f not in graph:
            return 0, [f]
        frame, callees = graph[f]
        best, chain = 0, []
        for c in sorted(callees):
            d, ch = go(c, stack | {f})
            if d > best:
                best, chain = d, ch
        memo[f] = (frame + best, [f] + chain)
        return memo[f]

    return go(root, frozenset())


def stack_base(nm_text):
    for line in nm_text.splitlines():
        p = line.split()
        if len(p) >= 3 and p[-1] == "__stack_base":
            return int(p[0], 16)
    raise ValueError("no __stack_base symbol")


def check(objdump_text, nm_text, top, reserve):
    depth, chain = max_depth(parse_objdump(objdump_text))
    avail = top - stack_base(nm_text)
    margin = avail - depth
    return {"depth": depth, "avail": avail, "margin": margin, "reserve": reserve,
            "ok": margin >= reserve, "chain": chain}


DEFAULT_TOOLS = "/localdev/smarton/tt-metal/runtime/sfpi/compiler/bin"


def check_elf(elf, tools=DEFAULT_TOOLS, top=0xFFB02000, reserve=128):
    od = subprocess.run([f"{tools}/riscv-tt-elf-objdump", "-d", "--no-show-raw-insn", elf],
                        check=True, capture_output=True, text=True).stdout
    nm = subprocess.run([f"{tools}/riscv-tt-elf-nm", elf], check=True, capture_output=True,
                        text=True).stdout
    r = check(od, nm, top, reserve)
    r["elf"] = elf
    return r


def find_elfs(root, since=0.0, name="brisc.elf"):
    """Paths of `name` files under root with mtime >= since, sorted."""
    out = []
    for d, _, files in os.walk(root):
        if name in files:
            p = os.path.join(d, name)
            try:
                if os.stat(p).st_mtime >= since:
                    out.append(p)
            except OSError:
                pass
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("elf", nargs="?")
    ap.add_argument("--scan", help="check every brisc.elf under this dir instead of one ELF")
    ap.add_argument("--since", type=float, default=0.0, help="with --scan: mtime >= this epoch")
    ap.add_argument("--tools", default=DEFAULT_TOOLS)
    ap.add_argument("--top", type=lambda s: int(s, 0), default=0xFFB02000)
    ap.add_argument("--reserve", type=int, default=128)
    a = ap.parse_args()
    if (a.elf is None) == (a.scan is None):
        ap.error("give one ELF or --scan <dir>")
    elfs = find_elfs(a.scan, a.since) if a.scan else [a.elf]
    rc = 0
    for e in elfs:
        r = check_elf(e, a.tools, a.top, a.reserve)
        print(json.dumps(r))
        rc |= 0 if r["ok"] else 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
