"""Tests for opt/brisc_stack_check.py (tasks #573/#574): frame sums over the direct
call graph, the margin verdict against --reserve and the --scan mode run.py uses."""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("bsc", HERE / "brisc_stack_check.py")
bsc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bsc)

OBJDUMP = """
ffb00000 <_start>:
ffb00004:  addi sp,sp,-16
ffb00008:  jal  ra,ffb00100 <kernel_main>
ffb00100 <kernel_main>:
ffb00104:  addi sp,sp,-4096
ffb00108:  jal  ra,ffb00200 <k2_range>
ffb0010c:  jal  ra,ffb00300 <__udivdi3>
ffb00110 <.L5>:
ffb00114:  j    ffb00110 <.L5>
ffb00200 <k2_range>:
ffb00204:  addi sp,sp,-32
ffb00300 <__udivdi3>:
ffb00304:  addi sp,sp,-48
ffb00308:  jal  ra,ffb00300 <__udivdi3>
"""


def nm(base):
    return f"{base:08x} B __stack_base\nffb00000 T _start\n"


def test_depth_follows_deepest_chain():
    g = bsc.parse_objdump(OBJDUMP)
    assert g["kernel_main"] == (4096, {"k2_range", "__udivdi3"})
    depth, chain = bsc.max_depth(g)
    assert depth == 16 + 4096 + 48  # self-recursion counted once
    assert chain == ["_start", "kernel_main", "__udivdi3"]


def test_margin_against_reserve():
    top = 0xFFB02000
    depth = 16 + 4096 + 48
    r = bsc.check(OBJDUMP, nm(top - depth - 112), top, 128)  # #573: +112 B before the fix
    assert (r["margin"], r["ok"]) == (112, False)
    r = bsc.check(OBJDUMP, nm(top - depth - 128), top, 128)
    assert (r["margin"], r["ok"]) == (128, True)


def _fake_tools(tmp_path, base):
    tools = tmp_path / "tools"
    tools.mkdir()
    (tmp_path / "od.txt").write_text(OBJDUMP)
    for name, body in (("objdump", f"cat {tmp_path}/od.txt"),
                       ("nm", f"printf '{base:08x} B __stack_base\\n'")):
        p = tools / f"riscv-tt-elf-{name}"
        p.write_text(f"#!/bin/sh\n{body}\n")
        p.chmod(0o755)
    return tools


def _scan(tmp_path, tools, since):
    return subprocess.run([sys.executable, str(HERE / "brisc_stack_check.py"), "--scan",
                           str(tmp_path / "cache"), "--since", str(since), "--tools", str(tools)],
                          capture_output=True, text=True)


def test_scan_checks_only_new_elfs(tmp_path):
    old = tmp_path / "cache/k/old_kern/h1/brisc/brisc.elf"
    new = tmp_path / "cache/k/new_kern/h2/brisc/brisc.elf"
    for p in (old, new):
        p.parent.mkdir(parents=True)
        p.write_bytes(b"")
    os.utime(old, (1000, 1000))
    os.utime(new, (5000, 5000))
    assert bsc.find_elfs(str(tmp_path / "cache"), 2000) == [str(new)]
    depth = 16 + 4096 + 48
    good = _scan(tmp_path, _fake_tools(tmp_path, 0xFFB02000 - depth - 400), 2000)
    assert good.returncode == 0 and good.stdout.count("\n") == 1 and "new_kern" in good.stdout


def test_scan_fails_short_margin(tmp_path):
    p = tmp_path / "cache/k/kern/h/brisc/brisc.elf"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"")
    bad = _scan(tmp_path, _fake_tools(tmp_path, 0xFFB02000 - (16 + 4096 + 48) - 112), 0)
    assert bad.returncode == 1 and '"margin": 112' in bad.stdout
