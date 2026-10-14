#!/usr/bin/env python3
"""A/B gate JIT-hash check (#656).

After an A/B run, list each arm's JIT cache (kernels/<name>/<hash>) and fail
when the kernel(s) the gate edits have the same hash set in both arms: that run
was an A/A comparison and its delta is noise (#647/#654).

Each arm is a JIT cache directory, or a text file with one path per line
(e.g. `find <cache> -path '*/kernels/*' -maxdepth 6 -type d` saved on the box).

  python3 opt/ab_jit_check.py --arm base=<cache|listing> --arm cand=<cache|listing> \
      --kernel <name> [--kernel <name> ...]

Also compares the md5 of each kernel's trisc/brisc/ncrisc ELFs per arm (directory
arms only). The tt-metal hash ignores source text, so an in-place source swap with
a per-arm cache can keep the hash but change the ELFs: that passes ("OK (ELF)").

Exit 0: every --kernel differs between the arms. 1: some --kernel is identical
or missing (the gate must fail or be flagged A/A). 2: usage error.
"""
import argparse
import hashlib
import os
import re
import sys

_KERNEL_RE = re.compile(r"(?:^|/)kernels/([^/]+)/([^/\s]+)")


def _paths(src):
    if os.path.isdir(src):
        for root, dirs, _ in os.walk(src):
            for d in dirs:
                yield os.path.join(root, d)
    else:
        with open(src) as f:
            for line in f:
                yield line.strip()


def kernel_hashes(src):
    """Map kernel name -> set of hashes found under src."""
    out = {}
    for p in _paths(src):
        m = _KERNEL_RE.search(p)
        if m:
            out.setdefault(m.group(1), set()).add(m.group(2))
    return out


def elf_md5s(src):
    """Map kernel name -> {<risc dir>/<file>: set of md5} for *.elf under a cache dir."""
    out = {}
    if not os.path.isdir(src):
        return out
    for root, _, files in os.walk(src):
        m = _KERNEL_RE.search(root + "/")
        if not m:
            continue
        for f in files:
            if f.endswith(".elf"):
                with open(os.path.join(root, f), "rb") as fh:
                    d = hashlib.md5(fh.read()).hexdigest()
                key = os.path.join(os.path.basename(root), f)
                out.setdefault(m.group(1), {}).setdefault(key, set()).add(d)
    return out


def check(arms, kernels, elfs=None):
    """Return (ok, report lines). arms: list of (label, {name: hashes}).
    elfs: optional [elf_md5s(arm a), elf_md5s(arm b)]."""
    lines, ok = [], True
    (la, a), (lb, b) = arms
    ea, eb = elfs or ({}, {})
    for k in kernels:
        ha, hb = a.get(k, set()), b.get(k, set())
        if not ha or not hb:
            ok = False
            lines.append(f"MISSING {k}: {la}={sorted(ha)} {lb}={sorted(hb)}")
        elif ha == hb and ea.get(k) and eb.get(k) and ea[k] != eb[k]:
            diff = sorted(n for n in set(ea[k]) | set(eb[k]) if ea[k].get(n) != eb[k].get(n))
            lines.append(f"OK (ELF) {k}: same hashes {sorted(ha)} but ELFs differ: {diff}")
        elif ha == hb:
            ok = False
            lines.append(f"A/A {k}: same hashes {sorted(ha)} in {la} and {lb}")
        else:
            lines.append(f"OK {k}: {la}={sorted(ha)} {lb}={sorted(hb)}")
    same = sorted(n for n in set(a) & set(b) if a[n] == b[n])
    if set(a) == set(b) and len(same) == len(a) and ea == eb:
        ok = False
        lines.append(f"A/A all {len(a)} kernels identical between {la} and {lb}")
    return ok, lines


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", action="append", required=True, help="label=path")
    ap.add_argument("--kernel", action="append", required=True)
    args = ap.parse_args(argv)
    if len(args.arm) != 2 or any("=" not in x for x in args.arm):
        ap.error("need exactly two --arm label=path")
    arms, elfs = [], []
    for spec in args.arm:
        label, path = spec.split("=", 1)
        if not os.path.exists(path):
            ap.error(f"{path} not found")
        arms.append((label, kernel_hashes(path)))
        elfs.append(elf_md5s(path))
    ok, lines = check(arms, args.kernel, elfs)
    print("\n".join(lines))
    print("JIT-HASH OK" if ok else "JIT-HASH FAIL: gate is A/A for the edited kernel(s)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
