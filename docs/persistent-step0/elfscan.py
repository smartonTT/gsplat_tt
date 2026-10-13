#!/usr/bin/env python3
"""t579: per-RISC ELF sizes and local-memory layout of every kernel build in a tt-metal JIT cache.

Runs on the measurement box, read-only (piped over ssh: `python3 - <cache root> < elfscan.py`).
One JSON line per ELF: kernel, hash, risc, mtime, load (PT_LOAD file bytes = what the kernel config
ring holds), per-section sizes and addresses, and the local-memory symbols (__stack_base etc.)."""
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

TOOLS = "/localdev/smarton/tt-metal/runtime/sfpi/compiler/bin"


def parse(p: Path):
    b = p.read_bytes()
    if b[:4] != b"\x7fELF" or b[4] != 1:
        return None
    phoff, shoff = struct.unpack_from("<II", b, 0x1C)
    phentsize, phnum, shentsize, shnum, shstrndx = struct.unpack_from("<HHHHH", b, 0x2A)
    segs = []
    for i in range(phnum):
        ptype, off, vaddr, paddr, filesz, memsz = struct.unpack_from("<IIIIII", b, phoff + i * phentsize)
        if ptype == 1:
            segs.append({"vaddr": vaddr, "paddr": paddr, "filesz": filesz, "memsz": memsz})
    sh = [struct.unpack_from("<IIIIIIIIII", b, shoff + i * shentsize) for i in range(shnum)]
    stro = sh[shstrndx][4]
    secs = {}
    for name, typ, flags, addr, off, size, *_ in sh:
        end = b.index(b"\0", stro + name)
        n = b[stro + name:end].decode()
        if n and (flags & 2):  # SHF_ALLOC
            secs[n] = {"addr": addr, "size": size, "nobits": typ == 8}
    return segs, secs


def syms(p: Path):
    out = {}
    try:
        nm = subprocess.run([f"{TOOLS}/riscv-tt-elf-nm", str(p)], capture_output=True, text=True).stdout
    except OSError:
        return out
    for line in nm.splitlines():
        f = line.split()
        if len(f) >= 3 and f[-1].startswith("__") and ("stack" in f[-1] or "ldm" in f[-1] or "bss" in f[-1]
                                                       or "data" in f[-1] or "fw_export" in f[-1]):
            out[f[-1]] = int(f[0], 16)
    return out


def main():
    root = Path(sys.argv[1])
    for d in sorted(root.glob("**/kernels/*/*")):
        if not d.is_dir():
            continue
        for e in sorted(d.glob("*/*.elf")):
            r = parse(e)
            if r is None:
                continue
            segs, secs = r
            print(json.dumps({"kernel": d.parent.name, "hash": d.name, "risc": e.parent.name,
                              "file": e.name, "mtime": int(e.stat().st_mtime),
                              "load": sum(s["filesz"] for s in segs), "segs": segs, "secs": secs,
                              "syms": syms(e), "path": str(e)}))


if __name__ == "__main__":
    main()
