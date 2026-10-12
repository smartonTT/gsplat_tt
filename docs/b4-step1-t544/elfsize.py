#!/usr/bin/env python3
"""t544: per-RISC kernel ELF sizes in a tt-metal JIT cache.

  elfsize.py <cache root> <kernel name>...

For each kernel name, every build hash under <cache>/**/kernels/<name>/ is listed
(newest first) with, per RISC ELF: the PT_LOAD file bytes (what the kernel config
ring holds: text + data, before 16 B alignment) and the .text section size."""
import struct
import sys
from pathlib import Path


def elf_sizes(p: Path):
    b = p.read_bytes()
    if b[:4] != b"\x7fELF" or b[4] != 1:
        return None
    phoff, shoff = struct.unpack_from("<II", b, 0x1C)
    phentsize, phnum, shentsize, shnum, shstrndx = struct.unpack_from("<HHHHH", b, 0x2A)
    load = 0
    for i in range(phnum):
        ptype, _, _, _, filesz, memsz = struct.unpack_from("<IIIIII", b, phoff + i * phentsize)
        if ptype == 1:
            load += filesz
    sh = [struct.unpack_from("<IIIIII", b, shoff + i * shentsize) for i in range(shnum)]
    stro = sh[shstrndx][4]
    text = 0
    for name, _, _, _, _, size in sh:
        end = b.index(b"\0", stro + name)
        if b[stro + name:end] == b".text":
            text += size
    return load, text


def main():
    root = Path(sys.argv[1])
    for name in sys.argv[2:]:
        dirs = sorted((d for d in root.glob(f"**/kernels/{name}/*") if d.is_dir()),
                      key=lambda d: d.stat().st_mtime, reverse=True)
        if not dirs:
            print(f"ELF {name}: not in cache")
        for d in dirs[:4]:
            parts = []
            tot = 0
            for e in sorted(d.glob("*/*.elf")):
                s = elf_sizes(e)
                if s:
                    parts.append(f"{e.parent.name}={s[0]}/{s[1]}")
                    tot += (s[0] + 15) // 16 * 16
            print(f"ELF {name} {d.name} load_total={tot} " + " ".join(parts))


if __name__ == "__main__":
    main()
