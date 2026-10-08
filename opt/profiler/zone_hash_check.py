#!/usr/bin/env python3
"""Pre-check device profiler zone names for 16-bit hash collisions.

tt-metal (tt_metal/tools/profiler/kernel_profiler.hpp) hashes each zone as
Hash16_CT(name "," __FILE__ "," __LINE__ ",KERNEL_PROFILER"): FNV-1a 32-bit,
then low16 ^ high16. __FILE__ is the absolute kernel path, so collisions
depend on the checkout path (#416: sort_ol_barrier vs fz_mv_dwb under
/localdev/smarton/gstt2-t416). Two different zone strings with the same hash
make ReadMeshDeviceProfilerResults throw, so check before a capture.

  zone_hash_check.py [--repo DIR] [--path-root ABS]
--path-root is the absolute repo path the kernels are compiled from
(default: --repo resolved). Exit 1 and print the pairs on a collision.
"""
import argparse
import os
import re
import sys

KERNEL_DIR = "render/kernels"
ZONE_RE = re.compile(r'\bDevice(?:ZoneScoped\w*|TimestampedData\w*|Recordevent\w*|Validate\w*)\s*\(\s*"([^"]*)"')


def hash16(s: str) -> int:
    h = 2166136261
    for b in s.encode():
        h = ((h ^ b) * 16777619) & 0xFFFFFFFF
    return ((h & 0xFFFF) ^ (h >> 16)) & 0xFFFF


def zone_string(name, path, line):
    return f"{name},{path},{line},KERNEL_PROFILER"


def scan(repo):
    """Yield (name, repo-relative path, line) for every named device zone."""
    base = os.path.join(repo, KERNEL_DIR)
    for d, _, files in os.walk(base):
        for f in sorted(files):
            if not f.endswith((".cpp", ".h", ".hpp", ".cc")):
                continue
            p = os.path.join(d, f)
            rel = os.path.relpath(p, repo)
            with open(p, errors="replace") as fh:
                for ln, text in enumerate(fh, 1):
                    if text.lstrip().startswith("//"):
                        continue
                    for m in ZONE_RE.finditer(text):
                        yield m.group(1), rel, ln


def collisions(zones, path_root):
    by_hash = {}
    for name, rel, ln in zones:
        s = zone_string(name, f"{path_root.rstrip('/')}/{rel}", ln)
        by_hash.setdefault(hash16(s), {})[s] = (name, rel, ln)
    return {h: list(v.values()) for h, v in by_hash.items() if len(v) > 1}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo", default=".")
    ap.add_argument("--path-root", default=None)
    a = ap.parse_args(argv)
    repo = os.path.abspath(a.repo)
    root = a.path_root or os.path.realpath(repo)
    zones = list(scan(repo))
    bad = collisions(zones, root)
    if not bad:
        print(f"[zone_hash_check] OK: {len(zones)} zones, no 16-bit hash collisions at {root}")
        return 0
    for h, items in sorted(bad.items()):
        print(f"[zone_hash_check] COLLISION hash=0x{h:04x} at {root}:", file=sys.stderr)
        for name, rel, ln in items:
            print(f"    {name}  {rel}:{ln}", file=sys.stderr)
    print("[zone_hash_check] rename one zone of each pair (or move it a line), "
          "or capture from a checkout at another path", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
