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
(default: --repo resolved). Exit 1 and print the pairs on a collision; exit 2 if
render/kernels is missing or has no zones (wrong --repo).
"""
import argparse
import os
import re
import sys

KERNEL_DIR = "render/kernels"
DEVICE_CALL = r'Device(?:ZoneScoped\w*|TimestampedData\w*|Recordevent\w*|Validate\w*)'
ZONE_RE = re.compile(r'\b' + DEVICE_CALL + r'\s*\(\s*"([^"]*)"')
# '#define W(name) DeviceZoneScopedN(name)': call sites of W use their own
# __FILE__/__LINE__ and literal name, so they hash like direct calls.
WRAPPER_RE = re.compile(r'^\s*#\s*define\s+(\w+)\s*\(\s*(\w+)[^)]*\)\s*' + DEVICE_CALL + r'\s*\(\s*\2\b')


def hash16(s: str) -> int:
    h = 2166136261
    for b in s.encode():
        h = ((h ^ b) * 16777619) & 0xFFFFFFFF
    return ((h & 0xFFFF) ^ (h >> 16)) & 0xFFFF


def zone_string(name, path, line):
    return f"{name},{path},{line},KERNEL_PROFILER"


def _sources(repo):
    base = os.path.join(repo, KERNEL_DIR)
    for d, _, files in os.walk(base):
        for f in sorted(files):
            if f.endswith((".cpp", ".h", ".hpp", ".cc")):
                p = os.path.join(d, f)
                with open(p, errors="replace") as fh:
                    yield os.path.relpath(p, repo), fh.read().splitlines()


def scan(repo):
    """Yield (name, repo-relative path, line) for every named device zone,
    direct or through a one-argument wrapper macro."""
    srcs = list(_sources(repo))
    wrappers = sorted({m.group(1) for _, lines in srcs for t in lines
                       for m in [WRAPPER_RE.match(t)] if m})
    alts = [DEVICE_CALL] + [re.escape(w) for w in wrappers]
    call_re = re.compile(r'\b(?:' + "|".join(alts) + r')\s*\(\s*"([^"]*)"')
    for rel, lines in srcs:
        for ln, text in enumerate(lines, 1):
            st = text.lstrip()
            if st.startswith("//") or WRAPPER_RE.match(text):
                continue
            for m in call_re.finditer(text):
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
    if not os.path.isdir(os.path.join(repo, KERNEL_DIR)):
        print(f"[zone_hash_check] ERROR: no {KERNEL_DIR} under {repo}", file=sys.stderr)
        return 2
    zones = list(scan(repo))
    if not zones:
        print(f"[zone_hash_check] ERROR: 0 zones found under {repo}/{KERNEL_DIR}", file=sys.stderr)
        return 2
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
