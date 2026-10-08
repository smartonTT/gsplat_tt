#!/usr/bin/env python3
"""Task #409: check a 30-view bicycle md5 list against the golden for the grid it ran on.

The output bits depend on the compute grid every stage partitions over, not on the dispatch
mode (docs/eth-dispatch-t397/README.md): eth at a capped 11x10 matches the worker golden.
So the golden is picked from the grid in the run log's "[DEV] dispatch ..." line (a
"stage grid capped to XxY" suffix wins). A grid without a golden fails; nothing is skipped.

  opt/md5_golden.py <md5 list> <run log>
    md5 list: `md5sum *` of the dumped views, sorted by name (what the bench scripts write);
    its own md5 (first 8 hex) is the list md5 compared here.
Prints one line (MD5_GOLDEN_OK / MD5_GOLDEN_FAIL) and exits 0 on a match, 1 on a mismatch,
2 if the grid is missing from the log or has no golden.
"""
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

# list md5 (8 hex) per stage grid; the full lists are in the files named alongside.
GOLDENS = {
    "11x10": "906e0435",  # worker dispatch, p100a and p150 (docs/matblend-ready-t273/t289/md5-golden-906e0435.txt)
    "12x10": "39d84b28",  # eth dispatch, p150 120 cores (docs/eth-dispatch-t397/out/md5-r1-E.txt)
}

_DEV = re.compile(r"^\[DEV\] .*?compute grid (\d+)x(\d+)(.*)$", re.M)
_CAP = re.compile(r"stage grid capped to (\d+)x(\d+)")


def grid_from_log(text: str) -> str | None:
    """Stage grid "XxY" from the last [DEV] open line, or None."""
    m = None
    for m in _DEV.finditer(text):
        pass
    if m is None:
        return None
    cap = _CAP.search(m.group(3))
    return f"{cap.group(1)}x{cap.group(2)}" if cap else f"{m.group(1)}x{m.group(2)}"


def list_md5(md5_list: bytes) -> str:
    return hashlib.md5(md5_list).hexdigest()[:8]


def check(md5_list: bytes, log_text: str) -> tuple[int, str]:
    grid = grid_from_log(log_text)
    got = list_md5(md5_list)
    n = sum(1 for ln in md5_list.decode(errors="replace").splitlines() if ln.strip())
    if grid is None:
        return 2, f"MD5_GOLDEN_FAIL no '[DEV] ... compute grid' line in the log (list {got}, {n} views)"
    want = GOLDENS.get(grid)
    if want is None:
        return 2, f"MD5_GOLDEN_FAIL grid {grid} has no golden ({', '.join(f'{g}: {h}' for g, h in GOLDENS.items())}); list {got}"
    if got != want:
        return 1, f"MD5_GOLDEN_FAIL grid {grid}: list {got} != golden {want} ({n} views)"
    return 0, f"MD5_GOLDEN_OK grid {grid}: list {got} = golden ({n}/{n} views)"


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: opt/md5_golden.py <md5 list> <run log>", file=sys.stderr)
        return 2
    rc, line = check(Path(argv[1]).read_bytes(), Path(argv[2]).read_text(errors="replace"))
    print(line)
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv))
