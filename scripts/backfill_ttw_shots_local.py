#!/usr/bin/env python3
"""Populate opt/metal-screenshots/ttw-NNN/ hero + hero_diff10 for report cards.

Uses the latest loop verify capture (opt/metal-screenshots/loop-ttw/hero.png) when
per-iter device shots are missing. The 10× diff is generated vs benchmarks/reference_v2
when no ref.png is present (same as build_report.ensure_hero_diff10 fallback).

This is a HEAD snapshot backfill — historical PSNR in iters.jsonl is unchanged.
Re-run scripts/backfill_ttw_shots.sh on device for per-iter renders at commit.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OPT = ROOT / "opt"
JSONL = OPT / "ttw" / "iters.jsonl"
SRC = OPT / "metal-screenshots" / "loop-ttw" / "hero.png"
SHOTS = OPT / "metal-screenshots"


def iter_dirs() -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for line in JSONL.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        n = r.get("iter")
        d = str(r.get("iter_dir") or "").strip() or (
            f"ttw-{int(n):03d}" if n is not None else ""
        )
        if d and d not in seen:
            seen.add(d)
            out.append(d)
    return out


def main() -> int:
    if not SRC.exists():
        print(f"missing source hero: {SRC}", file=sys.stderr)
        return 1
    sys.path.insert(0, str(OPT))
    import build_report as br  # noqa: E402

    n_copied = 0
    for d in iter_dirs():
        dest = SHOTS / d
        hero = dest / "hero.png"
        diff = dest / "hero_diff10.png"
        if hero.exists() and diff.exists():
            continue
        dest.mkdir(parents=True, exist_ok=True)
        if not hero.exists():
            shutil.copy2(SRC, hero)
            n_copied += 1
        br.ensure_hero_diff10(d)
    br.main()
    print(f"backfilled heroes for {n_copied} dirs; ran build_report")
    return 0


if __name__ == "__main__":
    sys.exit(main())
