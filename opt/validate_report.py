#!/usr/bin/env python3
"""End-of-iteration REPORT validator.

Runs AFTER opt/build_report.py regenerates opt/REPORT.html (wired into the
tt-workflows Definition-of-Done gate, scripts/done.sh; ttw.toml require_report=1).
Exits non-zero and prints "REPORT INVALID: <reason>" if the report is stale,
missing descriptions, or carries unreasonable fields for recent iterations — so a
bad report FAILS the iteration instead of silently slipping through.

Why this exists: the trend graph used to cap at the (stale) metal-iters.jsonl
count and never plot the live opt/ttw/iters.jsonl rows, and report.sh `touch`es
the html even when the generator errors — so a stale graph passed the old mtime
check. These checks close both gaps with CONTENT-based assertions.

Checks
  1. Graph not stale (data): the chart series build_report feeds to the figure
     plots through the latest iteration present in opt/ttw/iters.jsonl.
  2. Graph not stale (content): the newest ttw row's idea text is present in the
     produced REPORT.html and the html embeds a chart image — proving the html
     was actually regenerated with current data (a bare `touch` cannot fake this).
  3. Freshness: REPORT.html mtime >= the newest input jsonl mtime.
  4. Descriptions: no row renders the empty-description placeholder, and every
     ttw row carries a non-empty idea/action.
  5. Recent-field sanity: ttw rows that recorded a metric have a sane numeric
     PSNR (0 < psnr < 200) and positive numeric stage timings; metric-less reject
     rows are allowed to show FAIL but must still carry idea + reason + timestamp.

Usage:  python3 opt/validate_report.py   (exit 0 = valid, non-zero = invalid)
"""
from __future__ import annotations

import sys
from pathlib import Path

OPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(OPT_DIR))

import build_report as br  # noqa: E402  (path inserted above)

REPORT_HTML = OPT_DIR / "REPORT.html"
TTW_ITERS = OPT_DIR / "ttw" / "iters.jsonl"
INPUT_JSONLS = [
    OPT_DIR / "iters.jsonl",
    OPT_DIR / "metal-iters.jsonl",
    TTW_ITERS,
]

EMPTY_DESC_MARKER = "no description recorded"
PSNR_MIN, PSNR_MAX = 0.0, 200.0


class Invalid(Exception):
    pass


def _max_ttw_iter(rows: list[dict]) -> int | None:
    nums = [int(r["iter"]) for r in rows if isinstance(r.get("iter"), int)]
    return max(nums) if nums else None


def main() -> int:
    checks: list[str] = []

    if not REPORT_HTML.exists():
        print(f"REPORT INVALID: {REPORT_HTML} does not exist (generator never wrote it)")
        return 1
    html = REPORT_HTML.read_text(errors="replace")

    ttw_rows = br.load_ttw_iters()

    try:
        # --- 1. Graph not stale (data) -------------------------------------
        if ttw_rows:
            max_ledger = _max_ttw_iter(ttw_rows)
            series = br.merged_iter_series()
            plotted = [
                int(r["iter_dir"].split("-")[1])
                for r in series
                if r.get("_source") == "ttw" and r.get("iter_dir", "").startswith("ttw-")
            ]
            if not plotted:
                raise Invalid(
                    "chart series contains NO live ttw rows — the graph is stale "
                    "(capped at the metal-iters.jsonl count, live loop excluded)"
                )
            max_plotted = max(plotted)
            if max_plotted < max_ledger:
                raise Invalid(
                    f"chart stops at ttw iter {max_plotted} but opt/ttw/iters.jsonl "
                    f"reaches iter {max_ledger} (stale graph)"
                )
            checks.append(
                f"graph plots through latest iteration "
                f"(charted iter {max_plotted} >= ledger iter {max_ledger})"
            )

            # --- 2. Graph not stale (content of produced html) -------------
            if "data:image/png;base64," not in html and "<svg" not in html:
                raise Invalid("REPORT.html embeds no chart image (generator likely errored)")
            newest = max(ttw_rows, key=lambda r: str(r.get("ts", "")))
            newest_idea = str(newest.get("idea") or "").strip()
            if newest_idea and newest_idea not in html:
                raise Invalid(
                    "newest ttw idea text is absent from REPORT.html — html was NOT "
                    "regenerated from current data (report.sh touch over a failed "
                    f"generator?). Missing: {newest_idea[:80]!r}"
                )
            checks.append("REPORT.html content reflects the newest ttw iteration (fresh, not touched)")
        else:
            checks.append("no ttw rows yet — graph-staleness checks skipped")

        # --- 3. Freshness (mtime) ----------------------------------------------
        report_mtime = REPORT_HTML.stat().st_mtime
        newest_input = None
        newest_input_mtime = 0.0
        for p in INPUT_JSONLS:
            if p.exists() and p.stat().st_mtime > newest_input_mtime:
                newest_input_mtime = p.stat().st_mtime
                newest_input = p
        if newest_input is not None and report_mtime + 1.0 < newest_input_mtime:
            raise Invalid(
                f"REPORT.html (mtime {report_mtime:.0f}) is older than input "
                f"{newest_input.name} (mtime {newest_input_mtime:.0f}) — not regenerated"
            )
        checks.append("REPORT.html mtime >= newest input jsonl")

        # --- 4. Descriptions present ------------------------------------------
        if EMPTY_DESC_MARKER in html:
            raise Invalid("a ledger row rendered with no description text")
        for r in ttw_rows:
            if not str(r.get("idea") or "").strip():
                raise Invalid(f"ttw iter {r.get('iter')} has empty idea/description")
        checks.append("every ledger row has a non-empty description")

        # --- 5. Recent-field sanity (ttw rows) --------------------------------
        for r in ttw_rows:
            it = r.get("iter")
            if not str(r.get("ts") or "").strip():
                raise Invalid(f"ttw iter {it} has no timestamp")
            metrics = r.get("metrics") or {}
            hero = metrics.get("hero_vs_ref")
            if isinstance(hero, (int, float)) and hero == hero:
                if not (PSNR_MIN < hero < PSNR_MAX):
                    raise Invalid(
                        f"ttw iter {it} PSNR {hero} out of sane range "
                        f"({PSNR_MIN}, {PSNR_MAX})"
                    )
                timings = r.get("timings") or {}
                # A stage may legitimately report 0.0 when it is folded into an
                # adjacent stage by the pipeline (e.g. since the sort->blend pipe
                # in cpp#121, blend exec runs chained inside sort and its cost is
                # attributed to sort, so blend=0.0). Only a NEGATIVE timing is
                # actually impossible/corrupt.
                for k, v in timings.items():
                    if isinstance(v, (int, float)) and v < 0:
                        raise Invalid(f"ttw iter {it} stage timing {k}={v} is negative")
            else:
                # metric-less reject row: allowed to FAIL but must carry context.
                if not str(r.get("reason") or "").strip():
                    raise Invalid(
                        f"ttw iter {it} recorded no metric AND no reason — "
                        f"un-diagnosable row"
                    )
        checks.append("recent rows have sane PSNR/timings (or a reason when metric-less)")

        # --- 6. Hero + 10× diff on every KEPT ttw ledger row ----------------
        # Blocked/reject rows are diagnostic only — no verify gate, no shots.
        shots_root = OPT_DIR / "metal-screenshots"
        kept_rows = [
            r
            for r in ttw_rows
            if str(r.get("decision") or "").lower() not in ("blocked", "reject")
        ]
        for r in kept_rows:
            it = r.get("iter")
            iter_dir = str(r.get("iter_dir") or "").strip()
            if not iter_dir and it is not None:
                iter_dir = f"ttw-{int(it):03d}"
            if not iter_dir:
                shot = str(r.get("screenshot") or "")
                if shot:
                    iter_dir = Path(shot).parent.name
            if not iter_dir:
                raise Invalid(f"ttw iter {it} has no iter_dir / screenshot path")
            base = shots_root / iter_dir
            hero = base / "hero.png"
            diff = base / "hero_diff10.png"
            if not diff.exists() and (base / "diff10x.png").exists():
                diff = base / "diff10x.png"
            if not hero.exists():
                raise Invalid(
                    f"ttw iter {it} ({iter_dir}) missing hero.png — "
                    f"re-run verify with --iter-dir {iter_dir}"
                )
            if not diff.exists():
                raise Invalid(
                    f"ttw iter {it} ({iter_dir}) missing hero_diff10.png — "
                    f"re-run verify or python3 opt/build_report.py"
                )
        if kept_rows:
            checks.append(
                f"every kept ttw row has hero.png + 10× diff ({len(kept_rows)} rows)"
            )

    except Invalid as e:
        for c in checks:
            print(f"  [PASS] {c}")
        print(f"REPORT INVALID: {e}")
        return 1

    print("===== REPORT validation =====")
    for c in checks:
        print(f"  [PASS] {c}")
    print("RESULT: REPORT VALID")
    return 0


if __name__ == "__main__":
    sys.exit(main())
