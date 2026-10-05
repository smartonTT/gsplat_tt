"""Regenerate opt/REPORT.html from opt/iters.jsonl + opt/metal-iters.jsonl + opt/ttw/iters.jsonl.

Also writes an identical copy to opt/ttw/REPORT.html (never the tt-workflows stub).

Regenerate after every iteration (keep or reject) and whenever opt/current-iter.json
changes (in-flight card at top of ledger).

CLI:
  python3 opt/build_report.py
  python3 opt/build_report.py --set-in-flight '<json object>'
  python3 opt/build_report.py --clear-in-flight
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import re
import statistics
import sys
from html import escape as html_escape
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


OPT_DIR = Path(__file__).resolve().parent
ITERS_JSONL = OPT_DIR / "iters.jsonl"
METAL_ITERS_JSONL = OPT_DIR / "metal-iters.jsonl"
TTW_ITERS_JSONL = OPT_DIR / "ttw" / "iters.jsonl"
CURRENT_ITER_JSON = OPT_DIR / "current-iter.json"
METAL_SCREENSHOTS_DIR = OPT_DIR / "metal-screenshots"
REPORT_HTML = OPT_DIR / "REPORT.html"
REPORT_HTML_TTW = OPT_DIR / "ttw" / "REPORT.html"
SCREENSHOTS_DIR = OPT_DIR / "screenshots"
VIEWS_PER_RUN = 30  # 30-view bicycle bench; iter sums always cover all 30 views
TARGET_MS_PER_FRAME = 1.0  # 1 ms per frame on bh-30 — the final goal
TARGET_SUM_MS = TARGET_MS_PER_FRAME * VIEWS_PER_RUN  # legacy helper (= 30 ms)
STAGE_KEYS = ("project_ms", "tile_assign_ms", "sort_ms", "cull_ms", "blend_ms")
STAGE_TIMING_KEYS = ("project", "tile_assign", "sort", "cull", "blend")
REF_DIR = OPT_DIR.parent / "benchmarks" / "reference_v2"


def now_ts_minutes() -> str:
    """Current time in the local timezone, minute resolution."""
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z").rstrip()


def format_ts_minutes(ts: str) -> str:
    """Parse ISO timestamp and format in local time (YYYY-MM-DD HH:MM)."""
    if not ts:
        return "—"
    try:
        s = ts.strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone().strftime("%Y-%m-%d %H:%M")
    except (ValueError, TypeError):
        if len(ts) >= 16:
            return ts[:16].replace("T", " ")
        return ts


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def load_current_iter() -> dict | None:
    if not CURRENT_ITER_JSON.exists():
        return None
    try:
        row = json.loads(CURRENT_ITER_JSON.read_text())
    except json.JSONDecodeError:
        return None
    return row if isinstance(row, dict) else None


def write_current_iter(row: dict) -> None:
    row = dict(row)
    row.setdefault("updated_at", datetime.now(timezone.utc).replace(microsecond=0).isoformat())
    CURRENT_ITER_JSON.parent.mkdir(parents=True, exist_ok=True)
    CURRENT_ITER_JSON.write_text(json.dumps(row, indent=2) + "\n")


def clear_current_iter() -> None:
    if CURRENT_ITER_JSON.exists():
        CURRENT_ITER_JSON.unlink()


def load_ttw_iters() -> list[dict]:
    return _read_jsonl(TTW_ITERS_JSONL)


def normalize_ttw_row(r: dict) -> dict:
    """Map tt-workflows iterlog entries into ledger card shape."""
    n = r.get("iter")
    idea = str(r.get("idea") or "")
    decision = str(r.get("decision") or "")
    if decision == "keep":
        verdict = "WIN"
    elif decision == "reject":
        verdict = "FAIL"
    else:
        verdict = decision.upper() if decision else "—"
    timings = r.get("timings") or {}
    stage_map = {
        "proj": "project_ms",
        "project": "project_ms",
        "ta": "tile_assign_ms",
        "tile_assign": "tile_assign_ms",
        "sort": "sort_ms",
        "cull": "cull_ms",
        "blend": "blend_ms",
    }
    per_stage: dict[str, float] = {}
    for src, dst in stage_map.items():
        v = timings.get(src)
        if isinstance(v, (int, float)) and v == v:
            per_stage[dst] = float(v)
    metrics = r.get("metrics") or {}
    hero = metrics.get("hero_vs_ref")
    if not isinstance(hero, (int, float)):
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and v == v:
                hero = v
                break
    # Prefer the measured per-view total (includes d2h/head/tail gaps); the
    # stage sum omits those and reads ~0.25 ms low.
    view_ms = timings.get("ms_view")
    if not (isinstance(view_ms, (int, float)) and view_ms == view_ms and view_ms > 0):
        view_ms = metrics.get("frame_ms_view")
    if isinstance(view_ms, (int, float)) and view_ms == view_ms and view_ms > 0:
        sum_ms = float(view_ms) * VIEWS_PER_RUN
    else:
        sum_ms = sum(per_stage.values()) * VIEWS_PER_RUN if per_stage else None
    ts = r.get("ts") or r.get("timestamp") or ""
    iter_dir = str(r.get("iter_dir") or "").strip()
    if not iter_dir and n is not None:
        iter_dir = f"ttw-{int(n):03d}"
    if not iter_dir:
        shot = str(r.get("screenshot") or "")
        if shot:
            iter_dir = Path(shot).parent.name
    if not iter_dir:
        iter_dir = "ttw-unknown"
    return {
        "iter_dir": iter_dir,
        "timestamp": ts,
        "verdict": verdict,
        "action": idea,
        "note": r.get("reason") or r.get("notes") or "",
        "hero_psnr_dB": hero,
        # hero_vs_ref == 100.0 is a capped "bit-identical to golden" marker,
        # not a PSNR against benchmarks/reference_v2; the card says so.
        "_hero_golden_marker": isinstance(hero, (int, float)) and hero >= 100.0,
        "ms_per_view": (sum_ms / VIEWS_PER_RUN) if sum_ms else None,
        "sum_total_ms": sum_ms,
        "per_stage_median_ms": per_stage,
        "psnr_per_view": {"hero": float(hero)} if isinstance(hero, (int, float)) else {},
        "_runtime": "blackhole",
        "_source": "ttw",
        "validator_reasoning": r.get("reason") or "",
        "buildid": r.get("buildid"),
        "tracy": r.get("tracy") or "",
        "iter": n,
        "device_screenshot": r.get("device_screenshot"),
    }


def normalize_in_flight_row(row: dict) -> dict:
    """Shape opt/current-iter.json for the same card renderer as completed iters."""
    timings = row.get("timings") or {}
    stage_map = {
        "proj": "project_ms",
        "ta": "tile_assign_ms",
        "sort": "sort_ms",
        "cull": "cull_ms",
        "blend": "blend_ms",
    }
    per_stage: dict[str, float] = {}
    for src, dst in stage_map.items():
        v = timings.get(src)
        if isinstance(v, (int, float)) and v == v:
            per_stage[dst] = float(v)
    metrics = row.get("metrics") or {}
    hero = metrics.get("hero_vs_ref")
    n = row.get("iter")
    idea = str(row.get("idea") or "")
    decision = row.get("decision")
    status = str(row.get("status") or "in_progress")
    if decision == "keep":
        verdict = "WIN"
    elif decision == "reject":
        verdict = "FAIL"
    elif decision:
        verdict = str(decision).upper()
    else:
        verdict = f"IN FLIGHT ({status})"
    iter_dir = row.get("iter_dir") or (f"ttw-{int(n):03d}" if n is not None else "current")
    return {
        "iter_dir": iter_dir,
        "timestamp": row.get("updated_at") or row.get("started_at") or "",
        "verdict": verdict,
        "action": idea,
        "note": row.get("note") or row.get("reason") or "",
        "hero_psnr_dB": hero,
        # hero_vs_ref == 100.0 is a capped "bit-identical to golden" marker,
        # not a PSNR against benchmarks/reference_v2; the card says so.
        "_hero_golden_marker": isinstance(hero, (int, float)) and hero >= 100.0,
        "per_stage_median_ms": per_stage,
        "psnr_per_view": {"hero": float(hero)} if isinstance(hero, (int, float)) else {},
        "_runtime": "blackhole",
        "_source": "in_flight",
        "_in_flight": True,
        "validator_reasoning": row.get("note") or "",
    }


def img_link(src: str, cls: str = "thumb", title: str = "") -> str:
    t = f' title="{title}"' if title else ""
    return (f'<a href="{src}" class="zoom" target="_blank" rel="noopener"{t}>'
            f'<img src="{src}" class="{cls}" alt="{title}"></a>')


def viewer_link(shot: dict, mode: str, title: str) -> str:
    """Thumb that opens the compare viewer (ref / candidate / diff / split).

    The data-img-* paths are relative to opt/REPORT.html; rebase_for_ttw
    rewrites them for the opt/ttw/ mirror like src/href.
    """
    src = _opt_href(shot["hero" if mode == "cand" else "diff"])
    t = html_escape(title, quote=True)
    return (f'<a href="{src}" class="zoom" target="_blank" rel="noopener" title="{t}" '
            f'data-viewer-mode="{mode}" data-viewer-label="{html_escape(shot["ref"], quote=True)}" '
            f'data-img-ref="{_opt_href(shot["ref"])}" data-img-cand="{_opt_href(shot["hero"])}" '
            f'data-img-diff="{_opt_href(shot["diff"])}">'
            f'<img src="{src}" class="thumb" alt="{t}"></a>')


def golden_badge(shot: dict) -> str:
    """md5/golden match as its own badge (never shown as the PSNR)."""
    gold = shot.get("golden")
    if not gold or not isinstance(shot.get("golden_match"), bool):
        return ""
    g = html_escape(gold)
    if shot["golden_match"]:
        return (f"<p class='golden-badge golden-ok' title='pixel-identical to {g}'>"
                f"✓ golden match <code>{g}</code></p>")
    return (f"<p class='golden-badge golden-diff' title='differs from {g}'>"
            f"✗ differs from golden <code>{g}</code></p>")


def _read_timing_jsonl(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    per_stage: dict[str, list[float]] = {}
    for tr in rows:
        for src_key, dst_key in zip(STAGE_TIMING_KEYS, STAGE_KEYS):
            v = tr.get(src_key)
            if isinstance(v, (int, float)) and v == v:
                per_stage.setdefault(dst_key, []).append(float(v))
    return {k: statistics.median(v) for k, v in per_stage.items() if v}


def enrich_stage_medians(row: dict, runtime: str = "cpu") -> dict:
    """Fill per_stage_median_ms from timing.jsonl when the jsonl row lacks it.
    Searches both flat (`<dir>/timing.jsonl`) and nested (`<dir>/<sub>/timing.jsonl`)
    layouts; for metal iters, prefers tt > cpu_cpp_mac > cpu subdir."""
    stages = dict(row.get("per_stage_median_ms") or {})
    if stages:
        return stages
    iter_dir = row.get("iter_dir", "")
    if not iter_dir:
        return stages
    base = OPT_DIR / ("metal-screenshots" if runtime == "blackhole" else "screenshots") / iter_dir
    if not base.exists():
        return stages
    stages = _read_timing_jsonl(base / "timing.jsonl")
    if stages:
        return stages
    for sub in ("tt", "cpu_cpp_mac", "cpu", "default"):
        stages = _read_timing_jsonl(base / sub / "timing.jsonl")
        if stages:
            return stages
    return stages


def rows_with_stages(rows: list[dict]) -> list[dict]:
    return [{**r, "per_stage_median_ms": enrich_stage_medians(r, r.get("_runtime", "cpu"))} for r in rows]


def _candidate_hero_dirs(iter_dir: str) -> list[Path]:
    """All plausible locations of an iter's hero.png — top-level + sub-runtime dirs.
    Sub-dirs (tt, cpu_cpp_mac, cpu, default) are common for metal iters that
    rendered multiple backends side by side."""
    if not iter_dir:
        return []
    out: list[Path] = []
    for base_dir in (SCREENSHOTS_DIR, OPT_DIR / "metal-screenshots"):
        root = base_dir / iter_dir
        if not root.exists():
            continue
        out.append(root)
        for sub in ("tt", "cpu_cpp_mac", "cpu", "default"):
            sub_path = root / sub
            if sub_path.is_dir():
                out.append(sub_path)
    return out


def _diff_image_names() -> tuple[str, ...]:
    return ("hero_diff10.png", "diff10x.png")


def _pick_diff_path(base: Path) -> str:
    rel = base.relative_to(OPT_DIR)
    for name in _diff_image_names():
        if (base / name).exists():
            return f"{rel.as_posix()}/{name}"
    return ""


def ensure_hero_diff10(iter_dir: str) -> None:
    """Write hero_diff10.png where missing.

    Skips dirs that already have hero_diff10.png (from a003_verify vs cpu_cpp_mb).
    Otherwise prefers ref.png in the same dir, then benchmarks/reference_v2/hero.png.
    """
    if not iter_dir:
        return
    import numpy as np
    from PIL import Image

    ref_v2 = REF_DIR / "hero.png"
    ref_v2_rgb = None
    if ref_v2.exists():
        ref_v2_rgb = np.asarray(Image.open(ref_v2).convert("RGB"), dtype=np.float64) / 255.0

    for d in _candidate_hero_dirs(iter_dir):
        hero = d / "hero.png"
        if not hero.exists():
            continue
        out = d / "hero_diff10.png"
        if out.exists():
            continue
        ref_rgb = None
        local_ref = d / "ref.png"
        if local_ref.exists():
            ref_rgb = np.asarray(Image.open(local_ref).convert("RGB"), dtype=np.float64) / 255.0
        elif ref_v2_rgb is not None:
            ref_rgb = ref_v2_rgb
        if ref_rgb is None:
            continue
        cand_rgb = np.asarray(Image.open(hero).convert("RGB"), dtype=np.float64) / 255.0
        if ref_rgb.shape != cand_rgb.shape:
            continue
        amp = np.clip(np.abs(ref_rgb - cand_rgb) * 10.0, 0.0, 1.0)
        Image.fromarray((amp * 255.0).astype(np.uint8)).save(out)


# --- Device screenshot requirement (user, 2026-10-05) -------------------------
# Every ttw iteration carries a `device_screenshot` object (see
# opt/ttw/ITERATION_CHECKLIST.md): the bicycle hero rendered ON DEVICE at the
# iteration's commit/config, a 10x diff and a PSNR against ONE named reference
# (`ref` for the PSNR, `diff_ref` for the diff; they must be the same file), the
# sweep md5, a golden-match badge (`golden`/`golden_match`, never the PSNR) and a
# written visual check for tile artifacts. Iters > SCREENSHOT_REQUIRED_AFTER
# must have it. Older iters predate the rule: they show whatever they have, with
# no placeholder, and are not checked here.
SCREENSHOT_REQUIRED_AFTER = 197
SCREENSHOT_FIELDS = ("hero", "diff", "ref", "diff_ref", "psnr_vs_ref", "md5", "commit",
                     "config", "visual_check")
SHOT_REF = "benchmarks/reference_v2/hero.png"
SHOT_GOLDEN = "tests/fixtures/hero/hero_golden_8bit.png"
REPO_ROOT = OPT_DIR.parent
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _shot_file_problem(rel: str, root: Path) -> str:
    p = root / rel
    if not p.is_file():
        return f"file {rel} does not exist"
    with p.open("rb") as f:
        if f.read(8) != PNG_MAGIC:
            return f"file {rel} is not a PNG"
    return ""


def device_screenshot_problems(r: dict, root: Path | None = None) -> list[str]:
    """Problems with a row's `device_screenshot` object ([] = valid)."""
    root = root or REPO_ROOT
    shot = r.get("device_screenshot")
    if not isinstance(shot, dict):
        return ["no device_screenshot object"]
    probs = []
    for k in SCREENSHOT_FIELDS:
        v = shot.get(k)
        if k == "psnr_vs_ref":
            ok = (isinstance(v, (int, float)) and v == v and v > 0) or v == "inf"
        else:
            ok = isinstance(v, str) and v.strip() != ""
        if not ok:
            probs.append(f"device_screenshot.{k} missing or empty")
    vc = str(shot.get("visual_check") or "").strip()
    if vc and len(vc) < 8:
        probs.append("device_screenshot.visual_check is too short to be a real note")
    for k in ("hero", "diff", "ref", "diff_ref", "golden"):
        rel = str(shot.get(k) or "").strip()
        if rel:
            fp = _shot_file_problem(rel, root)
            if fp:
                probs.append(f"device_screenshot.{k}: {fp}")
    ref, diff_ref = str(shot.get("ref") or ""), str(shot.get("diff_ref") or "")
    if ref and diff_ref and ref != diff_ref:
        probs.append(f"device_screenshot: PSNR reference {ref} and diff reference {diff_ref} differ; "
                     f"both must name the same reference")
    if shot.get("golden") and not isinstance(shot.get("golden_match"), bool):
        probs.append("device_screenshot.golden_match must be true/false when golden is set")
    hero = str(shot.get("hero") or "")
    if hero in (SHOT_REF, ref) or any(part.startswith("cpu") for part in Path(hero).parts[:-1]):
        probs.append(f"device_screenshot.hero {hero} is a CPU/reference image, not a device render")
    return probs


def _psnr(a, b) -> float:
    import numpy as np
    mse = float(((a - b) ** 2).mean())
    return float("inf") if mse == 0 else 10.0 * float(np.log10(255.0 ** 2 / mse))


def screenshot_pixel_problems(shot: dict, root: Path | None = None) -> list[str]:
    """Recompute a device_screenshot's numbers from its images ([] = consistent).

    PSNR(hero, ref) must match `psnr_vs_ref` (0.01 dB), the diff must be
    |hero - diff_ref| * 10 (1 LSB), and `golden_match` must say whether hero is
    byte-identical in pixels to `golden`. Also checks a nested `knob_on` shot.
    """
    import numpy as np
    from PIL import Image
    root = root or REPO_ROOT
    probs: list[str] = []
    for name, s in (("device_screenshot", shot), ("device_screenshot.knob_on", shot.get("knob_on"))):
        if not isinstance(s, dict) or not s.get("hero"):
            continue
        ref_rel = s.get("ref") or shot.get("ref")
        diff_ref_rel = s.get("diff_ref") or shot.get("diff_ref")
        try:
            rgb = lambda rel: np.asarray(Image.open(root / rel).convert("RGB"), dtype=np.float64)
            hero = rgb(s["hero"])
            ref = rgb(ref_rel)
            p = _psnr(hero, ref)
            want = s.get("psnr_vs_ref")
            want_f = float("inf") if want == "inf" else float(want)
            if not (p == want_f or abs(p - want_f) <= 0.01):
                probs.append(f"{name}.psnr_vs_ref {want} but PSNR({s['hero']}, {ref_rel}) = {p:.3f}")
            if s.get("diff"):
                want_diff = np.clip(np.abs(hero - rgb(diff_ref_rel)) * 10.0, 0, 255)
                if np.abs(rgb(s["diff"]) - want_diff).max() > 1.0:
                    probs.append(f"{name}.diff {s['diff']} is not |hero - {diff_ref_rel}| * 10")
            gold_rel = s.get("golden") or shot.get("golden")
            if gold_rel and isinstance(s.get("golden_match", shot.get("golden_match")), bool):
                same = bool(np.array_equal(hero, rgb(gold_rel)))
                if same != s.get("golden_match", shot.get("golden_match")):
                    probs.append(f"{name}.golden_match says {not same} but hero vs {gold_rel} "
                                 f"is {'identical' if same else 'different'}")
        except Exception as e:  # unreadable image, bad number, ...
            probs.append(f"{name}: cannot recompute PSNR/diff ({type(e).__name__}: {e})")
    return probs


def screenshot_status(r: dict, root: Path | None = None) -> tuple[str, list[str]]:
    """('ok' | 'missing' | 'legacy', problems) for a ttw row."""
    it = r.get("iter")
    has_shot = r.get("device_screenshot") is not None
    if not has_shot and (not isinstance(it, int) or it <= SCREENSHOT_REQUIRED_AFTER):
        return "legacy", []
    if has_shot:
        probs = device_screenshot_problems(r, root)
        return ("missing" if probs else "ok"), probs
    return "missing", ["no device_screenshot object"]


def check_device_screenshots(rows: list[dict], root: Path | None = None,
                             pixels: bool = False) -> list[str]:
    """Errors over all ttw rows; `pixels` also recomputes PSNR/diff from the images."""
    errors: list[str] = []
    for r in rows:
        status, probs = screenshot_status(r, root)
        if status == "ok" and pixels:
            probs = screenshot_pixel_problems(r["device_screenshot"], root)
        errors.extend(f"ttw iter {r.get('iter')}: {p}" for p in probs)
    return errors


def _opt_href(rel: str) -> str:
    """Repo-relative path -> href relative to opt/REPORT.html."""
    return rel[4:] if rel.startswith("opt/") else "../" + rel


def load_metal_iters() -> list[dict]:
    if not METAL_ITERS_JSONL.exists():
        return []
    rows = []
    for line in METAL_ITERS_JSONL.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


_VERDICT_COLORS = {
    "PASS": "#2a9d8f",
    "BLOCKED": "#e76f51",
    "NEEDS_REVIEW": "#e9c46a",
    "FAIL": "#c44536",
}


def _pick_hero_psnr(r: dict) -> tuple[str, str]:
    """Return (label, value_str) for the most relevant hero PSNR metric in a row."""
    keys_priority = [
        ("hero_psnr_dB_new_ref_vs_absolute_GT_unculled", "vs absGT"),
        ("hero_psnr_dB_cpu_cpp_mac_vs_absolute_GT_unculled", "vs absGT"),
        ("hero_psnr_dB_tt_vs_cpu_30view", "tt vs cpu"),
        ("hero_psnr_dB_tt_vs_fixture", "tt vs fix"),
        ("hero_psnr_dB_cpu_vs_fixture", "cpu vs fix"),
        ("hero_psnr_dB_cpu_cpp_vs_fixture", "cpp vs fix"),
        ("hero_psnr_dB", "hero"),
    ]
    for k, label in keys_priority:
        v = r.get(k)
        if isinstance(v, (int, float)) and v == v:
            return label, f"{v:.1f}"
    psnr_d = r.get("psnr_per_view") or {}
    finite = [v for v in psnr_d.values() if isinstance(v, (int, float)) and v != float("inf") and v == v]
    if finite:
        return "min-30", f"{min(finite):.1f}"
    return "—", "—"


def metal_section(rows: list[dict]) -> str:
    if not rows:
        return """
<section>
  <h2>Metal port — TT-as-emulator (amendment-002)</h2>
  <p>Target: bh-30 P150 Blackhole, 1 ms/frame. No metal iters logged yet.</p>
</section>
"""
    head = ("<tr><th>iter</th><th>time</th><th>verdict</th><th>action</th>"
            "<th>sum_ms</th><th>PSNR</th><th>note</th></tr>")
    body = ""
    for r in reversed(rows):
        verdict = r.get("verdict", "")
        v_color = _VERDICT_COLORS.get(verdict, "#777")
        verdict_html = f"<span style='color:{v_color};font-weight:600'>{verdict}</span>"
        label, psnr_str = _pick_hero_psnr(r)
        psnr_html = f"<b>{psnr_str}</b> <small style='color:#777'>{label}</small>"
        sum_ms = r.get("sum_total_ms") or r.get("sum_total_ms_cpu_cpp_mac_30view") or r.get("sum_total_ms_tt") or r.get("sum_total_ms_cpu")
        sum_ms_str = f"{sum_ms:.1f}" if isinstance(sum_ms, (int, float)) and sum_ms == sum_ms else "—"
        ts = format_ts_minutes(r.get("timestamp", ""))
        note = (r.get("note") or "")
        note_short = note[:160] + ("…" if len(note) > 160 else "")
        body += (
            f"<tr><td><a href='metal-screenshots/{r['iter_dir']}/'>{r['iter_dir']}</a></td>"
            f"<td><small>{ts}</small></td>"
            f"<td>{verdict_html}</td><td><small>{r.get('action','')}</small></td>"
            f"<td>{sum_ms_str}</td><td>{psnr_html}</td>"
            f"<td><small title='{note}'>{note_short}</small></td></tr>"
        )
    return (f"<section><h2>Metal port — TT-as-emulator (amendment-002)</h2>"
            f"<p>Reference of record: <code>cpu_cpp_mb</code> backend; PSNR gated against "
            f"<code>benchmarks/reference_v2/</code> (regenerated 2026-05-28, validated 72.7 dB vs absolute GT).</p>"
            f"<table class='ledger'>{head}{body}</table></section>")


def current_state_section(metal_rows: list[dict]) -> str:
    state_path = OPT_DIR / "metal-supervisor-state.json"
    if not state_path.exists():
        return ""
    state = json.loads(state_path.read_text())
    phase = state.get("phase", "unknown")
    next_action = state.get("next_action", "")
    blockers = state.get("blockers") or []
    validations = state.get("validations") or []
    fixes = state.get("fixes_landed") or []
    last_gates = state.get("last_gates") or {}

    phase_color = "#2a9d8f" if "ready" in phase or "unblocked" in phase else (
        "#e76f51" if "blocked" in phase else "#264653"
    )
    phase_html = f"<span style='color:{phase_color};font-weight:600'>{phase}</span>"

    blockers_html = ""
    if blockers:
        blockers_html = "<h3 style='color:#e76f51'>Blockers</h3><ul>" + "".join(
            f"<li><b>{b.get('id','')}:</b> {b.get('summary','')}</li>" for b in blockers
        ) + "</ul>"

    validations_html = ""
    if validations:
        validations_html = "<h3 style='color:#2a9d8f'>Validations</h3><ul>" + "".join(
            f"<li><b>{v.get('id','')}:</b> {v.get('summary','')}</li>" for v in validations
        ) + "</ul>"

    fixes_html = ""
    if fixes:
        fixes_html = "<h3>Recent fixes landed</h3><ul>" + "".join(
            f"<li><b>{f.get('id','')}:</b> {f.get('summary','')}</li>" for f in fixes
        ) + "</ul>"

    gates_html = ""
    if last_gates:
        rows_html = "".join(
            f"<tr><th>{k}</th><td>{v}</td></tr>"
            for k, v in last_gates.items()
        )
        gates_html = f"<h3>Last gates</h3><table class='kv'>{rows_html}</table>"

    ref_thumb = ""
    ref_hero = OPT_DIR.parent / "benchmarks" / "reference_v2" / "hero.png"
    if ref_hero.exists():
        ref_thumb = (
            f"<div style='float:right;margin-left:16px;text-align:center;font-size:11px;color:#777'>"
            f"<a href='../benchmarks/reference_v2/hero.png' target='_blank'>"
            f"<img src='../benchmarks/reference_v2/hero.png' style='max-width:220px;border-radius:4px;border:1px solid #ddd'>"
            f"</a><br>reference_v2/hero.png<br>(regen 2026-05-28, cf=1/16384)</div>"
        )

    updated = state.get("updated_at", "")
    return f"""
<section style='background:#f1faee;border-left:4px solid #2a9d8f;padding:12px 16px;margin-bottom:16px;overflow:hidden'>
  {ref_thumb}
  <h2 style='margin-top:0'>Current state &mdash; <small>{updated}</small></h2>
  <table class='kv'>
    <tr><th>Phase</th><td>{phase_html}</td></tr>
    <tr><th>Next action</th><td>{next_action}</td></tr>
  </table>
  {fixes_html}
  {validations_html}
  {blockers_html}
  {gates_html}
</section>
"""


def load_iters() -> list[dict]:
    if not ITERS_JSONL.exists():
        return []
    rows = []
    for line in ITERS_JSONL.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def plot_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _iter_num(iter_dir: str) -> int | None:
    """Extract leading numeric iter index from an iter_dir like 'iter-057-foo'."""
    if not iter_dir.startswith("iter-"):
        return None
    rest = iter_dir[len("iter-"):]
    head = rest.split("-", 1)[0]
    try:
        return int(head)
    except ValueError:
        return None


MIN_ITER_NUM = 20


def _normalize_metal_row(r: dict) -> dict:
    """Convert a metal-iters.jsonl row into the shape `fig_combined` expects.

    Picks the best-available sum_total_ms and hero/min PSNR across the
    heterogeneous keys metal iters use (sum_total_ms_tt, sum_total_ms_cpu,
    hero_psnr_dB_tt_vs_*, etc.). Tagged with `_runtime = 'blackhole'` so the
    plot can distinguish marker shape; CPU iters get `'cpu'`.
    """
    sum_priority = [
        "sum_total_ms_tt",                      # Blackhole device
        "sum_total_ms_30view",                  # amendment-002 supervisor iters
        "sum_total_ms",                         # untagged (early metal iters)
        "sum_total_ms_cpu_cpp_mac_30view",      # Mac CPU validation runs
        "sum_total_ms_cpu_cpp_mac",
        "sum_total_ms_cpu",                     # numpy cpu on bh-30
    ]
    sum_ms = None
    sum_src = None
    for k in sum_priority:
        v = r.get(k)
        if isinstance(v, (int, float)) and v == v:
            sum_ms, sum_src = float(v), k
            break
    # Synthesize sum from ms_per_view * 30 for supervisor exit rows.
    if sum_ms is None:
        mpv = r.get("final_ms_per_view") or r.get("ms_per_view")
        if isinstance(mpv, (int, float)) and mpv == mpv:
            sum_ms, sum_src = float(mpv) * 30.0, "ms_per_view*30"

    psnr_priority = [
        ("hero_psnr_dB_tt_vs_cpu_30view",                "tt vs cpu30"),
        ("hero_psnr_dB_tt_vs_fixture",                   "tt vs fix"),
        ("hero_psnr_tt_vs_cpu_cpp_mb_dB",                "tt vs cpu_cpp_mb"),
        ("final_psnr_tt_vs_cpu_cpp_mb_dB",               "tt vs cpu_cpp_mb"),
        ("final_psnr_from_packs_vs_cpu_cpp_mb_dB",       "packs vs cpu_cpp_mb"),
        ("hero_psnr_dB",                                 "hero"),
        ("hero_psnr_dB_new_ref_vs_absolute_GT_unculled", "ref vs absGT"),
        ("hero_psnr_dB_cpu_cpp_mac_vs_bh30_cpu",         "mac vs bh30cpu"),
    ]
    psnr = None
    psnr_src = None
    for k, label in psnr_priority:
        v = r.get(k)
        if isinstance(v, (int, float)) and v == v:
            psnr, psnr_src = float(v), label
            break
    # Infinity flag (TtBackend delegating to cpu_cpp_mb → bit-identical) plots as ∞.
    if psnr is None and r.get("psnr_tt_vs_cpu_infinity") is True:
        psnr, psnr_src = float("inf"), "tt vs cpu_cpp_mb"

    # Disambiguate same iter_dir at different timestamps with the action tag.
    ts_short = (r.get("timestamp") or "")[11:16]  # HH:MM
    short_label = r.get("iter_dir", "").removeprefix("metal-")
    if r.get("action") and r["action"] not in {"start", "record", "commit", "backburner"}:
        short_label = f"{short_label}\n{r['action'][:18]}"
    elif ts_short:
        short_label = f"{short_label}\n{ts_short}"

    # Only show a sum on the GRAPH if it's a comparable 30-view bicycle bench.
    # The 256² fixture validation runs (sum_total_ms_cpu_cpp_mac without
    # _30view suffix) report ~200 ms and would skew the scale; keep their
    # PSNR but drop the sum for plotting purposes.
    comparable_sum_sources = {
        "sum_total_ms_tt",
        "sum_total_ms_30view",
        "sum_total_ms_cpu_cpp_mac_30view",
        "sum_total_ms_cpu",
        "ms_per_view*30",
    }
    graph_sum = sum_ms if sum_src in comparable_sum_sources else None

    # Carry through per-stage subtimings (project/tile_assign/sort/blend) so the
    # graph's subtiming lines render for the metal/Blackhole port iters too —
    # these are the per-frame medians already (median over the 30 views).
    per_stage = {}
    raw_stage = r.get("per_stage_median_ms") or {}
    if isinstance(raw_stage, dict):
        for k in STAGE_KEYS:
            v = raw_stage.get(k)
            if isinstance(v, (int, float)) and v == v:
                per_stage[k] = float(v)

    return {
        "iter_dir": r.get("iter_dir", ""),
        "_label": short_label,
        "timestamp": r.get("timestamp", ""),
        "verdict": r.get("verdict", ""),
        "action": r.get("action", ""),
        "sum_total_ms": graph_sum,
        "_sum_total_ms_any": sum_ms,
        "_sum_src": sum_src,
        "per_stage_median_ms": per_stage,
        "psnr_per_view": {"hero": psnr} if psnr is not None else {},
        "_psnr_src": psnr_src,
        "_runtime": "blackhole",
        "class": r.get("class", "metal"),
        "validator_reasoning": r.get("note", ""),
    }


# iter-057 is the bicycle-scene CPU baseline. Anything before it was on
# stitch_doll (700k splats vs bicycle's 6.1M) and isn't directly comparable;
# we drop it from the chart so the trajectory stays on a single scene.
BICYCLE_START_ITER_NUM = 57


def _ttw_chart_rows() -> list[dict]:
    """Live tt-workflows loop rows (opt/ttw/iters.jsonl), normalized for the
    chart and deduped against any metal-iters.jsonl rows that already cover the
    same ttw-NNN iter (so a ttw row is never double-counted/double-plotted)."""
    metal_ttw_nums: set[int] = set()
    for r in load_metal_iters():
        d = r.get("iter_dir", "")
        if d.startswith("ttw-"):
            tail = d[4:].split("-", 1)[0]
            if tail.isdigit():
                metal_ttw_nums.add(int(tail))
    out: list[dict] = []
    for r in load_ttw_iters():
        nr = normalize_ttw_row(r)
        d = nr.get("iter_dir", "")
        if d.startswith("ttw-"):
            tail = d.split("-", 1)[1]
            if tail.isdigit() and int(tail) in metal_ttw_nums:
                continue
        out.append(nr)
    return out


def merged_iter_series() -> list[dict]:
    """Bicycle-scene CPU baseline (iter-057) followed by Blackhole/metal iters
    AND the live tt-workflows loop iters in a single chronological stream.
    Pre-bicycle CPU sprint iters (stitch_doll) are kept in iters.jsonl and the
    ledger but excluded from the chart for scene consistency.

    Both metal-iters.jsonl and ttw/iters.jsonl are Blackhole-runtime rows; they
    are merged and sorted by timestamp together so the chart's "Mxx" markers and
    x-axis run through the latest iteration (matching the ledger), rather than
    stopping at the end of the stale metal-iters.jsonl ledger."""
    cpu_rows = [
        {**r, "_runtime": "cpu", "_label": r.get("iter_dir", "")[5:].lstrip("0123456789-")[:18]}
        for r in load_iters()
        if (n := _iter_num(r.get("iter_dir", ""))) is not None and n >= BICYCLE_START_ITER_NUM
    ]
    metal_rows = [_normalize_metal_row(r) for r in load_metal_iters()]
    ttw_rows = _ttw_chart_rows()
    bh_rows = sorted(metal_rows + ttw_rows, key=lambda r: r.get("timestamp", ""))
    return cpu_rows + bh_rows


def fig_combined(rows: list[dict]) -> str:
    """Single combined figure showing the whole timeline from CPU optimization
    sprint into the Blackhole port. CPU iters get round markers; Blackhole
    iters get diamond markers and the boundary is annotated with a vertical
    line + label. Both runtimes plot on the same time axis."""
    enriched = rows_with_stages(rows)
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(13, 7.5), sharex=True,
        gridspec_kw={"height_ratios": [3, 2], "hspace": 0.08},
    )

    xs = list(range(len(enriched)))
    labels = [r.get("_label") or r.get("iter_dir", "")[:14] for r in enriched]
    iter_nums = [_iter_num(r.get("iter_dir", "")) for r in enriched]
    runtimes = [r.get("_runtime", "cpu") for r in enriched]

    # Find the CPU→Blackhole boundary (first blackhole index in the merged list).
    bh_start_idx = next((i for i, rt in enumerate(runtimes) if rt == "blackhole"), None)

    # --- TOP: total + per-stage on log Y ---
    def split_xy_by_runtime(ys_all):
        xs_cpu, ys_cpu, xs_bh, ys_bh = [], [], [], []
        for x, y, rt in zip(xs, ys_all, runtimes):
            if y is None or (isinstance(y, float) and y != y):
                continue
            if rt == "blackhole":
                xs_bh.append(x); ys_bh.append(y)
            else:
                xs_cpu.append(x); ys_cpu.append(y)
        return xs_cpu, ys_cpu, xs_bh, ys_bh

    # Convert 30-view sum_total_ms into ms/frame for plotting.
    ys = [
        (r.get("sum_total_ms") / VIEWS_PER_RUN) if isinstance(r.get("sum_total_ms"), (int, float)) and r.get("sum_total_ms") == r.get("sum_total_ms") else None
        for r in enriched
    ]
    ys_valid = [(y if (y is not None and y > 0) else None) for y in ys]
    xs_cpu, ys_cpu, xs_bh, ys_bh = split_xy_by_runtime(ys_valid)
    ax_top.set_yscale("log")
    if ys_cpu:
        ax_top.plot(xs_cpu, ys_cpu, color="#264653", linewidth=1.8, zorder=3,
                    label="ms/frame — CPU (Mac, cpu_cpp_mb)")
        colors_cpu = ["#2a9d8f" if enriched[x].get("action") == "commit" else "#e76f51"
                      for x in xs_cpu]
        ax_top.scatter(xs_cpu, ys_cpu, c=colors_cpu, s=58, zorder=4,
                       edgecolors="white", linewidths=1.4, marker="o")
    if ys_bh:
        ax_top.plot(xs_bh, ys_bh, color="#9d4edd", linewidth=2.0, zorder=3,
                    linestyle="--", label="ms/frame — Blackhole (bh-30 P150)")
        verdict_color = {"PASS":"#2a9d8f","BLOCKED":"#e76f51","NEEDS_REVIEW":"#e9c46a",
                         "BASELINE":"#1d3557","IN_PROGRESS":"#9d4edd",
                         "ACCEPTED_FLOOR":"#f4a261"}
        colors_bh = [verdict_color.get(enriched[x].get("verdict",""), "#9d4edd") for x in xs_bh]
        ax_top.scatter(xs_bh, ys_bh, c=colors_bh, s=140, zorder=6,
                       edgecolors="black", linewidths=1.2, marker="D")
        for x, y in zip(xs_bh, ys_bh):
            ax_top.annotate(f"{y:.2f}", (x, y), xytext=(0, -14),
                            textcoords="offset points", ha="center", fontsize=7,
                            color="#5a189a", fontweight="bold")

    # Show BH iters with NO sum as down-triangle placeholders at the bottom of
    # the panel so they don't disappear from the top graph entirely.
    if bh_idx := [i for i, rt in enumerate(runtimes) if rt == "blackhole"]:
        bh_no_sum = [i for i in bh_idx if enriched[i].get("sum_total_ms") is None]
        if bh_no_sum:
            ax_top_ymin = min(ys_cpu + ys_bh + [0.1]) * 0.5 if (ys_cpu or ys_bh) else 0.1
            ax_top.scatter(bh_no_sum, [ax_top_ymin] * len(bh_no_sum), marker="v",
                           s=90, color="#9d4edd", alpha=0.55, zorder=4,
                           edgecolors="black", linewidths=0.8,
                           label="Blackhole iter (no timing recorded — PSNR only)")

    stage_palette = {
        "project_ms":    "#e76f51",
        "tile_assign_ms":"#f4a261",
        "sort_ms":       "#e9c46a",
        "blend_ms":      "#2a9d8f",
    }
    stage_marker = {
        "project_ms":     "o",
        "tile_assign_ms": "s",
        "sort_ms":        "^",
        "blend_ms":       "P",
    }
    for k in STAGE_KEYS:
        ys_s_all = [(r.get("per_stage_median_ms") or {}).get(k) for r in enriched]
        pts = [(x, y) for x, y in zip(xs, ys_s_all) if isinstance(y, (int, float)) and y == y and y > 0]
        if not pts:
            continue
        xs_s = [p[0] for p in pts]
        ys_s = [p[1] for p in pts]
        ax_top.plot(xs_s, ys_s, marker=stage_marker.get(k, "."), markersize=5,
                    linewidth=1.3, alpha=0.8, color=stage_palette.get(k),
                    label=k.replace("_ms", "") + " (per-frame median)")
        # Annotate the most-recent value of this subtiming so the breakdown is
        # readable at a glance (these are the per-stage medians per frame).
        lx, ly = xs_s[-1], ys_s[-1]
        ax_top.annotate(f"{ly:.1f}", (lx, ly), xytext=(4, 3),
                        textcoords="offset points", fontsize=6.5,
                        color=stage_palette.get(k), fontweight="bold")

    ax_top.axhline(TARGET_MS_PER_FRAME, color="#e9c46a", linestyle="--", linewidth=1.0,
                   alpha=0.7, label=f"target {TARGET_MS_PER_FRAME:.0f} ms/frame")
    if bh_start_idx is not None:
        for ax in (ax_top, ax_bot):
            ax.axvline(bh_start_idx - 0.5, color="#9d4edd", linestyle=":",
                       linewidth=1.8, alpha=0.85, zorder=1)
        # Place the boundary label INSIDE the panel at the bottom-left of the BH
        # region so it doesn't fight with the title or legend.
        ax_top.text(
            bh_start_idx - 0.45, ax_top.get_ylim()[0] * 1.4,
            "Blackhole port begins →",
            fontsize=9, color="#9d4edd", fontweight="bold",
            ha="left", va="bottom",
        )
    ax_top.set_ylabel("ms per frame (log scale)")
    ax_top.set_title("Per-frame timing — bicycle scene (6.1M splats, 30 views averaged)   "
                     "○ CPU baseline    ◇ Blackhole (with timing)    ▽ Blackhole (PSNR only)",
                     fontsize=11)
    ax_top.legend(loc="lower left", fontsize=8, ncol=2, framealpha=0.92)
    ax_top.grid(alpha=0.25, which="both")

    # --- BOTTOM: min-PSNR (finite plotted; all-inf iters get a ∞ marker) ---
    psnr_mins: list[float | None] = []
    psnr_inf_flags: list[bool] = []
    for r in enriched:
        psnr_d = r.get("psnr_per_view") or {}
        vals = [v for v in psnr_d.values() if isinstance(v, (int, float)) and v == v]
        finite = [v for v in vals if v != float("inf")]
        if finite:
            psnr_mins.append(min(finite))
            psnr_inf_flags.append(False)
        elif vals:
            psnr_mins.append(None)
            psnr_inf_flags.append(True)
        else:
            psnr_mins.append(None)
            psnr_inf_flags.append(False)

    # CPU iters: solid line + circles. Blackhole iters: dashed + diamonds.
    xs_cpu_p, ys_cpu_p, xs_bh_p, ys_bh_p = [], [], [], []
    for x, y, rt in zip(xs, psnr_mins, runtimes):
        if y is None: continue
        (xs_bh_p if rt == "blackhole" else xs_cpu_p).append(x)
        (ys_bh_p if rt == "blackhole" else ys_cpu_p).append(y)
    if ys_cpu_p:
        ax_bot.plot(xs_cpu_p, ys_cpu_p, color="#1d3557", marker="o", markersize=5,
                    linewidth=1.4, label="min PSNR — CPU (Mac)")
    if ys_bh_p:
        ax_bot.plot(xs_bh_p, ys_bh_p, color="#9d4edd", marker="D", markersize=7,
                    linewidth=1.4, linestyle="--",
                    markeredgecolor="black", markeredgewidth=0.8,
                    label="hero PSNR — Blackhole (bh-30)")

    xs_inf = [x for x, f in zip(xs, psnr_inf_flags) if f]
    if xs_inf:
        Y_INF = 95.0
        ax_bot.scatter(xs_inf, [Y_INF] * len(xs_inf), marker="*", s=140, color="#2a9d8f",
                       zorder=5, edgecolors="white", linewidths=1.0,
                       label="bit-identical to reference (∞ dB)")
        for x in xs_inf:
            ax_bot.annotate("∞", (x, Y_INF), xytext=(0, 8), textcoords="offset points",
                            ha="center", fontsize=10, color="#2a9d8f", fontweight="bold")
    ax_bot.axhline(PSNR_FLOOR, color="#e9c46a", linestyle="--", linewidth=1.0,
                   alpha=0.7, label=f"floor {PSNR_FLOOR:.0f} dB")
    ax_bot.set_ylabel("PSNR (dB)")
    ax_bot.set_xlabel(f"iter (bicycle baseline iter-{BICYCLE_START_ITER_NUM:03d} → Blackhole port)")
    ax_bot.set_ylim(0, 110)
    ax_bot.legend(loc="lower left", fontsize=8)
    ax_bot.grid(alpha=0.25)

    # X ticks: with very few iters now (bicycle only), label every one.
    cpu_idx = [i for i, rt in enumerate(runtimes) if rt == "cpu"]
    bh_idx  = [i for i, rt in enumerate(runtimes) if rt == "blackhole"]
    tick_labels = []
    for i in xs:
        if runtimes[i] == "cpu":
            tick_labels.append(f"{iter_nums[i]:03d}" if iter_nums[i] is not None else "")
        else:
            tick_labels.append(f"M{bh_idx.index(i)+1:02d}")
    ax_bot.set_xticks(xs)
    ax_bot.set_xticklabels(tick_labels, fontsize=8)

    # Bottom descriptors: every iter visible now since count is small.
    y_min = ax_bot.get_ylim()[0]
    for i, lab in enumerate(labels):
        color = "#9d4edd" if runtimes[i] == "blackhole" else "#264653"
        ax_bot.annotate(
            lab.replace("\n", " "), xy=(i, y_min), xytext=(0, -18),
            textcoords="offset points",
            ha="right", va="top", fontsize=7, color=color, rotation=45,
            annotation_clip=False,
        )
    fig.subplots_adjust(bottom=0.28)
    return plot_b64(fig)


def status_section(rows: list[dict]) -> str:
    if not rows:
        return '<section><h2>Status</h2><p>No iters yet. Phase 0 just landed.</p></section>'
    committed = [r for r in rows if r.get("action") == "commit"]
    best = min((r["sum_total_ms"] for r in committed), default=float("inf"))
    best_str = f"{best / VIEWS_PER_RUN:.2f} ms/frame" if best != float("inf") else "n/a"
    last = rows[-1]
    last_str = f"{last['iter_dir']} → {last['verdict']} / {last['action']}"
    escalations = [r for r in rows if r.get("high_promotion_priority")]
    esc_html = ""
    if escalations:
        esc_html = "<div class='escalations'><b>ESCALATIONS</b><ul>" + "".join(
            f"<li>{r['iter_dir']}: {r.get('validator_reasoning', '')}</li>" for r in escalations
        ) + "</ul></div>"
    return f"""
<section>
  <h2>Status</h2>
  {esc_html}
  <table class='kv'>
    <tr><th>Last iter</th><td>{last_str}</td></tr>
    <tr><th>Best ms/frame (committed)</th><td>{best_str}</td></tr>
    <tr><th>Target</th><td>&lt; {TARGET_MS_PER_FRAME:.0f} ms/frame</td></tr>
    <tr><th>Iters so far</th><td>{len(rows)} ({sum(1 for r in rows if r.get('action')=='commit')} committed)</td></tr>
  </table>
</section>
"""


def _find_hero_paths(iter_dir: str, runtime: str) -> tuple[str, str] | None:
    """Locate (hero_src, diff_src) relative paths for an iter dir.

    Prefers a backend-specific subdir hero (tt/, cpu_cpp_mac/, cpu/, default/)
    because those are the iter's *actual* render output. The top-level hero is
    only used when no subdir hero exists; it is otherwise the backfilled
    scene-reference render from scripts/backfill_missing_shots.sh or
    scripts/log_iter.sh, which does NOT represent the iter's state and
    misleadingly produces a near-zero diff vs reference_v2/hero.png.
    """
    base = OPT_DIR / ("metal-screenshots" if runtime == "blackhole" else "screenshots") / iter_dir
    if not base.exists():
        return None
    prefix = ("metal-screenshots/" if runtime == "blackhole" else "screenshots/") + iter_dir
    sub_priority = ("tt", "cpu_cpp_mac", "cpu", "default") if runtime == "blackhole" else ("cpu_cpp_mac", "tt", "cpu", "default")
    for sub in sub_priority:
        sub_base = base / sub
        if (sub_base / "hero.png").exists():
            diff = _pick_diff_path(sub_base)
            return (f"{prefix}/{sub}/hero.png", diff)
    if (base / "hero.png").exists():
        diff = _pick_diff_path(base)
        return (f"{prefix}/hero.png", diff)
    return None


def _hero_source_kind(iter_dir: str, runtime: str) -> str:
    """`'iter'` if hero is from an iter-native backend subdir (actual render
    output), `'backfill'` if it's only the top-level scene-reference render
    from a post-hoc backfill, or `''` if neither exists."""
    base = OPT_DIR / ("metal-screenshots" if runtime == "blackhole" else "screenshots") / iter_dir
    if not base.exists():
        return ""
    for sub in ("tt", "cpu_cpp_mac", "cpu", "default"):
        if (base / sub / "hero.png").exists():
            return "iter"
    if (base / "hero.png").exists():
        return "backfill"
    return ""


def _preview_html(iter_dir: str, runtime: str) -> str:
    paths = _find_hero_paths(iter_dir, runtime)
    if not paths:
        return "<span style='color:#bbb;font-size:11px'>no shots</span>"
    hero_src, diff_src = paths
    parts = [f"<a href='{hero_src}' target='_blank'><img src='{hero_src}' class='ledger-thumb' alt=''></a>"]
    if diff_src:
        parts.append(f"<a href='{diff_src}' target='_blank'><img src='{diff_src}' class='ledger-thumb' alt='10× diff'></a>")
    return "".join(parts)


def _stage_table_html(stages: dict, sum_ms: float | None) -> str:
    """All times rendered as ms/frame (stage medians are already per-frame;
    sum_ms is the 30-view total so we divide by VIEWS_PER_RUN)."""
    if not stages and sum_ms is None:
        return "<p style='color:#bbb;font-size:12px;margin:6px 0'>no stage timings recorded</p>"
    rows_html = ""
    total = sum((v for v in stages.values() if isinstance(v, (int, float))), 0.0)
    for k in STAGE_KEYS:
        v = stages.get(k)
        if isinstance(v, (int, float)) and v == v:
            pct = (v / total * 100.0) if total > 0 else 0.0
            bar_w = max(2.0, min(100.0, pct))
            label = k.replace("_ms", "")
            rows_html += (
                f"<tr><th>{label}</th>"
                f"<td style='text-align:right;font-variant-numeric:tabular-nums'>{v:.2f} ms/frame</td>"
                f"<td style='width:160px'><div style='background:#264653;height:8px;width:{bar_w:.1f}%;border-radius:2px'></div></td>"
                f"<td style='color:#777'>{pct:.1f}%</td></tr>"
            )
    if not rows_html:
        return "<p style='color:#bbb;font-size:12px;margin:6px 0'>no stage timings recorded</p>"
    sum_row = ""
    if sum_ms is not None:
        per_frame = sum_ms / float(VIEWS_PER_RUN)
        sum_row = (
            f"<tr style='border-top:1px solid #ccc'><th>total</th>"
            f"<td style='text-align:right;font-weight:600;font-variant-numeric:tabular-nums'>{per_frame:.2f} ms/frame</td>"
            f"<td></td><td></td></tr>"
        )
    return f"<table class='stages'>{rows_html}{sum_row}</table>"


def _iter_card_html(r: dict, runtime: str, position_label: str = "") -> str:
    """Exactly the old backburner card structure.

    Layout (visual): h3 with [position_label] + priority + iter_dir — verdict,
    then a single line "ms_per_frame=X, psnr=Y", then a `reason` paragraph
    with the full note. Thumbs (hero.png + hero_diff10.png at 100 px) on the
    right. Nothing more.

    `position_label` is "Mxx" for the xth Blackhole iter or "iter-NNN" for the
    CPU iter index (matches the markers used on the trajectory chart).
    """
    iter_dir = r.get("iter_dir", "")
    verdict = r.get("verdict", "")
    priority = "⭐" if r.get("high_promotion_priority") else ""
    pos_html = f"<span class='iter-pos'>{position_label}</span> " if position_label else ""

    psnr_d = r.get("psnr_per_view") or {}
    finite = [v for v in psnr_d.values() if isinstance(v, (int, float)) and v != float("inf") and v == v]
    if r.get("_hero_golden_marker"):
        psnr_min_str = "golden match (bit-identical to hero_golden_8bit; not a PSNR vs reference_v2)"
    elif finite:
        psnr_min_str = f"{min(finite):.1f} dB"
    elif psnr_d and any(v == float("inf") for v in psnr_d.values() if isinstance(v, (int, float))):
        psnr_min_str = "∞ dB"
    else:
        label, val = _pick_hero_psnr(r)
        if val == "—" and r.get("psnr_tt_vs_cpu_infinity") is True:
            psnr_min_str = "∞ dB"
        elif val != "—":
            psnr_min_str = f"{val} dB ({label})"
        else:
            psnr_min_str = "n/a"

    sum_ms_30 = (r.get("sum_total_ms") or r.get("_sum_total_ms_any")
                 or r.get("sum_total_ms_cpu_cpp_mac_30view")
                 or r.get("sum_total_ms_tt") or r.get("sum_total_ms_cpu")
                 or r.get("sum_total_ms_30view"))
    if sum_ms_30 is None and isinstance(r.get("final_ms_per_view") or r.get("ms_per_view"), (int, float)):
        sum_ms_30 = float(r.get("final_ms_per_view") or r.get("ms_per_view")) * 30.0
    if isinstance(sum_ms_30, (int, float)) and sum_ms_30 == sum_ms_30:
        sum_ms_str = f"{sum_ms_30 / VIEWS_PER_RUN:.2f} ms/frame"
    else:
        sum_ms_str = "n/a"

    ensure_hero_diff10(iter_dir)
    preview_paths = _find_hero_paths(iter_dir, runtime)
    thumb_html = ""
    if preview_paths:
        hero_src, diff_src = preview_paths
        thumb_html = img_link(hero_src)
        if diff_src:
            thumb_html += img_link(diff_src)
        else:
            thumb_html += (
                "<span style='color:#c44536;font-size:11px'>missing 10× diff "
                f"(expected hero_diff10.png under {iter_dir}/)</span>"
            )
    elif iter_dir:
        thumb_html = (
            "<span style='color:#c44536;font-size:11px'>missing hero screenshot "
            f"(run verify with --iter-dir {iter_dir})</span>"
        )

    shot_html = ""
    if r.get("_source") == "ttw":
        status, probs = screenshot_status(r)
        shot = r.get("device_screenshot") or {}
        if status == "ok":
            psnr = shot["psnr_vs_ref"]
            psnr_txt = "∞" if psnr == "inf" else f"{float(psnr):.2f}"
            ref = shot["ref"]
            ref_name = html_escape(ref)
            psnr_min_str = f"{psnr_txt} dB vs {ref_name}"
            thumb_html = (
                viewer_link(shot, "cand", "device hero (click to compare)")
                + viewer_link(shot, "diff", f"10x diff vs {ref} (click to compare)")
                + f"<p class='shot-cap'>device hero · 10× diff · PSNR {psnr_txt} dB<br>"
                f"ref: <code>{ref_name}</code></p>"
                + golden_badge(shot)
            )
            device = f" on {html_escape(shot['device'])}" if shot.get("device") else ""
            shot_html = (
                f"<p class='shot-meta'>device screenshot{device}: "
                f"<code>{html_escape(shot['commit'])}</code> {html_escape(shot['config'])} · "
                f"sweep md5 <code>{html_escape(shot['md5'])}</code> · "
                f"PSNR {psnr_txt} dB vs <code>{ref_name}</code> (diff vs the same)<br>"
                f"visual check: {html_escape(shot['visual_check'])}</p>"
            )
        elif status == "missing":
            thumb_html = (
                "<div class='shot-missing'>device screenshot MISSING - required<br>"
                f"<small>{html_escape('; '.join(probs))}</small></div>"
            )
        elif not preview_paths:
            # Predates the screenshot rule and never had one: show nothing.
            thumb_html = ""

    # Description: the `action` (the idea/what-was-tried) is the primary
    # human-readable line. For metal rows action is a slug + a descriptive
    # `note`/commit message; for ttw rows action IS the full idea and the
    # note/validator_reasoning is the gate string (e.g. "63.85 >= 63.6"). Show
    # both so every row has a real description, never just a bare gate string.
    action = (r.get("action") or "").strip()
    note = (r.get("validator_reasoning") or r.get("note") or "").strip()
    desc_html = ""
    if action:
        desc_html += f"<p class='idea-desc'>{action}</p>"
    if note and note != action:
        desc_html += f"<p class='reason'>{note}</p>"
    if not desc_html:
        desc_html = "<p class='reason' style='color:#bbb'>— no description recorded —</p>"

    commit = (r.get("commit") or r.get("commit_sha") or "").strip()
    buildid = r.get("buildid")
    if not commit and isinstance(buildid, dict):
        cpp_tok = str(buildid.get("cpp") or "").split()
        if len(cpp_tok) >= 2:
            commit = cpp_tok[1]
    build_html = f"<p class='iter-build'><code>{commit}</code></p>" if commit else ""

    # Clickable Tracy trace link — a per-iteration deliverable, exactly like the
    # hero/diff screenshots. Prefer the row's `tracy` field (repo-relative,
    # e.g. opt/profiler/ttw-NNN/render.tracy); else auto-discover a per-iter
    # trace. Only emit the link when the .tracy actually exists (no dead links).
    tracy_rel = (r.get("tracy") or "").strip()
    if not tracy_rel and iter_dir:
        for cand in (OPT_DIR / "profiler" / iter_dir / "render.tracy",):
            if cand.is_file():
                tracy_rel = "opt/" + str(cand.relative_to(OPT_DIR))
                break
    tracy_html = ""
    if tracy_rel:
        abs_tracy = OPT_DIR.parent / tracy_rel
        if abs_tracy.is_file():
            href = tracy_rel[4:] if tracy_rel.startswith("opt/") else tracy_rel
            tracy_cmd = f"tracy {tracy_rel}"
            tracy_html = (
                f"<p class='iter-tracy'><a href='{href}' "
                f"title='open in Tracy profiler'>🔬 Tracy trace</a> "
                f"<code class='tracy-cmd' title='copy &amp; run to open in Tracy' "
                f"style='font-size:11px;background:#f0f0f0;padding:1px 6px;"
                f"border-radius:3px;user-select:all;cursor:text'>{tracy_cmd}</code></p>"
            )
        else:
            tracy_html = (
                "<p class='iter-tracy' style='color:#c44536;font-size:11px'>"
                f"Tracy trace missing ({tracy_rel})</p>"
            )

    ts_raw = r.get("timestamp") or r.get("ts") or r.get("updated_at") or r.get("started_at") or ""
    ts_disp = format_ts_minutes(ts_raw)
    in_flight = r.get("_in_flight") is True
    row_class = "backburner-row in-flight-row" if in_flight else "backburner-row"

    # Caveat fires only when:
    #   - the displayed hero is a top-level (no backend subdir) image AND
    #   - the row has no freshly-measured hero_psnr_dB (i.e., logged via the
    #     old `backfill_missing_shots.sh` against current code rather than
    #     `scripts/log_iter.sh` which writes hero_psnr_dB at render time).
    # In that case the hero is a scene-reference render, NOT the iter's
    # commit-state output, so a near-zero diff next to a poor historical PSNR
    # is an artifact of the backfill — not of the iter.
    hero_kind = _hero_source_kind(iter_dir, runtime)
    freshly_measured = (
        isinstance(r.get("hero_psnr_dB"), (int, float))
        or r.get("hero_psnr_dB_infinity") is True
    )
    caveat_html = ""
    if hero_kind == "backfill" and not freshly_measured:
        caveat_html = (
            "<p class='caveat'>caveat: this iter never preserved its per-backend output. "
            "Both the hero AND <code>benchmarks/reference_v2/hero.png</code> were rendered "
            "with <em>current HEAD</em> (post-hoc backfill), so the 10× diff is ~0 by "
            "construction. It does <strong>not</strong> show what this iter actually "
            "produced. The PSNR is the iter's historical measurement at its commit. "
            "Use <code>scripts/rerender_at_commit.sh</code> to get the true diff.</p>"
        )

    return f"""
<div class='{row_class}'>
  <div class='backburner-meta'>
    <h3>{pos_html}{priority} {iter_dir} — {verdict}</h3>
    <p class='iter-ts'>{ts_disp}</p>
    <p>ms_per_frame={sum_ms_str}, psnr={psnr_min_str}</p>
    {caveat_html}
    {desc_html}
    {build_html}
    {shot_html}
    {tracy_html}
  </div>
  <div class='backburner-thumbs'>{thumb_html}</div>
</div>
"""


def in_flight_section() -> str:
    row = load_current_iter()
    if not row:
        return ""
    if row.get("status") in ("done", "complete", "finished"):
        return ""
    norm = normalize_in_flight_row(row)
    card = _iter_card_html(norm, "blackhole", "NOW")
    return f"""
<section class='in-flight-banner'>
  <h2>Current iteration (in flight)</h2>
  <p style='color:#777;font-size:12px;margin-top:0'>Live state from <code>opt/current-iter.json</code>;
  refreshed on every <code>build_report.py</code> run while work is active.</p>
  {card}
</section>
"""


THROUGHPUT_JSONL = OPT_DIR / "ttw" / "throughput.jsonl"


def throughput_section() -> str:
    """Secondary metric (task #275): back-to-back throughput, ms/frame.

    Rows come from opt/ttw/throughput.jsonl (render/run.py --back-to-back). They
    never feed the primary ms/view latency, the iteration ledger or the GPU anchor."""
    rows = _read_jsonl(THROUGHPUT_JSONL)
    if not rows:
        return ""
    body = []
    for r in sorted(rows, key=lambda r: r.get("ts", ""), reverse=True):
        ms = float(r["b2b_ms_frame"])
        lat = r.get("latency_ms_view")
        drop = r.get("b2b_drop_ms_frame")
        rounds = ", ".join(f"{x:.3f}" for x in r.get("b2b_ms_frame_rounds", []))
        md5 = html_escape(str(r.get("md5", "")))
        match = " (golden match)" if r.get("golden_match") else ""
        src = html_escape(str(r.get("source", "")))
        body.append(
            f"<tr><td>{html_escape(format_ts_minutes(r.get('ts', '')))}</td>"
            f"<td>iter-{r.get('iter_ref', '?')} @ <code>{html_escape(str(r.get('commit', '')))}</code></td>"
            f"<td>{html_escape(str(r.get('board', '')))}</td>"
            f"<td><b>{ms:.3f}</b> ({1000.0 / ms:.1f} FPS)<br><small>rounds {rounds}</small></td>"
            f"<td>{'&mdash;' if drop is None else f'{float(drop):.3f}'}</td>"
            f"<td>{'&mdash;' if lat is None else f'{float(lat):.3f}'}</td>"
            f"<td><code>{md5}</code>{match}</td>"
            f"<td><a href='{_opt_href(src)}'>{src}</a></td></tr>")
    return f"""
<section style='border-left:4px solid #8d99ae;padding:8px 16px'>
  <h2 style='margin-top:0'>Throughput, back-to-back
    <span style='background:#8d99ae;color:#fff;padding:1px 8px;border-radius:10px;
      font-size:12px'>SECONDARY METRIC</span></h2>
  <p>Frames rendered back to back with no host work between views
  (<code>render/run.py --back-to-back</code>): wall time / frames. The <b>primary
  metric stays per-view latency</b> (ms/view, as in the ledger and the GPU comparison);
  this table does not change it. All rows measured on device, 3 untraced rounds.
  Kept = frames kept and checked byte for byte against a check pass;
  dropped = frames dropped as <code>render()</code> returns, like a viewer.</p>
  <table class='rows'>
    <tr><th>Measured</th><th>Tip</th><th>Board</th>
        <th>Throughput, back-to-back, ms/frame (kept)</th><th>dropped, ms/frame</th>
        <th>Latency, same session, ms/view</th><th>Sweep md5</th><th>Source</th></tr>
    {"".join(body)}
  </table>
</section>
"""


def ledger_section(rows: list[dict]) -> str:
    """Unified ledger as big backburner-style cards, one per iter.
    Sorted by timestamp descending so the newest (Blackhole) iters appear on top."""
    metal_rows_raw = sorted(load_metal_iters(), key=lambda r: r.get("timestamp", ""))
    metal_norm = [_normalize_metal_row(r) for r in metal_rows_raw]
    # _normalize_metal_row strips raw keys we need for the big card; keep the original alongside.
    for norm, raw in zip(metal_norm, metal_rows_raw):
        for k, v in raw.items():
            norm.setdefault(k, v)
    bicycle_cpu = [r for r in rows
                   if (n := _iter_num(r.get("iter_dir", ""))) is not None
                   and n >= BICYCLE_START_ITER_NUM]
    pre_cpu = [r for r in rows
               if (n := _iter_num(r.get("iter_dir", ""))) is not None
               and n < BICYCLE_START_ITER_NUM]
    ttw_rows = [normalize_ttw_row(r) for r in load_ttw_iters()]
    metal_ttw_nums: set[int] = set()
    for r in metal_rows_raw:
        d = r.get("iter_dir", "")
        if d.startswith("ttw-"):
            tail = d[4:].split("-", 1)[0]
            if tail.isdigit():
                metal_ttw_nums.add(int(tail))
    ttw_rows = [
        r for r in ttw_rows
        if not (r.get("iter_dir", "").startswith("ttw-")
                and int(r["iter_dir"].split("-", 1)[1]) in metal_ttw_nums)
    ]
    merged = (
        [{**r, "_runtime": "cpu", "_scene": "bicycle"} for r in bicycle_cpu]
        + [{**r, "_scene": "bicycle"} for r in metal_norm]
        + [{**r, "_scene": "bicycle"} for r in ttw_rows]
    )
    merged.sort(key=lambda r: r.get("timestamp", ""), reverse=True)

    # Assign chronological position labels: Mxx for metal iters (1-indexed in
    # timestamp-ascending order, matching the chart markers), iter-NNN for CPU.
    metal_chrono = sorted(
        [r for r in merged if r.get("_runtime") == "blackhole"],
        key=lambda r: r.get("timestamp", "")
    )
    metal_label_for_id = {id(r): f"M{i+1:02d}" for i, r in enumerate(metal_chrono)}
    def _label(r: dict) -> str:
        if r.get("_runtime") == "blackhole":
            return metal_label_for_id.get(id(r), "")
        d = r.get("iter_dir", "")
        n = _iter_num(d)
        return f"iter-{n:03d}" if n is not None else ""

    body = "".join(_iter_card_html(r, r.get("_runtime", "cpu"), _label(r)) for r in merged)
    pre_body = "".join(
        _iter_card_html({**r, "_runtime": "cpu"}, "cpu", _label({**r, "_runtime": "cpu"}))
        for r in reversed(pre_cpu)
    )
    pre_section = ""
    if pre_body:
        pre_section = (
            f"<details style='margin-top:16px'><summary><b>Pre-bicycle CPU sprint "
            f"({len(pre_cpu)} iters on stitch_doll — collapsed)</b></summary>"
            f"{pre_body}</details>"
        )
    return (f"<section><h2>Ledger — bicycle scene ({len(merged)} iters)</h2>"
            f"<p style='color:#777;font-size:12px;margin-top:0'>"
            f"All iters since the scene pivot to bicycle (6.1M splats) on iter-057. "
            f"CPU and Blackhole interleaved by timestamp (newest first). "
            f"Earlier stitch_doll-scene CPU sprint folded below.</p>"
            f"{body}{pre_section}</section>")


def _legacy_table_ledger_unused(rows: list[dict]) -> str:
    """Old compressed-table ledger (kept for reference, not called)."""
    head = ("<tr><th>preview</th><th>iter</th><th>runtime</th><th>verdict</th><th>action</th>"
            "<th>sum_ms</th><th>PSNR</th><th>class</th><th>commit / note</th></tr>")
    metal_rows_raw = sorted(load_metal_iters(), key=lambda r: r.get("timestamp", ""))
    metal_norm = [_normalize_metal_row(r) for r in metal_rows_raw]
    bicycle_cpu = [r for r in rows
                   if (n := _iter_num(r.get("iter_dir", ""))) is not None
                   and n >= BICYCLE_START_ITER_NUM]
    pre_cpu = [r for r in rows
               if (n := _iter_num(r.get("iter_dir", ""))) is not None
               and n < BICYCLE_START_ITER_NUM]
    merged = (
        [{**r, "_runtime": "cpu", "_scene": "bicycle"} for r in bicycle_cpu]
        + [{**r, "_scene": "bicycle"} for r in metal_norm]
    )
    merged.sort(key=lambda r: r.get("timestamp", ""), reverse=True)

    def _row_html(r: dict, in_section: str) -> str:
        psnr_d = r.get("psnr_per_view") or {}
        finite = [v for v in psnr_d.values()
                  if isinstance(v, (int, float)) and v != float("inf") and v == v]
        if finite:
            psnr_str = f"{min(finite):.1f}"
        elif psnr_d and any(v == float("inf") for v in psnr_d.values()
                            if isinstance(v, (int, float))):
            psnr_str = "∞"
        else:
            label, val = _pick_hero_psnr(r)
            psnr_str = (f"{val} <small style='color:#999'>{label}</small>"
                        if val != "—" else "—")
        sum_ms = r.get("sum_total_ms")
        if sum_ms is None:
            sum_ms = r.get("_sum_total_ms_any")
        sum_src = r.get("_sum_src", "")
        src_tag = ""
        if sum_src:
            short_src = (sum_src.replace("sum_total_ms_", "")
                                .replace("cpu_cpp_mac_30view", "mac-30v")
                                .replace("cpu_cpp_mac", "mac")
                                .replace("_30view", "-30v"))
            src_tag = f" <small style='color:#999'>{short_src}</small>"
        sum_ms_str = (f"{sum_ms:.1f}{src_tag}"
                      if isinstance(sum_ms, (int, float)) and sum_ms == sum_ms
                      else "—")
        runtime = r.get("_runtime", "cpu")
        rt_html = (
            "<span style='color:#9d4edd;font-weight:600'>◇ Blackhole</span>"
            if runtime == "blackhole"
            else "<span style='color:#264653'>○ CPU</span>"
        )
        verdict = r.get("verdict", "")
        v_color = _VERDICT_COLORS.get(verdict, {"REVERT":"#c44536","ACCEPT":"#2a9d8f",
                                                "BASELINE":"#1d3557"}.get(verdict, "#666"))
        verdict_html = f"<span style='color:{v_color};font-weight:600'>{verdict}</span>"
        commit_or_note = ((r.get("commit_sha") or "")[:8]
                          if r.get("commit_sha")
                          else (r.get("validator_reasoning") or
                                r.get("note") or "")[:60])
        link_prefix = ("metal-screenshots/" if runtime == "blackhole"
                       else "screenshots/")
        row_bg = "#fafaff" if runtime == "blackhole" else ""
        ensure_hero_diff10(r.get("iter_dir", ""))
        preview = _preview_html(r.get("iter_dir", ""), runtime)
        # Clickable Tracy trace link (a per-iteration deliverable, like the
        # hero/diff screenshots): row `tracy` field or an auto-discovered
        # opt/profiler/<iter_dir>/render.tracy. Only link when the file exists.
        tracy_rel = (r.get("tracy") or "").strip()
        if not tracy_rel and r.get("iter_dir"):
            cand = OPT_DIR / "profiler" / r["iter_dir"] / "render.tracy"
            if cand.is_file():
                tracy_rel = "opt/" + str(cand.relative_to(OPT_DIR))
        tracy_cell = ""
        if tracy_rel and (OPT_DIR.parent / tracy_rel).is_file():
            href = tracy_rel[4:] if tracy_rel.startswith("opt/") else tracy_rel
            tracy_cmd = f"tracy {tracy_rel}"
            tracy_cell = (f"<br><a href='{href}' title='open in Tracy profiler' "
                          f"style='font-size:11px'>🔬 Tracy</a>"
                          f"<br><code title='copy &amp; run to open in Tracy' "
                          f"style='font-size:10px;color:#555;user-select:all'>{tracy_cmd}</code>")
        return (
            f"<tr style='background:{row_bg}'>"
            f"<td>{preview}</td>"
            f"<td><a href='{link_prefix}{r['iter_dir']}/'>{r['iter_dir']}</a>{tracy_cell}</td>"
            f"<td>{rt_html}</td>"
            f"<td>{verdict_html}</td>"
            f"<td><small>{r.get('action','')}</small></td>"
            f"<td>{sum_ms_str}</td>"
            f"<td>{psnr_str}</td>"
            f"<td><small>{r.get('class','')}</small></td>"
            f"<td><code><small>{commit_or_note}</small></code></td>"
            f"</tr>"
        )

    body = "".join(_row_html(r, "main") for r in merged)
    pre_body = "".join(_row_html({**r, "_runtime": "cpu", "_scene": "stitch_doll"}, "pre")
                       for r in reversed(pre_cpu))
    pre_section = ""
    if pre_body:
        pre_section = (
            f"<details style='margin-top:12px'><summary><b>Pre-bicycle CPU sprint "
            f"({len(pre_cpu)} iters on stitch_doll — collapsed)</b></summary>"
            f"<table class='ledger'>{head}{pre_body}</table></details>"
        )
    return (f"<section><h2>Ledger — bicycle scene</h2>"
            f"<p style='color:#777;font-size:12px;margin-top:0'>"
            f"All iters since the scene pivot to bicycle (6.1M splats) on "
            f"iter-057. CPU and Blackhole interleaved by timestamp; "
            f"earlier stitch_doll-scene CPU sprint is below in a fold.</p>"
            f"<table class='ledger'>{head}{body}</table>"
            f"{pre_section}</section>")


CULL_JSONL = OPT_DIR / "cull_tune.jsonl"
CULL_SUMMARY = OPT_DIR / "cull_tune_summary.json"
PSNR_FLOOR = 68.0


def load_cull_tune() -> list[dict]:
    if not CULL_JSONL.exists():
        return []
    return [json.loads(line) for line in CULL_JSONL.read_text().splitlines() if line.strip()]


def cull_tune_section() -> str:
    summary = {}
    if CULL_SUMMARY.exists():
        summary = json.loads(CULL_SUMMARY.read_text())
    rows = load_cull_tune()
    if not summary and not rows:
        return ""

    rec = summary.get("recommended", {})
    head = (
        f"<tr><th>assign</th><td>{rec.get('assign_mode', '—')}</td></tr>"
        f"<tr><th>contrib_floor</th><td>1/{rec.get('contrib_floor_n', 0):.0f}</td></tr>"
        f"<tr><th>transmittance</th><td>1/{rec.get('transmittance_n', 0):.0f}</td></tr>"
        f"<tr><th>min_opacity</th><td>{rec.get('min_opacity', 0):.5f}</td></tr>"
        f"<tr><th>k_cap</th><td>{rec.get('k_cap', 0)}</td></tr>"
        f"<tr><th>worst PSNR vs GT</th><td>{summary.get('worst_psnr', 0):.2f} dB</td></tr>"
        f"<tr><th>worst max_abs</th><td>{summary.get('worst_max_abs', 0):.4f}</td></tr>"
        f"<tr><th>mean ms / view</th><td>{summary.get('mean_ms', 0):.1f}</td></tr>"
        f"<tr><th>Maha vs iso ms</th><td>{summary.get('maha_ms', 0):.1f} / {summary.get('iso_ms', 0):.1f}</td></tr>"
    )
    iter_rows = ""
    for r in rows:
        if r.get("phase") not in ("final", "baseline_loose") and r.get("search") != "contrib_floor_n":
            continue
        cfg = r.get("config") or {}
        iter_rows += (
            f"<tr><td>{r.get('phase','')}</td>"
            f"<td>{cfg.get('assign_mode','')}</td>"
            f"<td>1/{cfg.get('contrib_floor_n',0):.0f}</td>"
            f"<td>{r.get('worst_psnr',0):.2f}</td>"
            f"<td>{r.get('worst_max_abs',0):.4f}</td>"
            f"<td>{r.get('mean_ms',0):.1f}</td>"
            f"<td>{r.get('ok','')}</td></tr>"
        )
    table = ""
    if iter_rows:
        table = (
            "<h3>Iteration log (selected)</h3>"
            "<table class='ledger'><tr><th>phase</th><th>mode</th><th>floor</th>"
            "<th>PSNR</th><th>max</th><th>ms</th><th>pass</th></tr>"
            f"{iter_rows}</table>"
        )
    return f"""
<section>
  <h2>Cull threshold tuning (2026-05-27)</h2>
  <p>Reference: numpy <b>true ground truth</b> — project with min_opacity=0,
  max_radius disabled, fixed 3σ AABB; tile_assign without per-pair Mahalanobis;
  alpha_blend without microblock cull. Quality floor: PSNR ≥ {PSNR_FLOOR} dB,
  max_abs ≤ 0.05 @ 1024² stitch_doll (+ orbit + close-zoom views).</p>
  <p>Production default: <b>Mahalanobis</b> per-pair and per-microblock cull with
  <code>contrib_floor=1/255</code> plus the GPU-3DGS per-pixel blend floor
  (iter-156, <code>BLEND_PIXEL_FLOOR</code> default on). Was 1/16384; that golden is
  archived under <code>tests/fixtures/hero/archive/</code>. PSNR vs old golden
  42-43 dB (faint haze dropped); independent review task #43 PASS, no seams.</p>
  <table class='kv'>{head}</table>
  {table}
  <p>Full log: <code>opt/cull_tune.jsonl</code></p>
</section>
"""


# TT anchor for the GPU ratio: the newest measured tip, i.e. the highest-numbered
# 'keep' row in opt/ttw/iters.jsonl with a measured 30-view 1024x1024 bicycle
# timings.ms_view, labelled with its board. Falls back to iter-180 (19.65 ms,
# docs/blend-diet-t146) only if no such row exists.
def tt_anchor() -> tuple[float, str]:
    best = None
    for r in load_ttw_iters():
        ms = (r.get("timings") or {}).get("ms_view")
        if r.get("decision") != "keep" or not isinstance(ms, (int, float)):
            continue
        if best is None or int(r.get("iter", -1)) > int(best.get("iter", -1)):
            best = r
    if best is None:
        return 19.65, "Blackhole P100 (yyzo-bh-07), iter-180 blend TRISC1 diet"
    m = best.get("metrics") if isinstance(best.get("metrics"), dict) else {}
    board = m.get("board") or "board not recorded"
    commit = (best.get("buildid") or {}).get("cpp", "")
    label = f"{board}, iter-{best.get('iter')} measured tip"
    if commit:
        label += f" ({commit})"
    return float(best["timings"]["ms_view"]), label


TT_ANCHOR_MS, TT_ANCHOR_LABEL = tt_anchor()

GPU_RESULT_JSON = OPT_DIR / "cpu-vs-tt" / "gpu_result.json"

# --- Published (NOT measured) GPU reference rows -------------------------
# Every figure below is quoted from its source publication. No number here was
# measured by this project, on our bench, or on any hardware we control.
PUBLISHED_GPU_ROWS = [
    {
        "id": "G1",
        "renderer": "INRIA 3DGS (diff-gaussian-rasterization, Kerbl et al. 2023)",
        "scene": "Mip-NeRF360 bicycle",
        "gpu": "NVIDIA RTX A6000",
        "res": "1920&times;1080",
        "speed": "93 FPS (Fig. 1 teaser, bicycle)",
        "ms": 10.75,
        "src": "https://arxiv.org/abs/2308.04079",
        "src_label": "arXiv:2308.04079",
    },
    {
        "id": "G2",
        "renderer": "INRIA 3DGS, pixel-normalized to our bench",
        "scene": "bicycle",
        "gpu": "NVIDIA RTX A6000",
        "res": "1024&times;1024 (normalized)",
        "speed": "183.8 FPS-equivalent",
        "ms": 5.44,
        "src": "https://arxiv.org/abs/2308.04079",
        "src_label": "derived from G1",
    },
    {
        "id": "G3",
        "renderer": "Kovini&cacute;/Stojkovi&cacute; TT line vs their CUDA reference",
        "scene": "their test scenes",
        "gpu": "NVIDIA GTX 4060",
        "res": "not stated",
        "speed": "their TT result &asymp; 1.6&times; slower than the 4060",
        "ms": None,
        "src": "",
        "src_label": "Slack DM D0C1CV1AJJV, 2026-09-14",
    },
]


def published_gpu_section() -> str:
    """Published-literature GPU rows. Always labelled 'published, not measured'."""
    doc_link = (
        "<a href='cpu-vs-tt-comparison.md'>cpu-vs-tt-comparison.md</a>"
        " &sect; Published GPU reference rows"
    )
    rows = []
    for r in PUBLISHED_GPU_ROWS:
        ms = "&mdash;" if r["ms"] is None else f"{r['ms']:.2f}"
        ratio = (
            "&mdash;"
            if r["ms"] is None
            else f"GPU <b>{TT_ANCHOR_MS / r['ms']:.1f}&times;</b> faster"
        )
        src = (
            f"<a href='{r['src']}' target='_blank'>{r['src_label']}</a>"
            if r["src"]
            else r["src_label"]
        )
        rows.append(
            f"<tr><td>{r['id']}</td><td>{r['renderer']}</td><td>{r['scene']}</td>"
            f"<td>{r['gpu']}</td><td>{r['res']}</td><td>{r['speed']}</td>"
            f"<td>{ms}</td><td>{ratio}</td><td>{src}</td></tr>"
        )
    body = "\n".join(rows)
    return f"""
<section style='background:#fdf3f3;border-left:4px solid #e76f51;padding:12px 16px'>
  <h2 style='margin-top:0'>GPU reference &mdash;
    <span style='background:#e76f51;color:#fff;padding:1px 8px;border-radius:10px;
      font-size:12px;letter-spacing:.5px'>PUBLISHED, NOT MEASURED</span></h2>
  <p>No CUDA host is reachable from this project, so the rows below are
  <b>figures quoted from their source publications</b>. They were <b>not</b>
  measured by this project, not run on our bench, and not run on any hardware we
  control. They exist only to give the charter's &ldquo;beat the GPU&rdquo;
  criterion an order-of-magnitude reference.</p>
  <table class='rows'>
    <tr><th>#</th><th>Renderer</th><th>Scene</th><th>GPU</th><th>Resolution</th>
        <th>Published speed</th><th>ms/view</th>
        <th>vs TT ({TT_ANCHOR_MS} ms/view)</th><th>Source</th></tr>
    {body}
  </table>
  <p><b>Normalization assumption (G2):</b>
  <code>ms<sub>1024&sup2;</sub> = ms<sub>1080p</sub> &times; (1024&middot;1024)/(1920&middot;1080)
  = 10.75 &times; 0.5059 = 5.44&nbsp;ms</code>, i.e. rasterization time is assumed
  <b>linear in pixel count</b> at fixed Gaussian count. Projection, tiling and the
  depth sort are per-Gaussian and do <i>not</i> shrink with resolution, so G2 is an
  <b>optimistic (too-fast)</b> normalization and the true 1024&sup2; A6000 number
  would be higher. Not corrected for: different reconstruction (paper's 30K-iter
  bicycle vs our 6,131,954-Gaussian <code>bicycle.ply</code>), full SH vs our
  SH-degree-0 colors, and a different camera set.</p>
  <p><b>TT anchor:</b> {TT_ANCHOR_MS} ms/view ({1000.0 / TT_ANCHOR_MS:.2f} FPS)
  &mdash; {TT_ANCHOR_LABEL}. Detail, sources and caveats: {doc_link}.</p>
</section>
"""


def gpu_reference_section() -> str:
    """GPU row for the charter's 'beat the GPU' criterion.

    Data-driven: fills in from opt/cpu-vs-tt/gpu_result.json as soon as
    bench/gpu_reference/run_gpu_bench.py has been run on a CUDA host. Until
    then it states plainly that no GPU is reachable -- never an estimate.
    """
    doc_link = (
        "<a href='cpu-vs-tt-comparison.md'>cpu-vs-tt-comparison.md</a>"
        " &sect; GPU reference"
    )
    if not GPU_RESULT_JSON.exists():
        return f"""
<section style='background:#fff8e6;border-left:4px solid #e9c46a;padding:12px 16px'>
  <h2 style='margin-top:0'>GPU reference &mdash; <span style='color:#b8860b'>not measured</span></h2>
  <p>The charter's success criterion is beating GPU performance, but
  <b>no NVIDIA GPU is reachable from this environment</b> (searched 2026-09-30):
  IRD offers no GPU architecture (<code>grayskull</code> / <code>wormhole</code> /
  <code>wormhole_b0</code> / <code>blackhole</code> / <code>compute</code> only),
  all 216 inventory machines report arch <code>blackhole</code>,
  <code>wormhole_b0</code> or <code>compute</code>, and every one of the 25
  ssh-reachable bare-metal hosts reports <b>0 NVIDIA PCI devices</b>. No cloud-GPU
  CLI or credential is present either.</p>
  <p><b>No number is shown rather than an estimated one.</b> The harness is
  committed and verified against this repo's camera math and PLY activations
  (bit-identical) &mdash; see
  <code>bench/gpu_reference/run_gpu_bench.py</code>. Run it on any CUDA host and
  this section fills itself in:</p>
  <p><code>python bench/gpu_reference/run_gpu_bench.py --backend gsplat --repeats 2
  &amp;&amp; python3 opt/build_report.py</code></p>
  <p>Unblocking needs a human to supply a GPU host or a cloud-GPU credential.
  Detail and the full search log: {doc_link}.</p>
  <table class='kv'>
    <tr><th>TT anchor for the eventual ratio</th>
        <td>{TT_ANCHOR_MS} ms/view ({1000.0 / TT_ANCHOR_MS:.2f} FPS) &mdash; {TT_ANCHOR_LABEL}</td></tr>
  </table>
</section>
"""

    r = json.loads(GPU_RESULT_JSON.read_text())
    gpu = r.get("gpu", {})
    gpu_name = gpu.get("name", "unknown GPU")
    avg = float(r["avg_frame_ms"])
    ratio = TT_ANCHOR_MS / avg
    verdict = (
        f"<span style='color:#2a9d8f;font-weight:600'>TT is {1 / ratio:.2f}&times; "
        f"faster</span>"
        if avg > TT_ANCHOR_MS
        else f"<span style='color:#e76f51;font-weight:600'>GPU is {ratio:.2f}&times; "
        f"faster</span>"
    )
    hero = ""
    hero_png = r.get("hero_png")
    if hero_png and (OPT_DIR.parent / hero_png).exists():
        rel = Path(hero_png).name
        hero = (
            f"<div style='float:right;margin-left:16px;text-align:center;"
            f"font-size:11px;color:#777'>"
            f"<a href='cpu-vs-tt/{rel}' target='_blank'>"
            f"<img src='cpu-vs-tt/{rel}' style='max-width:220px;border-radius:4px;"
            f"border:1px solid #ddd'></a><br>GPU hero render</div>"
        )
    return f"""
<section style='background:#f1faee;border-left:4px solid #2a9d8f;padding:12px 16px;overflow:hidden'>
  {hero}
  <h2 style='margin-top:0'>GPU reference &mdash; measured</h2>
  <table class='kv'>
    <tr><th>GPU</th><td>{gpu_name} &middot; {gpu.get('vram_gb', '?')} GB &middot;
        sm{gpu.get('sm', '?')} &middot; host {gpu.get('host', '?')}</td></tr>
    <tr><th>Rasterizer</th><td>{r.get('backend')} {r.get('backend_version', '')}</td></tr>
    <tr><th>Bench</th><td>{r.get('scene')} &middot; {r.get('n_gaussians', 0):,} gaussians
        &middot; {r['image_size'][0]}&times;{r['image_size'][1]}
        &middot; {r.get('n_timed_views')} timed views (hero warmup excluded)</td></tr>
    <tr><th>GPU frame</th><td><b>{avg} ms/view</b> avg &middot;
        p50 {r.get('p50_frame_ms')} &middot; min {r.get('min_frame_ms')} &middot;
        max {r.get('max_frame_ms')} &middot; {r.get('fps_from_avg')} FPS</td></tr>
    <tr><th>TT frame</th><td>{TT_ANCHOR_MS} ms/view avg &mdash; {TT_ANCHOR_LABEL}</td></tr>
    <tr><th>TT vs GPU</th><td>{verdict}</td></tr>
    <tr><th>GPU hero PSNR</th><td>{r.get('hero_psnr_db')} dB vs
        <code>benchmarks/reference_v2/hero.png</code></td></tr>
    <tr><th>Measured</th><td>{r.get('timestamp', '')}</td></tr>
  </table>
  <p>Raw per-view timings: <code>opt/cpu-vs-tt/gpu_result.json</code>.
  Bench-identity notes and caveats: {doc_link}.</p>
</section>
"""


def algorithm_snapshot(rows: list[dict]) -> str:
    return """
<section>
  <h2>Algorithm snapshot</h2>
  <ul>
    <li><b>cpu</b> (numpy): per-tile-per-pixel <code>alpha_blend</code>. Algorithm
        spec, slow (~45 s / 30 views). Bit-truth.</li>
    <li><b>Absolute GT</b>: any backend with <code>cull_disabled=True</code> &mdash;
        alpha-blend over every Gaussian, no culling at all. Use to validate any
        culled render.</li>
    <li><b>cpu_cpp_mb</b> (production reference): C++ pybind extension. Diagonal AABB
        from k=√(2·ln(ω·16384)) with k_cap=3; Mahalanobis cull per-pair and
        per-microblock. <b>max_radius</b> default <code>0</code> = min(H,W)/2 cap.
        Matches numpy <code>cpu</code> at 72 dB (≤1 LSB float32 noise) and matches
        absolute GT at 72.7 dB on hero. Runs in ~4 s / 30 views.</li>
    <li><b>tt</b> (target): same C++ pipeline as cpu_cpp_mb with one stage at a
        time swapped to a TT-metal kernel (plan-amendment-002). PSNR-gated:
        <code>tt</code> hero PSNR &ge; <code>cpu_cpp_mb</code> hero &minus; 0.5 dB.</li>
    <li><b>contrib_floor</b> = 1/255 with the per-pixel blend floor (iter-156,
        <code>BLEND_PIXEL_FLOOR</code> default on). The old 1/16384 golden is archived
        under <code>tests/fixtures/hero/archive/</code>; PSNR vs old golden 42-43 dB;
        independent review task #43 PASS, no seams.</li>
    <li><b>Reference views</b>: <code>benchmarks/reference_v2/</code> @ 1024×1024,
        regenerated 2026-05-28 with current code (cpu_cpp_mb @ cf=1/16384). The
        prior cad8f91 snapshot is preserved at
        <code>benchmarks/reference_v2.cad8f91.bak/</code> but scored only 30.6 dB
        vs absolute GT, so cherrypicks <code>a4da48a</code>+<code>2e7ad9a</code>
        replaced it.</li>
  </ul>
</section>
"""


def build_html(rows: list[dict]) -> str:
    figs_html = ""
    merged = merged_iter_series()
    if merged:
        figs_html = f"""
<section>
  <img src='{fig_combined(merged)}' style='width:100%;max-width:1300px'>
</section>
"""
    css = """
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 1200px; margin: 24px auto; padding: 0 16px; color: #1d3557; }
  h1 { font-size: 22px; border-bottom: 2px solid #264653; padding-bottom: 8px; }
  h2 { font-size: 17px; margin-top: 28px; color: #264653; }
  h3 { font-size: 14px; margin: 6px 0 2px; }
  table { border-collapse: collapse; width: 100%; font-size: 13px; }
  table.kv th { text-align: right; padding: 4px 10px; color: #777; font-weight: 500; }
  table.kv td { padding: 4px 10px; }
  table.ledger th { background: #f1faee; padding: 6px 8px; text-align: left; border-bottom: 1px solid #ddd; font-weight: 600; }
  table.ledger td { padding: 5px 8px; border-bottom: 1px solid #eee; }
  table.ledger tr:hover { background: #f8f8f8; }
  .escalations { background: #ffe5e0; border-left: 4px solid #e76f51; padding: 8px 12px; margin-bottom: 12px; }
  a img.thumb, a img { cursor: zoom-in; }
  .thumb { height: 100px; border-radius: 4px; }
  .ledger-thumb { height: 56px; border-radius: 3px; margin-right: 4px; vertical-align: middle; border: 1px solid #ddd; }
  .backburner-row { display: flex; gap: 16px; padding: 10px 0; border-bottom: 1px solid #eee; }
  .backburner-meta { flex: 1; }
  .backburner-thumbs a { display: inline-block; margin-right: 6px; }
  .backburner-thumbs img { height: 100px; border-radius: 4px; }
  .reason { color: #555; font-size: 12px; }
  .idea-desc { color: #1d3557; font-size: 13px; font-weight: 500; margin: 4px 0 2px; }
  .iter-build { color: #999; font-size: 11px; margin: 2px 0 0; }
  .caveat { color: #b8860b; font-size: 11px; font-style: italic; margin: 4px 0; }
  .iter-pos { display: inline-block; min-width: 48px; padding: 1px 6px; margin-right: 6px; background: #1d3557; color: #fff; font-size: 11px; font-weight: 700; border-radius: 3px; letter-spacing: 0.4px; }
  .iter-ts { color: #777; font-size: 11px; margin: 0 0 4px; font-variant-numeric: tabular-nums; }
  .report-meta { color: #555; font-size: 12px; margin: 0 0 16px; }
  .in-flight-banner { background: #fff8e6; border: 1px solid #e9c46a; border-radius: 6px; padding: 12px 16px; margin-bottom: 20px; }
  .in-flight-row { background: #fffdf5; border-left: 4px solid #e9c46a; padding-left: 12px; }
  .reason { color: #555; font-size: 12px; }
  code { background: #f1faee; padding: 1px 4px; border-radius: 3px; font-size: 11px; }
  .shot-missing { background: #fde2e1; border: 2px solid #c44536; color: #c44536; font-weight: 700; font-size: 12px; padding: 10px 12px; border-radius: 4px; max-width: 260px; }
  .shot-missing small { font-weight: 400; }
  .shot-cap { color: #555; font-size: 11px; margin: 2px 0 0; }
  .shot-meta { color: #555; font-size: 11px; margin: 4px 0 0; }
  .golden-badge { display: inline-block; font-size: 11px; margin: 4px 0 0; padding: 1px 6px; border-radius: 3px; }
  .golden-ok { background: #e3f4e1; color: #2d6a4f; border: 1px solid #95d5b2; }
  .golden-diff { background: #fff3cd; color: #8a6d00; border: 1px solid #e9c46a; }
  #lightbox { display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.85); z-index: 1000; cursor: zoom-out; align-items: center; justify-content: center; }
  #lightbox img { max-width: 96vw; max-height: 96vh; }
  #viewer { display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.9); z-index: 1001; flex-direction: column; align-items: center; justify-content: center; color: #eee; font-size: 13px; }
  #viewer .vw-bar { display: flex; gap: 14px; align-items: center; margin-bottom: 8px; flex-wrap: wrap; justify-content: center; }
  #viewer .vw-bar label { cursor: pointer; }
  #viewer .vw-bar code { background: #333; color: #eee; }
  #viewer button { background: #444; color: #eee; border: 1px solid #777; border-radius: 3px; padding: 2px 10px; cursor: pointer; }
  #viewer .vw-stage { position: relative; display: inline-block; line-height: 0; user-select: none; touch-action: none; }
  #viewer .vw-stage img { max-width: 94vw; max-height: 84vh; display: block; }
  #viewer .vw-stage img.vw-over { position: absolute; left: 0; top: 0; width: 100%; height: 100%; }
  #viewer .vw-line { position: absolute; top: 0; bottom: 0; width: 2px; margin-left: -1px; background: #fff; box-shadow: 0 0 3px #000; cursor: ew-resize; }
  #viewer .vw-line::after { content: ''; position: absolute; top: 50%; left: -9px; width: 16px; height: 16px; margin-top: -9px; border: 2px solid #fff; border-radius: 50%; background: rgba(0,0,0,0.5); }
  #viewer .vw-hint { color: #aaa; font-size: 11px; margin-top: 6px; }
</style>
<script>
// Compare viewer for device screenshots: ref / candidate / diff / split.
// Self-contained (no external scripts or CSS). Image paths come from the
// clicked link's data-img-* attributes, already relative to this file.
(function () {
  var V = null, pos = 50, dragging = false, dragged = false;
  function q(sel) { return V.querySelector(sel); }
  function setPos(p) {
    pos = Math.max(0, Math.min(100, p));
    // ref on the left of the line, candidate on the right.
    q('.vw-over').style.clipPath = 'inset(0 ' + (100 - pos) + '% 0 0)';
    q('.vw-line').style.left = pos + '%';
  }
  function setMode(m) {
    var r = q('input[value="' + m + '"]');
    if (r) r.checked = true;
    var split = m === 'split';
    q('.vw-base').src = V.dataset[split ? 'cand' : m];
    q('.vw-over').style.display = split ? 'block' : 'none';
    q('.vw-line').style.display = split ? 'block' : 'none';
    if (split) setPos(pos);
  }
  function open(a) {
    V = document.getElementById('viewer');
    V.dataset.ref = a.getAttribute('data-img-ref');
    V.dataset.cand = a.getAttribute('data-img-cand');
    V.dataset.diff = a.getAttribute('data-img-diff');
    q('.vw-over').src = V.dataset.ref;
    q('.vw-ref-name').textContent = a.getAttribute('data-viewer-label') || V.dataset.ref;
    V.style.display = 'flex';
    setMode(a.getAttribute('data-viewer-mode') || 'cand');
  }
  function close() { if (V) V.style.display = 'none'; }
  function fromEvent(e) {
    var r = q('.vw-stage').getBoundingClientRect();
    setPos((e.clientX - r.left) / r.width * 100);
  }
  document.addEventListener('click', function (e) {
    var lb = document.getElementById('lightbox');
    if (e.target.closest('#lightbox')) { lb.style.display = 'none'; return; }
    if (dragged) { dragged = false; return; }  // a split drag that ended off the image
    if (e.target.closest('.vw-close') || e.target.id === 'viewer') { close(); return; }
    var a = e.target.closest('a.zoom');
    if (!a) return;
    e.preventDefault();
    if (a.hasAttribute('data-img-ref')) { open(a); return; }
    if (!lb) return;
    lb.querySelector('img').src = a.getAttribute('href');
    lb.style.display = 'flex';
  });
  document.addEventListener('change', function (e) {
    if (e.target.name === 'vw-mode') setMode(e.target.value);
  });
  document.addEventListener('pointerdown', function (e) {
    if (!V || V.style.display !== 'flex' || !q('input[value="split"]').checked) return;
    if (!e.target.closest('.vw-stage')) return;
    dragging = true; e.preventDefault(); fromEvent(e);
  });
  document.addEventListener('pointermove', function (e) { if (dragging) fromEvent(e); });
  document.addEventListener('pointerup', function () { dragged = dragging; dragging = false; });
  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape') {
      var lb = document.getElementById('lightbox'); if (lb) lb.style.display = 'none';
      close(); return;
    }
    if (!V || V.style.display !== 'flex') return;
    var modes = {'1': 'ref', '2': 'cand', '3': 'diff', '4': 'split'};
    if (modes[e.key]) setMode(modes[e.key]);
    else if (e.key === 'ArrowLeft') { setMode('split'); setPos(pos - 5); }
    else if (e.key === 'ArrowRight') { setMode('split'); setPos(pos + 5); }
  });
})();
</script>
"""
    n_metal = len(load_metal_iters())
    n_ttw = len(load_ttw_iters())
    n_cpu = len(rows)
    inflight = load_current_iter()
    inflight_note = ""
    if inflight and inflight.get("status") not in ("done", "complete", "finished"):
        inflight_note = f" · in-flight: iter {inflight.get('iter', '?')} ({inflight.get('status', '?')})"
    meta = (
        f"<p class='report-meta'>Generated {now_ts_minutes()} · "
        f"{n_metal} metal + {n_ttw} ttw + {n_cpu} cpu ledger rows{inflight_note}</p>"
    )
    return f"""<!DOCTYPE html>
<html lang='en'>
<head><meta charset='utf-8'><title>gstt2 — Optimization Report</title>{css}</head>
<body>
<h1>gstt2 — Optimization Report</h1>
{meta}
{in_flight_section()}
{figs_html}
{throughput_section()}
{published_gpu_section()}
{gpu_reference_section()}
{ledger_section(rows)}
{algorithm_snapshot(rows)}
<div id='lightbox'><img src='' alt='enlarged screenshot'></div>
<div id='viewer'>
  <div class='vw-bar'>
    <label><input type='radio' name='vw-mode' value='ref'> ref</label>
    <label><input type='radio' name='vw-mode' value='cand'> candidate</label>
    <label><input type='radio' name='vw-mode' value='diff'> diff</label>
    <label><input type='radio' name='vw-mode' value='split'> split</label>
    <span>ref: <code class='vw-ref-name'></code></span>
    <button type='button' class='vw-close'>close (Esc)</button>
  </div>
  <div class='vw-stage'><img class='vw-base' alt='viewer image'><img class='vw-over' alt='reference'><div class='vw-line'></div></div>
  <div class='vw-hint'>split: ref left, candidate right; drag the line (all the way left = whole candidate). Keys 1-4 switch, arrows move the line.</div>
</div>
</body>
</html>
"""


def write_reports(html: str) -> None:
    # No trailing whitespace (empty template slots leave indented blank lines):
    # `ttp push` runs `git diff --check` and refuses them.
    html = "\n".join(line.rstrip() for line in html.split("\n"))
    REPORT_HTML.write_text(html)
    REPORT_HTML_TTW.parent.mkdir(parents=True, exist_ok=True)
    REPORT_HTML_TTW.write_text(rebase_for_ttw(html))


_REL_ATTR_RE = re.compile(
    r"""(\b(?:src|href|data-img-ref|data-img-cand|data-img-diff)=(["']))(?![a-zA-Z][a-zA-Z0-9+.-]*:|/|#)([^"']+\2)""")


def rebase_for_ttw(html: str) -> str:
    """Prefix relative src/href/data-img-* paths with ../ so the opt/ttw/ mirror resolves them."""
    return _REL_ATTR_RE.sub(r"\1../\3", html)


def main() -> None:
    rows = load_iters()
    html = build_html(rows)
    write_reports(html)
    n_ledger = len(load_metal_iters()) + len(load_ttw_iters())
    print(f"wrote {REPORT_HTML}  ({n_ledger} ledger rows, {len(rows)} cpu iters)")
    print(f"wrote {REPORT_HTML_TTW}  (mirror, relative paths rebased to ../)")
    errors = check_device_screenshots(load_ttw_iters(), pixels=True)
    if errors:
        for e in errors:
            print(f"SCREENSHOT MISSING: {e}")
        print("REPORT INVALID: device screenshot requirement not met "
              "(see opt/ttw/ITERATION_CHECKLIST.md)")
        sys.exit(2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Regenerate opt/REPORT.html (+ ttw mirror).")
    ap.add_argument(
        "--set-in-flight",
        metavar="JSON",
        help="Write opt/current-iter.json then regenerate reports.",
    )
    ap.add_argument(
        "--clear-in-flight",
        action="store_true",
        help="Remove opt/current-iter.json then regenerate reports.",
    )
    args = ap.parse_args()
    if args.set_in_flight:
        write_current_iter(json.loads(args.set_in_flight))
        main()
    elif args.clear_in_flight:
        clear_current_iter()
        main()
    else:
        main()
