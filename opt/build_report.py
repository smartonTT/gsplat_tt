"""Regenerate opt/REPORT.html from opt/iters.jsonl.

Single self-contained HTML. Sections:
  1. Status banner       (current iter, best, target, escalations)
  2. Backburner          (REJECT/NEEDS_REVIEW iters with reasons + thumbnails)
  3. Profile line graphs (sum-ms over iters; per-stage breakdown; PSNR floor)
  4. Per-iter ledger     (sortable HTML table)
  5. Algorithm snapshot  (current pipeline state, brief)

The supervisor's decide_and_log.py invokes this after every iter.
"""
from __future__ import annotations

import base64
import io
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


OPT_DIR = Path(__file__).resolve().parent
ITERS_JSONL = OPT_DIR / "iters.jsonl"
METAL_ITERS_JSONL = OPT_DIR / "metal-iters.jsonl"
METAL_SCREENSHOTS_DIR = OPT_DIR / "metal-screenshots"
REPORT_HTML = OPT_DIR / "REPORT.html"
SCREENSHOTS_DIR = OPT_DIR / "screenshots"
TARGET_SUM_MS = 1000.0
STAGE_KEYS = ("project_ms", "tile_assign_ms", "sort_ms", "blend_ms")
STAGE_TIMING_KEYS = ("project", "tile_assign", "sort", "blend")
REF_DIR = OPT_DIR.parent / "benchmarks" / "reference_v2"


def img_link(src: str, cls: str = "thumb") -> str:
    return f'<a href="{src}" target="_blank" rel="noopener"><img src="{src}" class="{cls}" alt=""></a>'


def enrich_stage_medians(row: dict) -> dict:
    """Fill per_stage_median_ms from timing.jsonl when the jsonl row lacks it."""
    stages = dict(row.get("per_stage_median_ms") or {})
    if stages:
        return stages
    timing_path = SCREENSHOTS_DIR / row.get("iter_dir", "") / "timing.jsonl"
    if not timing_path.exists():
        return stages
    timing_rows = [
        json.loads(line)
        for line in timing_path.read_text().splitlines()
        if line.strip()
    ]
    per_stage: dict[str, list[float]] = {}
    for tr in timing_rows:
        for src_key, dst_key in zip(STAGE_TIMING_KEYS, STAGE_KEYS):
            if src_key in tr:
                per_stage.setdefault(dst_key, []).append(float(tr[src_key]))
    return {k: statistics.median(v) for k, v in per_stage.items()}


def rows_with_stages(rows: list[dict]) -> list[dict]:
    return [{**r, "per_stage_median_ms": enrich_stage_medians(r)} for r in rows]


def ensure_hero_diff10(iter_dir: str) -> None:
    """Write hero_diff10.png as 10× per-channel abs color diff vs reference."""
    if not iter_dir:
        return
    iter_path = SCREENSHOTS_DIR / iter_dir
    hero = iter_path / "hero.png"
    ref = REF_DIR / "hero.png"
    if not hero.exists() or not ref.exists():
        return
    import numpy as np
    from PIL import Image

    ref_rgb = np.asarray(Image.open(ref).convert("RGB"), dtype=np.float64) / 255.0
    cand_rgb = np.asarray(Image.open(hero).convert("RGB"), dtype=np.float64) / 255.0
    if ref_rgb.shape != cand_rgb.shape:
        return
    amp = np.clip(np.abs(ref_rgb - cand_rgb) * 10.0, 0.0, 1.0)
    Image.fromarray((amp * 255.0).astype(np.uint8)).save(iter_path / "hero_diff10.png")


def load_metal_iters() -> list[dict]:
    if not METAL_ITERS_JSONL.exists():
        return []
    rows = []
    for line in METAL_ITERS_JSONL.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def metal_section(rows: list[dict]) -> str:
    if not rows:
        return """
<section>
  <h2>Metal port (Phase 5)</h2>
  <p>Target: bh-30 P150 Blackhole, 1 ms/frame. See
  <a href='plan-amendment-001-metal-port.md'>plan-amendment-001-metal-port.md</a>.
  No metal iters logged yet.</p>
</section>
"""
    head = "<tr><th>iter</th><th>verdict</th><th>action</th><th>sum_ms</th><th>min_psnr</th><th>class</th></tr>"
    body = ""
    for r in reversed(rows):
        psnr_d = r.get("psnr_per_view") or {}
        finite = [v for v in psnr_d.values() if isinstance(v, (int, float)) and v != float("inf") and v == v]
        hero_psnr = r.get("hero_psnr_dB")
        if hero_psnr is not None and hero_psnr == hero_psnr:
            psnr_min_str = f"{hero_psnr:.1f}"
        elif finite:
            psnr_min_str = f"{min(finite):.1f}"
        else:
            psnr_min_str = "—"
        sum_ms = r.get("sum_total_ms")
        sum_ms_str = f"{sum_ms:.1f}" if isinstance(sum_ms, (int, float)) and sum_ms == sum_ms else "—"
        body += (
            f"<tr><td><a href='metal-screenshots/{r['iter_dir']}/'>{r['iter_dir']}</a></td>"
            f"<td>{r.get('verdict','')}</td><td>{r.get('action','')}</td>"
            f"<td>{sum_ms_str}</td><td>{psnr_min_str}</td>"
            f"<td>{r.get('class','')}</td></tr>"
        )
    return f"<section><h2>Metal port (Phase 5)</h2><table class='ledger'>{head}{body}</table></section>"


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


def fig_sum_ms(rows: list[dict]) -> str:
    fig, ax = plt.subplots(figsize=(10, 3.5))
    xs = list(range(len(rows)))
    ys = [max(r.get("sum_total_ms", float("nan")), 1e-3) for r in rows]
    colors = ["#2a9d8f" if r.get("action") == "commit" else "#e76f51" for r in rows]
    ax.set_yscale("log")
    ax.plot(xs, ys, color="#264653", linewidth=1.5, zorder=1)
    ax.scatter(xs, ys, c=colors, s=46, zorder=2, edgecolors="white", linewidths=1.2)
    ax.axhline(TARGET_SUM_MS, color="#e9c46a", linestyle="--", linewidth=1.0, label=f"target {TARGET_SUM_MS:.0f} ms")
    for i, r in enumerate(rows):
        ax.annotate(r.get("iter_dir", "")[:12], (xs[i], ys[i]), fontsize=6, alpha=0.65,
                    xytext=(0, 6), textcoords="offset points", ha="center")
    ax.set_xlabel("iter index (oldest → newest)")
    ax.set_ylabel("sum-of-30 ms (log scale)")
    ax.set_title("30-frame benchmark total (lower is better)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.25, which="both")
    return plot_b64(fig)


def fig_psnr(rows: list[dict]) -> str:
    fig, ax = plt.subplots(figsize=(10, 3.5))
    xs = list(range(len(rows)))
    psnr_mins = []
    for r in rows:
        psnr_d = r.get("psnr_per_view") or {}
        finite = [v for v in psnr_d.values() if isinstance(v, (int, float)) and v != float("inf") and v == v]
        psnr_mins.append(min(finite) if finite else None)
    valid_xs = [x for x, y in zip(xs, psnr_mins) if y is not None]
    valid_ys = [y for y in psnr_mins if y is not None]
    if valid_ys:
        ax.plot(valid_xs, valid_ys, color="#1d3557", marker="o", markersize=4, linewidth=1.2)
    ax.axhline(PSNR_FLOOR, color="#e9c46a", linestyle="--", linewidth=1.0, label=f"floor {PSNR_FLOOR:.0f} dB")
    ax.set_xlabel("iter index")
    ax.set_ylabel("min PSNR across views (dB)")
    ax.set_title("Visual fidelity floor (higher is better; ∞ = bit-identical)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.25)
    return plot_b64(fig)


def fig_stages(rows: list[dict]) -> str:
    """Line plot of per-stage median ms/frame across iter history."""
    enriched = rows_with_stages(rows)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    xs = list(range(len(enriched)))
    plotted = False
    for k in STAGE_KEYS:
        ys = [(r.get("per_stage_median_ms") or {}).get(k) for r in enriched]
        xs_plot = [x for x, y in zip(xs, ys) if y is not None]
        ys_plot = [y for y in ys if y is not None]
        if ys_plot:
            ax.plot(xs_plot, ys_plot, marker=".", label=k.replace("_ms", ""), linewidth=1.2)
            plotted = True
    if not plotted:
        ax.text(0.5, 0.5, "no per-stage timing data", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("iter index (oldest → newest)")
    ax.set_ylabel("median ms / frame")
    ax.set_title("Per-stage breakdown")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.25)
    if plotted:
        ax.set_ylim(bottom=0)
    return plot_b64(fig)


def status_section(rows: list[dict]) -> str:
    if not rows:
        return '<section><h2>Status</h2><p>No iters yet. Phase 0 just landed.</p></section>'
    committed = [r for r in rows if r.get("action") == "commit"]
    best = min((r["sum_total_ms"] for r in committed), default=float("inf"))
    best_str = f"{best:.1f} ms" if best != float("inf") else "n/a"
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
    <tr><th>Best sum-of-30 (committed)</th><td>{best_str}</td></tr>
    <tr><th>Target</th><td>&lt; {TARGET_SUM_MS:.0f} ms</td></tr>
    <tr><th>Iters so far</th><td>{len(rows)} ({sum(1 for r in rows if r.get('action')=='commit')} committed)</td></tr>
  </table>
</section>
"""


def backburner_section(rows: list[dict]) -> str:
    parked = [r for r in rows if r.get("action") == "backburner"]
    if not parked:
        return ''
    items = []
    for r in parked:
        priority = "⭐" if r.get("high_promotion_priority") else ""
        verdict = r.get("verdict", "")
        psnr_d = r.get("psnr_per_view") or {}
        finite = [v for v in psnr_d.values() if isinstance(v, (int, float)) and v != float("inf") and v == v]
        psnr_min_str = f"{min(finite):.1f} dB" if finite else "n/a"
        thumb_dir = SCREENSHOTS_DIR / r.get("iter_dir", "")
        hero = thumb_dir / "hero.png"
        diff = thumb_dir / "hero_diff10.png"
        ensure_hero_diff10(r.get("iter_dir", ""))
        hero_src = f"screenshots/{r['iter_dir']}/hero.png"
        diff_src = f"screenshots/{r['iter_dir']}/hero_diff10.png"
        thumb_html = ""
        if hero.exists():
            thumb_html = img_link(hero_src)
        if diff.exists():
            thumb_html += img_link(diff_src)
        items.append(f"""
<div class='backburner-row'>
  <div class='backburner-meta'>
    <h3>{priority} {r['iter_dir']} — {verdict}</h3>
    <p>sum_ms={r.get('sum_total_ms', 0):.1f}, psnr_min={psnr_min_str}</p>
    <p class='reason'>{r.get('validator_reasoning', '')}</p>
  </div>
  <div class='backburner-thumbs'>{thumb_html}</div>
</div>
""")
    return f"<section><h2>Backburner ({len(parked)})</h2>{''.join(reversed(items))}</section>"


def ledger_section(rows: list[dict]) -> str:
    head = "<tr><th>iter</th><th>verdict</th><th>action</th><th>sum_ms</th><th>min_psnr</th><th>class</th><th>commit</th></tr>"
    body = ""
    for r in reversed(rows):
        psnr_d = r.get("psnr_per_view") or {}
        finite = [v for v in psnr_d.values() if isinstance(v, (int, float)) and v != float("inf") and v == v]
        psnr_min_str = f"{min(finite):.1f}" if finite else "—"
        sha = (r.get("commit_sha") or "")[:8]
        body += (
            f"<tr><td><a href='screenshots/{r['iter_dir']}/'>{r['iter_dir']}</a></td>"
            f"<td>{r.get('verdict','')}</td><td>{r.get('action','')}</td>"
            f"<td>{r.get('sum_total_ms',0):.1f}</td><td>{psnr_min_str}</td>"
            f"<td>{r.get('class','')}</td><td><code>{sha}</code></td></tr>"
        )
    return f"<section><h2>Ledger</h2><table class='ledger'>{head}{body}</table></section>"


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
  <code>contrib_floor=1/16384</code> (~68.6 dB worst vs true GT; ~18 ms/view).</p>
  <table class='kv'>{head}</table>
  {table}
  <p>Full log: <code>opt/cull_tune.jsonl</code></p>
</section>
"""


def algorithm_snapshot(rows: list[dict]) -> str:
    return """
<section>
  <h2>Algorithm snapshot</h2>
  <p>See <a href='plan.md'>plan.md</a> for the frozen plan and
  <a href='microblock-cpu-spec.md'>microblock-cpu-spec.md</a> for the
  microblock binning contract.</p>
  <ul>
    <li><b>cpu-ref</b> (numpy): existing per-tile-per-pixel <code>alpha_blend</code>.
        Algorithm spec, slow. Bit-truth.</li>
    <li><b>Ground truth</b>: numpy alpha_blend, all quality culls off
        (min_opacity=0, max_radius=-1, no Mahalanobis/microblock cull).</li>
    <li><b>cpu_cpp default</b>: diagonal AABB from k=sqrt(2·ln(ω/floor)) with
        floor=1/16384, k_cap=3, min k=3; Mahalanobis cull on tile assign + blend.</li>
    <li><b>contrib_floor</b> = 1/16384 (68.6 dB worst vs true GT; max_abs 0.0038).</li>
    <li><b>Reference views</b>: <code>benchmarks/reference_v2/</code> @ 1024×1024,
        orbit sweep + close-zoom.</li>
    <li><b>Metal</b>: tt-metal on bh-30 (P150); see plan-amendment-001-metal-port.md.</li>
  </ul>
</section>
"""


def build_html(rows: list[dict]) -> str:
    figs_html = ""
    if rows:
        figs_html = f"""
<section>
  <h2>Profile</h2>
  <img src='{fig_sum_ms(rows)}' style='width:100%;max-width:1100px'>
  <img src='{fig_psnr(rows)}' style='width:100%;max-width:1100px'>
  <img src='{fig_stages(rows)}' style='width:100%;max-width:1100px'>
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
  .backburner-row { display: flex; gap: 16px; padding: 10px 0; border-bottom: 1px solid #eee; }
  .backburner-meta { flex: 1; }
  .backburner-thumbs a { display: inline-block; margin-right: 6px; }
  .backburner-thumbs img { height: 100px; border-radius: 4px; }
  a img.thumb, a img { cursor: zoom-in; }
  .thumb { height: 100px; border-radius: 4px; }
  .reason { color: #555; font-size: 12px; }
  code { background: #f1faee; padding: 1px 4px; border-radius: 3px; font-size: 11px; }
</style>
"""
    return f"""<!DOCTYPE html>
<html lang='en'>
<head><meta charset='utf-8'><title>gsplat_tt_2 — Optimization Report</title>{css}</head>
<body>
<h1>gsplat_tt_3 — Optimization Report</h1>
<p>Local-Mac CPU-first sprint. Target: 30-frame sum-of-ms below {TARGET_SUM_MS:.0f} ms at PSNR ≥ {PSNR_FLOOR:.0f} dB vs true GT (all culls off).</p>
{status_section(rows)}
{cull_tune_section()}
{figs_html}
{backburner_section(rows)}
{ledger_section(rows)}
{metal_section(load_metal_iters())}
{algorithm_snapshot(rows)}
</body>
</html>
"""


def main():
    rows = load_iters()
    REPORT_HTML.write_text(build_html(rows))
    print(f"wrote {REPORT_HTML}  ({len(rows)} iters)")


if __name__ == "__main__":
    main()
