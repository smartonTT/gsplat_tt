"""Builds REPORT.html and graph PNGs from iters.jsonl + per-iter screenshot dirs.

Run from the repo root or anywhere:
    python docs/optimization-log/build_report.py

Idempotent; safe to re-run after every iter.
"""
from __future__ import annotations

import json
from collections import defaultdict
from html import escape
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs" / "optimization-log"
GRAPHS = OUT / "graphs"
SHOTS = OUT / "screenshots"
GRAPHS.mkdir(parents=True, exist_ok=True)

VERDICT_COLOR = {
    "KEEP": "#2ca02c",
    "NEEDS_REVIEW": "#ff7f0e",
    "REJECT": "#d62728",
}
ACTION_COLOR = {
    "commit": "#2ca02c",
    "no_commit_valid_but_not_faster": "#7f7f7f",
    "backburner": "#d62728",
}


def load_iters() -> list[dict]:
    p = OUT / "iters.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def _psnr_min(r: dict) -> float | None:
    psnr = r.get("psnr_per_view") or {}
    vals = [v for v in psnr.values() if v is not None and v != float("inf")]
    return min(vals) if vals else None


def graph_combined(iters: list[dict]) -> None:
    """Trajectory graph: every iter (not just at-or-better), with per-stage timings.

    Top subplot — log y: kernel_ms, total_ms (end-to-end host wall), blend_ms
    (host-side blend including daemon IPC), with PSNR_min on the right axis.
    Bottom subplot — linear y: prep_ms (project + tile_assign) and sort_ms,
    the host-side stages that flank the device kernel.

    Stage timings (total/blend/prep/sort) come from
    metrics.json["stage_medians"]; iters that don't have them just plot
    NaN for those series. Dot color encodes the action verdict.
    """
    nan = float("nan")
    kept = [r for r in iters if r.get("kernel_ms_median") is not None]
    if not kept:
        return

    labels = [r["iter_dir"] for r in kept]
    actions = [r.get("action", "unknown") for r in kept]
    medians = [r.get("kernel_ms_median", nan) for r in kept]
    psnr_min = [(_psnr_min(r) if _psnr_min(r) is not None else nan) for r in kept]

    def stage(key):
        return [r.get("stage_medians", {}).get(key, nan) for r in kept]

    total_ms = stage("total_ms")
    blend_ms = stage("blend_ms")
    project_ms = stage("project_ms")
    tile_assign_ms = stage("tile_assign_ms")
    sort_ms = stage("sort_ms")
    prep_ms = [
        (p + t) if (p == p and t == t) else nan  # NaN-safe sum
        for p, t in zip(project_ms, tile_assign_ms)
    ]

    xs = list(range(len(labels)))
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(14, 9), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )

    # Top: kernel + total + blend (log) with PSNR on twin axis.
    ax_top.plot(xs, total_ms, "-", color="#7f7f7f", linewidth=1.2, label="total (host wall)")
    ax_top.plot(xs, blend_ms, "-", color="#ff7f0e", linewidth=1.2, label="blend (host)")
    ax_top.plot(xs, medians, "-", color="#1f77b4", linewidth=1.8, label="kernel median", zorder=3)
    dot_colors = [ACTION_COLOR.get(a, "#999999") for a in actions]
    ax_top.scatter(xs, medians, c=dot_colors, s=44, zorder=4, edgecolors="white", linewidths=0.7)
    ax_top.axhline(1.0, color="#000000", linestyle=":", linewidth=1, label="target 1.0 ms")
    ax_top.set_ylabel("ms (log)")
    ax_top.set_yscale("log")
    ax_top.grid(True, alpha=0.3, which="both")
    # Force the y-range to include host-wall values (total_ms can exceed 300 ms).
    # Autoscale alone undershot when only 3 iters had stage_medians; the resulting
    # ~150 ms ceiling clipped the total/blend lines off the chart entirely.
    finite_top = [v for v in (medians + total_ms + blend_ms) if v == v]
    if finite_top:
        ax_top.set_ylim(0.5, max(finite_top) * 1.5)

    ax_db = ax_top.twinx()
    ax_db.plot(xs, psnr_min, "--", color="#9467bd", linewidth=1.2, marker="s",
               markersize=4, label="PSNR min (dB)")
    ax_db.axhline(35.0, color="#9467bd", linestyle=":", linewidth=1, alpha=0.6,
                  label="PSNR gate 35 dB")
    ax_db.set_ylabel("PSNR (dB)")
    ax_db.set_ylim(0, 75)

    h1, l1 = ax_top.get_legend_handles_labels()
    h2, l2 = ax_db.get_legend_handles_labels()
    ax_top.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8, ncol=2)
    ax_top.set_title("Per-iter trajectory — kernel + host stages + PSNR (all iters)")

    # Bottom: prep + sort (linear). These are CPU pre-blend stages.
    ax_bot.plot(xs, prep_ms, "-", color="#2ca02c", linewidth=1.2, marker="o",
                markersize=3, label="prep (project + tile_assign)")
    ax_bot.plot(xs, sort_ms, "-", color="#d62728", linewidth=1.2, marker="o",
                markersize=3, label="sort")
    ax_bot.set_xlabel("iter")
    ax_bot.set_ylabel("ms")
    ax_bot.grid(True, alpha=0.3)
    ax_bot.set_xticks(xs)
    ax_bot.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax_bot.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(GRAPHS / "graph-progress.png", dpi=120)
    plt.close(fig)


def graph_class_progress(iters: list[dict]) -> None:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in iters:
        counts[r.get("class", "unknown")][r.get("action", "unknown")] += 1
    classes = list(counts.keys())
    actions = ["commit", "no_commit_valid_but_not_faster", "backburner"]
    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(len(classes))
    for action in actions:
        ys = np.array([counts[c].get(action, 0) for c in classes])
        ax.bar(classes, ys, bottom=bottom, label=action, color=ACTION_COLOR.get(action, "#999999"))
        bottom += ys
    ax.set_xlabel("optimization class")
    ax.set_ylabel("iter count")
    ax.set_title("Iters per class × action")
    ax.legend()
    fig.tight_layout()
    fig.savefig(GRAPHS / "graph-class-progress.png", dpi=110)
    plt.close(fig)


def _brief_for_iter(iter_name: str) -> dict:
    """Returns {hypothesis, outcome} for an iter.

    hypothesis: the full ## Hypothesis section text (excluding fenced code
                blocks, capped ~1200 chars).
    outcome:    validator reasoning if validator.json exists, else lesson/
                correction note from iters.jsonl if present, else "".
    Both fields fall back to "" rather than slug-derived stand-ins, so the
    UI can decide whether to render an empty paragraph.
    """
    hypothesis = ""
    md = OUT / f"{iter_name}.md"
    if md.exists():
        text = md.read_text()
        if "## Hypothesis" in text:
            after = text.split("## Hypothesis", 1)[1]
            # Stop at next "## " heading.
            chunks = after.split("\n## ", 1)
            section = chunks[0]
            # Strip fenced code blocks for the brief.
            cleaned: list[str] = []
            in_code = False
            for line in section.splitlines():
                if line.strip().startswith("```"):
                    in_code = not in_code
                    continue
                if in_code:
                    continue
                cleaned.append(line.rstrip())
            # Collapse runs of blank lines.
            collapsed: list[str] = []
            prev_blank = False
            for s in cleaned:
                if not s.strip():
                    if not prev_blank:
                        collapsed.append("")
                    prev_blank = True
                else:
                    collapsed.append(s)
                    prev_blank = False
            hypothesis = "\n".join(collapsed).strip()
            if len(hypothesis) > 1200:
                hypothesis = hypothesis[:1200].rstrip() + "…"

    outcome = ""
    val_path = OUT / "screenshots" / iter_name / "validator.json"
    if val_path.exists():
        try:
            outcome = (json.loads(val_path.read_text()).get("reasoning") or "").strip()
        except Exception:
            outcome = ""
    if not outcome:
        # Look in iters.jsonl for a lesson / correction_note attached to this iter.
        try:
            for line in (OUT / "iters.jsonl").read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("iter_dir") == iter_name:
                    note = row.get("lesson") or row.get("correction") or row.get("correction_note")
                    if note:
                        outcome = note.strip()
                        break
        except Exception:
            pass

    return {"hypothesis": hypothesis, "outcome": outcome}


def render_card(r: dict) -> str:
    iter_name = r["iter_dir"]
    shot_dir = f"screenshots/{iter_name}"
    psnr = r.get("psnr_per_view", {})
    per_view = r.get("per_view_median", {})
    psnr_min = min(psnr.values()) if psnr else float("nan")
    badge_color = ACTION_COLOR.get(r.get("action"), "#999999")
    badge = r.get("action", "unknown")
    verdict_badge = r.get("verdict", "?")
    brief = _brief_for_iter(iter_name)
    hypothesis_html = (
        "<p class=\"brief hypothesis\"><b>Hypothesis.</b> "
        + escape(brief["hypothesis"]).replace("\n", "<br>")
        + "</p>"
    ) if brief["hypothesis"] else ""
    outcome_html = (
        "<p class=\"brief outcome\"><b>Outcome.</b> "
        + escape(brief["outcome"]).replace("\n", "<br>")
        + "</p>"
    ) if brief["outcome"] else ""

    # Try to load validator JSON for the checklist render
    val_path = OUT / "screenshots" / iter_name / "validator.json"
    if val_path.exists():
        val = json.loads(val_path.read_text())
        checks_html = "".join(
            f"<li>{'✓' if c['result'] == 'pass' else '✗'} {escape(c['name'])}: {escape(c.get('evidence',''))[:120]}</li>"
            for c in val.get("visual_checks", [])
        )
        reasoning = escape(val.get("reasoning", ""))
    else:
        checks_html = "<li>(no validator.json)</li>"
        reasoning = ""

    # iter-008 and other DEVICE_FAIL/deadlock iters write null for kernel_ms_median
    # — guard the format calls so a single broken row doesn't blow up the whole report.
    def _ms(key):
        v = r.get(key)
        return f"{v:.2f}" if isinstance(v, (int, float)) and v == v else "—"

    def _pv(key):
        v = (r.get("per_view_median") or {}).get(key)
        return f"{v:.2f}" if isinstance(v, (int, float)) and v == v else "—"

    def _psnr(key):
        v = (r.get("psnr_per_view") or {}).get(key)
        return f"{v:.1f}" if isinstance(v, (int, float)) and v == v else "—"

    psnr_min_str = f"{psnr_min:.1f}" if isinstance(psnr_min, (int, float)) and psnr_min == psnr_min else "—"

    return f"""
<section class="iter-card">
  <header>
    <h3>{escape(iter_name)}
      <span class="badge" style="background:{badge_color}">{escape(badge)}</span>
      <span class="badge verdict">{escape(verdict_badge)}</span>
      <span class="class-tag">{escape(r.get('class', '?'))}</span>
      {f'<a href="{escape(r["iter_dir"])}.md">md</a>' if (OUT / f"{r['iter_dir']}.md").exists() else ''}
      <span class="ts">{escape(r.get('timestamp', ''))}</span>
    </h3>
  </header>
  {hypothesis_html}
  {outcome_html}
  <div class="metrics">
    <div class="ms">kernel: <b>{_ms('kernel_ms_median')} ms</b> median · {_ms('kernel_ms_p99')} ms p99</div>
    <div class="per-view">per-view: hero {_pv('hero')} / side {_pv('side')} / top {_pv('top')}</div>
    <div class="psnr">PSNR: hero {_psnr('hero')} / side {_psnr('side')} / top {_psnr('top')} (min <b>{psnr_min_str}</b>)</div>
  </div>
  <div class="thumbs">
    <figure><a href="{shot_dir}/hero.png" target="_blank"><img src="{shot_dir}/hero.png" alt="hero"></a><figcaption>hero</figcaption></figure>
    <figure><a href="{shot_dir}/side.png" target="_blank"><img src="{shot_dir}/side.png" alt="side"></a><figcaption>side</figcaption></figure>
    <figure><a href="{shot_dir}/top.png" target="_blank"><img src="{shot_dir}/top.png" alt="top"></a><figcaption>top</figcaption></figure>
    <figure><a href="{shot_dir}/hero_diff10.png" target="_blank"><img src="{shot_dir}/hero_diff10.png" alt="hero diff10"></a><figcaption>hero × 10</figcaption></figure>
    <figure><a href="{shot_dir}/side_diff10.png" target="_blank"><img src="{shot_dir}/side_diff10.png" alt="side diff10"></a><figcaption>side × 10</figcaption></figure>
    <figure><a href="{shot_dir}/top_diff10.png" target="_blank"><img src="{shot_dir}/top_diff10.png" alt="top diff10"></a><figcaption>top × 10</figcaption></figure>
  </div>
  <details>
    <summary>Validator checks ({verdict_badge})</summary>
    <ul>{checks_html}</ul>
    <p class="reasoning">{reasoning}</p>
  </details>
</section>
"""


def render_html(iters: list[dict]) -> str:
    iter_count = len(iters)
    committed = [r for r in iters if r.get("action") == "commit"]
    best = min((r["kernel_ms_median"] for r in committed), default=float("inf"))
    best_str = f"{best:.2f} ms" if best != float("inf") else "n/a"
    cards = "\n".join(render_card(r) for r in reversed(iters))

    backburner_md_path = OUT / "BACKBURNER.md"
    backburner_html = ""
    if backburner_md_path.exists():
        backburner_html = "<pre>" + escape(backburner_md_path.read_text()) + "</pre>"

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>gsplat_tt opt-v2 — autonomous loop</title>
  <link rel="stylesheet" href="report.css">
</head>
<body>
  <header class="top">
    <h1>gsplat_tt opt-v2 — autonomous loop</h1>
    <div class="topbar">
      iters: {iter_count} · current best: {best_str} · target: 1.0 ms
    </div>
    <p><a href="STATUS.md">STATUS</a> · <a href="BACKBURNER.md">BACKBURNER</a> · <a href="iters.jsonl">iters.jsonl</a></p>
  </header>
  <section class="graphs">
    <a href="graphs/graph-progress.png" target="_blank"><img src="graphs/graph-progress.png" alt="progress: kernel ms + PSNR"></a>
    <a href="graphs/graph-class-progress.png" target="_blank"><img src="graphs/graph-class-progress.png" alt="class progress"></a>
  </section>
  <section class="iters">
    <h2>Per-iter cards (newest first)</h2>
    {cards}
  </section>
  <section class="backburner">
    <h2>BACKBURNER</h2>
    {backburner_html}
  </section>
</body>
</html>
"""


def main() -> None:
    iters = load_iters()
    graph_combined(iters)
    if iters:
        graph_class_progress(iters)
    # Drop stale per-metric graphs from the pre-combined layout, if present.
    for stale in ("graph-kernel-ms.png", "graph-kernel-ms-per-view.png", "graph-psnr.png"):
        p = GRAPHS / stale
        if p.exists():
            p.unlink()
    (OUT / "REPORT.html").write_text(render_html(iters))
    print(f"built REPORT.html for {len(iters)} iters")


if __name__ == "__main__":
    main()
