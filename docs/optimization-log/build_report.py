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
    """Single combined graph: kernel ms (log, left axis) + PSNR (linear, right axis).

    Plots only at-or-better iters: each entry must have kernel_ms_median <=
    the running best so far (iter-0 anchors). This is the "committed trajectory"
    — regressed iters and historical opt-stable data are excluded.
    """
    nan = float("nan")
    best = float("inf")
    kept: list[dict] = []
    for r in iters:
        ms = r.get("kernel_ms_median")
        if ms is None:
            continue
        if ms <= best:
            kept.append(r)
            best = ms

    labels = [r["iter_dir"] for r in kept]
    medians = [r.get("kernel_ms_median", nan) for r in kept]
    hero = [r.get("per_view_median", {}).get("hero", nan) for r in kept]
    side = [r.get("per_view_median", {}).get("side", nan) for r in kept]
    top  = [r.get("per_view_median", {}).get("top",  nan) for r in kept]
    psnr_min = [(_psnr_min(r) if _psnr_min(r) is not None else nan) for r in kept]
    actions = [r.get("action", "unknown") for r in kept]

    xs = list(range(len(labels)))
    fig, ax_ms = plt.subplots(figsize=(14, 6))
    ax_ms.plot(xs, medians, "-", color="#1f77b4", linewidth=1.6, label="kernel median", zorder=3)
    colors = [ACTION_COLOR.get(a, "#999999") for a in actions]
    ax_ms.scatter(xs, medians, c=colors, s=44, zorder=4, edgecolors="white", linewidths=0.7)
    ax_ms.plot(xs, hero, "-", color="#1f77b4", alpha=0.35, linewidth=0.9, label="hero")
    ax_ms.plot(xs, side, "-", color="#2ca02c", alpha=0.35, linewidth=0.9, label="side")
    ax_ms.plot(xs, top,  "-", color="#d62728", alpha=0.35, linewidth=0.9, label="top")
    ax_ms.axhline(1.0, color="#000000", linestyle=":", linewidth=1, label="target 1.0 ms")
    ax_ms.set_xlabel("iter")
    ax_ms.set_ylabel("kernel ms (log)")
    ax_ms.set_yscale("log")
    ax_ms.grid(True, alpha=0.3)
    ax_ms.set_xticks(xs)
    ax_ms.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)

    ax_db = ax_ms.twinx()
    ax_db.plot(xs, psnr_min, "--", color="#9467bd", linewidth=1.2, marker="s",
               markersize=4, label="PSNR min (dB)")
    ax_db.axhline(35.0, color="#9467bd", linestyle=":", linewidth=1, alpha=0.6,
                  label="PSNR gate 35 dB")
    ax_db.set_ylabel("PSNR (dB)")
    ax_db.set_ylim(0, 55)

    h1, l1 = ax_ms.get_legend_handles_labels()
    h2, l2 = ax_db.get_legend_handles_labels()
    ax_ms.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8, ncol=2)
    ax_ms.set_title("Kernel ms + PSNR — committed trajectory (at-or-better iters)")
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


def _brief_for_iter(iter_name: str) -> str:
    """One-line description: first paragraph after '## Hypothesis' in the iter's
    md file (capped). Falls back to the iter's slug if no md exists."""
    md = OUT / f"{iter_name}.md"
    if md.exists():
        text = md.read_text()
        if "## Hypothesis" in text:
            after = text.split("## Hypothesis", 1)[1].strip()
            # First non-empty, non-heading paragraph.
            para_lines: list[str] = []
            for line in after.splitlines():
                s = line.strip()
                if not s:
                    if para_lines:
                        break
                    continue
                if s.startswith("#"):
                    if para_lines:
                        break
                    continue
                para_lines.append(s)
            brief = " ".join(para_lines)
            if brief:
                return brief[:280] + ("…" if len(brief) > 280 else "")
    # Fallback: pretty-print the slug.
    slug = iter_name.split("-", 2)[-1] if iter_name.startswith("iter-") else iter_name
    return slug.replace("-", " ")


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

    return f"""
<section class="iter-card">
  <header>
    <h3>{escape(iter_name)}
      <span class="badge" style="background:{badge_color}">{escape(badge)}</span>
      <span class="badge verdict">{escape(verdict_badge)}</span>
      <span class="class-tag">{escape(r.get('class', '?'))}</span>
      {f'<a href="../../{escape(r["iter_dir"])}.md">md</a>' if r.get('commit_sha') else ''}
      <span class="ts">{escape(r.get('timestamp', ''))}</span>
    </h3>
  </header>
  <p class="brief">{escape(brief)}</p>
  <div class="metrics">
    <div class="ms">kernel: <b>{r.get('kernel_ms_median', float('nan')):.2f} ms</b> median · {r.get('kernel_ms_p99', float('nan')):.2f} ms p99</div>
    <div class="per-view">per-view: hero {per_view.get('hero', float('nan')):.2f} / side {per_view.get('side', float('nan')):.2f} / top {per_view.get('top', float('nan')):.2f}</div>
    <div class="psnr">PSNR: hero {psnr.get('hero', float('nan')):.1f} / side {psnr.get('side', float('nan')):.1f} / top {psnr.get('top', float('nan')):.1f} (min <b>{psnr_min:.1f}</b>)</div>
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
