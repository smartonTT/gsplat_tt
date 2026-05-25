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


def graph_kernel_ms(iters: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    xs = list(range(len(iters)))
    medians = [r.get("kernel_ms_median", float("nan")) for r in iters]
    p99s = [r.get("kernel_ms_p99", float("nan")) for r in iters]
    colors = [ACTION_COLOR.get(r.get("action"), "#999999") for r in iters]
    ax.plot(xs, medians, "-", color="#666666", linewidth=1, label="median")
    ax.plot(xs, p99s, "--", color="#bbbbbb", linewidth=1, label="p99")
    ax.scatter(xs, medians, c=colors, s=40, zorder=3)
    if iters:
        baseline = max((r["kernel_ms_median"] for r in iters if r["iter_dir"].startswith("iter-000")), default=None)
        if baseline:
            ax.axhline(baseline, color="#cccccc", linestyle=":", label=f"iter-0 baseline ({baseline:.1f} ms)")
    ax.axhline(1.0, color="#000000", linestyle=":", linewidth=1, label="target 1.0 ms")
    ax.set_xlabel("iter index")
    ax.set_ylabel("kernel ms")
    ax.set_yscale("log")
    ax.set_title("Kernel ms (median + p99), colored by action")
    ax.legend()
    fig.tight_layout()
    fig.savefig(GRAPHS / "graph-kernel-ms.png", dpi=110)
    plt.close(fig)


def graph_kernel_ms_per_view(iters: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    xs = list(range(len(iters)))
    for view, color in [("hero", "#1f77b4"), ("side", "#2ca02c"), ("top", "#d62728")]:
        ys = [r.get("per_view_median", {}).get(view, float("nan")) for r in iters]
        ax.plot(xs, ys, "-o", color=color, markersize=4, label=view)
    ax.axhline(1.0, color="#000000", linestyle=":", linewidth=1, label="target")
    ax.set_xlabel("iter index")
    ax.set_ylabel("kernel ms (per-view median)")
    ax.set_yscale("log")
    ax.set_title("Per-view kernel ms median")
    ax.legend()
    fig.tight_layout()
    fig.savefig(GRAPHS / "graph-kernel-ms-per-view.png", dpi=110)
    plt.close(fig)


def graph_psnr(iters: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    xs = list(range(len(iters)))
    mins = [min(r.get("psnr_per_view", {"x": 0}).values()) for r in iters]
    colors = [ACTION_COLOR.get(r.get("action"), "#999999") for r in iters]
    ax.scatter(xs, mins, c=colors, s=40)
    ax.axhline(100.0, color="#000000", linestyle="--", linewidth=1, label="kernel-algebra floor (100 dB)")
    ax.axhline(50.0, color="#888888", linestyle="--", linewidth=1, label="binning/sort floor (50 dB)")
    ax.set_xlabel("iter index")
    ax.set_ylabel("min PSNR across 3 views (dB)")
    ax.set_title("PSNR min per iter with class floors")
    ax.legend()
    fig.tight_layout()
    fig.savefig(GRAPHS / "graph-psnr.png", dpi=110)
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


def render_card(r: dict) -> str:
    iter_name = r["iter_dir"]
    shot_dir = f"screenshots/{iter_name}"
    psnr = r.get("psnr_per_view", {})
    per_view = r.get("per_view_median", {})
    psnr_min = min(psnr.values()) if psnr else float("nan")
    badge_color = ACTION_COLOR.get(r.get("action"), "#999999")
    badge = r.get("action", "unknown")
    verdict_badge = r.get("verdict", "?")

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
  <div class="metrics">
    <div class="ms">kernel: <b>{r.get('kernel_ms_median', float('nan')):.2f} ms</b> median · {r.get('kernel_ms_p99', float('nan')):.2f} ms p99</div>
    <div class="per-view">per-view: hero {per_view.get('hero', float('nan')):.2f} / side {per_view.get('side', float('nan')):.2f} / top {per_view.get('top', float('nan')):.2f}</div>
    <div class="psnr">PSNR: hero {psnr.get('hero', float('nan')):.1f} / side {psnr.get('side', float('nan')):.1f} / top {psnr.get('top', float('nan')):.1f} (min <b>{psnr_min:.1f}</b>)</div>
  </div>
  <div class="thumbs">
    <figure><img src="{shot_dir}/hero.png" alt="hero"><figcaption>hero</figcaption></figure>
    <figure><img src="{shot_dir}/side.png" alt="side"><figcaption>side</figcaption></figure>
    <figure><img src="{shot_dir}/top.png" alt="top"><figcaption>top</figcaption></figure>
    <figure><img src="{shot_dir}/hero_diff10.png" alt="hero diff10"><figcaption>hero × 10</figcaption></figure>
    <figure><img src="{shot_dir}/side_diff10.png" alt="side diff10"><figcaption>side × 10</figcaption></figure>
    <figure><img src="{shot_dir}/top_diff10.png" alt="top diff10"><figcaption>top × 10</figcaption></figure>
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
    <img src="graphs/graph-kernel-ms.png" alt="kernel ms">
    <img src="graphs/graph-kernel-ms-per-view.png" alt="per-view kernel ms">
    <img src="graphs/graph-psnr.png" alt="psnr">
    <img src="graphs/graph-class-progress.png" alt="class progress">
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
    if iters:
        graph_kernel_ms(iters)
        graph_kernel_ms_per_view(iters)
        graph_psnr(iters)
        graph_class_progress(iters)
    (OUT / "REPORT.html").write_text(render_html(iters))
    print(f"built REPORT.html for {len(iters)} iters")


if __name__ == "__main__":
    main()
