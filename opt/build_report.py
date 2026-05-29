"""Regenerate opt/REPORT.html from opt/iters.jsonl + opt/metal-iters.jsonl.

Single self-contained HTML. Sections:
  1. Profile line graphs (sum-ms over iters; per-stage breakdown; PSNR floor)
  2. Per-iter ledger     (one big card per iter — hero/diff thumbnails,
                          stage timings, PSNR, action, full note)
  3. Algorithm snapshot  (current pipeline state, brief)

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
VIEWS_PER_RUN = 30  # 30-view bicycle bench; iter sums always cover all 30 views
TARGET_MS_PER_FRAME = 1.0  # 1 ms per frame on bh-30 — the final goal
TARGET_SUM_MS = TARGET_MS_PER_FRAME * VIEWS_PER_RUN  # legacy helper (= 30 ms)
STAGE_KEYS = ("project_ms", "tile_assign_ms", "sort_ms", "blend_ms")
STAGE_TIMING_KEYS = ("project", "tile_assign", "sort", "blend")
REF_DIR = OPT_DIR.parent / "benchmarks" / "reference_v2"


def img_link(src: str, cls: str = "thumb") -> str:
    return f'<a href="{src}" target="_blank" rel="noopener"><img src="{src}" class="{cls}" alt=""></a>'


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


def ensure_hero_diff10(iter_dir: str) -> None:
    """Write hero_diff10.png as 10× per-channel abs color diff vs reference,
    in EVERY location where hero.png exists. Always regenerates so a refreshed
    reference produces correct diffs (no stale stitch-vs-bicycle artifacts)."""
    if not iter_dir:
        return
    ref = REF_DIR / "hero.png"
    if not ref.exists():
        return
    import numpy as np
    from PIL import Image

    ref_rgb = np.asarray(Image.open(ref).convert("RGB"), dtype=np.float64) / 255.0
    for d in _candidate_hero_dirs(iter_dir):
        hero = d / "hero.png"
        if not hero.exists():
            continue
        cand_rgb = np.asarray(Image.open(hero).convert("RGB"), dtype=np.float64) / 255.0
        if ref_rgb.shape != cand_rgb.shape:
            continue
        amp = np.clip(np.abs(ref_rgb - cand_rgb) * 10.0, 0.0, 1.0)
        Image.fromarray((amp * 255.0).astype(np.uint8)).save(d / "hero_diff10.png")


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
  <p>Target: bh-30 P150 Blackhole, 1 ms/frame. Plan:
  <a href='plan-amendment-002-tt-emulator-port.md'>plan-amendment-002-tt-emulator-port.md</a>.
  No metal iters logged yet.</p>
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
        ts = r.get("timestamp", "")[:19].replace("T", " ")
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
            f"<p>Plan: <a href='plan-amendment-002-tt-emulator-port.md'>plan-amendment-002-tt-emulator-port.md</a>. "
            f"Reference of record: <code>cpu_cpp_mb</code> backend; PSNR gated against "
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


def merged_iter_series() -> list[dict]:
    """Bicycle-scene CPU baseline (iter-057) followed by Blackhole/metal iters
    in timestamp order. Pre-bicycle CPU sprint iters (stitch_doll) are kept
    in iters.jsonl and the ledger but excluded from the chart for scene
    consistency."""
    cpu_rows = [
        {**r, "_runtime": "cpu", "_label": r.get("iter_dir", "")[5:].lstrip("0123456789-")[:18]}
        for r in load_iters()
        if (n := _iter_num(r.get("iter_dir", ""))) is not None and n >= BICYCLE_START_ITER_NUM
    ]
    metal_rows_raw = sorted(
        load_metal_iters(), key=lambda r: r.get("timestamp", "")
    )
    metal_rows = [_normalize_metal_row(r) for r in metal_rows_raw]
    return cpu_rows + metal_rows


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
        if (base / sub / "hero.png").exists():
            diff = ""
            if (base / sub / "hero_diff10.png").exists():
                diff = f"{prefix}/{sub}/hero_diff10.png"
            return (f"{prefix}/{sub}/hero.png", diff)
    if (base / "hero.png").exists():
        diff = "" if not (base / "hero_diff10.png").exists() else f"{prefix}/hero_diff10.png"
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
    if finite:
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

    note = (r.get("validator_reasoning") or r.get("note") or "").strip()

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
<div class='backburner-row'>
  <div class='backburner-meta'>
    <h3>{pos_html}{priority} {iter_dir} — {verdict}</h3>
    <p>ms_per_frame={sum_ms_str}, psnr={psnr_min_str}</p>
    {caveat_html}
    <p class='reason'>{note}</p>
  </div>
  <div class='backburner-thumbs'>{thumb_html}</div>
</div>
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
    merged = (
        [{**r, "_runtime": "cpu", "_scene": "bicycle"} for r in bicycle_cpu]
        + [{**r, "_scene": "bicycle"} for r in metal_norm]
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
        return (
            f"<tr style='background:{row_bg}'>"
            f"<td>{preview}</td>"
            f"<td><a href='{link_prefix}{r['iter_dir']}/'>{r['iter_dir']}</a></td>"
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
  <p>See <a href='plan.md'>plan.md</a> for the frozen plan,
  <a href='plan-amendment-002-tt-emulator-port.md'>plan-amendment-002-tt-emulator-port.md</a>
  for the TT port architecture, and
  <a href='microblock-cpu-spec.md'>microblock-cpu-spec.md</a> for the microblock
  binning contract.</p>
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
    <li><b>contrib_floor</b> = 1/16384 (set in <code>benchmarks/cameras_v2.json</code>;
        Pipeline default in <code>gsplat/pipeline.py</code>).</li>
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
  .caveat { color: #b8860b; font-size: 11px; font-style: italic; margin: 4px 0; }
  .iter-pos { display: inline-block; min-width: 48px; padding: 1px 6px; margin-right: 6px; background: #1d3557; color: #fff; font-size: 11px; font-weight: 700; border-radius: 3px; letter-spacing: 0.4px; }
  .reason { color: #555; font-size: 12px; }
  code { background: #f1faee; padding: 1px 4px; border-radius: 3px; font-size: 11px; }
</style>
"""
    return f"""<!DOCTYPE html>
<html lang='en'>
<head><meta charset='utf-8'><title>gstt2 — Optimization Report</title>{css}</head>
<body>
<h1>gstt2 — Optimization Report</h1>
{figs_html}
{ledger_section(rows)}
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
