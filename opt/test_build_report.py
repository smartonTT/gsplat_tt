"""python3 opt/test_build_report.py — ms_per_view uses measured view total."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import build_report as b

t = {"project": 3.0, "tile_assign": 1.0, "sort": 3.0, "blend": 5.0}
r = b.normalize_ttw_row({"iter": 1, "timings": dict(t, ms_view=12.25)})
assert abs(r["ms_per_view"] - 12.25) < 1e-9, r["ms_per_view"]
r = b.normalize_ttw_row({"iter": 1, "timings": t, "metrics": {"frame_ms_view": 12.5}})
assert abs(r["ms_per_view"] - 12.5) < 1e-9, r["ms_per_view"]
r = b.normalize_ttw_row({"iter": 1, "timings": t})
assert abs(r["ms_per_view"] - 12.0) < 1e-9, r["ms_per_view"]

# GPU anchor = newest 'keep' row with a measured ms_view, with its board.
_orig = b.load_ttw_iters
b.load_ttw_iters = lambda: [
    {"iter": 5, "decision": "keep", "timings": {"ms_view": 20.0}, "metrics": {"board": "A"}},
    {"iter": 7, "decision": "keep", "timings": {"ms_view": 14.5}, "metrics": {"board": "bh-x p100a"}},
    {"iter": 8, "decision": "rebaseline", "timings": {"ms_view": 16.0}},
    {"iter": 9, "decision": "diagnostic"},
]
ms, label = b.tt_anchor()
b.load_ttw_iters = _orig
assert ms == 14.5 and "bh-x p100a" in label and "iter-7" in label, (ms, label)
assert "latency (legacy)" in label, label

# Throughput (task #275) is a separate, labeled secondary table; empty without rows.
_orig = b._read_jsonl
b._read_jsonl = lambda p: [] if p == b.THROUGHPUT_JSONL else _orig(p)
assert b.throughput_section() == ""
b._read_jsonl = lambda p: [{"ts": "2026-10-05T14:58:00-0400", "iter_ref": 199,
    "commit": "abc1234", "board": "bh-x p100a", "b2b_ms_frame": 11.618,
    "b2b_ms_frame_rounds": [11.6, 11.62], "b2b_drop_ms_frame": 11.59,
    "latency_ms_view": 11.639, "md5": "46a725ab", "golden_match": True,
    "source": "docs/throughput-t275.md"}] if p == b.THROUGHPUT_JSONL else _orig(p)
sec = b.throughput_section()
b._read_jsonl = _orig
assert "Throughput, back-to-back" in sec and "HISTORY (before #465)" in sec, sec
assert "11.618" in sec and "86.1 FPS" in sec and "11.639" in sec and "46a725ab" in sec, sec
assert "href='../docs/throughput-t275.md'" in sec, sec
print("ok")


def test_rebase_for_ttw_prefixes_relative_paths_only():
    import build_report as br
    html = ('<img src="metal-screenshots/ttw-1/hero.png"><img src="data:image/png;base64,AA">'
            '<a href="../docs/x.md"></a><a href="https://x"></a><a href="#top"></a>')
    out = br.rebase_for_ttw(html)
    assert 'src="../metal-screenshots/ttw-1/hero.png"' in out
    assert 'src="data:image/png;base64,AA"' in out
    assert 'href="../../docs/x.md"' in out
    assert 'href="https://x"' in out and 'href="#top"' in out


def test_conclusion_section_links_doc_when_present(tmp_path):
    import build_report as br
    orig = br.CONCLUSION_MD
    try:
        br.CONCLUSION_MD = tmp_path / "missing.md"
        assert br.conclusion_section() == ""
        md = tmp_path / "conclusion.md"
        md.write_text("# Conclusion: stop at iter 207\n\nbody\n")
        br.CONCLUSION_MD = md
        sec = br.conclusion_section()
        assert "href='../docs/conclusion.md'" in sec and "stop at iter 207" in sec, sec
        assert "href=\"../../docs/conclusion.md\"" in br.rebase_for_ttw(sec.replace("'", '"'))
    finally:
        br.CONCLUSION_MD = orig


def test_conclusion_section_shows_p150_line(tmp_path):
    import json
    import build_report as br
    orig = br.CONCLUSION_MD, br.P150_JSON
    try:
        md = tmp_path / "conclusion.md"
        md.write_text("# Conclusion\n")
        br.CONCLUSION_MD, br.P150_JSON = md, tmp_path / "missing.json"
        assert "p150" not in br.conclusion_section()
        p = tmp_path / "p150.json"
        p.write_text(json.dumps({"board": "bh-30 (Blackhole p150b)", "tag": "best-iter-207",
            "ms_view": 12.906, "fps": 77.5, "md5": "906e0435", "md5_views": "30/30",
            "hero_psnr_vs_ref": 42.51, "p100a_ms_view": 10.907, "gpu_g1_ms": 10.75,
            "gpu_g1_label": "G1 published, not measured", "doc": "docs/p150-bench-bh30.md"}))
        br.P150_JSON = p
        sec = br.conclusion_section()
        assert "p150 (diagnostic, not an iteration)" in sec and "12.906 ms/view (77.5 FPS)" in sec, sec
        assert "p100a 10.907 ms" in sec and "published, not measured 10.75 ms" in sec, sec
        assert "p150 / G1 = 1.20x" in sec and "href='../docs/p150-bench-bh30.md'" in sec, sec
    finally:
        br.CONCLUSION_MD, br.P150_JSON = orig


def test_stop_line_section_final_best_levers_and_gate():
    import build_report as br
    orig = br.load_ttw_iters
    try:
        br.load_ttw_iters = lambda: []
        assert br.stop_line_section() == ""
        br.load_ttw_iters = lambda: [
            {"iter": 138, "decision": "keep", "timings": {"ms_view": 173.1}, "metrics": {"board": "bh-07 p100a"}},
            {"iter": 216, "decision": "keep", "timings": {"ms_view": 8.043},
             "metrics": {"board": "bh-04 p100a", "md5": "906e0435", "p150_board": "bh-30 p150",
                         "p150_frame_ms_view": 7.705, "p150_md5": "39d84b28", "p150_hero_psnr_vs_ref": 42.51}},
        ]
        sec = br.stop_line_section()
        assert "Conclusions / stop line" in sec and "best-iter-216" in sec and "e13e9f6b" in sec, sec
        assert "7.705 ms/view (129.8 FPS)" in sec and "39d84b28" in sec and "42.51 dB" in sec, sec
        assert "1.40&times;</b> faster than GPU G1" in sec and "published, not measured" in sec, sec
        assert "8.043 ms/view, md5 906e0435" in sec and "22.5&times;" in sec and "21.5&times;" in sec, sec
        assert "at least 0.15 ms/view" in sec and "GSPLAT_TT_CHUNK_CULL=1" in sec, sec
        assert "GSPLAT_TT_PFWC_DEAL=lpt" in sec and "7.807" in sec and "8.95" in sec, sec
    finally:
        br.load_ttw_iters = orig


def _b2b_row(it, b2b, passes, board="bh-x p100a", **kw):
    m = {"board": board, "ms_view_b2b": b2b, "ms_view_b2b_passes": passes,
         "ms_view_latency": 7.8, "commit": "abc1234", "build": "tip", "md5": "906e0435"}
    m.update(kw)
    return {"iter": it, "decision": "rebaseline", "metrics": m}


def test_b2b_anchor_headline_and_validator_rule():
    rows = [
        {"iter": 216, "decision": "keep", "timings": {"ms_view": 8.0}, "metrics": {"board": "p"}},
        _b2b_row(217, 9.5, [9.4, 9.5, 9.6]),
        _b2b_row(218, 9.1, [9.0, 9.1, 9.3], board="bh-30 p150"),
    ]
    orig = b.load_ttw_iters
    b.load_ttw_iters = lambda: rows
    try:
        ms, label = b.tt_anchor()
        sec = b.b2b_headline_section()
    finally:
        b.load_ttw_iters = orig
    # The anchor is the best b2b, not the (lower) legacy latency of iter 216.
    assert ms == 9.1 and "bh-30 p150" in label and "back-to-back" in label, (ms, label)
    assert "Headline: back-to-back ms/view" in sec and "9.100 ms/view b2b" in sec, sec
    assert "published, not measured" in sec and "1.18&times;</b> faster" in sec, sec
    assert "passes 9.000, 9.100, 9.300" in sec and "latency (legacy)" in sec, sec
    assert str(b.KEEP_GATE_MS) in sec and "42.4" in sec, sec
    # Validator rule: rows up to 216 need nothing; newer rows need >= 3 passes
    # whose median is the headline value.
    assert b.check_b2b(rows) == []
    assert b.check_b2b([{"iter": 200, "decision": "keep", "metrics": {}}]) == []
    assert b.check_b2b([{"iter": 219, "decision": "reject", "metrics": {}}]) == []
    assert "no metrics.ms_view_b2b" in b.check_b2b([{"iter": 219, "decision": "keep", "metrics": {}}])[0]
    assert ">= 3" in b.check_b2b([_b2b_row(219, 9.0, [9.0, 9.0])])[0]
    assert "not the median" in b.check_b2b([_b2b_row(219, 9.0, [9.1, 9.2, 9.3])])[0]


def test_legacy_cards_say_latency_legacy():
    r = b.normalize_ttw_row({"iter": 210, "decision": "keep", "timings": {"ms_view": 8.2}})
    assert r["ms_view_b2b"] is None
    r2 = b.normalize_ttw_row(_b2b_row(217, 9.5, [9.4, 9.5, 9.6]))
    assert r2["ms_view_b2b"] == 9.5 and r2["ms_view_latency"] == 7.8
