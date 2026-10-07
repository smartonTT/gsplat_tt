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
assert "Throughput, back-to-back" in sec and "SECONDARY METRIC" in sec, sec
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
