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
