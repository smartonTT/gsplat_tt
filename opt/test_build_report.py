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
print("ok")
