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
print("ok")
