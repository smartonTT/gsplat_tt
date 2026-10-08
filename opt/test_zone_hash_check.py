"""opt/profiler/zone_hash_check.py (task #426): reproduces the #416 Tracy zone collision."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "opt" / "profiler"))
import zone_hash_check as z  # noqa: E402

T416 = "/localdev/smarton/gstt2-t416"
DF = "render/kernels/dataflow"


def _fixture(tmp_path):
    d = tmp_path / DF
    d.mkdir(parents=True)
    a = ["\n"] * 433 + ['    DeviceZoneScopedN("sort_ol_barrier");\n']
    b = ["\n"] * 1043 + ['    DeviceTimestampedData("fz_mv_dwb", fz_dwb);\n']
    (d / "sort_bin_onelaunch.cpp").write_text("".join(a))
    (d / "sort_subchunk_materialize.cpp").write_text("".join(b))
    return tmp_path


def test_hash16_matches_416_collision():
    a = z.zone_string("sort_ol_barrier", f"{T416}/{DF}/sort_bin_onelaunch.cpp", 434)
    b = z.zone_string("fz_mv_dwb", f"{T416}/{DF}/sort_subchunk_materialize.cpp", 1044)
    assert z.hash16(a) == z.hash16(b) == 0x1486


def test_check_fails_at_t416_path(tmp_path, capsys):
    repo = _fixture(tmp_path)
    assert z.main(["--repo", str(repo), "--path-root", T416]) == 1
    err = capsys.readouterr().err
    assert "sort_ol_barrier" in err and "fz_mv_dwb" in err and "0x1486" in err


def test_check_passes_at_other_path(tmp_path):
    repo = _fixture(tmp_path)
    assert z.main(["--repo", str(repo), "--path-root", "/localdev/smarton/gstt2"]) == 0
