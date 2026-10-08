"""Off-device test for the t413 per-job TRISC1 gap report (docs/xvpin-tracy/ana407.py job_gaps)."""
import importlib.util
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
US = 1350  # cycles per us at 1350 MHz


def _load():
    spec = importlib.util.spec_from_file_location("ana407", ROOT / "docs/xvpin-tracy/ana407.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_job_gaps_start_between_tail(tmp_path, capsys):
    m = _load()
    c = (1, 2)
    Z = defaultdict(list)
    # mat 0..1000 us; jobs 100-300, 340-600, 700-900 -> start 100, between 40 + 100, tail 100
    Z[(0, 3, "mat_cull_mask")] = [(c, "TRISC_1", 0, 1000 * US)]
    Z[(0, 3, "mj_job")] = [(c, "TRISC_1", a * US, b * US) for a, b in ((100, 300), (340, 600), (700, 900))]
    Z[(0, 3, "mj_wait")] = [(c, "TRISC_1", a * US, b * US) for a, b in ((0, 100), (300, 340), (600, 700), (900, 1000))]
    Z[(0, 3, "mat_ol_sort")] = [(c, "NCRISC", 0, 100 * US)]           # covers the whole start gap
    Z[(0, 3, "mat_cull_wait")] = [(c, "BRISC", 900 * US, 950 * US)]   # half of the tail
    Z[(0, 3, "tile_blend_sfpu")] = [(c, "TRISC_1", 1000 * US, 1400 * US)]
    Z[(0, 3, "rd_l1_bulk")] = [(c, "NCRISC", 1000 * US, 1010 * US)] * 4  # 4 blend jobs of 100 us
    D = defaultdict(lambda: defaultdict(int))
    D[0][(c, "TRISC_1", "fz_m_band")] = 500 * US
    D[0][(c, "TRISC_1", "fz_m_copy")] = 10 * US
    D[0][(c, "TRISC_0", "fz_u_nj")] = 3
    CL = {(0, 3): {c: (0, 1400 * US)}}
    out = tmp_path / "jobs.csv"
    m.job_gaps(Z, D, CL, 1, [c], str(out))
    txt = capsys.readouterr().out
    row = out.read_text().splitlines()[1].split(",")
    assert row[3] == "3"                                    # jobs
    assert abs(float(row[5]) - 0.1) < 1e-6                  # start
    assert abs(float(row[6]) - 0.14) < 1e-6                 # between sum
    assert abs(float(row[7]) - 0.1) < 1e-6                  # between max
    assert abs(float(row[8]) - 0.1) < 1e-6                  # tail
    assert abs(float(row[9]) - (0.66 - 0.51)) < 1e-6        # in-job idle = 660 - 500 - 10 us
    assert "start    " in txt and "NCRISC:mat_ol_sort 100%" in txt
    assert "BRISC:mat_cull_wait 50%" in txt
    assert "WARNING" not in txt
    # whole 100 us blend jobs fit: start 1, between 0 + 1, tail 1 -> 0.3 ms; crit -0.3
    assert "whole blend jobs   fillable 0.300 mean-core; crit -0.300" in txt
