"""Off-device test for the MATCULL_TRISC_FILL counter report (opt/profiler/fill_zones.py)."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HDR = ("ARCH: blackhole, CHIP_FREQ[MHz]: 1350, Max Compute Cores: 120\n"
       "PCIe slot, core_x, core_y, RISC processor type, timer_id, time[cycles since reset], "
       "data, run host ID, trace id, trace id counter, zone name, type, source line, "
       "source file, meta data\n")
FRAME_CYC = 150 * 1350 * 1000
MS = 1350 * 1000


def zone(cx, risc, name, s, e):
    return [f"0,{cx},2,{risc},1,{s},0,0,,,{name},ZONE_START,1,k.cpp,\n",
            f"0,{cx},2,{risc},1,{e},0,0,,,{name},ZONE_END,1,k.cpp,\n"]


def ts_data(cx, risc, name, t, v):
    return f"0,{cx},2,{risc},1,{t},{v},0,,,{name},TS_DATA,1,k.cpp,\n"


def test_fill_zones_report(tmp_path):
    lines = []
    for f in range(3):  # frame 0 is the warmup and is dropped
        b = 10 * MS + f * FRAME_CYC
        for cx in (1, 2):
            lines += zone(cx, "NCRISC", "proj_count", b, b + 1000)
            m0 = b + 2 * MS
            m1 = m0 + cx * MS  # core 1 mat 1 ms, core 2 mat 2 ms
            lines += zone(cx, "TRISC_1", "mat_cull_mask", m0, m1)
            for r in ("NCRISC", "BRISC"):
                lines += zone(cx, r, "sort_subchunk_mat", m0, m1)
                lines.append(ts_data(cx, r, "fz_mv_dw", m1, MS // 10))
                lines.append(ts_data(cx, r, "fz_mv_dwn", m1, 4))
                for k in ("dwb", "dwm", "rd", "sort", "perm", "wr", "meta", "big", "jobs", "recs"):
                    lines.append(ts_data(cx, r, f"fz_mv_{k}", m1, 0))
            u = {"tot": cx * MS, "pick": MS // 2, "fillw": 0, "fill": 1000 * 30, "nb": 10,
                 "nr": 1000, "nj": 4}
            for k, v in u.items():
                lines.append(ts_data(cx, "TRISC_0", f"fz_u_{k}", m1, v))
            for k, v in {"tot": cx * MS, "mbw": 0, "copy": 0, "band": MS // 4}.items():
                lines.append(ts_data(cx, "TRISC_1", f"fz_m_{k}", m1, v))
            for k, v in {"tot": cx * MS, "mbw": 0, "regw": 0, "pack": 0, "patchw": 0,
                         "patch": 1000 * 12}.items():
                lines.append(ts_data(cx, "TRISC_2", f"fz_p_{k}", m1, v))
    p = tmp_path / "d.csv"
    p.write_text(HDR + "".join(lines))
    r = subprocess.run([sys.executable, str(ROOT / "opt/profiler/fill_zones.py"), str(p),
                        "--percore", str(tmp_path / "pc.csv")],
                       capture_output=True, text=True, check=True)
    got = {ln.split()[0]: ln.split()[1:] for ln in r.stdout.splitlines()[2:] if ln.strip()}
    assert r.stdout.startswith("views=2 ")
    assert [float(x) for x in got["mat_end"]] == [1.5, 2.0]
    assert [float(x) for x in got["nc_dw"]] == [0.1, 0.1]
    assert [float(x) for x in got["t0_idle(pick)"]] == [0.5, 0.5]
    assert [float(x) for x in got["t0_busy_frac"]] == [0.625, 0.75]
    assert [float(x) for x in got["t0_fill_cyc_rec"]] == [30.0, 30.0]
    assert [float(x) for x in got["t2_patch_cyc_rec"]] == [12.0, 12.0]
    assert [float(x) for x in got["t1_idle(tot-band)"]] == [1.25, 1.75]
    assert float(got["mat_tail(max-mean"][1]) == 0.5
    assert (tmp_path / "pc.csv").read_text().startswith("core,")
