"""emit_cores.py --weights must not divide by zero when the capture has no record
counts (iter-207 EMIT captures): it falls back to kMoverSpeedP150 / emit time (#359)."""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "opt", "profiler", "emit_cores.py")
HDR = "x,y,core_x,core_y,risc,t,data,a,b,c,zone,type\n"


def _row(x, y, risc, t, zone, typ, data=0):
    return f"0,{x},{y},{risc},0,{t},{data},0,0,0,{zone},{typ}\n"


def test_weights_without_record_counts(tmp_path):
    p = tmp_path / "dev.csv"
    rows = ["hdr\n", HDR]
    # mover (1,2): BRISC 1000 cycles, NCRISC 2000 cycles, no ep_nrec records
    for risc, dur in (("BRISC", 1000), ("NCRISC", 2000)):
        rows += [_row(1, 2, risc, 0, "sort_ol_emit", "ZONE_START"),
                 _row(1, 2, risc, dur, "sort_ol_emit", "ZONE_END")]
    p.write_text("".join(rows))
    out = subprocess.run([sys.executable, SCRIPT, str(p), "1", "--weights"],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "no record counts" in out.stdout
    line = [l for l in out.stdout.splitlines() if l.strip().startswith("{1, 2,")][0]
    b, n = [int(v) for v in line.strip(" {},").split(",")[2:]]
    assert b > n  # BRISC took half the time of NCRISC, so it is faster


def test_weights_no_known_mover(tmp_path):
    p = tmp_path / "dev.csv"
    rows = ["hdr\n", HDR]
    # mover (0,0) is not in kMoverSpeedP150
    for risc in ("BRISC", "NCRISC"):
        rows += [_row(0, 0, risc, 0, "sort_ol_emit", "ZONE_START"),
                 _row(0, 0, risc, 1000, "sort_ol_emit", "ZONE_END")]
    p.write_text("".join(rows))
    out = subprocess.run([sys.executable, SCRIPT, str(p), "1", "--weights"],
                         capture_output=True, text=True)
    assert out.returncode != 0
    assert "kMoverSpeedP150" in out.stderr
    assert "ZeroDivisionError" not in out.stderr
