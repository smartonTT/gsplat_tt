"""Off-device test for the chunked Tracy capture stitcher (opt/profiler/stitch_device_csv.py)."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "opt" / "profiler"))
from stitch_device_csv import read_csv, segment_frames  # noqa: E402

HDR = ("ARCH: blackhole, CHIP_FREQ[MHz]: 1350, Max Compute Cores: 120\n"
       "PCIe slot, core_x, core_y, RISC processor type, timer_id, time[cycles since reset], "
       "data, run host ID, trace id, trace id counter, zone name, type, source line, "
       "source file, meta data\n")
FRAME_CYC = 150 * 1350 * 1000  # 150 ms between frames


def row(cx, t, zone, typ, risc="NCRISC"):
    return f"0,{cx},2,{risc},1,{t},0,0,,,{zone},{typ},1,k.cpp,\n"


def chunk(path, n_frames, t0, cores=(1, 2), drop_anchor_in=None):
    lines = []
    for f in range(n_frames):
        base = t0 + f * FRAME_CYC
        for cx in cores:
            lines.append(row(cx, base - 50, "NCRISC-FW", "ZONE_START"))  # FW wraps the kernel
            if not (drop_anchor_in == f and cx == cores[-1]):
                lines.append(row(cx, base, "proj_count", "ZONE_START"))
                lines.append(row(cx, base + 1000 * cx, "proj_count", "ZONE_END"))
            lines.append(row(cx, base + 5_000_000, "NCRISC-FW", "ZONE_END"))
    path.write_text(HDR + "".join(lines))
    return len(lines)


def test_segments_frames_with_fw_rows_before_anchor(tmp_path):
    p = tmp_path / "c.csv"
    chunk(p, 3, 10_000_000)
    _, lines, ts = read_csv(p)
    frame, n, cores = segment_frames(lines, ts)
    assert (n, cores) == (3, 2)
    # the FW start 50 cycles before each anchor belongs to that anchor's frame
    assert list(frame) == [f for f in range(3) for _ in range(8)]


def test_uneven_frame_is_rejected(tmp_path):
    p = tmp_path / "c.csv"
    chunk(p, 3, 10_000_000, drop_anchor_in=1)
    _, lines, ts = read_csv(p)
    with pytest.raises(ValueError, match="uneven"):
        segment_frames(lines, ts)


def test_stitch_drops_warmups_and_orders_chunks(tmp_path):
    a, b, out = tmp_path / "a.csv", tmp_path / "b.csv", tmp_path / "s.csv"
    chunk(a, 4, 10_000_000)
    chunk(b, 3, 10_000_000)  # device reset between jobs: same timestamps as chunk a
    r = subprocess.run([sys.executable, str(ROOT / "opt/profiler/stitch_device_csv.py"),
                        "-o", str(out), str(a), str(b)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "frames=5" in r.stdout.splitlines()[-1]
    _, lines, ts = read_csv(out)
    assert len(lines) == 5 * 8
    frame, n, _ = segment_frames(lines, ts)
    assert n == 5
    assert all(x < y for x, y in zip(ts[:24], ts[24:]))  # chunk b shifted after chunk a


def test_long_marker_free_stretch_inside_frame(tmp_path):
    # Like sort_bucket_emit: a zone with no markers for ~100 ms, longer than the
    # ~50 ms host gap between frames. A widest-gap split would cut inside it.
    ms = 1350 * 1000
    lines = []
    for f in range(3):
        base = 10 * ms + f * 150 * ms
        for cx in (1, 2):
            lines += [row(cx, base - 50, "NCRISC-FW", "ZONE_START"),
                      row(cx, base, "proj_count", "ZONE_START"),
                      row(cx, base + ms, "proj_count", "ZONE_END"),
                      row(cx, base + 2 * ms, "emit", "ZONE_START"),
                      row(cx, base + 100 * ms, "emit", "ZONE_END"),
                      row(cx, base + 100 * ms + 10, "NCRISC-FW", "ZONE_END")]
    p = tmp_path / "c.csv"
    p.write_text(HDR + "".join(lines))
    _, lines, ts = read_csv(p)
    frame, n, _ = segment_frames(lines, ts)
    assert n == 3
    assert list(frame) == [f for f in range(3) for _ in range(12)]


def test_t142_precull_orphan_kernel_markers(tmp_path):
    """Real t142-pc capture trimmed to the 3 cores whose NCRISC-KERNEL START went stale
    (~3 s before the capture): used to raise 'zones straddle frame boundaries'."""
    import gzip
    p = tmp_path / "pc.csv"
    p.write_bytes(gzip.decompress((ROOT / "tests/fixtures/profiler/t142_pc_3cores.csv.gz").read_bytes()))
    _, lines, ts = read_csv(p)
    frame, n, cores = segment_frames(lines, ts)
    assert (n, cores) == (11, 9)
    orph = [lines[i].split(",")[1:3] + [lines[i].split(",")[11]] for i in np.nonzero(frame < 0)[0]]
    assert sorted(orph) == sorted([["12", "7", t] for t in ("ZONE_START", "ZONE_END")]
                                  + [["14", "2", t] for t in ("ZONE_START", "ZONE_END")]
                                  + [["15", "9", t] for t in ("ZONE_START", "ZONE_END")])
    # no kept NCRISC-KERNEL span is longer than a frame (was 341 ms with the stale START)
    assert ts[frame >= 0].max() - ts[frame >= 0].min() < 11 * 200 * 1350 * 1000
