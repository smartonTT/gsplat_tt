"""Task #639 (review #635): the L-A prefetch must not be gated off, and an inert run must fail."""
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "render/host/tile_assign_device.cpp")
LA = os.path.join(ROOT, "docs/la-prefetch-t628")


def test_prefetch_gate_not_host_free():
    src = open(SRC).read()
    # The early-return list of `if (prefetch) { ... }` in fused_k2_impl.
    m = re.search(r"if \(prefetch\) \{(.*?)return false;", src, re.S)
    assert m, "prefetch gate not found"
    assert "host_free_mp_enabled" not in m.group(1), "gate requires the constexpr-false host-free flag"


def test_overflow_falls_back_to_normal_path():
    src = open(SRC).read()
    m = re.search(r"la_mread\[2\] != 0 && !env_config::host_free_mp_enabled\(\)\) \{(.*?)\n            \}", src, re.S)
    assert m, "la_wait overflow branch not found"
    body = m.group(1)
    assert "ctx->fold_ran = true" in body and "/*prepare=*/false" in body


def _phases(*files):
    return subprocess.run([sys.executable, os.path.join(LA, "phases.py"), *files], capture_output=True, text=True)


def test_phases_fails_inert_lah():
    r = _phases(os.path.join(LA, "out/htl-r1-lah.txt.gz"))  # t628: L-A never ran
    assert r.returncode == 1 and "FAIL no la_wait zone" in r.stdout


def test_phases_passes_live_lah():
    r = _phases(os.path.join(LA, "out-t639/htl-r1-lah.txt.gz"))
    assert r.returncode == 0 and " W=" in r.stdout, r.stdout
