"""Task #584: BLEND_EXP1 kernel source check (docs/exp1-t584/check_exp1.py)."""
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TT_METAL = os.environ.get("TT_METAL_HOME", os.path.expanduser("~/dev/tt-metal"))
OPS_H = os.path.join(TT_METAL, "tt_metal/tt-llk/tt_llk_blackhole/common/inc/ckernel_ops.h")


@pytest.mark.skipif(not os.path.exists(OPS_H), reason="needs the tt-metal ckernel_ops.h")
def test_blend_exp1_source():
    r = subprocess.run([sys.executable, os.path.join(ROOT, "docs/exp1-t584/check_exp1.py"), TT_METAL],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "check_exp1: OK" in r.stdout
    # Task #646: the pad-filled bodies are checked too (equivalent, 4 ops fewer).
    assert "A2 singles with SWAP_FILL: 32/32" in r.stdout
