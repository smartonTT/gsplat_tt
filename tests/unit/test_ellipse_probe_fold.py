"""Task #633: GSPLAT_TT_ELLIPSE_PROBE (#538) must reach K2 wherever it runs.

Under the B4a fold (default on since iter 226) the next view's K2 runs on BRISC in
matblend_brisc.cpp, built by blend_device.cpp, not by tile_assign_device.cpp. Without the
define there, the probe only covers the first view's K2 and a b2b A/B measures nothing.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_fold_brisc_gets_probe_define():
    src = (ROOT / "render/host/blend_device.cpp").read_text()
    m = re.search(r"if \(b4a_k2_in_blend\(\)\) \{(.*?)\n        \}\n        ctx\.fz_brisc = CreateKernel", src, re.S)
    assert m, "B4a K2 define block before fz_brisc not found"
    block = m.group(1)
    assert "GSPLAT_TT_ELLIPSE_PROBE" in block
    assert 'd["K2_ELLIPSE_PROBE"] = "1"' in block


def test_folded_k2_includes_probe_code():
    src = (ROOT / "render/kernels/dataflow/matblend_brisc.cpp").read_text()
    assert '#include "pfwc_fuse.h"' in src
    assert "#ifdef K2_ELLIPSE_PROBE" in (ROOT / "render/kernels/dataflow/pfwc_fuse.h").read_text()
