"""Model test for task #61: the blend writer packs the final 8-bit image on device.

The host used to read back 3 bf16 tiles per screen tile, widen them to fp32,
undo the microblock permutation and hand a float image to Python, which then
quantized it as uint8(clip(x, 0, 1) * 255). The writer now does the whole thing
with integer ops (render/kernels/dataflow/img_pack_u8.h). This test compiles
that header on the host and checks it against the old path:

1. bf16_to_u8 equals the numpy float32 quantization for all 65536 bf16 patterns
   (NaN excluded: never produced by the blend), and
2. pack_channel places every tile slot at the pixel mb_perm_img_of_dev() maps it
   to (the table the old host assemble used).
"""
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

HDR_DIR = Path(__file__).resolve().parents[2] / "render" / "kernels" / "dataflow"

HARNESS = r"""
#include <cstdio>
#include "img_pack_u8.h"
int main() {
    for (uint32_t b = 0; b < 65536u; b++) std::printf("%u\n", img_pack_u8::bf16_to_u8(b));
    // Placement: one pass per bit of the slot index dev; slot dev holds 1.0
    // (0x3F80 -> 255) if that bit is set, else 0. Python rebuilds dev per pixel.
    static uint32_t tile[3][512];
    static uint8_t stage[32 * 96];
    for (uint32_t bit = 0; bit < 10; bit++) {
        for (uint32_t ch = 0; ch < 3; ch++)
            for (uint32_t k = 0; k < 512; k++) {
                uint32_t lo = (((2 * k) >> bit) & 1u) ? 0x3F80u : 0u;
                uint32_t hi = (((2 * k + 1) >> bit) & 1u) ? 0x3F80u : 0u;
                tile[ch][k] = lo | (hi << 16);
            }
        for (uint32_t ch = 0; ch < 3; ch++)
            img_pack_u8::pack_channel(tile[ch], stage, ch);
        for (uint32_t p = 0; p < 32 * 96; p++) std::printf("%u\n", stage[p]);
    }
    return 0;
}
"""


def _mb_perm_img_of_dev():
    """Python port of blend_device.cpp mb_perm_img_of_dev(): dev slot -> i*32+j."""
    t = np.zeros(1024, dtype=np.int64)
    for v in range(32):
        pv = [r * 32 + c for r in range(32) for c in range(32)
              if 2 * (r // 2) + (c & 1) == v]
        my, mx = v >> 2, v & 3
        tv = [(my * 4 + dr) * 32 + mx * 8 + dc for dr in range(4) for dc in range(8)]
        for k in range(32):
            t[pv[k]] = tv[k]
    return t


@pytest.fixture(scope="module")
def harness_out(tmp_path_factory):
    cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if cxx is None:
        pytest.skip("no C++ compiler")
    d = tmp_path_factory.mktemp("img_pack_u8")
    (d / "h.cpp").write_text(HARNESS)
    subprocess.run([cxx, "-std=c++17", "-O2", f"-I{HDR_DIR}", str(d / "h.cpp"),
                    "-o", str(d / "h")], check=True)
    out = subprocess.run([str(d / "h")], check=True, capture_output=True, text=True)
    return np.array(out.stdout.split(), dtype=np.int64)


def test_bf16_to_u8_matches_numpy(harness_out):
    got = harness_out[:65536]
    bits = np.arange(65536, dtype=np.uint32) << 16
    f = bits.view(np.float32)
    ok = ~np.isnan(f)
    ref = (np.clip(f, np.float32(0.0), np.float32(1.0)) * np.float32(255.0)).astype(np.uint8)
    assert np.array_equal(got[ok], ref[ok].astype(np.int64))


def test_pack_channel_matches_mb_permutation(harness_out):
    stages = harness_out[65536:].reshape(10, 32, 32, 3)
    # Slot dev was flagged 255 in stage `bit` iff bit `bit` of dev is set.
    dev_at = np.zeros((32, 32, 3), dtype=np.int64)
    for bit in range(10):
        vals = stages[bit]
        assert set(np.unique(vals)) <= {0, 255}
        dev_at |= (vals == 255).astype(np.int64) << bit
    # All three channels land on the same pixel for the same slot.
    assert np.array_equal(dev_at[..., 0], dev_at[..., 1])
    assert np.array_equal(dev_at[..., 0], dev_at[..., 2])
    tbl = _mb_perm_img_of_dev()
    img_of_dev = np.zeros(1024, dtype=np.int64)
    img_of_dev[dev_at[..., 0].reshape(-1)] = np.arange(1024)
    assert np.array_equal(img_of_dev, tbl)
