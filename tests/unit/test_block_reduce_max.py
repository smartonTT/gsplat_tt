"""Host-side checks for iter 036 block-wide MAX(T) early-termination gate.

Validates fp32 bit-pattern threshold compare used after reduce<MAX, SCALAR>.
Device reduce smoke test runs separately via tt-metal reduce_hw gtest when built.
"""
import struct

import pytest

T_THRESH_BITS = 0x38D1B717  # fp32(1e-4)


def _f32_bits(x: float) -> int:
    return struct.unpack(">I", struct.pack(">f", x))[0]


def tmax_saturated(t_max_bits: int) -> bool:
    """True when max(T) < 1e-4 (all pixels saturated after Stage F)."""
    return t_max_bits < T_THRESH_BITS


class TestBlockReduceMaxGate:
    def test_zero_max_is_saturated(self):
        assert tmax_saturated(_f32_bits(0.0))

    def test_sub_threshold_is_saturated(self):
        assert tmax_saturated(_f32_bits(1e-5))

    def test_at_threshold_not_saturated(self):
        assert not tmax_saturated(T_THRESH_BITS)

    def test_above_threshold_not_saturated(self):
        assert not tmax_saturated(_f32_bits(0.5))

    def test_background_pixel_keeps_tile_active(self):
        """One pixel at T=1.0 prevents block_saturated even if others are 0."""
        assert not tmax_saturated(_f32_bits(1.0))

    def test_fp32_reduce_output_used_not_bf16_raw(self):
        """Stage F2 reads fp32 scalar from CB_T_MAX, not raw bf16 CB_T_STATE."""
        fp32_max_from_reduce = _f32_bits(1.0)
        assert not tmax_saturated(fp32_max_from_reduce)
        # Raw bf16 1.0 bit pattern in low16 is NOT a valid fp32 max read.
        bf16_one_misread = 0x3C00
        assert tmax_saturated(bf16_one_misread)  # would false-trigger if misread
