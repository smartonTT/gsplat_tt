"""Task #724: train mode (fp32 op/colour) keeps decode-ahead on, with 128 B ring slots.

Source checks: the host no longer forces DA 0 in train mode, the kernel's DA guard
admits the fp32 form, the host ring matches the kernel slot size, and TRISC0's fp32
slot words reach the same DEST slots on TRISC1 as blend_stage_coeffs (bit for bit).
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HOST = (ROOT / "render/host/blend_device.cpp").read_text()
TRAIN_H = (ROOT / "render/host/blend_train.h").read_text()
KERN = (ROOT / "render/kernels/compute/alpha_blend_compute_mb.cpp").read_text()


def _body(src, name):
    m = re.search(r"inline void " + name + r"\(.*?\n}\n", src, re.S)
    assert m, name
    return m.group(0)


def test_host_train_keeps_decode_ahead_default():
    m = re.search(r"const std::string blend_da =(.*?);", HOST, re.S)
    assert m
    assert "blend_train" not in m.group(1), "train mode must not force DA 0"
    assert 'std::string("2")' in m.group(1)


def test_ring_size_matches_kernel_slots():
    assert "da_slot_bytes() / 64u" in HOST
    assert re.search(r"da_slot_bytes\(\) \{ return enabled\(\) \? 128u : 64u; \}", TRAIN_H)
    m = re.search(r"#if BLEND_DA_F32\nconstexpr uint32_t DA_SLOT_BYTES = (\d+);\n#else\n"
                  r"constexpr uint32_t DA_SLOT_BYTES = (\d+);", KERN)
    assert m and (int(m.group(1)), int(m.group(2))) == (128, 64)


def test_kernel_guard_admits_train():
    assert "(BLEND_SFPU_UNORM == 1 || BLEND_DA_F32)" in KERN
    assert re.search(r"#if defined\(BLEND_TRAIN\) && BLEND_TRAIN\n#define BLEND_DA_F32 1", KERN)


def test_f32_slot_words_reach_the_same_dest_slots():
    put = _body(KERN, "da_put_rec_f32")
    stage = _body(KERN, "da_stage_f32")
    # TRISC0: word 2k = lo half, 2k+1 = hi half of value k (L0 USHORT, L0 HI16_ONLY).
    lo_hi = re.findall(r"s\[(\d+)\] = \((\w+) (& 0xffffu|>> 16)\) \+ (klo|khi);", put)
    order = []
    for i, v, half, k in lo_hi:
        i = int(i)
        assert (half == "& 0xffffu") == (i % 2 == 0) == (k == "klo"), (i, v, half, k)
        if i % 2 == 0:
            order.append(v)
    assert len(lo_hi) == 18
    assert "s[DA_W_MASK] = mask;" in put
    # The kernel's slot fits 18 words + mask + kind.
    assert 18 + 2 <= 128 // 4
    # TRISC1: the two words of value k go out, then one SFPSTORE to its slot.
    stores = re.findall(r"TTI_SFPSTORE\(0, 0, 7, A0 \+ 2u \* (S_\w+)\);", stage)
    pushes = len(re.findall(r"ib\[0\] = v[01];", stage))
    assert pushes == 18
    # Same value -> slot map as blend_stage_coeffs(a, b, c, d, e, op, cr, cg, cb).
    slot_of = {"d": "S_MX", "e": "S_MY", "a": "S_A", "b": "S_B", "c": "S_C",
               "op": "S_OP", "cr": "S_CR", "cg": "S_CG", "cbv": "S_CB"}
    assert stores == [slot_of[v] for v in order]
    # The loads walk the slot in order, two words ahead of their pushes.
    loads = [int(x) for x in re.findall(r"= s\[(\d+)\]", stage)]
    assert loads == list(range(18))
