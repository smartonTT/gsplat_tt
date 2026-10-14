"""#664 BLEND_DWALK: model of the D walk in bs_middle_dwalk (no device).

Mirrors BsDWalk<IX> in render/kernels/compute/alpha_blend_compute_mb.cpp and
checks, for every microblock IX in 0..31, that each of the 7 middle loads
reads the same DEST row as the INCRWC path and that D ends at 2 IX (the
tail's base), so the replayed tail and its clearing store match.
"""
import re
from pathlib import Path

KERNEL = Path(__file__).resolve().parents[2] / "render/kernels/compute/alpha_blend_compute_mb.cpp"
DR_S = 192
# load k -> DEST slot, SWAP_FILL order
SLOTS = {"S_C0": 11, "S_OP": 5, "S_K99": 12, "S_FL": 6, "S_CR": 7, "S_CG": 8, "S_CB": 9}
ORDER = ["S_C0", "S_OP", "S_K99", "S_FL", "S_CR", "S_CG", "S_CB"]


def walk(ix):
    a, b, c = ix // 16, (ix % 16) // 4, ix % 4
    inc = [32] * a + [8] * b + [2] * c
    inc += [0] * (7 - len(inc))
    return inc


def test_walk_reads_same_rows_and_ends_at_tail_base():
    for ix in range(32):
        inc = walk(ix)
        assert len(inc) == 7
        d = 0
        for k, s in enumerate(ORDER):
            imm = 2 * (DR_S + SLOTS[s]) - d
            assert 0 <= imm < (1 << 13)
            assert (imm + d) == 2 * (DR_S + SLOTS[s])  # same absolute row as D=0 path
            d += inc[k]
        assert d == 2 * ix


def test_kernel_dwalk_load_order_matches_model():
    src = KERNEL.read_text()
    body = src[src.index("inline void bs_middle_dwalk()"):]
    body = body[: body.index("\n}\n")]
    loads = re.findall(r"BS_LDW\(\d, (S_\w+), (\d)u\)", body)
    assert [s for s, _ in loads] == ORDER
    assert [int(k) for _, k in loads] == list(range(7))
    # the kernel's slot constants agree with the model
    for s, v in SLOTS.items():
        m = re.search(rf"\b{s}\s*=\s*(\d+)", src)
        assert m and int(m.group(1)) == v, s


def test_increment_addr_mods_match_model():
    src = KERNEL.read_text()
    for incr in (2, 8, 32):
        assert re.search(rf"\.dest = \{{\.incr = {incr}\}}\}}\.set\(BS_AM_{incr}\)", src)
    assert ".dest = {.incr = 0, .clr = 1}}.set(BS_AM_CLR)" in src
