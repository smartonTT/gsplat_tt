"""Task #741: fp32 op/colour side bucket (GSPLAT_TT_TRAIN_OPC_SIDE=1).

The TOWN emit stamps bucket slot c + 1 in rec[6] bits 16..31 and the mover
writes the record's fp32 [op, r, g, b] to side entry t * cap + c (1 KB pages of
64 entries). The blend reader reads the subchunk's pages in windows and permutes
them by stamp; any unstamped record keeps the per-record gather. This replays the
reader's window loop on random subchunks against the direct lookup and checks
the wiring (defines, runtime args, the shifted coordinator noc_xy args).
"""
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
READER = (ROOT / "render/kernels/dataflow/reader_alpha_blend_mb_devcull.cpp").read_text()
MOVER = (ROOT / "render/kernels/dataflow/sort_bin_onelaunch.cpp").read_text()
TOWN = (ROOT / "render/kernels/compute/sort_ol_town_compute.cpp").read_text()
SORT_HOST = (ROOT / "render/host/sort_device.cpp").read_text()
BLEND_HOST = (ROOT / "render/host/blend_device.cpp").read_text()
TRAIN_H = (ROOT / "render/host/blend_train.h").read_text()

WIN = 128


def stamp(opr, c):
    """sort_ol_town_compute.cpp d[6] under TRAIN_OPC_SIDE."""
    return ((opr & 0xFFFF) | ((c + 1) << 16)) if c + 1 <= 0xFFFF else (opr & 0xFFFF)


def reader_side(recs, side_pages, tile_id, cap):
    """Python transcription of the reader's side path; None = gather fallback."""
    st = [r[6] >> 16 for r in recs]
    if not recs or 0 in st:
        return None
    cs = [s - 1 for s in st]
    page0 = tile_id * (cap // 64)
    dst = [None] * len(recs)
    reads = 0
    pa, pz = min(cs) // 64, max(cs) // 64
    while pa <= pz:
        pe = pz + 1 if pz - pa < WIN else pa + WIN
        need = {c // 64 for c in cs if pa <= c // 64 < pe}
        stg = {p: side_pages[page0 + p] for p in need}
        reads += len(need)
        for g, c in enumerate(cs):
            if pa * 64 <= c < pe * 64:
                p = c // 64
                dst[g] = stg[p][(c % 64) * 4:(c % 64) * 4 + 4]
        pa += WIN
    assert all(d is not None for d in dst)
    return dst, reads


def test_side_matches_lookup():
    rng = random.Random(741)
    for cap in (64, 4096, 16384):
        tiles = 3
        side = [[rng.random() for _ in range(4)] for _ in range(tiles * cap)]
        pages = [sum(side[p * 64:p * 64 + 64], []) for p in range(tiles * cap // 64)]
        for tile_id in range(tiles):
            for n_rec in sorted({min(n, cap) for n in (1, 63, 64, 65, 4096)}):
                slots = rng.sample(range(cap), n_rec)
                recs = [[0] * 6 + [stamp(rng.randrange(1 << 16), c), 0] for c in slots]
                got, reads = reader_side(recs, pages, tile_id, cap)
                assert got == [side[tile_id * cap + c] for c in slots]
                assert reads == len({c // 64 for c in slots})


def test_unstamped_falls_back():
    recs = [[0] * 6 + [stamp(5, 3), 0], [0] * 6 + [0x1234, 0]]
    assert reader_side(recs, [], 0, 64) is None
    assert stamp(0x1234, 0xFFFF) == 0x1234  # slot past 16 bits: no stamp
    assert stamp(0xABCD0042, 9) == 0x000A0042  # the cull's bits 0..15 kept


def test_wiring():
    assert "GSPLAT_TT_TRAIN_OPC_SIDE" in TRAIN_H and "kSideReaderArg = 29u" in TRAIN_H
    # TOWN: side ring after the record ring and run starts; stamp in rec[6].
    assert "OL_RING_TILES * D * REC_BYTES + OL_RING_TILES * 4u" in TOWN
    assert "static_assert(!(TRAIN_OPC_SIDE && PFWC_REC32)" in TOWN
    # Mover: same side ring address, arg 30, coordinator noc_xy from 31.
    assert "OL_RING_TILES * D * REC_BYTES + OL_RING_TILES * 4u" in MOVER
    assert "side_gen{get_arg_val<uint32_t>(30), 1024u}" in MOVER
    assert "get_arg_val<uint32_t>(31 + r)" in MOVER and "get_arg_val<uint32_t>(30 + r)" not in MOVER
    # Host sort: both emit arg lists carry arg 30 before the coordinator insert.
    assert SORT_HOST.count("// arg 30") == 2
    assert 'defines["TRAIN_OPC_SIDE"] = "1"' in SORT_HOST and 'tdef["TRAIN_OPC_SIDE"] = "1"' in SORT_HOST
    assert "GSPLAT_TT_TRAIN_OPC_SIDE=1 needs GSPLAT_TT_PFWC_REC32=0" in SORT_HOST
    assert 'register_buffer("train_opc_side"' in SORT_HOST
    assert "kOneLaunchRingTiles * depth * 16u" in SORT_HOST
    # Blend: reader args 29/30 after 28, define.
    assert 'reader_defines["TRAIN_OPC_SIDE"] = "1"' in BLEND_HOST
    i28 = BLEND_HOST.index("reader_args.push_back(train_opc_addr);")
    assert BLEND_HOST.index("reader_args.push_back(side_cap != 0u ? side_addr : 0u);") > i28
    assert "get_arg_val<uint32_t>(29);  // task #741" in READER
    assert "get_arg_val<uint32_t>(30);" in READER
