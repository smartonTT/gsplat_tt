"""Task #730: the train-mode fp32 op/colour gather moved to the blend writer.

With GSPLAT_TT_TRAIN_OPC_WRITER=1 the reader queues {slab address, count} on
CB 14 and the writer gathers into CB_TRAIN_OPC. This checks the wiring (defines,
CB, runtime arg) and replays the writer's loop on random records against the
direct per-record lookup.
"""
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
READER = (ROOT / "render/kernels/dataflow/reader_alpha_blend_mb_devcull.cpp").read_text()
WRITER = (ROOT / "render/kernels/dataflow/writer_alpha_blend.cpp").read_text()
HOST = (ROOT / "render/host/blend_device.cpp").read_text()
TRAIN_H = (ROOT / "render/host/blend_train.h").read_text()


def writer_gather(recs, opc_pages, batch):
    """Python transcription of the writer's serve() loop."""
    n_rec = len(recs)
    dst = [None] * n_rec
    g0 = 0
    while g0 < n_rec:
        g1 = g0 + batch if n_rec - g0 > batch else n_rec
        stage = [opc_pages[recs[g][7] >> 2] for g in range(g0, g1)]
        for g in range(g0, g1):
            s = (recs[g][7] & 3) * 4
            dst[g] = stage[g - g0][s:s + 4]
        g0 = g1
    return dst


def test_writer_gather_matches_lookup():
    rng = random.Random(730)
    n_scene = 5000
    opc = [[rng.random() for _ in range(4)] for _ in range(n_scene)]
    pages = [sum(opc[p * 4:p * 4 + 4], []) + [0.0] * (16 - 4 * len(opc[p * 4:p * 4 + 4]))
             for p in range((n_scene + 3) // 4)]
    for n_rec in (1, 63, 64, 65, 4096):
        recs = [[0] * 7 + [rng.randrange(n_scene)] for _ in range(n_rec)]
        for batch in (1, 64, 256):
            got = writer_gather(recs, pages, batch)
            assert got == [opc[r[7]] for r in recs]


def test_wiring():
    assert "kOpcReqCb = 14" in TRAIN_H and "kOpcWriterArg = 11u" in TRAIN_H
    assert "GSPLAT_TT_TRAIN_OPC_WRITER" in TRAIN_H
    assert 'reader_defines["TRAIN_OPC_WRITER"] = "1"' in HOST
    assert 'writer_defines["TRAIN_OPC_WRITER"] = "1"' in HOST
    assert "kOpcReqCb, 16, 4" in HOST
    assert "writer_args.push_back(train_opc_addr)" in HOST
    assert "CB_OPC_REQ = BLEND_CB_BASE + 14" in READER
    assert "CB_OPC_REQ = BLEND_CB_BASE + 14" in WRITER
    assert "get_arg_val<uint32_t>(11)" in WRITER
    # The writer's train loop must end on the sentinel and skip the blocking loop.
    assert "ended = true;\n            break;" in WRITER
    # No other kernel of the blend program uses CB id 14.
    for k in ("render/kernels/compute/alpha_blend_compute_mb.cpp",):
        assert "BLEND_CB_BASE + 14" not in (ROOT / k).read_text()
