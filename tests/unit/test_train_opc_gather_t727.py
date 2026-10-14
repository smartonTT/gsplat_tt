"""Task #727: the train-mode fp32 op/colour gather in the blend reader.

The reader may batch more reads per barrier (TRAIN_OPC_BATCH) and skip records whose
slab mask (rec[3]) is 0 (TRAIN_OPC_SKIP_DEAD). This replays the kernel's two loops
(issue into stage slots, then copy) on random records and checks that every record
the compute reads (mask != 0) gets its own fp32 entry, for each knob setting, and
that the host wires the knobs.
"""
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
READER = (ROOT / "render/kernels/dataflow/reader_alpha_blend_mb_devcull.cpp").read_text()
HOST = (ROOT / "render/host/blend_device.cpp").read_text()


def gather(recs, opc_pages, batch, skip_dead):
    """Python transcription of the reader loops (stage slot n holds one 64 B page)."""
    L = len(recs)
    dst = [None] * L
    g0 = 0
    while g0 < L:
        stage = []
        g1 = g0
        while g1 < L and len(stage) < batch:
            if skip_dead and recs[g1][3] == 0:
                g1 += 1
                continue
            stage.append(opc_pages[recs[g1][7] >> 2])
            g1 += 1
        n = 0
        for g in range(g0, g1):
            if skip_dead and recs[g][3] == 0:
                continue
            gid = recs[g][7]
            dst[g] = stage[n][(gid & 3) * 4:(gid & 3) * 4 + 4]
            n += 1
        g0 = g1
    return dst


def test_gather_knobs_give_the_same_live_entries():
    rng = random.Random(727)
    n_gauss = 5000
    opc = [[rng.getrandbits(32) for _ in range(4)] for _ in range(n_gauss)]
    pages = [sum(opc[p * 4:p * 4 + 4], []) + [0] * (16 - 4 * len(opc[p * 4:p * 4 + 4]))
             for p in range((n_gauss + 3) // 4)]
    for L in (1, 63, 64, 65, 3000, 8191):
        recs = [[0, 0, 0, rng.choice([0, 0, 1, 0x55]), 0, 0, 0, rng.randrange(n_gauss)]
                for _ in range(L)]
        for batch in (1, 64, 2048):
            for skip in (False, True):
                dst = gather(recs, pages, batch, skip)
                for g, r in enumerate(recs):
                    if r[3] != 0 or not skip:
                        assert dst[g] == opc[r[7]], (L, batch, skip, g)


def test_kernel_defaults_and_bounds():
    assert re.search(r"#define TRAIN_OPC_BATCH 64u", READER)
    assert re.search(r"#define TRAIN_OPC_SKIP_DEAD 0", READER)
    assert "TRAIN_OPC_BATCH * 64u <= 128u * 1024u" in READER
    # skip-dead must test the same word in both loops (the slab mask, rec[3])
    assert READER.count("recs[g1 * 8u + 3u] == 0u") == 1
    assert READER.count("recs[g * 8u + 3u] == 0u") == 1


def test_host_wires_the_knobs():
    for env, define in (("GSPLAT_TT_TRAIN_OPC_BATCH", "TRAIN_OPC_BATCH"),
                        ("GSPLAT_TT_TRAIN_OPC_SKIPDEAD", "TRAIN_OPC_SKIP_DEAD"),
                        ("GSPLAT_TT_T727_DIAG", "T727_DIAG")):
        assert env in HOST and f'reader_defines["{define}"]' in HOST
