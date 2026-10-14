"""Task #755: 3-slot op/colour ring (GSPLAT_TT_TRAIN_OPC_RING3=1).

CB_BUCKET_BULK and CB_TRAIN_OPC (= CB_BUCKET, id 9) get 3 fixed slots of
BULK_REC_SLOT pages, so the blend reader gathers two subchunks ahead of the
compute. CB_TRAIN_OPC pages shrink to 32 B (a slot holds MB_BUCKET_FIT 16 B
entries) and the gather staging moves to its own CB (id 15). Checks the slot
math and the wiring; flag off keeps the 2-slot layout.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
READER = (ROOT / "render/kernels/dataflow/reader_alpha_blend_mb_devcull.cpp").read_text()
BLEND_HOST = (ROOT / "render/host/blend_device.cpp").read_text()
TRAIN_H = (ROOT / "render/host/blend_train.h").read_text()

FIT = 8192                      # kBucketFit / MB_BUCKET_FIT
BULK_REC_SLOT = (FIT + 1) >> 1  # pages per slot, reader and compute


def test_slot_math():
    # Bulk: 64 B pages, 3 slots; each slot holds FIT 32 B slab records.
    bulk_pages = FIT // 2 * 3
    assert bulk_pages % BULK_REC_SLOT == 0 and bulk_pages // BULK_REC_SLOT == 3
    assert BULK_REC_SLOT * 64 == FIT * 32
    # Op/colour: 32 B pages, 3 slots; each slot holds FIT 16 B entries (dst[g * 4]).
    assert BULK_REC_SLOT * 32 == FIT * 16
    # L1 added vs the 2-slot layout (512 + 512 KB): bulk +256 KB, opc -128 KB, stage +4 KB.
    added = bulk_pages * 64 + bulk_pages * 32 + 64 * 64 - (FIT * 64 + FIT * 64)
    assert added == 132 * 1024


def test_host_wiring():
    assert 'getenv("GSPLAT_TT_TRAIN_OPC_RING3")' in TRAIN_H
    assert "return enabled() && on && !opc_on_writer() && !opc_side();" in TRAIN_H
    assert "constexpr uint32_t kOpcStageCb = 15;" in TRAIN_H
    assert "cb_cfg(CB_BUCKET, 32u, kBucketFit / 2u * 3u, DataFormat::Float32);" in BLEND_HOST
    assert "opc_ring3 ? kBucketFit / 2u * 3u : kBucketFit" in BLEND_HOST
    assert "cb_cfg(gsplat_tt::blend_train::kOpcStageCb, 64u, batch, DataFormat::UInt32);" in BLEND_HOST
    assert 'reader_defines["TRAIN_OPC_RING3"] = "1";' in BLEND_HOST
    # CB 15 is free in the blend program (no other cb_cfg / constexpr id 15).
    ids = re.findall(r"constexpr uint32_t CB_\w+\s*=\s*(\d+);", BLEND_HOST.split("add_mat_part")[0])
    assert "15" not in ids


def test_reader_staging():
    assert "constexpr uint32_t CB_OPC_STAGE = BLEND_CB_BASE + 15;" in READER
    assert "const uint32_t stage = get_write_ptr(CB_OPC_STAGE);" in READER
    assert "const uint32_t stage = side + TRAIN_OPC_STAGE;" in READER
    # The gather reads into the staging base, never at a fixed slot offset.
    assert "stage + n * 64u, 64u);" in READER
    assert "side + TRAIN_OPC_STAGE + n * 64u" not in READER
    assert '#error "TRAIN_OPC_RING3 needs the reader gather' in READER
