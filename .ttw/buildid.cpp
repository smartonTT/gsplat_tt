id=58
sha=63249dc
ts=2026-06-04T23:52:42-0700
desc=iter 102: M3: move 32-bit microblock mask into slab word3 (dead depth key); cull writer does aligned per-batch 64B page RMW of sort_subchunk_payload word3 (strided 4B misaligned both NoC ends -> 21.85dB, page RMW -> anchor); blend reads mask from rec[3]; deleted DRAM cull_masks buffer, ensure_resident_buffers, CB_BMASK_BULK + the blend reader's cull_masks/cull_mask_base args/accessors and bulk mask load
bin=df835eea49ff1991
