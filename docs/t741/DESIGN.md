# t741: fp32 op/colour side slab (design + status)

## Finding
PFWC_REC32 is on by default (render/host/env_config.h:311). Its 32 B blendrec
`[a,b,c,mx,my,u01,u23,dep]` holds no fp32 op/colour, so the #730 premise
("blendrec words 5..8 are fp32 op/colour") only holds with GSPLAT_TT_PFWC_REC32=0:
the 64 B blendrec then has r[5..8] = op, cr, cg, cb (writer_pfwc_split.cpp:468-471).

## Design (source-slot stamp; materialize unchanged)
Knob: `blend_train::opc_side()` (GSPLAT_TT_TRAIN_OPC_SIDE=1, train mode, not with
TRAIN_OPC_WRITER), render/host/blend_train.h. Host must throw if side && rec32.

1. TOWN TRISC (sort_ol_town_compute.cpp process(), ~l.118-163), under define
   TRAIN_OPC_SIDE (static_assert !PFWC_REC32): on offset change also load
   cp[5..8]; per record write them to
   `sring + (t*D + (c&(D-1)))*16`, sring = h[H_RING] + OL_RING_TILES*D*32 + OL_RING_TILES*4
   (pass OL_RING_TILES + TRAIN_OPC_SIDE in tdef, sort_device.cpp ~1086);
   `d[6] = (opr & 0xFFFF) | ((c+1) << 16)` when c+1 <= 0xFFFF, else opr & 0xFFFF.
   Cull reads w6 & 0xffff only (mat_cull_compute.cpp:164, materialize 224/297);
   train compute ignores w6.
2. Mover (sort_bin_onelaunch.cpp flush_run l.638): also
   `noc_async_write(side_l1 + (t*D + (s0&(D-1)))*16, get_noc_addr(t*tile_pages + s0/64, side_gen) + (s0%64)*16, n*16)`
   with `InterleavedAddrGen<true> side_gen{get_arg_val(30), 1024}`, side_l1 as sring.
   Same low-6-bit alignment on both ends. Coordinator noc_xy args move 30+r -> 31+r
   (l.458); host pushes the side address as arg 30 at both emit arg sites
   (sort_device.cpp ~2522 and ~3594, before the coordinator insert).
3. Host emit: CB 13 ring_bytes += kOneLaunchRingTiles*depth*16 in side mode
   (+128 KB/mover; movers are 567680 B now, check L1 fits); side buffer
   num_tiles*g_ol_tile_cap*16 B, 1 KB pages, grown with buf_ol_bucket in
   ensure_onelaunch_buffers (l.2185), register "train_opc_side";
   `blend_train::side_tile_cap() = ol_frame && town ? g_ol_tile_cap : 0` per frame.
4. Blend host (blend_device.cpp ~982): after arg 28 push arg 29 = side address
   (0 when side_tile_cap()==0) and arg 30 = tile cap; define TRAIN_OPC_SIDE.
5. Reader (reader_alpha_blend_mb_devcull.cpp gather block l.483-553): scan the
   L_sub records' stamps `rec[g*8+6]>>16`; any 0 or side addr 0 -> existing gather.
   Else slots c = stamp-1, pages tile_id*(cap/64) + cmin/64 .. cmax/64, windows of
   128 pages (128 KB staging at side+TRAIN_OPC_STAGE): read 1 KB pages, barrier,
   copy stg[(c - pa*64)*4 ..] to dst[g*4 ..] for the window's records.
   Check that the reader's tile_id is the bucket tile index materialize uses
   (page0 = tile_id*(ol_tile_cap/64)).

A/B: arms on (default) vs side (GSPLAT_TT_PFWC_REC32=0 GSPLAT_TT_TRAIN_OPC_SIDE=1)
+ A/A + rec32off-only (isolates the REC32 cost), dumps first (md5 3fada021,
check_dump), ab_jit_check, hero screenshot. Reuse docs/t734/drive.sh + remote.sh.
