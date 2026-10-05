#!/usr/bin/env bash
# Syntax-only compile of render kernels / host drivers against minimal stub
# tt-metal, sfpi and dataflow headers (task #99). For no-device tasks on a Mac:
# catches C++ errors in kernel and host code before a device run. It does NOT
# validate tt-metal / sfpi API details (overloads, register pressure).
#   tests/syntax_stub/check.sh            (from the repo root)
set -u
cd "$(dirname "$0")/../.."
ST=tests/syntax_stub
CXX=${CXX:-c++}
fail=0
chk() { local what=$1; shift; if out=$("$CXX" -fsyntax-only "$@" 2>&1 | grep -E "error" ); then echo "FAIL $what"; echo "$out" | head -10; fail=1; else echo "ok   $what"; fi; }
DF="-std=c++17 -I$ST -Irender/kernels/dataflow"
for f in writer_pfwc_vis.cpp gather_vis_scan.cpp gather_vis_scatter.cpp gather_visible_scatter.cpp \
         tile_assign_bbox.cpp writer_pfwc.cpp; do chk "$f" $DF render/kernels/dataflow/$f; done
chk "tile_assign_scatter.cpp" $DF render/kernels/dataflow/tile_assign_scatter.cpp
chk "tile_assign_scatter.cpp TA_K2_AABB" $DF -DTA_K2_AABB=1 render/kernels/dataflow/tile_assign_scatter.cpp
chk "reader_pfwc.cpp" $DF render/kernels/dataflow/reader_pfwc.cpp
chk "sort_bin_onelaunch.cpp" $DF render/kernels/dataflow/sort_bin_onelaunch.cpp
chk "sort_bin_onelaunch.cpp EMIT_PUBOC" $DF -DEMIT_PUBOC=1u render/kernels/dataflow/sort_bin_onelaunch.cpp
chk "sort_bin_onelaunch.cpp v2 PB8 RING8" $DF -DEMIT_PUBOC=1u -DOL_PB=8u -DOL_RING=8u -DOL_WIN_PAGES=1024u render/kernels/dataflow/sort_bin_onelaunch.cpp
chk "sort_bin_onelaunch.cpp v2 PB16 RING2 no PUBOC" $DF -DOL_PB=16u -DOL_RING=2u render/kernels/dataflow/sort_bin_onelaunch.cpp
chk "sort_bin_onelaunch.cpp v2 PB8 RING8 BREC_HALF256" $DF -DEMIT_PUBOC=1u -DOL_PB=8u -DOL_RING=8u -DOL_BREC_BULK=1 -DOL_BREC_HALF=256u render/kernels/dataflow/sort_bin_onelaunch.cpp
chk "sort_bin_onelaunch.cpp v2 PB8 RING8 BREC_BULK=0" $DF -DEMIT_PUBOC=1u -DOL_PB=8u -DOL_RING=8u -DOL_BREC_BULK=0 -DOL_BREC_HALF=128u render/kernels/dataflow/sort_bin_onelaunch.cpp
chk "sort_bin_onelaunch.cpp v2 PB8 RING8 BREC_HALF256 EMIT_TOWN" $DF -DEMIT_PUBOC=1u -DOL_PB=8u -DOL_RING=8u -DOL_BREC_BULK=1 -DOL_BREC_HALF=256u -DOL_EMIT_TOWN=1 render/kernels/dataflow/sort_bin_onelaunch.cpp
chk "sort_bin_onelaunch.cpp v2 EMIT_TOWN EMIT_PROF" $DF -DEMIT_PUBOC=1u -DOL_PB=8u -DOL_RING=8u -DOL_BREC_BULK=1 -DOL_BREC_HALF=256u -DOL_EMIT_TOWN=1 -DOL_EMIT_PROF=1 render/kernels/dataflow/sort_bin_onelaunch.cpp
chk "sort_subchunk_materialize.cpp" $DF render/kernels/dataflow/sort_subchunk_materialize.cpp
chk "sort_subchunk_materialize.cpp SORT_ONELAUNCH" $DF -DSORT_ONELAUNCH=1 render/kernels/dataflow/sort_subchunk_materialize.cpp
chk "sort_subchunk_materialize.cpp OL_MAT_SELECT" $DF -DSORT_ONELAUNCH=1 -DOL_MAT_SELECT=1 -DOL_MAT_PART=4096u render/kernels/dataflow/sort_subchunk_materialize.cpp
chk "sort_subchunk_materialize.cpp OL_MAT_SELECT FUSE_CULL" $DF -DSORT_ONELAUNCH=1 -DOL_MAT_SELECT=1 -DFUSE_CULL=1 -DFUSE_CULL_DEPTH=2 render/kernels/dataflow/sort_subchunk_materialize.cpp
chk "sort_subchunk_materialize.cpp SORT_ONELAUNCH MATBLEND_FUSE" $DF -DSORT_ONELAUNCH=1 -DMATBLEND_FUSE=1 render/kernels/dataflow/sort_subchunk_materialize.cpp
chk "reader_alpha_blend_mb_devcull.cpp MATBLEND_FUSE" $DF -DMB_BUCKET_FIT=8192u -DMATBLEND_FUSE=1 -DBLEND_CB_BASE=32 -DBLEND_CTA_BASE=12 render/kernels/dataflow/reader_alpha_blend_mb_devcull.cpp
chk "writer_alpha_blend.cpp BLEND_CB_BASE" $DF -DBLEND_CB_BASE=32 -DBLEND_CTA_BASE=12 render/kernels/dataflow/writer_alpha_blend.cpp
FZ="-DSORT_ONELAUNCH=1 -DFUSE_CULL=1 -DFUSE_CULL_DEPTH=2u -DMATCULL_FOLD=0 -DMATBLEND_FUSE=1 -DBLEND_CB_BASE=32 -DBLEND_CTA_BASE=24"
chk "matblend_ncrisc.cpp" $DF $FZ -DMB_BUCKET_FIT=8192u -DMB_TILE_L1_MASKS=1 render/kernels/dataflow/matblend_ncrisc.cpp
chk "matblend_brisc.cpp" $DF $FZ -DMAT_CB_BASE=16 render/kernels/dataflow/matblend_brisc.cpp
chk "writer_pfwc_fuse.cpp" $DF render/kernels/dataflow/writer_pfwc_fuse.cpp
chk "writer_pfwc_fuse.cpp EMIT_PUBOC" $DF -DEMIT_PUBOC=1 render/kernels/dataflow/writer_pfwc_fuse.cpp
chk "writer_pfwc_fuse.cpp FUSE_ABL=7" $DF -DFUSE_ABL=7u render/kernels/dataflow/writer_pfwc_fuse.cpp
chk "gather_vis_scatter.cpp EMIT_PUBOC" $DF -DEMIT_PUBOC=1 render/kernels/dataflow/gather_vis_scatter.cpp
chk "tile_assign_scatter_seg.cpp" $DF render/kernels/dataflow/tile_assign_scatter_seg.cpp
chk "tile_assign_scatter_seg.cpp TA_CB_OFFSET" $DF -DTA_CB_OFFSET=16 render/kernels/dataflow/tile_assign_scatter_seg.cpp
chk "tile_assign_scatter_seg.cpp K2_DIET" $DF -DK2_DIET=1 render/kernels/dataflow/tile_assign_scatter_seg.cpp
chk "tile_assign_scatter_seg.cpp K2_DIET TA_CB_OFFSET" $DF -DK2_DIET=1 -DTA_CB_OFFSET=16 render/kernels/dataflow/tile_assign_scatter_seg.cpp
chk "reader_pfwc.cpp PFWC_VIS" $DF -DPFWC_VIS=1 render/kernels/dataflow/reader_pfwc.cpp
for r in 0 1; do
  chk "writer_pfwc_split.cpp ROLE $r" $DF -DWSPLIT_ROLE=$r render/kernels/dataflow/writer_pfwc_split.cpp
  chk "writer_pfwc_split.cpp ROLE $r EMIT_PUBOC" $DF -DWSPLIT_ROLE=$r -DEMIT_PUBOC=1 render/kernels/dataflow/writer_pfwc_split.cpp
  chk "writer_pfwc_split.cpp ROLE $r EMIT_PUBOC PFWC_STEPCYC" $DF -DWSPLIT_ROLE=$r -DEMIT_PUBOC=1 -DPFWC_STEPCYC=1 render/kernels/dataflow/writer_pfwc_split.cpp
  chk "writer_pfwc_split.cpp ROLE $r EMIT_PUBOC PFWC_RD_COLS PFWC_STEPCYC" $DF -DWSPLIT_ROLE=$r -DEMIT_PUBOC=1 -DPFWC_RD_COLS=1 -DPFWC_STEPCYC=1 render/kernels/dataflow/writer_pfwc_split.cpp
done
CP="-std=c++20 -DTRISC_MATH=1 -I$ST -I$ST/api -Wno-unknown-attributes"
chk "project_pfwc_compute.cpp" $CP render/kernels/compute/project_pfwc_compute.cpp
chk "project_pfwc_compute.cpp PFWC_VIS" $CP -DPFWC_VIS=1 render/kernels/compute/project_pfwc_compute.cpp
chk "project_pfwc_compute.cpp PFWC_VIS PFWC_PRECULL" $CP -DPFWC_VIS=1 -DPFWC_PRECULL=1 render/kernels/compute/project_pfwc_compute.cpp
chk "project_pfwc_compute.cpp PFWC_VIS PFWC_PRECULL PRECULL_PC" $CP -DPFWC_VIS=1 -DPFWC_PRECULL=1 -DPRECULL_PC=1 render/kernels/compute/project_pfwc_compute.cpp
for tr in TRISC_UNPACK TRISC_MATH TRISC_PACK; do
  chk "sort_ol_town_compute.cpp $tr" ${CP/-DTRISC_MATH=1/} -D$tr=1 -DOL_RING=8u render/kernels/compute/sort_ol_town_compute.cpp
  chk "sort_ol_town_compute.cpp $tr EMIT_PROF" ${CP/-DTRISC_MATH=1/} -D$tr=1 -DOL_RING=8u -DOL_EMIT_PROF=1 render/kernels/compute/sort_ol_town_compute.cpp
done
chk "project_pfwc_compute.cpp PFWC_VIS PFWC_PRECULL PRECULL_PC PFWC_WSPLIT" $CP -DPFWC_VIS=1 -DPFWC_PRECULL=1 -DPRECULL_PC=1 -DPFWC_WSPLIT=1 render/kernels/compute/project_pfwc_compute.cpp
chk "project_pfwc_compute.cpp PFWC_COVCAM_SFPU" $CP -DPFWC_COVCAM_SFPU=1 render/kernels/compute/project_pfwc_compute.cpp
chk "project_pfwc_compute.cpp PFWC_COVCAM_SFPU PFWC_VIS PFWC_PRECULL" $CP -DPFWC_COVCAM_SFPU=1 -DPFWC_VIS=1 -DPFWC_PRECULL=1 render/kernels/compute/project_pfwc_compute.cpp
chk "project_pfwc_compute.cpp PFWC_COVCAM_SFPU PFWC_VIS PFWC_PRECULL PFWC_WSPLIT" $CP -DPFWC_COVCAM_SFPU=1 -DPFWC_VIS=1 -DPFWC_PRECULL=1 -DPRECULL_PC=1 -DPFWC_WSPLIT=1 render/kernels/compute/project_pfwc_compute.cpp
for tr in TRISC_UNPACK TRISC_MATH TRISC_PACK; do
  chk "project_pfwc_compute.cpp $tr PFWC_COV2D_SFPU" ${CP/-DTRISC_MATH=1/} -D$tr=1 -DPFWC_COV2D_SFPU=1 -DPFWC_COVCAM_SFPU=1 render/kernels/compute/project_pfwc_compute.cpp
  chk "project_pfwc_compute.cpp $tr PFWC_COV2D_SFPU PFWC_VIS PFWC_PRECULL PFWC_WSPLIT STEPCYC" ${CP/-DTRISC_MATH=1/} -D$tr=1 -DPFWC_COV2D_SFPU=1 -DPFWC_COVCAM_SFPU=1 -DPFWC_VIS=1 -DPFWC_PRECULL=1 -DPRECULL_PC=1 -DPFWC_WSPLIT=1 -DPFWC_STEPCYC=1 -DPFWC_STEPRISC=9 render/kernels/compute/project_pfwc_compute.cpp
done
# matblend_compute.cpp is not stub-checked: the stubs lack the blend compute
# headers (fill.h, sfpu exp), as for alpha_blend_compute_mb.cpp; the device build checks it.
HS="-std=c++20 -I$ST -Irender/host -Isrc -Wno-mismatched-tags"
for f in pfwc_device.cpp gather_visible_device.cpp tile_assign_device.cpp sort_device.cpp; do chk "$f" $HS render/host/$f; done
PB=$(python3 -c "import pybind11; print(pybind11.get_include())" 2>/dev/null)
PYH=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['include'])" 2>/dev/null)
[ -f "$PYH/Python.h" ] || PYH=$(ls -d /Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/*/Headers 2>/dev/null | head -1)
if [ -n "$PB" ] && [ -f "$PYH/Python.h" ]; then chk "render.cpp" $HS -I"$PB" -I"$PYH" render/host/render.cpp
else echo "skip render.cpp (no pybind11 / Python.h)"; fi
exit $fail
