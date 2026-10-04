set -u
T=/localdev/smarton/tt-metal; W=/localdev/smarton/gstt2-t186-scratch; mkdir -p $W; cd $W
CXX=/opt/tenstorrent/sfpi/compiler/bin/riscv-tt-elf-g++; OD=/opt/tenstorrent/sfpi/compiler/bin/riscv-tt-elf-objdump
[ -x $CXX ] || CXX=$T/runtime/sfpi/compiler/bin/riscv-tt-elf-g++
INC="-I$W -I$T -I$T/tt_metal -I$T/tt_metal/include -I$T/tt_metal/hw/inc -I$T/tt_metal/hw/inc/internal/tt-1xx -I$T/tt_metal/hw/inc/internal/tt-1xx/blackhole -I$T/tt_metal/hw/inc/internal/tt-1xx/blackhole/blackhole_defines -I$T/tt_metal/hw/inc/internal/tt-1xx/blackhole/noc -I$T/tt_metal/hw/ckernels/blackhole/metal/common -I$T/tt_metal/hw/ckernels/blackhole/metal/llk_io -I$T/tt_metal/tt-llk/common -I$T/tt_metal/hw/inc/debug -I$T/tt_metal/tt-llk/tt_llk_blackhole/common/inc -I$T/tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc -I$T/tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib -I$T/tt_metal/api -I$T/tt_metal/hostdevcommon/api -I$T/tt_metal/hw/firmware/src/tt-1xx -I$T/tt_metal/hw/inc/api -I$T/tt_metal/hw/inc/internal"
for VAR in ${VARS:-base}; do
 for R in BRISC NCRISC; do for P in 1 0; do for F in 1 0; do
  o=$W/o_${VAR}_${R}_P${P}_F${F}
  $CXX -mcpu=tt-bh -std=c++17 -O3 -fno-exceptions -fno-rtti -fno-use-cxa-atexit -ffast-math -fstack-usage -Wno-deprecated-declarations \
   -DDATA_FORMATS_DEFINED=1 -DROUTING_FW_ENABLED -DKERNEL_COMPILE_TIME_ARGS=2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2 -fno-tree-loop-distribute-patterns -DTENSIX_FIRMWARE -DARCH_BLACKHOLE -DCOMPILE_FOR_$R -DPROCESSOR_INDEX=$([ $R = BRISC ] && echo 0 || echo 1) -DKERNEL_BUILD -DLOCAL_MEM_EN=0 -DNOC_INDEX=0 -DNOC_MODE=0 -DNUM_DRAM_BANKS=8 -DNUM_L1_BANKS=140 -DIS_NOT_POW2_NUM_L1_BANKS=1 -DIS_NOT_POW2_NUM_DRAM_BANKS=0 -DLOG_BASE_2_OF_NUM_DRAM_BANKS=3 -DPCIE_NOC_X=19 -DPCIE_NOC_Y=24 -DDISPATCH_MESSAGE_ADDR=0 -DPCIE_NOC1_X=0 -DPCIE_NOC1_Y=0 \
   -DEMIT_PUBOC=1u -DOL_PB=4u -DOL_RING=2u -DOL_RING_TILES=1024u -DOL_EMIT_PROF=$P -DOL_EMIT_FOLD=$F $INC -I$W/$VAR -c $T/tt_metal/hw/firmware/src/tt-1xx/$( [ $R = BRISC ] && echo brisck || echo ncrisck ).cc -o $o.o > $o.err 2>&1
  echo "$VAR $R P=$P F=$F rc=$? $(grep -E 'kernel_main' $o.su 2>/dev/null)"
 done; done; done
done
