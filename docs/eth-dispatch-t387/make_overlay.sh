#!/bin/bash
# t387: tt-metal 437bc366 lists 14 ETH dispatch cores ([0,0]..[0,13]) for every Blackhole
# product in core_descriptors/blackhole_140_arch_eth_dispatch.yaml, but the bh-30 p150b has
# ETH harvesting mask 0x120 (12 live ETH cores, logical 0..11). Opening with
# DispatchCoreType::ETH then throws "No core coordinate found at location: (0, 12, ETH, LOGICAL)"
# in L1BankingAllocator::generate_config. tt-metal reads the core descriptors from
# TT_METAL_RUNTIME_ROOT, so this builds a symlink overlay of the (read-only, shared) viewer
# tt-metal whose only real file is a copy of that yaml with the dispatch list cut to the
# live ETH count. The tt-metal checkout itself is not changed.
#   bash make_overlay.sh <tt-metal dir> <overlay dir> <live eth cores>   (on the box)
set -eu
SRC=${1:?tt-metal}; OV=${2:?overlay}; N=${3:-12}
rm -rf "$OV"; mkdir -p "$OV/tt_metal/core_descriptors"
for e in "$SRC"/* "$SRC"/.[!.]*; do [ -e "$e" ] && [ "$(basename "$e")" != tt_metal ] && ln -s "$e" "$OV/"; done
for e in "$SRC"/tt_metal/*; do [ "$(basename "$e")" != core_descriptors ] && ln -s "$e" "$OV/tt_metal/"; done
for e in "$SRC"/tt_metal/core_descriptors/*; do ln -s "$e" "$OV/tt_metal/core_descriptors/"; done
Y=tt_metal/core_descriptors/blackhole_140_arch_eth_dispatch.yaml
rm "$OV/$Y"
list=$(python3 -c "print(', '.join('[0, %d]' % i for i in range($N)))")
sed -E "s/^( *)\[\[0, 0\], \[0, 1\].*\[0, 13\]\]$/\1[$list]/" "$SRC/$Y" > "$OV/$Y"
echo "overlay $OV: $(grep -c "\[0, $((N - 1))\]\]" "$OV/$Y") dispatch lists cut to $N ETH cores"
diff "$SRC/$Y" "$OV/$Y" | head -4
