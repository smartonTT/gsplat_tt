// Syntax-only stub of the tt-metal pinned host memory API (tasks #367/#374/#386).
#pragma once
#include <cstdint>
#include <memory>
#include <optional>
#include "../host_api.hpp"
#include "../host_buffer.hpp"
namespace tt::tt_metal::experimental {
struct NocAddr { uint32_t pcie_xy_enc = 0; uint64_t addr = 0; };
class PinnedMemory { public:
    static std::shared_ptr<PinnedMemory> Create(distributed::MeshDevice&, const distributed::MeshCoordinateRangeSet&,
                                                HostBuffer&, bool map_to_noc = false);
    std::optional<NocAddr> get_noc_addr(int device_id) const;
};
}  // namespace tt::tt_metal::experimental
