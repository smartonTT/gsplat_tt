// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// device_state — Stage-A device-resident shared state for gsplat_tt.
//
// Purpose (amendment-002 phase 2): provide a single point of truth for the
// Tenstorrent MeshDevice and a named-buffer registry that subsequent device
// kernels (cov2d / screen_xy / valid_mask / blend) can read from without
// re-uploading from the host. The handoff doc prescribes keeping means_cam
// resident on device across the project→cov2d→finalize chain; this header
// is the registry that makes that possible.
//
// API contract:
//   get_device()        — lazy-init a shared_ptr<MeshDevice> to device 0.
//                          Returns the same instance across calls.
//   is_initialized()    — has get_device() been called?
//   shutdown()          — release the registered buffers and close the device
//                          if it was initialized. Idempotent. This is the
//                          ONLY place we explicitly close() the MeshDevice;
//                          per-stage shutdown functions just reset their own
//                          local pointers.
//   register_buffer(k,b)— store a shared_ptr<MeshBuffer> under key k. If a
//                          buffer already exists at k, it's replaced.
//   get_buffer(k)       — retrieve buffer by key; returns nullptr if absent.
//   clear_buffers()     — drop all registered buffers (e.g. on N change).

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace tt {
namespace tt_metal {
namespace distributed {
class MeshBuffer;
class MeshDevice;
class MeshCommandQueue;
}  // namespace distributed
}  // namespace tt_metal
}  // namespace tt

namespace gsplat_tt {
namespace device_state {

std::shared_ptr<tt::tt_metal::distributed::MeshDevice> get_device();
bool is_initialized();
void shutdown();

tt::tt_metal::distributed::MeshCommandQueue* command_queue();

void register_buffer(
    const std::string& key,
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> buffer);
std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> get_buffer(
    const std::string& key);
void clear_buffers();

// Host-cached sort→blend handoff (GSPLAT_TT_SORT_BLEND_PIPE): P_kept and the
// padded cull-mask footprint published during sort; lets blend size resident
// buffers without a blocking D2H of sort_P_kept between stages.
void set_sort_blend_pipe_scalars(uint32_t p_kept, uint32_t mask_elems);
bool get_sort_blend_pipe_scalars(uint32_t* p_kept, uint32_t* mask_elems);

// Task #170 (K2 fold): the fused K2 (tile_assign_fused_k2) counted each
// mover's pairs per tile into row 2 * core + mover of `buf` (row_pages 64 B
// pages per row) for the page split it took (P_pub pairs over num_cores cores,
// BRISC's share `permille`). The one-launch sort takes the record (take
// clears it) and skips its count pass when the split is its own. Cleared by
// every tile_assign frame that did not write the rows.
struct K2CountRows {
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> buf;
    uint32_t num_cores = 0;
    uint32_t row_pages = 0;
    uint32_t num_tiles = 0;
    uint32_t P_pub = 0;
    uint32_t permille = 0;
    std::size_t bytes = 0;  // allocated size of buf
};
void set_k2_count_rows(const K2CountRows& rows);
bool take_k2_count_rows(K2CountRows* rows);
void clear_k2_count_rows();

// ROUTE C (GSPLAT_TT_BUCKET_MASK): the microblock-cull contrib_floor + the
// cull_disabled flag, published by render_full_py before the sort call so the
// sort-stage bucket-cull pass can run the SAME SFPU mask math the blend cull
// used (the blend host args never reach the sort driver).
void set_bucket_cull_params(float contrib_floor, bool cull_disabled);
bool get_bucket_cull_params(float* contrib_floor, bool* cull_disabled);

// Phase B (GSPLAT_TT_CHUNK_FUSION): tile grid for AABB in gather scatter.
void set_chunk_fusion_tile_grid(int tiles_x, int tiles_y, int tile_size);
bool get_chunk_fusion_tile_grid(int* tiles_x, int* tiles_y, int* tile_size);

// When true, sort left the publish kernel in-flight; the next blend enqueue
// on the shared CQ must not Finish until cull+blend (or an explicit drain).
void mark_sort_publish_pending();
bool sort_publish_pending();
void clear_sort_publish_pending();

// Detach stage MeshWorkload contexts (leak) so process exit never runs
// ProgramImpl after MeshDevice teardown. Safe to call from tt_device_shutdown.
void leak_stage_contexts_on_exit();

}  // namespace device_state
}  // namespace gsplat_tt
