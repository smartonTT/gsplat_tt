// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gsplat_tt/device_state.h"

#include <cstdlib>
#include <mutex>
#include <unordered_map>

#include "gsplat_tt/blend.h"
#include "gsplat_tt/gather_visible.h"
#include "gsplat_tt/pfwc.h"
#include "gsplat_tt/project.h"
#include "gsplat_tt/sort.h"
#include "gsplat_tt/tile_assign.h"

#include <tt-metalium/distributed.hpp>

namespace gsplat_tt {
namespace device_state {

namespace {

struct State {
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> mesh_device;
    std::unordered_map<
        std::string,
        std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>>
        buffers;
    std::mutex mu;
    bool sort_publish_pending = false;
    bool sort_pipe_scalars_valid = false;
    uint32_t sort_pipe_p_kept = 0;
    uint32_t sort_pipe_mask_elems = 0;
};

State& state() {
    static State s;
    return s;
}

}  // namespace

// Leak compiled MeshWorkload contexts at process exit so ProgramImpl is never
// destroyed after MeshDevice teardown (SIGSEGV). Do not MeshDevice::close() here;
// tt_metal's own atexit owns ShmResourceTracker (explicit close -> double-free).
void leak_stage_contexts_on_exit() {
    gsplat_tt::device_shutdown();
    gsplat_tt::project_device_shutdown();
    gsplat_tt::pfwc_device_shutdown();
    gsplat_tt::tile_assign_device_shutdown();
    gsplat_tt::sort_device_shutdown();
    gsplat_tt::gather_visible_device_shutdown();
}

namespace {

void register_exit_leak_once() {
    static std::once_flag once;
    std::call_once(once, []() {
        std::atexit(+[]() { leak_stage_contexts_on_exit(); });
    });
}

}  // namespace

std::shared_ptr<tt::tt_metal::distributed::MeshDevice> get_device() {
    register_exit_leak_once();
    auto& s = state();
    std::lock_guard<std::mutex> lock(s.mu);
    if (!s.mesh_device) {
        constexpr int device_id = 0;
        s.mesh_device = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(device_id);
    }
    return s.mesh_device;
}

bool is_initialized() {
    auto& s = state();
    std::lock_guard<std::mutex> lock(s.mu);
    return static_cast<bool>(s.mesh_device);
}

tt::tt_metal::distributed::MeshCommandQueue* command_queue() {
    auto dev = get_device();
    return &dev->mesh_command_queue();
}

void shutdown() {
    auto& s = state();
    std::lock_guard<std::mutex> lock(s.mu);
    s.buffers.clear();
    // Do not MeshDevice::close() here: leaked stage contexts still hold
    // MeshWorkload programs that tt_metal must not destroy after close().
    // Drop our registry only; OS reclaims the device on process exit.
}

void register_buffer(
    const std::string& key,
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> buffer) {
    auto& s = state();
    std::lock_guard<std::mutex> lock(s.mu);
    s.buffers[key] = std::move(buffer);
}

std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> get_buffer(
    const std::string& key) {
    auto& s = state();
    std::lock_guard<std::mutex> lock(s.mu);
    auto it = s.buffers.find(key);
    if (it == s.buffers.end()) {
        return nullptr;
    }
    return it->second;
}

void clear_buffers() {
    auto& s = state();
    std::lock_guard<std::mutex> lock(s.mu);
    s.buffers.clear();
}

void set_sort_blend_pipe_scalars(uint32_t p_kept, uint32_t mask_elems) {
    auto& s = state();
    std::lock_guard<std::mutex> lock(s.mu);
    s.sort_pipe_scalars_valid = true;
    s.sort_pipe_p_kept = p_kept;
    s.sort_pipe_mask_elems = mask_elems;
}

bool get_sort_blend_pipe_scalars(uint32_t* p_kept, uint32_t* mask_elems) {
    auto& s = state();
    std::lock_guard<std::mutex> lock(s.mu);
    if (!s.sort_pipe_scalars_valid) {
        return false;
    }
    if (p_kept) {
        *p_kept = s.sort_pipe_p_kept;
    }
    if (mask_elems) {
        *mask_elems = s.sort_pipe_mask_elems;
    }
    return true;
}

void mark_sort_publish_pending() {
    auto& s = state();
    std::lock_guard<std::mutex> lock(s.mu);
    s.sort_publish_pending = true;
}

bool sort_publish_pending() {
    auto& s = state();
    std::lock_guard<std::mutex> lock(s.mu);
    return s.sort_publish_pending;
}

void clear_sort_publish_pending() {
    auto& s = state();
    std::lock_guard<std::mutex> lock(s.mu);
    s.sort_publish_pending = false;
}

}  // namespace device_state
}  // namespace gsplat_tt
