// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gsplat_tt/device_state.h"

#include <mutex>
#include <unordered_map>

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
};

State& state() {
    static State s;
    return s;
}

}  // namespace

std::shared_ptr<tt::tt_metal::distributed::MeshDevice> get_device() {
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
    if (s.mesh_device) {
        s.mesh_device->close();
        s.mesh_device.reset();
    }
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

}  // namespace device_state
}  // namespace gsplat_tt
