// Syntax-only stub of the tt-metal HostBuffer (pinned output, blend_device.cpp).
#pragma once
#include <cstddef>
#include <memory>
#include "host_api.hpp"
namespace tt::stl { template <class T> struct Span { Span(T*, std::size_t) {} }; }
namespace tt::tt_metal {
struct MemoryPin { MemoryPin() = default; template <class P> explicit MemoryPin(const P&) {} };
struct HostBuffer { template <class T> HostBuffer(tt::stl::Span<T>, MemoryPin) {} };
}  // namespace tt::tt_metal
