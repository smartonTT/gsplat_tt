// Syntax-only stub of the tt-metal dataflow API (task #99 local checks).
#pragma once
#include <cstdint>
#define DeviceZoneScopedN(name) (void)name
template <typename T> T get_arg_val(int);
uint32_t get_write_ptr(uint32_t cb);
uint32_t get_read_ptr(uint32_t cb);
uint32_t get_tile_size(uint32_t cb);
void cb_reserve_back(uint32_t, uint32_t);
void cb_push_back(uint32_t, uint32_t);
void cb_wait_front(uint32_t, uint32_t);
void cb_pop_front(uint32_t, uint32_t);
template <uint32_t Base> struct TensorAccessorArgs {
    static constexpr uint32_t next_compile_time_args_offset() { return Base + 1; }
};
template <class A> struct TensorAccessor {
    TensorAccessor(const A&, uint32_t, uint32_t) {}
};
template <class A> TensorAccessor(const A&, uint32_t, uint32_t) -> TensorAccessor<A>;
template <bool D> struct InterleavedAddrGen { uint32_t bank_base_address; uint32_t page_size; };
template <class A> uint64_t get_noc_addr(uint32_t page, const TensorAccessor<A>&, uint32_t off = 0);
template <bool D> uint64_t get_noc_addr(uint32_t page, const InterleavedAddrGen<D>&, uint32_t off = 0);
uint64_t get_noc_addr(uint32_t x, uint32_t y, uint32_t addr);
void noc_async_read(uint64_t src, uint32_t dst, uint32_t size);
void noc_async_write(uint32_t src, uint64_t dst, uint32_t size);
template <class A> void noc_async_read_tile(uint32_t id, const TensorAccessor<A>&, uint32_t dst);
template <class A> void noc_async_write_tile(uint32_t id, const TensorAccessor<A>&, uint32_t src);
void noc_async_read_barrier();
void noc_async_write_barrier();
void noc_async_writes_flushed();
