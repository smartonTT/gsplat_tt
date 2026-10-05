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
    uint64_t get_noc_addr(uint32_t page, uint32_t off = 0) const;  // task #280 (writer_alpha_blend)
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
// Semaphores (task #106).
#define tt_l1_ptr
uint32_t get_semaphore(uint32_t id);
void noc_semaphore_wait(volatile uint32_t* sem, uint32_t val);
void noc_semaphore_set(volatile uint32_t* sem, uint32_t val);
void noc_semaphore_inc(uint64_t addr, uint32_t incr);
void noc_async_atomic_barrier();
void invalidate_l1_cache();
#ifndef tt_reg_ptr
#define tt_reg_ptr  // task #202 (EMIT_PROF variants)
#endif
#ifndef RISCV_DEBUG_REG_WALL_CLOCK_L
#define RISCV_DEBUG_REG_WALL_CLOCK_L 0xFFB121F0u
#endif
void DeviceTimestampedData(const char*, uint64_t);  // task #202 (EMIT_PROF variants)
// Non-blocking CB / NoC polls (task #207).
bool cb_pages_reservable_at_back(int32_t operand, int32_t num_pages);
bool cb_pages_available_at_front(int32_t operand, int32_t num_pages);
extern uint8_t noc_index;
bool ncrisc_noc_reads_flushed(uint32_t noc);
// NoC register reads (task #232: the physical NoC0 column).
uint32_t NOC_CMD_BUF_READ_REG(uint32_t noc, uint32_t buf, uint32_t addr);
#ifndef NOC_NODE_ID
#define NOC_NODE_ID 0x44u
#endif
#ifndef NOC_NODE_ID_MASK
#define NOC_NODE_ID_MASK ((((uint64_t)0x1) << 6) - 1)
#endif
// NoC atomic claim counter (task #280: reader_alpha_blend_mb_devcull).
constexpr uint32_t noc_mode = 0, write_at_cmd_buf = 0, NOC_UNICAST_WRITE_VC = 1;
template <uint32_t Mode, bool RetAddr>
void noc_fast_atomic_increment(uint8_t noc, uint32_t cmd_buf, uint64_t addr, uint32_t vc, uint32_t incr,
                               uint32_t wrap, bool linked, bool posted, uint32_t ret_addr);
