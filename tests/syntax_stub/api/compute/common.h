#pragma once
#include <cstdint>
// Thread sections (task #202): -DTRISC_UNPACK / -DTRISC_PACK select those, else MATH.
#if defined(TRISC_UNPACK) || defined(TRISC_PACK)
#define MATH(x)
#else
#define MATH(x) x
#endif
#ifdef TRISC_UNPACK
#define UNPACK(x) x
#else
#define UNPACK(x)
#endif
#ifdef TRISC_PACK
#define PACK(x) x
#else
#define PACK(x)
#endif
struct LocalCBInterface { uint32_t fifo_size, fifo_limit, fifo_page_size, fifo_num_pages, fifo_rd_ptr, fifo_wr_ptr; };
LocalCBInterface& get_local_cb_interface(uint32_t cb);
namespace ckernel {
enum class ThreadId : uint32_t { UnpackThreadId = 0, MathThreadId = 1, PackThreadId = 2 };
void mailbox_write(ThreadId, uint32_t);
uint32_t mailbox_read(ThreadId);
}  // namespace ckernel
template <typename T> T get_arg_val(int);
void cb_reserve_back(uint32_t, uint32_t); void cb_push_back(uint32_t, uint32_t);
void cb_wait_front(uint32_t, uint32_t); void cb_pop_front(uint32_t, uint32_t);
void tile_regs_acquire(); void tile_regs_commit(); void tile_regs_wait(); void tile_regs_release();
void pack_tile(uint32_t, uint32_t); void copy_tile_to_dst_init_short(uint32_t); void copy_tile(uint32_t, uint32_t, uint32_t);
void init_sfpu(uint32_t, uint32_t); void add_binary_tile_init(); void mul_binary_tile_init();
void recip_tile_init(); void sqrt_tile_init(); void relu_tile_init(); void rounding_op_tile_init();
void mul_unary_tile(uint32_t, uint32_t); void add_unary_tile(uint32_t, uint32_t);
void add_binary_tile(uint32_t, uint32_t, uint32_t); void mul_binary_tile(uint32_t, uint32_t, uint32_t);
void recip_tile(uint32_t); void sqrt_tile(uint32_t); void relu_tile(uint32_t); void ceil_tile(uint32_t);
