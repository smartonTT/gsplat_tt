# t231: TRISC0 decode-ahead staging for blend (GSPLAT_TT_BLEND_DECODE_AHEAD)

Design from docs/blend-trisc1-t230 section 5 (S2a, S2). Kernel:
`render/kernels/compute/alpha_blend_compute_mb.cpp` (decode-ahead block after
`blend_t_readback`, `blend_da_walk`). Host: `render/host/blend_device.cpp` (CB 10 ring,
`BLEND_DECODE_AHEAD` define).

## 1. What it does

t230 found that blend is limited by TRISC1 (MATH) instruction issue, not the SFPU.
Per live record TRISC1 spent about 77 RISC cycles staging the 7 coefficients
(14 SFPLOADI immediates built with shifts, masks and adds) and about 27 cycles
scanning records. TRISC0 (UNPACK) sits idle during the blend loop.

- **Knob 1 (S2a).** TRISC0 walks the same records. For each live record it writes the
  14 SFPLOADI words, already encoded, plus the record's microblock mask to an L1 ring
  (CB 10, 64 slots of 64 B, 4160 B). TRISC1 still scans and does the T readbacks; for
  each live record it takes the next slot, loads the 14 words and pushes them with the
  same SFPU ops in the same order as `BLEND_RAW_STAGE` (t146, md5-identical).
- **Knob 2 (S2).** The ring holds only live records plus markers: a readback marker at
  the same record position as the `g_seen % 512` trigger, and an end-of-subchunk
  marker. TRISC1 reads only the ring: no scan, no T reduce.
- TRISC0 runs its own copy of the T reduce at each readback (same function, same
  packed T tile, fenced), so it masks records exactly as TRISC1 did.
- Counters: produced count (TRISC0) and consumed count (TRISC1) in a 64 B ring header,
  volatile, with `fence` before each read of the other side's counter. TRISC0 fences
  the slot stores before it publishes. The counts run over the whole launch.
- Deadlock: TRISC0 publishes every slot (and the S2 readback marker) before it blocks
  in a readback, and its ring-full wait only needs TRISC1 to consume. The existing
  MATH-to-UNPACK end-of-subchunk ack is kept, so TRISC0 never runs into the next
  subchunk's slab early.
- Mailbox order is unchanged: TRISC0 mailboxes the ring address to TRISC1 once, before
  the first `get_tile_address`.
- TRISC2 (PACK) only takes part in the readbacks, so it steps from one to the next.

Compile-only check (`build.sh`, on the host, no device): all nine TRISC0/1/2 ELFs build
clean at knobs 0, 1 and 2. TRISC1 text: 12924 B (0), 12984 B (1), 11948 B (2).
TRISC1 consume path at knob 2 (disassembly): ring check, 14 `lw` interleaved with
14 `sw` to the instruction buffer and the 17 fixed SFPU ops, two counter stores, then
the unchanged jump dispatch. The TRISC0 producer is about 55 instructions per live
record plus a `fence`.

## 2. Device gate

Pending.
