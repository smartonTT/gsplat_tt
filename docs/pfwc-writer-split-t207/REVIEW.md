# t207 pfwc writer split: code-read review (task #220)

Reviewed: `f495e91..4063ed7` on `smarton/tt-project-opt` (the reviewed files are unchanged at
the branch tip). No device run; the device A/B task does the md5 smoke.

**Verdict: may proceed to the device A/B.** No blocking problems found. Three small hardening
notes below, none needed for the first device run.

## Checks run (Mac)

- `tests/unit/run_cpp.sh tests/unit/test_pfwc_wsplit.cpp`: PASS.
- `tests/syntax_stub/check.sh`: all ok, including both writer roles (plain, `EMIT_PUBOC`,
  `EMIT_PUBOC`+`PFWC_STEPCYC`) and the compute kernel with `PFWC_WSPLIT`.

## Findings by focus area

**Mailbox ordering.** Correct. Senders store the message, `fence`, then set the semaphore
(PREFIX at `writer_pfwc_split.cpp:330-333`, OPEN at `:390-392`). Receivers spin in
`wait_until`, which calls `invalidate_l1_cache()` before every flag read and once more before
the message reads. PREFIX words (0, 1) and OPEN words (16..63) do not overlap, so an OPEN that
lands before the receiver reads PREFIX (flag goes PREFIX -> OPEN) leaves m/pr intact; the
`*f != F_EMPTY` test accepts both states. The receiver expects OPEN iff `m0 % 16 != 0`, and
the sender sends it iff its end `pg.slot != 0`; both equal `(m_prev + vc) % 16`, so they agree.

**Deadlock.** None found. Every wait depends only on earlier chunks: chunk k needs compute
output k (compute is in order, so k-1 is already out), PREFIX/OPEN from k-1, and EMPTY of
slot(k+1), last used by chunk k-3 and emptied when that chunk ends. Compute needs input k
(NCRISC's reader, polled in every NCRISC wait) and a free slot in its parity set (chunk k-4
popped). Slot reuse is every 4 chunks (`slot_index` alternates per receiver), so there is no
cycle. The NCRISC tail `while (rd_k < num_chunks) rd_poll()` only depends on BRISC popping
even chunks, which never waits on NCRISC past its last sent OPEN. `num_chunks` 0 and 1 are
handled (BRISC writes counts(0, 0); NCRISC only reads).

**NCRISC non-blocking reader.** Matches `reader_pfwc.cpp` with `PFWC_VIS` and the fuse
stride: same 10 CBs (0..8, 30), tile `chunk_start + k * stride`, all-or-nothing reserve, push
after `ncrisc_noc_reads_flushed`. Shared read counters with the color reads only delay a
push; the `noc_async_read_barrier` at `:274` also completes pending reader reads, and the next
poll pushes them. Input depth 3 is applied to all 10 inputs (`in_depth`).

**Head-page merge and tail.** Equal to `writer_pfwc_fuse.cpp`. A chunk that starts mid-page
stages that page as the head; `finish` merges words `[0, head_s)` from OPEN and writes the
page only if the chunk filled it, otherwise `export_open` forwards it (after the merge, so the
chain carries every earlier word). The last chunk pads `[slot, 16)` with dep 0, offs = segment
pairs, aabb 0 and writes once, as the fused tail. Records are whole 64 B pages, each RISC
flushes only its own `[gs, ge)` run, so no page or 16 B word is written twice.

**L1 budget.** Recomputed: inputs 10x3x4 KB, two output sets 2x10x2x4 KB, scratch 17x2x4 KB,
VOP 4 KB, mask 128 B, two staging CBs 2x33,408 B, mailbox 1 KB = 498,048 B (486.4 KiB), as
claimed. No `BufferType::L1` buffers exist in `render/host`, so this fits BH L1 with the extra
8 KB kernel config buffer.

**CB ids 41..52.** No clash: the pfwc program uses 0..40 (38 only without fuse); the compute
`OCB()` maps 9..16 -> 41..48 and 35/36 -> 49/50, matching `pfwc_wsplit::odd_cb` and the
writer's `LO`/`HI` (static_asserts in both the writer and the test). Max id 52 < 64
(`NUM_CIRCULAR_BUFFERS` on Blackhole). `OCB` uses the per-core chunk index, not the tile id,
so parity matches the writers. No compute code reads an output CB back.

**Host wiring.** Runtime args: 25 writer args + 4 semaphore ids, NCRISC adds the 9 input bases
and the opacity at 29..38; 19 compile-time accessor args for NCRISC. `FUSE_ABL` is dropped
with a warning. Defines per role are consistent with the single-writer path.

## Non-blocking notes

1. `writer_pfwc_split.cpp:271`: `cb_pages_available_at_front` does not invalidate the L1
   cache, and `rd_poll` returns early (before its `invalidate_l1_cache`) once all chunks are
   read or while reads are pending. Harmless today: the BH RISC data cache is off by default
   and `invalidate_l1_cache` is only a `fence`. Would spin on a stale value only if
   `TT_METAL_ENABLE_L1_DATA_CACHE_RISCVS` enables NCRISC's cache. Hardening: call
   `invalidate_l1_cache()` in that loop (or use `wait_until`).
2. The receiver sets EMPTY (`:324`, `:385`) without a fence after its last message load.
   Safe in practice (in-order core, values are consumed before the store), a `fence` would
   make it explicit.
3. With the split, CBs 37 (mask) and 39 (opacity) are still allocated but unused (~4.2 KB),
   and the +8 KB kernel config is added whenever the knob is on, even with fuse off. Both
   trivial.

The model's own estimate (README) puts the split alone at 0.02-0.12 ms/view, below the
0.3 ms gate; its value depends on the SFPU fusion landing. The default stays 0, as designed.
