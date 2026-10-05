// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Task #274 (GSPLAT_TT_K2_TRISC=1, docs/k2-split-t274): the segment K2's pair
// loop on the core's idle TRISCs. Each mover (tile_assign_scatter_seg.cpp)
// owns one job CB (CB_JOB, + 16 on BRISC like its scratch CB) holding two jobs:
// job 0 for TRISC 0 (BRISC) / TRISC 2 (NCRISC), job 1 for TRISC 1 (both, BRISC
// first). A job is pages [pg0, pg0 + npg) of the mover's range
// (pfwc_fuse::k2_jobs); the mover reads its stream window (diet_window) into
// the job, fills the header and sets GO2 then GO. The TRISC
// (k2_trisc_compute.cpp) zeroes the job's count row, runs
// emit_pairs_diet_from over WinIo into the job's gid / tid pages and sets
// DONE. The mover, after its own pages, waits for DONE, clears GO, writes the
// job's pages and adds its counts to its count row. Task #291: the mover posts
// the jobs before their windows land and fills them in chunks (H_FILL). Jobs are posted every
// launch, npg = 0 when there is no work, so the TRISCs never wait forever.
#pragma once

#include <cstdint>

#ifndef K2_TCAP
#define K2_TCAP 640u  // window and pair pages per job
#endif

namespace k2_trisc {

constexpr uint32_t CB_JOB = 1;  // + 16 (TA_CB_OFFSET) for BRISC
constexpr uint32_t MAGIC = 0x274A0B51u, MAGIC2 = 0x9C3D1E27u;
enum : uint32_t { H_GO, H_GO2, H_DONE, H_PG0, H_NPG, H_C, H_LO, H_NIN, H_PPUB, H_TX, H_NSEG, H_TAB, H_SPAN, H_TS, H_TE, H_FILL };
// H_TS / H_TE: TRISC wall clock at GO seen / before DONE (K2_PROF builds only).
// H_FILL (task #291): window pages in L1 so far; the mover raises it per chunk.
constexpr uint32_t PB = 64;
constexpr uint32_t FOLD_TILES = 1024;  // == K2_FOLD_TILES
constexpr uint32_t CNT_OFF = PB;
constexpr uint32_t LOFS_OFF = CNT_OFF + FOLD_TILES * 4u;
constexpr uint32_t BOX_OFF = LOFS_OFF + K2_TCAP * PB;
constexpr uint32_t GID_OFF = BOX_OFF + K2_TCAP * PB;
constexpr uint32_t TID_OFF = GID_OFF + K2_TCAP * PB;
constexpr uint32_t JOB_BYTES = TID_OFF + K2_TCAP * PB;
constexpr uint32_t CB_BYTES = 2u * JOB_BYTES + 64u;  // + alignment slack

// The job area: the CB's base rounded up to 64 B (mover: get_write_ptr,
// TRISC: the CB interface's read pointer; equal in a fresh CB).
inline uint32_t job_addr(uint32_t cb_base, uint32_t j) { return ((cb_base + 63u) & ~63u) + j * JOB_BYTES; }

}  // namespace k2_trisc
