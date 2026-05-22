#!/usr/bin/env python3
"""Microbench: SHM write path vs pipe write path for FRM2 frame data.

Simulates the two approaches used by blend() at 1024×1024 stitch_doll
(~1.59 M entries):

  Pipe path  — write (attr, gids, offsets, px, py) to a BytesIO / os.pipe,
               read image back from BytesIO / os.pipe.
  SHM path   — memcpy (attr, gids, offsets, px, py) into SharedMemory regions,
               memcpy image back from SharedMemory.

Both paths do the same amount of data movement from the Python side.
The SHM path shows what the *Python-side* overhead will be once the
daemon reads from SHM instead of the pipe (the daemon-side memcpy cost
is symmetric and is already counted in daemon_rt, not save_npy/load_npy).

Usage:
    python tests/bench_shm_vs_pipe.py [--entries N] [--tiles T] [--reps R]
"""
from __future__ import annotations

import argparse
import io
import os
import statistics
import struct
import time
from multiprocessing.shared_memory import SharedMemory

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--entries", type=int, default=1_591_353,
                   help="Total (Gaussian,tile) pairs (default: 1.59M = 1024x1024 stitch_doll)")
    p.add_argument("--tiles",   type=int, default=1024,
                   help="Number of tiles (default: 1024 = 1024x1024 / 32x32)")
    p.add_argument("--H",       type=int, default=1024)
    p.add_argument("--W",       type=int, default=1024)
    p.add_argument("--reps",    type=int, default=50)
    return p.parse_args()


def bench_pipe_write(attr, gids, offsets, px, py, image_fp32, reps):
    """Simulate the save_npy (FRM2 write) + load_npy (image read) pipe path."""
    total_entries = attr.shape[0]
    offsets_count = offsets.shape[0]
    H, W = image_fp32.shape[:2]

    save_times = []
    load_times = []
    for _ in range(reps):
        buf = io.BytesIO()
        t0 = time.perf_counter()
        # FRM2 header
        buf.write(struct.pack("<6I", 0x46524D32, H, W, total_entries, offsets_count, 0))
        buf.write(memoryview(attr))
        buf.write(memoryview(gids))
        buf.write(memoryview(offsets))
        buf.write(memoryview(px))
        buf.write(memoryview(py))
        t1 = time.perf_counter()
        save_times.append((t1 - t0) * 1e3)

        # Simulate reading image back from stdout
        img_bytes = io.BytesIO(image_fp32.tobytes())
        dst = np.empty_like(image_fp32)
        t2 = time.perf_counter()
        mv = memoryview(dst).cast("B")
        img_bytes.readinto(mv)
        t3 = time.perf_counter()
        load_times.append((t3 - t2) * 1e3)

    return save_times, load_times


def bench_shm_write(
    attr, gids, offsets, px, py, image_fp32,
    shm_attr, shm_gids, shm_offs, shm_px, shm_py, shm_out_view,
    reps,
):
    """Simulate the SHM path: copyto shm_in regions + copyto from shm_out."""
    total_entries = attr.shape[0]
    offsets_count = offsets.shape[0]
    px_n = px.size
    H, W = image_fp32.shape[:2]
    n = H * W * 3

    save_times = []
    load_times = []
    # Pre-write image into shm_out (daemon would do this; here we benchmark
    # only the Python-side memcpy cost)
    np.copyto(shm_out_view[:n], image_fp32.reshape(-1))

    for _ in range(reps):
        t0 = time.perf_counter()
        # Header + data into SHM
        np.copyto(shm_attr[:total_entries],     attr[:total_entries])
        np.copyto(shm_gids[:total_entries],     gids[:total_entries])
        np.copyto(shm_offs[:offsets_count],     offsets[:offsets_count])
        np.copyto(shm_px[:px_n],               px.reshape(-1))
        np.copyto(shm_py[:px_n],               py.reshape(-1))
        # 24-byte FRM2 header write (tiny; included for completeness)
        _ = struct.pack("<6I", 0x46524D32, H, W, total_entries, offsets_count, 1)
        t1 = time.perf_counter()
        save_times.append((t1 - t0) * 1e3)

        # Read image back from SHM
        dst = np.empty_like(image_fp32)
        t2 = time.perf_counter()
        np.copyto(dst.reshape(-1), shm_out_view[:n])
        t3 = time.perf_counter()
        load_times.append((t3 - t2) * 1e3)

    return save_times, load_times


def print_stats(label: str, times: list[float]) -> None:
    med = statistics.median(times)
    mn  = min(times)
    mx  = max(times)
    print(f"  {label:<18s}  med={med:6.2f} ms  min={mn:6.2f} ms  max={mx:6.2f} ms")


def main() -> None:
    args = parse_args()
    P = args.entries
    T = args.tiles
    H, W = args.H, args.W
    reps = args.reps

    print(f"Bench SHM vs pipe  entries={P:,}  tiles={T}  H={H} W={W}  reps={reps}")
    print()

    # ---- Allocate synthetic frame data (matches real blend() dtypes) ----
    rng = np.random.default_rng(42)
    attr      = np.ascontiguousarray(rng.random((P, 5), dtype=np.float32))
    gids      = np.ascontiguousarray(rng.integers(0, 341426, P, dtype=np.uint32))
    offsets   = np.ascontiguousarray(rng.random(T + 1, dtype=np.float32))
    px        = np.ascontiguousarray(rng.random((T, 32, 32), dtype=np.float32))
    py        = np.ascontiguousarray(rng.random((T, 32, 32), dtype=np.float32))
    image_fp32 = np.ascontiguousarray(rng.random((H, W, 3), dtype=np.float32))

    max_entries   = P + P // 4
    max_tiles     = T + 64
    max_offsets   = max_tiles + 1
    max_px_floats = max_tiles * 32 * 32
    max_pixels    = H * W

    SHM_HDR = 64
    attr_off  = SHM_HDR
    gids_off  = attr_off  + max_entries * 5 * 4
    offs_off  = gids_off  + max_entries * 4
    px_off    = offs_off  + max_offsets * 4
    py_off    = px_off    + max_px_floats * 4
    shm_in_sz = py_off    + max_px_floats * 4

    shm_out_sz = max_pixels * 3 * 4  # fp32

    shm_in  = SharedMemory(create=True, size=shm_in_sz)
    shm_out = SharedMemory(create=True, size=shm_out_sz)
    try:
        buf_in = memoryview(shm_in.buf)
        shm_attr = np.frombuffer(buf_in[attr_off : gids_off], dtype=np.float32).reshape(max_entries, 5)
        shm_gids = np.frombuffer(buf_in[gids_off : offs_off], dtype=np.uint32)
        shm_offs = np.frombuffer(buf_in[offs_off : px_off],   dtype=np.float32)
        shm_px   = np.frombuffer(buf_in[px_off   : py_off],   dtype=np.float32)
        shm_py   = np.frombuffer(buf_in[py_off   : shm_in_sz], dtype=np.float32)
        shm_out_view = np.frombuffer(memoryview(shm_out.buf)[:shm_out_sz], dtype=np.float32)

        # ---- Warmup ----
        for _ in range(5):
            bench_pipe_write(attr, gids, offsets, px, py, image_fp32, 1)
            bench_shm_write(attr, gids, offsets, px, py, image_fp32,
                            shm_attr, shm_gids, shm_offs, shm_px, shm_py,
                            shm_out_view, 1)

        # ---- Pipe ----
        pipe_save, pipe_load = bench_pipe_write(attr, gids, offsets, px, py, image_fp32, reps)
        print("Pipe path (BytesIO, simulates stdin write + stdout read):")
        print_stats("save_npy (write)", pipe_save)
        print_stats("load_npy (read)",  pipe_load)
        print()

        # ---- SHM ----
        shm_save, shm_load = bench_shm_write(
            attr, gids, offsets, px, py, image_fp32,
            shm_attr, shm_gids, shm_offs, shm_px, shm_py,
            shm_out_view, reps,
        )
        print("SHM path (copyto shm_in / shm_out):")
        print_stats("save_npy (copyto)", shm_save)
        print_stats("load_npy (copyto)", shm_load)
        print()

        # ---- Summary ----
        pipe_tot_med = statistics.median(pipe_save) + statistics.median(pipe_load)
        shm_tot_med  = statistics.median(shm_save)  + statistics.median(shm_load)
        print(f"Totals (median):  pipe={pipe_tot_med:.2f} ms  shm={shm_tot_med:.2f} ms  "
              f"speedup={pipe_tot_med/shm_tot_med:.1f}×")
        print(f"  save_npy:  {statistics.median(pipe_save):.2f} ms → {statistics.median(shm_save):.2f} ms  "
              f"(−{statistics.median(pipe_save)-statistics.median(shm_save):.2f} ms)")
        print(f"  load_npy:  {statistics.median(pipe_load):.2f} ms → {statistics.median(shm_load):.2f} ms  "
              f"(−{statistics.median(pipe_load)-statistics.median(shm_load):.2f} ms)")

        # Data sizes for reference
        save_mb = (P * 5 * 4 + P * 4 + (T+1) * 4 + T * 1024 * 4 * 2) / 1e6
        load_mb = H * W * 3 * 4 / 1e6
        print(f"\nData moved  save={save_mb:.1f} MB  load={load_mb:.1f} MB")
    finally:
        # Release numpy views before closing SHM (avoid "cannot close exported pointers")
        del shm_attr, shm_gids, shm_offs, shm_px, shm_py, shm_out_view, buf_in
        shm_in.close();  shm_in.unlink()
        shm_out.close(); shm_out.unlink()


if __name__ == "__main__":
    main()
