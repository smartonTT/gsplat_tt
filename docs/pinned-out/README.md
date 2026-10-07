# Pinned-host output (task #367): p100a A/B

`GSPLAT_TT_OUT_PINNED=1` (default 0) makes the blend writer write the u8 image
rows straight into a pinned, NoC-mapped host buffer
(`experimental::PinnedMemory::Create(..., map_to_noc=true)`, page-aligned, same
rows x pitch layout as `res_out`) instead of the DRAM image. The writer barriers
on write acks before it exits; after `Finish` the host copies the buffer into
`image_out` inside the `d2h` span. With the flag at 0 the kernel (no
`OUT_PINNED` define) and its runtime args are unchanged.

Code: b00e6453. Driver: `drive.sh` (c8e744c6). Box: p100a yyzo-bh-04 (measurement
reservation, under `ttp lock p100`). One build (sha c8e744c6, so_md5 df11323b),
30-view bicycle sweep per run, rounds alternate base/pin. Raw output in `out/`.

## Correctness

- md5 906e0435 on 30/30 views in every run, flag on and off
  (`out/md5-r*-*.txt`, `ALL_VIEWS_IDENTICAL`).
- With the flag on the writer never writes the DRAM image, so a matching md5 on
  all 30 views shows the frames really come from the pinned buffer.
- Device hero with the flag on: `opt/metal-screenshots/t367-pinned-out/hero.png`,
  diff `hero_diff10.png`. PSNR 42.51 dB vs `benchmarks/reference_v2/hero.png`,
  golden 8-bit match, max 0 LSB vs golden. Checked by eye: no tile seams; the diff
  shows only the usual edge structure (spokes, foliage).

## A/B (ms/view, stage means over 30 views)

| run | avg_frame | view_total | blend | d2h |
|---|---|---|---|---|
| r1 base | 10.930 | 10.892 | 7.070 | 0.254 |
| r1 pin  | 10.905 | 10.867 | 7.111 | 0.250 |
| r2 pin  | 10.880 | 10.845 | 7.121 | 0.234 |
| r2 base | 10.924 | 10.887 | 7.081 | 0.255 |
| r3 base | 10.926 | 10.891 | 7.041 | 0.259 |
| r3 pin  | 10.917 | 10.884 | 7.088 | 0.257 |
| **base mean** | **10.927** | 10.890 | 7.064 | 0.256 |
| **pin mean**  | **10.901** | 10.865 | 7.107 | 0.247 |
| delta | -0.026 (-0.2%) | -0.025 | +0.043 | -0.009 |

(r0 pin was a smoke run: 10.959 / d2h 0.291, warm-up included, not counted.)

## Reading

On the p100a the flag is neutral (-0.2%, inside noise). The completion-queue
read was never the cost here: what `d2h` measures in both arms is the 3 MB host
memcpy (~0.25 ms, ~12 GB/s, cold lines just written by DMA). The pinned path
removes the read but keeps that copy, and the PCIe writes from the blend writer
add ~0.04 ms to blend.

On the p150 bh-30 #366 measured d2h at 0.47 ms, so ~0.2 ms there is the read
path itself and this flag may recover it; only a p150 A/B can tell.

The bigger lever is the copy: handing the pinned buffer out directly
(zero-copy, double-buffered so the next frame does not overwrite the one the
caller still holds) would remove the ~0.25 ms memcpy on both boxes.
