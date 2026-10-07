# t370: where the host time goes on the p150 bh-30 (iter 210)

Question: after #365 (iter 210), bh-30 p150 runs 11.07 ms/view vs 10.89 on the p100a yyzo-bh-04.
The device stages are faster on the p150, the host stages slower: d2h 0.44 vs 0.23, publish_host
0.31 vs 0.17, pfwc_rtargs 0.10 vs 0.05, sort_mat 0.12 vs 0.04 (`docs/p150-emit-throughput`
P1 runs). Where does that host time go, and is >= 0.3 ms/view of it ours to recover?

Answer: only the host work between views is on the critical path; the in-view host work
(publish_host, bin_layout, sort_mat) runs while the device is still in the emit. The between-view
host time is ~SPAN_GAP ms/view on bh-30 (d2h ~0.44, of it ~0.22 host copy and ~0.21 read path,
plus rtargs + enqueue ~0.19). No single host stage that is ours is worth >= 0.3 ms, so **no code
change**; the one lever above the gate is cross-view overlap with a double-buffered pinned output
(follow-up).

## Host facts (host-only probe, no device use: `host_probe.sh`, `copy_bench.c`, `out/probe-*.txt`)

| | bh-30 (p150b) | yyzo-bh-04 (p100a) | ratio |
|---|---|---|---:|
| CPU | 2x AMD EPYC 7352 (Zen 2), 24 cores in our cpuset, governor performance, boosts to ~3.1 GHz | Ryzen 5 7600X (Zen 4), 12 threads, up to 5.45 GHz | |
| scalar dependent-multiply loop | 1.30 ns/iter | 0.776 ns/iter | **1.68x** |
| 3 MB memcpy, source and destination cache-hot | 0.075-0.083 ms | 0.099 ms | 0.8x |
| 3 MB memcpy, cold source (512 MB ring, like lines just written by DMA), reused destination | **0.225 ms** | **0.180 ms** | 1.25x |
| 3 MB memcpy, cold source, fresh mmap'd destination | 1.73-1.77 ms | 0.90-0.95 ms | 1.9x |
| first touch of 3 MB fresh pages alone (768 4 KB faults) | 1.56 ms | 0.57 ms | 2.7x |
| card PCIe link | Gen4 16 GT/s x16 (card max Gen5), NUMA node 0 | Gen5 32 GT/s x16 | |
| load during the probe (viewer running, idle) | load avg 3.7-7.1; mutagen-agent 122 % CPU, viewer python 16-58 % | load avg 2.0; mutagen-agent 1.9 % | |

The viewer is stopped for every bench, so its CPU use does not enter the bench numbers.
mutagen-agent stays at ~1.2 cores the whole time (not ours; never stopped); #356 pinned the bench
away from it and saw -0.06 ms (noise), so it is not a cause either.

## Which host stages are on the critical path

`render()` is serial per view: project (host rtargs + enqueue, then a device wait), sort (wait for
the K2 rows, then host layout + publish + mat launch), blend (wait on `Finish`), d2h (blocking read).

- In-view host work is hidden. The host `blend` stage (mat+blend enqueue to `Finish`) is 6.575 ms
  on bh-30, while the device mat+blend takes 6.06 ms (#366, same code in 209 and 210): the host
  enqueues the mat ~0.5 ms before the device finishes the emit. bin_layout + publish_host + sort_mat
  (0.08 + 0.31 + 0.12 = 0.51 ms on bh-30) all run inside that slack. Making them 2x faster gives 0.
- Between views the device waits for the host: the d2h read and copy, then the next view's c2w,
  pfwc runtime args and enqueue before the first pfwc kernel starts.

DEVICE_SPAN_SECTION

## d2h split on bh-30 (for the #367 bh-30 A/B and the zero-copy follow-up)

d2h is one blocking `enqueue_read_mesh_buffer` of the 3 MB image straight into the fresh
`py::array` (`blend_device.cpp`, direct path, no assemble). The destination does not page-fault
in the latency bench: a fresh-page copy would cost >= 1.7 ms on bh-30, and d2h is 0.44.

| part | bh-30 | p100a | source |
|---|---:|---:|---|
| d2h stage (P1, stage mean) | 0.436 | 0.228 | `docs/p150-emit-throughput` r1-P1 logs |
| host copy of 3 MB from cold lines (model) | 0.225 | 0.180 | `copy_bench.c` cold |
| rest: completion-queue read path (dispatch + PCIe Gen4 vs Gen5 + per-page handshakes) | **~0.21** | ~0.05 | difference |

So on bh-30 about half of d2h is the read path, which `GSPLAT_TT_OUT_PINNED=1` removes (minus the
~0.04 ms the pinned writes add to blend on the p100a), and half is the host copy, which only a
zero-copy hand-out removes. Expected pinned gain on bh-30: ~0.15-0.2 ms/view (to be measured by the
queued bh-30 A/B, not by this task). The copy is overlapped with the transfer page by page in the
CQ path, so the split is approximate.

## Conclusions

1. The 2x host stage times on bh-30 come from the CPU (Zen 2 at ~3.1 GHz vs Zen 4 at 5.45 GHz,
   1.7x per scalar op, 2.7x per page fault) and, for d2h, from the read path over PCIe Gen4. They
   are not caused by the viewer (stopped during benches) or by anything we run.
2. publish_host, bin_layout, sort_mat: hidden behind the emit on both boards. Not worth work.
3. rtargs + enqueue: ~0.19 ms on bh-30, under the gate alone.
4. d2h: ~0.21 read path (pinned output, #367, A/B queued) + ~0.22 host copy (zero-copy).
5. The lever that clears the gate on the p150 is cross-view overlap: hand out a double-buffered
   pinned output and enqueue view N+1's project before the host consumes view N, so the device
   never waits for d2h/rtargs/enqueue. Bound on bh-30 = BOUND_BH30; on the p100a ~0.24-0.36 ms (#275).

## Screenshot

No code changed and no new arm can become an iteration, so there is no new hero. The device
profiler run used the iter-210 tree (4f4e812b render code = 6640239b); md5 vs the 906e0435 golden
list is in `out-bh30/`.

## Files

- `copy_bench.c`, `host_probe.sh`: host-only probe; `out/probe-bh30.txt`, `out/probe-p100a.txt`
- `idle_gaps.py` (Tracy host+device), `dev_span.py` (device log): per-view span and idle
- `out-bh30/`: the iter-210 bh-30 run (`drive_bh30.sh` from `docs/p150-emit-throughput`, ARMS=P1
  TRACY=P1, 1 round)
