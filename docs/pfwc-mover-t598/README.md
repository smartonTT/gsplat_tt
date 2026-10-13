# t598: pfwc mover floor (t484 lever 3): shelved at the model step

**Verdict: shelved before build.** No kernel code, no device run, nothing to land. The two parts of
t484 lever 3 (batch the survivor record writes; deal chunks to BRISC/NCRISC by survivor count)
together model at **+0.084 to +0.109 ms/view as an upper bound** (writes free and perfect
balance, neither reachable) and **about +0.05-0.07 realistic**. The kill rule is < 0.1 ms/view
b2b. Spec A (t596, pfwc survivor compaction) stays shelved: the mover floor does not drop.

No device, no ssh, no ird. CPU model only, from t484's measured per-core STEPCYC capture.

## Step 0: not a repeat

- B1 (coalesced bin emit, t534) is the sort emit, not the pfwc writer. Different kernel.
- t439 (LPT deal, shelved) dealt tiles across cores. This lever deals chunks between the two
  movers inside a core. Different.
- t467 (REC32, landed) narrowed the record. t481 (SKIP_RGB, landed) dropped reads. Neither
  touched the record loop or the deal.

## What the code does today (writer_pfwc_split.cpp, pfwc_wsplit.h)

- Part (a) is mostly done already. Records are staged in L1 (`RL` = 16 per bank, up to
  `NB_MAX` = 8 banks) and `flush_rec` sends one NoC write per bank run, not one per record. The
  dep/offs/aabb pages go out as three 64 B writes per 16 survivors. What is left is the flush
  time `fl` (write issue + `noc_async_writes_flushed`): 0.08 ms per mover per launch on the mean
  core, 0.09-0.11 on the slowest. t484's 112-117 cycles per survivor is **rec minus flush**:
  the loop's 11 L1 loads, 8 staging stores and the page bookkeeping. Batching writes does not
  touch it.
- Part (b): compute packs odd chunks into a second output CB set (`OCB()`), so the deal is fixed
  when a chunk is packed. A deal by running survivor count must be chosen before the movers have
  classified that chunk. The model's upper bound ignores this (perfect per-core balance).

## Model (`model598.py`, output `model-out.txt`)

Input: t484 arm sr (iter 222 + SKIP_RGB, p100 11x10, 6600 core-launches). Per core-launch:
`end = max(T + tail, max(busy_B, busy_N) + lead)`, where T is TRISC_1 wall minus back-pressure
(median 1.634 ms) and busy = cls + pfx + rec + opn + tail. lead and tail are fitted on measured
core ends: lead 0.245, tail 0.045, mean error 0.017 ms. The model window is 1.777 vs 1.772
measured. Projection to iter 227: T minus ACQ_FUSE's 0.106 (t489), then T and classify x 0.907
(B3 prune, N 6.13M -> 5.56M). Survivors unchanged.

Iter 227 projection, pfwc window gain (pfwc is on the b2b critical path, so this carries 1:1 to
b2b per t484):

| scenario | fitted lead 0.245 | lead 0.15 | lead 0.10 |
|---|---:|---:|---:|
| base window | 1.677 | 1.604 | 1.569 |
| (a) flush time halved | +0.040 | +0.035 | +0.033 |
| (a) flush time 0 (upper bound) | +0.077 | +0.068 | +0.064 |
| (b) perfect survivor balance (upper bound) | +0.038 | +0.034 | +0.032 |
| **(a)+(b) upper bounds together** | **+0.109** | **+0.096** | **+0.084** |
| ref, not this lever: rec loop at 90 cyc/survivor | +0.105 | +0.093 | +0.082 |
| ref, not this lever: rec loop at 70 cyc/survivor | +0.198 | +0.138 | +0.103 |

The fitted lead is the most mover-bound reading. It is likely too pessimistic: it projects the
ACQ_FUSE + prune cut as -0.10 ms, while measured b2b and project-stage cuts were -0.106 (t489)
and -0.128 (t536). A less mover-bound pfwc (smaller lead) makes this lever worth less, not more.
Only the upper bound under the most pessimistic fit reaches 0.1. A realistic build (flush halved,
about 75% of ideal balance) lands near +0.06.

## What would reopen it

The record loop itself, not its writes: 112 cycles per survivor for 11 loads + 8 stores is
~10 cycles per word. At 70-90 cycles per survivor the model gives +0.08 to +0.20 ms/view. A
microbenchmark of that loop on a Blackhole mover (loads grouped per tile, or 2 survivors per
iteration) is the cheap first step. It would also raise Spec A's ceiling (t596).

## Reproduce

```
git show origin/smarton/tt-project-opt:docs/pfwc-profile-t484/out/pfwc-sr.csv.gz > /tmp/sr.csv.gz
python3 docs/pfwc-mover-t598/model598.py /tmp/sr.csv.gz            # fitted lead
LEAD=0.10 python3 docs/pfwc-mover-t598/model598.py /tmp/sr.csv.gz  # sensitivity
```

`ana.py` is a copy of docs/pfwc-profile-t484/ana.py (CSV parser).
