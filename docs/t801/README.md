# #801 F1 prep: k-grouped light-row order (T2_KGROUP, default off)

Lever 1 of docs/training-plan/t799/PLAN.md (origin/ttp/t799-plan-refresh-trainer-breakdown @60c8a542).

## Design (no kernel source change)
Phase-B groups are tied to the param page (P/M/V) layout, so light rows cannot just be reordered in
phase B. Instead T2_KGROUP=1 adds **phase L**: the phase-A kernels (t2g_reader/t2g_compute/t2h_writer,
T2_PHASE_A) compiled with T2_HL=64.
- Light rows of active groups with 2 <= k <= kcap are sorted by k descending (rows ascending on
  ties) and cut into 64-lane groups; their table pages go into htab from page JL = nhrec / HL.
- Lane i of page j writes hsum record j*64+i; metab marks the row HFLAG|record, like a heavy lane.
- Each lane keeps its own (P, count), so its slot run and sum order are unchanged.
- Phase B then runs 1 plane per active group (k = 0/1 lanes plus records). Groups go to cores by
  LPT, at most LMAX = 320 per core (runtime args 3 + LMAX; the 341-word limit is from memory, not
  checked).

## Numbers (pair model, light planes = sum over 64-row groups of max min(k, kcap))
| counts | kcap | in order | k-grouped |
|---|---|---|---|
| Mac model (docs/t801/kgroup_model.py, fixture tests/fixtures/t801) | 16 | 191,492 | 51,449 |
| Mac model | 8 | 166,530 | 50,717 |
| hero device counts (tests/fixtures/hero/t2_counts.npz) | 16 | 191,445 | 51,441 |
| hero device counts | 8 | 166,535 | 50,713 |

Device plan on the hero counts (kcap 8): phase-B weight max 1937 -> 774 per core; phase L 34,847
planes, max 318 per core, 122-123 groups per core. Phase A is unchanged (max 1014).
Estimate only, not measured: B ~2.6 ms + L ~0.95 ms vs the current ~7.3 ms of B (#784 per-plane costs).

## Exactness
Sums equal the in-order split and segment_reduce bit for bit
(tests/unit/test_t2_split.py::test_device_split_sums_kgroup_bit_equal). The one exception is a
zero sum whose slots are all -0.0: its sign depends on the +0.0 padding planes, the same as in #784.
Adam does not see this sign (m starts at +0, v uses g^2).

## Risks for the device A/B (F1b)
- Phase B now reads up to ~30 scattered 64 B hsum records per group.
- Phase L reads per-lane slot runs of 2-8 slots (small NoC reads).
- htab upload per view grows by about 7 MB on the hero view (13.6k pages x 512 B).
- T2_KGROUP=1 has not been compiled or run: there is no device or tt-metal build in this task.
