# t222: pfwc feed-side cuts on top of the t221 default — SHELVE

Analysis only, no device run. Input: the t221 Tracy capture of the current default
(`GSPLAT_TT_PFWC_WRITER_SPLIT=1` + `GSPLAT_TT_PFWC_COVCAM_SFPU=1`, STEPCYC=1, views 0-3 plus
warm-up, 110 cores): `docs/pfwc-writer-split-t207/dev-t221/out/prof-both-dev.csv.gz`.
Script: `feed_slack.py`, output: `feed_slack.txt`.

## Scope change

The task first asked whether COVCAM_SFPU + writer split (+ a 528 B TRISC1 code trim) clear
the 0.3 ms/view gate. #221 already measured that: default (split + cov_cam) is
13.604 ms/view vs 14.268 both-off (-0.664), md5 46a725ab, and both are on by default
(f031ba0). So the question here is only: what feed-side (reader/writer) cut is left on
top of that default?

## Model

Per (core, launch), feed slack = max(BRISC, NCRISC) writer wall - TRISC busy (wall minus
input wait). On the makespan: max over cores of max(feed, TRISC) wall minus max over cores
of TRISC busy. That is what a zero-cost feed (instant first input, no writer tail) could
still save.

| | ms per view (traced) |
|---|---:|
| TRISC wall, mean core | 2.212 |
| TRISC0 input wait (first chunk + stalls) | 0.036 |
| writer tail after TRISC ends (BRISC / NCRISC) | 0.013 / 0.012 |
| feed - TRISC busy, per core p50 / p90 / max | 0.036 / 0.059 / 0.096 |
| **feed slack on the makespan, views 1-3 (launches 1-4)** | **0.045** (max 0.060, warm-up launch) |

Both writers wait 1.0-1.05 ms of their 2.22 ms (t221 table). TRISC wait is ≤0.036 ms on
TRISC0 and 0.001 on TRISC1/2. pfwc is compute-bound; the feed is idle half the time.

## Verdict: SHELVE every pfwc feed-side cut for now

- Upper bound for any reader/writer change (2:1 chunk deal, dynamic chunk claim, faster
  first read, smaller tail) is ~0.045 ms/view traced, ~7x below the 0.3 gate.
- The 528 B TRISC1 trim is moot. The split program is 92496 B and needs +24 KB of kernel
  config buffer anyway (t221), and +24 KB costs nothing visible (base at +24 KB: 14.19-14.34
  vs 14.235 tip). No trim is proposed.
- No specific feed cut is proposed. The feed only matters again if TRISC compute drops below
  the writers' busy time: ~1.12 ms (BRISC cls+rec) and ~1.19 ms + reader polls (NCRISC). That
  needs t197 lever 2 (one SFPU pass for cov2d a/b/c + conic + radii). On TRISC1 those steps
  are 0.185 + 0.187 + 0.136 + 0.073 + 0.209 + 0.207 = 0.997 ms of the 2.21 ms wall.
  After lever 2, NCRISC (writer + reader) is the likely new limit. A 2:1 BRISC:NCRISC chunk
  deal would be the first feed cut to model then.

## Caveats

- Traced numbers (STEPCYC counters on). Untraced pfwc runs a little faster, so the slack is
  about the same or smaller.
- p100a (yyzo-bh-07) capture, views 0-3 only, like t197/t221.
- The project stage is 3.346 ms/view untraced (default verify), and pfwc is ~2.3 ms of it.
  The other ~1.0 ms (K2 and the gaps) is outside this model.
