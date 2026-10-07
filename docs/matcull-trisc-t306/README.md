# t306: mat cull fill/patch on the TRISCs (MATCULL_TRISC_FILL)

Candidate 1 from `docs/mat-mover-opcount-t304.md`, built on iter 206 (fused mat+blend).
Env `GSPLAT_TT_MATCULL_TRISC_FILL` (define `MATCULL_TRISC_FILL`). Off by default in t306;
**on by default since t315** (unset = on, `=0` = off; `render/host/matcull_trisc_fill.h`).
Confirming A/B and iter 207: `docs/iter207-t315/README.md`.

- Movers (`sort_subchunk_materialize.cpp`): post one `{slab,n}` job page per slab on CB 7/23
  instead of `cull_slab`; the emit of item i moves after the read+sort of item i+1 and waits
  for the done page first.
- TRISC0 (`mat_cull_compute.cpp`, UNPACK): invalidates its data cache, fills the coefficient
  tiles from the slab, 128 records per batch.
- TRISC2 (PACK): patches word3 in the slab, fences, pushes done.

## Untraced A/B (yyzo-bh-07 p100a, bicycle 30 views, 1024x1024, commit 66f6ea3)

| round | off (ms/view) | on (ms/view) | on − off |
|---|---:|---:|---:|
| r1 (on, base) | 11.076 | 10.990 | −0.086 |
| r2 (base, on) | 11.085 | 10.856 | −0.229 |
| mean | 11.081 | 10.923 | **−0.158** |

md5 list 906e0435 (hero 86524912) in all 4 runs, 30/30: bit-identical.
Logs and md5 lists: `out/`.

## Verdict (t306): not kept under the 0.3 ms gate; superseded by t315

−0.16 ms/view is under the 0.3 ms gate (both rounds under). The default stays off, so there is
no new iteration, tag or screenshot. The t304 model predicted 0.55 ms (range 0.30-0.72); the
measured gain is about half the pessimistic end.

Not measured: the new path has no Tracy zones, so a MATCULL_PROF capture cannot say why the gain
fell short (candidates: mover waits on the done page, TRISC0 fill slower per record than the 0.83
ratio assumed, or the 2 movers per core contend for the one TRISC pipeline). That needs zones on
the mover done-wait and on the TRISC0 fill / TRISC2 patch loops first.

## Lever A gate

The default config is unchanged, so t297's profile still applies: TRISC idle in mat 2.415 ms per
core, far above the 0.6 ms gate. With TRISC_FILL on, the model puts ~1.13 ms of that on the TRISCs,
leaving ~1.3 ms; this is a model number, not measured.
