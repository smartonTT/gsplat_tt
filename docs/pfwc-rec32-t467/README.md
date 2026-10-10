# t467: narrower pfwc record (GSPLAT_TT_PFWC_REC32), code-only

Status: implemented behind `GSPLAT_TT_PFWC_REC32=1` (default 0). Not measured on device yet.

## Bytes per survivor (default chain: fused pfwc, writer split, EMIT_PUBOC, one-launch town sort, bulk brec)

| | today | REC32 |
|---|---|---|
| pfwc blendrec write | 64 B (words a,b,c,mx,my,op,cr,cg,cb,-,u01,u23,dep, rest 0) | 32 B (a,b,c,mx,my,u01,u23,dep), 2 per 64 B page |
| pfwc dep/offs/aabb | 12 B | 12 B |
| pfwc total write | 76 B | 44 B |
| emit blendrec read | 64 B | 32 B |

The emit uses only words 0-4 and 10-12, so op/colour (words 5-8) were dead bytes.
At about 1.57 M survivors/view (t232: about 14.3k per core x 110 cores; re-check on p150's 120 cores),
pfwc writes drop from about 119 to about 69 MB/view, and the emit's blendrec read from about 100 to about 50 MB/view.
Both are estimates. Measure them before claiming anything.

Output is bit-identical by construction (same words, new addresses). Expect md5 39d84b28.

## How it works
- Writer (writer_pfwc_split.cpp, pfwc_wsplit.h Rec32Stage): gaussian g goes to page g>>1, half (g&1)*32 B.
  A chunk that starts odd fills the upper half of the previous chunk's open page. That half page is handed
  across chunks in OPEN mailbox words MSG_REC..MSG_REC+7. The core's odd last page gets an upper half of 0.
- Emit (sort_bin_onelaunch.cpp brec_pg/brec_ho, sort_ol_town_compute.cpp): reads page g>>1 and offsets
  into it. Bulk runs span brec_pg(gb)-brec_pg(ga)+1 pages. Word indices for op/cg/dep move from 10/11/12 to 5/6/7.
- Guards: the host throws unless pfwc is fused with the writer split and the sort is one-launch town with bulk brec.

## Lever (b): device-side survivor count, no host read: not done
In b2b the early sort is enqueued on CQ0 right behind K2. CQ1 reads proj_M and the count rows and uploads
the layout while the sort runs, so the host read is off the device critical path. The best case is the
inter-program idle, 0.05-0.3 ms/view (#464), and only if mat ever waits on the host layout upload.
That is not worth the code, so this task skipped it.

## Risks
- pfwc is TRISC-bound (t232: TRISC is about the core wall; writer rec is about 0.8 ms). The pfwc gain may be small.
  The surer gain is the emit read and the DRAM traffic.
- +64 B staging L1 per writer RISC (CB_FUSE / CB_STG_ODD), plus a 64 B head page at l1_mask+128.
- The legacy mat overflow path (sort_subchunk_materialize, aos[9..11]), DUMP_PROJ and the parity tools assume
  the 64 B layout. They are not in the default chain. REC32 is refused outside the guarded chain.
- The 16-bit town offset still fits (BREC_HALF*64 <= 65536; tested in test_pfwc_rec32).

## Tests (host)
- tests/unit/test_pfwc_rec32.cpp models the writer protocol over random chunk splits and nb in {1,2,3,7,8}.
  It checks that each page is written once with the right halves, and that the emit's bulk addressing finds every record.
- tests/syntax_stub/check.sh gains REC32 variants of writer_pfwc_split (ROLE 0/1), sort_bin_onelaunch
  (town, no town) and sort_ol_town_compute. All pass. The FAILs in sort_subchunk_materialize and matblend_* existed before this task.

## Device A/B (p150, one run at a time)
```
ttp lock p100 -- <sync + build as usual>
ttp lock p100 -- env GSPLAT_TT_PFWC_REC32=0 python3 render/run.py --back-to-back   # 30 bicycle views
ttp lock p100 -- env GSPLAT_TT_PFWC_REC32=1 python3 render/run.py --back-to-back
```
Keep it only if b2b ms/view drops, md5 matches 39d84b28 (or hero PSNR >= 42.4 dB vs benchmarks/reference_v2/hero.png),
and the hero.png screenshot and diff, rendered on device, are checked by eye for tile artifacts. Also record
pfwc and sort device time per pass (B2B_STAGES) to see where the gain lands.
