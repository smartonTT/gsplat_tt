# Slack thread notes — D0C1CV1AJJV (Steve Marton ↔ Mateja Stojkovic DM)

Source: https://tenstorrent.slack.com/archives/D0C1CV1AJJV/p1789161791870189
Retrieved 2026-09-30 via Glean (`mcp__glean__search` / `read_document` on the
`tenstorrent.enterprise.slack.com` mirror of the same permalink). Slack desktop
caches on this machine contain no message bodies for this DM; no Slack token or
export is configured.

## Verbatim content of the thread

**Steve Marton — 2026-09-11 21:23**
> Hey, I talked with Vanja back in May about optimizing gaussian splatting. I
> don't see him on slack anymore. Are either of you still interested in gaussian
> splatting? I got a 12x speedup on the bicycle scene
> https://github.com/smartonTT/gsplat_tt/pull/3 — which is still nowhere near GPU
> performance, sadly, and I had to drop it due to higher priority projects. I
> won't be able to work on it in the foreseeable future. I just remembered about
> it today, and I thought I'd mention it to you guys, so the work doesn't go to
> waste, in case either of you still care about it. Let me know if you do.

**Mateja Stojkovic — 2026-09-14 09:28** (with a PPTX attachment)
> Hi Steve! Since you last talked to Vanja, we made some good progress on perf
> side as well! I think we got to around 1.6x slower then gpu on the scenes we
> tested on (the reference cuda card is gtx4060, that was the one we had at
> home). I will send you the presentation we made for the work we did (on the
> last slide you can see the speed-ups with improved kernels and algorithms we
> used). Also, thanks for remembering us, i will take a look at your work, maybe
> we can steal some ideas from there as well!

That is the whole thread. The only other technical content is inside the attached
deck (Slack file `F0C1PMDG12M`), which Glean has **not** indexed — see Blocked
below.

## Technically actionable points from the thread itself

1. **There is a reference number to beat: 1.6× slower than a GTX 4060.** The
   Kovinić/Stojkovic line got a single TT card to within 1.6× of a mid-range
   consumer GPU on their test scenes. Our target (beat GPU) is therefore known to
   be within ~2× of somebody's already-achieved TT result, not orders away. This
   is the single most useful fact in the thread: it sets a concrete, credible
   ceiling and says the remaining gap is algorithmic, not architectural.
2. **Their gains came from "improved kernels *and algorithms*", not just kernel
   micro-optimization.** This matches our own history: iters 022-034 (AABB /
   opacity-aware / Mahalanobis culling) moved the needle; iters 039-081 of
   CB-fusion and init-dropping produced sub-1% each. Algorithm-level changes are
   where the remaining 1.6× lives.
3. **Their baseline is a different codebase lineage** (Kovelja009/gsplat_tt →
   Vanja's alpha-blend kernel) from ours (smartonTT/gsplat_tt PR #3, the 12×
   line). Cross-pollination is explicitly invited by Mateja, so their algorithm
   choices are fair game to adopt.

## Candidate optimizations for gsplat-tt (mapped)

| Candidate | Rationale / origin | Status in our repo |
|---|---|---|
| Sub-tile (4×8 SFPU-block) culling and blending instead of whole-32×32-tile splat processing | Steve's own May diagnosis in #dit-project and the Jonathan Su DM: a GPU rejects at 32-thread granularity, we reject only at 32×32, so we do up to 32× wasted work. This is the largest single known inefficiency. | partially present — `render/kernels/compute/microblock_cull_compute.cpp`, `tile_l1_cull_sfpu.hpp` exist; worth re-measuring how much of the 32× is actually recovered |
| Matrix-engine (FPU/GEMM) formulation of the splat | tt-splat (kinginu, tt-awesome) replaces `exp` + depth-sorted alpha with a **polynomial splat + order-independent weighted-sum blending**, turning the pipeline into GEMM → activation → GEMM. Uses the matrix engine that 3DGS on GPUs leaves idle. | not tried; big, and order-independence also removes our host sort (11 ms) |
| Skip negligible tiles entirely | Jiwon Lee / Hyunggi Chang proposal notes: "many tiles have negligible impact and could be skipped; TT's MIMD cores can exploit irregular sparsity where GPUs can only do structured sparsity." | our contrib_floor lever is the per-pair version; whole-tile skip is a different axis |
| GSCore (ASPLOS '24) as an algorithmic reference | cited in the KAIST 3DGS-on-TT proposal as the design basis for a 3DGS rendering-pipeline accelerator | not consulted yet |

## Related material found while retrieving the thread (all cite-able)

- Steve's 2026-05-27 #dit-project update: kernel "tapped out"; root cause is
  per-32×32-tile uniform splat processing; plan was 4×8 SFPU-block culling; C++
  CPU renderer rewrite went 0.2 → 100 fps (metal was 2 fps at the 0.2 fps point).
  https://tenstorrent.enterprise.slack.com/archives/C094N67K9R7/p1779900235641199
- Vanja Kovinić's original alpha-blend-only kernel demo on a single n150:
  https://tenstorrent.enterprise.slack.com/archives/C3GEYPU73/p1778174078368719
- `tt-splat` — matrix-native 3DGS on Blackhole: https://github.com/kinginu/tt-splat
- Steve's own design doc "32x32 Bin 3DGS Renderer with FPU Optimization and TT
  Implementation Notes":
  https://docs.google.com/document/d/1BQbvP5oSEy_W7tfC0skrX8YPR_lUSV4_jmQnOp38z2I

## Blocked: the attached deck

The per-kernel speedup table Mateja refers to ("on the last slide you can see the
speed-ups with improved kernels and algorithms we used") is in the PPTX attached
to the 2026-09-14 message (Slack file id `SLACK2_File_E08AYGP54BT_F0C1PMDG12M`).
Glean returns `richDocumentData.status: NOT_FOUND` for it — attachment bytes are
not indexed.

To unblock, any one of:
- a Slack user/bot token with `files:read` on that DM (then
  `files.info` + `url_private_download` for `F0C1PMDG12M`), or
- the deck re-shared into a Glean-indexed location (Google Drive / SharePoint), or
- Steve downloading it manually from the Slack client into `docs/`.
