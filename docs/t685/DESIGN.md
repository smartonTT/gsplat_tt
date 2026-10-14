# T1 device blend backward: kernel design (#685, continued in a follow-up)

Done in #685: `gsplat/blend_backward_dev.py` (prep files, numpy model of this kernel,
`check` against `blend_backward(...)['slots']`) and `tests/unit/test_blend_backward_dev.py`.
Model vs ref on `benchmarks/train_oracle/t0_sh{0,3}_64x64_n50.npz`: 2.8e-5 / 3.6e-5 (bar 1e-2).
Usage: `python3 -m gsplat.blend_backward_dev prep <npz> <dir> --emulate`,
`python3 -m gsplat.blend_backward_dev check <npz> <dir>/slots.f32`.

Not written yet: `render/train/kernels/{bwd_reader,blend_bwd_compute,bwd_writer}.cpp`,
`render/train/blend_bwd_bench.cpp`, `render/train/CMakeLists.txt` (pattern: render/bench),
`docs/t685/drive.sh` (pattern: docs/sync-t676/drive.sh, one `ttp lock p100`, opt/sync_remote.sh).

## Dst (fp32, dst_full_sync_en: 8 tiles; dst_reg index = tile*32 + vector)
0 T (0 = dead or outside image), 1 BH = Ig - C_acc.g, 2-4 dL/dimage, 5 x ramp, 6 y ramp,
7 ACC: vectors 0-8 per-lane term sums (zeroed per pair), 9+3i..11+3i reduced results of
pair i of a batch (i < 7), 30 S0 (w), 31 S1 (ga). Vector V lane (row j, col k) = packed
element 64*(V>>1) + 16j + 2k + (V&1) (matches tt-llk load_face_data: SFPLOAD addr = 2V).

## Per visited vector, body<IX> (noinline sfpi, 32 instantiations, mask bits ascending)
u = X - mx; v = Y - my; pw = A'uu + B'uv + C'vv (A'=-A/2, B'=-B, C'=-C/2)
w = exp(min(pw,0)) (_sfpu_exp_21f_bf16_<true>); al = op*w; ga = pw>0 ? 0 : al
if al > AMAX: al = AMAX, w = 0, ga = 0.   S0 = w, S1 = ga.
om = 1-al; Tn = T*om; if al >= AMIN: if Tn >= TMIN { T = Tn; at = al*T_old;
  acc1..3 += at*g0..2; cg = g.c; BH -= at*cg; rc = approx_recip(om) + 2 NR steps;
  da = T_old*cg - BH*rc; acc0 += da*S0; G = da*S1; acc7 += G u; acc4 += G u u;
  acc5 += G u v; acc8 += G v; acc6 += G v v } else T = 0.
AMIN/AMAX/TMIN as exact fp32 bit defines (1/255 0x3B808081, 0.99 0x3F7D70A4, 1e-4 0x38D1B717).
Keep <= 8 live vFloats (no spills in sfpi).

## Pair finish (noinline, template on pair index i)
acc7 = A acc7 + B acc8; acc8 = B acc7 + C acc8 (raw A,B,C); acc4 *= -0.5; acc5 = -acc5;
acc6 *= -0.5. Reduce groups (7,8,4,5) (6,0,1,2) (3,0,0,0): SFPLOAD LREG0-3, SFPTRANSP,
SFPADD tree to LREG0, horizontal reduce as tt-llk horizontal_reduce phases 1-3 (sum in col 7,
no rotate), SFPSTORE to vector 9+3i+gk. Output term q of pair i is packed element
64*(RV>>1) + 16*(q%4) + 14 + (RV&1), RV = 9+3i+q/4.

## Flow
Reader (RISCV_1): ramps -> CB_RAMP once; per tile 5 planes -> CB_PLANE; per batch of <= 7
records one 512 B CB_REC page (slot 0 header {n, first-of-tile}) and a CB_SLOT page.
Compute: copy_tile planes to dst 0-4 on first-of-tile (UnpackToDestFp32), SFPU section,
MATH->UNPACK mailbox before cb_pop_front(CB_REC), pack dst 7 per batch with the forward's
non_zeroing_pack_release (alpha_blend_compute_mb.cpp:1061). Wall-clock stats per stage
(setup, bodies, finish, pack sync, pairs, bodies count, record wait) to a stats page.
Writer (RISCV_0): 9 floats per pair from CB_OUT into 64 B slot page `slot`.
Host: tiles split over cores by pair count; fp32_dest_acc_en, HiFi3, approx off.
