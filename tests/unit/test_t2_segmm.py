"""#713: numpy model of the T2 FPU segment reduce (t2_reader.cpp segment tiles, t2_compute.cpp
matmuls, fold, Dst transpose, SFPTRANSP regroup) against a direct per-lane segment sum."""
import numpy as np


def logical(flat):  # face-linear 1024 -> 32x32
    f = flat.reshape(2, 2, 16, 16)  # face row, face col, r, c
    return f.transpose(0, 2, 1, 3).reshape(32, 32)


def flat_of(t):
    return t.reshape(2, 16, 2, 16).transpose(0, 2, 1, 3).reshape(1024)


def vec(tile_flat, v):  # SFPU vector v of a tile: rows 4(v//2)..+3, column half v%2 -> 4x8
    rows = tile_flat.reshape(64, 16)
    return rows[4 * (v // 2):4 * (v // 2) + 4, 8 * (v % 2):8 * (v % 2) + 8].copy()


def set_vec(tile_flat, v, x):
    rows = tile_flat.reshape(64, 16)
    rows[4 * (v // 2):4 * (v // 2) + 4, 8 * (v % 2):8 * (v % 2) + 8] = x


def model(slots, P, cnt):
    s0, s1 = P[0], P[63] + cnt[63]
    pg0 = s0 // 64
    npg = (s1 - 1) // 64 - pg0 + 1 if s1 > s0 else 0
    acc = np.zeros((4, 32, 32))
    li = 0
    for q in range(npg):
        pb = (pg0 + q) * 64
        A = np.zeros(4 * 1024)
        for sl in range(max(s0, pb), min(s1, pb + 64)):
            while li < 63 and sl >= P[li] + cnt[li]:
                li += 1
            j = sl - pb
            rho = (j & 15) | ((j >> 5) << 4)
            t = ((j >> 4) & 1) * 2 + (li >> 5)
            r = li & 31
            A[t * 1024 + (((r >> 4) << 1) | (rho >> 4)) * 256 + (r & 15) * 16 + (rho & 15)] = 1
        page = np.zeros((64, 16))
        n = min(64, len(slots) - pb)
        page[:n] = slots[pb:pb + n]
        B = logical(page.reshape(1024))
        for t in range(4):
            acc[t] += np.einsum("ij,jk->ik", logical(A[t * 1024:(t + 1) * 1024]), B)
    d = [flat_of(acc[t]).copy() for t in range(4)]
    for v in range(8):  # fold
        for lo, hi in ((0, 2), (1, 3)):
            for f0, f1 in ((0, 8), (16, 24)):
                set_vec(d[lo], f0 + v, vec(d[lo], f0 + v) + vec(d[hi], f1 + v))
    t4, t5 = flat_of(logical(d[0]).T), flat_of(logical(d[1]).T)
    out = np.zeros(1024)
    for m in range(3):
        for h in range(2):
            regs = [vec(t4, 2 * m + h), vec(t4, 8 + 2 * m + h), vec(t5, 2 * m + h), vec(t5, 8 + 2 * m + h)]
            tr = [np.stack([regs[k][i] for k in range(4)]) for i in range(4)]  # SFPTRANSP
            for i in range(4):
                set_vec(out, 8 * m + 2 * i + h, tr[i])
    return out


def test_segment_matmul_layout():
    rng = np.random.default_rng(713)
    for trial in range(6):
        cnt = rng.poisson(8 if trial % 2 else 1, 64)
        cnt[rng.integers(0, 64, 5)] = 0
        start = int(rng.integers(0, 200))
        P = start + np.concatenate([[0], np.cumsum(cnt)[:-1]])
        slots = rng.standard_normal((start + cnt.sum() + 100, 16))
        out = model(slots, P, cnt)
        for s in range(9):
            for i in range(64):
                ref = slots[P[i]:P[i] + cnt[i], s].sum()
                assert abs(out[64 * s + i] - ref) < 1e-9, (trial, s, i)


def tf32(x):  # matmul source registers keep 10 mantissa bits (truncation)
    return (np.asarray(x, np.float32).view(np.uint32) & np.uint32(0xFFFFE000)).view(np.float32)


def srca(x):  # FPU SrcA at HiFi4: 8 explicit mantissa bits (tech_reports/matrix_engine)
    return (np.asarray(x, np.float32).view(np.uint32) & np.uint32(0xFFFF8000)).view(np.float32)


def bf16_trunc(x):
    return (np.asarray(x, np.float32).view(np.uint32) & np.uint32(0xFFFF0000)).view(np.float32)


def hi_lo(x):  # run-2 split: hi exact in tf32, lo = x - hi
    x = np.asarray(x, np.float32)
    hi = tf32(x)
    return hi, x - hi


def pieces(x):  # t2_bench.cpp slot split: three bf16 pieces, each the truncation of the rest
    r, out = np.asarray(x, np.float32), []
    for _ in range(3):
        out.append(bf16_trunc(r))
        r = (r - out[-1]).astype(np.float32)
    return out


def test_slot_split_precision():
    # Slots sit in SrcA, which keeps 8 mantissa bits: a raw fp32 page (#713 run 1) and the tf32
    # hi + fp32 lo pair (run 2) both lose ~2^-9 and failed the T0 oracle (worst grad 0.17 / 0.13);
    # three bf16 pieces pass SrcA unchanged and sum back to the fp32 slot exactly.
    rng = np.random.default_rng(7131)
    x = (rng.standard_normal(1 << 16) * np.exp(rng.uniform(-20, 20, 1 << 16))).astype(np.float32)
    ref = x.astype(np.float64)
    rel = lambda y: (np.abs(y - ref) / np.abs(ref)).max()  # noqa: E731
    hi, lo = hi_lo(x)
    assert rel(srca(x).astype(np.float64)) > 2.0 ** -10
    assert rel(srca(hi).astype(np.float64) + srca(lo)) > 2.0 ** -10
    p = pieces(x)
    for q in p:
        assert (bf16_trunc(q) == q).all() and (srca(q) == q).all()
    assert (sum(q.astype(np.float64) for q in p) == ref).all()
