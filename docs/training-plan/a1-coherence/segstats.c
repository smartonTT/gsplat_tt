// Task #688 (A1): per-segment sortedness stats for the sort warm-start sizing.
// For each segment [off[s], off[s+1]) of keys (already in warm-start order):
//   desc[s] = number of descents (key[j] < key[j-1]); runs = desc + 1
//   inv[s]  = number of inversions (pairs i<j with key[i] > key[j])
//   lis[s]  = length of the longest strictly increasing subsequence
// Keys are distinct within a segment (rank of (depth, gid) in the new view).
// Build: cc -O2 -shared -fPIC -o segstats.so segstats.c
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static int64_t merge_count(int32_t *a, int32_t *tmp, int64_t n) {
    if (n < 2) return 0;
    int64_t h = n / 2;
    int64_t c = merge_count(a, tmp, h) + merge_count(a + h, tmp, n - h);
    int64_t i = 0, j = h, k = 0;
    while (i < h && j < n) {
        if (a[i] <= a[j]) tmp[k++] = a[i++];
        else { tmp[k++] = a[j++]; c += h - i; }
    }
    while (i < h) tmp[k++] = a[i++];
    while (j < n) tmp[k++] = a[j++];
    memcpy(a, tmp, n * sizeof(int32_t));
    return c;
}

void segstats(const int32_t *keys, const int64_t *off, int64_t nseg,
              int64_t *desc, int64_t *inv, int64_t *lis) {
    int64_t maxn = 0;
    for (int64_t s = 0; s < nseg; ++s) {
        int64_t n = off[s + 1] - off[s];
        if (n > maxn) maxn = n;
    }
    int32_t *buf = (int32_t *)malloc((maxn + 1) * sizeof(int32_t));
    int32_t *tmp = (int32_t *)malloc((maxn + 1) * sizeof(int32_t));
    int32_t *tails = (int32_t *)malloc((maxn + 1) * sizeof(int32_t));
    for (int64_t s = 0; s < nseg; ++s) {
        const int32_t *k = keys + off[s];
        int64_t n = off[s + 1] - off[s];
        int64_t d = 0;
        for (int64_t j = 1; j < n; ++j) d += k[j] < k[j - 1];
        desc[s] = d;
        if (d == 0) { inv[s] = 0; lis[s] = n; continue; }
        memcpy(buf, k, n * sizeof(int32_t));
        inv[s] = merge_count(buf, tmp, n);
        int64_t L = 0;
        for (int64_t j = 0; j < n; ++j) {
            int64_t lo = 0, hi = L;
            while (lo < hi) { int64_t m = (lo + hi) / 2; if (tails[m] < k[j]) lo = m + 1; else hi = m; }
            tails[lo] = k[j];
            if (lo == L) ++L;
        }
        lis[s] = L;
    }
    free(buf); free(tmp); free(tails);
}
