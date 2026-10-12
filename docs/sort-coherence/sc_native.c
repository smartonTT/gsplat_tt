// CPU helpers for the t529 sort-coherence study (docs/sort-coherence/STUDY.md).
// Build: cc -O3 -shared -fPIC -o sc_native.so sc_native.c -lpthread
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// ---- per-tile inversion counts ------------------------------------------------
// keys[off[t] .. off[t+1]) is tile t's list in the order under test; count pairs
// i<j with keys[i] > keys[j] (strict). Also the max displacement of an element
// from its sorted position (stable), i.e. the window a k-sorted fix-up needs.
static int64_t merge_count(float* a, float* tmp, int64_t n) {
    if (n < 2) return 0;
    int64_t m = n / 2, inv = merge_count(a, tmp, m) + merge_count(a + m, tmp, n - m);
    int64_t i = 0, j = m, k = 0;
    while (i < m && j < n) {
        if (a[j] < a[i]) { inv += m - i; tmp[k++] = a[j++]; }
        else tmp[k++] = a[i++];
    }
    while (i < m) tmp[k++] = a[i++];
    while (j < n) tmp[k++] = a[j++];
    memcpy(a, tmp, n * sizeof(float));
    return inv;
}

typedef struct { const float* k; int64_t i; } KI;
static int cmp_ki(const void* x, const void* y) {
    const KI* a = x; const KI* b = y;
    if (a->k[a->i] < b->k[b->i]) return -1;
    if (a->k[a->i] > b->k[b->i]) return 1;
    return (a->i < b->i) ? -1 : (a->i > b->i);
}

void tile_inversions(int64_t ntiles, const int64_t* off, const float* keys, int64_t* inv_out,
                     int64_t* maxdisp_out) {
    int64_t maxn = 0;
    for (int64_t t = 0; t < ntiles; t++) if (off[t + 1] - off[t] > maxn) maxn = off[t + 1] - off[t];
    float* a = malloc((maxn + 1) * sizeof(float));
    float* tmp = malloc((maxn + 1) * sizeof(float));
    KI* ki = malloc((maxn + 1) * sizeof(KI));
    for (int64_t t = 0; t < ntiles; t++) {
        int64_t n = off[t + 1] - off[t];
        const float* k = keys + off[t];
        memcpy(a, k, n * sizeof(float));
        inv_out[t] = merge_count(a, tmp, n);
        int64_t md = 0;
        if (inv_out[t] > 0) {
            for (int64_t i = 0; i < n; i++) { ki[i].k = k; ki[i].i = i; }
            qsort(ki, n, sizeof(KI), cmp_ki);
            for (int64_t r = 0; r < n; r++) {
                int64_t d = ki[r].i - r; if (d < 0) d = -d;
                if (d > md) md = d;
            }
        }
        maxdisp_out[t] = md;
    }
    free(a); free(tmp); free(ki);
}

// ---- alpha blend ------------------------------------------------------------------
// Standard 3DGS front-to-back blend per 32x32 tile, in the list order given.
// alpha = min(0.99, op * exp(-0.5 d^T conic d)); skip alpha < 1/255; stop a pixel
// when T would fall below 1e-4. Black background. img is H*W*3 float32.
typedef struct {
    int64_t t0, t1, step, tiles_x, W, H;
    const int64_t* off; const int32_t* gid;
    const float *mx, *my, *ca, *cb, *cc, *op, *col;
    float* img;
} BlendJob;

static void* blend_worker(void* p) {
    BlendJob* j = p;
    for (int64_t t = j->t0; t < j->t1; t += j->step) {
        int64_t tx = t % j->tiles_x, ty = t / j->tiles_x;
        for (int py = 0; py < 32; py++) for (int px = 0; px < 32; px++) {
            int64_t X = tx * 32 + px, Y = ty * 32 + py;
            if (X >= j->W || Y >= j->H) continue;
            float fx = X + 0.5f, fy = Y + 0.5f, T = 1.f, r = 0, g = 0, b = 0;
            for (int64_t i = j->off[t]; i < j->off[t + 1]; i++) {
                int32_t q = j->gid[i];
                float dx = fx - j->mx[q], dy = fy - j->my[q];
                float pw = -0.5f * (j->ca[q] * dx * dx + j->cc[q] * dy * dy) - j->cb[q] * dx * dy;
                if (pw > 0.f) continue;
                float al = j->op[q] * expf(pw);
                if (al > 0.99f) al = 0.99f;
                if (al < 1.f / 255.f) continue;
                float nT = T * (1.f - al);
                if (nT < 1e-4f) break;
                float w = al * T;
                r += w * j->col[3 * q]; g += w * j->col[3 * q + 1]; b += w * j->col[3 * q + 2];
                T = nT;
            }
            float* o = j->img + 3 * (Y * j->W + X);
            o[0] = r; o[1] = g; o[2] = b;
        }
    }
    return NULL;
}

void blend(int64_t ntiles, int64_t tiles_x, int64_t W, int64_t H, const int64_t* off,
           const int32_t* gid, const float* mx, const float* my, const float* ca, const float* cb,
           const float* cc, const float* op, const float* col, float* img, int nthreads) {
    pthread_t th[64]; BlendJob jb[64];
    if (nthreads > 64) nthreads = 64;
    // Interleave tiles across threads for balance.
    for (int i = 0; i < nthreads; i++) {
        BlendJob b = {i, ntiles, nthreads, tiles_x, W, H,
                      off, gid, mx, my, ca, cb, cc, op, col, img};
        jb[i] = b;
        pthread_create(&th[i], NULL, blend_worker, &jb[i]);
    }
    for (int i = 0; i < nthreads; i++) pthread_join(th[i], NULL);
}
