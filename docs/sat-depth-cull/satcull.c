// Task #675 (sat-depth cull step 0): CPU pair emit + per-tile saturation scan.
// Built by step0.py: cc -O3 -shared -fPIC -o satcull.so satcull.c -lpthread
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>

#define TS 32

// Pair emit: bbox tiles of (mx +- rx, my +- ry), kept when the peak in-tile
// contribution op*exp(-0.5*m2) at the closest tile point is >= floor
// (same test as gsplat.rasterization.get_tile_assignments).
int64_t emit_pairs(int n, const float* mx, const float* my, const float* A, const float* B,
                   const float* C, const float* op, const float* rx, const float* ry, int W, int H,
                   float floor_, int32_t* out_tile, int32_t* out_g, int64_t cap) {
    const int tx_n = (W + TS - 1) / TS, ty_n = (H + TS - 1) / TS;
    int64_t p = 0;
    for (int g = 0; g < n; g++) {
        int x0 = (int)floorf((mx[g] - rx[g]) / TS), x1 = (int)floorf((mx[g] + rx[g]) / TS);
        int y0 = (int)floorf((my[g] - ry[g]) / TS), y1 = (int)floorf((my[g] + ry[g]) / TS);
        if (x0 < 0) x0 = 0;
        if (y0 < 0) y0 = 0;
        if (x1 > tx_n - 1) x1 = tx_n - 1;
        if (y1 > ty_n - 1) y1 = ty_n - 1;
        for (int ty = y0; ty <= y1; ty++)
            for (int tx = x0; tx <= x1; tx++) {
                float px = fminf(fmaxf(mx[g], (float)(tx * TS)), (float)(tx * TS + TS));
                float py = fminf(fmaxf(my[g], (float)(ty * TS)), (float)(ty * TS + TS));
                float dx = px - mx[g], dy = py - my[g];
                float m2 = A[g] * dx * dx + 2.f * B[g] * dx * dy + C[g] * dy * dy;
                if (op[g] * expf(-0.5f * m2) < floor_) continue;
                if (p >= cap) return -1;
                out_tile[p] = ty * tx_n + tx;
                out_g[p] = g;
                p++;
            }
    }
    return p;
}

typedef struct {
    int t0, t1, W, H, tx_n;
    const int64_t* off;   // tile offsets into the sorted pair list (ntiles+1)
    const int32_t* gid;   // sorted pair -> gaussian
    const float *mx, *my, *A, *B, *C, *op, *sxx, *syy;  // conic A,B,C; cov diag sxx, syy
    float eps;
    int32_t* tile_sat;    // out: index in tile list of the pair after which all px T<eps, or -1
    int32_t* pix_g;       // out: per pixel the gaussian whose blend took T below eps, or -1
    int64_t* blended;     // out: pairs scanned per tile (= sat+1 or n)
} job_t;

static void* run(void* arg) {
    job_t* j = (job_t*)arg;
    float T[TS * TS];
    for (int t = j->t0; t < j->t1; t++) {
        const int tx = t % j->tx_n, ty = t / j->tx_n;
        const int X0 = tx * TS, Y0 = ty * TS;
        const int w = (X0 + TS <= j->W) ? TS : j->W - X0, h = (Y0 + TS <= j->H) ? TS : j->H - Y0;
        for (int i = 0; i < TS * TS; i++) T[i] = 1.f;
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++) j->pix_g[(int64_t)(Y0 + y) * j->W + X0 + x] = -1;
        int live = w * h;
        const int64_t s = j->off[t], e = j->off[t + 1];
        j->tile_sat[t] = -1;
        j->blended[t] = e - s;
        for (int64_t k = s; k < e && live > 0; k++) {
            const int g = j->gid[k];
            const float o = j->op[g];
            const float m2max = 2.f * logf(255.f * o);  // alpha >= 1/255 region
            if (m2max <= 0.f) continue;
            const float hx = sqrtf(m2max * j->sxx[g]) + 1.f, hy = sqrtf(m2max * j->syy[g]) + 1.f;
            int xa = (int)floorf(j->mx[g] - hx - X0), xb = (int)ceilf(j->mx[g] + hx - X0);
            int ya = (int)floorf(j->my[g] - hy - Y0), yb = (int)ceilf(j->my[g] + hy - Y0);
            if (xa < 0) xa = 0;
            if (ya < 0) ya = 0;
            if (xb > w - 1) xb = w - 1;
            if (yb > h - 1) yb = h - 1;
            for (int y = ya; y <= yb; y++) {
                const float dy = (Y0 + y + 0.5f) - j->my[g];
                for (int x = xa; x <= xb; x++) {
                    float* tp = &T[y * TS + x];
                    if (*tp < j->eps) continue;
                    const float dx = (X0 + x + 0.5f) - j->mx[g];
                    const float pw = -0.5f * (j->A[g] * dx * dx + 2.f * j->B[g] * dx * dy + j->C[g] * dy * dy);
                    if (pw > 0.f) continue;
                    float a = o * expf(pw);
                    if (a > 0.99f) a = 0.99f;
                    if (a < 1.f / 255.f) continue;
                    *tp *= (1.f - a);
                    if (*tp < j->eps) {
                        j->pix_g[(int64_t)(Y0 + y) * j->W + X0 + x] = g;
                        live--;
                    }
                }
            }
            if (live == 0) {
                j->tile_sat[t] = (int32_t)(k - s);
                j->blended[t] = k - s + 1;
            }
        }
    }
    return NULL;
}

static int g_next, g_ntiles;

static void* worker(void* arg) {
    job_t j = *(job_t*)arg;
    for (;;) {
        const int t = __atomic_fetch_add(&g_next, 1, __ATOMIC_RELAXED);
        if (t >= g_ntiles) break;
        j.t0 = t;
        j.t1 = t + 1;
        run(&j);
    }
    return NULL;
}

void sat_scan(int ntiles, int W, int H, int nthreads, const int64_t* off, const int32_t* gid,
              const float* mx, const float* my, const float* A, const float* B, const float* C,
              const float* op, const float* sxx, const float* syy, float eps, int32_t* tile_sat,
              int32_t* pix_g, int64_t* blended) {
    // Dynamic tile queue: deep tiles do not stall a wave.
    pthread_t th[64];
    if (nthreads > 64) nthreads = 64;
    job_t j = {0, 0, W, H, (W + TS - 1) / TS, off, gid, mx, my, A, B, C, op, sxx, syy, eps,
               tile_sat, pix_g, blended};
    g_next = 0;
    g_ntiles = ntiles;
    for (int i = 0; i < nthreads; i++) pthread_create(&th[i], NULL, worker, &j);
    for (int i = 0; i < nthreads; i++) pthread_join(th[i], NULL);
}
