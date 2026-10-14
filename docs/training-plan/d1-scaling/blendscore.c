// Task #691 (D1): importance score for the pruned timing proxies.
// score[g] += alpha * T over every (sampled) pixel of every train view, front to back per
// 32x32 tile, 3DGS rules: alpha = min(0.99, op * exp(-0.5 * d^T conic d)), skip alpha < 1/255,
// stop when T would fall below 1e-4. Pixels are sampled on a stride x stride grid (pixel centre
// x + 0.5, y + 0.5). Pairs come in (tile, depth) order with per-tile offsets.
#include <math.h>
#include <stdint.h>
void blendscore(const int64_t *off, int64_t ntiles, const int32_t *tile_of_seg, const int32_t *gid,
                const float *mx, const float *my, const float *ca, const float *cb, const float *cc,
                const float *op, const int32_t *glob, int W, int H, int tiles_x, int stride,
                double *score) {
  for (int64_t s = 0; s < ntiles; s++) {
    int t = tile_of_seg[s];
    int x0 = (t % tiles_x) * 32, y0 = (t / tiles_x) * 32;
    for (int py = y0; py < y0 + 32 && py < H; py += stride)
      for (int px = x0; px < x0 + 32 && px < W; px += stride) {
        float T = 1.0f, fx = px + 0.5f, fy = py + 0.5f;
        for (int64_t k = off[s]; k < off[s + 1]; k++) {
          int g = gid[k];
          float dx = fx - mx[g], dy = fy - my[g];
          float pw = -0.5f * (ca[g] * dx * dx + 2.0f * cb[g] * dx * dy + cc[g] * dy * dy);
          if (pw > 0.0f) continue;
          float a = op[g] * expf(pw);
          if (a > 0.99f) a = 0.99f;
          if (a < 1.0f / 255.0f) continue;
          float Tn = T * (1.0f - a);
          if (Tn < 1e-4f) break;
          score[glob[g]] += (double)(a * T);
          T = Tn;
        }
      }
  }
}
