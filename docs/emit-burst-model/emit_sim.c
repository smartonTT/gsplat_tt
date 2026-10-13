// Task #637: sequential model of the one-launch sort emit's per-tile run ring
// (sort_bin_onelaunch.cpp step 5, OL_RING = R, OL_RING_DEPTH = D).
// One mover stream = the screen-tile ids of its records in emit order. Time
// advances one unit per record. A record whose tile cursor k starts a run
// (k % R == 0, k >= D) waits until the run D records back has left L1, which
// happens L units after that run's last record was packed (flush latency).
// Out, per stream: end time (records + stall units), stall events, and the
// burst histogram: a burst = consecutive records of one tile, each <= L units
// after the previous one; hist[min(len, 1023)] += len (records in bursts of len).
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

void sim(const int32_t* tiles, const int64_t* off, int32_t nstreams, int32_t ntiles,
         int32_t R, int32_t D, int64_t L, int64_t* end_out, int64_t* events_out, int64_t* hist) {
    const int32_t nr = D / R;  // runs in flight per tile
    int64_t* fin = malloc(sizeof(int64_t) * ntiles * nr);  // flush-done time per ring run slot
    int64_t* cur = malloc(sizeof(int64_t) * ntiles);
    int64_t* last = malloc(sizeof(int64_t) * ntiles);
    int64_t* blen = malloc(sizeof(int64_t) * ntiles);
    for (int32_t s = 0; s < nstreams; s++) {
        memset(cur, 0, sizeof(int64_t) * ntiles);
        memset(blen, 0, sizeof(int64_t) * ntiles);
        for (int64_t i = 0; i < (int64_t)ntiles * nr; i++) fin[i] = INT64_MIN / 2;
        for (int32_t t = 0; t < ntiles; t++) last[t] = INT64_MIN / 2;
        int64_t T = 0, ev = 0, rec = 0;
        for (int64_t i = off[s]; i < off[s + 1]; i++) {
            const int32_t t = tiles[i];
            const int64_t k = cur[t]++;
            // burst bookkeeping (stream position, no stalls)
            if (rec - last[t] <= L) blen[t]++;
            else { if (blen[t]) hist[blen[t] < 1023 ? blen[t] : 1023] += blen[t]; blen[t] = 1; }
            last[t] = rec++;
            if (k % R == 0 && k >= D) {
                const int64_t need = fin[(int64_t)t * nr + (k / R) % nr];
                if (T < need) { T = need; ev++; }
            }
            T += 1;
            if (k % R == R - 1) fin[(int64_t)t * nr + (k / R) % nr] = T + L;
        }
        for (int32_t t = 0; t < ntiles; t++)
            if (blen[t]) hist[blen[t] < 1023 ? blen[t] : 1023] += blen[t];
        end_out[s] = T;
        events_out[s] = ev;
    }
    free(fin); free(cur); free(last); free(blen);
}

// sim3: the same ring rule on the TOWN emit (task #202): TRISC t % 3 owns tile t and packs its
// records in stream order, 3 time units per record (so a balanced stream runs at 1 unit per
// record). Batches of B records; a TRISC starts batch k after its own batch k-1 and after every
// TRISC finished batch k - SLOTS (slot reuse). L is in the same units. Out: end time per stream.
void sim3(const int32_t* tiles, const int64_t* off, int32_t nstreams, int32_t ntiles,
          int32_t R, int32_t D, int64_t L, int32_t B, int32_t SLOTS, int64_t* end_out) {
    const int32_t nr = D / R;
    int64_t* fin = malloc(sizeof(int64_t) * ntiles * nr);
    int64_t* cur = malloc(sizeof(int64_t) * ntiles);
    int64_t* alld = malloc(sizeof(int64_t) * (SLOTS + 1));
    for (int32_t s = 0; s < nstreams; s++) {
        memset(cur, 0, sizeof(int64_t) * ntiles);
        for (int64_t i = 0; i < (int64_t)ntiles * nr; i++) fin[i] = INT64_MIN / 2;
        int64_t done[3] = {0, 0, 0};
        for (int32_t j = 0; j <= SLOTS; j++) alld[j] = 0;
        const int64_t n = off[s + 1] - off[s];
        int64_t kb = 0;
        for (int64_t b0 = 0; b0 < n; b0 += B, kb++) {
            const int64_t b1 = (b0 + B < n) ? b0 + B : n;
            const int64_t gate = (kb >= SLOTS) ? alld[(kb - SLOTS) % (SLOTS + 1)] : 0;
            int64_t mx = 0;
            for (int32_t o = 0; o < 3; o++) {
                int64_t T = done[o] > gate ? done[o] : gate;
                for (int64_t i = off[s] + b0; i < off[s] + b1; i++) {
                    const int32_t t = tiles[i];
                    if (t % 3 != o) continue;
                    const int64_t k = cur[t]++;
                    if (k % R == 0 && k >= D) {
                        const int64_t need = fin[(int64_t)t * nr + (k / R) % nr];
                        if (T < need) T = need;
                    }
                    T += 3;
                    if (k % R == R - 1) fin[(int64_t)t * nr + (k / R) % nr] = T + L;
                }
                done[o] = T;
                if (T > mx) mx = T;
            }
            alld[kb % (SLOTS + 1)] = mx;
        }
        end_out[s] = done[0] > done[1] ? (done[0] > done[2] ? done[0] : done[2]) : (done[1] > done[2] ? done[1] : done[2]);
    }
    free(fin); free(cur); free(alld);
}
