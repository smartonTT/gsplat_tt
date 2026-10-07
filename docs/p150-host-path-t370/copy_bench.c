/* t370: host-only model of the d2h host copy (no device). The completion-queue reader (and the
 * #367 pinned path) copies the 3 MB u8 image from memory the card just wrote by DMA, so the
 * source lines are not in the CPU caches. Times one 3 MB memcpy (median of N) for:
 *   warm   source and destination both cache-hot (the #366 memcpy_bench case)
 *   cold   source cycled through a 512 MB ring (never cached), destination reused
 *   coldf  cold source, fresh mmap'd destination each time (page faults, like a new py::array)
 *   pf     page-fault cost alone: touch one byte per 4 KB page of a fresh 3 MB buffer
 * Also a scalar loop (ns/iter) as a single-thread CPU speed probe.
 *   cc -O2 -o copy_bench copy_bench.c && ./copy_bench */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <sys/mman.h>
static volatile char sink;
#define SZ (1024u * 1024u * 3u)
#define N 200
#define RING 170 /* 170 x 3 MB = 510 MB */
static char* fresh(void) { return mmap(0, SZ, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0); }
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }
static int cmp(const void* a, const void* b) { double x = *(const double*)a, y = *(const double*)b; return (x > y) - (x < y); }
static double med(double* v) { qsort(v, N, sizeof(double), cmp); return v[N / 2]; }
int main(void) {
    double t[N]; char* ring = malloc((size_t)SZ * RING); char* dst = malloc(SZ); char* src = malloc(SZ);
    memset(ring, 1, (size_t)SZ * RING); memset(dst, 2, SZ); memset(src, 3, SZ);
    for (int i = 0; i < N; i++) { double a = now(); memcpy(dst, src, SZ); t[i] = now() - a; }
    printf("warm  %.3f ms\n", med(t));
    for (int i = 0; i < N; i++) { char* s = ring + (size_t)SZ * (i % RING); double a = now(); memcpy(dst, s, SZ); t[i] = now() - a; }
    printf("cold  %.3f ms\n", med(t));
    for (int i = 0; i < N; i++) { char* s = ring + (size_t)SZ * (i % RING); double a = now(); char* d = fresh(); memcpy(d, s, SZ); t[i] = now() - a; sink = d[SZ - 1]; munmap(d, SZ); }
    printf("coldf %.3f ms\n", med(t));
    for (int i = 0; i < N; i++) { double a = now(); volatile char* d = fresh(); for (size_t o = 0; o < SZ; o += 4096) d[o] = 1; t[i] = now() - a; munmap((void*)d, SZ); }
    printf("pf    %.3f ms\n", med(t));
    volatile uint64_t x = 1; double a = now(); for (uint64_t i = 0; i < 200000000ull; i++) x = x * 6364136223846793005ull + i;
    printf("scalar %.3f ns/iter\n", (now() - a) * 1e6 / 2e8);
    return 0;
}
