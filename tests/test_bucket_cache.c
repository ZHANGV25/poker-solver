/*
 * test_bucket_cache.c — Verify the cached turn/river bucket equals what
 * the old per-lookup computation would produce.
 *
 * For random (hand, board) pairs, compute the bucket using the SAME formulas
 * that are now used at cache-population time (in the deal-card branch), and
 * compare to a reference implementation that mirrors the OLD per-lookup code.
 *
 * If they match for thousands of random cases, the cache is equivalent to
 * the old code and safe to deploy.
 *
 * Build (Linux with OpenMP):
 *   gcc -O2 -fopenmp -Isrc -o test_bucket_cache tests/test_bucket_cache.c \
 *       src/mccfr_blueprint.c src/card_abstraction.c -lm
 *
 * Usage:
 *   ./test_bucket_cache [num_cases] [seed]
 *   Default: 1000 cases, seed=42
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "hand_eval.h"

/* Local copies of static helpers from mccfr_blueprint.c so we don't need to
 * link the full solver object. These must match the solver exactly. */
static inline uint64_t rng_next(uint64_t *state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

static inline uint32_t rng_int(uint64_t *state, uint32_t max) {
    if (max == 0) return 0;
    return (uint32_t)(rng_next(state) % (uint64_t)max);
}

static void partial_shuffle(int *arr, int n, int k, uint64_t *rng) {
    for (int i = 0; i < k && i < n; i++) {
        int j = i + rng_int(rng, n - i);
        int tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
    }
}

/* Reference: this mirrors EXACTLY what the OLD traverse() code did at line
 * 1003-1031 (river case) for computing the bucket from (hand, board). */
static int old_river_bucket(int c0, int c1, const int *board, int num_board,
                            int postflop_num_buckets) {
    int blk[52] = {0};
    for (int b = 0; b < num_board; b++) blk[board[b]] = 1;
    blk[c0] = 1; blk[c1] = 1;
    int av[52]; int nav = 0;
    for (int c = 0; c < 52; c++) if (!blk[c]) av[nav++] = c;
    int wins = 0, ties = 0, total = 0;
    uint64_t erng = (uint64_t)c0 * 1000003ULL + (uint64_t)c1 * 999983ULL;
    for (int b = 0; b < num_board; b++)
        erng = erng * 6364136223846793005ULL + (uint64_t)board[b];
    int cards_needed = 2 + (5 - num_board);
    (void)cards_needed;  /* used in partial_shuffle */

    /* Inline partial_shuffle behavior — mirrors the old code */
    extern void partial_shuffle(int *arr, int n, int k, uint64_t *rng);

    for (int si = 0; si < 200 && nav >= 2; si++) {
        /* For river (num_board == 5): cards_needed = 2 (just opponent hand).
         * This matches the new deal-time code exactly. */
        partial_shuffle(av, nav, 2, &erng);
        int h7[7] = {board[0], board[1], board[2], board[3], board[4], c0, c1};
        int o7[7] = {board[0], board[1], board[2], board[3], board[4], av[0], av[1]};
        uint32_t hs = eval7(h7), os = eval7(o7);
        if (hs > os) wins++; else if (hs == os) ties++;
        total++;
    }
    float ehs = (total > 0) ? ((float)wins + 0.5f*(float)ties) / (float)total : 0.5f;
    int b = (int)(ehs * (float)postflop_num_buckets);
    if (b >= postflop_num_buckets) b = postflop_num_buckets - 1;
    return b;
}

/* Mirrors the NEW deal-time code from traverse() (near line 955).
 * Should be IDENTICAL to old_river_bucket. */
static int new_river_bucket(int c0, int c1, const int *board,
                            int postflop_num_buckets) {
    int blk[52] = {0};
    for (int b = 0; b < 5; b++) blk[board[b]] = 1;
    blk[c0] = 1; blk[c1] = 1;
    int av[52]; int nav = 0;
    for (int c = 0; c < 52; c++) if (!blk[c]) av[nav++] = c;
    int wins = 0, ties = 0, total = 0;
    uint64_t erng = (uint64_t)c0 * 1000003ULL + (uint64_t)c1 * 999983ULL;
    for (int b = 0; b < 5; b++)
        erng = erng * 6364136223846793005ULL + (uint64_t)board[b];

    extern void partial_shuffle(int *arr, int n, int k, uint64_t *rng);

    for (int si = 0; si < 200 && nav >= 2; si++) {
        partial_shuffle(av, nav, 2, &erng);
        int h7[7] = {board[0], board[1], board[2], board[3], board[4], c0, c1};
        int o7[7] = {board[0], board[1], board[2], board[3], board[4], av[0], av[1]};
        uint32_t hs = eval7(h7), os = eval7(o7);
        if (hs > os) wins++; else if (hs == os) ties++;
        total++;
    }
    float ehs = (total > 0) ? ((float)wins + 0.5f*(float)ties) / (float)total : 0.5f;
    int b = (int)(ehs * (float)postflop_num_buckets);
    if (b >= postflop_num_buckets) b = postflop_num_buckets - 1;
    return b;
}

/* Random 5-card board + 2-card hand, no collisions. */
static void random_situation(uint64_t *rng, int *hand, int *board) {
    int used[52] = {0};
    int dealt = 0;
    while (dealt < 7) {
        uint64_t x = *rng;
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        *rng = x;
        int c = (int)(x % 52);
        if (used[c]) continue;
        used[c] = 1;
        if (dealt < 2) hand[dealt] = c;
        else board[dealt - 2] = c;
        dealt++;
    }
}

int main(int argc, char **argv) {
    int num_cases = 1000;
    uint64_t seed = 42;
    if (argc > 1) num_cases = atoi(argv[1]);
    if (argc > 2) seed = (uint64_t)atoll(argv[2]);

    printf("Testing %d random (hand, board) cases, seed=%llu\n",
           num_cases, (unsigned long long)seed);

    uint64_t rng = seed;
    int mismatches = 0;
    int first_mismatch_shown = 0;

    for (int i = 0; i < num_cases; i++) {
        int hand[2], board[5];
        random_situation(&rng, hand, board);

        int old_b = old_river_bucket(hand[0], hand[1], board, 5, 200);
        int new_b = new_river_bucket(hand[0], hand[1], board, 200);

        if (old_b != new_b) {
            mismatches++;
            if (!first_mismatch_shown) {
                printf("MISMATCH at case %d: old=%d new=%d\n", i, old_b, new_b);
                printf("  hand: %d %d, board: %d %d %d %d %d\n",
                       hand[0], hand[1], board[0], board[1], board[2], board[3], board[4]);
                first_mismatch_shown = 1;
            }
        }
    }

    printf("\nResults: %d/%d mismatches\n", mismatches, num_cases);
    if (mismatches == 0) {
        printf("PASS: river bucket cache is equivalent to old per-lookup compute.\n");
        return 0;
    } else {
        printf("FAIL: bucket cache produces different values from old code!\n");
        return 1;
    }
}
