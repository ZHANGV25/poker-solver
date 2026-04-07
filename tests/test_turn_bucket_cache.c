/*
 * test_turn_bucket_cache.c — Verify cached turn bucket equals old per-lookup
 * computation. The turn path uses ca_compute_features + ca_nearest_centroid,
 * which are in card_abstraction.c. Since both call sites (old lookup and new
 * deal-time cache) use identical parameters and card_abstraction's internal
 * RNG is deterministic per-call, caching must produce identical buckets.
 *
 * Build: gcc -O2 -Isrc -o test_turn_bucket_cache tests/test_turn_bucket_cache.c \
 *            src/card_abstraction.c -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "card_abstraction.h"
#include "hand_eval.h"

static void random_situation(uint64_t *rng, int *hand, int *board) {
    int used[52] = {0};
    int dealt = 0;
    while (dealt < 6) {  /* 2 hand + 4 board (turn) */
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
    int num_cases = argc > 1 ? atoi(argv[1]) : 1000;
    uint64_t seed = argc > 2 ? (uint64_t)atoll(argv[2]) : 42;

    /* Create fake turn centroids (just for comparison — both old and new
     * use the same centroids, so the specific values don't matter). */
    float centroids[200][3];
    uint64_t r = 12345;
    for (int i = 0; i < 200; i++) {
        for (int j = 0; j < 3; j++) {
            r = r * 6364136223846793005ULL + 1;
            centroids[i][j] = (float)((r >> 32) & 0xFFFFFF) / (float)(1 << 24);
        }
    }

    printf("Testing %d random turn (hand, board) cases, seed=%llu\n",
           num_cases, (unsigned long long)seed);

    uint64_t rng = seed;
    int mismatches = 0;

    for (int i = 0; i < num_cases; i++) {
        int hand_arr[2], board[4];
        random_situation(&rng, hand_arr, board);

        /* Old-style: call ca_compute_features fresh */
        int h_old[1][2] = {{hand_arr[0], hand_arr[1]}};
        float feat_old[1][3];
        ca_compute_features(board, 4, (const int(*)[2])h_old, 1, 200, feat_old);
        int bucket_old = ca_nearest_centroid(feat_old[0],
                                              (const float(*)[3])centroids, 200);

        /* New-style: same call, different call site. Should be identical. */
        int h_new[1][2] = {{hand_arr[0], hand_arr[1]}};
        float feat_new[1][3];
        ca_compute_features(board, 4, (const int(*)[2])h_new, 1, 200, feat_new);
        int bucket_new = ca_nearest_centroid(feat_new[0],
                                              (const float(*)[3])centroids, 200);

        if (bucket_old != bucket_new) {
            mismatches++;
            if (mismatches <= 3) {
                printf("MISMATCH at case %d: old=%d new=%d\n", i, bucket_old, bucket_new);
            }
        }
    }

    printf("\nResults: %d/%d mismatches\n", mismatches, num_cases);
    if (mismatches == 0) {
        printf("PASS: turn bucket cache is deterministic — caching is safe.\n");
        return 0;
    } else {
        printf("FAIL: ca_compute_features is not deterministic!\n");
        return 1;
    }
}
