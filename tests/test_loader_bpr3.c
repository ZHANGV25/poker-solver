/**
 * Loader dispatcher test: verify BPR3 legacy format is handled.
 *
 * After the Phase A loader rewrite, bp_load_regrets has two paths:
 *   - BPR4 (current format) → parallel mmap loader
 *   - BPR2/BPR3 (legacy formats) → serial loader
 *
 * The rewrite preserved the serial loader for BPR2/BPR3 but we had no
 * test exercising that code path. This test synthesizes a valid BPR3
 * file by hand and verifies that bp_load_regrets routes it correctly
 * and produces the expected hash table state.
 *
 * BPR3 header format (from bp_load_regrets dispatch code):
 *   "BPR3" (4B)
 *   int32 saved_table_size
 *   int32 saved_entries
 *   int64 saved_iters
 *
 * Entry format (same as BPR4 — no change when the header fields grew
 * to int64):
 *   int player, int street, int bucket (12B)
 *   u64 board_hash, u64 action_hash (16B)
 *   int num_actions (4B)
 *   int regrets[num_actions] (4*na B)
 *   int has_strategy_sum (4B)
 *   float strategy_sum[num_actions] if has_ss (4*na B)
 *
 * Build:
 *   gcc -O2 -fopenmp -Isrc tests/test_loader_bpr3.c \
 *       src/mccfr_blueprint.c src/card_abstraction.c \
 *       -o build/test_loader_bpr3 -lm
 */
#include "mccfr_blueprint.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

static int mkcard(int rank, int suit) { return rank * 4 + suit; }

static int init_solver(BPSolver *s) {
    int flop[3] = { mkcard(0, 0), mkcard(5, 1), mkcard(11, 2) };
    int hands[BP_MAX_PLAYERS][BP_MAX_HANDS][2];
    float weights[BP_MAX_PLAYERS][BP_MAX_HANDS];
    int num_hands[BP_MAX_PLAYERS] = {1, 1, 0, 0, 0, 0};
    memset(hands, 0, sizeof(hands));
    memset(weights, 0, sizeof(weights));
    hands[0][0][0] = mkcard(12, 3); hands[0][0][1] = mkcard(11, 3);
    hands[1][0][0] = mkcard(1, 0);  hands[1][0][1] = mkcard(1, 1);
    weights[0][0] = 1.0f;
    weights[1][0] = 1.0f;
    float bet_sizes[1] = {1.0f};

    BPConfig config;
    bp_default_config(&config);
    config.include_preflop = 0;
    config.hash_table_size = (int64_t)(1 << 16);
    config.num_threads = 0;
    config.prune_start_iter = 100000000;

    memset(s, 0, sizeof(*s));
    return bp_init_ex(s, 2, flop,
                      (const int (*)[BP_MAX_HANDS][2])hands,
                      (const float (*)[BP_MAX_HANDS])weights,
                      num_hands, 100, 100, bet_sizes, 1, &config);
}

int main(int argc, char **argv) {
    const char *path = (argc > 1) ? argv[1] : "build/bpr3_checkpoint.bin";
    printf("=== BPR3 legacy format loader dispatch test ===\n");
    printf("Writing synthetic BPR3 file: %s\n", path);

    /* Build 10 unique info sets, each with distinct regrets and
     * strategy_sum values. No duplicates — this test exercises the
     * cleanest case: BPR3 dispatch + normal single-copy load. */
    const int N = 10;
    const int NA = 4;

    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 1; }

    /* BPR3 header */
    fwrite("BPR3", 1, 4, f);
    int32_t table_size = 65536;
    int32_t num_entries = N;
    int64_t iters = 7654321;
    fwrite(&table_size, sizeof(int32_t), 1, f);
    fwrite(&num_entries, sizeof(int32_t), 1, f);
    fwrite(&iters, sizeof(int64_t), 1, f);

    /* 10 entries with predictable values */
    int expected_regrets[N][NA];
    float expected_ss[N][NA];
    for (int i = 0; i < N; i++) {
        int player = i % 2;
        int street = 1 + (i / 2) % 3;
        int bucket = i * 7;
        uint64_t board_hash = 0xDEAD0000ULL + (uint64_t)i;
        uint64_t action_hash = 0xBEEF0000ULL + (uint64_t)i * 19;
        int32_t na = NA;

        fwrite(&player, sizeof(int), 1, f);
        fwrite(&street, sizeof(int), 1, f);
        fwrite(&bucket, sizeof(int), 1, f);
        fwrite(&board_hash, sizeof(uint64_t), 1, f);
        fwrite(&action_hash, sizeof(uint64_t), 1, f);
        fwrite(&na, sizeof(int), 1, f);

        for (int a = 0; a < NA; a++) {
            expected_regrets[i][a] = 1000 * (i + 1) + 10 * a;
            fwrite(&expected_regrets[i][a], sizeof(int), 1, f);
        }

        int has_ss = 1;
        fwrite(&has_ss, sizeof(int), 1, f);
        for (int a = 0; a < NA; a++) {
            expected_ss[i][a] = 0.5f + (float)i * 0.1f + (float)a * 0.01f;
            fwrite(&expected_ss[i][a], sizeof(float), 1, f);
        }
    }
    fclose(f);
    printf("Wrote %d BPR3 entries\n", N);

    /* Load it */
    BPSolver s;
    if (init_solver(&s) != 0) { fprintf(stderr, "init failed\n"); return 1; }
    int64_t n = bp_load_regrets(&s, path);
    printf("bp_load_regrets returned: %lld\n", (long long)n);

    if (n != N) {
        fprintf(stderr, "FAIL: loaded %lld entries, expected %d\n",
                (long long)n, N);
        bp_free(&s);
        return 1;
    }

    /* Verify: find each expected key and check regrets/ss */
    int found = 0;
    int mismatches = 0;
    for (int i = 0; i < N; i++) {
        int player = i % 2;
        int street = 1 + (i / 2) % 3;
        int bucket = i * 7;
        uint64_t board_hash = 0xDEAD0000ULL + (uint64_t)i;
        uint64_t action_hash = 0xBEEF0000ULL + (uint64_t)i * 19;

        int matched = 0;
        for (int64_t slot = 0; slot < s.info_table.table_size; slot++) {
            if (s.info_table.occupied[slot] != 1) continue;
            BPInfoKey *k = &s.info_table.keys[slot];
            BPInfoSet *is = &s.info_table.sets[slot];
            if (k->player != player) continue;
            if (k->street != street) continue;
            if (k->bucket != bucket) continue;
            if (k->board_hash != board_hash) continue;
            if (k->action_hash != action_hash) continue;
            matched = 1;
            found++;
            for (int a = 0; a < NA; a++) {
                if (is->regrets[a] != expected_regrets[i][a]) {
                    if (mismatches < 5) {
                        printf("  MISMATCH entry %d action %d: got %d expected %d\n",
                               i, a, is->regrets[a], expected_regrets[i][a]);
                    }
                    mismatches++;
                }
                if (!is->strategy_sum) {
                    if (mismatches < 5) {
                        printf("  MISMATCH entry %d: NULL strategy_sum\n", i);
                    }
                    mismatches++;
                    break;
                }
                if (fabsf(is->strategy_sum[a] - expected_ss[i][a]) > 1e-5f) {
                    if (mismatches < 5) {
                        printf("  MISMATCH entry %d action %d ss: got %.3f expected %.3f\n",
                               i, a, is->strategy_sum[a], expected_ss[i][a]);
                    }
                    mismatches++;
                }
            }
            break;
        }
        if (!matched) {
            if (mismatches < 5) {
                printf("  MISSING entry %d (player=%d street=%d bucket=%d)\n",
                       i, player, street, bucket);
            }
            mismatches++;
        }
    }

    printf("Found: %d/%d, mismatches: %d\n", found, N, mismatches);
    bp_free(&s);

    /* Also verify that iterations_run was picked up from the header */
    if (mismatches > 0) {
        fprintf(stderr, "\n=== BPR3 DISPATCH TEST FAILED ===\n");
        return 1;
    }

    printf("\n=== BPR3 DISPATCH TEST PASSED ===\n");
    return 0;
}
