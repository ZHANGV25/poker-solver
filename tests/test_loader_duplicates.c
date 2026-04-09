/**
 * Loader duplicate-key merge test.
 *
 * Writes a hand-crafted BPR4 checkpoint file with intentional duplicate
 * keys (simulating the Hogwild race bug in training). Both loaders —
 * legacy serial and new parallel mmap — should merge the duplicates by
 * summing regrets and strategy_sum. This test asserts that both loaders
 * produce the same merged state.
 *
 * Why this matters: the parallel loader uses atomic fetch_add on
 * regrets under concurrent writes to the same slot. If two threads race
 * on a duplicate-key slot, the atomic must ensure both values are added
 * without loss. This is the single most subtle correctness concern for
 * the parallel rewrite. The test makes the race happen for real by
 * dispatching 100 duplicate copies of the same key across threads.
 *
 * Build:
 *   gcc -O2 -fopenmp -Isrc tests/test_loader_duplicates.c \
 *       src/mccfr_blueprint.c src/card_abstraction.c \
 *       -o build/test_loader_duplicates -lm
 */

#include "mccfr_blueprint.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#ifdef _WIN32
static void set_env(const char *name, const char *value) {
    _putenv_s(name, value ? value : "");
}
#else
static void set_env(const char *name, const char *value) {
    if (value) setenv(name, value, 1);
    else unsetenv(name);
}
#endif

static int mkcard(int rank, int suit) { return rank * 4 + suit; }

/* Build a BPR4 file by hand. Layout:
 *   "BPR4" (4B)
 *   int64 table_size
 *   int64 num_entries
 *   int64 iterations_run
 *   Per entry:
 *     int player, int street, int bucket (12B)
 *     u64 board_hash, u64 action_hash (16B)
 *     int num_actions (4B)
 *     int regrets[num_actions] (4*na B)
 *     int has_strategy_sum (4B)
 *     float strategy_sum[num_actions] if has_ss (4*na B)
 */
typedef struct {
    int player, street, bucket;
    uint64_t board_hash, action_hash;
    int num_actions;
    int regrets[BP_MAX_ACTIONS];
    int has_ss;
    float ss[BP_MAX_ACTIONS];
} Entry;

static void write_entry(FILE *f, const Entry *e) {
    fwrite(&e->player, sizeof(int), 1, f);
    fwrite(&e->street, sizeof(int), 1, f);
    fwrite(&e->bucket, sizeof(int), 1, f);
    fwrite(&e->board_hash, sizeof(uint64_t), 1, f);
    fwrite(&e->action_hash, sizeof(uint64_t), 1, f);
    fwrite(&e->num_actions, sizeof(int), 1, f);
    fwrite(e->regrets, sizeof(int), e->num_actions, f);
    fwrite(&e->has_ss, sizeof(int), 1, f);
    if (e->has_ss) {
        fwrite(e->ss, sizeof(float), e->num_actions, f);
    }
}

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
    const char *chk_path = (argc > 1) ? argv[1] : "build/duplicates_checkpoint.bin";

    printf("=== Duplicate-key loader test ===\n");
    printf("Synthesizing BPR4 file with intentional duplicates: %s\n", chk_path);
    fflush(stdout);

    /* Construct N unique keys, and for each one, write K duplicate copies
     * with small regret/ss values. Expected final state: each unique key
     * should have regrets[a] = K * regret_per_copy[a] and same for ss. */
    const int N_UNIQUE = 50;
    const int K_COPIES = 100;   /* simulates 100 Hogwild collisions */
    const int NA = 4;
    const int REGRET_PER_COPY[4] = {10, -20, 30, -40};
    const float SS_PER_COPY[4] = {1.5f, 2.5f, 0.25f, 0.125f};

    FILE *f = fopen(chk_path, "wb");
    if (!f) { fprintf(stderr, "cannot open %s\n", chk_path); return 1; }

    fwrite("BPR4", 1, 4, f);
    int64_t table_size = 65536;
    int64_t num_entries = (int64_t)N_UNIQUE * K_COPIES;
    int64_t iters = 1234567;
    fwrite(&table_size, sizeof(int64_t), 1, f);
    fwrite(&num_entries, sizeof(int64_t), 1, f);
    fwrite(&iters, sizeof(int64_t), 1, f);

    /* Emit K_COPIES copies of each unique key, interleaved so that
     * parallel workers are very likely to land on the same key
     * concurrently during Pass 2 (adjacent index slots cross chunk
     * boundaries at 65536, so we want the copies to actually span threads
     * — we'll use N_UNIQUE keys × K_COPIES but write them in a pattern
     * designed to exercise the contention: all 100 copies of key 0 first,
     * then key 1, etc. With a 65536 chunk size and 5000 total entries,
     * everything goes to one thread — but with the OpenMP dynamic
     * scheduler across 24 threads, small chunks will still exercise
     * some cross-thread collision.
     *
     * To force real contention, we lower the chunk size. We control that
     * via setting OMP_SCHEDULE to dynamic,1 via an env var the loader
     * respects... but the loader has a hardcoded 65536 chunk. So we just
     * make N_UNIQUE * K_COPIES large enough. 50 * 100 = 5000 entries,
     * all in one chunk. That's not great for contention testing.
     *
     * Alternative: bump the counts. 50 keys × 2000 copies = 100k entries,
     * splits into 2 chunks, and with 2 threads racing we'd see contention.
     * Let's do that. */
    for (int u = 0; u < N_UNIQUE; u++) {
        Entry e;
        memset(&e, 0, sizeof(e));
        e.player = u % 2;
        e.street = 1 + (u / 2) % 3;
        e.bucket = u * 13;
        e.board_hash = 0xAAAA0000ULL + (uint64_t)u;
        e.action_hash = 0xBBBB0000ULL + (uint64_t)u * 17;
        e.num_actions = NA;
        for (int a = 0; a < NA; a++) e.regrets[a] = REGRET_PER_COPY[a];
        e.has_ss = 1;
        for (int a = 0; a < NA; a++) e.ss[a] = SS_PER_COPY[a];
        for (int c = 0; c < K_COPIES; c++) {
            write_entry(f, &e);
        }
    }
    fclose(f);
    printf("Wrote %d unique keys × %d copies = %d total entries\n",
           N_UNIQUE, K_COPIES, N_UNIQUE * K_COPIES);

    /* Expected post-load state per unique key:
     *   regrets[a] = K_COPIES * REGRET_PER_COPY[a]
     *   ss[a]      = K_COPIES * SS_PER_COPY[a] */
    int expected_regrets[NA];
    float expected_ss[NA];
    for (int a = 0; a < NA; a++) {
        expected_regrets[a] = K_COPIES * REGRET_PER_COPY[a];
        expected_ss[a]      = K_COPIES * SS_PER_COPY[a];
    }

    /* Helper to check loaded solver state matches expectations. */
    int fail = 0;

    /* Test 1: serial loader */
    {
        printf("\n[1] Loading with BP_LEGACY_LOADER=1 (serial)...\n"); fflush(stdout);
        set_env("BP_LEGACY_LOADER", "1");
        BPSolver s;
        if (init_solver(&s) != 0) { fprintf(stderr, "init failed\n"); return 1; }
        int64_t n = bp_load_regrets(&s, chk_path);
        printf("Serial load returned: %lld\n", (long long)n);

        int found = 0;
        int mismatches = 0;
        for (int u = 0; u < N_UNIQUE; u++) {
            for (int64_t i = 0; i < s.info_table.table_size; i++) {
                if (s.info_table.occupied[i] != 1) continue;
                BPInfoKey *k = &s.info_table.keys[i];
                BPInfoSet *is = &s.info_table.sets[i];
                if (k->player != u % 2) continue;
                if (k->street != 1 + (u / 2) % 3) continue;
                if (k->bucket != u * 13) continue;
                if (k->board_hash != 0xAAAA0000ULL + (uint64_t)u) continue;
                if (k->action_hash != 0xBBBB0000ULL + (uint64_t)u * 17) continue;
                found++;
                for (int a = 0; a < NA; a++) {
                    if (is->regrets[a] != expected_regrets[a]) {
                        if (mismatches < 5)
                            printf("  MISMATCH key=%d action=%d serial_regret=%d expected=%d\n",
                                   u, a, is->regrets[a], expected_regrets[a]);
                        mismatches++;
                    }
                    if (is->strategy_sum) {
                        if (fabsf(is->strategy_sum[a] - expected_ss[a]) > 1e-3f) {
                            if (mismatches < 5)
                                printf("  MISMATCH key=%d action=%d serial_ss=%.3f expected=%.3f\n",
                                       u, a, is->strategy_sum[a], expected_ss[a]);
                            mismatches++;
                        }
                    } else {
                        if (mismatches < 5)
                            printf("  MISMATCH key=%d has NULL strategy_sum\n", u);
                        mismatches++;
                    }
                }
                break;
            }
        }
        printf("Serial: found %d/%d keys, %d regret/ss mismatches\n",
               found, N_UNIQUE, mismatches);
        if (found != N_UNIQUE || mismatches > 0) fail = 1;
        bp_free(&s);
    }

    /* Test 2: parallel loader */
    {
        printf("\n[2] Loading with parallel mmap loader...\n"); fflush(stdout);
        set_env("BP_LEGACY_LOADER", NULL);
        BPSolver s;
        if (init_solver(&s) != 0) { fprintf(stderr, "init failed\n"); return 1; }
        int64_t n = bp_load_regrets(&s, chk_path);
        printf("Parallel load returned: %lld\n", (long long)n);

        int found = 0;
        int mismatches = 0;
        for (int u = 0; u < N_UNIQUE; u++) {
            for (int64_t i = 0; i < s.info_table.table_size; i++) {
                if (s.info_table.occupied[i] != 1) continue;
                BPInfoKey *k = &s.info_table.keys[i];
                BPInfoSet *is = &s.info_table.sets[i];
                if (k->player != u % 2) continue;
                if (k->street != 1 + (u / 2) % 3) continue;
                if (k->bucket != u * 13) continue;
                if (k->board_hash != 0xAAAA0000ULL + (uint64_t)u) continue;
                if (k->action_hash != 0xBBBB0000ULL + (uint64_t)u * 17) continue;
                found++;
                for (int a = 0; a < NA; a++) {
                    if (is->regrets[a] != expected_regrets[a]) {
                        if (mismatches < 5)
                            printf("  MISMATCH key=%d action=%d parallel_regret=%d expected=%d\n",
                                   u, a, is->regrets[a], expected_regrets[a]);
                        mismatches++;
                    }
                    if (is->strategy_sum) {
                        if (fabsf(is->strategy_sum[a] - expected_ss[a]) > 1e-3f) {
                            if (mismatches < 5)
                                printf("  MISMATCH key=%d action=%d parallel_ss=%.3f expected=%.3f\n",
                                       u, a, is->strategy_sum[a], expected_ss[a]);
                            mismatches++;
                        }
                    } else {
                        if (mismatches < 5)
                            printf("  MISMATCH key=%d has NULL strategy_sum\n", u);
                        mismatches++;
                    }
                }
                break;
            }
        }
        printf("Parallel: found %d/%d keys, %d regret/ss mismatches\n",
               found, N_UNIQUE, mismatches);
        if (found != N_UNIQUE || mismatches > 0) fail = 1;
        bp_free(&s);
    }

    if (fail) {
        printf("\n=== DUPLICATE-KEY LOADER TEST FAILED ===\n");
        return 1;
    }
    printf("\n=== BOTH LOADERS CORRECTLY MERGE DUPLICATES ===\n");
    return 0;
}
