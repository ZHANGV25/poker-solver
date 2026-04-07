/*
 * Measure actual info set lookup miss rate.
 * Runs the full unified solver with counters enabled.
 * Reports misses per iteration after the hash table fills.
 *
 * Build: gcc -O2 -g -fopenmp -I src tests/test_miss_counter.c \
 *        src/mccfr_blueprint.c src/card_abstraction.c -o test_miss -lm -lpthread
 * Run:   ./test_miss [threads] [hash_size]
 */
#include "mccfr_blueprint.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

extern void bp_get_miss_stats(int64_t *total, int64_t *miss);
extern void bp_reset_miss_stats(void);

int main(int argc, char **argv) {
    int num_threads = argc > 1 ? atoi(argv[1]) : 32;
    int hash_size = argc > 2 ? atoi(argv[2]) : (1 << 20); /* 1M default */
    /* Phase 1: fill the table. Phase 2: measure misses. */
    int fill_iters = 500000;   /* enough to fill 1M table */
    int measure_iters = 200000; /* measure after fill */

    printf("Miss rate test: %d threads, %d hash slots\n", num_threads, hash_size);
    printf("Phase 1: %d iters (fill table)\n", fill_iters);
    printf("Phase 2: %d iters (measure misses)\n\n", measure_iters);

    BPConfig config;
    bp_default_config(&config);
    config.num_threads = num_threads;
    config.include_preflop = 1;
    config.hash_table_size = hash_size;
    config.discount_stop_iter = 100;
    config.discount_interval = 50;
    config.prune_start_iter = 1000000000; /* no pruning */
    config.snapshot_start_iter = 1000000000; /* no snapshots */
    config.strategy_interval = 100000;

    float postflop_bet_sizes[] = {0.5f, 1.0f, 2.0f};
    float preflop_bet_sizes[] = {0.5f, 1.0f, 2.0f, 3.0f};

    BPSolver solver;
    memset(&solver, 0, sizeof(solver));

    int ret = bp_init_unified(&solver, 6,
                               50, 100, 10000,
                               postflop_bet_sizes, 3,
                               preflop_bet_sizes, 4,
                               &config);
    if (ret != 0) { fprintf(stderr, "init failed: %d\n", ret); return 1; }

    /* Load texture cache AFTER init (init already precomputed, but this
     * replaces it — useful when init was modified to skip). For now,
     * init still precomputes. The cache just verifies compatibility. */
    extern int bp_load_texture_cache(BPSolver *s, const char *path);
    bp_load_texture_cache(&solver, "/tmp/texture_cache.bin");
    printf("Solver initialized.\n");

    /* Phase 1: fill the table */
    printf("Phase 1: filling hash table...\n");
    bp_reset_miss_stats();
    ret = bp_solve(&solver, fill_iters);

    int64_t total1, miss1;
    bp_get_miss_stats(&total1, &miss1);
    printf("  After %d iters: info_sets=%d/%d, lookups=%lld, misses=%lld (%.2f%%)\n",
           fill_iters, bp_num_info_sets(&solver), hash_size,
           (long long)total1, (long long)miss1,
           total1 > 0 ? (double)miss1 / total1 * 100 : 0);

    /* Phase 2: measure miss rate with full table */
    printf("\nPhase 2: measuring miss rate with full table...\n");
    bp_reset_miss_stats();
    ret = bp_solve(&solver, measure_iters);

    int64_t total2, miss2;
    bp_get_miss_stats(&total2, &miss2);
    double miss_rate = total2 > 0 ? (double)miss2 / total2 * 100 : 0;
    double misses_per_iter = (double)miss2 / measure_iters;
    double lookups_per_iter = (double)total2 / measure_iters;

    printf("  Lookups: %lld (%.1f per iter)\n", (long long)total2, lookups_per_iter);
    printf("  Misses:  %lld (%.1f per iter)\n", (long long)miss2, misses_per_iter);
    printf("  Miss rate: %.2f%%\n", miss_rate);

    printf("\n=== RESULT ===\n");
    printf("With %d hash slots (full), each iteration has %.1f lookups and %.1f misses.\n",
           hash_size, lookups_per_iter, misses_per_iter);
    printf("Miss rate: %.2f%% — each miss returns 0 instead of the true game value.\n", miss_rate);

    if (miss_rate > 50)
        printf("CRITICAL: >50%% miss rate. Tree is vastly larger than table.\n");
    else if (miss_rate > 10)
        printf("HIGH: >10%% miss rate. Significant value bias.\n");
    else if (miss_rate > 1)
        printf("MODERATE: >1%% miss rate. Some value bias.\n");
    else
        printf("LOW: <1%% miss rate. Table is adequate.\n");

    bp_free(&solver);
    return 0;
}
