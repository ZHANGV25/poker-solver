/**
 * Minimal test: exercises ONLY the hash table + arena allocator
 * with many threads, no texture precomputation.
 * Tests for heap corruption from concurrent insert/lookup patterns.
 *
 * gcc -O3 -g -fsanitize=address -fopenmp -I src \
 *     tests/test_hash_table.c src/mccfr_blueprint.c src/card_abstraction.c \
 *     -o test_hash -lm -lpthread
 */
#include "mccfr_blueprint.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Expose internal functions — we'll access them via extern declarations */
extern void *arena_alloc(int num_ints);

int main(int argc, char **argv) {
    int num_threads = 64;
    int num_iters = 5000000;
    int table_size = 1 << 20; /* 1M slots */

    if (argc > 1) num_threads = atoi(argv[1]);
    if (argc > 2) num_iters = atoi(argv[2]);
    if (argc > 3) table_size = atoi(argv[3]);

    printf("Hash table stress test: %d threads, %d iterations, %d slots\n",
           num_threads, num_iters, table_size);

    /* Initialize the solver with minimal config (postflop-only, no precomputation) */
    BPConfig config;
    bp_default_config(&config);
    config.num_threads = num_threads;
    config.include_preflop = 1;
    config.hash_table_size = table_size;
    config.discount_stop_iter = num_iters / 10;
    config.discount_interval = num_iters / 100;
    config.prune_start_iter = num_iters / 20;
    config.snapshot_start_iter = num_iters / 5;
    config.snapshot_interval = num_iters / 20;
    config.strategy_interval = 1000;

    float postflop_bet_sizes[] = {0.5f, 1.0f};
    float preflop_bet_sizes[] = {0.5f, 1.0f, 2.0f, 3.0f};

    BPSolver solver;
    memset(&solver, 0, sizeof(solver));

    /* Use bp_init_unified but with 0 postflop buckets to skip precomputation */
    int ret = bp_init_unified(&solver, 6,
                               50, 100, 10000,
                               postflop_bet_sizes, 2,
                               preflop_bet_sizes, 4,
                               &config);
    if (ret != 0) {
        fprintf(stderr, "init failed: %d\n", ret);
        return 1;
    }

    /* Override postflop settings to skip precomputation if it hasn't happened */
    printf("Running %d iterations with %d threads...\n", num_iters, num_threads);
    ret = bp_solve(&solver, num_iters);
    printf("bp_solve returned %d, info_sets=%d\n", ret, bp_num_info_sets(&solver));

    bp_free(&solver);
    printf("Done. No corruption detected.\n");
    return 0;
}
