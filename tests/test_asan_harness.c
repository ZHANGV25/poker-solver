/**
 * Minimal test harness for ASan/TSan testing of the blueprint solver.
 * Runs a short unified solve with multiple threads to detect heap corruption.
 *
 * Compile:
 *   gcc -O1 -g -fsanitize=address -fopenmp -I src \
 *       tests/test_asan_harness.c src/mccfr_blueprint.c src/card_abstraction.c \
 *       -o test_asan -lm -lpthread
 *
 * Run:
 *   OMP_STACKSIZE=64m ASAN_OPTIONS=detect_odr_violation=0:halt_on_error=1 ./test_asan
 */

#include "mccfr_blueprint.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    int num_threads = 8;
    int iterations = 100000;
    int hash_size = 1 << 22;  /* 4M slots — small for testing */

    if (argc > 1) num_threads = atoi(argv[1]);
    if (argc > 2) iterations = atoi(argv[2]);
    if (argc > 3) hash_size = atoi(argv[3]);

    printf("ASan test: %d threads, %d iterations, %d hash slots\n",
           num_threads, iterations, hash_size);

    BPConfig config;
    bp_default_config(&config);
    config.num_threads = num_threads;
    config.include_preflop = 1;
    config.hash_table_size = hash_size;

    /* Scale thresholds for short run */
    config.discount_stop_iter = iterations * 35 / 1000;
    if (config.discount_stop_iter < 100) config.discount_stop_iter = 100;
    config.discount_interval = config.discount_stop_iter / 10;
    if (config.discount_interval < 10) config.discount_interval = 10;
    config.prune_start_iter = iterations * 17 / 1000;
    if (config.prune_start_iter < 50) config.prune_start_iter = 50;
    config.snapshot_start_iter = iterations * 7 / 100;
    config.snapshot_interval = iterations * 17 / 1000;
    if (config.snapshot_interval < 50) config.snapshot_interval = 50;
    config.strategy_interval = 100;

    float postflop_bet_sizes[] = {0.5f, 1.0f, 2.0f};
    float preflop_bet_sizes[] = {0.5f, 1.0f, 2.0f, 3.0f};

    BPSolver solver;
    memset(&solver, 0, sizeof(solver));

    int ret = bp_init_unified(&solver, 6,
                               50, 100, 10000,
                               postflop_bet_sizes, 3,
                               preflop_bet_sizes, 4,
                               &config);
    if (ret != 0) {
        fprintf(stderr, "bp_init_unified failed: %d\n", ret);
        return 1;
    }

    printf("Solver initialized. Running %d iterations...\n", iterations);
    ret = bp_solve(&solver, iterations);
    if (ret != 0) {
        fprintf(stderr, "bp_solve failed: %d\n", ret);
    }

    printf("Done. Info sets: %d\n", bp_num_info_sets(&solver));
    bp_free(&solver);
    printf("Freed. No corruption detected by ASan.\n");
    return 0;
}
