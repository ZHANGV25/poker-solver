/*
 * Measure game tree size (unique info sets) for different bet size configs.
 * Uses a huge hash table so nothing misses — we just count how many slots fill.
 * Runs for a fixed number of iters and reports info set count + miss rate.
 *
 * Build: gcc -O2 -g -I src tests/test_tree_size.c src/mccfr_blueprint.c src/card_abstraction.c -o test_tree -lm
 * Run:   ./test_tree <postflop_sizes> <preflop_sizes> <hash_size> <iters>
 *
 * Examples:
 *   ./test_tree 1 4 4194304 500000    # 1 postflop size, 4 preflop, 4M hash
 *   ./test_tree 2 4 4194304 500000    # 2 postflop sizes
 *   ./test_tree 3 4 4194304 500000    # 3 postflop sizes (current)
 */
#include "mccfr_blueprint.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

extern void bp_get_miss_stats(int64_t *total, int64_t *miss);
extern void bp_reset_miss_stats(void);
extern int bp_load_texture_cache(BPSolver *s, const char *path);

int main(int argc, char **argv) {
    int n_post = argc > 1 ? atoi(argv[1]) : 3;
    int n_pre = argc > 2 ? atoi(argv[2]) : 4;
    int hash_size = argc > 3 ? atoi(argv[3]) : (1 << 22); /* 4M */
    int iters = argc > 4 ? atoi(argv[4]) : 500000;
    int n_buckets = argc > 5 ? atoi(argv[5]) : 200;

    float all_post[] = {0.5f, 1.0f, 2.0f};
    float all_pre[] = {0.5f, 1.0f, 2.0f, 3.0f};

    if (n_post > 3) n_post = 3;
    if (n_pre > 4) n_pre = 4;

    printf("Config: %d postflop sizes, %d preflop sizes, %d buckets, %d hash, %d iters\n",
           n_post, n_pre, n_buckets, hash_size, iters);
    printf("Postflop sizes:");
    for (int i = 0; i < n_post; i++) printf(" %.1f", all_post[i]);
    printf("\nPreflop sizes:");
    for (int i = 0; i < n_pre; i++) printf(" %.1f", all_pre[i]);
    printf("\n\n");

    BPConfig config;
    bp_default_config(&config);
    config.num_threads = 1;
    config.include_preflop = 1;
    config.hash_table_size = hash_size;
    config.discount_stop_iter = 100;
    config.discount_interval = 50;
    config.prune_start_iter = 1000000000;
    config.snapshot_start_iter = 1000000000;
    config.strategy_interval = 100000;
    config.postflop_num_buckets = n_buckets;

    BPSolver solver;
    memset(&solver, 0, sizeof(solver));

    int ret = bp_init_unified(&solver, 6,
                               50, 100, 10000,
                               all_post, n_post,
                               all_pre, n_pre,
                               &config);
    if (ret != 0) { fprintf(stderr, "init failed\n"); return 1; }

    /* Run in phases to track fill rate */
    int phase_size = iters / 5;
    for (int phase = 0; phase < 5; phase++) {
        bp_reset_miss_stats();
        bp_solve(&solver, phase_size);

        int64_t total, miss;
        bp_get_miss_stats(&total, &miss);
        int is = bp_num_info_sets(&solver);
        double fill = (double)is / hash_size * 100;
        double miss_pct = total > 0 ? (double)miss / total * 100 : 0;

        printf("  After %dK iters: %d IS (%.1f%% full), miss=%.1f%% (%lld/%lld)\n",
               (phase + 1) * phase_size / 1000, is, fill, miss_pct,
               (long long)miss, (long long)total);
    }

    printf("\nFinal: %d info sets in %d hash slots (%.1f%% full)\n",
           bp_num_info_sets(&solver), hash_size,
           (double)bp_num_info_sets(&solver) / hash_size * 100);

    bp_free(&solver);
    return 0;
}
