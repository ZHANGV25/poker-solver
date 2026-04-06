/*
 * test_hash_dedup.c — Stress test for info_table_find_or_create de-dup.
 *
 * Spawns N threads via bp_solve, then scans the hash table for duplicate
 * keys. With the old code, UTG root entries (hottest node) would get
 * duplicated under high thread contention.
 *
 * Build (Linux with OpenMP):
 *   gcc -O2 -fopenmp -Isrc -o test_hash_dedup \
 *       tests/test_hash_dedup.c src/mccfr_blueprint.c src/card_abstraction.c -lm
 *
 * Usage:
 *   ./test_hash_dedup [num_threads] [iterations]
 *   Default: 192 threads, 100000 iterations
 */
#include "mccfr_blueprint.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static int keys_equal(const BPInfoKey *a, const BPInfoKey *b) {
    return a->player == b->player &&
           a->street == b->street &&
           a->bucket == b->bucket &&
           a->board_hash == b->board_hash &&
           a->action_hash == b->action_hash;
}

int main(int argc, char **argv) {
    int num_threads = 192;
    int iterations = 100000;

    if (argc > 1) num_threads = atoi(argv[1]);
    if (argc > 2) iterations = atoi(argv[2]);

    printf("De-dup stress test: %d threads, %d iterations\n",
           num_threads, iterations);
    fflush(stdout);

    /* Init solver — small hash table to maximize probe chain contention */
    BPConfig config;
    bp_default_config(&config);
    config.num_threads = num_threads;
    config.hash_table_size = 50000;  /* tiny table = long chains = max contention */
    config.include_preflop = 1;
    config.postflop_num_buckets = 200;  /* match cached texture_cache.bin */
    /* Disable pruning/discount so all iterations hit the same code paths */
    config.prune_start_iter = iterations + 1;
    config.discount_stop_iter = 0;
    config.snapshot_start_iter = iterations + 1;

    char buf[524288];
    BPSolver *s = (BPSolver *)buf;
    float postflop[] = {0.5, 1.0};
    float preflop[] = {0.5, 0.7, 1.0};

    int ret = bp_init_unified(s, 6, 50, 100, 10000,
                               postflop, 2, preflop, 3, &config);
    if (ret != 0) {
        printf("FAIL: init returned %d\n", ret);
        return 1;
    }

    float tier0[] = {0.5, 0.7, 1.0};
    float tier1[] = {0.7, 1.0};
    float tier2[] = {1.0};
    float tier3[] = {8.0};
    bp_set_preflop_tier(s, 0, tier0, 3, 4);
    bp_set_preflop_tier(s, 1, tier1, 2, 4);
    bp_set_preflop_tier(s, 2, tier2, 1, 4);
    bp_set_preflop_tier(s, 3, tier3, 1, 4);

    printf("Running %d iterations with %d threads...\n", iterations, num_threads);
    fflush(stdout);

    bp_solve(s, iterations);

    printf("Solve done. %lld info sets created.\n",
           (long long)s->info_table.num_entries);
    printf("Scanning for duplicate keys...\n");
    fflush(stdout);

    /* Scan every occupied pair (i, j) where j > i for key equality.
     * For a 50K table this is O(n^2) but n is small (only occupied slots). */
    BPInfoTable *t = &s->info_table;

    /* Collect all occupied indices first */
    int64_t *occ = (int64_t*)malloc(sizeof(int64_t) * (size_t)(t->num_entries + 1000));
    int64_t n_occ = 0;
    for (int64_t i = 0; i < t->table_size; i++) {
        if (t->occupied[i] == 1) {
            occ[n_occ++] = i;
        }
    }
    printf("Occupied slots: %lld (reported: %lld)\n",
           (long long)n_occ, (long long)t->num_entries);

    int total_dups = 0;
    for (int64_t a = 0; a < n_occ; a++) {
        for (int64_t b = a + 1; b < n_occ; b++) {
            if (keys_equal(&t->keys[occ[a]], &t->keys[occ[b]])) {
                total_dups++;
                if (total_dups <= 20) {
                    BPInfoKey *k = &t->keys[occ[a]];
                    printf("  DUP #%d: player=%d street=%d bucket=%d "
                           "ah=%016llx (slots %lld, %lld)\n",
                           total_dups, k->player, k->street, k->bucket,
                           (unsigned long long)k->action_hash,
                           (long long)occ[a], (long long)occ[b]);
                }
            }
        }
    }

    free(occ);
    bp_free(s);

    printf("\n========================================\n");
    if (total_dups == 0) {
        printf("PASS: 0 duplicate keys in %lld info sets.\n", (long long)n_occ);
        return 0;
    } else {
        printf("FAIL: %d duplicate keys detected!\n", total_dups);
        return 1;
    }
}
