/**
 * Loader correctness diff test.
 *
 * Trains a small toy solver, saves the regret table, then loads it twice:
 * once with the legacy serial loader (BP_LEGACY_LOADER=1) and once with
 * the new parallel mmap loader (default). Asserts that both loaders
 * produce byte-identical hash table state:
 *   - Same set of occupied slots (same keys at same slot indices)
 *   - Same num_actions per slot
 *   - Same regrets[] per slot
 *   - Same strategy_sum[] per slot (NULL vs non-NULL and float values)
 *
 * If any mismatch exists, exits non-zero with a detailed diff report.
 *
 * This is the primary safety net for the Tier 2 loader rewrite: any
 * divergence means the parallel loader has a bug, and the EC2 run
 * against the 1.5B v2 checkpoint cannot be trusted.
 *
 * Build:
 *   gcc -O2 -fopenmp -Isrc tests/test_loader_diff.c src/mccfr_blueprint.c \
 *       src/card_abstraction.c -o build/test_loader_diff -lm
 * Run:
 *   ./build/test_loader_diff /tmp/toy_checkpoint.bin
 */

#include "mccfr_blueprint.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#ifdef _WIN32
#include <stdlib.h>
static void set_env(const char *name, const char *value) {
    _putenv_s(name, value ? value : "");
}
#else
#include <stdlib.h>
static void set_env(const char *name, const char *value) {
    if (value) setenv(name, value, 1);
    else unsetenv(name);
}
#endif

static int mkcard(int rank, int suit) { return rank * 4 + suit; }

/* Canonical dump of the hash table for diffing. We don't care about
 * slot ORDER (both loaders use the same find_or_create which assigns
 * slots deterministically based on the key hash), but to be robust to
 * any future change in slot assignment we sort by (player, street,
 * bucket, board_hash, action_hash) before comparing. */
typedef struct {
    BPInfoKey key;
    int num_actions;
    int regrets[BP_MAX_ACTIONS];
    int has_strategy_sum;
    float strategy_sum[BP_MAX_ACTIONS];
} EntryDump;

static int entry_cmp(const void *a, const void *b) {
    const EntryDump *ea = (const EntryDump*)a;
    const EntryDump *eb = (const EntryDump*)b;
    if (ea->key.player != eb->key.player) return ea->key.player - eb->key.player;
    if (ea->key.street != eb->key.street) return ea->key.street - eb->key.street;
    if (ea->key.bucket != eb->key.bucket) return ea->key.bucket - eb->key.bucket;
    if (ea->key.board_hash < eb->key.board_hash) return -1;
    if (ea->key.board_hash > eb->key.board_hash) return 1;
    if (ea->key.action_hash < eb->key.action_hash) return -1;
    if (ea->key.action_hash > eb->key.action_hash) return 1;
    return 0;
}

static int64_t dump_table(const BPSolver *s, EntryDump **out) {
    int64_t n = 0;
    for (int64_t i = 0; i < s->info_table.table_size; i++) {
        if (s->info_table.occupied[i] == 1) n++;
    }
    EntryDump *buf = (EntryDump*)calloc((size_t)n, sizeof(EntryDump));
    if (!buf) { *out = NULL; return -1; }
    int64_t k = 0;
    for (int64_t i = 0; i < s->info_table.table_size; i++) {
        if (s->info_table.occupied[i] != 1) continue;
        const BPInfoSet *is = &s->info_table.sets[i];
        EntryDump *d = &buf[k++];
        d->key = s->info_table.keys[i];
        d->num_actions = is->num_actions;
        for (int a = 0; a < is->num_actions; a++)
            d->regrets[a] = is->regrets[a];
        if (is->strategy_sum) {
            d->has_strategy_sum = 1;
            for (int a = 0; a < is->num_actions; a++)
                d->strategy_sum[a] = is->strategy_sum[a];
        } else {
            d->has_strategy_sum = 0;
        }
    }
    qsort(buf, (size_t)n, sizeof(EntryDump), entry_cmp);
    *out = buf;
    return n;
}

static int init_toy_solver(BPSolver *s, int train_iters) {
    (void)train_iters;
    /* Same toy game as test_phase_1_3_synthetic.c but using bp_init_ex
     * directly so we don't depend on that test file. */
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
    config.prune_start_iter = 100000000;  /* effectively disable */

    memset(s, 0, sizeof(*s));
    return bp_init_ex(s, 2, flop,
                      (const int (*)[BP_MAX_HANDS][2])hands,
                      (const float (*)[BP_MAX_HANDS])weights,
                      num_hands, 100, 100, bet_sizes, 1, &config);
}

int main(int argc, char **argv) {
    const char *chk_path = (argc > 1) ? argv[1] : "/tmp/toy_checkpoint.bin";
    int train_iters = (argc > 2) ? atoi(argv[2]) : 500000;

    printf("=== Loader diff test ===\n");
    printf("Checkpoint: %s\n", chk_path);
    printf("Train iters: %d\n", train_iters);
    fflush(stdout);

    /* Step 1: Train and save a toy checkpoint. */
    BPSolver s1;
    if (init_toy_solver(&s1, train_iters) != 0) {
        fprintf(stderr, "init failed\n"); return 1;
    }
    printf("\n[1/5] Training toy solver (%d iters)...\n", train_iters); fflush(stdout);
    if (bp_solve(&s1, train_iters) != 0) {
        fprintf(stderr, "bp_solve failed\n"); bp_free(&s1); return 1;
    }
    int64_t orig_entries = bp_num_info_sets(&s1);
    printf("Trained: %lld info sets\n", (long long)orig_entries);

    printf("\n[2/5] Saving to %s...\n", chk_path); fflush(stdout);
    if (bp_save_regrets(&s1, chk_path) != 0) {
        fprintf(stderr, "save failed\n"); bp_free(&s1); return 1;
    }
    /* Dump the ORIGINAL in-memory state too, for three-way comparison. */
    EntryDump *dump_orig = NULL;
    int64_t n_orig = dump_table(&s1, &dump_orig);
    printf("Original dump: %lld entries\n", (long long)n_orig);
    bp_free(&s1);

    /* Step 2: Load with legacy serial loader. */
    BPSolver s_serial;
    if (init_toy_solver(&s_serial, 0) != 0) {
        fprintf(stderr, "init failed\n"); return 1;
    }
    printf("\n[3/5] Loading with BP_LEGACY_LOADER=1 (serial)...\n"); fflush(stdout);
    set_env("BP_LEGACY_LOADER", "1");
    int64_t n_serial = bp_load_regrets(&s_serial, chk_path);
    if (n_serial <= 0) {
        fprintf(stderr, "serial load failed: %lld\n", (long long)n_serial);
        bp_free(&s_serial); return 1;
    }
    EntryDump *dump_serial = NULL;
    int64_t dn_serial = dump_table(&s_serial, &dump_serial);
    printf("Serial dump: %lld entries\n", (long long)dn_serial);
    bp_free(&s_serial);

    /* Step 3: Load with parallel mmap loader. */
    BPSolver s_parallel;
    if (init_toy_solver(&s_parallel, 0) != 0) {
        fprintf(stderr, "init failed\n"); return 1;
    }
    printf("\n[4/5] Loading with parallel mmap loader (default)...\n"); fflush(stdout);
    set_env("BP_LEGACY_LOADER", NULL);
    int64_t n_par = bp_load_regrets(&s_parallel, chk_path);
    if (n_par <= 0) {
        fprintf(stderr, "parallel load failed: %lld\n", (long long)n_par);
        bp_free(&s_parallel); return 1;
    }
    EntryDump *dump_par = NULL;
    int64_t dn_par = dump_table(&s_parallel, &dump_par);
    printf("Parallel dump: %lld entries\n", (long long)dn_par);
    bp_free(&s_parallel);

    /* Step 4: Three-way diff. */
    printf("\n[5/5] Diffing dumps...\n"); fflush(stdout);
    int fail = 0;
    if (n_orig != dn_serial || n_orig != dn_par) {
        printf("FAIL: entry count mismatch: orig=%lld serial=%lld parallel=%lld\n",
               (long long)n_orig, (long long)dn_serial, (long long)dn_par);
        fail = 1;
    } else {
        int64_t n = n_orig;
        int64_t n_diff_serial = 0;
        int64_t n_diff_par = 0;
        for (int64_t i = 0; i < n; i++) {
            const EntryDump *o = &dump_orig[i];
            const EntryDump *sv = &dump_serial[i];
            const EntryDump *pv = &dump_par[i];

            int serial_ok =
                o->key.player == sv->key.player &&
                o->key.street == sv->key.street &&
                o->key.bucket == sv->key.bucket &&
                o->key.board_hash == sv->key.board_hash &&
                o->key.action_hash == sv->key.action_hash &&
                o->num_actions == sv->num_actions &&
                o->has_strategy_sum == sv->has_strategy_sum;
            if (serial_ok) {
                for (int a = 0; a < o->num_actions; a++) {
                    if (o->regrets[a] != sv->regrets[a]) { serial_ok = 0; break; }
                    if (o->has_strategy_sum &&
                        fabsf(o->strategy_sum[a] - sv->strategy_sum[a]) > 1e-5f) {
                        serial_ok = 0; break;
                    }
                }
            }
            if (!serial_ok) n_diff_serial++;

            int par_ok =
                o->key.player == pv->key.player &&
                o->key.street == pv->key.street &&
                o->key.bucket == pv->key.bucket &&
                o->key.board_hash == pv->key.board_hash &&
                o->key.action_hash == pv->key.action_hash &&
                o->num_actions == pv->num_actions &&
                o->has_strategy_sum == pv->has_strategy_sum;
            if (par_ok) {
                for (int a = 0; a < o->num_actions; a++) {
                    if (o->regrets[a] != pv->regrets[a]) { par_ok = 0; break; }
                    if (o->has_strategy_sum &&
                        fabsf(o->strategy_sum[a] - pv->strategy_sum[a]) > 1e-5f) {
                        par_ok = 0; break;
                    }
                }
            }
            if (!par_ok) n_diff_par++;

            /* Print first few diffs for debuggability. */
            if (!serial_ok && n_diff_serial <= 5) {
                printf("  SERIAL DIFF [%lld]: key(%d,%d,%d,%016llx,%016llx)\n",
                       (long long)i, o->key.player, o->key.street, o->key.bucket,
                       (unsigned long long)o->key.board_hash,
                       (unsigned long long)o->key.action_hash);
            }
            if (!par_ok && n_diff_par <= 5) {
                printf("  PARALLEL DIFF [%lld]: key(%d,%d,%d,%016llx,%016llx)\n",
                       (long long)i, o->key.player, o->key.street, o->key.bucket,
                       (unsigned long long)o->key.board_hash,
                       (unsigned long long)o->key.action_hash);
                printf("    orig regrets: ");
                for (int a = 0; a < o->num_actions; a++) printf("%d ", o->regrets[a]);
                printf("\n    par  regrets: ");
                for (int a = 0; a < pv->num_actions; a++) printf("%d ", pv->regrets[a]);
                printf("\n");
                if (o->has_strategy_sum) {
                    printf("    orig ss: ");
                    for (int a = 0; a < o->num_actions; a++) printf("%.3f ", o->strategy_sum[a]);
                    printf("\n    par  ss: ");
                    for (int a = 0; a < pv->num_actions; a++) printf("%.3f ", pv->strategy_sum[a]);
                    printf("\n");
                }
            }
        }
        printf("  serial vs orig diffs: %lld / %lld\n", (long long)n_diff_serial, (long long)n);
        printf("  parallel vs orig diffs: %lld / %lld\n", (long long)n_diff_par, (long long)n);
        if (n_diff_serial > 0 || n_diff_par > 0) fail = 1;
    }

    free(dump_orig); free(dump_serial); free(dump_par);

    if (fail) {
        printf("\n=== LOADER DIFF TEST FAILED ===\n");
        return 1;
    }
    printf("\n=== ALL LOADER DIFFS MATCH — parallel mmap loader is correct ===\n");
    return 0;
}
