/*
 * Strategy consistency checker for BPR3 regret checkpoints.
 * Verifies poker-knowledge invariants in a single pass.
 *
 * Build: gcc -O2 -o strategy_checks verification/strategy_checks.c -lm
 * Usage: ./strategy_checks <checkpoint.bin>
 *
 * Output: JSON lines, one per check, for easy parsing by the Python wrapper.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#define MAX_ACTIONS 20

static const char *POS_NAMES[] = {"SB","BB","UTG","MP","CO","BTN"};

/* Preflop pair bucket indices */
static const int PAIR_BUCKETS[] = {0,25,48,69,88,105,120,133,144,153,160,165,168};
#define NUM_PAIR_BUCKETS 13

static int is_pair_bucket(int b) {
    for (int i = 0; i < NUM_PAIR_BUCKETS; i++)
        if (PAIR_BUCKETS[i] == b) return 1;
    return 0;
}

static void regret_match(const int *regrets, float *strat, int na) {
    float total = 0;
    for (int i = 0; i < na; i++) {
        float v = regrets[i] > 0 ? (float)regrets[i] : 0;
        strat[i] = v;
        total += v;
    }
    if (total > 0) {
        for (int i = 0; i < na; i++) strat[i] /= total;
    } else {
        for (int i = 0; i < na; i++) strat[i] = 1.0f / na;
    }
}

static uint64_t root_action_hash(void) {
    return 0xFEDCBA9876543210ULL;
}

/* ── Accumulators ──────────────────────────────────────────────────── */

/* Preflop check: AA fold frequency per position */
typedef struct {
    int count;
    float fold_sum;
} FoldAccum;

/* Stats for action distribution checks */
typedef struct {
    long long total;
    long long has_exact_zero;      /* any action with prob exactly 0.0 */
    long long has_exact_one;       /* any action with prob exactly 1.0 */
    long long num_actions_hist[MAX_ACTIONS + 1]; /* histogram of num_actions */
} GeneralStats;

/* Per-street bet/check frequency */
typedef struct {
    long long count;
    double bet_freq_sum;      /* sum of (1 - check/call prob) across entries */
    double check_freq_sum;    /* sum of check/call prob */
} StreetBetStats;

/* River near-nuts fold tracking */
typedef struct {
    int count;
    int fold_above_5pct;
    float max_fold;
} RiverNutsFold;

/* Preflop weak hand raise tracking from UTG */
typedef struct {
    int count;
    int raise_above_80pct;
    float max_raise;
    int worst_bucket;
} WeakUTGRaise;

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "/opt/blueprint_unified/regrets_latest.bin";
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return 1; }

    char *iobuf = malloc(16 * 1024 * 1024);
    setvbuf(f, iobuf, _IOFBF, 16 * 1024 * 1024);

    /* Header */
    char magic[4];
    int table_size, num_entries_hdr;
    int64_t iterations_run;
    fread(magic, 1, 4, f);
    int is_v3 = (memcmp(magic, "BPR3", 4) == 0);
    int is_v2 = (memcmp(magic, "BPR2", 4) == 0);
    if (!is_v3 && !is_v2) { fprintf(stderr, "Bad magic\n"); return 1; }
    fread(&table_size, 4, 1, f);
    fread(&num_entries_hdr, 4, 1, f);
    if (is_v3) {
        fread(&iterations_run, 8, 1, f);
    } else {
        int i32; fread(&i32, 4, 1, f);
        iterations_run = (int64_t)i32;
    }

    /* ── Accumulators ──────────────────────────────────────────────── */

    /* 1. AA (bucket 0) fold per position at root */
    FoldAccum aa_fold[6] = {{0}};

    /* 2. 72o (bucket 167) fold per position at root */
    FoldAccum sevtwo_fold[6] = {{0}};

    /* 3. Pocket pairs fold from BTN at root */
    FoldAccum pair_btn_fold[NUM_PAIR_BUCKETS] = {{0}};

    /* 4. Weak hands (bucket > 140) raise from UTG at root */
    WeakUTGRaise weak_utg = {0, 0, 0.0f, -1};

    /* 5. River near-nuts (bucket > 190) fold */
    RiverNutsFold river_nuts = {0, 0, 0.0f};

    /* 6. Per-street bet/check frequency */
    StreetBetStats street_stats[4] = {{0}};

    /* 7. General degenerate strategy detection */
    GeneralStats gen = {0};

    uint64_t root_ah = root_action_hash();

    int player, street, bucket, na, has_sum;
    uint64_t board_hash, action_hash;
    int regrets[MAX_ACTIONS];
    float strat[MAX_ACTIONS];

    long long total = 0;

    while (1) {
        if (fread(&player, 4, 1, f) != 1) break;
        if (fread(&street, 4, 1, f) != 1) break;
        if (fread(&bucket, 4, 1, f) != 1) break;
        if (fread(&board_hash, 8, 1, f) != 1) break;
        if (fread(&action_hash, 8, 1, f) != 1) break;
        if (fread(&na, 4, 1, f) != 1) break;
        if (na < 1 || na > MAX_ACTIONS) break;
        if (fread(regrets, 4, na, f) != (size_t)na) break;
        if (fread(&has_sum, 4, 1, f) != 1) break;
        if (has_sum) {
            float ss[MAX_ACTIONS];
            if (fread(ss, 4, na, f) != (size_t)na) break;
        }

        regret_match(regrets, strat, na);
        total++;

        /* ── General checks (all streets) ──────────────────────────── */
        if (na <= MAX_ACTIONS) gen.num_actions_hist[na]++;
        gen.total++;

        int has_zero = 0, has_one = 0;
        for (int i = 0; i < na; i++) {
            if (strat[i] == 0.0f) has_zero = 1;
            if (strat[i] == 1.0f) has_one = 1;
        }
        if (has_zero) gen.has_exact_zero++;
        if (has_one) gen.has_exact_one++;

        /* ── Preflop root checks ───────────────────────────────────── */
        if (street == 0 && action_hash == root_ah && player >= 0 && player < 6) {
            float fold_p = (na > 0) ? strat[0] : 0.0f;

            /* AA fold */
            if (bucket == 0) {
                aa_fold[player].count++;
                aa_fold[player].fold_sum += fold_p;
            }

            /* 72o fold */
            if (bucket == 167) {
                sevtwo_fold[player].count++;
                sevtwo_fold[player].fold_sum += fold_p;
            }

            /* Pocket pairs from BTN */
            if (player == 5) { /* BTN */
                for (int pi = 0; pi < NUM_PAIR_BUCKETS; pi++) {
                    if (bucket == PAIR_BUCKETS[pi]) {
                        pair_btn_fold[pi].count++;
                        pair_btn_fold[pi].fold_sum += fold_p;
                    }
                }
            }

            /* Weak hands from UTG (position 2) */
            if (player == 2 && bucket > 140) {
                float raise_p = 0;
                for (int a = 2; a < na; a++) raise_p += strat[a];
                weak_utg.count++;
                if (raise_p > 0.80f) {
                    weak_utg.raise_above_80pct++;
                    if (raise_p > weak_utg.max_raise) {
                        weak_utg.max_raise = raise_p;
                        weak_utg.worst_bucket = bucket;
                    }
                }
            }
        }

        /* ── Postflop checks ──────────────────────────────────────── */
        if (street >= 1 && street <= 3) {
            /* Bet vs check frequency */
            float check_p = (na > 1) ? strat[1] : 0.0f; /* action 1 = check/call */
            /* fold is action 0 (if facing bet), but for bet/check stats
             * we care about the non-fold actions */
            float bet_p = 0;
            for (int a = 2; a < na; a++) bet_p += strat[a];
            street_stats[street].count++;
            street_stats[street].bet_freq_sum += bet_p;
            street_stats[street].check_freq_sum += check_p;
        }

        /* River near-nuts fold */
        if (street == 3 && bucket > 190) {
            float fold_p = (na > 0) ? strat[0] : 0.0f;
            river_nuts.count++;
            if (fold_p > 0.05f) {
                river_nuts.fold_above_5pct++;
                if (fold_p > river_nuts.max_fold)
                    river_nuts.max_fold = fold_p;
            }
        }

        if (total % 100000000 == 0)
            fprintf(stderr, "  %lld entries...\n", total);
    }
    fclose(f);
    free(iobuf);

    /* ── Output results as structured text ─────────────────────────── */

    printf("STRATEGY_CHECKS iterations=%lld entries=%lld\n\n", (long long)iterations_run, total);

    /* Check 1: AA never folds (fold < 1%) */
    printf("=== CHECK: AA_NEVER_FOLDS ===\n");
    int aa_pass = 1;
    for (int p = 0; p < 6; p++) {
        if (aa_fold[p].count > 0) {
            float avg_fold = aa_fold[p].fold_sum / aa_fold[p].count;
            printf("  %s: fold=%.4f %s\n", POS_NAMES[p], avg_fold,
                   avg_fold < 0.01f ? "OK" : "FAIL");
            if (avg_fold >= 0.01f) aa_pass = 0;
        }
    }
    printf("RESULT: %s\n\n", aa_pass ? "PASS" : "FAIL");

    /* Check 2: 72o folds >90% from UTG/MP/CO */
    printf("=== CHECK: 72o_FOLDS_EARLY ===\n");
    int sevtwo_pass = 1;
    int early_positions[] = {2, 3, 4}; /* UTG, MP, CO */
    for (int i = 0; i < 3; i++) {
        int p = early_positions[i];
        if (sevtwo_fold[p].count > 0) {
            float avg_fold = sevtwo_fold[p].fold_sum / sevtwo_fold[p].count;
            printf("  %s: fold=%.4f %s\n", POS_NAMES[p], avg_fold,
                   avg_fold > 0.90f ? "OK" : "FAIL");
            if (avg_fold <= 0.90f) sevtwo_pass = 0;
        } else {
            printf("  %s: no data\n", POS_NAMES[p]);
        }
    }
    printf("RESULT: %s\n\n", sevtwo_pass ? "PASS" : "FAIL");

    /* Check 3: Pocket pairs fold <50% from BTN */
    printf("=== CHECK: PAIRS_BTN_DONT_FOLD ===\n");
    int pairs_pass = 1;
    static const char *PAIR_NAMES[] = {"AA","KK","QQ","JJ","TT","99","88","77","66","55","44","33","22"};
    for (int pi = 0; pi < NUM_PAIR_BUCKETS; pi++) {
        if (pair_btn_fold[pi].count > 0) {
            float avg_fold = pair_btn_fold[pi].fold_sum / pair_btn_fold[pi].count;
            printf("  %s (bucket %d): fold=%.4f %s\n",
                   PAIR_NAMES[pi], PAIR_BUCKETS[pi], avg_fold,
                   avg_fold < 0.50f ? "OK" : "FAIL");
            if (avg_fold >= 0.50f) pairs_pass = 0;
        }
    }
    printf("RESULT: %s\n\n", pairs_pass ? "PASS" : "FAIL");

    /* Check 4: No weak hand (bucket > 140) raises >80% from UTG */
    printf("=== CHECK: WEAK_UTG_NO_RAISE ===\n");
    printf("  Checked %d weak-hand entries from UTG\n", weak_utg.count);
    printf("  Raise >80%%: %d entries\n", weak_utg.raise_above_80pct);
    if (weak_utg.raise_above_80pct > 0) {
        printf("  Max raise: %.4f at bucket %d\n", weak_utg.max_raise, weak_utg.worst_bucket);
    }
    printf("RESULT: %s\n\n", weak_utg.raise_above_80pct == 0 ? "PASS" : "FAIL");

    /* Check 5: River near-nuts (bucket > 190) doesn't fold >5% */
    printf("=== CHECK: RIVER_NUTS_DONT_FOLD ===\n");
    printf("  Checked %d river near-nuts entries (bucket > 190)\n", river_nuts.count);
    printf("  Fold >5%%: %d entries\n", river_nuts.fold_above_5pct);
    if (river_nuts.fold_above_5pct > 0) {
        printf("  Max fold: %.4f\n", river_nuts.max_fold);
    }
    int nuts_pass = (river_nuts.count == 0) ||
                    ((float)river_nuts.fold_above_5pct / river_nuts.count < 0.05f);
    printf("RESULT: %s\n\n", nuts_pass ? "PASS" : "FAIL");

    /* Check 6: Flop avg bet freq < avg check freq */
    printf("=== CHECK: FLOP_CHECK_MORE_THAN_BET ===\n");
    if (street_stats[1].count > 0) {
        double avg_bet = street_stats[1].bet_freq_sum / street_stats[1].count;
        double avg_check = street_stats[1].check_freq_sum / street_stats[1].count;
        printf("  Flop avg bet freq: %.4f\n", avg_bet);
        printf("  Flop avg check/call freq: %.4f\n", avg_check);
        printf("RESULT: %s\n\n", avg_bet < avg_check ? "PASS" : "FAIL");
    } else {
        printf("  No flop entries\n");
        printf("RESULT: SKIP\n\n");
    }

    /* Check 7: Turn bet freq > Flop bet freq */
    printf("=== CHECK: TURN_BET_MORE_THAN_FLOP ===\n");
    if (street_stats[1].count > 0 && street_stats[2].count > 0) {
        double flop_bet = street_stats[1].bet_freq_sum / street_stats[1].count;
        double turn_bet = street_stats[2].bet_freq_sum / street_stats[2].count;
        printf("  Flop avg bet freq: %.4f\n", flop_bet);
        printf("  Turn avg bet freq: %.4f\n", turn_bet);
        printf("RESULT: %s\n\n", turn_bet > flop_bet ? "PASS" : "FAIL");
    } else {
        printf("  Insufficient data\n");
        printf("RESULT: SKIP\n\n");
    }

    /* Check 8: No degenerate strategies (exact 0.0 or 1.0 for >50% of info sets) */
    printf("=== CHECK: NO_DEGENERATE_STRATEGIES ===\n");
    double zero_pct = gen.total > 0 ? (double)gen.has_exact_zero / gen.total * 100 : 0;
    double one_pct = gen.total > 0 ? (double)gen.has_exact_one / gen.total * 100 : 0;
    printf("  Exact 0.0 in any action: %lld / %lld (%.1f%%)\n",
           gen.has_exact_zero, gen.total, zero_pct);
    printf("  Exact 1.0 in any action: %lld / %lld (%.1f%%)\n",
           gen.has_exact_one, gen.total, one_pct);
    int degen_pass = (zero_pct < 50.0) && (one_pct < 50.0);
    printf("RESULT: %s\n\n", degen_pass ? "PASS" : "FAIL");

    /* Check 9: Action count distribution */
    printf("=== CHECK: ACTION_COUNT_DISTRIBUTION ===\n");
    long long low_action = gen.num_actions_hist[1] + gen.num_actions_hist[2];
    long long mid_action = 0;
    for (int i = 3; i <= 6; i++) mid_action += gen.num_actions_hist[i];
    long long high_action = 0;
    for (int i = 7; i <= MAX_ACTIONS; i++) high_action += gen.num_actions_hist[i];
    printf("  1-2 actions: %lld (%.1f%%)\n", low_action,
           gen.total > 0 ? (double)low_action / gen.total * 100 : 0);
    printf("  3-6 actions: %lld (%.1f%%)\n", mid_action,
           gen.total > 0 ? (double)mid_action / gen.total * 100 : 0);
    printf("  7+  actions: %lld (%.1f%%)\n", high_action,
           gen.total > 0 ? (double)high_action / gen.total * 100 : 0);
    for (int i = 1; i <= 8; i++) {
        if (gen.num_actions_hist[i] > 0)
            printf("    na=%d: %lld\n", i, gen.num_actions_hist[i]);
    }
    /* Most entries should have 3-6 actions */
    int action_pass = (gen.total == 0) || (mid_action >= low_action);
    printf("RESULT: %s\n\n", action_pass ? "PASS" : "FAIL");

    /* Summary */
    printf("=== SUMMARY ===\n");
    int checks_passed = aa_pass + sevtwo_pass + pairs_pass +
                        (weak_utg.raise_above_80pct == 0) +
                        nuts_pass + degen_pass + action_pass;
    int checks_total = 7;
    /* Add conditional checks */
    if (street_stats[1].count > 0) {
        double avg_bet = street_stats[1].bet_freq_sum / street_stats[1].count;
        double avg_check = street_stats[1].check_freq_sum / street_stats[1].count;
        checks_total++;
        if (avg_bet < avg_check) checks_passed++;
    }
    if (street_stats[1].count > 0 && street_stats[2].count > 0) {
        double flop_bet = street_stats[1].bet_freq_sum / street_stats[1].count;
        double turn_bet = street_stats[2].bet_freq_sum / street_stats[2].count;
        checks_total++;
        if (turn_bet > flop_bet) checks_passed++;
    }
    printf("  %d / %d checks passed\n", checks_passed, checks_total);

    return (checks_passed == checks_total) ? 0 : 1;
}
