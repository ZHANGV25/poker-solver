/*
 * Extract preflop strategies for ALL 6 positions by scanning all info sets.
 *
 * Unlike extract_all_preflop.c which only finds UTG (hardcoded root hash),
 * this scans every entry with street==0 and aggregates by (player, bucket).
 * For non-UTG positions, there are multiple root scenarios (one per
 * preceding action sequence). We aggregate by averaging regret-matched
 * strategies across all scenarios.
 *
 * Build: gcc -O2 -o extract_positions tests/extract_all_positions.c -lm
 * Usage: ./extract_positions <checkpoint.bin>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define MAX_ACTIONS 20

static const char *LABELS[] = {
    "AA","AKs","AKo","AQs","AQo","AJs","AJo","ATs","ATo","A9s","A9o","A8s","A8o",
    "A7s","A7o","A6s","A6o","A5s","A5o","A4s","A4o","A3s","A3o","A2s","A2o",
    "KK","KQs","KQo","KJs","KJo","KTs","KTo","K9s","K9o","K8s","K8o","K7s","K7o",
    "K6s","K6o","K5s","K5o","K4s","K4o","K3s","K3o","K2s","K2o",
    "QQ","QJs","QJo","QTs","QTo","Q9s","Q9o","Q8s","Q8o","Q7s","Q7o","Q6s","Q6o",
    "Q5s","Q5o","Q4s","Q4o","Q3s","Q3o","Q2s","Q2o",
    "JJ","JTs","JTo","J9s","J9o","J8s","J8o","J7s","J7o","J6s","J6o","J5s","J5o",
    "J4s","J4o","J3s","J3o","J2s","J2o",
    "TT","T9s","T9o","T8s","T8o","T7s","T7o","T6s","T6o","T5s","T5o","T4s","T4o",
    "T3s","T3o","T2s","T2o",
    "99","98s","98o","97s","97o","96s","96o","95s","95o","94s","94o","93s","93o",
    "92s","92o",
    "88","87s","87o","86s","86o","85s","85o","84s","84o","83s","83o","82s","82o",
    "77","76s","76o","75s","75o","74s","74o","73s","73o","72s","72o",
    "66","65s","65o","64s","64o","63s","63o","62s","62o",
    "55","54s","54o","53s","53o","52s","52o",
    "44","43s","43o","42s","42o",
    "33","32s","32o",
    "22"
};
static const char *POS[] = {"SB","BB","UTG","MP","CO","BTN"};

static void regret_match(const int *regrets, float *strat, int na) {
    float sum = 0;
    for (int i = 0; i < na; i++) {
        strat[i] = regrets[i] > 0 ? (float)regrets[i] : 0;
        sum += strat[i];
    }
    if (sum > 0) for (int i = 0; i < na; i++) strat[i] /= sum;
    else for (int i = 0; i < na; i++) strat[i] = 1.0f / na;
}

/* Accumulated strategy per (player, bucket) */
typedef struct {
    float fold_sum;
    float call_sum;
    float raise_sum;
    int count;  /* number of info sets aggregated */
} Agg;

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "regrets_latest.bin";
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return 1; }

    char *iobuf = malloc(16 * 1024 * 1024);
    setvbuf(f, iobuf, _IOFBF, 16 * 1024 * 1024);

    /* Read header */
    char magic[4];
    int64_t table_size, num_entries, iters;
    fread(magic, 1, 4, f);
    int is_v4 = (memcmp(magic, "BPR4", 4) == 0);
    int is_v3 = (memcmp(magic, "BPR3", 4) == 0);
    int is_v2 = (memcmp(magic, "BPR2", 4) == 0);
    if (!is_v4 && !is_v3 && !is_v2) {
        fprintf(stderr, "Bad magic: %.4s\n", magic); return 1;
    }
    if (is_v4) {
        fread(&table_size, 8, 1, f);
        fread(&num_entries, 8, 1, f);
        fread(&iters, 8, 1, f);
    } else {
        int ts32, ne32;
        fread(&ts32, 4, 1, f); table_size = ts32;
        fread(&ne32, 4, 1, f); num_entries = ne32;
        if (is_v3) fread(&iters, 8, 1, f);
        else { int i32; fread(&i32, 4, 1, f); iters = i32; }
    }

    printf("Checkpoint: format=%s entries=%lld iters=%lld\n\n",
           is_v4 ? "BPR4" : is_v3 ? "BPR3" : "BPR2",
           (long long)num_entries, (long long)iters);

    /* Aggregated strategies: [player][bucket] */
    Agg agg[6][169];
    memset(agg, 0, sizeof(agg));

    int player, street, bucket, na, has_sum;
    uint64_t board_hash, action_hash;
    int regrets[MAX_ACTIONS];
    long long total = 0, preflop_count = 0;

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

        /* Collect all preflop info sets */
        if (street == 0 && player >= 0 && player < 6 &&
            bucket >= 0 && bucket < 169 && na >= 2) {

            float strat[MAX_ACTIONS];
            regret_match(regrets, strat, na);

            /* Aggregate: fold = strat[0], call = strat[1], raise = sum of rest */
            float fold_p = strat[0];
            float call_p = strat[1];
            float raise_p = 0;
            for (int a = 2; a < na; a++) raise_p += strat[a];

            Agg *a = &agg[player][bucket];
            a->fold_sum += fold_p;
            a->call_sum += call_p;
            a->raise_sum += raise_p;
            a->count++;
            preflop_count++;
        }

        total++;
        if (total % 200000000 == 0)
            fprintf(stderr, "  %lldM entries scanned...\n", total / 1000000);
    }
    fclose(f);
    free(iobuf);

    printf("Scanned %lld entries, %lld preflop\n\n", total, preflop_count);

    /* Print per-position info set counts */
    printf("Preflop info sets per position:\n");
    for (int p = 0; p < 6; p++) {
        int total_is = 0, buckets_found = 0;
        for (int b = 0; b < 169; b++) {
            total_is += agg[p][b].count;
            if (agg[p][b].count > 0) buckets_found++;
        }
        printf("  %s: %d info sets across %d/169 buckets (avg %.1f scenarios/bucket)\n",
               POS[p], total_is, buckets_found,
               buckets_found > 0 ? (float)total_is / buckets_found : 0);
    }
    printf("\n");

    /* Print strategy table for each position */
    for (int p = 0; p < 6; p++) {
        printf("========== %s (player %d) ==========\n", POS[p], p);
        printf("%-5s  %5s %5s %5s %5s  scenarios\n",
               "Hand", "fold", "call", "raise", "best");

        int n_fold = 0, n_call = 0, n_raise = 0, n_missing = 0;

        for (int b = 0; b < 169; b++) {
            Agg *a = &agg[p][b];
            if (a->count == 0) {
                printf("%-5s  [NOT FOUND]\n", LABELS[b]);
                n_missing++;
                continue;
            }

            float fold_p = a->fold_sum / a->count;
            float call_p = a->call_sum / a->count;
            float raise_p = a->raise_sum / a->count;

            const char *best;
            if (fold_p >= call_p && fold_p >= raise_p) { best = "FOLD"; n_fold++; }
            else if (call_p >= fold_p && call_p >= raise_p) { best = "CALL"; n_call++; }
            else { best = "RAISE"; n_raise++; }

            printf("%-5s  %5.1f %5.1f %5.1f   %-5s  %4d\n",
                   LABELS[b], fold_p*100, call_p*100, raise_p*100, best, a->count);
        }

        printf("\nSummary: %d fold, %d call, %d raise, %d missing\n\n",
               n_fold, n_call, n_raise, n_missing);
    }

    /* Quick sanity checks */
    printf("========== SANITY CHECKS ==========\n");
    int issues = 0;

    /* AA should raise from all positions */
    for (int p = 0; p < 6; p++) {
        Agg *a = &agg[p][0];
        if (a->count == 0) { printf("WARN: AA not found for %s\n", POS[p]); issues++; continue; }
        float raise_p = a->raise_sum / a->count;
        if (raise_p < 0.5) {
            printf("FAIL: %s AA raise=%.0f%% (should be >50%%)\n", POS[p], raise_p*100);
            issues++;
        }
    }

    /* 32o should fold from UTG/MP */
    for (int p = 2; p <= 3; p++) {
        Agg *a = &agg[p][167]; /* 32o */
        if (a->count == 0) continue;
        float fold_p = a->fold_sum / a->count;
        if (fold_p < 0.5) {
            printf("FAIL: %s 32o fold=%.0f%% (should be >50%%)\n", POS[p], fold_p*100);
            issues++;
        }
    }

    /* Later positions should be looser than earlier (fewer folds for marginal hands) */
    int marginal_buckets[] = {86, 99, 120}; /* TT, T9o, 88 */
    for (int mi = 0; mi < 3; mi++) {
        int b = marginal_buckets[mi];
        Agg *utg = &agg[2][b], *btn = &agg[5][b];
        if (utg->count == 0 || btn->count == 0) continue;
        float utg_raise = utg->raise_sum / utg->count;
        float btn_raise = btn->raise_sum / btn->count;
        if (utg_raise > btn_raise + 0.15) {
            printf("WARN: %s raises more from UTG (%.0f%%) than BTN (%.0f%%)\n",
                   LABELS[b], utg_raise*100, btn_raise*100);
            issues++;
        }
    }

    if (issues == 0) printf("All sanity checks passed.\n");
    else printf("\n%d issues found.\n", issues);

    return 0;
}
