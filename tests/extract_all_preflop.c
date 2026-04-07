/*
 * Extract ALL 169 preflop class strategies for ALL 6 positions.
 * Prints a full strategy table sorted by expected action.
 *
 * Build: gcc -O2 -o extract_all tests/extract_all_preflop.c -lm
 * Usage: ./extract_all <checkpoint.bin>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

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

typedef struct {
    int player;
    int bucket;
    int na;
    int regrets[MAX_ACTIONS];
    int found;
} Root;

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "regrets_2B.bin";
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return 1; }

    char *iobuf = malloc(16 * 1024 * 1024);
    setvbuf(f, iobuf, _IOFBF, 16 * 1024 * 1024);

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
        else { int i; fread(&i, 4, 1, f); iters = i; }
    }

    printf("Checkpoint: format=%s table=%lld entries=%lld iters=%lld\n\n",
           is_v4 ? "BPR4" : is_v3 ? "BPR3" : "BPR2",
           (long long)table_size, (long long)num_entries, (long long)iters);

    uint64_t root_ah = 0xFEDCBA9876543210ULL;

    /* Storage for all 6 × 169 roots */
    Root roots[6][169];
    memset(roots, 0, sizeof(roots));

    int player, street, bucket, na, has_sum;
    uint64_t board_hash, action_hash;
    int regrets[MAX_ACTIONS];
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

        if (street == 0 && action_hash == root_ah &&
            player >= 0 && player < 6 && bucket >= 0 && bucket < 169) {
            Root *r = &roots[player][bucket];
            r->player = player;
            r->bucket = bucket;
            r->na = na;
            memcpy(r->regrets, regrets, na * sizeof(int));
            r->found = 1;
        }

        total++;
        if (total % 200000000 == 0) fprintf(stderr, "  %lldM entries...\n", total/1000000);
    }
    fclose(f);
    free(iobuf);

    /* Count found */
    int found_total = 0;
    for (int p = 0; p < 6; p++)
        for (int b = 0; b < 169; b++)
            if (roots[p][b].found) found_total++;
    printf("Found %d / %d preflop root info sets\n\n", found_total, 6*169);

    /* Print strategy table for each position */
    for (int p = 0; p < 6; p++) {
        printf("========== %s (player %d) ==========\n", POS[p], p);
        printf("%-5s  %5s %5s %5s %5s   action\n", "Hand", "fold", "call", "raise", "best");

        int n_fold = 0, n_call = 0, n_raise = 0, n_missing = 0;

        for (int b = 0; b < 169; b++) {
            Root *r = &roots[p][b];
            if (!r->found) {
                printf("%-5s  [NOT FOUND]\n", LABELS[b]);
                n_missing++;
                continue;
            }

            float strat[MAX_ACTIONS];
            regret_match(r->regrets, strat, r->na);

            /* Aggregate: fold = strat[0], call = strat[1], raise = sum of rest */
            float fold_p = 0, call_p = 0, raise_p = 0;
            if (r->na >= 1) fold_p = strat[0];
            if (r->na >= 2) call_p = strat[1];
            for (int a = 2; a < r->na; a++) raise_p += strat[a];

            /* Determine dominant action */
            const char *best;
            if (fold_p >= call_p && fold_p >= raise_p) { best = "FOLD"; n_fold++; }
            else if (call_p >= fold_p && call_p >= raise_p) { best = "CALL"; n_call++; }
            else { best = "RAISE"; n_raise++; }

            /* Flag suspicious: trash hand raising or premium folding */
            int suspicious = 0;
            /* Pairs AA-TT and broadway should raise */
            if (b <= 4 || b == 25 || b == 48 || b == 68 || b == 86) {
                if (fold_p > 0.5) suspicious = 1;
            }
            /* Trash (bottom 30 classes) should mostly fold from UTG/MP */
            if (b >= 140 && p >= 2 && p <= 3) {
                if (raise_p > 0.5) suspicious = 1;
            }

            printf("%-5s  %5.1f %5.1f %5.1f   %-5s%s\n",
                   LABELS[b], fold_p*100, call_p*100, raise_p*100, best,
                   suspicious ? "  *** SUSPICIOUS ***" : "");
        }

        printf("\nSummary: %d fold, %d call, %d raise, %d missing\n\n",
               n_fold, n_call, n_raise, n_missing);
    }

    /* Sanity checks */
    printf("========== SANITY CHECKS ==========\n");
    int issues = 0;

    /* Check all positions for AA raising */
    for (int p = 0; p < 6; p++) {
        Root *r = &roots[p][0]; /* AA = bucket 0 */
        if (!r->found) { printf("WARN: AA not found for %s\n", POS[p]); issues++; continue; }
        float strat[MAX_ACTIONS];
        regret_match(r->regrets, strat, r->na);
        float raise_p = 0;
        for (int a = 2; a < r->na; a++) raise_p += strat[a];
        if (raise_p < 0.5) {
            printf("FAIL: %s AA raise=%.0f%% (should be >50%%)\n", POS[p], raise_p*100);
            issues++;
        }
    }

    /* Check UTG trash hands fold */
    int trash[] = {140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,
                   155,156,157,158,159,160,161,162,163,164,165,166,167,168};
    for (int ti = 0; ti < 29; ti++) {
        int b = trash[ti];
        Root *r = &roots[2][b]; /* UTG = player 2 */
        if (!r->found) continue;
        float strat[MAX_ACTIONS];
        regret_match(r->regrets, strat, r->na);
        float fold_p = strat[0];
        if (fold_p < 0.5) {
            printf("FAIL: UTG %s fold=%.0f%% (should be >50%%)\n", LABELS[b], fold_p*100);
            issues++;
        }
    }

    /* Check that later positions are looser than earlier */
    /* UTG should fold more than BTN for marginal hands */
    int marginal[] = {84, 120, 150}; /* J3s, 88, 63o */
    for (int mi = 0; mi < 3; mi++) {
        int b = marginal[mi];
        Root *utg = &roots[2][b], *btn = &roots[5][b];
        if (!utg->found || !btn->found) continue;
        float s_utg[MAX_ACTIONS], s_btn[MAX_ACTIONS];
        regret_match(utg->regrets, s_utg, utg->na);
        regret_match(btn->regrets, s_btn, btn->na);
        if (s_utg[0] < s_btn[0] - 0.1) {
            printf("WARN: %s folds less from UTG (%.0f%%) than BTN (%.0f%%)\n",
                   LABELS[b], s_utg[0]*100, s_btn[0]*100);
            issues++;
        }
    }

    if (issues == 0) printf("All sanity checks passed.\n");
    else printf("\n%d issues found.\n", issues);

    return 0;
}
