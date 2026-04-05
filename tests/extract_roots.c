/*
 * Extract ROOT preflop strategies for all 6 positions.
 * Root = "folds to me" scenario for each position.
 *
 * Action hashes computed for:
 *   UTG: []              (first to act)
 *   MP:  [0]             (UTG folds)
 *   CO:  [0,0]           (UTG,MP fold)
 *   BTN: [0,0,0]         (UTG,MP,CO fold)
 *   SB:  [0,0,0,0]       (UTG,MP,CO,BTN fold)
 *   BB:  [0,0,0,0,1]     (all fold, SB completes)
 *
 * Build: gcc -O2 -o extract_roots tests/extract_roots.c -lm
 * Usage: ./extract_roots <checkpoint.bin>
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
/* Preflop acting order: UTG(2), MP(3), CO(4), BTN(5), SB(0), BB(1) */
static const char *POS[] = {"SB","BB","UTG","MP","CO","BTN"};
static const int ACTING_ORDER[] = {2, 3, 4, 5, 0, 1}; /* who acts 1st..6th */

static inline uint64_t hash_combine(uint64_t a, uint64_t b) {
    a ^= b + 0x9e3779b97f4a7c15ULL + (a << 6) + (a >> 2);
    return a;
}

static uint64_t compute_action_hash(const int *actions, int n) {
    uint64_t h = 0xFEDCBA9876543210ULL;
    for (int i = 0; i < n; i++)
        h = hash_combine(h, (uint64_t)actions[i] * 17 + 3);
    return h;
}

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
    float strategy_sum[MAX_ACTIONS];
    int has_sum;
    int found;
} Root;

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "regrets_latest.bin";
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
        else { int i32; fread(&i32, 4, 1, f); iters = i32; }
    }
    printf("Checkpoint: %lld entries, %lld iterations\n\n",
           (long long)num_entries, (long long)iters);

    /* Compute "folds to me" hash for each acting position.
     * Preflop order: UTG(2), MP(3), CO(4), BTN(5), SB(0), BB(1)
     * Action index 0 = fold, 1 = call/check */
    uint64_t root_hashes[6];
    int root_players[6];

    /* UTG (acts 1st): no prior actions */
    int acts0[] = {};
    root_hashes[0] = compute_action_hash(acts0, 0);
    root_players[0] = 2; /* UTG = player 2 */

    /* MP (acts 2nd): UTG folds */
    int acts1[] = {0};
    root_hashes[1] = compute_action_hash(acts1, 1);
    root_players[1] = 3;

    /* CO (acts 3rd): UTG, MP fold */
    int acts2[] = {0, 0};
    root_hashes[2] = compute_action_hash(acts2, 2);
    root_players[2] = 4;

    /* BTN (acts 4th): UTG, MP, CO fold */
    int acts3[] = {0, 0, 0};
    root_hashes[3] = compute_action_hash(acts3, 3);
    root_players[3] = 5;

    /* SB (acts 5th): UTG, MP, CO, BTN fold */
    int acts4[] = {0, 0, 0, 0};
    root_hashes[4] = compute_action_hash(acts4, 4);
    root_players[4] = 0;

    /* BB (acts 6th): all fold, SB completes (call = action 1) */
    int acts5[] = {0, 0, 0, 0, 1};
    root_hashes[5] = compute_action_hash(acts5, 5);
    root_players[5] = 1;

    printf("Root hashes (folds-to-me scenarios):\n");
    const char *root_labels[] = {"UTG open", "MP open", "CO open",
                                  "BTN open", "SB open", "BB vs limp"};
    for (int i = 0; i < 6; i++)
        printf("  %s (player %d): hash=%016llx\n",
               root_labels[i], root_players[i], (unsigned long long)root_hashes[i]);
    printf("\n");

    /* Storage: [root_idx][bucket] */
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
        float ss[MAX_ACTIONS];
        memset(ss, 0, sizeof(ss));
        if (has_sum) {
            if (fread(ss, 4, na, f) != (size_t)na) break;
        }

        if (street == 0 && bucket >= 0 && bucket < 169) {
            for (int ri = 0; ri < 6; ri++) {
                if (player == root_players[ri] && action_hash == root_hashes[ri]) {
                    Root *r = &roots[ri][bucket];
                    r->player = player;
                    r->bucket = bucket;
                    r->na = na;
                    memcpy(r->regrets, regrets, na * sizeof(int));
                    memcpy(r->strategy_sum, ss, na * sizeof(float));
                    r->has_sum = has_sum;
                    r->found = 1;
                }
            }
        }

        total++;
        if (total % 200000000 == 0)
            fprintf(stderr, "  %lldM entries...\n", total / 1000000);

        /* Early exit: stop scanning once all 6×169 root entries found */
        int all_found = 1;
        for (int ri = 0; ri < 6 && all_found; ri++)
            for (int b = 0; b < 169 && all_found; b++)
                if (!roots[ri][b].found) all_found = 0;
        if (all_found) {
            fprintf(stderr, "  All roots found at entry %lldM, stopping early\n",
                    total / 1000000);
            break;
        }
    }
    fclose(f);
    free(iobuf);

    /* Print strategy tables */
    for (int ri = 0; ri < 6; ri++) {
        int found = 0;
        for (int b = 0; b < 169; b++) if (roots[ri][b].found) found++;

        printf("========== %s — %s (player %d) [%d/169 found] ==========\n",
               root_labels[ri], POS[root_players[ri]], root_players[ri], found);
        printf("%-5s  %5s %5s %5s   %-5s  | %5s %5s %5s   %-5s\n",
               "Hand", "fold", "call", "raise", "best",
               "aFld", "aCll", "aRse", "avg");

        int n_fold = 0, n_call = 0, n_raise = 0, n_miss = 0;
        int a_fold = 0, a_call = 0, a_raise = 0;
        for (int b = 0; b < 169; b++) {
            Root *r = &roots[ri][b];
            if (!r->found) { n_miss++; continue; }

            /* Regret-matched (current snapshot) */
            float strat[MAX_ACTIONS];
            regret_match(r->regrets, strat, r->na);

            float fold_p = (r->na >= 1) ? strat[0] : 0;
            float call_p = (r->na >= 2) ? strat[1] : 0;
            float raise_p = 0;
            for (int a = 2; a < r->na; a++) raise_p += strat[a];

            const char *best;
            if (fold_p >= call_p && fold_p >= raise_p) { best = "FOLD"; n_fold++; }
            else if (call_p >= fold_p && call_p >= raise_p) { best = "CALL"; n_call++; }
            else { best = "RAISE"; n_raise++; }

            /* Average strategy from strategy_sum */
            float af = 0, ac = 0, ar = 0;
            const char *abest = "---";
            if (r->has_sum) {
                float sum = 0;
                for (int a = 0; a < r->na; a++) sum += r->strategy_sum[a];
                if (sum > 0) {
                    af = (r->na >= 1) ? r->strategy_sum[0] / sum : 0;
                    ac = (r->na >= 2) ? r->strategy_sum[1] / sum : 0;
                    ar = 0;
                    for (int a = 2; a < r->na; a++) ar += r->strategy_sum[a] / sum;

                    if (af >= ac && af >= ar) { abest = "FOLD"; a_fold++; }
                    else if (ac >= af && ac >= ar) { abest = "CALL"; a_call++; }
                    else { abest = "RAISE"; a_raise++; }
                }
            }

            printf("%-5s  %5.1f %5.1f %5.1f   %-5s  | %5.1f %5.1f %5.1f   %-5s\n",
                   LABELS[b], fold_p*100, call_p*100, raise_p*100, best,
                   af*100, ac*100, ar*100, abest);
        }
        printf("Summary: %d fold, %d call, %d raise, %d missing\n",
               n_fold, n_call, n_raise, n_miss);
        printf("Avg:     %d fold, %d call, %d raise\n\n",
               a_fold, a_call, a_raise);
    }

    /* Sanity checks */
    printf("========== SANITY CHECKS ==========\n");
    int issues = 0;
    for (int ri = 0; ri < 6; ri++) {
        Root *r = &roots[ri][0]; /* AA */
        if (!r->found) { printf("WARN: AA not found for %s\n", root_labels[ri]); issues++; continue; }
        float s[MAX_ACTIONS]; regret_match(r->regrets, s, r->na);
        float rp = 0; for (int a = 2; a < r->na; a++) rp += s[a];
        if (rp < 0.8)
            printf("CHECK: %s AA raise=%.0f%% (expected >80%%)\n", root_labels[ri], rp*100);
        else
            printf("  OK: %s AA raise=%.0f%%\n", root_labels[ri], rp*100);
    }
    /* Check 32o folds from early positions */
    for (int ri = 0; ri < 3; ri++) {
        Root *r = &roots[ri][167]; /* 32o */
        if (!r->found) continue;
        float s[MAX_ACTIONS]; regret_match(r->regrets, s, r->na);
        if (s[0] < 0.5) {
            printf("CHECK: %s 32o fold=%.0f%% (expected >50%%)\n", root_labels[ri], s[0]*100);
            issues++;
        }
    }
    if (issues == 0) printf("\nAll checks passed.\n");

    return 0;
}
