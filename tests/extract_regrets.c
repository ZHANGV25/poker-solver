/*
 * Extract raw regret values for specific preflop classes at the UTG root.
 * Prints both regrets and strategies so we can see exactly what the solver
 * "thinks" and whether fold_regret vs raise_regret is trending correctly.
 *
 * Build: gcc -O2 -o extract_regrets tests/extract_regrets.c -lm
 * Usage: ./extract_regrets /opt/blueprint_unified/regrets_latest.bin
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#define MAX_ACTIONS 20

static const char *PREFLOP_LABELS[] = {
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

static const char *POS_NAMES[] = {"SB","BB","UTG","MP","CO","BTN"};
static const char *ACT_NAMES[] = {"fold","call","r0.5x","r1x","r2x","r3x","allin","a7"};

static void regret_match(const int *regrets, float *strat, int na) {
    float total = 0;
    for (int i = 0; i < na; i++) {
        float v = regrets[i] > 0 ? (float)regrets[i] : 0;
        strat[i] = v;
        total += v;
    }
    if (total > 0) for (int i = 0; i < na; i++) strat[i] /= total;
    else for (int i = 0; i < na; i++) strat[i] = 1.0f / na;
}

typedef struct {
    int player;
    int bucket;
    int regrets[MAX_ACTIONS];
    int na;
    uint64_t action_hash;
    float strategy_sum[MAX_ACTIONS];
    int has_sum;
} Entry;

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "/opt/blueprint_unified/regrets_latest.bin";
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return 1; }

    char *iobuf = malloc(16 * 1024 * 1024);
    setvbuf(f, iobuf, _IOFBF, 16 * 1024 * 1024);

    char magic[4];
    int table_size, num_entries_hdr;
    int64_t iterations_run;
    fread(magic, 1, 4, f);
    int is_v3 = (memcmp(magic, "BPR3", 4) == 0);
    if (!is_v3 && memcmp(magic, "BPR2", 4) != 0) {
        fprintf(stderr, "Bad magic\n"); return 1;
    }
    fread(&table_size, 4, 1, f);
    fread(&num_entries_hdr, 4, 1, f);
    if (is_v3) fread(&iterations_run, 8, 1, f);
    else { int i32; fread(&i32, 4, 1, f); iterations_run = i32; }

    printf("Checkpoint: %s format, table=%d, entries=%d, iters=%lld\n\n",
           is_v3 ? "BPR3" : "BPR2", table_size, num_entries_hdr, (long long)iterations_run);

    uint64_t root_ah = 0xFEDCBA9876543210ULL;

    /* Buckets we want to examine */
    int target_buckets[] = {0, 1, 2, 25, 48, 84, 120, 150, 155, 165, 166, 167, 168};
    int n_targets = sizeof(target_buckets) / sizeof(target_buckets[0]);

    /* Collect all preflop root entries for all positions */
    Entry roots[6 * 169]; /* max 6 positions * 169 buckets */
    int n_roots = 0;

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
        float ss[MAX_ACTIONS] = {0};
        if (has_sum) {
            if (fread(ss, 4, na, f) != (size_t)na) break;
        }

        /* Collect preflop roots (street=0, root action hash) */
        if (street == 0 && action_hash == root_ah && n_roots < 6 * 169) {
            Entry *e = &roots[n_roots];
            e->player = player;
            e->bucket = bucket;
            e->na = na;
            e->action_hash = action_hash;
            e->has_sum = has_sum;
            memcpy(e->regrets, regrets, na * sizeof(int));
            memcpy(e->strategy_sum, ss, na * sizeof(float));
            n_roots++;
        }

        /* Also collect NON-root preflop entries for 32s/32o to check for hash issues */

        total++;
        if (total % 200000000 == 0)
            fprintf(stderr, "  %lld entries...\n", total);
    }
    fclose(f);
    free(iobuf);

    printf("Scanned %lld entries, collected %d preflop roots\n\n", total, n_roots);

    /* Print raw regrets for target buckets at each position */
    printf("=== RAW REGRET VALUES (preflop root, all positions) ===\n");
    printf("Actions: 0=fold 1=call 2=r0.5x 3=r1x 4=r2x 5=r3x 6=allin\n\n");

    for (int pos = 0; pos < 6; pos++) {
        printf("--- %s (player %d) ---\n", POS_NAMES[pos], pos);
        for (int ti = 0; ti < n_targets; ti++) {
            int bkt = target_buckets[ti];
            for (int r = 0; r < n_roots; r++) {
                if (roots[r].player == pos && roots[r].bucket == bkt) {
                    Entry *e = &roots[r];
                    float strat[MAX_ACTIONS];
                    regret_match(e->regrets, strat, e->na);

                    const char *label = bkt < 169 ? PREFLOP_LABELS[bkt] : "???";
                    printf("  %3d %-4s [na=%d]: ", bkt, label, e->na);
                    printf("regrets={");
                    for (int a = 0; a < e->na; a++)
                        printf("%s%d", a ? "," : "", e->regrets[a]);
                    printf("} → strat={");
                    for (int a = 0; a < e->na; a++)
                        printf("%s%.3f", a ? "," : "", strat[a]);
                    printf("}");
                    if (e->has_sum) {
                        printf(" sum={");
                        for (int a = 0; a < e->na; a++)
                            printf("%s%.0f", a ? "," : "", e->strategy_sum[a]);
                        printf("}");
                    }
                    printf("\n");
                    break;
                }
            }
        }
        printf("\n");
    }

    /* Special analysis: 32s vs 32o divergence */
    printf("=== 32s vs 32o DIVERGENCE ANALYSIS ===\n");
    for (int pos = 0; pos < 6; pos++) {
        Entry *e32s = NULL, *e32o = NULL;
        for (int r = 0; r < n_roots; r++) {
            if (roots[r].player == pos && roots[r].bucket == 166) e32s = &roots[r];
            if (roots[r].player == pos && roots[r].bucket == 167) e32o = &roots[r];
        }
        if (e32s && e32o) {
            printf("  %s: 32s na=%d, 32o na=%d", POS_NAMES[pos], e32s->na, e32o->na);
            if (e32s->na != e32o->na) printf(" *** MISMATCH ***");
            printf("\n");
            printf("    32s regrets: ");
            for (int a = 0; a < e32s->na; a++) printf("%8d ", e32s->regrets[a]);
            printf("\n");
            printf("    32o regrets: ");
            for (int a = 0; a < e32o->na; a++) printf("%8d ", e32o->regrets[a]);
            printf("\n");
            /* Check if they have the same sign pattern */
            int same_signs = 1;
            int min_na = e32s->na < e32o->na ? e32s->na : e32o->na;
            for (int a = 0; a < min_na; a++) {
                int s1 = (e32s->regrets[a] > 0) - (e32s->regrets[a] < 0);
                int s2 = (e32o->regrets[a] > 0) - (e32o->regrets[a] < 0);
                if (s1 != s2) { same_signs = 0; break; }
            }
            printf("    Same sign pattern: %s\n\n", same_signs ? "YES" : "NO *** DIVERGED ***");
        }
    }

    /* Fold regret trend: for trash hands, is fold regret positive and growing? */
    printf("=== FOLD REGRET ANALYSIS (trash hands at UTG) ===\n");
    int trash_buckets[] = {150, 155, 165, 166, 167, 168};
    int n_trash = sizeof(trash_buckets) / sizeof(trash_buckets[0]);
    for (int ti = 0; ti < n_trash; ti++) {
        int bkt = trash_buckets[ti];
        for (int r = 0; r < n_roots; r++) {
            if (roots[r].player == 2 && roots[r].bucket == bkt) { /* UTG = player 2 */
                Entry *e = &roots[r];
                const char *label = bkt < 169 ? PREFLOP_LABELS[bkt] : "???";
                int fold_reg = e->regrets[0];
                int max_raise_reg = 0;
                for (int a = 2; a < e->na; a++)
                    if (e->regrets[a] > max_raise_reg) max_raise_reg = e->regrets[a];
                printf("  %-4s (bkt %3d): fold_reg=%d, max_raise_reg=%d, fold_leads=%s\n",
                       label, bkt, fold_reg, max_raise_reg,
                       fold_reg > max_raise_reg ? "YES" : "NO");
                break;
            }
        }
    }

    return 0;
}
