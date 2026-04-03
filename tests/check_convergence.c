/*
 * Fast convergence checker for BPR2 regret checkpoints.
 * Parses the full file in a single pass with buffered I/O.
 *
 * Build: gcc -O2 -o check_convergence check_convergence.c -lm
 * Usage: ./check_convergence /opt/blueprint_unified/regrets_latest.bin
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#define MAX_ACTIONS 20

/* Preflop class ordering: 0=AA, 1=AKs, 2=AKo, ..., 12=22, 13-168=suited/offsuit */
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
static const char *ACT_NAMES[] = {"fold","call","r0.5x","r1x","r2x","r3x","r4x"};

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

/* Root action hash = hash of empty sequence */
static uint64_t root_action_hash(void) {
    return 0xFEDCBA9876543210ULL;
}

typedef struct {
    int player;
    int bucket;
    int regrets[MAX_ACTIONS];
    int na;
} PreflopRoot;

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "/opt/blueprint_unified/regrets_latest.bin";
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return 1; }

    /* Set large buffer for fast sequential reads */
    char *iobuf = malloc(16 * 1024 * 1024);
    setvbuf(f, iobuf, _IOFBF, 16 * 1024 * 1024);

    /* Header */
    char magic[4];
    int table_size, num_entries_hdr;
    int64_t iterations_run;
    fread(magic, 1, 4, f);
    int is_v3 = (memcmp(magic, "BPR3", 4) == 0);
    int is_v2 = (memcmp(magic, "BPR2", 4) == 0);
    if (!is_v3 && !is_v2) { fprintf(stderr, "Bad magic (expected BPR2/BPR3)\n"); return 1; }
    fread(&table_size, 4, 1, f);
    fread(&num_entries_hdr, 4, 1, f);
    if (is_v3) {
        fread(&iterations_run, 8, 1, f);
    } else {
        int iters32; fread(&iters32, 4, 1, f);
        iterations_run = (int64_t)iters32;
    }
    printf("Header: table=%d entries_hdr=%d iters=%lld\n\n", table_size, num_entries_hdr, (long long)iterations_run);

    /* Stats */
    long long street_counts[4] = {0,0,0,0};
    double street_regret_sum[4] = {0,0,0,0};
    long long uniform_count = 0, converged_count = 0, total = 0;
    int max_regret = 0, min_regret = 0;

    /* Preflop root collection */
    PreflopRoot preflop_roots[1024];
    int n_roots = 0;

    uint64_t root_ah = root_action_hash();

    int player, street, bucket, na, has_sum;
    uint64_t board_hash, action_hash;
    int regrets[MAX_ACTIONS];
    float strat[MAX_ACTIONS];

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

        if (street >= 0 && street <= 3) {
            street_counts[street]++;
            int mr = 0;
            for (int i = 0; i < na; i++) {
                int ar = abs(regrets[i]);
                if (ar > mr) mr = ar;
                if (regrets[i] > max_regret) max_regret = regrets[i];
                if (regrets[i] < min_regret) min_regret = regrets[i];
            }
            street_regret_sum[street] += mr;
        }

        regret_match(regrets, strat, na);
        float max_p = 0;
        for (int i = 0; i < na; i++) if (strat[i] > max_p) max_p = strat[i];
        if (fabsf(max_p - 1.0f/na) < 0.05f) uniform_count++;
        if (max_p > 0.70f) converged_count++;

        /* Collect preflop roots */
        if (street == 0 && action_hash == root_ah && n_roots < 1024) {
            preflop_roots[n_roots].player = player;
            preflop_roots[n_roots].bucket = bucket;
            preflop_roots[n_roots].na = na;
            memcpy(preflop_roots[n_roots].regrets, regrets, na * sizeof(int));
            n_roots++;
        }

        total++;
        if (total % 100000000 == 0)
            printf("  %lld entries...\n", total);
    }
    fclose(f);
    free(iobuf);

    printf("\nActual entries: %lld (header said %d)\n", total, num_entries_hdr);

    printf("\n=== Distribution by street ===\n");
    const char *snames[] = {"Preflop","Flop","Turn","River"};
    for (int i = 0; i < 4; i++) {
        if (street_counts[i] > 0) {
            double avg = street_regret_sum[i] / street_counts[i];
            double pct = (double)street_counts[i] / total * 100;
            printf("  %8s: %12lld (%5.1f%%)  avg|regret|=%.0f\n", snames[i], street_counts[i], pct, avg);
        }
    }

    printf("\n=== Convergence ===\n");
    printf("  Max regret: %d\n", max_regret);
    printf("  Min regret: %d (floor: -310,000,000)\n", min_regret);
    printf("  Near-uniform: %lld / %lld (%.1f%%)\n", uniform_count, total, (double)uniform_count/total*100);
    printf("  Dominant >70%%: %lld / %lld (%.1f%%)\n", converged_count, total, (double)converged_count/total*100);

    printf("\n=== Preflop root strategies ===\n");
    int show_buckets[] = {0,1,2,3,4,5,10,25,48,84,120,150,165,166,167,168};
    int n_show = sizeof(show_buckets)/sizeof(show_buckets[0]);

    for (int pos = 0; pos < 6; pos++) {
        printf("\n  %s:\n", POS_NAMES[pos]);
        for (int si = 0; si < n_show; si++) {
            int bkt = show_buckets[si];
            for (int r = 0; r < n_roots; r++) {
                if (preflop_roots[r].player == pos && preflop_roots[r].bucket == bkt) {
                    int rna = preflop_roots[r].na;
                    regret_match(preflop_roots[r].regrets, strat, rna);
                    const char *label = bkt < 169 ? PREFLOP_LABELS[bkt] : "???";
                    printf("    %3d %-4s: ", bkt, label);
                    for (int a = 0; a < rna && a < 7; a++)
                        printf("%s=%.2f ", ACT_NAMES[a], strat[a]);
                    printf("\n");
                    break;
                }
            }
        }
    }

    printf("\n=== Sanity checks ===\n");
    /* Find AA (bucket 0) for any position */
    for (int r = 0; r < n_roots; r++) {
        if (preflop_roots[r].bucket == 0) {
            regret_match(preflop_roots[r].regrets, strat, preflop_roots[r].na);
            int folds = strat[0] > 0.5f;
            printf("  AA (%s): %s (fold=%.2f)\n", POS_NAMES[preflop_roots[r].player],
                   folds ? "FAIL - AA folding >50%!" : "OK - raises", strat[0]);
            break;
        }
    }
    for (int r = 0; r < n_roots; r++) {
        if (preflop_roots[r].bucket == 168) {
            regret_match(preflop_roots[r].regrets, strat, preflop_roots[r].na);
            int folds = strat[0] > 0.5f;
            printf("  32o (%s): %s (fold=%.2f)\n", POS_NAMES[preflop_roots[r].player],
                   folds ? "OK - folds" : "FAIL - 32o not folding!", strat[0]);
            break;
        }
    }

    return 0;
}
