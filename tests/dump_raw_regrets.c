/* Dump RAW regret values (not regret-matched) for specific info sets.
 * Shows actual int32 regret accumulations to diagnose convergence issues. */
#include <stdio.h>
#include <string.h>
#include <stdint.h>

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

int main(int argc, char **argv) {
    const char *path = argv[1];
    FILE *f = fopen(path, "rb");
    if (!f) { perror("open"); return 1; }

    char magic[4]; fread(magic, 1, 4, f);
    int64_t table_size, num_entries, iters;
    if (memcmp(magic,"BPR4",4)==0) {
        fread(&table_size,8,1,f); fread(&num_entries,8,1,f); fread(&iters,8,1,f);
    } else {
        int ts,ne; fread(&ts,4,1,f); fread(&ne,4,1,f);
        table_size=ts; num_entries=ne;
        if (memcmp(magic,"BPR3",4)==0) fread(&iters,8,1,f);
        else { int i; fread(&i,4,1,f); iters=i; }
    }
    printf("Checkpoint: %lld entries, %lld iters\n\n",
           (long long)num_entries, (long long)iters);

    /* Target hashes */
    int acts0[] = {};
    uint64_t utg_root = compute_action_hash(acts0, 0);

    /* Buckets to look for: AA=0, KK=25, 32o=167, 87s=131?
     * Actually the bucket indices match LABELS order:
     * AA=0, AKs=1, AKo=2, ..., KK=25, ..., 32o=167, 22=168 */

    /* Bucket indices: AA=0, AKs=1, AKo=2, ..., KK=25, ..., QQ=48, JJ=69,
     * TT=88, 99=105, 88=120, 77=133, 66=144, 55=153, 44=160, 33=165, 22=168 */
    int targets[][2] = {
        {2, 0},    /* UTG, AA */
        {2, 25},   /* UTG, KK */
        {2, 2},    /* UTG, AKo */
        {2, 88},   /* UTG, TT */
        {2, 105},  /* UTG, 99 */
        {2, 120},  /* UTG, 88 */
        {2, 133},  /* UTG, 77 */
        {2, 144},  /* UTG, 66 */
        {2, 153},  /* UTG, 55 */
        {2, 160},  /* UTG, 44 */
        {2, 165},  /* UTG, 33 */
        {2, 168},  /* UTG, 22 */
        {5, 88},   /* BTN, TT (for comparison) */
        {5, 105},  /* BTN, 99 (for comparison) */
    };
    int n_targets = 14;
    const char *target_labels[] = {
        "UTG AA", "UTG KK", "UTG AKo",
        "UTG TT", "UTG 99", "UTG 88", "UTG 77", "UTG 66",
        "UTG 55", "UTG 44", "UTG 33", "UTG 22",
        "BTN TT", "BTN 99"
    };

    /* BTN root hash: UTG folds, MP folds, CO folds → BTN acts */
    int acts_btn[] = {0, 0, 0};
    uint64_t btn_root = compute_action_hash(acts_btn, 3);

    int found[14] = {0};
    int player, street, bucket, na, has_sum;
    uint64_t board_hash, action_hash;

    for (long long e = 0; e < num_entries; e++) {
        if (fread(&player, 4, 1, f) != 1) break;
        fread(&street, 4, 1, f);
        fread(&bucket, 4, 1, f);
        fread(&board_hash, 8, 1, f);
        fread(&action_hash, 8, 1, f);
        fread(&na, 4, 1, f);
        if (na < 1 || na > 20) break;
        int regrets[20];
        fread(regrets, 4, na, f);
        fread(&has_sum, 4, 1, f);
        float strategy_sum[20];
        if (has_sum) fread(strategy_sum, 4, na, f);

        for (int t = 0; t < n_targets; t++) {
            if (found[t]) continue;
            int tp = targets[t][0];
            int tb = targets[t][1];
            uint64_t expected_hash = (tp == 2) ? utg_root : btn_root;

            if (player == tp && street == 0 && bucket == tb &&
                action_hash == expected_hash) {
                printf("=== %s (player=%d bucket=%d na=%d) ===\n",
                       target_labels[t], player, bucket, na);
                printf("  Raw regrets: [");
                for (int a = 0; a < na; a++)
                    printf("%s%d", a?", ":"", regrets[a]);
                printf("]\n");

                if (has_sum) {
                    printf("  Strategy sum: [");
                    for (int a = 0; a < na; a++)
                        printf("%s%.1f", a?", ":"", strategy_sum[a]);
                    printf("]\n");
                } else {
                    printf("  Strategy sum: NULL\n");
                }

                /* Regret match for comparison */
                float strat[20];
                float sum = 0;
                for (int a = 0; a < na; a++) {
                    strat[a] = regrets[a] > 0 ? (float)regrets[a] : 0;
                    sum += strat[a];
                }
                if (sum > 0) for (int a = 0; a < na; a++) strat[a] /= sum;
                else for (int a = 0; a < na; a++) strat[a] = 1.0f / na;

                printf("  Regret-matched: [");
                for (int a = 0; a < na; a++)
                    printf("%s%.1f%%", a?", ":"", strat[a]*100);
                printf("]\n");

                /* Check for regret floor/ceiling hits */
                int at_floor = 0, at_ceiling = 0;
                for (int a = 0; a < na; a++) {
                    if (regrets[a] <= -310000000) at_floor++;
                    if (regrets[a] >= 310000000) at_ceiling++;
                }
                if (at_floor || at_ceiling)
                    printf("  WARNING: %d at floor, %d at ceiling\n",
                           at_floor, at_ceiling);
                printf("\n");
                found[t] = 1;
            }
        }

        if (e % 200000000 == 0 && e > 0)
            fprintf(stderr, "  %lldM entries...\n", e/1000000);
        /* Early exit once all targets found */
        int all = 1;
        for (int t = 0; t < n_targets; t++) if (!found[t]) { all = 0; break; }
        if (all) break;
    }
    fclose(f);

    /* Report missing */
    for (int t = 0; t < n_targets; t++)
        if (!found[t]) printf("NOT FOUND: %s\n", target_labels[t]);

    return 0;
}
