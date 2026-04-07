/* Dump first 20 action_hashes for each player at street=0 (preflop) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static inline uint64_t hash_combine(uint64_t seed, uint64_t val) {
    return seed ^ (val * 0x9E3779B97F4A7C15ULL + 0x517CC1B727220A95ULL
                   + (seed << 12) + (seed >> 4));
}

static uint64_t compute_action_hash(const int *actions, int n) {
    uint64_t h = 0xFEDCBA9876543210ULL;
    for (int i = 0; i < n; i++)
        h = hash_combine(h, (uint64_t)actions[i] * 17 + 3);
    return h;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "/tmp/verify_regrets.bin";
    FILE *f = fopen(path, "rb");
    if (!f) { perror("open"); return 1; }

    char magic[4]; fread(magic, 1, 4, f);
    int is_v4 = (memcmp(magic, "BPR4", 4) == 0);
    if (is_v4) { int64_t x; fread(&x,8,1,f); fread(&x,8,1,f); fread(&x,8,1,f); }
    else { int x; fread(&x,4,1,f); fread(&x,4,1,f);
           if (memcmp(magic,"BPR3",4)==0) { int64_t y; fread(&y,8,1,f); }
           else { fread(&x,4,1,f); } }

    /* Print expected hashes */
    printf("Expected root hashes:\n");
    int a0[] = {}; printf("  UTG root: %016llx\n", (unsigned long long)compute_action_hash(a0, 0));
    int a1[] = {0}; printf("  MP  root (UTG folds=action 0): %016llx\n", (unsigned long long)compute_action_hash(a1, 1));
    int a2[] = {0,0}; printf("  CO  root: %016llx\n", (unsigned long long)compute_action_hash(a2, 2));
    printf("\n");

    /* Collect first few unique hashes per player at street 0 */
    int player, street, bucket, na, has_sum;
    uint64_t board_hash, action_hash;
    int count[6] = {0};
    uint64_t first_hashes[6][30];
    memset(first_hashes, 0, sizeof(first_hashes));

    while (1) {
        if (fread(&player, 4, 1, f) != 1) break;
        if (fread(&street, 4, 1, f) != 1) break;
        if (fread(&bucket, 4, 1, f) != 1) break;
        if (fread(&board_hash, 8, 1, f) != 1) break;
        if (fread(&action_hash, 8, 1, f) != 1) break;
        if (fread(&na, 4, 1, f) != 1) break;
        if (na < 1 || na > 20) break;
        int reg[20]; fread(reg, 4, na, f);
        fread(&has_sum, 4, 1, f);
        if (has_sum) { float ss[20]; fread(ss, 4, na, f); }

        if (street == 0 && player >= 0 && player < 6) {
            /* Check if this hash is already recorded */
            int dup = 0;
            for (int i = 0; i < count[player] && i < 30; i++)
                if (first_hashes[player][i] == action_hash) { dup = 1; break; }
            if (!dup && count[player] < 30) {
                first_hashes[player][count[player]] = action_hash;
                count[player]++;
            }
        }
    }
    fclose(f);

    const char *POS[] = {"SB","BB","UTG","MP","CO","BTN"};
    for (int p = 0; p < 6; p++) {
        printf("%s (player %d): %d unique action_hashes (first 30):\n", POS[p], p, count[p]);
        for (int i = 0; i < count[p]; i++)
            printf("  %016llx\n", (unsigned long long)first_hashes[p][i]);
        printf("\n");
    }

    return 0;
}
