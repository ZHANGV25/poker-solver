/* Run 100 iterations via the DLL, save checkpoint, and scan it
 * respecting the entry count from the header. */
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <dlfcn.h>

#define HC(seed, val) ((seed) ^ ((val) * 0x9E3779B97F4A7C15ULL + 0x517CC1B727220A95ULL + ((seed) << 12) + ((seed) >> 4)))

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "/tmp/verify_regrets.bin";

    /* Parse checkpoint WITH entry count limit */
    FILE *f = fopen(path, "rb");
    if (!f) { perror("open"); return 1; }

    char magic[4]; fread(magic, 1, 4, f);
    int64_t table_size = 0, num_entries = 0, iters = 0;
    if (memcmp(magic, "BPR4", 4) == 0) {
        fread(&table_size, 8, 1, f);
        fread(&num_entries, 8, 1, f);
        fread(&iters, 8, 1, f);
    } else {
        int ts, ne;
        fread(&ts, 4, 1, f); table_size = ts;
        fread(&ne, 4, 1, f); num_entries = ne;
        if (memcmp(magic, "BPR3", 4) == 0) fread(&iters, 8, 1, f);
        else { int i; fread(&i, 4, 1, f); iters = i; }
    }
    printf("Header: format=%.4s entries=%lld iters=%lld\n",
           magic, (long long)num_entries, (long long)iters);

    uint64_t root = 0xFEDCBA9876543210ULL;
    uint64_t after_fold = HC(root, 3ULL);
    printf("Expected: root=%016llx, after_fold=%016llx\n\n",
           (unsigned long long)root, (unsigned long long)after_fold);

    /* Scan entries up to num_entries only */
    int player, street, bucket, na, has_sum;
    uint64_t board_hash, action_hash;

    int found_root = 0, found_fold = 0;
    int player_counts[6] = {0};
    int preflop_counts[6] = {0};

    for (long long e = 0; e < num_entries; e++) {
        if (fread(&player, 4, 1, f) != 1) { printf("EARLY EOF at entry %lld\n", e); break; }
        fread(&street, 4, 1, f);
        fread(&bucket, 4, 1, f);
        fread(&board_hash, 8, 1, f);
        fread(&action_hash, 8, 1, f);
        fread(&na, 4, 1, f);
        if (na < 1 || na > 20) { printf("BAD na=%d at entry %lld\n", na, e); break; }
        int reg[20]; fread(reg, 4, na, f);
        fread(&has_sum, 4, 1, f);
        if (has_sum) { float ss[20]; fread(ss, 4, na, f); }

        if (player >= 0 && player < 6) player_counts[player]++;
        if (street == 0 && player >= 0 && player < 6) preflop_counts[player]++;

        if (action_hash == root) found_root++;
        if (action_hash == after_fold) found_fold++;

        /* Print first 5 preflop entries for player 3 (MP) */
        if (street == 0 && player == 3 && preflop_counts[3] <= 5) {
            printf("  MP preflop #%d: bucket=%d na=%d hash=%016llx\n",
                   preflop_counts[3], bucket, na,
                   (unsigned long long)action_hash);
        }
    }
    fclose(f);

    printf("\nEntry counts by player:\n");
    const char *POS[] = {"SB","BB","UTG","MP","CO","BTN"};
    for (int p = 0; p < 6; p++)
        printf("  %s: %d total, %d preflop\n", POS[p], player_counts[p], preflop_counts[p]);
    printf("\nRoot hash found: %d times\n", found_root);
    printf("After-fold hash found: %d times\n", found_fold);

    return 0;
}
