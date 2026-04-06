/* Search checkpoint for specific action_hashes */
#include <stdio.h>
#include <string.h>
#include <stdint.h>

static inline uint64_t hash_combine(uint64_t seed, uint64_t val) {
    return seed ^ (val * 0x9E3779B97F4A7C15ULL + 0x517CC1B727220A95ULL
                   + (seed << 12) + (seed >> 4));
}

int main(int argc, char **argv) {
    const char *path = argv[1];
    FILE *f = fopen(path, "rb");
    char magic[4]; fread(magic, 1, 4, f);
    if (memcmp(magic,"BPR4",4)==0) { int64_t x; fread(&x,8,1,f); fread(&x,8,1,f); fread(&x,8,1,f); }
    else { int x; fread(&x,4,1,f); fread(&x,4,1,f);
           if (memcmp(magic,"BPR3",4)==0) { int64_t y; fread(&y,8,1,f); }
           else { int y; fread(&y,4,1,f); } }

    /* Hashes to search for */
    uint64_t root = 0xFEDCBA9876543210ULL;
    uint64_t after_fold = hash_combine(root, (uint64_t)0 * 17 + 3);
    uint64_t after_2fold = hash_combine(after_fold, (uint64_t)0 * 17 + 3);

    printf("Searching for:\n");
    printf("  UTG root:  %016llx\n", (unsigned long long)root);
    printf("  After 1 fold: %016llx\n", (unsigned long long)after_fold);
    printf("  After 2 folds: %016llx\n", (unsigned long long)after_2fold);
    printf("\n");

    int player, street, bucket, na, has_sum;
    uint64_t board_hash, action_hash;
    int found_root=0, found_1f=0, found_2f=0;
    long long total=0;

    while (1) {
        if (fread(&player, 4, 1, f) != 1) break;
        if (fread(&street, 4, 1, f) != 1) break;
        if (fread(&bucket, 4, 1, f) != 1) break;
        if (fread(&board_hash, 8, 1, f) != 1) break;
        if (fread(&action_hash, 8, 1, f) != 1) break;
        if (fread(&na, 4, 1, f) != 1) break;
        if (na<1||na>20) break;
        int r[20]; fread(r, 4, na, f);
        fread(&has_sum, 4, 1, f);
        if (has_sum) { float ss[20]; fread(ss, 4, na, f); }

        if (action_hash == root && !found_root) {
            printf("FOUND root hash: player=%d street=%d bucket=%d na=%d\n", player, street, bucket, na);
            found_root++;
        }
        if (action_hash == after_fold && found_1f < 3) {
            printf("FOUND after-1-fold: player=%d street=%d bucket=%d na=%d\n", player, street, bucket, na);
            found_1f++;
        }
        if (action_hash == after_2fold && found_2f < 3) {
            printf("FOUND after-2-folds: player=%d street=%d bucket=%d na=%d\n", player, street, bucket, na);
            found_2f++;
        }
        total++;
    }
    fclose(f);
    printf("\nScanned %lld entries. Found: root=%d, 1fold=%d, 2fold=%d\n",
           total, found_root, found_1f, found_2f);
    return 0;
}
