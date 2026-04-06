/* Search for specific hashes with ANY player/bucket combination */
#include <stdio.h>
#include <string.h>
#include <stdint.h>

static inline uint64_t hash_combine(uint64_t seed, uint64_t val) {
    return seed ^ (val * 0x9E3779B97F4A7C15ULL + 0x517CC1B727220A95ULL
                   + (seed << 12) + (seed >> 4));
}

int main(int argc, char **argv) {
    FILE *f = fopen(argv[1], "rb");
    char magic[4]; fread(magic, 1, 4, f);
    if (memcmp(magic,"BPR4",4)==0) { int64_t x; fread(&x,8,1,f); fread(&x,8,1,f); fread(&x,8,1,f); }
    else { int x; fread(&x,4,1,f); fread(&x,4,1,f);
           if (memcmp(magic,"BPR3",4)==0) { int64_t y; fread(&y,8,1,f); }
           else { int y; fread(&y,4,1,f); } }

    uint64_t root = 0xFEDCBA9876543210ULL;
    /* Search for: root, and first 12 "after one action" hashes */
    uint64_t targets[13];
    targets[0] = root;
    for (int a = 0; a < 12; a++)
        targets[a+1] = hash_combine(root, (uint64_t)a * 17 + 3);

    int player, street, bucket, na, has_sum;
    uint64_t board_hash, action_hash;
    int found[13] = {0};
    long long total = 0;

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

        for (int t = 0; t < 13; t++) {
            if (action_hash == targets[t] && found[t] < 3) {
                printf("target[%d] (%016llx): player=%d street=%d bucket=%d na=%d\n",
                       t, (unsigned long long)targets[t], player, street, bucket, na);
                found[t]++;
            }
        }
        total++;
    }
    fclose(f);

    printf("\nResults (%lld entries scanned):\n", total);
    printf("  root hash: %d matches\n", found[0]);
    for (int a = 0; a < 12; a++)
        printf("  after action[%d]: %d matches\n", a, found[a+1]);
    return 0;
}
