/* Find all preflop info sets for MP (player 3) with bucket 0 (AA).
 * Print their action_hashes and try to match against known sequences. */
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

    /* Precompute hashes for all possible "UTG takes action X" sequences */
    uint64_t root = 0xFEDCBA9876543210ULL;
    printf("Expected hashes for 'UTG does action X, MP acts':\n");
    for (int a = 0; a < 12; a++) {
        int seq[] = {a};
        uint64_t h = root;
        h = hash_combine(h, (uint64_t)a * 17 + 3);
        printf("  action[%d]: %016llx\n", a, (unsigned long long)h);
    }
    printf("\n");

    int player, street, bucket, na, has_sum;
    uint64_t board_hash, action_hash;
    int count = 0;

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

        /* Print all MP preflop entries for bucket 0 (AA) */
        if (street == 0 && player == 3 && bucket == 0) {
            printf("MP bucket=0(AA): hash=%016llx na=%d regs=[%d,%d,%d...]\n",
                   (unsigned long long)action_hash, na, r[0], na>1?r[1]:0, na>2?r[2]:0);
            count++;
            if (count >= 20) break;
        }
    }
    fclose(f);
    printf("\nFound %d entries for MP/AA\n", count);
    return 0;
}
