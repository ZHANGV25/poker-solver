/*
 * diagnose_checkpoint_v2.c — Full duplicate scan across ALL preflop entries.
 *
 * Uses a hash set to detect duplicate (player, bucket, action_hash) tuples
 * across all 34M+ preflop info sets, not just root.
 *
 * Build:  cc -O2 -o diagnose_v2 diagnose_checkpoint_v2.c
 * Usage:  aws s3 cp s3://bucket/checkpoint.bin - | ./diagnose_v2
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#define MAX_ACTIONS 16

/* ── Hash set for duplicate detection ── */
/* Key: (player, bucket, action_hash) packed into a struct.
 * Table: open addressing, linear probing, 64M slots (~1.5GB). */

#define DUP_TABLE_SIZE (1 << 26)  /* 64M slots */
#define DUP_TABLE_MASK (DUP_TABLE_SIZE - 1)

typedef struct {
    int player;
    int bucket;
    uint64_t action_hash;
} DupKey;

typedef struct {
    DupKey key;
    int count;
    int occupied;
    /* Regrets from first copy (to compare with second) */
    int first_all_zero;
    int first_max_abs;
} DupSlot;

static DupSlot *dup_table;

static uint64_t dup_hash(int player, int bucket, uint64_t ah) {
    uint64_t h = ah;
    h ^= (uint64_t)player * 0x9e3779b97f4a7c15ULL;
    h ^= (uint64_t)bucket * 0x517cc1b727220a95ULL;
    h ^= (h >> 33);
    h *= 0xff51afd7ed558ccdULL;
    h ^= (h >> 33);
    return h;
}

/* Returns pointer to slot. Sets slot->count = 1 on first insert. */
static DupSlot *dup_find_or_create(int player, int bucket, uint64_t ah) {
    uint64_t h = dup_hash(player, bucket, ah);
    for (int probe = 0; probe < 4096; probe++) {
        uint64_t idx = (h + probe) & DUP_TABLE_MASK;
        DupSlot *s = &dup_table[idx];
        if (!s->occupied) {
            s->occupied = 1;
            s->key.player = player;
            s->key.bucket = bucket;
            s->key.action_hash = ah;
            s->count = 0;
            s->first_all_zero = 0;
            s->first_max_abs = 0;
            return s;
        }
        if (s->key.player == player && s->key.bucket == bucket &&
            s->key.action_hash == ah) {
            return s;
        }
    }
    return NULL; /* table full — shouldn't happen with 64M slots for 34M entries */
}

/* ── Buffered stdin reader ── */
#define BUFSIZE (64 * 1024 * 1024)
static unsigned char *g_buf;
static size_t g_pos, g_len;

static void buf_init(void) {
    g_buf = malloc(BUFSIZE);
    g_pos = g_len = 0;
}

static int buf_read(void *dst, size_t n) {
    unsigned char *out = (unsigned char *)dst;
    while (n > 0) {
        if (g_pos >= g_len) {
            g_len = fread(g_buf, 1, BUFSIZE, stdin);
            g_pos = 0;
            if (g_len == 0) return -1;
        }
        size_t avail = g_len - g_pos;
        size_t take = n < avail ? n : avail;
        memcpy(out, g_buf + g_pos, take);
        g_pos += take;
        out += take;
        n -= take;
    }
    return 0;
}

/* ── Bucket names ── */
static const char *RANKS = "23456789TJQKA";
static char bucket_names[169][8];
static const char *player_names[] = {"SB","BB","UTG","MP","CO","BTN"};

static void init_bucket_names(void) {
    int n = 0;
    for (int r0 = 12; r0 >= 0; r0--) {
        for (int r1 = r0; r1 >= 0; r1--) {
            if (r0 == r1) {
                sprintf(bucket_names[n++], "%c%c", RANKS[r0], RANKS[r1]);
            } else {
                sprintf(bucket_names[n++], "%c%cs", RANKS[r0], RANKS[r1]);
                sprintf(bucket_names[n++], "%c%co", RANKS[r0], RANKS[r1]);
            }
        }
    }
}

int main(void) {
    init_bucket_names();
    buf_init();

    /* Allocate dup table: 64M slots × 40 bytes ≈ 2.5GB */
    dup_table = (DupSlot*)calloc(DUP_TABLE_SIZE, sizeof(DupSlot));
    if (!dup_table) {
        fprintf(stderr, "ERROR: can't allocate dup table (need ~2.5GB)\n");
        return 1;
    }
    printf("Allocated dup table: %d slots (%.1f GB)\n",
           DUP_TABLE_SIZE, (double)DUP_TABLE_SIZE * sizeof(DupSlot) / 1e9);
    fflush(stdout);

    /* Header */
    char magic[4];
    if (buf_read(magic, 4) < 0 || memcmp(magic, "BPR4", 4) != 0) {
        fprintf(stderr, "ERROR: not BPR4\n");
        return 1;
    }
    int64_t table_size, num_entries, iterations;
    buf_read(&table_size, 8);
    buf_read(&num_entries, 8);
    buf_read(&iterations, 8);
    printf("Checkpoint: %lld entries, %lld iters\n",
           (long long)num_entries, (long long)iterations);
    fflush(stdout);

    int64_t total_preflop = 0;
    int64_t total_postflop = 0;
    int64_t total_preflop_dups = 0;
    int64_t total_preflop_zero = 0;
    int64_t total_preflop_tiny = 0;
    int64_t total_postflop_dups = 0;
    int64_t total_postflop_zero = 0;

    time_t t0 = time(NULL);

    for (int64_t i = 0; i < num_entries; i++) {
        if (i > 0 && (i % 50000000) == 0) {
            time_t now = time(NULL);
            double elapsed = difftime(now, t0);
            double rate = (double)i / elapsed;
            double eta = (double)(num_entries - i) / rate;
            printf("  %lldM/%lldM (%lld%%) %.1fM/s ETA %.0fs | "
                   "preflop: %lldM (%lld dups, %lld zero) postflop: %lldM (%lld dups, %lld zero)\n",
                   (long long)(i/1000000), (long long)(num_entries/1000000),
                   (long long)(100*i/num_entries), rate/1e6, eta,
                   (long long)(total_preflop/1000000), (long long)total_preflop_dups,
                   (long long)(total_preflop_zero/1000000),
                   (long long)(total_postflop/1000000), (long long)total_postflop_dups,
                   (long long)(total_postflop_zero/1000000));
            fflush(stdout);
        }

        int player, street, bucket, na;
        uint64_t board_hash, action_hash;
        if (buf_read(&player, 4) < 0) break;
        buf_read(&street, 4);
        buf_read(&bucket, 4);
        buf_read(&board_hash, 8);
        buf_read(&action_hash, 8);
        buf_read(&na, 4);
        if (na < 1 || na > MAX_ACTIONS) na = MAX_ACTIONS;

        int regrets[MAX_ACTIONS];
        buf_read(regrets, 4 * na);

        int has_ss;
        buf_read(&has_ss, 4);
        if (has_ss) {
            float ss[MAX_ACTIONS];
            buf_read(ss, 4 * na);
        }

        /* Check regret stats */
        int all_zero = 1;
        int max_abs = 0;
        for (int a = 0; a < na; a++) {
            if (regrets[a] != 0) all_zero = 0;
            int ab = regrets[a] < 0 ? -regrets[a] : regrets[a];
            if (ab > max_abs) max_abs = ab;
        }

        if (street == 0) {
            total_preflop++;
            if (all_zero) total_preflop_zero++;
            else if (max_abs < 10000) total_preflop_tiny++;

            DupSlot *slot = dup_find_or_create(player, bucket, action_hash);
            if (slot) {
                slot->count++;
                if (slot->count == 1) {
                    slot->first_all_zero = all_zero;
                    slot->first_max_abs = max_abs;
                }
                if (slot->count == 2) {
                    total_preflop_dups++;
                }
            }
        } else {
            total_postflop++;
            if (all_zero) total_postflop_zero++;
        }
    }

    time_t now = time(NULL);
    double elapsed = difftime(now, t0);
    printf("\nDone: %.0fs (%.1fM/s)\n", elapsed, (double)num_entries/elapsed/1e6);

    /* ── Report ── */
    printf("\n======================================================================\n");
    printf("PREFLOP:  %lld entries, %lld all-zero (%.0f%%), %lld tiny\n",
           (long long)total_preflop, (long long)total_preflop_zero,
           100.0 * total_preflop_zero / (total_preflop ? total_preflop : 1),
           (long long)total_preflop_tiny);
    printf("POSTFLOP: %lld entries, %lld all-zero (%.0f%%)\n",
           (long long)total_postflop, (long long)total_postflop_zero,
           100.0 * total_postflop_zero / (total_postflop ? total_postflop : 1));

    printf("\n--- PREFLOP DUPLICATES: %lld ---\n", (long long)total_preflop_dups);

    /* Collect stats on duplicates by player */
    int dups_by_player[6] = {0};
    int dups_by_player_zero[6] = {0};  /* first copy was all-zero */
    int total_dup_entries = 0;

    /* Print first 50 examples */
    int shown = 0;
    for (uint64_t i = 0; i < DUP_TABLE_SIZE; i++) {
        DupSlot *s = &dup_table[i];
        if (!s->occupied || s->count <= 1) continue;
        total_dup_entries++;
        if (s->key.player >= 0 && s->key.player < 6) {
            dups_by_player[s->key.player] += s->count - 1; /* extra copies */
            if (s->first_all_zero) dups_by_player_zero[s->key.player]++;
        }
        if (shown < 50) {
            const char *pname = (s->key.player >= 0 && s->key.player < 6) ?
                                 player_names[s->key.player] : "?";
            const char *bname = (s->key.bucket >= 0 && s->key.bucket < 169) ?
                                 bucket_names[s->key.bucket] : "?";
            printf("  %s %-4s ah=%016llx -> %d copies (1st: %s, max|r|=%d)\n",
                   pname, bname, (unsigned long long)s->key.action_hash,
                   s->count,
                   s->first_all_zero ? "ZERO" : "has data",
                   s->first_max_abs);
            shown++;
        }
    }

    printf("\nDuplicate info sets by position:\n");
    for (int p = 0; p < 6; p++) {
        printf("  %s: %d duplicate info sets",
               player_names[p], dups_by_player[p]);
        if (dups_by_player_zero[p])
            printf(" (%d with zero first-copy)", dups_by_player_zero[p]);
        printf("\n");
    }

    printf("\nTotal unique info sets with duplicates: %d\n", total_dup_entries);
    printf("Total extra (wasted) entries: %lld\n", (long long)total_preflop_dups);

    free(dup_table);
    free(g_buf);
    printf("\nDone.\n");
    return 0;
}
