/*
 * diagnose_checkpoint.c — Fast streaming BPR4 checkpoint scanner.
 *
 * Reads from stdin, reports:
 *   - All-zero regret info sets (preflop)
 *   - Tiny regret info sets (preflop, max|r| < 10000)
 *   - Preflop root completeness (all 169 buckets * 6 players)
 *   - Duplicate preflop keys
 *
 * Build:  cc -O2 -o diagnose_checkpoint diagnose_checkpoint.c
 * Usage:  aws s3 cp s3://bucket/checkpoint.bin - | ./diagnose_checkpoint
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#define ROOT_AH 0xFEDCBA9876543210ULL
#define TINY_THRESH 10000
#define MAX_ACTIONS 16
#define MAX_ANOMALIES 500

static const char *RANKS = "23456789TJQKA";
static char bucket_names[169][8];
static const char *player_names[] = {"SB","BB","UTG","MP","CO","BTN"};

static void init_bucket_names(void) {
    int n = 0;
    for (int r0 = 12; r0 >= 0; r0--) {
        for (int r1 = r0; r1 >= 0; r1--) {
            if (r0 == r1) {
                sprintf(bucket_names[n], "%c%c", RANKS[r0], RANKS[r1]);
                n++;
            } else {
                sprintf(bucket_names[n], "%c%cs", RANKS[r0], RANKS[r1]);
                n++;
                sprintf(bucket_names[n], "%c%co", RANKS[r0], RANKS[r1]);
                n++;
            }
        }
    }
}

/* Buffered stdin reader */
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

static int buf_skip(size_t n) {
    while (n > 0) {
        if (g_pos >= g_len) {
            g_len = fread(g_buf, 1, BUFSIZE, stdin);
            g_pos = 0;
            if (g_len == 0) return -1;
        }
        size_t avail = g_len - g_pos;
        size_t take = n < avail ? n : avail;
        g_pos += take;
        n -= take;
    }
    return 0;
}

/* Preflop root storage: regrets for each (player, bucket) */
typedef struct {
    int regrets[MAX_ACTIONS];
    float strategy_sum[MAX_ACTIONS];
    int na;
    int has_ss;
    int present;
} RootEntry;

static RootEntry root[6][169];

/* Anomaly lists */
typedef struct {
    int player, bucket, na;
    uint64_t action_hash;
    int regrets[MAX_ACTIONS];
    float ss[MAX_ACTIONS];
    int has_ss;
    int max_abs;
} Anomaly;

static Anomaly zeros[MAX_ANOMALIES];
static int n_zeros = 0;
static Anomaly tinys[MAX_ANOMALIES];
static int n_tinys = 0;

/* Simple dup detection for preflop root: just track seen */
static int root_seen[6][169]; /* 0 = not seen, 1 = seen, 2+ = dup */

int main(void) {
    init_bucket_names();
    buf_init();

    /* Header */
    char magic[4];
    if (buf_read(magic, 4) < 0 || memcmp(magic, "BPR4", 4) != 0) {
        fprintf(stderr, "ERROR: not a BPR4 file\n");
        return 1;
    }
    int64_t table_size, num_entries, iterations;
    buf_read(&table_size, 8);
    buf_read(&num_entries, 8);
    buf_read(&iterations, 8);
    printf("Checkpoint: %lld entries, %lld iters, table=%lld\n",
           (long long)num_entries, (long long)iterations, (long long)table_size);
    fflush(stdout);

    int64_t total_all_zero_preflop = 0;
    int64_t total_tiny_preflop = 0;
    int64_t total_preflop = 0;
    int64_t total_with_ss = 0;
    int64_t total_dup_root = 0;

    time_t t0 = time(NULL);

    for (int64_t i = 0; i < num_entries; i++) {
        if (i > 0 && (i % 50000000) == 0) {
            time_t now = time(NULL);
            double elapsed = difftime(now, t0);
            double rate = (double)i / elapsed;
            double eta = (double)(num_entries - i) / rate;
            printf("  %lldM/%lldM (%lld%%) %.1fM/s ETA %.0fs\n",
                   (long long)(i/1000000), (long long)(num_entries/1000000),
                   (long long)(100*i/num_entries), rate/1e6, eta);
            fflush(stdout);
        }

        /* Read key: player(4) + street(4) + bucket(4) + board_hash(8) + action_hash(8) + na(4) = 32 */
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

        float ss[MAX_ACTIONS];
        memset(ss, 0, sizeof(ss));
        if (has_ss) {
            total_with_ss++;
            buf_read(ss, 4 * na);
        }

        /* Only analyze preflop */
        if (street != 0) continue;
        total_preflop++;

        /* Check all-zero */
        int all_zero = 1;
        int max_abs = 0;
        for (int a = 0; a < na; a++) {
            if (regrets[a] != 0) all_zero = 0;
            int ab = regrets[a] < 0 ? -regrets[a] : regrets[a];
            if (ab > max_abs) max_abs = ab;
        }

        if (all_zero) {
            total_all_zero_preflop++;
            if (n_zeros < MAX_ANOMALIES) {
                Anomaly *z = &zeros[n_zeros++];
                z->player = player; z->bucket = bucket;
                z->action_hash = action_hash; z->na = na;
                z->has_ss = has_ss; z->max_abs = 0;
                memcpy(z->regrets, regrets, 4 * na);
                memcpy(z->ss, ss, 4 * na);
            }
        } else if (max_abs < TINY_THRESH) {
            total_tiny_preflop++;
            if (n_tinys < MAX_ANOMALIES) {
                Anomaly *t = &tinys[n_tinys++];
                t->player = player; t->bucket = bucket;
                t->action_hash = action_hash; t->na = na;
                t->has_ss = has_ss; t->max_abs = max_abs;
                memcpy(t->regrets, regrets, 4 * na);
                memcpy(t->ss, ss, 4 * na);
            }
        }

        /* Track root entries */
        if (action_hash == ROOT_AH && player >= 0 && player < 6 &&
            bucket >= 0 && bucket < 169) {
            if (root_seen[player][bucket]) {
                total_dup_root++;
            }
            root_seen[player][bucket]++;
            RootEntry *r = &root[player][bucket];
            r->present = 1;
            r->na = na;
            r->has_ss = has_ss;
            memcpy(r->regrets, regrets, 4 * na);
            memcpy(r->strategy_sum, ss, 4 * na);
        }
    }

    time_t now = time(NULL);
    double elapsed = difftime(now, t0);
    printf("\nDone: %.0fs (%.1fM entries/s)\n", elapsed,
           (double)num_entries / elapsed / 1e6);

    /* ── Report ── */
    printf("\n======================================================================\n");
    printf("Total preflop entries: %lld\n", (long long)total_preflop);
    printf("Entries with strategy_sum: %lld\n", (long long)total_with_ss);

    /* Duplicates */
    printf("\n--- DUPLICATE ROOT KEYS: %lld ---\n", (long long)total_dup_root);
    for (int p = 0; p < 6; p++)
        for (int b = 0; b < 169; b++)
            if (root_seen[p][b] > 1)
                printf("  %s %s -> %d copies\n",
                       player_names[p], bucket_names[b], root_seen[p][b]);

    /* All-zero preflop */
    printf("\n--- ALL-ZERO PREFLOP: %lld ---\n", (long long)total_all_zero_preflop);
    printf("  Root all-zero:\n");
    for (int p = 0; p < 6; p++)
        for (int b = 0; b < 169; b++)
            if (root[p][b].present) {
                int az = 1;
                for (int a = 0; a < root[p][b].na; a++)
                    if (root[p][b].regrets[a] != 0) { az = 0; break; }
                if (az) {
                    printf("    %s %-4s na=%d", player_names[p], bucket_names[b],
                           root[p][b].na);
                    if (root[p][b].has_ss) {
                        printf(" ss=[");
                        for (int a = 0; a < root[p][b].na; a++)
                            printf("%s%.1f", a ? "," : "", root[p][b].strategy_sum[a]);
                        printf("]");
                    }
                    printf("\n");
                }
            }

    /* Tiny preflop */
    printf("\n--- TINY PREFLOP (0 < max|r| < %d): %lld ---\n",
           TINY_THRESH, (long long)total_tiny_preflop);
    printf("  Root tiny:\n");
    for (int p = 0; p < 6; p++)
        for (int b = 0; b < 169; b++)
            if (root[p][b].present) {
                int mx = 0;
                for (int a = 0; a < root[p][b].na; a++) {
                    int ab = root[p][b].regrets[a] < 0 ? -root[p][b].regrets[a] : root[p][b].regrets[a];
                    if (ab > mx) mx = ab;
                }
                if (mx > 0 && mx < TINY_THRESH) {
                    printf("    %s %-4s max|r|=%d [", player_names[p],
                           bucket_names[b], mx);
                    for (int a = 0; a < root[p][b].na; a++)
                        printf("%s%+d", a ? "," : "", root[p][b].regrets[a]);
                    printf("]");
                    if (root[p][b].has_ss) {
                        printf(" ss=[");
                        for (int a = 0; a < root[p][b].na; a++)
                            printf("%s%.1f", a ? "," : "", root[p][b].strategy_sum[a]);
                        printf("]");
                    }
                    printf("\n");
                }
            }

    /* Root completeness */
    int present = 0, missing = 0;
    for (int p = 0; p < 6; p++)
        for (int b = 0; b < 169; b++)
            if (root[p][b].present) present++;
            else missing++;
    printf("\n--- PREFLOP ROOT: %d/1014 present, %d missing ---\n", present, missing);
    if (missing > 0) {
        printf("  Missing:\n");
        int shown = 0;
        for (int p = 0; p < 6; p++)
            for (int b = 0; b < 169; b++)
                if (!root[p][b].present && shown++ < 30)
                    printf("    %s %s\n", player_names[p], bucket_names[b]);
        if (missing > 30)
            printf("    ... and %d more\n", missing - 30);
    }

    /* Non-root all-zero examples */
    printf("\n--- NON-ROOT ALL-ZERO PREFLOP (first 30) ---\n");
    int shown = 0;
    for (int i = 0; i < n_zeros && shown < 30; i++) {
        if (zeros[i].action_hash == ROOT_AH) continue;
        printf("  %s %-4s ah=%016llx na=%d\n",
               player_names[zeros[i].player < 6 ? zeros[i].player : 0],
               zeros[i].bucket < 169 ? bucket_names[zeros[i].bucket] : "?",
               (unsigned long long)zeros[i].action_hash, zeros[i].na);
        shown++;
    }

    /* Non-root tiny examples */
    printf("\n--- NON-ROOT TINY PREFLOP (first 30) ---\n");
    shown = 0;
    for (int i = 0; i < n_tinys && shown < 30; i++) {
        if (tinys[i].action_hash == ROOT_AH) continue;
        printf("  %s %-4s ah=%016llx max|r|=%d [",
               player_names[tinys[i].player < 6 ? tinys[i].player : 0],
               tinys[i].bucket < 169 ? bucket_names[tinys[i].bucket] : "?",
               (unsigned long long)tinys[i].action_hash, tinys[i].max_abs);
        for (int a = 0; a < tinys[i].na; a++)
            printf("%s%+d", a ? "," : "", tinys[i].regrets[a]);
        printf("]\n");
        shown++;
    }

    free(g_buf);
    printf("\nDone.\n");
    return 0;
}
