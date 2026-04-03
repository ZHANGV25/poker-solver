/*
 * Self-Play Simulator — Simulate hands using blueprint strategies.
 *
 * Loads a BPR3 checkpoint into a hash table, then simulates 6-player
 * NLHE hands using regret-matched strategies for action selection.
 *
 * Build:
 *   gcc -O2 -o self_play verification/self_play.c -lm
 *   (needs ~80 GB RAM for full checkpoint, or much less for test data)
 *
 * Usage:
 *   ./self_play <checkpoint.bin> [num_hands] [seed]
 *
 * Output: JSON-formatted metrics by position.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

/* ── Card representation ─────────────────────────────────────────── */

#define CARD_RANK(c) ((c) >> 2)
#define CARD_SUIT(c) ((c) & 3)
#define MAKE_CARD(rank, suit) (((rank) << 2) | (suit))

#define NUM_CARDS 52
#define NUM_PLAYERS 6
#define MAX_ACTIONS 20
#define BIG_BLIND 100
#define SMALL_BLIND 50
#define INITIAL_STACK 10000  /* 100 BB */
#define ROOT_ACTION_HASH 0xFEDCBA9876543210ULL

/* ── 7-card hand evaluation (inline from hand_eval.h logic) ──────── */

static uint32_t eval5(int c0, int c1, int c2, int c3, int c4) {
    int ranks[5] = {CARD_RANK(c0), CARD_RANK(c1), CARD_RANK(c2),
                    CARD_RANK(c3), CARD_RANK(c4)};
    int suits[5] = {CARD_SUIT(c0), CARD_SUIT(c1), CARD_SUIT(c2),
                    CARD_SUIT(c3), CARD_SUIT(c4)};

    for (int i = 1; i < 5; i++) {
        int key = ranks[i]; int j = i - 1;
        while (j >= 0 && ranks[j] < key) { ranks[j+1] = ranks[j]; j--; }
        ranks[j+1] = key;
    }

    int is_flush = (suits[0]==suits[1] && suits[1]==suits[2] &&
                    suits[2]==suits[3] && suits[3]==suits[4]);
    int is_straight = 0, straight_high = ranks[0];
    if (ranks[0]-ranks[4]==4 && ranks[0]!=ranks[1] && ranks[1]!=ranks[2] &&
        ranks[2]!=ranks[3] && ranks[3]!=ranks[4])
        is_straight = 1;
    if (ranks[0]==12 && ranks[1]==3 && ranks[2]==2 && ranks[3]==1 && ranks[4]==0)
        { is_straight = 1; straight_high = 3; }

    if (is_straight && is_flush) return (9<<20) | (straight_high<<16);
    if (is_flush) return (6<<20)|(ranks[0]<<16)|(ranks[1]<<12)|(ranks[2]<<8)|(ranks[3]<<4)|ranks[4];
    if (is_straight) return (5<<20) | (straight_high<<16);

    int counts[13] = {0};
    for (int i = 0; i < 5; i++) counts[ranks[i]]++;

    int quads=-1, trips=-1, p1=-1, p2=-1;
    for (int r = 12; r >= 0; r--) {
        if (counts[r]==4) quads=r;
        else if (counts[r]==3) trips=r;
        else if (counts[r]==2) { if (p1<0) p1=r; else p2=r; }
    }

    if (quads>=0) { int k=-1; for (int r=12;r>=0;r--) if (counts[r]>0&&r!=quads){k=r;break;} return (8<<20)|(quads<<16)|(k<<12); }
    if (trips>=0 && p1>=0) return (7<<20)|(trips<<16)|(p1<<12);
    if (trips>=0) { int k0=-1,k1=-1; for(int r=12;r>=0;r--) if(counts[r]>0&&r!=trips){if(k0<0)k0=r;else k1=r;} return (4<<20)|(trips<<16)|(k0<<12)|(k1<<8); }
    if (p1>=0 && p2>=0) { int k=-1; for(int r=12;r>=0;r--) if(counts[r]>0&&r!=p1&&r!=p2){k=r;break;} return (3<<20)|(p1<<16)|(p2<<12)|(k<<8); }
    if (p1>=0) { int k[3],ki=0; for(int r=12;r>=0&&ki<3;r--) if(counts[r]>0&&r!=p1) k[ki++]=r; return (2<<20)|(p1<<16)|(k[0]<<12)|(k[1]<<8)|(k[2]<<4); }
    return (1<<20)|(ranks[0]<<16)|(ranks[1]<<12)|(ranks[2]<<8)|(ranks[3]<<4)|ranks[4];
}

static uint32_t eval7(const int cards[7]) {
    static const int combos[21][5] = {
        {0,1,2,3,4},{0,1,2,3,5},{0,1,2,3,6},{0,1,2,4,5},{0,1,2,4,6},{0,1,2,5,6},
        {0,1,3,4,5},{0,1,3,4,6},{0,1,3,5,6},{0,1,4,5,6},{0,2,3,4,5},{0,2,3,4,6},
        {0,2,3,5,6},{0,2,4,5,6},{0,3,4,5,6},{1,2,3,4,5},{1,2,3,4,6},{1,2,3,5,6},
        {1,2,4,5,6},{1,3,4,5,6},{2,3,4,5,6}
    };
    uint32_t best = 0;
    for (int i = 0; i < 21; i++) {
        uint32_t v = eval5(cards[combos[i][0]], cards[combos[i][1]],
                           cards[combos[i][2]], cards[combos[i][3]], cards[combos[i][4]]);
        if (v > best) best = v;
    }
    return best;
}

/* ── RNG ─────────────────────────────────────────────────────────── */

static uint64_t rng_state;

static inline uint64_t rng_next(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}

static inline int rng_int(int n) {
    return (int)(rng_next() % (uint64_t)n);
}

static inline double rng_double(void) {
    return (double)(rng_next() & 0x1FFFFFFFFFFFFFULL) / (double)0x20000000000000ULL;
}

static void shuffle_deck(int *deck, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rng_int(i + 1);
        int tmp = deck[i]; deck[i] = deck[j]; deck[j] = tmp;
    }
}

/* ── Preflop hand class mapping (169 classes) ────────────────────── */

static int hand_to_preflop_class(int c0, int c1) {
    int r0 = CARD_RANK(c0), r1 = CARD_RANK(c1);
    int suited = (CARD_SUIT(c0) == CARD_SUIT(c1));
    if (r0 < r1) { int t = r0; r0 = r1; r1 = t; }

    /* Build class index matching the ordering in check_convergence.c:
     * For each high rank (A=12 down to 2=0), pairs first, then
     * suited+offsuit combos with lower ranks.
     * AA=0, AKs=1, AKo=2, AQs=3, ... KK=25, KQs=26, ... */
    int idx = 0;
    for (int hi = 12; hi > r0; hi--) {
        idx++; /* pair */
        idx += 2 * (hi); /* suited+offsuit combos with each lower rank */
    }
    if (r0 == r1) {
        return idx; /* pair */
    }
    idx++; /* skip the pair for this high rank */
    /* Suited/offsuit combos: for r0, go through lo from r0-1 down to r1 */
    for (int lo = r0 - 1; lo > r1; lo--) {
        idx += 2; /* suited + offsuit */
    }
    idx += suited ? 0 : 1; /* suited first, then offsuit */
    return idx;
}

/* ── EHS computation (Monte Carlo, simplified) ───────────────────── */

static float compute_ehs(int c0, int c1, const int *board, int num_board, int n_samples) {
    int blocked[52] = {0};
    blocked[c0] = 1; blocked[c1] = 1;
    for (int i = 0; i < num_board; i++) blocked[board[i]] = 1;

    int avail[52], n_avail = 0;
    for (int c = 0; c < 52; c++)
        if (!blocked[c]) avail[n_avail++] = c;

    int cards_needed = 2 + (5 - num_board);
    if (n_avail < cards_needed) return 0.5f;

    int wins = 0, ties = 0, total = 0;
    for (int s = 0; s < n_samples; s++) {
        /* Partial shuffle */
        for (int i = 0; i < cards_needed && i < n_avail; i++) {
            int j = i + rng_int(n_avail - i);
            int tmp = avail[i]; avail[i] = avail[j]; avail[j] = tmp;
        }

        int full_board[5];
        for (int i = 0; i < num_board; i++) full_board[i] = board[i];
        for (int i = num_board; i < 5; i++) full_board[i] = avail[2 + (i - num_board)];

        int hero[7] = {full_board[0],full_board[1],full_board[2],full_board[3],full_board[4],c0,c1};
        int opp[7] = {full_board[0],full_board[1],full_board[2],full_board[3],full_board[4],avail[0],avail[1]};

        uint32_t hs = eval7(hero), os = eval7(opp);
        if (hs > os) wins++;
        else if (hs == os) ties++;
        total++;
    }
    return total > 0 ? ((float)wins + 0.5f * (float)ties) / (float)total : 0.5f;
}

static int ehs_to_bucket(float ehs, int num_buckets) {
    int b = (int)(ehs * num_buckets);
    if (b >= num_buckets) b = num_buckets - 1;
    if (b < 0) b = 0;
    return b;
}

/* ── Info set hash table ─────────────────────────────────────────── */

typedef struct {
    int player;
    int street;
    int bucket;
    uint64_t board_hash;
    uint64_t action_hash;
} InfoKey;

typedef struct {
    int num_actions;
    int regrets[MAX_ACTIONS];
} InfoEntry;

/* Simple open-addressing hash table */
#define HT_SIZE (1 << 24)  /* 16M slots — fits test data, real data needs bigger */
#define HT_MASK (HT_SIZE - 1)

static InfoKey *ht_keys;
static InfoEntry *ht_entries;
static int *ht_occupied;
static int ht_count = 0;
static int ht_actual_size;

/* Forward declaration — defined below with other hash functions */
static uint64_t hash_combine(uint64_t a, uint64_t b);

static uint64_t hash_key(InfoKey k) {
    /* Match mccfr_blueprint.c info_table hash: hash_combine chain */
    uint64_t h = hash_combine(k.board_hash, k.action_hash);
    h = hash_combine(h, (uint64_t)k.player);
    h = hash_combine(h, (uint64_t)k.street);
    h = hash_combine(h, (uint64_t)k.bucket);
    return h;
}

static int ht_init(int size) {
    ht_actual_size = size;
    ht_keys = (InfoKey *)calloc(size, sizeof(InfoKey));
    ht_entries = (InfoEntry *)calloc(size, sizeof(InfoEntry));
    ht_occupied = (int *)calloc(size, sizeof(int));
    if (!ht_keys || !ht_entries || !ht_occupied) return -1;
    return 0;
}

static void ht_insert(InfoKey key, const int *regrets, int na) {
    uint64_t h = hash_key(key);
    int idx = (int)(h & (ht_actual_size - 1));
    while (ht_occupied[idx]) {
        idx = (idx + 1) & (ht_actual_size - 1);
    }
    ht_keys[idx] = key;
    ht_entries[idx].num_actions = na;
    for (int i = 0; i < na && i < MAX_ACTIONS; i++)
        ht_entries[idx].regrets[i] = regrets[i];
    ht_occupied[idx] = 1;
    ht_count++;
}

static InfoEntry *ht_lookup(InfoKey key) {
    uint64_t h = hash_key(key);
    int idx = (int)(h & (ht_actual_size - 1));
    int probes = 0;
    while (ht_occupied[idx] && probes < 1000) {
        InfoKey *k = &ht_keys[idx];
        if (k->player == key.player && k->street == key.street &&
            k->bucket == key.bucket && k->board_hash == key.board_hash &&
            k->action_hash == key.action_hash) {
            return &ht_entries[idx];
        }
        idx = (idx + 1) & (ht_actual_size - 1);
        probes++;
    }
    return NULL;
}

/* ── Checkpoint loading ──────────────────────────────────────────── */

static int64_t load_checkpoint(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return -1; }

    char *iobuf = malloc(16 * 1024 * 1024);
    setvbuf(f, iobuf, _IOFBF, 16 * 1024 * 1024);

    char magic[4];
    int table_size, num_entries;
    int64_t iterations;
    fread(magic, 1, 4, f);
    int is_v3 = (memcmp(magic, "BPR3", 4) == 0);
    int is_v2 = (memcmp(magic, "BPR2", 4) == 0);
    if (!is_v3 && !is_v2) { fprintf(stderr, "Bad magic\n"); fclose(f); free(iobuf); return -1; }
    fread(&table_size, 4, 1, f);
    fread(&num_entries, 4, 1, f);
    if (is_v3) { fread(&iterations, 8, 1, f); }
    else { int i32; fread(&i32, 4, 1, f); iterations = (int64_t)i32; }

    fprintf(stderr, "Loading: %d entries, %lld iterations\n", num_entries, (long long)iterations);

    /* Size hash table to ~2x entries for good load factor */
    int ht_size_bits = 14; /* start at 16K */
    while ((1 << ht_size_bits) < num_entries * 2) ht_size_bits++;
    if (ht_size_bits > 30) ht_size_bits = 30;
    int ht_sz = 1 << ht_size_bits;
    fprintf(stderr, "Hash table: %d slots (%d bits)\n", ht_sz, ht_size_bits);

    if (ht_init(ht_sz) < 0) {
        fprintf(stderr, "Failed to allocate hash table\n");
        fclose(f); free(iobuf); return -1;
    }

    int player, street, bucket, na, has_sum;
    uint64_t board_hash, action_hash;
    int regrets[MAX_ACTIONS];
    int loaded = 0;

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

        InfoKey key = {player, street, bucket, board_hash, action_hash};
        ht_insert(key, regrets, na);
        loaded++;

        if (loaded % 10000000 == 0)
            fprintf(stderr, "  %d entries loaded...\n", loaded);
    }

    fclose(f);
    free(iobuf);
    fprintf(stderr, "Loaded %d entries\n", loaded);
    return iterations;
}

/* ── Regret matching ─────────────────────────────────────────────── */

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

static int sample_action(const float *strat, int na) {
    double r = rng_double();
    double cum = 0;
    for (int i = 0; i < na; i++) {
        cum += strat[i];
        if (r < cum) return i;
    }
    return na - 1;
}

/* ── Hash functions (must match mccfr_blueprint.c exactly) ────────── */

static uint64_t hash_combine(uint64_t a, uint64_t b) {
    a ^= b + 0x9e3779b97f4a7c15ULL + (a << 6) + (a >> 2);
    return a;
}

static uint64_t compute_board_hash(const int *board, int n) {
    uint64_t h = 0x123456789ABCDEFULL;
    for (int i = 0; i < n; i++)
        h = hash_combine(h, (uint64_t)board[i] * 31 + 7);
    return h;
}

/* Canonicalize board for suit-isomorphic info set hashing.
 * Ported from mccfr_blueprint.c:canonicalize_board(). */
static void canonicalize_board(const int *board, int num_board, int *canon_out) {
    if (num_board == 0) return;

    int sorted[3], canon_flop[3];
    int suit_map[4] = {-1, -1, -1, -1};

    /* Sort flop cards by rank descending */
    for (int i = 0; i < 3 && i < num_board; i++) sorted[i] = board[i];
    for (int a = 0; a < 2; a++)
        for (int b = a+1; b < 3; b++)
            if ((sorted[a]>>2) < (sorted[b]>>2)) {
                int tmp = sorted[a]; sorted[a] = sorted[b]; sorted[b] = tmp;
            }

    int r0 = sorted[0]>>2, r1 = sorted[1]>>2, r2 = sorted[2]>>2;
    int s0 = sorted[0]&3, s1 = sorted[1]&3, s2 = sorted[2]&3;

    if (r0 == r1 && r1 == r2) {
        canon_flop[0] = r0*4+3; canon_flop[1] = r1*4+2; canon_flop[2] = r2*4+1;
    } else if (r0 == r1 || r1 == r2) {
        if (r0 == r1) {
            if (s2 == s0 || s2 == s1) {
                canon_flop[0] = r0*4+3; canon_flop[1] = r1*4+2; canon_flop[2] = r2*4+3;
            } else {
                canon_flop[0] = r0*4+3; canon_flop[1] = r1*4+2; canon_flop[2] = r2*4+1;
            }
        } else {
            if (s0 == s1 || s0 == s2) {
                canon_flop[0] = r0*4+3; canon_flop[1] = r1*4+3; canon_flop[2] = r2*4+2;
            } else {
                canon_flop[0] = r0*4+3; canon_flop[1] = r1*4+2; canon_flop[2] = r2*4+1;
            }
        }
    } else {
        if (s0 == s1 && s1 == s2) {
            canon_flop[0] = r0*4+3; canon_flop[1] = r1*4+3; canon_flop[2] = r2*4+3;
        } else if (s0 == s1) {
            canon_flop[0] = r0*4+3; canon_flop[1] = r1*4+3; canon_flop[2] = r2*4+2;
        } else if (s0 == s2) {
            canon_flop[0] = r0*4+3; canon_flop[1] = r1*4+2; canon_flop[2] = r2*4+3;
        } else if (s1 == s2) {
            canon_flop[0] = r0*4+2; canon_flop[1] = r1*4+3; canon_flop[2] = r2*4+3;
        } else {
            canon_flop[0] = r0*4+3; canon_flop[1] = r1*4+2; canon_flop[2] = r2*4+1;
        }
    }

    /* Build suit mapping from sorted actual -> canonical */
    for (int i = 0; i < 3; i++) {
        int as = sorted[i] & 3;
        int cs = canon_flop[i] & 3;
        if (suit_map[as] == -1) suit_map[as] = cs;
    }
    int nc = 0;
    for (int i = 0; i < 4; i++) {
        if (suit_map[i] == -1) {
            while (nc < 4) {
                int used = 0;
                for (int j = 0; j < 4; j++)
                    if (suit_map[j] == nc) { used = 1; break; }
                if (!used) break;
                nc++;
            }
            if (nc < 4) suit_map[i] = nc++;
            else suit_map[i] = 0;
        }
    }

    for (int i = 0; i < 3 && i < num_board; i++)
        canon_out[i] = canon_flop[i];
    for (int i = 3; i < num_board; i++) {
        int rank = board[i] >> 2;
        int suit = board[i] & 3;
        canon_out[i] = rank * 4 + suit_map[suit];
    }
}

/* ── Action hash (must match mccfr_blueprint.c:compute_action_hash) ─ */

static int action_history[256];
static int action_history_len = 0;

static uint64_t compute_action_hash(const int *actions, int num_actions) {
    uint64_t h = 0xFEDCBA9876543210ULL;
    for (int i = 0; i < num_actions; i++)
        h = hash_combine(h, (uint64_t)actions[i] * 17 + 3);
    return h;
}

/* ── Game simulation ─────────────────────────────────────────────── */

typedef struct {
    /* Per-position accumulators */
    double winnings[NUM_PLAYERS];   /* total chips won/lost */
    long long hands_played;
    long long showdowns;            /* hands reaching showdown */
    long long folds_to_first_bet;   /* folded to first bet */
    long long vpip[NUM_PLAYERS];    /* voluntarily put money in pot */
    double pot_total;               /* total pot sizes */
    long long hands_won[NUM_PLAYERS];
    long long lookups_found;
    long long lookups_missed;
} SimStats;

static void simulate_hand(SimStats *stats, int ehs_samples) {
    int deck[52];
    for (int i = 0; i < 52; i++) deck[i] = i;
    shuffle_deck(deck, 52);

    /* Deal hole cards */
    int hole[NUM_PLAYERS][2];
    int di = 0;
    for (int p = 0; p < NUM_PLAYERS; p++) {
        hole[p][0] = deck[di++];
        hole[p][1] = deck[di++];
    }

    /* Board cards */
    int board[5];
    for (int i = 0; i < 5; i++) board[i] = deck[di++];

    /* Game state */
    int stack[NUM_PLAYERS];
    int bet[NUM_PLAYERS];
    int folded[NUM_PLAYERS];
    int all_in[NUM_PLAYERS];
    for (int p = 0; p < NUM_PLAYERS; p++) {
        stack[p] = INITIAL_STACK;
        bet[p] = 0;
        folded[p] = 0;
        all_in[p] = 0;
    }

    /* Post blinds: SB=0, BB=1 */
    bet[0] = SMALL_BLIND; stack[0] -= SMALL_BLIND;
    bet[1] = BIG_BLIND; stack[1] -= BIG_BLIND;
    int pot = SMALL_BLIND + BIG_BLIND;

    int active_players = NUM_PLAYERS;
    int went_to_showdown = 0;

    /* Track VPIP (posting blinds doesn't count) */
    int has_vpip[NUM_PLAYERS];
    memset(has_vpip, 0, sizeof(has_vpip));

    /* Preflop: UTG(2) acts first, then MP(3), CO(4), BTN(5), SB(0), BB(1) */
    int preflop_order[] = {2, 3, 4, 5, 0, 1};
    /* Postflop: SB(0), BB(1), UTG(2), MP(3), CO(4), BTN(5) */
    int postflop_order[] = {0, 1, 2, 3, 4, 5};

    int streets[] = {0, 1, 2, 3}; /* preflop, flop, turn, river */
    int board_cards_per_street[] = {0, 3, 4, 5};

    action_history_len = 0;
    uint64_t action_hash = ROOT_ACTION_HASH;
    int first_bet_happened = 0;

    for (int si = 0; si < 4; si++) {
        int street = streets[si];
        int num_board = board_cards_per_street[si];
        uint64_t board_hash;
        if (num_board > 0) {
            int canon[5];
            canonicalize_board(board, num_board, canon);
            board_hash = compute_board_hash(canon, num_board);
        } else {
            board_hash = 0;
        }

        int *order = (street == 0) ? preflop_order : postflop_order;

        /* Reset bets for new street (except preflop) */
        if (street > 0) {
            for (int p = 0; p < NUM_PLAYERS; p++) bet[p] = 0;
        }

        int current_bet = (street == 0) ? BIG_BLIND : 0;
        int last_raiser = -1;
        int actions_taken = 0;
        int players_acted = 0;

        /* Keep going until everyone has acted and action is closed */
        int max_rounds = NUM_PLAYERS * 4; /* prevent infinite loops */
        int round = 0;

        for (round = 0; round < max_rounds; round++) {
            int pi = round % NUM_PLAYERS;
            int p = order[pi];

            if (folded[p] || all_in[p]) continue;
            if (round >= NUM_PLAYERS && p == last_raiser) break;

            /* Compute bucket */
            int bucket;
            if (street == 0) {
                bucket = hand_to_preflop_class(hole[p][0], hole[p][1]);
            } else {
                float ehs = compute_ehs(hole[p][0], hole[p][1], board, num_board, ehs_samples);
                bucket = ehs_to_bucket(ehs, 200);
            }

            /* Look up strategy */
            InfoKey key = {p, street, bucket, board_hash, action_hash};
            InfoEntry *entry = ht_lookup(key);

            int action;
            if (entry) {
                stats->lookups_found++;
                float strat[MAX_ACTIONS];
                regret_match(entry->regrets, strat, entry->num_actions);
                action = sample_action(strat, entry->num_actions);
            } else {
                stats->lookups_missed++;
                /* Default: call 70%, fold 30% if facing bet; check 80%, bet 20% otherwise */
                double r = rng_double();
                if (bet[p] < current_bet) {
                    action = (r < 0.30) ? 0 : 1; /* 30% fold, 70% call */
                } else {
                    action = (r < 0.80) ? 1 : 2; /* 80% check, 20% bet */
                }
            }

            /* Execute action */
            int facing_bet = (current_bet > bet[p]);
            int action_type; /* 0=fold, 1=check/call, 2+=raise */

            if (facing_bet) {
                if (action == 0) {
                    /* Fold */
                    folded[p] = 1;
                    active_players--;
                    action_type = 0;
                    if (!first_bet_happened) {
                        stats->folds_to_first_bet++;
                        first_bet_happened = 1;
                    }
                } else if (action == 1) {
                    /* Call */
                    int call_amount = current_bet - bet[p];
                    if (call_amount > stack[p]) call_amount = stack[p];
                    stack[p] -= call_amount;
                    pot += call_amount;
                    bet[p] += call_amount;
                    if (!has_vpip[p]) { has_vpip[p] = 1; stats->vpip[p]++; }
                    action_type = 1;
                } else {
                    /* Raise */
                    int call_amount = current_bet - bet[p];
                    float raise_mult = 1.0f; /* default 1x pot */
                    if (action == 2) raise_mult = 0.5f;
                    else if (action == 3) raise_mult = 1.0f;
                    else if (action == 4) raise_mult = 2.0f;
                    else raise_mult = 3.0f;

                    int raise_amount = (int)(pot * raise_mult) + call_amount;
                    if (raise_amount > stack[p]) raise_amount = stack[p];
                    stack[p] -= raise_amount;
                    pot += raise_amount;
                    bet[p] += raise_amount;
                    current_bet = bet[p];
                    last_raiser = p;
                    if (!has_vpip[p]) { has_vpip[p] = 1; stats->vpip[p]++; }
                    action_type = action;
                }
            } else {
                if (action <= 1) {
                    /* Check */
                    action_type = 1;
                } else {
                    /* Bet (open) */
                    float bet_mult = 0.5f;
                    if (action == 2) bet_mult = 0.5f;
                    else if (action == 3) bet_mult = 1.0f;
                    else if (action == 4) bet_mult = 2.0f;
                    else bet_mult = 3.0f;

                    int bet_amount = (int)(pot * bet_mult);
                    if (bet_amount < BIG_BLIND) bet_amount = BIG_BLIND;
                    if (bet_amount > stack[p]) bet_amount = stack[p];
                    stack[p] -= bet_amount;
                    pot += bet_amount;
                    bet[p] = bet_amount;
                    current_bet = bet_amount;
                    last_raiser = p;
                    first_bet_happened = 1;
                    if (!has_vpip[p] && street == 0) { has_vpip[p] = 1; stats->vpip[p]++; }
                    action_type = action;
                }
            }

            if (action_history_len < 256)
                action_history[action_history_len++] = action_type;
            action_hash = compute_action_hash(action_history, action_history_len);
            actions_taken++;

            if (active_players <= 1) break;

            /* Check if betting round is complete */
            if (actions_taken >= active_players && last_raiser < 0) break;
        }

        if (active_players <= 1) break;
    }

    /* Determine winner(s) and compute all players' results */
    stats->pot_total += pot;

    int winner_share[NUM_PLAYERS];
    memset(winner_share, 0, sizeof(winner_share));

    if (active_players == 1) {
        for (int p = 0; p < NUM_PLAYERS; p++) {
            if (!folded[p]) {
                winner_share[p] = pot;
                stats->hands_won[p]++;
                break;
            }
        }
    } else {
        /* Showdown */
        went_to_showdown = 1;
        stats->showdowns++;

        uint32_t best_str = 0;
        int winners[NUM_PLAYERS], n_winners = 0;

        for (int p = 0; p < NUM_PLAYERS; p++) {
            if (folded[p]) continue;
            int cards[7] = {board[0],board[1],board[2],board[3],board[4],hole[p][0],hole[p][1]};
            uint32_t str = eval7(cards);
            if (str > best_str) {
                best_str = str;
                n_winners = 1;
                winners[0] = p;
            } else if (str == best_str) {
                winners[n_winners++] = p;
            }
        }

        int share = pot / n_winners;
        int remainder = pot - share * n_winners;
        for (int w = 0; w < n_winners; w++) {
            winner_share[winners[w]] = share + (w == 0 ? remainder : 0);
            stats->hands_won[winners[w]]++;
        }
    }

    /* Update all players' winnings: net = (chips_remaining + share) - start */
    for (int p = 0; p < NUM_PLAYERS; p++) {
        int net = (stack[p] + winner_share[p]) - INITIAL_STACK;
        stats->winnings[p] += net;
    }

    stats->hands_played++;
}

/* ── Main ────────────────────────────────────────────────────────── */

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "verification/test_checkpoint.bin";
    int num_hands = argc > 2 ? atoi(argv[2]) : 10000;
    uint64_t seed = argc > 3 ? (uint64_t)atoll(argv[3]) : (uint64_t)time(NULL);
    int ehs_samples = argc > 4 ? atoi(argv[4]) : 100;

    rng_state = seed;
    if (rng_state == 0) rng_state = 0xDEADBEEF;

    /* Load checkpoint */
    int64_t iterations = load_checkpoint(path);
    if (iterations < 0) return 1;

    fprintf(stderr, "\nSimulating %d hands (seed=%llu, ehs_samples=%d)...\n",
            num_hands, (unsigned long long)seed, ehs_samples);

    /* Run simulation */
    SimStats stats;
    memset(&stats, 0, sizeof(stats));

    for (int h = 0; h < num_hands; h++) {
        simulate_hand(&stats, ehs_samples);
        if ((h + 1) % 10000 == 0)
            fprintf(stderr, "  %d / %d hands...\n", h + 1, num_hands);
    }

    /* Output results */
    static const char *POS_NAMES[] = {"SB","BB","UTG","MP","CO","BTN"};

    printf("SELF_PLAY iterations=%lld hands=%lld seed=%llu\n\n",
           (long long)iterations, stats.hands_played, (unsigned long long)seed);

    printf("=== WIN RATE (bb/100) ===\n");
    double total_wr = 0;
    for (int p = 0; p < NUM_PLAYERS; p++) {
        double bb100 = stats.winnings[p] / (double)stats.hands_played * 100.0 / BIG_BLIND;
        printf("  %s: %+.2f bb/100\n", POS_NAMES[p], bb100);
        total_wr += bb100;
    }
    printf("  Sum: %+.2f bb/100\n", total_wr);
    printf("WINRATE_SUM_CHECK: %s (threshold: +/- 2.0)\n\n",
           fabs(total_wr) < 2.0 ? "PASS" : "FAIL");

    printf("=== SHOWDOWN FREQUENCY ===\n");
    double sd_pct = (double)stats.showdowns / stats.hands_played * 100;
    printf("  Showdown: %lld / %lld (%.1f%%)\n", stats.showdowns, stats.hands_played, sd_pct);
    printf("SHOWDOWN_CHECK: %s (expected: 25-30%%)\n\n",
           (sd_pct >= 15 && sd_pct <= 45) ? "PASS" : "FAIL");

    printf("=== FOLD TO FIRST BET ===\n");
    double ftb_pct = (double)stats.folds_to_first_bet / stats.hands_played * 100;
    printf("  Fold to first bet: %.1f%%\n", ftb_pct);
    printf("FOLD_TO_BET_CHECK: %s (expected: 40-60%%)\n\n",
           (ftb_pct >= 20 && ftb_pct <= 80) ? "PASS" : "FAIL");

    printf("=== AVERAGE POT SIZE ===\n");
    double avg_pot = stats.pot_total / stats.hands_played;
    printf("  Average pot: %.0f chips (%.1f BB)\n", avg_pot, avg_pot / BIG_BLIND);
    printf("POT_SIZE_CHECK: %s (expected: 2-20 BB)\n\n",
           (avg_pot / BIG_BLIND >= 1.0 && avg_pot / BIG_BLIND <= 50.0) ? "PASS" : "FAIL");

    printf("=== VPIP BY POSITION ===\n");
    for (int p = 0; p < NUM_PLAYERS; p++) {
        double vpip_pct = (double)stats.vpip[p] / stats.hands_played * 100;
        printf("  %s: %.1f%%\n", POS_NAMES[p], vpip_pct);
    }
    /* BTN should be highest, UTG lowest */
    double btn_vpip = (double)stats.vpip[5] / stats.hands_played;
    double utg_vpip = (double)stats.vpip[2] / stats.hands_played;
    printf("VPIP_ORDER_CHECK: %s (BTN > UTG expected)\n\n",
           btn_vpip >= utg_vpip ? "PASS" : "FAIL");

    printf("=== LOOKUP STATS ===\n");
    long long total_lookups = stats.lookups_found + stats.lookups_missed;
    printf("  Found: %lld / %lld (%.1f%%)\n",
           stats.lookups_found, total_lookups,
           total_lookups > 0 ? (double)stats.lookups_found / total_lookups * 100 : 0);
    printf("  Missed (used default): %lld\n\n", stats.lookups_missed);

    printf("=== SUMMARY ===\n");
    int checks = 0, passed = 0;
    checks++; if (fabs(total_wr) < 2.0) passed++;
    checks++; if (sd_pct >= 15 && sd_pct <= 45) passed++;
    checks++; if (ftb_pct >= 20 && ftb_pct <= 80) passed++;
    checks++; if (avg_pot / BIG_BLIND >= 1.0 && avg_pot / BIG_BLIND <= 50.0) passed++;
    checks++; if (btn_vpip >= utg_vpip) passed++;
    printf("  %d / %d checks passed\n", passed, checks);

    return (passed == checks) ? 0 : 1;
}
