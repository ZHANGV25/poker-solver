/**
 * mccfr_blueprint.c — Production N-player external-sampling MCCFR
 *
 * Matches the Pluribus blueprint training algorithm:
 *   - External-sampling MCCFR with Linear CFR + regret-based pruning
 *   - OpenMP Hogwild-style parallelism (lock-free shared regret tables)
 *   - Integer regrets (int32) with floor at -310M
 *   - Card abstraction via hand-to-bucket mapping
 *   - Multi-street: flop -> turn -> river -> showdown
 *
 * Thread safety: each thread runs independent iterations with its own
 * RNG state. Regret/strategy updates are unsynchronized (Hogwild-style).
 * This is safe because external sampling visits a sparse subset of info
 * sets per iteration, so collisions are rare and the noise is negligible.
 */

#include "mccfr_blueprint.h"
#include "card_abstraction.h"
#include "hand_eval.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#ifdef _POSIX_VERSION
#include <sched.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

/* ── Lookup miss counter (for diagnosing hash table saturation) ───── */
static int64_t g_lookup_total = 0;
static int64_t g_lookup_miss = 0;

void bp_get_miss_stats(int64_t *total, int64_t *miss) {
    *total = __atomic_load_n(&g_lookup_total, __ATOMIC_RELAXED);
    *miss = __atomic_load_n(&g_lookup_miss, __ATOMIC_RELAXED);
}

void bp_reset_miss_stats(void) {
    __atomic_store_n(&g_lookup_total, 0, __ATOMIC_RELAXED);
    __atomic_store_n(&g_lookup_miss, 0, __ATOMIC_RELAXED);
}

/* ── RNG (xorshift64, thread-safe via separate states) ────────────── */

static inline uint64_t rng_next(uint64_t *state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

static inline int rng_int(uint64_t *state, int n) {
    return (int)(rng_next(state) % (uint64_t)n);
}

static inline float rng_float(uint64_t *state) {
    return (float)(rng_next(state) & 0xFFFFFF) / (float)0x1000000;
}

/* Fisher-Yates partial shuffle: pick k items from array of n */
static void partial_shuffle(int *arr, int n, int k, uint64_t *rng) {
    for (int i = 0; i < k && i < n; i++) {
        int j = i + rng_int(rng, n - i);
        int tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
    }
}

/* ── Regret arena allocator ──────────────────────────────────────── */
/* Eliminates per-info-set malloc overhead. Each calloc(4, sizeof(int))
 * adds 16-32 bytes of glibc malloc header, doubling memory for tiny arrays.
 * Arena allocates in bulk 64MB chunks, returning aligned 32-byte blocks. */

#define ARENA_CHUNK_SIZE (64 * 1024 * 1024)  /* 64 MB per chunk */
#define ARENA_BLOCK_SIZE 64                    /* 64 bytes = 16 ints max */

typedef struct ArenaChunk {
    char *data;
    int used;       /* bytes used in this chunk */
    int capacity;
    struct ArenaChunk *next;
} ArenaChunk;

typedef struct {
    ArenaChunk *head;
    int total_chunks;
} RegretArena;

static RegretArena g_arena = {NULL, 0};

static void arena_init(void) {
    /* Nothing to do — lazy allocation on first use */
}

/* Thread-safe arena_alloc using atomic fetch-add on chunk->used.
 * New chunk allocation uses a simple spinlock (rare path). */
static int g_arena_lock = 0;

static void *arena_alloc(int num_ints) {
    int size = num_ints * (int)sizeof(int);
    if (size < ARENA_BLOCK_SIZE) size = ARENA_BLOCK_SIZE;
    /* Round up to 8-byte alignment */
    size = (size + 7) & ~7;

retry:;
    ArenaChunk *chunk = g_arena.head;
    if (chunk) {
        int old = __atomic_fetch_add(&chunk->used, size, __ATOMIC_RELAXED);
        if (old + size <= chunk->capacity) {
            return chunk->data + old;
        }
        /* Chunk full — fall through to allocate new one */
    }

    /* Rare path: allocate new chunk under spinlock */
    int expected = 0;
    if (!__atomic_compare_exchange_n(&g_arena_lock, &expected, 1,
                                     0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
        /* Another thread is allocating — spin with pause then retry */
        for (int spins = 0; __atomic_load_n(&g_arena_lock, __ATOMIC_ACQUIRE); spins++) {
            #ifdef __x86_64__
            __builtin_ia32_pause();
            #endif
            if (spins > 10000) { /* yield after ~10us of spinning */
                #ifdef _OPENMP
                #pragma omp taskyield
                #endif
                spins = 0;
            }
        }
        goto retry;
    }

    /* Double-check after acquiring lock */
    chunk = g_arena.head;
    if (chunk && chunk->used + size <= chunk->capacity) {
        __atomic_store_n(&g_arena_lock, 0, __ATOMIC_RELEASE);
        goto retry;
    }

    ArenaChunk *c = (ArenaChunk*)malloc(sizeof(ArenaChunk));
    if (!c) { __atomic_store_n(&g_arena_lock, 0, __ATOMIC_RELEASE); return NULL; }
    c->data = (char*)calloc(1, ARENA_CHUNK_SIZE);
    if (!c->data) { free(c); __atomic_store_n(&g_arena_lock, 0, __ATOMIC_RELEASE); return NULL; }
    c->capacity = ARENA_CHUNK_SIZE;
    c->used = 0;
    c->next = g_arena.head;
    g_arena.head = c;
    g_arena.total_chunks++;
    __atomic_store_n(&g_arena_lock, 0, __ATOMIC_RELEASE);
    goto retry;
}

static void arena_free_all(void) {
    ArenaChunk *c = g_arena.head;
    while (c) {
        ArenaChunk *next = c->next;
        free(c->data);
        free(c);
        c = next;
    }
    g_arena.head = NULL;
    g_arena.total_chunks = 0;
}

/* ── Hash table ───────────────────────────────────────────────────── */

static uint64_t hash_combine(uint64_t a, uint64_t b) {
    a ^= b + 0x9e3779b97f4a7c15ULL + (a << 6) + (a >> 2);
    return a;
}

static uint64_t compute_board_hash(const int *board, int num_board) {
    uint64_t h = 0x123456789ABCDEFULL;
    for (int i = 0; i < num_board; i++)
        h = hash_combine(h, (uint64_t)board[i] * 31 + 7);
    return h;
}

/* Canonicalize a board for info set hashing.
 * Maps actual board cards to suit-isomorphic canonical cards so that
 * boards with the same texture share info set entries.
 * canon_out must have space for num_board cards. */
static void canonicalize_board(const int *board, int num_board, int *canon_out) {
    if (num_board == 0) return;

    /* Step 1: Canonicalize the flop (first 3 cards) */
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

    /* Determine canonical suits based on pattern (same logic as traversal) */
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

    /* Build suit mapping from sorted actual → canonical */
    for (int i = 0; i < 3; i++) {
        int as = sorted[i] & 3;
        int cs = canon_flop[i] & 3;
        if (suit_map[as] == -1) suit_map[as] = cs;
    }
    /* Assign unmapped suits to remaining canonical suits */
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

    /* Output canonical board */
    for (int i = 0; i < 3 && i < num_board; i++)
        canon_out[i] = canon_flop[i];

    /* Turn and river: apply suit mapping */
    for (int i = 3; i < num_board; i++) {
        int rank = board[i] >> 2;
        int suit = board[i] & 3;
        canon_out[i] = rank * 4 + suit_map[suit];
    }
}

static uint64_t compute_action_hash(const int *actions, int num_actions) {
    uint64_t h = 0xFEDCBA9876543210ULL;
    for (int i = 0; i < num_actions; i++)
        h = hash_combine(h, (uint64_t)actions[i] * 17 + 3);
    return h;
}

static void info_table_init(BPInfoTable *t, int64_t table_size) {
    t->table_size = table_size;
    t->keys = (BPInfoKey*)calloc((size_t)table_size, sizeof(BPInfoKey));
    t->sets = (BPInfoSet*)calloc((size_t)table_size, sizeof(BPInfoSet));
    t->occupied = (int*)calloc((size_t)table_size, sizeof(int));
    t->num_entries = 0;
}

/* Lock-free find-or-create using atomic CAS on the occupied flag.
 *
 * occupied states: 0 = empty, 1 = ready, 2 = being initialized
 *
 * Protocol:
 *   1. CAS(occupied[idx], 0 -> 2): we won the slot, initialize it
 *   2. If occupied[idx] == 2: another thread is initializing, spin until 1
 *   3. If occupied[idx] == 1: slot is ready, check if key matches
 *
 * This is fully lock-free for lookups (common case) and per-slot atomic
 * for insertions (no global lock, no contention between different slots). */
static inline int key_eq(const BPInfoKey *a, const BPInfoKey *b) {
    return a->player == b->player &&
           a->street == b->street &&
           a->bucket == b->bucket &&
           a->board_hash == b->board_hash &&
           a->action_hash == b->action_hash;
}

/* Spin until a state=2 slot finishes initialization (becomes 1).
 * The initializer already won its CAS, so it WILL complete — the only
 * question is when. On 192-core machines, OS preemption can delay the
 * initializer by milliseconds, so we use a long spin (~100ms) and then
 * yield to avoid burning a core indefinitely. */
static inline void spin_until_ready(const int *flag) {
    int spins = 0;
    while (__atomic_load_n(flag, __ATOMIC_ACQUIRE) == 2) {
        #ifdef __x86_64__
        __builtin_ia32_pause();
        #elif defined(__aarch64__)
        __asm__ volatile("yield");
        #endif
        if (++spins > 10000000) {   /* ~10ms, yield then retry */
            spins = 0;
            #ifdef _POSIX_VERSION
            sched_yield();
            #endif
        }
    }
}

static int64_t info_table_find_or_create(BPInfoTable *t, BPInfoKey key,
                                          int num_actions) {
    uint64_t h = hash_combine(key.board_hash, key.action_hash);
    h = hash_combine(h, (uint64_t)key.player);
    h = hash_combine(h, (uint64_t)key.street);
    h = hash_combine(h, (uint64_t)key.bucket);
    int64_t slot = (int64_t)(h % (uint64_t)t->table_size);

    for (int probe = 0; probe < 4096; probe++) {
        int64_t idx = (slot + probe) % t->table_size;
        int state = __atomic_load_n(&t->occupied[idx], __ATOMIC_ACQUIRE);

        if (state == 1) {
            if (key_eq(&t->keys[idx], &key)) return idx;
            continue;
        }

        if (state == 0) {
            int expected = 0;
            if (__atomic_compare_exchange_n(&t->occupied[idx], &expected, 2,
                                            0, __ATOMIC_ACQ_REL,
                                            __ATOMIC_ACQUIRE)) {
                t->sets[idx].num_actions = num_actions;
                t->sets[idx].regrets = (int*)arena_alloc(num_actions);
                t->sets[idx].strategy_sum = NULL;
                t->keys[idx] = key;
                __atomic_store_n(&t->occupied[idx], 1, __ATOMIC_RELEASE);

                /* De-dup check: scan backwards from our position to the
                 * hash start. If the same key was published at an earlier
                 * slot by another thread, we are the duplicate — return
                 * the earlier slot.
                 *
                 * BUG FIX: must spin-wait on state=2 slots during the scan.
                 * Without this, if thread A is still initializing slot S+2
                 * (state=2) when thread B scans, B skips it, misses the
                 * duplicate, and both copies accumulate separate regrets. */
                for (int p2 = 0; p2 < probe; p2++) {
                    int64_t idx2 = (slot + p2) % t->table_size;
                    int st2 = __atomic_load_n(&t->occupied[idx2], __ATOMIC_ACQUIRE);
                    if (st2 == 2) {
                        spin_until_ready(&t->occupied[idx2]);
                        st2 = __atomic_load_n(&t->occupied[idx2], __ATOMIC_ACQUIRE);
                    }
                    if (st2 == 1 && key_eq(&t->keys[idx2], &key)) {
                        /* Earlier copy exists — we're the duplicate.
                         * Slot is wasted (can't reclaim with linear
                         * probing) but regrets go to the right place. */
                        return idx2;
                    }
                }
                __atomic_fetch_add(&t->num_entries, 1, __ATOMIC_RELAXED);
                return idx;
            }
            state = __atomic_load_n(&t->occupied[idx], __ATOMIC_ACQUIRE);
        }

        if (state == 2) {
            /* Another thread is initializing this slot. We MUST wait for it
             * to finish — skipping past could cause us to create a duplicate
             * at a later slot. spin_until_ready handles OS preemption with
             * yield fallback. */
            spin_until_ready(&t->occupied[idx]);
            if (__atomic_load_n(&t->occupied[idx], __ATOMIC_ACQUIRE) == 1) {
                if (key_eq(&t->keys[idx], &key)) return idx;
            }
            continue;
        }
    }
    return -1;
}

/* Allocate strategy_sum for an info set (lazy, for round 1 only).
 * Uses arena allocator (same as regrets) to avoid billions of small
 * heap callocs that cause fragmentation and interact poorly with
 * glibc's thread-local arenas under high concurrency. Arena memory
 * is zeroed (calloc'd chunks), so float 0.0f is correct (IEEE 754). */
static void ensure_strategy_sum(BPInfoSet *is) {
    if (__atomic_load_n((void**)&is->strategy_sum, __ATOMIC_ACQUIRE) == NULL) {
        float *buf = (float*)arena_alloc(is->num_actions);
        if (!buf) return;
        float *expected = NULL;
        if (!__atomic_compare_exchange_n(&is->strategy_sum, &expected, buf,
                                          0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
            /* CAS lost — buf is wasted arena space (can't free individually).
             * This is rare (only on concurrent first access) and each waste
             * is ≤32 bytes. Acceptable trade-off vs heap corruption risk. */
        }
    }
}

static void info_table_free(BPInfoTable *t) {
    /* Both regrets and strategy_sum are arena-allocated — freed in bulk */
    arena_free_all();
    free(t->keys);
    free(t->sets);
    free(t->occupied);
}

/* ── Helpers ──────────────────────────────────────────────────────── */

static inline int cards_conflict(int a0, int a1, int b0, int b1) {
    return (a0 == b0 || a0 == b1 || a1 == b0 || a1 == b1);
}

static inline int card_in_set(int card, const int *set, int n) {
    for (int i = 0; i < n; i++)
        if (set[i] == card) return 1;
    return 0;
}

/* Regret matching from integer regrets.
 * With bucket-in-key, regrets is [num_actions] flat (one bucket per info set). */
static void regret_match(const int *regrets, float *strategy, int num_actions) {
    float sum = 0;
    for (int a = 0; a < num_actions; a++) {
        float pos = (regrets[a] > 0) ? (float)regrets[a] : 0.0f;
        strategy[a] = pos;
        sum += pos;
    }
    if (sum > 0) {
        float inv = 1.0f / sum;
        for (int a = 0; a < num_actions; a++)
            strategy[a] *= inv;
    } else {
        float u = 1.0f / num_actions;
        for (int a = 0; a < num_actions; a++)
            strategy[a] = u;
    }
}

static int sample_action(const float *strategy, int num_actions, uint64_t *rng) {
    float r = rng_float(rng);
    float cumulative = 0;
    for (int a = 0; a < num_actions; a++) {
        cumulative += strategy[a];
        if (r <= cumulative) return a;
    }
    return num_actions - 1;
}

/* Get the bucket index for a hand on a given street */
static inline int get_bucket(const BPSolver *s, int street, int player, int hand_idx) {
    if (!s->use_buckets) return hand_idx;
    if (street < 0 || street > 3) return hand_idx;
    return s->bucket_map[street][player][hand_idx];
}

/* Get num_buckets for a player on a street */
static inline int get_num_buckets(const BPSolver *s, int street, int player) {
    if (!s->use_buckets) return s->num_hands[player];
    if (street < 0 || street > 3) return s->num_hands[player];
    return s->num_buckets[street][player];
}

/* Map num_board to street index: 0=preflop, 3=flop(1), 4=turn(2), 5=river(3) */
static inline int board_to_street(int num_board) {
    if (num_board == 0) return 0; /* preflop */
    if (num_board <= 3) return 1; /* flop */
    if (num_board == 4) return 2; /* turn */
    return 3; /* river */
}

/* ── Betting tree (generated on-the-fly) ─────────────────────────── */

#define ACT_FOLD    0
#define ACT_CHECK   1
#define ACT_CALL    2
#define ACT_BET     3

typedef struct {
    int type;
    int bet_idx;
    int amount;
} BPAction;

static int generate_actions(BPAction *out, int max_out,
                            int pot, int stack, int to_call,
                            int num_raises, int max_raises,
                            const float *bet_sizes, int num_bet_sizes) {
    int n = 0;
    if (to_call > 0 && n < max_out) {
        out[n].type = ACT_FOLD; out[n].bet_idx = -1; out[n].amount = 0; n++;
    }
    if (n < max_out) {
        out[n].type = (to_call > 0) ? ACT_CALL : ACT_CHECK;
        out[n].bet_idx = -1; out[n].amount = to_call; n++;
    }
    if (num_raises < max_raises) {
        int added_allin = 0;
        for (int i = 0; i < num_bet_sizes && n < max_out; i++) {
            int ba;
            if (to_call == 0) ba = (int)(bet_sizes[i] * pot);
            else ba = to_call + (int)(bet_sizes[i] * (pot + to_call));
            if (ba >= stack) ba = stack;
            if (ba <= to_call) continue;
            if (ba >= stack) { if (added_allin) continue; added_allin = 1; }
            out[n].type = ACT_BET; out[n].bet_idx = i; out[n].amount = ba; n++;
        }
        if (!added_allin && stack > to_call && n < max_out) {
            out[n].type = ACT_BET; out[n].bet_idx = num_bet_sizes;
            out[n].amount = stack; n++;
        }
    }
    return n;
}

/* ── Traversal state ─────────────────────────────────────────────── */

typedef struct {
    BPSolver *solver;
    uint64_t *rng;           /* pointer to thread-local RNG */
    int traverser;
    int64_t iteration;
    int use_pruning;         /* 1 if this iteration uses pruning */
    int sampled_hands[BP_MAX_PLAYERS];

    int board[5];
    int num_board;
    int active[BP_MAX_PLAYERS];
    int num_active;
    int bets[BP_MAX_PLAYERS];       /* per-street bets (reset each street) */
    int invested[BP_MAX_PLAYERS];   /* CUMULATIVE total invested across ALL streets */
    int has_acted[BP_MAX_PLAYERS];
    int pot;
    int stacks[BP_MAX_PLAYERS];     /* remaining stack per player */
    int num_raises;

    /* Per-traversal postflop bucket map (for unified solver).
     * Computed once when flop is dealt, reused for turn/river.
     * postflop_bucket[hand_idx] = bucket index (0..199). */
    int postflop_bucket[BP_MAX_HANDS];
    int postflop_num_buckets_actual;
    int postflop_buckets_computed;   /* 1 if buckets have been computed for this flop */

    /* Canonical board for info set keys. Boards with the same texture
     * (suit-isomorphic pattern) map to the same canonical board, so
     * strategically equivalent boards share info sets. This matches
     * Pluribus, which uses ~665M action sequences vs our previous ~1.3B
     * when keying on actual dealt cards.
     * suit_map[actual_suit] = canonical_suit (-1 = unmapped). Built when
     * the flop is canonicalized, applied to turn/river cards. */
    int canon_board[5];
    int num_canon_board;
    int suit_map[4];  /* actual suit -> canonical suit */

    int action_history[256];
    int history_len;
} TraversalState;

static float traverse(TraversalState *ts, int acting_order_idx,
                      const int *acting_order, int num_in_order);

static int count_active(const int *active, int n) {
    int c = 0;
    for (int i = 0; i < n; i++) if (active[i]) c++;
    return c;
}

static int next_active(const int *acting_order, int num_in_order,
                        const int *active, int np, int cur_idx) {
    for (int i = 1; i <= num_in_order; i++) {
        int idx = (cur_idx + i) % num_in_order;
        int p = acting_order[idx];
        if (p < np && active[p]) return idx;
    }
    return -1;
}

static int round_done(const int *bets, const int *active,
                      const int *has_acted, int np) {
    int mx = 0;
    for (int i = 0; i < np; i++)
        if (active[i] && bets[i] > mx) mx = bets[i];
    for (int i = 0; i < np; i++) {
        if (!active[i]) continue;
        if (!has_acted[i]) return 0;
        if (bets[i] != mx) return 0;
    }
    return 1;
}

/* N-player showdown: compare all active players' hands */
static float eval_showdown_n(BPSolver *s, const int *board,
                              int traverser, const int *active,
                              const int *sampled_hands, int pot,
                              const int *invested) {
    int NP = s->num_players;

    /* If the traverser folded, they can't win anything at showdown. */
    if (!active[traverser])
        return -(float)invested[traverser];

    int th = sampled_hands[traverser];
    int tc0 = s->hands[traverser][th][0];
    int tc1 = s->hands[traverser][th][1];

    if (card_in_set(tc0, board, 5) || card_in_set(tc1, board, 5))
        return 0;

    /* Evaluate all active players' hands */
    uint32_t strength[BP_MAX_PLAYERS];
    int valid[BP_MAX_PLAYERS];  /* 1 if hand is valid (no card conflicts) */
    int cards[7];
    cards[0] = board[0]; cards[1] = board[1]; cards[2] = board[2];
    cards[3] = board[3]; cards[4] = board[4];

    for (int p = 0; p < NP; p++) {
        valid[p] = 0;
        if (!active[p]) continue;
        int oh = sampled_hands[p];
        int oc0 = s->hands[p][oh][0], oc1 = s->hands[p][oh][1];
        if (p != traverser && (cards_conflict(tc0, tc1, oc0, oc1) ||
            card_in_set(oc0, board, 5) || card_in_set(oc1, board, 5)))
            continue;
        cards[5] = oc0; cards[6] = oc1;
        strength[p] = eval7(cards);
        valid[p] = 1;
    }
    /* Traverser is active (checked above), evaluate their hand */
    cards[5] = tc0; cards[6] = tc1;
    strength[traverser] = eval7(cards);
    valid[traverser] = 1;

    /* Side pot calculation: sort unique investment levels, compute each
     * pot layer, award to the best hand among eligible players.
     * Eligible = active AND invested >= this layer's threshold. */
    int levels[BP_MAX_PLAYERS];
    int n_levels = 0;
    for (int p = 0; p < NP; p++) {
        if (!active[p] || !valid[p]) continue;
        /* Insert invested[p] into sorted unique levels */
        int inv = invested[p];
        int found = 0;
        for (int i = 0; i < n_levels; i++)
            if (levels[i] == inv) { found = 1; break; }
        if (!found) {
            int pos = n_levels;
            for (int i = 0; i < n_levels; i++)
                if (inv < levels[i]) { pos = i; break; }
            for (int i = n_levels; i > pos; i--) levels[i] = levels[i-1];
            levels[pos] = inv;
            n_levels++;
        }
    }

    /* If all players invested equally (common case), skip side pot logic */
    if (n_levels <= 1) {
        int n_tied = 0;
        int trav_wins = 1;
        for (int p = 0; p < NP; p++) {
            if (!active[p] || !valid[p]) continue;
            if (strength[p] > strength[traverser]) { trav_wins = 0; }
            if (strength[p] == strength[traverser]) n_tied++;
        }
        if (!trav_wins)
            return -(float)invested[traverser];
        return (float)pot / (float)n_tied - (float)invested[traverser];
    }

    /* Side pots: for each investment level, compute the pot layer and
     * award it to the best hand among players who invested at least that much */
    float trav_winnings = 0;
    int prev_level = 0;
    for (int li = 0; li < n_levels; li++) {
        int level = levels[li];
        int layer_per_player = level - prev_level;
        if (layer_per_player <= 0) continue;

        /* Count contributors and find winner(s) for this layer */
        int layer_pot = 0;
        uint32_t best = 0;
        int n_eligible = 0;
        for (int p = 0; p < NP; p++) {
            /* All active players who invested >= level contribute */
            if (invested[p] >= level) {
                layer_pot += layer_per_player;
            }
            /* Only valid active players are eligible to win */
            if (active[p] && valid[p] && invested[p] >= level) {
                if (strength[p] > best) best = strength[p];
                n_eligible++;
            }
        }
        /* Also count folded players' contributions to this layer */
        for (int p = 0; p < NP; p++) {
            if (!active[p] && invested[p] > prev_level) {
                int contrib = invested[p] - prev_level;
                if (contrib > layer_per_player) contrib = layer_per_player;
                layer_pot += contrib;
            }
        }

        /* Award to winner(s) */
        if (n_eligible > 0 && strength[traverser] == best &&
            active[traverser] && valid[traverser] && invested[traverser] >= level) {
            int n_winners = 0;
            for (int p = 0; p < NP; p++)
                if (active[p] && valid[p] && invested[p] >= level && strength[p] == best)
                    n_winners++;
            trav_winnings += (float)layer_pot / (float)n_winners;
        }

        prev_level = level;
    }

    return trav_winnings - (float)invested[traverser];
}

/* ── Main traversal ──────────────────────────────────────────────── */

static float traverse(TraversalState *ts, int acting_order_idx,
                      const int *acting_order, int num_in_order) {
    BPSolver *s = ts->solver;
    int NP = s->num_players;
    int n_active = count_active(ts->active, NP);

    /* Terminal: all folded */
    if (n_active <= 1) {
        for (int p = 0; p < NP; p++) {
            if (ts->active[p]) {
                if (p == ts->traverser)
                    return (float)(ts->pot - ts->invested[ts->traverser]);
                else
                    return -(float)ts->invested[ts->traverser];
            }
        }
        return 0;
    }

    /* Round complete -> next street or showdown */
    if (round_done(ts->bets, ts->active, ts->has_acted, NP)) {
        if (ts->num_board >= 5) {
            return eval_showdown_n(s, ts->board, ts->traverser,
                                    ts->active, ts->sampled_hands,
                                    ts->pot, ts->invested);
        }

        /* Build blocked set: board cards + all active players' cards */
        int blocked[52] = {0};
        for (int b = 0; b < ts->num_board; b++) blocked[ts->board[b]] = 1;
        for (int p = 0; p < NP; p++) {
            if (!ts->active[p]) continue;
            int h = ts->sampled_hands[p];
            blocked[s->hands[p][h][0]] = 1;
            blocked[s->hands[p][h][1]] = 1;
        }

        int valid[52], nv = 0;
        for (int c = 0; c < 52; c++)
            if (!blocked[c]) valid[nv++] = c;
        if (nv == 0) return 0;

        TraversalState next = *ts;
        memset(next.bets, 0, sizeof(next.bets));
        memset(next.has_acted, 0, sizeof(next.has_acted));
        next.num_raises = 0;

        if (ts->num_board == 0) {
            /* Preflop -> Flop: deal 3 cards */
            if (nv < 3) return 0;
            partial_shuffle(valid, nv, 3, ts->rng);
            next.board[0] = valid[0];
            next.board[1] = valid[1];
            next.board[2] = valid[2];
            next.num_board = 3;

            /* Look up precomputed 200-bucket mapping for this flop texture.
             * The cache was built at init time for all 1,755 canonical textures.
             * The randomly dealt flop won't match canonical suits, so we
             * canonicalize first: sort ranks descending, assign canonical suits
             * matching the pattern (rainbow, monotone, flush draw, etc.).
             * Bucket assignment depends only on EHS which is suit-isomorphic,
             * so any texture with the same rank pattern + suit pattern gives
             * the same bucket mapping relative to hand indices. */
            if (s->num_cached_textures > 0) {
                /* Canonicalize: sort cards by rank descending */
                int sorted[3] = {next.board[0], next.board[1], next.board[2]};
                for (int a = 0; a < 2; a++)
                    for (int b = a+1; b < 3; b++)
                        if ((sorted[a]>>2) < (sorted[b]>>2)) {
                            int tmp = sorted[a]; sorted[a] = sorted[b]; sorted[b] = tmp;
                        }

                int r0 = sorted[0]>>2, r1 = sorted[1]>>2, r2 = sorted[2]>>2;
                int s0 = sorted[0]&3, s1 = sorted[1]&3, s2 = sorted[2]&3;

                /* Determine canonical suits based on pattern */
                int canon[3];
                if (r0 == r1 && r1 == r2) {
                    /* Trips: rainbow */
                    canon[0] = r0*4+3; canon[1] = r1*4+2; canon[2] = r2*4+1;
                } else if (r0 == r1 || r1 == r2) {
                    /* Paired board. Flush draw = kicker shares suit with a paired card.
                     * Paired cards always have different suits (same rank). */
                    if (r0 == r1) {
                        /* Cards 0,1 are the pair, card 2 is the kicker.
                         * Flush draw if kicker suit matches either paired card. */
                        if (s2 == s0 || s2 == s1) {
                            /* FD: kicker shares suit with a paired card.
                             * Canon: pair different suits (s,h), kicker shares with first (s). */
                            canon[0] = r0*4+3; canon[1] = r1*4+2; canon[2] = r2*4+3;
                        } else {
                            /* Rainbow: all three suits different */
                            canon[0] = r0*4+3; canon[1] = r1*4+2; canon[2] = r2*4+1;
                        }
                    } else {
                        /* r1 == r2: cards 1,2 are the pair, card 0 is the kicker.
                         * Flush draw if kicker suit matches either paired card. */
                        if (s0 == s1 || s0 == s2) {
                            /* FD: kicker shares suit with a paired card. */
                            canon[0] = r0*4+3; canon[1] = r1*4+3; canon[2] = r2*4+2;
                        } else {
                            /* Rainbow */
                            canon[0] = r0*4+3; canon[1] = r1*4+2; canon[2] = r2*4+1;
                        }
                    }
                } else {
                    /* Unpaired: check suit pattern */
                    if (s0 == s1 && s1 == s2) {
                        /* Monotone */
                        canon[0] = r0*4+3; canon[1] = r1*4+3; canon[2] = r2*4+3;
                    } else if (s0 == s1) {
                        canon[0] = r0*4+3; canon[1] = r1*4+3; canon[2] = r2*4+2;
                    } else if (s0 == s2) {
                        canon[0] = r0*4+3; canon[1] = r1*4+2; canon[2] = r2*4+3;
                    } else if (s1 == s2) {
                        canon[0] = r0*4+2; canon[1] = r1*4+3; canon[2] = r2*4+3;
                    } else {
                        /* Rainbow */
                        canon[0] = r0*4+3; canon[1] = r1*4+2; canon[2] = r2*4+1;
                    }
                }

                uint64_t flop_hash = compute_board_hash(canon, 3);
                int found = -1;
                for (int t = 0; t < s->num_cached_textures; t++) {
                    if (s->texture_hash_keys[t] == flop_hash) {
                        found = t;
                        break;
                    }
                }
                if (found >= 0) {
                    int *cache_row = &s->texture_bucket_cache[found * BP_MAX_HANDS];
                    memcpy(next.postflop_bucket, cache_row, BP_MAX_HANDS * sizeof(int));
                    next.postflop_num_buckets_actual = s->postflop_num_buckets;
                    next.postflop_buckets_computed = 1;
                }

                /* Store canonical flop and build suit mapping for turn/river.
                 * suit_map[actual_suit] = canonical_suit.
                 * We map each actual suit from sorted[] to its canonical suit
                 * from canon[]. Unmapped suits get the next available canonical
                 * suit (for turn/river cards with new suits). */
                next.canon_board[0] = canon[0];
                next.canon_board[1] = canon[1];
                next.canon_board[2] = canon[2];
                next.num_canon_board = 3;

                for (int i = 0; i < 4; i++) next.suit_map[i] = -1;
                for (int i = 0; i < 3; i++) {
                    int actual_suit = sorted[i] & 3;
                    int canon_suit = canon[i] & 3;
                    if (next.suit_map[actual_suit] == -1)
                        next.suit_map[actual_suit] = canon_suit;
                }
                /* Assign unmapped suits to remaining canonical suits */
                int next_canon = 0;
                for (int i = 0; i < 4; i++) {
                    if (next.suit_map[i] == -1) {
                        /* Find next unused canonical suit */
                        while (next_canon < 4) {
                            int used = 0;
                            for (int j = 0; j < 4; j++)
                                if (next.suit_map[j] == next_canon) { used = 1; break; }
                            if (!used) break;
                            next_canon++;
                        }
                        if (next_canon < 4)
                            next.suit_map[i] = next_canon++;
                        else
                            next.suit_map[i] = 0; /* fallback */
                    }
                }
            }

            /* Postflop acting order: SB(0), BB(1), UTG(2), ..., BTN(5) */
            int postflop_order[BP_MAX_PLAYERS];
            for (int i = 0; i < NP; i++) postflop_order[i] = i;
            return traverse(&next, 0, postflop_order, NP);
        } else {
            /* Flop->Turn or Turn->River: deal 1 card */
            int dealt = valid[rng_int(ts->rng, nv)];
            next.board[next.num_board++] = dealt;
            /* Canonicalize the dealt card using the flop's suit mapping */
            if (next.num_canon_board < 5) {
                int rank = dealt >> 2;
                int suit = dealt & 3;
                int canon_suit = (next.suit_map[suit] >= 0) ? next.suit_map[suit] : suit;
                next.canon_board[next.num_canon_board++] = rank * 4 + canon_suit;
            }
            return traverse(&next, 0, acting_order, num_in_order);
        }
    }

    /* Current player */
    int ap = acting_order[acting_order_idx];
    if (!ts->active[ap]) {
        int ni = next_active(acting_order, num_in_order, ts->active, NP, acting_order_idx);
        if (ni < 0) return 0;
        return traverse(ts, ni, acting_order, num_in_order);
    }

    /* Generate actions */
    int mx = 0;
    for (int p = 0; p < NP; p++)
        if (ts->active[p] && ts->bets[p] > mx) mx = ts->bets[p];
    int to_call = mx - ts->bets[ap];
    if (to_call < 0) to_call = 0;

    BPAction actions[BP_MAX_ACTIONS];
    int remaining_stack = ts->stacks[ap];
    /* Use preflop bet sizes when on preflop, postflop sizes otherwise.
     * Pluribus: preflop has up to 14 raise sizes (fine-grained),
     * postflop has 1-3 sizes for later rounds. */
    const float *cur_bet_sizes;
    int cur_num_bet_sizes;
    int max_raises = 3;
    if (ts->num_board == 0 && s->num_preflop_bet_sizes > 0) {
        /* Tiered preflop: different sizes per raise level (Pluribus-style).
         * Level 0 = open, 1 = 3-bet, 2 = 4-bet, 3 = 5-bet. */
        if (s->num_preflop_tiers > 0) {
            int level = ts->num_raises;
            if (level >= s->num_preflop_tiers) level = s->num_preflop_tiers - 1;
            cur_bet_sizes = s->preflop_tiered_sizes[level];
            cur_num_bet_sizes = s->num_preflop_tiered_sizes[level];
        } else {
            cur_bet_sizes = s->preflop_bet_sizes;
            cur_num_bet_sizes = s->num_preflop_bet_sizes;
        }
        max_raises = (s->preflop_max_raises > 0) ? s->preflop_max_raises : 4;
    } else if (ts->num_raises > 0 && s->num_subsequent_bet_sizes > 0) {
        /* Postflop subsequent raise: fewer sizes (Pluribus: {1x pot, all-in}) */
        cur_bet_sizes = s->subsequent_bet_sizes;
        cur_num_bet_sizes = s->num_subsequent_bet_sizes;
    } else {
        /* Postflop first raise: full sizes (Pluribus: {0.5x, 1x, all-in}) */
        cur_bet_sizes = s->bet_sizes;
        cur_num_bet_sizes = s->num_bet_sizes;
    }
    int na = generate_actions(actions, BP_MAX_ACTIONS, ts->pot, remaining_stack,
                              to_call, ts->num_raises, max_raises,
                              cur_bet_sizes, cur_num_bet_sizes);
    if (na == 0) return 0;

    /* Info set lookup — bucket is part of the key (Pluribus-style).
     * Pluribus uses separate 200-bucket abstractions per street. */
    int street = board_to_street(ts->num_board);
    int bucket;
    if (street == 0) {
        /* Preflop: 169 lossless classes */
        bucket = get_bucket(s, street, ap, ts->sampled_hands[ap]);
    } else if (street == 1 && ts->postflop_buckets_computed) {
        /* Flop: use precomputed k-means texture buckets */
        bucket = ts->postflop_bucket[ts->sampled_hands[ap]];
    } else if (street == 2 && ts->num_board == 4 && s->turn_centroids_k > 0) {
        /* Turn: compute [EHS, PPot, NPot] features for this hand, then
         * map to nearest precomputed k-means centroid. Matches Pluribus:
         * k-means on domain-specific features (Johanson 2013). */
        int hand[1][2];
        hand[0][0] = s->hands[ap][ts->sampled_hands[ap]][0];
        hand[0][1] = s->hands[ap][ts->sampled_hands[ap]][1];
        float feat[1][3];
        ca_compute_features(ts->board, 4, (const int(*)[2])hand, 1, 200, feat);
        bucket = ca_nearest_centroid(feat[0], (const float(*)[3])s->turn_centroids,
                                      s->turn_centroids_k);
    } else if (street >= 2 && ts->num_board >= 4) {
        /* River (or turn fallback): EHS percentile with 200 MC samples. */
        int h = ts->sampled_hands[ap];
        int c0 = s->hands[ap][h][0], c1 = s->hands[ap][h][1];
        int blk[52] = {0};
        for (int b = 0; b < ts->num_board; b++) blk[ts->board[b]] = 1;
        blk[c0] = 1; blk[c1] = 1;
        int av[52]; int nav = 0;
        for (int c = 0; c < 52; c++) if (!blk[c]) av[nav++] = c;
        int wins = 0, ties = 0, total = 0;
        uint64_t erng = (uint64_t)c0 * 1000003ULL + (uint64_t)c1 * 999983ULL;
        for (int b = 0; b < ts->num_board; b++)
            erng = erng * 6364136223846793005ULL + (uint64_t)ts->board[b];
        int cards_needed = 2 + (5 - ts->num_board);
        for (int si = 0; si < 200 && nav >= cards_needed; si++) {
            partial_shuffle(av, nav, cards_needed, &erng);
            int fb[5];
            for (int b = 0; b < ts->num_board; b++) fb[b] = ts->board[b];
            for (int b = ts->num_board; b < 5; b++)
                fb[b] = av[2 + (b - ts->num_board)];
            int h7[7] = {fb[0], fb[1], fb[2], fb[3], fb[4], c0, c1};
            int o7[7] = {fb[0], fb[1], fb[2], fb[3], fb[4], av[0], av[1]};
            uint32_t hs = eval7(h7), os = eval7(o7);
            if (hs > os) wins++; else if (hs == os) ties++;
            total++;
        }
        float ehs = (total > 0) ? ((float)wins + 0.5f*(float)ties) / (float)total : 0.5f;
        bucket = (int)(ehs * (float)s->postflop_num_buckets);
        if (bucket >= s->postflop_num_buckets) bucket = s->postflop_num_buckets - 1;
    } else {
        bucket = get_bucket(s, street, ap, ts->sampled_hands[ap]);
    }

    BPInfoKey key;
    key.player = ap;
    key.street = street;
    key.bucket = bucket;
    /* Board is NOT in the key — the bucket already abstracts the board's
     * strategic impact. Including board_hash would create separate info sets
     * for every canonical board, defeating the purpose of bucketing and
     * inflating the tree from ~665M (Pluribus) to billions.
     * Preflop: board_hash is constant (no board). Postflop: bucket captures
     * the hand+board situation via EHS/k-means clustering. */
    key.board_hash = 0;
    key.action_hash = compute_action_hash(ts->action_history, ts->history_len);

    int64_t is_slot = info_table_find_or_create(&s->info_table, key, na);
    __atomic_fetch_add(&g_lookup_total, 1, __ATOMIC_RELAXED);
    if (is_slot < 0) {
        __atomic_fetch_add(&g_lookup_miss, 1, __ATOMIC_RELAXED);
        return 0;
    }
    BPInfoSet *is = &s->info_table.sets[is_slot];

    /* Guard: if hash collision causes action count mismatch, use stored count
     * to prevent buffer overflow on regrets (arena) and strategy_sum (heap). */
    if (na != is->num_actions) {
        na = is->num_actions;
    }

    /* Regret matching */
    float strategy[BP_MAX_ACTIONS];
    regret_match(is->regrets, strategy, na);

    int next_order = next_active(acting_order, num_in_order, ts->active, NP, acting_order_idx);
    if (next_order < 0) next_order = acting_order_idx;

    if (ap == ts->traverser) {
        /* Traverser: explore all actions (with optional pruning).
         * Pluribus Algorithm 1 (TRAVERSE-MCCFR-P): only explored actions
         * get regret updates. Pruned actions are left unchanged. */
        float action_values[BP_MAX_ACTIONS];
        int explored[BP_MAX_ACTIONS];
        float node_value = 0;

        for (int a = 0; a < na; a++) {
            explored[a] = 0;
            /* Pruning: skip actions with very negative regret.
             * Pluribus exceptions: never prune on river (num_board >= 5),
             * never prune fold (leads to terminal node). */
            if (ts->use_pruning && is->regrets[a] < BP_PRUNE_THRESHOLD) {
                if (ts->num_board < 5 && actions[a].type != ACT_FOLD) {
                    action_values[a] = 0;
                    continue;
                }
            }

            explored[a] = 1;
            TraversalState child = *ts;
            child.action_history[child.history_len++] = a;

            if (actions[a].type == ACT_FOLD) {
                child.active[ap] = 0;
            } else if (actions[a].type == ACT_CHECK) {
                child.has_acted[ap] = 1;
            } else if (actions[a].type == ACT_CALL) {
                child.bets[ap] = mx;
                child.invested[ap] += to_call;
                child.stacks[ap] -= to_call;
                child.pot += to_call;
                child.has_acted[ap] = 1;
            } else {
                /* amount = total new chips committed this action.
                 * For a raise: amount includes to_call + raise.
                 * New street-bet level = current bet + amount. */
                int amount = actions[a].amount;
                /* Cap at remaining stack */
                if (amount > child.stacks[ap]) amount = child.stacks[ap];
                child.bets[ap] += amount;
                child.invested[ap] += amount;
                child.stacks[ap] -= amount;
                child.pot += amount;
                child.has_acted[ap] = 1;
                for (int p = 0; p < NP; p++)
                    if (p != ap && child.active[p]) child.has_acted[p] = 0;
                child.num_raises++;
            }

            action_values[a] = traverse(&child, next_order, acting_order, num_in_order);
            node_value += strategy[a] * action_values[a];
        }

        /* Update integer regrets for EXPLORED actions only (Hogwild: no lock).
         * Pluribus Algorithm 1: pruned actions' regrets are left unchanged.
         * Previously we updated all actions, giving pruned actions
         * delta = -node_value, pushing them deeper negative and preventing
         * recovery (the root cause of the call trap at early positions). */
        for (int a = 0; a < na; a++) {
            if (!explored[a]) continue;
            float raw_delta = action_values[a] - node_value;
            int delta;
            if (raw_delta > 2e9f) delta = (int)2e9;
            else if (raw_delta < -2e9f) delta = (int)-2e9;
            else delta = (int)raw_delta;
            int64_t tmp = (int64_t)is->regrets[a] + (int64_t)delta;
            if (tmp < BP_REGRET_FLOOR) is->regrets[a] = BP_REGRET_FLOOR;
            else if (tmp > BP_REGRET_CEILING) is->regrets[a] = BP_REGRET_CEILING;
            else is->regrets[a] = (int)tmp;
        }

        /* Strategy sum: accumulate for preflop (street 0) every 10007
         * iterations. Matches Pluribus: UPDATE-STRATEGY called every
         * Strategy Interval (10K) for first betting round only.
         * Using 10007 (prime) instead of 10000 to avoid aliasing with
         * traverser cycling: gcd(6, 10000)=2 caused only players 1,3,5
         * to accumulate, permanently excluding SB(0), UTG(2), CO(4).
         * gcd(6, 10007)=1, so all 6 players get equal accumulation. */
        if (street == 0 && (ts->iteration % 10007) == 0) {
            ensure_strategy_sum(is);
            for (int a = 0; a < na; a++)
                is->strategy_sum[a] += strategy[a];
        }

        return node_value;

    } else {
        /* Non-traverser: sample one action from current strategy.
         * Matches Pluribus Algorithm 1: opponents play according to σ(I). */
        int sampled = sample_action(strategy, na, ts->rng);

        TraversalState child = *ts;
        child.action_history[child.history_len++] = sampled;

        if (actions[sampled].type == ACT_FOLD) {
            child.active[ap] = 0;
        } else if (actions[sampled].type == ACT_CHECK) {
            child.has_acted[ap] = 1;
        } else if (actions[sampled].type == ACT_CALL) {
            child.bets[ap] = mx;
            child.invested[ap] += to_call;
            child.stacks[ap] -= to_call;
            child.pot += to_call;
            child.has_acted[ap] = 1;
        } else {
            int amount = actions[sampled].amount;
            if (amount > child.stacks[ap]) amount = child.stacks[ap];
            child.bets[ap] += amount;
            child.invested[ap] += amount;
            child.stacks[ap] -= amount;
            child.pot += amount;
            child.has_acted[ap] = 1;
            for (int p = 0; p < NP; p++)
                if (p != ap && child.active[p]) child.has_acted[p] = 0;
            child.num_raises++;
        }

        return traverse(&child, next_order, acting_order, num_in_order);
    }
}

/* ── Strategy snapshots (rounds 2-4, Pluribus-style) ─────────────── */

/* Pluribus accumulates strategy_sum only for round 1. For rounds 2-4, it
 * saves snapshots of the current strategy to disk periodically (every
 * ~200 minutes after 800 minutes). The blueprint for rounds 2-4 is the
 * average of these snapshots.
 *
 * We implement this by periodically adding the current regret-matched
 * strategy into strategy_sum for ALL streets (not just round 1) when
 * we're past the snapshot_start_iter threshold. This accumulates the
 * snapshot average directly in memory rather than saving to disk. */

static void accumulate_snapshot(BPInfoTable *t) {
    float strat_buf[BP_MAX_ACTIONS];
    for (int64_t i = 0; i < t->table_size; i++) {
        if (t->occupied[i] != 1) continue;
        BPInfoSet *is = &t->sets[i];
        int na = is->num_actions;

        if (!is->strategy_sum) {
            float *buf = (float*)arena_alloc(na);
            if (buf) {
                is->strategy_sum = buf;  /* Single-threaded (omp single), no CAS needed */
            }
        }

        regret_match(is->regrets, strat_buf, na);
        for (int a = 0; a < na; a++)
            is->strategy_sum[a] += strat_buf[a];
    }
}

/* ── Linear CFR discount ─────────────────────────────────────────── */

static void apply_discount(BPInfoTable *t, float discount) {
    for (int64_t i = 0; i < t->table_size; i++) {
        if (!t->occupied[i]) continue;
        BPInfoSet *is = &t->sets[i];
        int na = is->num_actions;
        for (int a = 0; a < na; a++) {
            is->regrets[a] = (int)((float)is->regrets[a] * discount);
            if (is->regrets[a] < BP_REGRET_FLOOR)
                is->regrets[a] = BP_REGRET_FLOOR;
        }
        if (is->strategy_sum) {
            for (int a = 0; a < na; a++)
                is->strategy_sum[a] *= discount;
        }
    }
}

/* ── Public API ──────────────────────────────────────────────────── */

void bp_default_config(BPConfig *config) {
    memset(config, 0, sizeof(BPConfig));
    /* Pluribus timing converted to iterations assuming ~1000 iter/min on 64 cores */
    config->discount_stop_iter = 400000;    /* 400 min * ~1000 iter/min */
    config->discount_interval  = 10000;     /* 10 min * ~1000 iter/min */
    config->prune_start_iter   = 200000;    /* 200 min */
    config->snapshot_start_iter = 800000;   /* 800 min */
    config->snapshot_interval  = 200000;    /* 200 min */
    config->strategy_interval  = 10000;     /* Pluribus: every 10K iterations */
    config->num_threads = 0;                /* auto */
    config->hash_table_size = 0;            /* auto */
    config->snapshot_dir = NULL;
}

int bp_init(BPSolver *s, int num_players,
            const int *flop,
            const int hands[][BP_MAX_HANDS][2],
            const float weights[][BP_MAX_HANDS],
            const int *num_hands,
            int starting_pot, int effective_stack,
            const float *bet_sizes, int num_bet_sizes) {
    BPConfig config;
    bp_default_config(&config);
    /* For backward compatibility: small hash table, single thread */
    config.hash_table_size = BP_HASH_SIZE_SMALL;
    config.num_threads = 1;
    return bp_init_ex(s, num_players, flop, hands, weights, num_hands,
                       starting_pot, effective_stack, bet_sizes, num_bet_sizes,
                       &config);
}

int bp_init_ex(BPSolver *s, int num_players,
                const int *flop,
                const int hands[][BP_MAX_HANDS][2],
                const float weights[][BP_MAX_HANDS],
                const int *num_hands,
                int starting_pot, int effective_stack,
                const float *bet_sizes, int num_bet_sizes,
                const BPConfig *config) {
    memset(s, 0, sizeof(BPSolver));
    s->num_players = num_players;
    memcpy(s->flop, flop, 3 * sizeof(int));
    s->config = *config;

    for (int p = 0; p < num_players; p++) {
        s->num_hands[p] = num_hands[p];
        for (int h = 0; h < num_hands[p]; h++) {
            s->hands[p][h][0] = hands[p][h][0];
            s->hands[p][h][1] = hands[p][h][1];
            s->weights[p][h] = weights[p][h];
        }
    }

    s->starting_pot = starting_pot;
    s->effective_stack = effective_stack;
    s->num_bet_sizes = num_bet_sizes;
    for (int i = 0; i < num_bet_sizes; i++)
        s->bet_sizes[i] = bet_sizes[i];

    /* Default: no card abstraction (identity mapping) */
    s->use_buckets = 0;
    for (int st = 0; st < 4; st++)
        for (int p = 0; p < num_players; p++) {
            s->num_buckets[st][p] = num_hands[p];
            for (int h = 0; h < num_hands[p]; h++)
                s->bucket_map[st][p][h] = h;
        }

    /* Hash table */
    int64_t ht_size = config->hash_table_size;
    if (ht_size <= 0)
        ht_size = (num_players > 2) ? (int64_t)BP_HASH_SIZE_MEDIUM : (int64_t)BP_HASH_SIZE_SMALL;
    info_table_init(&s->info_table, ht_size);

    /* RNG states — one per thread */
    int nt = config->num_threads;
    if (nt <= 0) {
        #ifdef _OPENMP
        nt = omp_get_max_threads();
        #else
        nt = 1;
        #endif
    }
    s->num_rng_states = nt;
    /* Allocate with 64-byte (cache-line) padding per RNG state to avoid
       false sharing between threads.  Each state is at index tid*8. */
    s->rng_states = (uint64_t*)malloc(nt * 8 * sizeof(uint64_t));
    memset(s->rng_states, 0, nt * 8 * sizeof(uint64_t));
    for (int i = 0; i < nt; i++)
        s->rng_states[i * 8] = 0xDEADBEEF12345678ULL + (uint64_t)i * 6364136223846793005ULL;

    return 0;
}

int bp_init_unified(BPSolver *s, int num_players,
                     int small_blind, int big_blind, int initial_stack,
                     const float *postflop_bet_sizes, int num_postflop_bet_sizes,
                     const float *preflop_bet_sizes, int num_preflop_bet_sizes,
                     const BPConfig *config) {
    memset(s, 0, sizeof(BPSolver));
    s->num_players = num_players;
    s->config = *config;
    s->config.include_preflop = 1;

    /* Blinds and stacks */
    s->small_blind = small_blind;
    s->big_blind = big_blind;
    s->initial_stack = initial_stack;
    /* starting_pot and effective_stack are computed per-iteration based on
     * who actually reaches the flop. For the init, set to full blind contributions. */
    s->starting_pot = small_blind + big_blind;
    s->effective_stack = initial_stack - big_blind; /* max chip cost so far */

    /* All players get all 1326 possible hands */
    int empty_board[3] = {0, 0, 0};
    int all_hands[BP_MAX_HANDS][2];

    /* Generate all C(52,2) = 1326 hands */
    int nh = 0;
    for (int c0 = 0; c0 < 52; c0++)
        for (int c1 = c0 + 1; c1 < 52; c1++) {
            if (nh >= BP_MAX_HANDS) break;
            all_hands[nh][0] = c0;
            all_hands[nh][1] = c1;
            nh++;
        }

    for (int p = 0; p < num_players; p++) {
        s->num_hands[p] = nh;
        for (int h = 0; h < nh; h++) {
            s->hands[p][h][0] = all_hands[h][0];
            s->hands[p][h][1] = all_hands[h][1];
            s->weights[p][h] = 1.0f;
        }
    }

    /* Bet sizes */
    s->num_bet_sizes = num_postflop_bet_sizes;
    for (int i = 0; i < num_postflop_bet_sizes && i < BP_MAX_ACTIONS; i++)
        s->bet_sizes[i] = postflop_bet_sizes[i];

    /* Postflop subsequent raises: {1x pot, all-in} per Pluribus */
    s->num_subsequent_bet_sizes = 1;
    s->subsequent_bet_sizes[0] = 1.0f; /* 1x pot; all-in is added automatically */

    s->num_preflop_bet_sizes = num_preflop_bet_sizes;
    for (int i = 0; i < num_preflop_bet_sizes && i < BP_MAX_ACTIONS; i++)
        s->preflop_bet_sizes[i] = preflop_bet_sizes[i];

    /* Tiered preflop: not set via bp_init_unified args, use bp_set_preflop_tiers() */
    s->num_preflop_tiers = 0;
    s->preflop_max_raises = 0;

    /* Default: 169 lossless preflop buckets (identity for postflop until set) */
    s->use_buckets = 1;

    /* Preflop: 169 lossless classes for all players.
     * 169 = 13 pairs + 78 suited + 78 offsuit.
     * class_id = f(rank_hi, rank_lo, suited). Same mapping as card_abstraction.c. */
    int hand_to_class[BP_MAX_HANDS];
    int class_map[13][13][2]; /* [rank_hi][rank_lo][suited] -> class_idx */
    memset(class_map, -1, sizeof(class_map));
    int n_classes = 0;
    for (int r0 = 12; r0 >= 0; r0--) {
        for (int r1 = r0; r1 >= 0; r1--) {
            if (r0 == r1) {
                class_map[r0][r1][0] = n_classes;
                class_map[r0][r1][1] = n_classes;
                n_classes++;
            } else {
                class_map[r0][r1][1] = n_classes; n_classes++; /* suited */
                class_map[r0][r1][0] = n_classes; n_classes++; /* offsuit */
            }
        }
    }
    for (int h = 0; h < nh; h++) {
        int r0 = all_hands[h][0] / 4;
        int r1 = all_hands[h][1] / 4;
        int suited = ((all_hands[h][0] % 4) == (all_hands[h][1] % 4)) ? 1 : 0;
        if (r0 < r1) { int tmp = r0; r0 = r1; r1 = tmp; }
        hand_to_class[h] = class_map[r0][r1][suited];
    }

    for (int p = 0; p < num_players; p++) {
        s->num_buckets[0][p] = n_classes; /* street 0 = preflop */
        for (int h = 0; h < nh; h++)
            s->bucket_map[0][p][h] = hand_to_class[h];
    }

    /* Postflop: identity mapping until bp_set_buckets is called */
    for (int st = 1; st <= 3; st++)
        for (int p = 0; p < num_players; p++) {
            s->num_buckets[st][p] = nh;
            for (int h = 0; h < nh; h++)
                s->bucket_map[st][p][h] = h;
        }

    /* Hash table — 3B default for tiered preflop (Pluribus: 665M action sequences) */
    int64_t ht_size = config->hash_table_size;
    if (ht_size <= 0)
        ht_size = (int64_t)BP_HASH_SIZE_LARGE;
    info_table_init(&s->info_table, ht_size);

    /* RNG states */
    int nt = config->num_threads;
    if (nt <= 0) {
        #ifdef _OPENMP
        nt = omp_get_max_threads();
        #else
        nt = 1;
        #endif
    }
    s->num_rng_states = nt;
    /* Allocate with 64-byte (cache-line) padding per RNG state to avoid
       false sharing between threads.  Each state is at index tid*8. */
    s->rng_states = (uint64_t*)malloc(nt * 8 * sizeof(uint64_t));
    memset(s->rng_states, 0, nt * 8 * sizeof(uint64_t));
    for (int i = 0; i < nt; i++)
        s->rng_states[i * 8] = 0xDEADBEEF12345678ULL + (uint64_t)i * 6364136223846793005ULL;

    /* Postflop bucketing: precompute 200-bucket EHS mapping for all 1,755 textures.
     * Pluribus precomputes card abstraction (bucket assignments) for all flop textures
     * once before training starts. We do the same here. */
    s->postflop_num_buckets = (config->postflop_num_buckets > 0) ? config->postflop_num_buckets : 200;
    s->postflop_ehs_samples = 500;  /* MC samples for precompute (Pluribus: ~500) */
    s->num_cached_textures = 0;
    s->texture_bucket_cache = (int*)calloc(BP_MAX_TEXTURES * (size_t)BP_MAX_HANDS, sizeof(int));

    printf("[BP] Unified init: %d players, blinds %d/%d, stack %d, "
           "preflop=%d sizes, postflop=%d sizes, hands=%d, "
           "preflop_buckets=%d, postflop_buckets=%d\n",
           num_players, small_blind, big_blind, initial_stack,
           num_preflop_bet_sizes, num_postflop_bet_sizes, nh,
           n_classes, s->postflop_num_buckets);
    /* Try loading precomputed texture cache from well-known paths */
    if (s->num_cached_textures == 0) {
        const char *cache_paths[] = {"/tmp/texture_cache.bin", "texture_cache.bin",
                                      "/opt/blueprint_unified/texture_cache.bin", NULL};
        for (int ci = 0; cache_paths[ci]; ci++) {
            if (bp_load_texture_cache(s, cache_paths[ci]) > 0) break;
        }
    }
    if (s->num_cached_textures > 0) {
        printf("[BP] Texture cache loaded (%d textures), skipping precompute\n",
               s->num_cached_textures);
    } else {
    printf("[BP] Precomputing postflop buckets for all flop textures...\n");
    fflush(stdout);

    {
        /* Generate all 1,755 unique flop textures (suit isomorphism).
         * Same logic as solve_scenarios.py:generate_all_textures(). */
        const char *RANKS_STR = "23456789TJQKA";
        int tex_count = 0;
        clock_t pc_start = clock();

        for (int r0 = 12; r0 >= 0; r0--) {
            for (int r1 = r0; r1 >= 0; r1--) {
                for (int r2 = r1; r2 >= 0; r2--) {
                    /* Determine suit variants */
                    int variants[5][3]; /* up to 5 suit patterns, each = [s0, s1, s2] */
                    int nv = 0;

                    if (r0 == r1 && r1 == r2) {
                        /* Trips: only rainbow */
                        variants[nv][0] = 3; variants[nv][1] = 2; variants[nv][2] = 1; nv++; /* s,h,d */
                    } else if (r0 == r1 || r1 == r2) {
                        /* Paired: rainbow and flush draw.
                         * Flush draw = kicker shares a suit with one paired card.
                         * Paired cards MUST have different suits (same rank).
                         * E.g., KsKh2s — 2s shares suit with Ks. */
                        variants[nv][0] = 3; variants[nv][1] = 2; variants[nv][2] = 1; nv++; /* rainbow: all different suits */
                        if (r0 == r1) {
                            /* Kicker (r2) shares suit with first paired card */
                            variants[nv][0] = 3; variants[nv][1] = 2; variants[nv][2] = 3; nv++; /* fd: r2 suit == r0 suit */
                        } else {
                            /* Kicker (r0) shares suit with second paired card (r1==r2 case) */
                            /* r1 and r2 have same rank, different suits. r0 shares suit with r1. */
                            variants[nv][0] = 3; variants[nv][1] = 3; variants[nv][2] = 2; nv++; /* fd: r0 suit == r1 suit */
                        }
                    } else {
                        /* Unpaired: rainbow, monotone, 3 flush draws */
                        variants[nv][0] = 3; variants[nv][1] = 2; variants[nv][2] = 1; nv++; /* r */
                        variants[nv][0] = 3; variants[nv][1] = 3; variants[nv][2] = 3; nv++; /* m */
                        variants[nv][0] = 3; variants[nv][1] = 3; variants[nv][2] = 2; nv++; /* fd12 */
                        variants[nv][0] = 3; variants[nv][1] = 2; variants[nv][2] = 3; nv++; /* fd13 */
                        variants[nv][0] = 2; variants[nv][1] = 3; variants[nv][2] = 3; nv++; /* fd23 */
                    }

                    for (int v = 0; v < nv && tex_count < BP_MAX_TEXTURES; v++) {
                        int board[3] = {
                            r0 * 4 + variants[v][0],
                            r1 * 4 + variants[v][1],
                            r2 * 4 + variants[v][2]
                        };

                        /* Generate hands for this flop */
                        int flop_blocked[52] = {0};
                        flop_blocked[board[0]] = 1;
                        flop_blocked[board[1]] = 1;
                        flop_blocked[board[2]] = 1;

                        int flop_hands[BP_MAX_HANDS][2];
                        int nh_flop = 0;
                        for (int c0 = 0; c0 < 52; c0++) {
                            if (flop_blocked[c0]) continue;
                            for (int c1 = c0 + 1; c1 < 52; c1++) {
                                if (flop_blocked[c1]) continue;
                                if (nh_flop >= BP_MAX_HANDS) goto precomp_hands_done;
                                flop_hands[nh_flop][0] = c0;
                                flop_hands[nh_flop][1] = c1;
                                nh_flop++;
                            }
                        }
                        precomp_hands_done:;

                        /* K-means bucketing on [EHS, positive_potential, negative_potential].
                         * Matches Pluribus: k-means on domain-specific features (Johanson 2013). */
                        int flop_buckets[BP_MAX_HANDS];
                        ca_assign_buckets_kmeans(board, 3,
                                                  (const int(*)[2])flop_hands, nh_flop,
                                                  s->postflop_num_buckets,
                                                  s->postflop_ehs_samples,
                                                  flop_buckets);

                        /* Store in cache: map flop-hand index to full 1326-hand index.
                         * Both enumerations use the same c0<c1 order, so we can
                         * just walk them in parallel. */
                        int *cache_row = &s->texture_bucket_cache[tex_count * BP_MAX_HANDS];
                        memset(cache_row, 0, BP_MAX_HANDS * sizeof(int));

                        int fi = 0;
                        for (int c0 = 0; c0 < 52 && fi < nh_flop; c0++) {
                            if (flop_blocked[c0]) continue;
                            for (int c1 = c0 + 1; c1 < 52 && fi < nh_flop; c1++) {
                                if (flop_blocked[c1]) continue;
                                /* Map (c0,c1) to full-hand index */
                                /* Full hand list: c0 in 0..51, c1 in c0+1..51
                                 * Index = sum(51-k for k=0..c0-1) + (c1 - c0 - 1)
                                 *       = c0*51 - c0*(c0-1)/2 + c1 - c0 - 1 */
                                int full_idx = c0 * 51 - c0 * (c0 - 1) / 2 + c1 - c0 - 1;
                                if (full_idx >= 0 && full_idx < BP_MAX_HANDS)
                                    cache_row[full_idx] = flop_buckets[fi];
                                fi++;
                            }
                        }

                        /* Compute canonical hash for this texture */
                        s->texture_hash_keys[tex_count] = compute_board_hash(board, 3);
                        tex_count++;
                    }
                }
            }
        }

        s->num_cached_textures = tex_count;
        double pc_elapsed = (double)(clock() - pc_start) / CLOCKS_PER_SEC;
        printf("[BP] Precomputed %d textures in %.1fs\n", tex_count, pc_elapsed);
    }
    } /* end else (skip if cache loaded) */

    /* Precompute turn k-means centroids.
     * Sample random turn boards, compute [EHS, PPot, NPot] features for all
     * valid hands, then run k-means to get 200 centroids. During traversal,
     * each hand's features are computed inline and mapped to nearest centroid.
     * This replaces floor(ehs * 200) percentile bucketing on the turn. */
    {
        printf("[BP] Precomputing turn k-means centroids...\n");
        clock_t tc_start = clock();

        int k = s->postflop_num_buckets;
        if (k > 200) k = 200;
        int n_turn_boards = 2000;
        int feat_samples = 200;

        /* Collect feature vectors from sampled turn boards */
        int max_feat = n_turn_boards * 1200;
        float (*all_feat)[3] = (float(*)[3])malloc((size_t)max_feat * 3 * sizeof(float));
        int n_feat = 0;

        uint64_t trng = 0xABCDEF0123456789ULL;
        for (int ti = 0; ti < n_turn_boards && n_feat < max_feat - 1200; ti++) {
            /* Sample random 4-card turn board */
            int deck[52];
            for (int c = 0; c < 52; c++) deck[c] = c;
            for (int i = 0; i < 4; i++) {
                int j = i + (int)(rng_next(&trng) % (uint64_t)(52 - i));
                int tmp = deck[i]; deck[i] = deck[j]; deck[j] = tmp;
            }
            int board[4] = {deck[0], deck[1], deck[2], deck[3]};

            /* Generate valid hands */
            int bblk[52] = {0};
            for (int b = 0; b < 4; b++) bblk[board[b]] = 1;
            int hands[BP_MAX_HANDS][2];
            int nh = 0;
            for (int c0 = 0; c0 < 52; c0++) {
                if (bblk[c0]) continue;
                for (int c1 = c0 + 1; c1 < 52; c1++) {
                    if (bblk[c1]) continue;
                    if (nh >= BP_MAX_HANDS) goto turn_hands_done;
                    hands[nh][0] = c0;
                    hands[nh][1] = c1;
                    nh++;
                }
            }
            turn_hands_done:;

            /* Compute [EHS, PPot, NPot] for all hands on this board */
            int batch = (nh > 300) ? 300 : nh;  /* subsample for speed */
            float feats[BP_MAX_HANDS][3];
            ca_compute_features(board, 4, (const int(*)[2])hands, batch,
                                feat_samples, feats);

            for (int h = 0; h < batch && n_feat < max_feat; h++) {
                all_feat[n_feat][0] = feats[h][0];
                all_feat[n_feat][1] = feats[h][1];
                all_feat[n_feat][2] = feats[h][2];
                n_feat++;
            }
        }

        /* Run k-means on all collected features.
         * Sort by EHS for percentile-seeded centroid initialization
         * (same method as ca_assign_buckets_kmeans). Without sorting,
         * seeds are effectively random → many empty clusters. */
        int *sorted_idx = (int*)malloc(n_feat * sizeof(int));
        for (int f = 0; f < n_feat; f++) sorted_idx[f] = f;
        /* Simple insertion sort on EHS (feature[0]) — n_feat is ~600K,
         * too large for insertion sort. Use qsort with a static array ref. */
        /* Store EHS + index pairs for sorting */
        typedef struct { float ehs; int idx; } EhsIdx;
        EhsIdx *sort_buf = (EhsIdx*)malloc(n_feat * sizeof(EhsIdx));
        for (int f = 0; f < n_feat; f++) {
            sort_buf[f].ehs = all_feat[f][0];
            sort_buf[f].idx = f;
        }
        /* qsort comparator defined as nested function (GCC extension) or
         * use a simple shell sort for portability */
        for (int gap = n_feat / 2; gap > 0; gap /= 2)
            for (int i = gap; i < n_feat; i++) {
                EhsIdx tmp = sort_buf[i];
                int j = i;
                while (j >= gap && sort_buf[j - gap].ehs > tmp.ehs) {
                    sort_buf[j] = sort_buf[j - gap];
                    j -= gap;
                }
                sort_buf[j] = tmp;
            }

        float (*centroids)[3] = (float(*)[3])malloc((size_t)k * 3 * sizeof(float));
        for (int c = 0; c < k; c++) {
            int si = (int)((float)c / k * n_feat);
            if (si >= n_feat) si = n_feat - 1;
            int fi = sort_buf[si].idx;
            centroids[c][0] = all_feat[fi][0];
            centroids[c][1] = all_feat[fi][1];
            centroids[c][2] = all_feat[fi][2];
        }
        free(sort_buf);
        free(sorted_idx);

        int *counts = (int*)calloc(k, sizeof(int));
        float (*sums)[3] = (float(*)[3])calloc((size_t)k * 3, sizeof(float));
        for (int iter = 0; iter < 20; iter++) {
            memset(counts, 0, k * sizeof(int));
            memset(sums, 0, (size_t)k * 3 * sizeof(float));
            for (int f = 0; f < n_feat; f++) {
                int best_c = 0;
                float best_d = 1e30f;
                for (int c = 0; c < k; c++) {
                    float d0 = all_feat[f][0] - centroids[c][0];
                    float d1 = all_feat[f][1] - centroids[c][1];
                    float d2 = all_feat[f][2] - centroids[c][2];
                    float d = d0*d0 + d1*d1 + d2*d2;
                    if (d < best_d) { best_d = d; best_c = c; }
                }
                counts[best_c]++;
                sums[best_c][0] += all_feat[f][0];
                sums[best_c][1] += all_feat[f][1];
                sums[best_c][2] += all_feat[f][2];
            }
            for (int c = 0; c < k; c++) {
                if (counts[c] > 0) {
                    centroids[c][0] = sums[c][0] / counts[c];
                    centroids[c][1] = sums[c][1] / counts[c];
                    centroids[c][2] = sums[c][2] / counts[c];
                }
            }
        }

        int actual_k = 0;
        for (int c = 0; c < k; c++) {
            if (counts[c] > 0 && actual_k < 200) {
                s->turn_centroids[actual_k][0] = centroids[c][0];
                s->turn_centroids[actual_k][1] = centroids[c][1];
                s->turn_centroids[actual_k][2] = centroids[c][2];
                actual_k++;
            }
        }
        s->turn_centroids_k = actual_k;

        free(all_feat); free(centroids); free(counts); free(sums);
        double tc_elapsed = (double)(clock() - tc_start) / CLOCKS_PER_SEC;
        printf("[BP] Turn centroids: %d clusters from %d samples in %.1fs\n",
               actual_k, n_feat, tc_elapsed);
    }

    return 0;
}

int bp_set_buckets(BPSolver *s, int street,
                    const int bucket_map[][BP_MAX_HANDS],
                    const int *num_buckets) {
    if (street < 0 || street > 3) return -1;
    s->use_buckets = 1;
    for (int p = 0; p < s->num_players; p++) {
        s->num_buckets[street][p] = num_buckets[p];
        for (int h = 0; h < s->num_hands[p]; h++)
            s->bucket_map[street][p][h] = bucket_map[p][h];
    }
    return 0;
}

/* Set tiered preflop bet sizes (Pluribus-style: fewer sizes at higher raise levels).
 * level 0 = open raise, 1 = 3-bet, 2 = 4-bet, 3 = 5-bet.
 * Call once per level after bp_init_unified, before bp_solve. */
int bp_set_preflop_tier(BPSolver *s, int level,
                         const float *sizes, int num_sizes,
                         int max_raises) {
    if (level < 0 || level >= 4) return -1;
    if (num_sizes > BP_MAX_ACTIONS) num_sizes = BP_MAX_ACTIONS;
    for (int i = 0; i < num_sizes; i++)
        s->preflop_tiered_sizes[level][i] = sizes[i];
    s->num_preflop_tiered_sizes[level] = num_sizes;
    if (level + 1 > s->num_preflop_tiers)
        s->num_preflop_tiers = level + 1;
    if (max_raises > 0)
        s->preflop_max_raises = max_raises;
    printf("[BP] Preflop tier %d: %d sizes [", level, num_sizes);
    for (int i = 0; i < num_sizes; i++)
        printf("%s%.2f", i ? ", " : "", sizes[i]);
    printf("], max_raises=%d\n", s->preflop_max_raises);
    return 0;
}

int bp_solve(BPSolver *s, int64_t max_iterations) {
    int NP = s->num_players;
    int acting_order[BP_MAX_PLAYERS];
    for (int i = 0; i < NP; i++) acting_order[i] = i;

    int nt = s->num_rng_states;
    #ifdef _OPENMP
    if (nt > 1) {
        omp_set_num_threads(nt);
    }
    #endif

    printf("[BP] Starting %d-player MCCFR: %lld iterations, %d threads, "
           "hash=%lld, buckets=%s\n",
           NP, (long long)max_iterations, nt, (long long)s->info_table.table_size,
           s->use_buckets ? "yes" : "no");

    #ifdef _OPENMP
    double t_start = omp_get_wtime();
    #else
    clock_t t_start = clock();
    #endif

    /* Incremental Linear CFR discount + strategy snapshots.
     *
     * Pluribus applies d = (T/interval)/(T/interval+1) every discount_interval
     * iterations for the first discount_stop_iter iterations. It also saves
     * strategy snapshots for rounds 2-4 after snapshot_start_iter.
     *
     * We use a single parallel region with explicit barriers between batches
     * to avoid repeated thread creation/teardown overhead. Thread 0 handles
     * discount and snapshot operations during the barrier. */
    /* Resume support: offset all iteration counters by previously completed iters.
     * After bp_load_regrets, iterations_run = saved_iters, so global_iter tracks
     * the cumulative total across checkpoint/resume cycles. */
    int64_t iter_offset = s->iterations_run;

    int64_t discount_count = 0;
    int64_t next_discount_at = s->config.discount_interval;
    /* Fast-forward discount_count to match resumed state */
    if (iter_offset > 0 && s->config.discount_interval > 0) {
        discount_count = iter_offset / s->config.discount_interval;
        next_discount_at = (discount_count + 1) * s->config.discount_interval;
    }
    int64_t batch_size = s->config.discount_interval;
    if (batch_size <= 0 || batch_size > max_iterations) batch_size = max_iterations;
    int64_t num_batches = (max_iterations + batch_size - 1) / batch_size;

    #ifdef _OPENMP
    #pragma omp parallel if(nt > 1)
    #endif
    {
        #ifdef _OPENMP
        int tid = omp_get_thread_num();
        #else
        int tid = 0;
        #endif
        uint64_t *my_rng = &s->rng_states[tid * 8];

        for (int64_t batch = 0; batch < num_batches; batch++) {
            int64_t batch_start = batch * batch_size + 1;
            int64_t batch_end = batch_start + batch_size - 1;
            if (batch_end > max_iterations) batch_end = max_iterations;

            /* All threads work on this batch's iterations */
            #ifdef _OPENMP
            #pragma omp for schedule(dynamic, 64)
            #endif
            for (int64_t iter = batch_start; iter <= batch_end; iter++) {
                int64_t global_iter = iter + iter_offset;
                int traverser = (int)((global_iter - 1) % NP);

                int use_pruning = 0;
                if (global_iter > s->config.prune_start_iter) {
                    use_pruning = (rng_float(my_rng) < BP_PRUNE_PROB) ? 1 : 0;
                }

                int sampled_hands[BP_MAX_PLAYERS];
                for (int p = 0; p < NP; p++)
                    sampled_hands[p] = rng_int(my_rng, s->num_hands[p]);

                int conflict = 0;
                for (int p = 0; p < NP && !conflict; p++) {
                    int c0 = s->hands[p][sampled_hands[p]][0];
                    int c1 = s->hands[p][sampled_hands[p]][1];
                    /* Board conflict only applies when starting from flop */
                    if (!s->config.include_preflop) {
                        if (card_in_set(c0, s->flop, 3) || card_in_set(c1, s->flop, 3))
                            { conflict = 1; break; }
                    }
                    /* Player-player conflict (always check) */
                    for (int q = 0; q < p; q++) {
                        int d0 = s->hands[q][sampled_hands[q]][0];
                        int d1 = s->hands[q][sampled_hands[q]][1];
                        if (cards_conflict(c0, c1, d0, d1)) { conflict = 1; break; }
                    }
                }
                if (conflict) continue;

                TraversalState ts;
                memset(&ts, 0, sizeof(ts));
                ts.solver = s;
                ts.rng = my_rng;
                ts.traverser = traverser;
                ts.iteration = global_iter;
                ts.use_pruning = use_pruning;
                memcpy(ts.sampled_hands, sampled_hands, sizeof(sampled_hands));

                if (s->config.include_preflop) {
                    /* Unified: start from preflop, no board cards yet.
                     * SB (player 0) and BB (player 1) post blinds.
                     * Preflop acting order: UTG(2), MP(3), CO(4), BTN(5), SB(0), BB(1)
                     * The blinds are forced bets — already in the pot. */
                    ts.num_board = 0;
                    for (int p = 0; p < NP; p++) ts.active[p] = 1;
                    ts.num_active = NP;
                    ts.pot = s->small_blind + s->big_blind;

                    for (int p = 0; p < NP; p++) {
                        ts.stacks[p] = s->initial_stack;
                        ts.invested[p] = 0;
                        ts.bets[p] = 0;
                        ts.has_acted[p] = 0;
                    }
                    /* SB posts small blind */
                    if (NP > 0) {
                        ts.bets[0] = s->small_blind;
                        ts.invested[0] = s->small_blind;
                        ts.stacks[0] -= s->small_blind;
                    }
                    /* BB posts big blind */
                    if (NP > 1) {
                        ts.bets[1] = s->big_blind;
                        ts.invested[1] = s->big_blind;
                        ts.stacks[1] -= s->big_blind;
                    }
                    /* BB has NOT acted yet (still needs to close the action).
                     * All other players have NOT acted. The blinds are forced
                     * bets, not voluntary actions. */
                    /* Preflop acting order: UTG first (player 2 in 6-max) */
                    int preflop_order[BP_MAX_PLAYERS];
                    for (int i = 0; i < NP; i++)
                        preflop_order[i] = (i + 2) % NP; /* UTG=2, MP=3, ..., SB=0, BB=1 */

                    traverse(&ts, 0, preflop_order, NP);
                } else {
                    /* Postflop-only: start from flop (backward compat) */
                    memcpy(ts.board, s->flop, 3 * sizeof(int));
                    ts.num_board = 3;
                    for (int p = 0; p < NP; p++) ts.active[p] = 1;
                    ts.num_active = NP;
                    ts.pot = s->starting_pot;
                    for (int p = 0; p < NP; p++) {
                        ts.stacks[p] = s->effective_stack;
                        ts.invested[p] = s->starting_pot / NP;
                    }
                    traverse(&ts, 0, acting_order, NP);
                }

                if (tid == 0) {
                    s->iterations_run = global_iter;
                    if (iter % 10000 == 0 || iter == 1 || iter == max_iterations) {
                        #ifdef _OPENMP
                        double elapsed = omp_get_wtime() - t_start;
                        #else
                        double elapsed = (double)(clock() - t_start) / CLOCKS_PER_SEC;
                        #endif
                        printf("[BP] iter %lld/%lld (global %lld), info sets: %lld, %.1fs\n",
                               (long long)iter, (long long)max_iterations, (long long)global_iter, (long long)s->info_table.num_entries, elapsed);
                        fflush(stdout);
                    }
                }
            }
            /* implicit barrier after omp for */

            /* Thread 0 applies discount and snapshots between batches.
             * Other threads wait at the barrier below. */
            #ifdef _OPENMP
            #pragma omp single
            #endif
            {
                /* Linear CFR discount: d = T/(T+1) every interval */
                int64_t global_batch_end = (int64_t)batch_end + iter_offset;
                if (global_batch_end <= s->config.discount_stop_iter &&
                    global_batch_end >= next_discount_at) {
                    discount_count++;
                    float t_val = (float)discount_count;
                    float discount = t_val / (t_val + 1.0f);
                    apply_discount(&s->info_table, discount);
                    next_discount_at = (discount_count + 1) * s->config.discount_interval;
                    printf("[BP] Applied Linear CFR discount #%lld: d=%.4f at iter %lld\n",
                           (long long)discount_count, discount, (long long)global_batch_end);
                }

                /* Strategy snapshots for rounds 2-4 */
                if (global_batch_end >= s->config.snapshot_start_iter &&
                    s->config.snapshot_interval > 0 &&
                    (global_batch_end % s->config.snapshot_interval) < batch_size) {
                    accumulate_snapshot(&s->info_table);
                    s->snapshots_saved++;
                    printf("[BP] Accumulated strategy snapshot #%lld at iter %lld\n",
                           (long long)s->snapshots_saved, (long long)global_batch_end);
                }
            }
            /* implicit barrier after omp single */
        }
    } /* end parallel */

    #ifdef _OPENMP
    double total_time = omp_get_wtime() - t_start;
    #else
    double total_time = (double)(clock() - t_start) / CLOCKS_PER_SEC;
    #endif
    printf("[BP] Done: %lld iterations, %lld info sets, %.1fs (%.0f iter/s)\n",
           (long long)max_iterations, (long long)s->info_table.num_entries, total_time,
           (double)max_iterations / total_time);

    return 0;
}

int bp_get_strategy(const BPSolver *s, int player,
                     const int *board, int num_board,
                     const int *action_seq, int seq_len,
                     float *strategy_out, int bucket) {
    BPInfoKey key;
    key.player = player;
    key.street = board_to_street(num_board);
    key.bucket = bucket;
    key.board_hash = 0;  /* Board abstracted by bucket, not in key */
    key.action_hash = compute_action_hash(action_seq, seq_len);

    uint64_t h = hash_combine(key.board_hash, key.action_hash);
    h = hash_combine(h, (uint64_t)key.player);
    h = hash_combine(h, (uint64_t)key.street);
    h = hash_combine(h, (uint64_t)key.bucket);
    int64_t slot = (int64_t)(h % (uint64_t)s->info_table.table_size);

    for (int probe = 0; probe < 1024; probe++) {
        int64_t idx = (slot + probe) % s->info_table.table_size;
        if (!s->info_table.occupied[idx]) return 0;
        if (key_eq(&s->info_table.keys[idx], &key)) {
            BPInfoSet *is = &s->info_table.sets[idx];
            int na = is->num_actions;

            if (is->strategy_sum) {
                float sum = 0;
                for (int a = 0; a < na; a++) {
                    float v = is->strategy_sum[a];
                    v = v > 0 ? v : 0;
                    strategy_out[a] = v;
                    sum += v;
                }
                if (sum > 0) {
                    for (int a = 0; a < na; a++) strategy_out[a] /= sum;
                } else {
                    for (int a = 0; a < na; a++) strategy_out[a] = 1.0f / na;
                }
            } else {
                regret_match(is->regrets, strategy_out, na);
            }
            return na;
        }
    }
    return 0;
}

int bp_get_regrets(const BPSolver *s, int player,
                    const int *board, int num_board,
                    const int *action_seq, int seq_len,
                    int *regrets_out, int bucket) {
    BPInfoKey key;
    key.player = player;
    key.street = board_to_street(num_board);
    key.bucket = bucket;
    key.board_hash = 0;
    key.action_hash = compute_action_hash(action_seq, seq_len);

    uint64_t h = hash_combine(key.board_hash, key.action_hash);
    h = hash_combine(h, (uint64_t)key.player);
    h = hash_combine(h, (uint64_t)key.street);
    h = hash_combine(h, (uint64_t)key.bucket);
    int64_t slot = (int64_t)(h % (uint64_t)s->info_table.table_size);

    for (int probe = 0; probe < 1024; probe++) {
        int64_t idx = (slot + probe) % s->info_table.table_size;
        if (!s->info_table.occupied[idx]) return 0;
        if (key_eq(&s->info_table.keys[idx], &key)) {
            BPInfoSet *is = &s->info_table.sets[idx];
            int na = is->num_actions;
            for (int a = 0; a < na; a++)
                regrets_out[a] = is->regrets[a];
            return na;
        }
    }
    return 0;
}

int64_t bp_num_info_sets(const BPSolver *s) {
    return s->info_table.num_entries;
}

int bp_save_regrets(const BPSolver *s, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;

    const BPInfoTable *t = &s->info_table;

    /* Header: "BPR4" = bucket-in-key format v4 (int64 table_size, int64 iterations) */
    fwrite("BPR4", 1, 4, f);
    fwrite(&t->table_size, sizeof(int64_t), 1, f);
    fwrite(&t->num_entries, sizeof(int64_t), 1, f);
    int64_t iters64 = s->iterations_run;
    fwrite(&iters64, sizeof(int64_t), 1, f);

    /* Entries — each info set has bucket in key, regrets[num_actions] only */
    int64_t written = 0;
    for (int64_t i = 0; i < t->table_size; i++) {
        if (t->occupied[i] != 1) continue;
        BPInfoKey *key = &t->keys[i];
        BPInfoSet *is = &t->sets[i];

        fwrite(&key->player, sizeof(int), 1, f);
        fwrite(&key->street, sizeof(int), 1, f);
        fwrite(&key->bucket, sizeof(int), 1, f);
        fwrite(&key->board_hash, sizeof(uint64_t), 1, f);
        fwrite(&key->action_hash, sizeof(uint64_t), 1, f);
        fwrite(&is->num_actions, sizeof(int), 1, f);

        fwrite(is->regrets, sizeof(int), is->num_actions, f);

        int has_ss = (is->strategy_sum != NULL) ? 1 : 0;
        fwrite(&has_ss, sizeof(int), 1, f);
        if (has_ss) {
            fwrite(is->strategy_sum, sizeof(float), is->num_actions, f);
        }
        written++;
    }

    fclose(f);
    printf("[BP] Saved %lld info sets to %s\n", (long long)written, path);
    return 0;
}

int64_t bp_load_regrets(BPSolver *s, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;

    /* Header */
    char magic[4];
    fread(magic, 1, 4, f);
    int is_v4 = (memcmp(magic, "BPR4", 4) == 0);
    int is_v3 = (memcmp(magic, "BPR3", 4) == 0);
    int is_v2 = (memcmp(magic, "BPR2", 4) == 0);
    if (!is_v4 && !is_v3 && !is_v2) {
        printf("[BP] ERROR: expected BPR2/BPR3/BPR4 format, got %.4s\n", magic);
        fclose(f);
        return -1;
    }

    int64_t saved_entries;
    int64_t saved_iters;
    if (is_v4) {
        int64_t saved_table_size64;
        fread(&saved_table_size64, sizeof(int64_t), 1, f);
        fread(&saved_entries, sizeof(int64_t), 1, f);
        fread(&saved_iters, sizeof(int64_t), 1, f);
    } else {
        int saved_table_size32, saved_entries32;
        fread(&saved_table_size32, sizeof(int), 1, f);
        fread(&saved_entries32, sizeof(int), 1, f);
        saved_entries = (int64_t)saved_entries32;
        if (is_v3) {
            fread(&saved_iters, sizeof(int64_t), 1, f);
        } else {
            int iters32; fread(&iters32, sizeof(int), 1, f);
            saved_iters = (int64_t)iters32;
        }
    }

    printf("[BP] Loading checkpoint: %lld info sets, %lld iterations\n",
           (long long)saved_entries, (long long)saved_iters);

    BPInfoTable *t = &s->info_table;
    int64_t loaded = 0;
    int64_t merged = 0;

    int tmp_regrets[BP_MAX_ACTIONS];
    float tmp_ss[BP_MAX_ACTIONS];

    for (int64_t e = 0; e < saved_entries; e++) {
        BPInfoKey key;
        int na;

        if (fread(&key.player, sizeof(int), 1, f) != 1) break;
        if (fread(&key.street, sizeof(int), 1, f) != 1) break;
        if (fread(&key.bucket, sizeof(int), 1, f) != 1) break;
        if (fread(&key.board_hash, sizeof(uint64_t), 1, f) != 1) break;
        if (fread(&key.action_hash, sizeof(uint64_t), 1, f) != 1) break;
        if (fread(&na, sizeof(int), 1, f) != 1) break;
        if (na > BP_MAX_ACTIONS) na = BP_MAX_ACTIONS;

        int64_t slot = info_table_find_or_create(t, key, na);
        if (slot < 0) {
            fseek(f, na * sizeof(int), SEEK_CUR);
            int has_ss;
            fread(&has_ss, sizeof(int), 1, f);
            if (has_ss) fseek(f, na * sizeof(float), SEEK_CUR);
            continue;
        }

        BPInfoSet *is = &t->sets[slot];

        /* Read regrets into temp buffer, then ADD to slot.
         * For fresh slots (arena zeroed): 0 + new = new (normal load).
         * For duplicate keys (Hogwild race bug): existing + new = merged.
         * This recovers split regrets from the duplicate-key bug where
         * two hash table entries accumulated partial regrets separately. */
        fread(tmp_regrets, sizeof(int), na, f);
        int is_dup = 0;
        for (int a = 0; a < na; a++) {
            if (is->regrets[a] != 0) { is_dup = 1; break; }
        }
        for (int a = 0; a < na; a++) {
            is->regrets[a] += tmp_regrets[a];
        }

        int has_ss;
        fread(&has_ss, sizeof(int), 1, f);
        if (has_ss) {
            if (!is->strategy_sum) {
                is->strategy_sum = (float*)arena_alloc(na);
            }
            if (is->strategy_sum) {
                fread(tmp_ss, sizeof(float), na, f);
                for (int a = 0; a < na; a++)
                    is->strategy_sum[a] += tmp_ss[a];
            } else {
                fseek(f, na * sizeof(float), SEEK_CUR);
            }
        }

        if (is_dup) {
            merged++;
            if (merged <= 50) {
                printf("[BP] Merged duplicate: player=%d street=%d bucket=%d "
                       "ah=%016llx\n", key.player, key.street, key.bucket,
                       (unsigned long long)key.action_hash);
            }
        }
        loaded++;
    }

    s->iterations_run = saved_iters;
    fclose(f);
    if (merged > 0) {
        printf("[BP] WARNING: merged %lld duplicate entries (Hogwild race bug). "
               "Split regrets have been recovered.\n", (long long)merged);
    }
    printf("[BP] Loaded %lld/%lld info sets (%lld merged), table %lld/%lld\n",
           (long long)loaded, (long long)saved_entries, (long long)merged,
           (long long)t->num_entries, (long long)t->table_size);
    return loaded;
}

int bp_save_texture_cache(const BPSolver *s, const char *path) {
    if (!s->texture_bucket_cache || s->num_cached_textures == 0) return -1;
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    fwrite("TXC1", 1, 4, f);
    fwrite(&s->num_cached_textures, sizeof(int), 1, f);
    fwrite(&s->postflop_num_buckets, sizeof(int), 1, f);
    fwrite(s->texture_hash_keys, sizeof(uint64_t), s->num_cached_textures, f);
    fwrite(s->texture_bucket_cache, sizeof(int),
           (size_t)s->num_cached_textures * BP_MAX_HANDS, f);
    fclose(f);
    printf("[BP] Saved texture cache: %d textures to %s\n",
           s->num_cached_textures, path);
    return 0;
}

int bp_load_texture_cache(BPSolver *s, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    char magic[4];
    fread(magic, 1, 4, f);
    if (memcmp(magic, "TXC1", 4) != 0) { fclose(f); return -1; }
    int num_tex, num_buckets;
    fread(&num_tex, sizeof(int), 1, f);
    fread(&num_buckets, sizeof(int), 1, f);
    if (num_tex > BP_MAX_TEXTURES || num_buckets != s->postflop_num_buckets) {
        fclose(f); return -1;
    }
    if (!s->texture_bucket_cache)
        s->texture_bucket_cache = (int*)calloc(BP_MAX_TEXTURES * (size_t)BP_MAX_HANDS, sizeof(int));
    fread(s->texture_hash_keys, sizeof(uint64_t), num_tex, f);
    fread(s->texture_bucket_cache, sizeof(int), (size_t)num_tex * BP_MAX_HANDS, f);
    s->num_cached_textures = num_tex;
    fclose(f);
    printf("[BP] Loaded texture cache: %d textures from %s\n", num_tex, path);
    return num_tex;
}

void bp_free(BPSolver *s) {
    info_table_free(&s->info_table);
    if (s->rng_states) free(s->rng_states);
    if (s->texture_bucket_cache) free(s->texture_bucket_cache);
    memset(s, 0, sizeof(BPSolver));
}

/* ── Full strategy export ────────────────────────────────────────── */

int bp_export_strategies(const BPSolver *s,
                          unsigned char *buf, size_t buf_size,
                          size_t *bytes_written) {
    const BPInfoTable *t = &s->info_table;

    /* First pass: compute required size */
    size_t total = 0;
    int count = 0;
    /* Header: 4 bytes magic + 4 bytes num_entries + 4 bytes num_players */
    total += 12;
    for (int64_t i = 0; i < t->table_size; i++) {
        if (t->occupied[i] != 1) continue;
        BPInfoSet *is = &t->sets[i];
        /* Key: player(1) + street(1) + bucket(2) + board_hash(8) + action_hash(8) = 20 bytes */
        /* Meta: num_actions(1) = 1 byte */
        /* Strategy: num_actions * 1 byte (uint8 quantized) */
        total += 20 + 1 + is->num_actions;
        count++;
    }

    *bytes_written = total;
    if (buf == NULL) return 0;
    if (buf_size < total) return -1;

    unsigned char *p = buf;

    memcpy(p, "BPS3", 4); p += 4;  /* v3: bucket-in-key format */
    memcpy(p, &count, 4); p += 4;
    int np = s->num_players;
    memcpy(p, &np, 4); p += 4;

    float strategy_buf[BP_MAX_ACTIONS];
    for (int64_t i = 0; i < t->table_size; i++) {
        if (t->occupied[i] != 1) continue;
        BPInfoKey *key = &t->keys[i];
        BPInfoSet *is = &t->sets[i];
        int na = is->num_actions;

        unsigned char player_byte = (unsigned char)key->player;
        unsigned char street_byte = (unsigned char)key->street;
        unsigned short bucket_short = (unsigned short)key->bucket;
        memcpy(p, &player_byte, 1); p += 1;
        memcpy(p, &street_byte, 1); p += 1;
        memcpy(p, &bucket_short, 2); p += 2;
        memcpy(p, &key->board_hash, 8); p += 8;
        memcpy(p, &key->action_hash, 8); p += 8;

        unsigned char na_byte = (unsigned char)na;
        memcpy(p, &na_byte, 1); p += 1;

        regret_match(is->regrets, strategy_buf, na);
        for (int a = 0; a < na; a++) {
            int q = (int)(strategy_buf[a] * 255.0f + 0.5f);
            if (q < 0) q = 0;
            if (q > 255) q = 255;
            *p++ = (unsigned char)q;
        }
    }

    return 0;
}

int bp_export_buckets(const BPSolver *s, int street, int player,
                       int *bucket_out) {
    if (street < 0 || street > 3 || player < 0 || player >= s->num_players)
        return 0;
    int nh = s->num_hands[player];
    for (int h = 0; h < nh; h++)
        bucket_out[h] = s->bucket_map[street][player][h];
    return nh;
}
