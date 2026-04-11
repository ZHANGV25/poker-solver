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
#include <errno.h>
#ifndef _WIN32
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sched.h>
#else
/* Windows/MSYS2 compatibility shims. The solver is Linux-first (training
 * runs on EC2) but Windows builds are needed for local tests and dev
 * iteration. These shims give us the few POSIX calls we actually use;
 * madvise is guarded separately by MADV_* ifdefs at each call site. */
#include <windows.h>
static inline void sched_yield(void) { SwitchToThread(); }
static inline long sysconf(int name) {
    (void)name;
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return (long)si.dwPageSize;
}
#define _SC_PAGESIZE 0
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

/* Lookup miss stats — stubbed out to avoid atomic contention on a single
 * global cache line (192 threads hammering the same int64_t at 17K iter/s
 * was creating ~300M atomic ops/sec on one cache line, saturating coherence). */
void bp_get_miss_stats(int64_t *total, int64_t *miss) {
    *total = 0; *miss = 0;
}
void bp_reset_miss_stats(void) { }

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

/* Per-thread arena slices: each thread holds a 1MB local slice and serves
 * sub-allocations without atomics. When the slice runs out, it grabs a
 * new slice from the global arena via ONE atomic op.
 *
 * Before: every arena_alloc was an atomic_fetch_add on chunk->used. With
 * 192 threads doing 10M+ allocations during snapshot-time strategy_sum
 * allocation, the cache line bounced 10M+ times across cores → 10+
 * minutes of contention per snapshot.
 *
 * After: each thread atomically reserves a 1MB slice (~16K allocations
 * worth of 64-byte blocks). Per-thread overhead drops 16000x.
 *
 * The slice is also pre-faulted (madvise POPULATE_WRITE) when it spans
 * new pages, eliminating the page-fault storm during snapshots. */
#define TLS_SLICE_SIZE (1024 * 1024)  /* 1MB per thread-local slice */

static int g_arena_lock = 0;
static __thread char *tls_arena_ptr = NULL;
static __thread char *tls_arena_end = NULL;

/* Reserve a fresh slice from the global arena. Allocates a new chunk
 * if needed. Returns 0 on success, -1 on OOM. */
static int arena_grab_slice(int min_size) {
    int slice_size = TLS_SLICE_SIZE;
    if (min_size > slice_size) slice_size = min_size;
    /* Round to 64 bytes for cache-line alignment of the slice itself */
    slice_size = (slice_size + 63) & ~63;

retry_global:;
    ArenaChunk *chunk = g_arena.head;
    if (chunk) {
        int old = __atomic_fetch_add(&chunk->used, slice_size, __ATOMIC_RELAXED);
        if (old + slice_size <= chunk->capacity) {
            tls_arena_ptr = chunk->data + old;
            tls_arena_end = tls_arena_ptr + slice_size;
            return 0;
        }
        /* Chunk full — fall through to allocate new one */
    }

    /* Need a new chunk — take the spinlock */
    int expected = 0;
    if (!__atomic_compare_exchange_n(&g_arena_lock, &expected, 1,
                                     0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
        for (int spins = 0; __atomic_load_n(&g_arena_lock, __ATOMIC_ACQUIRE); spins++) {
            #ifdef __x86_64__
            __builtin_ia32_pause();
            #endif
            if (spins > 10000) { sched_yield(); spins = 0; }
        }
        goto retry_global;
    }

    /* Double-check */
    chunk = g_arena.head;
    if (chunk && chunk->used + slice_size <= chunk->capacity) {
        __atomic_store_n(&g_arena_lock, 0, __ATOMIC_RELEASE);
        goto retry_global;
    }

    /* Allocate a new chunk */
    ArenaChunk *c = (ArenaChunk*)malloc(sizeof(ArenaChunk));
    if (!c) { __atomic_store_n(&g_arena_lock, 0, __ATOMIC_RELEASE); return -1; }
    c->data = (char*)calloc(1, ARENA_CHUNK_SIZE);
    if (!c->data) {
        free(c);
        __atomic_store_n(&g_arena_lock, 0, __ATOMIC_RELEASE);
        return -1;
    }
    c->capacity = ARENA_CHUNK_SIZE;
    c->used = 0;
    c->next = g_arena.head;
    g_arena.head = c;
    g_arena.total_chunks++;

    /* Pre-fault the new chunk's pages so future per-iteration writes
     * don't trigger mm_lock contention via page faults. madvise is fast
     * (~10ms for 64MB) and eliminates the per-snapshot fault storm. */
    #ifdef MADV_POPULATE_WRITE
    madvise(c->data, ARENA_CHUNK_SIZE, MADV_POPULATE_WRITE);
    #endif
    #ifdef MADV_HUGEPAGE
    madvise(c->data, ARENA_CHUNK_SIZE, MADV_HUGEPAGE);
    #endif

    __atomic_store_n(&g_arena_lock, 0, __ATOMIC_RELEASE);
    goto retry_global;
}

static void *arena_alloc(int num_ints) {
    int size = num_ints * (int)sizeof(int);
    if (size < ARENA_BLOCK_SIZE) size = ARENA_BLOCK_SIZE;
    /* Round up to 8-byte alignment */
    size = (size + 7) & ~7;

    /* Fast path: serve from thread-local slice without any atomics. */
    if (tls_arena_ptr && tls_arena_ptr + size <= tls_arena_end) {
        void *ptr = tls_arena_ptr;
        tls_arena_ptr += size;
        return ptr;
    }

    /* Slow path: grab a new slice (one atomic op per ~16K allocations). */
    if (arena_grab_slice(size) != 0) return NULL;
    void *ptr = tls_arena_ptr;
    tls_arena_ptr += size;
    return ptr;
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
    /* Note: TLS pointers in worker threads still point into freed memory.
     * They'll be reset on the next arena_grab_slice() call. Safe as long
     * as no thread does arena_alloc between free_all and the next solve. */
    tls_arena_ptr = NULL;
    tls_arena_end = NULL;
}

/* ── Hash table ───────────────────────────────────────────────────── */

/* Hash table probe caps. INSERT and READ MUST be equal — making them differ
 * silently corrupts data: an entry placed at insert-distance > READ cap becomes
 * invisible to readers but still occupies a slot. Both at 4096 means the table
 * is correctly observable up to ~99% load (above which insertion_failures fires
 * and we know to bump table_size in the next run). */
#define HASH_PROBE_LIMIT_INSERT 4096
#define HASH_PROBE_LIMIT_READ   4096

/* splitmix64 — high-quality 64-bit avalanche mixer used throughout the
 * solver's hash functions. Defined here (rather than later) so hash_combine
 * and friends can use it. Same mixer as the texture lookup hashmap. */
static inline uint64_t bp_mix64(uint64_t x) {
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27; x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

/* Legacy-mixer mode for loading and re-querying v2 checkpoints that were
 * trained with the boost-style hash_combine. Set to 1 via bp_set_legacy_hash_mixer()
 * BEFORE calling bp_load_regrets on a v2 checkpoint. The v2 training run
 * (2026-04-07 22:18 UTC) launched BEFORE commit 48da71b (2026-04-08 01:51 UTC)
 * so the 1.5B checkpoint's action_hash values are derived from the old mixer.
 * Phase 1.3's traverse_ev re-queries by key, so compute_action_hash MUST
 * produce values matching what training stored, or every non-root lookup fails.
 *
 * IMPORTANT — separation of concerns:
 *   - compute_action_hash VALUE is what gets stored in the key field and must
 *     match training byte-for-byte. Controlled by g_legacy_hash_mixer.
 *   - Slot derivation (info_table_find_or_create and traverse_ev probe) uses
 *     hash_combine on the composite key (board_hash, action_hash, player,
 *     street, bucket). This is purely for placement and only needs to be
 *     internally consistent within a single process run. Always use splitmix64
 *     here regardless of the legacy flag, because boost-style slot derivation
 *     on the v2 checkpoint produces catastrophic probe-chain clustering
 *     (20K insertion failures observed, ~2900x slower lookups than splitmix64).
 *     First attempt at this fix tied both together and made Phase 1.3 EV walk
 *     ~200x slower than needed — killed after 6.5 min with no iter progress.
 */
static int g_legacy_hash_mixer = 0;

void bp_set_legacy_hash_mixer(int enabled) {
    g_legacy_hash_mixer = enabled ? 1 : 0;
    printf("[BP] Action-hash mixer mode: %s (slot-derivation mixer: splitmix64 always)\n",
           g_legacy_hash_mixer ? "legacy (boost)" : "splitmix64");
    fflush(stdout);
}

/* Bug 6 fix: splitmix64 hash_combine for slot derivation in the main hash
 * table. Used by info_table_find_or_create and traverse_ev probe loops.
 * Always splitmix64 regardless of legacy flag — see comment above.
 *
 * The previous boost::hash_combine
 *     a ^= b + 0x9e3779b97f4a7c15 + (a << 6) + (a >> 2);
 * has known weak distribution on small structured inputs. splitmix64 gives
 * ~uniform distribution over uint64. */
static inline uint64_t hash_combine(uint64_t a, uint64_t b) {
    return bp_mix64(a ^ bp_mix64(b));
}

/* Legacy boost-style combiner, used ONLY by compute_action_hash when
 * g_legacy_hash_mixer is enabled. Reproduces the exact bits that v2 training
 * wrote into key.action_hash, so lookups on a v2 checkpoint find the right
 * entries. Do NOT call this from anywhere else. */
static inline uint64_t hash_combine_boost_legacy(uint64_t a, uint64_t b) {
    a ^= b + 0x9e3779b97f4a7c15ULL + (a << 6) + (a >> 2);
    return a;
}

static uint64_t compute_board_hash(const int *board, int num_board) {
    uint64_t h = 0x123456789ABCDEFULL;
    if (g_legacy_hash_mixer) {
        /* Match the boost-style hash values stored in texture_cache.bin
         * (created 2026-04-07 before commit 48da71b). This ensures
         * texture_index_lookup() finds the right precomputed bucket row. */
        for (int i = 0; i < num_board; i++)
            h = hash_combine_boost_legacy(h, (uint64_t)board[i] * 31 + 7);
    } else {
        for (int i = 0; i < num_board; i++)
            h = hash_combine(h, (uint64_t)board[i] * 31 + 7);
    }
    return h;
}

/* Bug 7 fix: O(1) texture lookup via open-addressing hashmap.
 *
 * Replaces the previous O(N) linear scan over `s->num_cached_textures`
 * (~1755 entries per scan, fired on every flop deal during traversal). At
 * 30K iter/s × 192 threads × ~1 flop deal/iter, the scan was costing
 * ~5-15% of solver throughput.
 *
 * The hashmap lives in `s->texture_hash_index[BP_TEXTURE_INDEX_SIZE]` and
 * stores indices into `s->texture_hash_keys[]`. -1 means empty.
 *
 * Hash mixer: bp_mix64 (splitmix64) — same mixer used by hash_combine
 * for the main hash table. Much better distribution on small integer
 * inputs than the previous boost::hash_combine. */

/* Look up a flop_hash in the texture hashmap. Returns the index into
 * texture_hash_keys[] / texture_bucket_cache rows, or -1 if not found. */
static inline int texture_index_lookup(const BPSolver *s, uint64_t flop_hash) {
    uint64_t mixed = bp_mix64(flop_hash);
    int idx = (int)(mixed & (uint64_t)(BP_TEXTURE_INDEX_SIZE - 1));
    /* Linear probe. With BP_MAX_TEXTURES=1760 entries in BP_TEXTURE_INDEX_SIZE=4096
     * slots (~43% load), expected probe length is ~1.4. Probe up to 32 slots
     * to bound the worst case. */
    for (int p = 0; p < 32; p++) {
        int slot = (idx + p) & (BP_TEXTURE_INDEX_SIZE - 1);
        int t = s->texture_hash_index[slot];
        if (t < 0) return -1;
        if (s->texture_hash_keys[t] == flop_hash) return t;
    }
    return -1;
}

/* Insert a (flop_hash, texture_index) pair into the hashmap. */
static inline void texture_index_insert(BPSolver *s, uint64_t flop_hash, int texture_idx) {
    uint64_t mixed = bp_mix64(flop_hash);
    int idx = (int)(mixed & (uint64_t)(BP_TEXTURE_INDEX_SIZE - 1));
    for (int p = 0; p < 32; p++) {
        int slot = (idx + p) & (BP_TEXTURE_INDEX_SIZE - 1);
        if (s->texture_hash_index[slot] < 0) {
            s->texture_hash_index[slot] = texture_idx;
            return;
        }
    }
    /* Should never reach here at our load factor (≤43% with 32 probe slots
     * and a uniform splitmix mixer, the probability of failing all 32 probes
     * is astronomically small). If we do, the texture is silently dropped
     * from the index and the lookup will return -1, falling back to the
     * caller's "not found" path. */
}

/* Rebuild the texture hashmap from `texture_hash_keys[0..num_cached_textures]`.
 * Called after the texture cache is precomputed or loaded from disk. */
static void texture_index_rebuild(BPSolver *s) {
    for (int i = 0; i < BP_TEXTURE_INDEX_SIZE; i++) {
        s->texture_hash_index[i] = -1;
    }
    for (int t = 0; t < s->num_cached_textures; t++) {
        texture_index_insert(s, s->texture_hash_keys[t], t);
    }
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
    if (g_legacy_hash_mixer) {
        /* v2 checkpoint mode: reproduce the byte-exact action_hash values
         * that training stored using boost-style hash_combine. */
        for (int i = 0; i < num_actions; i++)
            h = hash_combine_boost_legacy(h, (uint64_t)actions[i] * 17 + 3);
    } else {
        /* v3+ mode: splitmix64 hash_combine. */
        for (int i = 0; i < num_actions; i++)
            h = hash_combine(h, (uint64_t)actions[i] * 17 + 3);
    }
    return h;
}

static void info_table_init(BPInfoTable *t, int64_t table_size) {
    t->table_size = table_size;
    t->keys = (BPInfoKey*)calloc((size_t)table_size, sizeof(BPInfoKey));
    t->sets = (BPInfoSet*)calloc((size_t)table_size, sizeof(BPInfoSet));
    t->occupied = (int*)calloc((size_t)table_size, sizeof(int));
    t->num_entries = 0;
    t->insertion_failures = 0;
    t->max_probe_observed = 0;
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
            sched_yield();
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

    for (int probe = 0; probe < HASH_PROBE_LIMIT_INSERT; probe++) {
        int64_t idx = (slot + probe) % t->table_size;
        int state = __atomic_load_n(&t->occupied[idx], __ATOMIC_ACQUIRE);

        if (state == 1) {
            if (key_eq(&t->keys[idx], &key)) {
                /* Track max probe distance even on existing-key hits, since
                 * later reads will need to walk this same distance. */
                if (probe > 0) {
                    int64_t cur = __atomic_load_n(&t->max_probe_observed, __ATOMIC_RELAXED);
                    while ((int64_t)probe > cur &&
                           !__atomic_compare_exchange_n(&t->max_probe_observed, &cur, (int64_t)probe,
                                                         1, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) { }
                }
                return idx;
            }
            continue;
        }

        if (state == 0) {
            int expected = 0;
            if (__atomic_compare_exchange_n(&t->occupied[idx], &expected, 2,
                                            0, __ATOMIC_ACQ_REL,
                                            __ATOMIC_ACQUIRE)) {
                t->sets[idx].num_actions = num_actions;
                t->sets[idx].regrets = (int*)arena_alloc(num_actions);
                /* Bug 9 fix: defensive NULL check. arena_alloc returns NULL
                 * on OOM. Without this check, the slot would be published
                 * with regrets=NULL and the next access (regret_match,
                 * regret update, etc.) would segfault. Roll back the slot
                 * state and report failure to the caller. The caller treats
                 * a -1 return as "this iteration's traversal cannot proceed"
                 * and skips the regret update — same handling as the
                 * probe-cap-exhausted case. */
                if (!t->sets[idx].regrets) {
                    __atomic_store_n(&t->occupied[idx], 0, __ATOMIC_RELEASE);
                    __atomic_fetch_add(&t->insertion_failures, 1, __ATOMIC_RELAXED);
                    return -1;
                }
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
                /* Track max probe distance for the successful insert. */
                if (probe > 0) {
                    int64_t cur = __atomic_load_n(&t->max_probe_observed, __ATOMIC_RELAXED);
                    while ((int64_t)probe > cur &&
                           !__atomic_compare_exchange_n(&t->max_probe_observed, &cur, (int64_t)probe,
                                                         1, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) { }
                }
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
    /* Probe cap exhausted — table is too full or pathologically clustered.
     * Caller will silently treat this as EV=0 which biases parent regrets.
     * Increment the failure counter so monitoring catches it. */
    __atomic_fetch_add(&t->insertion_failures, 1, __ATOMIC_RELAXED);
    /* Also record max probe = HASH_PROBE_LIMIT_INSERT for diagnostics. */
    {
        int64_t cur = __atomic_load_n(&t->max_probe_observed, __ATOMIC_RELAXED);
        while ((int64_t)HASH_PROBE_LIMIT_INSERT > cur &&
               !__atomic_compare_exchange_n(&t->max_probe_observed, &cur, (int64_t)HASH_PROBE_LIMIT_INSERT,
                                             1, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) { }
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

/* Phase 1.3: allocate action_evs accumulator for an info set. Lazy
 * allocation on first traverser visit during the EV walk. Arena-backed.
 * See docs/PHASE_1_3_DESIGN.md for the algorithm. */
static void ensure_action_evs(BPInfoSet *is) {
    if (__atomic_load_n((void**)&is->action_evs, __ATOMIC_ACQUIRE) == NULL) {
        float *buf = (float*)arena_alloc(is->num_actions);
        if (!buf) return;
        float *expected = NULL;
        if (!__atomic_compare_exchange_n(&is->action_evs, &expected, buf,
                                          0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
            /* CAS lost, buf wasted — same trade-off as strategy_sum. */
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

/* Phase 1.3: Extract the AVERAGE strategy σ̄ from strategy_sum (or fall
 * back to regret-matched σ if strategy_sum is NULL for this info set).
 * Mirrors the computation inside bp_get_strategy() but operates on a
 * BPInfoSet pointer directly (no hash lookup).
 *
 * This is the sampling distribution used by traverse_ev() at both
 * traverser and opponent decision nodes — we want to measure action EVs
 * under the blueprint's average strategy, not the current regret-matched
 * strategy that training was mid-optimizing. */
static void avg_strategy(const BPInfoSet *is, float *out, int num_actions) {
    float *ss = __atomic_load_n((float**)&is->strategy_sum, __ATOMIC_ACQUIRE);
    if (ss) {
        float sum = 0;
        for (int a = 0; a < num_actions; a++) {
            float v = ss[a];
            if (v < 0) v = 0;
            out[a] = v;
            sum += v;
        }
        if (sum > 0) {
            float inv = 1.0f / sum;
            for (int a = 0; a < num_actions; a++) out[a] *= inv;
            return;
        }
    }
    /* No strategy_sum or zero sum — fall back to regret matching.
     * This happens for info sets that were only visited during pruning or
     * never had strategy_sum accumulated. Best we can do. */
    regret_match(is->regrets, out, num_actions);
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

/* Flop bucket cache — hoisted out of TraversalState to avoid copying
 * 5.3 KB on every action expansion (`TraversalState child = *ts;`).
 * Allocated on the stack in the function that deals the flop, shared
 * via pointer with all descendant traversal states. */
typedef struct {
    int buckets[BP_MAX_HANDS];   /* hand_idx -> flop bucket */
    int num_buckets_actual;
    int computed;
} FlopBucketCache;

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

    /* Flop bucket cache — pointer to stack-allocated cache from the
     * function that dealt the flop. NULL before flop is dealt. */
    FlopBucketCache *flop_cache;

    /* Per-iteration turn/river bucket cache. Each iteration has a fixed
     * set of sampled hands, so (hand, board) -> bucket is invariant.
     * Cached at deal time to avoid the 200-sample Monte Carlo EHS
     * recompute on every info set lookup (was the dominant cost:
     * ~12,000 eval7 calls/iteration -> now ~12 per iteration).
     * -1 = not computed. Indexed by player. */
    int turn_bucket[BP_MAX_PLAYERS];
    int river_bucket[BP_MAX_PLAYERS];

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

        /* Flop bucket cache: allocated on this function's stack when the
         * flop is dealt. Lives until this function returns (which happens
         * AFTER all recursive children complete, since traverse is
         * recursive and returns from deepest node first). */
        FlopBucketCache flop_cache_local;

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
                /* Bug 7 fix: O(1) hashmap lookup instead of O(N) linear scan */
                int found = texture_index_lookup(s, flop_hash);
                if (found >= 0) {
                    int *cache_row = &s->texture_bucket_cache[found * BP_MAX_HANDS];
                    memcpy(flop_cache_local.buckets, cache_row, BP_MAX_HANDS * sizeof(int));
                    flop_cache_local.num_buckets_actual = s->postflop_num_buckets;
                    flop_cache_local.computed = 1;
                    next.flop_cache = &flop_cache_local;
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

            /* Pre-compute turn/river bucket cache for all active players
             * ONCE when the card is dealt. This replaces the per-info-set
             * 200-sample Monte Carlo EHS recompute that was the dominant
             * cost (~12,000 eval7 calls/iteration -> ~12). */
            if (next.num_board == 4 && s->turn_centroids_k > 0) {
                /* Turn: ca_compute_features + nearest centroid */
                for (int p = 0; p < NP; p++) {
                    next.turn_bucket[p] = -1;
                    if (!next.active[p]) continue;
                    int ph[1][2];
                    ph[0][0] = s->hands[p][next.sampled_hands[p]][0];
                    ph[0][1] = s->hands[p][next.sampled_hands[p]][1];
                    float feat[1][3];
                    ca_compute_features(next.board, 4, (const int(*)[2])ph, 1, 200, feat);
                    next.turn_bucket[p] = ca_nearest_centroid(
                        feat[0], (const float(*)[3])s->turn_centroids,
                        s->turn_centroids_k);
                }
            } else if (next.num_board == 5) {
                /* River: 200-sample EHS per player. This is the expensive
                 * version of the same compute that used to run per-info-set. */
                for (int p = 0; p < NP; p++) {
                    next.river_bucket[p] = -1;
                    if (!next.active[p]) continue;
                    int ph = next.sampled_hands[p];
                    int c0 = s->hands[p][ph][0], c1 = s->hands[p][ph][1];
                    int blk[52] = {0};
                    for (int b = 0; b < 5; b++) blk[next.board[b]] = 1;
                    blk[c0] = 1; blk[c1] = 1;
                    int av[52]; int nav = 0;
                    for (int c = 0; c < 52; c++) if (!blk[c]) av[nav++] = c;
                    int wins = 0, ties = 0, total = 0;
                    uint64_t erng = (uint64_t)c0 * 1000003ULL + (uint64_t)c1 * 999983ULL;
                    for (int b = 0; b < 5; b++)
                        erng = erng * 6364136223846793005ULL + (uint64_t)next.board[b];
                    for (int si = 0; si < 200 && nav >= 2; si++) {
                        partial_shuffle(av, nav, 2, &erng);
                        int h7[7] = {next.board[0], next.board[1], next.board[2],
                                     next.board[3], next.board[4], c0, c1};
                        int o7[7] = {next.board[0], next.board[1], next.board[2],
                                     next.board[3], next.board[4], av[0], av[1]};
                        uint32_t hs = eval7(h7), os = eval7(o7);
                        if (hs > os) wins++; else if (hs == os) ties++;
                        total++;
                    }
                    float ehs = (total > 0) ? ((float)wins + 0.5f*(float)ties) / (float)total : 0.5f;
                    int b = (int)(ehs * (float)s->postflop_num_buckets);
                    if (b >= s->postflop_num_buckets) b = s->postflop_num_buckets - 1;
                    next.river_bucket[p] = b;
                }
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
    } else if (street == 1 && ts->flop_cache && ts->flop_cache->computed) {
        /* Flop: use precomputed k-means texture buckets (cached in
         * FlopBucketCache on the stack of the function that dealt the flop). */
        bucket = ts->flop_cache->buckets[ts->sampled_hands[ap]];
    } else if (street == 2 && ts->num_board == 4 && s->turn_centroids_k > 0 &&
               ts->turn_bucket[ap] >= 0) {
        /* Turn: use pre-computed bucket from deal-time cache. */
        bucket = ts->turn_bucket[ap];
    } else if (street >= 2 && ts->num_board == 5 && ts->river_bucket[ap] >= 0) {
        /* River: use pre-computed bucket from deal-time cache. */
        bucket = ts->river_bucket[ap];
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
    if (is_slot < 0) {
        return 0;
    }
    BPInfoSet *is = &s->info_table.sets[is_slot];

    /* Bug 2 fix: clamp DOWN only, never up.
     *
     * If a hash collision causes the stored slot to have MORE actions than the
     * caller's local na, raising local na would cause subsequent code to
     * access actions[BP_MAX_ACTIONS] indices that generate_actions never
     * initialized — uninitialized stack data dispatched as if it were a
     * valid BPAction. The previous unconditional clamp set na = is->num_actions
     * unconditionally, exposing this bug on the rare hash collision case.
     *
     * Lowering only is always safe: the stack-allocated arrays (strategy[],
     * action_values[], explored[], actions[]) are sized BP_MAX_ACTIONS, and
     * generate_actions populated the first `local_na` entries. Reducing na
     * means we just iterate over fewer of the populated entries. */
    if (na > is->num_actions) {
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
                /* Bug 1 fix: cap call against remaining stack. In equal-stack
                 * 6-max with our betting tree this case is unreachable in
                 * practice (the BET branch caps bets at stack so to_call <=
                 * stacks[ap] always holds), but the call branch lacked the
                 * defensive cap. Adding it makes the call branch first-class
                 * bug-free regardless of upstream tree assumptions. */
                int call_amount = (to_call > child.stacks[ap]) ? child.stacks[ap] : to_call;
                child.bets[ap] += call_amount;
                child.invested[ap] += call_amount;
                child.stacks[ap] -= call_amount;
                child.pot += call_amount;
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

        /* Strategy sum: accumulate for preflop (street 0) on every traverser
         * visit. This matches Bug 6's documented fix in BLUEPRINT_BUGS.md:
         * "Remove the interval check entirely. Accumulate strategy_sum on
         * every traverser visit for preflop. This is cheap (preflop info
         * sets are tiny) and ensures all 6 players accumulate."
         *
         * The previous gate `% 10007` was a Bug 6 regression — re-added with
         * a coprime constant to fix the gcd(6, 10000)=2 aliasing while
         * preserving sparsity. Without the gate there is no aliasing issue
         * (every visit accumulates regardless of iteration parity), so all
         * 6 players accumulate equally. */
        if (street == 0) {
            ensure_strategy_sum(is);
            if (is->strategy_sum) {
                for (int a = 0; a < na; a++)
                    is->strategy_sum[a] += strategy[a];
            }
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
            /* Bug 1 fix: cap call against remaining stack. See traverser
             * branch above for the rationale. */
            int call_amount = (to_call > child.stacks[ap]) ? child.stacks[ap] : to_call;
            child.bets[ap] += call_amount;
            child.invested[ap] += call_amount;
            child.stacks[ap] -= call_amount;
            child.pot += call_amount;
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

/* ── Phase 1.3: σ̄-sampled EV walk ────────────────────────────────── */

/* traverse_ev() is a read-only sibling of traverse() that computes per-action
 * expected values under the average strategy σ̄. Changes from traverse():
 *   1. Strategies sourced from avg_strategy() (strategy_sum normalized, with
 *      regret_match fallback), not from regret_match on current regrets.
 *   2. At traverser nodes, exhaustively enumerates actions (same as traverse),
 *      but instead of updating regrets, accumulates action_values[a] into
 *      is->action_evs[a] and increments is->ev_visit_count.
 *   3. At opponent nodes, samples one action from σ̄ (external sampling).
 *   4. No regret updates, no strategy_sum updates. Read-only with respect
 *      to training state.
 *   5. Pruning DISABLED — we want to measure EVs for ALL actions, including
 *      those training pruned away (they may still matter for biased
 *      continuation strategies at depth-limited leaves).
 *
 * See docs/PHASE_1_3_DESIGN.md for the full math. */
static float traverse_ev(TraversalState *ts, int acting_order_idx,
                         const int *acting_order, int num_in_order);

static float traverse_ev(TraversalState *ts, int acting_order_idx,
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

        FlopBucketCache flop_cache_local;

        if (ts->num_board == 0) {
            /* Preflop -> Flop: deal 3 cards */
            if (nv < 3) return 0;
            partial_shuffle(valid, nv, 3, ts->rng);
            next.board[0] = valid[0];
            next.board[1] = valid[1];
            next.board[2] = valid[2];
            next.num_board = 3;

            /* Look up precomputed 200-bucket mapping for this flop texture
             * (same canonicalization logic as traverse() — see the extended
             * comment there for the reasoning). */
            if (s->num_cached_textures > 0) {
                int sorted[3] = {next.board[0], next.board[1], next.board[2]};
                for (int a = 0; a < 2; a++)
                    for (int b = a+1; b < 3; b++)
                        if ((sorted[a]>>2) < (sorted[b]>>2)) {
                            int tmp = sorted[a]; sorted[a] = sorted[b]; sorted[b] = tmp;
                        }

                int r0 = sorted[0]>>2, r1 = sorted[1]>>2, r2 = sorted[2]>>2;
                int s0 = sorted[0]&3, s1 = sorted[1]&3, s2 = sorted[2]&3;

                int canon[3];
                if (r0 == r1 && r1 == r2) {
                    canon[0] = r0*4+3; canon[1] = r1*4+2; canon[2] = r2*4+1;
                } else if (r0 == r1 || r1 == r2) {
                    if (r0 == r1) {
                        if (s2 == s0 || s2 == s1) {
                            canon[0] = r0*4+3; canon[1] = r1*4+2; canon[2] = r2*4+3;
                        } else {
                            canon[0] = r0*4+3; canon[1] = r1*4+2; canon[2] = r2*4+1;
                        }
                    } else {
                        if (s0 == s1 || s0 == s2) {
                            canon[0] = r0*4+3; canon[1] = r1*4+3; canon[2] = r2*4+2;
                        } else {
                            canon[0] = r0*4+3; canon[1] = r1*4+2; canon[2] = r2*4+1;
                        }
                    }
                } else {
                    if (s0 == s1 && s1 == s2) {
                        canon[0] = r0*4+3; canon[1] = r1*4+3; canon[2] = r2*4+3;
                    } else if (s0 == s1) {
                        canon[0] = r0*4+3; canon[1] = r1*4+3; canon[2] = r2*4+2;
                    } else if (s0 == s2) {
                        canon[0] = r0*4+3; canon[1] = r1*4+2; canon[2] = r2*4+3;
                    } else if (s1 == s2) {
                        canon[0] = r0*4+2; canon[1] = r1*4+3; canon[2] = r2*4+3;
                    } else {
                        canon[0] = r0*4+3; canon[1] = r1*4+2; canon[2] = r2*4+1;
                    }
                }

                uint64_t flop_hash = compute_board_hash(canon, 3);
                int found = texture_index_lookup(s, flop_hash);
                if (found >= 0) {
                    int *cache_row = &s->texture_bucket_cache[found * BP_MAX_HANDS];
                    memcpy(flop_cache_local.buckets, cache_row, BP_MAX_HANDS * sizeof(int));
                    flop_cache_local.num_buckets_actual = s->postflop_num_buckets;
                    flop_cache_local.computed = 1;
                    next.flop_cache = &flop_cache_local;
                }

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
                int next_canon = 0;
                for (int i = 0; i < 4; i++) {
                    if (next.suit_map[i] == -1) {
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
                            next.suit_map[i] = 0;
                    }
                }
            }

            int postflop_order[BP_MAX_PLAYERS];
            for (int i = 0; i < NP; i++) postflop_order[i] = i;
            return traverse_ev(&next, 0, postflop_order, NP);
        } else {
            /* Flop->Turn or Turn->River: deal 1 card */
            int dealt = valid[rng_int(ts->rng, nv)];
            next.board[next.num_board++] = dealt;
            if (next.num_canon_board < 5) {
                int rank = dealt >> 2;
                int suit = dealt & 3;
                int canon_suit = (suit >= 0 && suit < 4) ? next.suit_map[suit] : 0;
                if (canon_suit < 0) canon_suit = 0;
                next.canon_board[next.num_canon_board++] = rank * 4 + canon_suit;
            }

            /* Recompute turn/river bucket cache at deal time (same as traverse) */
            if (next.num_board == 4 && s->turn_centroids_k > 0) {
                for (int p = 0; p < NP; p++) {
                    next.turn_bucket[p] = -1;
                    if (!next.active[p]) continue;
                    int ph[1][2];
                    ph[0][0] = s->hands[p][next.sampled_hands[p]][0];
                    ph[0][1] = s->hands[p][next.sampled_hands[p]][1];
                    float feat[1][3];
                    ca_compute_features(next.board, 4, (const int(*)[2])ph, 1, 200, feat);
                    next.turn_bucket[p] = ca_nearest_centroid(
                        feat[0], (const float(*)[3])s->turn_centroids,
                        s->turn_centroids_k);
                }
            } else if (next.num_board == 5) {
                for (int p = 0; p < NP; p++) {
                    next.river_bucket[p] = -1;
                    if (!next.active[p]) continue;
                    int ph = next.sampled_hands[p];
                    int c0 = s->hands[p][ph][0], c1 = s->hands[p][ph][1];
                    int blk[52] = {0};
                    for (int b = 0; b < 5; b++) blk[next.board[b]] = 1;
                    blk[c0] = 1; blk[c1] = 1;
                    int av[52]; int nav = 0;
                    for (int c = 0; c < 52; c++) if (!blk[c]) av[nav++] = c;
                    int wins = 0, ties = 0, total = 0;
                    uint64_t erng = (uint64_t)c0 * 1000003ULL + (uint64_t)c1 * 999983ULL;
                    for (int b = 0; b < 5; b++)
                        erng = erng * 6364136223846793005ULL + (uint64_t)next.board[b];
                    for (int si = 0; si < 200 && nav >= 2; si++) {
                        partial_shuffle(av, nav, 2, &erng);
                        int h7[7] = {next.board[0], next.board[1], next.board[2],
                                     next.board[3], next.board[4], c0, c1};
                        int o7[7] = {next.board[0], next.board[1], next.board[2],
                                     next.board[3], next.board[4], av[0], av[1]};
                        uint32_t hs = eval7(h7), os = eval7(o7);
                        if (hs > os) wins++; else if (hs == os) ties++;
                        total++;
                    }
                    float ehs = (total > 0) ? ((float)wins + 0.5f*(float)ties) / (float)total : 0.5f;
                    int b = (int)(ehs * (float)s->postflop_num_buckets);
                    if (b >= s->postflop_num_buckets) b = s->postflop_num_buckets - 1;
                    next.river_bucket[p] = b;
                }
            }

            return traverse_ev(&next, 0, acting_order, num_in_order);
        }
    }

    /* Current player */
    int ap = acting_order[acting_order_idx];
    if (!ts->active[ap]) {
        int ni = next_active(acting_order, num_in_order, ts->active, NP, acting_order_idx);
        if (ni < 0) return 0;
        return traverse_ev(ts, ni, acting_order, num_in_order);
    }

    /* Generate actions (same as traverse) */
    int mx = 0;
    for (int p = 0; p < NP; p++)
        if (ts->active[p] && ts->bets[p] > mx) mx = ts->bets[p];
    int to_call = mx - ts->bets[ap];
    if (to_call < 0) to_call = 0;

    BPAction actions[BP_MAX_ACTIONS];
    int remaining_stack = ts->stacks[ap];
    const float *cur_bet_sizes;
    int cur_num_bet_sizes;
    int max_raises = 3;
    if (ts->num_board == 0 && s->num_preflop_bet_sizes > 0) {
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
        cur_bet_sizes = s->subsequent_bet_sizes;
        cur_num_bet_sizes = s->num_subsequent_bet_sizes;
    } else {
        cur_bet_sizes = s->bet_sizes;
        cur_num_bet_sizes = s->num_bet_sizes;
    }
    int na = generate_actions(actions, BP_MAX_ACTIONS, ts->pot, remaining_stack,
                              to_call, ts->num_raises, max_raises,
                              cur_bet_sizes, cur_num_bet_sizes);
    if (na == 0) return 0;

    /* Info set lookup — read-only. We do NOT create new info sets here;
     * if the info set wasn't visited during training, we skip it (return 0).
     * This is safe because the blueprint is the ground truth — we only
     * measure EVs against what training actually produced. */
    int street = board_to_street(ts->num_board);
    int bucket;
    if (street == 0) {
        bucket = get_bucket(s, street, ap, ts->sampled_hands[ap]);
    } else if (street == 1 && ts->flop_cache && ts->flop_cache->computed) {
        bucket = ts->flop_cache->buckets[ts->sampled_hands[ap]];
    } else if (street == 2 && ts->num_board == 4 && s->turn_centroids_k > 0 &&
               ts->turn_bucket[ap] >= 0) {
        bucket = ts->turn_bucket[ap];
    } else if (street >= 2 && ts->num_board == 5 && ts->river_bucket[ap] >= 0) {
        bucket = ts->river_bucket[ap];
    } else {
        bucket = get_bucket(s, street, ap, ts->sampled_hands[ap]);
    }

    BPInfoKey key;
    key.player = ap;
    key.street = street;
    key.bucket = bucket;
    key.board_hash = 0;
    key.action_hash = compute_action_hash(ts->action_history, ts->history_len);

    /* Read-only lookup — DO NOT create. Info sets that training never
     * visited are outside the blueprint's support and we have no data
     * for them. Return 0 (neutral EV) in that case. */
    uint64_t h = hash_combine(key.board_hash, key.action_hash);
    h = hash_combine(h, (uint64_t)key.player);
    h = hash_combine(h, (uint64_t)key.street);
    h = hash_combine(h, (uint64_t)key.bucket);
    int64_t slot = (int64_t)(h % (uint64_t)s->info_table.table_size);

    BPInfoSet *is = NULL;
    for (int probe = 0; probe < HASH_PROBE_LIMIT_READ; probe++) {
        int64_t idx = (slot + probe) % s->info_table.table_size;
        int state = __atomic_load_n(&s->info_table.occupied[idx], __ATOMIC_ACQUIRE);
        if (state == 0) break;
        if (state == 2) continue;  /* another thread initializing, skip */
        if (key_eq(&s->info_table.keys[idx], &key)) {
            is = &s->info_table.sets[idx];
            break;
        }
    }
    if (is == NULL) return 0;  /* info set not in blueprint — skip */

    /* Clamp down only — same safety as traverse(). */
    if (na > is->num_actions) {
        na = is->num_actions;
    }

    /* Sample strategy from σ̄ (average strategy), not regret-matched σ. */
    float strategy[BP_MAX_ACTIONS];
    avg_strategy(is, strategy, na);

    int next_order = next_active(acting_order, num_in_order, ts->active, NP, acting_order_idx);
    if (next_order < 0) next_order = acting_order_idx;

    if (ap == ts->traverser) {
        /* Traverser: exhaustively enumerate all actions (NO pruning — we
         * want EVs for everything, including pruned actions). For each
         * action, compute the child subtree EV and accumulate into
         * is->action_evs[a]. */
        float action_values[BP_MAX_ACTIONS];
        float node_value = 0;

        for (int a = 0; a < na; a++) {
            TraversalState child = *ts;
            child.action_history[child.history_len++] = a;

            if (actions[a].type == ACT_FOLD) {
                child.active[ap] = 0;
            } else if (actions[a].type == ACT_CHECK) {
                child.has_acted[ap] = 1;
            } else if (actions[a].type == ACT_CALL) {
                int call_amount = (to_call > child.stacks[ap]) ? child.stacks[ap] : to_call;
                child.bets[ap] += call_amount;
                child.invested[ap] += call_amount;
                child.stacks[ap] -= call_amount;
                child.pot += call_amount;
                child.has_acted[ap] = 1;
            } else {
                int amount = actions[a].amount;
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

            action_values[a] = traverse_ev(&child, next_order, acting_order, num_in_order);
            node_value += strategy[a] * action_values[a];
        }

        /* Accumulate per-action EVs for this info set. Hogwild-style: each
         * thread does atomic adds into the same accumulator. Per-action
         * EVs are float32 so we need a compare-and-swap loop (no atomic
         * float add in C11), OR we accept slight precision loss from
         * non-atomic += and rely on the averaging to smooth it out. The
         * existing regret updates use non-atomic += (Hogwild accepts this),
         * so we match that pattern. */
        ensure_action_evs(is);
        if (is->action_evs) {
            for (int a = 0; a < na; a++)
                is->action_evs[a] += action_values[a];
            __atomic_fetch_add(&is->ev_visit_count, 1, __ATOMIC_RELAXED);
        }

        return node_value;

    } else {
        /* Non-traverser: sample one action from σ̄. */
        int sampled = sample_action(strategy, na, ts->rng);

        TraversalState child = *ts;
        child.action_history[child.history_len++] = sampled;

        if (actions[sampled].type == ACT_FOLD) {
            child.active[ap] = 0;
        } else if (actions[sampled].type == ACT_CHECK) {
            child.has_acted[ap] = 1;
        } else if (actions[sampled].type == ACT_CALL) {
            int call_amount = (to_call > child.stacks[ap]) ? child.stacks[ap] : to_call;
            child.bets[ap] += call_amount;
            child.invested[ap] += call_amount;
            child.stacks[ap] -= call_amount;
            child.pot += call_amount;
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

        return traverse_ev(&child, next_order, acting_order, num_in_order);
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

/* Called from within an existing outer parallel region. Uses omp for to
 * distribute work across the team's threads. Does NOT use a new parallel
 * region (avoids nested parallelism issues). */
static void accumulate_snapshot(BPInfoTable *t) {
    float strat_buf[BP_MAX_ACTIONS];
    #pragma omp for schedule(static, 65536) nowait
    for (int64_t i = 0; i < t->table_size; i++) {
        if (t->occupied[i] != 1) continue;
        BPInfoSet *is = &t->sets[i];
        int na = is->num_actions;

        ensure_strategy_sum(is);
        float *ss = __atomic_load_n((float**)&is->strategy_sum, __ATOMIC_ACQUIRE);
        if (!ss) continue;

        regret_match(is->regrets, strat_buf, na);
        for (int a = 0; a < na; a++)
            ss[a] += strat_buf[a];
    }
}

/* ── Linear CFR discount ─────────────────────────────────────────── */

/* Called from within an existing outer parallel region. */
static void apply_discount(BPInfoTable *t, float discount) {
    #pragma omp for schedule(static, 65536) nowait
    for (int64_t i = 0; i < t->table_size; i++) {
        if (!t->occupied[i]) continue;
        BPInfoSet *is = &t->sets[i];
        int na = is->num_actions;
        for (int a = 0; a < na; a++) {
            is->regrets[a] = (int)((float)is->regrets[a] * discount);
            if (is->regrets[a] < BP_REGRET_FLOOR)
                is->regrets[a] = BP_REGRET_FLOOR;
        }
        /* F3 fix: only discount strategy_sum on round 1 (street 0). Per the
         * Pluribus paper (Supp. p. 14-15), Linear CFR discounts both regrets
         * AND average strategies during the discount phase, but the average
         * strategy that gets discounted is the round-1 average (phi). The
         * rounds 2-4 strategy_sum is constructed from snapshots taken AFTER
         * the discount phase ends, and Pluribus does NOT discount those.
         *
         * Currently this fix is defensive — discount_stop_iter (e.g. 35M)
         * is always less than snapshot_start_iter (e.g. 70M) under our
         * canonical Python config, so rounds 2-4 strategy_sum is NULL when
         * apply_discount fires and the original `if (is->strategy_sum)`
         * check skipped them. But that's a config-fragile invariant. Adding
         * an explicit street filter makes the discount semantics correct
         * regardless of how the timing parameters get tuned. */
        if (is->strategy_sum && t->keys[i].street == 0) {
            for (int a = 0; a < na; a++)
                is->strategy_sum[a] *= discount;
        }
    }
}

/* ── Public API ──────────────────────────────────────────────────── */

void bp_default_config(BPConfig *config) {
    memset(config, 0, sizeof(BPConfig));
    /* Pluribus timing as fractions of training duration. The previous defaults
     * were calibrated to "1000 iter/min" (Pluribus's hardware), which is ~30x
     * slower than ours and produces wildly wrong behavior on modern HW.
     *
     * These defaults assume a ~1B iteration baseline target, scaled from
     * Pluribus's actual wall-clock fractions:
     *   discount_stop:  3.47% (Pluribus 400 min / 11520 min)
     *   prune_start:    1.74% (Pluribus 200 min / 11520 min)
     *   snapshot_start: 6.94% (Pluribus 800 min / 11520 min)
     *   snapshot_interval: 1.74% (Pluribus 200 min / 11520 min)
     *
     * Source: pluribus_technical_details.md §1, supplementary materials of
     * Brown & Sandholm 2019.
     *
     * Production callers (precompute/blueprint_worker_unified.py) should
     * override these via the args.iterations target. These defaults are a
     * safe fallback for any caller that forgets to override. */
    config->discount_stop_iter  = 35000000;    /* 3.5% of 1B */
    config->discount_interval   =   875000;    /* 0.087% of 1B */
    config->prune_start_iter    = 17000000;    /* 1.7% of 1B */
    config->snapshot_start_iter = 70000000;    /* 7% of 1B */
    config->snapshot_interval   = 17000000;    /* 1.7% of 1B */
    config->strategy_interval   =    10000;    /* Pluribus's only iter-count threshold */
    config->num_threads         = 0;           /* auto */
    config->hash_table_size     = 0;           /* auto = BP_HASH_SIZE_LARGE */
    config->snapshot_dir        = NULL;
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
    /* Bug 14 fix: only allocate the texture cache if the caller hasn't already
     * loaded one via bp_load_texture_cache(). The canonical Python driver
     * (blueprint_worker_unified.py) calls bp_load_texture_cache BEFORE
     * bp_init_unified, and without this guard the previous unconditional
     * calloc here would leak the loaded buffer (~9 MB). */
    if (!s->texture_bucket_cache) {
        s->num_cached_textures = 0;
        s->texture_bucket_cache = (int*)calloc(BP_MAX_TEXTURES * (size_t)BP_MAX_HANDS, sizeof(int));
    }

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
        /* Bug 7 fix: build the texture lookup hashmap */
        texture_index_rebuild(s);
        double pc_elapsed = (double)(clock() - pc_start) / CLOCKS_PER_SEC;
        printf("[BP] Precomputed %d textures in %.1fs\n", tex_count, pc_elapsed);
    }
    } /* end else (skip if cache loaded) */

    /* Precompute turn k-means centroids.
     * Sample random turn boards, compute [EHS, PPot, NPot] features for all
     * valid hands, then run k-means to get 200 centroids. During traversal,
     * each hand's features are computed inline and mapped to nearest centroid.
     * This replaces floor(ehs * 200) percentile bucketing on the turn.
     *
     * Skip the precompute if a cached centroids file is found. The file is
     * tiny (~2.4 KB for 200 centroids × 3 floats × 4 bytes) and saves
     * 6 minutes of single-threaded compute on resume. */
    {
        const char *centroid_paths[] = {
            "/tmp/turn_centroids.bin",
            "/dev/shm/turn_centroids.bin",
            "turn_centroids.bin",
        };
        int loaded_centroids = 0;
        for (int i = 0; i < 3 && !loaded_centroids; i++) {
            FILE *f = fopen(centroid_paths[i], "rb");
            if (!f) continue;
            char magic[4];
            int saved_k;
            if (fread(magic, 1, 4, f) == 4 && memcmp(magic, "TCN1", 4) == 0 &&
                fread(&saved_k, sizeof(int), 1, f) == 1 &&
                saved_k > 0 && saved_k <= 200) {
                if (fread(s->turn_centroids, sizeof(float), saved_k * 3, f)
                    == (size_t)(saved_k * 3)) {
                    s->turn_centroids_k = saved_k;
                    loaded_centroids = 1;
                    printf("[BP] Loaded %d turn centroids from %s\n",
                           saved_k, centroid_paths[i]);
                }
            }
            fclose(f);
        }
        if (!loaded_centroids) {
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

        /* Save the freshly-computed centroids for future runs. */
        FILE *fc = fopen("/tmp/turn_centroids.bin", "wb");
        if (fc) {
            fwrite("TCN1", 1, 4, fc);
            fwrite(&actual_k, sizeof(int), 1, fc);
            fwrite(s->turn_centroids, sizeof(float), actual_k * 3, fc);
            fclose(fc);
            printf("[BP] Saved turn centroids to /tmp/turn_centroids.bin\n");
        }
        } /* end if (!loaded_centroids) */
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

    /* Bug C/F sanity check: warn if any timing field looks dangerously stale.
     * The previous defaults (400000, 200000, 800000) were calibrated to
     * ~1000 iter/min hardware and produce wildly wrong behavior at our rate.
     * If a caller passed those literal values, scream so the operator notices. */
    if (s->config.discount_stop_iter > 0 && s->config.discount_stop_iter < 1000000) {
        fprintf(stderr, "[BP] WARNING: discount_stop_iter=%lld is suspiciously small. "
                "Pluribus uses 3.5%% of training (e.g. 35M for a 1B target, 280M for 8B).\n",
                (long long)s->config.discount_stop_iter);
    }
    if (s->config.prune_start_iter > 0 && s->config.prune_start_iter < 500000) {
        fprintf(stderr, "[BP] WARNING: prune_start_iter=%lld is suspiciously small. "
                "Pluribus uses 1.74%% of training.\n",
                (long long)s->config.prune_start_iter);
    }

    printf("[BP] Starting %d-player MCCFR: %lld iterations, %d threads, "
           "hash=%lld, buckets=%s\n",
           NP, (long long)max_iterations, nt, (long long)s->info_table.table_size,
           s->use_buckets ? "yes" : "no");

    /* Pre-fault all hash table pages to avoid kernel mm_lock contention
     * during the first few minutes of solving. With lazy allocation, the
     * first random access to each empty slot triggers a page fault that
     * holds a kernel lock. With 192 threads doing random probes, this
     * serializes through mm_lock and tanks throughput to ~5K iter/s.
     *
     * Strategy: madvise(MADV_POPULATE_WRITE) on the page-aligned interior
     * of each array. madvise requires page-aligned addresses, but calloc
     * doesn't guarantee that. We round the start up and the length down
     * to page boundaries — the few KB of un-aligned head/tail will get
     * lazily faulted, which is fine since it's microscopic. */
    {
        BPInfoTable *t = &s->info_table;
        printf("[BP] Pre-faulting hash table pages...\n");
        fflush(stdout);
        #ifdef _OPENMP
        double prefault_start = omp_get_wtime();
        #endif

        long pagesize = sysconf(_SC_PAGESIZE);
        if (pagesize <= 0) pagesize = 4096;

        struct { void *addr; size_t len; const char *name; } regions[] = {
            { t->occupied, (size_t)t->table_size * sizeof(int), "occupied" },
            { t->keys,     (size_t)t->table_size * sizeof(BPInfoKey), "keys" },
            { t->sets,     (size_t)t->table_size * sizeof(BPInfoSet), "sets" },
        };

        for (size_t r = 0; r < 3; r++) {
            uintptr_t start = (uintptr_t)regions[r].addr;
            uintptr_t end = start + regions[r].len;
            uintptr_t aligned_start = (start + pagesize - 1) & ~((uintptr_t)pagesize - 1);
            uintptr_t aligned_end = end & ~((uintptr_t)pagesize - 1);
            if (aligned_end <= aligned_start) continue;
            size_t aligned_len = aligned_end - aligned_start;

            /* Request transparent huge pages. With 138GB working set and
             * random hash probes, the dTLB (3072 4K entries on Zen4) misses
             * on nearly every probe. 2MB huge pages cut the page table
             * working set 512x, letting more translations fit in the TLB
             * hierarchy. Huge TLB misses are also cheaper (one fewer level). */
            #ifdef MADV_HUGEPAGE
            madvise((void*)aligned_start, aligned_len, MADV_HUGEPAGE);
            #endif

            int used_madvise = 0;
            #ifdef MADV_POPULATE_WRITE
            if (madvise((void*)aligned_start, aligned_len, MADV_POPULATE_WRITE) == 0) {
                used_madvise = 1;
            } else {
                fprintf(stderr, "[BP] madvise(%s) failed: %s — falling back to write loop\n",
                        regions[r].name, strerror(errno));
            }
            #endif

            if (!used_madvise) {
                /* Explicit page-stride writes. Volatile prevents the compiler
                 * from optimizing the read-write pair away. Each write to a
                 * virgin page triggers CoW, allocating a real physical page. */
                volatile char *p = (volatile char*)aligned_start;
                #pragma omp parallel for schedule(static)
                for (size_t i = 0; i < aligned_len; i += (size_t)pagesize) {
                    char v = p[i];
                    p[i] = v;
                }
            }
        }

        #ifdef _OPENMP
        printf("[BP] Pre-fault complete in %.1fs\n",
               omp_get_wtime() - prefault_start);
        #else
        printf("[BP] Pre-fault complete\n");
        #endif
        fflush(stdout);
    }

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

    /* Batch size for the OpenMP parallel-for loop. This is INDEPENDENT
     * of discount_interval — they used to be the same variable, which
     * tanked parallelism: with discount_interval=10000 and chunk_size=64,
     * a batch had only 156 chunks for 192 threads → 36+ idle threads at
     * every barrier.
     *
     * Rules:
     *   - During discount phase: batch_size MUST be <= discount_interval
     *     so we don't miss discount triggers (one discount per batch).
     *   - After discount phase: use a much larger batch (10M) so 192
     *     threads with chunk 64 have ~150K chunks of work — full parallelism.
     *   - Snapshots fire when (global_batch_end % snapshot_interval) < batch_size,
     *     which works for both small and large batch sizes.
     *
     * Bug 10 fix: batch_size is recomputed inside the batch loop below
     * (rather than once at solve start) so that a chunk crossing the
     * discount→post-discount boundary correctly switches to the larger
     * batch_size mid-chunk instead of using the small discount_interval
     * for the entire chunk. */
    int64_t POSTDISCOUNT_BATCH_SIZE = 10000000;
    int64_t initial_batch_size;
    int64_t global_start_iter = iter_offset + 1;
    if (global_start_iter <= s->config.discount_stop_iter) {
        initial_batch_size = s->config.discount_interval;
        if (initial_batch_size <= 0) initial_batch_size = max_iterations;
    } else {
        initial_batch_size = POSTDISCOUNT_BATCH_SIZE;
    }
    if (initial_batch_size > max_iterations) initial_batch_size = max_iterations;
    /* num_batches is an upper-bound estimate for logging and the loop
     * counter; the actual number of batches may be slightly different if
     * batch_size changes mid-chunk at the discount boundary. We use a
     * generous upper bound (assuming smallest possible batch size). */
    int64_t num_batches_estimate = (max_iterations + initial_batch_size - 1) / initial_batch_size;
    printf("[BP] batch_size=%lld (~%lld batches), discount_phase=%s\n",
           (long long)initial_batch_size, (long long)num_batches_estimate,
           (global_start_iter <= s->config.discount_stop_iter) ? "yes" : "no");
    fflush(stdout);

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

        /* Bug 10 fix: this loop replaces a fixed-num_batches `for` loop with
         * a position-tracking `while` loop so batch_size can be recomputed
         * mid-chunk when crossing the discount→post-discount boundary. */
        int64_t cur_iter_in_chunk = 1;
        int64_t batch = 0;
        while (cur_iter_in_chunk <= max_iterations) {
            /* Recompute batch_size for THIS batch based on current global iter
             * (not just the start of the chunk). Switches from
             * discount_interval to POSTDISCOUNT_BATCH_SIZE the moment we cross
             * the discount_stop_iter threshold. */
            int64_t cur_global_iter = cur_iter_in_chunk + iter_offset;
            int64_t batch_size;
            if (cur_global_iter <= s->config.discount_stop_iter) {
                batch_size = s->config.discount_interval;
                if (batch_size <= 0) batch_size = max_iterations - cur_iter_in_chunk + 1;
            } else {
                batch_size = POSTDISCOUNT_BATCH_SIZE;
            }
            int64_t batch_start = cur_iter_in_chunk;
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
                /* Only zero the scalar state fields that need to be zero.
                 * Arrays are written as needed. This avoids 6.3 KB of memset
                 * per iteration (20+ GB/s of L1 bandwidth wasted across 192
                 * threads at full speed). */
                ts.solver = s;
                ts.rng = my_rng;
                ts.traverser = traverser;
                ts.iteration = global_iter;
                ts.use_pruning = use_pruning;
                ts.num_raises = 0;
                ts.num_canon_board = 0;
                ts.history_len = 0;
                ts.flop_cache = NULL;
                for (int p = 0; p < BP_MAX_PLAYERS; p++) {
                    ts.turn_bucket[p] = -1;
                    ts.river_bucket[p] = -1;
                }
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
                        ts.bets[p] = 0;       /* Previously zeroed by removed memset */
                        ts.has_acted[p] = 0;  /* Previously zeroed by removed memset */
                    }
                    traverse(&ts, 0, acting_order, NP);
                }

                /* Bug γ fix: atomic monotonic update so resume from a partial
                 * checkpoint doesn't underflow the iter offset. Previously only
                 * tid 0 wrote, with no atomic, and could write a value behind
                 * what other threads had already executed. Use a CAS loop with
                 * RELEASE so the saved iter count is always >= the highest
                 * executed iter at the time of the save. */
                {
                    int64_t cur = __atomic_load_n(&s->iterations_run, __ATOMIC_RELAXED);
                    while (global_iter > cur &&
                           !__atomic_compare_exchange_n(&s->iterations_run, &cur, global_iter,
                                                         1, __ATOMIC_RELEASE, __ATOMIC_RELAXED)) { }
                }
            }
            /* implicit barrier after omp for */

            /* Decide whether to discount/snapshot in omp single, then run
             * the parallel work OUTSIDE the single block using the existing
             * team. This avoids nested parallelism which was silently
             * serializing accumulate_snapshot/apply_discount.
             *
             * The variables below are thread-private. We use `copyprivate`
             * on the omp single to broadcast the values to all threads. */
            int do_discount = 0;
            int do_snapshot = 0;
            float discount_value = 0.0f;
            int64_t global_batch_end = (int64_t)batch_end + iter_offset;

            #ifdef _OPENMP
            #pragma omp single copyprivate(do_discount, do_snapshot, discount_value)
            #endif
            {
                /* Per-batch progress print */
                #ifdef _OPENMP
                double elapsed = omp_get_wtime() - t_start;
                #else
                double elapsed = (double)(clock() - t_start) / CLOCKS_PER_SEC;
                #endif
                double rate = (batch_end > 0 && elapsed > 0) ? (double)batch_end / elapsed : 0.0;
                printf("[BP] batch %lld/~%lld done at iter %lld (global %lld), "
                       "%.1fs, %.0f iter/s, info sets: %lld\n",
                       (long long)(batch + 1), (long long)num_batches_estimate,
                       (long long)batch_end, (long long)global_batch_end,
                       elapsed, rate, (long long)s->info_table.num_entries);
                fflush(stdout);

                /* Decide whether to discount this batch */
                if (global_batch_end <= s->config.discount_stop_iter &&
                    global_batch_end >= next_discount_at) {
                    discount_count++;
                    float t_val = (float)discount_count;
                    discount_value = t_val / (t_val + 1.0f);
                    next_discount_at = (discount_count + 1) * s->config.discount_interval;
                    do_discount = 1;
                    printf("[BP] Applying Linear CFR discount #%lld: d=%.4f at iter %lld\n",
                           (long long)discount_count, discount_value, (long long)global_batch_end);
                }

                /* Decide whether to snapshot */
                if (global_batch_end >= s->config.snapshot_start_iter &&
                    s->config.snapshot_interval > 0 &&
                    (global_batch_end % s->config.snapshot_interval) < batch_size) {
                    do_snapshot = 1;
                    s->snapshots_saved++;
                    printf("[BP] Accumulating strategy snapshot #%lld at iter %lld\n",
                           (long long)s->snapshots_saved, (long long)global_batch_end);
                    fflush(stdout);
                }
            }
            /* After copyprivate: all threads have do_discount/do_snapshot/
             * discount_value set to the same values. */

            /* Run the discount/snapshot in parallel using the existing
             * thread team. These functions use `#pragma omp for nowait`. */
            if (do_discount) {
                apply_discount(&s->info_table, discount_value);
            }
            if (do_snapshot) {
                accumulate_snapshot(&s->info_table);
            }
            /* Explicit barrier: ensure all threads finish before next batch. */
            #ifdef _OPENMP
            #pragma omp barrier
            #endif

            /* Advance to the next batch position. Bug 10 fix: this is now
             * inside the while loop so the loop variable advances by the
             * actual batch_size used (which may differ from the previous
             * iteration if we crossed the discount→post-discount boundary). */
            cur_iter_in_chunk = batch_end + 1;
            batch++;
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

    for (int probe = 0; probe < HASH_PROBE_LIMIT_READ; probe++) {
        int64_t idx = (slot + probe) % s->info_table.table_size;
        /* Bug 8 fix: use ACQUIRE load + state-machine handling. The previous
         * code used a non-atomic `!s->info_table.occupied[idx]` check which
         * (a) didn't use an acquire barrier so reads of keys[idx] and
         * sets[idx] could be reordered before the state load, and (b) treated
         * state==2 (initializing) as state==1 (ready), allowing key_eq to
         * compare against partially-initialized keys. With proper handling we
         * spin-wait on state==2 slots and only proceed once they're ready. */
        int state = __atomic_load_n(&s->info_table.occupied[idx], __ATOMIC_ACQUIRE);
        if (state == 0) return 0;
        if (state == 2) {
            spin_until_ready((const int*)&s->info_table.occupied[idx]);
            state = __atomic_load_n(&s->info_table.occupied[idx], __ATOMIC_ACQUIRE);
            if (state != 1) continue;
        }
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

    for (int probe = 0; probe < HASH_PROBE_LIMIT_READ; probe++) {
        int64_t idx = (slot + probe) % s->info_table.table_size;
        /* Bug 8 fix: same state-machine handling as bp_get_strategy above. */
        int state = __atomic_load_n(&s->info_table.occupied[idx], __ATOMIC_ACQUIRE);
        if (state == 0) return 0;
        if (state == 2) {
            spin_until_ready((const int*)&s->info_table.occupied[idx]);
            state = __atomic_load_n(&s->info_table.occupied[idx], __ATOMIC_ACQUIRE);
            if (state != 1) continue;
        }
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

void bp_get_table_stats(const BPSolver *s,
                         int64_t *out_entries,
                         int64_t *out_table_size,
                         int64_t *out_insertion_failures,
                         int64_t *out_max_probe_observed) {
    /* Relaxed loads — these are stat counters with no synchronization
     * requirements, racy reads are acceptable. */
    if (out_entries)
        *out_entries = __atomic_load_n(&s->info_table.num_entries, __ATOMIC_RELAXED);
    if (out_table_size)
        *out_table_size = s->info_table.table_size;
    if (out_insertion_failures)
        *out_insertion_failures = __atomic_load_n(&s->info_table.insertion_failures, __ATOMIC_RELAXED);
    if (out_max_probe_observed)
        *out_max_probe_observed = __atomic_load_n(&s->info_table.max_probe_observed, __ATOMIC_RELAXED);
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

/* ── Regret loader ────────────────────────────────────────────────
 *
 * Two implementations share the same public entry point bp_load_regrets:
 *
 *   (a) load_regrets_serial: legacy fread-based loader. Always used for
 *       BPR2/BPR3 format (which we don't generate anymore but still have
 *       to be able to read). Also the fallback when BP_LEGACY_LOADER=1.
 *
 *   (b) load_regrets_mmap_parallel: BPR4-only. mmaps the whole file,
 *       walks once serially to build an offset index, then dispatches
 *       threads to process the index in parallel. ~20x faster on
 *       r6i.8xlarge for the 1.5B v2 checkpoint (60 GB).
 *
 * Set BP_LEGACY_LOADER=1 in the environment to force the serial loader
 * on BPR4 files too — useful for correctness diffing old vs new.
 *
 * Both loaders MUST produce an identical final hash table state (same
 * slot assignments, same regret sums, same strategy_sum state). A
 * round-trip test in tests/test_phase_1_3_synthetic.c validates this on
 * a toy checkpoint.
 *
 * The key correctness invariants preserved in BOTH loaders:
 *   1. Regrets are ADDED, not written. Fresh arena slots have zeroed
 *      regrets, so + is idempotent on fresh loads, and recovers split
 *      regrets on duplicate-key Hogwild collisions.
 *   2. strategy_sum is lazily arena-allocated on first encounter.
 *   3. info_table_find_or_create uses the existing Hogwild-safe CAS;
 *      its return slot is thread-safe to write to (each thread writes
 *      to its own distinct slot after the CAS). */

static int64_t load_regrets_serial(BPSolver *s, FILE *f,
                                    int is_v4, int is_v3, int is_v2,
                                    int64_t saved_entries,
                                    int64_t saved_iters) {
    (void)is_v3; (void)is_v2;
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

        /* Read regrets into a temp buffer, check if the slot already
         * has data (= duplicate key from the Hogwild race bug), then
         * ADD to the slot. For fresh slots (arena zeroed): 0 + new =
         * new. For duplicate keys: existing + new = merged sum.
         *
         * NOTE: the is_dup counter is a LOWER BOUND, not an exact
         * count. If a pair of duplicate regrets happens to sum to
         * zero in every slot, a third duplicate would see all-zero
         * regrets and be mis-reported as "not a duplicate". The
         * arithmetic is still correct (the add is idempotent on the
         * zero state); only the `[BP] WARNING: merged N` log line
         * would under-count. Rare in practice on non-trivially
         * trained checkpoints. */
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

    t->insertion_failures = 0;
    t->max_probe_observed = 0;

    if (merged > 0) {
        printf("[BP] WARNING: merged %lld duplicate entries (Hogwild race bug). "
               "Split regrets have been recovered.\n", (long long)merged);
    }
    printf("[BP] Loaded %lld/%lld info sets (%lld merged) [serial], table %lld/%lld\n",
           (long long)loaded, (long long)saved_entries, (long long)merged,
           (long long)t->num_entries, (long long)t->table_size);
    return loaded;
}

/* ── mmap compat shim ──────────────────────────────────────────────
 *
 * Linux: standard POSIX mmap.
 * Windows (MSYS2): CreateFileMapping + MapViewOfFile.
 *
 * We only need a read-only private mapping of the whole file. */
typedef struct {
    const unsigned char *base;
    size_t size;
#ifdef _WIN32
    HANDLE h_file;
    HANDLE h_map;
#else
    int fd;
#endif
} BPFileMap;

static int bp_file_map_open(BPFileMap *m, const char *path) {
    memset(m, 0, sizeof(*m));
#ifdef _WIN32
    m->h_file = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (m->h_file == INVALID_HANDLE_VALUE) return -1;
    LARGE_INTEGER sz;
    if (!GetFileSizeEx(m->h_file, &sz)) {
        CloseHandle(m->h_file); return -1;
    }
    m->size = (size_t)sz.QuadPart;
    m->h_map = CreateFileMappingA(m->h_file, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!m->h_map) { CloseHandle(m->h_file); return -1; }
    m->base = (const unsigned char*)MapViewOfFile(m->h_map, FILE_MAP_READ, 0, 0, 0);
    if (!m->base) { CloseHandle(m->h_map); CloseHandle(m->h_file); return -1; }
    return 0;
#else
    m->fd = open(path, O_RDONLY);
    if (m->fd < 0) return -1;
    struct stat st;
    if (fstat(m->fd, &st) < 0) { close(m->fd); return -1; }
    m->size = (size_t)st.st_size;
    m->base = (const unsigned char*)mmap(NULL, m->size, PROT_READ,
                                          MAP_PRIVATE, m->fd, 0);
    if (m->base == MAP_FAILED) { close(m->fd); return -1; }
    /* MADV_SEQUENTIAL hints the kernel to aggressively read ahead and
     * drop pages behind us. Good for a streaming one-pass walk. */
#ifdef MADV_SEQUENTIAL
    madvise((void*)m->base, m->size, MADV_SEQUENTIAL);
#endif
    return 0;
#endif
}

static void bp_file_map_close(BPFileMap *m) {
#ifdef _WIN32
    if (m->base) UnmapViewOfFile(m->base);
    if (m->h_map) CloseHandle(m->h_map);
    if (m->h_file != INVALID_HANDLE_VALUE) CloseHandle(m->h_file);
#else
    if (m->base && m->base != MAP_FAILED)
        munmap((void*)m->base, m->size);
    if (m->fd >= 0) close(m->fd);
#endif
    memset(m, 0, sizeof(*m));
}

/* BPR4 parallel loader using mmap + two-pass index.
 *
 * Pass 1 (serial, single thread):
 *   Walk the file from offset data_start (just past the 28-byte BPR4
 *   header) to eof. For each entry, read num_actions and has_ss in
 *   order to know its on-disk size, then record the entry's byte
 *   offset into an int64 array. No hash inserts, no allocations, just
 *   linear pointer arithmetic. Pages stream into cache as we touch
 *   them. Expected time: ~60-90s for 1.2B entries on r6i.8xlarge.
 *
 * Pass 2 (parallel, #OMP threads):
 *   Each thread takes a chunk of the index range. For each entry, read
 *   the full entry directly from the mmap'd pointer, call
 *   info_table_find_or_create, and merge regrets + strategy_sum into
 *   the slot. The find-or-create CAS is already lock-free; the only
 *   thing we need to preserve is the "fresh slot zeroed -> add is
 *   safe" invariant, which holds because the arena is calloc'd.
 *
 * Merge safety: two threads MAY race on the same slot if the original
 * training run had duplicate keys (Bug 11 / Hogwild race) — the serial
 * loader relies on "previous += new" being sequential, but under
 * parallelism two threads could race on the same regret[] slot. We fix
 * this with atomic fetch_add on each regret array entry. strategy_sum
 * allocation uses the same CAS pattern as ensure_strategy_sum. */
static int64_t load_regrets_mmap_parallel(BPSolver *s, const char *path,
                                           int64_t saved_entries,
                                           int64_t saved_iters) {
    BPFileMap mfile;
    if (bp_file_map_open(&mfile, path) != 0) {
        fprintf(stderr, "[BP] mmap failed on %s, falling back to serial\n", path);
        return -2;  /* caller will retry with serial */
    }
    printf("[BP] mmap: %zu bytes\n", mfile.size); fflush(stdout);

    /* Data starts just past the 28-byte BPR4 header (magic 4 + table_size 8 +
     * num_entries 8 + iterations 8). */
    const size_t data_start = 28;
    const unsigned char *base = mfile.base;
    size_t pos = data_start;
    size_t eof = mfile.size;

    /* ── Pass 1: build offset index ──
     *
     * Pre-allocate enough slots for saved_entries; if the walk finds fewer
     * than that (truncated file), we shrink. On corrupt file we bail.
     * Allocating 8 bytes × 1.2B entries = 9.6 GB — not tiny but manageable
     * on r6i.8xlarge (256 GB) and on the PC's 64 GB for the 200M checkpoint
     * (which only needs ~1.6 GB of index). */
    int64_t *offsets = (int64_t*)malloc((size_t)saved_entries * sizeof(int64_t));
    if (!offsets) {
        fprintf(stderr, "[BP] OOM allocating %lld-entry offset index\n",
                (long long)saved_entries);
        bp_file_map_close(&mfile);
        return -1;
    }

    printf("[BP] Pass 1: building offset index (serial walk)...\n"); fflush(stdout);
    double t0 = 0.0;
#ifdef _OPENMP
    t0 = omp_get_wtime();
#endif

    int64_t n_indexed = 0;
    int64_t last_progress = 0;
    for (; n_indexed < saved_entries; n_indexed++) {
        if (pos + 32 > eof) {
            fprintf(stderr, "[BP] Pass 1: unexpected EOF at offset %zu, "
                    "entry %lld / %lld\n", pos, (long long)n_indexed,
                    (long long)saved_entries);
            break;
        }
        offsets[n_indexed] = (int64_t)pos;

        /* Read na without touching regrets — we need it to compute the
         * entry size. Key fields are: player(4) + street(4) + bucket(4)
         * + board_hash(8) + action_hash(8) = 28 bytes, then na at offset
         * 28 from entry start. */
        int na;
        memcpy(&na, base + pos + 28, sizeof(int));
        if (na < 0 || na > BP_MAX_ACTIONS) {
            fprintf(stderr, "[BP] Pass 1: corrupt na=%d at offset %zu\n",
                    na, pos);
            free(offsets); bp_file_map_close(&mfile); return -1;
        }

        /* Advance past key(28) + na(4) + regrets(4*na) + has_ss(4) */
        size_t next = pos + 28 + 4 + (size_t)(4 * na) + 4;
        if (next > eof) {
            fprintf(stderr, "[BP] Pass 1: truncated regrets at entry %lld\n",
                    (long long)n_indexed);
            break;
        }
        int has_ss;
        memcpy(&has_ss, base + pos + 28 + 4 + (size_t)(4 * na), sizeof(int));
        if (has_ss) {
            next += (size_t)(4 * na);
            if (next > eof) {
                fprintf(stderr, "[BP] Pass 1: truncated strategy_sum at entry %lld\n",
                        (long long)n_indexed);
                break;
            }
        }
        pos = next;

        if (n_indexed - last_progress >= 100000000) {
            last_progress = n_indexed;
#ifdef _OPENMP
            double dt = omp_get_wtime() - t0;
            printf("[BP]   indexed %lld / %lld entries (%.1fs, %.0f M/s)\n",
                   (long long)n_indexed, (long long)saved_entries, dt,
                   (double)n_indexed / (dt * 1e6));
#else
            printf("[BP]   indexed %lld / %lld entries\n",
                   (long long)n_indexed, (long long)saved_entries);
#endif
            fflush(stdout);
        }
    }

#ifdef _OPENMP
    double pass1_sec = omp_get_wtime() - t0;
    printf("[BP] Pass 1 done: %lld entries in %.1fs\n",
           (long long)n_indexed, pass1_sec);
#else
    printf("[BP] Pass 1 done: %lld entries\n", (long long)n_indexed);
#endif
    fflush(stdout);

    /* ── Pass 2: parallel hash inserts ── */
    BPInfoTable *t = &s->info_table;
    int64_t loaded = 0;
    int64_t merged = 0;
    int64_t dropped = 0;

#ifdef _OPENMP
    t0 = omp_get_wtime();
#endif
    printf("[BP] Pass 2: parallel hash inserts (%d threads)...\n",
#ifdef _OPENMP
           omp_get_max_threads()
#else
           1
#endif
    ); fflush(stdout);

    int64_t progress_stride = n_indexed / 20;
    if (progress_stride < 1) progress_stride = 1;

#ifdef _OPENMP
    #pragma omp parallel reduction(+:loaded,merged,dropped)
#endif
    {
        int64_t local_loaded = 0;
        int64_t local_merged = 0;
        int64_t local_dropped = 0;

#ifdef _OPENMP
        #pragma omp for schedule(dynamic, 65536)
#endif
        for (int64_t i = 0; i < n_indexed; i++) {
            size_t ep = (size_t)offsets[i];
            /* Decode entry header. Field-by-field memcpy so no alignment
             * assumptions on the mmap region (on x86 unaligned is fine,
             * but being explicit is cheaper than debugging later). */
            BPInfoKey key;
            memcpy(&key.player,      base + ep + 0,  sizeof(int));
            memcpy(&key.street,      base + ep + 4,  sizeof(int));
            memcpy(&key.bucket,      base + ep + 8,  sizeof(int));
            memcpy(&key.board_hash,  base + ep + 12, sizeof(uint64_t));
            memcpy(&key.action_hash, base + ep + 20, sizeof(uint64_t));
            int na;
            memcpy(&na, base + ep + 28, sizeof(int));
            if (na > BP_MAX_ACTIONS) na = BP_MAX_ACTIONS;

            int64_t slot = info_table_find_or_create(t, key, na);
            if (slot < 0) {
                local_dropped++;
                continue;
            }

            BPInfoSet *is = &t->sets[slot];

            /* Merge regrets. Under parallel load, two threads may race on
             * this slot only if the source checkpoint has duplicate keys
             * (Hogwild race bug in training). We use atomic fetch_add to
             * be safe — for the 99.9999% fresh-slot case this compiles
             * to a LOCK ADD which is cheap on x86; for the rare duplicate
             * case it guarantees correctness.
             *
             * Detecting "is this a duplicate?" for reporting is racy but
             * we only use it for logging, not for correctness. We check
             * if any regret was already non-zero BEFORE our add (not
             * atomic with the add — that's fine since we only want an
             * approximate duplicate count). */
            const int *src_regrets = (const int*)(base + ep + 32);
            int is_dup = 0;
            for (int a = 0; a < na; a++) {
                int prev = __atomic_load_n(&is->regrets[a], __ATOMIC_RELAXED);
                if (prev != 0) is_dup = 1;
                int v;
                memcpy(&v, src_regrets + a, sizeof(int));
                __atomic_fetch_add(&is->regrets[a], v, __ATOMIC_RELAXED);
            }

            size_t has_ss_off = ep + 32 + (size_t)(4 * na);
            int has_ss;
            memcpy(&has_ss, base + has_ss_off, sizeof(int));
            if (has_ss) {
                /* Lazy strategy_sum allocation. Same CAS pattern as
                 * ensure_strategy_sum but inlined here to avoid the extra
                 * function call in the hot path. */
                float *ss = __atomic_load_n((float**)&is->strategy_sum, __ATOMIC_ACQUIRE);
                if (!ss) {
                    float *buf = (float*)arena_alloc(na);
                    if (buf) {
                        float *expected = NULL;
                        if (__atomic_compare_exchange_n(&is->strategy_sum, &expected, buf,
                                                         0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
                            ss = buf;
                        } else {
                            /* Another thread won — reload and use theirs.
                             * buf is wasted arena space. */
                            ss = __atomic_load_n((float**)&is->strategy_sum, __ATOMIC_ACQUIRE);
                        }
                    }
                }
                if (ss) {
                    const float *src_ss = (const float*)(base + has_ss_off + 4);
                    /* Atomic float add via CAS loop (no atomic_float_add in
                     * C11). Each slot is independent; contention is only
                     * on duplicate keys. */
                    for (int a = 0; a < na; a++) {
                        float add;
                        memcpy(&add, src_ss + a, sizeof(float));
                        int *slot_i = (int*)&ss[a];
                        union { float f; int i; } old_u, new_u;
                        do {
                            old_u.i = __atomic_load_n(slot_i, __ATOMIC_RELAXED);
                            new_u.f = old_u.f + add;
                        } while (!__atomic_compare_exchange_n(
                            slot_i, &old_u.i, new_u.i,
                            0, __ATOMIC_RELAXED, __ATOMIC_RELAXED));
                    }
                }
            }

            if (is_dup) local_merged++;
            local_loaded++;

            if ((i % progress_stride) == 0 && i > 0) {
#ifdef _OPENMP
                if (omp_get_thread_num() == 0) {
                    double dt = omp_get_wtime() - t0;
                    printf("[BP]   Pass 2 progress: %lld / %lld (%.1fs)\n",
                           (long long)i, (long long)n_indexed, dt);
                    fflush(stdout);
                }
#endif
            }
        }

        loaded += local_loaded;
        merged += local_merged;
        dropped += local_dropped;
    }

#ifdef _OPENMP
    double pass2_sec = omp_get_wtime() - t0;
    printf("[BP] Pass 2 done: %lld loaded, %lld merged, %lld dropped in %.1fs\n",
           (long long)loaded, (long long)merged, (long long)dropped, pass2_sec);
#else
    printf("[BP] Pass 2 done: %lld loaded, %lld merged, %lld dropped\n",
           (long long)loaded, (long long)merged, (long long)dropped);
#endif
    fflush(stdout);

    free(offsets);
    bp_file_map_close(&mfile);

    s->iterations_run = saved_iters;

    t->insertion_failures = 0;
    t->max_probe_observed = 0;

    if (merged > 0) {
        printf("[BP] WARNING: merged %lld duplicate entries (Hogwild race bug). "
               "Split regrets have been recovered.\n", (long long)merged);
    }
    printf("[BP] Loaded %lld/%lld info sets (%lld merged) [parallel], table %lld/%lld\n",
           (long long)loaded, (long long)saved_entries, (long long)merged,
           (long long)t->num_entries, (long long)t->table_size);
    return loaded;
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

    /* Dispatch:
     *   - BPR4 by default → parallel mmap loader
     *   - BPR4 with BP_LEGACY_LOADER=1 → serial loader (for correctness diff)
     *   - BPR2/BPR3 → serial loader (always — these formats are legacy) */
    const char *legacy = getenv("BP_LEGACY_LOADER");
    int force_serial = (legacy && legacy[0] == '1');

    if (is_v4 && !force_serial) {
        fclose(f);  /* parallel loader opens its own mmap */
        int64_t n = load_regrets_mmap_parallel(s, path, saved_entries, saved_iters);
        if (n != -2) return n;  /* success or hard failure */
        /* -2 = mmap failed, fall back to serial — re-open and seek past header */
        f = fopen(path, "rb");
        if (!f) return -1;
        fseek(f, 28, SEEK_SET);
    }

    int64_t n = load_regrets_serial(s, f, is_v4, is_v3, is_v2,
                                     saved_entries, saved_iters);
    fclose(f);
    return n;
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
    /* Bug 7 fix: build the texture lookup hashmap */
    texture_index_rebuild(s);
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

        /* Bug B fix: export the AVERAGE strategy from strategy_sum, not the
         * current regret-matched strategy. The average strategy is what
         * converges to Nash in CFR; the regret-matched current strategy is
         * just an instantaneous noisy snapshot.
         *
         * Mirrors bp_get_strategy(): normalize strategy_sum, fall back to
         * regret_match if no strategy_sum was accumulated for this info set
         * (rare — only newly-created entries that have never seen a snapshot
         * or a periodic accumulation event). */
        if (is->strategy_sum) {
            float sum = 0;
            for (int a = 0; a < na; a++) {
                float v = is->strategy_sum[a];
                if (v < 0) v = 0;
                strategy_buf[a] = v;
                sum += v;
            }
            if (sum > 0) {
                for (int a = 0; a < na; a++) strategy_buf[a] /= sum;
            } else {
                regret_match(is->regrets, strategy_buf, na);
            }
        } else {
            regret_match(is->regrets, strategy_buf, na);
        }

        for (int a = 0; a < na; a++) {
            int q = (int)(strategy_buf[a] * 255.0f + 0.5f);
            if (q < 0) q = 0;
            if (q > 255) q = 255;
            *p++ = (unsigned char)q;
        }
    }

    return 0;
}

/* ── Phase 1.3: σ̄-sampled action EV computation + export ─────────── */

int bp_compute_action_evs(BPSolver *s, int64_t num_iterations) {
    int NP = s->num_players;
    if (NP <= 0) return -1;
    if (s->info_table.num_entries == 0) {
        fprintf(stderr, "[BP1.3] No info sets in table — run bp_solve or bp_load_regrets first\n");
        return -1;
    }

    int nt = s->config.num_threads;
    if (nt <= 0) {
        #ifdef _OPENMP
        nt = omp_get_max_threads();
        #else
        nt = 1;
        #endif
    }

    /* Make sure each thread has its own RNG state. bp_solve() allocates these.
     * If we're running after bp_load_regrets without a prior bp_solve, they
     * may not exist. Allocate on demand. */
    if (!s->rng_states || s->num_rng_states < nt * 8) {
        if (s->rng_states) free(s->rng_states);
        s->rng_states = (uint64_t*)calloc(nt * 8, sizeof(uint64_t));
        if (!s->rng_states) return -1;
        s->num_rng_states = nt * 8;
        /* Seed each thread's RNG with a distinct value — use wall time
         * XOR'd with the thread index so the streams don't collide. */
        uint64_t base_seed = (uint64_t)time(NULL) ^ 0xDEADBEEFCAFEBABEULL;
        for (int t = 0; t < nt; t++) {
            s->rng_states[t * 8] = base_seed + (uint64_t)t * 0x9E3779B97F4A7C15ULL;
            if (s->rng_states[t * 8] == 0) s->rng_states[t * 8] = 1;
        }
    }

    printf("[BP1.3] Computing action EVs: %lld iterations, %d threads, %lld info sets\n",
           (long long)num_iterations, nt, (long long)s->info_table.num_entries);
    fflush(stdout);

    /* Pre-allocate action_evs arena storage for every loaded info set
     * in one sequential pass BEFORE the walk starts. Without this, the
     * walk's first visit to each info set calls ensure_action_evs() →
     * arena_alloc(), which hits a global atomic fetch_add inside
     * arena_grab_slice() on slice refill. With 32 threads all allocating
     * tiny (4-6 action) buffers concurrently during the first N iters,
     * cache-line bouncing on the arena head pointer stalls everyone.
     *
     * One-time sequential pre-allocation eliminates this: single thread
     * walks the table, calls arena_alloc for each occupied slot, stores
     * the result. No concurrent arena churn during the walk. Expected
     * memory cost: ~1.2B info sets × 4 actions avg × 4B float = ~20 GB
     * (same as it would have been on-demand during the walk, just up
     * front and deterministic).
     *
     * This was the main throughput bottleneck observed in prod run 2
     * on 2026-04-11 (50M walk running ~1800 iter/s vs design's ~100K). */
    {
        #ifdef _OPENMP
        double t_prealloc = omp_get_wtime();
        #else
        clock_t t_prealloc = clock();
        #endif
        int64_t prealloc_count = 0;
        for (int64_t i = 0; i < s->info_table.table_size; i++) {
            if (s->info_table.occupied[i] != 1) continue;
            BPInfoSet *is = &s->info_table.sets[i];
            /* Skip if somehow already allocated (shouldn't happen in
             * a fresh bp_compute_action_evs call). */
            if (is->action_evs != NULL) continue;
            float *buf = (float*)arena_alloc(is->num_actions);
            if (buf) {
                is->action_evs = buf;
                prealloc_count++;
            }
        }
        #ifdef _OPENMP
        double prealloc_elapsed = omp_get_wtime() - t_prealloc;
        #else
        double prealloc_elapsed = (double)(clock() - t_prealloc) / CLOCKS_PER_SEC;
        #endif
        printf("[BP1.3] Pre-allocated action_evs for %lld info sets in %.1fs "
               "(eliminates per-visit arena contention during walk)\n",
               (long long)prealloc_count, prealloc_elapsed);
        fflush(stdout);
    }

    #ifdef _OPENMP
    double t_start = omp_get_wtime();
    omp_set_num_threads(nt);
    #else
    clock_t t_start = clock();
    #endif

    int64_t progress_interval = num_iterations / 20;  /* 5% progress marks */
    if (progress_interval < 1) progress_interval = 1;

    /* Shared atomic iteration counter for progress reporting. The old
     * tid==0 + iter%interval scheme was broken with schedule(dynamic,64)
     * because tid 0 only processes a sparse random subset of iters. */
    int64_t completed_iters = 0;
    int64_t next_progress_mark = progress_interval;

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

        #ifdef _OPENMP
        #pragma omp for schedule(dynamic, 64)
        #endif
        for (int64_t iter = 1; iter <= num_iterations; iter++) {
            int traverser = (int)((iter - 1) % NP);

            int sampled_hands[BP_MAX_PLAYERS];
            for (int p = 0; p < NP; p++)
                sampled_hands[p] = rng_int(my_rng, s->num_hands[p]);

            /* Card conflict check (same as bp_solve) */
            int conflict = 0;
            for (int p = 0; p < NP && !conflict; p++) {
                int c0 = s->hands[p][sampled_hands[p]][0];
                int c1 = s->hands[p][sampled_hands[p]][1];
                if (!s->config.include_preflop) {
                    if (card_in_set(c0, s->flop, 3) || card_in_set(c1, s->flop, 3))
                        { conflict = 1; break; }
                }
                for (int q = 0; q < p; q++) {
                    int d0 = s->hands[q][sampled_hands[q]][0];
                    int d1 = s->hands[q][sampled_hands[q]][1];
                    if (cards_conflict(c0, c1, d0, d1)) { conflict = 1; break; }
                }
            }
            if (conflict) continue;

            TraversalState ts;
            ts.solver = s;
            ts.rng = my_rng;
            ts.traverser = traverser;
            ts.iteration = iter;
            ts.use_pruning = 0;  /* NEVER prune in the EV walk */
            ts.num_raises = 0;
            ts.num_canon_board = 0;
            ts.history_len = 0;
            ts.flop_cache = NULL;
            for (int p = 0; p < BP_MAX_PLAYERS; p++) {
                ts.turn_bucket[p] = -1;
                ts.river_bucket[p] = -1;
            }
            memcpy(ts.sampled_hands, sampled_hands, sizeof(sampled_hands));

            if (s->config.include_preflop) {
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
                if (NP > 0) {
                    ts.bets[0] = s->small_blind;
                    ts.invested[0] = s->small_blind;
                    ts.stacks[0] -= s->small_blind;
                }
                if (NP > 1) {
                    ts.bets[1] = s->big_blind;
                    ts.invested[1] = s->big_blind;
                    ts.stacks[1] -= s->big_blind;
                }
                int preflop_order[BP_MAX_PLAYERS];
                for (int i = 0; i < NP; i++)
                    preflop_order[i] = (i + 2) % NP;

                traverse_ev(&ts, 0, preflop_order, NP);
            } else {
                memcpy(ts.board, s->flop, 3 * sizeof(int));
                ts.num_board = 3;
                for (int p = 0; p < NP; p++) ts.active[p] = 1;
                ts.num_active = NP;
                ts.pot = s->starting_pot;
                for (int p = 0; p < NP; p++) {
                    ts.stacks[p] = s->effective_stack;
                    ts.invested[p] = s->starting_pot / NP;
                    ts.bets[p] = 0;
                    ts.has_acted[p] = 0;
                }
                int postflop_order[BP_MAX_PLAYERS];
                for (int i = 0; i < NP; i++) postflop_order[i] = i;
                traverse_ev(&ts, 0, postflop_order, NP);
            }

            /* Progress reporting via shared atomic counter — whichever
             * thread crosses a 5% mark prints. Works with any OpenMP
             * schedule. */
            int64_t my_count = __atomic_add_fetch(&completed_iters, 1, __ATOMIC_RELAXED);
            int64_t cur_mark = __atomic_load_n(&next_progress_mark, __ATOMIC_RELAXED);
            if (my_count >= cur_mark) {
                /* Try to claim this mark so only one thread prints per threshold. */
                int64_t new_mark = cur_mark + progress_interval;
                if (__atomic_compare_exchange_n(&next_progress_mark, &cur_mark, new_mark,
                                                 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
                    #ifdef _OPENMP
                    double elapsed = omp_get_wtime() - t_start;
                    #else
                    double elapsed = (double)(clock() - t_start) / CLOCKS_PER_SEC;
                    #endif
                    double frac = (double)my_count / (double)num_iterations;
                    double rate = my_count / (elapsed > 0 ? elapsed : 1);
                    printf("[BP1.3] iter %lld/%lld (%.0f%%), %.1fs, %.0f iter/s\n",
                           (long long)my_count, (long long)num_iterations,
                           frac * 100.0, elapsed, rate);
                    fflush(stdout);
                }
            }
        }
    }

    #ifdef _OPENMP
    double total_time = omp_get_wtime() - t_start;
    #else
    double total_time = (double)(clock() - t_start) / CLOCKS_PER_SEC;
    #endif

    /* Report how many info sets got visited */
    int64_t visited = 0;
    for (int64_t i = 0; i < s->info_table.table_size; i++) {
        if (s->info_table.occupied[i] != 1) continue;
        if (s->info_table.sets[i].ev_visit_count > 0) visited++;
    }

    printf("[BP1.3] Done: %lld iterations, %.1fs, %lld/%lld info sets visited (%.1f%%)\n",
           (long long)num_iterations, total_time,
           (long long)visited, (long long)s->info_table.num_entries,
           100.0 * (double)visited / (double)s->info_table.num_entries);
    fflush(stdout);

    return 0;
}

int bp_export_action_evs(const BPSolver *s,
                          unsigned char *buf, size_t buf_size,
                          size_t *bytes_written) {
    const BPInfoTable *t = &s->info_table;

    /* First pass: compute required size */
    size_t total = 0;
    int count = 0;
    /* Header: 4 bytes magic + 4 bytes num_entries */
    total += 8;
    for (int64_t i = 0; i < t->table_size; i++) {
        if (t->occupied[i] != 1) continue;
        BPInfoSet *is = &t->sets[i];
        if (is->ev_visit_count <= 0) continue;  /* skip unvisited */
        if (is->action_evs == NULL) continue;   /* shouldn't happen if visit>0 */
        /* Key: player(1) + street(1) + bucket(2) + action_hash(8) = 12 bytes
         * Meta: num_actions(1) = 1 byte
         * EVs:  4 * num_actions bytes (float32) */
        total += 12 + 1 + 4 * is->num_actions;
        count++;
    }

    *bytes_written = total;
    if (buf == NULL) return 0;
    if (buf_size < total) return -1;

    unsigned char *p = buf;
    memcpy(p, "BPR3", 4); p += 4;
    memcpy(p, &count, 4); p += 4;

    for (int64_t i = 0; i < t->table_size; i++) {
        if (t->occupied[i] != 1) continue;
        BPInfoKey *key = &t->keys[i];
        BPInfoSet *is = &t->sets[i];
        if (is->ev_visit_count <= 0) continue;
        if (is->action_evs == NULL) continue;

        unsigned char player_byte = (unsigned char)key->player;
        unsigned char street_byte = (unsigned char)key->street;
        unsigned short bucket_short = (unsigned short)key->bucket;
        memcpy(p, &player_byte, 1); p += 1;
        memcpy(p, &street_byte, 1); p += 1;
        memcpy(p, &bucket_short, 2); p += 2;
        memcpy(p, &key->action_hash, 8); p += 8;

        unsigned char na_byte = (unsigned char)is->num_actions;
        memcpy(p, &na_byte, 1); p += 1;

        /* Write average EV per action: sum / visit_count */
        float inv_count = 1.0f / (float)is->ev_visit_count;
        for (int a = 0; a < is->num_actions; a++) {
            float avg_ev = is->action_evs[a] * inv_count;
            memcpy(p, &avg_ev, 4); p += 4;
        }
    }

    return 0;
}

static int cmp_int64(const void *a, const void *b) {
    int64_t av = *(const int64_t*)a;
    int64_t bv = *(const int64_t*)b;
    if (av < bv) return -1;
    if (av > bv) return 1;
    return 0;
}

int bp_get_ev_visit_stats(const BPSolver *s, BPEVVisitStats *out) {
    if (!out) return -1;
    memset(out, 0, sizeof(*out));

    const BPInfoTable *t = &s->info_table;

    /* First pass: count visited entries to size the percentile array. */
    int64_t n = 0;
    for (int64_t i = 0; i < t->table_size; i++) {
        if (t->occupied[i] != 1) continue;
        if (t->sets[i].ev_visit_count > 0) n++;
    }
    out->total_visited = n;
    if (n == 0) return 0;

    /* Second pass: collect visit counts into a contiguous array for
     * sort-based percentiles. For large tables (1.2B × 8B = 9.6 GB),
     * this is a one-time transient allocation at export. */
    int64_t *visits = (int64_t*)malloc((size_t)n * sizeof(int64_t));
    if (!visits) {
        fprintf(stderr, "[BP1.3] bp_get_ev_visit_stats: OOM for %lld "
                "int64s — returning total_visited only\n", (long long)n);
        return 0;  /* partial fill: total_visited set, percentiles zero */
    }
    int64_t k = 0;
    for (int64_t i = 0; i < t->table_size; i++) {
        if (t->occupied[i] != 1) continue;
        const BPInfoSet *is = &t->sets[i];
        if (is->ev_visit_count > 0) {
            visits[k++] = (int64_t)is->ev_visit_count;
            if (is->ev_visit_count < 5) out->below_5++;
            if (is->ev_visit_count < 100) out->below_100++;
            if (is->ev_visit_count >= 1000) out->above_1000++;
        }
    }

    qsort(visits, (size_t)n, sizeof(int64_t), cmp_int64);
    out->min_visits = visits[0];
    out->max_visits = visits[n - 1];
    out->p10_visits = visits[(int64_t)((double)n * 0.10)];
    out->p50_visits = visits[(int64_t)((double)n * 0.50)];
    out->p90_visits = visits[(int64_t)((double)n * 0.90)];
    out->p99_visits = visits[(int64_t)((double)n * 0.99)];
    free(visits);
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
