/**
 * mccfr_blueprint.h — Production N-player external-sampling MCCFR
 *
 * Matches the exact Pluribus blueprint training algorithm:
 *   - N-player (2-6) external-sampling MCCFR
 *   - Multi-street: flop -> turn -> river -> showdown
 *   - Linear CFR discount: every DISCOUNT_INTERVAL iters for first DISCOUNT_PERIOD iters
 *   - Regret-based pruning: skip actions with regret < PRUNE_THRESHOLD (95% of iters)
 *   - Integer regrets (int32) with floor at REGRET_FLOOR
 *   - Card abstraction: hand_to_bucket mapping (200 buckets per street)
 *   - OpenMP parallel iterations with thread-local RNG (Hogwild-style)
 *   - Strategy snapshots to disk (not continuous accumulation for rounds 2-4)
 *
 * Pluribus parameters (from supplementary):
 *   - 64 cores, 8 days, 12400 CPU-hours, <512GB RAM
 *   - Discount: d = (T/interval)/(T/interval+1) every 10 min for first 400 min
 *   - Pruning: after 200 min, threshold -300M, 95% of iterations
 *   - Regret floor: -310M
 *   - 665M action sequences, 414M encountered
 *   - Regrets as int32, lazy allocation
 */
#ifndef MCCFR_BLUEPRINT_H
#define MCCFR_BLUEPRINT_H

#include <stdint.h>
#include <stddef.h>

/* ── Limits ──────────────────────────────────────────────────────────── */

#define BP_MAX_PLAYERS    6
#define BP_MAX_HANDS      1326   /* max hands per player (all possible 2-card combos) */
#define BP_MAX_ACTIONS    16
#define BP_MAX_BOARD      5

/* Hash table sizing.
 * Per-texture: ~5-50M info sets depending on player count and bet sizes.
 * BP_HASH_SIZE_MEDIUM (64M slots) fits ~45M info sets at 70% load.
 * Memory: 64M * (key=20 + set=16 + occupied=4) = ~2.5GB metadata.
 * For small test runs, use BP_HASH_SIZE_SMALL. */
#define BP_HASH_SIZE_LARGE  (1 << 30)   /* 1.07B slots (~44GB metadata) — bucket-in-key needs more */
#define BP_HASH_SIZE_MEDIUM (1 << 26)   /* 64M slots (~2.5GB metadata) — per-texture */
#define BP_HASH_SIZE_SMALL  (1 << 22)   /* 4M slots (~160MB metadata) — testing only */

/* Pluribus-matched constants */
#define BP_REGRET_FLOOR      (-310000000)   /* minimum regret per action */
#define BP_REGRET_CEILING     310000000     /* maximum regret per action (prevents int32 overflow) */
#define BP_PRUNE_THRESHOLD   (-300000000)   /* skip actions below this */
#define BP_PRUNE_PROB        0.95f          /* fraction of iters that prune */

/* ── Data structures ─────────────────────────────────────────────────── */

/* Info set key — includes bucket (Pluribus-style: one info set per bucket).
 * This means each hash table slot stores regrets[num_actions] only,
 * reducing memory ~200x vs storing regrets[num_actions * num_buckets]. */
typedef struct {
    int player;
    int street;              /* 0=preflop, 1=flop, 2=turn, 3=river */
    int bucket;              /* card abstraction bucket index */
    uint64_t board_hash;
    uint64_t action_hash;
} BPInfoKey;

/* Info set data — integer regrets for memory efficiency.
 * Pluribus stores regrets as int32, strategies derived from regrets.
 * With bucket-in-key, each info set stores regrets for ONE bucket only.
 * strategy_sum is only maintained for round 1 (preflop).
 * For rounds 2-4, snapshots of current strategy are saved to disk. */
typedef struct {
    int num_actions;
    int *regrets;          /* [num_actions], int32 */
    float *strategy_sum;   /* [num_actions] or NULL (lazy) */
} BPInfoSet;

/* Hash table (open addressing, linear probing) */
typedef struct {
    BPInfoKey *keys;
    BPInfoSet *sets;
    int *occupied;
    int num_entries;
    int table_size;        /* actual hash table size (configurable) */
} BPInfoTable;

/* Blueprint solver configuration */
typedef struct {
    /* Timing-based thresholds (converted to iteration counts).
     * Pluribus used wall-clock minutes; we convert to iterations
     * based on estimated iterations/minute on the target hardware.
     * Uses int64 because full Pluribus-equivalent compute on fast hardware
     * requires ~100B+ iterations, exceeding INT32_MAX. */
    int64_t discount_stop_iter;     /* stop Linear CFR discount after this (Pluribus: 400 min) */
    int64_t discount_interval;      /* apply discount every N iterations (Pluribus: ~10 min) */
    int64_t prune_start_iter;       /* start pruning after this (Pluribus: 200 min) */
    int64_t snapshot_start_iter;    /* start saving snapshots after this (Pluribus: 800 min) */
    int64_t snapshot_interval;      /* save snapshot every N iterations (Pluribus: ~200 min) */
    int64_t strategy_interval;      /* update round-1 avg strategy every N iters (Pluribus: 10K) */
    int num_threads;            /* OpenMP threads (0 = auto) */
    int hash_table_size;        /* 0 = auto (BP_HASH_SIZE_SMALL or _LARGE based on players) */
    const char *snapshot_dir;   /* directory for strategy snapshots (NULL = no snapshots) */
    int include_preflop;        /* 1 = start from preflop (unified Pluribus-style), 0 = start from flop */
    int postflop_num_buckets;   /* 0 = default (200). Reduce to shrink game tree. */
} BPConfig;

/* Blueprint solver state */
typedef struct {
    /* Players */
    int num_players;
    int hands[BP_MAX_PLAYERS][BP_MAX_HANDS][2]; /* actual cards */
    float weights[BP_MAX_PLAYERS][BP_MAX_HANDS];
    int num_hands[BP_MAX_PLAYERS];

    /* Card abstraction: hand -> bucket mapping per street.
     * bucket_map[street][player][hand_idx] = bucket index.
     * street: 0=preflop, 1=flop, 2=turn, 3=river.
     * When use_buckets=0, bucket = hand_idx (identity mapping). */
    int use_buckets;
    int bucket_map[4][BP_MAX_PLAYERS][BP_MAX_HANDS];
    int num_buckets[4][BP_MAX_PLAYERS]; /* buckets per street per player */

    /* Board */
    int flop[3];

    /* Bet sizing (postflop first raise: Pluribus turn/river = 0.5x, 1x, all-in) */
    float bet_sizes[BP_MAX_ACTIONS];
    int num_bet_sizes;

    /* Postflop subsequent raises (Pluribus turn/river = 1x, all-in) */
    float subsequent_bet_sizes[BP_MAX_ACTIONS];
    int num_subsequent_bet_sizes;

    /* Preflop bet sizing (Pluribus: 1-14 sizes per decision point) */
    float preflop_bet_sizes[BP_MAX_ACTIONS];
    int num_preflop_bet_sizes;

    /* Pot, stacks, and blinds */
    int starting_pot;       /* pot at start of postflop (when include_preflop=0) */
    int effective_stack;    /* stack at start of postflop (when include_preflop=0) */
    int small_blind;        /* SB amount in chips (e.g. 50 for $50/$100) */
    int big_blind;          /* BB amount in chips (e.g. 100) */
    int initial_stack;      /* starting stack per player (e.g. 10000 for 100BB) */

    /* Info set storage */
    BPInfoTable info_table;

    /* Config */
    BPConfig config;

    /* RNG state — one per thread for OpenMP */
    uint64_t *rng_states;    /* [num_threads] */
    int num_rng_states;

    /* Postflop bucket config for unified solver.
     * When include_preflop=1, postflop buckets are precomputed for all 1,755
     * flop textures at init time. During traversal, the canonical flop hash
     * is used to look up the precomputed 200-bucket mapping. */
    int postflop_num_buckets;  /* target buckets per street (Pluribus: 200) */
    int postflop_ehs_samples;  /* MC samples for EHS precompute (Pluribus: ~500) */

    /* Precomputed flop texture bucket cache.
     * Key: canonical flop hash (from 3 sorted ranks + suit pattern).
     * Value: bucket_map[1326] for that texture.
     * Allocated at init, freed by bp_free. */
    #define BP_MAX_TEXTURES 1760
    int *texture_bucket_cache;     /* [BP_MAX_TEXTURES * BP_MAX_HANDS] flat array */
    uint64_t texture_hash_keys[BP_MAX_TEXTURES];  /* hash key per texture */
    int num_cached_textures;

    /* Stats */
    int64_t iterations_run;
    int64_t snapshots_saved;
} BPSolver;

/* ── Public API ──────────────────────────────────────────────────────── */

#ifdef _WIN32
#define BP_EXPORT __declspec(dllexport)
#else
#define BP_EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initialize with default Pluribus-matched config.
 */
BP_EXPORT void bp_default_config(BPConfig *config);

/**
 * Initialize the blueprint solver.
 */
BP_EXPORT int bp_init(BPSolver *s, int num_players,
                      const int *flop,
                      const int hands[][BP_MAX_HANDS][2],
                      const float weights[][BP_MAX_HANDS],
                      const int *num_hands,
                      int starting_pot, int effective_stack,
                      const float *bet_sizes, int num_bet_sizes);

/**
 * Initialize with explicit config.
 */
BP_EXPORT int bp_init_ex(BPSolver *s, int num_players,
                          const int *flop,
                          const int hands[][BP_MAX_HANDS][2],
                          const float weights[][BP_MAX_HANDS],
                          const int *num_hands,
                          int starting_pot, int effective_stack,
                          const float *bet_sizes, int num_bet_sizes,
                          const BPConfig *config);

/**
 * Set card abstraction buckets for a given street.
 * Must be called after bp_init, before bp_solve.
 *
 * street: 0=preflop, 1=flop, 2=turn, 3=river
 * bucket_map[player][hand_idx] = bucket index
 * num_buckets[player] = number of buckets for this player
 */
BP_EXPORT int bp_set_buckets(BPSolver *s, int street,
                              const int bucket_map[][BP_MAX_HANDS],
                              const int *num_buckets);

/**
 * Run external-sampling MCCFR with Pluribus optimizations.
 * Supports OpenMP parallelism, pruning, Linear CFR discount.
 */
BP_EXPORT int bp_solve(BPSolver *s, int max_iterations);

/**
 * Extract weighted-average strategy at a specific info set.
 */
BP_EXPORT int bp_get_strategy(const BPSolver *s, int player,
                               const int *board, int num_board,
                               const int *action_seq, int seq_len,
                               float *strategy_out, int bucket);

BP_EXPORT int bp_num_info_sets(const BPSolver *s);
BP_EXPORT void bp_free(BPSolver *s);

/**
 * Save/load precomputed texture bucket cache to skip the 65-90 min
 * precompute on subsequent launches. Format: TXC1 header + hash keys
 * + bucket mappings. ~9.3 MB file for 1755 textures × 1326 hands.
 */
BP_EXPORT int bp_save_texture_cache(const BPSolver *s, const char *path);
BP_EXPORT int bp_load_texture_cache(BPSolver *s, const char *path);

/**
 * Initialize a unified preflop-through-river solver (Pluribus-style).
 *
 * All 6 players get all 1326 hands. Preflop uses 169 lossless classes.
 * Postflop uses card abstraction via bp_set_buckets().
 *
 * Traversal starts from preflop with SB/BB posting, then UTG→MP→CO→BTN→SB→BB
 * acting in order. After preflop round, 3 flop cards are dealt, then
 * flop→turn→river betting rounds proceed as in the standard postflop solver.
 *
 * This matches Pluribus Algorithm 1 exactly: one unified MCCFR solve.
 *
 * Args:
 *   s: solver state (caller allocates)
 *   num_players: 6 for full table (2-6 supported)
 *   small_blind, big_blind: blind amounts in chips (e.g. 50, 100)
 *   initial_stack: starting stack per player in chips (e.g. 10000 for 100BB)
 *   postflop_bet_sizes, num_postflop_bet_sizes: bet fractions for flop/turn/river
 *   preflop_bet_sizes, num_preflop_bet_sizes: bet fractions for preflop raises
 *   config: solver configuration
 */
BP_EXPORT int bp_init_unified(BPSolver *s, int num_players,
                                int small_blind, int big_blind, int initial_stack,
                                const float *postflop_bet_sizes, int num_postflop_bet_sizes,
                                const float *preflop_bet_sizes, int num_preflop_bet_sizes,
                                const BPConfig *config);

/**
 * Export ALL info set strategies to a binary buffer.
 *
 * For each occupied slot in the hash table, writes:
 *   BPInfoKey (player, board_hash, action_hash)  — 20 bytes
 *   uint8 num_actions                             — 1 byte
 *   uint8 num_hands                               — 1 byte (actually num_buckets)
 *   uint8[num_actions * num_hands] strategy       — quantized to 0-255
 *
 * Strategies are derived from regrets via regret matching (not raw regrets).
 * Quantization: strategy[a] * 255, rounded, clamped to [0, 255].
 *
 * Args:
 *   s: solved state
 *   buf: output buffer (must be pre-allocated)
 *   buf_size: size of buf in bytes
 *   bytes_written: output, actual bytes written
 *
 * Returns 0 on success, -1 if buffer too small.
 * Call with buf=NULL to query required buffer size via bytes_written.
 */
BP_EXPORT int bp_export_strategies(const BPSolver *s,
                                    unsigned char *buf, size_t buf_size,
                                    size_t *bytes_written);

/**
 * Save the full regret table to a binary file for checkpoint/resume.
 *
 * Writes all occupied hash table slots (keys + regrets + strategy_sum)
 * to a binary file that can be reloaded via bp_load_regrets().
 *
 * Format:
 *   Header: "BPRG" (4B) + table_size (4B) + num_entries (4B) + iterations_run (4B)
 *   For each occupied slot:
 *     BPInfoKey (player 4B, street 4B, board_hash 8B, action_hash 8B) = 24B
 *     num_actions (4B), num_hands (4B) = 8B
 *     regrets[num_actions * num_hands] (int32) = variable
 *     has_strategy_sum (4B)
 *     if has_strategy_sum: strategy_sum[num_actions * num_hands] (float32) = variable
 *
 * Returns 0 on success, -1 on write error.
 */
BP_EXPORT int bp_save_regrets(const BPSolver *s, const char *path);

/**
 * Load a previously saved regret table from a checkpoint file.
 *
 * Must be called AFTER bp_init_unified or bp_init_ex (hash table must exist).
 * Loaded entries are inserted into the existing hash table. If the table is
 * too small for the saved entries, excess entries are silently dropped.
 *
 * Returns number of entries loaded, or -1 on error.
 */
BP_EXPORT int bp_load_regrets(BPSolver *s, const char *path);

/**
 * Export EHS values and bucket assignments for all hands.
 *
 * Writes per-hand: float32 EHS, int32 bucket_index
 * Total: num_hands * 8 bytes
 *
 * Args:
 *   s: solver state (after bp_set_buckets)
 *   street: which street's bucketing (1=flop, 2=turn, 3=river)
 *   player: which player
 *   bucket_out: [num_hands] bucket assignments
 *
 * Returns num_hands.
 */
BP_EXPORT int bp_export_buckets(const BPSolver *s, int street, int player,
                                 int *bucket_out);

#ifdef __cplusplus
}
#endif

#endif /* MCCFR_BLUEPRINT_H */
