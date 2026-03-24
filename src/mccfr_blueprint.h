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

/* ── Limits ──────────────────────────────────────────────────────────── */

#define BP_MAX_PLAYERS    6
#define BP_MAX_HANDS      1326   /* max hands per player (all possible 2-card combos) */
#define BP_MAX_ACTIONS    8
#define BP_MAX_BOARD      5

/* Hash table sizing.
 * Per-texture: ~5-50M info sets depending on player count and bet sizes.
 * BP_HASH_SIZE_MEDIUM (64M slots) fits ~45M info sets at 70% load.
 * Memory: 64M * (key=20 + set=16 + occupied=4) = ~2.5GB metadata.
 * For small test runs, use BP_HASH_SIZE_SMALL. */
#define BP_HASH_SIZE_LARGE  (1 << 29)   /* 536M slots (~21GB metadata) — full Pluribus */
#define BP_HASH_SIZE_MEDIUM (1 << 26)   /* 64M slots (~2.5GB metadata) — per-texture */
#define BP_HASH_SIZE_SMALL  (1 << 22)   /* 4M slots (~160MB metadata) — testing only */

/* Pluribus-matched constants */
#define BP_REGRET_FLOOR      (-310000000)   /* minimum regret per action */
#define BP_PRUNE_THRESHOLD   (-300000000)   /* skip actions below this */
#define BP_PRUNE_PROB        0.95f          /* fraction of iters that prune */

/* ── Data structures ─────────────────────────────────────────────────── */

/* Info set key */
typedef struct {
    int player;
    uint64_t board_hash;
    uint64_t action_hash;
} BPInfoKey;

/* Info set data — integer regrets for memory efficiency.
 * Pluribus stores regrets as int32, strategies derived from regrets.
 * strategy_sum is only maintained for round 1 (preflop).
 * For rounds 2-4, snapshots of current strategy are saved to disk. */
typedef struct {
    int num_actions;
    int num_hands;         /* actually num_buckets when using abstraction */
    int *regrets;          /* [num_actions * num_hands], int32 */
    float *strategy_sum;   /* [num_actions * num_hands] or NULL (lazy) */
    float *current_strategy; /* [num_actions * num_hands], transient */
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
     * Default: ~1000 iter/min on 64 cores -> 400 min = 400K iters. */
    int discount_stop_iter;     /* stop Linear CFR discount after this (Pluribus: 400 min) */
    int discount_interval;      /* apply discount every N iterations (Pluribus: ~10 min) */
    int prune_start_iter;       /* start pruning after this (Pluribus: 200 min) */
    int snapshot_start_iter;    /* start saving snapshots after this (Pluribus: 800 min) */
    int snapshot_interval;      /* save snapshot every N iterations (Pluribus: ~200 min) */
    int strategy_interval;      /* update round-1 avg strategy every N iters (Pluribus: 10K) */
    int num_threads;            /* OpenMP threads (0 = auto) */
    int hash_table_size;        /* 0 = auto (BP_HASH_SIZE_SMALL or _LARGE based on players) */
    const char *snapshot_dir;   /* directory for strategy snapshots (NULL = no snapshots) */
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

    /* Bet sizing */
    float bet_sizes[BP_MAX_ACTIONS];
    int num_bet_sizes;

    /* Pot and stacks */
    int starting_pot;
    int effective_stack;

    /* Info set storage */
    BPInfoTable info_table;

    /* Config */
    BPConfig config;

    /* RNG state — one per thread for OpenMP */
    uint64_t *rng_states;    /* [num_threads] */
    int num_rng_states;

    /* Stats */
    int iterations_run;
    int snapshots_saved;
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
                               float *strategy_out, int hand_idx);

BP_EXPORT int bp_num_info_sets(const BPSolver *s);
BP_EXPORT void bp_free(BPSolver *s);

#ifdef __cplusplus
}
#endif

#endif /* MCCFR_BLUEPRINT_H */
