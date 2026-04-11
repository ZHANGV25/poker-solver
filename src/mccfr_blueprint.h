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
#define BP_HASH_SIZE_3B     ((int64_t)3000000000) /* 3B slots (~180GB metadata) — full unified solve */
#define BP_HASH_SIZE_LARGE  (1 << 30)   /* 1.07B slots (~44GB metadata) — bucket-in-key needs more */
#define BP_HASH_SIZE_MEDIUM (1 << 26)   /* 64M slots (~2.5GB metadata) — per-texture */
#define BP_HASH_SIZE_SMALL  (1 << 22)   /* 4M slots (~160MB metadata) — testing only */

/* Pluribus-matched constants */
#define BP_REGRET_FLOOR      (-310000000)   /* minimum regret per action */
#define BP_REGRET_CEILING     2000000000    /* ~int32 max. Pluribus has no explicit ceiling (only
                                            * the implicit int32 max). The old 310M ceiling caused
                                            * dominant actions to saturate 7x too early, losing
                                            * ordering information when multiple actions competed.
                                            * int64 intermediate arithmetic prevents overflow. */
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
 *
 * strategy_sum lifecycle (NOTE: this differs from Pluribus — see SOLVER_CONFIG.md §10):
 *   - Round 1: accumulated continuously by traverse() at mccfr_blueprint.c:1283-1289
 *     on every traverser visit (full regret-matched distribution added).
 *   - Rounds 2-4: accumulated in-memory by accumulate_snapshot() at
 *     mccfr_blueprint.c:1343-1358, called from bp_solve at snapshot barriers.
 *     Note that accumulate_snapshot does NOT filter by street, so round 1 also
 *     gets snapshot-accumulated in addition to per-visit accumulation.
 *
 * Pluribus uses a sparse sampled UPDATE-STRATEGY for round 1 (every 10K iters,
 * one path sampled, single phi[I,a] += 1) and on-disk snapshots for rounds 2-4
 * to halve memory. We use full-distribution per-visit + in-memory snapshots,
 * which is algorithmically defensible (lower variance, more memory). */
typedef struct {
    int num_actions;
    int *regrets;          /* [num_actions], int32 */
    float *strategy_sum;   /* [num_actions] or NULL (lazy) */
    float *action_evs;     /* [num_actions] or NULL (populated by bp_compute_action_evs).
                            * Phase 1.3: sum of per-action expected values observed during
                            * the σ̄-sampled EV walk. Average EV = action_evs[a] / ev_visit_count.
                            * See docs/PHASE_1_3_DESIGN.md. Arena-allocated lazily on first
                            * traverser visit during the EV walk. NULL until then. */
    int ev_visit_count;    /* Phase 1.3: number of traverser visits that contributed to
                            * action_evs during the EV walk. Used to normalize the
                            * accumulated EVs into averages at export time. Relaxed atomic. */
} BPInfoSet;

/* Hash table (open addressing, linear probing) */
typedef struct {
    BPInfoKey *keys;
    BPInfoSet *sets;
    int *occupied;
    int64_t num_entries;
    int64_t table_size;             /* actual hash table size (supports up to 4B+) */
    int64_t insertion_failures;     /* atomic, lifetime count of probe-cap-exceeded returns */
    int64_t max_probe_observed;     /* atomic, lifetime maximum probe distance */
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
    /* DEPRECATED — F2 fix in v3 (see V3_PLAN.md Phase 2.8). This field is set
     * by ~30 callers but never read by mccfr_blueprint.c. The original Pluribus
     * design uses Strategy Interval = 10K to gate sampled UPDATE-STRATEGY calls
     * for round 1; our implementation accumulates strategy_sum on every visit
     * (see SOLVER_CONFIG.md §10), so this field is dead code. Kept in the
     * struct for ABI safety with the 30+ ctypes callers; ignore in new code. */
    int64_t strategy_interval;
    int num_threads;            /* OpenMP threads (0 = auto) */
    int64_t hash_table_size;    /* 0 = auto. Supports >2B (e.g. 3B for 376GB instance) */
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

    /* Tiered preflop sizing: per-raise-level arrays (Pluribus-style).
     * Level 0 = open raise, 1 = 3-bet, 2 = 4-bet, 3 = 5-bet.
     * If num_preflop_tiers > 0, these override preflop_bet_sizes. */
    float preflop_tiered_sizes[4][BP_MAX_ACTIONS];
    int num_preflop_tiered_sizes[4];  /* sizes per level */
    int num_preflop_tiers;            /* 0 = use flat preflop_bet_sizes */
    int preflop_max_raises;           /* 0 = default (4) */

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
    /* Bug 7 fix: open-addressing hashmap on top of texture_hash_keys[].
     * Sized to next power of 2 above 2 * BP_MAX_TEXTURES → 4096 buckets,
     * load factor ≤43% so probe chains stay under 3-4 on average. The hash
     * mixer is splitmix64-style. Replaces the previous O(N) linear scan
     * (~1755 comparisons per flop deal). Populated by
     * `texture_index_rebuild()` after the texture cache is loaded or
     * precomputed. */
    #define BP_TEXTURE_INDEX_SIZE 4096
    int *texture_bucket_cache;     /* [BP_MAX_TEXTURES * BP_MAX_HANDS] flat array */
    uint64_t texture_hash_keys[BP_MAX_TEXTURES];  /* hash key per texture */
    int texture_hash_index[BP_TEXTURE_INDEX_SIZE]; /* open-addressed: -1 = empty, else idx into texture_hash_keys */
    int num_cached_textures;

    /* Precomputed turn k-means centroids for bucketing.
     * 200 centroids in [EHS, PPot, NPot] feature space.
     * During traversal, each hand's features are computed inline and
     * mapped to nearest centroid. Matches Pluribus bucketing approach. */
    float turn_centroids[200][3];
    int turn_centroids_k;

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
 * Set tiered preflop bet sizes (Pluribus-style: fewer sizes at higher raise levels).
 * level: 0 = open raise, 1 = 3-bet, 2 = 4-bet, 3 = 5-bet.
 * Call once per level after bp_init_unified, before bp_solve.
 * max_raises: total preflop raise cap (set on first call, e.g. 4).
 */
BP_EXPORT int bp_set_preflop_tier(BPSolver *s, int level,
                                   const float *sizes, int num_sizes,
                                   int max_raises);

/**
 * Run external-sampling MCCFR with Pluribus optimizations.
 * Supports OpenMP parallelism, pruning, Linear CFR discount.
 */
BP_EXPORT int bp_solve(BPSolver *s, int64_t max_iterations);

/**
 * Extract weighted-average strategy at a specific info set.
 */
BP_EXPORT int bp_get_strategy(const BPSolver *s, int player,
                               const int *board, int num_board,
                               const int *action_seq, int seq_len,
                               float *strategy_out, int bucket);

/**
 * Extract raw integer regrets at a specific info set.
 * regrets_out must have space for at least BP_MAX_ACTIONS ints.
 * Returns num_actions, or 0 if info set not found.
 */
BP_EXPORT int bp_get_regrets(const BPSolver *s, int player,
                              const int *board, int num_board,
                              const int *action_seq, int seq_len,
                              int *regrets_out, int bucket);

BP_EXPORT int64_t bp_num_info_sets(const BPSolver *s);
BP_EXPORT void bp_free(BPSolver *s);

/**
 * Hash table health stats. All four out-pointers may be NULL to skip.
 * Safe to call concurrently with bp_solve (counters use relaxed atomics).
 *
 * insertion_failures: lifetime count of info_table_find_or_create calls that
 *   exhausted the 4096-probe cap and silently returned -1. Should always be 0
 *   in a properly-sized table. Nonzero indicates strategy_sum corruption.
 * max_probe_observed: lifetime maximum probe distance for any successful
 *   insertion. Approaching 4096 indicates the table is too full.
 */
BP_EXPORT void bp_get_table_stats(const BPSolver *s,
                                   int64_t *out_entries,
                                   int64_t *out_table_size,
                                   int64_t *out_insertion_failures,
                                   int64_t *out_max_probe_observed);

/**
 * Enable the legacy boost-style hash_combine mixer. Must be called BEFORE
 * bp_load_regrets when loading a v2 checkpoint (trained before commit
 * 48da71b on 2026-04-08 01:51 UTC). v2's stored action_hash values were
 * computed with the boost-style mixer; if Phase 1.3's traverse_ev runs
 * with splitmix64, every non-root info set lookup fails because the
 * recomputed action_hashes don't match the stored ones.
 */
BP_EXPORT void bp_set_legacy_hash_mixer(int enabled);

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
BP_EXPORT int64_t bp_load_regrets(BPSolver *s, const char *path);

/**
 * Phase 1.3: Compute per-action expected values under the average strategy σ̄.
 *
 * Runs num_iterations of a modified MCCFR traversal where:
 *   - Strategies at all decision nodes are sampled from strategy_sum (the
 *     average strategy), not regret-matched from current regrets.
 *   - At traverser nodes, per-action EVs `action_values[a]` are accumulated
 *     into `is->action_evs[a]` and `is->ev_visit_count` is incremented.
 *   - Regrets and strategy_sum are NOT updated (read-only with respect to
 *     the underlying blueprint).
 *
 * After this runs, `action_evs[a] / ev_visit_count` is an unbiased estimator
 * of `v̄(I, a)` — the counterfactual value of action `a` at info set `I`
 * under the average strategy. See docs/PHASE_1_3_DESIGN.md for the math.
 *
 * Must be called AFTER bp_load_regrets or bp_solve so that strategy_sum is
 * populated. Must be called BEFORE bp_export_action_evs.
 *
 * Multi-threaded via OpenMP (same thread count as bp_solve).
 *
 * Returns 0 on success.
 */
BP_EXPORT int bp_compute_action_evs(BPSolver *s, int64_t num_iterations);

/**
 * Phase 1.3: Export per-action EVs to a binary buffer.
 *
 * For each occupied slot with ev_visit_count > 0, writes:
 *   player(1) + street(1) + bucket(2) + action_hash(8) + num_actions(1)
 *   + float32[num_actions] avg_ev (= action_evs[a] / ev_visit_count)
 *
 * Header: "BPR3" (4) + u32 num_entries
 *
 * Info sets with ev_visit_count == 0 are skipped. Consumer should fall back
 * to equity approximation at those info sets.
 *
 * Args:
 *   s: solver state (after bp_compute_action_evs)
 *   buf: output buffer (pre-allocated; pass NULL to query required size)
 *   buf_size: size of buf in bytes
 *   bytes_written: output, actual bytes written (or required size if buf=NULL)
 *
 * Returns 0 on success, -1 if buffer too small.
 */
BP_EXPORT int bp_export_action_evs(const BPSolver *s,
                                    unsigned char *buf, size_t buf_size,
                                    size_t *bytes_written);

/**
 * Phase 1.3: Aggregate visit count statistics across all info sets that
 * the EV walk touched (ev_visit_count > 0).
 *
 * The exported BPR3 section only writes the NORMALIZED average EV per
 * action, not the raw visit count. Downstream sentinels need the visit
 * distribution to judge EV confidence and flag low-traffic info sets.
 * This function reports the aggregate stats cheaply at export time.
 *
 * Fills the caller-provided struct with:
 *   total_visited: number of info sets with ev_visit_count > 0
 *   min/p10/p50/p90/p99/max: percentile visit counts across those sets
 *   below_5: count with ev_visit_count < 5 (low-confidence)
 *   below_100: count with ev_visit_count < 100
 *   above_1000: count with ev_visit_count >= 1000 (high-confidence)
 *
 * Safe to call after bp_compute_action_evs. Iterates the hash table once.
 */
typedef struct {
    int64_t total_visited;
    int64_t min_visits;
    int64_t p10_visits;
    int64_t p50_visits;
    int64_t p90_visits;
    int64_t p99_visits;
    int64_t max_visits;
    int64_t below_5;
    int64_t below_100;
    int64_t above_1000;
} BPEVVisitStats;

BP_EXPORT int bp_get_ev_visit_stats(const BPSolver *s, BPEVVisitStats *out);

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
