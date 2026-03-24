/**
 * mccfr_blueprint.h — N-player external-sampling MCCFR for blueprint computation
 *
 * Implements the exact Pluribus blueprint training algorithm:
 *   - N-player (2-6) external-sampling MCCFR
 *   - Multi-street: flop → turn → river → showdown
 *   - Linear CFR: regrets *= t/(t+1), strategy sum weighted by t
 *   - One traverser per iteration, cycles through all N players
 *   - Non-traverser actions: sampled according to current strategy
 *   - Chance nodes: sampled (one card per deal)
 *   - Info sets indexed by (board_hash, node_index, street)
 *
 * This is a CPU solver designed for offline precompute (EC2).
 * Memory scales linearly with info sets visited (not tree size).
 *
 * Pluribus used this with:
 *   - 64 cores, ~8 days (12,400 CPU-hours)
 *   - 200-bucket card abstraction (we use exact hands, limited to 200)
 *   - External sampling over actions and chance
 */
#ifndef MCCFR_BLUEPRINT_H
#define MCCFR_BLUEPRINT_H

#include <stdint.h>

#define BP_MAX_PLAYERS    6
#define BP_MAX_HANDS      200    /* per player */
#define BP_MAX_ACTIONS    8
#define BP_MAX_BOARD      5

/* Info set key: uniquely identifies a decision point across iterations.
 * (player, board_hash, betting_sequence_hash) → info set */
typedef struct {
    int player;
    uint64_t board_hash;     /* hash of visible board cards */
    uint64_t action_hash;    /* hash of betting history at this node */
} BPInfoKey;

/* Info set data: regrets + strategy sums for one decision point */
typedef struct {
    int num_actions;
    int num_hands;
    float *regrets;           /* [num_actions * num_hands] */
    float *strategy_sum;      /* [num_actions * num_hands] */
    float *current_strategy;  /* [num_actions * num_hands] */
} BPInfoSet;

/* Hash table for info sets (open addressing) */
#define BP_HASH_SIZE (1 << 22)  /* 4M slots (~128MB) */

typedef struct {
    BPInfoKey *keys;
    BPInfoSet *sets;
    int *occupied;            /* 1 if slot is used */
    int num_entries;
} BPInfoTable;

/* Blueprint solver state */
typedef struct {
    /* Players */
    int num_players;
    int hands[BP_MAX_PLAYERS][BP_MAX_HANDS][2];
    float weights[BP_MAX_PLAYERS][BP_MAX_HANDS];
    int num_hands[BP_MAX_PLAYERS];

    /* Board */
    int flop[3];

    /* Bet sizing */
    float bet_sizes[BP_MAX_ACTIONS];
    int num_bet_sizes;

    /* Pot and stacks */
    int starting_pot;        /* pot at flop start (in chips, scale=100) */
    int effective_stack;     /* per player */

    /* Info set storage */
    BPInfoTable info_table;

    /* RNG state for sampling */
    uint64_t rng_state;

    /* Stats */
    int iterations_run;
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
 * Initialize the blueprint solver.
 *
 * Args:
 *   s: solver state (caller-allocated)
 *   num_players: 2-6
 *   flop: 3 flop cards
 *   hands/weights/num_hands: per-player hand ranges
 *   starting_pot, effective_stack: in chips (scale=100)
 *   bet_sizes, num_bet_sizes: bet fractions
 */
BP_EXPORT int bp_init(BPSolver *s, int num_players,
                      const int *flop,
                      const int hands[][BP_MAX_HANDS][2],
                      const float weights[][BP_MAX_HANDS],
                      const int *num_hands,
                      int starting_pot, int effective_stack,
                      const float *bet_sizes, int num_bet_sizes);

/**
 * Run external-sampling MCCFR for the specified iterations.
 * Each iteration: pick one traverser, sample opponent actions + chance.
 *
 * Returns 0 on success.
 */
BP_EXPORT int bp_solve(BPSolver *s, int max_iterations);

/**
 * Extract weighted-average strategy at a specific info set.
 *
 * Args:
 *   s: solved state
 *   player: which player's strategy
 *   board: current board cards
 *   num_board: 3 (flop), 4 (turn), 5 (river)
 *   action_seq: sequence of actions taken (indices into action list)
 *   seq_len: length of action_seq
 *   strategy_out: [num_actions] output, normalized weighted average
 *   hand_idx: which hand in this player's range
 *
 * Returns num_actions, or 0 if info set not found.
 */
BP_EXPORT int bp_get_strategy(const BPSolver *s, int player,
                               const int *board, int num_board,
                               const int *action_seq, int seq_len,
                               float *strategy_out, int hand_idx);

/**
 * Get the number of info sets created.
 */
BP_EXPORT int bp_num_info_sets(const BPSolver *s);

/**
 * Free solver resources.
 */
BP_EXPORT void bp_free(BPSolver *s);

#ifdef __cplusplus
}
#endif

#endif /* MCCFR_BLUEPRINT_H */
