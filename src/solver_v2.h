/**
 * solver_v2.h — Pluribus-style depth-limited DCFR solver
 *
 * Key changes from v1:
 *   - Linear CFR (alpha=1, beta=1, gamma=1) matching Pluribus
 *   - Final iteration strategy (not average)
 *   - 4 continuation strategies at leaf nodes
 *   - On-the-fly river blueprint evaluation at leaves
 *   - Precomputed river strengths for all runouts
 */
#ifndef SOLVER_V2_H
#define SOLVER_V2_H

#include <stdint.h>

#define MAX_ACTIONS_V2    8
#define MAX_HANDS_V2   1326
#define MAX_BOARD_V2      5
#define MAX_RAISES_V2     2
#define NUM_CONT_STRATS   4  /* unmodified, fold-biased, call-biased, raise-biased */

/* Node types */
#define NODE_V2_DECISION   0
#define NODE_V2_FOLD       1
#define NODE_V2_SHOWDOWN   2
#define NODE_V2_LEAF       3  /* depth-limited: 4 continuation strategies */

/* ── Tree node ─────────────────────────────────────────────────────────── */

typedef struct {
    int type;
    int player;          /* 0=OOP, 1=IP, -1=terminal */
    int num_actions;
    int children[MAX_ACTIONS_V2];
    int pot;             /* pot in chips (scale=100) */
    int bets[2];         /* each player's total bet this street */
} NodeV2;

/* ── Per-info-set data ─────────────────────────────────────────────────── */

typedef struct {
    int num_actions;
    int num_hands;
    float *regrets;           /* [num_actions * num_hands] */
    float *strategy_sum;      /* [num_actions * num_hands] for Linear CFR weighting */
    float *current_strategy;  /* [num_actions * num_hands] last iteration */
} InfoSetV2;

/* ── Precomputed river data for leaf evaluation ────────────────────────── */

typedef struct {
    uint32_t **strengths;  /* [num_rivers][num_hands] eval7 results */
    int *river_cards;      /* [num_rivers] card indices */
    int num_rivers;
    int num_hands;
} RiverStrengthTable;

/* ── Continuation strategy for leaf nodes ──────────────────────────────── */

typedef struct {
    /* Per-hand action frequencies from the blueprint for the next street.
     * blueprint_freqs[hand_idx * num_actions + action_idx] = P(action|hand)
     * This is used to generate the 4 biased variants. */
    float *blueprint_freqs;   /* [num_hands * num_actions] or NULL */
    int num_actions;          /* number of actions in blueprint */
    int num_hands;
} ContinuationData;

/* ── Solver state ──────────────────────────────────────────────────────── */

typedef struct {
    /* Board */
    int board[MAX_BOARD_V2];
    int num_board;

    /* Ranges */
    int hands[2][MAX_HANDS_V2][2];   /* [player][hand_idx][card0,card1] */
    float weights[2][MAX_HANDS_V2];  /* initial weights (Bayesian-narrowed) */
    int num_hands[2];

    /* Game tree */
    NodeV2 *nodes;
    int num_nodes;

    /* Info sets (one per decision node) */
    InfoSetV2 *info_sets;

    /* Precomputed hand strengths (for showdown nodes) */
    uint32_t *hand_strengths[2];

    /* Precomputed river strengths for ALL runouts (for leaf evaluation) */
    /* Only populated for turn solves (num_board == 4) */
    RiverStrengthTable river_table[2];

    /* Continuation data for leaf evaluation */
    ContinuationData *cont_data;  /* per leaf node, or NULL */
    int num_leaves;
    int *leaf_indices;  /* node indices that are leaf type */

    /* Leaf value cache: [leaf_idx][strategy_k][hand_idx] */
    /* Precomputed once, reused across DCFR iterations */
    float **leaf_values;  /* [num_leaves * NUM_CONT_STRATS][num_hands] */

    /* Bet sizing */
    float bet_sizes[MAX_ACTIONS_V2];
    int num_bet_sizes;

    /* Pot and stacks */
    int starting_pot;
    int effective_stack;

    /* Solver state */
    int iterations_run;
    float exploitability;
} SolverV2;

/* ── Public API ────────────────────────────────────────────────────────── */

/**
 * Initialize solver for a single-street subgame.
 */
int sv2_init(SolverV2 *s,
             const int *board, int num_board,
             const int hands0[][2], const float *weights0, int num_hands0,
             const int hands1[][2], const float *weights1, int num_hands1,
             int starting_pot, int effective_stack,
             const float *bet_sizes, int num_bet_sizes);

/**
 * Precompute river strength tables for all possible river cards.
 * Call after sv2_init for turn solves (num_board == 4).
 * This is the key optimization: compute once, reuse across iterations.
 */
int sv2_precompute_river_strengths(SolverV2 *s);

/**
 * Precompute leaf values for all 4 continuation strategies.
 * Uses the river strength tables to evaluate each strategy variant
 * against the current narrowed ranges. Must be called before solving.
 *
 * If blueprint_freqs is NULL, uses equity-based leaf values (fallback).
 */
int sv2_compute_leaf_values(SolverV2 *s);

/**
 * Run Linear CFR (Pluribus-style DCFR with alpha=beta=gamma=1).
 * Returns after max_iterations or when exploitability < target.
 */
float sv2_solve(SolverV2 *s, int max_iterations, float target_exploitability);

/**
 * Get the FINAL ITERATION strategy for a specific hand.
 * This is what Pluribus uses (not average strategy).
 */
void sv2_get_strategy(const SolverV2 *s, int player, int hand_idx,
                      float *strategy_out);

/**
 * Get final iteration strategies for ALL hands.
 */
void sv2_get_all_strategies(const SolverV2 *s, int player,
                            float *strategy_out);

/**
 * Compute exploitability using best-response traversal.
 */
float sv2_exploitability(SolverV2 *s);

/**
 * Free all memory.
 */
void sv2_free(SolverV2 *s);

#endif /* SOLVER_V2_H */
