/**
 * solver.h — Depth-limited Linear CFR poker solver
 *
 * Solves a single-street subgame with:
 *   - Weighted (Bayesian-narrowed) ranges for both players
 *   - 4 continuation strategies at leaf nodes (Pluribus-style)
 *   - O(N+M) prefix-sum showdown evaluation
 *   - Linear CFR: d = t/(t+1), matching Pluribus exactly
 *   - Regret-based pruning
 *   - 16-bit compressed regret storage
 */
#ifndef SOLVER_H
#define SOLVER_H

#include <stdint.h>

/* ── Constants ─────────────────────────────────────────────────────────── */

#define MAX_ACTIONS       8    /* max bet sizes + check + fold + call + allin */
#define MAX_HANDS      1326    /* C(52,2) */
#define MAX_BOARD         5
#define MAX_RAISES        2    /* max raises per street */
#define NUM_CONTINUATION  4    /* Pluribus: unmodified, fold-biased, call-biased, raise-biased */

/* Node types */
#define NODE_DECISION     0
#define NODE_FOLD         1
#define NODE_SHOWDOWN     2
#define NODE_CHANCE       3    /* turn/river card dealt */
#define NODE_LEAF         4    /* depth-limited leaf → continuation strategies */

/* ── Data structures ───────────────────────────────────────────────────── */

/**
 * A single node in the game tree.
 * Stored in a flat array (struct of arrays would be faster for SIMD,
 * but struct approach is clearer for initial implementation).
 */
typedef struct {
    int type;               /* NODE_DECISION, NODE_FOLD, NODE_SHOWDOWN, NODE_LEAF */
    int player;             /* 0 = OOP, 1 = IP, -1 = terminal/chance */
    int num_actions;        /* number of child actions */
    int children[MAX_ACTIONS]; /* child node indices */
    int pot;                /* pot size at this node (in chips, scale=100) */
    int stack;              /* remaining effective stack */
    int bets[2];            /* total amount each player has bet this street */
} TreeNode;

/**
 * Per-information-set storage for DCFR.
 * One InfoSet per (node, player) pair.
 * Stores regrets and cumulative strategy for all hands × actions.
 */
typedef struct {
    int node_idx;
    int num_actions;
    int num_hands;          /* number of valid hands for this player at this node */
    float *regrets;         /* [num_actions * num_hands] cumulative regrets */
    float *cum_strategy;    /* [num_actions * num_hands] cumulative strategy */
    float *current_strategy; /* [num_actions * num_hands] current iteration strategy */
} InfoSet;

/**
 * Continuation value at a leaf node.
 * For each continuation strategy k and each hand h:
 *   leaf_values[k * num_hands + h] = expected value of playing
 *   strategy k from this point forward.
 */
typedef struct {
    float *values;          /* [NUM_CONTINUATION * num_hands_per_player * 2] for both players */
} LeafValues;

/**
 * Pre-sorted hand strength data for O(N+M) showdown evaluation.
 */
typedef struct {
    int *sorted_indices;    /* hand indices sorted by strength (ascending) */
    uint32_t *strengths;    /* hand strength for each hand index */
    int num_hands;
} SortedHands;

/**
 * The solver configuration and state.
 */
typedef struct {
    /* Board */
    int board[MAX_BOARD];
    int num_board;          /* 3=flop, 4=turn, 5=river */

    /* Ranges: hands[player][i] = {card0, card1}, weights[player][i] = reach weight */
    int hands[2][MAX_HANDS][2];
    float weights[2][MAX_HANDS];
    int num_hands[2];

    /* Game tree */
    TreeNode *nodes;
    int num_nodes;

    /* Info sets */
    InfoSet *info_sets;
    int num_info_sets;

    /* Showdown evaluation (precomputed per board) */
    SortedHands sorted[2];
    uint32_t *hand_strengths[2]; /* precomputed eval7 strength per hand per player */

    /* Scratch buffers (avoid malloc per iteration) */
    float *scratch_reach[2]; /* temporary reach arrays */
    float *scratch_cfv;      /* temporary CFV array */

    /* Leaf continuation values (if depth-limited) */
    LeafValues *leaf_values; /* one per leaf node, or NULL for river */
    int num_leaves;

    /* Bet sizing */
    float bet_sizes[MAX_ACTIONS]; /* as fraction of pot (0.33, 0.75, etc.) */
    int num_bet_sizes;

    /* Pot and stacks (in chips, scale = 100 for 0.01 BB precision) */
    int starting_pot;
    int effective_stack;

    /* Linear CFR parameters (retained for API compat, not used by discount) */
    float alpha;
    float beta;
    float gamma;

    /* Results */
    int iterations_run;
    float exploitability;
} Solver;

/* ── Public API ────────────────────────────────────────────────────────── */

/**
 * Initialize solver with board, ranges, pot, stacks, bet sizes.
 * Builds game tree, sorts hands, allocates info sets.
 * Returns 0 on success, -1 on error.
 */
int solver_init(Solver *s,
                const int *board, int num_board,
                const int hands0[][2], const float *weights0, int num_hands0,
                const int hands1[][2], const float *weights1, int num_hands1,
                int starting_pot, int effective_stack,
                const float *bet_sizes, int num_bet_sizes);

/**
 * Set leaf continuation values for depth-limited solving.
 * leaf_values[leaf_idx][k * num_hands + h] = EV for hand h under strategy k.
 * If not called, solver uses showdown evaluation at all terminals (river mode).
 */
int solver_set_leaf_values(Solver *s, int leaf_idx,
                           const float *values_p0, const float *values_p1);

/**
 * Run DCFR for the specified number of iterations.
 * Returns exploitability as fraction of pot.
 */
float solver_solve(Solver *s, int max_iterations, float target_exploitability);

/**
 * Get the converged strategy for a specific hand at the root node.
 * Writes action probabilities to `strategy_out` (length = root num_actions).
 * player: 0=OOP, 1=IP.
 * Returns the EV for this hand.
 */
float solver_get_strategy(const Solver *s, int player, int hand_idx,
                          float *strategy_out);

/**
 * Get strategy for ALL hands at the root node.
 * Writes to strategy_out[hand_idx * num_actions + action_idx].
 * Also writes EVs to ev_out[hand_idx] if ev_out != NULL.
 */
void solver_get_all_strategies(const Solver *s, int player,
                               float *strategy_out, float *ev_out);

/**
 * Compute exploitability of the current strategy.
 * Returns exploitability as fraction of pot.
 */
float solver_exploitability(Solver *s);

/**
 * Free all allocated memory.
 */
void solver_free(Solver *s);

#endif /* SOLVER_H */
