/**
 * street_solve.cuh — N-player single-street GPU solver
 *
 * Pluribus-style depth-limited search for 2-6 player subgames.
 * Solves ONE betting round on GPU with externally-provided leaf values.
 *
 * N-player design (matching Pluribus):
 *   - Tree has up to SS_MAX_PLAYERS decision-making players
 *   - CFR cycles through all N players as traversers
 *   - At depth-limit leaves, ALL N remaining players simultaneously
 *     choose among 4 continuation strategies
 *   - Showdown compares all remaining hands (pot split for ties)
 *   - Fold awards pot to remaining players (or last player standing)
 *
 * Betting rotation:
 *   Players act in position order. In a single-street tree:
 *   - Preflop: SB(0), BB(1), UTG(2), MP(3), CO(4), BTN(5)
 *   - Postflop: SB(0), BB(1), UTG(2), ..., BTN(last)
 *   Players who have folded are skipped. The tree builder takes
 *   an array of active player indices and rotates through them.
 *
 * Continuation strategies at leaves (from Pluribus paper):
 *   At each non-river leaf, ALL remaining players simultaneously
 *   choose among 4 continuation strategies. Modeled as N sequential
 *   (but unobserved) decision nodes. CFR converges over all players'
 *   choices jointly.
 */
#ifndef STREET_SOLVE_CUH
#define STREET_SOLVE_CUH

#include <stdint.h>

/* ── Limits ──────────────────────────────────────────────────────────── */

#define SS_MAX_PLAYERS    6       /* max players in a subgame (6-max) */
#define SS_MAX_HANDS      200     /* max hands per player */
#define SS_MAX_ACTIONS    8       /* max actions per decision node */
#define SS_MAX_TREE_NODES 4096    /* max nodes (larger for N-player trees) */
#define SS_MAX_BOARD      5
#define SS_MAX_LEAVES     64      /* max depth-limit leaves before cont expansion */
#define SS_NUM_CONT_STRATS 4      /* unmodified, fold-biased, call-biased, raise-biased */
#define SS_MAX_RAISES     3       /* max raises per street */

/* Node types */
#define SS_NODE_DECISION  0
#define SS_NODE_FOLD      1       /* a player folded; node stores who folded */
#define SS_NODE_SHOWDOWN  2       /* river terminal — N-way showdown */
#define SS_NODE_LEAF      4       /* depth-limit leaf — external value */

/* ── Flat tree node ──────────────────────────────────────────────────── */

typedef struct {
    int type;                     /* SS_NODE_DECISION/FOLD/SHOWDOWN/LEAF */
    int player;                   /* which player acts (0..N-1), or -1 for terminal */
    int num_children;             /* for decision: num actions */
    int first_child;              /* index into children array */
    int pot;                      /* total pot in chips */
    int bets[SS_MAX_PLAYERS];     /* each player's total bet this street */
    int board_cards[5];
    int num_board;
    int leaf_idx;                 /* for SS_NODE_LEAF: index into leaf_values */
    int fold_player;              /* for SS_NODE_FOLD: which player folded (-1 if N/A) */
    int num_players;              /* number of active (non-folded) players at this node */
    int active_players[SS_MAX_PLAYERS]; /* which players are still active (1=active, 0=folded) */
} SSNode;

/* ── Input to the GPU solver ────────────────────────────────────────── */

typedef struct {
    /* Tree (built by CPU, uploaded to GPU) */
    SSNode *nodes;
    int *children;
    int num_nodes;
    int num_children_total;

    /* Players */
    int num_players;              /* N: number of players in this subgame (2-6) */

    /* Hands and weights — per player */
    int hands[SS_MAX_PLAYERS][SS_MAX_HANDS][2];
    float weights[SS_MAX_PLAYERS][SS_MAX_HANDS];
    int num_hands[SS_MAX_PLAYERS];

    /* Board */
    int board[SS_MAX_BOARD];
    int num_board;

    /* Leaf values: leaf_values[leaf_idx * num_players * max_hands + p * max_hands + h]
     * EV for player p, hand h, at leaf leaf_idx. */
    float *leaf_values;
    int num_leaves;

    /* BFS level ordering */
    int *level_order;
    int *node_depth;
    int max_depth;

    /* Node type indices */
    int *decision_node_indices;
    int num_decision_nodes;
    int *showdown_node_indices;
    int num_showdown_nodes;
    int *leaf_node_indices;
    int num_leaf_nodes;
    int *fold_node_indices;
    int num_fold_nodes;

    /* Pot and stacks */
    int starting_pot;
    int effective_stack;

    /* Whether this is a river solve */
    int is_river;

    /* A3 strategy freezing (Pluribus safe subgame re-solving).
     * Per-node array: frozen_action[node_idx] = action index that hero took
     * at this node (-1 if not frozen). At frozen nodes, hero's strategy is
     * locked to 100% on the taken action during all CFR iterations.
     * This ensures the re-solve is "safe" — it won't exploit hero's past actions.
     * Set to NULL if no freezing needed. */
    int *frozen_action;
} SSTreeData;

/* ── Output ──────────────────────────────────────────────────────────── */

typedef struct {
    /* Strategy at the root node — final iteration */
    float *root_strategy;         /* [num_actions][num_hands[root_player]] */
    int root_num_actions;
    int root_player;

    /* Per-hand EV at root for ALL players */
    float *root_ev;               /* [num_players][max_hands] */

    /* Weighted average strategies at ALL decision nodes */
    float *avg_strategies;
    int *avg_strategy_node_ids;
    int num_avg_nodes;

    /* Number of players (for interpreting arrays) */
    int num_players;
    int max_hands;
} SSOutput;

/* ── Host API ────────────────────────────────────────────────────────── */

#ifdef _WIN32
#define SS_EXPORT __declspec(dllexport)
#else
#define SS_EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Build an N-player single-street betting tree on CPU.
 *
 * Args:
 *   board, num_board: current board cards
 *   starting_pot: pot at start of this street (chips, scale=100)
 *   effective_stack: remaining stack per player (assumes equal stacks)
 *   bet_sizes, num_bet_sizes: bet size fractions
 *   num_players: number of active players (2-6)
 *   acting_order: array of player indices in acting order (length num_players)
 *                 e.g., [0,1] for heads-up OOP/IP, [0,1,2] for 3-way
 *   is_river: 1 if river (showdown terminals), 0 otherwise (leaf terminals)
 *   use_cont_strats: 1 to expand leaves with continuation strategy nodes
 *   tree_out: output tree data
 */
SS_EXPORT int ss_build_tree(
    const int *board, int num_board,
    int starting_pot, int effective_stack,
    const float *bet_sizes, int num_bet_sizes,
    int num_players, const int *acting_order,
    int is_river, int use_cont_strats,
    SSTreeData *tree_out
);

/**
 * Solve the N-player single-street tree on GPU using Linear CFR.
 *
 * CFR cycles through all num_players traversers per iteration.
 * Returns 0 on success.
 */
SS_EXPORT int ss_solve_gpu(
    SSTreeData *tree_data,
    int max_iterations,
    SSOutput *output
);

SS_EXPORT void ss_free_tree(SSTreeData *tree_data);
SS_EXPORT void ss_free_output(SSOutput *output);

#ifdef __cplusplus
}
#endif

#endif /* STREET_SOLVE_CUH */
