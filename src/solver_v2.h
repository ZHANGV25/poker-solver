/**
 * solver_v2.h — Pluribus-style multi-street solver
 *
 * Aligned with: Brown & Sandholm, Science 2019 (Pluribus)
 *   - Linear CFR (iteration-weighted regret discounting)
 *   - Final iteration strategy for play
 *   - Weighted average strategy for Bayesian belief updating
 *   - Multi-street solving: flop → turn → river → showdown
 *   - Chance sampling over deal-out cards between streets
 *   - O(N+M) prefix-sum showdown evaluation
 *
 * Tree structure:
 *   - Single-street betting tree built per street
 *   - At end of betting round (non-river), solver iterates over all
 *     possible next cards internally (chance sampling)
 *   - Each next-card subtree is a fresh single-street solve
 *   - Showdown evaluation at river terminal nodes
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
#define NODE_V2_DECISION      0
#define NODE_V2_FOLD          1
#define NODE_V2_SHOWDOWN      2
#define NODE_V2_CHANCE        3  /* deal next card, iterate over all possibilities */

/* ── Tree node ─────────────────────────────────────────────────────────── */

typedef struct {
    int type;
    int player;          /* 0=OOP, 1=IP, -1=terminal/chance */
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
    float *strategy_sum;      /* [num_actions * num_hands] iteration-weighted sum */
    float *current_strategy;  /* [num_actions * num_hands] last iteration */
} InfoSetV2;

/* ── Single-street subgame (one per street being solved) ──────────────── */

typedef struct {
    NodeV2 *nodes;
    int num_nodes;
    int nodes_capacity;
    InfoSetV2 *info_sets;
    int info_sets_capacity;
} StreetTree;

/* ── Solver state ──────────────────────────────────────────────────────── */

typedef struct {
    /* Board */
    int board[MAX_BOARD_V2];
    int num_board;            /* initial board cards (3 for flop, 4 for turn, 5 for river) */

    /* Ranges */
    int hands[2][MAX_HANDS_V2][2];   /* [player][hand_idx][card0,card1] */
    float weights[2][MAX_HANDS_V2];  /* initial weights (Bayesian-narrowed) */
    int num_hands[2];

    /* Root street tree (the street we're solving from) */
    StreetTree root_tree;

    /* Per-river-board hand strengths: computed on-the-fly during traversal */
    /* We cache the most recent river board's strengths to avoid recomputation */
    uint32_t *cached_strengths[2];
    int cached_river_board[5];
    int cached_river_valid;

    /* Bet sizing (shared across streets for simplicity) */
    float bet_sizes[MAX_ACTIONS_V2];
    int num_bet_sizes;

    /* Pot and stacks */
    int starting_pot;
    int effective_stack;

    /* Multi-street: per-street info sets stored in a hash table.
     * Key = (street_board_hash, node_index_in_street_tree).
     * This allows info sets to persist across CFR iterations for
     * turn/river subtrees that are rebuilt each iteration. */
    /* For efficiency, we use a flat array indexed by a compact key. */

    /* Turn subtree info sets: [turn_card_idx][node_idx] */
    /* turn_card_idx = index into the list of non-blocked turn cards */
    InfoSetV2 **turn_info_sets;    /* [num_turn_cards][max_turn_nodes] */
    int *turn_cards;               /* list of valid turn card indices */
    int num_turn_cards;
    int max_turn_nodes;

    /* River subtree info sets: [turn_card_idx][river_card_idx][node_idx] */
    /* Stored flat: river_info_sets[turn_idx * num_river_per_turn + river_idx][node_idx] */
    InfoSetV2 **river_info_sets;   /* [num_turn * max_river_per_turn][max_river_nodes] */
    int max_river_per_turn;
    int max_river_nodes;

    /* Solver state */
    int iterations_run;
    float exploitability;
} SolverV2;

/* ── Public API ────────────────────────────────────────────────────────── */

int sv2_init(SolverV2 *s,
             const int *board, int num_board,
             const int hands0[][2], const float *weights0, int num_hands0,
             const int hands1[][2], const float *weights1, int num_hands1,
             int starting_pot, int effective_stack,
             const float *bet_sizes, int num_bet_sizes);

float sv2_solve(SolverV2 *s, int max_iterations, float target_exploitability);

void sv2_get_strategy(const SolverV2 *s, int player, int hand_idx,
                      float *strategy_out);

void sv2_get_strategy_at_node(const SolverV2 *s,
                              const int *action_seq, int seq_len,
                              int player, int hand_idx,
                              float *strategy_out, int *num_actions_out);

void sv2_get_all_strategies(const SolverV2 *s, int player,
                            float *strategy_out);

void sv2_get_average_strategy(const SolverV2 *s, int player, int hand_idx,
                              float *strategy_out);

float sv2_exploitability(SolverV2 *s);

void sv2_free(SolverV2 *s);

#endif /* SOLVER_V2_H */
