/**
 * flop_solve.cuh — Full flop-through-river GPU solver
 *
 * Materializes the complete multi-street game tree on GPU and runs
 * level-batched Linear CFR over all ~62K nodes in parallel.
 *
 * Architecture:
 *   - CPU builds the full tree (flop betting → 49 turn cards →
 *     turn betting → ~46 river cards each → river betting → showdown)
 *   - Tree + hand data uploaded to GPU once
 *   - Each CFR iteration: 2 passes per traverser
 *     1. Top-down: propagate reach (one kernel per BFS level)
 *     2. Bottom-up: compute CFV at terminals, propagate up, update regrets
 *   - All ~62K nodes processed in parallel
 *   - Hand strengths precomputed per showdown node on GPU
 *
 * Memory for 80 hands, 2 bet sizes:
 *   Tree structure: ~62K nodes × 64 bytes = ~4MB
 *   Regrets + strategy: ~20K decision nodes × 4 actions × 80 hands × 12B = ~77MB
 *   CFV + reach: ~62K nodes × 80 hands × 8B = ~40MB
 *   Total: ~121MB (fits easily in 12GB RTX 3060)
 *
 * Expected performance:
 *   ~5-20ms per CFR iteration (vs 18s on CPU)
 *   200 iterations: 1-4 seconds for full flop solve
 */
#ifndef FLOP_SOLVE_CUH
#define FLOP_SOLVE_CUH

#include <stdint.h>

/* Limits for the full multi-street tree */
#define FS_MAX_HANDS      200     /* max hands per player */
#define FS_MAX_ACTIONS    6       /* max actions per decision node */
#define FS_MAX_TREE_NODES 100000  /* max total nodes across all streets */
#define FS_MAX_BOARD      5

/* Node types (same as CPU solver) */
#define FS_NODE_DECISION  0
#define FS_NODE_FOLD      1
#define FS_NODE_SHOWDOWN  2
#define FS_NODE_CHANCE    3  /* deal next card: children are per-card subtrees */

/* ── Flat tree node ───────────────────────────────────────────────────── */

typedef struct {
    int type;
    int player;           /* 0=OOP, 1=IP, -1=terminal/chance */
    int num_children;     /* for decision: num actions. for chance: num cards */
    int first_child;      /* index of first child in children array */
    int pot;              /* pot in chips */
    int bets[2];
    int board_cards[5];   /* full board at this node (for showdown eval) */
    int num_board;        /* number of board cards at this node */
} FSNode;

/* ── Input to the GPU solver ──────────────────────────────────────────── */

typedef struct {
    /* Tree (built by CPU, uploaded to GPU) */
    FSNode *nodes;          /* flat array of all tree nodes */
    int *children;          /* flat array: children[node.first_child + i] = child node index */
    int num_nodes;
    int num_children_total; /* total entries in children array */

    /* Hands and weights */
    int hands[2][FS_MAX_HANDS][2];
    float weights[2][FS_MAX_HANDS];
    int num_hands[2];

    /* BFS level ordering for the tree */
    int *level_order;       /* node indices in BFS order */
    int *node_depth;        /* depth of each node */
    int max_depth;

    /* Which nodes are decision nodes (for regret/strategy allocation) */
    int *decision_node_indices;  /* list of decision node indices */
    int num_decision_nodes;

    /* Showdown node list (for precomputing hand strengths) */
    int *showdown_node_indices;
    int num_showdown_nodes;
} FSTreeData;

/* ── Output ───────────────────────────────────────────────────────────── */

typedef struct {
    /* Strategy at the root node (flop root) */
    float *root_strategy;     /* [num_actions][num_hands[root_player]] */
    int root_num_actions;
    int root_player;

    /* Per-hand EV at root */
    float *root_ev;           /* [2][num_hands] */

    /* ── Extended output for precompute (Pluribus blueprint extraction) ── */

    /* Weighted average strategies at ALL decision nodes.
     * all_avg_strategies[node_idx * FS_MAX_ACTIONS * max_hands + a * max_hands + h]
     * Only allocated when fs_solve_gpu is called with extract_all=1. */
    float *all_avg_strategies;  /* [num_nodes * FS_MAX_ACTIONS * max_hands] */
    float *all_cfv;             /* [num_nodes * max_hands] per-hand CFV at each node */
    int max_hands;              /* max(num_hands[0], num_hands[1]) */

    /* Turn root identification.
     * turn_root_indices[i] = node index of the i-th turn root.
     * turn_root_cards[i]   = which card was dealt (0-51). */
    int *turn_root_indices;
    int *turn_root_cards;
    int num_turn_roots;
} FSOutput;

/* ── Host API ─────────────────────────────────────────────────────────── */

#ifdef _WIN32
#define FS_EXPORT __declspec(dllexport)
#else
#define FS_EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Build the full flop-through-river tree on CPU.
 * Caller must free tree_out->nodes, children, level_order, etc.
 */
FS_EXPORT int fs_build_tree(
    const int *flop_board,       /* 3 flop cards */
    int starting_pot,
    int effective_stack,
    const float *bet_sizes,
    int num_bet_sizes,
    FSTreeData *tree_out
);

/**
 * Solve the full tree on GPU using Linear CFR.
 *
 * The tree must be built first via fs_build_tree().
 * Hands and weights must be populated in tree_data.
 *
 * Returns 0 on success.
 */
FS_EXPORT int fs_solve_gpu(
    FSTreeData *tree_data,
    int max_iterations,
    FSOutput *output
);

/**
 * Free tree data allocated by fs_build_tree().
 */
FS_EXPORT void fs_free_tree(FSTreeData *tree_data);

/**
 * Free output data.
 */
FS_EXPORT void fs_free_output(FSOutput *output);

/**
 * Solve with full strategy extraction for precompute.
 *
 * Like fs_solve_gpu() but also extracts:
 *   - Weighted average strategies at all decision nodes
 *   - Per-hand CFV at all nodes
 *   - Turn root node identification
 *
 * Used by the precompute pipeline to generate blueprint data.
 */
FS_EXPORT int fs_solve_gpu_extract_all(
    FSTreeData *tree_data,
    int max_iterations,
    FSOutput *output
);

#ifdef __cplusplus
}
#endif

#endif /* FLOP_SOLVE_CUH */
