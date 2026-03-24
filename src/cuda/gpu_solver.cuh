/**
 * gpu_solver.cuh — CUDA solver for batch precomputation
 *
 * Supports:
 *   - Single-street solving (river, turn, flop)
 *   - Full flop-through-river solving with multi-level strategy extraction
 *   - Linear CFR (iteration-weighted regret discounting, matching Pluribus)
 *   - Batch mode: solve hundreds of textures simultaneously
 *   - Level-batched architecture (GPUGT paper, arXiv 2408.14778)
 *
 * GPU memory estimate for 80-hand full-game tree:
 *   ~500 nodes × 6 actions × 80 hands × 12 bytes = ~2.9 MB per texture
 *   RTX 3060 12GB → ~4000 textures simultaneously
 */
#ifndef GPU_SOLVER_CUH
#define GPU_SOLVER_CUH

#include <stdint.h>

#define GPU_MAX_ACTIONS    6
#define GPU_MAX_HANDS    400
#define GPU_MAX_NODES    512    /* increased for full-game trees */
#define GPU_MAX_BOARD      5
#define GPU_MAX_STREETS    4    /* preflop, flop, turn, river */

/* Node types */
#define GPU_NODE_DECISION  0
#define GPU_NODE_FOLD      1
#define GPU_NODE_SHOWDOWN  2
#define GPU_NODE_LEAF      3
#define GPU_NODE_CHANCE    4    /* deal next card (turn/river) */

/* ── Game tree (flat, GPU-friendly) ────────────────────────────────────── */

typedef struct {
    int type;
    int player;           /* 0=OOP, 1=IP, -1=terminal/chance */
    int num_actions;
    int children[GPU_MAX_ACTIONS];
    int pot;
    int bets[2];
    int street;           /* 0=flop, 1=turn, 2=river */
    int chance_cards;     /* for CHANCE nodes: number of possible cards */
    int chance_start;     /* index into chance_children array */
} GPUNode;

/* ── Per-texture solve data ────────────────────────────────────────────── */

typedef struct {
    int board[GPU_MAX_BOARD];
    int num_board;

    int hands[2][GPU_MAX_HANDS][2];
    float weights[2][GPU_MAX_HANDS];
    int num_hands[2];

    uint32_t strengths[2][GPU_MAX_HANDS];

    int num_nodes;
    int starting_pot;
    int effective_stack;
} TextureData;

/* ── Solver state per texture (GPU memory) ─────────────────────────────── */

typedef struct {
    float *regrets;
    float *strategy_sum;
    float *current_strategy;
} TextureSolverState;

/* ── Result extraction options ─────────────────────────────────────────── */

typedef struct {
    int extract_flop_root;     /* extract strategies at flop root */
    int extract_turn_roots;    /* extract strategies at each turn deal node */
    int extract_river_roots;   /* extract strategies at each river deal node */
    int flop_root_node;        /* node index of flop root */
    int *turn_root_nodes;      /* array of turn root node indices */
    int num_turn_roots;
    float *flop_strategies;    /* output: [2][num_hands][num_actions] */
    float *turn_strategies;    /* output: [num_turn_roots][2][num_hands][num_actions] */
} ExtractionConfig;

/* ── Host API ──────────────────────────────────────────────────────────── */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Solve a batch of textures on the GPU using Linear CFR.
 *
 * Args:
 *   textures: array of TextureData (host memory)
 *   num_textures: number of textures to solve
 *   tree: shared game tree
 *   num_tree_nodes: number of nodes in the tree
 *   max_iterations: Linear CFR iterations
 *   results_out: per-hand strategy output [num_textures][2][GPU_MAX_HANDS][GPU_MAX_ACTIONS]
 *
 * Returns 0 on success.
 */
int gpu_solve_batch(
    const TextureData *textures,
    int num_textures,
    const GPUNode *tree,
    int num_tree_nodes,
    int max_iterations,
    float *results_out
);

/**
 * Solve with multi-level strategy extraction.
 * Extracts strategies at flop root AND turn roots per runout.
 */
int gpu_solve_batch_multilevel(
    const TextureData *textures,
    int num_textures,
    const GPUNode *tree,
    int num_tree_nodes,
    int max_iterations,
    ExtractionConfig *config,
    float *results_out
);

int gpu_get_info(int *cuda_cores, size_t *free_mem, size_t *total_mem);

#ifdef __cplusplus
}
#endif

#endif /* GPU_SOLVER_CUH */
