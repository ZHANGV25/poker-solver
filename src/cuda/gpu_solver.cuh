/**
 * gpu_solver.cuh — CUDA DCFR solver for batch flop precomputation
 *
 * Solves multiple flop textures in parallel on the GPU.
 * Each texture gets a thread block. Within each block, hand pairs
 * are parallelized across threads.
 *
 * Architecture:
 *   - One CUDA kernel per DCFR iteration
 *   - Regrets and strategies stored in GPU global memory
 *   - Showdown evaluation parallelized across hand pairs
 *   - Batch mode: solve hundreds of textures simultaneously
 */
#ifndef GPU_SOLVER_CUH
#define GPU_SOLVER_CUH

#include <stdint.h>

#define GPU_MAX_ACTIONS    6
#define GPU_MAX_HANDS    400   /* max hands per player after filtering */
#define GPU_MAX_NODES    128   /* max tree nodes per street */
#define GPU_MAX_BOARD      5

/* Node types */
#define GPU_NODE_DECISION  0
#define GPU_NODE_FOLD      1
#define GPU_NODE_SHOWDOWN  2
#define GPU_NODE_LEAF      3

/* ── Game tree (flat, GPU-friendly) ────────────────────────────────────── */

typedef struct {
    int type;
    int player;
    int num_actions;
    int children[GPU_MAX_ACTIONS];
    int pot;
    int bets[2];
} GPUNode;

/* ── Per-texture solve data ────────────────────────────────────────────── */

typedef struct {
    /* Board */
    int board[GPU_MAX_BOARD];
    int num_board;

    /* Hands and weights (both players) */
    int hands[2][GPU_MAX_HANDS][2];
    float weights[2][GPU_MAX_HANDS];
    int num_hands[2];

    /* Precomputed hand strengths (for river/showdown) */
    uint32_t strengths[2][GPU_MAX_HANDS];

    /* Game tree (shared across all textures with same bet sizes) */
    int num_nodes;

    /* Pot and stack */
    int starting_pot;
    int effective_stack;
} TextureData;

/* ── Solver state per texture (GPU memory) ─────────────────────────────── */

typedef struct {
    /* Regrets: [num_nodes][GPU_MAX_ACTIONS][GPU_MAX_HANDS] */
    float *regrets;
    /* Strategy sum (for extracting average strategy if needed) */
    float *strategy_sum;
    /* Current strategy: [num_nodes][GPU_MAX_ACTIONS][GPU_MAX_HANDS] */
    float *current_strategy;
} TextureSolverState;

/* ── Host API ──────────────────────────────────────────────────────────── */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Solve a batch of textures on the GPU.
 *
 * Args:
 *   textures: array of TextureData (host memory)
 *   num_textures: number of textures to solve
 *   tree: shared game tree (same structure for all textures)
 *   num_tree_nodes: number of nodes in the tree
 *   max_iterations: DCFR iterations per texture
 *   results_out: per-hand strategy output [num_textures][2][GPU_MAX_HANDS][GPU_MAX_ACTIONS]
 *                allocated by caller
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
 * Query available GPU memory and compute capability.
 */
int gpu_get_info(int *cuda_cores, size_t *free_mem, size_t *total_mem);

#ifdef __cplusplus
}
#endif

#endif /* GPU_SOLVER_CUH */
