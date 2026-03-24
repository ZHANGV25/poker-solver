/**
 * gpu_mccfr.cuh — Batch Outcome-Sampling MCCFR on GPU
 *
 * Novel approach: instead of external sampling (traverse ALL traverser
 * actions — inherently sequential/branching), we use OUTCOME SAMPLING
 * where every player's action is sampled, producing a single linear path
 * per iteration. This is trivially parallelizable on GPU — launch
 * thousands of independent trajectories simultaneously.
 *
 * Algorithm (Outcome-Sampling MCCFR with importance weighting):
 *   For each batch of K trajectories in parallel:
 *     1. Sample hands for all players (with card-conflict rejection)
 *     2. At each decision node, sample ONE action from exploration policy:
 *          sigma_sample(a) = eps/|A| + (1-eps)*sigma(a)
 *     3. Follow sampled path to terminal (fold/showdown)
 *     4. Walk back: compute importance-weighted counterfactual regrets
 *     5. atomicAdd regret updates to shared info set tables
 *   After each batch:
 *     - Apply Linear CFR discount: regrets *= t/(t+1)
 *     - Regret-match to recompute strategies
 *
 * Multi-street: the FULL game tree (flop→turn→river) is pre-built on CPU
 * as flat arrays with chance nodes for dealt cards. Each trajectory samples
 * one turn card and one river card as part of its path.
 *
 * Info sets are indexed directly by (decision_node_id, hand_idx) — no hash
 * table needed because the pre-built tree gives each action history a
 * unique node ID.
 *
 * Memory layout (SoA for coalesced GPU access):
 *   regrets:      [num_decision_nodes * GM_MAX_ACTIONS * max_hands]
 *   strategy_sum: [num_decision_nodes * GM_MAX_ACTIONS * max_hands]
 *   strategy:     [num_decision_nodes * GM_MAX_ACTIONS * max_hands]
 *
 * References:
 *   - Lanctot et al., "Monte Carlo Sampling for Regret Minimization
 *     in Extensive Games", NeurIPS 2009 (outcome sampling formulation)
 *   - Brown & Sandholm, "Superhuman AI for multiplayer poker", Science 2019
 *   - Brown & Sandholm, "Solving Imperfect-Information Games via
 *     Discounted Regret Minimization", AAAI 2019
 */
#ifndef GPU_MCCFR_CUH
#define GPU_MCCFR_CUH

#include <stdint.h>

/* ── Limits ──────────────────────────────────────────────────────────── */

#define GM_MAX_PLAYERS    6
#define GM_MAX_HANDS      169     /* max hands per player (exact card combos) */
#define GM_MAX_ACTIONS    6       /* fold, check/call, bet33, bet75, bet150, allin */
#define GM_MAX_BOARD      5
#define GM_MAX_RAISES     3
#define GM_MAX_PATH_LEN   64      /* max decision nodes along any root-to-terminal path */
#define GM_MAX_BUCKETS    64      /* max equity buckets per player */

/* Batch size: number of simultaneous outcome-sampling trajectories.
 * Each trajectory = 1 GPU thread. RTX 3060 has 3584 CUDA cores,
 * 28 SMs × 1536 threads/SM = ~43K max concurrent threads.
 * We use 16K-32K for good occupancy while leaving memory headroom. */
#define GM_DEFAULT_BATCH_SIZE  16384

/* Exploration parameter for sampling policy:
 * sigma_sample(a) = eps/|A| + (1-eps)*sigma(a)
 * Higher eps = more exploration = higher variance but better coverage.
 * Pluribus supplementary uses eps=0.6 for external sampling;
 * outcome sampling benefits from similar or slightly higher. */
#define GM_EXPLORATION_EPS  0.6f

/* ── Node types ──────────────────────────────────────────────────────── */

#define GM_NODE_DECISION  0       /* player chooses an action */
#define GM_NODE_CHANCE    1       /* deal a card (turn or river) */
#define GM_NODE_FOLD      2       /* terminal: someone folded */
#define GM_NODE_SHOWDOWN  3       /* terminal: river showdown */

/* ── Flat tree node ──────────────────────────────────────────────────── */

typedef struct {
    int type;                     /* GM_NODE_DECISION/CHANCE/FOLD/SHOWDOWN */
    int player;                   /* acting player (0..N-1), or -1 for terminal/chance */
    int num_children;             /* number of child edges */
    int first_child;              /* index into children[] array */

    /* For GM_NODE_CHANCE: card dealing */
    int street;                   /* 0=flop betting, 1=turn betting, 2=river betting */

    /* Pot and bets */
    int pot;
    int bets[GM_MAX_PLAYERS];

    /* Active players */
    int active[GM_MAX_PLAYERS];
    int num_active;

    /* For GM_NODE_FOLD: which player folded */
    int fold_player;

    /* For decision nodes: info set index (dense, 0..num_decision_nodes-1) */
    int decision_idx;

    /* For decision nodes: number of legal actions */
    int num_actions;

    /* Parent index (for backward pass) — -1 if root */
    int parent;
    /* Which child index of parent leads to this node */
    int parent_action;
} GMNode;

/* ── Input to the GPU solver ────────────────────────────────────────── */

typedef struct {
    /* Tree (built by CPU, uploaded to GPU) */
    GMNode *nodes;
    int *children;                /* children[node.first_child + a] = child node idx */
    int num_nodes;
    int num_children_total;

    /* Decision node mapping (dense index → node index) */
    int *decision_node_map;       /* decision_node_map[decision_idx] = node index */
    int num_decision_nodes;

    /* Players */
    int num_players;

    /* Hands — per player, cards as (c0, c1) pairs */
    int hands[GM_MAX_PLAYERS][GM_MAX_HANDS][2];
    int num_hands[GM_MAX_PLAYERS];
    int max_hands;                /* max over all players' num_hands */

    /* Board (flop, fixed) */
    int flop[3];

    /* Bet sizing */
    float bet_sizes[GM_MAX_ACTIONS];
    int num_bet_sizes;

    /* Pot and stacks */
    int starting_pot;
    int effective_stack;

    /* Card abstraction (equity buckets).
     * When use_buckets=1, regret/strategy arrays are indexed by bucket
     * instead of hand. All hands in the same bucket share strategies.
     * Hands are still sampled exactly for card-conflict checks and
     * showdown evaluation — only the info set indexing uses buckets. */
    int use_buckets;                /* 0 = exact hands, 1 = use buckets */
    int hand_to_bucket[GM_MAX_PLAYERS][GM_MAX_HANDS]; /* hand_idx -> bucket */
    int num_buckets[GM_MAX_PLAYERS];
    int max_buckets;                /* max over all players' num_buckets */
} GMTreeData;

/* ── Solver configuration ────────────────────────────────────────────── */

typedef struct {
    int max_iterations;           /* outer iterations (each = 1 batch of trajectories) */
    int batch_size;               /* trajectories per batch (default: GM_DEFAULT_BATCH_SIZE) */
    float exploration_eps;        /* epsilon for exploration (default: GM_EXPLORATION_EPS) */
    int print_every;              /* print progress every N iterations (0 = silent) */
} GMSolveConfig;

/* ── Output ──────────────────────────────────────────────────────────── */

typedef struct {
    /* Weighted-average strategy at every decision node.
     * Indexed by bucket when use_buckets=1, by hand otherwise. */
    float *avg_strategy;          /* [num_decision_nodes * GM_MAX_ACTIONS * max_slots] */
    int num_decision_nodes;
    int max_hands;                /* max hands (for non-bucketed mode) */
    int num_players;
    int use_buckets;              /* mirrors tree_data->use_buckets */
    int max_buckets;              /* mirrors tree_data->max_buckets */

    /* Decision node info: which player, how many actions */
    int *decision_players;        /* [num_decision_nodes] */
    int *decision_num_actions;    /* [num_decision_nodes] */

    /* Stats */
    int iterations_run;
    int total_trajectories;
    float solve_time_ms;
} GMOutput;

/* ── Host API ────────────────────────────────────────────────────────── */

#ifdef _WIN32
#define GM_EXPORT __declspec(dllexport)
#else
#define GM_EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Build multi-street game tree on CPU.
 *
 * Builds the FULL flop→turn→river tree with chance nodes.
 * Chance nodes have one child per possible dealt card.
 * The tree is stored as flat arrays suitable for GPU upload.
 *
 * For 6-player with 4 bet sizes and 3 max raises:
 *   - Single street: ~300-1000 nodes
 *   - With turn (47 cards): ~15K-50K nodes
 *   - With river (46 cards): ~700K-2.3M nodes
 *   - Memory: ~100-300 MB for the tree
 *
 * To keep memory manageable, we support a `max_turn_cards` parameter
 * that limits the number of turn cards enumerated (sampling the rest).
 * Similarly `max_river_cards`. Set to 0 for "enumerate all".
 *
 * Returns 0 on success.
 */
GM_EXPORT int gm_build_tree(
    const int *flop, int num_players,
    const int *acting_order,
    int starting_pot, int effective_stack,
    const float *bet_sizes, int num_bet_sizes,
    int max_turn_cards,           /* 0 = all 47, else sample this many */
    int max_river_cards,          /* 0 = all 46, else sample this many */
    GMTreeData *tree_out
);

/**
 * Run batch outcome-sampling MCCFR on GPU.
 *
 * Each iteration launches `config.batch_size` independent trajectories.
 * Each trajectory samples one complete path through the game tree,
 * computes importance-weighted counterfactual regrets, and updates
 * shared regret tables via atomicAdd.
 *
 * After each iteration:
 *   - Linear CFR discount: regrets *= t/(t+1)
 *   - Recompute strategies via regret matching
 *
 * Returns 0 on success.
 */
GM_EXPORT int gm_solve_gpu(
    GMTreeData *tree_data,
    GMSolveConfig *config,
    GMOutput *output
);

/**
 * Extract the weighted-average strategy for a specific decision node and hand.
 *
 * Returns num_actions, fills strategy_out[0..num_actions-1].
 * Returns 0 if decision_idx is out of range.
 */
GM_EXPORT int gm_get_strategy(
    const GMOutput *output,
    int decision_idx,
    int hand_idx,
    float *strategy_out
);

GM_EXPORT void gm_free_tree(GMTreeData *tree_data);
GM_EXPORT void gm_free_output(GMOutput *output);

/**
 * Utility: print tree statistics.
 */
GM_EXPORT void gm_print_tree_stats(const GMTreeData *tree_data);

#ifdef __cplusplus
}
#endif

#endif /* GPU_MCCFR_CUH */
