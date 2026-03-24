/**
 * gpu_mccfr.cu — Batch Outcome-Sampling MCCFR on GPU
 *
 * Novel GPU-parallel MCCFR for 6-player NLHE blueprint computation.
 *
 * Architecture:
 *   1. CPU builds full multi-street game tree (flop→turn→river) as flat arrays
 *   2. Upload tree + regret/strategy arrays to GPU
 *   3. Per iteration:
 *      a. Kernel: regret_match — compute current strategy from regrets
 *      b. Kernel: batch_traverse — K parallel outcome-sampling trajectories
 *         Each thread: sample hands, walk tree sampling all actions,
 *         reach terminal, walk back computing importance-weighted regrets,
 *         atomicAdd to shared regret tables
 *      c. Kernel: linear_cfr_discount — regrets *= t/(t+1)
 *   4. Download weighted-average strategies
 *
 * Key insight: outcome sampling produces ONE linear path per iteration,
 * making it trivially parallelizable — no branching, no warp divergence.
 */

#include "gpu_mccfr.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        return -1; \
    } \
} while(0)

/* ═══════════════════════════════════════════════════════════════════════
 * PART 1: CPU — Multi-street tree construction
 * ═══════════════════════════════════════════════════════════════════════ */

typedef struct {
    GMNode *nodes;
    int num_nodes;
    int cap_nodes;
    int *children;
    int num_children;
    int cap_children;
    int num_decision_nodes;
} GMTreeBuilder;

static void tb_init(GMTreeBuilder *tb) {
    tb->cap_nodes = 4096;
    tb->nodes = (GMNode*)malloc(tb->cap_nodes * sizeof(GMNode));
    tb->num_nodes = 0;
    tb->cap_children = 16384;
    tb->children = (int*)malloc(tb->cap_children * sizeof(int));
    tb->num_children = 0;
    tb->num_decision_nodes = 0;
}

static int tb_alloc_node(GMTreeBuilder *tb) {
    if (tb->num_nodes >= tb->cap_nodes) {
        tb->cap_nodes *= 2;
        tb->nodes = (GMNode*)realloc(tb->nodes, tb->cap_nodes * sizeof(GMNode));
    }
    int idx = tb->num_nodes++;
    memset(&tb->nodes[idx], 0, sizeof(GMNode));
    tb->nodes[idx].player = -1;
    tb->nodes[idx].fold_player = -1;
    tb->nodes[idx].decision_idx = -1;
    tb->nodes[idx].parent = -1;
    tb->nodes[idx].parent_action = -1;
    return idx;
}

static int tb_alloc_children(GMTreeBuilder *tb, int count) {
    while (tb->num_children + count > tb->cap_children) {
        tb->cap_children *= 2;
        tb->children = (int*)realloc(tb->children, tb->cap_children * sizeof(int));
    }
    int start = tb->num_children;
    tb->num_children += count;
    return start;
}

/* Helpers */
static int count_active_gm(const int *active, int n) {
    int c = 0;
    for (int i = 0; i < n; i++) if (active[i]) c++;
    return c;
}

static int max_bet_gm(const int *bets, const int *active, int n) {
    int mx = 0;
    for (int i = 0; i < n; i++)
        if (active[i] && bets[i] > mx) mx = bets[i];
    return mx;
}

static int round_done_gm(const int *bets, const int *active,
                          const int *has_acted, int np) {
    int mx = max_bet_gm(bets, active, np);
    for (int i = 0; i < np; i++) {
        if (!active[i]) continue;
        if (!has_acted[i]) return 0;
        if (bets[i] != mx) return 0;
    }
    return 1;
}

static int next_active_gm(const int *acting_order, int num_in_order,
                            const int *active, int np, int cur_idx) {
    for (int i = 1; i <= num_in_order; i++) {
        int idx = (cur_idx + i) % num_in_order;
        int p = acting_order[idx];
        if (p < np && active[p]) return idx;
    }
    return -1;
}

/* Build one street's betting tree, recursively.
 * When the round completes, returns a terminal/chance node instead of recursing
 * into the next street (that's handled by build_multi_street). */
static int build_betting_round(
    GMTreeBuilder *tb, int np,
    const int *acting_order, int num_in_order, int cur_order_idx,
    int pot, int stack, int *bets, int *has_acted, int *active,
    int num_raises, const float *bet_sizes, int num_bet_sizes,
    int street, /* 0=flop, 1=turn, 2=river */
    int parent_node, int parent_action_idx
) {
    int n_active = count_active_gm(active, np);

    /* Terminal: only 1 player left (everyone else folded) */
    if (n_active <= 1) {
        int idx = tb_alloc_node(tb);
        tb->nodes[idx].type = GM_NODE_FOLD;
        tb->nodes[idx].pot = pot;
        memcpy(tb->nodes[idx].bets, bets, np * sizeof(int));
        memcpy(tb->nodes[idx].active, active, np * sizeof(int));
        tb->nodes[idx].num_active = n_active;
        tb->nodes[idx].street = street;
        tb->nodes[idx].parent = parent_node;
        tb->nodes[idx].parent_action = parent_action_idx;
        /* Find who folded: it's whoever caused this state.
         * Actually for fold terminals, fold_player is set by the action that led here. */
        for (int p = 0; p < np; p++)
            if (active[p]) { tb->nodes[idx].fold_player = -1; break; }
        return idx;
    }

    /* Round complete — transition to next street or showdown */
    if (round_done_gm(bets, active, has_acted, np)) {
        if (street == 2) {
            /* River round done → showdown */
            int idx = tb_alloc_node(tb);
            tb->nodes[idx].type = GM_NODE_SHOWDOWN;
            tb->nodes[idx].pot = pot;
            memcpy(tb->nodes[idx].bets, bets, np * sizeof(int));
            memcpy(tb->nodes[idx].active, active, np * sizeof(int));
            tb->nodes[idx].num_active = n_active;
            tb->nodes[idx].street = street;
            tb->nodes[idx].parent = parent_node;
            tb->nodes[idx].parent_action = parent_action_idx;
            return idx;
        } else {
            /* Flop/turn round done → chance node (deal next card) */
            int idx = tb_alloc_node(tb);
            tb->nodes[idx].type = GM_NODE_CHANCE;
            tb->nodes[idx].pot = pot;
            memcpy(tb->nodes[idx].bets, bets, np * sizeof(int));
            memcpy(tb->nodes[idx].active, active, np * sizeof(int));
            tb->nodes[idx].num_active = n_active;
            tb->nodes[idx].street = street;
            tb->nodes[idx].parent = parent_node;
            tb->nodes[idx].parent_action = parent_action_idx;
            /* Children will be added by build_multi_street */
            return idx;
        }
    }

    /* Find current acting player */
    int acting_player = acting_order[cur_order_idx];
    if (!active[acting_player]) {
        int next_idx = next_active_gm(acting_order, num_in_order,
                                        active, np, cur_order_idx);
        if (next_idx < 0) {
            /* Shouldn't happen */
            int idx = tb_alloc_node(tb);
            tb->nodes[idx].type = GM_NODE_SHOWDOWN;
            tb->nodes[idx].pot = pot;
            tb->nodes[idx].parent = parent_node;
            tb->nodes[idx].parent_action = parent_action_idx;
            return idx;
        }
        return build_betting_round(tb, np, acting_order, num_in_order,
                                    next_idx, pot, stack, bets, has_acted, active,
                                    num_raises, bet_sizes, num_bet_sizes, street,
                                    parent_node, parent_action_idx);
    }

    /* Decision node */
    int node = tb_alloc_node(tb);
    tb->nodes[node].type = GM_NODE_DECISION;
    tb->nodes[node].player = acting_player;
    tb->nodes[node].pot = pot;
    memcpy(tb->nodes[node].bets, bets, np * sizeof(int));
    memcpy(tb->nodes[node].active, active, np * sizeof(int));
    tb->nodes[node].num_active = n_active;
    tb->nodes[node].street = street;
    tb->nodes[node].decision_idx = tb->num_decision_nodes++;
    tb->nodes[node].parent = parent_node;
    tb->nodes[node].parent_action = parent_action_idx;

    int mx = max_bet_gm(bets, active, np);
    int to_call = mx - bets[acting_player];
    if (to_call < 0) to_call = 0;

    int next_order = next_active_gm(acting_order, num_in_order, active, np, cur_order_idx);
    if (next_order < 0) next_order = cur_order_idx;

    int temp_children[GM_MAX_ACTIONS];
    int nc = 0;

    /* Fold (only if facing a bet) */
    if (to_call > 0 && nc < GM_MAX_ACTIONS) {
        int new_active[GM_MAX_PLAYERS];
        memcpy(new_active, active, np * sizeof(int));
        new_active[acting_player] = 0;
        int child = build_betting_round(tb, np, acting_order, num_in_order,
                                          next_order, pot, stack, bets, has_acted,
                                          new_active, num_raises, bet_sizes, num_bet_sizes,
                                          street, node, nc);
        temp_children[nc++] = child;
    }

    /* Check / Call */
    if (nc < GM_MAX_ACTIONS) {
        int new_bets[GM_MAX_PLAYERS], new_ha[GM_MAX_PLAYERS];
        memcpy(new_bets, bets, np * sizeof(int));
        memcpy(new_ha, has_acted, np * sizeof(int));
        new_bets[acting_player] = mx;
        new_ha[acting_player] = 1;
        int new_pot = pot + to_call;
        int child = build_betting_round(tb, np, acting_order, num_in_order,
                                          next_order, new_pot, stack, new_bets, new_ha,
                                          active, num_raises, bet_sizes, num_bet_sizes,
                                          street, node, nc);
        temp_children[nc++] = child;
    }

    /* Bets / Raises */
    if (num_raises < GM_MAX_RAISES) {
        int added_allin = 0;
        for (int i = 0; i < num_bet_sizes && nc < GM_MAX_ACTIONS; i++) {
            int ba;
            if (to_call == 0)
                ba = (int)(bet_sizes[i] * pot);
            else
                ba = to_call + (int)(bet_sizes[i] * (pot + to_call));
            if (ba >= stack) ba = stack;
            if (ba <= to_call) continue;
            if (ba >= stack) {
                if (added_allin) continue;
                added_allin = 1;
            }

            int new_bets[GM_MAX_PLAYERS], new_ha[GM_MAX_PLAYERS];
            memcpy(new_bets, bets, np * sizeof(int));
            memcpy(new_ha, has_acted, np * sizeof(int));
            new_bets[acting_player] = mx + ba;
            new_ha[acting_player] = 1;
            for (int p = 0; p < np; p++)
                if (p != acting_player && active[p]) new_ha[p] = 0;
            int new_pot = pot + ba;

            int child = build_betting_round(tb, np, acting_order, num_in_order,
                                              next_order, new_pot, stack, new_bets, new_ha,
                                              active, num_raises + 1, bet_sizes, num_bet_sizes,
                                              street, node, nc);
            temp_children[nc++] = child;
        }
        /* Explicit all-in */
        if (!added_allin && stack > to_call && nc < GM_MAX_ACTIONS) {
            int new_bets[GM_MAX_PLAYERS], new_ha[GM_MAX_PLAYERS];
            memcpy(new_bets, bets, np * sizeof(int));
            memcpy(new_ha, has_acted, np * sizeof(int));
            new_bets[acting_player] = mx + stack;
            new_ha[acting_player] = 1;
            for (int p = 0; p < np; p++)
                if (p != acting_player && active[p]) new_ha[p] = 0;
            int child = build_betting_round(tb, np, acting_order, num_in_order,
                                              next_order, pot + stack, stack, new_bets, new_ha,
                                              active, num_raises + 1, bet_sizes, num_bet_sizes,
                                              street, node, nc);
            temp_children[nc++] = child;
        }
    }

    tb->nodes[node].num_actions = nc;
    tb->nodes[node].num_children = nc;
    int start = tb_alloc_children(tb, nc);
    for (int i = 0; i < nc; i++) tb->children[start + i] = temp_children[i];
    tb->nodes[node].first_child = start;

    return node;
}

/* Build the full multi-street tree: flop betting → chance (turn) →
 * turn betting → chance (river) → river betting → showdown.
 *
 * Strategy: build flop betting tree first, then for each chance node
 * (end of flop), enumerate turn cards and build turn betting trees,
 * and so on for river. */
static void expand_chance_nodes(
    GMTreeBuilder *tb, int np,
    const int *acting_order, int num_in_order,
    int stack, const float *bet_sizes, int num_bet_sizes,
    const int *flop,
    int max_turn_cards, int max_river_cards
) {
    /* Find all chance nodes and expand them.
     * We process iteratively: first expand all street-0 chance nodes (flop→turn),
     * then all street-1 chance nodes (turn→river). */
    for (int target_street = 0; target_street <= 1; target_street++) {
        int next_street = target_street + 1;

        /* Collect indices of chance nodes at this street */
        int *chance_nodes = (int*)malloc(tb->num_nodes * sizeof(int));
        int n_chance = 0;
        for (int i = 0; i < tb->num_nodes; i++) {
            if (tb->nodes[i].type == GM_NODE_CHANCE &&
                tb->nodes[i].street == target_street &&
                tb->nodes[i].num_children == 0) {
                chance_nodes[n_chance++] = i;
            }
        }

        for (int ci = 0; ci < n_chance; ci++) {
            int cn_idx = chance_nodes[ci];

            /* Cache chance node data before any realloc (build_betting_round
             * can realloc tb->nodes, invalidating pointers) */
            int cn_pot = tb->nodes[cn_idx].pot;
            int cn_active[GM_MAX_PLAYERS];
            memcpy(cn_active, tb->nodes[cn_idx].active, np * sizeof(int));

            /* Determine blocked cards */
            int blocked[52] = {0};
            blocked[flop[0]] = 1;
            blocked[flop[1]] = 1;
            blocked[flop[2]] = 1;

            /* If this is a river chance node (target_street=1), we need
             * to find the turn card. Walk up parents to find the street-0
             * chance node and determine which child was taken. */
            if (target_street == 1) {
                int walk = cn_idx;
                int chance0 = -1;
                while (walk >= 0) {
                    if (tb->nodes[walk].type == GM_NODE_CHANCE &&
                        tb->nodes[walk].street == 0) {
                        chance0 = walk;
                        break;
                    }
                    walk = tb->nodes[walk].parent;
                }
                if (chance0 >= 0) {
                    /* Find which child of chance0 is on our ancestor path */
                    int child_walk = cn_idx;
                    while (child_walk >= 0 && tb->nodes[child_walk].parent != chance0)
                        child_walk = tb->nodes[child_walk].parent;
                    if (child_walk >= 0) {
                        int action_at_chance = tb->nodes[child_walk].parent_action;
                        /* Build valid card list (without flop) to map action→card */
                        int valid[52], nv = 0;
                        for (int c = 0; c < 52; c++)
                            if (!blocked[c]) valid[nv++] = c;
                        if (action_at_chance >= 0 && action_at_chance < nv)
                            blocked[valid[action_at_chance]] = 1;
                    }
                }
            }

            /* Enumerate valid cards to deal */
            int n_valid = 0;
            for (int c = 0; c < 52; c++)
                if (!blocked[c]) n_valid++;

            /* Limit number of cards if requested */
            int n_deal = n_valid;
            if (target_street == 0 && max_turn_cards > 0 && max_turn_cards < n_deal)
                n_deal = max_turn_cards;
            if (target_street == 1 && max_river_cards > 0 && max_river_cards < n_deal)
                n_deal = max_river_cards;

            /* Build children: one subtree per dealt card */
            int *card_children = (int*)malloc(n_deal * sizeof(int));
            for (int ci2 = 0; ci2 < n_deal; ci2++) {
                /* Start a new betting round for the next street */
                int new_bets[GM_MAX_PLAYERS] = {0};
                int new_ha[GM_MAX_PLAYERS] = {0};

                /* Re-read cn_idx data (nodes may have been realloc'd) */
                int child = build_betting_round(
                    tb, np, acting_order, num_in_order, 0,
                    cn_pot, stack, new_bets, new_ha, cn_active,
                    0, /* num_raises reset */
                    bet_sizes, num_bet_sizes,
                    next_street, cn_idx, ci2
                );
                card_children[ci2] = child;
            }

            /* Store children on the chance node (nodes pointer is stable now) */
            int start = tb_alloc_children(tb, n_deal);
            for (int ci2 = 0; ci2 < n_deal; ci2++)
                tb->children[start + ci2] = card_children[ci2];
            tb->nodes[cn_idx].num_children = n_deal;
            tb->nodes[cn_idx].first_child = start;

            free(card_children);
        }

        free(chance_nodes);
    }
}

/* ── Public: build multi-street tree ────────────────────────────────── */

int gm_build_tree(
    const int *flop, int num_players,
    const int *acting_order,
    int starting_pot, int effective_stack,
    const float *bet_sizes, int num_bet_sizes,
    int max_turn_cards, int max_river_cards,
    GMTreeData *tree_out
) {
    memset(tree_out, 0, sizeof(GMTreeData));

    GMTreeBuilder tb;
    tb_init(&tb);

    /* Build flop betting round */
    int init_bets[GM_MAX_PLAYERS] = {0};
    int init_ha[GM_MAX_PLAYERS] = {0};
    int init_active[GM_MAX_PLAYERS] = {0};
    for (int i = 0; i < num_players; i++) init_active[i] = 1;

    int root = build_betting_round(
        &tb, num_players, acting_order, num_players, 0,
        starting_pot, effective_stack, init_bets, init_ha, init_active,
        0, bet_sizes, num_bet_sizes, 0, /* street=flop */
        -1, -1 /* no parent */
    );

    printf("[GM] Flop betting tree: %d nodes, %d decision nodes\n",
           tb.num_nodes, tb.num_decision_nodes);

    /* Expand chance nodes: flop→turn, then turn→river */
    expand_chance_nodes(&tb, num_players, acting_order, num_players,
                        effective_stack, bet_sizes, num_bet_sizes, flop,
                        max_turn_cards, max_river_cards);

    printf("[GM] Full tree: %d nodes, %d decision nodes, %d children\n",
           tb.num_nodes, tb.num_decision_nodes, tb.num_children);

    /* Build decision node map */
    int *dec_map = (int*)calloc(tb.num_decision_nodes, sizeof(int));
    for (int i = 0; i < tb.num_nodes; i++) {
        if (tb.nodes[i].type == GM_NODE_DECISION) {
            int di = tb.nodes[i].decision_idx;
            if (di >= 0 && di < tb.num_decision_nodes)
                dec_map[di] = i;
        }
    }

    /* Fill output */
    tree_out->nodes = tb.nodes;
    tree_out->children = tb.children;
    tree_out->num_nodes = tb.num_nodes;
    tree_out->num_children_total = tb.num_children;
    tree_out->decision_node_map = dec_map;
    tree_out->num_decision_nodes = tb.num_decision_nodes;
    tree_out->num_players = num_players;
    memcpy(tree_out->flop, flop, 3 * sizeof(int));
    tree_out->num_bet_sizes = num_bet_sizes;
    for (int i = 0; i < num_bet_sizes; i++)
        tree_out->bet_sizes[i] = bet_sizes[i];
    tree_out->starting_pot = starting_pot;
    tree_out->effective_stack = effective_stack;

    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════
 * PART 2: GPU — Device code (hand eval, kernels)
 * ═══════════════════════════════════════════════════════════════════════ */

/* ── 5-card evaluator (device) ─────────────────────────────────────── */

__device__ uint32_t gm_eval5(int c0, int c1, int c2, int c3, int c4) {
    int r[5] = {c0 >> 2, c1 >> 2, c2 >> 2, c3 >> 2, c4 >> 2};
    int s[5] = {c0 & 3, c1 & 3, c2 & 3, c3 & 3, c4 & 3};
    /* Sort ranks descending (insertion sort) */
    for (int i = 1; i < 5; i++) {
        int k = r[i], j = i - 1;
        while (j >= 0 && r[j] < k) { r[j + 1] = r[j]; j--; }
        r[j + 1] = k;
    }
    int fl = (s[0]==s[1] && s[1]==s[2] && s[2]==s[3] && s[3]==s[4]);
    int st = 0, sh = r[0];
    if (r[0]-r[4]==4 && r[0]!=r[1] && r[1]!=r[2] && r[2]!=r[3] && r[3]!=r[4]) st=1;
    if (r[0]==12 && r[1]==3 && r[2]==2 && r[3]==1 && r[4]==0) { st=1; sh=3; }
    if (st && fl) return (9u<<20)|(sh<<16);
    if (fl) return (6u<<20)|(r[0]<<16)|(r[1]<<12)|(r[2]<<8)|(r[3]<<4)|r[4];
    if (st) return (5u<<20)|(sh<<16);
    int cn[13]={0};
    for (int i=0;i<5;i++) cn[r[i]]++;
    int q=-1,t=-1,p1=-1,p2=-1;
    for (int i=12;i>=0;i--) {
        if (cn[i]==4) q=i; else if (cn[i]==3) t=i;
        else if (cn[i]==2) { if (p1<0) p1=i; else p2=i; }
    }
    if (q>=0) { int k=-1; for (int i=12;i>=0;i--) if (cn[i]>0&&i!=q){k=i;break;} return (8u<<20)|(q<<16)|(k<<12); }
    if (t>=0&&p1>=0) return (7u<<20)|(t<<16)|(p1<<12);
    if (t>=0) { int k0=-1,k1=-1; for (int i=12;i>=0;i--) if (cn[i]>0&&i!=t){if(k0<0)k0=i;else k1=i;} return (4u<<20)|(t<<16)|(k0<<12)|(k1<<8); }
    if (p1>=0&&p2>=0) { int k=-1; for (int i=12;i>=0;i--) if (cn[i]>0&&i!=p1&&i!=p2){k=i;break;} return (3u<<20)|(p1<<16)|(p2<<12)|(k<<8); }
    if (p1>=0) { int k[3]; int ki=0; for (int i=12;i>=0&&ki<3;i--) if (cn[i]>0&&i!=p1) k[ki++]=i; return (2u<<20)|(p1<<16)|(k[0]<<12)|(k[1]<<8)|(k[2]<<4); }
    return (1u<<20)|(r[0]<<16)|(r[1]<<12)|(r[2]<<8)|(r[3]<<4)|r[4];
}

__device__ uint32_t gm_eval7(const int c[7]) {
    /* 21 combinations of 5 from 7 */
    static const int cb[21][5] = {
        {0,1,2,3,4},{0,1,2,3,5},{0,1,2,3,6},{0,1,2,4,5},{0,1,2,4,6},
        {0,1,2,5,6},{0,1,3,4,5},{0,1,3,4,6},{0,1,3,5,6},{0,1,4,5,6},
        {0,2,3,4,5},{0,2,3,4,6},{0,2,3,5,6},{0,2,4,5,6},{0,3,4,5,6},
        {1,2,3,4,5},{1,2,3,4,6},{1,2,3,5,6},{1,2,4,5,6},{1,3,4,5,6},
        {2,3,4,5,6}
    };
    uint32_t best = 0;
    for (int i = 0; i < 21; i++) {
        uint32_t v = gm_eval5(c[cb[i][0]], c[cb[i][1]], c[cb[i][2]],
                               c[cb[i][3]], c[cb[i][4]]);
        if (v > best) best = v;
    }
    return best;
}

/* ── Kernel 1: Regret matching ─────────────────────────────────────── */

/* One block per decision node, one thread per slot (hand or bucket).
 * Reads regrets[], writes strategy[] via regret matching.
 * max_s = max_hands when not bucketed, max_buckets when bucketed. */
__global__ void gm_regret_match_kernel(
    const int *d_decision_node_map,  /* [num_dec_nodes] -> node idx */
    const GMNode *d_nodes,
    float *d_regrets,                /* [num_dec * MAX_ACTIONS * max_s] */
    float *d_strategy,               /* [num_dec * MAX_ACTIONS * max_s] */
    const int *d_num_slots,          /* [MAX_PLAYERS]: num_hands or num_buckets */
    int num_dec_nodes,
    int max_s                        /* max_hands or max_buckets */
) {
    int di = blockIdx.x;            /* decision node index */
    int s = threadIdx.x;            /* slot index (hand or bucket) */
    if (di >= num_dec_nodes) return;

    int node_idx = d_decision_node_map[di];
    int player = d_nodes[node_idx].player;
    int na = d_nodes[node_idx].num_actions;
    int ns = d_num_slots[player];

    if (s >= ns || na <= 0) return;

    int base = di * GM_MAX_ACTIONS * max_s;

    float sum = 0.0f;
    float strat[GM_MAX_ACTIONS];
    for (int a = 0; a < na; a++) {
        float r = d_regrets[base + a * max_s + s];
        float pos = (r > 0.0f) ? r : 0.0f;
        strat[a] = pos;
        sum += pos;
    }

    if (sum > 0.0f) {
        float inv = 1.0f / sum;
        for (int a = 0; a < na; a++)
            d_strategy[base + a * max_s + s] = strat[a] * inv;
    } else {
        float u = 1.0f / (float)na;
        for (int a = 0; a < na; a++)
            d_strategy[base + a * max_s + s] = u;
    }
}

/* ── Kernel 2: Batch outcome-sampling traversal ────────────────────── */

/* Each thread executes ONE complete outcome-sampling trajectory.
 *
 * Steps per thread:
 * 1. Sample hands for all players (rejection sampling for card conflicts)
 * 2. Walk down the tree from root, sampling one action at each node:
 *    - Decision: sample from exploration policy sigma_sample
 *    - Chance: sample one card uniformly
 *    Record the path (node indices, actions, strategy probs)
 * 3. At terminal: compute payoff
 * 4. Walk back up: compute importance-weighted counterfactual regrets
 *    for the traverser, atomicAdd to regret tables
 *
 * Importance weighting for outcome sampling:
 *   At traverser's info set I along the path, for the sampled action a_s:
 *     For each action a:
 *       if a == a_s:
 *         regret_update = (u(z) * w_{-i}) / q(a_s|I) - v(I)
 *       else:
 *         regret_update = -v(I)
 *     where:
 *       u(z) = terminal payoff for traverser
 *       w_{-i} = product of opponents' strategy probs / sampling probs (tail weight)
 *       q(a|I) = sampling probability of action a at info set I
 *       v(I) = sum_a sigma(a|I) * q_value(a)
 *
 *   Simplified outcome sampling update (Lanctot 2009, Theorem 2):
 *     Let pi_{-i}(z) = product of all opponents' strategy probs on the path
 *     Let pi_sample(z) = product of all sampling probs on the path
 *     Let W = pi_{-i}(z) / (pi_sample(z) / pi_sample_i(z))
 *           = pi_{-i}(z) * pi_sample_i(z_I→z) / pi_sample_{-i}(z)
 *     (this is the importance weight from opponents)
 *
 *     At info set I of traverser with sampled action a_s:
 *       For each action a:
 *         r(a) += W * [1(a==a_s) * u(z) / q(a_s|I) - v(I)]
 *
 *   In practice, we use the "stochastically-weighted" form:
 *     regret(I, a) += W_tail * [1(a==a_s)/q(a_s|I) * u(z)] - W_tail * sigma(a|I) * u_avg
 *     where W_tail accounts for opponents below this node.
 */
__global__ void gm_batch_traverse_kernel(
    const GMNode *d_nodes,
    const int *d_children,
    const int *d_decision_node_map,
    const float *d_strategy,           /* current strategy, indexed by slot */
    float *d_regrets,                  /* atomicAdd target, indexed by slot */
    float *d_strategy_sum,             /* atomicAdd target, indexed by slot */
    const int *d_hands,                /* [MAX_PLAYERS * MAX_HANDS * 2] */
    const int *d_num_hands,
    const int *d_flop,                 /* [3] */
    const int *d_hand_to_bucket,       /* [MAX_PLAYERS * MAX_HANDS] or NULL */
    int use_buckets,                   /* 0=exact hands, 1=bucket abstraction */
    int num_players,
    int max_s,                         /* max slots: max_hands or max_buckets */
    int traverser,                     /* which player is traversing */
    int iteration,                     /* current iteration (for strategy sum weight) */
    float eps,                         /* exploration epsilon */
    int batch_size,
    int num_dec_nodes
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size) return;

    /* Per-thread RNG (cuRAND initialized externally, we use a simple LCG here
     * seeded from tid + iteration for reproducibility) */
    /* Using a fast LCG-based PRNG for speed */
    unsigned long long rng = (unsigned long long)tid * 6364136223846793005ULL +
                              (unsigned long long)iteration * 1442695040888963407ULL + 1;
    #define RNG_NEXT(rng) ((rng) = (rng) * 6364136223846793005ULL + 1442695040888963407ULL)
    #define RNG_FLOAT(rng) ((float)((RNG_NEXT(rng) >> 33) & 0x7FFFFFFF) / (float)0x7FFFFFFF)
    #define RNG_INT(rng, n) ((int)((RNG_NEXT(rng) >> 33) % (unsigned long long)(n)))

    /* Step 1: Sample hands for all players */
    int sampled_hands[GM_MAX_PLAYERS];
    int hand_cards[GM_MAX_PLAYERS][2]; /* cached for conflict check */

    /* Rejection sampling for card conflicts */
    int attempts = 0;
    int valid = 0;
    while (!valid && attempts < 100) {
        attempts++;
        valid = 1;
        for (int p = 0; p < num_players; p++) {
            int nh = d_num_hands[p];
            sampled_hands[p] = RNG_INT(rng, nh);
            int base = p * GM_MAX_HANDS * 2 + sampled_hands[p] * 2;
            hand_cards[p][0] = d_hands[base];
            hand_cards[p][1] = d_hands[base + 1];

            /* Check vs flop */
            for (int f = 0; f < 3; f++) {
                if (hand_cards[p][0] == d_flop[f] || hand_cards[p][1] == d_flop[f])
                    { valid = 0; break; }
            }
            if (!valid) break;

            /* Check vs other players */
            for (int q = 0; q < p; q++) {
                if (hand_cards[p][0] == hand_cards[q][0] ||
                    hand_cards[p][0] == hand_cards[q][1] ||
                    hand_cards[p][1] == hand_cards[q][0] ||
                    hand_cards[p][1] == hand_cards[q][1])
                    { valid = 0; break; }
            }
            if (!valid) break;
        }
    }
    if (!valid) return; /* give up after 100 attempts */

    /* Step 2: Walk down the tree, recording path */
    /* Path recording: store decision nodes visited, actions taken, probabilities */
    int path_dec_idx[GM_MAX_PATH_LEN];    /* decision_idx of visited decision nodes */
    int path_action[GM_MAX_PATH_LEN];     /* action taken at each */
    float path_sigma[GM_MAX_PATH_LEN];    /* sigma(a|I) at each (actual strategy prob) */
    float path_qsample[GM_MAX_PATH_LEN];  /* q(a|I) at each (sampling prob) */
    int path_player[GM_MAX_PATH_LEN];     /* player who acted */
    int path_len = 0;

    int current = 0; /* root node index */
    int board[5];
    board[0] = d_flop[0]; board[1] = d_flop[1]; board[2] = d_flop[2];
    int num_board = 3;

    /* Track blocked cards for chance sampling */
    /* Using a 64-bit bitmask for 52 cards */
    unsigned long long blocked = 0ULL;
    blocked |= (1ULL << d_flop[0]) | (1ULL << d_flop[1]) | (1ULL << d_flop[2]);
    for (int p = 0; p < num_players; p++) {
        blocked |= (1ULL << hand_cards[p][0]) | (1ULL << hand_cards[p][1]);
    }

    float terminal_payoff = 0.0f;
    int reached_terminal = 0;

    /* Walk down tree */
    for (int step = 0; step < 200 && !reached_terminal; step++) {
        GMNode node = d_nodes[current];

        switch (node.type) {
        case GM_NODE_DECISION: {
            int player = node.player;
            int na = node.num_actions;
            int di = node.decision_idx;
            int hand = sampled_hands[player];

            if (na <= 0 || di < 0) { reached_terminal = 1; break; }

            /* Map hand to slot (bucket if using abstraction, hand otherwise) */
            int slot = hand;
            if (use_buckets && d_hand_to_bucket != 0)
                slot = d_hand_to_bucket[player * GM_MAX_HANDS + hand];

            /* Read current strategy for this slot */
            int base = di * GM_MAX_ACTIONS * max_s;
            float sigma[GM_MAX_ACTIONS];
            for (int a = 0; a < na; a++)
                sigma[a] = d_strategy[base + a * max_s + slot];

            /* Compute sampling policy: q(a) = eps/|A| + (1-eps)*sigma(a) */
            float q[GM_MAX_ACTIONS];
            float eps_uniform = eps / (float)na;
            for (int a = 0; a < na; a++)
                q[a] = eps_uniform + (1.0f - eps) * sigma[a];

            /* Sample action from q */
            float r = RNG_FLOAT(rng);
            float cum = 0.0f;
            int sampled_a = na - 1;
            for (int a = 0; a < na; a++) {
                cum += q[a];
                if (r <= cum) { sampled_a = a; break; }
            }

            /* Record in path (store slot, not hand, for regret indexing) */
            if (path_len < GM_MAX_PATH_LEN) {
                path_dec_idx[path_len] = di;
                path_action[path_len] = sampled_a;
                path_sigma[path_len] = sigma[sampled_a];
                path_qsample[path_len] = q[sampled_a];
                path_player[path_len] = player;
                path_len++;
            }

            /* Accumulate strategy sum */
            {
                float w = (float)iteration;
                atomicAdd(&d_strategy_sum[base + sampled_a * max_s + slot],
                          w * sigma[sampled_a]);
            }

            /* Follow child */
            current = d_children[node.first_child + sampled_a];
            break;
        }

        case GM_NODE_CHANCE: {
            /* Sample one child uniformly (each child = one dealt card).
             * Children were built by expand_chance_nodes in order of
             * valid_cards[0..n_deal-1], where valid_cards is the sorted
             * list of unblocked cards at tree-build time (board + no player
             * cards since hands aren't known at build time).
             *
             * At runtime we don't need to reconstruct the card list — we just
             * pick a child uniformly. The board card for evaluation will be
             * computed from the child's subtree context at showdown.
             *
             * However, we DO need to track which card was dealt so that
             * showdown evaluation has the correct 5-card board. We reconstruct
             * the valid card list (excluding only flop + previously dealt
             * street cards, NOT player hands — matching tree-build behavior)
             * and index into it.
             */
            int nc = node.num_children;
            if (nc == 0) { reached_terminal = 1; break; }

            int child_idx = RNG_INT(rng, nc);

            /* Reconstruct the card that child_idx corresponds to.
             * At tree build time, blocked = flop + (turn card if river street).
             * We reconstruct the same blocked set: flop only for turn deal,
             * flop + turn for river deal. Player hands are NOT blocked at
             * tree build time. */
            unsigned long long build_blocked = 0ULL;
            build_blocked |= (1ULL << d_flop[0]) | (1ULL << d_flop[1]) | (1ULL << d_flop[2]);
            /* If this is a river chance node (street >= 1), the turn card
             * is board[3] (already dealt in a prior chance node). */
            if (num_board >= 4)
                build_blocked |= (1ULL << board[3]);

            /* Build same valid card list as tree builder */
            int build_valid[52];
            int build_nv = 0;
            for (int c = 0; c < 52; c++)
                if (!(build_blocked & (1ULL << c)))
                    build_valid[build_nv++] = c;

            if (child_idx < build_nv && num_board < 5) {
                int dealt = build_valid[child_idx];
                board[num_board] = dealt;
                blocked |= (1ULL << dealt);
                num_board++;
            }

            current = d_children[node.first_child + child_idx];
            break;
        }

        case GM_NODE_FOLD: {
            /* Someone folded — find who's still in and who won */
            reached_terminal = 1;

            /* Traverser's payoff:
             * If traverser is still active → they win their share
             * If traverser folded → they lose their bets */
            if (!node.active[traverser]) {
                /* Traverser folded or was already out */
                terminal_payoff = -(float)node.bets[traverser];
            } else {
                /* Traverser is still in (someone else folded until only we remain) */
                if (node.num_active <= 1) {
                    terminal_payoff = (float)(node.pot - node.bets[traverser]);
                } else {
                    /* Multiple players still active but this is a fold terminal?
                     * This means someone just folded and play continues.
                     * But our tree should only have fold terminals when n_active <= 1.
                     * Treat as: traverser's share. */
                    terminal_payoff = (float)(node.pot - node.bets[traverser])
                                     / (float)node.num_active;
                }
            }
            break;
        }

        case GM_NODE_SHOWDOWN: {
            reached_terminal = 1;

            if (!node.active[traverser]) {
                terminal_payoff = -(float)node.bets[traverser];
                break;
            }

            /* Need full 5-card board for showdown */
            if (num_board < 5) {
                /* Deal remaining cards randomly for unbiased evaluation */
                for (int b = num_board; b < 5; b++) {
                    int remaining = 0;
                    for (int c = 0; c < 52; c++)
                        if (!(blocked & (1ULL << c))) remaining++;
                    if (remaining <= 0) break;
                    int pick = RNG_INT(rng, remaining);
                    int count = 0;
                    for (int c = 0; c < 52; c++) {
                        if (!(blocked & (1ULL << c))) {
                            if (count == pick) {
                                board[b] = c;
                                blocked |= (1ULL << c);
                                break;
                            }
                            count++;
                        }
                    }
                    num_board++;
                }
            }

            /* Evaluate traverser's hand */
            int tc0 = hand_cards[traverser][0];
            int tc1 = hand_cards[traverser][1];
            int cards_t[7] = {board[0], board[1], board[2], board[3], board[4], tc0, tc1};
            uint32_t trav_str = gm_eval7(cards_t);

            /* Compare against all active opponents */
            int n_tied = 1;
            int trav_wins = 1; /* assume traverser wins until proven otherwise */
            for (int p = 0; p < num_players; p++) {
                if (p == traverser || !node.active[p]) continue;
                int oc0 = hand_cards[p][0];
                int oc1 = hand_cards[p][1];
                int cards_o[7] = {board[0], board[1], board[2], board[3], board[4], oc0, oc1};
                uint32_t opp_str = gm_eval7(cards_o);
                if (opp_str > trav_str) { trav_wins = 0; break; }
                else if (opp_str == trav_str) n_tied++;
            }

            if (!trav_wins) {
                terminal_payoff = -(float)node.bets[traverser];
            } else {
                float share = (float)node.pot / (float)n_tied;
                terminal_payoff = share - (float)node.bets[traverser];
            }
            break;
        }

        default:
            reached_terminal = 1;
            break;
        }
    }

    if (!reached_terminal) return; /* path too long, skip */

    /* Step 4: Walk back — importance-weighted regret updates.
     *
     * Lanctot et al. (2009), Theorem 2, outcome sampling MCCFR:
     *
     * At traverser i's info set I along sampled terminal history z:
     *
     *   Let a_s = sampled action at I
     *   Let W = pi_{-i}(z) / pi_{-i}^q(z)
     *         = product of sigma(a)/q(a) for ALL non-traverser nodes on z
     *   Let S_i = pi_i(z[I→z]) / pi_i^q(z[I→z])
     *           = product of sigma(a)/q(a) for traverser nodes BELOW I on z
     *
     *   Regret update:
     *     r~(I, a) = W * u(z) * [I(a==a_s) / q(a_s|I)  -  S_i]
     *
     * This decomposes as:
     *   For a == a_s: r~(I, a_s) = W * u * (1/q(a_s|I) - S_i)
     *   For a != a_s: r~(I, a)   = W * u * (0           - S_i) = -W * u * S_i
     *
     * We track two running products bottom-up:
     *   W_opp:  product of sigma/q for opponent nodes (importance weight)
     *   S_trav: product of sigma/q for traverser nodes below current
     */
    float u = terminal_payoff;

    /* Pre-compute W_opp (full product for all opponent nodes on path)
     * and per-node S_trav (traverser tail below each node). */
    float w_opp = 1.0f;
    for (int k = 0; k < path_len; k++) {
        float ratio = path_sigma[k] / fmaxf(path_qsample[k], 1e-8f);
        if (path_player[k] != traverser)
            w_opp *= ratio;
    }

    /* Walk bottom-up, maintaining S_trav = product of sigma/q for
     * traverser nodes BELOW the current node k. */
    float s_trav = 1.0f;

    for (int k = path_len - 1; k >= 0; k--) {
        if (path_player[k] == traverser) {
            int di = path_dec_idx[k];
            int hand = sampled_hands[traverser];
            int slot = hand;
            if (use_buckets && d_hand_to_bucket != 0)
                slot = d_hand_to_bucket[traverser * GM_MAX_HANDS + hand];
            int na = d_nodes[d_decision_node_map[di]].num_actions;
            int base = di * GM_MAX_ACTIONS * max_s;

            float q_as = fmaxf(path_qsample[k], 1e-8f);
            int a_s = path_action[k];

            /* Lanctot Eq: r~(I, a) = W * u * [I(a==a_s)/q(a_s) - S_i] */
            for (int a = 0; a < na; a++) {
                float regret_incr;
                if (a == a_s) {
                    regret_incr = w_opp * u * (1.0f / q_as - s_trav);
                } else {
                    regret_incr = w_opp * u * (-s_trav);
                }
                if (regret_incr > 1e5f) regret_incr = 1e5f;
                if (regret_incr < -1e5f) regret_incr = -1e5f;

                atomicAdd(&d_regrets[base + a * max_s + slot], regret_incr);
            }

            /* Update S_trav: include this traverser node's ratio */
            float sigma_as = fmaxf(path_sigma[k], 1e-8f);
            s_trav *= sigma_as / q_as;
        }
        /* Opponent nodes don't affect s_trav (already in w_opp) */
    }

    #undef RNG_NEXT
    #undef RNG_FLOAT
    #undef RNG_INT
}

/* ── Kernel 3: CFR+ regret floor ──────────────────────────────────── */

/* CFR+ floors negative regrets at 0 every iteration.
 * This is more appropriate than Linear CFR discount for outcome sampling
 * because outcome sampling visits each info set very rarely per batch.
 * A global discount factor like t/(t+1) would destroy accumulated regrets
 * at unvisited info sets, causing strategies to collapse.
 * CFR+ instead floors negatives at 0, which preserves positive regrets
 * and has proven convergence guarantees (Tammelin et al., 2015). */
__global__ void gm_regret_floor_kernel(
    float *d_regrets,
    int num_dec_nodes,
    int max_s
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_dec_nodes * GM_MAX_ACTIONS * max_s;
    if (idx >= total) return;

    if (d_regrets[idx] < 0.0f) d_regrets[idx] = 0.0f;
}

/* ═══════════════════════════════════════════════════════════════════════
 * PART 3: Host orchestration
 * ═══════════════════════════════════════════════════════════════════════ */

int gm_solve_gpu(
    GMTreeData *tree_data,
    GMSolveConfig *config,
    GMOutput *output
) {
    int NP = tree_data->num_players;
    int num_dec = tree_data->num_decision_nodes;
    int use_buckets = tree_data->use_buckets;
    int max_s = use_buckets ? tree_data->max_buckets : tree_data->max_hands;
    int batch_size = config->batch_size > 0 ? config->batch_size : GM_DEFAULT_BATCH_SIZE;
    int max_iter = config->max_iterations;
    float eps = config->exploration_eps > 0 ? config->exploration_eps : GM_EXPLORATION_EPS;

    if (num_dec == 0) {
        fprintf(stderr, "[GM] No decision nodes in tree\n");
        return -1;
    }
    if (max_s <= 0) {
        fprintf(stderr, "[GM] max_slots is 0 (set max_hands or max_buckets)\n");
        return -1;
    }

    printf("[GM] GPU MCCFR: %d players, %d dec nodes, %s=%d, batch=%d\n",
           NP, num_dec, use_buckets ? "buckets" : "hands", max_s, batch_size);

    /* ── Allocate GPU memory ──────────────────────────────────────── */

    /* Tree */
    GMNode *d_nodes;
    int *d_children;
    int *d_decision_node_map;
    CUDA_CHECK(cudaMalloc(&d_nodes, tree_data->num_nodes * sizeof(GMNode)));
    CUDA_CHECK(cudaMalloc(&d_children, tree_data->num_children_total * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_decision_node_map, num_dec * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_nodes, tree_data->nodes,
                           tree_data->num_nodes * sizeof(GMNode), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_children, tree_data->children,
                           tree_data->num_children_total * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_decision_node_map, tree_data->decision_node_map,
                           num_dec * sizeof(int), cudaMemcpyHostToDevice));

    /* Hands */
    int *d_hands;
    int hands_size = GM_MAX_PLAYERS * GM_MAX_HANDS * 2;
    int *h_hands_flat = (int*)calloc(hands_size, sizeof(int));
    for (int p = 0; p < NP; p++)
        for (int h = 0; h < tree_data->num_hands[p]; h++) {
            h_hands_flat[(p * GM_MAX_HANDS + h) * 2] = tree_data->hands[p][h][0];
            h_hands_flat[(p * GM_MAX_HANDS + h) * 2 + 1] = tree_data->hands[p][h][1];
        }
    CUDA_CHECK(cudaMalloc(&d_hands, hands_size * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_hands, h_hands_flat, hands_size * sizeof(int), cudaMemcpyHostToDevice));
    free(h_hands_flat);

    int *d_num_hands;
    CUDA_CHECK(cudaMalloc(&d_num_hands, GM_MAX_PLAYERS * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_num_hands, tree_data->num_hands,
                           GM_MAX_PLAYERS * sizeof(int), cudaMemcpyHostToDevice));

    /* Flop */
    int *d_flop;
    CUDA_CHECK(cudaMalloc(&d_flop, 3 * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_flop, tree_data->flop, 3 * sizeof(int), cudaMemcpyHostToDevice));

    /* Regrets, strategy, strategy_sum — indexed by slot (hand or bucket) */
    size_t arr_size = (size_t)num_dec * GM_MAX_ACTIONS * max_s * sizeof(float);
    float *d_regrets, *d_strategy, *d_strategy_sum;
    CUDA_CHECK(cudaMalloc(&d_regrets, arr_size));
    CUDA_CHECK(cudaMalloc(&d_strategy, arr_size));
    CUDA_CHECK(cudaMalloc(&d_strategy_sum, arr_size));
    CUDA_CHECK(cudaMemset(d_regrets, 0, arr_size));
    CUDA_CHECK(cudaMemset(d_strategy, 0, arr_size));
    CUDA_CHECK(cudaMemset(d_strategy_sum, 0, arr_size));

    /* Hand-to-bucket mapping (upload to GPU if using buckets) */
    int *d_hand_to_bucket = NULL;
    if (use_buckets) {
        int htb_size = GM_MAX_PLAYERS * GM_MAX_HANDS;
        int *h_htb = (int*)calloc(htb_size, sizeof(int));
        for (int p = 0; p < NP; p++)
            for (int h = 0; h < tree_data->num_hands[p]; h++)
                h_htb[p * GM_MAX_HANDS + h] = tree_data->hand_to_bucket[p][h];
        CUDA_CHECK(cudaMalloc(&d_hand_to_bucket, htb_size * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_hand_to_bucket, h_htb, htb_size * sizeof(int), cudaMemcpyHostToDevice));
        free(h_htb);
    }

    /* Num slots per player (hands or buckets) */
    int *d_num_slots;
    int h_num_slots[GM_MAX_PLAYERS];
    for (int p = 0; p < NP; p++)
        h_num_slots[p] = use_buckets ? tree_data->num_buckets[p] : tree_data->num_hands[p];
    CUDA_CHECK(cudaMalloc(&d_num_slots, GM_MAX_PLAYERS * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_num_slots, h_num_slots, GM_MAX_PLAYERS * sizeof(int), cudaMemcpyHostToDevice));

    printf("[GM] GPU memory: %.1f MB for regrets/strategy (%zu bytes each)\n",
           3.0f * arr_size / (1024.0f * 1024.0f), arr_size);

    /* ── Timing ───────────────────────────────────────────────────── */
    cudaEvent_t start_event, stop_event;
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));
    CUDA_CHECK(cudaEventRecord(start_event));

    /* ── Main iteration loop ──────────────────────────────────────── */

    int threads_per_block = 256;

    for (int iter = 1; iter <= max_iter; iter++) {
        /* Cycle traverser through all players */
        for (int trav = 0; trav < NP; trav++) {

            /* Kernel 1: Regret matching — compute current strategy */
            {
                int blocks = num_dec;
                int threads = (max_s < 256) ? max_s : 256;
                gm_regret_match_kernel<<<blocks, threads>>>(
                    d_decision_node_map, d_nodes,
                    d_regrets, d_strategy,
                    d_num_slots, num_dec, max_s
                );
            }

            /* Kernel 2: Batch traverse — K parallel trajectories */
            {
                int blocks = (batch_size + threads_per_block - 1) / threads_per_block;
                gm_batch_traverse_kernel<<<blocks, threads_per_block>>>(
                    d_nodes, d_children, d_decision_node_map,
                    d_strategy, d_regrets, d_strategy_sum,
                    d_hands, d_num_hands, d_flop,
                    d_hand_to_bucket, use_buckets,
                    NP, max_s, trav,
                    iter * NP + trav,
                    eps, batch_size, num_dec
                );
            }
        }

        /* Kernel 3: CFR+ regret floor (clamp negatives to 0) */
        {
            int total_elems = num_dec * GM_MAX_ACTIONS * max_s;
            int blocks = (total_elems + threads_per_block - 1) / threads_per_block;
            gm_regret_floor_kernel<<<blocks, threads_per_block>>>(
                d_regrets, num_dec, max_s
            );
        }

        CUDA_CHECK(cudaGetLastError());

        if (config->print_every > 0 &&
            (iter % config->print_every == 0 || iter == 1 || iter == max_iter)) {
            CUDA_CHECK(cudaDeviceSynchronize());
            printf("[GM] iter %d/%d, trajectories so far: %d\n",
                   iter, max_iter, iter * NP * batch_size);
            fflush(stdout);
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    /* ── Timing ───────────────────────────────────────────────────── */
    CUDA_CHECK(cudaEventRecord(stop_event));
    CUDA_CHECK(cudaEventSynchronize(stop_event));
    float elapsed_ms;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event));
    printf("[GM] Solve time: %.1f ms (%d iterations × %d traversers × %d batch = %lld trajectories)\n",
           elapsed_ms, max_iter, NP, batch_size,
           (long long)max_iter * NP * batch_size);

    /* ── Download results ─────────────────────────────────────────── */
    memset(output, 0, sizeof(GMOutput));
    output->num_decision_nodes = num_dec;
    output->max_hands = max_s;  /* max_slots (hands or buckets) */
    output->num_players = NP;
    output->use_buckets = use_buckets;
    output->max_buckets = use_buckets ? tree_data->max_buckets : 0;
    output->iterations_run = max_iter;
    output->total_trajectories = max_iter * NP * batch_size;
    output->solve_time_ms = elapsed_ms;

    /* Download strategy_sum for average strategy extraction */
    output->avg_strategy = (float*)malloc(arr_size);
    CUDA_CHECK(cudaMemcpy(output->avg_strategy, d_strategy_sum, arr_size, cudaMemcpyDeviceToHost));

    /* Download decision node info */
    output->decision_players = (int*)malloc(num_dec * sizeof(int));
    output->decision_num_actions = (int*)malloc(num_dec * sizeof(int));
    for (int di = 0; di < num_dec; di++) {
        int ni = tree_data->decision_node_map[di];
        output->decision_players[di] = tree_data->nodes[ni].player;
        output->decision_num_actions[di] = tree_data->nodes[ni].num_actions;
    }

    /* ── Cleanup GPU ──────────────────────────────────────────────── */
    cudaFree(d_nodes);
    cudaFree(d_children);
    cudaFree(d_decision_node_map);
    cudaFree(d_hands);
    cudaFree(d_num_hands);
    cudaFree(d_flop);
    cudaFree(d_num_slots);
    if (d_hand_to_bucket) cudaFree(d_hand_to_bucket);
    cudaFree(d_regrets);
    cudaFree(d_strategy);
    cudaFree(d_strategy_sum);
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    return 0;
}

/* ── Strategy extraction ──────────────────────────────────────────── */

int gm_get_strategy(
    const GMOutput *output,
    int decision_idx,
    int hand_idx,
    float *strategy_out
) {
    if (decision_idx < 0 || decision_idx >= output->num_decision_nodes)
        return 0;

    int na = output->decision_num_actions[decision_idx];
    int max_h = output->max_hands;
    int base = decision_idx * GM_MAX_ACTIONS * max_h;

    float sum = 0.0f;
    for (int a = 0; a < na; a++) {
        float v = output->avg_strategy[base + a * max_h + hand_idx];
        v = (v > 0.0f) ? v : 0.0f;
        strategy_out[a] = v;
        sum += v;
    }

    if (sum > 0.0f) {
        for (int a = 0; a < na; a++) strategy_out[a] /= sum;
    } else {
        float u = 1.0f / (float)na;
        for (int a = 0; a < na; a++) strategy_out[a] = u;
    }

    return na;
}

/* ── Tree stats ───────────────────────────────────────────────────── */

void gm_print_tree_stats(const GMTreeData *tree_data) {
    int n_decision = 0, n_chance = 0, n_fold = 0, n_showdown = 0;
    for (int i = 0; i < tree_data->num_nodes; i++) {
        switch (tree_data->nodes[i].type) {
        case GM_NODE_DECISION:  n_decision++; break;
        case GM_NODE_CHANCE:    n_chance++; break;
        case GM_NODE_FOLD:      n_fold++; break;
        case GM_NODE_SHOWDOWN:  n_showdown++; break;
        }
    }
    printf("[GM] Tree stats:\n");
    printf("  Total nodes:    %d\n", tree_data->num_nodes);
    printf("  Decision nodes: %d\n", n_decision);
    printf("  Chance nodes:   %d\n", n_chance);
    printf("  Fold terminals: %d\n", n_fold);
    printf("  Showdowns:      %d\n", n_showdown);
    printf("  Children total: %d\n", tree_data->num_children_total);
    printf("  Players:        %d\n", tree_data->num_players);

    int max_s = tree_data->use_buckets ? tree_data->max_buckets : tree_data->max_hands;
    size_t arr_size = (size_t)n_decision * GM_MAX_ACTIONS * max_s * sizeof(float);
    printf("  Abstraction:    %s (max_slots=%d)\n",
           tree_data->use_buckets ? "BUCKETED" : "exact hands", max_s);
    printf("  Regret array:   %.1f MB\n", arr_size / (1024.0f * 1024.0f));
    printf("  Total GPU mem:  %.1f MB (3 arrays + tree)\n",
           (3.0f * arr_size + tree_data->num_nodes * sizeof(GMNode)) / (1024.0f * 1024.0f));
}

/* ── Cleanup ──────────────────────────────────────────────────────── */

void gm_free_tree(GMTreeData *tree_data) {
    free(tree_data->nodes);
    free(tree_data->children);
    free(tree_data->decision_node_map);
    memset(tree_data, 0, sizeof(GMTreeData));
}

void gm_free_output(GMOutput *output) {
    free(output->avg_strategy);
    free(output->decision_players);
    free(output->decision_num_actions);
    memset(output, 0, sizeof(GMOutput));
}
