/**
 * street_solve.cu — N-player single-street GPU solver
 *
 * Pluribus-style depth-limited search for 2-6 player subgames.
 *
 * Phase 1: CPU builds N-player single-street betting tree
 * Phase 2: Upload to GPU, precompute hand strengths (river only)
 * Phase 3: Level-batched Linear CFR iterations on GPU
 *          (cycles through all N traversers per iteration)
 * Phase 4: Download strategies and EVs for all players
 *
 * N-player generalization:
 *   - Tree nodes track which players are active (not folded)
 *   - Betting rotates through active players in position order
 *   - Fold removes one player, play continues with remaining
 *   - When only 1 player remains, they win (fold terminal)
 *   - Showdown compares all remaining players' hands
 *   - CFR: each iteration traverses for ONE player, cycling through all N
 *   - Reach: one array per opponent (N-1 reach arrays per traverser)
 *   - Continuation strategies: all N players choose sequentially at leaves
 */

#include "street_solve.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        return -1; \
    } \
} while(0)

/* ═══════════════════════════════════════════════════════════════════════
 * PART 1: CPU tree construction (N-player)
 * ═══════════════════════════════════════════════════════════════════════ */

typedef struct {
    SSNode *nodes;
    int num_nodes;
    int cap_nodes;
    int *children;
    int num_children;
    int cap_children;
    int num_leaves;
} SSTreeBuilder;

static void stb_init(SSTreeBuilder *tb) {
    tb->cap_nodes = 512;
    tb->nodes = (SSNode*)malloc(tb->cap_nodes * sizeof(SSNode));
    tb->num_nodes = 0;
    tb->cap_children = 2048;
    tb->children = (int*)malloc(tb->cap_children * sizeof(int));
    tb->num_children = 0;
    tb->num_leaves = 0;
}

static int stb_alloc_node(SSTreeBuilder *tb) {
    if (tb->num_nodes >= tb->cap_nodes) {
        tb->cap_nodes *= 2;
        tb->nodes = (SSNode*)realloc(tb->nodes, tb->cap_nodes * sizeof(SSNode));
    }
    int idx = tb->num_nodes++;
    memset(&tb->nodes[idx], 0, sizeof(SSNode));
    tb->nodes[idx].player = -1;
    tb->nodes[idx].leaf_idx = -1;
    tb->nodes[idx].fold_player = -1;
    return idx;
}

static int stb_alloc_children(SSTreeBuilder *tb, int count) {
    while (tb->num_children + count > tb->cap_children) {
        tb->cap_children *= 2;
        tb->children = (int*)realloc(tb->children, tb->cap_children * sizeof(int));
    }
    int start = tb->num_children;
    tb->num_children += count;
    return start;
}

static void set_node_board(SSNode *n, const int *board, int num_board) {
    for (int i = 0; i < num_board && i < 5; i++) n->board_cards[i] = board[i];
    n->num_board = num_board;
}

static void set_node_active(SSNode *n, const int *active, int num_players) {
    n->num_players = 0;
    for (int i = 0; i < num_players && i < SS_MAX_PLAYERS; i++) {
        n->active_players[i] = active[i];
        if (active[i]) n->num_players++;
    }
}

/* Count active players */
static int count_active(const int *active, int n) {
    int c = 0;
    for (int i = 0; i < n; i++) if (active[i]) c++;
    return c;
}

/* Find next active player after `current` in acting order */
static int next_active_player(const int *acting_order, int num_in_order,
                               const int *active, int num_players,
                               int current_order_idx) {
    for (int i = 1; i <= num_in_order; i++) {
        int idx = (current_order_idx + i) % num_in_order;
        int p = acting_order[idx];
        if (p < num_players && active[p]) return idx;
    }
    return -1; /* shouldn't happen if >1 active */
}

/* Find the max bet among active players */
static int max_bet(const int *bets, const int *active, int n) {
    int mx = 0;
    for (int i = 0; i < n; i++)
        if (active[i] && bets[i] > mx) mx = bets[i];
    return mx;
}

/* Check if all active players have acted and bets are equal */
static int round_complete(const int *bets, const int *active, const int *has_acted,
                          int num_players) {
    int mx = max_bet(bets, active, num_players);
    for (int i = 0; i < num_players; i++) {
        if (!active[i]) continue;
        if (!has_acted[i]) return 0;
        if (bets[i] != mx) return 0;
    }
    return 1;
}

/* Build N-player single-street betting tree recursively.
 * acting_order: array of player indices in position order
 * current_order_idx: index into acting_order for who acts next
 * bets: each player's total bet this street
 * has_acted: whether each player has had at least one chance to act
 * active: which players are still in the hand (not folded)
 * num_raises: raises so far this street */
static int build_street_tree_n(
    SSTreeBuilder *tb, int is_river, int num_players,
    const int *acting_order, int num_in_order,
    int current_order_idx, int pot, int stack,
    int *bets, int *has_acted, int *active,
    int num_raises, const float *bet_sizes, int num_bet_sizes,
    const int *board, int num_board
) {
    int n_active = count_active(active, num_players);

    /* Only 1 player left — they win */
    if (n_active <= 1) {
        int idx = stb_alloc_node(tb);
        tb->nodes[idx].type = SS_NODE_FOLD;
        /* Find the winner (last active player) */
        for (int i = 0; i < num_players; i++)
            if (active[i]) { tb->nodes[idx].player = i; break; }
        tb->nodes[idx].pot = pot;
        for (int i = 0; i < num_players; i++) tb->nodes[idx].bets[i] = bets[i];
        set_node_board(&tb->nodes[idx], board, num_board);
        set_node_active(&tb->nodes[idx], active, num_players);
        return idx;
    }

    /* Check if round is complete */
    if (round_complete(bets, active, has_acted, num_players)) {
        int idx = stb_alloc_node(tb);
        if (is_river) {
            tb->nodes[idx].type = SS_NODE_SHOWDOWN;
        } else {
            tb->nodes[idx].type = SS_NODE_LEAF;
            tb->nodes[idx].leaf_idx = tb->num_leaves++;
        }
        tb->nodes[idx].pot = pot;
        for (int i = 0; i < num_players; i++) tb->nodes[idx].bets[i] = bets[i];
        set_node_board(&tb->nodes[idx], board, num_board);
        set_node_active(&tb->nodes[idx], active, num_players);
        return idx;
    }

    /* Current player to act */
    int acting_player = acting_order[current_order_idx];

    /* Skip folded players */
    if (!active[acting_player]) {
        int next_idx = next_active_player(acting_order, num_in_order,
                                           active, num_players, current_order_idx);
        if (next_idx < 0) {
            /* Shouldn't happen */
            int idx = stb_alloc_node(tb);
            tb->nodes[idx].type = SS_NODE_SHOWDOWN;
            tb->nodes[idx].pot = pot;
            for (int i = 0; i < num_players; i++) tb->nodes[idx].bets[i] = bets[i];
            set_node_board(&tb->nodes[idx], board, num_board);
            set_node_active(&tb->nodes[idx], active, num_players);
            return idx;
        }
        return build_street_tree_n(tb, is_river, num_players,
                                    acting_order, num_in_order,
                                    next_idx, pot, stack,
                                    bets, has_acted, active,
                                    num_raises, bet_sizes, num_bet_sizes,
                                    board, num_board);
    }

    int node = stb_alloc_node(tb);
    tb->nodes[node].type = SS_NODE_DECISION;
    tb->nodes[node].player = acting_player;
    tb->nodes[node].pot = pot;
    for (int i = 0; i < num_players; i++) tb->nodes[node].bets[i] = bets[i];
    set_node_board(&tb->nodes[node], board, num_board);
    set_node_active(&tb->nodes[node], active, num_players);

    int mx = max_bet(bets, active, num_players);
    int to_call = mx - bets[acting_player];
    if (to_call < 0) to_call = 0;

    int temp_children[16];
    int nc = 0;

    int next_order = next_active_player(acting_order, num_in_order,
                                         active, num_players, current_order_idx);
    if (next_order < 0) next_order = current_order_idx; /* fallback */

    /* ── Fold (only if facing a bet) ────────────────── */
    if (to_call > 0) {
        int new_active[SS_MAX_PLAYERS];
        memcpy(new_active, active, num_players * sizeof(int));
        new_active[acting_player] = 0;

        int fold_child = build_street_tree_n(tb, is_river, num_players,
                                              acting_order, num_in_order,
                                              next_order, pot, stack,
                                              bets, has_acted, new_active,
                                              num_raises, bet_sizes, num_bet_sizes,
                                              board, num_board);
        temp_children[nc++] = fold_child;
    }

    /* ── Check or Call ──────────────────────────────── */
    {
        int new_bets[SS_MAX_PLAYERS];
        int new_has_acted[SS_MAX_PLAYERS];
        memcpy(new_bets, bets, num_players * sizeof(int));
        memcpy(new_has_acted, has_acted, num_players * sizeof(int));
        new_bets[acting_player] = mx;
        new_has_acted[acting_player] = 1;
        int new_pot = pot + to_call;

        int call_child = build_street_tree_n(tb, is_river, num_players,
                                              acting_order, num_in_order,
                                              next_order, new_pot, stack,
                                              new_bets, new_has_acted, active,
                                              num_raises, bet_sizes, num_bet_sizes,
                                              board, num_board);
        temp_children[nc++] = call_child;
    }

    /* ── Bet / Raise ────────────────────────────────── */
    if (num_raises < SS_MAX_RAISES && nc < 14) {
        int added_allin = 0;
        for (int i = 0; i < num_bet_sizes && nc < 14; i++) {
            int bet_amount;
            if (to_call == 0)
                bet_amount = (int)(bet_sizes[i] * pot);
            else
                bet_amount = to_call + (int)(bet_sizes[i] * (pot + to_call));

            if (bet_amount >= stack) bet_amount = stack;
            if (bet_amount <= to_call) continue;

            if (bet_amount >= stack) {
                if (added_allin) continue;
                added_allin = 1;
            }

            int new_bets[SS_MAX_PLAYERS];
            int new_has_acted[SS_MAX_PLAYERS];
            memcpy(new_bets, bets, num_players * sizeof(int));
            memcpy(new_has_acted, has_acted, num_players * sizeof(int));
            new_bets[acting_player] = mx + bet_amount;
            new_has_acted[acting_player] = 1;
            /* Reset has_acted for other players (they need to respond) */
            for (int p = 0; p < num_players; p++)
                if (p != acting_player && active[p]) new_has_acted[p] = 0;
            int new_pot = pot + bet_amount;
            int new_stack = stack; /* simplified: same stack for all */

            int raise_child = build_street_tree_n(tb, is_river, num_players,
                                                   acting_order, num_in_order,
                                                   next_order, new_pot, new_stack,
                                                   new_bets, new_has_acted, active,
                                                   num_raises + 1, bet_sizes, num_bet_sizes,
                                                   board, num_board);
            temp_children[nc++] = raise_child;
        }

        /* Explicit all-in if not covered */
        if (!added_allin && stack > to_call && nc < 14) {
            int ba = stack;
            int new_bets[SS_MAX_PLAYERS];
            int new_has_acted[SS_MAX_PLAYERS];
            memcpy(new_bets, bets, num_players * sizeof(int));
            memcpy(new_has_acted, has_acted, num_players * sizeof(int));
            new_bets[acting_player] = mx + ba;
            new_has_acted[acting_player] = 1;
            for (int p = 0; p < num_players; p++)
                if (p != acting_player && active[p]) new_has_acted[p] = 0;
            int new_pot = pot + ba;

            int ai_child = build_street_tree_n(tb, is_river, num_players,
                                                acting_order, num_in_order,
                                                next_order, new_pot, stack,
                                                new_bets, new_has_acted, active,
                                                num_raises + 1, bet_sizes, num_bet_sizes,
                                                board, num_board);
            temp_children[nc++] = ai_child;
        }
    }

    /* Store children */
    int start = stb_alloc_children(tb, nc);
    for (int i = 0; i < nc; i++) tb->children[start + i] = temp_children[i];
    tb->nodes[node].first_child = start;
    tb->nodes[node].num_children = nc;

    return node;
}

/* Expand leaf nodes with N-player continuation strategy decision nodes.
 * For each leaf, add N sequential decision nodes (one per active player),
 * each with 4 children. Total terminals per leaf = 4^N. */
static void expand_continuation_leaves_n(SSTreeBuilder *tb, int num_players,
                                          const int *active_at_root) {
    int orig_count = tb->num_nodes;
    typedef struct {
        int idx; int pot; int bets[SS_MAX_PLAYERS]; int board[5]; int nb;
        int active[SS_MAX_PLAYERS]; int n_active;
    } LeafInfo;

    LeafInfo *leaves = (LeafInfo*)malloc(orig_count * sizeof(LeafInfo));
    int nl = 0;
    for (int i = 0; i < orig_count; i++) {
        if (tb->nodes[i].type != SS_NODE_LEAF) continue;
        LeafInfo *li = &leaves[nl++];
        li->idx = i;
        li->pot = tb->nodes[i].pot;
        memcpy(li->bets, tb->nodes[i].bets, sizeof(li->bets));
        memcpy(li->board, tb->nodes[i].board_cards, sizeof(li->board));
        li->nb = tb->nodes[i].num_board;
        memcpy(li->active, tb->nodes[i].active_players, sizeof(li->active));
        li->n_active = tb->nodes[i].num_players;
    }

    tb->num_leaves = 0;

    for (int li_idx = 0; li_idx < nl; li_idx++) {
        LeafInfo *li = &leaves[li_idx];

        /* Collect active player indices for this leaf */
        int active_list[SS_MAX_PLAYERS];
        int n_act = 0;
        for (int p = 0; p < num_players; p++)
            if (li->active[p]) active_list[n_act++] = p;

        if (n_act == 0) continue;

        /* Build chain of N decision nodes, each with 4 children.
         * We build this recursively: player 0 picks, then player 1 picks, etc.
         * At the bottom, we have 4^N terminal leaf nodes. */

        /* For efficiency with N>2, limit to 4^N terminals.
         * N=2: 16 terminals, N=3: 64, N=4: 256, N=5: 1024, N=6: 4096.
         * For N>3, this gets large. Pluribus handles this by only having
         * 2-3 players remaining at most leaves (others have folded). */

        /* Convert original leaf to first active player's decision */
        int orig = li->idx;
        tb->nodes[orig].type = SS_NODE_DECISION;
        tb->nodes[orig].player = active_list[0];
        tb->nodes[orig].leaf_idx = -1;

        /* Recursive helper: build continuation tree for remaining players */
        /* We'll use iterative approach with a stack to avoid deep recursion */

        /* For simplicity with variable N, we build level by level.
         * Level 0: 1 node (player active_list[0] decides among 4)
         * Level 1: 4 nodes (player active_list[1] decides among 4 each)
         * Level k: 4^k nodes (player active_list[k] decides)
         * Level n_act: 4^n_act terminal leaves */

        /* Track current level's parent nodes */
        int *cur_parents = (int*)malloc(sizeof(int));
        cur_parents[0] = orig;
        int n_parents = 1;

        for (int level = 0; level < n_act; level++) {
            int player = active_list[level];
            int is_last = (level == n_act - 1);
            int n_next = n_parents * SS_NUM_CONT_STRATS;
            int *next_parents = (int*)malloc(n_next * sizeof(int));
            int ni = 0;

            for (int pi = 0; pi < n_parents; pi++) {
                int parent = cur_parents[pi];

                if (level > 0) {
                    /* Set parent as decision node for this player */
                    tb->nodes[parent].type = SS_NODE_DECISION;
                    tb->nodes[parent].player = player;
                }

                int p_children[SS_NUM_CONT_STRATS];
                for (int s = 0; s < SS_NUM_CONT_STRATS; s++) {
                    if (is_last) {
                        /* Terminal leaf */
                        int term = stb_alloc_node(tb);
                        tb->nodes[term].type = SS_NODE_LEAF;
                        tb->nodes[term].leaf_idx = tb->num_leaves++;
                        tb->nodes[term].pot = li->pot;
                        memcpy(tb->nodes[term].bets, li->bets, sizeof(li->bets));
                        set_node_board(&tb->nodes[term], li->board, li->nb);
                        set_node_active(&tb->nodes[term], li->active, num_players);
                        p_children[s] = term;
                    } else {
                        /* Intermediate: next player's decision */
                        int inter = stb_alloc_node(tb);
                        tb->nodes[inter].pot = li->pot;
                        memcpy(tb->nodes[inter].bets, li->bets, sizeof(li->bets));
                        set_node_board(&tb->nodes[inter], li->board, li->nb);
                        set_node_active(&tb->nodes[inter], li->active, num_players);
                        p_children[s] = inter;
                        next_parents[ni++] = inter;
                    }
                }

                int cs = stb_alloc_children(tb, SS_NUM_CONT_STRATS);
                for (int s = 0; s < SS_NUM_CONT_STRATS; s++)
                    tb->children[cs + s] = p_children[s];
                tb->nodes[parent].first_child = cs;
                tb->nodes[parent].num_children = SS_NUM_CONT_STRATS;
            }

            free(cur_parents);
            if (is_last) {
                free(next_parents);
                cur_parents = NULL;
            } else {
                cur_parents = next_parents;
                n_parents = ni;
            }
        }
        if (cur_parents) free(cur_parents);
    }

    free(leaves);
}


/* ═══════════════════════════════════════════════════════════════════════
 * PART 2: GPU hand evaluation (same as before)
 * ═══════════════════════════════════════════════════════════════════════ */

__device__ uint32_t ss_eval5(int c0, int c1, int c2, int c3, int c4) {
    int r[5] = {c0 >> 2, c1 >> 2, c2 >> 2, c3 >> 2, c4 >> 2};
    int s[5] = {c0 & 3, c1 & 3, c2 & 3, c3 & 3, c4 & 3};
    for (int i = 1; i < 5; i++) {
        int k = r[i], j = i - 1;
        while (j >= 0 && r[j] < k) { r[j + 1] = r[j]; j--; }
        r[j + 1] = k;
    }
    int fl = (s[0] == s[1] && s[1] == s[2] && s[2] == s[3] && s[3] == s[4]);
    int st = 0, sh = r[0];
    if (r[0] - r[4] == 4 && r[0] != r[1] && r[1] != r[2] && r[2] != r[3] && r[3] != r[4]) st = 1;
    if (r[0] == 12 && r[1] == 3 && r[2] == 2 && r[3] == 1 && r[4] == 0) { st = 1; sh = 3; }
    if (st && fl) return (9u << 20) | (sh << 16);
    if (fl) return (6u << 20) | (r[0] << 16) | (r[1] << 12) | (r[2] << 8) | (r[3] << 4) | r[4];
    if (st) return (5u << 20) | (sh << 16);
    int cn[13] = {0};
    for (int i = 0; i < 5; i++) cn[r[i]]++;
    int q = -1, t = -1, p1 = -1, p2 = -1;
    for (int i = 12; i >= 0; i--) {
        if (cn[i] == 4) q = i;
        else if (cn[i] == 3) t = i;
        else if (cn[i] == 2) { if (p1 < 0) p1 = i; else p2 = i; }
    }
    if (q >= 0) { int k = -1; for (int i = 12; i >= 0; i--) if (cn[i] > 0 && i != q) { k = i; break; } return (8u << 20) | (q << 16) | (k << 12); }
    if (t >= 0 && p1 >= 0) return (7u << 20) | (t << 16) | (p1 << 12);
    if (t >= 0) { int k0 = -1, k1 = -1; for (int i = 12; i >= 0; i--) if (cn[i] > 0 && i != t) { if (k0 < 0) k0 = i; else k1 = i; } return (4u << 20) | (t << 16) | (k0 << 12) | (k1 << 8); }
    if (p1 >= 0 && p2 >= 0) { int k = -1; for (int i = 12; i >= 0; i--) if (cn[i] > 0 && i != p1 && i != p2) { k = i; break; } return (3u << 20) | (p1 << 16) | (p2 << 12) | (k << 8); }
    if (p1 >= 0) { int k[3], ki = 0; for (int i = 12; i >= 0 && ki < 3; i--) if (cn[i] > 0 && i != p1) k[ki++] = i; return (2u << 20) | (p1 << 16) | (k[0] << 12) | (k[1] << 8) | (k[2] << 4); }
    return (1u << 20) | (r[0] << 16) | (r[1] << 12) | (r[2] << 8) | (r[3] << 4) | r[4];
}

__device__ uint32_t ss_eval7(const int c[7]) {
    const int cb[21][5] = {
        {0,1,2,3,4},{0,1,2,3,5},{0,1,2,3,6},{0,1,2,4,5},{0,1,2,4,6},{0,1,2,5,6},
        {0,1,3,4,5},{0,1,3,4,6},{0,1,3,5,6},{0,1,4,5,6},{0,2,3,4,5},{0,2,3,4,6},
        {0,2,3,5,6},{0,2,4,5,6},{0,3,4,5,6},{1,2,3,4,5},{1,2,3,4,6},{1,2,3,5,6},
        {1,2,4,5,6},{1,3,4,5,6},{2,3,4,5,6}};
    uint32_t b = 0;
    for (int i = 0; i < 21; i++) {
        uint32_t v = ss_eval5(c[cb[i][0]], c[cb[i][1]], c[cb[i][2]], c[cb[i][3]], c[cb[i][4]]);
        if (v > b) b = v;
    }
    return b;
}

/* Precompute hand strengths for all players at showdown nodes */
__global__ void ss_precompute_strengths(
    const SSNode *nodes, const int *showdown_indices, int num_showdowns,
    const int *hands,  /* [SS_MAX_PLAYERS * SS_MAX_HANDS * 2] */
    const int *num_hands_per_player, /* [SS_MAX_PLAYERS] */
    int num_players,
    uint32_t *strengths  /* [num_showdowns][SS_MAX_PLAYERS][SS_MAX_HANDS] */
) {
    int sd_idx = blockIdx.x;
    int hand = threadIdx.x;
    if (sd_idx >= num_showdowns) return;

    int node_idx = showdown_indices[sd_idx];
    const SSNode *node = &nodes[node_idx];

    for (int p = 0; p < num_players; p++) {
        int nh = num_hands_per_player[p];
        if (hand >= nh) continue;
        if (!node->active_players[p]) {
            strengths[(sd_idx * SS_MAX_PLAYERS + p) * SS_MAX_HANDS + hand] = 0;
            continue;
        }

        int hc0 = hands[(p * SS_MAX_HANDS + hand) * 2];
        int hc1 = hands[(p * SS_MAX_HANDS + hand) * 2 + 1];

        int blocked = 0;
        for (int b = 0; b < node->num_board; b++)
            if (hc0 == node->board_cards[b] || hc1 == node->board_cards[b]) { blocked = 1; break; }

        if (blocked) {
            strengths[(sd_idx * SS_MAX_PLAYERS + p) * SS_MAX_HANDS + hand] = 0;
        } else {
            int c7[7] = {node->board_cards[0], node->board_cards[1], node->board_cards[2],
                         node->board_cards[3], node->board_cards[4], hc0, hc1};
            strengths[(sd_idx * SS_MAX_PLAYERS + p) * SS_MAX_HANDS + hand] = ss_eval7(c7);
        }
    }
}


/* ═══════════════════════════════════════════════════════════════════════
 * PART 3: GPU kernels — N-player level-batched CFR
 *
 * Key difference from 2-player: each kernel takes `traverser` and
 * `num_players` parameters. Reach/CFV arrays are indexed by
 * [node_idx * max_hands + hand] for the traverser's opponent reach.
 * Since we have N-1 opponents, we compute a single "opponent reach
 * product" per hand: product of all opponents' reach probabilities.
 * But actually in poker CFR, the "opponent reach" for a traverser
 * is the product of ALL other players' reach. For N players, this
 * is: reach_-i = product_{j != i} reach_j(h_j).
 *
 * For the GPU, we maintain one reach array per player:
 *   reach[player * num_nodes * max_hands + node * max_hands + hand]
 * and the opponent-reach for fold/showdown is computed by summing
 * over non-conflicting opponent hand combos × their reach products.
 * ═══════════════════════════════════════════════════════════════════════ */

/* Regret-based pruning threshold */
#define SS_PRUNE_THRESHOLD (-10000.0f)

/* Batched regret matching at decision nodes */
__global__ void ss_batch_regret_match(
    const SSNode *nodes, const int *batch_nodes, int batch_size,
    float *regrets, float *strategy, int max_hands,
    const int *num_hands_arr, int num_players, const int *d_iteration
) {
    int bi = blockIdx.x;
    int hand = threadIdx.x;
    if (bi >= batch_size) return;
    int node_idx = batch_nodes[bi];
    const SSNode *n = &nodes[node_idx];
    if (n->type != SS_NODE_DECISION) return;
    int acting = n->player;
    if (acting < 0 || acting >= num_players) return;
    int nh = num_hands_arr[acting];
    if (hand >= nh) return;
    int na = n->num_children;
    int base = node_idx * SS_MAX_ACTIONS * max_hands;

    int iteration = *d_iteration;
    int prune_active = (iteration > 10) && (((hand * 7 + iteration) % 20) != 0);

    float sum = 0;
    for (int a = 0; a < na; a++) {
        float r = regrets[base + a * max_hands + hand];
        if (prune_active && r < SS_PRUNE_THRESHOLD) r = 0;
        sum += (r > 0) ? r : 0;
    }
    if (sum > 0) {
        float inv = 1.0f / sum;
        for (int a = 0; a < na; a++) {
            float r = regrets[base + a * max_hands + hand];
            if (prune_active && r < SS_PRUNE_THRESHOLD) r = 0;
            strategy[base + a * max_hands + hand] = ((r > 0) ? r : 0) * inv;
        }
    } else {
        float u = 1.0f / na;
        for (int a = 0; a < na; a++)
            strategy[base + a * max_hands + hand] = u;
    }
}

/* Batched reach propagation for decision nodes.
 * `reach_player` is the player whose reach we're propagating.
 * If this player is acting at the node, multiply reach by strategy.
 * Otherwise, copy reach unchanged to children. */
__global__ void ss_batch_propagate_reach(
    const SSNode *nodes, const int *children_arr, const int *batch_nodes, int batch_size,
    float *reach, const float *strategy, int reach_player, int max_hands,
    const int *num_hands_arr, int num_players
) {
    int bi = blockIdx.x;
    int hand = threadIdx.x;
    if (bi >= batch_size) return;
    int node_idx = batch_nodes[bi];
    const SSNode *n = &nodes[node_idx];
    if (n->type != SS_NODE_DECISION) return;
    int nh = num_hands_arr[reach_player];
    if (hand >= nh) return;
    int na = n->num_children;
    int acting = n->player;

    /* Reach array is per-player: reach[node * max_hands + hand] */
    float pr = reach[node_idx * max_hands + hand];

    if (acting == reach_player) {
        /* Acting player: multiply by strategy */
        int base = node_idx * SS_MAX_ACTIONS * max_hands;
        for (int a = 0; a < na; a++) {
            int child = children_arr[n->first_child + a];
            reach[child * max_hands + hand] = pr * strategy[base + a * max_hands + hand];
        }
    } else {
        /* Non-acting player: copy reach to all children */
        for (int a = 0; a < na; a++) {
            int child = children_arr[n->first_child + a];
            reach[child * max_hands + hand] = pr;
        }
    }
}

/* N-player fold value.
 * When a player folds, remaining players split the pot.
 * In practice for our tree: when only 1 player remains (all others folded),
 * that player wins. Otherwise the fold just removes one player and play continues
 * (handled by the tree structure, not this kernel).
 *
 * For CFR, the fold terminal value for the traverser:
 *   If traverser is the last remaining player → wins the pot
 *   If traverser folded → loses their bet
 *   If traverser is still in but not last → handled by tree continuation
 *
 * Our tree structure creates fold terminals only when n_active==1.
 * The winner is stored in node->player.
 *
 * The traverser's payoff is constant for a given hand (does not depend on
 * opponent hand strengths), so cfv = payoff * sum_over_valid_opponent_combos(reach_product).
 * For N=2 this is just payoff * sum(opp_reach).
 * For N>2 we enumerate opponent hand combos to correctly exclude
 * inter-opponent card conflicts. */
__global__ void ss_batch_fold_value(
    const SSNode *nodes, const int *batch_nodes, int batch_size,
    float *cfv,
    const float *reach_all,  /* [num_players][num_nodes * max_hands] */
    int traverser, int max_hands,
    const int *num_hands_arr, int num_players,
    const int *hands, int starting_pot,
    int num_nodes  /* for reach indexing */
) {
    int bi = blockIdx.x;
    int hand = threadIdx.x;
    if (bi >= batch_size) return;
    int node_idx = batch_nodes[bi];
    const SSNode *node = &nodes[node_idx];
    int nh_trav = num_hands_arr[traverser];
    if (hand >= nh_trav) return;

    int winner = node->player;  /* last remaining player */
    int hc0 = hands[(traverser * SS_MAX_HANDS + hand) * 2];
    int hc1 = hands[(traverser * SS_MAX_HANDS + hand) * 2 + 1];

    /* Payoff for traverser */
    float payoff;
    if (traverser == winner) {
        /* Traverser wins — gets pot minus their own contribution */
        payoff = (float)(node->pot - node->bets[traverser]);
    } else {
        /* Traverser lost (folded earlier or someone else won) */
        payoff = -(float)node->bets[traverser];
    }

    /* Collect non-traverser players (all players except traverser, regardless
     * of active status — folded players still have reach that matters for
     * the card-removal weighting) */
    int opp[SS_MAX_PLAYERS];
    int n_opp = 0;
    for (int p = 0; p < num_players; p++) {
        if (p == traverser) continue;
        opp[n_opp++] = p;
    }

    /* ── Fast path: single opponent (N=2) ── */
    if (n_opp == 1) {
        int p = opp[0];
        int nh_p = num_hands_arr[p];
        float opp_sum = 0;
        for (int o = 0; o < nh_p; o++) {
            int oc0 = hands[(p * SS_MAX_HANDS + o) * 2];
            int oc1 = hands[(p * SS_MAX_HANDS + o) * 2 + 1];
            if (hc0 == oc0 || hc0 == oc1 || hc1 == oc0 || hc1 == oc1) continue;
            int blocked = 0;
            for (int b = 0; b < node->num_board; b++)
                if (oc0 == node->board_cards[b] || oc1 == node->board_cards[b]) { blocked = 1; break; }
            if (blocked) continue;
            opp_sum += reach_all[p * num_nodes * max_hands + node_idx * max_hands + o];
        }
        cfv[node_idx * max_hands + hand] = payoff * opp_sum;
        return;
    }

    /* ── Exact N-player: enumerate opponent hand combos ── */
    int nh_opp[SS_MAX_PLAYERS];
    for (int i = 0; i < n_opp; i++)
        nh_opp[i] = num_hands_arr[opp[i]];

    float total_reach = 0;

    /* Odometer iteration over all opponent hand combos */
    int idx[SS_MAX_PLAYERS - 1];
    for (int i = 0; i < n_opp; i++) idx[i] = 0;

    while (1) {
        int opp_cards[SS_MAX_PLAYERS - 1][2];
        float reach_prod = 1.0f;
        int valid = 1;

        for (int i = 0; i < n_opp; i++) {
            int p = opp[i];
            int o = idx[i];
            int oc0 = hands[(p * SS_MAX_HANDS + o) * 2];
            int oc1 = hands[(p * SS_MAX_HANDS + o) * 2 + 1];
            opp_cards[i][0] = oc0;
            opp_cards[i][1] = oc1;

            /* Check conflict with traverser's cards */
            if (hc0 == oc0 || hc0 == oc1 || hc1 == oc0 || hc1 == oc1) { valid = 0; break; }

            /* Check conflict with board */
            for (int b = 0; b < node->num_board; b++)
                if (oc0 == node->board_cards[b] || oc1 == node->board_cards[b]) { valid = 0; break; }
            if (!valid) break;

            float w = reach_all[p * num_nodes * max_hands + node_idx * max_hands + o];
            reach_prod *= w;

            /* Check conflict with all previous opponents' cards */
            for (int j = 0; j < i; j++) {
                if (oc0 == opp_cards[j][0] || oc0 == opp_cards[j][1] ||
                    oc1 == opp_cards[j][0] || oc1 == opp_cards[j][1]) {
                    valid = 0;
                    break;
                }
            }
            if (!valid) break;
        }

        if (valid && reach_prod > 0.0f) {
            total_reach += reach_prod;
        }

        /* Advance to next combo (odometer increment) */
        int carry = n_opp - 1;
        while (carry >= 0) {
            idx[carry]++;
            if (idx[carry] < nh_opp[carry]) break;
            idx[carry] = 0;
            carry--;
        }
        if (carry < 0) break;
    }

    cfv[node_idx * max_hands + hand] = payoff * total_reach;
}

/* N-player showdown value (2-player fast path + exact N-player).
 * Compare traverser's hand against ALL remaining opponents simultaneously.
 * Pot is split among players with the highest hand strength.
 *
 * For N=2: O(M) pairwise comparison (unchanged).
 * For N>2: Enumerate all opponent hand combinations O(M^(N-1)).
 *   For each combo, check card conflicts, find the max strength among all
 *   active players, and award pot to winner(s) (split on tie).
 *   Traverser's EV = sum over combos of (payoff * product of opp reaches).
 *
 * In practice N>2 showdowns mostly have 3 players (rarely 4+), since
 * most players fold before showdown, so the inner loop is ~M^2 = 40K. */
__global__ void ss_batch_showdown_value(
    const SSNode *nodes, const int *batch_nodes, int batch_size,
    float *cfv,
    const float *reach_all,
    int traverser, int max_hands,
    const int *num_hands_arr, int num_players,
    const int *hands,
    const uint32_t *strengths, const int *sd_local_map,
    int num_nodes
) {
    int bi = blockIdx.x;
    int hand = threadIdx.x;
    if (bi >= batch_size) return;
    int node_idx = batch_nodes[bi];
    const SSNode *node = &nodes[node_idx];
    int nh_trav = num_hands_arr[traverser];
    if (hand >= nh_trav) return;
    if (!node->active_players[traverser]) { cfv[node_idx * max_hands + hand] = 0; return; }

    int sd_idx = sd_local_map[node_idx];
    uint32_t hs = strengths[(sd_idx * SS_MAX_PLAYERS + traverser) * SS_MAX_HANDS + hand];
    if (hs == 0) { cfv[node_idx * max_hands + hand] = 0; return; }

    int hc0 = hands[(traverser * SS_MAX_HANDS + hand) * 2];
    int hc1 = hands[(traverser * SS_MAX_HANDS + hand) * 2 + 1];

    /* Collect active opponents */
    int opp[SS_MAX_PLAYERS];
    int n_opp = 0;
    for (int p = 0; p < num_players; p++) {
        if (p == traverser) continue;
        if (!node->active_players[p]) continue;
        opp[n_opp++] = p;
    }

    /* Traverser's payoff: pot won minus amount invested.
     * If traverser wins outright: pot - bets[traverser]
     * If traverser ties with k others: pot/k - bets[traverser]  (k = number of winners including traverser)
     * If traverser loses: -bets[traverser] */
    float pot = (float)node->pot;
    float trav_bet = (float)node->bets[traverser];

    /* ────────────────────────────────────────────────────────
     * Fast path: N=2 (exactly one opponent)
     * This is the original correct logic, just cleaned up.
     * ──────────────────────────────────────────────────────── */
    if (n_opp == 1) {
        int p = opp[0];
        int nh_p = num_hands_arr[p];
        float val = 0;
        for (int o = 0; o < nh_p; o++) {
            int oc0 = hands[(p * SS_MAX_HANDS + o) * 2];
            int oc1 = hands[(p * SS_MAX_HANDS + o) * 2 + 1];
            if (hc0 == oc0 || hc0 == oc1 || hc1 == oc0 || hc1 == oc1) continue;
            uint32_t os = strengths[(sd_idx * SS_MAX_PLAYERS + p) * SS_MAX_HANDS + o];
            if (os == 0) continue;
            float w = reach_all[p * num_nodes * max_hands + node_idx * max_hands + o];
            if (hs > os)      val += w * (pot - trav_bet);
            else if (hs < os) val -= w * trav_bet;
            /* tie: val += w * (pot / 2.0f - trav_bet) -- i.e. split pot */
            else              val += w * (pot * 0.5f - trav_bet);
        }
        cfv[node_idx * max_hands + hand] = val;
        return;
    }

    /* ────────────────────────────────────────────────────────
     * Exact N-player showdown (n_opp >= 2)
     *
     * We enumerate all combinations of opponent hands.
     * For n_opp==2 this is O(M^2); for n_opp==3, O(M^3), etc.
     * We handle up to SS_MAX_PLAYERS-1 = 5 opponents via a
     * recursive-iteration approach using a stack of indices.
     *
     * For each combination:
     *   1. Check card conflicts between all opponents
     *   2. Compute reach product = product of each opp's reach
     *   3. Find max strength among traverser + all opponents
     *   4. Count how many players share that max strength
     *   5. Traverser's payoff = (pot / n_winners) - trav_bet if winner,
     *      else -trav_bet
     * ──────────────────────────────────────────────────────── */

    /* Pre-cache per-opponent data for the inner loops */
    int nh_opp[SS_MAX_PLAYERS];
    for (int i = 0; i < n_opp; i++)
        nh_opp[i] = num_hands_arr[opp[i]];

    float val = 0;

    /* Iterative enumeration of opponent hand combos using index stack.
     * idx[i] = hand index for opponent i. We iterate in odometer order. */
    int idx[SS_MAX_PLAYERS - 1];  /* max 5 opponents */
    for (int i = 0; i < n_opp; i++) idx[i] = 0;

    /* Outer loop: iterate over all combos */
    while (1) {
        /* ── Gather this combo's cards, strengths, reaches ── */
        int opp_cards[SS_MAX_PLAYERS - 1][2];
        uint32_t opp_str[SS_MAX_PLAYERS - 1];
        float reach_prod = 1.0f;
        int valid = 1;

        for (int i = 0; i < n_opp; i++) {
            int p = opp[i];
            int o = idx[i];
            int oc0 = hands[(p * SS_MAX_HANDS + o) * 2];
            int oc1 = hands[(p * SS_MAX_HANDS + o) * 2 + 1];
            opp_cards[i][0] = oc0;
            opp_cards[i][1] = oc1;

            /* Check conflict with traverser's cards */
            if (hc0 == oc0 || hc0 == oc1 || hc1 == oc0 || hc1 == oc1) { valid = 0; break; }

            uint32_t os = strengths[(sd_idx * SS_MAX_PLAYERS + p) * SS_MAX_HANDS + o];
            if (os == 0) { valid = 0; break; }
            opp_str[i] = os;

            float w = reach_all[p * num_nodes * max_hands + node_idx * max_hands + o];
            reach_prod *= w;

            /* Check conflict with all previous opponents' cards */
            for (int j = 0; j < i; j++) {
                if (oc0 == opp_cards[j][0] || oc0 == opp_cards[j][1] ||
                    oc1 == opp_cards[j][0] || oc1 == opp_cards[j][1]) {
                    valid = 0;
                    break;
                }
            }
            if (!valid) break;
        }

        if (valid && reach_prod > 0.0f) {
            /* Find max strength among traverser + all opponents */
            uint32_t max_str = hs;  /* traverser's strength */
            for (int i = 0; i < n_opp; i++) {
                if (opp_str[i] > max_str) max_str = opp_str[i];
            }

            /* Determine traverser's payoff */
            if (hs < max_str) {
                /* Traverser loses */
                val += reach_prod * (-trav_bet);
            } else {
                /* Traverser has max strength; count total winners (including traverser) */
                int n_winners = 1;  /* traverser */
                for (int i = 0; i < n_opp; i++) {
                    if (opp_str[i] == max_str) n_winners++;
                }
                val += reach_prod * (pot / (float)n_winners - trav_bet);
            }
        }

        /* ── Advance to next combo (odometer increment) ── */
        int carry = n_opp - 1;
        while (carry >= 0) {
            idx[carry]++;
            if (idx[carry] < nh_opp[carry]) break;
            idx[carry] = 0;
            carry--;
        }
        if (carry < 0) break;  /* all combos exhausted */
    }

    cfv[node_idx * max_hands + hand] = val;
}

/* Leaf value kernel — reads external leaf values.
 * leaf_values layout: [leaf_idx * num_players * max_hands + player * max_hands + hand]
 *
 * For N>2, enumerate opponent hand combos to correctly handle
 * inter-opponent card conflicts (same approach as fold kernel). */
__global__ void ss_batch_leaf_value(
    const SSNode *nodes, const int *batch_nodes, int batch_size,
    float *cfv,
    const float *reach_all,
    int traverser, int max_hands,
    const int *num_hands_arr, int num_players,
    const float *leaf_values,
    const int *hands,
    int num_nodes
) {
    int bi = blockIdx.x;
    int hand = threadIdx.x;
    if (bi >= batch_size) return;
    int node_idx = batch_nodes[bi];
    const SSNode *node = &nodes[node_idx];
    int nh_trav = num_hands_arr[traverser];
    if (hand >= nh_trav) return;
    int leaf_idx = node->leaf_idx;
    if (leaf_idx < 0) { cfv[node_idx * max_hands + hand] = 0; return; }

    float ev = leaf_values[leaf_idx * num_players * max_hands + traverser * max_hands + hand];

    int hc0 = hands[(traverser * SS_MAX_HANDS + hand) * 2];
    int hc1 = hands[(traverser * SS_MAX_HANDS + hand) * 2 + 1];

    /* Collect active opponents */
    int opp[SS_MAX_PLAYERS];
    int n_opp = 0;
    for (int p = 0; p < num_players; p++) {
        if (p == traverser) continue;
        if (!node->active_players[p]) continue;
        opp[n_opp++] = p;
    }

    /* ── Fast path: single opponent (N=2) ── */
    if (n_opp <= 1) {
        float opp_sum = 0;
        if (n_opp == 1) {
            int p = opp[0];
            int nh_p = num_hands_arr[p];
            for (int o = 0; o < nh_p; o++) {
                int oc0 = hands[(p * SS_MAX_HANDS + o) * 2];
                int oc1 = hands[(p * SS_MAX_HANDS + o) * 2 + 1];
                if (hc0 == oc0 || hc0 == oc1 || hc1 == oc0 || hc1 == oc1) continue;
                int blocked = 0;
                for (int b = 0; b < node->num_board; b++)
                    if (oc0 == node->board_cards[b] || oc1 == node->board_cards[b]) { blocked = 1; break; }
                if (blocked) continue;
                opp_sum += reach_all[p * num_nodes * max_hands + node_idx * max_hands + o];
            }
        } else {
            opp_sum = 1.0f;  /* no opponents active */
        }
        cfv[node_idx * max_hands + hand] = ev * opp_sum;
        return;
    }

    /* ── Exact N-player: enumerate opponent hand combos ── */
    int nh_opp[SS_MAX_PLAYERS];
    for (int i = 0; i < n_opp; i++)
        nh_opp[i] = num_hands_arr[opp[i]];

    float total_reach = 0;

    int idx[SS_MAX_PLAYERS - 1];
    for (int i = 0; i < n_opp; i++) idx[i] = 0;

    while (1) {
        int opp_cards[SS_MAX_PLAYERS - 1][2];
        float reach_prod = 1.0f;
        int valid = 1;

        for (int i = 0; i < n_opp; i++) {
            int p = opp[i];
            int o = idx[i];
            int oc0 = hands[(p * SS_MAX_HANDS + o) * 2];
            int oc1 = hands[(p * SS_MAX_HANDS + o) * 2 + 1];
            opp_cards[i][0] = oc0;
            opp_cards[i][1] = oc1;

            /* Check conflict with traverser's cards */
            if (hc0 == oc0 || hc0 == oc1 || hc1 == oc0 || hc1 == oc1) { valid = 0; break; }

            /* Check conflict with board */
            for (int b = 0; b < node->num_board; b++)
                if (oc0 == node->board_cards[b] || oc1 == node->board_cards[b]) { valid = 0; break; }
            if (!valid) break;

            float w = reach_all[p * num_nodes * max_hands + node_idx * max_hands + o];
            reach_prod *= w;

            /* Check conflict with all previous opponents' cards */
            for (int j = 0; j < i; j++) {
                if (oc0 == opp_cards[j][0] || oc0 == opp_cards[j][1] ||
                    oc1 == opp_cards[j][0] || oc1 == opp_cards[j][1]) {
                    valid = 0;
                    break;
                }
            }
            if (!valid) break;
        }

        if (valid && reach_prod > 0.0f) {
            total_reach += reach_prod;
        }

        /* Advance to next combo (odometer increment) */
        int carry = n_opp - 1;
        while (carry >= 0) {
            idx[carry]++;
            if (idx[carry] < nh_opp[carry]) break;
            idx[carry] = 0;
            carry--;
        }
        if (carry < 0) break;
    }

    cfv[node_idx * max_hands + hand] = ev * total_reach;
}

/* CFV propagation + regret update (bottom-up).
 * For N-player CFR: only the traverser's regrets are updated.
 * CFV propagation at decision nodes where traverser acts:
 *   cfv[node] = sum_a strategy[a] * cfv[child_a]
 * At opponent decision nodes:
 *   cfv[node] = sum_a cfv[child_a]  (opponent's strategy already in reach) */
__global__ void ss_batch_propagate_cfv(
    const SSNode *nodes, const int *children_arr, const int *batch_nodes, int batch_size,
    float *cfv, float *regrets, float *strategy_sum, const float *strategy,
    int traverser, int max_hands, const int *num_hands_arr, int num_players,
    const int *d_iteration
) {
    int iteration = *d_iteration;
    int bi = blockIdx.x;
    int hand = threadIdx.x;
    if (bi >= batch_size) return;
    int node_idx = batch_nodes[bi];
    const SSNode *n = &nodes[node_idx];
    if (n->type != SS_NODE_DECISION) return;
    int nh_trav = num_hands_arr[traverser];
    if (hand >= nh_trav) return;
    int na = n->num_children;

    if (n->player == traverser) {
        float val = 0;
        int base = node_idx * SS_MAX_ACTIONS * max_hands;
        for (int a = 0; a < na; a++) {
            int child = children_arr[n->first_child + a];
            val += strategy[base + a * max_hands + hand] * cfv[child * max_hands + hand];
        }
        cfv[node_idx * max_hands + hand] = val;
        /* Update regrets */
        for (int a = 0; a < na; a++) {
            int child = children_arr[n->first_child + a];
            regrets[base + a * max_hands + hand] += cfv[child * max_hands + hand] - val;
        }
        /* Linear CFR discount */
        float d = (float)iteration / ((float)iteration + 1.0f);
        for (int a = 0; a < na; a++)
            regrets[base + a * max_hands + hand] *= d;
        /* Accumulate weighted strategy sum */
        for (int a = 0; a < na; a++)
            strategy_sum[base + a * max_hands + hand] +=
                (float)iteration * strategy[base + a * max_hands + hand];
    } else {
        /* Non-traverser decision node: cfv = sum of children cfv */
        float val = 0;
        for (int a = 0; a < na; a++) {
            int child = children_arr[n->first_child + a];
            val += cfv[child * max_hands + hand];
        }
        cfv[node_idx * max_hands + hand] = val;
    }
}


/* ═══════════════════════════════════════════════════════════════════════
 * PART 4: Host orchestration
 * ═══════════════════════════════════════════════════════════════════════ */

extern "C" SS_EXPORT int ss_build_tree(
    const int *board, int num_board,
    int starting_pot, int effective_stack,
    const float *bet_sizes, int num_bet_sizes,
    int num_players, const int *acting_order,
    int is_river, int use_cont_strats,
    SSTreeData *tree_out
) {
    memset(tree_out, 0, sizeof(SSTreeData));

    SSTreeBuilder tb;
    stb_init(&tb);

    /* Initialize betting state */
    int bets[SS_MAX_PLAYERS] = {0};
    int has_acted[SS_MAX_PLAYERS] = {0};
    int active[SS_MAX_PLAYERS] = {0};
    for (int i = 0; i < num_players; i++) active[i] = 1;

    build_street_tree_n(&tb, is_river, num_players,
                        acting_order, num_players,
                        0, starting_pot, effective_stack,
                        bets, has_acted, active,
                        0, bet_sizes, num_bet_sizes, board, num_board);

    printf("[SS] After betting tree: %d nodes, %d leaves\n", tb.num_nodes, tb.num_leaves);

    if (use_cont_strats && !is_river && tb.num_leaves > 0) {
        expand_continuation_leaves_n(&tb, num_players, active);
        printf("[SS] After cont strat expansion: %d nodes, %d terminal leaves\n",
               tb.num_nodes, tb.num_leaves);
    }

    /* Transfer to output */
    tree_out->nodes = tb.nodes;
    tree_out->children = tb.children;
    tree_out->num_nodes = tb.num_nodes;
    tree_out->num_children_total = tb.num_children;
    tree_out->num_leaves = tb.num_leaves;
    tree_out->is_river = is_river;
    tree_out->starting_pot = starting_pot;
    tree_out->effective_stack = effective_stack;
    tree_out->num_players = num_players;
    for (int i = 0; i < num_board; i++) tree_out->board[i] = board[i];
    tree_out->num_board = num_board;

    /* BFS level order */
    int N = tb.num_nodes;
    tree_out->level_order = (int*)malloc(N * sizeof(int));
    tree_out->node_depth = (int*)calloc(N, sizeof(int));
    {
        int *queue = (int*)malloc(N * sizeof(int));
        int qh = 0, qt = 0, li = 0;
        queue[qt++] = 0;
        while (qh < qt) {
            int n = queue[qh++];
            tree_out->level_order[li++] = n;
            int d = tree_out->node_depth[n];
            if (d > tree_out->max_depth) tree_out->max_depth = d;
            SSNode *nd = &tb.nodes[n];
            for (int a = 0; a < nd->num_children; a++) {
                int child = tb.children[nd->first_child + a];
                tree_out->node_depth[child] = d + 1;
                queue[qt++] = child;
            }
        }
        free(queue);
    }

    /* Classify nodes */
    tree_out->num_decision_nodes = 0;
    tree_out->num_showdown_nodes = 0;
    tree_out->num_leaf_nodes = 0;
    tree_out->num_fold_nodes = 0;
    for (int i = 0; i < N; i++) {
        switch (tb.nodes[i].type) {
            case SS_NODE_DECISION: tree_out->num_decision_nodes++; break;
            case SS_NODE_SHOWDOWN: tree_out->num_showdown_nodes++; break;
            case SS_NODE_LEAF:     tree_out->num_leaf_nodes++; break;
            case SS_NODE_FOLD:     tree_out->num_fold_nodes++; break;
        }
    }
    tree_out->decision_node_indices = (int*)malloc((tree_out->num_decision_nodes + 1) * sizeof(int));
    tree_out->showdown_node_indices = (int*)malloc((tree_out->num_showdown_nodes + 1) * sizeof(int));
    tree_out->leaf_node_indices = (int*)malloc((tree_out->num_leaf_nodes + 1) * sizeof(int));
    tree_out->fold_node_indices = (int*)malloc((tree_out->num_fold_nodes + 1) * sizeof(int));

    int di = 0, si = 0, li2 = 0, fi = 0;
    for (int i = 0; i < N; i++) {
        switch (tb.nodes[i].type) {
            case SS_NODE_DECISION: tree_out->decision_node_indices[di++] = i; break;
            case SS_NODE_SHOWDOWN: tree_out->showdown_node_indices[si++] = i; break;
            case SS_NODE_LEAF:     tree_out->leaf_node_indices[li2++] = i; break;
            case SS_NODE_FOLD:     tree_out->fold_node_indices[fi++] = i; break;
        }
    }

    printf("[SS] Tree: %d nodes (%d decision, %d fold, %d showdown, %d leaf), "
           "%d players, depth=%d\n",
           N, tree_out->num_decision_nodes, tree_out->num_fold_nodes,
           tree_out->num_showdown_nodes, tree_out->num_leaf_nodes,
           num_players, tree_out->max_depth);

    return 0;
}

extern "C" SS_EXPORT int ss_solve_gpu(
    SSTreeData *td, int max_iterations, SSOutput *output
) {
    int N = td->num_nodes;
    int NP = td->num_players;
    int max_h = 0;
    for (int p = 0; p < NP; p++)
        if (td->num_hands[p] > max_h) max_h = td->num_hands[p];
    if (max_h == 0 || NP == 0) return -1;

    printf("[SS] GPU solve: %d nodes, %d players, max_hands=%d, %d iterations\n",
           N, NP, max_h, max_iterations);

    /* ── Device memory ─────────────────────────── */
    SSNode *d_nodes;
    int *d_children, *d_hands, *d_num_hands;
    float *d_regrets, *d_strategy, *d_strategy_sum, *d_cfv;
    float *d_reach_all;  /* [NP][N * max_h] — one reach array per player */
    float *d_weights;
    float *d_leaf_values = NULL;
    uint32_t *d_strengths = NULL;

    size_t node_sz = N * sizeof(SSNode);
    size_t child_sz = td->num_children_total * sizeof(int);
    size_t state_sz = (size_t)N * SS_MAX_ACTIONS * max_h * sizeof(float);
    size_t cfv_sz = (size_t)N * max_h * sizeof(float);
    size_t reach_sz = (size_t)NP * N * max_h * sizeof(float);
    size_t hands_sz = SS_MAX_PLAYERS * SS_MAX_HANDS * 2 * sizeof(int);
    size_t weights_sz = SS_MAX_PLAYERS * SS_MAX_HANDS * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_nodes, node_sz));
    CUDA_CHECK(cudaMalloc(&d_children, child_sz));
    CUDA_CHECK(cudaMalloc(&d_hands, hands_sz));
    CUDA_CHECK(cudaMalloc(&d_num_hands, SS_MAX_PLAYERS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_weights, weights_sz));
    CUDA_CHECK(cudaMalloc(&d_regrets, state_sz));
    CUDA_CHECK(cudaMalloc(&d_strategy, state_sz));
    CUDA_CHECK(cudaMalloc(&d_strategy_sum, state_sz));
    CUDA_CHECK(cudaMalloc(&d_cfv, cfv_sz));
    CUDA_CHECK(cudaMalloc(&d_reach_all, reach_sz));

    /* Flatten hands for GPU */
    int *h_hands = (int*)calloc(SS_MAX_PLAYERS * SS_MAX_HANDS * 2, sizeof(int));
    float *h_weights = (float*)calloc(SS_MAX_PLAYERS * SS_MAX_HANDS, sizeof(float));
    int h_num_hands[SS_MAX_PLAYERS] = {0};
    for (int p = 0; p < NP; p++) {
        h_num_hands[p] = td->num_hands[p];
        for (int h = 0; h < td->num_hands[p]; h++) {
            h_hands[(p * SS_MAX_HANDS + h) * 2] = td->hands[p][h][0];
            h_hands[(p * SS_MAX_HANDS + h) * 2 + 1] = td->hands[p][h][1];
            h_weights[p * SS_MAX_HANDS + h] = td->weights[p][h];
        }
    }

    CUDA_CHECK(cudaMemcpy(d_nodes, td->nodes, node_sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_children, td->children, child_sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_hands, h_hands, hands_sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_num_hands, h_num_hands, SS_MAX_PLAYERS * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weights, h_weights, weights_sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_regrets, 0, state_sz));
    CUDA_CHECK(cudaMemset(d_strategy, 0, state_sz));
    CUDA_CHECK(cudaMemset(d_strategy_sum, 0, state_sz));

    /* ── Upload leaf values ──────────────── */
    if (td->num_leaf_nodes > 0 && td->leaf_values != NULL) {
        size_t leaf_sz = (size_t)td->num_leaves * NP * max_h * sizeof(float);
        CUDA_CHECK(cudaMalloc(&d_leaf_values, leaf_sz));
        CUDA_CHECK(cudaMemcpy(d_leaf_values, td->leaf_values, leaf_sz, cudaMemcpyHostToDevice));
    }

    /* ── Showdown precompute (river only) ─── */
    int *d_showdown_idx = NULL;
    int *d_sd_local = NULL;
    if (td->is_river && td->num_showdown_nodes > 0) {
        size_t strength_sz = (size_t)td->num_showdown_nodes * SS_MAX_PLAYERS * SS_MAX_HANDS * sizeof(uint32_t);
        CUDA_CHECK(cudaMalloc(&d_strengths, strength_sz));
        CUDA_CHECK(cudaMalloc(&d_showdown_idx, td->num_showdown_nodes * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_showdown_idx, td->showdown_node_indices,
                               td->num_showdown_nodes * sizeof(int), cudaMemcpyHostToDevice));

        ss_precompute_strengths<<<td->num_showdown_nodes, SS_MAX_HANDS>>>(
            d_nodes, d_showdown_idx, td->num_showdown_nodes,
            d_hands, d_num_hands, NP, d_strengths);
        CUDA_CHECK(cudaDeviceSynchronize());

        int *h_sd_local = (int*)calloc(N, sizeof(int));
        for (int i = 0; i < td->num_showdown_nodes; i++)
            h_sd_local[td->showdown_node_indices[i]] = i;
        CUDA_CHECK(cudaMalloc(&d_sd_local, N * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_sd_local, h_sd_local, N * sizeof(int), cudaMemcpyHostToDevice));
        free(h_sd_local);
    }

    /* ── Build level groups ──────────────── */
    int **level_nodes = (int**)calloc(td->max_depth + 1, sizeof(int*));
    int *level_counts = (int*)calloc(td->max_depth + 1, sizeof(int));
    int *level_caps = (int*)calloc(td->max_depth + 1, sizeof(int));
    for (int i = 0; i < N; i++) {
        int d = td->node_depth[i];
        if (level_counts[d] >= level_caps[d]) {
            level_caps[d] = level_caps[d] ? level_caps[d] * 2 : 64;
            level_nodes[d] = (int*)realloc(level_nodes[d], level_caps[d] * sizeof(int));
        }
        level_nodes[d][level_counts[d]++] = i;
    }
    int **d_level_nodes = (int**)malloc((td->max_depth + 1) * sizeof(int*));
    for (int d = 0; d <= td->max_depth; d++) {
        if (level_counts[d] > 0) {
            CUDA_CHECK(cudaMalloc(&d_level_nodes[d], level_counts[d] * sizeof(int)));
            CUDA_CHECK(cudaMemcpy(d_level_nodes[d], level_nodes[d],
                                   level_counts[d] * sizeof(int), cudaMemcpyHostToDevice));
        } else {
            d_level_nodes[d] = NULL;
        }
    }

    /* ── Upload terminal node lists ──────── */
    int *d_fold_nodes = NULL, *d_sd_nodes_list = NULL, *d_leaf_nodes_list = NULL;
    if (td->num_fold_nodes > 0) {
        CUDA_CHECK(cudaMalloc(&d_fold_nodes, td->num_fold_nodes * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_fold_nodes, td->fold_node_indices,
                               td->num_fold_nodes * sizeof(int), cudaMemcpyHostToDevice));
    }
    if (td->num_showdown_nodes > 0) {
        CUDA_CHECK(cudaMalloc(&d_sd_nodes_list, td->num_showdown_nodes * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_sd_nodes_list, td->showdown_node_indices,
                               td->num_showdown_nodes * sizeof(int), cudaMemcpyHostToDevice));
    }
    if (td->num_leaf_nodes > 0) {
        CUDA_CHECK(cudaMalloc(&d_leaf_nodes_list, td->num_leaf_nodes * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_leaf_nodes_list, td->leaf_node_indices,
                               td->num_leaf_nodes * sizeof(int), cudaMemcpyHostToDevice));
    }

    int *d_dec_nodes;
    CUDA_CHECK(cudaMalloc(&d_dec_nodes, td->num_decision_nodes * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_dec_nodes, td->decision_node_indices,
                           td->num_decision_nodes * sizeof(int), cudaMemcpyHostToDevice));

    int *d_iter;
    CUDA_CHECK(cudaMalloc(&d_iter, sizeof(int)));

    int starting_pot = td->starting_pot;
    int block_size = 128;
    if (max_h > 128) block_size = ((max_h + 31) / 32) * 32;

    printf("[SS] %d levels, %d folds, %d showdowns, %d leaves, block=%d\n",
           td->max_depth + 1, td->num_fold_nodes, td->num_showdown_nodes,
           td->num_leaf_nodes, block_size);
    fflush(stdout);

    /* ═══════════════════════════════════════════
     * Main CFR loop — NO CUDA graph for N-player
     * (graph capture doesn't support variable traverser)
     *
     * Each iteration: cycle through all N traversers.
     * For each traverser:
     *   1. Regret matching at all decision nodes
     *   2. Propagate reach for all NON-traverser players
     *   3. Compute terminal values (fold, showdown, leaf)
     *   4. Propagate CFV bottom-up + update traverser's regrets
     * ═══════════════════════════════════════════ */

    for (int iter = 1; iter <= max_iterations; iter++) {
        CUDA_CHECK(cudaMemcpy(d_iter, &iter, sizeof(int), cudaMemcpyHostToDevice));

        for (int trav = 0; trav < NP; trav++) {
            /* 1. Regret matching */
            ss_batch_regret_match<<<td->num_decision_nodes, block_size>>>(
                d_nodes, d_dec_nodes, td->num_decision_nodes,
                d_regrets, d_strategy, max_h, d_num_hands, NP, d_iter);

            /* 2. Propagate reach for each non-traverser player */
            for (int rp = 0; rp < NP; rp++) {
                if (rp == trav) continue;

                /* Initialize this player's reach at root */
                float *d_rp_reach = d_reach_all + (size_t)rp * N * max_h;
                CUDA_CHECK(cudaMemset(d_rp_reach, 0, N * max_h * sizeof(float)));
                CUDA_CHECK(cudaMemcpy(d_rp_reach,
                                       h_weights + rp * SS_MAX_HANDS,
                                       td->num_hands[rp] * sizeof(float),
                                       cudaMemcpyHostToDevice));

                /* Top-down propagation */
                for (int d = 0; d <= td->max_depth; d++) {
                    if (level_counts[d] > 0) {
                        ss_batch_propagate_reach<<<level_counts[d], block_size>>>(
                            d_nodes, d_children, d_level_nodes[d], level_counts[d],
                            d_rp_reach, d_strategy, rp, max_h,
                            d_num_hands, NP);
                    }
                }
            }

            /* 3. Terminal values */
            CUDA_CHECK(cudaMemset(d_cfv, 0, cfv_sz));

            if (td->num_fold_nodes > 0) {
                ss_batch_fold_value<<<td->num_fold_nodes, block_size>>>(
                    d_nodes, d_fold_nodes, td->num_fold_nodes,
                    d_cfv, d_reach_all, trav, max_h,
                    d_num_hands, NP, d_hands, starting_pot, N);
            }
            if (td->num_showdown_nodes > 0 && d_strengths != NULL) {
                ss_batch_showdown_value<<<td->num_showdown_nodes, block_size>>>(
                    d_nodes, d_sd_nodes_list, td->num_showdown_nodes,
                    d_cfv, d_reach_all, trav, max_h,
                    d_num_hands, NP, d_hands, d_strengths, d_sd_local, N);
            }
            if (td->num_leaf_nodes > 0 && d_leaf_values != NULL) {
                ss_batch_leaf_value<<<td->num_leaf_nodes, block_size>>>(
                    d_nodes, d_leaf_nodes_list, td->num_leaf_nodes,
                    d_cfv, d_reach_all, trav, max_h,
                    d_num_hands, NP, d_leaf_values, d_hands, N);
            }

            /* 4. Bottom-up CFV propagation + regret update */
            for (int d = td->max_depth; d >= 0; d--) {
                if (level_counts[d] > 0) {
                    ss_batch_propagate_cfv<<<level_counts[d], block_size>>>(
                        d_nodes, d_children, d_level_nodes[d], level_counts[d],
                        d_cfv, d_regrets, d_strategy_sum, d_strategy,
                        trav, max_h, d_num_hands, NP, d_iter);
                }
            }
        }

        CUDA_CHECK(cudaDeviceSynchronize());

        if (iter % 50 == 0 || iter == max_iterations || iter == 1) {
            printf("[SS] iter %d/%d\n", iter, max_iterations);
            fflush(stdout);
        }
    }

    /* ═══════════════════════════════════════════
     * Extract output
     * ═══════════════════════════════════════════ */

    float *h_strategy = (float*)malloc(state_sz);
    float *h_strategy_sum = (float*)malloc(state_sz);
    float *h_cfv = (float*)malloc(cfv_sz);
    CUDA_CHECK(cudaMemcpy(h_strategy, d_strategy, state_sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_strategy_sum, d_strategy_sum, state_sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_cfv, d_cfv, cfv_sz, cudaMemcpyDeviceToHost));

    /* Root strategy (final iteration) */
    int root_na = td->nodes[0].num_children;
    int root_player = td->nodes[0].player;
    int root_nh = td->num_hands[root_player];
    output->root_num_actions = root_na;
    output->root_player = root_player;
    output->root_strategy = (float*)malloc(root_na * root_nh * sizeof(float));
    for (int a = 0; a < root_na; a++)
        for (int h = 0; h < root_nh; h++)
            output->root_strategy[a * root_nh + h] =
                h_strategy[0 * SS_MAX_ACTIONS * max_h + a * max_h + h];

    /* Root EV for all players */
    output->root_ev = (float*)calloc(NP * max_h, sizeof(float));
    for (int h = 0; h < max_h; h++)
        output->root_ev[h] = h_cfv[0 * max_h + h];

    output->num_players = NP;
    output->max_hands = max_h;

    /* Weighted average strategies at all decision nodes */
    output->num_avg_nodes = td->num_decision_nodes;
    output->avg_strategy_node_ids = (int*)malloc(td->num_decision_nodes * sizeof(int));
    output->avg_strategies = (float*)calloc(
        (size_t)td->num_decision_nodes * SS_MAX_ACTIONS * max_h, sizeof(float));

    for (int di = 0; di < td->num_decision_nodes; di++) {
        int nidx = td->decision_node_indices[di];
        output->avg_strategy_node_ids[di] = nidx;
        int player = td->nodes[nidx].player;
        if (player < 0 || player >= NP) continue;
        int nh = td->num_hands[player];
        int na = td->nodes[nidx].num_children;

        for (int h = 0; h < nh; h++) {
            float sum = 0;
            for (int a = 0; a < na; a++) {
                float v = h_strategy_sum[nidx * SS_MAX_ACTIONS * max_h + a * max_h + h];
                if (v < 0) v = 0;
                sum += v;
            }
            float inv = (sum > 0) ? (1.0f / sum) : (1.0f / na);
            for (int a = 0; a < na; a++) {
                float v = h_strategy_sum[nidx * SS_MAX_ACTIONS * max_h + a * max_h + h];
                if (v < 0) v = 0;
                output->avg_strategies[di * SS_MAX_ACTIONS * max_h + a * max_h + h] =
                    (sum > 0) ? (v * inv) : inv;
            }
        }
    }

    /* ── Cleanup ──────────────────────────── */
    free(h_strategy); free(h_strategy_sum); free(h_cfv);
    free(h_hands); free(h_weights);

    for (int d = 0; d <= td->max_depth; d++) {
        if (d_level_nodes[d]) cudaFree(d_level_nodes[d]);
        if (level_nodes[d]) free(level_nodes[d]);
    }
    free(d_level_nodes); free(level_nodes); free(level_counts); free(level_caps);

    cudaFree(d_nodes); cudaFree(d_children); cudaFree(d_hands);
    cudaFree(d_num_hands); cudaFree(d_weights);
    cudaFree(d_regrets); cudaFree(d_strategy); cudaFree(d_strategy_sum);
    cudaFree(d_cfv); cudaFree(d_reach_all);
    cudaFree(d_dec_nodes); cudaFree(d_iter);
    if (d_leaf_values) cudaFree(d_leaf_values);
    if (d_strengths) cudaFree(d_strengths);
    if (d_showdown_idx) cudaFree(d_showdown_idx);
    if (d_sd_local) cudaFree(d_sd_local);
    if (d_fold_nodes) cudaFree(d_fold_nodes);
    if (d_sd_nodes_list) cudaFree(d_sd_nodes_list);
    if (d_leaf_nodes_list) cudaFree(d_leaf_nodes_list);

    return 0;
}

extern "C" SS_EXPORT void ss_free_tree(SSTreeData *td) {
    if (td->nodes) free(td->nodes);
    if (td->children) free(td->children);
    if (td->level_order) free(td->level_order);
    if (td->node_depth) free(td->node_depth);
    if (td->decision_node_indices) free(td->decision_node_indices);
    if (td->showdown_node_indices) free(td->showdown_node_indices);
    if (td->leaf_node_indices) free(td->leaf_node_indices);
    if (td->fold_node_indices) free(td->fold_node_indices);
    if (td->leaf_values) free(td->leaf_values);
    memset(td, 0, sizeof(SSTreeData));
}

extern "C" SS_EXPORT void ss_free_output(SSOutput *out) {
    if (out->root_strategy) free(out->root_strategy);
    if (out->root_ev) free(out->root_ev);
    if (out->avg_strategies) free(out->avg_strategies);
    if (out->avg_strategy_node_ids) free(out->avg_strategy_node_ids);
    memset(out, 0, sizeof(SSOutput));
}
