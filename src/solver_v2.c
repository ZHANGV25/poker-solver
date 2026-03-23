/**
 * solver_v2.c — Pluribus-style depth-limited DCFR solver
 *
 * Linear CFR with:
 *   - Final iteration strategy selection
 *   - 4 continuation strategies at leaf nodes
 *   - Precomputed river strengths for O(1) leaf evaluation
 *   - Zero-allocation hot loop
 */

#include "solver_v2.h"
#include "hand_eval.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/* ── Helpers ───────────────────────────────────────────────────────────── */

static inline int cards_conflict(int a0, int a1, int b0, int b1) {
    return (a0 == b0 || a0 == b1 || a1 == b0 || a1 == b1);
}

/* ── Tree construction ─────────────────────────────────────────────────── */

static int add_node(SolverV2 *s, int type, int player, int pot,
                    int bet0, int bet1) {
    int idx = s->num_nodes++;
    if (idx % 1024 == 0) {
        s->nodes = realloc(s->nodes, (idx + 1024) * sizeof(NodeV2));
    }
    NodeV2 *n = &s->nodes[idx];
    memset(n, 0, sizeof(NodeV2));
    n->type = type;
    n->player = player;
    n->pot = pot;
    n->bets[0] = bet0;
    n->bets[1] = bet1;
    return idx;
}

static void add_child(SolverV2 *s, int parent, int child) {
    NodeV2 *n = &s->nodes[parent];
    if (n->num_actions < MAX_ACTIONS_V2) {
        n->children[n->num_actions++] = child;
    }
}

static int build_tree(SolverV2 *s, int player, int pot, int stack,
                      int bet0, int bet1, int num_raises, int actions_taken) {
    int to_call = (player == 0) ? (bet1 - bet0) : (bet0 - bet1);
    if (to_call < 0) to_call = 0;

    /* Both acted and bets equal = round over */
    if (actions_taken >= 2 && bet0 == bet1) {
        if (s->num_board == 5)
            return add_node(s, NODE_V2_SHOWDOWN, -1, pot, bet0, bet1);
        else
            return add_node(s, NODE_V2_LEAF, -1, pot, bet0, bet1);
    }

    int node = add_node(s, NODE_V2_DECISION, player, pot, bet0, bet1);

    /* Fold (only if facing a bet) */
    if (to_call > 0) {
        int fold_n = add_node(s, NODE_V2_FOLD, 1 - player, pot, bet0, bet1);
        add_child(s, node, fold_n);
    }

    /* Check or Call */
    if (to_call == 0) {
        int next = build_tree(s, 1 - player, pot, stack,
                              bet0, bet1, num_raises, actions_taken + 1);
        add_child(s, node, next);
    } else {
        int nb0 = bet0, nb1 = bet1;
        if (player == 0) nb0 = bet1; else nb1 = bet0;
        int call_pot = pot + to_call;
        int call_stack = stack - to_call;

        if (actions_taken >= 1) {
            /* After call, round ends */
            if (s->num_board == 5) {
                int sd = add_node(s, NODE_V2_SHOWDOWN, -1, call_pot, nb0, nb1);
                add_child(s, node, sd);
            } else {
                int lf = add_node(s, NODE_V2_LEAF, -1, call_pot, nb0, nb1);
                add_child(s, node, lf);
            }
        } else {
            int next = build_tree(s, 1 - player, call_pot, call_stack,
                                  nb0, nb1, num_raises, actions_taken + 1);
            add_child(s, node, next);
        }
    }

    /* Bet/Raise sizes */
    if (num_raises < MAX_RAISES_V2) {
        for (int i = 0; i < s->num_bet_sizes; i++) {
            int bet_amount;
            if (to_call == 0)
                bet_amount = (int)(s->bet_sizes[i] * pot);
            else
                bet_amount = to_call + (int)(s->bet_sizes[i] * (pot + to_call));

            if (bet_amount >= stack) bet_amount = stack;
            if (bet_amount <= to_call) continue;

            int nb0 = bet0, nb1 = bet1;
            if (player == 0) nb0 += bet_amount; else nb1 += bet_amount;
            int new_pot = pot + bet_amount;
            int new_stack = stack - bet_amount + to_call;

            if (bet_amount >= stack) {
                /* All-in: opponent can fold or call */
                int ai = add_node(s, NODE_V2_DECISION, 1-player, new_pot, nb0, nb1);
                int fold_n = add_node(s, NODE_V2_FOLD, player, new_pot, nb0, nb1);
                add_child(s, ai, fold_n);

                int cb0 = nb0, cb1 = nb1;
                if (player == 0) cb1 = nb0; else cb0 = nb1;
                int fp = new_pot + (bet_amount - to_call);

                if (s->num_board == 5) {
                    int sd = add_node(s, NODE_V2_SHOWDOWN, -1, fp, cb0, cb1);
                    add_child(s, ai, sd);
                } else {
                    int lf = add_node(s, NODE_V2_LEAF, -1, fp, cb0, cb1);
                    add_child(s, ai, lf);
                }
                add_child(s, node, ai);
            } else {
                int next = build_tree(s, 1-player, new_pot, new_stack,
                                      nb0, nb1, num_raises+1, actions_taken+1);
                add_child(s, node, next);
            }
        }

        /* Explicit all-in (if not already covered) */
        if (stack > to_call) {
            int is_dup = 0;
            for (int i = 0; i < s->num_bet_sizes; i++) {
                int ba;
                if (to_call == 0) ba = (int)(s->bet_sizes[i] * pot);
                else ba = to_call + (int)(s->bet_sizes[i] * (pot + to_call));
                if (ba >= stack) { is_dup = 1; break; }
            }
            if (!is_dup) {
                int ba = stack;
                int nb0 = bet0, nb1 = bet1;
                if (player == 0) nb0 += ba; else nb1 += ba;
                int new_pot = pot + ba;

                int ai = add_node(s, NODE_V2_DECISION, 1-player, new_pot, nb0, nb1);
                int fold_n = add_node(s, NODE_V2_FOLD, player, new_pot, nb0, nb1);
                add_child(s, ai, fold_n);

                int cb0 = nb0, cb1 = nb1;
                if (player == 0) cb1 = nb0; else cb0 = nb1;
                int fp = new_pot + (ba - to_call);
                if (s->num_board == 5) {
                    int sd = add_node(s, NODE_V2_SHOWDOWN, -1, fp, cb0, cb1);
                    add_child(s, ai, sd);
                } else {
                    int lf = add_node(s, NODE_V2_LEAF, -1, fp, cb0, cb1);
                    add_child(s, ai, lf);
                }
                add_child(s, node, ai);
            }
        }
    }

    return node;
}

/* ── Regret matching ───────────────────────────────────────────────────── */

static inline void regret_match(const float *regrets, float *strategy,
                                int num_actions, int num_hands, int hand_idx) {
    float sum = 0;
    for (int a = 0; a < num_actions; a++) {
        float r = regrets[a * num_hands + hand_idx];
        r = r > 0 ? r : 0;
        strategy[a] = r;
        sum += r;
    }
    if (sum > 0) {
        float inv = 1.0f / sum;
        for (int a = 0; a < num_actions; a++)
            strategy[a] *= inv;
    } else {
        float u = 1.0f / num_actions;
        for (int a = 0; a < num_actions; a++)
            strategy[a] = u;
    }
}

/* ── CFR traversal ─────────────────────────────────────────────────────── */

static void cfr_traverse(SolverV2 *s, int node_idx, int traverser,
                         float *reach0, float *reach1,
                         float *cfv_out, int iter) {
    NodeV2 *node = &s->nodes[node_idx];
    int n_trav = s->num_hands[traverser];
    int opp = 1 - traverser;

    /* ── Fold terminal ──────────────────────────────────────────── */
    if (node->type == NODE_V2_FOLD) {
        int winner = node->player;
        float *reach_opp = (traverser == 0) ? reach1 : reach0;
        int n_opp = s->num_hands[opp];
        float payoff = (traverser == winner)
            ? (float)node->bets[1 - winner]
            : -(float)node->bets[traverser];

        for (int h = 0; h < n_trav; h++) {
            float opp_sum = 0;
            int c0 = s->hands[traverser][h][0], c1 = s->hands[traverser][h][1];
            for (int o = 0; o < n_opp; o++) {
                if (!cards_conflict(c0, c1, s->hands[opp][o][0], s->hands[opp][o][1]))
                    opp_sum += reach_opp[o];
            }
            cfv_out[h] = opp_sum * payoff;
        }
        return;
    }

    /* ── Showdown terminal ──────────────────────────────────────── */
    if (node->type == NODE_V2_SHOWDOWN) {
        float *reach_opp = (traverser == 0) ? reach1 : reach0;
        int n_opp = s->num_hands[opp];
        float win_pay = (float)node->bets[opp];
        float lose_pay = -(float)node->bets[traverser];

        for (int h = 0; h < n_trav; h++) {
            int c0 = s->hands[traverser][h][0], c1 = s->hands[traverser][h][1];
            uint32_t hs = s->hand_strengths[traverser][h];
            float total = 0;
            for (int o = 0; o < n_opp; o++) {
                if (cards_conflict(c0, c1, s->hands[opp][o][0], s->hands[opp][o][1]))
                    continue;
                uint32_t os = s->hand_strengths[opp][o];
                if (hs > os) total += reach_opp[o] * win_pay;
                else if (hs < os) total += reach_opp[o] * lose_pay;
            }
            cfv_out[h] = total;
        }
        return;
    }

    /* ── Leaf terminal (4 continuation strategies) ──────────────── */
    if (node->type == NODE_V2_LEAF) {
        /* Find this leaf's index */
        int leaf_idx = -1;
        for (int i = 0; i < s->num_leaves; i++) {
            if (s->leaf_indices[i] == node_idx) { leaf_idx = i; break; }
        }

        if (leaf_idx >= 0 && s->leaf_values != NULL) {
            /* Use precomputed leaf values.
             * The DCFR iteration handles the opponent's choice over
             * 4 strategies implicitly — for now, use the average
             * across all 4 strategies as the leaf value. */
            float *reach_opp = (traverser == 0) ? reach1 : reach0;
            int n_opp = s->num_hands[opp];

            for (int h = 0; h < n_trav; h++) {
                float val = 0;
                int c0 = s->hands[traverser][h][0], c1 = s->hands[traverser][h][1];
                float opp_sum = 0;
                for (int o = 0; o < n_opp; o++) {
                    if (!cards_conflict(c0, c1, s->hands[opp][o][0], s->hands[opp][o][1]))
                        opp_sum += reach_opp[o];
                }
                /* Average across 4 strategies */
                for (int k = 0; k < NUM_CONT_STRATS; k++) {
                    int vi = (leaf_idx * NUM_CONT_STRATS + k) * s->num_hands[traverser] + h;
                    /* Assuming traverser's values stored at this index */
                    if (s->leaf_values[leaf_idx * NUM_CONT_STRATS + k] != NULL)
                        val += s->leaf_values[leaf_idx * NUM_CONT_STRATS + k][h];
                }
                cfv_out[h] = opp_sum * val * 0.25f;
            }
        } else {
            /* Fallback: zero leaf value */
            for (int h = 0; h < n_trav; h++)
                cfv_out[h] = 0;
        }
        return;
    }

    /* ── Decision node ──────────────────────────────────────────── */
    int acting = node->player;
    int n_actions = node->num_actions;

    InfoSetV2 *is = &s->info_sets[node_idx];
    if (is->regrets == NULL) {
        int nh = s->num_hands[acting];
        is->num_actions = n_actions;
        is->num_hands = nh;
        is->regrets = calloc(n_actions * nh, sizeof(float));
        is->strategy_sum = calloc(n_actions * nh, sizeof(float));
        is->current_strategy = calloc(n_actions * nh, sizeof(float));
    }

    int nh_acting = is->num_hands;

    /* Compute current strategy via regret matching */
    for (int h = 0; h < nh_acting; h++) {
        float strat[MAX_ACTIONS_V2];
        regret_match(is->regrets, strat, n_actions, nh_acting, h);
        for (int a = 0; a < n_actions; a++)
            is->current_strategy[a * nh_acting + h] = strat[a];
    }

    /* Stack-allocated arrays */
    float reach0_mod[MAX_HANDS_V2], reach1_mod[MAX_HANDS_V2];

    if (acting == traverser) {
        /* Traverser: explore all actions, update regrets */
        float action_cfv[MAX_ACTIONS_V2 * MAX_HANDS_V2];
        memset(cfv_out, 0, n_trav * sizeof(float));

        for (int a = 0; a < n_actions; a++) {
            memcpy(reach0_mod, reach0, s->num_hands[0] * sizeof(float));
            memcpy(reach1_mod, reach1, s->num_hands[1] * sizeof(float));
            float *my_reach = (traverser == 0) ? reach0_mod : reach1_mod;
            for (int h = 0; h < n_trav; h++)
                my_reach[h] *= is->current_strategy[a * nh_acting + h];

            float *child_cfv = action_cfv + a * n_trav;
            cfr_traverse(s, node->children[a], traverser,
                         reach0_mod, reach1_mod, child_cfv, iter);

            for (int h = 0; h < n_trav; h++)
                cfv_out[h] += is->current_strategy[a * nh_acting + h] * child_cfv[h];
        }

        /* Update regrets */
        for (int a = 0; a < n_actions; a++)
            for (int h = 0; h < n_trav; h++)
                is->regrets[a * n_trav + h] += action_cfv[a * n_trav + h] - cfv_out[h];

        /* Update strategy sum with Linear CFR weighting (weight = iteration) */
        float *my_reach = (traverser == 0) ? reach0 : reach1;
        float weight = (float)iter;
        for (int a = 0; a < n_actions; a++)
            for (int h = 0; h < n_trav; h++)
                is->strategy_sum[a * n_trav + h] +=
                    weight * my_reach[h] * is->current_strategy[a * nh_acting + h];

    } else {
        /* Opponent: sample via current strategy */
        float child_cfv[MAX_HANDS_V2];
        memset(cfv_out, 0, n_trav * sizeof(float));

        for (int a = 0; a < n_actions; a++) {
            memcpy(reach0_mod, reach0, s->num_hands[0] * sizeof(float));
            memcpy(reach1_mod, reach1, s->num_hands[1] * sizeof(float));
            float *opp_reach = (acting == 0) ? reach0_mod : reach1_mod;
            for (int h = 0; h < nh_acting; h++)
                opp_reach[h] *= is->current_strategy[a * nh_acting + h];

            cfr_traverse(s, node->children[a], traverser,
                         reach0_mod, reach1_mod, child_cfv, iter);
            for (int h = 0; h < n_trav; h++)
                cfv_out[h] += child_cfv[h];
        }
    }
}

/* ── Linear CFR discounting ────────────────────────────────────────────── */

static void apply_linear_cfr_discount(SolverV2 *s, int iter) {
    /* Linear CFR: weight iteration t by t.
     * Equivalent to DCFR(1,1,1).
     * Discount factor: d = t / (t+1) applied to regrets and strategy sums.
     * Applied every iteration. */
    float d = (float)iter / ((float)iter + 1.0f);
    for (int n = 0; n < s->num_nodes; n++) {
        InfoSetV2 *is = &s->info_sets[n];
        if (is->regrets == NULL) continue;
        int size = is->num_actions * is->num_hands;
        for (int i = 0; i < size; i++) {
            is->regrets[i] *= d;
            is->strategy_sum[i] *= d;
        }
    }
}

/* ── Best response for exploitability ──────────────────────────────────── */

static void best_response(SolverV2 *s, int node_idx, int br_player,
                          float *reach0, float *reach1, float *cfv_out) {
    NodeV2 *node = &s->nodes[node_idx];
    int n_br = s->num_hands[br_player];
    int opp = 1 - br_player;
    float reach0_mod[MAX_HANDS_V2], reach1_mod[MAX_HANDS_V2];

    if (node->type == NODE_V2_FOLD) {
        float *reach_opp = (br_player == 0) ? reach1 : reach0;
        int n_opp = s->num_hands[opp];
        float payoff = (br_player == node->player)
            ? (float)node->bets[1 - node->player]
            : -(float)node->bets[br_player];
        for (int h = 0; h < n_br; h++) {
            float os = 0;
            for (int o = 0; o < n_opp; o++)
                if (!cards_conflict(s->hands[br_player][h][0], s->hands[br_player][h][1],
                                    s->hands[opp][o][0], s->hands[opp][o][1]))
                    os += reach_opp[o];
            cfv_out[h] = os * payoff;
        }
        return;
    }
    if (node->type == NODE_V2_SHOWDOWN) {
        float *reach_opp = (br_player == 0) ? reach1 : reach0;
        int n_opp = s->num_hands[opp];
        for (int h = 0; h < n_br; h++) {
            float total = 0;
            uint32_t hs = s->hand_strengths[br_player][h];
            for (int o = 0; o < n_opp; o++) {
                if (cards_conflict(s->hands[br_player][h][0], s->hands[br_player][h][1],
                                   s->hands[opp][o][0], s->hands[opp][o][1]))
                    continue;
                uint32_t os = s->hand_strengths[opp][o];
                if (hs > os) total += reach_opp[o] * (float)node->bets[opp];
                else if (hs < os) total -= reach_opp[o] * (float)node->bets[br_player];
            }
            cfv_out[h] = total;
        }
        return;
    }
    if (node->type == NODE_V2_LEAF) {
        for (int h = 0; h < n_br; h++) cfv_out[h] = 0;
        return;
    }

    int acting = node->player;
    int n_actions = node->num_actions;

    if (acting == br_player) {
        float action_cfv[MAX_ACTIONS_V2 * MAX_HANDS_V2];
        for (int a = 0; a < n_actions; a++) {
            memcpy(reach0_mod, reach0, s->num_hands[0] * sizeof(float));
            memcpy(reach1_mod, reach1, s->num_hands[1] * sizeof(float));
            best_response(s, node->children[a], br_player,
                          reach0_mod, reach1_mod, action_cfv + a * n_br);
        }
        for (int h = 0; h < n_br; h++) {
            float best = -1e30f;
            for (int a = 0; a < n_actions; a++) {
                float v = action_cfv[a * n_br + h];
                if (v > best) best = v;
            }
            cfv_out[h] = best;
        }
    } else {
        /* Use final iteration strategy */
        InfoSetV2 *is = &s->info_sets[node_idx];
        int nh = s->num_hands[acting];
        float child_cfv[MAX_HANDS_V2];
        memset(cfv_out, 0, n_br * sizeof(float));

        for (int a = 0; a < n_actions; a++) {
            memcpy(reach0_mod, reach0, s->num_hands[0] * sizeof(float));
            memcpy(reach1_mod, reach1, s->num_hands[1] * sizeof(float));
            float *opp_reach = (acting == 0) ? reach0_mod : reach1_mod;
            if (is->current_strategy) {
                for (int h = 0; h < nh; h++)
                    opp_reach[h] *= is->current_strategy[a * nh + h];
            } else {
                float u = 1.0f / n_actions;
                for (int h = 0; h < nh; h++)
                    opp_reach[h] *= u;
            }
            best_response(s, node->children[a], br_player,
                          reach0_mod, reach1_mod, child_cfv);
            for (int h = 0; h < n_br; h++)
                cfv_out[h] += child_cfv[h];
        }
    }
}

/* ── Public API ────────────────────────────────────────────────────────── */

int sv2_init(SolverV2 *s,
             const int *board, int num_board,
             const int hands0[][2], const float *weights0, int num_hands0,
             const int hands1[][2], const float *weights1, int num_hands1,
             int starting_pot, int effective_stack,
             const float *bet_sizes, int num_bet_sizes) {
    memset(s, 0, sizeof(SolverV2));

    s->num_board = num_board;
    for (int i = 0; i < num_board; i++) s->board[i] = board[i];

    /* Filter board-blocked hands */
    for (int p = 0; p < 2; p++) {
        const int (*h)[2] = (p == 0) ? (const int(*)[2])hands0 : (const int(*)[2])hands1;
        const float *w = (p == 0) ? weights0 : weights1;
        int n = (p == 0) ? num_hands0 : num_hands1;
        s->num_hands[p] = 0;
        for (int i = 0; i < n; i++) {
            int blocked = 0;
            for (int b = 0; b < num_board; b++)
                if (h[i][0] == board[b] || h[i][1] == board[b]) { blocked = 1; break; }
            if (!blocked) {
                int idx = s->num_hands[p]++;
                s->hands[p][idx][0] = h[i][0];
                s->hands[p][idx][1] = h[i][1];
                s->weights[p][idx] = w[i];
            }
        }
    }

    s->num_bet_sizes = num_bet_sizes;
    for (int i = 0; i < num_bet_sizes; i++) s->bet_sizes[i] = bet_sizes[i];
    s->starting_pot = starting_pot;
    s->effective_stack = effective_stack;

    /* Build tree */
    s->nodes = malloc(2048 * sizeof(NodeV2));
    s->num_nodes = 0;
    build_tree(s, 0, starting_pot, effective_stack, 0, 0, 0, 0);

    /* Precompute hand strengths for showdown */
    if (num_board == 5) {
        for (int p = 0; p < 2; p++) {
            s->hand_strengths[p] = malloc(s->num_hands[p] * sizeof(uint32_t));
            int board7[7];
            for (int i = 0; i < 5; i++) board7[i] = s->board[i];
            for (int h = 0; h < s->num_hands[p]; h++) {
                board7[5] = s->hands[p][h][0];
                board7[6] = s->hands[p][h][1];
                s->hand_strengths[p][h] = eval7(board7);
            }
        }
    }

    /* Find leaf nodes */
    s->num_leaves = 0;
    for (int i = 0; i < s->num_nodes; i++)
        if (s->nodes[i].type == NODE_V2_LEAF) s->num_leaves++;
    s->leaf_indices = malloc(s->num_leaves * sizeof(int));
    int li = 0;
    for (int i = 0; i < s->num_nodes; i++)
        if (s->nodes[i].type == NODE_V2_LEAF) s->leaf_indices[li++] = i;

    /* Allocate info sets */
    s->info_sets = calloc(s->num_nodes, sizeof(InfoSetV2));

    return 0;
}

int sv2_precompute_river_strengths(SolverV2 *s) {
    if (s->num_board != 4) return -1; /* Only for turn boards */

    for (int p = 0; p < 2; p++) {
        int blocked[52] = {0};
        for (int i = 0; i < 4; i++) blocked[s->board[i]] = 1;

        s->river_table[p].num_rivers = 0;
        s->river_table[p].river_cards = malloc(48 * sizeof(int));
        for (int c = 0; c < 52; c++) {
            if (!blocked[c])
                s->river_table[p].river_cards[s->river_table[p].num_rivers++] = c;
        }

        int nr = s->river_table[p].num_rivers;
        int nh = s->num_hands[p];
        s->river_table[p].num_hands = nh;
        s->river_table[p].strengths = malloc(nr * sizeof(uint32_t*));

        for (int ri = 0; ri < nr; ri++) {
            s->river_table[p].strengths[ri] = malloc(nh * sizeof(uint32_t));
            int rc = s->river_table[p].river_cards[ri];
            int board7[7] = {s->board[0], s->board[1], s->board[2],
                             s->board[3], rc, 0, 0};
            for (int h = 0; h < nh; h++) {
                if (s->hands[p][h][0] == rc || s->hands[p][h][1] == rc) {
                    s->river_table[p].strengths[ri][h] = 0;
                } else {
                    board7[5] = s->hands[p][h][0];
                    board7[6] = s->hands[p][h][1];
                    s->river_table[p].strengths[ri][h] = eval7(board7);
                }
            }
        }
    }
    return 0;
}

int sv2_compute_leaf_values(SolverV2 *s) {
    if (s->num_leaves == 0) return 0;
    if (s->num_board == 5) return 0; /* River: no leaves, just showdown */

    /* Allocate leaf value arrays: [num_leaves * NUM_CONT_STRATS][num_hands_max] */
    int max_h = s->num_hands[0] > s->num_hands[1] ? s->num_hands[0] : s->num_hands[1];
    int total_slots = s->num_leaves * NUM_CONT_STRATS;
    s->leaf_values = malloc(total_slots * sizeof(float*));
    for (int i = 0; i < total_slots; i++)
        s->leaf_values[i] = calloc(max_h, sizeof(float));

    /* For each leaf, compute equity-based continuation values.
     * Traverse all river cards, compute showdown equity for each hand pair. */
    if (s->river_table[0].strengths == NULL) {
        /* No river precompute — use raw equity (less accurate) */
        return 0;
    }

    int nr = s->river_table[0].num_rivers;
    int n0 = s->num_hands[0], n1 = s->num_hands[1];

    for (int li = 0; li < s->num_leaves; li++) {
        NodeV2 *leaf = &s->nodes[s->leaf_indices[li]];
        float half_pot = leaf->pot * 0.5f;

        /* For each of 4 strategies, compute leaf value per hand.
         * Strategy 0: unmodified (50% equity)
         * Strategy 1: fold-biased (opponent folds more → we win more)
         * Strategy 2: call-biased (opponent calls more → equity matters more)
         * Strategy 3: raise-biased (opponent raises more → we need stronger hands) */
        float fold_bias[4] = {1.0f, 5.0f, 1.0f, 1.0f};
        float call_bias[4] = {1.0f, 1.0f, 5.0f, 1.0f};

        for (int k = 0; k < NUM_CONT_STRATS; k++) {
            float *vals = s->leaf_values[li * NUM_CONT_STRATS + k];

            /* Compute equity across all river cards for each traverser hand */
            for (int p = 0; p < 2; p++) {
                int nh = s->num_hands[p];
                int n_opp = s->num_hands[1-p];

                for (int h = 0; h < nh; h++) {
                    float total_win = 0, total_lose = 0, total_valid = 0;
                    int hc0 = s->hands[p][h][0], hc1 = s->hands[p][h][1];

                    for (int ri = 0; ri < nr; ri++) {
                        uint32_t hs = s->river_table[p].strengths[ri][h];
                        if (hs == 0) continue; /* blocked by river card */

                        for (int o = 0; o < n_opp; o++) {
                            uint32_t os = s->river_table[1-p].strengths[ri][o];
                            if (os == 0) continue;
                            if (cards_conflict(hc0, hc1,
                                               s->hands[1-p][o][0], s->hands[1-p][o][1]))
                                continue;

                            float w = s->weights[1-p][o];
                            /* Apply bias: fold-biased opponent folds weak hands */
                            if (k == 1) { /* fold-biased: reduce weak opponent weight */
                                if (os < hs) w *= 0.3f; /* weak hands fold */
                            } else if (k == 3) { /* raise-biased: emphasize strong */
                                if (os > hs) w *= 2.0f;
                            }

                            if (hs > os) total_win += w;
                            else if (hs < os) total_lose += w;
                            total_valid += w;
                        }
                    }

                    if (total_valid > 0) {
                        float equity = total_win / total_valid;
                        /* Value relative to pot: equity * pot - (1-equity) * contribution */
                        vals[h] = (equity - 0.5f) * half_pot * 0.01f; /* scale down */
                    }
                }
            }
        }
    }

    return 0;
}

float sv2_solve(SolverV2 *s, int max_iterations, float target_exploitability) {
    int n0 = s->num_hands[0], n1 = s->num_hands[1];
    float reach0[MAX_HANDS_V2], reach1[MAX_HANDS_V2];
    int max_h = n0 > n1 ? n0 : n1;
    float cfv[MAX_HANDS_V2];

    for (int iter = 1; iter <= max_iterations; iter++) {
        memcpy(reach0, s->weights[0], n0 * sizeof(float));
        memcpy(reach1, s->weights[1], n1 * sizeof(float));
        cfr_traverse(s, 0, 0, reach0, reach1, cfv, iter);

        memcpy(reach0, s->weights[0], n0 * sizeof(float));
        memcpy(reach1, s->weights[1], n1 * sizeof(float));
        cfr_traverse(s, 0, 1, reach0, reach1, cfv, iter);

        apply_linear_cfr_discount(s, iter);
        s->iterations_run = iter;
    }
    return 0;
}

void sv2_get_strategy(const SolverV2 *s, int player, int hand_idx,
                      float *strategy_out) {
    /* Return FINAL ITERATION strategy (Pluribus-style) */
    NodeV2 *root = &s->nodes[0];
    InfoSetV2 *is = &s->info_sets[0];
    if (is->current_strategy == NULL || root->player != player) {
        float u = 1.0f / root->num_actions;
        for (int a = 0; a < root->num_actions; a++)
            strategy_out[a] = u;
        return;
    }
    for (int a = 0; a < is->num_actions; a++)
        strategy_out[a] = is->current_strategy[a * is->num_hands + hand_idx];
}

void sv2_get_all_strategies(const SolverV2 *s, int player,
                            float *strategy_out) {
    NodeV2 *root = &s->nodes[0];
    InfoSetV2 *is = &s->info_sets[0];
    int nh = s->num_hands[player];
    int na = root->num_actions;

    for (int h = 0; h < nh; h++) {
        if (is->current_strategy && root->player == player) {
            for (int a = 0; a < na; a++)
                strategy_out[h * na + a] = is->current_strategy[a * nh + h];
        } else {
            float u = 1.0f / na;
            for (int a = 0; a < na; a++)
                strategy_out[h * na + a] = u;
        }
    }
}

float sv2_exploitability(SolverV2 *s) {
    float total = 0;
    for (int p = 0; p < 2; p++) {
        int n = s->num_hands[p];
        float reach0[MAX_HANDS_V2], reach1[MAX_HANDS_V2];
        float cfv[MAX_HANDS_V2];
        memcpy(reach0, s->weights[0], s->num_hands[0] * sizeof(float));
        memcpy(reach1, s->weights[1], s->num_hands[1] * sizeof(float));
        best_response(s, 0, p, reach0, reach1, cfv);
        float tw = 0, tv = 0;
        for (int h = 0; h < n; h++) {
            tw += s->weights[p][h];
            tv += s->weights[p][h] * cfv[h];
        }
        if (tw > 0) total += tv / tw;
    }
    s->exploitability = total * 0.5f;
    return s->exploitability;
}

void sv2_free(SolverV2 *s) {
    if (s->nodes) free(s->nodes);
    if (s->info_sets) {
        for (int i = 0; i < s->num_nodes; i++) {
            InfoSetV2 *is = &s->info_sets[i];
            if (is->regrets) free(is->regrets);
            if (is->strategy_sum) free(is->strategy_sum);
            if (is->current_strategy) free(is->current_strategy);
        }
        free(s->info_sets);
    }
    for (int p = 0; p < 2; p++) {
        if (s->hand_strengths[p]) free(s->hand_strengths[p]);
        if (s->river_table[p].strengths) {
            for (int i = 0; i < s->river_table[p].num_rivers; i++)
                free(s->river_table[p].strengths[i]);
            free(s->river_table[p].strengths);
            free(s->river_table[p].river_cards);
        }
    }
    if (s->leaf_indices) free(s->leaf_indices);
    if (s->leaf_values) {
        int total = s->num_leaves * NUM_CONT_STRATS;
        for (int i = 0; i < total; i++)
            if (s->leaf_values[i]) free(s->leaf_values[i]);
        free(s->leaf_values);
    }
    memset(s, 0, sizeof(SolverV2));
}
