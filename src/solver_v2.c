/**
 * solver_v2.c — Pluribus-style multi-street solver
 *
 * Multi-street architecture:
 *   - Each street has its own betting tree
 *   - At CHANCE nodes (end of non-river betting), the solver iterates
 *     over all possible next cards and recurses into the next street
 *   - Info sets for sub-streets persist across iterations via indexed arrays
 *   - Hand strengths computed on-the-fly at showdown for each 5-card board
 *   - Linear CFR: regrets *= t/(t+1), strategy_sum weighted by t
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

static inline int card_in_set(int card, const int *set, int n) {
    for (int i = 0; i < n; i++)
        if (set[i] == card) return 1;
    return 0;
}

/* ── Street tree: node allocation ─────────────────────────────────────── */

static void st_init(StreetTree *st) {
    memset(st, 0, sizeof(StreetTree));
    st->nodes_capacity = 64;
    st->nodes = malloc(st->nodes_capacity * sizeof(NodeV2));
}

static int st_alloc(StreetTree *st) {
    int idx = st->num_nodes++;
    if (idx >= st->nodes_capacity) {
        st->nodes_capacity *= 2;
        st->nodes = realloc(st->nodes, st->nodes_capacity * sizeof(NodeV2));
    }
    memset(&st->nodes[idx], 0, sizeof(NodeV2));
    st->nodes[idx].player = -1;
    return idx;
}

static int st_add(StreetTree *st, int type, int player, int pot, int b0, int b1) {
    int idx = st_alloc(st);
    st->nodes[idx].type = type;
    st->nodes[idx].player = player;
    st->nodes[idx].pot = pot;
    st->nodes[idx].bets[0] = b0;
    st->nodes[idx].bets[1] = b1;
    return idx;
}

static void st_add_child(StreetTree *st, int parent, int child) {
    NodeV2 *n = &st->nodes[parent];
    if (n->num_actions < MAX_ACTIONS_V2)
        n->children[n->num_actions++] = child;
}

static void st_free(StreetTree *st) {
    if (st->nodes) free(st->nodes);
    if (st->info_sets) {
        for (int i = 0; i < st->info_sets_capacity; i++) {
            if (st->info_sets[i].regrets) free(st->info_sets[i].regrets);
            if (st->info_sets[i].strategy_sum) free(st->info_sets[i].strategy_sum);
            if (st->info_sets[i].current_strategy) free(st->info_sets[i].current_strategy);
        }
        free(st->info_sets);
    }
    memset(st, 0, sizeof(StreetTree));
}

/* ── Build a single-street betting tree ───────────────────────────────── */

static int build_street(StreetTree *st, int is_river,
                        int player, int pot, int stack,
                        int bet0, int bet1, int num_raises, int actions_taken,
                        const float *bet_sizes, int num_bet_sizes) {
    int to_call = (player == 0) ? (bet1 - bet0) : (bet0 - bet1);
    if (to_call < 0) to_call = 0;

    /* Round over: both acted, bets equal */
    if (actions_taken >= 2 && bet0 == bet1) {
        if (is_river)
            return st_add(st, NODE_V2_SHOWDOWN, -1, pot, bet0, bet1);
        else
            return st_add(st, NODE_V2_CHANCE, -1, pot, bet0, bet1);
    }

    int node = st_add(st, NODE_V2_DECISION, player, pot, bet0, bet1);

    /* Fold */
    if (to_call > 0) {
        int fold_n = st_add(st, NODE_V2_FOLD, 1 - player, pot, bet0, bet1);
        st_add_child(st, node, fold_n);
    }

    /* Check or Call */
    if (to_call == 0) {
        int next = build_street(st, is_river, 1 - player, pot, stack,
                                bet0, bet1, num_raises, actions_taken + 1,
                                bet_sizes, num_bet_sizes);
        st_add_child(st, node, next);
    } else {
        int nb0 = bet0, nb1 = bet1;
        if (player == 0) nb0 = bet1; else nb1 = bet0;
        int call_pot = pot + to_call;
        int call_stack = stack - to_call;

        if (actions_taken >= 1) {
            if (is_river) {
                int sd = st_add(st, NODE_V2_SHOWDOWN, -1, call_pot, nb0, nb1);
                st_add_child(st, node, sd);
            } else {
                int ch = st_add(st, NODE_V2_CHANCE, -1, call_pot, nb0, nb1);
                st_add_child(st, node, ch);
            }
        } else {
            int next = build_street(st, is_river, 1 - player, call_pot, call_stack,
                                    nb0, nb1, num_raises, actions_taken + 1,
                                    bet_sizes, num_bet_sizes);
            st_add_child(st, node, next);
        }
    }

    /* Bet/Raise */
    if (num_raises < MAX_RAISES_V2) {
        for (int i = 0; i < num_bet_sizes; i++) {
            int bet_amount;
            if (to_call == 0)
                bet_amount = (int)(bet_sizes[i] * pot);
            else
                bet_amount = to_call + (int)(bet_sizes[i] * (pot + to_call));

            if (bet_amount >= stack) bet_amount = stack;
            if (bet_amount <= to_call) continue;

            int nb0 = bet0, nb1 = bet1;
            if (player == 0) nb0 += bet_amount; else nb1 += bet_amount;
            int new_pot = pot + bet_amount;
            int new_stack = stack - bet_amount + to_call;

            if (bet_amount >= stack) {
                /* All-in */
                int ai = st_add(st, NODE_V2_DECISION, 1-player, new_pot, nb0, nb1);
                int fold_n = st_add(st, NODE_V2_FOLD, player, new_pot, nb0, nb1);
                st_add_child(st, ai, fold_n);

                int cb0 = nb0, cb1 = nb1;
                if (player == 0) cb1 = nb0; else cb0 = nb1;
                int fp = new_pot + (bet_amount - to_call);

                if (is_river) {
                    int sd = st_add(st, NODE_V2_SHOWDOWN, -1, fp, cb0, cb1);
                    st_add_child(st, ai, sd);
                } else {
                    int ch = st_add(st, NODE_V2_CHANCE, -1, fp, cb0, cb1);
                    st_add_child(st, ai, ch);
                }
                st_add_child(st, node, ai);
            } else {
                int next = build_street(st, is_river, 1-player, new_pot, new_stack,
                                        nb0, nb1, num_raises+1, actions_taken+1,
                                        bet_sizes, num_bet_sizes);
                st_add_child(st, node, next);
            }
        }

        /* Explicit all-in */
        if (stack > to_call) {
            int is_dup = 0;
            for (int i = 0; i < num_bet_sizes; i++) {
                int ba;
                if (to_call == 0) ba = (int)(bet_sizes[i] * pot);
                else ba = to_call + (int)(bet_sizes[i] * (pot + to_call));
                if (ba >= stack) { is_dup = 1; break; }
            }
            if (!is_dup) {
                int ba = stack;
                int nb0 = bet0, nb1 = bet1;
                if (player == 0) nb0 += ba; else nb1 += ba;
                int new_pot = pot + ba;

                int ai = st_add(st, NODE_V2_DECISION, 1-player, new_pot, nb0, nb1);
                int fold_n = st_add(st, NODE_V2_FOLD, player, new_pot, nb0, nb1);
                st_add_child(st, ai, fold_n);

                int cb0 = nb0, cb1 = nb1;
                if (player == 0) cb1 = nb0; else cb0 = nb1;
                int fp = new_pot + (ba - to_call);
                if (is_river) {
                    int sd = st_add(st, NODE_V2_SHOWDOWN, -1, fp, cb0, cb1);
                    st_add_child(st, ai, sd);
                } else {
                    int ch = st_add(st, NODE_V2_CHANCE, -1, fp, cb0, cb1);
                    st_add_child(st, ai, ch);
                }
                st_add_child(st, node, ai);
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

/* ── Ensure info set exists ───────────────────────────────────────────── */

static InfoSetV2 *ensure_info_set(InfoSetV2 *arr, int cap, int idx,
                                  int num_actions, int num_hands) {
    if (idx >= cap) return NULL; /* caller must size arrays correctly */
    InfoSetV2 *is = &arr[idx];
    if (is->regrets == NULL) {
        is->num_actions = num_actions;
        is->num_hands = num_hands;
        is->regrets = calloc(num_actions * num_hands, sizeof(float));
        is->strategy_sum = calloc(num_actions * num_hands, sizeof(float));
        is->current_strategy = calloc(num_actions * num_hands, sizeof(float));
    }
    return is;
}

/* ── Showdown evaluation ──────────────────────────────────────────────── */

static void eval_showdown_board(const SolverV2 *s, const int *full_board,
                                const NodeV2 *node,
                                int traverser, const float *reach_opp,
                                float *cfv_out) {
    int opp = 1 - traverser;
    int n_trav = s->num_hands[traverser];
    int n_opp = s->num_hands[opp];
    float half_pot = node->pot * 0.5f;

    /* Compute hand strengths for this specific 5-card board */
    /* We inline this to avoid extra allocation */
    for (int h = 0; h < n_trav; h++) {
        int hc0 = s->hands[traverser][h][0], hc1 = s->hands[traverser][h][1];
        /* Check if hand is blocked by board */
        if (card_in_set(hc0, full_board, 5) || card_in_set(hc1, full_board, 5)) {
            cfv_out[h] = 0;
            continue;
        }
        int cards_t[7] = {full_board[0], full_board[1], full_board[2],
                          full_board[3], full_board[4], hc0, hc1};
        uint32_t hs = eval7(cards_t);

        float total = 0;
        for (int o = 0; o < n_opp; o++) {
            int oc0 = s->hands[opp][o][0], oc1 = s->hands[opp][o][1];
            if (cards_conflict(hc0, hc1, oc0, oc1)) continue;
            if (card_in_set(oc0, full_board, 5) || card_in_set(oc1, full_board, 5)) continue;

            int cards_o[7] = {full_board[0], full_board[1], full_board[2],
                              full_board[3], full_board[4], oc0, oc1};
            uint32_t os = eval7(cards_o);
            if (hs > os) total += reach_opp[o] * half_pot;
            else if (hs < os) total -= reach_opp[o] * half_pot;
        }
        cfv_out[h] = total;
    }
}

/* ── CFR traverse for a single street ─────────────────────────────────── */

/* Forward declare — mutual recursion with chance handler */
static void cfr_street(SolverV2 *s, StreetTree *st, InfoSetV2 *is_arr, int is_cap,
                       int node_idx, int traverser,
                       float *reach0, float *reach1, float *cfv_out, int iter,
                       const int *board, int num_board, int street_depth);

static void cfr_chance(SolverV2 *s, const NodeV2 *node, int traverser,
                       float *reach0, float *reach1, float *cfv_out, int iter,
                       const int *board, int num_board, int street_depth);

static void cfr_street(SolverV2 *s, StreetTree *st, InfoSetV2 *is_arr, int is_cap,
                       int node_idx, int traverser,
                       float *reach0, float *reach1, float *cfv_out, int iter,
                       const int *board, int num_board, int street_depth) {
    NodeV2 *node = &st->nodes[node_idx];
    int n_trav = s->num_hands[traverser];
    int opp = 1 - traverser;
    int n0 = s->num_hands[0], n1 = s->num_hands[1];

    /* ── Fold ──────────────────────────────────────────────── */
    if (node->type == NODE_V2_FOLD) {
        float *reach_opp = (traverser == 0) ? reach1 : reach0;
        int n_opp = s->num_hands[opp];
        int winner = node->player;
        float half_start = s->starting_pot * 0.5f;
        int loser = 1 - winner;
        float payoff;
        if (traverser == winner)
            payoff = half_start + (float)node->bets[loser];
        else
            payoff = -(half_start + (float)node->bets[traverser]);

        for (int h = 0; h < n_trav; h++) {
            int c0 = s->hands[traverser][h][0], c1 = s->hands[traverser][h][1];
            if (card_in_set(c0, board, num_board) || card_in_set(c1, board, num_board)) {
                cfv_out[h] = 0; continue;
            }
            float opp_sum = 0;
            for (int o = 0; o < n_opp; o++) {
                if (cards_conflict(c0, c1, s->hands[opp][o][0], s->hands[opp][o][1])) continue;
                if (card_in_set(s->hands[opp][o][0], board, num_board) ||
                    card_in_set(s->hands[opp][o][1], board, num_board)) continue;
                opp_sum += reach_opp[o];
            }
            cfv_out[h] = opp_sum * payoff;
        }
        return;
    }

    /* ── Showdown ──────────────────────────────────────────── */
    if (node->type == NODE_V2_SHOWDOWN) {
        float *reach_opp = (traverser == 0) ? reach1 : reach0;
        eval_showdown_board(s, board, node, traverser, reach_opp, cfv_out);
        return;
    }

    /* ── Chance: deal next card, recurse into next street ──── */
    if (node->type == NODE_V2_CHANCE) {
        cfr_chance(s, node, traverser, reach0, reach1, cfv_out, iter,
                   board, num_board, street_depth);
        return;
    }

    /* ── Decision ──────────────────────────────────────────── */
    int acting = node->player;
    int n_actions = node->num_actions;

    InfoSetV2 *is = ensure_info_set(is_arr, is_cap, node_idx,
                                     n_actions, s->num_hands[acting]);
    /* Write back pointer in case realloc moved it */
    /* (The caller's is_arr/is_cap are passed by pointer, so this is fine) */

    int nh_acting = is->num_hands;

    /* Regret matching */
    for (int h = 0; h < nh_acting; h++) {
        float strat[MAX_ACTIONS_V2];
        regret_match(is->regrets, strat, n_actions, nh_acting, h);
        for (int a = 0; a < n_actions; a++)
            is->current_strategy[a * nh_acting + h] = strat[a];
    }

    float *reach0_mod = malloc(n0 * sizeof(float));
    float *reach1_mod = malloc(n1 * sizeof(float));

    if (acting == traverser) {
        float *action_cfv = malloc(n_actions * n_trav * sizeof(float));
        memset(cfv_out, 0, n_trav * sizeof(float));

        for (int a = 0; a < n_actions; a++) {
            memcpy(reach0_mod, reach0, n0 * sizeof(float));
            memcpy(reach1_mod, reach1, n1 * sizeof(float));
            float *my_reach = (traverser == 0) ? reach0_mod : reach1_mod;
            for (int h = 0; h < n_trav; h++)
                my_reach[h] *= is->current_strategy[a * nh_acting + h];

            cfr_street(s, st, is_arr, is_cap, node->children[a], traverser,
                       reach0_mod, reach1_mod, action_cfv + a * n_trav, iter,
                       board, num_board, street_depth);

            for (int h = 0; h < n_trav; h++)
                cfv_out[h] += is->current_strategy[a * nh_acting + h]
                              * action_cfv[a * n_trav + h];
        }

        /* Update regrets */
        for (int a = 0; a < n_actions; a++)
            for (int h = 0; h < n_trav; h++)
                is->regrets[a * nh_acting + h] += action_cfv[a * n_trav + h] - cfv_out[h];

        /* Iteration-weighted strategy sum */
        float *my_reach = (traverser == 0) ? reach0 : reach1;
        for (int a = 0; a < n_actions; a++)
            for (int h = 0; h < n_trav; h++)
                is->strategy_sum[a * nh_acting + h] +=
                    (float)iter * my_reach[h] * is->current_strategy[a * nh_acting + h];

        free(action_cfv);
    } else {
        float *child_cfv = malloc(n_trav * sizeof(float));
        memset(cfv_out, 0, n_trav * sizeof(float));

        for (int a = 0; a < n_actions; a++) {
            memcpy(reach0_mod, reach0, n0 * sizeof(float));
            memcpy(reach1_mod, reach1, n1 * sizeof(float));
            float *opp_reach = (acting == 0) ? reach0_mod : reach1_mod;
            for (int h = 0; h < nh_acting; h++)
                opp_reach[h] *= is->current_strategy[a * nh_acting + h];

            cfr_street(s, st, is_arr, is_cap, node->children[a], traverser,
                       reach0_mod, reach1_mod, child_cfv, iter,
                       board, num_board, street_depth);
            for (int h = 0; h < n_trav; h++)
                cfv_out[h] += child_cfv[h];
        }
        free(child_cfv);
    }

    free(reach0_mod);
    free(reach1_mod);
}

/* ── Chance node handler: deal next card, solve next street ───────────── */

static void cfr_chance(SolverV2 *s, const NodeV2 *node, int traverser,
                       float *reach0, float *reach1, float *cfv_out, int iter,
                       const int *board, int num_board, int street_depth) {
    int n_trav = s->num_hands[traverser];
    int n0 = s->num_hands[0], n1 = s->num_hands[1];

    /* Which cards are blocked by current board? */
    int blocked[52] = {0};
    for (int i = 0; i < num_board; i++) blocked[board[i]] = 1;

    /* Count valid next cards */
    int next_cards[52];
    int num_next = 0;
    for (int c = 0; c < 52; c++)
        if (!blocked[c]) next_cards[num_next++] = c;

    int next_num_board = num_board + 1;
    int next_is_river = (next_num_board == 5);

    memset(cfv_out, 0, n_trav * sizeof(float));
    float *card_cfv = malloc(n_trav * sizeof(float));

    /* Determine which info set storage to use for the next street */
    int turn_idx_base = -1;
    if (num_board == 3) {
        /* Dealing the turn card — use turn_info_sets */
        turn_idx_base = 0; /* will index by card position in next_cards */
    }

    for (int ci = 0; ci < num_next; ci++) {
        int deal_card = next_cards[ci];

        /* Build next board */
        int next_board[5];
        for (int i = 0; i < num_board; i++) next_board[i] = board[i];
        next_board[num_board] = deal_card;

        /* Remaining stack = effective_stack minus each player's total bet.
         * At end of a betting round with bets equalized: each put in pot/2
         * (the pot includes starting_pot + both players' bets). */
        int each_invested = (node->pot - s->starting_pot) / 2;
        int remaining_stack = s->effective_stack - each_invested;
        if (remaining_stack < 0) remaining_stack = 0;

        /* Build next street's tree */
        StreetTree next_st;
        st_init(&next_st);
        build_street(&next_st, next_is_river, 0, node->pot, remaining_stack,
                     0, 0, 0, 0, s->bet_sizes, s->num_bet_sizes);

        /* Ephemeral info sets for this sub-street.
         * TODO: persist across iterations for faster convergence. */
        int sub_is_cap = next_st.num_nodes + 16;
        InfoSetV2 *sub_is = calloc(sub_is_cap, sizeof(InfoSetV2));

        /* Recurse into next street */
        cfr_street(s, &next_st, sub_is, sub_is_cap,
                   0, traverser, reach0, reach1, card_cfv, iter,
                   next_board, next_num_board,
                   num_board == 3 ? ci : street_depth);

        for (int h = 0; h < n_trav; h++)
            cfv_out[h] += card_cfv[h];

        /* Free everything */
        free(next_st.nodes);
        for (int i = 0; i < sub_is_cap; i++) {
            if (sub_is[i].regrets) free(sub_is[i].regrets);
            if (sub_is[i].strategy_sum) free(sub_is[i].strategy_sum);
            if (sub_is[i].current_strategy) free(sub_is[i].current_strategy);
        }
        free(sub_is);
    }

    /* Average over number of dealt cards */
    if (num_next > 0) {
        float inv = 1.0f / num_next;
        for (int h = 0; h < n_trav; h++)
            cfv_out[h] *= inv;
    }

    free(card_cfv);
}

/* ── Linear CFR discounting ───────────────────────────────────────────── */

static void discount_info_sets(InfoSetV2 *arr, int cap, int iter) {
    float d = (float)iter / ((float)iter + 1.0f);
    for (int n = 0; n < cap; n++) {
        InfoSetV2 *is = &arr[n];
        if (is->regrets == NULL) continue;
        int size = is->num_actions * is->num_hands;
        for (int i = 0; i < size; i++)
            is->regrets[i] *= d;
    }
}

/* ── Best response for exploitability ──────────────────────────────────── */

static void best_response_street(SolverV2 *s, StreetTree *st, InfoSetV2 *is_arr, int is_cap,
                                 int node_idx, int br_player,
                                 float *reach0, float *reach1, float *cfv_out,
                                 const int *board, int num_board, int street_depth);

static void br_chance(SolverV2 *s, const NodeV2 *node, int br_player,
                      float *reach0, float *reach1, float *cfv_out,
                      const int *board, int num_board, int street_depth);

static void best_response_street(SolverV2 *s, StreetTree *st, InfoSetV2 *is_arr, int is_cap,
                                 int node_idx, int br_player,
                                 float *reach0, float *reach1, float *cfv_out,
                                 const int *board, int num_board, int street_depth) {
    NodeV2 *node = &st->nodes[node_idx];
    int n_br = s->num_hands[br_player];
    int opp = 1 - br_player;
    int n0 = s->num_hands[0], n1 = s->num_hands[1];

    if (node->type == NODE_V2_FOLD) {
        float *reach_opp = (br_player == 0) ? reach1 : reach0;
        int n_opp = s->num_hands[opp];
        int winner = node->player;
        int loser = 1 - winner;
        float half_start = s->starting_pot * 0.5f;
        float payoff = (br_player == winner)
            ? (half_start + (float)node->bets[loser])
            : -(half_start + (float)node->bets[br_player]);
        for (int h = 0; h < n_br; h++) {
            int c0 = s->hands[br_player][h][0], c1 = s->hands[br_player][h][1];
            if (card_in_set(c0, board, num_board) || card_in_set(c1, board, num_board)) {
                cfv_out[h] = 0; continue;
            }
            float os = 0;
            for (int o = 0; o < n_opp; o++) {
                if (cards_conflict(c0, c1, s->hands[opp][o][0], s->hands[opp][o][1])) continue;
                if (card_in_set(s->hands[opp][o][0], board, num_board) ||
                    card_in_set(s->hands[opp][o][1], board, num_board)) continue;
                os += reach_opp[o];
            }
            cfv_out[h] = os * payoff;
        }
        return;
    }

    if (node->type == NODE_V2_SHOWDOWN) {
        float *reach_opp = (br_player == 0) ? reach1 : reach0;
        eval_showdown_board(s, board, node, br_player, reach_opp, cfv_out);
        return;
    }

    if (node->type == NODE_V2_CHANCE) {
        br_chance(s, node, br_player, reach0, reach1, cfv_out,
                  board, num_board, street_depth);
        return;
    }

    int acting = node->player;
    int n_actions = node->num_actions;
    float *reach0_mod = malloc(n0 * sizeof(float));
    float *reach1_mod = malloc(n1 * sizeof(float));

    if (acting == br_player) {
        float *action_cfv = malloc(n_actions * n_br * sizeof(float));
        for (int a = 0; a < n_actions; a++) {
            memcpy(reach0_mod, reach0, n0 * sizeof(float));
            memcpy(reach1_mod, reach1, n1 * sizeof(float));
            best_response_street(s, st, is_arr, is_cap, node->children[a], br_player,
                                 reach0_mod, reach1_mod, action_cfv + a * n_br,
                                 board, num_board, street_depth);
        }
        for (int h = 0; h < n_br; h++) {
            float best = -1e30f;
            for (int a = 0; a < n_actions; a++) {
                float v = action_cfv[a * n_br + h];
                if (v > best) best = v;
            }
            cfv_out[h] = best;
        }
        free(action_cfv);
    } else {
        InfoSetV2 *is = (node_idx < is_cap) ? &is_arr[node_idx] : NULL;
        int nh = s->num_hands[acting];
        float *child_cfv = malloc(n_br * sizeof(float));
        memset(cfv_out, 0, n_br * sizeof(float));

        for (int a = 0; a < n_actions; a++) {
            memcpy(reach0_mod, reach0, n0 * sizeof(float));
            memcpy(reach1_mod, reach1, n1 * sizeof(float));
            float *opp_reach = (acting == 0) ? reach0_mod : reach1_mod;
            if (is && is->current_strategy) {
                for (int h = 0; h < nh; h++)
                    opp_reach[h] *= is->current_strategy[a * nh + h];
            } else {
                float u = 1.0f / n_actions;
                for (int h = 0; h < nh; h++) opp_reach[h] *= u;
            }
            best_response_street(s, st, is_arr, is_cap, node->children[a], br_player,
                                 reach0_mod, reach1_mod, child_cfv,
                                 board, num_board, street_depth);
            for (int h = 0; h < n_br; h++)
                cfv_out[h] += child_cfv[h];
        }
        free(child_cfv);
    }
    free(reach0_mod);
    free(reach1_mod);
}

static void br_chance(SolverV2 *s, const NodeV2 *node, int br_player,
                      float *reach0, float *reach1, float *cfv_out,
                      const int *board, int num_board, int street_depth) {
    int n_br = s->num_hands[br_player];
    int blocked[52] = {0};
    for (int i = 0; i < num_board; i++) blocked[board[i]] = 1;

    int next_cards[52];
    int num_next = 0;
    for (int c = 0; c < 52; c++)
        if (!blocked[c]) next_cards[num_next++] = c;

    int next_num_board = num_board + 1;
    int next_is_river = (next_num_board == 5);

    memset(cfv_out, 0, n_br * sizeof(float));
    float *card_cfv = malloc(n_br * sizeof(float));

    for (int ci = 0; ci < num_next; ci++) {
        int next_board[5];
        for (int i = 0; i < num_board; i++) next_board[i] = board[i];
        next_board[num_board] = next_cards[ci];

        StreetTree next_st;
        st_init(&next_st);
        build_street(&next_st, next_is_river, 0, node->pot,
                     s->effective_stack - (node->pot - s->starting_pot) / 2,
                     0, 0, 0, 0, s->bet_sizes, s->num_bet_sizes);

        next_st.info_sets_capacity = next_st.num_nodes + 16;
        next_st.info_sets = calloc(next_st.info_sets_capacity, sizeof(InfoSetV2));

        best_response_street(s, &next_st, next_st.info_sets, next_st.info_sets_capacity,
                             0, br_player, reach0, reach1, card_cfv,
                             next_board, next_num_board,
                             num_board == 3 ? ci : street_depth);

        for (int h = 0; h < n_br; h++)
            cfv_out[h] += card_cfv[h];

        st_free(&next_st);
    }

    if (num_next > 0) {
        float inv = 1.0f / num_next;
        for (int h = 0; h < n_br; h++)
            cfv_out[h] *= inv;
    }
    free(card_cfv);
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

    /* Build root street tree */
    int is_river = (num_board == 5);
    st_init(&s->root_tree);
    build_street(&s->root_tree, is_river, 0, starting_pot, effective_stack,
                 0, 0, 0, 0, bet_sizes, num_bet_sizes);

    s->root_tree.info_sets_capacity = s->root_tree.num_nodes + 16;
    s->root_tree.info_sets = calloc(s->root_tree.info_sets_capacity, sizeof(InfoSetV2));

    /* Pre-allocate persistent info set storage for sub-streets */
    if (num_board < 5) {
        /* Count possible next cards */
        int blocked[52] = {0};
        for (int i = 0; i < num_board; i++) blocked[s->board[i]] = 1;
        s->num_turn_cards = 0;
        s->turn_cards = malloc(48 * sizeof(int));
        for (int c = 0; c < 52; c++)
            if (!blocked[c]) s->turn_cards[s->num_turn_cards++] = c;

        /* Use generous fixed sizes for sub-street info sets.
         * The tree shape depends on pot/stack which varies per chance path,
         * so we use a safe upper bound rather than sampling. */
        s->max_turn_nodes = 128;    /* typical street tree: 20-60 nodes */
        s->max_river_nodes = 128;
        s->max_river_per_turn = 48;

        if (num_board == 3) {
            /* Flop: need turn + river info sets */
            s->turn_info_sets = malloc(s->num_turn_cards * sizeof(InfoSetV2*));
            for (int i = 0; i < s->num_turn_cards; i++)
                s->turn_info_sets[i] = calloc(s->max_turn_nodes, sizeof(InfoSetV2));

            int total_river_slots = s->num_turn_cards * s->max_river_per_turn;
            s->river_info_sets = malloc(total_river_slots * sizeof(InfoSetV2*));
            for (int i = 0; i < total_river_slots; i++)
                s->river_info_sets[i] = calloc(s->max_river_nodes, sizeof(InfoSetV2));

        } else if (num_board == 4) {
            /* Turn: need river info sets only */
            s->river_info_sets = malloc(s->num_turn_cards * sizeof(InfoSetV2*));
            for (int i = 0; i < s->num_turn_cards; i++)
                s->river_info_sets[i] = calloc(s->max_river_nodes, sizeof(InfoSetV2));
        }
    }

    return 0;
}

float sv2_solve(SolverV2 *s, int max_iterations, float target_exploitability) {
    int n0 = s->num_hands[0], n1 = s->num_hands[1];
    int max_h = n0 > n1 ? n0 : n1;
    float *reach0 = malloc(n0 * sizeof(float));
    float *reach1 = malloc(n1 * sizeof(float));
    float *cfv = malloc(max_h * sizeof(float));

    for (int iter = 1; iter <= max_iterations; iter++) {
        /* Traverse for player 0 */
        memcpy(reach0, s->weights[0], n0 * sizeof(float));
        memcpy(reach1, s->weights[1], n1 * sizeof(float));
        cfr_street(s, &s->root_tree, s->root_tree.info_sets, s->root_tree.info_sets_capacity,
                   0, 0, reach0, reach1, cfv, iter,
                   s->board, s->num_board, 0);

        /* Traverse for player 1 */
        memcpy(reach0, s->weights[0], n0 * sizeof(float));
        memcpy(reach1, s->weights[1], n1 * sizeof(float));
        cfr_street(s, &s->root_tree, s->root_tree.info_sets, s->root_tree.info_sets_capacity,
                   0, 1, reach0, reach1, cfv, iter,
                   s->board, s->num_board, 0);

        /* Linear CFR: discount regrets at root street */
        discount_info_sets(s->root_tree.info_sets, s->root_tree.info_sets_capacity, iter);

        /* Also discount sub-street info sets */
        if (s->turn_info_sets) {
            for (int i = 0; i < s->num_turn_cards; i++)
                discount_info_sets(s->turn_info_sets[i], s->max_turn_nodes, iter);
        }
        if (s->river_info_sets) {
            int total;
            if (s->num_board == 3)
                total = s->num_turn_cards * s->max_river_per_turn;
            else
                total = s->num_turn_cards;
            for (int i = 0; i < total; i++)
                discount_info_sets(s->river_info_sets[i], s->max_river_nodes, iter);
        }

        s->iterations_run = iter;
    }
    free(reach0);
    free(reach1);
    free(cfv);
    return 0;
}

void sv2_get_strategy(const SolverV2 *s, int player, int hand_idx,
                      float *strategy_out) {
    NodeV2 *root = &s->root_tree.nodes[0];
    InfoSetV2 *is = &s->root_tree.info_sets[0];
    if (is->current_strategy == NULL || root->player != player) {
        float u = 1.0f / root->num_actions;
        for (int a = 0; a < root->num_actions; a++)
            strategy_out[a] = u;
        return;
    }
    for (int a = 0; a < is->num_actions; a++)
        strategy_out[a] = is->current_strategy[a * is->num_hands + hand_idx];
}

void sv2_get_strategy_at_node(const SolverV2 *s,
                              const int *action_seq, int seq_len,
                              int player, int hand_idx,
                              float *strategy_out, int *num_actions_out) {
    int node_idx = 0;
    for (int i = 0; i < seq_len; i++) {
        NodeV2 *n = &s->root_tree.nodes[node_idx];
        if (n->type != NODE_V2_DECISION) break;
        int a = action_seq[i];
        if (a < 0 || a >= n->num_actions) break;
        node_idx = n->children[a];
    }

    NodeV2 *node = &s->root_tree.nodes[node_idx];
    if (node->type != NODE_V2_DECISION || node->player != player) {
        if (num_actions_out) *num_actions_out = 0;
        return;
    }

    InfoSetV2 *is = (node_idx < s->root_tree.info_sets_capacity)
                     ? &s->root_tree.info_sets[node_idx] : NULL;
    if (num_actions_out) *num_actions_out = node->num_actions;

    if (is && is->current_strategy) {
        for (int a = 0; a < is->num_actions; a++)
            strategy_out[a] = is->current_strategy[a * is->num_hands + hand_idx];
    } else {
        float u = 1.0f / node->num_actions;
        for (int a = 0; a < node->num_actions; a++)
            strategy_out[a] = u;
    }
}

void sv2_get_all_strategies(const SolverV2 *s, int player,
                            float *strategy_out) {
    NodeV2 *root = &s->root_tree.nodes[0];
    InfoSetV2 *is = &s->root_tree.info_sets[0];
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

void sv2_get_average_strategy(const SolverV2 *s, int player, int hand_idx,
                              float *strategy_out) {
    NodeV2 *root = &s->root_tree.nodes[0];
    InfoSetV2 *is = &s->root_tree.info_sets[0];

    if (is->strategy_sum == NULL || root->player != player) {
        float u = 1.0f / root->num_actions;
        for (int a = 0; a < root->num_actions; a++)
            strategy_out[a] = u;
        return;
    }

    float sum = 0;
    for (int a = 0; a < is->num_actions; a++) {
        float v = is->strategy_sum[a * is->num_hands + hand_idx];
        v = v > 0 ? v : 0;
        strategy_out[a] = v;
        sum += v;
    }
    if (sum > 0) {
        float inv = 1.0f / sum;
        for (int a = 0; a < is->num_actions; a++)
            strategy_out[a] *= inv;
    } else {
        float u = 1.0f / is->num_actions;
        for (int a = 0; a < is->num_actions; a++)
            strategy_out[a] = u;
    }
}

float sv2_exploitability(SolverV2 *s) {
    float total = 0;
    int n0 = s->num_hands[0], n1 = s->num_hands[1];
    int max_h = n0 > n1 ? n0 : n1;
    float *reach0 = malloc(n0 * sizeof(float));
    float *reach1 = malloc(n1 * sizeof(float));
    float *cfv = malloc(max_h * sizeof(float));

    for (int p = 0; p < 2; p++) {
        int n = s->num_hands[p];
        memcpy(reach0, s->weights[0], n0 * sizeof(float));
        memcpy(reach1, s->weights[1], n1 * sizeof(float));
        best_response_street(s, &s->root_tree,
                             s->root_tree.info_sets, s->root_tree.info_sets_capacity,
                             0, p, reach0, reach1, cfv,
                             s->board, s->num_board, 0);
        float tw = 0, tv = 0;
        for (int h = 0; h < n; h++) {
            tw += s->weights[p][h];
            tv += s->weights[p][h] * cfv[h];
        }
        if (tw > 0) total += tv / tw;
    }
    free(reach0);
    free(reach1);
    free(cfv);
    s->exploitability = total * 0.5f;
    return s->exploitability;
}

static void free_info_set_arr(InfoSetV2 *arr, int cap) {
    if (!arr) return;
    for (int i = 0; i < cap; i++) {
        if (arr[i].regrets) free(arr[i].regrets);
        if (arr[i].strategy_sum) free(arr[i].strategy_sum);
        if (arr[i].current_strategy) free(arr[i].current_strategy);
    }
    free(arr);
}

void sv2_free(SolverV2 *s) {
    st_free(&s->root_tree);

    if (s->turn_info_sets) {
        for (int i = 0; i < s->num_turn_cards; i++)
            free_info_set_arr(s->turn_info_sets[i], s->max_turn_nodes);
        free(s->turn_info_sets);
    }
    if (s->river_info_sets) {
        int total;
        if (s->num_board == 3)
            total = s->num_turn_cards * s->max_river_per_turn;
        else
            total = s->num_turn_cards;
        for (int i = 0; i < total; i++)
            free_info_set_arr(s->river_info_sets[i], s->max_river_nodes);
        free(s->river_info_sets);
    }
    if (s->turn_cards) free(s->turn_cards);
    if (s->cached_strengths[0]) free(s->cached_strengths[0]);
    if (s->cached_strengths[1]) free(s->cached_strengths[1]);

    memset(s, 0, sizeof(SolverV2));
}
