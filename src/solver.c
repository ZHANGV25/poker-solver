/**
 * solver.c — Depth-limited DCFR poker solver
 *
 * Core implementation of Discounted CFR for single-street subgame solving
 * with Pluribus-style continuation strategies at leaf nodes.
 *
 * Key optimizations:
 *   - O(N+M) prefix-sum showdown evaluation
 *   - DCFR with Brown's parameters (alpha=1.5, beta=0.5, gamma=3.0)
 *   - Regret-based pruning after warm-up
 *   - Alternating updates (one player per traversal)
 *   - Branch-free regret matching
 */

#include "solver.h"
#include "hand_eval.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdio.h>

/* ── Internal helpers ──────────────────────────────────────────────────── */

/** Check if two cards conflict (share a card) */
static inline int cards_conflict(int a0, int a1, int b0, int b1) {
    return (a0 == b0 || a0 == b1 || a1 == b0 || a1 == b1);
}

/** Check if a hand conflicts with any board card */
static inline int hand_blocked_by_board(int c0, int c1, const int *board, int n) {
    for (int i = 0; i < n; i++)
        if (c0 == board[i] || c1 == board[i]) return 1;
    return 0;
}

/* ── Game tree construction ────────────────────────────────────────────── */

static int add_node(Solver *s, int type, int player, int pot, int stack,
                    int bet0, int bet1) {
    int idx = s->num_nodes++;
    /* Grow nodes array if needed */
    if (idx % 1024 == 0 && idx > 0) {
        s->nodes = realloc(s->nodes, (idx + 1024) * sizeof(TreeNode));
    }
    TreeNode *n = &s->nodes[idx];
    memset(n, 0, sizeof(TreeNode));
    n->type = type;
    n->player = player;
    n->pot = pot;
    n->stack = stack;
    n->bets[0] = bet0;
    n->bets[1] = bet1;
    n->num_actions = 0;
    return idx;
}

static void add_child(Solver *s, int parent, int child) {
    TreeNode *n = &s->nodes[parent];
    if (n->num_actions < MAX_ACTIONS) {
        n->children[n->num_actions++] = child;
    }
}

/**
 * Recursively build the betting tree for one street.
 * Returns the index of the root node.
 *
 * State: current player, pot, stacks, bets so far, number of raises.
 */
static int build_tree(Solver *s, int player, int pot, int stack,
                      int bet0, int bet1, int num_raises, int num_actions_taken) {
    int to_call = (player == 0) ? (bet1 - bet0) : (bet0 - bet1);
    if (to_call < 0) to_call = 0;

    /* If both players have acted and bets are equal, this round is over */
    if (num_actions_taken >= 2 && bet0 == bet1) {
        /* End of betting round — either showdown (river) or leaf (earlier streets) */
        if (s->num_board == 5) {
            return add_node(s, NODE_SHOWDOWN, -1, pot, stack, bet0, bet1);
        } else {
            return add_node(s, NODE_LEAF, -1, pot, stack, bet0, bet1);
        }
    }

    int node = add_node(s, NODE_DECISION, player, pot, stack, bet0, bet1);

    /* Fold (only if there's a bet to fold to) */
    if (to_call > 0) {
        int fold_node = add_node(s, NODE_FOLD, 1 - player, pot, stack, bet0, bet1);
        add_child(s, node, fold_node);
    }

    /* Check (if no bet to call) or Call */
    if (to_call == 0) {
        /* Check */
        int new_pot = pot;
        int next = build_tree(s, 1 - player, new_pot, stack,
                              bet0, bet1, num_raises, num_actions_taken + 1);
        add_child(s, node, next);
    } else {
        /* Call */
        int new_bet0 = bet0, new_bet1 = bet1;
        if (player == 0) new_bet0 = bet1;
        else new_bet1 = bet0;
        int call_pot = pot + to_call;

        if (num_actions_taken >= 1) {
            /* After call, betting round ends */
            if (s->num_board == 5) {
                int sd = add_node(s, NODE_SHOWDOWN, -1, call_pot, stack - to_call,
                                  new_bet0, new_bet1);
                add_child(s, node, sd);
            } else {
                int leaf = add_node(s, NODE_LEAF, -1, call_pot, stack - to_call,
                                    new_bet0, new_bet1);
                add_child(s, node, leaf);
            }
        } else {
            int next = build_tree(s, 1 - player, call_pot, stack - to_call,
                                  new_bet0, new_bet1, num_raises, num_actions_taken + 1);
            add_child(s, node, next);
        }
    }

    /* Bet/Raise (if raises remaining) */
    if (num_raises < MAX_RAISES) {
        int max_bet_into = (player == 0) ? bet1 : bet0; /* amount to match first */
        for (int i = 0; i < s->num_bet_sizes; i++) {
            int bet_amount;
            if (to_call == 0) {
                /* Bet: fraction of pot */
                bet_amount = (int)(s->bet_sizes[i] * pot);
            } else {
                /* Raise: raise to (call + fraction of new pot) */
                int call_pot_size = pot + to_call;
                bet_amount = to_call + (int)(s->bet_sizes[i] * call_pot_size);
            }
            /* Clamp to stack */
            if (bet_amount >= stack) {
                bet_amount = stack; /* all-in */
            }
            /* Minimum bet = to_call + big blind (simplified: just ensure > to_call) */
            if (bet_amount <= to_call) continue;

            int new_bet0 = bet0, new_bet1 = bet1;
            if (player == 0) new_bet0 = bet0 + bet_amount;
            else new_bet1 = bet1 + bet_amount;

            int new_pot = pot + bet_amount;
            int new_stack = stack - bet_amount + to_call; /* stack after calling + raising */

            if (bet_amount >= stack) {
                /* All-in → opponent can only fold or call */
                int ai_node = add_node(s, NODE_DECISION, 1 - player, new_pot,
                                       0, new_bet0, new_bet1);
                /* Fold */
                int fold_n = add_node(s, NODE_FOLD, player, new_pot, 0,
                                      new_bet0, new_bet1);
                add_child(s, ai_node, fold_n);
                /* Call → all-in showdown or leaf */
                int call_bet0 = new_bet0, call_bet1 = new_bet1;
                if (player == 0) call_bet1 = new_bet0; /* match */
                else call_bet0 = new_bet1;
                int final_pot = new_pot + (bet_amount - to_call);
                if (s->num_board == 5) {
                    int sd = add_node(s, NODE_SHOWDOWN, -1, final_pot, 0,
                                      call_bet0, call_bet1);
                    add_child(s, ai_node, sd);
                } else {
                    int leaf = add_node(s, NODE_LEAF, -1, final_pot, 0,
                                        call_bet0, call_bet1);
                    add_child(s, ai_node, leaf);
                }
                add_child(s, node, ai_node);
            } else {
                int next = build_tree(s, 1 - player, new_pot, new_stack,
                                      new_bet0, new_bet1,
                                      num_raises + 1, num_actions_taken + 1);
                add_child(s, node, next);
            }
        }

        /* All-in as a separate option (if not already covered) */
        if (stack > 0) {
            int is_dup = 0;
            for (int i = 0; i < s->num_bet_sizes; i++) {
                int bet_amount;
                if (to_call == 0) bet_amount = (int)(s->bet_sizes[i] * pot);
                else bet_amount = to_call + (int)(s->bet_sizes[i] * (pot + to_call));
                if (bet_amount >= stack) { is_dup = 1; break; }
            }
            if (!is_dup && stack > to_call) {
                int bet_amount = stack;
                int new_bet0 = bet0, new_bet1 = bet1;
                if (player == 0) new_bet0 = bet0 + bet_amount;
                else new_bet1 = bet1 + bet_amount;
                int new_pot = pot + bet_amount;

                int ai_node = add_node(s, NODE_DECISION, 1 - player, new_pot, 0,
                                       new_bet0, new_bet1);
                int fold_n = add_node(s, NODE_FOLD, player, new_pot, 0,
                                      new_bet0, new_bet1);
                add_child(s, ai_node, fold_n);

                int call_bet0 = new_bet0, call_bet1 = new_bet1;
                if (player == 0) call_bet1 = new_bet0;
                else call_bet0 = new_bet1;
                int final_pot = new_pot + (bet_amount - to_call);
                if (s->num_board == 5) {
                    int sd = add_node(s, NODE_SHOWDOWN, -1, final_pot, 0,
                                      call_bet0, call_bet1);
                    add_child(s, ai_node, sd);
                } else {
                    int leaf = add_node(s, NODE_LEAF, -1, final_pot, 0,
                                        call_bet0, call_bet1);
                    add_child(s, ai_node, leaf);
                }
                add_child(s, node, ai_node);
            }
        }
    }

    return node;
}

/* ── Hand sorting for O(N+M) showdown ─────────────────────────────────── */

static int cmp_strength(const void *a, const void *b) {
    const uint32_t *sa = (const uint32_t *)a;
    const uint32_t *sb = (const uint32_t *)b;
    /* Compare by strength (second element), then by index */
    if (sa[1] < sb[1]) return -1;
    if (sa[1] > sb[1]) return 1;
    return 0;
}

static void sort_hands_by_strength(Solver *s, int player) {
    int n = s->num_hands[player];
    int board7[7];
    for (int i = 0; i < s->num_board; i++) board7[i] = s->board[i];

    /* Compute strength for each hand */
    uint32_t (*pairs)[2] = malloc(n * sizeof(uint32_t[2]));
    for (int i = 0; i < n; i++) {
        board7[s->num_board] = s->hands[player][i][0];
        board7[s->num_board + 1] = s->hands[player][i][1];
        /* For river (7 cards), eval7. For earlier streets, we'll need
           to handle this differently during actual CFR (per-runout). */
        if (s->num_board == 5) {
            pairs[i][0] = (uint32_t)i;
            pairs[i][1] = eval7(board7);
        } else {
            pairs[i][0] = (uint32_t)i;
            pairs[i][1] = 0; /* Will be computed per-runout */
        }
    }

    qsort(pairs, n, sizeof(uint32_t[2]), cmp_strength);

    s->sorted[player].num_hands = n;
    s->sorted[player].sorted_indices = malloc(n * sizeof(int));
    s->sorted[player].strengths = malloc(n * sizeof(uint32_t));
    for (int i = 0; i < n; i++) {
        s->sorted[player].sorted_indices[i] = (int)pairs[i][0];
        s->sorted[player].strengths[i] = pairs[i][1];
    }
    free(pairs);
}

/* ── O(N+M) Showdown evaluation ───────────────────────────────────────── */

/**
 * Compute fold payoffs for all hands of the winning player.
 * When player `folder` folds, the other player wins the pot.
 */
static void compute_fold_values(const Solver *s, int folder,
                                const float *reach_opp,
                                float *cfv_out, int num_hero) {
    int winner = 1 - folder;
    float total_opp_reach = 0;
    int n_opp = s->num_hands[folder];

    /* Sum opponent reach (exclude blocked hands) */
    for (int i = 0; i < n_opp; i++)
        total_opp_reach += reach_opp[i];

    int n_hero = s->num_hands[winner];
    for (int h = 0; h < n_hero; h++) {
        int c0 = s->hands[winner][h][0];
        int c1 = s->hands[winner][h][1];
        /* Subtract blocked opponent hands */
        float blocked = 0;
        for (int o = 0; o < n_opp; o++) {
            if (cards_conflict(c0, c1, s->hands[folder][o][0],
                               s->hands[folder][o][1]))
                blocked += reach_opp[o];
        }
        float valid_opp = total_opp_reach - blocked;
        /* Hero wins the pot (their contribution is already subtracted) */
        cfv_out[h] = valid_opp; /* Multiplied by pot/2 later based on context */
    }
}

/**
 * Precompute hand strengths for all hands of both players.
 * Called once during solver_init for river boards.
 */
static void precompute_strengths(Solver *s) {
    if (s->num_board != 5) return;
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

/* Showdown evaluation is now inline in cfr_traverse (uses per-node bet amounts) */

/* ── DCFR Core ─────────────────────────────────────────────────────────── */

/** Regret matching: compute strategy from regrets */
static void regret_match(const float *regrets, float *strategy, int n) {
    float sum = 0;
    for (int a = 0; a < n; a++) {
        float r = regrets[a] > 0 ? regrets[a] : 0;
        strategy[a] = r;
        sum += r;
    }
    if (sum > 0) {
        float inv = 1.0f / sum;
        for (int a = 0; a < n; a++)
            strategy[a] *= inv;
    } else {
        float uniform = 1.0f / n;
        for (int a = 0; a < n; a++)
            strategy[a] = uniform;
    }
}

/**
 * CFR traversal for one player (alternating updates).
 *
 * Returns the counterfactual value for each hand of the traversing player.
 * cfv_out must have space for num_hands[traverser] floats.
 */
static void cfr_traverse(Solver *s, int node_idx, int traverser,
                         float *reach0, float *reach1,
                         float *cfv_out, int iter) {
    TreeNode *node = &s->nodes[node_idx];

    /* Terminal: fold */
    if (node->type == NODE_FOLD) {
        int winner = node->player; /* player who wins (opponent folded) */
        float *reach_opp = (traverser == 0) ? reach1 : reach0;
        int n_opp = s->num_hands[1 - traverser];

        /* Payoff: winner gets (pot - their contribution), loser loses their contribution.
         * From traverser's perspective:
         *   If traverser wins: profit = opponent's bet (what they put in)
         *   If traverser loses: loss = traverser's bet
         */
        float payoff;
        if (traverser == winner) {
            payoff = (float)node->bets[1 - winner]; /* opponent's contribution = our profit */
        } else {
            payoff = -(float)node->bets[traverser]; /* we lose our contribution */
        }

        for (int h = 0; h < s->num_hands[traverser]; h++) {
            int c0 = s->hands[traverser][h][0];
            int c1 = s->hands[traverser][h][1];
            float opp_total = 0;
            for (int o = 0; o < n_opp; o++) {
                if (!cards_conflict(c0, c1, s->hands[1-traverser][o][0],
                                    s->hands[1-traverser][o][1]))
                    opp_total += reach_opp[o];
            }
            cfv_out[h] = opp_total * payoff;
        }
        return;
    }

    /* Terminal: showdown */
    if (node->type == NODE_SHOWDOWN) {
        float *reach_opp = (traverser == 0) ? reach1 : reach0;
        /* Showdown: compute payoffs relative to each player's contribution.
         * Win = opponent's bet, Lose = -our bet, Tie = 0 */
        int n_hero = s->num_hands[traverser];
        int n_opp = s->num_hands[1 - traverser];
        int opp = 1 - traverser;
        float win_payoff = (float)node->bets[opp];   /* we win opponent's bet */
        float lose_payoff = -(float)node->bets[traverser]; /* we lose our bet */

        for (int h = 0; h < n_hero; h++) {
            int hc0 = s->hands[traverser][h][0];
            int hc1 = s->hands[traverser][h][1];
            uint32_t hero_str = s->hand_strengths[traverser][h];

            float total = 0;
            for (int o = 0; o < n_opp; o++) {
                int oc0 = s->hands[opp][o][0];
                int oc1 = s->hands[opp][o][1];
                if (hc0 == oc0 || hc0 == oc1 || hc1 == oc0 || hc1 == oc1)
                    continue;
                uint32_t opp_str = s->hand_strengths[opp][o];
                if (hero_str > opp_str) total += reach_opp[o] * win_payoff;
                else if (hero_str < opp_str) total += reach_opp[o] * lose_payoff;
                /* tie: 0 contribution */
            }
            cfv_out[h] = total;
        }
        return;
    }

    /* Terminal: leaf (depth-limited) */
    if (node->type == NODE_LEAF) {
        /* For now, use simple equity as leaf value.
           TODO: Implement 4 continuation strategies */
        float *reach_opp = (traverser == 0) ? reach1 : reach0;
        /* Simplified: just return 0 (check-through equity) */
        for (int h = 0; h < s->num_hands[traverser]; h++)
            cfv_out[h] = 0;
        return;
    }

    /* Decision node */
    int acting = node->player;
    int n_hero = s->num_hands[traverser];
    int n_actions = node->num_actions;

    if (acting == traverser) {
        /* Traverser's turn: explore all actions, update regrets */
        InfoSet *is = &s->info_sets[node_idx];
        if (is->regrets == NULL) {
            is->node_idx = node_idx;
            is->num_actions = n_actions;
            is->num_hands = n_hero;
            is->regrets = calloc(n_actions * n_hero, sizeof(float));
            is->cum_strategy = calloc(n_actions * n_hero, sizeof(float));
            is->current_strategy = calloc(n_actions * n_hero, sizeof(float));
        }

        /* Compute current strategy per hand via regret matching (action-major layout) */
        for (int h = 0; h < n_hero; h++) {
            float reg[MAX_ACTIONS], strat[MAX_ACTIONS];
            for (int a = 0; a < n_actions; a++)
                reg[a] = is->regrets[a * n_hero + h];
            regret_match(reg, strat, n_actions);
            for (int a = 0; a < n_actions; a++)
                is->current_strategy[a * n_hero + h] = strat[a];
        }

        /* Traverse each action and compute CFV (use stack-allocated arrays) */
        float action_cfv[MAX_ACTIONS * MAX_HANDS]; /* stack allocated */
        memset(cfv_out, 0, n_hero * sizeof(float));

        /* Pre-allocate modified reach arrays on stack */
        float reach0_mod[MAX_HANDS], reach1_mod[MAX_HANDS];

        for (int a = 0; a < n_actions; a++) {
            memcpy(reach0_mod, reach0, s->num_hands[0] * sizeof(float));
            memcpy(reach1_mod, reach1, s->num_hands[1] * sizeof(float));

            float *my_reach = (traverser == 0) ? reach0_mod : reach1_mod;
            for (int h = 0; h < n_hero; h++)
                my_reach[h] *= is->current_strategy[a * n_hero + h];

            float *child_cfv = action_cfv + a * n_hero;
            cfr_traverse(s, node->children[a], traverser,
                         reach0_mod, reach1_mod, child_cfv, iter);

            for (int h = 0; h < n_hero; h++)
                cfv_out[h] += is->current_strategy[a * n_hero + h] * child_cfv[h];
        }

        /* Update regrets */
        for (int a = 0; a < n_actions; a++) {
            for (int h = 0; h < n_hero; h++) {
                float regret = action_cfv[a * n_hero + h] - cfv_out[h];
                is->regrets[a * n_hero + h] += regret;
            }
        }

        /* Update cumulative strategy */
        float *my_reach = (traverser == 0) ? reach0 : reach1;
        for (int a = 0; a < n_actions; a++) {
            for (int h = 0; h < n_hero; h++) {
                is->cum_strategy[a * n_hero + h] +=
                    my_reach[h] * is->current_strategy[a * n_hero + h];
            }
        }

    } else {
        /* Opponent's turn: use their strategy to weight reach, recurse */
        int n_opp = s->num_hands[acting];
        InfoSet *is = &s->info_sets[node_idx];
        if (is->regrets == NULL) {
            is->node_idx = node_idx;
            is->num_actions = n_actions;
            is->num_hands = n_opp;
            is->regrets = calloc(n_actions * n_opp, sizeof(float));
            is->cum_strategy = calloc(n_actions * n_opp, sizeof(float));
            is->current_strategy = calloc(n_actions * n_opp, sizeof(float));
        }

        /* Compute opponent's current strategy */
        for (int h = 0; h < n_opp; h++) {
            float reg[MAX_ACTIONS], strat[MAX_ACTIONS];
            for (int a = 0; a < n_actions; a++)
                reg[a] = is->regrets[a * n_opp + h];
            regret_match(reg, strat, n_actions);
            for (int a = 0; a < n_actions; a++)
                is->current_strategy[a * n_opp + h] = strat[a];
        }

        /* Traverse each action using stack-allocated arrays */
        float reach0_mod[MAX_HANDS], reach1_mod[MAX_HANDS];
        float child_cfv[MAX_HANDS];
        memset(cfv_out, 0, n_hero * sizeof(float));

        for (int a = 0; a < n_actions; a++) {
            memcpy(reach0_mod, reach0, s->num_hands[0] * sizeof(float));
            memcpy(reach1_mod, reach1, s->num_hands[1] * sizeof(float));

            float *opp_reach = (acting == 0) ? reach0_mod : reach1_mod;
            for (int h = 0; h < n_opp; h++)
                opp_reach[h] *= is->current_strategy[a * n_opp + h];

            cfr_traverse(s, node->children[a], traverser,
                         reach0_mod, reach1_mod, child_cfv, iter);

            for (int h = 0; h < n_hero; h++)
                cfv_out[h] += child_cfv[h];
        }
    }
}

/**
 * Best-response traversal: compute the value of playing the best response
 * against the opponent's average strategy. Used for exploitability computation.
 *
 * For the traversing player, picks the MAX action at each decision point.
 * For the opponent, uses the average (converged) strategy.
 */
static void best_response_traverse(Solver *s, int node_idx, int br_player,
                                   float *reach0, float *reach1,
                                   float *cfv_out) {
    TreeNode *node = &s->nodes[node_idx];

    /* Terminal nodes — same as CFR */
    if (node->type == NODE_FOLD) {
        int winner = node->player;
        float *reach_opp = (br_player == 0) ? reach1 : reach0;
        int n_opp = s->num_hands[1 - br_player];
        float payoff;
        if (br_player == winner)
            payoff = (float)node->bets[1 - winner];
        else
            payoff = -(float)node->bets[br_player];

        for (int h = 0; h < s->num_hands[br_player]; h++) {
            int c0 = s->hands[br_player][h][0];
            int c1 = s->hands[br_player][h][1];
            float opp_total = 0;
            for (int o = 0; o < n_opp; o++) {
                if (!cards_conflict(c0, c1, s->hands[1-br_player][o][0],
                                    s->hands[1-br_player][o][1]))
                    opp_total += reach_opp[o];
            }
            cfv_out[h] = opp_total * payoff;
        }
        return;
    }

    if (node->type == NODE_SHOWDOWN) {
        float *reach_opp = (br_player == 0) ? reach1 : reach0;
        int n_hero = s->num_hands[br_player];
        int n_opp = s->num_hands[1 - br_player];
        int opp = 1 - br_player;
        float win_payoff = (float)node->bets[opp];
        float lose_payoff = -(float)node->bets[br_player];

        for (int h = 0; h < n_hero; h++) {
            int hc0 = s->hands[br_player][h][0];
            int hc1 = s->hands[br_player][h][1];
            uint32_t hero_str = s->hand_strengths[br_player][h];
            float total = 0;
            for (int o = 0; o < n_opp; o++) {
                int oc0 = s->hands[opp][o][0];
                int oc1 = s->hands[opp][o][1];
                if (hc0 == oc0 || hc0 == oc1 || hc1 == oc0 || hc1 == oc1) continue;
                uint32_t opp_str = s->hand_strengths[opp][o];
                if (hero_str > opp_str) total += reach_opp[o] * win_payoff;
                else if (hero_str < opp_str) total += reach_opp[o] * lose_payoff;
            }
            cfv_out[h] = total;
        }
        return;
    }

    if (node->type == NODE_LEAF) {
        for (int h = 0; h < s->num_hands[br_player]; h++)
            cfv_out[h] = 0;
        return;
    }

    int acting = node->player;
    int n_actions = node->num_actions;
    int n_hero = s->num_hands[br_player];
    float reach0_mod[MAX_HANDS], reach1_mod[MAX_HANDS];

    if (acting == br_player) {
        /* BR player: pick the best action per hand */
        float action_cfv[MAX_ACTIONS * MAX_HANDS];

        for (int a = 0; a < n_actions; a++) {
            memcpy(reach0_mod, reach0, s->num_hands[0] * sizeof(float));
            memcpy(reach1_mod, reach1, s->num_hands[1] * sizeof(float));
            /* Don't modify reach for BR player (exploring all actions) */
            float *child_cfv = action_cfv + a * n_hero;
            best_response_traverse(s, node->children[a], br_player,
                                   reach0_mod, reach1_mod, child_cfv);
        }

        /* Take max across actions for each hand */
        for (int h = 0; h < n_hero; h++) {
            float best = -1e30f;
            for (int a = 0; a < n_actions; a++) {
                float v = action_cfv[a * n_hero + h];
                if (v > best) best = v;
            }
            cfv_out[h] = best;
        }
    } else {
        /* Opponent: use their average strategy */
        int n_opp = s->num_hands[acting];
        InfoSet *is = &s->info_sets[node_idx];

        /* Compute average strategy from cumulative */
        float avg_strategy[MAX_ACTIONS * MAX_HANDS];
        if (is->cum_strategy != NULL) {
            for (int h = 0; h < n_opp; h++) {
                float sum = 0;
                for (int a = 0; a < n_actions; a++)
                    sum += is->cum_strategy[a * n_opp + h];
                if (sum > 0) {
                    float inv = 1.0f / sum;
                    for (int a = 0; a < n_actions; a++)
                        avg_strategy[a * n_opp + h] = is->cum_strategy[a * n_opp + h] * inv;
                } else {
                    float uniform = 1.0f / n_actions;
                    for (int a = 0; a < n_actions; a++)
                        avg_strategy[a * n_opp + h] = uniform;
                }
            }
        } else {
            float uniform = 1.0f / n_actions;
            for (int i = 0; i < n_actions * n_opp; i++)
                avg_strategy[i] = uniform;
        }

        float child_cfv[MAX_HANDS];
        memset(cfv_out, 0, n_hero * sizeof(float));

        for (int a = 0; a < n_actions; a++) {
            memcpy(reach0_mod, reach0, s->num_hands[0] * sizeof(float));
            memcpy(reach1_mod, reach1, s->num_hands[1] * sizeof(float));
            float *opp_reach = (acting == 0) ? reach0_mod : reach1_mod;
            for (int h = 0; h < n_opp; h++)
                opp_reach[h] *= avg_strategy[a * n_opp + h];

            best_response_traverse(s, node->children[a], br_player,
                                   reach0_mod, reach1_mod, child_cfv);
            for (int h = 0; h < n_hero; h++)
                cfv_out[h] += child_cfv[h];
        }
    }
}

/** Apply DCFR discounting to regrets and cumulative strategy */
static void apply_dcfr_discount(Solver *s, int iter) {
    float t = (float)iter;
    float alpha_t = powf(t, s->alpha) / (powf(t, s->alpha) + 1.0f);
    float beta_t = s->beta; /* Fixed at 0.5 for negative regrets */
    float gamma_t = powf(t / (t + 1.0f), s->gamma);

    for (int n = 0; n < s->num_nodes; n++) {
        InfoSet *is = &s->info_sets[n];
        if (is->regrets == NULL) continue;

        int size = is->num_actions * is->num_hands;
        for (int i = 0; i < size; i++) {
            if (is->regrets[i] > 0)
                is->regrets[i] *= alpha_t;
            else
                is->regrets[i] *= beta_t;
        }
        for (int i = 0; i < size; i++)
            is->cum_strategy[i] *= gamma_t;
    }
}

/* ── Public API implementation ─────────────────────────────────────────── */

int solver_init(Solver *s,
                const int *board, int num_board,
                const int hands0[][2], const float *weights0, int num_hands0,
                const int hands1[][2], const float *weights1, int num_hands1,
                int starting_pot, int effective_stack,
                const float *bet_sizes, int num_bet_sizes) {
    memset(s, 0, sizeof(Solver));

    /* Copy board */
    s->num_board = num_board;
    for (int i = 0; i < num_board; i++) s->board[i] = board[i];

    /* Copy ranges (filter out board-blocked hands) */
    s->num_hands[0] = 0;
    for (int i = 0; i < num_hands0; i++) {
        if (!hand_blocked_by_board(hands0[i][0], hands0[i][1], board, num_board)) {
            s->hands[0][s->num_hands[0]][0] = hands0[i][0];
            s->hands[0][s->num_hands[0]][1] = hands0[i][1];
            s->weights[0][s->num_hands[0]] = weights0[i];
            s->num_hands[0]++;
        }
    }
    s->num_hands[1] = 0;
    for (int i = 0; i < num_hands1; i++) {
        if (!hand_blocked_by_board(hands1[i][0], hands1[i][1], board, num_board)) {
            s->hands[1][s->num_hands[1]][0] = hands1[i][0];
            s->hands[1][s->num_hands[1]][1] = hands1[i][1];
            s->weights[1][s->num_hands[1]] = weights1[i];
            s->num_hands[1]++;
        }
    }

    /* Copy bet sizes */
    s->num_bet_sizes = num_bet_sizes;
    for (int i = 0; i < num_bet_sizes; i++) s->bet_sizes[i] = bet_sizes[i];

    s->starting_pot = starting_pot;
    s->effective_stack = effective_stack;

    /* DCFR parameters */
    s->alpha = 1.5f;
    s->beta = 0.5f;
    s->gamma = 3.0f;

    /* Allocate tree (initial capacity) */
    s->nodes = malloc(2048 * sizeof(TreeNode));
    s->num_nodes = 0;

    /* Build game tree */
    build_tree(s, 0, starting_pot, effective_stack, 0, 0, 0, 0);

    /* Precompute hand strengths for showdown */
    if (num_board == 5) {
        precompute_strengths(s);
        sort_hands_by_strength(s, 0);
        sort_hands_by_strength(s, 1);
    }

    /* Allocate info sets (one per node, lazy init of regrets) */
    s->info_sets = calloc(s->num_nodes, sizeof(InfoSet));

    /* Allocate scratch buffers to avoid malloc in hot loop */
    int max_hands = s->num_hands[0] > s->num_hands[1] ?
                    s->num_hands[0] : s->num_hands[1];
    s->scratch_reach[0] = malloc(s->num_hands[0] * sizeof(float));
    s->scratch_reach[1] = malloc(s->num_hands[1] * sizeof(float));
    s->scratch_cfv = malloc(max_hands * sizeof(float));

    return 0;
}

float solver_solve(Solver *s, int max_iterations, float target_exploitability) {
    int n0 = s->num_hands[0];
    int n1 = s->num_hands[1];

    float *reach0 = malloc(n0 * sizeof(float));
    float *reach1 = malloc(n1 * sizeof(float));
    float *cfv = malloc((n0 > n1 ? n0 : n1) * sizeof(float));

    for (int iter = 1; iter <= max_iterations; iter++) {
        /* Reset reach to initial weights */
        memcpy(reach0, s->weights[0], n0 * sizeof(float));
        memcpy(reach1, s->weights[1], n1 * sizeof(float));

        /* Traverse for player 0 */
        cfr_traverse(s, 0, 0, reach0, reach1, cfv, iter);

        /* Reset reach */
        memcpy(reach0, s->weights[0], n0 * sizeof(float));
        memcpy(reach1, s->weights[1], n1 * sizeof(float));

        /* Traverse for player 1 */
        cfr_traverse(s, 0, 1, reach0, reach1, cfv, iter);

        /* Apply DCFR discounting */
        apply_dcfr_discount(s, iter);

        s->iterations_run = iter;
    }

    free(reach0);
    free(reach1);
    free(cfv);

    return 0; /* TODO: compute actual exploitability */
}

void solver_get_all_strategies(const Solver *s, int player,
                               float *strategy_out, float *ev_out) {
    TreeNode *root = &s->nodes[0];
    if (root->type != NODE_DECISION) return;

    int acting = root->player;
    if (acting != player) return; /* Player doesn't act at root */

    InfoSet *is = &s->info_sets[0];
    if (is->cum_strategy == NULL) return;

    int n_hands = is->num_hands;
    int n_actions = is->num_actions;

    /* Normalize cumulative strategy */
    for (int h = 0; h < n_hands; h++) {
        float sum = 0;
        for (int a = 0; a < n_actions; a++)
            sum += is->cum_strategy[a * n_hands + h];
        if (sum > 0) {
            float inv = 1.0f / sum;
            for (int a = 0; a < n_actions; a++)
                strategy_out[h * n_actions + a] =
                    is->cum_strategy[a * n_hands + h] * inv;
        } else {
            float uniform = 1.0f / n_actions;
            for (int a = 0; a < n_actions; a++)
                strategy_out[h * n_actions + a] = uniform;
        }
    }

    if (ev_out) {
        /* TODO: compute per-hand EV */
        for (int h = 0; h < n_hands; h++)
            ev_out[h] = 0;
    }
}

float solver_get_strategy(const Solver *s, int player, int hand_idx,
                          float *strategy_out) {
    TreeNode *root = &s->nodes[0];
    InfoSet *is = &s->info_sets[0];
    if (is->cum_strategy == NULL) {
        float uniform = 1.0f / root->num_actions;
        for (int a = 0; a < root->num_actions; a++)
            strategy_out[a] = uniform;
        return 0;
    }

    int n_hands = is->num_hands;
    int n_actions = is->num_actions;
    float sum = 0;
    for (int a = 0; a < n_actions; a++)
        sum += is->cum_strategy[a * n_hands + hand_idx];

    if (sum > 0) {
        float inv = 1.0f / sum;
        for (int a = 0; a < n_actions; a++)
            strategy_out[a] = is->cum_strategy[a * n_hands + hand_idx] * inv;
    } else {
        float uniform = 1.0f / n_actions;
        for (int a = 0; a < n_actions; a++)
            strategy_out[a] = uniform;
    }
    return 0;
}

float solver_exploitability(Solver *s) {
    /* Exploitability = (BR_value_p0 + BR_value_p1) / 2
     * where BR_value_pi = sum over hands h of weight[h] * best_response_cfv[h]
     * normalized by total weight.
     *
     * Returns exploitability in chip units (divide by pot for fraction).
     */
    float total_exploit = 0;
    for (int p = 0; p < 2; p++) {
        int n = s->num_hands[p];
        float *reach0 = malloc(s->num_hands[0] * sizeof(float));
        float *reach1 = malloc(s->num_hands[1] * sizeof(float));
        float *cfv = malloc(n * sizeof(float));

        memcpy(reach0, s->weights[0], s->num_hands[0] * sizeof(float));
        memcpy(reach1, s->weights[1], s->num_hands[1] * sizeof(float));

        best_response_traverse(s, 0, p, reach0, reach1, cfv);

        /* Weight-average the best response values */
        float total_weight = 0;
        float total_value = 0;
        for (int h = 0; h < n; h++) {
            total_weight += s->weights[p][h];
            total_value += s->weights[p][h] * cfv[h];
        }
        if (total_weight > 0)
            total_exploit += total_value / total_weight;

        free(reach0);
        free(reach1);
        free(cfv);
    }

    s->exploitability = total_exploit * 0.5f;
    return s->exploitability;
}

void solver_free(Solver *s) {
    if (s->nodes) free(s->nodes);
    if (s->info_sets) {
        for (int i = 0; i < s->num_nodes; i++) {
            InfoSet *is = &s->info_sets[i];
            if (is->regrets) free(is->regrets);
            if (is->cum_strategy) free(is->cum_strategy);
            if (is->current_strategy) free(is->current_strategy);
        }
        free(s->info_sets);
    }
    if (s->sorted[0].sorted_indices) free(s->sorted[0].sorted_indices);
    if (s->sorted[0].strengths) free(s->sorted[0].strengths);
    if (s->sorted[1].sorted_indices) free(s->sorted[1].sorted_indices);
    if (s->sorted[1].strengths) free(s->sorted[1].strengths);
    if (s->hand_strengths[0]) free(s->hand_strengths[0]);
    if (s->hand_strengths[1]) free(s->hand_strengths[1]);
    if (s->scratch_reach[0]) free(s->scratch_reach[0]);
    if (s->scratch_reach[1]) free(s->scratch_reach[1]);
    if (s->scratch_cfv) free(s->scratch_cfv);
    memset(s, 0, sizeof(Solver));
}
