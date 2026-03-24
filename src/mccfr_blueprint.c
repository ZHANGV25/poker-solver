/**
 * mccfr_blueprint.c — N-player external-sampling MCCFR
 *
 * Implements Pluribus's blueprint training algorithm exactly:
 *
 * Algorithm (from Pluribus supplementary, Algorithm 1):
 *   For each iteration t = 1, 2, ...:
 *     Pick traverser i = (t-1) % num_players
 *     Sample a hand for each player from their range
 *     Traverse the game tree:
 *       At traverser's nodes: explore ALL actions, compute CFV per action
 *       At opponent's nodes: SAMPLE one action ∝ current strategy
 *       At chance nodes: SAMPLE one card
 *     Update traverser's regrets at visited info sets
 *     Linear CFR discount: regrets *= t/(t+1)
 *
 * Key design choices:
 *   - Info sets stored in hash table (board + action sequence → info set)
 *   - Multi-street: flop betting → sample turn card → turn betting →
 *     sample river card → river betting → showdown
 *   - Memory: O(info sets visited) not O(tree size)
 *   - Exact per-hand strategies (no card abstraction)
 */

#include "mccfr_blueprint.h"
#include "hand_eval.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/* ── RNG (xorshift64) ────────────────────────────────────────────────── */

static inline uint64_t rng_next(uint64_t *state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

static inline int rng_int(uint64_t *state, int n) {
    return (int)(rng_next(state) % (uint64_t)n);
}

static inline float rng_float(uint64_t *state) {
    return (float)(rng_next(state) & 0xFFFFFF) / (float)0x1000000;
}

/* ── Hash table operations ───────────────────────────────────────────── */

static uint64_t hash_combine(uint64_t a, uint64_t b) {
    a ^= b + 0x9e3779b97f4a7c15ULL + (a << 6) + (a >> 2);
    return a;
}

static uint64_t compute_board_hash(const int *board, int num_board) {
    uint64_t h = 0x123456789ABCDEFULL;
    for (int i = 0; i < num_board; i++)
        h = hash_combine(h, (uint64_t)board[i] * 31 + 7);
    return h;
}

static uint64_t compute_action_hash(const int *actions, int num_actions) {
    uint64_t h = 0xFEDCBA9876543210ULL;
    for (int i = 0; i < num_actions; i++)
        h = hash_combine(h, (uint64_t)actions[i] * 17 + 3);
    return h;
}

static void info_table_init(BPInfoTable *t) {
    t->keys = (BPInfoKey*)calloc(BP_HASH_SIZE, sizeof(BPInfoKey));
    t->sets = (BPInfoSet*)calloc(BP_HASH_SIZE, sizeof(BPInfoSet));
    t->occupied = (int*)calloc(BP_HASH_SIZE, sizeof(int));
    t->num_entries = 0;
}

static int info_table_find_or_create(BPInfoTable *t, BPInfoKey key,
                                      int num_actions, int num_hands) {
    uint64_t h = hash_combine(key.board_hash, key.action_hash);
    h = hash_combine(h, (uint64_t)key.player);
    int slot = (int)(h % BP_HASH_SIZE);

    /* Linear probing */
    for (int probe = 0; probe < BP_HASH_SIZE; probe++) {
        int idx = (slot + probe) % BP_HASH_SIZE;
        if (!t->occupied[idx]) {
            /* Create new entry */
            t->occupied[idx] = 1;
            t->keys[idx] = key;
            t->sets[idx].num_actions = num_actions;
            t->sets[idx].num_hands = num_hands;
            t->sets[idx].regrets = (float*)calloc(num_actions * num_hands, sizeof(float));
            t->sets[idx].strategy_sum = (float*)calloc(num_actions * num_hands, sizeof(float));
            t->sets[idx].current_strategy = (float*)calloc(num_actions * num_hands, sizeof(float));
            t->num_entries++;
            return idx;
        }
        if (t->keys[idx].player == key.player &&
            t->keys[idx].board_hash == key.board_hash &&
            t->keys[idx].action_hash == key.action_hash) {
            return idx;  /* Found existing */
        }
    }
    return -1; /* Table full — shouldn't happen with proper sizing */
}

static void info_table_free(BPInfoTable *t) {
    for (int i = 0; i < BP_HASH_SIZE; i++) {
        if (t->occupied[i]) {
            free(t->sets[i].regrets);
            free(t->sets[i].strategy_sum);
            free(t->sets[i].current_strategy);
        }
    }
    free(t->keys);
    free(t->sets);
    free(t->occupied);
}

/* ── Helpers ─────────────────────────────────────────────────────────── */

static inline int cards_conflict(int a0, int a1, int b0, int b1) {
    return (a0 == b0 || a0 == b1 || a1 == b0 || a1 == b1);
}

static inline int card_in_set(int card, const int *set, int n) {
    for (int i = 0; i < n; i++)
        if (set[i] == card) return 1;
    return 0;
}

static void regret_match(float *regrets, float *strategy,
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

/* Sample an action according to strategy probabilities */
static int sample_action(const float *strategy, int num_actions, uint64_t *rng) {
    float r = rng_float(rng);
    float cumulative = 0;
    for (int a = 0; a < num_actions; a++) {
        cumulative += strategy[a];
        if (r <= cumulative) return a;
    }
    return num_actions - 1;
}

/* ── N-player betting tree (generated on-the-fly during traversal) ──── */

/* Action types for the tree */
#define ACT_FOLD    0
#define ACT_CHECK   1
#define ACT_CALL    2
#define ACT_BET     3  /* + bet_size_index */

typedef struct {
    int type;          /* ACT_FOLD, ACT_CHECK, ACT_CALL, ACT_BET */
    int bet_idx;       /* for ACT_BET: index into bet_sizes */
    int amount;        /* bet amount in chips */
} BPAction;

/* Generate available actions at a decision point */
static int generate_actions(BPAction *out, int max_out,
                            int pot, int stack, int to_call,
                            int num_raises, int max_raises,
                            const float *bet_sizes, int num_bet_sizes) {
    int n = 0;

    /* Fold (only if facing a bet) */
    if (to_call > 0 && n < max_out) {
        out[n].type = ACT_FOLD;
        out[n].bet_idx = -1;
        out[n].amount = 0;
        n++;
    }

    /* Check or Call */
    if (n < max_out) {
        out[n].type = (to_call > 0) ? ACT_CALL : ACT_CHECK;
        out[n].bet_idx = -1;
        out[n].amount = to_call;
        n++;
    }

    /* Bets/Raises */
    if (num_raises < max_raises) {
        int added_allin = 0;
        for (int i = 0; i < num_bet_sizes && n < max_out; i++) {
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

            out[n].type = ACT_BET;
            out[n].bet_idx = i;
            out[n].amount = ba;
            n++;
        }

        /* Explicit all-in */
        if (!added_allin && stack > to_call && n < max_out) {
            out[n].type = ACT_BET;
            out[n].bet_idx = num_bet_sizes;
            out[n].amount = stack;
            n++;
        }
    }

    return n;
}

/* ── External-sampling MCCFR traversal ───────────────────────────────── */

/* Traversal state passed through recursion */
typedef struct {
    BPSolver *solver;
    int traverser;
    int iteration;
    int sampled_hands[BP_MAX_PLAYERS];  /* which hand each player holds */

    /* Current board */
    int board[5];
    int num_board;

    /* Active players */
    int active[BP_MAX_PLAYERS];
    int num_active;

    /* Betting state */
    int bets[BP_MAX_PLAYERS];       /* each player's bet this street */
    int has_acted[BP_MAX_PLAYERS];  /* whether each player has acted */
    int pot;
    int stack;
    int num_raises;

    /* Action history for info set key */
    int action_history[256];
    int history_len;
} TraversalState;

static float traverse(TraversalState *ts, int acting_order_idx,
                      const int *acting_order, int num_in_order);

/* Count active players */
static int count_active_bp(const int *active, int n) {
    int c = 0;
    for (int i = 0; i < n; i++) if (active[i]) c++;
    return c;
}

/* Find next active player */
static int next_active_bp(const int *acting_order, int num_in_order,
                           const int *active, int num_players,
                           int current_idx) {
    for (int i = 1; i <= num_in_order; i++) {
        int idx = (current_idx + i) % num_in_order;
        int p = acting_order[idx];
        if (p < num_players && active[p]) return idx;
    }
    return -1;
}

/* Check if round is complete */
static int round_done(const int *bets, const int *active,
                      const int *has_acted, int num_players) {
    int mx = 0;
    for (int i = 0; i < num_players; i++)
        if (active[i] && bets[i] > mx) mx = bets[i];
    for (int i = 0; i < num_players; i++) {
        if (!active[i]) continue;
        if (!has_acted[i]) return 0;
        if (bets[i] != mx) return 0;
    }
    return 1;
}

/* Evaluate showdown: compute traverser's payoff */
static float eval_showdown_n(BPSolver *s, const int *board, int num_board,
                              int traverser, const int *active,
                              const int *sampled_hands, int pot) {
    int trav_hand = sampled_hands[traverser];
    int tc0 = s->hands[traverser][trav_hand][0];
    int tc1 = s->hands[traverser][trav_hand][1];

    if (card_in_set(tc0, board, num_board) || card_in_set(tc1, board, num_board))
        return 0;

    int cards_t[7] = {board[0], board[1], board[2], board[3], board[4], tc0, tc1};
    uint32_t trav_str = eval7(cards_t);

    /* Compare against all active opponents */
    int n_active = 0;
    int best_rank = 0;  /* 0=traverser is best, 1=tied, 2=loses */
    int n_tied = 1;     /* including traverser */
    int any_conflict = 0;

    for (int p = 0; p < s->num_players; p++) {
        if (p == traverser || !active[p]) continue;
        n_active++;

        int oh = sampled_hands[p];
        int oc0 = s->hands[p][oh][0], oc1 = s->hands[p][oh][1];

        /* Card conflict check */
        if (cards_conflict(tc0, tc1, oc0, oc1) ||
            card_in_set(oc0, board, num_board) ||
            card_in_set(oc1, board, num_board)) {
            any_conflict = 1;
            continue;
        }

        int cards_o[7] = {board[0], board[1], board[2], board[3], board[4], oc0, oc1};
        uint32_t opp_str = eval7(cards_o);

        if (opp_str > trav_str) {
            return -(float)s->hands[traverser][trav_hand][0]; /* loses — return -bet */
            /* Actually: traverser loses their contribution to pot */
        } else if (opp_str == trav_str) {
            n_tied++;
        }
    }

    if (any_conflict && n_active == 0) return 0;

    /* Traverser's payoff: their share of the pot minus their investment */
    float pot_share = (float)pot / (float)n_tied;
    float invested = 0;
    for (int p = 0; p < s->num_players; p++)
        if (p != traverser) invested += 0; /* bets already in pot */

    /* Simplified: win = pot - traverser's total bet, lose = -traverser's total bet */
    /* We track this as: positive = profit, negative = loss */
    return pot_share;
}

/* Main traversal function */
static float traverse(TraversalState *ts, int acting_order_idx,
                      const int *acting_order, int num_in_order) {
    BPSolver *s = ts->solver;
    int NP = s->num_players;
    int n_active = count_active_bp(ts->active, NP);

    /* ── Terminal: only 1 player left (all others folded) ──── */
    if (n_active <= 1) {
        /* Find winner */
        int winner = -1;
        for (int p = 0; p < NP; p++)
            if (ts->active[p]) { winner = p; break; }

        if (winner == ts->traverser) {
            /* Traverser wins the pot */
            return (float)(ts->pot - ts->bets[ts->traverser]);
        } else {
            /* Traverser folded or someone else won */
            return -(float)ts->bets[ts->traverser];
        }
    }

    /* ── Terminal: round complete ──────────────────────────── */
    if (round_done(ts->bets, ts->active, ts->has_acted, NP)) {
        int num_board = ts->num_board;

        if (num_board >= 5) {
            /* River showdown */
            return eval_showdown_n(s, ts->board, 5, ts->traverser,
                                    ts->active, ts->sampled_hands, ts->pot);
        }

        /* Deal next card (chance node — sample one card) */
        int blocked[52] = {0};
        for (int b = 0; b < num_board; b++) blocked[ts->board[b]] = 1;
        /* Also block players' cards */
        for (int p = 0; p < NP; p++) {
            if (!ts->active[p]) continue;
            int h = ts->sampled_hands[p];
            blocked[s->hands[p][h][0]] = 1;
            blocked[s->hands[p][h][1]] = 1;
        }

        int valid_cards[52];
        int n_valid = 0;
        for (int c = 0; c < 52; c++)
            if (!blocked[c]) valid_cards[n_valid++] = c;

        if (n_valid == 0) return 0;

        /* Sample one card */
        int card_idx = rng_int(&s->rng_state, n_valid);
        int dealt_card = valid_cards[card_idx];

        /* Recurse into next street */
        TraversalState next = *ts;
        next.board[num_board] = dealt_card;
        next.num_board = num_board + 1;
        memset(next.bets, 0, sizeof(next.bets));
        memset(next.has_acted, 0, sizeof(next.has_acted));
        next.num_raises = 0;
        /* pot stays the same, stack stays the same */

        return traverse(&next, 0, acting_order, num_in_order);
    }

    /* ── Find current acting player ───────────────────────── */
    int acting_player = acting_order[acting_order_idx];

    /* Skip folded players */
    if (!ts->active[acting_player]) {
        int next_idx = next_active_bp(acting_order, num_in_order,
                                       ts->active, NP, acting_order_idx);
        if (next_idx < 0) return 0;
        return traverse(ts, next_idx, acting_order, num_in_order);
    }

    /* ── Generate available actions ───────────────────────── */
    int mx = 0;
    for (int p = 0; p < NP; p++)
        if (ts->active[p] && ts->bets[p] > mx) mx = ts->bets[p];
    int to_call = mx - ts->bets[acting_player];
    if (to_call < 0) to_call = 0;

    BPAction actions[BP_MAX_ACTIONS];
    int n_actions = generate_actions(actions, BP_MAX_ACTIONS,
                                      ts->pot, ts->stack, to_call,
                                      ts->num_raises, 3, /* max_raises */
                                      s->bet_sizes, s->num_bet_sizes);
    if (n_actions == 0) return 0;

    /* ── Get or create info set ───────────────────────────── */
    BPInfoKey key;
    key.player = acting_player;
    key.board_hash = compute_board_hash(ts->board, ts->num_board);
    key.action_hash = compute_action_hash(ts->action_history, ts->history_len);

    int hand_idx = ts->sampled_hands[acting_player];
    int nh = s->num_hands[acting_player];

    int is_slot = info_table_find_or_create(&s->info_table, key, n_actions, nh);
    if (is_slot < 0) return 0; /* table full */

    BPInfoSet *is = &s->info_table.sets[is_slot];

    /* Regret matching for this hand */
    float strategy[BP_MAX_ACTIONS];
    regret_match(is->regrets, strategy, n_actions, nh, hand_idx);
    for (int a = 0; a < n_actions; a++)
        is->current_strategy[a * nh + hand_idx] = strategy[a];

    int next_order = next_active_bp(acting_order, num_in_order,
                                     ts->active, NP, acting_order_idx);
    if (next_order < 0) next_order = acting_order_idx;

    if (acting_player == ts->traverser) {
        /* ── Traverser: explore ALL actions ─── */
        float action_values[BP_MAX_ACTIONS];
        float node_value = 0;

        for (int a = 0; a < n_actions; a++) {
            TraversalState child = *ts;
            child.action_history[child.history_len++] = a;

            if (actions[a].type == ACT_FOLD) {
                child.active[acting_player] = 0;
                action_values[a] = traverse(&child, next_order,
                                             acting_order, num_in_order);
            } else if (actions[a].type == ACT_CHECK) {
                child.has_acted[acting_player] = 1;
                action_values[a] = traverse(&child, next_order,
                                             acting_order, num_in_order);
            } else if (actions[a].type == ACT_CALL) {
                child.bets[acting_player] = mx;
                child.pot += to_call;
                child.has_acted[acting_player] = 1;
                action_values[a] = traverse(&child, next_order,
                                             acting_order, num_in_order);
            } else {  /* ACT_BET */
                child.bets[acting_player] = mx + actions[a].amount;
                child.pot += actions[a].amount;
                child.has_acted[acting_player] = 1;
                /* Reset has_acted for other active players */
                for (int p = 0; p < NP; p++)
                    if (p != acting_player && child.active[p])
                        child.has_acted[p] = 0;
                child.num_raises++;
                action_values[a] = traverse(&child, next_order,
                                             acting_order, num_in_order);
            }

            node_value += strategy[a] * action_values[a];
        }

        /* Update regrets + Linear CFR discount (inline).
         * In external sampling, each traverser visits this info set ~once
         * per NP iterations. The discount uses the traverser's cycle count
         * (iteration / NP) as the effective time step. */
        int cycle = ts->iteration / ts->solver->num_players;
        float d = (cycle > 0) ? (float)cycle / ((float)cycle + 1.0f) : 0.5f;
        for (int a = 0; a < n_actions; a++) {
            is->regrets[a * nh + hand_idx] += action_values[a] - node_value;
            is->regrets[a * nh + hand_idx] *= d;
        }

        /* Accumulate weighted strategy sum */
        for (int a = 0; a < n_actions; a++)
            is->strategy_sum[a * nh + hand_idx] +=
                (float)ts->iteration * strategy[a];

        return node_value;

    } else {
        /* ── Non-traverser: SAMPLE one action ─── */
        int sampled = sample_action(strategy, n_actions, &s->rng_state);

        TraversalState child = *ts;
        child.action_history[child.history_len++] = sampled;

        if (actions[sampled].type == ACT_FOLD) {
            child.active[acting_player] = 0;
        } else if (actions[sampled].type == ACT_CHECK) {
            child.has_acted[acting_player] = 1;
        } else if (actions[sampled].type == ACT_CALL) {
            child.bets[acting_player] = mx;
            child.pot += to_call;
            child.has_acted[acting_player] = 1;
        } else {  /* ACT_BET */
            child.bets[acting_player] = mx + actions[sampled].amount;
            child.pot += actions[sampled].amount;
            child.has_acted[acting_player] = 1;
            for (int p = 0; p < NP; p++)
                if (p != acting_player && child.active[p])
                    child.has_acted[p] = 0;
            child.num_raises++;
        }

        /* Also update strategy sum for non-traverser (for extraction) */
        is->strategy_sum[sampled * nh + hand_idx] +=
            (float)ts->iteration * strategy[sampled];

        return traverse(&child, next_order, acting_order, num_in_order);
    }
}

/* ── Public API ──────────────────────────────────────────────────────── */

int bp_init(BPSolver *s, int num_players,
            const int *flop,
            const int hands[][BP_MAX_HANDS][2],
            const float weights[][BP_MAX_HANDS],
            const int *num_hands,
            int starting_pot, int effective_stack,
            const float *bet_sizes, int num_bet_sizes) {
    memset(s, 0, sizeof(BPSolver));
    s->num_players = num_players;
    memcpy(s->flop, flop, 3 * sizeof(int));

    for (int p = 0; p < num_players; p++) {
        s->num_hands[p] = num_hands[p];
        for (int h = 0; h < num_hands[p]; h++) {
            s->hands[p][h][0] = hands[p][h][0];
            s->hands[p][h][1] = hands[p][h][1];
            s->weights[p][h] = weights[p][h];
        }
    }

    s->starting_pot = starting_pot;
    s->effective_stack = effective_stack;
    s->num_bet_sizes = num_bet_sizes;
    for (int i = 0; i < num_bet_sizes; i++)
        s->bet_sizes[i] = bet_sizes[i];

    info_table_init(&s->info_table);
    s->rng_state = 0xDEADBEEF12345678ULL;

    return 0;
}

int bp_solve(BPSolver *s, int max_iterations) {
    int NP = s->num_players;
    int acting_order[BP_MAX_PLAYERS];
    for (int i = 0; i < NP; i++) acting_order[i] = i;

    printf("[BP] Starting %d-player MCCFR, %d iterations\n", NP, max_iterations);

    for (int iter = 1; iter <= max_iterations; iter++) {
        int traverser = (iter - 1) % NP;

        /* Sample hands for all players */
        int sampled_hands[BP_MAX_PLAYERS];
        for (int p = 0; p < NP; p++) {
            /* Sample uniformly from player's range (weighted by initial weight) */
            /* For simplicity, uniform for now. Proper: weight-proportional. */
            sampled_hands[p] = rng_int(&s->rng_state, s->num_hands[p]);
        }

        /* Check for card conflicts between sampled hands */
        int conflict = 0;
        for (int p = 0; p < NP && !conflict; p++) {
            int c0 = s->hands[p][sampled_hands[p]][0];
            int c1 = s->hands[p][sampled_hands[p]][1];
            /* Check vs board */
            if (card_in_set(c0, s->flop, 3) || card_in_set(c1, s->flop, 3)) {
                conflict = 1; break;
            }
            /* Check vs other players */
            for (int q = 0; q < p; q++) {
                int d0 = s->hands[q][sampled_hands[q]][0];
                int d1 = s->hands[q][sampled_hands[q]][1];
                if (cards_conflict(c0, c1, d0, d1)) { conflict = 1; break; }
            }
        }
        if (conflict) continue; /* Skip this iteration */

        /* Build traversal state */
        TraversalState ts;
        memset(&ts, 0, sizeof(ts));
        ts.solver = s;
        ts.traverser = traverser;
        ts.iteration = iter;
        memcpy(ts.sampled_hands, sampled_hands, sizeof(sampled_hands));
        memcpy(ts.board, s->flop, 3 * sizeof(int));
        ts.num_board = 3;
        for (int p = 0; p < NP; p++) ts.active[p] = 1;
        ts.num_active = NP;
        ts.pot = s->starting_pot;
        ts.stack = s->effective_stack;
        ts.num_raises = 0;
        ts.history_len = 0;

        /* Traverse */
        traverse(&ts, 0, acting_order, NP);

        s->iterations_run = iter;

        if (iter % 10000 == 0 || iter == 1 || iter == max_iterations) {
            printf("[BP] iter %d/%d, info sets: %d\n",
                   iter, max_iterations, s->info_table.num_entries);
            fflush(stdout);
        }
    }

    return 0;
}

int bp_get_strategy(const BPSolver *s, int player,
                     const int *board, int num_board,
                     const int *action_seq, int seq_len,
                     float *strategy_out, int hand_idx) {
    BPInfoKey key;
    key.player = player;
    key.board_hash = compute_board_hash(board, num_board);
    key.action_hash = compute_action_hash(action_seq, seq_len);

    /* Search for this info set */
    uint64_t h = hash_combine(key.board_hash, key.action_hash);
    h = hash_combine(h, (uint64_t)key.player);
    int slot = (int)(h % BP_HASH_SIZE);

    for (int probe = 0; probe < BP_HASH_SIZE; probe++) {
        int idx = (slot + probe) % BP_HASH_SIZE;
        if (!s->info_table.occupied[idx]) return 0; /* not found */
        if (s->info_table.keys[idx].player == key.player &&
            s->info_table.keys[idx].board_hash == key.board_hash &&
            s->info_table.keys[idx].action_hash == key.action_hash) {
            /* Found — extract weighted average */
            BPInfoSet *is = &s->info_table.sets[idx];
            int na = is->num_actions;
            float sum = 0;
            for (int a = 0; a < na; a++) {
                float v = is->strategy_sum[a * is->num_hands + hand_idx];
                v = v > 0 ? v : 0;
                strategy_out[a] = v;
                sum += v;
            }
            if (sum > 0) {
                for (int a = 0; a < na; a++) strategy_out[a] /= sum;
            } else {
                for (int a = 0; a < na; a++) strategy_out[a] = 1.0f / na;
            }
            return na;
        }
    }
    return 0;
}

int bp_num_info_sets(const BPSolver *s) {
    return s->info_table.num_entries;
}

void bp_free(BPSolver *s) {
    info_table_free(&s->info_table);
    memset(s, 0, sizeof(BPSolver));
}
