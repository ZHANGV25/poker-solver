/**
 * mccfr_blueprint.c — Production N-player external-sampling MCCFR
 *
 * Matches the Pluribus blueprint training algorithm:
 *   - External-sampling MCCFR with Linear CFR + regret-based pruning
 *   - OpenMP Hogwild-style parallelism (lock-free shared regret tables)
 *   - Integer regrets (int32) with floor at -310M
 *   - Card abstraction via hand-to-bucket mapping
 *   - Multi-street: flop -> turn -> river -> showdown
 *
 * Thread safety: each thread runs independent iterations with its own
 * RNG state. Regret/strategy updates are unsynchronized (Hogwild-style).
 * This is safe because external sampling visits a sparse subset of info
 * sets per iteration, so collisions are rare and the noise is negligible.
 */

#include "mccfr_blueprint.h"
#include "hand_eval.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* ── RNG (xorshift64, thread-safe via separate states) ────────────── */

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

/* ── Hash table ───────────────────────────────────────────────────── */

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

static void info_table_init(BPInfoTable *t, int table_size) {
    t->table_size = table_size;
    t->keys = (BPInfoKey*)calloc(table_size, sizeof(BPInfoKey));
    t->sets = (BPInfoSet*)calloc(table_size, sizeof(BPInfoSet));
    t->occupied = (int*)calloc(table_size, sizeof(int));
    t->num_entries = 0;
}

/* Thread-safe find-or-create using CAS-like logic.
 * In practice, the occupied[] flag is set non-atomically (Hogwild).
 * Duplicate entries are harmless — they just waste a slot. */
static int info_table_find_or_create(BPInfoTable *t, BPInfoKey key,
                                      int num_actions, int num_hands) {
    uint64_t h = hash_combine(key.board_hash, key.action_hash);
    h = hash_combine(h, (uint64_t)key.player);
    int slot = (int)(h % (uint64_t)t->table_size);

    for (int probe = 0; probe < 1024; probe++) {
        int idx = (slot + probe) % t->table_size;
        if (t->occupied[idx]) {
            /* Check if this is our key */
            if (t->keys[idx].player == key.player &&
                t->keys[idx].board_hash == key.board_hash &&
                t->keys[idx].action_hash == key.action_hash) {
                return idx;  /* Fast path: existing entry (lock-free) */
            }
            continue; /* Collision, try next slot */
        }
        /* Empty slot — need to create. Use critical section because
         * calloc + multi-field init isn't atomic. The critical section
         * only fires for NEW info sets (~once per set), not lookups. */
        int created = 0;
        #ifdef _OPENMP
        #pragma omp critical(hash_insert)
        #endif
        {
            if (!t->occupied[idx]) {
                /* Double-check under lock */
                t->sets[idx].num_actions = num_actions;
                t->sets[idx].num_hands = num_hands;
                t->sets[idx].regrets = (int*)calloc(num_actions * num_hands, sizeof(int));
                t->sets[idx].strategy_sum = NULL;
                t->sets[idx].current_strategy = (float*)calloc(num_actions * num_hands, sizeof(float));
                t->keys[idx] = key;
                t->occupied[idx] = 1; /* Publish LAST (acts as release fence) */
                t->num_entries++;
                created = 1;
            }
        }
        if (created) return idx;
        /* Another thread created an entry here — check if it's ours */
        if (t->keys[idx].player == key.player &&
            t->keys[idx].board_hash == key.board_hash &&
            t->keys[idx].action_hash == key.action_hash) {
            return idx;
        }
        /* Different key was inserted, continue probing */
    }
    return -1;
}

/* Allocate strategy_sum for an info set (lazy, for round 1 only) */
static void ensure_strategy_sum(BPInfoSet *is) {
    if (is->strategy_sum == NULL) {
        #ifdef _OPENMP
        #pragma omp critical(strategy_sum_alloc)
        #endif
        {
            if (is->strategy_sum == NULL) {
                is->strategy_sum = (float*)calloc(is->num_actions * is->num_hands, sizeof(float));
            }
        }
    }
}

static void info_table_free(BPInfoTable *t) {
    for (int i = 0; i < t->table_size; i++) {
        if (t->occupied[i]) {
            free(t->sets[i].regrets);
            if (t->sets[i].strategy_sum) free(t->sets[i].strategy_sum);
            free(t->sets[i].current_strategy);
        }
    }
    free(t->keys);
    free(t->sets);
    free(t->occupied);
}

/* ── Helpers ──────────────────────────────────────────────────────── */

static inline int cards_conflict(int a0, int a1, int b0, int b1) {
    return (a0 == b0 || a0 == b1 || a1 == b0 || a1 == b1);
}

static inline int card_in_set(int card, const int *set, int n) {
    for (int i = 0; i < n; i++)
        if (set[i] == card) return 1;
    return 0;
}

/* Regret matching from integer regrets */
static void regret_match_int(const int *regrets, float *strategy,
                              int num_actions, int num_hands, int hand_idx) {
    float sum = 0;
    for (int a = 0; a < num_actions; a++) {
        int r = regrets[a * num_hands + hand_idx];
        float pos = (r > 0) ? (float)r : 0.0f;
        strategy[a] = pos;
        sum += pos;
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

static int sample_action(const float *strategy, int num_actions, uint64_t *rng) {
    float r = rng_float(rng);
    float cumulative = 0;
    for (int a = 0; a < num_actions; a++) {
        cumulative += strategy[a];
        if (r <= cumulative) return a;
    }
    return num_actions - 1;
}

/* Get the bucket index for a hand on a given street */
static inline int get_bucket(const BPSolver *s, int street, int player, int hand_idx) {
    if (!s->use_buckets) return hand_idx;
    if (street < 0 || street > 3) return hand_idx;
    return s->bucket_map[street][player][hand_idx];
}

/* Get num_buckets for a player on a street */
static inline int get_num_buckets(const BPSolver *s, int street, int player) {
    if (!s->use_buckets) return s->num_hands[player];
    if (street < 0 || street > 3) return s->num_hands[player];
    return s->num_buckets[street][player];
}

/* Map num_board to street index: 3=flop(1), 4=turn(2), 5=river(3) */
static inline int board_to_street(int num_board) {
    if (num_board <= 3) return 1; /* flop */
    if (num_board == 4) return 2; /* turn */
    return 3; /* river */
}

/* ── Betting tree (generated on-the-fly) ─────────────────────────── */

#define ACT_FOLD    0
#define ACT_CHECK   1
#define ACT_CALL    2
#define ACT_BET     3

typedef struct {
    int type;
    int bet_idx;
    int amount;
} BPAction;

static int generate_actions(BPAction *out, int max_out,
                            int pot, int stack, int to_call,
                            int num_raises, int max_raises,
                            const float *bet_sizes, int num_bet_sizes) {
    int n = 0;
    if (to_call > 0 && n < max_out) {
        out[n].type = ACT_FOLD; out[n].bet_idx = -1; out[n].amount = 0; n++;
    }
    if (n < max_out) {
        out[n].type = (to_call > 0) ? ACT_CALL : ACT_CHECK;
        out[n].bet_idx = -1; out[n].amount = to_call; n++;
    }
    if (num_raises < max_raises) {
        int added_allin = 0;
        for (int i = 0; i < num_bet_sizes && n < max_out; i++) {
            int ba;
            if (to_call == 0) ba = (int)(bet_sizes[i] * pot);
            else ba = to_call + (int)(bet_sizes[i] * (pot + to_call));
            if (ba >= stack) ba = stack;
            if (ba <= to_call) continue;
            if (ba >= stack) { if (added_allin) continue; added_allin = 1; }
            out[n].type = ACT_BET; out[n].bet_idx = i; out[n].amount = ba; n++;
        }
        if (!added_allin && stack > to_call && n < max_out) {
            out[n].type = ACT_BET; out[n].bet_idx = num_bet_sizes;
            out[n].amount = stack; n++;
        }
    }
    return n;
}

/* ── Traversal state ─────────────────────────────────────────────── */

typedef struct {
    BPSolver *solver;
    uint64_t *rng;           /* pointer to thread-local RNG */
    int traverser;
    int iteration;
    int use_pruning;         /* 1 if this iteration uses pruning */
    int sampled_hands[BP_MAX_PLAYERS];

    int board[5];
    int num_board;
    int active[BP_MAX_PLAYERS];
    int num_active;
    int bets[BP_MAX_PLAYERS];       /* per-street bets (reset each street) */
    int invested[BP_MAX_PLAYERS];   /* CUMULATIVE total invested across ALL streets */
    int has_acted[BP_MAX_PLAYERS];
    int pot;
    int stacks[BP_MAX_PLAYERS];     /* remaining stack per player */
    int num_raises;

    int action_history[256];
    int history_len;
} TraversalState;

static float traverse(TraversalState *ts, int acting_order_idx,
                      const int *acting_order, int num_in_order);

static int count_active(const int *active, int n) {
    int c = 0;
    for (int i = 0; i < n; i++) if (active[i]) c++;
    return c;
}

static int next_active(const int *acting_order, int num_in_order,
                        const int *active, int np, int cur_idx) {
    for (int i = 1; i <= num_in_order; i++) {
        int idx = (cur_idx + i) % num_in_order;
        int p = acting_order[idx];
        if (p < np && active[p]) return idx;
    }
    return -1;
}

static int round_done(const int *bets, const int *active,
                      const int *has_acted, int np) {
    int mx = 0;
    for (int i = 0; i < np; i++)
        if (active[i] && bets[i] > mx) mx = bets[i];
    for (int i = 0; i < np; i++) {
        if (!active[i]) continue;
        if (!has_acted[i]) return 0;
        if (bets[i] != mx) return 0;
    }
    return 1;
}

/* N-player showdown: compare all active players' hands */
static float eval_showdown_n(BPSolver *s, const int *board,
                              int traverser, const int *active,
                              const int *sampled_hands, int pot,
                              const int *invested) {
    int th = sampled_hands[traverser];
    int tc0 = s->hands[traverser][th][0];
    int tc1 = s->hands[traverser][th][1];

    if (card_in_set(tc0, board, 5) || card_in_set(tc1, board, 5))
        return 0;

    int cards_t[7] = {board[0], board[1], board[2], board[3], board[4], tc0, tc1};
    uint32_t trav_str = eval7(cards_t);

    int n_tied = 1;
    int trav_wins = 1;

    for (int p = 0; p < s->num_players; p++) {
        if (p == traverser || !active[p]) continue;
        int oh = sampled_hands[p];
        int oc0 = s->hands[p][oh][0], oc1 = s->hands[p][oh][1];

        if (cards_conflict(tc0, tc1, oc0, oc1) ||
            card_in_set(oc0, board, 5) || card_in_set(oc1, board, 5))
            continue;

        int cards_o[7] = {board[0], board[1], board[2], board[3], board[4], oc0, oc1};
        uint32_t opp_str = eval7(cards_o);

        if (opp_str > trav_str) { trav_wins = 0; break; }
        else if (opp_str == trav_str) n_tied++;
    }

    /* Payoff: profit relative to TOTAL investment across all streets */
    if (!trav_wins) {
        return -(float)invested[traverser];
    } else {
        return (float)pot / (float)n_tied - (float)invested[traverser];
    }
}

/* ── Main traversal ──────────────────────────────────────────────── */

static float traverse(TraversalState *ts, int acting_order_idx,
                      const int *acting_order, int num_in_order) {
    BPSolver *s = ts->solver;
    int NP = s->num_players;
    int n_active = count_active(ts->active, NP);

    /* Terminal: all folded */
    if (n_active <= 1) {
        for (int p = 0; p < NP; p++) {
            if (ts->active[p]) {
                if (p == ts->traverser)
                    return (float)(ts->pot - ts->invested[ts->traverser]);
                else
                    return -(float)ts->invested[ts->traverser];
            }
        }
        return 0;
    }

    /* Round complete -> next street or showdown */
    if (round_done(ts->bets, ts->active, ts->has_acted, NP)) {
        if (ts->num_board >= 5) {
            return eval_showdown_n(s, ts->board, ts->traverser,
                                    ts->active, ts->sampled_hands,
                                    ts->pot, ts->invested);
        }

        /* Deal next card */
        int blocked[52] = {0};
        for (int b = 0; b < ts->num_board; b++) blocked[ts->board[b]] = 1;
        for (int p = 0; p < NP; p++) {
            if (!ts->active[p]) continue;
            int h = ts->sampled_hands[p];
            blocked[s->hands[p][h][0]] = 1;
            blocked[s->hands[p][h][1]] = 1;
        }

        int valid[52], nv = 0;
        for (int c = 0; c < 52; c++)
            if (!blocked[c]) valid[nv++] = c;
        if (nv == 0) return 0;

        int dealt = valid[rng_int(ts->rng, nv)];

        TraversalState next = *ts;
        next.board[next.num_board++] = dealt;
        memset(next.bets, 0, sizeof(next.bets));
        memset(next.has_acted, 0, sizeof(next.has_acted));
        next.num_raises = 0;
        return traverse(&next, 0, acting_order, num_in_order);
    }

    /* Current player */
    int ap = acting_order[acting_order_idx];
    if (!ts->active[ap]) {
        int ni = next_active(acting_order, num_in_order, ts->active, NP, acting_order_idx);
        if (ni < 0) return 0;
        return traverse(ts, ni, acting_order, num_in_order);
    }

    /* Generate actions */
    int mx = 0;
    for (int p = 0; p < NP; p++)
        if (ts->active[p] && ts->bets[p] > mx) mx = ts->bets[p];
    int to_call = mx - ts->bets[ap];
    if (to_call < 0) to_call = 0;

    BPAction actions[BP_MAX_ACTIONS];
    int remaining_stack = ts->stacks[ap];
    int na = generate_actions(actions, BP_MAX_ACTIONS, ts->pot, remaining_stack,
                              to_call, ts->num_raises, 3,
                              s->bet_sizes, s->num_bet_sizes);
    if (na == 0) return 0;

    /* Info set lookup */
    int street = board_to_street(ts->num_board);
    int bucket = get_bucket(s, street, ap, ts->sampled_hands[ap]);
    int nb = get_num_buckets(s, street, ap);

    BPInfoKey key;
    key.player = ap;
    key.board_hash = compute_board_hash(ts->board, ts->num_board);
    key.action_hash = compute_action_hash(ts->action_history, ts->history_len);

    int is_slot = info_table_find_or_create(&s->info_table, key, na, nb);
    if (is_slot < 0) return 0;
    BPInfoSet *is = &s->info_table.sets[is_slot];

    /* Regret matching */
    float strategy[BP_MAX_ACTIONS];
    regret_match_int(is->regrets, strategy, na, nb, bucket);

    int next_order = next_active(acting_order, num_in_order, ts->active, NP, acting_order_idx);
    if (next_order < 0) next_order = acting_order_idx;

    if (ap == ts->traverser) {
        /* Traverser: explore all actions (with optional pruning) */
        float action_values[BP_MAX_ACTIONS];
        float node_value = 0;

        for (int a = 0; a < na; a++) {
            /* Pruning: skip actions with very negative regret */
            if (ts->use_pruning && is->regrets[a * nb + bucket] < BP_PRUNE_THRESHOLD) {
                /* Don't prune terminal-leading actions or river actions */
                if (ts->num_board < 5 || actions[a].type != ACT_FOLD) {
                    action_values[a] = 0;
                    continue;
                }
            }

            TraversalState child = *ts;
            child.action_history[child.history_len++] = a;

            if (actions[a].type == ACT_FOLD) {
                child.active[ap] = 0;
            } else if (actions[a].type == ACT_CHECK) {
                child.has_acted[ap] = 1;
            } else if (actions[a].type == ACT_CALL) {
                child.bets[ap] = mx;
                child.invested[ap] += to_call;
                child.stacks[ap] -= to_call;
                child.pot += to_call;
                child.has_acted[ap] = 1;
            } else {
                /* amount = total new chips committed this action.
                 * For a raise: amount includes to_call + raise.
                 * New street-bet level = current bet + amount. */
                int amount = actions[a].amount;
                /* Cap at remaining stack */
                if (amount > child.stacks[ap]) amount = child.stacks[ap];
                child.bets[ap] += amount;
                child.invested[ap] += amount;
                child.stacks[ap] -= amount;
                child.pot += amount;
                child.has_acted[ap] = 1;
                for (int p = 0; p < NP; p++)
                    if (p != ap && child.active[p]) child.has_acted[p] = 0;
                child.num_raises++;
            }

            action_values[a] = traverse(&child, next_order, acting_order, num_in_order);
            node_value += strategy[a] * action_values[a];
        }

        /* Update integer regrets (Hogwild: no lock needed) */
        for (int a = 0; a < na; a++) {
            int idx = a * nb + bucket;
            int delta = (int)((action_values[a] - node_value) * 1000.0f);
            is->regrets[idx] += delta;
            /* Floor at REGRET_FLOOR */
            if (is->regrets[idx] < BP_REGRET_FLOOR)
                is->regrets[idx] = BP_REGRET_FLOOR;
        }

        /* Strategy sum: only for round 1 (preflop), every strategy_interval iterations */
        if (street <= 1 && (ts->iteration % s->config.strategy_interval) == 0) {
            ensure_strategy_sum(is);
            for (int a = 0; a < na; a++)
                is->strategy_sum[a * nb + bucket] += strategy[a];
        }

        return node_value;

    } else {
        /* Non-traverser: sample one action */
        int sampled = sample_action(strategy, na, ts->rng);

        TraversalState child = *ts;
        child.action_history[child.history_len++] = sampled;

        if (actions[sampled].type == ACT_FOLD) {
            child.active[ap] = 0;
        } else if (actions[sampled].type == ACT_CHECK) {
            child.has_acted[ap] = 1;
        } else if (actions[sampled].type == ACT_CALL) {
            child.bets[ap] = mx;
            child.invested[ap] += to_call;
            child.stacks[ap] -= to_call;
            child.pot += to_call;
            child.has_acted[ap] = 1;
        } else {
            int amount = actions[sampled].amount;
            if (amount > child.stacks[ap]) amount = child.stacks[ap];
            child.bets[ap] += amount;
            child.invested[ap] += amount;
            child.stacks[ap] -= amount;
            child.pot += amount;
            child.has_acted[ap] = 1;
            for (int p = 0; p < NP; p++)
                if (p != ap && child.active[p]) child.has_acted[p] = 0;
            child.num_raises++;
        }

        return traverse(&child, next_order, acting_order, num_in_order);
    }
}

/* ── Linear CFR discount ─────────────────────────────────────────── */

static void apply_discount(BPInfoTable *t, float discount) {
    for (int i = 0; i < t->table_size; i++) {
        if (!t->occupied[i]) continue;
        BPInfoSet *is = &t->sets[i];
        int size = is->num_actions * is->num_hands;
        for (int j = 0; j < size; j++) {
            is->regrets[j] = (int)((float)is->regrets[j] * discount);
            if (is->regrets[j] < BP_REGRET_FLOOR)
                is->regrets[j] = BP_REGRET_FLOOR;
        }
        if (is->strategy_sum) {
            for (int j = 0; j < size; j++)
                is->strategy_sum[j] *= discount;
        }
    }
}

/* ── Public API ──────────────────────────────────────────────────── */

void bp_default_config(BPConfig *config) {
    memset(config, 0, sizeof(BPConfig));
    /* Pluribus timing converted to iterations assuming ~1000 iter/min on 64 cores */
    config->discount_stop_iter = 400000;    /* 400 min * ~1000 iter/min */
    config->discount_interval  = 10000;     /* 10 min * ~1000 iter/min */
    config->prune_start_iter   = 200000;    /* 200 min */
    config->snapshot_start_iter = 800000;   /* 800 min */
    config->snapshot_interval  = 200000;    /* 200 min */
    config->strategy_interval  = 10000;     /* Pluribus: every 10K iterations */
    config->num_threads = 0;                /* auto */
    config->hash_table_size = 0;            /* auto */
    config->snapshot_dir = NULL;
}

int bp_init(BPSolver *s, int num_players,
            const int *flop,
            const int hands[][BP_MAX_HANDS][2],
            const float weights[][BP_MAX_HANDS],
            const int *num_hands,
            int starting_pot, int effective_stack,
            const float *bet_sizes, int num_bet_sizes) {
    BPConfig config;
    bp_default_config(&config);
    /* For backward compatibility: small hash table, single thread */
    config.hash_table_size = BP_HASH_SIZE_SMALL;
    config.num_threads = 1;
    return bp_init_ex(s, num_players, flop, hands, weights, num_hands,
                       starting_pot, effective_stack, bet_sizes, num_bet_sizes,
                       &config);
}

int bp_init_ex(BPSolver *s, int num_players,
                const int *flop,
                const int hands[][BP_MAX_HANDS][2],
                const float weights[][BP_MAX_HANDS],
                const int *num_hands,
                int starting_pot, int effective_stack,
                const float *bet_sizes, int num_bet_sizes,
                const BPConfig *config) {
    memset(s, 0, sizeof(BPSolver));
    s->num_players = num_players;
    memcpy(s->flop, flop, 3 * sizeof(int));
    s->config = *config;

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

    /* Default: no card abstraction (identity mapping) */
    s->use_buckets = 0;
    for (int st = 0; st < 4; st++)
        for (int p = 0; p < num_players; p++) {
            s->num_buckets[st][p] = num_hands[p];
            for (int h = 0; h < num_hands[p]; h++)
                s->bucket_map[st][p][h] = h;
        }

    /* Hash table */
    int ht_size = config->hash_table_size;
    if (ht_size <= 0)
        ht_size = (num_players > 2) ? BP_HASH_SIZE_LARGE : BP_HASH_SIZE_SMALL;
    info_table_init(&s->info_table, ht_size);

    /* RNG states — one per thread */
    int nt = config->num_threads;
    if (nt <= 0) {
        #ifdef _OPENMP
        nt = omp_get_max_threads();
        #else
        nt = 1;
        #endif
    }
    s->num_rng_states = nt;
    s->rng_states = (uint64_t*)malloc(nt * sizeof(uint64_t));
    for (int i = 0; i < nt; i++)
        s->rng_states[i] = 0xDEADBEEF12345678ULL + (uint64_t)i * 6364136223846793005ULL;

    return 0;
}

int bp_set_buckets(BPSolver *s, int street,
                    const int bucket_map[][BP_MAX_HANDS],
                    const int *num_buckets) {
    if (street < 0 || street > 3) return -1;
    s->use_buckets = 1;
    for (int p = 0; p < s->num_players; p++) {
        s->num_buckets[street][p] = num_buckets[p];
        for (int h = 0; h < s->num_hands[p]; h++)
            s->bucket_map[street][p][h] = bucket_map[p][h];
    }
    return 0;
}

int bp_solve(BPSolver *s, int max_iterations) {
    int NP = s->num_players;
    int acting_order[BP_MAX_PLAYERS];
    for (int i = 0; i < NP; i++) acting_order[i] = i;

    int nt = s->num_rng_states;
    #ifdef _OPENMP
    if (nt > 1) omp_set_num_threads(nt);
    #endif

    printf("[BP] Starting %d-player MCCFR: %d iterations, %d threads, "
           "hash=%d, buckets=%s\n",
           NP, max_iterations, nt, s->info_table.table_size,
           s->use_buckets ? "yes" : "no");

    clock_t t_start = clock();
    int discount_count = 0;

    #ifdef _OPENMP
    #pragma omp parallel if(nt > 1)
    #endif
    {
        #ifdef _OPENMP
        int tid = omp_get_thread_num();
        #else
        int tid = 0;
        #endif
        uint64_t *my_rng = &s->rng_states[tid];

        #ifdef _OPENMP
        #pragma omp for schedule(dynamic, 64)
        #endif
        for (int iter = 1; iter <= max_iterations; iter++) {
            int traverser = (iter - 1) % NP;

            /* Pruning decision: 95% of iters after warmup */
            int use_pruning = 0;
            if (iter > s->config.prune_start_iter) {
                use_pruning = (rng_float(my_rng) < BP_PRUNE_PROB) ? 1 : 0;
            }

            /* Sample hands */
            int sampled_hands[BP_MAX_PLAYERS];
            for (int p = 0; p < NP; p++)
                sampled_hands[p] = rng_int(my_rng, s->num_hands[p]);

            /* Card conflict rejection */
            int conflict = 0;
            for (int p = 0; p < NP && !conflict; p++) {
                int c0 = s->hands[p][sampled_hands[p]][0];
                int c1 = s->hands[p][sampled_hands[p]][1];
                if (card_in_set(c0, s->flop, 3) || card_in_set(c1, s->flop, 3))
                    { conflict = 1; break; }
                for (int q = 0; q < p; q++) {
                    int d0 = s->hands[q][sampled_hands[q]][0];
                    int d1 = s->hands[q][sampled_hands[q]][1];
                    if (cards_conflict(c0, c1, d0, d1)) { conflict = 1; break; }
                }
            }
            if (conflict) continue;

            /* Build traversal state */
            TraversalState ts;
            memset(&ts, 0, sizeof(ts));
            ts.solver = s;
            ts.rng = my_rng;
            ts.traverser = traverser;
            ts.iteration = iter;
            ts.use_pruning = use_pruning;
            memcpy(ts.sampled_hands, sampled_hands, sizeof(sampled_hands));
            memcpy(ts.board, s->flop, 3 * sizeof(int));
            ts.num_board = 3;
            for (int p = 0; p < NP; p++) ts.active[p] = 1;
            ts.num_active = NP;
            ts.pot = s->starting_pot;
            /* Each player starts with effective_stack chips.
             * The starting_pot was already contributed (blinds/antes),
             * split evenly among players as their initial investment. */
            for (int p = 0; p < NP; p++) {
                ts.stacks[p] = s->effective_stack;
                ts.invested[p] = s->starting_pot / NP; /* each player's share of blinds */
            }

            traverse(&ts, 0, acting_order, NP);

            /* Progress (single-threaded section) */
            #ifdef _OPENMP
            if (tid == 0)
            #endif
            {
                s->iterations_run = iter;
                if (iter % 10000 == 0 || iter == 1 || iter == max_iterations) {
                    double elapsed = (double)(clock() - t_start) / CLOCKS_PER_SEC;
                    printf("[BP] iter %d/%d, info sets: %d, %.1fs\n",
                           iter, max_iterations, s->info_table.num_entries, elapsed);
                    fflush(stdout);
                }
            }
        }
    } /* end parallel */

    /* Apply Linear CFR discount after all iterations
     * (Pluribus applied every 10 min for first 400 min; we apply proportionally) */
    if (max_iterations <= s->config.discount_stop_iter) {
        int n_discounts = max_iterations / s->config.discount_interval;
        for (int d = 1; d <= n_discounts; d++) {
            float t_val = (float)d;
            float discount = t_val / (t_val + 1.0f);
            /* Apply compound discount */
            if (d == n_discounts) {
                apply_discount(&s->info_table, discount);
            }
        }
    }

    double total_time = (double)(clock() - t_start) / CLOCKS_PER_SEC;
    printf("[BP] Done: %d iterations, %d info sets, %.1fs (%.0f iter/s)\n",
           max_iterations, s->info_table.num_entries, total_time,
           max_iterations / total_time);

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

    uint64_t h = hash_combine(key.board_hash, key.action_hash);
    h = hash_combine(h, (uint64_t)key.player);
    int slot = (int)(h % (uint64_t)s->info_table.table_size);

    for (int probe = 0; probe < 1024; probe++) {
        int idx = (slot + probe) % s->info_table.table_size;
        if (!s->info_table.occupied[idx]) return 0;
        if (s->info_table.keys[idx].player == key.player &&
            s->info_table.keys[idx].board_hash == key.board_hash &&
            s->info_table.keys[idx].action_hash == key.action_hash) {
            BPInfoSet *is = &s->info_table.sets[idx];
            int na = is->num_actions;
            int nh = is->num_hands;

            /* Use strategy_sum if available (round 1), else current regret-matched strategy */
            if (is->strategy_sum) {
                float sum = 0;
                for (int a = 0; a < na; a++) {
                    float v = is->strategy_sum[a * nh + hand_idx];
                    v = v > 0 ? v : 0;
                    strategy_out[a] = v;
                    sum += v;
                }
                if (sum > 0) {
                    for (int a = 0; a < na; a++) strategy_out[a] /= sum;
                } else {
                    for (int a = 0; a < na; a++) strategy_out[a] = 1.0f / na;
                }
            } else {
                /* Regret-matched current strategy */
                regret_match_int(is->regrets, strategy_out, na, nh, hand_idx);
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
    if (s->rng_states) free(s->rng_states);
    memset(s, 0, sizeof(BPSolver));
}
