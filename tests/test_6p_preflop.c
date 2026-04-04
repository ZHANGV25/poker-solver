/*
 * 6-player preflop-only external-sampling MCCFR.
 * Tests whether trash hands converge to fold from UTG.
 * No postflop — goes directly to equity-based showdown.
 *
 * Build: gcc -O2 -o test_6p tests/test_6p_preflop.c -lm
 * Run:   ./test_6p [iterations]   (default: 50M, takes ~30s)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#define NP 6
#define N_CLASSES 169
#define N_ACTIONS 6  /* fold, call, r0.5x, r1x, r2x, r3x */
#define SB_AMT 50
#define BB_AMT 100
#define STACK 10000

/* Preflop hand strength (0-1 scale, approximate heads-up equity).
 * Computed from standard preflop equity charts. */
static float STRENGTH[N_CLASSES];

static const char *LABELS[] = {
    "AA","AKs","AKo","AQs","AQo","AJs","AJo","ATs","ATo","A9s","A9o","A8s","A8o",
    "A7s","A7o","A6s","A6o","A5s","A5o","A4s","A4o","A3s","A3o","A2s","A2o",
    "KK","KQs","KQo","KJs","KJo","KTs","KTo","K9s","K9o","K8s","K8o","K7s","K7o",
    "K6s","K6o","K5s","K5o","K4s","K4o","K3s","K3o","K2s","K2o",
    "QQ","QJs","QJo","QTs","QTo","Q9s","Q9o","Q8s","Q8o","Q7s","Q7o","Q6s","Q6o",
    "Q5s","Q5o","Q4s","Q4o","Q3s","Q3o","Q2s","Q2o",
    "JJ","JTs","JTo","J9s","J9o","J8s","J8o","J7s","J7o","J6s","J6o","J5s","J5o",
    "J4s","J4o","J3s","J3o","J2s","J2o",
    "TT","T9s","T9o","T8s","T8o","T7s","T7o","T6s","T6o","T5s","T5o","T4s","T4o",
    "T3s","T3o","T2s","T2o",
    "99","98s","98o","97s","97o","96s","96o","95s","95o","94s","94o","93s","93o",
    "92s","92o",
    "88","87s","87o","86s","86o","85s","85o","84s","84o","83s","83o","82s","82o",
    "77","76s","76o","75s","75o","74s","74o","73s","73o","72s","72o",
    "66","65s","65o","64s","64o","63s","63o","62s","62o",
    "55","54s","54o","53s","53o","52s","52o",
    "44","43s","43o","42s","42o",
    "33","32s","32o",
    "22"
};
static const char *POS[] = {"SB","BB","UTG","MP","CO","BTN"};
static const char *ACT[] = {"fold","call","r0.5x","r1x","r2x","r3x"};

static void init_strength(void) {
    /* Approximate preflop equity percentages (heads-up vs random hand).
     * Sources: standard preflop equity tables. */
    float eq[] = {
        85.3, 67.0, 65.4, 66.2, 64.5, 65.4, 63.6, 65.0, 63.0, 63.4, 61.0, 62.1, 59.9,  /* AA-A8o */
        60.7, 58.4, 59.3, 57.0, 59.9, 57.4, 58.8, 56.4, 57.8, 55.4, 56.8, 54.3,        /* A7s-A2o */
        82.4, 63.4, 61.4, 63.0, 60.6, 62.6, 60.4, 60.6, 58.2, 58.4, 55.8, 57.3, 54.6,  /* KK-K7o */
        56.2, 53.3, 54.9, 52.0, 54.1, 51.3, 53.3, 50.4, 52.4, 49.4,                     /* K6s-K2o */
        79.9, 60.3, 58.2, 59.5, 57.2, 57.8, 55.0, 56.2, 53.2, 54.6, 51.4, 53.7, 50.3,  /* QQ-Q6o */
        52.3, 48.9, 51.0, 47.8, 50.1, 46.8, 49.3, 46.0,                                 /* Q5s-Q2o */
        77.5, 57.5, 55.4, 55.8, 53.3, 54.3, 51.6, 52.8, 49.8, 51.2, 47.7, 49.2, 45.8,  /* JJ-J5o */
        48.0, 44.5, 47.0, 43.4, 46.0, 42.4,                                              /* J4s-J2o */
        75.1, 54.3, 52.0, 52.6, 50.0, 50.4, 47.6, 48.9, 45.7, 47.3, 44.0, 45.3, 41.7,  /* TT-T3o */
        44.1, 40.5, 43.2, 39.6,                                                           /* T2s-T2o */
        72.1, 51.1, 48.9, 49.0, 46.4, 47.3, 44.4, 45.4, 42.2, 43.6, 40.3, 42.5, 39.0,  /* 99-93o */
        41.2, 37.7,                                                                        /* 92s-92o */
        69.1, 48.2, 45.6, 46.2, 43.3, 44.1, 41.0, 42.6, 39.2, 41.0, 37.3, 39.6, 35.8,  /* 88-82o */
        66.2, 45.7, 42.9, 43.5, 40.5, 41.7, 38.4, 39.9, 36.3, 38.6, 34.7,               /* 77-72o */
        63.3, 43.2, 40.1, 41.2, 37.8, 39.6, 36.1, 37.7, 33.9,                            /* 66-62o */
        60.3, 41.1, 37.8, 39.4, 35.7, 37.6, 34.0,                                        /* 55-52o */
        57.0, 38.5, 35.1, 36.6, 33.1,                                                     /* 44-42o */
        54.0, 36.4, 32.9,                                                                  /* 33-32o */
        50.3                                                                                /* 22 */
    };
    for (int i = 0; i < N_CLASSES; i++)
        STRENGTH[i] = eq[i] / 100.0f;
}

/* RNG */
static uint64_t g_rng;
static inline uint64_t rng_next(void) {
    g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17;
    return g_rng;
}
static inline float rng_float(void) { return (float)(rng_next() & 0xFFFFFF) / 16777216.0f; }
static inline int rng_int(int n) { return (int)(rng_next() % (uint64_t)n); }

/* Regret table: [player][class][action_sequence_hash]
 * For preflop-only, the action sequences are manageable.
 * We store regrets for the ROOT node only (first action for each position). */
static int regrets[NP][N_CLASSES][N_ACTIONS];

static void regret_match(const int *reg, float *s, int na) {
    float sum = 0;
    for (int i = 0; i < na; i++) { s[i] = reg[i] > 0 ? (float)reg[i] : 0; sum += s[i]; }
    if (sum > 0) for (int i = 0; i < na; i++) s[i] /= sum;
    else for (int i = 0; i < na; i++) s[i] = 1.0f / na;
}

static int sample_action(const float *s, int na) {
    float r = rng_float(), c = 0;
    for (int i = 0; i < na; i++) { c += s[i]; if (r <= c) return i; }
    return na - 1;
}

/* Simplified preflop game:
 * - SB posts 50, BB posts 100
 * - Acting order: UTG(2), MP(3), CO(4), BTN(5), SB(0), BB(1)
 * - Each player: fold, call(to BB=100), or raise to various sizes
 * - After all act, showdown among remaining players using equity
 * - Only one round of action (no re-raises for simplicity) */

static float simulate_hand(int traverser, int classes[NP]) {
    int active[NP]; for (int i = 0; i < NP; i++) active[i] = 1;
    int invested[NP]; memset(invested, 0, sizeof(invested));
    invested[0] = SB_AMT; invested[1] = BB_AMT;
    int pot = SB_AMT + BB_AMT;
    int current_bet = BB_AMT;

    /* Preflop acting order: UTG(2), MP(3), CO(4), BTN(5), SB(0), BB(1) */
    int order[NP] = {2, 3, 4, 5, 0, 1};

    /* Raise amounts: fold(0), call(100), r0.5x, r1x, r2x, r3x */
    /* Raise = call + fraction * (pot + call) */

    float traverser_values[N_ACTIONS];
    float traverser_strat[N_ACTIONS];
    int traverser_acted = 0;

    for (int oi = 0; oi < NP; oi++) {
        int p = order[oi];
        if (!active[p]) continue;

        int to_call = current_bet - invested[p];
        if (to_call < 0) to_call = 0;

        /* Generate actions */
        int na = 0;
        int action_amounts[N_ACTIONS];
        /* fold */
        if (to_call > 0) { action_amounts[na++] = -1; } /* fold */
        /* call */
        action_amounts[na++] = to_call;
        /* raises */
        float bet_fracs[] = {0.5f, 1.0f, 2.0f, 3.0f};
        for (int i = 0; i < 4 && na < N_ACTIONS; i++) {
            int raise_to = current_bet + (int)(bet_fracs[i] * (float)(pot + to_call));
            int amount = raise_to - invested[p];
            if (amount <= to_call) continue;
            if (amount > STACK - invested[p]) amount = STACK - invested[p];
            action_amounts[na++] = amount;
        }

        if (p == traverser) {
            /* Traverser: explore all actions */
            regret_match(regrets[p][classes[p]], traverser_strat, na);
            traverser_acted = 1;

            for (int a = 0; a < na; a++) {
                /* Simulate this action */
                int saved_active[NP], saved_invested[NP];
                memcpy(saved_active, active, sizeof(active));
                memcpy(saved_invested, invested, sizeof(invested));
                int saved_pot = pot, saved_bet = current_bet;

                if (action_amounts[a] == -1) {
                    /* fold */
                    active[p] = 0;
                } else {
                    int amt = action_amounts[a];
                    invested[p] += amt;
                    pot += amt;
                    if (invested[p] > current_bet) current_bet = invested[p];
                }

                /* Continue with remaining players (all sample) */
                int remaining_active = 0;
                for (int j = 0; j < NP; j++) if (active[j]) remaining_active++;

                float val;
                if (remaining_active <= 1 || action_amounts[a] == -1) {
                    /* If folded or only one left, compute directly */
                    if (!active[traverser]) {
                        val = -(float)invested[traverser];
                        if (invested[traverser] == 0 && action_amounts[a] == -1) val = 0;
                    } else {
                        /* Continue to let remaining opponents act, then showdown */
                        /* Simplified: just go to showdown with current state */
                        goto showdown_calc;
                    }
                } else {
                    showdown_calc:;
                    /* Let remaining opponents (after traverser) sample and act */
                    for (int oj = oi + 1; oj < NP; oj++) {
                        int pp = order[oj];
                        if (!active[pp] || pp == traverser) continue;
                        int tc = current_bet - invested[pp];
                        if (tc < 0) tc = 0;

                        /* Opponent's actions: fold or call (simplified - no re-raise) */
                        float opp_strat[2];
                        int opp_reg[2] = {0, 0}; /* No stored regrets for opponents beyond root */
                        /* Use root regrets as approximation */
                        int opp_na = tc > 0 ? 2 : 1; /* fold/call or just check */
                        regret_match(regrets[pp][classes[pp]], opp_strat, N_ACTIONS);
                        /* Map to fold/call: fold prob = strat[0], call prob = 1-strat[0] */
                        float fold_p = (tc > 0) ? opp_strat[0] : 0;
                        if (rng_float() < fold_p) {
                            active[pp] = 0;
                        } else {
                            invested[pp] = current_bet;
                            pot += tc;
                        }
                    }

                    /* Check if only one left */
                    remaining_active = 0;
                    int winner = -1;
                    for (int j = 0; j < NP; j++) if (active[j]) { remaining_active++; winner = j; }

                    if (remaining_active <= 1) {
                        if (winner == traverser)
                            val = (float)(pot - invested[traverser]);
                        else
                            val = -(float)invested[traverser];
                    } else {
                        /* Showdown: best hand wins */
                        float best = -1; int n_best = 0;
                        for (int j = 0; j < NP; j++) {
                            if (!active[j]) continue;
                            /* Add noise to equity for variance */
                            float s = STRENGTH[classes[j]] + (rng_float() - 0.5f) * 0.3f;
                            if (s > best) { best = s; n_best = 1; }
                            else if (fabsf(s - best) < 0.001f) n_best++;
                        }
                        /* Did traverser win? */
                        float ts = STRENGTH[classes[traverser]] + (rng_float() - 0.5f) * 0.3f;
                        if (ts >= best - 0.001f)
                            val = (float)pot / (float)n_best - (float)invested[traverser];
                        else
                            val = -(float)invested[traverser];
                    }
                }

                traverser_values[a] = val;

                /* Restore state */
                memcpy(active, saved_active, sizeof(active));
                memcpy(invested, saved_invested, sizeof(invested));
                pot = saved_pot;
                current_bet = saved_bet;
            }

            /* Compute node value and update regrets */
            float node_val = 0;
            for (int a = 0; a < na; a++)
                node_val += traverser_strat[a] * traverser_values[a];

            for (int a = 0; a < na; a++) {
                int delta = (int)((traverser_values[a] - node_val) * 10.0f);
                int64_t tmp = (int64_t)regrets[p][classes[p]][a] + delta;
                if (tmp > 310000000) tmp = 310000000;
                if (tmp < -310000000) tmp = -310000000;
                regrets[p][classes[p]][a] = (int)tmp;
            }

            /* After traverser acts, sample their action for downstream */
            int trav_act = sample_action(traverser_strat, na);
            if (action_amounts[trav_act] == -1) {
                active[p] = 0;
            } else {
                invested[p] += action_amounts[trav_act];
                pot += action_amounts[trav_act];
                if (invested[p] > current_bet) current_bet = invested[p];
            }

            return node_val; /* Return after traverser acts */

        } else {
            /* Non-traverser: sample one action from root strategy */
            float strat[N_ACTIONS];
            regret_match(regrets[p][classes[p]], strat, na);
            int act = sample_action(strat, na);

            if (action_amounts[act] == -1) {
                active[p] = 0;
            } else {
                int amt = action_amounts[act];
                invested[p] += amt;
                pot += amt;
                if (invested[p] > current_bet) current_bet = invested[p];
            }
        }
    }

    /* All acted, showdown */
    int remaining = 0;
    for (int i = 0; i < NP; i++) if (active[i]) remaining++;
    if (remaining <= 1) {
        for (int i = 0; i < NP; i++) {
            if (active[i]) {
                if (i == traverser) return (float)(pot - invested[traverser]);
                else return -(float)invested[traverser];
            }
        }
    }
    /* Equity showdown */
    float best = -1; int n_best = 0;
    for (int j = 0; j < NP; j++) {
        if (!active[j]) continue;
        float s = STRENGTH[classes[j]] + (rng_float() - 0.5f) * 0.3f;
        if (s > best) { best = s; n_best = 1; }
        else if (fabsf(s - best) < 0.001f) n_best++;
    }
    float ts = STRENGTH[classes[traverser]] + (rng_float() - 0.5f) * 0.3f;
    if (ts >= best - 0.001f)
        return (float)pot / (float)n_best - (float)invested[traverser];
    return -(float)invested[traverser];
}

int main(int argc, char **argv) {
    int max_iters = argc > 1 ? atoi(argv[1]) : 50000000;
    g_rng = 98765432101234567ULL;
    init_strength();
    memset(regrets, 0, sizeof(regrets));

    printf("6-player preflop-only MCCFR: %d iterations\n", max_iters);
    printf("Blinds: %d/%d, Stack: %d\n\n", SB_AMT, BB_AMT, STACK);

    for (int iter = 1; iter <= max_iters; iter++) {
        int traverser = iter % NP;
        int classes[NP];
        for (int p = 0; p < NP; p++) classes[p] = rng_int(N_CLASSES);

        simulate_hand(traverser, classes);

        if (iter % 5000000 == 0 || iter == max_iters) {
            printf("=== iter %dM ===\n", iter / 1000000);
            printf("UTG root strategies:\n");
            int show[] = {0, 1, 25, 48, 68, 84, 120, 150, 165, 166, 167, 168};
            const char *slabels[] = {"AA","AKs","KK","QQ","JJ","J3s","88","63o","33","32s","32o","22"};
            for (int si = 0; si < 12; si++) {
                int c = show[si];
                float s[N_ACTIONS];
                regret_match(regrets[2][c], s, N_ACTIONS); /* player 2 = UTG */
                printf("  %-4s: ", slabels[si]);
                for (int a = 0; a < N_ACTIONS; a++) {
                    if (s[a] >= 0.01f) printf("%s=%.0f%% ", ACT[a], s[a]*100);
                }
                printf(" (fold_reg=%d)\n", regrets[2][c][0]);
            }
            printf("\n");
        }
    }
    return 0;
}
