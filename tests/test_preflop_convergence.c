/*
 * Minimal 2-player preflop-only MCCFR to verify convergence.
 * Tests whether 32o converges to fold and AA converges to raise.
 * No postflop, no board cards — pure preflop decision.
 *
 * Build: gcc -O2 -o test_preflop tests/test_preflop_convergence.c -lm
 * Run:   ./test_preflop [iterations]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

/* Preflop hand equity table (approximate heads-up equity for each of 169 classes).
 * Index matches ca_preflop_classes: 0=AA, 1=AKs, ..., 165=33, 166=32s, 167=32o, 168=22 */
static float EQUITY[169]; /* computed at init */

/* RNG */
static uint64_t rng_state = 12345678901234567ULL;
static inline uint64_t rng_next(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}
static inline float rng_float(void) {
    return (float)(rng_next() & 0xFFFFFF) / (float)0x1000000;
}
static inline int rng_int(int n) { return (int)(rng_next() % (uint64_t)n); }

/* Approximate equity for 169 classes (heads-up, rough values) */
static void init_equity(void) {
    /* Pairs: AA=85%, KK=82%, ..., 22=50% */
    float pair_eq[] = {0.85,0.82,0.80,0.77,0.75,0.72,0.69,0.66,0.63,0.60,0.57,0.54,0.50};
    /* AA=0, KK=25, QQ=48, JJ=68, TT=86, 99=102, 88=116, 77=128, 66=138, 55=146, 44=152, 33=165, 22=168 */
    int pair_idx[] = {0,25,48,68,86,102,116,128,138,146,152,165,168};
    for (int i = 0; i < 13; i++) EQUITY[pair_idx[i]] = pair_eq[i];

    /* Non-pairs: rough equity based on high card + connectedness */
    for (int i = 0; i < 169; i++) {
        if (EQUITY[i] > 0) continue; /* already set (pair) */
        /* Rough heuristic: higher index = weaker hand */
        EQUITY[i] = 0.55f - (float)i * 0.002f;
        if (EQUITY[i] < 0.30f) EQUITY[i] = 0.30f;
        if (EQUITY[i] > 0.65f) EQUITY[i] = 0.65f;
    }
    /* Specific overrides for hands we care about */
    EQUITY[1] = 0.67f;   /* AKs */
    EQUITY[2] = 0.65f;   /* AKo */
    EQUITY[84] = 0.36f;  /* J3s */
    EQUITY[150] = 0.33f; /* 63o */
    EQUITY[166] = 0.32f; /* 32s */
    EQUITY[167] = 0.31f; /* 32o */
}

/* Game: 2 players, SB=50, BB=100, stack=10000
 * Actions: 0=fold, 1=call(100), 2=raise(300), 3=raise(600) */
#define N_ACTIONS 4
#define N_CLASSES 169
#define SB 50
#define BB 100

/* Regret tables: [player][class][action] */
static int regrets[2][N_CLASSES][N_ACTIONS];

static void regret_match(const int *reg, float *strat, int na) {
    float sum = 0;
    for (int i = 0; i < na; i++) {
        strat[i] = reg[i] > 0 ? (float)reg[i] : 0;
        sum += strat[i];
    }
    if (sum > 0) for (int i = 0; i < na; i++) strat[i] /= sum;
    else for (int i = 0; i < na; i++) strat[i] = 1.0f / na;
}

static int sample_action(const float *strat, int na) {
    float r = rng_float();
    float cum = 0;
    for (int i = 0; i < na; i++) {
        cum += strat[i];
        if (r <= cum) return i;
    }
    return na - 1;
}

/* Simulate one hand with given actions, return payoff for player 0 */
static float play_hand(int class0, int class1, int act0, int act1) {
    float eq0 = EQUITY[class0];
    /* Player 0 is SB (posts 50), player 1 is BB (posts 100) */

    if (act0 == 0) return -SB; /* SB folds, loses SB */

    /* SB calls (act0=1) or raises */
    int invest0, invest1;
    if (act0 == 1) {
        /* SB limps to 100 */
        invest0 = BB; /* call to 100 */
        /* BB can check (always, since limp) */
        invest1 = BB;
    } else if (act0 == 2) {
        /* SB raises to 300 */
        invest0 = 300;
        if (act1 == 0) return SB + BB; /* BB folds, SB wins BB */
        invest1 = 300; /* BB calls */
    } else {
        /* SB raises to 600 */
        invest0 = 600;
        if (act1 == 0) return SB + BB;
        invest1 = 600;
    }

    /* Showdown: use equity to determine winner */
    float pot = (float)(invest0 + invest1);
    if (rng_float() < eq0)
        return pot - (float)invest0; /* win */
    else
        return -(float)invest0; /* lose */
}

int main(int argc, char **argv) {
    int max_iters = argc > 1 ? atoi(argv[1]) : 5000000;
    init_equity();
    memset(regrets, 0, sizeof(regrets));

    printf("2-player preflop MCCFR: %d iterations\n\n", max_iters);

    for (int iter = 1; iter <= max_iters; iter++) {
        int traverser = iter % 2;
        int opp = 1 - traverser;

        /* Sample hands */
        int class_t = rng_int(N_CLASSES);
        int class_o = rng_int(N_CLASSES);

        /* Opponent strategy */
        float opp_strat[N_ACTIONS];
        regret_match(regrets[opp][class_o], opp_strat, N_ACTIONS);
        int opp_act = sample_action(opp_strat, N_ACTIONS);

        /* Traverser explores all actions */
        float strat[N_ACTIONS];
        regret_match(regrets[traverser][class_t], strat, N_ACTIONS);

        float values[N_ACTIONS];
        float node_value = 0;

        for (int a = 0; a < N_ACTIONS; a++) {
            float v;
            if (traverser == 0) {
                v = play_hand(class_t, class_o, a, opp_act);
            } else {
                /* Player 1 is BB. Swap perspective. */
                /* When BB "folds", they check (or fold to a raise) */
                v = -play_hand(class_o, class_t, opp_act, a);
            }
            values[a] = v;
            node_value += strat[a] * v;
        }

        /* Update regrets (with ceiling) */
        for (int a = 0; a < N_ACTIONS; a++) {
            int delta = (int)((values[a] - node_value) * 10.0f);
            int64_t tmp = (int64_t)regrets[traverser][class_t][a] + delta;
            if (tmp > 310000000) tmp = 310000000;
            if (tmp < -310000000) tmp = -310000000;
            regrets[traverser][class_t][a] = (int)tmp;
        }

        /* Progress */
        if (iter % 1000000 == 0 || iter == max_iters) {
            printf("iter %dM:\n", iter / 1000000);
            int show[] = {0, 25, 48, 150, 165, 166, 167, 168};
            const char *labels[] = {"AA","KK","QQ","63o","33","32s","32o","22"};
            for (int si = 0; si < 8; si++) {
                int c = show[si];
                float s[N_ACTIONS];
                regret_match(regrets[0][c], s, N_ACTIONS);
                printf("  SB %-4s: fold=%.2f call=%.2f r300=%.2f r600=%.2f  (reg: %d %d %d %d)\n",
                       labels[si], s[0], s[1], s[2], s[3],
                       regrets[0][c][0], regrets[0][c][1],
                       regrets[0][c][2], regrets[0][c][3]);
            }
            printf("\n");
        }
    }

    return 0;
}
