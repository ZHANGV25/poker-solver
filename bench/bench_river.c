/**
 * bench_river.c — Benchmark: river-only DCFR solve
 *
 * Tests the solver on a concrete river scenario:
 *   Board: Qs As 2d 7h 4c
 *   OOP: CO open range (narrowed)
 *   IP: BB defend range (narrowed)
 *   Pot: 6.5 BB, Effective stack: 90 BB
 *   Bet sizes: 33%, 75%, all-in
 *
 * Reports: tree size, iterations/sec, convergence, sample strategies.
 */

#include "../src/solver.h"
#include "../src/hand_eval.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ── Range generation ──────────────────────────────────────────────────── */

/** Generate all possible 2-card hands not blocked by board */
static int generate_all_hands(const int *board, int n_board,
                              int hands[][2], float *weights) {
    int count = 0;
    for (int c0 = 0; c0 < 51; c0++) {
        int blocked0 = 0;
        for (int b = 0; b < n_board; b++)
            if (c0 == board[b]) { blocked0 = 1; break; }
        if (blocked0) continue;

        for (int c1 = c0 + 1; c1 < 52; c1++) {
            int blocked1 = 0;
            for (int b = 0; b < n_board; b++)
                if (c1 == board[b]) { blocked1 = 1; break; }
            if (blocked1) continue;

            hands[count][0] = c0;
            hands[count][1] = c1;
            weights[count] = 1.0f;
            count++;
        }
    }
    return count;
}

/** Generate a narrowed range (simulate ~80 hands from a typical range) */
static int generate_narrowed_range(const int *board, int n_board,
                                   int hands[][2], float *weights,
                                   int target_count) {
    /* Generate all hands first, then take every Nth to simulate narrowing */
    int all_hands[MAX_HANDS][2];
    float all_weights[MAX_HANDS];
    int total = generate_all_hands(board, n_board, all_hands, all_weights);

    int step = total / target_count;
    if (step < 1) step = 1;

    int count = 0;
    for (int i = 0; i < total && count < target_count; i += step) {
        hands[count][0] = all_hands[i][0];
        hands[count][1] = all_hands[i][1];
        weights[count] = 1.0f;
        count++;
    }
    return count;
}

/* ── Timer ─────────────────────────────────────────────────────────────── */

static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* ── Main ──────────────────────────────────────────────────────────────── */

int main(void) {
    printf("=== Poker Solver Benchmark: River DCFR ===\n\n");

    /* Board: Qs As 2d 7h 4c */
    int board[5];
    board[0] = parse_card("Qs");
    board[1] = parse_card("As");
    board[2] = parse_card("2d");
    board[3] = parse_card("7h");
    board[4] = parse_card("4c");

    printf("Board: Qs As 2d 7h 4c\n");

    /* Test with different range sizes */
    int range_sizes[] = {30, 60, 100, 200, 500, 1000};
    int n_tests = sizeof(range_sizes) / sizeof(range_sizes[0]);

    float bet_sizes[] = {0.33f, 0.75f}; /* 33% and 75% pot + all-in is added automatically */
    int n_bets = 2;

    int pot = 650;   /* 6.5 BB * 100 scale */
    int stack = 9000; /* 90 BB * 100 scale */

    for (int t = 0; t < n_tests; t++) {
        int target = range_sizes[t];

        /* Generate ranges */
        int hands0[MAX_HANDS][2], hands1[MAX_HANDS][2];
        float weights0[MAX_HANDS], weights1[MAX_HANDS];
        int n0 = generate_narrowed_range(board, 5, hands0, weights0, target);
        int n1 = generate_narrowed_range(board, 5, hands1, weights1, target);

        printf("\n--- Range size: %d (OOP=%d, IP=%d hands) ---\n", target, n0, n1);

        /* Init solver */
        Solver s;
        int err = solver_init(&s, board, 5,
                              (const int(*)[2])hands0, weights0, n0,
                              (const int(*)[2])hands1, weights1, n1,
                              pot, stack, bet_sizes, n_bets);
        if (err) {
            printf("Error initializing solver\n");
            continue;
        }

        printf("Tree nodes: %d\n", s.num_nodes);
        printf("Hands: OOP=%d, IP=%d\n", s.num_hands[0], s.num_hands[1]);

        /* Benchmark: 100 iterations */
        int iters = 100;
        double t0 = get_time_ms();
        solver_solve(&s, iters, 0.01f);
        double t1 = get_time_ms();

        double elapsed_ms = t1 - t0;
        double iters_per_sec = iters / (elapsed_ms / 1000.0);

        printf("100 iterations: %.1f ms (%.0f iter/sec)\n", elapsed_ms, iters_per_sec);

        /* Project time for different iteration counts */
        printf("Projected: 200 iter = %.1f ms, 500 iter = %.1f ms, 1000 iter = %.1f ms\n",
               elapsed_ms * 2, elapsed_ms * 5, elapsed_ms * 10);

        /* Show sample strategy for first hand */
        float strat[MAX_ACTIONS];
        solver_get_strategy(&s, 0, 0, strat);
        printf("OOP hand 0 strategy: ");
        /* Map action indices to labels */
        TreeNode *root = &s.nodes[0];
        for (int a = 0; a < root->num_actions; a++) {
            TreeNode *child = &s.nodes[root->children[a]];
            const char *label;
            if (child->type == NODE_FOLD) label = "Fold";
            else if (child->type == NODE_SHOWDOWN || child->type == NODE_LEAF)
                label = "Check/Call";
            else {
                /* It's a bet/raise subtree */
                label = "Bet";
            }
            printf("%s=%.1f%% ", label, strat[a] * 100);
        }
        printf("\n");

        /* Now solve with more iterations and time it */
        solver_free(&s);
        solver_init(&s, board, 5,
                    (const int(*)[2])hands0, weights0, n0,
                    (const int(*)[2])hands1, weights1, n1,
                    pot, stack, bet_sizes, n_bets);

        t0 = get_time_ms();
        solver_solve(&s, 500, 0.01f);
        t1 = get_time_ms();
        printf("500 iterations: %.1f ms\n", t1 - t0);

        solver_free(&s);
    }

    printf("\n=== Benchmark complete ===\n");
    return 0;
}
