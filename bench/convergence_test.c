/**
 * convergence_test.c — Measure exploitability convergence across iterations
 *
 * Tests with realistic narrowed ranges (50-80 hands) and measures
 * how quickly exploitability decreases.
 */
#include "../src/solver.h"
#include "../src/hand_eval.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static int generate_hands(const int *board, int n_board,
                          int hands[][2], float *weights, int max_count) {
    int count = 0;
    for (int c0 = 0; c0 < 51 && count < max_count; c0++) {
        int blocked = 0;
        for (int b = 0; b < n_board; b++)
            if (c0 == board[b]) { blocked = 1; break; }
        if (blocked) continue;
        for (int c1 = c0 + 1; c1 < 52 && count < max_count; c1++) {
            int blocked2 = 0;
            for (int b = 0; b < n_board; b++)
                if (c1 == board[b]) { blocked2 = 1; break; }
            if (blocked2) continue;
            hands[count][0] = c0;
            hands[count][1] = c1;
            weights[count] = 1.0f;
            count++;
        }
    }
    return count;
}

int main(void) {
    printf("=== Convergence Test: DCFR River Solver ===\n\n");

    int board[5];
    board[0] = parse_card("Qs");
    board[1] = parse_card("As");
    board[2] = parse_card("2d");
    board[3] = parse_card("7h");
    board[4] = parse_card("4c");

    float bet_sizes[] = {0.33f, 0.75f};
    int pot = 1000;   /* 10 BB * 100 */
    int stack = 9000; /* 90 BB * 100 */

    int range_sizes[] = {20, 40, 60, 80};
    int n_range_tests = 4;

    for (int r = 0; r < n_range_tests; r++) {
        int target = range_sizes[r];

        int hands0[MAX_HANDS][2], hands1[MAX_HANDS][2];
        float w0[MAX_HANDS], w1[MAX_HANDS];
        int n0 = generate_hands(board, 5, hands0, w0, target);
        int n1 = generate_hands(board, 5, hands1, w1, target);

        printf("--- %d hands per player ---\n", target);

        int iter_points[] = {10, 25, 50, 100, 200, 500, 1000};
        int n_points = 7;
        int prev_iters = 0;

        Solver s;
        solver_init(&s, board, 5,
                    (const int(*)[2])hands0, w0, n0,
                    (const int(*)[2])hands1, w1, n1,
                    pot, stack, bet_sizes, 2);

        printf("  Tree: %d nodes, Hands: OOP=%d IP=%d\n\n", s.num_nodes, s.num_hands[0], s.num_hands[1]);
        printf("  %8s  %10s  %12s  %12s\n", "Iters", "Time(ms)", "Exploit", "Exploit/Pot");

        for (int p = 0; p < n_points; p++) {
            int target_iters = iter_points[p];
            int delta = target_iters - prev_iters;

            double t0 = get_time_ms();
            solver_solve(&s, delta, 0.0001f);
            double t1 = get_time_ms();

            /* Compute exploitability */
            float exploit = solver_exploitability(&s);
            float exploit_pct = exploit / (float)pot * 100.0f;

            printf("  %8d  %10.1f  %12.1f  %10.2f%%\n",
                   target_iters, t1 - t0, exploit, exploit_pct);

            prev_iters = target_iters;
        }

        /* Show sample strategies at convergence */
        printf("\n  Sample OOP strategies (after 1000 iter):\n");
        TreeNode *root = &s.nodes[0];
        for (int h = 0; h < s.num_hands[0] && h < 10; h++) {
            char c0[3], c1[3];
            format_card(s.hands[0][h][0], c0);
            format_card(s.hands[0][h][1], c1);
            float strat[MAX_ACTIONS];
            solver_get_strategy(&s, 0, h, strat);
            printf("    %s%s: ", c0, c1);
            for (int a = 0; a < root->num_actions; a++)
                printf("%.0f%% ", strat[a] * 100);
            printf("\n");
        }

        solver_free(&s);
        printf("\n");
    }

    printf("=== Done ===\n");
    return 0;
}
