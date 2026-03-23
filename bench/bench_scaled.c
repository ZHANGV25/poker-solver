/**
 * bench_scaled.c — Benchmark with realistic narrowed range sizes
 *
 * Tests convergence and timing for range sizes typical of
 * Bayesian-narrowed ranges after flop+turn actions.
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
        int blocked0 = 0;
        for (int b = 0; b < n_board; b++)
            if (c0 == board[b]) { blocked0 = 1; break; }
        if (blocked0) continue;

        for (int c1 = c0 + 1; c1 < 52 && count < max_count; c1++) {
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

int main(void) {
    printf("=== Scaled Benchmark: River DCFR ===\n\n");

    int board[5];
    board[0] = parse_card("Qs");
    board[1] = parse_card("As");
    board[2] = parse_card("2d");
    board[3] = parse_card("7h");
    board[4] = parse_card("4c");

    float bet_sizes[] = {0.33f, 0.75f};
    int pot = 650;
    int stack = 9000;

    int test_sizes[] = {10, 20, 40, 60, 80, 100};
    int n_tests = 6;

    for (int t = 0; t < n_tests; t++) {
        int target = test_sizes[t];

        int hands0[MAX_HANDS][2], hands1[MAX_HANDS][2];
        float w0[MAX_HANDS], w1[MAX_HANDS];
        int n0 = generate_hands(board, 5, hands0, w0, target);
        int n1 = generate_hands(board, 5, hands1, w1, target);

        printf("--- %d hands per player ---\n", target);

        Solver s;
        solver_init(&s, board, 5,
                    (const int(*)[2])hands0, w0, n0,
                    (const int(*)[2])hands1, w1, n1,
                    pot, stack, bet_sizes, 2);

        printf("  Tree: %d nodes, Hands: OOP=%d IP=%d\n",
               s.num_nodes, s.num_hands[0], s.num_hands[1]);

        /* Benchmark iterations */
        int iter_counts[] = {100, 200, 500};
        for (int i = 0; i < 3; i++) {
            solver_free(&s);
            solver_init(&s, board, 5,
                        (const int(*)[2])hands0, w0, n0,
                        (const int(*)[2])hands1, w1, n1,
                        pot, stack, bet_sizes, 2);

            double t0 = get_time_ms();
            solver_solve(&s, iter_counts[i], 0.01f);
            double t1 = get_time_ms();

            printf("  %d iter: %.0f ms (%.0f iter/sec)\n",
                   iter_counts[i], t1 - t0,
                   iter_counts[i] / ((t1 - t0) / 1000.0));
        }

        solver_free(&s);
        printf("\n");
    }

    printf("=== Done ===\n");
    return 0;
}
