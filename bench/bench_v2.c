/**
 * bench_v2.c — Full benchmark of solver_v2 (Pluribus-style)
 *
 * Measures:
 * 1. River solve (direct showdown)
 * 2. Turn solve (with precomputed river strengths + leaf eval)
 * 3. Strategy sanity check
 * 4. Exploitability convergence
 * 5. Final iteration vs average strategy comparison
 */
#include "../src/solver_v2.h"
#include "../src/hand_eval.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double get_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

static int gen_hands(const int *board, int n_board, int hands[][2],
                     float *weights, int max) {
    int count = 0;
    int blocked[52] = {0};
    for (int i = 0; i < n_board; i++) blocked[board[i]] = 1;
    for (int c0 = 0; c0 < 51 && count < max; c0++) {
        if (blocked[c0]) continue;
        for (int c1 = c0+1; c1 < 52 && count < max; c1++) {
            if (blocked[c1]) continue;
            hands[count][0] = c0;
            hands[count][1] = c1;
            weights[count] = 1.0f;
            count++;
        }
    }
    return count;
}

int main(void) {
    printf("=============================================================\n");
    printf("  SOLVER V2 BENCHMARK (Pluribus-style Linear CFR)\n");
    printf("=============================================================\n\n");

    float bet_sizes[] = {0.33f, 0.75f};

    /* ── Test 1: River solve ──────────────────────────────────────── */
    printf("1. RIVER SOLVE (direct showdown)\n");
    {
        int board[5] = {parse_card("Qs"), parse_card("As"), parse_card("2d"),
                        parse_card("7h"), parse_card("4c")};

        int sizes[] = {40, 60, 80, 100};
        for (int si = 0; si < 4; si++) {
            int N = sizes[si];
            int h0[MAX_HANDS_V2][2], h1[MAX_HANDS_V2][2];
            float w0[MAX_HANDS_V2], w1[MAX_HANDS_V2];
            int n0 = gen_hands(board, 5, h0, w0, N);
            int n1 = gen_hands(board, 5, h1, w1, N);

            SolverV2 s;
            sv2_init(&s, board, 5, (const int(*)[2])h0, w0, n0,
                     (const int(*)[2])h1, w1, n1, 1000, 9000, bet_sizes, 2);

            double t0 = get_time_us();
            sv2_solve(&s, 500, 0.01f);
            double ms = (get_time_us() - t0) / 1000.0;

            float exploit = sv2_exploitability(&s);
            printf("   N=%d: %.0fms (exploit=%.4f%%)\n", N, ms, exploit/1000*100);
            sv2_free(&s);
        }
    }
    printf("\n");

    /* ── Test 2: Turn solve with precomputed river strengths ──────── */
    printf("2. TURN SOLVE (precomputed river strengths + leaf eval)\n");
    {
        int board[4] = {parse_card("Qs"), parse_card("As"), parse_card("2d"),
                        parse_card("7h")};

        int sizes[] = {40, 60, 80, 100};
        for (int si = 0; si < 4; si++) {
            int N = sizes[si];
            int h0[MAX_HANDS_V2][2], h1[MAX_HANDS_V2][2];
            float w0[MAX_HANDS_V2], w1[MAX_HANDS_V2];
            int n0 = gen_hands(board, 4, h0, w0, N);
            int n1 = gen_hands(board, 4, h1, w1, N);

            SolverV2 s;
            sv2_init(&s, board, 4, (const int(*)[2])h0, w0, n0,
                     (const int(*)[2])h1, w1, n1, 1000, 9000, bet_sizes, 2);

            /* Precompute river strengths */
            double pc_t0 = get_time_us();
            sv2_precompute_river_strengths(&s);
            double pc_ms = (get_time_us() - pc_t0) / 1000.0;

            /* Compute leaf values */
            double lv_t0 = get_time_us();
            sv2_compute_leaf_values(&s);
            double lv_ms = (get_time_us() - lv_t0) / 1000.0;

            /* Solve */
            double solve_t0 = get_time_us();
            sv2_solve(&s, 500, 0.01f);
            double solve_ms = (get_time_us() - solve_t0) / 1000.0;

            double total = pc_ms + lv_ms + solve_ms;

            printf("   N=%d: precomp=%.0fms leaf=%.0fms solve=%.0fms TOTAL=%.0fms\n",
                   N, pc_ms, lv_ms, solve_ms, total);

            /* Show tree info */
            printf("         nodes=%d leaves=%d hands=[%d,%d]\n",
                   s.num_nodes, s.num_leaves, s.num_hands[0], s.num_hands[1]);

            sv2_free(&s);
        }
    }
    printf("\n");

    /* ── Test 3: Strategy sanity ──────────────────────────────────── */
    printf("3. STRATEGY SANITY CHECK\n");
    {
        int board[5] = {parse_card("Qs"), parse_card("As"), parse_card("2d"),
                        parse_card("7h"), parse_card("4c")};
        int h0[4][2], h1[4][2];
        float w0[4] = {1,1,1,1}, w1[4] = {1,1,1,1};

        h0[0][0] = parse_card("Ah"); h0[0][1] = parse_card("Kh"); /* TPTK */
        h0[1][0] = parse_card("Qh"); h0[1][1] = parse_card("Qc"); /* trips */
        h0[2][0] = parse_card("Jh"); h0[2][1] = parse_card("Th"); /* JT */
        h0[3][0] = parse_card("6h"); h0[3][1] = parse_card("5h"); /* air */

        h1[0][0] = parse_card("Ac"); h1[0][1] = parse_card("Kc");
        h1[1][0] = parse_card("3c"); h1[1][1] = parse_card("3d");
        h1[2][0] = parse_card("Tc"); h1[2][1] = parse_card("9c");
        h1[3][0] = parse_card("8c"); h1[3][1] = parse_card("8d");

        float bs[] = {0.75f};
        SolverV2 s;
        sv2_init(&s, board, 5, (const int(*)[2])h0, w0, 4,
                 (const int(*)[2])h1, w1, 4, 1000, 5000, bs, 1);
        sv2_solve(&s, 1000, 0.0f);

        const char *names[] = {"AhKh(TPTK)", "QhQc(trips)", "JhTh(JT)", "6h5h(air)"};
        printf("   OOP final-iteration strategies:\n");
        for (int h = 0; h < 4; h++) {
            float strat[MAX_ACTIONS_V2];
            sv2_get_strategy(&s, 0, h, strat);
            float bet_total = 0;
            for (int a = 1; a < s.nodes[0].num_actions; a++) bet_total += strat[a];
            printf("   %s: check=%.0f%% bet=%.0f%%\n",
                   names[h], strat[0]*100, bet_total*100);
        }
        printf("   Expected: QQ bets most, AK bets often, 65 checks always\n");
        sv2_free(&s);
    }
    printf("\n");

    /* ── Test 4: Multi-table simulation ──────────────────────────── */
    printf("4. MULTI-TABLE SIMULATION (8 concurrent river solves)\n");
    {
        int boards[8][5] = {
            {parse_card("Qs"), parse_card("As"), parse_card("2d"), parse_card("7h"), parse_card("4c")},
            {parse_card("Kh"), parse_card("Td"), parse_card("5s"), parse_card("3c"), parse_card("8h")},
            {parse_card("Jc"), parse_card("9d"), parse_card("6h"), parse_card("2s"), parse_card("Ah")},
            {parse_card("Tc"), parse_card("8s"), parse_card("4d"), parse_card("Kd"), parse_card("3h")},
            {parse_card("Ac"), parse_card("Qd"), parse_card("7s"), parse_card("5h"), parse_card("9c")},
            {parse_card("Ks"), parse_card("Jd"), parse_card("3s"), parse_card("6c"), parse_card("2h")},
            {parse_card("Qh"), parse_card("Ts"), parse_card("8d"), parse_card("4h"), parse_card("Ac")},
            {parse_card("Ad"), parse_card("9s"), parse_card("5c"), parse_card("Jh"), parse_card("7d")},
        };

        double total_t0 = get_time_us();
        for (int t = 0; t < 8; t++) {
            int h0[MAX_HANDS_V2][2], h1[MAX_HANDS_V2][2];
            float w0[MAX_HANDS_V2], w1[MAX_HANDS_V2];
            int n0 = gen_hands(boards[t], 5, h0, w0, 60);
            int n1 = gen_hands(boards[t], 5, h1, w1, 60);

            SolverV2 s;
            sv2_init(&s, boards[t], 5, (const int(*)[2])h0, w0, n0,
                     (const int(*)[2])h1, w1, n1, 1000, 9000, bet_sizes, 2);
            sv2_solve(&s, 500, 0.01f);
            sv2_free(&s);
        }
        double total_ms = (get_time_us() - total_t0) / 1000.0;
        printf("   8 river solves sequential: %.0fms (%.0fms avg)\n",
               total_ms, total_ms / 8);
        printf("   With 8 threads: ~%.0fms wall time\n", total_ms / 8);
    }
    printf("\n");

    printf("=============================================================\n");
    printf("  BENCHMARK COMPLETE\n");
    printf("=============================================================\n");
    return 0;
}
