/**
 * cross_validate2.c — Extended cross-validation with more iterations
 */
#include "../src/solver.h"
#include "../src/hand_eval.h"
#include <stdio.h>
#include <math.h>
#include <time.h>

static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(void) {
    printf("=== Extended Cross-Validation ===\n\n");

    int board[5];
    board[0] = parse_card("Qs"); board[1] = parse_card("As");
    board[2] = parse_card("2d"); board[3] = parse_card("7h");
    board[4] = parse_card("4c");

    int hands0[4][2], hands1[4][2];
    float w0[4] = {1,1,1,1}, w1[4] = {1,1,1,1};

    hands0[0][0] = parse_card("Ah"); hands0[0][1] = parse_card("Kh");
    hands0[1][0] = parse_card("Qh"); hands0[1][1] = parse_card("Qc");
    hands0[2][0] = parse_card("Jh"); hands0[2][1] = parse_card("Th");
    hands0[3][0] = parse_card("6h"); hands0[3][1] = parse_card("5h");

    hands1[0][0] = parse_card("Ac"); hands1[0][1] = parse_card("Kc");
    hands1[1][0] = parse_card("3c"); hands1[1][1] = parse_card("3d");
    hands1[2][0] = parse_card("Tc"); hands1[2][1] = parse_card("9c");
    hands1[3][0] = parse_card("8c"); hands1[3][1] = parse_card("8d");

    float bet_sizes[] = {0.75f};
    int pot = 1000, stack = 5000;
    const char *hand_names[] = {"AhKh(TPTK)", "QhQc(trips)", "JhTh(JT)", "6h5h(air)"};

    int iter_counts[] = {1000, 5000, 10000, 50000};
    for (int t = 0; t < 4; t++) {
        Solver s;
        solver_init(&s, board, 5,
                    (const int(*)[2])hands0, w0, 4,
                    (const int(*)[2])hands1, w1, 4,
                    pot, stack, bet_sizes, 1);

        double t0 = get_time_ms();
        solver_solve(&s, iter_counts[t], 0.0f);
        double t1 = get_time_ms();

        float exploit = solver_exploitability(&s);
        printf("--- %d iterations (%.0fms) exploit=%.6f%% ---\n",
               iter_counts[t], t1-t0, exploit/pot*100);

        TreeNode *root = &s.nodes[0];
        for (int h = 0; h < 4; h++) {
            float strat[MAX_ACTIONS];
            solver_get_strategy(&s, 0, h, strat);
            float check = strat[0]*100;
            float bet = 0;
            for (int a = 1; a < root->num_actions; a++) bet += strat[a]*100;
            printf("  %s: Chk=%.1f%% Bet=%.1f%%\n", hand_names[h], check, bet);
        }

        solver_free(&s);
        printf("\n");
    }

    printf("Rust reference (1000 iter):\n");
    printf("  AhKh(TPTK): Chk=0.0%%  Bet=100.0%%\n");
    printf("  QhQc(trips): Chk=16.2%% Bet=83.8%%\n");
    printf("  JhTh(JT):   Chk=100.0%% Bet=0.0%%\n");
    printf("  6h5h(air):  Chk=21.3%% Bet=78.7%%\n");

    return 0;
}
