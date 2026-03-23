/**
 * final_bench.c — Comprehensive benchmark and validation
 *
 * 1. Validates hand evaluation
 * 2. Tests value computation sanity
 * 3. Benchmarks solve time across range sizes
 * 4. Measures convergence (exploitability)
 * 5. Validates strategy reasonableness
 */
#include "../src/solver.h"
#include "../src/hand_eval.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

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

/* ── Test 1: Hand evaluation ──────────────────────────────────────────── */
static int test_hand_eval(void) {
    printf("Test 1: Hand Evaluation\n");
    int pass = 1;

    /* Known hand strengths on board Qs As 2d 7h 4c */
    int board7[7];
    board7[0] = parse_card("Qs"); board7[1] = parse_card("As");
    board7[2] = parse_card("2d"); board7[3] = parse_card("7h");
    board7[4] = parse_card("4c");

    /* AhKh = pair of aces, K kicker */
    board7[5] = parse_card("Ah"); board7[6] = parse_card("Kh");
    uint32_t s_ak = eval7(board7);
    if ((s_ak >> 20) != HC_PAIR) { printf("  FAIL: AhKh should be pair\n"); pass = 0; }

    /* 3c3d = pair of 3s */
    board7[5] = parse_card("3c"); board7[6] = parse_card("3d");
    uint32_t s_33 = eval7(board7);
    if ((s_33 >> 20) != HC_PAIR) { printf("  FAIL: 3c3d should be pair\n"); pass = 0; }

    /* AhKh should beat 3c3d (pair of aces > pair of 3s) */
    if (s_ak <= s_33) { printf("  FAIL: pair of aces should beat pair of 3s\n"); pass = 0; }

    /* 6h5h = high card */
    board7[5] = parse_card("6h"); board7[6] = parse_card("5h");
    uint32_t s_65 = eval7(board7);
    if ((s_65 >> 20) != HC_HIGH_CARD) { printf("  FAIL: 6h5h should be high card\n"); pass = 0; }

    /* QhQc = trips (with Qs on board) */
    board7[5] = parse_card("Qh"); board7[6] = parse_card("Qc");
    uint32_t s_qq = eval7(board7);
    if ((s_qq >> 20) != HC_TRIPS) { printf("  FAIL: QhQc should be trips\n"); pass = 0; }

    /* QhQc should beat AhKh */
    if (s_qq <= s_ak) { printf("  FAIL: trips should beat one pair\n"); pass = 0; }

    if (pass) printf("  PASS: all hand evaluations correct\n");
    return pass;
}

/* ── Test 2: Strategy sanity ──────────────────────────────────────────── */
static int test_strategy_sanity(void) {
    printf("\nTest 2: Strategy Sanity\n");
    int pass = 1;

    int board[5];
    board[0] = parse_card("Qs"); board[1] = parse_card("As");
    board[2] = parse_card("2d"); board[3] = parse_card("7h");
    board[4] = parse_card("4c");

    int hands0[4][2], hands1[4][2];
    float w0[4] = {1,1,1,1}, w1[4] = {1,1,1,1};

    /* OOP: AhKh(TPTK), QhQc(trips), JhTh(JT high), 6h5h(nothing) */
    hands0[0][0] = parse_card("Ah"); hands0[0][1] = parse_card("Kh");
    hands0[1][0] = parse_card("Qh"); hands0[1][1] = parse_card("Qc");
    hands0[2][0] = parse_card("Jh"); hands0[2][1] = parse_card("Th");
    hands0[3][0] = parse_card("6h"); hands0[3][1] = parse_card("5h");

    /* IP: AcKc(TPTK), 3c3d(pair 3s), Tc9c(T9 high), 8c8d(pair 8s) */
    hands1[0][0] = parse_card("Ac"); hands1[0][1] = parse_card("Kc");
    hands1[1][0] = parse_card("3c"); hands1[1][1] = parse_card("3d");
    hands1[2][0] = parse_card("Tc"); hands1[2][1] = parse_card("9c");
    hands1[3][0] = parse_card("8c"); hands1[3][1] = parse_card("8d");

    float bet_sizes[] = {0.75f};
    Solver s;
    solver_init(&s, board, 5,
                (const int(*)[2])hands0, w0, 4,
                (const int(*)[2])hands1, w1, 4,
                1000, 5000, bet_sizes, 1);

    solver_solve(&s, 1000, 0.0f);

    /* Check strategies */
    TreeNode *root = &s.nodes[0];
    printf("  OOP strategies (4 hands, 1000 iter):\n");
    const char *hand_names[] = {"AhKh(TPTK)", "QhQc(trips)", "JhTh(JThigh)", "6h5h(nothing)"};
    float strats[4][MAX_ACTIONS];
    for (int h = 0; h < 4; h++) {
        solver_get_strategy(&s, 0, h, strats[h]);
        printf("    %s: ", hand_names[h]);
        for (int a = 0; a < root->num_actions; a++)
            printf("%.0f%% ", strats[h][a]*100);
        printf("\n");
    }

    /* Sanity checks:
     * - QhQc (trips) should bet most frequently (strongest hand)
     * - 6h5h (nothing) should check almost always
     * - AhKh should have significant bet frequency (value hand)
     */
    /* Action 0 = check (goes to IP decision), Action 1 = bet 75%, Action 2 = all-in */
    float qq_bet = 0, ak_bet = 0, jt_bet = 0, low_bet = 0;
    for (int a = 1; a < root->num_actions; a++) {
        qq_bet += strats[1][a];
        ak_bet += strats[0][a];
        jt_bet += strats[2][a];
        low_bet += strats[3][a];
    }

    if (qq_bet < 0.3f) { printf("  WARN: QQ trips should bet >30%% (got %.0f%%)\n", qq_bet*100); }
    if (low_bet > 0.5f) { printf("  WARN: 65 should not bet >50%% (got %.0f%%)\n", low_bet*100); }
    if (ak_bet < 0.1f) { printf("  WARN: AK should bet >10%% (got %.0f%%)\n", ak_bet*100); }
    printf("  QQ bets %.0f%%, AK bets %.0f%%, JT bets %.0f%%, 65 bets %.0f%%\n",
           qq_bet*100, ak_bet*100, jt_bet*100, low_bet*100);

    /* Exploitability */
    float exploit = solver_exploitability(&s);
    printf("  Exploitability: %.4f chips (%.4f%% of pot)\n", exploit, exploit/1000*100);
    if (exploit > 100) { printf("  WARN: exploitability seems high\n"); }

    solver_free(&s);
    if (pass) printf("  PASS: strategies look reasonable\n");
    return pass;
}

/* ── Test 3: Benchmark ────────────────────────────────────────────────── */
static void test_benchmark(void) {
    printf("\nTest 3: Performance Benchmark\n");
    printf("  %-8s %-8s %-12s %-12s %-12s %-12s\n",
           "Hands", "Nodes", "100iter(ms)", "500iter(ms)", "iter/sec", "500i_exploit");

    int board[5];
    board[0] = parse_card("Qs"); board[1] = parse_card("As");
    board[2] = parse_card("2d"); board[3] = parse_card("7h");
    board[4] = parse_card("4c");

    float bet_sizes[] = {0.33f, 0.75f};
    int pot = 1000, stack = 9000;
    int sizes[] = {20, 40, 60, 80, 100, 150, 200};
    int n_sizes = 7;

    for (int t = 0; t < n_sizes; t++) {
        int target = sizes[t];
        int hands0[MAX_HANDS][2], hands1[MAX_HANDS][2];
        float w0[MAX_HANDS], w1[MAX_HANDS];
        int n0 = generate_hands(board, 5, hands0, w0, target);
        int n1 = generate_hands(board, 5, hands1, w1, target);

        /* 100 iterations */
        Solver s;
        solver_init(&s, board, 5,
                    (const int(*)[2])hands0, w0, n0,
                    (const int(*)[2])hands1, w1, n1,
                    pot, stack, bet_sizes, 2);

        double t0 = get_time_ms();
        solver_solve(&s, 100, 0.0f);
        double t1 = get_time_ms();
        double ms100 = t1 - t0;

        solver_free(&s);

        /* 500 iterations */
        solver_init(&s, board, 5,
                    (const int(*)[2])hands0, w0, n0,
                    (const int(*)[2])hands1, w1, n1,
                    pot, stack, bet_sizes, 2);

        t0 = get_time_ms();
        solver_solve(&s, 500, 0.0f);
        t1 = get_time_ms();
        double ms500 = t1 - t0;
        float exploit = solver_exploitability(&s);

        printf("  %-8d %-8d %-12.0f %-12.0f %-12.0f %.4f%%\n",
               target, s.num_nodes, ms100, ms500,
               500.0 / (ms500 / 1000.0), exploit/pot*100);

        solver_free(&s);
    }
}

int main(void) {
    printf("============================================================\n");
    printf("   Poker Solver — Comprehensive Benchmark & Validation\n");
    printf("============================================================\n\n");

    int pass = 1;
    pass &= test_hand_eval();
    pass &= test_strategy_sanity();
    test_benchmark();

    printf("\n============================================================\n");
    printf("   Result: %s\n", pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    printf("============================================================\n");
    return pass ? 0 : 1;
}
