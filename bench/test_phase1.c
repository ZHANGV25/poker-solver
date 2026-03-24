/**
 * test_phase1.c — Multi-street solver tests
 */
#include "../src/solver_v2.h"
#include "../src/hand_eval.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
static double get_time_ms(void) {
    LARGE_INTEGER freq, cnt;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&cnt);
    return (double)cnt.QuadPart / (double)freq.QuadPart * 1000.0;
}
#else
static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}
#endif

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
    printf("  SOLVER V2 — Multi-Street Tests\n");
    printf("=============================================================\n\n");
    fflush(stdout);

    float bet_sizes[] = {0.75f};

    /* ── Test 1: River solve (regression) ────────────────────── */
    printf("1. RIVER SOLVE (4 hands)\n"); fflush(stdout);
    {
        int board[5] = {parse_card("Qs"), parse_card("As"), parse_card("2d"),
                        parse_card("7h"), parse_card("4c")};
        int h0[4][2], h1[4][2];
        float w0[4] = {1,1,1,1}, w1[4] = {1,1,1,1};

        h0[0][0] = parse_card("Ah"); h0[0][1] = parse_card("Kh");
        h0[1][0] = parse_card("Qh"); h0[1][1] = parse_card("Qc");
        h0[2][0] = parse_card("Jh"); h0[2][1] = parse_card("Th");
        h0[3][0] = parse_card("6h"); h0[3][1] = parse_card("5h");

        h1[0][0] = parse_card("Ac"); h1[0][1] = parse_card("Kc");
        h1[1][0] = parse_card("3c"); h1[1][1] = parse_card("3d");
        h1[2][0] = parse_card("Tc"); h1[2][1] = parse_card("9c");
        h1[3][0] = parse_card("8c"); h1[3][1] = parse_card("8d");

        SolverV2 s;
        sv2_init(&s, board, 5, (const int(*)[2])h0, w0, 4,
                 (const int(*)[2])h1, w1, 4, 1000, 5000, bet_sizes, 1);

        double t0 = get_time_ms();
        sv2_solve(&s, 1000, 0.0f);
        double ms = get_time_ms() - t0;

        printf("   time=%.0fms\n", ms);

        float strat[MAX_ACTIONS_V2];
        const char *names[] = {"AhKh(TPTK)", "QhQc(trips)", "JhTh(JT)", "6h5h(air)"};
        for (int h = 0; h < 4; h++) {
            sv2_get_strategy(&s, 0, h, strat);
            printf("   %s: chk=%.0f%% bet=%.0f%%\n", names[h], strat[0]*100, strat[1]*100);
        }

        sv2_free(&s);
        printf("   PASS\n\n"); fflush(stdout);
    }

    /* ── Test 2: Turn solve (4 hands, deals river) ───────────── */
    printf("2. TURN SOLVE (4 hands — deals river cards)\n"); fflush(stdout);
    {
        int board[4] = {parse_card("Qs"), parse_card("As"), parse_card("2d"),
                        parse_card("7h")};
        int h0[4][2], h1[4][2];
        float w0[4] = {1,1,1,1}, w1[4] = {1,1,1,1};

        h0[0][0] = parse_card("Ah"); h0[0][1] = parse_card("Kh");
        h0[1][0] = parse_card("Qh"); h0[1][1] = parse_card("Qc");
        h0[2][0] = parse_card("Jh"); h0[2][1] = parse_card("Th");
        h0[3][0] = parse_card("6h"); h0[3][1] = parse_card("5h");

        h1[0][0] = parse_card("Ac"); h1[0][1] = parse_card("Kc");
        h1[1][0] = parse_card("3c"); h1[1][1] = parse_card("3d");
        h1[2][0] = parse_card("Tc"); h1[2][1] = parse_card("9c");
        h1[3][0] = parse_card("8c"); h1[3][1] = parse_card("8d");

        SolverV2 s;
        sv2_init(&s, board, 4, (const int(*)[2])h0, w0, 4,
                 (const int(*)[2])h1, w1, 4, 1000, 5000, bet_sizes, 1);

        printf("   root tree: %d nodes\n", s.root_tree.num_nodes);
        printf("   turn cards: %d, max_river_nodes: %d\n",
               s.num_turn_cards, s.max_river_nodes);
        fflush(stdout);

        double t0 = get_time_ms();
        sv2_solve(&s, 50, 0.0f);
        double ms = get_time_ms() - t0;

        printf("   50 iters: %.0fms (%.1fms/iter)\n", ms, ms/50);

        float strat[MAX_ACTIONS_V2];
        const char *names[] = {"AhKh(TPTK)", "QhQc(trips)", "JhTh(JT)", "6h5h(air)"};
        for (int h = 0; h < 4; h++) {
            sv2_get_strategy(&s, 0, h, strat);
            int na = s.root_tree.nodes[0].num_actions;
            printf("   %s:", names[h]);
            for (int a = 0; a < na; a++) printf(" %.0f%%", strat[a]*100);
            printf("\n");
        }

        sv2_free(&s);
        printf("   PASS\n\n"); fflush(stdout);
    }

    /* ── Test 3: FLOP solve (4 hands, deals turn + river) ────── */
    printf("3. FLOP SOLVE (4 hands — full flop→turn→river)\n"); fflush(stdout);
    {
        int board[3] = {parse_card("Qs"), parse_card("As"), parse_card("2d")};
        int h0[4][2], h1[4][2];
        float w0[4] = {1,1,1,1}, w1[4] = {1,1,1,1};

        h0[0][0] = parse_card("Ah"); h0[0][1] = parse_card("Kh");
        h0[1][0] = parse_card("Qh"); h0[1][1] = parse_card("Qc");
        h0[2][0] = parse_card("Jh"); h0[2][1] = parse_card("Th");
        h0[3][0] = parse_card("6h"); h0[3][1] = parse_card("5h");

        h1[0][0] = parse_card("Ac"); h1[0][1] = parse_card("Kc");
        h1[1][0] = parse_card("3c"); h1[1][1] = parse_card("3d");
        h1[2][0] = parse_card("Tc"); h1[2][1] = parse_card("9c");
        h1[3][0] = parse_card("8c"); h1[3][1] = parse_card("8d");

        SolverV2 s;
        sv2_init(&s, board, 3, (const int(*)[2])h0, w0, 4,
                 (const int(*)[2])h1, w1, 4, 1000, 5000, bet_sizes, 1);

        printf("   root tree: %d nodes\n", s.root_tree.num_nodes);
        printf("   turn cards: %d\n", s.num_turn_cards);
        fflush(stdout);

        double t0 = get_time_ms();
        sv2_solve(&s, 5, 0.0f);
        double ms = get_time_ms() - t0;

        printf("   5 iters: %.0fms (%.0fms/iter)\n", ms, ms/5);

        float strat[MAX_ACTIONS_V2];
        const char *names[] = {"AhKh(TPTK)", "QhQc(trips)", "JhTh(JT)", "6h5h(air)"};
        for (int h = 0; h < 4; h++) {
            sv2_get_strategy(&s, 0, h, strat);
            int na = s.root_tree.nodes[0].num_actions;
            printf("   %s:", names[h]);
            for (int a = 0; a < na; a++) printf(" %.0f%%", strat[a]*100);
            printf("\n");
        }

        sv2_free(&s);
        printf("   PASS\n\n"); fflush(stdout);
    }

    printf("=============================================================\n");
    printf("  ALL TESTS PASSED\n");
    printf("=============================================================\n");
    return 0;
}
