/**
 * quick_test.c — Minimal test of the solver
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

int main(void) {
    printf("=== Quick Solver Test ===\n\n");

    /* Board: Qs As 2d 7h 4c (river) */
    int board[5];
    board[0] = parse_card("Qs");
    board[1] = parse_card("As");
    board[2] = parse_card("2d");
    board[3] = parse_card("7h");
    board[4] = parse_card("4c");

    /* Tiny range: just 5 hands each */
    int hands0[5][2], hands1[5][2];
    float w0[5], w1[5];

    /* OOP hands */
    hands0[0][0] = parse_card("Ah"); hands0[0][1] = parse_card("Kh"); w0[0] = 1.0f;
    hands0[1][0] = parse_card("Kd"); hands0[1][1] = parse_card("Qd"); w0[1] = 1.0f;
    hands0[2][0] = parse_card("Jh"); hands0[2][1] = parse_card("Th"); w0[2] = 1.0f;
    hands0[3][0] = parse_card("9d"); hands0[3][1] = parse_card("8d"); w0[3] = 1.0f;
    hands0[4][0] = parse_card("6h"); hands0[4][1] = parse_card("5h"); w0[4] = 1.0f;

    /* IP hands */
    hands1[0][0] = parse_card("Ac"); hands1[0][1] = parse_card("Kc"); w1[0] = 1.0f;
    hands1[1][0] = parse_card("Qc"); hands1[1][1] = parse_card("Jc"); w1[1] = 1.0f;
    hands1[2][0] = parse_card("Tc"); hands1[2][1] = parse_card("9c"); w1[2] = 1.0f;
    hands1[3][0] = parse_card("8c"); hands1[3][1] = parse_card("7c"); w1[3] = 1.0f;
    hands1[4][0] = parse_card("6c"); hands1[4][1] = parse_card("5c"); w1[4] = 1.0f;

    float bet_sizes[] = {0.33f, 0.75f};
    int pot = 650;
    int stack = 9000;

    printf("Initializing solver...\n");
    Solver s;
    int err = solver_init(&s, board, 5,
                          (const int(*)[2])hands0, w0, 5,
                          (const int(*)[2])hands1, w1, 5,
                          pot, stack, bet_sizes, 2);
    if (err) {
        printf("ERROR: solver_init failed\n");
        return 1;
    }

    printf("Tree: %d nodes\n", s.num_nodes);
    printf("Hands: OOP=%d, IP=%d\n", s.num_hands[0], s.num_hands[1]);

    /* Print tree structure */
    printf("\nTree structure:\n");
    for (int i = 0; i < s.num_nodes && i < 30; i++) {
        TreeNode *n = &s.nodes[i];
        const char *type_str = "???";
        switch (n->type) {
            case NODE_DECISION: type_str = "DECISION"; break;
            case NODE_FOLD: type_str = "FOLD"; break;
            case NODE_SHOWDOWN: type_str = "SHOWDOWN"; break;
            case NODE_LEAF: type_str = "LEAF"; break;
        }
        printf("  [%d] %s player=%d actions=%d pot=%d children=",
               i, type_str, n->player, n->num_actions, n->pot);
        for (int a = 0; a < n->num_actions; a++)
            printf("%d ", n->children[a]);
        printf("\n");
    }

    printf("\nSolving (10 iterations)...\n");
    double t0 = get_time_ms();
    solver_solve(&s, 10, 0.01f);
    double t1 = get_time_ms();
    printf("Done: %.1f ms\n", t1 - t0);

    printf("\nSolving (100 iterations)...\n");
    solver_free(&s);
    solver_init(&s, board, 5,
                (const int(*)[2])hands0, w0, 5,
                (const int(*)[2])hands1, w1, 5,
                pot, stack, bet_sizes, 2);
    t0 = get_time_ms();
    solver_solve(&s, 100, 0.01f);
    t1 = get_time_ms();
    printf("Done: %.1f ms\n", t1 - t0);

    /* Show strategies */
    printf("\nOOP strategies (root node):\n");
    TreeNode *root = &s.nodes[0];
    for (int h = 0; h < s.num_hands[0]; h++) {
        char c0[3], c1[3];
        format_card(s.hands[0][h][0], c0);
        format_card(s.hands[0][h][1], c1);
        float strat[MAX_ACTIONS];
        solver_get_strategy(&s, 0, h, strat);
        printf("  %s%s: ", c0, c1);
        for (int a = 0; a < root->num_actions; a++) {
            printf("%.1f%% ", strat[a] * 100);
        }
        printf("\n");
    }

    solver_free(&s);
    printf("\n=== Test complete ===\n");
    return 0;
}
