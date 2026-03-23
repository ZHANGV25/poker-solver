/**
 * debug_values.c — Debug fold/showdown value computation
 *
 * Tests with a known scenario where we can verify correctness manually:
 * - River board: Qs As 2d 7h 4c
 * - OOP has AhKh (top pair top kicker) and 6h5h (nothing)
 * - IP has AcKc (top pair top kicker) and 3c3d (pocket threes, no pair on board)
 * - Pot: 10 BB, Stack: 90 BB
 *
 * Expected behavior:
 * - AhKh vs AcKc = tie (both have TPTK)
 * - AhKh vs 3c3d = AhKh wins (pair of aces beats pair of threes)
 * - 6h5h vs AcKc = 6h5h loses (high card vs pair of aces)
 * - 6h5h vs 3c3d = 6h5h loses (high card vs pair of threes)
 *
 * So OOP's AhKh should bet for value, 6h5h should check/fold or bluff.
 */
#include "../src/solver.h"
#include "../src/hand_eval.h"
#include <stdio.h>
#include <string.h>

int main(void) {
    printf("=== Debug: Value Computation ===\n\n");

    int board[5];
    board[0] = parse_card("Qs");
    board[1] = parse_card("As");
    board[2] = parse_card("2d");
    board[3] = parse_card("7h");
    board[4] = parse_card("4c");

    /* Verify hand evaluation */
    printf("Hand evaluation check:\n");
    struct { const char *name; int cards[7]; } tests[] = {
        {"AhKh (TPTK)", {board[0], board[1], board[2], board[3], board[4],
                         parse_card("Ah"), parse_card("Kh")}},
        {"AcKc (TPTK)", {board[0], board[1], board[2], board[3], board[4],
                         parse_card("Ac"), parse_card("Kc")}},
        {"3c3d (pair 3s)", {board[0], board[1], board[2], board[3], board[4],
                            parse_card("3c"), parse_card("3d")}},
        {"6h5h (nothing)", {board[0], board[1], board[2], board[3], board[4],
                            parse_card("6h"), parse_card("5h")}},
    };
    for (int i = 0; i < 4; i++) {
        uint32_t str = eval7(tests[i].cards);
        int cat = str >> 20;
        const char *cat_names[] = {"?","HighCard","Pair","TwoPair","Trips",
                                   "Straight","Flush","FullHouse","Quads","StraightFlush"};
        printf("  %s: strength=%u category=%s\n",
               tests[i].name, str, cat < 10 ? cat_names[cat] : "???");
    }

    /* Set up solver */
    int hands0[2][2], hands1[2][2];
    float w0[2] = {1.0f, 1.0f}, w1[2] = {1.0f, 1.0f};

    hands0[0][0] = parse_card("Ah"); hands0[0][1] = parse_card("Kh"); /* TPTK */
    hands0[1][0] = parse_card("6h"); hands0[1][1] = parse_card("5h"); /* nothing */
    hands1[0][0] = parse_card("Ac"); hands1[0][1] = parse_card("Kc"); /* TPTK */
    hands1[1][0] = parse_card("3c"); hands1[1][1] = parse_card("3d"); /* pair 3s */

    /* Just one bet size: 75% pot */
    float bet_sizes[] = {0.75f};
    int pot = 1000;   /* 10 BB * 100 */
    int stack = 9000; /* 90 BB * 100 */

    printf("\nSetup: OOP={AhKh, 6h5h} vs IP={AcKc, 3c3d}\n");
    printf("Pot=10BB, Stack=90BB, Bet=75%%pot\n\n");

    Solver s;
    solver_init(&s, board, 5,
                (const int(*)[2])hands0, w0, 2,
                (const int(*)[2])hands1, w1, 2,
                pot, stack, bet_sizes, 1);

    printf("Tree: %d nodes\n", s.num_nodes);
    for (int i = 0; i < s.num_nodes; i++) {
        TreeNode *n = &s.nodes[i];
        const char *types[] = {"DECISION","FOLD","SHOWDOWN","CHANCE","LEAF"};
        printf("  [%d] %s p=%d actions=%d pot=%d bets=[%d,%d] children=",
               i, n->type < 5 ? types[n->type] : "?",
               n->player, n->num_actions, n->pot, n->bets[0], n->bets[1]);
        for (int a = 0; a < n->num_actions; a++)
            printf("%d ", n->children[a]);
        printf("\n");
    }

    /* Solve */
    printf("\nSolving 1000 iterations...\n");
    solver_solve(&s, 1000, 0.001f);

    /* Show strategies */
    printf("\nOOP strategies:\n");
    for (int h = 0; h < s.num_hands[0]; h++) {
        char c0[3], c1[3];
        format_card(s.hands[0][h][0], c0);
        format_card(s.hands[0][h][1], c1);
        float strat[MAX_ACTIONS];
        solver_get_strategy(&s, 0, h, strat);
        printf("  %s%s: ", c0, c1);
        /* Label actions */
        TreeNode *root = &s.nodes[0];
        for (int a = 0; a < root->num_actions; a++) {
            TreeNode *child = &s.nodes[root->children[a]];
            if (child->type == NODE_FOLD) printf("Fold=%.1f%% ", strat[a]*100);
            else if (child->type == NODE_SHOWDOWN) printf("Check=%.1f%% ", strat[a]*100);
            else if (child->type == NODE_DECISION) {
                /* Figure out if this is a bet or raise by checking pot increase */
                int bet_amount = child->pot - root->pot;
                printf("Bet%d=%.1f%% ", bet_amount, strat[a]*100);
            }
            else printf("?=%.1f%% ", strat[a]*100);
        }
        printf("\n");
    }

    printf("\nExpected behavior:\n");
    printf("  AhKh: should bet for value (beats 3c3d, ties AcKc)\n");
    printf("  6h5h: should check/fold mostly (loses to everything)\n");

    solver_free(&s);
    printf("\n=== Done ===\n");
    return 0;
}
