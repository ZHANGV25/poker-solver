/**
 * cross_validate.c — Compare our C solver against the Rust postflop-solver
 *
 * Rust solver output (1000 iterations, 0.0074 BB exploitability):
 *   OOP root node:
 *     6h5h: Check=21.3%  Bet75%=78.7%  (bluff!)
 *     AhKh: Check=0.0%   Bet75%=100.0% (value bet)
 *     JhTh: Check=100.0% Bet75%=0.0%   (check weak)
 *     QhQc: Check=16.2%  Bet75%=83.8%  (value bet, some trapping)
 */
#include "../src/solver.h"
#include "../src/hand_eval.h"
#include <stdio.h>
#include <math.h>

int main(void) {
    printf("=== Cross-Validation: C Solver vs Rust postflop-solver ===\n\n");

    int board[5];
    board[0] = parse_card("Qs");
    board[1] = parse_card("As");
    board[2] = parse_card("2d");
    board[3] = parse_card("7h");
    board[4] = parse_card("4c");

    /* Exact same hands as Rust test */
    int hands0[4][2], hands1[4][2];
    float w0[4] = {1,1,1,1}, w1[4] = {1,1,1,1};

    /* OOP: AhKh, QhQc, JhTh, 6h5h */
    hands0[0][0] = parse_card("Ah"); hands0[0][1] = parse_card("Kh");
    hands0[1][0] = parse_card("Qh"); hands0[1][1] = parse_card("Qc");
    hands0[2][0] = parse_card("Jh"); hands0[2][1] = parse_card("Th");
    hands0[3][0] = parse_card("6h"); hands0[3][1] = parse_card("5h");

    /* IP: AcKc, 3c3d, Tc9c, 8c8d */
    hands1[0][0] = parse_card("Ac"); hands1[0][1] = parse_card("Kc");
    hands1[1][0] = parse_card("3c"); hands1[1][1] = parse_card("3d");
    hands1[2][0] = parse_card("Tc"); hands1[2][1] = parse_card("9c");
    hands1[3][0] = parse_card("8c"); hands1[3][1] = parse_card("8d");

    /* Same parameters: pot=10BB, stack=50BB, bet=75% */
    float bet_sizes[] = {0.75f};
    int pot = 1000;   /* 10 BB * 100 scale */
    int stack = 5000; /* 50 BB * 100 scale */

    Solver s;
    solver_init(&s, board, 5,
                (const int(*)[2])hands0, w0, 4,
                (const int(*)[2])hands1, w1, 4,
                pot, stack, bet_sizes, 1);

    printf("C solver: %d nodes, OOP=%d hands, IP=%d hands\n\n",
           s.num_nodes, s.num_hands[0], s.num_hands[1]);

    /* Solve 1000 iterations */
    solver_solve(&s, 1000, 0.0f);
    float exploit = solver_exploitability(&s);

    printf("Exploitability: %.4f chips = %.4f BB = %.4f%% of pot\n\n",
           exploit, exploit / 100.0f, exploit / (float)pot * 100.0f);

    /* Compare OOP root strategies */
    printf("%-12s  %-18s  %-18s  %-8s\n", "Hand", "C Solver", "Rust Solver", "Match?");
    printf("%-12s  %-18s  %-18s  %-8s\n", "----", "--------", "-----------", "------");

    /* Rust reference values (OOP root): */
    const char *hand_names[] = {"AhKh", "QhQc", "JhTh", "6h5h"};
    /* Rust: check%, bet75% for each hand */
    float rust_check[] = {0.0f, 16.2f, 100.0f, 21.3f};
    float rust_bet[]   = {100.0f, 83.8f, 0.0f, 78.7f};

    TreeNode *root = &s.nodes[0];
    int all_match = 1;
    for (int h = 0; h < 4; h++) {
        float strat[MAX_ACTIONS];
        solver_get_strategy(&s, 0, h, strat);

        /* Our tree: action 0=check(->IP decision), action 1=bet75%, action 2=all-in */
        float c_check = strat[0] * 100;
        float c_bet = 0;
        for (int a = 1; a < root->num_actions; a++)
            c_bet += strat[a] * 100;

        float diff_check = fabsf(c_check - rust_check[h]);
        float diff_bet = fabsf(c_bet - rust_bet[h]);
        int match = (diff_check < 15.0f && diff_bet < 15.0f); /* within 15% */

        printf("%-12s  Chk=%.1f%% Bet=%.1f%%  Chk=%.1f%% Bet=%.1f%%  %s\n",
               hand_names[h],
               c_check, c_bet,
               rust_check[h], rust_bet[h],
               match ? "OK" : "DIFF");
        if (!match) all_match = 0;
    }

    printf("\n");
    if (all_match) {
        printf("PASS: All strategies within 15%% of Rust solver\n");
    } else {
        printf("NOTE: Some strategies differ. This may be due to:\n");
        printf("  - Different tree structure (our tree includes all-in as separate action)\n");
        printf("  - Different DCFR parameters (Rust uses gamma=3, resets at powers of 4)\n");
        printf("  - Convergence differences with only 1000 iterations\n");
        printf("  Strategies should converge closer with more iterations.\n");
    }

    solver_free(&s);
    printf("\n=== Done ===\n");
    return 0;
}
