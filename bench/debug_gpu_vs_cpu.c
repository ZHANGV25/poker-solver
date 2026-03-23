/**
 * debug_gpu_vs_cpu.c — CPU reference for GPU comparison
 *
 * Solves the EXACT same spot as the GPU benchmark:
 *   Board: Qs As 2d 7h 4c
 *   80 hands per player (first 80 non-blocked)
 *   Same tree structure (check/bet75/allin at root)
 *   200 iterations
 *
 * Prints per-hand strategies so we can compare with GPU output.
 */
#include "../src/solver_v2.h"
#include "../src/hand_eval.h"
#include <stdio.h>
#include <time.h>

static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static int gen_hands(const int *board, int nb, int hands[][2],
                     float *weights, int max) {
    int count = 0;
    int blocked[52] = {0};
    for (int i = 0; i < nb; i++) blocked[board[i]] = 1;
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
    printf("=== CPU Reference (solver_v2, same spot as GPU bench) ===\n\n");

    int board[5] = {parse_card("Qs"), parse_card("As"), parse_card("2d"),
                    parse_card("7h"), parse_card("4c")};

    int h0[MAX_HANDS_V2][2], h1[MAX_HANDS_V2][2];
    float w0[MAX_HANDS_V2], w1[MAX_HANDS_V2];
    int n0 = gen_hands(board, 5, h0, w0, 80);
    int n1 = gen_hands(board, 5, h1, w1, 80);

    printf("Board: Qs As 2d 7h 4c\n");
    printf("Hands: OOP=%d IP=%d\n\n", n0, n1);

    /* Use same bet sizes as GPU bench: 75% and all-in via 3 actions */
    float bet_sizes[] = {0.75f};
    SolverV2 s;
    sv2_init(&s, board, 5,
             (const int(*)[2])h0, w0, n0,
             (const int(*)[2])h1, w1, n1,
             1000, 9000, bet_sizes, 1);

    printf("Tree: %d nodes, %d leaves\n", s.num_nodes, s.num_leaves);
    printf("Tree structure:\n");
    for (int i = 0; i < s.num_nodes && i < 30; i++) {
        const char *types[] = {"DECISION","FOLD","SHOWDOWN","LEAF"};
        printf("  [%d] %s p=%d actions=%d pot=%d bets=[%d,%d]",
               i, s.nodes[i].type < 4 ? types[s.nodes[i].type] : "?",
               s.nodes[i].player, s.nodes[i].num_actions,
               s.nodes[i].pot, s.nodes[i].bets[0], s.nodes[i].bets[1]);
        if (s.nodes[i].num_actions > 0) {
            printf(" children=");
            for (int a = 0; a < s.nodes[i].num_actions; a++)
                printf("%d ", s.nodes[i].children[a]);
        }
        printf("\n");
    }

    printf("\nSolving 200 iterations (Linear CFR, final iteration strategy)...\n");
    double t0 = get_time_ms();
    sv2_solve(&s, 200, 0.0f);
    double elapsed = get_time_ms() - t0;
    printf("Done: %.0fms\n\n", elapsed);

    /* Print first 10 OOP hands */
    printf("OOP strategies (first 10 hands):\n");
    printf("%-8s  ", "Hand");
    for (int a = 0; a < s.nodes[0].num_actions; a++) printf("Act%d    ", a);
    printf("\n");

    for (int h = 0; h < s.num_hands[0] && h < 10; h++) {
        char c0[3], c1[3];
        format_card(s.hands[0][h][0], c0);
        format_card(s.hands[0][h][1], c1);
        float strat[MAX_ACTIONS_V2];
        sv2_get_strategy(&s, 0, h, strat);
        printf("%-4s%-4s  ", c0, c1);
        for (int a = 0; a < s.nodes[0].num_actions; a++)
            printf("%.1f%%   ", strat[a] * 100);
        printf("\n");
    }

    /* Print some specific hands we know the answer for */
    printf("\nKey hands:\n");
    const char *key_hands[][2] = {
        {"Ah","Kh"}, {"Qh","Qc"}, {"Jh","Th"}, {"6h","5h"},
        {"Ac","Kc"}, {"3c","3d"}, {"Tc","9c"}, {"8c","8d"},
    };
    const char *key_names[] = {
        "AhKh(TPTK)", "QhQc(trips)", "JhTh(JT)", "6h5h(air)",
        "AcKc(TPTK)", "3c3d(pair3)", "Tc9c(T9)", "8c8d(pair8)",
    };

    for (int k = 0; k < 8; k++) {
        int c0 = parse_card(key_hands[k][0]);
        int c1 = parse_card(key_hands[k][1]);
        /* Find in range */
        int found = -1;
        for (int h = 0; h < s.num_hands[0]; h++) {
            if ((s.hands[0][h][0] == c0 && s.hands[0][h][1] == c1) ||
                (s.hands[0][h][0] == c1 && s.hands[0][h][1] == c0)) {
                found = h; break;
            }
        }
        if (found < 0) {
            /* Try IP */
            for (int h = 0; h < s.num_hands[1]; h++) {
                if ((s.hands[1][h][0] == c0 && s.hands[1][h][1] == c1) ||
                    (s.hands[1][h][0] == c1 && s.hands[1][h][1] == c0)) {
                    found = h; break;
                }
            }
            if (found >= 0) {
                float strat[MAX_ACTIONS_V2];
                sv2_get_strategy(&s, 1, found, strat);
                float bet = 0;
                for (int a = 1; a < s.nodes[0].num_actions; a++) bet += strat[a];
                printf("  IP  %-14s: check=%.0f%% bet=%.0f%%\n",
                       key_names[k], strat[0]*100, bet*100);
            }
        } else {
            float strat[MAX_ACTIONS_V2];
            sv2_get_strategy(&s, 0, found, strat);
            float bet = 0;
            for (int a = 1; a < s.nodes[0].num_actions; a++) bet += strat[a];
            printf("  OOP %-14s: check=%.0f%% bet=%.0f%%\n",
                   key_names[k], strat[0]*100, bet*100);
        }
    }

    float exploit = sv2_exploitability(&s);
    printf("\nExploitability: %.4f chips (%.4f%% of pot)\n",
           exploit, exploit / 1000 * 100);

    sv2_free(&s);
    return 0;
}
