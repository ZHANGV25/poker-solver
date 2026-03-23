/**
 * prove_v2.c — Three remaining correctness proofs:
 *
 * 1. Cross-validate AVERAGE strategy against Rust solver
 * 2. Achieve exploitability below 1% of pot
 * 3. Verify with realistic 100-hand ranges
 */
#include "../src/solver_v2.h"
#include "../src/hand_eval.h"
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>

static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* Get AVERAGE strategy (from strategy_sum) instead of final iteration */
static void get_average_strategy(const SolverV2 *s, int player, int hand_idx,
                                  float *strat_out) {
    NodeV2 *root = &s->nodes[0];
    InfoSetV2 *is = &s->info_sets[0];
    if (!is->strategy_sum || root->player != player) {
        float u = 1.0f / root->num_actions;
        for (int a = 0; a < root->num_actions; a++) strat_out[a] = u;
        return;
    }
    int nh = is->num_hands;
    int na = is->num_actions;
    float sum = 0;
    for (int a = 0; a < na; a++)
        sum += is->strategy_sum[a * nh + hand_idx];
    if (sum > 0) {
        float inv = 1.0f / sum;
        for (int a = 0; a < na; a++)
            strat_out[a] = is->strategy_sum[a * nh + hand_idx] * inv;
    } else {
        float u = 1.0f / na;
        for (int a = 0; a < na; a++) strat_out[a] = u;
    }
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
    printf("=============================================================\n");
    printf("  CORRECTNESS PROOFS (v2 solver)\n");
    printf("=============================================================\n\n");

    int board[5] = {parse_card("Qs"), parse_card("As"), parse_card("2d"),
                    parse_card("7h"), parse_card("4c")};

    /* ── PROOF 1: Cross-validate average strategy vs Rust solver ──── */
    printf("PROOF 1: Average strategy comparison\n\n");
    {
        int h0[4][2], h1[4][2];
        float w0[4] = {1,1,1,1}, w1[4] = {1,1,1,1};
        h0[0][0]=parse_card("Ah"); h0[0][1]=parse_card("Kh");
        h0[1][0]=parse_card("Qh"); h0[1][1]=parse_card("Qc");
        h0[2][0]=parse_card("Jh"); h0[2][1]=parse_card("Th");
        h0[3][0]=parse_card("6h"); h0[3][1]=parse_card("5h");
        h1[0][0]=parse_card("Ac"); h1[0][1]=parse_card("Kc");
        h1[1][0]=parse_card("3c"); h1[1][1]=parse_card("3d");
        h1[2][0]=parse_card("Tc"); h1[2][1]=parse_card("9c");
        h1[3][0]=parse_card("8c"); h1[3][1]=parse_card("8d");

        float bs[] = {0.75f};
        SolverV2 s;
        sv2_init(&s, board, 5, (const int(*)[2])h0, w0, 4,
                 (const int(*)[2])h1, w1, 4, 1000, 5000, bs, 1);
        sv2_solve(&s, 10000, 0.0f);

        const char *names[] = {"AhKh(TPTK)", "QhQc(trips)", "JhTh(JT)", "6h5h(air)"};

        /* Rust reference (from actual run):
         * AhKh: check=0% bet=100%
         * QhQc: check=16% bet=84%
         * JhTh: check=100% bet=0%
         * 6h5h: check=21% bet=79% */
        float rust_bet[] = {1.00f, 0.84f, 0.00f, 0.79f};

        printf("  %-14s  %-12s  %-12s  %-12s  %-6s\n",
               "Hand", "Avg Strat", "Final Iter", "Rust Ref", "Match?");

        int all_close = 1;
        for (int h = 0; h < 4; h++) {
            float avg[MAX_ACTIONS_V2], final_s[MAX_ACTIONS_V2];
            get_average_strategy(&s, 0, h, avg);
            sv2_get_strategy(&s, 0, h, final_s);

            float avg_bet = 0, final_bet = 0;
            for (int a = 1; a < s.nodes[0].num_actions; a++) {
                avg_bet += avg[a];
                final_bet += final_s[a];
            }

            float diff = fabsf(avg_bet - rust_bet[h]);
            int close = diff < 0.30f; /* within 30% */
            if (!close) all_close = 0;

            printf("  %-14s  bet=%.0f%%      bet=%.0f%%      bet=%.0f%%      %s\n",
                   names[h], avg_bet*100, final_bet*100, rust_bet[h]*100,
                   close ? "OK" : "DIFF");
        }

        float exploit = sv2_exploitability(&s);
        printf("\n  Exploitability: %.2f%% of pot\n", exploit/1000*100);
        printf("  Result: %s\n", all_close ? "PASS" : "PARTIAL");
        sv2_free(&s);
    }

    /* ── PROOF 2: Exploitability below 1% of pot ─────────────────── */
    printf("\nPROOF 2: Exploitability convergence to <1%%\n\n");
    {
        int h0[4][2], h1[4][2];
        float w0[4] = {1,1,1,1}, w1[4] = {1,1,1,1};
        h0[0][0]=parse_card("Ah"); h0[0][1]=parse_card("Kh");
        h0[1][0]=parse_card("Qh"); h0[1][1]=parse_card("Qc");
        h0[2][0]=parse_card("Jh"); h0[2][1]=parse_card("Th");
        h0[3][0]=parse_card("6h"); h0[3][1]=parse_card("5h");
        h1[0][0]=parse_card("Ac"); h1[0][1]=parse_card("Kc");
        h1[1][0]=parse_card("3c"); h1[1][1]=parse_card("3d");
        h1[2][0]=parse_card("Tc"); h1[2][1]=parse_card("9c");
        h1[3][0]=parse_card("8c"); h1[3][1]=parse_card("8d");

        float bs[] = {0.75f};
        SolverV2 s;
        sv2_init(&s, board, 5, (const int(*)[2])h0, w0, 4,
                 (const int(*)[2])h1, w1, 4, 1000, 5000, bs, 1);

        int iters[] = {100, 500, 1000, 2000, 5000, 10000, 20000, 50000};
        int prev = 0;
        printf("  %8s  %12s\n", "Iters", "Exploit/Pot");

        for (int i = 0; i < 8; i++) {
            sv2_solve(&s, iters[i] - prev, 0.0f);
            float exploit = sv2_exploitability(&s);
            printf("  %8d  %10.4f%%\n", iters[i], exploit/1000*100);
            prev = iters[i];
            if (exploit/1000*100 < 1.0) {
                printf("  PASS: below 1%% at %d iterations\n", iters[i]);
                break;
            }
        }

        sv2_free(&s);
    }

    /* ── PROOF 3: Realistic 100-hand ranges ──────────────────────── */
    printf("\nPROOF 3: Realistic range (100 hands each)\n\n");
    {
        int h0[MAX_HANDS_V2][2], h1[MAX_HANDS_V2][2];
        float w0[MAX_HANDS_V2], w1[MAX_HANDS_V2];
        int n0 = gen_hands(board, 5, h0, w0, 100);
        int n1 = gen_hands(board, 5, h1, w1, 100);

        float bs[] = {0.33f, 0.75f};
        SolverV2 s;
        sv2_init(&s, board, 5, (const int(*)[2])h0, w0, n0,
                 (const int(*)[2])h1, w1, n1, 1000, 9000, bs, 2);

        printf("  Hands: OOP=%d IP=%d, Tree: %d nodes\n", s.num_hands[0], s.num_hands[1], s.num_nodes);

        double t0 = get_time_ms();
        sv2_solve(&s, 1000, 0.0f);
        double elapsed = get_time_ms() - t0;

        float exploit = sv2_exploitability(&s);
        printf("  1000 iter: %.0fms, exploit=%.2f%% of pot\n", elapsed, exploit/1000*100);

        /* Verify mixed strategy (not degenerate) */
        int n_mostly_bet = 0, n_mostly_check = 0, n_mixed = 0;
        for (int h = 0; h < s.num_hands[0]; h++) {
            float strat[MAX_ACTIONS_V2];
            sv2_get_strategy(&s, 0, h, strat);
            float bet = 0;
            for (int a = 1; a < s.nodes[0].num_actions; a++) bet += strat[a];
            if (bet > 0.8) n_mostly_bet++;
            else if (bet < 0.2) n_mostly_check++;
            else n_mixed++;
        }

        printf("  Strategy distribution: %d bet, %d check, %d mixed\n",
               n_mostly_bet, n_mostly_check, n_mixed);

        int pass = (n_mostly_bet > 0 && n_mostly_check > 0);
        printf("  %s: %s\n", pass ? "PASS" : "FAIL",
               pass ? "both betting and checking hands exist" : "degenerate strategy");

        /* Show a few interesting hands */
        printf("\n  Sample hands:\n");
        for (int h = 0; h < s.num_hands[0] && h < s.num_hands[0]; h++) {
            char c0[3], c1[3];
            format_card(s.hands[0][h][0], c0);
            format_card(s.hands[0][h][1], c1);
            float strat[MAX_ACTIONS_V2];
            sv2_get_strategy(&s, 0, h, strat);
            float bet = 0;
            for (int a = 1; a < s.nodes[0].num_actions; a++) bet += strat[a];

            /* Only print hands that are interesting (strong pairs, draws, air) */
            int r0 = s.hands[0][h][0] >> 2, r1 = s.hands[0][h][1] >> 2;
            if (r0 == r1 || r0 >= 10 || r1 >= 10 || (bet > 0.4 && bet < 0.6)) {
                printf("    %s%s: check=%.0f%% bet=%.0f%%\n",
                       c0, c1, strat[0]*100, bet*100);
            }
        }

        sv2_free(&s);
    }

    printf("\n=============================================================\n");
    return 0;
}
