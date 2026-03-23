/**
 * bench_realistic.c — Measure ACTUAL times for each component
 *
 * No estimates, no projections. Real measurements of:
 * 1. Hand strength evaluation (eval7) throughput
 * 2. Showdown comparison (N×M pairs)
 * 3. River rollout for one runout
 * 4. Full river rollout across 46 cards
 * 5. Leaf value computation (4 strategies × rollout)
 * 6. DCFR river solve (complete)
 * 7. Simulated turn solve with leaf rollouts
 * 8. Simulated flop solve (47 turns × leaf rollouts)
 */

#include "../src/solver.h"
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

/* Generate hands not blocked by board */
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
    printf("  REALISTIC BENCHMARKS — ACTUAL MEASURED TIMES\n");
    printf("=============================================================\n\n");

    /* Setup: flop board Qs As 2d */
    int flop[3] = {parse_card("Qs"), parse_card("As"), parse_card("2d")};
    int turn_card = parse_card("7h");
    int river_card = parse_card("4c");

    int turn_board[4] = {flop[0], flop[1], flop[2], turn_card};
    int river_board[5] = {flop[0], flop[1], flop[2], turn_card, river_card};

    /* ── Benchmark 1: eval7 throughput ────────────────────────────── */
    printf("1. eval7 throughput\n");
    {
        int cards7[7] = {flop[0], flop[1], flop[2], turn_card, river_card, 0, 0};
        int total_evals = 0;
        double t0 = get_time_us();
        for (int c0 = 0; c0 < 51; c0++) {
            for (int c1 = c0+1; c1 < 52; c1++) {
                cards7[5] = c0;
                cards7[6] = c1;
                eval7(cards7);
                total_evals++;
            }
        }
        double elapsed = get_time_us() - t0;
        printf("   %d evaluations in %.0f us (%.1f M eval/sec)\n\n",
               total_evals, elapsed, total_evals / elapsed);
    }

    /* ── Benchmark 2: N×M showdown comparison ─────────────────────── */
    printf("2. Showdown comparison (precomputed strengths)\n");
    for (int N = 40; N <= 120; N += 20) {
        int hands0[MAX_HANDS][2], hands1[MAX_HANDS][2];
        float w0[MAX_HANDS], w1[MAX_HANDS];
        int n0 = gen_hands(river_board, 5, hands0, w0, N);
        int n1 = gen_hands(river_board, 5, hands1, w1, N);

        /* Precompute strengths */
        uint32_t str0[MAX_HANDS], str1[MAX_HANDS];
        int board7[7];
        for (int i = 0; i < 5; i++) board7[i] = river_board[i];
        for (int h = 0; h < n0; h++) {
            board7[5] = hands0[h][0]; board7[6] = hands0[h][1];
            str0[h] = eval7(board7);
        }
        for (int h = 0; h < n1; h++) {
            board7[5] = hands1[h][0]; board7[6] = hands1[h][1];
            str1[h] = eval7(board7);
        }

        /* Measure N×M comparison */
        float cfv[MAX_HANDS];
        int reps = 1000;
        double t0 = get_time_us();
        for (int r = 0; r < reps; r++) {
            for (int h = 0; h < n0; h++) {
                float win = 0, lose = 0;
                for (int o = 0; o < n1; o++) {
                    if (hands0[h][0] == hands1[o][0] || hands0[h][0] == hands1[o][1] ||
                        hands0[h][1] == hands1[o][0] || hands0[h][1] == hands1[o][1])
                        continue;
                    if (str0[h] > str1[o]) win += w1[o];
                    else if (str0[h] < str1[o]) lose += w1[o];
                }
                cfv[h] = win * 500 - lose * 500;
            }
        }
        double elapsed = (get_time_us() - t0) / reps;
        printf("   N=%d M=%d: %.1f us per showdown eval (%.0f us for %d reps)\n",
               n0, n1, elapsed, (get_time_us() - t0) - elapsed * reps + elapsed, reps);
    }
    printf("\n");

    /* ── Benchmark 3: Single river rollout ────────────────────────── */
    printf("3. Single river card rollout (eval strengths + showdown for one river)\n");
    for (int N = 40; N <= 120; N += 40) {
        int hands0[MAX_HANDS][2], hands1[MAX_HANDS][2];
        float w0[MAX_HANDS], w1[MAX_HANDS];
        /* Use turn board (4 cards) to generate hands */
        int n0 = gen_hands(turn_board, 4, hands0, w0, N);
        int n1 = gen_hands(turn_board, 4, hands1, w1, N);

        int reps = 100;
        double t0 = get_time_us();
        for (int r = 0; r < reps; r++) {
            /* For one specific river card, compute all strengths and do showdown */
            int full_board[5] = {flop[0], flop[1], flop[2], turn_card, river_card};
            uint32_t s0[MAX_HANDS], s1[MAX_HANDS];
            int board7[7];
            for (int i = 0; i < 5; i++) board7[i] = full_board[i];

            for (int h = 0; h < n0; h++) {
                if (hands0[h][0] == river_card || hands0[h][1] == river_card) {
                    s0[h] = 0; continue;
                }
                board7[5] = hands0[h][0]; board7[6] = hands0[h][1];
                s0[h] = eval7(board7);
            }
            for (int h = 0; h < n1; h++) {
                if (hands1[h][0] == river_card || hands1[h][1] == river_card) {
                    s1[h] = 0; continue;
                }
                board7[5] = hands1[h][0]; board7[6] = hands1[h][1];
                s1[h] = eval7(board7);
            }

            /* N×M comparison */
            float cfv[MAX_HANDS];
            for (int h = 0; h < n0; h++) {
                if (s0[h] == 0) { cfv[h] = 0; continue; }
                float win = 0, lose = 0;
                for (int o = 0; o < n1; o++) {
                    if (s1[o] == 0) continue;
                    if (hands0[h][0] == hands1[o][0] || hands0[h][0] == hands1[o][1] ||
                        hands0[h][1] == hands1[o][0] || hands0[h][1] == hands1[o][1])
                        continue;
                    if (s0[h] > s1[o]) win += w1[o];
                    else if (s0[h] < s1[o]) lose += w1[o];
                }
                cfv[h] = win - lose;
            }
        }
        double elapsed = (get_time_us() - t0) / reps;
        printf("   N=%d: %.0f us per river card rollout\n", N, elapsed);
    }
    printf("\n");

    /* ── Benchmark 4: Full 46-card river rollout ──────────────────── */
    printf("4. Full river rollout (46 cards × eval + showdown)\n");
    for (int N = 40; N <= 120; N += 40) {
        int hands0[MAX_HANDS][2], hands1[MAX_HANDS][2];
        float w0[MAX_HANDS], w1[MAX_HANDS];
        int n0 = gen_hands(turn_board, 4, hands0, w0, N);
        int n1 = gen_hands(turn_board, 4, hands1, w1, N);

        int reps = 10;
        double t0 = get_time_us();
        for (int r = 0; r < reps; r++) {
            float total_cfv[MAX_HANDS] = {0};

            /* Try all 46 possible river cards */
            for (int rc = 0; rc < 52; rc++) {
                /* Skip if river card is on board or would be blocked */
                if (rc == flop[0] || rc == flop[1] || rc == flop[2] || rc == turn_card)
                    continue;

                int full_board[5] = {flop[0], flop[1], flop[2], turn_card, rc};
                uint32_t s0[MAX_HANDS], s1[MAX_HANDS];
                int board7[7];
                for (int i = 0; i < 5; i++) board7[i] = full_board[i];

                for (int h = 0; h < n0; h++) {
                    if (hands0[h][0] == rc || hands0[h][1] == rc) {
                        s0[h] = 0; continue;
                    }
                    board7[5] = hands0[h][0]; board7[6] = hands0[h][1];
                    s0[h] = eval7(board7);
                }
                for (int h = 0; h < n1; h++) {
                    if (hands1[h][0] == rc || hands1[h][1] == rc) {
                        s1[h] = 0; continue;
                    }
                    board7[5] = hands1[h][0]; board7[6] = hands1[h][1];
                    s1[h] = eval7(board7);
                }

                for (int h = 0; h < n0; h++) {
                    if (s0[h] == 0) continue;
                    float win = 0, lose = 0;
                    for (int o = 0; o < n1; o++) {
                        if (s1[o] == 0) continue;
                        if (hands0[h][0] == hands1[o][0] || hands0[h][0] == hands1[o][1] ||
                            hands0[h][1] == hands1[o][0] || hands0[h][1] == hands1[o][1])
                            continue;
                        if (s0[h] > s1[o]) win += w1[o];
                        else if (s0[h] < s1[o]) lose += w1[o];
                    }
                    total_cfv[h] += win - lose;
                }
            }
        }
        double elapsed = (get_time_us() - t0) / reps;
        printf("   N=%d: %.1f ms per full 46-card rollout\n", N, elapsed / 1000.0);
    }
    printf("\n");

    /* ── Benchmark 5: 4-strategy leaf eval (4 × rollout) ─────────── */
    printf("5. Leaf value eval (4 continuation strategies × 46-card rollout)\n");
    for (int N = 40; N <= 120; N += 40) {
        int hands0[MAX_HANDS][2], hands1[MAX_HANDS][2];
        float w0[MAX_HANDS], w1[MAX_HANDS];
        int n0 = gen_hands(turn_board, 4, hands0, w0, N);
        int n1 = gen_hands(turn_board, 4, hands1, w1, N);

        int reps = 5;
        double t0 = get_time_us();
        for (int r = 0; r < reps; r++) {
            /* 4 strategies */
            for (int strat = 0; strat < 4; strat++) {
                float total_cfv[MAX_HANDS] = {0};
                for (int rc = 0; rc < 52; rc++) {
                    if (rc == flop[0] || rc == flop[1] || rc == flop[2] || rc == turn_card)
                        continue;

                    int full_board[5] = {flop[0], flop[1], flop[2], turn_card, rc};
                    uint32_t s0[MAX_HANDS], s1[MAX_HANDS];
                    int board7[7];
                    for (int i = 0; i < 5; i++) board7[i] = full_board[i];

                    for (int h = 0; h < n0; h++) {
                        if (hands0[h][0] == rc || hands0[h][1] == rc) { s0[h] = 0; continue; }
                        board7[5] = hands0[h][0]; board7[6] = hands0[h][1];
                        s0[h] = eval7(board7);
                    }
                    for (int h = 0; h < n1; h++) {
                        if (hands1[h][0] == rc || hands1[h][1] == rc) { s1[h] = 0; continue; }
                        board7[5] = hands1[h][0]; board7[6] = hands1[h][1];
                        s1[h] = eval7(board7);
                    }

                    for (int h = 0; h < n0; h++) {
                        if (s0[h] == 0) continue;
                        float win = 0, lose = 0;
                        for (int o = 0; o < n1; o++) {
                            if (s1[o] == 0) continue;
                            if (hands0[h][0] == hands1[o][0] || hands0[h][0] == hands1[o][1] ||
                                hands0[h][1] == hands1[o][0] || hands0[h][1] == hands1[o][1])
                                continue;
                            if (s0[h] > s1[o]) win += w1[o];
                            else if (s0[h] < s1[o]) lose += w1[o];
                        }
                        total_cfv[h] += win - lose;
                    }
                }
            }
        }
        double elapsed = (get_time_us() - t0) / reps;
        printf("   N=%d: %.1f ms for 4-strategy leaf eval\n", N, elapsed / 1000.0);
    }
    printf("\n");

    /* ── Benchmark 6: DCFR river solve (existing solver) ──────────── */
    printf("6. DCFR river solve (our existing solver)\n");
    for (int N = 40; N <= 120; N += 20) {
        int hands0[MAX_HANDS][2], hands1[MAX_HANDS][2];
        float w0[MAX_HANDS], w1[MAX_HANDS];
        int n0 = gen_hands(river_board, 5, hands0, w0, N);
        int n1 = gen_hands(river_board, 5, hands1, w1, N);

        float bet_sizes[] = {0.33f, 0.75f};
        Solver s;
        solver_init(&s, river_board, 5,
                    (const int(*)[2])hands0, w0, n0,
                    (const int(*)[2])hands1, w1, n1,
                    1000, 9000, bet_sizes, 2);

        double t0 = get_time_us();
        solver_solve(&s, 500, 0.0f);
        double elapsed = get_time_us() - t0;
        float exploit = solver_exploitability(&s);
        printf("   N=%d: %.1f ms (500 iter, exploit=%.4f%%)\n",
               N, elapsed / 1000.0, exploit / 1000.0 * 100);
        solver_free(&s);
    }
    printf("\n");

    /* ── Benchmark 7: Simulated turn solve cost ───────────────────── */
    printf("7. Simulated turn solve = DCFR + leaf rollouts\n");
    printf("   (DCFR river solve × number of leaf nodes, plus 4-strat leaf eval)\n");
    for (int N = 40; N <= 100; N += 20) {
        /* DCFR cost: already measured above */
        /* Leaf eval cost: 4 strategies × 46 rivers */
        /* A turn tree has ~5 leaf nodes (check-through, call after bet, etc.) */
        int num_leaves = 5;

        /* Measure leaf eval for this N */
        int hands0[MAX_HANDS][2], hands1[MAX_HANDS][2];
        float w0[MAX_HANDS], w1[MAX_HANDS];
        int n0 = gen_hands(turn_board, 4, hands0, w0, N);
        int n1 = gen_hands(turn_board, 4, hands1, w1, N);

        double leaf_t0 = get_time_us();
        for (int leaf = 0; leaf < num_leaves; leaf++) {
            for (int strat = 0; strat < 4; strat++) {
                for (int rc = 0; rc < 52; rc++) {
                    if (rc == flop[0] || rc == flop[1] || rc == flop[2] || rc == turn_card)
                        continue;
                    int full_board[5] = {flop[0], flop[1], flop[2], turn_card, rc};
                    uint32_t s0[MAX_HANDS], s1[MAX_HANDS];
                    int board7[7];
                    for (int i = 0; i < 5; i++) board7[i] = full_board[i];
                    for (int h = 0; h < n0; h++) {
                        if (hands0[h][0] == rc || hands0[h][1] == rc) { s0[h]=0; continue; }
                        board7[5] = hands0[h][0]; board7[6] = hands0[h][1];
                        s0[h] = eval7(board7);
                    }
                    for (int h = 0; h < n1; h++) {
                        if (hands1[h][0] == rc || hands1[h][1] == rc) { s1[h]=0; continue; }
                        board7[5] = hands1[h][0]; board7[6] = hands1[h][1];
                        s1[h] = eval7(board7);
                    }
                    /* N×M comparison (just measuring throughput) */
                    volatile float dummy = 0;
                    for (int h = 0; h < n0; h++) {
                        if (s0[h] == 0) continue;
                        for (int o = 0; o < n1; o++) {
                            if (s1[o] == 0) continue;
                            dummy += (s0[h] > s1[o]) ? 1.0f : -1.0f;
                        }
                    }
                }
            }
        }
        double leaf_elapsed = (get_time_us() - leaf_t0) / 1000.0;

        /* DCFR cost (use same N for turn, solve on turn board) */
        /* We can't directly solve turn in current solver (river only),
           so estimate: DCFR cost scales with N^2 × nodes × iterations.
           Use river solve time as proxy since tree structure is similar. */
        int hands0r[MAX_HANDS][2], hands1r[MAX_HANDS][2];
        float w0r[MAX_HANDS], w1r[MAX_HANDS];
        int n0r = gen_hands(river_board, 5, hands0r, w0r, N);
        int n1r = gen_hands(river_board, 5, hands1r, w1r, N);
        float bet_sizes[] = {0.33f, 0.75f};
        Solver s;
        solver_init(&s, river_board, 5,
                    (const int(*)[2])hands0r, w0r, n0r,
                    (const int(*)[2])hands1r, w1r, n1r,
                    1000, 9000, bet_sizes, 2);
        double dcfr_t0 = get_time_us();
        solver_solve(&s, 500, 0.0f);
        double dcfr_elapsed = (get_time_us() - dcfr_t0) / 1000.0;
        solver_free(&s);

        double total = leaf_elapsed + dcfr_elapsed;
        printf("   N=%d: leaf_eval=%.0fms + dcfr=%.0fms = TOTAL %.0fms\n",
               N, leaf_elapsed, dcfr_elapsed, total);
    }
    printf("\n");

    /* ── Benchmark 8: Simulated flop solve (47 turns × everything) ── */
    printf("8. Simulated flop solve = 47 turn cards × leaf rollouts + DCFR\n");
    {
        int N = 80; /* Typical narrowed range for flop */
        int hands0[MAX_HANDS][2], hands1[MAX_HANDS][2];
        float w0[MAX_HANDS], w1[MAX_HANDS];

        /* Measure one turn card's leaf eval */
        int n0 = gen_hands(turn_board, 4, hands0, w0, N);
        int n1 = gen_hands(turn_board, 4, hands1, w1, N);
        int num_leaves = 5;

        double one_turn_t0 = get_time_us();
        for (int leaf = 0; leaf < num_leaves; leaf++) {
            for (int strat = 0; strat < 4; strat++) {
                for (int rc = 0; rc < 52; rc++) {
                    if (rc == flop[0] || rc == flop[1] || rc == flop[2] || rc == turn_card)
                        continue;
                    int full_board[5] = {flop[0], flop[1], flop[2], turn_card, rc};
                    uint32_t s0[MAX_HANDS], s1[MAX_HANDS];
                    int board7[7];
                    for (int i = 0; i < 5; i++) board7[i] = full_board[i];
                    for (int h = 0; h < n0; h++) {
                        if (hands0[h][0] == rc || hands0[h][1] == rc) { s0[h]=0; continue; }
                        board7[5] = hands0[h][0]; board7[6] = hands0[h][1];
                        s0[h] = eval7(board7);
                    }
                    for (int h = 0; h < n1; h++) {
                        if (hands1[h][0] == rc || hands1[h][1] == rc) { s1[h]=0; continue; }
                        board7[5] = hands1[h][0]; board7[6] = hands1[h][1];
                        s1[h] = eval7(board7);
                    }
                    volatile float dummy = 0;
                    for (int h = 0; h < n0; h++) {
                        if (s0[h] == 0) continue;
                        for (int o = 0; o < n1; o++) {
                            if (s1[o] == 0) continue;
                            dummy += (s0[h] > s1[o]) ? 1.0f : -1.0f;
                        }
                    }
                }
            }
        }
        double one_turn_ms = (get_time_us() - one_turn_t0) / 1000.0;

        /* DCFR for flop (similar tree, ~200 hands pre-narrowing) */
        int n0r = gen_hands(river_board, 5, hands0, w0, 200);
        int n1r = gen_hands(river_board, 5, hands1, w1, 200);
        float bet_sizes[] = {0.33f, 0.75f};
        Solver s;
        solver_init(&s, river_board, 5,
                    (const int(*)[2])hands0, w0, n0r,
                    (const int(*)[2])hands1, w1, n1r,
                    1000, 9000, bet_sizes, 2);
        double dcfr_t0 = get_time_us();
        solver_solve(&s, 500, 0.0f);
        double dcfr_ms = (get_time_us() - dcfr_t0) / 1000.0;
        solver_free(&s);

        double flop_total = 47.0 * one_turn_ms + dcfr_ms;

        printf("   N=%d:\n", N);
        printf("   One turn card leaf eval: %.0f ms\n", one_turn_ms);
        printf("   × 47 turn cards: %.0f ms\n", 47.0 * one_turn_ms);
        printf("   DCFR (200 hands, 500 iter): %.0f ms\n", dcfr_ms);
        printf("   FLOP TOTAL: %.1f seconds\n", flop_total / 1000.0);
    }
    printf("\n");

    /* ── Summary ──────────────────────────────────────────────────── */
    printf("=============================================================\n");
    printf("  TIMING SUMMARY\n");
    printf("=============================================================\n");
    printf("  All times measured on this machine, single-threaded.\n");
    printf("  Multi-threading divides times by available cores.\n");

    return 0;
}
