/**
 * bench_flop_optimized.c — Measure flop solve with two key optimizations:
 *
 * 1. Precompute ALL river hand strengths for ALL 2,162 runouts once upfront,
 *    then reuse across iterations (eliminates redundant eval7 calls)
 *
 * 2. Parallelize leaf evaluation across turn cards using OpenMP
 *
 * Measures each optimization independently and combined.
 */

#include "../src/solver.h"
#include "../src/hand_eval.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

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

/* ── Precomputed river strengths table ───────────────────────────────── */

/**
 * Precompute hand strengths for ALL possible river completions.
 * For a given turn board (4 cards), there are 48 possible river cards.
 * For each river card, compute eval7 for all hands.
 *
 * Storage: strengths[river_idx][hand_idx] = uint32_t strength
 * river_idx = 0..47 (mapped from card 0..51 skipping board cards)
 */
typedef struct {
    uint32_t **strengths;     /* [num_rivers][num_hands] */
    int *river_cards;         /* [num_rivers] actual card values */
    int num_rivers;
    int num_hands;
} PrecomputedRiverStrengths;

static PrecomputedRiverStrengths precompute_river_strengths(
        const int *turn_board, /* 4 cards */
        const int hands[][2], int num_hands) {
    PrecomputedRiverStrengths prs;
    int blocked[52] = {0};
    for (int i = 0; i < 4; i++) blocked[turn_board[i]] = 1;

    /* Count valid river cards */
    prs.num_rivers = 0;
    prs.river_cards = malloc(48 * sizeof(int));
    for (int c = 0; c < 52; c++) {
        if (!blocked[c]) {
            prs.river_cards[prs.num_rivers++] = c;
        }
    }

    prs.num_hands = num_hands;
    prs.strengths = malloc(prs.num_rivers * sizeof(uint32_t*));

    for (int ri = 0; ri < prs.num_rivers; ri++) {
        prs.strengths[ri] = malloc(num_hands * sizeof(uint32_t));
        int rc = prs.river_cards[ri];
        int board7[7] = {turn_board[0], turn_board[1], turn_board[2],
                         turn_board[3], rc, 0, 0};

        for (int h = 0; h < num_hands; h++) {
            if (hands[h][0] == rc || hands[h][1] == rc) {
                prs.strengths[ri][h] = 0; /* blocked */
                continue;
            }
            board7[5] = hands[h][0];
            board7[6] = hands[h][1];
            prs.strengths[ri][h] = eval7(board7);
        }
    }
    return prs;
}

static void free_prs(PrecomputedRiverStrengths *prs) {
    for (int i = 0; i < prs->num_rivers; i++)
        free(prs->strengths[i]);
    free(prs->strengths);
    free(prs->river_cards);
}

/* ── Leaf evaluation using precomputed strengths ─────────────────────── */

/**
 * Evaluate one leaf node's continuation value using precomputed strengths.
 * For each of 4 strategies, for each river card, do N×M showdown.
 */
static double leaf_eval_precomputed(
        const PrecomputedRiverStrengths *prs0,
        const PrecomputedRiverStrengths *prs1,
        const int hands0[][2], int n0,
        const int hands1[][2], int n1,
        const float *w1,
        float *cfv_out /* [n0] */) {
    double t0 = get_time_us();

    memset(cfv_out, 0, n0 * sizeof(float));

    /* 4 continuation strategies */
    for (int strat = 0; strat < 4; strat++) {
        /* For each river card */
        for (int ri = 0; ri < prs0->num_rivers; ri++) {
            const uint32_t *s0 = prs0->strengths[ri];
            const uint32_t *s1 = prs1->strengths[ri];

            /* N×M showdown with precomputed strengths */
            for (int h = 0; h < n0; h++) {
                if (s0[h] == 0) continue;
                float win = 0, lose = 0;
                int hc0 = hands0[h][0], hc1 = hands0[h][1];
                for (int o = 0; o < n1; o++) {
                    if (s1[o] == 0) continue;
                    if (hc0 == hands1[o][0] || hc0 == hands1[o][1] ||
                        hc1 == hands1[o][0] || hc1 == hands1[o][1])
                        continue;
                    if (s0[h] > s1[o]) win += w1[o];
                    else if (s0[h] < s1[o]) lose += w1[o];
                }
                cfv_out[h] += (win - lose);
            }
        }
    }

    return (get_time_us() - t0) / 1000.0;
}

/* ── Leaf evaluation WITHOUT precomputed strengths (original) ────────── */

static double leaf_eval_naive(
        const int *turn_board,
        const int hands0[][2], int n0,
        const int hands1[][2], int n1,
        const float *w1,
        float *cfv_out) {
    double t0 = get_time_us();

    memset(cfv_out, 0, n0 * sizeof(float));

    for (int strat = 0; strat < 4; strat++) {
        for (int rc = 0; rc < 52; rc++) {
            if (rc == turn_board[0] || rc == turn_board[1] ||
                rc == turn_board[2] || rc == turn_board[3])
                continue;

            int full_board[5] = {turn_board[0], turn_board[1],
                                 turn_board[2], turn_board[3], rc};
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
                cfv_out[h] += win - lose;
            }
        }
    }

    return (get_time_us() - t0) / 1000.0;
}

/* ── Main ────────────────────────────────────────────────────────────── */

int main(void) {
    printf("=============================================================\n");
    printf("  FLOP OPTIMIZATION BENCHMARKS\n");
    printf("=============================================================\n\n");

#ifdef _OPENMP
    printf("OpenMP: enabled (%d threads available)\n\n", omp_get_max_threads());
#else
    printf("OpenMP: NOT enabled (single-threaded)\n\n");
#endif

    int flop[3] = {parse_card("Qs"), parse_card("As"), parse_card("2d")};
    int turn_card = parse_card("7h");
    int turn_board[4] = {flop[0], flop[1], flop[2], turn_card};
    int river_board[5] = {flop[0], flop[1], flop[2], turn_card, parse_card("4c")};

    int range_sizes[] = {40, 60, 80, 100, 120};
    int num_sizes = 5;
    int num_leaves = 5;

    for (int si = 0; si < num_sizes; si++) {
        int N = range_sizes[si];
        int hands0[MAX_HANDS][2], hands1[MAX_HANDS][2];
        float w0[MAX_HANDS], w1[MAX_HANDS];
        int n0 = gen_hands(turn_board, 4, hands0, w0, N);
        int n1 = gen_hands(turn_board, 4, hands1, w1, N);

        printf("--- N=%d hands per player ---\n", N);

        /* 1. Precompute river strengths (one-time cost) */
        double precomp_t0 = get_time_us();
        PrecomputedRiverStrengths prs0 = precompute_river_strengths(
            turn_board, (const int(*)[2])hands0, n0);
        PrecomputedRiverStrengths prs1 = precompute_river_strengths(
            turn_board, (const int(*)[2])hands1, n1);
        double precomp_ms = (get_time_us() - precomp_t0) / 1000.0;
        printf("  Precompute river strengths: %.1f ms (one-time)\n", precomp_ms);

        /* 2. Naive leaf eval (original — recomputes eval7 every time) */
        float cfv_naive[MAX_HANDS];
        double naive_total = 0;
        for (int leaf = 0; leaf < num_leaves; leaf++) {
            naive_total += leaf_eval_naive(turn_board,
                (const int(*)[2])hands0, n0,
                (const int(*)[2])hands1, n1, w1, cfv_naive);
        }
        printf("  Naive leaf eval (%d leaves): %.0f ms\n", num_leaves, naive_total);

        /* 3. Precomputed leaf eval (no redundant eval7) */
        float cfv_precomp[MAX_HANDS];
        double precomp_total = 0;
        for (int leaf = 0; leaf < num_leaves; leaf++) {
            precomp_total += leaf_eval_precomputed(&prs0, &prs1,
                (const int(*)[2])hands0, n0,
                (const int(*)[2])hands1, n1, w1, cfv_precomp);
        }
        printf("  Precomputed leaf eval (%d leaves): %.0f ms\n",
               num_leaves, precomp_total);
        printf("  Speedup: %.1fx\n", naive_total / precomp_total);

        /* 4. OpenMP parallel across turn cards for flop solve */
#ifdef _OPENMP
        double omp_t0 = get_time_us();
        float all_leaf_cfv[47][MAX_HANDS];

        #pragma omp parallel for schedule(dynamic)
        for (int tc_idx = 0; tc_idx < 48; tc_idx++) {
            int tc = -1;
            int idx = 0;
            for (int c = 0; c < 52; c++) {
                if (c == flop[0] || c == flop[1] || c == flop[2]) continue;
                if (idx == tc_idx) { tc = c; break; }
                idx++;
            }
            if (tc < 0) continue;

            /* For this turn card, generate hands and precompute strengths */
            int tb[4] = {flop[0], flop[1], flop[2], tc};
            int lh0[MAX_HANDS][2], lh1[MAX_HANDS][2];
            float lw0[MAX_HANDS], lw1[MAX_HANDS];
            int ln0 = gen_hands(tb, 4, lh0, lw0, N);
            int ln1 = gen_hands(tb, 4, lh1, lw1, N);

            PrecomputedRiverStrengths lprs0 = precompute_river_strengths(
                tb, (const int(*)[2])lh0, ln0);
            PrecomputedRiverStrengths lprs1 = precompute_river_strengths(
                tb, (const int(*)[2])lh1, ln1);

            /* Evaluate leaves */
            for (int leaf = 0; leaf < num_leaves; leaf++) {
                float lcfv[MAX_HANDS];
                leaf_eval_precomputed(&lprs0, &lprs1,
                    (const int(*)[2])lh0, ln0,
                    (const int(*)[2])lh1, ln1, lw1, lcfv);
            }

            free_prs(&lprs0);
            free_prs(&lprs1);
        }
        double omp_ms = (get_time_us() - omp_t0) / 1000.0;
        printf("  OpenMP parallel flop leaf eval (47 turns): %.0f ms\n", omp_ms);
#endif

        /* 5. Sequential flop solve (47 turns × precomputed leaf eval) */
        double seq_t0 = get_time_us();
        for (int tc_idx = 0; tc_idx < 48; tc_idx++) {
            int tc = -1;
            int idx = 0;
            for (int c = 0; c < 52; c++) {
                if (c == flop[0] || c == flop[1] || c == flop[2]) continue;
                if (idx == tc_idx) { tc = c; break; }
                idx++;
            }
            if (tc < 0) continue;

            int tb[4] = {flop[0], flop[1], flop[2], tc};
            int lh0[MAX_HANDS][2], lh1[MAX_HANDS][2];
            float lw0[MAX_HANDS], lw1[MAX_HANDS];
            int ln0 = gen_hands(tb, 4, lh0, lw0, N);
            int ln1 = gen_hands(tb, 4, lh1, lw1, N);

            PrecomputedRiverStrengths lprs0 = precompute_river_strengths(
                tb, (const int(*)[2])lh0, ln0);
            PrecomputedRiverStrengths lprs1 = precompute_river_strengths(
                tb, (const int(*)[2])lh1, ln1);

            for (int leaf = 0; leaf < num_leaves; leaf++) {
                float lcfv[MAX_HANDS];
                leaf_eval_precomputed(&lprs0, &lprs1,
                    (const int(*)[2])lh0, ln0,
                    (const int(*)[2])lh1, ln1, lw1, lcfv);
            }

            free_prs(&lprs0);
            free_prs(&lprs1);
        }
        double seq_ms = (get_time_us() - seq_t0) / 1000.0;
        printf("  Sequential flop leaf eval (47 turns): %.0f ms\n", seq_ms);

        /* 6. Add DCFR cost for total flop solve */
        int rh0[MAX_HANDS][2], rh1[MAX_HANDS][2];
        float rw0[MAX_HANDS], rw1[MAX_HANDS];
        int rn0 = gen_hands(river_board, 5, rh0, rw0, N > 150 ? 150 : N);
        int rn1 = gen_hands(river_board, 5, rh1, rw1, N > 150 ? 150 : N);
        float bet_sizes[] = {0.33f, 0.75f};
        Solver s;
        solver_init(&s, river_board, 5,
                    (const int(*)[2])rh0, rw0, rn0,
                    (const int(*)[2])rh1, rw1, rn1,
                    1000, 9000, bet_sizes, 2);
        double dcfr_t0 = get_time_us();
        solver_solve(&s, 500, 0.0f);
        double dcfr_ms = (get_time_us() - dcfr_t0) / 1000.0;
        solver_free(&s);

        printf("\n  === TOTAL FLOP SOLVE (N=%d) ===\n", N);
        printf("  Sequential: leaf=%.0fms + dcfr=%.0fms = %.1f sec\n",
               seq_ms, dcfr_ms, (seq_ms + dcfr_ms) / 1000.0);
#ifdef _OPENMP
        printf("  Parallel:   leaf=%.0fms + dcfr=%.0fms = %.1f sec\n",
               omp_ms, dcfr_ms, (omp_ms + dcfr_ms) / 1000.0);
#endif
        printf("\n  === TOTAL TURN SOLVE (N=%d) ===\n", N);
        printf("  Precomputed: leaf=%.0fms + dcfr=%.0fms = %.0f ms\n",
               precomp_total + precomp_ms, dcfr_ms,
               precomp_total + precomp_ms + dcfr_ms);

        free_prs(&prs0);
        free_prs(&prs1);
        printf("\n");
    }

    return 0;
}
