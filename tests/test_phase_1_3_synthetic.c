/**
 * Phase 1.3 synthetic ground-truth test.
 *
 * Builds a minimal 2-player postflop toy game (1 hand per player, fixed flop,
 * 1 pot-sized bet, 100 chips start) where the true per-action EVs v̄(I,a)
 * under the blueprint's average strategy σ̄ can be independently computed in
 * Python by exhaustively enumerating the turn/river chance tree. This test:
 *
 *   1. Initializes the toy solver via bp_init.
 *   2. Trains to convergence via bp_solve (10M iterations — the tree is tiny).
 *   3. Runs bp_compute_action_evs (20M iterations) to populate action_evs[].
 *   4. Exports strategies (BPS3) and action EVs (BPR3) to a temporary .bps
 *      file at the path provided on the command line.
 *   5. Dumps a JSON summary of basic counts.
 *
 * The Python side (tests/test_phase_1_3_synthetic.py) then loads the .bps,
 * enumerates the game tree, and asserts:
 *   (a) |Σ σ̄[a]·v̄(I,a) - v̄(I)| < threshold for every visited info set
 *   (b) C-exported action_evs match Python-enumerated ground truth to
 *       within sampling noise
 *
 * Build:
 *   make test_phase_1_3
 * Run:
 *   ./build/test_phase_1_3 /tmp/phase_1_3_synthetic.bps
 *
 * If this test fails, Phase 1.3 has a bug in the math or the accumulator —
 * do NOT proceed to EC2 verification until it's green.
 */

#include "mccfr_blueprint.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

/* Card encoding: 0..51, rank = card>>2, suit = card&3.
 * Ranks: 0=2, 1=3, ..., 11=K, 12=A.  Suits: 0=c, 1=d, 2=h, 3=s. */
static int mkcard(int rank, int suit) { return rank * 4 + suit; }

static const char *rank_str(int r) {
    static const char *s[13] = {"2","3","4","5","6","7","8","9","T","J","Q","K","A"};
    return s[r];
}
static const char *suit_str(int s) {
    static const char *ss[4] = {"c","d","h","s"};
    return ss[s];
}
static void print_card(int c) {
    printf("%s%s", rank_str(c >> 2), suit_str(c & 3));
}

int main(int argc, char **argv) {
    const char *out_path = (argc > 1) ? argv[1] : "/tmp/phase_1_3_synthetic.bps";
    int64_t train_iters = (argc > 2) ? atoll(argv[2]) : 10000000;
    int64_t ev_iters    = (argc > 3) ? atoll(argv[3]) : 20000000;

    printf("=== Phase 1.3 synthetic ground-truth test ===\n");
    printf("Output: %s\n", out_path);
    printf("Train iters: %lld\n", (long long)train_iters);
    printf("EV walk iters: %lld\n", (long long)ev_iters);
    fflush(stdout);

    /* Toy game setup:
     *   2 players, postflop-only (include_preflop=0).
     *   Fixed flop: 2c 7d Kh (rainbow, no obvious draws).
     *   P0 hand: AsKs (top pair top kicker + nut flush draw... wait, Kh is
     *            on board so it's top pair with backdoor spade flush).
     *   P1 hand: 3c 3d (under-pair, small pocket pair).
     *   Starting pot: 100 chips, effective stack: 100 chips (1-SPR).
     *   Bet size: pot-sized only (1.0).
     *
     * Why these cards: deterministic per-runout showdown (no ties most of
     * the time), clear equity edge for P0 that narrows on certain turns
     * (3x, 7x pair the board for P1), providing varied per-action EV values
     * across the tree.
     *
     * Why 1 hand per player: with default identity bucket mapping and 1
     * hand, each player has exactly 1 bucket per street. The info set
     * count becomes purely a function of the betting tree shape, not the
     * card abstraction. Training converges fast on a tiny tree. */

    int flop[3] = {
        mkcard(0, 0),  /* 2c */
        mkcard(5, 1),  /* 7d */
        mkcard(11, 2), /* Kh */
    };

    int hands_p0[BP_MAX_HANDS][2];
    int hands_p1[BP_MAX_HANDS][2];
    memset(hands_p0, 0, sizeof(hands_p0));
    memset(hands_p1, 0, sizeof(hands_p1));
    hands_p0[0][0] = mkcard(12, 3); /* As */
    hands_p0[0][1] = mkcard(11, 3); /* Ks */
    hands_p1[0][0] = mkcard(1, 0);  /* 3c */
    hands_p1[0][1] = mkcard(1, 1);  /* 3d */

    /* Assemble into the [players][hands][2] shape bp_init expects. */
    int hands[BP_MAX_PLAYERS][BP_MAX_HANDS][2];
    memset(hands, 0, sizeof(hands));
    hands[0][0][0] = hands_p0[0][0];
    hands[0][0][1] = hands_p0[0][1];
    hands[1][0][0] = hands_p1[0][0];
    hands[1][0][1] = hands_p1[0][1];

    float weights[BP_MAX_PLAYERS][BP_MAX_HANDS];
    memset(weights, 0, sizeof(weights));
    weights[0][0] = 1.0f;
    weights[1][0] = 1.0f;

    int num_hands[BP_MAX_PLAYERS] = {1, 1, 0, 0, 0, 0};

    float bet_sizes[1] = {1.0f}; /* pot-sized bet */

    printf("\nFlop: "); print_card(flop[0]); printf(" "); print_card(flop[1]); printf(" "); print_card(flop[2]); printf("\n");
    printf("P0 hand: "); print_card(hands_p0[0][0]); printf(" "); print_card(hands_p0[0][1]); printf("\n");
    printf("P1 hand: "); print_card(hands_p1[0][0]); printf(" "); print_card(hands_p1[0][1]); printf("\n");
    printf("Starting pot: 100, effective stack: 100, bet size: 1.0x pot\n\n");
    fflush(stdout);

    /* Use bp_init_ex so we can set a real hash table size and multi-thread.
     * bp_init defaults to BP_HASH_SIZE_SMALL (4M slots) and 1 thread, which
     * is fine for the tree size but wastes wall time on training. */
    BPConfig config;
    bp_default_config(&config);
    config.include_preflop = 0;
    config.hash_table_size = (int64_t)(1 << 16); /* 64K slots — toy tree has <1000 info sets */
    config.num_threads = 0; /* auto (OpenMP max) */
    /* Disable pruning — on such a tiny tree pruning would starve some
     * actions of visits. Set thresholds impossibly high to effectively
     * disable. */
    config.prune_start_iter = train_iters * 2;
    /* Leave Linear CFR discount on the defaults — it's fine. */

    BPSolver s;
    memset(&s, 0, sizeof(s));

    int rc = bp_init_ex(&s, 2, flop,
                        (const int (*)[BP_MAX_HANDS][2])hands,
                        (const float (*)[BP_MAX_HANDS])weights,
                        num_hands,
                        100, 100,
                        bet_sizes, 1,
                        &config);
    if (rc != 0) {
        fprintf(stderr, "FATAL: bp_init_ex returned %d\n", rc);
        return 1;
    }
    printf("Solver initialized.\n"); fflush(stdout);

    /* Train. */
    printf("\nTraining %lld iterations...\n", (long long)train_iters); fflush(stdout);
    double t0 = (double)clock() / CLOCKS_PER_SEC;
    rc = bp_solve(&s, train_iters);
    double t1 = (double)clock() / CLOCKS_PER_SEC;
    if (rc != 0) {
        fprintf(stderr, "FATAL: bp_solve returned %d\n", rc);
        bp_free(&s);
        return 1;
    }
    int64_t n_is = bp_num_info_sets(&s);
    printf("Training done: %.1fs CPU, %lld info sets\n",
           t1 - t0, (long long)n_is);
    fflush(stdout);

    /* Compute per-action EVs. */
    printf("\nRunning σ̄-sampled EV walk (%lld iterations)...\n",
           (long long)ev_iters);
    fflush(stdout);
    t0 = (double)clock() / CLOCKS_PER_SEC;
    rc = bp_compute_action_evs(&s, ev_iters);
    t1 = (double)clock() / CLOCKS_PER_SEC;
    if (rc != 0) {
        fprintf(stderr, "FATAL: bp_compute_action_evs returned %d\n", rc);
        bp_free(&s);
        return 1;
    }
    printf("EV walk done: %.1fs CPU\n", t1 - t0);
    fflush(stdout);

    /* ── Stage 1 sanity checks (C-side, before export) ───────────────
     *
     * These are weak but necessary conditions on the EV walk's output.
     * They don't verify the math is CORRECT (that requires independent
     * enumeration in Python), but they catch the obvious failure modes:
     *
     *   1. No NaN or infinity in any normalized EV
     *   2. All normalized EVs within plausible chip range [-200, 200]
     *      (max stack + max pot swing in this toy game)
     *   3. Every info set with ev_visit_count > 0 has a non-NULL
     *      action_evs pointer (lazy allocator worked)
     *   4. σ̄-weighted sum of action EVs is bounded
     *   5. At least one visited info set exists at each street reached
     *
     * If any of these fail, Phase 1.3 has a bug and we stop immediately.
     * If they pass, we write the bundle; Python verifier does the strong
     * checks. */
    printf("\n── Stage 1: C-side sanity checks ──\n"); fflush(stdout);
    {
        const float MAX_CHIPS = 300.0f;  /* 200 is the hard ceiling; allow a little slack */
        int64_t n_visited = 0;
        int64_t n_zero_visits = 0;
        int64_t n_null_action_evs = 0;
        int64_t n_nan = 0;
        int64_t n_out_of_range = 0;
        int64_t n_weighted_out_of_range = 0;
        float max_abs_ev = 0.0f;
        float max_abs_weighted = 0.0f;
        int per_street_visited[4] = {0, 0, 0, 0};

        for (int64_t i = 0; i < s.info_table.table_size; i++) {
            if (s.info_table.occupied[i] != 1) continue;
            BPInfoSet *is = &s.info_table.sets[i];
            BPInfoKey *k = &s.info_table.keys[i];
            int na = is->num_actions;

            if (is->ev_visit_count == 0) { n_zero_visits++; continue; }
            n_visited++;
            if (k->street >= 0 && k->street < 4) per_street_visited[k->street]++;

            if (is->action_evs == NULL) { n_null_action_evs++; continue; }

            /* Normalize and check each action EV */
            float visits_f = (float)is->ev_visit_count;
            float norm_evs[BP_MAX_ACTIONS];
            for (int a = 0; a < na; a++) {
                norm_evs[a] = is->action_evs[a] / visits_f;
                /* NaN check: NaN != NaN */
                if (norm_evs[a] != norm_evs[a]) { n_nan++; continue; }
                float av = norm_evs[a] < 0 ? -norm_evs[a] : norm_evs[a];
                if (av > max_abs_ev) max_abs_ev = av;
                if (av > MAX_CHIPS) n_out_of_range++;
            }

            /* σ̄-weighted sum check */
            float strat[BP_MAX_ACTIONS];
            if (is->strategy_sum) {
                float ssum = 0;
                for (int a = 0; a < na; a++) {
                    float v = is->strategy_sum[a];
                    if (v < 0) v = 0;
                    strat[a] = v;
                    ssum += v;
                }
                if (ssum > 0) {
                    for (int a = 0; a < na; a++) strat[a] /= ssum;
                } else {
                    for (int a = 0; a < na; a++) strat[a] = 1.0f / na;
                }
            } else {
                for (int a = 0; a < na; a++) strat[a] = 1.0f / na;
            }

            float weighted = 0;
            for (int a = 0; a < na; a++) weighted += strat[a] * norm_evs[a];
            float wav = weighted < 0 ? -weighted : weighted;
            if (wav > max_abs_weighted) max_abs_weighted = wav;
            if (wav > MAX_CHIPS) n_weighted_out_of_range++;
        }

        printf("  total info sets:         %lld\n", (long long)n_is);
        printf("  with ev_visit_count>0:   %lld\n", (long long)n_visited);
        printf("  with zero visits:        %lld\n", (long long)n_zero_visits);
        printf("  visits per street:       flop=%d turn=%d river=%d\n",
               per_street_visited[1], per_street_visited[2], per_street_visited[3]);
        printf("  NULL action_evs (bug):   %lld\n", (long long)n_null_action_evs);
        printf("  NaN EVs:                 %lld\n", (long long)n_nan);
        printf("  |EV| > %.0f chips:          %lld\n", MAX_CHIPS, (long long)n_out_of_range);
        printf("  |Σσ̄·EV| > %.0f chips:       %lld\n", MAX_CHIPS, (long long)n_weighted_out_of_range);
        printf("  max |EV| observed:       %.2f\n", max_abs_ev);
        printf("  max |Σσ̄·EV| observed:    %.2f\n", max_abs_weighted);

        int fail = 0;
        if (n_visited == 0)                { fprintf(stderr, "FAIL: no info sets visited\n"); fail = 1; }
        if (n_null_action_evs > 0)         { fprintf(stderr, "FAIL: %lld visited info sets have NULL action_evs\n", (long long)n_null_action_evs); fail = 1; }
        if (n_nan > 0)                     { fprintf(stderr, "FAIL: %lld NaN EVs\n", (long long)n_nan); fail = 1; }
        if (n_out_of_range > 0)            { fprintf(stderr, "FAIL: %lld EVs out of chip range\n", (long long)n_out_of_range); fail = 1; }
        if (n_weighted_out_of_range > 0)   { fprintf(stderr, "FAIL: %lld σ̄-weighted sums out of chip range\n", (long long)n_weighted_out_of_range); fail = 1; }
        if (per_street_visited[1] == 0)    { fprintf(stderr, "FAIL: no flop info sets visited\n"); fail = 1; }

        if (fail) {
            fprintf(stderr, "\n=== STAGE 1 SANITY CHECKS FAILED ===\n");
            bp_free(&s);
            return 2;
        }
        printf("  ✓ All Stage 1 sanity checks passed.\n");
        fflush(stdout);
    }

    /* Export strategies to a buffer. */
    size_t strat_needed = 0;
    bp_export_strategies(&s, NULL, 0, &strat_needed);
    printf("\nStrategies buffer needed: %zu bytes\n", strat_needed);
    unsigned char *strat_buf = (unsigned char*)malloc(strat_needed);
    if (!strat_buf) { fprintf(stderr, "OOM strat buf\n"); bp_free(&s); return 1; }
    size_t strat_written = 0;
    bp_export_strategies(&s, strat_buf, strat_needed, &strat_written);
    printf("Strategies exported: %zu bytes\n", strat_written);

    /* Export action EVs to a buffer. */
    size_t ev_needed = 0;
    bp_export_action_evs(&s, NULL, 0, &ev_needed);
    printf("Action EVs buffer needed: %zu bytes\n", ev_needed);
    unsigned char *ev_buf = (unsigned char*)malloc(ev_needed);
    if (!ev_buf) { fprintf(stderr, "OOM ev buf\n"); free(strat_buf); bp_free(&s); return 1; }
    size_t ev_written = 0;
    bp_export_action_evs(&s, ev_buf, ev_needed, &ev_written);
    printf("Action EVs exported: %zu bytes\n", ev_written);

    /* Write raw (uncompressed) test bundle file. Format is test-specific,
     * NOT the production .bps format — the Python reader for this test
     * knows how to parse it. This keeps the test self-contained (no LZMA
     * dependency from the C side).
     *
     * Test bundle layout:
     *   magic "T13\0" (4B)
     *   u32 strategies_size
     *   u32 action_evs_size
     *   u32 num_players
     *   s32 flop[3]
     *   s32 p0_hand[2]
     *   s32 p1_hand[2]
     *   s32 starting_pot
     *   s32 effective_stack
     *   u32 num_bet_sizes (always 1 for this test)
     *   f32 bet_sizes[1]
     *   strategies_size bytes of BPS3-inner data (starts with "BPS3" magic)
     *   action_evs_size bytes of BPR3-inner data (starts with "BPR3" magic)
     */
    FILE *f = fopen(out_path, "wb");
    if (!f) {
        fprintf(stderr, "FATAL: cannot open %s for writing\n", out_path);
        free(strat_buf); free(ev_buf); bp_free(&s);
        return 1;
    }
    fwrite("T13\0", 1, 4, f);
    uint32_t ss32 = (uint32_t)strat_written;
    uint32_t es32 = (uint32_t)ev_written;
    uint32_t np32 = 2;
    fwrite(&ss32, sizeof(uint32_t), 1, f);
    fwrite(&es32, sizeof(uint32_t), 1, f);
    fwrite(&np32, sizeof(uint32_t), 1, f);
    int32_t flop32[3] = {flop[0], flop[1], flop[2]};
    fwrite(flop32, sizeof(int32_t), 3, f);
    int32_t p0_hand[2] = {hands[0][0][0], hands[0][0][1]};
    int32_t p1_hand[2] = {hands[1][0][0], hands[1][0][1]};
    fwrite(p0_hand, sizeof(int32_t), 2, f);
    fwrite(p1_hand, sizeof(int32_t), 2, f);
    int32_t spot = 100;
    int32_t estk = 100;
    fwrite(&spot, sizeof(int32_t), 1, f);
    fwrite(&estk, sizeof(int32_t), 1, f);
    uint32_t nbs = 1;
    fwrite(&nbs, sizeof(uint32_t), 1, f);
    float bs_f = 1.0f;
    fwrite(&bs_f, sizeof(float), 1, f);
    fwrite(strat_buf, 1, strat_written, f);
    fwrite(ev_buf, 1, ev_written, f);
    fclose(f);
    printf("\nWrote test bundle to %s (%zu bytes)\n",
           out_path, 4 + 3*sizeof(uint32_t) + 3*sizeof(int32_t) +
                       2*sizeof(int32_t) + 2*sizeof(int32_t) +
                       2*sizeof(int32_t) + sizeof(uint32_t) + sizeof(float) +
                       strat_written + ev_written);

    free(strat_buf);
    free(ev_buf);
    bp_free(&s);

    printf("\n=== C side complete. Run tests/test_phase_1_3_synthetic.py "
           "to verify. ===\n");
    return 0;
}
