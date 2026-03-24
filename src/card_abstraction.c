/**
 * card_abstraction.c — Fast EHS computation + percentile bucketing
 *
 * Uses hand_eval.h for 7-card evaluation via Monte Carlo sampling.
 * Designed for speed: ~1000 hands × 1000 samples = ~1M eval7 calls,
 * each taking ~200ns = ~0.2 seconds total on modern CPU.
 */

#include "card_abstraction.h"
#include "hand_eval.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ── RNG ──────────────────────────────────────────────────────────── */

static inline uint64_t ca_rng_next(uint64_t *state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

static inline int ca_rng_int(uint64_t *state, int n) {
    return (int)(ca_rng_next(state) % (uint64_t)n);
}

/* Fisher-Yates partial shuffle: pick k items from array of n */
static void partial_shuffle(int *arr, int n, int k, uint64_t *rng) {
    for (int i = 0; i < k && i < n; i++) {
        int j = i + ca_rng_int(rng, n - i);
        int tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
    }
}

/* ── EHS computation ─────────────────────────────────────────────── */

int ca_compute_ehs(
    const int *board, int num_board,
    const int hands[][2], int num_hands,
    int n_samples,
    float *ehs_out
) {
    if (num_hands <= 0 || n_samples <= 0) return -1;

    /* Build blocked card set from board */
    int board_blocked[52] = {0};
    for (int i = 0; i < num_board; i++)
        board_blocked[board[i]] = 1;

    uint64_t rng_state = 0xCAFEBABE12345678ULL;

    for (int h = 0; h < num_hands; h++) {
        int c0 = hands[h][0], c1 = hands[h][1];

        /* Cards blocked by this hand + board */
        int blocked[52];
        memcpy(blocked, board_blocked, sizeof(blocked));
        blocked[c0] = 1;
        blocked[c1] = 1;

        /* Available cards for opponent + board completion */
        int avail[52];
        int n_avail = 0;
        for (int c = 0; c < 52; c++)
            if (!blocked[c]) avail[n_avail++] = c;

        int wins = 0, ties = 0, total = 0;
        int cards_needed = 2 + (5 - num_board); /* 2 opp + remaining board */

        for (int s = 0; s < n_samples; s++) {
            if (n_avail < cards_needed) break;

            /* Shuffle and pick: first 2 = opponent, rest = board completion */
            partial_shuffle(avail, n_avail, cards_needed, &rng_state);

            int oc0 = avail[0], oc1 = avail[1];

            /* Build full 5-card board */
            int full_board[5];
            int nb_copy = (num_board <= 5) ? num_board : 5;
            memcpy(full_board, board, nb_copy * sizeof(int));
            for (int b = num_board; b < 5; b++)
                full_board[b] = avail[2 + (b - num_board)];

            /* Evaluate both hands */
            int hero_cards[7] = {full_board[0], full_board[1], full_board[2],
                                  full_board[3], full_board[4], c0, c1};
            int opp_cards[7] = {full_board[0], full_board[1], full_board[2],
                                 full_board[3], full_board[4], oc0, oc1};

            uint32_t hero_str = eval7(hero_cards);
            uint32_t opp_str = eval7(opp_cards);

            if (hero_str > opp_str) wins++;
            else if (hero_str == opp_str) ties++;
            total++;
        }

        ehs_out[h] = (total > 0) ? ((float)wins + 0.5f * (float)ties) / (float)total : 0.5f;
    }

    return 0;
}

/* ── Percentile bucketing ────────────────────────────────────────── */

/* Comparison function for sorting by EHS */
typedef struct { int idx; float ehs; } EhsPair;

static int cmp_ehs(const void *a, const void *b) {
    float ea = ((const EhsPair*)a)->ehs;
    float eb = ((const EhsPair*)b)->ehs;
    if (ea < eb) return -1;
    if (ea > eb) return 1;
    return 0;
}

int ca_assign_buckets(
    const float *ehs, int num_hands,
    int num_buckets,
    int *bucket_out
) {
    if (num_hands <= 0) return 0;
    int actual = (num_buckets < num_hands) ? num_buckets : num_hands;

    /* Sort by EHS */
    EhsPair *pairs = (EhsPair*)malloc(num_hands * sizeof(EhsPair));
    for (int i = 0; i < num_hands; i++) {
        pairs[i].idx = i;
        pairs[i].ehs = ehs[i];
    }
    qsort(pairs, num_hands, sizeof(EhsPair), cmp_ehs);

    /* Assign buckets: equal-population percentile */
    for (int rank = 0; rank < num_hands; rank++) {
        int bucket = rank * actual / num_hands;
        if (bucket >= actual) bucket = actual - 1;
        bucket_out[pairs[rank].idx] = bucket;
    }

    free(pairs);
    return actual;
}

/* ── Hand generation ─────────────────────────────────────────────── */

int ca_generate_hands(
    const int *board, int num_board,
    int hands_out[][2]
) {
    int blocked[52] = {0};
    for (int i = 0; i < num_board; i++)
        blocked[board[i]] = 1;

    int n = 0;
    for (int c0 = 0; c0 < 52; c0++) {
        if (blocked[c0]) continue;
        for (int c1 = c0 + 1; c1 < 52; c1++) {
            if (blocked[c1]) continue;
            if (n >= CA_MAX_HANDS) goto done;
            hands_out[n][0] = c0;
            hands_out[n][1] = c1;
            n++;
        }
    }
done:
    return n;
}

/* ── Preflop hand classes (169 strategically unique) ─────────────── */

int ca_preflop_classes(
    int classes_out[][2],
    int *hand_to_class,
    const int hands[][2], int num_hands
) {
    /* 169 classes: 13 pairs + 78 suited + 78 offsuit
     * Encoding: pairs at 0-12, suited at 13-90, offsuit at 91-168
     * Class = f(rank0, rank1, suited) */
    int n_classes = 0;
    int class_map[13][13][2]; /* [rank_hi][rank_lo][suited] = class_idx */
    memset(class_map, -1, sizeof(class_map));

    /* Build class map */
    for (int r0 = 12; r0 >= 0; r0--) {
        for (int r1 = r0; r1 >= 0; r1--) {
            if (r0 == r1) {
                /* Pair */
                class_map[r0][r1][0] = n_classes;
                class_map[r0][r1][1] = n_classes; /* same class for suited/offsuit pair */
                classes_out[n_classes][0] = MAKE_CARD(r0, 0);
                classes_out[n_classes][1] = MAKE_CARD(r1, 1);
                n_classes++;
            } else {
                /* Suited */
                class_map[r0][r1][1] = n_classes;
                classes_out[n_classes][0] = MAKE_CARD(r0, 0);
                classes_out[n_classes][1] = MAKE_CARD(r1, 0);
                n_classes++;
                /* Offsuit */
                class_map[r0][r1][0] = n_classes;
                classes_out[n_classes][0] = MAKE_CARD(r0, 0);
                classes_out[n_classes][1] = MAKE_CARD(r1, 1);
                n_classes++;
            }
        }
    }

    /* Map each hand to its class */
    for (int h = 0; h < num_hands; h++) {
        int r0 = CARD_RANK(hands[h][0]);
        int r1 = CARD_RANK(hands[h][1]);
        int suited = (CARD_SUIT(hands[h][0]) == CARD_SUIT(hands[h][1])) ? 1 : 0;
        /* Ensure r0 >= r1 */
        if (r0 < r1) { int tmp = r0; r0 = r1; r1 = tmp; }
        hand_to_class[h] = class_map[r0][r1][suited];
    }

    return n_classes;
}
