/**
 * card_abstraction.h — EHS computation + k-means bucketing for card abstraction
 *
 * Computes Expected Hand Strength (EHS) for poker hands via Monte Carlo
 * sampling, then clusters into buckets using k-means on domain features.
 *
 * Pluribus uses 169 lossless preflop buckets and 200 lossy buckets
 * per postflop street, clustered via k-means on domain-specific features
 * (EHS + hand potential), per Johanson et al. 2013.
 *
 * Two bucketing methods available:
 *   ca_assign_buckets()        — fast percentile bucketing (1D, EHS only)
 *   ca_assign_buckets_kmeans() — k-means on [EHS, positive_potential, negative_potential]
 */
#ifndef CARD_ABSTRACTION_H
#define CARD_ABSTRACTION_H

#include <stdint.h>

#define CA_MAX_HANDS 1326    /* max possible 2-card hands from 52 cards */

#ifdef _WIN32
#define CA_EXPORT __declspec(dllexport)
#else
#define CA_EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Compute Expected Hand Strength for a set of hands on a given board.
 *
 * EHS = (wins + 0.5*ties) / total against random opponent hands,
 * averaged over random board completions (for flop/turn).
 *
 * Args:
 *   board: board cards (3=flop, 4=turn, 5=river)
 *   num_board: number of board cards
 *   hands: array of (c0, c1) pairs
 *   num_hands: number of hands
 *   n_samples: Monte Carlo samples per hand (more = more accurate)
 *   ehs_out: [num_hands] output array of EHS values in [0, 1]
 *
 * Returns 0 on success.
 */
CA_EXPORT int ca_compute_ehs(
    const int *board, int num_board,
    const int hands[][2], int num_hands,
    int n_samples,
    float *ehs_out
);

/**
 * Assign hands to buckets via percentile bucketing on EHS.
 *
 * Sorts hands by EHS, divides into num_buckets equal-population groups.
 * Bucket 0 = weakest hands, bucket num_buckets-1 = strongest.
 *
 * Args:
 *   ehs: [num_hands] EHS values (from ca_compute_ehs)
 *   num_hands: number of hands
 *   num_buckets: target number of buckets
 *   bucket_out: [num_hands] output bucket index for each hand
 *
 * Returns actual number of buckets (may be < num_buckets if fewer hands).
 */
CA_EXPORT int ca_assign_buckets(
    const float *ehs, int num_hands,
    int num_buckets,
    int *bucket_out
);

/**
 * Generate all 1326 possible 2-card hands from a 52-card deck,
 * excluding cards in the given board.
 *
 * Args:
 *   board: cards to exclude
 *   num_board: number of board cards
 *   hands_out: [CA_MAX_HANDS][2] output array
 *
 * Returns number of hands generated.
 */
CA_EXPORT int ca_generate_hands(
    const int *board, int num_board,
    int hands_out[][2]
);

/**
 * Generate the 169 strategically unique preflop hand classes.
 * Each class is (rank1, rank2, suited), mapped to a representative hand.
 *
 * Args:
 *   classes_out: [169][2] representative (c0, c1) for each class
 *   hand_to_class: [1326] maps each of the 1326 hands to its class index
 *   hands: [1326][2] all possible hands (from ca_generate_hands with empty board)
 *   num_hands: 1326
 *
 * Returns 169.
 */
/**
 * Assign hands to buckets via k-means clustering on domain features.
 *
 * Features per hand (Johanson et al. 2013 / Pluribus):
 *   [0] EHS — expected hand strength vs random opponent
 *   [1] Positive potential — P(behind now but ahead after more cards)
 *   [2] Negative potential — P(ahead now but behind after more cards)
 *
 * K-means with Lloyd's algorithm, 20 iterations, percentile-seeded centroids.
 *
 * Args:
 *   board, num_board: current board cards
 *   hands: array of (c0, c1) pairs
 *   num_hands: number of hands
 *   num_buckets: target k
 *   n_samples: Monte Carlo samples for EHS + potential computation
 *   bucket_out: [num_hands] output bucket index for each hand
 *
 * Returns actual number of buckets used.
 */
CA_EXPORT int ca_assign_buckets_kmeans(
    const int *board, int num_board,
    const int hands[][2], int num_hands,
    int num_buckets, int n_samples,
    int *bucket_out
);

/**
 * Compute [EHS, PPot, NPot] feature vectors for a set of hands.
 * Used for k-means bucketing and centroid precomputation.
 */
CA_EXPORT void ca_compute_features(
    const int *board, int num_board,
    const int hands[][2], int num_hands,
    int n_samples,
    float features[][3]
);

/**
 * Find the nearest centroid for a single 3D feature vector.
 * Returns the centroid index (0 to k-1).
 */
static inline int ca_nearest_centroid(
    const float feat[3], const float centroids[][3], int k
) {
    float best_dist = 1e30f;
    int best = 0;
    for (int c = 0; c < k; c++) {
        float d0 = feat[0] - centroids[c][0];
        float d1 = feat[1] - centroids[c][1];
        float d2 = feat[2] - centroids[c][2];
        float dist = d0*d0 + d1*d1 + d2*d2;
        if (dist < best_dist) { best_dist = dist; best = c; }
    }
    return best;
}

CA_EXPORT int ca_preflop_classes(
    int classes_out[][2],
    int *hand_to_class,
    const int hands[][2], int num_hands
);

#ifdef __cplusplus
}
#endif

#endif /* CARD_ABSTRACTION_H */
