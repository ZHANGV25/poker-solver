/**
 * flop_accel.cuh — GPU-accelerated flop leaf evaluation
 *
 * At flop CHANCE nodes, instead of CPU iterating over 49×46 runouts,
 * launch a single GPU kernel that computes per-hand CFV across ALL
 * runouts in parallel.
 *
 * Each thread: one (traverser_hand, turn_card, river_card) combination
 * Thread block: one turn card, threads handle river cards × hands
 *
 * Input:  flop board (3 cards), both players' hands + reach probabilities
 * Output: per-hand CFV for the traverser at the chance node
 *
 * This replaces the O(H × T × R × H_opp) CPU loop with a GPU kernel
 * where H=hands, T=turn cards, R=river cards, H_opp=opponent hands.
 */
#ifndef FLOP_ACCEL_CUH
#define FLOP_ACCEL_CUH

#include <stdint.h>

#define FA_MAX_HANDS 400
#define FA_MAX_BOARD 5

typedef struct {
    /* Flop board (3 cards) */
    int board[3];

    /* Both players' hands */
    int hands[2][FA_MAX_HANDS][2];
    float reach[2][FA_MAX_HANDS];  /* current reach probabilities */
    int num_hands[2];

    /* Pot at the chance node */
    float half_pot;

    /* Which player is the traverser */
    int traverser;
} FlopAccelInput;

typedef struct {
    /* Per-hand CFV for the traverser, averaged over all runouts */
    float cfv[FA_MAX_HANDS];
} FlopAccelOutput;

#ifdef _WIN32
#define FA_EXPORT __declspec(dllexport)
#else
#define FA_EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

FA_EXPORT int flop_accel_eval(const FlopAccelInput *input, FlopAccelOutput *output);
FA_EXPORT int flop_accel_init(void);
FA_EXPORT void flop_accel_cleanup(void);

#ifdef __cplusplus
}
#endif

#endif /* FLOP_ACCEL_CUH */
