/* Trace actual action hashes at each position's first decision.
 * Links against the solver DLL and runs 10 iterations, printing
 * every preflop info set creation with its hash. */
#include <stdio.h>
#include <string.h>
#include <stdint.h>

/* Reproduce the hash functions exactly */
static inline uint64_t hash_combine(uint64_t seed, uint64_t val) {
    return seed ^ (val * 0x9E3779B97F4A7C15ULL + 0x517CC1B727220A95ULL
                   + (seed << 12) + (seed >> 4));
}

/* Simulate: what hash does the solver compute for "UTG folds"?
 * Action index for fold = 0 (first in generate_actions when to_call > 0).
 *
 * But wait — when UTG is the traverser and explores fold (action a=0),
 * the child gets action_history[0] = 0. But what if the generate_actions
 * for UTG doesn't start with fold? Let's check all possibilities. */
int main(void) {
    uint64_t root = 0xFEDCBA9876543210ULL;

    printf("Root hash (no actions): %016llx\n\n", (unsigned long long)root);

    /* UTG preflop: to_call > 0 (faces BB), so fold IS action 0.
     * Actions: fold(0), call(1), raise_0(2), ..., raise_7(9), maybe allin(10)
     *
     * After UTG does action A, child history = [A], len = 1.
     * MP's info set key has action_hash = compute_action_hash([A], 1). */

    printf("After UTG takes action A, MP sees hash:\n");
    for (int a = 0; a <= 11; a++) {
        uint64_t h = root;
        h = hash_combine(h, (uint64_t)a * 17 + 3);
        printf("  A=%2d: %016llx", a, (unsigned long long)h);
        if (a == 0) printf("  (fold)");
        if (a == 1) printf("  (call)");
        if (a >= 2) printf("  (raise %d)", a-2);
        printf("\n");
    }

    printf("\nAfter UTG folds(0) + MP takes action B:\n");
    uint64_t h1 = hash_combine(root, (uint64_t)0 * 17 + 3);
    for (int b = 0; b <= 11; b++) {
        uint64_t h = hash_combine(h1, (uint64_t)b * 17 + 3);
        printf("  B=%2d: %016llx\n", b, (unsigned long long)h);
    }

    /* Alternative theory: maybe the action stored is NOT the index
     * but something else, like the BPAction.bet_idx or amount.
     * Let's check: in traverse(), line 1046/1119:
     *   child.action_history[child.history_len++] = a;  (traverser)
     *   child.action_history[child.history_len++] = sampled;  (non-traverser)
     * Both use the index 'a' or 'sampled' into the actions[] array.
     * This IS the local action index (0, 1, 2, ..., na-1). */

    /* Another theory: maybe there's a bug where history_len isn't incrementing
     * properly, or the memset zeroes are persisting. Let me check:
     * compute_action_hash with actions=[0], len=1:
     *   h = root
     *   h = hash_combine(h, 0*17+3) = hash_combine(h, 3)
     *
     * But compute_action_hash with actions=[0,0,...], len=0:
     *   h = root (no iterations)
     *
     * If history_len is stuck at 0, all nodes would have the root hash.
     * But we found UTG root entries, so at least UTG's lookup uses len=0. */

    /* Maybe the issue is simpler: UTG's info set is looked up BEFORE any
     * action, so its hash is root (len=0). After UTG takes action, the CHILD
     * has len=1. MP's info set uses the child's hash. This should work.
     *
     * Let me check: maybe the MP entries in the checkpoint just don't include
     * the "UTG folds" scenario because at 5M iterations, UTG's fold is rare
     * for some buckets. But we found ~480K MP preflop entries, so SOME
     * scenarios are explored. Just not the "UTG folds" one?
     *
     * At 5M iterations: UTG is traverser ~833K times. UTG explores all
     * actions including fold. But after fold, MP is a NON-TRAVERSER, so
     * MP's action is sampled and only ONE info set is created per traversal.
     * Over 833K iterations, that's 833K MP info sets created at the
     * "UTG folds" hash — across 169 buckets = ~4,930 per bucket.
     *
     * So the entries SHOULD exist. Unless they hash to the same slot as
     * other entries and got overwritten? No, the table uses open addressing
     * and is only 64% full. */

    /* Let me try a completely different theory: maybe the checkpoint only
     * saves entries with non-zero regrets, and the MP root entries haven't
     * been visited enough to develop non-zero regrets. But the save function
     * saves ALL occupied entries regardless of regret values. */

    /* Last theory: maybe there's a bug in my extract tool's checkpoint parsing.
     * If the BPR4 header is parsed wrong, all subsequent entries are shifted
     * and the hashes are garbage. But the UTG root WAS found correctly...
     * unless that was a coincidence.
     *
     * Let me verify: the UTG root has bucket=126, not bucket=0 (AA).
     * The find_hash tool only prints the first match. Bucket 126 is T2s.
     * This is expected — the scan hits T2s before AA in the hash table. */

    printf("\n");
    printf("If none of the above match the actual MP hashes (b9f0..., b9f1...),\n");
    printf("then the solver is building a longer history before MP acts.\n");
    printf("Possible: inactive player skipping adds to history? Or blind\n");
    printf("posting adds actions? Check the traverse function carefully.\n");

    return 0;
}
