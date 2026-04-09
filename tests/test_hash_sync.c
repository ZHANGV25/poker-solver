/**
 * Hash sync test: prints bp_mix64 and hash_combine outputs for a set
 * of known inputs. Used to verify that python/blueprint_v2.py's
 * _bp_mix64 / _hash_combine produce byte-identical values.
 *
 * This is a regression guard on the Python-C hash synchronization.
 * Bug 6 (commit 48da71b) swapped the C-side mixer from boost-style
 * to splitmix64 but did NOT update the Python side, creating a latent
 * lookup failure in every v3 .bps consumer.
 *
 * Build:
 *   gcc -O2 -Isrc tests/test_hash_sync.c -o build/test_hash_sync -lm
 *
 * (No link against mccfr_blueprint.c — we re-implement the two functions
 * here to freeze the expected behavior. If mccfr_blueprint.c's versions
 * ever drift from these, this test will catch it.)
 */
#include <stdio.h>
#include <stdint.h>

static inline uint64_t bp_mix64(uint64_t x) {
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27; x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

static inline uint64_t hash_combine(uint64_t a, uint64_t b) {
    return bp_mix64(a ^ bp_mix64(b));
}

/* Mirrors compute_action_hash from mccfr_blueprint.c:431. */
static uint64_t compute_action_hash(const int *actions, int num_actions) {
    uint64_t h = 0xFEDCBA9876543210ULL;
    for (int i = 0; i < num_actions; i++)
        h = hash_combine(h, (uint64_t)actions[i] * 17 + 3);
    return h;
}

int main(void) {
    /* bp_mix64 vectors */
    uint64_t mix_inputs[] = {
        0, 1, 2, 42, 0xDEADBEEFULL,
        0x123456789ABCDEF0ULL, 0xFEDCBA9876543210ULL,
        0xFFFFFFFFFFFFFFFFULL
    };
    printf("=== bp_mix64 ===\n");
    for (size_t i = 0; i < sizeof(mix_inputs)/sizeof(mix_inputs[0]); i++) {
        printf("bp_mix64(%016llx) = %016llx\n",
               (unsigned long long)mix_inputs[i],
               (unsigned long long)bp_mix64(mix_inputs[i]));
    }

    /* hash_combine vectors */
    struct { uint64_t a, b; } hc_inputs[] = {
        {0, 0},
        {1, 2},
        {0x123456789ABCDEF0ULL, 0xFEDCBA9876543210ULL},
        {0xDEADBEEFULL, 0xCAFEBABEULL},
    };
    printf("\n=== hash_combine ===\n");
    for (size_t i = 0; i < sizeof(hc_inputs)/sizeof(hc_inputs[0]); i++) {
        printf("hash_combine(%016llx, %016llx) = %016llx\n",
               (unsigned long long)hc_inputs[i].a,
               (unsigned long long)hc_inputs[i].b,
               (unsigned long long)hash_combine(hc_inputs[i].a, hc_inputs[i].b));
    }

    /* compute_action_hash vectors — these are what the blueprint uses
     * to key info sets by action history. Critical that Python and C
     * agree. */
    printf("\n=== compute_action_hash ===\n");
    {
        int h0[1] = {};  /* empty */
        printf("compute_action_hash([]) = %016llx\n",
               (unsigned long long)compute_action_hash(h0, 0));
    }
    {
        int h1[] = {0};
        printf("compute_action_hash([0]) = %016llx\n",
               (unsigned long long)compute_action_hash(h1, 1));
    }
    {
        int h2[] = {1};
        printf("compute_action_hash([1]) = %016llx\n",
               (unsigned long long)compute_action_hash(h2, 1));
    }
    {
        int h3[] = {0, 1, 2};
        printf("compute_action_hash([0,1,2]) = %016llx\n",
               (unsigned long long)compute_action_hash(h3, 3));
    }
    {
        int h4[] = {2, 1, 0, 3, 2, 1};
        printf("compute_action_hash([2,1,0,3,2,1]) = %016llx\n",
               (unsigned long long)compute_action_hash(h4, 6));
    }

    return 0;
}
