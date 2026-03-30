/*
 * Stress test for lock-free hash table duplicate detection.
 * Spawns N threads, each inserting the same set of keys.
 * Verifies that num_entries matches the actual unique count.
 *
 * Build: gcc -O2 -pthread -o test_hashtable_dedup tests/test_hashtable_dedup.c src/mccfr_blueprint.c src/card_abstraction.c -I src -lm
 * Run:   ./test_hashtable_dedup
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <assert.h>
#include "mccfr_blueprint.h"

#define NUM_THREADS 32
#define NUM_KEYS 10000
#define TABLE_SIZE 65536  /* small table for testing */
#define ITERS_PER_THREAD 5  /* each thread inserts all keys this many times */

/* We need access to the internals. Declare the table struct directly. */
typedef struct {
    int player, street, bucket;
    uint64_t board_hash, action_hash;
} TestKey;

/* Global test table — we'll use the bp_* functions via the solver */
static BPInfoKey test_keys[NUM_KEYS];

static void init_test_keys(void) {
    for (int i = 0; i < NUM_KEYS; i++) {
        test_keys[i].player = i % 6;
        test_keys[i].street = (i / 6) % 4;
        test_keys[i].bucket = i % 200;
        test_keys[i].board_hash = (uint64_t)i * 2654435761ULL;
        test_keys[i].action_hash = (uint64_t)(i + 1) * 2246822519ULL;
    }
}

/* We need direct access to info_table_find_or_create.
 * Since it's static, we'll replicate the hash table here with
 * the same logic. Instead, let's just test via a simple wrapper. */

/* Minimal hash table implementation matching mccfr_blueprint.c */
typedef struct {
    BPInfoKey *keys;
    int *occupied;  /* 0=empty, 1=ready, 2=initializing */
    int table_size;
    int num_entries;
} TestTable;

static uint64_t hash_combine_test(uint64_t a, uint64_t b) {
    a ^= b + 0x9e3779b97f4a7c15ULL + (a << 6) + (a >> 2);
    return a;
}

static int key_eq_test(const BPInfoKey *a, const BPInfoKey *b) {
    return a->player == b->player && a->street == b->street &&
           a->bucket == b->bucket && a->board_hash == b->board_hash &&
           a->action_hash == b->action_hash;
}

static int table_find_or_create(TestTable *t, BPInfoKey key) {
    uint64_t h = hash_combine_test(key.board_hash, key.action_hash);
    h = hash_combine_test(h, (uint64_t)key.player);
    h = hash_combine_test(h, (uint64_t)key.street);
    h = hash_combine_test(h, (uint64_t)key.bucket);
    int slot = (int)(h % (uint64_t)t->table_size);

    for (int probe = 0; probe < 4096; probe++) {
        int idx = (slot + probe) % t->table_size;
        int state = __atomic_load_n(&t->occupied[idx], __ATOMIC_ACQUIRE);

        if (state == 1) {
            if (key_eq_test(&t->keys[idx], &key)) return idx;
            continue;
        }

        if (state == 0) {
            int expected = 0;
            if (__atomic_compare_exchange_n(&t->occupied[idx], &expected, 2,
                                            0, __ATOMIC_ACQ_REL,
                                            __ATOMIC_ACQUIRE)) {
                t->keys[idx] = key;
                __atomic_store_n(&t->occupied[idx], 1, __ATOMIC_RELEASE);

                /* De-dup: check for earlier copy */
                for (int p2 = 0; p2 < probe; p2++) {
                    int idx2 = (slot + p2) % t->table_size;
                    if (__atomic_load_n(&t->occupied[idx2], __ATOMIC_ACQUIRE) == 1) {
                        if (key_eq_test(&t->keys[idx2], &key)) {
                            return idx2;  /* duplicate — return earlier */
                        }
                    }
                }
                __atomic_fetch_add(&t->num_entries, 1, __ATOMIC_RELAXED);
                return idx;
            }
            state = __atomic_load_n(&t->occupied[idx], __ATOMIC_ACQUIRE);
        }

        if (state == 2) {
            int spins = 0;
            while (__atomic_load_n(&t->occupied[idx], __ATOMIC_ACQUIRE) == 2) {
                if (++spins > 1000000) break;
                #ifdef __x86_64__
                __builtin_ia32_pause();
                #endif
            }
            if (__atomic_load_n(&t->occupied[idx], __ATOMIC_ACQUIRE) == 1) {
                if (key_eq_test(&t->keys[idx], &key)) return idx;
            }
            continue;
        }
    }
    return -1;
}

static TestTable g_table;

typedef struct {
    int thread_id;
    int insertions;
} ThreadArg;

static void* thread_func(void *arg) {
    ThreadArg *ta = (ThreadArg*)arg;
    int count = 0;
    for (int iter = 0; iter < ITERS_PER_THREAD; iter++) {
        for (int i = 0; i < NUM_KEYS; i++) {
            int slot = table_find_or_create(&g_table, test_keys[i]);
            if (slot >= 0) count++;
        }
    }
    ta->insertions = count;
    return NULL;
}

int main(void) {
    printf("=== Hash Table De-dup Stress Test ===\n");
    printf("Threads: %d, Keys: %d, Table: %d slots, Iters/thread: %d\n",
           NUM_THREADS, NUM_KEYS, TABLE_SIZE, ITERS_PER_THREAD);

    init_test_keys();

    g_table.table_size = TABLE_SIZE;
    g_table.keys = calloc(TABLE_SIZE, sizeof(BPInfoKey));
    g_table.occupied = calloc(TABLE_SIZE, sizeof(int));
    g_table.num_entries = 0;

    pthread_t threads[NUM_THREADS];
    ThreadArg args[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; i++) {
        args[i].thread_id = i;
        args[i].insertions = 0;
        pthread_create(&threads[i], NULL, thread_func, &args[i]);
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    /* Count actual unique entries by scanning occupied slots */
    int actual_occupied = 0;
    for (int i = 0; i < TABLE_SIZE; i++) {
        if (g_table.occupied[i] == 1) actual_occupied++;
    }

    /* Count actual unique keys (some occupied slots may be duplicates) */
    int unique_keys = 0;
    for (int i = 0; i < TABLE_SIZE; i++) {
        if (g_table.occupied[i] != 1) continue;
        int is_dup = 0;
        for (int j = 0; j < i; j++) {
            if (g_table.occupied[j] == 1 && key_eq_test(&g_table.keys[j], &g_table.keys[i])) {
                is_dup = 1;
                break;
            }
        }
        if (!is_dup) unique_keys++;
    }

    int duplicates = actual_occupied - unique_keys;

    printf("\nResults:\n");
    printf("  num_entries counter: %d\n", g_table.num_entries);
    printf("  Actual occupied slots: %d\n", actual_occupied);
    printf("  Unique keys: %d (expected: %d)\n", unique_keys, NUM_KEYS);
    printf("  Duplicate slots: %d\n", duplicates);
    printf("  Counter accuracy: %.1f%% (should be ~100%%)\n",
           (double)g_table.num_entries / unique_keys * 100.0);

    int pass = 1;
    if (g_table.num_entries != unique_keys) {
        printf("\n  FAIL: num_entries (%d) != unique_keys (%d)\n",
               g_table.num_entries, unique_keys);
        pass = 0;
    }
    if (unique_keys != NUM_KEYS) {
        printf("\n  FAIL: unique_keys (%d) != expected (%d)\n",
               unique_keys, NUM_KEYS);
        pass = 0;
    }
    if (duplicates > NUM_KEYS / 100) {  /* allow <1% waste */
        printf("\n  WARN: %d duplicate slots (%.1f%% waste)\n",
               duplicates, (double)duplicates / actual_occupied * 100);
    }

    printf("\n%s\n", pass ? "ALL TESTS PASSED" : "TESTS FAILED");

    free(g_table.keys);
    free(g_table.occupied);
    return pass ? 0 : 1;
}
