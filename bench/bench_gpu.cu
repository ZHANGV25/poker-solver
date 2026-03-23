/**
 * bench_gpu.cu — Benchmark GPU solver on local RTX 3060
 *
 * Tests:
 * 1. GPU info query
 * 2. Single texture river solve on GPU
 * 3. Batch solve (100 textures)
 * 4. Compare GPU vs CPU timing
 */

#include "../src/cuda/gpu_solver.cuh"
#include "../src/hand_eval.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double get_time_ms(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* Build a simple river betting tree (same structure as CPU solver) */
static int build_simple_tree(GPUNode *tree) {
    int n = 0;

    /* Node 0: OOP decision (check/bet33/bet75/allin) */
    tree[n].type = GPU_NODE_DECISION;
    tree[n].player = 0;
    tree[n].pot = 1000;
    tree[n].bets[0] = 0; tree[n].bets[1] = 0;
    tree[n].num_actions = 3;
    tree[n].children[0] = 1;  /* check -> IP decision */
    tree[n].children[1] = 8;  /* bet 75% -> IP decision */
    tree[n].children[2] = 15; /* all-in -> IP decision */
    n++;

    /* Node 1: IP decision after check */
    tree[n].type = GPU_NODE_DECISION;
    tree[n].player = 1;
    tree[n].pot = 1000;
    tree[n].bets[0] = 0; tree[n].bets[1] = 0;
    tree[n].num_actions = 3;
    tree[n].children[0] = 2; /* check -> showdown */
    tree[n].children[1] = 3; /* bet 75% -> OOP decision */
    tree[n].children[2] = 6; /* all-in -> OOP decision */
    n++;

    /* Node 2: showdown (check-check) */
    tree[n].type = GPU_NODE_SHOWDOWN;
    tree[n].player = -1;
    tree[n].pot = 1000;
    tree[n].bets[0] = 0; tree[n].bets[1] = 0;
    tree[n].num_actions = 0;
    n++;

    /* Node 3: OOP facing bet 75% after check-bet */
    tree[n].type = GPU_NODE_DECISION;
    tree[n].player = 0;
    tree[n].pot = 1750;
    tree[n].bets[0] = 0; tree[n].bets[1] = 750;
    tree[n].num_actions = 2;
    tree[n].children[0] = 4; /* fold */
    tree[n].children[1] = 5; /* call */
    n++;

    /* Node 4: fold (IP wins) */
    tree[n].type = GPU_NODE_FOLD;
    tree[n].player = 1; /* IP wins */
    tree[n].pot = 1750;
    tree[n].bets[0] = 0; tree[n].bets[1] = 750;
    tree[n].num_actions = 0;
    n++;

    /* Node 5: call showdown */
    tree[n].type = GPU_NODE_SHOWDOWN;
    tree[n].player = -1;
    tree[n].pot = 2500;
    tree[n].bets[0] = 750; tree[n].bets[1] = 750;
    tree[n].num_actions = 0;
    n++;

    /* Node 6: OOP facing all-in after check */
    tree[n].type = GPU_NODE_DECISION;
    tree[n].player = 0;
    tree[n].pot = 10000;
    tree[n].bets[0] = 0; tree[n].bets[1] = 9000;
    tree[n].num_actions = 2;
    tree[n].children[0] = 4; /* fold (reuse) */
    tree[n].children[1] = 7; /* call */
    n++;

    /* Node 7: all-in showdown */
    tree[n].type = GPU_NODE_SHOWDOWN;
    tree[n].player = -1;
    tree[n].pot = 19000;
    tree[n].bets[0] = 9000; tree[n].bets[1] = 9000;
    tree[n].num_actions = 0;
    n++;

    /* Node 8: IP facing bet 75% */
    tree[n].type = GPU_NODE_DECISION;
    tree[n].player = 1;
    tree[n].pot = 1750;
    tree[n].bets[0] = 750; tree[n].bets[1] = 0;
    tree[n].num_actions = 2;
    tree[n].children[0] = 9;  /* fold */
    tree[n].children[1] = 10; /* call */
    n++;

    /* Node 9: fold (OOP wins) */
    tree[n].type = GPU_NODE_FOLD;
    tree[n].player = 0;
    tree[n].pot = 1750;
    tree[n].bets[0] = 750; tree[n].bets[1] = 0;
    tree[n].num_actions = 0;
    n++;

    /* Node 10: call showdown */
    tree[n].type = GPU_NODE_SHOWDOWN;
    tree[n].player = -1;
    tree[n].pot = 2500;
    tree[n].bets[0] = 750; tree[n].bets[1] = 750;
    tree[n].num_actions = 0;
    n++;

    /* Nodes 11-14: padding for tree[n].children references */
    for (int i = n; i < 16; i++) {
        tree[i].type = GPU_NODE_SHOWDOWN;
        tree[i].player = -1;
        tree[i].pot = 1000;
        tree[i].bets[0] = 0; tree[i].bets[1] = 0;
        tree[i].num_actions = 0;
    }

    /* Node 15: IP facing all-in */
    tree[15].type = GPU_NODE_DECISION;
    tree[15].player = 1;
    tree[15].pot = 10000;
    tree[15].bets[0] = 9000; tree[15].bets[1] = 0;
    tree[15].num_actions = 2;
    tree[15].children[0] = 16; /* fold */
    tree[15].children[1] = 17; /* call */

    tree[16].type = GPU_NODE_FOLD;
    tree[16].player = 0;
    tree[16].pot = 10000;
    tree[16].bets[0] = 9000; tree[16].bets[1] = 0;
    tree[16].num_actions = 0;

    tree[17].type = GPU_NODE_SHOWDOWN;
    tree[17].player = -1;
    tree[17].pot = 19000;
    tree[17].bets[0] = 9000; tree[17].bets[1] = 9000;
    tree[17].num_actions = 0;

    return 18;
}

/* Generate hands for a texture */
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
    printf("  GPU SOLVER BENCHMARK (RTX 3060)\n");
    printf("=============================================================\n\n");

    /* 1. GPU info */
    printf("1. GPU Info\n");
    int cores; size_t free_mem, total_mem;
    gpu_get_info(&cores, &free_mem, &total_mem);
    printf("\n");

    /* 2. Build tree */
    GPUNode tree[GPU_MAX_NODES];
    memset(tree, 0, sizeof(tree));
    int num_nodes = build_simple_tree(tree);
    printf("2. Tree: %d nodes\n\n", num_nodes);

    /* 3. Single texture solve */
    printf("3. Single texture river solve\n");
    {
        TextureData tex;
        memset(&tex, 0, sizeof(tex));
        tex.board[0] = parse_card("Qs"); tex.board[1] = parse_card("As");
        tex.board[2] = parse_card("2d"); tex.board[3] = parse_card("7h");
        tex.board[4] = parse_card("4c"); tex.num_board = 5;
        tex.starting_pot = 1000; tex.effective_stack = 9000;
        tex.num_nodes = num_nodes;

        int h0[GPU_MAX_HANDS][2], h1[GPU_MAX_HANDS][2];
        float w0[GPU_MAX_HANDS], w1[GPU_MAX_HANDS];
        tex.num_hands[0] = gen_hands(tex.board, 5, h0, w0, 80);
        tex.num_hands[1] = gen_hands(tex.board, 5, h1, w1, 80);
        memcpy(tex.hands[0], h0, sizeof(h0));
        memcpy(tex.hands[1], h1, sizeof(h1));
        memcpy(tex.weights[0], w0, sizeof(w0));
        memcpy(tex.weights[1], w1, sizeof(w1));

        /* Precompute strengths on CPU for comparison */
        for (int p = 0; p < 2; p++) {
            for (int h = 0; h < tex.num_hands[p]; h++) {
                int cards[7] = {tex.board[0],tex.board[1],tex.board[2],
                                tex.board[3],tex.board[4],
                                tex.hands[p][h][0],tex.hands[p][h][1]};
                tex.strengths[p][h] = eval7(cards);
            }
        }

        int out_stride = 2 * GPU_MAX_HANDS * GPU_MAX_ACTIONS;
        float *results = (float*)calloc(out_stride, sizeof(float));

        double t0 = get_time_ms();
        int err = gpu_solve_batch(&tex, 1, tree, num_nodes, 200, results);
        double elapsed = get_time_ms() - t0;

        if (err == 0) {
            printf("   Solved in %.0f ms (200 iterations)\n", elapsed);

            /* Show a few strategies */
            printf("   OOP root strategies (first 5 hands):\n");
            for (int h = 0; h < 5 && h < tex.num_hands[0]; h++) {
                char c0[3], c1[3];
                format_card(tex.hands[0][h][0], c0);
                format_card(tex.hands[0][h][1], c1);
                printf("     %s%s: ", c0, c1);
                for (int a = 0; a < 3; a++) {
                    float s = results[0 * GPU_MAX_HANDS * GPU_MAX_ACTIONS +
                                      h * GPU_MAX_ACTIONS + a];
                    printf("%.0f%% ", s * 100);
                }
                printf("\n");
            }
        } else {
            printf("   ERROR: gpu_solve_batch returned %d\n", err);
        }
        free(results);
    }
    printf("\n");

    /* 4. Batch solve */
    printf("4. Batch solve (100 textures, 200 iterations each)\n");
    {
        int batch_size = 100;
        TextureData *textures = (TextureData*)calloc(batch_size, sizeof(TextureData));

        /* Generate different boards */
        for (int t = 0; t < batch_size; t++) {
            textures[t].num_board = 5;
            textures[t].starting_pot = 1000;
            textures[t].effective_stack = 9000;
            textures[t].num_nodes = num_nodes;

            /* Vary the board slightly */
            int base_board[5] = {parse_card("Qs"), parse_card("As"), parse_card("2d"),
                                 parse_card("7h"), parse_card("4c")};
            /* Shift river card */
            int river = (parse_card("4c") + t * 2) % 52;
            /* Skip if conflicts with flop/turn */
            while (river == base_board[0] || river == base_board[1] ||
                   river == base_board[2] || river == base_board[3])
                river = (river + 1) % 52;
            base_board[4] = river;
            memcpy(textures[t].board, base_board, 5 * sizeof(int));

            int h[GPU_MAX_HANDS][2];
            float w[GPU_MAX_HANDS];
            textures[t].num_hands[0] = gen_hands(textures[t].board, 5, h, w, 60);
            memcpy(textures[t].hands[0], h, sizeof(h));
            memcpy(textures[t].weights[0], w, sizeof(w));
            textures[t].num_hands[1] = gen_hands(textures[t].board, 5, h, w, 60);
            memcpy(textures[t].hands[1], h, sizeof(h));
            memcpy(textures[t].weights[1], w, sizeof(w));

            /* Precompute strengths */
            for (int p = 0; p < 2; p++) {
                for (int hi = 0; hi < textures[t].num_hands[p]; hi++) {
                    int cards[7];
                    for (int i = 0; i < 5; i++) cards[i] = textures[t].board[i];
                    cards[5] = textures[t].hands[p][hi][0];
                    cards[6] = textures[t].hands[p][hi][1];
                    textures[t].strengths[p][hi] = eval7(cards);
                }
            }
        }

        int out_stride = 2 * GPU_MAX_HANDS * GPU_MAX_ACTIONS;
        float *results = (float*)calloc(batch_size * out_stride, sizeof(float));

        double t0 = get_time_ms();
        int err = gpu_solve_batch(textures, batch_size, tree, num_nodes, 200, results);
        double elapsed = get_time_ms() - t0;

        if (err == 0) {
            printf("   %d textures solved in %.0f ms (%.1f ms/texture)\n",
                   batch_size, elapsed, elapsed / batch_size);
            printf("   Throughput: %.0f textures/sec\n", batch_size / (elapsed/1000));
        } else {
            printf("   ERROR: %d\n", err);
        }

        free(textures);
        free(results);
    }

    printf("\n=============================================================\n");
    printf("  BENCHMARK COMPLETE\n");
    printf("=============================================================\n");
    return 0;
}
