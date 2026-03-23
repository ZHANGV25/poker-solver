/**
 * bench_local_precompute.cu — Can we precompute locally on RTX 3060?
 *
 * Tests: solve 500 textures in one batch (simulating ~1/3 of a scenario)
 * If this takes <10 seconds, local precompute is feasible.
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

/* Build simple tree (18 nodes, check/bet75/allin) */
static int build_tree(GPUNode *tree) {
    memset(tree, 0, GPU_MAX_NODES * sizeof(GPUNode));
    /* Node 0: OOP check/bet/allin */
    tree[0].type=GPU_NODE_DECISION; tree[0].player=0; tree[0].num_actions=3;
    tree[0].children[0]=1; tree[0].children[1]=8; tree[0].children[2]=15;
    tree[0].pot=1000; tree[0].bets[0]=0; tree[0].bets[1]=0;
    /* Node 1: IP after check */
    tree[1].type=GPU_NODE_DECISION; tree[1].player=1; tree[1].num_actions=3;
    tree[1].children[0]=2; tree[1].children[1]=3; tree[1].children[2]=6;
    tree[1].pot=1000;
    /* Node 2: check-check showdown */
    tree[2].type=GPU_NODE_SHOWDOWN; tree[2].player=-1; tree[2].pot=1000;
    /* Node 3: OOP facing bet after check */
    tree[3].type=GPU_NODE_DECISION; tree[3].player=0; tree[3].num_actions=2;
    tree[3].children[0]=4; tree[3].children[1]=5;
    tree[3].pot=1750; tree[3].bets[0]=0; tree[3].bets[1]=750;
    tree[4].type=GPU_NODE_FOLD; tree[4].player=1; tree[4].pot=1750; tree[4].bets[0]=0; tree[4].bets[1]=750;
    tree[5].type=GPU_NODE_SHOWDOWN; tree[5].player=-1; tree[5].pot=2500; tree[5].bets[0]=750; tree[5].bets[1]=750;
    /* Node 6: OOP facing allin after check */
    tree[6].type=GPU_NODE_DECISION; tree[6].player=0; tree[6].num_actions=2;
    tree[6].children[0]=4; tree[6].children[1]=7;
    tree[6].pot=10000; tree[6].bets[0]=0; tree[6].bets[1]=9000;
    tree[7].type=GPU_NODE_SHOWDOWN; tree[7].player=-1; tree[7].pot=19000; tree[7].bets[0]=9000; tree[7].bets[1]=9000;
    /* Node 8: IP facing bet */
    tree[8].type=GPU_NODE_DECISION; tree[8].player=1; tree[8].num_actions=2;
    tree[8].children[0]=9; tree[8].children[1]=10;
    tree[8].pot=1750; tree[8].bets[0]=750; tree[8].bets[1]=0;
    tree[9].type=GPU_NODE_FOLD; tree[9].player=0; tree[9].pot=1750; tree[9].bets[0]=750; tree[9].bets[1]=0;
    tree[10].type=GPU_NODE_SHOWDOWN; tree[10].player=-1; tree[10].pot=2500; tree[10].bets[0]=750; tree[10].bets[1]=750;
    /* Padding */
    for (int i=11; i<15; i++) { tree[i].type=GPU_NODE_SHOWDOWN; tree[i].player=-1; tree[i].pot=1000; }
    /* Node 15: IP facing allin */
    tree[15].type=GPU_NODE_DECISION; tree[15].player=1; tree[15].num_actions=2;
    tree[15].children[0]=16; tree[15].children[1]=17;
    tree[15].pot=10000; tree[15].bets[0]=9000; tree[15].bets[1]=0;
    tree[16].type=GPU_NODE_FOLD; tree[16].player=0; tree[16].pot=10000; tree[16].bets[0]=9000;
    tree[17].type=GPU_NODE_SHOWDOWN; tree[17].player=-1; tree[17].pot=19000; tree[17].bets[0]=9000; tree[17].bets[1]=9000;
    return 18;
}

int main(void) {
    printf("=============================================================\n");
    printf("  LOCAL PRECOMPUTE FEASIBILITY TEST\n");
    printf("=============================================================\n\n");

    int cores; size_t free_mem, total_mem;
    gpu_get_info(&cores, &free_mem, &total_mem);

    GPUNode tree[GPU_MAX_NODES];
    int num_nodes = build_tree(tree);

    /* Test batch sizes: 100, 500, 1000, 1755 */
    int batch_sizes[] = {100, 500, 1000, 1755};
    int iters = 200;

    for (int bi = 0; bi < 4; bi++) {
        int batch = batch_sizes[bi];

        /* Check memory */
        size_t mem_per_tex = num_nodes * GPU_MAX_ACTIONS * GPU_MAX_HANDS * sizeof(float) * 2 +
                             num_nodes * GPU_MAX_HANDS * sizeof(float) * 2 +
                             sizeof(TextureData);
        size_t total_gpu_mem = batch * mem_per_tex;
        if (total_gpu_mem > free_mem * 0.9) {
            printf("Batch %d: would need %.1f GB, only %.1f GB free — SKIP\n\n",
                   batch, total_gpu_mem/1e9, free_mem/1e9);
            continue;
        }

        TextureData *textures = (TextureData*)calloc(batch, sizeof(TextureData));
        for (int t = 0; t < batch; t++) {
            textures[t].num_board = 5;
            textures[t].starting_pot = 1000;
            textures[t].effective_stack = 9000;
            int base[5] = {parse_card("Qs"), parse_card("As"), parse_card("2d"),
                           parse_card("7h"), parse_card("4c")};
            int river = (base[4] + t * 2) % 52;
            while (river==base[0]||river==base[1]||river==base[2]||river==base[3])
                river = (river+1)%52;
            base[4] = river;
            memcpy(textures[t].board, base, 5*sizeof(int));
            int h[GPU_MAX_HANDS][2]; float w[GPU_MAX_HANDS];
            textures[t].num_hands[0] = gen_hands(textures[t].board, 5, h, w, 80);
            memcpy(textures[t].hands[0], h, sizeof(h)); memcpy(textures[t].weights[0], w, sizeof(w));
            textures[t].num_hands[1] = gen_hands(textures[t].board, 5, h, w, 80);
            memcpy(textures[t].hands[1], h, sizeof(h)); memcpy(textures[t].weights[1], w, sizeof(w));
            for (int p=0;p<2;p++) for (int hi=0;hi<textures[t].num_hands[p];hi++) {
                int cards[7]; for(int i=0;i<5;i++)cards[i]=textures[t].board[i];
                cards[5]=textures[t].hands[p][hi][0]; cards[6]=textures[t].hands[p][hi][1];
                textures[t].strengths[p][hi]=eval7(cards);
            }
        }

        int out_stride = 2 * GPU_MAX_HANDS * GPU_MAX_ACTIONS;
        float *results = (float*)calloc(batch * out_stride, sizeof(float));

        double t0 = get_time_ms();
        gpu_solve_batch(textures, batch, tree, num_nodes, iters, results);
        double elapsed = get_time_ms() - t0;

        printf("Batch %d textures × %d iterations:\n", batch, iters);
        printf("  Time: %.1f sec (%.1f ms/texture)\n", elapsed/1000, elapsed/batch);
        printf("  Throughput: %.0f textures/sec\n", batch/(elapsed/1000));
        printf("  1 scenario (1755 tex): %.1f sec\n", 1755.0 / (batch/(elapsed/1000)));
        printf("  27 scenarios: %.1f min\n\n", 27 * 1755.0 / (batch/(elapsed/1000)) / 60);

        free(textures);
        free(results);
    }

    printf("=============================================================\n");
    return 0;
}
