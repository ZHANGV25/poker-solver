/**
 * sanity_test.cu — GPU vs CPU sanity check with known hands
 *
 * Uses the 4-hand toy game where we know the correct answers:
 *   OOP: AhKh(TPTK), QhQc(trips), JhTh(JT), 6h5h(air)
 *   IP: AcKc(TPTK), 3c3d(pair3), Tc9c(T9), 8c8d(pair8)
 *   Board: Qs As 2d 7h 4c
 *
 * Expected: QQ bets most, AK bets often, 65 checks always
 */
#include "../src/cuda/gpu_solver.cuh"
#include "../src/hand_eval.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Build tree matching CPU solver_v2 output (27 nodes) */
static int build_tree(GPUNode *tree) {
    memset(tree, 0, GPU_MAX_NODES * sizeof(GPUNode));

    /* Helper macro for MSVC compatibility (no compound literals in C++) */
    #define SET_NODE(idx, t, p, na, c0, c1, c2, c3, pot_v, b0, b1) \
        tree[idx].type=t; tree[idx].player=p; tree[idx].num_actions=na; \
        tree[idx].children[0]=c0; tree[idx].children[1]=c1; \
        tree[idx].children[2]=c2; tree[idx].children[3]=c3; \
        tree[idx].pot=pot_v; tree[idx].bets[0]=b0; tree[idx].bets[1]=b1;

    SET_NODE(0,  GPU_NODE_DECISION, 0, 3, 1,15,24,0, 1000, 0,0);
    SET_NODE(1,  GPU_NODE_DECISION, 1, 3, 2,3,12,0,  1000, 0,0);
    SET_NODE(2,  GPU_NODE_SHOWDOWN,-1, 0, 0,0,0,0,   1000, 0,0);
    SET_NODE(3,  GPU_NODE_DECISION, 0, 4, 4,5,6,9,   1750, 0,750);
    SET_NODE(4,  GPU_NODE_FOLD,     1, 0, 0,0,0,0,   1750, 0,750);
    SET_NODE(5,  GPU_NODE_SHOWDOWN,-1, 0, 0,0,0,0,   2500, 750,750);
    SET_NODE(6,  GPU_NODE_DECISION, 1, 2, 7,8,0,0,   4375, 2625,750);
    SET_NODE(7,  GPU_NODE_FOLD,     0, 0, 0,0,0,0,   4375, 2625,750);
    SET_NODE(8,  GPU_NODE_SHOWDOWN,-1, 0, 0,0,0,0,   6250, 2625,2625);
    SET_NODE(9,  GPU_NODE_DECISION, 1, 2, 10,11,0,0, 10000, 8250,750);
    SET_NODE(10, GPU_NODE_FOLD,     0, 0, 0,0,0,0,   10000, 8250,750);
    SET_NODE(11, GPU_NODE_SHOWDOWN,-1, 0, 0,0,0,0,   17500, 8250,8250);
    SET_NODE(12, GPU_NODE_DECISION, 0, 2, 13,14,0,0, 10000, 0,9000);
    SET_NODE(13, GPU_NODE_FOLD,     1, 0, 0,0,0,0,   10000, 0,9000);
    SET_NODE(14, GPU_NODE_SHOWDOWN,-1, 0, 0,0,0,0,   19000, 9000,9000);
    SET_NODE(15, GPU_NODE_DECISION, 1, 4, 16,17,18,21, 1750, 750,0);
    SET_NODE(16, GPU_NODE_FOLD,     0, 0, 0,0,0,0,   1750, 750,0);
    SET_NODE(17, GPU_NODE_SHOWDOWN,-1, 0, 0,0,0,0,   2500, 750,750);
    SET_NODE(18, GPU_NODE_DECISION, 0, 2, 19,20,0,0, 4375, 750,2625);
    SET_NODE(19, GPU_NODE_FOLD,     1, 0, 0,0,0,0,   4375, 750,2625);
    SET_NODE(20, GPU_NODE_SHOWDOWN,-1, 0, 0,0,0,0,   6250, 2625,2625);
    SET_NODE(21, GPU_NODE_DECISION, 0, 2, 22,23,0,0, 10000, 750,8250);
    SET_NODE(22, GPU_NODE_FOLD,     1, 0, 0,0,0,0,   10000, 750,8250);
    SET_NODE(23, GPU_NODE_SHOWDOWN,-1, 0, 0,0,0,0,   17500, 8250,8250);
    SET_NODE(24, GPU_NODE_DECISION, 1, 2, 25,26,0,0, 10000, 9000,0);
    SET_NODE(25, GPU_NODE_FOLD,     0, 0, 0,0,0,0,   10000, 9000,0);
    SET_NODE(26, GPU_NODE_SHOWDOWN,-1, 0, 0,0,0,0,   19000, 9000,9000);

    return 27;
}

int main(void) {
    printf("=== GPU vs CPU Sanity Test ===\n\n");

    GPUNode tree[GPU_MAX_NODES];
    int num_nodes = build_tree(tree);

    TextureData tex;
    memset(&tex, 0, sizeof(tex));
    tex.board[0] = parse_card("Qs"); tex.board[1] = parse_card("As");
    tex.board[2] = parse_card("2d"); tex.board[3] = parse_card("7h");
    tex.board[4] = parse_card("4c"); tex.num_board = 5;
    tex.starting_pot = 1000; tex.effective_stack = 9000;

    /* OOP: AhKh, QhQc, JhTh, 6h5h */
    tex.hands[0][0][0]=parse_card("Ah"); tex.hands[0][0][1]=parse_card("Kh");
    tex.hands[0][1][0]=parse_card("Qh"); tex.hands[0][1][1]=parse_card("Qc");
    tex.hands[0][2][0]=parse_card("Jh"); tex.hands[0][2][1]=parse_card("Th");
    tex.hands[0][3][0]=parse_card("6h"); tex.hands[0][3][1]=parse_card("5h");
    tex.num_hands[0] = 4;

    /* IP: AcKc, 3c3d, Tc9c, 8c8d */
    tex.hands[1][0][0]=parse_card("Ac"); tex.hands[1][0][1]=parse_card("Kc");
    tex.hands[1][1][0]=parse_card("3c"); tex.hands[1][1][1]=parse_card("3d");
    tex.hands[1][2][0]=parse_card("Tc"); tex.hands[1][2][1]=parse_card("9c");
    tex.hands[1][3][0]=parse_card("8c"); tex.hands[1][3][1]=parse_card("8d");
    tex.num_hands[1] = 4;

    for (int a = 0; a < 4; a++) tex.weights[0][a] = 1.0f;
    for (int a = 0; a < 4; a++) tex.weights[1][a] = 1.0f;

    /* Precompute strengths */
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

    printf("GPU solve (1000 iterations)...\n");
    int err = gpu_solve_batch(&tex, 1, tree, num_nodes, 1000, results);
    if (err) { printf("GPU solve failed: %d\n", err); return 1; }

    const char *oop_names[] = {"AhKh(TPTK)", "QhQc(trips)", "JhTh(JT)", "6h5h(air)"};
    const char *ip_names[] = {"AcKc(TPTK)", "3c3d(pair3)", "Tc9c(T9)", "8c8d(pair8)"};

    printf("\nGPU OOP strategies:\n");
    for (int h = 0; h < 4; h++) {
        printf("  %-14s: ", oop_names[h]);
        float total_bet = 0;
        for (int a = 0; a < 3; a++) {
            float s = results[0 * GPU_MAX_HANDS * GPU_MAX_ACTIONS +
                              h * GPU_MAX_ACTIONS + a];
            if (a == 0) printf("check=%.0f%% ", s*100);
            else { printf("bet%d=%.0f%% ", a, s*100); total_bet += s; }
        }
        printf("(total bet=%.0f%%)\n", total_bet*100);
    }

    printf("\nExpected (from CPU solver_v2, 1000 iter):\n");
    printf("  AhKh(TPTK):    check=23%% bet=77%%\n");
    printf("  QhQc(trips):   check=29%% bet=71%%\n");
    printf("  JhTh(JT):      check=77%% bet=23%%\n");
    printf("  6h5h(air):     check=100%% bet=0%%\n");

    free(results);
    printf("\n=== Done ===\n");
    return 0;
}
