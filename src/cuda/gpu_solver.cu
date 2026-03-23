/**
 * gpu_solver.cu — Level-batched CUDA DCFR solver
 *
 * Architecture (inspired by GPUGT paper, arXiv 2408.14778):
 *   - Game tree flattened by BFS level
 *   - Each DCFR iteration = two passes:
 *     1. Top-down: compute reach probabilities (strategy × parent reach)
 *     2. Bottom-up: compute CFV from terminals, propagate up, update regrets
 *   - Each pass = O(depth) sequential kernel launches
 *   - Within each kernel: parallelize across textures × hands
 *
 * Memory layout:
 *   All per-node data is stored in flat arrays indexed by [texture][node][hand]
 *   Nodes are ordered by BFS level for sequential processing.
 *
 * For batch precompute: solve many textures simultaneously on one GPU.
 * RTX 3060 (12GB): ~2000 textures per batch at 80 hands each.
 */

#include "gpu_solver.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        return -1; \
    } \
} while(0)

/* ── GPU hand evaluation ──────────────────────────────────────────────── */

__device__ uint32_t d_eval5(int c0, int c1, int c2, int c3, int c4) {
    int r[5] = {c0>>2, c1>>2, c2>>2, c3>>2, c4>>2};
    int s[5] = {c0&3, c1&3, c2&3, c3&3, c4&3};
    /* Sort descending */
    for (int i=1;i<5;i++) {
        int kr=r[i]; int j=i-1;
        while(j>=0 && r[j]<kr) { r[j+1]=r[j]; j--; }
        r[j+1]=kr;
    }
    int flush = (s[0]==s[1]&&s[1]==s[2]&&s[2]==s[3]&&s[3]==s[4]);
    int straight=0, shi=r[0];
    if (r[0]-r[4]==4 && r[0]!=r[1] && r[1]!=r[2] && r[2]!=r[3] && r[3]!=r[4]) straight=1;
    if (r[0]==12&&r[1]==3&&r[2]==2&&r[3]==1&&r[4]==0) { straight=1; shi=3; }
    if (straight&&flush) return (9u<<20)|(shi<<16);
    if (flush) return (6u<<20)|(r[0]<<16)|(r[1]<<12)|(r[2]<<8)|(r[3]<<4)|r[4];
    if (straight) return (5u<<20)|(shi<<16);
    int cnt[13]={0}; for(int i=0;i<5;i++) cnt[r[i]]++;
    int q=-1,t=-1,p1=-1,p2=-1;
    for(int i=12;i>=0;i--) {
        if(cnt[i]==4) q=i; else if(cnt[i]==3) t=i;
        else if(cnt[i]==2) { if(p1<0)p1=i; else p2=i; }
    }
    if(q>=0){int k=-1;for(int i=12;i>=0;i--)if(cnt[i]>0&&i!=q){k=i;break;}return(8u<<20)|(q<<16)|(k<<12);}
    if(t>=0&&p1>=0)return(7u<<20)|(t<<16)|(p1<<12);
    if(t>=0){int k0=-1,k1=-1;for(int i=12;i>=0;i--)if(cnt[i]>0&&i!=t){if(k0<0)k0=i;else k1=i;}return(4u<<20)|(t<<16)|(k0<<12)|(k1<<8);}
    if(p1>=0&&p2>=0){int k=-1;for(int i=12;i>=0;i--)if(cnt[i]>0&&i!=p1&&i!=p2){k=i;break;}return(3u<<20)|(p1<<16)|(p2<<12)|(k<<8);}
    if(p1>=0){int k[3],ki=0;for(int i=12;i>=0&&ki<3;i--)if(cnt[i]>0&&i!=p1)k[ki++]=i;return(2u<<20)|(p1<<16)|(k[0]<<12)|(k[1]<<8)|(k[2]<<4);}
    return(1u<<20)|(r[0]<<16)|(r[1]<<12)|(r[2]<<8)|(r[3]<<4)|r[4];
}

__device__ uint32_t d_eval7(const int cards[7]) {
    const int c[21][5] = {
        {0,1,2,3,4},{0,1,2,3,5},{0,1,2,3,6},{0,1,2,4,5},{0,1,2,4,6},{0,1,2,5,6},
        {0,1,3,4,5},{0,1,3,4,6},{0,1,3,5,6},{0,1,4,5,6},{0,2,3,4,5},{0,2,3,4,6},
        {0,2,3,5,6},{0,2,4,5,6},{0,3,4,5,6},{1,2,3,4,5},{1,2,3,4,6},{1,2,3,5,6},
        {1,2,4,5,6},{1,3,4,5,6},{2,3,4,5,6}
    };
    uint32_t best=0;
    for(int i=0;i<21;i++){
        uint32_t v=d_eval5(cards[c[i][0]],cards[c[i][1]],cards[c[i][2]],cards[c[i][3]],cards[c[i][4]]);
        if(v>best)best=v;
    }
    return best;
}

/* ── Kernel: precompute hand strengths for a river board ───────────────── */

__global__ void precompute_strengths_kernel(
    const TextureData *textures,
    int num_textures
) {
    int tex = blockIdx.x;
    int hand = threadIdx.x;
    if (tex >= num_textures) return;

    const TextureData *t = &textures[tex];
    if (t->num_board != 5) return;

    for (int p = 0; p < 2; p++) {
        if (hand < t->num_hands[p]) {
            int cards[7];
            for (int i = 0; i < 5; i++) cards[i] = t->board[i];
            cards[5] = t->hands[p][hand][0];
            cards[6] = t->hands[p][hand][1];
            /* Write directly to texture data (needs to be in device memory) */
            /* This kernel is called on device-resident TextureData */
            ((TextureData*)&textures[tex])->strengths[p][hand] = d_eval7(cards);
        }
    }
}

/* ── Kernel: regret matching (compute current strategy from regrets) ──── */

__global__ void regret_match_kernel(
    float *regrets,          /* [num_textures * num_nodes * max_actions * max_hands] */
    float *current_strategy, /* same layout */
    const int *node_players, /* [num_nodes] which player acts */
    const int *node_num_actions, /* [num_nodes] */
    const int *hands_per_player, /* [num_textures * 2] */
    int num_textures,
    int num_nodes,
    int max_actions,
    int max_hands,
    int target_node          /* which node to process */
) {
    int tex = blockIdx.x;
    int hand = threadIdx.x;
    if (tex >= num_textures) return;

    int player = node_players[target_node];
    int na = node_num_actions[target_node];
    int nh = hands_per_player[tex * 2 + player];
    if (hand >= nh) return;

    int base = ((tex * num_nodes + target_node) * max_actions) * max_hands;

    float sum = 0;
    for (int a = 0; a < na; a++) {
        float r = regrets[base + a * max_hands + hand];
        sum += (r > 0) ? r : 0;
    }

    if (sum > 0) {
        float inv = 1.0f / sum;
        for (int a = 0; a < na; a++) {
            float r = regrets[base + a * max_hands + hand];
            current_strategy[base + a * max_hands + hand] = ((r > 0) ? r : 0) * inv;
        }
    } else {
        float u = 1.0f / na;
        for (int a = 0; a < na; a++)
            current_strategy[base + a * max_hands + hand] = u;
    }
}

/* ── Kernel: compute terminal values (fold/showdown) ───────────────────── */

__global__ void terminal_value_kernel(
    const TextureData *textures,
    const GPUNode *tree,
    float *cfv,              /* [num_textures * num_nodes * max_hands] output */
    const float *reach_opp,  /* [num_textures * max_hands] opponent reach */
    int num_textures,
    int num_nodes,
    int max_hands,
    int target_node,
    int traverser
) {
    int tex = blockIdx.x;
    int hand = threadIdx.x;
    if (tex >= num_textures) return;

    const TextureData *t = &textures[tex];
    int nh = t->num_hands[traverser];
    if (hand >= nh) return;

    const GPUNode *node = &tree[target_node];
    int opp = 1 - traverser;
    int n_opp = t->num_hands[opp];
    int reach_base = tex * max_hands;
    int cfv_idx = (tex * num_nodes + target_node) * max_hands + hand;

    int hc0 = t->hands[traverser][hand][0];
    int hc1 = t->hands[traverser][hand][1];

    if (node->type == GPU_NODE_FOLD) {
        int winner = node->player;
        float payoff = (traverser == winner)
            ? (float)node->bets[1 - winner]
            : -(float)node->bets[traverser];
        float val = 0;
        for (int o = 0; o < n_opp; o++) {
            int oc0 = t->hands[opp][o][0], oc1 = t->hands[opp][o][1];
            if (hc0==oc0||hc0==oc1||hc1==oc0||hc1==oc1) continue;
            val += reach_opp[reach_base + o] * payoff;
        }
        cfv[cfv_idx] = val;
    }
    else if (node->type == GPU_NODE_SHOWDOWN) {
        uint32_t hs = t->strengths[traverser][hand];
        float win_pay = (float)node->bets[opp];
        float lose_pay = -(float)node->bets[traverser];
        float val = 0;
        for (int o = 0; o < n_opp; o++) {
            int oc0 = t->hands[opp][o][0], oc1 = t->hands[opp][o][1];
            if (hc0==oc0||hc0==oc1||hc1==oc0||hc1==oc1) continue;
            uint32_t os = t->strengths[opp][o];
            float w = reach_opp[reach_base + o];
            if (hs > os) val += w * win_pay;
            else if (hs < os) val += w * lose_pay;
        }
        cfv[cfv_idx] = val;
    }
    else if (node->type == GPU_NODE_LEAF) {
        cfv[cfv_idx] = 0; /* Leaf continuation value */
    }
}

/* ── Kernel: propagate CFV from children to parent decision node ────────── */

__global__ void propagate_cfv_kernel(
    const GPUNode *tree,
    float *cfv,               /* [num_textures * num_nodes * max_hands] */
    const float *cur_strategy,/* [num_textures * num_nodes * max_actions * max_hands] */
    const int *hands_per_player,
    int num_textures,
    int num_nodes,
    int max_actions,
    int max_hands,
    int target_node,
    int traverser
) {
    int tex = blockIdx.x;
    int hand = threadIdx.x;
    if (tex >= num_textures) return;

    const GPUNode *node = &tree[target_node];
    if (node->type != GPU_NODE_DECISION) return;

    int acting = node->player;
    int nh_trav = hands_per_player[tex * 2 + traverser];
    if (hand >= nh_trav) return;

    int na = node->num_actions;
    int cfv_parent = (tex * num_nodes + target_node) * max_hands + hand;

    if (acting == traverser) {
        /* Traverser's node: CFV = sum(strategy[a] * child_cfv[a]) */
        float val = 0;
        int strat_base = ((tex * num_nodes + target_node) * max_actions) * max_hands;
        int nh_acting = hands_per_player[tex * 2 + acting];
        for (int a = 0; a < na; a++) {
            int child = node->children[a];
            float child_val = cfv[(tex * num_nodes + child) * max_hands + hand];
            float s = cur_strategy[strat_base + a * max_hands + hand];
            val += s * child_val;
        }
        cfv[cfv_parent] = val;
    } else {
        /* Opponent's node: CFV = sum over all actions of child_cfv
         * (opponent's reach already encoded in the child terminal values) */
        float val = 0;
        for (int a = 0; a < na; a++) {
            int child = node->children[a];
            val += cfv[(tex * num_nodes + child) * max_hands + hand];
        }
        cfv[cfv_parent] = val;
    }
}

/* ── Kernel: update regrets ────────────────────────────────────────────── */

__global__ void update_regrets_kernel(
    const GPUNode *tree,
    float *regrets,
    float *cfv,
    const float *cur_strategy,
    const int *hands_per_player,
    int num_textures,
    int num_nodes,
    int max_actions,
    int max_hands,
    int target_node,
    int traverser,
    int iteration
) {
    int tex = blockIdx.x;
    int hand = threadIdx.x;
    if (tex >= num_textures) return;

    const GPUNode *node = &tree[target_node];
    if (node->type != GPU_NODE_DECISION || node->player != traverser) return;

    int nh = hands_per_player[tex * 2 + traverser];
    if (hand >= nh) return;

    int na = node->num_actions;
    int parent_cfv_idx = (tex * num_nodes + target_node) * max_hands + hand;
    float node_val = cfv[parent_cfv_idx];
    int reg_base = ((tex * num_nodes + target_node) * max_actions) * max_hands;

    for (int a = 0; a < na; a++) {
        int child = node->children[a];
        float child_val = cfv[(tex * num_nodes + child) * max_hands + hand];
        float regret = child_val - node_val;
        int reg_idx = reg_base + a * max_hands + hand;
        regrets[reg_idx] += regret;
    }

    /* Linear CFR discount */
    float d = (float)iteration / ((float)iteration + 1.0f);
    for (int a = 0; a < na; a++) {
        int idx = reg_base + a * max_hands + hand;
        regrets[idx] *= d;
    }
}

/* ── Host: batch solve ─────────────────────────────────────────────────── */

extern "C" int gpu_solve_batch(
    const TextureData *h_textures,
    int num_textures,
    const GPUNode *h_tree,
    int num_tree_nodes,
    int max_iterations,
    float *results_out
) {
    printf("[GPU] Batch solve: %d textures, %d tree nodes, %d iterations\n",
           num_textures, num_tree_nodes, max_iterations);

    /* Device memory */
    TextureData *d_textures;
    GPUNode *d_tree;
    float *d_regrets, *d_strategy, *d_cfv, *d_reach;
    int *d_node_players, *d_node_num_actions, *d_hands_per_player;

    size_t tex_size = num_textures * sizeof(TextureData);
    size_t tree_size = num_tree_nodes * sizeof(GPUNode);
    int state_stride = num_tree_nodes * GPU_MAX_ACTIONS * GPU_MAX_HANDS;
    size_t state_size = (size_t)num_textures * state_stride * sizeof(float);
    size_t cfv_stride = num_tree_nodes * GPU_MAX_HANDS;
    size_t cfv_size = (size_t)num_textures * cfv_stride * sizeof(float);
    size_t reach_size = (size_t)num_textures * GPU_MAX_HANDS * sizeof(float);

    printf("[GPU] Memory: state=%.1f MB, cfv=%.1f MB, reach=%.1f MB\n",
           state_size/1e6, cfv_size/1e6, reach_size/1e6);

    CUDA_CHECK(cudaMalloc(&d_textures, tex_size));
    CUDA_CHECK(cudaMalloc(&d_tree, tree_size));
    CUDA_CHECK(cudaMalloc(&d_regrets, state_size));
    CUDA_CHECK(cudaMalloc(&d_strategy, state_size));
    CUDA_CHECK(cudaMalloc(&d_cfv, cfv_size));
    CUDA_CHECK(cudaMalloc(&d_reach, reach_size));

    /* Node metadata arrays */
    int *h_node_players = (int*)malloc(num_tree_nodes * sizeof(int));
    int *h_node_num_actions = (int*)malloc(num_tree_nodes * sizeof(int));
    int *h_hands_pp = (int*)malloc(num_textures * 2 * sizeof(int));

    for (int n = 0; n < num_tree_nodes; n++) {
        h_node_players[n] = h_tree[n].player;
        h_node_num_actions[n] = h_tree[n].num_actions;
    }
    for (int t = 0; t < num_textures; t++) {
        h_hands_pp[t*2] = h_textures[t].num_hands[0];
        h_hands_pp[t*2+1] = h_textures[t].num_hands[1];
    }

    CUDA_CHECK(cudaMalloc(&d_node_players, num_tree_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_node_num_actions, num_tree_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_hands_per_player, num_textures * 2 * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_textures, h_textures, tex_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tree, h_tree, tree_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_node_players, h_node_players, num_tree_nodes * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_node_num_actions, h_node_num_actions, num_tree_nodes * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_hands_per_player, h_hands_pp, num_textures * 2 * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_regrets, 0, state_size));
    CUDA_CHECK(cudaMemset(d_strategy, 0, state_size));

    /* Precompute hand strengths for river boards */
    precompute_strengths_kernel<<<num_textures, GPU_MAX_HANDS>>>(d_textures, num_textures);
    CUDA_CHECK(cudaDeviceSynchronize());

    /* BFS level order of nodes */
    int *level_order = (int*)malloc(num_tree_nodes * sizeof(int));
    int *node_depth = (int*)calloc(num_tree_nodes, sizeof(int));
    int max_depth = 0;
    {
        /* BFS to determine depths */
        int *queue = (int*)malloc(num_tree_nodes * sizeof(int));
        int qhead = 0, qtail = 0;
        queue[qtail++] = 0;
        node_depth[0] = 0;
        int lo_idx = 0;
        while (qhead < qtail) {
            int n = queue[qhead++];
            level_order[lo_idx++] = n;
            if (node_depth[n] > max_depth) max_depth = node_depth[n];
            for (int a = 0; a < h_tree[n].num_actions; a++) {
                int child = h_tree[n].children[a];
                node_depth[child] = node_depth[n] + 1;
                queue[qtail++] = child;
            }
        }
        free(queue);
    }

    dim3 grid(num_textures);
    dim3 block(GPU_MAX_HANDS);

    /* ── Main DCFR loop ─────────────────────────────────────── */
    for (int iter = 1; iter <= max_iterations; iter++) {
        for (int traverser = 0; traverser < 2; traverser++) {
            /* Initialize reach with weights */
            /* For simplicity, copy weights to reach (done per iteration) */
            {
                float *h_reach = (float*)malloc(reach_size);
                for (int t = 0; t < num_textures; t++) {
                    for (int h = 0; h < GPU_MAX_HANDS; h++) {
                        h_reach[t * GPU_MAX_HANDS + h] =
                            (h < h_textures[t].num_hands[1-traverser])
                            ? h_textures[t].weights[1-traverser][h] : 0;
                    }
                }
                CUDA_CHECK(cudaMemcpy(d_reach, h_reach, reach_size, cudaMemcpyHostToDevice));
                free(h_reach);
            }

            /* Top-down: compute strategy at each decision node */
            for (int lo = 0; lo < num_tree_nodes; lo++) {
                int n = level_order[lo];
                if (h_tree[n].type == GPU_NODE_DECISION) {
                    regret_match_kernel<<<grid, block>>>(
                        d_regrets, d_strategy, d_node_players, d_node_num_actions,
                        d_hands_per_player, num_textures, num_tree_nodes,
                        GPU_MAX_ACTIONS, GPU_MAX_HANDS, n);
                }
            }
            CUDA_CHECK(cudaDeviceSynchronize());

            /* Bottom-up: compute CFV from terminals */
            CUDA_CHECK(cudaMemset(d_cfv, 0, cfv_size));

            /* First: terminal nodes */
            for (int lo = num_tree_nodes - 1; lo >= 0; lo--) {
                int n = level_order[lo];
                if (h_tree[n].type == GPU_NODE_FOLD ||
                    h_tree[n].type == GPU_NODE_SHOWDOWN ||
                    h_tree[n].type == GPU_NODE_LEAF) {
                    terminal_value_kernel<<<grid, block>>>(
                        d_textures, d_tree, d_cfv, d_reach,
                        num_textures, num_tree_nodes, GPU_MAX_HANDS,
                        n, traverser);
                }
            }
            CUDA_CHECK(cudaDeviceSynchronize());

            /* Then: propagate up through decision nodes (bottom-up) */
            for (int lo = num_tree_nodes - 1; lo >= 0; lo--) {
                int n = level_order[lo];
                if (h_tree[n].type == GPU_NODE_DECISION) {
                    propagate_cfv_kernel<<<grid, block>>>(
                        d_tree, d_cfv, d_strategy, d_hands_per_player,
                        num_textures, num_tree_nodes,
                        GPU_MAX_ACTIONS, GPU_MAX_HANDS, n, traverser);
                    CUDA_CHECK(cudaDeviceSynchronize());

                    /* Update regrets at traverser's nodes */
                    if (h_tree[n].player == traverser) {
                        update_regrets_kernel<<<grid, block>>>(
                            d_tree, d_regrets, d_cfv, d_strategy,
                            d_hands_per_player, num_textures, num_tree_nodes,
                            GPU_MAX_ACTIONS, GPU_MAX_HANDS, n, traverser, iter);
                    }
                }
            }
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        if (iter % 50 == 0 || iter == max_iterations) {
            printf("[GPU] Iteration %d/%d\n", iter, max_iterations);
        }
    }

    /* Extract final strategies (current_strategy at root node) */
    float *h_strategy = (float*)malloc(state_size);
    CUDA_CHECK(cudaMemcpy(h_strategy, d_strategy, state_size, cudaMemcpyDeviceToHost));

    int out_stride = 2 * GPU_MAX_HANDS * GPU_MAX_ACTIONS;
    for (int t = 0; t < num_textures; t++) {
        for (int p = 0; p < 2; p++) {
            int nh = h_textures[t].num_hands[p];
            /* Root node = index 0 */
            int base = (t * num_tree_nodes + 0) * GPU_MAX_ACTIONS * GPU_MAX_HANDS;
            for (int h = 0; h < nh; h++) {
                for (int a = 0; a < GPU_MAX_ACTIONS; a++) {
                    results_out[t * out_stride + p * GPU_MAX_HANDS * GPU_MAX_ACTIONS +
                                h * GPU_MAX_ACTIONS + a] =
                        h_strategy[base + a * GPU_MAX_HANDS + h];
                }
            }
        }
    }

    free(h_strategy);
    free(h_node_players);
    free(h_node_num_actions);
    free(h_hands_pp);
    free(level_order);
    free(node_depth);
    cudaFree(d_textures);
    cudaFree(d_tree);
    cudaFree(d_regrets);
    cudaFree(d_strategy);
    cudaFree(d_cfv);
    cudaFree(d_reach);
    cudaFree(d_node_players);
    cudaFree(d_node_num_actions);
    cudaFree(d_hands_per_player);

    return 0;
}

extern "C" int gpu_get_info(int *cuda_cores, size_t *free_mem, size_t *total_mem) {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    *cuda_cores = prop.multiProcessorCount * 128;
    cudaMemGetInfo(free_mem, total_mem);
    printf("[GPU] %s: %d SMs, %d cores, %.1f GB free / %.1f GB total\n",
           prop.name, prop.multiProcessorCount, *cuda_cores,
           *free_mem / 1e9, *total_mem / 1e9);
    return 0;
}
