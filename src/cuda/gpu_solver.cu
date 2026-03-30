/**
 * gpu_solver.cu — Level-batched CUDA solver with Linear CFR
 *
 * Aligned with Pluribus:
 *   - Linear CFR: regrets discounted by t/(t+1) each iteration
 *   - Strategy sum weighted by iteration for average strategy extraction
 *   - Supports single-street and multi-street trees
 *   - Multi-level strategy extraction (flop root + turn roots)
 *
 * Architecture (inspired by GPUGT paper, arXiv 2408.14778):
 *   - Game tree flattened by BFS level
 *   - Each iteration = two passes:
 *     1. Top-down: propagate reach probabilities
 *     2. Bottom-up: compute CFV from terminals, propagate up, update regrets
 *   - Each pass = O(depth) sequential kernel launches
 *   - Within each kernel: parallelize across textures × hands
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

/* ── Kernel: precompute hand strengths ─────────────────────────────────── */

__global__ void precompute_strengths_kernel(
    const TextureData *textures, int num_textures
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
            ((TextureData*)&textures[tex])->strengths[p][hand] = d_eval7(cards);
        }
    }
}

/* ── Kernel: regret matching ──────────────────────────────────────────── */

__global__ void regret_match_kernel(
    float *regrets, float *current_strategy,
    const int *node_players, const int *node_num_actions,
    const int *hands_per_player,
    int num_textures, int num_nodes, int max_actions, int max_hands,
    int target_node
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

/* ── Kernel: terminal values ──────────────────────────────────────────── */

__global__ void terminal_value_kernel(
    const TextureData *textures, const GPUNode *tree,
    float *cfv, const float *node_reach,
    int num_textures, int num_nodes, int max_hands,
    int target_node, int traverser
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
    int reach_base = (tex * num_nodes + target_node) * max_hands;
    int cfv_idx = (tex * num_nodes + target_node) * max_hands + hand;
    int hc0 = t->hands[traverser][hand][0];
    int hc1 = t->hands[traverser][hand][1];

    if (node->type == GPU_NODE_FOLD) {
        int winner = node->player;
        int loser = 1 - winner;
        float half_start = t->starting_pot * 0.5f;
        float payoff = (traverser == winner)
            ? (half_start + (float)node->bets[loser])
            : -(half_start + (float)node->bets[traverser]);
        float val = 0;
        for (int o = 0; o < n_opp; o++) {
            int oc0 = t->hands[opp][o][0], oc1 = t->hands[opp][o][1];
            if (hc0==oc0||hc0==oc1||hc1==oc0||hc1==oc1) continue;
            val += node_reach[reach_base + o] * payoff;
        }
        cfv[cfv_idx] = val;
    } else if (node->type == GPU_NODE_SHOWDOWN) {
        uint32_t hs = t->strengths[traverser][hand];
        float half_pot = node->pot * 0.5f;
        float val = 0;
        for (int o = 0; o < n_opp; o++) {
            int oc0 = t->hands[opp][o][0], oc1 = t->hands[opp][o][1];
            if (hc0==oc0||hc0==oc1||hc1==oc0||hc1==oc1) continue;
            uint32_t os = t->strengths[opp][o];
            float w = node_reach[reach_base + o];
            if (hs > os) val += w * half_pot;
            else if (hs < os) val -= w * half_pot;
        }
        cfv[cfv_idx] = val;
    } else if (node->type == GPU_NODE_LEAF) {
        cfv[cfv_idx] = 0;
    }
}

/* ── Kernel: reach propagation ────────────────────────────────────────── */

__global__ void propagate_reach_kernel(
    const GPUNode *tree, float *node_reach, const float *cur_strategy,
    const int *hands_per_player,
    int num_textures, int num_nodes, int max_actions, int max_hands,
    int parent_node, int acting_player
) {
    int tex = blockIdx.x;
    int hand = threadIdx.x;
    if (tex >= num_textures) return;
    const GPUNode *parent = &tree[parent_node];
    if (parent->type != GPU_NODE_DECISION) return;
    int nh_acting = hands_per_player[tex * 2 + parent->player];
    int nh_opp = hands_per_player[tex * 2 + acting_player];

    if (parent->player == acting_player && hand < nh_acting) {
        int pr_idx = (tex * num_nodes + parent_node) * max_hands + hand;
        float pr = node_reach[pr_idx];
        int sb = ((tex * num_nodes + parent_node) * max_actions) * max_hands;
        for (int a = 0; a < parent->num_actions; a++) {
            int child = parent->children[a];
            float s = cur_strategy[sb + a * max_hands + hand];
            node_reach[(tex * num_nodes + child) * max_hands + hand] = pr * s;
        }
    } else if (parent->player != acting_player && hand < nh_opp) {
        int pr_idx = (tex * num_nodes + parent_node) * max_hands + hand;
        float pr = node_reach[pr_idx];
        for (int a = 0; a < parent->num_actions; a++) {
            int child = parent->children[a];
            node_reach[(tex * num_nodes + child) * max_hands + hand] = pr;
        }
    }
}

/* ── Kernel: CFV propagation + regret update ──────────────────────────── */

__global__ void propagate_cfv_kernel(
    const GPUNode *tree, float *cfv, const float *cur_strategy,
    const int *hands_per_player,
    int num_textures, int num_nodes, int max_actions, int max_hands,
    int target_node, int traverser
) {
    int tex = blockIdx.x;
    int hand = threadIdx.x;
    if (tex >= num_textures) return;
    const GPUNode *node = &tree[target_node];
    if (node->type != GPU_NODE_DECISION) return;
    int nh_trav = hands_per_player[tex * 2 + traverser];
    if (hand >= nh_trav) return;
    int na = node->num_actions;
    int cfv_parent = (tex * num_nodes + target_node) * max_hands + hand;

    if (node->player == traverser) {
        float val = 0;
        int sb = ((tex * num_nodes + target_node) * max_actions) * max_hands;
        for (int a = 0; a < na; a++) {
            int child = node->children[a];
            float cv = cfv[(tex * num_nodes + child) * max_hands + hand];
            val += cur_strategy[sb + a * max_hands + hand] * cv;
        }
        cfv[cfv_parent] = val;
    } else {
        float val = 0;
        for (int a = 0; a < na; a++) {
            int child = node->children[a];
            val += cfv[(tex * num_nodes + child) * max_hands + hand];
        }
        cfv[cfv_parent] = val;
    }
}

__global__ void update_regrets_kernel(
    const GPUNode *tree, float *regrets, float *strategy_sum,
    float *cfv, const float *cur_strategy,
    const int *hands_per_player,
    int num_textures, int num_nodes, int max_actions, int max_hands,
    int target_node, int traverser, int iteration
) {
    int tex = blockIdx.x;
    int hand = threadIdx.x;
    if (tex >= num_textures) return;
    const GPUNode *node = &tree[target_node];
    if (node->type != GPU_NODE_DECISION || node->player != traverser) return;
    int nh = hands_per_player[tex * 2 + traverser];
    if (hand >= nh) return;
    int na = node->num_actions;
    int pcfv = (tex * num_nodes + target_node) * max_hands + hand;
    float nv = cfv[pcfv];
    int rb = ((tex * num_nodes + target_node) * max_actions) * max_hands;

    for (int a = 0; a < na; a++) {
        int child = node->children[a];
        float cv = cfv[(tex * num_nodes + child) * max_hands + hand];
        regrets[rb + a * max_hands + hand] += cv - nv;
    }
    /* Linear CFR discount */
    float d = (float)iteration / ((float)iteration + 1.0f);
    for (int a = 0; a < na; a++)
        regrets[rb + a * max_hands + hand] *= d;

    /* Accumulate iteration-weighted strategy sum */
    for (int a = 0; a < na; a++)
        strategy_sum[rb + a * max_hands + hand] +=
            (float)iteration * cur_strategy[rb + a * max_hands + hand];
}

/* ── Host: batch solve ─────────────────────────────────────────────────── */

extern "C" int gpu_solve_batch(
    const TextureData *h_textures, int num_textures,
    const GPUNode *h_tree, int num_tree_nodes,
    int max_iterations, float *results_out
) {
    printf("[GPU] Linear CFR batch: %d textures, %d nodes, %d iters\n",
           num_textures, num_tree_nodes, max_iterations);

    TextureData *d_textures; GPUNode *d_tree;
    float *d_regrets, *d_strategy, *d_strategy_sum, *d_cfv, *d_node_reach;
    int *d_node_players, *d_node_num_actions, *d_hands_per_player;

    size_t tex_sz = num_textures * sizeof(TextureData);
    size_t tree_sz = num_tree_nodes * sizeof(GPUNode);
    size_t state_sz = (size_t)num_textures * num_tree_nodes * GPU_MAX_ACTIONS * GPU_MAX_HANDS * sizeof(float);
    size_t cfv_sz = (size_t)num_textures * num_tree_nodes * GPU_MAX_HANDS * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_textures, tex_sz));
    CUDA_CHECK(cudaMalloc(&d_tree, tree_sz));
    CUDA_CHECK(cudaMalloc(&d_regrets, state_sz));
    CUDA_CHECK(cudaMalloc(&d_strategy, state_sz));
    CUDA_CHECK(cudaMalloc(&d_strategy_sum, state_sz));
    CUDA_CHECK(cudaMalloc(&d_cfv, cfv_sz));
    CUDA_CHECK(cudaMalloc(&d_node_reach, cfv_sz));

    int *h_np = (int*)malloc(num_tree_nodes * sizeof(int));
    int *h_na = (int*)malloc(num_tree_nodes * sizeof(int));
    int *h_hp = (int*)malloc(num_textures * 2 * sizeof(int));
    for (int n = 0; n < num_tree_nodes; n++) {
        h_np[n] = h_tree[n].player; h_na[n] = h_tree[n].num_actions;
    }
    for (int t = 0; t < num_textures; t++) {
        h_hp[t*2] = h_textures[t].num_hands[0];
        h_hp[t*2+1] = h_textures[t].num_hands[1];
    }

    CUDA_CHECK(cudaMalloc(&d_node_players, num_tree_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_node_num_actions, num_tree_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_hands_per_player, num_textures * 2 * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_textures, h_textures, tex_sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tree, h_tree, tree_sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_node_players, h_np, num_tree_nodes * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_node_num_actions, h_na, num_tree_nodes * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_hands_per_player, h_hp, num_textures * 2 * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_regrets, 0, state_sz));
    CUDA_CHECK(cudaMemset(d_strategy, 0, state_sz));
    CUDA_CHECK(cudaMemset(d_strategy_sum, 0, state_sz));

    precompute_strengths_kernel<<<num_textures, GPU_MAX_HANDS>>>(d_textures, num_textures);
    CUDA_CHECK(cudaDeviceSynchronize());

    /* BFS level order */
    int *lo = (int*)malloc(num_tree_nodes * sizeof(int));
    { int *q = (int*)malloc(num_tree_nodes * sizeof(int));
      int qh=0,qt=0,li=0; q[qt++]=0;
      while(qh<qt) { int n=q[qh++]; lo[li++]=n;
        for(int a=0;a<h_tree[n].num_actions;a++) q[qt++]=h_tree[n].children[a]; }
      free(q); }

    dim3 grid(num_textures); dim3 block(GPU_MAX_HANDS);

    for (int iter = 1; iter <= max_iterations; iter++) {
        for (int trav = 0; trav < 2; trav++) {
            int opp = 1-trav;
            for (int i=0;i<num_tree_nodes;i++) { int n=lo[i];
                if(h_tree[n].type==GPU_NODE_DECISION)
                    regret_match_kernel<<<grid,block>>>(d_regrets,d_strategy,d_node_players,
                        d_node_num_actions,d_hands_per_player,num_textures,num_tree_nodes,
                        GPU_MAX_ACTIONS,GPU_MAX_HANDS,n); }
            CUDA_CHECK(cudaDeviceSynchronize());

            { float *hr=(float*)calloc((size_t)num_textures*num_tree_nodes*GPU_MAX_HANDS,sizeof(float));
              for(int t=0;t<num_textures;t++) for(int h=0;h<h_textures[t].num_hands[opp];h++)
                hr[(t*num_tree_nodes+0)*GPU_MAX_HANDS+h]=h_textures[t].weights[opp][h];
              CUDA_CHECK(cudaMemcpy(d_node_reach,hr,cfv_sz,cudaMemcpyHostToDevice)); free(hr); }

            for(int i=0;i<num_tree_nodes;i++) { int n=lo[i];
                if(h_tree[n].type==GPU_NODE_DECISION) {
                    propagate_reach_kernel<<<grid,block>>>(d_tree,d_node_reach,d_strategy,
                        d_hands_per_player,num_textures,num_tree_nodes,GPU_MAX_ACTIONS,
                        GPU_MAX_HANDS,n,opp);
                    CUDA_CHECK(cudaDeviceSynchronize()); } }

            CUDA_CHECK(cudaMemset(d_cfv, 0, cfv_sz));
            for(int i=num_tree_nodes-1;i>=0;i--) { int n=lo[i];
                if(h_tree[n].type==GPU_NODE_FOLD||h_tree[n].type==GPU_NODE_SHOWDOWN||h_tree[n].type==GPU_NODE_LEAF)
                    terminal_value_kernel<<<grid,block>>>(d_textures,d_tree,d_cfv,d_node_reach,
                        num_textures,num_tree_nodes,GPU_MAX_HANDS,n,trav); }
            CUDA_CHECK(cudaDeviceSynchronize());

            for(int i=num_tree_nodes-1;i>=0;i--) { int n=lo[i];
                if(h_tree[n].type==GPU_NODE_DECISION) {
                    propagate_cfv_kernel<<<grid,block>>>(d_tree,d_cfv,d_strategy,d_hands_per_player,
                        num_textures,num_tree_nodes,GPU_MAX_ACTIONS,GPU_MAX_HANDS,n,trav);
                    CUDA_CHECK(cudaDeviceSynchronize());
                    if(h_tree[n].player==trav)
                        update_regrets_kernel<<<grid,block>>>(d_tree,d_regrets,d_strategy_sum,d_cfv,
                            d_strategy,d_hands_per_player,num_textures,num_tree_nodes,
                            GPU_MAX_ACTIONS,GPU_MAX_HANDS,n,trav,iter); } }
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        if(iter%50==0||iter==max_iterations)
            printf("[GPU] iter %d/%d\n",iter,max_iterations);
    }

    float *hs=(float*)malloc(state_sz);
    CUDA_CHECK(cudaMemcpy(hs,d_strategy,state_sz,cudaMemcpyDeviceToHost));
    int os=2*GPU_MAX_HANDS*GPU_MAX_ACTIONS;
    for(int t=0;t<num_textures;t++) for(int p=0;p<2;p++) {
        int nh=h_textures[t].num_hands[p];
        int b=(t*num_tree_nodes+0)*GPU_MAX_ACTIONS*GPU_MAX_HANDS;
        for(int h=0;h<nh;h++) for(int a=0;a<GPU_MAX_ACTIONS;a++)
            results_out[t*os+p*GPU_MAX_HANDS*GPU_MAX_ACTIONS+h*GPU_MAX_ACTIONS+a]=
                hs[b+a*GPU_MAX_HANDS+h]; }

    free(hs); free(h_np); free(h_na); free(h_hp); free(lo);
    cudaFree(d_textures); cudaFree(d_tree); cudaFree(d_regrets);
    cudaFree(d_strategy); cudaFree(d_strategy_sum); cudaFree(d_cfv);
    cudaFree(d_node_reach); cudaFree(d_node_players);
    cudaFree(d_node_num_actions); cudaFree(d_hands_per_player);
    return 0;
}

extern "C" int gpu_solve_batch_multilevel(
    const TextureData *textures, int num_textures,
    const GPUNode *tree, int num_tree_nodes,
    int max_iterations, ExtractionConfig *config, float *results_out
) {
    /* Run the standard batch solve first */
    int ret = gpu_solve_batch(textures, num_textures, tree, num_tree_nodes,
                              max_iterations, results_out);
    if (ret != 0) return ret;

    if (!config) return 0;

    /* Re-upload the final strategy_sum from GPU to extract at additional nodes.
     * The standard solve already downloaded root strategies into results_out.
     * For multilevel extraction, we need strategies at turn root nodes too.
     *
     * Since gpu_solve_batch already freed GPU state, we re-extract from the
     * strategy data that was computed. The most efficient approach is to
     * re-run a minimal solve (1 iteration) just to get the regret-matched
     * strategy at all nodes, then extract from the desired nodes.
     *
     * However, for correctness, we re-solve with the full iteration count
     * and extract during the final iteration. This is the simplest correct
     * approach -- a future optimization could cache GPU state across calls. */

    /* For now, extract turn root strategies from the results buffer.
     * The full strategy array is indexed by node, so we can extract
     * strategies at any node from the solve results. */

    /* Re-solve to get strategies at all nodes (not just root) */
    size_t state_sz = (size_t)num_textures * num_tree_nodes *
                      GPU_MAX_ACTIONS * GPU_MAX_HANDS * sizeof(float);
    size_t cfv_sz = (size_t)num_textures * num_tree_nodes *
                    GPU_MAX_HANDS * sizeof(float);
    size_t tex_sz = num_textures * sizeof(TextureData);
    size_t tree_sz = num_tree_nodes * sizeof(GPUNode);

    TextureData *d_textures; GPUNode *d_tree;
    float *d_regrets, *d_strategy, *d_strategy_sum, *d_cfv, *d_node_reach;
    int *d_node_players, *d_node_num_actions, *d_hands_per_player;

    CUDA_CHECK(cudaMalloc(&d_textures, tex_sz));
    CUDA_CHECK(cudaMalloc(&d_tree, tree_sz));
    CUDA_CHECK(cudaMalloc(&d_regrets, state_sz));
    CUDA_CHECK(cudaMalloc(&d_strategy, state_sz));
    CUDA_CHECK(cudaMalloc(&d_strategy_sum, state_sz));
    CUDA_CHECK(cudaMalloc(&d_cfv, cfv_sz));
    CUDA_CHECK(cudaMalloc(&d_node_reach, cfv_sz));

    int *h_np = (int*)malloc(num_tree_nodes * sizeof(int));
    int *h_na = (int*)malloc(num_tree_nodes * sizeof(int));
    int *h_hp = (int*)malloc(num_textures * 2 * sizeof(int));
    if (!h_np || !h_na || !h_hp) goto cleanup_ml;
    for (int n = 0; n < num_tree_nodes; n++) {
        h_np[n] = tree[n].player; h_na[n] = tree[n].num_actions;
    }
    for (int t = 0; t < num_textures; t++) {
        h_hp[t*2] = textures[t].num_hands[0];
        h_hp[t*2+1] = textures[t].num_hands[1];
    }

    CUDA_CHECK(cudaMalloc(&d_node_players, num_tree_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_node_num_actions, num_tree_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_hands_per_player, num_textures * 2 * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_textures, textures, tex_sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tree, tree, tree_sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_node_players, h_np, num_tree_nodes * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_node_num_actions, h_na, num_tree_nodes * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_hands_per_player, h_hp, num_textures * 2 * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_regrets, 0, state_sz));
    CUDA_CHECK(cudaMemset(d_strategy_sum, 0, state_sz));

    precompute_strengths_kernel<<<num_textures, GPU_MAX_HANDS>>>(d_textures, num_textures);
    CUDA_CHECK(cudaDeviceSynchronize());

    {   /* BFS level order */
        int *lo = (int*)malloc(num_tree_nodes * sizeof(int));
        int *q = (int*)malloc(num_tree_nodes * sizeof(int));
        if (!lo || !q) { free(lo); free(q); goto cleanup_ml; }
        int qh=0,qt=0,li=0; q[qt++]=0;
        while(qh<qt) { int n=q[qh++]; lo[li++]=n;
            for(int a=0;a<tree[n].num_actions;a++) q[qt++]=tree[n].children[a]; }
        free(q);

        dim3 grid(num_textures); dim3 block(GPU_MAX_HANDS);

        for (int iter = 1; iter <= max_iterations; iter++) {
            for (int trav = 0; trav < 2; trav++) {
                int opp = 1-trav;
                for (int i=0;i<num_tree_nodes;i++) { int n=lo[i];
                    if(tree[n].type==GPU_NODE_DECISION)
                        regret_match_kernel<<<grid,block>>>(d_regrets,d_strategy,d_node_players,
                            d_node_num_actions,d_hands_per_player,num_textures,num_tree_nodes,
                            GPU_MAX_ACTIONS,GPU_MAX_HANDS,n); }
                CUDA_CHECK(cudaDeviceSynchronize());

                { float *hr=(float*)calloc((size_t)num_textures*num_tree_nodes*GPU_MAX_HANDS,sizeof(float));
                  if(!hr) { free(lo); goto cleanup_ml; }
                  for(int t=0;t<num_textures;t++) for(int h=0;h<textures[t].num_hands[opp];h++)
                    hr[(t*num_tree_nodes+0)*GPU_MAX_HANDS+h]=textures[t].weights[opp][h];
                  CUDA_CHECK(cudaMemcpy(d_node_reach,hr,cfv_sz,cudaMemcpyHostToDevice)); free(hr); }

                for(int i=0;i<num_tree_nodes;i++) { int n=lo[i];
                    if(tree[n].type==GPU_NODE_DECISION) {
                        propagate_reach_kernel<<<grid,block>>>(d_tree,d_node_reach,d_strategy,
                            d_hands_per_player,num_textures,num_tree_nodes,GPU_MAX_ACTIONS,
                            GPU_MAX_HANDS,n,opp);
                        CUDA_CHECK(cudaDeviceSynchronize()); } }

                CUDA_CHECK(cudaMemset(d_cfv, 0, cfv_sz));
                for(int i=num_tree_nodes-1;i>=0;i--) { int n=lo[i];
                    if(tree[n].type==GPU_NODE_FOLD||tree[n].type==GPU_NODE_SHOWDOWN||tree[n].type==GPU_NODE_LEAF)
                        terminal_value_kernel<<<grid,block>>>(d_textures,d_tree,d_cfv,d_node_reach,
                            num_textures,num_tree_nodes,GPU_MAX_HANDS,n,trav); }
                CUDA_CHECK(cudaDeviceSynchronize());

                for(int i=num_tree_nodes-1;i>=0;i--) { int n=lo[i];
                    if(tree[n].type==GPU_NODE_DECISION) {
                        propagate_cfv_kernel<<<grid,block>>>(d_tree,d_cfv,d_strategy,d_hands_per_player,
                            num_textures,num_tree_nodes,GPU_MAX_ACTIONS,GPU_MAX_HANDS,n,trav);
                        CUDA_CHECK(cudaDeviceSynchronize());
                        if(tree[n].player==trav)
                            update_regrets_kernel<<<grid,block>>>(d_tree,d_regrets,d_strategy_sum,d_cfv,
                                d_strategy,d_hands_per_player,num_textures,num_tree_nodes,
                                GPU_MAX_ACTIONS,GPU_MAX_HANDS,n,trav,iter); } }
                CUDA_CHECK(cudaDeviceSynchronize());
            }
        }

        /* Extract strategies at all requested nodes */
        float *hs = (float*)malloc(state_sz);
        if (!hs) { free(lo); goto cleanup_ml; }
        CUDA_CHECK(cudaMemcpy(hs, d_strategy_sum, state_sz, cudaMemcpyDeviceToHost));

        /* Helper: extract weighted-average strategy at a given node */
        auto extract_node = [&](int node_idx, float *out, int out_stride) {
            for (int t = 0; t < num_textures; t++) {
                for (int p = 0; p < 2; p++) {
                    int nh = textures[t].num_hands[p];
                    int b = (t * num_tree_nodes + node_idx) * GPU_MAX_ACTIONS * GPU_MAX_HANDS;
                    for (int h = 0; h < nh; h++) {
                        float sum = 0;
                        for (int a = 0; a < GPU_MAX_ACTIONS; a++) {
                            float v = hs[b + a * GPU_MAX_HANDS + h];
                            sum += (v > 0) ? v : 0;
                        }
                        float inv = (sum > 0) ? 1.0f / sum : 1.0f / tree[node_idx].num_actions;
                        for (int a = 0; a < GPU_MAX_ACTIONS; a++) {
                            float v = hs[b + a * GPU_MAX_HANDS + h];
                            int idx = t * out_stride + p * GPU_MAX_HANDS * GPU_MAX_ACTIONS
                                      + h * GPU_MAX_ACTIONS + a;
                            out[idx] = (sum > 0) ? ((v > 0 ? v : 0) * inv) : inv;
                        }
                    }
                }
            }
        };

        /* Extract flop root strategies */
        if (config->extract_flop_root && config->flop_strategies) {
            int out_stride = 2 * GPU_MAX_HANDS * GPU_MAX_ACTIONS;
            extract_node(config->flop_root_node, config->flop_strategies, out_stride);
        }

        /* Extract turn root strategies */
        if (config->extract_turn_roots && config->turn_strategies && config->turn_root_nodes) {
            int out_stride = 2 * GPU_MAX_HANDS * GPU_MAX_ACTIONS;
            for (int ti = 0; ti < config->num_turn_roots; ti++) {
                int node_idx = config->turn_root_nodes[ti];
                float *dest = config->turn_strategies +
                              ti * num_textures * out_stride;
                extract_node(node_idx, dest, out_stride);
            }
        }

        free(hs);
        free(lo);
    }

cleanup_ml:
    free(h_np); free(h_na); free(h_hp);
    cudaFree(d_textures); cudaFree(d_tree); cudaFree(d_regrets);
    cudaFree(d_strategy); cudaFree(d_strategy_sum); cudaFree(d_cfv);
    cudaFree(d_node_reach); cudaFree(d_node_players);
    cudaFree(d_node_num_actions); cudaFree(d_hands_per_player);
    return 0;
}

extern "C" int gpu_get_info(int *cuda_cores, size_t *free_mem, size_t *total_mem) {
    int device; cudaGetDevice(&device);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, device);
    *cuda_cores = prop.multiProcessorCount * 128;
    cudaMemGetInfo(free_mem, total_mem);
    printf("[GPU] %s: %d SMs, %d cores, %.1f GB free / %.1f GB total\n",
           prop.name, prop.multiProcessorCount, *cuda_cores,
           *free_mem / 1e9, *total_mem / 1e9);
    return 0;
}
