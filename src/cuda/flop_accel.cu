/**
 * flop_accel.cu — GPU-accelerated flop chance-node evaluation
 *
 * Parallelizes the O(H × T × R × H_opp) showdown evaluation across
 * all turn×river runouts on the GPU.
 *
 * Grid:  (num_turn_cards) blocks
 * Block: (num_river_cards_per_turn) threads, each accumulates over opp hands
 *
 * Each thread computes, for ONE specific (turn, river) runout:
 *   For each traverser hand h:
 *     eval7(board + turn + river + hand_h) vs eval7(board + turn + river + hand_o)
 *     weighted by reach[opp][o]
 *
 * Results are atomically accumulated into per-hand CFV buffers.
 *
 * Memory: all input data in constant/shared memory for fast access.
 */

#include "flop_accel.cuh"
#include <stdio.h>
#include <cuda_runtime.h>

/* ── Device hand evaluation (same as gpu_solver.cu) ───────────────────── */

__device__ uint32_t fa_eval5(int c0, int c1, int c2, int c3, int c4) {
    int r[5] = {c0>>2, c1>>2, c2>>2, c3>>2, c4>>2};
    int s[5] = {c0&3, c1&3, c2&3, c3&3, c4&3};
    for (int i=1;i<5;i++) {
        int kr=r[i], j=i-1;
        while(j>=0 && r[j]<kr) { r[j+1]=r[j]; j--; }
        r[j+1]=kr;
    }
    int flush=(s[0]==s[1]&&s[1]==s[2]&&s[2]==s[3]&&s[3]==s[4]);
    int straight=0, shi=r[0];
    if(r[0]-r[4]==4&&r[0]!=r[1]&&r[1]!=r[2]&&r[2]!=r[3]&&r[3]!=r[4]) straight=1;
    if(r[0]==12&&r[1]==3&&r[2]==2&&r[3]==1&&r[4]==0){straight=1;shi=3;}
    if(straight&&flush) return(9u<<20)|(shi<<16);
    if(flush) return(6u<<20)|(r[0]<<16)|(r[1]<<12)|(r[2]<<8)|(r[3]<<4)|r[4];
    if(straight) return(5u<<20)|(shi<<16);
    int cnt[13]={0}; for(int i=0;i<5;i++) cnt[r[i]]++;
    int q=-1,t=-1,p1=-1,p2=-1;
    for(int i=12;i>=0;i--){
        if(cnt[i]==4)q=i;else if(cnt[i]==3)t=i;
        else if(cnt[i]==2){if(p1<0)p1=i;else p2=i;}
    }
    if(q>=0){int k=-1;for(int i=12;i>=0;i--)if(cnt[i]>0&&i!=q){k=i;break;}return(8u<<20)|(q<<16)|(k<<12);}
    if(t>=0&&p1>=0)return(7u<<20)|(t<<16)|(p1<<12);
    if(t>=0){int k0=-1,k1=-1;for(int i=12;i>=0;i--)if(cnt[i]>0&&i!=t){if(k0<0)k0=i;else k1=i;}return(4u<<20)|(t<<16)|(k0<<12)|(k1<<8);}
    if(p1>=0&&p2>=0){int k=-1;for(int i=12;i>=0;i--)if(cnt[i]>0&&i!=p1&&i!=p2){k=i;break;}return(3u<<20)|(p1<<16)|(p2<<12)|(k<<8);}
    if(p1>=0){int k[3],ki=0;for(int i=12;i>=0&&ki<3;i--)if(cnt[i]>0&&i!=p1)k[ki++]=i;return(2u<<20)|(p1<<16)|(k[0]<<12)|(k[1]<<8)|(k[2]<<4);}
    return(1u<<20)|(r[0]<<16)|(r[1]<<12)|(r[2]<<8)|(r[3]<<4)|r[4];
}

__device__ uint32_t fa_eval7(const int cards[7]) {
    const int c[21][5] = {
        {0,1,2,3,4},{0,1,2,3,5},{0,1,2,3,6},{0,1,2,4,5},{0,1,2,4,6},{0,1,2,5,6},
        {0,1,3,4,5},{0,1,3,4,6},{0,1,3,5,6},{0,1,4,5,6},{0,2,3,4,5},{0,2,3,4,6},
        {0,2,3,5,6},{0,2,4,5,6},{0,3,4,5,6},{1,2,3,4,5},{1,2,3,4,6},{1,2,3,5,6},
        {1,2,4,5,6},{1,3,4,5,6},{2,3,4,5,6}
    };
    uint32_t best=0;
    for(int i=0;i<21;i++){
        uint32_t v=fa_eval5(cards[c[i][0]],cards[c[i][1]],cards[c[i][2]],
                            cards[c[i][3]],cards[c[i][4]]);
        if(v>best)best=v;
    }
    return best;
}

/* ── Main kernel ──────────────────────────────────────────────────────── */

/**
 * Each block = one turn card.
 * Each thread within block = one river card.
 * Thread loops over all traverser hands and opponent hands.
 *
 * Output: d_cfv[hand] accumulated atomically across all threads.
 */
__global__ void flop_eval_kernel(
    const int *d_board,          /* [3] flop cards */
    const int *d_hands_trav,     /* [n_trav][2] traverser hands */
    const int *d_hands_opp,      /* [n_opp][2] opponent hands */
    const float *d_reach_opp,    /* [n_opp] opponent reach */
    int n_trav, int n_opp,
    float half_pot,
    const int *d_turn_cards,     /* [n_turn] valid turn cards */
    const int *d_river_cards,    /* [n_turn][max_river] river cards per turn */
    const int *d_num_river,      /* [n_turn] number of river cards per turn */
    int max_river,
    float *d_cfv                 /* [n_trav] output, atomically accumulated */
) {
    int turn_idx = blockIdx.x;
    int river_thread = threadIdx.x;

    int turn_card = d_turn_cards[turn_idx];
    int num_river = d_num_river[turn_idx];
    if (river_thread >= num_river) return;

    int river_card = d_river_cards[turn_idx * max_river + river_thread];

    /* Full 5-card board */
    int board5[5] = {d_board[0], d_board[1], d_board[2], turn_card, river_card};

    /* For each traverser hand */
    for (int h = 0; h < n_trav; h++) {
        int hc0 = d_hands_trav[h * 2];
        int hc1 = d_hands_trav[h * 2 + 1];

        /* Skip if hand conflicts with board */
        if (hc0 == turn_card || hc1 == turn_card ||
            hc0 == river_card || hc1 == river_card ||
            hc0 == board5[0] || hc1 == board5[0] ||
            hc0 == board5[1] || hc1 == board5[1] ||
            hc0 == board5[2] || hc1 == board5[2])
            continue;

        int cards_h[7] = {board5[0], board5[1], board5[2], board5[3], board5[4], hc0, hc1};
        uint32_t hs = fa_eval7(cards_h);

        float val = 0;
        for (int o = 0; o < n_opp; o++) {
            int oc0 = d_hands_opp[o * 2];
            int oc1 = d_hands_opp[o * 2 + 1];

            /* Card conflict checks */
            if (oc0 == hc0 || oc0 == hc1 || oc1 == hc0 || oc1 == hc1) continue;
            if (oc0 == turn_card || oc1 == turn_card ||
                oc0 == river_card || oc1 == river_card ||
                oc0 == board5[0] || oc1 == board5[0] ||
                oc0 == board5[1] || oc1 == board5[1] ||
                oc0 == board5[2] || oc1 == board5[2])
                continue;

            int cards_o[7] = {board5[0], board5[1], board5[2], board5[3], board5[4], oc0, oc1};
            uint32_t os = fa_eval7(cards_o);

            float w = d_reach_opp[o];
            if (hs > os) val += w * half_pot;
            else if (hs < os) val -= w * half_pot;
        }

        /* Atomic accumulate */
        atomicAdd(&d_cfv[h], val);
    }
}

/* ── Persistent GPU buffers ───────────────────────────────────────────── */

static int *d_board_buf = NULL;
static int *d_hands_trav_buf = NULL;
static int *d_hands_opp_buf = NULL;
static float *d_reach_opp_buf = NULL;
static int *d_turn_cards_buf = NULL;
static int *d_river_cards_buf = NULL;
static int *d_num_river_buf = NULL;
static float *d_cfv_buf = NULL;
static int initialized = 0;

extern "C" FA_EXPORT int flop_accel_init(void) {
    if (initialized) return 0;

    cudaMalloc(&d_board_buf, 3 * sizeof(int));
    cudaMalloc(&d_hands_trav_buf, FA_MAX_HANDS * 2 * sizeof(int));
    cudaMalloc(&d_hands_opp_buf, FA_MAX_HANDS * 2 * sizeof(int));
    cudaMalloc(&d_reach_opp_buf, FA_MAX_HANDS * sizeof(float));
    cudaMalloc(&d_turn_cards_buf, 49 * sizeof(int));
    cudaMalloc(&d_river_cards_buf, 49 * 48 * sizeof(int));
    cudaMalloc(&d_num_river_buf, 49 * sizeof(int));
    cudaMalloc(&d_cfv_buf, FA_MAX_HANDS * sizeof(float));

    initialized = 1;
    printf("[GPU] Flop accelerator initialized\n");
    return 0;
}

extern "C" FA_EXPORT void flop_accel_cleanup(void) {
    if (!initialized) return;
    cudaFree(d_board_buf);
    cudaFree(d_hands_trav_buf);
    cudaFree(d_hands_opp_buf);
    cudaFree(d_reach_opp_buf);
    cudaFree(d_turn_cards_buf);
    cudaFree(d_river_cards_buf);
    cudaFree(d_num_river_buf);
    cudaFree(d_cfv_buf);
    initialized = 0;
}

extern "C" FA_EXPORT int flop_accel_eval(const FlopAccelInput *input, FlopAccelOutput *output) {
    if (!initialized) flop_accel_init();

    int trav = input->traverser;
    int opp = 1 - trav;
    int n_trav = input->num_hands[trav];
    int n_opp = input->num_hands[opp];

    /* Build turn/river card lists */
    int blocked[52] = {0};
    for (int i = 0; i < 3; i++) blocked[input->board[i]] = 1;

    int turn_cards[49];
    int num_turn = 0;
    for (int c = 0; c < 52; c++)
        if (!blocked[c]) turn_cards[num_turn++] = c;

    int river_cards[49][48];
    int num_river[49];
    int max_river = 0;

    for (int ti = 0; ti < num_turn; ti++) {
        int tc = turn_cards[ti];
        num_river[ti] = 0;
        for (int c = 0; c < 52; c++) {
            if (!blocked[c] && c != tc)
                river_cards[ti][num_river[ti]++] = c;
        }
        if (num_river[ti] > max_river) max_river = num_river[ti];
    }

    /* Flatten hands for GPU */
    int h_trav[FA_MAX_HANDS * 2], h_opp[FA_MAX_HANDS * 2];
    for (int h = 0; h < n_trav; h++) {
        h_trav[h*2] = input->hands[trav][h][0];
        h_trav[h*2+1] = input->hands[trav][h][1];
    }
    for (int o = 0; o < n_opp; o++) {
        h_opp[o*2] = input->hands[opp][o][0];
        h_opp[o*2+1] = input->hands[opp][o][1];
    }

    /* Flatten river cards */
    int *river_flat = (int*)malloc(num_turn * max_river * sizeof(int));
    memset(river_flat, 0, num_turn * max_river * sizeof(int));
    for (int ti = 0; ti < num_turn; ti++)
        for (int ri = 0; ri < num_river[ti]; ri++)
            river_flat[ti * max_river + ri] = river_cards[ti][ri];

    /* Copy to GPU */
    cudaMemcpy(d_board_buf, input->board, 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hands_trav_buf, h_trav, n_trav * 2 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hands_opp_buf, h_opp, n_opp * 2 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_reach_opp_buf, input->reach[opp], n_opp * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_turn_cards_buf, turn_cards, num_turn * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_river_cards_buf, river_flat, num_turn * max_river * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_num_river_buf, num_river, num_turn * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_cfv_buf, 0, n_trav * sizeof(float));

    /* Launch: one block per turn card, threads = max river cards */
    int threads = max_river;
    if (threads > 256) threads = 256; /* cap for occupancy */

    flop_eval_kernel<<<num_turn, threads>>>(
        d_board_buf, d_hands_trav_buf, d_hands_opp_buf, d_reach_opp_buf,
        n_trav, n_opp, input->half_pot,
        d_turn_cards_buf, d_river_cards_buf, d_num_river_buf,
        max_river, d_cfv_buf);

    cudaDeviceSynchronize();

    /* Read back and normalize */
    float h_cfv[FA_MAX_HANDS];
    cudaMemcpy(h_cfv, d_cfv_buf, n_trav * sizeof(float), cudaMemcpyDeviceToHost);

    /* Normalize: divide by total number of runouts */
    int total_runouts = 0;
    for (int ti = 0; ti < num_turn; ti++) total_runouts += num_river[ti];

    float inv = (total_runouts > 0) ? 1.0f / total_runouts : 0;
    for (int h = 0; h < n_trav; h++)
        output->cfv[h] = h_cfv[h] * inv;

    free(river_flat);
    return 0;
}
