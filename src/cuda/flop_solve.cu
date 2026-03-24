/**
 * flop_solve.cu — Full flop-through-river GPU solver
 *
 * Phase 1: CPU builds complete multi-street tree (~62K nodes)
 * Phase 2: Upload to GPU, precompute hand strengths at showdown nodes
 * Phase 3: Level-batched Linear CFR iterations on GPU
 * Phase 4: Download root strategy
 */

#include "flop_solve.cuh"
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

#define MAX_RAISES 1

/* ═══════════════════════════════════════════════════════════════════════
 * PART 1: CPU tree construction
 * ═══════════════════════════════════════════════════════════════════════ */

/* Dynamic arrays for tree building */
typedef struct {
    FSNode *nodes;
    int num_nodes;
    int cap_nodes;
    int *children;
    int num_children;
    int cap_children;
} TreeBuilder;

static void tb_init(TreeBuilder *tb) {
    tb->cap_nodes = 8192;
    tb->nodes = (FSNode*)malloc(tb->cap_nodes * sizeof(FSNode));
    tb->num_nodes = 0;
    tb->cap_children = 32768;
    tb->children = (int*)malloc(tb->cap_children * sizeof(int));
    tb->num_children = 0;
}

static int tb_alloc_node(TreeBuilder *tb) {
    if (tb->num_nodes >= tb->cap_nodes) {
        tb->cap_nodes *= 2;
        tb->nodes = (FSNode*)realloc(tb->nodes, tb->cap_nodes * sizeof(FSNode));
    }
    int idx = tb->num_nodes++;
    memset(&tb->nodes[idx], 0, sizeof(FSNode));
    tb->nodes[idx].player = -1;
    return idx;
}

static int tb_alloc_children(TreeBuilder *tb, int count) {
    while (tb->num_children + count > tb->cap_children) {
        tb->cap_children *= 2;
        tb->children = (int*)realloc(tb->children, tb->cap_children * sizeof(int));
    }
    int start = tb->num_children;
    tb->num_children += count;
    return start;
}

/* Build a single-street betting tree, returning root node index */
static int build_betting(TreeBuilder *tb, int is_river,
                         int player, int pot, int stack,
                         int bet0, int bet1, int num_raises, int actions_taken,
                         const float *bet_sizes, int num_bet_sizes,
                         const int *board, int num_board) {
    int to_call = (player == 0) ? (bet1 - bet0) : (bet0 - bet1);
    if (to_call < 0) to_call = 0;

    /* Round over */
    if (actions_taken >= 2 && bet0 == bet1) {
        int idx = tb_alloc_node(tb);
        tb->nodes[idx].type = is_river ? FS_NODE_SHOWDOWN : FS_NODE_CHANCE;
        tb->nodes[idx].pot = pot;
        tb->nodes[idx].bets[0] = bet0;
        tb->nodes[idx].bets[1] = bet1;
        for (int i = 0; i < num_board; i++) tb->nodes[idx].board_cards[i] = board[i];
        tb->nodes[idx].num_board = num_board;
        return idx;
    }

    int node = tb_alloc_node(tb);
    tb->nodes[node].type = FS_NODE_DECISION;
    tb->nodes[node].player = player;
    tb->nodes[node].pot = pot;
    tb->nodes[node].bets[0] = bet0;
    tb->nodes[node].bets[1] = bet1;
    for (int i = 0; i < num_board; i++) tb->nodes[node].board_cards[i] = board[i];
    tb->nodes[node].num_board = num_board;

    /* Collect children in a temp buffer, then copy to flat array */
    int temp_children[8];
    int nc = 0;

    /* Fold */
    if (to_call > 0) {
        int fold = tb_alloc_node(tb);
        tb->nodes[fold].type = FS_NODE_FOLD;
        tb->nodes[fold].player = 1 - player; /* winner */
        tb->nodes[fold].pot = pot;
        tb->nodes[fold].bets[0] = bet0;
        tb->nodes[fold].bets[1] = bet1;
        for (int i = 0; i < num_board; i++) tb->nodes[fold].board_cards[i] = board[i];
        tb->nodes[fold].num_board = num_board;
        temp_children[nc++] = fold;
    }

    /* Check or Call */
    if (to_call == 0) {
        int next = build_betting(tb, is_river, 1-player, pot, stack,
                                 bet0, bet1, num_raises, actions_taken+1,
                                 bet_sizes, num_bet_sizes, board, num_board);
        temp_children[nc++] = next;
    } else {
        int nb0 = bet0, nb1 = bet1;
        if (player == 0) nb0 = bet1; else nb1 = bet0;
        int call_pot = pot + to_call;
        int call_stack = stack - to_call;

        if (actions_taken >= 1) {
            int term = tb_alloc_node(tb);
            tb->nodes[term].type = is_river ? FS_NODE_SHOWDOWN : FS_NODE_CHANCE;
            tb->nodes[term].pot = call_pot;
            tb->nodes[term].bets[0] = nb0;
            tb->nodes[term].bets[1] = nb1;
            for (int i = 0; i < num_board; i++) tb->nodes[term].board_cards[i] = board[i];
            tb->nodes[term].num_board = num_board;
            temp_children[nc++] = term;
        } else {
            int next = build_betting(tb, is_river, 1-player, call_pot, call_stack,
                                     nb0, nb1, num_raises, actions_taken+1,
                                     bet_sizes, num_bet_sizes, board, num_board);
            temp_children[nc++] = next;
        }
    }

    /* Bets/Raises */
    if (num_raises < MAX_RAISES) {
        int added_allin = 0;
        for (int i = 0; i < num_bet_sizes; i++) {
            int ba;
            if (to_call == 0) ba = (int)(bet_sizes[i] * pot);
            else ba = to_call + (int)(bet_sizes[i] * (pot + to_call));
            if (ba >= stack) ba = stack;
            if (ba <= to_call) continue;

            int nb0 = bet0, nb1 = bet1;
            if (player == 0) nb0 += ba; else nb1 += ba;
            int new_pot = pot + ba;
            int new_stack = stack - ba + to_call;

            if (ba >= stack) {
                if (added_allin) continue;
                added_allin = 1;
                /* All-in: opponent folds or calls */
                int ai = tb_alloc_node(tb);
                tb->nodes[ai].type = FS_NODE_DECISION;
                tb->nodes[ai].player = 1-player;
                tb->nodes[ai].pot = new_pot;
                tb->nodes[ai].bets[0] = nb0;
                tb->nodes[ai].bets[1] = nb1;
                for (int b = 0; b < num_board; b++) tb->nodes[ai].board_cards[b] = board[b];
                tb->nodes[ai].num_board = num_board;

                int ai_children[2];
                int ai_nc = 0;

                int f = tb_alloc_node(tb);
                tb->nodes[f].type = FS_NODE_FOLD;
                tb->nodes[f].player = player;
                tb->nodes[f].pot = new_pot;
                tb->nodes[f].bets[0] = nb0; tb->nodes[f].bets[1] = nb1;
                for (int b = 0; b < num_board; b++) tb->nodes[f].board_cards[b] = board[b];
                tb->nodes[f].num_board = num_board;
                ai_children[ai_nc++] = f;

                int cb0 = nb0, cb1 = nb1;
                if (player == 0) cb1 = nb0; else cb0 = nb1;
                int fp = new_pot + (ba - to_call);
                int sd = tb_alloc_node(tb);
                sd = tb->num_nodes - 1;
                tb->nodes[sd].type = is_river ? FS_NODE_SHOWDOWN : FS_NODE_CHANCE;
                tb->nodes[sd].pot = fp;
                tb->nodes[sd].bets[0] = cb0; tb->nodes[sd].bets[1] = cb1;
                for (int b = 0; b < num_board; b++) tb->nodes[sd].board_cards[b] = board[b];
                tb->nodes[sd].num_board = num_board;
                ai_children[ai_nc++] = sd;

                int ai_start = tb_alloc_children(tb, ai_nc);
                for (int j = 0; j < ai_nc; j++) tb->children[ai_start+j] = ai_children[j];
                tb->nodes[ai].first_child = ai_start;
                tb->nodes[ai].num_children = ai_nc;

                temp_children[nc++] = ai;
            } else {
                int next = build_betting(tb, is_river, 1-player, new_pot, new_stack,
                                         nb0, nb1, num_raises+1, actions_taken+1,
                                         bet_sizes, num_bet_sizes, board, num_board);
                temp_children[nc++] = next;
            }
        }

        /* Explicit all-in if not covered */
        if (!added_allin && stack > to_call && nc < 7) {
            int ba = stack;
            int nb0 = bet0, nb1 = bet1;
            if (player == 0) nb0 += ba; else nb1 += ba;
            int new_pot = pot + ba;

            int ai = tb_alloc_node(tb);
            tb->nodes[ai].type = FS_NODE_DECISION;
            tb->nodes[ai].player = 1-player;
            tb->nodes[ai].pot = new_pot;
            tb->nodes[ai].bets[0] = nb0; tb->nodes[ai].bets[1] = nb1;
            for (int b = 0; b < num_board; b++) tb->nodes[ai].board_cards[b] = board[b];
            tb->nodes[ai].num_board = num_board;

            int ai_children[2]; int ai_nc = 0;
            int f = tb_alloc_node(tb);
            tb->nodes[f].type = FS_NODE_FOLD;
            tb->nodes[f].player = player;
            tb->nodes[f].pot = new_pot;
            tb->nodes[f].bets[0] = nb0; tb->nodes[f].bets[1] = nb1;
            for (int b = 0; b < num_board; b++) tb->nodes[f].board_cards[b] = board[b];
            tb->nodes[f].num_board = num_board;
            ai_children[ai_nc++] = f;

            int cb0 = nb0, cb1 = nb1;
            if (player == 0) cb1 = nb0; else cb0 = nb1;
            int fp = new_pot + (ba - to_call);
            int sd = tb_alloc_node(tb);
            tb->nodes[sd].type = is_river ? FS_NODE_SHOWDOWN : FS_NODE_CHANCE;
            tb->nodes[sd].pot = fp;
            tb->nodes[sd].bets[0] = cb0; tb->nodes[sd].bets[1] = cb1;
            for (int b = 0; b < num_board; b++) tb->nodes[sd].board_cards[b] = board[b];
            tb->nodes[sd].num_board = num_board;
            ai_children[ai_nc++] = sd;

            int ai_start = tb_alloc_children(tb, ai_nc);
            for (int j = 0; j < ai_nc; j++) tb->children[ai_start+j] = ai_children[j];
            tb->nodes[ai].first_child = ai_start;
            tb->nodes[ai].num_children = ai_nc;

            temp_children[nc++] = ai;
        }
    }

    /* Store children */
    int start = tb_alloc_children(tb, nc);
    for (int i = 0; i < nc; i++) tb->children[start+i] = temp_children[i];
    tb->nodes[node].first_child = start;
    tb->nodes[node].num_children = nc;

    return node;
}

/* Expand chance nodes: for each unexpanded CHANCE node, add one child per card.
 * Two-pass: first snapshot node data, then expand (safe against realloc). */
static void expand_chance_nodes(TreeBuilder *tb,
                                const float *sub_bet_sizes, int sub_num_bet_sizes) {
    int orig_count = tb->num_nodes;

    /* Pass 1: collect unexpanded chance nodes and snapshot their data */
    typedef struct { int idx; int nb; int board[5]; int pot; int bet0; } ChInfo;
    ChInfo *infos = (ChInfo*)malloc(orig_count * sizeof(ChInfo));
    int num_ch = 0;
    for (int i = 0; i < orig_count; i++) {
        if (tb->nodes[i].type != FS_NODE_CHANCE) continue;
        if (tb->nodes[i].num_board >= 5) continue;
        if (tb->nodes[i].num_children > 0) continue; /* already expanded */
        ChInfo *ci = &infos[num_ch++];
        ci->idx = i;
        ci->nb = tb->nodes[i].num_board;
        for (int b = 0; b < 5; b++) ci->board[b] = tb->nodes[i].board_cards[b];
        ci->pot = tb->nodes[i].pot;
        ci->bet0 = tb->nodes[i].bets[0];
    }

    /* Pass 2: expand using snapshots (tb->nodes may realloc during build_betting) */
    for (int j = 0; j < num_ch; j++) {
        int blocked[52] = {0};
        for (int b = 0; b < infos[j].nb; b++) blocked[infos[j].board[b]] = 1;

        int next_cards[49];
        int num_next = 0;
        for (int c = 0; c < 52; c++)
            if (!blocked[c]) next_cards[num_next++] = c;

        int is_river = (infos[j].nb + 1 == 5);

        int start = tb_alloc_children(tb, num_next);
        for (int ci = 0; ci < num_next; ci++) {
            int new_board[5];
            for (int b = 0; b < infos[j].nb; b++) new_board[b] = infos[j].board[b];
            new_board[infos[j].nb] = next_cards[ci];

            int stack = 10000 - infos[j].bet0;
            if (stack < 0) stack = 0;

            int subtree = build_betting(tb, is_river, 0, infos[j].pot, stack,
                                        0, 0, 0, 0,
                                        sub_bet_sizes, sub_num_bet_sizes,
                                        new_board, infos[j].nb + 1);
            tb->children[start + ci] = subtree;
        }
        /* Node index is still valid (append-only) */
        tb->nodes[infos[j].idx].first_child = start;
        tb->nodes[infos[j].idx].num_children = num_next;
    }

    free(infos);
}

/* ═══════════════════════════════════════════════════════════════════════
 * PART 2: GPU hand evaluation
 * ═══════════════════════════════════════════════════════════════════════ */

__device__ uint32_t fs_eval5(int c0, int c1, int c2, int c3, int c4) {
    int r[5]={c0>>2,c1>>2,c2>>2,c3>>2,c4>>2};
    int s[5]={c0&3,c1&3,c2&3,c3&3,c4&3};
    for(int i=1;i<5;i++){int k=r[i],j=i-1;while(j>=0&&r[j]<k){r[j+1]=r[j];j--;}r[j+1]=k;}
    int fl=(s[0]==s[1]&&s[1]==s[2]&&s[2]==s[3]&&s[3]==s[4]);
    int st=0,sh=r[0];
    if(r[0]-r[4]==4&&r[0]!=r[1]&&r[1]!=r[2]&&r[2]!=r[3]&&r[3]!=r[4])st=1;
    if(r[0]==12&&r[1]==3&&r[2]==2&&r[3]==1&&r[4]==0){st=1;sh=3;}
    if(st&&fl)return(9u<<20)|(sh<<16);
    if(fl)return(6u<<20)|(r[0]<<16)|(r[1]<<12)|(r[2]<<8)|(r[3]<<4)|r[4];
    if(st)return(5u<<20)|(sh<<16);
    int cn[13]={0};for(int i=0;i<5;i++)cn[r[i]]++;
    int q=-1,t=-1,p1=-1,p2=-1;
    for(int i=12;i>=0;i--){if(cn[i]==4)q=i;else if(cn[i]==3)t=i;else if(cn[i]==2){if(p1<0)p1=i;else p2=i;}}
    if(q>=0){int k=-1;for(int i=12;i>=0;i--)if(cn[i]>0&&i!=q){k=i;break;}return(8u<<20)|(q<<16)|(k<<12);}
    if(t>=0&&p1>=0)return(7u<<20)|(t<<16)|(p1<<12);
    if(t>=0){int k0=-1,k1=-1;for(int i=12;i>=0;i--)if(cn[i]>0&&i!=t){if(k0<0)k0=i;else k1=i;}return(4u<<20)|(t<<16)|(k0<<12)|(k1<<8);}
    if(p1>=0&&p2>=0){int k=-1;for(int i=12;i>=0;i--)if(cn[i]>0&&i!=p1&&i!=p2){k=i;break;}return(3u<<20)|(p1<<16)|(p2<<12)|(k<<8);}
    if(p1>=0){int k[3],ki=0;for(int i=12;i>=0&&ki<3;i--)if(cn[i]>0&&i!=p1)k[ki++]=i;return(2u<<20)|(p1<<16)|(k[0]<<12)|(k[1]<<8)|(k[2]<<4);}
    return(1u<<20)|(r[0]<<16)|(r[1]<<12)|(r[2]<<8)|(r[3]<<4)|r[4];
}

__device__ uint32_t fs_eval7(const int c[7]) {
    const int cb[21][5]={
        {0,1,2,3,4},{0,1,2,3,5},{0,1,2,3,6},{0,1,2,4,5},{0,1,2,4,6},{0,1,2,5,6},
        {0,1,3,4,5},{0,1,3,4,6},{0,1,3,5,6},{0,1,4,5,6},{0,2,3,4,5},{0,2,3,4,6},
        {0,2,3,5,6},{0,2,4,5,6},{0,3,4,5,6},{1,2,3,4,5},{1,2,3,4,6},{1,2,3,5,6},
        {1,2,4,5,6},{1,3,4,5,6},{2,3,4,5,6}};
    uint32_t b=0;
    for(int i=0;i<21;i++){uint32_t v=fs_eval5(c[cb[i][0]],c[cb[i][1]],c[cb[i][2]],c[cb[i][3]],c[cb[i][4]]);if(v>b)b=v;}
    return b;
}

/* ═══════════════════════════════════════════════════════════════════════
 * PART 3: GPU kernels for level-batched CFR
 * ═══════════════════════════════════════════════════════════════════════ */

/* Precompute hand strengths at showdown nodes */
__global__ void fs_precompute_strengths(
    const FSNode *nodes, const int *showdown_indices, int num_showdowns,
    const int *hands, int num_hands_0, int num_hands_1,
    uint32_t *strengths  /* [num_showdowns][2][max_hands] */
) {
    int sd_idx = blockIdx.x;
    int hand = threadIdx.x;
    if (sd_idx >= num_showdowns) return;

    int node_idx = showdown_indices[sd_idx];
    const FSNode *node = &nodes[node_idx];

    for (int p = 0; p < 2; p++) {
        int nh = (p == 0) ? num_hands_0 : num_hands_1;
        if (hand >= nh) continue;

        int hc0 = hands[(p * FS_MAX_HANDS + hand) * 2];
        int hc1 = hands[(p * FS_MAX_HANDS + hand) * 2 + 1];

        /* Check board conflict */
        int blocked = 0;
        for (int b = 0; b < node->num_board; b++)
            if (hc0 == node->board_cards[b] || hc1 == node->board_cards[b]) { blocked = 1; break; }

        if (blocked) {
            strengths[(sd_idx * 2 + p) * FS_MAX_HANDS + hand] = 0;
        } else {
            int c7[7] = {node->board_cards[0], node->board_cards[1], node->board_cards[2],
                         node->board_cards[3], node->board_cards[4], hc0, hc1};
            strengths[(sd_idx * 2 + p) * FS_MAX_HANDS + hand] = fs_eval7(c7);
        }
    }
}

/* Regret matching at a decision node */
__global__ void fs_regret_match(
    const FSNode *nodes, float *regrets, float *strategy,
    int node_idx, int max_hands, int num_hands_0, int num_hands_1
) {
    int hand = threadIdx.x + blockIdx.x * blockDim.x;
    const FSNode *n = &nodes[node_idx];
    int nh = (n->player == 0) ? num_hands_0 : num_hands_1;
    if (hand >= nh) return;

    int na = n->num_children;
    int base = node_idx * FS_MAX_ACTIONS * max_hands;

    float sum = 0;
    for (int a = 0; a < na; a++) {
        float r = regrets[base + a * max_hands + hand];
        sum += (r > 0) ? r : 0;
    }
    if (sum > 0) {
        float inv = 1.0f / sum;
        for (int a = 0; a < na; a++) {
            float r = regrets[base + a * max_hands + hand];
            strategy[base + a * max_hands + hand] = ((r > 0) ? r : 0) * inv;
        }
    } else {
        float u = 1.0f / na;
        for (int a = 0; a < na; a++)
            strategy[base + a * max_hands + hand] = u;
    }
}

/* Terminal value: fold */
__global__ void fs_fold_value(
    const FSNode *nodes, float *cfv, const float *reach,
    int node_idx, int traverser, int max_hands,
    int num_hands_0, int num_hands_1,
    const int *hands, int starting_pot
) {
    int hand = threadIdx.x + blockIdx.x * blockDim.x;
    int n_trav = (traverser == 0) ? num_hands_0 : num_hands_1;
    if (hand >= n_trav) return;

    const FSNode *node = &nodes[node_idx];
    int opp = 1 - traverser;
    int n_opp = (opp == 0) ? num_hands_0 : num_hands_1;
    int winner = node->player;
    int loser = 1 - winner;

    float half_start = starting_pot * 0.5f;
    float payoff = (traverser == winner)
        ? (half_start + (float)node->bets[loser])
        : -(half_start + (float)node->bets[traverser]);

    int hc0 = hands[(traverser * FS_MAX_HANDS + hand) * 2];
    int hc1 = hands[(traverser * FS_MAX_HANDS + hand) * 2 + 1];

    float opp_sum = 0;
    for (int o = 0; o < n_opp; o++) {
        int oc0 = hands[(opp * FS_MAX_HANDS + o) * 2];
        int oc1 = hands[(opp * FS_MAX_HANDS + o) * 2 + 1];
        if (hc0==oc0||hc0==oc1||hc1==oc0||hc1==oc1) continue;
        /* Check board conflict for opponent */
        int blocked = 0;
        for (int b = 0; b < node->num_board; b++)
            if (oc0 == node->board_cards[b] || oc1 == node->board_cards[b]) { blocked = 1; break; }
        if (blocked) continue;
        opp_sum += reach[node_idx * max_hands + o];
    }
    cfv[node_idx * max_hands + hand] = opp_sum * payoff;
}

/* Terminal value: showdown */
__global__ void fs_showdown_value(
    const FSNode *nodes, float *cfv, const float *reach,
    int node_idx, int traverser, int max_hands,
    int num_hands_0, int num_hands_1,
    const int *hands, const uint32_t *strengths,
    int sd_local_idx  /* index into showdown_indices */
) {
    int hand = threadIdx.x + blockIdx.x * blockDim.x;
    int n_trav = (traverser == 0) ? num_hands_0 : num_hands_1;
    if (hand >= n_trav) return;

    const FSNode *node = &nodes[node_idx];
    int opp = 1 - traverser;
    int n_opp = (opp == 0) ? num_hands_0 : num_hands_1;
    float half_pot = node->pot * 0.5f;

    uint32_t hs = strengths[(sd_local_idx * 2 + traverser) * FS_MAX_HANDS + hand];
    if (hs == 0) { cfv[node_idx * max_hands + hand] = 0; return; }

    int hc0 = hands[(traverser * FS_MAX_HANDS + hand) * 2];
    int hc1 = hands[(traverser * FS_MAX_HANDS + hand) * 2 + 1];

    float val = 0;
    for (int o = 0; o < n_opp; o++) {
        int oc0 = hands[(opp * FS_MAX_HANDS + o) * 2];
        int oc1 = hands[(opp * FS_MAX_HANDS + o) * 2 + 1];
        if (hc0==oc0||hc0==oc1||hc1==oc0||hc1==oc1) continue;
        uint32_t os = strengths[(sd_local_idx * 2 + opp) * FS_MAX_HANDS + o];
        if (os == 0) continue;
        float w = reach[node_idx * max_hands + o];
        if (hs > os) val += w * half_pot;
        else if (hs < os) val -= w * half_pot;
    }
    cfv[node_idx * max_hands + hand] = val;
}

/* Propagate reach through a decision node (top-down) */
__global__ void fs_propagate_reach(
    const FSNode *nodes, const int *children_arr,
    float *reach, const float *strategy,
    int node_idx, int opp, int max_hands,
    int num_hands_0, int num_hands_1
) {
    int hand = threadIdx.x + blockIdx.x * blockDim.x;
    const FSNode *n = &nodes[node_idx];
    int acting = n->player;
    int na = n->num_children;

    if (acting == opp) {
        int nh = (opp == 0) ? num_hands_0 : num_hands_1;
        if (hand >= nh) return;
        float pr = reach[node_idx * max_hands + hand];
        int base = node_idx * FS_MAX_ACTIONS * max_hands;
        for (int a = 0; a < na; a++) {
            int child = children_arr[n->first_child + a];
            float s = strategy[base + a * max_hands + hand];
            reach[child * max_hands + hand] = pr * s;
        }
    } else {
        /* Non-acting player: copy reach to all children */
        int nh = (opp == 0) ? num_hands_0 : num_hands_1;
        if (hand >= nh) return;
        float pr = reach[node_idx * max_hands + hand];
        for (int a = 0; a < na; a++) {
            int child = children_arr[n->first_child + a];
            reach[child * max_hands + hand] = pr;
        }
    }
}

/* Propagate reach through a chance node (top-down) — copy to all children */
__global__ void fs_propagate_reach_chance(
    const FSNode *nodes, const int *children_arr,
    float *reach, int node_idx, int max_hands, int num_hands_opp
) {
    int hand = threadIdx.x + blockIdx.x * blockDim.x;
    if (hand >= num_hands_opp) return;
    const FSNode *n = &nodes[node_idx];
    float pr = reach[node_idx * max_hands + hand];
    for (int a = 0; a < n->num_children; a++) {
        int child = children_arr[n->first_child + a];
        reach[child * max_hands + hand] = pr;
    }
}

/* Propagate CFV up from children (bottom-up) for a decision node */
__global__ void fs_propagate_cfv(
    const FSNode *nodes, const int *children_arr,
    float *cfv, const float *strategy,
    int node_idx, int traverser, int max_hands,
    int num_hands_0, int num_hands_1
) {
    int hand = threadIdx.x + blockIdx.x * blockDim.x;
    int n_trav = (traverser == 0) ? num_hands_0 : num_hands_1;
    if (hand >= n_trav) return;

    const FSNode *n = &nodes[node_idx];
    int na = n->num_children;

    if (n->player == traverser) {
        float val = 0;
        int base = node_idx * FS_MAX_ACTIONS * max_hands;
        for (int a = 0; a < na; a++) {
            int child = children_arr[n->first_child + a];
            float cv = cfv[child * max_hands + hand];
            val += strategy[base + a * max_hands + hand] * cv;
        }
        cfv[node_idx * max_hands + hand] = val;
    } else {
        float val = 0;
        for (int a = 0; a < na; a++) {
            int child = children_arr[n->first_child + a];
            val += cfv[child * max_hands + hand];
        }
        cfv[node_idx * max_hands + hand] = val;
    }
}

/* Propagate CFV up from chance node children (average) */
__global__ void fs_propagate_cfv_chance(
    const FSNode *nodes, const int *children_arr,
    float *cfv, int node_idx, int traverser, int max_hands,
    int n_trav
) {
    int hand = threadIdx.x + blockIdx.x * blockDim.x;
    if (hand >= n_trav) return;

    const FSNode *n = &nodes[node_idx];
    float val = 0;
    for (int a = 0; a < n->num_children; a++) {
        int child = children_arr[n->first_child + a];
        val += cfv[child * max_hands + hand];
    }
    /* Average over dealt cards */
    cfv[node_idx * max_hands + hand] = val / (float)n->num_children;
}

/* Update regrets at a decision node */
__global__ void fs_update_regrets(
    const FSNode *nodes, const int *children_arr,
    float *regrets, float *strategy_sum, float *cfv, const float *strategy,
    int node_idx, int traverser, int max_hands,
    int num_hands_0, int num_hands_1, int iteration
) {
    int hand = threadIdx.x + blockIdx.x * blockDim.x;
    const FSNode *n = &nodes[node_idx];
    if (n->player != traverser) return;
    int nh = (traverser == 0) ? num_hands_0 : num_hands_1;
    if (hand >= nh) return;

    int na = n->num_children;
    float nv = cfv[node_idx * max_hands + hand];
    int base = node_idx * FS_MAX_ACTIONS * max_hands;

    for (int a = 0; a < na; a++) {
        int child = children_arr[n->first_child + a];
        float cv = cfv[child * max_hands + hand];
        regrets[base + a * max_hands + hand] += cv - nv;
    }

    /* Linear CFR discount */
    float d = (float)iteration / ((float)iteration + 1.0f);
    for (int a = 0; a < na; a++)
        regrets[base + a * max_hands + hand] *= d;

    /* Accumulate weighted strategy sum */
    for (int a = 0; a < na; a++)
        strategy_sum[base + a * max_hands + hand] +=
            (float)iteration * strategy[base + a * max_hands + hand];
}

/* ═══════════════════════════════════════════════════════════════════════
 * PART 3B: BATCHED kernels (one launch per BFS level)
 *
 * Grid: (num_nodes_in_batch), Block: (max_hands)
 * blockIdx.x → selects which node from the batch
 * threadIdx.x → selects which hand
 * ═══════════════════════════════════════════════════════════════════════ */

/* Batched regret matching */
__global__ void fs_batch_regret_match(
    const FSNode *nodes, const int *batch_nodes, int batch_size,
    float *regrets, float *strategy, int max_hands,
    int num_hands_0, int num_hands_1
) {
    int bi = blockIdx.x;
    int hand = threadIdx.x;
    if (bi >= batch_size) return;
    int node_idx = batch_nodes[bi];
    const FSNode *n = &nodes[node_idx];
    if (n->type != FS_NODE_DECISION) return;
    int nh = (n->player == 0) ? num_hands_0 : num_hands_1;
    if (hand >= nh) return;
    int na = n->num_children;
    int base = node_idx * FS_MAX_ACTIONS * max_hands;
    float sum = 0;
    for (int a = 0; a < na; a++) {
        float r = regrets[base + a * max_hands + hand];
        sum += (r > 0) ? r : 0;
    }
    if (sum > 0) {
        float inv = 1.0f / sum;
        for (int a = 0; a < na; a++) {
            float r = regrets[base + a * max_hands + hand];
            strategy[base + a * max_hands + hand] = ((r > 0) ? r : 0) * inv;
        }
    } else {
        float u = 1.0f / na;
        for (int a = 0; a < na; a++)
            strategy[base + a * max_hands + hand] = u;
    }
}

/* Batched reach propagation (handles both decision and chance nodes) */
__global__ void fs_batch_propagate_reach(
    const FSNode *nodes, const int *children_arr, const int *batch_nodes, int batch_size,
    float *reach, const float *strategy, int opp, int max_hands,
    int num_hands_0, int num_hands_1
) {
    int bi = blockIdx.x;
    int hand = threadIdx.x;
    if (bi >= batch_size) return;
    int node_idx = batch_nodes[bi];
    const FSNode *n = &nodes[node_idx];

    if (n->type == FS_NODE_DECISION) {
        int acting = n->player;
        int na = n->num_children;
        if (acting == opp) {
            int nh = (opp == 0) ? num_hands_0 : num_hands_1;
            if (hand >= nh) return;
            float pr = reach[node_idx * max_hands + hand];
            int base = node_idx * FS_MAX_ACTIONS * max_hands;
            for (int a = 0; a < na; a++) {
                int child = children_arr[n->first_child + a];
                reach[child * max_hands + hand] = pr * strategy[base + a * max_hands + hand];
            }
        } else {
            int nh = (opp == 0) ? num_hands_0 : num_hands_1;
            if (hand >= nh) return;
            float pr = reach[node_idx * max_hands + hand];
            for (int a = 0; a < na; a++) {
                int child = children_arr[n->first_child + a];
                reach[child * max_hands + hand] = pr;
            }
        }
    } else if (n->type == FS_NODE_CHANCE) {
        int nh = (opp == 0) ? num_hands_0 : num_hands_1;
        if (hand >= nh) return;
        float pr = reach[node_idx * max_hands + hand];
        for (int a = 0; a < n->num_children; a++) {
            int child = children_arr[n->first_child + a];
            reach[child * max_hands + hand] = pr;
        }
    }
}

/* Batched terminal value: fold nodes.
 * Separated from showdown for clarity and to allow different block sizes. */
__global__ void fs_batch_fold_value(
    const FSNode *nodes, const int *batch_nodes, int batch_size,
    float *cfv, const float *reach, int traverser, int max_hands,
    int num_hands_0, int num_hands_1,
    const int *hands, int starting_pot
) {
    __shared__ float s_opp_reach[FS_MAX_HANDS];
    __shared__ int s_opp_cards[FS_MAX_HANDS * 2];

    int bi = blockIdx.x;
    int hand = threadIdx.x;
    if (bi >= batch_size) return;
    int node_idx = batch_nodes[bi];
    const FSNode *node = &nodes[node_idx];
    int opp = 1 - traverser;
    int n_opp = (opp == 0) ? num_hands_0 : num_hands_1;
    int n_trav = (traverser == 0) ? num_hands_0 : num_hands_1;

    if (hand < n_opp) {
        s_opp_reach[hand] = reach[node_idx * max_hands + hand];
        s_opp_cards[hand * 2] = hands[(opp * FS_MAX_HANDS + hand) * 2];
        s_opp_cards[hand * 2 + 1] = hands[(opp * FS_MAX_HANDS + hand) * 2 + 1];
    }
    __syncthreads();

    if (hand >= n_trav) return;
    int hc0 = hands[(traverser * FS_MAX_HANDS + hand) * 2];
    int hc1 = hands[(traverser * FS_MAX_HANDS + hand) * 2 + 1];

    int winner = node->player;
    int loser = 1 - winner;
    float half_start = starting_pot * 0.5f;
    float payoff = (traverser == winner)
        ? (half_start + (float)node->bets[loser])
        : -(half_start + (float)node->bets[traverser]);
    float opp_sum = 0;
    for (int o = 0; o < n_opp; o++) {
        int oc0 = s_opp_cards[o * 2], oc1 = s_opp_cards[o * 2 + 1];
        if (hc0==oc0||hc0==oc1||hc1==oc0||hc1==oc1) continue;
        int blocked = 0;
        for (int b = 0; b < node->num_board; b++)
            if (oc0==node->board_cards[b]||oc1==node->board_cards[b]) {blocked=1;break;}
        if (blocked) continue;
        opp_sum += s_opp_reach[o];
    }
    cfv[node_idx * max_hands + hand] = opp_sum * payoff;
}

/* Batched showdown value with shared memory.
 * Simple O(N*M) but with opponent data cached in shared memory. */
__global__ void fs_batch_showdown_value(
    const FSNode *nodes, const int *batch_nodes, int batch_size,
    float *cfv, const float *reach, int traverser, int max_hands,
    int num_hands_0, int num_hands_1,
    const int *hands,
    const uint32_t *strengths, const int *sd_local_map,
    const int *unused
) {
    __shared__ float s_opp_reach[FS_MAX_HANDS];
    __shared__ uint32_t s_opp_str[FS_MAX_HANDS];
    __shared__ int s_opp_cards[FS_MAX_HANDS * 2];

    int bi = blockIdx.x;
    int hand = threadIdx.x;
    if (bi >= batch_size) return;
    int node_idx = batch_nodes[bi];
    const FSNode *node = &nodes[node_idx];
    int opp = 1 - traverser;
    int n_opp = (opp == 0) ? num_hands_0 : num_hands_1;
    int n_trav = (traverser == 0) ? num_hands_0 : num_hands_1;
    int sd_idx = sd_local_map[node_idx];

    /* Cooperatively load opponent data into shared memory */
    if (hand < n_opp) {
        s_opp_reach[hand] = reach[node_idx * max_hands + hand];
        s_opp_str[hand] = strengths[(sd_idx * 2 + opp) * FS_MAX_HANDS + hand];
        s_opp_cards[hand * 2] = hands[(opp * FS_MAX_HANDS + hand) * 2];
        s_opp_cards[hand * 2 + 1] = hands[(opp * FS_MAX_HANDS + hand) * 2 + 1];
    }
    __syncthreads();

    if (hand >= n_trav) return;
    uint32_t hs = strengths[(sd_idx * 2 + traverser) * FS_MAX_HANDS + hand];
    if (hs == 0) { cfv[node_idx * max_hands + hand] = 0; return; }

    int hc0 = hands[(traverser * FS_MAX_HANDS + hand) * 2];
    int hc1 = hands[(traverser * FS_MAX_HANDS + hand) * 2 + 1];
    float half_pot = node->pot * 0.5f;
    float val = 0;

    for (int o = 0; o < n_opp; o++) {
        int oc0 = s_opp_cards[o * 2], oc1 = s_opp_cards[o * 2 + 1];
        if (hc0==oc0||hc0==oc1||hc1==oc0||hc1==oc1) continue;
        uint32_t os = s_opp_str[o];
        if (os == 0) continue;
        float w = s_opp_reach[o];
        if (hs > os) val += w * half_pot;
        else if (hs < os) val -= w * half_pot;
    }
    cfv[node_idx * max_hands + hand] = val;
}

/* Batched CFV propagation + regret update (bottom-up).
 * `d_iteration` is a device pointer — allows CUDA graph replay with different iteration values. */
__global__ void fs_batch_propagate_cfv(
    const FSNode *nodes, const int *children_arr, const int *batch_nodes, int batch_size,
    float *cfv, float *regrets, float *strategy_sum, const float *strategy,
    int traverser, int max_hands, int num_hands_0, int num_hands_1, const int *d_iteration
) {
    int iteration = *d_iteration;
    int bi = blockIdx.x;
    int hand = threadIdx.x;
    if (bi >= batch_size) return;
    int node_idx = batch_nodes[bi];
    const FSNode *n = &nodes[node_idx];
    int n_trav = (traverser == 0) ? num_hands_0 : num_hands_1;
    if (hand >= n_trav) return;

    if (n->type == FS_NODE_DECISION) {
        int na = n->num_children;
        if (n->player == traverser) {
            float val = 0;
            int base = node_idx * FS_MAX_ACTIONS * max_hands;
            for (int a = 0; a < na; a++) {
                int child = children_arr[n->first_child + a];
                val += strategy[base + a * max_hands + hand] * cfv[child * max_hands + hand];
            }
            cfv[node_idx * max_hands + hand] = val;
            /* Update regrets */
            for (int a = 0; a < na; a++) {
                int child = children_arr[n->first_child + a];
                regrets[base + a * max_hands + hand] += cfv[child * max_hands + hand] - val;
            }
            /* Linear CFR discount */
            float d = (float)iteration / ((float)iteration + 1.0f);
            for (int a = 0; a < na; a++)
                regrets[base + a * max_hands + hand] *= d;
            /* Accumulate weighted strategy sum */
            for (int a = 0; a < na; a++)
                strategy_sum[base + a * max_hands + hand] +=
                    (float)iteration * strategy[base + a * max_hands + hand];
        } else {
            float val = 0;
            for (int a = 0; a < na; a++) {
                int child = children_arr[n->first_child + a];
                val += cfv[child * max_hands + hand];
            }
            cfv[node_idx * max_hands + hand] = val;
        }
    } else if (n->type == FS_NODE_CHANCE) {
        float val = 0;
        for (int a = 0; a < n->num_children; a++) {
            int child = children_arr[n->first_child + a];
            val += cfv[child * max_hands + hand];
        }
        cfv[node_idx * max_hands + hand] = val / (float)n->num_children;
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * PART 4: Host orchestration
 * ═══════════════════════════════════════════════════════════════════════ */

extern "C" FS_EXPORT int fs_build_tree(
    const int *flop_board, int starting_pot, int effective_stack,
    const float *bet_sizes, int num_bet_sizes,
    FSTreeData *tree_out
) {
    memset(tree_out, 0, sizeof(FSTreeData));

    TreeBuilder tb;
    tb_init(&tb);

    /* Build flop betting tree (full bet sizes) */
    build_betting(&tb, 0, 0, starting_pot, effective_stack,
                  0, 0, 0, 0, bet_sizes, num_bet_sizes,
                  flop_board, 3);

    printf("[FS] After flop: %d nodes\n", tb.num_nodes);

    /* Turn: use simplified sizing (1 bet size) to keep tree manageable.
     * Full sizing on turn would create ~5M nodes which is too large.
     * Single 75% pot bet captures most of the strategic value. */
    float turn_bet[] = {0.75f};
    expand_chance_nodes(&tb, turn_bet, 1);
    printf("[FS] After turn expansion: %d nodes\n", tb.num_nodes);

    /* River: also simplified (1 bet size) */
    float river_bet[] = {0.75f};
    expand_chance_nodes(&tb, river_bet, 1);
    printf("[FS] After river expansion: %d nodes (children: %d)\n",
           tb.num_nodes, tb.num_children);

    /* Transfer to output */
    tree_out->nodes = tb.nodes;
    tree_out->children = tb.children;
    tree_out->num_nodes = tb.num_nodes;
    tree_out->num_children_total = tb.num_children;

    /* BFS level order */
    tree_out->level_order = (int*)malloc(tb.num_nodes * sizeof(int));
    tree_out->node_depth = (int*)calloc(tb.num_nodes, sizeof(int));
    {
        int *queue = (int*)malloc(tb.num_nodes * sizeof(int));
        int qh = 0, qt = 0, li = 0;
        queue[qt++] = 0;
        while (qh < qt) {
            int n = queue[qh++];
            tree_out->level_order[li++] = n;
            int d = tree_out->node_depth[n];
            if (d > tree_out->max_depth) tree_out->max_depth = d;
            FSNode *nd = &tb.nodes[n];
            for (int a = 0; a < nd->num_children; a++) {
                int child = tb.children[nd->first_child + a];
                tree_out->node_depth[child] = d + 1;
                queue[qt++] = child;
            }
        }
        free(queue);
    }

    /* Find decision and showdown nodes */
    tree_out->num_decision_nodes = 0;
    tree_out->num_showdown_nodes = 0;
    for (int i = 0; i < tb.num_nodes; i++) {
        if (tb.nodes[i].type == FS_NODE_DECISION) tree_out->num_decision_nodes++;
        if (tb.nodes[i].type == FS_NODE_SHOWDOWN) tree_out->num_showdown_nodes++;
    }
    tree_out->decision_node_indices = (int*)malloc(tree_out->num_decision_nodes * sizeof(int));
    tree_out->showdown_node_indices = (int*)malloc(tree_out->num_showdown_nodes * sizeof(int));
    int di = 0, si = 0;
    for (int i = 0; i < tb.num_nodes; i++) {
        if (tb.nodes[i].type == FS_NODE_DECISION) tree_out->decision_node_indices[di++] = i;
        if (tb.nodes[i].type == FS_NODE_SHOWDOWN) tree_out->showdown_node_indices[si++] = i;
    }

    printf("[FS] Tree: %d nodes, %d decision, %d showdown, depth=%d\n",
           tb.num_nodes, tree_out->num_decision_nodes,
           tree_out->num_showdown_nodes, tree_out->max_depth);

    return 0;
}

extern "C" FS_EXPORT int fs_solve_gpu(
    FSTreeData *td, int max_iterations, FSOutput *output
) {
    int N = td->num_nodes;
    int nh0 = td->num_hands[0], nh1 = td->num_hands[1];
    int max_h = nh0 > nh1 ? nh0 : nh1;

    printf("[FS] GPU solve: %d nodes, hands=[%d,%d], %d iterations\n",
           N, nh0, nh1, max_iterations);

    /* Device memory */
    FSNode *d_nodes;
    int *d_children, *d_hands, *d_showdown_idx;
    float *d_regrets, *d_strategy, *d_strategy_sum, *d_cfv, *d_reach;
    float *d_weights;
    uint32_t *d_strengths;

    size_t node_sz = N * sizeof(FSNode);
    size_t child_sz = td->num_children_total * sizeof(int);
    size_t state_sz = (size_t)N * FS_MAX_ACTIONS * max_h * sizeof(float);
    size_t cfv_sz = (size_t)N * max_h * sizeof(float);
    size_t hands_sz = 2 * FS_MAX_HANDS * 2 * sizeof(int);
    size_t weights_sz = 2 * FS_MAX_HANDS * sizeof(float);
    size_t strength_sz = (size_t)td->num_showdown_nodes * 2 * FS_MAX_HANDS * sizeof(uint32_t);

    printf("[FS] Memory: state=%.1f MB, cfv=%.1f MB, strengths=%.1f MB, total=%.1f MB\n",
           state_sz*3/1e6, cfv_sz*2/1e6, strength_sz/1e6,
           (state_sz*3 + cfv_sz*2 + node_sz + child_sz + strength_sz)/1e6);

    CUDA_CHECK(cudaMalloc(&d_nodes, node_sz));
    CUDA_CHECK(cudaMalloc(&d_children, child_sz));
    CUDA_CHECK(cudaMalloc(&d_hands, hands_sz));
    CUDA_CHECK(cudaMalloc(&d_weights, weights_sz));
    CUDA_CHECK(cudaMalloc(&d_regrets, state_sz));
    CUDA_CHECK(cudaMalloc(&d_strategy, state_sz));
    CUDA_CHECK(cudaMalloc(&d_strategy_sum, state_sz));
    CUDA_CHECK(cudaMalloc(&d_cfv, cfv_sz));
    CUDA_CHECK(cudaMalloc(&d_reach, cfv_sz));
    CUDA_CHECK(cudaMalloc(&d_showdown_idx, td->num_showdown_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_strengths, strength_sz));

    /* Flatten hands for GPU */
    int *h_hands = (int*)calloc(2 * FS_MAX_HANDS * 2, sizeof(int));
    float *h_weights = (float*)calloc(2 * FS_MAX_HANDS, sizeof(float));
    for (int p = 0; p < 2; p++) {
        for (int h = 0; h < td->num_hands[p]; h++) {
            h_hands[(p * FS_MAX_HANDS + h) * 2] = td->hands[p][h][0];
            h_hands[(p * FS_MAX_HANDS + h) * 2 + 1] = td->hands[p][h][1];
            h_weights[p * FS_MAX_HANDS + h] = td->weights[p][h];
        }
    }

    CUDA_CHECK(cudaMemcpy(d_nodes, td->nodes, node_sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_children, td->children, child_sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_hands, h_hands, hands_sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weights, h_weights, weights_sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_showdown_idx, td->showdown_node_indices,
                           td->num_showdown_nodes * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_regrets, 0, state_sz));
    CUDA_CHECK(cudaMemset(d_strategy, 0, state_sz));
    CUDA_CHECK(cudaMemset(d_strategy_sum, 0, state_sz));

    /* Precompute hand strengths at all showdown nodes */
    printf("[FS] Precomputing strengths at %d showdown nodes...\n", td->num_showdown_nodes);
    if (td->num_showdown_nodes > 0) {
        fs_precompute_strengths<<<td->num_showdown_nodes, FS_MAX_HANDS>>>(
            d_nodes, d_showdown_idx, td->num_showdown_nodes,
            d_hands, nh0, nh1, d_strengths);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    printf("[FS] Strengths done\n");

    /* Build showdown node local index map */
    int *sd_local = (int*)calloc(N, sizeof(int));
    for (int i = 0; i < td->num_showdown_nodes; i++)
        sd_local[td->showdown_node_indices[i]] = i;

    int threads = 256;
    int blocks = (max_h + threads - 1) / threads;

    /* ── Build level groups: group nodes by BFS depth ───── */
    /* This lets us launch ONE kernel per level instead of per node */
    int **level_nodes = (int**)calloc(td->max_depth + 1, sizeof(int*));
    int *level_counts = (int*)calloc(td->max_depth + 1, sizeof(int));
    int *level_caps = (int*)calloc(td->max_depth + 1, sizeof(int));
    for (int i = 0; i < N; i++) {
        int d = td->node_depth[i];
        if (level_counts[d] >= level_caps[d]) {
            level_caps[d] = level_caps[d] ? level_caps[d] * 2 : 256;
            level_nodes[d] = (int*)realloc(level_nodes[d], level_caps[d] * sizeof(int));
        }
        level_nodes[d][level_counts[d]++] = i;
    }

    /* Upload per-level node index arrays to GPU */
    int **d_level_nodes = (int**)malloc((td->max_depth + 1) * sizeof(int*));
    for (int d = 0; d <= td->max_depth; d++) {
        if (level_counts[d] > 0) {
            CUDA_CHECK(cudaMalloc(&d_level_nodes[d], level_counts[d] * sizeof(int)));
            CUDA_CHECK(cudaMemcpy(d_level_nodes[d], level_nodes[d],
                                   level_counts[d] * sizeof(int), cudaMemcpyHostToDevice));
        } else {
            d_level_nodes[d] = NULL;
        }
    }

    /* Upload sd_local map to GPU */
    int *d_sd_local;
    CUDA_CHECK(cudaMalloc(&d_sd_local, N * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_sd_local, sd_local, N * sizeof(int), cudaMemcpyHostToDevice));

    int starting_pot = td->nodes[0].pot;

    /* ── Upload batch index arrays to GPU ───────────────── */
    int *d_dec_nodes;
    CUDA_CHECK(cudaMalloc(&d_dec_nodes, td->num_decision_nodes * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_dec_nodes, td->decision_node_indices,
                           td->num_decision_nodes * sizeof(int), cudaMemcpyHostToDevice));

    int *h_terminal_nodes = (int*)malloc(N * sizeof(int));
    int num_terminals = 0;
    for (int i = 0; i < N; i++)
        if (td->nodes[i].type == FS_NODE_FOLD || td->nodes[i].type == FS_NODE_SHOWDOWN)
            h_terminal_nodes[num_terminals++] = i;
    int *d_terminal_nodes;
    CUDA_CHECK(cudaMalloc(&d_terminal_nodes, num_terminals * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_terminal_nodes, h_terminal_nodes,
                           num_terminals * sizeof(int), cudaMemcpyHostToDevice));

    /* Pre-upload opponent weight arrays for both traversers */
    float *d_opp_weights[2];
    for (int t = 0; t < 2; t++) {
        int opp = 1 - t;
        int n_opp = (opp == 0) ? nh0 : nh1;
        CUDA_CHECK(cudaMalloc(&d_opp_weights[t], n_opp * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_opp_weights[t], h_weights + opp * FS_MAX_HANDS,
                               n_opp * sizeof(float), cudaMemcpyHostToDevice));
    }

    int block_size = 128;
    if (max_h > 128) block_size = ((max_h + 31) / 32) * 32;

    /* Separate fold and showdown terminal node lists */
    int *h_fold_nodes = (int*)malloc(N * sizeof(int));
    int *h_sd_nodes = (int*)malloc(N * sizeof(int));
    int num_folds = 0, num_sds = 0;
    for (int i = 0; i < N; i++) {
        if (td->nodes[i].type == FS_NODE_FOLD) h_fold_nodes[num_folds++] = i;
        if (td->nodes[i].type == FS_NODE_SHOWDOWN) h_sd_nodes[num_sds++] = i;
    }
    int *d_fold_nodes, *d_sd_nodes_list;
    CUDA_CHECK(cudaMalloc(&d_fold_nodes, (num_folds + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sd_nodes_list, (num_sds + 1) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_fold_nodes, h_fold_nodes, num_folds * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sd_nodes_list, h_sd_nodes, num_sds * sizeof(int), cudaMemcpyHostToDevice));

    int *d_sorted_opp_idx = NULL; /* unused — kept for API compat */

    /* Device-side iteration counter (for CUDA graph compatibility) */
    int *d_iter;
    CUDA_CHECK(cudaMalloc(&d_iter, sizeof(int)));

    printf("[FS] %d levels, %d folds, %d showdowns, block=%d\n",
           td->max_depth + 1, num_folds, num_sds, block_size);
    fflush(stdout);

    /* ══════════════════════════════════════════════════════
     * CUDA GRAPH CAPTURE
     *
     * Capture one full iteration (both traversers) as a graph.
     * The only per-iteration state is `d_iter` which we update
     * via cudaMemcpyAsync before each graph launch.
     * ══════════════════════════════════════════════════════ */
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cudaGraph_t graph = NULL;
    cudaGraphExec_t graph_exec = NULL;

    /* Run iteration 1 with graph capture */
    {
        int iter_val = 1;
        CUDA_CHECK(cudaMemcpy(d_iter, &iter_val, sizeof(int), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

        for (int trav = 0; trav < 2; trav++) {
            int opp = 1 - trav;

            fs_batch_regret_match<<<td->num_decision_nodes, block_size, 0, stream>>>(
                d_nodes, d_dec_nodes, td->num_decision_nodes,
                d_regrets, d_strategy, max_h, nh0, nh1);

            cudaMemsetAsync(d_reach, 0, cfv_sz, stream);
            cudaMemcpyAsync(d_reach, d_opp_weights[trav],
                            ((opp == 0) ? nh0 : nh1) * sizeof(float),
                            cudaMemcpyDeviceToDevice, stream);

            for (int d = 0; d <= td->max_depth; d++) {
                if (level_counts[d] > 0) {
                    fs_batch_propagate_reach<<<level_counts[d], block_size, 0, stream>>>(
                        d_nodes, d_children, d_level_nodes[d], level_counts[d],
                        d_reach, d_strategy, opp, max_h, nh0, nh1);
                }
            }

            cudaMemsetAsync(d_cfv, 0, cfv_sz, stream);
            if (num_folds > 0) {
                fs_batch_fold_value<<<num_folds, block_size, 0, stream>>>(
                    d_nodes, d_fold_nodes, num_folds,
                    d_cfv, d_reach, trav, max_h, nh0, nh1,
                    d_hands, starting_pot);
            }
            if (num_sds > 0) {
                fs_batch_showdown_value<<<num_sds, block_size, 0, stream>>>(
                    d_nodes, d_sd_nodes_list, num_sds,
                    d_cfv, d_reach, trav, max_h, nh0, nh1,
                    d_hands, d_strengths, d_sd_local, d_sorted_opp_idx);
            }

            for (int d = td->max_depth; d >= 0; d--) {
                if (level_counts[d] > 0) {
                    fs_batch_propagate_cfv<<<level_counts[d], block_size, 0, stream>>>(
                        d_nodes, d_children, d_level_nodes[d], level_counts[d],
                        d_cfv, d_regrets, d_strategy_sum, d_strategy,
                        trav, max_h, nh0, nh1, d_iter);
                }
            }
        }

        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0));
    }

    printf("[FS] CUDA graph captured. Starting iterations...\n");
    fflush(stdout);

    /* ── Main CFR loop: replay graph ─────────────────── */
    for (int iter = 1; iter <= max_iterations; iter++) {
        /* Update iteration counter on device */
        CUDA_CHECK(cudaMemcpy(d_iter, &iter, sizeof(int), cudaMemcpyHostToDevice));

        /* Launch the captured graph */
        CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        if (iter % 50 == 0 || iter == max_iterations || iter == 1) {
            printf("[FS] iter %d/%d\n", iter, max_iterations);
            fflush(stdout);
        }
    }

    /* Cleanup graph */
    cudaGraphExecDestroy(graph_exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    free(h_fold_nodes); free(h_sd_nodes);

    /* Extract root strategy */
    int root_na = td->nodes[0].num_children;
    int root_player = td->nodes[0].player;
    int root_nh = (root_player == 0) ? nh0 : nh1;

    output->root_num_actions = root_na;
    output->root_player = root_player;
    output->root_strategy = (float*)malloc(root_na * root_nh * sizeof(float));

    float *h_strat = (float*)malloc(state_sz);
    CUDA_CHECK(cudaMemcpy(h_strat, d_strategy, state_sz, cudaMemcpyDeviceToHost));

    for (int a = 0; a < root_na; a++)
        for (int h = 0; h < root_nh; h++)
            output->root_strategy[a * root_nh + h] =
                h_strat[0 * FS_MAX_ACTIONS * max_h + a * max_h + h];

    /* Cleanup */
    free(h_strat);
    free(h_hands);
    free(h_weights);
    free(sd_local);
    free(h_terminal_nodes);
    for (int d = 0; d <= td->max_depth; d++) {
        if (d_level_nodes[d]) cudaFree(d_level_nodes[d]);
        if (level_nodes[d]) free(level_nodes[d]);
    }
    free(d_level_nodes); free(level_nodes); free(level_counts); free(level_caps);
    cudaFree(d_sd_local); cudaFree(d_dec_nodes); cudaFree(d_terminal_nodes);
    cudaFree(d_opp_weights[0]); cudaFree(d_opp_weights[1]); cudaFree(d_iter);
    cudaFree(d_fold_nodes); cudaFree(d_sd_nodes_list);
    if (d_sorted_opp_idx) cudaFree(d_sorted_opp_idx);
    cudaFree(d_nodes); cudaFree(d_children); cudaFree(d_hands);
    cudaFree(d_weights); cudaFree(d_regrets); cudaFree(d_strategy);
    cudaFree(d_strategy_sum); cudaFree(d_cfv); cudaFree(d_reach);
    cudaFree(d_showdown_idx); cudaFree(d_strengths);

    return 0;
}

extern "C" FS_EXPORT void fs_free_tree(FSTreeData *td) {
    if (td->nodes) free(td->nodes);
    if (td->children) free(td->children);
    if (td->level_order) free(td->level_order);
    if (td->node_depth) free(td->node_depth);
    if (td->decision_node_indices) free(td->decision_node_indices);
    if (td->showdown_node_indices) free(td->showdown_node_indices);
    memset(td, 0, sizeof(FSTreeData));
}

extern "C" FS_EXPORT void fs_free_output(FSOutput *out) {
    if (out->root_strategy) free(out->root_strategy);
    if (out->root_ev) free(out->root_ev);
    memset(out, 0, sizeof(FSOutput));
}
