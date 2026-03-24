/* Minimal test: just build the tree on CPU and report sizes */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Inline the tree builder from flop_solve.cu (C-compatible subset) */
#define FS_MAX_HANDS 200
#define FS_MAX_ACTIONS 6
#define FS_NODE_DECISION 0
#define FS_NODE_FOLD 1
#define FS_NODE_SHOWDOWN 2
#define FS_NODE_CHANCE 3
#define MAX_RAISES 1

typedef struct {
    int type, player, num_children, first_child, pot;
    int bets[2], board_cards[5], num_board;
} FSNode;

typedef struct {
    FSNode *nodes; int num_nodes, cap_nodes;
    int *children; int num_children, cap_children;
} TB;

static void tb_init(TB *t) {
    t->cap_nodes=8192; t->nodes=malloc(t->cap_nodes*sizeof(FSNode)); t->num_nodes=0;
    t->cap_children=32768; t->children=malloc(t->cap_children*sizeof(int)); t->num_children=0;
}
static int tb_an(TB *t) {
    if(t->num_nodes>=t->cap_nodes){t->cap_nodes*=2;t->nodes=realloc(t->nodes,t->cap_nodes*sizeof(FSNode));}
    int i=t->num_nodes++; memset(&t->nodes[i],0,sizeof(FSNode)); t->nodes[i].player=-1; return i;
}
static int tb_ac(TB *t, int n) {
    while(t->num_children+n>t->cap_children){t->cap_children*=2;t->children=realloc(t->children,t->cap_children*sizeof(int));}
    int s=t->num_children; t->num_children+=n; return s;
}

static int build_bet(TB *tb, int is_river, int player, int pot, int stack,
                     int b0, int b1, int nr, int at,
                     const float *bs, int nbs, const int *board, int nb) {
    int tc=(player==0)?(b1-b0):(b0-b1); if(tc<0)tc=0;
    if(at>=2&&b0==b1){
        int i=tb_an(tb);
        tb->nodes[i].type=is_river?FS_NODE_SHOWDOWN:FS_NODE_CHANCE;
        tb->nodes[i].pot=pot; tb->nodes[i].bets[0]=b0; tb->nodes[i].bets[1]=b1;
        for(int j=0;j<nb;j++) tb->nodes[i].board_cards[j]=board[j];
        tb->nodes[i].num_board=nb;
        return i;
    }
    int node=tb_an(tb);
    tb->nodes[node].type=FS_NODE_DECISION; tb->nodes[node].player=player;
    tb->nodes[node].pot=pot; tb->nodes[node].bets[0]=b0; tb->nodes[node].bets[1]=b1;
    for(int j=0;j<nb;j++) tb->nodes[node].board_cards[j]=board[j];
    tb->nodes[node].num_board=nb;
    int tc_arr[8]; int nc=0;
    if(tc>0){int f=tb_an(tb);tb->nodes[f].type=FS_NODE_FOLD;tb->nodes[f].player=1-player;
        tb->nodes[f].pot=pot;tb->nodes[f].bets[0]=b0;tb->nodes[f].bets[1]=b1;
        for(int j=0;j<nb;j++)tb->nodes[f].board_cards[j]=board[j];tb->nodes[f].num_board=nb;
        tc_arr[nc++]=f;}
    if(tc==0){
        tc_arr[nc++]=build_bet(tb,is_river,1-player,pot,stack,b0,b1,nr,at+1,bs,nbs,board,nb);
    } else {
        int nb0=b0,nb1=b1; if(player==0)nb0=b1;else nb1=b0;
        int cp=pot+tc;
        if(at>=1){int t2=tb_an(tb);tb->nodes[t2].type=is_river?FS_NODE_SHOWDOWN:FS_NODE_CHANCE;
            tb->nodes[t2].pot=cp;tb->nodes[t2].bets[0]=nb0;tb->nodes[t2].bets[1]=nb1;
            for(int j=0;j<nb;j++)tb->nodes[t2].board_cards[j]=board[j];tb->nodes[t2].num_board=nb;
            tc_arr[nc++]=t2;}
        else tc_arr[nc++]=build_bet(tb,is_river,1-player,cp,stack-tc,nb0,nb1,nr,at+1,bs,nbs,board,nb);
    }
    if(nr<MAX_RAISES){
        for(int i=0;i<nbs&&nc<7;i++){
            int ba;
            if(tc==0)ba=(int)(bs[i]*pot); else ba=tc+(int)(bs[i]*(pot+tc));
            if(ba>=stack)ba=stack; if(ba<=tc)continue;
            int nb0=b0,nb1=b1; if(player==0)nb0+=ba;else nb1+=ba;
            int np=pot+ba,ns=stack-ba+tc;
            if(ba>=stack){
                int ai=tb_an(tb);tb->nodes[ai].type=FS_NODE_DECISION;tb->nodes[ai].player=1-player;
                tb->nodes[ai].pot=np;tb->nodes[ai].bets[0]=nb0;tb->nodes[ai].bets[1]=nb1;
                for(int j=0;j<nb;j++)tb->nodes[ai].board_cards[j]=board[j];tb->nodes[ai].num_board=nb;
                int f=tb_an(tb);tb->nodes[f].type=FS_NODE_FOLD;tb->nodes[f].player=player;
                tb->nodes[f].pot=np;for(int j=0;j<nb;j++)tb->nodes[f].board_cards[j]=board[j];tb->nodes[f].num_board=nb;
                int cb0=nb0,cb1=nb1;if(player==0)cb1=nb0;else cb0=nb1;int fp=np+(ba-tc);
                int sd=tb_an(tb);tb->nodes[sd].type=is_river?FS_NODE_SHOWDOWN:FS_NODE_CHANCE;
                tb->nodes[sd].pot=fp;for(int j=0;j<nb;j++)tb->nodes[sd].board_cards[j]=board[j];tb->nodes[sd].num_board=nb;
                int as=tb_ac(tb,2);tb->children[as]=f;tb->children[as+1]=sd;
                tb->nodes[ai].first_child=as;tb->nodes[ai].num_children=2;
                tc_arr[nc++]=ai;
            } else {
                tc_arr[nc++]=build_bet(tb,is_river,1-player,np,ns,nb0,nb1,nr+1,at+1,bs,nbs,board,nb);
            }
        }
    }
    int s=tb_ac(tb,nc);for(int i=0;i<nc;i++)tb->children[s+i]=tc_arr[i];
    tb->nodes[node].first_child=s;tb->nodes[node].num_children=nc;
    return node;
}

static void expand_chance(TB *tb, const float *bs, int nbs) {
    int orig=tb->num_nodes;
    /* First pass: collect all chance node indices and their data
     * (before any realloc can invalidate pointers) */
    int *chance_list = malloc(orig * sizeof(int));
    int num_chance = 0;
    for(int i=0;i<orig;i++){
        if(tb->nodes[i].type!=FS_NODE_CHANCE && tb->nodes[i].num_children==0)
            if(tb->nodes[i].type==FS_NODE_CHANCE) chance_list[num_chance++]=i;
        if(tb->nodes[i].type==FS_NODE_CHANCE && tb->nodes[i].num_board<5 && tb->nodes[i].num_children==0)
            chance_list[num_chance++]=i;
    }
    /* Snapshot the data we need from each chance node */
    typedef struct { int idx; int nb; int board[5]; int pot; int bet0; } ChanceInfo;
    ChanceInfo *infos = malloc(num_chance * sizeof(ChanceInfo));
    for(int j=0;j<num_chance;j++){
        int i=chance_list[j];
        infos[j].idx=i;
        infos[j].nb=tb->nodes[i].num_board;
        for(int b=0;b<5;b++) infos[j].board[b]=tb->nodes[i].board_cards[b];
        infos[j].pot=tb->nodes[i].pot;
        infos[j].bet0=tb->nodes[i].bets[0];
    }
    /* Second pass: expand each chance node using cached data */
    for(int j=0;j<num_chance;j++){
        int i=infos[j].idx;
        int nb=infos[j].nb;
        int blocked[52]={0};
        for(int b=0;b<nb;b++) blocked[infos[j].board[b]]=1;
        int nc_cards[49]; int nnc=0;
        for(int c=0;c<52;c++) if(!blocked[c]) nc_cards[nnc++]=c;
        int is_riv=(nb+1==5);
        int s=tb_ac(tb,nnc);
        for(int ci=0;ci<nnc;ci++){
            int new_b[5];for(int b=0;b<nb;b++)new_b[b]=infos[j].board[b];
            new_b[nb]=nc_cards[ci];
            int stk=10000-infos[j].bet0;if(stk<0)stk=0;
            int subtree=build_bet(tb,is_riv,0,infos[j].pot,stk,
                                   0,0,0,0,bs,nbs,new_b,nb+1);
            /* Re-derive children pointer (realloc may have moved it) */
            tb->children[s+ci]=subtree;
        }
        /* Safe: nodes[i] index is still valid (we only append, never delete) */
        tb->nodes[i].first_child=s;
        tb->nodes[i].num_children=nnc;
    }
    free(chance_list);
    free(infos);
    printf("  expanded %d chance nodes\n", num_chance); fflush(stdout);
}

int main(void) {
    printf("Tree build test\n"); fflush(stdout);
    TB tb; tb_init(&tb);
    int board[3]={40,48,0}; /* Qs As 2c — just card ints */
    float bs[]={0.75f};
    build_bet(&tb,0,0,1000,5000,0,0,0,0,bs,1,board,3);
    printf("Flop: %d nodes\n",tb.num_nodes); fflush(stdout);

    float turn_bs[]={0.75f};
    expand_chance(&tb,turn_bs,1);
    printf("After turn: %d nodes, %d children\n",tb.num_nodes,tb.num_children); fflush(stdout);

    float river_bs[]={0.75f};
    expand_chance(&tb,river_bs,1);
    printf("After river: %d nodes, %d children\n",tb.num_nodes,tb.num_children); fflush(stdout);

    int dec=0,sd=0,ch=0,fold=0;
    for(int i=0;i<tb.num_nodes;i++){
        if(tb.nodes[i].type==FS_NODE_DECISION)dec++;
        if(tb.nodes[i].type==FS_NODE_SHOWDOWN)sd++;
        if(tb.nodes[i].type==FS_NODE_CHANCE)ch++;
        if(tb.nodes[i].type==FS_NODE_FOLD)fold++;
    }
    printf("Types: %d decision, %d fold, %d showdown, %d chance\n",dec,fold,sd,ch);
    printf("Memory: %.1f MB nodes, %.1f MB children\n",
           tb.num_nodes*sizeof(FSNode)/1e6, tb.num_children*4/1e6);

    free(tb.nodes);free(tb.children);
    printf("DONE\n");
    return 0;
}
