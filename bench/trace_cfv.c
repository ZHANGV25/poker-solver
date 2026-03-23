/**
 * trace_cfv.c — Trace CPU CFV computation at every node
 *
 * Same 4-hand game. Prints the CFV for each hand at each node
 * after solving, so we can compare with GPU values.
 *
 * Also prints the strategy at every decision node.
 */
#include "../src/solver_v2.h"
#include "../src/hand_eval.h"
#include <stdio.h>
#include <string.h>

/* We need access to the internals for debugging.
 * Rerun one CFR iteration and print everything. */

static const char *oop_names[] = {"AhKh", "QhQc", "JhTh", "6h5h"};
static const char *ip_names[] = {"AcKc", "3c3d", "Tc9c", "8c8d"};

static void print_strategy(const SolverV2 *s, int node_idx) {
    InfoSetV2 *is = &s->info_sets[node_idx];
    if (!is->current_strategy) return;

    int player = s->nodes[node_idx].player;
    int na = is->num_actions;
    int nh = is->num_hands;
    const char **names = (player == 0) ? oop_names : ip_names;

    printf("  Strategy at node %d (player %d, %d actions):\n", node_idx, player, na);
    for (int h = 0; h < nh; h++) {
        printf("    %s: ", names[h]);
        for (int a = 0; a < na; a++) {
            float s_val = is->current_strategy[a * nh + h];
            printf("a%d=%.1f%% ", a, s_val * 100);
        }
        printf("\n");
    }
}

static void compute_cfv_debug(SolverV2 *s, int node_idx, int traverser,
                               float *reach0, float *reach1, float *cfv_out) {
    NodeV2 *node = &s->nodes[node_idx];
    int n_trav = s->num_hands[traverser];
    int opp = 1 - traverser;
    const char **trav_names = (traverser == 0) ? oop_names : ip_names;

    if (node->type == NODE_V2_FOLD) {
        int winner = node->player;
        float *reach_opp = (traverser == 0) ? reach1 : reach0;
        int n_opp = s->num_hands[opp];
        float payoff = (traverser == winner)
            ? (float)node->bets[1 - winner]
            : -(float)node->bets[traverser];

        printf("  [%d] FOLD winner=%d payoff=%.0f\n", node_idx, winner, payoff);
        for (int h = 0; h < n_trav; h++) {
            float opp_sum = 0;
            int c0 = s->hands[traverser][h][0], c1 = s->hands[traverser][h][1];
            for (int o = 0; o < n_opp; o++) {
                if (c0==s->hands[opp][o][0]||c0==s->hands[opp][o][1]||
                    c1==s->hands[opp][o][0]||c1==s->hands[opp][o][1]) continue;
                opp_sum += reach_opp[o];
            }
            cfv_out[h] = opp_sum * payoff;
            printf("    %s: opp_reach=%.3f cfv=%.1f\n", trav_names[h], opp_sum, cfv_out[h]);
        }
        return;
    }

    if (node->type == NODE_V2_SHOWDOWN) {
        float *reach_opp = (traverser == 0) ? reach1 : reach0;
        int n_opp = s->num_hands[opp];
        float win_pay = (float)node->bets[opp];
        float lose_pay = -(float)node->bets[traverser];

        printf("  [%d] SHOWDOWN win_pay=%.0f lose_pay=%.0f\n", node_idx, win_pay, lose_pay);
        for (int h = 0; h < n_trav; h++) {
            int c0 = s->hands[traverser][h][0], c1 = s->hands[traverser][h][1];
            uint32_t hs = s->hand_strengths[traverser][h];
            float total = 0;
            for (int o = 0; o < n_opp; o++) {
                if (c0==s->hands[opp][o][0]||c0==s->hands[opp][o][1]||
                    c1==s->hands[opp][o][0]||c1==s->hands[opp][o][1]) continue;
                uint32_t os = s->hand_strengths[opp][o];
                float w = reach_opp[o];
                if (hs > os) total += w * win_pay;
                else if (hs < os) total += w * lose_pay;
            }
            cfv_out[h] = total;
            printf("    %s: cfv=%.1f\n", trav_names[h], cfv_out[h]);
        }
        return;
    }

    int acting = node->player;
    int na = node->num_actions;
    InfoSetV2 *is = &s->info_sets[node_idx];
    int nh_acting = s->num_hands[acting];

    printf("  [%d] DECISION player=%d actions=%d\n", node_idx, acting, na);

    /* Print current strategy */
    if (is->current_strategy) {
        const char **act_names = (acting == 0) ? oop_names : ip_names;
        for (int h = 0; h < nh_acting; h++) {
            printf("    strategy %s: ", act_names[h]);
            for (int a = 0; a < na; a++)
                printf("%.1f%% ", is->current_strategy[a * nh_acting + h] * 100);
            printf("\n");
        }
    }

    /* Print opponent reach at this node */
    float *reach_opp = (traverser == 0) ? reach1 : reach0;
    printf("    opp reach: ");
    const char **opp_names_arr = (opp == 0) ? oop_names : ip_names;
    for (int o = 0; o < s->num_hands[opp]; o++)
        printf("%s=%.3f ", opp_names_arr[o], reach_opp[o]);
    printf("\n");

    float reach0_mod[MAX_HANDS_V2], reach1_mod[MAX_HANDS_V2];

    if (acting == traverser) {
        float action_cfv[MAX_ACTIONS_V2 * MAX_HANDS_V2];
        memset(cfv_out, 0, n_trav * sizeof(float));

        for (int a = 0; a < na; a++) {
            memcpy(reach0_mod, reach0, s->num_hands[0] * sizeof(float));
            memcpy(reach1_mod, reach1, s->num_hands[1] * sizeof(float));
            float *my_reach = (traverser == 0) ? reach0_mod : reach1_mod;
            for (int h = 0; h < n_trav; h++)
                my_reach[h] *= is->current_strategy[a * nh_acting + h];

            printf("  --- Action %d (child %d) ---\n", a, node->children[a]);
            float *child_cfv = action_cfv + a * n_trav;
            compute_cfv_debug(s, node->children[a], traverser,
                              reach0_mod, reach1_mod, child_cfv);

            for (int h = 0; h < n_trav; h++)
                cfv_out[h] += is->current_strategy[a * nh_acting + h] * child_cfv[h];
        }

        printf("  Node %d CFV:\n", node_idx);
        for (int h = 0; h < n_trav; h++)
            printf("    %s: node_cfv=%.1f\n", trav_names[h], cfv_out[h]);

    } else {
        float child_cfv[MAX_HANDS_V2];
        memset(cfv_out, 0, n_trav * sizeof(float));

        for (int a = 0; a < na; a++) {
            memcpy(reach0_mod, reach0, s->num_hands[0] * sizeof(float));
            memcpy(reach1_mod, reach1, s->num_hands[1] * sizeof(float));
            float *opp_r = (acting == 0) ? reach0_mod : reach1_mod;
            for (int h = 0; h < nh_acting; h++)
                opp_r[h] *= is->current_strategy[a * nh_acting + h];

            printf("  --- Action %d (child %d) ---\n", a, node->children[a]);
            compute_cfv_debug(s, node->children[a], traverser,
                              reach0_mod, reach1_mod, child_cfv);

            for (int h = 0; h < n_trav; h++)
                cfv_out[h] += child_cfv[h];
        }

        printf("  Node %d CFV:\n", node_idx);
        for (int h = 0; h < n_trav; h++)
            printf("    %s: node_cfv=%.1f\n", trav_names[h], cfv_out[h]);
    }
}

int main(void) {
    printf("=== CPU CFV Trace (4-hand game) ===\n\n");

    int board[5] = {parse_card("Qs"), parse_card("As"), parse_card("2d"),
                    parse_card("7h"), parse_card("4c")};

    int h0[4][2], h1[4][2];
    float w0[4] = {1,1,1,1}, w1[4] = {1,1,1,1};

    h0[0][0]=parse_card("Ah"); h0[0][1]=parse_card("Kh");
    h0[1][0]=parse_card("Qh"); h0[1][1]=parse_card("Qc");
    h0[2][0]=parse_card("Jh"); h0[2][1]=parse_card("Th");
    h0[3][0]=parse_card("6h"); h0[3][1]=parse_card("5h");
    h1[0][0]=parse_card("Ac"); h1[0][1]=parse_card("Kc");
    h1[1][0]=parse_card("3c"); h1[1][1]=parse_card("3d");
    h1[2][0]=parse_card("Tc"); h1[2][1]=parse_card("9c");
    h1[3][0]=parse_card("8c"); h1[3][1]=parse_card("8d");

    float bs[] = {0.75f};
    SolverV2 s;
    sv2_init(&s, board, 5, (const int(*)[2])h0, w0, 4,
             (const int(*)[2])h1, w1, 4, 1000, 5000, bs, 1);

    /* Solve first */
    printf("Solving 1000 iterations...\n\n");
    sv2_solve(&s, 1000, 0.0f);

    /* Now trace one final traversal for OOP (traverser=0) */
    printf("=== Tracing CFV for OOP (traverser=0) ===\n\n");
    float reach0[4] = {1,1,1,1}, reach1[4] = {1,1,1,1};
    float cfv[4];
    compute_cfv_debug(&s, 0, 0, reach0, reach1, cfv);

    printf("\n=== Final Root CFV for OOP ===\n");
    for (int h = 0; h < 4; h++)
        printf("  %s: %.1f\n", oop_names[h], cfv[h]);

    printf("\n=== Final Strategies ===\n");
    for (int h = 0; h < 4; h++) {
        float strat[MAX_ACTIONS_V2];
        sv2_get_strategy(&s, 0, h, strat);
        float bet = 0;
        for (int a = 1; a < s.nodes[0].num_actions; a++) bet += strat[a];
        printf("  OOP %s: check=%.1f%% bet=%.1f%%\n", oop_names[h], strat[0]*100, bet*100);
    }

    sv2_free(&s);
    return 0;
}
