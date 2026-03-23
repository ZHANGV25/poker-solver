# Known Gaps and Their Impact

## Gap 1: Preflop Ranges Are Semi-Binary

**What we have:** `ranges.json` with 42 range entries. 18 are pure binary (hand is in or out). 24 have partial weights, but ONLY at 0.5 (50% frequency). No hands have 0.3, 0.7, or other intermediate frequencies.

**What GTO actually is:** Every hand has a precise frequency. For example, CO's RFI range might include K9o at 37% frequency, not "in" or "out." The boundary hands (hands on the edge of the range) should be played at mixed frequencies.

**Impact:**
- The ~20 hands at the boundary of each range are either always played (when they should sometimes fold) or never played (when they should sometimes open).
- Overplays ~10% of the range and underplays ~10%.
- For most hands deep inside the range (AA, KK, AKs) this makes zero difference.
- For boundary hands (K7o, Q9o, etc.) the error is up to 100% (always play vs never play).

**Effort to fix:** MEDIUM. Need a proper preflop solver (or scrape GTO Wizard's preflop charts) to get exact frequencies for all 169 hand classes × 6 positions × all scenarios. Could use our own CFR solver on a simplified preflop game tree, or just look up published GTO preflop ranges.

**Workaround:** The 0.5 weights we do have cover the most important boundary hands. The impact on postflop play is modest because postflop strategies adjust to whatever range enters.

---

## Gap 2: Flop Blueprint Without Range Narrowing

**What we have:** Precomputed per-hand strategies for the flop, solved with FULL preflop ranges. When villain acts on the flop, we narrow their range. But the strategy hero uses on the FLOP itself was computed against villain's full range.

**What Pluribus does:** Re-solves the flop with 4 continuation strategies even before anyone has acted, upgrading from the coarse blueprint to a fine-grained solution.

**Why our approach is actually okay for the first action:**
- Before anyone acts on the flop, both players have their full preflop ranges.
- Our blueprint was solved with exactly those full ranges.
- The blueprint IS the correct answer for the first flop action.
- Pluribus re-solves because its blueprint used lossy card abstraction (200 buckets). Ours uses exact per-hand strategies with no abstraction.

**Where it's NOT okay:**
- When villain bets and hero faces a decision, villain's range is narrowed.
- Our blueprint's response was computed against villain's FULL range.
- If villain's actual range (after betting) is much different from the full range, our response is suboptimal.

**Impact:** LOW to MEDIUM. The first flop action is correct. Subsequent flop actions (facing a bet/raise) use strategies computed against the wrong range. The error scales with how much villain's range changed — on most flops, the betting range is ~60-70% of the full range, so the error is modest.

**Effort to fix:** MEDIUM. Re-solve the flop from the current decision point when facing an action, using narrowed ranges. This uses the same re-solve infrastructure as the turn/river, but with 47 turn cards × leaf evaluation making it ~268ms (measured). Feasible but adds complexity.

---

## Gap 3: Flop Blueprint Coverage

**What we have:**
- 4 of 27 scenarios have full per-hand strategies (CO_vs_BB_srp, MP_vs_BB_srp, UTG_vs_BB_srp, UTG_vs_SB_3bp)
- 2 scenarios are partially upgraded (BTN_vs_BB_srp 53%, UTG_vs_MP_srp 46%)
- 21 scenarios have aggregate-only strategies (useless for per-hand advice)

**Impact:** HIGH for the 21 missing scenarios. When we encounter a BTN_vs_BB_3bp pot, we have no per-hand flop strategy — we can only give aggregate advice ("range bets 50%") which doesn't help with specific hands.

**Effort to fix:** LOW (with GPU solver). Run local precompute with the Rust solver (~6 minutes on GPU for all 47K textures). This is blocked on the fact that the Rust solver does the full flop-through-river solve, not our C solver. Need to run `precompute/solve_scenarios.py` locally.

---

## Gap 4: Turn Blueprint for Continuation Strategies

**What we have:** Nothing. No precomputed turn strategies.

**What we need:** Per-hand action frequencies at the turn root node for each of 47 possible turn cards, for each flop texture. This data is used to:
1. Generate the 4 biased continuation strategies at turn leaf nodes
2. Narrow ranges after turn actions

**Impact:** HIGH. Without turn blueprints, our turn re-solve uses equity-based heuristic leaf values instead of proper continuation strategies. This misses implied odds and future street dynamics.

**Effort to fix:** MEDIUM. The Rust solver already computes turn strategies during its flop-through-river solve — we just need to extract and save them. Requires modifying the Rust solver's output format to include turn node strategies per runout.

---

## Gap 5: Multiway Pots

**What we have:** Heads-up solver only. When 3+ players see the flop, we pick hero vs "most relevant" villain and ignore the rest.

**Impact:** MEDIUM. ~25% of flop pots are multiway in 6-max. Our strategies will overbluff and underdefend in multiway spots.

**Effort to fix:** LARGE. Requires a multiplayer CFR solver (3+ players). Tree size explodes, convergence is slower, and Nash equilibrium isn't well-defined for >2 players. This is a research-level problem.

**Workaround:** Heuristic adjustments for multiway (reduce bluff frequency by 50%, tighten calling range). This captures ~80% of the multiway effect.

---

## Gap 6: Exploitability Computation Needs Fixing for Larger Ranges

**What we have:** Exploitability works for small ranges (4 hands: converges to <1%). For 100 hands, shows 15.4% after 1000 iterations — unclear if this is correct or a bug.

**Impact:** MEDIUM. Without reliable exploitability for large ranges, we can't measure convergence quality for production use.

**Effort to fix:** LOW. The best-response function likely needs the same careful payoff treatment as the CFR traversal. Debug by comparing best-response values at specific nodes.

---

## Priority Order

1. **Gap 3** (Flop blueprint coverage) — run local precompute, LOW effort, HIGH impact
2. **Gap 4** (Turn blueprints) — modify Rust solver output, MEDIUM effort, HIGH impact
3. **Gap 1** (Preflop frequencies) — get proper GTO ranges, MEDIUM effort, MEDIUM impact
4. **Gap 6** (Exploitability for large ranges) — debug best-response, LOW effort, MEDIUM impact
5. **Gap 2** (Flop re-solve when facing action) — wire existing infrastructure, MEDIUM effort, LOW-MEDIUM impact
6. **Gap 5** (Multiway) — research problem, LARGE effort, MEDIUM impact
