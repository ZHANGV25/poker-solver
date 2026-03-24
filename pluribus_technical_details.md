# Pluribus Technical Details - Complete Extraction

Source: "Superhuman AI for multiplayer poker" (Brown & Sandholm, Science 2019)
- Main paper: DOI 10.1126/science.aay2400
- Supplementary materials: 30-page PDF

---

## 1. BLUEPRINT COMPUTATION

### Algorithm: External-Sampling Monte Carlo CFR (MCCFR)
- Variant: External-sampling MCCFR with two key improvements (Linear CFR + negative-regret pruning)
- Full pseudocode: Algorithm 1 in supplementary (pages 17-19), three functions:
  - `MCCFR-P(T)` - main loop
  - `TRAVERSE-MCCFR(h, Pi)` - standard traversal
  - `TRAVERSE-MCCFR-P(h, Pi)` - traversal with pruning
  - `UPDATE-STRATEGY(h, Pi)` - average strategy update (first betting round only)
  - `CALCULATE-STRATEGY(R(Ii), Ii)` - regret matching

### Training Time & Hardware
- **Wall clock time: 8 days**
- **Total compute: 12,400 CPU core hours**
- **Hardware: Single 64-core server** (large shared memory node on Bridges supercomputer)
  - Four 16-core Intel Xeon E5-8860 v3 CPUs
- **Memory used: < 512 GB** (node had 3 TB available, less than 0.5 TB needed)
- **Cloud cost estimate: ~$144** at spot instance rates
- **No GPUs used at any point**

### Number of Iterations
- Not given as a raw iteration count. Training ran for 8 days continuously.
- Iterations are measured in time (minutes), not count, for the pruning/discounting thresholds.

### Linear CFR Specifics
- **Discounting applied for the first 400 minutes of training**
- **Discount interval: every 10 minutes**
- **Discount factor: d = (T/10) / (T/10 + 1)** where T = minutes elapsed since start
- Both regrets AND average strategies are multiplied by d every 10 minutes
- **Discounting stops after 400 minutes** (cost of multiplications not worth the benefit)
- Linear CFR speeds up convergence by approximately **3x** vs traditional MCCFR

### Regret-Based Pruning
- **Pruning threshold (C): -300,000,000** (negative 300 million)
- **Pruning starts after: 200 minutes of training**
- **After 200 minutes**: 95% of iterations use pruning, 5% explore all actions
- **Pruning exceptions**: Actions on the final betting round (river) are NOT pruned. Actions leading immediately to terminal nodes are NOT pruned.
- **Pruning decision**: Made once per iteration for all actions (not per-action), reducing RNG calls
- Estimated speedup from pruning: **~2x**

### Regret Floor
- **Floor on regret: -310,000,000** for every action
- This prevents integer overflow and makes it easier to unprune actions that later improve

### Memory Storage Format
- **Regrets stored as 4-byte integers** (not 8-byte doubles)
- Memory for regrets in an action sequence **allocated only when first encountered** (lazy allocation)
  - Exception: first betting round allocated up front (it's small)
  - This reduced memory by **more than 2x** (many action sequences never occur in 6-player)

### Total Action Sequences
- **664,845,654 total action sequences** in blueprint action abstraction
- **413,507,309 actually encountered** during training
- Memory only allocated for encountered sequences

### Average Strategy Storage
- For the **first betting round**: traditional average strategy (phi) maintained in memory
  - Updated every 10,000 iterations (Strategy Interval = 10,000)
  - Only tracked on first betting round (UPDATE-STRATEGY returns if betting_round > 0)
- For **rounds 2-4**: snapshots of current strategy saved to disk every 200 minutes after initial 800 minutes
  - Blueprint = average of these snapshots
  - This reduced memory by **nearly half** vs maintaining average strategy in memory

---

## 2. CARD ABSTRACTION

### Preflop (Round 1)
- **Lossless abstraction: 169 strategically distinct hands**
- (There are 169 strategically unique preflop hands when suits are treated as equivalent)

### Flop (Round 2)
- **200 buckets** (lossy abstraction)
- On average **6,434 infosets per abstract infoset bucket** on the flop
- Bucketed using **k-means clustering on domain-specific features** (ref [26]: Johanson et al. 2013)

### Turn (Round 3)
- **200 buckets** (lossy abstraction)
- Same method as flop

### River (Round 4)
- **200 buckets** (lossy abstraction)
- Same method as flop

### Strategically Unique Information Situations (if lossless were used everywhere)
- Preflop: **169**
- Flop: **1,286,792**
- Turn: **55,190,538**
- River: **2,428,287,420**

### Real-Time Search Abstraction
- **Current round: lossless information abstraction** (no lossy bucketing for the round being searched)
- **Future rounds: 500 buckets per round** (lossy, finer than blueprint's 200)
- Buckets for future rounds determined **separately for each flop** using:
  - Potential-aware abstraction (ref [48])
  - Clustering based on **Earth Mover's Distance** (ref [27]: Ganzfried & Sandholm 2014)
  - Combined with future hand potential (ref [28]: Brown, Ganzfried, Sandholm 2015)

---

## 3. ACTION ABSTRACTION

### Blueprint Action Abstraction
- Between **1 and 14 different raise sizes** per decision point
- All raise sizes expressed as **fractions of pot size**
- Sizes chosen by hand based on what earlier Pluribus versions used with significant positive probability
- **First betting round**: particularly fine-grained (Pluribus usually doesn't search here)
- **Second betting round**: more coarse
- **Third and fourth betting rounds**:
  - First raise in round: at most **3 sizes** (0.5x pot, 1x pot, or all-in)
  - Subsequent raises: at most **2 sizes** (1x pot or all-in)
- Fold and call always included when legal

### Real-Time Search Action Abstraction
- Between **1 and 6 raise sizes** per decision point
- Between **100 and 2,000 total action sequences** in the subgame
- If opponent chooses off-tree action: action is added to subgame model and re-searched from root

---

## 4. REAL-TIME SEARCH

### Algorithm 2: Nested Search (page 26)
```
1: I <- empty          # Initialize current infoset as start of game
2: G_root <- G(I)      # Initialize subgame root as public node of I
3: sigma <- sigma_blueprint   # Initialize strategy as blueprint

OPPONENTTURN(a):
4:  if action a not in abstraction for infoset I:
5:    for each node h in G(I):
6:      AddAction(h, a)     # Add a as legal action
7:    sigma <- Search(G_root)  # Re-search from root
8:  I <- I.a                   # Advance infoset
9:  CheckNewRound()

OURTURN:
10:  a ~ sigma(I)       # Sample action from strategy
11:  frozen(I) = True   # Lock this infoset's strategy
12:  I <- I.a           # Advance infoset
13:  CheckNewRound()

CHECKNEWROUND:
14:  if BettingRound(G(I)) > BettingRound(G_root):
15:    G_root = G(I)       # Update root to new round
16:    sigma <- Search(G_root)  # Re-search from new root
```

### When Blueprint vs Search
- **First betting round**: Usually plays blueprint strategy
  - Exception: if opponent raises more than **$100 off** from any blueprint raise size AND no more than **4 players remaining**, then search is used
  - For slightly off-tree first-round bets: uses **pseudo-harmonic action translation** to map to nearest blueprint size
- **Second, third, and fourth betting rounds**: Real-time search ALWAYS used

### Subgame Rooting
- Root = start of the **current betting round** (not the current decision point)
- Root is a probability distribution over nodes in a public state G
- Probability of node h = pi_sigma(h) / sum(pi_sigma(h') for h' in G)
- When a new round starts, root moves to start of new round
- Uses **unsafe search** (assumes opponents followed Pluribus's computed strategy)
  - Unsafe search is ~4x faster than safe alternatives in 6-player (most hands folded preflop)

### Subgame Depth Limits
- **First round search**: extends to end of the round; leaf nodes at chance nodes at start of round 2
- **Second round with > 2 players**: leaf nodes at chance nodes at start of round 3 OR immediately after the second raise action, whichever is earlier
- **All other cases**: subgame extends to **end of the game** (no depth limit)

### 4 Continuation Strategies at Leaf Nodes
Each player chooses one (or a mixture) of 4 strategies for play beyond leaf nodes:

1. **Unmodified blueprint strategy**
2. **Blueprint biased toward folding**: fold probability multiplied by **5**, then renormalized
3. **Blueprint biased toward calling**: call probability multiplied by **5**, then renormalized
4. **Blueprint biased toward raising**: all raise probabilities multiplied by **5**, then renormalized

- Choice must be identical for all leaf nodes in the same infoset
- The choice is treated as another "action" in the subgame, selected via MCCFR
- **Compression**: continuation strategies compressed by sampling one action per abstract infoset according to probabilities, storing only that action using minimum bits necessary

### Leaf Value Computation
- Terminal node values estimated by **rolling out the remainder of the game** according to the list of continuation strategies that all players chose
- NOT a static evaluation - it's a rollout under the chosen continuation strategies

### CFR Variant During Search
- **Large subgames or early game**: Monte Carlo Linear CFR (same as blueprint)
- **Smaller subgames**: Optimized **vector-based Linear CFR** that samples one set of public board cards per thread (ref [42])
- **Plays the strategy from the FINAL iteration** (not the weighted average)
  - But sigma for belief updates uses the weighted average
  - Final iteration strategy helps avoid poor actions not fully eliminated in average

### Time Budget Per Decision
- **1 to 33 seconds per subgame** depending on situation
- **Average: ~20 seconds per hand** (playing against itself in 6-player)
- Roughly 2x faster than professional humans

### Range/Belief Updates (Bayesian)
- Maintains probability distribution over **1326 possible private card pairs** per player
- Initially each pair has probability **1/1326**
- Updated via **Bayes' rule** based on strategy profile sigma:
  - If search hasn't been conducted yet: sigma = blueprint
  - Otherwise: sigma = output of previously-run search
- When a round ends and new root is established: beliefs are updated based on all actions in the completed round

### Frozen Actions
- When Pluribus acts, its strategy at that infoset is **frozen** for that hand
- Only frozen for its actual hand, not other possible hands
- Opponent strategies are NOT frozen
- When a new round starts, all strategies before the new root are effectively frozen

---

## 5. INFRASTRUCTURE

### Blueprint Size
- **< 512 GB memory** needed during training
- Compressed to fit in **< 128 GB** for live play
- Specific compression method: average strategy snapshots (rounds 2-4) + lazy allocation

### Real-Time Play Hardware
- **Two 14-core Intel Haswell E5-2695 v3 CPUs** (28 cores total)
- **< 128 GB RAM**
- Single shared memory node
- No GPUs

### For Comparison
- AlphaGo (2016): 1920 CPUs + 280 GPUs
- Deep Blue (1997): 480 custom chips
- Libratus (2017): 100 CPUs

---

## 6. GAME PARAMETERS

- **Starting stack: $10,000** per player per hand (100 big blinds)
- **Blinds: $50 / $100** (small blind / big blind)
- **6 players**
- **Minimum raise: $100**
- Subsequent raise must be >= previous raise increment
- Maximum raise = remaining chips

---

## 7. KEY ALGORITHM CONSTANTS SUMMARY

| Parameter | Value |
|-----------|-------|
| Blueprint training time | 8 days |
| Blueprint CPU core hours | 12,400 |
| Blueprint cloud cost | ~$144 |
| Training cores | 64 |
| Training memory | < 512 GB |
| Play cores | 28 |
| Play memory | < 128 GB |
| Linear CFR discount period | First 400 minutes |
| Discount interval | 10 minutes |
| Discount formula | d = (T/10)/(T/10+1) |
| Pruning start | After 200 minutes |
| Pruning probability | 95% of iterations |
| Pruning threshold (C) | -300,000,000 |
| Regret floor | -310,000,000 |
| Regret storage | 4-byte integers |
| Strategy Interval | 10,000 iterations |
| Snapshot interval (post-800min) | Every 200 minutes |
| Preflop buckets | 169 (lossless) |
| Flop/Turn/River buckets (blueprint) | 200 per round |
| Flop/Turn/River buckets (search) | 500 per round |
| Continuation strategies (k) | 4 |
| Continuation bias multiplier | 5x |
| Search raise sizes | 1-6 per decision point |
| Blueprint raise sizes | 1-14 per decision point |
| Search action sequences | 100-2,000 |
| Blueprint action sequences (total) | 664,845,654 |
| Blueprint action sequences (encountered) | 413,507,309 |
| Search time per subgame | 1-33 seconds |
| Average time per hand | ~20 seconds |
| Off-tree threshold (round 1) | > $100 from blueprint size |
| Max players for round-1 search | 4 |
| Private card pairs tracked | 1,326 |
