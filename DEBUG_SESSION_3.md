# Debug Session 3: Deep Research — Why Does Pluribus Converge and We Don't?

Paste this into a new Claude Code conversation. Working directory: `/Users/victor/Documents/Dev/poker-solver/`

Private repo: `ZHANGV25/poker-solver-dev`

---

## Context

We're building a 6-player no-limit hold'em blueprint solver matching Pluribus (Brown & Sandholm, Science 2019). External-sampling MCCFR with Linear CFR discount and regret-based pruning on EC2 (c7a.metal-48xl, 192 cores, 376 GB RAM).

**We have spent an entire day debugging convergence issues.** The solver runs, late positions (BTN, SB) converge perfectly, but early positions (UTG, MP) show pathological behavior where hands that should raise instead call or fold. Multiple fix attempts have failed. We need deep research into WHY.

---

## What Works

- BTN: All 15 pocket pairs raise correctly at 500M iterations (85-100% raise)
- SB: Converges well (1-3 opponents)
- Premiums from UTG eventually converge with enough iterations (AA reached 87% raise at 1.5B in one run)
- All code bugs found and fixed (board_hash, int32 overflow, heap corruption, int64, tiered sizing)
- Position gradient is correct (UTG tightest → BTN widest)

## What's Broken

UTG (5 opponents) and MP (4-5 opponents) show three failure modes for pocket pairs:

1. **Call trap**: TT/99 converge to 100% call instead of raise. Call regret freezes when call dominates (same mechanism as fold lock-in). Monotonically worsening: TT went 63%R→41%R→34%R→12%R across checkpoints.

2. **Fold lock-in**: 44/33/22 locked at 100% fold across all checkpoints. Fold regret frozen at positive value (fold_value=0 for UTG who has no blinds invested).

3. **Uniform stuck**: 88/AKo show 9.1/9.1/81.8 (all regrets ≤ 0, uniform default) after 2B+ iterations.

## Fix Attempts (All Failed)

### Attempt 1: Exempt preflop from pruning
- **Theory**: Pruned raise actions (below -300M) can't recover
- **Result**: Broke fold lock-in (99/TT started raising) but revealed call trap underneath. TT declined from 63% raise to 12% raise over 2B iterations. Strategy_sum average also showed call preference — confirmed structural.
- **Reverted**: Pluribus doesn't exempt preflop from pruning

### Attempt 2: Average Strategy Sampling (Lanctot et al. 2012)
- **Theory**: Non-traversers sample from average strategy instead of current, giving "UTG raised" nodes more training data
- **Result**: Made things WORSE. AA showed zero signal (uniform) at 500M. Bootstrap problem: early average strategy is uniform → opponents sample uniformly → no EV differentiation → stays uniform.
- **Reverted**: Pluribus uses standard ES

### Attempt 3: Remove 10x regret scaling
- **Theory**: The `* 10.0f` in regret delta wasn't in Pluribus, amplifies noise, causes premature ceiling/floor saturation
- **Result**: Combined with AS (Attempt 2), so can't isolate. Currently testing WITHOUT AS.
- **Status**: Still in code (removed), currently running

### Current run (Attempt 4): Pure Pluribus-matching
- No 10x, no AS, standard pruning, fixed bucketing, strategy_sum every 10007 iters
- **50M probe**: All pocket pairs raising (best early result ever)
- **100M probe**: AA:C85 KK:C65 — AA/KK flipped to call while JJ/TT still raise (INVERTED)
- **Status**: Running, waiting for 200M+ probes (discount ends at 175M)

## The Core Mechanism

When one action dominates the strategy (say call at 80%):
1. `node_value ≈ call_value` (weighted by 80% strategy)
2. `regret[call] += call_value - call_value ≈ 0` → **call regret freezes**
3. `regret[raise_i] += raise_i_value - call_value` → depends on raise quality
4. With 8 raise sizes, each gets ~1% of non-traverser sampling → opponents at "UTG raised size X" get 80x less training data than "UTG called" → raise values are unreliable
5. Unreliable raise values → raise regrets don't grow → call stays dominant

This doesn't affect BTN because fewer opponents = less noise in raise values = signal breaks through faster.

## Pluribus Alignment Audit

Every algorithmic parameter now matches Pluribus EXCEPT:

| Parameter | Pluribus | Us | Risk |
|-----------|----------|-----|------|
| Open raise sizes | Hand-selected 1-14 | 8 fixed [0.4,0.5,0.7,1.0,1.5,2.5,4.0,8.0] | **Unknown** |
| Total compute | 12,400 core-hours | ~2,000 max | Known gap |

The raise sizes are the last remaining deviation we haven't tested. Pluribus says sizes were "chosen by hand based on what earlier Pluribus versions used with significant positive probability." Web sources show Pluribus opens to ~2.0-2.25 BB typically.

## What to Investigate

### 1. Is the 8-size fragmentation really the problem?
Pluribus used "up to 14" sizes for preflop and it worked. More sizes than us. So fragmentation alone doesn't explain it. What's different about their 14 vs our 8?

### 2. Is there a subtle algorithmic deviation we're missing?
Read the Pluribus supplementary materials (Algorithm 1 pseudocode) extremely carefully. Our `pluribus_technical_details.md` has a summary but may miss nuances. The supplementary PDF is at: https://noambrown.github.io/papers/19-Science-Superhuman_Supp.pdf

Specific things to check:
- Does UPDATE-STRATEGY use the current strategy σ or something else?
- Is there exploration in CALCULATE-STRATEGY beyond standard regret matching?
- How exactly does the pruning decision work? (per-iteration for all actions, or per-action?)
- Does Pluribus use any warm-start or initial regret seeding?
- Is there any mention of convergence issues with many actions?

### 3. The regret ceiling (+310M) — is this causing information loss?
Pluribus mentions a regret FLOOR (-310M) but not a ceiling. We added +310M to prevent overflow. Could this be causing problems? If dominant actions hit +310M ceiling, they all look equal (same as the uniform problem).

### 4. Could the discount phase be the issue?
Our discount is proportionally scaled. Pluribus uses wall-clock time. The cumulative discount effect is: product(T/(T+1) for T=1..40) = 1/41 ≈ 0.024. This means 97.6% of early regret signal is erased. With our iteration-based scaling, is this ratio the same?

### 5. Noam Brown's thesis
His 2020 CMU thesis "Equilibrium Finding for Large Adversarial Imperfect-Information Games" may discuss convergence issues and solutions in more detail than the Science paper. Available at: http://reports-archive.adm.cs.cmu.edu/anon/2020/CMU-CS-20-132.pdf

### 6. The open-source Pluribus implementations
Several exist on GitHub. Do any of them have the same convergence issues? How do they handle raise sizes?
- https://github.com/apcode/pluribus-poker-AI
- https://github.com/Agnar22/Pluribus

## Key Files

- `src/mccfr_blueprint.c` — solver (traverse at ~line 712, pruning at ~1054, regret update at ~1099, non-traverser at ~1124, strategy_sum at ~1110)
- `src/mccfr_blueprint.h` — data structures, constants
- `src/card_abstraction.c` — bucketing (ca_compute_features, ca_assign_buckets_kmeans)
- `pluribus_technical_details.md` — our extraction of the Pluribus paper
- `docs/BLUEPRINT_BUGS.md` — all bugs found and fixed (8 total)
- `docs/BLUEPRINT_CHRONICLE.md` — full narrative timeline of what was tried
- `DEBUG_SESSION_2.md` — original debug session that started this investigation

## EC2 Instance Running

Instance `i-0de1597e03994a548` (c7a.metal-48xl) is running with the Pluribus-exact config. Probes every 50M iterations uploaded to `s3://poker-blueprint-unified/probes/`.

Check latest: `aws s3 cp s3://poker-blueprint-unified/probes/probe_latest.txt -`

## What We Need

1. Deep research into the Pluribus algorithm — read the supplementary, the thesis, the open-source implementations
2. Identify ANY remaining deviation from Pluribus, no matter how small
3. Understand whether 8 raise sizes should cause convergence failure when Pluribus uses 14
4. If possible, find evidence of others encountering the same convergence issue
5. Propose a fix that is GROUNDED IN EVIDENCE, not theoretical — we've had enough theoretical fixes that failed

Do NOT implement anything. Research only. Read code, read papers, compare. Report findings.
