# References and Sources

## Academic Papers

### Pluribus (Primary Reference)
- **Paper:** "Superhuman AI for multiplayer poker" — Brown & Sandholm, Science 365(6456), 2019
- **DOI:** 10.1126/science.aay2400
- **Main PDF:** https://noambrown.github.io/papers/19-Science-Superhuman.pdf
- **Supplementary (30 pages, contains Algorithm 1 & 2 pseudocode):** https://noambrown.github.io/papers/19-Science-Superhuman_Supp.pdf
- **Key details we use:**
  - Algorithm 2: real-time search procedure (subgame rooting at round start)
  - 4 continuation strategies at leaf nodes (unmodified, fold×5, call×5, raise×5)
  - Linear CFR = DCFR(α=1, β=1, γ=1) for blueprint, final iteration strategy during play
  - Bayesian range updates using most recent search strategy
  - Unsafe search (no gadget game), mitigated by round-rooting

### Libratus
- **Paper:** "Superhuman AI for heads-up no-limit poker: Libratus beats top professionals" — Brown & Sandholm, Science 2018
- **DOI:** 10.1126/science.aao1733
- **PDF:** https://noambrown.github.io/papers/17-Science-Superhuman.pdf
- **Key details:** Safe subgame solving with gadget game, nested re-solving, blueprint + real-time refinement

### DCFR (Our Algorithm)
- **Paper:** "Solving Imperfect-Information Games via Discounted Regret Minimization" — Brown & Sandholm, AAAI 2019
- **arXiv:** https://arxiv.org/pdf/1809.04040
- **Key details:** Recommended params α=1.5, β=0, γ=2. We use CFR+ (regret floor at 0) which is a special case.

### Depth-Limited Solving
- **Paper:** "Depth-Limited Solving for Imperfect-Information Games" — Brown, Sandholm, Amos, NeurIPS 2018
- **arXiv:** https://arxiv.org/abs/1805.08195
- **PDF:** https://proceedings.neurips.cc/paper_files/paper/2018/file/34306d99c63613fad5b2a140398c0420-Paper.pdf
- **Key details:** Multi-valued leaf states, 4 continuation strategies, bias approach for generating strategies

### Safe Subgame Solving
- **Paper:** "Safe and Nested Subgame Solving for Imperfect-Information Games" — Brown & Sandholm, NeurIPS 2017
- **arXiv:** https://arxiv.org/abs/1705.02955
- **Key details:** Gadget game construction, alternative payoffs, safety guarantees

### GPU-Accelerated CFR
- **Paper:** "GPU-Accelerated Counterfactual Regret Minimization" — University of Toronto CPRG, 2024
- **arXiv:** https://arxiv.org/html/2408.14778v1
- **Code:** https://github.com/uoftcprg/gpugt
- **Key details:** CFR reformulated as batched matrix-vector operations per tree level, 203x speedup over C++ on toy games. We use the batch-across-textures approach (many small independent solves) rather than their within-game batching.

### Supremus
- **Paper:** "Unlocking the Potential of Deep Counterfactual Value Networks" — 2020
- **arXiv:** https://arxiv.org/abs/2007.10442
- **Key details:** Full CUDA poker AI, 1000 DCFR iterations in 0.8s on GPU. Uses neural network value functions at leaves.

### CFR+
- **Paper:** "Solving Large Imperfect Information Games Using CFR+" — Tammelin et al., IJCAI 2015
- **arXiv:** https://arxiv.org/abs/1407.5042
- **Key details:** Floor negative regrets at 0, alternating updates. We use this variant.

### ReBeL
- **Paper:** "Combining Deep Reinforcement Learning and Search for Imperfect-Information Games" — Brown et al., 2020
- **arXiv:** https://arxiv.org/abs/2007.13544
- **Key details:** Public Belief States, learned value networks for depth-limited search. Future direction.

### Regret-Based Pruning
- **Paper:** "Regret-Based Pruning in Extensive-Form Games" — Brown & Sandholm, NeurIPS 2015/2016
- **arXiv:** https://arxiv.org/abs/1609.03234
- **Key details:** Skip actions with regret < threshold in 95% of iterations. Pluribus uses threshold ≈ -300M.

## Open Source Implementations

### Noam Brown's Reference Solver
- **Repo:** https://github.com/noambrown/poker_solver
- **What it is:** River-only subgame solver with CFR/CFR+/DCFR/MCCFR
- **Key code:** `Trainer` class (full traversal), `MCCFRTrainer` (sampled), O(N+M) showdown via prefix sums
- **We referenced:** DCFR parameters (α=1.5, β=0, γ=2), hand evaluation approach, strategy extraction

### b-inary/postflop-solver (Rust)
- **Repo:** https://github.com/b-inary/postflop-solver
- **What it is:** Production-grade NLHE postflop solver, faster than PioSOLVER
- **Desktop app:** https://github.com/b-inary/desktop-postflop
- **Key details:** DCFR with γ=3 and strategy reset at powers of 4, 16-bit compression, rayon parallelism, suit isomorphism
- **We use:** As the Rust solver binary (`tbl-engine.exe`) for precomputing flop blueprints. Our C solver was cross-validated against it.
- **Benchmarks:** 19.8s vs PioSOLVER's 22.9s on standard test (6 threads, 0.5% exploitability)

### GPUGT
- **Repo:** https://github.com/uoftcprg/gpugt
- **What it is:** Python/CuPy GPU CFR implementation (companion to the paper)
- **License:** MIT
- **We referenced:** Level-batched kernel architecture concept

### Other Reference Solvers
- **TexasSolver:** https://github.com/bupticybee/TexasSolver — Open source C++ NLHE solver
- **Fossana/discounted-cfr:** https://github.com/Fossana/discounted-cfr-poker-solver — Java DCFR for turn/river
- **OMPEval:** https://github.com/zekyll/OMPEval — Fast hand evaluator (272M/s), 200KB tables
- **PokerHandEvaluator:** https://github.com/HenryRLee/PokerHandEvaluator — 144KB lookup tables

## Our Prior Work

### Poker Engine 2026 (CMU Hackathon)
- **Repo:** https://github.com/ZHANGV25/poker-engine-2026
- **What it was:** Custom 27-card poker variant for CMU × Jump Trading tournament
- **Key learnings we applied:**
  - C DCFR solver with 4.8x speedup over Python
  - Hero range tracking is critical (the biggest miss — "a 20-minute fix that could have changed the outcome")
  - Double-narrowing bug (pre-narrowing + solver-internal narrowing overcounts)
  - Blueprint backward-induction outperforms runtime solving for acting-first decisions
  - Depth-limited solver with continuation values bridges the gap
  - LZMA compression with lazy loading for blueprint data

### ACR Poker HUD
- **Repo:** https://github.com/ZHANGV25/ACRPoker-Hud-PC
- **What it is:** The HUD application that consumes the poker-solver engine
- **Key files:**
  - `cdp_ui.py` — Main HUD entry point with tkinter overlay
  - `src/cdp_reader.py` — Chrome DevTools Protocol reader for ACR game state
  - `solver/ranges.json` — Preflop GTO ranges (42 entries, semi-binary)
  - `solver/flop_solutions/` — Precomputed flop blueprints (47K files, ~3.4 GB)
  - `solver/solver-cli/` — Rust postflop-solver wrapper (`tbl-engine.exe`)
  - `solver/flop_lookup.py` — Suit isomorphism + blueprint lookup
  - `solver/action_history.py` — Preflop action reconstruction + HandTracker

## Commercial References (Not Open Source)

- **PioSOLVER:** https://piosolver.com — Industry standard NLHE solver. CPU-only, C/C++. Our benchmark reference.
- **GTO+:** Commercial solver, CPU-only. Comparable to PioSOLVER.
- **GTO Wizard:** https://gtowizard.com — Cloud-based GTO training tool. Has precomputed solutions for all spots. Potential source for preflop frequencies if we need them.

## CMU Course Materials
- **Lecture slides on Pluribus:** https://www.cs.cmu.edu/~sandholm/cs15-888F24/Lecture_14_Pluribus_and_depth-limited_subgame_solving.pdf
- **Brown's thesis:** http://reports-archive.adm.cs.cmu.edu/anon/2020/CMU-CS-20-132.pdf — "Equilibrium Finding for Large Adversarial Imperfect-Information Games"

## Key Concepts Quick Reference

| Concept | Where to Find |
|---------|--------------|
| CFR algorithm pseudocode | Pluribus supplementary, Algorithm 1 |
| Real-time search procedure | Pluribus supplementary, Algorithm 2 |
| 4 continuation strategies | Depth-limited paper, Section 4; Pluribus paper, "Search" section |
| Bayesian range updates | Pluribus supplementary, range tracking section |
| DCFR parameters | DCFR paper, Table 1 |
| Safe subgame solving / gadget | Safe subgame paper, Section 3 |
| O(N+M) showdown evaluation | Brown's poker_solver, `VectorEvaluator` class |
| Suit isomorphism | postflop-solver `isomorphism.rs`; our `precompute_flops.py` |
| Hand evaluation | OMPEval for benchmarks; our `hand_eval.h` (21× eval5 approach) |
