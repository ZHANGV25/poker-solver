# Path A: SaaS Poker Solver — Commercialization Plan

**Decision date:** 2026-03-31 | **Updated:** 2026-04-01
**Product:** 6-player No-Limit Hold'em solver as a cloud-based training/analysis tool
**Positioning:** "The only solver that does 6-player postflop"

---

## Why This Matters

The multiway postflop gap. GTO Wizard solved multiway preflop (up to 9 players, Feb 2026)
and 3-way postflop (Aug 2025). But they cap at 3 players postflop.
The real game regularly has 4-6 players seeing flops. Nobody solves that.

| Product | Preflop | Postflop | Approach | Price |
|---------|---------|----------|----------|-------|
| GTO Wizard Ultra | 9 players | **3 players max** | Neural net + CFR hybrid, on-demand | $229-279/mo |
| PioSOLVER | HU only | HU only | CFR+, desktop | EUR 450-800 one-time |
| MonkerSolver | N-way (slow) | 3-4 (weeks) | MCCFR, desktop, 256GB+ RAM | EUR 499 one-time |
| Simple 3-Way | — | 3 players | Cloud | $249/yr |
| **Us** | **6 players** | **6 players, <100ms** | **MCCFR blueprint + GPU search** | TBD |

Our edge is specifically **6-player postflop**. GTO Wizard owns multiway preflop.
We don't need to compete there — we need to own the postflop multiway space.

---

## Competitive Landscape (Updated April 2026)

### GTO Wizard — The Real Threat

GTO Wizard is further along on multiway than initially assessed:
- **Feb 2026:** Shipped multiway preflop solving up to 9 players (Ultra tier, $229-279/mo)
- **Aug 2025:** Shipped 3-way postflop solving
- **Approach:** Neural net + CFR hybrid trained on billions of hands. Solves on-demand (no precomputed blueprint). This is a fundamentally different architecture from ours.
- **Roadmap:** Their blog says scaling postflop beyond 3 players is planned but gives no timeline.
- **Public API:** In development for researchers/hobbyists.

**Why they can't easily get to 6-player postflop:**
- Their on-demand approach means solving a 6-player postflop tree in seconds from scratch. The game tree complexity grows combinatorially — 6-player flop trees are orders of magnitude larger than 3-player.
- Their neural net estimates value at depth limits. Training a net accurate enough for 6-player interactions requires vastly more data and compute.
- They took 2+ years from acquiring Ruse AI (2023) to shipping 3-way postflop (Aug 2025). Scaling to 6 is harder than scaling from 2 to 3.
- **Estimated 12-18 months before they could ship 6-player postflop** (unchanged).

### Our Architectural Advantage

We use a different approach that's better suited to full 6-player:
- **Precomputed blueprint:** 8 days of MCCFR on 72 cores produces a 413M info-set strategy covering all 6 players, preflop through river. The hard work is done offline.
- **GPU real-time search:** At runtime, we only solve the current subgame (one street), using the blueprint for depth-limited leaf values. This is fast (<100ms) because the search space is small.
- **Pluribus-exact:** Our algorithm matches the published Science 2019 paper exactly — external-sampling MCCFR with Linear CFR, pruning, 4 continuation strategies, A3 freezing. This converges to Nash equilibrium, not a neural approximation.

The tradeoff: we can't solve arbitrary custom spots on demand (our strategies are within our abstraction). GTO Wizard can. But for the standard 6-max cash game that 90%+ of players play, our blueprint covers it.

### Other Competitors

- **PioSOLVER:** Desktop-only, HU-only. Won't pivot to multiway. Not a threat.
- **MonkerSolver:** Development stalled. Takes weeks for 3-way. Not a threat.
- **Deepsolver:** Newer entrant, HU only. Not a threat for multiway.

---

## Product Tiers & Pricing

Anchored against GTO Wizard Ultra ($229-279/mo) — they charge this for 3-way.
We offer 6-way, which is strictly superior.

### Free Tier (Marketing Funnel)
- 6-player preflop strategy viewer (all positions, 100bb)
- Limited postflop: 5 spot queries/day
- Strategy visualization (color-coded range grids)
- Purpose: drive signups, content sharing, word-of-mouth

### Pro ($79/mo or $59/mo annual)
- Full preflop blueprint access (all stack depths)
- 50 postflop queries/day (any street, any texture)
- Hand history upload (PokerStars, GGPoker, ACR formats)
- GTO trainer / drill mode (daily practice spots)
- Basic aggregated reports

### Elite ($179/mo or $139/mo annual)
- Unlimited postflop queries
- Custom bet sizing trees
- Node locking (freeze opponent strategies for exploitative analysis)
- Advanced aggregated reports (flop/turn/river heatmaps)
- Range vs. range equity analysis
- Priority solve queue

### API / B2B ($500+/mo)
- Programmatic access to solver engine
- Batch solving (1000+ spots/day)
- Custom integration support
- For: training sites, poker rooms, content creators

### Revenue Projections (Conservative)

| Scenario | Users | Avg Revenue/User | ARR |
|----------|-------|-----------------|-----|
| Year 1 (launch) | 500-2,000 | $80/mo | $480K-$1.9M |
| Year 2 (growth) | 2,000-8,000 | $90/mo | $2.2M-$8.6M |
| Year 3 (mature) | 5,000-15,000 | $100/mo | $6M-$18M |

---

## Feature Roadmap

### Phase 1: Preflop-Only Launch (Target: 2-3 weeks after blueprint completes)

Launch with preflop only. Zero compute cost at any scale — static data served from
Vercel. Postflop goes behind a waitlist to validate demand before committing to GPU
infrastructure ($200-500/mo EC2 burn).

**Three access levels:**
- **No account (landing page visitor):** One position visible (UTG, 100bb, open).
  Interactive grid, can hover hands and see frequencies. AI Coach visible on a few
  hands, rest blurred with "sign up to see." This is the demo.
- **Free account (email signup):** Full preflop access. All positions, scenarios, stack
  depths. AI Coach on every hand. Preflop trainer with explanations. Range comparison.
  This is generous on purpose — it builds the user base and generates word of mouth.
- **Pro account ($49/mo, introduced later with postflop):** Everything free has, plus
  postflop solver, deeper trainer, hand history upload, session reports.

**No paid tiers at launch.** Free only for 4-6 weeks while user base grows and postflop
is built. Introducing Pro when there are thousands of free users who already trust the
product converts better than charging day one with no social proof.

**Must have for launch:**
1. Web-based preflop strategy viewer
   - All 6 positions (UTG, MP, CO, BTN, SB, BB)
   - Color-coded range grids with raise/call/fold/3bet frequencies
   - Stack depth selection (50bb, 100bb, 150bb, 200bb)
   - Scenario selection (open, vs open, vs 3-bet, vs 4-bet)
2. AI Strategy Coach — pregenerated explanations on every hand (see section below)
   - Click any hand → panel slides out with 2-3 sentence explanation
   - 16,224 explanations pregenerated as static JSON (~$2 to generate)
   - Zero runtime cost, zero latency, zero hallucination risk
3. Preflop trainer / drill mode
   - Random spot generation, user picks action, reveal GTO answer + AI explanation
   - Session stats (accuracy %, EV loss tracking)
   - Already built in frontend — wire to real blueprint data
4. Range comparison across positions
   - Side-by-side diff view (already built)
   - Add: single-hand position sweep — show how one hand's strategy changes
     across all 6 positions in one view (e.g., "AJo: UTG fold 72%, MP fold 55%,
     CO raise 80%, BTN raise 95%")
5. Multi-way impact toggle (landing page hero feature)
   - Toggle between HU baseline ranges and 6-player ranges
   - Watch the grid reshape in real time
   - Uses publicly available HU data as baseline, our blueprint as 6-player
   - Single most powerful marketing visual — demonstrates the product's
     reason to exist in one interaction
6. User accounts (Supabase auth, already built)
   - No Stripe at launch — no paid tier yet
7. Feedback button + Discord link
   - Simple text form → Supabase table
   - Discord server with #suggestions, #bugs, #general channels
8. Public roadmap page
   - Static page: what's built, what's in progress, what's planned
   - Replaces FAQ for "when is postflop coming?"
9. Postflop waitlist
   - Signup form on the postflop page (already shows "coming soon")
   - Track signups to gauge demand before investing in GPU infra

### Phase 2: Postflop Launch (4-8 weeks after Phase 1)

Only proceed if postflop waitlist shows strong demand. Introduces paid tier.

**Must have:**
1. Postflop spot solver
   - User inputs: positions, board, stack depth, action history
   - Returns strategy for each hand/bucket at decision point
   - Color-coded grid + EV display
   - GPU-accelerated: ~200ms per solve on RTX 3060
2. "What would villain do?" — Reverse range viewer
   - Toggle "Hero" / "Villain" on any solved spot
   - Shows villain's strategy weighted by Bayesian-narrowed range
   - Opacity layer: hands still in villain's range are bright, narrowed-out
     hands are ghosted
   - Aggregate bar: "Villain's likely next action: bet large 45%, bet small
     20%, check 35%" — weighted across entire narrowed range
   - "If villain does X" branches: click to walk the game tree forward one
     step, see hero's response, villain's range narrows further
   - AI Coach works from villain's perspective too: "If villain has KsQh here,
     the solver says they bet 85% of the time because they have top pair good
     kicker on a board where most draws missed."
   - Why this is unique: every other solver shows both players' strategies,
     but none show the narrowed range — the Bayesian update of what villain
     likely has based on observed actions. Our solver does this in real time.
3. AI Coach — postflop runtime explanations
   - Classify-then-phrase pipeline (see AI Coach section below)
   - Deterministic reason classification in our code, LLM only phrases it
   - Temperature 0, structured input, output validation, aggressive caching
4. Hand replay with solver overlay
   - Upload hand history → replay street by street
   - At each decision point: show solver's recommendation overlaid on
     what the user actually did, with EV difference
   - AI Coach explains each deviation
   - First 6-player hand replay tool that exists
5. Session leak report ("Table Scanner")
   - Upload batch of hands → one-page report of biggest leaks
   - Ranked by total EV lost: "You lost 12bb by c-betting too often in
     3-way pots" / "You lost 8bb folding too much to river bets"
   - AI Coach explains each leak
   - Actionable, specific, tied to real hands
6. Stripe billing (Pro tier: $49/mo)
7. Hand history parsing (already built in API — ACR, Stars, GG, Winamax, 888)

### Phase 3: Growth Features (Months 3-6)

8. Node locking (exploitative adjustments)
9. Aggregated reports (flop/turn/river strategy heatmaps across textures)
10. ICM / tournament mode
11. API access for power users
12. Mobile-responsive polish (or native app)

### AI Strategy Coach (All Tiers — Signature Feature)

**The problem:** The #1 complaint across all poker solvers is "it tells me WHAT to do
but not WHY." GTO Wizard, PioSOLVER, and every other tool output raw frequencies
(raise 60%, call 30%, fold 10%) with no explanation. Players memorize numbers without
building intuition.

**The feature:** An LLM (Claude) explains solver decisions in plain language, grounded
in the actual solver output. Not generic poker tips — specific explanations tied to the
exact board, ranges, and EV data from the solver.

**Critical design decision: NOT a chat.** This is a click-to-explain panel, not a
conversational interface. Chat invites open-ended questions that lead to hallucination.
The explanation box gives a controlled, grounded output for a fixed input (this hand,
this position, this scenario). The user clicks a hand on the range grid, a panel slides
out with 2-3 sentences explaining why.

**Example UX:**
> **Td9s — Under the Gun, 100bb, Open**
>
> Fold 72% · Call 0% · Raise 28%
>
> This hand is a marginal open from early position. When you raise, multiple players
> behind you will call with hands that dominate you — AT, KT, and overpairs all have
> you crushed. The solver raises occasionally to balance its range with suited broadways,
> but folds most of the time because the risk of playing a dominated hand out of position
> against multiple callers outweighs the reward.

**Available at every tier (including free).** Rationale:
- The people who need explanations most are the free/low-tier users still learning
- The cost is negligible (see pregeneration below)
- It's the single biggest differentiator vs every competitor — making it free makes it
  our brand identity, not a locked upsell
- A free user who can tap "why?" on any preflop hand and get a real explanation is a
  user who tells their friends about the product

**Tier gating — not the explanation, but the depth of what gets explained:**
- **Free:** AI Coach on preflop decisions (preflop viewer + explanations = best free
  poker tool that exists)
- **Pro:** AI Coach on postflop decisions, trainer feedback, hand history analysis
- **Elite:** AI Coach everywhere + deeper analysis ("how does this change if villain
  is tighter/looser"), comparative explanations across positions

#### Preflop Pregeneration (All Explanations Pre-Computed)

Preflop is small enough to pregenerate every explanation. No runtime API calls needed.

**Matrix:**
- 169 hand classes × 6 positions × 4 scenarios × 4 stack depths = **16,224 explanations**
- ~30-40% have non-obvious strategies worth detailed explanations (~6,000)
- Trivial spots (72o UTG = fold) get simple templated explanations

**Cost to pregenerate all 16,224:** ~$2 total using Claude Haiku
(~150 input tokens + ~80 output tokens per explanation at Haiku pricing).

**Implementation:**
- Pregenerate all explanations as a JSON file alongside preflop data
- Store as `{ position, hand, scenario, stack_depth } → explanation string`
- Serve as static lookups — zero latency, zero API cost, works offline
- Can review every explanation by hand before shipping (only 16K entries)
- Regenerate the full set when blueprint data updates

#### Postflop Runtime Explanations

Postflop combinations are too large to pregenerate (boards × ranges × actions × stacks).
Use a classify-then-phrase pipeline at runtime:

**Step 1 — Deterministic classification (our code, no LLM):**
```
Solver data → reason tag
Examples: "value_raise", "bluff_raise", "fold_dominated", "draw_priced_out",
          "deny_equity", "range_balance", "position_disadvantage"
```
Based on hero equity, EV gaps, range composition, board texture. The reasoning is
computed by our code — not the LLM.

**Step 2 — LLM phrasing (Claude Haiku, temperature 0):**
Feed structured solver data + reason tag. The model's job is translation, not analysis.

**Structured input format:**
```json
{
  "hero_hand": "Td9h",
  "hero_position": "BB",
  "board": "Ks9d4h",
  "street": "flop",
  "villain_position": "CO",
  "villain_action": "bet 75% pot",
  "villain_range_summary": {
    "top_pair_plus": 0.35, "flush_draws": 0.15,
    "overcards": 0.25, "air": 0.25
  },
  "solver_actions": {
    "check_raise": { "frequency": 0.73, "ev": 2.3 },
    "call": { "frequency": 0.22, "ev": 0.8 },
    "fold": { "frequency": 0.05, "ev": 0.0 }
  },
  "hero_equity_vs_range": 0.54,
  "reason_tag": "value_raise"
}
```

**System prompt constraints:**
- Only reference numbers provided in the input
- Never invent equity percentages, frequencies, or EV values
- Exactly 2-3 sentences
- Temperature 0 for deterministic output

**Step 3 — Output validation (our code):**
- Check: does explanation mention numbers not in the input? Flag.
- Check: does it reference cards not on the board? Flag.
- Check: longer than 4 sentences? Truncate or regenerate.

**Step 4 — Cache aggressively:**
Same board texture + same hand class + same action pattern → serve cached explanation.
Most users hit common spots. Cache hit rate will be high.

**Runtime cost per postflop explanation:** ~$0.0003 (Haiku). Negligible even at scale.

#### Hallucination Mitigation Summary

| Layer | What It Does |
|-------|-------------|
| Structured input | LLM sees only exact solver data, can't invent context |
| Reason classification | Reasoning is deterministic code, LLM only phrases it |
| Temperature 0 | Same input → same output, no creative drift |
| Output validation | Flag any numbers/cards not in input before showing user |
| Short output (2-3 sentences) | Hallucination risk scales with length |
| Pregeneration (preflop) | No runtime LLM at all for preflop — fully static |
| Caching (postflop) | Common spots served from cache, no regeneration |

**Competitive landscape:**
- Octopi Poker ($15/mo) has "George" AI assistant — navigation helper, not grounded
  in real solver output.
- AceSolver claims AI "why" explanations — controversial accuracy, banned from
  r/Poker_Theory. Community skeptical.
- GTO Wizard — no AI explanation feature. Raw frequencies only.
- PioSOLVER — nothing. Pure data.
- **Nobody has LLM explanations grounded in actual solver output.** This is an open lane.

**Why we're uniquely positioned:** Our solver provides richer context than anyone else
can feed an LLM — actual narrowed ranges from Bayesian updates, bucket compositions,
per-action EVs, and blueprint probabilities. GTO Wizard's neural net approach doesn't
expose this intermediate data the same way.

### What Competitors Do Well (Copy These)

**GTO Wizard:**
- Instant results (cloud pre-computed) — no waiting for solves
- Drill/trainer mode — gamified learning, daily streaks
- Clean UI — range grids, color coding, intuitive navigation
- Free tier that hooks users — rotating free spots, preflop charts
- Content machine — blog articles rank for poker strategy keywords
- Regional ambassadors — Japan expansion via Yokosawa (1M YouTube subs)

**PioSOLVER:**
- Deep customization — arbitrary bet sizes, node locking, scripting
- Accuracy — gold standard for heads-up GTO accuracy
- Batch analysis — solve hundreds of spots automatically
- Community scripts — shared configurations from coaches

**What we can do that nobody else can:**
- 6-player postflop solving — GTO Wizard caps at 3, MonkerSolver is unusable beyond 3
- "What does GTO actually look like 4-way on a wet board?" — still unanswerable by any other product
- MCCFR blueprint convergence (theoretically grounded Nash equilibrium, not neural approximation)
- Sub-100ms GPU search for 6-player subgames

**What GTO Wizard does better (be honest about this):**
- On-demand custom solving (any spot, any parameters) — we're limited to our abstraction
- 9-player preflop (we do 6)
- Product maturity (years of polish, 51-200 employees, 14+ poker site integrations)
- GTO Trainer, HH analyzer, aggregated reports, node locking — features we haven't built yet
- Neural net approach generalizes to novel situations; our blueprint is fixed

**Our positioning should be:**
- NOT "the first multiplayer solver" (GTO Wizard does multiway preflop + 3-way postflop)
- YES "the only solver that handles 4-5-6 player postflop spots"
- Emphasize the blueprint rigor: "413M info sets, 8 days of compute, Pluribus-exact algorithm"
- Price undercut: same or better multiway for less than GTO Wizard Ultra ($229-279/mo)

---

## Tech Architecture (SaaS)

### Backend
- **API server**: Python (FastAPI) or Node.js
- **Solver engine**: C/CUDA binaries called via subprocess or ctypes
- **Blueprint storage**: S3 — load into memory on server startup
- **Job queue**: Redis + worker for async postflop solves
- **Database**: PostgreSQL (users, billing, hand histories, saved spots)
- **Auth**: Clerk or Auth0
- **Payments**: Stripe

### Frontend
- **Framework**: Next.js (React)
- **Key components**:
  - Range grid viewer (13x13 matrix, color-coded)
  - Action frequency bars
  - Street navigator (preflop → flop → turn → river)
  - Hand history viewer
  - Trainer/drill interface
- **Hosting**: Vercel

### Infrastructure
- **GPU server**: Single dedicated box with RTX 3060+ for real-time solves
  - Or: cloud GPU instances (but more expensive)
  - 67ms per river solve = can handle ~15 concurrent users on one GPU
  - Scale: add GPUs as users grow
- **Blueprint serving**: Pre-load into RAM on server boot (~30-50GB)
- **CDN**: Cloudflare for static assets

### Scale Estimates
- 1 RTX 3060 handles ~15 QPS (river solves) or ~4 QPS (flop solves)
- 1,000 concurrent users probably generate ~10-50 solve requests/sec (bursty)
- Need 1-4 GPUs initially, scale with demand
- Blueprint RAM: ~30-50GB per server instance

---

## Go-to-Market Strategy

### Phase 1: Pre-Launch (Weeks 1-4)

**Content blitz — establish credibility before product exists:**

1. **Blog posts / Twitter threads** (publish 2-3 per week):
   - "What 6-Player GTO Actually Looks Like (It's Not What You Think)"
   - "The First Solver-Verified UTG Opening Strategy for 6-Max"
   - "How I Built the Only 6-Player Poker Solver at CMU"
   - "Why Every Multiway Strategy Article You've Read Is Wrong"
   - "HU Solver vs 6-Player Solver: How Much Does Multiway Change Your Strategy?"
   - Include actual solver output screenshots (range grids, EV comparisons)

2. **Forum seeding** (TwoPlusTwo + Reddit r/poker):
   - Post solver results in strategy discussion threads
   - "I've been working on a 6-player solver — here's what it says about [common spot]"
   - Answer multiway strategy questions with actual solver output
   - Don't hard-sell; let results speak

3. **Landing page + email capture:**
   - Domain: nexusgto.com (purchased, deployed on Vercel)
   - Show sample outputs (preflop ranges, postflop strategies)
   - "Join the waitlist for early access"
   - Goal: 500-2,000 emails before launch

4. **YouTube outreach:**
   - Contact 3-5 poker strategy channels for "sneak peek" collaborations
   - Targets: Raise Your Edge, SplitSuit, Jonathan Little, Finding Equilibrium
   - Offer free access in exchange for an honest review video
   - "CMU student builds the only 6-player postflop solver" is a compelling hook

### Phase 2: Launch (Weeks 5-8)

5. **Launch with free tier + Pro tier:**
   - Email the waitlist
   - Post launch announcement on TwoPlusTwo, Reddit, Twitter
   - Product Hunt launch (generates press even if poker isn't their core audience)

6. **Influencer seeding:**
   - Give free Elite access to 10-20 well-known players/coaches
   - They'll post about it organically if the product is good

7. **PR angle:**
   - "CMU student replicates Facebook's Pluribus AI" — tech press loves this angle
   - Pitch to TechCrunch, Hacker News, The Verge
   - Cross-post to CMU channels (university loves this kind of story)

### Phase 3: Growth (Months 3-6)

8. **GTO Trainer launch** (stickiest feature — reduces churn):
   - Daily drill spots
   - Accuracy tracking / leaderboards
   - Focus on multiway-specific drills (no other product can do this)

9. **Content flywheel:**
   - Weekly "Solver Spot of the Week" on YouTube/Twitter
   - Monthly deep-dive articles on multiway theory
   - Guest appearances on poker podcasts (Red Chip, Thinking Poker, Just Hands)

10. **Partnership outreach:**
    - Training sites (Upswing, Run It Once) — license solver data for their courses
    - Poker rooms (GGPoker, PokerStars) — game integrity / broadcast overlay
    - Coaching platforms — integrate solver analysis into coaching workflows

### Phase 4: Scale or Exit (Months 6-12)

11. **If traction is strong:** Raise seed funding, expand team, build mobile
12. **If acquisition interest:** GTO Wizard is the natural acquirer. They have multiway preflop but cap at 3-way postflop. Our 6-player postflop engine fills their biggest gap. They've acquired solver tech before (Ruse AI, 2023).
13. **B2B expansion:** Trading firms (SIG, Jane Street), academic licensing

---

## Marketing Channels (Ranked by Expected ROI)

| Channel | Cost | Expected Impact | Notes |
|---------|------|----------------|-------|
| **TwoPlusTwo posts** | Free | High | Where serious players discuss tools |
| **Reddit r/poker** | Free | High | Viral potential for novel solver output |
| **Twitter/X threads** | Free | High | Poker thought leaders retweet interesting analysis |
| **YouTube collaborations** | Free access to creators | Very high | Strategy channels have 50K-800K subs |
| **Blog SEO** | Time only | Medium (slow) | Long-term flywheel; "multiway solver" queries have no competition |
| **Poker podcasts** | Free (guest appearances) | Medium | Niche but high-trust audience |
| **Tech press (HN, TC)** | Free (pitch) | Medium (spike) | One-time traffic spike, good for credibility |
| **Paid ads** | $$$$ | Low | Poker players are ad-blind; organic works better |

---

## Key Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Blueprint doesn't converge well | Medium | Critical | Validate exploitability before launch; can re-run with more compute |
| GTO Wizard ships 6-player postflop faster than expected | Low-Medium | High | They took 2+ years for 3-way. But they have 51-200 employees and neural net infrastructure. Move fast, consider acquisition positioning. |
| GTO Wizard's on-demand approach is more flexible | High | Medium | Our blueprint covers standard 6-max cash. For custom spots, we'd need to add on-demand solving or accept the limitation. |
| Low demand ("3-way postflop is enough") | Medium | High | Content showing 4-6 way strategies differ dramatically from 3-way. Free tier proves value. |
| GPU server costs too high at scale | Low | Medium | Optimize solve caching; pre-compute common spots |
| UX/UI can't match GTO Wizard polish | Medium | Medium | Hire a designer; MVP can be functional-first. GTO Wizard users complain about complexity — simpler can be better. |
| Patent risk (Pluribus IP) | Low | Medium | Algorithm is published in Science; implementation is independent |
| GTO Wizard launches public API first | Medium | Low | They announced it's in development. If they ship it, our B2B angle weakens but consumer product unaffected. |

---

## Legal Checklist

- [ ] Register business entity (LLC or C-Corp)
- [ ] Terms of Service: "For educational and training purposes only, not for use during live play"
- [ ] Privacy Policy (GDPR if serving EU users)
- [ ] Payment processing (Stripe — works fine for poker training tools)
- [ ] Patent search on Brown/Sandholm filings (due diligence)
- [ ] No gambling license needed (training tool, not a poker room)

---

## Immediate Next Steps (April 2026)

**DONE (as of April 1):**
- ~~Landing page~~ → live at nexusgto.com (Vercel)
- ~~Waitlist~~ → wired to Supabase, rate limited
- ~~Frontend MVP~~ → preflop viewer, postflop (coming soon), 404, mobile responsive
- ~~Repo protection~~ → public repo stripped, code in private poker-solver-dev
- ~~Domain~~ → nexusgto.com purchased and deployed

**NEXT:**
1. **April 3-4**: Blueprint compute finishes → download, validate, terminate EC2
2. **April 5-7**: Wire blueprint into runtime → test full query pipeline
3. **April 7-10**: FastAPI backend serving real preflop data + postflop GPU solves
4. **April 10-14**: Connect frontend to real solver API, restore postflop page
5. **April 14-21**: Content creation (3-5 articles with real solver output screenshots)
6. **April 21-28**: Forum seeding (TwoPlusTwo, Reddit) + YouTube outreach
7. **May 1-7**: Auth (Clerk) + billing (Stripe) integration
8. **May 7-21**: GTO Trainer (the #1 retention feature — interactive drilling, scored decisions)
9. **May 21-30**: Public launch with free + Pro tiers
10. **June**: Hand history upload, aggregated reports

---

## Financial Summary

| Item | Cost |
|------|------|
| Blueprint compute (one-time) | ~$235 |
| Domain + hosting (annual) | ~$200 |
| GPU server (monthly) | ~$100-300 (dedicated box) or $500+ (cloud) |
| Stripe fees | 2.9% + $0.30 per transaction |
| Vercel (frontend hosting) | Free tier initially |
| Total monthly burn (pre-revenue) | ~$300-600 |

**Break-even**: ~5-8 Pro subscribers ($79/mo × 5 = $395/mo)

This is an extremely capital-efficient business. No VC needed unless scaling aggressively.
