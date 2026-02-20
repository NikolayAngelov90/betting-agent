# Brainstorming Session Results

**Session Date:** 2026-02-19
**Facilitator:** AI Brainstorming Facilitator
**Participant:** Niki

## Session Start

**Approach:** AI-Recommended Techniques
**Sequence chosen:**
1. First Principles Thinking (creative) — ~15 min
2. SCAMPER Method (structured) — ~20 min
3. Assumption Reversal (deep) — ~15 min

**Context:** Improve ML prediction models in a football betting agent — richer features + incremental learning from settled picks.

## Executive Summary

**Topic:** ML model improvement for football betting agent — richer features + incremental learning

**Session Goals:** (1) Add more feature dependencies per match for better prediction accuracy. (2) Enable models to incrementally learn from settled picks after CI pipeline settles yesterday's matches.

**Techniques Used:** First Principles Thinking → SCAMPER → Assumption Reversal

**Total Ideas Generated:** 11 concrete ideas across 3 categories

### Key Themes Identified:

1. **Recency beats volume** — recent data weighted properly outperforms full flat history
2. **The best signals are already in the DB** — bookmaker odds, match dates, standings all unused as features
3. **The online learning loop is almost free** — one CI workflow line closes the feedback loop
4. **Style is the frontier** — the highest long-term edge but requires new data sources

## Technique Sessions

### Technique 1: First Principles Thinking

**Goal:** Strip assumptions, find what truly drives match outcomes.

**5 Fundamental Principles identified:**

| Principle | What to measure |
|---|---|
| **Momentum** | Weighted recent form + late-goal trend (goals in 75'+) + post-loss response rate |
| **Player quality** | Key player availability (injuries/suspensions) + squad depth drop-off index |
| **Style matchup** | Attack type vs defensive shape — press rate, possession%, shots-on-target ratio |
| **External pressure** | Match stakes index: derby flag, relegation zone proximity, title race position, cup elimination pressure |
| **Situational context** | Rest days between matches, travel distance (away), midweek European game flag |

**Decision:** Implement all 5 as new feature groups.

---

### Technique 2: SCAMPER Method

**7 lenses applied to the current ML pipeline:**

| Lens | Idea | Priority |
|---|---|---|
| **S — Substitute** | Replace raw form string with exponential decay form score (recent games weighted 3×). Form-adjusted Elo that reacts faster. | 2 |
| **C — Combine** | Style-xG interaction term: high-press team + high xG-against = counter-attack vulnerability flag | 6 |
| **A — Adapt** | Adapt financial RSI/MACD to team performance — detect "overbought" winning streaks likely to regress | 5 |
| **M — Modify** | Style-weighted H2H: only count H2H meetings with similar formations/styles (depends on C) | 7 |
| **P — Put to other use** | Bookmaker implied probability (Bet365/Pinnacle odds already in DB) fed as a feature — market consensus is one of the strongest predictors available | 1 |
| **E — Eliminate** | Auto-prune features with correlation > 0.8 to reduce model noise | 4 |
| **R — Reverse** | Settled pick result → immediate training signal. Online learning loop: daily CI settle → model weight update | 3 |

### Technique 3: Assumption Reversal

**5 assumptions reversed:**

| Assumption | Reversal |
|---|---|
| Models train once on historical CSV | Models retrain **daily** after settlement |
| Settled result = statistics only | Settled result = **training sample** |
| All historical matches equally valuable | Recent settled picks weighted higher via sliding window |
| Training separate from prediction | Training + prediction in **same CI loop** |
| More data = better | Rolling window of last ~2000 matches beats full history |

**Online learning architecture chosen — Option B (Daily Batch Retrain on Sliding Window):**

```
Daily CI Pipeline:
  --update  → fetch fixtures + odds + xG
  --settle  → mark picks win/loss + write actual scores to Match table
  --train   → retrain on last 2000 matches (includes yesterday's settled results)
  --picks   → predict with model that learned from yesterday
```

**Why Option B:**
- Works with existing XGBoost/RF — no architecture swap needed
- `train_ml_models()` already uses `order_by(match_date.desc()).limit(2000)` — sliding window built in
- `--settle` already writes `actual_home_goals/away_goals` to Match records — already valid training samples
- Gap is just: `--train` not called in CI. Add one line to workflow.

---

## Idea Categorization

### Immediate Opportunities

_Ideas ready to implement now_

1. Bookmaker implied probability as ML feature (Bet365/Pinnacle odds already in DB — zero new data needed)
2. Exponential decay form score replacing raw W/D/L string
3. Add `--train` step to CI after `--settle` (one line in workflow YAML)
4. Rest days between matches + midweek European game flag (match dates already in DB)

### Future Innovations

_Ideas requiring development/research_

5. Auto-prune correlated features (correlation matrix > 0.8 threshold)
6. Financial RSI/MACD adapted to team scoring streaks
7. Match stakes index (relegation proximity via standings API already available)
8. Squad depth drop-off index via injury scraper integration

### Moonshots

_Ambitious, transformative concepts_

9. Style matchup features (possession%, press rate — needs StatsBomb/Understat integration)
10. Style-weighted H2H (depends on style data above)
11. Travel distance fatigue model

### Insights and Learnings

_Key realizations from the session_

1. **The biggest quick win is already in the DB** — bookmaker implied probability is unused signal sitting in the odds table
2. **Online learning is already 90% built** — `train_ml_models()` already uses a recency-ordered sliding window; the only missing piece is calling `--train` in CI after `--settle`
3. **Momentum is the most underserved feature** — form string treats all results equally; decay weighting + late-goal trends = significant signal improvement for low effort
4. **Style vs style is the hardest but highest potential** — no current data source; needs external integration but could be the biggest edge long-term

## Action Planning

### Top 3 Priority Ideas

#### #1 Priority: Daily CI online learning loop + bookmaker odds as ML feature

- Rationale: Two highest-impact changes with lowest effort. Bookmaker implied probability is the single strongest unused signal — data already in DB. Daily `--train` after `--settle` closes the feedback loop so the model learns from every real prediction it makes.
- Next steps: (1) Add `home_implied_prob`, `away_implied_prob`, `draw_implied_prob` features to `feature_engineer.py` by querying Bet365/Pinnacle odds from DB. (2) Add `--train` step to CI workflow after `--settle`. (3) Run tests.
- Resources needed: Existing codebase only — no new data sources
- Timeline: Implement now (this session)

#### #2 Priority: Exponential decay form score + situational context features

- Rationale: Form string currently weights a 6-week-old win the same as last week's. Decay scoring is a pure math change. Rest days and midweek flag are computable from match dates already in DB — zero new API calls.
- Next steps: (1) Replace `home_form` / `away_form` raw string with decay-weighted score in `feature_engineer.py`. (2) Add `home_rest_days`, `away_rest_days`, `home_midweek_flag`, `away_midweek_flag` features. (3) Retrain.
- Resources needed: Existing match date data in DB
- Timeline: Implement now (this session)

#### #3 Priority: Match stakes index + auto-prune correlated features

- Rationale: Standings API already wired in — relegation proximity and title race position are computable. Feature pruning removes noise that confuses LR/RF.
- Next steps: (1) Add `home_relegation_gap`, `away_relegation_gap`, `home_title_gap`, `away_title_gap` from standings. (2) Add correlation pruning in `train_ml_models()` using pandas `corr()`. (3) Retrain and compare accuracy.
- Resources needed: Existing standings data from API-Football
- Timeline: Implement after priorities 1 & 2

## Reflection and Follow-up

### What Worked Well

- First Principles surfaced 5 clean feature categories in minutes
- SCAMPER's P-lens (bookmaker odds as feature) was the highest-value discovery — hiding in plain sight
- Assumption Reversal clarified that online learning is already 90% built

### Areas for Further Exploration

- Style matchup data (StatsBomb / Understat integration)
- RSI/MACD momentum indicators applied to team performance
- Travel distance fatigue modelling

### Recommended Follow-up Techniques

- Morphological Analysis — once style data is available, systematically map all feature combinations
- Five Whys — when model accuracy plateaus, diagnose which feature categories are underperforming

### Questions That Emerged

- What minimum sample size of settled picks is needed before daily retraining meaningfully improves accuracy?
- Should bookmaker odds be used as raw implied probability or as a calibrated probability (removing bookmaker margin)?
- Is a 2000-match sliding window optimal, or should it be tuned per league?

### Next Session Planning

- **Suggested topics:** Style data integration + RSI momentum features
- **Recommended timeframe:** After 30 days of daily retraining data has accumulated
- **Preparation needed:** Evaluate StatsBomb free tier / Understat scraping options for possession/press data

---

_Session facilitated using the BMAD CIS brainstorming framework_
