# Betting Agent — Full Predictive Performance Audit & Improvement

You are a senior quantitative sports-betting engineer, football data scientist, ML engineer and statistical modeller.

Repository:

https://github.com/NikolayAngelov90/betting-agent

Your mission is NOT to simply refactor the code.

Your mission is to determine whether every part of this system actually helps produce better-calibrated football probabilities, better value identification, better CLV and ultimately better long-term expected ROI.

The target is NOT "guaranteed winning bets" — that is impossible.

The target is:

> maximize out-of-sample predictive quality and risk-adjusted expected value using ONLY legally accessible and currently available free data sources already used by the application, plus genuinely free public sources if they can be integrated reliably.

Do not make the model more complicated unless the additional complexity produces measurable out-of-sample improvement.

---

# IMPORTANT: READ THE ENTIRE REPOSITORY FIRST

Before changing ANY code:

Read the entire repository.

Do not limit yourself to:

* README
* models
* betting_agent.py

Inspect every relevant Python file, configuration file, workflow, SQL migration, test, scraper, feature module, model, reporting module and data-processing module.

Pay particular attention to:

* src/agent/
* src/models/
* src/features/
* src/data/
* src/scrapers/
* src/betting/
* src/reporting/
* config/
* migrations/
* tests/
* .github/workflows/
* docs/
* FEATURES.md

Also inspect the recent egress optimizations and history mirror/cache implementation.

Do not undo the previous egress optimizations.

---

# PHASE 1 — BUILD THE COMPLETE SYSTEM MAP

Create a complete data-flow map:

FREE DATA SOURCES
↓
DATA INGESTION
↓
DATABASE
↓
HISTORICAL DATA
↓
FEATURE ENGINEERING
↓
INDIVIDUAL MODELS
↓
ENSEMBLE
↓
CALIBRATION
↓
BOOKMAKER PROBABILITIES
↓
VALUE / EV
↓
BET SELECTION
↓
RISK MANAGEMENT
↓
CLAUDE REVIEW
↓
FINAL PICK
↓
SETTLEMENT
↓
LEARNING
↓
MODEL UPDATE

For every stage identify:

* inputs
* outputs
* database reads
* database writes
* transformations
* assumptions
* potential information leakage
* potential bias
* potential stale data
* potential duplicated information
* potential statistical errors

---

# PHASE 2 — AUDIT EVERY FUNCTION

For every meaningful function in the application:

1. Explain what the function does.
2. Explain what data it uses.
3. Explain why that data is relevant.
4. Determine whether the implementation is statistically correct.
5. Determine whether it contains leakage.
6. Determine whether it duplicates another feature/model.
7. Determine whether it adds predictive information.
8. Determine whether the data is available at prediction time.
9. Determine whether the calculation is robust with small samples.
10. Determine whether it should be modified, removed or retained.

Create a table:

| Function | Purpose | Data | Predictive value | Leakage risk | Problem | Recommendation |
| -------- | ------- | ---- | ---------------- | ------------ | ------- | -------------- |

Do NOT automatically change every function.

---

# PHASE 3 — DATA SOURCE AUDIT

The application currently uses multiple free data sources.

Audit each one individually.

Known sources include:

* Flashscore
* API-Football
* football-data.org
* The Odds API
* Open-Meteo
* bookmaker odds
* Claude web research

For every source determine:

* what information is collected
* how frequently it is updated
* whether historical coverage is sufficient
* whether the information is available BEFORE kickoff
* whether the timestamp is stored
* whether the data can accidentally leak post-match information
* data quality
* missing-data rate
* duplicate-data rate
* usefulness for prediction

Do NOT assume that more data = better predictions.

---

# PHASE 4 — FREE DATA SOURCE RESEARCH

Investigate whether there are additional genuinely free football data sources that can improve predictions.

Only consider sources that are:

* legally accessible
* genuinely free
* reasonably reliable
* automatable
* available without paid API credits
* useful for pre-match prediction

Potential categories:

### Team performance

* recent results
* home/away strength
* xG
* shots
* shots on target
* possession
* attacking efficiency
* defensive efficiency

### Squad

* injuries
* suspensions
* expected absences
* player availability
* lineup information when available before kickoff

### Context

* rest days
* fixture congestion
* travel
* competition importance
* qualification/relegation/title pressure

### Market

* opening odds
* current odds
* closing odds
* line movement
* bookmaker consensus

### Match environment

* weather
* temperature
* wind
* precipitation
* pitch conditions where reliably available

### Referee

* cards
* penalties
* fouls
* home/away tendencies

But do NOT add a source merely because it exists.

For every candidate source estimate:

* predictive value
* reliability
* update frequency
* implementation cost
* free-tier limitations
* risk of scraping instability
* expected impact on model performance

---

# PHASE 5 — FEATURE ENGINEERING AUDIT

The repository currently contains 14 feature sections.

Audit every feature individually.

Known categories include:

* team form
* Poisson strengths
* Elo
* H2H
* league position
* international competition
* xG
* extended statistics
* referee
* momentum
* bookmaker probabilities
* odds movement
* situational context
* league statistics
* weather

For EVERY feature determine:

### A. Is it predictive?

Does it contain genuine signal?

### B. Is it redundant?

Does another feature already encode the same information?

### C. Is it correctly calculated?

Check formulas.

### D. Is the historical window appropriate?

Compare:

* 3 matches
* 5 matches
* 10 matches
* 20 matches
* exponentially weighted history

Do not assume 10 is optimal.

### E. Is it time-safe?

A feature must contain ONLY information available at the prediction timestamp.

### F. Does it behave differently across leagues?

Check whether normalization is required.

### G. Is the sample size sufficient?

Avoid noisy features derived from tiny samples.

### H. Does missing-data handling introduce bias?

Audit every fallback/default value.

---

# PHASE 6 — TIME LEAKAGE AUDIT

This is extremely important.

Perform a strict walk-forward leakage audit.

For every feature ask:

> "If this prediction were generated at 10:00 on match day, could the system actually know this value?"

Look specifically for:

* future match results
* future league standings
* post-match xG
* post-match injuries
* final odds used when only opening odds were available
* future referee statistics
* future team strength
* future Bayesian weights
* future calibration parameters
* future EV thresholds
* settlement information accidentally entering training
* data loaded after kickoff
* data ordering bugs

Do not accept "the database contains it" as proof that it was available at prediction time.

---

# PHASE 7 — POISSON / DIXON-COLes AUDIT

Deeply review:

src/models/poisson_model.py

Check:

* attack strength
* defence strength
* home advantage
* Dixon-Coles correction
* rho
* time decay
* xG integration
* league normalization
* low-score probabilities
* parameter estimation
* regularization
* sparse teams
* promoted teams
* national teams
* international matches

Test whether the current:

rho = -0.13

and

180-day half-life

are actually optimal.

Do not assume these values are correct.

Use walk-forward backtesting to compare reasonable alternatives.

---

# PHASE 8 — ELO AUDIT

Deeply review:

src/models/elo_system.py

Test:

* K-factor
* home advantage
* season regression
* initial ratings
* newly promoted teams
* international teams
* strength of schedule
* margin-of-victory handling
* recency weighting
* league strength normalization

Determine whether Elo is genuinely adding signal beyond Poisson and ML.

---

# PHASE 9 — MACHINE LEARNING AUDIT

Deeply review:

src/models/ml_models.py

Audit:

* Logistic Regression
* Random Forest
* XGBoost
* feature preprocessing
* missing values
* class imbalance
* hyperparameters
* feature scaling
* feature selection
* probability outputs
* training window
* retraining frequency

Check whether every model actually improves the ensemble.

Run ablation tests:

ML

ML + Elo

ML + Poisson

ML + bookmaker

full ensemble

Remove any model that consistently hurts out-of-sample performance.

---

# PHASE 10 — ENSEMBLE AUDIT

Deeply review:

src/models/ensemble.py

The current architecture uses:

* Poisson
* XGBoost
* Random Forest
* Elo
* bookmaker probabilities

Audit whether:

* weights are optimal
* weights are static when they should be dynamic
* bookmaker weight is too high
* bookmaker probability is being counted twice
* ML models already contain bookmaker features
* correlated models are being overweighted

Test multiple ensemble approaches:

1. Current weighted average
2. Optimized static weights
3. League-specific weights
4. Market-specific weights
5. Time-decayed weights
6. Bayesian weights
7. Stacked/meta-model approach

Only keep changes that improve genuine out-of-sample results.

---

# PHASE 11 — BOOKMAKER BLEND AUDIT

This is a critical area.

The current system uses a large bookmaker blend.

Determine whether bookmaker probabilities are:

* correctly margin-adjusted
* normalized
* based on the correct market
* based on current vs opening odds
* duplicated in ML features
* overly dominant

Test:

0% bookmaker

20%

40%

50%

60%

70%

80%

and determine which produces the best out-of-sample results.

Do not optimize purely for accuracy.

Also measure:

* CLV
* Brier score
* log loss
* ROI
* calibration

---

# PHASE 12 — PROBABILITY CALIBRATION

Deeply audit:

src/models/probability_calibration.py

Determine whether calibration is:

* statistically valid
* trained only on historical information
* walk-forward safe
* sufficiently sampled
* market-specific
* league-specific where justified

Compare:

* no calibration
* isotonic
* Platt scaling / logistic calibration
* beta calibration if appropriate

Do not overfit calibration.

---

# PHASE 13 — BAYESIAN WEIGHTS

Deeply audit:

src/models/bayesian_weights.py

Determine whether:

* prior is sensible
* decay is appropriate
* league-specific weights have enough data
* market-specific weights have enough data
* cold-start behaviour is correct
* uncertainty is correctly represented
* recent performance is over-weighted

Test different decay periods.

---

# PHASE 14 — BETTING VALUE ENGINE

Audit every calculation involving:

* implied probability
* bookmaker margin
* expected value
* edge
* Kelly
* confidence
* model agreement
* stake
* odds filtering

Verify formulas mathematically.

Particularly verify that EV is calculated against the correct market probability.

Check whether bookmaker margin is removed correctly before comparison.

---

# PHASE 15 — RISK MANAGEMENT AUDIT

Review:

* Kelly fraction
* maximum stake
* drawdown circuit breaker
* correlation filter
* daily exposure
* per-league cap
* per-match cap
* odds range
* confidence thresholds
* EV threshold
* model divergence guard

Determine whether any rule accidentally removes profitable bets.

Backtest:

current risk management

vs

no risk management

vs

optimized risk management.

Measure:

* ROI
* maximum drawdown
* Sharpe-like risk-adjusted return
* volatility
* losing streaks
* bankroll survival

---

# PHASE 16 — MARKET SELECTION

Determine which markets the system is actually good at predicting.

Analyze separately:

* Home Win
* Draw
* Away Win
* Over 2.5
* Under 2.5
* BTTS
* team goals
* other markets present in the code

Do NOT assume all markets should be treated identically.

Calculate performance by:

* market
* league
* odds range
* probability range
* model confidence
* season
* time period

Remove markets that consistently destroy expected value.

---

# PHASE 17 — ODDS MOVEMENT / CLV

Deeply analyze odds movement.

Determine whether:

opening odds

↓

current odds

↓

closing odds

are available and timestamped correctly.

Calculate Closing Line Value where possible.

CLV should become one of the main diagnostics.

A strategy that consistently gets positive CLV but has short-term negative ROI should NOT automatically be removed.

---

# PHASE 18 — BACKTESTING

This is mandatory.

Do NOT optimize parameters against the entire historical dataset.

Use chronological walk-forward validation.

Example:

TRAIN
2023 → 2024

TEST
2025

then:

TRAIN
2023 → 2025

TEST
2026

Never allow future information into training.

Use realistic prediction timestamps.

---

# PHASE 19 — AVOID OVERFITTING

This is extremely important.

Do not optimize dozens of parameters simultaneously.

If you test:

* 10 half-lives
* 10 thresholds
* 10 ensemble weights
* 10 calibration parameters

you will almost certainly overfit.

Use:

* sensible parameter ranges
* nested or rolling validation where practical
* out-of-sample evaluation
* holdout periods

Prefer robust parameters that perform reasonably across multiple periods over parameters that produce one spectacular backtest.

---

# PHASE 20 — PERFORMANCE METRICS

Do NOT use ROI alone.

Track at minimum:

### Prediction quality

* Log Loss
* Brier Score
* calibration error
* accuracy

### Betting quality

* ROI
* yield
* CLV
* average odds
* hit rate
* EV realized

### Risk

* maximum drawdown
* volatility
* losing streak
* bankroll trajectory

### Stability

* performance by league
* performance by market
* performance by odds range
* performance by month

---

# PHASE 21 — CLAUDE REVIEW AUDIT

The current pipeline performs a Claude web-research review before finalizing picks.

Audit this very carefully.

Determine whether Claude:

* adds genuine predictive information
* merely follows bookmaker consensus
* introduces confirmation bias
* changes statistically good picks into worse picks
* uses information that was unavailable at the actual decision timestamp
* leaks future information
* changes picks without a quantitative edge
* disproportionately favours popular teams

Create an experiment:

MODEL PICK

vs

CLAUDE REVIEWED PICK

Compare independently.

If Claude review does not improve out-of-sample performance, recommend disabling or restricting it.

Do NOT assume AI review improves predictions simply because it can research more information.

---

# PHASE 22 — FEATURE ABLATION

Run systematic feature-group ablation.

Compare:

Baseline

* Team form
* xG
* Elo
* H2H
* standings
* referee
* momentum
* odds
* odds movement
* injuries
* weather
* situational
* league statistics

For each group calculate its incremental contribution.

Identify:

HIGH VALUE FEATURES

LOW VALUE FEATURES

REDUNDANT FEATURES

HARMFUL FEATURES

NOISY FEATURES

Remove features that consistently reduce out-of-sample performance.

---

# PHASE 23 — FREE DATA PRIORITIZATION

After the audit, produce:

## Tier 1 — Must have

Free sources/features that clearly improve predictions.

## Tier 2 — Useful

Potential improvement but limited evidence.

## Tier 3 — Not worth implementing

Data that adds complexity without measurable benefit.

Do NOT add data merely to make the system look more sophisticated.

---

# PHASE 24 — IMPROVE THE SYSTEM

Only after completing the audit:

Implement the improvements that have strong evidence.

For every change provide:

FILE

FUNCTION

OLD BEHAVIOUR

NEW BEHAVIOUR

WHY IT SHOULD IMPROVE PREDICTION

BACKTEST RESULT

RISK

---

# HARD RULES

## Rule 1

Never optimize against future results.

## Rule 2

Never use post-match information in pre-match features.

## Rule 3

Never claim a strategy is profitable based only on training data.

## Rule 4

Never add complexity without measurable benefit.

## Rule 5

Do not destroy the existing egress optimizations.

## Rule 6

Do not replace working components just because another technique is newer.

## Rule 7

Do not optimize purely for hit rate.

## Rule 8

Do not optimize purely for ROI.

## Rule 9

Prefer calibrated probabilities over raw classification accuracy.

## Rule 10

Prefer robust out-of-sample improvements over impressive backtest improvements.

---

# IMPLEMENTATION STRATEGY

Do this in stages.

### Stage 1

Complete repository audit.

NO CODE CHANGES.

Produce the full findings report.

### Stage 2

Identify the top 10 improvements by expected predictive impact.

### Stage 3

Implement only the highest-confidence improvements.

### Stage 4

Add regression tests.

### Stage 5

Run walk-forward backtests.

### Stage 6

Compare BEFORE vs AFTER.

### Stage 7

Only keep changes that demonstrate meaningful improvement.

---

# FINAL REPORT

At the end produce a comprehensive report with:

## 1. Executive summary

What is currently strong and weak.

## 2. Architecture assessment

Score 0–10.

## 3. Data quality assessment

Score 0–10.

## 4. Feature quality

Top 20 useful features.

Top 20 weak/redundant features.

## 5. Model quality

Poisson

Elo

ML

Ensemble

Bookmaker

Bayesian

Calibration

Claude review

## 6. Leakage audit

List every potential leakage issue discovered.

## 7. Backtest results

Before vs after.

## 8. Market performance

Performance by market.

## 9. League performance

Performance by league.

## 10. Odds-range performance

Performance by odds bands.

## 11. CLV analysis

If sufficient historical odds data exists.

## 12. Risk analysis

Drawdown and bankroll behaviour.

## 13. Free data opportunities

Ranked by expected value.

## 14. Code changes

Every modified file and function.

## 15. Remaining weaknesses

What cannot currently be solved with available free data.

---

# MOST IMPORTANT FINAL QUESTION

Answer this explicitly:

> If this system had to operate for the next 12 months using only the currently available free data sources, what are the 5 changes most likely to improve its long-term risk-adjusted betting performance?

Rank them from #1 to #5.

Do not choose based on how interesting the technology is.

Choose based on evidence from the repository, statistical reasoning and out-of-sample results.