# Stage 3 — Evidence-Driven Betting Model Reconstruction

You have completed the full audit.

Do NOT start another generic audit.

Use the audit findings and statistical tests as the source of truth for this iteration.

The current evidence indicates that the system does NOT currently demonstrate a reliable predictive edge over bookmaker probabilities.

Therefore, the objective of this stage is NOT to make the system look more sophisticated.

The objective is to:

1. Fix the statistical and implementation defects.
2. Fix the evaluation infrastructure.
3. Establish a trustworthy baseline.
4. Improve the model only where out-of-sample evidence supports the change.
5. Remove components that demonstrably hurt performance.
6. Never optimize against the same data used to evaluate the improvement.

---

# CURRENT AUDIT FINDINGS

Treat these findings as established until independently disproven:

### Current predictive performance

* 1,018 settled picks analyzed.
* 5,771 out-of-sample matches in walk-forward testing.
* Logistic divergence coefficient:
  +0.245
  z = 1.11
  p = 0.265
* Brier:
  model = 0.2499
  constant baseline = 0.2498
  raw 1/odds = 0.2410
* Blend sweep:
  w=0 → log-loss 1.0412
  w=0.60 → 0.9983
  w=1.0 → 0.9929
* Realized ROI = approximately -3.6%.

Interpretation:

The current model has NOT demonstrated a statistically reliable edge over the bookmaker market.

Do not claim otherwise.

---

# CRITICAL PRINCIPLE

From this point forward:

NO optimization without out-of-sample validation.

NO parameter tuning against the final evaluation period.

NO hard-coded threshold based on settled outcomes without a holdout.

NO "this looks better" decisions.

NO optimizing for accuracy alone.

NO optimizing for ROI alone.

Use:

* log loss
* Brier score
* calibration
* CLV
* ROI
* drawdown
* stability across time

as separate measurements.

---

# PHASE 1 — CREATE A TRUSTWORTHY BASELINE

Before changing the model, create a reproducible baseline experiment.

The baseline must include:

### Market-only baseline

1 / de-vigged bookmaker probability

### Raw market baseline

1 / odds

### Current production model

Current ensemble exactly as shipped.

### Elo-only

### Poisson-only

### ML-only

### Market + model

All evaluated on identical chronological matches.

Produce:

| Model | Log Loss | Brier | Calibration | ROI | CLV |
| ----- | -------: | ----: | ----------: | --: | --: |

If CLV is unavailable, explicitly report that.

This baseline becomes immutable.

---

# PHASE 2 — FIX THE PLAIN SOFTWARE BUGS

Fix the two confirmed implementation bugs first.

## Bug 1

`src/agent/betting_agent.py`

Around the identified CV/report code:

`ml_models._models`

The actual attribute is:

`models`

Fix this.

Add a regression test proving that the CV/report path actually executes.

---

## Bug 2

`src/models/bayesian_weights.py`

`update()` currently writes league/global parameters regardless of `market`.

Fix the market separation.

The update for:

1X2

must not contaminate:

O/U

BTTS

or other markets.

Add tests proving market isolation.

---

# PHASE 3 — FIX BAYESIAN WEIGHT LEARNING

The audit found:

* Poisson receives 872 observations
* Elo receives 436
* Bayesian normalization collapses near-identical accuracies toward ~0.33
* the current learner cannot meaningfully distinguish model quality
* there is evidence of double updating

Do not simply change constants.

Re-design the weighting logic around predictive loss.

The weight-learning objective should be based on:

* out-of-sample log loss
* optionally Brier score as secondary metric

NOT classification accuracy.

The system must:

* update only from genuinely new settled predictions
* never update twice from the same prediction
* keep market-specific state
* keep league-specific state only where sample size is sufficient
* have a global fallback
* prevent tiny samples from dominating
* use sensible shrinkage toward global weights

Add tests for:

* duplicate update
* market isolation
* league isolation
* cold start
* insufficient sample size
* Bayesian update correctness

---

# PHASE 4 — REMOVE UNSUPPORTED HARD-CODED GATES

The audit identified approximately 15 hard-coded rules fitted on only 30–100 settled bets without holdout validation.

Examples include:

* over_3.5 exclusion
* club_pick_min_ev
* club BTTS ban
* wc_mismatch rules
* other outcome-derived gates

DO NOT blindly delete them.

Instead:

1. Locate every such rule.
2. Document why it exists.
3. Identify the historical data used to establish it.
4. Determine whether it was fitted using future outcomes.
5. Move it behind configuration/feature flags.
6. Remove it from production unless it survives proper walk-forward validation.

Every rule must have:

TRAIN PERIOD

VALIDATION PERIOD

HOLDOUT PERIOD

If a rule does not survive out-of-sample testing, remove it.

---

# PHASE 5 — FIX EV SELECTION

The audit found:

EV quintile Q3:

claimed EV = +11.6%

realized ROI = -19.3%

while Q1:

claimed EV = -4.8%

realized ROI = +2.4%.

This strongly suggests the current EV calculation and/or selection logic is unreliable.

Audit the entire EV pipeline.

Verify mathematically:

implied probability

↓

vig removal

↓

model probability

↓

edge

↓

expected value

↓

stake

↓

selection

Do not simply lower or raise the EV threshold.

First determine WHY claimed EV is anti-predictive.

Check for:

* bookmaker margin
* probability normalization
* double counting
* stale odds
* incorrect outcome mapping
* incorrect decimal odds
* model calibration
* market selection
* leakage
* data timestamp problems

Only after fixing the calculation should thresholds be evaluated.

---

# PHASE 6 — BOOKMAKER MARKET AS THE PRIMARY BASELINE

The audit shows that bookmaker probabilities currently outperform the model.

Therefore, do NOT try to force the model to beat the market immediately.

Build the architecture around:

MARKET BASELINE

*

INCREMENTAL MODEL SIGNAL

The model should answer:

> "Can our independent information improve the bookmaker probability?"

not:

> "Can our model predict football better than the bookmaker?"

Test:

Market only

Market + Elo

Market + Poisson

Market + ML

Market + all independent signals

The model must earn its weight.

If adding a component worsens out-of-sample log loss, reduce or remove it.

---

# PHASE 7 — STOP DOUBLE COUNTING THE MARKET

Audit every location where bookmaker information enters the system.

Create a complete dependency map.

Determine whether bookmaker probabilities appear in:

* ML features
* ensemble weights
* final bookmaker blend
* EV calculation
* calibration
* Bayesian weights
* Claude review

If the same market information is used multiple times, quantify the double counting.

The final architecture should have one clearly defined market signal.

---

# PHASE 8 — POISSON RECONSTRUCTION

Audit the Poisson implementation.

The audit found:

* 180-day half-life is not optimal
* 365–730 days perform better
* rho = -0.13 hurts 1X2
* rho has essentially no effect on O/U 2.5

Do not simply change:

180 → 365

and

rho = -0.13 → 0.

Run a controlled walk-forward experiment.

Test a small, theoretically justified parameter set.

For example:

half-life:

180

365

540

730

rho:

-0.10

-0.05

0

+0.05

Do NOT perform an unrestricted grid search.

Choose parameters based on robustness across multiple periods.

If rho consistently hurts performance:

remove the correction for the affected market.

Do not force Dixon-Coles into every market.

---

# PHASE 9 — ELO RE-EVALUATION

The audit found:

Elo log-loss = 1.0279

Poisson log-loss = 1.0385

Elo is currently the strongest standalone model.

Investigate why Elo is underweighted.

Test:

Elo alone

Market + Elo

Market + Poisson

Market + Elo + Poisson

Determine incremental value.

Also test:

* K factor
* home advantage
* season regression
* initial rating
* recency

Only test a small number of theoretically sensible alternatives.

Do NOT overfit Elo parameters.

---

# PHASE 10 — DATA COVERAGE

The audit found:

xG coverage ≈ 1.9%

shots/possession/corners ≈ 2.4%

injuries = 17 rows

Poisson xG has effectively never activated.

Therefore:

DO NOT continue optimizing models around data that barely exists.

First establish the best achievable model using high-coverage data.

Then investigate free data expansion.

---

# PHASE 11 — UNDERSTAT xG

Investigate the feasibility of integrating Understat as a genuinely free source.

Before implementation determine:

* legal/technical accessibility
* historical coverage
* league coverage
* update frequency
* reliability
* scraping stability
* rate limits
* whether historical xG can be mapped to existing fixtures

Target:

increase xG coverage from ~1.9% toward meaningful coverage.

Do NOT promise 60%.

Measure actual coverage after integration.

If integration is unreliable, do not make it a critical dependency.

---

# PHASE 12 — FREE DATA EXPANSION

Prioritize free information that is available BEFORE kickoff.

Investigate, in order:

1. xG
2. shots
3. lineups
4. injuries
5. suspensions
6. rest days
7. fixture congestion
8. referee
9. odds movement
10. weather

For each source measure:

coverage

freshness

historical availability

data quality

predictive contribution

Only integrate sources that can be evaluated historically.

---

# PHASE 13 — CLOSING LINE VALUE

This is now a major infrastructure priority.

Current state:

0 of 124,158 odds rows are stored after pick day.

Therefore current `avg_clv` is NOT genuine CLV.

It is merely:

model_probability - 1 / odds

Rename or remove this metric.

Do not call it CLV.

---

## Build genuine CLV tracking

Use historical football-data.co.uk CSVs where appropriate.

Use the available free Odds API quota for forward snapshots.

Record:

* bookmaker
* market
* selection
* odds
* timestamp

At minimum capture:

opening/current price

and

closing price.

Implement a proper CLV calculation.

Example concept:

pick price = 2.20

closing price = 2.00

The bet obtained a favorable line move.

Do not hard-code the formula without considering:

* market type
* decimal odds
* margin
* outcome

---

# PHASE 14 — MODEL VS MARKET CONTRIBUTION

After CLV infrastructure exists, measure:

Does the model consistently select prices that subsequently move in its direction?

This is one of the strongest diagnostics available.

Measure:

* average CLV
* median CLV
* CLV by market
* CLV by league
* CLV by odds range
* CLV by model edge

If positive CLV is absent, do not claim predictive edge.

---

# PHASE 15 — CLAUDE REVIEW

Do not remove Claude review yet.

But isolate it experimentally.

Every pick should ideally record:

PRE_CLAUDE_MODEL_RESULT

POST_CLAUDE_RESULT

CLAUDE_CHANGED_PICK

CLAUDE_REASON

Then compare:

model-only

vs

Claude-adjusted

with identical opportunities.

Do not use the review itself as evidence that the model improved.

Also ensure Claude cannot access post-kickoff information when the decision is supposed to be pre-match.

---

# PHASE 16 — RETRAINING

Audit:

`ml_retrain_days`

The audit found that training currently runs every day despite configuration indicating 3 days.

Fix the scheduling logic.

Training should occur according to configuration.

Do not retrain simply because the workflow executes.

---

# PHASE 17 — FEATURE ABLATION

After fixing the infrastructure, run feature-group ablation.

Compare:

baseline market

market + Elo

market + Poisson

market + form

market + xG

market + injuries

market + referee

market + momentum

market + situational

market + odds movement

full model

For each group report:

incremental log-loss improvement

incremental Brier improvement

calibration improvement

CLV impact

ROI impact

sample size

statistical uncertainty

Remove consistently harmful feature groups.

---

# PHASE 18 — WALK-FORWARD VALIDATION

Use chronological validation.

Never random shuffle.

Example:

Train:
2022–2023

Validate:
2024

Train:
2022–2024

Validate:
2025

Train:
2022–2025

Holdout:
2026

The final holdout must NEVER be used to choose parameters.

If historical coverage makes these exact periods impossible, choose equivalent chronological windows.

Document the exact methodology.

---

# PHASE 19 — STATISTICAL SIGNIFICANCE

For every claimed improvement calculate uncertainty.

Do not say:

"ROI increased from 2% to 5%, therefore improvement."

Instead determine whether the difference is meaningful.

Use appropriate:

* confidence intervals
* bootstrap
* permutation tests
* paired tests where applicable

For ROI, account for the fact that betting outcomes are dependent on odds and not simply classification accuracy.

---

# PHASE 20 — DO NOT CHASE NOISE

If the evidence says:

"the model cannot currently beat the market"

that is an acceptable result.

The correct engineering decision may be:

* market-only betting
* fewer markets
* fewer leagues
* no bets until sufficient edge
* paper-trading mode
* data collection mode

Do NOT manufacture an edge by tuning thresholds until historical ROI becomes positive.

---

# PHASE 21 — PRODUCTION SAFETY

Preserve:

* Supabase egress optimizations
* history cache
* incremental sync
* existing tests
* database correctness
* scheduler reliability

Do not introduce unnecessary database traffic.

Any new data source must be evaluated for:

* request frequency
* caching
* rate limits
* egress
* failure handling

---

# IMPLEMENTATION ORDER

Execute in this exact order:

### Step 1

Fix software bugs.

### Step 2

Fix Bayesian weighting.

### Step 3

Fix EV calculation and remove unsupported thresholds.

### Step 4

Create immutable market/model baseline.

### Step 5

Implement genuine CLV tracking.

### Step 6

Fix retraining schedule.

### Step 7

Re-evaluate Elo and Poisson.

### Step 8

Improve high-coverage features.

### Step 9

Investigate/integrate free xG source.

### Step 10

Run final walk-forward comparison.

Do NOT jump directly to Step 9.

---

# BEFORE EVERY MAJOR CHANGE

Record:

CURRENT METRIC

EXPECTED IMPROVEMENT

HYPOTHESIS

DATA USED

VALIDATION PERIOD

Then implement.

After implementation:

MEASURED RESULT

If the result is negative or statistically inconclusive:

REVERT THE CHANGE.

Do not keep changes simply because they make the code more sophisticated.

---

# FINAL ACCEPTANCE CRITERIA

The new system is considered improved ONLY if it demonstrates one or more of:

1. Better out-of-sample log loss.
2. Better Brier score.
3. Better calibration.
4. Positive and persistent CLV.
5. Better risk-adjusted ROI.
6. Better stability across time/markets/leagues.

A higher backtest ROI alone is NOT sufficient.

---

# FINAL REPORT

Produce:

## A. Bugs fixed

## B. Statistical defects fixed

## C. Architecture changes

## D. Data sources added

## E. Features removed

## F. Features added

## G. Model changes

## H. Baseline vs final metrics

Include:

* Log Loss
* Brier
* calibration
* CLV
* ROI
* max drawdown
* sample size

## I. Market-specific results

## J. League-specific results

## K. Remaining weaknesses

## L. Changes reverted because they failed

## M. Recommended production configuration

Finally answer:

> Does the evidence now justify allowing the model to influence bookmaker probabilities?

Answer ONLY:

YES

NO

or

INSUFFICIENT EVIDENCE

and explain why.

Do not use optimism as a substitute for statistical evidence.