# Stage 5 — Production Hardening, Paper Trading & Prospective CLV Experiment

Stage 4 is complete.

Do NOT perform another model optimization.

Do NOT add new ML algorithms.

Do NOT add large numbers of new features.

Do NOT tune betting thresholds to improve historical ROI.

The current evidence is:

* 457 tests passing
* 0 regressions
* historical bookmaker corruption was substantially worse than initially believed
* seven bookmakers and 13,274 rows were affected
* Pinnacle was also partially contaminated
* the corruption mechanism is now structurally prevented by `market_spec.py`
* clean EV analysis still shows severe miscalibration:
  model overstates true rate by approximately +8.60pp
* increasing EV threshold makes holdout ROI worse
* Elo and Poisson are significantly worse standalone than the clean bookmaker market
* no tested model blend has demonstrated a statistically significant improvement over bookmaker probabilities
* genuine CLV infrastructure exists but currently has 0% data coverage
* production database migration 003 has not yet been deliberately applied
* current correct production decision is PAPER TRADING ONLY

The purpose of Stage 5 is therefore:

> Build a trustworthy prospective experiment that can finally determine whether the system has genuine predictive/value information beyond the bookmaker market.

---

# PHASE 1 — DO NOT TOUCH THE MODEL

Freeze the current model architecture.

Do NOT:

* retune Elo
* retune Poisson
* add ML algorithms
* add new feature families
* optimize EV thresholds
* optimize confidence thresholds
* increase bookmaker/model blend based on backtest noise

The model is now an experimental subject.

We need to observe it prospectively.

Create/document a model version identifier so every future prediction can be associated with the exact model configuration that generated it.

Example:

`model_version = stage5_baseline_YYYYMMDD`

Every prediction must retain this version.

---

# PHASE 2 — PRODUCTION DATABASE MIGRATION

Inspect migration 003 carefully.

Determine exactly which columns/tables/indexes it creates.

Before applying anything:

1. Verify migration is idempotent.
2. Verify nullable changes cannot break existing rows.
3. Verify existing data is preserved.
4. Verify no destructive operation exists.
5. Verify indexes are appropriate.
6. Verify the migration is compatible with the existing Supabase/Postgres environment.

Do NOT rely on:

`create_tables()`

auto-adding production columns.

The migration should be applied deliberately through the project's normal migration mechanism.

After migration:

verify production schema.

Specifically verify:

* `closing_odds`
* closing timestamp
* any CLV fields
* model version
* required prediction metadata

Do NOT alter unrelated production schema.

---

# PHASE 3 — DATABASE SAFETY

Before enabling paper trading:

verify that the application can:

1. create prediction
2. store odds
3. store prediction timestamp
4. capture closing odds
5. settle prediction
6. calculate result
7. calculate CLV
8. update statistics

without errors.

Run this against a safe test path where possible.

Do NOT perform destructive production tests.

---

# PHASE 4 — PAPER TRADING MODE

Set:

`paper_trading_mode = true`

The system must continue generating complete predictions.

The difference is:

NO REAL BETTING ACTION.

Every hypothetical bet must be stored exactly as if it were a real candidate.

Store:

* fixture
* league
* kickoff
* market
* selection
* bookmaker
* opening/current odds
* prediction timestamp
* model probability
* bookmaker probability
* predicted edge
* predicted EV
* confidence
* final recommendation
* model version
* Claude decision if applicable

The paper-trading record must be immutable after prediction except for explicitly defined fields such as closing odds and settlement result.

---

# PHASE 5 — CLOSING LINE CAPTURE

This is the highest priority feature.

Inspect:

`capture_closing_lines.py`

Make it production-safe.

Requirements:

### Timing

Capture closing/current odds approximately:

60–90 minutes before kickoff.

Make the window configurable.

Do not assume that one exact minute is always available.

### Idempotency

Running the script twice must NOT create duplicate closing records.

### Retry

Transient API failures should retry safely.

### Missing odds

If closing odds cannot be captured:

record:

`closing_capture_status = missing`

Do not invent a closing price.

### Late capture

If the capture happens after kickoff:

mark it invalid.

Do not use it for CLV.

### Wrong market

Never match:

Home/Away

with

1X2.

Use the centralized `market_spec`.

### Wrong selection

Verify selection identity before attaching closing odds.

---

# PHASE 6 — ODDS SOURCE EFFICIENCY

The system previously had Supabase egress problems.

Therefore inspect the closing-line workflow specifically for unnecessary database and API traffic.

Requirements:

* fetch only required fixtures
* fetch only required markets
* avoid repeated queries
* batch where appropriate
* cache where possible
* never repeatedly download identical odds
* avoid `SELECT *`
* avoid N+1 queries

Measure:

database requests

API requests

rows transferred

approximate egress

per closing capture run.

Do not solve CLV by creating a new egress problem.

---

# PHASE 7 — CLOSING LINE DATA MODEL

Audit the schema.

For every closing-line record we need enough information to reconstruct exactly what happened.

At minimum:

* fixture_id
* market_type
* selection
* bookmaker
* odds
* captured_at
* kickoff_at
* source
* validity/status

If model/pick association is needed, preserve it without duplicating unnecessary data.

Keep raw odds.

Do not store only derived CLV.

The raw closing price must remain available for future recalculation.

---

# PHASE 8 — CLV IMPLEMENTATION AUDIT

Review the existing `clv.py`.

Verify mathematically that:

CLV uses:

PICK PRICE

versus

CLOSING PRICE

and does NOT use:

model probability

as a substitute for the closing market.

Maintain the existing regression test:

`test_clv_is_not_model_edge`

Add additional tests for:

* favorable movement
* unfavorable movement
* unchanged odds
* missing close
* invalid timestamp
* bookmaker mismatch
* market mismatch
* selection mismatch
* different decimal odds
* multiple bookmakers

Document the exact CLV formula.

---

# PHASE 9 — CLV NORMALIZATION

Determine whether CLV should be measured using:

1. raw decimal price movement
2. implied probability movement
3. de-vig probability movement

Do not assume one is universally correct.

For the primary production metric choose the most statistically appropriate approach for each market type.

Keep secondary raw metrics for diagnostics.

Document:

why the primary metric was selected.

---

# PHASE 10 — BOOKMAKER CONSISTENCY

Because bookmaker corruption was severe, apply the Stage 4 market validation rules to closing odds too.

A closing price should only be accepted if:

* market structure is valid
* odds are valid
* timestamp is valid
* bookmaker/market mapping is valid
* overround is plausible for the market
* source data is not obviously corrupted

Do NOT simply discard entire bookmakers.

Validation must be per:

`(match, bookmaker, market)`

as established in Stage 4.

---

# PHASE 11 — PAPER-TRADING DASHBOARD / REPORT

Create a useful report for prospective observations.

At minimum show:

### Volume

* total paper picks
* picks per market
* picks per league
* picks per bookmaker

### Pricing

* average odds
* median odds
* average predicted probability
* average market probability
* average predicted EV

### CLV

* CLV coverage %
* average CLV
* median CLV
* positive CLV %
* CLV by market
* CLV by odds bucket

### Outcomes

* ROI
* win rate
* Brier
* log loss

Do not call any metric "CLV" unless it uses genuine closing prices.

---

# PHASE 12 — CRITICAL DATA SEPARATION

Maintain strict separation between:

### Prediction-time information

and

### Post-prediction information.

At prediction time the system may use ONLY information available before the prediction timestamp.

Closing odds must NEVER enter:

* model features
* model probability
* prediction
* Claude decision

They are evaluation data only.

Likewise:

match result

must never enter the prediction.

---

# PHASE 13 — CLAUDE REVIEW ISOLATION

Keep Claude review in the experiment, but isolate it.

Record:

`pre_claude_probability`

`pre_claude_selection`

`pre_claude_ev`

`claude_changed_decision`

`post_claude_selection`

`post_claude_reason`

Do not overwrite the original model prediction.

We need to answer later:

Does Claude improve decisions?

or

Does Claude merely move them toward the market?

Do NOT conclude either yet.

---

# PHASE 14 — MODEL VS MARKET EXPERIMENT

Every paper pick must allow the following counterfactual comparison:

### Market-only

What would happen if we used the bookmaker market directly?

### Model-only

What would happen using the model probability?

### Current production blend

What happens with the current blend?

### Selected bet

What did the system actually choose?

This allows us to measure incremental value without needing to rerun historical predictions.

---

# PHASE 15 — EV CALIBRATION TRACKING

Because Stage 4 found:

model overstates true rate by ~8.6pp

track calibration prospectively.

Create buckets such as:

0–5% predicted edge

5–10%

10–15%

15–20%

20%+

For each bucket record:

* number of predictions
* average predicted probability
* realized probability
* Brier
* ROI
* CLV

Do not optimize thresholds yet.

We are measuring calibration.

---

# PHASE 16 — SAMPLE SIZE

Do not make production model decisions from small samples.

Minimum evaluation target:

### First checkpoint

100 valid closing-line picks.

Purpose:

data quality only.

No model decision.

### Second checkpoint

200 valid closing-line picks.

Purpose:

initial CLV signal.

### Third checkpoint

500+ valid closing-line picks.

Purpose:

meaningful model-vs-market evaluation.

At each checkpoint report uncertainty.

---

# PHASE 17 — STATISTICAL TESTS

At each checkpoint calculate:

* log loss difference vs market
* Brier difference vs market
* CLV confidence interval
* ROI confidence interval
* bootstrap where appropriate
* permutation test where appropriate

Do NOT use:

"ROI is positive"

as proof of edge.

Do NOT use:

"CLV is positive"

from a tiny sample as proof either.

---

# PHASE 18 — MARKET-SPECIFIC ANALYSIS

Do NOT pool all markets blindly.

Analyze separately:

1X2

O/U 2.5

BTTS

other active markets

For each report:

sample

log loss

Brier

CLV

ROI

calibration

Only pool markets when statistically justified.

---

# PHASE 19 — LEAGUE ANALYSIS

Do NOT optimize leagues yet.

Just collect data.

Record:

league

country

market

model

bookmaker

CLV

result

After sufficient observations, determine whether some leagues consistently show useful signal.

Do not draw conclusions from tiny league samples.

---

# PHASE 20 — DATA QUALITY MONITORING

Create explicit health checks.

Alert if:

* closing capture coverage < 80%
* duplicate closing records appear
* invalid timestamps appear
* invalid market mappings appear
* bookmaker corruption exceeds expected levels
* API failures increase
* Supabase egress increases unexpectedly
* prediction records are missing closing associations
* settlement failures occur

Do not silently continue with corrupted evaluation data.

---

# PHASE 21 — FREE ODDS API QUOTA

Inspect the existing Odds API integration.

The audit previously indicated approximately 468/500 requests remained.

Do not waste quota.

Determine exactly:

* requests per day
* fixtures per request
* markets per request
* batching capability
* caching
* retry behaviour

Calculate expected monthly usage.

The goal is to capture closing lines without exceeding the free quota.

If the current design cannot achieve sufficient coverage under the free quota:

report the limitation.

Do NOT introduce a paid dependency.

---

# PHASE 22 — SCHEDULING

Determine the existing scheduler:

GitHub Actions / cron / CI / application scheduler.

Add:

`capture_closing_lines`

at the appropriate point before kickoff.

Avoid creating a separate workflow for every match.

Prefer batched execution.

Make the timing configurable.

Document timezone handling explicitly.

Bulgaria may switch between EET/EEST, so do not hard-code UTC offsets.

---

# PHASE 23 — PRODUCTION SAFETY CHECK

Before recommending activation, run a final checklist:

[ ] migration applied

[ ] schema verified

[ ] paper trading enabled

[ ] prediction records complete

[ ] closing capture scheduled

[ ] closing capture tested

[ ] no duplicate closing records

[ ] CLV formula tested

[ ] market validation active

[ ] no Home/Away → 1X2 collision possible

[ ] no post-prediction leakage

[ ] Supabase egress acceptable

[ ] Odds API quota sustainable

[ ] settlement works

[ ] existing 457+ tests pass

---

# PHASE 24 — DO NOT COMMIT OR DEPLOY

Do NOT commit.

Do NOT deploy.

Do NOT apply production migration automatically.

Prepare everything and report the exact commands/actions that I need to execute manually.

For production migration provide:

1. migration file
2. exact command
3. expected result
4. verification query
5. rollback consideration

Do not execute destructive production operations yourself.

---

# PHASE 25 — FINAL REPORT

Create:

`docs/stage5-paper-trading-YYYY-MM-DD.md`

Include:

## 1. Objective

## 2. Changes made

## 3. Database migration

## 4. Paper trading architecture

## 5. Closing line capture

## 6. CLV calculation

## 7. Odds API usage

## 8. Supabase usage impact

## 9. Data quality safeguards

## 10. Tests

## 11. Exact manual production steps

## 12. First 100-pick evaluation plan

## 13. 200-pick evaluation plan

## 14. 500-pick evaluation plan

## 15. What we intentionally did NOT change

---

# FINAL DECISION

At the end answer:

### Is the system ready for real-money betting?

ONLY:

YES

or

NO

Given the current evidence, the expected answer is:

NO.

The system should remain:

PAPER TRADING ONLY

until genuine prospective CLV and model-vs-market evidence has been collected.

---

# MOST IMPORTANT RULE

Do not try to make the numbers look better.

The purpose of Stage 5 is to collect clean evidence.

If the evidence eventually shows that:

MARKET ONLY > MODEL

then recommend market-only.

If:

MARKET + MODEL > MARKET

with positive CLV and robust out-of-sample evidence, then recommend the model.

If the evidence is inconclusive:

keep paper trading.

A simpler profitable system is preferable to a sophisticated unproven one.
