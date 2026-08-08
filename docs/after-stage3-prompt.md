# Stage 4 — Production Hardening, Clean Market Data & Genuine CLV Validation

You have completed Stage 3.

Do NOT perform another broad model optimization.

The next objective is to make the system statistically trustworthy and production-ready after the discovery of the bookmaker market corruption bug.

The most important discovery was:

`API-Football "Home/Away"` was incorrectly mapped to `1X2`.

This caused two-way draw-excluded prices to overwrite genuine 1X2 bookmaker prices.

All previous experiments involving bookmaker probabilities must therefore be treated as potentially contaminated.

We now need a clean baseline.

---

# CURRENT STATE

Stage 3 established:

* 422 tests passing.
* Consensus de-vigged bookmaker probability improved log-loss from 1.0034 → 0.9931.
* 0.80 bookmaker blend was accepted relative to 0.60.
* Poisson 540d / rho=0 was accepted standalone.
* All six outcome-fitted betting gates failed holdout validation.
* `ml_retrain_days` was already correctly respected.
* Understat integration is currently NOT verified and must not be assumed.
* Genuine CLV infrastructure has been added but does not yet contain sufficient closing-line data.
* Current final conclusion is:

INSUFFICIENT EVIDENCE.

Do NOT turn this into a claim of profitability.

---

# PHASE 1 — COMPLETELY VALIDATE THE MARKET CORRUPTION FIX

Audit the entire bookmaker ingestion pipeline.

The fix must guarantee that:

`Home/Away`

and

`1X2`

are separate market types everywhere.

Search the entire repository for:

* BET_TYPE_MAP
* Home/Away
* 1X2
* home/draw/away
* bookmaker probabilities
* implied probabilities
* odds normalization
* odds persistence
* consensus calculations

Verify that no two-way market can overwrite a three-way market.

Add explicit tests:

### Test 1

Home/Away must never populate 1X2.

### Test 2

1X2 must always contain three outcomes.

### Test 3

A bookmaker may offer both markets simultaneously.

Both must survive independently.

### Test 4

Overround for a genuine 1X2 market should be mathematically plausible.

### Test 5

Two-way market probabilities must sum approximately to 1 after de-vig.

### Test 6

Three-way probabilities must sum approximately to 1 after de-vig.

### Test 7

Duplicate bookmaker rows cannot silently overwrite another market.

---

# PHASE 2 — HISTORICAL DATA INTEGRITY AUDIT

Because the previous data was contaminated, determine exactly which historical records are affected.

Quantify:

* number of corrupted rows
* affected bookmakers
* affected matches
* affected dates
* affected markets
* affected features
* affected predictions
* affected ML training records

Create a report:

| Dataset | Total | Potentially corrupted | Safe |
| ------- | ----: | --------------------: | ---: |

Do NOT silently delete data.

Determine whether historical odds can be reconstructed from the source.

If reconstruction is possible:

rebuild them.

If reconstruction is impossible:

mark them as contaminated and exclude them from clean model evaluation.

Do not mix contaminated and clean data.

---

# PHASE 3 — CREATE A CLEAN EVALUATION DATASET

Create an immutable concept of:

`clean_evaluation_dataset`

It must contain only data that:

* has correct market mapping
* has valid odds
* has correct timestamps
* has no post-match information
* has known prediction timestamp
* has known market type
* has valid outcome
* has sufficient bookmaker coverage

Document exactly how a record qualifies.

Every future model experiment must use this dataset.

---

# PHASE 4 — REBUILD THE BASELINE

Do NOT use previous model results as the baseline.

Re-run the key experiments on clean data.

At minimum compare:

### Baseline A

Raw bookmaker probability.

### Baseline B

De-vigged bookmaker consensus.

### Baseline C

Elo.

### Baseline D

Poisson.

### Baseline E

Current ML model.

### Baseline F

Market + Elo.

### Baseline G

Market + Poisson.

### Baseline H

Market + Elo + Poisson.

### Baseline I

Current production ensemble.

Use exactly the same matches for every comparison where possible.

Report:

* sample size
* log loss
* Brier
* calibration
* ROI
* CLV if available

Do not compare models using different samples without clearly explaining why.

---

# PHASE 5 — VERIFY BOOKMAKER CONSENSUS

The audit showed:

Consensus de-vig:

1.0034 → 0.9931

This is currently the strongest evidence.

Now investigate why.

Measure bookmaker agreement.

For every match calculate:

* number of bookmakers
* median probability
* mean probability
* dispersion
* min/max
* overround
* disagreement

Test whether consensus quality improves with:

2 bookmakers

3 bookmakers

4 bookmakers

5+ bookmakers

Do NOT automatically require many bookmakers if doing so destroys coverage.

Determine the optimal minimum coverage threshold using walk-forward validation.

---

# PHASE 6 — BOOKMAKER BLEND

Current evidence:

0.60 was worse.

0.80 is better than 0.60.

Pure market still has the best point estimate:

market = 0.9906

blend 0.80 = 0.9926

Therefore:

DO NOT increase or decrease the blend merely because of theory.

Test only:

0.80

0.90

1.00

on the clean evaluation dataset.

If pure market remains best:

recommend:

`bookmaker_blend_weight = 1.0`

for that market.

That is an acceptable outcome.

The model must earn the right to influence the market probability.

---

# PHASE 7 — SEPARATE MARKETS

Do NOT assume one optimal architecture exists for every market.

Evaluate independently:

* 1X2
* Over 2.5
* Under 2.5
* BTTS
* any other active market

For each determine:

best market baseline

best model

best blend

best calibration

best EV behaviour

best CLV behaviour

If the model adds value only to one market:

keep it there.

Do not force a global ensemble.

---

# PHASE 8 — IMPLEMENT PROPER CLV CAPTURE

This is now the most important remaining measurement system.

The application must record:

### At prediction time

* fixture_id
* market
* selection
* bookmaker
* odds
* timestamp
* model probability
* market probability
* selected price

### At closing time

* same bookmaker
* same market
* same selection
* closing odds
* closing timestamp

Do NOT substitute:

`model probability - 1/odds`

for CLV.

That is not CLV.

---

# PHASE 9 — CLOSING LINE SCHEDULER

Inspect:

`capture_closing_lines.py`

Determine:

* how it is triggered
* how often it runs
* whether it runs before kickoff
* whether odds are refreshed sufficiently close to kickoff
* whether it can miss fixtures
* whether it retries
* whether duplicate captures are prevented

Target a configurable capture window.

For example:

60–90 minutes before kickoff.

Do not hard-code this if the existing architecture supports configuration.

---

# PHASE 10 — CLV FORMULA

Implement a documented CLV calculation.

Do NOT rely on a simplistic formula without considering market type.

For a selected price:

pick_odds

and

closing_odds

calculate a consistent price-based CLV metric.

Also store:

* absolute odds movement
* implied probability movement
* normalized/de-vig movement where possible

Keep the raw prices so the metric can be recalculated later.

---

# PHASE 11 — CLV DATA QUALITY

Create monitoring for:

* missing closing odds
* stale closing odds
* closing odds after kickoff
* wrong market
* wrong selection
* bookmaker mismatch
* duplicate closing rows

Never calculate CLV from invalid pairs.

Expose:

`clv_coverage_rate`

Example:

1,000 picks

850 valid closing lines

CLV coverage = 85%.

---

# PHASE 12 — PAPER-TRADING MODE

Until sufficient CLV data exists:

DO NOT use CLV to change production weights.

Create or verify a paper-trading mode.

Record what the system WOULD have selected.

Do not require real bets.

Collect:

* prediction
* odds
* model probability
* market probability
* closing odds
* outcome

This gives us clean prospective data.

---

# PHASE 13 — MINIMUM SAMPLE POLICY

Do not make a model decision after 10 or 20 picks.

For CLV evaluation:

minimum target:

~200 valid picks.

Prefer:

300–500+

before making major production changes.

Evaluate:

overall

1X2

O/U

BTTS

and individual leagues only when sample sizes justify it.

---

# PHASE 14 — RECHECK EV

After fixing the bookmaker corruption:

re-run the EV analysis.

The previous result:

Q3 EV +11.6% → ROI -19.3%

may have been caused partly by corrupted bookmaker probabilities.

Do NOT assume that finding remains valid.

Recalculate:

claimed EV quintiles

vs

realized ROI

on clean data.

Also calculate:

claimed edge

vs

CLV.

If high predicted EV does not correspond to positive CLV, investigate before changing thresholds.

---

# PHASE 15 — EV THRESHOLDS

Do NOT decide yet whether:

`min_expected_value = 0.05`

should be removed.

First establish:

1. clean market probabilities
2. correct EV
3. calibration
4. CLV

Then test:

EV threshold = 0%

2%

5%

7.5%

10%

But use walk-forward validation.

If the market itself is approximately efficient, a positive historical EV threshold may simply select noisy model errors.

---

# PHASE 16 — CONFIDENCE THRESHOLD

Same principle for:

`min_confidence = 0.55`

Do not optimize it yet.

Determine whether confidence correlates with:

* log loss
* Brier
* CLV
* realized ROI

If confidence does not predict quality, remove it from decision logic.

---

# PHASE 17 — REMOVE FALSE CLV

Search the repository for:

`avg_clv`

and all references to CLV.

Ensure no internal metric is still incorrectly labelled CLV.

Rename any metric that is actually:

model edge

predicted edge

EV

or

probability divergence.

This is important because dashboards and Claude prompts must not receive misleading metrics.

---

# PHASE 18 — CLAUDE REVIEW

Keep Claude review isolated.

For every prediction record:

BEFORE_CLAUDE

AFTER_CLAUDE

and whether Claude changed:

* selection
* odds
* confidence
* EV
* final recommendation

Do not let Claude override the mathematical probability silently.

Claude may provide contextual evidence, but the pipeline must preserve the original quantitative prediction.

Eventually compare:

MODEL ONLY

vs

MODEL + CLAUDE

using prospective data.

---

# PHASE 19 — DATA SOURCE INTEGRITY

Do NOT integrate Understat yet.

The previous investigation showed that the expected scraper payload is currently unavailable.

Before adding any new data source:

prove:

* access works
* historical data exists
* coverage exists
* timestamps exist
* data can be mapped to fixtures
* it can be collected reliably

Only then integrate.

Do not build architecture around an unverified source.

---

# PHASE 20 — ELO / POISSON

Do not perform another major parameter search yet.

Keep the Stage 3 accepted changes provisionally:

Poisson:

540-day half-life

rho = 0

But verify them again on the CLEAN dataset.

Elo:

retain current configuration unless clean data demonstrates a problem.

If the clean dataset changes the ranking:

document it before modifying anything.

---

# PHASE 21 — PRODUCTION CONFIGURATION

At the end, recommend a production configuration based ONLY on clean evidence.

Potential outcomes include:

### Configuration A

Market-only.

### Configuration B

Market + Elo.

### Configuration C

Market + Poisson.

### Configuration D

Market + model ensemble.

Do NOT assume C or D is better because it is more sophisticated.

---

# PHASE 22 — DO NOT OVER-ENGINEER

Do NOT:

* add new ML algorithms
* add deep learning
* add neural networks
* add dozens of features
* optimize hundreds of parameters
* scrape unreliable websites
* add paid data sources
* increase API usage unnecessarily

until the clean market + CLV baseline is established.

The current priority is measurement quality, not model complexity.

---

# PHASE 23 — TESTING

Maintain all existing tests.

Add tests for:

* market separation
* historical data integrity
* clean dataset filtering
* bookmaker consensus
* de-vig calculation
* CLV calculation
* CLV timestamp validation
* duplicate closing line prevention
* EV calculation
* market-specific Bayesian weights
* Claude pre/post state
* paper trading

Run the complete suite.

Target:

0 regressions.

---

# PHASE 24 — FINAL REPORT

Produce:

## 1. Historical corruption

Exactly how many records were affected.

## 2. Clean dataset

How many records remain.

## 3. Clean baseline

Market vs model.

## 4. Market performance

By market.

## 5. Model contribution

Incremental improvement over market.

## 6. CLV infrastructure

Coverage and correctness.

## 7. EV validation

Clean EV vs realized performance.

## 8. Production configuration

Exact recommended values.

## 9. Changes made

Every file and important function.

## 10. Tests

Total tests and new tests.

## 11. Remaining uncertainty

What we still do not know.

---

# FINAL DECISION

At the end answer:

### A. Does the model currently demonstrate a statistically reliable edge over the market?

YES / NO / INSUFFICIENT EVIDENCE

### B. Does the model improve bookmaker probabilities?

YES / NO / INSUFFICIENT EVIDENCE

### C. Is genuine CLV now being measured?

YES / NO

### D. Is the EV calculation trustworthy?

YES / NO / INSUFFICIENT EVIDENCE

### E. What should production use today?

Choose exactly one:

MARKET ONLY

MARKET + MODEL

MODEL ONLY

PAPER TRADING ONLY

---

# IMPORTANT

Do NOT commit changes.

Do NOT modify production configuration merely to increase apparent ROI.

Do NOT declare profitability.

Do NOT optimize again until the clean dataset and CLV pipeline have sufficient prospective observations.

The goal of Stage 4 is to make the system trustworthy enough that the next 200–500 predictions can finally answer whether it has genuine predictive value.
