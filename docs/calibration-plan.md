# Probability Calibration Layer

**Status: EVALUATED 2026-07-21 — built, backtested, NOT ENABLED (acceptance
criteria failed in the right direction).**

The time-split backtest (fit on 819 picks ≤ 06-30, evaluated on 91 picks
07-01+) showed:

- **Eval Brier got WORSE with calibration** (0.240 raw → 0.267 calibrated).
- **The raw July predictions are already calibrated**: eval buckets
  pred 51% → actual 50% (−1pp), 59% → 62% (+3pp), 67% → 70% (+3pp).
- The June-era overconfidence this plan targeted was removed by the process
  changes shipped in the meantime (bookmaker blend 0.60, market-blend forced
  picks, blend/EV floors, market-anchored Claude review). Applying the
  June-fitted map to July over-corrects (+12pp underconfident) and would gut
  pick volume.

**What shipped instead:** `src/models/probability_calibration.py`
(ProbabilityCalibrator — isotonic per market family, persisted to
`data/models/probability_calibration.json`), applied in `EnsemblePredictor`
behind `models.probability_calibration_enabled` (**default false**), refit
after every settle in `learn_from_settled()` with a logged 30-day drift check.
**If the drift check warns (actual − predicted < −5pp over 30d), flip the flag
to true** — the freshly-fitted map activates without code changes.

---

*Original design below, kept for the day the flag is needed.*

## Why (the evidence, from 871 settled picks as of 2026-07-16)

Every predicted-probability bucket is systematically overconfident:

| avg predicted | n | actual win rate | gap |
|---|---|---|---|
| 0.47 | 142 | 0.32 | −15pp |
| 0.55 | 327 | 0.49 | −6pp |
| 0.64 | 239 | 0.54 | −10pp |
| 0.75 | 132 | 0.61 | −14pp |
| 0.83 | 23  | 0.74 | −9pp |

Consequences: "value" EV is fictional (the +EV gate passes losing bets), Kelly
stakes are oversized on the worst picks (France–Spain @ 4.0% stake, lost), and
the confidence floor filters on a meaningless number.

## What

A monotonic calibration mapping `p_cal = f(p_raw)` fitted on settled picks and
applied to ensemble probabilities **after** the bookmaker blend, **before** EV /
confidence gating and Kelly sizing.

- **Method:** isotonic regression on (predicted_probability, won) pairs from
  `saved_picks` (fallback: Platt / simple shrink `p_cal = a·p_raw + b` if the
  isotonic fit is too jumpy at n<1000). Fit per market family if data allows
  (1X2 vs goals vs BTTS — BTTS has the worst miscalibration), global otherwise.
- **Persistence:** `data/models/probability_calibration.json`, refit weekly in
  `learn_from_settled()` (same pattern as `calibration.json` / Bayesian weights).
- **Application point:** `EnsemblePredictor.predict()` output, one place — so
  value gating, forced-pick blending, the review menu, and Kelly all see
  calibrated numbers automatically.

## Interactions to re-tune (the reason this waits)

1. `betting.min_confidence` (0.55) and `min_expected_value` (0.05) were tuned
   against INFLATED probabilities — calibrated probs will pass fewer picks;
   thresholds likely need lowering (e.g. conf 0.52, EV 0.03) or repicking from
   the backtest.
2. Forced-pick blend (50% model + 50% market) — with a calibrated model the
   blend weight may deserve to move toward the model.
3. `wc_mismatch_fav_prob` (0.65) and the club blend floor (0.55) compare against
   model numbers — recheck against calibrated scale.
4. Kelly: calibrated probs shrink stakes on exactly the picks that lost big.

## Acceptance (backtest before enabling)

- Run `--backtest-rolling` (rolling-origin) with and without calibration on the
  full settled history: require Brier score improvement AND no win-rate drop on
  the picks the calibrated pipeline would still have made.
- Bucket table above recomputed on calibrated probs: every bucket gap within
  ±3pp.
- Paper-run one week (log calibrated picks alongside live ones) before flipping.

## Out of scope

Claude review prompts (already market-anchored), settlement, scrapers.
