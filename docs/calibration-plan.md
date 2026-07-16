# Probability Calibration Layer — staged for after the WC final (2026-07-19)

**Status: STAGED — do not implement before the World Cup final has settled.**
The pipeline is currently winning (62%+ since 2026-07-08) and this change
reshapes every probability, EV, threshold, and Kelly stake — it needs a
backtest, not a mid-tournament hot-swap.

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
