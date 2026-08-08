# Full Predictive Performance Audit — 2026-08-07

Executed against `docs/reviewing-optimazation-prompt.md`. **Stage 1 only: no code was
changed.** Every number below is measured, not asserted — from the production Supabase
database (read-only) and from walk-forward backtests written for this audit.

Evidence scripts (scratchpad, not committed): `audit_perf.py`, `audit_perf2.py`,
`audit_perf3.py`, `backtest_poisson.py`, `backtest_blend.py`.

Data used: 1,018 settled picks (2026-02-28 → 2026-08-05, all settled), 38,517 matches,
317,657 odds rows.

---

## 1. Executive summary

**The system is well engineered and statistically inert.**

The pipeline is genuinely impressive as software: sharding, egress control, HMAC-signed
model artifacts, a completeness-checked preload cache, concurrency-safe settlement,
idempotent picks. That work is sound and should be preserved.

The predictive core does not work, and this is now measurable three independent ways:

| Test | Result |
|---|---|
| **Pick level** — logistic regression, outcome ~ market + (model − market) | model's disagreement with the market: coef +0.245, z = 1.11, **p = 0.265**. LR test p = 0.265 |
| **Scoring rules** on the same 1,018 picks | model Brier **0.2499** vs constant-0.514 Brier **0.2498** vs raw `1/odds` Brier **0.2410** |
| **Match level** — 1,710 out-of-sample matches, blend sweep | log-loss falls **monotonically** as bookmaker weight rises: 1.0412 at w=0 → **0.9929 at w=1.0**. No interior optimum |

The model's probabilities are **indistinguishable from a constant**, and **worse than the
raw vigged bookmaker price** it is being blended with. Realised flat ROI is −3.6% on 1,018
bets — approximately the bookmaker margin. That is the exact signature of a bettor with
zero edge paying the vig.

Critically, the model's declared edge is not merely absent — it is **anti-informative in
the middle of its own range**:

| Claimed-EV quintile | avg claimed EV | realised flat ROI |
|---|---|---|
| Q1 | −4.8% | **+2.4%** |
| Q2 | +6.4% | −8.4% |
| Q3 | +11.6% | **−19.3%** |
| Q4 | +18.1% | −6.8% |
| Q5 | +38.0% | +14.1% (8 longshot bets carry it) |

The picks the model rates as *least* valuable are the ones that win. `min_expected_value:
0.05` is therefore an adverse-selection filter, not a value filter.

Second-order finding, equally important: **nothing in the settled record is statistically
significant.** Every segment — market, league, model-agreement level, Claude KEEP/CHANGE —
has p > 0.15 against its break-even rate. A permutation test on league ROI spread gives
**p = 0.407**: league differences are indistinguishable from noise. Yet the codebase
contains roughly fifteen hard-coded thresholds and exclusions, each justified by a comment
citing a small settled sample (`over_3.5` "proven loser, 32 picks"; `btts_yes` banned for
club forced picks on 32 picks; `club_pick_min_ev: -0.05` from 41 picks). These are
noise-fitted parameters selected on the same data used to evaluate them. This violates the
prompt's Rules 3, 4 and 10 and is the second-largest source of harm after the fictional EV.

What genuinely works: the ingestion layer, the settlement grader, the egress
optimisations, the concurrency safety, and the Elo model (which quietly outperforms
Poisson and is under-weighted).

---

## 2. Architecture assessment — **7 / 10**

Strong: separation of concerns, async fan-out with bounded concurrency, the preload cache
with a *provable* completeness rule (`src/features/preload_cache.py` is the best-designed
file in the repo), DB-level dedup constraint on `saved_picks`, batched settlement with a
correction window, sharding keyed on `match_id` so per-match invariants never straddle
shards.

Weak:
- `src/agent/betting_agent.py` is **4,808 lines** and owns ingestion orchestration, pick
  generation, portfolio management, settlement, tuning, training, calibration and CLI. It
  is the single biggest maintenance risk.
- Learned state (`data/models/`) is not version-controlled and survives only via a GitHub
  Actions cache with a `restore-keys` prefix match. Cache eviction silently degrades the
  system to Poisson+Elo with no alert.
- Local artifacts are already stale: `ml_models.pkl` and `goals_model.pkl` last written
  **2026-03-31**, `bayesian_weights.json` **2026-03-30**.
- `data/models/feature_list.json` is absent, so `MLModels.predict()` logs a warning on
  every single prediction.

---

## 3. Data quality assessment — **3 / 10**

This is the finding that most surprised me. Coverage across 38,360 completed matches:

| Field | Rows populated | Coverage |
|---|---|---|
| goals | 38,360 | 100% |
| **xG** | **735** | **1.9%** |
| **shots / possession / corners / dangerous attacks** | **933** | **2.4%** |
| referee | 7,591 | 19.8% |
| venue (needed for weather) | 3,088 | 8.0% |
| **injuries (entire table)** | **17 rows** | ≈0% |

xG by year: 2022 **0**, 2023 **0**, 2024 **0**, 2025 **25**, 2026 **710**.

Direct consequences, all verified in code:

1. `models.poisson_use_xg: true` with `poisson_xg_min_coverage: 0.35` — actual coverage is
   1.9%, so `use_xg_global` is **never** true. The xG-Poisson feature has never once
   activated in production.
2. `MLModels.fit()` prunes features with >80% zeros. xG features, all extended-stat
   features (shots, SoT, possession, corners, dangerous attacks, saves, offsides, free
   kicks), and all injury features are ≈98% zero → **pruned every training run**.
3. The `xg_regression_penalty` in `value_calculator.find_value_bets()` reads
   `home_xg_overperformance`, which is 0.0 for ~98% of teams. It effectively never fires.
4. `FEATURES.md` and the README describe 14 feature families. In practice **six** carry
   real data: team form (goals only), H2H, Poisson strengths, Elo, bookmaker implied
   probabilities, and situational rest/fatigue.

---

## 4. Feature quality

### Top features that actually carry signal

Ranked by the evidence available (the ML model's own importance list is unavailable —
`feature_list.json` is missing — so this is ranked by measured predictive contribution and
data availability):

1. `home_implied_prob` / `draw_implied_prob` / `away_implied_prob` — the only features
   proven to beat a constant
2. `over25_implied_prob`, `under25_implied_prob`
3. `btts_yes_implied_prob`
4. Elo rating difference (Elo beats Poisson head-to-head, §5)
5. Poisson `home_xg` / `away_xg` (from raw goals)
6. `league_avg_goals`, `league_over25_rate`, `league_btts_rate` (real league priors)
7. `league_home_win_rate`, `league_draw_rate`
8. `home_overall_goals_scored_per_match` / conceded, and away equivalents
9. `decay_form_score` (both sides)
10. `quality_adjusted_ppg` — but see the leakage note in §6
11. `h2h_total_meetings`, `h2h_home_win_pct` (only when ≥5 meetings)
12. `home_rest_days` / `away_rest_days` / `rest_days_diff`
13. `home_fatigue_index` / `away_fatigue_index` (congestion is real, well-computed)
14. `home_matches_14d` / `away_matches_14d`
15. `is_international_match`
16. `home_over15_implied_prob` / `away_over15_implied_prob`
17. `position_difference` — see the caveat in §6
18. `home_overall_clean_sheets` / `failed_to_score`
19. `wc_points_diff`, `wc_gd_diff` (tournament group stage only)
20. `home_win_streak` / `losing_streak` (weak, but non-degenerate)

### Weak, redundant, or structurally dead features

| # | Feature family | Verdict |
|---|---|---|
| 1–8 | `home/away_xg_avg`, `xg_against_avg`, `xg_overperformance`, `xg_matches`, `xg_for_diff`, `xg_against_diff` | **Dead** — 1.9% coverage, pruned as sparse |
| 9–16 | `shots_per_game_avg`, `shots_on_target_per_game_avg`, `possession_avg`, `corners_per_game_avg`, `dangerous_attacks_per_game_avg`, `saves_per_game_avg`, `offsides_per_game_avg`, `free_kicks_per_game_avg` (×2 sides) | **Dead** — 2.4% coverage |
| 17–21 | `home_odds_movement`, `away_odds_movement`, `over25_odds_movement`, `max_abs_movement`, `movement_direction` | **Dead** — only 8.4% of odds rows have an opening value that differs from current, so ≈92% zeros → pruned |
| 22–27 | `home_injury_*`, `away_injury_*` | **Structurally excluded** — `create_features(for_training=True)` skips injuries entirely, so they never enter `feature_names`; at predict time `_align_features` therefore drops them. They can never reach the ML model. 17 injury rows exist anyway |
| 28–33 | `weather_temp_c`, `weather_wind_kmh`, `weather_precip_mm`, `weather_is_raining`, `weather_is_windy`, `weather_available` | **Structurally excluded** — same train/serve asymmetry as injuries, plus 8% venue coverage. The Open-Meteo call runs on every live fixture and its output is discarded by the ML path |
| 34–40 | `referee_cards_per_match_avg`, `referee_fouls_per_match_avg`, `referee_over25_rate`, `referee_avg_yellow_cards`, `referee_avg_red_cards`, `referee_matches`, `referee_goals_per_match_avg` | **Mostly dead** — 19.8% coverage; fouls/cards are 2.4% |
| 41–44 | `league_position`, `title_gap`, `relegation_gap`, `in_relegation_zone` | **Wrong, not merely weak** — `_get_league_standings` ranks teams by raw points over their **last 50 league matches regardless of season**. Teams with different match counts are compared directly. This is a rolling-strength ranking mislabelled as a league table; `in_relegation_zone` is meaningless |
| 45–48 | `home_rsi`, `away_rsi`, `home_macd`, `away_macd` | Momentum on a 14-match window of {0,1,3} points. RSI here is a monotone transform of points-per-game — i.e. it duplicates `points_per_match`. Redundant by construction |
| 49–50 | `intl_experience_diff`, `intl_quality_diff` | Non-zero for a small minority of fixtures |

### Window audit (Phase 5D)

Every form window is hard-coded to 10 matches (`create_features` lines 476–482); standings
use 50; momentum uses 14. **No comparison of 3/5/10/20/EWMA was ever run.** The exponential
`decay_form_score` (λ = 0.85) is the only adaptive one. Given that the ML model contributes
nothing measurable at present, tuning windows is not the priority — but the claim that 10
is optimal has no support in the repo.

---

## 5. Model quality

### Walk-forward backtest design

Train on everything before a cutoff, evaluate on the next 60 days. Five cutoffs:
2025-08-01, 2025-11-01, 2026-02-01, 2026-05-01, 2026-07-01. 5,771 out-of-sample matches
(1,710 with bookmaker prices). No parameter was fitted on the evaluation windows.

### Poisson (Phase 7)

Half-life and ρ swept on identical out-of-sample matches, 1X2 log-loss:

| half-life (days) | ρ = 0 | ρ = −0.13 |
|---|---|---|
| 90 | 1.0494 | 1.0524 |
| **180 (shipped)** | 1.0364 | **1.0385** |
| 365 | 1.0322 | 1.0338 |
| 730 | **1.0314** | 1.0328 |
| none | 1.0321 | 1.0335 |

- **180 days is not optimal.** 365–730 days is consistently better; the improvement from
  180 → 730 is 0.007 nats. Small, but monotone and consistent across the sweep. 90 days is
  clearly worse — the model is being starved of data, not kept fresh.
- **ρ = −0.13 makes 1X2 worse at every half-life.** That is expected — Dixon-Coles targets
  the low-score cells — but the corrected matrix is used for 1X2 as well.
- **ρ does not help the market it was designed for either.** Over/Under 2.5 log-loss:
  ρ = −0.13 → 0.7083, ρ = 0 → 0.7084. Indistinguishable.
- Therefore `_estimate_league_rhos()` — a `scipy.optimize.minimize_scalar` over every
  league × every match, calling `poisson.pmf` inside the objective, on every `fit()` — buys
  **nothing measurable** and costs real CI time.
- Shrinkage (`shrinkage_sample_cap: 100`), the unknown-team hash offset, and national-team
  strength separation are all sensible and correctly implemented.

### Elo (Phase 8) — **the underrated component**

Same 5,771 matches:

| model | log-loss | Brier | accuracy |
|---|---|---|---|
| **Elo** | **1.0279** | **0.6162** | **48.7%** |
| Poisson (180, −0.13) | 1.0385 | 0.6229 | 47.6% |
| home-bias prior (0.44/0.26/0.30) | 1.0683 | 0.6458 | 44.8% |
| uniform | 1.0986 | 0.6667 | 44.8% |

Elo beats Poisson on all three metrics, using five columns and 170 lines of code. Yet:
- config prior weights Poisson **0.25** vs Elo **0.20**;
- the Bayesian learner has recorded **872 observations for Poisson and 436 for Elo** — see
  the double-counting bug in §6;
- `_calculate_probabilities` uses a crude linear draw model (`0.28 − |diff|/2000`, floored
  at 0.15) that has never been fitted. Fitting an ordered-logit or a simple
  draw-probability curve on the same data would be a cheap, real gain.
- No league-strength normalisation: a Romanian Liga-1 1600 and a Premier League 1600 are
  treated as equal. With cross-league fixtures (CL/EL/ECL) in the pool, this is a genuine
  bias.

### Machine learning (Phase 9)

Cannot be evaluated on merit, because it is not currently contributing:
- Bayesian weights show **`ml` with n = 0 observations in every league** — pure prior. The
  learner has never received an ML outcome.
- `calibration.json` contains only `poisson` and `elo`.
- `ml_models.pkl` is from **2026-03-31** locally.
- `feature_list.json` is missing.
- The per-class CV diagnostic in `train_ml_models` (line 3295) reads
  `self.predictor.ml_models._models` — the attribute is `models`. It raises `AttributeError`
  every run and is swallowed by the surrounding `except`. That diagnostic has never printed.

The training methodology itself is *good*: chronological 80/20 hold-out taken before any
fitting, `TimeSeriesSplit` inside the training split only, isotonic calibration wrappers on
RF/XGB/LGBM, sparse/zero-variance/correlation pruning, importance pruning with a re-fit.
The problem is upstream: after pruning, the surviving feature set is mostly form + league
priors + bookmaker probabilities, and the bookmaker probabilities are the only ones with
signal — which the ensemble then blends in *again* at 60% (double-counting, §6).

### Ensemble (Phase 10) & bookmaker blend (Phase 11) — **the decisive experiment**

On 1,710 identical out-of-sample matches with real prices, sweeping the blend weight:

| bookmaker weight | 1X2 log-loss | Brier | accuracy |
|---|---|---|---|
| 0.00 (pure Poisson+Elo) | 1.0308 | 0.6186 | 48.8% |
| 0.20 | 1.0164 | 0.6085 | 50.2% |
| 0.40 | 1.0057 | 0.6008 | 52.0% |
| 0.50 | 1.0016 | 0.5980 | 52.3% |
| **0.60 (shipped)** | 0.9983 | 0.5957 | 52.1% |
| 0.80 | 0.9940 | 0.5931 | 52.3% |
| **1.00 (pure market)** | **0.9929** | **0.5930** | **52.8%** |

Over/Under 2.5, same matches:

| bookmaker weight | log-loss |
|---|---|
| 0.00 | 0.7083 |
| 0.40 | 0.6856 |
| **0.60 (shipped)** | 0.6787 |
| 0.75 | 0.6752 |
| **1.00** | **0.6724** |

**Monotone in both markets. There is no interior optimum.** Every unit of model weight
subtracted from the market makes the forecast worse out of sample. The June decision to
raise the blend from 0.40 → 0.60 was correct and under-shot; the data says the right
answer under the current model is 1.0.

This is corroborated by the settled record: monthly claimed EV collapsed to ≈0 after the
0.60 change, and realised ROI moved from −16.2% (May) to −0.6% (July) / +0.9% (August).

Additional ensemble problems:
- **Double counting.** The ML models are trained on features that include
  `home_implied_prob` etc. The ensemble then blends the bookmaker's 1X2 in *again* at 60%
  in `predict()`. The market's opinion enters twice, at an uncontrolled effective weight.
- **The 1X2 output is not a coherent distribution.** After the goals blend, `predict()`
  mutates `ensemble_1x2` in place, renormalises, then dampens, then renormalises, then
  hard-caps, then renormalises again (lines 299–377). Each step is defensible; composed,
  the mapping from inputs to output is not analysable.
- **The goal-market "decisiveness" adjustment** (lines 210–235) — `goal_boost =
  (max_win_prob − 0.40) × 0.15`, `draw_penalty = (draw_prob − 0.25) × 0.20`,
  `btts_boost = (competitiveness − 0.50) × 0.10` — is six magic constants with no
  derivation and no backtest anywhere in the repo.
- `INTERNATIONAL_LEAGUES` halves the Poisson weight and multiplies Elo by 1.5 for CL/EL/ECL
  — sensible in direction, but the multipliers are invented.

### Bayesian weights (Phase 13) — **two real bugs**

1. **`update()` writes to `_league_params` and `_global_params` regardless of `market`**
   (`bayesian_weights.py` lines 112–147). `tune_ensemble_weights` calls it once with
   `market="1X2"` and once with `market="goals"` for Poisson, but only once for Elo. Result
   in the live file: Poisson n = 872, Elo n = 436 — exactly 2×. Poisson's league-level rate
   is a blend of its 1X2 accuracy and its over/under accuracy.
2. **The weight mapping cannot discriminate.** Weights are
   `E[Beta] / Σ E[Beta]`. Live global values: Poisson 425/(425+373) = 0.533, Elo
   222/(222+182) = 0.549, ML 0.55 (prior). Normalised → **0.33 / 0.34 / 0.34**. Because all
   three "accuracies" on a 3-way task sit in a narrow band, normalising them compresses
   every difference away. A model at 40% and one at 55% get 0.42 and 0.58. The learner
   converges to uniform by construction and can never express "Elo is better than Poisson"
   — which the backtest says it is.
3. Accuracy (argmax-correct) is the wrong criterion for weighting probability forecasters;
   log-loss or Brier is. Draws are almost never the argmax, so a model's draw calibration
   is invisible to the learner.

### Probability calibration (Phase 12)

`ProbabilityCalibrator` is correctly written — isotonic, per market family, MIN_FAMILY=120
/ MIN_GLOBAL=300, clipped, graceful identity fallback. It is **disabled**
(`probability_calibration_enabled: false`) and the persisted file is empty. The 2026-07-21
decision to leave it off was defensible at the time.

But it fits on the **chosen picks only** — a selection-biased sample — and is evaluated on
the same picks. There is no holdout. The honest caveat is in the docstring; the practice
still has no validation.

`calibrate_from_pick_outcomes()` is worse: it computes `factor = 1 − mean(|pred − actual|)`
floored at 0.85. This is **direction-blind**: an under-confident model would be shrunk
toward 0.5 exactly as an over-confident one is. `pick_calibration.json` is currently `{}`.

Measured calibration on the 1,018 settled picks:

| model prob band | n | avg predicted | actual win rate | error |
|---|---|---|---|---|
| 0.45–0.50 | 150 | 0.473 | **0.353** | −12.0 pp |
| 0.50–0.55 | 178 | 0.525 | 0.461 | −6.4 pp |
| 0.55–0.60 | 215 | 0.576 | 0.540 | −3.6 pp |
| 0.60–0.65 | 183 | 0.623 | 0.525 | −9.8 pp |
| 0.65–0.70 | 105 | 0.671 | 0.581 | −9.0 pp |
| 0.70–0.80 | 140 | 0.744 | 0.607 | −13.7 pp |

Compare the market on the *same* picks:

| implied band | n | avg implied | actual | market error | model error |
|---|---|---|---|---|---|
| <0.40 | 102 | 0.353 | 0.333 | **−1.9 pp** | −14.8 pp |
| 0.40–0.50 | 286 | 0.446 | 0.437 | **−0.9 pp** | −9.9 pp |
| 0.50–0.60 | 326 | 0.551 | 0.531 | **−2.0 pp** | −7.6 pp |
| 0.60–0.70 | 243 | 0.642 | 0.617 | **−2.4 pp** | −4.3 pp |
| >0.70 | 61 | 0.734 | 0.672 | −6.2 pp | −11.1 pp |

The market's residual error is ≈ the vig, uniformly. The model's is 4–15 pp of genuine
overconfidence.

### Claude review (Phase 21)

**Verdict: not yet measurable. The instrumentation is right; the sample is not there.**

| segment | n | win% | break-even | flat ROI | binomial p |
|---|---|---|---|---|---|
| Claude KEEP | 100 | 55.0% | 60.9% | **−8.7%** | 0.259 |
| Claude CHANGE | 79 | 62.0% | 57.5% | **+9.1%** | 0.429 |
| not reviewed | 839 | 49.9% | 52.1% | −4.1% | 0.213 |

The KEEP-vs-CHANGE gap (+17.8 pp of ROI) is suggestive but **not significant** at these
sample sizes, and the reviewed period is 2026-07-08 → 2026-08-05, which is confounded with
the 0.60 blend change and the July upturn. The unreviewed control group inside that window
is **n = 2** — there is effectively no control.

The proper counterfactual — `model_result`, the model's original pick graded independently
— was added in commit `4957e0a` and is populated on only **19 picks** (7 of them CHANGE).
`get_claude_added_value()` requires n ≥ 10 CHANGE pairs and currently returns `{}`.

On the 7 available CHANGE pairs: both won 1, only-Claude won 2, only-model won 1, neither 3.
That is nothing.

**One methodological problem worth fixing regardless of the outcome.**
`_recent_review_stats()` injects into the decision prompt a line like
`KEEP: 14/26 won (54%) | CHANGE: 7/8 won (88%)`, with the stated intent of giving Claude
"evidence-based encouragement to act on research instead of deferring to the saved pick."
That is an 8-sample statistic being used as an instruction to change more often. The same
applies to `_recent_selection_stats()`, which feeds per-selection win rates at `min_n = 8`.
This is noise being laundered into a prompt directive, and it makes the KEEP/CHANGE split
partly self-fulfilling — which further contaminates the very experiment that is supposed to
measure the review's value.

What the review *does* demonstrably do: it moves picks **toward the market**. Reviewed picks
have `p_model − 1/odds` of −0.1% (KEEP) and **−3.9%** (CHANGE), versus +8.1% for unreviewed
picks. Given everything in §5, moving toward the market is the correct direction — and it is
the most plausible explanation for why reviewed picks look better.

---

## 6. Leakage audit (Phase 6)

**Clean (verified):**
- Team form, H2H, xG, momentum, situational, WC tournament features all filter on
  `as_of_date` in both the preload and live paths.
- Elo ratings are correctly set to `None` during training (`create_features` line 473) so
  full-history Elo does not leak into past opponent-quality weights. This was a real trap
  and it is handled.
- The preload-cache completeness rule (`preload_cache._resolve`) is sound: a cached slice
  is only served when it provably equals the live query.
- `tune_ensemble_weights` refits Poisson/Elo with `as_of_date=oldest_pick_date` before
  scoring.
- Settlement grades against the **regulation** score, and `_grade_selection` returns `None`
  on unknown selections rather than defaulting to a loss.

**Leaks and skews found:**

| # | Issue | Severity |
|---|---|---|
| L1 | **Asymmetric leakage in `tune_ensemble_weights`.** Poisson and Elo are refit as-of the oldest settled pick, but ML predictions come from the *currently loaded pickles*, which were trained on data that includes those very matches. The model comparison that sets ensemble weights therefore hands ML a look-ahead advantage its rivals are denied. (Masked today only because ML eval usually yields nothing — `ml` n = 0.) | **High** |
| L2 | **In-sample parameter selection at scale.** `excluded_markets`, `club_pick_min_ev`, `club_pick_min_blend`, `wc_mismatch_fav_prob`, `wc_mismatch_dog_xg`, the club BTTS-Yes ban, the agreement bonuses, the contrarian bonus — all chosen by inspecting settled outcomes, none validated on a holdout. Permutation test says the segment differences are noise (league p = 0.407). This is the classic form of "optimising against future results" the prompt's Rule 1 forbids | **High** |
| L3 | **Train/serve odds skew.** `_get_bookmaker_features(match_id)` and `_get_odds_movement_features(match_id)` take no `as_of_date`. For a training match they read the last-stored pre-kickoff price; at prediction time they read a price ~6.3 h before kickoff (measured median). Not future information — 0% of odds rows are written after the pick day — but the two distributions differ | Medium |
| L4 | **Probability calibration and pick calibration fit and evaluate on the same settled picks**, with no holdout and with selection bias (only the picks we chose) | Medium |
| L5 | **Prompt-mediated feedback loop.** `_recent_review_stats` / `_recent_selection_stats` feed settled outcomes back into the decision prompt at `min_n = 8`. Outcomes influence future decisions through the LLM without any statistical gate | Medium |
| L6 | Referee preload window is anchored to `date.today() − 365d`, not to `as_of_date`. Safe in practice (the completeness rule falls back to the live query), but the two paths are not equivalent by construction the way the others are | Low |

---

## 7. Backtest results — before vs after

**No "after" is reported, because Stage 1 mandates no code changes.** What the backtests
establish is the size of the prize:

| configuration | 1X2 log-loss (1,710 OOS matches) | vs shipped |
|---|---|---|
| shipped: Poisson(180, −0.13) + Elo, blend 0.60 | 0.9983 | — |
| blend 0.80 | 0.9940 | −0.0043 |
| **blend 1.00 (pure de-vigged market)** | **0.9929** | **−0.0054** |
| Poisson half-life 730, ρ=0, blend 0.60 (est.) | ≈0.9975 | −0.0008 |

The blend weight is worth roughly **seven times** more than the Poisson parameter tuning.
That is the whole story of this audit in one line.

---

## 8. Market performance (Phase 16)

| selection | n | win% | break-even | flat ROI | claimed EV | p |
|---|---|---|---|---|---|---|
| Over 2.5 Goals | 293 | 54.3% | 55.1% | −0.7% | +13.8% | 0.769 |
| Home Win | 254 | 49.6% | 48.4% | **+2.8%** | +17.2% | 0.707 |
| BTTS Yes | 120 | 52.5% | 55.4% | −4.8% | +7.3% | 0.582 |
| Home Over 1.5 | 102 | 43.1% | 48.5% | −11.0% | +13.0% | 0.322 |
| Over 1.5 Goals | 69 | 69.6% | — | +0.4% | +7.2% | — |
| Away Win | 67 | 43.3% | 48.9% | −12.6% | +11.9% | 0.394 |
| Under 3.5 Goals | 45 | 55.6% | 66.2% | −16.5% | +8.9% | 0.155 |
| Over 3.5 Goals | 33 | 36.4% | 45.9% | −22.0% | +38.5% | 0.299 |
| Under 2.5 Goals | 11 | 27.3% | — | −48.0% | +11.3% | — |

**Nothing here is significant.** The permutation test on the selection-level ROI spread
gives p = 0.087 — borderline, and driven entirely by the small-n tails.

Two operational notes:
- The two largest markets, Over 2.5 (n=293) and Home Win (n=254), both land within ~1 pp of
  their break-even rate. That is the "no edge, paying the vig" signature again, at the only
  sample sizes large enough to say anything.
- **`config/config.yaml` and `config/config.example.yaml` have diverged.** The example
  excludes `over_3.5` ("proven loser"); the live config does **not**. The live config also
  sets `max_total_kelly_pct: 0`, disabling the daily exposure cap that the example
  documents as 40%. Whatever the intent, the running system is not the documented one.

---

## 9. League performance (Phase 9)

Worst five: `spain/laliga2` −33.6% (n=45), `england/championship` −33.4% (19),
`netherlands/eredivisie` −26.5% (33), `sweden/allsvenskan` −24.1% (43),
`italy/serie-b` −24.0% (21).

Best five: `romania/liga-1` +48.0% (29), `greece/super-league` +38.9% (14),
`germany/2-bundesliga` +30.6% (12), `france/ligue-2` +30.1% (13),
`turkey/super-lig` +25.2% (25).

**Permutation test: observed spread 81.6 pp, p = 0.407.** With 1,018 picks spread over 30+
leagues, a spread this large is exactly what random assignment produces. **Do not act on
this table.** Any league-level exclusion or weighting derived from it is noise-fitting.

---

## 10. Odds-range performance (Phase 10)

| odds band | n | win% | flat ROI | claimed EV |
|---|---|---|---|---|
| 1.00–1.40 | 45 | 71.1% | −4.2% | +6.8% |
| 1.40–1.60 | 173 | 60.7% | −8.1% | +2.9% |
| 1.60–1.80 | 234 | 58.5% | −1.2% | +4.8% |
| 1.80–2.00 | 167 | 49.7% | −5.8% | +14.4% |
| 2.00–2.50 | 280 | 45.4% | +0.1% | +20.0% |
| 2.50–3.50 | 111 | 31.5% | **−13.1%** | **+33.1%** |
| 3.50+ | 8 | 50.0% | +82.1% | +73.2% |

The pattern is the favourite-longshot bias appearing in the model, not the market: the
higher the odds, the larger the model's claimed edge and the worse the outcome — until
n collapses to 8. `min_odds: 1.50` combined with `divergence > 2.0` rejection is doing some
work, but the 2.50–3.50 band remains a systematic leak.

---

## 11. CLV analysis (Phase 17) — **the metric does not exist**

This is the most consequential gap in the whole system.

- **0 of 124,158** odds rows on picked matches carry a timestamp after the pick day. Odds
  are captured once, a median of **6.3 hours before kickoff**, and never refreshed.
- `opening_odds` is populated on 80.6% of rows, but on only **8.4%** of those does it
  differ from the current value. The line-movement signal that the odds-movement features
  are built on is 92% absent.
- Therefore **closing-line value cannot be computed for a single pick, past or future.**
- The `avg_clv` reported by `get_stats()` and shown in `--stats` / `--report` is
  `predicted_probability − 1/odds` (line 2616). That is not CLV. It is the model's *own
  declared edge*. It currently reads **+6.3%**, which reads as "we consistently beat the
  market by 6.3 points" when the truth is "we consistently disagree with the market by 6.3
  points, and are wrong to". This metric has been actively misleading every performance
  report.

The prompt says a strategy with positive CLV and short-term negative ROI should not be
removed. That protection cannot be applied here, because CLV is unmeasured. Fixing this is
recommendation #2.

---

## 12. Risk analysis (Phase 15)

| staking | final | turnover | ROI | max drawdown | longest losing streak | per-bet Sharpe |
|---|---|---|---|---|---|---|
| flat 1 u | −36.3 u | 1,018 u | −3.6% | 59.9 u (5.9% of turnover) | 9 | −0.036 |
| Kelly % | −66.5 u | 2,633 u | −2.5% | 173.3 u (6.6% of turnover) | 9 | −0.022 |

Observations:
- Risk *control* is working: 4% max stake, 0.25 fractional Kelly, agreement scaling, and
  the correlation filter keep drawdown modest and the equity curve smooth. Bankroll
  survival is not at risk.
- The drawdown circuit breaker and `_auto_calibrate_ev_threshold` respond to **n = 30–40**
  samples — far below the noise floor. At a true ROI of −3.6% and typical odds, a 40-pick
  window has a standard error of roughly ±16 pp. The breaker is reacting almost entirely to
  noise.
- Worse, the EV auto-calibration **tightens `min_ev` during cold streaks** — pushing
  selection *up* the claimed-EV scale, which §1 shows is where the losses are. The
  mechanism is directionally wrong given the measured EV/ROI relationship.
- Kelly is computed on `predicted_probability`, which the calibration table shows is 4–14
  pp too high. Even at 0.25 fractional Kelly, sizing on inflated probabilities systematically
  over-stakes. That the Kelly ROI (−2.5%) beats the flat ROI (−3.6%) is a small mercy from
  the agreement scaling, not evidence that the sizing is right.

---

## 13. Free data opportunities (Phases 3, 4, 23)

Current sources: Flashscore (scraping, fixtures/results/stats/referee/venue), API-Football
(100 req/day free — the binding constraint), football-data.org (free, 9 leagues, 10/min),
The Odds API, football-data.co.uk CSVs (history), Open-Meteo (free, no key), Claude web
research.

### Tier 1 — clear, measurable improvement

| source | what it fixes | cost | risk |
|---|---|---|---|
| **football-data.co.uk closing-odds columns** (`PSCH/PSCD/PSCA`, `B365CH/D/A`, `AvgCH/D/A`) | The loader already downloads these CSVs for history but ignores the closing-odds columns. They give a **retrospective closing line for thousands of past matches in 20+ leagues**, immediately enabling: real CLV, a properly de-vigged market probability for training, and a market-anchored target. This single source turns CLV from unmeasurable into measurable on historical data | Low — parsing columns already being downloaded | None |
| **The Odds API closing snapshot** (free tier; the `wagyu-sports` MCP on this machine reports **468 of 500 monthly requests remaining**) | One request per league shortly before kickoff, stored as a separate `closing_odds` column, gives **forward** CLV. ~30 requests/day is out of budget on 500/month, but one snapshot per matchday for the top 6 leagues is affordable | Low | Free-tier limit |
| **Understat** (free, scrapeable, top-5 leagues + RFPL, 2014→present) | Takes xG coverage from **1.9% → ~60%** for the leagues that matter. This is the difference between the xG feature family being dead and being real. `soccerdata` (PyPI, MIT) wraps it | Medium | Scraping fragility; check ToS |
| **ClubElo** (`http://api.clubelo.com/`, free, daily, no key) | Cross-league-normalised, well-calibrated club Elo. Fixes the league-strength normalisation gap in the in-house Elo, and gives an independent second opinion for the ensemble | Low | Coverage is European clubs only |

### Tier 2 — plausible, limited evidence

- **FBref / StatsBomb via `soccerdata`** — shots, SoT, possession, pressures, at far better
  coverage than the current 2.4%. Cloudflare-protected and rate-limited; treat as a nightly
  batch, not a live dependency.
- **API-Football lineups** (~1 request per fixture, ~1 h before kickoff) — confirmed
  lineups are one of the few genuinely pre-match signals the market prices slowly. Blocked
  by the 100/day free tier unless the odds fallback is trimmed.
- **openfootball / football.db** — free fixture and result data, useful as a cross-check for
  the duplicate-team problem noted in project memory.

### Tier 3 — not worth implementing

- **More weather.** Venue coverage is 8%, and weather features are structurally excluded
  from ML training anyway. Fix the plumbing before adding data.
- **More referee data.** 19.8% coverage, and the fouls/cards columns it depends on are at
  2.4%. Referee effects on goals are small and swamped by team quality.
- **Injury scraping at the current scale.** 17 rows total. Either commit to a real source
  (API-Football, quota-permitting) or drop the feature family honestly rather than shipping
  zeros.
- **Additional bookmakers.** 12 books are already stored; the median is already stable. More
  books adds egress, not signal.

### On MCP servers and access

- `wagyu-sports` **works** and has a live Odds-API key with **468/500 monthly requests
  remaining** — usable today for the closing-line snapshot in Tier 1.
- `soccer-server` (API-Football wrapper) is available but shares the same free-tier quota
  ceiling as the in-repo scraper; it adds convenience, not capacity.
- `claude.ai Supabase` MCP is connected and can read the production project directly.
- **`claude.ai Google Calendar` is not authorised** in this session and I could not run the
  OAuth flow here. It is not needed for this audit; if you want it available later,
  authorise it from your claude.ai connector settings.
- Nothing else was blocked. I did not need any access I did not have.

---

## 14. Code changes

**None. Stage 1 is audit-only, as the prompt specifies.**

For Stage 3, the concrete defects found are listed below with file and line so the work is
ready to start:

| # | File | Location | Defect |
|---|---|---|---|
| 1 | `src/models/bayesian_weights.py` | `update()` 112–147 | League/global params updated regardless of `market` → Poisson double-counted (872 vs 436 obs), 1X2 and goals accuracy mixed |
| 2 | `src/models/bayesian_weights.py` | `_params_to_weights()` 213–221 | Normalising near-identical accuracies compresses all differences → weights converge to uniform; cannot express "Elo > Poisson" |
| 3 | `src/agent/betting_agent.py` | 3295 | `ml_models._models` — attribute is `models`. CV classification report has never run |
| 4 | `src/agent/betting_agent.py` | `tune_ensemble_weights` 2762–2890 | Poisson/Elo refit as-of; ML uses full-data pickles. Asymmetric look-ahead in the weight comparison |
| 5 | `src/agent/betting_agent.py` | `get_stats` 2614–2621 | `avg_clv` is `p − 1/odds`, not CLV. Mislabelled in every `--stats` / `--report` output |
| 6 | `src/features/feature_engineer.py` | 499–505, 653–658 | Injury and weather features excluded from training but computed at predict time → dropped by `_align_features`. Dead work on every fixture |
| 7 | `src/features/team_features.py` | `_get_league_standings` 647–696 | "League position" is a rolling 50-match points ranking across seasons, with unequal match counts. `in_relegation_zone` / `title_gap` are not what they claim |
| 8 | `src/models/poisson_model.py` | `_estimate_league_rhos` 325–393 | Per-league MLE over every league × match; measured benefit on both 1X2 and O/U 2.5 is nil |
| 9 | `config/config.yaml` vs `config.example.yaml` | — | Diverged: `over_3.5` excluded in example only; `max_total_kelly_pct: 0` (cap disabled) in live |
| 10 | `src/reporting/match_briefing.py` | `_recent_review_stats` 784–816, `_recent_selection_stats` 751–782 | Injects 8-sample statistics into the decision prompt as directives; contaminates the KEEP/CHANGE experiment |
| 11 | `src/agent/betting_agent.py` | `_auto_calibrate_ev_threshold` 3790–3889 | Tightens `min_ev` on cold streaks, selecting *higher* claimed-EV picks — the cohort that loses most. n=40 is below the noise floor |
| 12 | `src/models/probability_calibration.py`, `calibrate_from_pick_outcomes` | — | Fit and evaluated on the same selection-biased picks; `calibrate_from_pick_outcomes` is direction-blind (`1 − mean\|err\|`) |
| 13 | `data/models/` | — | Not version-controlled; survives only via GH Actions cache. `feature_list.json` missing → warning on every predict |
| 14 | `src/models/ensemble.py` | `predict()` 299–394 | Blend → dampen → renormalise → hard-cap → renormalise chain; output is not an analysable function of inputs. Bookmaker enters twice (ML features + explicit blend) |
| 15 | `src/models/ensemble.py` | 210–235 | Six undocumented magic constants in the goal-market "decisiveness" adjustment, never backtested |

---

## 15. Remaining weaknesses that free data cannot solve

- **Beating a de-vigged Pinnacle/Bet365 line on 1X2 and O/U 2.5 with public pre-match data
  is, for practical purposes, not achievable.** These are the most efficient football
  markets in existence. The measured gap (market log-loss 0.9929 vs best model 1.0279) is
  not a tuning gap; it is an information gap. The market prices lineups, team news, weather,
  motivation and sharp money that no free feed exposes before kickoff.
- **Injuries and confirmed lineups** — the one pre-match input with real residual value —
  have no reliable free source at 30-league scale.
- **Sample size.** At a true edge of ±2%, distinguishing skill from noise needs thousands of
  bets. At ~200 picks/month, that is years. Every in-season decision made on 30–100 picks
  will be noise-driven, no matter how the code is written.
- **Where an edge could plausibly exist** is not in these markets: lower-division and
  Nordic/Eastern European leagues, less liquid derivative markets (corners, cards, team
  totals, Asian handicaps), and early-line-vs-closing-line movement. All of them require
  the closing-line infrastructure that does not yet exist.

---

## Top 10 improvements by expected predictive impact (Stage 2)

1. **Set `bookmaker_blend_weight` to 1.0 for 1X2 and O/U, or stop betting those markets.**
   The blend sweep is monotone on 1,710 out-of-sample matches; every point of model weight
   makes the forecast worse.
2. **Build closing-line infrastructure.** Add a `closing_odds` column; backfill from the
   football-data.co.uk closing columns already being downloaded; snapshot forward closing
   lines via The Odds API. Then make CLV the primary diagnostic.
3. **Fix the `avg_clv` mislabel.** It is the model's declared edge, not CLV, and it has been
   reading +6.3% while realised ROI is −3.6%.
4. **Freeze all noise-fitted thresholds and re-derive them on a holdout** (or delete them).
   League/market exclusions, `club_pick_min_ev`, `wc_mismatch_*`, the agreement and
   contrarian bonuses. Permutation test: league p = 0.407.
5. **Replace the Bayesian weight learner's accuracy criterion with log-loss, fix the
   market double-count, and use softmax-over-log-loss so weights can actually diverge.**
6. **Promote Elo, demote Poisson** — Elo wins on log-loss, Brier and accuracy. Fit the draw
   curve; add league-strength normalisation.
7. **Move Poisson half-life to 365–730 days and set `dixon_coles_rho: 0.0`**; delete
   `_estimate_league_rhos`. Measured, small, free.
8. **Fix the train/serve feature asymmetry.** Either include injuries and weather in
   training (with historical values) or remove them from the predict path. Today the work is
   done and discarded.
9. **Restore ML observability**: fix the `_models` typo, write `feature_list.json`, feed the
   Bayesian learner real ML outcomes, and remove the ML look-ahead advantage in tuning.
10. **Get xG coverage above 50%** via Understat/`soccerdata`, so the xG feature family and
    xG-Poisson stop being dead code.

---

## The most important final question

> If this system had to operate for the next 12 months using only currently available free
> data sources, what are the 5 changes most likely to improve its long-term risk-adjusted
> betting performance?

**#1 — Stop betting against the closing line on 1X2 and Over/Under 2.5.**
Either take the bookmaker blend to 1.0 (making the system a market-follower that bets only
on genuine price dispersion between books) or exit these markets. Three independent tests
say the model subtracts information: pick-level p = 0.265, model Brier ≈ constant Brier,
and a blend sweep that is monotone to w = 1.0. This alone converts a −3.6% ROI into
approximately −vig, and removes the −13% to −19% claimed-EV cohorts.

**#2 — Build the closing line, then make CLV the only metric that governs decisions.**
Without it every decision this system makes is being judged on 30–100-sample ROI, which is
pure noise. With it, edge becomes visible in weeks instead of years. Backfill from the
football-data.co.uk CSVs already downloaded; snapshot forward lines with the free Odds API
quota. This is the highest-leverage engineering change in the repo because it makes every
subsequent change testable.

**#3 — Delete every threshold that was fitted on settled outcomes without a holdout.**
League exclusions, market exclusions, `club_pick_min_ev`, `wc_mismatch_*`, agreement and
contrarian bonuses, and the prompt-injected win-rate lines in the Claude review. The
permutation test (league p = 0.407) says these encode noise, and each one narrows the
betting universe in a direction chosen by chance. Removing them raises expected performance
by removing negative-expectation constraints, not by adding signal.

**#4 — Fix the ensemble's weighting machinery so it can express what the data says.**
Concretely: score models by log-loss not argmax-accuracy; stop double-counting the
bookmaker (it enters the ML features *and* the explicit blend); fix the Bayesian
double-update; use a mapping that can actually separate a 48.7%-accuracy model from a
47.6% one. Today the learner is mathematically incapable of preferring Elo over Poisson
even though Elo is better on every metric.

**#5 — Fix data coverage before adding any modelling sophistication.**
xG at 1.9%, shots at 2.4%, injuries at 17 rows, weather structurally excluded from
training. Roughly half of the advertised feature set is computed, pruned, and thrown away
every run. Understat via `soccerdata` takes xG to ~60% on the leagues that matter; fixing
the train/serve asymmetry makes weather and injuries usable at all. Until this is done,
every model improvement is being applied to a feature matrix that is mostly zeros — and
no amount of ensemble tuning fixes that.

**Not chosen, and why:** stacked meta-models, neural approaches, more markets, more
bookmakers, more leagues. All would add complexity to a system whose measured predictive
content is currently indistinguishable from a constant. Rule 4.
