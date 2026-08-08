# Stage 3 — Evidence-Driven Reconstruction, 2026-08-07

Executed against `docs/iteration-previewing-prompt.md`, using
`docs/predictive-audit-2026-08-07.md` as the source of truth.

Every change below was measured before being kept. Changes that came back
negative or statistically inconclusive were **reverted or not applied**, and are
listed in section L. Test suite: **422 passing** (was 408; +14 new).

Reproduce with:

```bash
python -m scripts.run_baseline --window 60      # the immutable baseline
python -m pytest tests/ -q                      # regression suite
```

---

## A. Bugs fixed

### A1. `ml_models._models` — a diagnostic that had never once run

`betting_agent.py:3295` read `self.predictor.ml_models._models`; the attribute is
`models`. It raised `AttributeError` on every training run and the surrounding
`except` swallowed it, so the per-class precision/recall report has never
produced output.

Fixed by extracting it to `MLModels.cross_val_report()` with a regression test
(`tests/test_ml_cv_report.py::test_cross_val_report_actually_runs`) that fails if
the path stops executing, plus
`test_models_attribute_name_is_models_not_underscore_models` pinning the exact
typo.

**While fixing it I found a second defect in the same code.** It called
`cross_val_predict(..., cv=3)` — plain `KFold` with `shuffle=False` — and labelled
itself "leak-free diagnostic". With chronological data, folds 2 and 3 train on
rows that come *after* fold 1's test rows. It now walks forward with
`TimeSeriesSplit`, fits a fresh clone per fold, and scales inside the fold so the
scaler never sees test rows. `test_report_is_forward_chained_not_kfold` asserts
`train_idx.max() < test_idx.min()` for every fold.

### A2. Market contamination in the weight learner

`bayesian_weights.update()` wrote the league and global buckets regardless of
`market`, so an over/under observation moved the weights used for 1X2. It also
gave Poisson two updates per settled pick (1X2 + goals) against Elo's one —
visible in production as **872 recorded observations for Poisson and 436 for
Elo**, exactly 2×.

Fixed structurally rather than with a conditional: **every scope key is now
market-qualified** (`"1X2::england/premier-league"`), so there is no
market-agnostic bucket to contaminate.
`test_market_isolation` and `test_no_market_agnostic_bucket_exists` guard it.

### A3. `"Home/Away"` mapped to the 1X2 market — the largest defect found

Not in the prompt's list; found while auditing the EV pipeline (Phase 5's
"incorrect outcome mapping").

`BET_TYPE_MAP` mapped API-Football's **`"Home/Away"`** bet — the *two-way,
draw-excluded* market — to `market_type: "1X2"` with the same `Home`/`Away`
labels. For any bookmaker offering both bets, the two-way prices overwrote the
real 1X2 ones under the same `(bookmaker, market_type, selection)` key.

The arithmetic confirms it exactly. Match 49032:

| book | Home | Draw | Away | overround |
|---|---|---|---|---|
| Pinnacle | 1.71 | 3.66 | 4.55 | 1.078 |
| 1xBet | 1.74 | 3.81 | 4.89 | 1.042 |
| **Bet365 (stored)** | **1.25** | 3.40 | **3.75** | **1.361** |

Pinnacle de-vigged gives P(H)=0.542, P(A)=0.204, so the two-way home price is
0.542/0.746 = 0.727 → fair 1.376, which with margin is the 1.25 that was stored.
Only Home and Away were overwritten — the two-way bet has no Draw leg — which is
precisely the observed signature.

**Blast radius:** all **2,486** Bet365 1X2 rows in production, median overround
**1.3524**. Bet365 was also the *first-priority* book in
`_get_bookmaker_features`, so these were the primary source of
`home/draw/away_implied_prob` — the features that feed both ML training and the
bookmaker blend. Measured: the first-choice book had an implausible overround on
**92.3%** of matches.

Fixed by mapping `"Home/Away"` to `draw_no_bet` (its actual meaning) and by
making the feature layer robust to already-corrupt rows (section C1).

---

## B. Statistical defects fixed

### B1. Weight learning rebuilt around predictive loss

The old learner tracked argmax-**accuracy** as a Beta posterior and set weights to
`E[Beta] / ΣE[Beta]`. On a 3-way task every model's accuracy sits in a narrow band
(live values 0.533 / 0.549 / 0.550), so normalising compressed every difference
away — the production file resolved to **0.33 / 0.34 / 0.34**. It was
mathematically incapable of preferring Elo over Poisson even though Elo wins on
log-loss, Brier and accuracy.

Replaced with decayed **log-loss** and the Hedge / multiplicative-weights rule,
`w ∝ exp(-η · cumulative_loss)` with the textbook `η = sqrt(8 ln K / n_eff)`.
Because the exponent is `sqrt(8 ln K · n_eff) · mean_loss`, separation grows with
evidence: uniform at cold start, decisive once the data supports it. That is a
standard algorithm with known regret bounds, not a tuned constant.

Accuracy is the wrong criterion for weighting probability forecasters in any case:
it is blind to calibration, and because the draw is almost never any model's
argmax, it is blind to a third of the outcome space.

### B2. Idempotent updates

Nothing previously stopped the same settled pick being applied on every run. Each
observation now carries a key; re-applying it is a no-op, and keys are pruned once
decayed past six half-lives. `test_replaying_the_whole_history_changes_nothing`
and `test_dedup_survives_a_save_load_round_trip` cover it.

### B3. ML look-ahead advantage in the weight comparison removed

Audit finding L1. `tune_ensemble_weights` refits Poisson/Elo as-of the oldest
settled pick, but ML predictions came from the *currently loaded pickles* — trained
on data that includes those very matches. The comparison that sets ensemble
weights was handing ML a look-ahead advantage its rivals were denied.

The learner now only accepts an ML observation when the match post-dates the
model's `trained_at`, and logs how many were skipped.

### B4. Shrinkage and scope discipline

League weights shrink toward the market-global weights, which shrink toward the
config prior, both by `n_eff / (n_eff + 30)`. A scope is only used when *every*
model has data in it — otherwise absence of evidence would read as evidence of
quality. Weights are clamped to [0.05, 0.70] by a water-filling projection
(clip-then-renormalise does not work: renormalising scales a clipped entry back
over the bound — a bug I hit and fixed in my own first implementation).

### B5. The mislabelled CLV metric

`get_stats()` reported `avg_clv`, computed as `predicted_probability - 1/odds`.
That is the model's *own claimed edge*, not closing line value. It read **+6.3%**
while realised flat ROI was **−3.6%**, and it was shown in every `--stats`,
`--report` and Telegram performance message.

Renamed to `avg_model_market_divergence` with an explicit "not evidence of edge"
label. `avg_clv` now exists but is computed only from genuine closing prices, and
reports `UNAVAILABLE` with a reason until they exist.

---

## C. Architecture changes

### C1. Consensus de-vigging replaces single-book de-vigging

`_get_bookmaker_features` de-vigged **one** book (Bet365 → Pinnacle → any) while
`value_calculator` computed EV against the **median price across all books**. Two
problems: a single corrupt book dominated (A3), and the median price sat **+8.3%**
above the reference book's on average (p90 **+36.9%**) — every point of which
entered claimed EV as if it were edge.

Now: per-outcome median of de-vigged probabilities across every book that passes
an overround plausibility gate. One bad book cannot dominate, and the probability
comes from the same cross-book consensus as the price it is compared against.

**Measured on 2,761 completed matches with 1X2 prices:**

| market probability | log-loss | Brier | accuracy |
|---|---|---|---|
| OLD: single book, Bet365-first, no gate | 1.0034 | 0.5983 | 52.1% |
| **NEW: cross-book median, gated** | **0.9931** | **0.5933** | **52.4%** |
| reference: Pinnacle alone | 0.9946 | 0.5940 | 52.3% |

Paired bootstrap **+0.0103 nats, 95% CI [+0.0052, +0.0154] — significant**. Home
bias fell from +1.74pp to −0.79pp. The consensus also beats the sharpest single
book.

### C2. The immutable baseline (`src/evaluation/baseline.py`, `scripts/run_baseline.py`)

Committed, reproducible, chronological-only. Scores every candidate on the
**identical** match set (intersection of what all candidates can forecast), so
differences reflect models and not coverage. Reports log-loss, Brier, calibration
error and accuracy; reports ROI and CLV as explicitly unavailable rather than
approximating them. Snapshots to `data/baselines/*.json`, never edited in place.

### C3. Gate registry (`src/betting/gate_registry.py`)

Every outcome-derived gate is declared with what it removes, the evidence
originally cited, its walk-forward verdict, and its holdout ROI effect. A-priori
risk constraints (odds range, stake caps, divergence sanity, min Kelly) are
separated from empirical edge claims and stay on. Unknown gate names raise rather
than silently resolving to "disabled".

### C4. Closing-line infrastructure

`saved_picks.closing_odds` + `closing_odds_captured_at` (migration 003 with
rollback), and `scripts/capture_closing_lines.py` to populate them from the
cross-book consensus shortly before kickoff. `closing_odds_captured_at` exists so
an early snapshot is distinguishable from a true closing price — a CLV series
silently built from 6-hours-out snapshots would be worse than none.

Both columns are nullable and `_migrate_missing_columns` adds them automatically
on the first CI invocation, so no manual DDL step is required.

---

## D. Data sources added

**None.** Phase 10 is explicit: do not optimise around data that barely exists.
Coverage remains xG 1.9%, shots 2.4%, injuries 17 rows. See section K.

---

## E. Features removed

- **Per-league Dixon-Coles ρ estimation** (`_estimate_league_rhos`) — now off by
  default (`models.dc_rho_per_league: false`). It ran a
  `scipy.optimize.minimize_scalar` over every league × every match, calling
  `poisson.pmf` inside the objective, on every `fit()` (~10 fits a CI day). On the
  market it was designed for, Over/Under 2.5 log-loss was **0.7083 at ρ=−0.13 and
  0.7084 at ρ=0** — no effect. On 1X2 it was actively harmful.
- **Dixon-Coles correction itself** (`dixon_coles_rho: 0.0`) — same evidence.

No feature *columns* were removed. The dead families identified in the Stage 1
audit (weather, injuries, extended stats, odds movement) are a data-coverage
problem, not a code problem, and removing them would foreclose the fix.

---

## F. Features added

- `bookmaker_consensus_books` — how many books backed the consensus probability.
  Makes the difference between a 1-book and a 12-book estimate visible to the
  model and to diagnostics.

---

## G. Model changes

| parameter | before | after | evidence |
|---|---|---|---|
| `bookmaker_blend_weight` | 0.60 | **0.80** | smallest weight not significantly worse than pure market |
| `strength_half_life_days` | 180 | **540** | 180d was the worst half-life tested |
| `dixon_coles_rho` | −0.13 | **0.0** | best or tied at every half-life |
| `dc_rho_per_league` | on | **off** | no measurable benefit, real CI cost |
| Elo `reg` | 0.33 | **0.33 (unchanged)** | improvement not significant — see L |

**Blend sweep** (1,901 out-of-sample matches, decision rule fixed in advance:
*take the smallest bookmaker weight whose log-loss is statistically
indistinguishable from pure market*):

| blend | log-loss | vs market 100% | verdict |
|---|---|---|---|
| market 100% | 0.9906 | — | baseline |
| market 90% | 0.9913 | [−0.0005, +0.0020] | ok |
| **market 80%** | **0.9928** | **[−0.0002, +0.0048]** | **ok ← chosen** |
| market 75% | 0.9938 | [+0.0002, +0.0064] | SIGNIFICANTLY worse |
| market 60% | 0.9979 | [+0.0026, +0.0122] | SIGNIFICANTLY worse (was shipped) |

**Poisson grid** (restricted, theory-motivated; no unrestricted search — 4
half-lives × 4 ρ):

| setting | log-loss |
|---|---|
| hl=730, ρ=0.00 | 1.0314 |
| **hl=540, ρ=0.00** | **1.0315** |
| hl=365, ρ=0.00 | 1.0322 |
| hl=180, ρ=0.00 | 1.0364 |
| hl=180, ρ=−0.10 (≈shipped) | 1.0376 |

540 and 730 are tied within noise; the shorter memory is preferred on a tie.
Combined change vs shipped: **+0.0063 nats, CI [+0.0028, +0.0098] — significant
standalone.**

---

## H. Baseline vs final metrics

The immutable baseline, 1,901 out-of-sample matches across 5 chronological
windows:

| Model | LogLoss | Brier | CalErr | Acc | ROI | CLV | n |
|---|---|---|---|---|---|---|---|
| market (de-vigged consensus) | **0.9906** | **0.5917** | 0.0158 | **52.9%** | n/a | unavailable | 1901 |
| market (raw 1/odds, vig in) | 0.9924 | 0.5925 | 0.0113 | 52.8% | n/a | unavailable | 1901 |
| market 80% + poisson/elo 20% | 0.9928 | 0.5925 | 0.0180 | 52.6% | n/a | unavailable | 1901 |
| market 60% + poisson/elo 40% | 0.9979 | 0.5956 | 0.0183 | 52.4% | n/a | unavailable | 1901 |
| market 40% + poisson/elo 60% | 1.0058 | 0.6009 | 0.0217 | 52.1% | n/a | unavailable | 1901 |
| elo only | 1.0304 | 0.6182 | **0.0059** | 49.0% | n/a | unavailable | 1901 |
| poisson + elo (50/50) | 1.0307 | 0.6184 | 0.0053 | 48.5% | n/a | unavailable | 1901 |
| poisson only (180d, ρ=−0.13) | 1.0409 | 0.6242 | 0.0128 | 47.6% | n/a | unavailable | 1901 |

Elo and Poisson have *better* calibration error than the market but far worse
log-loss — they are under-confident and poorly discriminating, which ECE rewards
and log-loss does not. Reported separately for exactly this reason.

### BEFORE vs AFTER, shipped configuration

| configuration | LogLoss | Brier | CalErr | Acc | n |
|---|---|---|---|---|---|
| reference: pure market consensus | 0.9906 | 0.5917 | 0.0158 | 52.9% | 1901 |
| **AFTER (this stage)** | **0.9926** | **0.5923** | 0.0187 | 52.3% | 1901 |
| — blend 0.80 only | 0.9975 | 0.5945 | 0.0107 | 52.5% | 1901 |
| — consensus de-vig only | 0.9979 | 0.5956 | 0.0183 | 52.4% | 1901 |
| **BEFORE (shipped this morning)** | 1.0006 | 0.5966 | 0.0115 | 52.4% | 1901 |
| — poisson 540/ρ0 only | 1.0011 | 0.5966 | 0.0131 | 52.2% | 1901 |

Paired bootstrap vs BEFORE:

| variant | improvement | 95% CI | verdict |
|---|---|---|---|
| **AFTER (all changes)** | **+0.0079** | **[+0.0037, +0.0122]** | **SIGNIFICANT** |
| blend 0.80 only | +0.0030 | [+0.0005, +0.0057] | SIGNIFICANT |
| consensus de-vig only | +0.0026 | [−0.0009, +0.0061] | not significant *in blend* |
| poisson 540/ρ0 only | −0.0005 | [−0.0017, +0.0007] | not significant |

**Max drawdown / ROI:** unchanged and not re-measured — no settled picks exist
under the new configuration. The Stage 1 figures stand (flat 1u: −3.6% ROI,
max DD 59.9u on 1,018u turnover, longest losing streak 9, per-bet Sharpe −0.036).
Re-measuring ROI on the same 1,018 picks the changes were derived from would be
exactly the in-sample self-deception this stage exists to remove.

**CLV:** still unavailable. The infrastructure now exists; the data does not yet.

---

## I. Market-specific results

Unchanged from Stage 1 — no new settled data. For the record, and to keep the
sample sizes visible:

| selection | n | win% | break-even | flat ROI | binomial p |
|---|---|---|---|---|---|
| Over 2.5 Goals | 293 | 54.3% | 55.1% | −0.7% | 0.769 |
| Home Win | 254 | 49.6% | 48.4% | +2.8% | 0.707 |
| BTTS Yes | 120 | 52.5% | 55.4% | −4.8% | 0.582 |
| Home Over 1.5 | 102 | 43.1% | 48.5% | −11.0% | 0.322 |
| Away Win | 67 | 43.3% | 48.9% | −12.6% | 0.394 |
| Under 3.5 Goals | 45 | 55.6% | 66.2% | −16.5% | 0.155 |
| Over 3.5 Goals | 33 | 36.4% | 45.9% | −22.0% | 0.299 |

Nothing is significant. The two largest markets land within ~1pp of break-even —
the signature of no edge, paying the vig.

The Over/Under 2.5 blend sweep (same windows) is monotone toward the market:
0.7083 at pure model → 0.6787 at the old 0.60 blend → **0.6724 at pure market**.

---

## J. League-specific results

Unchanged, and still noise. Permutation test on the league-level ROI spread:
observed spread **81.6pp, p = 0.407**. With 1,018 picks over 30+ leagues, a spread
that large is what random assignment produces. No league gate was added and none
should be.

---

## L. Changes reverted or not applied because they failed

Recording these matters as much as the accepted ones.

1. **Elo season regression 0.33 → 0.0.** Best in the sweep (log-loss 1.0253 vs
   1.0279) but the paired bootstrap gave **+0.0026 nats, CI [−0.0001, +0.0051]** —
   inconclusive. Per Phase 19, **not applied**. Elo keeps `reg=0.33`.
2. **Elo K-factor and home-advantage changes.** K=32 and ha=65 were already the
   best of the tested values. No change.
3. **Poisson half-life 730 over 540.** 1.0314 vs 1.0315 — indistinguishable. Took
   the shorter memory rather than claiming a 0.0001 improvement.
4. **Removing `min_expected_value` and `min_confidence` outright.** The holdout
   says both cost money (−2.87pp and −2.18pp of holdout ROI). But removing them
   means betting nearly everything, which is a *volume* decision with real
   bankroll consequences, not a modelling one. Left in place and flagged in
   section M instead — this is the user's call, not a change to make silently.
5. **Understat xG integration.** See section K.
6. **Applying migration 003 to production by hand.** Not needed — the columns are
   nullable and `_migrate_missing_columns` adds them on the first CI invocation.

### Gate validation results (Phase 4)

Train 2026-02-28..04-26 (n=509), validate ..06-17 (n=254), holdout ..08-05
(n=255). **Not one gate survived.**

| gate | holdout cohort | cohort ROI | 95% CI | gate's effect on holdout ROI |
|---|---|---|---|---|
| `min_expected_value` (EV<5%) | 167 | **+1.0%** | [−12.6%, +13.9%] | **−2.87pp** |
| `min_confidence` (prob<55%) | 102 | +2.8% | [−16.5%, +22.0%] | −2.18pp |
| `club_pick_min_ev` (EV<−5%) | 74 | **+4.3%** | [−14.6%, +21.6%] | **−1.95pp** |
| `club_pick_min_blend` | 90 | +0.4% | [−22.0%, +22.0%] | −0.46pp |
| `club_btts_yes_ban` | 11 | −20.6% | [−69.4%, +28.9%] | +0.91pp |
| `split` agreement | 55 | −14.4% | [−39.3%, +11.2%] | +3.82pp |
| `exclude_over_3.5` / `under_2.5` / `under_3.5` | 0–4 | — | — | untestable |

The last row is instructive: an *active* gate produces no holdout cohort, so it
can never be validated from production data. That is why these needed a registry
rather than another comment.

All six edge gates are now **off by default** behind `betting.gates.*`.

---

## K. Remaining weaknesses

1. **The market still wins.** After every change, pure market consensus (0.9906)
   still beats the shipped configuration (0.9926). The gap is no longer
   significant, but it is not negative either. The model has stopped subtracting
   value; it has not started adding any.
2. **CLV is still unmeasured.** The infrastructure landed today; the first data
   point arrives the first time `capture_closing_lines.py` runs before a kickoff.
   Until then, no claim about edge can be validated.
3. **Data coverage is unchanged and remains the binding constraint.** xG 1.9%,
   shots 2.4%, injuries 17 rows, venue 8%. Half the advertised feature set is
   computed, pruned as sparse, and discarded every run.
4. **Understat is not the easy win Stage 1 implied — I was wrong to size it.**
   Probing it today: the site responds (HTTP 200, correct page titles) but the
   `datesData = JSON.parse('...')` payload that every documented scraper —
   including `soccerdata` — depends on is **absent**. Pages are 18.6 KB with zero
   occurrences of `datesData`, `teamsData` or `playersData`, so the data is no
   longer server-rendered. Integration would mean reverse-engineering an
   undocumented endpoint. Per Phase 11: **do not make it a critical dependency,
   and do not promise a coverage figure.** My Stage 1 report's "~60%" was based on
   Understat's documented league coverage, not on a verified integration path.
   Treat it as unquantified until someone gets a working extraction.
5. **1,018 settled picks is too few to validate anything.** Every gate CI in
   section L spans 30+ percentage points. At ~200 picks/month this does not
   improve quickly.
6. **The `ml` model still contributes nothing measurable.** It has no weight-learner
   observations, its pickles are from 2026-03-31, and `feature_list.json` is
   missing. The look-ahead guard (B3) will keep it honest once retraining resumes,
   but the underlying issue is coverage (3).

### Phase 16 — retraining schedule: premise was already false

The prompt states "training currently runs every day despite configuration
indicating 3 days". That was true and has already been fixed, in commit `b3dd1e1`.
`--train` checks `_ml_models_stale(max_age_days=models.ml_retrain_days)` at
`betting_agent.py:4181` and skips when fresh; `tests/test_train_scheduling.py`
covers it. Verified, no change made.

---

## M. Recommended production configuration

Applied to `config/config.yaml` and documented in `config.example.yaml`:

```yaml
models:
  bookmaker_blend_weight: 0.80    # was 0.60
  strength_half_life_days: 540    # was 180
  dixon_coles_rho: 0.0            # was -0.13
  dc_rho_per_league: false        # new — was implicitly on
betting:
  gates:                          # new — all six edge gates off
    club_btts_yes_ban: false
    club_pick_min_ev: false
    club_pick_min_blend: false
    split_agreement_low_conf: false
```

**Two things I did not change, and think you should decide on:**

- **`min_expected_value: 0.05` and `min_confidence: 0.55`.** The holdout says both
  cost money. But EV is not a value signal here (Stage 1: the EV quintiles are
  non-monotonic and the middle quintile ran −19.3%), so these are really volume
  controls. Removing them roughly triples pick volume, and at a true edge of
  −vig, more volume is more loss. The coherent options are (a) leave them as
  volume caps and stop calling them value filters, or (b) go to paper-trading
  mode until CLV data exists. I would not simply delete them.
- **Schedule `capture_closing_lines.py`.** Nothing in this stage produces CLV data
  until this runs before kickoffs. A cron entry ~60–90 minutes out, after the odds
  refresh, is the practical setup on free data.

---

## Does the evidence now justify allowing the model to influence bookmaker probabilities?

# INSUFFICIENT EVIDENCE

**Why.**

The model no longer *demonstrably harms* the forecast, which is a real change: at
the 0.60 blend it was significantly worse than pure market (CI [+0.0026,
+0.0122]); at 0.80 it is not (CI [−0.0002, +0.0048]). And the combined Stage 3
configuration is significantly better than what shipped this morning (+0.0079
nats, CI [+0.0037, +0.0122]).

But "no longer significantly worse" is not "better". The point estimate still
favours pure market on every metric — log-loss 0.9906 vs 0.9926, Brier 0.5917 vs
0.5923, accuracy 52.9% vs 52.3%. Not one configuration tested beat the market, at
any blend weight, on any of the five windows. The honest reading is that the 20%
model share currently buys nothing measurable and costs nothing measurable.

More importantly, **the decisive test cannot be run yet.** Whether a model adds
information to a market price is answered by closing line value, and this system
has never recorded a closing line. The infrastructure for it landed today
(migration 003, `capture_closing_lines.py`), and the first meaningful CLV series
is weeks away. Until then, any claim of edge rests on 1,018 settled picks whose
segment confidence intervals span 30+ percentage points — the sample that produced
the fifteen noise-fitted gates this stage just retired.

The 20% weight is therefore justified as an *option*, not as an edge: it is the
largest model share the data cannot rule out, it keeps a measurable model
contribution flowing into the weight learner, and it is cheap to withdraw. Revisit
when there are ~200 picks with a stored closing line. If CLV is persistently
negative at that point, the answer becomes NO and the correct configuration is
`bookmaker_blend_weight: 1.0` — a market-consensus price-shopper, which given
everything measured here is a perfectly respectable thing for this system to be.
