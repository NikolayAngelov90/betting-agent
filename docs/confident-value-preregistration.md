# Stage 31 pre-registration — confident picks at a good price

Written 2026-11-12, **before any outcome is read for this analysis**. The
commit that adds this file is the registration; its hash is quoted in the
report and in `scripts/confident_backtest.py`. Everything below is fixed by that
commit. A result obtained by a route this document does not describe is
exploratory and is labelled so.

Objective, as stated by the operator: *the safest picks that still carry a
good price. Good price = taken odds ≥ 1.70. One pick per match. No fixed daily
count. The models weighted together, not the market alone.*

## 0. What has already been seen (declared, not hidden)

The holdout is **not pristine**. Stage 30 (`7cbbddd`) opened kickoffs
2026-09-01 … 2026-11-09 once, for a different rule (best same-snapshot price,
edge against the Shin anchor). From that I know, and cannot un-know:

* the anchor is calibrated on the holdout (39.88% won vs 39.87% fair, 90,655
  pairs) — so a market-only probability is expected to be honest;
* a best-price rule at odds 1.50–2.20 returned +21.36% on n=108 — a secondary
  look, at **best** prices, not the median the pipeline takes;
* the leg-level band test read outcomes across the whole era by implied
  probability band at the cross-book median.

Nothing has been read for: a blended probability, any `P_min`, the agreement
filter, hit rate at the taken price in the 1.70+ bands, picks per day, or the
closing edge. Availability counts used to write this document read **no score
column and no pick result** (the queries select none).

## 1. Periods and population

| | kickoff dates | role |
|---|---|---|
| development | 2026-02-28 … 2026-08-31 | fitting, exploration |
| **holdout** | 2026-09-01 … 2026-11-09 | **opened once**, by `--open-holdout` |

Population: the Stage 30 loader (`scripts/sharp_anchor.load`), unchanged —
completed club fixtures (`league NOT LIKE 'world/%'`) with at least one odds
row, a real kickoff (no sub-second or pre-06:00 phantom), no
`training_exclusion_reason` in the corrupt set (contaminated 1X2, corrupt team
identity). Odds rows written at or after kickoff are dropped. Leg sets that do
not canonicalise (`market_spec` returns no spec, or an overlapping one such as
double chance) are excluded and counted.

**Leg families** (the only ones a model probability exists for):

* `1X2` — Home / Draw / Away
* `over_under` — Over / Under at lines 1.5, 2.5, 3.5, 4.5
* `btts` — Yes / No
* `team_goals` — Home / Away, Over / Under at lines 0.5, 1.5

Every other line is excluded and counted.

**Pickable legs** are the ones the pipeline can express and settle: the
17 selections of `_market_specs` in these families (1X2 x3, Over/Under
1.5-4.5 x8, BTTS x2, team Over 0.5/1.5 x4). A team-goals *Under* is priced
and enters the blend fit, but the grader does not know it, so it is never
picked - in the backtest or by the flag. A two-leg set with one gradeable
leg is graded by complement. `betting.excluded_markets` is a gate
of the **current** rule and is **not** applied to the new rule; the market mix
of every cell is reported instead.

## 2. Prices (hard rule 3)

* **Anchor**: the Pinnacle of the **same feed** (`Pinnacle` for API-Football,
  `TheOddsAPI-pinnacle` for The Odds API), de-vigged with **Shin**
  (`src/evaluation/devig.shin`) — the estimator Stage 30's control selected.
* **Taken price**: the pipeline's own policy, `_find_best_odds` — the
  **median** across non-Flashscore books, a single book allowed — restricted
  to books of the anchor's feed that carry the **complete** leg set within
  **120 s** of the anchor set (the loader's rule; the anchor itself counts as
  a book). When both feeds carry an anchor set for a leg, the feed with more
  books in that leg's median is used; ties go to The Odds API.
* Never the best price across books. The cross-feed median the live pipeline
  computes (no snapshot restriction) is reported once as a sensitivity.
* **Stated limit**: `odds` keeps the *last* pre-kickoff write. For most
  fixtures that is the morning run, which is the price the pipeline took; for
  fixtures refreshed later it is a later price. Part D measures the price
  path where one exists.

## 3. Part B — the blended probability

Sources, per leg:

| source | 1X2 | binary families |
|---|---|---|
| market — Shin on the same-feed anchor | yes | yes |
| Elo — pre-match ratings, season-boundary regression, draw curve refit monthly on prior matches only (`residual_harness.history_pass("season")`) | yes | **no opinion** |
| Poisson — `PoissonModel.fit(as_of_date=…)`, refit **weekly** (each Monday 00:00, matches before it only), `predict()` for that week's fixtures | yes | yes |
| ML | **absent** — see below | **absent** |

**ML is ABSENT, and that is stated rather than estimated.** No per-leg
out-of-sample ML probability exists: there is no predictions table, and a
model probability is stored only on the 2,961 saved picks — legs selected
*because* the model disagreed with the price. Fitting a weight on a selected
subset would measure the selection. ML's share is `UNEVALUATED`.

**Form**: a linear opinion pool per family, `p = Σ wₖ pₖ`, `wₖ ≥ 0`,
`Σ wₖ = 1`, renormalised over the leg set. Four weight vectors: `1X2`
(market, Elo, Poisson), `over_under`, `btts`, `team_goals` (market, Poisson).

**A source with no opinion is dropped, not defaulted.** Poisson has none on a
fixture when either team is absent from that week's fit; the remaining weights
are renormalised for that set, and such sets are left out of the weight fit
and counted.

**Fit**: minimise mean log-loss of the realised outcome over leg sets, by
exhaustive search on the simplex at step 0.01 (deterministic, no optimiser
state). Probabilities clipped to [1e-6, 1−1e-6].

**Reported**:

1. *Walk-forward, development only*: monthly folds, weights fit on kickoffs
   before the month, scored on the month; first scored month is the first
   with ≥ 300 training sets. Out-of-sample log-loss of the blend minus the
   market's, with a fixture-bootstrap 95% CI (4,000 draws).
2. *The registered weights*: fit on all of development; CIs from 1,000
   fixture-bootstrap refits.
3. *Holdout*: the same delta under the frozen registered weights.

**A second fit, declared now**: the same with the **deployed** Elo
(calendar-year regression, fixed draw curve). The corrected Elo is not in
production, so the wiring in Part E can only use the deployed one; the
deployed-Elo weights are what the flag would read. Both weight sets are
reported.

**If the market takes ~100%, the report says so in those words** — the models
add no measurable information beyond the price — and the blend is carried
forward anyway as the registered probability.

**Stated limit**: `PoissonModel.fit` weights by days before *today*, not
before `as_of_date`. The weights are proportional either way, so weighted
means are unchanged; anything in the fit that thresholds on an absolute weight
is not, and is not corrected here.

### Agreement

A leg is in **agreement** when every source that has an opinion ranks its
selection first within the leg set:

* `1X2`: market, Elo **and** Poisson (three sources);
* binary families: market **and** Poisson (two sources — Elo has none). These
  are reported in their own row and never counted as three-way agreement.

**Test**: among legs with taken price ≥ 1.70, hit rate with n and
fixture-bootstrap CI for agree vs disagree, and the difference with its CI —
on development and on the holdout, pooled and per family. Verdict on the
holdout pooled difference: `RAISES` if its CI excludes zero above, `LOWERS` if
below, otherwise `NO DETECTABLE EFFECT at this n`.

## 4. Parts C and H — the selection rule

An **eligible leg** has taken price ≥ 1.70 (section 2). One pick per fixture:
among the fixture's qualifying legs, the highest blended probability; ties to
the higher taken price, then to the leg key in sorted order.

Odds bands, fixed: **[1.70, 2.00)**, **[2.00, 2.50)**, **[2.50, 3.50]**.

| rule | taken price | qualifies when | grid | cells |
|---|---|---|---|---|
| **C** — unbounded | ≥ 1.70 | blended p ≥ `P_min` | `P_min` ∈ {0.50, 0.52, 0.54, 0.56, 0.58} × agreement {off, on} | 10 |
| **lane 1 — confident** | [1.70, 2.00) | blended p ≥ `P_min` | same grid | 10 |
| **lane 2 — value** | [2.00, 3.50] | blended p × taken ≥ 1 + e | e ∈ {0.00, 0.02, 0.05} × agreement {off, on} | 6 |

**26 cells.** Each is computed on development (exploratory) and on the
holdout (the decision). Lanes are reported separately everywhere and never
pooled.

Per cell and period: n picks, fixtures, hit rate with a fixture-bootstrap 95%
CI, **predicted hit rate beside it** (mean blended probability of the picks),
average taken odds, ROI at one unit per pick with its CI, market mix, and the
same four figures per odds band.

**Picks per day** per cell: median and range, zero-pick days, and the
distribution on days with < 10, 10–40 and > 40 fixtures in the population.

**The current rule, paired.** The current rule is the pipeline's saved picks
on the same fixtures (`disposition IS NULL`, valid evidence, gradeable). As of
s5.22 the saved pick is the model's pick, so the comparator is graded on the
**model** selection (`model_result` where a review changed the pick, else
`result`); the reviewed final pick is shown beside it. Reported: its hit rate,
average odds and ROI on each period and per band, and for every cell the
paired difference in hit rate on fixtures both rules picked, with a
fixture-bootstrap CI.

**The pre-experiment period** (picks dated ≤ 2026-08-05; the count is reported
against the 1,018 named in the directive) restated per band with today's
gates applied to the stored fields: odds in [1.50, 10.0], EV ≥ 0.05,
probability ≥ 0.52, and probability ≥ 0.55 or EV × probability ≥ 0.038,
market not in `excluded_markets`.

**The blocker, stated before it is measured.** `max_odds` 10.0 permits long
prices; the confidence gate removes them. `min_confidence` is 0.55 with a
sliding scale down to a **hard-coded** floor of 0.52
(`value_calculator.high_ev_min_confidence`). At odds 2.00 a probability of
0.52 is a claimed edge of +4%; at 2.50, +30%; at 3.50, +82%. Reported: legs
per day, per band, that clear every price and EV gate and are removed by the
confidence gate alone — computed with the **blended** probability standing in
for the ensemble's, and labelled `simulated` for that reason.

**The arithmetic ceiling.** A leg with no edge at taken price `o` wins
`1 / (o × (1 + m))`, where `m` is the margin of the taken price over the
anchor's fair price. At `o` = 1.70 that is 58.8% at `m` = 0 and about 56% at
`m` = 5%. The measured `m` on eligible legs is reported; a realised hit rate
above the ceiling is either an edge or noise, and the CI says which it can be.

## 5. Part D — are the generous prices real

Population: fixtures in `odds_snapshots` (append-only since 2026-08-25) where
`TheOddsAPI-pinnacle` was observed at two or more pre-kickoff instants at
least **one hour** apart. Counted before writing this: **319 fixtures**, all
in September–November 2026, all on The Odds API feed (the API-Football anchor
has one observation per fixture, so it contributes none). Markets: `1X2` and
`over_under`, the two that feed carries.

* instant `t₀` = the fixture's **first** anchor snapshot;
* taken price at `t₀` = the median across that feed's books within 120 s of it;
* a leg is **generous** when taken × Shin fair at `t₀` − 1 > 0;
* **closing edge** = taken(`t₀`) × Shin fair at the anchor's **last**
  pre-kickoff snapshot − 1.

Primary: generous legs with taken ≥ 1.70 — mean closing edge with a
fixture-bootstrap CI, beside the mean instant edge. Secondary: all prices;
and **per book**, each book's own `t₀` price in place of the median.

Verdict: `REAL` if the CI of the mean closing edge lies above zero; `SLOW` if
it lies below zero; otherwise `UNRESOLVED at this n`. The retained share
(closing ÷ instant) is reported either way. No outcome is read in this part.

**Quote age.** From the commit that lands Part A each snapshot carries the
book's own `last_update`. Age = `observed_at − book_last_update`; **fresh** =
age ≤ 600 s. A snapshot with no `last_update` is `UNEVALUATED`, never fresh.
Today the fresh subset has n = 0 by construction and is reported as that.

## 6. Part F — the decision table

One row per cell, grouped by rule, lanes never pooled: holdout hit rate with
CI, predicted hit rate, average odds, ROI with CI, picks per day (median and
range). **Nothing is recommended.** One cell per rule group (C, lane 1, lane 2)
is *marked*: the highest
hit rate whose 95% CI lower bound exceeds the current rule's holdout hit rate
(model pick, pick grain) — if any exists. A mixed card is shown as arithmetic
only: lane 1 share × lane 1 hit rate + lane 2 share × lane 2 hit rate, at
shares 90/10, 75/25 and 50/50.

## 7. Multiplicity (hard rule 1)

| family of comparisons | count |
|---|---|
| selection cells, holdout | 26 |
| blend vs market, per family | 4 |
| agreement, pooled + per family | 5 |
| closing edge, primary | 1 |
| **declared** | **36** |

At 95% that is **1.8 expected false positives** if nothing is true. The mark
in section 6 uses the 95% interval the directive names; the Bonferroni
interval for 26 cells (99.81%) is printed beside it, and a mark that does not
survive it is labelled as not surviving. Band splits, per-book rows,
development figures and the picks-per-day tables are descriptive.

## 8. What would count against the rule

* predicted hit rate above realised in most cells — the blend is
  overconfident exactly where the rule selects (a winner's curse);
* a marked cell on the holdout that development does not echo;
* a hit rate that rises with `P_min` on development and not on the holdout;
* generous prices whose closing edge is not positive — the qualifying legs
  are then the ones the median was slow on;
* a result that depends on the Kambi-family books Stage 30 could not explain;
  every cell is re-run without them and shown.

## 9. My prediction, written before the run

The market takes at least 90% of the weight in every family and the blend's
log-loss is indistinguishable from the market's. Because the blend is then
nearly the anchor's fair probability, a leg qualifies at `P_min` ≥ 0.54 only
where the median price is generous against Pinnacle; realised hit rates sit at
or below predicted and under the 58.8% ceiling; no cell's lower bound clears
the current rule's hit rate once the 26-cell multiplicity is counted; lane 2
has too few picks to say anything. If that is wrong, this paragraph is the
record of what I expected.
