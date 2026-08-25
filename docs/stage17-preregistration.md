# Stage 17 — Pre-Registration

**Written and committed BEFORE any analysis.** Nothing in this file was chosen
by looking at a relationship between a predictor and the target.

## What was looked at before writing this

Full disclosure, because "pre-registered" is worthless if the boundary is vague.
Exactly one query was run before this file was written, returning only substrate
extent needed to choose a date split:

```
odds rows                     365,181
with opening_odds             323,362  (88.5%)
opening_odds <> odds_value     21,295  (5.8%)
date range                    2026-02-28 .. 2026-08-24
matches / bookmakers            3,246 / 59
```

No predictor, no target, no relationship, no per-league or per-book breakdown.

## Held-out period

| | |
| --- | --- |
| **TRAINING** | `2026-02-28` .. `2026-06-30` inclusive |
| **HELD OUT — SEALED** | `2026-07-01` .. `2026-08-24` inclusive |

Split by match kickoff date (`matches.match_date`), not by row timestamp, so a
fixture cannot straddle the boundary.

The sealed period will not be queried, described, counted, or used for feature
selection or target definition until Part D. Part A's substrate description may
report **row counts** for the sealed period (needed to state coverage honestly)
but no distributional or relational statistic.

## The target

**M = O_early / O_late − 1**, in percent.

- `O_early` = `odds.opening_odds` — first price this system saw for that
  `(match, bookmaker, market, selection)`.
- `O_late` = `odds.odds_value` — last price this system saw for the same key.
- **M > 0 means the price SHORTENED** (odds fell) after first sight. That is the
  direction that produces positive CLV for a backer who took `O_early`.

Same units and same sign convention as `clv.price_clv = taken/closing − 1`, so
every number here is directly comparable to Stage 16's thresholds.

**Market:** `1X2` only, selections normalised across the two vocabularies known
to exist (`Home`/`Home Win`, `Away`/`Away Win`). 1X2 is chosen because it is the
only market with a complete three-way book, which H2 requires for an overround.

## Hypotheses

Each is falsifiable and directional. **No hypothesis may be added later, and a
hypothesis not in this list may not be tested in Part D.**

- **H1 — MOMENTUM.** Prices that have already moved continue to move the same
  way. *Direction:* among selections in the **top quintile** of absolute prior
  drift, mean M is positive when prior drift was a shortening, negative when it
  was a drifting-out.
- **H2 — CROSS-BOOK DISAGREEMENT.** A price that is an outlier against the
  cross-book consensus reverts toward it. *Direction:* for selections whose
  `O_early` is in the **top quintile** of "above cross-book median implied
  price", mean M > 0 (the outlier-high price shortens).
- **H3 — INJURIES.** Injury records move lines against the affected team.
  *Direction:* for matches with ≥1 injury record for one team and 0 for the
  other, mean M > 0 on the **opposing** team's selection.
- **H4 — TIME.** Movement magnitude grows with elapsed time between the two
  observations. *Direction:* mean |M| increases monotonically across quintiles
  of elapsed time. (Also serves as Part B2's nuisance null: movement explained
  by the passage of time alone.)

**Subset definition is fixed at QUINTILES (top 20%) for every continuous
predictor**, chosen a priori. No other cut-point may be used. This is the rule
that prevents Stage 17 from producing the next generation of the ~15
data-fitted thresholds the 2026-08-07 audit found.

## Decision rule

Carried over from Stage 16, unchanged:

| threshold | value | provenance |
| --- | --- | --- |
| best-line break-even | **+1.85%** | MEASURED 2026-08-25, 3,241 matches |
| minimum decision-relevant | **+2.00%** | DERIVED, Stage 16 Part B |
| comfortable | +4.00% | DERIVED, Stage 16 Part B |

- **SIGNAL** — at least one hypothesis's pre-registered subset shows mean M with
  a **95% CI lower bound above +1.85%**, in the training period **and** the
  held-out period, with the direction as stated above.
- **NO SIGNAL** — no subset clears that in training; **or** a training candidate
  fails to replicate held-out. *A held-out contradiction means the training
  result was noise. No adjust-and-re-run.*
- **Significance** is reported at Bonferroni-corrected α = 0.0125 (4
  hypotheses). The economic threshold is binding regardless: a statistically
  robust +0.3% effect is not a finding.

## Capacity requirement

A candidate must apply to **≥10 fixtures per month** to count as actionable.
Below that it is reported as a curiosity. A +3% effect on four fixtures a month
is a different proposition from the same effect across the card.

## Known threats, registered in advance

1. **`opening_odds` is first-sight-by-this-system, not the market open.** M is
   therefore a lower bound on true market movement, and is biased by when this
   system happened to look.
2. **Survivorship.** `prune_old_odds(keep_days=400)` preserves rows for matches
   with saved picks. Any population older than the prune horizon over-represents
   matches this system bet on. To be quantified in Part A2.
3. **THE HABIT.** Two selection vocabularies are known. Any query returning
   implausibly few rows is a vocabulary bug until proven otherwise.
4. **Overwrite semantics.** `odds` is unique on
   `(match_id, bookmaker, market_type, selection)` and is overwritten on refresh.
   There are exactly two observations per key, never a path.

---

*Committed before analysis. Stage 17, 2026-08-25.*
