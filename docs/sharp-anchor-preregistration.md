# Sharp-Anchor Pre-Registration — selection by price dispersion

**Registered 2026-11-12, before any match result is read for this analysis.**
The three steps completed before this document — the synthetic control, the
estimator evidence and the anchor coverage — run a query that does not select
the score columns at all (`load(with_results=False)`).

Measurement only. Nothing here adopts a rule, a venue or a threshold.

---

## 1. THE HYPOTHESIS

> **H-SHARP.** Legs on which a named bookmaker's own price exceeds the
> sharp-anchor fair price by at least the threshold return a **positive ROI**
> at that bookmaker's price.

One-sided: the alternative of interest is ROI > 0. A negative or
indistinguishable ROI is a refutation of the rule as a source of value, and is
reported as one.

**My expectation, stated so it can be held against me:** not supported. The
record so far says the market is the best model we hold and that margins
exceed every gap measured. A soft book priced above a sharp fair price is at
least as likely to be a stale or restricted line as a mistake.

---

## 2. WHAT IS ALREADY KNOWN — disclosed, because it bounds the seal

* **Outcome-free, done before this document:**
  * *the synthetic control*: each of four estimators is exact on its own
    margin model and fails the others; Shin ≡ additive for two-way sets;
    proportional fails an additive book by 1.4–2.8pp.
  * *the estimator evidence*: against the same-snapshot exchange price
    (~1% overround), **Shin** has the smallest weighted band deviation on the
    same-feed anchor — **0.100pp** against 0.193 (proportional), 0.192
    (additive), 0.210 (power), n=2,866 pairs. On the API-Football anchor
    (n=1,006, cross-feed pairs inside 120s) the evidence is **mixed**:
    proportional 0.148, Shin 0.206, with Shin flatter in the top bands.
  * *coverage*: an anchor leg set exists on 4,666 of 4,859 fixtures (96.0%).
* **Outcomes already read, on a different quantity:** the band test of
  2026-11-11 read every leg's result at the cross-book MEDIAN price over the
  whole record, including what is the holdout period here. So market
  calibration by band is known for the holdout. **The registered quantity —
  the ROI of anchor-relative edge-selected legs at a named book's price — has
  not been computed for any period.**

---

## 3. DEFINITIONS — fixed here

**Unit.** A leg: one (fixture, market, line, side, selection).

**Anchor.** Pinnacle, **from the same feed as the takeable book** —
`Pinnacle` for API-Football books, `TheOddsAPI-pinnacle` for TheOddsAPI books.
Never across feeds: within a feed 97.8–99.8% of rows sit inside 120s of each
other (median 0.0s); across feeds the median is 255s and p90 is 8.1 hours.

**Fair probability.** The **Shin** de-vig of the anchor's complete leg set.
Proportional and power are reported as **sensitivity analyses**, labelled as
such; they do not replace the primary.

**Takeable price.** A named book's **own** price for the selection, from a
complete, in-band leg set of that book, where every row of both the book's set
and the anchor's set was written within **120 seconds** of each other and
**strictly before kickoff**. No cross-time best line. The anchor itself is
never a takeable book.

**Edge.** `takeable price × fair probability − 1`.

**Thresholds.** `edge ≥ 2%`, `≥ 3%`, `≥ 5%`. **Primary: 3%.**

**Implausible edges.** `edge > 15%` is excluded from the primary as a presumed
data error (a mis-keyed line or a palpable price) and **counted**. The
uncapped figure is reported beside it as a sensitivity.

**Exchanges** (`matchbook`, `betfair_ex_eu`) are reported in their own rows and
**excluded from the pooled primary**: their commission is not in the quoted
price and their depth is `UNAVAILABLE`.

**Grading.** The deployed `_grade_selection` on the canonical selection. Legs
that cannot be graded, or are void, are excluded and counted.

**Exclusions.** `training_exclusion_reason` in {`corrupt_1x2_assignment`,
`corrupt_team_identity`, `phantom_kickoff_now_stamp`}; kickoffs that are
placeholders; leg sets outside the market's registered overround band; rows
written at or after kickoff.

---

## 4. THE PERIODS, AND THE SEAL

| | kickoff dates | role |
|---|---|---|
| **development** | 2026-02-28 … 2026-08-31 | measured first; exploratory |
| **HOLDOUT** | **2026-09-01 … 2026-11-09** | **the registered test; opened ONCE** |

The holdout is read by exactly one command, `--open-holdout`, run once after
the development figures are reported. Nothing is tuned between the two: the
thresholds, the estimator, the tolerance, the cap and the pooling rule are all
fixed in this document.

**Order of the stage:** development → holdout → and only then the residual
harness and the Elo measurements, which read outcomes across both periods.

---

## 5. THE STATISTICS

**Pooled primary.** One bet per leg, unit stake: among the non-exchange books
whose price qualifies at the threshold, the **highest same-snapshot price**.
(A maximum across books at one instant is a real takeable price; a maximum
across time is not.)

**Per book.** For each book: every leg where that book's own price qualifies.

For every row: **n, fixtures, win rate, average odds and ROI together**, with a
95% CI on ROI from a bootstrap over **fixtures** (4,000 draws).

**Also reported separately:** how many qualifying legs fall at odds
**1.50–2.20**, and their figures.

---

## 6. THE DECISION RULE — on the HOLDOUT, pooled primary, threshold 3%

* **n < 100 legs over 50 fixtures** → `INSUFFICIENT`. No verdict.
* **95% CI lower bound on ROI > 0** → `SUPPORTED on the holdout`. One
  confirmation on ten weeks; a candidate for a forward test, not an adoption.
* **CI spans zero** → `NOT SUPPORTED at this n`, with the interval stated.
* **CI upper bound < 0** → `REFUTED`.

**Multiplicity, declared.** One primary test. The 2% and 5% thresholds, the
per-book rows, the odds-range subset and the two sensitivity estimators are
**secondary and exploratory**: 2 thresholds + the books + 2 estimators is on
the order of thirty looks, so an isolated secondary interval excluding zero is
expected by chance and is not a finding. A secondary result is reported with
that label and supports nothing on its own.

**What the development period is for.** It is reported in full and decides
nothing. If development and holdout disagree in sign, the holdout governs and
the disagreement is reported as instability.

---

## 7. WHAT WOULD REFUTE H-SHARP

* the registered `REFUTED` outcome;
* a `NOT SUPPORTED` holdout following a positive development period — the
  development figure was then selection or noise;
* a positive pooled result that disappears when the anchor estimator changes
  between Shin, power and proportional — the "edge" was then the de-vig, which
  is the error this stage exists to stop repeating.

**And what a positive result would NOT establish:** that the prices are
takeable at size, that the accounts survive, or that the books are reachable
by the operator. The books supplying qualifying prices are listed and the list
stops; which are usable is the operator's answer.

*Registered 2026-11-12. `scripts/sharp_anchor.py`, `src/evaluation/devig.py`.*
