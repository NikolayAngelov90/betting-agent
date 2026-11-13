# Prospective registration — `confident_value` at P_min 0.54, cohort s5.23

Committed **before the first s5.23 pick**. The switch is commit `s5.23` /
`stage5_baseline_20260807.5bca73`; the first scheduled run on it is the one
on **2026-11-15**. *(Corrected before publication: the scheduled run of
2026-11-14 started at 06:49 UTC on `e0cacb6`, the previous code, because the
switch had not yet been pushed. Its picks are s5.22. Every date below was
moved one day; no n changed.)* Everything below is fixed by the commit that adds this
file, and its hash is quoted in the ledger.

## 1. Why this rule is running

**The operator's decision, in his words: switched — not because an edge was
found. None was.** The standard rule's hit rate is bought with prices below
1.70 that he has rejected.

| Stage 31 holdout, kickoffs 2026-09-01 … 11-09, grain = pick | n | hit rate | predicted | avg odds | ROI |
|---|---|---|---|---|---|
| standard rule, as saved | 1,456 | 58.3% [55.8, 60.9] | 62.8% | 1.63 | −5.5% [−9.6, −1.2] |
| `confident_value`, P_min 0.54 | 448 | 57.4% [52.7, 61.8] | 55.6% | 1.73 | −0.5% [−8.5, +7.2] |

Fixture bootstrap, 4,000 draws. Registration of that backtest: `4fcdd83`.

## 2. What is expected

**~57% hit rate at average odds ~1.73, ROI near zero with an interval that
includes zero.** That is the claim, and the only one. It is not a claim of an
edge, and a positive return over any short period is not evidence of one.

| quantity | expected | source |
|---|---|---|
| hit rate | ~57% | holdout 57.4% [52.7, 61.8] |
| predicted hit rate (mean blended probability of the picks) | ~55.6% | holdout |
| average taken odds | ~1.73 | holdout |
| ROI per 1-unit pick | ~0, interval including zero | holdout −0.5% [−8.5, +7.2] |
| picks per calendar day | 5.6–6.4 | dev+holdout 1,440 / 255 days; holdout 448 / 70 |
| picks per day with fixtures | median 6.5, range 0–21 | holdout |
| zero-pick days | ~1 in 11 days with fixtures, almost all on days with < 10 fixtures | holdout 5 of 56, all five on < 10-fixture days |

## 3. What is measured, and how

Grain: one pick per fixture. Population: `saved_picks` with `model_version =
stage5_baseline_20260807.5bca73`, `disposition IS NULL`, `evidence_status IS
NULL`, settled win/loss (voids leave all three figures). Method: fixture
bootstrap, 4,000 draws, percentile intervals.

1. **Hit rate, average odds and ROI — always together**, and the **predicted
   hit rate beside the realised one**. The settlement message carries the
   first three for each day's batch (`settlement_summary_line`).
2. **Picks per day, including zero-pick days**, from `pick_runs` (one row per
   run: `PICKS_MADE`, `NONE_QUALIFIED`, `NO_FIXTURES`, `NO_PRICED_FIXTURES`,
   `INCOMPLETE`). A day with no row is a failed run, not a zero.
3. **The paired comparison, prospectively.** The standard rule is still
   computed for every fixture and its top pick stored in `shadow_picks`
   (never sent, never staked). On fixtures where both exist: the difference in
   hit rate, average odds and ROI, graded on read with the deployed grader.
   Hit rate is never compared alone.

**Executable:** `python -m scripts.confident_backtest --prospective` reads all three
and prints them with the separation progress below. On the night of the switch
it reads cohort `5bca73`, 0 picks, and both tables **not installed** — they are
created by the first run on the new code, and the report says so rather than
reporting zero days.

## 4. When can 57% be told apart

One-sample, two-sided 5%, at a true rate of 57%, against a **fixed**
reference (`scripts/confident_backtest.n_to_separate`, tested). Dates from
**2026-11-15**, at the holdout rate (6.4 picks per calendar day) and the
slower development-plus-holdout rate (5.6, which carries the summer break):

| told apart from | expected 95% interval excludes it (50% power) | 80% power |
|---|---|---|
| **the 53.7% zero-edge ceiling** at 1.70 | **865 picks — 2027-03-31 … 2027-04-18** | **1,785 picks — 2027-08-21 … 2027-09-28** |
| **the standard rule's 58.3%** | 5,572 picks — 2029-04-04 … 2029-07-29 | 11,319 picks — 2031-09-19 … 2032-05-12 |

> **The comparison with 58.3% cannot be decided on this horizon.** A
> 1.3-point difference in hit rate needs five to eleven thousand picks — two
> and a half to five and a half years at today's rate — and that is treating
> 58.3% as known. It is an estimate itself, so the real requirement is
> roughly double. **The comparison that matters is not hit rate alone**: it
> is hit rate at the price, which is ROI, read beside the shadow.
>
> **Against the zero-edge ceiling the answer comes in months**: by spring
> 2027 the interval is expected to exclude 53.7% if the true rate is 57%; by
> autumn 2027 with 80% power.

The 53.7% ceiling is `1 / (1.70 × (1 + 0.0948))`, the mean margin of the
taken price over the anchor's fair price on eligible holdout legs. Above
1.70 the ceiling is lower; it is quoted at the floor because that is where
the rule's picks sit (average 1.73).

## 5. A rule this registration adds

**No threshold is changed on a single period again.** The P_min 0.56 cell
read 62.6% on 147 holdout picks — the best-looking cell of the grid — and
failed its development echo: 56.1% on 244 picks, against a predicted 56.6%.
It was a registered refuter and it fired. A future change to `p_min`,
`min_price`, `require_agreement` or the blend weights needs: a
pre-registration committed before any outcome is read for it, the change to
hold on **two** periods, and its own cohort bump (every one of those keys is
in `TRACKED_KEYS`).

## 6. What would count against the rule, decided now

* **Realised below predicted** by more than the interval allows — the blend
  overconfident exactly where the rule selects.
* **The hit-rate interval's upper bound falls below 53.7%** — the picks win
  less often than a zero-edge pick at their price.
* **Average odds drift below 1.70 or the picks drift above ~2.0** — the rule
  is no longer selecting what it was registered on.
* **The shadow's ROI interval sits entirely above the new rule's** — the
  standard rule's short prices turn out to pay, against the holdout.
* **A day recorded `NONE_QUALIFIED` with many fixtures** — at P_min 0.54 the
  holdout's zero-pick days all had fewer than ten fixtures.

None of these is a trigger to change a threshold; each is a trigger for a
registered stage.
