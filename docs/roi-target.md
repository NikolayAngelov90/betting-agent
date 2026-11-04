# THE TARGET IS ROI — registered 2026-11-04

## The identity that keeps being re-learned

> **Win rate, average odds and edge are one relationship, not three goals. A
> target that fixes win rate AND price has already fixed the edge, and the
> edge is the only one of the three that has to be earned.**

For a flat stake, `ROI = p·(o − 1) − (1 − p) = p·o − 1`. So naming a win rate
`p` and an average price `o` names the ROI exactly, and naming any two of
{win rate, price, ROI} determines the third. There is no configuration in
which a win rate is a goal independently of the price it was taken at.

The "70% at average odds 1.70" figure is therefore **+19% ROI**, and it
implies a **+12.3-point** edge over a de-vigged 1.70 price (58.8% fair).
That is **two to three times the largest externally documented
favourite–longshot gap** — `external, Footiqo, 2025/26, one bookmaker, five
leagues: +6.32pp at n=32, +3.74pp at n=134` — and it is asked for in the most
liquid part of the market, where any such gap is smallest.

## THE OPERATING TARGET: +3% ROI

Chosen because the edge it requires is **the size of the one anomaly anybody
has documented**, not because it is modest. At 1.70 it needs ~2.9 points of
edge, against +3.74pp observed at n=134 in the external study's 70–79% band.

**The target is the ROI. The pairs that deliver it are a curve, not a point:**

| average taken odds | win rate needed for +3% ROI |
| --- | --- |
| 1.55 | **66.5%** |
| 1.70 | **60.6%** |
| 1.90 | **54.2%** |
| 2.10 | 49.0% |

Any point on that curve is the same achievement. Moving along it is not
progress.

## THE STRETCH FIGURE: +10% ROI, for reporting only

Recorded so it has a name. **It sizes no experiment, sets no threshold, and
justifies no configuration.** If it appears it will appear as a surprise, and
it will be reported as one.

## THE CURRENT POSITION, ON THE SAME AXIS

`measured` 2026-11-04, grain=PICK, population=PAPER settled
(`disposition IS NULL`, `result IN ('win','loss')`), 2026-08-11 → 2026-11-03,
n=1,583 picks over 1,573 fixtures:

| | |
| --- | --- |
| ROI | **−5.68%**, 95% CI over fixtures **[−9.69%, −1.74%]** |
| win rate | **58.4%** |
| average taken odds | **1.63** |

**The interval excludes zero**, so the loss is a measured fact and not noise.

> **The distance to the operating target is 8.7 points of ROI**, from −5.68%
> to +3%. At an average price of 1.63 that is a win rate of 58.4% → 63.2%:
> **4.8 points of win rate at constant price**, or the equivalent move in
> price at constant win rate.

## HOW LONG IT TAKES TO CONFIRM — the n, derived

Distinguishing +3% ROI from zero at 80% power, α=0.05 one-sided, using the
**measured** per-pick P&L dispersion of the paper record and a bootstrap over
fixtures. The figure and its derivation are reported beside the target from
the start, because a target whose confirmation time is unknown cannot be
managed.

See the ledger entry of 2026-11-04 for the computed n and the implied
calendar time at the measured pick rate.

## PROVENANCE RULES THAT TRAVEL WITH THIS TARGET

* win rate and ROI are reported **together**, with average taken odds, never
  one alone
* every external figure keeps `external, source named, date fetched`, and the
  Footiqo gaps keep `one bookmaker, five leagues, n per band stated` — their
  own limitations section says as much and it is quoted, not paraphrased
* no ratio over a mixed population without the pooling named; paper and live
  are separate eras and `is_paper` is collinear with time

*Registered 2026-11-04. No model changed, no threshold moved.*
