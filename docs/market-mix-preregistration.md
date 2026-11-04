# MARKET MIX — PRE-REGISTRATION

**Committed 2026-11-05, before the restricted population's ROI was read.**

---

## H-MIX — the hypothesis

> **Restricting the pick population to markets whose MEASURED cost is ≤ 1.5%
> improves ROI by approximately the cost difference, independent of any
> edge.**

If the cheap markets cost ~1.1% and the expensive ones ~6–7%, the predicted
improvement is **of the order of 5 points of ROI**, and it is predicted to
arrive whether or not the model has any edge at all.

## WHY THIS IS NOT A DATA-MINED SELECTION

**The market is selected by its MARGIN, which is a property of the price, not
by its observed return, which is a property of the outcome.** The overround is
computed from the quoted odds and is knowable before any bet settles; it
would be the same number if every result had gone the other way.

That is the whole difference from the selectivity test this project already
ran, which found **11 of 1,000 interval comparisons separable against 50
expected by chance** — that test ranked segments by their *realised* ROI, so
the winners were selected on the noise they contained. Ranking by cost cannot
do that: cost carries no outcome information.

**The mechanism is arithmetic.** For a flat stake,
`ROI = p·o − 1`. The overround is the amount by which `Σ 1/o` exceeds 1, so a
market with a 6% margin prices every selection about 5 points worse than one
with a 1% margin for the same true probability. Paying less for the same bet
is not a strategy discovery.

## THE DECISION RULE, stated before the number is read

| result | decision |
| --- | --- |
| restricted ROI improves by **≥ half** the cost difference, with a fixture-bootstrap CI excluding zero | the restriction is **supported** and goes to Niki as a cohort decision |
| improvement is **positive but < half** the cost difference, or its CI includes zero | **not adopted.** The cost argument is incomplete and the gap is reported as unexplained |
| restricted ROI is **not better**, or is worse | **refuted as stated** — see below |

**No threshold discovered in the measurement may justify a decision in this
stage.** The 1.5% cut is fixed here, before looking; it is not re-chosen on
where our markets happen to fall.

## WHAT WOULD REFUTE IT

> **If the restricted population's ROI is not better by roughly the cost
> difference, then the model's edge differs BY MARKET and the cost argument is
> incomplete.**

That is a **finding, not a failure**. It would mean the expensive markets were
carrying some compensating edge — the model may be genuinely better at Team
Goals than at 1X2 — and that cost and edge are not separable the way this
hypothesis assumes. It would redirect the work from execution to
per-market calibration.

## THE COUNTERFACTUAL LIMITATION, stated in advance

This is a **counterfactual on an existing record, not a backtest of a
strategy.** The selections were made by a model that scored **all** markets
and ranked them against each other; restricting the population after the fact
**does not establish that the same model would have produced those selections
as its first choices** from a cheap-markets-only universe.

**For the counterfactual to transfer, all of the following would have to be
true:**

1. the model's per-market probability estimates are unchanged when the
   expensive markets are removed from the candidate set — i.e. the markets are
   scored independently and not normalised against one another
2. the EV threshold, the daily exposure cap and the per-match pick cap would
   admit the same cheap-market selections once the expensive ones no longer
   compete for those slots — removing competitors can only **add** picks, so
   the restricted record is a **lower bound** on volume and the realised mix
   would differ
3. the correlated-pick filter would not drop different picks when the
   candidate set changes
4. the cheap markets' liquidity and limits are no worse than the expensive
   ones at the prices actually reachable

None of those four is established by this measurement, and 2 and 3 are known
to be false in detail, because both filters operate on the joint candidate
set. **The honest reading is a cost arithmetic check, not a forecast of
returns.**

## POPULATIONS AND METHOD

Paper and live are **never pooled** (`is_paper` is collinear with time: live
ends 2026-08-10). Settled picks only, `disposition IS NULL`,
`result IN ('win','loss')`. Bootstrap **resamples fixtures**, 4,000 draws.
Win rate, ROI and average taken odds are reported **together**, always, with
the measured cost in the same table. External figures keep their source,
date, bookmaker, league set and per-band n.

**The vocabulary reconciliation is part of the method, not a convenience:**
`saved_picks.selection` and `odds.selection` name the same bet differently
("Home Win" vs "Home", "Over 2.5 Goals" vs "Over 2.5"). The mapping is read
**from the pricer's own `selection_map`**, parsed out of
`value_calculator.py`, so the analysis cannot use a second vocabulary that
drifts from the one that set the prices.

## WHAT THIS DOES NOT DO

Adopts nothing, changes no config, moves no threshold, spends no credits, and
touches no registered statistic. ρ=0.42, σ, `TARGET_N`, the floor and the
168-credit reservation are unrelated and untouched.

*Committed 2026-11-05 before the restricted ROI was computed.*
