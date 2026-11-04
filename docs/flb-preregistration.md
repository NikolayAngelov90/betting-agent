# FAVOURITE–LONGSHOT BIAS — PRE-REGISTRATION

**Written 2026-11-04 and committed BEFORE our own data was queried.** The
external table below was read first; our settled record was not.

---

## THE EXTERNAL LEAD, in its own terms

`external, Footiqo, 2025/26 season, 1,727 Big Five fixtures, ONE bookmaker
(1xBet opening/closing), fetched 2026-11-04`

De-vigged implied probability against the observed outcome rate:

| band | gap (pp) | n |
| --- | --- | --- |
| 40–49% | **+1.29** | 591 |
| 50–59% | **−1.34** | 424 |
| 60–69% | **+3.21** | 266 |
| 70–79% | **+3.74** | 134 |
| 80–89% | **+6.32** | 32 |

The pattern is a favourite–longshot bias: the de-vigged price **understates**
the outcome rate, and increasingly so with probability — except at 50–59%,
where it reverses.

**Their study is one season, five leagues, one bookmaker, and they say so
themselves.** That limitation is theirs, not an inference of ours, and it is
why nothing here treats their gaps as parameters.

---

## H-FLB — the hypothesis, directional

> **De-vigged implied probability understates the outcome rate, and the
> understatement increases with probability.**

Directional, one-sided per band. The alternative of interest is a **positive**
gap; a negative gap is a different finding and is reported as a
contradiction, not as a weaker version of this one.

## THE BANDS — fixed in advance at the external study's boundaries

`[0.40, 0.50)`, `[0.50, 0.60)`, `[0.60, 0.70)`, `[0.70, 0.80)`, `[0.80, 0.90)`

**Not chosen on our data.** They are the external study's own boundaries, so
no boundary can be moved after seeing where our mass sits. Picks outside
[0.40, 0.90) are reported as an out-of-band count and tested in no band.

## DE-VIGGING — stated before measuring

The gap is observed rate minus **de-vigged** implied probability, where
de-vigging is **proportional normalisation** of the market's own probabilities
for the same market on the same fixture: `p_devig = p_implied / Σ p_implied`
over that market's selections. We use `market_probability` where it is
populated, which is this project's stored market consensus, and we report the
coverage of that column rather than assuming it.

## THE DECISION RULE

A band shows the bias **present** when all three hold:

1. **n ≥ 100 picks in the band, over ≥ 50 distinct fixtures.** Below that the
   band is reported `INSUFFICIENT` and not tested. A band with n=12 is not a
   result; the external study's own top band is n=32 and they report it with
   that caveat.
2. the gap's **95% one-sided lower bound exceeds zero**, bootstrapped over
   **fixtures** (4,000 resamples, resampling match ids and taking all their
   picks)
3. the **economic bar** below is cleared

## THE ECONOMIC BAR — the gap must survive the real price

A gap against the **de-vigged** price is not money. The price actually
reachable carries the overround, and this project's registered best-line
break-even is **+1.85%**.

> **A band passes economically only when its lower bound exceeds the
> overround measured at the prices we actually took in that band.**

The overround is measured per band from our own stored prices, not assumed,
and reported beside the gap. A gap of +3.21pp at a de-vigged price means
nothing if the reachable price carries a 4% margin.

## MULTIPLICITY — stated before testing

**Five bands, fixed in advance. One test per band. No post-hoc band
selection, no merging of adjacent bands, no re-banding on our own
distribution.** Five one-sided tests at α=0.05 give an expected 0.25 false
positives; **a single band clearing is therefore weak evidence and is reported
as such.** The pattern claim — monotone increase with probability — is the
hypothesis, and a single band in isolation does not support it.

The standard to beat is this project's own selectivity result: **11 of 1,000
interval comparisons separated where 50 are expected by chance.** Against that
baseline, anything short of several bands moving together in the predicted
direction is noise.

## POPULATIONS, SEPARATED

Paper and live are **separate** and never pooled: `is_paper` is collinear with
time here (live ends 2026-08-10, paper starts 08-11). Settled picks only,
`disposition IS NULL`, `result IN ('win','loss')`. Each reported figure names
its population, n, fixture count and date span.

## WHAT THE LIKELY ANSWER IS, SAID BEFORE LOOKING

**Insufficient n in the upper bands.** Our settled paper record is roughly
1,500 picks whose average taken odds are about 1.6 — an implied probability
near 0.62 — so the mass sits in the middle bands and the 80–89% band may be
nearly empty. **If a band is short, it is reported as short.** That is the
expected outcome and it is not a failure of the test.

## WHAT THIS PRE-REGISTRATION DOES NOT DO

It fits no model, changes no threshold, adopts no market and sizes no
experiment. ρ=0.42, σ=5.937%, `TARGET_N`, the floor and the 168-credit
reservation are untouched and unrelated.

*Committed 2026-11-04 before our data was queried.*
