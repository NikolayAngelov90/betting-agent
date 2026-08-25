# Stage 17 — Is There Any Signal Left?

Stage 16 is closed: `STAGE 16 — CHECKPOINT DERIVED`. Four independent tests now agree that the model adds nothing over the market price, and the fourth was the sharpest: betting the close at retail prices predicts −4.64% ROI, the settled record shows −5.396% over 1,320 picks, and the 0.76pp difference — the model's entire contribution — sits inside its own standard error.

Niki has decided to return to the predictive core. This stage begins that, and it begins by refusing the obvious move.

**The obvious move is to build a better model. Do not.** The 2026-08-07 audit established that this model's probabilities are indistinguishable from a constant *and worse than raw `1/odds`*, and that the bookmaker blend sweep improves monotonically to `w = 1.0`. That is not a model that needs tuning. It is a model whose information content is negative relative to the market, produced by exactly the process — more features, more ensemble members, more tuned thresholds — that another round of improvement would repeat.

So this stage asks the prior question:

> **Is there any information available to this system that is not already in the price it can take?**

If the answer is no, no amount of modelling helps and the project is finished as a betting system. That is a legitimate outcome and it must be reachable from this prompt.

---

## 0. The reframe this stage rests on

The system does not need to predict match outcomes. It needs to beat **the price it can actually take**.

Those are different problems, and the second is strictly easier:

* predicting outcomes means beating the closing line, which aggregates every participant's information. Four tests say this system cannot.
* beating the taken price means identifying prices that will **move in your favour before the close** — which is what CLV measures, and what the instrument built in Stages 13–16 now measures honestly.

The current system takes prices at ~10:30 UTC and the close arrives hours later. Every pick is therefore already a bet on line movement, made by a model trained on outcomes. Nobody has asked whether anything in this data predicts *movement*.

**And movement research is nearly free.** Outcome research needed settled results; movement can be studied on every match that has an opening price and a later price — tens of thousands of rows already in the database, at **zero API credits**. The credit constraint that dominated Stage 15 does not apply to this stage at all.

**Stage 16 also removed the other constraint.** Excluding a decision-relevant effect needs ~17 observations at 80% power, not 500. So a candidate that works can be validated in weeks, and one that does not can be killed in weeks. The seven-month arithmetic is gone.

---

## 1. Pre-registration — write this before you look at anything

The audit's second-largest finding was roughly fifteen thresholds fitted to the data they were then evaluated on. This stage will produce the next generation of those unless it is pre-registered.

Before any analysis, commit to `docs/stage17-preregistration.md` and commit it to git:

* **the held-out period**, defined by date, and untouched until Part D
* **the hypotheses**, each stated as a falsifiable claim with a direction
* **the decision rule** — what result would make you say "there is signal" and what result would make you say "there is none"
* **the effect size that matters**, carried over from Stage 16: a predictable movement component must clear the overround to be worth anything. Best-line break-even is +1.85%; the minimum decision-relevant CLV is +2%.

State in the report that this file was committed before the analysis, and give its commit hash.

Nothing in this stage may be justified by a threshold discovered during it.

---

## PART A — What data actually exists, and what it cannot support

No modelling. Establish the substrate and its limits.

### A1. The price history is thinner than it looks

`odds` is unique on `(match_id, bookmaker, market_type, selection)` and is **overwritten on every refresh**. There is no price history — only `opening_odds`, frozen at first sight, and whatever the current row holds.

Establish precisely:

* what `opening_odds` means operationally — first sight by this system, which is not the market's open. How long after the market opened does this system typically first see a price?
* how many matches have both an opening and a materially later price, and what the distribution of the interval between them is
* whether `opening_odds` is per-book, and whether books' first-sight times are comparable
* what fraction of the odds history is usable for movement research after those constraints

### A2. Survivorship and selection

The odds table prunes rows older than 400 days, preserving rows for matches with saved picks. That is a selection mechanism: matches the system bet on retain their odds; others may not.

Quantify it. Any movement study run on the surviving population without accounting for this is measuring the system's own pick selection, not the market.

### A3. State what the substrate can and cannot answer

One paragraph, plainly. If the honest answer is that this data can only support a weak version of the question, say so now rather than discovering it in Part C.

---

## PART B — Is movement predictable at all?

### B1. Define the target precisely

Movement from the price this system could have taken to a later price, expressed in the same units as CLV so it is comparable to Stage 16's thresholds. Say exactly which two prices, at which times, for which market, and why.

### B2. Establish the null first

Before testing any predictor, establish what "no skill" looks like:

* the unconditional distribution of movement — mean, spread, and whether it is centred on zero
* what fraction of movement is explained by the passage of time alone
* the base rate a coin-flip predictor would achieve

Stage 16 found a 21.7% point mass at exactly zero in the CLV data and investigated it before accepting it. Apply the same suspicion here: a movement distribution with unexpected structure is a data artefact until proven otherwise.

### B3. Test the predictors the system already has

Use the existing 14 feature sections. Do not build new ones yet — the question is whether *anything already computed* carries movement information, and it is a cheaper question than whether something new might.

Give particular attention to the sections that have a prior reason to relate to movement rather than to outcomes:

* **odds movement** — opening vs current change, direction, magnitude. Momentum in prices is the most-documented effect in this space and the system already computes it while using it to predict outcomes instead.
* **injuries** — Stage 14 established these reach only the Claude review prompt and never the model. If injury news moves lines, this is information the system holds and does not use.
* **cross-book disagreement** — the de-vigging layer already computes per-book overrounds. Disagreement between books is a candidate signal that has never been tested as one.
* **timing** — how long before kickoff the price was taken.

Report each as a measured effect with its confidence interval, on the training period only. The held-out period stays sealed.

### B4. The honest null result

If nothing clears noise, say so. The report should be as short and as clear in that case as in any other, and it should state what would have had to be true for the answer to be different.

---

## PART C — Is any predictable component large enough to matter?

Statistical significance is not the bar. Stage 16 set the bar and it is economic:

* a predictable movement component must exceed the **overround** to be worth acting on — best-line break-even +1.85%, minimum decision-relevant +2%, comfortable +4%
* it must survive the prices this system can actually reach, not idealised ones
* it must be large enough to stake meaningfully under `kelly_fraction: 0.25`

For any candidate that clears B, report its size against those thresholds. A statistically robust +0.3% effect is not a finding; it is a smaller version of the same nothing.

Also state the capacity honestly: how many picks per month would such a signal apply to? A +3% effect on four fixtures a month is a different proposition from the same effect across the card.

---

## PART D — The held-out period, and only now

Open the sealed period. Run exactly the tests pre-registered in §1, with no additions.

Report:

* each pre-registered hypothesis, its pre-registered decision rule, and the result
* every difference between the training-period result and the held-out result
* anything you wanted to test and did not, because it was not pre-registered — list it for a future stage rather than running it

**If the held-out result contradicts the training result, the training result was noise.** Say so and stop. Do not adjust and re-run; that is how the fifteen thresholds happened.

---

## PART E — What this stage recommends, and what it does not do

No code changes. No config changes. No new model. No touching the frozen model, the ensemble, the blend weight, or any threshold. `paper_trading_mode` stays `true`.

Produce one of three recommendations, with the evidence attached:

1. **There is a candidate signal.** Describe it, its measured size against the economic threshold, its capacity, and what a minimal test of it would look like — remembering that ~17 observations suffice to exclude a decision-relevant effect, so a live test is weeks, not months.
2. **There is no signal in this data, but there is a specific reason to think one exists elsewhere.** Name the data that would be needed and what it would cost.
3. **There is no signal and no identified route to one.** Then the honest conclusion is that this system cannot beat the market with the information available to it, and the project is complete as a betting system. Say it plainly, with the four independent confirmations behind it, and let Niki decide what the software becomes.

Outcome 3 is not a failure of this stage. It is the answer four earlier tests have been pointing at, stated once and for all with the price data included.

---

## F. Hard rules

1. **Pre-registration before analysis.** Commit hash in the report. A hypothesis not in that file may not be tested in Part D.
2. **The held-out period stays sealed** until Part D. Not peeked at, not used for feature selection, not used to choose a target definition.
3. **No number without provenance** — `measured`, `simulated`, `assumed`, `from literature` — with a date.
4. **No threshold discovered during this stage may be used to justify a result in it.**
5. **A single anomalous result is evidence about the measurement first.** Investigate distributional oddities before accepting them, as Stage 16 did with the point mass at zero.
6. **One definition, and a guard.** Watch for THE HABIT in the data layer specifically — Stage 16 found `odds.selection` carrying both `Home` and `Home Win`, and Stage 15 found two market taxonomies. Any query in this stage that returns implausibly few rows is a vocabulary bug until proven otherwise.
7. **No production changes at all.** This stage reads. If something needs writing, it needs a different stage.
8. If any invariant in `tests/test_experiment_invariants.py` fails: **STOP, do not fix, report.** Invariants 2 and 3 are known defective; the declaration must not cite the invariant count.

Declare:

`STAGE 17 — SIGNAL FOUND` / `STAGE 17 — NO SIGNAL, ROUTE IDENTIFIED` / `STAGE 17 — NO SIGNAL, NO ROUTE`

whichever the evidence supports, and none of them if the substrate cannot support the question — in which case declare `STAGE 17 — SUBSTRATE INSUFFICIENT` and say what would be needed.
