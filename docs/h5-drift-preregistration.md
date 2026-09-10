# H5 — PRE-KICKOFF DRIFT: pre-registration

**Registered 2026-09-03, before any of the tests below were run.**

**H5 is the first quantity measured in this project that sits on the right side
of the vig.** Every other candidate has come in below break-even — H2 at
**+0.705%** before it turned out to be the two-way trap, MODEL CLV at
**−0.587%**. The observed drift is **+3.92%**, against a **+1.85%** best-line
break-even and a **+2%** decision-relevant threshold.

**And it does not need what H1 needs.** H1 asks whether movement predicts
movement and therefore requires three separated observations. **Drift needs
only two — a taken price and a later price — which is exactly what the current
policy produces for free.** 1,470 two-point keys are already held and the count
grows daily at zero credit cost. **Nothing here requires a purchase.**

---

## WHAT HAS ALREADY BEEN SEEN — declared, so this registration is not false

**This was measured on 2026-09-03 while deriving H1's σ, before H5 was
formulated.** It is the observation that prompted H5 and it cannot be
un-seen:

| market | selection | n (keys) | mean | sd |
| --- | --- | --- | --- | --- |
| 1X2 | Home | 283 | **+3.69%** | 9.21% |
| 1X2 | Away | 283 | **−1.82%** | 10.53% |
| 1X2 | Draw | 283 | +1.68% | 3.78% |
| over_under | Over 2.5 | 144 | −0.86% | 4.52% |
| over_under | Under 2.5 | 144 | +1.83% | 4.76% |
| over_under | Over 3.5 | 25 | −2.10% | 1.78% |
| over_under | Under 3.5 | 25 | +2.26% | 1.69% |

Fixture-level, 1X2 Home, one price per fixture: **mean +3.92%, σ 9.39%, n=15
fixtures.**

**Everything below has NOT been looked at**: the picked-versus-unpicked control,
the taken-selection direction test, and every breakdown by league, price band
and lead time. **No query answering any registered question has been run.**

---

## THE THREE QUESTIONS, and what each outcome would mean

### Q1 — Is the drift REAL, or is it selection bias?

**The control the pick population makes available:** fixtures in the **same
league on the same day** that the system did **not** pick. Snapshots are written
for every fixture in a refreshed league, not only picked ones, so the control
exists without new spend.

| outcome | condition | meaning |
| --- | --- | --- |
| **MARKET EFFECT** | picked and unpicked Home both drift, and the gap between them is **< 1 percentage point** | the drift is a property of the market, not of this system's choices |
| **SELECTION BIAS** | picked Home drifts **≥ 2pp more** than unpicked Home | the system picks Homes whose price is about to lengthen — it is systematically on the wrong side, which is itself a finding |
| **MIXED** | gap between 1pp and 2pp | both present; report both components and do not claim either |

### Q2 — Is it HOME-specific, or does everything drift the way the system took it?

**The distinction decides whether this is a market effect or a pricing
artefact of WHEN this system takes prices.**

For every pick, measure the movement of **the selection actually taken**.

| outcome | condition | meaning |
| --- | --- | --- |
| **PRICING ARTEFACT** | the taken selection drifts out by **>+1.85%** across **all** market types, Home and Away and Over and Under alike | this system takes stale or generous prices that then correct. Nothing to do with Home; it is a statement about our timing, and the remedy is when we take prices, not what we back |
| **HOME-SPECIFIC** | 1X2 Home drifts out while at least one other taken selection drifts **in** | a directional market phenomenon |
| **NEITHER** | taken-selection drift < +1.85% overall | the +3.92% was a fixture-selection artefact of the 15 |

#### Q2 IS NOT ONLY DIAGNOSTIC — it has an operational consequence, and it points the opposite way to a decision already taken

**If every taken selection drifts out in the direction it was taken, then a
LONGER LEAD IS WORSE**, because more adverse drift accumulates between taking
the price and the close.

**Stage 21 moved the cron to 03:00 partly on the opposite claim.** Recorded
there, verbatim:

> *"Lead roughly doubles, and … this is the one effect that HELPS the MODEL
> series rather than merely protecting it: a longer lead leaves more room for
> the price to move before the close, which is exactly what CLV measures."*

**"More room for the price to move" is only a benefit if the movement is
unbiased.** A **PRICING ARTEFACT** outcome on Q2 says it is not — the price
moves against the taken side systematically — and then a longer lead is a
larger loss, not a larger opportunity.

| Q2 outcome | consequence for the 03:00 cron |
| --- | --- |
| **PRICING ARTEFACT** | **the lead-time rationale REVERSES.** 03:00 still buys delay tolerance, which is measured and real, but its CLV benefit becomes a CLV cost |
| **HOME-SPECIFIC** | the rationale survives for non-Home selections and is void for Home |
| **NEITHER** | the rationale is untouched and remains untested |

> **The lead-time benefit has never been measured. It was argued, not
> demonstrated, and Q2 is the test that would settle it in either direction.**
> Recorded here before the sample completes so the result cannot be read
> selectively afterwards.

### Q3 — Does it hold, or concentrate?

Breakdowns, each reported whether or not it is flattering: **by league**, **by
price band** (odds < 2.0, 2.0–3.5, > 3.5), and **by lead time** (< 6h, 6–12h,
> 12h from pick to kickoff).

| outcome | condition |
| --- | --- |
| **BROAD** | drift > +1.85% in **≥ 60%** of strata holding ≥ 10 fixtures |
| **CONCENTRATED** | drift is carried by **< 3 leagues** or by a single price band |

**A concentrated effect is not disqualifying and is not tradable as a general
rule.** It becomes a narrower hypothesis about those strata, and the segment
warning applies: **`settled-pick-segments-are-noise` established that per-league
splits on small samples produce spurious structure** (ROI spread p=0.407). **Q3
is descriptive. No threshold will be fitted to whatever it shows.**

---

## SAMPLE SIZE AND STOPPING RULE

**Fixed now, so the analysis cannot be run repeatedly until it passes.**

* **Minimum n: 50 fixtures with a two-point 1X2 Home observation ≥30 minutes
  apart.** Currently **15**.
* **Analysis runs ONCE, when n ≥ 50 is reached.** Not before, and not again
  after a disappointing result.
* At σ = 9.39% and δ = 2%, **n=50 gives ~45% power** — under-powered for the
  decision-relevant effect and **adequate for the +3.92% actually observed**
  (n≈34 at δ=4%). **Stated in advance: a null at n=50 does NOT close the
  question**; it bounds the effect below ~4%.
* **Re-derive σ at the same time.** The H1 sample-size decision depends on it
  and it currently rests on 15 fixtures.

**Expected date: ~3 weeks (2026-09-24), at the observed accumulation rate.**

---

## WHY THIS IS WORTH REGISTERING RATHER THAN JUST RUNNING

**The direction is coherent with what is already known**, which is exactly when
a spurious result is most persuasive: if Home odds systematically lengthen
before kickoff, then backing Home early yields negative CLV — **consistent with
the measured MODEL CLV of −0.587%** on a portfolio that takes Home some of the
time.

> **A finding that explains an existing puzzle is the easiest kind to believe
> and the hardest to check.** That is the reason for fixing the outcomes now.

*Registered 2026-09-03. n=15 at registration; analysis at n≥50.*

---

# RESULT — the registered analysis, run once, 2026-09-10

**n = 129 fixtures against a registered minimum of 50. Run once. Outcomes were
fixed on 2026-09-03 and are not restated here in softened form.**

## First: the harness was defective, and that is disclosed rather than buried

The analysis executed on 2026-09-10 and **Q1 and Q2 returned no result** — not
a null, an error. Both faults were in my measurement code:

| question | fault | direction of the fix |
| --- | --- | --- |
| **Q2** | membership was tested as `(match_id, odds_snapshots.market_type, odds_snapshots.selection) ∈ {(match_id, saved_picks.market, saved_picks.selection)}`. **The two tables use different vocabularies** — snapshots say `('1X2','Home')`, `saved_picks` says `('1X2','Home Win')`; snapshots say `('over_under','Over 2.5')`, `saved_picks` says `('Over 2.5','Over 2.5 Goals')`. **The intersection is empty by construction**, so Q2 reported n=0. | fixed using the alias declarations that **already existed** — `market_spec.MARKET_SPECS.legs` and `capture_closing_lines.SELECTION_SPEC`. **No third table was introduced**; a third table is how the first two came to disagree. |
| **Q1** | the control was built with `WHERE sp.disposition IS NULL`, so fixtures whose picks had been **consolidated** counted as never-picked. | fixed — a consolidated row was still a stake that was taken. **Correcting it made the control smaller, from 1 to 0.** |

**A correction that is allowed to change a verdict needs its own control.**
Q3 was untouched by both faults, so it is that control: it must reproduce the
first execution exactly. **It does** — 6/25 leagues above +1.85%, identical
price bands, identical lead-time cells. The correction moved what it was
supposed to move and nothing else.

**Correcting a measurement defect is not re-running after a disappointing
result, because no result was produced.** The stopping rule forbids running
again *after seeing an outcome*; Q1 and Q2 had no outcome to see. The
distinction is recorded here so it cannot be claimed retrospectively.

## SAMPLE and σ

| quantity | value |
| --- | --- |
| fixtures with a two-point 1X2-Home observation ≥30 min apart | **129** (registered minimum 50 — **MET**) |
| sensitivity: reading *"two-point"* as **at least** two rather than exactly two | **also 129** — the ambiguity in my own wording turns out to be moot |
| **fixture-level 1X2 Home drift** | **+0.488%**, sd 5.937, **95% CI [−0.536%, +1.513%]** |

> ### The +3.92% that prompted H5 does not survive its own registration.
> The interval's **upper bound (+1.513%) sits below the +1.85% best-line
> break-even**, and the interval spans zero. At n=15 it was +3.92%; at n=129 it
> is **+0.488%**.

**σ re-derived, as the registration required: 5.937% at fixture level, against
the 9.39% the registration carried from n=15.** This is not cosmetic — **H1's
sample size is a function of it**:

    n = 6.185 * sigma^2 / delta^2,  delta = 2%
      sigma = 9.39%  ->  n = 137     (the figure H1 has been carrying)
      sigma = 5.937% ->  n = 55

**H1's required purchase falls by roughly 60%.** The registration flagged that
"anything materially larger than 3.3% moves the required n above 50 and the
purchase must be resized or refused" — it is still above 50, so the resize
stands, but at 55 rather than 137.

## Q1 — REAL, OR SELECTION BIAS?  → **NOT EVALUABLE**

| | |
| --- | --- |
| PICKED fixtures (1X2 Home) | n=129, mean +0.488% |
| UNPICKED fixtures (1X2 Home) | **n=0** |

**The registration asserted a fact about the pipeline that is false.** Verbatim:

> *"Snapshots are written for every fixture in a refreshed league, not only
> picked ones, so the control exists without new spend."*

**Measured:** 450 fixtures carry a pre-kickoff snapshot; 129 carry the
**two-point** observation H5 requires; **all 129 were picked.** Of the 33
never-picked fixtures holding any snapshot, **30 hold exactly one** — and one
observation cannot supply a drift measurement.

**The cause is a gate, not a shortage.** `theodds_scraper._imminent_league_fixtures`
takes `require_pending_pick: bool = True`, and `refresh_and_capture.py` — the
**only** writer of a *second* pre-kickoff observation — passes
`require_pending_pick=not args.any_fixture`, i.e. **True by default**. A fixture
gets re-priced **because** it carries a pending pick. Single snapshots for
unpicked fixtures come from the daily `update()`, once per day, which is what I
was thinking of when I wrote the sentence — and it is not the same thing.

> ### The control does not exist, and waiting will not create it.
> More time produces more **picked** fixtures. The control population is empty
> **by construction of the collection policy**, so Q1 is not
> "under-powered pending more data" — it is **unanswerable with the data this
> pipeline collects**.

**The remedy exists and costs credits:** `refresh_and_capture.py --any-fixture`
already flips the gate. Deliberately refreshing an unpicked control sample is
the only way to answer Q1, and it competes with the H1 purchase for the same
budget under the TheOddsAPI credit gate. **Not taken here; recorded as the
precondition for any future Q1.**

## Q2 — HOME-SPECIFIC, OR DOES EVERYTHING DRIFT THE WAY IT WAS TAKEN? → **NEITHER**

**Naive intervals over 814 observations would overstate precision** — the
observations are (fixture × bookmaker × selection) and one fixture's price move
appears once per bookmaker. Reported under the project's cluster bootstrap
(`src.evaluation.clv._boot`), clustered on fixture:

| taken selection | n obs | fixtures | mean | cluster 95% CI | deff | crosses +1.85%? |
| --- | --- | --- | --- | --- | --- | --- |
| **ALL taken selections** | 814 | 61 | **−0.607%** | **[−1.296%, +0.135%]** | ~11.5 | **NO** |
| taken `over_under` | 426 | 42 | +0.198% | [−0.414%, +0.825%] | ~5.5 | NO |
| taken 1X2 **Home** | 302 | 15 | −0.607% | [−1.625%, +0.649%] | ~14.2 | NO |
| taken 1X2 Away | 86 | **4** | −4.597% | **bootstrap declined — <5 fixtures** | — | **not interpretable** |

*Intervals are cluster-bootstrap estimates at seed 0.* **`analysis/h5_drift_analysis.py` is the preserved harness and reproduces every number in this section exactly** — it carries a banner stating that re-running it does not create a new verdict.

**The design effect is ~11.** Effective n on the headline row is **51.7**, not
814. Quoting the naive [−0.818%, −0.396%] would have made a clustered sample of
61 fixtures look like 814 independent draws — **the error direction that makes a
null look significant**, which is the reason `_boot` exists.

**Against the registered bands:**

* **PRICING ARTEFACT** required the taken selection to drift out by **>+1.85%
  across all markets**. Overall it is **−0.607%**, and the two markets disagree
  in sign. **Not met, and reversed in sign** — taken prices *shorten*.
* **HOME-SPECIFIC** required **1X2 Home to drift out** while another taken
  selection drifted in. **Home drifts in (−0.607%). Not met.**
* **NEITHER** — *"taken-selection drift < +1.85% overall"*. **This is the
  outcome**, and the registration wrote its meaning in advance: **"the +3.92%
  was a fixture-selection artefact of the 15."**

**The Away row is reported and refused in the same line.** −4.597% on **four
fixtures** is the exact shape of finding this project has twice been burned by,
and `_boot` declining it is the guard working rather than a gap in the output.

### Q2's operational consequence for the 03:00 cron

The registration bound Q2 to Stage 21's lead-time rationale in advance. The
**NEITHER** row reads: *"the rationale is untouched and remains untested."*

> ### The Stage 21 lead-time claim stays **UNVERIFIED**, and Q3 shows it cannot be verified from this data at all.

**Two independent reasons, and the second is the stronger:**

1. Q2 came out **NEITHER**, which the registration defined as leaving the
   rationale untested.
2. **The data contains almost no lead-time variation to test it with.** Q3's
   lead-time strata:

   | bucket | observations | **fixtures** |
   | --- | --- | --- |
   | <6h | 42 | **2** |
   | 6–12h | 2,573 | **127** |
   | >12h | 0 | **0** |

   **127 of 129 fixtures sit in one bucket.** The 03:00 cron produces a
   near-constant lead by design, so *"a longer lead leaves more room for the
   price to move"* has no contrast available to it. **UNVERIFIED here does not
   mean "not yet enough data" — it means this collection regime cannot produce
   the comparison.** Testing it requires deliberately varying when prices are
   taken, which is a policy change, not an accumulation.

## Q3 — BROAD, OR CONCENTRATED? → **NOT BROAD** *(descriptive; no threshold fitted)*

**Leagues above +1.85%: 6 of 25 strata holding ≥10 observations = 24%**, against
a registered **BROAD** threshold of ≥60%.

| price band (first price) | n | mean | 95% CI (naive) |
| --- | --- | --- | --- |
| < 2.0 | 1,191 | −0.760% | [−1.019%, −0.502%] |
| **2.0–3.5** | 1,082 | **+2.305%** | [+1.942%, +2.668%] |
| > 3.5 | 342 | −0.296% | [−1.357%, +0.765%] |

**The 2.0–3.5 band is the only cell above break-even, and it is exactly the kind
of cell this project has a standing memory about.** `settled-pick-segments-are-noise`
records that per-segment splits on small samples produce spurious structure
(ROI spread p=0.407). **Q3 was registered as descriptive with no threshold to be
fitted to it, and none is.** The band interval above is naive and unclustered;
it is printed as a description, not as evidence.

## What the run establishes

| | |
| --- | --- |
| the **+3.92%** prompting observation | **falsified at the registered n** — +0.488%, CI upper bound below break-even |
| **Q1** | **NOT EVALUABLE** — control empty by construction; the registration's premise about the pipeline was wrong |
| **Q2** | **NEITHER** — taken selections drift −0.607% (cluster CI crosses zero), not out |
| **Q3** | **NOT BROAD** — 24% of strata; one price band above break-even, not fitted |
| **σ** | **5.937%**, not 9.39% — **H1's required n falls from 137 to 55** |
| **Stage 21 lead-time claim** | **UNVERIFIED**, and untestable under the current constant-lead regime |

> **The registration did the job it was written for.** It fixed the outcome
> bands before the sample completed, so a headline that had already been
> over-claimed in this project's own ledger could be retired on evidence rather
> than defended. **And it failed in one place — a premise about the pipeline
> stated as fact and never checked — which is recorded above rather than
> quietly dropped.**

*Run 2026-09-10 at n=129. Analysis is closed. Q1 reopens only if an unpicked
control is deliberately collected.*

---

**Q1 IS FILED UNDER `UNADDRESSABLE BY SELF-OBSERVATION` — named 2026-09-10.**
It is not an under-powered control; **the policy generates the population, so
the population cannot control for the policy.** More time yields more *picked*
fixtures. Disposition (a) — `--any-fixture` — is available at credit cost and
has not been taken. Third instance of the class, alongside Stage 21's lead-time
claim and the Saturday margin boundary, with H1 reclassified into it.
See `docs/unaddressable-by-self-observation.md`.
