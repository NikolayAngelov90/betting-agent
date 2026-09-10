# DOES THE CLAUDE REVIEW ADD ANYTHING?

**Measured 2026-09-10 from data already held. Nothing was built. No threshold
was fitted. Every interval is cluster-bootstrapped on FIXTURE, and the paired
difference is reported as an OBSERVED DIFFERENCE, never as a causal claim.**

---

# PART 1 — MAKING THE REVIEW'S ABSENCE MEASURABLE

**`review_action = none` currently conflates several distinct states, and the
2026-08-25 audit found 35 picks it could not explain.** This establishes the
size of the gap and whether the logs can separate the causes retrospectively.
**It proposes a field; it does not add one.**

## The gap is 142, not 979

| | |
| --- | --- |
| picks total | 1,761 |
| `review_action IS NULL` | **979** |
| …of which **predate the feature** (first review 2026-07-08) | **837** |
| **…NULL on or after that date — the real gap** | **142** |
| …of the 142, PAPER picks (the experiment) | **139** |

> **837 of the 979 are not a gap at all — the review did not exist yet.**
> Quoting 979 as "unexplained" would have been the same error as reading a bare
> count without its gate. The number that matters is **142**.

## Every one of the 142 is retrospectively classifiable

**Cached CI logs reach back to 2026-03-01, and every date carrying a
post-feature NULL has one.** Classification is per-PICK, joining each NULL pick
to the briefing lines for its own fixture:

| cause | picks | paper | evidence in the log |
| --- | --- | --- | --- |
| **`auth_unavailable`** | **121** | 121 | `You've hit your session limit · resets HH:MMam (UTC)` then `Your credit balance is too low to access the Anthropic API` |
| **`decision_discarded`** | **12** | 12 | `Briefing decision [M]: action=KEEP\|CHANGE` present in the log, `review_action` NULL in the database |
| **`fixture_reviewed_other_pick`** | **9** | 6 | a SIBLING pick on the same fixture carries the verdict |
| `not_attempted` | **0** | 0 | — |

**`not_attempted` is empty. Every absence has a recorded cause.** The
information exists; it is simply not written down in a queryable form.

### Two corrections found while measuring, and both changed the answer

**1. `fixture_reviewed_other_pick` is not a failure and was nearly reported as
one.** The first pass classified 21 picks as `decision_discarded`. Checking one
— pick 1019, `SJK vs HJK Helsinki`, 2026-08-03 — showed pick **1018** on the
*same fixture* carrying `CHANGE`, which is the decision the log records.
**The review is FIXTURE-scoped: one decision per fixture, and a second pick on
that fixture receives nothing.** That is by design, and 9 of the 21 were it.

> **A cause counted per pick from a mechanism that runs per fixture will
> manufacture a defect that does not exist.** The proposed field must record
> which pick a decision attached to, or it reproduces the same confusion.

**2. The remaining 12 ARE genuine discarded decisions, and that is a real
defect.** Verified on `SK Rapid vs Paide`, 2026-08-12:

```
Claude Code briefing failed for SK Rapid vs Paide … session limit
Briefing decision [SK Rapid vs Paide]: action=CHANGE     <-- a verdict WAS produced
```

**Both picks on that fixture (1106, 1108) hold `review_action = NULL`**, while
`FC Copenhagen` (CHANGE) and `GKS Katowice` (KEEP) from the same run persisted
correctly. **A decision was computed, logged, and lost.** Recorded here; not
pursued, per the standing rule.

## The control group the absence creates

**2026-09-09 (13 picks) and 2026-09-10 (7 picks) are 100% NULL.** Both days the
subscription hit its session limit and the API fallback had no credit balance.

> ### 20 picks on which FINAL is MODEL by construction, with no intervention.
> **That is a natural control group — and it is exactly the control H5's Q1
> lacked.** Right now it is indistinguishable, in the database, from a
> pipeline failure.

**The pipeline already fails safe.** `Pick review [M]: no decision returned —
model pick left unchanged` is the correct behaviour and it happened every time.
**What is missing is the label, not the behaviour.**

## PROPOSED — an explicit outcome field, recorded where the review returns

**Not built. Proposed for a decision.**

| value | meaning |
| --- | --- |
| `not_attempted` | the review never fired for this pick |
| `auth_unavailable` | it fired and no credential path was usable |
| `attempted_failed` | it ran and errored, or returned nothing parseable |
| `decision_discarded` | it produced a verdict that was not applied |
| `keep` | applied, KEEP |
| `change` | applied, CHANGE |

**Written at the point the review returns**, not inferred later. Three
properties matter more than the enum:

1. **It must be NOT NULL with a default of `not_attempted`**, so a pick can
   never again be silent about its own review. A nullable field reproduces the
   gap it was added to close.
2. **It must record which pick a decision attached to**, or
   `fixture_reviewed_other_pick` becomes a phantom `decision_discarded` — as it
   did above, in this very analysis.
3. **`auth_unavailable` becomes a labelled cohort**, so a no-auth day is a
   usable control rather than a hole. **That is the whole value of the change:
   the 20 picks of 09-09/09-10 become a control group instead of a gap.**

---

# PART 2 — WHAT THE DUAL ATTRIBUTION SAYS

**The instrument was built for this question and has accumulated for a month.**

## The two series

| series | n | fixtures | mean price CLV | cluster 95% CI | crosses +1.85%? | crosses 0? |
| --- | --- | --- | --- | --- | --- | --- |
| **MODEL** | 105 | 105 | **+0.256%** | [−0.306%, +0.862%] | **NO** | **YES** |
| **FINAL** | 123 | 123 | **+0.130%** | [−0.413%, +0.697%] | **NO** | **YES** |

**Neither series reaches break-even and both intervals contain zero.**
Design effect 1.00 — one observation per fixture, as s5.9 now guarantees.

## The paired subset — an observed difference, not a causal claim

| subset | n | mean (final − model) | cluster 95% CI |
| --- | --- | --- | --- |
| all paired picks | 96 | **−0.101%** | [−0.561%, +0.345%] |
| **`review = CHANGE` only** | 25 | **−0.389%** | [−2.136%, +1.367%] |
| `review = KEEP` | 41 | **exactly 0** | — by construction |

**Only 23 of 96 pairs are non-zero.** An unchanged pick contributes exactly
zero because both attributions carry the same price, so **the paired mean is
diluted by every KEEP**, and the CHANGE subset is where the review actually
acted. **Both cross zero.**

## Settled picks — ROI and average taken odds, not win rate alone

| group | n | wins | win % | avg taken odds | P/L (units) | **flat ROI** | avg EV at taken price |
| --- | --- | --- | --- | --- | --- | --- | --- |
| KEEP | 444 | 258 | 58.1% | 1.619 | −27.26 | **−6.14%** | +0.62% |
| **CHANGE** | 312 | 188 | 60.3% | 1.704 | +5.43 | **+1.74%** | **−6.85%** |
| none | 132 | 74 | 56.1% | 1.669 | −9.38 | **−7.10%** | −1.32% |

> **CHANGE is the only group with positive ROI, and the only group whose
> modelled EV is strongly negative.** That inversion is the most interesting
> number here and it is also the one most likely to be noise.

**So it was tested rather than admired:**

| | |
| --- | --- |
| CHANGE ROI | +1.74%, cluster 95% CI **[−7.90%, +10.95%]** — crosses zero |
| KEEP ROI | −6.14%, cluster 95% CI **[−13.81%, +1.36%]** — crosses zero |
| **CHANGE − KEEP** | **+7.88 pp**, cluster 95% CI **[−3.99 pp, +19.92 pp]**, **bootstrap p = 0.202** |

## CHANGE picks and negative EV

| window | n | negative EV at taken price |
| --- | --- | --- |
| README's recorded figure, 90 days | — | **73%** |
| **measured 2026-09-10** (all CHANGE picks fall inside 90 days) | 324 | **87.0%** |
| KEEP, same basis | 456 | 59.4% |
| none | 140 | 78.6% |

**The README's 73% is out of date in the unflattering direction: it is now
87.0%.** All 324 CHANGE picks fall within the last 90 days, so the windows are
the same and the figure has genuinely moved.

---

# THE ANSWER

> ## On every instrument available, the review's effect is indistinguishable from zero.

| instrument | result | verdict |
| --- | --- | --- |
| paired CLV, all picks | −0.101% [−0.561%, +0.345%] | **crosses zero** |
| paired CLV, CHANGE only | −0.389% [−2.136%, +1.367%] | **crosses zero** |
| flat ROI, CHANGE − KEEP | +7.88 pp [−3.99, +19.92], **p = 0.202** | **crosses zero** |
| MODEL vs break-even | +0.256%, upper bound +0.862% | **below +1.85%** |
| FINAL vs break-even | +0.130%, upper bound +0.697% | **below +1.85%** |

**Against Stage 16's established thresholds: `+1.85%` break-even is not
approached by either series, and `p = 0.202` sits on the wrong side of the
`p > 0.15` line Stage 16 found for every settled segment — including
KEEP/CHANGE.** This is not a new finding. **It is the same finding, one month
of additional data later, and it has not moved.**

**The honest conclusion is that there is nothing here to learn from.** A
+7.88 pp ROI gap that cannot be distinguished from zero at n=756 is not a
signal to train on; it is the shape a null takes when the sample is small and
the variance is large.

**The follow-on question — whether the review earns its place at all — is a
decision for Niki and a separate stage.** It is not answered here, and nothing
in this document should be read as answering it. What can be said: **the review
costs a subscription, fails on roughly one day in seven, bypasses the value
gates by design, and produces picks that are 87% negative-EV at the taken
price — and its measured effect on both CLV and ROI is zero.**

---

# THE CONSTRAINT ON ANY FUTURE DESIGN

> ## Training the model on the review's decisions would make MODEL and FINAL dependent, and destroy the comparison that measures the review.

**This is the same class as EXP-1.**

**The dual attribution works because the two series are independent by
construction.** MODEL is the frozen Stage 5 selection; FINAL is what was
persisted after the review. `final − model` is meaningful **only while the
model that produced MODEL has never seen a review decision.**

**The moment a learner reads `review_action`** — to weight markets Claude
prefers, to calibrate on KEEP/CHANGE outcomes, to learn which selections get
overturned — **MODEL stops being a control and becomes a downstream product of
FINAL.** The difference between them then measures the feedback loop, not the
review, and **it will look like the review is adding value precisely in
proportion to how much the model has learned to imitate it.**

**That failure is invisible in the output.** The number still computes, the
interval still narrows, and nothing fails. It is the reason EXP-1's paths remain
gated and the reason `live_only()` was not modified when the reporting change
needed it.

**Any future design must keep the measurement and the learning apart:**

* **a learner may not read `review_action`, the FINAL attribution, or anything
  derived from them** — the existing `valid_evidence()` gate is the enforcement
  point, and this belongs in it;
* **if the review's decisions are ever to be learned from, the comparison must
  be abandoned first and explicitly**, with a cohort bump recording that MODEL
  is no longer a control;
* **the two cannot both be had.** Measuring the review and learning from it are
  mutually exclusive, and the choice must be made openly rather than arrived at
  by an innocuous-looking feature.

*Measured 2026-09-10. Read-only: no schema, no code, no production data changed.*
