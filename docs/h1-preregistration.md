# H1 — PRE-REGISTRATION

**Written 2026-09-17, before the policy change exists and before any H1
observation exists.** Nothing this registration depends on can have been shaped
by H1 data, because **there is none**: a query for series carrying three
pre-kickoff observations separated by ≥30 minutes returns **0 across 0 fixtures**
(98,929 snapshot rows, 2026-08-25 to 2026-09-17, phantom and corrupt rows
excluded). That is the classification `UNADDRESSABLE BY SELF-OBSERVATION`
restated as a measurement.

---

## THE HYPOTHESIS

> **H1. Pre-kickoff price movement is positively autocorrelated: the direction
> and size of the move from t0→t1 predicts the move from t1→t2, within the same
> fixture and the same price series.**

**Directional and one-sided.** Momentum, not mean-reversion. A significant
NEGATIVE correlation falsifies H1 as stated and is reported as a separate
finding, not as support.

**The estimator is a correlation, not a location shift** — `n = 6.185·σ²/δ²`
sizes a mean and does not transfer, which is why H1 carries its own n.

---

## THE EFFECT SIZE — CARRIED FROM STAGE 16, NOT RE-DERIVED

| threshold | meaning |
| --- | --- |
| **+1.85%** | best-line break-even. Below this no action is profitable. |
| **+2.00%** | minimum decision-relevant effect |
| **+4.00%** | comfortable — would change behaviour without argument |

**These are properties of the market this project bets into.** They are inherited
from Stage 16 and from H1's own registration; nothing here re-derives them.

**Expected captured gain is `ρ·σ·E[z|acted]`.**

| | | provenance |
| --- | --- | --- |
| σ | **5.937%** | measured 2026-09-10, n=129, **fixture-level** median move |
| `E[z|acted]` | **0.798** = √(2/π) | two-sided timing |
| **ρ registered** | **0.42** | derived 2026-09-16 from δ=+2% and the policy |
| **n** | **33–39** | `((z_a+z_b)/z_r)² + 3`, band across +2.00% to +1.85% |

### The acting policy is FORCED, which is why ρ is not a free parameter

**The pipeline selects picks on EV. By the time any price is observed the fixture
is already chosen, so a drift signal can change WHEN a price is taken and never
WHICH fixture is bet.** Every pick gets a timing decision and the signal's sign
chooses direction — that is two-sided timing, and it is the only policy the
architecture permits. **EFF-1**: an effect size is a property of the phenomenon
*and* the policy that would act on it, so a ρ registered without a fixed policy
registers nothing.

---

## THE DECISION RULE — FIXED NOW

| observed | reading |
| --- | --- |
| **r ≥ 0.42, p < 0.05 one-sided** | **SIGNAL, and actionable.** Captured gain ≥ +2.00%, clear of break-even. |
| **0 < r < 0.42, p < 0.05** | **SIGNAL, NOT ACTIONABLE.** Real autocorrelation below the cost of acting. Reported as such and **not** used to justify a timing change. |
| **p ≥ 0.05** | **NULL** — see the bound below. |
| **r < 0, p < 0.05** | **FALSIFIED AS STATED.** Mean-reversion; a separate question, not a weaker H1. |

### WHAT A NULL AT n = 33–39 ACTUALLY EXCLUDES — the likeliest result, stated first

A null does not mean "no effect". It means the effect is **smaller than this**:

| n | 95% one-sided upper bound on ρ (at r ≈ 0) | implied captured gain | vs +1.85% |
| --- | --- | --- | --- |
| 33 | **0.292** | **1.38%** | **below break-even** |
| 39 | **0.267** | **1.27%** | **below break-even** |

> ### A null at n = 33–39 establishes that pre-kickoff momentum cannot pay for itself: the captured gain is bounded below the best-line break-even, so acting on it loses to the overround.
>
> **That is a decision, not an absence of one** — it retires timing as a lever
> and closes the last untested hypothesis in this project.

**It does NOT exclude** a small real autocorrelation (ρ up to ~0.27), a larger
effect in a market or lead-time band this sample does not cover, or momentum
under a policy that could select fixtures — which this architecture cannot.

---

## THE CONTROL — QUERIED BEFORE BEING REGISTERED

**H1 is SELF-CONTROLLED. The comparison is within-fixture and within-series:
t0→t1 against t1→t2, same match, same bookmaker, same market, same selection.**
There is no control group and none is claimed.

**H5's Q1 registered a control group that did not exist**, because the refresh
requires a pending pick and every qualifying fixture was therefore picked. **That
error is structurally impossible here** — the control is the earlier interval of
the same series — but the underlying fact was queried rather than assumed, and it
constrains the conclusion:

| queried 2026-09-17 | result |
| --- | --- |
| fixtures carrying a separated pre-kickoff gap | **142** |
| **of those, PICKED** | **142 (100%)** |
| distinct bookmakers carrying them | **TheOddsAPI only** |
| existing gap length | p10 **354** · p50 **486** · p90 **645** minutes |
| observation lead time before kickoff | p10 **26** · p50 **415** · p90 **658** minutes |

> **The sample is picked fixtures only, and that bounds GENERALISATION, not
> validity.** H1 will be answered for the fixtures this pipeline bets, which is
> the population the answer would be acted on for. **Stated here so it cannot be
> reported later as though it covered all fixtures.**

**The two intervals must be comparable in length or the within-fixture control is
not like-for-like.** The design achieves that by construction: with a refresh
cadence of 120 minutes, t0→t1 and t1→t2 are each ~120 minutes. **Any series whose
two intervals differ by more than 2× is excluded before analysis.**

---

## CONTAMINATION CONTROLS — EVERY PRIOR POSITIVE RESULT HERE DIED TO ONE OF THESE

| control | how it is applied |
| --- | --- |
| **the overround band** | H2 read **+0.705%** and was the two-way (draw-excluded) trap. **Only prices whose book overround falls in (1.005, 1.25) enter.** Computed per (fixture, book, market) at each observation, not pooled. |
| **provider as a STRATUM, never pooled** | API-Football's median movement was **+37.3%** against The Odds API's **−0.61%** — different instruments, not different samples. **Measured: 100% of existing separated series are TheOddsAPI**, so the collection is single-provider by construction. Provider is recorded on every row and reported; if any API-Football row enters, it is analysed as a separate stratum or dropped, never merged. |
| **the phantom class** | **510 permanent `phantom_kickoff_now_stamp` rows, 34.7% of August.** Every query touching `matches` carries `training_exclusion_reason IS NULL`. **An aggregate contamination rate is not a bound on any single query**, so the predicate is applied per query rather than argued once. |
| **duplicate fixture identity** | **One observation per FIXTURE IDENTITY, not per `match_id`.** Three known guarantee violations came from that distinction. Series are grouped through the s5.9 fixture-group resolution; two rows that are one fixture contribute one series. |
| **aggregation level** | **One actionable price per fixture** — the best available line for the selection actually picked — **never the mean across books and markets.** H5's first σ came back at **1.05%** against a true **7.67%** for exactly this reason: **a factor of fifty in n.** |

---

## WHAT WOULD MAKE ME DISTRUST THE RESULT

* **a σ materially different from 5.937%** in the collected window — the sizing
  is built on it, and a different σ means a different n;
* **asymmetric intervals** — if t1→t2 is systematically longer than t0→t1,
  autocorrelation and drift are confounded;
* **a single extreme fixture carrying the correlation** — a single anomalous
  result is evidence about the measurement first. The result is reported with and
  without the most influential observation.

---

## THE REAL COST — this is a TRADE, not a bill

**168 credits is not the whole of it, and quoting only that reads as a cost.**

| | | provenance |
| --- | --- | --- |
| measured consumption | **~34.6 credits/day** | `api_budget`: 400 used in September, 437 in August |
| spendable budget | **400/month** | free tier 500 − safety margin 50, self-imposed budget 400 |
| **days of pipeline the free tier funds** | **~12/month** | 400 ÷ 34.6 — **which is why OPS-4 opened on the 12th** |
| H1 collection | **≈106–168 credits** | window 360, 2 days |
| **what that actually costs** | **≈3–5 days of closing-line capture** | not the 2 days it runs |

> ### H1 costs roughly five days of capture, not two. The collection runs for two days; the credits it spends are five days of the pipeline's oxygen.

**And the trade is still clearly favourable, on reasoning already in the record.**
Stage 16 established that **lost captures buy precision on a RESOLVED axis**:

* **MODEL's upper bound is +0.107%** against a **+1.85%** requirement — the
  question of whether the model beats the closing line is answered, and more
  observations move a bound that is already an order of magnitude inside the
  threshold;
* the **500-observation target was over-specified about twenty-nine-fold**.

**H1's question is OPEN. The capture's question is CLOSED.** Spending a closed
axis's precision to answer an open question is the trade, and it is the right way
round.

---

*Registered 2026-09-17, before the policy change and before any H1 observation
exists. No credit spent, no config changed, no collection begun.*

---

## COHORT NEUTRALITY — RESTATED 2026-09-20, BECAUSE THE FIRST VERSION RESTED ON A MECHANISM THAT CANNOT EXECUTE

**The claim is unchanged: the H1 collection does not require an `s5.15` cohort
break. The reason it holds is not the reason first registered.**

### What was registered on 09-17, and why it does not stand

> *"NEUTRAL, conditionally. 0 of 18 runs overlap under the current cron, margin
> 41 minutes. Neutral by construction if the runner gates on `daily-picks`
> completion; without that gate it rests on a thin margin."*

The picks-run guard was that gate, and it was approved in part as protection for
H1's widened 360-minute refresh window. **Measured 2026-09-20 over 231
closing-lines runs, 08-10 to 09-20, against the same tables the guard reads:**

| | |
| --- | --- |
| total runs | **231** |
| fail condition 1 — outside the 03:00-16:00 window | **143** |
| fail condition 2 — today's run had already written picks | **76** |
| **pass 1 AND 2** | **12** |
| of those, zero candidate leagues | **12** |
| condition 3 ever evaluated | **0** |
| **WOULD DECLINE** | **0 / 231 = 0.0%** |

**And the reason is structural, not statistical.** A candidate league needs a
fixture kicking off within the window that **already carries a pick**;
condition 2 requires that **no pick was written today**; and the measured
fixture-ingestion horizon is **11.2-12.0 hours — same-day only**, so any fixture
kicking off in the next two hours was created today and any pick on it was made
today. **Conditions 2 and 3 exclude each other by construction.**

> ### An argument resting on a mechanism that cannot execute is not an argument. The guard is recorded as REDUNDANT, not protective, and this registration does not rely on it.

### What the neutrality actually rests on — the candidate rule

**The same measurement that excludes the guard excludes the exposure.**

`_imminent_league_fixtures` yields a candidate league only when a fixture in the
window **carries a pending pick**. So:

| state | candidates | refresh | exposure |
| --- | --- | --- | --- |
| **no picks written today** | **none** — nothing carries a pending pick | **no requests made** | **none: there is nothing to contaminate and nothing is fetched** |
| picks written today | some | requests made | a pick already taken persists `taken_odds` at pick time, so a refresh cannot alter it |

**A pending picks run and an active refresh cannot coexist**, because the
predicate that authorises a refresh is the presence of the very output whose
absence defines "pending". The widened 360-minute window changes **how far ahead**
a candidate is looked for; it does not change **what makes one a candidate**.

**Three independent statements, and the conclusion needs only the first two:**

1. **The candidate rule** — no picks, no candidates, no requests. Structural,
   and it is the load-bearing one.
2. **`taken_odds` is persisted at pick time** — a refresh cannot retroactively
   change the price a recorded pick was taken at. Structural.
3. 0 of 231 runs would have declined; 0 of 18 overlap the first refresh under
   the current cron, margin 41 minutes. **Empirical, and now corroborating
   rather than load-bearing.**

> ### VERDICT: NEUTRAL, UNCONDITIONALLY — by the candidate rule and `taken_odds`, not by the guard. No `s5.15` required.
>
> The 09-17 verdict was *"NEUTRAL, conditionally"* and named the guard as the
> condition. **The condition is removed rather than satisfied**: the state the
> guard was meant to police is one in which no refresh happens at all.

### What this does NOT claim

* **Not that the guard is harmful.** It fails open, has declined nothing in
  production, and costs nothing while unreachable. Removing it is a separate
  decision and is not proposed here.
* **Not that the guard was wrong to build.** Its decline branch was forced on
  09-19 and behaves as specified. **It is correct and unreachable**, which are
  different properties.
* **Not that no refresh can ever overlap a picks run.** It claims the overlap
  requires today's picks to exist, and in that case the exposure is bounded by
  `taken_odds` rather than by the guard.

**If the fixture-ingestion horizon ever extends beyond same-day, statement 1
weakens and this argument must be re-derived** — the mutual exclusion of
conditions 2 and 3 depends on it, and so does the candidate rule's bite. Stated
here so the dependency is visible rather than assumed.

*Restated 2026-09-20. No credit spent, no config changed, no collection begun.*
