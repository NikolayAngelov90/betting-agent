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

---

## AMENDMENT 1 — 2026-10-10. THE CADENCE PREMISE IS FALSE AS MEASURED.

**Recorded before collection, as required. No credit has been spent on H1.**

### What the registration assumed

> "The design achieves that by construction: with a refresh cadence of 120
> minutes, t0→t1 and t1→t2 are each ~120 minutes."

That sentence is the only place the design's **slot spacing** is justified, and
it is stated as holding *by construction*. It does not hold. The construction
was the `closing-lines` cron — `47 10 * * *` plus
`17 11,13,15,17,19,21,23 * * *`, eight nominal slots 120 minutes apart — and
the scheduler does not run it.

### What actually fires — MEASURED 2026-10-10

Source: `gh run list --workflow=closing-lines.yml --limit 200`, `event=schedule`
only. The listing returned exactly 200 rows, so it is **TRUNCATED and every
count below is a lower bound**; it spans 2026-09-11 → 2026-10-10.

| quantity | nominal | **measured** |
| --- | --- | --- |
| slots per day | 8 | **median 4, mean 4.2, min 1, max 6** (n=47 days) |
| delay from the nominal slot | 0 | median **+13.7 min**, mean +13.4, sd 16.8, range −16.2…+42.2 |
| consecutive gap | 120 min | median **207 min** |
| gaps below 120 min | 0 | **29 of 91** — and all 29 are also below 90 |

The **shape** matters more than the averages. Every day with a real card shows
the same pattern, in minutes between consecutive slots:

```
2026-09-13  [824, 12, 196, 194]      2026-09-18  [ 19, 208, 194]
2026-09-14  [935, 20, 239, 175]      2026-09-19  [771,  24, 216, 208]
2026-09-15  [ 16, 281, 197]          2026-09-20  [796,  19, 211, 199]
2026-09-16  [ 14, 281, 213]          2026-10-03  [781,  12, 212, 207]
2026-09-17  [ 12, 287, 194]          2026-10-09  [ 14, 349]
```

An overnight gap, then **a pair 12–24 minutes apart**, then roughly 200-minute
gaps. The pair is **below `MIN_GAP_MINUTES = 30`**, so the two runs in it are one
separated instant, not two: the second slot is spent and buys no separation.
A fixture's 360-minute window therefore contains at most **two** separated
instants, and a qualifying triple needs three.

### The consequence, simulated against measured slots and real fixtures

Twelve days with ≥10 fixtures (the empty-card period excluded), real fixture
rows from production, the slot times above, and the **authoritative**
`qualifying_triple`:

```
min_interval=120  credits=316  QUALIFYING=0
min_interval=110  credits=316  QUALIFYING=0
min_interval=100  credits=316  QUALIFYING=0
min_interval= 90  credits=316  QUALIFYING=0
min_interval= 60  credits=316  QUALIFYING=0
min_interval= 30  credits=316  QUALIFYING=0
```

**`min_interval` is not the binding constraint, at any value.** 316 credits
would buy zero trajectories.

### Correcting a figure reported on 2026-10-09

The previous entry reported "0 qualifying at `min_interval=120`, 36 at 110".
That simulation used the **nominal** cron times. It is arithmetically correct
for a schedule that fires as written, and the premise is false: against measured
slots both values give 0. The predicate accounts for none of the difference —
re-run with the authoritative `qualifying_triple` the nominal simulation still
returns 0 and 36, so the entire change is nominal-versus-measured slot times.
**A simulation carries the provenance of its inputs, and "nominal cron" is an
assumption, not a measurement.**

### What a qualifying triple actually requires

`qualifying_triple` needs each gap ≥ 30 min and `hi/lo` **strictly** < 2.0,
inside the 360-minute window. Measured by evaluating the predicate:

| gaps (min) | span | verdict |
| --- | --- | --- |
| 13 / 217 | 230 | rejected — the first gap is below MIN_GAP |
| 30 / 60 | 90 | rejected — ratio exactly 2.0 is **not** < 2.0 |
| 45 / 90 | 135 | rejected — ratio exactly 2.0 |
| 60 / 120 | 180 | rejected — ratio exactly 2.0 |
| 100 / 120 | 220 | **qualifies** |
| 120 / 120 | 240 | **qualifies** |
| 90 / 150 | 240 | **qualifies** |
| 150 / 180 | 330 | **qualifies** |

So the requirement is **three slots 90–150 minutes apart inside the six hours
before kickoff**, with the two intervals within a factor of two of each other.

### The amendment

**No change to `min_interval` is made, because the measurement does not justify
one.** Amending it would be a threshold picked to look like a fix while the
mechanism — slot spacing — went unaddressed.

What is amended is the **premise**: the registration's "by construction" claim
is withdrawn and replaced by a requirement that must be satisfied by the
collection mechanism, whatever that turns out to be:

> **H1-C1 (amended).** The three observations must be spaced 90–150 minutes
> apart, each gap ≥ 30 minutes and the two gaps within a factor < 2.0, inside
> the 360 minutes before kickoff. **This spacing must be produced
> deterministically and verified per fixture, not inferred from a cron
> expression.** `min_interval` must then be set strictly below the achieved
> spacing with margin for jitter; at a 120-minute spacing and the measured
> jitter (sd 16.8 min, max +42.2), **90** satisfies that. The number is
> conditional on the spacing mechanism and is not adopted until one exists.

**This is a sizing defect, not a threshold chosen on an outcome.** The filter
(`min_interval`) and the cadence (the cron) were specified independently and
cancel; nothing about H1's effect size, σ, ρ or `TARGET_N` is touched, and no
collected observation was examined to pick any number here. Every figure above
is `measured` or `simulated` with its date and query window, as recorded.

### Open, and in front of Niki

The GitHub scheduler drops roughly half the slots and batches what it does fire,
so **no cron expression can deliver the required spacing reliably** — that is
what the 12–24 minute pairs are. The repository is **public**, so Actions
minutes are free and unmetered, which makes a single scheduled job that issues
its three requests with in-job waits the only mechanism measured here that is
immune to slot dropping. Cost at three requests per league-day:
`3 × 2 = 6 credits per league-day`, so **~22 credits/day is 3–4 leagues** and
seven days fits inside the 168-credit reservation. Choosing that mechanism is a
decision, not a fix, and it is not taken here.

*Amended 2026-10-10 in commit `abb053e`. No credit spent, no collection begun,
`min_interval` unchanged at 120 pending a spacing mechanism. The measurements
above are reproducible from that commit: the slot series via
`gh run list --workflow=closing-lines.yml --limit 200`, the simulation against
`scripts/h1_collection_check.qualifying_triple` as shipped there.*

---

# AMENDMENT 1 — appended 2026-10-18, BEFORE any re-run. The statistics are UNCHANGED.

The first window closed on 2026-10-17 at **n=4 of 39**, floor 33, **power
0.073** at ρ=0.42. No analysis was run. The window under-collected because the
pipeline that writes H1's input was dead 10-13 → 10-17; the full account is in
`docs/ci-audit-ledger.md` under the entry of 2026-10-18.

Credits reset **2026-11-01**, which is the earliest a re-run can start.

## WHAT IS NOT AMENDED

**The statistics are untouched.** ρ=0.42, σ=5.937%, `TARGET_N` 39, floor 33,
the n=33–39 band from `((z_a+z_b)/z_r)² + 3`, the null's interpretation, the
168-credit reservation and the eight-day window all stand exactly as
registered. Nothing about the first window's result licenses changing them —
`n=4` is **not** a reduced-power result and **`NO DATA` is not a null**, so
there is no finding to update them against.

**C1–C3 are not amendments.** The input precondition
(`scripts/h1_precondition.py`), the per-league skip reasons and the
zero-collection alarm change what is **observable**, not what is **collected**.
A run that would have collected a trajectory before still collects it; a run
that would have collected nothing now says which of three reasons applies. No
selection rule moves.

## WHAT **IS** AMENDED, AND WHY — ONE ITEM

> **The density selection changes from CARD density to BAND density.**

As registered, the pinned leagues were chosen by fixture density **per day**.
That is the wrong denominator, and the measurement that shows it was taken on
the live card of 2026-10-18:

```
H1_PRECONDITION state=NO_FIXTURE_IN_BAND
  68 fixture(s) on the card, 9 anywhere in (240,360] minutes,
  0 of those in the 4 pinned league(s)
```

**68 fixtures, 9 usable, 0 pinned.** A league can be dense per day and
contribute nothing, because a collection round needs the fixture still unkicked
when the **last** request fires. With 3 rounds at 120-minute gaps that is
`T+240`, so the usable band is `(T+240, T+360]` — and a fixture at `T+60` sits
inside the registered 360-minute window while being useless.

**The mechanical justification**: `BAND_LO_MIN = (H1_ROUNDS - 1) *
H1_GAP_MINUTES`. It is derived from the registered round count and gap, not
chosen — if either changes, the band moves with it. Selecting on the window
instead of the band overstates collectable yield by exactly the fixtures that
kick off mid-collection.

**This is the second time the density premise has inverted.** The first was the
weekend card: bigger and worth less. Recorded so the pattern is on file rather
than rediscovered.

### The effect on the design, stated rather than implied

Re-selecting by band density changes **which leagues are pinned**, therefore
**which fixtures are sampled**, therefore the collected population. That is why
it is an amendment and not a fix. It does **not** change ρ, σ, n, the floor, the
credit bound or the analysis — it changes the sampling frame from "leagues with
many fixtures" to "leagues with many fixtures H1 can actually follow", which is
the frame the registered n was always computing for.

## THE UNDER-COLLECTING-RATE DECISION RULE, STATED BEFORE COLLECTION

Registered now, with its date, so it cannot be set after the rate is known:

| checkpoint | condition | decision |
| --- | --- | --- |
| **day 3** of the window | trajectories < **8** | STOP. Re-select the pinned leagues by band density and restart; three days at under a quarter-rate cannot reach 33 in eight. |
| **day 5** | trajectories < **18** | STOP and report. The window cannot reach the floor; continuing spends credits for an underpowered result. |
| **day 8** (close) | trajectories ≥ **33** | run the registered analysis, unchanged. |
| day 8 | trajectories < 33 | **no analysis.** Report n and its power, as on 2026-10-18. |

The day-3 and day-5 figures are the straight-line rate to 33 over eight days
(4.125/day), floored: `ceil(4.125 × 2)` = 9 → **8** allows one slow start, and
`ceil(4.125 × 4)` = 17 → **18** does not. Stated as a rate, not as a verdict,
and dated **2026-10-18** — before the re-run exists.

## PRECONDITIONS ON STARTING AT ALL

A re-run does not begin until all three hold, because the first window failed
on the first of them:

1. `daily-picks` has produced a non-empty card on each of the **three**
   preceding days — H1 samples what that job writes, and it wrote 8, 4 and 1
   fixtures on the three days it mattered.
2. `H1_PRECONDITION` reports `OK`, not `NO_CARD` and not
   `NO_FIXTURE_IN_BAND`, on the morning of the start.
3. The zero-collection alarm is wired, so a silent day is seen on the day it
   happens rather than at the window's close.

**Whether to re-run at all is Niki's**, at a cost of **168 TheOddsAPI
credits** over eight days, earliest start **2026-11-01**.
