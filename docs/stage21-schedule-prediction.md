# Stage 21 — Schedule Change: Pre-Registered Prediction

**Written and committed BEFORE the first firing of `0 3 * * *`.**
At the time of writing (2026-08-30), the cron change is pushed and the new
schedule has never fired. The first firing is **2026-08-31, a Monday**.

A schedule change is a claim. This project measures its claims rather than
discussing them afterwards.

## What changed and why the numbers below are what they are

`daily-picks` moved `37 9 * * *` → `0 3 * * *` (09:37 → 03:00 UTC).

The choice was computed against three measured constraints:

| constraint | measured 2026-08-30 | consequence |
| --- | --- | --- |
| earliest kickoff, any day | **10:04 UTC** (Sunday p05 10:06) | a ~20-min run must **start by 09:45 UTC** |
| settlement | latest KO 19:30 → football ends ~21:30 | cron must follow results publication |
| odds availability | 90% of fixtures priced ≥24h out | does not bind |

**03:00 therefore buys 6h45m of delay tolerance** (09:45 − 03:00) with 5h30m of
settlement margin.

## THE PREDICTION

**Start time** — GitHub's measured delay envelope is **0.5h–5.7h historically**,
with **10h21m and 11h21m observed on 2026-08-27/28**.

1. **If the envelope returns to historical, the run starts between 03:00 and
   08:42 UTC.**
2. **Anything up to 09:45 UTC still covers the whole card** — that is the
   tolerance the choice was computed against.
3. **Beyond 09:45 UTC, fixtures are missed and the 6h45m margin was
   insufficient.**

### Three outcomes, defined before looking

| outcome | condition | meaning |
| --- | --- | --- |
| **WITHIN TOLERANCE** | start ≤ **08:42 UTC** | delay inside the historical envelope; margin unused |
| **LATE BUT COVERING** | **08:42 < start ≤ 09:45 UTC** | margin consumed but the card is intact — the choice worked *because* of the change |
| **BEYOND TOLERANCE** | start > **09:45 UTC** | fixtures missed; 6h45m was not enough and the residual in B5 is realised |

Also recorded, from the run: `model_version` on any pick must be
`stage5_baseline_20260807.dfe302` (`s5.8`), and the pick **lead-time
distribution**, whose projection was 7.1h at the p10 kickoff, 10.7h at the
median and 15.4h at p90 — against a measured on-time baseline of 4.4–8.8h.

## Monday's card, and how much a delay would cost

**Known card for 2026-08-31**, MEASURED 2026-08-30 from football-data.org
(its 9 competitions only):

```
8 matches, all TIMED
  SA  16:30Z  Lecce v Roma            PD  17:30Z  Osasuna v Getafe
  SA  18:45Z  Atalanta v Bologna      PL  19:00Z  Aston Villa v Arsenal
  PPL 19:15Z  Braga v Vitoria         PPL 19:15Z  Benfica v Estoril
  PD  19:30Z  Barcelona v Rayo        BSA 23:00Z  Remo v Coritiba
```

**Earliest kickoff in that subset: 16:30 UTC.**

**Monday's historical profile** (MEASURED, n=185, August 2026, all 30 configured
leagues): earliest **10:34**, p10 10:39, **median 13:13**, latest 19:30.

**Fraction of a Monday card already kicked off, by run start:**

| run start | historical Monday (n=185) | tomorrow's known card (n=8) |
| --- | --- | --- |
| 03:00 (no delay) | **0%** | 0% |
| 08:42 (+5h42, historical max) | **0%** | 0% |
| 09:45 (tolerance limit) | **0%** | 0% |
| 14:00 (+11h, observed max) | **80%** | **0%** |

## THE CAVEAT, registered before the run rather than discovered after it

> **Tomorrow may not test the margin at all.**

Within football-data.org's coverage, tomorrow's earliest kickoff is **16:30 UTC**
— so *any* delay up to ~13 hours is harmless for those eight fixtures. Even the
observed 11h21m maximum would cost **nothing** on that subset.

**A `WITHIN TOLERANCE` result tomorrow is therefore weak evidence.** It would
confirm the run fires and the cohort stamps correctly, but it would **not**
demonstrate that 6h45m of margin is sufficient, because the card would not have
required it.

The full 30-league card may still contain earlier fixtures that
football-data.org does not cover — Monday's historical earliest is 10:34 — and
if it does, the test is stronger. **That is not known in advance and is not
claimed.**

**The margin is genuinely tested only on a day whose card starts early: a
Wednesday (median KO 10:28) or a Saturday/Sunday (median 14:00, p05 ~10:06).**
Those are the days to read the result on, and this file should be checked against
the first such day rather than against tomorrow alone.

---

*Committed before the first firing of the new cron. Stage 21, 2026-08-30.*

---

# SECOND CHECKPOINT — registered 2026-08-30, before either day arrives

## Monday's result must not be cited as evidence about the margin

**This sentence is the reason this section exists now rather than after a pass
makes the question feel settled.**

Monday 2026-08-31 tests **that the cron fires** and that picks stamp `s5.8`. It
**cannot** test the margin: tomorrow's earliest known kickoff is **16:30 UTC**, so
every delay up to ~13 hours is harmless. A `WITHIN TOLERANCE` result there would
be **true and uninformative** about the thing the choice was computed against.

> **A `WITHIN TOLERANCE` result on 2026-08-31 is not evidence that 6h45m of
> margin is sufficient, and must not be recorded as such.**

## The day that does test it

**Kickoff profiles under the new 03:00 cron** (MEASURED 2026-08-30, August 2026):

| day | n | earliest | p05 | median | lost @09:45 | lost @12:00 | lost @14:00 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Wed** | 119 | 10:15 | 10:15 | **10:28** | 0% | **84%** | 84% |
| Sat | 293 | 10:15 | 10:45 | 14:00 | 0% | 20.5% | 50.5% |
| Sun | 325 | **10:04** | 10:07 | 15:00 | 0% | 29.8% | 42.8% |

**The two days fail in different shapes, and that is why both are registered:**

* **Wednesday is a CLIFF.** Its card is concentrated in a ~15-minute band around
  10:15–10:30, so the loss goes **0% → 84% between a 7h15m and a 9h delay**.
  There is no partial failure — the day is either intact or mostly gone.
* **Saturday is a GRADIENT.** Losses accumulate: 20.5% at +9h, 50.5% at +11h.

### PRIMARY — first Wednesday: **2026-09-02**

### SECONDARY — first weekend day: **Saturday 2026-09-05**

## Outcomes, fixed in advance — defined on the CARD, not the start time

**The start time is only a proxy. The card is what the change was for**, so each
outcome is defined on the fraction of that day's fixtures already kicked off when
the run actually began.

| outcome | condition |
| --- | --- |
| **MARGIN HELD** | **0%** of the day's card had kicked off at the actual start |
| **MARGIN CONSUMED** | **>0% and ≤20%** lost — the change helped and was nearly exhausted |
| **MARGIN INSUFFICIENT** | **>20%** lost — 6h45m was not enough; B5's residual is realised |

## The measurement to take on each day

Not the start time alone. All four, from the run and the database:

1. **actual start time**, and the delay from 03:00
2. **fixtures inside `max_days_ahead` at the moment the run executed**
3. **how many of those had already kicked off** — the unpickable remainder
4. **the fraction of that day's whole card lost**, which decides the outcome above

Plus, for continuity with the Stage 21 arithmetic: the **pick lead-time
distribution**, projected at 7.1h (p10 kickoff) / 10.7h (median) / 15.4h (p90)
against a measured on-time baseline of 4.4–8.8h and 2.1h on the late run.

## What each result would mean

* **Wednesday MARGIN HELD** — the strongest single result available. The cliff
  day survived, so the margin covered the delay on the day least able to absorb
  it.
* **Wednesday MARGIN INSUFFICIENT** — expect ~84%, not a small number, because of
  the cliff. That is B5 realised and the answer is not a further cron shift: at
  84% the pipeline needs splitting, not moving.
* **Saturday differing from Wednesday** — informative rather than contradictory.
  A gradient day losing 20% while the cliff day loses 0% is the delay landing
  between the two thresholds, and that is a *measurement of the delay*, not a
  failure of the change.

**If the first Wednesday's card is unusually late — as Monday's is — the same
caveat applies and the checkpoint moves to the next Wednesday.** Check the card
before reading the result, not after.

*Registered before either day, 2026-08-30.*

---

# CORRECTIONS, measured on the first firing (2026-08-31)

**Recorded after the run, and labelled as such. The outcomes above are NOT
revised — only the arithmetic behind one boundary, and one projection that could
not have been met.**

## 1. The 09:45 boundary was derived from an ASSUMED run length

> **`09:45` came from *"a ~20-min run must start by 09:45 UTC"* against a 10:04
> earliest kickoff. The ~20 minutes was assumed and never labelled as an
> assumption.**

**MEASURED 2026-08-31, run `33375724727`:**

| | |
| --- | --- |
| run start → run end | **63m 37s** (09:02:40 → 10:06:17) |
| run start → **picks written** | **34m 35s** (09:02:37 → 09:37) |

**Picks are what the deadline is about**, so 34m35s is the figure that matters,
not the full 63m.

> ### CORRECTED BOUNDARY: **~09:29 UTC**, and the margin is **6h 29m**, not 6h45m.

**Wednesday must be read against 09:29.** The three outcome bands keep their
meaning; only the LATE-BUT-COVERING/BEYOND-TOLERANCE line moves, from 09:45 to
09:29. Today's 09:02:37 start sits inside the corrected boundary, so **no
recorded outcome changes.**

**The general point, which is the reusable half:** *a registered boundary derived
from an assumed value carries that assumption's provenance, and this one was
never labelled.* Every number in this file that is not marked MEASURED should be
read as carrying the same risk.

## 2. The lead-time doubling did not occur, and could not have

**Projected: 7.1 / 10.7 / 15.4h. MEASURED (n=20): p10 6.4h · median 7.9h · p90
9.6h** — below projection at every point.

**This is not a failure of the projection; it is the projection's premise not
being met.** It assumed picks written at **~03:20**. They were written at
**09:37**, six hours later, because the run was delayed.

> **The cron change's benefit to the MODEL series is CONTINGENT on the cron
> firing near its time. At a six-hour delay it buys coverage margin only.**

Today's median of **7.9h sits inside the pre-change on-time baseline of
4.4–8.8h** — the lead was normal, not improved. The projection stands as
written, unmet, and is testable only on a day the run starts near 03:00.

## 3. LATE BUT COVERING was earned by the card, not by the margin

Today started 09:02:37 — **6h 02m 37s late, outside the 5.7h historical envelope**
— and lost **0%** of a 28-fixture card. But the card's earliest kickoff was
**16:00 UTC**, so any delay under ~13h would have scored the same.

**The limit registered against WITHIN TOLERANCE applies here unchanged:**

> **A `WITHIN TOLERANCE` result on 2026-08-31 is not evidence that 6h45m of
> margin is sufficient, and must not be recorded as such.**

**Nor is a LATE BUT COVERING result. Wednesday 2026-09-02 remains the test**, and
its card must be checked for lateness before its result is read.

*Corrections measured and recorded 2026-08-31, after the first firing.*

---

# WEDNESDAY 2026-09-02 — THE PRIMARY CHECKPOINT IS VOID, AND THE REASON IS A PHANTOM ARTIFACT

## Timing of this entry, stated exactly rather than implied

| event | UTC |
| --- | --- |
| external card established (public calendar) | ~07:26 |
| August Wednesday profile recomputed | ~07:20 |
| **run `33603479759` started** | **07:25:16** |
| this entry committed | ~07:30 |

> **THIS IS NOT A PRE-REGISTRATION AND MUST NOT BE READ AS ONE.** The run began
> while the card was being established. What follows is derived **only** from
> sources independent of the run — a public fixture calendar and August history
> — and **no output of run `33603479759` has been read**. But the wall clock is
> the wall clock, and the entry lands four minutes after the run started.

## 1. TODAY'S EARLIEST KICKOFF, from outside this system

**The registered pre-check could not be answered internally**: there were zero
`matches` rows for 2026-09-02 at 07:20 UTC, because those rows are created by
the run being qualified.

**Two external sources were tried and failed before one worked** — recorded so
"no data" is not mistaken for "no fixtures":

| source | result |
| --- | --- |
| football-data.org `/v4/matches` | **0 matches for 08-31 → 09-05**, but 1 for 09-06. It also returned 0 for 2026-08-31, a day the registered file itself recorded as **8 matches from this same source**. **Self-inconsistent, therefore unusable as evidence** — not treated as "no fixtures". |
| `soccer-server` MCP | **403 Forbidden** on the server's own key. Unavailable, not empty. |
| **public fixture calendar (web)** | **usable** |

**TODAY'S CARD (2026-09-02), from the public calendar:**

| competition | fixtures | kickoff (local BST) | **UTC** |
| --- | --- | --- | --- |
| EFL Championship | 4 | 19:45–20:00 | **18:45–19:00** |
| EFL League One | 5 | 19:45–20:00 | **18:45–19:00** |
| Scottish Premiership | 3 | 19:45 | **18:45** |

> ### EARLIEST KICKOFF TODAY: ~18:45 UTC.

**Limitation stated:** the listing is UK-centric and may miss fixtures in other
configured leagues. It is corroborated below by August's own real Wednesdays and
by yesterday's card (17:30–19:00), and early September is an international
break, which is why the card is small and domestic-lower-league only.

## 2. WHAT TODAY CAN DISCRIMINATE: **NOTHING ABOUT THE MARGIN**

**With an 18:45 UTC earliest kickoff, a run starting any time before ~18:00
loses nothing.** The corrected boundary is **09:29**. A delay of *fifteen hours*
would still score 0% lost.

> **Today is UNINFORMATIVE about the margin, for the same reason Monday and
> Tuesday were. All three registered outcomes are NOT reachable: only MARGIN
> HELD is.**

**Registered before reading the run's card measurements.**

## 3. WHY WEDNESDAY WAS CHOSEN — AND WHY THAT CHOICE WAS AN ARTIFACT

The second checkpoint registered this profile, and designated Wednesday PRIMARY
because of it:

| day | n | earliest | p05 | median | lost @09:45 | lost @12:00 |
| --- | --- | --- | --- | --- | --- | --- |
| **Wed** | 119 | 10:15 | 10:15 | **10:28** | 0% | **84%** |

**Recomputed today, August 2026, with and without phantom rows:**

| | n | earliest | p05 | median | latest |
| --- | --- | --- | --- | --- | --- |
| **WITH phantoms (as registered)** | **119** | `10:15:29.254552` | `10:15:29.6` | `10:28:45.419404` | 19:00 |
| **WITHOUT phantoms** | **19** | **16:00:00** | 16:00 | **19:00:00** | 19:00 |

**The registered figures carry MICROSECONDS. That is the phantom signature** —
`match_date` equal to `created_at`, the row stamped with the pipeline's own
wall-clock time instead of a kickoff.

**And the clock they carry is the old cron's:**

| phantom Wednesday rows, by hour | count |
| --- | --- |
| **10:00** | **71** |
| **11:00** | **29** |

`37 9 * * *` plus run time lands at 10:00–11:00 UTC. **The "Wednesday cliff at
10:15–10:28" is the previous schedule's execution window, recorded as kickoff
times.**

> ## THE WEDNESDAY CLIFF DOES NOT EXIST.
>
> **100 of the 119 "Wednesday fixtures" were phantoms. The 19 real ones start
> at 16:00 and have a median of 19:00** — Wednesday is one of the most
> delay-tolerant days in the week, not the least.

**The specific claims now withdrawn:**

* *"Wednesday is a CLIFF … 0% → 84% between a 7h15m and a 9h delay"* — **the
  84% was phantoms.**
* *"Wednesday, not Saturday, is the least delay-tolerant day — 60% of its card
  gone at one hour"* — recorded as the *unexpected half* of the B1 finding.
  **It was an artifact, and being unexpected is exactly why it should have been
  checked harder.**
* **PRIMARY CHECKPOINT = Wednesday** — chosen on that artifact, so the choice
  was wrong at the moment it was made.

**Saturday and Sunday largely survive the same test**, which is why this went
unnoticed: Sat earliest 10:15 with phantoms and 10:15 without (n=294→267); Sun
10:04 with, 10:15 without (n=325→226). **Wednesday was uniquely corrupted
because it had few real fixtures (19) and many phantoms (100)** — the smallest
real denominator in the week.

### The rule this is an instance of

The phantom class was found, quantified (**1.280% by count, 2.817% by decay
weight**) and excluded from the *model* datasets. **It was never excluded from
the SCHEDULE analysis**, which was written later, by me, querying `matches`
directly.

> **An exclusion is a property of a query, not of a dataset.** Establishing that
> a contaminated class exists does not protect any analysis that does not filter
> it, and every new analysis re-inherits the contamination by default.

This is the same shape as the `deff` predicate correction recorded yesterday:
**a measurement is only as good as the filter it applies, and neither the data
nor the tooling will remind you.**

## 4. WHERE THE CHECKPOINT ACTUALLY GOES

**Recomputed on phantom-free August data:**

| day | n (real) | earliest | median |
| --- | --- | --- | --- |
| Wed | 19 | 16:00 | 19:00 |
| **Sat** | **267** | **10:15** | 15:00 |
| **Sun** | **226** | **10:15** | 15:00 |

> **The margin is testable on SATURDAY and SUNDAY, and on no other day
> measured.** Both keep a 10:15 earliest after phantoms are removed, and both
> have real denominators in the hundreds.
>
> ### The checkpoint moves to Saturday 2026-09-05 — which was already registered as SECONDARY, and is now the only candidate.

**The registered Saturday profile survives the phantom test and its outcome
bands stand as written**: MARGIN HELD at 0% lost, MARGIN CONSUMED >0–20%,
MARGIN INSUFFICIENT >20%. **Its gradient shape — 20.5% at +9h, 50.5% at +11h —
should be re-derived phantom-free before Saturday rather than trusted**, since
it came from the same table.

**The pre-registered meaning of a Wednesday MARGIN INSUFFICIENT — "expect ~84%,
and at that level the answer is splitting the pipeline, not shifting the cron"
— is withdrawn with the profile it rested on.** Nothing about today can trigger
it.

## 5. What today still measures

Today remains worth recording as a **second WITHIN TOLERANCE data point** on the
delay envelope, and as `s5.9`'s second live day. **It is not a margin test.**

**Run `33603479759` started `2026-09-02T07:25:16Z` — delay 4h 25m 16s, WITHIN
TOLERANCE (≤08:42), the second consecutive firing inside the historical
0.5–5.7h envelope.** The four card measurements follow when it completes.

*Recorded 2026-09-02 ~07:30 UTC, four minutes after the run began, from sources
independent of it.*

---

# SATURDAY 2026-09-05 — CORRECTED THRESHOLDS, registered 2026-09-02

**Three days before the day they govern.** Full derivation in
`docs/ci-audit-ledger.md`; the operative numbers are repeated here because a
reader of this file must not have to know the ledger entry exists — **that is
rule 3, and this pointer is the remedy.**

**Saturday is now the PRIMARY checkpoint** (Wednesday's cliff was phantoms, see
the entry above). **Sunday 2026-09-06 is secondary.**

**Phantom-free, August 2026 — n=267 (Sat), n=226 (Sun):**

| run start | **Sat lost** | **Sun lost** | *(Sat as registered, contaminated)* |
| --- | --- | --- | --- |
| **09:40** — the corrected boundary | **0.0%** | **0.0%** | — |
| 11:00 | 4.1% | 1.8% | *9.2%* |
| 12:00 | **12.4%** | **15.5%** | *20.5%* |
| 13:00 | 16.1% | 26.1% | — |
| 14:00 | **45.3%** | **34.1%** | *50.5%* |

**Bands unchanged. What moves is where the start has to land:**

| outcome | condition | **start time** |
| --- | --- | --- |
| **MARGIN HELD** | 0% lost | **≤ ~09:40** |
| **MARGIN CONSUMED** | >0% and ≤20% | **~09:40 → ~13:20** |
| **MARGIN INSUFFICIENT** | >20% | **beyond ~13:20** |

**The 20% crossing moves ~80 minutes later, from ~12:00 to ~13:20.**

**And the boundary itself is 09:40, not 09:29 and not 09:45.** The 10:04
earliest that anchored both earlier figures was a phantom row; the real earliest
is 10:15, and picks land 34m35s after start.

*Registered 2026-09-02, before the day.*

---

# SATURDAY'S ACTUAL CARD — registered 2026-09-03, two days ahead

**Taken from an external source, corroborated by two independent lookups.**

| | |
| --- | --- |
| fixtures listed | **29** |
| **earliest kickoff** | **12:30 BST = 11:30 UTC** |
| Premier League | 7 · EFL Championship 11 · League One 11 · League Two 10 |

**Saturday's card is REAL and EARLY — the margin is testable, and this is the
first day this week of which that is true.** Wednesday (16:30) and Thursday
(16:00) could not test it, and both returned MARGIN HELD that proves only that
the cron fires.

**But this card's earliest is 75 minutes later than the 10:15 the August
profile is built on**, so on the card that actually exists:

| | August-derived | **this Saturday** |
| --- | --- | --- |
| first loss at a start of | ~10:15 | **~11:30** |
| MARGIN HELD requires start ≤ | 09:40 | **~11:30** |

> **The 09:40 boundary is CONSERVATIVE for this Saturday by about 1h50m.**
>
> **A MARGIN HELD result on 2026-09-05 demonstrates that the margin covers a
> start up to ~11:30 UTC on a 29-fixture early card. It does NOT demonstrate
> the 09:40 boundary**, which needs a start between 09:40 and 11:30 to be
> tested at all, and a start past 11:30 to be falsified.

*Registered before the day, with hours between measurement and commit.*
