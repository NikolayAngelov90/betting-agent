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
