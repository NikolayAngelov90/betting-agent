# Pre-Registration — Is GitHub's Scheduler Delay a Function of the Scheduled Hour?

**Registered 2026-10-13, BEFORE any analysis.** Nothing in this file has been
tested at the time of writing. No cron is changed by it.

H1 is mid-collection until 2026-10-17. No pricing schedule, no H1 cron and no
`daily-picks` cron is touched before then.

---

## THE OBSERVATION

Four regimes, and the delay is **monotone in the scheduled hour**:

| cron (UTC) | n | query window / provenance | max delay | mean |
| --- | --- | --- | --- | --- |
| `0 0 * * *` | **9** | `measured` 2026-10-10, single-cron regime, 28d window | 5h04m | **4h31m** (sd 18.2) |
| `0 3 * * *` | **32** | `measured` 2026-10-02, closed regime | **6h35m** | not recorded |
| `0 4 * * 1-5` | **1** | `measured` 2026-10-12, run 38243861903 | **7h38m** (458 min) | — |
| `37 9 * * *` | **39 @ `--limit 120` / 16 @ `--limit 60`** | `measured` 2026-10-02 and 2026-10-09 | **11h21m** | not recorded |

Each figure belongs to its own row. **`11h21m` is `37 9`'s maximum and has
been misattributed once already**; it is not and has never been the envelope of
any other cron.

> **`n=1` supports no distributional claim.** The `0 4` figure enters this
> register as a **single observation that replaced the measured maximum** — 458
> replacing 304 — and as nothing else. It is not a mean, not a spread, and not
> evidence of a distribution.

## THE HYPOTHESIS

> **GitHub's scheduler delay is a function of the scheduled hour**, such that a
> cron moved later is delivered later by more than the nominal shift. A later
> cron therefore buys **less than its face value**, and the weekday move from
> `0 0` to `0 4` returned approximately what the earlier move to `00:00` had
> bought.

Arithmetic of that last clause, so it is falsifiable rather than rhetorical:
`0 0` was ready at **06:16**; `0 4` with the single observed delay is ready at
**12:49**. The nominal shift was 4 h; the realised shift is **6h33m**.

## THE FALSIFIABLE PREDICTION

**If the hypothesis is true**, returning weekdays to `0 0 * * 1-5` will restore
the weekday delay distribution toward the `0 0` regime:

* **mean toward ~4h30m**, and
* **maximum below ~5h30m**

measured over a window containing **at least n=10 weekday scheduled runs**, so
the comparison is against `0 0`'s n=9 and not against a handful.

**If the hypothesis is false**, the weekday delays under `0 0 * * 1-5` will
look like the `0 4` observation rather than like the old `0 0` regime — i.e. a
mean materially above 5h and a maximum above 6h — and the lateness will have
been a property of the **calendar**, not the hour.

## THE DECISION RULE, STATED NOW

| result after the trial window (n>=10 weekday runs) | decision |
| --- | --- |
| mean <= 5h00m **and** max <= 5h30m | **ABANDON THE SPLIT.** Weekdays return to `0 0` permanently; the hour was the cause and `0 4` was buying negative margin. |
| mean > 5h00m **or** max > 6h00m | **KEEP A SPLIT**, because the hour is not the lever and the deadline still differs by day type. The weekday hour is then chosen on the measured delay, not on the nominal one. |
| anything between | **INCONCLUSIVE — the split stays as it is** and the window extends. An inconclusive result is not a licence to pick the preferred answer. |

The deadline question is **separate and unchanged**: weekday and weekend have
different earliest kickoffs, which is why a split exists at all. This trial is
only about whether a later *hour* buys margin.

## THE CONFOUND, NAMED BEFORE MEASURING

> **The four regimes are also four consecutive time periods.** `37 9` ran before
> `0 3`, which ran before `0 0`, which ran before `0 4`. **Hour and calendar are
> not separable from this data.** A platform-wide change in scheduler load, a
> change in GitHub's queueing, or a seasonal effect would produce the same
> monotone table with no hour effect whatsoever.
>
> So the table above **cannot** establish the hypothesis, and no amount of
> re-reading it will. It can only motivate a prospective test.

A second confound: the `0 4` row is also the only row from a **split** cron
(`* * 1-5`). A day-of-week-restricted schedule may be queued differently from
an everyday one. n=1 cannot distinguish that either.

## THE ONE FREE TEST AVAILABLE

`closing-lines` fires at **several hours within a single calendar period**. If
its delays can be grouped by scheduled hour, that separates hour from calendar
**inside one regime** — the only such population in this repository, and it
costs nothing to measure.

**Stated before measuring:** its delay attribution carries a **2-hour ceiling**
(`CENSORED`), because runs are matched to the nearest preceding cron and its
crons are 2 hours apart. Whether that ceiling **destroys** the comparison or
merely **truncates** it is itself a finding to report, not to assume:

* it **truncates** if the hour effect within the ceiling is smaller than the
  ceiling — the mean of a censored sample is biased low but the *ordering*
  across hours may survive;
* it **destroys** the comparison if the delays routinely exceed 2 hours, in
  which case every run is attributed to the wrong cron and the grouping is
  noise.

A ceiling imposed by an attribution rule is not an observation, so any mean
computed under it is reported as censored or not reported.

## WHAT HAPPENS UNTIL 2026-10-17

No cron change. The mechanism is:

1. the schedule-margin report, printed every audit run, with the breach
   **acknowledged** (`ACKNOWLEDGED_MARGIN_BREACH`, reviewed 2026-10-18) so it
   reports without alarming until it worsens; and
2. **manual dispatch** when the operator judges it necessary — the probe's
   alert now carries the earliest kickoff and the last safe dispatch time in
   both clocks.

## SUCCESSOR CONTRACT

This file is amended only by appending. The decision rule above is **not**
rewritten after the result is known; if it turns out to be the wrong rule, the
amendment says so and explains why, with the original left in place.
