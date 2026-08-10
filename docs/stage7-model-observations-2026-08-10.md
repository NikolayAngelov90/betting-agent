# Stage 7 — Model Observations (NOT implemented)

> **⚠️ CORRECTED BY STAGE 10.2 (2026-08-10).** The model identity quoted below
> was computed from the local, gitignored `config/config.yaml`, which CI
> overwrites and production has never executed. The deployed configuration is
> `config/config.example.yaml`. Correct identities:
> `stage5_baseline_20260807.326fcf` (CODE_REVISION `s5.1`, Stages 5-7) and
> `stage5_baseline_20260807.485823` (CODE_REVISION `s5.2`, Stages 8-10).
> Everything else in this report stands. See
> `docs/stage10.2-model-identity-2026-08-10.md`.

**Date:** 2026-08-10
**Status:** observations only. No model change was made. `model_version` remains
`stage5_baseline_20260807.ac04cc`, `CODE_REVISION = s5.1`.

Stage 7 was an operational stage and the model is frozen. These were found while
auditing the pick pipeline and are recorded here so the decision is deliberate.

---

## Observation 1 — the correlation filter has no over↔under cross pairs

**Severity: real. Decide before the experiment collects data.**

### What was seen

Checking recent pick batches for duplicate matches (30-day window, production,
read-only) turned up three matches carrying two picks each:

| Date | Match | Picks |
|---|---|---|
| 2026-08-08 | Estrela vs Sporting CP (PT) | Over 2.5 @1.53 **+ Under 3.5 @1.58** |
| 2026-08-04 | Levski Sofia vs Kairat Almaty (UCL) | Home Win @1.65 + Over 2.5 @2.25 |
| 2026-08-03 | SJK vs HJK Helsinki (FIN) | Over 2.5 @1.53 + Double Chance 1X @1.79 |

Two picks per match is the intended `max_picks_per_match: 2`. The second and
third pairs are declared in `_CORRELATED_PAIRS` and were kept on composite score
— working as designed.

The first is not. **Over 2.5 and Under 3.5 are two rungs of the same totals
ladder pointing in opposite directions.** Together they are a bet that the match
ends with exactly 3 goals.

There is no duplicate-fixture bug: a 60-day check for the same team pairing under
different `match_id`s returned empty, so the `apifootball_id`-first matching is
holding.

### Why it slips through

`_CORRELATED_PAIRS` in [betting_agent.py:4083-4136](../src/agent/betting_agent.py#L4083-L4136)
declares same-direction ladders:

```python
("Over 1.5 Goals",  "Over 2.5 Goals"),
("Over 2.5 Goals",  "Over 3.5 Goals"),
("Under 2.5 Goals", "Under 3.5 Goals"),
("Under 3.5 Goals", "Under 4.5 Goals"),
```

and the 1X2 ↔ Double Chance / DNB overlaps. It declares **no over↔under pair at
all**. Any `Over X.5` + `Under Y.5` with `Y > X` on one match passes untouched.

### Why it matters

Both legs individually cleared the EV bar, so the pair is not an arithmetic
error. The problem is sizing. Kelly assumes bets are roughly independent; here
both stakes ride on a single match's goal count, so the position is one
concentrated bet on a narrow window — exactly the over-concentration the filter
exists to prevent — sized as if it were two diversified ones.

It is also a signal worth surfacing: the model is claiming the market underprices
*both* tails of the same distribution. That is coherent only if it is putting
unusual mass on exactly 3 goals, and it should be visible when it happens rather
than passing silently.

### Proposed fix (not applied)

Add the nested cross pairs to `_CORRELATED_PAIRS`:

```python
# Opposite rungs of one totals ladder: winning both requires the goal count
# to land in the gap between the lines. Two stakes, one narrow bet.
("Over 1.5 Goals", "Under 2.5 Goals"),
("Over 1.5 Goals", "Under 3.5 Goals"),
("Over 2.5 Goals", "Under 3.5 Goals"),
("Over 2.5 Goals", "Under 4.5 Goals"),
("Over 3.5 Goals", "Under 4.5 Goals"),
```

Four lines of data, no logic change — the existing composite-score tiebreak then
drops the weaker leg.

### Why it was not applied

It changes which picks are emitted, so it requires a `CODE_REVISION` bump to
`s5.2` and a new `model_version`.

**The timing argument is the whole point.** Right now that costs nothing: 0
valid closing lines have been collected, so no evidence is invalidated. Once the
experiment is running, the same fix splits the sample across two model versions
and delays a decision-grade CLV read by however long has elapsed. If it is going
to be fixed, fix it before the first paper picks land.

Frequency is low — 1 occurrence in 30 days, ~13 Under-3.5 picks per 90 days
against 105 Over-2.5 — so the cost of *not* fixing it is small and bounded. The
cost of fixing it mid-experiment is not.

---

## Observation 2 — ~36% of picks are structurally outside CLV measurement

**Severity: bounds the experiment. Not a defect.**

The Odds API request is `h2h,totals`, which populates `1X2` and `over_under`
only. Team Goals (20.0% of picks), BTTS (14.1%) and Double Chance (2.3%) come
from the daily API-Football scrape and are never refreshed near kickoff, so no
valid closing line can exist for them.

This is covered in §7b of the operational report. Recorded here because it is a
*model-selection* observation as much as an operational one: the pick generator
allocates over a third of its output to markets the experiment cannot grade.

Two coherent responses, neither implemented:

1. **Accept.** Measure CLV on the 64% that is measurable, and treat the rest as
   unmeasured. Cheapest, and 64% is enough to answer the question. Recommended.
2. **Restrict paper picks to measurable markets** for the duration of the
   experiment. Gets ~100% coverage of what is emitted and reaches 500 valid
   closing lines faster — but it changes the pick distribution, so it is a model
   change with a new `model_version`, and it makes the paper record
   unrepresentative of what the system would actually bet.

Option 2 is tempting and probably wrong: it optimises the measurement at the cost
of measuring the real thing.

---

## Not investigated

Stage 7 did not revisit the Stage 1–3 conclusion that the model adds no
information over the bookmaker price. Nothing here bears on it. These are
pick-construction and measurement-scope observations, not evidence about
predictive skill.
