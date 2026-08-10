# Stage 7 — Operational Paper Trading & Prospective CLV

> **⚠️ CORRECTED BY STAGE 10.2 (2026-08-10).** The model identity quoted below
> was computed from the local, gitignored `config/config.yaml`, which CI
> overwrites and production has never executed. The deployed configuration is
> `config/config.example.yaml`. Correct identities:
> `stage5_baseline_20260807.326fcf` (CODE_REVISION `s5.1`, Stages 5-7) and
> `stage5_baseline_20260807.485823` (CODE_REVISION `s5.2`, Stages 8-10).
> Everything else in this report stands. See
> `docs/stage10.2-model-identity-2026-08-10.md`.

**Date:** 2026-08-10
**Model:** `stage5_baseline_20260807.ac04cc` — **unchanged**, `CODE_REVISION = s5.1`
**Status:** uncommitted, not pushed, not deployed
**Tests:** 540 passed

---

## 1. Executive summary

Stage 7 set out to make the frozen Stage-5 system capable of collecting clean
prospective evidence. It found three defects that would each, independently,
have produced a *plausible-looking but false* experimental result. All three are
fixed. None of them touched the model.

**The one that matters most:** the closing-line capture stamped
`closing_odds_captured_at` with **the script's run time**, not the time the
price was observed. Every validity check in `clv.validate_pair` then passed by
construction — the script only ever runs inside the window, so a price the odds
table had been holding for ten hours was recorded as a *closing* line.

That is not a theoretical risk. Measured on production: the median odds row for
a matched fixture is **281–340 minutes old at kickoff**, and only **16–25%** of
rows fall inside the 180-minute closing window. For the 36% of picks in markets
the pre-kickoff refresh does not cover (team goals, BTTS, double chance), the
"closing" price would have been *the same database row the pick was priced
from* — yielding a CLV of exactly 0.00%, which reads as **closing-line parity**.
The system would have reported that it prices in line with the market when in
fact it had measured nothing at all.

Capture now stamps the price's own observation time and rejects prices older
than the CLV window, so such picks come back `missing` — honest, and countable
as a coverage gap rather than as evidence.

**Second:** `is_paper` had been written since Stage 5 but was **never read**.
Enabling paper trading without fixing that would have pooled measurement-only
picks into live ROI, the Telegram performance report, the EV-threshold
calibrator, the Bayesian weight learner and the probability calibrator — letting
the experiment's own output rewrite the model it exists to measure. Now filtered
at 7 sites, with 12 isolation tests.

**Third:** the credit ledger and The Odds API's own counter were independent.
Checked against the live account today: the provider reports **95 credits used**
this month; the ledger read **0**. A fifth of the free tier had been spent
outside this pipeline and every budget decision was being made against a number
wrong in the dangerous direction. The ledger now reconciles upward from the
response headers.

Paper trading is **enabled**. The experiment can start as soon as the odds-refresh
cron runs. Expected time to 100 valid closing lines: **≈21 days**.

---

## 2. Stage-6 verification

Stage 6's work was re-derived from the code rather than taken on trust.

| Claim | Verdict |
|---|---|
| `eu + h2h,totals` = 2 credits/request | Confirmed — `REGIONS × MARKETS` in [odds_quota.py:85-87](../src/data/odds_quota.py#L85-L87) |
| ~258 credits/month for the selected strategy | Re-measured at 212 cr/month on 30 d, 170 on 90 d |
| 90% historical coverage | Re-measured at 88% for the pick-driven strategy |
| Model untouched | Confirmed — `model_version` still `stage5_baseline_20260807.ac04cc` |
| 511 tests passing | Confirmed as the Stage-6 baseline; now 540 |

Stage 6's headline finding — that the daily 09:37 UTC pipeline leaves odds 8–10 h
stale at kickoff — is confirmed and **quantified** in §7 below.

---

## 3. Fixture-ingestion findings

Production, read-only, at 2026-08-10 08:25 UTC:

```
fixtures_total    67
fixtures_future    0
fixtures_today     0
max_match_date    2026-08-09 19:30
```

**Zero future fixtures is expected, not broken.** The ingestion horizon is
TODAY-ONLY: `daily-picks.yml` runs at 09:37 UTC and fetches that day's fixtures.
At 08:25 UTC the cron had not yet fired, so the newest rows are still yesterday's.

The horizon costs little: **95.7% of picks kick off between 11:00 and 22:00 UTC**,
i.e. after the daily run. The ~4% that kick off before it are unreachable
regardless of odds strategy.

It does, however, mean the odds-refresh job **selects zero leagues until the
daily ingestion has run**. The two crons are correctly ordered (09:37, then
11:17 onward), but a delayed daily run — GitHub's scheduler has been observed
0.5–5.7 h late on this repo — pushes the first useful refresh slot later. The
2-hourly spread absorbs this.

---

## 4. Odds API quota model

Recalculated from the current code, not from the Stage-5 estimate.

```
credits = requests × regions × markets
        = requests × 1 × 2          (eu; h2h,totals)
```

Both markets are required: `h2h` serves 1X2 picks (32.3% of recent picks) and
`totals` serves the over/under family (31.3%). Dropping `totals` would halve the
unit cost and remove a third of the experiment.

| Guard | Value | Where |
|---|---|---|
| Free tier | 500 / calendar month | provider |
| Self-imposed budget | 400 | `DEFAULT_MONTHLY_BUDGET` |
| Safety margin | 50 | `DEFAULT_SAFETY_MARGIN` |
| Per-run ceiling | 24 credits | `DEFAULT_MAX_CREDITS_PER_RUN` |

The per-run ceiling was sized from measurement, not intuition: across 313 charged
runs in the simulated history the per-run cost was mean 3.7, p95 10, p99 14, max
20. A cap of 24 clips **0%** of observed runs while bounding a runaway (a
fixture-data glitch that made all 27 leagues look imminent) to 12 leagues.

### Ledger ↔ provider reconciliation (new)

Checked against the live account today:

```
provider:  95 used, 405 remaining
ledger:     0 used
```

The ledger starts each month at zero and only knows about spend that goes
through it. The provider counts everything on the key — including this repo's
MCP client and any manual call. Budget 400 + out-of-band 95 = 495, inside the
tier by luck alone.

`OddsApiQuota.reconcile()` now adopts the provider's count from the
`x-requests-used` response header after each batch. It only ever raises: a
provider count *below* the ledger means a month boundary or a different key, and
spending more on the strength of that guess is the wrong direction. It uses
`ApiBudgetStore.raise_used_to()` rather than `claim()`, because a claim is
refused above the limit — exactly wrong for recording spend that has already
happened, where the refusal would silently preserve the too-low number.

---

## 5. Selected refresh strategy

Three strategies were measured on production fixture history. **C wins on both
axes** and is what ships.

| Strategy | Selection rule | Credits/month (30 d) | Credits/month (90 d) | CLV coverage |
|---|---|---|---|---|
| A — cheapest | 1 refresh/league/day | 238 | 210 | 8% |
| B — broadest | every league with a fixture in window | 340 | 296 | 84–85% |
| **C — pick-driven** | **league has a fixture in window AND a pick awaiting a close** | **212** | **170** | **88%** |

C is cheaper *and* better covered because the constraint is aligned with the
goal: a credit is spent only where it can actually produce a closing line.
Measured on production history, ~3.6 mapped leagues per day carry a pending
pick, against ~7.2 with any fixture and 27 configured. That gap is the entire
saving.

Implemented as `refresh_imminent(window_minutes, min_interval_minutes,
require_pending_pick=True)` in [theodds_scraper.py:902](../src/scrapers/theodds_scraper.py#L902).

---

## 6. Cost simulation

`scripts/simulate_odds_quota.py` replays production fixture history through the
real selection code.

```
window            120 min
min interval      180 min
credits/month     212 (30 d replay) · 170 (90 d replay)
worst single day  46 credits
per-run cost      mean 3.7 · p95 10 · p99 14 · max 20
budget            400 (+50 margin held back)
free tier         500
headroom          ~47% of tier unused at the measured rate
```

Adding the 95 credits already spent out-of-band this month, the projected August
total is ~307 of 500 — comfortable, and now visible to the ledger.

---

## 7. Closing-line capture validation

The pipeline `prediction → taken odds → late refresh → closing capture → CLV`
was traced end to end. Three defects were found in the capture stage.

### 7a. Run time was being recorded as observation time — **critical**

`closing_odds_captured_at` was set to `now` (the moment the script ran).
`clv.validate_pair` uses that field for its lead check. Since the script only
ever runs for picks kicking off within 90 minutes, **the check could not fail**,
no matter how old the underlying price was.

Measured on production, for matches that carried a pick in the last 30 days:

| Market | Rows | Min lead | **Median lead** | **% within 180 min** |
|---|---|---|---|---|
| 1X2 | 9,012 | 36 min | **281 min** | **25.4%** |
| over_under | 7,267 | 36 min | **302 min** | **21.2%** |
| team_goals | 7,337 | 39 min | **316 min** | **16.9%** |
| btts | 900 | 39 min | **316 min** | **16.0%** |
| double_chance | 402 | 39 min | **340 min** | **16.4%** |

So 75–84% of available prices are *not* closing prices, and all of them would
have been stored as such.

**Fix.** `consensus_close()` now takes the odds row's own `timestamp`, drops
rows observed before `kickoff − max_lead`, and returns the oldest contributing
row's timestamp as `observed_at`. That value — not `now` — becomes
`closing_odds_captured_at`. Capture and validation share one window constant
(`clv.DEFAULT_MAX_CAPTURE_LEAD`), so anything capture will store is something
CLV will accept; a test asserts that property directly.

**Consequence, stated plainly:** coverage will now be *lower* and *true*. Prices
outside the window are recorded `missing`. No check was weakened to raise
coverage; one that was vacuous was made real.

### 7b. Markets the refresh never touches

The Odds API request is `h2h,totals`, which populates `1X2` and `over_under`
only. Team goals, BTTS and double chance come from the daily API-Football scrape
and are **never refreshed near kickoff**.

| Market family | Share of picks (90 d) | Closing line obtainable? |
|---|---|---|
| 1X2 | 32.3% | yes (`h2h`) |
| Over/Under 1.5–3.5 | 31.3% | yes (`totals`, main line) |
| Team Goals | 20.0% | **no** |
| BTTS | 14.1% | **no** |
| Double Chance | 2.3% | **no** |

**≈36% of picks are structurally outside prospective CLV measurement.** Before
7a's fix these would have silently returned CLV ≈ 0.00% from a stale row. They
are now honestly `missing`. Widening the API request to cover them would
increase the per-request cost proportionally (`credits = regions × markets`) and
is **not** recommended at the free tier — the 64% that is measurable is enough
to answer the question.

### 7c. Stale backlog and an N+1 write-back

Migration 003 backfilled all 1,070 historical picks to
`closing_capture_status = 'pending'`, and the window filter is
`match_date <= now + 90 min` with no lower bound. The first production run
therefore sweeps every one of them.

That was benign for correctness — they are all `late` by definition — but the
old code built `match_ids` from *all* rows before the late check, so it read
their odds too, and wrote results back with one `session.get()` per pick: 1,070
round trips, against a docstring promising two queries.

**Fix.** Late rows are partitioned out *before* the odds query, and status-only
outcomes go back as one bulk `UPDATE` per status. Verified by test: a 50-pick
stale backlog plus one live pick reads **3** odds rows, not 153, in ≤6 queries.

### Validity rules — unchanged

Correct match · correct market · correct selection · valid decimal odds · valid
timestamp · before kickoff · inside the closing window · valid bookmaker · valid
market structure · plausible overround · no corrupt source-market collision.
All still enforced. `missing` / `late` / `invalid` are recorded explicitly; no
price is ever invented or substituted.

---

## 8. Paper-trading safety verification

### The finding: `is_paper` was written but never read

Stage 5 added the column and stamped it on every save. Nothing consumed it. Had
paper mode been enabled on that basis, measurement-only picks would have flowed
into:

| Consumer | What it would have corrupted |
|---|---|
| `get_stats()` | headline ROI, win rate, Brier — and the Telegram performance report |
| `rolling_backtest()` | the reported equity curve |
| `_auto_calibrate_ev_threshold()` | `min_ev` — **which bets get taken** |
| cold-streak market breakdown | which markets get suppressed |
| `tune_ensemble_weights()` | Bayesian ensemble weights — **the frozen model** |
| `calibrate_from_pick_outcomes()` | per-model calibration factors |
| probability-calibration drift check | the calibration refit trigger |

The last three are the serious ones: the experiment's own output would have
rewritten the model it exists to measure, while `model_version` stayed constant
and claimed nothing had changed.

**Fix.** A single module-level predicate in
[betting_agent.py](../src/agent/betting_agent.py), applied at all 7 sites:

```python
def _live_only():
    return or_(SavedPick.is_paper.is_(False), SavedPick.is_paper.is_(None))
```

NULL counts as live: production backfilled all 1,070 rows to `false`
(verified today — `is_paper=false: 1070`, no NULLs), but a deployment that added
the column without a default would read NULL, and those rows are real history.

12 tests in `tests/test_paper_live_isolation.py` cover it, built so leakage is
unmissable: live picks all win at 2.0, paper picks all lose, so any contamination
moves ROI from +100% to 0%.

### The message is the betting action

The Telegram picks message *is* how bets get placed — by hand, by a person
reading it. A measurement-only pick that looks identical to a live recommendation
is a money-safety problem, not a cosmetic one. `send_daily_picks(paper_mode=True)`
now prepends:

> **🧪 PAPER TRADING — DO NOT BET REAL MONEY**
> *Recorded for measurement only. These picks are excluded from the live record
> and exist to collect closing-line data.*

A test asserts the banner appears when `paper_mode=True` and — equally important
— does **not** appear when it is `False`.

### Activation

`paper_trading_mode: true` in both `config/config.yaml` and
`config/config.example.yaml`. CI copies the example over the real config, so the
next scheduled run is paper. The value is asserted in a test, so leaving paper
mode requires a documented decision that breaks a test, not a quiet config edit.

---

## 9. CI scheduling

Two workflows, deliberately not merged into one.

| Workflow | Cron (UTC) | Sofia (EEST) | Purpose |
|---|---|---|---|
| `daily-picks.yml` | `37 9 * * *` | 12:37 | fixture ingestion → predictions → settlement → report |
| `closing-lines.yml` | `17 11,13,15,17,19,21,23 * * *` | 14:17–02:17 | imminent odds refresh → closing capture |

They stay separate because their cadences are irreconcilable: ingestion and
prediction are once-daily, while a closing line must be taken within 180 minutes
of each kickoff spread across a 12-hour evening window. Folding capture into the
daily job would collect nothing — which is precisely the Stage-6 finding.

All schedules are UTC. Bulgaria observes EET/EEST and shifts on the last Sundays
of March and October, so the *local* time of these runs moves by an hour twice a
year. Nothing in the logic reads local time — every kickoff comparison uses
stored UTC — so only the convenience of the slots moves.

The off-hour minutes (`:37`, `:17`) avoid the top-of-hour scheduling crush.

---

## 10. Supabase egress audit

Every query added by Stages 6–7 is column-projected, index-backed and bounded.

| Query | Projection | Bound | Measured |
|---|---|---|---|
| `_imminent_league_fixtures` | `Match.id, league, match_date, Team.name` | `ix_match_fixture_date`, window only | handful of rows |
| `_leagues_refreshed_since` | `Odds.match_id, timestamp` | window matches only | small |
| capture Q1 — pending picks | 7 columns, joined to `Match` | `closing_capture_status='pending'` + horizon | one round trip |
| capture Q2 — odds | 6 columns | live match ids × needed markets only | **3 rows** where the old code read 153 |
| capture write-back | — | one bulk `UPDATE` per status | was 1,070 round trips |
| `paper_trading_report` | projected | `days` + `model_version` | bounded |

No `SELECT *`. No N+1 remains in the scheduled path. The market restriction on
Q2 alone cut a full-history run from 104,117 rows to ~7,600; the late-row
partition cuts the routine case to near zero.

---

## 11. Test results

```
540 passed, 15 warnings in 124.52s
```

New in Stage 7 (29 tests over the Stage-6 baseline of 511):

| File | Tests | Covers |
|---|---|---|
| `test_paper_live_isolation.py` | 12 | paper picks excluded from live ROI and from every loop that changes future predictions; NULL treated as live; paper visible to the experiment report; Telegram banner |
| `test_odds_quota_and_refresh.py` | +9 | per-run ceiling (4), provider reconciliation (5) |
| `test_odds_quota_and_refresh.py` | +2 | concurrency: 12 racing workers cannot oversell the budget; an overlapping run cannot respend the first run's claim |
| `test_closing_capture.py` | +6 | stale price rejected; fresh price still captured; `captured_at` is the price time; capture and CLV share one window; backlog reads no odds; backlog-only run issues no odds query |

The 15 warnings are pre-existing `datetime.utcnow()` deprecations.

---

## 12. Production verification

Read-only. No rows created, altered or deleted. No new migrations applied.

```
now (UTC)                 2026-08-10 08:25
fixtures_total            67          fixtures_future     0
odds newest               2026-08-09 10:26
odds rows (7 d)           16,699
picks_total               1,070       settled            1,048
picks_pending             22          (all kicked off 2026-08-09)
is_paper = false          1,070       is_paper NULL      0
model_version set         0
closing_odds set          0
closing_capture_status    pending × 1,070
CLV coverage              0.0%
Odds API ledger           0 / 400 used (August)
Odds API provider          95 used, 405 remaining
```

Current-season market inventory (last 3 days): 1X2 across **27 books**,
over_under across **19**, team_goals 3, double_chance 2, btts 2 — spanning 92
matches. Bookmaker breadth is sufficient for a median consensus close on the two
measurable market families.

`model_version` is set on 0 rows, confirming no pick has been generated since
Stage 5 froze the model. The first paper pick will be the first row to carry it.

---

## 13. Experiment contamination safeguards

| Risk | Guard |
|---|---|
| Paper outcomes rewrite the model | `_live_only()` at 7 sites, 12 tests |
| Paper picks inflate the live record | same predicate; `get_stats`, `rolling_backtest` |
| Paper picks read as real bets | mandatory Telegram banner, asserted both ways |
| Model drifts mid-experiment | `model_version` recorded per pick; a config or `CODE_REVISION` change mints a new one |
| A stale price is scored as a close | observation-time stamping + shared window constant |
| Coverage inflated by unmeasurable markets | `missing` recorded explicitly, never substituted |
| Out-of-band credit spend exhausts the tier | ledger reconciles upward from provider headers |
| Two runs spend the same credit | conditional `UPDATE ... WHERE used + n <= ceiling`, tested under 12-thread contention |
| Two runs double-capture a pick | only `status='pending'` is considered; idempotency tested |

`model_version` is the linchpin: any analysis must group by it, and a mid-flight
change makes that visible rather than silent.

---

## 14. 100 / 200 / 500-pick measurement plan

Rate, measured on the last 30 days: **229 picks / 30 days = 7.63 per day**, of
which **164 / 229 = 71.6%** are in CLV-eligible markets. At Stage 6's measured
88% capture coverage:

```
7.63 × 0.716 × 0.88 ≈ 4.8 valid closing lines per day
```

| Milestone | ETA | What it can answer | What it cannot |
|---|---|---|---|
| **100** | ~21 days | Is the capture pipeline sound? Is coverage as predicted? Is mean CLV plausibly non-zero? | Nothing about profitability. A 100-pick ROI is noise. |
| **200** | ~42 days | Sign of mean CLV with a usable CI. Calibration by market family. | Still no ROI conclusion. |
| **500** | ~104 days (~3.5 months) | Whether mean CLV is reliably positive — the decision-grade question. | Segment-level claims; those need far more. |

**Stop rules.** If CLV coverage after 100 picks is below ~60% of eligible picks,
the collection pipeline is the problem — fix it before reading any CLV number.
If mean CLV at 500 is not distinguishable from zero by a paired bootstrap, the
Stage 1–3 conclusion stands: the model adds no information over the market, and
no real-money betting is justified.

**Real money remains off the table at every milestone below 500 valid closing
lines**, regardless of how good short-term ROI looks. A profitable-looking 100
picks is exactly what a −3.6% ROI system produces sometimes.

---

## 15. Remaining blockers

Nothing blocks starting the experiment. Two things bound what it can conclude,
and one needs a decision.

1. **~36% of picks are unmeasurable for CLV** (§7b). Structural, not a bug.
   Accept it and measure the 64%, or widen the API markets and pay proportionally
   more per request. Recommendation: accept.
2. **`ODDS_API_KEY` is absent from the local `.env`** — present in GitHub secrets
   since 2026-02-15, so CI works, but the refresh path cannot be exercised
   locally (today's dry run skipped with `ODDS_API_KEY not configured`). Not a
   blocker; a note for anyone testing locally.
3. **A same-match hedge slips the correlation filter.** See
   `docs/stage7-model-observations-2026-08-10.md`. Needs your decision because
   fixing it changes emitted picks and therefore `model_version`.

---

## 16. Exact next steps

1. **Review this report and the model-observations doc.** In particular, decide
   on the correlation-filter gap — fixing it now is free (0 closing lines
   collected); in two weeks it invalidates the run.
2. **Commit Stages 6 + 7 together** — one commit, local only. Not done: Stage 7
   said do not commit.
3. **Push and let `closing-lines.yml` run.** First scheduled slot 11:17 UTC. The
   first run sweeps the 1,070-row legacy backlog to `late` — expected, one-time,
   ~3 odds rows read.
4. **After 24 h, check `scripts/paper_trading_report.py --operational`** for:
   credits claimed vs provider, leagues refreshed, capture status breakdown,
   coverage. Expect the first captures within 48 h.
5. **At ~20 valid closing lines, sanity-check the leads.** Every captured row
   should show an observation time inside 180 minutes of kickoff. If they cluster
   at the boundary, tighten the cron rather than the check.
6. **At 100, run the milestone review** in §14 — pipeline health only, no
   profitability reading.

---

## Appendix — what did *not* change

The model is untouched. `model_version` is still
`stage5_baseline_20260807.ac04cc` and `CODE_REVISION` is still `s5.1`, verified
against the live config after all Stage-7 edits.

`bookmaker_blend_weight` remains 0.80. All six edge gates remain disabled.
`min_expected_value` and `min_confidence` are unchanged. No feature, threshold or
weight was tuned.

The `_live_only()` filter is the one change that touches a prediction-affecting
path, and it is a no-op today: production holds 0 paper picks, so every filtered
query returns exactly what it returned before. Its purpose is to *preserve* the
freeze once paper picks exist, not to alter it — which is why `CODE_REVISION`
does not move.

**"The system can now collect data" is not "the model works."** Stages 1–3
established, by three independent tests, that this model adds no information
over the bookmaker price. Nothing in Stage 7 revisits that. The market remains
the benchmark, CLV remains the primary prospective signal, and paper trading
remains mandatory.
