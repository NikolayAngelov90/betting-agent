# Stage 6 — Free Odds API Optimization & Prospective CLV Collection

> **⚠️ CORRECTED BY STAGE 10.2 (2026-08-10).** The model identity quoted below
> was computed from the local, gitignored `config/config.yaml`, which CI
> overwrites and production has never executed. The deployed configuration is
> `config/config.example.yaml`. Correct identities:
> `stage5_baseline_20260807.326fcf` (CODE_REVISION `s5.1`, Stages 5-7) and
> `stage5_baseline_20260807.485823` (CODE_REVISION `s5.2`, Stages 8-10).
> Everything else in this report stands. See
> `docs/stage10.2-model-identity-2026-08-10.md`.

Executed against `docs/stage-6-prompt.md`.

**511 tests passing, 0 regressions** (483 after Stage 5, +28).
**Nothing committed, nothing pushed, nothing deployed, no new migration.**

---

## A correction to the Stage 5 report, up front

The Stage 6 brief inherits a figure from my Stage 5 report that measurement shows
was wrong.

Stage 5 said the current design costs **~1,620 credits/month (3.2× over the free
tier)**, derived as `27 leagues × 2 credits × 30 days`. That assumed all 27
mapped leagues are requested every day. The code does not do that:
`update()` calls `_leagues_with_today_fixtures()` and requests **only leagues
that have a fixture that day**.

Replaying 176 days of real fixture history through the shipped selection logic:

| | leagues/day | credits/run |
|---|---|---|
| mean | **7.2** | 14.5 |
| median | 5 | 10 |
| p90 | 19 | 38 |
| max (2026-02-28, Sat) | 24 | 48 |

**Real current cost: ~434 credits/month** — inside the 500 free tier, but with
essentially no margin and a worst day of 48. So the quota problem is real but
smaller than reported, and it is a *margin* problem rather than a 3× overshoot.

The **freshness** problem from Stage 5 stands unchanged and is the one that
actually blocks the experiment.

---

## Before

| metric | value |
|---|---|
| requests/day | 7.2 mean, 24 max (leagues with a fixture) |
| credits/day | 14.5 mean, 48 max |
| credits/month | **~434** of 500 free tier — no margin |
| schedule | one run, `37 9 * * *` (09:37 UTC) |
| CLV coverage capability | **0 valid closing lines, ever** |

Odds are captured ~09:37 UTC; European club matches kick off in the evening. By
kickoff the stored price is 8–10 hours old, and `clv.validate_pair` rejects any
snapshot taken more than 180 minutes before kickoff. The experiment could run
indefinitely and collect nothing.

## After

| metric | value |
|---|---|
| requests/day | 4.3 mean, 25 max (leagues with a *pending pick* imminent) |
| credits/day | **8.6 mean**, 50 worst observed |
| credits/month | **~258** (simulated on 134 days of real pick history) |
| quota safety margin | budget 400, margin 50 → 350 spendable; 500 free tier |
| schedule | `17 11,13,15,17,19,21,23 * * *` — 7 runs/day, 2-hourly |
| expected CLV coverage | **90%** of picks on Odds-API-covered leagues |

---

## Why the obvious design was rejected

The brief's Phase 3 example assumes imminent-fixture refresh saves money:
"3 leagues × 2 = 6 credits rather than 27 × 2 = 54". Measured, it does the
opposite — because a league gets refreshed *repeatedly* as successive fixtures
approach:

| design | credits/month |
|---|---|
| current (all today's leagues, once/day) | 434 |
| imminent fixtures, 90-min window, hourly | **1,289** |
| imminent fixtures, 120-min window, hourly | **1,604** |
| imminent fixtures, 180-min window, hourly | **2,184** |

The saving comes from Phase 5's priority, not Phase 3's window: refresh only
leagues that have a **pending pick awaiting a closing line**. Measured per day:

- 27 leagues configured
- **7.2** have a fixture
- **3.6** have a fixture *and* a pending pick

That last gap is the entire Stage 6 economy.

### Parameter choice, by a rule fixed before looking

*Fewest scheduled runs subject to: ≤400 credits/month, ≥95% pick coverage, and
window ≤150 min so captures land comfortably inside the 180-minute CLV validity
limit.*

| run every | window | min gap | credits/month | coverage |
|---|---|---|---|---|
| 1h | 90 | 180 | 288 | 100% |
| **2h** | **120** | **180** | **256** | **96%** |
| 2h | 180 | 0 | 426 | 100% |
| 3h | 90 | 180 | 164 | 54% |

**Chosen: 2-hourly, 120-minute window, 180-minute minimum interval.**

---

## Changes

| File | Why |
|---|---|
| `src/data/odds_quota.py` **(new)** | Monthly claim-before-spend credit ledger. Reuses the existing `ApiBudgetStore` conditional-UPDATE protocol with a month keyed by its first day — **no migration required**. The pre-existing guard read `x-requests-remaining` *after* a response, which cannot stop a concurrent burst and silently resets to "spend freely" if the Actions cache is lost. |
| `src/scrapers/theodds_scraper.py` | `_imminent_league_fixtures` (selection + skip reasons), `_leagues_refreshed_since` (dedup derived from the odds table, not a cache file), `refresh_imminent` (orchestration, time-injectable). `update()`'s fetch/persist tail extracted to `_fetch_and_persist` so both paths share one implementation. Persist window generalised from hard-coded "today" to the fixtures actually passed — a 23:00 run legitimately wants a 00:30 kickoff. |
| `scripts/refresh_and_capture.py` **(new)** | The single scheduled job: refresh imminent odds, then capture immediately while they are freshest. `--dry-run`, `--status`. |
| `scripts/simulate_odds_quota.py` **(new)** | Phase 13 simulation against real history. |
| `.github/workflows/closing-lines.yml` **(new)** | 2-hourly schedule, `concurrency: cancel-in-progress: false`, minimal dependency install, failure alert. |
| `config/config.{yaml,example.yaml}` | `odds_api.monthly_credit_budget: 400`, `odds_api.safety_margin_credits: 50`, `betting.odds_refresh_window_minutes: 120`, `betting.odds_refresh_min_interval_minutes: 180`. |
| `tests/test_odds_quota_and_refresh.py` **(new)** | 28 tests. |

---

## Tests

| | |
|---|---|
| total | **511 passing** |
| new in Stage 6 | **28** |
| failures | 0 |
| regressions | **0** |

Covering the brief's checklist: quota below/above limit, safety margin, partial
grant, accumulation, **month boundary**; fixture inside/outside window, already
started, completed/postponed, multi-league, unmapped league, already-captured
pick, urgency ordering; recently-refreshed skip, stale refresh, non-Odds-API
rows not counting as a refresh, duplicate run; **no API call when the budget is
exhausted**, dry run spends nothing, no import-time `load_dotenv`, no
credentials in the module, workflow cannot enable real money or generate picks.

Closing-capture validation tests (fresh/stale/post-kickoff/corrupt
bookmaker/consensus/market mismatch) already exist from Stages 4–5 and are
unchanged — **no validation rule was weakened**.

---

## Quota simulation (Phase 13)

134 days of real pick history, 2-hourly runs, 120-min window, 180-min interval:

```
mean credits/day        8.6      median 6      p95 22
worst observed day      50       (2026-05-10 Sun, 25 requests, 44 picks)
PROJECTED month         258      within the 400 budget, free tier 500
pick coverage           90%
```

| scenario | days | mean credits | max | coverage |
|---|---|---|---|---|
| normal weekday | 89 | 5.9 | 18 | 91% |
| normal weekend | 45 | 14.0 | 50 | 90% |
| busiest day (Sun 2026-05-10) | 1 | 50.0 | 50 | 100% |
| European competition day | 29 | 5.1 | 8 | 85% |

**Worst reasonable case.** If *every* day were as busy as the busiest observed
Sunday, the month would cost 1,500 credits — over the free tier. That is not a
realistic season shape, but it is exactly what the guard exists for: the ledger
hard-stops at 400 and logs the refusal. The system cannot silently exceed the
budget; it degrades to collecting fewer closing lines, which is the correct
direction to fail.

---

## Odds API cost model — validated empirically

The whole design rests on `credits = regions × markets`. I isolated it with two
live requests against a **separate** Odds API key (the `wagyu-sports` MCP), never
the project's key:

| request | remaining after | delta |
|---|---|---|
| `regions=eu, markets=h2h,totals` | 406 | — |
| `regions=eu, markets=h2h` | **405** | **1 credit** |

A 1-region, 1-market request costs exactly 1 credit, so the shipped
`eu` + `h2h,totals` request costs **2**. Confirmed.

The same call also confirmed the batching the design depends on: **one league
request returned all 9 upcoming Allsvenskan fixtures with 20 bookmakers each**.
Total validation spend: 3 credits, none of it from the project's quota.
`get_quota_info` is free (406/94 before and after).

### Market necessity (Phase 7)

Neither market can be dropped. Proven from the pick record, not asserted:

| market family | source | settled picks | consumed by |
|---|---|---|---|
| `h2h` → `1X2` | The Odds API | **330** | value calculator, `home/draw/away_implied_prob`, CLV for 1X2 picks |
| `totals` → `over_under` | The Odds API | **473** | Over/Under 1.5–4.5 selections, `over25_implied_prob`, CLV for goals picks |

Removing either blinds CLV for a large share of picks. No market was changed.

**Honest coverage ceiling:** 267 picks are on markets The Odds API's
`h2h,totals` does not cover (Team Goals 134, BTTS 123, Double Chance 10), and 41
picks are on leagues it does not carry (Romania, Bulgaria, Finland). Those can
only get a closing price from API-Football rows already in the database.

---

## Supabase egress (Phase 11)

Every new query is column-projected. Measured against production:

| query | count | shape |
|---|---|---|
| `_imminent_league_fixtures` | 1–3 | projected 4 columns over fixtures in the window, using `ix_match_fixture_date`; a second projected pass for away names; one `DISTINCT match_id` for pending picks |
| `_leagues_refreshed_since` | 1 | projected `GROUP BY league, max(timestamp)` |
| `capture_closing_lines` | 2 (+1 write) | already market-filtered in Stage 5 |

No `SELECT *`, no N+1, no full-history scan. Scope is the handful of fixtures
inside a 120-minute window, not a day or a season.

**Pre-existing issue left alone:** `_get_today_fixtures` (used by the *daily*
`update()`, not by this path) does `session.query(Match)` — all 45 columns — plus
a `session.get(Team, …)` per match, an N+1. The new path avoids both. Fixing it
touches the working daily pipeline for no Stage 6 benefit, so it is reported
rather than changed.

---

## CLV readiness

**Can genuine closing lines now be collected? Yes — mechanically. Not yet in
practice, for one reason outside this stage's scope.**

- expected capture window: **0–120 minutes before kickoff**, inside the
  180-minute validity limit with 60 minutes of margin for scheduler delay
- expected coverage: **90%** of picks on Odds-API-covered leagues; ~76% of all
  picks once unmapped leagues and uncovered markets are accounted for
- current coverage: **0%** — unchanged, and correctly so

### Remaining blocker

**The database currently holds zero future fixtures.** Latest fixture is
2026-08-09 19:30 UTC; it is now 2026-08-10. The dry run therefore selected 0
leagues — correctly, because there is genuinely nothing imminent to refresh.

This is not a Stage 6 defect: the live Odds API returned 9 Allsvenskan fixtures
kicking off today, so the *fixtures exist* — the daily ingestion has not written
them yet (the top-5 leagues start mid-August and the pipeline has not run into
the new season). The closing-line job depends on fixture ingestion running
first; until it does, it will keep selecting 0 leagues and spending 0 credits,
which is the correct failure mode.

**Nothing about this justifies weakening validation.** A lower number of valid
closing lines is preferable to a larger number of invalid ones.

---

## Model integrity

Explicitly confirmed — no model behaviour changed in this stage:

- **no model parameters changed** — `bookmaker_blend_weight` 0.80,
  `strength_half_life_days` 540, `dixon_coles_rho` 0.0, Elo untouched
- **no feature engineering changed**
- **no betting thresholds changed** — `min_expected_value` 0.05,
  `min_confidence` 0.55 as before
- **no gates changed** — all six edge gates remain off in `gate_registry`
- **no `model_version` semantics changed** — the identifier is
  `stage5_baseline_20260807.ac04cc`, and none of the Stage 6 config keys are in
  `TRACKED_KEYS`, so the frozen model keeps its version. That is deliberate:
  scheduling and quota do not change what a prediction *means*.
- **no CLV formula changed**, no `market_spec` rule changed, no pick-selection
  logic changed
- `paper_trading_mode` remains **false** — Stage 6 did not enable it, and did
  not enable real-money betting

---

## Final decision

# NOT READY — BLOCKERS REMAIN

The quota and freshness work is done and validated: the design fits the free
tier with margin (258 of 500, guarded at 400), the cost model is empirically
confirmed, the schedule places captures inside the CLV validity window, and 511
tests pass with no regressions.

But two things must happen before the prospective experiment can begin
collecting, and neither is a Stage 6 deliverable:

1. **Fixture ingestion must run into the new season.** There are 0 future
   fixtures in the database, so there is nothing to refresh or capture.
2. **`paper_trading_mode` must be turned on** so picks are recorded as
   measurement-only, per Stage 5's recommendation.

Until both are true, `clv_coverage_rate` stays at 0% and no model-vs-market
conclusion can be drawn. The system is not profitable and no edge is claimed.
