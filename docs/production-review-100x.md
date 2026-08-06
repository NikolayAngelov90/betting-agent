# Production review — betting-agent at 100× traffic

**Reviewer stance:** senior staff engineer, pre-scale readiness review.
**Scale assumption:** 100× today → ~13,200 fixtures/day, ~3.8 M matches, ~31 M odds
rows, ~100 K+ tracked picks. Free-plan limits ignored.
**Method:** every finding below is backed by code, an `EXPLAIN (ANALYZE, BUFFERS)`
against the live database, or a measured constant. Nothing is inferred from
smell alone.

---

---

## Implementation status (updated 2026-08-06)

The S2 prerequisites and the three quick wins are **done and verified**. S1-1
(sharding) is not — and deliberately so: it was blocked on exactly these fixes.

| Item | Status | Evidence |
|---|---|---|
| S2-1 saved_picks dedup race | **Fixed** | `ix_saved_picks_dedup` unique index (migration 002) + `INSERT ... ON CONFLICT DO NOTHING RETURNING id`. Interleaved-writer test drives both sessions past the read check before either writes. |
| S2-2 odds upsert race | **Fixed** | `ON CONFLICT DO UPDATE` with `COALESCE` preserving `opening_odds`. Test inserts a rival row mid-batch and asserts the batch survives. |
| S2-4 bind-parameter ceiling | **Fixed** | `src/data/sql_helpers.py` `id_in()` → `= ANY(:array)`. Measured: 30,000 ids goes from **30,000 bind params / 469 KB of SQL** to **1 param / 80 chars**. |
| S2-3 mirror concurrency hole | **Fixed** | `_SyncLock` (fcntl/msvcrt) around the whole sync, with a timeout that proceeds rather than deadlocks. |
| S1-2 per-process API budget | **Fixed** | `api_budget` table + `ApiBudgetStore` atomic claim. Test: two scraper instances share one ceiling. |
| S1-3 per-team COUNT N+1 | **Fixed** | Two grouped aggregates. Production EXPLAIN: **91 ms for 10 teams in one round trip**, vs ~55 ms *each*; now O(1) round trips. |
| S1-4 settle N+1 + unbounded fetch | **Fixed** | Batched match prime into the identity map; `pending` bounded by `betting.settle_batch_size` (2000), oldest first, with a backlog warning. |
| S3-1 prune seq scan | **Fixed** | Rewritten to drive off `matches` with an id cursor. Production EXPLAIN: **1,774 ms → 14 ms**. |
| Workflow timeout overcommit | **Fixed** | `backfill-wc` hoisted into its own job (it is conditional and rarely runs, but its 90 min counted against the main cap). Main job worst case now **355 min inside a 360 min cap**; auto-trigger preserved by moving the coverage probe with it. |
| S1-1 sharding | **Code done, workflow left single-process** | `--picks --shard i/N --out FILE` + `--picks --collect DIR`. The portfolio phase is extracted into `finalize_picks()` and runs **once** over the union, because the exposure cap is a property of the day, not the shard. 19 tests, incl. shard-then-collect ≡ unsharded for N ∈ {2,3,4,8}. The GitHub matrix is documented in the workflow but not enabled: at current volume the runner setup costs more than it saves. |

Sharding surfaced a **pre-existing reproducibility bug**: the portfolio ranking
score was not a total order, and Python's sort is stable, so tied picks were
resolved by whatever order analysis happened to emit. Re-running `--picks` with
fixtures in a different order produced *different bets*. Ranking is now keyed on
`(-score, match_id, selection)`. Two regression tests pin it; both fail on the
pre-fix code.

Two defects were found *during* implementation and are also fixed: `api_budget`
was created without RLS (Supabase advisor **ERROR**), and migration 001's trigger
function had a mutable `search_path` (advisor **WARN**). Both cleared; the
trigger was re-probed on production afterwards and still behaves correctly.

**Tests: 271 → 356.** 23 of the 25 new concurrency tests fail against the
pre-fix code.

---

## Verdict

**The application logic mostly scales. The execution model does not.**

This is a single-process, single-runner, once-a-day batch job that assumes it can
hold the entire problem in one Python heap for four hours. At 100× the dominant
failures are not slow queries — they are *architectural*: the pipeline has no way
to run two of anything at once, and the moment you try, three separate pieces of
shared state silently corrupt (API budget, pick dedup, history mirror).

The single most important number in this review: **the workflow's step timeouts
already sum to 445 minutes inside a 240-minute job cap.** The system is over
budget on wall-clock *today*, at 1×.

Ranked below. S1 = cannot function at 100×. S2 = data corruption under
concurrency. S3 = measurable waste. S4 = reliability/observability.

---

# S1 — Blocking at 100×

## S1-1. The execution model cannot be scaled horizontally at all

**Evidence.**

```
.github/workflows/daily-picks.yml
  job timeout      : 240 min
  step timeouts    : [90, 65, 15, 50, 150, 60, 15]  = 445 min
  overcommit       : 205 min beyond the job cap
  concurrency      : group: daily-picks, cancel-in-progress: false   ← serialises, never parallelises
  schedule         : one cron, '37 9 * * *'
```

Seven sequential `python -m src.agent.betting_agent` invocations on one runner.
There is no sharding key anywhere — no `--shard i/n`, no league partition, no
work queue. Adding a second runner would not split the work; it would duplicate
it (and corrupt state — see all of S2).

**At 100×:** the `--picks` step alone analyses 13,200 fixtures at
`ANALYSIS_CONCURRENCY=5`. Even at an optimistic 1 s/fixture that is 44 minutes,
and it is followed by *one Claude subprocess spawn per pick*
([match_briefing.py:1291](../src/reporting/match_briefing.py#L1291)) — 13,200
`claude -p` processes, serialised, each doing web research. That step is budgeted
150 minutes. It will not finish; it will not come close.

**Measurable impact:** the job dies at 240 min. Because GitHub *cancels* the job
rather than failing a step, `continue-on-error: true` does not save it and the
day produces nothing.

**Proposed change (large, unavoidable):**
1. Shard the fixture-processing steps by a deterministic key (`match_id % N`) and
   run N matrix jobs. This requires fixing S2-1/S2-2/S2-3 first — sharding on top
   of today's shared state produces duplicate picks and blown API quota.
2. Move the per-pick LLM review out of the critical path onto a queue with its
   own concurrency budget; it is the only step whose cost is linear in picks *and*
   measured in minutes each.
3. Immediately (independent of scale): **reduce the step timeouts so they sum to
   ≤ 240**, or raise the job cap. Today's configuration cannot complete a
   worst-case day by construction.

---

## S1-2. API-Football budget is per-process in-memory — quota enforcement is fiction

**Evidence.** [apifootball_scraper.py:408](../src/scrapers/apifootball_scraper.py#L408)

```python
self._requests_today = 0          # ← reset on every construction
self._daily_limit = 100
```

`_requests_today` is never persisted or read back. Every one of the seven CLI
processes constructs a fresh scraper and starts from zero. The 100/day cap is
enforced *per process*, so today's real ceiling is already ~700, not 100 — the
system only stays under quota because most steps do not call the API.

**At 100×:** with any sharding (S1-1), N workers × 100 requests each. The
"budget" numbers threaded through `remaining_budget()`, `BUDGET_XG`,
`BUDGET_RESERVE` and the odds semaphore
([`_make_odds_semaphore`](../src/scrapers/apifootball_scraper.py#L1508)) all
become decorative.

**Measurable impact:** real API spend scales with worker count, not with a
configured limit. On a paid API-Football tier this is a direct, unbounded cost
line.

**Proposed change:** move the counter to the database — a `api_budget(day,
provider, used)` row incremented with `UPDATE ... SET used = used + 1 WHERE day =
$1 AND used < $2 RETURNING used`. Atomic, works across processes, and the
`RETURNING` tells the caller whether it won the token. Small change; it is the
prerequisite for any parallelism.

---

## S1-3. `_unanalyzable_today()` — one `COUNT` round trip per team

**Evidence.** [betting_agent.py:708-716](../src/agent/betting_agent.py#L708-L716)

```python
for m in club:
    for tid in (m.home_team_id, m.away_team_id):
        if tid not in counts:
            counts[tid] = session.query(Match.id).filter(...).count()
```

Measured on production, one team: `Aggregate ... Execution Time: 4.314 ms`,
75 buffers. Add ~50 ms Supabase pooler RTT → **~55 ms per team**.

Measured batched alternative (one query, `unnest(array[...])` + `GROUP BY`) for
10 teams: **20.2 ms total** — 27× faster at 10 teams, and it is O(1) round trips
regardless of team count.

**At 100×:** 26,400 distinct teams × 55 ms = **~24 minutes** of pure latency,
inside a step budgeted 65 minutes, to compute a boolean per fixture.

**Proposed change:** replace the loop with the single `GROUP BY` shown above.
~15 lines. Impact: 24 min → ~3 s.

---

## S1-4. `settle_predictions()` — unbounded fetch plus a per-pick `session.get`

**Evidence.** [betting_agent.py:2090-2130](../src/agent/betting_agent.py#L2090-L2130)

```python
pending = session.query(SavedPick).filter(SavedPick.result.is_(None)).all()   # no limit
already_settled = session.query(SavedPick).filter(
    SavedPick.settled_at >= utcnow() - timedelta(days=3), ...).all()
for sp in already_settled:
    m = session.get(Match, sp.match_id)        # ← N+1
...
for pick in all_picks_to_process:
    match = session.get(Match, pick.match_id)  # ← N+1
```

Three problems compounding: `pending` has no bound (it grows without limit if
settlement ever falls behind — exactly what happens when upstream results are
late), the 3-day correction window scales with pick volume, and both loops issue
one primary-key lookup per pick.

**At 100×:** ~13,200 picks/day → a 3-day correction window of ~40,000 picks, each
triggering a `session.get(Match, ...)`. At ~50 ms RTT that is **~33 minutes** in a
step budgeted 15 minutes. The step dies, picks stay unsettled, `pending` grows,
and the next run is worse — a positive feedback loop into permanent failure.

**Proposed change:**
- Bound `pending` (`LIMIT`, oldest first) and loop until drained, so a backlog
  degrades gracefully instead of timing out.
- Replace both `session.get` loops with one batched `Match.id.in_(...)` fetch
  keyed into a dict — the same pattern already used in `tune_ensemble_weights`.
- Column-project both `SavedPick` reads (they pull 26 columns including
  `review_reason VARCHAR(500)`).

Impact: ~33 min → seconds, and removes an unbounded-growth failure mode.

---

# S2 — Data corruption under concurrency

These are latent today only because `concurrency: cancel-in-progress: false`
serialises runs. **Every one of them fires the moment S1-1 is fixed.** They must
be fixed *before* sharding, not after.

## S2-1. `saved_picks` has no unique constraint — dedup is a TOCTOU race

**Evidence.** Live index inventory:

```
saved_picks : saved_picks_pkey        UNIQUE  (id)
saved_picks : ix_saved_picks_match_id         (match_id)      ← not unique
```

Dedup is entirely application-level read-then-write
([betting_agent.py:1714-1733](../src/agent/betting_agent.py#L1714-L1733)):
`SELECT ... .first()` → `if existing: continue` → `session.add(...)`. Two
concurrent workers both read "no duplicate" and both insert.

**Impact:** duplicated tracked bets. Not cosmetic — it corrupts *every* downstream
statistic: win rate, ROI, Brier score, the Bayesian weight learner, and the
drawdown circuit breaker that sizes real stakes.

**Proposed change:** `CREATE UNIQUE INDEX CONCURRENTLY ix_saved_picks_dedup ON
saved_picks (match_id, selection, pick_date);` and convert the insert to
`INSERT ... ON CONFLICT DO NOTHING`. The database becomes the arbiter; the
application check stays as a cheap fast path. (Check for existing duplicates
before adding the index.)

## S2-2. Odds upsert is read-then-write against a UNIQUE index — a conflict kills the whole batch

**Evidence.** `odds` *does* have `ix_odds_match_bookie_market UNIQUE (match_id,
bookmaker, market_type, selection)`. But
[`_save_odds_from_set`](../src/scrapers/apifootball_scraper.py#L1710) builds an
in-memory `existing_index` and then inserts whatever is missing.

Under concurrency, two writers both miss, both insert, and the loser gets
`IntegrityError`. On PostgreSQL a constraint violation **aborts the entire
transaction** — so it is not one lost odds row, it is every odds row in that
fixture's batch, and `get_session()` re-raises, killing the step.

**Proposed change:** `INSERT ... ON CONFLICT (match_id, bookmaker, market_type,
selection) DO UPDATE SET odds_value = EXCLUDED.odds_value` — preserving
`opening_odds` with `COALESCE(odds.opening_odds, EXCLUDED.odds_value)`. One
statement, no read, race-free, and it also removes the per-match `SELECT`.

## S2-3. History mirror: concurrent syncs can silently persist stale values *(my code, from the previous pass)*

**Evidence.** [history_mirror.py](../src/data/history_mirror.py) `_write_atomic`
writes the Parquet file and the metadata file as two separate atomic renames.
That is correct for *crash* safety (which is what it was designed for) but not
for *concurrency*:

```
A: writes parquet_A (watermark_A)
B: writes parquet_B (watermark_B, older)
A: writes meta_A   (watermark_A)
→ meta_A + parquet_B
```

Next sync resumes from `watermark_A` and never re-fetches the rows in
`(watermark_B, watermark_A]` that exist only in `parquet_A`. The row-count
reconcile catches this **only if those rows were inserts**. If they were
*updates* — a settled result, an xG backfill — the count matches and the stale
values persist indefinitely.

**Impact:** silently wrong model inputs, with no error and no log line. This is
the worst failure mode in the review because it is invisible.

**Proposed change:** write both artefacts as a single file (embed the metadata in
the Parquet key-value metadata, or write one `.npz`/directory and rename once),
**or** take an exclusive `fcntl`/`msvcrt` lock on a sidecar file for the duration
of the sync. The lock is ~10 lines and also prevents N workers doing N redundant
full resyncs on a cold cache.

## S2-4. PostgreSQL's 65,535 bind-parameter ceiling

**Evidence.** [feature_engineer.py:163-166](../src/features/feature_engineer.py#L163-L166)

```python
or_(Match.home_team_id.in_(all_team_ids),
    Match.away_team_id.in_(all_team_ids))     # ← the same list, bound twice
```

`all_team_ids` is a Python set rendered as inline bind parameters. At 100×,
13,200 fixtures → ~26,400 distinct teams → **52,800 bind parameters in one
statement**, against a hard protocol limit of 65,535. `preload_batch` also passes
`match_ids` to queries 1 and 2.

**Impact:** not slow — a hard `PROTOCOL_VIOLATION` failure. `preload_batch`
catches its own exceptions and sets `_preload_cache = None`, so the pipeline
silently degrades to per-fixture queries: correct results, but 13,200 × ~10
round trips ≈ hours.

**Proposed change:** pass large id sets as a single array parameter
(`= ANY(:ids)` via `postgresql.ARRAY`), or a temporary table / `VALUES` join.
One bind parameter regardless of size, and it plans better than a giant `IN`.

---

# S3 — Measurable waste

## S3-1. `prune_old_odds` is an unbounded single-transaction DELETE with a sequential scan

**Evidence.** `EXPLAIN (ANALYZE, BUFFERS)` on production, today:

```
Delete on odds  (actual time=1770.371..1770.375 rows=0)
  ->  Hash Join
        ->  Seq Scan on odds  (actual time=47.475..1708.385 rows=214683)
              Filter: (NOT (ANY (match_id = (hashed SubPlan 1).col1)))
Execution Time: 1773.721 ms
```

It scans **214,683 of 317,657 odds rows and deletes zero**, every single day,
inside `daily_update()`.

**At 100×:** ~31.7 M rows scanned (~3–4 min), and on the days it *does* match, it
deletes potentially millions of rows in one transaction — a long `RowExclusiveLock`,
a WAL spike, and table bloat that autovacuum then has to chase.

**Proposed change:** batch it (`DELETE ... WHERE id IN (SELECT id ... LIMIT
10000)` in a loop, committing each batch) and switch `NOT IN` to `NOT EXISTS`
(also the NULL-safe idiom). Add a partial index or drive it off `match_date` so
the scan is an index range, not a full table read. Impact: 1.8 s → ~50 ms on
no-op days; bounded lock duration on real ones.

## S3-2. Thread pool and connection pool are sized independently

**Evidence.** `pool_size=5, max_overflow=10` → **15 connections max**
([database.py:83-89](../src/data/database.py#L83-L89)). `run_in_executor(None,
...)` uses the default executor: `min(32, cpu_count + 4)` = **24 threads** on this
machine, more on a bigger runner.

It works today only because two unrelated constants happen to be small:
`BATCH_SIZE = 10` and `ANALYSIS_CONCURRENCY = 5`. Nothing enforces the
relationship, and `ANALYSIS_CONCURRENCY` is documented as a tunable env var —
setting it to 20 produces `QueuePool limit of size 5 overflow 10 reached` after a
30-second block, presenting as a mysterious timeout.

**Proposed change:** derive one from the other (`pool_size = max(concurrency,
batch_size) + headroom`), or assert the invariant at startup. Zero runtime cost,
removes a foot-gun that will absolutely be stepped on during a scale-up.

## S3-3. `_sync_create_features` builds and tears down an event loop per match

**Evidence.** [betting_agent.py:41-49](../src/agent/betting_agent.py#L41-L49) —
`asyncio.new_event_loop()` … `loop.close()` inside the per-match function that
runs in a worker thread.

**At 100×:** 13,200 loop create/destroy cycles per training run. Measurable but
second-order (~1 ms each ≈ 13 s); worth fixing only while touching that code.
Listed for completeness, not urgency — a thread-local loop reused across the
batch removes it.

## S3-4. Unbounded in-process caches

- `_analysis_cache` ([betting_agent.py:103](../src/agent/betting_agent.py#L103)) —
  holds a full `MatchAnalysis` per fixture, cleared only in `get_daily_picks`.
  13,200 analyses × predictions + recommendations is hundreds of MB.
- `match_history` module cache holds **every** completed match: 3.8 M `MatchRow`
  objects at 100× ≈ **1 GB resident**, in every process, whether it needs history
  or not.

**Proposed change:** the history cache is the one that matters. At 3.8 M rows the
right representation is not a Python object list but the pandas/Arrow frame the
mirror already produces — Elo and Poisson could consume columnar arrays directly
(`numpy` vectorised) instead of iterating objects. That is a real refactor, but
it is also the only way this stays in memory at 100×.

---

# S4 — Reliability and observability

## S4-1. `continue-on-error: true` on every core step

Seven core steps all swallow failure. There *is* a compensating alert step, so
this is not silent — but the pipeline has no concept of a step being *required*.
A day where `--update` fails still runs `--picks`, which generates picks from
stale odds and sends them to Telegram as if nothing happened.

**Proposed change:** classify steps. `--update` failing should skip `--picks`,
not proceed with stale inputs. `if: steps.update.outcome == 'success'` on the
picks step is a one-line change with a real correctness payoff.

## S4-2. No retry or backoff on database operations

`pool_pre_ping=True` handles a dead connection at checkout, and `pool_recycle=300`
handles Supabase scale-to-zero. Neither handles a transient failure *mid-statement*
— a pooler restart, a failover, a deadlock. There is no `tenacity`-style retry
anywhere in the DB layer.

**At 100×** on a busier instance, transient errors go from "never" to "daily".

**Proposed change:** wrap `get_session()` in a retry for the retryable SQLSTATE
classes (`40001` serialization failure, `40P01` deadlock, `08006` connection
failure). Must be paired with S2-1/S2-2 idempotency — retrying a non-idempotent
write is worse than failing.

## S4-3. No idempotency key on a pipeline run

A re-run redoes everything: re-scrapes, re-analyses, re-spawns LLM reviews. The
only guards are the ad-hoc ones (`briefings_sent.json`, the pick dedup that S2-1
shows is racy). At 100× a re-run is hours of duplicated API spend.

---

## Things I checked and am *not* flagging

Being explicit about the negative results, so this reads as a review rather than
a list of everything I could think of:

- **Telegram message limits** — `_send_chunked` already splits at 4000 chars and
  handles the supergroup-migration error. Fine at 100×.
- **`odds` unique index** — correctly defined; the problem is how the code writes
  through it (S2-2), not the schema.
- **Column projections and the history mirror's incremental sync** — from the
  previous pass; the sync is O(changes) and verified against production.
- **`teams` indexes** — added last pass, planner confirmed using them.
- **`expire_on_commit=False`** — deliberate and documented; the detached-instance
  trap it prevents is real. Not a leak risk given short-lived sessions.
- **The `WHEN (OLD.* IS DISTINCT FROM NEW.*)` trigger** — correct, cannot
  self-fire, verified on production.

---

## Recommended sequencing

The dependencies matter more than the individual fixes:

| Order | Item | Why first |
|---|---|---|
| 1 | S2-1, S2-2, S2-4, S1-2 | **Prerequisites for any parallelism.** Sharding before these is actively harmful. |
| 2 | S1-3, S1-4, S3-1 | Pure wins, no dependencies, ~1 day of work, removes ~57 min/day of latency at 100×. |
| 3 | S2-3 | Small, but the failure is invisible — fix before anyone runs two workers. |
| 4 | S4-1, S3-2 | One-line-ish changes with real payoff. |
| 5 | S1-1 | The big one. Only tractable after 1–4. |
| 6 | S3-4 | Columnar history. Needed around 10×, not before. |

**Quick wins with the best measured ratio:** S1-3 (24 min → 3 s, ~15 lines),
S3-1 (1.8 s/day of pointless scanning → ~50 ms, ~10 lines), S1-4 (33 min →
seconds, ~30 lines). Those three are roughly a day's work and remove the two
timeout-driven failure loops.
