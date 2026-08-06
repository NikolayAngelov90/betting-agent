# Egress & correctness — follow-up pass

**Date:** 2026-08-06
**Project:** `betting-agent` (`nhlurscyrlvpjzapmqcr`), Supabase Postgres 17, eu-central-1
**Predecessor:** [egress-audit.md](egress-audit.md) — pass 1 took ~150–165 MB/day down to ~30–37 MB/day.

---

## Remaining issues fixed

### 1. Preload and live feature generation were not identical *(Priority 1)*

**Root cause.** `preload_batch` cached history for exactly the two teams playing
the fixture. Seven feature branches then read that cache and treated *"team not
present"* or *"fewer rows than I asked for"* as **"this team has no history"**,
returning zeroed features instead of asking the database. Three families broke:

| Family | What actually happened |
|---|---|
| **League standings** | `_get_league_standings` ranks every club in the division. The other ~18 clubs were not in the cache, so they scored **0 points** — the fixture's own two teams came out 1st and 2nd, every time. `league_position`, `title_gap`, `relegation_gap` and `position_difference` were therefore meaningless in the `--picks` path *and* in ML training. |
| **Referee stats** | The cached branch scanned `team_history` for matches with that official — i.e. only those two clubs' games — and skipped the 365-day window the live query applies. Averages came from a handful of matches instead of the referee's last 30. |
| **Head-to-head** | Found while writing the regression tests: meetings were read from the 60-row / 365-day team window, so older meetings silently vanished. The adversarial fixture shows **preload=5 vs live=10** meetings. |

**Fix.** The question "can the cache answer this exactly?" now has a provable
answer, asked in one place ([src/features/preload_cache.py](../src/features/preload_cache.py))
instead of seven.

Cached row lists are *prefixes of the true history in descending date order*.
Filtering a descending prefix by any predicate yields a descending prefix of the
filtered history. So a cached answer equals `... ORDER BY match_date DESC LIMIT n`
exactly when **either** at least `n` cached rows match the predicate (every row
the database would return is at least as recent as our n-th, so it is in our
prefix) **or** the list is not truncated at all. Anything else returns `None`
and the caller runs its normal live query.

```python
# before — src/features/team_features.py
rows = preload_cache.get("team_history", {}).get(team_id, [])
filtered = [m for m in rows if ...][:num_matches]
if not rows_subset:
    return self._empty_form_features()      # ← a cache miss, reported as "no history"

# after
rows_subset = _pc.team_rows(preload_cache, team_id, limit=num_matches, predicate=...)
if rows_subset is not None:
    ...                                      # provably exact
# otherwise fall through to the live query, unchanged
```

`preload_batch` also gained the scopes those wider families need, so the
fallback stays rare:

| Scope | Source | Why |
|---|---|---|
| team history (34 cols) | query | the fixture's own teams; needs the stat columns |
| referee history (11 cols) | query, `row_number()` capped at the live `LIMIT 30`, same 365-day window | an official's matches span the whole division |
| **league history** | **derived** from the in-memory completed-match history | standings need every club; querying it would cost ~3.2 MB on a busy 32-league matchday for rows the process already holds |
| **H2H history** | **derived**, same source | needs the complete record of a pairing |

Deriving 5 and 6 is possible because the history the models already load carries
exactly the columns those two consume (ids, date, goals, league) — so they are
free.

**This changes predictions.** The pre-fix `--picks` and `--train` paths fed the
model garbage league-position features and truncated H2H. Correcting them moves
model output — for the better, but it is a real change, not a no-op.

### 2. `--train` ignored `models.ml_retrain_days` *(Priority 3)*

**Classification: a Python bug in the CLI branch.** Not configuration (the key
is read correctly elsewhere) and not the workflow (its step is literally named
*"Retrain ML models (if stale)"* — the intent was always conditional).

`daily_update()` *does* check `_ml_models_stale()`. But CI runs
`--update --skip-ml-retrain` so the retrain is deferred to a dedicated `--train`
step with its own timeout budget. The staleness decision was therefore made in
the `--update` process and thrown away, and `--train` — the process that
actually trains — never asked. The pipeline rebuilt features for 500 matches and
refit Poisson/Elo **every day** despite `ml_retrain_days: 3`.

Fixed in [betting_agent.py](../src/agent/betting_agent.py) by gating the branch
on `_ml_models_stale(max_age_days=config.models.ml_retrain_days)`, with
`--train --force` retained as the operator escape hatch.

### 3. `analyze_fixture` re-downloaded odds it had just preloaded

`get_daily_picks` calls `preload_batch` with exactly the fixture ids it is about
to analyse, then `analyze_fixture` queried every odds row again in the same
process. Now reads the cache, falling back to the query for any fixture outside
the batch.

### 4. Two more unprojected reads

`injury_scraper` selected `DISTINCT` over all 47 match columns to use four of
them; `flashscore_scraper`'s stat-enrichment query fetched whole rows to use
`(id, flashscore_id)`.

---

## Cross-process history mirror *(Priority 2)*

```
PostgreSQL → incremental sync → local Parquet → Arrow/pandas → training & prediction
```

[src/data/history_mirror.py](../src/data/history_mirror.py). After pass 1 the
completed-match history cost 3.8 MB **per process**, and the daily job runs
seven. The mirror lives in `data/models/`, which `actions/cache` already
persists — so one copy serves all seven steps of a run *and* survives to the
next day.

**Measured:** 1,202 kB Parquet for 38,219 rows; 4.8 MB in memory as a frame.

Correctness rules, each with a test:

| Rule | Why |
|---|---|
| Watermark is the **newest `updated_at` actually received**, never `now()` | a row committed mid-sync cannot be skipped |
| Next sync re-asks with `>=`, not `>` | a row sharing the watermark's exact timestamp is re-fetched rather than missed; merging is by primary key so the overlap is harmless |
| Delta query does **not** filter `is_fixture`/`home_goals` | membership is re-decided per row, so a fixture gaining a result enters and a cleared result leaves |
| Row-count reconcile every sync | `updated_at` cannot record a delete; a mismatch forces a full resync |
| Parquet written to temp + atomic rename, metadata written **after** | a crash between the two leaves an older watermark → next run re-fetches a small overlap idempotently |
| Schema version in metadata | a column-set change forces a resync instead of serving a frame missing a column |

Every failure path — no `updated_at`, no pyarrow, SQLite, corrupt file, missing
metadata — degrades to reading from the database. **Degrading costs egress,
never correctness.**

Verified on production: the delta query uses `ix_matches_updated_at`
(Index Scan, 5 buffers, 0.105 ms).

---

## Database migrations

Files: [migrations/001_history_mirror_and_indexes.sql](../migrations/001_history_mirror_and_indexes.sql)
and its `.rollback.sql`. **Applied to production and verified.**

### `matches.updated_at` — required, not optional

The mirror must answer "what changed since my last sync?". `matches` recorded
only `created_at`, which detects inserts but not updates — and this pipeline
updates existing rows constantly: settlement writes goals onto a row inserted
days earlier as a fixture; `backfill_match_stats` writes xG onto rows years old;
the Flashscore scraper rewrites stats in place. Without a modification timestamp
the mirror can only redownload everything (the 12 MB read we are eliminating) or
serve stale rows to the models.

No stock alternative exists: `xmin` is a wrapping 32-bit transaction id, is not
indexable, and cannot be range-queried across a freeze.

### The trigger — why not SQLAlchemy `onupdate=`

`onupdate` fires only for ORM UPDATEs on mapped attributes. This codebase also
writes through bulk `query(...).update(synchronize_session=False)`, raw `text()`
statements in `database.py` and `scripts/`, and the Neon→Supabase merge scripts.
A row updated through any of those would be **invisible to the mirror forever** —
a silent, permanent staleness bug. The trigger is the only place that sees every
writer.

The `WHEN (OLD.* IS DISTINCT FROM NEW.*)` clause skips no-op updates, which
scrapers issue constantly; without it the mirror would redownload rows that did
not change. Comparing at BEFORE-time is safe because `updated_at` has not been
touched yet, so the trigger cannot re-fire on its own write.

**Verified on production** with a throwaway row (since deleted): a no-op update
left `updated_at` unchanged; a real change moved it.

### Indexes — measured, not speculative

`pg_stat_user_tables` recorded **58,189 sequential scans on `teams` having read
57,342,537 tuples**, on a 1,286-row / 16-page table with no index but its primary
key. EXPLAIN (ANALYZE, BUFFERS) on production, before and after:

| Query | Call sites | Before | After | Gain |
|---|---|---|---|---|
| `WHERE name = $1` | apifootball step 1, flashscore, footballdataorg, historical_loader | Seq Scan `cost=0.00..32.60` buffers=10, 689 filtered | Index Scan `cost=0.28..2.50` buffers=3 | **13× cost, 3× buffers** |
| `WHERE league = $1` | standings team list, fuzzy candidate pools | Seq Scan `cost=0.00..32.60` buffers=16, 1260 filtered | Bitmap Heap Scan `cost=1.58..16.95` buffers=7 | **1.9× cost, 2.3× buffers** |
| `WHERE apifootball_team_id = $1` | apifootball step 0 — the id-first match that stops team duplication; misses scan the whole table | Seq Scan `cost=0.00..32.60` buffers=10 | Index Scan `cost=0.28..2.50` buffers=2 | **13× cost** |
| `WHERE updated_at >= $1` | mirror delta sync | *(column did not exist)* | Index Scan buffers=5, 0.105 ms | new |

The planner chose the index in all four cases. Write cost is negligible —
`teams` gains ~1,300 rows/year.

**Rollback** drops the trigger, function, column and all four indexes. It is safe
at any time: `history_mirror.supports()` probes for the column and degrades to a
database read when it is gone. Predictions are unaffected; only egress rises.

---

## Estimated additional egress reduction

Modelled on a busy in-season matchday reconstructed from production
(2024-10-05: 132 matches, 32 leagues, 253 teams, 16 referees), using wire sizes
measured on the live database.

| # | Source | Per day | Note |
|---|---|---:|---|
| 1 | `preload_batch` team history (34 cols, 236 B/row) | **3.07 MB** | picks 1.53 + train 0.83 amortised + tuning 0.71 |
| 2 | `preload_batch` odds (6 cols, 77 B/row) | **0.71 MB** | ~9,240 rows |
| 3 | Occasional full mirror resync (cache miss ≈ 1 day in 10) | **0.38 MB** | 3.8 MB amortised |
| 4 | SavedPick reads (projected, 76 B/row) | **0.20 MB** | several call sites |
| 5 | Scraper budget-bounded lists | **0.10 MB** | xG / stats / detail, 25–100 rows each |
| 6 | **History mirror delta sync** | **0.06 MB** | ~300 changed rows × 143 B |
| 7 | Referee scope | **0.05 MB** | 16 × 30 × 95 B |
| 8 | `get_daily_picks` fixture list (unprojected) | **0.05 MB** | 132 × 368 B |
| 9 | Injury / league / H2H scopes | ~0 | derived or 4-column |
| 10 | Count, probe and aggregate queries | ~0 | tens of bytes each |

**Total ≈ 4.6 MB/day busy (~140 MB/month); ~1.2 MB/day in the current quiet
season.**

| | Per day | Per month | vs 5 GB quota |
|---|---:|---:|---:|
| Before pass 1 | 150–165 MB | 4.5–5.0 GB | **at/over the limit** |
| After pass 1 | 30–37 MB | ~1.0 GB | 5× headroom |
| **After pass 2** | **~4.6 MB** | **~140 MB** | **~36× headroom** |

Additional reduction from this pass: **~85–88%**. Cumulative: **~97%**.

---

## CPU improvements

These were not the objective and are side effects, but they are real:

- **`teams` lookups**: 58k sequential scans of 985 rows each become index scans
  of 2–7 buffers. 12.2 ms → 0.75 ms on the name lookup.
- **`~10 → ~1` model fits' worth of history I/O per process**, and with the
  mirror the per-process read becomes a local Parquet load (~1.2 MB from disk)
  instead of a network round trip.
- **`--train` runs 1 day in 3 instead of 3 in 3** — two days out of three now
  skip 500 feature builds and a Poisson/Elo refit entirely. This is the single
  largest wall-clock saving in the pipeline (the step is budgeted 50 minutes).
- **`injury_scraper`** no longer asks Postgres to `SELECT DISTINCT` over 47
  columns to use four.
- **N+1 removal**: `match_briefing`'s WC dedup went from 3 queries per fixture
  (2 × `session.get(Team)` + a `COUNT`) to two batch queries for the whole list.

---

## Files modified

**New**

| File | Purpose |
|---|---|
| `src/features/preload_cache.py` | completeness-checked cache accessors — the rule that makes preload ≡ live |
| `src/data/history_mirror.py` | Parquet mirror with incremental sync |
| `migrations/001_history_mirror_and_indexes.sql` | forward migration, justified inline |
| `migrations/001_history_mirror_and_indexes.rollback.sql` | rollback |
| `tests/test_history_mirror.py` | 19 tests |
| `tests/test_train_scheduling.py` | 7 tests |
| `tests/test_sql_compiles_on_postgres.py` | 14 tests |

**Changed**

| File | Change |
|---|---|
| `src/features/feature_engineer.py` | referee/league/H2H scopes; xG + situational branches on the new accessors |
| `src/features/team_features.py` | form / international / momentum branches; league-scope routing |
| `src/features/h2h_features.py` | pairing scope |
| `src/agent/betting_agent.py` | `--train` staleness gate; `analyze_fixture` odds from cache |
| `src/data/models.py` | `matches.updated_at` + index; three `teams` indexes |
| `src/data/match_history.py` | three-tier load: memory → mirror → database |
| `src/scrapers/injury_scraper.py`, `flashscore_scraper.py` | two projections |
| `.github/workflows/daily-picks.yml` | cache step documents the mirror |
| `requirements.txt` | `pyarrow>=15.0.0` (optional at runtime) |
| `tests/test_features.py` | 5 tests updated to the new contract; the MagicMock `preload_batch` test rewritten against a real database |

---

## Before vs after

| | Before pass 2 | After |
|---|---|---|
| Preload vs live features | diverged on standings, referee, H2H | **identical, no exclusions**, at `rel=1e-9` |
| League position features | fixture's teams always 1st and 2nd | true division ranking |
| Referee stats | computed from 2 clubs' games, no date window | last 30 matches in 365 days, all clubs |
| H2H meetings | truncated by the 60-row / 365-day window (5 of 10) | complete |
| Cache miss semantics | silently returned zeros | returns `None`, caller queries |
| `--train` | every day, ignoring `ml_retrain_days: 3` | only when stale, `--force` to override |
| History read | 3.8 MB × ~5–7 processes/day | ~0.06 MB/day of deltas |
| Change detection | `(count, max_id, max_date)` probe | `updated_at` + trigger, catches every writer |
| `teams` lookups | 58k seq scans, 57M tuples | index scans |
| Tests | 271 | **329** |

---

## Validation

- **Full suite: 329 passed.** (271 before this pass; +58 new.)
- **Feature equality with no exclusions**, on an adversarial fixture built to
  break the cache: ~960 days of history so the 365-day window truncates; 12
  clubs so standings have real spread; a referee who mostly officiates *other*
  clubs' matches; H2H meetings outside the window. Asserted across four
  configurations — live window, training window, `as_of_date`, and a deliberately
  starved cache (`cap_per_team=3, league_cap=5, referee_cap=2`) that forces
  fallback almost everywhere.
- **The new tests fail on the pre-fix code** — verified by stashing: standings
  order `[2,1,3,4,5,6…]` vs `[7,3,6,8,10,12…]`, referee stats all-zero vs 17.61
  fouls/match, H2H 5 vs 10. The `--train` tests: 2 of 7 fail on baseline.
- **Model output equality**: Elo ratings and Poisson strengths compared
  database-sourced vs mirror-sourced to `abs=1e-12`.
- **SQL compilation**: every modified query rendered to PostgreSQL and asserted
  on. This is not ceremony — it caught a real defect in pass 1 (a projected join
  with no determinable left side) that SQLite and the MagicMock tests both
  missed.
- **Production**: migration applied and verified; trigger behaviour probed with a
  throwaway row; the mirror's delta query EXPLAINed against live data.

---

## Remaining technical debt

1. **`preload_batch` query 3 is now 67% of all remaining egress** (3.07 MB/day).
   It cannot come from the mirror because the mirror deliberately carries only
   the 9 goal-level columns. See the final section.
2. **`get_daily_picks`' fixture list is still unprojected** (47 columns, ~132
   rows = 49 kB/day). Left alone deliberately: it is a critical path, the payoff
   is under 50 kB, and the risk/benefit does not justify touching it.
3. **Scraper write paths still fetch whole rows.** They must — they mutate what
   they read — and they are bounded by a day's fixtures or an API budget.
4. **The mirror's Postgres path has not been executed end-to-end.** Its SQL was
   compiled, EXPLAINed and run read-only against production, and its logic is
   covered by 19 SQLite tests, but no process has actually built the Parquet file
   from Supabase. The first CI run will.
5. **`_migrate_missing_indexes()` will try to create the three `teams` indexes**
   on the next `--init`. They already exist, so the inspector skips them; harmless
   but worth knowing.
6. **`odds` (317k rows, 66 MB) is never mirrored.** Correct — odds change
   constantly, so an incremental sync would approach a full one daily.

---

## Risk assessment

| Risk | Severity | Mitigation |
|---|---|---|
| **Predictions change.** Standings, referee and H2H features were wrong and are now right, so model output moves. | **High — expected and intended** | Flagged prominently. Worth watching the next few days of picks; the ML models should be retrained (they were trained on the broken features). |
| ML models currently in `data/models/` were trained on corrupted league-position features | Medium | They will retrain on the next staleness trigger. Forcing `--train --force` once is the faster route. |
| Mirror serves stale rows | Low | Trigger catches every writer; watermark uses received-max not `now()`; `>=` overlap; count reconcile; schema version. Nineteen tests. |
| Mirror corrupts or is interrupted | Low | Atomic rename, metadata written second, every failure path degrades to a database read. |
| Trigger overhead on writes | Low | One assignment per changed row; no-ops skipped by the `WHEN` clause. |
| `actions/cache` eviction | Low | First sync rebuilds; costs 3.8 MB once. |
| pyarrow unavailable | Low | `MirrorUnavailable` → database read. Suite skips those tests via `importorskip`. |
| Migration rollback needed | Low | Rollback SQL provided and safe at any time; code probes for the column. |
| More live queries than before when the cache cannot prove exactness | Low–Medium | By design — correctness over round trips. The added scopes keep it rare; the starved-cache test proves the fallback path is correct. |

---

## Is further egress reduction realistically possible?

**Yes — one meaningful step remains, and it is not an architectural change.**

The remaining 4.6 MB/day is dominated by a single query: `preload_batch`'s
34-column team history (3.07 MB/day, ~67%). It exists because feature
engineering needs shots, possession, corners, cards, saves, offsides and free
kicks — columns the mirror does not carry.

**Widening the mirror to those columns would make that query free.** Cost:
104 B/row → 236 B/row, so the Parquet file goes from ~1.2 MB to roughly 3 MB and
the in-memory frame from 4.8 MB to ~11 MB; a cold resync costs 9 MB instead of
3.8 MB. That would take the pipeline to **~1.5 MB/day (~45 MB/month)**. Same
architecture, same sync logic, one wider column list — but it trades a bigger
cached artifact and more resident memory for ~3 MB/day, which at 36× headroom is
not obviously worth doing today. It is the right move if the fixture volume grows
substantially.

**Below roughly 1 MB/day, no.** What is left is irreducible without changing what
the system does:

- **Odds** (~0.71 MB/day) are the one thing that genuinely changes every run —
  caching them is caching the signal.
- **Delta syncs** (~0.06 MB/day) are already proportional to real change; the
  only way lower is fetching less than what changed, i.e. being stale.
- **Fixture metadata, referee scope, SavedPick reads** are already
  column-projected and bounded by a day's activity.

So: the floor with the current architecture is about **1–1.5 MB/day**, we are at
**4.6 MB/day**, and the gap is one optional mirror-schema widening. Against a
5 GB quota, all three numbers are noise — the egress problem is solved, and the
remaining work is optimisation for its own sake rather than for the bill.
