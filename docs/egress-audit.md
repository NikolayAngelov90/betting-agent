# Supabase egress audit — betting-agent

**Date:** 2026-08-06
**Project:** `betting-agent` (`nhlurscyrlvpjzapmqcr`), Postgres 17.6, eu-central-1
**Goal:** cut egress under the 5 GB/month free-plan quota without changing behaviour.

---

## 0. Scope correction

The brief assumes the Supabase JS/PostgREST client (`select("*")`, RPC, Storage,
Auth). **This project does not use it.** It talks to Supabase Postgres directly
over the connection pooler via SQLAlchemy + psycopg2 (`DATABASE_URL`,
[src/data/database.py:69-89](../src/data/database.py#L69-L89)).

So the inventory the brief asks for maps like this:

| Asked for | Reality here |
|---|---|
| Supabase clients | One: `DatabaseManager` (SQLAlchemy engine, pool_size 5, overflow 10, recycle 300s). SQLite fallback when `DATABASE_URL` is unset. |
| `select("*")` | `session.query(Match)` — the ORM equivalent, and just as costly. 154 query sites across 16 modules. |
| RPC | None. |
| Storage requests | None. All `storage.*` tables are empty; models/caches live on the GitHub Actions filesystem. |
| Auth requests | None. All `auth.*` tables have 0 rows. No `refresh_token` traffic. |
| Realtime | None. `messages`/`subscription` empty. |

Everything below therefore concerns **SQL result-set size**, which is exactly
what Supabase bills as egress.

### Table sizes (live)

| Table | Rows | Total size | Avg wire bytes/row for `SELECT *` |
|---|---:|---:|---:|
| `odds` | 317,657 | 66 MB | 88 B |
| `matches` | 38,344 (38,219 completed) | 17 MB | **333 B** (46 columns) |
| `saved_picks` | 1,018 | 392 kB | ~335 B (26 columns) |
| `players` | 2,315 | 296 kB | — |
| `teams` | 1,286 | 224 kB | — |
| `injuries` | 17 | 144 kB | — |

"Wire bytes" is the psycopg2 text-protocol `DataRow` size: 6 bytes per row plus
a 4-byte length prefix and the text encoding for each column. Postgres 17 has no
libpq compression, so wire bytes ≈ billed egress.

---

## 1. Dependency graph

```
GitHub Actions cron '37 9 * * *'  →  ONE job, SEVEN Python processes
│
├─ --backfill-wc      (conditional)
├─ --update           → daily_update()
│                         ├─ scrapers write fixtures/odds/results
│                         └─ predictor.fit() ─────────────┐
│                            (+1 more if backfill ran)    │
├─ --settle           → settle_predictions()              │
│                       learn_from_settled()              │
│                         ├─ predictor.fit() ─────────────┤
│                         └─ tune_ensemble_weights()      │
│                              ├─ fit(as_of_date) ────────┤     Elo.fit()
│                              └─ fit()  [restore] ───────┼──►  SELECT matches.*
├─ --train            → train_ml_models()                 │     WHERE is_fixture=false
│                         ├─ predictor.fit() ─────────────┤       AND home_goals IS NOT NULL
│                         ├─ SELECT matches.* LIMIT 500   │     (NO LIMIT)
│                         └─ preload_batch(cutoff=0)      │     38,219 rows × 333 B
├─ --picks            → predictor.fit() ──────────────────┤     = 12.0 MB  ✗ per call
│                       preload_batch(today's fixtures)   │
│                       analyze_fixture() × ~90           │     Poisson.fit()
├─ --update-results   → scrape_results()                  └──►  same query LIMIT 5000
└─ --settle (again)   → learn_from_settled()  ×3 fits           = 1.65 MB ✗ per call
```

The returned data feeds:

```
Elo.fit()      →  ratings dict          →  ensemble.predict() / feature_engineer.elo_ratings
Poisson.fit()  →  team strengths        →  ensemble.predict()
preload_batch  →  _preload_cache        →  create_features() → ML models
get_stats()    →  stats dict            →  Telegram --report / --stats
```

**The critical structural fact:** those seven CLI invocations are seven separate
OS processes. Every one starts with empty in-memory state, so `predictor.fit()`
ran the full-table query from scratch **~10 times a day**, and `Elo.fit()` asks
for 46 columns while consuming 5.

---

## 2. Top 10 egress sources (before)

| # | Source | File | Per call | Calls/day | Per day |
|---|---|---|---:|---:|---:|
| 1 | `Elo.fit()` — `SELECT matches.*`, no LIMIT | `models/elo_system.py:55` | 12.0 MB | ~10 | **~120 MB** |
| 2 | `Poisson.fit()` — `SELECT matches.*` LIMIT 5000 | `models/poisson_model.py:85` | 1.65 MB | ~10 | **~17 MB** |
| 3 | `preload_batch()` training history — 10,578 rows × 46 cols | `features/feature_engineer.py:127` | 3.4 MB | 1 | 3.4 MB |
| 4 | `preload_batch()` picks history + odds | same | ~1.5 MB | 1 | 1.5 MB |
| 5 | Tuning preload (`cutoff_days=0`) | `agent/betting_agent.py:2617` | ~1 MB | 2 | 2 MB |
| 6 | Scraper team resolution — full `Team` rows per unresolved name | `scrapers/apifootball_scraper.py:1387`, `flashscore_scraper.py:1689` | ~110 kB | 100s | ~2 MB |
| 7 | `get_stats()` — `SELECT saved_picks.*` full table | `agent/betting_agent.py:2293` | 333 kB | 2–3 | ~1 MB |
| 8 | `analyze_fixture()` per-match `SELECT odds.*` | `agent/betting_agent.py:773` | ~1.5 kB | ~90 | ~1 MB |
| 9 | Per-fixture feature queries (form, H2H, xG, referee, league, situational) | `features/*` | small | 100s | ~1 MB |
| 10 | Injury features — `SELECT injuries.*` + lazy-loaded `i.player` (N+1) | `features/injury_features.py:46` | small | ~180 | <0.5 MB |

**Total ≈ 150–165 MB/day ≈ 4.5–5.0 GB/month** — squarely at the 5 GB free-tier
ceiling. Sources 1 and 2 alone are ~83%.

---

## 3. Optimisations implemented

### 3.1 Shared, column-projected match-history cache (the big one)

**File:** `src/data/match_history.py` (new), wired into
`src/models/elo_system.py` and `src/models/poisson_model.py`.

**Problem.** Elo and Poisson each ran `session.query(Match)` — SQLAlchemy emits
`SELECT matches.*`, all 46 columns — over the same slice of the table, on every
one of the ~10 `predictor.fit()` calls a day. Elo had no `LIMIT` at all. Elo
reads 5 of those columns; Poisson reads 8.

**Why it causes egress.** 38,219 rows × 333 B = 12.0 MB shipped per Elo fit,
of which ~2.3 MB is data anyone looks at. Multiplied by ~10 fits.

**Fix.** One module fetches the 9-column superset once per process, ordered
`(match_date, id)`, and both models filter `league` / `as_of_date` / `limit` in
Python against the in-memory rows.

```python
# before — models/elo_system.py
matches = (session.query(Match)
           .filter(Match.is_fixture == False, Match.home_goals.isnot(None))
           .order_by(Match.match_date.asc()).all())          # SELECT matches.*  (46 cols)

# after
matches = get_completed_matches(db, league=league, as_of_date=as_of_date)
# → SELECT matches.id, match_date, home_team_id, away_team_id,
#          home_goals, away_goals, home_xg, away_xg, league
```

**Freshness.** The cache cannot go stale mid-process: `daily_update` deliberately
re-fits after a backfill inserts matches, and settlement turns fixtures into
completed matches. Before serving, the cache runs a `(count, max(id),
max(match_date))` aggregate probe — one row, ~50 bytes — and refetches when it
changes. Both scenarios are covered by tests.

**Measured impact.**

| | per fetch | fetches/day | per day |
|---|---:|---:|---:|
| Before (Elo) | 12.0 MB | 10 | 120 MB |
| Before (Poisson) | 1.65 MB | 10 | 17 MB |
| **After (shared)** | **3.8 MB** | **5–7** (once per process that fits) | **19–27 MB** |

**Reduction: ~111 MB/day (≈81% of model-fitting egress, ≈68% of all egress).**

### 3.2 Column projection across the feature pipeline

Every query below was fetching whole entities and immediately projecting them
into dicts in Python — the columns crossed the network and were then discarded.

| File | Query | Columns before → after |
|---|---|---:|
| `features/feature_engineer.py:77` | preload query 1 — fixture metadata | 46 → 9 |
| `features/feature_engineer.py:98` | preload query 2 — odds | 8 → 6 |
| `features/feature_engineer.py:127` | preload query 3 — team history | 46 → 34 |
| `features/feature_engineer.py:550` | `_get_xg_features` | 46 → 5 |
| `features/feature_engineer.py:668` | referee features | 46 → 8 |
| `features/feature_engineer.py:771` | bookmaker odds | 8 → 4 |
| `features/feature_engineer.py:908` | odds movement | 8 → 5 |
| `features/feature_engineer.py:997` | league baselines | 46 → 2 |
| `features/feature_engineer.py:1127` | situational/fatigue | 46 → 7 |
| `features/team_features.py:55` | `get_form_features` | 46 → 20 |
| `features/team_features.py:493` | `get_international_form` | 46 → 20 |
| `features/team_features.py:569` | `get_momentum_indicators` | 46 → 3 |
| `features/team_features.py:652` | `_get_league_standings` | 6 → 2 |
| `features/h2h_features.py:58` | H2H meetings | 46 → 3 |
| `agent/betting_agent.py:777` | `analyze_fixture` odds | 8 → 5 |
| `agent/betting_agent.py:631` | low-coverage fixture scan | 46 → 2 |
| `agent/betting_agent.py:697` | `_unanalyzable_today` | 46 → 4 |
| `agent/betting_agent.py:2293` | `get_stats` full picks table | 26 → 7 |
| `agent/betting_agent.py:2456` | rolling backtest picks | 26 → 6 |
| `agent/betting_agent.py:2525` | `tune_ensemble_weights` picks | 26 → 6 |
| `agent/betting_agent.py:2572` | tuning match lookup | 46 → 4 |
| `agent/betting_agent.py:2906` | ML training match list | 46 → 4 |
| `agent/betting_agent.py:4041` | injury-staleness team lookup | 46 → 2 |
| `reporting/match_briefing.py:555` | briefing odds | 8 → 5 |
| `scrapers/apifootball_scraper.py:1387` | fuzzy team candidates | 5 → 2 |
| `scrapers/flashscore_scraper.py:1689` | Flashscore team resolution | 5 → 2 |

Measured samples: `saved_picks` full scan **333 kB → 76 kB (−77%)**; odds reads
**−43%**; training preload **3.4 MB → 2.4 MB (−29%)**.

**Reduction: ~4–6 MB/day.**

### 3.3 N+1 and round-trip elimination

**`features/injury_features.py:46`** — `session.query(Injury).join(Player)` returned
whole `Injury` rows and then lazy-loaded `i.player` per injury, firing one extra
`SELECT players.*` each. Now a single query selects only `Player.is_key_player,
Player.position`; the squad query selects only `Player.position`.

**`reporting/match_briefing.py:372`** — the WC-fixture dedup ran three queries per
fixture (two `session.get(Team, …)` plus a `COUNT(odds)`). Now two batch queries
for the whole fixture list: `(Team.id, Team.name) WHERE id IN (…)` and a grouped
`COUNT(*) … GROUP BY match_id`.

**`scrapers/flashscore_scraper.py:1689`** — scanned a whole league of full `Team`
rows per scraped name; now scans an `(id, name)` projection and materialises only
the winning row as an ORM entity (the caller mutates it, so it must stay one).

**Reduction: ~2 MB/day + several hundred fewer round-trips.**

---

## 4. Result

| | Per day | Per month |
|---|---:|---:|
| Before | ~150–165 MB | ~4.5–5.0 GB |
| After | ~30–37 MB | **~0.9–1.1 GB** |

**Estimated reduction: ~78%.** Headroom against the 5 GB free-plan quota goes
from roughly 0× to about 4.5×.

### Verification

- Full suite: **271 passed** (255 pre-existing + 16 new).
- `tests/test_match_history_cache.py` — 12 tests: cached rows are byte-identical
  to a reference implementation that replays the original ORM query (unfiltered,
  `as_of_date` as both `date` and `datetime`, `league` filter, `newest_first` +
  `limit`); Elo ratings match a reference chronological pass to 1e-9; Poisson
  strengths are stable across the fit → fit(as_of_date) → fit cycle; the cache
  refreshes on both new inserts and fixtures gaining a result.
- `tests/test_feature_projection_equivalence.py` — 4 tests: the whole
  `create_features` pipeline run against real SQLite, preloaded vs live, must
  agree on all 249 features; plus non-triviality guards so "both paths return
  zeros" cannot pass.
- Every rewritten query was compiled to Postgres SQL and inspected. **This caught
  a real break**: the projected injury join had no determinable left side and
  needed an explicit `select_from(Injury)` — the existing MagicMock-based tests
  could not have seen it.

---

## 5. Remaining opportunities (not implemented)

**a. Cross-process history cache (largest remaining lever, ~19–27 MB/day → ~0.2 MB/day).**
The seven CLI steps share one GitHub Actions filesystem, and `data/models/` is
already persisted via `actions/cache`. Mirroring the 9-column history to a local
Parquet/SQLite file there would make all seven processes share one fetch — and
across days, only fetch the delta. It needs a reliable change signal: an
`updated_at` column on `matches` with a `BEFORE UPDATE` trigger (additive, no
behaviour change). Not done here because it means DDL on the production database,
which is a bigger commitment than the brief authorises.

**b. Push `cap_per_team` into SQL.** `preload_batch` fetches every completed match
for ~295 teams (10,578 rows) and then keeps at most 200 per team in Python. A
`row_number() OVER (PARTITION BY team ORDER BY match_date DESC)` window would
filter server-side. Fiddly because each row belongs to two teams via an `OR`.

**c. `--train` runs unconditionally.** `.github/workflows/daily-picks.yml:192`
runs `--train` every day, but `models.ml_retrain_days` is 3 — `daily_update`
already checks staleness, the standalone `--train` branch does not. Gating it on
`_ml_models_stale()` would drop the training preload (~2.4 MB) on ~2 days in 3.
Behaviour change, so flagging rather than doing.

**d. Missing indexes on `teams` (latency/CPU, not egress).** `pg_stat_user_tables`
shows **58,186 sequential scans on `teams` reading 57.3 M tuples** — the table has
no index beyond its primary key, so every `filter_by(name=…)`,
`filter_by(league=…)` and `filter_by(apifootball_team_id=…)` scans all 1,286 rows.
The brief says not to optimise CPU, so these are recommended, not applied:

```sql
CREATE INDEX CONCURRENTLY ix_teams_name    ON teams (name);
CREATE INDEX CONCURRENTLY ix_teams_league  ON teams (league);
CREATE INDEX CONCURRENTLY ix_teams_afid    ON teams (apifootball_team_id);
```

Adding them to `Team.__table_args__` would auto-create them via
`_migrate_missing_indexes()`. The Supabase performance advisor currently reports
no lints; this is below its threshold.

**e. `matches` seq scans are correct.** 414 seq scans averaging 44 k rows is the
Elo full-history read — a sequential scan is the right plan for "read the whole
table". Now that it happens once per process instead of ~1.4× per fit, it is
also ~10× rarer.

---

## 6. Trade-offs and things to know

1. **Memory for time.** The history cache holds ~38 k lightweight `MatchRow`
   objects (`__slots__`, 9 fields) — roughly 10–15 MB of process RSS. That is a
   deliberate trade against ~111 MB/day of egress, and the brief rules RAM out of
   scope.

2. **Tie ordering is now deterministic, and was not before.** The old queries
   ordered by `match_date` alone; ties came back in whatever order Postgres chose.
   The cache sorts by `(match_date, id)`. For Poisson's `LIMIT 5000` this can pick
   a different row from a same-timestamp pair than a given past run did — but the
   old behaviour was not reproducible either, so this is strictly an improvement.
   It does mean a fit is not guaranteed bit-identical to a specific historical run.

3. **One extra tiny query per fit.** The freshness probe costs ~50 bytes and one
   aggregate scan per `fit()`. Net egress is overwhelmingly positive; if you ever
   want it gone, `match_history.invalidate()` can be called explicitly from the
   scrapers' write paths instead.

4. **Write paths were left alone.** Scraper queries that fetch an entity and then
   mutate it (`_save_api_id`, result updates, odds upserts) still select whole
   rows — they have to, and they are bounded to a day's fixtures.

5. **A pre-existing quirk surfaced, not caused by this work.** League-standings
   and referee features differ between the preload path and the live path, because
   `preload_batch` caches history only for the fixture's own two teams while those
   two feature families need league-wide and referee-wide data. Confirmed by
   running the new equivalence test against the pre-optimisation code, where it
   diverges to the identical values. Worth a separate fix; excluded (and
   documented) in the test so it does not mask real regressions.

---

## 7. Files modified

| File | Change |
|---|---|
| `src/data/match_history.py` | **new** — process-wide 9-column history cache with freshness probe |
| `src/models/elo_system.py` | `fit()` reads the cache; dropped the full-table ORM query |
| `src/models/poisson_model.py` | `fit()` reads the cache; body dedented out of the now-unneeded session block |
| `src/features/feature_engineer.py` | 9 queries column-projected (incl. all 3 preload queries) |
| `src/features/team_features.py` | 4 queries column-projected |
| `src/features/h2h_features.py` | H2H query column-projected |
| `src/features/injury_features.py` | N+1 lazy-load removed; join projected to 2 player columns |
| `src/agent/betting_agent.py` | 9 queries column-projected |
| `src/reporting/match_briefing.py` | odds projected; per-fixture team/count lookups batched |
| `src/scrapers/apifootball_scraper.py` | fuzzy team candidate pool projected to `(id, name)` |
| `src/scrapers/flashscore_scraper.py` | team resolution scans a projection, materialises only the hit |
| `tests/test_match_history_cache.py` | **new** — 12 equivalence + freshness tests |
| `tests/test_feature_projection_equivalence.py` | **new** — 4 end-to-end pipeline equivalence tests |
