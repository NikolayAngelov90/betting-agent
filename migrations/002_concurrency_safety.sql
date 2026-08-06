-- =============================================================================
-- Migration 002 — make concurrent writers safe
--
-- Target: Supabase Postgres 17 (project betting-agent / nhlurscyrlvpjzapmqcr)
-- Rollback: migrations/002_concurrency_safety.rollback.sql
--
-- Prerequisite for running more than one pipeline worker. Today the workflow
-- serialises runs (`concurrency: cancel-in-progress: false`), which is the only
-- reason the races below have not fired. Sharding the pipeline without this
-- migration would actively corrupt data.
-- =============================================================================


-- -----------------------------------------------------------------------------
-- 1. saved_picks dedup constraint
--
-- THE RACE: save_picks() dedups by reading first --
--
--     existing = session.query(SavedPick).filter(
--         SavedPick.match_id == ..., SavedPick.selection == ...,
--         SavedPick.pick_date == ...).first()
--     if existing: continue
--     session.add(SavedPick(...))
--
-- Two workers both read "no duplicate" and both insert. The table had no unique
-- index, so nothing stopped them.
--
-- WHY IT MATTERS MORE THAN IT LOOKS: a duplicated pick is not a cosmetic double
-- entry. Every downstream statistic is computed by counting saved_picks rows —
-- win rate, ROI, Brier score, the Bayesian per-league weight learner, and the
-- drawdown circuit breaker that scales real stake sizes. One duplicate skews
-- all of them in the same direction.
--
-- Verified zero existing violations before creating this index.
--
-- NOTE: written without CONCURRENTLY because a migration runner wraps this file
-- in a transaction and CREATE INDEX CONCURRENTLY cannot run inside one. The
-- table is ~1k rows, so the exclusive lock is held for milliseconds. On a large
-- table, run it separately with CONCURRENTLY instead.
-- -----------------------------------------------------------------------------
CREATE UNIQUE INDEX IF NOT EXISTS ix_saved_picks_dedup
    ON saved_picks (match_id, selection, pick_date);


-- -----------------------------------------------------------------------------
-- 2. Cross-process API request budget
--
-- THE PROBLEM: ApiFootballScraper tracks spend in `self._requests_today`, an
-- instance attribute initialised to 0. It is never persisted. The daily job runs
-- seven separate Python processes, each constructing its own scraper, so the
-- "100 requests/day" cap is enforced per process — the real ceiling is already
-- ~700/day. With any parallelism it becomes N_workers x 100, and every
-- budget-derived number in the scraper (BUDGET_XG, BUDGET_RESERVE, the odds
-- semaphore capacity) is computed from a counter that does not mean anything.
--
-- This table makes the counter global. Callers claim quota with a single
-- conditional UPDATE that is atomic under concurrency:
--
--     UPDATE api_budget SET used = used + :n
--      WHERE day = :day AND provider = :p AND used + :n <= limit_
--  RETURNING used;
--
-- No row returned => the claim was refused because the budget is exhausted.
-- The row lock Postgres takes for the UPDATE serialises concurrent claimants,
-- so two workers can never both spend the last request.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS api_budget (
    day        date         NOT NULL,
    provider   varchar(50)  NOT NULL,
    used       integer      NOT NULL DEFAULT 0,
    limit_     integer      NOT NULL,
    updated_at timestamp    NOT NULL DEFAULT now(),
    PRIMARY KEY (day, provider)
);

-- Retention: one row per provider per day is tiny, but nothing prunes it, so
-- keep it honest with an index for the cleanup query.
CREATE INDEX IF NOT EXISTS ix_api_budget_day ON api_budget (day);


-- -----------------------------------------------------------------------------
-- 3. Index supporting the batched odds prune
--
-- prune_old_odds ran one unbounded DELETE whose plan was a sequential scan of
-- the whole odds table:
--
--     Delete on odds (actual time=1770.371..1770.375 rows=0)
--       -> Seq Scan on odds (actual rows=214683)
--     Execution Time: 1773.721 ms
--
-- It scanned 214,683 of 317,657 rows to delete ZERO, every day. The rewrite
-- selects victim ids in bounded batches driven off match_date, which needs the
-- match side to be cheap to filter. ix_match_fixture_date already covers
-- (is_fixture, match_date); this adds the date-only entry point the prune uses.
-- -----------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS ix_matches_match_date ON matches (match_date);

ANALYZE saved_picks;
ANALYZE matches;
