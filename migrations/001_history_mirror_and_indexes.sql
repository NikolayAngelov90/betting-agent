-- =============================================================================
-- Migration 001 — change tracking for the local history mirror + teams indexes
--
-- Target: Supabase Postgres 17 (project betting-agent / nhlurscyrlvpjzapmqcr)
-- Rollback: migrations/001_history_mirror_and_indexes.rollback.sql
--
-- Every statement below is additive and reversible. Nothing is dropped, no
-- column type changes, no data is rewritten except the one-time backfill of a
-- newly added column.
-- =============================================================================


-- -----------------------------------------------------------------------------
-- 1. matches.updated_at
--
-- WHY THIS IS REQUIRED (not a nice-to-have):
--
-- The local Parquet history mirror has to answer "what changed since my last
-- sync?" on every process start. `matches` currently records only created_at,
-- which detects INSERTs but not UPDATEs — and this pipeline updates existing
-- match rows constantly:
--
--   · settlement writes home_goals/away_goals onto a row inserted days earlier
--     as a fixture (is_fixture flips false, the row ENTERS the completed set)
--   · backfill_match_stats writes xG and shot stats onto rows that are already
--     completed and years old
--   · the Flashscore result scraper rewrites stats in place
--
-- Without a modification timestamp the mirror has exactly two options, both
-- unacceptable: redownload the whole table every sync (which is the 12 MB read
-- we are trying to eliminate), or serve stale rows to the models (wrong xG,
-- wrong results, wrong predictions).
--
-- There is no stock alternative. `xmin` is a 32-bit wrapping transaction id,
-- is not indexable, and cannot be range-queried across a freeze. Postgres has
-- no built-in row-modification timestamp.
--
-- Cost: one 8-byte column on 38k rows (~300 kB) plus a one-time table rewrite
-- (17 MB table — seconds).
-- -----------------------------------------------------------------------------
ALTER TABLE matches
    ADD COLUMN IF NOT EXISTS updated_at timestamp NOT NULL DEFAULT now();

-- Seed from created_at so existing rows carry a meaningful value rather than
-- "everything changed at migration time".
UPDATE matches
   SET updated_at = created_at
 WHERE created_at IS NOT NULL
   AND updated_at > created_at;


-- -----------------------------------------------------------------------------
-- 2. Trigger to maintain it
--
-- WHY A TRIGGER AND NOT SQLAlchemy's onupdate=:
--
-- SQLAlchemy's onupdate only fires for ORM-issued UPDATEs on mapped attributes.
-- This codebase also writes through paths the ORM never sees:
--   · bulk `query(...).update(synchronize_session=False)` statements
--   · raw `text()` DDL/DML in database.py and scripts/
--   · scripts/merge_old_neon_to_supabase.py, migrate_to_supabase.py
--   · any future psql / dashboard edit
-- A row updated through any of those without bumping updated_at would be
-- invisible to the mirror FOREVER — a silent, permanent staleness bug. The
-- trigger is the only place that sees every writer.
--
-- The WHEN clause skips no-op updates: scrapers frequently rewrite a row with
-- identical values, and bumping updated_at for those would make the mirror
-- redownload rows that did not actually change. Comparing old.* to new.* at
-- BEFORE-time is safe — updated_at itself has not been touched yet, so the
-- trigger cannot re-fire on its own write.
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION set_matches_updated_at()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_matches_updated_at ON matches;
CREATE TRIGGER trg_matches_updated_at
    BEFORE UPDATE ON matches
    FOR EACH ROW
    WHEN (OLD.* IS DISTINCT FROM NEW.*)
    EXECUTE FUNCTION set_matches_updated_at();


-- -----------------------------------------------------------------------------
-- 3. Index on matches(updated_at)
--
-- QUERY:   SELECT <9 cols> FROM matches WHERE updated_at >= $1 ORDER BY updated_at
-- RUNS:    once per Python process (7 per CI day), plus any manual run
--
-- Without it, every incremental sync is a full sequential scan of 38k rows /
-- 1046 pages to find the handful that changed — cheap in bytes but pointless
-- CPU, and it grows linearly with the table. With it, a range scan touches
-- only the changed rows.
-- -----------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS ix_matches_updated_at ON matches (updated_at);


-- -----------------------------------------------------------------------------
-- 4. teams indexes — evidence-based, measured on this database
--
-- pg_stat_user_tables: 58,189 sequential scans on `teams` having read
-- 57,342,537 tuples. The table has 1,286 rows in 16 pages and NO index other
-- than its primary key, so every name / league / api-id lookup reads all of it.
-- These lookups run once per scraped team name — hundreds of times per day
-- across apifootball, flashscore, footballdataorg and historical_loader.
--
-- Measured with EXPLAIN (ANALYZE, BUFFERS) on production, before and after:
--
--  (a) WHERE name = $1                  [apifootball _resolve_team step 1,
--                                        flashscore _get_or_create_team,
--                                        footballdataorg, historical_loader]
--      before: Seq Scan   cost=0.00..32.60  buffers=10  689 rows filtered
--      after:  Index Scan cost=0.28..2.50   buffers=3
--      → 13x lower cost, 3x fewer buffers
--
--  (b) WHERE league = $1                [_get_league_standings team list,
--                                        fuzzy candidate pools]
--      before: Seq Scan         cost=0.00..32.60  buffers=16  1260 filtered
--      after:  Bitmap Heap Scan cost=1.58..16.95  buffers=7
--      → 1.9x lower cost, 2.3x fewer buffers
--
--  (c) WHERE apifootball_team_id = $1   [apifootball _resolve_team step 0 —
--                                        the id-first match that stops teams
--                                        being duplicated; runs for EVERY
--                                        ingested fixture, and misses scan the
--                                        whole table because there is no row]
--      before: Seq Scan   cost=0.00..32.60  buffers=10
--      after:  Index Scan cost=0.28..2.50   buffers=2
--      → 13x lower cost
--
-- The planner chose the index in all three cases, so these are not speculative.
-- Write cost is negligible: `teams` gains ~1,300 rows/year.
--
-- NOTE: run these three CONCURRENTLY if you apply them by hand outside a
-- transaction. They are written plain here because the table is 16 pages and
-- the lock is held for milliseconds, and because CREATE INDEX CONCURRENTLY
-- cannot run inside the transaction a migration runner wraps around this file.
-- -----------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS ix_teams_name ON teams (name);
CREATE INDEX IF NOT EXISTS ix_teams_league ON teams (league);
CREATE INDEX IF NOT EXISTS ix_teams_apifootball_team_id ON teams (apifootball_team_id);

ANALYZE teams;
ANALYZE matches;
