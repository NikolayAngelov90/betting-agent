-- =============================================================================
-- Rollback for migration 002.
--
-- SAFE, but note what you lose:
--
--   * Dropping ix_saved_picks_dedup returns pick deduplication to the
--     application-level read-then-write check. That is correct only while runs
--     are serialised — do not roll this back and then run workers in parallel.
--   * Dropping api_budget makes ApiFootballScraper fall back to its in-process
--     counter (the code probes for the table and degrades). Quota accounting
--     becomes per-process again.
--
-- The code tolerates both: saved-pick inserts fall back to plain ORM adds, and
-- the budget falls back to instance state.
-- =============================================================================

DROP INDEX IF EXISTS ix_saved_picks_dedup;

DROP INDEX IF EXISTS ix_api_budget_day;
DROP TABLE IF EXISTS api_budget;

DROP INDEX IF EXISTS ix_matches_match_date;

ANALYZE saved_picks;
ANALYZE matches;
