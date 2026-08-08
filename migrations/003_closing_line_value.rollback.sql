-- =============================================================================
-- Rollback for migration 003.
--
-- SAFE for the pipeline: SavedPick.closing_odds is nullable everywhere it is
-- read, and get_stats() already handles the "no closing price on any pick" case
-- by reporting CLV as UNAVAILABLE with a reason.
--
-- WHAT YOU LOSE: every closing price captured so far, and with it the only
-- diagnostic that can distinguish a strategy beating the market from one having
-- a lucky month. Export the column before running this if any CLV history has
-- accumulated:
--
--   COPY (SELECT id, closing_odds, closing_odds_captured_at
--         FROM saved_picks WHERE closing_odds IS NOT NULL)
--   TO '/tmp/clv_backup.csv' CSV HEADER;
-- =============================================================================

ALTER TABLE saved_picks DROP COLUMN IF EXISTS closing_odds_captured_at;
ALTER TABLE saved_picks DROP COLUMN IF EXISTS closing_odds;

ANALYZE saved_picks;
