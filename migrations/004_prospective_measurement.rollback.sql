-- =============================================================================
-- Rollback for migration 004.
--
-- SAFE: all three columns are nullable and every reader uses getattr/COALESCE
-- with a default, so the pipeline degrades to its pre-Stage-4 behaviour —
-- picks are still saved, they simply carry no record of the market probability
-- they were compared against.
--
-- WHAT YOU LOSE: the ability to evaluate the model's claimed edge without
-- re-deriving the market from odds rows that may since have been overwritten.
-- That reconstruction is what Stage 4 had to do, and it is why 194 of 1,018
-- settled picks could not be placed in the clean dataset at all.
--
-- Export before rolling back if any prospective data has accumulated:
--
--   COPY (SELECT id, market_probability, market_books, is_paper
--         FROM saved_picks WHERE market_probability IS NOT NULL)
--   TO '/tmp/prospective_backup.csv' CSV HEADER;
-- =============================================================================

ALTER TABLE saved_picks DROP COLUMN IF EXISTS is_paper;
ALTER TABLE saved_picks DROP COLUMN IF EXISTS market_books;
ALTER TABLE saved_picks DROP COLUMN IF EXISTS market_probability;

ANALYZE saved_picks;
