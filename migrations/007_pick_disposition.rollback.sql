-- =============================================================================
-- Rollback for migration 007.
--
-- SAFE for the pipeline: every reader treats NULL as "the pick stands", so
-- dropping the column returns the system to Stage 12 semantics.
--
-- WHAT YOU LOSE
--
--   * the distinction between a pick that was taken and one superseded by the
--     review's consolidation branch. Both become indistinguishable, and the
--     superseded ones re-enter the FINAL CLV series as if they had been bet —
--     double-counting the sibling pick they were consolidated into.
--
--   * any Part B void marking, which would make defective fixtures look live
--     again.
--
-- IMPORTANT: rolling this back does NOT restore the old `session.delete()`
-- behaviour, and must not. Deleting a pick destroys its pick_observations, and
-- the MODEL observation is the only record of the frozen model's taken price —
-- unreconstructable, because the odds table keeps one row per
-- (match, bookmaker, market, selection) and overwrites it on every refresh.
--
-- Export first if any row carries a disposition:
--
--   COPY (SELECT id, match_id, pick_date, market, selection, disposition
--         FROM saved_picks WHERE disposition IS NOT NULL)
--   TO '/tmp/stage13_dispositions_backup.csv' CSV HEADER;
-- =============================================================================

ALTER TABLE saved_picks DROP COLUMN IF EXISTS disposition;

ANALYZE saved_picks;
