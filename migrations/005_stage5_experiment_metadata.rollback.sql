-- =============================================================================
-- Rollback for migration 005.
--
-- SAFE for the pipeline: every column is nullable and every reader uses
-- getattr/COALESCE with a default. The system degrades to its Stage 4 behaviour
-- — predictions are still saved, settled and measured; they simply carry no
-- record of which model configuration produced them and no explicit
-- closing-capture status.
--
-- WHAT YOU LOSE
--
--   * model_version — cohorts from different configurations become
--     indistinguishable again, which is the exact ambiguity that made Stages
--     1-4 results hard to pool.
--   * closing_capture_status — a NULL closing_odds returns to being ambiguous
--     (never captured / rejected / captured late), so CLV coverage can no
--     longer be stated honestly.
--
-- Export before rolling back if any prospective data has accumulated:
--
--   COPY (SELECT id, model_version, closing_capture_status,
--                closing_bookmaker_count, closing_fair_probability, pre_claude_ev
--         FROM saved_picks
--         WHERE model_version IS NOT NULL
--            OR closing_capture_status <> 'pending')
--   TO '/tmp/stage5_backup.csv' CSV HEADER;
-- =============================================================================

DROP INDEX IF EXISTS ix_saved_picks_closing_pending;

ALTER TABLE saved_picks DROP COLUMN IF EXISTS pre_claude_ev;
ALTER TABLE saved_picks DROP COLUMN IF EXISTS closing_fair_probability;
ALTER TABLE saved_picks DROP COLUMN IF EXISTS closing_bookmaker_count;
ALTER TABLE saved_picks DROP COLUMN IF EXISTS closing_capture_status;
ALTER TABLE saved_picks DROP COLUMN IF EXISTS model_version;

ANALYZE saved_picks;
