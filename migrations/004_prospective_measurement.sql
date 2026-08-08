-- =============================================================================
-- Migration 004 — record what the model was disagreeing WITH, and support
--                 paper trading
--
-- Target: Supabase Postgres 17 (project betting-agent / nhlurscyrlvpjzapmqcr)
-- Rollback: migrations/004_prospective_measurement.rollback.sql
-- Depends on: 003 (closing line value)
--
-- WHY
--
-- Stage 4 could not answer "was the model's claimed edge real?" from stored data
-- alone. `saved_picks.predicted_probability` recorded the model's number, but
-- the de-vigged market consensus it was being compared against was never stored
-- — it had to be re-derived from the odds table, and those rows had since been
-- overwritten by later refreshes (and, for 2,548 matches, corrupted outright by
-- the Home/Away -> 1X2 mapping bug).
--
-- Storing the market's opinion at prediction time makes every future evaluation
-- a lookup instead of a reconstruction:
--
--   market_probability  de-vigged consensus probability for THIS selection,
--                       from the same cross-book median the blend uses
--   market_books        how many plausible books backed that consensus, so a
--                       one-book estimate is distinguishable from a twelve-book
--                       one rather than both reading as "the market"
--
-- PAPER TRADING
--
--   is_paper            the pick was recorded for measurement, not offered as a
--                       recommendation. Stage 3 found that an ACTIVE gate leaves
--                       no holdout cohort — six of the eleven gates tested were
--                       untestable for exactly that reason. Recording suppressed
--                       candidates as paper picks is what makes those gates
--                       measurable prospectively.
--
-- All three are nullable with a safe default, so existing rows stay valid and
-- `_migrate_missing_columns` can add them without a backfill.
--
-- EGRESS: three narrow columns on a ~1k-row table, no index. Negligible.
-- =============================================================================

ALTER TABLE saved_picks
    ADD COLUMN IF NOT EXISTS market_probability DOUBLE PRECISION;

ALTER TABLE saved_picks
    ADD COLUMN IF NOT EXISTS market_books INTEGER;

ALTER TABLE saved_picks
    ADD COLUMN IF NOT EXISTS is_paper BOOLEAN DEFAULT FALSE;

COMMENT ON COLUMN saved_picks.market_probability IS
    'De-vigged cross-book consensus probability for this selection at prediction '
    'time. The quantity predicted_probability should be judged against.';

COMMENT ON COLUMN saved_picks.market_books IS
    'Number of bookmakers whose market passed the overround plausibility gate '
    'and contributed to market_probability.';

COMMENT ON COLUMN saved_picks.is_paper IS
    'TRUE when the pick was recorded for measurement only (paper trading), not '
    'offered as a live recommendation. Excluded from ROI reporting by default.';

ANALYZE saved_picks;
