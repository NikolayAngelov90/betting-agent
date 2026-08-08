-- =============================================================================
-- Migration 005 — Stage 5 prospective-experiment metadata
--
-- Target: Supabase Postgres 17 (project betting-agent / nhlurscyrlvpjzapmqcr)
-- Rollback: migrations/005_stage5_experiment_metadata.rollback.sql
-- Depends on: 003 (closing line), 004 (prospective measurement)
--
-- WHY
--
-- Stage 5 turns the model into an experimental subject observed prospectively.
-- That requires three things the schema could not express:
--
-- 1. WHICH MODEL produced a prediction.
--    Stages 1-4 changed the blend weight, the Poisson half-life, rho, the
--    de-vigging rule and six betting gates. Every one of those changes silently
--    altered what `predicted_probability` means, and nothing in the row recorded
--    it — so a settled pick from March and one from August were pooled as if
--    they came from the same system. `model_version` stamps the configuration
--    that generated each prediction so future cohorts can be separated.
--
-- 2. WHETHER A CLOSING PRICE IS USABLE.
--    A NULL closing_odds is ambiguous: not captured yet, captured and rejected,
--    or captured after kickoff? Averaging over that ambiguity is how a CLV
--    series quietly becomes unrepresentative. `closing_capture_status` records
--    the outcome explicitly, and never invents a price.
--
-- 3. WHAT THE CLOSING PRICE ACTUALLY IS.
--    A bare number cannot be re-validated. Storing the bookmaker count and the
--    de-vigged closing probability alongside it keeps the raw evidence, so CLV
--    can be recomputed later under a different definition (Phase 9 leaves the
--    primary normalisation open on purpose).
--
-- Plus `pre_claude_ev`, completing the pre/post-review snapshot that
-- model_market / model_selection / model_probability already provide.
--
-- SAFETY
--
-- Additive only. Every column is nullable (closing_capture_status carries a
-- DEFAULT, which PostgreSQL applies without rewriting existing rows). No data is
-- modified, no constraint is added to existing columns, nothing is dropped.
-- IF NOT EXISTS on every statement makes the whole migration idempotent.
--
-- EGRESS: six narrow columns on a ~1k-row table. One partial index on
-- closing_capture_status, which the capture job and the coverage report both
-- filter by; it indexes only the rows still awaiting capture, so it stays tiny.
-- =============================================================================

ALTER TABLE saved_picks
    ADD COLUMN IF NOT EXISTS model_version VARCHAR(64);

ALTER TABLE saved_picks
    ADD COLUMN IF NOT EXISTS closing_capture_status VARCHAR(16) DEFAULT 'pending';

ALTER TABLE saved_picks
    ADD COLUMN IF NOT EXISTS closing_bookmaker_count INTEGER;

ALTER TABLE saved_picks
    ADD COLUMN IF NOT EXISTS closing_fair_probability DOUBLE PRECISION;

ALTER TABLE saved_picks
    ADD COLUMN IF NOT EXISTS pre_claude_ev DOUBLE PRECISION;

-- Partial index: the capture job asks "which picks still need a closing price?"
-- on every run, and the coverage report asks the complement. Indexing only the
-- unresolved rows keeps this at a few hundred entries rather than the whole
-- table, and it shrinks as captures complete.
CREATE INDEX IF NOT EXISTS ix_saved_picks_closing_pending
    ON saved_picks (closing_capture_status)
    WHERE closing_capture_status = 'pending';

COMMENT ON COLUMN saved_picks.model_version IS
    'Identifier of the exact model configuration that produced this prediction, e.g. stage5_baseline_20260807. Lets cohorts from different configurations be separated instead of silently pooled.';

COMMENT ON COLUMN saved_picks.closing_capture_status IS
    'pending | captured | missing | late | invalid. Explicit so a NULL closing_odds is never ambiguous. Only rows with status=captured may contribute to CLV.';

COMMENT ON COLUMN saved_picks.closing_bookmaker_count IS
    'How many bookmaker markets passed validation and contributed to closing_odds.';

COMMENT ON COLUMN saved_picks.closing_fair_probability IS
    'De-vigged consensus probability at closing time. Kept alongside the raw price so margin-free CLV can be computed without re-querying.';

COMMENT ON COLUMN saved_picks.pre_claude_ev IS
    'Expected value of the model''s original selection, snapshotted before the Claude review could change it.';

ANALYZE saved_picks;
