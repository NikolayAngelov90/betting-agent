-- Rollback for 008. Drops both columns and the partial index.
-- Destructive: any exclusion marking is lost. The marking script is
-- idempotent, so re-running it after a re-apply restores the same state.
DROP INDEX IF EXISTS ix_matches_training_exclusion;
ALTER TABLE matches DROP COLUMN IF EXISTS training_exclusion_reason;
ALTER TABLE saved_picks DROP COLUMN IF EXISTS evidence_status;
