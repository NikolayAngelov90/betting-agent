-- =============================================================================
-- Rollback for migration 001.
--
-- Order matters: the trigger references the function, so it goes first.
--
-- SAFE TO RUN AT ANY TIME. Dropping matches.updated_at disables the local
-- history mirror's incremental sync; src/data/history_mirror.py probes for the
-- column and, when it is absent, falls back to the in-process cache with the
-- (count, max_id, max_date) freshness probe. Predictions are unaffected — only
-- egress goes back up to roughly 3.8 MB per process.
--
-- Dropping the teams indexes only restores the previous sequential scans.
-- =============================================================================

DROP TRIGGER IF EXISTS trg_matches_updated_at ON matches;
DROP FUNCTION IF EXISTS set_matches_updated_at();

DROP INDEX IF EXISTS ix_matches_updated_at;
ALTER TABLE matches DROP COLUMN IF EXISTS updated_at;

DROP INDEX IF EXISTS ix_teams_name;
DROP INDEX IF EXISTS ix_teams_league;
DROP INDEX IF EXISTS ix_teams_apifootball_team_id;

ANALYZE teams;
ANALYZE matches;
