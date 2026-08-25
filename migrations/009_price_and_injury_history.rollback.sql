-- Rollback for 009. Destructive: all accumulated price and injury history is
-- lost and CANNOT be reconstructed, because the source rows were overwritten in
-- `odds` and `injuries` as they always were. Re-applying gives an empty table
-- that starts accumulating again from that moment.
--
-- `odds.first_seen_at` is dropped with the same consequence: it was never
-- backfilled, so the only values it held were the ones observed since 009 was
-- applied.
DROP INDEX IF EXISTS ix_injury_obs_observed_at;
DROP INDEX IF EXISTS ix_injury_obs_team_time;
DROP TABLE IF EXISTS injury_observations;

ALTER TABLE odds DROP COLUMN IF EXISTS first_seen_at;

DROP INDEX IF EXISTS ix_odds_snapshots_observed_at;
DROP INDEX IF EXISTS ix_odds_snapshots_key_time;
DROP TABLE IF EXISTS odds_snapshots;
