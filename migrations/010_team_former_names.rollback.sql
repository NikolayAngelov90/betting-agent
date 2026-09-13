-- Rollback for 010. Drops the table and its index.
--
-- SAFE: nothing else references it. `resolve_team()` treats a missing table as
-- "no former names known" and falls through to its remaining steps, so the
-- pipeline degrades to pre-Stage-23 behaviour rather than failing — the
-- resurrections return, which is the observable cost of the rollback.
DROP INDEX IF EXISTS ix_team_former_names_team;
DROP TABLE IF EXISTS team_former_names;
