-- =============================================================================
-- Rollback for migration 006.
--
-- SAFE for the pipeline in the sense that nothing crashes: `saved_picks` is
-- untouched by 006, so settlement, the Telegram picks message, the odds
-- refresh and the FINAL-series closing capture all keep working exactly as
-- they did before. The observation writes degrade to no-ops (the ORM model is
-- guarded and logs at debug level when the table is absent).
--
-- WHAT YOU LOSE
--
--   * The MODEL attribution series, permanently and unrecoverably for every
--     pick where the Claude review changed the selection. `saved_picks.odds`
--     holds the FINAL price; the model's taken price lives ONLY here. Once
--     these rows are gone it cannot be reconstructed — the odds table keeps
--     one row per (match, bookmaker, market, selection) and overwrites it on
--     every refresh, and inverting the stored EV is unsound for Draw No Bet.
--
--   * Every closing observation attributed to MODEL, and the paired
--     model-vs-final comparison built on it.
--
-- The FINAL series survives: its close is still written to
-- saved_picks.closing_odds by the existing capture path.
--
-- EXPORT FIRST if any prospective observation has accumulated. This is not
-- optional advice — the model prices in this table are the experiment:
--
--   COPY (SELECT * FROM pick_observations)
--   TO '/tmp/stage10_pick_observations_backup.csv' CSV HEADER;
-- =============================================================================

DROP INDEX IF EXISTS ix_pick_observations_pick_id;
DROP INDEX IF EXISTS ix_pick_observations_pending;

DROP TABLE IF EXISTS pick_observations;
