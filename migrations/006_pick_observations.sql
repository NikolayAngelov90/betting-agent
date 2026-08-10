-- =============================================================================
-- Migration 006 — dual CLV attribution storage (`pick_observations`)
--
-- Target: Supabase Postgres 17 (project betting-agent / nhlurscyrlvpjzapmqcr)
-- Rollback: migrations/006_pick_observations.rollback.sql
-- Depends on: 005 (experiment metadata)
--
-- WHY
--
-- The experiment asks two separate questions:
--
--   MODEL — did the FROZEN Stage 5 selection's price move the right way?
--   FINAL — did the selection actually taken (after the Claude KEEP/CHANGE
--           review) move the right way?
--
-- Stage 9 proved `saved_picks` cannot hold both answers, for two independent
-- reasons:
--
-- 1. THE MODEL'S TAKEN PRICE IS DESTROYED.
--    On a CHANGE, `_apply_decision` assigns `primary.odds = float(new.odds)`.
--    model_market / model_selection / model_probability are deliberately
--    preserved — but not the price. CLV is `taken / closing - 1`, and a
--    probability is not a price. Nor can the price be recovered afterwards:
--    ix_odds_match_bookie_market is UNIQUE on
--    (match_id, bookmaker, market_type, selection), so the odds table keeps
--    exactly one row per book and every refresh overwrites it. There is no
--    price history to look back into.
--
--    Inverting the stored EV (`odds = (ev + 1) / p`) looks like a way out and
--    is not: `_market_ev` scales Draw No Bet by P(decisive), which is not
--    stored, so a DNB pick would silently yield a wrong price. DNB is an
--    enabled market with zero picks so far — the trap is armed but unfired.
--
-- 2. THERE IS ONLY ONE SET OF CLOSING COLUMNS.
--    A changed pick needs TWO closing observations in TWO different markets.
--    `saved_picks` has one closing_odds / closing_captured_at /
--    closing_capture_status / closing_bookmaker_count /
--    closing_fair_probability, and no free column can carry a second without
--    abusing its meaning.
--
-- Measured on production before this migration: of 1,070 picks, 71 carry a
-- model snapshot — 49 identical (one observation, two attributions) and 22
-- genuine changes, none of which can produce a model CLV. `both_measurable`
-- was 0. That zero is what this table exists to fix, prospectively.
--
-- SHAPE
--
-- One row per (pick, attribution). `taken_odds` is written at pick-save time,
-- BEFORE the review can overwrite anything — that is the only moment the
-- model's price exists. UNIQUE (pick_id, attribution) keeps the two
-- observations distinguishable and prevents either from being duplicated.
--
-- On an unchanged pick both rows carry the same market, selection and price,
-- so closing capture resolves ONE close and writes it to both: one API
-- observation, two attributions, and still one fixture in the statistics.
--
-- SAFETY
--
-- Purely additive. No column on `saved_picks` is added, altered, renamed or
-- dropped; no existing row is read or written by this migration. The table is
-- created empty and is NOT backfilled — historical rows keep their truthful
-- `no_model_snapshot` / `model_taken_price_not_recorded` states, because
-- reconstructing their prices is exactly the unsound operation described above.
-- IF NOT EXISTS makes the migration idempotent.
--
-- ON DELETE CASCADE: an observation is meaningless without its pick, and the
-- pipeline never deletes picks except in the review's consolidation branch,
-- where the pick genuinely ceases to exist.
--
-- EGRESS: two narrow rows per pick, ~8 picks/day. The pending index is partial
-- so it only covers rows still awaiting capture.
-- =============================================================================

CREATE TABLE IF NOT EXISTS pick_observations (
    id                   SERIAL PRIMARY KEY,
    pick_id              INTEGER NOT NULL
                         REFERENCES saved_picks(id)
                         ON DELETE CASCADE,

    -- 'model' = the frozen Stage 5 selection, 'final' = the persisted one.
    attribution          VARCHAR(8) NOT NULL,
    market               VARCHAR(50) NOT NULL,
    selection            VARCHAR(100) NOT NULL,

    -- The price at the moment of the pick. Never reconstructed.
    taken_odds           DOUBLE PRECISION NOT NULL,
    taken_at             TIMESTAMP NOT NULL,

    closing_odds         DOUBLE PRECISION,
    closing_captured_at  TIMESTAMP,
    closing_status       VARCHAR(16) NOT NULL DEFAULT 'pending',
    closing_book_count   INTEGER,
    closing_fair_prob    DOUBLE PRECISION,

    CONSTRAINT uq_pick_observations_pick_attribution
        UNIQUE (pick_id, attribution),

    -- Enforced in the database, not only in application code: a third
    -- attribution value would silently create a series nothing reports on.
    CONSTRAINT ck_pick_observations_attribution
        CHECK (attribution IN ('model', 'final'))
);

-- The capture job filters on exactly this predicate.
CREATE INDEX IF NOT EXISTS ix_pick_observations_pending
    ON pick_observations (closing_status)
    WHERE closing_status = 'pending';

-- The report joins observations back to their picks.
CREATE INDEX IF NOT EXISTS ix_pick_observations_pick_id
    ON pick_observations (pick_id);

ANALYZE pick_observations;
