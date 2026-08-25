-- Stage 18, Part C — stop discarding the data.
--
-- Additive only. No backfill, no UPDATE, no DELETE. Every existing row is
-- untouched and every new column starts NULL.
--
-- COHORT NEUTRALITY. Nothing here changes what any model reads. `odds` keeps
-- its unique constraint and its exact shape, so every current-price consumer
-- reads precisely what it read yesterday and Stage 3's column-projection egress
-- work is preserved BY CONSTRUCTION rather than by care. History accumulates in
-- a separate table that no model touches.
--
-- ---------------------------------------------------------------------------
-- 1. odds_snapshots — the price PATH, not the price
--
-- `odds` is unique on (match_id, bookmaker, market_type, selection) and is
-- overwritten on every refresh, so it holds exactly two observations per key —
-- `opening_odds` and the current value — and never a third. Stage 17 could not
-- test momentum for that reason: predicting movement t1->t2 from movement
-- t0->t1 needs three points.
--
-- DELIBERATELY NO UNIQUE CONSTRAINT on the natural key. That absence IS the
-- feature; a second row for the same key at a later `observed_at` is the whole
-- point. The index is (key, observed_at) so a trajectory reads as one range
-- scan.
--
-- Sized in Part A from measured write volume: ~92,000 rows/month at 211 bytes
-- all-in = ~19.4 MB/month, reaching ~45% of the 500 MB free tier at six months
-- and ~68% at twelve. The 400-day prune first bites 2027-04-04.
CREATE TABLE IF NOT EXISTS odds_snapshots (
    id           BIGSERIAL PRIMARY KEY,
    match_id     INTEGER NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    bookmaker    VARCHAR(100) NOT NULL,
    market_type  VARCHAR(50)  NOT NULL,
    selection    VARCHAR(50)  NOT NULL,
    odds_value   DOUBLE PRECISION NOT NULL,
    observed_at  TIMESTAMP NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_odds_snapshots_key_time
    ON odds_snapshots (match_id, bookmaker, market_type, selection, observed_at);

CREATE INDEX IF NOT EXISTS ix_odds_snapshots_observed_at
    ON odds_snapshots (observed_at);

-- ---------------------------------------------------------------------------
-- 2. odds.first_seen_at — when THIS SYSTEM first saw the price
--
-- `opening_odds` is frozen at first sight and carries no timestamp, so "how
-- long before kickoff was this price taken" has been unanswerable. That is
-- Stage 17's H4.
--
-- `matches.created_at` cannot proxy for it: MEASURED 2026-08-25, 53.5% of match
-- rows were created AFTER their own kickoff, mean +14 days, because they are
-- backfill stamps rather than first-sight stamps.
--
-- Existing rows stay NULL and are NOT backfilled. A guessed first-seen time is
-- worse than an absent one — it would look like evidence.
ALTER TABLE odds
    ADD COLUMN IF NOT EXISTS first_seen_at TIMESTAMP;

-- ---------------------------------------------------------------------------
-- 3. injury_observations — what was known, and WHEN it was known
--
-- `injuries` holds current status only and is overwritten. MEASURED 2026-08-25:
-- 34 rows in the entire database, 10 teams, all dated 2026-08-17/18, while the
-- CI audit shows runs fetching 128-198 injuries in March-May. The history was
-- fetched and discarded daily.
--
-- `observed_at` is the point of the table. An injury that moves a line moves it
-- when the NEWS ARRIVES; a current-status snapshot cannot distinguish a
-- two-week-old absence from this morning's announcement.
--
-- COHORT-NEUTRAL: Stage 14 established injuries reach only the Claude review
-- prompt and never the model. Retaining history changes no prediction.
CREATE TABLE IF NOT EXISTS injury_observations (
    id           BIGSERIAL PRIMARY KEY,
    player_id    INTEGER,
    team_id      INTEGER NOT NULL,
    injury_type  VARCHAR(100),
    status       VARCHAR(50),
    start_date   DATE,
    source       VARCHAR(50),
    observed_at  TIMESTAMP NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_injury_obs_team_time
    ON injury_observations (team_id, observed_at);

CREATE INDEX IF NOT EXISTS ix_injury_obs_observed_at
    ON injury_observations (observed_at);
