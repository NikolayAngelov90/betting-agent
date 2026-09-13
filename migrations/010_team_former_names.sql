-- Stage 23 — an operation that REMOVES an identifier must RECORD it.
--
-- THE LESSON THIS TABLE EXISTS TO ENFORCE, stated first because it generalises
-- past team names: s5.10 merged 129 team rows on 2026-09-10 and ran
-- `DELETE FROM teams` without keeping the names it removed. Within a day 26 of
-- them were back, and by 09-13 the count was 44 — every one an EXACT match to a
-- name the merge had deleted.
--
--     THE MERGE REMOVED THE ROW AND LEFT THE REASON THE ROW EXISTED.
--
-- The scrapers go on emitting "Lens" because Flashscore goes on calling it
-- "Lens". Nothing in the database remembered that "Lens" had been resolved to
-- "Racing Club de Lens", so the next scrape created it again. Measured: 100% of
-- 44 resurrections are exact matches to merged-away names — no diacritic,
-- punctuation or token differences at all.
--
-- Additive only. No backfill of existing columns, no UPDATE, no DELETE. The
-- table starts empty and is populated by the merge that removes the names.
--
-- COHORT: this table alone changes nothing. `resolve_team()` consulting it is
-- the selection-affecting part and carries the bump.

CREATE TABLE IF NOT EXISTS team_former_names (
    -- The removed name, EXACTLY as the source wrote it. Not normalised: the
    -- lookup that uses this is an exact string match, deliberately. A fuzzy
    -- comparison here would reintroduce the ratio-and-cross-product hazard that
    -- `team_names_similar`'s union fix was measured against and refused —
    -- 38 new matches, roughly half absurd (`ac milan` == `manchester utd`).
    name           TEXT        NOT NULL,

    -- The surviving row the name now resolves to.
    team_id        INTEGER     NOT NULL REFERENCES teams(id) ON DELETE CASCADE,

    -- Where the mapping came from, so a later reader can tell a merge-derived
    -- entry from a hand-added one without guessing. Hand-added entries are
    -- expected to be rare and should be argued for in review.
    source         TEXT        NOT NULL DEFAULT 'merge',

    -- Which revision performed the removal. s5.10's entries are backfilled from
    -- its apply log; every later merge writes its own at merge time.
    revision       TEXT,

    recorded_at    TIMESTAMP   NOT NULL DEFAULT (now() AT TIME ZONE 'utc'),

    -- A name resolves to exactly ONE club or it is not usable as an identity.
    -- The PK enforces that: a second merge trying to claim a name already
    -- claimed by a different club fails loudly instead of silently rebinding
    -- it, which is the same refusal s5.10 applied to the two `Sporting Clube`
    -- rows rather than guessing between `Sporting CP` and `Braga`.
    PRIMARY KEY (name)
);

-- The lookup is by name (the PK, already indexed). This index serves the
-- reverse question — "what names did this club used to have?" — which is what
-- an audit of a merge needs and what a future un-merge would need.
CREATE INDEX IF NOT EXISTS ix_team_former_names_team
    ON team_former_names (team_id);
