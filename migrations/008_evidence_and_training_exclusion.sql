-- Stage 13 (s5.3) — two orthogonal exclusion facts.
--
-- Additive only. No backfill, no UPDATE, no DELETE: every existing row keeps
-- NULL, which means "included" for both columns. The rows that need marking are
-- marked by an explicit, reviewable script, not by this migration.
--
-- ---------------------------------------------------------------------------
-- 1. saved_picks.evidence_status
--
-- `disposition` answers "why did this leave the live record" and is read by
-- _live_only(). It cannot also answer "is this valid evidence about the model",
-- because a pick can be both superseded AND built on corrupt inputs, and under
-- a single field the later write would erase the earlier reason.
--
-- NULL          the observation is valid evidence
-- 'void_corrupt_features'
--               the fixture was real and the wager genuine, but the model's
--               inputs described a different club — so it is not evidence about
--               this model. Read by _valid_evidence() at the five learning and
--               measurement sites, and by series_clv in the paper report.
--
-- Write-once, enforced by a @validates on the ORM model: exclusion can be
-- applied later but never undone, so a wrong stamp is unrecoverable.
ALTER TABLE saved_picks
    ADD COLUMN IF NOT EXISTS evidence_status VARCHAR(32);

-- ---------------------------------------------------------------------------
-- 2. matches.training_exclusion_reason
--
-- A reason string rather than a boolean, so a future reader knows WHY a match
-- was detached without consulting a commit message. NULL = included in fitting.
--
-- 'corrupt_team_identity'
--               one participant's row belongs to a different club. Measured:
--               29 matches across four rows (Telstar/Maccabi Tel Aviv,
--               SK Rapid/Rapid Bucuresti, St. Pauli/Pau FC, Levski Sofia).
--
-- Honoured at match_history._base_filter() and at the four direct queries that
-- do not route through it — feature_engineer.py's own historical query
-- foremost, because excluding these from training while the feature pipeline
-- still reads them would leave the contamination exactly where it does harm.
ALTER TABLE matches
    ADD COLUMN IF NOT EXISTS training_exclusion_reason VARCHAR(48);

CREATE INDEX IF NOT EXISTS ix_matches_training_exclusion
    ON matches (training_exclusion_reason)
    WHERE training_exclusion_reason IS NOT NULL;
