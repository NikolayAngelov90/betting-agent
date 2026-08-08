-- =============================================================================
-- Migration 003 — store the closing line so CLV can actually be computed
--
-- Target: Supabase Postgres 17 (project betting-agent / nhlurscyrlvpjzapmqcr)
-- Rollback: migrations/003_closing_line_value.rollback.sql
--
-- WHY
--
-- The 2026-08-07 predictive audit found that closing line value — the strongest
-- available diagnostic for a betting model — could not be computed for a single
-- pick, past or future:
--
--   * 0 of 124,158 odds rows on picked matches carried a timestamp after the
--     pick day. Odds are captured once, a median of 6.3 hours before kickoff,
--     and never refreshed.
--   * opening_odds is populated on 80.6% of rows but differs from the current
--     value on only 8.4% of them, so even the movement that IS captured is
--     mostly absent.
--   * get_stats() reported "avg_clv", but the quantity was
--     predicted_probability - 1/odds — the model's own claimed edge, not CLV.
--     It read +6.3% while realised flat ROI was -3.6%.
--
-- CLV matters because it separates a strategy that is genuinely beating the
-- market from one that is merely running hot or cold: persistent positive CLV
-- with short-run negative ROI is a working strategy having a bad month, and
-- persistent negative CLV is a losing strategy regardless of its ROI so far.
--
-- WHAT THIS ADDS
--
--   saved_picks.closing_odds              the consensus decimal price for THIS
--                                         selection at (or nearest to) kickoff
--   saved_picks.closing_odds_captured_at  when that snapshot was taken, so a
--                                         late capture can be distinguished
--                                         from a true closing price
--
-- Both are nullable: every existing row stays valid and simply has no CLV.
-- get_stats() reports CLV only over the picks that have a closing price and
-- states the sample size, rather than averaging a partially-populated column.
--
-- EGRESS: two nullable columns on a ~1k-row table. Negligible, and no index is
-- added — CLV is computed in a full scan of settled picks that already runs.
-- =============================================================================

ALTER TABLE saved_picks
    ADD COLUMN IF NOT EXISTS closing_odds DOUBLE PRECISION;

ALTER TABLE saved_picks
    ADD COLUMN IF NOT EXISTS closing_odds_captured_at TIMESTAMP;

COMMENT ON COLUMN saved_picks.closing_odds IS
    'Consensus decimal price for this selection at kickoff. NULL until '
    'scripts/capture_closing_lines.py runs for the fixture. Genuine CLV = '
    'odds / closing_odds - 1.';

COMMENT ON COLUMN saved_picks.closing_odds_captured_at IS
    'When closing_odds was captured. A snapshot taken well before kickoff is '
    'not a closing line; this column makes that visible instead of silently '
    'degrading the CLV series.';

ANALYZE saved_picks;
