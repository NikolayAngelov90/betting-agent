"""Database models for Football Betting Agent."""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime,
    CheckConstraint, ForeignKey, Date, Text, JSON, Index, UniqueConstraint
)
from sqlalchemy.orm import declarative_base, relationship

from src.utils.logger import utcnow

Base = declarative_base()


class Team(Base):
    """Team model."""

    __tablename__ = 'teams'

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    country = Column(String(50))
    league = Column(String(100))
    apifootball_team_id = Column(Integer)  # API-Football team ID for history backfill
    created_at = Column(DateTime, default=utcnow)

    # Team lookup happens once per scraped name across four scrapers, and the
    # table had no index beyond its primary key — pg_stat recorded 58k sequential
    # scans having read 57M tuples. EXPLAIN on production picked an index scan for
    # all three columns (see migrations/001_history_mirror_and_indexes.sql).
    __table_args__ = (
        Index("ix_teams_name", "name"),
        Index("ix_teams_league", "league"),
        Index("ix_teams_apifootball_team_id", "apifootball_team_id"),
    )

    # Relationships
    home_matches = relationship("Match", foreign_keys="Match.home_team_id", back_populates="home_team")
    away_matches = relationship("Match", foreign_keys="Match.away_team_id", back_populates="away_team")
    players = relationship("Player", back_populates="team")
    injuries = relationship("Injury", back_populates="team")

    def __repr__(self):
        return f"<Team(name='{self.name}', league='{self.league}')>"


class Match(Base):
    """Match model."""

    __tablename__ = 'matches'

    id = Column(Integer, primary_key=True)
    home_team_id = Column(Integer, ForeignKey('teams.id'), nullable=False)
    away_team_id = Column(Integer, ForeignKey('teams.id'), nullable=False)
    match_date = Column(DateTime, nullable=False)
    league = Column(String(100))
    season = Column(String(20))

    # Results (NULL for fixtures)
    home_goals = Column(Integer)
    away_goals = Column(Integer)
    ht_home_goals = Column(Integer)  # Halftime
    ht_away_goals = Column(Integer)

    # Statistics
    home_shots = Column(Integer)
    away_shots = Column(Integer)
    home_shots_on_target = Column(Integer)
    away_shots_on_target = Column(Integer)
    home_possession = Column(Float)
    away_possession = Column(Float)
    home_corners = Column(Integer)
    away_corners = Column(Integer)
    home_fouls = Column(Integer)
    away_fouls = Column(Integer)
    home_yellow_cards = Column(Integer)
    away_yellow_cards = Column(Integer)
    home_red_cards = Column(Integer)
    away_red_cards = Column(Integer)

    # Expected Goals (from API-Football or Flashscore)
    home_xg = Column(Float)
    away_xg = Column(Float)

    # Extended statistics (from Flashscore)
    home_dangerous_attacks = Column(Integer)
    away_dangerous_attacks = Column(Integer)
    home_saves = Column(Integer)          # Goalkeeper saves
    away_saves = Column(Integer)
    home_offsides = Column(Integer)
    away_offsides = Column(Integer)
    home_free_kicks = Column(Integer)
    away_free_kicks = Column(Integer)

    # Match context (from Flashscore match detail page)
    referee = Column(String(100))
    venue = Column(String(150))
    venue_capacity = Column(Integer)

    # Score detail (for cup/playoff matches)
    regulation_home_goals = Column(Integer)  # Score at 90 min (excl. extra time)
    regulation_away_goals = Column(Integer)
    penalty_home_score = Column(Integer)
    penalty_away_score = Column(Integer)

    # Round / stage name (from API-Football, e.g. "Group Stage - 1", "Quarter-finals")
    round = Column(String(100))

    # API-Football fixture ID for cross-referencing
    apifootball_id = Column(Integer)

    # Flashscore short match ID (e.g. "G8MZEpbl") — used to scrape odds page
    flashscore_id = Column(String(20))

    # Match status
    is_fixture = Column(Boolean, default=False)
    created_at = Column(DateTime, default=utcnow)

    # Change marker for the local Parquet history mirror's incremental sync.
    #
    # On PostgreSQL the authoritative writer is the trg_matches_updated_at
    # trigger from migration 001, which also supplies the NOT NULL / DEFAULT
    # now() that the live column carries. The trigger is what catches the bulk
    # `query().update()` statements and raw SQL that SQLAlchemy's onupdate never
    # sees. The Python-side default/onupdate here keeps the column meaningful on
    # SQLite (dev only — no trigger, and no mirror either), and is deliberately
    # nullable so _migrate_missing_columns can add it to an existing dev DB
    # without needing a backfill.
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow, index=True)

    # Indexes on the columns hit by every form/xG/momentum query:
    #  · (home_team_id, is_fixture, match_date) — home form queries
    #  · (away_team_id, is_fixture, match_date) — away form queries
    #  · (is_fixture, match_date)               — upcoming fixtures scan
    #  · (league, is_fixture)                   — standings cache build
    __table_args__ = (
        Index("ix_match_home_team_fixture_date", "home_team_id", "is_fixture", "match_date"),
        Index("ix_match_away_team_fixture_date", "away_team_id", "is_fixture", "match_date"),
        Index("ix_match_fixture_date", "is_fixture", "match_date"),
        Index("ix_match_league_fixture", "league", "is_fixture"),
        # Date-only entry point for the batched odds prune, which selects victim
        # match ids by age. ix_match_fixture_date leads with is_fixture, so it
        # cannot serve a bare match_date range.
        Index("ix_matches_match_date", "match_date"),
    )

    # Relationships
    home_team = relationship("Team", foreign_keys=[home_team_id], back_populates="home_matches")
    away_team = relationship("Team", foreign_keys=[away_team_id], back_populates="away_matches")
    odds = relationship("Odds", back_populates="match")
    def __repr__(self):
        return f"<Match({self.home_team.name if self.home_team else 'TBD'} vs {self.away_team.name if self.away_team else 'TBD'}, {self.match_date})>"


class Player(Base):
    """Player model."""

    __tablename__ = 'players'

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    team_id = Column(Integer, ForeignKey('teams.id'), index=True)
    position = Column(String(50))
    is_key_player = Column(Boolean, default=False)

    # Relationships
    team = relationship("Team", back_populates="players")
    injuries = relationship("Injury", back_populates="player")

    def __repr__(self):
        return f"<Player(name='{self.name}', position='{self.position}')>"


class Injury(Base):
    """Injury model."""

    __tablename__ = 'injuries'

    id = Column(Integer, primary_key=True)
    player_id = Column(Integer, ForeignKey('players.id'), index=True)
    team_id = Column(Integer, ForeignKey('teams.id'), index=True)
    injury_type = Column(String(100))
    start_date = Column(Date)
    status = Column(String(50))  # 'out', 'doubtful', 'available'
    source = Column(String(200))
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)

    # Relationships
    player = relationship("Player", back_populates="injuries")
    team = relationship("Team", back_populates="injuries")

    def __repr__(self):
        return f"<Injury({self.player.name if self.player else 'Unknown'}, {self.injury_type}, {self.status})>"


class Odds(Base):
    """Odds model."""

    __tablename__ = 'odds'

    id = Column(Integer, primary_key=True)
    match_id = Column(Integer, ForeignKey('matches.id'), nullable=False)
    bookmaker = Column(String(50))
    market_type = Column(String(50))  # '1X2', 'over_under', 'btts', 'asian_handicap'
    selection = Column(String(50))
    odds_value = Column(Float, nullable=False)
    opening_odds = Column(Float, nullable=True)  # First-seen odds value (never overwritten)
    timestamp = Column(DateTime, default=utcnow)

    # Relationships
    match = relationship("Match", back_populates="odds")

    __table_args__ = (
        Index('ix_odds_match_bookie_market', 'match_id', 'bookmaker', 'market_type', 'selection', unique=True),
    )

    def __repr__(self):
        return f"<Odds({self.bookmaker}, {self.market_type}: {self.selection} @ {self.odds_value})>"


class SavedPick(Base):
    """Saved daily pick for tracking results and statistics."""

    __tablename__ = 'saved_picks'

    id = Column(Integer, primary_key=True)
    # index: hit by every per-match dedup, review lookup, footer read, and
    # settlement join — flagged by the Supabase performance advisor.
    match_id = Column(Integer, ForeignKey('matches.id'), nullable=False, index=True)
    pick_date = Column(Date, nullable=False)
    match_name = Column(String(200))
    league = Column(String(100))

    # Bet details
    market = Column(String(50))       # '1X2', 'Over 2.5', 'BTTS'
    selection = Column(String(100))   # 'Home Win', 'Over 2.5 Goals', 'BTTS Yes'
    odds = Column(Float)
    predicted_probability = Column(Float)
    expected_value = Column(Float)
    confidence = Column(Float)
    kelly_stake_percentage = Column(Float)
    risk_level = Column(String(20))
    used_fallback_odds = Column(Boolean, default=False)
    model_agreement = Column(String(20))  # unanimous/majority/split/solo — for analysis & filtering
    # Claude pick-review outcome (NULL = not reviewed). Makes the review's value
    # measurable: win rate of KEEP vs CHANGE picks answers "does the review help?"
    review_action = Column(String(10))    # 'KEEP' or 'CHANGE'
    review_reason = Column(String(500))   # Claude's one-line justification
    # The MODEL's ORIGINAL pick, snapshotted at save time and never overwritten
    # when Claude applies a CHANGE. Lets us measure Claude's true added value
    # (final win-rate vs what the model alone would have done) and, later,
    # distil it. On a KEEP these equal selection/market/predicted_probability.
    model_market = Column(String(50))
    model_selection = Column(String(100))
    model_probability = Column(Float)
    model_result = Column(String(10))     # win/loss/void of the model's ORIGINAL pick

    # --- Stage 5 experiment metadata -----------------------------------------
    # Which model configuration produced this prediction. Stages 1-4 changed the
    # blend weight, Poisson half-life, rho, the de-vigging rule and six betting
    # gates; none of it was recorded, so picks from different systems were
    # pooled in the same statistics. See src/models/model_version.py.
    model_version = Column(String(64))

    # pending | captured | missing | late | invalid.
    # Explicit so a NULL closing_odds is never ambiguous (never captured?
    # rejected? captured after kickoff?). Only 'captured' may feed CLV.
    closing_capture_status = Column(String(16), default="pending")
    closing_bookmaker_count = Column(Integer)
    closing_fair_probability = Column(Float)

    # The model's own EV for its ORIGINAL selection, snapshotted before the
    # Claude review could change anything. Completes the pre/post pair alongside
    # model_market / model_selection / model_probability.
    pre_claude_ev = Column(Float)

    # --- Prospective-measurement columns (Stage 4, Phase 12) -----------------
    # The market's own opinion AT PREDICTION TIME, stored alongside the model's.
    # Without it, nothing downstream can reconstruct what the model was
    # disagreeing with: `predicted_probability` was recorded but the de-vigged
    # consensus it was compared against was not, so every retrospective question
    # of the form "was the model's edge real?" had to re-derive the market from
    # odds rows that had since been overwritten.
    market_probability = Column(Float)      # de-vigged consensus for this selection
    market_books = Column(Integer)          # how many books backed that consensus

    # Recorded for measurement only, not offered as a live recommendation.
    # Set when betting.paper_trading_mode is on. Paper picks are excluded from
    # ROI reporting by default so they cannot flatter or damage the live record.
    is_paper = Column(Boolean, default=False)

    # Closing line for THIS selection — the consensus decimal price at (or as
    # close as possible to) kickoff. Populated by scripts/capture_closing_lines.py.
    #
    # Without it, closing line value cannot be computed at all: the 2026-08-07
    # audit found 0 of 124,158 odds rows on picked matches carried a timestamp
    # after the pick day, so the odds table holds a ~6h-before-kickoff snapshot
    # and nothing later. CLV is the strongest available diagnostic for a betting
    # model — a strategy with persistent positive CLV and short-run negative ROI
    # is working, and one with negative CLV is not, however its ROI looks.
    closing_odds = Column(Float)
    closing_odds_captured_at = Column(DateTime)

    # Result (NULL = pending)
    result = Column(String(10))       # 'win', 'loss', 'void', or NULL
    actual_home_goals = Column(Integer)
    actual_away_goals = Column(Integer)
    settled_at = Column(DateTime)

    created_at = Column(DateTime, default=utcnow)

    # Dedup was application-level read-then-write, which two concurrent workers
    # both pass. A duplicated pick is not cosmetic: every statistic here is a
    # count over this table — win rate, ROI, Brier, the Bayesian weight learner,
    # and the drawdown breaker that sizes real stakes. The database arbitrates
    # now; save_picks() still checks first as a cheap fast path.
    __table_args__ = (
        Index("ix_saved_picks_dedup", "match_id", "selection", "pick_date",
              unique=True),
    )

    # Relationships
    match = relationship("Match")

    def __repr__(self):
        status = self.result or "pending"
        return f"<SavedPick({self.match_name}: {self.selection} @ {self.odds} — {status})>"


class ApiBudget(Base):
    """Cross-process daily request budget for an external API.

    ``ApiFootballScraper`` counted spend in ``self._requests_today``, an instance
    attribute reset to 0 on construction and never persisted — so the "100
    requests/day" cap was enforced *per process*, and the daily job runs seven of
    them. This table makes the counter global, and quota is claimed with one
    conditional UPDATE whose row lock serialises concurrent claimants.
    """

    __tablename__ = 'api_budget'

    day = Column(Date, primary_key=True)
    provider = Column(String(50), primary_key=True)
    used = Column(Integer, nullable=False, default=0)
    # Trailing underscore: `limit` is a reserved word in several dialects.
    limit_ = Column("limit_", Integer, nullable=False)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)

    __table_args__ = (
        Index("ix_api_budget_day", "day"),
    )

    def __repr__(self):
        return f"<ApiBudget({self.provider} {self.day}: {self.used}/{self.limit_})>"


class PickObservation(Base):
    """One (pick, attribution) CLV observation — Stage 10, migration 006.

    The experiment measures two series and they are different bets:

        model — the FROZEN Stage 5 selection
        final — the selection actually persisted, after the Claude review

    ``taken_odds`` is written at pick-save time, BEFORE the review can
    overwrite anything, because that is the only moment the model's price
    exists. On a CHANGE, ``_apply_decision`` assigns
    ``primary.odds = float(new.odds)`` and the model's price is gone from
    ``saved_picks`` for good: the odds table holds one row per
    (match, bookmaker, market, selection) and overwrites it on every refresh,
    so there is no price history to recover it from. Inverting the stored EV is
    unsound for Draw No Bet, whose EV is scaled by an unstored P(decisive).

    On an unchanged pick both rows carry the same market, selection and price.
    Closing capture then resolves ONE close and attributes it to both — one API
    observation, two attributions, and still one fixture in the statistics.
    """

    __tablename__ = 'pick_observations'

    id = Column(Integer, primary_key=True)
    pick_id = Column(Integer, ForeignKey('saved_picks.id', ondelete='CASCADE'),
                     nullable=False, index=True)

    #: Declared so SQLAlchemy's unit of work knows a pick must be INSERTed
    #: before its observations. A bare column-level ForeignKey gives the DDL
    #: the constraint but gives the mapper no ordering dependency, so a session
    #: that adds both in one flush can emit them the wrong way round and trip
    #: the constraint.
    pick = relationship("SavedPick", backref="observations")

    #: 'model' or 'final'. Constrained in the database as well as here, so a
    #: third value cannot quietly create a series nothing reports on.
    attribution = Column(String(8), nullable=False)
    market = Column(String(50), nullable=False)
    selection = Column(String(100), nullable=False)

    #: The price at the moment of the pick. Never reconstructed.
    taken_odds = Column(Float, nullable=False)
    taken_at = Column(DateTime, nullable=False)

    closing_odds = Column(Float)
    closing_captured_at = Column(DateTime)
    closing_status = Column(String(16), nullable=False, default='pending')
    closing_book_count = Column(Integer)
    closing_fair_prob = Column(Float)

    __table_args__ = (
        UniqueConstraint('pick_id', 'attribution',
                         name='uq_pick_observations_pick_attribution'),
        CheckConstraint("attribution IN ('model', 'final')",
                        name='ck_pick_observations_attribution'),
        Index('ix_pick_observations_pending', 'closing_status'),
    )

    def __repr__(self):
        return (f"<PickObservation(pick={self.pick_id} {self.attribution} "
                f"{self.selection} @{self.taken_odds} "
                f"close={self.closing_odds} {self.closing_status})>")
