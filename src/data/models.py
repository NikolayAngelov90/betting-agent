"""Database models for Football Betting Agent."""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime,
    ForeignKey, Date, Text, JSON, Index
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
    flashscore_id = Column(String(50))
    transfermarkt_id = Column(String(50))
    apifootball_team_id = Column(Integer)  # API-Football team ID for history backfill
    created_at = Column(DateTime, default=utcnow)

    # Relationships
    home_matches = relationship("Match", foreign_keys="Match.home_team_id", back_populates="home_team")
    away_matches = relationship("Match", foreign_keys="Match.away_team_id", back_populates="away_team")
    players = relationship("Player", back_populates="team")
    injuries = relationship("Injury", back_populates="team")
    news = relationship("News", back_populates="team")

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

    # API-Football fixture ID for cross-referencing
    apifootball_id = Column(Integer)

    # Flashscore short match ID (e.g. "G8MZEpbl") — used to scrape odds page
    flashscore_id = Column(String(20))

    # Match status
    is_fixture = Column(Boolean, default=False)
    created_at = Column(DateTime, default=utcnow)

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
    team_id = Column(Integer, ForeignKey('teams.id'))
    position = Column(String(50))
    market_value = Column(Float)
    is_key_player = Column(Boolean, default=False)
    transfermarkt_id = Column(String(50))

    # Relationships
    team = relationship("Team", back_populates="players")
    injuries = relationship("Injury", back_populates="player")

    def __repr__(self):
        return f"<Player(name='{self.name}', position='{self.position}')>"


class Injury(Base):
    """Injury model."""

    __tablename__ = 'injuries'

    id = Column(Integer, primary_key=True)
    player_id = Column(Integer, ForeignKey('players.id'))
    team_id = Column(Integer, ForeignKey('teams.id'))
    injury_type = Column(String(100))
    start_date = Column(Date)
    expected_return = Column(Date)
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


class News(Base):
    """News model."""

    __tablename__ = 'news'

    id = Column(Integer, primary_key=True)
    team_id = Column(Integer, ForeignKey('teams.id'))
    headline = Column(Text)
    content = Column(Text)
    source = Column(String(100))
    url = Column(String(500))
    sentiment_score = Column(Float)  # -1 to +1
    published_at = Column(DateTime)
    created_at = Column(DateTime, default=utcnow)

    # Relationships
    team = relationship("Team", back_populates="news")

    def __repr__(self):
        return f"<News({self.headline[:50]}..., {self.source})>"


class SavedPick(Base):
    """Saved daily pick for tracking results and statistics."""

    __tablename__ = 'saved_picks'

    id = Column(Integer, primary_key=True)
    match_id = Column(Integer, ForeignKey('matches.id'), nullable=False)
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

    # Result (NULL = pending)
    result = Column(String(10))       # 'win', 'loss', 'void', or NULL
    actual_home_goals = Column(Integer)
    actual_away_goals = Column(Integer)
    settled_at = Column(DateTime)

    created_at = Column(DateTime, default=utcnow)

    # Relationships
    match = relationship("Match")

    def __repr__(self):
        status = self.result or "pending"
        return f"<SavedPick({self.match_name}: {self.selection} @ {self.odds} — {status})>"


