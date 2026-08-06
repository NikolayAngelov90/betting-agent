"""Every query touched by the egress work must compile to valid PostgreSQL.

The test suite runs on SQLite. That is fine for logic, but it cannot catch a
query that is well-formed for SQLite and invalid — or subtly different — on the
production dialect. Two real defects were found exactly this way during the
work: a projected join with no determinable left side (needed an explicit
``select_from``), and a window-function scope whose subquery had to be aliased.

So: render each shape with the PostgreSQL dialect and assert on the SQL text.
Compilation failures raise; the assertions pin the properties that matter —
that projections really are projections, and that no query smuggles in a
``SELECT matches.*``.
"""

import pytest
from sqlalchemy import create_engine, func, or_, tuple_
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import Session

from src.data.models import Match, Odds, Team, Player, Injury, SavedPick


@pytest.fixture(scope="module")
def session():
    return Session(bind=create_engine("sqlite://"))


def render(query) -> str:
    """Compile to PostgreSQL with literals inlined. Raises on invalid SQL."""
    return str(query.statement.compile(
        dialect=postgresql.dialect(),
        compile_kwargs={"literal_binds": True},
    ))


ALL_MATCH_COLUMNS = len(Match.__table__.columns)


def selected_columns(sql: str) -> int:
    head = sql.split("\nFROM")[0]
    return head.count(",") + 1


class TestProjectionsAreRealProjections:
    """A projection that quietly selects everything is not a projection."""

    def test_match_table_is_wide_enough_for_this_to_matter(self):
        assert ALL_MATCH_COLUMNS >= 40, (
            f"matches has {ALL_MATCH_COLUMNS} columns; the projection work "
            f"assumed ~46")

    def test_history_core_projection(self, session):
        from src.data.match_history import _CORE_COLUMNS
        sql = render(session.query(*_CORE_COLUMNS).filter(
            Match.is_fixture == False, Match.home_goals.isnot(None)))  # noqa: E712
        assert selected_columns(sql) == 9
        assert "matches.venue" not in sql
        assert "matches.created_at" not in sql

    def test_form_features_projection(self, session):
        sql = render(session.query(
            Match.home_team_id, Match.away_team_id,
            Match.home_goals, Match.away_goals,
            Match.home_shots, Match.away_shots,
            Match.home_shots_on_target, Match.away_shots_on_target,
            Match.home_possession, Match.away_possession,
            Match.home_corners, Match.away_corners,
            Match.home_dangerous_attacks, Match.away_dangerous_attacks,
            Match.home_saves, Match.away_saves,
            Match.home_offsides, Match.away_offsides,
            Match.home_free_kicks, Match.away_free_kicks,
        ).filter(Match.is_fixture == False))  # noqa: E712
        assert selected_columns(sql) == 20 < ALL_MATCH_COLUMNS

    def test_odds_projection(self, session):
        sql = render(session.query(
            Odds.market_type, Odds.selection, Odds.odds_value,
            Odds.bookmaker, Odds.opening_odds,
        ).filter(Odds.match_id == 1))
        assert selected_columns(sql) == 5
        assert "odds.timestamp" not in sql

    def test_saved_picks_projection_excludes_the_wide_text_column(self, session):
        sql = render(session.query(
            SavedPick.result, SavedPick.odds, SavedPick.kelly_stake_percentage,
            SavedPick.pick_date, SavedPick.market,
            SavedPick.predicted_probability, SavedPick.used_fallback_odds,
        ))
        assert selected_columns(sql) == 7
        # review_reason is VARCHAR(500) and dominated the old full-row payload.
        assert "review_reason" not in sql


class TestTrickyShapesCompile:
    """Shapes SQLite would accept but PostgreSQL might not."""

    def test_injury_join_has_an_explicit_left_side(self, session):
        """Selecting only Player columns leaves the FROM ambiguous.

        Without select_from(Injury) SQLAlchemy raises InvalidRequestError —
        this is the defect the compile check originally caught.
        """
        sql = render(session.query(
            Player.is_key_player, Player.position,
        ).select_from(Injury).join(
            Player, Injury.player_id == Player.id
        ).filter(
            Injury.team_id == 1,
            Injury.status.in_(["out", "doubtful"]),
        ))
        assert "FROM injuries JOIN players" in sql
        assert "players.is_key_player" in sql
        assert "injuries.injury_type" not in sql

    def test_referee_window_function_scope(self, session):
        """The referee scope caps per-referee rows with row_number()."""
        rn = func.row_number().over(
            partition_by=Match.referee,
            order_by=(Match.match_date.desc(), Match.id.desc()),
        ).label("rn")
        sub = session.query(
            Match.id, Match.match_date, Match.referee,
            Match.home_goals, Match.away_goals,
            Match.home_yellow_cards, Match.away_yellow_cards,
            Match.home_red_cards, Match.away_red_cards,
            Match.home_fouls, Match.away_fouls,
            rn,
        ).filter(
            Match.referee.in_(["Mike Dean"]),
            Match.is_fixture == False,  # noqa: E712
            Match.home_goals.isnot(None),
        ).subquery()
        sql = render(session.query(sub).filter(sub.c.rn <= 30))
        assert "ROW_NUMBER() OVER" in sql.upper()
        assert "PARTITION BY matches.referee" in sql

    def test_grouped_odds_count(self, session):
        sql = render(session.query(Odds.match_id, func.count(Odds.id))
                     .filter(Odds.match_id.in_([1, 2]))
                     .group_by(Odds.match_id))
        assert "GROUP BY odds.match_id" in sql
        assert "count(odds.id)" in sql

    def test_row_value_in_for_pairings(self, session):
        """Row-value IN is valid in PostgreSQL; assert it renders as one."""
        sql = render(session.query(Match.id).filter(
            tuple_(Match.home_team_id, Match.away_team_id).in_([(1, 2), (2, 1)])))
        assert "(matches.home_team_id, matches.away_team_id) IN" in sql

    def test_incremental_sync_query(self, session):
        from datetime import datetime
        sql = render(session.query(
            Match.id, Match.match_date, Match.home_team_id, Match.away_team_id,
            Match.home_goals, Match.away_goals, Match.home_xg, Match.away_xg,
            Match.league, Match.updated_at, Match.is_fixture,
        ).filter(Match.updated_at >= datetime(2026, 1, 1)
                 ).order_by(Match.updated_at.asc()))
        assert "matches.updated_at >=" in sql
        assert "ORDER BY matches.updated_at ASC" in sql
        # Deliberately unfiltered on is_fixture/home_goals: membership of the
        # completed set is decided per row after the fetch.
        assert "matches.is_fixture =" not in sql

    def test_or_filtered_team_history(self, session):
        sql = render(session.query(
            Match.match_date, Match.home_team_id, Match.away_team_id,
            Match.home_goals, Match.away_goals,
            Match.regulation_home_goals, Match.regulation_away_goals,
        ).filter(
            Match.is_fixture == False,  # noqa: E712
            or_(Match.home_team_id == 1, Match.away_team_id == 1),
        ).order_by(Match.match_date.desc()).limit(10))
        assert "LIMIT 10" in sql
        assert selected_columns(sql) == 7


class TestIndexedColumnsExistOnTheModel:
    """The migration and the ORM must not drift apart."""

    def test_matches_has_updated_at(self):
        assert "updated_at" in Match.__table__.columns

    def test_teams_indexes_are_declared(self):
        names = {ix.name for ix in Team.__table__.indexes}
        assert {"ix_teams_name", "ix_teams_league",
                "ix_teams_apifootball_team_id"} <= names

    def test_updated_at_is_indexed(self):
        indexed = {
            col.name
            for ix in Match.__table__.indexes
            for col in ix.columns
        }
        assert "updated_at" in indexed
