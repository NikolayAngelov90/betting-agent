"""A price with no known age is UNEVALUATED, never fresh. Stage 31 Part A.

`odds_snapshots.observed_at` is OUR clock: when we fetched the quote. It says
nothing about when the BOOK last changed it. Stage 30 found three Kambi books
winning 7-10pp above the anchor's fair price and could not say whether they
were generous or slow, because the one field that separates the two -- the
bookmaker's own `last_update`, which The Odds API returns on every bookmaker
and every market -- was read off the wire and dropped.

It is now stored with every snapshot, as `book_last_update`.

    THE ODDS API   per bookmaker AND per market. The market's stamp is the
                   quote's own; the bookmaker's is the fallback.
    API-FOOTBALL   one `update` per FIXTURE. That is not a bookmaker's own
                   stamp, so its rows carry NULL -- UNEVALUATED.
    OLDER ROWS     NULL, never backfilled.

THE CONTROL THE DIRECTIVE NAMES: two books in ONE response, with different
`last_update`, are stored differently. Driven end to end through
`_save_game_odds` on a scratch database.
"""
import ast
import pathlib
from datetime import datetime

import pytest

import src.data.database as db_mod
from src.data.models import Base, Match, Odds, OddsSnapshot, Team
from src.data.price_history import (
    FRESH_MAX_AGE_S,
    QUOTE_FRESH,
    QUOTE_STALE,
    QUOTE_UNEVALUATED,
    parse_book_last_update,
    quote_age_seconds,
    quote_freshness,
    record_price,
)
from src.scrapers.theodds_scraper import TheOddsScraper

HOME, AWAY = "Arsenal", "Chelsea"
MID = 9001


def _mgr(tmp_path, name="age.db"):
    mgr = db_mod.DatabaseManager(
        config=type("C", (), {"database": {"sqlite_path": str(tmp_path / name)}})())
    Base.metadata.create_all(mgr.engine)
    assert not mgr.is_postgres
    with mgr.get_session() as s:
        s.add(Team(id=1, name=HOME))
        s.add(Team(id=2, name=AWAY))
        s.flush()
        s.add(Match(id=MID, league="england/premier-league",
                    match_date=datetime(2026, 11, 21, 15, 0),
                    is_fixture=True, home_team_id=1, away_team_id=2))
        s.commit()
    return mgr


def _scraper(mgr):
    sc = TheOddsScraper.__new__(TheOddsScraper)
    sc.config = type("C", (), {"get": staticmethod(lambda *a, **k: None)})()
    sc.api_key = "test-key"
    sc.db = mgr
    sc._remaining_requests = None
    return sc


def _h2h(last_update=None):
    m = {"key": "h2h", "outcomes": [
        {"name": HOME, "price": 2.10}, {"name": AWAY, "price": 3.40},
        {"name": "Draw", "price": 3.50}]}
    if last_update is not None:
        m["last_update"] = last_update
    return m


def _totals(last_update=None):
    m = {"key": "totals", "outcomes": [
        {"name": "Over", "point": 2.5, "price": 1.90},
        {"name": "Under", "point": 2.5, "price": 1.95}]}
    if last_update is not None:
        m["last_update"] = last_update
    return m


def _book(key, markets, last_update=None):
    b = {"key": key, "markets": markets}
    if last_update is not None:
        b["last_update"] = last_update
    return b


def _save(mgr, bookmakers):
    game = {"home_team": HOME, "away_team": AWAY, "bookmakers": bookmakers}
    with mgr.get_session() as s:
        n = _scraper(mgr)._save_game_odds(game, MID, [], session=s)
        s.commit()
    return n


def _stamps(mgr):
    """{(bookmaker, market): {book_last_update, ...}} from the price path."""
    out = {}
    with mgr.get_session() as s:
        for r in s.query(OddsSnapshot).filter_by(match_id=MID).all():
            out.setdefault((r.bookmaker, r.market_type), set()).add(
                r.book_last_update)
    return out


# ═══════════════════════ the named control, driven ══════════════════════════

def test_TWO_BOOKS_in_ONE_response_are_stored_with_their_OWN_stamps(tmp_path):
    mgr = _mgr(tmp_path)
    written = _save(mgr, [
        _book("pinnacle", [_h2h("2026-11-21T09:58:41Z")],
              last_update="2026-11-21T09:58:41Z"),
        _book("unibet_se", [_h2h("2026-11-21T03:12:05Z")],
              last_update="2026-11-21T03:12:05Z"),
    ])
    assert written == 6
    got = _stamps(mgr)
    pin = got[("TheOddsAPI-pinnacle", "1X2")]
    uni = got[("TheOddsAPI-unibet_se", "1X2")]
    assert pin == {datetime(2026, 11, 21, 9, 58, 41)}
    assert uni == {datetime(2026, 11, 21, 3, 12, 5)}
    assert pin != uni, (
        "two books fetched in one response carry the SAME last_update: the "
        "stamp stored is the response's, not the bookmaker's")


def test_both_books_share_ONE_observed_at_and_DIFFER_in_age(tmp_path):
    """Which is the whole point: one fetch, two ages. `observed_at` cannot
    tell these apart and `book_last_update` can."""
    mgr = _mgr(tmp_path)
    _save(mgr, [
        _book("pinnacle", [_h2h("2026-11-21T09:58:41Z")]),
        _book("unibet_se", [_h2h("2026-11-21T03:12:05Z")]),
    ])
    with mgr.get_session() as s:
        rows = s.query(OddsSnapshot).filter_by(match_id=MID).all()
        by_book = {}
        for r in rows:
            by_book.setdefault(r.bookmaker, []).append(
                (r.observed_at, r.book_last_update))
    spread = max(o for v in by_book.values() for o, _ in v) - min(
        o for v in by_book.values() for o, _ in v)
    assert spread.total_seconds() < 5, "one response, one fetch"
    ages = {b: {quote_age_seconds(o, lu) for o, lu in v}
            for b, v in by_book.items()}
    pin = min(ages["TheOddsAPI-pinnacle"])
    uni = min(ages["TheOddsAPI-unibet_se"])
    assert uni - pin == pytest.approx(6 * 3600 + 46 * 60 + 36, abs=5)


def test_the_MARKET_stamp_wins_over_the_BOOKMAKER_stamp(tmp_path):
    """One book, two markets, two stamps. The bookmaker-level stamp is the
    newest of its markets, so using it for both would understate the age of
    the older market."""
    mgr = _mgr(tmp_path)
    _save(mgr, [_book(
        "pinnacle",
        [_h2h("2026-11-21T09:58:41Z"), _totals("2026-11-21T07:00:00Z")],
        last_update="2026-11-21T09:58:41Z")])
    got = _stamps(mgr)
    assert got[("TheOddsAPI-pinnacle", "1X2")] == {
        datetime(2026, 11, 21, 9, 58, 41)}
    assert got[("TheOddsAPI-pinnacle", "over_under")] == {
        datetime(2026, 11, 21, 7, 0, 0)}


def test_the_BOOKMAKER_stamp_is_the_FALLBACK_when_the_market_has_none(tmp_path):
    mgr = _mgr(tmp_path)
    _save(mgr, [_book("pinnacle", [_h2h()],
                      last_update="2026-11-21T09:58:41Z")])
    assert _stamps(mgr)[("TheOddsAPI-pinnacle", "1X2")] == {
        datetime(2026, 11, 21, 9, 58, 41)}


# ═══════════════════ no known age is UNEVALUATED, never fresh ═══════════════

def test_NO_STAMP_is_stored_as_NULL_and_reads_UNEVALUATED(tmp_path):
    """The known-wrong control. A writer that defaulted the stamp to its own
    fetch time would store age zero here -- the freshest quote in the table,
    made out of not knowing."""
    mgr = _mgr(tmp_path)
    _save(mgr, [_book("pinnacle", [_h2h()])])
    with mgr.get_session() as s:
        rows = s.query(OddsSnapshot).filter_by(match_id=MID).all()
        assert len(rows) == 3
        for r in rows:
            assert r.book_last_update is None, (
                f"a quote with no last_update was stamped "
                f"{r.book_last_update}: its age is now a made-up number")
            assert quote_age_seconds(
                r.observed_at, r.book_last_update) is None
            assert quote_freshness(
                r.observed_at, r.book_last_update) == QUOTE_UNEVALUATED


@pytest.mark.parametrize("raw", [None, "", "   ", "yesterday", 1731405521,
                                 "2026-13-45T99:00:00Z", {}, []])
def test_an_UNREADABLE_stamp_is_NONE_never_now(raw):
    assert parse_book_last_update(raw) is None


def test_a_READABLE_stamp_becomes_NAIVE_UTC():
    assert parse_book_last_update("2026-11-21T09:58:41Z") == datetime(
        2026, 11, 21, 9, 58, 41)
    # an offset is CONVERTED, not dropped
    assert parse_book_last_update("2026-11-21T11:58:41+02:00") == datetime(
        2026, 11, 21, 9, 58, 41)
    naive = datetime(2026, 11, 21, 9, 58, 41)
    assert parse_book_last_update(naive) == naive


def test_the_THREE_STATES_and_the_boundary():
    obs = datetime(2026, 11, 21, 10, 0, 0)
    fresh = datetime(2026, 11, 21, 9, 55, 0)
    assert quote_freshness(obs, fresh) == QUOTE_FRESH
    edge = datetime(2026, 11, 21, 9, 50, 0)
    assert quote_age_seconds(obs, edge) == FRESH_MAX_AGE_S
    assert quote_freshness(obs, edge) == QUOTE_FRESH
    assert quote_freshness(
        obs, datetime(2026, 11, 21, 9, 49, 59)) == QUOTE_STALE
    assert quote_freshness(obs, None) == QUOTE_UNEVALUATED
    assert quote_freshness(None, fresh) == QUOTE_UNEVALUATED


def test_a_stamp_AFTER_our_observation_is_NOT_clamped_to_fresh():
    """A negative age is a clock disagreement. Clamping it to zero would
    manufacture the freshest quote in the table."""
    obs = datetime(2026, 11, 21, 10, 0, 0)
    later = datetime(2026, 11, 21, 10, 5, 0)
    assert quote_age_seconds(obs, later) == -300.0
    assert quote_freshness(obs, later) == QUOTE_STALE


# ═════════════════════ what did NOT change, and what cannot ═════════════════

def test_the_ODDS_table_is_UNTOUCHED(tmp_path):
    """The stamp goes to the price PATH only. `odds` keeps its columns and its
    contents, so every current-price consumer reads what it read before."""
    assert "book_last_update" not in {c.name for c in Odds.__table__.columns}
    assert "book_last_update" in {
        c.name for c in OddsSnapshot.__table__.columns}
    mgr = _mgr(tmp_path)
    _save(mgr, [_book("pinnacle", [_h2h("2026-11-21T09:58:41Z")])])
    with mgr.get_session() as s:
        got = {o.selection: o.odds_value for o in
               s.query(Odds).filter_by(match_id=MID, market_type="1X2")}
    assert got == {"Home": 2.10, "Draw": 3.50, "Away": 3.40}


def test_the_COLUMN_IS_NULLABLE_so_the_auto_migration_can_add_it(tmp_path):
    """`_migrate_missing_columns` adds only NULLABLE columns. Driven: a
    database whose `odds_snapshots` predates the column gains it, and its old
    rows read NULL -- UNEVALUATED, not backfilled."""
    col = OddsSnapshot.__table__.columns["book_last_update"]
    assert col.nullable is True and col.default is None
    mgr = _mgr(tmp_path, "old.db")
    with mgr.engine.begin() as c:
        c.exec_driver_sql("DROP TABLE odds_snapshots")
        c.exec_driver_sql(
            "CREATE TABLE odds_snapshots (id INTEGER PRIMARY KEY, "
            "match_id INTEGER NOT NULL, bookmaker VARCHAR(100) NOT NULL, "
            "market_type VARCHAR(50) NOT NULL, selection VARCHAR(50) NOT NULL, "
            "odds_value FLOAT NOT NULL, observed_at DATETIME NOT NULL)")
        c.exec_driver_sql(
            "INSERT INTO odds_snapshots (match_id, bookmaker, market_type, "
            "selection, odds_value, observed_at) VALUES "
            f"({MID}, 'Pinnacle', '1X2', 'Home', 2.1, '2026-11-01 08:00:00')")
    mgr._migrate_missing_columns()
    with mgr.get_session() as s:
        rows = s.query(OddsSnapshot).all()
        assert [(r.bookmaker, r.book_last_update) for r in rows] == [
            ("Pinnacle", None)]


def test_API_FOOTBALL_passes_NO_stamp_and_that_is_DECLARED():
    """From the AST, by set equality: the keywords API-Football's one
    `_record_price` call passes. `book_last_update` is not among them, so its
    rows are NULL by construction. The Odds API's call passes it."""
    def _kw(path):
        tree = ast.parse(pathlib.Path(path).read_text(encoding="utf-8"))
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and getattr(n.func, "id", "") == "_record_price"]
        assert len(calls) == 1, (path, len(calls))
        return {k.arg for k in calls[0].keywords}

    base = {"match_id", "bookmaker", "market_type", "selection", "odds_value",
            "observed_at"}
    assert _kw("src/scrapers/apifootball_scraper.py") == base
    assert _kw("src/scrapers/theodds_scraper.py") == base | {"book_last_update"}


def test_record_price_STORES_WHAT_IT_IS_GIVEN(tmp_path):
    mgr = _mgr(tmp_path)
    with mgr.get_session() as s:
        record_price(s, match_id=MID, bookmaker="X", market_type="1X2",
                     selection="Home", odds_value=2.0,
                     observed_at=datetime(2026, 11, 21, 10, 0, 0),
                     book_last_update="2026-11-21T09:00:00Z")
        record_price(s, match_id=MID, bookmaker="Y", market_type="1X2",
                     selection="Home", odds_value=2.0,
                     observed_at=datetime(2026, 11, 21, 10, 0, 0))
        s.commit()
        got = {r.bookmaker: r.book_last_update
               for r in s.query(OddsSnapshot).all()}
    assert got == {"X": datetime(2026, 11, 21, 9, 0, 0), "Y": None}


def test_the_FINGERPRINT_DID_NOT_MOVE():
    """Instrumentation, not selection. The cohort is the one Part G left."""
    from src.models.model_version import CODE_REVISION, model_version
    from src.utils.config import Config
    from tests.experiment_pins import (CODE_REVISION_PIN,
                                       FROZEN_MODEL_VERSION)
    assert CODE_REVISION == CODE_REVISION_PIN
    assert model_version(
        Config("config/config.example.yaml")) == FROZEN_MODEL_VERSION


# ═════════════════════ the census: three states, never a rate of nothing ════

def test_the_CENSUS_keeps_UNEVALUATED_out_of_the_rate():
    """Four quotes: one fresh, one stale, two with no stamp. The fresh share
    is over the two EVALUATED quotes -- 50% -- and the two unknowns are
    counted beside it, not inside it."""
    from scripts.quote_age import tally
    obs = datetime(2026, 11, 21, 10, 0, 0)
    r = tally([
        ("TheOddsAPI-pinnacle", obs, datetime(2026, 11, 21, 9, 58, 0)),
        ("TheOddsAPI-unibet_se", obs, datetime(2026, 11, 21, 3, 0, 0)),
        ("Pinnacle", obs, None),
        ("Bet365", obs, None),
    ])
    assert r["rows"] == 4
    assert r["overall"] == {QUOTE_FRESH: 1, QUOTE_STALE: 1,
                            QUOTE_UNEVALUATED: 2}
    assert r["fresh_rate_of_evaluated"] == 0.5
    assert r["by_feed"]["apifootball"][QUOTE_UNEVALUATED] == 2
    assert r["by_feed"]["apifootball"]["fresh_rate_of_evaluated"] is None
    assert r["by_book"]["TheOddsAPI-unibet_se"]["median_age_s"] == 25200.0


def test_NO_STAMPED_QUOTE_AT_ALL_is_a_NONE_rate_not_a_zero():
    """The state production is in today: 251,064 quotes, none stamped. A
    rate of 0.0 would read as "every quote is stale"."""
    from scripts.quote_age import tally
    obs = datetime(2026, 11, 21, 10, 0, 0)
    r = tally([("Pinnacle", obs, None)] * 3)
    assert r["overall"][QUOTE_UNEVALUATED] == 3
    assert r["fresh_rate_of_evaluated"] is None
    assert tally([])["fresh_rate_of_evaluated"] is None


#: `measured` 2026-11-12, production, grain = one stored quote, BEFORE the
#: first run that carries the column.
CENSUS_AT_INSTALL = {"rows": 251064, "fresh": 0, "stale": 0,
                     "unevaluated": 251064, "column_present": False,
                     "observed": ("2026-08-25", "2026-11-09")}


def test_the_INSTALL_DAY_reading_is_ALL_UNEVALUATED():
    c = CENSUS_AT_INSTALL
    assert c["fresh"] + c["stale"] + c["unevaluated"] == c["rows"]
    assert c["unevaluated"] == c["rows"] and c["column_present"] is False
