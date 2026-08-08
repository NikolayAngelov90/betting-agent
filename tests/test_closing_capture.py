"""Stage 5, Phases 3, 5, 8 and 10 — closing-line capture and CLV.

Covers the full prospective lifecycle against SQLite (never production):
create prediction -> store odds -> capture closing -> settle -> compute CLV,
plus the ten CLV movement/validity cases Phase 8 requires.
"""

from datetime import datetime, timedelta

import pytest

from src.data.market_spec import check_overround, devig
from src.evaluation.clv import CLVInvalid, compute, validate_pair


# ═══════════════════════════════════════════ Phase 8 — the ten required cases

KO = datetime(2026, 9, 1, 19, 0)
CAP = KO - timedelta(minutes=60)


def _base(**over):
    d = dict(taken_odds=2.20, closing_odds=2.00,
             pick_market="1X2", closing_market="1X2",
             pick_selection="Home Win", closing_selection="Home Win",
             kickoff=KO, captured_at=CAP)
    d.update(over)
    return d


def test_1_favourable_movement():
    """Took 2.20, closed 2.00 — the market came to us."""
    r = compute(taken_odds=2.20, closing_odds=2.00)
    assert r.price_clv == pytest.approx(0.10)
    assert r.beat_close is True
    assert r.prob_clv < 0          # our implied prob was LOWER than the close's


def test_2_unfavourable_movement():
    r = compute(taken_odds=1.90, closing_odds=2.10)
    assert r.price_clv == pytest.approx(1.90 / 2.10 - 1)
    assert r.price_clv < 0
    assert r.beat_close is False


def test_3_unchanged_odds_is_exactly_zero():
    r = compute(taken_odds=2.00, closing_odds=2.00)
    assert r.price_clv == pytest.approx(0.0)
    assert r.prob_clv == pytest.approx(0.0)
    assert r.beat_close is False    # zero is not "beat"


def test_4_missing_close_is_refused_not_defaulted():
    with pytest.raises(CLVInvalid):
        validate_pair(**_base(closing_odds=None))
    with pytest.raises(CLVInvalid):
        compute(taken_odds=2.2, closing_odds=None)


def test_5_invalid_timestamp_after_kickoff():
    with pytest.raises(CLVInvalid, match="AFTER kickoff"):
        validate_pair(**_base(captured_at=KO + timedelta(seconds=1)))


def test_5b_invalid_timestamp_far_before_kickoff():
    with pytest.raises(CLVInvalid, match="beyond"):
        validate_pair(**_base(captured_at=KO - timedelta(hours=8)))


def test_6_bookmaker_mismatch_is_not_silently_accepted():
    """The pair carries no bookmaker identity, so a consensus close is compared
    against a consensus take. Mixing a single-book close with a consensus take
    would be a category error — guarded by keeping both sides consensus-derived
    in capture_closing_lines, and asserted here on the market/selection identity
    that the pair DOES carry."""
    with pytest.raises(CLVInvalid, match="market mismatch"):
        validate_pair(**_base(closing_market="draw_no_bet"))


def test_7_market_mismatch_home_away_vs_1x2():
    """The exact corruption class: a two-way price must never be compared to a
    three-way pick."""
    with pytest.raises(CLVInvalid, match="market mismatch"):
        validate_pair(**_base(pick_market="1X2", closing_market="draw_no_bet"))


def test_8_selection_mismatch():
    with pytest.raises(CLVInvalid, match="selection mismatch"):
        validate_pair(**_base(closing_selection="Away Win"))


def test_9_different_decimal_odds_scales():
    """CLV is scale-free: a 10% better price is +10% at any price level.

    (1.00 is not a valid decimal price — it pays nothing — so the short end of
    the range starts at 1.10/1.21.)
    """
    for taken, closing in [(1.21, 1.10), (2.20, 2.00), (11.0, 10.0), (5.5, 5.0)]:
        assert compute(taken_odds=taken, closing_odds=closing).price_clv == \
            pytest.approx(0.10)


def test_9b_a_price_of_one_is_rejected_as_invalid():
    """Decimal 1.00 returns only the stake; it is not a tradeable price."""
    with pytest.raises(CLVInvalid):
        compute(taken_odds=1.10, closing_odds=1.00)


def test_10_multiple_bookmakers_use_the_median_consensus():
    from scripts.capture_closing_lines import consensus_close

    class _R:
        def __init__(self, bk, sel, odds, mt="1X2"):
            self.bookmaker, self.selection = bk, sel
            self.odds_value, self.market_type = odds, mt

    rows = []
    for bk, (h, d, a) in {
        "A": (2.00, 3.50, 4.00),
        "B": (2.10, 3.40, 3.90),
        "C": (1.90, 3.60, 4.10),
    }.items():
        rows += [_R(bk, "Home", h), _R(bk, "Draw", d), _R(bk, "Away", a)]

    price, fair, n = consensus_close(rows, "1X2", None, None, 0)
    assert n == 3
    assert price == pytest.approx(2.00)      # median of 1.90/2.00/2.10
    assert 0 < fair < 1


# ═════════════════════════════════════ Phase 10 — validation on CLOSING odds

def test_corrupt_closing_book_is_excluded_from_the_consensus():
    """The Stage 4 gate applies to closing prices too. Real production values."""
    from scripts.capture_closing_lines import consensus_close

    class _R:
        def __init__(self, bk, sel, odds, mt="1X2"):
            self.bookmaker, self.selection = bk, sel
            self.odds_value, self.market_type = odds, mt

    rows = []
    for bk, (h, d, a) in {
        "Bet365": (1.25, 3.40, 3.75),     # two-way contamination, overround 1.361
        "Pinnacle": (1.71, 3.66, 4.55),
        "1xBet": (1.74, 3.81, 4.89),
    }.items():
        rows += [_R(bk, "Home", h), _R(bk, "Draw", d), _R(bk, "Away", a)]

    price, fair, n = consensus_close(rows, "1X2", None, None, 0)
    assert n == 2, "the corrupt book was not excluded from the closing consensus"
    assert price == pytest.approx((1.71 + 1.74) / 2)


def test_flashscore_never_contributes_to_a_close():
    from scripts.capture_closing_lines import consensus_close

    class _R:
        def __init__(self, bk, sel, odds, mt="1X2"):
            self.bookmaker, self.selection = bk, sel
            self.odds_value, self.market_type = odds, mt

    rows = []
    for bk in ("Flashscore", "Pinnacle"):
        rows += [_R(bk, "Home", 1.71), _R(bk, "Draw", 3.66), _R(bk, "Away", 4.55)]
    _, _, n = consensus_close(rows, "1X2", None, None, 0)
    assert n == 1


def test_overlapping_market_yields_price_but_no_fair_probability():
    """Double chance cannot be de-vigged, so closing_fair_probability stays NULL
    while the raw price comparison still works."""
    from scripts.capture_closing_lines import consensus_close

    class _R:
        def __init__(self, bk, sel, odds):
            self.bookmaker, self.selection = bk, sel
            self.odds_value, self.market_type = odds, "double_chance"

    rows = [_R("A", "Double Chance 1X", 1.22), _R("A", "Double Chance 12", 1.30),
            _R("A", "Double Chance X2", 1.83)]
    price, fair, n = consensus_close(rows, "double_chance", None, None, 0)
    assert n == 1 and price == pytest.approx(1.22)
    assert fair is None


# ═══════════════════════════════════════ Phase 5 — mapping completeness

def test_every_tradeable_selection_is_mappable():
    """A selection the pipeline can pick but the capturer cannot map becomes a
    permanent hole in CLV coverage. Production hit this with 'Double Chance 1X'."""
    from scripts.capture_closing_lines import SELECTION_SPEC
    from src.betting.value_calculator import ValueBettingCalculator

    ensemble = {k: 0.5 for k in (
        "home_win", "draw", "away_win", "over_1.5", "over_2.5", "over_3.5",
        "over_4.5", "under_1.5", "under_2.5", "under_3.5", "under_4.5",
        "btts_yes", "btts_no", "home_over_0.5", "away_over_0.5",
        "home_over_1.5", "away_over_1.5", "dc_1x", "dc_12", "dc_x2",
        "dnb_home", "dnb_away")}
    calc = ValueBettingCalculator.__new__(ValueBettingCalculator)
    selections = {s for _, s, _, _ in calc._market_specs(ensemble)}

    missing = selections - set(SELECTION_SPEC)
    assert not missing, (
        f"selections the pipeline can pick but capture_closing_lines cannot map: "
        f"{sorted(missing)}")


def test_selection_spec_leg_indices_are_within_their_market():
    from scripts.capture_closing_lines import SELECTION_SPEC
    from src.data.market_spec import get_spec

    for selection, (market_type, _line, _side, leg) in SELECTION_SPEC.items():
        spec = get_spec(market_type)
        assert spec is not None, f"{selection} -> undeclared market {market_type}"
        assert 0 <= leg < spec.arity, (
            f"{selection} points at leg {leg} of a {spec.arity}-leg market")


# ═══════════════════════ Phase 3 — full prospective lifecycle (SQLite only)

def test_full_lifecycle_predict_capture_settle_clv(tmp_path, monkeypatch):
    """create prediction -> odds -> closing capture -> settle -> CLV, end to end.

    conftest strips DATABASE_URL, so this runs against a temp SQLite file and
    can never reach production.
    """
    import src.data.database as db_mod
    from src.data.models import Base, Match, Odds, SavedPick

    monkeypatch.setattr(db_mod, "_db_manager", None, raising=False)
    mgr = db_mod.DatabaseManager(
        config=type("C", (), {"database": {"sqlite_path": str(tmp_path / "t.db")}})())
    Base.metadata.create_all(mgr.engine)

    kickoff = datetime.utcnow() + timedelta(minutes=45)
    with mgr.get_session() as s:
        s.add(Match(id=1, home_team_id=1, away_team_id=2, match_date=kickoff,
                    league="x/y", is_fixture=True))
        # 1. store odds — three books, one of them corrupt
        for bk, (h, d, a) in {
            "Pinnacle": (2.00, 3.50, 4.00),
            "1xBet": (2.10, 3.40, 3.90),
            "Bet365": (1.25, 3.40, 3.75),      # corrupt: overround 1.361
        }.items():
            for sel, o in (("Home", h), ("Draw", d), ("Away", a)):
                s.add(Odds(match_id=1, bookmaker=bk, market_type="1X2",
                           selection=sel, odds_value=o))
        # 2. create prediction
        s.add(SavedPick(
            id=1, match_id=1, pick_date=kickoff.date(), match_name="A vs B",
            market="1X2", selection="Home Win", odds=2.20,
            predicted_probability=0.55, expected_value=0.21, confidence=0.55,
            kelly_stake_percentage=1.0, closing_capture_status="pending",
            model_version="test_v1", is_paper=True))
        s.commit()

    monkeypatch.setattr(db_mod, "_db_manager", mgr, raising=False)
    from scripts import capture_closing_lines as cap
    monkeypatch.setattr(cap, "get_db", lambda: mgr)

    # 3. capture the closing line
    stats = cap.capture(within_minutes=120)
    assert stats["captured"] == 1, stats

    with mgr.get_session() as s:
        pick = s.get(SavedPick, 1)
        assert pick.closing_capture_status == "captured"
        # Corrupt Bet365 excluded -> median of 2.00 and 2.10.
        assert pick.closing_odds == pytest.approx(2.05)
        assert pick.closing_bookmaker_count == 2
        assert pick.closing_odds_captured_at is not None
        assert 0 < pick.closing_fair_probability < 1

        # 4. settle
        m = s.get(Match, 1)
        m.is_fixture, m.home_goals, m.away_goals = False, 2, 0
        pick.result, pick.actual_home_goals, pick.actual_away_goals = "win", 2, 0
        s.commit()

        # 5. CLV from the stored raw prices
        r = compute(taken_odds=pick.odds, closing_odds=pick.closing_odds,
                    kickoff=kickoff, captured_at=pick.closing_odds_captured_at)
        assert r.price_clv == pytest.approx(2.20 / 2.05 - 1)
        assert r.beat_close is True
        validate_pair(
            taken_odds=pick.odds, closing_odds=pick.closing_odds,
            pick_market=pick.market, closing_market=pick.market,
            pick_selection=pick.selection, closing_selection=pick.selection,
            kickoff=kickoff, captured_at=pick.closing_odds_captured_at)


def test_capture_is_idempotent(tmp_path, monkeypatch):
    """A second run must not re-capture or overwrite an existing closing price."""
    import src.data.database as db_mod
    from src.data.models import Base, Match, Odds, SavedPick

    mgr = db_mod.DatabaseManager(
        config=type("C", (), {"database": {"sqlite_path": str(tmp_path / "i.db")}})())
    Base.metadata.create_all(mgr.engine)
    kickoff = datetime.utcnow() + timedelta(minutes=45)
    with mgr.get_session() as s:
        s.add(Match(id=1, home_team_id=1, away_team_id=2, match_date=kickoff,
                    league="x/y", is_fixture=True))
        for bk in ("Pinnacle", "1xBet"):
            for sel, o in (("Home", 2.0), ("Draw", 3.5), ("Away", 4.0)):
                s.add(Odds(match_id=1, bookmaker=bk, market_type="1X2",
                           selection=sel, odds_value=o))
        s.add(SavedPick(id=1, match_id=1, pick_date=kickoff.date(),
                        market="1X2", selection="Home Win", odds=2.20,
                        closing_capture_status="pending"))
        s.commit()

    from scripts import capture_closing_lines as cap
    monkeypatch.setattr(cap, "get_db", lambda: mgr)

    first = cap.capture(within_minutes=120)
    assert first["captured"] == 1
    with mgr.get_session() as s:
        captured_at = s.get(SavedPick, 1).closing_odds_captured_at
        price = s.get(SavedPick, 1).closing_odds

    second = cap.capture(within_minutes=120)
    assert second["captured"] == 0
    assert second["considered"] == 0, "an already-captured pick was reconsidered"
    with mgr.get_session() as s:
        pick = s.get(SavedPick, 1)
        assert pick.closing_odds == price
        assert pick.closing_odds_captured_at == captured_at


def test_late_capture_is_marked_not_captured(tmp_path, monkeypatch):
    import src.data.database as db_mod
    from src.data.models import Base, Match, Odds, SavedPick

    mgr = db_mod.DatabaseManager(
        config=type("C", (), {"database": {"sqlite_path": str(tmp_path / "l.db")}})())
    Base.metadata.create_all(mgr.engine)
    started = datetime.utcnow() - timedelta(minutes=10)   # already kicked off
    with mgr.get_session() as s:
        s.add(Match(id=1, home_team_id=1, away_team_id=2, match_date=started,
                    league="x/y", is_fixture=True))
        for sel, o in (("Home", 2.0), ("Draw", 3.5), ("Away", 4.0)):
            s.add(Odds(match_id=1, bookmaker="Pinnacle", market_type="1X2",
                       selection=sel, odds_value=o))
        s.add(SavedPick(id=1, match_id=1, pick_date=started.date(),
                        market="1X2", selection="Home Win", odds=2.20,
                        closing_capture_status="pending"))
        s.commit()

    from scripts import capture_closing_lines as cap
    monkeypatch.setattr(cap, "get_db", lambda: mgr)
    stats = cap.capture(within_minutes=120)
    assert stats["late"] == 1 and stats["captured"] == 0
    with mgr.get_session() as s:
        pick = s.get(SavedPick, 1)
        assert pick.closing_capture_status == "late"
        assert pick.closing_odds is None, "a late capture must not set a price"


def test_missing_price_is_recorded_not_invented(tmp_path, monkeypatch):
    import src.data.database as db_mod
    from src.data.models import Base, Match, SavedPick

    mgr = db_mod.DatabaseManager(
        config=type("C", (), {"database": {"sqlite_path": str(tmp_path / "m.db")}})())
    Base.metadata.create_all(mgr.engine)
    kickoff = datetime.utcnow() + timedelta(minutes=45)
    with mgr.get_session() as s:
        s.add(Match(id=1, home_team_id=1, away_team_id=2, match_date=kickoff,
                    league="x/y", is_fixture=True))
        s.add(SavedPick(id=1, match_id=1, pick_date=kickoff.date(),
                        market="1X2", selection="Home Win", odds=2.20,
                        closing_capture_status="pending"))
        s.commit()   # no odds at all

    from scripts import capture_closing_lines as cap
    monkeypatch.setattr(cap, "get_db", lambda: mgr)
    stats = cap.capture(within_minutes=120)
    assert stats["missing"] == 1
    with mgr.get_session() as s:
        pick = s.get(SavedPick, 1)
        assert pick.closing_capture_status == "missing"
        assert pick.closing_odds is None


# ═══════════════════════════════════ Phase 1 — model version identifier

def test_model_version_changes_when_a_tracked_setting_changes():
    from src.models.model_version import model_version

    class _Cfg:
        def __init__(self, blend):
            self._v = {"models.bookmaker_blend_weight": blend}

        def get(self, key, default=None):
            return self._v.get(key, default)

    a = model_version(_Cfg(0.80))
    b = model_version(_Cfg(0.90))
    assert a != b, "changing the blend weight left the model version unchanged"
    assert a == model_version(_Cfg(0.80)), "version is not deterministic"
    assert a.startswith("stage5_baseline_")


def test_model_version_ignores_untracked_noise():
    """League lists and tokens churn constantly; they must not make every
    prediction look like a new model."""
    from src.models.model_version import model_version

    class _Cfg:
        def __init__(self, leagues):
            self._v = {"models.bookmaker_blend_weight": 0.80,
                       "scraping.flashscore_leagues": leagues}

        def get(self, key, default=None):
            return self._v.get(key, default)

    assert model_version(_Cfg(["a"])) == model_version(_Cfg(["a", "b", "c"]))


def test_model_version_never_raises():
    from src.models.model_version import model_version

    class _Broken:
        def get(self, key, default=None):
            raise RuntimeError("config exploded")

    assert model_version(_Broken()) == "unknown"


# ═══════════════════════ Safety — scripts must not mutate env at import

def test_scripts_do_not_load_dotenv_at_import():
    """REGRESSION. conftest pops DATABASE_URL so no test can reach production.
    An import-time load_dotenv() puts it straight back, and a SQLite unit test
    then writes to the live database — which happened while building this file
    (caught only by an IntegrityError on a colliding primary key).

    Environment mutation is a side effect and belongs in an entry point.
    """
    import ast
    import pathlib

    for name in ("capture_closing_lines", "run_baseline", "run_clean_baseline"):
        path = pathlib.Path("scripts") / f"{name}.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        # Only statements that EXECUTE at import. A load_dotenv() inside a
        # function body is fine — that is exactly the fix — so descend into
        # module-level statements but stop at any def/class boundary.
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)):
                continue
            for call in ast.walk(node):
                if (isinstance(call, ast.Call)
                        and isinstance(call.func, ast.Name)
                        and call.func.id == "load_dotenv"):
                    raise AssertionError(
                        f"{path} calls load_dotenv() at import time — this "
                        f"re-introduces DATABASE_URL and lets tests reach "
                        f"production. Move it into main().")


def test_importing_capture_script_does_not_set_database_url(monkeypatch):
    import importlib
    import os
    import sys

    monkeypatch.delenv("DATABASE_URL", raising=False)
    sys.modules.pop("scripts.capture_closing_lines", None)
    importlib.import_module("scripts.capture_closing_lines")
    assert "DATABASE_URL" not in os.environ, (
        "importing the capture script set DATABASE_URL — tests would hit production")
