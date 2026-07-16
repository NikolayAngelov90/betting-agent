"""Tests for the 'predictions only' Claude pick-review path.

Claude reviews every saved pick and may CHANGE the selection; the in-memory pick
objects must be synced from the DB so the Telegram summary reflects the switch.
No briefing article is posted in this mode.
"""

import asyncio
import re
from datetime import date, datetime
from types import SimpleNamespace

import pytest

from src.data.database import DatabaseManager
from src.data.models import Team, Match, SavedPick
from src.reporting.match_briefing import MatchBriefingService
from src.utils.config import get_config

_CYRILLIC = re.compile(r"[Ѐ-ӿ]")


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    mgr = DatabaseManager(config=SimpleNamespace(
        database={"sqlite_path": str(tmp_path / "review_test.db")}
    ))
    assert not mgr.is_postgres, "test DB must be SQLite, not production Postgres"
    mgr.create_tables()
    return mgr


def _rec(match_id, **kw):
    base = dict(match_id=match_id, market="over_under", selection="Over 2.5 Goals",
                odds=2.10, predicted_probability=0.51, expected_value=0.05,
                confidence=0.51, kelly_stake_percentage=1.5, risk_level="medium")
    base.update(kw)
    return SimpleNamespace(**base)


def _svc(db):
    return MatchBriefingService(SimpleNamespace(
        db=db, config=get_config(), telegram=None, agent=None))


def test_sync_recs_reflects_switched_pick(db):
    """After Claude switches the saved pick, the in-memory rec must show it."""
    with db.get_session() as s:
        h, a = Team(name="Djurgarden"), Team(name="Hacken")
        s.add_all([h, a]); s.flush()
        m = Match(home_team_id=h.id, away_team_id=a.id,
                  match_date=datetime(2026, 7, 6, 18, 0), league="sweden/allsvenskan",
                  is_fixture=True)
        s.add(m); s.flush()
        # DB pick is "Home Over 1.5" @1.55 (what Claude switched it to).
        s.add(SavedPick(match_id=m.id, pick_date=date.today(),
                        match_name="Djurgarden vs Hacken", league="sweden/allsvenskan",
                        market="team_goals", selection="Home Over 1.5", odds=1.55,
                        predicted_probability=0.6, expected_value=0.08, confidence=0.6,
                        kelly_stake_percentage=2.0, risk_level="low"))
        s.commit()
        mid = m.id

    # rec still holds the pre-switch selection.
    rec = _rec(mid, market="over_under", selection="Over 2.5 Goals", odds=2.10)
    _svc(db)._sync_recs_from_db(mid, [rec])

    assert rec.selection == "Home Over 1.5"
    assert rec.odds == 1.55
    assert rec.market == "team_goals"
    assert rec.expected_value == 0.08


def test_sync_recs_no_saved_pick_leaves_rec_untouched(db):
    rec = _rec(999, selection="Over 2.5 Goals")
    _svc(db)._sync_recs_from_db(999, [rec])
    assert rec.selection == "Over 2.5 Goals"  # unchanged when no DB pick exists


def test_finalize_noop_without_claude_auth(db, monkeypatch):
    """With no Claude auth, review must no-op and leave picks untouched."""
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    rec = _rec(1, selection="Over 2.5 Goals")
    reviewed = asyncio.run(_svc(db).finalize_picks_with_claude([rec]))
    assert reviewed == 0
    assert rec.selection == "Over 2.5 Goals"


def test_finalize_empty_picks(db):
    assert asyncio.run(_svc(db).finalize_picks_with_claude([])) == 0


def _decision_svc():
    """Service with no DB/agent needed — _research_and_write only uses config."""
    return MatchBriefingService(SimpleNamespace(
        db=None, config=get_config(), telegram=None, agent=None))


def test_decision_only_returns_no_prose_and_english_prompt(monkeypatch):
    """Review path: Claude returns ONLY a decision, prompt is English (no bg)."""
    svc = _decision_svc()
    captured = {}

    async def fake_api(system, user, match_name):
        captured["system"] = system
        captured["user"] = user
        return ("<<<DECISION>>>\naction: CHANGE\nmarket_key: home_over_1.5\n"
                "confidence: 0.62\nreason: strong favourite\n<<<END>>>")

    monkeypatch.setattr(svc, "_resolve_backend", lambda: "anthropic_api")
    monkeypatch.setattr(svc, "_call_anthropic_api", fake_api)

    menu = [{"market_key": "home_over_1.5", "selection": "Home Over 1.5",
             "odds": 1.55, "probability": 0.6}]
    briefing, decision = asyncio.run(svc._research_and_write(
        "USA vs Bosnia", "model dossier here", lineup_aware=False,
        has_lineups=False, menu=menu, decision_only=True))

    # No briefing prose is returned or generated.
    assert briefing == ""
    assert decision["action"] == "CHANGE"
    assert decision["market_key"] == "home_over_1.5"
    # The decision-only prompt is English and forbids prose — no Bulgarian.
    assert "ONLY the decision block" in captured["system"]
    assert "Do NOT write an article" in captured["system"]
    assert not _CYRILLIC.search(captured["system"])
    assert not _CYRILLIC.search(captured["user"])
    # De-anchoring: the menu shows the market-implied probability alongside the
    # model's (1.55 odds -> 65%), and the prompt warns the model is overconfident
    # and prefers shorter odds on close calls.
    assert "market 65%" in captured["user"]
    assert "OVERCONFIDENT" in captured["user"]
    assert "SHORTER odds" in captured["user"]
    # Switch-more guidance (KEEP 53% vs CHANGE 92% in tracked results): Claude
    # is told not to defer to the saved pick out of caution.
    assert "SWITCH decisions have materially outperformed" in captured["user"]


def test_decision_only_no_menu_skips_api_call(monkeypatch):
    """No priced selections ⇒ no decision, and the API is never called."""
    svc = _decision_svc()
    called = {"api": False}

    async def fake_api(system, user, match_name):
        called["api"] = True
        return "x"

    monkeypatch.setattr(svc, "_resolve_backend", lambda: "anthropic_api")
    monkeypatch.setattr(svc, "_call_anthropic_api", fake_api)

    briefing, decision = asyncio.run(svc._research_and_write(
        "USA vs Bosnia", "dossier", lineup_aware=False, has_lineups=False,
        menu=[], decision_only=True))
    assert briefing == "" and decision is None
    assert called["api"] is False


def test_apply_decision_records_keep_on_saved_pick(db):
    """The review outcome must be persisted so KEEP-vs-CHANGE win rates are
    measurable — a KEEP ruling stamps review_action/review_reason on the pick."""
    with db.get_session() as s:
        h, a = Team(name="Argentina"), Team(name="Egypt")
        s.add_all([h, a]); s.flush()
        m = Match(home_team_id=h.id, away_team_id=a.id,
                  match_date=datetime(2026, 7, 8, 19, 0),
                  league="world/fifa-world-cup", is_fixture=True)
        s.add(m); s.flush()
        s.add(SavedPick(match_id=m.id, pick_date=date.today(),
                        match_name="Argentina vs Egypt", league="world/fifa-world-cup",
                        market="over_under", selection="Over 2.5 Goals", odds=1.70,
                        predicted_probability=0.6, expected_value=0.02, confidence=0.6,
                        kelly_stake_percentage=1.0, risk_level="low"))
        s.commit()
        mid = m.id

    svc = _svc(db)
    bound = svc._apply_decision(
        mid, {"action": "KEEP", "reason": "market and research agree"},
        None, [], "", "")
    assert bound is True
    with db.get_session() as s:
        row = s.query(SavedPick).filter(SavedPick.match_id == mid).one()
        assert row.review_action == "KEEP"
        assert row.review_reason == "market and research agree"
        assert row.selection == "Over 2.5 Goals"  # unchanged by KEEP


def test_recent_review_stats_formats_keep_change(db):
    """KEEP/CHANGE win rates are computed from settled reviewed picks."""
    with db.get_session() as s:
        h, a = Team(name="A"), Team(name="B")
        s.add_all([h, a]); s.flush()
        m = Match(home_team_id=h.id, away_team_id=a.id,
                  match_date=datetime(2026, 7, 10, 18, 0),
                  league="world/fifa-world-cup", is_fixture=False,
                  home_goals=1, away_goals=0)
        s.add(m); s.flush()
        # 7 KEEP (3 wins), 5 CHANGE (5 wins) — 12 settled reviewed picks.
        for i in range(7):
            s.add(SavedPick(match_id=m.id, pick_date=date.today(),
                            match_name="A vs B", league="world/fifa-world-cup",
                            market="1X2", selection="Home Win", odds=1.7,
                            review_action="KEEP",
                            result="win" if i < 3 else "loss"))
        for i in range(5):
            s.add(SavedPick(match_id=m.id, pick_date=date.today(),
                            match_name="A vs B", league="world/fifa-world-cup",
                            market="1X2", selection="Home Win", odds=1.7,
                            review_action="CHANGE", result="win"))
        s.commit()

    out = _svc(db)._recent_review_stats()
    assert "CHANGE: 5/5 won (100%)" in out
    assert "KEEP: 3/7 won (43%)" in out


def test_recent_review_stats_empty_below_minimum(db):
    """Fewer than 10 settled reviewed picks -> no stats line (too noisy)."""
    assert _svc(db)._recent_review_stats() == ""
