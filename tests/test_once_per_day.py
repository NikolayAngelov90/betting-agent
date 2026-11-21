"""One card per day, DRIVEN. Stage 34, 2026-11-21.

The card is published by an external dispatch at 10:30 Europe/Sofia with a
late fallback cron behind it; either can start `daily-picks`, and GitHub
queues a second run behind the first. These tests drive the REAL code -- the
workflow's `once` gate (`scripts/once_per_day.main`), the real `--picks` CLI
path in `betting_agent.main`, and `pick_runs` written to and read from a real
SQLite database -- and count what reached the client. A stand-in agent supplies
the day's picks; everything that decides whether to send is the production
code.

THE WORKFLOW'S ORDER IS REPRODUCED, NOT ASSUMED: `daily-picks.yml`'s
concurrency group is workflow-level and never cancels, so a second run starts
only after the first has finished. Each `_workflow_run` below is one whole run:
the gate first, and the picks path only if the gate let it through.
"""
import asyncio
import datetime as dt
import pathlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from scripts import once_per_day
from src.data import pick_runs as pr


@pytest.fixture
def db(tmp_path, monkeypatch):
    from src.data.database import DatabaseManager
    monkeypatch.delenv("DATABASE_URL", raising=False)
    mgr = DatabaseManager(config=SimpleNamespace(
        database={"sqlite_path": str(tmp_path / "once.db")}))
    assert not mgr.is_postgres
    mgr.create_tables()
    return mgr


class _Client:
    """Every client message the run sent: the card, or the zero-pick line."""

    def __init__(self):
        self.messages = []

    async def send_daily_picks(self, picks, **kw):
        self.messages.append(("card" if picks else
                              f"line:{kw.get('no_picks_outcome')}", len(picks)))


def _agent(db, client, *, picks=1, fail=False, errors=0):
    a = MagicMock()
    a.db = db
    a.shutdown = AsyncMock()
    a.telegram = client
    a.telegram.enabled = True
    a.config = SimpleNamespace(get=lambda k, d=None: {
        "betting.selection_mode": "confident_value",
        "notifications.suppress_picks_summary": False}.get(k, d))
    # A REAL BetRecommendation: the send reads `match_id` (0 skips the
    # injury lookup) and the console listing after it reads the rest.
    from src.betting.value_calculator import BetRecommendation
    made = [BetRecommendation(
        match="A vs B", match_id=0, market="1X2", selection="Home Win",
        odds=1.75, predicted_probability=0.55, expected_value=-0.04,
        confidence=0.55, kelly_stake_percentage=0.0, recommended_stake=0.0,
        reasoning="r", risk_level="low", league="spain/laliga2")
        for _ in range(picks)]

    async def _get_daily_picks(**kw):
        if fail:
            raise RuntimeError("planted crash before any outcome is recorded")
        a.last_pick_run = {"stage": "EVALUATED", "slate": 5, "fixtures": 5,
                           "errors": errors}
        return made, made, []
    a.get_daily_picks = _get_daily_picks
    a.get_stats = MagicMock(return_value={})
    return a


def _picks_path(agent):
    """The real `--picks` CLI path, as the workflow's picks step runs it."""
    from src.agent.betting_agent import main
    with patch("src.agent.betting_agent.FootballBettingAgent",
               return_value=agent), \
         patch("src.agent.betting_agent._configure_cli_runtime"), \
         patch("sys.argv", ["betting_agent", "--picks", "--force"]):
        asyncio.run(main())


def _workflow_run(db, client, today, capsys=None, **agent_kw):
    """One whole `daily-picks` run: the `once` gate, then -- only if the gate
    let it through -- the picks path. Returns whether the gate stopped it."""
    out_file = pathlib.Path(db.config.database["sqlite_path"]).with_suffix(
        ".out")
    out_file.write_text("", encoding="utf-8")
    with patch.dict("os.environ", {"GITHUB_OUTPUT": str(out_file)}):
        once_per_day.main(db=db, today=today)
    already = "already_ran=true" in out_file.read_text(encoding="utf-8")
    if not already:
        try:
            _picks_path(_agent(db, client, **agent_kw))
        except RuntimeError:
            pass                     # the planted crash: the run failed
    return already


def _outcomes(db, today):
    from src.data.models import PickRun
    with db.get_session() as s:
        return [r.outcome for r in s.query(PickRun)
                .filter(PickRun.pick_date == today)
                .order_by(PickRun.recorded_at).all()]


TODAY = dt.date.today()


# ── THE FOUR CONTROLS ────────────────────────────────────────────────────────

def test_DISPATCH_then_SCHEDULE_one_card(db):
    client = _Client()
    assert _workflow_run(db, client, TODAY) is False      # the 10:30 dispatch
    assert _workflow_run(db, client, TODAY) is True       # the fallback cron
    assert client.messages == [("card", 1)], client.messages
    assert _outcomes(db, TODAY) == [pr.PICKS_MADE, pr.ALREADY_RAN_TODAY]


def test_SCHEDULE_only_one_card(db):
    """A missed dispatch: the fallback alone produces the day's card."""
    client = _Client()
    assert _workflow_run(db, client, TODAY) is False
    assert client.messages == [("card", 1)]
    assert _outcomes(db, TODAY) == [pr.PICKS_MADE]


def test_TWO_DISPATCHES_one_card(db):
    client = _Client()
    _workflow_run(db, client, TODAY)
    _workflow_run(db, client, TODAY)
    _workflow_run(db, client, TODAY)                      # and a third start
    assert client.messages == [("card", 1)], client.messages
    assert _outcomes(db, TODAY).count(pr.ALREADY_RAN_TODAY) == 2


@pytest.mark.parametrize("first", ["crash", "incomplete"])
def test_a_FAILED_first_run_leaves_the_day_open_and_the_second_SENDS(db,
                                                                     first):
    """A crash before any outcome is written, and a run that recorded
    INCOMPLETE (errors, no pick): neither completes the day."""
    client = _Client()
    if first == "crash":
        _workflow_run(db, client, TODAY, fail=True)
    else:
        _workflow_run(db, client, TODAY, picks=0, errors=2)
    assert client.messages == [], "the failed run reached the client"
    assert _workflow_run(db, client, TODAY) is False, (
        "the gate closed a day that no run completed")
    assert client.messages == [("card", 1)]


# ── AND THE CASES AROUND THEM ────────────────────────────────────────────────

def test_a_ZERO_PICK_day_is_COMPLETED_too_and_the_line_is_not_sent_twice(db):
    client = _Client()
    _workflow_run(db, client, TODAY, picks=0)
    _workflow_run(db, client, TODAY, picks=0)
    assert client.messages == [("line:NONE_QUALIFIED", 0)], client.messages
    assert _outcomes(db, TODAY) == [pr.NONE_QUALIFIED, pr.ALREADY_RAN_TODAY]


def test_a_GATE_that_FAILS_OPEN_is_caught_by_the_IN_RUN_guard(db, capsys):
    """The `once` job errors (fail open); the picks step still refuses."""
    client = _Client()
    _workflow_run(db, client, TODAY)
    _picks_path(_agent(db, client))                       # gate bypassed
    assert client.messages == [("card", 1)], client.messages
    assert "ALREADY RAN TODAY" in capsys.readouterr().out
    assert _outcomes(db, TODAY)[-1] == pr.ALREADY_RAN_TODAY


def test_YESTERDAY_does_not_close_TODAY(db):
    """A completed row for yesterday leaves today open."""
    client = _Client()
    pr.record(db, pick_date=TODAY - dt.timedelta(days=1),
              selection_mode="confident_value", outcome=pr.PICKS_MADE,
              fixtures_on_slate=5, fixtures_evaluated=5, analysis_errors=0,
              picks_saved=3)
    assert _workflow_run(db, client, TODAY) is False
    assert client.messages == [("card", 1)]


def test_the_LOSS_ALERTS_literal_equals_the_modules():
    from scripts import pipeline_loss_alert as pla
    assert pla.NOT_A_DAY_OUTCOME == pr.NOT_A_DAY_OUTCOME

def test_the_gate_FAILS_OPEN_on_an_unreadable_database(tmp_path, capsys):
    broken = SimpleNamespace(get_session=lambda: (_ for _ in ()).throw(
        RuntimeError("database down")))
    out = tmp_path / "out"
    with patch.dict("os.environ", {"GITHUB_OUTPUT": str(out)}):
        rc = once_per_day.main(db=broken, today=TODAY)
    written = out.read_text(encoding="utf-8")
    assert "already_ran=false" in written, written
    assert rc == 0 and "FAILING OPEN" in capsys.readouterr().out


def test_ALREADY_RAN_never_COMPLETES_a_day():
    assert pr.ALREADY_RAN_TODAY not in pr.COMPLETED
    assert pr.ALREADY_RAN_TODAY in pr.NOT_A_DAY_OUTCOME


def test_ALREADY_RAN_is_never_the_DAYS_outcome_for_any_reader():
    """The three readers that take the latest row per day as the day's
    outcome skip it -- otherwise a no-op run would overwrite the answer."""
    for path in ("scripts/pipeline_loss_alert.py", "scripts/first_run_check.py",
                 "scripts/confident_backtest.py"):
        src = pathlib.Path(path).read_text(encoding="utf-8")
        assert "ALREADY_RAN_TODAY" in src, f"{path} does not skip it"


# ── THE WORKFLOW CARRIES IT ──────────────────────────────────────────────────

def test_the_WORKFLOW_skips_the_whole_job_on_ALREADY_RAN_and_fails_open():
    wf = yaml.safe_load(pathlib.Path(".github/workflows/daily-picks.yml")
                        .read_text(encoding="utf-8"))
    main_job = wf["jobs"]["daily-picks"]
    assert list(wf["jobs"])[0] == "daily-picks", "the main job moved"
    assert main_job["needs"] == "once"
    cond = main_job["if"]
    assert "needs.once.outputs.already_ran != 'true'" in cond
    assert "!cancelled()" in cond, (
        "without it a FAILED gate would skip the job and lose the card")
    once = wf["jobs"]["once"]
    assert once["outputs"]["already_ran"] == "${{ steps.once.outputs.already_ran }}"
    assert any("scripts.once_per_day" in (s.get("run") or "")
               for s in once["steps"])
    # runs never overlap: workflow-level group, never cancelled
    assert wf["concurrency"] == {"group": "daily-picks",
                                 "cancel-in-progress": False}
