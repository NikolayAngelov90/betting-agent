"""A zero-pick day is a STATE, not a failure. Stage 32 Part A, 2026-11-13.

`confident_value` has zero-pick days by design -- 5 of 56 holdout days at
P_min 0.54. Three things had to change before it could be switched on, and
each is DRIVEN here, with a known-wrong control beside it:

  1. THE LOSS ALERT counted every day with a card and no pick as a loss. It
     now has three states -- PICKS MADE, NO FIXTURE QUALIFIED, PIPELINE
     FAILED -- printed differently, and only the third is a loss. The burden
     of proof is on "nothing qualified": the run must have RECORDED that it
     evaluated the whole slate without an error. Silence stays a loss.

  2. THE CLIENT hears one plain line on a day the rule produced no pick --
     and nothing on a day the pipeline broke, so a failure is never
     announced as a quiet day.

  3. THE NEW MODE MAKES NO FORCED PICKS and does NOT apply
     `excluded_markets`. Both are driven through the method `analyze_fixture`
     calls, against the standard mode doing the opposite.
"""
import ast
import asyncio
import datetime as dt
import pathlib
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
import yaml

from scripts import pipeline_loss_alert as loss
from src.data import pick_runs as pr

D = dt.date(2026, 11, 13)


# ════════════════════════ 1. the record a run leaves ════════════════════════

@pytest.mark.parametrize("stage, picks, expect", [
    ({"stage": "EVALUATED", "fixtures": 12, "errors": 0}, 3, pr.PICKS_MADE),
    ({"stage": "EVALUATED", "fixtures": 12, "errors": 0}, 0, pr.NONE_QUALIFIED),
    ({"stage": "EVALUATED", "fixtures": 12, "errors": 1}, 0, pr.INCOMPLETE),
    ({"stage": "EVALUATED", "fixtures": None, "errors": 0}, 0, pr.INCOMPLETE),
    ({"stage": "EVALUATED", "fixtures": 0, "errors": 0}, 0, pr.NO_FIXTURES),
    ({"stage": "NO_FIXTURES", "fixtures": 0}, 0, pr.NO_FIXTURES),
    ({"stage": "NO_PRICED_FIXTURES", "fixtures": 0}, 0, pr.NO_PRICED_FIXTURES),
    ({"stage": "ALREADY_GENERATED"}, 0, None),
    ({"stage": "SHARD", "fixtures": 5, "errors": 0}, 0, None),
    (None, 0, pr.INCOMPLETE),
    ({"stage": "SOMETHING_NEW"}, 0, pr.INCOMPLETE),
    (None, 2, pr.PICKS_MADE),
])
def test_the_OUTCOME_a_run_records(stage, picks, expect):
    """"None qualified" needs a completed, error-free evaluation of a
    non-empty slate. Anything short of that -- including no stage at all --
    is INCOMPLETE, never NONE_QUALIFIED by default."""
    assert pr.classify(stage, picks) == expect


def _mgr(tmp_path, name="runs.db"):
    import src.data.database as db_mod
    from src.data.models import Base
    mgr = db_mod.DatabaseManager(config=type(
        "C", (), {"database": {"sqlite_path": str(tmp_path / name)}})())
    Base.metadata.create_all(mgr.engine)
    assert not mgr.is_postgres
    return mgr


def test_the_RECORD_is_written_and_a_BROKEN_db_does_not_raise(tmp_path):
    from src.data.models import PickRun
    mgr = _mgr(tmp_path)
    assert pr.record(mgr, pick_date=D, selection_mode="confident_value",
                     outcome=pr.NONE_QUALIFIED, fixtures_on_slate=14,
                     fixtures_evaluated=12, analysis_errors=0, picks_saved=0,
                     model_version="x.abc123") is True
    with mgr.get_session() as s:
        rows = [(r.pick_date, r.outcome, r.fixtures_evaluated, r.picks_saved)
                for r in s.query(PickRun).all()]
    assert rows == [(D, "NONE_QUALIFIED", 12, 0)]
    # an outcome outside the vocabulary is refused, not stored
    assert pr.record(mgr, pick_date=D, selection_mode="x", outcome="FINE",
                     fixtures_on_slate=1, fixtures_evaluated=1,
                     analysis_errors=0, picks_saved=0) is False

    class _Broken:
        def get_session(self):
            raise RuntimeError("database gone")
    assert pr.record(_Broken(), pick_date=D, selection_mode="x",
                     outcome=pr.PICKS_MADE, fixtures_on_slate=1,
                     fixtures_evaluated=1, analysis_errors=0,
                     picks_saved=1) is False


def test_the_ALERT_and_the_RECORD_agree_on_what_evaluated_and_empty_means():
    """The alert keeps a literal copy so it imports nothing from `src` at
    load. Set equality keeps the copy honest."""
    assert set(loss.EVALUATED_EMPTY) == set(pr.EVALUATED_EMPTY)
    assert set(pr.EVALUATED_EMPTY) == {pr.NONE_QUALIFIED, pr.NO_FIXTURES}


# ════════════════════ 2. the loss alert: three states, driven ═══════════════

def _day(back, card, picks, outcome=None, fe=None):
    return loss.DayLoss(D - timedelta(days=back), card, picks,
                        card_discovered=bool(card), outcome=outcome,
                        fixtures_evaluated=fe)


def test_THREE_STATES_and_each_is_PRINTED_DIFFERENTLY():
    made = _day(2, 30, 6, pr.PICKS_MADE, 30)
    none = _day(1, 12, 0, pr.NONE_QUALIFIED, 12)
    failed = _day(0, 28, 0, None)
    assert [loss.day_state(r) for r in (made, none, failed)] == [
        loss.STATE_PICKS_MADE, loss.STATE_NONE_QUALIFIED,
        loss.STATE_PIPELINE_FAILED]
    text = loss.format_day_states([made, none, failed]).split(chr(10))
    assert len(text) == 3
    assert "PICKS MADE" in text[0] and "6 pick(s)" in text[0]
    assert "NO FIXTURE QUALIFIED" in text[1] and "not a loss" in text[1]
    assert "all 12 fixture(s) evaluated" in text[1]
    assert "PIPELINE FAILED" in text[2] and "A LOSS" in text[2]
    assert len({t.split()[1] for t in text}) == 3


def test_a_NONE_QUALIFIED_day_ENDS_the_streak_and_a_FAILED_one_does_not():
    rows = [_day(3, 30, 0, None), _day(2, 25, 0, None),
            _day(1, 12, 0, pr.NONE_QUALIFIED, 12), _day(0, 28, 0, None)]
    assert [r.day for r in loss.zero_pick_streak(rows)] == [D]
    # KNOWN-WRONG: the same days with no recorded outcome are all losses --
    # which is exactly what the alert said before Stage 32
    bare = [r._replace(outcome=None) for r in rows]
    assert len(loss.zero_pick_streak(bare)) == 4


@pytest.mark.parametrize("outcome", [pr.INCOMPLETE, pr.NO_PRICED_FIXTURES,
                                     None])
def test_SILENCE_and_PARTIAL_evaluations_STAY_losses(outcome):
    """A broken odds feed looks exactly like 'no priced fixtures', and an
    evaluation that raised cannot say nothing qualified."""
    r = _day(0, 20, 0, outcome)
    assert loss.day_state(r) == loss.STATE_PIPELINE_FAILED
    assert loss.zero_pick_streak([r]) == [r]


def test_a_NO_FIXTURES_record_makes_an_EMPTY_CARD_a_state_not_a_loss():
    """Before Stage 32 a zero card read as 'never looked' and counted. The
    international window (2026-11-10..12) would have alerted for three days."""
    looked = _day(0, 0, 0, pr.NO_FIXTURES, 0)
    assert loss.day_state(looked) == loss.STATE_NONE_QUALIFIED
    assert "no fixtures to evaluate" in loss.format_day_states([looked])
    never = _day(0, 0, 0, None)
    assert loss.day_state(never) == loss.STATE_PIPELINE_FAILED


def test_the_ALERT_READS_the_recorded_outcome_from_the_DATABASE(tmp_path):
    """Driven on scratch SQLite through `load_recent(session=...)`, which
    loads no `.env` and never calls `get_db()` -- a test of this reader
    cannot reach production."""
    from src.data.models import Base, Match, SavedPick, Team
    mgr = _mgr(tmp_path, "alert.db")
    with mgr.get_session() as s:
        s.add(Team(id=1, name="A"))
        s.add(Team(id=2, name="B"))
        s.flush()
        for i, (back, n) in enumerate(((2, 3), (1, 2), (0, 2))):
            for j in range(n):
                s.add(Match(id=100 + 10 * i + j, home_team_id=1,
                            away_team_id=2, league="x/y",
                            match_date=datetime.combine(
                                D - timedelta(days=back),
                                dt.time(18, 0))))
        s.flush()
        s.add(SavedPick(match_id=100, pick_date=D - timedelta(days=2),
                        market="1X2", selection="Home Win", odds=1.8,
                        created_at=datetime.combine(D - timedelta(days=2),
                                                    dt.time(7, 0))))
        s.commit()
    pr.record(mgr, pick_date=D - timedelta(days=1),
              selection_mode="confident_value", outcome=pr.NONE_QUALIFIED,
              fixtures_on_slate=2, fixtures_evaluated=2, analysis_errors=0,
              picks_saved=0)
    with mgr.get_session() as s:
        rows, _earliest, _basis = loss.load_recent(days=2, today=D, session=s)
    states = [(r.day, loss.day_state(r)) for r in rows]
    assert states == [
        (D - timedelta(days=2), loss.STATE_PICKS_MADE),
        (D - timedelta(days=1), loss.STATE_NONE_QUALIFIED),
        (D, loss.STATE_PIPELINE_FAILED)]
    assert [r.day for r in loss.zero_pick_streak(rows)] == [D]


def test_a_database_WITHOUT_the_table_is_read_CONSERVATIVELY(tmp_path):
    """Before the first run creates `pick_runs`, every zero-pick day is a
    loss, as it always was -- not an error."""
    from src.data.models import Match, Team
    mgr = _mgr(tmp_path, "old.db")
    with mgr.engine.begin() as c:
        c.exec_driver_sql("DROP TABLE pick_runs")
    with mgr.get_session() as s:
        s.add(Team(id=1, name="A"))
        s.add(Team(id=2, name="B"))
        s.flush()
        s.add(Match(id=1, home_team_id=1, away_team_id=2, league="x/y",
                    match_date=datetime.combine(D, dt.time(18, 0))))
        s.commit()
    with mgr.get_session() as s:
        rows, _e, _b = loss.load_recent(days=0, today=D, session=s)
    assert [loss.day_state(r) for r in rows] == [loss.STATE_PIPELINE_FAILED]


# ═══════════════════════ 3. the client line on a zero-pick day ══════════════

OPERATOR_WORDS = ("pipeline", "run", "rule", "fixture", "error", "fail",
                  "model", "cohort", "card", "evaluat", "confident", "alert",
                  "operator", "paper")


@pytest.mark.parametrize("outcome", [pr.NONE_QUALIFIED, pr.NO_FIXTURES])
def test_the_CLIENT_LINE_is_ONE_plain_line(outcome):
    from src.reporting.telegram_bot import no_picks_line
    line = no_picks_line(outcome, D)
    assert line is not None and chr(10) not in line
    assert len(line) <= 90, len(line)
    assert line.startswith("No picks today (13 Nov 2026)")
    low = line.lower()
    for word in OPERATOR_WORDS:
        assert word not in low, (word, line)


def test_the_two_lines_say_DIFFERENT_things():
    from src.reporting.telegram_bot import no_picks_line
    a = no_picks_line(pr.NONE_QUALIFIED, D)
    b = no_picks_line(pr.NO_FIXTURES, D)
    assert a == "No picks today (13 Nov 2026): no match met the criteria."
    assert b == ("No picks today (13 Nov 2026): there are no matches in the "
                 "leagues we cover.")


@pytest.mark.parametrize("outcome", [pr.INCOMPLETE, pr.NO_PRICED_FIXTURES,
                                     pr.PICKS_MADE, None, "ANYTHING"])
def test_a_FAILURE_never_becomes_a_client_line(outcome):
    from src.reporting.telegram_bot import no_picks_line
    assert no_picks_line(outcome, D) is None


class _Notifier:
    """A notifier whose two destinations are observable and distinct."""

    def __new__(cls):
        from src.reporting import telegram_bot as tb

        class N(tb.TelegramNotifier):
            def __init__(self):
                self.config = None
                self.enabled = True
                self.bot_token = "t"
                self.chat_id = "CLIENT"
                self.operator_chat_id = "OPERATOR"
                self._bot = None
                self._last_send_error = ""
                self.client, self.operator = [], []

            def _get_bot(self):
                return object()

            async def _send_chunked(self, message, header="", *, to=None,
                                    report="report"):
                self.client.append((report, to, header + message))

            async def _send_message(self, text, **_k):
                self.client.append(("message", None, text))

            async def send_alert(self, text, **_k):
                self.operator.append(text)
        return N()


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def test_DRIVEN_a_NONE_QUALIFIED_day_sends_ONE_line_to_the_CLIENT(
        tmp_path, monkeypatch):
    from src.reporting import telegram_bot as tb
    monkeypatch.setattr(tb, "_NO_PICKS_NOTICE_STATE", tmp_path / "np.txt")
    n = _Notifier()
    _run(n.send_daily_picks([], no_picks_outcome=pr.NONE_QUALIFIED))
    assert len(n.client) == 1 and n.operator == []
    report, to, body = n.client[0]
    assert to == "CLIENT" and report == "no picks today"
    assert body.startswith("No picks today (") and chr(10) not in body
    # a re-run the same day does not repeat it
    _run(n.send_daily_picks([], no_picks_outcome=pr.NONE_QUALIFIED))
    assert len(n.client) == 1


def test_KNOWN_WRONG_a_BROKEN_day_sends_the_CLIENT_NOTHING(tmp_path,
                                                          monkeypatch):
    """The old operator path, unchanged: no outcome, or an outcome that is
    not 'evaluated in full', reaches the operator and never the client."""
    from src.reporting import telegram_bot as tb
    monkeypatch.setattr(tb, "_NO_PICKS_NOTICE_STATE", tmp_path / "np.txt")
    for outcome in (None, pr.INCOMPLETE, pr.NO_PRICED_FIXTURES):
        n = _Notifier()
        _run(n.send_daily_picks([], no_picks_outcome=outcome))
        assert n.client == [], (outcome, n.client)
        assert len(n.operator) == 1


def test_the_PICKS_PATH_sends_the_line_ONLY_on_an_evaluated_empty_day():
    """From the AST of `main()`: the one `send_daily_picks([], ...)` call in
    the no-picks branch sits inside `if _outcome in
    _pick_runs.EVALUATED_EMPTY`, and that branch never calls the validator
    that refuses an unknown mode."""
    src = pathlib.Path("src/agent/betting_agent.py").read_text(
        encoding="utf-8")
    tree = ast.parse(src)
    sends = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and (ast.get_source_segment(
                src, node.test) or "") == \
                "_outcome in _pick_runs.EVALUATED_EMPTY":
            for c in ast.walk(node):
                if isinstance(c, ast.Call) and getattr(
                        c.func, "attr", "") == "send_daily_picks":
                    sends.append(c)
    assert len(sends) == 1
    kw = {k.arg for k in sends[0].keywords}
    assert kw == {"no_picks_outcome"}
    branch = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                  and (ast.get_source_segment(src, n.test) or "")
                  == "not picks"
                  and any(isinstance(c, ast.Constant) and c.value ==
                          "No value picks found for today."
                          for c in ast.walk(n)))
    calls = {getattr(c.func, "attr", None) or getattr(c.func, "id", "")
             for c in ast.walk(branch) if isinstance(c, ast.Call)}
    # ANCHORED: the set is populated -- the branch is found and it does call
    # the sender -- so the disjointness below is a finding, not an empty set.
    assert "send_daily_picks" in calls and "print" in calls
    assert calls.isdisjoint({"mode_of", "rule_from_config"}), calls


# ═════════════════ 4. no forced picks, no excluded_markets — DRIVEN ══════════

T0 = datetime(2026, 11, 21, 8, 0, 0)


def _rows(book, home, draw, away, o35=None, u35=None):
    out = [(book, "1X2", "Home", home, T0), (book, "1X2", "Draw", draw, T0),
           (book, "1X2", "Away", away, T0)]
    if o35:
        out += [(book, "over_under", "Over 3.5", o35, T0),
                (book, "over_under", "Under 3.5", u35, T0)]
    return out


def _cfg(over=None):
    data = yaml.safe_load(pathlib.Path(
        "config/config.example.yaml").read_text(encoding="utf-8"))
    for dotted, value in (over or {}).items():
        node = data
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    def get(key, default=None):
        node = data
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node
    return SimpleNamespace(get=get, betting=data["betting"])


PRED = {
    "elo": {"home_win": 0.40, "draw": 0.29, "away_win": 0.31},
    "poisson": {"home_win": 0.41, "draw": 0.28, "away_win": 0.31,
                "over_3.5": 0.56, "home_xg": 1.9, "away_xg": 1.7},
}
PRED["ensemble"] = dict(PRED["poisson"], **{
    "under_3.5": 0.44, "over_2.5": 0.74, "over_1.5": 0.90})


def _agent(config, rows):
    from src.agent.betting_agent import FootballBettingAgent
    from src.betting.value_calculator import ValueBettingCalculator
    a = FootballBettingAgent.__new__(FootballBettingAgent)
    a.config = config
    a.value_calculator = ValueBettingCalculator(config)
    a.predictor = None
    a._odds_rows_with_time = lambda match_id, kickoff=None: list(rows)
    return a


def _odds_data(rows):
    return [{"market_type": m, "selection": s, "odds_value": v,
             "bookmaker": b, "opening_odds": v} for b, m, s, v, _t in rows]


def _finish(agent, rows, forced_calls):
    def _forced(*_a, **_k):
        from src.betting.value_calculator import BetRecommendation
        forced_calls.append(1)
        return BetRecommendation(
            match="A vs B", match_id=7, market="1X2", selection="Home Win",
            odds=2.30, predicted_probability=0.41, expected_value=-0.05,
            confidence=0.41, kelly_stake_percentage=0.5,
            recommended_stake=0.5, reasoning="forced", risk_level="high")
    agent._should_force_pick = lambda league, cov: True
    agent.value_calculator.find_best_bet = _forced
    return agent._recommendations_for_fixture(
        PRED, _odds_data(rows), "A vs B", {}, home_team_name="A",
        away_team_name="B", match_id=7, league="england/premier-league",
        kickoff=T0 + timedelta(hours=6), home_id=None, away_id=None,
        coverage_score=0.95, low_coverage=False)


# a 1X2 where nothing reaches p >= 0.54 at a price >= 1.70
NOTHING = _rows("Pinnacle", 2.35, 3.40, 3.10) + _rows("Bet365", 2.30, 3.35,
                                                      3.05)


def test_DRIVEN_the_NEW_MODE_makes_ZERO_FORCED_PICKS():
    """Since s5.23 the standard rule is still COMPUTED for every fixture and
    stored as a shadow -- so the forced pick IS built once, and it lands in
    the shadow, never in the recommendations the caller saves and sends."""
    calls = []
    agent = _agent(_cfg({"betting.selection_mode": "confident_value"}),
                   NOTHING)
    recs, confident = _finish(agent, NOTHING, calls)
    assert confident is True
    assert recs == [], (
        "the new mode forced a pick onto a fixture its rule declined")
    assert calls == [1], "the standard rule's forced pick is computed once"
    shadow = agent._standard_shadow[7]
    assert [r.reasoning for r in shadow] == ["forced"], (
        "the forced pick should be in the standard-rule shadow")


def test_KNOWN_WRONG_the_STANDARD_mode_DOES_force_one_on_the_same_fixture():
    calls = []
    recs, confident = _finish(_agent(_cfg({
        "betting.selection_mode": "standard"}), NOTHING), NOTHING, calls)
    assert confident is False
    assert calls == [1] and len(recs) == 1 and recs[0].reasoning == "forced"


# Over 3.5 is the best qualifying leg: fair ~0.55 at 1.75. `over_3.5` is in
# the deployed `excluded_markets`.
OVER35 = (_rows("Pinnacle", 2.35, 3.40, 3.10, o35=1.75, u35=2.18)
          + _rows("Bet365", 2.30, 3.35, 3.05, o35=1.76, u35=2.15))


def test_DRIVEN_the_NEW_MODE_does_NOT_apply_EXCLUDED_MARKETS():
    cfg = _cfg({"betting.selection_mode": "confident_value"})
    assert "over_3.5" in cfg.get("betting.excluded_markets")
    recs, _ = _finish(_agent(cfg, OVER35), OVER35, [])
    assert [r.selection for r in recs] == ["Over 3.5 Goals"]
    assert recs[0].odds >= 1.70 and recs[0].predicted_probability >= 0.54


def test_KNOWN_WRONG_the_STANDARD_rule_DOES_exclude_it():
    cfg = _cfg({"betting.selection_mode": "standard"})
    std = _agent(cfg, OVER35).value_calculator.find_value_bets(
        PRED, _odds_data(OVER35), "A vs B", {}, home_team_name="A",
        away_team_name="B", match_id=7, league="england/premier-league")
    assert all(r.selection != "Over 3.5 Goals" for r in std)


def test_ANALYZE_FIXTURE_calls_the_driven_method_and_nothing_else_selects():
    """From the AST: `analyze_fixture` reaches the selection only through
    `_recommendations_for_fixture`, so the two tests above drive what it
    runs."""
    src = pathlib.Path("src/agent/betting_agent.py").read_text(
        encoding="utf-8")
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef)
              and n.name == "analyze_fixture")
    called = {getattr(c.func, "attr", "") for c in ast.walk(fn)
              if isinstance(c, ast.Call)}
    # ANCHORED by the positive membership; then nothing else selects.
    assert "_recommendations_for_fixture" in called
    direct = {"find_value_bets", "find_best_bet", "_forced_pick",
              "_select_recommendations"}
    assert called.isdisjoint(direct), sorted(called & direct)


# ════════════ 5. the settlement message: hit rate, odds, return TOGETHER ════

def test_the_SETTLEMENT_LINE_carries_all_three_and_leaves_voids_out():
    from src.reporting.telegram_bot import settlement_summary_line
    picks = [{"result": "win", "odds": 1.80}, {"result": "win", "odds": 1.75},
             {"result": "loss", "odds": 1.72}, {"result": "void", "odds": 2.0},
             {"result": "loss", "odds": 1.90}]
    line = settlement_summary_line(picks)
    # 2 of 4; mean odds 1.7925; return (0.80 + 0.75 - 1 - 1) / 4 = -11.25%
    assert line == ("Hit rate 2/4 (50%) · average odds 1.79 · return -11.2% "
                    "per 1-unit pick")
    assert "nothing settled" in settlement_summary_line(
        [{"result": "void", "odds": 2.0}])


def test_the_SETTLEMENT_REPORT_shows_the_line_and_NOT_the_old_batch_figure():
    n = _Notifier()
    settled = [{
        "match_name": "A vs B", "match": "A vs B",
        "league": "england/premier-league", "market": "Over 2.5",
        "selection": "Over 2.5 Goals", "odds": 1.8, "result": "win",
        "score": "2-1", "stake": 0.5, "kelly_stake_percentage": 0.5,
        "expected_value": 0.0, "confidence": 0.55,
        "predicted_probability": 0.55, "model_agreement": "solo",
        "review_action": None, "is_paper": True,
        "pick_date": dt.date.today() - timedelta(days=1)}]
    _run(n.send_settlement_report(settled, stats={}))
    body = chr(10).join(m for _r, _t, m in n.client)
    assert "Hit rate 1/1 (100%) · average odds 1.80 · return +80.0%" in body
    assert "Batch:" not in body, (
        "the stake-weighted 'Batch' figure is back beside a per-unit return")
