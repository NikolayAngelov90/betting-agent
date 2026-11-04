"""The client chat is reachable from TWO senders, by construction.

THE FIRST FIX MISSED FIVE PATHS BECAUSE ITS CENSUS LOOKED AT `src/`.
`scripts/ci_alert.py` calls `deliver_alert(text)` with no destination, and
`deliver_alert` fell back to `TELEGRAM_CHAT_ID` — the CLIENT chat. Five
callers reach it:

    ci-audit.yml                    CI audit findings
    reminders.yml                   reminder verdicts and failures
    daily-picks.yml                 the no-picks loss alert
    fixture-horizon-probe.yml       the probe message
    scripts/paper_trading_report.py --telegram  (via ci_alert.send)

    A CENSUS'S POPULATION IS THE REPOSITORY, NOT A DIRECTORY.

THE SEAL IS STRUCTURAL, NOT AN INSTRUCTION. `deliver_alert` now defaults to
`TELEGRAM_OPERATOR_CHAT_ID`, and `_send_message` / `_send_one` /
`_send_chunked` take the destination as an argument whose default is the
operator id. The client id is obtainable only where it is passed explicitly,
so a sixth caller added tomorrow cannot reach it by omission.

PART C IS DRIVEN, NOT SAMPLED. The previous measurement said 1.95 msgs/day
before and after with alerts firing zero times — which cannot distinguish "no
longer sends" from "did not send during the sample". Each drive below runs the
real path with a recording transport and reads the destination it addressed.
"""

import asyncio
import datetime as dt
import glob
import os

import pytest
import yaml

CLIENT = "CLIENT-CHAT-9999"
OPERATOR = "OPERATOR-CHAT-1111"


# ═══════════════════ A. the destination each path resolves ══════════════════

@pytest.fixture
def recorder(monkeypatch):
    """Records every (destination, text) the delivery layer is handed."""
    sent = []

    def _fake_post(url, data=None, timeout=None, **_k):
        sent.append((str((data or {}).get("chat_id")), (data or {}).get("text")))

        class _R:
            status_code = 200

            @staticmethod
            def json():
                return {"ok": True, "result": {"message_id": 1}}
        return _R()

    import src.reporting.alert_delivery as ad
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", CLIENT)
    monkeypatch.setenv("TELEGRAM_OPERATOR_CHAT_ID", OPERATOR)
    for name in ("requests", "urllib"):
        if hasattr(ad, name):
            pass
    monkeypatch.setattr(ad, "_post", _fake_post, raising=False)
    return sent


def test_DELIVER_ALERT_DEFAULTS_TO_THE_OPERATOR_not_the_client(monkeypatch):
    """The one line that was the leak, read directly."""
    import inspect

    import ast
    import textwrap

    import src.reporting.alert_delivery as ad

    # THE AST, NOT RAW TEXT AND NOT `code_of`.
    #
    # Raw text is wrong: the comment beside the fixed line QUOTES
    # `TELEGRAM_CHAT_ID` to explain what the leak was, so a negative
    # assertion would match the explanation rather than the code.
    #
    # `code_of` would strip that comment, but importing `tests.conftest`
    # drags this file into the CAPTURE GATE -- a test about which chat a
    # message reaches would then be able to stop price capture. That has
    # happened twice before.
    #
    # Reading the environment lookups out of the parse tree is exact, needs
    # no helper, and cannot match a comment at all.
    tree = ast.parse(textwrap.dedent(inspect.getsource(ad.deliver_alert)))
    env_keys = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and isinstance(node.func.value, ast.Attribute)
                and node.func.value.attr == "environ"
                and node.args
                and isinstance(node.args[0], ast.Constant)):
            env_keys.add(node.args[0].value)
    assert "TELEGRAM_OPERATOR_CHAT_ID" in env_keys, (
        f"deliver_alert reads {sorted(env_keys)} and no longer defaults to "
        f"the operator chat; every ci_alert caller would reach the client "
        f"chat again")
    assert "TELEGRAM_CHAT_ID" not in env_keys, (
        f"deliver_alert can still resolve the CLIENT chat id from the "
        f"environment ({sorted(env_keys)}). That is the leak, not a "
        f"fallback.")


def test_THE_CLIENT_ID_IS_READ_BY_EXACTLY_THE_TWO_CLIENT_SENDERS():
    """`self.chat_id` is passed from the card, its resend, and the settlement.

    Read from the AST so a new sender cannot quietly join them.
    """
    import ast
    import io
    src = io.open("src/reporting/telegram_bot.py", encoding="utf-8").read()
    tree = ast.parse(src)
    holders = set()
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(fn):
            if (isinstance(node, ast.Attribute) and node.attr == "chat_id"
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "self"):
                holders.add(fn.name)
    allowed = {"__init__",            # defines it
               "send_daily_picks",    # the card
               "resend_daily_picks",  # the same card again
               "send_settlement_report",
               "_send_message"}       # migration bookkeeping only
    assert holders <= allowed, (
        f"these functions read the CLIENT chat id and must not: "
        f"{sorted(holders - allowed)}")
    for required in ("send_daily_picks", "send_settlement_report"):
        assert required in holders, (
            f"{required} no longer names the client chat, so the card or the "
            f"settlement is going somewhere else")


def test_KNOWN_WRONG_a_SEND_FROM_AN_ALERT_PATH_IS_CAUGHT():
    """Prove the test above can fail.

    Simulates reintroducing `to=self.chat_id` inside `send_alert` and
    confirms the AST check names it.
    """
    import ast
    src = (
        "class N:\n"
        "    def __init__(self):\n"
        "        self.chat_id = ''\n"
        "    async def send_alert(self, t):\n"
        "        await self._send_chunked(t, to=self.chat_id)\n")
    tree = ast.parse(src)
    holders = set()
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(fn):
            if (isinstance(node, ast.Attribute) and node.attr == "chat_id"
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "self"):
                holders.add(fn.name)
    allowed = {"__init__", "send_daily_picks", "resend_daily_picks",
               "send_settlement_report", "_send_message"}
    assert "send_alert" in holders - allowed, (
        "the AST check would NOT notice an alert path taking the client id, "
        "so the test above proves nothing")


# ═══════════════════ the workflow env, asserted from YAML ═══════════════════

def test_THE_CLIENT_ID_IS_IN_ONE_WORKFLOW_ENV_ONLY():
    """A workflow step that never sends a card must not hold the client id."""
    holders = {}
    for f in sorted(glob.glob(".github/workflows/*.yml")):
        raw = open(f, encoding="utf-8").read()
        for i, line in enumerate(raw.split("\n"), 1):
            st = line.strip()
            if st.startswith("TELEGRAM_CHAT_ID:"):
                holders.setdefault(os.path.basename(f), []).append(i)
    assert set(holders) <= {"daily-picks.yml"}, (
        f"the CLIENT chat id is in the env of workflows that do not send a "
        f"card: {holders}. Those steps call `scripts.ci_alert`, which is how "
        f"ci-audit and reminders reached the client chat.")
    assert "daily-picks.yml" in holders, (
        "no workflow holds the client id, so the card cannot be sent at all")
    assert len(holders["daily-picks.yml"]) == 1, (
        f"the client id appears {len(holders['daily-picks.yml'])} times in "
        f"daily-picks.yml; it belongs at the job level only, where the picks "
        f"and settlement steps run")


def test_EVERY_ci_alert_STEP_CARRIES_THE_OPERATOR_ID():
    """The five alert paths must have a destination, or they send nothing."""
    found = 0
    for f in sorted(glob.glob(".github/workflows/*.yml")):
        raw = open(f, encoding="utf-8").read()
        if "scripts.ci_alert" not in raw:
            continue
        found += 1
        assert "TELEGRAM_OPERATOR_CHAT_ID" in raw, (
            f"{os.path.basename(f)} invokes scripts.ci_alert but carries no "
            f"operator chat id, so its alerts go nowhere silently")
    assert found >= 3, (
        f"only {found} workflow(s) reference scripts.ci_alert; the census "
        f"found four, so one has been renamed and may be unguarded")


# ═══════════════════ C. FIVE DRIVES ════════════════════════════════════════

from src.reporting.telegram_bot import TelegramNotifier


class _N(TelegramNotifier):
    """A notifier whose transport is recorded, with both ids configured.

    A REAL SUBCLASS, declared here. The first version built the class with
    `type(...)` and reassigned `__class__`, which resolved `_send_chunked` to
    the BASE implementation and attempted a real send -- the overrides were
    never reached and both client drives reported nothing delivered.
    """

    def __init__(self):
        self.config = None
        self.enabled = True
        self.bot_token = "tok"
        self.chat_id = CLIENT
        self.operator_chat_id = OPERATOR
        self._bot = None
        self._last_send_error = ""
        self.seen = []

    def _get_bot(self):
        return object()

    async def _send_chunked(self, message, header="", *, to=None, report=""):
        self.seen.append((to, report, message))
        return type("M", (), {"message_id": 1})()

    async def _send_message(self, text, to=None):
        self.seen.append((to, "message", text))
        return type("M", (), {"message_id": 1})()


def _run(c):
    return asyncio.new_event_loop().run_until_complete(c)


def _pick():
    from src.betting.value_calculator import BetRecommendation
    return BetRecommendation(
        match="A vs B", match_id=1, market="Over 2.5",
        selection="Over 2.5 Goals", odds=1.8, predicted_probability=0.6,
        expected_value=0.08, confidence=0.65, kelly_stake_percentage=1.5,
        recommended_stake=1.5, reasoning="", risk_level="low",
        league="england/premier-league", model_agreement="solo",
        match_date=dt.datetime(2026, 11, 4, 20, 0))


def _client_msgs(n):
    return [x for x in n.seen if x[0] == CLIENT]


def test_DRIVE_1_a_CI_AUDIT_FAILURE_reaches_the_operator_only(monkeypatch):
    """The path that was actually leaking."""
    seen = {}

    def _fake(text, *, token=None, chat_id=None, sleep=None):
        from src.reporting.alert_delivery import DeliveryResult
        seen["chat_id"] = chat_id
        return DeliveryResult(True, 1, "", False)

    import src.reporting.alert_delivery as ad
    monkeypatch.setattr(ad, "deliver_alert", _fake)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", CLIENT)
    monkeypatch.setenv("TELEGRAM_OPERATOR_CHAT_ID", OPERATOR)
    import importlib
    ca = importlib.import_module("scripts.ci_alert")
    importlib.reload(ca)
    ca.send("(!) ci-audit found 3 NEW findings")
    assert seen.get("chat_id") != CLIENT, (
        f"a ci-audit finding addressed {seen.get('chat_id')!r} — the client "
        f"chat. This is the exact message Niki was still receiving.")


@pytest.mark.parametrize("kind,body", [
    ("VERDICT", "REMINDER DUE: the H1 acknowledgement expired"),
    ("FAILURE", "reminders step FAILED — pytest exit 1"),
])
def test_DRIVE_2_reminders_reach_the_operator_only(monkeypatch, kind, body):
    """Both halves: a dated verdict and a real failure."""
    seen = {}

    def _fake(text, *, token=None, chat_id=None, sleep=None):
        from src.reporting.alert_delivery import DeliveryResult
        seen["chat_id"] = chat_id
        return DeliveryResult(True, 1, "", False)

    import src.reporting.alert_delivery as ad
    monkeypatch.setattr(ad, "deliver_alert", _fake)
    import importlib
    ca = importlib.reload(importlib.import_module("scripts.ci_alert"))
    ca.send(body)
    assert seen.get("chat_id") != CLIENT, f"{kind} reached the client chat"


def test_DRIVE_3_a_CRASH_WITH_NO_FINDING_reaches_the_operator_only(monkeypatch):
    seen = {}

    def _fake(text, *, token=None, chat_id=None, sleep=None):
        from src.reporting.alert_delivery import DeliveryResult
        seen["chat_id"] = chat_id
        return DeliveryResult(True, 1, "", False)

    import src.reporting.alert_delivery as ad
    monkeypatch.setattr(ad, "deliver_alert", _fake)
    import importlib
    ca = importlib.reload(importlib.import_module("scripts.ci_alert"))
    ca.send("step(s) FAILED — update; no finding produced")
    assert seen.get("chat_id") != CLIENT


def test_DRIVE_4_a_SUCCESSFUL_PICKS_RUN_reaches_the_client(monkeypatch):
    from src.reporting import telegram_bot as tb
    monkeypatch.setattr(tb, "_picks_sent_today", lambda: False)
    monkeypatch.setattr(tb, "_mark_picks_sent_today", lambda: None)
    n = _N()
    _run(n.send_daily_picks([_pick()], force=True, paper_mode=True))
    msgs = _client_msgs(n)
    assert len(msgs) == 1, f"the card did not reach the client chat: {n.seen}"
    assert "A vs B" in msgs[0][2]


def test_DRIVE_5_a_SETTLEMENT_reaches_the_client_WITH_its_population():
    from src.reporting.experiment_record import ExperimentRecord
    import inspect
    sig = inspect.signature(ExperimentRecord)
    kw = {k: v for k, v in (
        ("wins", 30), ("losses", 20), ("settled", 50), ("cohorts", 3),
        ("win_rate", 0.60), ("flat_roi", -0.055), ("avg_odds", 1.70),
        ("pl_units", -2.8), ("first_date", "2026-08-11"),
        ("last_date", "2026-11-03")) if k in sig.parameters}
    rec = ExperimentRecord(**kw)
    n = _N()
    settled = [{
        "match_name": "A vs B", "match": "A vs B",
        "league": "england/premier-league", "market": "Over 2.5",
        "selection": "Over 2.5 Goals", "odds": 1.8, "result": "win",
        "score": "2-1", "stake": 1.5, "kelly_stake_percentage": 1.5,
        "expected_value": 0.08, "confidence": 0.65,
        "predicted_probability": 0.6, "model_agreement": "solo",
        "review_action": "KEEP", "is_paper": True,
        "pick_date": dt.date(2026, 11, 3)}]
    _run(n.send_settlement_report(settled, stats={}, experiment=rec))
    msgs = _client_msgs(n)
    assert msgs, f"the settlement did not reach the client chat: {n.seen}"
    body = msgs[0][2]
    # THE PERFORMANCE FIGURE RIDES INSIDE THE SETTLEMENT, IN CLIENT LANGUAGE.
    # This test first asserted "PAPER", "3 cohorts" and "avg odds" -- the
    # OPERATOR wording, which is exactly what was removed on 2026-11-04. The
    # population is still named, in words a client can read.
    assert "no real money has been staked" in body.lower(), (
        "the settlement does not say money is not at stake")
    assert "11 Aug 2026" in body, "no start date in client form"
    assert "50 settled picks" in body, "no settled count"
    assert "60.0%" in body and "level stakes" in body and "average price" in body, (
        "win rate appears without its return and its price")
    assert "cohort" not in body.lower(), (
        "the cohort count reached the client settlement")


# ══════════ A. the settlement carries the figure, in CLIENT language ════════

def _rec():
    from src.reporting.experiment_record import ExperimentRecord
    import inspect
    sig = inspect.signature(ExperimentRecord)
    kw = {k: v for k, v in (
        ("wins", 30), ("losses", 20), ("settled", 50), ("cohorts", 3),
        ("win_rate", 0.60), ("flat_roi", -0.055), ("avg_odds", 1.70),
        ("pl_units", -2.8), ("first_date", "2026-08-11"),
        ("last_date", "2026-11-03")) if k in sig.parameters}
    return ExperimentRecord(**kw)


def _settlement_body():
    n = _N()
    settled = [{
        "match_name": "A vs B", "match": "A vs B",
        "league": "england/premier-league", "market": "Over 2.5",
        "selection": "Over 2.5 Goals", "odds": 1.8, "result": "win",
        "score": "2-1", "stake": 1.5, "kelly_stake_percentage": 1.5,
        "expected_value": 0.08, "confidence": 0.65,
        "predicted_probability": 0.6, "model_agreement": "solo",
        "review_action": "KEEP", "is_paper": True,
        "pick_date": dt.date(2026, 11, 3)}]
    _run(n.send_settlement_report(settled, stats={}, experiment=_rec()))
    msgs = [x for x in n.seen if x[0] == CLIENT]
    assert msgs, f"the settlement did not reach the client chat: {n.seen}"
    return msgs[0][2]


def test_THE_SETTLEMENT_CARRIES_THE_OVERALL_FIGURE_AND_THE_PERIOD():
    """One message, both things: the day's results and the running total."""
    body = _settlement_body()
    assert "30 wins" in body and "20 losses" in body, body[:500]
    assert "60.0%" in body, "the win rate is missing"
    assert "50 settled picks" in body, "the settled count is missing"
    # a win rate never appears without its price and its return
    assert "-5.6%" in body and "level stakes" in body, "no return figure"
    assert "average price 1.70" in body, "no average price"
    assert "11 Aug 2026" in body, f"no start date in client form: {body[:400]}"


def test_THE_SETTLEMENT_SAYS_PLAINLY_THAT_NO_MONEY_IS_STAKED():
    """A reader who is not Niki must be able to tell these are simulated."""
    body = _settlement_body()
    assert "no real money has been staked" in body.lower(), (
        f"the settlement does not say money is not at stake: {body[:500]}")
    assert "simulated" in body.lower()


def test_KNOWN_WRONG_THE_SETTLEMENT_CARRIES_NO_OPERATOR_VOCABULARY():
    """The control. `format_block` would fail every one of these.

    It renders "Experiment: CLV", "MODEL:", "FINAL:", "Stage 16", "p > 0.15"
    and "3 cohorts are 3 different configurations" — which is what the client
    settlement contained until 2026-11-04.
    """
    from src.reporting.experiment_record import (OPERATOR_VOCABULARY,
                                                 format_block)
    body = _settlement_body()
    leaked = [w for w in OPERATOR_VOCABULARY if w.lower() in body.lower()]
    assert not leaked, (
        f"operator vocabulary reached the client settlement: {leaked}")
    # and prove the check discriminates: the OPERATOR block trips it
    operator = "\n".join(format_block(_rec(), html=False))
    tripped = [w for w in OPERATOR_VOCABULARY if w.lower() in operator.lower()]
    assert len(tripped) >= 4, (
        f"the operator block only tripped {tripped}; the vocabulary list is "
        f"too weak to have caught the old settlement")


# ══════════ B. the operator channel is unset BY DECISION ════════════════════

def test_AN_UNSET_OPERATOR_ID_ANNOTATES_AND_NEVER_REACHES_THE_CLIENT(monkeypatch):
    """The decision: GitHub carries operator information, not Telegram.

    `TELEGRAM_OPERATOR_CHAT_ID` is deliberately not configured. With no
    destination `deliver_alert` must take the ANNOTATION path and must not
    reach the client chat.
    """
    import src.reporting.alert_delivery as ad
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", CLIENT)
    monkeypatch.delenv("TELEGRAM_OPERATOR_CHAT_ID", raising=False)
    posted = []
    monkeypatch.setattr(ad, "_post", lambda *a, **k: posted.append(a),
                        raising=False)
    annotated = []
    monkeypatch.setattr(ad, "_annotate",
                        lambda m: (annotated.append(m), True)[1])
    res = ad.deliver_alert("a model-freshness finding")
    assert posted == [], (
        "an alert was POSTED with no operator id configured; the only "
        "destination available was the client chat")
    assert annotated, "nothing was annotated, so the finding has NO carrier"
    assert res.ok is False and "not configured" in res.detail
    assert res.surfaced is True, (
        "the annotation path did not report that it surfaced the "
        "message, so nothing proves it has a carrier")


def test_KNOWN_WRONG_a_REINTRODUCED_CLIENT_FALLBACK_IS_CAUGHT(monkeypatch):
    """Prove the test above can fail.

    Simulates the defect that produced this whole exchange: an
    `or os.environ["TELEGRAM_CHAT_ID"]` fallback behind the operator id.
    """
    import os
    posted = []

    def _leaky_deliver(text, *, token=None, chat_id=None, sleep=None):
        dest = (chat_id or os.environ.get("TELEGRAM_OPERATOR_CHAT_ID", "")
                or os.environ.get("TELEGRAM_CHAT_ID", ""))   # the old defect
        if dest:
            posted.append(dest)
        return None

    monkeypatch.setenv("TELEGRAM_CHAT_ID", CLIENT)
    monkeypatch.delenv("TELEGRAM_OPERATOR_CHAT_ID", raising=False)
    _leaky_deliver("a model-freshness finding")
    assert posted == [CLIENT], (
        "the simulated fallback did NOT reach the client chat, so the test "
        "above is not demonstrably catching anything")


def test_NO_CODE_PATH_FALLS_BACK_TO_THE_CLIENT_ID():
    """Read every environment lookup in the delivery layer from the AST."""
    import ast
    import io
    src = io.open("src/reporting/alert_delivery.py", encoding="utf-8").read()
    keys = set()
    for node in ast.walk(ast.parse(src)):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and isinstance(node.func.value, ast.Attribute)
                and node.func.value.attr == "environ"
                and node.args
                and isinstance(node.args[0], ast.Constant)):
            keys.add(node.args[0].value)
    assert "TELEGRAM_CHAT_ID" not in keys, (
        f"the delivery layer can still resolve the CLIENT chat id from the "
        f"environment ({sorted(keys)}). That is the defect this stage ended.")
    assert "TELEGRAM_OPERATOR_CHAT_ID" in keys


# ══════ A2. the count and the period describe the SAME population ══════════

def _seeded_record(tmp_path, settled_days, unsettled_days):
    """Build a real ExperimentRecord over a seeded database."""
    import datetime as _dt

    from src.data.database import DatabaseManager
    from src.data.models import Base, Match, SavedPick, Team
    from src.reporting.experiment_record import build
    mgr = DatabaseManager(
        config=type("C", (), {"database": {"sqlite_path": str(tmp_path / "e.db")}})())
    Base.metadata.create_all(mgr.engine)
    with mgr.get_session() as s:
        s.add(Team(id=1, name="A")); s.add(Team(id=2, name="B"))
        pid = 1
        for d, res in ([(d, "win") for d in settled_days]
                       + [(d, None) for d in unsettled_days]):
            s.add(Match(id=100 + pid, league="x",
                        match_date=_dt.datetime.combine(d, _dt.time(20, 0)),
                        is_fixture=False, home_team_id=1, away_team_id=2))
            s.add(SavedPick(
                id=pid, match_id=100 + pid, pick_date=d,
                match_name="A vs B", league="x", market="Over 2.5",
                selection="Over 2.5 Goals", odds=1.80,
                predicted_probability=0.6, expected_value=0.08,
                confidence=0.65, kelly_stake_percentage=1.5,
                risk_level="low", model_agreement="solo",
                used_fallback_odds=False, is_paper=True, result=res))
            pid += 1
        s.commit()
    return build(mgr), mgr


def test_the_COUNT_AND_THE_PERIOD_COME_FROM_THE_SAME_POPULATION(tmp_path):
    """An UNSETTLED pick must not widen the period it is not counted in.

    MEASURED 2026-11-04 on production before the fix: 1,645 settled picks
    paired with a period ending 2026-11-04, which 13 unsettled picks had
    widened by a day. A client cannot check that and will not know to ask.
    """
    settled = [dt.date(2026, 8, 11), dt.date(2026, 9, 1), dt.date(2026, 11, 3)]
    unsettled = [dt.date(2026, 11, 20)]          # later, and NOT counted
    rec, _ = _seeded_record(tmp_path, settled, unsettled)
    assert rec.settled == len(settled), rec.settled
    assert str(rec.last_date) == "2026-11-03", (
        f"the period ends {rec.last_date}, but the last SETTLED pick is "
        f"2026-11-03. An unsettled pick widened a period it does not enter.")
    assert str(rec.first_date) == "2026-08-11"
    body = "\n".join(__import__(
        "src.reporting.experiment_record", fromlist=["client_block"]
    ).client_block(rec, html=False))
    assert f"from {len(settled)} settled picks" in body, body
    assert "3 Nov 2026" in body and "20 Nov" not in body, body


def test_KNOWN_WRONG_an_UNFILTERED_MAX_WOULD_DIFFER(tmp_path):
    """Prove the test above is checking something.

    If the seeded unsettled pick did not extend past the settled ones, the
    assertion would hold whether or not the query were filtered.
    """
    from sqlalchemy import text
    settled = [dt.date(2026, 8, 11), dt.date(2026, 11, 3)]
    unsettled = [dt.date(2026, 11, 20)]
    _rec_, mgr = _seeded_record(tmp_path, settled, unsettled)
    with mgr.get_session() as s:
        unf = s.execute(text(
            "SELECT max(pick_date) FROM saved_picks WHERE is_paper = 1")).scalar()
        fil = s.execute(text(
            "SELECT max(pick_date) FROM saved_picks "
            "WHERE is_paper = 1 AND result IS NOT NULL")).scalar()
    assert str(unf) != str(fil), (
        f"unfiltered {unf} equals filtered {fil}, so this fixture cannot "
        f"distinguish the two queries and the test above proves nothing")
    assert str(unf).startswith("2026-11-20") and str(fil).startswith("2026-11-03")


def test_EVERY_NUMBER_IN_THE_CLIENT_BLOCK_RECONCILES(tmp_path):
    """wins + losses + void == settled, and the rate is of the decided.

    MEASURED on production: 929 + 663 = 1,592 against 1,645 settled, leaving
    53 VOID picks that a client adding the two numbers could not account for.
    """
    from src.reporting.experiment_record import ExperimentRecord, client_block
    import inspect
    sig = inspect.signature(ExperimentRecord)
    kw = {k: v for k, v in (
        ("wins", 929), ("losses", 663), ("settled", 1645), ("cohorts", 15),
        ("win_rate", 929 / 1592), ("flat_roi", -0.057), ("avg_odds", 1.62),
        ("pl_units", -93.0), ("first_date", "2026-08-11"),
        ("last_date", "2026-11-03")) if k in sig.parameters}
    body = "\n".join(client_block(ExperimentRecord(**kw), html=False))
    assert "929 wins" in body and "663 losses" in body
    assert "53 void" in body, (
        f"the 53 void picks are invisible, so 929 + 663 does not reach the "
        f"1645 stated: {body}")
    assert "1592 decided" in body, "the percentage's base is not stated"
    assert "58.4%" in body
