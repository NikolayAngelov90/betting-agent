"""Telegram is a CLIENT channel. Three things may reach it, ever.

    1. the daily picks card
    2. the next day's settlement
    3. overall performance

Other people read this channel, so operator content does not belong in it:
a red run, a suspended provider, a below-threshold model, a credit warning.
GitHub already notifies about runs.

OPERATOR CONTENT IS NOT DELETED, IT IS REROUTED. `send_alert` now addresses
`operator_chat_id`, and when that is unset `deliver_alert` writes an
`::error::` annotation and a GitHub step-summary entry — a carrier that does
not depend on Telegram. "Not to the client channel" and "not to nothing" are
both satisfied without a silent redirect.

THREE KNOWN-WRONG CONTROLS, because the first two alone would pass against a
channel that sent nothing at all:

  a FAILED run sends nothing to the client channel
  a SUCCESSFUL pick run sends the card
  a SETTLEMENT sends the settlement
"""

import asyncio
import datetime as dt

import pytest

from src.reporting import telegram_bot as tb


class _Notifier(tb.TelegramNotifier):
    """A notifier whose two destinations are observable and distinct."""

    def __init__(self, operator=""):
        self.config = None
        self.enabled = True
        self.bot_token = "t"
        self.chat_id = "CLIENT"
        self.operator_chat_id = operator
        self._bot = None
        self._last_send_error = ""
        self.client_msgs = []
        self.operator_msgs = []

    def _get_bot(self):
        return object()

    async def _send_chunked(self, message, header=None, report="", **_k):
        self.client_msgs.append((report, message))

    async def _send_message(self, text, **_k):
        self.client_msgs.append(("message", text))
        return object()


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def _pick():
    from src.betting.value_calculator import BetRecommendation
    return BetRecommendation(
        match="A vs B", match_id=1, market="Over 2.5",
        selection="Over 2.5 Goals", odds=1.8, predicted_probability=0.6,
        expected_value=0.08, confidence=0.65, kelly_stake_percentage=1.5,
        recommended_stake=1.5, reasoning="", risk_level="low",
        league="england/premier-league", model_agreement="solo",
        match_date=dt.datetime(2026, 11, 4, 20, 0))


# ════════════════════ the sender census, pinned ═════════════════════════════

CLIENT_SENDERS = ("send_daily_picks", "send_settlement_report",
                  "send_performance_report")
OPERATOR_SENDERS = ("send_alert", "send_welcome_message")


def test_EVERY_SENDER_IS_CLASSIFIED_and_the_census_is_pinned():
    """A new sender must be classified here before it can ship."""
    found = {n for n in dir(tb.TelegramNotifier)
             if n.startswith("send_") and callable(getattr(tb.TelegramNotifier, n))}
    classified = set(CLIENT_SENDERS) | set(OPERATOR_SENDERS)
    assert found == classified, (
        f"unclassified sender(s): {sorted(found - classified)}; "
        f"classified but absent: {sorted(classified - found)}. Every sender "
        f"is CLIENT or OPERATOR, and a new one is a decision, not a default.")


def test_the_ALERT_PATH_ADDRESSES_THE_OPERATOR_CHANNEL(monkeypatch):
    """Observable: the chat id `deliver_alert` is handed."""
    seen = {}

    def _fake(text, *, token=None, chat_id=None, sleep=None):
        seen["chat_id"] = chat_id
        seen["text"] = text
        return tb_result()

    def tb_result():
        from src.reporting.alert_delivery import DeliveryResult
        return DeliveryResult(True, 1, "", False)

    import src.reporting.alert_delivery as ad
    monkeypatch.setattr(ad, "deliver_alert", _fake)
    n = _Notifier(operator="OPERATOR")
    _run(n.send_alert("a below-threshold model"))
    assert seen["chat_id"] == "OPERATOR", (
        f"the alert was addressed to {seen['chat_id']!r}; operator content "
        f"must not reach the client channel")
    assert n.client_msgs == [], "an alert reached the client channel"


def test_an_UNSET_OPERATOR_CHANNEL_DOES_NOT_FALL_BACK_TO_THE_CLIENT(monkeypatch):
    """The content still has a carrier, and it is NOT the client channel."""
    seen = {}

    def _fake(text, *, token=None, chat_id=None, sleep=None):
        from src.reporting.alert_delivery import DeliveryResult
        seen["chat_id"] = chat_id
        return DeliveryResult(False, 0, "not configured", True)

    import src.reporting.alert_delivery as ad
    monkeypatch.setattr(ad, "deliver_alert", _fake)
    n = _Notifier(operator="")
    _run(n.send_alert("a suspended provider"))
    assert seen["chat_id"] == "", seen
    assert n.client_msgs == [], (
        "with no operator channel the alert fell back to the CLIENT channel, "
        "which is the redirect this change exists to prevent")


# ═══════════════ CONTROL 1 — a failed run sends nothing ════════════════════

def test_CONTROL_1_a_FAILED_RUN_SENDS_NOTHING_TO_THE_CLIENT(monkeypatch):
    """No picks produced: the client channel stays silent.

    `send_daily_picks` with an empty list used to send "No value picks found
    for today." — a run-status message in a channel other people read.
    """
    n = _Notifier()
    _run(n.send_daily_picks([]))
    assert n.client_msgs == [], (
        f"a failed/empty run put {len(n.client_msgs)} message(s) in the "
        f"client channel: {[m[0] for m in n.client_msgs]}")


# ═══════════════ CONTROL 2 — a successful pick run sends the card ══════════

def test_CONTROL_2_a_SUCCESSFUL_PICK_RUN_SENDS_THE_CARD(monkeypatch):
    monkeypatch.setattr(tb, "_picks_sent_today", lambda: False)
    monkeypatch.setattr(tb, "_mark_picks_sent_today", lambda: None)
    n = _Notifier()
    _run(n.send_daily_picks([_pick()], force=True, paper_mode=True))
    assert len(n.client_msgs) == 1, n.client_msgs
    report, body = n.client_msgs[0]
    assert report == "daily picks"
    assert "A vs B" in body and "Over 2.5 Goals" in body, body[:200]


# ═══════════════ CONTROL 3 — a settlement sends the settlement ═════════════

def test_CONTROL_3_a_SETTLEMENT_SENDS_THE_SETTLEMENT():
    n = _Notifier()
    # `send_settlement_report` consumes DICTS (`pick.get(...)`), as
    # `settle_predictions()` yields them — not ORM rows.
    settled = [{
        "match_name": "A vs B", "match": "A vs B",
        "league": "england/premier-league",
        "market": "Over 2.5", "selection": "Over 2.5 Goals", "odds": 1.8,
        "result": "win", "score": "2-1",
        "stake": 1.5, "kelly_stake_percentage": 1.5, "expected_value": 0.08,
        "confidence": 0.65, "predicted_probability": 0.6,
        "model_agreement": "solo", "review_action": "KEEP",
        "is_paper": True, "pick_date": dt.date(2026, 11, 3),
    }]
    _run(n.send_settlement_report(settled, stats={}))
    assert n.client_msgs, "the settlement did not reach the client channel"


# ═══════════════ the population travels WITH the number ════════════════════

def test_the_PERFORMANCE_FIGURE_CARRIES_ITS_POPULATION():
    """A client-facing percentage without its population is the mixed-ratio
    defect, now in front of other people."""
    from src.reporting.experiment_record import ExperimentRecord, format_block
    import inspect
    sig = inspect.signature(ExperimentRecord)
    kw = {}
    for name, p in sig.parameters.items():
        if p.default is not inspect.Parameter.empty:
            continue
        kw[name] = {"int": 0, "float": 0.0, "str": ""}.get(
            getattr(p.annotation, "__name__", ""), None)
    for k, v in (("wins", 30), ("losses", 20), ("settled", 50),
                 ("cohorts", 3), ("win_rate", 0.60), ("flat_roi", -0.055),
                 ("avg_odds", 1.70), ("pl_units", -2.8),
                 ("first_date", "2026-08-11"), ("last_date", "2026-11-03")):
        if k in sig.parameters:
            kw[k] = v
    rec = ExperimentRecord(**kw)
    body = "\n".join(format_block(rec, html=False))
    assert "PAPER" in body, body
    assert "2026-08-11" in body and "3 cohorts" in body, body
    # rule 1: never a win rate alone
    assert "60.0%" in body and "ROI" in body and "avg odds" in body, body
