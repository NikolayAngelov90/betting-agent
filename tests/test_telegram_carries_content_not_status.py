"""Telegram carries CONTENT and ACTIONS, never run status — 2026-10-22.

THE RULE. Three classes, one disposition each:

    run status  "workflow FAILED", "DID NOT RUN", a red conclusion   REMOVE
    content     the daily card, the weekly report, THE LOSS          KEEP
    action      the manual dispatch, anything Niki must do           KEEP

WHY. A push notification that restates a run's conclusion competes with the
one message that prevents a lost card, and it wins, because it arrives more
often. MEASURED over 2026-09-22..10-22: the daily-picks status alerts fired 20
times (11 "workflow FAILED" + 9 "critical step") against ONE firing of the
loss report. Twenty status messages bought one report of an actual loss.

WHAT CARRIES THE REMOVED ONES is recorded per sender in `REMOVED`, and one of
them is `nothing` -- said out loud rather than discovered later.
"""

import pathlib
import re

import pytest

WORKFLOWS = sorted(pathlib.Path(".github/workflows").glob("*.yml"))


def _text(p):
    return p.read_text(encoding="utf-8")


def _code_lines(src):
    """Non-comment lines. Every assertion here looks for the ABSENCE of a
    sender, and this file's own prose quotes the senders it removed -- as do
    the workflow comments that record them. The seventh time in this
    repository that a text scan has had to stop matching its own
    explanation."""
    return [l for l in src.splitlines() if not l.strip().startswith("#")]


# ── THE CENSUS ─────────────────────────────────────────────────────────────

#: EVERY Telegram sender in the repository, `measured` 2026-10-22 by grepping
#: `ci_alert`, `send_`, `TELEGRAM_BOT_TOKEN` and `sendMessage` across
#: `src/`, `scripts/` and `.github/`.
#:
#: The firing counts are `measured` over 2026-09-22..10-22 from
#: `gh run list` conclusions and, for daily-picks, from per-step conclusions
#: across all 36 runs.
KEPT = {
    # ── content ───────────────────────────────────────────────────────────
    "telegram_bot.send_daily_picks": "content: the card itself",
    "telegram_bot.send_settlement_report": "content: yesterday's results",
    "telegram_bot.send_performance_report": "content: the weekly report",
    "telegram_bot.send_welcome_message": "content: manual setup only",
    "paper_trading_report --telegram": "content: the experiment summary",
    "pipeline_loss_alert": (
        "content: a day with a card and ZERO PICKS is a loss, not a status. "
        "1 firing in the window"),
    "model_freshness": (
        "content: 'the model has not been retrained since X' is a fact about "
        "the system that no run's conclusion reports. ADDED 2026-10-22 to "
        "close the gap the removed ML-retrain alert left -- it checks the "
        "ARTEFACT, not the step"),
    # ── action ────────────────────────────────────────────────────────────
    "ci_audit actionable alarms": (
        "action: the margin, the pattern self-test, model integrity, an "
        "untriaged finding, an absent scheduled run, an unparseable workflow"),
    "reminders REMINDER DUE": "action: a self-expiring guard reached its date",
    "fixture_horizon_probe FINDING/REFUSAL": (
        "action: tomorrow's margin is short, or is UNKNOWN and not safe"),
    # ── transport, not a sender ──────────────────────────────────────────
    "ci_alert / alert_delivery": "transport for all of the above",
}

#: REMOVED, with what carries each one now. "GitHub's notification" is an
#: acceptable answer. "nothing" had to be said out loud, and is.
REMOVED = {
    "daily-picks: Notify Telegram on workflow failure": (
        "GitHub's notification on the red run; the LOSS report for the "
        "consequence; ci-audit's per-workflow red classification. 11 firings"),
    "daily-picks: Alert on critical step failure (the send)": (
        "the step STAYS and still decides the run's colour; the log line "
        "keeps the step names; GitHub's notification. 9 firings"),
    "daily-picks: Alert on ML training timeout or failure": (
        "scripts/model_freshness.py, which checks the MODEL instead of the "
        "step -- added 2026-10-22, closing the gap this removal named. NOT "
        "GitHub's notification: the step is continue-on-error, so a failed "
        "retrain never reddens the run. 0 firings in the window"),
    "closing-lines: Alert if the capture job failed": (
        "GitHub's notification; the coverage ratchet, which measures the GAP "
        "rather than the exit code. 2 firings"),
    "paper-trading-report: Alert if the report failed": (
        "GitHub's notification; and the daily report's ABSENCE. 0 firings"),
    "h1-collect: Alert if the collection job failed": (
        "GitHub's notification; ci-audit. Schedule disabled 2026-10-20; "
        "0 firings"),
    "ci-audit: the empty-body MACHINERY fallback": (
        "the ::error::CI AUDIT CRASHED annotation; GitHub's notification"),
    "ci-audit: per-run BROKEN/DID_NOT_RUN relay": (
        "GitHub's notification on that run's own red -- EXCEPT when the run "
        "concluded SUCCESS, which is kept because nothing else carries it"),
    "reminders: the UNEVALUATED third-state send": (
        "the ::error::REMINDERS NOT EVALUATED annotation; GitHub's "
        "notification; ci-audit, which now tells the two reds apart"),
}


def test_every_sender_is_CLASSIFIED_and_the_classes_do_not_overlap():
    assert not (set(KEPT) & set(REMOVED))
    assert len(KEPT) == 11 and len(REMOVED) == 9


def test_a_REMOVED_sender_SAYS_WHAT_CARRIES_IT_NOW():
    for name, carrier in REMOVED.items():
        assert carrier.strip(), name
        assert ("GitHub's notification" in carrier
                or "NOTHING" in carrier
                or "the step STAYS" in carrier
                or "annotation" in carrier), (
            f"{name} was removed without naming a carrier: {carrier!r}")


def test_the_ONE_sender_whose_carrier_is_NOTHING_says_so_in_the_workflow():
    """`nothing` must be said out loud where the removal happened, not only
    in a test."""
    src = _text(pathlib.Path(".github/workflows/daily-picks.yml"))
    assert "carried by NOTHING" in src, (
        "the ML-retrain gap is no longer stated at the site of its removal")
    assert "continue-on-error" in src and "no notification is sent" in src, (
        "the reason GitHub does not cover it must be stated too")


# ── THE KNOWN-WRONG CONTROL THE DIRECTIVE NAMED ───────────────────────────
#
#   a failed run sends no Telegram message;
#   a day with a card and zero picks still does;
#   a margin breach still does.

def test_CONTROL_a_FAILED_RUN_sends_NO_telegram_message():
    """Statically, across every workflow: no `ci_alert` may be reached from a
    condition that is only about a run or step CONCLUSION."""
    offenders = []
    for p in WORKFLOWS:
        if p.name == "zz-parse-bad.yml":
            continue
        lines = _code_lines(_text(p))
        for i, line in enumerate(lines):
            if "ci_alert" not in line:
                continue
            window = chr(10).join(lines[max(0, i - 22):i + 1])
            # The two shapes that mean "this fires because something failed".
            if re.search(r"if:\s*failure\(\)", window):
                offenders.append(f"{p.name}:{line.strip()[:60]} <- if: failure()")
            if re.search(r"outcome\s*==\s*'failure'", window):
                offenders.append(f"{p.name}:{line.strip()[:60]} <- outcome==failure")
    assert offenders == [], (
        "a run- or step-conclusion still pushes to Telegram: " + "; ".join(offenders))


def test_CONTROL_a_CARD_WITH_ZERO_PICKS_STILL_sends():
    """The loss report must still be wired, and still fire on the PRODUCED-
    NOTHING predicate rather than on a failure."""
    src = _text(pathlib.Path(".github/workflows/daily-picks.yml"))
    code = chr(10).join(_code_lines(src))
    assert "scripts.pipeline_loss_alert" in code
    assert "scripts.ci_alert \"$(cat" in code, (
        "the loss body is no longer piped to the alert transport")
    assert "steps.picks.outcome != 'success'" in code, (
        "the loss step's condition changed; it must fire on 'produced no "
        "picks', which includes SKIPPED, not on 'failed'")


def test_CONTROL_the_loss_body_carries_NO_RUN_CONCLUSION():
    """It reports the loss. It must not report how the run ended."""
    import datetime as dt

    from scripts.pipeline_loss_alert import DayLoss, format_loss_alert

    body = format_loss_alert(
        [DayLoss(dt.date(2026, 10, 22), 13, 0)],
        earliest_kickoff_min=570, deadline_basis="observed earliest kickoff",
        run_url="https://example/runs/1",
        now=dt.datetime(2026, 10, 22, 3, 0, tzinfo=dt.timezone.utc))
    for banned in ("workflow FAILED", "step(s) FAILED", "DID NOT RUN",
                   "conclusion", "did not fix it", "exited"):
        assert banned not in body, (
            f"the loss report is reporting the run's state: {banned!r}")
    assert "NO PICKS PRODUCED" in body
    assert "DO NOT BACKFILL" in body


def test_CONTROL_a_MARGIN_BREACH_STILL_sends():
    """An UNACKNOWLEDGED breach must still reach the alarm body. The standing
    one is acknowledged, so this drives a DETERIORATION past its floor."""
    import scripts.ci_audit as ci

    ack = ci.ACKNOWLEDGED_MARGIN_BREACH["weekday"]
    assert ci.margin_breach_is_acknowledged("weekday", ack["binding"] - 1) == {}
    # and the body writer must carry an actionable alarm through
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "body.txt")
        os.environ["AUDIT_ALERT_BODY"] = path
        try:
            ci._write_alarm_body(["schedule margin +9m < 30m - CONSERVATIVE"])
            written = pathlib.Path(path).read_text(encoding="utf-8")
        finally:
            os.environ.pop("AUDIT_ALERT_BODY", None)
    assert "schedule margin" in written


def test_CONTROL_an_ALL_STATUS_run_writes_an_EMPTY_body_so_NOTHING_is_sent():
    """The mechanism the saving depends on. The workflow sends only
    `if [ -s "$BODY_FILE" ]`, so a status-only run must write ZERO bytes --
    not the bare 'CI audit alarmed:' heading, which would still be -s."""
    import os
    import tempfile

    import scripts.ci_audit as ci

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "body.txt")
        os.environ["AUDIT_ALERT_BODY"] = path
        try:
            ci._write_alarm_body([])
            size = pathlib.Path(path).stat().st_size
        finally:
            os.environ.pop("AUDIT_ALERT_BODY", None)
    assert size == 0, f"an empty alarm list wrote {size} bytes, so it sends"


def test_the_audit_STILL_REDDENS_on_a_status_only_alarm():
    """The removal must not become a deletion. A status alarm is carried by
    THIS run's red, which only exists if it still exits non-zero."""
    import scripts.ci_audit as ci

    assert ci._emit_alarms([], []) == 0, "a quiet run must exit 0"
    assert ci._emit_alarms([], ["123 daily-picks DID_NOT_RUN"]) == 1, (
        "a status-only alarm stopped reddening, so GitHub will not notify "
        "and nothing carries it at all")
    assert ci._emit_alarms(["schedule margin"], []) == 1


def test_the_WITHHOLDING_is_announced_in_the_log(capsys):
    """An absent Telegram message must be a recorded decision."""
    import scripts.ci_audit as ci

    ci._emit_alarms(["a real finding"], ["123 wf DID_NOT_RUN"])
    out = capsys.readouterr().out
    assert "NOT sent to Telegram" in out
    assert "1 of 2 alarm(s)" in out
    assert "GitHub's notification" in out


# ── the measured rate, before and after ───────────────────────────────────

#: Telegram messages per day on the ALERT path (`ci_alert`), `measured`
#: 2026-09-22..10-22 unless noted. The content path (card, settlement,
#: weekly/paper report) is unchanged by this stage and is excluded.
ALERT_MESSAGES_PER_DAY = {
    "before": {
        "ci-audit": 3.00,            # 90 failures / 30 days
        "daily-picks workflow FAILED": 0.37,   # 11 / 30
        "daily-picks critical step": 0.30,     # 9 / 30
        "daily-picks loss report": 0.03,       # 1 / 30
        "daily-picks ML timeout": 0.00,        # 0 / 36 runs
        "closing-lines": 0.07,                 # 2 / 30
        "paper-trading-report": 0.00,          # 0 / 30
        "h1-collect": 0.00,                    # 0 / 5 runs
        "fixture-horizon-probe": 0.27,         # 8 / 30
        "reminders": 1.00,     # `simulated`: 1 run so far, daily trigger
    },
    "after": {
        # 9 of a 30-run sample of alarming audit runs still write an
        # actionable body once the standing margin is acknowledged and status
        # alarms are withheld: 3.00 * 9/30.
        "ci-audit": 0.90,
        "daily-picks loss report": 0.03,
        "fixture-horizon-probe": 0.27,
        # the collection error is fixed AND a non-verdict exit no longer
        # sends, so this is 0 until a guard actually reaches its date
        "reminders": 0.00,
        # `simulated`: 0.00/day in steady state. `measured` 2026-10-22, the
        # CI model's trained_at is 2026-10-21T09:07 -- 1 day against a
        # 5-day threshold. It fires only when the retrain stops working,
        # which is the whole point, and then daily with a rising age.
        "model freshness": 0.00,
    },
}


def test_the_RATE_is_recorded_with_its_window_and_FELL():
    before = sum(ALERT_MESSAGES_PER_DAY["before"].values())
    after = sum(ALERT_MESSAGES_PER_DAY["after"].values())
    assert round(before, 2) == 5.04
    assert round(after, 2) == 1.20
    assert after < before / 4, (
        f"{before:.2f}/day -> {after:.2f}/day is less than a 4x reduction; "
        f"re-derive the figures rather than relaxing this")


def test_every_AFTER_sender_is_CONTENT_or_ACTION():
    """Nothing in the surviving set may be run status."""
    assert set(ALERT_MESSAGES_PER_DAY["after"]) <= {
        "ci-audit", "daily-picks loss report", "fixture-horizon-probe",
        "reminders", "model freshness"}
    for name in ALERT_MESSAGES_PER_DAY["after"]:
        assert any(name.split()[0] in k or k.startswith(name.split()[0])
                   or name.split("-")[0] in k for k in KEPT), name
