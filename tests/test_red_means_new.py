"""Red means NEW, UNACKNOWLEDGED, or DETERIORATED — 2026-10-23.

`ci-audit` reddened on every run for two weeks and the answer was always "one
finding, a verdict by design". A workflow that reddens whenever a KNOWN state
exists reddens every day, and then red carries no information — which is how
GitHub's own notification list became the only channel being read.

    NEW            no recorded occurrence                     -> RED
    DETERIORATED   worse than the recorded figure              -> RED
    UNACKNOWLEDGED recorded, but its review date has passed    -> RED
    UNCHANGED      within the derived tolerance of the record  -> prints

AND MACHINERY ALWAYS REDDENS, whatever the finding state. That is handled
outside the classifier, so no classification can make a broken audit quiet.

THE FOUR CONTROLS THE RULE IS ONLY AS GOOD AS are all driven below, through
`_emit_alarms` rather than against `classify_finding` alone: the exit code is
what the workflow reads, and a classifier that returns the right word while
the emitter ignores it would pass a test on the word.
"""

import datetime as dt
import io
import os
import pathlib
import tempfile
from contextlib import redirect_stdout

import pytest

import scripts.ci_audit as ci

KEY = "margin.schedule"
REC = ci.KNOWN_FINDINGS[KEY]


def _emit(keyed=(), alarmed=(), status=()):
    """`(exit_code, stdout, telegram_body)` from the real emitter."""
    buf = io.StringIO()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "body.txt")
        os.environ["AUDIT_ALERT_BODY"] = path
        try:
            with redirect_stdout(buf):
                code = ci._emit_alarms(list(alarmed), list(status),
                                       list(keyed))
            body = pathlib.Path(path).read_text(encoding="utf-8") \
                if os.path.exists(path) else ""
        finally:
            os.environ.pop("AUDIT_ALERT_BODY", None)
    return code, buf.getvalue(), body


# ── THE FOUR CONTROLS ──────────────────────────────────────────────────────

def test_CONTROL_1_an_UNCHANGED_acknowledged_finding_is_GREEN():
    """The whole point. The recorded figure, re-reported, must not redden."""
    code, out, body = _emit(
        keyed=[(KEY, REC["figure"], "schedule margin +21m < 30m")])
    assert code == 0, out
    assert "KNOWN, does not redden" in out
    assert "UNCHANGED" in out
    # IT PRINTS, with its figure and its review date
    assert "+20.67" in out and REC["review"] in out, out
    assert body == "", "a known finding was pushed to Telegram"
    assert "no NEW, DETERIORATED or UNACKNOWLEDGED finding" in out


def test_CONTROL_2_the_SAME_finding_DETERIORATED_is_RED():
    """Worse than the record by more than the derived tolerance."""
    worse = REC["figure"] - ci.FINDING_TOLERANCE_MINUTES - 0.01
    code, out, body = _emit(keyed=[(KEY, worse, "schedule margin")])
    assert code == 1, out
    assert "DETERIORATED" in out
    assert "::error::audit alarm" in out
    assert "schedule margin" in body, "a deteriorated finding was withheld"


def test_CONTROL_3_a_NEW_finding_is_RED():
    """No recorded occurrence at all."""
    code, out, body = _emit(
        keyed=[("margin.invented", -5.0, "something never seen before")])
    assert code == 1, out
    assert "NEW" in out and "no recorded occurrence" in out
    assert "something never seen before" in body


def test_CONTROL_4_a_CRASH_WITH_NO_FINDING_is_RED():
    """Machinery failure must never be quiet.

    `_emit_alarms` is not what reddens a crash — `_main_guarded` returns
    `EXIT_CRASH` and the workflow branches on it — so this control checks BOTH
    halves: the emitter is honest about having nothing (exit 0), and the crash
    path is what carries the red.
    """
    code, out, _body = _emit()
    assert code == 0, "the emitter invented a finding it was not given"
    assert "Machinery failure is handled separately" not in out, (
        "with nothing at all it must say nothing, not reassure")

    # the crash path, driven
    import unittest.mock as mock
    with mock.patch.object(ci, "main", side_effect=RuntimeError("boom")):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = ci._main_guarded()
    assert rc == ci.EXIT_CRASH, rc
    assert rc != 0, "a crash exited zero"
    crash = buf.getvalue()
    assert "CRASHED" in crash
    assert "NOTHING was audited" in crash, (
        "the crash banner no longer says that nothing was audited, which is "
        "the one thing a reader must not assume")


# ── the tolerance is derived, and compared in the right direction ─────────

def test_the_TOLERANCE_is_DERIVED_from_the_measured_single_point_movement():
    assert ci.FINDING_TOLERANCE_MINUTES == ci.ACKNOWLEDGED_MARGIN_BREACH_JITTER
    assert ci.FINDING_TOLERANCE_MINUTES == 11.32
    assert ci.FINDING_TOLERANCE_COUNT == 0, (
        "one more absent scheduled run is a new fact, not jitter")
    src = pathlib.Path("scripts/ci_audit.py").read_text(encoding="utf-8")
    i = src.index("FINDING_TOLERANCE_MINUTES = ")
    head = src[max(0, i - 1400):i]
    for move in ("+5.53", "+2.79", "-11.32", "-2.24", "+1.77"):
        assert move in head, (
            f"the single-point move {move} is no longer recorded beside the "
            f"tolerance derived from it")


@pytest.mark.parametrize("delta,expect", [
    (0.0, "UNCHANGED"),
    (+50.0, "UNCHANGED"),                       # better is never worse
    (-ci.FINDING_TOLERANCE_MINUTES, "UNCHANGED"),       # exactly at the edge
    (-ci.FINDING_TOLERANCE_MINUTES - 0.01, "DETERIORATED"),
])
def test_the_MINUTES_comparison_treats_LOWER_as_worse(delta, expect):
    state, _why = ci.classify_finding(KEY, REC["figure"] + delta)
    assert state == expect, (delta, state)


def test_the_COUNT_comparison_treats_HIGHER_as_worse():
    """A figure's meaning is not recoverable from its value, so the direction
    comes from the tolerance kind."""
    saved = dict(ci.KNOWN_FINDINGS)
    ci.KNOWN_FINDINGS["t.count"] = {
        "figure": 3, "tolerance": "count", "since": "2026-10-23",
        "review": "2099-01-01", "why": "x", "action": "y"}
    try:
        assert ci.classify_finding("t.count", 3)[0] == "UNCHANGED"
        assert ci.classify_finding("t.count", 2)[0] == "UNCHANGED"
        assert ci.classify_finding("t.count", 4)[0] == "DETERIORATED"
    finally:
        ci.KNOWN_FINDINGS.clear()
        ci.KNOWN_FINDINGS.update(saved)


def test_an_EXPIRED_review_date_makes_it_UNACKNOWLEDGED_whatever_the_figure():
    """A deferral with a date nothing reads is not a deferral."""
    state, why = ci.classify_finding(KEY, REC["figure"],
                                     today=dt.date(2099, 1, 1))
    assert state == "UNACKNOWLEDGED"
    assert REC["review"] in why
    # and the EXACT figure does not rescue it
    code, out, _ = _emit(keyed=[(KEY, REC["figure"], "x")])
    assert code == 0, "today it is unexpired, so this control is mis-dated"


def test_an_UNPARSEABLE_review_date_is_UNACKNOWLEDGED():
    saved = dict(ci.KNOWN_FINDINGS)
    ci.KNOWN_FINDINGS[KEY] = dict(REC, review="not-a-date")
    try:
        assert ci.classify_finding(KEY, REC["figure"])[0] == "UNACKNOWLEDGED"
    finally:
        ci.KNOWN_FINDINGS.clear()
        ci.KNOWN_FINDINGS.update(saved)


def test_the_RECORD_carries_its_FIGURE_its_REVIEW_and_its_ACTION():
    for field in ("figure", "tolerance", "since", "review", "why", "action"):
        assert REC.get(field), f"the recorded finding has no {field}"
    assert REC["figure"] == 20.665799601890512, (
        "the recorded figure is not the full-precision reading; a predicate "
        "reading a rounded display absorbs a breach it was never shown")
    assert dt.date.fromisoformat(REC["review"]) > dt.date(2026, 10, 23)


# ── the status split survives the keying ────────────────────────────────

def test_a_STATUS_alarm_still_REDDENS_and_still_misses_TELEGRAM():
    code, out, body = _emit(status=["123 daily-picks DID_NOT_RUN"])
    assert code == 1
    assert "NOT sent to Telegram" in out
    assert body == ""


def test_an_UNKEYED_alarm_is_treated_as_NEW_and_reddens():
    """Correct for every unkeyed site today: they are all rare, and a per-run
    verdict is new for each run by construction."""
    code, out, body = _emit(alarmed=["3 pattern(s) FAIL the self-test"])
    assert code == 1
    assert "pattern(s) FAIL" in body


# ── what each workflow reddens on tomorrow ─────────────────────────────

#: A GREEN AUDIT MUST BE A PREDICTION, NOT A RELIEF. Written down the day the
#: rule shipped, so tomorrow's colour is checkable against what was expected.
#:
#: `ci-audit`  GREEN, unless: a run scores BROKEN or DID_NOT_RUN that is not
#:             in RESOLVED_RUNS (NEW); the schedule margin deteriorates past
#:             +9.35 (the acknowledged floor) once the lag series reaches
#:             n>=4 and stops being UNSEPARABLE; the prospective/actionable
#:             margin fires; a pattern self-test fails; or the audit crashes.
#: `reminders` GREEN, unless: fewer than 9 reminders RUN (NOT EVALUATED), or
#:             one of the nine fails. The only date-driven one between now and
#:             2026-11-01 is H1_RESERVATION_ACK_UNTIL on 2026-11-01.
TOMORROW = {
    "ci-audit": "GREEN",
    "reminders": "GREEN",
}


def test_TOMORROWS_COLOUR_is_PREDICTED_and_the_exceptions_enumerated():
    assert set(TOMORROW) == {"ci-audit", "reminders"}
    assert all(v == "GREEN" for v in TOMORROW.values())
    src = pathlib.Path("tests/test_red_means_new.py").read_text(
        encoding="utf-8")
    # each prediction carries its own falsifiers, or it is a hope
    for frag in ("RESOLVED_RUNS", "+9.35", "UNSEPARABLE", "crashes",
                 "NOT EVALUATED", "2026-11-01"):
        assert frag in src, f"the prediction does not name {frag}"


def test_the_MARGIN_is_UNSEPARABLE_TODAY_so_it_cannot_redden_either_way():
    """Today's state, recorded: the cron move of 2026-10-22 emptied the lag
    series, so n is below `LAG_CLUSTER_MIN_N` and the margin reports without
    alarming. It is neither clear nor breached."""
    m = ci.schedule_margin([(None, 272.0)], 60, deadline_min=570,
                           day_type="weekday")
    assert m["cluster_state"] == "UNSEPARABLE"
    assert m["alarm"] is False
    assert "neither clear nor breached" in m["no_alarm_reason"]
