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

#: A RECORD OF OUR OWN, BECAUSE THE CONTROLS MUST NOT BREAK WHEN A REAL
#: FINDING IS DISCHARGED — 2026-11-07.
#:
#: These four controls used to drive the classifier through the LIVE
#: `margin.schedule` record, asserting its exact figure and its exact review
#: date. So when that finding was discharged by its own action (its margin
#: moved +14.81m past its threshold and its date review was retired as
#: uninformative), FOUR CONTROLS FAILED — and none of them was testing
#: anything that had changed. A control coupled to live data reports the
#: data moving as a mechanism defect.
#:
#: The mechanism is now driven through `SYNTHETIC`, which carries a figure, a
#: tolerance and a FUTURE review date so each of the four states is
#: reachable. The live record gets its own test below, asserting the shape a
#: discharged finding must have — which is a different claim, deliberately.
SYNTHETIC_KEY = "synthetic.margin"
SYNTHETIC = {
    "figure": 20.665799601890512,
    "tolerance": "minutes",
    "since": "2026-10-22",
    "review": "2099-01-01",
    "why": "a synthetic record, so the four controls own their own inputs",
    "action": "none; this record exists to drive the classifier",
}


@pytest.fixture(autouse=True)
def _synthetic_record():
    """`SYNTHETIC_KEY` present for every test in this module, removed after."""
    ci.KNOWN_FINDINGS[SYNTHETIC_KEY] = dict(SYNTHETIC)
    try:
        yield
    finally:
        ci.KNOWN_FINDINGS.pop(SYNTHETIC_KEY, None)


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
        keyed=[(SYNTHETIC_KEY, SYNTHETIC["figure"],
                "schedule margin +21m < 30m")])
    assert code == 0, out
    assert "KNOWN, does not redden" in out
    assert "UNCHANGED" in out
    # IT PRINTS, with its figure and its review date
    assert "+20.67" in out and SYNTHETIC["review"] in out, out
    assert body == "", "a known finding was pushed to Telegram"
    assert "no NEW, DETERIORATED or UNACKNOWLEDGED finding" in out


def test_CONTROL_2_the_SAME_finding_DETERIORATED_is_RED():
    """Worse than the record by more than the derived tolerance.

    SYNTHETIC, because the live record now carries a `floor` and the floor is
    checked FIRST -- so driving this through it would exercise the relapse
    path and assert the wrong word. Two mechanisms, two tests.
    """
    worse = SYNTHETIC["figure"] - ci.FINDING_TOLERANCE_MINUTES - 0.01
    code, out, body = _emit(keyed=[(SYNTHETIC_KEY, worse, "schedule margin")])
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
    """SYNTHETIC: this is the TOLERANCE's behaviour, with no floor in the way.

    The live record's floor is tested by its own control; mixing them would
    make the direction-of-worse claim depend on which threshold fired.
    """
    state, _why = ci.classify_finding(SYNTHETIC_KEY,
                                      SYNTHETIC["figure"] + delta)
    assert state == expect, (delta, state)
    assert "floor" not in SYNTHETIC, (
        "the synthetic record gained a floor, so these four cases no longer "
        "isolate the relative tolerance")


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
    state, why = ci.classify_finding(SYNTHETIC_KEY, SYNTHETIC["figure"],
                                     today=dt.date(2099, 1, 2))
    assert state == "UNACKNOWLEDGED"
    assert SYNTHETIC["review"] in why
    # and the EXACT figure does not rescue it
    code, out, _ = _emit(keyed=[(SYNTHETIC_KEY, SYNTHETIC["figure"], "x")])
    assert code == 0, "today it is unexpired, so this control is mis-dated"


def test_an_UNPARSEABLE_review_date_is_UNACKNOWLEDGED():
    ci.KNOWN_FINDINGS[SYNTHETIC_KEY] = dict(SYNTHETIC, review="not-a-date")
    assert ci.classify_finding(
        SYNTHETIC_KEY, SYNTHETIC["figure"])[0] == "UNACKNOWLEDGED"


def test_EVERY_LIVE_RECORD_carries_its_FIGURE_and_a_TRIGGER():
    """The live table's own contract, which is NOT the controls' contract.

    A record may carry a date review or a mechanism trigger, but it must
    carry one of them: a figure with neither cannot ever be looked at again.
    """
    for key, rec in ci.KNOWN_FINDINGS.items():
        for field in ("figure", "tolerance", "since", "why", "action"):
            assert rec.get(field) is not None, f"{key} has no {field}"
        assert rec.get("review") or rec.get("review_trigger"), (
            f"{key} carries neither a `review` date nor a `review_trigger`. "
            f"A recorded figure with no way back to it is a permanent "
            f"silence, which is the opposite of an acknowledgement")
        assert isinstance(rec["figure"], float), (
            f"{key}'s figure is not a float; a predicate reading a rounded "
            f"display absorbs a breach it was never shown")


def test_the_DISCHARGED_SCHEDULE_MARGIN_keeps_what_it_superseded():
    """`margin.schedule` was discharged 2026-11-07 BY ITS OWN ACTION.

    The action it registered on 2026-10-22 was "the parallel picks series
    reaching n=10". That landed on 2026-11-06 at n=12, `FULL_CARD_PICKS_MAX`
    was substituted 88.1 -> 40.35, and the margin moved +14.81m past its
    threshold. So this is a discharge, not a re-acknowledgement, and the
    retired figure stays with its date and its regime.
    """
    assert REC["figure"] == pytest.approx(35.4755, abs=0.001)
    assert REC["clears_threshold"] is True
    assert REC["figure"] > ci.MARGIN_ALARM_MIN, (
        f"the discharged figure {REC['figure']:.2f} no longer clears the "
        f"{ci.MARGIN_ALARM_MIN}m threshold, so it is a BREACH again and must "
        f"be acknowledged rather than recorded as resolved")
    # the date review is retired ON PURPOSE, and the trigger replaces it
    assert "review" not in REC, (
        "a date review is back on a quantity that only moves when a "
        "mechanism moves. It fired on 2026-11-06 against a figure that had "
        "already improved by 14.8 minutes -- a reminder with no information")
    for mech in ("cron", "FULL_CARD_PICKS_MAX", "lateness"):
        assert mech in REC["review_trigger"], (
            f"`review_trigger` no longer names {mech}, so a reader cannot "
            f"tell what would make this figure stale")
    # RETIRED, NOT DELETED
    sup = REC["superseded"]
    assert sup["figure"] == 20.665799601890512
    assert sup["since"] == "2026-10-22" and sup["review"] == "2026-11-06"
    assert "88.1" in sup["regime"] and "SERIAL" in sup["regime"]
    assert sup["figure"] < ci.MARGIN_ALARM_MIN < REC["figure"], (
        "the superseded figure no longer sits below the threshold with the "
        "new one above it, which is the whole shape of the discharge")
    moved = REC["figure"] - sup["figure"]
    assert moved == pytest.approx(14.81, abs=0.01), f"{moved:.2f}"


def test_KNOWN_WRONG_A_DISCHARGE_DOES_NOT_SILENCE_A_RELAPSE():
    """THE CONTROL THAT FOUND A REAL SILENCE, kept as the reason `floor` exists.

    Recording the discharged figure (+35.48) with a RELATIVE tolerance of
    11.32m made every value down to +24.16 read as UNCHANGED -- including the
    band 24.16..30.00, which is BELOW the registered 30-minute threshold and
    is a genuine breach. **Recording an improvement had bought a 5.84-minute
    silence**, and it would have reached a run.

    An absolute threshold cannot be expressed by a relative tolerance, so the
    record carries `floor` and the classifier returns `RELAPSED` for it.
    """
    # the silent band, measured from the two constants rather than asserted
    tol_edge = REC["figure"] - ci.FINDING_TOLERANCE_MINUTES
    assert tol_edge < ci.MARGIN_ALARM_MIN, (
        "the tolerance edge no longer sits below the threshold, so the "
        "discharge no longer creates a silent band and this control is "
        "describing a condition that has gone")
    silent_band = ci.MARGIN_ALARM_MIN - tol_edge
    assert silent_band == pytest.approx(5.84, abs=0.01), f"{silent_band:.2f}"

    # a figure INSIDE that band is a breach, and must redden anyway
    inside = (tol_edge + ci.MARGIN_ALARM_MIN) / 2.0
    assert inside > tol_edge, "not inside the tolerance"
    assert inside < ci.MARGIN_ALARM_MIN, "not below the threshold"
    state, why = ci.classify_finding(KEY, inside)
    assert state == "RELAPSED", (
        f"a margin of {inside:+.2f} -- below the {ci.MARGIN_ALARM_MIN}m "
        f"threshold but within tolerance of the discharged figure -- "
        f"classified {state}. That is the silence the floor closes")
    assert "floor" in why
    assert "RELAPSED" in ci.FINDING_STATES_THAT_REDDEN
    code, out, body = _emit(keyed=[(KEY, inside, "schedule margin relapsed")])
    assert code == 1, out
    assert "RELAPSED" in out
    assert "schedule margin" in body, "a relapse was withheld from Telegram"

    # and a figure ABOVE the threshold still prints rather than reddening,
    # or the floor would have replaced one silence with a daily red
    assert ci.classify_finding(KEY, ci.MARGIN_ALARM_MIN + 0.5)[0] == "UNCHANGED"


def test_the_FLOOR_IS_THE_REGISTERED_THRESHOLD_and_is_not_edited():
    """`floor` must BE `MARGIN_ALARM_MIN`, not a copy that can drift."""
    assert REC["floor"] == float(ci.MARGIN_ALARM_MIN)
    assert ci.MARGIN_ALARM_MIN == 30, (
        "the registered threshold moved; it was stated 2026-10-10 and this "
        "stage does not edit it")


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
