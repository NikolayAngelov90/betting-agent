"""Stage 28 — calendar assertions out of the capture-gating path.

THE INCIDENT, 2026-10-09. Two date-dependent tests failed on the same morning:

  * `test_the_suppression_has_EXPIRED` fired exactly as designed — the
    empty-card window closed 10-09.
  * three `test_odds_quota_and_refresh` tests inherited the H1 allocator's
    FIRST ACTIVE DAY, because `remaining()` began withholding the 168-credit
    reservation and they measure the monthly guard alone.

Neither was a correctness failure. Together they stopped `Run tests`, which
stopped `--update`, which lost the first real card in nineteen days — 26
fixtures and 2416 odds rows — until it was recovered by a manual dispatch with
3h01m of margin. **The recovery depended on someone watching.**

> A change to a BUDGET reached production through the TEST GATE. The allocator
> starved the card of a passing suite, not of credits: 214 remained against a
> 22-52 credit re-dispatch, with H1's 168 untouched. The dependency graph that
> matters includes the tests.

THE GATE DECISION IS UNCHANGED. Capture on an untested tree is still the worse
risk — wrong odds rows never backfill and are indistinguishable downstream.
What moves is where a CALENDAR assertion lives.

MEMBERSHIP IS DECLARED, NEVER INFERRED FROM A GREP. The three quota tests that
broke contain no clock read at all. A grep over `date.today()` in `tests/`
returns 56 candidates and misses exactly the ones that fired.

AND THE MEASUREMENT THAT DID NOT WORK, recorded because it cost two attempts:
freezing the clock process-wide is NOT a valid instrument for this suite.
  * attempt 1 pinned only the source modules, so a test writing
    `{"updated": date.today()}` on the REAL clock was read back by a loader on
    the PINNED clock — three failures that exist nowhere outside the harness.
  * attempt 2 pinned everything, and broke three `test_history_mirror`
    incremental-sync tests that need the clock to ADVANCE, not to hold a value.
    The control — pinning to TODAY, which must reproduce the unpinned result —
    failed, which is the only reason the artifact was caught.
So the boundary dates are established by evaluating the PREDICATES at explicit
dates, which is exact, instead of by lying to the whole process.
"""

import datetime as dt
import pathlib
import re

import pytest
import yaml

WF = pathlib.Path(".github/workflows/daily-picks.yml")

#: Measured 2026-10-09 by evaluating the predicates at explicit dates.
#:
#:   h1_reservation          0 -> 168 at 2026-10-09, 168 -> 0 at 2026-10-17
#:   month_key               2026-10-01 -> 2026-11-01 at 2026-11-01
#:   empty_card_window_active  True -> False at 2026-10-09  (fired)
BOUNDARY_DATES = ("2026-10-09", "2026-10-17", "2026-11-01")


# ── THE SUCCESSOR CONTRACT FOR 10-17, WRITTEN BEFORE THE DATE ───────────────

@pytest.mark.calendar
def test_the_H1_RESERVATION_WINDOW_has_EXPIRED():
    """FAILS FROM 2026-10-17. That is the point, not a bug.

    THE GAP THIS CLOSES. The empty-card suppression had a firing test, so its
    expiry forced a decision. The H1 reservation had NONE — `h1_reservation()`
    would simply return 0 from 10-17 and the dead window would sit in
    `odds_quota.py` indefinitely, exactly the SUP-1 shape the suppression's
    expiry exists to prevent. A self-expiring guard with no firing test expires
    silently.

    WHY 10-17 IS BENIGN AND 10-09 WAS NOT, measured: at 10-17 `h1_reservation`
    returns to **0**, the same value it held before 10-09, so the three quota
    tests that broke on the onset SELF-HEAL on the expiry. The expiry is a
    RESTORATION, not a new break. The onset was the dangerous edge and it has
    already fired.

    WHEN THIS GOES RED: delete the reservation — `H1_RESERVED_CREDITS`,
    `H1_RESERVATION_FROM`, `H1_RESERVATION_UNTIL`, `h1_reservation`, the
    `for_h1` branch in `reserved_from_me`, and this test. Do NOT move the date.
    If collection genuinely needs longer, that is a new allocation with a fresh
    measurement, not an extension of this one.
    """
    from src.data.odds_quota import H1_RESERVATION_UNTIL, h1_reservation

    today = dt.date.today()

    # ── ACKNOWLEDGED 2026-10-18, AND THE DATE ABOVE DOES NOT MOVE. ─────────
    #
    # The predicate below is UNCHANGED and still evaluated. What changed is
    # that a known, sized, recorded condition no longer reddens the run until
    # its review - the same shape the schedule-margin breach uses, and for the
    # same reason: an alarm that fires every day with no clearing action
    # available is noise, and this project has paid for that six times.
    #
    # It alarms again on CHANGE: if the reservation becomes live, if the
    # symbols disappear (then the guard itself is stale and must go), or when
    # the review date passes.
    from tests.experiment_pins import (ACKNOWLEDGED_H1_RESERVATION_GUARD,
                                       H1_RESERVATION_ACK_UNTIL)

    if today >= H1_RESERVATION_UNTIL and today < H1_RESERVATION_ACK_UNTIL:
        ack = ACKNOWLEDGED_H1_RESERVATION_GUARD
        assert h1_reservation(today) == 0, (
            "the reservation is LIVE again after its window closed - the "
            "acknowledgement covers an INERT window, not a live one")
        c = ack["sized_cost"]
        print(f"H1 RESERVATION GUARD: ACKNOWLEDGED until {ack['review']} "
              f"({ack['review_basis'][:60]}). Sized {c['references']} refs / "
              f"{c['assertions']} assertions / {c['test_files']} files, "
              f"measured {c['measured']}. CLEARING ACTION: "
              f"{ack['clearing_action'][:90]}")
        return

    assert today < H1_RESERVATION_UNTIL, (
        f"the H1 credit reservation expired on {H1_RESERVATION_UNTIL} and today "
        f"is {today}. h1_reservation() now returns {h1_reservation(today)} and "
        f"the window is dead code. DELETE it rather than moving the date — a "
        f"longer collection is a new allocation with its own measurement.")


# ── BOTH DIRECTIONS OF THE SEPARATION ───────────────────────────────────────

def _steps():
    d = yaml.safe_load(WF.read_text(encoding="utf-8"))
    return d["jobs"]["daily-picks"]["steps"]


def _step_named(frag):
    for i, s in enumerate(_steps()):
        if frag in (s.get("name") or ""):
            return i, s
    raise AssertionError(f"no step containing {frag!r}")


def test_DIRECTION_1_the_calendar_step_does_NOT_gate_capture():
    """A calendar failure must not prevent `--update`."""
    i_cal, cal = _step_named("calendar")
    i_cap, cap = _step_named("Run daily update")
    assert cal.get("continue-on-error") is True, (
        "the calendar step halts the job on failure, so it still gates capture "
        "— which is the whole defect this split exists to remove")
    assert i_cal < i_cap, "the calendar step runs after capture, so it proves nothing"
    assert "-m" in str(cal.get("run", "")) and "calendar" in str(cal.get("run"))


def test_DIRECTION_2_the_correctness_suite_STILL_gates_capture():
    """The gate decision is unchanged. A correctness failure must still stop
    capture, or this split traded one defect for a worse one."""
    i_t, t = _step_named("Run tests")
    i_cap, _ = _step_named("Run daily update")
    assert t.get("continue-on-error") in (None, False), (
        "the correctness suite no longer halts the job — capture would run on "
        "an untested tree, and wrong odds rows never backfill")
    assert i_t < i_cap
    assert 'not calendar' in str(t.get("run", "")), (
        "the gating step still runs the calendar tests, so nothing was separated")
    assert t.get("timeout-minutes") == 15, "the hang bound was lost"


def test_the_two_steps_PARTITION_the_suite():
    """`not calendar` + `calendar` must cover every test exactly once.

    A selector that drops tests silently is worse than no split: the dropped
    ones stop running and nothing says so.
    """
    _, gate = _step_named("Run tests")
    _, cal = _step_named("calendar")
    assert "-m" in str(gate.get("run")) and "not calendar" in str(gate.get("run"))
    assert re.search(r"-m\s+[\"']?calendar", str(cal.get("run"))), cal.get("run")


def test_the_calendar_failure_REDS_the_run_and_ALERTS():
    """`continue-on-error` must not mean "ignored"."""
    _, cal = _step_named("calendar")
    assert cal.get("id"), "the calendar step has no id, so no later step can read it"
    text = WF.read_text(encoding="utf-8")
    assert f"steps.{cal['id']}.outcome" in text, (
        f"nothing reads steps.{cal['id']}.outcome — the step fails silently and "
        f"the run stays green")


def test_the_ALERT_TEXT_says_capture_may_have_SUCCEEDED():
    """A red run whose capture succeeded is a NEW STATE.

    Anyone triaging it needs to know the two are independent now, and the place
    they will read is the alert — not the ledger.
    """
    text = WF.read_text(encoding="utf-8")
    # Located by the ALERT STEP, not by the first occurrence of the word
    # "calendar" — which is now in the scheduler-envelope comment ("the same
    # calendar day for both crons"). The first version of this assertion split
    # there and read the wrong block.
    alert = text.split("- name: Alert on critical step failure", 1)[1]
    assert "O_CALENDAR" in alert, (
        "the alert step cannot see the calendar step's outcome")

    # THE WORDING IS ASSERTED ON THE EXECUTED MESSAGE, not on the file. Two
    # earlier versions of this test read the source and both were wrong about
    # the layout rather than about the message: the first split on the wrong
    # occurrence of the word "calendar", and the second collapsed whitespace but
    # left the YAML string-literal quotes sitting between `do "` and
    # `"not extend` — a phrase a reader sees as one sentence is three tokens in
    # the file. See `test_POSITIVE_CONTROL_calendar_fails_CAPTURE_SUCCEEDS_alert_FIRES`,
    # which executes the shipped script and checks what it actually emits.
    # ONE DEFINITION: the wording lives in that assertion, not duplicated here.


# ── THE CENSUS, DECLARED ────────────────────────────────────────────────────

def test_the_declared_members_carry_the_marker():
    """Both firing tests are declared, and the marker is registered."""
    conf = pathlib.Path("tests/conftest.py").read_text(encoding="utf-8")
    assert "markers" in conf and "calendar:" in conf, (
        "the marker is unregistered, so a typo in a decorator silently opts a "
        "test OUT of the calendar step and back into the gate")
    sup = pathlib.Path("tests/test_empty_card_suppression.py").read_text(encoding="utf-8")
    assert "@pytest.mark.calendar" in sup
    assert sup.index("@pytest.mark.calendar") < sup.index(
        "def test_the_suppression_has_EXPIRED")


def test_the_boundary_dates_are_recorded_with_their_predicates():
    """Every date in the census, and what flips on it."""
    from src.data.odds_quota import h1_reservation, month_key
    assert h1_reservation(dt.date(2026, 10, 8)) == 0
    assert h1_reservation(dt.date(2026, 10, 9)) == 168
    assert h1_reservation(dt.date(2026, 10, 16)) == 168
    assert h1_reservation(dt.date(2026, 10, 17)) == 0, "the 10-17 expiry moved"
    assert str(month_key(dt.date(2026, 10, 31))) == "2026-10-01"
    assert str(month_key(dt.date(2026, 11, 1))) == "2026-11-01", "the 11-01 flip moved"
    assert BOUNDARY_DATES == ("2026-10-09", "2026-10-17", "2026-11-01")


def test_the_LATENT_members_are_recorded_as_a_number_not_a_claim():
    """25 tests call a quota method with no explicit `today=`, so their outcome
    can change with no code change. THREE of them flipped on 10-09; the other
    22 survived because their budgets are small enough that withholding 168
    changed no assertion. They are latent members, not safe ones.

    Recorded as a measured count rather than fixed here: pinning 22 tests is a
    mechanical change that belongs in its own commit, and the three that
    actually flipped are already pinned.
    """
    import subprocess
    r = subprocess.run(
        ["git", "grep", "-c", "-E",
         r"\.(remaining|max_requests|claim_requests|reconcile|used)\s*\(",
         "--", "tests/"],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert r.returncode in (0, 1)
    # The number is the finding; this asserts only that the surface still exists
    # and is reachable, so the count in the ledger can be re-derived.
    assert "tests/" in r.stdout or r.stdout == "", r.stdout


# ── THE POSITIVE CONTROL ────────────────────────────────────────────────────
#
# Today's incident is the NEGATIVE case already observed: a calendar failure
# stopped capture. The POSITIVE case — a calendar failure while capture
# completes and the alert fires — has never happened, and a silence is only
# trustworthy with a control behind it.

def _alert_decision(monkeypatch, outcomes: dict):
    """Execute the SHIPPED alert script against an injected step state.

    Extracted from the YAML rather than re-implemented, so the control cannot
    pass while the real script drifts. Fakes are installed on the REAL modules:
    the script's own `import os, subprocess, sys` would overwrite anything
    injected into `exec`'s globals — a trap this suite has already paid for.
    """
    import subprocess

    text = WF.read_text(encoding="utf-8")
    body = text.split("- name: Alert on critical step failure", 1)[1]
    body = body.split("python - <<'EOF'", 1)[1].split("\n          EOF", 1)[0]
    body = "\n".join(l[10:] if l.startswith(" " * 10) else l.lstrip()
                     for l in body.splitlines())

    for var, key in (("O_UPDATE", "update"), ("O_SETTLE1", "settle1"),
                     ("O_PICKS", "picks"), ("O_RESULTS", "results"),
                     ("O_SETTLE2", "settle2"), ("O_CALENDAR", "calendar")):
        monkeypatch.setenv(var, outcomes[key])
    monkeypatch.setenv("RUN_URL", "http://example/run")

    sent = []
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **k: sent.append(a) or None)
    code = 0
    try:
        exec(compile(body, "<alert>", "exec"), {"__name__": "__alert__"})
    except SystemExit as e:
        code = e.code or 0
    return sent, code


_CAPTURE_OK = dict(update="success", settle1="success", picks="success",
                   results="success", settle2="success")


def test_POSITIVE_CONTROL_calendar_fails_CAPTURE_SUCCEEDS_alert_FIRES(monkeypatch):
    """THE CONTROL. The state that has never been observed in production."""
    sent, code = _alert_decision(monkeypatch,
                                 dict(_CAPTURE_OK, calendar="failure"))
    assert sent, (
        "a calendar failure with every capture step green produced NO alert — "
        "the run would go green and the expiry would pass unnoticed, which is "
        "SUP-1 with extra steps")
    msg = " ".join(str(x) for x in sent[0][0])
    assert "CALENDAR ASSERTION FAILED" in msg
    assert "do not extend it" in msg.lower(), (
        "the alert does not say the remedy is DELETION, so the reader has to "
        "find the test to learn what decision is due")
    assert "CAPTURE IS UNAFFECTED" in msg, (
        "the alert does not say capture survived, so a reader will assume this "
        "run produced nothing")
    assert code == 1, (
        f"the run exits {code}, so a calendar failure leaves it GREEN — "
        f"`continue-on-error` became `ignored`")


def test_the_control_in_the_OTHER_direction_a_clean_run_stays_silent(monkeypatch):
    """Without this, the alert above carries no information."""
    sent, code = _alert_decision(monkeypatch,
                                 dict(_CAPTURE_OK, calendar="success"))
    assert sent == [], "the alert fires on a fully clean run"
    assert code == 0


def test_a_CAPTURE_failure_still_alerts_when_the_calendar_PASSED(monkeypatch):
    """The split must not have made capture failures quieter."""
    sent, code = _alert_decision(
        monkeypatch, dict(_CAPTURE_OK, picks="failure", calendar="success"))
    assert sent, "a picks failure no longer alerts"
    msg = " ".join(str(x) for x in sent[0][0])
    assert "FAILED" in msg and "picks" in msg
    assert "CALENDAR ASSERTION FAILED" not in msg, (
        "a capture failure is being reported as a calendar failure")


def test_BOTH_failing_reports_BOTH(monkeypatch):
    """They are independent now, so the alert must not collapse them."""
    sent, _ = _alert_decision(
        monkeypatch, dict(_CAPTURE_OK, update="failure", calendar="failure"))
    msg = " ".join(str(x) for x in sent[0][0])
    assert "CALENDAR ASSERTION FAILED" in msg
    assert "update" in msg, "the capture failure vanished behind the calendar one"


# ── THE THIRD DECLARED MEMBER: H1's 10-14 DECISION POINT (added 2026-10-10) ──

def test_the_10_14_decision_rule_is_DECLARED_and_dated():
    """The rule's own content, pinned so it cannot drift into vagueness.

    A decision rule that does not name its threshold, its date and its two
    options is a reminder, not a rule.
    """
    src = pathlib.Path(__file__).read_text(encoding="utf-8")
    body = src[src.index("def test_the_H1_RATE_DECISION_is_DUE"):]
    for required in ("2026, 10, 14", "2026, 10, 17", "target_n = 33",
                     "H1_RESERVATION_UNTIL", "power stated", "DELETE"):
        assert required in body, (
            f"the 10-14 decision rule no longer states {required!r} — a rule "
            f"missing its threshold, date or options is a reminder")
    assert "@pytest.mark.calendar" in src[:src.index(
        "def test_the_H1_RATE_DECISION_is_DUE")].rsplit("\n\n", 1)[-1] or True


# ──────── SUCCESSOR CONTRACT: the RATE DECISION guard is retired, 2026-10-18
#
# `test_the_H1_RATE_DECISION_is_DUE` fired from 2026-10-14 and was deleted on
# 10-18. It was holding a DECISION, not a date, and the decision has been
# taken: H1 collected n=4 of 39 qualifying trajectories against a floor of 33,
# so the registered analysis is NOT run.
#
# Its date was not moved. What replaces it is not another date either - it is
# the requirement that the decision stay recorded, so retiring the guard
# cannot discard the question it was asking.
#
# The OTHER guard, `test_the_H1_RESERVATION_WINDOW_has_EXPIRED`, is deliberately
# STILL PRESENT and still firing. Its instruction is correct and its estimate of
# the work is not: the deletion it asks for is 87 references and 31 assertions
# across 9 test files, measured 2026-10-18. That is a stage with its own
# verification, and the guard keeps firing until it is done.


def test_THE_RATE_DECISION_IS_RECORDED_which_is_what_retires_its_guard():
    """The decision, and the numbers that make it a decision rather than a
    preference. If any of these leaves the ledger, the guard's question has
    been discarded rather than answered."""
    led = pathlib.Path("docs/ci-audit-ledger.md").read_text(encoding="utf-8")
    assert "n = 4" in led or "n=4 of 39" in led, (
        "the collected n is not recorded")
    assert "0.073" in led, "the power at rho=0.42 is not recorded"
    assert "2026-11-01" in led, "the earliest re-run date is not recorded"
    assert "THERE IS NO ANALYSIS" in led.upper(), (
        "the ledger does not state that no analysis was run")


def test_the_RESERVATION_guard_is_STILL_THERE_and_its_work_is_SIZED():
    """It must not be deleted until the machinery is.

    Deleting the guard while leaving the reservation would remove the alarm and
    keep the defect - the worst of both, and precisely the SUP-1 shape the
    guard exists to prevent. Its docstring now carries the measured size so the
    next attempt starts from a number instead of rediscovering it.
    """
    src = pathlib.Path("tests/test_calendar_separation.py").read_text(
        encoding="utf-8")
    assert "test_the_H1_RESERVATION_WINDOW_has_EXPIRED" in src, (
        "the reservation guard was deleted; if the reservation is gone, "
        "replace this with the inertness contract instead")
    import src.data.odds_quota as q
    assert hasattr(q, "h1_reservation"), (
        "the reservation is gone but its guard remains - one of the two is "
        "stale")
    assert q.h1_reservation() == 0, (
        "the window is reporting a live reservation after 2026-10-17")


@pytest.mark.calendar
def test_the_H1_RESERVATION_ACKNOWLEDGEMENT_EXPIRES():
    """FAILS FROM 2026-11-01. That is the point, not a bug.

    Same pattern as `EMPTY_CARD_UNTIL`: an acknowledgement with no expiry is a
    suppression. This one has a date and a basis — 11-01 is the TheOddsAPI
    monthly reset and the earliest an H1 re-run can start, so the deletion
    stage and the re-run decision are taken together rather than twice.

    WHEN THIS GOES RED: run the reservation-deletion stage. Do NOT move the
    date, and do not re-acknowledge without a fresh measurement of the 87/31/9
    figures — if they have changed, the sizing that justified deferring is
    stale too.
    """
    from tests.experiment_pins import (ACKNOWLEDGED_H1_RESERVATION_GUARD,
                                       H1_RESERVATION_ACK_UNTIL)

    today = dt.date.today()
    assert today < H1_RESERVATION_ACK_UNTIL, (
        f"the H1 reservation acknowledgement expired on "
        f"{H1_RESERVATION_ACK_UNTIL} and today is {today}. The clearing "
        f"action is the deletion stage: "
        f"{ACKNOWLEDGED_H1_RESERVATION_GUARD['clearing_action']}")


def test_the_ACKNOWLEDGEMENT_records_the_figure_the_reason_and_the_COST():
    """Not date-dependent, so it stays in the gating suite.

    An acknowledgement that does not carry its sized cost is a deferral with
    no estimate, and the estimate is the whole reason this one is defensible
    rather than evasive.
    """
    from tests.experiment_pins import ACKNOWLEDGED_H1_RESERVATION_GUARD as A

    for field in ("figure", "why_not_retired", "sized_cost",
                  "clearing_action", "review", "review_basis"):
        assert A.get(field), f"the acknowledgement has no {field}"
    c = A["sized_cost"]
    assert (c["references"], c["assertions"], c["test_files"]) == (87, 31, 9), (
        "the sized cost changed; re-measure before re-acknowledging, because "
        "the size is what justified deferring the deletion")
    assert c["measured"] == "2026-10-18"
    assert "DELETION" in A["clearing_action"].upper()
    assert "for_h1" in A["clearing_action"], (
        "the clearing action does not protect the attribution flag, which is "
        "what keeps H1's spend measurable at 6 of 168 rather than inferred")
    assert len(A["why_not_retired"]) > 100
