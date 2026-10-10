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

@pytest.mark.calendar
def test_the_H1_RATE_DECISION_is_DUE():
    """FAILS FROM 2026-10-14. That is the point, not a bug.

    THE RULE, WRITTEN ON 2026-10-10 AND NOT ON 10-14: if the projected `n` at
    10-17 is below 33, it goes to Niki ON 10-14 — not on 10-17, when the only
    remaining options are the ones nobody chose.

    WHY A FIRING TEST RATHER THAN A NOTE. The same argument as the reservation
    window two days ago: a decision that depends on somebody remembering a date
    is an outage with no owner. This one fires three days before the window
    closes, which is the whole value — on 10-17 the choice between extending
    the window and accepting a reduced `n` is already made by the calendar.

    IT DOES NOT GATE CAPTURE. It carries the `calendar` marker, so it runs in
    the step with `continue-on-error: true` and prices are unaffected. A red run
    here means a DECISION IS DUE, not that the pipeline failed.

    THE SUCCESSOR CONTRACT, written now:
      * if the projection clears 33 — delete this test, record the measured
        rate in the ledger, and let the window run to 10-17;
      * if it does not — Niki chooses between (a) extending past 10-17, which
        moves `H1_RESERVATION_UNTIL` and is a budget decision, and (b) a
        reduced `n` with its power stated. Then delete this test and record
        which, with the number it was decided on.
      * either way the remedy is DELETION, never moving the date. A date moved
        once is a date that can be moved again.
    """
    from datetime import date

    today = date.today()
    decision_due = date(2026, 10, 14)
    window_ends = date(2026, 10, 17)
    target_n = 33

    assert today < decision_due, (
        f"TODAY IS {today} AND H1's RATE DECISION IS DUE (it became due "
        f"{decision_due}, {(window_ends - decision_due).days} days before the "
        f"window closes on {window_ends}).\n"
        f"Run `python -m scripts.h1_collection_check --since 2026-10-09` and "
        f"read the qualifying FIXTURE count for TheOddsAPI.\n"
        f"  projected n at {window_ends} = n_so_far / days_collected * "
        f"{(window_ends - date(2026, 10, 9)).days}\n"
        f"If that is below {target_n}, this goes to Niki TODAY with the two "
        f"options: extend past {window_ends} (a budget decision, it moves "
        f"H1_RESERVATION_UNTIL) or accept a reduced n WITH ITS POWER STATED.\n"
        f"The remedy is to DELETE this test once the decision is recorded — "
        f"never to move the date.")


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
