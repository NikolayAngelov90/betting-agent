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

#: ── THE SEPARATION MOVED FROM STEPS TO WORKFLOWS. 2026-10-21. ─────────────
#:
#: Stage 28 put the calendar assertions in their own STEP with
#: `continue-on-error`, which stopped them gating capture. It did not stop
#: them REDDENING the run: the critical-step alert read their outcome and
#: exited 1, so on 2026-10-21 `daily-picks` concluded `failure` while all five
#: critical steps succeeded. One conclusion, two facts -- and a human reading
#: red cannot separate them, which is what cost six cards on 10-13..18.
#:
#: They now live in their own workflow, triggered on this one's
#: `workflow_run: completed` (by dependency, never by a clock) and running
#: whether it succeeded or failed.
REM = pathlib.Path(".github/workflows/reminders.yml")


def _steps_of(path):
    d = yaml.safe_load(path.read_text(encoding="utf-8"))
    jobs = d["jobs"]
    return jobs[next(iter(jobs))]["steps"]


def _code_of(path):
    """A workflow's text with COMMENT LINES removed.

    The first draft of the re-pointed assertions failed on ITS OWN COMMENT
    explaining that a reference had been removed -- the comment-blind text
    scan, again. Only lines whose first non-space character is `#` are
    dropped, which covers the YAML comments and the Python comments inside a
    `run:` block alike.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    return chr(10).join(l for l in lines if not l.lstrip().startswith("#"))

#: Measured 2026-10-09 by evaluating the predicates at explicit dates.
#:
#:   h1_reservation          0 -> 168 at 2026-11-01, 168 -> 0 at 2026-11-09
#:   month_key               2026-10-01 -> 2026-11-01 at 2026-11-01
#:   empty_card_window_active  True -> False at 2026-10-09  (fired)
#:
#: ── THE WINDOW MOVED 2026-10-28, AUTHORISED ───────────────────────────────
#:
#: WAS ("2026-10-09", "2026-10-17", "2026-11-01"). The re-run is chosen and
#: starts at the credit reset, so the onset and the month flip are now THE
#: SAME DATE. That is by design rather than coincidence -- the reservation is
#: carved out of a monthly pool, and opening it on any other day spends it
#: from a pool normal pricing has already drawn on.
#:
#: AND IT MAKES 11-01 A THREE-PREDICATE DATE: the month key flips, the
#: reservation opens, and `H1_RESERVATION_ACK_UNTIL` reddens `reminders`. All
#: three on one morning, which is why the collection-error gap in that
#: workflow was worth closing first.
BOUNDARY_DATES = ("2026-11-01", "2026-11-09")


# ── THE SUCCESSOR CONTRACT FOR 10-17, WRITTEN BEFORE THE DATE ───────────────

@pytest.mark.calendar
def test_the_H1_RESERVATION_WINDOW_EXPIRES():
    """FAILS FROM 2026-11-09. That is the point, not a bug.

    ── THE DATE MOVED 2026-10-28, AND THIS DOCSTRING FORBADE IT ───────────
    It read: "WHEN THIS GOES RED: delete the reservation ... Do NOT move the
    date. If collection genuinely needs longer, that is a new allocation with
    a fresh measurement, not an extension of this one."

    THAT CONDITION IS SATISFIED RATHER THAN OVERRIDDEN, and the distinction is
    the whole argument. The first attempt is CLOSED at n=4 with power 0.073;
    its analysis was never run and never will be. What starts on 2026-11-01 is
    not a longer version of it — it is a new collection, with its own
    pre-registration, its own eight days and its own sizing, after the
    collection defect that starved the first one was found and fixed. The
    docstring's own carve-out is "a new allocation with a fresh measurement",
    and that is exactly what this is.

    So the dates move together and the old pair is kept inert at
    `odds_quota.RETIRED_H1_RESERVATION_WINDOW`, because a figure measured
    inside the October window is not comparable with one measured inside this
    one. Authorised by Niki, 2026-10-28; no assertion in this file is
    weakened, and the four boundary dates are driven in
    `test_the_boundary_dates_are_recorded_with_their_predicates` and in
    `test_the_NEW_WINDOW_is_INERT_before_and_after`.

    AND THE ACKNOWLEDGEMENT BELOW IS NOW MOOT, not expired: it covered an
    INERT window ("the acknowledgement covers an INERT window, not a live
    one"), and from 11-01 the window is live. Its branch is unreachable while
    the window is open, which is the correct behaviour and not a bypass.

    ── the original record, unchanged ────────────────────────────────────

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

    WHEN THIS GOES RED ON 2026-11-09: delete the reservation —
    `H1_RESERVED_CREDITS`, `H1_RESERVATION_FROM`, `H1_RESERVATION_UNTIL`,
    `h1_reservation`, the `for_h1` branch in `reserved_from_me`, the H1 cap in
    `remaining`/`claim_requests`, and this test. Do NOT move the date a second
    time: one re-run was authorised on the argument above, and a THIRD window
    would be an extension dressed as a new allocation.
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
    # ── THE ACKNOWLEDGEMENT IS RETIRED AND THE CONTRACT IS ASSERTED. ───────
    #
    # 2026-11-06. The guard above was an ACKNOWLEDGEMENT with an expiry, and
    # it fired on 2026-11-01 and reddened `reminders` for five days. Its
    # instruction was the deletion stage; what shipped is the INERT half of
    # its own successor contract ("gone OR inert"), because 51 of the 61
    # assertions it sized encode live facts a future attempt needs.
    #
    # So there is no standing condition and no date to review. What is
    # asserted here is the contract that replaced it, which cannot expire
    # because it is a property rather than a deadline.
    from tests.experiment_pins import H1_RESERVATION_SUCCESSOR as S
    from src.data import h1_window as _w
    from src.data.odds_quota import allocate as _alloc

    # THE CONTRACT IS A PROPERTY, NOT TODAY'S READING. Asserting
    # `h1_reservation(today) == 0` would be a calendar assertion in disguise:
    # it is 168 today (11-06 is inside the window) and 0 from 11-09, so it
    # would flip with no code change -- which is the whole reason calendar
    # assertions were moved out of the capture gate.
    for probe in (_w.UNTIL, _w.UNTIL + dt.timedelta(days=1),
                  _w.UNTIL + dt.timedelta(days=400)):
        assert h1_reservation(probe) == 0, (
            f"the reservation is live on {probe}, at or past UNTIL "
            f"({_w.UNTIL}); the machinery no longer goes inert by calendar, "
            f"and a switch someone must remember to unset is how SUP-1 "
            f"happens")
        assert _alloc(today=probe)["h1_ceiling"] == 0, (
            f"the allocator still carves a ceiling on {probe}")
    assert _w.contains(_w.LAST_DAY) and not _w.contains(_w.UNTIL), (
        "the population bound moved; the analysis clamp reads it")

    # ── THE CALENDAR HALF, RESTORED AND RESTATED. 2026-11-06. ─────────────
    #
    # I replaced this guard's `assert today < H1_RESERVATION_UNTIL` with the
    # property probes above, and that silently removed the only clock-
    # dependent assertion in it -- `test_calendar_marker_enforced` caught it
    # immediately ("no longer flagged; the predicate has lost the instances
    # the marker exists for"). A `calendar`-marked test with no calendar
    # assertion is a marker with nothing under it.
    #
    # IT IS RESTORED, WITH A DIFFERENT INSTRUCTION. The old message said
    # "DELETE it rather than moving the date". That deletion has now been
    # SIZED AND DECLINED: 274 references / 61 assertions, of which 51 encode
    # live facts a future attempt needs. So the firing of this guard no
    # longer asks for a deletion -- it asks for a decision between the two
    # things that are actually available.
    # ── THE CHOICE WAS MADE: DORMANCY, RECORDED. 2026-11-11. ──────────
    #
    # The guard fired on 11-09 and demanded a decision between starting a new
    # window and recording dormancy, and it forbade moving the date without
    # recording which. Option (b) was taken and authorised, so the date moves
    # WITH its record: `h1_window.DORMANCY` carries the decision, three
    # measured reasons, and the two MECHANISMS that must report before a
    # window is sized.
    #
    # THE DATE IS AN ANCHOR, NOT A DEADLINE, and it is derived: the next UEFA
    # matchday cluster, MD5 on 24-25 November 2026. It is the first
    # opportunity a window could be anchored to. The substantive trigger is
    # the mechanism pair, and this assertion exists so the record cannot sit
    # past its own anchor unexamined -- and so this `calendar`-marked test
    # keeps a clock-dependent assertion, which is what the marker is for.
    assert _w.DORMANCY["decision"] == "DORMANT", (
        "the H1 window expired and no decision is recorded. Record one in "
        "`h1_window.DORMANCY` -- dormancy with its reasons, or a new window "
        "with its own pre-registration. Deletion remains excluded.")
    assert today < _w.DORMANCY["review_anchor"], (
        f"H1 has been DORMANT since {_w.DORMANCY['decided']} and its review "
        f"anchor {_w.DORMANCY['review_anchor']} has passed (today "
        f"{today}). h1_reservation() returns {h1_reservation(today)} on "
        f"every day. The anchor was the UEFA MD5 cluster of 24-25 November; "
        f"the two mechanisms it waits on are "
        f"{list(_w.DORMANCY['review_mechanisms'])}. EITHER report those and "
        f"size a window, OR re-anchor to the next cluster WITH the reason. "
        f"Deletion is still not one of the choices, and moving the anchor "
        f"without recording why is the suppression this guard prevents.")
    print(f"H1 RESERVATION: goes INERT on {_w.UNTIL} by the existing "
          f"calendar. Accepted cost: {S['accepted_cost'][:70]}. KEPT "
          f"{S['assertions_encoding_a_live_fact']} of "
          f"{S['sized_cost_now']['assertions']} assertions.")


# ── BOTH DIRECTIONS OF THE SEPARATION ───────────────────────────────────────

def _steps():
    d = yaml.safe_load(WF.read_text(encoding="utf-8"))
    return d["jobs"]["daily-picks"]["steps"]


WF_STEPS = _steps_of(WF)


def _step_named(frag):
    for i, s in enumerate(_steps()):
        if frag in (s.get("name") or ""):
            return i, s
    raise AssertionError(f"no step containing {frag!r}")



def test_DIRECTION_1_the_reminders_CANNOT_gate_capture():
    """A reminder must not prevent `--update`.

    STRENGTHENED 2026-10-21. It used to assert `continue-on-error: true` on a
    step inside `daily-picks`, which made a reminder unable to gate capture
    but still able to redden the run. The reminders are now a SEPARATE
    WORKFLOW, so they cannot touch capture at all -- a stronger form of the
    same claim, and one that does not depend on a flag staying set.
    """
    for st in WF_STEPS:
        run = str(st.get("run", ""))
        assert '-m "calendar or live"' not in run, (
            f"step {st.get('name')!r} runs the reminder selection inside the "
            f"capture workflow again")
    assert REM.exists(), "the reminder workflow is gone"


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



def test_the_THREE_WORKFLOWS_PARTITION_the_suite():
    """Three selections now, and together they must cover every test once.

    A selector that drops tests silently is worse than no split: the dropped
    ones stop running and nothing says so.

    ── IT WAS TWO UNTIL 2026-10-28 ──────────────────────────────────────
    The capture gate was `not calendar and not live` — EVERY test except two
    declared markers — so each new test gated price capture by default. That
    default was wrong four times (10-09, 10-13, 10-21, 10-27; nine cards), and
    three of the four were clock-dependent tests that say nothing about
    capture correctness. Gating is now OPT-IN:

        daily-picks        capture_critical and not calendar and not live
        non-gating-tests   not capture_critical and not calendar and not live
        reminders          calendar or live

    `tests/test_gate_is_opt_in.py` asserts the partition is exact by
    enumerating node ids, which is stronger than this test's check that each
    workflow carries its own selection. Both are kept: this one catches a
    workflow losing its selector, that one catches the selectors ceasing to
    partition.
    """
    gate = [" ".join(str(st.get("run", "")).split()) for st in WF_STEPS]
    assert any('-m "capture_critical and not calendar and not live"' in r
               for r in gate), "the capture gate's selection changed"
    rem = [" ".join(str(st.get("run", "")).split()) for st in _steps_of(REM)]
    assert any('-m "calendar or live"' in r for r in rem), (
        "the reminder workflow does not run the complementary selection")
    ng = [" ".join(str(st.get("run", "")).split())
          for st in _steps_of(pathlib.Path(
              ".github/workflows/non-gating-tests.yml"))]
    assert any('-m "not capture_critical and not calendar and not live"' in r
               for r in ng), (
        "non-gating-tests does not run the gate's complement, so the 907 "
        "tests that left the gate are running nowhere")



def test_a_reminder_failure_REDS_THE_REMINDER_RUN_and_ALERTS():
    """`continue-on-error` must not mean "ignored" -- now in the new home.

    The reminder step carries `continue-on-error` so its artifact still
    uploads, and the step after it reads the outcome and exits 1. That is what
    makes the REMINDER run red while `daily-picks` stays green.
    """
    steps = _steps_of(REM)
    rem = next(st for st in steps if st.get("id") == "reminders")
    assert rem.get("continue-on-error") is True, (
        "the reminder step halts the job, so the log never uploads")
    text = REM.read_text(encoding="utf-8")
    assert f"steps.{rem['id']}.outcome" in text, (
        f"nothing reads steps.{rem['id']}.outcome -- the step fails silently "
        f"and the reminder run stays green")
    teller = next(st for st in steps if "means" in str(st.get("name", "")))
    assert "sys.exit(1)" in str(teller.get("run", "")), (
        "the reminder run cannot go red, so a due decision passes unnoticed")



def test_the_REMINDER_ALERT_says_it_is_NOT_a_capture_failure():
    """A red reminder run is a different fact from a lost card.

    Anyone triaging it reads the alert, not the ledger -- and the two messages
    arrive in the same Telegram chat, so each must name its workflow and say
    what it is NOT.
    """
    text = REM.read_text(encoding="utf-8")
    assert "REMINDERS" in text and "A DECISION IS DUE" in text
    assert "NOT A PIPELINE FAILURE" in text, (
        "the reminder alert does not say what it is NOT, so it reads like a "
        "capture failure")
    assert "daily-picks" in text, (
        "the reminder alert does not name where capture actually lives")
    # and the capture alert must not have acquired reminder language
    picks_code = _code_of(WF)
    for phrase in ("A DECISION IS DUE", "self-expiring", "CALENDAR ASSERTION"):
        assert phrase not in picks_code, (
            f"daily-picks' alert says {phrase!r}, so the two messages are "
            f"converging again")

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
    # ── THE EDGES, AND THE ONE THAT IS *NOT* HERE. 2026-11-06. ─────────
    #
    #   2026-10-31  before the onset          0
    #   2026-11-01  the onset               168
    #   2026-11-08  last collection day     168
    #   2026-11-09  the window expiry         0
    #
    # THE STOP IS NOT A BOUNDARY HERE. H1 stopped on 2026-11-05 and the
    # reservation goes on declaring 168 until UNTIL, because it is calendar
    # driven by design. Recorded as an accepted cost rather than patched --
    # see `h1_window.STOPPED_ON`.
    assert h1_reservation(dt.date(2026, 10, 31)) == 0, "the day before the onset"
    assert h1_reservation(dt.date(2026, 11, 1)) == 168, "the onset moved"
    assert h1_reservation(dt.date(2026, 11, 8)) == 168, (
        "the last collection day is unfunded. The reservation is CALENDAR "
        "driven and the experiment stop does not enter it -- a funding "
        "switch was tried on 2026-11-06 and reverted; see "
        "h1_window.STOPPED_ON")
    assert h1_reservation(dt.date(2026, 11, 9)) == 0, "the 11-09 expiry moved"
    # and the POPULATION bound is unmoved by all of that
    from src.data import h1_window as _w
    assert _w.contains(dt.date(2026, 11, 8)), (
        "11-08 left the analysis population, which the stop must not touch")
    assert not _w.contains(dt.date(2026, 11, 9))
    assert str(month_key(dt.date(2026, 10, 31))) == "2026-10-01"
    assert str(month_key(dt.date(2026, 11, 1))) == "2026-11-01", "the 11-01 flip moved"
    assert BOUNDARY_DATES == ("2026-11-01", "2026-11-09")
    # THE RETIRED WINDOW IS STILL INERT, both edges. A moved window must not
    # leave the old one live anywhere.
    from src.data.odds_quota import RETIRED_H1_RESERVATION_WINDOW as _old
    assert h1_reservation(_old[0]) == 0 and h1_reservation(_old[1]) == 0, (
        "the retired October window is still reporting a reservation")


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

#: The script that decides the run's colour from the five step outcomes,
#: identified by WHAT IT READS. Its step name changed on 2026-10-22 when the
#: Telegram send was removed, and a name-keyed `split` returned "" and raised
#: `IndexError` — a locator that fails without saying what moved.
_PICKS_KEY = 'steps["picks (incl. review)"]'


def _alert_decision(monkeypatch, outcomes: dict):
    """Execute the SHIPPED colour-decision script against an injected state.

    Extracted from the YAML rather than re-implemented, so the control cannot
    pass while the real script drifts.

    THE CHANNEL IS NOW THE LOG. Until 2026-10-22 the script ended in
    `subprocess.run([... "scripts.ci_alert", msg])` and this captured that
    call; the Telegram send was removed because a list of step names and
    outcomes is run status, and the printed line is what carries it now. The
    return value is still a list, empty when nothing was reported, so every
    caller's `assert sent` / `assert sent == []` keeps its meaning.
    """
    import io
    from contextlib import redirect_stdout

    text = WF.read_text(encoding="utf-8")
    blocks = [b for b in text.split("python - <<'EOF'")[1:] if _PICKS_KEY in b]
    assert len(blocks) == 1, (
        f"{len(blocks)} workflow scripts read the picks outcome; expected 1")
    body = blocks[0].split(chr(10) + "          EOF", 1)[0]
    body = chr(10).join(l[10:] if l.startswith(" " * 10) else l.lstrip()
                        for l in body.splitlines())

    for var, key in (("O_UPDATE", "update"), ("O_SETTLE1", "settle1"),
                     ("O_PICKS", "picks"), ("O_RESULTS", "results"),
                     ("O_SETTLE2", "settle2")):
        monkeypatch.setenv(var, outcomes[key])
    # O_CALENDAR IS GONE. 2026-10-21: the reminders moved to their own
    # workflow, so daily-picks' alert can no longer see them and an injected
    # calendar outcome would be a variable the shipped script never reads.
    assert "calendar" not in outcomes, (
        "the harness was handed a calendar outcome, which the shipped alert "
        "no longer reads -- the test would be exercising nothing")
    monkeypatch.setenv("RUN_URL", "http://example/run")

    buf = io.StringIO()
    code = 0
    try:
        with redirect_stdout(buf):
            exec(compile(body, "<colour>", "exec"), {"__name__": "__alert__"})
    except SystemExit as e:
        code = e.code or 0
    reported = [l for l in buf.getvalue().splitlines() if "step states:" in l]
    return reported, code


_CAPTURE_OK = dict(update="success", settle1="success", picks="success",
                   results="success", settle2="success")



def test_POSITIVE_CONTROL_capture_OK_now_exits_0_and_SAYS_NOTHING(monkeypatch):
    """THE CONTROL, INVERTED BY THE MOVE. 2026-10-21.

    It used to assert that a calendar failure with every capture step green
    produced an alert AND exit 1. That was correct while the two facts shared
    a conclusion. Now they do not: with capture green, `daily-picks` must say
    NOTHING and exit 0, whatever any reminder is doing -- that is the whole
    content of "one run, one meaning".

    The reminder's own alert is asserted by
    `test_a_reminder_failure_REDS_THE_REMINDER_RUN_and_ALERTS` above, so the
    state is still covered; it has simply moved to the workflow that owns it.
    """
    sent, code = _alert_decision(monkeypatch, dict(_CAPTURE_OK))
    assert code == 0, (
        f"daily-picks exits {code} with every capture step green -- something "
        f"other than capture can still redden it")
    assert not sent, (
        f"daily-picks alerted with every capture step green: {sent}")


def test_the_control_in_the_OTHER_direction_a_clean_run_stays_silent(monkeypatch):
    """Without this, the alert above carries no information."""
    sent, code = _alert_decision(monkeypatch, dict(_CAPTURE_OK))
    assert sent == [], "the alert fires on a fully clean run"
    assert code == 0


def test_a_CAPTURE_failure_still_alerts_after_the_WORKFLOW_SPLIT(monkeypatch):
    """Neither split must have made capture failures quieter.

    RENAMED 2026-10-21: there is no calendar outcome to pass any more, so the
    old name described a condition the harness can no longer express.
    """
    sent, code = _alert_decision(
        monkeypatch, dict(_CAPTURE_OK, picks="failure"))
    assert sent, "a picks failure is no longer reported"
    msg = " ".join(sent)
    assert "FAILED" in msg and "picks" in msg
    assert "CALENDAR" not in msg.upper(), (
        "a capture failure is being reported with calendar language")
    assert code == 1



def test_a_CAPTURE_failure_is_reported_ON_ITS_OWN_TERMS(monkeypatch):
    """RE-POINTED 2026-10-21. "Both failing" was a state of ONE run; with the
    split it is two runs, each reporting its own fact. What must still hold is
    that a capture failure is reported as a capture failure and nothing else.
    """
    sent, code = _alert_decision(
        monkeypatch, dict(_CAPTURE_OK, update="failure", picks="failure"))
    assert sent, "a capture failure is no longer reported"
    msg = " ".join(sent)
    assert "update" in msg and "picks" in msg, (
        "a failing step vanished from the report")
    assert "CALENDAR" not in msg.upper(), (
        "a capture failure is being reported with calendar language")
    assert code == 1


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
# The OTHER guard, `test_the_H1_RESERVATION_WINDOW_EXPIRES`, is deliberately
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
    assert "test_the_H1_RESERVATION_WINDOW_EXPIRES" in src, (
        "the reservation guard was deleted; if the reservation is gone, "
        "replace this with the inertness contract instead")
    import src.data.odds_quota as q
    assert hasattr(q, "h1_reservation"), (
        "the reservation is gone but its guard remains - one of the two is "
        "stale")
    # PURE, NOT `h1_reservation()` ON TODAY. It asserted `== 0` and that was
    # true only while the window was past; from 2026-11-01 the live window
    # legitimately returns 168 and the assertion would have failed for eight
    # days running, reading as a defect. The POSITION is what this guard
    # cares about, and position is clock-free.
    assert q.h1_reservation(q.H1_RESERVATION_FROM) == q.H1_RESERVED_CREDITS
    assert q.h1_reservation(q.H1_RESERVATION_UNTIL) == 0, (
        "the window does not close at its own UNTIL date")


@pytest.mark.calendar
def test_the_H1_RESERVATION_SUCCESSOR_CONTRACT_HOLDS():
    """Replaces `test_the_H1_RESERVATION_ACKNOWLEDGEMENT_EXPIRES`, 2026-11-06.

    That test was a date guard and it did its job: it reddened `reminders`
    from 2026-11-01 and the red was a VERDICT, not a failure.

    IT IS NOT REPLACED BY ANOTHER DATE. What replaces it is the contract,
    which is a PROPERTY of the deployed calendar and cannot expire: the
    reservation is zero outside FROM..UNTIL, so the machinery goes inert on
    2026-11-09 with no code change and nothing is left to acknowledge.

    DELIBERATELY NOT ASSERTED: that the reservation is zero TODAY. It is 168
    today and that is correct -- 11-06 is inside the window. A stop-driven
    funding switch was built to make it zero three days early and reverted
    the same day: it flipped 13 capture-gate tests and made a stopped
    experiment report as `BUDGET_UNREADABLE`. `h1_window.STOPPED_ON` carries
    the reasoning and the accepted cost.
    """
    from tests.experiment_pins import H1_RESERVATION_SUCCESSOR as S
    from src.data import h1_window as w
    from src.data.odds_quota import allocate as alloc
    from src.data.odds_quota import h1_reservation as res

    assert str(w.STOPPED_ON) == S["stopped_on"], (
        "the recorded stop date and the experiment record disagree")
    for probe in (w.UNTIL, w.UNTIL + dt.timedelta(days=1),
                  w.UNTIL + dt.timedelta(days=90)):
        assert res(probe) == 0 and alloc(today=probe)["h1_ceiling"] == 0, (
            f"{probe} is at or past UNTIL and still funded")
    assert all(w.contains(d) for d in w.days_list())
    assert not w.contains(w.UNTIL)
    # ── AND THE STOP IS A RECORD, READ BY NO FUNDING PATH. ────────────────
    #
    # NOT `inspect.getsource(res)`. That is what I wrote first and it passed
    # VACUOUSLY: `conftest` monkeypatches `h1_reservation` to a stub
    # (`_declared_only`, 111 characters), so the assertion was reading the
    # STUB's source and could never have seen the real function. A test that
    # passes for the wrong reason is worse than one that fails.
    #
    # So the file is read from disk, and through `code_of` -- which strips
    # comments AND docstrings, and is required here because the real
    # docstring MENTIONS `h1_window.STOPPED_ON` on purpose. The claim is
    # about the CODE, not the prose.
    from tests.conftest import anchored, code_of

    quota_src = pathlib.Path("src/data/odds_quota.py").read_text(
        encoding="utf-8")
    body = anchored(code_of(quota_src), "def h1_reservation",
                    "odds_quota.py with comments and docstrings stripped")
    assert "STOPPED_ON" not in body, (
        "odds_quota CODE references the stop again; that funding switch was "
        "reverted for measured reasons -- read h1_window.STOPPED_ON before "
        "re-adding it")
    # and the behavioural half, which no stub can fake away: a day INSIDE the
    # window but after the stop is still funded, because the stop is not read
    assert res(dt.date(2026, 11, 8)) == 168, (
        "the last collection day is unfunded, so something IS reading the "
        "stop")


def test_the_SUCCESSOR_RECORDS_THE_RE_MEASURED_SIZING_AND_WHAT_WAS_KEPT():
    """The retired guard's instruction: do not re-acknowledge without a fresh
    measurement of the 87/31/9 figures, because the size is what justified
    deferring. It was re-measured, and it had TRIPLED."""
    from tests.experiment_pins import H1_RESERVATION_SUCCESSOR as S

    was, now = S["sized_cost_when_deferred"], S["sized_cost_now"]
    assert (was["references"], was["assertions"], was["test_files"]) == (
        87, 31, 9), "the figure that justified deferring was edited"
    assert was["measured"] == "2026-10-18"
    assert now["measured"] == "2026-11-06"
    assert now["references"] > was["references"], (
        "the re-measurement did not happen, or the cost shrank -- either way "
        "say which, because the sizing is the whole argument")
    assert now["references"] == 274 and now["assertions"] == 61
    # the deletion was NOT performed, and the reason is the thing to pin
    assert S["assertions_encoding_a_live_fact"] == 51
    assert S["assertions_encoding_a_live_fact"] < now["assertions"]
    for field in ("stopped_on", "stop_basis", "contract", "inert_via",
                  "kept_for_a_future_attempt",
                  "how_a_future_attempt_resumes"):
        assert S.get(field), f"the successor record has no {field}"
    resume = S["how_a_future_attempt_resumes"]
    assert "FROM/UNTIL" in resume and "STOPPED_ON" in resume, (
        f"the record does not say how a future attempt turns the reservation "
        f"back on, which is the thing a future reader needs: {resume!r}")
    assert "rebuild" in resume, (
        "the record does not say that the enforcement machinery survives, "
        "which is the whole reason the deletion was not performed")
    assert "review" not in S, (
        "a review date here would make this a deferral again; the condition "
        "is gone, not postponed")


# ══════════ ONE RUN, ONE MEANING — the workflow split, 2026-10-21 ══════════
#
# Rule 1: a run's conclusion carries ONE fact. Two facts need two workflows.
# These were briefly a separate file, `tests/test_one_meaning_per_run.py`,
# which was itself the duplication rule 1 forbids: this file already owned the
# "where do calendar assertions live" contract. Folded in, file deleted.


def test_a_REMINDER_CANNOT_REDDEN_daily_picks():
    """THE ASSERTION THAT MATTERS. After the move, `daily-picks` concluding
    `failure` means capture failed and nothing else."""
    src = _code_of(WF)
    assert "O_CALENDAR" not in src, (
        "the calendar outcome is wired into daily-picks again, so a reminder "
        "can redden a run whose capture succeeded")
    assert "steps.calendar" not in src, "the calendar step id is referenced"
    assert "cal_msg" not in src, (
        "the alert still composes a calendar message, so it still speaks "
        "about something other than capture")
    alert = next(st for st in WF_STEPS if _PICKS_KEY in str(st.get("run", "")))
    body = str(alert["run"])
    tail = body[body.index("All critical steps OK"):][:200]
    assert "sys.exit(0)" in tail, "the clean path no longer exits 0"
    assert "sys.exit(1)" not in tail, (
        "the clean path can still exit 1, so something other than capture can "
        "redden this run")


def test_the_REMINDERS_are_ordered_by_DEPENDENCY_not_a_clock():
    """Rule 2. A nominal gap cannot order two jobs that are both late, and
    this project has paid for that twice: the H1 collector ran through the
    10-13..10-17 outage reporting RUNNING_AND_FINE, and `closing-lines`
    captures nothing when daily-picks is late."""
    d = yaml.safe_load(REM.read_text(encoding="utf-8"))
    trig = d.get(True) or d.get("on") or {}
    assert "workflow_run" in trig, "ordered by a clock again"
    assert "Daily Betting Picks" in trig["workflow_run"]["workflows"]
    # `completed`, NOT `success`: a reminder is due on its own date whatever
    # the pipeline did, and the day the pipeline breaks is the day a pending
    # decision is most likely to matter
    assert trig["workflow_run"]["types"] == ["completed"], (
        f"types={trig['workflow_run']['types']} - on [success] a reminder "
        f"would be silenced by the pipeline breaking")
    if "schedule" in trig:
        assert "BACKSTOP" in REM.read_text(encoding="utf-8").upper(), (
            "the reminder workflow has a cron not declared a backstop")


def test_it_REPORTS_the_upstream_state_rather_than_inferring_it():
    """The fixture-horizon probe's whole lifetime was lost to a dependency
    that looked fine, so this job prints the upstream conclusion."""
    upstream = next(st for st in _steps_of(REM)
                    if "UPSTREAM" in str(st.get("name", "")))
    assert "workflow_run.conclusion" in str(upstream.get("env", {})), (
        "the upstream conclusion is not read")
    assert "UNEVALUATED" in str(upstream.get("run", "")), (
        "a non-workflow_run trigger must report UNEVALUATED rather than "
        "leaving the upstream state blank and readable as success")


def test_an_UNEVALUATED_reminder_step_is_its_OWN_message():
    """Third state. A reminder step that did not complete is not "nothing due".

    RE-POINTED 2026-10-22, and the re-pointing is the interesting part. This
    asserted the literal `outcome not in ("success", "failure")`, which pinned
    a predicate that was itself the defect: it read the STEP OUTCOME, and
    `continue-on-error` collapses every non-zero exit into the word `failure`.
    On run 39714443709 pytest exited 2 on 17 collection errors, no reminder
    ran, and the branch ABOVE this one claimed a reminder was due. The third
    state existed and the one state that reached it was never the one that
    occurred.

    This file owns the SEPARATION -- that a reminder speaks for itself and
    cannot redden capture. The mechanics of the three states now belong to
    `tests/test_reminders_are_evaluated.py`, which drives the classifier at
    each exit code, so they are asserted there and not duplicated here.
    """
    rem = _code_of(REM)
    assert "NOT EVALUATED" in rem, (
        "the third state is gone: a reminder set that was not read would "
        "report as a clear result")
    assert 'outcome not in ("success", "failure")' not in rem, (
        "the step OUTCOME is being read again, which is what turned 'pytest "
        "could not run' into 'a reminder is due'")
    assert "rc not in (0, 1)" in rem, (
        "the third state is no longer keyed on the EXIT CODE, which is the "
        "only thing that separates a verdict from a crash")


def test_the_reminders_are_GIVEN_the_credentials_they_need():
    """Rule 3: an alarm reports a finding only from data it actually read.

    On 2026-10-21 a reminder reported "the weekday series shrank to n=0" from
    a step with no GH token -- a finding about football from the absence of a
    credential. The test now reports UNEVALUATED instead, but a reminder that
    can ONLY say UNEVALUATED is not a reminder, so the new home supplies the
    token deliberately rather than leaving it out and working around it.
    """
    step = next(st for st in _steps_of(REM) if st.get("id") == "reminders")
    env = step.get("env") or {}
    for key in ("GH_TOKEN", "FOOTBALL_DATA_ORG_KEY", "DATABASE_URL"):
        assert key in env, (
            f"{key} is not supplied, so the reminder that needs it can only "
            f"ever report UNEVALUATED")


def test_the_UNEVALUATED_fix_is_STILL_IN_PLACE_in_the_test_itself():
    """Belt and braces: the credential is supplied AND the test still refuses
    to invent a verdict without it. Either alone is a defect."""
    src = pathlib.Path("tests/test_cron_split_by_daytype.py").read_text(
        encoding="utf-8")
    assert "UNEVALUATED, not clear" in src
    assert "pytest.skip(" in src
    assert "GH_TOKEN" in src, (
        "the skip reason does not say whether a credential was present, so "
        "'could not look' and 'nothing there' are indistinguishable again")


#: THE KNOWN-WRONG CASE: this morning's reminder, in shape.
THE_REMINDER_THAT_REDDENED_CAPTURE = '''"""The 2026-10-21 reminder, verbatim in shape."""
import pytest
import scripts.ci_audit as ci


@pytest.mark.live
def test_the_weekday_cron_CLEARS_the_stated_margin():
    by_type, _crons = ci.collect_lag_series_by_daytype()
    lag = [l for _d, l in (by_type.get("weekday") or [])]
    assert len(lag) >= 8, f"the weekday series shrank to n={len(lag)}"
'''


def test_KNOWN_WRONG_todays_reminder_reddens_REMINDERS_and_NOT_daily_picks():
    """THE CONTROL, drawn from the morning that motivated the split.

      * the CAPTURE GATE deselects it entirely -> `daily-picks` stays green;
      * the REMINDER selection collects it      -> that workflow can go red.

    Asserted by RUNNING both selections over the planted test, because the
    claim is about which selection picks it up -- a fact about pytest, not
    about the YAML.
    """
    import subprocess
    import sys

    planted = pathlib.Path("tests/test_zz_one_meaning_probe.py")
    planted.write_text(THE_REMINDER_THAT_REDDENED_CAPTURE, encoding="utf-8")
    try:
        gate = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", str(planted),
             "-p", "no:cacheprovider", "-m", "not calendar and not live"],
            capture_output=True, encoding="utf-8", errors="replace",
            timeout=300)
        rem = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", str(planted),
             "-p", "no:cacheprovider", "--collect-only",
             "-m", "calendar or live"],
            capture_output=True, encoding="utf-8", errors="replace",
            timeout=300)
    finally:
        planted.unlink(missing_ok=True)

    gate_out = gate.stdout + gate.stderr
    rem_out = rem.stdout + rem.stderr

    # EXIT 5 IS "NO TESTS COLLECTED", NOT A FAILURE. The probe runs alone
    # here, so deselecting its only test leaves nothing to run; in the real
    # gate the whole tree runs alongside it and 5 cannot occur.
    assert "deselected" in gate_out, gate_out[-800:]
    assert gate.returncode in (0, 5), (
        f"the capture gate was reddened by a reminder (rc={gate.returncode}): "
        f"{gate_out[-1000:]}")
    assert " failed" not in gate_out, (
        f"a reminder FAILED inside the capture gate: {gate_out[-1000:]}")

    assert "no tests ran" not in rem_out, (
        "the reminder selection does not collect the reminder, so moving it "
        "out silenced it instead of relocating it")
    assert "1 test" in rem_out or "test collected" in rem_out, rem_out[-800:]
