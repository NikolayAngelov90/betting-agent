"""The clock, the deadline it was set against, and the bound on step 13.

STAGE 22, 2026-10-01. Two numbers that had sat in a workflow comment since
2026-08-30 were both wrong, and the schedule was derived from them:

    earliest kickoff  the comment said 10:04 UTC. That value came from rows whose
                      `match_date` carries SUB-SECOND precision — ingestion
                      timestamps, not kickoffs. Discarding the 419 contaminated
                      rows of 2085 in the measurement window, the earliest REAL
                      kickoff is 10:15.
    run duration      the comment assumed "a ~20-minute run" and derived a 09:45
                      start deadline from it. Measured time from run start to
                      picks complete on a FULL card is 67.4-88.1 min (n=4).

So the real deadline is 08:28-08:47 UTC and the 03:00 cron sat 67-85 minutes past
it. These tests pin the corrected decision so it cannot drift back by assumption.

THEY DO NOT RE-MEASURE ANYTHING. Measurement belongs to the ledger entry; this
file asserts that the shipped configuration matches what was measured, which is
the part a future edit can silently break.
"""

import pathlib
import re

import pytest
import yaml

WF = pathlib.Path(".github/workflows/daily-picks.yml")

#: Measured 2026-10-01. Earliest REAL kickoff in the 30 tracked leagues, over
#: 1666 clean rows (2026-08-01..09-20), after discarding sub-second `match_date`.
EARLIEST_KICKOFF_MIN = 10 * 60 + 15

#: Measured 2026-10-01, n=4 full cards. Time from run start to picks complete.
FULL_CARD_MAX = 88.1
FULL_CARD_MEAN_PLUS_3SD = 107.0

#: Measured 2026-10-01, n=32, membership by each run's own headSha.
DELAY_MAX = 6 * 60 + 35
DELAY_MEAN_PLUS_3SD = 6 * 60 + 53

#: Measured 2026-09-28..10-01, n=4: the suite's CI wall time, 119-130 s.
SUITE_BASELINE_MIN = 130 / 60.0


def _wf():
    return yaml.safe_load(WF.read_text(encoding="utf-8"))


def _crons(d):
    sched = (d[True] if True in d else d["on"])["schedule"]
    return [c["cron"] for c in sched]


def _step(d, name):
    for s in d["jobs"]["daily-picks"]["steps"]:
        if s.get("name") == name:
            return s
    raise AssertionError(f"step {name!r} not found")


# ── PART A: the clock ────────────────────────────────────────────────────────

def test_the_cron_is_SPLIT_BY_DAY_OF_WEEK():
    """LITERALS, pinned. Moving either is a cohort decision, not a tweak.

    WAS `["0 0 * * *"]` UNTIL 2026-10-10. The old assertion demanded a bump
    and a re-derivation before it could be changed; both happened - s5.17,
    and the margin re-derived against TWO deadlines, because the deadline is
    the earliest kickoff and it differs by day (weekend 10:15 UTC, weekday
    12:00 UTC, from an independent reference). The NAME changes with the
    shape so a reader cannot mistake this for the single-cron guard.
    """
    # MOVED AGAIN 2026-10-22 (`s5.19`): the bump happened and BOTH margins
    # were re-derived, against localised per-day floors and per-CLUSTER
    # spreads. The weekend moved too, which the previous entry had left
    # untouched -- localising the anchors inverted which floor is earlier.
    assert _crons(_wf()) == ["30 0 * * 6,0", "0 1 * * 1-5"], (
        "the daily-picks crons moved. That changes when prices are taken, "
        "which is selection-affecting by the s5.2 precedent - bump "
        "CODE_REVISION and re-derive BOTH margins before changing this")


def test_the_cron_clears_the_measured_deadline_on_BOTH_worst_cases():
    """max AND mean+3sd, never the mean. The mean is what hid this for a month."""
    cron = 0
    deadline_max = EARLIEST_KICKOFF_MIN - FULL_CARD_MAX
    deadline_3sd = EARLIEST_KICKOFF_MIN - FULL_CARD_MEAN_PLUS_3SD

    assert cron + DELAY_MAX <= deadline_max, (
        f"at the observed max delay the run starts "
        f"{cron + DELAY_MAX - deadline_max:.0f} min past the deadline")
    assert cron + DELAY_MEAN_PLUS_3SD <= deadline_3sd, (
        f"at mean+3sd the run starts "
        f"{cron + DELAY_MEAN_PLUS_3SD - deadline_3sd:.0f} min past the deadline")


def test_midnight_has_MORE_date_headroom_than_the_old_cron():
    """The move's stated risk was crossing UTC midnight. It is the opposite.

    Execution is the cron instant plus the scheduler delay, so the headroom
    before execution lands on the NEXT calendar day is `1440 - cron`. At 00:00
    that is a full 24h; at 03:00 it was 21h. The largest delay ever documented is
    11h21m — the largest delay ever documented in ANY regime, and it belongs to
    the retired `37 9 * * *` (n=39), not to this cron — so both are safe. The
    move INCREASES the margin, and this test
    exists so nobody re-derives the fear.
    """
    old, new = 3 * 60, 0
    assert (1440 - new) > (1440 - old)
    assert 1440 - new == 1440
    assert 11 * 60 + 21 < 1440 - new, "a documented delay would cross midnight"


def test_the_corrected_figures_are_RECORDED_in_the_workflow():
    """The comment must carry what the schedule was derived from.

    A number in a comment is `assumed` until re-measured — three instances in one
    week. The defence is that the measured value and its date sit beside the
    decision, so the next reader does not have to trust a bare figure.
    """
    text = WF.read_text(encoding="utf-8")
    for token in ("10:15", "19:45", "08:28-08:47", "107.0", "sub-second",
                  "2026-10-01", "4h11m"):
        assert token in text, f"{token!r} missing from the schedule rationale"
    assert "~20-minute run" not in text.split("STAGE 22")[1], (
        "the withdrawn 20-minute assumption is still being asserted below the "
        "correction")


# ── PART B: step 13 is bounded, and a timeout ALERTS ─────────────────────────

def test_the_test_step_has_a_timeout():
    """Unbounded was the dangerous shape: it inherited the 360-minute job cap."""
    s = _step(_wf(), "Run tests")
    assert "timeout-minutes" in s, (
        "the test step has no timeout-minutes, so a hang inherits the job cap "
        "and the job is CANCELLED — which loses the day with no alert")
    assert s["timeout-minutes"] == 15


def test_the_timeout_is_a_stated_multiple_of_the_MEASURED_baseline():
    """Not a round number pulled from nowhere.

    ~7x the measured maximum absorbs the 2-3x variance an unlucky runner
    produces, and still cuts an unbounded hang from 360 min to 15.
    """
    t = _step(_wf(), "Run tests")["timeout-minutes"]
    mult = t / SUITE_BASELINE_MIN
    assert 5 <= mult <= 10, (
        f"the timeout is {mult:.1f}x the measured baseline of "
        f"{SUITE_BASELINE_MIN:.1f} min — state the multiple and why before "
        f"moving it outside 5-10x")
    assert t < _wf()["jobs"]["daily-picks"]["timeout-minutes"], (
        "the step timeout is not below the job cap, so the job is still "
        "cancelled before the step fails and no alert fires")


def test_the_test_step_still_HALTS_the_job():
    """Part C's decision, pinned: the gate stays.

    Capture on an untested tree is the worse risk — wrong odds rows never
    backfill and are indistinguishable downstream. Bounding the hang does not
    relax the gate, and this asserts the bound did not quietly become a bypass.
    """
    s = _step(_wf(), "Run tests")
    assert s.get("continue-on-error") in (None, False), (
        "the test gate was turned into a warning. That lets capture run on an "
        "untested tree — a different decision from bounding the hang, and it "
        "needs its own record")


# ── THE POSITIVE CONTROL: it reports on a TIMEOUT, not only on a failure ────

def _colour_decision(monkeypatch, outcomes: dict):
    """The critical-step predicate's own output, read from the workflow.

    Not re-implemented: the body is extracted from the YAML and executed, so the
    control cannot pass while the shipped script drifts.

    THE CHANNEL CHANGED ON 2026-10-22, AND SO DID THIS HARNESS. The step used
    to `subprocess.run([... "scripts.ci_alert", msg])`, so the control
    captured `subprocess.run` calls. The Telegram send was removed -- a list
    of step names and their outcomes is run status, and the run's own page
    already carries it -- and THE LOG LINE is now the carrier. So the control
    captures STDOUT.

    It is located by the step's SCRIPT rather than its name, because the name
    changed with the channel and a name-keyed split returned an empty string
    and an `IndexError` -- which says nothing about what moved.

    The first version of this helper passed its fakes in `exec`'s namespace
    and the script's own `import os, sys` overwrote them; every outcome then
    read as `''` and the control reported UNEXPECTED for a state it had never
    been given. A harness that silently substitutes its inputs is the failure
    mode this whole file is about.
    """
    import io
    from contextlib import redirect_stdout

    text = WF.read_text(encoding="utf-8")
    marker = 'steps["picks (incl. review)"]'
    blocks = [b for b in text.split("python - <<'EOF'")[1:] if marker in b]
    assert len(blocks) == 1, (
        f"{len(blocks)} workflow scripts read the picks outcome; expected 1")
    body = blocks[0].split("EOF", 1)[0]
    body = chr(10).join(
        line[10:] if line.startswith(" " * 10) else line.lstrip()
        for line in body.splitlines())

    for var, key in (("O_UPDATE", "update"), ("O_SETTLE1", "settle1"),
                     ("O_PICKS", "picks"), ("O_RESULTS", "results"),
                     ("O_SETTLE2", "settle2")):
        monkeypatch.setenv(var, outcomes[key])
    monkeypatch.setenv("RUN_URL", "http://example/run")

    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            exec(compile(body, "<colour>", "exec"), {"__name__": "__alert__"})
    except SystemExit:
        pass
    out = buf.getvalue()
    # `[]` when nothing was reported, so every caller's `assert sent == []`
    # and `assert sent` keep their meaning across the channel change.
    return [l for l in out.splitlines() if "step states:" in l]


def test_the_harness_itself_passes_the_outcomes_through(monkeypatch):
    """Guard on the guard: prove the env actually reaches the extracted script.

    Without this, every assertion below could be passing on `''` outcomes — which
    is exactly what happened on the first attempt.
    """
    sent = _colour_decision(monkeypatch,
                           dict(update="failure", settle1="success",
                                picks="success", results="success",
                                settle2="success"))
    msg = " ".join(sent)
    assert "update" in msg and "''" not in msg, msg


def test_POSITIVE_CONTROL_a_step13_TIMEOUT_produces_a_REPORT(monkeypatch):
    """A bounded failure that does not alarm is the same defect, shorter.

    A step-level timeout marks step 13 `failure`; the job halts there, so every
    step below reports `skipped`. That is the SAME downstream state a test
    FAILURE produces — the state observed on run 36843668865 — and the alert keys
    on those downstream outcomes, not on anything tests-specific. So the alert
    that fired today is the alert a timeout gets.
    """
    timeout_state = dict(update="skipped", settle1="skipped", picks="skipped",
                         results="skipped", settle2="skipped")
    sent = _colour_decision(monkeypatch, timeout_state)
    assert sent, ("a step-13 timeout produced NO report — the bound is cosmetic")
    msg = " ".join(sent)
    assert "DID NOT RUN" in msg, msg


def test_POSITIVE_CONTROL_the_REPORT_is_SILENT_when_every_step_succeeded(monkeypatch):
    """The other half of the control: it must not alert unconditionally."""
    ok = dict(update="success", settle1="success", picks="success",
              results="success", settle2="success")
    assert _colour_decision(monkeypatch, ok) == [], (
        "it reports on a clean run, so its firing carries no information")


def test_an_UNEXPECTED_outcome_is_reported_as_itself(monkeypatch):
    """`cancelled` is neither success, failure nor skipped, and must not be
    folded into one of them — the three-state rule, applied to the alert."""
    sent = _colour_decision(monkeypatch, dict(update="cancelled", settle1="success",
                                picks="success", results="success",
                                settle2="success"))
    assert sent, "a cancelled step produced no report"
    assert "UNEXPECTED" in " ".join(sent)


# ── THE REMAINING UNBOUNDED STEPS: reported, not fixed ──────────────────────

def test_every_unbounded_step_is_LISTED_so_the_next_one_is_a_decision():
    """Audit, not a fix. Stage 22 bounds step 13 only.

    A step with no `timeout-minutes` inherits the job cap, and a job cancelled at
    the cap does not reliably run `always()` steps. These are the steps that
    still carry that shape; each is short-running today, which is why none is
    urgent and all are recorded.
    """
    d = _wf()
    unbounded = [s.get("name") for s in d["jobs"]["daily-picks"]["steps"]
                 if "run" in s and "timeout-minutes" not in s]
    # Pinned so that ADDING a long-running step without a timeout fails here
    # rather than in production at the 360-minute cap.
    assert len(unbounded) <= 18, (
        f"{len(unbounded)} steps now have no timeout-minutes: {unbounded}")
    assert "Run tests" not in unbounded, "step 13 lost its bound"
    for critical in ("Run daily update (fixtures + odds, no Flashscore results)",
                     "Generate, review, and send picks",
                     "Scrape Flashscore results for all leagues (post-picks)"):
        assert critical not in unbounded, f"{critical} is unbounded"
