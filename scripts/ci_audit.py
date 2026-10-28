"""Mechanical half of the daily CI audit. Stdlib only.

    python -m scripts.ci_audit --unaudited          # runs not yet in the ledger
    python -m scripts.ci_audit --since 2026-08-11 --until 2026-08-13
    python -m scripts.ci_audit --run 31482430418    # one run

Stage 14, Part D. `.claude/commands/daily-ci-audit.md` drives this; the command
carries the judgement, this carries the counting.

WHY A SCRIPT AND NOT ONLY A PROMPT
----------------------------------
Part A's manual pass took a day and found 1 BROKEN and 9 DEGRADED runs that CI
had reported as `success`. The value was never in the reading — it was in
looking at every step of every run instead of the tail of the failed ones. That
part is mechanical and should not depend on anyone's stamina.

What is NOT mechanical, and stays in the command: deciding whether a zero is a
defect or a quiet day, and writing the note a future reader needs.

THE ASSERTIONS ARE SELF-CALIBRATING, DELIBERATELY
-------------------------------------------------
Every threshold here is relative to what this pipeline recently did, never to a
hand-maintained list. A hardcoded league list rots; "a unit that produced data
within the last N days produced none today" adapts as coverage changes and stays
silent on a card where nothing was ever expected.

This is the 2026-08-07 audit's lesson applied to alerting: thresholds fitted to
a snapshot are noise generators.

`conclusion: success` is not evidence. Every core step in `daily-picks` carries
`continue-on-error: true`, so a run is green whenever the runner survived.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import pathlib
import re
import subprocess
import sys
from collections import defaultdict
from typing import Dict, List, NamedTuple, Optional
# ── THE REPO ROOT GOES ON sys.path BEFORE ANY `scripts.` IMPORT ────────────
#
# `python scripts/ci_audit.py` puts `scripts/` on sys.path[0]; `python -m
# scripts.ci_audit` puts the repo root there. A module-level
# `from scripts.x import y` therefore works under one invocation and raises
# ModuleNotFoundError at IMPORT under the other.
#
# That shipped on 2026-10-10 and produced THREE reds reported as "CI audit
# alarmed - see the audit step for the run ids" when there were no run ids and
# nothing had been audited for ~13 hours. The same job already invoked
# `python -m scripts.ci_alert`, so the two forms disagreed inside one file and
# only one of them was exercised.
#
# A CRASH BOUNDARY CANNOT CATCH THIS - the import runs before any handler in
# this module is reachable - so the fix is to make the invocation form
# IRRELEVANT rather than to pick one and hope.
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

# Only the labelled percentile table is needed here; it is stdlib-only, so
# this step keeps its no-install property.
from scripts.schedule_deadline import (
    CLEARING_ACTION,
    DEADLINE_FLOOR as _DEADLINE_FLOOR,
    DEADLINE_FLOOR_PROVENANCE as _DEADLINE_FLOOR_PROVENANCE,
    DEADLINE_PERCENTILES as _DEADLINE_PERCENTILES,
    deadline_floor_for as _deadline_floor_for,
    combined_deadline,
    day_type,
    format_prospective,
    format_two_margins,
    format_reference_deadline,
    prospective_margin,
    structural_breach_is_acknowledged,
    STRUCTURAL_ACK,
    two_margins,
    reference_earliest)

LEDGER = pathlib.Path("docs/ci-audit-ledger.md")
LOGS = pathlib.Path("ci_logs")
#: ── EVERY WORKFLOW, AND THE SET IS GUARDED ────────────────────────────────
#:
#: `h1-collect.yml` and `fixture-horizon-probe.yml` were added 2026-10-11, and
#: THE OMISSION WAS STRUCTURAL FOR THE THIRD AND FOURTH TIME. `ci-audit.yml`
#: itself was absent from this tuple until 2026-09-27 and cost eight unrecorded
#: reds; the same shape then repeated for both workflows created on 2026-10-10,
#: because creating a workflow and registering it are two acts and only one of
#: them is forced by anything.
#:
#: `test_every_workflow_on_disk_is_AUDITED_or_EXCLUDED` now makes them one:
#: a new file under `.github/workflows/` fails the suite until it is either
#: listed here or named in `UNAUDITED_WORKFLOWS` with a reason.
WORKFLOWS = ("daily-picks.yml", "closing-lines.yml", "paper-trading-report.yml",
             "ci-audit.yml", "h1-collect.yml", "fixture-horizon-probe.yml",
             # ADDED 2026-10-21 WITH THE WORKFLOW ITSELF, in the same commit.
             # `ci-audit.yml` was created on 2026-09-25 and added here a month
             # later, and eight of its red runs survived unrecorded in the
             # gap. A workflow and its registration ship together or the gap
             # happens again.
             "reminders.yml",
             # ADDED 2026-10-28 WITH THE WORKFLOW ITSELF, same rule, and the
             # guard enforced it: `test_every_workflow_on_disk_is_AUDITED_or_
             # EXCLUDED` failed the moment the file existed and before it had
             # ever run, which is the registration arriving BEFORE the first
             # red rather than a month after it.
             #
             # It carries the 907 tests that no longer gate price capture. Its
             # red means "a defect that does not touch capture" — a third
             # distinct meaning, alongside daily-picks' "capture is at risk"
             # and reminders' "a decision is due".
             "non-gating-tests.yml")

#: Workflow files deliberately outside the audit, each with its reason. Empty
#: today: every workflow is audited. An entry here is a decision, not an
#: oversight, and the guard above is what makes that distinction real.
UNAUDITED_WORKFLOWS: Dict[str, str] = {}

#: Workflows that are LISTED and LEDGERED but never ALARMED on.
#:
#: `ci-audit.yml` was added to WORKFLOWS on 2026-09-27. It had been absent, and
#: THE ABSENCE WAS NOT DELIBERATE — `WORKFLOWS` was last edited 2026-08-24 and
#: `ci-audit.yml` was created 2026-09-25, a month later. There was no reason and
#: no rationale to preserve; the workflow simply did not exist. Eight of its red
#: runs survived unrecorded because a human happened to read them.
#:
#: BUT THE LOOP IS REAL AND MUST NOT BE REINTRODUCED. If the audit alarms on its
#: own verdicts: run N fails -> N+1 sees N's failure, alarms, fails -> N+2 sees
#: N+1's failure -> perpetual red, self-sustaining, with nothing able to clear
#: it. That is CLR-2 exactly: an alarm whose scope is defined by a condition its
#: own remedy cannot end.
#:
#: A one-run lag does not break it — it only delays each link in the chain. What
#: breaks it is the split this file already uses everywhere else: REPORTING and
#: FAILING are different policies. The audit's own runs are reported and take a
#: ledger row; they never enter `alarmed`.
#:
#: The cost is stated rather than hidden: a genuinely broken ci-audit will not
#: alarm about itself. Its failures are visible in the ledger and in the Actions
#: list, which is where the eight were found by hand.
#: `fixture-horizon-probe` joins on 2026-10-11, for a DIFFERENT reason than
#: `ci-audit` and with the same cost stated.
#:
#: `ci-audit` is here to break a feedback loop. The probe has no loop; it is
#: here because ITS RED IS ALREADY ITS ALERT. The probe exits 1 on a legitimate
#: finding — tomorrow's margin is short — and on a refused clock, and it fires
#: its own Telegram alert naming the clearing action. The audit alarming on
#: that again would report "the probe went red", which a reader misreads as
#: "the probe is broken" when the probe is working exactly as designed.
#:
#: THE COST, the same one `ci-audit` pays: a GENUINELY broken probe will not
#: alarm here. It is bounded rather than absent — the probe's own
#: `if: failure()` step alerts on a crash too, and its runs are LISTED and take
#: ledger rows, which is where the eight unrecorded `ci-audit` reds were
#: eventually found by hand.
#:
#: `h1-collect` is deliberately NOT here. It exits 0 on every terminal stop
#: state, so a red there means a crash or a timeout — a real finding about a
#: sized experiment losing a day, and nothing else reports it.
NEVER_ALARM_WORKFLOWS = ("ci-audit", "fixture-horizon-probe")

#: How far back "recently produced data" looks. Not a threshold on the metric —
#: a window on the pipeline's own history.
LOOKBACK_RUNS = 7

ANSI = re.compile(r"\x1b\[[0-9;]*m")


# ─────────────────────────────────────────────────────────── evidence sources

def _sh(*args: str) -> str:
    """Always decode UTF-8, never the locale codec.

    `text=True` alone decodes with the platform encoding. On a Windows console
    that is cp1251, and a CI log containing any byte outside it kills the reader
    thread, leaving `.stdout` as None and the caller writing None to a file.
    Third instance of this class in Stage 14 — the DEL-2 harness and the emoji
    in `ci_alert` were the other two. CI is Linux/UTF-8 and would never have
    shown it.
    """
    r = subprocess.run(args, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    return r.stdout or ""


#: Verdicts that are NOT a result. A row carrying one of these is a note saying
#: "come back later", and until 2026-09-13 nothing came back.
PROVISIONAL_VERDICTS = ("IN_PROGRESS", "UNAUDITABLE")

#: `##[error]` ANNOTATIONS THIS REPOSITORY EMITS ON PURPOSE, each paired with
#: the file that prints it. Identified BY SOURCE, not by how routine the text
#: looks: `test_every_self_emitted_annotation_names_a_REAL_emitter` asserts the
#: named file still contains the pattern, so an entry cannot survive its
#: emitter being deleted and cannot be added for a message nothing prints.
#:
#: These are the known-benign baseline. A detector whose baseline equals its
#: threshold has no dynamic range, which is exactly what happened: every
#: alarming `ci-audit` run emitted two of these and scored DEGRADED for an
#: unexplained absorbed failure.
SELF_EMITTED_ANNOTATIONS = (
    ("::error::audit alarm", "scripts/ci_audit.py"),
    ("::error::CI audit alarmed", ".github/workflows/ci-audit.yml"),
    ("::error::Pick generation", ".github/workflows/daily-picks.yml"),
    # Added 2026-10-12. Both were unregistered, so the probe's own
    # annotations counted as FOREIGN in its audited log. Each is followed by
    # that step exiting non-zero - ALARM exits 1, REFUSED exits 3 - so each
    # explains exactly one exit and the remainder stays honest. A refusal with
    # a SILENT non-zero exit would read as an absorbed failure, which is why
    # the designed refusal is annotated too; the TEXT is what separates it
    # from a finding, not the presence of the annotation.
    ("::error::PROBE ALARM", "scripts/fixture_horizon_probe.py"),
    ("::error::PROBE REFUSED", "scripts/fixture_horizon_probe.py"),
    # Added 2026-10-21 with the workflow that emits it. It is followed
    # immediately by `sys.exit(1)`, so it explains exactly one exit and the
    # remainder stays honest.
    ("::error::REMINDER DUE", ".github/workflows/reminders.yml"),
)

#: ── WORKFLOWS WHOSE RED IS A VERDICT, NOT A FAILURE. 2026-10-21. ──────────
#:
#: `reminders.yml` exists to go red when a self-expiring guard reaches its
#: date. Alarming on that is `fixtures_zero_active` again -- an alarm that
#: fires daily with nothing a reader can do about it. But SILENCING THE
#: WORKFLOW IS NOT AVAILABLE: that is precisely what hid eight `ci-audit` reds
#: and the probe's `BROKEN`, and it is why `NEVER_ALARM_WORKFLOWS` is kept as
#: small as it is.
#:
#: So the red is CLASSIFIED instead, three-valued:
#:
#:   VERDICT         red WITH the registered marker -> does NOT alarm
#:   FAILURE         red WITHOUT it                 -> DOES alarm
#:   UNCLASSIFIABLE  cancelled, empty log, or the marker cannot be read
#:
#: A crashed reminder workflow -- a syntax error, a missing credential, a
#: traceback -- carries no marker and therefore still alarms. Both directions
#: are controlled in `tests/test_red_is_classified_per_workflow.py`.
#:
#: THE MARKER IS THE RENDERED FORM, because that is what THIS TOOL OBSERVES.
#: GitHub consumes the workflow command: the log holds `##[error]X`, and
#: `::error::X` survives only in the `##[group]Run ...` echo of the step's own
#: source. Pinning the source literal is the exact mistake that made the
#: annotation matcher count echoed script text and miss every real annotation
#: -- measured on three runs 2026-10-13.
DELIBERATE_RED_MARKERS = {
    ".github/workflows/reminders.yml": (
        "##[error]REMINDER DUE",
        "a self-expiring guard reached its date; this workflow exists to say "
        "so, and price capture is reported separately by daily-picks"),
}

# ── THE MARKER CERTIFIED A LIE, ONCE, AND IT IS WORTH THE PARAGRAPH ─────────
#
# `reminders.yml` had exactly one run before this was found: 39714443709,
# 2026-10-22 03:00 UTC. It carried `##[error]REMINDER DUE` and a non-zero
# exit, so `classify_red` returned VERDICT -- correctly, by its own rules.
#
# NO REMINDER WAS DUE. pytest had exited 2 on 17 collection errors, six
# reminders were SELECTED AND NONE RAN, and the classifier downstream read
# `steps.reminders.outcome == 'failure'` as "a reminder is due" and printed
# the marker itself. The same selection locally was `6 passed, 1 skipped`.
#
# A REGISTERED MARKER IS ONLY AS HONEST AS THE PREDICATE THAT PRINTS IT. This
# table cannot check that; it can only check that the marker is present. So
# the fix is in the emitter -- it now branches on pytest's EXIT CODE, where 1
# is a verdict and 2/3/4/5 are "not evaluated" with a different marker that is
# deliberately NOT registered here, because that red is a failure.
#
# The audit needed no change to classify it correctly afterwards: a red
# without this marker already returns FAILURE.

#: Where `UNCLASSIFIABLE` goes, stated rather than left to the reader: INSIDE
#: `alarmed`. "The marker could not be read" is not "there was no marker" --
#: the third-state rule -- and a workflow whose red cannot be classified is
#: the case where silence has cost the most here. It is the conservative side
#: and it is the side a cancelled run lands on, which is noisy by design.
UNCLASSIFIABLE_ALARMS = True

#: VERDICT requires the marker AND the non-zero exit it explains. A marker
#: with no exit means the step printed the annotation and then succeeded, so
#: the red came from somewhere else and is NOT explained -- that is the
#: co-occurrence precondition, and ignoring it is what let the partition
#: report a phantom remainder of 1 on every red audit run.
def classify_red(workflow: str, log: str, facts: Dict[str, object]) -> tuple:
    """`(state, why)` for a run that concluded `failure`.

    `workflow` is the FILE PATH, as `WORKFLOWS` spells it, because a `name:`
    can be edited without the file moving.
    """
    entry = DELIBERATE_RED_MARKERS.get(workflow)
    if entry is None:
        return ("FAILURE", "no registered marker for this workflow, so every "
                           "red is a failure")
    marker, why = entry
    if log is None or len(log.strip()) < _MIN_AUDITABLE_LOG_BYTES:
        return ("UNCLASSIFIABLE",
                f"log is empty or too short to read ({0 if log is None else len(log.strip())} "
                f"bytes), so the marker could not be looked for")
    if str(facts.get("run_conclusion") or "").lower() == "cancelled":
        return ("UNCLASSIFIABLE", "the run was cancelled, so it never reached "
                                  "the point where it would have marked itself")
    if marker not in log:
        return ("FAILURE",
                f"red WITHOUT {marker!r} - a crashed reminder workflow must "
                f"still be heard")
    if not facts.get("exit_annotations"):
        return ("UNCLASSIFIABLE",
                f"{marker!r} is present but NO non-zero exit accompanies it, "
                f"so the marker explains nothing and the red is unaccounted "
                f"for")
    return ("VERDICT", why)

# ────────────────────── RESOLVED RUNS (CLR-2, Stage 27, 2026-10-02) ──────────
#
# An alarm cleared by TIME teaches waiting. The 10-01 `DID_NOT_RUN` was fixed in
# `a4d5228` and alarmed eight more times, clearing only when `--since yesterday`
# rolled past the run. CLR-2 requires an action a person can take, and narrowing
# the scope is not it — the scope was narrowed once already, which is how this
# shape arrived.
#
# So a run leaves the alarm's scope BECAUSE A ROW SAYS SO. Recording the
# resolution IS the clearing action, and it changes the predicate.
#
# RECORDED, NOT SILENCING: a resolved run still appears in the table with its
# verdict and its resolution. What changes is only whether it enters `alarmed`.
RESOLVED_RUNS = {
    # run id        (resolving commit, reason)
    # ── THE SIX-DAY OUTAGE, RESOLVED 2026-10-18 BY 7514a50 ─────────────────
    #
    # One finding, six reds, and the partition was IDENTICAL on all six:
    # exit=2 self=1 foreign=0 unexplained_nonzero_exit=1 tracebacks=0
    # steps_failed=[]. `Run tests` failed on a test that asserted a hard-coded
    # date was absent from a rolling 14-day window over a live `gh run list`;
    # the date aged out, and every downstream step was SKIPPED -> the verdict
    # was correctly `DID_NOT_RUN` and the audit correctly alarmed on all six
    # days. 137 fixtures, zero picks.
    #
    # The one unexplained exit is TRUE, not a matcher defect: `Run tests`
    # exits non-zero and emits no registered annotation of its own, while the
    # critical-step alert emits `::error::Pick generation`, which explains the
    # other exit. `steps_failed` is empty because that list is parsed from the
    # workflow's own text, which names only the CORE steps - the known
    # step-name parser gap, not a finding.
    "38389790394": ("7514a50", "six-day outage: the live-state test in the "
                               "gate; marked `live` and the gate narrowed"),
    "38541025164": ("7514a50", "same finding, day 2"),
    "38708288664": ("7514a50", "same finding, day 3"),
    "38875569032": ("7514a50", "same finding, day 4"),
    "38999614321": ("7514a50", "same finding, day 5"),
    "39094759323": ("7514a50", "same finding, day 6 — recovered by a "
                               "dispatch the same morning, 65 picks on a "
                               "68-fixture card, 93 min before kickoff"),
    "36843668865": ("a4d5228",
                    "the month-boundary contradiction in "
                    "test_credit_reading_freshness was bounded; the test gate "
                    "that stopped --update can no longer fire on the 1st"),
    # ── 2026-10-27: THE FOURTH OCCURRENCE OF THE SAME CLASS ────────────────
    #
    # `tests/test_model_freshness.py::test_the_EXIT_CODE_is_0_FRESH_and_1_
    # FINDING` failed `assert 1 == 0` and stopped `Run tests`, so `update`,
    # `settle (pre-picks)` and `picks` were all SKIPPED and the day's card was
    # lost. Recovered by a dispatch at 10:25 the same morning.
    #
    # THE SHAPE, AND IT IS MINE. The test built its fixture from a FROZEN
    # module constant (`NOW = dt.datetime(2026, 10, 22, 12, 0, ...)`) and then
    # asserted on `mf.main(["--quiet"])`, which computes
    # `datetime.now(timezone.utc)` ITSELF and takes no `now` argument. One
    # side frozen, the other live: as real time passed the "fresh" artefact
    # aged beyond the staleness threshold and the exit code turned 1.
    #
    # WHY THE MARKER SCAN DID NOT REFUSE IT, measured 2026-10-28: the scan's
    # frozen-versus-moving logic is right, but it learns which functions are
    # clock-derived from the HAND-MAINTAINED `LIVE_NAMES` / `LIVE_QUALIFIED` /
    # `SEAMLESS_READERS` sets, and NO clock-reading function in `scripts/` has
    # ever been in them -- 16 top-level functions there read the clock with no
    # `now` seam and 0 of the 16 are registered. "A reader nobody registered
    # is invisible" is enumerated in `UNCOVERABLE`; this is that gap
    # materialising, at a cost of one card.
    #
    # Resolved by `b9dd01e`, which makes the fixture clock-RELATIVE
    # (`now = datetime.now(utc)`) so both sides move together -- the
    # discrimination `test_BOTH_SIDES_MOVING_TOGETHER_is_NOT_refused` already
    # names as legitimate.
    "40457295067": ("b9dd01e",
                    "fourth capture-gate card loss: a frozen NOW fixture "
                    "asserted against mf.main(), which reads the clock with "
                    "no seam; the fixture is now clock-relative"),
    # THE MARKER'S FIRST REAL EXERCISE (2026-10-10). Recorded nine alarms late,
    # which is itself the measurement: 37885390196 failed at 04:45 on 10-09 and
    # EVERY ci-audit run from 05:14 on 10-09 to 05:43 on 10-10 alarmed on it —
    # nine consecutive reds, one finding, cause fixed the same day.
    #
    # CHECKABLE REASON, not "fixed": `Run tests` failed on two DATE-DEPENDENT
    # assertions and nothing else — test_the_suppression_has_EXPIRED (the
    # empty-card window closed 10-09, by design) and three
    # test_odds_quota_and_refresh tests that inherited the allocator's first
    # active day. No correctness test failed. 582751d pinned those tests to the
    # regime each measures; 0ed4cad moved every calendar assertion OUT of the
    # capture-gating step, so a date-dependent failure can no longer skip
    # `--update` and lose a card.
    #
    # WHAT IS *NOT* CLAIMED: the gate itself is unchanged, and a CORRECTNESS
    # failure still stops capture. That is the accepted cost, re-affirmed on
    # 10-09 and recorded in the ledger — not something this row resolves.
    "37885390196": ("582751d+0ed4cad",
                    "Run tests failed on two DATE-DEPENDENT assertions only "
                    "(suppression expiry 10-09 + three quota tests inheriting "
                    "the allocator's first day); both pinned to their regime, "
                    "and calendar assertions moved out of the capture gate so "
                    "a calendar failure can no longer skip --update"),
    # ── 2026-10-21: THE SAME SHAPE, A THIRD TIME, AND IT WAS MINE ──────────
    #
    # Partition IDENTICAL to the six-day outage above:
    #   exit=2 self=1 foreign=0 unexplained_nonzero_exit=1 tracebacks=0
    #   steps_failed=[]
    # and the cause is the same class: a test in the CAPTURE-GATING selection
    # that reads LIVE EXTERNAL STATE. 7514a50 resolved that one on 2026-10-18
    # by marking it `live`; I put a new one in on 2026-10-20 in 269aea9, and
    # today was its first CI exposure.
    #
    # TWO assertions failed, both added in the preceding 24 hours, both
    # passing locally and failing only in CI:
    #   * test_the_weekday_cron_CLEARS... called
    #     `collect_lag_series_by_daytype()`, which queries the GitHub API. The
    #     test step has no GH credentials, so the series came back EMPTY and
    #     the n>=8 guard fired. The GUARD was right; where it ran was not.
    #   * test_the_COUNTS_and_the_MEDIAN... pinned finding 13, whose predicate
    #     is `not Path("data/models/feature_list.json").exists()` -- a
    #     WORKING-TREE read. `actions/cache` restores that file in CI and a dev
    #     box does not have it, so the live count was 7 in CI and 8 locally.
    #
    # NOT the pipefail change in 06e54c6: that touched ONE file
    # (paper-trading-report.yml) and this run predates it (headSha b8d91c1).
    #
    # Card status: NOTHING IRREVERSIBLE. The run died at the gate before
    # discovery, pricing or saving -- 0 fixtures, 0 odds, 0 picks -- and was
    # recovered by a dispatch at 08:29Z the same morning.
    "39532208006": ("465146a",
                    "two assertions added in the previous 24h, both in the "
                    "capture-gating selection and both environment-dependent: "
                    "a GitHub-API series read (now marked `live`) and a "
                    "working-tree predicate (now asserted separately with its "
                    "reason, not pinned). Third occurrence of this shape; "
                    "test_no_live_state_in_the_capture_gate added to stop a "
                    "fourth"),
}

#: Matches a ledger row's run id whether or not it is bolded. The original
#: pattern required a bare `| 123… |` and silently skipped every
#: bolded row — which is why the 09-10 pass re-listed six runs that were
#: already audited, and why this function over-reported ever since.
_LEDGER_ROW = re.compile(r"^\|\s*\**\s*(\d{9,})\s*\**\s*\|(.*)$", re.M)


def audited_run_ids() -> set:
    """Run ids carrying a FINAL verdict in the ledger.

    TWO DEFECTS, BOTH FOUND BY USING IT.

    1. It matched only unbolded rows, so `| **34199783169** | **daily-picks** |`
       read as unaudited. On 2026-09-10 that made `--unaudited` report 59 runs of
       which six already had verdicts. Over-reporting wastes work; it does not
       hide anything, which is why it survived.

    2. It counted ANY row as audited, including `IN_PROGRESS` and
       `UNAUDITABLE` — verdicts that explicitly mean "no verdict yet". Run
       34745992077 carried `IN_PROGRESS` on 2026-09-13 and was re-listed only
       because a human asked. **That one hides something**: a provisional
       verdict is permanent by default when the query that finds work treats it
       as finished.

    THE SECOND IS THE THIRD-STATE PROBLEM ONE LEVEL UP. Naming the state stopped
    it reading as health; nothing was obliged to act on the name. A rule that
    fires only when someone remembers it is not a practice.
    """
    if not LEDGER.exists():
        return set()
    out = set()
    for rid, rest in _LEDGER_ROW.findall(LEDGER.read_text(encoding="utf-8")):
        if _is_provisional(rest):
            continue          # re-list it: this row is a request, not a result
        out.add(rid)
    return out


#: `disc[fs=2c/-m fdo=0c/0m af=86c/2m]`, and the older `disc[fs=2c fdo=0c af=86c]`.
_LEDGER_DISC = re.compile(
    r"disc\[\s*fs=([\-0-9]+)c(?:/([\-0-9]+)m)?\s+"
    r"fdo=([\-0-9]+)c(?:/([\-0-9]+)m)?\s+"
    r"af=([\-0-9]+)c(?:/([\-0-9]+)m)?\s*\]")

_LEDGER_WORKFLOW = re.compile(
    r"^\|\s*\**\s*(\d{9,})\s*\**\s*\|\s*\**\s*([a-z-]+)\s*\**\s*\|(.*)$", re.M)


def _n(text: Optional[str]) -> Optional[int]:
    """`'86'` -> 86; `'-'` and absent -> None. `-` means NOT REPORTED."""
    if text is None or text == "-":
        return None
    try:
        return int(text)
    except ValueError:
        return None


def ledger_history(workflow: str, limit: int = None) -> List[Dict[str, object]]:
    """Per-source figures for this workflow's PAST runs, read from the ledger.

    THE AUDIT WAS DESTROYING ITS OWN INPUT. `history` was assembled only from
    the runs in the current pass:

        by_wf = defaultdict(list)
        for r in runs:
            hits = assertions(facts, by_wf[r["workflow"]])
            by_wf[r["workflow"]].append(facts)

    and `--unaudited` excludes every run the ledger already holds. So
    `produced_recently()` could not see the last day a source produced the
    moment that day was recorded, and the per-source discovery check —
    the one written after Flashscore's 88-day silent death — was skipped by its
    own "an empty history means nothing can be said" branch.

    FOUND 2026-09-25, on an outage in its fourth day. It fired on 09-22 only
    because 09-21 happened to be unaudited in the same batch; had 09-21 been
    audited alone it would never have fired at all. An alarm that works once per
    outage, by coincidence of batching, is not an alarm.

        Recording the finding removed the evidence that would fire it again.

    The ledger is the durable record, so history comes from there. Rows are
    returned oldest-first to match the in-pass convention, and `-` is read as
    None — NOT REPORTED is not zero, and conflating them is the collapse this
    file has closed four times already.
    """
    if not LEDGER.exists():
        return []
    rows: List[Dict[str, object]] = []
    for rid, wf, rest in _LEDGER_WORKFLOW.findall(LEDGER.read_text(encoding="utf-8")):
        if wf != workflow:
            continue
        m = _LEDGER_DISC.search(rest)
        if not m:
            continue
        fs_c, fs_m, fdo_c, fdo_m, af_c, af_m = m.groups()
        facts: Dict[str, object] = {}
        for ckey, mkey, cval, mval in (
                ("src_flashscore_fixtures", "src_flashscore_matched", fs_c, fs_m),
                ("src_footballdataorg_fixtures", "src_footballdataorg_matched", fdo_c, fdo_m),
                ("src_apifootball_fixtures", "src_apifootball_matched", af_c, af_m)):
            c, mm = _n(cval), _n(mval)
            if c is not None:
                facts[ckey] = c
            if mm is not None:
                facts[mkey] = mm
        if facts:
            rows.append(facts)
    return rows[-limit:] if limit else rows


def _is_provisional(row_rest: str) -> bool:
    """True when a ledger row's VERDICT CELL is provisional.

    Checked per CELL, never as a substring of the row. The first version tested
    `"IN_PROGRESS" in rest` and immediately re-listed 34745992077 — whose row
    says DEGRADED and whose NOTE says "Was IN_PROGRESS when first audited".

    That is the same error this ledger has recorded three times: a definition
    read as an occurrence. The word appearing in a row is not the row carrying
    that verdict, exactly as a workflow's YAML echoed into a log is not a failed
    step and a test named after a string is not that string occurring.

    A cell counts as provisional when, stripped of bold markers, it BEGINS with
    a provisional verdict — so `IN_PROGRESS` and `UNAUDITABLE — empty log` both
    match while a note mentioning either does not.
    """
    for cell in row_rest.split("|"):
        c = cell.replace("*", "").replace("~", "").strip().upper()
        if any(c.startswith(v) for v in PROVISIONAL_VERDICTS):
            return True
    return False


class RunListing(list):
    """A list of runs that knows whether it is the WHOLE list.

    THE THIRD STATE, NAMED. `returned == limit` does not mean "that is all
    there is" — it means the query stopped counting and nobody can tell which.
    A complete listing and a listing cut off at its ceiling are the same object
    until one of them says so.

    This is the fourth instance of the class in this file and the same one the
    odds path closed three times:

        ``[]`` vs ``None``                 measured-and-empty vs never-measured
        429-with-credits vs 429-with-zero  rate-limited vs exhausted
        no-hits vs no-log                  a clean run vs an unreadable one
        **returned == limit**              **complete vs truncated**

    `list_runs`'s own docstring already recorded this defect for
    `closing-lines` at ``limit=40`` — *"which read as 'no closing-lines runs in
    the window' rather than as a truncated query"* — and it recurred at 250 for
    `daily-picks` on 2026-09-16, hiding 24 unaudited runs. **Knowledge present,
    caller not consulting it.**

    RAISING THE LIMIT MOVES THE CLIFF; NAMING THE STATE REMOVES IT. A bigger
    number is right until the next workflow outgrows it, and it fails the same
    silent way when it does.
    """

    #: Workflows whose query came back at the cap, so their history is UNKNOWN
    #: beyond that point rather than exhausted. `{workflow: limit}`.
    truncated: Dict[str, int]

    def __init__(self, rows=(), truncated=None):
        super().__init__(rows)
        self.truncated = dict(truncated or {})

    @property
    def complete(self) -> bool:
        return not self.truncated

    def warning(self) -> Optional[str]:
        """The sentence a caller must print, or None when the listing is whole."""
        if self.complete:
            return None
        parts = ", ".join(f"{wf} (limit {n})"
                          for wf, n in sorted(self.truncated.items()))
        return (f"TRUNCATED LISTING: {parts} returned exactly as many runs as "
                f"were asked for, so anything older is UNKNOWN, not absent. "
                f"Counts below are a LOWER BOUND. Re-run with a higher --limit "
                f"to see further back.")


def list_runs(since: Optional[str], until: Optional[str],
              limit: int = 250) -> RunListing:
    """`limit` is per workflow and must outrun the busiest one.

    closing-lines fires roughly every two hours, so 40 covered five days and
    silently dropped everything older — which read as "no closing-lines runs in
    the window" rather than as a truncated query.

    THE RETURN VALUE NOW CARRIES WHETHER IT IS COMPLETE. See `RunListing`: a
    workflow that came back at exactly `limit` is recorded as truncated, and
    every caller that reports a COUNT must say so. The filtering below is
    applied AFTER the cap, so `since`/`until` cannot be used to argue the cap
    was not reached — the cap is a property of the query, not of the window.
    """
    out: List[dict] = []
    truncated: Dict[str, int] = {}
    for wf in WORKFLOWS:
        raw = _sh("gh", "run", "list", "--workflow", wf, "--limit", str(limit),
                  "--json", "databaseId,startedAt,conclusion,event,status")
        try:
            rows = json.loads(raw or "[]")
        except json.JSONDecodeError:
            rows = []
        if len(rows) >= limit:
            truncated[wf.replace(".yml", "")] = limit
        for r in rows:
            day = (r.get("startedAt") or "")[:10]
            if since and day < since:
                continue
            if until and day > until:
                continue
            r["workflow"] = wf.replace(".yml", "")
            out.append(r)
    return RunListing(sorted(out, key=lambda r: r.get("startedAt") or ""),
                      truncated)


def workflow_states() -> Optional[Dict[str, str]]:
    """`{workflow name: state}` from GitHub, or None when it cannot be asked.

    THE ONE FAILURE MODE THIS TOOL COULD NOT SEE. Every check here reads RUNS.
    A disabled workflow produces none, so "disabled" and "nothing was
    scheduled" were the same observation — the fifth instance of that shape,
    applied to the scheduler itself rather than to anything it runs.

    Found 2026-09-25 after nothing had run for 7h48m. The answer that day was
    `active` for all four, and it was obtained by a command OUTSIDE this tool —
    which is the gap, independently of the answer.

    `None`, NOT `{}`. An empty mapping would read as "no workflows exist", and
    conflating "could not look" with "found nothing" is the collapse this file
    has now closed five times.
    """
    raw = _sh("gh", "workflow", "list", "--all", "--json", "name,state")
    if not raw.strip():
        return None
    try:
        rows = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(rows, list):
        return None
    return {str(r.get("name")): str(r.get("state")) for r in rows
            if isinstance(r, dict) and r.get("name")}


def fetch_log(run_id: str) -> str:
    """Full log of every step. Cached under ci_logs/ so a re-audit is free."""
    d = LOGS / f"run_{run_id}"
    f = d / "full.log"
    if not f.exists() or f.stat().st_size == 0:
        d.mkdir(parents=True, exist_ok=True)
        f.write_text(_sh("gh", "run", "view", run_id, "--log"),
                     encoding="utf-8", errors="replace")
    return ANSI.sub("", f.read_text(encoding="utf-8", errors="replace"))


# ───────────────────────────────────────────────────────────── counted facts

# ══════════ EVERY PATTERN IS TESTED AT REGISTRATION ══════════════════════════
#
#     2026-10-21. A detector's health must be observable in its QUIET state.
#
# WHY THIS EXISTS. `model_integrity_failures` and `model_unsigned_loads` were
# registered on 2026-10-12, matched their log lines, and COULD NOT FIRE: the
# patterns have no numeric capture group, `int()` raised, the key never reached
# `facts`, and `facts.get(...)` was always falsy. TEN DAYS.
#
# The fallback printed "matched but produced no number ... NOT COUNTED" the
# whole time -- but ONLY WHEN THE PATTERN MATCHED, which needs a real integrity
# failure to happen first. A diagnostic gated on the condition cannot report
# that the detector is incapable of firing.
#
# So every pattern now carries an EXAMPLE LINE it must match, and the self-test
# pushes that line through `extract()` and requires a usable value. A pattern
# that matches but yields nothing fails HERE, on the next CI run, and not on
# the day its condition first occurs.
#
# THE EXAMPLE IS THE THIRD FIELD OF THE REGISTRATION, beside the pattern and
# (for the self-emitted annotations) the emitter. It cannot rot relative to the
# pattern, because the self-test asserts the match in both directions.
PATTERN_EXAMPLES = {
    "picks_saved": "PAPER TRADING: saving 7 pick(s) for 2026-10-21",
    "picks_saved_live": "Saved 7 new picks to the database",
    "observations": "pick_observations: 14 written for today",
    "af_requests": "API-Football update complete (51 requests used)",
    "injuries_saved": "Injury update: saved 86 injuries from 30 fixtures",
    "odds_rows": "TheOddsAPI update complete: 240 odds rows",
    "credits": "theoddsapi used = 334 credits this month",
    "clv_pairs": "12 valid CLV pairs captured",
    "fixtures_created": "API-Football: creating new fixture A vs B",
    "reviews": "Briefing decision [KEEP] for Team A vs Team B",
    "settled": "Settled 11 picks",
    "reminder_collect_errors": (
        "::warning::REMINDERS COLLECTION ERRORS - 3 test collection error(s) "
        "were reported; selected reminders produced no failures. No due "
        "decision was made."),
    "credits_claimed": "credits claimed  : 8",
    "captured": "captured         : 16",
    "capture_missing": "9 captured, 3 missing",
    "no_rows": "result=no_rows league=france/ligue-1",
    "decisions_discarded": "Could not apply briefing decision for 53181",
    "unpriced_fixtures": "UNPRICED FIXTURES: 4 rows with no price",
    "unpriced_check_dead": "UNPRICED FIXTURE CHECK DID NOT RUN",
    "implausible_rows": "IMPLAUSIBLE ATTRIBUTIONS: 2 team rows",
    "implausible_check_dead": "IMPLAUSIBLE ATTRIBUTION CHECK DID NOT RUN",
    "report_sent": "Performance report sent to Telegram",
    "duplicate_fixture_refusals": "cause=duplicate_rows_one_fixture match=53181",
    "picks_run_guard_declined": "PICKS-RUN GUARD: DECLINING to save picks",
    "model_integrity_failures":
        "MODEL INTEGRITY FAILURE path=data/models/ml_models.pkl action=investigate",
    "model_unsigned_loads":
        "MODEL SIGNATURE MISSING path=data/models/elo.pkl action=run --train",
    "experiment_record_built": "EXPERIMENT RECORD: n=1250 settled",
    "fixtures_zero_active":
        "returned 0 fixtures for france/ligue-1 - the page yielded NO ROWS AT "
        "ALL, expected >1 for active season",
    "no_fixtures_at_all": "No fixtures found for 2026-10-21",
    "fixtures_scraped": "Scraped 30 fixtures from france/ligue-1",
    "src_flashscore_fixtures": "Scraped 30 fixtures from france/ligue-1",
    "src_footballdataorg_fixtures":
        "football-data.org: 4 scores updated, 6 new fixtures added",
    "src_apifootball_fixtures": "API-Football: creating new fixture A vs B",
    "src_apifootball_matched":
        "API-Football fixtures 2026-10-21: 0 created, 33 updated",
    "src_flashscore_matched": "Flashscore: 12 fixtures updated",
    "src_footballdataorg_matched": "football-data.org: 4 scores updated",
}

#: Padding, so `extract()` does not refuse the synthetic log as unauditable.
_SELFTEST_PAD = "selftest" * 40


def selftest_patterns(patterns: dict = None, examples: dict = None) -> dict:
    """Apply every registered pattern to its example and require a value.

    Returns `{"passed": [...], "failed": [(key, why)], "no_example": [...]}`.

    THE TWO WAYS A PATTERN CAN BE BROKEN ARE BOTH CHECKED:
      * it does not match its own example -- the example rotted, or the
        pattern was edited and the example was not;
      * it matches and `extract()` still yields nothing -- the
        model-integrity shape, which is the one that cost ten days.
    """
    import io
    from contextlib import redirect_stdout

    patterns = PATTERNS if patterns is None else patterns
    examples = PATTERN_EXAMPLES if examples is None else examples
    passed, failed, missing = [], [], []

    # ── COMPILATION FIRST, AND IT SHORT-CIRCUITS. ──────────────────────────
    # `extract()` applies EVERY pattern, so one uncompilable regex makes it
    # raise on any input at all -- including on another key's example. So the
    # compile check cannot be folded into the loop below: it has to run first,
    # and when it finds anything the extract phase is not attempted, because
    # there is nothing it could report that is not downstream of this.
    uncompilable = []
    for key, pat in patterns.items():
        try:
            re.compile(pat)
        except re.error as e:
            uncompilable.append((key, f"the pattern is not a valid regex: {e}"
                                      f" - extract() cannot run at all while "
                                      f"it is registered"))
    if uncompilable:
        return {"passed": [], "failed": uncompilable,
                "no_example": sorted(set(patterns) - set(examples))}

    for key, pat in patterns.items():
        ex = examples.get(key)
        if ex is None:
            missing.append(key)
            continue
        if not re.search(pat, ex):
            failed.append((key, "the pattern does NOT match its own example"))
            continue
        buf = io.StringIO()
        with redirect_stdout(buf):
            facts = extract(ex + chr(10) + _SELFTEST_PAD)
        if facts.get(key) in (None, 0):
            failed.append((key, f"matched its example and yielded "
                                f"{facts.get(key)!r} - this is the "
                                f"model_integrity shape"))
            continue
        passed.append(key)
    return {"passed": passed, "failed": failed, "no_example": missing}


def format_pattern_selftest(r: dict) -> str:
    out = [f"pattern_selftest[{len(r['passed'])} pass, "
           f"{len(r['failed'])} FAIL, {len(r['no_example'])} no example]"]
    for key, why in r["failed"]:
        out.append(f"  !! {key}: {why}")
    for key in r["no_example"]:
        out.append(f"  .. {key}: no example registered, so its health is "
                   f"UNEVALUATED rather than proven")
    return chr(10).join(out)


#: ── `!!` IS RESERVED FOR FINDINGS. 2026-10-21. ────────────────────────────
#:
#: MEASURED on a real audit run (39630251098's conditions, re-run locally):
#: **10 marked lines, of which 1 was a finding.** The other nine were routine
#: — two `TRUNCATED LISTING`, five `no ledger history`, one `LOWER BOUND`, one
#: `workflow record(s) ... does not own` — and they print on every run. A
#: reader who must filter the marker is doing the marker's job.
#:
#: THE RULE: a line carries `!!` **iff its condition also appends to
#: `alarmed`.** Everything else is `note:` — the information is kept, the
#: marker is not. Nine emitter sites were demoted 2026-10-21; **six** now keep
#: the marker and all six alarm: the pattern self-test, the actionable margin,
#: the structural margin's deterioration, the schedule margin, and — added
#: 2026-10-22 — an unevaluated workflow state and a missing cron for
#: tomorrow's day type.
#:
#: THE RESIDUE OF 2026-10-21 IS RESOLVED, LINE BY LINE. Three of the nine
#: demoted lines said "UNEVALUATED, not safe" or "UNKNOWN" and alarmed on
#: nothing, so their words claimed more than the alarm set delivered. Decided
#: 2026-10-22 on this project's own test for an alarm — rare, actionable, AND
#: CLEARED BY AN ACTION — and the third condition is what separated them:
#:
#:   * A DAY TYPE WITH NO LATENESS SERIES → **REWORDED.** It ends when the
#:     next scheduled run fires, with nobody doing anything, so an alarm here
#:     would be cleared by TIME and CLR-2 exists because that teaches
#:     waiting. The line now says "margin NOT COMPUTED … neither clear nor
#:     breached" and states why it does not alarm.
#:   * `gh workflow list` UNREADABLE → **ALARMS.** It is the only line with
#:     no alternative carrier: a disabled workflow produces no runs and
#:     nothing else here can detect one, so "could not look" leaves a
#:     specific detector blind. Cleared by restoring GH_TOKEN, which is an
#:     action. Zero occurrences to date, so it costs nothing in volume.
#:   * NO PARSEABLE CRON FOR TOMORROW'S DAY TYPE → **ALARMS.** Read as a
#:     condition rather than as a margin, it says `daily-picks` is not
#:     scheduled to run tomorrow. Cleared by editing the workflow.
#: Keys computed by their OWN block further down `extract()`, not by the
#: generic counting loop. Listed by name so the loop skips them outright: a
#: key with two owners is a key whose warning fires on the owner that did not
#: get there first. See the comment at the skip.
_OWNED_BY_A_DEDICATED_BRANCH = frozenset({
    "src_flashscore_fixtures", "src_footballdataorg_fixtures",
    "src_apifootball_fixtures", "src_apifootball_matched",
    "src_flashscore_matched", "src_footballdataorg_matched",
})

PATTERNS = {
    "picks_saved": r"PAPER TRADING: saving (\d+) pick",
    "picks_saved_live": r"Saved (\d+) new pick",
    "observations": r"pick_observations.*?(\d+) written",
    "af_requests": r"API-Football update complete \((\d+) requests used",
    "injuries_saved": r"Injury update: saved (\d+) injuries from (\d+) fixtures",
    "odds_rows": r"TheOddsAPI update complete: (\d+) odds rows",
    "credits": r"= (\d+) credits",
    "clv_pairs": r"(\d+) valid CLV pair",
    "fixtures_created": r"creating new fixture",
    "reviews": r"Briefing decision \[",
    "settled": r"Settled (\d+) picks",
    # ── A GREEN REMINDERS RUN THAT DECIDED NOTHING. 2026-10-28. ───────────
    #
    # `2c537b1` correctly stopped treating `rc == 1` as "a reminder is due":
    # with `--continue-on-collection-errors` pytest also exits 1 for a module
    # it could not import, so the exit code alone conflated a verdict with an
    # import error. Its collection-error branch prints `::warning::` and
    # EXITS 0.
    #
    # MEASURED 2026-10-28: none of the 35 registered patterns matched that
    # warning line, so the run concluded SUCCESS, carried no marker, and was
    # examined by nothing. A reminder module that cannot be imported NEVER
    # FIRES, and the whole point of `reminders.yml` is to fire on a date.
    # PNC-1 in its purest form: the warning is a producer with no consumer.
    #
    # It matters in four days. `H1_RESERVATION_ACK_UNTIL` reddens on
    # 2026-11-01, and a collection error that morning would have passed in
    # silence on a green run.
    "reminder_collect_errors": r"REMINDERS COLLECTION ERRORS - (\d+) test",
    # closing-lines evidence. Without these the audit reported 25 of 27 runs
    # CLEAN against a manual pass that found 9 DEGRADED — it was reading
    # daily-picks vocabulary against closing-lines logs and finding nothing.
    "credits_claimed": r"credits claimed\s*:\s*(\d+)",
    "captured": r"captured\s*:\s*(\d+)",
    "capture_missing": r"captured, (\d+) missing",
    "no_rows": r"result=no_rows",
    # A Claude KEEP/CHANGE verdict that was computed and then thrown away. The
    # manual pass caught these; the first version of this script did not, and
    # that was the single run where the two disagreed. The A1 cascade defect
    # surfaced here for four days as a swallowed NotNullViolation.
    "decisions_discarded": r"Could not apply",
    # Stage 21 follow-up. A fixture with no odds is indistinguishable from one
    # nobody wanted to bet. Alarms only on the identity class (no API-Football
    # id); odds-budget coverage is logged at INFO and deliberately not counted
    # here, because 15 of 18 on 2026-08-29 were budget and would have drowned
    # the 3 that were not.
    "unpriced_fixtures": r"UNPRICED FIXTURES: (\d+) row",
    "unpriced_check_dead": r"UNPRICED FIXTURE CHECK DID NOT RUN",
    # ING-1 step 1. A team row whose fixtures span >1 domestic country.
    "implausible_rows": r"IMPLAUSIBLE ATTRIBUTIONS: (\d+) team row",
    "implausible_check_dead": r"IMPLAUSIBLE ATTRIBUTION CHECK DID NOT RUN",
    "report_sent": r"(?:Performance report sent to Telegram|Settlement report sent to Telegram)",
    # s5.9's OWN refusal, distinct from the ordinary per-match cap. Both log
    # `PICK_REJECTED reason=same_fixture_limit`; only the duplicate case says
    # so, and on 2026-09-04 the cap fired and was nearly recorded as s5.9's
    # first catch.
    "duplicate_fixture_refusals": r"cause=duplicate_rows_one_fixture",
    # PRE-REGISTERED AND CURRENTLY MATCHES NOTHING. The picks-run guard was
    # reverted on 2026-09-04; this pattern waits for its redesign. It is here
    # BEFORE the guard rather than after because the guard shipped once already
    # with a log line this file had no pattern for, and five closing-lines runs
    # scored CLEAN while captures were stopped all day.
    "picks_run_guard_declined": r"PICKS-RUN GUARD: DECLINING",
    # MODEL INTEGRITY, wired 2026-10-12. Both markers were already printed by
    # `src/models/ml_models.py` and read by NOTHING - not this file, not an
    # alert. So the one benefit that does NOT depend on the HMAC key being
    # secret, catching accidental corruption or a key rotation, was itself
    # unobservable: the handler logs at ERROR and the agent continues on
    # Poisson + Elo.
    #
    # They are plain markers rather than `::error::` DELIBERATELY - see the
    # comment at the emitter. A self-emitted annotation with no non-zero exit
    # of its own would let `min(self_emitted, exits)` absorb a genuine
    # unexplained exit.
    "model_integrity_failures": r"MODEL INTEGRITY FAILURE",
    "model_unsigned_loads": r"MODEL SIGNATURE MISSING",
    # The experiment record that replaces the frozen live all-time in the
    # Telegram reports. Registered BEFORE the code that emits it, deliberately:
    # the picks-run guard shipped with a log line this file had no pattern for
    # and five runs scored CLEAN while captures were stopped all day. A report
    # that silently stops carrying the block must be visible here.
    "experiment_record_built": r"EXPERIMENT RECORD: n=(\d+)",
    # Stage 19. The audit was blind to a day that discovered nothing: 2026-08-26
    # analysed 0 fixtures against a card of six real matches and was flagged
    # only for the API-Football suspension. It surfaced because Niki looked at a
    # football calendar.
    #
    # Keyed on the scraper's OWN warning, which already excludes
    # `off_season_leagues` — so a genuinely dormant league does not fire it, and
    # the assertion needs no threshold of its own to get wrong.
    # STAGE 20 Part C. Anchored on "NO ROWS AT ALL", which the scraper now
    # emits only when the PAGE yielded nothing — not when a league simply has
    # no fixtures inside the window it was queried for. The old pattern fired
    # 21 times on 2026-08-27, all false positives, which is how a check gets
    # ignored.
    "fixtures_zero_active": r"returned 0 fixtures for \S+ . the page yielded NO ROWS AT ALL, expected .1 for active season",
    "no_fixtures_at_all": r"No fixtures found for \d{4}-\d{2}-\d{2}",
    "fixtures_scraped": r"Scraped (\d+) fixtures from",
    # PER-SOURCE discovery. The aggregate assertion shipped earlier in Stage 19
    # would NOT have fired on 2026-05-31, the day Flashscore went silent:
    # flashscore=0, football-data.org=0, apifootball=13, so the TOTAL was
    # healthy and nothing alarmed. It then stayed silent for 88 days.
    #
    # A fallback that substitutes silently makes the primary's failure
    # invisible. Redundancy that is not checked PER COMPONENT is not
    # redundancy — it is one working source and two unverified claims. Each
    # source is therefore watched on its own, regardless of the others.
    "src_flashscore_fixtures": r"Scraped (\d+) fixtures from",
    "src_footballdataorg_fixtures": r"football-data\.org: \d+ scores updated, (\d+) new fixtures added",
    "src_apifootball_fixtures": r"API-Football: creating new fixture",
    # MATCHED, not created. A source that fetches 325 fixtures and successfully
    # matches every one onto an existing row CREATES nothing — and scored zero,
    # identical to a source that returned nothing at all. It fired a false
    # positive on every daily-picks run from 2026-09-03 to 09-08.
    #
    # Creation measures NOVELTY, and novelty legitimately falls to zero
    # whenever another source got there first. That is the normal steady state,
    # not a fault. Liveness is `created + matched > 0`.
    "src_apifootball_matched": r"API-Football fixtures [0-9-]+: \d+ created, (\d+) updated",
    "src_flashscore_matched": r"Fuzzy-merged|Flashscore: \d+ fixtures? updated",
    "src_footballdataorg_matched": r"football-data\.org: (\d+) scores updated",
}


#: ── THE COMMAND ECHO IS NOT OUTPUT. 2026-10-23. ──────────────────────────
#:
#: GitHub prints every `run:` block into the log before executing it, inside a
#: `##[group]Run …` / `##[endgroup]` pair and coloured with the ANSI sequence
#: below. Those lines are the workflow's SOURCE, not its output, and matching
#: them reports a definition as an occurrence.
#:
#: THIS HAS NOW COST TWO FINDINGS, both of them false and both about the same
#: pattern. On 2026-10-13 the `step(s) FAILED` matcher read its own emitting
#: line and called two DEGRADED runs BROKEN; the remedy then was to require a
#: letter after the dash, which excludes `… FAILED — " + ", ".join(failed)`.
#: On 2026-10-23 it read a COMMENT I had written inside that same `run:` block
#: — `# It sent "⚠️ Daily picks: step(s) FAILED — update, picks". A list of …`
#: — which has a real step name after the dash and sails through the anchor.
#: The audit reported `core step(s) reported failure: update, picks` for a run
#: whose every critical step SUCCEEDED.
#:
#:     An anchor narrows a pattern. Removing the echo removes the class.
#:
#: AND THE RULE THAT OUTLIVES IT:
#:
#:     A detector that reads logs containing its own source will match
#:     itself. Comment-blindness is a property of the INPUT PATH, not of
#:     the scanner.
#:
#: This project applied comment-blindness to the CODE eleven times --
#: `conftest.code_of` exists for it -- and never once to the LOG, although the
#: log contains the code. The question belongs to the path: strip what is not
#: output BEFORE any pattern runs, in `extract` rather than at each call site,
#: so a caller cannot forget.
#:
#: ASKED IN THE OTHER DIRECTION TOO, once, across every candidate: is there
#: anywhere the audit matches against a file it also WRITES? The census and
#: its verdicts are in `tests/test_source_echo_is_stripped.py`
#: (`SELF_MATCH_CENSUS`) -- `pnc_scan`'s `KNOWN_INSTANCES` was exactly this
#: shape and is guarded; the ledger, the cron parse and the audit's reading of
#: its own run logs are measured clean or clean by design.
#:
#: Pinned to what the CONSUMER observes, as the `##[error]` marker rule
#: already is: the `##[group]Run ` / `##[endgroup]` boundaries, which survive
#: `fetch_log`'s ANSI stripping. An `_ECHO_COLOUR` constant was written first
#: and DELETED when the strip widened to the whole region -- it had no
#: consumer left, and a producer with none is PNC-1 whatever its comment says.


def strip_non_output(log: str) -> str:
    """`log` with everything that is NOT step OUTPUT removed.

    TWO FILTERS, both of the same class: text that is a DEFINITION being read
    as an OCCURRENCE.

      1. GitHub's echoed `run:` blocks (see the note above).
      2. pytest's verbose test-ID lines. On 2026-10-23 the `tracebacks`
         counter read 1 for a `daily-picks` run whose every step succeeded,
         because `tests/test_red_is_classified_per_workflow.py` is
         PARAMETRISED ON THE STRING "Traceback (most recent call last)" and
         pytest prints that parameter in the test's name. A test fixture
         describing a traceback was counted as one.

    A pytest `-v` progress line is `tests/x.py::test_y PASSED [ 12%]`: every
    character of it is a test name or a parameter, never program output. A
    real traceback inside a FAILED test appears on its own lines and survives.


    Conservative on both failure modes: an unterminated group stops stripping
    at the next step boundary rather than eating the rest of the log, and a
    line is dropped on the colour alone only when it is INSIDE a group.
    """
    if not log:
        return log or ""
    out, in_echo = [], False
    for line in log.splitlines(keepends=True):
        if "##[group]Run " in line:
            in_echo = True
            continue
        # pytest TEST-ID LINES. The log is `job<TAB>step<TAB>timestamp text`,
        # so the test id is looked for after the last tab.
        _tail = line.rsplit("	", 1)[-1].lstrip()
        _tail = _tail.split(" ", 1)[-1].lstrip() if _tail[:2].isdigit() else _tail
        if _tail.startswith("tests/") and "::" in _tail:
            continue
        if in_echo:
            if "##[endgroup]" in line:
                in_echo = False
                continue
            # ── EVERY LINE IN THE REGION, NOT ONLY THE COLOURED ONES. ─────
            #
            # The first version dropped a line only if it carried the echo
            # colour, and that version PASSED a direct test and CHANGED
            # NOTHING in production: `fetch_log` runs `ANSI.sub("", ...)`
            # before `extract` ever sees the log, so there is no colour left
            # to match. I had tested the function against a raw `gh` log and
            # called it a fix for the pipeline.
            #
            #     Testing the function is not testing the path the data takes.
            #
            # `##[group]Run …` to `##[endgroup]` IS the echo region by
            # definition -- the command, then `shell:` and `env:` -- and no
            # step OUTPUT appears inside it. Only `Run ` groups are matched,
            # so `##[group]Operating System` and its siblings are untouched.
            continue
        out.append(line)
    return "".join(out)


def extract(log: str) -> Dict[str, object]:
    """Counts, not adjectives.

    NON-OUTPUT IS STRIPPED FIRST. See `strip_non_output`: every
    pattern below would otherwise be able to match the workflow's own source,
    and two have.
    """
    log = strip_non_output(log)
    f: Dict[str, object] = {}
    for key, pat in PATTERNS.items():
        ms = re.findall(pat, log)
        if not ms:
            continue
        # `src_apifootball_fixtures` added 2026-09-27. The warning offered two
        # remedies; this is the right one. Its pattern —
        # "API-Football: creating new fixture" — carries NO NUMBER, because the
        # line is emitted once per fixture, so the count IS the occurrence count.
        # Inventing a capture group would mean inventing a number. The key was
        # already being set correctly 40 lines below; the warning fired only
        # because the generic loop reached it first and `int()` raised on a
        # whole-match string. It was an inverted indicator: it printed
        # NOT COUNTED exactly when API-Football WAS creating fixtures.
        # ── KEYS A DEDICATED BRANCH OWNS ARE SKIPPED HERE. 2026-10-21. ──────
        #
        # THIRD RECURRENCE IN THIS ONE LIST. `unpriced_check_dead` was added
        # after it was silently dropped for two days;
        # `src_apifootball_fixtures` after it printed NOT COUNTED exactly when
        # API-Football WAS creating fixtures; and `src_flashscore_matched`
        # does it today, on `Flashscore: 12 fixtures updated` -- an ORDINARY
        # line on every run that scrapes Flashscore.
        #
        # THAT NOISE IS NOT COSMETIC. The same message is how a dead detector
        # reports itself, and the model-integrity alarms sat dead for TEN DAYS
        # behind it. A benign routine case must not share a message with a
        # rare real one, so the family is suppressed AT THE SOURCE rather than
        # relabelled: the generic loop has no business computing a key that
        # the per-source block below owns outright.
        if key in _OWNED_BY_A_DEDICATED_BRANCH:
            continue
        if key in ("fixtures_created", "reviews", "no_rows",
                   "src_apifootball_fixtures",
                   "decisions_discarded", "fixtures_zero_active",
                   "no_fixtures_at_all", "unpriced_check_dead",
                   "implausible_check_dead",
                   "duplicate_fixture_refusals", "picks_run_guard_declined",
                   "report_sent",
                   # ── ADDED 2026-10-21, AND THESE TWO COULD NOT FIRE ──────
                   #
                   # Found by the surface census, which asked which surfaces
                   # had ever been proved CAPABLE of firing. Both patterns
                   # match plain markers with no numeric capture group, so
                   # `int()` raised, the key never reached `facts`, and
                   # `facts.get("model_integrity_failures")` was always
                   # falsy. The model-integrity and model-signature alarms
                   # were REGISTERED, PATTERN-MATCHED AND INCAPABLE OF
                   # ALARMING since they were written on 2026-10-12.
                   #
                   # The fallback was telling us: it prints "matched but
                   # produced no number ... NOT COUNTED" — but only when the
                   # pattern matches, which needs a real integrity failure to
                   # happen first. The same shape as
                   # `unpriced_check_dead`, named three lines below, which
                   # this list was extended for once already.
                   "model_integrity_failures", "model_unsigned_loads"):
            f[key] = len(ms)
        elif key == "injuries_saved":
            f["injuries_saved"] = int(ms[-1][0])
            f["injury_fixtures"] = int(ms[-1][1])
        else:
            try:
                f[key] = int(ms[-1] if isinstance(ms[-1], str) else ms[-1][0])
            except (TypeError, ValueError):
                # A FALLBACK MUST RECORD THAT IT FIRED. This `pass` silently
                # dropped `unpriced_check_dead` for two days: the pattern has
                # no numeric capture group, so int() raised, the key never
                # reached `facts`, and the assertion built to catch a dead
                # check was itself dead. A pattern that matches but cannot be
                # counted is a bug in the pattern, and it now says so.
                print(f"  ci_audit: pattern {key!r} matched but produced no "
                      f"number — add it to the count-style keys above, or give "
                      f"it a numeric capture group. NOT COUNTED.")
    # Per-source: sum every occurrence (Flashscore logs one line per league),
    # count occurrences for API-Football (one line per fixture created).
    # Only set when the run ACTUALLY attempted fixture discovery. A
    # closing-lines run does not scrape fixtures, and reporting `fs=0` there
    # would conflate "did not report" with "reported nothing" — the exact
    # distinction this summary exists to preserve.
    if "Scraping fixtures:" in log or "fixtures from" in log:
        f["src_flashscore_fixtures"] = sum(
            int(x) for x in re.findall(PATTERNS["src_flashscore_fixtures"], log))
    _fdo = re.findall(PATTERNS["src_footballdataorg_fixtures"], log)
    if _fdo:
        f["src_footballdataorg_fixtures"] = sum(int(x) for x in _fdo)
    if "API-Football" in log:
        f["src_apifootball_fixtures"] = len(
            re.findall(PATTERNS["src_apifootball_fixtures"], log))
    # The matched counterparts. Summed, because a run reports one line per date
    # and a source is alive if it matched anything on any of them.
    _afm = re.findall(PATTERNS["src_apifootball_matched"], log)
    if _afm:
        f["src_apifootball_matched"] = sum(int(x) for x in _afm)
    _fsm = re.findall(PATTERNS["src_flashscore_matched"], log)
    if _fsm:
        f["src_flashscore_matched"] = len(_fsm)
    _fdm = re.findall(PATTERNS["src_footballdataorg_matched"], log)
    if _fdm:
        f["src_footballdataorg_matched"] = sum(int(x) for x in _fdm)

    # ---- DEL-3: per-report sequence integrity ---------------------------
    # Parsed as a RECORD, not counted as an occurrence. A report's delivery is
    # one fact with several fields, and the three assertions below each read a
    # different field. Counting matches would collapse them back together.
    #
    # `REPORT_DELIVERY report=<name> chunks=N sent=K failed=<list|none>
    #  terminator=<yes|no> attempts=N`
    _rd = re.findall(
        r"REPORT_DELIVERY report=(.+?) chunks=(\d+) sent=(\d+) "
        r"failed=(\S+) terminator=(yes|no) attempts=(\d+)", log)
    if _rd:
        f["report_deliveries"] = [
            {"report": r[0], "chunks": int(r[1]), "sent": int(r[2]),
             "failed": ([] if r[3] == "none"
                        else [int(x) for x in r[3].split(",") if x.isdigit()]),
             "terminator": r[4] == "yes", "attempts": int(r[5])}
            for r in _rd]
    f["report_incomplete"] = len(re.findall(r"REPORT INCOMPLETE:", log))

    # ---- s5.13: the resolution record, split BY STEP ---------------------
    # `TEAM_RESOLVE name=... step=... team=...`. Counted per step, never as a
    # total, for the reason every other split here exists: the total is the
    # number that cannot answer the question. On 2026-09-14/15 the creations
    # AND the step-2 interceptions were both zero, and a total of "0 new rows"
    # reads as success while being equally consistent with nothing having been
    # attempted. These two counts separate those.
    _tr = re.findall(r"TEAM_RESOLVE name=(?:'[^']*'|\"[^\"]*\") step=(\w+)", log)
    if _tr:
        steps: Dict[str, int] = {}
        for st in _tr:
            steps[st] = steps.get(st, 0) + 1
        f["team_resolve_steps"] = steps

    _fx = re.findall(PATTERNS["fixtures_scraped"], log)
    if _fx:
        f["fixtures_scraped"] = sum(int(x) for x in _fx)
        f["fixture_attempts"] = len(_fx)
    f["errors"] = len(re.findall(r"\| ERROR +\|", log))
    f["tracebacks"] = len(re.findall(r"Traceback \(most recent call last\)", log))
    f["account_suspended"] = "account suspended" in log
    f["telegram_sent"] = len(re.findall(r"Telegram message sent", log))
    # A FAILED ATTEMPT IS NOT A LOST MESSAGE — and it was, until DEL-3 added
    # retry on 2026-09-17. On 09-19 one chunk of five timed out, the retry
    # recovered it, all five were delivered, and this counter reported
    # "1 alert failed to deliver". THE COUNTER PREDATES THE MECHANISM IT
    # COUNTS, which is the third instance here after `af=0` and
    # `fixtures_zero_active` — and both of those cried wolf for days first.
    #
    # Three states, kept apart rather than summed:
    #   LOST        a part the retry did not recover, or a DEL-1 alert that
    #               exhausted its attempts. This is the only one that alarms.
    #   RECOVERED   an attempt failed and a later attempt succeeded.
    #   UNEXPLAINED a raw failure no REPORT_DELIVERY or ALERT line accounts
    #               for — reported as UNKNOWN, never silently as either.
    f["telegram_attempts_failed"] = len(re.findall(
        r"Failed to send Telegram message", log))
    f["telegram_alerts_lost"] = len(re.findall(
        r"ALERT NOT DELIVERED", log, re.I))

    # DERIVED AFTER ITS INPUTS, and the first version was not: this block sat
    # above the telegram counters, read `telegram_attempts_failed` before it
    # existed, and silently reported 0 recovered. A derived value computed
    # ahead of its inputs is a zero that looks like a measurement.
    _rd_list = f.get("report_deliveries") or []
    _parts_lost = sum(len(d["failed"]) for d in _rd_list)
    # Attempts beyond one per chunk ARE retries, and each retry implies a prior
    # failed attempt the sequence record already accounts for.
    _retries = sum(max(0, d["attempts"] - d["chunks"]) for d in _rd_list)
    _raw = f["telegram_attempts_failed"]
    f["telegram_lost"] = _parts_lost + f["telegram_alerts_lost"]
    f["telegram_recovered"] = min(_raw, _retries)
    f["telegram_unexplained"] = max(
        0, _raw - _retries - _parts_lost - f["telegram_alerts_lost"])
    # Anchored to start with a STEP NAME, not a quote. GitHub echoes each
    # `run:` block into the log, so the workflow's own source line —
    #   msg = ("... step(s) FAILED — " + ", ".join(failed)
    # — matches a naive pattern and reports every run BROKEN. Found by running
    # this script against the window the manual pass had already audited: it
    # called two DEGRADED runs BROKEN. A definition is not an occurrence.
    # ══ A COSMETIC NORMALISATION OF TEXT A DETECTOR MATCHES IS A CHANGE TO
    #    THE DETECTOR. Recorded 2026-10-22, from a near-miss. ══════════════
    #
    # Removing an unrelated Telegram send from `daily-picks.yml`, the em dash
    # in `step(s) FAILED —` was normalised to a hyphen. The pattern below
    # required the em dash and the workflow's text was its ONLY source, so
    # that one character would have emptied `steps_failed` and
    # `steps_not_run` **permanently** -- two detectors gone, and NO TEST
    # WOULD HAVE FAILED, because a regex that stops matching reports nothing
    # rather than erroring. It was caught by a test asserting the audit can
    # still read the line, not by reading the diff.
    #
    # THE RULE, in the same family as pinning markers to `##[error]` rather
    # than to the producer's `::error::`:
    #
    #   * pin to WHAT THE CONSUMER OBSERVES, not to what the producer writes;
    #   * treat every edit to matched text as an edit to the MECHANISM, and
    #     test the mechanism in the same change;
    #   * where the text is outside this file's control, accept both forms,
    #     so the detector does not depend on a character nobody knows is
    #     load-bearing.
    #
    # The workflow keeps the em dash and says there that it is load-bearing;
    # this pattern no longer depends on which one arrives.
    f["steps_failed"] = re.findall(
        r"step\(s\) FAILED [—-] ([A-Za-z][^\n]{0,120})", log)
    # SKIPPED IS THE THIRD STATE, and it is the one that occurred. Run
    # 35498465743 (2026-09-20) halted at `Run tests` — the only core step with
    # no continue-on-error — so update, settle and picks were SKIPPED, the
    # workflow printed `All critical steps OK`, and this tool scored the run
    # CLEAN. A run whose core steps did not execute is neither clean nor
    # broken; nothing crashed and nothing ran.
    #
    # Read from BOTH shapes, because the ledger must stay readable across the
    # workflow change that introduced the second one:
    #   old  All critical steps OK: {'picks (incl. review)': 'skipped', ...}
    #   new  step(s) DID NOT RUN — picks (incl. review), update
    f["steps_not_run"] = re.findall(
        r"step\(s\) DID NOT RUN [—-] ([A-Za-z][^\n]{0,120})", log)
    if not f["steps_not_run"]:
        for blob in re.findall(r"All critical steps OK: (\{[^\n}]{0,400}\})", log):
            named = [n for n, o in re.findall(r"'([^']+)':\s*'([^']+)'", blob)
                     if o == "skipped"]
            if named:
                f["steps_not_run"].append(", ".join(named))

    # AN ABSORBED FAILURE, WHICH WAS THE LAST OPAQUE SLICE OF THIS BLIND SPOT.
    #
    # A step's `outcome` is NOT exposed by the REST API — the fields are
    # completed_at, conclusion, name, number, started_at, status. Under
    # `continue-on-error` a failing step therefore reports `conclusion: success`
    # and post-hoc triage from the API alone cannot tell it from a passing one.
    # `daily-picks` has NINE such steps. Five echo their outcome into an `env:`
    # block, two are inferable from a later step's `if:`, and THREE — the
    # camoufox download, the Claude Code CLI install and the weekly report —
    # were recoverable only from this annotation, which this tool did not parse.
    #
    # Rule 1 again, in the tool, on the gap the tool exists for.
    #
    # TWO KINDS OF `##[error]`, KEPT APART. A step exiting non-zero and a script
    # deliberately annotating an error are different facts, and this pipeline
    # emits the second on purpose — DEL-2's `::error::Pick generation FAILED`
    # and this tool's own `::error::audit alarm` both render as `##[error]`.
    # Counting them together would make every red run look like it had an
    # absorbed failure.
    # WHICH LEAGUES REPORTED A ZERO. The suppressible findings are run-level or
    # source-level and name no league, so the third condition cannot be read
    # from the finding text — it is read from the log, which does name them.
    # PER-LEAGUE DISCOVERY, from the single `FS_DISCOVERY` line. Carries the
    # league, so a finding built from it is league-scoped and condition 3 can be
    # applied PER FINDING instead of per run.
    f["league_discovery"] = [
        {"league": lg, "state": st, "rows": rows,
         "earliest": None if e == "None" else e,
         "cutoff": None if c == "None" else c, "kept": int(k)}
        for lg, st, rows, e, c, k in re.findall(
            r"FS_DISCOVERY ([a-z0-9/-]+): state=(\S+) rows=(\S+) "
            r"earliest_parsed=(\S+) cutoff=(\S+) kept=(\d+)", log)]

    # RETAINED FOR HISTORICAL LOGS ONLY. Every log written before 2026-09-27
    # carries the three old phrasings and no `FS_DISCOVERY` line, and the ledger
    # has to stay readable across that change — the same reason `ledger_history`
    # parses two `disc[...]` formats. New runs populate `league_discovery`
    # instead; this is dead for them and deliberately kept.
    f["zero_fixture_leagues"] = sorted(set(
        re.findall(r"Scraped 0 fixtures from ([a-z0-9/-]+)", log)
    ) | set(
        re.findall(r"Flashscore: ([a-z0-9/-]+) has no fixtures within", log)
    ) | set(
        re.findall(r"Flashscore returned 0 fixtures for ([a-z0-9/-]+)", log)
    ))
    # ── THE THREE-WAY PARTITION (Stage 27, 2026-10-02) ──────────────────────
    #
    # `##[error]` was split two ways: step exits and everything else. On every
    # alarming `ci-audit` run the count is 2, and BOTH are the audit's own
    # verdict exits — so the detector's expected baseline equalled its
    # threshold and it had NO DYNAMIC RANGE. A genuine third absorbed failure
    # on a `ci-audit` run was invisible.
    #
    # The third bucket is DELIBERATE ANNOTATION EMITTED BY THIS REPOSITORY,
    # identified by SOURCE: every pattern in `SELF_EMITTED_ANNOTATIONS` is
    # pinned to the file that prints it, and a test asserts the emitter still
    # contains it. That is what stops the registry drifting into "errors that
    # look routine".
    f["exit_annotations"] = len(re.findall(
        r"##\[error\]Process completed with exit code \d+", log))
    # MATCHED IN THE RENDERED FORM, NOT THE SOURCE FORM. The registry spells
    # each pattern `::error::X` because that is the literal the emitter must
    # contain, and `test_every_self_emitted_annotation_names_a_REAL_emitter`
    # checks exactly that. But GitHub CONSUMES the workflow command: what the
    # log holds is `##[error]X`. The only place `::error::` survives verbatim
    # is the `##[group]Run ...` echo of a step's own shell source.
    #
    # So the old matcher counted ECHOED SCRIPT TEXT and MISSED every real
    # annotation. MEASURED 2026-10-13 on three alarming ci-audit runs
    # (38337483365, 38246516896, 38308788037), identical on all three:
    #
    #     ##[error] total 4   exits 2   self OLD 1 -> NEW 2
    #     unexplained_nonzero_exit  OLD 1 -> NEW 0
    #
    # A FALSE ABSORBED FAILURE ON EVERY RED AUDIT RUN. The detector was
    # rebuilt in Stage 12.1 precisely to have dynamic range, and it was
    # reporting a phantom remainder of 1 the whole time - two errors that
    # partially cancelled, which is why the total still looked plausible.
    f["self_emitted_annotations"] = sum(
        len(re.findall(re.escape("##[error]" + pat.split("::error::", 1)[1]),
                       log))
        for pat, _src in SELF_EMITTED_ANNOTATIONS if "::error::" in pat)
    f["foreign_annotations"] = (len(re.findall(r"##\[error\]", log))
                                - f["exit_annotations"]
                                - f["self_emitted_annotations"])
    # Retained names, so the ledger and the older rows stay readable.
    f["steps_nonzero_exit"] = f["exit_annotations"]
    f["error_annotations"] = (f["self_emitted_annotations"]
                              + f["foreign_annotations"])

    # THE SIGNAL IS THE REMAINDER. Each deliberate annotation we print is
    # followed by its step exiting non-zero — the script returns 1, or the echo
    # step is followed by `exit 1` — so each explains AT MOST ONE exit. `min`
    # rather than subtraction so a log carrying more annotations than exits
    # cannot drive the remainder negative and read as "fewer than none".
    f["self_emitted_exits"] = min(f["self_emitted_annotations"],
                                  f["exit_annotations"])
    f["unexplained_nonzero_exit"] = (f["exit_annotations"]
                                     - f["self_emitted_exits"])
    return f



# ───────────────────────────────── the empty-card suppression, WITH AN EXPIRY

#: The international break, MEASURED from an external reference on 2026-09-27:
#: `openfootball/football.json` 2026-27 shows ZERO fixtures in all eight covered
#: leagues from 2026-09-21 through 2026-10-08, and the card returns on 10-09
#: with 7 fixtures, then 41 on 10-10. Every one of the eight has its last
#: fixture on 09-20 and its next on 10-09 or 10-10 — a 19-20 day gap.
#:
#: THE CARD IS ABSENT FOR THESE EIGHT LEAGUES ONLY, AND THE PIPELINE'S CARD WAS
#: NEVER EMPTY. Corrected 2026-10-01 from production: inside this very window the
#: pipeline had 34 fixtures and 26 saved picks on five days — 09-21 (2 fx / 2 pk),
#: 09-25 (1/1), 09-26 (25/17), 09-27 (5/5), 09-28 (1/1) — every one of them in an
#: UNCOVERED league (romania/liga-1, spain/laliga2, england/league-one,
#: england/league-two) and ZERO in the covered eight. The scraper confirms the
#: eight independently: all 30 leagues report `none-in-range` with 100+ rows
#: present and the eight's earliest kickoff at 10-09/10-10.
#:
#: So the window's DATES AND SCOPE SURVIVE — the per-league narrowing is what
#: saved them — but no text here may say "the card is absent" without naming the
#: eight. Said of all 30 it is false, and it was already false when first written:
#: `disc[fs=18c]` and `disc[fs=5c]` were printed on the two runs immediately
#: preceding that claim.
#:
#: So "zero fixtures" is the EXPECTED state for these dates IN THESE EIGHT
#: LEAGUES, and the discovery alarms would fire daily with a known cause until
#: they were tuned out.
#:
#: SUPPRESSING THEM IS SUP-1 UNLESS IT EXPIRES. A guard that silences the alarms
#: about its own subject would make the first real collapse after 10-09
#: invisible. So:
#:
#:   * the window is a literal date pair, not a flag anyone must remember
#:   * suppressed findings are still PRINTED, tagged, with the grid cell that
#:     justifies each — suppressed is not unobserved
#:   * `test_empty_card_suppression_has_expired` FAILS from 10-09 onward, so
#:     the suppression deletes itself rather than becoming permanent
#:
#: CLR-2: the action that clears the suppressed condition is THE CARD RETURNING,
#: which arrives by CALENDAR rather than by repair. This is the first
#: suppression in this project cleared by time rather than by a fix — which is
#: exactly why it needs the expiry test and not a judgement call.
#: THE EIGHT LEAGUES THE REFERENCE ACTUALLY COVERS. A LITERAL, for the same
#: reason `EMPTY_CARD_UNTIL` is: a set read from config is a switch, and a
#: switch is what someone forgets.
#:
#: THE THIRD CONDITION EXISTS BECAUSE THE SUPPRESSION EXCEEDED ITS EVIDENCE.
#: `openfootball/football.json` 2026-27 answers for these eight and returns 404
#: for the other 22 of the 30 configured leagues. Suppressing a discovery zero
#: across all 30 on the strength of a grid covering 8 would have recorded a
#: genuine failure in the other 22 as "expected" for eleven days, when nothing
#: established that it was.
#:
#: The residual was large and is now MEASURED, 2026-09-28 and again in production
#: 2026-10-01: on 09-19 the uncovered 22 produced 75 of 116 tracked fixtures, on
#: 09-26 all 25, on 09-27 all 5 and on 09-28 all 1. Every zero is the break and
#: not a defect — all 30 leagues report `none-in-range` with rows present (27-126)
#: and a future earliest kickoff, `no-rows` is 0 on every run, and production
#: agrees with the 09-28 probe on all 30 leagues' state, row count and earliest
#: date. The uncovered 22 resume EARLIER than the eight: spain/laliga2 10-02,
#: england/league-one and league-two 10-03, finland/veikkausliiga 10-07,
#: romania/liga-1 10-08 — all confirmed from the scraper, not from the grid.
EMPTY_CARD_COVERED_LEAGUES = frozenset({
    "england/premier-league", "england/championship", "spain/laliga",
    "germany/bundesliga", "italy/serie-a", "france/ligue-1",
    "netherlands/eredivisie", "portugal/primeira-liga",
})

#: Pulls the league out of a league-scoped finding, so condition 3 reads the
#: finding rather than the run.
_LEAGUE_IN_FINDING = re.compile(r"^discovery: ([a-z0-9/-]+) found 0 fixtures")

# ───────────────────────────── the schedule margin, MEASURED daily (Stage 23)
#
# `2026-11-06` came out of a linear fit with R^2 = 0.451 — half the variance
# unexplained — and a fitted date cited bare becomes a measurement within two
# weeks. That class has already cost this project three errors in one week, so
# the date is kept ONLY as `simulated, R^2=0.451` beside a number computed from
# the live series every day.

#: Weekend earliest kickoff, UTC minutes. The deadline is WEEKEND-ONLY: Sat/Sun
#: kick off from 10:15 while Mon-Fri start 14:30-15:00, so this is the binding
#: case and weekday runs carry four hours more slack.
#:
#: MEASURED 2026-10-01 on 1666 rows with real kickoff times — the 419
#: sub-second-precision rows are `datetime.now()` phantoms, not kickoffs, and
#: 10:04 was one of them. INDEPENDENTLY VERIFIED against openfootball, which is
#: not the data that produced it: Sparta Rotterdam v Feyenoord on 2026-08-09
#: reads 12:15 CEST there and 10:15 UTC here, and it is openfootball's own
#: earliest eredivisie kickoff of the season.
#: ── TWO DEADLINES, 2026-10-10 ───────────────────────────────────────────────
#:
#: The deadline is the EARLIEST KICKOFF, and it differs by day type. Measured
#: against an INDEPENDENT reference (Flashscore fixtures pages probed with
#: max_days_ahead=8, converted from Europe/Kyiv to UTC after verifying the
#: offset against the production `matches` table, which stores UTC):
#:
#:     weekend  10:15 UTC (Sun), 10:30 (Sat)   -> 10:15 binds
#:     weekday  12:00 UTC (Mon)                -> 12:00 binds
#:              15:00 (Fri), 16:45 (Tue-Thu)
#:
#: THE REFERENCE DOES NOT REPRODUCE THE 14:30 FIGURE that was on file for
#: weekdays; it gives 12:00, which is 2h30 stricter. The stricter measured
#: value is used, because a figure verified against the data that produced it
#: is not verified.

#: ── RETIRED AS BOUNDS 2026-10-10, kept as LABELLED PERCENTILES ────────────
#:
#: These were used as deadlines. Measured over 148 weekdays and 66 weekends of
#: clean rows, they are percentiles of a date-varying quantity:
#:
#:     14:30 weekday -> the ~12th percentile. 87.8% of weekdays start LATER;
#:                      18 of 148 start EARLIER, and `0 4` BREACHES on those.
#:     10:15 weekend -> the ~5th percentile. Weekends at 10:00 occur, and the
#:                      weekend cron clears even those by 198 minutes.
#:
#: The margin is now computed against `discovered_deadline()`. These remain
#: only as the fallback a single-cron or undiscovered-card path reports, and
#: they arrive WITH their percentile label so no reader mistakes one for a
#: bound. `test_no_margin_computation_uses_a_deadline_LITERAL` keeps the
#: literals out of this file.
#:
#: THE TWO PROVENANCE FACTS THAT MUST STAY BESIDE THE CONSTANT, because a
#: deadline without them has been wrong twice:
#:
#: 1. 10:04 WAS A PHANTOM. The earlier figure came from rows whose `match_date`
#:    carries sub-second precision — `datetime.now()` ingestion stamps, not
#:    kickoffs, 419 of them. Every query here that takes a kickoff extremum
#:    applies the sub-second exclusion; uncorrected the weekday deadline reads
#:    07:53 and 111 of 112 sub-14:30 rows in one league are phantoms.
#: 2. INDEPENDENTLY VERIFIED against openfootball, which is not the data that
#:    produced it: Sparta Rotterdam v Feyenoord on 2026-08-09 reads 12:15 CEST
#:    there and 10:15 UTC here, and it is openfootball's own earliest
#:    eredivisie kickoff of the season.
# ── THE FALLBACK IS THE DERIVED FLOOR NOW, NOT THE PERCENTILE. 2026-10-19 ──
#
# `_DEADLINE_PERCENTILES` held 12:00 / 10:15 and was the fallback on 1087 of
# 1090 days, two and a half hours laxer than our own measured earliest weekday
# kickoff. That gap WAS the exposure tail (max 150m, above 60m on 8 of 715
# weekday days); it never belonged to the reference. The floor is derived as
# `ceil_15(ready_binding + 60 registered)` per day type, from each day type's
# OWN series -- 09:45 weekday, 09:15 weekend. The percentile record is kept
# under its own name because it is still the right label for what it holds.
#: ── DERIVED PER LOOKUP, FOR TOMORROW. LOCALISED 2026-10-22. ──────────────
#:
#: The anchor is a LOCAL time now and the UTC minute is a function of the day,
#: so this mapping resolves `_DEADLINE_FLOOR` on every lookup rather than
#: snapshotting it at import. `_DEADLINE_FLOOR` itself answers for TOMORROW,
#: which is the day every schedule question here concerns.
#:
#: On 2026-10-22 it reads weekday 09:30 / weekend 09:00 UTC; from 2026-10-25
#: it reads 10:30 / 10:00. `schedule_margins_by_daytype` does NOT use it --
#: it asks `deadline_floor_for` with the day it is serving, so the margin
#: cannot be computed from an import-time value on a day that straddles the
#: transition.
class _DeadlineMinForTomorrow(dict):
    def __getitem__(self, day_type):
        return _DEADLINE_FLOOR[day_type][0]

    def get(self, day_type, default=None):
        try:
            return self[day_type]
        except KeyError:
            return default

    def keys(self):
        return _DEADLINE_FLOOR.keys()

    def items(self):
        return ((k, self[k]) for k in self.keys())

    def values(self):
        return (self[k] for k in self.keys())

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(list(self.keys()))


DEADLINE_MIN_BY_DAYTYPE = _DeadlineMinForTomorrow()

#: Which percentile each one is, for the report line.
DEADLINE_BASIS_BY_DAYTYPE = {
    k: f"{v[0]} ({v[1]})" for k, v in _DEADLINE_FLOOR_PROVENANCE.items()
}

#: The WORSE of the two, kept under its old name so a single-cron regime and
#: every existing caller still reads one conservative number.
#: THE WORSE OF THE TWO, derived on access for the same reason as above. It
#: is a property-like callable's value, not a frozen int, so a process that
#: outlives a DST transition cannot keep serving the old season's figure.
def schedule_deadline_min() -> int:
    return min(DEADLINE_MIN_BY_DAYTYPE.values())


#: Kept under its old name for the fourteen call sites that read one
#: conservative number. Resolved at import; every margin path that knows its
#: day uses `deadline_floor_for` instead.
SCHEDULE_DEADLINE_MIN = schedule_deadline_min()

#: MEASURED 2026-10-10, scheduled daily-picks runs, regime `0 0 * * *`,
#: n=10: wall-clock minutes from run start to run end.
MARGIN_RUN_DURATIONS = [72.3, 53.3, 29.0, 41.8, 47.5, 37.0, 40.8,
                        45.4, 43.2, 42.3]

#: ── EACH DAY TYPE'S OWN DURATION SPREAD. 2026-10-19. ──────────────────────
#:
#: `MARGIN_RUN_DURATIONS` above is ONE pooled series (n=10, the `0 0 * * *`
#: regime, measured 2026-10-10) and the prospective margin used it for BOTH day
#: types. Borrowing the other day type's spread is precisely what `0 4` got
#: wrong, and the weekend's is half again as wide: mean+3sd 148.8 against the
#: weekday's 100.9.
#:
#: `measured` 2026-10-19 from every scheduled `daily-picks` run GitHub still
#: lists (--limit 120), wall-clock minutes from run start to run end, split by
#: the day of week the run STARTED on. Pooled across cron regimes on purpose:
#: a run's duration does not depend on when it was triggered, and pooling is
#: both better powered (n=78 / n=34 against n=10) and more conservative
#: (weekday mean+3sd 100.9 against 59.4 for the current regime alone).
#:
#:   weekday  n=78  max 100.8  mean 42.5  sd 19.5  mean+3sd 100.9
#:   weekend  n=34  max 129.2  mean 54.8  sd 31.3  mean+3sd 148.8
#: ── IT WAS MEASURING THE WRONG QUANTITY. CORRECTED 2026-10-21. ───────────
#:
#: The series below was `run start -> run END`, and the margin asks a
#: different question: will the CARD BE SAVED AND SENT before the earliest
#: kickoff? Everything after the picks step is YESTERDAY's business --
#: `Save briefings-sent guard`, the Flashscore results scrape, `Settle late
#: results`, the alerts and the cache saves -- and none of it can affect
#: whether today's card beats today's kickoff. Verified against the workflow's
#: own step order, not assumed.
#:
#: THE POST-PICKS TAIL IS NOT SMALL: `measured` 2026-10-21 over n=111
#: scheduled runs, mean 12.8 min, sd 8.7, min 1.7, max 49.6.
#:
#:   weekday (n=79)     mean    sd     max   mean+3sd    binding margin
#:   full run           43.0  19.1   100.8     100.2       +38.0
#:   TIME-TO-PICKS      29.7  18.6    76.7      85.6       +52.6
#:                                            -> +14.6 recovered, of which
#:                                               LEVEL +13.3, SPREAD +1.3
#:
#: FOURTEEN OF THE NINETEEN-POINT-EIGHT MINUTES WERE AN INSTRUMENTATION
#: ERROR, NOT A DURATION PROBLEM. No step got faster; the margin stopped
#: charging the card for work done after it was delivered. The registered
#: 60-minute minimum is untouched, and so is the deadline.
#:
#: THE OLD SERIES IS KEPT BELOW AT `MARGIN_RUN_DURATIONS_FULL_BY_DAYTYPE`.
#: A correction that deletes what it corrected cannot be checked, and the
#: full-run figure is still the right one for any question about runner cost.
#:
#: `measured` 2026-10-21 from every scheduled `daily-picks` run GitHub still
#: lists (--limit 120), as
#:     (completion of "Generate, review, and send picks") - (run start),
#: split by the day of week the run STARTED on, pooled across cron regimes for
#: the same reason as before: a run's duration does not depend on when it was
#: triggered, and pooling is better powered and more conservative.
MARGIN_RUN_DURATIONS_BY_DAYTYPE = {
    "weekday": [
        4.1, 4.4, 5.1, 5.1, 5.3, 5.6, 6.1, 6.7, 7.0, 8.0, 8.1, 9.6, 9.9,
        10.3, 10.6, 11.0, 11.3, 11.6, 12.4, 12.6, 13.5, 18.5, 18.5, 18.8,
        19.4, 19.7, 19.9, 20.2, 20.5, 20.6, 20.9, 22.6, 24.0, 24.3, 24.7,
        24.8, 25.6, 25.9, 26.2, 26.6, 27.9, 27.9, 28.5, 28.9, 29.4, 29.6,
        31.4, 31.7, 32.8, 34.1, 34.2, 34.5, 36.7, 36.9, 37.3, 37.5, 38.8,
        39.9, 40.6, 41.1, 41.4, 41.9, 42.4, 42.6, 44.9, 45.8, 46.1, 48.7,
        54.0, 58.4, 59.8, 60.8, 61.5, 61.8, 64.7, 67.4, 71.4, 75.8, 76.7,
    ],
    "weekend": [
        4.5, 5.2, 5.9, 7.0, 7.3, 8.3, 9.5, 10.1, 24.8, 27.8, 28.5, 31.0,
        31.8, 33.8, 34.1, 36.2, 39.2, 41.2, 49.1, 58.0, 59.6, 64.1, 66.1,
        68.3, 70.3, 79.5, 81.6, 81.8, 83.0, 88.1, 90.4, 114.8,
    ],
}

#: THE SUPERSEDED SERIES, run start to run END, retained so the correction
#: above can be checked rather than taken on trust. `measured` 2026-10-19.
#: Use it for runner-cost questions; it is the WRONG quantity for the margin.
MARGIN_RUN_DURATIONS_FULL_BY_DAYTYPE = {

    "weekday": [
        9.4, 9.8, 13.6, 13.9, 14.9, 15.5, 15.6, 17.3, 18.6, 19.9, 21.9, 22.4,
        24.7, 25.4, 26.1, 27.3, 27.7, 27.8, 28.1, 29.0, 30.2, 30.3, 30.9, 31.1,
        31.6, 33.2, 33.4, 33.7, 33.7, 34.5, 35.2, 35.2, 36.1, 36.1, 37.0, 39.4,
        39.8, 39.9, 40.8, 41.8, 42.0, 42.1, 42.3, 42.5, 42.8, 43.5, 43.7, 44.4,
        45.8, 46.4, 46.9, 47.1, 47.5, 48.6, 49.2, 50.0, 50.4, 51.8, 53.0, 53.1,
        55.5, 56.6, 57.6, 59.1, 59.9, 62.2, 63.7, 66.2, 66.5, 67.1, 67.7, 68.0,
        70.5, 78.5, 82.4, 88.1, 96.7, 100.8,
    ],
    "weekend": [
        10.7, 12.9, 13.1, 13.6, 14.2, 19.4, 23.8, 31.0, 35.6, 36.4, 36.8, 37.3,
        40.4, 41.1, 41.5, 42.7, 43.2, 45.4, 47.2, 53.1, 53.8, 70.1, 72.0, 72.3,
        78.0, 78.3, 79.1, 92.5, 92.8, 97.3, 101.2, 101.3, 104.4, 129.2,
    ],
}

#: Time from run start to picks complete on a FULL card (>=40 fixtures).
#: MEASURED 2026-10-01, n=4: 67.4-88.1 min, mean 76.4, sd 10.2.
#:
#: ── NOT RE-DERIVED 2026-10-21, AND THE REASON IS MEASURED ─────────────────
#:
#: The review was parallelised in `cccb0e6` at 21:29 on 2026-10-21, and the
#: newest `daily-picks` run is 08:29 the same day. **n = 0 post-change runs.**
#: So this constant still describes the SERIAL regime, and the standing
#: `schedule margin` finding does not clear.
#:
#: AND IT CANNOT BE RE-DERIVED FROM FULL-CARD RUNS ON ANY USEFUL HORIZON.
#: `measured` 2026-10-21 over 2026-07-01..2026-10-21: 73 weekday days carried
#: a card and only **6 of them (8%) had >=40 fixtures** -- 0.4 full-card
#: weekdays per week, median gap 6 days. Reaching n=10 takes ~25 weeks and
#: n=20 ~50. Replacing an n=4 constant with an n=5 one is not progress.
#:
#: WHY A SMALL n IS REFUSED RATHER THAN NOTED: this project measured its own
#: instability. The weekday LATENESS series moved mean+3sd 417.5 -> 428.9 on a
#: SINGLE new point (n=7 -> 8) because that point lowered the mean and raised
#: the sd, and 3sd dominates; the margin went +40 -> +37 overnight. A 3sd
#: bound on single digits is not a bound.
MIN_N_TO_REDERIVE_PICKS_BOUND = 10

#: THE FEASIBLE ROUTE, recorded so the next attempt does not repeat the
#: infeasible one. The full-card restriction exists because a full card was
#: the worst case -- and the parallelisation is precisely what removes most of
#: the card-size dependence: modelled over n=72 runs, the picks step's sd
#: falls 11.71 -> 2.37. So the constant should become a time-to-picks bound
#: over ALL cards, for which n=79 weekday runs are already measured
#: (max 76.7, mean+3sd 85.6, `measured` 2026-10-21, SERIAL regime).
#:
#: THAT STILL NEEDS POST-CHANGE RUNS, because the sd collapse is a MODEL until
#: production confirms it -- but ordinary weekdays arrive at ~5/week rather
#: than 0.4, so n=10 is about two weeks rather than twenty-five.
#:
#: REGIMES ARE NOT POOLED. Serial and parallel runs are different populations
#: under the same rule that keeps the `37 9`, `0 3` and `0 0` cron series
#: apart, so the post-change series starts empty and is not seeded with these.
REDERIVE_PICKS_BOUND_PLAN = (
    "re-point FULL_CARD_PICKS_MAX at ALL-CARD time-to-picks once n >= "
    "MIN_N_TO_REDERIVE_PICKS_BOUND post-cccb0e6 weekday runs exist; do NOT "
    "pool them with the serial runs, and report max and mean+3sd together "
    "with the regime named")
FULL_CARD_PICKS_MAX = 88.1
FULL_CARD_PICKS_3SD = 107.0

# ══════════ A CHANGE OF DEFINITION, NOT A RE-MEASUREMENT ════════════════════
#
#     2026-10-21. Rule 1: recorded as a change of definition, with the
#     population each version described.
#
# THE OLD DEFINITION KEEPS ITS NAME AND ITS LABEL. `FULL_CARD_PICKS_MAX` was
# the MAX time-to-picks over FULL-CARD (>=40 fixture) runs in the SERIAL
# review regime -- n=4, measured 2026-10-01, 67.4-88.1 min. That population is
# retired, not wrong, and the constant is still what the margin uses until the
# replacement has enough points.
#
# WHY THE DEFINITION CHANGES RATHER THAN THE MEASUREMENT BEING REPEATED.
# Full-card weekdays arrive at 0.4/week (6 of 73 days, measured 2026-10-21),
# so n=10 of them is ~25 weeks. All-card weekdays arrive at ~5/week, so n=10
# is about TWO WEEKS. That trade would be a straight loss of conservatism
# under the serial regime -- but the parallelisation is exactly what removes
# the card-size dependence the full-card restriction existed to capture:
# modelled over n=72 runs, the picks step's sd falls 11.71 -> 2.37 and its max
# 55.05 -> 13.76. So the common population now bounds the rare one far more
# tightly than it did before, and the restriction has stopped earning its 25
# weeks.
#
# REGIMES ARE NOT POOLED, so the new series starts at n=0 and the four serial
# full-card runs do not contribute to it. Membership is by `headSha`: a run
# counts only if its own commit contains the parallel review.
PICKS_BOUND_OLD_POPULATION = (
    "MAX time-to-picks over FULL-CARD (>=40 fixture) weekday runs, SERIAL "
    "review regime; n=4, measured 2026-10-01, range 67.4-88.1, mean 76.4, "
    "sd 10.2")
PICKS_BOUND_NEW_POPULATION = (
    "time-to-picks over ALL-CARD weekday runs, PARALLEL review regime "
    "(REVIEW_CONCURRENCY=4, first in cccb0e6, 2026-10-21 21:29); starts at "
    "n=0 because regimes are not pooled")

#: The commit that introduced the parallel review. A run belongs to the new
#: population only if its own `headSha` is this commit or a descendant of it.
PARALLEL_REVIEW_SINCE = "cccb0e6"
PARALLEL_REVIEW_SINCE_UTC = "2026-10-21T21:29"


def _as_datetime(s):
    """A GitHub ISO-8601 timestamp as an aware `datetime`, or `None`.

    `None` RATHER THAN A RAISE, and that is the whole point. This is called on
    fields GitHub may omit -- `completedAt` is absent on a step that never
    finished -- and on 2026-10-21 the call site named a function that did not
    exist at all, so every audit run on `c2eef6d` died with a `NameError` and
    NOTHING was audited for three runs. The caller reads `None` as "this run
    does not contribute", never as a zero.
    """
    if not s:
        return None
    try:
        return _dt.datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return None


_PICKS_SERIES_CACHE: Dict[int, dict] = {}


def parallel_regime_picks_series(limit: int = 120):
    """`{'n','mean','sd','mean_plus_3sd','max','to_ten','members'}`.

    Time-to-picks for every scheduled weekday `daily-picks` run whose commit
    CONTAINS `PARALLEL_REVIEW_SINCE`. `git merge-base --is-ancestor` decides
    membership, so a run on a branch that predates the change cannot sneak in
    on a later timestamp.

    Returns `n=0` and `mean_plus_3sd=None` when the series is empty, which is
    the honest state and NOT a zero.
    """
    import statistics

    if limit in _PICKS_SERIES_CACHE:
        return _PICKS_SERIES_CACHE[limit]

    # AN UNREADABLE `gh` IS UNEVALUATED, NOT AN EMPTY SERIES -- AND IT TOOK
    # TWO GOES. The first version wrote `json.loads(_sh(...) or "[]")`, which
    # looks like a safe default and is the opposite: NO OUTPUT AT ALL became
    # `[]`, so the except clause only ever caught malformed non-empty JSON and
    # an unreadable `gh` reported n=0. That is precisely the confusion this
    # guard was added to prevent, because n=0 is the state the series is
    # legitimately in today -- the two would have been indistinguishable
    # exactly when it matters. Caught by
    # `test_an_EMPTY_series_is_n0_and_an_UNREADABLE_gh_is_UNEVALUATED`.
    def _unevaluated(reason):
        return {"n": None, "members": [], "state": "UNEVALUATED",
                "regime": PICKS_BOUND_NEW_POPULATION,
                "needed": MIN_N_TO_REDERIVE_PICKS_BOUND,
                "mean": None, "sd": None, "mean_plus_3sd": None, "max": None,
                "to_ten_weeks": None, "to_ten_date": "UNEVALUATED",
                "reason": reason + " - so whether any post-change run exists "
                                   "is UNKNOWN, which is distinct from "
                                   "knowing there is none"}

    raw = _sh("gh", "run", "list", "--workflow", "daily-picks.yml",
              "--limit", str(limit), "--json",
              "databaseId,createdAt,event,headSha")
    if not (raw or "").strip():
        return _unevaluated("`gh run list` produced NO OUTPUT")
    try:
        rows = json.loads(raw)
    except json.JSONDecodeError:
        return _unevaluated("`gh run list` produced output that is not JSON")
    # ONE git CALL, NOT ONE PER RUN. The first draft shelled out to
    # `git merge-base --is-ancestor` inside the loop -- up to 120 subprocess
    # spawns, twice per audit (the clock and the margin alert both ask), which
    # timed out a test file. The set of commits CONTAINING the change is one
    # `rev-list`, and membership is then a lookup.
    _descendants = {PARALLEL_REVIEW_SINCE[:7]} | {
        c[:7] for c in _sh("git", "rev-list",
                           f"{PARALLEL_REVIEW_SINCE}..HEAD").split()}
    vals, members = [], []
    for r in rows:
        if r.get("event") != "schedule":
            continue
        sha = r.get("headSha") or ""
        if not sha:
            continue
        if sha[:7] not in _descendants:
            continue                         # the run predates the change
        started = _as_datetime(r.get("createdAt"))
        if started is None or started.weekday() >= 5:
            continue
        ttp = _time_to_picks(str(r["databaseId"]), started)
        if ttp is not None:
            vals.append(ttp)
            members.append(r["databaseId"])
    n = len(vals)
    out = {"n": n, "members": members,
           "regime": PICKS_BOUND_NEW_POPULATION,
           "needed": MIN_N_TO_REDERIVE_PICKS_BOUND}
    if not vals:
        out.update({"mean": None, "sd": None, "mean_plus_3sd": None,
                    "max": None})
    else:
        m = statistics.mean(vals)
        sd = statistics.pstdev(vals) if n > 1 else 0.0
        out.update({"mean": m, "sd": sd, "mean_plus_3sd": m + 3 * sd,
                    "max": max(vals)})
    # PROJECTED DATE TO n=10, from the measured arrival rate of scheduled
    # weekday runs: ~5 per week. Labelled `simulated`, because it assumes the
    # schedule keeps firing and no run fails before its picks step.
    missing = max(0, MIN_N_TO_REDERIVE_PICKS_BOUND - n)
    out["to_ten_weeks"] = round(missing / 5.0, 1)
    out["to_ten_date"] = (
        (_dt.date.today() + _dt.timedelta(days=round(missing / 5.0 * 7)))
        .isoformat() if missing else "reached")
    _PICKS_SERIES_CACHE[limit] = out
    return out


def _time_to_picks(run_id: str, started):
    """Minutes from run start to the picks step completing, or None."""
    try:
        jobs = json.loads(_sh("gh", "run", "view", run_id, "--json", "jobs")
                          or "{}").get("jobs") or []
    except json.JSONDecodeError:
        return None
    for j in jobs:
        for st in j.get("steps") or []:
            if st.get("name") == PICKS_STEP_NAME and st.get("completedAt"):
                done = _as_datetime(st["completedAt"])
                if done is not None:
                    return (done - started).total_seconds() / 60.0
    return None


#: The step whose completion ends "time to picks". Everything after it is
#: yesterday's business and cannot affect today's kickoff.
PICKS_STEP_NAME = "Generate, review, and send picks"


def format_picks_bound_status(s: dict) -> str:
    """One block, every run, so the clock is visible rather than remembered."""
    if s.get("state") == "UNEVALUATED":
        return chr(10).join([
            "picks_bound[UNEVALUATED]",
            f"  {s.get('reason', 'the series could not be read')}",
            f"  margin uses    : the OLD constant, {FULL_CARD_PICKS_MAX} min",
        ])
    if s["n"] == 0:
        body = (f"n=0 -- the series is EMPTY, which is not a zero. No "
                f"scheduled weekday run has yet carried {PARALLEL_REVIEW_SINCE}")
    else:
        body = (f"n={s['n']} mean={s['mean']:.1f} sd={s['sd']:.1f} "
                f"mean+3sd={s['mean_plus_3sd']:.1f} max={s['max']:.1f}")
    return chr(10).join([
        f"picks_bound[REGIME CHANGE IN PROGRESS]",
        f"  since          : {PARALLEL_REVIEW_SINCE} "
        f"({PARALLEL_REVIEW_SINCE_UTC} UTC)",
        f"  new population : {body}",
        f"                   {s['regime']}",
        f"  needed         : n>={s['needed']} "
        + (f"-> ~{s['to_ten_weeks']} weeks, about {s['to_ten_date']} "
           f"(simulated from ~5 scheduled weekday runs/week)"
           if s["to_ten_date"] != "reached" else "-> REACHED, re-derive now"),
        f"  margin uses    : the OLD constant, {FULL_CARD_PICKS_MAX} min",
        f"                   {PICKS_BOUND_OLD_POPULATION}",
        f"  so the margin is CONSERVATIVE: that figure is a full-card MAX from "
        f"the SERIAL regime, and the parallel review cut the picks step's sd "
        f"11.71 -> 2.37. The true margin is very probably BETTER than reported.",
    ])



#: The trailing window the delay series is computed over. STATED, not fitted —
#: 28 days is four weeks of crons, long enough to hold the weekly pattern and
#: short enough to follow a drift rather than average it away.
MARGIN_TRAILING_DAYS = 28

#: THE DECISION FIRES BELOW THIS. Chosen on its own terms, not fitted to the
#: current value: 30 minutes is LESS THAN ONE SD of the observed delay
#: (36.7 min), so at the threshold a single one-sigma-late day already misses
#: the card. Declared here with the current margin known to be +94 min, and
#: deliberately not derived from it.
MARGIN_ALARM_MIN = 30

#: The instrument to reach for when it fires, recorded now so the decision has
#: an option rather than a scramble. The deadline is weekend-only, so a
#: weekend-specific schedule is far cheaper than splitting a 31-step workflow,
#: and Mon-Fri margin is currently being spent against a constraint that does
#: not exist on those days.
MARGIN_REMEDY = (
    # REWRITTEN 2026-10-20. It read "weekend-specific cron (Sat/Sun earlier); "
    # "Mon-Fri binds at 14:30 not 10:15, so the split is the second option not "
    # "the first" -- which cited the RETIRED 14:30/10:15 percentiles and a
    # day-type split that has existed since 2026-10-10. A remedy naming
    # constants that no longer exist points the reader at a decision already
    # taken.
    "the deadline is now the OBSERVED EARLIEST KICKOFF (weekday 09:30, "
    "weekend 09:45) and no deadline choice clears this: the kickoff is a fact "
    "and the ready time is ours. TODAY: dispatch daily-picks manually that "
    "morning, which skips the measured 3h58m-5h04m scheduler lateness. "
    # RE-STATED 2026-10-21, because every figure in the old sentence was
    # superseded within a day. "~7 min in the injury loop" was the SLEEP, and
    # only the pacing part was recoverable (3.1 min, shipped). "~9 min in the
    # AF upsert loop" was never real -- that loop costs ~4 SECONDS. And the
    # "19.8-minute shortfall" was computed against a duration term that
    # measured run start to run END. A remedy quoting retired numbers points
    # the reader at work that does not exist, which is why the previous
    # version of this text was rewritten for the same reason on 2026-10-20.
    "DURABLY: an earlier cron, or a faster run. MEASURED 2026-10-21 over "
    "n=72 runs, against the structural margin: parallelising the Claude "
    "review behind an atomic marker buys +30.5m on its own, because it cuts "
    "the SPREAD (sd 18.4 -> 10.8) and the bound is mean+3sd; moving the ML "
    "retrain off the picks path buys +3.5m; the shipped request-pacing "
    "change buys +4.0m once it reaches the series. The remaining structural "
    "deficit is 7.4m, so retrain + pacing very nearly closes it and the "
    "review closes it outright")


#: ── `mean+3sd` IS CONSERVATIVE BY ACCIDENT. 2026-10-22. ───────────────────
#:
#: Every margin in this file is computed from `mean + 3sd` of a lag series,
#: and on 2026-10-22 the weekday series was found to be **BIMODAL UNDER ONE
#: UNCHANGED CRON** (`0 0 * * 1-5` throughout): 255.9-304.9 through 10-09,
#: a ten-day gap, then 128.4-144.1. `measured` the same day:
#:
#:     weekday n=10   mean 220.0   sd 74.1   mean+3sd 442.3   MAX 304.9
#:
#: (sd is the SAMPLE sd, n-1, which is what `schedule_margin` uses; the
#: population sd gives 70.3 / 430.9. Two figures for one quantity is how a
#: record stops being checkable, so the one that BINDS is the one quoted.)
#:
#: **THE BOUND EXCEEDS THE OBSERVED MAXIMUM BY 137.5 MINUTES.** Three sd of a
#: two-cluster series is mostly the DISTANCE BETWEEN THE CLUSTERS, not tail
#: risk, so the margin is buying two hours of headroom against an event never
#: observed.
#:
#: THE DIRECTION OF THAT ACCIDENT IS NOT GUARANTEED. It is conservative while
#: the clusters straddle the mean. Let the lower cluster accumulate until it
#: dominates and the mean falls toward it while the sd stays wide; let a third
#: cluster appear ABOVE both and the bound understates the tail instead.
#: Nothing in `mean+3sd` knows which of those is happening, and the one case
#: already observed moved the bound 417.5 -> 428.9 **on a single new point**.
#:
#: THE SUCCESSOR IS AN EMPIRICAL QUANTILE. It needs no segmentation decision,
#: it cannot be dragged by between-cluster distance, and it is never outside
#: the data. p95 is named rather than p99 because p99 is unreachable for this
#: series -- see `QUANTILE_SUCCESSOR`.
#:
#: AND IT IS NOT SWITCHED TO YET, because at n=10 **neither statistic is
#: supported**: `mean+3sd` extrapolates past every observation, and the p95
#: quantile is simply the maximum. Saying so is the honest state. Both are
#: printed side by side, each labelled, so the switch is a comparison rather
#: than a replacement.
QUANTILE_SUCCESSOR = {
    "statistic": "p95",
    #: USABLE WHEN AT LEAST FIVE OBSERVATIONS SIT STRICTLY ABOVE IT, i.e.
    #: n >= 5/(1-p). FIVE, derived rather than chosen: one new point moved
    #: `mean+3sd` by 11.4 minutes at n=9 -> n=10 (`measured` 2026-10-20/21),
    #: and a statistic that five points support cannot be moved that far by a
    #: single arrival.
    "min_n": 100,
    "min_n_rule": "n >= 5/(1-p); p95 -> 100, p99 -> 500",
    #: p99 would need n >= 500. At the measured ~5 scheduled weekday runs per
    #: week that is ~100 weeks, so it is named as out of reach rather than
    #: offered as an option with no date.
    "p99_min_n": 500,
    "arrival_per_week": 5,
    "projected_usable": "2027-03-11",      # simulated: (100-10)/5 weeks
    "why_not_now": ("at n=10 NEITHER statistic is supported: mean+3sd "
                    "extrapolates 137.5 minutes past the observed maximum, "
                    "and "
                    "the p95 quantile is just that maximum"),
}


#: ── THE BIMODALITY TEST. 2026-10-22. ─────────────────────────────────────
#:
#: A bimodal series must not be summarised by a statistic that assumes it is
#: not, so the margin is computed PER CLUSTER and the pessimistic one governs.
#: The split is at the largest gap between consecutive sorted values, and the
#: series counts as bimodal only when that gap dwarfs the typical one.
#:
#: `measured` 2026-10-22, the two live series:
#:
#:   weekday  n=10  sorted 128.4 .. 304.9   max gap 111.8  median 6.4  = 17.33
#:   weekend  n=6   sorted 219.3 .. 293.7   max gap  32.7  median 12.5 =  2.61
#:
#: 8.0 sits between them with a wide margin on both sides. IT IS PROVISIONAL
#: AND SAYS SO: a ratio threshold derived from the only two series available is
#: close to circular, and the honest fix is the empirical quantile, which needs
#: no split at all -- see `QUANTILE_SUCCESSOR`, usable at n>=100.
LAG_CLUSTER_GAP_RATIO = 8.0
LAG_CLUSTER_PROVENANCE = (
    "largest consecutive gap / median consecutive gap; measured 2026-10-22 as "
    "17.33 (weekday, n=10) and 2.61 (weekend, n=6), so 8.0 separates them. "
    "PROVISIONAL: a threshold derived from two series is near-circular, and "
    "the p95 successor removes the need for any split at n>=100")


#: The smallest n at which the split rule can reach a verdict at all. Below
#: this there is no median gap worth the name, so "it did not split" is
#: indistinguishable from "it is not bimodal" -- and binding on the pooled
#: figure would be binding a mean+3sd on one or two observations.
LAG_CLUSTER_MIN_N = 4


def cluster_state(lags) -> str:
    """`BIMODAL` / `UNIMODAL` / `UNSEPARABLE`.

    THE THIRD STATE IS THE POINT. `split_lag_clusters` returns
    `(vals, None)` both when a series is genuinely unimodal and when it is too
    small to tell, and those are different facts: the first licenses binding
    on the pooled figure, the second licenses nothing. At n=1 -- which is
    where the 2026-10-22 cron move left both series -- `mean+3sd` equals the
    single observation and a margin computed from it would read as a
    measurement.
    """
    import statistics
    vals = sorted(float(v) for v in lags if v is not None)
    if len(vals) < LAG_CLUSTER_MIN_N:
        return "UNSEPARABLE"
    gaps = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
    med = statistics.median(gaps)
    widest = max(gaps)
    # ── A ZERO MEDIAN GAP IS NOT "CANNOT TELL". ──────────────────────────
    #
    # The first version returned UNSEPARABLE whenever `med <= 0`, and
    # `test_a_breaching_margin_DOES_alarm` caught it immediately: five
    # IDENTICAL observations have every gap zero, which is a single point
    # mass — the opposite of bimodal — and the rule was silencing a genuine
    # breach by calling it unseparable.
    #
    # So the two zero cases are told apart: every gap zero is UNIMODAL, and a
    # zero median with a non-zero widest gap is an infinite ratio, i.e. the
    # most bimodal a series can be.
    if widest == 0:
        return "UNIMODAL"
    if med <= 0:
        return "BIMODAL"
    return ("BIMODAL" if widest / med >= LAG_CLUSTER_GAP_RATIO
            else "UNIMODAL")


def split_lag_clusters(lags):
    """`(slow, fast)` when the series is bimodal, else `(sorted(lags), None)`.

    SLOW FIRST, because the slow cluster is the one that governs a safety
    choice and returning it first makes a caller that ignores the second
    element conservative rather than optimistic.

    Callers that need to tell "not bimodal" from "cannot tell" must ask
    `cluster_state`; this function collapses both into `(vals, None)`.
    """
    import statistics
    vals = sorted(float(v) for v in lags)
    if len(vals) < LAG_CLUSTER_MIN_N:
        return vals, None
    gaps = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
    med = statistics.median(gaps)
    widest = max(gaps)
    if med <= 0 or widest / med < LAG_CLUSTER_GAP_RATIO:
        return vals, None
    i = gaps.index(widest)
    fast, slow = vals[:i + 1], vals[i + 1:]
    # Each side must hold enough points to have a spread at all; otherwise
    # the "cluster" is one outlier and the split is noise.
    if len(fast) < 2 or len(slow) < 2:
        return vals, None
    return slow, fast


def empirical_quantile(values, p: float):
    """The `p`-quantile by nearest-rank, or `None` when undefined.

    NEAREST-RANK, NOT INTERPOLATED, and `None` below n = 1/(1-p): an
    interpolated p95 of nine points is a number with no observation above it,
    which is the same extrapolation `mean+3sd` is being retired for. Returning
    `None` makes "not enough data" a state rather than a plausible figure.
    """
    vals = sorted(v for v in values if v is not None)
    n = len(vals)
    if n == 0 or n < 1 / (1 - p):
        return None
    import math
    return vals[min(n - 1, int(math.ceil(p * n)) - 1)]


def quantile_is_usable(n: int, p: float = 0.95) -> bool:
    """Whether `n` supports the quantile well enough to BIND on it."""
    return n >= 5 / (1 - p)


def schedule_margin(series, cron_minutes: int,
                    deadline_min: int = SCHEDULE_DEADLINE_MIN,
                    day_type: str = None):
    """`dict` of the margin to ONE deadline, from a measured lag series.

    `deadline_min` defaults to the WORSE of the two day-type deadlines, so a
    caller that does not say which day type it means gets the conservative
    answer rather than a wrong one.

    `series` is `[(date, lag_minutes), ...]` for runs whose OWN headSha carries
    the cron in force — the `390a4be` rule, because a lag is only comparable
    within one schedule.

    Returns both the `max` and the `mean+3sd` margin. Never the mean alone: the
    mean is what hid a negative margin for a month.
    """
    if not series:
        return {"n": 0, "reason": "no runs under the current cron yet",
                "deadline_min": deadline_min, "day_type": day_type}
    lags = [l for _, l in series]
    n = len(lags)
    mean = sum(lags) / n
    sd = (sum((x - mean) ** 2 for x in lags) / (n - 1)) ** 0.5 if n > 1 else 0.0
    out = {"n": n, "min": min(lags), "max": max(lags), "mean": mean, "sd": sd,
           "mean_plus_3sd": mean + 3 * sd, "cron": cron_minutes}
    # THE SUCCESSOR STATISTIC, COMPUTED AND PRINTED BUT NOT USED. See
    # `QUANTILE_SUCCESSOR` for why it is not switched to yet.
    out["p95"] = empirical_quantile(lags, 0.95)
    out["p99"] = empirical_quantile(lags, 0.99)
    out["quantile_usable"] = quantile_is_usable(n, 0.95)
    for key, delay, picks in (("margin_max", max(lags), FULL_CARD_PICKS_MAX),
                              ("margin_3sd", mean + 3 * sd, FULL_CARD_PICKS_3SD)):
        out[key] = deadline_min - (cron_minutes + delay + picks)
    # FAIL CLOSED on the worse of the two.
    out["pooled_binding"] = min(out["margin_max"], out["margin_3sd"])

    # ── PER CLUSTER, AND THE PESSIMISTIC ONE GOVERNS. 2026-10-22. ─────────
    #
    # The pooled figures above stay, labelled, because they are what every
    # earlier entry in the ledger quotes. But on a BIMODAL series `mean+3sd`
    # measures the distance between the clusters: for the weekday series it
    # reads 442.3 against an observed maximum of 304.9, so the pooled margin
    # is 137.5 minutes of headroom against an event never seen -- and no cron
    # can clear 60 minutes against it. Deriving a cron from that figure is
    # fitting the schedule to an artefact.
    #
    # So when the series splits, the margin is computed on EACH cluster and
    # `binding` comes from the SLOW one. The fast cluster is reported and
    # never relied on: nothing explains the gap, so nothing licenses assuming
    # it persists.
    _slow, _fast = split_lag_clusters(lags)
    out["bimodal"] = _fast is not None
    out["clusters"] = {}
    for _name, _vals in (("slow", _slow), ("fast", _fast)):
        if not _vals:
            continue
        _m = sum(_vals) / len(_vals)
        _sd = (sum((x - _m) ** 2 for x in _vals) / (len(_vals) - 1)) ** 0.5 \
            if len(_vals) > 1 else 0.0
        _c = {"n": len(_vals), "mean": _m, "sd": _sd, "max": max(_vals),
              "mean_plus_3sd": _m + 3 * _sd}
        _c["margin_max"] = deadline_min - (
            cron_minutes + _c["max"] + FULL_CARD_PICKS_MAX)
        _c["margin_3sd"] = deadline_min - (
            cron_minutes + _c["mean_plus_3sd"] + FULL_CARD_PICKS_3SD)
        _c["binding"] = min(_c["margin_max"], _c["margin_3sd"])
        out["clusters"][_name] = _c
    out["cluster_state"] = cluster_state(lags)
    if out["bimodal"]:
        out["binding"] = out["clusters"]["slow"]["binding"]
        out["binding_from"] = "slow cluster"
    else:
        out["binding"] = out["pooled_binding"]
        out["binding_from"] = ("the whole series (not bimodal)"
                               if out["cluster_state"] == "UNIMODAL"
                               else "the whole series, UNSEPARABLE at this n")
    # ── UNSEPARABLE REPORTS AND DOES NOT ALARM. 2026-10-23. ──────────────
    #
    # A third state, distinct from clear and from breached. Below
    # LAG_CLUSTER_MIN_N the split rule cannot reach a verdict, so "it did not
    # split" carries no information -- and `mean+3sd` of one or two points is
    # that point. Binding on it would be the "absent instrumentation is not a
    # zero" error with extra arithmetic.
    #
    # EVERY FIGURE STILL PRINTS. This suppresses the ALARM, not the report:
    # the margin, its basis and the reason are all on the line, and the
    # conservative reading named beside them.
    out["alarm"] = (out["binding"] < MARGIN_ALARM_MIN
                    and out["cluster_state"] != "UNSEPARABLE")
    if out["cluster_state"] == "UNSEPARABLE":
        out["no_alarm_reason"] = (
            f"n={n} is below {LAG_CLUSTER_MIN_N}, so the series cannot be "
            f"separated into clusters and mean+3sd is not a bound - this is "
            f"neither clear nor breached")
    out["deadline_min"] = deadline_min
    out["day_type"] = day_type
    return out


def current_cron_minutes(sha: str = None):
    """The cron in `daily-picks.yml`, as UTC minutes. `None` if not a single cron.

    Read from the SHA when given, so membership is decided by what the run
    actually carried rather than by what the worktree carries now.
    """
    path = ".github/workflows/daily-picks.yml"
    text = (_sh("git", "show", f"{sha}:{path}") if sha
            else pathlib.Path(path).read_text(encoding="utf-8"))
    crons = re.findall(r"^\s*- cron:\s*'([^']+)'", text, re.M)
    if len(crons) != 1:
        return None
    m = re.match(r"^(\d+)\s+(\d+)\s+\*\s+\*\s+\*$", crons[0].strip())
    return int(m.group(2)) * 60 + int(m.group(1)) if m else None


#: Where `collect_lag_series_by_daytype` reports the shas it could NOT read.
#: A separate key and not a silent exclusion, because the exclusion wore the
#: `UNEVALUATED: no runs under the current cron yet` label and was read as
#: "the cron is new" for as long as the shallow clone lasted.
UNREADABLE_KEY = "__unreadable__"


def current_crons_by_daytype(sha: str = None,
                             want_exprs: bool = False) -> dict:
    """`{'weekend': minutes|None, 'weekday': minutes|None}` from the workflow.

    Added 2026-10-10 with the split. `current_cron_minutes` returns None for
    anything that is not exactly one `m h * * *`, which a split schedule never
    is — so without this the margin would report UNEVALUATED from the day the
    split landed, and "the schedule cannot be assessed" would be
    indistinguishable from "the schedule is fine".

    A cron whose day-of-week field is `*` applies to BOTH day types.
    """
    path = ".github/workflows/daily-picks.yml"
    text = (_sh("git", "show", f"{sha}:{path}") if sha
            else pathlib.Path(path).read_text(encoding="utf-8"))
    # UNREADABLE IS NOT "A DIFFERENT CRON". `_sh` returns "" when `git show`
    # fails, which it does for every sha outside a SHALLOW CLONE —
    # `actions/checkout` defaults to `fetch-depth: 1`.
    #
    # MEASURED 2026-10-11: that made the whole series empty in CI and reported
    # it as `UNEVALUATED: no runs under the current cron yet`, which is
    # indistinguishable from "the cron is new". It had looked fine for weeks
    # only because every run in the 28-day window shared the then-HEAD sha —
    # older CI runs show n=6 and n=4 — and it broke the moment commits became
    # frequent. A membership rule that degrades silently over-excludes, and
    # over-exclusion wearing the UNEVALUATED label is worse than either.
    # Third-state family - and NOT "instance thirteen", which an earlier
    # draft of this comment asserted. The ledger's highest enumerable
    # ordinal for that family is NINE, claimed twice for two different
    # findings. docs/rule-tallies.md has the list; there is no reliable
    # ordinal, and a source comment is the last place to invent one.
    if sha and not text.strip():
        return None
    out = {"weekend": None, "weekday": None}
    exprs = {"weekend": None, "weekday": None}
    for expr in re.findall(r"^\s*- cron:\s*'([^']+)'", text, re.M):
        parts = expr.strip().split()
        if len(parts) != 5 or not parts[0].isdigit() or not parts[1].isdigit():
            continue
        minutes = int(parts[1]) * 60 + int(parts[0])
        dow = parts[4]
        # Which day types does this cron cover? 0 and 7 are Sunday, 6 Saturday.
        days = set()
        if dow == "*":
            days = set(range(7))
        else:
            for piece in dow.split(","):
                if "-" in piece:
                    a, b = piece.split("-")
                    days |= set(range(int(a), int(b) + 1))
                elif piece.isdigit():
                    days.add(int(piece) % 7)
        if days & {0, 6}:
            out["weekend"] = minutes
            exprs["weekend"] = expr.strip()
        if days & {1, 2, 3, 4, 5}:
            out["weekday"] = minutes
            exprs["weekday"] = expr.strip()
    # TWO READERS, ONE PARSE. An earlier draft of this fix returned the
    # expressions as an `__exprs__` key inside the same dict -- which
    # `schedule_margins_by_daytype` iterates with `crons.items()`, so a dict
    # value would have reached `cron_minutes // 60`. A sentinel key in a
    # mapping somebody else iterates is a landmine; a second return value is
    # not. And a SECOND PARSE would have been THE HABIT: one rule, two
    # definitions, free to drift.
    return (out, exprs) if want_exprs else out


def schedule_margins_by_daytype(series_by_type: dict, crons: dict,
                                day=None) -> dict:
    """One margin per day type, and the TRIGGER FIRES ON THE WORSE.

    `day` IS THE DAY THE MARGIN SERVES and defaults to TOMORROW, because the
    deadline is a LOCAL fact converted per day (localised 2026-10-22). On
    2026-10-24 tomorrow is a UTC+2 day while today is still UTC+3, and a
    stored UTC constant cannot tell those apart.

    A weekday series and a weekend series are DISTINCT POPULATIONS under the
    same change: they have different crons, different deadlines and different
    n. They are reported apart and never pooled — the same rule that keeps the
    `37 9`, `0 3` and `0 0` series separate.
    """
    _day = day or (_dt.date.today() + _dt.timedelta(days=1))
    out = {}
    for day_type, cron_minutes in crons.items():
        if cron_minutes is None:
            continue
        # THE DAY IS PASSED, NOT ASSUMED. `day` defaults to tomorrow, which
        # is what this margin is about; on 2026-10-24 that is a UTC+2 day
        # while today is still UTC+3, and a stored constant cannot tell the
        # difference.
        out[day_type] = schedule_margin(
            series_by_type.get(day_type, []), cron_minutes,
            deadline_min=_deadline_floor_for(day_type, _day)[0],
            day_type=day_type)
    # A DAY TYPE WITH NO SERIES IS UNEVALUATED, NOT SAFE. The weekday series
    # starts EMPTY the moment the split lands, because membership requires the
    # run's own headSha to carry the cron in force and no weekday run has fired
    # under `0 4` yet. Reporting that as a passing margin would be the exact
    # "absent instrumentation is not a zero" error.
    evaluated = {k: v for k, v in out.items() if "binding" in v}
    out["unevaluated"] = sorted(k for k in out if k not in evaluated)
    if evaluated:
        worst = min(evaluated.values(), key=lambda m: m["binding"])
        out["worse"] = worst["day_type"]
        out["alarm"] = bool(worst["alarm"])
    else:
        out["worse"] = None
        out["alarm"] = False
    return out


def collect_lag_series(today: _dt.date = None, limit: int = 120):
    """`(series, cron_minutes)` for runs under the CURRENT cron only.

    MEMBERSHIP BY EACH RUN'S OWN headSha. Pooling two cron regimes is the
    `390a4be` error — a 09:37 run read as an 11h30m outlier of a 03:00 series —
    and it is prevented here rather than corrected later.
    """
    today = today or _dt.date.today()
    cron_now = current_cron_minutes()
    if cron_now is None:
        return [], None
    raw = _sh("gh", "run", "list", "--workflow", "daily-picks.yml",
              "--limit", str(limit), "--json",
              "databaseId,event,createdAt,headSha")
    try:
        runs = json.loads(raw or "[]")
    except json.JSONDecodeError:
        return [], cron_now
    by_sha: Dict[str, Optional[int]] = {}
    series = []
    for r in runs:
        if r.get("event") != "schedule":
            continue
        try:
            c = _dt.datetime.fromisoformat(r["createdAt"].replace("Z", "+00:00"))
        except (KeyError, ValueError):
            continue
        c = c.replace(tzinfo=None)
        if (today - c.date()).days > MARGIN_TRAILING_DAYS:
            continue
        sha = r.get("headSha") or ""
        if sha not in by_sha:
            by_sha[sha] = current_cron_minutes(sha)
        if by_sha[sha] != cron_now:
            continue                      # a different schedule: NOT comparable
        sched = c.replace(hour=cron_now // 60, minute=cron_now % 60,
                          second=0, microsecond=0)
        if c < sched:
            sched -= _dt.timedelta(days=1)
        series.append((c.date(), (c - sched).total_seconds() / 60.0))
    return series, cron_now


#: ── THE SCHEDULER SERIES, FIVE OF THEM, NEVER POOLED ──────────────────────
#:
#: Five populations is where pooling becomes tempting. Each carries its own n,
#: its own QUERY WINDOW, and its own deadline basis. `11h21m` belongs to
#: `37 9 * * *` and to nothing else.
#:
#: The three retired regimes are recorded figures, not live queries: their runs
#: are outside the 28-day trailing window and re-deriving them would need the
#: `--limit` that produced them, which is exactly the page-size trap.
RETIRED_SERIES = (
    # (cron, n, query window / provenance, deadline basis at the time)
    ("37 9 * * *", "39 @ --limit 120 / 16 @ --limit 60",
     "measured 2026-10-02 and 2026-10-09; n IS a function of the query window",
     "09:45 assumed, never measured"),
    ("0 3 * * *", "32", "measured 2026-10-02, closed regime",
     "10:15, treated as a bound"),
    ("0 0 * * *", "9", "measured 2026-10-10, single-cron regime, 28d window",
     "10:15, treated as a bound"),
)


def format_series_census(series_by_type: dict, crons: dict) -> str:
    """All five, apart, with n and basis. Printed every run."""
    lines = [
        "scheduler series (FIVE populations, never pooled):",
        "  ATTRIBUTION: each run is matched to the cron that governs ITS OWN "
        "DAY OF WEEK,",
        "  and only if its own headSha carried that cron. Nearest-preceding "
        "attribution would",
        "  misread a weekend 04:31 start as 31 min late against `0 4` "
        "(-240 min of error) and a",
        "  weekday 08:31 start as 511 min late against `0 0` (+240). The week "
        "is partitioned by",
        "  day of week, so no run can have both crons as a predecessor.",
    ]
    for cron, n, window, basis in RETIRED_SERIES:
        lines.append(f"  RETIRED  {cron:16s} n={n:34s} deadline basis: {basis}")
        lines.append(f"           {'':16s}   {window}")
    from scripts.schedule_deadline import DEADLINE_FLOOR

    # ── THE LIVE LABEL WAS A LITERAL, AND IT WENT STALE IN A DAY. ─────────
    #
    # This read `expr = ("0 0 * * 6,0" if ... else "0 4 * * 1-5")`. It was
    # correct when written and WRONG from 566c927, where the weekday cron was
    # reverted to `0 0 * * 1-5` -- so the report named `0 4` while the margin
    # printed beside it (+280m) was measured under `0 0`. A label that
    # contradicts its own data is worse than no label: the series attribution
    # is sound, and the only wrong thing on the line is the one field a reader
    # uses to decide WHICH schedule the numbers describe.
    #
    # Now derived from the same parse that produces the minutes, so the two
    # cannot disagree.
    _live = current_crons_by_daytype(want_exprs=True)
    live_exprs = (_live[1] if _live else {})
    for day_type in ("weekend", "weekday"):
        cron_min = crons.get(day_type)
        n = len(series_by_type.get(day_type) or [])
        expr = live_exprs.get(day_type) or "UNRESOLVED"
        _, pct, prov = DEADLINE_FLOOR[day_type]
        state = f"n={n}" if n else "n=0 UNEVALUATED, not clear"
        lines.append(f"  LIVE     {expr:16s} {state:36s} "
                     f"deadline basis: {pct} ({prov.split(';')[0]})")
        if cron_min is None:
            lines.append(f"           {'':16s}   no cron resolved for this day type")
    # THE EXCLUSION, COUNTED. An n that fell because a sha was unreadable is
    # not the same observation as an n that fell because no run fired, and
    # until 2026-10-11 both printed as `n=0 UNEVALUATED`.
    bad = series_by_type.get(UNREADABLE_KEY) or []
    if bad:
        lines.append(
            f"  note: {len(bad)} run(s) EXCLUDED because their headSha could not "
            f"be read, so their cron is unknown and their lag is not "
            f"attributable. This is a DEGRADED census, not a small one. "
            f"CLEARING ACTION: run the audit on a full clone "
            f"(`actions/checkout` with `fetch-depth: 0`); shas: "
            + ", ".join(sh[:7] for sh in bad[:6])
            + (" ..." if len(bad) > 6 else ""))
    else:
        lines.append("  every scheduled run in the window had a READABLE sha "
                     "- no run was excluded for an unknown cron")
    return chr(10).join(lines)


def collect_lag_series_by_daytype(today: _dt.date = None, limit: int = 120):
    """`({'weekend': series, 'weekday': series}, crons)` under the CURRENT crons.

    MEMBERSHIP BY EACH RUN'S OWN headSha, and now also by the run's OWN DAY OF
    WEEK. A weekday run and a weekend run fire from different crons against
    different deadlines, so pooling them is the `390a4be` error one axis over:
    a 04:00 weekday run read as a four-hour outlier of a 00:00 series.
    """
    today = today or _dt.date.today()
    crons = current_crons_by_daytype() or {"weekend": None, "weekday": None}
    out = {"weekend": [], "weekday": []}
    unreadable = set()
    try:
        runs = json.loads(_sh(
            "gh", "run", "list", "--workflow", "daily-picks.yml",
            "--limit", str(limit),
            "--json", "createdAt,headSha,event") or "[]")
    except json.JSONDecodeError:
        return out, crons
    for r in runs:
        if r.get("event") != "schedule":
            continue
        try:
            c = _dt.datetime.fromisoformat(
                r["createdAt"].replace("Z", "+00:00")).replace(tzinfo=None)
        except (KeyError, ValueError):
            continue
        if (today - c.date()).days > MARGIN_TRAILING_DAYS:
            continue
        day_type = "weekend" if c.date().weekday() >= 5 else "weekday"
        cron_now = crons.get(day_type)
        if cron_now is None:
            continue
        # The run must have CARRIED this cron for its lag to be comparable.
        carried = current_crons_by_daytype(r.get("headSha") or "")
        if carried is None:
            unreadable.add(r.get("headSha") or "?")
            continue
        if carried.get(day_type) != cron_now:
            continue
        sched = c.replace(hour=cron_now // 60, minute=cron_now % 60,
                          second=0, microsecond=0)
        if c < sched:
            sched -= _dt.timedelta(days=1)
        out[day_type].append((c.date(), (c - sched).total_seconds() / 60.0))
    out[UNREADABLE_KEY] = sorted(unreadable)
    return out, crons


def format_schedule_margins(margins: dict) -> str:
    """Both day types, apart, with the trigger on the WORSE of the two."""
    if not margins:
        return ("schedule_margin[UNEVALUATED: no cron resolved for either day "
                "type — this is NOT a clean bill]")
    lines = []
    for day_type in ("weekend", "weekday"):
        m = margins.get(day_type)
        if not m:
            lines.append(f"  {day_type:8s} UNEVALUATED (no cron)")
            continue
        dl = m["deadline_min"]
        lines.append(f"  {day_type:8s} deadline {dl // 60:02d}:{dl % 60:02d}  "
                     + format_schedule_margin(m))
    worse = margins.get("worse")
    unev = margins.get("unevaluated") or []
    lines.append(f"  TRIGGER fires on the WORSE day type: {worse}"
                 + ("  ** DECISION FIRES **" if margins.get("alarm") else ""))
    if unev:
        # ── REWORDED 2026-10-22, AND NOT MADE TO ALARM. ───────────────────
        #
        # It said "is UNEVALUATED, not safe". "Not safe" asserts a safety
        # property, and this line cannot evaluate one -- it is reporting that
        # there is nothing to evaluate. Words that claim more than the alarm
        # set delivers are what the marker reservation of 2026-10-21 left
        # behind, named rather than fixed; this is the fix.
        #
        # WHY REWORD RATHER THAN ALARM, which was the other option: an alarm
        # here would be CLEARED BY TIME. The state ends when the next
        # scheduled run fires under that cron, with nobody doing anything,
        # and CLR-2 exists because an alarm cleared by waiting teaches
        # waiting. The two siblings below -- an unreadable `gh workflow list`
        # and a missing cron -- are cleared by an EDIT, so they do alarm.
        lines.append(
            f"  note: {', '.join(unev)}: margin NOT COMPUTED - no run has yet "
            f"fired under its cron, so there is no lateness series. This is "
            f"neither clear nor breached, and it ends on its own when a run "
            f"fires, so it does not alarm. Until then the conservative "
            f"reading is the one the margin would give at this cron's worst "
            f"observed lateness, and the action available is a manual "
            f"dispatch on that day type.")
    return chr(10).join(lines)


def format_schedule_margin(m) -> str:
    """One line, printed EVERY day whether it alarms or not."""
    if not m.get("n"):
        return f"schedule_margin[UNEVALUATED: {m.get('reason', 'no series')}]"
    def hm(v):
        s = "-" if v < 0 else ""
        v = abs(v)
        return f"{s}{int(v)//60}h{int(v)%60:02d}m"
    # ── BOTH STATISTICS, SIDE BY SIDE, EACH LABELLED. 2026-10-22. ─────────
    #
    # `mean+3sd` BINDS; the p95 quantile is printed beside it and binds on
    # nothing. The weekday series is bimodal under one unchanged cron, so
    # `mean+3sd` is measuring between-cluster distance and currently exceeds
    # the observed maximum by 137.5 minutes -- conservative BY ACCIDENT, in a
    # direction nothing guarantees. The switch is a comparison rather than a
    # replacement, and it does not happen before n reaches
    # QUANTILE_SUCCESSOR["min_n"].
    _q = m.get("p95")
    _qtxt = ("UNDEFINED (n too small for any point to sit above it)"
             if _q is None else hm(_q))
    _verdict = ("BINDS" if m.get("quantile_usable")
                else f"not usable until n>={QUANTILE_SUCCESSOR['min_n']}")
    _lines = [
        f"schedule_margin[n={m['n']} window={MARGIN_TRAILING_DAYS}d "
        f"delay max={hm(m['max'])} mean+3sd={hm(m['mean_plus_3sd'])} "
        f"| POOLED margin max={m['margin_max']:+.0f}m "
        f"3sd={m['margin_3sd']:+.0f}m pooled_binding="
        f"{m.get('pooled_binding', m['binding']):+.0f}m]",
    ]
    # ── PER CLUSTER, AND WHICH ONE BINDS. 2026-10-22. ────────────────────
    #
    # The pooled line above is kept because every earlier ledger entry quotes
    # it, but on a bimodal series it is an artefact: `mean+3sd` measures the
    # distance BETWEEN clusters. So each cluster is printed with its own n,
    # and the line says which one `binding` came from. The fast cluster is
    # reported and never relied on.
    if m.get("bimodal"):
        _lines.append(f"{'':10s}  BIMODAL (gap ratio >= "
                      f"{LAG_CLUSTER_GAP_RATIO}): the pooled mean+3sd "
                      f"measures BETWEEN-CLUSTER distance, not tail risk")
        # PNC-1: the threshold's provenance existed as a constant and nothing
        # printed it. This is where a reader asks "why does it split, and why
        # at 8.0?", so it is answered here rather than left in the source.
        _lines.append(f"{'':10s}  split rule: {LAG_CLUSTER_PROVENANCE}")
    for _name in ("slow", "fast"):
        _c = (m.get("clusters") or {}).get(_name)
        if not _c:
            continue
        _gov = "GOVERNS" if _name == "slow" and m.get("bimodal") else \
            ("binds" if _name == "slow" else "reported, NOT relied on")
        _lines.append(
            f"{'':10s}  {_name:4s}[n={_c['n']} mean={hm(_c['mean'])} "
            f"max={hm(_c['max'])} mean+3sd={hm(_c['mean_plus_3sd'])} "
            f"| margin max={_c['margin_max']:+.0f}m "
            f"3sd={_c['margin_3sd']:+.0f}m binding={_c['binding']:+.0f}m] "
            f"{_gov}")
    if m.get("cluster_state") == "UNSEPARABLE":
        _lines.append(
            f"{'':10s}  UNSEPARABLE: {m.get('no_alarm_reason', '')}. It "
            f"REPORTS and does NOT alarm. The conservative reading until the "
            f"series separates is the retired cron's SLOW cluster.")
    _lines.append(
        f"{'':10s}  BINDING {m['binding']:+.0f}m from {m['binding_from']}, "
        f"threshold {MARGIN_ALARM_MIN}m"
        + ("  ** DECISION FIRES **" if m["alarm"] else
           ("  (no alarm: UNSEPARABLE)"
            if m.get("cluster_state") == "UNSEPARABLE" else "")))
    _lines.append(
        f"{'':10s}  successor[p95 delay={_qtxt} ({_verdict}); mean+3sd is "
        f"{'ABOVE' if m['mean_plus_3sd'] > m['max'] else 'within'} the "
        f"observed max {hm(m['max'])}]")
    return chr(10).join(_lines)


# ───────────────────── A RUN THAT NEVER FIRED, AND A FILE GITHUB REFUSED
#
# NEITHER HAD A DETECTOR, and `DID_NOT_RUN` is not one despite its name: it is
# a verdict on `steps_not_run` — a run that EXISTS whose core steps were
# skipped. A workflow that produced no run at all has no row to verdict, so
# every run-based check in this file is blind to it.
#
#     A workflow GitHub refuses does not fail, it disappears. Invalid YAML is
#     an ABSENCE, not a red run.
#
# MEASURED 2026-10-13, and the second half of that is WRONG in a useful way:
# GitHub DOES create a run for an unparseable workflow. Pushing a duplicate
# `env:` to main produced run 38238629301 four seconds later — `event=push`,
# `conclusion=failure`, and NAMED BY THE FILE'S PATH rather than its `name:`
# field, because the `name:` could not be parsed. `gh run list --workflow
# <file>` lists it. So the signal existed and nothing read it; the probe is in
# NEVER_ALARM_WORKFLOWS, so the silence was SCOPE, not timing.
#
# The path-as-name is the discriminator, and it is GitHub's own parser talking.

#: The measured maximum scheduler lateness, used ONLY to decide whether today's
#: missing run is still plausibly coming. MEASURED 2026-10-12: the first
#: weekday run under `0 4` fired 458 minutes late, which replaced the previous
#: maximum of 304 (n=9, under the retired `0 0`/`37 9` regimes). A bound that a
#: single observation has already doubled is not a distribution — it is used
#: here as a floor for PENDING and for nothing else.
OBSERVED_LATENESS_MAX_MIN = 458


def is_unparseable_run(r: dict) -> bool:
    """ONE definition of the predicate.

    It was briefly two - the collector filtered and the formatter counted
    whatever it was handed - so a formatter fed raw rows reported every failed
    run as an unparseable workflow. Caught by passing raw rows to it in a
    test, which is what a test is for.
    """
    return (str(r.get("workflowName") or "").startswith(".github/workflows/")
            and str(r.get("conclusion") or "").lower() == "failure")


def unparseable_workflow_runs(limit: int = 60) -> List[dict]:
    """Runs whose workflow name is a PATH — GitHub could not parse the file.

    This is the only signal GitHub offers, and it is not reachable any other
    way: `gh api .../actions/workflows` reported `state: active` for a file
    containing the exact defect it refuses (measured 2026-10-13), so state is
    not a parse verdict. The dispatch endpoint DOES return GitHub's parser
    error — `422 failed to parse workflow: (Line: N, Col: M)` — but only for a
    file carrying a `workflow_dispatch` trigger, and it CREATES A RUN when the
    file is valid, so it cannot be used as a gate.
    """
    try:
        runs = json.loads(_sh(
            "gh", "run", "list", "--limit", str(limit),
            "--json", "databaseId,workflowName,event,conclusion,createdAt,"
                      "headBranch") or "[]")
    except json.JSONDecodeError:
        return []
    return [r for r in runs if is_unparseable_run(r)]


def format_unparseable(runs: List[dict], default_branch: str = "main") -> str:
    """Alarm on the DEFAULT branch; report a branch push without alarming.

    A branch can be deleted, and an alarm on a deleted branch's push cannot be
    cleared by any action - the `fixtures_zero_active` shape for the seventh
    time. On the default branch it is the real thing: the workflow is not
    scheduled until the file parses, and no other check sees it.
    """
    runs = [r for r in runs if is_unparseable_run(r)]
    on_main = [r for r in runs if r.get("headBranch") == default_branch]
    other = [r for r in runs if r.get("headBranch") != default_branch]
    lines = []
    if on_main:
        lines.append(
            f"::error::{len(on_main)} UNPARSEABLE WORKFLOW push(es) on "
            f"{default_branch}: "
            + ", ".join(f"{r['databaseId']} {r['workflowName']}"
                        for r in on_main)
            + ". GitHub could not parse the file, so it is NOT SCHEDULED - "
              "the run is named by its PATH because its `name:` could not be "
              "read. CLEARING ACTION: fix the YAML and push; `gh api -X POST "
              ".../dispatches` returns GitHub's own parser error with a line "
              "and column.")
    else:
        lines.append(f"unparseable_workflows[0 on {default_branch}]")
    if other:
        lines.append(
            f"  {len(other)} on other branches, reported and NOT alarmed "
            f"(a deleted branch's push cannot be cleared by any action): "
            + ", ".join(sorted({str(r.get("headBranch")) for r in other})))
    return chr(10).join(lines)


def missing_scheduled_runs(workflow: str = "daily-picks.yml",
                           days: int = 14,
                           today: _dt.date = None) -> dict:
    """Days whose scheduled run NEVER EXISTED, and today's PENDING state.

    THREE STATES, because two would lie. A past day with no scheduled run is
    MISSING — a card that was never produced. Today with no run yet is
    PENDING, not missing: the first weekday run under `0 4` arrived **458
    minutes** late (measured 2026-10-12), so "it is 07:00 and nothing has run"
    is not evidence of absence. Reporting PENDING as MISSING would have
    alarmed on a run that was merely very late; reporting MISSING as PENDING
    would never alarm at all.
    """
    today = today or _dt.date.today()
    # ── MIDNIGHT IS FALSY, AND IT KILLED THIS DETECTOR. 2026-10-18. ────────
    #
    # This read `if not any(crons.values())`. A cron is stored as MINUTES PAST
    # MIDNIGHT, so `0 0` is the integer 0 and `any([0, 0])` is False. While
    # the weekday cron was `0 4` (240) the guard was satisfied by the weekday
    # value alone and the detector worked. The revert to `0 0` at 566c927
    # made BOTH day types 0, and this function has returned UNEVALUATED on
    # every call since -- silently, because UNEVALUATED is a legitimate state
    # and its consumer skips on it.
    #
    # The detector it killed is the one built for the 422 window: "a workflow
    # GitHub refuses does not fail, it disappears". It was dead for the whole
    # of its first day, and the skip read as a local environment quirk.
    #
    # AND THE TWO CAUSES WERE ONE REASON. `current_crons_by_daytype()` returns
    # None for an unreadable tree (the shallow-clone third state) and a dict
    # otherwise; `or {}` flattened the first into the second, so an unreadable
    # sha reported "no cron resolved" -- which is a statement about the
    # WORKFLOW, not about our ability to read it.
    crons = current_crons_by_daytype()
    if crons is None:
        return {"state": "UNEVALUATED",
                "reason": "the workflow tree is unreadable at this sha "
                          "(shallow clone? needs fetch-depth: 0)"}
    if not crons or all(v is None for v in crons.values()):
        return {"state": "UNEVALUATED",
                "reason": "no cron resolved for either day type"}
    try:
        runs = json.loads(_sh(
            "gh", "run", "list", "--workflow", workflow, "--limit", "120",
            "--json", "createdAt,event") or "[]")
    except json.JSONDecodeError:
        return {"state": "UNEVALUATED", "reason": "run list unreadable"}

    seen = set()
    for r in runs:
        if r.get("event") != "schedule":
            continue
        try:
            seen.add(_dt.datetime.fromisoformat(
                r["createdAt"].replace("Z", "+00:00")).date())
        except (KeyError, ValueError):
            continue

    missing, pending = [], None
    for back in range(days, -1, -1):
        day = today - _dt.timedelta(days=back)
        dtype = "weekend" if day.weekday() >= 5 else "weekday"
        cron = crons.get(dtype)
        if cron is None:
            continue                      # no cron governs that day type
        if day in seen:
            continue
        if day == today:
            now = _dt.datetime.now(_dt.timezone.utc).replace(
                tzinfo=None)
            elapsed = (now - _dt.datetime.combine(
                day, _dt.time(cron // 60, cron % 60))).total_seconds() / 60
            if elapsed > OBSERVED_LATENESS_MAX_MIN:
                missing.append(day)
            elif elapsed > 0:
                pending = (day, int(elapsed))
        else:
            missing.append(day)
    return {"state": "CHECKED", "missing": missing, "pending": pending,
            "window_days": days, "crons": crons}


def format_missing_runs(m: dict) -> str:
    if m.get("state") != "CHECKED":
        return (f"missing_runs[UNEVALUATED: {m.get('reason')} - this is NOT a "
                f"clean bill]")
    lines = []
    if m["missing"]:
        lines.append(
            f"::error::{len(m['missing'])} scheduled run(s) NEVER EXISTED in "
            f"the last {m['window_days']}d: "
            + ", ".join(str(d) for d in m["missing"])
            + ". These are not late runs; they must NOT enter any delay "
              "series. CLEARING ACTION: check the workflow file parses "
              "(`gh run list` shows a push-event failure named by PATH when "
              "it does not) and that the cron still matches the day type.")
    else:
        lines.append(f"missing_runs[0 in {m['window_days']}d - every day "
                     f"governed by a cron has a scheduled run]")
    if m["pending"]:
        day, elapsed = m["pending"]
        lines.append(
            f"  today ({day}) has no scheduled run yet: {elapsed} min past "
            f"nominal, against a measured maximum of "
            f"{OBSERVED_LATENESS_MAX_MIN} min. PENDING, not missing - a run "
            f"458 min late has already been observed, so this is not yet "
            f"evidence of absence.")
    return chr(10).join(lines)


EMPTY_CARD_FROM = _dt.date(2026, 9, 21)
#: Exclusive. The COVERED EIGHT's card returns here (10-09, 7 fixtures; 41 on
#: 10-10). It is NOT the date the pipeline's card returns — uncovered leagues
#: play throughout, and the next of them is spain/laliga2 on 10-02.
EMPTY_CARD_UNTIL = _dt.date(2026, 10, 9)

#: Findings whose cause is the measured empty card. Matched as substrings of the
#: assertion text, and DELIBERATELY NARROW: only the discovery-zero family. A
#: traceback, a lost message or a skipped step is not explained by an empty card
#: and must still alarm.
EMPTY_CARD_SUPPRESSIBLE = (
    "NO FIXTURES FOUND for the day",
    "fixture scrape(s) attempted, 0 fixtures found in total",
    "0 created AND 0 matched",
    "returned 0 fixtures",
)


def empty_card_window_active(today: _dt.date = None) -> bool:
    """True while the suppression is live. False from 10-09 — by date, not flag."""
    return (today or _dt.date.today()) < EMPTY_CARD_UNTIL


class SuppressionCensus(NamedTuple):
    """VAC-1's counter. The OUTCOME of a filter cannot report that it ran.

    `suppressed=0` was byte-identical for "evaluated 29 findings and cleared
    none" and "was handed nothing and evaluated nothing" — the 09-27 production
    run printed the second and read as the first, and only reading
    `assertions()` and the `disc` field separated them.

    So the census reports the SIZE OF THE SET THE FILTER WAS HANDED, not just
    what it rejected:

      examined    findings the partition evaluated; 0 when it short-circuited
      candidates  of those, the ones ELIGIBLE for suppression (a league-scoped
                  discovery finding, or a run-level suppressible one). This is
                  the number that says whether the filter had anything to work
                  on at all.
      suppressed  cleared
      alarmed     examined - suppressed
      reason      why `examined` is 0, when it is: the filter did not engage
                  (window/expired) or was handed no findings

    PRINTED UNCONDITIONALLY, for the AF_LEAGUE_FILTER reason: a line that
    appears only in the interesting case is a line whose absence means nothing.
    """

    examined: int
    candidates: int
    suppressed: int
    alarmed: int
    reason: str = ""

    def __str__(self) -> str:
        return (f"suppression[examined={self.examined} "
                f"candidates={self.candidates} suppressed={self.suppressed} "
                f"alarmed={self.alarmed}"
                + (f" NOT-ENGAGED:{self.reason}]" if self.reason else "]"))


def _is_suppression_candidate(h: str) -> bool:
    """Eligible for suppression at all — league-scoped, or run-level suppressible.

    Kept next to the partition and used BY it, so the census cannot drift from
    the decision it describes. A second implementation of this predicate would
    be the three-phrasings habit again.
    """
    return bool(_LEAGUE_IN_FINDING.search(h)) or \
        any(m in h for m in EMPTY_CARD_SUPPRESSIBLE)


def partition_empty_card(hits, run_date, today=None, zero_leagues=None):
    """`(kept, suppressed, census)`. Suppressed carry the reference cell that
    excuses them; the census reports what was EXAMINED — see `SuppressionCensus`.

    THREE conditions, all required:

      1. the RUN's date is inside the measured empty window
      2. the suppression itself has not expired (gates on TODAY, so an old run
         audited after 10-09 is not retroactively re-alarmed)
      3. EVERY league that reported a zero is one the reference COVERS

    The third was added the same day as the first two, because the suppression
    was applied to 30 leagues on the strength of a grid covering 8. A discovery
    zero from any of the other 22 is unreferenced and must still alarm.

    `zero_leagues=None` means the leagues were not determined — which FAILS
    CLOSED and suppresses nothing, rather than assuming the benign case.
    """
    # SHORT-CIRCUITS EXAMINE NOTHING, and say so. Before the census these
    # returned the same `[]` as a filter that evaluated every finding and
    # cleared none.
    _cand = sum(1 for h in hits if _is_suppression_candidate(h))
    if not (run_date and EMPTY_CARD_FROM <= run_date < EMPTY_CARD_UNTIL):
        return list(hits), [], SuppressionCensus(
            0, _cand, 0, 0, "window" if run_date else "run-date-unparseable")
    if not empty_card_window_active(today):
        return list(hits), [], SuppressionCensus(0, _cand, 0, 0, "expired")
    kept, supp = [], []
    for h in hits:
        # CONDITION 3, PER FINDING. A league-scoped finding carries its own
        # league, so it is cleared on its own evidence. Before 2026-09-27 the
        # findings were run-level and this could only be applied to the whole
        # run — which meant one unreferenced league in a set of 29 left
        # everything unexplained.
        _m = _LEAGUE_IN_FINDING.search(h)
        if _m:
            if _m.group(1) in EMPTY_CARD_COVERED_LEAGUES:
                supp.append(f"{h}  [SUPPRESSED: reference openfootball 2026-27 "
                            f"shows 0 fixtures for this league on {run_date}; "
                            f"card returns {EMPTY_CARD_UNTIL}]")
            else:
                kept.append(h)          # unreferenced league -> still alarms
            continue
        # RUN-LEVEL findings keep the old all-or-nothing rule: they name no
        # league, so they are only clearable when EVERY zero-reporting league is
        # covered. `None` or `[]` suppresses nothing — fail closed.
        if not zero_leagues or (set(zero_leagues) - EMPTY_CARD_COVERED_LEAGUES):
            kept.append(h)
        elif any(m in h for m in EMPTY_CARD_SUPPRESSIBLE):
            supp.append(f"{h}  [SUPPRESSED: reference openfootball 2026-27 shows "
                        f"0 fixtures in all 8 covered leagues on {run_date}; "
                        f"card returns {EMPTY_CARD_UNTIL}]")
        else:
            kept.append(h)
    return kept, supp, SuppressionCensus(
        examined=len(hits), candidates=_cand, suppressed=len(supp),
        alarmed=len(kept),
        # ENGAGED BUT HANDED NOTHING is its own state, and it is the one the
        # 09-27 run was in. Distinguished from `examined>0, suppressed=0`.
        reason="" if hits else "no-findings")


# ───────────────────────────────────────────────── self-calibrating assertions

def assertions(facts: Dict[str, object],
               history: List[Dict[str, object]]) -> List[str]:
    """Fire only when a unit that RECENTLY produced data produces none.

    `history` is the same fact dict for the previous runs of this workflow, most
    recent last. An empty history means nothing can be said — and saying nothing
    is correct, not a pass.
    """
    hits: List[str] = []

    def produced_recently(key: str) -> bool:
        return any((h.get(key) or 0) > 0 for h in history[-LOOKBACK_RUNS:])

    for key, label in (("picks_saved", "picks"),
                       ("odds_rows", "Odds API rows"),
                       ("injuries_saved", "injuries"),
                       ("clv_pairs", "valid CLV pairs")):
        if key in facts and (facts.get(key) or 0) == 0 and produced_recently(key):
            hits.append(
                f"{label} = 0, but this workflow produced {label} within the "
                f"last {LOOKBACK_RUNS} runs")

    # PER-SOURCE discovery, on the day's first run only.
    #
    # The aggregate version shipped earlier in Stage 19 would NOT have fired on
    # 2026-05-31, the day Flashscore went silent: flashscore=0,
    # football-data.org=0, API-Football=13, so the TOTAL looked healthy. It then
    # stayed silent for 88 days.
    #
    # A fallback that substitutes silently makes the primary's failure
    # invisible. Redundancy that is not checked PER COMPONENT is not redundancy
    # — it is one working source and two unverified claims.
    if facts.get("is_first_run_of_day", True):
        for key, mkey, label in (
                ("src_flashscore_fixtures", "src_flashscore_matched",
                 "Flashscore fixtures"),
                ("src_footballdataorg_fixtures", "src_footballdataorg_matched",
                 "football-data.org fixtures"),
                ("src_apifootball_fixtures", "src_apifootball_matched",
                 "API-Football fixtures")):
            if key not in facts or not produced_recently(key):
                continue
            created = facts.get(key) or 0
            matched = facts.get(mkey) or 0
            # ALIVE IF created + matched > 0. Zero creations alone is novelty
            # falling to zero, which happens whenever another source reached
            # the fixture first — the steady state, not a fault.
            if created + matched > 0:
                continue
            hits.append(
                f"{label}: 0 created AND 0 matched while other sources still "
                f"produce — this source produced within the last "
                f"{LOOKBACK_RUNS} runs and is now silent on both counts")

    # A REMINDER THAT COULD NOT BE IMPORTED DID NOT RUN, and the run is GREEN.
    # Unconditional: unlike the rules above this is not "a producer fell
    # silent", so it needs no history and must fire on the FIRST occurrence —
    # a reminder's whole job is to fire once, on a date, and a missed date
    # cannot be caught up.
    _ce = facts.get("reminder_collect_errors")
    if _ce:
        hits.append(
            f"reminders: {_ce} test COLLECTION ERROR(S) and the run concluded "
            f"green — a reminder module that cannot be imported never fires, "
            f"and no due decision was made. This is NOT 'no reminder was due'")

    picks = facts.get("picks_saved") or facts.get("picks_saved_live") or 0
    obs = facts.get("observations")
    if picks and obs is not None and obs != 2 * picks:
        hits.append(f"pick_observations {obs} != 2 x {picks} picks saved")

    # NOT self-calibrating, and deliberately so: spending a credit and getting
    # no rows back is wrong on the first occurrence, not relative to history.
    # This is the condition the manual pass used to mark 9 runs DEGRADED.
    if facts.get("no_rows"):
        hits.append(
            f"{facts['no_rows']} league request(s) returned no_rows — credits "
            "were spent and the provider returned an empty event list")
    if (facts.get("credits_claimed") or 0) > 0 and (facts.get("captured") or 0) == 0:
        hits.append(
            f"{facts['credits_claimed']} credit(s) claimed, 0 closing lines "
            "captured")

    # Also not self-calibrating: a review decision that was computed and then
    # discarded is wrong on the first occurrence. The reviewed pick silently
    # keeps whatever the model chose, and the KEEP/CHANGE record gains a gap.
    if facts.get("decisions_discarded"):
        hits.append(
            f"{facts['decisions_discarded']} briefing decision(s) DISCARDED — "
            "the review ran and its verdict was thrown away")

    # Stage 19 — discovery. NOT self-calibrating: a fixture scrape that returns
    # nothing for a league the scraper itself calls in-season is wrong on the
    # first occurrence. MEASURED 2026-08-26: this had been true on EVERY run
    # since 2026-05-30 — 88 days, 200+ attempts, zero fixtures — and nothing
    # reported it, because API-Football was quietly covering for it until its
    # suspension on 08-19.
    if facts.get("fixtures_zero_active"):
        hits.append(
            f"{facts['fixtures_zero_active']} active-season league(s) returned "
            "0 fixtures — discovery produced nothing the scraper expected")
    if facts.get("fixture_attempts") and not facts.get("fixtures_scraped"):
        hits.append(
            f"{facts['fixture_attempts']} fixture scrape(s) attempted, "
            "0 fixtures found in total")
    if facts.get("no_fixtures_at_all"):
        hits.append("NO FIXTURES FOUND for the day — nothing was analysed")

    # NOT self-calibrating: a fixture that could not be priced because a team
    # would not resolve is wrong on the first occurrence, not relative to
    # history. It is the shape that hid for three days as "nobody bet that one".
    if facts.get("unpriced_fixtures"):
        hits.append(
            f"{facts['unpriced_fixtures']} row(s) carry NO ODDS while "
            "same-league peers do (cause not established: unresolved team, "
            "or a duplicate row holding the same fixture)")

    # A check that did not run is not a clean check.
    if facts.get("implausible_rows"):
        hits.append(
            f"{facts['implausible_rows']} team row(s) carry fixtures from more "
            "than one DOMESTIC country — a club plays in one pyramid, so one "
            "group belongs to another club (which group is NOT decided here)")

    if facts.get("implausible_check_dead"):
        hits.append("the implausible-attribution check DID NOT RUN — no "
                    "evidence either way, not a clean result")

    if facts.get("unpriced_check_dead"):
        hits.append("the unpriced-fixture check DID NOT RUN — no evidence "
                    "either way, not a clean result")

    if facts.get("duplicate_fixture_refusals"):
        hits.append(
            f"{facts['duplicate_fixture_refusals']} pick(s) refused because two "
            "ROWS were one FIXTURE — s5.9 acting, not the ordinary per-match cap")

    if facts.get("picks_run_guard_declined"):
        hits.append("the odds refresh DECLINED — closing-line capture stopped "
                    "for this run; every pending pick will be rejected as late")

    # A report that stops carrying the experiment block is silent otherwise:
    # the message still sends, it just says less. Absence is the finding.
    if facts.get("report_sent") and not facts.get("experiment_record_built"):
        hits.append("a report was sent WITHOUT the experiment record block — "
                    "the paper series silently stopped being reported")

    if facts.get("tracebacks"):
        hits.append(f"{facts['tracebacks']} traceback(s) in the log")
    if facts.get("account_suspended"):
        hits.append("API-Football reported the account suspended")
    if facts.get("telegram_lost"):
        hits.append(
            f"{facts['telegram_lost']} message(s) LOST — a report part the "
            "retry did not recover, or an alert that exhausted its attempts")

    if facts.get("telegram_unexplained"):
        hits.append(
            f"{facts['telegram_unexplained']} Telegram failure(s) UNEXPLAINED "
            "— no REPORT_DELIVERY or ALERT line accounts for them, so whether "
            "they were recovered is UNKNOWN (not assumed either way)")

    # ---- DEL-3, three assertions, each on a DIFFERENT failure shape -------
    #
    # A lost middle chunk and a stream that stopped early are not the same
    # event and do not present the same way. One leaves a hole and ends
    # normally; the other ends early and leaves no hole. A single assertion
    # would catch whichever was written first and miss the other — which is
    # precisely how the defect survived: `_send_chunked` returned the LAST
    # chunk's Message, so it was blind to holes and blind to nothing else.
    for d in (facts.get("report_deliveries") or []):
        # 1. A HOLE. Parts failed; the report reached a reader incomplete.
        if d["failed"]:
            hits.append(
                f"the {d['report']} was delivered INCOMPLETE — part(s) "
                f"{d['failed']} of {d['chunks']} failed to send; a reader sees "
                f"a report that ends normally")
        # 2. AN EARLY END. No terminator, so the stream stopped before the
        #    last part and nothing in it says so.
        elif not d["terminator"]:
            hits.append(
                f"the {d['report']} carries NO TERMINATOR — the final part "
                f"never sent, so the message a reader has is truncated with "
                f"nothing marking the cut")
        # 3. ARITHMETIC. sent + failed must equal chunks. A mismatch with an
        #    empty `failed` list means a part went missing WITHOUT being
        #    recorded, which is the original defect returning by another route.
        if d["sent"] + len(d["failed"]) != d["chunks"]:
            hits.append(
                f"the {d['report']}'s parts do not add up: {d['sent']} sent + "
                f"{len(d['failed'])} failed != {d['chunks']} chunks — a part "
                f"went missing without being recorded")
    if facts.get("steps_failed"):
        hits.append(f"core step(s) reported failure: {facts['steps_failed'][-1]}")
    # PER-LEAGUE DISCOVERY FINDINGS. One per league that found nothing, so a
    # mixed day is PARTIALLY explained rather than wholly unexplained. `kept`
    # and `off-season` produce no finding — a league that delivered fixtures and
    # a league the config calls dormant are both expected.
    for d in (facts.get("league_discovery") or []):
        if d["state"] in ("none-in-range", "no-rows"):
            hits.append(
                f"discovery: {d['league']} found 0 fixtures "
                f"(state={d['state']}, rows={d['rows']}, "
                f"earliest_parsed={d['earliest']}, cutoff={d['cutoff']})")

    if facts.get("steps_not_run"):
        hits.append(f"core step(s) DID NOT RUN: {facts['steps_not_run'][-1]} "
                    "— nothing crashed and nothing ran")

    # ABSORBED FAILURE. A step exited non-zero and NOTHING ELSE in this audit
    # accounts for it — not the workflow's own failure alert, not the skip
    # detection. Under `continue-on-error` that step reports
    # `conclusion: success` to the API, so this annotation is the only record.
    #
    # NOT self-calibrating: a step exiting non-zero is wrong on the first
    # occurrence, like a spent credit that returned no rows.
    #
    # THE GUARDS EXIST BECAUSE THE FIRST VERSION DOUBLE-REPORTED. It fired on
    # the 09-24 closing-lines psycopg failures — and `closing-lines` carries
    # ZERO continue-on-error steps, so nothing there was absorbed and the word
    # was simply wrong. The claim is about a failure that is OTHERWISE
    # INVISIBLE, so every other way the audit already sees one disqualifies it:
    #
    #   steps_failed    the workflow's own alert named a failed core step
    #   steps_not_run   a step was skipped, so the job halted visibly
    #   tracebacks      the failure printed a traceback and is already BROKEN
    #
    # Found by running it against the real logs rather than the synthetic ones.
    # SATURATION REMOVED 2026-10-02: the predicate is the REMAINDER after the
    # repository's own deliberate annotations are subtracted, not the raw exit
    # count. On an alarming `ci-audit` run that remainder is 0, where the raw
    # count was 2 — its own baseline.
    if (facts.get("unexplained_nonzero_exit")
            and not facts.get("steps_failed")
            and not facts.get("steps_not_run")
            and not facts.get("tracebacks")):
        # WORDING CORRECTED 2026-09-27, on its first real firing. It said the
        # steps "were ABSORBED", and on the eight ci-audit runs that is true of
        # one of the two exits and false of the other — the second exit IS the
        # job's red. The count is of NON-ZERO EXITS; whether each was absorbed
        # depends on that step's continue-on-error, which the API does not
        # expose. Claiming absorption for all of them over-attributes.
        hits.append(
            f"{facts['unexplained_nonzero_exit']} step(s) exited NON-ZERO with "
            f"no traceback and no named failed step "
            f"({facts.get('self_emitted_exits', 0)} further exit(s) explained "
            f"by this repository's own deliberate annotations) — any of them "
            f"under continue-on-error reports conclusion: success, so this "
            f"annotation is the only record")
    return hits


def discovery_summary(facts: Dict[str, object]) -> str:
    """`disc[fs=N fdo=N af=N]` — per source, never an aggregate.

    Printed on every daily-picks row and carried into the ledger note. The
    total is the number that hid Flashscore's death from 2026-05-30 to
    2026-08-26; only the per-source split makes a silent substitution visible.
    Absent (rather than 0) is shown as `-`, because "did not report" and
    "reported nothing" are different facts.

    Shown as `created/matched` since 2026-09-09. A bare creation count read
    identically for a healthy source that matched everything and a dead one
    that returned nothing — it cried wolf on every daily-picks run for five
    days, which trains the reader to skip the very line that would catch the
    88-day failure this assertion exists for.
    """
    keys = (("fs", "src_flashscore_fixtures", "src_flashscore_matched"),
            ("fdo", "src_footballdataorg_fixtures", "src_footballdataorg_matched"),
            ("af", "src_apifootball_fixtures", "src_apifootball_matched"))
    if not any(k in facts for _, k, _m in keys):
        return ""
    parts = []
    for label, ckey, mkey in keys:
        c = facts[ckey] if ckey in facts else "-"
        m = facts[mkey] if mkey in facts else "-"
        parts.append(f"{label}={c}c/{m}m")
    return "disc[" + " ".join(parts) + "]"


def resolution_summary(facts: Dict[str, object]) -> str:
    """`resolve[provider_id=N former_name=N exact=N strict=N create=N]`.

    The composition of team identity resolution for a run, from s5.13's
    `TEAM_RESOLVE` line. Printed beside `disc[...]` because it answers the
    question `disc[...]` cannot: a fixture count says how much was scraped, this
    says how each team NAME in it was decided.

    `former_name` is the one to watch — `resolve_team` step 2 had never fired in
    production as of 2026-09-16, so its first non-zero here is the first
    evidence that branch is reachable at all.
    """
    steps = facts.get("team_resolve_steps")
    if not steps:
        return ""
    order = ("provider_id", "former_name", "exact_name", "strict",
             "create", "no_match")
    parts = [f"{k}={steps[k]}" for k in order if steps.get(k)]
    return "resolve[" + " ".join(parts) + "]" if parts else ""


#: A log shorter than this carries no step output — a run that is still
#: executing, or whose log could not be fetched. Not a threshold on quality:
#: a real daily-picks log is ~600k and the smallest real capture log ~10k.
_MIN_AUDITABLE_LOG_BYTES = 200


def verdict(facts: Dict[str, object], hits: List[str],
            log: str = None) -> str:
    """CLEAN means the checks ran and found nothing. It cannot mean nothing ran.

    THIRD INSTANCE OF THE SAME COLLAPSE IN THIS TOOL. `[]` versus `None` in the
    odds path, 429-with-credits versus 429-with-zero, and now an empty result
    versus an unmeasured one — all three returned the same value for "found
    nothing" and "could not look".

    Found 2026-09-13: run 34745992077 (daily-picks, 09-13 07:44) had a
    ZERO-BYTE cached log and this function returned CLEAN. No assertion can fire
    against an empty file, so `hits` was empty, so the run scored clean. A
    verdict from no evidence.

        No evidence is not a clean verdict.

    `UNAUDITABLE` is named and counted so it appears in the ledger as its own
    row rather than inflating the CLEAN count — which is exactly how nine
    DEGRADED runs went unnoticed in the pass this tool was built to replace.
    """
    # A run still executing has no verdict to give. Its log is empty because
    # `gh run view --log` returns nothing until completion, NOT because
    # anything failed — found 2026-09-13 on run 34745992077, which was 20
    # minutes old and mid-step-15 when it was audited and reported UNAUDITABLE.
    # Checked BEFORE the log test so an in-flight run is never mistaken for a
    # missing one: both look identical from the log alone, and only the run
    # metadata separates them.
    if str(facts.get("run_status") or "").lower() in ("in_progress", "queued",
                                                      "waiting", "pending",
                                                      "requested"):
        return "IN_PROGRESS"
    if log is not None and len(log.strip()) < _MIN_AUDITABLE_LOG_BYTES:
        return "UNAUDITABLE"
    if facts.get("steps_failed") or facts.get("tracebacks"):
        return "BROKEN"
    # FOURTH INSTANCE, and the vocabulary is GitHub's own. `outcome` is
    # three-valued and every reader here treated it as two. A run whose core
    # steps were SKIPPED did not crash — nothing to call BROKEN — and did not
    # work — nothing to call CLEAN. Checked AFTER `steps_failed`, because a run
    # that both failed and skipped is broken first.
    if facts.get("steps_not_run"):
        return "DID_NOT_RUN"
    return "DEGRADED" if hits else "CLEAN"


# ────────────────────────────────────────────────────────────────────── main

#: AN ACKNOWLEDGED STANDING BREACH. CLR-2, applied to a condition rather than
#: an event.
#:
#: MEASURED 2026-10-13: seven consecutive `ci-audit` runs went red with the
#: IDENTICAL line `schedule margin -85m < 30m`. Seven reds, ONE finding - the
#: same shape as the nine reds of 10-10. The cause is a single observation (the
#: 2026-10-12 weekday run, 458 min late) sitting inside a 28-day window, so the
#: alarm would fire on EVERY audit run for 28 days with nothing a reader could
#: do about it. That is `fixtures_zero_active` for the seventh time: an alarm
#: that cries wolf until it is ignored, and it reaches that state in a day.
#:
#: So a breach that has been SEEN, RECORDED and has a dated decision attached
#: is REPORTED and does not alarm. It alarms again the moment it gets WORSE
#: than the acknowledged figure, because deterioration is an event.
#:
#: The clearing action is an action a person takes: the cron decision registered
#: in `docs/scheduler-hour-hypothesis.md`, due after 2026-10-17.
# ── THE 10-18 REVIEW, CONDUCTED. THE ACKNOWLEDGEMENT IS NOT RENEWED. ───────
#
# It worked exactly as designed, and the first thing it did on a real test was
# REFUSE. Measured 2026-10-18: the weekday series grew n=1 -> n=5 and the
# binding margin deteriorated from -85m to **-241m**, so
# `margin_breach_is_acknowledged('weekday', -241.4)` returns {} and the margin
# alarms again. Deterioration is an event.
#
#     weekday `0 4`   n=5  min 283  mean 357 (5h57m)  max 458 (7h38m)  sd 77
#     weekend `0 0`   n=6  min 219  mean 253 (4h13m)  max 294 (4h54m)  sd 27
#
# The entry below is LEFT AT -85 ON PURPOSE. Re-acknowledging at -241 would
# absorb a condition that got four times worse, which is the one thing an
# acknowledgement must never do. It alarms until the cron decision is taken —
# that is the clearing action, and it is now the blocking item.
# ── THE ACKNOWLEDGEMENT IS RETIRED 2026-10-20: OBSOLETE, NOT MERELY EXPIRED ──
#
# It read `binding=-85.0, since=2026-10-12, review=2026-10-18` and acknowledged
# ONE observation: the first weekday run under `0 4` firing 458 min late.
#
# THREE THINGS ENDED IT, and only the first was noticed:
#
#   1. its review date passed on 2026-10-18 -- and NOTHING ENFORCED THAT.
#      `margin_breach_is_acknowledged` compares only `binding`, never
#      `review`, so an acknowledgement whose review date has gone by keeps
#      absorbing. Measured today: the weekday binding is **+21m** against a
#      30m threshold, a real breach, and it was being absorbed silently by a
#      guard that expired two days ago.
#   2. the decision it was WAITING FOR was taken: the weekday cron was
#      reverted to `0 0` on 2026-10-18, so the `0 4` regime it describes no
#      longer exists.
#   3. its figure is meaningless now. `-85.0` was a `0 4` number; under `0 0`
#      the binding is +21m. An acknowledgement that cannot be compared to the
#      thing it covers is not a deferral, it is a hole.
#
# So it is removed rather than renewed, and the breach it was covering now
# REPORTS. `margin_breach_is_acknowledged` is kept -- an empty register is the
# correct state for "nothing is deferred", and keeping the function means the
# next deferral is a data entry rather than a code change. It now also refuses
# an entry whose review date has passed, so this cannot recur.
#: ── THE STANDING WEEKDAY BREACH, ACKNOWLEDGED 2026-10-22 ──────────────────
#:
#: THE FIGURE. `measured` 2026-10-22: the weekday binding is
#: **20.665799601890512**, against `MARGIN_ALARM_MIN` of 30. It DISPLAYS as
#: `+21`, and the display is not what is stored -- on 2026-10-21 an
#: acknowledgement recorded at the rounded `19.0` correctly REFUSED a real
#: binding of 18.89232917906861, and that refusal is the reason this entry
#: carries full precision in `measured_at` and a deliberate FLOOR in
#: `binding`.
#:
#: WHY IT CANNOT CLEAR. The deadline is the OBSERVED EARLIEST KICKOFF (weekday
#: 09:30) and the ready time is ours; no choice of deadline moves a kickoff.
#: `schedule-deadline-is-data-not-a-constant` already drove exposure to zero
#: days over 60m and the reference bound on 0 of 834 days, so the remaining
#: deficit is cron-and-duration, not deadline. It is true on essentially every
#: weekday -- MEASURED over a 30-run sample of alarming `ci-audit` runs, this
#: alarm appeared in 30 of 30 -- and no action available to the operator
#: clears it on the day. That is CLR-2's definition of a thing that must
#: report rather than alarm.
#:
#: AND THE READING IS CONSERVATIVE, TWICE OVER, which is why acknowledging it
#: is not merely tolerating it:
#:
#:   1. THE PICKS TERM IS FROM A RETIRED REGIME. It uses
#:      `FULL_CARD_PICKS_MAX = 88.1`, a full-card MAX from the SERIAL review
#:      (n=4, 2026-10-01). The replacement series under the parallel review
#:      stands at n=1 and its one observation is 35.1 minutes.
#:   2. THE LATENESS TERM IS BIMODAL, and `mean+3sd` over a bimodal series
#:      measures BETWEEN-CLUSTER SPREAD, not tail risk. `measured` 2026-10-22,
#:      weekday n=10: 2026-10-02..10-09 give 255.9-304.9 minutes, then a
#:      ten-day gap, then 2026-10-19..10-22 give 128.4-144.1. The cron did NOT
#:      change across that gap -- it is `0 0 * * 1-5` throughout -- so this is
#:      the environment, not a regime of ours, and splitting the series on an
#:      eyeballed gap at n=4 is the error
#:      `settled-pick-segments-are-noise` records. Pooled, the spread does the
#:      work: adding the first low point moved the binding -97.91 minutes in
#:      one step while the LEVEL improved.
#:
#: THE CLEARING ACTION, and it is an action with a date rather than a wait:
#: the parallel-regime series reaching n=10 and `FULL_CARD_PICKS_MAX` being
#: re-derived from it. Until then the operator dispatches `daily-picks` by
#: hand on the morning, which skips the measured 3h58m-5h04m scheduler
#: lateness outright.
#:
#: THE FLOOR, AND WHY IT IS NOT THE READING. `binding` below is the figure the
#: acknowledgement covers DOWN TO, not the figure measured. It sits
#: `JITTER` minutes under the reading because the binding moves on its own as
#: ordinary points arrive: `measured` 2026-10-22 by recomputing it over
#: growing prefixes of the weekday series, the single-point moves were +5.53,
#: +2.79, -11.32, -2.24 and +1.77 minutes (excluding -97.91, which is the
#: cluster boundary entering the series). The largest ordinary move is 11.32,
#: so a floor at `reading - 11.32` alarms on a real deterioration and not on
#: the next scheduled run. An acknowledgement pinned to the reading itself
#: would re-alarm within a day or two and the daily red would be back.
ACKNOWLEDGED_MARGIN_BREACH_JITTER = 11.32
ACKNOWLEDGED_MARGIN_BREACH: dict = {
    "weekday": {
        "since": "2026-10-22",
        "measured_at": 20.665799601890512,
        "binding": 20.665799601890512 - ACKNOWLEDGED_MARGIN_BREACH_JITTER,
        "jitter": ACKNOWLEDGED_MARGIN_BREACH_JITTER,
        "why": ("the deadline is the observed earliest kickoff and the ready "
                "time is ours, so no deadline choice clears it; and the "
                "reading is CONSERVATIVE on both terms - the picks bound is a "
                "full-card MAX from the retired SERIAL regime (88.1m, n=4) "
                "while the parallel series' one observation is 35.1m, and the "
                "lateness series is BIMODAL (n=10: 255.9-304.9 then "
                "128.4-144.1, same cron) so mean+3sd is measuring "
                "between-cluster spread rather than tail risk"),
        "action": ("dispatch daily-picks by hand on the morning (it skips the "
                   "measured 3h58m-5h04m scheduler lateness); the durable "
                   "clearing action is the parallel-regime picks series "
                   "reaching n=10 and FULL_CARD_PICKS_MAX being re-derived "
                   "from it"),
        "review": "2026-11-04",
    },
}


def margin_breach_is_acknowledged(day_type: str, binding: float) -> dict:
    """The acknowledgement, or `{}`. Worse than acknowledged is NOT covered."""
    ack = ACKNOWLEDGED_MARGIN_BREACH.get(day_type)
    if not ack:
        return {}
    # AN EXPIRED ACKNOWLEDGEMENT ABSORBS NOTHING. Added 2026-10-20: the
    # previous entry's `review` date passed on 2026-10-18 and it kept
    # absorbing a +21m weekday breach, because this function compared only
    # `binding`. A deferral with a date that nothing reads is not a deferral.
    review = ack.get("review")
    if review:
        try:
            if _dt.date.today() > _dt.date.fromisoformat(review):
                return {}
        except ValueError:
            return {}          # an unparseable date absorbs nothing either
    # MORE negative than the recorded figure is a NEW observation, and a
    # standing acknowledgement must not absorb a deterioration.
    return ack if binding >= ack["binding"] else {}


def _write_alarm_body(alarmed: List[str]) -> None:
    """Hand the REASONS to the workflow, so the alert is not a guess.

    Until 2026-10-13 the alert read "a run scored BROKEN or DID_NOT_RUN, or
    the audit itself failed" for EVERY alarm. On 2026-10-12 the actual cause
    was `schedule margin -85m < 30m`, which is none of the three - the same
    defect as the probe's single fixed sentence, in the tool that found it.
    """
    path = _os.environ.get("AUDIT_ALERT_BODY")
    if not path:
        return
    # AN EMPTY LIST WRITES AN EMPTY FILE, and that is load-bearing. The
    # workflow sends the alert only `if [ -s "$BODY_FILE" ]`, so this is how
    # "every alarm this run raised was run status" becomes "no Telegram
    # message". Writing the bare heading would have left a 17-byte file and
    # pushed a message with no reasons in it.
    body = ("CI audit alarmed:" + chr(10)
            + chr(10).join(f"  - {a}" for a in alarmed)) if alarmed else ""
    try:
        pathlib.Path(path).write_text(body, encoding="utf-8")
    except OSError as exc:                      # pragma: no cover
        print(f"could not write the alarm body: {exc}")


#: ── RED MEANS NEW, UNACKNOWLEDGED, OR DETERIORATED. 2026-10-23. ──────────
#:
#: `ci-audit` reddened on every run for two weeks and the answer was always
#: "one finding, a verdict by design". A workflow that reddens whenever a
#: KNOWN state exists reddens every day, and then red carries no information
#: at all -- which is how the only channel Niki reads became GitHub's own
#: notification list.
#:
#: So a finding is classified before it decides the exit code:
#:
#:   NEW            no recorded occurrence                   -> RED
#:   DETERIORATED   worse than the recorded figure            -> RED
#:   UNACKNOWLEDGED recorded but its review date has passed    -> RED
#:   UNCHANGED      within the tolerance of the recorded one  -> prints
#:
#: AND MACHINERY ALWAYS REDDENS, whatever the finding state. That is handled
#: outside this table, in `_main_guarded`'s exit 2 and in the workflow's
#: crash branch, so no classification here can silence a broken audit.
#:
#: THE TOLERANCE IS DERIVED, NOT CHOSEN. For a margin-shaped figure it is the
#: largest ORDINARY single-point move of the binding, `measured` 2026-10-22 by
#: recomputing it over growing prefixes of the weekday series: +5.53, +2.79,
#: -11.32, -2.24, +1.77 (the -97.91 cluster boundary excluded). Same figure
#: the standing acknowledgement's floor uses, and for the same reason -- a
#: statistic that moves 11 minutes on one arrival must not call that a
#: deterioration.
#:
#: FOR A COUNT the tolerance is ZERO: one more absent scheduled run, or one
#: more failing pattern, is a new fact and not jitter.
FINDING_TOLERANCE_MINUTES = ACKNOWLEDGED_MARGIN_BREACH_JITTER      # 11.32
FINDING_TOLERANCE_COUNT = 0

#: THE RECORDED OCCURRENCES. A source-level table, deliberately: CI cannot
#: write back, and in this project recording a figure is already a human act
#: with an argument beside it (`RESOLVED_RUNS`, `ACKNOWLEDGED_MARGIN_BREACH`,
#: `STRUCTURAL_ACK`). Writing the figure down IS the acknowledgement.
#:
#: `figure` is the value last reported; `tolerance` names which of the two
#: derived tolerances applies; `review` is the date after which it reddens
#: again whatever its figure.
KNOWN_FINDINGS: Dict[str, dict] = {
    "margin.schedule": {
        "figure": 20.665799601890512,
        "tolerance": "minutes",
        "since": "2026-10-22",
        # 2026-11-06 here is the projected date the parallel picks series
        # reaches n=10, NOT the R^2=0.451 fit that shares the date elsewhere
        # in this file. Two different projections landing on one day is
        # exactly how a figure acquires the wrong provenance.
        "review": "2026-11-06",
        "why": ("the deadline is the observed earliest kickoff and the ready "
                "time is ours; the slow cluster governs and the cron was "
                "derived from it"),
        "action": ("dispatch daily-picks by hand that morning; durably, the "
                   "parallel picks series reaching n=10"),
    },
}


def classify_finding(key: str, figure, today=None) -> tuple:
    """`(state, why)` for one finding. `state` decides whether red is earned.

    LOWER IS WORSE for a margin and HIGHER IS WORSE for a count, so the
    comparison direction comes from the tolerance kind rather than from the
    sign of the number -- a figure's meaning is not recoverable from its
    value.
    """
    rec = KNOWN_FINDINGS.get(key)
    if rec is None:
        return ("NEW", "no recorded occurrence for this finding")
    today = today or _dt.date.today()
    review = rec.get("review")
    if review:
        try:
            if today > _dt.date.fromisoformat(review):
                return ("UNACKNOWLEDGED",
                        f"recorded, but its review date {review} has passed")
        except ValueError:
            return ("UNACKNOWLEDGED",
                    f"recorded with an unparseable review date {review!r}")
    if figure is None or rec.get("figure") is None:
        return ("UNCHANGED", "recorded, and neither occurrence carries a "
                             "figure to compare")
    kind = rec.get("tolerance", "minutes")
    # TWO STATEMENTS, NOT A TUPLE. `tol, worse = TOL, figure > rec + tol`
    # reads `tol` on the right before the left side binds it --
    # `UnboundLocalError`, and the count branch had never been exercised until
    # the control for it was written. A tolerance kind that nothing drove was
    # dead code wearing a parameter's clothes.
    if kind == "count":
        tol = FINDING_TOLERANCE_COUNT
        worse = figure > rec["figure"] + tol
    else:
        tol = FINDING_TOLERANCE_MINUTES
        worse = figure < rec["figure"] - tol
    if worse:
        return ("DETERIORATED",
                f"{figure:+.2f} against the recorded {rec['figure']:+.2f} "
                f"(tolerance {tol} {kind})")
    return ("UNCHANGED",
            f"{figure:+.2f} is within {tol} {kind} of the recorded "
            f"{rec['figure']:+.2f}; review {review}")


#: The states that earn a red. UNCHANGED prints and does not.
FINDING_STATES_THAT_REDDEN = ("NEW", "DETERIORATED", "UNACKNOWLEDGED")


def _emit_alarms(alarmed: List[str], alarmed_status: List[str],
                 keyed=()) -> int:
    """Print every alarm, push only the actionable ones, return the exit code.

    `keyed` is `[(finding_key, figure, text), ...]` -- findings that carry
    their identity from the site that raised them, so a KNOWN one can print
    without spending the run's colour. See `KNOWN_FINDINGS`.

    THE EXIT CODE COVERS BOTH LISTS, deliberately. A status alarm is not sent
    to Telegram, and the thing that carries it instead is THIS RUN'S RED --
    which only exists if a status alarm still exits non-zero. Stop reddening
    and the information has nowhere left to go, which would turn a removal
    into a deletion.
    """
    both = alarmed + alarmed_status
    if not both and not keyed:
        return 0

    # ── CLASSIFY THE KEYED FINDINGS BEFORE THE EXIT CODE. 2026-10-23. ────
    #
    # A keyed finding carries its identity and its figure from the site that
    # RAISED it, never recovered from its wording. Unkeyed alarms are treated
    # as NEW and always redden, which is correct for every one of them today:
    # the pattern self-test, model integrity, an absent scheduled run, an
    # untriaged finding, an unparseable workflow, an unevaluated workflow
    # state, a missing cron and the actionable margin are all RARE, and a
    # per-run verdict relay is new for each run by construction.
    quiet, loud = [], []
    for key, figure, text in keyed:
        state, why = classify_finding(key, figure)
        line = f"{text}  [{state}: {why}]"
        # EVERY KEYED FINDING IS ACTIONABLE TODAY. A keyed STATUS finding
        # would need routing away from Telegram the way `alarmed_status` is;
        # the seam for it was built and removed rather than left empty, since
        # a parameter no caller populates is not a seam.
        (quiet if state not in FINDING_STATES_THAT_REDDEN
         else loud).append(line)
    reddens = alarmed + alarmed_status + loud
    for line in quiet:
        # IT PRINTS, WITH ITS FIGURE AND ITS REVIEW DATE. A known state is
        # reported; it just stops spending the run's colour.
        print(f"note: KNOWN, does not redden: {line}")
    if not reddens:
        print("no NEW, DETERIORATED or UNACKNOWLEDGED finding: exit 0. "
              "Machinery failure is handled separately and always reddens.")
        return 0
    print(chr(10) + "::error::audit alarm — " + "; ".join(reddens))
    _status_n = len(alarmed_status)
    if _status_n > 0:
        # SAID OUT LOUD, every time, so the absence of a Telegram message is
        # a recorded decision rather than something to rediscover.
        print(f"note: {_status_n} of {len(reddens)} alarm(s) above "
              f"RESTATE a run's own conclusion, so they are NOT sent to "
              f"Telegram - GitHub's notification on that red run carries "
              f"them, and this run's red carries the rest.")
    _write_alarm_body(alarmed + loud)
    return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--unaudited", action="store_true",
                    help="only runs with no ledger row")
    ap.add_argument("--since")
    ap.add_argument("--until")
    ap.add_argument("--run", help="one run id")
    ap.add_argument("--limit", type=int, default=250)
    # REPORTING AND FAILING ARE DIFFERENT POLICIES, so the flag is explicit and
    # the default is REPORT. DEGRADED is this pipeline's ordinary state — eight
    # implausible-attribution rows every day — and failing on it would make red
    # mean nothing, which is precisely the noise DEL-2 was narrowed to avoid.
    # BROKEN and DID_NOT_RUN are different: they say the pipeline did not do its
    # job, which is the same axis DEL-2 chose.
    ap.add_argument("--fail-on", default="",
                    help="comma-separated verdicts that exit non-zero "
                         "(e.g. BROKEN,DID_NOT_RUN). Default: report only.")
    a = ap.parse_args()

    if a.run:
        # Fetch the run's real status: --run used to synthesise a row with no
        # status at all, so an in-flight run audited this way could never
        # report IN_PROGRESS. The single-run path is the one used to chase an
        # anomaly, which is exactly when the distinction matters most.
        _st = ""
        try:
            _st = json.loads(_sh("gh", "run", "view", a.run, "--json",
                                 "status") or "{}").get("status", "")
        except Exception:
            pass
        runs = [{"databaseId": int(a.run), "workflow": "?", "status": _st,
                 "startedAt": "", "conclusion": "?"}]
        listing_warning = None
    else:
        listing = list_runs(a.since, a.until, a.limit)
        # CONSULTED, not merely available. The defect this exists for was never
        # that the truncation was unknowable — it was that nothing asked.
        listing_warning = listing.warning()
        runs = list(listing)
        if a.unaudited:
            done = audited_run_ids()
            runs = [r for r in runs if str(r["databaseId"]) not in done]

    if listing_warning:
        print(f"note: {listing_warning}\n")

    fail_on = {v.strip().upper() for v in (a.fail_on or "").split(",") if v.strip()}
    alarmed: List[str] = []
    # ── TELEGRAM CARRIES CONTENT AND ACTIONS, NEVER RUN STATUS. 2026-10-22 ──
    #
    # TWO LISTS, NOT ONE, and the split is made where the alarm is RAISED
    # rather than by matching its wording later -- pattern-matching a message
    # to recover its class is how two alarms spent nine days registered and
    # incapable of firing.
    #
    # `alarmed_status` holds alarms that only RESTATE a run's own conclusion.
    # They still print, still set the exit code and still redden this
    # workflow, so GitHub's notification carries them. They are not pushed to
    # a phone, because a phone cannot act on them.
    alarmed_status: List[str] = []
    # ── KEYED FINDINGS. 2026-10-23. ────────────────────────────────────────
    #
    # `(finding_key, figure, text)` for findings that carry their identity and
    # their figure from the site that RAISES them. `_emit_alarms` classifies
    # each against `KNOWN_FINDINGS`, and only NEW / DETERIORATED /
    # UNACKNOWLEDGED earn a red. Everything in `alarmed` is treated as NEW and
    # reddens, which is correct for all of them today.
    _keyed: List[tuple] = []

    # ── THE PATTERN SELF-TEST, AT REGISTRATION AND BEFORE ANY RUN ─────────
    #
    # Placed here, before the early return on "no runs to audit", for the same
    # reason the workflow-state check below is: a registration defect is not
    # about any particular run, and putting it after the return would make it
    # unreachable exactly when the audit has nothing else to do.
    #
    # It is a FINDING, not machinery: a pattern that cannot count is a
    # detector that cannot fire, which is what the model-integrity alarms were
    # for ten days.
    _pst = selftest_patterns()
    print()
    print(format_pattern_selftest(_pst))
    # THE REGIME CLOCK, every run, so it is visible rather than remembered.
    print()
    print(format_picks_bound_status(parallel_regime_picks_series()))
    if _pst["failed"]:
        alarmed.append(
            f"{len(_pst['failed'])} registered pattern(s) FAIL the self-test: "
            + ", ".join(k for k, _ in _pst["failed"])
            + " - each matches its own example and still yields no value, so "
              "the detector cannot fire. CLEARING ACTION: give the pattern a "
              "numeric capture group, or add the key to the count-style list "
              "in extract().")

    # WORKFLOW STATE, CHECKED BEFORE THE EARLY RETURN.
    #
    # IT WAS FIRST PLACED AFTER IT, WHICH MADE IT UNREACHABLE IN THE ONE
    # CASE IT EXISTS FOR: a disabled workflow produces NO RUNS, so
    # `if not runs: return 0` printed "No runs to audit." and exited 0
    # before the check ran. Found by simulating a disabled workflow rather
    # than by reading the code — the same ordering family as PNC-1. Not self-calibrating and
    # not behind `--fail-on`: a scheduled workflow that is not `active` is wrong
    # on the FIRST occurrence, the way a spent credit returning no rows is. It
    # also cannot be inferred from runs, which is the whole reason it is here.
    _states = workflow_states()
    if _states is None:
        # ── MADE TO ALARM 2026-10-22. ─────────────────────────────────────
        #
        # It was a `note:` that said "UNKNOWN" and alarmed on nothing. Of the
        # three lines in that shape this is the one with NO ALTERNATIVE
        # CARRIER: the comment above says it, and it is true -- a disabled
        # workflow produces no runs, so nothing else in this audit can detect
        # one. When this check cannot look, that detector is blind and no
        # other line goes quiet to tell you.
        #
        # AND IT IS CLEARED BY AN ACTION, which is what separates it from the
        # day-type line above: restoring `gh` auth or the token clears it, so
        # it does not teach waiting. MEASURED: zero occurrences to date, so
        # alarming costs nothing in volume.
        print("!! WORKFLOW STATE UNEVALUATED: `gh workflow list` could not be "
              "read, so this run did NOT check whether any workflow is "
              "disabled. A disabled workflow produces no runs and cannot be "
              "detected any other way, so this is a BLIND DETECTOR, not a "
              "quiet one. CLEARING ACTION: restore `gh` auth (GH_TOKEN in "
              "the job env) and re-run.\n")
        alarmed.append(
            "workflow state UNEVALUATED - `gh workflow list` unreadable, so "
            "the ONLY detector for a disabled workflow did not run. CLEARING "
            "ACTION: restore GH_TOKEN in the audit job and re-run")
    else:
        # SCOPED TO `WORKFLOWS`, added 2026-10-13. `gh workflow list` returns
        # every record GitHub HOLDS, not every file this repository has: a
        # scratch workflow pushed to a deleted branch stayed listed as
        # `active` for at least a day after the branch was gone (measured
        # 2026-10-13, `zz-parse-bad.yml`). An unscoped check would alarm on a
        # workflow that no longer exists the moment GitHub marks it inactive,
        # which is an alarm nobody can clear by any action in this repo.
        # `gh workflow list` keys a VALID workflow by its `name:` field and an
        # orphaned one by its PATH, so matching on filename alone excludes
        # every real workflow - caught by driving it rather than by reading
        # it. The owned set is therefore built FROM THE TREE: each file in
        # `WORKFLOWS` contributes both its display name and its filename.
        _owned = set(WORKFLOWS)
        for _w in WORKFLOWS:
            try:
                for _line in (pathlib.Path(".github/workflows")
                              / _w).read_text(encoding="utf-8").splitlines():
                    if _line.startswith("name:"):
                        _owned.add(_line.split(":", 1)[1].strip())
                        break
            except OSError:
                continue
        _unowned = {n for n in _states
                    if n not in _owned and n.rsplit("/", 1)[-1] not in _owned}
        if _unowned:
            print(f"note: {len(_unowned)} workflow record(s) GitHub holds that "
                  f"this repository does not own, so their state is NOT "
                  f"checked: {', '.join(sorted(_unowned))}. A record outlives "
                  f"the branch that carried it; an alarm on one could not be "
                  f"cleared by any action here.")
        _off = {n: st for n, st in _states.items()
                if st.lower() != "active" and n in _owned}
        for _name, _state in sorted(_off.items()):
            alarmed.append(f"workflow {_name!r} state={_state}")
            print(f"::error::workflow {_name!r} is {_state}, not active — it "
                  f"produces no runs, so every run-based check below is blind "
                  f"to it. Re-enable with `gh workflow enable`.")
        if _off:
            print()

    if not runs:
        # "No runs" is the reading most changed by a truncated query, so the
        # warning is repeated rather than assumed to have been read above.
        print("No runs to audit."
              + ("  (SEE THE TRUNCATION WARNING — this may be a cut-off "
                 "query, not an empty window.)" if listing_warning else ""))
        # A state alarm must survive the early return: a disabled workflow
        # produces no runs, so this is the branch it arrives on.
        return _emit_alarms(alarmed, alarmed_status, _keyed)

    # HISTORY IS WHAT IS KNOWN, NOT WHAT IS IN THIS PASS. Seeded from the
    # ledger before the loop, so recording a finding no longer erases the
    # evidence that would fire it again. See `ledger_history`.

    by_wf: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for _wf in {r["workflow"] for r in runs}:
        seed = ledger_history(_wf, LOOKBACK_RUNS)
        if seed:
            by_wf[_wf] = list(seed)
        else:
            # NOT a pass. The per-source checks below will be skipped for this
            # workflow, and silence there would read as coverage.
            print(f"note: no ledger history for {_wf}: per-source discovery checks "
                  f"are UNEVALUATED for it, not passed\n")

    print(f"{'run':<12} {'workflow':<14} {'started':<17} {'verdict':<10} findings")
    print("-" * 100)
    seen_days: set = set()
    for r in runs:
        rid = str(r["databaseId"])
        log = fetch_log(rid)
        facts = extract(log)
        # Run metadata, not log-derived: the only evidence that
        # separates "still running" from "log unavailable".
        facts["run_status"] = r.get("status")
        # Added 2026-10-21 for `classify_red`: a cancelled run never reached
        # the point where it would have marked itself, which is
        # UNCLASSIFIABLE rather than FAILURE.
        facts["run_conclusion"] = r.get("conclusion")
        # Only the DAY'S FIRST run of a workflow exercises discovery from cold.
        # A same-day re-run legitimately finds no NEW fixtures, because the
        # first run already added them — so applying the per-source check to
        # every run fires on 2026-03-03, a day discovery was working fine.
        _day = (r["workflow"], (r.get("startedAt") or "")[:10])
        facts["is_first_run_of_day"] = _day not in seen_days
        seen_days.add(_day)
        hits = assertions(facts, by_wf[r["workflow"]])
        by_wf[r["workflow"]].append(facts)
        # The empty card is MEASURED, not assumed, and the suppression expires
        # on 2026-10-09 by date. Suppressed findings are printed below.
        try:
            _rd = _dt.date.fromisoformat((r.get("startedAt") or "")[:10])
        except ValueError:
            _rd = None
        hits, _suppressed, _census = partition_empty_card(
            hits, _rd, zero_leagues=facts.get("zero_fixture_leagues"))
        v = verdict(facts, hits, log)
        # STAGE 19 item 2: per-source discovery figures are printed on EVERY
        # daily-picks row, verdict or not, and belong in the ledger note.
        # The AGGREGATE is the number that lied for three months: a healthy
        # total hid a dead source for 88 days. A reader must not have to
        # reconstruct which source produced what.
        disc = discovery_summary(facts)
        res = resolution_summary(facts)
        _pre = "  ".join(x for x in (disc, res) if x)
        # The audit reports on itself and never alarms on itself — see
        # NEVER_ALARM_WORKFLOWS for why a one-run lag would not break the loop.
        # CLR-2. A resolved run leaves the ALARM's scope because a row says so,
        # and stays in the TABLE because it happened. Recording the resolution
        # is the clearing action and it changes this predicate.
        # ── A RED MEANS "FAILURE" ONLY FOR WORKFLOWS WHOSE REDS ARE
        # FAILURES. 2026-10-21, rule 1: the declaration is READ, never
        # assumed. `reminders.yml` is designed to redden when a guard reaches
        # its date; a crashed reminders run carries no marker and still
        # alarms. Silencing the workflow is NOT available -- that is what hid
        # eight ci-audit reds and the probe's BROKEN.
        _red_state, _red_why = (None, None)
        if str(r.get("conclusion") or "").lower() == "failure" \
                and r["workflow"] in DELIBERATE_RED_MARKERS:
            _red_state, _red_why = classify_red(r["workflow"], log, facts)
            print(f"  red classified: {_red_state} - {_red_why}")
        if _red_state == "UNCLASSIFIABLE" and UNCLASSIFIABLE_ALARMS:
            # STATUS: it is a statement ABOUT A RED CONCLUSION, which GitHub
            # has already notified on. What this adds is the classification,
            # and that is a thing to read in the log, not to be woken by.
            alarmed_status.append(
                f"{rid} {r['workflow']} red UNCLASSIFIABLE - {_red_why}")
        # Read IMMEDIATELY ABOVE the predicate it gates. It used to sit at the
        # top of this block, and the red-classification inserted between them
        # put 582 characters of code in the gap -- which broke the CLR-2
        # guard's window and, more to the point, separated a value from its
        # only use.
        _resolved = RESOLVED_RUNS.get(str(rid))
        if v in fail_on and r["workflow"] not in NEVER_ALARM_WORKFLOWS \
                and not _resolved and _red_state != "VERDICT":
            # ── STATUS OR FINDING, DECIDED BY WHETHER GITHUB ALREADY SAID IT ─
            #
            # `DID_NOT_RUN`, and `BROKEN` on a run that CONCLUDED FAILURE,
            # restate the run's own red: GitHub's notification carries them
            # and a second copy on a phone adds nothing actionable.
            #
            # `BROKEN` on a run that CONCLUDED SUCCESS is the opposite case
            # and it is why this is not a blanket removal. MEASURED today:
            # run 39710051629, `daily-picks`, scheduled, conclusion SUCCESS,
            # scored BROKEN on a traceback in its log. GitHub notified NOBODY
            # about that run. Drop it from Telegram too and the only carrier
            # left would be a line in an audit log.
            _restates_the_red = (
                v == "DID_NOT_RUN"
                or str(r.get("conclusion") or "").lower() == "failure")
            # ── NOT KEYED, AND THAT IS A DECISION. 2026-10-23. ────────────
            #
            # A per-run verdict already has an owner: `RESOLVED_RUNS`. A
            # keyed version was built and REMOVED rather than shipped, because
            # it would have been a SECOND mechanism suppressing the same
            # alarm, and a run suppressed for two different reasons has no
            # single place to look.
            #
            # AND THE EXISTING ONE DEMONSTRABLY WORKS. The 30-run sample of
            # 2026-10-22 showed `daily-picks DID_NOT_RUN` in 30 of 30 firings
            # — five real events relayed 12, 10, 5, 2 and 1 times — and all
            # five are in `RESOLVED_RUNS`, recorded 2026-10-18 by `7514a50`
            # and `465146a`. Re-audited today over 2026-10-13 onward, 156
            # rows including those five, they produce ZERO alarms. The noise
            # measured in that sample predates the resolution, not the rule.
            #
            #     The mechanism was not missing. It was not filled in.
            (alarmed_status if _restates_the_red else alarmed).append(
                f"{rid} {r['workflow']} {v}"
                + ("" if _restates_the_red else
                   " - ON A RUN THAT CONCLUDED SUCCESS, so no GitHub "
                   "notification was sent for it"))

        # ── THE FOURTH ALARM SITE, added 2026-10-12 and written down ────────
        #
        # Model integrity is OUTSIDE the verdict axis on purpose. A tampered
        # or corrupt pickle does not make the run BROKEN - the agent catches
        # it, declines to load, and produces picks from Poisson + Elo - so the
        # run can score CLEAN while the artefact is unusable. DEGRADED is not
        # in `--fail-on` either, so routing this through `hits` would print it
        # and alert nobody.
        #
        # It is safe to alarm because it has DYNAMIC RANGE, measured BEFORE
        # wiring: 0 integrity failures and 0 unsigned loads across 1,400 load
        # events in 239 runs of the cached corpus (measured 2026-10-12). An
        # alarm whose baseline equals its threshold is the
        # `fixtures_zero_active` shape; this one's baseline is zero.
        #
        # CLR-2: the clearing action is in the alert text and it is an action
        # a person takes. The emitter carries `action=` for both markers too.
        if facts.get("model_integrity_failures"):
            alarmed.append(
                f"{rid} {r['workflow']} MODEL INTEGRITY FAILURE x"
                f"{facts['model_integrity_failures']} - a signed artefact did "
                f"not match its signature. CLEARING ACTION: investigate the "
                f"cached pair, then re-train to re-sign. Do NOT delete it.")
        elif facts.get("model_unsigned_loads"):
            alarmed.append(
                f"{rid} {r['workflow']} MODEL SIGNATURE MISSING x"
                f"{facts['model_unsigned_loads']} - a pickle was loaded "
                f"UNVERIFIED because no .sig sat beside it. CLEARING ACTION: "
                f"run `--train` so the pair is re-saved with a signature.")
        print(f"{rid:<12} {r['workflow']:<14} {(r.get('startedAt') or '')[:16]:<17} "
              f"{v:<10} {((_pre + '  ') if _pre else '') + '; '.join(hits)}"[:190])
        for h in hits[1:]:
            print(f"{'':<56} {h[:60]}")
        # RECORDED, not hidden. Suppressed is not unobserved.
        for h in _suppressed:
            print(f"{'':<56} ~ {h[:150]}")
        # RECORDED, not silencing. The verdict above is unchanged; only the
        # alarm's scope moved, and the row says why and by which commit.
        if _resolved:
            print(f"{'':<56} = RESOLVED by {_resolved[0]}: {_resolved[1][:120]}")
        # VAC-1. UNCONDITIONAL: printed whether or not anything was suppressed,
        # so that "the filter cleared nothing" and "the filter was handed
        # nothing" are different lines rather than the same absence.
        print(f"{'':<56} {_census}")
    # STAGE 23. The schedule margin, EVERY day, alarm or not. `2026-11-06` is a
    # fit with R^2=0.451 and must never stand alone as a date; this is the
    # measured number beside it.
    _by_type, _crons = collect_lag_series_by_daytype()
    # TOMORROW, NAMED HERE AND PASSED. The deadline is a LOCAL fact converted
    # per day (localised 2026-10-22), and `_tomorrow` further down is the same
    # day -- it is hoisted so the margin and the alarm cannot disagree about
    # which day they are talking about across a DST transition.
    _tomorrow = _dt.date.today() + _dt.timedelta(days=1)
    _margins = schedule_margins_by_daytype(_by_type, _crons, day=_tomorrow)
    print()
    # PNC-1, FOUND 2026-10-11. `format_series_census` carried the attribution
    # rule and the per-series n, and NOTHING CALLED IT — it was exercised only
    # by its own test, so the rule it states had never been printed in CI.
    # PNC-1 again. The ordinal that stood here ("thirteenth") was an
    # accreted count; the tally audit of 2026-10-12 enumerated the
    # register and found 11 verified of 14 claimed - see
    # docs/rule-tallies.md. Cite the register, not a running number.
    print(format_series_census(_by_type, _crons))
    print()
    print(format_schedule_margins(_margins))

    # ── A RUN THAT NEVER FIRED, AND A FILE GITHUB REFUSED ───────────────────
    # Both printed EVERY run. Neither had any detector before 2026-10-13, and
    # `DID_NOT_RUN` is not one: it verdicts `steps_not_run` on a run that
    # exists. A missing run has no row to verdict.
    print()
    _missing = missing_scheduled_runs()
    print(format_missing_runs(_missing))
    if _missing.get("missing"):
        alarmed.append(
            f"{len(_missing['missing'])} scheduled daily-picks run(s) NEVER "
            f"EXISTED: {', '.join(str(d) for d in _missing['missing'])} - not "
            f"late, ABSENT. CLEARING ACTION: confirm the workflow file parses "
            f"and the cron matches the day type.")

    print()
    _unparseable = unparseable_workflow_runs()
    print(format_unparseable(_unparseable))

    # ── THE DIAGNOSED-AND-UNACTIONED COUNT, REPORTED DAILY. 2026-10-20. ───
    #
    # The weather producer was diagnosed on 2026-08-07 and deleted on
    # 2026-10-20: SEVENTY-FOUR DAYS. Nothing counted it in between, so
    # nothing could notice. `scripts/finding_register.py` references the
    # source doc and line and holds the status itself -- the ledger's own
    # entries are NEVER rewritten, a rule that has held four times.
    #
    # REPORT-ONLY, like `pnc_scan`: a triaged backlog is a decision, not a
    # finding, so the count does not alarm. What DOES alarm is an UNTRIAGED
    # live finding, because an unmarked entry is how 74 days happened.
    #
    # CLR-2: the clearing action is a FIX, a DELETION, or a DECLARATION with
    # a re-triage date, and each changes the predicate -- a fix or deletion
    # flips `is_live` to False, a declaration flips `untriaged` to empty.
    try:
        from scripts.finding_register import evaluate as _fr_eval
        from scripts.finding_register import untriaged as _fr_untriaged
        _fr = _fr_eval()
        _fr_live = [r for r in _fr if r["is_live"] is True]
        _fr_unev = [r for r in _fr if r["is_live"] is None]
        _fr_bad = _fr_untriaged(_fr)
        print(f"{chr(10)}diagnosed_findings[{len(_fr)} harvested, "
              f"{len(_fr_live)} still live, {len(_fr_unev)} UNEVALUATED, "
              f"{len(_fr_bad)} untriaged | median age "
              f"{max((r['age_days'] for r in _fr_live), default=0)}d | "
              f"source: docs/predictive-audit-2026-08-07.md]")
        for _r in sorted(_fr_live, key=lambda x: x["id"]):
            print(f"  {_r['outcome']:8s} [{_r['id']:2d}] {_r['file']} :: "
                  f"{_r['location'][:46]}"
                  + (f"  re-triage {_r['retriage']}" if _r.get("retriage")
                     else ""))
        if _fr_bad:
            alarmed.append(
                f"{len(_fr_bad)} diagnosed finding(s) UNTRIAGED: "
                + ", ".join(str(r["id"]) for r in _fr_bad)
                + " - a fix, a deletion, or a declaration with a date")
    except Exception as _fr_err:            # noqa: BLE001
        print(f"{chr(10)}diagnosed_findings[UNEVALUATED: "
              f"{type(_fr_err).__name__} - could not read the register, which "
              f"is NOT an empty backlog]")
    _unp_main = [r for r in _unparseable if r.get("headBranch") == "main"]
    if _unp_main:
        alarmed.append(
            f"{len(_unp_main)} unparseable workflow push(es) on main: "
            + ", ".join(str(r["workflowName"]) for r in _unp_main)
            + " - GitHub refused the file, so that workflow is NOT scheduled.")
    _margin = _margins.get(_margins.get("worse")) or {}
    _cron = _crons.get(_margins.get("worse"))

    # ── THE PROSPECTIVE MARGIN COMES BACK, FROM THE REFERENCE — 2026-10-19 ─
    #
    # IT LIVED IN THE PROBE AND THE PROBE IS OFF. A DB-dependent line was
    # added here on 2026-10-10 and could never have produced its number: this
    # step passes `GH_TOKEN` and nothing else and installs no packages, so
    # without `DATABASE_URL` it fell back to SQLite, which holds no fixtures,
    # and read `UNKNOWN: the card for that day is not discovered` on every
    # run. It was removed and handed to `fixture_horizon_probe.py`, which was
    # disabled on 2026-10-18 after 240 page-loads produced 8 refusals and
    # ZERO deadlines -- so from that day **nothing computed it at all**.
    #
    # The objection that removed it was `credentials AND psycopg2 AND
    # SQLAlchemy on every audit run`. The reference pays none of that: ONE
    # stdlib HTTPS GET and one secret, no database, no driver, no package.
    # That is the whole reason this line can live here now.
    #
    # COVERAGE IS NOT A BOUND. `measured` 2026-10-19 the reference covers 9 of
    # the 30 configured leagues, so its minimum errs LATE. It is combined with
    # the day-type constant by `min`, which can only tighten; the basis names
    # which side bound the answer and the coverage is printed every run.
    # `_tomorrow` is HOISTED above, beside the margin, so these two cannot
    # end up describing different days across a transition.
    _ref = reference_earliest(_tomorrow)
    _ahead = combined_deadline(_tomorrow, _ref)
    print()
    print(format_reference_deadline(_ahead))
    print(f"  basis: {_ahead['basis']}")
    if _ref.get("leagues_absent"):
        print(f"  reference covered : {', '.join(_ref['leagues_present']) or 'NONE'}")
        print(f"  reference did NOT : {', '.join(_ref['leagues_absent'])}")
    _ahead_cron = _crons.get(day_type(_tomorrow))

    # ── TWO MARGINS, AND THE ALARM MOVES TO THE ACTIONABLE ONE. 2026-10-21 ─
    #
    # One number was carrying two facts. `min(reference, floor)` always
    # returned the floor, because the floor IS the observed earliest kickoff,
    # so every day reported a structural statement about the pipeline's
    # duration dressed as a day-specific deadline. Measured 2026-10-21: floor
    # 09:30 against today's actual earliest kickoff of 15:30.
    #
    # CLR-2: the alarm fires on the quantity its clearing action changes. A
    # manual dispatch changes the DAY-SPECIFIC margin and does nothing to the
    # structural one, which is the duration problem.
    #
    # FIRING RATE, measured over the record (n=904 days, 2024-01-01..):
    #     structural  629/904 = 69.6%
    #     actionable    4/605 =  0.7%   (605 days the reference could see)
    # The alarm becoming rare is the point. The structural term stays printed
    # every run, so nothing is hidden by the change -- and it has to stay,
    # because the actionable margin's residual EXCEEDS the registered 60 on
    # 51.6% of days and so cannot carry the safety claim on its own.
    _m2 = two_margins(
        _tomorrow, _ref, _ahead_cron,
        [lag for _d, lag in (_by_type.get(day_type(_tomorrow)) or [])],
        MARGIN_RUN_DURATIONS_BY_DAYTYPE[day_type(_tomorrow)])
    print(format_two_margins(_m2))
    _act, _struct = _m2["actionable"], _m2["structural"]
    if _act.get("verdict") == "ALARM":
        print(f"!! ACTIONABLE MARGIN {_act['binding']:+.0f} min for "
              f"{_tomorrow} is below {_act['min_margin']} min. "
              f"CLEARING ACTION: {CLEARING_ACTION}")
        alarmed.append(
            f"actionable margin {_act['binding']:+.0f}m for {_tomorrow} "
            f"< {_act['min_margin']}m (deadline "
            f"{_act['deadline_min'] // 60:02d}:"
            f"{_act['deadline_min'] % 60:02d} from {_ahead['reference']}, "
            f"{_act['coverage']}, reference {_ahead['reference_state']}; "
            f"residual p50={_m2['residual']['p50']}m so this is an UPPER "
            f"BOUND)")
    # REPORTED AND ACKNOWLEDGED, NEVER ALARMED. It is true every day and no
    # action available to the operator clears it, which is the definition of
    # the thing CLR-2 says must not be an alarm.
    if _struct.get("verdict") == "ALARM":
        # ── ACKNOWLEDGED, BUT ONLY WHILE IT HOLDS. 2026-10-21. ─────────────
        #
        # This printed the acknowledgement unconditionally, so a structural
        # margin that DETERIORATED would have gone on reporting "does NOT
        # alarm" with a stale figure beside it. The predicate compares the
        # figure AND the review date -- the retired
        # `ACKNOWLEDGED_MARGIN_BREACH` compared only the figure and absorbed
        # a breach for days past its own review date.
        _deficit = _struct["binding"] - _struct["min_margin"]
        _sack = structural_breach_is_acknowledged(_deficit)
        if _sack:
            print(f"   structural margin {_struct['binding']:+.0f} min for "
                  f"{_tomorrow} is below {_struct['min_margin']} min "
                  f"(deficit {_deficit:+.1f}) and is ACKNOWLEDGED since "
                  f"{_sack['since']}, review {_sack['review']}. It does NOT "
                  f"alarm: {_sack['action']}")
        else:
            print(f"!! STRUCTURAL MARGIN {_struct['binding']:+.0f} min for "
                  f"{_tomorrow} (deficit {_deficit:+.1f}) is NO LONGER "
                  f"COVERED by the acknowledgement made "
                  f"{STRUCTURAL_ACK['since']} at "
                  f"{STRUCTURAL_ACK['binding']:+.1f} (review "
                  f"{STRUCTURAL_ACK['review']}): it has either DETERIORATED "
                  f"or passed its review date. CLEARING ACTION: "
                  f"{STRUCTURAL_ACK['action']}")
            alarmed.append(
                f"structural margin deficit {_deficit:+.1f}m for {_tomorrow} "
                f"is worse than the acknowledged "
                f"{STRUCTURAL_ACK['binding']:+.1f}m, or the acknowledgement "
                f"(review {STRUCTURAL_ACK['review']}) has expired")

    if _cron is None:
        # ── MADE TO ALARM 2026-10-22. ─────────────────────────────────────
        #
        # It was a `note:` ending "UNEVALUATED, not safe" and alarming on
        # nothing. Read what the condition actually is: the day type the
        # trigger fires on -- TOMORROW's -- has NO PARSEABLE CRON in
        # `daily-picks.yml`. That is not an unknown margin, it is a workflow
        # that will not run tomorrow, and the margin being uncomputable is
        # the least of it.
        #
        # CLEARED BY AN EDIT: add or repair the cron. So, unlike the day-type
        # line in `format_series_census`, an alarm here names an action
        # someone performs rather than a wait.
        print(f"!! NO CRON FOR TOMORROW'S DAY TYPE "
              f"({_margins.get('worse') or 'unknown'}): `daily-picks.yml` "
              f"carries no parseable `m h * * *` entry for it, so the "
              f"workflow is NOT SCHEDULED to run and the margin cannot be "
              f"computed. CLEARING ACTION: add or repair that cron in "
              f"`.github/workflows/daily-picks.yml`.")
        alarmed.append(
            f"no parseable cron for tomorrow's day type "
            f"({_margins.get('worse') or 'unknown'}) in daily-picks.yml - the "
            f"workflow is NOT SCHEDULED for it and the margin is uncomputable. "
            f"CLEARING ACTION: add or repair the cron entry")
    elif _margin.get("alarm"):
        # ── THE MARK FOLLOWS THE ALARM, NOT THE CONDITION. 2026-10-21. ───
        #
        # This printed `!!` BEFORE consulting the acknowledgement, so an
        # ACKNOWLEDGED breach -- one that deliberately does not alarm -- still
        # wore the finding marker. Found by demonstrating the quiet state:
        # the acknowledgement was injected, the margin stopped alarming, and
        # the mark stayed. The rule has to apply to its own first instance.
        _ack = margin_breach_is_acknowledged(
            _margins.get("worse") or "", _margin.get("binding") or 0.0)
        if not _ack:
            print(f"!! SCHEDULE MARGIN {_margin['binding']:+.0f} min is below "
                  f"the {MARGIN_ALARM_MIN}-minute threshold. Remedy on file: "
                  f"{MARGIN_REMEDY}")
        else:
            print(f"note: schedule margin {_margin['binding']:+.0f} min is "
                  f"below the {MARGIN_ALARM_MIN}-minute threshold and is "
                  f"ACKNOWLEDGED, so it reports and does not alarm. Remedy on "
                  f"file: {MARGIN_REMEDY}")
        if _ack:
            # ── THE ACKNOWLEDGED PATH SAYS EVERYTHING THE ALARM SAID. ──────
            #
            # Three things it used not to. The `why` was truncated at 88
            # characters, which cut the conservatism statement off entirely;
            # the floor was printed as if it were the reading; and the
            # reading itself did not appear, so a reader could not tell how
            # much headroom the acknowledgement had left.
            #
            # An acknowledgement that reports LESS than the alarm it replaces
            # is a silencer.
            _pb_ack = parallel_regime_picks_series()
            print(f"  ACKNOWLEDGED since {_ack['since']}, review "
                  f"{_ack['review']}.")
            print(f"  reading now {_margin['binding']:+.2f}m; acknowledged "
                  f"DOWN TO a FLOOR of {_ack['binding']:+.2f}m "
                  f"(= the {_ack['measured_at']:+.2f}m measured on "
                  f"{_ack['since']} less {_ack['jitter']}m of measured "
                  f"single-point jitter). It alarms on DETERIORATION past "
                  f"that floor, not on the level.")
            print(f"  WHY IT CANNOT CLEAR, and why the reading is "
                  f"CONSERVATIVE: {_ack['why']}")
            print(f"  CLEARING ACTION: {_ack['action']}")
            print(f"  the replacement series is at n={_pb_ack['n']}/"
                  f"{MIN_N_TO_REDERIVE_PICKS_BOUND}"
                  + (f", about {_pb_ack['to_ten_date']}"
                     if _pb_ack["to_ten_date"] not in ("reached",
                                                       "UNEVALUATED")
                     else f" ({_pb_ack['to_ten_date']})"))
        else:
            # RULE 3: a margin computed from a RETIRED REGIME'S constant is
            # labelled wherever it appears, the alert included. The operator
            # dispatching by hand needs to know what he is insuring against.
            _pb = parallel_regime_picks_series()
            # ── KEYED, SO A KNOWN FIGURE PRINTS INSTEAD OF REDDENING. ─────
            #
            # This is the finding that reddened `ci-audit` every day for two
            # weeks. It now carries its key and its FIGURE from here, and
            # `_emit_alarms` classifies it NEW / DETERIORATED / UNACKNOWLEDGED
            # / UNCHANGED against `KNOWN_FINDINGS["margin.schedule"]`. Only
            # the first three earn a red; UNCHANGED prints with its figure and
            # its review date.
            _keyed.append(("margin.schedule", _margin["binding"],
                f"schedule margin {_margin['binding']:+.0f}m "
                f"< {MARGIN_ALARM_MIN}m - CONSERVATIVE: computed from "
                f"FULL_CARD_PICKS_MAX={FULL_CARD_PICKS_MAX}m, a full-card MAX "
                f"from the RETIRED SERIAL review regime (n=4, 2026-10-01). "
                f"The parallel review cut the picks step's sd 11.71 -> 2.37, "
                f"so the true margin is very probably BETTER than this. The "
                f"replacement series is at n={_pb['n']}/"
                f"{MIN_N_TO_REDERIVE_PICKS_BOUND}"
                + (f", ~{_pb['to_ten_weeks']} weeks to go (about "
                   f"{_pb['to_ten_date']})" if _pb["to_ten_date"] != "reached"
                   else ", READY TO RE-DERIVE")))

    if listing_warning:
        # Printed AGAIN under the table: the number a reader carries away is
        # the one at the bottom of a long listing, and that is the number the
        # truncation qualifies.
        print(f"\nnote: {listing_warning}")
        print(f"note: {len(runs)} run(s) listed above is a LOWER BOUND.")
    return _emit_alarms(alarmed, alarmed_status, _keyed)


#: Exit codes, so a red run says WHICH KIND it is.
#:
#: A red `ci-audit` is a VERDICT by design: it exits 1 when it finds something.
#: But an uncaught exception also exits 1, and on 2026-10-10 three consecutive
#: reds were a `ModuleNotFoundError` at import reported as "CI audit alarmed -
#: see the audit step for the run ids" when there were no run ids and nothing
#: had been audited at all. The workflow could not tell the two apart because
#: the process gave it the same number.
EXIT_CLEAN, EXIT_FINDING, EXIT_CRASH = 0, 1, 2


def _main_guarded() -> int:
    """`main()` with its own crash boundary, so machinery announces itself."""
    try:
        return main()
    except SystemExit:
        raise
    except BaseException as exc:                 # noqa: BLE001 - deliberate
        import traceback

        traceback.print_exc()
        print(f"\n::error::CI AUDIT CRASHED "
              f"This is MACHINERY, not a finding: NOTHING was audited by this "
              f"run, and nothing has been audited since the last run that "
              f"exited 0 or 1. Do not read the absence of findings as a "
              f"clean bill of health.")
        return EXIT_CRASH


if __name__ == "__main__":
    raise SystemExit(_main_guarded())
