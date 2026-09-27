"""The empty-card suppression, and the test that deletes it.

THE MEASUREMENT IT RESTS ON, 2026-09-27: `openfootball/football.json` 2026-27 —
a reference independent of Flashscore and API-Football — shows ZERO fixtures in
all eight covered leagues from 2026-09-21 through 2026-10-08. Every one of the
eight has its last fixture on 09-20 and its next on 10-09 or 10-10. The card
returns 10-09 with 7 fixtures and 41 on 10-10.

So "zero fixtures" is the EXPECTED state for eleven more days, and the discovery
alarms would otherwise fire daily with a known cause until they were tuned out —
which is how `fixtures_zero_active` became noise three times.

BUT SUPPRESSING THEM IS SUP-1 UNLESS IT EXPIRES. A guard that silences the alarms
about its own subject makes the first real collapse after 10-09 invisible. The
three requirements, each pinned below:

    1. the end date is a LITERAL, not a flag anyone must remember to unset
    2. `test_the_suppression_has_EXPIRED` FAILS from 10-09, so the suppression
       deletes itself rather than becoming permanent
    3. suppressed findings are still PRINTED, tagged with the grid cell that
       justifies each — suppressed is not unobserved

CLR-2: the action that clears the suppressed condition is THE CARD RETURNING,
which arrives by CALENDAR rather than by repair. **This is the first suppression
in this project cleared by time rather than by a fix** — and that is precisely
why it cannot be left to a judgement call.
"""

import datetime as dt
import importlib.util
import pathlib

import pytest

_spec = importlib.util.spec_from_file_location(
    "_ci_audit_empty", pathlib.Path("scripts/ci_audit.py"))
ci = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ci)

IN_WINDOW = dt.date(2026, 9, 27)
BEFORE = dt.date(2026, 9, 20)
AFTER = dt.date(2026, 10, 9)

ZERO_HITS = [
    "NO FIXTURES FOUND for the day — nothing was analysed",
    "30 fixture scrape(s) attempted, 0 fixtures found in total",
    "Flashscore fixtures: 0 created AND 0 matched while other sources produce",
]
REAL_HITS = [
    "2 traceback(s) in the log",
    "1 message(s) LOST — a report part the retry did not recover",
    "core step(s) DID NOT RUN: update, picks (incl. review)",
]


# ── REQUIREMENT 2, FIRST, BECAUSE IT IS THE ONE THAT MATTERS ─────────────────

def test_the_suppression_has_EXPIRED():
    """FAILS FROM 2026-10-09. That is the point, not a bug.

    When this goes red the card has returned and the suppression must be
    DELETED — this test, `EMPTY_CARD_FROM`, `EMPTY_CARD_UNTIL`,
    `EMPTY_CARD_SUPPRESSIBLE`, `empty_card_window_active`,
    `partition_empty_card`, and its call site in `main()`.

    Do not extend the date to make this pass. Extending it is the decision the
    expiry exists to force someone to make deliberately, and it needs a fresh
    measurement from the reference, not a nudge.
    """
    today = dt.date.today()
    assert today < ci.EMPTY_CARD_UNTIL, (
        f"the empty-card suppression expired on {ci.EMPTY_CARD_UNTIL} and today "
        f"is {today}. The card has returned; DELETE the suppression rather than "
        f"moving the date. If the break was genuinely extended, re-measure it "
        f"against openfootball and record the new grid before changing anything.")


def test_the_end_date_is_a_LITERAL_not_a_config_flag():
    """A flag someone must unset is the failure mode, not the fix."""
    src = pathlib.Path("scripts/ci_audit.py").read_text(encoding="utf-8")
    assert "EMPTY_CARD_UNTIL = _dt.date(2026, 10, 9)" in src
    assert "config.get" not in src.split("EMPTY_CARD_FROM")[1][:400], (
        "the window reads from config — then it is a switch, and a switch left "
        "on is how SUP-1 happens")


def test_the_window_closes_by_DATE_and_not_by_anything_else():
    assert ci.empty_card_window_active(dt.date(2026, 10, 8)) is True
    assert ci.empty_card_window_active(dt.date(2026, 10, 9)) is False
    assert ci.empty_card_window_active(dt.date(2027, 1, 1)) is False


# ── REQUIREMENT 3: suppressed is not unobserved ──────────────────────────────

def test_suppressed_findings_are_RETURNED_not_dropped():
    kept, supp = ci.partition_empty_card(ZERO_HITS, IN_WINDOW, today=IN_WINDOW)
    assert kept == []
    assert len(supp) == len(ZERO_HITS), "a finding vanished instead of being tagged"


def test_each_suppressed_finding_CARRIES_ITS_JUSTIFICATION():
    _, supp = ci.partition_empty_card(ZERO_HITS, IN_WINDOW, today=IN_WINDOW)
    for s in supp:
        assert "[SUPPRESSED:" in s
        assert "openfootball 2026-27" in s, "no reference named"
        assert "2026-09-27" in s, "the run's own date is not in the justification"
        assert "card returns 2026-10-09" in s, "no expiry in the record"


def test_the_original_finding_text_SURVIVES_inside_the_tag():
    _, supp = ci.partition_empty_card(
        ["NO FIXTURES FOUND for the day — nothing was analysed"],
        IN_WINDOW, today=IN_WINDOW)
    assert supp[0].startswith("NO FIXTURES FOUND for the day"), (
        "the finding was rewritten rather than annotated — a reader cannot "
        "grep the ledger for the original text")


# ── THE SUPPRESSION IS NARROW, WHICH IS THE WHOLE SAFETY ARGUMENT ────────────

def test_a_REAL_failure_inside_the_window_is_NOT_suppressed():
    """An empty card explains zero fixtures. It explains nothing else.

    A traceback, a lost message or a skipped step is not excused by the
    calendar, and suppressing one would be the SUP-1 this design exists to
    avoid.
    """
    kept, supp = ci.partition_empty_card(REAL_HITS, IN_WINDOW, today=IN_WINDOW)
    assert kept == REAL_HITS, f"a real failure was suppressed: {supp}"
    assert supp == []


def test_a_mixed_run_keeps_the_real_and_tags_the_expected():
    kept, supp = ci.partition_empty_card(
        ZERO_HITS + REAL_HITS, IN_WINDOW, today=IN_WINDOW)
    assert kept == REAL_HITS
    assert len(supp) == len(ZERO_HITS)


# ── TWO CONDITIONS, BOTH REQUIRED ────────────────────────────────────────────

def test_a_run_BEFORE_the_window_is_not_suppressed():
    """09-20 was the last normal day, with 31 fixtures. Nothing to excuse."""
    kept, supp = ci.partition_empty_card(ZERO_HITS, BEFORE, today=IN_WINDOW)
    assert kept == ZERO_HITS and supp == []


def test_a_run_AFTER_the_window_is_not_suppressed():
    kept, supp = ci.partition_empty_card(ZERO_HITS, AFTER, today=IN_WINDOW)
    assert kept == ZERO_HITS and supp == []


def test_an_in_window_run_audited_AFTER_the_expiry_is_not_re_alarmed():
    """The second condition. Re-auditing 09-27 in November must not resurrect
    eleven days of alarms that were correct to suppress at the time — nor
    suppress them silently, which is why `empty_card_window_active` gates on
    TODAY and the window gates on the RUN's date. Both are required."""
    kept, supp = ci.partition_empty_card(
        ZERO_HITS, IN_WINDOW, today=dt.date(2026, 11, 1))
    assert kept == ZERO_HITS, "the expired suppression still applied"
    assert supp == []


def test_a_run_with_no_parseable_date_is_NOT_suppressed():
    """Fail closed. An unknown date cannot license a suppression."""
    kept, supp = ci.partition_empty_card(ZERO_HITS, None, today=IN_WINDOW)
    assert kept == ZERO_HITS and supp == []
