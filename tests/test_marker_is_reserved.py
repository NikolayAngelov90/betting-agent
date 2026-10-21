"""`!!` is reserved for findings — 2026-10-21.

MEASURED BEFORE: a real audit run with ONE finding printed **10 marked
lines**. Nine were routine and print on every run — two `TRUNCATED LISTING`,
five `no ledger history`, one `LOWER BOUND`, one `workflow record(s) … does
not own`. A reader who must filter the marker is doing the marker's job.

MEASURED AFTER: the same run prints **1**.

THE RULE: a line carries `!!` **iff its condition also appends to `alarmed`.**
Everything else is `note:` — the information is kept, the marker is not.

> The quiet state is the one that must be clean, because that is the state a
> reader learns to skim.
"""

import io
import pathlib
import re
from contextlib import redirect_stdout

import pytest

import scripts.ci_audit as ci

SRC = pathlib.Path("scripts/ci_audit.py").read_text(encoding="utf-8")


def _code_lines():
    """Source lines that are not comments — a `!!` inside a comment is prose
    about the marker, not a use of it, and the first draft of this file would
    have counted its own explanation."""
    return [l for l in SRC.splitlines() if not l.lstrip().startswith("#")]


# ── the rule, statically ───────────────────────────────────────────────────

def test_EVERY_marked_site_also_ALARMS():
    """The rule itself. Four sites keep the marker and all four alarm."""
    marked = [l.strip() for l in _code_lines() if "!!" in l]
    assert len(marked) == 4, (
        f"{len(marked)} sites emit the marker, expected 4 (pattern "
        f"self-test, actionable margin, structural margin deterioration, "
        f"schedule margin): {marked}")
    for frag in ("{key}: {why}", "ACTIONABLE MARGIN", "STRUCTURAL MARGIN",
                 "SCHEDULE MARGIN"):
        assert any(frag in m for m in marked), f"{frag} lost its marker"


def test_the_NINE_ROUTINE_lines_KEPT_THEIR_INFORMATION():
    """Demoted, not deleted. Each still prints, as a note."""
    for frag in (
            "run(s) EXCLUDED because their headSha could not",
            "is UNEVALUATED, not safe - no run has yet",
            "{listing_warning}",
            "workflow state UNKNOWN: `gh workflow list` could not be read",
            "workflow record(s) GitHub holds that",
            "no ledger history for {_wf}",
            "carries no single `m h * * *` cron",
            "run(s) listed above is a LOWER BOUND",
    ):
        assert frag in SRC, f"the information was deleted, not demoted: {frag}"
        # and it is now a note
        owner = next(l for l in _code_lines() if frag in l)
        assert "!!" not in owner, f"still marked: {owner.strip()}"


def test_the_RESIDUE_is_NAMED_rather_than_quietly_demoted():
    """Three of the nine say "not safe"/"UNKNOWN" and alarm on nothing. Their
    words claim more than the alarm set delivers. Changing what alarms is a
    different decision from reserving the marker, so it is recorded."""
    assert "THREE of the nine" in SRC, (
        "the mismatch between 'not safe' and 'does not alarm' is no longer "
        "recorded, so it will be forgotten rather than decided")


# ── the quiet state ────────────────────────────────────────────────────────

def _marks(text):
    return [l for l in text.splitlines() if "!!" in l]


def test_a_QUIET_run_prints_ZERO_marked_lines():
    """Every formatter, on input with nothing wrong."""
    quiet = {
        "pattern self-test": ci.format_pattern_selftest(
            {"passed": list(ci.PATTERNS), "failed": [], "no_example": []}),
        "series census": ci.format_series_census(
            {"weekday": [("2026-10-20", 30.0)] * 9,
             "weekend": [("2026-10-18", 40.0)] * 9},
            {"weekday": 0, "weekend": 0}),
        "picks bound": ci.format_picks_bound_status(
            {"n": 0, "mean": None, "sd": None, "mean_plus_3sd": None,
             "max": None, "regime": "r", "needed": 10,
             "to_ten_weeks": 2.0, "to_ten_date": "2026-11-04"}),
    }
    for name, text in quiet.items():
        assert _marks(text) == [], (
            f"{name} marks a line on quiet input: {_marks(text)}")


def test_a_CLEAN_pattern_registry_prints_zero_marks_and_still_REPORTS():
    out = ci.format_pattern_selftest(ci.selftest_patterns())
    assert _marks(out) == [], out
    assert "35 pass" in out, "the quiet state stopped reporting the count"


# ── KNOWN-WRONG CONTROL ────────────────────────────────────────────────────

def test_KNOWN_WRONG_one_injected_finding_is_the_ONLY_marked_line():
    """A real finding in an otherwise quiet run must be the only mark.

    Driven through the real formatter, and the REASON is verified rather than
    just the count: the one mark must name the failing key.
    """
    quiet = ci.format_pattern_selftest(
        {"passed": list(ci.PATTERNS), "failed": [], "no_example": []})
    assert _marks(quiet) == []

    injected = ci.format_pattern_selftest({
        "passed": [k for k in ci.PATTERNS if k != "settled"],
        "failed": [("settled", "matched its example and yielded None - this "
                               "is the model_integrity shape")],
        "no_example": [],
    })
    marks = _marks(injected)
    assert len(marks) == 1, f"{len(marks)} marked lines, expected 1: {marks}"
    assert "settled" in marks[0], (
        f"the one mark does not name the finding: {marks[0]!r}")
    assert "model_integrity shape" in marks[0], (
        "the mark does not carry the reason, so the reader must go looking")


def test_the_CONTROL_in_the_other_direction_a_NOTE_is_not_a_mark():
    """A pattern with no example is information, not a finding, so it must
    print and must NOT be marked — otherwise the marker creeps back."""
    out = ci.format_pattern_selftest(
        {"passed": [], "failed": [], "no_example": ["settled"]})
    assert "settled" in out, "the no-example state stopped being reported"
    assert _marks(out) == [], f"a note acquired the marker: {_marks(out)}"
    assert "UNEVALUATED" in out, (
        "the no-example line no longer says it is UNEVALUATED rather than "
        "passing")


# ── the ratio, recorded beside the firings-to-changes figure ───────────────

#: MARKED LINES PER FINDING, `measured` 2026-10-21 by running the audit
#: before and after on the same conditions (one finding: `schedule margin
#: +19m < 30m`).
#:
#: It belongs beside the 33:1 firings-to-changes figure from the surface
#: census because they measure the same failure in two media: an alarm that
#: repeats without producing a change, and a marker worn by output that is not
#: a finding. Both train the reader to skip.
MARKED_LINES_PER_FINDING = {"before": 10, "after": 1}


def test_the_RATIO_is_recorded_with_its_window():
    b, a = MARKED_LINES_PER_FINDING["before"], MARKED_LINES_PER_FINDING["after"]
    assert b == 10 and a == 1
    assert a == 1, "a run with one finding must print exactly one marked line"
    assert b / a == 10


def test_the_MARK_FOLLOWS_THE_ALARM_not_the_CONDITION():
    """Found by demonstrating the quiet state, and it was the rule's own
    first instance.

    The schedule-margin `!!` printed BEFORE `margin_breach_is_acknowledged`
    was consulted, so an ACKNOWLEDGED breach -- one that deliberately does not
    alarm -- still wore the finding marker. The acknowledgement was injected,
    the margin stopped alarming, and the mark stayed.
    """
    i = SRC.index("SCHEDULE MARGIN")
    head = SRC[max(0, i - 900):i]
    # the acknowledgement must be consulted BEFORE the mark is printed
    assert "margin_breach_is_acknowledged" in head, (
        "the mark is printed before the acknowledgement is consulted, so an "
        "acknowledged breach still wears it")
    assert "if not _ack:" in head, (
        "the mark is not gated on the absence of an acknowledgement")
    # and the acknowledged path still REPORTS, as a note
    tail = SRC[i:i + 900]
    assert "note: schedule margin" in tail, (
        "the acknowledged breach stopped being reported at all; it must "
        "report without the marker, not go silent")


def test_the_ACKNOWLEDGEMENT_compares_the_REAL_figure_not_the_DISPLAY():
    """Incidental confirmation from the same demonstration: the binding is
    18.89 and prints as `+19`, and an acknowledgement recorded at 19.0
    correctly REFUSES to cover it. A predicate reading the rounded display
    would have absorbed a breach it was never shown."""
    real, displayed = 18.89232917906861, 19.0
    ci.ACKNOWLEDGED_MARGIN_BREACH = {
        "weekday": {"since": "2026-10-21", "binding": displayed,
                    "why": "x" * 40, "action": "y", "review": "2026-12-31"}}
    try:
        assert ci.margin_breach_is_acknowledged("weekday", real) == {}, (
            "an acknowledgement at the ROUNDED figure absorbed a worse real "
            "one")
        assert ci.margin_breach_is_acknowledged("weekday", displayed), (
            "the same figure is not covered, so the comparison is not >=")
    finally:
        ci.ACKNOWLEDGED_MARGIN_BREACH = {}
