"""The s5.19 decision record and the live picks bound cannot be confused.

WHY THIS FILE EXISTS. 2026-11-06.

`FULL_CARD_PICKS_MAX` was substituted FORWARD — 88.1 (serial, FULL-CARD, n=4,
measured 2026-10-01) to 40.35 (parallel, ALL-CARD, n=12, measured
2026-11-06) — while the assertions that record WHY the weekday cron moved to
`0 1` kept 88.1.

    A DECISION'S RECORD KEEPS THE FIGURES THE DECISION WAS MADE ON.

Re-pointing those assertions at the live constant would make them assert an
argument that was never made: that `0 1` was chosen against a 40.35 bound,
which would have implied a *different* cron (the latest boundary clearing 60
moves 60 -> 105 weekday under the new bound). The argument and the pipeline
state are two different things and must not be able to drift into each other.

SO THE SEPARATION IS ASSERTED IN BOTH DIRECTIONS, and the known-wrong control
SWAPS them and requires the failure — because "they are separate" is
unverifiable unless confusing them is shown to break something.
"""
import ast
import pathlib

import pytest

import scripts.ci_audit as ci

ROOT = pathlib.Path(__file__).resolve().parent.parent
CRON_TESTS = ROOT / "tests" / "test_cron_moved.py"
SRC = CRON_TESTS.read_text(encoding="utf-8")

#: The tests whose assertions record the s5.19 ARGUMENT. Each must read the
#: decision record and never the live constant.
ARGUMENT_TESTS = (
    "test_the_CRON_is_the_LATEST_that_clears_60_on_BOTH_pairings",
    "test_the_MARGIN_at_BOTH_FLOORS_clears_the_registered_minimum",
    "test_the_MOVE_SPENDS_REAL_MARGIN_and_is_NOT_a_repair",
    "test_the_POOLED_reading_is_the_ARTEFACT_that_made_it_look_like_a_repair",
    "test_the_REJECTED_CRON_is_a_BREACH_by_the_projects_OWN_instrument",
    "test_the_BREACH_CLEARED_BY_A_MEASUREMENT_CORRECTION_not_the_pipeline",
)


def _sites(text):
    """`{enclosing test: {names read}}` for every picks-bound read."""
    tree = ast.parse(text)
    fns = [(n.lineno, n.end_lineno, n.name) for n in ast.walk(tree)
           if isinstance(n, ast.FunctionDef)]
    out = {}
    for i, line in enumerate(text.splitlines(), 1):
        if "FULL_CARD_PICKS" not in line and "_S519" not in line:
            continue
        owner = next((nm for a, b, nm in fns if a <= i <= b), "<module>")
        seen = out.setdefault(owner, set())
        if "ci.FULL_CARD_PICKS" in line:
            seen.add("LIVE")
        if "_S519[" in line:
            seen.add("RECORD")
    return out


# ── THE TWO FIGURES ARE DIFFERENT, OR NOTHING BELOW MEANS ANYTHING ─────────

def test_the_TWO_FIGURES_DIFFER_so_the_separation_is_testable():
    """If they were equal, every assertion here would pass either way."""
    rec = ci.PICKS_BOUND_AT_S519_DECISION
    assert rec["max"] != ci.FULL_CARD_PICKS_MAX, (
        f"the decision record and the live bound are both "
        f"{ci.FULL_CARD_PICKS_MAX}; the separation cannot be verified while "
        f"they agree, so this file is UNEVALUATED rather than passing")
    assert rec["mean_plus_3sd"] != ci.FULL_CARD_PICKS_3SD
    assert rec["max"] == 88.1 and rec["mean_plus_3sd"] == 107.0
    assert ci.FULL_CARD_PICKS_MAX == pytest.approx(40.35)
    assert ci.FULL_CARD_PICKS_3SD == pytest.approx(64.59)


def test_the_RECORD_CARRIES_ITS_DATE_REGIME_n_AND_SUCCESSOR():
    """A figure kept for history is useless without what produced it."""
    rec = ci.PICKS_BOUND_AT_S519_DECISION
    assert rec["measured"] == "2026-10-01"
    assert rec["n"] == 4
    assert "SERIAL" in rec["regime"] and "FULL-CARD" in rec["regime"]
    assert "s5.19" in rec["used_for"]
    sup = rec["superseded_by"]
    assert sup["n"] == 12 and sup["measured"] == "2026-11-06"
    assert "PARALLEL" in sup["regime"] and "ALL-CARD" in sup["regime"]
    assert sup["max"] == pytest.approx(ci.FULL_CARD_PICKS_MAX), (
        "the record's successor no longer names the live bound, so a reader "
        "cannot get from the historical figure to the current one")
    assert rec.get("why_not_updated"), (
        "nothing says why this record is not updated with the live figure")


# ── DIRECTION 1: the ARGUMENT must not read the LIVE constant ──────────────

def test_DIRECTION_1_the_ARGUMENT_assertions_READ_ONLY_THE_RECORD():
    sites = _sites(SRC)
    for name in ARGUMENT_TESTS:
        assert name in sites, (
            f"{name} no longer reads a picks bound at all, so this check "
            f"verifies nothing about it -- was it renamed or deleted?")
        assert "LIVE" not in sites[name], (
            f"{name} records the s5.19 ARGUMENT and reads the LIVE constant. "
            f"The argument was made against 88.1; asserting it against the "
            f"live bound claims a decision that was never taken.")
        assert "RECORD" in sites[name], f"{name} reads neither figure"


# ── DIRECTION 2: the LIVE path must not read the RECORD ────────────────────

def test_DIRECTION_2_the_LIVE_MARGIN_PATH_READS_ONLY_THE_LIVE_CONSTANT():
    """`ci_audit`'s own margin computation must never reach for the record."""
    audit = (ROOT / "scripts" / "ci_audit.py").read_text(encoding="utf-8")
    tree = ast.parse(audit)
    offenders = []
    for n in ast.walk(tree):
        if not isinstance(n, ast.FunctionDef):
            continue
        body = ast.unparse(n)
        if "PICKS_BOUND_AT_S519_DECISION" in body:
            offenders.append(n.name)
    assert offenders == [], (
        f"these functions in ci_audit read the s5.19 DECISION RECORD: "
        f"{offenders}. The live margin path must read the live constant; the "
        f"record is for the argument, not for arithmetic.")
    # ── AND schedule_deadline MUST NOT REACH FOR EITHER -- BY AST. ────────
    #
    # NOT a raw-text `not in`. A comment mentioning either name would satisfy
    # that falsely, and this file's own commentary names both -- so the claim
    # is made over NAMES THE PARSER SEES, which comments and docstrings are
    # not. (It also keeps this site out of the raw-match backlog, whose
    # clearing action is exactly this.)
    sd_tree = ast.parse(
        (ROOT / "scripts" / "schedule_deadline.py").read_text(encoding="utf-8"))
    referenced = {n.id for n in ast.walk(sd_tree) if isinstance(n, ast.Name)}
    referenced |= {n.attr for n in ast.walk(sd_tree)
                   if isinstance(n, ast.Attribute)}
    # THE ANCHOR, as a size rather than a truthiness: a negative claim over
    # a name set verifies nothing unless the set is demonstrably the module's.
    assert len(referenced) >= 50, (
        f"the parse produced only {len(referenced)} name(s) from "
        f"schedule_deadline, so this is not the module being checked and the "
        f"absence claims below verify nothing")
    assert "MIN_ACCEPTABLE_MARGIN_MIN" in referenced, (
        "a known-present name is missing from the parse, so the haystack is "
        "not what it should be")
    for name in ("FULL_CARD_PICKS_MAX", "FULL_CARD_PICKS_3SD",
                 "PICKS_BOUND_AT_S519_DECISION"):
        assert name not in referenced, (
            f"schedule_deadline now references {name} IN CODE; the derived "
            f"deadline was independent of the picks bound, and that "
            f"independence is why substituting it did not cascade into the "
            f"floor")


# ── THE KNOWN-WRONG CONTROL: SWAPPING THEM MUST FAIL ──────────────────────

def test_KNOWN_WRONG_SWAPPING_THEM_FAILS():
    """The control. Without it, "they are separate" is unfalsifiable.

    Each direction's predicate is re-run against a MUTATED copy of the source
    in which the two reads are exchanged. Both must then report an offender.
    """
    # direction 1, swapped: the argument tests now read the live constant
    swapped = SRC.replace('_S519["max"]', "ci.FULL_CARD_PICKS_MAX") \
                 .replace('_S519["mean_plus_3sd"]', "ci.FULL_CARD_PICKS_3SD")
    assert swapped != SRC, "the swap changed nothing, so it tests nothing"
    sites = _sites(swapped)
    caught = [n for n in ARGUMENT_TESTS
              if "LIVE" in sites.get(n, set())]
    assert len(caught) == len(ARGUMENT_TESTS), (
        f"swapping the reads was NOT caught for "
        f"{set(ARGUMENT_TESTS) - set(caught)}; direction 1 cannot detect the "
        f"confusion it exists to prevent")

    # direction 2, swapped: ci_audit's margin path now reads the record
    audit = (ROOT / "scripts" / "ci_audit.py").read_text(encoding="utf-8")
    planted = audit.replace(
        "def verdict(", "def _planted():\n"
        "    return PICKS_BOUND_AT_S519_DECISION['max']\n\n\ndef verdict(", 1)
    assert planted != audit, "the plant did not apply"
    tree = ast.parse(planted)
    found = [n.name for n in ast.walk(tree)
             if isinstance(n, ast.FunctionDef)
             and "PICKS_BOUND_AT_S519_DECISION" in ast.unparse(n)]
    assert "_planted" in found, (
        "direction 2 cannot find a function that reads the decision record, "
        "so its empty result on the real file proves nothing")


def test_KNOWN_WRONG_an_EQUAL_PAIR_is_UNEVALUATED_not_clean():
    """If the two figures ever agree, the separation stops being testable and
    this file must say so rather than passing."""
    rec = dict(ci.PICKS_BOUND_AT_S519_DECISION, max=ci.FULL_CARD_PICKS_MAX)
    assert rec["max"] == ci.FULL_CARD_PICKS_MAX
    # the real guard is test_the_TWO_FIGURES_DIFFER; this pins that a mutated
    # equal pair is exactly what it refuses
    assert ci.PICKS_BOUND_AT_S519_DECISION["max"] != ci.FULL_CARD_PICKS_MAX, (
        "the live and recorded figures now agree, which makes every "
        "separation assertion in this file vacuous")
