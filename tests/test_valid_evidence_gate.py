"""Stage 13 (s5.3) — every learner and measurer must gate on `valid_evidence()`.

This guard exists because a hand enumeration was wrong, and it has now been
widened twice because the guard itself was wrong in the same way.

**Pass 1 — the hand count.** Seven `_live_only()` call sites, five classified as
learners. That counts sites which ROUTE THROUGH the shared predicate, and is
structurally blind to sites that bypass it — the lesson `feature_engineer`'s
hand-copied `is_fixture == False` already taught.

**Pass 2 — scan for readers instead of callers.** Two more in `betting_agent`:
`get_claude_added_value` (a real gap, and measuring over paper picks too) and
`_reset_stale_ml_calibration` (a sentinel, exempt).

**Pass 3 — the spelling and the scope.** Pass 2 matched only
`result.isnot(None)` and scanned only `betting_agent.py`. Both too narrow.
`match_briefing` writes it as `result.in_(["win", "loss"])`, in a different
file, and that blindness hid the most consequential finding of the stage:

    _recent_selection_stats   settled outcomes -> the KEEP/CHANGE prompt
    _recent_review_stats      settled outcomes -> the KEEP/CHANGE prompt

Neither filtered paper picks — `is_paper` appeared zero times in that file. So
paper-pick outcomes were computed into statistics, injected into the decision
prompt, and used by Claude to choose the pick the FINAL series measures. The
isolation was real on the model path and absent on the review path.

A third site turned up with them: `probability_calibration.fit_from_db`, which
excluded paper picks correctly but not picks whose features described a
different club.

The generalisable lesson, worth more than any of the three: **a guard that
recognises one dialect of a predicate reports a clean population it never
looked at.**
"""

import collections
import pathlib
import re

import pytest

from tests.experiment_pins import EVIDENCE_GATE_EXEMPTIONS

#: An exemption must name exactly ONE category. The generic form fits
#: everywhere, which is what makes it paste-able past review.
EXEMPT_MARKER = re.compile(
    r"evidence-gate:\s*NOT GATED\s*\((populates|repairs|resolves)\)")

#: "Settled" has more than one spelling; see the module docstring.
SETTLED = re.compile(
    r"SavedPick\.result\s*\.\s*isnot\(\s*None\s*\)"
    r"|SavedPick\.result\s*\.\s*in_\("
    r"|SavedPick\.result\s*==\s*[\"']"
    r"|SavedPick\.model_result\s*\.\s*isnot\(\s*None\s*\)"
    r"|SavedPick\.model_result\s*\.\s*in_\(")

#: Every module that can reach saved_picks, not one file.
SCANNED = (sorted(pathlib.Path("src").rglob("*.py"))
           + sorted(pathlib.Path("scripts").rglob("*.py")))

#: Functions that consume settled outcomes to adjust or measure the model.
#: Named explicitly: this is a classification, and it should be argued about in
#: review rather than inferred by a regex.
LEARNERS_AND_MEASURERS = {
    "tune_ensemble_weights",
    "learn_from_settled",
    "calibrate_from_pick_outcomes",
    "_auto_calibrate_ev_threshold",
    "rolling_backtest",
    "get_claude_added_value",
    "fit_from_db",
    "_recent_selection_stats",
    "_recent_review_stats",
}


def _bodies():
    """{(path, function): source} across every scanned module."""
    out = collections.defaultdict(list)
    for path in SCANNED:
        current = "<module>"
        for line in path.read_text(encoding="utf-8").splitlines():
            m = re.match(r"\s*(?:async\s+)?def\s+(\w+)", line)
            if m:
                current = m.group(1)
            out[(str(path), current)].append(line)
    return {k: chr(10).join(v) for k, v in out.items()}


def _find(fn):
    return {k: v for k, v in _bodies().items() if k[1] == fn}


@pytest.mark.parametrize("fn", sorted(LEARNERS_AND_MEASURERS))
def test_every_learner_gates_on_valid_evidence(fn):
    found = _find(fn)
    assert found, f"{fn} no longer exists — update the classification"
    for (path, _), body in found.items():
        assert "valid_evidence()" in body, (
            f"{path}:{fn} consumes settled pick outcomes to adjust or measure "
            "the model but does not gate on valid_evidence(). A pick whose "
            "features described a different club would teach from an outcome "
            "the model never actually predicted.")


@pytest.mark.parametrize("fn", ["_recent_selection_stats", "_recent_review_stats",
                                "fit_from_db", "get_claude_added_value"])
def test_the_review_path_also_excludes_paper_picks(fn):
    """The leak itself, pinned per site.

    These four feed statistics or calibration from settled outcomes. Three had
    no paper filter at all, and two of those fed the KEEP/CHANGE prompt — so
    the experiment was informing the review that produces the FINAL series it
    measures.
    """
    for (path, _), body in _find(fn).items():
        assert "live_only()" in body or "is_paper" in body, (
            f"{path}:{fn} reads settled outcomes without excluding paper "
            "picks — measurement-only rows would inform a decision that "
            "persists a real pick")


def test_no_ungated_reader_of_settled_picks():
    """The generalisation — the part a hand count cannot do."""
    nl = chr(10)
    offenders = []
    for (path, fn), body in _bodies().items():
        if not SETTLED.search(body):
            continue
        if "valid_evidence()" in body or EXEMPT_MARKER.search(body):
            continue
        offenders.append(f"{path}:{fn}")

    assert not offenders, (
        "function(s) reading settled pick outcomes with neither "
        "valid_evidence() nor an exemption marker." + nl + nl +
        "THE RULE: live_only() asks whether the WAGER HAPPENED. "
        "valid_evidence() asks whether the observation says anything true "
        "ABOUT THE MODEL. A learner needs both; the settled record needs only "
        "the first." + nl + nl +
        "  learns/measures -> add valid_evidence() to the filter" + nl +
        "  otherwise       -> mark it at the query:" + nl +
        "      # evidence-gate: NOT GATED (populates|repairs|resolves) - why" +
        nl + nl + "Ungated:" + nl + "  " + (nl + "  ").join(sorted(offenders)))


def test_the_number_of_exemptions_is_pinned():
    """Observe the population, not only the shape."""
    found = []
    for path in SCANNED:
        for n, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), 1):
            if EXEMPT_MARKER.search(line):
                found.append(f"{path}:{n}")
    assert len(found) == EVIDENCE_GATE_EXEMPTIONS, (
        f"{len(found)} evidence-gate exemption(s), pin says "
        f"{EVIDENCE_GATE_EXEMPTIONS}. If the new one is legitimate, say so by "
        "editing EVIDENCE_GATE_EXEMPTIONS in tests/experiment_pins.py."
        + chr(10) + "  " + (chr(10) + "  ").join(found))


def test_the_settled_record_is_deliberately_not_gated():
    """Where the split earns its keep.

    `get_stats()` is the ROI record and the cold-streak alert is risk
    management on real money. Both ask whether the bet happened, which
    `evidence_status` does not answer. Gating them would silently drop three
    real settled wagers from the published record.
    """
    for (path, _), body in _find("get_stats").items():
        assert EXEMPT_MARKER.search(body), (
            "get_stats lost its exemption marker — if it were gated, the "
            "settled record would change and the money would not have")
        assert "valid_evidence()" not in body, (
            "get_stats is now gated on evidence — the wagers were real")


def test_the_predicates_live_in_one_place():
    """The root cause of the leak was a hand-copied filter.

    `match_briefing` wrote its own because the predicates lived inside
    `betting_agent`. They now live in `src/data/pick_filters.py`, so the next
    module that needs them imports rather than reinvents.
    """
    mod = pathlib.Path("src/data/pick_filters.py").read_text(encoding="utf-8")
    assert "def live_only()" in mod and "def valid_evidence()" in mod
    briefing = pathlib.Path(
        "src/reporting/match_briefing.py").read_text(encoding="utf-8")
    assert "from src.data.pick_filters import" in briefing
