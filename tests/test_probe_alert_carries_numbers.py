"""The probe's exit taxonomy and the alert's two numbers — 2026-10-12.

WHAT HAPPENED. The probe shipped 2026-10-10 and has run **once**, on
2026-10-11 at 21:45 UTC (run 38177680734). It died after **21 seconds** with
`ModuleNotFoundError: No module named 'aiohttp'` — the workflow installed a
hand-curated package list that omitted it, while `requirements.txt` carried it
the whole time. The `if: failure()` step then sent the operator:

    "Tomorrow's picks may land AFTER the first kickoff. ACTION: dispatch
     daily-picks manually tomorrow morning ..."

— a **margin finding, from a job that computed no margin at all**. The text
carried a disjunction ("if instead the probe refused or could not verify its
clock") that the reader had no way to resolve.

    A designed refusal must be distinguishable from a crash, in the exit
    status and in the log. And an alert that asks the operator to subtract a
    runtime from a kickoff he has to look up himself has not told him
    anything he did not have to work out.

So: four exit codes, three alert texts, and a body that carries both numbers
in both clocks.
"""

import datetime as dt
import pathlib
import re

import pytest

import scripts.fixture_horizon_probe as probe


# ── THE EXIT TAXONOMY ────────────────────────────────────────────────────────

def test_the_FOUR_exit_codes_are_distinct_and_a_REFUSAL_is_not_a_CRASH():
    """Hard rule 1. Until today all three non-zero outcomes exited 1."""
    codes = (probe.EXIT_CLEAR, probe.EXIT_ALARM, probe.EXIT_CRASH,
             probe.EXIT_REFUSED)
    assert codes == (0, 1, 2, 3)
    assert len(set(codes)) == 4
    # and the LOG classifies itself too, not only the exit status
    for code in (probe.EXIT_ALARM, probe.EXIT_CRASH, probe.EXIT_REFUSED):
        assert probe.MARKER[code].startswith("PROBE ")
    assert probe.MARKER[probe.EXIT_CRASH] != probe.MARKER[probe.EXIT_REFUSED]


def test_a_CRASH_returns_2_and_announces_itself_as_MACHINERY(monkeypatch,
                                                             tmp_path):
    """The exact failure of run 38177680734, driven.

    Asserting on the BODY as well as the code: the code is what the workflow
    branches on, and the body is what the operator reads.
    """
    def _boom(argv=None):
        raise ModuleNotFoundError("No module named 'aiohttp'")

    monkeypatch.setattr(probe, "main", _boom)
    out = tmp_path / "body.txt"
    monkeypatch.setenv("PROBE_ALERT_BODY", str(out))

    assert probe._main_guarded([]) == probe.EXIT_CRASH

    body = out.read_text(encoding="utf-8")
    assert "PROBE CRASHED" in body
    assert "MACHINERY, not a finding" in body
    assert "NOTHING was probed" in body
    assert "do not read the absence of a deadline as a safe one" in body
    # it must NOT read as a margin finding
    assert "may land AFTER the first kickoff" not in body


def test_a_clean_run_returns_0_through_the_guard(monkeypatch):
    """The control for the above: the boundary must not convert a pass."""
    monkeypatch.setattr(probe, "main", lambda argv=None: probe.EXIT_CLEAR)
    assert probe._main_guarded([]) == probe.EXIT_CLEAR


def test_SystemExit_passes_through_the_guard(monkeypatch):
    """`--help` exits via SystemExit and must not be reported as a crash."""
    def _exit(argv=None):
        raise SystemExit(0)

    monkeypatch.setattr(probe, "main", _exit)
    with pytest.raises(SystemExit):
        probe._main_guarded([])


# ── THE TWO NUMBERS, RENDERED ────────────────────────────────────────────────

#: A known kickoff, so both numbers are arithmetic a reader can check.
#: 2026-10-13 is a Tuesday; Europe/Sofia is EEST (+03:00) on that date.
DAY = dt.date(2026, 10, 13)
KICKOFF_MIN = 11 * 60 + 30            # 11:30 UTC


def test_POSITIVE_CONTROL_the_body_carries_BOTH_numbers_in_BOTH_clocks():
    """Part E's control. An alert body never rendered in a test is a formatter
    with no consumer — which is this project's most-repeated shape."""
    body = probe.format_dispatch_alert(
        DAY, KICKOFF_MIN, "netherlands/eerste-divisie",
        basis="deadline DISCOVERED by the probe", margin=-37.0,
        run_url="https://example.invalid/run/1")

    # 1. THE EARLIEST KICKOFF, both clocks, with the league named
    assert "EARLIEST KICKOFF  11:30 UTC" in body, body
    assert "14:30" in body, "the Europe/Sofia time is missing"
    assert "EEST" in body or "+0300" in body, "the local clock is not NAMED"
    assert "netherlands/eerste-divisie" in body, "the league is not named"

    # 2. THE LAST SAFE DISPATCH = 11:30 - 88.1 - 15 = 09:46 UTC
    assert "LAST SAFE DISPATCH 09:46 UTC" in body, body
    assert "12:46" in body, "the local last-safe-dispatch is missing"

    # the arithmetic is SHOWN, not just the result
    assert "88m time-to-picks" in body
    assert "NOT the mean" in body
    assert "15m safety margin" in body
    assert "n=4" in body and "67.4-88.1" in body, (
        "the upper bound carries no provenance")

    # 3. the BASIS of the deadline, per hard rule 3
    assert "DISCOVERED by the probe" in body

    assert "-37m" in body or "projected margin -37" in body, body
    assert "gh workflow run daily-picks.yml" in body
    assert "https://example.invalid/run/1" in body


def test_EVERY_time_in_the_body_NAMES_ITS_CLOCK():
    """Five instances of "establish the clock" rest on one ledger site. An
    alert printing a bare hour is the next one, so no HH:MM may stand alone."""
    body = probe.format_dispatch_alert(
        DAY, KICKOFF_MIN, "x/y", basis="deadline DISCOVERED by the probe")
    for line in body.splitlines():
        for m in re.finditer(r"\b(\d{2}:\d{2})\b", line):
            after = line[m.end():m.end() + 14]
            assert ("UTC" in after or "EEST" in after or "EET" in after
                    or "+0" in after or "unavailable" in after), (
                f"a bare time {m.group(1)!r} with no clock named: {line!r}")


def test_the_UPPER_BOUND_is_used_and_not_the_MEAN():
    """Hard rule 4, asserted on the arithmetic rather than on the prose.

    mean 76.4 would give 10:00 UTC; the upper bound 88.1 gives 09:46. Fourteen
    minutes, and the whole purpose of the figure is to not be late.
    """
    assert probe.PICKS_UPPER_BOUND_MIN == 88.1
    body = probe.format_dispatch_alert(DAY, KICKOFF_MIN, None, basis="b")
    assert "09:46" in body
    assert "10:00 UTC" not in body, "the MEAN was used, not the upper bound"


def test_a_pre_dawn_kickoff_says_THE_DAY_BEFORE():
    """An edge the arithmetic produces and a reader would otherwise misread:
    kickoff 01:00 means the last safe dispatch is the previous evening."""
    body = probe.format_dispatch_alert(
        DAY, 60, "x/y", basis="deadline DISCOVERED by the probe")
    assert "THE DAY BEFORE" in body, body


def test_the_body_states_the_BASIS_whichever_it_IS():
    """Hard rule 3. A fallback-derived deadline must say so — an alert whose
    basis is not stated is indistinguishable from one that measured
    something."""
    discovered = probe.format_dispatch_alert(
        DAY, KICKOFF_MIN, "x/y", basis="deadline DISCOVERED by the probe")
    fallback = probe.format_dispatch_alert(
        DAY, 12 * 60, None,
        basis="deadline from the 12:00 weekday FALLBACK constant, not probed")
    assert "DISCOVERED by the probe" in discovered
    assert "FALLBACK constant, not probed" in fallback
    assert discovered != fallback


# ── THE WIRING ───────────────────────────────────────────────────────────────

def test_the_workflow_BRANCHES_on_the_exit_code():
    """One fixed sentence for every non-zero exit is the defect this replaces.
    """
    wf = pathlib.Path(".github/workflows/fixture-horizon-probe.yml").read_text(
        encoding="utf-8")
    assert "probe_exit" in wf, "the exit code is not captured"
    assert 'CODE" = "2"' in wf and 'CODE" = "3"' in wf, (
        "the alert does not branch on crash vs refusal")
    assert "PROBE CRASHED" in wf and "PROBE REFUSED" in wf
    assert "BODY_FILE" in wf, "the rendered body is not used by the alert"


def test_the_probe_installs_from_REQUIREMENTS_not_a_curated_list():
    """The cause of run 38177680734, pinned.

    A hand-curated list is a second definition of the dependency set, and the
    two diverge the moment an import is added upstream. `aiohttp` was in
    requirements.txt the whole time.
    """
    wf = pathlib.Path(".github/workflows/fixture-horizon-probe.yml").read_text(
        encoding="utf-8")
    assert "pip install -r requirements.txt -c constraints.txt" in wf, (
        "the probe is back on a hand-curated dependency list")
    assert "aiohttp" in pathlib.Path("requirements.txt").read_text(
        encoding="utf-8"), "aiohttp left requirements.txt"
