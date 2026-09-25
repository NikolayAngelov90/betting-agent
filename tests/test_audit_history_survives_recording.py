"""An audit must not destroy its own input.

FOUND 2026-09-25, on an outage in its fourth day.

`produced_recently()` decides whether a source that USED to produce has gone
silent. Its history was assembled from the runs in the current pass:

    by_wf = defaultdict(list)
    for r in runs:
        hits = assertions(facts, by_wf[r["workflow"]])
        by_wf[r["workflow"]].append(facts)

and `--unaudited` excludes every run the ledger already holds. So the last day
a source produced became invisible THE MOMENT IT WAS RECORDED, and the
per-source discovery check — written after Flashscore's 88-day silent death —
was skipped by its own "an empty history means nothing can be said" branch.

    Recording the finding removed the evidence that would fire it again.

Measured: with `--since 2026-09-19` the check fired on 09-22, 09-23 and 09-24.
With `--unaudited` it fired on 09-22 alone, and only because 09-21 happened to
be unaudited in the same batch. An alarm that works once per outage, by
coincidence of batching, is not an alarm — and wiring it to production in that
state would have produced a daily silent pass that READ AS COVERAGE.

The fix: history comes from the ledger, which is the durable record.
"""

import importlib.util
import pathlib
import sys

import pytest

_spec = importlib.util.spec_from_file_location(
    "_ci_audit_hist", pathlib.Path("scripts/ci_audit.py"))
ci = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ci)


LEDGER_ROWS = """
| **35429871254** | daily-picks | 09-19 07:38 | **DEGRADED** | `disc[fs=104c fdo=40c af=12c]` = 156 created |
| **35577396913** | daily-picks | 09-21 08:20 | **DEGRADED** | `disc[fs=2c fdo=0c af=86c]` · 2 picks |
| **35702692750** | daily-picks | 09-22 08:02 | **DEGRADED** | `disc[fs=0c fdo=0c af=0c]` · 0 picks |
| **35071608733** | daily-picks | 09-16 08:02 | **DEGRADED** | `disc[fs=17c/-m fdo=4c/5m af=0c/29m]` |
| 33000000001 | closing-lines | 09-16 20:05 | CLEAN | |
"""


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    p = tmp_path / "ledger.md"
    p.write_text(LEDGER_ROWS, encoding="utf-8")
    monkeypatch.setattr(ci, "LEDGER", p)
    return p


def test_the_ledger_yields_history_for_the_right_workflow(ledger):
    rows = ci.ledger_history("daily-picks")
    assert len(rows) == 4
    assert ci.ledger_history("closing-lines") == [], (
        "a row with no disc[...] became history — it carries no per-source "
        "figure, so it can say nothing")


def test_BOTH_ledger_formats_parse():
    """`disc[fs=2c fdo=0c af=86c]` and `disc[fs=17c/-m fdo=4c/5m af=0c/29m]`.

    The format changed mid-project. A parser that reads only the current one
    would silently treat every older row as no history — which is the same
    failure with a different cause.
    """
    m = ci._LEDGER_DISC.search("x `disc[fs=2c fdo=0c af=86c]` y")
    assert m and m.group(1) == "2" and m.group(5) == "86"
    m2 = ci._LEDGER_DISC.search("`disc[fs=17c/-m fdo=4c/5m af=0c/29m]`")
    # The regex captures the literal "-"; `_n` is what turns it into None, so
    # the two responsibilities are asserted separately rather than conflated.
    assert m2 and m2.group(1) == "17" and m2.group(2) == "-"
    assert ci._n(m2.group(2)) is None
    assert m2.group(4) == "5" and m2.group(6) == "29"


def test_a_dash_is_NOT_REPORTED_and_not_zero():
    """`-m` means the run did not report a matched count.

    Reading it as 0 would let "did not report" satisfy a check about "produced
    nothing" — the collapse this file has closed four times.
    """
    assert ci._n("-") is None
    assert ci._n(None) is None
    assert ci._n("0") == 0
    assert ci._n("86") == 86


def test_produced_recently_SEES_the_last_good_day_through_the_ledger(ledger):
    """THE FIX. 09-21 carries fs=2c and is in the ledger, not in the pass."""
    hist = ci.ledger_history("daily-picks", ci.LOOKBACK_RUNS)
    assert any((h.get("src_flashscore_fixtures") or 0) > 0 for h in hist), (
        "the last day Flashscore produced is invisible — the per-source check "
        "will be skipped and its silence will read as coverage")


def test_the_assertion_FIRES_on_a_silent_source_given_ledger_history(ledger):
    """End to end, on the shape of 09-23: fs=0 with fs>0 in the ledger."""
    facts = {
        "is_first_run_of_day": True,
        "src_flashscore_fixtures": 0,
        "src_footballdataorg_fixtures": 0,
        "src_apifootball_fixtures": 0,
    }
    hits = ci.assertions(facts, ci.ledger_history("daily-picks", ci.LOOKBACK_RUNS))
    assert any("Flashscore fixtures: 0 created AND 0 matched" in h for h in hits), hits


def test_it_does_NOT_fire_when_the_source_never_produced(ledger):
    """The self-calibration must survive the fix.

    A source that has never produced is not a regression, and firing on it is
    the cry-wolf behaviour that got an earlier aggregate check ignored.
    """
    facts = {"is_first_run_of_day": True, "src_flashscore_fixtures": 0}
    hits = ci.assertions(facts, [{"src_flashscore_fixtures": 0}])
    assert not any("Flashscore" in h for h in hits), hits


def test_an_empty_ledger_still_says_nothing_rather_than_passing(tmp_path, monkeypatch):
    """No history is not a clean bill, and the caller announces it.

    `assertions` keeps its "nothing can be said" branch — that part was
    correct. What was wrong was the ledger being unreadable to it.
    """
    p = tmp_path / "empty.md"
    p.write_text("no rows here\n", encoding="utf-8")
    monkeypatch.setattr(ci, "LEDGER", p)
    assert ci.ledger_history("daily-picks") == []
    facts = {"is_first_run_of_day": True, "src_flashscore_fixtures": 0}
    assert not any("Flashscore" in h for h in ci.assertions(facts, []))


def test_history_is_oldest_first_so_the_lookback_window_slices_correctly(ledger):
    """`history[-LOOKBACK_RUNS:]` takes the MOST RECENT runs.

    Returning newest-first would make the window select the oldest rows, and
    the check would answer about the wrong days without erroring.
    """
    rows = ci.ledger_history("daily-picks")
    assert rows[0]["src_flashscore_fixtures"] == 104     # 09-19, first in file
    assert rows[-1]["src_flashscore_fixtures"] == 17     # 09-16 row, last in file
    assert ci.ledger_history("daily-picks", 1) == [rows[-1]]


# ── THE POLICY THE WIRING NEEDED ─────────────────────────────────────────────

WORKFLOW = pathlib.Path(".github/workflows/ci-audit.yml")


def test_the_audit_is_wired_and_cannot_audit_itself_from_inside():
    """`gh run view --log` is empty until a run completes.

    So the invocation has to be a SEPARATE run, not a step in daily-picks.
    Pinned because the obvious design is the impossible one.
    """
    assert WORKFLOW.is_file(), "the audit is not wired to anything"
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "workflow_run" in text
    daily = pathlib.Path(".github/workflows/daily-picks.yml").read_text(encoding="utf-8")
    assert "ci_audit" not in daily, (
        "ci_audit was added as a step inside daily-picks — it cannot read its "
        "own log, so it would audit nothing")


def test_red_is_NARROW_and_degraded_only_reports():
    """DEGRADED is this pipeline's ordinary state — eight rows every day.

    Failing on it would make red mean nothing, which is the noise DEL-2 was
    deliberately narrowed to avoid. BROKEN and DID_NOT_RUN are the axis.
    """
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "--fail-on BROKEN,DID_NOT_RUN" in text
    assert "--fail-on BROKEN,DID_NOT_RUN,DEGRADED" not in text


def test_the_alert_fires_on_ANY_finding_even_though_red_does_not():
    """Reporting and failing are separated, so the alert is wider than red."""
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "steps.audit.outcome != 'success'" in text
    assert "ci_alert" in text


# ── WORKFLOW STATE: the one thing no run-based check can see ─────────────────
#
# FOUND 2026-09-25 after 7h48m with nothing running. Every check in this tool
# reads RUNS; a disabled workflow produces none, so "disabled" and "nothing was
# scheduled" were the same observation. The answer that day was `active` for all
# four — and it was obtained by a command OUTSIDE the tool, which is the gap
# regardless of the answer.

def test_a_disabled_workflow_alarms_even_with_ZERO_runs(monkeypatch, capsys):
    """THE CASE IT EXISTS FOR, and the case the first version could not reach.

    The check was first placed AFTER `if not runs: return 0`. A disabled
    workflow produces no runs, so that branch printed "No runs to audit." and
    exited 0 before the check ran — the check was unreachable in precisely the
    situation it was written for. Found by simulating it, not by reading it.
    """
    monkeypatch.setattr(ci, "workflow_states",
                        lambda: {"Daily Betting Picks": "disabled_inactivity",
                                 "Closing Line Capture": "active"})
    monkeypatch.setattr(sys, "argv", ["ci_audit", "--since", "2030-01-01"])
    code = ci.main()
    out = capsys.readouterr().out
    assert code == 1, "a disabled workflow exited 0"
    assert "disabled_inactivity" in out
    assert "not active" in out
    assert "audit alarm" in out


def test_it_is_NOT_behind_fail_on(monkeypatch, capsys):
    """Not self-calibrating. A disabled scheduled workflow is wrong on the
    FIRST occurrence, like a spent credit that returned no rows — so it alarms
    without the caller opting in."""
    monkeypatch.setattr(ci, "workflow_states",
                        lambda: {"X": "disabled_manually"})
    monkeypatch.setattr(sys, "argv", ["ci_audit", "--since", "2030-01-01"])
    assert ci.main() == 1          # no --fail-on given
    assert "disabled_manually" in capsys.readouterr().out


def test_all_active_is_SILENT(monkeypatch, capsys):
    """The normal case must add no noise, or it will be tuned out."""
    monkeypatch.setattr(ci, "workflow_states", lambda: {"A": "active", "B": "active"})
    monkeypatch.setattr(sys, "argv", ["ci_audit", "--since", "2030-01-01"])
    ci.main()
    out = capsys.readouterr().out
    assert "not active" not in out
    assert "state UNKNOWN" not in out


def test_an_unreadable_gh_is_UNKNOWN_and_says_so(monkeypatch, capsys):
    """`None`, never `{}`. "Could not look" is not "found nothing" — the fifth
    time this file has had to separate those two."""
    monkeypatch.setattr(ci, "_sh", lambda *a: "")
    assert ci.workflow_states() is None
    monkeypatch.setattr(ci, "_sh", lambda *a: "}{ not json")
    assert ci.workflow_states() is None

    monkeypatch.setattr(ci, "workflow_states", lambda: None)
    monkeypatch.setattr(sys, "argv", ["ci_audit", "--since", "2030-01-01"])
    ci.main()
    out = capsys.readouterr().out
    assert "state UNKNOWN" in out
    assert "indistinguishable from a quiet one" in out
