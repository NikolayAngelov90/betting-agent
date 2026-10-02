"""Stage 27 — the three reporting defects and the dead ceiling.

Four instruments were wrong in four different ways, and none of them lost data:

  A  `##[error]` was split two ways, and on every alarming `ci-audit` run BOTH
     annotations were the audit's own verdict exits. **A detector whose expected
     baseline equals its threshold has no dynamic range** — a genuine third
     absorbed failure there was invisible. Fixed by a third bucket, identified
     BY SOURCE: every pattern is pinned to the file that emits it.

  B  the 10-01 `DID_NOT_RUN` alarm was remediated in `a4d5228` and fired eight
     more times, clearing only when `--since yesterday` rolled past the run.
     **An alarm cleared by time teaches waiting.** CLR-2 wants an action, so
     recording the resolution IS the action and it changes the predicate.

  C  "the apparatus is not working" fired OUTSIDE the collection window, where
     zero trajectories is correct. NOT-STARTED and STARTED-AND-FAILING are the
     third-state family, in the check's own headline.

  D  `CREDIT_CEILING = 200` was unreachable: the 168 reservation always bound
     first. Two independent bounds over one pool with the weaker one DEAD.

Every silence below has a positive control, because a positive control is the
only reason to trust a silence.
"""

import datetime as dt
import importlib.util
import pathlib

import pytest

_spec = importlib.util.spec_from_file_location(
    "_ci_audit_sat", pathlib.Path("scripts/ci_audit.py"))
ci = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ci)

from scripts.h1_collection_check import (
    APPARATUS_STATES,
    COLLECTION_STATES,
    COLLECTION_TERMINAL_STATES,
    CREDIT_CEILING,
    TARGET_N,
    apparatus_state,
    collection_stop,
    effective_bound,
)
from src.data.odds_quota import H1_RESERVED_CREDITS


# ══════════════════ A: SATURATION REMOVED ═══════════════════════════════════

_AUDIT_LOG = (
    "some normal output\n"
    "::error::audit alarm — 36843668865 daily-picks DID_NOT_RUN\n"
    "##[error]audit alarm — 36843668865 daily-picks DID_NOT_RUN\n"
    "##[error]Process completed with exit code 1.\n"
    "::error::CI audit alarmed — see the audit step for the run ids.\n"
    "##[error]CI audit alarmed — see the audit step for the run ids.\n"
    "##[error]Process completed with exit code 1.\n"
)


def test_the_three_counts_PARTITION_the_annotations_exactly():
    """exit + self-emitted + foreign == every `##[error]` in the log.

    A partition that does not sum is two overlapping counts wearing the name.
    """
    import re
    for log in (_AUDIT_LOG, "", "##[error]Something foreign\n",
                _AUDIT_LOG + "##[error]Process completed with exit code 2.\n"):
        f = ci.extract(log)
        total = len(re.findall(r"##\[error\]", log))
        assert (f["exit_annotations"] + f["self_emitted_annotations"]
                + f["foreign_annotations"]) == total, (
            f"counts do not partition: {f['exit_annotations']} + "
            f"{f['self_emitted_annotations']} + {f['foreign_annotations']} "
            f"!= {total}")


def test_the_audit_own_alarm_leaves_NO_remainder():
    """The baseline is subtracted, so an alarming `ci-audit` run is explained."""
    f = ci.extract(_AUDIT_LOG)
    assert f["exit_annotations"] == 2
    assert f["self_emitted_annotations"] == 2
    assert f["self_emitted_exits"] == 2
    assert f["unexplained_nonzero_exit"] == 0, (
        "the audit's own two verdict exits still read as unexplained — the "
        "detector is still saturated at its own baseline")


def test_POSITIVE_CONTROL_a_THIRD_exit_from_a_FOREIGN_source_moves_the_verdict():
    """THE CONTROL. If the subtraction removed the signal with the baseline,
    this is where it shows."""
    foreign = _AUDIT_LOG + "##[error]Process completed with exit code 127.\n"
    f = ci.extract(foreign)
    assert f["exit_annotations"] == 3
    assert f["self_emitted_exits"] == 2
    assert f["unexplained_nonzero_exit"] == 1, (
        "a third, foreign non-zero exit was absorbed by the subtraction — the "
        "baseline removal took the signal with it")

    hits = ci.assertions(f, [])
    assert any("exited NON-ZERO" in h for h in hits), (
        "the finding does not fire on an unexplained exit, so the verdict "
        "cannot move")
    # and the clean case must stay silent
    assert not any("exited NON-ZERO" in h for h in ci.assertions(
        ci.extract(_AUDIT_LOG), [])), (
        "the finding fires on the audit's own baseline — nothing was fixed")


def test_every_self_emitted_annotation_names_a_REAL_emitter():
    """BY SOURCE, enforced. An entry cannot survive its emitter being deleted,
    and cannot be added for a message nothing prints."""
    assert ci.SELF_EMITTED_ANNOTATIONS
    for pat, src in ci.SELF_EMITTED_ANNOTATIONS:
        p = pathlib.Path(src)
        assert p.is_file(), f"{pat!r} names a source that does not exist: {src}"
        assert pat in p.read_text(encoding="utf-8"), (
            f"{pat!r} is registered as self-emitted but {src} does not contain "
            f"it — the registry has drifted into guessing from the text")


def test_a_foreign_annotation_is_NOT_counted_as_self_emitted():
    f = ci.extract("##[error]Some action blew up\n")
    assert f["self_emitted_annotations"] == 0
    assert f["foreign_annotations"] == 1


def test_the_remainder_cannot_go_NEGATIVE():
    """More annotations than exits must not read as 'fewer than none'."""
    log = ("::error::audit alarm x\n##[error]audit alarm x\n"
           "::error::CI audit alarmed y\n##[error]CI audit alarmed y\n")
    f = ci.extract(log)
    assert f["exit_annotations"] == 0
    assert f["unexplained_nonzero_exit"] == 0


# ══════════════════ B: RESOLVED MARKER ══════════════════════════════════════

def test_the_eight_are_backfilled_with_commit_and_reason():
    r = ci.RESOLVED_RUNS.get("36843668865")
    assert r, "the 10-01 DID_NOT_RUN is not recorded as resolved"
    commit, reason = r
    assert commit == "a4d5228"
    assert "month-boundary" in reason and len(reason) > 40, (
        "the resolution carries no reason a future reader could check")


def test_CLR2_the_clearing_action_CHANGES_the_predicate():
    """Stated explicitly: recording the resolution is the action, and it moves
    the alarm. If the predicate ignored the marker, recording it would be
    theatre."""
    src = pathlib.Path("scripts/ci_audit.py").read_text(encoding="utf-8")
    block = src.split("_resolved = RESOLVED_RUNS.get", 1)[1][:400]
    assert "not _resolved" in block, (
        "RESOLVED_RUNS is read but does not gate `alarmed` — the clearing "
        "action does not change the predicate")


def test_the_resolution_is_RECORDED_not_silencing():
    """A resolved run still appears in the table, with its verdict."""
    src = pathlib.Path("scripts/ci_audit.py").read_text(encoding="utf-8")
    assert 'RESOLVED by' in src, "nothing prints the resolution"
    # the verdict is computed before the marker is consulted, so it is unchanged
    assert src.index("v = verdict(facts, hits, log)") < \
        src.index("_resolved = RESOLVED_RUNS.get"), (
        "the marker is consulted BEFORE the verdict — then it could change it, "
        "and a resolved run would stop carrying what happened")


def test_an_UNRESOLVED_run_still_alarms():
    """The positive control for the marker: it must not clear everything."""
    assert "99999999999" not in ci.RESOLVED_RUNS
    src = pathlib.Path("scripts/ci_audit.py").read_text(encoding="utf-8")
    assert "alarmed.append" in src


# ══════════════════ C: THIRD OUTCOME REGISTERED ═════════════════════════════

def test_the_apparatus_states_are_registered_and_exhaustive():
    assert APPARATUS_STATES == ("DID_NOT_START", "RUNNING_AND_FAILING",
                                "RUNNING_AND_FINE")


@pytest.mark.parametrize("in_window,raw,n,expected,alarms", [
    (False, 103, 0, "DID_NOT_START", False),
    (False, 0, 0, "DID_NOT_START", False),
    (False, 400, 12, "DID_NOT_START", False),
    (True, 103, 0, "RUNNING_AND_FAILING", True),
    (True, 400, 12, "RUNNING_AND_FINE", False),
])
def test_all_three_apparatus_states(in_window, raw, n, expected, alarms):
    a = apparatus_state(in_window=in_window, raw_rows=raw, n_fixtures=n)
    assert a.state == expected
    assert a.alarms is alarms


def test_a_correct_zero_OUTSIDE_the_window_does_not_alarm():
    """The 2026-10-02 case: 103 observations, 0 trajectories, window not open."""
    a = apparatus_state(in_window=False, raw_rows=103, n_fixtures=0)
    assert a.state == "DID_NOT_START" and a.alarms is False
    assert "HAS NOT STARTED" in a.message
    assert "not working" not in a.message, (
        "a correct zero is still being described as a broken apparatus")


def test_POSITIVE_CONTROL_the_alarm_STILL_fires_inside_the_window():
    """The only reason to trust the silence above."""
    a = apparatus_state(in_window=True, raw_rows=103, n_fixtures=0)
    assert a.state == "RUNNING_AND_FAILING"
    assert a.alarms is True, (
        "nothing alarms for a genuinely failing apparatus — the third state "
        "was added by deleting the detector")
    assert "not working" in a.message


def test_the_three_apparatus_states_PRINT_DIFFERENTLY():
    seen = {}
    for kw in (dict(in_window=False, raw_rows=103, n_fixtures=0),
               dict(in_window=True, raw_rows=103, n_fixtures=0),
               dict(in_window=True, raw_rows=400, n_fixtures=12)):
        a = apparatus_state(**kw)
        seen[a.state] = str(a)
    assert set(seen) == set(APPARATUS_STATES)
    assert len(set(seen.values())) == 3


def test_the_window_comes_from_ONE_definition():
    """`in_window` must derive from the reservation window, not a second
    calendar — a third date literal here is THE HABIT."""
    src = pathlib.Path("scripts/h1_collection_check.py").read_text(encoding="utf-8")
    assert "from src.data.odds_quota import h1_reservation" in src
    assert "h1_reservation() > 0" in src


# ══════════════════ D: CEILINGS COLLAPSED ═══════════════════════════════════

def test_there_is_ONE_credit_bound():
    """`CREDIT_CEILING` caps the reservation; it is not a parallel limb."""
    assert effective_bound(H1_RESERVED_CREDITS) == H1_RESERVED_CREDITS
    assert effective_bound(CREDIT_CEILING + 50) == CREDIT_CEILING, (
        "a reservation above the registered ceiling is honoured — the "
        "registration no longer caps the allocation")
    assert effective_bound(10) == 10


def test_CEILING_HIT_HAS_a_reachable_path_and_it_is_STATED():
    """The dead state is alive, by a named input: a reservation exceeding the
    registered ceiling, which is a misconfiguration the state names."""
    s = collection_stop(n_fixtures=0, credits_spent=CREDIT_CEILING,
                        raw_rows=9, reservation=CREDIT_CEILING + 50)
    assert s.state == "CEILING_HIT"
    assert "misconfiguration" in s.reason
    assert "EXCEEDS" in s.reason


def test_the_normal_arrangement_reports_the_RESERVATION():
    s = collection_stop(n_fixtures=0, credits_spent=H1_RESERVED_CREDITS,
                        raw_rows=9, reservation=H1_RESERVED_CREDITS)
    assert s.state == "RESERVATION_EXHAUSTED"
    assert str(CREDIT_CEILING) in s.reason, (
        "the registered ceiling is not mentioned, so a reader cannot tell it "
        "was never reached")


def test_an_UNREADABLE_ledger_is_its_OWN_state_not_a_ceiling_event():
    """The split. Folding this into CEILING_HIT reported a halt-for-ignorance
    as a halt-at-a-bound."""
    s = collection_stop(n_fixtures=5, credits_spent=None, raw_rows=9)
    assert s.state == "BUDGET_UNREADABLE"
    assert s.halt is True
    assert "NOT a bound being reached" in s.reason


def test_the_terminal_states_are_FIVE_and_all_DISTINGUISHABLE():
    """Four became five — a SPLIT, not an addition, and reported as such."""
    assert len(COLLECTION_TERMINAL_STATES) == 5
    assert COLLECTION_STATES[0] == "COLLECTING"
    cases = [
        dict(n_fixtures=TARGET_N, credits_spent=10, raw_rows=9,
             reservation=H1_RESERVED_CREDITS),
        dict(n_fixtures=0, credits_spent=H1_RESERVED_CREDITS, raw_rows=9,
             reservation=H1_RESERVED_CREDITS),
        dict(n_fixtures=0, credits_spent=CREDIT_CEILING, raw_rows=9,
             reservation=CREDIT_CEILING + 50),
        dict(n_fixtures=0, credits_spent=H1_RESERVED_CREDITS, raw_rows=0,
             reservation=H1_RESERVED_CREDITS),
        dict(n_fixtures=0, credits_spent=None, raw_rows=9,
             reservation=H1_RESERVED_CREDITS),
    ]
    lines = {}
    for kw in cases:
        s = collection_stop(**kw)
        lines[s.state] = str(s)
    assert set(lines) == set(COLLECTION_TERMINAL_STATES), set(lines)
    assert len(set(lines.values())) == 5, "two terminal states print the same"


# ══════════════════ E: ENVELOPE RE-ATTRIBUTED ═══════════════════════════════
#
# `11h21m` belongs to `37 9 * * *` (n=39) and never to `0 3 * * *`, whose
# measured maximum was 6h35m. A figure without its regime gets re-pooled within
# a month, and that has already happened twice: once citing the pooled envelope
# as this cron's, and once reading a 09:37 run as an 11h30m outlier of a 03:00
# series.

#: Live sites that cite a scheduler delay figure. The ledger is EXCLUDED on
#: purpose: its entries record what was believed when written, and rewriting
#: them would be deleting the record rather than correcting it.
_DELAY_CITING_FILES = (
    ".github/workflows/daily-picks.yml",
    ".github/workflows/ci-audit.yml",
    ".github/workflows/closing-lines.yml",
    "src/scrapers/theodds_scraper.py",
    "tests/test_picks_run_guard.py",
    "tests/test_schedule_margin.py",
)


def test_every_LIVE_citation_of_the_envelope_names_its_regime():
    """A delay figure must carry the cron it was measured under.

    The check is deliberately coarse — the regime marker must appear within a
    few lines of the figure — because the failure it prevents is a bare number
    being read as the current cron's envelope.
    """
    import re
    offenders = []
    for path in _DELAY_CITING_FILES:
        lines = pathlib.Path(path).read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            if not re.search(r"11h21m|0\.5-5\.7h|0\.5h to 11h", line):
                continue
            window = "\n".join(lines[max(0, i - 6):i + 7])
            if not re.search(r"37 9 \* \* \*|RETIRED|retired|pre-08-30|regime",
                             window):
                offenders.append(f"{path}:{i + 1}: {line.strip()[:90]}")
    assert not offenders, (
        "a delay figure is cited with no regime named nearby:\n  "
        + "\n  ".join(offenders))


def test_the_canonical_per_regime_table_exists_with_all_three_n():
    """One place carries the table, and the others point at it."""
    wf = pathlib.Path(".github/workflows/daily-picks.yml").read_text(encoding="utf-8")
    block = wf.split("THE ENVELOPE, RE-ATTRIBUTED", 1)
    assert len(block) == 2, "the canonical statement is gone"
    t = block[1][:2000]
    for token in ("37 9 * * *", "0 3 * * *", "0 0 * * *",
                  "39", "32", "11h21m", "6h35m", "4h15m", "headSha"):
        assert token in t, f"the per-regime table is missing {token!r}"


def test_the_three_series_are_never_POOLED_in_the_collector():
    src = pathlib.Path("scripts/ci_audit.py").read_text(encoding="utf-8")
    body = src.split("def collect_lag_series", 1)[1].split("\ndef ", 1)[0]
    assert "if by_sha[sha] != cron_now" in body
    assert "NOT comparable" in body


def test_n_equals_1_supports_no_distributional_claim():
    """The current cron has one observation. `sd` is 0 by construction and
    `mean+3sd` collapses onto it — so the margin is a reading, not a bound."""
    m = ci.schedule_margin([(dt.date(2026, 10, 2), 255.0)], 0)
    assert m["n"] == 1
    assert m["sd"] == 0.0
    assert m["mean_plus_3sd"] == m["max"] == 255.0, (
        "mean+3sd differs from max at n=1, so one of them is inventing spread")
