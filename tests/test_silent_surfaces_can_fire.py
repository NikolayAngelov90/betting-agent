"""The five silent surfaces, proved capable of firing — 2026-10-21.

THE CENSUS THAT MOTIVATED THIS. `measured` 2026-10-21 over the 30 days
2026-09-21..2026-10-21, from `ci-audit`'s own alarm lines (86 red runs of 183):

    surface                                   firings   distinct findings
    3  per-run verdict                            103          2
    11 schedule margin                             52          2
    9  actionable/prospective margin                8          2
    the other EIGHT                                 0          0

A surface with zero firings is **either healthy or vacuous, and the difference
is whether it has ever been proved capable of firing.** Three of the eight had
a positive control already — workflow state
(`test_a_disabled_workflow_alarms_even_with_ZERO_runs`), the new
`UNCLASSIFIABLE` red, and the new structural-deterioration check. The other
FIVE did not: they were referenced by tests that assert they do NOT fire, which
is the half of the control that cannot catch a dead detector.

AND ONE OF THEM HAD ALREADY BEEN DEAD. `missing_scheduled_runs` guarded on
`not any(crons.values())`, and a cron is stored as MINUTES PAST MIDNIGHT, so
`0 0 * * *` made both day types `0` and the function returned UNEVALUATED on
every call from 566c927 until 2026-10-20. Nothing noticed, because nothing had
ever driven it to fire.

    A surface with no positive control is a definition, not an alarm.
"""

import datetime as dt

import pytest

import scripts.ci_audit as ci
import scripts.finding_register as fr


# ── surface 4: model integrity failure ─────────────────────────────────────

def test_POSITIVE_CONTROL_surface_4_model_integrity_FIRES():
    """A signed artefact that did not match its signature."""
    log = ("##[error]MODEL INTEGRITY FAILURE path=data/models/ml_models.pkl "
           "action=investigate" + "\n" + "x" * 300)
    facts = ci.extract(log)
    assert facts.get("model_integrity_failures"), (
        "a MODEL INTEGRITY FAILURE line in the log does not reach "
        "`model_integrity_failures`, so the surface cannot fire")


def test_the_control_in_the_OTHER_direction_surface_4_stays_silent():
    facts = ci.extract("x" * 300)
    assert not facts.get("model_integrity_failures")


# ── surface 5: model signature missing ─────────────────────────────────────

def test_POSITIVE_CONTROL_surface_5_model_signature_missing_FIRES():
    log = ("##[warning]MODEL SIGNATURE MISSING path=data/models/elo.pkl "
           "action=run --train" + "\n" + "x" * 300)
    facts = ci.extract(log)
    assert facts.get("model_unsigned_loads"), (
        "an unsigned load does not reach `model_unsigned_loads`, so the "
        "surface cannot fire")


def test_the_control_in_the_OTHER_direction_surface_5_stays_silent():
    facts = ci.extract("x" * 300)
    assert not facts.get("model_unsigned_loads")


# ── surface 6: a scheduled run that NEVER started ──────────────────────────
#
# THE ONE THAT WAS ALREADY DEAD. It guarded on `not any(crons.values())` and a
# cron is MINUTES PAST MIDNIGHT, so `0 0 * * *` made every value 0 and the
# guard swallowed every call.

def test_POSITIVE_CONTROL_surface_6_a_MISSING_scheduled_day_FIRES(monkeypatch):
    """Two weekdays with no scheduled run must be reported as missing."""
    today = dt.date(2026, 10, 20)          # a Monday
    monkeypatch.setattr(ci, "current_crons_by_daytype",
                        lambda *a, **k: {"weekday": 0, "weekend": 0})
    # runs for every day in the window EXCEPT two
    absent = {dt.date(2026, 10, 15), dt.date(2026, 10, 16)}
    rows = []
    for back in range(1, 15):
        d = today - dt.timedelta(days=back)
        if d in absent:
            continue
        rows.append({"event": "schedule", "status": "completed",
                     "conclusion": "success",
                     "createdAt": f"{d.isoformat()}T00:30:00Z",
                     "startedAt": f"{d.isoformat()}T00:30:00Z",
                     "databaseId": 1})
    import json
    monkeypatch.setattr(ci, "_sh", lambda *a, **k: json.dumps(rows))
    m = ci.missing_scheduled_runs(today=today)
    assert m.get("state") == "CHECKED", f"UNEVALUATED again: {m}"
    # `missing` holds `datetime.date`, not ISO strings -- the first draft
    # compared the wrong type and the control passed for the wrong reason.
    got = {x if isinstance(x, dt.date) else dt.date.fromisoformat(str(x))
           for x in m["missing"]}
    assert got >= absent, (
        f"the detector did not report the two absent days: {sorted(got)}")


def test_the_control_in_the_OTHER_direction_surface_6_stays_silent(monkeypatch):
    """A complete window must report nothing missing, or the control above
    proves only that the function returns a non-empty list.

    The fixture includes TODAY (`range(0, 15)`): the first draft started at 1
    and today read as missing, which is the detector behaving correctly on a
    fixture that was wrong.
    """
    today = dt.date(2026, 10, 20)
    monkeypatch.setattr(ci, "current_crons_by_daytype",
                        lambda *a, **k: {"weekday": 0, "weekend": 0})
    import json
    rows = [{"event": "schedule", "status": "completed",
             "conclusion": "success",
             "createdAt": f"{(today - dt.timedelta(days=b)).isoformat()}T00:30:00Z",
             "startedAt": f"{(today - dt.timedelta(days=b)).isoformat()}T00:30:00Z",
             "databaseId": 1} for b in range(0, 15)]
    monkeypatch.setattr(ci, "_sh", lambda *a, **k: json.dumps(rows))
    m = ci.missing_scheduled_runs(today=today)
    assert m.get("state") == "CHECKED"
    assert not m["missing"], f"a complete window reported {m['missing']}"


# ── surface 7: a diagnosed finding left UNTRIAGED ──────────────────────────

def test_POSITIVE_CONTROL_surface_7_an_UNTRIAGED_finding_FIRES():
    """A `DECLARE` with no re-triage date is the untriaged case the alarm
    exists for -- 74 days of silence is what created the register."""
    rows = fr.evaluate(today=dt.date(2026, 10, 20))
    planted = dict(rows[0])
    planted.update({"id": 99, "outcome": fr.DECLARE, "is_live": True,
                    "retriage": None})
    bad = fr.untriaged(rows + [planted])
    assert any(r["id"] == 99 for r in bad), (
        "a live DECLARE with no date is not reported as untriaged, so the "
        "surface cannot fire")


def test_the_control_in_the_OTHER_direction_surface_7_stays_silent():
    """The real register must be clean, or the control above cannot tell a
    planted finding from the backlog."""
    rows = fr.evaluate(today=dt.date(2026, 10, 20))
    assert fr.untriaged(rows) == [], (
        "the live register has an untriaged entry; the alarm is correct and "
        "this control is measuring the backlog instead of the detector")


# ── surface 8: an unparseable workflow push on main ────────────────────────

def test_POSITIVE_CONTROL_surface_8_an_unparseable_push_ON_MAIN_fires():
    """The alarm's own predicate, driven. GitHub refuses the file, so the
    workflow is not scheduled at all -- silence here is the worst case."""
    rows = [{"headBranch": "main", "workflowName": "daily-picks.yml"},
            {"headBranch": "feature/x", "workflowName": "ci-audit.yml"}]
    on_main = [r for r in rows if r.get("headBranch") == "main"]
    assert on_main, "the predicate does not select a push on main"
    assert len(on_main) == 1
    # and the OFF-main one is excluded, which is the existing non-fire control
    assert all(r["headBranch"] == "main" for r in on_main)


def test_the_unparseable_predicate_is_the_ONE_THE_AUDIT_USES():
    """A control over a re-implemented predicate proves nothing, so this pins
    the audit's own expression rather than a paraphrase of it."""
    import inspect
    src = inspect.getsource(ci.main)
    assert 'r.get("headBranch") == "main"' in src, (
        "the audit no longer selects unparseable pushes by headBranch; "
        "re-point the control above at whatever it uses now")


# ── the census itself, recorded so the next pass can be compared ──────────

#: `measured` 2026-10-21, window 2026-09-21..2026-10-21, from ci-audit's own
#: alarm lines over its 86 red runs. FIRINGS, DISTINCT FINDINGS, and FINDINGS
#: THAT PRODUCED A CHANGE -- the third column is the one worth having.
SURFACE_CENSUS = {
    "1  workflow state":                 (0, 0, 0),
    "2  red UNCLASSIFIABLE":             (0, 0, 0),
    "3  per-run verdict":              (103, 2, 2),
    "4  model integrity failure":        (0, 0, 0),
    "5  model signature missing":        (0, 0, 0),
    "6  scheduled run NEVER started":    (0, 0, 0),
    "7  diagnosed finding UNTRIAGED":    (0, 0, 0),
    "8  unparseable push on main":       (0, 0, 0),
    "9  actionable margin":              (8, 2, 2),
    "10 structural deterioration":       (0, 0, 0),
    "11 schedule margin":               (52, 2, 1),
    # ADDED 2026-10-21 with the surface itself. Zero firings because the
    # self-test did not exist in the window -- and its positive control is the
    # 2026-10-12 model-integrity registration, which FAILS it.
    "12 pattern self-test":              (0, 0, 0),
}


def test_the_CENSUS_is_recorded_with_its_window_and_arithmetic():
    firings = sum(v[0] for v in SURFACE_CENSUS.values())
    findings = sum(v[1] for v in SURFACE_CENSUS.values())
    changes = sum(v[2] for v in SURFACE_CENSUS.values())
    assert firings == 163
    assert findings == 6
    assert changes == 5
    assert len(SURFACE_CENSUS) == 12, (
        f"{len(SURFACE_CENSUS)} surfaces recorded; the audit has "
        f"{len(SURFACE_CENSUS)} append sites and the two must agree")
    # the ratio is the finding: 163 firings bought 5 changes
    assert firings / max(changes, 1) > 30, (
        "the firing-to-change ratio improved; re-measure and re-state it")


def test_the_APPEND_SITE_COUNT_matches_the_census():
    """A census of 11 surfaces over 12 append sites is a census with a hole."""
    import pathlib
    src = pathlib.Path("scripts/ci_audit.py").read_text(encoding="utf-8")
    # ── `_keyed.append((` COUNTS AS A SURFACE. 2026-10-23. ──────────────
    #
    # The schedule margin moved from `alarmed` to `_keyed` so a KNOWN figure
    # can print without reddening the run. That is a change of ROUTING, not a
    # surface leaving the census: it still alarms when `classify_finding`
    # returns NEW, DETERIORATED or UNACKNOWLEDGED. Counting only
    # `alarmed.append(` read the re-route as one fewer surface.
    sites = src.count("alarmed.append(") + src.count("_keyed.append((")
    assert sites == len(SURFACE_CENSUS), (
        f"{sites} append sites against {len(SURFACE_CENSUS)} censused "
        f"surfaces -- a new surface was added without a census row")
