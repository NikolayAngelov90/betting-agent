"""A detector on the MODEL, not on the retrain step — 2026-10-22.

THE GAP IT CLOSES. On 2026-10-22 the "ML training step killed by CI timeout"
Telegram alert was removed as run status, and the removal was recorded with
its hole named out loud:

  * the retrain step carries `continue-on-error: true`, so a failed retrain
    does not redden the run and GitHub notifies nobody;
  * `_ml_models_stale()` repairs a SINGLE failure on the next run;
  * a PERSISTENT failure had no detector at all -- staleness was
    `logger.info`-only, and the ensemble degrades quietly to Poisson + Elo.

MEASURED 2026-10-22, and the two answers differ, which is the point of
checking the artefact rather than the step:

    CI (the cache production uses)   trained_at 2026-10-21T09:07  ->  1 day
    this working copy                trained_at 2026-03-31T07:35  -> 205 days

CI is healthy. So this detector ships as a GUARD, not as a repair, and the
205-day local copy is what it was driven against.

THE THRESHOLD IS DERIVED. `ceil(ml_retrain_days + p99 scheduling gap)` =
`ceil(3 + 1.95)` = 5 days, where the gap is `measured` 2026-10-22 over n=183
gaps between consecutive SCHEDULED daily-picks runs. p95 (1.04) and p99 (1.95)
both ceil to 5, so the figure does not turn on which quantile is chosen.
"""

import datetime as dt
import json
import pathlib
import pickle

import pytest

import scripts.model_freshness as mf

NOW = dt.datetime(2026, 10, 22, 12, 0, tzinfo=dt.timezone.utc)


def _write(models_dir, name, trained_at):
    p = models_dir / name
    p.write_bytes(pickle.dumps({"trained_at": trained_at, "junk": 1}))
    return p


@pytest.fixture
def models(tmp_path, monkeypatch):
    d = tmp_path / "models"
    d.mkdir()
    monkeypatch.setattr(mf, "MODELS_DIR", d)
    monkeypatch.setattr(mf, "SIDECAR", d / "retrain_provenance.json")
    return d


# ── the threshold is derived, not chosen ──────────────────────────────────

def test_the_THRESHOLD_is_DERIVED_from_the_measured_cadence():
    assert mf.SCHEDULING_GAP_P99_DAYS == 1.95
    assert mf.stale_threshold_days(3) == 5
    assert mf.stale_threshold_days(7) == 9, (
        "the threshold must follow ml_retrain_days; a constant 5 would stop "
        "being derived the moment the cadence changed")


def test_the_THRESHOLD_does_not_turn_on_the_QUANTILE_CHOICE():
    """p95 1.04 and p99 1.95 both ceil to 5. Worth pinning, because the
    quantile-versus-mean+3sd question is still OPEN elsewhere in this project
    and this is the one place it happens not to matter."""
    import math
    assert math.ceil(3 + 1.04) == math.ceil(3 + 1.95) == 5


def test_the_GAP_carries_its_PROVENANCE_with_n_and_a_date():
    p = mf.SCHEDULING_GAP_PROVENANCE
    assert "n=183" in p and "2026-10-22" in p
    assert "p99 1.95" in p and "9.18" in p
    assert "outage" in p, (
        "the excluded p100 must be excluded for a stated reason, or the "
        "threshold looks like it was fitted to miss it")


# ── CONTROL 1: a successful retrain is SILENT ─────────────────────────────

def test_CONTROL_a_FRESH_model_is_SILENT_and_exits_0(models, capsys):
    for name in mf.ARTEFACTS:
        _write(models, name, (NOW - dt.timedelta(days=1)).isoformat())
    rows = mf.read_artefacts(models)
    a = mf.assess(rows, mf.record_provenance(rows, "run-1"), NOW, 3)
    assert a["state"] == "FRESH"
    assert a["findings"] == []
    assert a["oldest_age_days"] == pytest.approx(1.0, abs=0.01)


@pytest.mark.parametrize("age_days", [0.1, 1.0, 3.0, 4.9])
def test_CONTROL_every_age_WITHIN_the_threshold_is_silent(models, age_days):
    for name in mf.ARTEFACTS:
        _write(models, name, (NOW - dt.timedelta(days=age_days)).isoformat())
    rows = mf.read_artefacts(models)
    a = mf.assess(rows, {}, NOW, 3)
    assert a["state"] == "FRESH", (age_days, a["findings"])


# ── CONTROL 2: a skipped or failed retrain ALARMS, each forced ────────────

def test_CONTROL_a_FAILED_retrain_leaves_an_OLD_stamp_and_ALARMS(models):
    """A retrain that fails writes nothing, so the OLD timestamp survives.
    Forced by writing one past the threshold -- which is exactly what the
    205-day working copy is."""
    for name in mf.ARTEFACTS:
        _write(models, name, (NOW - dt.timedelta(days=205.1)).isoformat())
    rows = mf.read_artefacts(models)
    a = mf.assess(rows, {}, NOW, 3)
    assert a["state"] == "STALE"
    assert len(a["findings"]) == 2
    assert "205.1 days old" in a["findings"][0]
    assert "threshold of 5" in a["findings"][0]


def test_CONTROL_a_SKIPPED_retrain_that_never_wrote_a_model_ALARMS(models):
    """`skipped` is the third state and it is the one that occurred on
    2026-09-20. An ABSENT artefact is not an old one -- no model was ever
    written here -- and it must not read as age zero."""
    rows = mf.read_artefacts(models)          # empty directory
    a = mf.assess(rows, {}, NOW, 3)
    assert a["state"] == "UNKNOWN", a
    assert len(a["findings"]) == 2
    assert all("ABSENT" in f for f in a["findings"])
    assert a["oldest_age_days"] is None, (
        "an absent artefact reported an age; absent is not zero")


def test_the_BOUNDARY_is_STRICTLY_PAST_the_threshold(models):
    for name in mf.ARTEFACTS:
        _write(models, name, (NOW - dt.timedelta(days=5.0)).isoformat())
    assert mf.assess(mf.read_artefacts(models), {}, NOW, 3)["state"] == "FRESH"
    for name in mf.ARTEFACTS:
        _write(models, name, (NOW - dt.timedelta(days=5.01)).isoformat())
    assert mf.assess(mf.read_artefacts(models), {}, NOW, 3)["state"] == "STALE"


# ── the third state, three ways ───────────────────────────────────────────

def test_a_PARTIAL_write_is_caught_which_a_step_outcome_cannot_be(models):
    """One artefact fresh, the other stale. The retrain step exits 0 and the
    run is green; only an outcome check sees it."""
    _write(models, "ml_models.pkl", (NOW - dt.timedelta(days=1)).isoformat())
    _write(models, "goals_model.pkl", (NOW - dt.timedelta(days=90)).isoformat())
    a = mf.assess(mf.read_artefacts(models), {}, NOW, 3)
    assert a["state"] == "STALE"
    assert len(a["findings"]) == 1
    assert "goals_model.pkl" in a["findings"][0]


def test_an_UNREADABLE_artefact_is_UNKNOWN_not_FRESH(models):
    _write(models, "ml_models.pkl", (NOW - dt.timedelta(days=1)).isoformat())
    (models / "goals_model.pkl").write_bytes(b"not a pickle at all")
    a = mf.assess(mf.read_artefacts(models), {}, NOW, 3)
    assert a["state"] == "UNKNOWN"
    assert "UNREADABLE" in a["findings"][0]
    assert "UNKNOWN" in a["findings"][0]


def test_a_MISSING_trained_at_is_UNKNOWN_not_ZERO(models):
    (models / "ml_models.pkl").write_bytes(pickle.dumps({"junk": 1}))
    _write(models, "goals_model.pkl", (NOW - dt.timedelta(days=1)).isoformat())
    a = mf.assess(mf.read_artefacts(models), {}, NOW, 3)
    assert a["state"] == "UNKNOWN"
    assert "NO trained_at" in a["findings"][0]


def test_an_UNPARSEABLE_trained_at_is_UNKNOWN(models):
    _write(models, "ml_models.pkl", "the thirty-first of never")
    _write(models, "goals_model.pkl", (NOW - dt.timedelta(days=1)).isoformat())
    a = mf.assess(mf.read_artefacts(models), {}, NOW, 3)
    assert a["state"] == "UNKNOWN"
    assert "unparseable" in a["findings"][0]


# ── "by which run" ────────────────────────────────────────────────────────

def test_the_WRITING_RUN_is_recorded_and_SURVIVES_a_later_run(models):
    """The pickle does not record its writing run and adding a key would
    change the SIGNED bytes for a reporting nicety, so the run is recorded
    beside the artefact. The check runs in the same job as `--train`, so
    "first observed" is the writing run whenever the stamp moved."""
    ta = (NOW - dt.timedelta(days=1)).isoformat()
    for name in mf.ARTEFACTS:
        _write(models, name, ta)
    known = mf.record_provenance(mf.read_artefacts(models), "run-AAA")
    assert known["ml_models.pkl"]["first_seen_run"] == "run-AAA"

    # a LATER run that retrains nothing must not claim authorship
    known2 = mf.record_provenance(mf.read_artefacts(models), "run-BBB")
    assert known2["ml_models.pkl"]["first_seen_run"] == "run-AAA", (
        "a run that observed an unchanged model took credit for writing it")

    # a NEW stamp belongs to the run that first saw it
    for name in mf.ARTEFACTS:
        _write(models, name, NOW.isoformat())
    known3 = mf.record_provenance(mf.read_artefacts(models), "run-CCC")
    assert known3["ml_models.pkl"]["first_seen_run"] == "run-CCC"
    assert json.loads(mf.SIDECAR.read_text(encoding="utf-8")) == known3


def test_a_MISSING_run_id_is_said_out_loud_not_left_blank(models):
    _write(models, "ml_models.pkl", (NOW - dt.timedelta(days=1)).isoformat())
    known = mf.record_provenance(mf.read_artefacts(models), "")
    assert "UNKNOWN" in known["ml_models.pkl"]["first_seen_run"]


def test_the_ALARM_NAMES_the_run_that_wrote_the_stale_model(models):
    ta = (NOW - dt.timedelta(days=90)).isoformat()
    for name in mf.ARTEFACTS:
        _write(models, name, ta)
    known = mf.record_provenance(mf.read_artefacts(models), "run-OLD")
    a = mf.assess(mf.read_artefacts(models), known, NOW, 3)
    assert "run-OLD" in a["findings"][0], a["findings"][0]


# ── the message is CONTENT, and it cannot become background ──────────────

def test_the_BODY_reports_the_SYSTEM_not_a_RUN(models):
    for name in mf.ARTEFACTS:
        _write(models, name, (NOW - dt.timedelta(days=205.1)).isoformat())
    a = mf.assess(mf.read_artefacts(models), {}, NOW, 3)
    body = mf.format_alert(a, 3)
    for banned in ("workflow FAILED", "step(s) FAILED", "exited",
                   "conclusion", "run #"):
        assert banned not in body, f"the body reports run status: {banned!r}"
    assert "HAS NOT BEEN RETRAINED" in body
    assert "205.1 days" in body
    assert "NOTHING LOOKS BROKEN" in body, (
        "the body must say why this needs reporting at all: the ensemble "
        "still produces picks from Poisson + Elo")
    assert "CLEARING ACTION" in body and "--train --force" in body
    assert "ceil(ml_retrain_days=3 + 1.95" in body, (
        "the body must show the threshold's derivation, not just its value")


def test_the_BODY_CANNOT_BECOME_A_REPEATED_IDENTICAL_ALARM(models):
    """`pipeline_loss_alert` exists because six identical messages became
    background. This one carries the AGE, which increases every day it stays
    broken, so consecutive messages differ by construction."""
    bodies = []
    for age in (6.0, 7.0, 8.0):
        for name in mf.ARTEFACTS:
            _write(models, name, (NOW - dt.timedelta(days=age)).isoformat())
        a = mf.assess(mf.read_artefacts(models), {}, NOW, 3)
        bodies.append(mf.format_alert(a, 3))
    assert len(set(bodies)) == 3, "two days of staleness render identically"


def test_the_BODY_is_ASCII_so_PRINTING_IT_cannot_KILL_THE_DETECTOR():
    """It opened with an emoji and `print(body)` raised UnicodeEncodeError on
    a cp1251 console. A detector that dies on its own output is worse than no
    detector."""
    src = pathlib.Path("scripts/model_freshness.py").read_text(
        encoding="utf-8")
    assert "⚠" not in src and "❓" not in src
    for name in mf.ARTEFACTS:
        pass
    body = mf.format_alert(
        {"state": "STALE", "threshold_days": 5, "oldest_age_days": 205.1,
         "findings": ["x"], "lines": ["  line"]}, 3)
    body.encode("cp1251")          # raises if a non-encodable char is back


# ── the exit code, and the wiring ────────────────────────────────────────

def test_the_EXIT_CODE_is_0_FRESH_and_1_FINDING(models, monkeypatch):
    monkeypatch.setenv("GITHUB_RUN_ID", "run-Z")
    now = dt.datetime.now(dt.timezone.utc)
    for name in mf.ARTEFACTS:
        _write(models, name, (now - dt.timedelta(days=1)).isoformat())
    assert mf.main(["--quiet"]) == 0
    for name in mf.ARTEFACTS:
        _write(models, name, (now - dt.timedelta(days=99)).isoformat())
    assert mf.main(["--quiet"]) == 1


def test_it_is_WIRED_and_does_NOT_key_on_the_RETRAIN_STEPS_OUTCOME():
    """The whole point. Keying on `steps.retrain_ml` would rebuild the sender
    it replaces."""
    import yaml

    wf = yaml.safe_load(pathlib.Path(".github/workflows/daily-picks.yml")
                        .read_text(encoding="utf-8"))
    steps = wf["jobs"]["daily-picks"]["steps"]
    step = next((s for s in steps
                 if "scripts.model_freshness" in str(s.get("run", ""))), None)
    assert step is not None, "the detector is not wired into daily-picks"
    cond = str(step.get("if", ""))
    assert "always()" in cond
    assert "retrain_ml" not in cond, (
        "it keys on the retrain STEP again, which is the thing it replaces")
    assert step.get("continue-on-error") is True, (
        "a stale model would redden the run; DEL-2 reserves this workflow's "
        "red for 'the day produced no picks', and a stale model still "
        "produced picks")
    env = step.get("env") or {}
    for key in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"):
        assert key in env, f"{key} is not available, so it cannot deliver"


def test_the_SIDECAR_is_GITIGNORED_like_its_siblings():
    """Generated, cache-carried, never a source. Every other artefact in
    `data/models/` is listed explicitly in `.gitignore`."""
    ignore = pathlib.Path(".gitignore").read_text(encoding="utf-8")
    assert "data/models/retrain_provenance.json" in ignore
