"""The integrity failure, made observable — 2026-10-12.

`INTEGRITY FAILURE` has been logged at ERROR since 2026-04-25 and read by
**nothing**: not `ci_audit`'s pattern set, not an alert, not a workflow. The
handler logs and returns, and the agent continues on Poisson + Elo. So the one
benefit of the HMAC that does **not** depend on the key being secret —
catching accidental corruption, a partial cache restore, or a key rotation —
was itself unobservable.

    An alarm with no wire to production is a definition, not an alarm.

MEASURED BEFORE WIRING, 2026-10-12, over the local audit log cache
(`ci_logs/`, 807 cached runs across all workflows):

    runs with a verified-path load   239      <- the POSITIVE marker
    total load events              1,400
    runs with no pickle present       13
    UNSIGNED LOADS (no .sig)           0
    INTEGRITY FAILURES                 0

The 239/1,400 line is what makes the two zeros mean something: the corpus
exercises the path. Without it, `0 unsigned loads` would be the same reading a
corpus containing no model loads at all would give — which is the vacuous
filter this project has now met three times.

THIS FILE USED A LOCAL COPY OF THE LOGURU SINK. It now uses the SHARED
`capture_loguru` fixture from `tests/conftest.py`, added 2026-10-12 together
with a collection-time refusal of `caplog` — because the first draft of this
file asserted on `caplog`, found it empty, and would have reported "the alert
does not fire" about an alert that fires. Three local copies of the sink
existed at that point and a fourth was being written. One rule, one definition.

THE LOAD IS DELIBERATELY STILL ALLOWED when a `.sig` is absent. Failing closed
is step 4 of the switch plan and it does not ship until the measured count has
held at zero over a window: an unsigned load that refuses would cost a card,
and a test gate has already cost one permanently.

NO FULL DIGEST, KEY OR PREIMAGE APPEARS HERE. `_mask` emits eight characters of
a sixty-four character digest. A digest is not a secret and masking it is not
what protects the key; the rule simply has no exception for values that look
harmless.
"""

import importlib
import pathlib
import pickle
import re

import pytest


@pytest.fixture()
def ml(monkeypatch):
    """The module, loaded with the opt-in so import does not raise."""
    monkeypatch.setenv("BETTING_AGENT_ALLOW_DEFAULT_HMAC", "1")
    monkeypatch.delenv("MODEL_HMAC_KEY", raising=False)
    import src.models.ml_models as m
    return importlib.reload(m)


# ── THE CONTROL ──────────────────────────────────────────────────────────────

def test_POSITIVE_CONTROL_corrupted_bytes_alert_with_the_right_content(
        ml, tmp_path, monkeypatch, capture_loguru):
    """Corrupt a saved model's bytes; the alert must fire and say what it is.

    Asserting on the ALERT rather than on the exception: the exception was
    always raised and always caught, which is precisely why nobody knew.
    """
    f = tmp_path / "ml_models.pkl"
    ml._safe_save({"hello": "world"}, f)
    assert f.with_suffix(".sig").exists(), "the control's premise failed"

    f.write_bytes(pickle.dumps({"hello": "tampered"}))
    monkeypatch.setenv("GITHUB_RUN_ID", "99887766")

    with capture_loguru("ERROR") as records:
        with pytest.raises(ml.IntegrityError):
            ml._safe_load(f)

    assert records, "no loguru record was captured - the sink is wrong"
    line = "\n".join(r["message"] for r in records)
    assert ml.INTEGRITY_FAILURE_MARKER in line, line
    assert "model=ml_models.pkl" in line
    assert "run=99887766" in line, "the run id is not in the alert"
    assert "expected=" in line and "actual=" in line
    assert "action=" in line, "CLR-2: no clearing action in the alert text"
    # MASKED: eight characters, and the two digests must differ or the
    # assertion would pass against a mask that discarded everything.
    masks = re.findall(r"(?:expected|actual)=([0-9a-f?]{4}\.\.[0-9a-f?]{4})",
                       line)
    assert len(masks) == 2, masks
    assert masks[0] != masks[1], (
        "the two masked digests are identical, so the mask is not carrying "
        "enough to compare them")


def test_NEGATIVE_CONTROL_an_intact_model_is_SILENT(ml, capture_loguru, tmp_path):
    """The other half. Without it the test above passes against a module that
    alerts on every load."""
    f = tmp_path / "ml_models.pkl"
    ml._safe_save({"hello": "world"}, f)
    with capture_loguru("WARNING") as records:
        state = ml._safe_load(f)
    assert state == {"hello": "world"}
    line = "\n".join(r["message"] for r in records)
    assert ml.INTEGRITY_FAILURE_MARKER not in line, line
    assert ml.SIGNATURE_MISSING_MARKER not in line, line


def test_a_MISSING_signature_alerts_and_STILL_LOADS(ml, capture_loguru, tmp_path):
    """Both halves of today's decision, in one place.

    It alerts — so the frequency becomes a measured number — and it loads, so
    nothing that works today stops working. Step 4 of the plan inverts the
    second half, and not before the first has been zero for a window.
    """
    f = tmp_path / "ml_models.pkl"
    f.write_bytes(pickle.dumps({"unsigned": True}))
    assert not f.with_suffix(".sig").exists()

    with capture_loguru("ERROR") as records:
        state = ml._safe_load(f)

    line = "\n".join(r["message"] for r in records)
    assert records, "no loguru record was captured - the sink is wrong"
    assert state == {"unsigned": True}, "the load was tightened in this stage"
    assert ml.SIGNATURE_MISSING_MARKER in line, line
    assert "state=loaded_UNVERIFIED" in line, (
        "the alert does not say the load was unverified, which is the whole "
        "distinction: absent is UNVERIFIABLE, not verified")
    assert "action=" in line


# ── THE WIRE ─────────────────────────────────────────────────────────────────

def test_the_audit_MATCHES_both_markers():
    """The wire itself. The markers existed for 170 days; the pattern set is
    what makes them reach a person."""
    import scripts.ci_audit as ci
    import src.models.ml_models as m

    for key, marker in (("model_integrity_failures",
                         m.INTEGRITY_FAILURE_MARKER),
                        ("model_unsigned_loads",
                         m.SIGNATURE_MISSING_MARKER)):
        assert key in ci.PATTERNS, f"{key} is not in the audit's pattern set"
        import re
        assert re.search(ci.PATTERNS[key], marker), (
            f"{key}'s pattern does not match the marker the emitter prints: "
            f"{ci.PATTERNS[key]!r} vs {marker!r}")


def test_both_markers_reach_the_ALARM_and_not_just_the_report():
    """`DEGRADED` is not in `--fail-on`, so a finding in `hits` prints and
    alerts nobody. These must be on the alarm path."""
    import inspect

    import scripts.ci_audit as ci

    import tests.conftest as conftest

    src = inspect.getsource(ci.main if hasattr(ci, "main") else ci)
    # ── COMMENT-STRIPPED BEFORE SPLITTING. 2026-10-21. ────────────────────
    # The anchor `facts.get("<key>")` now also appears in a COMMENT in the
    # count-style key list -- the comment that records these two alarms were
    # incapable of firing until that list was extended. Splitting on the raw
    # source landed the window inside that prose. Same defect this suite has
    # hit repeatedly; `conftest.code_of` tokenises and drops comments.
    whole = conftest.code_of(inspect.getsource(ci))
    for key in ("model_integrity_failures", "model_unsigned_loads"):
        block = whole.split(f'facts.get("{key}")', 1)
        assert len(block) == 2, f"{key} is never read from facts"
        assert "alarmed.append" in block[1][:400], (
            f"{key} is read but does not reach `alarmed` within the branch")
    assert "CLEARING ACTION" in inspect.getsource(ci)


def test_the_markers_are_NOT_registered_error_annotations():
    """THE TRAP THIS AVOIDED, pinned.

    `::error::` was the obvious wiring. `ci_audit` computes its absorbed-
    failure signal as a REMAINDER:

        unexplained_nonzero_exit = exit_annotations - min(self_emitted, exits)

    Every entry in `SELF_EMITTED_ANNOTATIONS` holds because its annotation is
    followed by that step exiting non-zero, so each explains at most one exit.
    An integrity failure does NOT exit non-zero — the handler returns and the
    agent carries on. Registering it would let `min` absorb a GENUINE
    unexplained exit elsewhere in the same log, blinding the detector in
    exactly the way it was rebuilt to stop.
    """
    import scripts.ci_audit as ci
    import src.models.ml_models as m

    registered = {pat for pat, _src in ci.SELF_EMITTED_ANNOTATIONS}
    for marker in (m.INTEGRITY_FAILURE_MARKER, m.SIGNATURE_MISSING_MARKER):
        assert not any(marker in pat for pat in registered), (
            f"{marker} was registered as a self-emitted annotation; it does "
            f"not exit non-zero, so it would absorb an unexplained exit")
    # COMMENT-BLIND. The emitter's own comment NAMES `::error::` to explain
    # why it does not print one, so a raw substring scan trips on the
    # documentation of the very decision it is checking.
    emitter = pathlib.Path("src/models/ml_models.py").read_text(
        encoding="utf-8")
    code = chr(10).join(l.split("#", 1)[0] for l in emitter.splitlines())
    assert "::error::" not in code, (
        "ml_models.py emits a GitHub error annotation. It must not: see above")


def test_NO_FULL_DIGEST_can_reach_a_log(ml, capture_loguru):
    """Masked form only, and the mask is checked rather than trusted."""
    full = "a" * 64
    masked = ml._mask(full)
    assert len(masked) == 10, masked
    assert full not in masked
    assert ml._mask("") == "??..??"
    assert ml._mask("short") == "??..??", "a short input leaks itself"


# ── THE MEASUREMENT THAT SIZES STEP 4 ────────────────────────────────────────

#: `measured` 2026-10-12 over `ci_logs/` — the audit's own cache, 807 runs.
#: Recorded here because step 4 of the switch plan (a missing `.sig` fails
#: closed) does not ship until this has held at zero over a window, and a
#: number with no provenance cannot gate anything.
UNSIGNED_LOAD_CENSUS = {
    "cached_runs": 807,
    "runs_with_a_load": 239,
    "load_events": 1400,
    "runs_with_no_pickle": 13,
    "unsigned_loads": 0,
    "integrity_failures": 0,
    "measured": "2026-10-12",
    "window": "the local audit log cache; GitHub retains run logs ~90 days, "
              "so the 324 daily-picks runs since 2026-02-15 CANNOT all be "
              "measured and this is not a census of them",
}


def test_the_census_is_recorded_with_its_LIMITS():
    """Hard rule: no number without provenance, and a window that is not the
    full population says so."""
    c = UNSIGNED_LOAD_CENSUS
    assert c["unsigned_loads"] == 0 and c["integrity_failures"] == 0
    assert c["runs_with_a_load"] > 0, (
        "the positive marker is zero, so the two zeros above are vacuous")
    assert "CANNOT all be measured" in c["window"]
    assert c["measured"] == "2026-10-12"
