"""Stage 13 (s5.3) — every cached derivative of excluded data must invalidate.

THE PRINCIPLE, which matters more than the inventory: an exclusion is only real
where every cache derived from the excluded data is invalidated. There were two
such caches — the Parquet history mirror and the ML pickles — and they are ONE
finding, not two. Anyone adding a third needs this sentence, not a list.

Both are restored by the daily workflow's `Restore database and models` step,
from the same artifact, so neither can be fixed by a one-time deletion: a stale
copy arrives from a previous run and is trusted on arrival. Both are therefore
stamped with the same filter generation and refuse themselves on mismatch.

The ML pickles are the subtler of the two, because they LOOK guarded:
`ml_retrain_days: 3` measures when a model was trained and says nothing about
what it was trained on, so a pickle full of contaminated matches reads as fresh.

The mirror is a cache of "completed matches", and what counts as one is now
narrower. A Parquet built before the exclusion filter existed holds rows the
current code must never see, and it does not announce itself: a cache that
answers looks exactly like a cache that answers correctly.

A one-time invalidation would not have been enough. The daily workflow restores
and saves this artifact, so a stale mirror can arrive from a previous run and be
trusted on arrival — local deletion never touches it. The stamp makes
invalidation automatic on every future change to the predicate.
"""

import json

import pytest

from src.data.history_mirror import HistoryMirror, filter_generation


def test_the_generation_is_derived_from_the_predicate_not_hand_maintained():
    """If it were a hand-maintained constant, it would rot like every other one.

    Deriving it from the source of `_base_filter` means changing the filter
    changes the stamp with no one remembering to do anything.
    """
    g = filter_generation()
    assert g and g != "unknown"
    assert len(g) == 12


def test_the_current_predicate_actually_contains_the_exclusion():
    """Guards against the stamp being computed over the wrong function."""
    import inspect

    from src.data.match_history import _HistoryCache

    src = inspect.getsource(_HistoryCache._base_filter)
    assert "training_exclusion_reason" in src, (
        "the generation stamp is derived from a predicate that does not "
        "mention the exclusion — the stamp would not change when it does")


def _write_meta(tmp_path, generation):
    mirror = HistoryMirror(directory=tmp_path)
    mirror.meta_path.parent.mkdir(parents=True, exist_ok=True)
    from src.data.history_mirror import SCHEMA_VERSION
    mirror.meta_path.write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "filter_generation": generation,
        "row_count": 12345,
        "watermark": "2026-08-01T00:00:00",
    }), encoding="utf-8")
    return mirror


def test_a_mirror_from_the_current_generation_is_accepted(tmp_path):
    mirror = _write_meta(tmp_path, filter_generation())
    meta = mirror._read_meta()
    assert meta.get("row_count") == 12345, (
        "a mirror built under the current filter was discarded")


def test_a_mirror_from_an_older_generation_is_refused(tmp_path):
    """The case that matters: stale artifact, current code.

    Verified the way the delete guard was — build the state, run against the
    new code, assert the fallback fired. `_read_meta` returning {} is what
    triggers a full resync from the database instead of trusting the Parquet.
    """
    mirror = _write_meta(tmp_path, "0000deadbeef")   # a previous filter
    meta = mirror._read_meta()
    assert meta == {}, (
        "a mirror built under a DIFFERENT exclusion filter was accepted — "
        "contaminated rows would be served to Poisson and Elo")


def test_a_mirror_with_no_generation_at_all_is_refused(tmp_path):
    """Every mirror predating this change has no stamp; none may be trusted."""
    mirror = HistoryMirror(directory=tmp_path)
    mirror.meta_path.parent.mkdir(parents=True, exist_ok=True)
    from src.data.history_mirror import SCHEMA_VERSION
    mirror.meta_path.write_text(json.dumps({
        "schema_version": SCHEMA_VERSION, "row_count": 999,
    }), encoding="utf-8")
    assert mirror._read_meta() == {}


def test_changing_the_predicate_changes_the_stamp(monkeypatch):
    """The property the whole mechanism rests on."""
    import inspect

    import src.data.history_mirror as hm

    before = hm.filter_generation()
    monkeypatch.setattr(
        inspect, "getsource",
        lambda _obj: "def _base_filter(): return (something_else,)")
    after = hm.filter_generation()
    assert before != after, (
        "the stamp did not move when the predicate changed — invalidation "
        "would not be automatic")


def test_the_predicate_is_closed_so_the_digest_is_complete():
    """The boundary of filter_generation(), pinned instead of assumed.

    The digest covers `_base_filter`'s source text. That is complete ONLY while
    the predicate references nothing outside itself — if it started using a
    module-level constant or a helper, changing that symbol would alter the
    filter without moving the digest, and a stale mirror would be accepted.

    `staticmethod` is the decorator, not a dependency.
    """
    import ast
    import inspect
    import textwrap

    from src.data.match_history import _HistoryCache

    tree = ast.parse(textwrap.dedent(
        inspect.getsource(_HistoryCache._base_filter)))
    referenced = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    referenced.discard("staticmethod")

    assert referenced <= {"Match"}, (
        "_base_filter now closes over " + repr(sorted(referenced - {"Match"}))
        + " — filter_generation() digests only this function's own source, so "
        "changing those symbols would alter the filter WITHOUT invalidating "
        "existing mirrors. Extend the digest to cover them; do not relax this.")


# ─────────────────────────────────────── the second cache: the ML artifacts

def _fitted_model(tmp_path, generation):
    """A saved model stamped with `generation`, written the way save() writes."""
    import numpy as np

    from src.models.ml_models import MLModels, _safe_save

    m = MLModels()
    state = {
        "models": {}, "calibrated_models": {}, "scaler": None,
        "feature_names": ["a", "b"], "is_fitted": True,
        "_kept_feature_mask": None, "_corr_drop_mask": None,
        "trained_at": "2026-08-22T10:00:00",     # two days old: "fresh"
        "training_filter_generation": generation,
    }
    tmp_path.mkdir(parents=True, exist_ok=True)
    _safe_save(state, tmp_path / "ml_models.pkl")
    return m


def test_a_model_from_the_current_generation_loads(tmp_path):
    from src.models.ml_models import training_filter_generation

    m = _fitted_model(tmp_path, training_filter_generation())
    m.load(str(tmp_path))
    assert m.is_fitted, "a model trained under the current filter was discarded"


def test_a_model_from_an_older_generation_is_refused_and_retrains(tmp_path):
    """The case the age check cannot see.

    `trained_at` is two days old, so `_ml_models_stale(max_age_days=3)` would
    call this fresh and skip --train. Only the generation stamp can tell that it
    was fitted on matches the current code excludes.

    Verified the way the mirror and the delete guard were: build the artifact,
    run it against current code, assert the fallback fired.
    """
    m = _fitted_model(tmp_path, "0000deadbeef")
    m.load(str(tmp_path))

    assert not m.is_fitted, (
        "a model trained under a DIFFERENT exclusion filter was loaded — the "
        "contaminated 29 would be back in the weights, and the age check would "
        "report the model as fresh")
    assert m.trained_at is None, (
        "trained_at survived — the staleness check would treat a refused "
        "artifact as recent and skip the retrain that must now happen")


def test_a_model_with_no_stamp_is_refused(tmp_path):
    """Every artifact predating s5.3 has no stamp; none may be trusted."""
    m = _fitted_model(tmp_path, None)
    m.load(str(tmp_path))
    assert not m.is_fitted


def test_both_caches_share_one_generation():
    """If they could disagree, one would be invalidated and the other not."""
    from src.data.history_mirror import filter_generation
    from src.models.ml_models import training_filter_generation

    assert training_filter_generation() == filter_generation()


# ─────────────────── the safety net that is load-bearing and looks redundant

def test_the_row_count_reconcile_still_consults_the_exclusion_filter():
    """A genuine cross-check between two independent paths. Keep it.

    HISTORY, because the reason changed. Until MIR-1 was fixed (Stage 14) this
    reconcile was the ONLY thing keeping excluded matches out of the Poisson
    fit: `_fetch_delta` re-admitted all 29 on every incremental sync and only
    the count disagreement (local=39236 db=39207) removed them again.

    That is no longer true. Membership is now decided once, in SQL, from
    `_base_filter()`, so the incremental path excludes them by construction.

    The reconcile stays anyway, and its value is now different and better: it is
    an independent cross-check. The incremental path maintains the mirror
    row-by-row; this counts the database directly. Two paths that must agree.
    Deletes still leave no trace in `updated_at`, so without it a deleted row
    would linger indefinitely.

    Worth knowing when judging whether to keep it: when MIR-1 was reproduced,
    this reconcile MASKED the outcome tests. "Is the excluded match gone from
    the mirror?" passed even against the broken code, because the reconcile
    removed it. Only the tests that measured the mechanism — the watermark and
    the source itself — failed. A safety net that repairs the symptom will hide
    the defect from any test that asks about outcomes.
    """
    import inspect

    from src.data.history_mirror import HistoryMirror

    src = inspect.getsource(HistoryMirror._completed_count)
    assert "training_exclusion_reason" in src, (
        "the row-count reconcile no longer excludes contaminated matches. It "
        "is the independent half of a two-path cross-check: the incremental "
        "sync maintains the mirror row-by-row, this counts the database. If "
        "they stop measuring the same population the check is worthless, and "
        "deletes — which leave no trace in updated_at — go unnoticed.")

    sync = inspect.getsource(HistoryMirror._sync_locked)
    assert "_completed_count" in sync, (
        "the sync no longer reconciles against a filtered count — see above")
