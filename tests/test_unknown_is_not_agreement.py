"""UNKNOWN IS NOT A MATCH. `[]` vs `None`, in the cache-validity mechanism.

`filter_generation()` digests the exclusion predicate's source so that a cache
built under a DIFFERENT predicate refuses itself. Until 2026-09-17 its failure
path returned the string ``"unknown"``.

    A CONSTANT COMPARES EQUAL TO ITSELF.

So every cache stamped while the source was unreadable carried the same
generation and validated against every other one. **The digest stopped
discriminating at exactly the moment validity became uncomputable, and reported
that as validity** — and announcing the failure at WARNING (Stage 25) did not
change that, because a caller comparing two strings still saw agreement.

    `"unknown" == "unknown"`  ->  served
    `None`                    ->  REBUILD

Both consumers share one generation, deliberately, so the two caches can never
disagree about what "excluded" means — which means both had the same hole and
both are fixed here:

    history_mirror._read_meta      the Parquet history mirror
    ml_models.load                 the trained-model pickles (x2 classes)

THE CHECK IS BEFORE THE EQUALITY, NOT AFTER. `None == None` is True, so a
generation of None on BOTH sides would otherwise read as a match — which is the
original defect wearing the new sentinel.
"""

import json

import pytest

import src.data.history_mirror as hm


def test_healthy_generation_is_a_digest_not_a_sentinel():
    got = hm.filter_generation()
    assert got is not None and isinstance(got, str) and len(got) == 12, got


def test_an_uncomputable_generation_is_None_not_a_constant(monkeypatch):
    """THE REGRESSION. A constant here is agreement between unrelated caches."""
    import inspect

    def _boom(*a, **k):
        raise OSError("source unavailable")

    monkeypatch.setattr(inspect, "getsource", _boom)
    got = hm.filter_generation()
    assert got is None, (
        f"filter_generation returned {got!r} when the predicate source was "
        f"unreadable. A constant compares equal to itself, so every cache "
        f"stamped with it validates against every other one.")


def _meta(tmp_path, stamped):
    m = hm.HistoryMirror(directory=tmp_path)
    m.dir.mkdir(parents=True, exist_ok=True)
    m.meta_path.write_text(json.dumps({
        "schema_version": hm.SCHEMA_VERSION,
        "filter_generation": stamped,
        "watermark": "2026-09-01T00:00:00",
        "row_count": 10,
    }), encoding="utf-8")
    return m


def test_the_mirror_refuses_when_the_CURRENT_generation_is_unknown(tmp_path, monkeypatch):
    m = _meta(tmp_path, "abcdef123456")
    monkeypatch.setattr(hm, "filter_generation", lambda: None)
    assert m._read_meta() == {}, (
        "the mirror was served while the current filter generation could not "
        "be computed — provenance is unestablished, which is not agreement")


def test_the_mirror_refuses_when_the_STAMP_is_unknown(tmp_path, monkeypatch):
    m = _meta(tmp_path, None)
    monkeypatch.setattr(hm, "filter_generation", lambda: "abcdef123456")
    assert m._read_meta() == {}


def test_None_on_BOTH_sides_is_STILL_a_refusal(tmp_path, monkeypatch):
    """`None == None` is True. That is the defect wearing the new sentinel.

    If the unknown check ever moves BELOW the equality test, this is the case
    that comes back — and it is the common one, because a source that is
    unreadable now was probably unreadable when the cache was written.
    """
    m = _meta(tmp_path, None)
    monkeypatch.setattr(hm, "filter_generation", lambda: None)
    assert m._read_meta() == {}, (
        "two unknowns compared equal and the cache was served — the check must "
        "precede the equality, not follow it")


def test_a_matching_generation_is_still_served(tmp_path, monkeypatch):
    """The refusal must not swallow the healthy case."""
    m = _meta(tmp_path, "abcdef123456")
    monkeypatch.setattr(hm, "filter_generation", lambda: "abcdef123456")
    assert m._read_meta().get("filter_generation") == "abcdef123456"


def test_both_consumers_share_one_generation():
    """A second opinion here would be two definitions of 'excluded'."""
    import src.models.ml_models as ml
    assert ml.training_filter_generation() == hm.filter_generation()


@pytest.mark.parametrize("cls_name", ["MLModels", "GoalsMLModel"])
def test_the_pickle_check_refuses_unknown_before_comparing(cls_name):
    """Both model classes carry the check, and both check before the equality."""
    import inspect
    import src.models.ml_models as ml

    src = inspect.getsource(getattr(ml, cls_name).load)
    assert "_want is None or _got is None" in src, (
        f"{cls_name}.load compares generations without refusing unknown first. "
        f"`None == None` restores a pickle whose training filter cannot be "
        f"established — the contaminated-weights case the check exists for.")
    assert src.index("_want is None or _got is None") < src.index("if _got != _want"), (
        f"{cls_name}.load tests equality BEFORE refusing unknown")
