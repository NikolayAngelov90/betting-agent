"""Regression tests for MLModels.cross_val_report (Stage 3, Phase 2 Bug 1).

The diagnostic this replaces read `ml_models._models` (the attribute is `models`),
so it raised AttributeError on every training run and the surrounding `except`
swallowed it — it had never once produced output. These tests prove the path
executes and that its split is genuinely forward-chained.
"""

import numpy as np
import pytest

from src.models.ml_models import MLModels


def _synthetic(n=240, seed=0):
    """A separable-ish 3-class problem so the report has all classes populated."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    logits = np.column_stack([
        X[:, 0] * 1.5,
        X[:, 1] * 0.5,
        X[:, 2] * 1.5,
    ])
    y = np.argmax(logits + rng.normal(scale=0.5, size=logits.shape), axis=1)
    return X, y.astype(int)


@pytest.fixture(scope="module")
def fitted():
    X, y = _synthetic()
    m = MLModels()
    m.fit(X, y, [f"f{i}" for i in range(X.shape[1])])
    return m, X, y


def test_cross_val_report_actually_runs(fitted):
    """The bug regression: this must return a populated report, not None and not
    raise. Before the fix the equivalent code path never executed at all."""
    model, X, y = fitted
    report = model.cross_val_report(X, y, [f"f{i}" for i in range(X.shape[1])])
    assert report is not None, "cross_val_report returned None on valid input"
    assert "accuracy" in report
    assert 0.0 <= report["accuracy"] <= 1.0
    # At least one class row present with the usual metric keys.
    class_rows = [k for k in report if k in ("Away", "Draw", "Home")]
    assert class_rows, f"no per-class rows in {sorted(report)}"
    for k in class_rows:
        assert {"precision", "recall", "f1-score", "support"} <= set(report[k])


def test_models_attribute_name_is_models_not_underscore_models():
    """Guards the exact typo: `ml_models._models` must stay a non-attribute."""
    m = MLModels()
    assert hasattr(m, "models")
    assert not hasattr(m, "_models")


def test_report_is_forward_chained_not_kfold(fitted, monkeypatch):
    """Every fold's training indices must precede its test indices.

    The old implementation used cross_val_predict(cv=3) — plain KFold with
    shuffle=False — so folds 2 and 3 trained on rows chronologically after
    fold 1's test rows while being labelled a "leak-free diagnostic".
    """
    model, X, y = fitted
    seen = []
    import src.models.ml_models as mm
    real_split = mm.TimeSeriesSplit.split

    def spy(self, Xa, *a, **kw):
        for tr, te in real_split(self, Xa, *a, **kw):
            seen.append((tr, te))
            yield tr, te

    monkeypatch.setattr(mm.TimeSeriesSplit, "split", spy)
    model.cross_val_report(X, y, [f"f{i}" for i in range(X.shape[1])])

    assert seen, "no folds were produced — the CV path did not run"
    for train_idx, test_idx in seen:
        assert train_idx.max() < test_idx.min(), (
            "a fold trained on data at or after its test window")


def test_returns_none_when_sample_too_small():
    m = MLModels()
    X, y = _synthetic(n=200)
    m.fit(X, y, [f"f{i}" for i in range(X.shape[1])])
    tiny_X, tiny_y = _synthetic(n=4, seed=3)
    assert m.cross_val_report(tiny_X, tiny_y) is None


def test_returns_none_when_no_models():
    m = MLModels()
    m.models = {}
    X, y = _synthetic(n=100)
    assert m.cross_val_report(X, y) is None


def test_degenerate_labels_do_not_raise(fitted):
    """A fold where a class is absent must not blow up the diagnostic."""
    model, X, _ = fitted
    y_const = np.zeros(len(X), dtype=int)
    report = model.cross_val_report(X, y_const, [f"f{i}" for i in range(X.shape[1])])
    assert report is None or "accuracy" in report
