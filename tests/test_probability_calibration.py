"""Tests for the isotonic probability calibrator (dormant by default).

Built per docs/calibration-plan.md; the 2026-07-21 acceptance backtest kept it
disabled (July predictions already calibrated), but the machinery must be ready
to activate via models.probability_calibration_enabled when drift returns.
"""

import random

import pytest

from src.models.probability_calibration import (
    ProbabilityCalibrator, SELECTION_FAMILY, key_family,
)


def _synthetic_pairs(n=600, overconfidence=0.10, seed=7):
    """Picks whose true win rate runs `overconfidence` below the prediction."""
    rng = random.Random(seed)
    pairs = []
    fams = ["1x2", "goals", "btts", "team_goals"]
    for _ in range(n):
        p = rng.uniform(0.40, 0.80)
        true_p = max(0.05, p - overconfidence)
        pairs.append((rng.choice(fams), p, 1 if rng.random() < true_p else 0))
    return pairs


def test_fit_pulls_overconfident_probabilities_down():
    cal = ProbabilityCalibrator()
    assert cal.fit(_synthetic_pairs()) is True
    # A 65% overconfident prediction must calibrate materially lower.
    calibrated = cal.apply("home_win", 0.65)
    assert calibrated < 0.65
    assert calibrated > 0.30  # and not absurdly low


def test_apply_is_monotonic():
    cal = ProbabilityCalibrator()
    cal.fit(_synthetic_pairs())
    xs = [0.40, 0.50, 0.60, 0.70, 0.80]
    ys = [cal.apply("over_2.5", x) for x in xs]
    assert all(a <= b + 1e-9 for a, b in zip(ys, ys[1:])), "must be monotonic"


def test_unfitted_is_identity():
    cal = ProbabilityCalibrator()
    assert cal.is_fitted is False
    assert cal.apply("home_win", 0.61) == 0.61


def test_load_missing_file_is_identity(tmp_path):
    cal = ProbabilityCalibrator.load(tmp_path / "nope.json")
    assert cal.is_fitted is False
    assert cal.apply("btts_yes", 0.55) == 0.55


def test_persistence_roundtrip(tmp_path):
    cal = ProbabilityCalibrator()
    cal.fit(_synthetic_pairs())
    path = tmp_path / "prob_cal.json"
    cal.save(path)
    loaded = ProbabilityCalibrator.load(path)
    assert loaded.is_fitted
    for p in (0.45, 0.55, 0.65, 0.75):
        assert loaded.apply("home_win", p) == pytest.approx(
            cal.apply("home_win", p), abs=1e-9)


def test_small_family_falls_back_to_global():
    # 400 global pairs but only a handful of team_goals — that family must
    # fall back to the global map rather than fit on noise.
    pairs = _synthetic_pairs(n=400)
    pairs = [t for t in pairs if t[0] != "team_goals"]
    pairs += [("team_goals", 0.6, 1) for _ in range(10)]
    cal = ProbabilityCalibrator()
    cal.fit(pairs)
    assert "team_goals" not in cal.maps
    assert "global" in cal.maps
    # the family key still calibrates — via the global map
    assert cal.apply("home_over_1.5", 0.6) != 0.6 or cal.apply("home_win", 0.6) == cal.apply("home_over_1.5", 0.6)


def test_too_little_data_refuses_to_fit():
    cal = ProbabilityCalibrator()
    assert cal.fit(_synthetic_pairs(n=100)) is False
    assert cal.is_fitted is False


def test_key_family_mapping():
    assert key_family("home_win") == "1x2"
    assert key_family("over_2.5") == "goals"
    assert key_family("under_1.5") == "goals"
    assert key_family("btts_yes") == "btts"
    assert key_family("home_over_1.5") == "team_goals"
    assert key_family("nonsense") is None
    assert SELECTION_FAMILY["BTTS Yes"] == "btts"
