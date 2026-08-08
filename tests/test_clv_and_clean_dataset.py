"""Stage 4, Phase 23 — CLV correctness, timestamp validity, and clean-dataset filtering."""

from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from src.evaluation import clv as clv_mod
from src.evaluation.clean_dataset import build
from src.evaluation.clv import (
    CLVInvalid,
    compute,
    coverage_report,
    validate_pair,
)

KO = datetime(2026, 8, 10, 19, 0)


def _pair(**over):
    base = dict(
        taken_odds=2.20, closing_odds=2.00,
        pick_market="1X2", closing_market="1X2",
        pick_selection="Home Win", closing_selection="Home Win",
        kickoff=KO, captured_at=KO - timedelta(minutes=60),
    )
    base.update(over)
    return base


# ───────────────────────────────────────────────── CLV formula

def test_clv_formula_matches_the_documented_definition():
    r = compute(taken_odds=2.20, closing_odds=2.00,
                kickoff=KO, captured_at=KO - timedelta(minutes=45))
    assert r.price_clv == pytest.approx(2.20 / 2.00 - 1)     # +10%
    assert r.prob_clv == pytest.approx(1 / 2.20 - 1 / 2.00)  # negative = good
    assert r.beat_close is True
    assert r.lead_minutes == pytest.approx(45)


def test_negative_clv_when_the_line_moved_against_us():
    r = compute(taken_odds=1.90, closing_odds=2.10)
    assert r.price_clv < 0
    assert r.beat_close is False


def test_fair_clv_is_optional_and_margin_free():
    r = compute(taken_odds=2.20, closing_odds=2.00,
                taken_fair=0.45, closing_fair=0.50)
    assert r.fair_clv == pytest.approx(0.50 / 0.45 - 1)
    assert compute(taken_odds=2.2, closing_odds=2.0).fair_clv is None


def test_clv_is_not_model_edge():
    """The metric that used to be called CLV read +6.3% while realised ROI was
    -3.6%. It depends on the model probability; genuine CLV does not."""
    import inspect
    src = inspect.getsource(compute)
    assert "predicted_probability" not in src
    assert "model" not in src.replace("model probability", "")


# ─────────────────────────────────────── timestamp / pair validity

def test_rejects_capture_after_kickoff():
    with pytest.raises(CLVInvalid, match="AFTER kickoff"):
        validate_pair(**_pair(captured_at=KO + timedelta(minutes=5)))


def test_rejects_capture_far_before_kickoff():
    with pytest.raises(CLVInvalid, match="beyond"):
        validate_pair(**_pair(captured_at=KO - timedelta(hours=6)))


def test_rejects_missing_capture_timestamp():
    with pytest.raises(CLVInvalid, match="no capture timestamp"):
        validate_pair(**_pair(captured_at=None))


def test_rejects_market_mismatch():
    with pytest.raises(CLVInvalid, match="market mismatch"):
        validate_pair(**_pair(closing_market="draw_no_bet"))


def test_rejects_selection_mismatch():
    with pytest.raises(CLVInvalid, match="selection mismatch"):
        validate_pair(**_pair(closing_selection="Away Win"))


def test_rejects_invalid_prices():
    with pytest.raises(CLVInvalid):
        validate_pair(**_pair(taken_odds=1.0))
    with pytest.raises(CLVInvalid):
        validate_pair(**_pair(closing_odds=None))


def test_accepts_a_valid_pair():
    validate_pair(**_pair())   # must not raise


# ──────────────────────────────────────────── coverage reporting

def _pick(**over):
    base = dict(odds=2.20, closing_odds=2.00, market="1X2", selection="Home Win",
                kickoff=KO, closing_odds_captured_at=KO - timedelta(minutes=60))
    base.update(over)
    return SimpleNamespace(**base)


def test_coverage_rate_and_stats():
    picks = [
        _pick(),                          # took 2.20, closed 2.00 -> BEAT the close
        _pick(closing_odds=2.40),         # took 2.20, closed 2.40 -> line moved AGAINST
        _pick(closing_odds=None),                                   # invalid
        _pick(closing_odds_captured_at=KO + timedelta(minutes=1)),  # after KO
        _pick(closing_odds_captured_at=None),                       # no timestamp
    ]
    cov = coverage_report(picks)
    assert cov.total_picks == 5
    assert cov.valid == 2
    assert cov.coverage_rate == pytest.approx(0.4)
    # One of the two valid pairs beat the close.
    assert cov.beat_close_rate == pytest.approx(0.5)
    assert cov.avg_price_clv == pytest.approx(((2.20 / 2.00 - 1) + (2.20 / 2.40 - 1)) / 2)
    assert "missing/invalid closing price" in cov.invalid_reasons
    assert "captured after kickoff" in cov.invalid_reasons
    assert "missing capture timestamp" in cov.invalid_reasons
    assert "clv_coverage_rate" in cov.render()


def test_coverage_with_no_valid_pairs_reports_na_not_zero():
    """Zero would read as 'we break even against the close'. It is 'unknown'."""
    cov = coverage_report([_pick(closing_odds=None)])
    assert cov.valid == 0
    assert cov.avg_price_clv is None
    assert "n/a" in cov.render()


def test_duplicate_capture_is_prevented_by_the_capture_query():
    """capture_closing_lines only selects picks whose closing_odds IS NULL, so a
    captured price is never overwritten by a later run."""
    import inspect
    from scripts import capture_closing_lines
    src = inspect.getsource(capture_closing_lines.capture)
    assert "closing_odds.is_(None)" in src


# ────────────────────────────────────────── clean dataset filtering

def _m(mid, hg=1, ag=0, d=datetime(2026, 5, 1)):
    return SimpleNamespace(id=mid, match_date=d, home_team_id=1, away_team_id=2,
                           home_goals=hg, away_goals=ag, league="x/y")


def _o(mid, book, sel, odds, mt="1X2", ts=None):
    return SimpleNamespace(match_id=mid, bookmaker=book, market_type=mt,
                           selection=sel, odds_value=odds, timestamp=ts)


def _book(mid, book, h, d, a, **kw):
    return [_o(mid, book, "Home", h, **kw), _o(mid, book, "Draw", d, **kw),
            _o(mid, book, "Away", a, **kw)]


def test_clean_dataset_excludes_corrupt_books_but_keeps_the_match():
    odds = (_book(1, "Bet365", 1.25, 3.40, 3.75)      # corrupt (overround 1.361)
            + _book(1, "Pinnacle", 1.71, 3.66, 4.55)
            + _book(1, "1xBet", 1.74, 3.81, 4.89))
    ds, man = build([_m(1)], odds, markets=("1X2",), min_books=2)
    assert len(ds) == 1
    assert ds[0].n_books["1X2"] == 2
    assert man.books_rejected["1X2: implausible overround (corruption gate)"] == 1
    assert 0.52 < ds[0].consensus("1X2")[0] < 0.57


def test_match_with_only_corrupt_books_is_rejected_entirely():
    ds, man = build([_m(1)], _book(1, "Bet365", 1.25, 3.40, 3.75),
                    markets=("1X2",), min_books=1)
    assert ds == []
    assert man.qualified_matches == 0


def test_quorum_is_enforced():
    odds = _book(1, "Pinnacle", 1.71, 3.66, 4.55)
    assert build([_m(1)], odds, markets=("1X2",), min_books=1)[0] != []
    assert build([_m(1)], odds, markets=("1X2",), min_books=2)[0] == []


def test_post_kickoff_odds_are_excluded():
    ko = datetime(2026, 5, 1)
    late = _book(1, "Pinnacle", 1.71, 3.66, 4.55, ts=ko + timedelta(hours=1))
    early = _book(1, "1xBet", 1.74, 3.81, 4.89, ts=ko - timedelta(hours=2))
    ds, man = build([_m(1, d=ko)], late + early, markets=("1X2",), min_books=1)
    assert man.books_rejected["odds timestamped after kickoff"] == 3
    assert ds[0].n_books["1X2"] == 1


def test_flashscore_display_prices_are_excluded():
    odds = (_book(1, "Flashscore", 1.71, 3.66, 4.55)
            + _book(1, "Pinnacle", 1.71, 3.66, 4.55))
    ds, man = build([_m(1)], odds, markets=("1X2",), min_books=1)
    assert man.books_rejected["excluded bookmaker (display-only prices)"] == 3
    assert ds[0].n_books["1X2"] == 1


def test_incomplete_match_is_rejected():
    m = _m(1)
    m.home_goals = None
    ds, man = build([m], _book(1, "Pinnacle", 1.71, 3.66, 4.55),
                    markets=("1X2",), min_books=1)
    assert ds == []
    assert man.rejected["no valid outcome (incomplete match)"] == 1


def test_manifest_is_renderable_and_auditable():
    ds, man = build([_m(1)], _book(1, "Pinnacle", 1.71, 3.66, 4.55),
                    markets=("1X2",), min_books=1)
    text = man.render()
    assert "matches qualified" in text
    assert "min surviving books" in text


def test_consensus_probabilities_sum_to_one():
    odds = (_book(1, "Pinnacle", 1.71, 3.66, 4.55)
            + _book(1, "1xBet", 1.74, 3.81, 4.89)
            + _book(1, "A", 1.80, 3.50, 4.40))
    ds, _ = build([_m(1)], odds, markets=("1X2",), min_books=2)
    assert sum(ds[0].consensus("1X2")) == pytest.approx(1.0, abs=1e-9)
