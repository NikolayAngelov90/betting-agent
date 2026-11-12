"""Stage 30: a correct fair price, the sharp-anchor backtest, the harness.

WHAT THIS FILE DEFENDS, and each is a way the stage could quietly rot:

  1. THE ESTIMATOR PASSES ITS SYNTHETIC CONTROL. The band test read a
     de-vigging artefact as a finding because its null was not zero. Here
     every estimator is driven against a perfectly calibrated book under
     four margin models -- live, not from pinned numbers.
  2. THE SEAL. The outcome-free steps run a query that does not select the
     score columns, and the holdout is read by one flag.
  3. THE REGISTERED VERDICT IS THE REGISTERED ONE: `NOT SUPPORTED at this
     n`, on the holdout, pooled, 3%. Secondary rows that exclude zero stay
     labelled secondary.
  4. THE HARNESS'S POSITIVE CONTROL: zero features reproduces the market
     exactly, a planted feature is found, and noise is not.
  5. THE ELO REPLICA IS THE PRODUCTION SYSTEM, and the season-label trap
     that inverted the first measurement is recorded.

NOTE ON IMPORTS. `scripts.sharp_anchor` and `scripts.residual_harness` keep
their database, market_spec and agent imports INSIDE functions.
"""
import ast
import pathlib
import random

import pytest

from scripts import residual_harness as rh
from scripts import sharp_anchor as sa
from src.evaluation.devig import (ESTIMATORS, EXACT_FOR, GENERATORS, additive,
                                  power, proportional, shin)


# ── 1. THE SYNTHETIC CONTROL, DRIVEN ─────────────────────────────────────

@pytest.fixture(scope="module")
def control():
    return sa.control()


def test_EVERY_ESTIMATOR_IS_EXACT_ON_THE_MODEL_IT_INVERTS(control):
    """The diagonal. If this fails, the estimator does not do what its name
    says and nothing downstream is a fair price."""
    for key, v in control.items():
        if v["exact_expected"]:
            assert v["max_abs_gap_pp"] < 1e-6, (
                f"{key}: an estimator leaves {v['max_abs_gap_pp']:.6f}pp on "
                f"the margin model it is supposed to invert")
            # a MULTIPLICATIVE book quotes above 1.0 for an extreme
            # favourite (0.95 * 1.066), which is not a price; those sets are
            # refused rather than estimated, and only a handful exist
            assert v["refused"] <= 5, key
            if v["generator"] != "multiplicative":
                assert v["refused"] == 0, key


def test_PROPORTIONAL_FAILS_THE_ADDITIVE_BOOK_which_is_the_band_tests_error(
        control):
    """The control the band test lacked, with the failure it would have
    shown: over a point of manufactured gap from a perfectly calibrated
    book."""
    for arity in (2, 3):
        v = control[f"{arity}way|M=0.066|additive|proportional"]
        assert v["max_abs_gap_pp"] > 1.0, (
            f"proportional de-vigging now leaves only "
            f"{v['max_abs_gap_pp']:.3f}pp on a {arity}-way equal-share book. "
            f"The known-wrong estimator has stopped being wrong, so this "
            f"control no longer controls")
    # and the gap has the band test's own SHAPE: negative low, positive high
    g = control["2way|M=0.066|additive|proportional"]["gap_pp"]
    assert g["0.30-0.40"] < 0 < g["0.70-0.80"] < g["0.80-0.90"]


def test_SHIN_IS_THE_ADDITIVE_ESTIMATOR_FOR_TWO_WAY_SETS(control):
    """A known identity, and the reason Shin passes the two-way control
    exactly. Driven on the grid and on a single quote."""
    for margin in sa.CONTROL_MARGINS:
        v = control[f"2way|M={margin}|additive|shin"]
        assert v["max_abs_gap_pp"] < 1e-6
    q = [0.82, 0.26]
    a, s = additive(q), shin(q)
    assert a == pytest.approx([0.78, 0.22], abs=1e-9)
    assert s == pytest.approx(a, abs=1e-8)


def test_NO_ESTIMATOR_AMONG_THE_THREE_IS_EXACT_ON_A_THREE_WAY_ADDITIVE_BOOK(
        control):
    """The honest wrinkle, pinned so it is not rounded away later: on a
    3-way equal-share book only the additive estimator is zero. So choosing
    Shin is a claim about real books, supported by evidence, not arithmetic.
    """
    for e in ("proportional", "power", "shin"):
        v = control[f"3way|M=0.035|additive|{e}"]
        assert v["max_abs_gap_pp"] > 0.1, (
            f"{e} is now exact on a three-way additive book; the recorded "
            f"wrinkle has gone and the registration's wording needs "
            f"restating")
    assert control["3way|M=0.035|additive|shin"]["max_abs_gap_pp"] < 0.5
    assert control["3way|M=0.035|additive|additive"]["max_abs_gap_pp"] < 1e-6


@pytest.mark.parametrize("est", [proportional, additive, power, shin])
def test_an_ESTIMATE_IS_A_PROBABILITY_VECTOR(est):
    for q in ([0.55, 0.50], [0.45, 0.30, 0.31], [0.80, 0.25]):
        p = est(q)
        assert p is not None
        assert sum(p) == pytest.approx(1.0, abs=1e-9)
        assert all(0.0 < x < 1.0 for x in p)


def test_an_INVALID_SET_IS_REFUSED_not_clipped():
    """`None` is the refusal. A set summing below 1, a price outside (0,1),
    and an additive leg that would go non-positive are three inputs with no
    fair price, and clipping any of them would be a silent third state."""
    for est in ESTIMATORS.values():
        assert est([0.40, 0.45]) is None          # arb-shaped: sum < 1
        assert est([1.20, 0.10]) is None
    # a longshot quoted below its own equal share of the margin
    assert additive([0.98, 0.01, 0.07]) is None   # 0.01 - 0.02 < 0
    # and one quoted just above it is estimated, not refused
    assert additive([0.97, 0.02, 0.06]) is not None


def test_the_GENERATORS_HIT_THEIR_MARGIN():
    for name, gen in GENERATORS.items():
        for p in ([0.6, 0.4], [0.5, 0.27, 0.23]):
            q = gen(p, 0.05)
            assert sum(q) == pytest.approx(1.05, abs=1e-6), name
    assert set(EXACT_FOR.values()) == set(ESTIMATORS)


# ── 2. THE SEAL ──────────────────────────────────────────────────────────

def _calls(fn_name: str, callee: str):
    tree = ast.parse(pathlib.Path("scripts/sharp_anchor.py").read_text(
        encoding="utf-8"))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == fn_name)
    return [n for n in ast.walk(fn) if isinstance(n, ast.Call)
            and getattr(n.func, "id", "") == callee]


@pytest.mark.parametrize("fn", ["select", "coverage"])
def test_the_OUTCOME_FREE_STEPS_LOAD_WITHOUT_RESULTS(fn):
    """Read from the AST: every `load(...)` in an outcome-free step passes
    `with_results=False` as a literal. Not a convention -- the query those
    steps run does not select the score columns at all."""
    calls = _calls(fn, "load")
    assert calls, f"`{fn}` no longer calls load(); the seal check is vacuous"
    for c in calls:
        kw = {k.arg: k.value for k in c.keywords}
        assert "with_results" in kw, f"{fn}: load() without with_results"
        assert isinstance(kw["with_results"], ast.Constant) \
            and kw["with_results"].value is False, (
            f"`{fn}` loads WITH results. It ran before the registration and "
            f"must not be able to read an outcome")


def test_the_SCORE_COLUMNS_ARE_SELECTED_ONLY_WITH_RESULTS():
    src = pathlib.Path("scripts/sharp_anchor.py").read_text(encoding="utf-8")
    assert ('score = ", m.home_goals, m.away_goals" if with_results else ""'
            in src), (
        "the loader no longer omits the score columns when results are not "
        "asked for; `with_results=False` has become a promise instead of a "
        "property of the query")


def test_the_HOLDOUT_RANGE_IS_READ_BY_ONE_FLAG_ONLY():
    """`HOLDOUT_UNTIL` bounds a result-reading load in exactly two places:
    the registered test and the post-hoc diagnostic that names itself."""
    src = pathlib.Path("scripts/sharp_anchor.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    users = set()
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef):
            continue
        for n in ast.walk(fn):
            if isinstance(n, ast.Name) and n.id == "HOLDOUT_UNTIL":
                users.add(fn.name)
    assert users == {"main", "platform_check"}, (
        f"HOLDOUT_UNTIL is read in {sorted(users)}. The holdout was "
        f"registered to be opened by `--open-holdout` alone; the only other "
        f"reader allowed is the post-hoc diagnostic, which says it is one")
    assert sa.platform_check.__doc__.startswith("POST-HOC")


def test_the_PERIODS_AND_THRESHOLDS_ARE_THE_REGISTERED_ONES():
    doc = pathlib.Path("docs/sharp-anchor-preregistration.md").read_text(
        encoding="utf-8")
    assert sa.EDGE_THRESHOLDS == (0.02, 0.03, 0.05)
    assert sa.PRIMARY_THRESHOLD == 0.03 and "**Primary: 3%.**" in doc
    assert sa.PRIMARY_ESTIMATOR == "shin"
    assert sa.EDGE_CAP == 0.15 and "`edge > 15%`" in doc
    assert sa.SNAPSHOT_TOLERANCE_S == 120.0 and "**120 seconds**" in doc
    assert str(sa.HOLDOUT_FROM) == "2026-09-01"
    assert "2026-09-01 … 2026-11-09" in doc
    assert sa.PREREGISTRATION_COMMIT == "7cbbddd"
    assert "before any match result is read" in doc


# ── 3. THE VERDICT ───────────────────────────────────────────────────────

def test_the_REGISTERED_VERDICT_is_NOT_SUPPORTED_and_its_interval_says_so():
    m = sa.MEASURED
    v = m["holdout_pooled"]["0.03"]
    assert m["registered_verdict"] == "NOT SUPPORTED at this n"
    assert v["n"] >= 100 and v["fixtures"] >= 50, "it would be INSUFFICIENT"
    lo, hi = v["ci"]
    assert lo < 0 < hi, (
        f"the holdout primary interval [{lo}, {hi}] no longer spans zero; "
        f"the registered verdict would be a different one")
    assert sa._verdict({"n": v["n"], "fixtures": v["fixtures"],
                        "ci95": [lo, hi]}) == m["registered_verdict"]


def test_the_SIGN_DEPENDS_ON_THE_ESTIMATOR_which_was_a_registered_refuter():
    """Section 7 of the registration: a positive pooled result that
    disappears when the anchor estimator changes was the de-vig all along.
    On the holdout the primary is positive under Shin and power and NEGATIVE
    under proportional."""
    e = sa.MEASURED["holdout_primary_by_estimator"]
    assert e["shin"] > 0 and e["power"] > 0 > e["proportional"]


def test_a_SECONDARY_INTERVAL_THAT_EXCLUDES_ZERO_stays_SECONDARY():
    s = sa.MEASURED["holdout_odds_1_50_2_20"]
    assert s["0.02"]["ci"][0] > 0, "the recorded secondary look has moved"
    assert "SECONDARY" in s["label"] and "thirty" in s["label"], (
        "the odds-range row excludes zero and is no longer labelled as one "
        "of about thirty declared exploratory looks")


def test_the_ANCHOR_IS_CALIBRATED_and_the_KAMBI_ROWS_VIOLATE_IT():
    """The reason the platform rows are reported as UNEXPLAINED. On every
    comparable pair the anchor's fair probability matches the win rate to
    under a tenth of a point; on three books sharing a platform it misses by
    seven to ten."""
    for period, v in sa.MEASURED[
            "anchor_calibration_all_comparable"].items():
        assert abs(v["win_rate"] - v["fair"]) < 0.001, period
    k = sa.MEASURED["kambi_family"]
    assert k["post_hoc"] is True
    for legs in ("home_legs", "away_legs"):
        assert k[legs]["win"] - k[legs]["fair"] > 0.06, legs
    assert k["holdout_primary_without_family"]["roi"] < 0, (
        "the holdout primary is no longer negative with the family removed; "
        "the statement that the pooled figure leans on it needs restating")
    assert "UNEXPLAINED, not an edge" in k["reading"]
    assert "last_update" in k["reading"]


# ── 4. THE HARNESS'S OWN CONTROLS, DRIVEN ────────────────────────────────

def _synthetic(n: int, beta: float, seed: int = 7):
    import datetime as dt
    rng = random.Random(seed)
    obs = []
    for i in range(n):
        off = rng.gauss(0.0, 0.8)
        z = rng.gauss(0.0, 1.0)
        noise = rng.gauss(0.0, 1.0)
        p = rh._sigmoid(off + beta * z)
        obs.append({"date": dt.datetime(2026, 1, 1) + dt.timedelta(
                        hours=6 * i),
                    "mid": i, "y": 1 if rng.random() < p else 0,
                    "offset": off, "planted": z, "noise": noise})
    return obs


def test_ZERO_FEATURES_REPRODUCES_THE_MARKET_EXACTLY():
    """The positive control. With nothing to fit, the model IS the offset."""
    r = rh.walk_forward(_synthetic(1500, 0.0), None)
    assert r["delta_ll"] == 0.0
    assert r["ll_model"] == r["ll_market"]
    assert r["folds"] >= 3


def test_a_PLANTED_FEATURE_IS_FOUND_and_NOISE_IS_NOT():
    """Known-right and known-wrong, on the same data. A harness that finds
    nothing finds nothing in noise too; it has to be shown to find something
    when something is there."""
    obs = _synthetic(4000, 0.5)
    planted = rh.walk_forward(obs, "planted")
    noise = rh.walk_forward(obs, "noise")
    assert planted["delta_ll"] < -0.01 and planted["delta_ci95"][1] < 0, (
        f"a feature with a true coefficient of 0.5 was not detected: delta "
        f"{planted['delta_ll']:+.5f}")
    assert planted["beta_full_sample"] == pytest.approx(0.5, abs=0.1)
    lo, hi = noise["delta_ci95"]
    assert lo <= 0.002 and hi >= -0.002 and noise["delta_ll"] > -0.002, (
        f"pure noise was scored as informative: delta "
        f"{noise['delta_ll']:+.5f} [{lo:+.5f},{hi:+.5f}]")


def test_a_FEATURE_NOBODY_CARRIES_is_UNEVALUATED_not_zero():
    r = rh.walk_forward(_synthetic(600, 0.0), "absent_feature")
    assert "UNEVALUATED" in r
    assert "delta_ll" not in r


def test_TOO_LITTLE_HISTORY_is_UNEVALUATED_not_a_score():
    r = rh.walk_forward(_synthetic(120, 0.0), "planted")
    assert "UNEVALUATED" in r and str(rh.MIN_TRAIN) in r["UNEVALUATED"]


def test_NO_CANDIDATE_ADDS_INFORMATION_and_the_figures_carry_their_folds():
    m = rh.MEASURED
    for name, (d, lo, hi, _b, _bl, _bh) in m["home_win"]["candidates"].items():
        assert hi > 0, (
            f"{name}'s out-of-sample interval [{lo},{hi}] now excludes zero "
            f"on the improving side -- that is a finding and needs more "
            f"than this stage's six folds behind it")
    assert m["home_win"]["zero_feature_delta"] == 0.0
    assert m["home_win"]["folds"] == 6 and m["over_2.5"]["folds"] == 7
    assert m["odds_era"] == ("2026-02-28", "2026-11-09")
    assert "thin" in m["harness_verdict"]


def test_IN_SAMPLE_SIGNIFICANCE_did_not_survive_OUT_OF_SAMPLE():
    """`form_diff` is the harness earning its keep: a coefficient whose
    in-sample interval excludes zero, and an out-of-sample delta that does
    not improve on the market."""
    d, _lo, _hi, _b, b_lo, _b_hi = rh.MEASURED["home_win"]["candidates"][
        "form_diff"]
    assert b_lo > 0, "form_diff's in-sample interval no longer excludes zero"
    assert d > 0, "form_diff now improves out of sample; restate the lesson"


def test_the_SLOPE_ON_REAL_OUTCOMES_AGREES_WITH_THE_SYNTHETIC_CONTROL():
    """Two independent lines to one conclusion: on real home-win outcomes
    the calibration slope spans zero under Shin and power and EXCLUDES zero
    under proportional -- the estimator the synthetic control fails."""
    s = rh.MEASURED["home_win"]["slope_beta"]
    assert s["shin"][1] < 0 < s["shin"][2]
    assert s["power"][1] < 0 < s["power"][2]
    assert s["proportional"][1] > 0, (
        "proportional de-vigging no longer leaves a calibration slope on "
        "real outcomes; the corroboration of Part A has gone")


def test_the_MODEL_PROBABILITY_ROW_IS_LABELLED_A_SELECTED_SUBSET():
    mp = rh.MEASURED["model_probability"]
    assert "SELECTED" in mp["population"]
    assert mp["folds"] == 2 and mp["scored"] < 500
    assert mp["delta"][1] < 0 < mp["delta"][2]


# ── 5. ELO AND xG ────────────────────────────────────────────────────────

def test_the_ELO_REPLICA_WAS_THE_PRODUCTION_SYSTEM():
    e = rh.MEASURED["elo"]
    assert e["replica_max_rating_diff"] == 0.0 and e["teams"] == 1450, (
        "the chronological pass no longer reproduces EloRatingSystem.fit() "
        "exactly, so Part D measured a different system")


def test_BOTH_ELO_CHANGES_HELP_and_NEITHER_IS_ADOPTED():
    e = rh.MEASURED["elo"]
    for name, (ll, d, lo, hi) in e["variants"].items():
        assert hi < 0, f"{name}: the interval [{lo},{hi}] no longer excludes 0"
        assert ll == pytest.approx(e["deployed_log_loss"] + d, abs=2e-5)
    both = e["variants"]["season|fitted"][1]
    assert both < e["variants"]["season|fixed"][1]
    assert both < e["variants"]["calendar|fitted"][1]
    # measured, not fixed: the deployed curve is untouched
    src = pathlib.Path("src/models/elo_system.py").read_text(encoding="utf-8")
    assert "max(0.15, 0.28 - abs(diff) / 2000.0)" in src
    assert "m_year > prev_year" in src


def test_the_SEASON_LABEL_TRAP_IS_RECORDED_because_it_inverted_the_result():
    """`matches.season` mixes two formats inside 31 leagues. A season
    variant keyed on it regressed a third of the teams every other fixture
    and measured WORSE than the deployed Elo; the corrected one is better."""
    t = rh.MEASURED["elo"]["season_label_trap"]
    assert t["regressions_with_raw_labels"] > 2 * t["true_transitions"]
    assert t["delta_measured_with_raw_labels"] > 0 > rh.MEASURED["elo"][
        "variants"]["season|fixed"][1]
    # FROM THE AST, as one equality: the function issues NO query at all.
    # It derives the key from each row's date, so anything it calls on a
    # session would be a read of the raw label coming back.
    tree = ast.parse(pathlib.Path("scripts/residual_harness.py").read_text(
        encoding="utf-8"))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef)
              and n.name == "_season_labels")
    queries = sorted({getattr(c.func, "attr", "") for c in ast.walk(fn)
                      if isinstance(c, ast.Call)}
                     & {"execute", "get_session", "query"})
    assert queries == [], (
        f"`_season_labels` issues a query again ({queries}); it must derive "
        f"the season from the date, because the stored label is the trap")


def test_UNDERSTAT_IS_NOT_PROPOSED_while_its_terms_are_unobtained():
    x = rh.MEASURED["xg"]
    assert x["understat_terms"] == "UNOBTAINED"
    assert x["backfill_proposed"] is False, (
        "an Understat backfill is proposed without its terms. Unclear is "
        "not permission")
    assert x["share"] < 0.05 < x["after_understat_backfill"] < 0.30
    assert len(rh.UNDERSTAT_LEAGUES) == x["understat_leagues_in_config"] == 5
