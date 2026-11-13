"""Stage 31 — the backtest of `confident_value`, and what it found.

Registered in docs/confident-value-preregistration.md (commit 4fcdd83) before
any outcome was read. The holdout was opened ONCE.

WHAT IT FOUND, in the terms the directive asked for:

    BLEND           the market takes 93-100% of the weight in every family
    AGREEMENT       raises the hit rate by selecting SHORTER prices; inside
                    the selection cells it adds nothing
    BACKTEST        no cell with a real interval clears the current rule
    PRICES          UNEVALUATED at the median (2 generous legs of 1,427)
    DECISION TABLE  nothing recommended; nothing marked on a real interval

THE DRIVEN HALF. The fit is exercised on synthetic sets where the answer is
known: a market that is exactly right takes all the weight against a noisy
model, and a planted model that knows more than the market is found. A
positive finding from a fit that could not have found one is not a finding.
"""
import ast
import datetime as dt
import pathlib
import random

import pytest

from scripts import confident_backtest as bt
from src.betting import confident_value as cv

M = bt.MEASURED
SRC = pathlib.Path("scripts/confident_backtest.py").read_text(encoding="utf-8")


# ═══════════════════════════ the registration ═══════════════════════════════

def test_the_REGISTRATION_is_named_and_the_grid_is_the_registered_size():
    assert bt.PREREGISTRATION_COMMIT == "4fcdd83" == M["preregistration"]
    assert bt.REGISTERED_CELLS == len(cv.registered_grid()) == 26
    assert bt.DECLARED_COMPARISONS == 26 + 4 + 5 + 1
    doc = pathlib.Path(
        "docs/confident-value-preregistration.md").read_text(encoding="utf-8")
    for needle in ("**26 cells.**", "| **declared** | **36** |",
                   "opened once", "1.8 expected false positives"):
        assert needle in doc, needle


def test_the_OUTCOME_FREE_load_does_not_SELECT_a_score():
    """From the AST of `load`: the score columns enter the query only through
    the `with_results` branch. An outcome-free step cannot read a result by
    accident, because the query it runs never asks for one."""
    tree = ast.parse(SRC)
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
              and n.name == "load")
    score_literals = [n for n in ast.walk(fn) if isinstance(n, ast.Constant)
                      and isinstance(n.value, str)
                      and "home_goals" in n.value]
    assert score_literals, "the score columns are no longer named in load()"
    guarded = set()
    for node in ast.walk(fn):
        if isinstance(node, (ast.IfExp, ast.If)) and "with_results" in (
                ast.get_source_segment(SRC, node.test) or ""):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Constant) and sub in score_literals:
                    guarded.add(id(sub))
    assert guarded == {id(n) for n in score_literals}


def test_the_HOLDOUT_never_fits():
    """From the AST of `run`: the only path that fits weights is the one
    guarded by `period == "dev"`, and the other asserts that weights were
    handed in. The CLI refuses `--open-holdout` without `--weights` too."""
    tree = ast.parse(SRC)
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
              and n.name == "run")
    asserts = [ast.get_source_segment(SRC, n.test) or ""
               for n in ast.walk(fn) if isinstance(n, ast.Assert)]
    assert asserts == ["weights_in is not None"], asserts
    fits = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
            and getattr(n.func, "id", "") in ("registered_weights",
                                              "walk_forward")]
    assert len(fits) == 2
    dev_branches = [n for n in ast.walk(fn) if isinstance(n, ast.If)
                    and (ast.get_source_segment(SRC, n.test) or "")
                    == 'period == "dev"']
    inside = {id(c) for b in dev_branches for stmt in b.body
              for c in ast.walk(stmt)}
    assert all(id(f) in inside for f in fits), (
        "a weight fit sits outside the development branch")
    assert "--open-holdout needs --weights" in SRC


def test_the_CONTROL_matched_the_Stage_30_loader_exactly():
    c = M["control"]
    assert c["leg_sets"] == 225581
    assert c["differing_from_stage30_loader"] == 0
    assert c["grid_cells"] == 26


# ═══════════════════ the fit, driven on sets with a known answer ════════════

def _synthetic(n, model_knows, seed=7):
    """`n` three-way sets. The TRUTH is drawn per set; the outcome is drawn
    from the truth. `model_knows=False`: the market quotes the truth and the
    models quote noise. `model_knows=True`: Poisson quotes the truth and the
    market quotes a blurred version of it."""
    rng = random.Random(seed)
    obs = []
    for i in range(n):
        a, b = sorted((rng.uniform(0.15, 0.7), rng.uniform(0.15, 0.7)))
        truth = [a * 0.9, (b - a) * 0.9 + 0.05, 0.0]
        truth[2] = 1.0 - truth[0] - truth[1]
        if min(truth) <= 0.02:
            continue
        noise = [rng.uniform(0.1, 0.8) for _ in range(3)]
        noise = [x / sum(noise) for x in noise]
        blurred = [0.5 * t + 0.5 / 3 for t in truth]
        u, winner, acc = rng.random(), 2, 0.0
        for k, t in enumerate(truth):
            acc += t
            if u < acc:
                winner = k
                break
        sources = ({"market": truth, "elo": noise, "poisson": noise[::-1]}
                   if not model_knows else
                   {"market": blurred, "elo": noise, "poisson": truth})
        obs.append({"mid": i, "family": "1X2", "sources": sources,
                    "winner": winner, "date": dt.date(2026, 3, 1)})
    return obs


def test_a_CALIBRATED_MARKET_takes_the_weight_against_noise():
    w = bt.fit_weights(_synthetic(3000, model_knows=False), "1X2")
    assert w["weights"]["market"] >= 0.95, w["weights"]
    assert w["log_loss"] <= w["market_log_loss"] + 1e-12


def test_a_PLANTED_model_that_knows_more_IS_FOUND():
    """The control for the finding. If the fit could not move weight off the
    market, "the market takes ~100%" would be a property of the fit."""
    w = bt.fit_weights(_synthetic(3000, model_knows=True), "1X2")
    assert w["weights"]["poisson"] >= 0.80, w["weights"]
    assert w["weights"]["market"] <= 0.20
    assert w["log_loss"] < w["market_log_loss"] - 0.01


def test_the_SIMPLEX_is_exhaustive_at_the_registered_step():
    two = bt._simplex(("market", "poisson"))
    three = bt._simplex(("market", "elo", "poisson"))
    assert len(two) == 101 and len(three) == 5151
    for g in two + three:
        assert sum(g.values()) == pytest.approx(1.0)
        assert min(g.values()) >= 0.0
    assert bt.GRID_STEP == 0.01


def test_a_set_with_a_MISSING_source_is_left_out_of_the_fit_and_counted():
    obs = _synthetic(400, model_knows=False)
    for o in obs[:50]:
        o["sources"] = dict(o["sources"], poisson=None)
    w = bt.fit_weights(obs, "1X2")
    assert w["dropped_missing_source"] == 50
    assert w["n_sets"] == len(obs) - 50


# ═══════════════════════════ the decision table ═════════════════════════════

def _cell(n, hit, lo, hi, lane):
    return {"n": n, "hit_rate": hit, "hit_ci95": [lo, hi],
            "hit_ci_bonferroni": [lo - 0.02, hi + 0.02],
            "predicted_hit_rate": 0.55, "avg_odds": 1.8, "roi": 0.0,
            "roi_ci95": [-0.1, 0.1], "lane": lane,
            "per_day": {"median": 1, "min": 0, "max": 3,
                        "zero_pick_days": 2, "days_with_fixtures": 10}}


def _holdout(cells, base=0.58):
    names = [r.name for r in cv.registered_grid()]
    full = {name: {"n": 0, "lane": r.lane}
            for name, r in zip(names, cv.registered_grid())}
    full.update(cells)
    return {"variants": {"corrected": {
        "cells": full,
        "current_rule": {"model_pick": {"hit_rate": base, "n": 1000}}}}}


def test_ONE_WINNING_PICK_is_marked_AS_WRITTEN_and_flagged_DEGENERATE():
    """The defect in the registered rule, reproduced. It names no minimum n,
    and a bootstrap of identical outcomes has zero width -- so one winning
    pick "clears" the current rule. The mark as written is kept; the table
    says there is no interval behind it."""
    d = bt.decision_table(_holdout({
        "L2 e>=0.00 agree": _cell(1, 1.0, 1.0, 1.0, cv.LANE_VALUE),
        "C p>=0.56 all": _cell(147, 0.626, 0.551, 0.701, "C")}))
    assert d["marked"] == {cv.LANE_VALUE: "L2 e>=0.00 agree"}
    assert d["marked_n"] == {cv.LANE_VALUE: 1}
    assert d["marked_with_a_real_interval"] == {}
    row = next(r for r in d["rows"] if r["cell"] == "L2 e>=0.00 agree")
    assert row["interval_is_degenerate"] is True


def test_a_cell_whose_LOWER_BOUND_clears_IS_marked_and_the_HIGHEST_wins():
    d = bt.decision_table(_holdout({
        "C p>=0.54 all": _cell(400, 0.66, 0.61, 0.71, "C"),
        "C p>=0.56 all": _cell(300, 0.70, 0.64, 0.76, "C"),
        "C p>=0.50 all": _cell(900, 0.60, 0.57, 0.63, "C")}))
    assert d["marked_with_a_real_interval"] == {"C": "C p>=0.56 all"}
    by = {r["cell"]: r for r in d["rows"]}
    assert by["C p>=0.50 all"]["clears_current_rule_95"] is False


def test_LANES_ARE_NEVER_POOLED_the_mixed_card_is_ARITHMETIC_ONLY():
    d = bt.decision_table(_holdout({
        "L1 p>=0.54 all": _cell(400, 0.56, 0.51, 0.61, cv.LANE_CONFIDENT),
        "L2 e>=0.02 all": _cell(40, 0.40, 0.25, 0.55, cv.LANE_VALUE)}))
    assert {r["lane"] for r in d["rows"]} == {
        "C", cv.LANE_CONFIDENT, cv.LANE_VALUE}
    mixed = d["mixed_card_arithmetic"]
    assert len(mixed) == 1
    assert mixed[0]["shares"]["90/10"] == pytest.approx(0.9 * 0.56 + 0.1 * 0.40)
    assert mixed[0]["shares"]["50/50"] == pytest.approx(0.48)
    assert bt.MIXED_SHARES == ((0.90, 0.10), (0.75, 0.25), (0.50, 0.50))


# ═════════════════════════ what was measured ════════════════════════════════

def test_BLEND_the_market_takes_AT_LEAST_93_percent_in_every_family():
    for variant in ("corrected", "deployed"):
        for fam, w in M["weights"][variant].items():
            assert w["weights"]["market"] >= 0.93, (variant, fam)
            # and "market only" is inside every weight interval
            assert w["ci95"]["market"][1] == 1.0, (variant, fam)
    assert M["weights"]["corrected"]["1X2"]["weights"] == {
        "market": 0.93, "elo": 0.06, "poisson": 0.01}
    assert M["weights"]["deployed"]["1X2"]["weights"] == {
        "market": 0.94, "elo": 0.04, "poisson": 0.02}
    assert M["ml_share"] == "UNEVALUATED"


def test_BLEND_is_never_better_than_the_market_out_of_sample():
    """Walk-forward on development and frozen on the holdout: not one family
    has an interval below zero. On the holdout team goals is WORSE."""
    for variant, fams in M["walk_forward_development"].items():
        for fam, w in fams.items():
            if w["n"] > 1:
                assert w["ci95"][1] > 0, (variant, fam)
    ho = M["delta_vs_market"]["holdout"]
    for fam, w in ho.items():
        if w["n"]:
            assert w["ci95"][1] >= 0, fam
    assert ho["team_goals"]["ci95"][0] > 0, (
        "on the holdout the team-goals blend is worse than the market")
    assert ho["btts"]["n"] == 0, "no BTTS anchor set exists in the holdout"


def test_the_DEPLOYED_weights_in_the_config_are_the_FITTED_ones():
    import yaml
    cfg = yaml.safe_load(pathlib.Path(
        "config/config.example.yaml").read_text(encoding="utf-8"))
    got = cfg["betting"]["confident_value"]["weights"]
    for fam, w in M["weights"]["deployed"].items():
        assert got[fam] == pytest.approx(w["weights"]), fam


def test_AGREEMENT_raises_the_hit_rate_by_selecting_SHORTER_PRICES():
    a = M["agreement_pooled"]["holdout"]
    assert a["verdict"] == "RAISES" and a["diff_ci95"][0] > 0
    assert a["agree"]["avg_odds"] < 2.0 < 3.0 < a["disagree"]["avg_odds"]
    # predicted beside realised: both groups are calibrated, so the gap is
    # the price, not information
    for side in ("agree", "disagree"):
        assert abs(a[side]["hit_rate"] - a[side]["predicted_hit_rate"]) < 0.01
    # and neither group returns a profit
    assert a["agree"]["roi"] < 0 and a["disagree"]["roi_ci95"][1] < 0


def test_AGREEMENT_adds_NOTHING_inside_the_selection_cells():
    """Where price and probability are already bounded, the filter moves the
    holdout hit rate by less than one point, in the WRONG direction."""
    cells = M["cells"]["holdout"]
    for p in ("0.50", "0.52", "0.54", "0.56"):
        both, agree = cells[f"C p>={p} all"], cells[f"C p>={p} agree"]
        assert agree["n"] < both["n"]
        assert agree["hit"] - both["hit"] < 0.01, p
        assert both["hit_ci95"][0] <= agree["hit"] <= both["hit_ci95"][1]


def test_BACKTEST_no_cell_with_a_real_interval_clears_the_current_rule():
    base = M["decision_table"]["current_rule_hit_rate"]
    assert base == pytest.approx(0.5831, abs=1e-4)
    cells = M["cells"]["holdout"]
    clearing = [n for n, c in cells.items()
                if c["n"] and c["hit_ci95"][0] > base and 0 < c["hit"] < 1]
    assert clearing == []
    assert M["decision_table"]["marked_with_a_real_interval"] == {}
    assert M["decision_table"]["marked_as_written"] == {
        "value": "L2 e>=0.00 agree"}
    assert M["decision_table"]["marked_n"] == {"value": 1}


def test_HIT_RATE_ODDS_AND_ROI_TRAVEL_TOGETHER_and_every_ROI_includes_zero():
    for per in ("development", "holdout"):
        for name, c in M["cells"][per].items():
            if not c["n"]:
                continue
            for k in ("hit", "hit_ci95", "predicted", "avg_odds", "roi",
                      "roi_ci95"):
                assert k in c, (per, name, k)
            assert c["avg_odds"] >= 1.70, (per, name)
    for name, c in M["cells"]["holdout"].items():
        if c["n"] >= 30:
            assert c["roi_ci95"][0] < 0 < c["roi_ci95"][1], name


def test_the_BEST_LOOKING_cell_is_NOT_ECHOED_by_development():
    """A registered refuter. P_min 0.56 reads 62.6% on the holdout and 56.1%
    on development, where its predicted hit rate is 56.6%."""
    ho = M["cells"]["holdout"]["C p>=0.56 all"]
    dv = M["cells"]["development"]["C p>=0.56 all"]
    assert (ho["n"], dv["n"]) == (147, 244)
    assert ho["hit"] == pytest.approx(0.6259, abs=1e-4)
    assert dv["hit"] == pytest.approx(0.5615, abs=1e-3)
    assert abs(dv["hit"] - dv["predicted"]) < 0.01
    assert ho["hit_ci95"][0] < M["decision_table"]["current_rule_hit_rate"]


def test_NO_FIXED_DAILY_COUNT_the_rule_has_zero_pick_days():
    cells = M["cells"]["holdout"]
    zero = {p: cells[f"C p>={p} all"]["zero_pick_days"]
            for p in ("0.50", "0.52", "0.54", "0.56", "0.58")}
    assert zero == {"0.50": 2, "0.52": 4, "0.54": 5, "0.56": 15, "0.58": 51}
    assert all(cells[f"C p>={p} all"]["days"] == 56 for p in zero)
    assert [cells[f"C p>={p} all"]["per_day_median"]
            for p in ("0.50", "0.54", "0.56", "0.58")] == [14.5, 6.5, 2.0, 0.0]


def test_LANE_2_has_TOO_FEW_PICKS_to_say_anything():
    cells = M["cells"]["holdout"]
    ns = [c["n"] for n, c in cells.items() if n.startswith("L2")]
    assert ns == [9, 3, 1, 1, 0, 0]


def test_the_CEILING_arithmetic():
    c = M["ceiling"]["holdout"]
    assert c["ceiling_at_min_price_zero_margin"] == pytest.approx(1 / 1.70,
                                                                  abs=1e-4)
    assert c["ceiling_at_min_price_measured_margin"] == pytest.approx(
        1 / (1.70 * (1 + c["mean_margin"])), abs=1e-3)
    assert 0.53 < c["ceiling_at_min_price_measured_margin"] < 0.54


def test_the_result_does_NOT_LEAN_ON_THE_KAMBI_BOOKS():
    base = M["cells"]["holdout"]
    without = M["sensitivity_holdout"]["without_kambi"]
    for name, c in without.items():
        if c["n"] >= 100:
            assert abs(c["hit"] - base[name]["hit"]) < 0.01, name


def test_PRICES_the_generous_median_BARELY_EXISTS():
    p = M["prices"]
    assert (p["fixtures_with_a_price_path"], p["legs"]) == (317, 1427)
    assert p["primary_median_taken_ge_1.70"]["n"] == 2, (
        "the registered primary population is two legs; its mechanical "
        "verdict is not a verdict")
    assert p["fresh_subset_n"] == 0 and p["column_present"] is False
    assert p["book_quotes_unevaluated"] == 20565


def test_PRICES_per_book_the_EXCHANGES_are_SLOW_and_the_KAMBI_books_are_not():
    books = M["prices"]["per_book_n_ge_30"]
    for ex in ("TheOddsAPI-matchbook", "TheOddsAPI-betfair_ex_eu"):
        assert books[ex]["verdict"] == "SLOW"
        assert books[ex]["instant_edge"] > 0 > books[ex]["closing_edge"]
    for kambi in ("TheOddsAPI-unibet_se", "TheOddsAPI-unibet_nl",
                  "TheOddsAPI-leovegas_se"):
        assert books[kambi]["verdict"] == "REAL"
        assert books[kambi]["closing_ci95"][0] > 0
        # long prices: this is not the 1.70-2.00 lane
        assert books[kambi]["avg_price"] > 3.5


def test_the_PRE_EXPERIMENT_count_is_the_one_the_directive_named():
    pe = M["pre_experiment"]
    assert pe["rows"] == 1018 and pe["settled_valid"] == 1012
    assert pe["until"] == "2026-08-05"
    assert pe["as_saved"]["all"]["n"] == 1012
    assert pe["with_todays_gates"]["all"]["n"] == 470


def test_the_BLOCKER_is_the_CONFIDENCE_GATE_and_it_is_measured_on_REAL_picks():
    """`max_odds` 10.0 permits long prices and the confidence gate removes
    them. On the pre-experiment picks, of those clearing every OTHER gate of
    today, the confidence gate alone removes 2% in 1.70-2.00, 41% in
    2.00-2.50 and 81% in 2.50-3.50."""
    g = M["pre_experiment"]["confidence_gate_alone"]
    share = {b: g[b]["removed"] / g[b]["clear_every_other_gate"]
             for b in ("1.70-2.00", "2.00-2.50", "2.50-3.50")}
    assert share["1.70-2.00"] == pytest.approx(3 / 190)
    assert share["2.00-2.50"] == pytest.approx(101 / 248)
    assert share["2.50-3.50"] == pytest.approx(87 / 107)
    assert share["1.70-2.00"] < share["2.00-2.50"] < share["2.50-3.50"]
    # and under a blend that is nearly the market, no leg at 2.00+ reaches it
    proxy = M["blocker_blend_proxy_holdout"]
    assert proxy["2.00-2.50"]["at_or_above_052"] == 0
    assert proxy["2.50-3.50"]["at_or_above_052"] == 0
