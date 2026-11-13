"""Selection mode `confident_value` — the rule, and the flag it sits behind.

Stage 31. The operator's objective: the safest picks that still carry a good
price. Good price = taken odds >= 1.70. One pick per match. No fixed daily
count. The models weighted together, not the market alone.

ONE DEFINITION. `src/betting/confident_value.py` is read by the backtest
(`scripts/confident_backtest.py`) and by the pipeline flag. The first half of
this file pins the rule; the second half drives the flag:

    flag OFF  -> the recommendations are byte-identical to `find_value_bets`
    flag ON   -> no pick below 1.70, none below P_min, at most one per
                 fixture, and a fixture with nothing qualifying yields NO
                 picks and NO error

NOTHING IS SWITCHED ON. `betting.selection_mode` is `standard` in the deployed
config, the new keys are not in TRACKED_KEYS, and the fingerprint is the one
Part G left.
"""
import ast
import pathlib
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
import yaml

from src.betting import confident_value as cv

EXAMPLE = "config/config.example.yaml"
T0 = datetime(2026, 11, 21, 8, 0, 0)


def _rows(book, t=T0, home=2.10, draw=3.50, away=3.60,
          over=1.90, under=1.95):
    return [
        (book, "1X2", "Home", home, t), (book, "1X2", "Draw", draw, t),
        (book, "1X2", "Away", away, t),
        (book, "over_under", "Over 2.5", over, t),
        (book, "over_under", "Under 2.5", under, t),
    ]


# ═════════════════════════════ leg sets and prices ══════════════════════════

def test_a_COMPLETE_set_is_built_and_an_INCOMPLETE_one_is_not():
    rows = _rows("Pinnacle") + [("Bet365", "1X2", "Home", 2.0, T0),
                                ("Bet365", "1X2", "Draw", 3.4, T0)]
    sets, _ = cv.build_sets(rows)
    assert ("Pinnacle", "1X2", None, None) in sets
    assert ("Pinnacle", "over_under", "2.5", None) in sets
    assert ("Bet365", "1X2", None, None) not in sets
    assert sets[("Pinnacle", "1X2", None, None)]["keys"] == [
        "Home", "Draw", "Away"]


def test_legs_from_TWO_SNAPSHOTS_are_REFUSED():
    rows = [("Pinnacle", "1X2", "Home", 2.1, T0),
            ("Pinnacle", "1X2", "Draw", 3.5, T0),
            ("Pinnacle", "1X2", "Away", 3.6, T0 + timedelta(hours=3))]
    sets, refused = cv.build_sets(rows)
    assert sets == {}
    assert refused == {"legs_not_one_snapshot": 1}


def test_the_TAKEN_PRICE_is_the_MEDIAN_and_never_the_best():
    rows = (_rows("Pinnacle", home=2.10) + _rows("Bet365", home=2.00)
            + _rows("1xBet", home=2.60, draw=3.20, away=3.20))
    priced = cv.slots(cv.build_sets(rows)[0])
    sl = priced[("1X2", None, None)]
    assert sl["books"] == 3 and sl["feed"] == "apifootball"
    assert sl["taken"][0] == 2.10, "the median of 2.00, 2.10, 2.60"
    assert sl["taken"][0] < 2.60, "the best price across books is not taken"
    assert sum(sl["fair"]) == pytest.approx(1.0)


def test_the_median_matches_the_PIPELINES_own_aggregation():
    assert cv.median([1.9]) == 1.9
    assert cv.median([2.0, 1.8, 2.4]) == 2.0
    assert cv.median([1.8, 2.0, 2.2, 2.6]) == pytest.approx(2.1)


def test_a_book_from_ANOTHER_FEED_or_ANOTHER_SNAPSHOT_is_not_in_the_median():
    late = T0 + timedelta(hours=5)
    rows = (_rows("Pinnacle", home=2.10)
            + _rows("TheOddsAPI-unibet_se", home=2.90,        # other feed
                    draw=3.00, away=3.00)
            + _rows("Bet365", t=late, home=2.80,              # other snapshot
                    draw=3.00, away=3.00))
    sl = cv.slots(cv.build_sets(rows)[0])[("1X2", None, None)]
    assert sl["books"] == 1 and sl["taken"][0] == 2.10
    # ...and the sensitivity switch that takes them all says so by name
    wide = cv.slots(cv.build_sets(rows)[0], cross_feed=True)[
        ("1X2", None, None)]
    assert wide["books"] == 3 and wide["taken"][0] == 2.80


def test_NO_ANCHOR_means_NO_SLOT_never_a_model_only_price():
    rows = _rows("Bet365") + _rows("1xBet")
    assert cv.slots(cv.build_sets(rows)[0]) == {}


def test_rows_with_NO_TIMESTAMP_cannot_claim_a_snapshot():
    rows = [(b, m, s, v, None) for b, m, s, v, _t in _rows("Pinnacle")]
    sets, _ = cv.build_sets(rows)
    assert sets, "the sets exist"
    assert cv.slots(sets) == {}, (
        "a set with no timestamp was treated as inside the snapshot")
    assert cv.slots(sets, require_snapshot=False) != {}


def test_EXCLUDE_drops_a_book_and_never_the_anchor():
    rows = _rows("Pinnacle", home=2.10) + _rows("Bet365", home=2.50)
    sl = cv.slots(cv.build_sets(rows)[0],
                  exclude=("Bet365", "Pinnacle"))[("1X2", None, None)]
    assert sl["books"] == 1 and sl["taken"][0] == 2.10


def test_BANDS_are_half_open_and_the_last_is_closed():
    assert cv.band_of(1.69) is None
    assert cv.band_of(1.70) == (1.70, 2.00)
    assert cv.band_of(2.00) == (2.00, 2.50)
    assert cv.band_of(2.50) == (2.50, 3.50)
    assert cv.band_of(3.50) == (2.50, 3.50)
    assert cv.band_of(3.51) is None


# ═════════════════════════ the blended probability ══════════════════════════

def test_the_POOL_is_a_weighted_mean_that_sums_to_one():
    sources = {"market": [0.5, 0.3, 0.2], "elo": [0.6, 0.2, 0.2],
               "poisson": [0.4, 0.3, 0.3]}
    p = cv.pool(sources, {"market": 0.5, "elo": 0.25, "poisson": 0.25})
    assert p == pytest.approx([0.5, 0.275, 0.225])
    assert sum(p) == pytest.approx(1.0)


def test_a_source_with_NO_OPINION_is_DROPPED_and_the_rest_renormalised():
    sources = {"market": [0.5, 0.5], "poisson": None}
    assert cv.pool(sources, {"market": 0.7, "poisson": 0.3}) == [0.5, 0.5]


def test_NO_MARKET_means_NO_BLEND():
    assert cv.pool({"market": None, "poisson": [0.6, 0.4]},
                   {"market": 0.5, "poisson": 0.5}) is None


def test_MARKET_ONLY_weights_return_the_anchor_exactly():
    fair = [0.46, 0.27, 0.27]
    assert cv.pool({"market": fair, "elo": [0.9, 0.05, 0.05]},
                   {"market": 1.0}) == pytest.approx(fair)


def test_a_MODEL_SET_is_in_the_sets_own_key_order_and_an_UNDER_is_1_minus():
    pred = {"home_win": 0.5, "draw": 0.3, "away_win": 0.2, "over_2.5": 0.6,
            "btts_yes": 0.55, "home_over_0.5": 0.8}
    assert cv.model_set("1X2", None, None, ["Home", "Draw", "Away"],
                        pred) == pytest.approx([0.5, 0.3, 0.2])
    assert cv.model_set("over_under", "2.5", None,
                        ["Over 2.5", "Under 2.5"],
                        pred) == pytest.approx([0.6, 0.4])
    assert cv.model_set("btts", None, None, ["Yes", "No"],
                        pred) == pytest.approx([0.55, 0.45])
    assert cv.model_set("team_goals", "0.5", "Home",
                        ["Home Over 0.5", "Home Under 0.5"],
                        pred) == pytest.approx([0.8, 0.2])
    # a line the model has no number for is NO OPINION, not a default
    assert cv.model_set("over_under", "5.5", None,
                        ["Over 5.5", "Under 5.5"], pred) is None
    assert cv.model_set("1X2", None, None, ["Home", "Draw", "Away"],
                        None) is None


def test_AGREEMENT_needs_every_opinion_and_a_strict_first():
    three = {"market": [0.5, 0.3, 0.2], "elo": [0.45, 0.3, 0.25],
             "poisson": [0.4, 0.35, 0.25]}
    assert cv.agrees(three, 0) is True
    assert cv.agrees(three, 1) is False
    split = dict(three, elo=[0.3, 0.3, 0.4])
    assert cv.agrees(split, 0) is False
    # a tie for first is not agreement
    assert cv.agrees({"market": [0.5, 0.5], "poisson": [0.6, 0.4]}, 0) is False
    # one opinion agrees with nobody
    assert cv.agrees({"market": [0.6, 0.4], "poisson": None}, 0) is False
    # the binary families: market and Poisson only, Elo has none
    assert cv.agrees({"market": [0.6, 0.4], "poisson": [0.55, 0.45]}, 0)


def test_ELO_HAS_NO_OPINION_on_the_binary_families():
    assert cv.FAMILY_SOURCES["1X2"] == ("market", "elo", "poisson")
    for fam in ("over_under", "btts", "team_goals"):
        assert cv.FAMILY_SOURCES[fam] == ("market", "poisson")


def test_PICKABLE_is_the_pipelines_OWN_market_list_for_these_families():
    """Set equality against `_market_specs`, read from the AST so this file
    does not have to construct a calculator."""
    tree = ast.parse(pathlib.Path(
        "src/betting/value_calculator.py").read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
              and n.name == "_market_specs")
    specs = {}
    for node in ast.walk(fn):
        if isinstance(node, ast.Tuple) and len(node.elts) == 4 \
                and isinstance(node.elts[0], ast.Constant):
            market, selection, _prob, key = node.elts
            specs[selection.value] = (market.value, key.value)
    families = {"1X2", "BTTS", "Team Goals"}
    wanted = {sel: mk for sel, mk in specs.items()
              if mk[0] in families or mk[0].startswith(("Over", "Under"))}
    assert cv.PICKABLE == wanted
    assert len(cv.PICKABLE) == 17


def test_a_TEAM_GOALS_UNDER_is_priced_but_NEVER_picked():
    assert cv.pipeline_selection("team_goals", "Home Over 0.5") \
        == "Home Over 0.5"
    assert cv.pipeline_selection("team_goals", "Home Under 0.5") is None
    assert cv.pipeline_selection("1X2", "Home") == "Home Win"
    assert cv.pipeline_selection("over_under", "Under 2.5") \
        == "Under 2.5 Goals"


# ═══════════════════════════ the selection rule ═════════════════════════════

def _leg(fixture, taken, p, agree=True, key="Home"):
    return cv.Leg(fixture=fixture, family="1X2",
                  key=("1X2", None, None, key), taken=taken, p=p,
                  agree=agree, selection="Home Win")


def test_ONE_PICK_PER_FIXTURE_the_highest_probability():
    legs = [_leg(1, 1.80, 0.55, key="Home"), _leg(1, 1.90, 0.58, key="Draw"),
            _leg(2, 1.75, 0.60)]
    picks = cv.select(legs, cv.Rule("r", p_min=0.50))
    assert [(p.fixture, p.p) for p in picks] == [(1, 0.58), (2, 0.60)]


def test_a_TIE_goes_to_the_higher_price_then_the_key():
    legs = [_leg(1, 1.80, 0.55, key="Home"), _leg(1, 1.95, 0.55, key="Draw")]
    assert cv.select(legs, cv.Rule("r", p_min=0.5))[0].taken == 1.95
    same = [_leg(1, 1.80, 0.55, key="Home"), _leg(1, 1.80, 0.55, key="Away")]
    assert cv.select(same, cv.Rule("r", p_min=0.5))[0].key[3] == "Away"
    assert cv.select(same[::-1], cv.Rule("r", p_min=0.5))[0].key[3] == "Away"


def test_BELOW_THE_PRICE_or_BELOW_P_MIN_never_qualifies():
    rule = cv.Rule("r", lo=1.70, p_min=0.54)
    assert not rule.qualifies(_leg(1, 1.69, 0.90))
    assert not rule.qualifies(_leg(1, 2.50, 0.5399))
    assert rule.qualifies(_leg(1, 1.70, 0.54))


def test_the_LANE_BOUNDS_and_the_EDGE_rule():
    lane1 = cv.Rule("l1", lo=1.70, hi=2.00, p_min=0.5)
    assert lane1.qualifies(_leg(1, 1.99, 0.6))
    assert not lane1.qualifies(_leg(1, 2.00, 0.6)), "[1.70, 2.00)"
    lane2 = cv.Rule("l2", lo=2.00, hi=3.50, hi_closed=True, edge=0.02)
    assert lane2.qualifies(_leg(1, 2.00, 0.51))            # 1.02
    assert not lane2.qualifies(_leg(1, 2.00, 0.505))       # 1.01
    assert lane2.qualifies(_leg(1, 3.50, 0.30))            # closed top
    assert not lane2.qualifies(_leg(1, 3.51, 0.90))


def test_the_AGREEMENT_filter_removes_a_leg_the_sources_split_on():
    rule = cv.Rule("r", p_min=0.5, require_agreement=True)
    assert rule.qualifies(_leg(1, 1.8, 0.6, agree=True))
    assert not rule.qualifies(_leg(1, 1.8, 0.6, agree=False))


def test_NOTHING_QUALIFYING_is_an_EMPTY_LIST_not_an_error():
    assert cv.select([_leg(1, 1.50, 0.9), _leg(2, 1.9, 0.3)],
                     cv.Rule("r", p_min=0.5)) == []
    assert cv.select([], cv.Rule("r", p_min=0.5)) == []


def test_the_REGISTERED_GRID_has_26_cells_and_three_groups():
    grid = cv.registered_grid()
    assert len(grid) == 26
    assert len({r.name for r in grid}) == 26
    by = {}
    for r in grid:
        by.setdefault(r.lane, []).append(r)
    assert {k: len(v) for k, v in by.items()} == {
        "C": 10, cv.LANE_CONFIDENT: 10, cv.LANE_VALUE: 6}
    assert {r.p_min for r in by["C"]} == set(cv.P_MIN_GRID)
    assert all(r.hi is None for r in by["C"])
    assert all((r.lo, r.hi, r.hi_closed) == (1.70, 2.00, False)
               for r in by[cv.LANE_CONFIDENT])
    assert all((r.lo, r.hi, r.hi_closed) == (2.00, 3.50, True)
               and r.p_min is None for r in by[cv.LANE_VALUE])
    assert {r.edge for r in by[cv.LANE_VALUE]} == set(cv.EDGE_GRID)
    assert cv.P_MIN_GRID == (0.50, 0.52, 0.54, 0.56, 0.58)
    assert cv.EDGE_GRID == (0.00, 0.02, 0.05)
    assert cv.MIN_PRICE == 1.70


# ═══════════════════════════ the flag: config ═══════════════════════════════

def _cfg(over=None, drop=()):
    """The deployed config, with overrides, as the object the code reads:
    `.get("a.b")` and `.betting`."""
    data = yaml.safe_load(pathlib.Path(EXAMPLE).read_text(encoding="utf-8"))
    for dotted, value in (over or {}).items():
        node = data
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    for dotted in drop:
        node = data
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node[part]
        del node[parts[-1]]

    def get(key, default=None):
        node = data
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node
    return SimpleNamespace(get=get, data=data, betting=data["betting"])


def test_the_DEPLOYED_config_says_STANDARD_and_carries_the_two_numbers():
    c = _cfg()
    assert c.get("betting.selection_mode") == cv.MODE_STANDARD
    assert cv.mode_of(c) == cv.MODE_STANDARD
    assert c.get("betting.confident_value.min_price") == 1.70
    assert c.get("betting.confident_value.p_min") in cv.P_MIN_GRID
    rule = cv.rule_from_config(c)
    assert rule.lo == 1.70 and rule.p_min == c.get(
        "betting.confident_value.p_min")


def test_an_ABSENT_key_is_STANDARD_and_a_TYPO_is_REFUSED():
    assert cv.mode_of(SimpleNamespace(get=lambda k, d=None: d)) \
        == cv.MODE_STANDARD
    with pytest.raises(ValueError):
        cv.mode_of(_cfg({"betting.selection_mode": "confident-value"}))


def test_the_rule_reads_its_NUMBERS_FROM_THE_CONFIG():
    rule = cv.rule_from_config(_cfg({
        "betting.confident_value.min_price": 1.85,
        "betting.confident_value.p_min": 0.58}))
    assert (rule.lo, rule.p_min) == (1.85, 0.58)


def test_MALFORMED_WEIGHTS_are_REFUSED():
    ok = cv.weights_from_config(_cfg())
    assert set(ok) == set(cv.FAMILY_SOURCES)
    for fam, vec in ok.items():
        assert sum(vec.values()) == pytest.approx(1.0)
        assert set(vec) <= set(cv.FAMILY_SOURCES[fam])
    for bad in ({"btts": {"market": 0.5, "elo": 0.5}},        # no opinion
                {"1X2": {"market": 0.7, "poisson": 0.2}},     # sums to 0.9
                {"1X2": {"market": 1.2, "poisson": -0.2}},    # negative
                {"corners": {"market": 1.0}}):                # unknown family
        with pytest.raises(ValueError):
            cv.weights_from_config(
                _cfg({"betting.confident_value.weights": bad}))


def test_NOTHING_IS_SWITCHED_ON_and_the_fingerprint_did_not_move():
    from src.models.model_version import (CODE_REVISION, TRACKED_KEYS,
                                          model_version)
    from src.utils.config import Config
    from tests.experiment_pins import (CODE_REVISION_PIN,
                                       FROZEN_MODEL_VERSION)
    assert CODE_REVISION == CODE_REVISION_PIN
    assert model_version(Config(EXAMPLE)) == FROZEN_MODEL_VERSION
    new_keys = [k for k in TRACKED_KEYS
                if "selection_mode" in k or "confident_value" in k]
    assert new_keys == [], (
        "the selection-mode keys are tracked now. That is what SWITCHING "
        "requires, with a cohort bump -- it is not part of wiring the flag")


# ═══════════════════════════ the flag: driven ═══════════════════════════════

PRED = {
    "poisson": {"home_win": 0.50, "draw": 0.27, "away_win": 0.23,
                "over_1.5": 0.78, "over_2.5": 0.56, "over_3.5": 0.31,
                "over_4.5": 0.14, "under_2.5": 0.44, "btts_yes": 0.55,
                "btts_no": 0.45, "home_over_0.5": 0.82,
                "away_over_0.5": 0.70, "home_over_1.5": 0.52,
                "away_over_1.5": 0.33, "home_xg": 1.6, "away_xg": 1.1},
    "elo": {"home_win": 0.48, "draw": 0.28, "away_win": 0.24},
}
# The ENSEMBLE claims more than the blend will: Over 2.5 at 0.62 against a
# median of 1.85 is a +15% edge to the standard rule, so the standard rule
# returns a pick and the flag-off equality below is not of two empty lists.
PRED["ensemble"] = dict(PRED["poisson"], **{
    "over_2.5": 0.62, "under_2.5": 0.38,
    "under_1.5": 0.22, "under_3.5": 0.69, "under_4.5": 0.86})


def _agent(config, rows):
    """A REAL calculator behind a bare agent: only the selection step runs,
    and the one database read it makes is replaced by `rows`."""
    from src.agent.betting_agent import FootballBettingAgent
    from src.betting.value_calculator import ValueBettingCalculator
    a = FootballBettingAgent.__new__(FootballBettingAgent)
    a.config = config
    a.value_calculator = ValueBettingCalculator(config)
    a.predictor = None
    a._odds_rows_with_time = lambda match_id, kickoff=None: list(rows)
    return a


def _odds_data(rows):
    return [{"market_type": m, "selection": s, "odds_value": v,
             "bookmaker": b, "opening_odds": v} for b, m, s, v, _t in rows]


def _select(config, rows, match_id=7):
    a = _agent(config, rows)
    return a._select_recommendations(
        PRED, _odds_data(rows), "A vs B", {},
        home_team_name="A", away_team_name="B", match_id=match_id,
        league="england/premier-league", kickoff=T0 + timedelta(hours=6))


GOOD = (_rows("Pinnacle", home=1.86, draw=3.70, away=4.60, over=1.84,
              under=2.02)
        + _rows("Bet365", home=1.90, draw=3.60, away=4.40, over=1.85,
                under=2.00)
        + _rows("1xBet", home=1.95, draw=3.60, away=4.30, over=1.90,
                under=2.00))


def test_FLAG_OFF_the_recommendations_are_BYTE_IDENTICAL():
    """Three configs that all mean `standard` -- the key absent, the key
    explicit, and the deployed file -- against `find_value_bets` called
    directly. The known-wrong half: with the flag ON the result differs, so
    the equality is not of two paths that both do nothing."""
    deployed = _cfg()
    direct = _agent(deployed, GOOD).value_calculator.find_value_bets(
        PRED, _odds_data(GOOD), "A vs B", {}, home_team_name="A",
        away_team_name="B", match_id=7, league="england/premier-league")
    assert direct, "the standard rule returned nothing: the equality is vacuous"
    absent = _cfg(drop=("betting.selection_mode",))
    assert absent.get("betting.selection_mode") is None
    for config in (deployed,
                   _cfg({"betting.selection_mode": "standard"}),
                   absent):
        recs, confident = _select(config, GOOD)
        assert confident is False
        assert repr(recs) == repr(direct)
    on, confident = _select(
        _cfg({"betting.selection_mode": "confident_value",
              "betting.confident_value.p_min": 0.50}), GOOD)
    assert confident is True
    assert on and repr(on) != repr(direct)


def test_FLAG_ON_one_pick_at_or_above_1_70_and_at_or_above_P_MIN():
    config = _cfg({"betting.selection_mode": "confident_value",
                   "betting.confident_value.p_min": 0.50})
    recs, confident = _select(config, GOOD)
    assert confident is True
    assert len(recs) == 1, "at most one pick per fixture"
    rec = recs[0]
    assert rec.odds >= 1.70
    assert rec.predicted_probability >= 0.50
    assert rec.selection in cv.PICKABLE
    assert rec.match_id == 7
    assert rec.market_books == 3


def test_FLAG_ON_raising_P_MIN_above_every_leg_gives_ZERO_PICKS_no_error():
    config = _cfg({"betting.selection_mode": "confident_value",
                   "betting.confident_value.p_min": 0.58})
    high, _ = _select(_cfg({"betting.selection_mode": "confident_value",
                            "betting.confident_value.p_min": 0.50}), GOOD)
    assert high and high[0].predicted_probability < 0.58
    recs, confident = _select(config, GOOD)
    assert recs == [] and confident is True


def test_FLAG_ON_a_price_below_1_70_is_never_picked():
    short = (_rows("Pinnacle", home=1.40, draw=4.90, away=8.00, over=1.45,
                   under=2.85)
             + _rows("Bet365", home=1.42, draw=4.75, away=7.50, over=1.44,
                     under=2.80))
    config = _cfg({"betting.selection_mode": "confident_value",
                   "betting.confident_value.p_min": 0.50})
    recs, _ = _select(config, short)
    for rec in recs:
        assert rec.odds >= 1.70
    # the favourite at 1.40 has by far the highest probability and is NOT it
    assert all(rec.selection != "Home Win" for rec in recs)


def test_FLAG_ON_no_anchor_no_odds_and_no_timestamp_each_give_ZERO_PICKS():
    config = _cfg({"betting.selection_mode": "confident_value",
                   "betting.confident_value.p_min": 0.50})
    assert _select(config, [])[0] == []
    assert _select(config, _rows("Bet365") + _rows("1xBet"))[0] == []
    untimed = [(b, m, s, v, None) for b, m, s, v, _t in GOOD]
    assert _select(config, untimed)[0] == []


def test_FLAG_ON_the_FORCED_PICK_does_not_refill_a_zero_pick_fixture():
    """From the AST of the analysis function: the forced-pick block is
    guarded on the confident flag, so a fixture where nothing qualifies
    stays at zero picks instead of being topped up by the old rule."""
    src = pathlib.Path("src/agent/betting_agent.py").read_text(
        encoding="utf-8")
    tree = ast.parse(src)
    guards = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        if any(isinstance(c, ast.Call)
               and getattr(c.func, "attr", "") == "_should_force_pick"
               for c in ast.walk(node)):
            guards.append(ast.get_source_segment(src, node.test) or "")
    assert guards, "the forced-pick block was not found"
    assert any("_confident" in g and "not recommendations" in g
               for g in guards), guards


def test_a_ZERO_PICK_DAY_puts_NOTHING_in_front_of_the_client():
    """The client channel carries the card, the settlement and performance.
    An empty card is not one of them: `send_daily_picks([])` sends the
    client nothing, and what it does send goes to the OPERATOR channel."""
    import asyncio

    from src.reporting import telegram_bot as tb

    class _N(tb.TelegramNotifier):
        def __init__(self):
            self.config = None
            self.enabled = True
            self.bot_token = "t"
            self.chat_id = "CLIENT"
            self.operator_chat_id = "OPERATOR"
            self._bot = None
            self._last_send_error = ""
            self.client, self.operator = [], []

        def _get_bot(self):
            return object()

        async def _send_chunked(self, message, header=None, report="", **_k):
            self.client.append(message)

        async def _send_message(self, text, **_k):
            self.client.append(text)

        async def send_alert(self, text, **_k):
            self.operator.append(text)

    n = _N()
    asyncio.new_event_loop().run_until_complete(n.send_daily_picks([]))
    assert n.client == [], n.client
    assert len(n.operator) == 1


def test_the_ZERO_PICK_LINE_cannot_raise_whatever_the_config_holds():
    """Found by `test_sharding`, which runs `main()` behind a mocked config.
    The branch that words the no-picks line first called `mode_of()`, which
    REFUSES an unknown value -- so a zero-pick day could raise, the one day
    the directive says must not error. From the AST: the no-picks branch
    compares the raw config value and calls nothing that can refuse."""
    src = pathlib.Path("src/agent/betting_agent.py").read_text(
        encoding="utf-8")
    tree = ast.parse(src)
    branch = None
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and (
                ast.get_source_segment(src, node.test) or "") == "not picks":
            printed = [c for c in ast.walk(node) if isinstance(c, ast.Constant)
                       and c.value == "No value picks found for today."]
            if printed:
                branch = node
    assert branch is not None, "the no-picks branch was not found"
    calls = {getattr(c.func, "attr", None) or getattr(c.func, "id", "")
             for stmt in branch.body
             for c in ast.walk(stmt) if isinstance(c, ast.Call)}
    assert calls == {"get", "print"}, (
        f"the no-picks branch calls {sorted(calls)}; it must only read the "
        f"config, never validate it")
    # the two lines it can print, and the standard one is unchanged
    texts = "".join(c.value for stmt in branch.body for c in ast.walk(stmt)
                    if isinstance(c, ast.Constant) and isinstance(c.value, str))
    assert "No value picks found for today." in texts
    assert "not a failure" in texts
