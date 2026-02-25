"""
Ad-hoc review tests for the three new feature groups.
Run with: python tests/_review_new_features.py
"""
import numpy as np
from unittest.mock import MagicMock

print("=" * 60)
print("A. _get_bookmaker_features() — extended markets")
print("=" * 60)

from src.features.feature_engineer import FeatureEngineer

fe = FeatureEngineer.__new__(FeatureEngineer)
fe.db = MagicMock()


def make_odds_row(market_type, bookmaker, selection, odds_value):
    r = MagicMock()
    r.market_type = market_type
    r.bookmaker = bookmaker
    r.selection = selection
    r.odds_value = odds_value
    return r


# Realistic odds for a match
test_rows = [
    make_odds_row("1X2",        "Bet365",        "Home Win",       1.80),
    make_odds_row("1X2",        "Bet365",        "Draw",           3.60),
    make_odds_row("1X2",        "Bet365",        "Away Win",       4.20),
    make_odds_row("over_under", "Bet365",        "Over 2.5",       1.75),
    make_odds_row("over_under", "Bet365",        "Under 2.5",      2.10),
    make_odds_row("over_under", "Bet365",        "Over 1.5",       1.30),
    make_odds_row("over_under", "Bet365",        "Under 1.5",      3.40),
    make_odds_row("btts",       "Bet365",        "Yes",            1.85),
    make_odds_row("btts",       "Bet365",        "No",             1.95),
    make_odds_row("team_goals", "API-Football",  "Home Over 1.5",  1.60),
    make_odds_row("team_goals", "API-Football",  "Home Under 1.5", 2.30),
    make_odds_row("team_goals", "API-Football",  "Away Over 1.5",  2.10),
    make_odds_row("team_goals", "API-Football",  "Away Under 1.5", 1.70),
]

mock_session = MagicMock()
mock_session.query.return_value.filter.return_value.all.return_value = test_rows
fe.db.get_session.return_value.__enter__ = MagicMock(return_value=mock_session)
fe.db.get_session.return_value.__exit__ = MagicMock(return_value=False)

result = fe._get_bookmaker_features(42)

# Expected values via margin removal
rh, rd, ra = 1/1.80, 1/3.60, 1/4.20
m3 = rh + rd + ra
exp_h  = round(rh/m3, 4)
exp_d  = round(rd/m3, 4)
exp_a  = round(ra/m3, 4)

ro25, ru25 = 1/1.75, 1/2.10
exp_o25 = round(ro25/(ro25+ru25), 4)

ro15, ru15 = 1/1.30, 1/3.40
exp_o15 = round(ro15/(ro15+ru15), 4)

ry, rn = 1/1.85, 1/1.95
exp_btts = round(ry/(ry+rn), 4)

rho, rhu = 1/1.60, 1/2.30
exp_home_o15 = round(rho/(rho+rhu), 4)

rao, rau = 1/2.10, 1/1.70
exp_away_o15 = round(rao/(rao+rau), 4)

all_ok = True
checks = [
    ("bookmaker_available",         result["bookmaker_available"],          1),
    ("home_implied_prob",           result["home_implied_prob"],            exp_h),
    ("draw_implied_prob",           result["draw_implied_prob"],            exp_d),
    ("away_implied_prob",           result["away_implied_prob"],            exp_a),
    ("1X2 probs sum to 1",          round(result["home_implied_prob"] +
                                          result["draw_implied_prob"] +
                                          result["away_implied_prob"], 4),  1.0),
    ("goals_bookmaker_available",   result["goals_bookmaker_available"],    1),
    ("over25_implied_prob",         result["over25_implied_prob"],          exp_o25),
    ("over25+under25 = 1",          round(result["over25_implied_prob"] +
                                          result["under25_implied_prob"], 4), 1.0),
    ("over15_implied_prob",         result["over15_implied_prob"],          exp_o15),
    ("over15+under15 = 1",          round(result["over15_implied_prob"] +
                                          result["under15_implied_prob"], 4), 1.0),
    ("btts_bookmaker_available",    result["btts_bookmaker_available"],     1),
    ("btts_yes_implied_prob",       result["btts_yes_implied_prob"],        exp_btts),
    ("btts_yes+no = 1",             round(result["btts_yes_implied_prob"] +
                                          result["btts_no_implied_prob"], 4), 1.0),
    ("team_goals_bookmaker_avail",  result["team_goals_bookmaker_available"], 1),
    ("home_over15_implied_prob",    result["home_over15_implied_prob"],     exp_home_o15),
    ("away_over15_implied_prob",    result["away_over15_implied_prob"],     exp_away_o15),
]

for name, got, expected in checks:
    ok = abs(float(got) - float(expected)) < 0.0001
    if not ok:
        all_ok = False
    print(f"  {'OK' if ok else 'FAIL'}  {name}: {got} (expected {expected})")

# No rows → safe defaults
mock_session.query.return_value.filter.return_value.all.return_value = []
defaults = fe._get_bookmaker_features(99)
assert defaults["bookmaker_available"] == 0
assert defaults["goals_bookmaker_available"] == 0
assert abs(defaults["home_implied_prob"] - 1/3) < 0.001
print("  OK  no-rows -> safe defaults returned")

# Bet365 preferred over API-Football
rows_mixed = [
    make_odds_row("over_under", "API-Football", "Over 2.5",  2.00),
    make_odds_row("over_under", "API-Football", "Under 2.5", 1.90),
    make_odds_row("over_under", "Bet365",       "Over 2.5",  1.75),
    make_odds_row("over_under", "Bet365",       "Under 2.5", 2.10),
]
mock_session.query.return_value.filter.return_value.all.return_value = rows_mixed
r = fe._get_bookmaker_features(77)
assert abs(r["over25_implied_prob"] - exp_o25) < 0.001, \
    f"Bet365 not preferred: got {r['over25_implied_prob']}, expected {exp_o25}"
print("  OK  Bet365 preferred over API-Football for over/under")

print()
print("=" * 60)
print("B. _get_league_features() — league baseline rates")
print("=" * 60)


def make_match(hg, ag):
    m = MagicMock()
    m.home_goals = hg
    m.away_goals = ag
    return m


# 10 matches: 6×(2-1), 2×(1-1), 2×(0-2)
# home_wins=6, draws=2, away_wins=2
# avg_goals = (6*3 + 2*2 + 2*2)/10 = (18+4+4)/10 = 2.6
# over_2.5 (>2 total): 2-1=3>2 yes (6), 1-1=2 no, 0-2=2 no -> 6/10=0.6
# btts: 2-1 yes (6), 1-1 yes (2), 0-2 no (2) -> 8/10=0.8
test_matches = [make_match(2, 1)] * 6 + [make_match(1, 1)] * 2 + [make_match(0, 2)] * 2

mock_session2 = MagicMock()
mock_session2.query.return_value \
             .filter.return_value \
             .order_by.return_value \
             .limit.return_value \
             .all.return_value = test_matches

fe2 = FeatureEngineer.__new__(FeatureEngineer)
fe2.db = MagicMock()
fe2.db.get_session.return_value.__enter__ = MagicMock(return_value=mock_session2)
fe2.db.get_session.return_value.__exit__ = MagicMock(return_value=False)

lf = fe2._get_league_features("england/premier-league")
checks_b = [
    ("league_home_win_rate",  lf["league_home_win_rate"],  0.6),
    ("league_draw_rate",      lf["league_draw_rate"],       0.2),
    ("league_away_win_rate",  lf["league_away_win_rate"],   0.2),
    ("league_avg_goals",      lf["league_avg_goals"],       2.6),
    ("league_over25_rate",    lf["league_over25_rate"],     0.6),
    ("league_btts_rate",      lf["league_btts_rate"],       0.8),
    ("league_matches_count",  lf["league_matches_count"],   10),
]
for name, got, expected in checks_b:
    ok = abs(float(got) - float(expected)) < 0.0001
    if not ok:
        all_ok = False
    print(f"  {'OK' if ok else 'FAIL'}  {name}: {got} (expected {expected})")

lf2 = fe2._get_league_features("england/premier-league")
assert lf2 is lf, "Cache miss — expected same dict object"
print("  OK  cache hit returns same dict")

# < 10 matches → defaults
mock_session2.query.return_value \
             .filter.return_value \
             .order_by.return_value \
             .limit.return_value \
             .all.return_value = test_matches[:5]
fe3 = FeatureEngineer.__new__(FeatureEngineer)
fe3.db = MagicMock()
fe3.db.get_session.return_value.__enter__ = MagicMock(return_value=mock_session2)
fe3.db.get_session.return_value.__exit__ = MagicMock(return_value=False)
lf_def = fe3._get_league_features("rare/league")
assert lf_def["league_matches_count"] == 0, "Expected defaults for < 10 matches"
print("  OK  < 10 matches -> defaults returned")

print()
print("=" * 60)
print("C. Ensemble bookmaker blend for goals markets")
print("=" * 60)

from src.models.ensemble import EnsemblePredictor

ens = EnsemblePredictor.__new__(EnsemblePredictor)
ens.config = MagicMock()
ens.config.get = lambda key, default=None: {
    "models.ensemble_weights": {},
    "models.bookmaker_blend_weight": 0.40,
}.get(key, default)
ens.weights = {"poisson": 0.25, "elo": 0.20, "xgboost": 0.35, "random_forest": 0.20}
ens.ml_models = MagicMock()
ens.ml_models.is_fitted = False
ens.goals_model = MagicMock()
ens.goals_model.is_fitted = False  # disabled in review test — GoalsMLModel blend not tested here

mock_poisson = MagicMock()
mock_poisson._team_strengths = {1: 1.0, 2: 1.0}
mock_poisson.predict.return_value = {
    "home_win": 0.40, "draw": 0.30, "away_win": 0.30,
    "over_1.5": 0.80, "over_2.5": 0.60, "over_3.5": 0.30,
    "under_2.5": 0.40,
    "btts_yes": 0.55, "btts_no": 0.45,
    "home_over_1.5": 0.65, "away_over_1.5": 0.50,
    "home_xg": 1.5, "away_xg": 1.1,
    "most_likely_score": "1-1",
}
mock_elo = MagicMock()
mock_elo.ratings = {1: 1500, 2: 1500}
mock_elo.predict.return_value = {"home_win": 0.40, "draw": 0.30, "away_win": 0.30}
ens.poisson = mock_poisson
ens.elo = mock_elo

blend = 0.40
bk_o25   = 0.75
bk_o15   = 0.88
bk_btts  = 0.62
bk_h15   = 0.72
bk_a15   = 0.58

feature_names = [
    "goals_bookmaker_available", "over25_implied_prob", "over15_implied_prob",
    "btts_bookmaker_available", "btts_yes_implied_prob",
    "team_goals_bookmaker_available", "home_over15_implied_prob", "away_over15_implied_prob",
]
feature_values = np.array([1.0, bk_o25, bk_o15, 1.0, bk_btts, 1.0, bk_h15, bk_a15])

result_with_bk = ens.predict(1, 2, features_vector=feature_values, feature_names=feature_names)
ens_bk = result_with_bk["ensemble"]

# Poisson btts_yes gets a small competitiveness boost before blending —
# retrieve the no-bookmaker result to get the exact pre-blend btts value.
result_no_bk = ens.predict(1, 2)
ens_no = result_no_bk["ensemble"]
poisson_btts_boosted = ens_no["btts_yes"]  # value before bookmaker blend

# Use the actual pre-blend values from the no-bookmaker path (ensemble applies
# goal_boost/draw_penalty adjustments on top of raw Poisson before the blend).
pre_blend_o25  = ens_no["over_2.5"]
pre_blend_o15  = ens_no["over_1.5"]

exp_o25_blend  = round(pre_blend_o25 * (1 - blend) + bk_o25  * blend, 4)
exp_o15_blend  = round(pre_blend_o15 * (1 - blend) + bk_o15  * blend, 4)
exp_btts_blend = round(poisson_btts_boosted * (1 - blend) + bk_btts * blend, 4)
exp_h15_blend  = round(0.65 * (1 - blend) + bk_h15  * blend, 4)
exp_a15_blend  = round(0.50 * (1 - blend) + bk_a15  * blend, 4)

checks_c = [
    ("over_2.5 blended",      ens_bk["over_2.5"],        exp_o25_blend),
    ("over_1.5 blended",      ens_bk["over_1.5"],        exp_o15_blend),
    ("btts_yes blended",      ens_bk["btts_yes"],        exp_btts_blend),
    ("home_over_1.5 blended", ens_bk["home_over_1.5"],   exp_h15_blend),
    ("away_over_1.5 blended", ens_bk["away_over_1.5"],   exp_a15_blend),
    # complement invariants
    ("over_2.5+under_2.5=1",  round(ens_bk["over_2.5"] + ens_bk["under_2.5"], 4), 1.0),
    ("btts_yes+no=1",         round(ens_bk["btts_yes"]  + ens_bk["btts_no"],  4), 1.0),
]
for name, got, expected in checks_c:
    ok = abs(float(got) - float(expected)) < 0.001
    if not ok:
        all_ok = False
    print(f"  {'OK' if ok else 'FAIL'}  {name}: {got} (expected {expected})")

# Without bookmaker features the blend must NOT apply
no_blend_ok = abs(ens_no["over_2.5"] - exp_o25_blend) > 0.01
if not no_blend_ok:
    all_ok = False
print(f"  {'OK' if no_blend_ok else 'FAIL'}  no-bookmaker path unaffected (over_2.5={ens_no['over_2.5']})")

# 1X2 probs must still sum to 1 after blend
ok_1x2 = abs(ens_bk["home_win"] + ens_bk["draw"] + ens_bk["away_win"] - 1.0) < 0.001
if not ok_1x2:
    all_ok = False
print(f"  {'OK' if ok_1x2 else 'FAIL'}  1X2 probs still sum to 1.0")

print()
print("=" * 60)
print(f"RESULT: {'ALL OK' if all_ok else 'SOME FAILED'}")
print("=" * 60)
