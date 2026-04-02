"""Value betting calculator and bet recommendation engine."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger()


@dataclass
class BetRecommendation:
    """A single bet recommendation with full analysis."""
    match: str
    match_id: int         # Database match ID for result tracking
    market: str           # '1X2', 'Over 2.5', 'BTTS', etc.
    selection: str        # 'Home Win', 'Over 2.5 Goals', 'BTTS Yes', etc.
    odds: float
    predicted_probability: float
    expected_value: float
    confidence: float
    kelly_stake_percentage: float
    recommended_stake: float
    reasoning: str
    risk_level: str       # 'low', 'medium', 'high'
    injury_impact: str = ""
    h2h_insight: str = ""
    form_insight: str = ""
    news_insight: str = ""
    used_fallback_odds: bool = False
    league: str = ""
    # Enhanced decision support fields
    model_agreement: str = ""       # 'unanimous', 'majority', 'split'
    models_for: str = ""            # e.g. "Poisson, Elo, ML"
    models_against: str = ""        # e.g. "Elo"
    home_xg_avg: float = 0.0       # Rolling xG average (home team)
    away_xg_avg: float = 0.0       # Rolling xG average (away team)
    xg_edge: str = ""              # xG-based insight string
    predicted_xg: str = ""         # e.g. "1.45 - 0.92"
    match_date: Optional[datetime] = None  # Kickoff time (UTC)
    contrarian_value: float = 0.0          # Model-vs-market divergence ratio (>1.3 = contrarian)


class ValueBettingCalculator:
    """Identifies value bets by comparing predicted probabilities to bookmaker odds."""

    def __init__(self, config=None):
        self.config = config or get_config()
        betting = self.config.betting
        self.min_odds = betting.get("min_odds", 1.30)
        self.max_odds = betting.get("max_odds", 10.0)
        self.min_ev = betting.get("min_expected_value", 0.03)       # 3% — professional standard
        self.min_confidence = betting.get("min_confidence", 0.58)   # 58% minimum model probability
        self.high_ev_min_confidence = 0.45  # hard floor — never go below 45% even with high EV
        # Sliding scale: EV × confidence combined score threshold.
        # Allows slightly sub-threshold confidence when EV compensates (and vice versa).
        # e.g. 56% conf + 7% EV = 0.039 → passes; 46% conf + 3.5% EV = 0.016 → rejected.
        self.min_ev_confidence_score = betting.get("min_ev_confidence_score", 0.035)
        self.kelly_fraction = betting.get("kelly_fraction", 0.25)
        self.max_stake_pct = betting.get("max_stake_percentage", 5.0)
        _VALID_MARKET_KEYS = {
            "home_win", "draw", "away_win",
            "over_1.5", "over_2.5", "over_3.5",
            "under_1.5", "under_2.5", "under_3.5",
            "btts_yes", "btts_no",
            "home_over_1.5", "away_over_1.5",
        }
        raw_excluded = set(betting.get("excluded_markets", []))
        unknown = raw_excluded - _VALID_MARKET_KEYS
        if unknown:
            from src.utils.logger import get_logger as _get_logger
            _get_logger().warning(
                f"Unknown excluded_markets keys (ignored): {sorted(unknown)}. "
                f"Valid keys: {sorted(_VALID_MARKET_KEYS)}"
            )
        self.excluded_markets = raw_excluded & _VALID_MARKET_KEYS
        # Reject picks where Kelly recommends less than this % of bankroll
        # (too marginal to justify the variance — usually <0.5% means near-zero edge)
        self.min_kelly_stake = betting.get("min_kelly_stake", 0.5)

    def find_value_bets(self, predictions: Dict, odds_data: List[Dict],
                        match_name: str = "",
                        context: Dict = None,
                        home_team_name: str = "",
                        away_team_name: str = "",
                        match_id: int = 0,
                        league: str = "") -> List[BetRecommendation]:
        """Find value bets by comparing predictions to available odds.

        Args:
            predictions: Ensemble prediction output
            odds_data: List of odds records from database
            match_name: Display name for the match
            context: Optional dict with injury/form/h2h/news insights
            home_team_name: Name of the home team (for odds matching)
            away_team_name: Name of the away team (for odds matching)

        Returns:
            List of BetRecommendation objects sorted by EV
        """
        context = context or {}
        ensemble = predictions.get("ensemble", {})
        recommendations = []
        _rejected: dict = {}  # reason → count for INFO-level summary

        def _reject(reason: str):
            _rejected[reason] = _rejected.get(reason, 0) + 1

        # Define markets to check
        markets = [
            ("1X2", "Home Win", ensemble.get("home_win", 0), "home_win"),
            ("1X2", "Draw", ensemble.get("draw", 0), "draw"),
            ("1X2", "Away Win", ensemble.get("away_win", 0), "away_win"),
            ("Over 1.5", "Over 1.5 Goals", ensemble.get("over_1.5", 0), "over_1.5"),
            ("Over 2.5", "Over 2.5 Goals", ensemble.get("over_2.5", 0), "over_2.5"),
            ("Over 3.5", "Over 3.5 Goals", ensemble.get("over_3.5", 0), "over_3.5"),
            ("Under 1.5", "Under 1.5 Goals", ensemble.get("under_1.5", 0), "under_1.5"),
            ("Under 2.5", "Under 2.5 Goals", ensemble.get("under_2.5", 0), "under_2.5"),
            ("Under 3.5", "Under 3.5 Goals", ensemble.get("under_3.5", 0), "under_3.5"),
            ("BTTS", "BTTS Yes", ensemble.get("btts_yes", 0), "btts_yes"),
            ("BTTS", "BTTS No", ensemble.get("btts_no", 0), "btts_no"),
            # Team goal line markets
            ("Team Goals", "Home Over 1.5", ensemble.get("home_over_1.5", 0), "home_over_1.5"),
            ("Team Goals", "Away Over 1.5", ensemble.get("away_over_1.5", 0), "away_over_1.5"),
        ]

        for market, selection, prob, market_key in markets:
            if market_key in self.excluded_markets:
                continue
            if prob < self.high_ev_min_confidence:
                _reject("prob<45%")
                continue  # hard floor — never go below 45%

            # Find best odds for this market/selection
            best_odds = self._find_best_odds(
                odds_data, market, selection,
                home_team_name=home_team_name, away_team_name=away_team_name,
            )

            # Skip selections with no real bookmaker odds — estimated odds
            # cannot be used for value betting since there is no market to beat.
            is_fallback = False
            if not best_odds:
                logger.debug(
                    f"No real odds for {match_name} {selection} "
                    f"(prob={prob:.0%}) — skipping (no bookmaker data)"
                )
                _reject("no odds")
                continue

            if not best_odds or best_odds < self.min_odds or best_odds > self.max_odds:
                _reject("odds out of range")
                continue

            ev = self.calculate_expected_value(prob, best_odds)
            if ev < self.min_ev:
                _reject("low EV")
                continue

            # Sliding scale: allow bets below min_confidence when EV compensates.
            # EV × confidence must exceed min_ev_confidence_score (default 0.035).
            # This replaces the old hard 10% EV bypass — now a smooth curve where
            # high EV compensates for slightly lower confidence and vice versa.
            if prob < self.min_confidence:
                ev_conf_score = ev * prob
                if ev_conf_score < self.min_ev_confidence_score:
                    _reject("low EV×conf")
                    continue

            # Model-market divergence: compute how much our model disagrees
            # with the bookmaker.  Used as both a guard (reject >2x) and as a
            # contrarian signal (1.3x–2.0x = genuine edge territory).
            implied_prob = 1.0 / best_odds if best_odds > 0 else 0
            divergence = prob / implied_prob if implied_prob > 0 else 0
            if not is_fallback and divergence > 2.0:
                logger.debug(
                    f"Rejecting {match_name} {selection}: model {prob:.0%} vs "
                    f"market {implied_prob:.0%} ({divergence:.1f}x divergence > 2.0x)"
                )
                _reject("divergence>2x")
                continue

            # Model agreement analysis (before Kelly so we can scale stake)
            agreement_info = self._check_model_agreement(
                predictions, market_key, selection,
                features_dict=context.get("features_dict"),
            )

            # Reject split-model picks that are also below minimum confidence.
            # High EV alone is not sufficient when models disagree AND confidence
            # is weak — this combination typically indicates miscalibration or
            # insufficient historical data rather than a genuine edge.
            if agreement_info.get("agreement") == "split" and prob < self.min_confidence:
                logger.debug(
                    f"Rejecting {match_name} {selection}: split models + "
                    f"confidence {prob:.0%} < {self.min_confidence:.0%}"
                )
                _reject("split+low conf")
                continue

            # Uncertainty-aware Kelly: scale stake by model agreement
            kelly_pct = self.kelly_criterion(
                prob, best_odds,
                agreement=agreement_info.get("agreement"),
            )
            # Skip bets where Kelly recommends a trivially small stake —
            # these are on the edge of profitability and not worth the variance
            if kelly_pct < self.min_kelly_stake:
                logger.debug(
                    f"Skipping {match_name} {selection}: "
                    f"Kelly {kelly_pct:.2f}% < min {self.min_kelly_stake}%"
                )
                _reject("low Kelly")
                continue
            risk = self._assess_risk(prob, best_odds, ev)

            # xG insight
            xg_info = self._build_xg_insight(context, ensemble, market, selection)

            reasoning = self._build_reasoning(
                market, selection, prob, best_odds, ev, context,
                agreement_info, xg_info,
            )

            rec = BetRecommendation(
                match=match_name,
                match_id=match_id,
                market=market,
                selection=selection,
                odds=best_odds,
                predicted_probability=round(prob, 4),
                expected_value=round(ev, 4),
                confidence=round(prob, 4),
                kelly_stake_percentage=round(kelly_pct, 2),
                recommended_stake=round(kelly_pct, 2),
                reasoning=reasoning,
                risk_level=risk,
                injury_impact=context.get("injury_impact", ""),
                h2h_insight=context.get("h2h_insight", ""),
                form_insight=context.get("form_insight", ""),
                news_insight=context.get("news_insight", ""),
                used_fallback_odds=is_fallback,
                league=league,
                model_agreement=agreement_info.get("agreement", ""),
                models_for=agreement_info.get("models_for", ""),
                models_against=agreement_info.get("models_against", ""),
                home_xg_avg=context.get("home_xg_avg", 0.0),
                away_xg_avg=context.get("away_xg_avg", 0.0),
                xg_edge=xg_info.get("insight", ""),
                predicted_xg=xg_info.get("predicted_xg", ""),
                contrarian_value=round(divergence, 2),
            )
            recommendations.append(rec)

        # Sort by EV * confidence * agreement bonus (best bets first).
        # Unanimous picks are promoted; split-model picks are demoted.
        # This ensures portfolio caps consume slots on the highest-conviction bets first.
        _agreement_bonus = {"unanimous": 1.15, "majority": 1.0, "split": 0.85, "unknown": 0.95}
        recommendations.sort(
            key=lambda r: r.expected_value * r.confidence
            * _agreement_bonus.get(r.model_agreement, 1.0),
            reverse=True,
        )
        if _rejected and not recommendations:
            _summary = ", ".join(f"{v} {k}" for k, v in sorted(_rejected.items(), key=lambda x: -x[1]))
            logger.info(f"  {match_name}: 0 picks — rejected: {_summary}")
        return recommendations

    @staticmethod
    def calculate_expected_value(probability: float, odds: float) -> float:
        """Calculate expected value of a bet.

        EV = (probability * odds) - 1
        Positive EV means the bet has value.
        """
        return (probability * odds) - 1.0

    # Scale Kelly stake by model agreement level
    _KELLY_AGREEMENT_SCALE = {
        "unanimous": 1.0,
        "majority": 0.80,
        "split": 0.60,
        "unknown": 0.75,
    }

    def kelly_criterion(self, probability: float, odds: float,
                        agreement: str = None) -> float:
        """Calculate optimal stake using fractional Kelly Criterion.

        Kelly % = (bp - q) / b
        where b = odds - 1, p = win probability, q = 1 - p

        When agreement is provided, the Kelly fraction is further scaled down
        for non-unanimous model agreement to reduce exposure on uncertain bets.

        Returns percentage of bankroll to stake (capped at max_stake_pct).
        """
        b = odds - 1.0
        p = probability
        q = 1.0 - p

        if b <= 0:
            return 0.0

        kelly = (b * p - q) / b

        if kelly <= 0:
            return 0.0

        # Apply fractional Kelly
        fractional = kelly * self.kelly_fraction

        # Scale by model agreement confidence
        if agreement:
            config_scale = self.config.get("betting.kelly_agreement_scale", {})
            scale = config_scale.get(
                agreement,
                self._KELLY_AGREEMENT_SCALE.get(agreement, 1.0),
            )
            fractional *= scale

        return min(fractional * 100, self.max_stake_pct)

    def _find_best_odds(self, odds_data: List[Dict], market: str,
                        selection: str,
                        home_team_name: str = "",
                        away_team_name: str = "") -> float:
        """Find the best (highest) odds for a given market and selection."""
        best = 0.0

        # Map our selection names to what's stored in the odds table
        # Odds API stores team names for 1X2, so include them
        selection_map = {
            # "Home Win" / "Away Win" are stored by Flashscore scraper;
            # legacy Odds-API records use "1"/"2" or team names.
            "Home Win": ["1", "Home", "home_win", "Home Win", home_team_name],
            "Draw": ["X", "Draw", "draw", "Draw"],
            "Away Win": ["2", "Away", "away_win", "Away Win", away_team_name],
            # Flashscore O/U stored without " Goals" suffix ("Over 2.5", etc.)
            "Over 2.5 Goals": ["Over 2.5", "Over", "Over 2.5 Goals"],
            "Under 2.5 Goals": ["Under 2.5", "Under", "Under 2.5 Goals"],
            "Over 1.5 Goals": ["Over 1.5", "Over 1.5 Goals"],
            "Under 1.5 Goals": ["Under 1.5", "Under 1.5 Goals"],
            "Over 3.5 Goals": ["Over 3.5", "Over 3.5 Goals"],
            "Under 3.5 Goals": ["Under 3.5", "Under 3.5 Goals"],
            "BTTS Yes": ["Yes", "BTTS Yes"],
            "BTTS No": ["No", "BTTS No"],
            # Team goal line markets (stored via API-Football team_goals market)
            "Home Over 1.5": ["Home Over 1.5", "Home Team Over 1.5", "Home - Over 1.5"],
            "Away Over 1.5": ["Away Over 1.5", "Away Team Over 1.5", "Away - Over 1.5"],
        }

        valid_selections = selection_map.get(selection, [selection])
        # Filter out empty strings
        valid_selections = [s for s in valid_selections if s]

        market_map = {
            "1X2": ["1X2", "h2h"],
            "Over 2.5": ["over_under", "totals"],
            "Under 2.5": ["over_under", "totals"],
            "Over 1.5": ["over_under", "totals"],
            "Under 1.5": ["over_under", "totals"],
            "Over 3.5": ["over_under", "totals"],
            "Under 3.5": ["over_under", "totals"],
            "BTTS": ["btts"],
            "Team Goals": ["team_goals", "team_over_under"],
        }

        valid_markets = market_map.get(market, [market])

        # Bookmakers excluded from EV odds: Flashscore stores display/fair odds,
        # not real betting lines — using them inflates EV calculations.
        _EXCLUDED_BOOKMAKERS = {"Flashscore"}

        for odds_record in odds_data:
            if odds_record.get("bookmaker", "") in _EXCLUDED_BOOKMAKERS:
                continue
            record_market = odds_record.get("market_type", "")
            record_selection = odds_record.get("selection", "")
            record_odds = odds_record.get("odds_value", 0)

            if record_market in valid_markets and record_selection in valid_selections:
                best = max(best, record_odds)

        return best

    def _assess_risk(self, probability: float, odds: float, ev: float) -> str:
        """Assess the risk level of a bet."""
        if probability >= 0.65 and odds < 2.5:
            return "low"
        elif probability >= 0.50 and odds < 5.0:
            return "medium"
        return "high"

    def _check_model_agreement(self, predictions: Dict, market_key: str,
                                selection: str,
                                features_dict: Dict = None) -> Dict:
        """Check how many models agree on this selection.

        Returns dict with agreement level and model names.
        """
        poisson = predictions.get("poisson", {})
        elo = predictions.get("elo", {})
        ml = predictions.get("ml", {}).get("ml_average") if predictions.get("ml") else None

        models_for = []
        models_against = []

        # For 1X2 markets: check if each model's top pick matches selection
        if market_key in ("home_win", "draw", "away_win"):
            for model_name, pred in [("Poisson", poisson), ("Elo", elo), ("ML", ml)]:
                if not pred:
                    continue
                top = max(["home_win", "draw", "away_win"],
                          key=lambda k: pred.get(k, 0))
                if top == market_key:
                    models_for.append(model_name)
                else:
                    models_against.append(model_name)
        else:
            # Goals/BTTS/team goal line markets — check all available signals
            # 1. Poisson
            if poisson:
                poisson_prob = poisson.get(market_key, 0)
                # For under/no markets, always derive from the complementary over/yes market
                if market_key.startswith("under_") or market_key == "btts_no":
                    complement_key = market_key.replace("under_", "over_").replace("btts_no", "btts_yes")
                    complement_prob = poisson.get(complement_key, 0)
                    if complement_prob > 0:
                        poisson_prob = 1.0 - complement_prob
                if poisson_prob > 0.50:
                    models_for.append("Poisson")
                else:
                    models_against.append("Poisson")

            # 2. GoalsML (available for over_2.5 / under_2.5)
            goals_ml_prob = predictions.get("goals_ml_over25")
            if goals_ml_prob is not None and market_key in ("over_2.5", "under_2.5"):
                effective = goals_ml_prob if market_key == "over_2.5" else (1.0 - goals_ml_prob)
                if effective > 0.50:
                    models_for.append("GoalsML")
                else:
                    models_against.append("GoalsML")

            # 3. Bookmaker implied probability
            if features_dict:
                _bk_map = {
                    "over_2.5": "over25_implied_prob",
                    "under_2.5": "under25_implied_prob",
                    "over_1.5": "over15_implied_prob",
                    "under_1.5": "under15_implied_prob",
                    "btts_yes": "btts_yes_implied_prob",
                    "btts_no": "btts_no_implied_prob",
                }
                bk_key = _bk_map.get(market_key)
                if bk_key:
                    bk_p = features_dict.get(bk_key, 0)
                    if bk_p > 0:
                        if bk_p > 0.50:
                            models_for.append("Bookmaker")
                        else:
                            models_against.append("Bookmaker")

        total = len(models_for) + len(models_against)
        if total == 0:
            agreement = "unknown"
        elif len(models_against) == 0:
            agreement = "unanimous"
        elif len(models_for) > len(models_against):
            agreement = "majority"
        else:
            agreement = "split"

        return {
            "agreement": agreement,
            "models_for": ", ".join(models_for),
            "models_against": ", ".join(models_against),
        }

    def _build_xg_insight(self, context: Dict, ensemble: Dict,
                           market: str, selection: str) -> Dict:
        """Build xG-based decision insight."""
        home_xg = ensemble.get("home_xg", 0)
        away_xg = ensemble.get("away_xg", 0)
        predicted_xg = f"{home_xg:.2f} - {away_xg:.2f}" if home_xg or away_xg else ""

        insight_parts = []

        # xG overperformance warning
        home_overperf = context.get("home_xg_overperformance", 0)
        away_overperf = context.get("away_xg_overperformance", 0)

        if home_overperf > 0.5:
            insight_parts.append("Home overperforming xG (regression risk)")
        elif home_overperf < -0.5:
            insight_parts.append("Home underperforming xG (due a bounce)")

        if away_overperf > 0.5:
            insight_parts.append("Away overperforming xG (regression risk)")
        elif away_overperf < -0.5:
            insight_parts.append("Away underperforming xG (due a bounce)")

        # xG-based goal insight
        total_xg = home_xg + away_xg
        if market in ("Over 2.5", "Over 1.5", "Over 3.5", "BTTS"):
            if total_xg > 2.8:
                insight_parts.append(f"High xG total ({total_xg:.1f}) supports goals")
            elif total_xg < 1.8:
                insight_parts.append(f"Low xG total ({total_xg:.1f}) cautionary for goals")

        return {
            "predicted_xg": predicted_xg,
            "insight": ". ".join(insight_parts) if insight_parts else "",
        }

    def _build_reasoning(self, market: str, selection: str, prob: float,
                         odds: float, ev: float, context: Dict,
                         agreement_info: Dict = None,
                         xg_info: Dict = None) -> str:
        """Build a human-readable reasoning string for the bet."""
        parts = [
            f"{selection} predicted at {prob:.0%} probability",
            f"with odds of {odds:.2f} giving {ev:.1%} expected value.",
        ]

        # Model agreement
        if agreement_info and agreement_info.get("agreement"):
            agr = agreement_info["agreement"]
            if agr == "unanimous":
                parts.append(f"All models agree ({agreement_info['models_for']}).")
            elif agr == "majority":
                parts.append(
                    f"Majority of models agree ({agreement_info['models_for']}); "
                    f"{agreement_info['models_against']} disagrees."
                )
            elif agr == "split":
                parts.append(
                    f"Models split: {agreement_info['models_for']} for, "
                    f"{agreement_info['models_against']} against."
                )

        # xG insight
        if xg_info and xg_info.get("insight"):
            parts.append(xg_info["insight"] + ".")

        if context.get("form_insight"):
            parts.append(context["form_insight"])
        if context.get("h2h_insight"):
            parts.append(context["h2h_insight"])
        if context.get("injury_impact"):
            parts.append(context["injury_impact"])

        return " ".join(parts)
