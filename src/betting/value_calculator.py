"""Value betting calculator and bet recommendation engine."""

from dataclasses import dataclass, field
from typing import Dict, List

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


class ValueBettingCalculator:
    """Identifies value bets by comparing predicted probabilities to bookmaker odds."""

    def __init__(self, config=None):
        self.config = config or get_config()
        betting = self.config.betting
        self.min_odds = betting.get("min_odds", 1.30)
        self.max_odds = betting.get("max_odds", 10.0)
        self.min_ev = betting.get("min_expected_value", 0.05)
        self.min_confidence = betting.get("min_confidence", 0.55)
        self.kelly_fraction = betting.get("kelly_fraction", 0.25)
        self.max_stake_pct = betting.get("max_stake_percentage", 5.0)

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

        # Define markets to check (goals-oriented picks only)
        markets = [
            ("1X2", "Home Win", ensemble.get("home_win", 0), "home_win"),
            ("1X2", "Draw", ensemble.get("draw", 0), "draw"),
            ("1X2", "Away Win", ensemble.get("away_win", 0), "away_win"),
            ("Over 1.5", "Over 1.5 Goals", ensemble.get("over_1.5", 0), "over_1.5"),
            ("Over 2.5", "Over 2.5 Goals", ensemble.get("over_2.5", 0), "over_2.5"),
            ("Over 3.5", "Over 3.5 Goals", ensemble.get("over_3.5", 0), "over_3.5"),
            ("BTTS", "BTTS Yes", ensemble.get("btts_yes", 0), "btts_yes"),
        ]

        # Allow lower confidence (45%) for high-EV longshot picks;
        # standard picks still need min_confidence (55%).
        high_ev_min_confidence = 0.45

        for market, selection, prob, market_key in markets:
            if prob < high_ev_min_confidence:
                continue  # hard floor — never go below 45%

            # Find best odds for this market/selection
            best_odds = self._find_best_odds(
                odds_data, market, selection,
                home_team_name=home_team_name, away_team_name=away_team_name,
            )

            # Fallback for markets without odds (e.g. free API tier):
            # use typical bookmaker prices so high-confidence model
            # predictions still appear as picks.
            is_fallback = False
            if not best_odds and prob >= high_ev_min_confidence:
                fallback_odds = {
                    "BTTS Yes": 1.80,
                    "Over 1.5 Goals": 1.45,
                    "Over 2.5 Goals": 1.90,
                    "Over 3.5 Goals": 2.50,
                }
                best_odds = fallback_odds.get(selection, 0)
                if best_odds:
                    is_fallback = True

            if not best_odds or best_odds < self.min_odds or best_odds > self.max_odds:
                continue

            ev = self.calculate_expected_value(prob, best_odds)
            if ev < self.min_ev:
                continue

            # Standard picks need min_confidence; only high-EV (>10%) can use 45-55%
            if prob < self.min_confidence and ev < 0.10:
                continue

            kelly_pct = self.kelly_criterion(prob, best_odds)
            risk = self._assess_risk(prob, best_odds, ev)

            # Model agreement analysis
            agreement_info = self._check_model_agreement(
                predictions, market_key, selection,
            )

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
            )
            recommendations.append(rec)

        # Sort by EV * confidence (best bets first)
        recommendations.sort(key=lambda r: r.expected_value * r.confidence, reverse=True)
        return recommendations

    @staticmethod
    def calculate_expected_value(probability: float, odds: float) -> float:
        """Calculate expected value of a bet.

        EV = (probability * odds) - 1
        Positive EV means the bet has value.
        """
        return (probability * odds) - 1.0

    def kelly_criterion(self, probability: float, odds: float) -> float:
        """Calculate optimal stake using fractional Kelly Criterion.

        Kelly % = (bp - q) / b
        where b = odds - 1, p = win probability, q = 1 - p

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

        # Apply fractional Kelly and cap
        fractional = kelly * self.kelly_fraction
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
            "Home Win": ["1", "Home", "home_win", home_team_name],
            "Draw": ["X", "Draw", "draw"],
            "Away Win": ["2", "Away", "away_win", away_team_name],
            "Over 2.5 Goals": ["Over 2.5", "Over"],
            "Under 2.5 Goals": ["Under 2.5", "Under"],
            "Over 1.5 Goals": ["Over 1.5"],
            "Over 3.5 Goals": ["Over 3.5"],
            "BTTS Yes": ["Yes", "BTTS Yes"],
            "BTTS No": ["No", "BTTS No"],
        }

        valid_selections = selection_map.get(selection, [selection])
        # Filter out empty strings
        valid_selections = [s for s in valid_selections if s]

        market_map = {
            "1X2": ["1X2", "h2h"],
            "Over 2.5": ["over_under", "totals"],
            "Under 2.5": ["over_under", "totals"],
            "Over 1.5": ["over_under", "totals"],
            "Over 3.5": ["over_under", "totals"],
            "BTTS": ["btts"],
        }

        valid_markets = market_map.get(market, [market])

        for odds_record in odds_data:
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
                                selection: str) -> Dict:
        """Check how many models agree on this selection.

        Returns dict with agreement level and model names.
        """
        poisson = predictions.get("poisson", {})
        elo = predictions.get("elo", {})
        ml = predictions.get("ml", {}).get("ml_average") if predictions.get("ml") else None

        models_for = []
        models_against = []

        # For 1X2 markets: check if each model's top pick matches selection
        selection_to_key = {
            "Home Win": "home_win", "Draw": "draw", "Away Win": "away_win",
        }

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
            # Goals/BTTS markets — check if model prob > 50%
            for model_name, pred in [("Poisson", poisson)]:
                if not pred:
                    continue
                model_prob = pred.get(market_key, 0)
                if model_prob > 0.50:
                    models_for.append(model_name)
                else:
                    models_against.append(model_name)

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
