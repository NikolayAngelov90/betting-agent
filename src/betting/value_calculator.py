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
    opening_odds: float = 0.0              # Opening line (first odds seen); 0 if unknown


class ValueBettingCalculator:
    """Identifies value bets by comparing predicted probabilities to bookmaker odds."""

    def __init__(self, config=None):
        self.config = config or get_config()
        betting = self.config.betting
        self.min_odds = betting.get("min_odds", 1.30)
        self.max_odds = betting.get("max_odds", 10.0)
        self.min_ev = betting.get("min_expected_value", 0.05)       # 5% — selective standard
        self.min_confidence = betting.get("min_confidence", 0.55)   # 55% minimum model probability
        self.high_ev_min_confidence = 0.52  # hard floor — model must give clear majority (>50%)
        # Sliding scale: EV × confidence combined score threshold.
        # Allows slightly sub-threshold confidence when EV compensates (and vice versa).
        # e.g. 56% conf + 8% EV = 0.045 → passes; 52% conf + 7% EV = 0.036 → rejected.
        self.min_ev_confidence_score = betting.get("min_ev_confidence_score", 0.038)
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

        markets = self._market_specs(ensemble)

        for market, selection, prob, market_key in markets:
            if market_key in self.excluded_markets:
                continue
            if prob < self.high_ev_min_confidence:
                _reject(f"prob<{self.high_ev_min_confidence:.0%}")
                continue  # hard floor — model probability below absolute minimum

            # Find best (median) odds for this market/selection
            best_odds = self._find_best_odds(
                odds_data, market, selection,
                home_team_name=home_team_name, away_team_name=away_team_name,
            )
            opening = self._find_opening_odds(
                odds_data, market, selection,
                home_team_name=home_team_name, away_team_name=away_team_name,
            )

            # Skip selections with no real bookmaker odds — estimated odds
            # cannot be used for value betting since there is no market to beat.
            if not best_odds:
                logger.debug(
                    f"No real odds for {match_name} {selection} "
                    f"(prob={prob:.0%}) — skipping (no bookmaker data)"
                )
                _reject("no odds")
                continue

            if best_odds < self.min_odds or best_odds > self.max_odds:
                _reject("odds out of range")
                continue

            ev = self.calculate_expected_value(prob, best_odds)
            if ev < self.min_ev:
                _reject("low EV")
                continue

            # Goals-market EV floor: when historical pick calibration shows
            # systematic overestimation (cal_factor < 0.90), require proportionally
            # higher EV to compensate.  Avoids picking goals markets where the model
            # consistently over-predicts but the EV margin is thin.
            _pick_cal = context.get("pick_calibration", {})
            _cal_key_map = {
                "over_2.5": "over_2.5", "under_2.5": "over_2.5",
                "over_1.5": "over_1.5", "under_1.5": "over_1.5",
                "btts_yes": "btts", "btts_no": "btts",
            }
            _cal_key = _cal_key_map.get(market_key)
            if _cal_key and _pick_cal:
                _mkt_cal = _pick_cal.get(_cal_key, 1.0)
                if _mkt_cal < 0.90:
                    _adj_min_ev = min(self.min_ev / _mkt_cal, self.min_ev + 0.05)
                    if ev < _adj_min_ev:
                        logger.debug(
                            f"Goals EV floor: {match_name} {selection} ev={ev:.2%} < "
                            f"adj_min_ev={_adj_min_ev:.2%} (pick_cal={_mkt_cal:.3f})"
                        )
                        _reject("goals_cal_ev_floor")
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
            if divergence > 2.0:
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

            # Extra guard: high-EV picks with below-threshold confidence require
            # unanimous model agreement.  This prevents miscalibrated outliers
            # (e.g. 49% prob × 3.64 odds = 78% EV) from slipping through — such
            # combinations typically signal data sparsity or model disagreement,
            # not a genuine edge. Genuine high-EV bets from confident predictions
            # will have unanimous support and are unaffected.
            if prob < self.min_confidence and ev > 0.50:
                if agreement_info.get("agreement") != "unanimous":
                    logger.debug(
                        f"Rejecting {match_name} {selection}: high EV ({ev:.0%}) "
                        f"but prob {prob:.0%} < min_confidence and agreement="
                        f"{agreement_info.get('agreement')} (not unanimous)"
                    )
                    _reject("high_ev_low_conf_not_unanimous")
                    continue

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

            # xG regression contradiction penalty: when xG signals that a team
            # has been scoring above expected (regression risk) and we are
            # betting on MORE goals / BTTS Yes, dampen the probability to
            # reflect mean-reversion pressure.  Applied before Kelly so that
            # severely penalised picks also fail minimum-stake checks.
            home_overperf = context.get("home_xg_overperformance", 0) or 0
            away_overperf = context.get("away_xg_overperformance", 0) or 0
            _xg_penalty = 0.0
            if market_key in ("over_2.5", "over_1.5", "over_3.5", "btts_yes"):
                # Both teams overperforming → strong regression signal against goals
                if home_overperf > 0.5 and away_overperf > 0.5:
                    _xg_penalty = 0.05
                elif home_overperf > 0.5 or away_overperf > 0.5:
                    _xg_penalty = 0.03
            elif market_key == "home_win" and home_overperf > 1.0:
                _xg_penalty = 0.03
            elif market_key == "away_win" and away_overperf > 1.0:
                _xg_penalty = 0.03
            if _xg_penalty > 0:
                prob_orig = prob
                prob = prob * (1.0 - _xg_penalty)
                ev = self.calculate_expected_value(prob, best_odds)
                logger.debug(
                    f"xG regression penalty ({_xg_penalty:.0%}) on {match_name} "
                    f"{selection}: prob {prob_orig:.1%} → {prob:.1%}, "
                    f"EV → {ev:.1%}"
                )
                # Re-check minimum thresholds after penalty
                if ev < self.min_ev:
                    _reject("xg_regression_penalty")
                    continue
                if prob < self.high_ev_min_confidence:
                    _reject("xg_regression_penalty_low_conf")
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
                used_fallback_odds=False,  # retained for DB compatibility; always False since we skip no-odds selections above
                league=league,
                model_agreement=agreement_info.get("agreement", ""),
                models_for=agreement_info.get("models_for", ""),
                models_against=agreement_info.get("models_against", ""),
                home_xg_avg=context.get("home_xg_avg", 0.0),
                away_xg_avg=context.get("away_xg_avg", 0.0),
                xg_edge=xg_info.get("insight", ""),
                predicted_xg=xg_info.get("predicted_xg", ""),
                contrarian_value=round(divergence, 2),
                opening_odds=round(opening, 2) if opening else 0.0,
            )
            recommendations.append(rec)

        # Sort by EV * confidence * agreement bonus (best bets first).
        # Unanimous picks are promoted; split-model picks are demoted.
        # This ensures portfolio caps consume slots on the highest-conviction bets first.
        _agreement_bonus = {"unanimous": 1.15, "solo": 1.05, "majority": 1.0, "split": 0.85, "unknown": 0.95}
        recommendations.sort(
            key=lambda r: r.expected_value * r.confidence
            * _agreement_bonus.get(r.model_agreement, 1.0),
            reverse=True,
        )
        if _rejected and not recommendations:
            _summary = ", ".join(f"{v} {k}" for k, v in sorted(_rejected.items(), key=lambda x: -x[1]))
            logger.info(f"  {match_name}: 0 picks — rejected: {_summary}")
        return recommendations

    def _market_specs(self, ensemble: Dict):
        """The (market, selection, prob, market_key) tuples evaluated for a match."""
        return [
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
            ("Team Goals", "Home Over 1.5", ensemble.get("home_over_1.5", 0), "home_over_1.5"),
            ("Team Goals", "Away Over 1.5", ensemble.get("away_over_1.5", 0), "away_over_1.5"),
        ]

    def find_best_bet(self, predictions: Dict, odds_data: List[Dict],
                      match_name: str = "", context: Dict = None,
                      home_team_name: str = "", away_team_name: str = "",
                      match_id: int = 0, league: str = "",
                      prefer_market: bool = False) -> Optional[BetRecommendation]:
        """Return the single best bet for a match, bypassing the value and
        confidence gates used by find_value_bets.

        Used for "pick every match" mode (WC): when no selection clears the normal
        value thresholds, this still returns the best bettable selection so the
        match gets a tracked pick. Only HARD sanity gates apply — real odds must
        exist and be in range. Kelly stake is clamped to the minimum so the pick
        carries a real (if small) stake. Returns None if nothing is bettable.

        prefer_market=True (set for thin-data / low-coverage WC fixtures): pick the
        selection the MARKET rates most likely (shortest odds ≥ floor) instead of
        the model's highest-EV selection. Settled data showed the model is
        systematically overconfident and has no edge, so on matches where it has
        almost no history, the market favourite maximises win rate — which is what
        a thin-data prediction should optimise for.
        """
        context = context or {}
        ensemble = predictions.get("ensemble", {})

        # Gather every bettable selection (real odds in range). No confidence
        # floor: this is the "pick every match" fallback, so even-match slates
        # still yield a pick.
        candidates = []  # (market, selection, prob, market_key, odds, divergence, ev)
        for market, selection, prob, market_key in self._market_specs(ensemble):
            if market_key in self.excluded_markets:
                continue
            best_odds = self._find_best_odds(
                odds_data, market, selection,
                home_team_name=home_team_name, away_team_name=away_team_name,
            )
            if not best_odds or best_odds < self.min_odds or best_odds > self.max_odds:
                continue
            implied = 1.0 / best_odds if best_odds > 0 else 0
            divergence = prob / implied if implied > 0 else 0
            ev = self.calculate_expected_value(prob, best_odds)
            candidates.append(
                (market, selection, prob, market_key, best_odds, divergence, ev)
            )

        if not candidates:
            return None

        # Mismatch handling (strong favourite vs weak rival). Settled WC data:
        # BTTS Yes loses in routs because the underdog gets shut out (France 3-0
        # Iraq, Canada 6-0 Qatar, Brazil 3-0 Haiti, Spain 4-0 Saudi). In those
        # games the FAVOURITE'S Over 1.5 / Over 2.5 wins instead. So when one side
        # is a clear favourite, drop BTTS and the underdog's team-total, and pick
        # the favourite's goal markets by win probability.
        hw = ensemble.get("home_win", 0.0)
        aw = ensemble.get("away_win", 0.0)
        fav_prob = max(hw, aw)
        is_mismatch = fav_prob >= 0.60
        if is_mismatch:
            fav_home = hw >= aw
            fav_over_key = "home_over_1.5" if fav_home else "away_over_1.5"
            dog_over_key = "away_over_1.5" if fav_home else "home_over_1.5"
            fav_win_key = "home_win" if fav_home else "away_win"
            # Drop bets that need the weak side to score / win.
            drop = {"btts_yes", dog_over_key,
                    "home_win" if not fav_home else "away_win", "draw"}
            pool = [c for c in candidates if c[3] not in drop] or candidates
            # Preference order: favourite scores 2+ → match has 3+ goals →
            # favourite wins. Pick the first tier that's priced; within it, the
            # higher model probability.
            for tier in ([fav_over_key], ["over_2.5"], [fav_win_key], ["over_1.5"]):
                tier_pool = [c for c in pool if c[3] in tier]
                if tier_pool:
                    best = max(tier_pool, key=lambda c: c[2])
                    market, selection, prob, market_key, best_odds, divergence, ev = best
                    return self._build_pick(
                        predictions, ensemble, context, match_name, match_id, league,
                        odds_data, market, selection, prob, market_key, best_odds,
                        home_team_name=home_team_name, away_team_name=away_team_name,
                    )
            # No favourite-friendly market priced → fall through to generic logic
            # on the reduced pool.
            candidates = pool

        if prefer_market:
            # Thin-data WC: trust the market, not the model's noisy probs. Pick the
            # selection with the highest market-implied probability (shortest odds)
            # at or above the odds floor → maximises win rate. Prefer the core
            # markets (1X2 / Over-Under) over team-goal props for a cleaner call.
            _core = {"home_win", "draw", "away_win",
                     "over_1.5", "over_2.5", "btts_yes"}
            pool = [c for c in candidates if c[3] in _core] or candidates
            best = min(pool, key=lambda c: c[4])  # lowest odds = market favourite
        else:
            # Forced picks must be SANE, not maximal-EV: with thin data the model's
            # probabilities are noisy and max-EV selects the most mispriced longshot
            # (e.g. Away Over 1.5 @ 6.50 on 31% confidence). Tier 1: highest EV among
            # selections with modest odds and modest model-market divergence. Tier 2
            # (nothing modest available): the model's most-likely selection.
            sane = [
                c for c in candidates
                if c[4] <= 3.50 and c[5] <= 1.50  # odds ≤ 3.5, divergence ≤ 1.5x
            ]
            if sane:
                best = max(sane, key=lambda c: c[6])  # highest EV among sane picks
            else:
                best = max(candidates, key=lambda c: c[2])  # highest model probability

        market, selection, prob, market_key, best_odds, divergence, ev = best
        return self._build_pick(
            predictions, ensemble, context, match_name, match_id, league, odds_data,
            market, selection, prob, market_key, best_odds,
            home_team_name=home_team_name, away_team_name=away_team_name,
        )

    def available_selections(self, predictions: Dict, odds_data: List[Dict],
                             home_team_name: str = "", away_team_name: str = "") -> List[Dict]:
        """Return the selections that have real bookmaker odds in range for a match.

        This is the menu the briefing LLM may switch a pick to — it can only choose
        a selection the market actually prices. Each entry: market, selection,
        market_key, model probability, and best odds.
        """
        ensemble = predictions.get("ensemble", {})
        out = []
        for market, selection, prob, market_key in self._market_specs(ensemble):
            if market_key in self.excluded_markets:
                continue
            best_odds = self._find_best_odds(
                odds_data, market, selection,
                home_team_name=home_team_name, away_team_name=away_team_name,
            )
            if not best_odds or best_odds < self.min_odds or best_odds > self.max_odds:
                continue
            out.append({
                "market": market, "selection": selection, "market_key": market_key,
                "probability": round(prob, 4), "odds": round(best_odds, 2),
            })
        return out

    def build_selection_pick(self, predictions: Dict, odds_data: List[Dict],
                             market_key: str, match_name: str = "", context: Dict = None,
                             home_team_name: str = "", away_team_name: str = "",
                             match_id: int = 0, league: str = "") -> Optional[BetRecommendation]:
        """Build a tracked pick for a SPECIFIC selection (chosen by the briefing LLM).

        Bypasses value/confidence gates; requires real odds in range. Returns None
        if the selection has no usable bookmaker odds.
        """
        context = context or {}
        ensemble = predictions.get("ensemble", {})
        spec = next(
            (s for s in self._market_specs(ensemble) if s[3] == market_key), None
        )
        if spec is None:
            return None
        market, selection, prob, _ = spec
        best_odds = self._find_best_odds(
            odds_data, market, selection,
            home_team_name=home_team_name, away_team_name=away_team_name,
        )
        if not best_odds or best_odds < self.min_odds or best_odds > self.max_odds:
            return None
        return self._build_pick(
            predictions, ensemble, context, match_name, match_id, league, odds_data,
            market, selection, prob, market_key, best_odds,
            home_team_name=home_team_name, away_team_name=away_team_name,
        )

    def _build_pick(self, predictions, ensemble, context, match_name, match_id, league,
                    odds_data, market, selection, prob, market_key, best_odds,
                    home_team_name="", away_team_name="") -> BetRecommendation:
        """Construct a BetRecommendation for a given (market, selection, odds).

        Shared by find_best_bet and build_selection_pick — no value gates, Kelly
        clamped to the minimum stake so the pick carries a real (if small) stake.
        """
        implied = 1.0 / best_odds if best_odds > 0 else 0
        divergence = prob / implied if implied > 0 else 0
        ev = self.calculate_expected_value(prob, best_odds)
        opening = self._find_opening_odds(
            odds_data, market, selection,
            home_team_name=home_team_name, away_team_name=away_team_name,
        )
        agreement_info = self._check_model_agreement(
            predictions, market_key, selection,
            features_dict=context.get("features_dict"),
        )
        kelly_pct = self.kelly_criterion(
            prob, best_odds, agreement=agreement_info.get("agreement"),
        )
        kelly_pct = max(kelly_pct, self.min_kelly_stake)  # ensure a real minimum stake
        risk = self._assess_risk(prob, best_odds, ev)
        xg_info = self._build_xg_insight(context, ensemble, market, selection)
        reasoning = self._build_reasoning(
            market, selection, prob, best_odds, ev, context, agreement_info, xg_info,
        )
        return BetRecommendation(
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
            used_fallback_odds=False,
            league=league,
            model_agreement=agreement_info.get("agreement", ""),
            models_for=agreement_info.get("models_for", ""),
            models_against=agreement_info.get("models_against", ""),
            home_xg_avg=context.get("home_xg_avg", 0.0),
            away_xg_avg=context.get("away_xg_avg", 0.0),
            xg_edge=xg_info.get("insight", ""),
            predicted_xg=xg_info.get("predicted_xg", ""),
            contrarian_value=round(divergence, 2),
            opening_odds=round(opening, 2) if opening else 0.0,
        )

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
        "solo": 0.85,     # single model, no dissent — reduce stake vs full consensus
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

        # Flashscore 1X2 odds are composite/display prices — exclude from EV.
        # Flashscore over_under odds ARE real bookmaker prices (best available
        # from their comparison page) — allow them as a fallback when no API
        # bookmaker (Bet365/Pinnacle/1xBet) provides the line (e.g. Over 1.5).
        _OVER_UNDER_MARKETS = {"over_under", "totals"}

        # Collect every real bookmaker price for this market+selection, plus
        # the best Flashscore over/under price as a fallback when no real
        # bookmaker covers the line.
        real_prices: list = []
        flashscore_ou_best = 0.0

        for odds_record in odds_data:
            bookie = odds_record.get("bookmaker", "")
            record_market = odds_record.get("market_type", "")
            record_selection = odds_record.get("selection", "")
            record_odds = odds_record.get("odds_value", 0) or 0

            if record_market not in valid_markets or record_selection not in valid_selections:
                continue
            if record_odds <= 1.0:
                continue

            if bookie == "Flashscore":
                if record_market in _OVER_UNDER_MARKETS:
                    flashscore_ou_best = max(flashscore_ou_best, record_odds)
            else:
                real_prices.append(record_odds)

        # Aggregation policy: when 2+ bookmakers price a market, use the
        # MEDIAN to dampen outliers. `max()` produces a positive bias because
        # an inflated price from a single bookie always passes through —
        # sharps publish Pinnacle/Bet365 at the consensus, but smaller books
        # mis-price minor leagues regularly.  Median tracks consensus while
        # still gaining a small edge over plain bookmaker average.
        # With a single bookmaker we have no choice — use that price.
        if real_prices:
            real_prices.sort()
            n = len(real_prices)
            if n == 1:
                best = real_prices[0]
            elif n % 2 == 1:
                best = real_prices[n // 2]
            else:
                best = (real_prices[n // 2 - 1] + real_prices[n // 2]) / 2
            if n >= 2:
                spread = real_prices[-1] - real_prices[0]
                if spread >= 0.30:
                    logger.debug(
                        f"Wide odds spread for {selection} ({market}): "
                        f"low={real_prices[0]:.2f} high={real_prices[-1]:.2f} "
                        f"spread={spread:.2f} — using median={best:.2f}"
                    )
        else:
            best = flashscore_ou_best

        return best

    def _find_opening_odds(self, odds_data: List[Dict], market: str,
                           selection: str,
                           home_team_name: str = "",
                           away_team_name: str = "") -> float:
        """Return the median opening line across bookmakers for this selection.

        Uses the same selection/market mapping as _find_best_odds but reads
        the opening_odds field instead of odds_value.  Returns 0.0 when no
        opening odds are recorded (feature is populated gradually).
        """
        selection_map = {
            "Home Win": ["1", "Home", "home_win", "Home Win", home_team_name],
            "Draw": ["X", "Draw", "draw", "Draw"],
            "Away Win": ["2", "Away", "away_win", "Away Win", away_team_name],
            "Over 2.5 Goals": ["Over 2.5", "Over", "Over 2.5 Goals"],
            "Under 2.5 Goals": ["Under 2.5", "Under", "Under 2.5 Goals"],
            "Over 1.5 Goals": ["Over 1.5", "Over 1.5 Goals"],
            "Under 1.5 Goals": ["Under 1.5", "Under 1.5 Goals"],
            "Over 3.5 Goals": ["Over 3.5", "Over 3.5 Goals"],
            "Under 3.5 Goals": ["Under 3.5", "Under 3.5 Goals"],
            "BTTS Yes": ["Yes", "BTTS Yes"],
            "BTTS No": ["No", "BTTS No"],
            "Home Over 1.5": ["Home Over 1.5", "Home Team Over 1.5", "Home - Over 1.5"],
            "Away Over 1.5": ["Away Over 1.5", "Away Team Over 1.5", "Away - Over 1.5"],
        }
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
        valid_selections = [s for s in selection_map.get(selection, [selection]) if s]
        valid_markets = market_map.get(market, [market])

        prices = []
        for rec in odds_data:
            if rec.get("bookmaker") == "Flashscore":
                continue
            if rec.get("market_type") not in valid_markets:
                continue
            if rec.get("selection") not in valid_selections:
                continue
            op = rec.get("opening_odds") or 0
            if op > 1.0:
                prices.append(op)

        if not prices:
            return 0.0
        prices.sort()
        n = len(prices)
        if n == 1:
            return prices[0]
        if n % 2 == 1:
            return prices[n // 2]
        return (prices[n // 2 - 1] + prices[n // 2]) / 2

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
        elif len(models_against) == 0 and len(models_for) == 1:
            agreement = "solo"      # only one model has a signal; no dissent but not true consensus
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
            elif agr == "solo":
                parts.append(f"Only {agreement_info['models_for']} signal (single model).")
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
