"""Main Football Betting Agent orchestrator."""

import asyncio
import json
import numpy as np
from datetime import datetime, date, timedelta
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict

from src.scrapers.flashscore_scraper import FlashscoreScraper
from src.scrapers.odds_scraper import OddsScraper
from src.scrapers.injury_scraper import InjuryScraper
from src.scrapers.news_scraper import NewsScraper
from src.scrapers.historical_loader import HistoricalDataLoader
from src.features.feature_engineer import FeatureEngineer
from src.models.ensemble import EnsemblePredictor
from src.betting.value_calculator import ValueBettingCalculator, BetRecommendation
from src.betting.bankroll_manager import BankrollManager
from src.reporting.telegram_bot import TelegramNotifier
from src.data.models import Match, Team, Odds, SavedPick
from src.data.database import get_db, init_db
from src.utils.config import get_config
from src.utils.logger import get_logger, setup_logger

logger = get_logger()


@dataclass
class MatchAnalysis:
    """Complete analysis result for a single match."""
    match_id: int
    match_name: str
    match_date: datetime
    league: str
    features: Dict
    predictions: Dict
    recommendations: List[BetRecommendation]
    injury_report: Dict
    news_summary: Dict


class FootballBettingAgent:
    """Main agent that orchestrates all components for betting predictions.

    Coordinates data collection, feature engineering, prediction models,
    and value betting calculations.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = get_config()

        # Setup logging
        log_cfg = self.config.logging
        setup_logger(
            log_level=log_cfg.get("level", "INFO"),
            log_file=log_cfg.get("log_file", "logs/betting_agent.log"),
        )

        # Initialize components
        self.db = init_db()
        self.scraper = FlashscoreScraper(self.config)
        self.odds_collector = OddsScraper(self.config)
        self.injury_tracker = InjuryScraper(self.config)
        self.news_aggregator = NewsScraper(self.config)
        self.historical_loader = HistoricalDataLoader(self.config)
        self.feature_engineer = FeatureEngineer()
        self.predictor = EnsemblePredictor(self.config)
        self.value_calculator = ValueBettingCalculator(self.config)
        self.bankroll = BankrollManager(config=self.config)
        self.telegram = TelegramNotifier(self.config)

        logger.info("Football Betting Agent initialized")

    async def daily_update(self):
        """Run the full daily data collection cycle."""
        logger.info("Starting daily update cycle")

        # 1. Historical data (bootstrap — loads CSV results from football-data.co.uk)
        try:
            await self.historical_loader.update()
        except Exception as e:
            logger.error(f"Historical data loading failed: {e}")

        # 2. Odds API (reliable, fast, provides fixtures + odds + recent scores)
        try:
            await self.odds_collector.update()
            logger.info("Odds API update complete")
        except Exception as e:
            logger.error(f"Odds API update failed: {e}")

        # 2. Flashscore scraping (disabled — Odds API provides fixtures + scores)
        # To re-enable, uncomment below:
        # try:
        #     await asyncio.wait_for(self.scraper.update(), timeout=300)
        #     logger.info("Flashscore update complete")
        # except asyncio.TimeoutError:
        #     logger.warning("Flashscore scraping timed out, skipping")
        #     self.scraper.close_driver()
        # except Exception as e:
        #     logger.error(f"Flashscore update failed: {e}")
        #     self.scraper.close_driver()

        # 4. Injury data
        try:
            await self.injury_tracker.update()
            logger.info("Injury update complete")
        except Exception as e:
            logger.error(f"Injury update failed: {e}")

        # 5. News aggregation
        try:
            await self.news_aggregator.update()
            logger.info("News update complete")
        except Exception as e:
            logger.error(f"News update failed: {e}")

        # 6. Fit/update prediction models
        try:
            self.predictor.fit()
            logger.info("Models fitted")
        except Exception as e:
            logger.error(f"Model fitting failed: {e}")

        logger.info("Daily update cycle complete")

    async def analyze_fixture(self, match_id: int) -> MatchAnalysis:
        """Run complete analysis on a single fixture.

        Args:
            match_id: Match database ID

        Returns:
            MatchAnalysis with predictions and recommendations
        """
        logger.info(f"Analyzing match {match_id}")

        with self.db.get_session() as session:
            match = session.get(Match, match_id)
            if not match:
                raise ValueError(f"Match {match_id} not found")

            home_team = session.get(Team, match.home_team_id)
            away_team = session.get(Team, match.away_team_id)
            home_team_name = home_team.name
            away_team_name = away_team.name
            match_name = f"{home_team_name} vs {away_team_name}"
            match_date = match.match_date
            league = match.league or ""
            home_id = match.home_team_id
            away_id = match.away_team_id

            # Get odds from DB
            odds_records = session.query(Odds).filter_by(match_id=match_id).all()
            odds_data = [
                {
                    "market_type": o.market_type,
                    "selection": o.selection,
                    "odds_value": o.odds_value,
                    "bookmaker": o.bookmaker,
                }
                for o in odds_records
            ]

        # Generate features
        features = await self.feature_engineer.create_features(match_id)
        feature_vector = self.feature_engineer.create_feature_vector(features)

        # Get predictions
        predictions = self.predictor.predict(home_id, away_id, feature_vector)

        # Get injury report
        from src.scrapers.injury_scraper import InjuryScraper
        injury_scraper = InjuryScraper()
        home_injuries = await injury_scraper.get_injury_summary(home_id)
        away_injuries = await injury_scraper.get_injury_summary(away_id)
        injury_report = {"home": home_injuries, "away": away_injuries}

        # Get news summary
        home_sentiment = await self.news_aggregator.get_team_sentiment(home_id)
        away_sentiment = await self.news_aggregator.get_team_sentiment(away_id)
        news_summary = {"home": home_sentiment, "away": away_sentiment}

        # Build context for reasoning
        context = self._build_context(features, injury_report, news_summary)

        # Find value bets
        recommendations = self.value_calculator.find_value_bets(
            predictions, odds_data, match_name, context,
            home_team_name=home_team_name, away_team_name=away_team_name,
            match_id=match_id,
        )

        return MatchAnalysis(
            match_id=match_id,
            match_name=match_name,
            match_date=match_date,
            league=league,
            features=features,
            predictions=predictions,
            recommendations=recommendations,
            injury_report=injury_report,
            news_summary=news_summary,
        )

    async def get_daily_picks(self, target_date: date = None,
                              min_ev: float = 0.05,
                              min_confidence: float = 0.55) -> List[BetRecommendation]:
        """Get high-confidence value betting picks for a specific date.

        Args:
            target_date: Date to get picks for (defaults to today)
            min_ev: Minimum expected value threshold
            min_confidence: Minimum confidence threshold (default 70%)

        Returns:
            List of BetRecommendation sorted by confidence
        """
        target = target_date or date.today()
        logger.info(f"Getting daily picks for {target}")

        with self.db.get_session() as session:
            day_start = datetime.combine(target, datetime.min.time())
            day_end = day_start + timedelta(days=1)
            fixtures = session.query(Match).filter(
                Match.is_fixture == True,
                Match.match_date >= day_start,
                Match.match_date < day_end,
            ).all()
            fixture_ids = [f.id for f in fixtures]

        if not fixture_ids:
            logger.info(f"No fixtures found for {target}")
            return []

        # Data coverage report — flag under-covered fixtures
        with self.db.get_session() as session:
            low_coverage = []
            for fid in fixture_ids:
                m = session.get(Match, fid)
                if not m:
                    continue
                cov = self.predictor.check_coverage(m.home_team_id, m.away_team_id)
                if cov["score"] < 1.0:
                    ht = session.get(Team, m.home_team_id)
                    at = session.get(Team, m.away_team_id)
                    name = f"{ht.name if ht else m.home_team_id} vs {at.name if at else m.away_team_id}"
                    low_coverage.append((name, cov["score"]))
            if low_coverage:
                logger.warning(
                    f"Data coverage gaps in {len(low_coverage)}/{len(fixture_ids)} fixtures:"
                )
                for name, score in low_coverage:
                    logger.warning(f"  {name}: coverage {score:.0%}")

        all_recommendations = []
        for match_id in fixture_ids:
            try:
                analysis = await self.analyze_fixture(match_id)
                for rec in analysis.recommendations:
                    if rec.expected_value >= min_ev and rec.confidence >= min_confidence:
                        all_recommendations.append(rec)
            except Exception as e:
                logger.error(f"Error analyzing match {match_id}: {e}")

        # Sort by confidence (highest first)
        all_recommendations.sort(
            key=lambda r: r.confidence,
            reverse=True,
        )

        logger.info(f"Found {len(all_recommendations)} high-confidence picks for {target}")

        # Save picks to database for tracking
        self._save_picks(all_recommendations, target)

        return all_recommendations

    def _save_picks(self, picks: List[BetRecommendation], pick_date: date):
        """Save picks to database for result tracking."""
        if not picks:
            return

        with self.db.get_session() as session:
            for pick in picks:
                # Skip if already saved (same match + selection + date)
                existing = session.query(SavedPick).filter(
                    SavedPick.match_id == pick.match_id,
                    SavedPick.selection == pick.selection,
                    SavedPick.pick_date == pick_date,
                ).first()
                if existing:
                    continue

                saved = SavedPick(
                    match_id=pick.match_id,
                    pick_date=pick_date,
                    match_name=pick.match,
                    market=pick.market,
                    selection=pick.selection,
                    odds=pick.odds,
                    predicted_probability=pick.predicted_probability,
                    expected_value=pick.expected_value,
                    confidence=pick.confidence,
                    kelly_stake_percentage=pick.kelly_stake_percentage,
                    risk_level=pick.risk_level,
                )
                session.add(saved)

            session.commit()
            logger.info(f"Saved {len(picks)} picks to database")

    def settle_predictions(self):
        """Check pending picks against actual match results and mark win/loss.

        Returns:
            List of dicts with settled pick details for reporting.
        """
        settled = []

        with self.db.get_session() as session:
            pending = session.query(SavedPick).filter(
                SavedPick.result.is_(None),
            ).all()

            if not pending:
                logger.info("No pending picks to settle")
                return settled

            for pick in pending:
                match = session.get(Match, pick.match_id)
                if not match or match.home_goals is None or match.away_goals is None:
                    continue  # Match not completed yet

                hg = match.home_goals
                ag = match.away_goals
                total = hg + ag
                btts = hg > 0 and ag > 0

                # Determine actual outcome for the selection
                won = False
                sel = pick.selection

                if sel == "Home Win":
                    won = hg > ag
                elif sel == "Draw":
                    won = hg == ag
                elif sel == "Away Win":
                    won = hg < ag
                elif sel == "Over 2.5 Goals":
                    won = total > 2.5
                elif sel == "Under 2.5 Goals":
                    won = total < 2.5
                elif sel == "Over 1.5 Goals":
                    won = total > 1.5
                elif sel == "Over 3.5 Goals":
                    won = total > 3.5
                elif sel == "BTTS Yes":
                    won = btts
                elif sel == "BTTS No":
                    won = not btts

                pick.result = "win" if won else "loss"
                pick.actual_home_goals = hg
                pick.actual_away_goals = ag
                pick.settled_at = datetime.utcnow()

                settled.append({
                    "match_name": pick.match_name,
                    "selection": pick.selection,
                    "odds": pick.odds,
                    "result": pick.result,
                    "score": f"{hg}-{ag}",
                    "stake": pick.kelly_stake_percentage,
                    "pick_date": pick.pick_date,
                })

            session.commit()

        logger.info(f"Settled {len(settled)} picks")
        return settled

    def get_stats(self) -> Dict:
        """Calculate prediction statistics over different time periods."""
        now = date.today()

        with self.db.get_session() as session:
            all_picks = session.query(SavedPick).all()

            if not all_picks:
                return {"total": 0}

            settled = [p for p in all_picks if p.result is not None]
            pending = [p for p in all_picks if p.result is None]

            def calc_stats(picks_list):
                if not picks_list:
                    return {"total": 0, "wins": 0, "losses": 0, "win_rate": 0.0, "roi": 0.0}

                wins = sum(1 for p in picks_list if p.result == "win")
                losses = sum(1 for p in picks_list if p.result == "loss")
                total = wins + losses
                win_rate = wins / total if total > 0 else 0.0

                # Simulate ROI with Kelly stakes
                total_staked = sum(p.kelly_stake_percentage for p in picks_list)
                total_profit = sum(
                    p.kelly_stake_percentage * (p.odds - 1) if p.result == "win"
                    else -p.kelly_stake_percentage
                    for p in picks_list
                )
                roi = total_profit / total_staked if total_staked > 0 else 0.0

                # Average odds
                avg_odds_wins = (
                    np.mean([p.odds for p in picks_list if p.result == "win"])
                    if wins > 0 else 0.0
                )
                avg_odds_losses = (
                    np.mean([p.odds for p in picks_list if p.result == "loss"])
                    if losses > 0 else 0.0
                )

                return {
                    "total": total,
                    "wins": wins,
                    "losses": losses,
                    "win_rate": round(win_rate, 4),
                    "roi": round(roi, 4),
                    "avg_odds_wins": round(float(avg_odds_wins), 2),
                    "avg_odds_losses": round(float(avg_odds_losses), 2),
                }

            # By time period
            yesterday = now - timedelta(days=1)
            week_ago = now - timedelta(days=7)
            month_ago = now - timedelta(days=30)

            yesterday_picks = [p for p in settled if p.pick_date == yesterday]
            week_picks = [p for p in settled if p.pick_date >= week_ago]
            month_picks = [p for p in settled if p.pick_date >= month_ago]

            # By market type
            markets = {}
            for p in settled:
                markets.setdefault(p.market, []).append(p)

            # Model coverage summary
            poisson_teams = len(self.predictor.poisson._team_strengths)
            elo_teams = len(self.predictor.elo.ratings)
            ml_fitted = self.predictor.ml_models.is_fitted

            stats = {
                "all_time": calc_stats(settled),
                "last_7_days": calc_stats(week_picks),
                "last_30_days": calc_stats(month_picks),
                "yesterday": calc_stats(yesterday_picks),
                "pending": len(pending),
                "by_market": {m: calc_stats(picks) for m, picks in markets.items()},
                "model_coverage": {
                    "poisson_teams": poisson_teams,
                    "elo_teams": elo_teams,
                    "ml_fitted": ml_fitted,
                },
            }

            return stats

    def tune_ensemble_weights(self):
        """Adjust ensemble weights based on recent prediction accuracy."""
        with self.db.get_session() as session:
            month_ago = date.today() - timedelta(days=30)
            settled = session.query(SavedPick).filter(
                SavedPick.result.isnot(None),
                SavedPick.pick_date >= month_ago,
            ).all()

        if len(settled) < 20:
            logger.info(f"Not enough settled picks for tuning ({len(settled)}, need 20+)")
            return

        # Re-predict each settled match with individual models
        model_correct = {"poisson": 0, "elo": 0, "ml": 0}
        model_total = {"poisson": 0, "elo": 0, "ml": 0}

        for pick in settled:
            if pick.market != "1X2":
                continue  # Only tune on 1X2 market (where all models contribute)

            try:
                with self.db.get_session() as session:
                    match = session.get(Match, pick.match_id)
                    if not match:
                        continue
                    home_id = match.home_team_id
                    away_id = match.away_team_id

                # Actual result
                if pick.actual_home_goals > pick.actual_away_goals:
                    actual = "home_win"
                elif pick.actual_home_goals == pick.actual_away_goals:
                    actual = "draw"
                else:
                    actual = "away_win"

                # Get individual model predictions
                poisson_pred = self.predictor.poisson.predict(home_id, away_id)
                elo_pred = self.predictor.elo.predict(home_id, away_id)

                for model_name, pred in [("poisson", poisson_pred), ("elo", elo_pred)]:
                    best = max(["home_win", "draw", "away_win"], key=lambda k: pred.get(k, 0))
                    model_total[model_name] += 1
                    if best == actual:
                        model_correct[model_name] += 1

                # ML model
                if self.predictor.ml_models.is_fitted:
                    model_total["ml"] += 1
                    # ML prediction would need features, approximate by checking if pick was correct
                    # since ensemble includes ML weight
                    if pick.result == "win":
                        model_correct["ml"] += 1

            except Exception:
                continue

        # Calculate accuracy per model
        accuracies = {}
        for model in ["poisson", "elo", "ml"]:
            if model_total[model] > 0:
                accuracies[model] = model_correct[model] / model_total[model]
            else:
                accuracies[model] = 0.25  # Default

        if sum(accuracies.values()) == 0:
            logger.info("No accuracy data to tune weights")
            return

        # Normalize to get new weights
        total_acc = sum(accuracies.values())
        new_weights = {
            "poisson": round(accuracies["poisson"] / total_acc, 3),
            "elo": round(accuracies["elo"] / total_acc, 3),
            "xgboost": round((accuracies["ml"] / total_acc) * 0.6, 3),  # 60% of ML weight to XGB
            "random_forest": round((accuracies["ml"] / total_acc) * 0.4, 3),  # 40% to RF
        }

        # Save tuned weights
        weights_path = Path("data/models/ensemble_weights.json")
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        weights_path.write_text(json.dumps(new_weights, indent=2))

        logger.info(f"Tuned ensemble weights: {new_weights}")
        logger.info(f"Model accuracies: {accuracies}")

        # Apply immediately
        self.predictor.weights = new_weights

        return new_weights

    async def train_ml_models(self, max_samples: int = 2000):
        """Train ML models on historical match data.

        Extracts features from completed matches, creates labels from results,
        and trains XGBoost/RandomForest/LogisticRegression models.

        Args:
            max_samples: Maximum matches to use for training (most recent).
                         More = better accuracy but slower. Default 2000.
        """
        logger.info("Starting ML model training...")

        # Fit Poisson/Elo first (needed for feature context)
        self.predictor.fit()

        # Get most recent completed matches with results
        with self.db.get_session() as session:
            matches = session.query(Match).filter(
                Match.is_fixture == False,
                Match.home_goals.isnot(None),
                Match.away_goals.isnot(None),
            ).order_by(Match.match_date.desc()).limit(max_samples).all()

            # Reverse to chronological order (oldest first) for time-series CV
            match_data = [
                {
                    "id": m.id,
                    "home_goals": m.home_goals,
                    "away_goals": m.away_goals,
                }
                for m in reversed(matches)
            ]

        logger.info(f"Using {len(match_data)} most recent matches for training")

        if len(match_data) < 100:
            logger.warning("Not enough matches for ML training (need 100+)")
            return

        # Build feature matrix and labels
        X_list = []
        y_list = []
        feature_names = None
        skipped = 0

        for i, md in enumerate(match_data):
            try:
                features = await self.feature_engineer.create_features(md["id"])
                if not features:
                    skipped += 1
                    continue

                vec = self.feature_engineer.create_feature_vector(features)
                if feature_names is None:
                    feature_names = self.feature_engineer.get_feature_names(features)

                # Only include if vector length matches (consistent features)
                if feature_names and len(vec) == len(feature_names):
                    X_list.append(vec)

                    # Label: 2=home_win, 1=draw, 0=away_win
                    if md["home_goals"] > md["away_goals"]:
                        y_list.append(2)
                    elif md["home_goals"] == md["away_goals"]:
                        y_list.append(1)
                    else:
                        y_list.append(0)
                else:
                    skipped += 1

            except Exception as e:
                skipped += 1
                continue

            # Progress logging
            if (i + 1) % 200 == 0:
                logger.info(f"Processed {i + 1}/{len(match_data)} matches ({len(X_list)} valid)...")

        if len(X_list) < 100:
            logger.warning(f"Only {len(X_list)} valid samples (skipped {skipped}), need 100+")
            return

        X = np.array(X_list)
        y = np.array(y_list)

        logger.info(f"Training on {len(X)} samples, {X.shape[1]} features (skipped {skipped})")

        # Train models
        self.predictor.ml_models.fit(X, y, feature_names)
        self.predictor.ml_models.save()

        # Log feature importance
        importance = self.predictor.ml_models.get_feature_importance()
        for model_name, features in importance.items():
            top5 = features[:5]
            logger.info(f"{model_name} top features: {', '.join(f'{n}={v:.3f}' for n, v in top5)}")

        logger.info("ML model training complete")

    def _build_context(self, features: Dict, injury_report: Dict,
                       news_summary: Dict) -> Dict:
        """Build context strings for bet reasoning."""
        context = {}

        # Form insight
        home_form = features.get("home_overall_form_string", "")
        away_form = features.get("away_overall_form_string", "")
        if home_form or away_form:
            context["form_insight"] = f"Home form: {home_form}, Away form: {away_form}."

        # H2H insight
        h2h_meetings = features.get("h2h_total_meetings", 0)
        if h2h_meetings > 0:
            h2h_home_pct = features.get("h2h_home_win_pct", 0)
            context["h2h_insight"] = (
                f"H2H: {h2h_meetings} meetings, "
                f"home wins {h2h_home_pct:.0%} of the time."
            )

        # Injury impact
        home_injured = injury_report.get("home", {}).get("total_injured", 0)
        away_injured = injury_report.get("away", {}).get("total_injured", 0)
        if home_injured or away_injured:
            context["injury_impact"] = (
                f"Injuries: home {home_injured} out, away {away_injured} out."
            )

        # News
        home_trend = news_summary.get("home", {}).get("trend", "neutral")
        away_trend = news_summary.get("away", {}).get("trend", "neutral")
        context["news_insight"] = f"News sentiment: home={home_trend}, away={away_trend}."

        return context

    async def shutdown(self):
        """Clean up resources."""
        self.scraper.close_driver()
        await self.scraper.close()
        await self.odds_collector.close()
        await self.injury_tracker.close()
        await self.news_aggregator.close()
        await self.historical_loader.close()
        logger.info("Agent shutdown complete")


async def main():
    """CLI entry point."""
    import sys

    agent = FootballBettingAgent()

    if len(sys.argv) < 2:
        print("Usage: python -m src.agent.betting_agent <command>")
        print("\nCommands:")
        print("  --init              Initialize database and collect data")
        print("  --update            Run daily data update")
        print("  --picks             Show today's value picks")
        print("  --settle            Settle pending picks with actual results")
        print("  --stats             Show prediction statistics")
        print("  --train             Train ML models on historical data")
        print("  --tune              Tune ensemble weights from recent results")
        print("  --analyze <id>      Analyze a specific match")
        print("  --telegram-setup    Setup Telegram bot notifications")
        print("  --telegram-test     Send a test Telegram message")
        return

    command = sys.argv[1]

    try:
        if command == "--init":
            print("Initializing database and running first data collection...")
            await agent.daily_update()
            print("Initialization complete.")

        elif command == "--update":
            print("Running daily update...")
            await agent.daily_update()
            print("Update complete.")

        elif command == "--train":
            print("Training ML models on historical data...")
            await agent.train_ml_models()
            print("ML training complete.")

        elif command == "--picks":
            agent.predictor.fit()
            picks = await agent.get_daily_picks()
            if not picks:
                print("No value picks found for today.")
            else:
                # Send to Telegram first (before console print which may fail on encoding)
                if agent.telegram.enabled:
                    stats = agent.get_stats()
                    await agent.telegram.send_daily_picks(picks, stats=stats)
                    print(f"\nPicks sent to Telegram!")

                for i, pick in enumerate(picks, 1):
                    # Encode-safe printing for non-ASCII team names
                    match_name = pick.match.encode("ascii", "replace").decode()
                    reasoning = pick.reasoning.encode("ascii", "replace").decode()
                    print(f"\n{'='*50}")
                    print(f"Pick #{i}: {match_name}")
                    print(f"  Bet: {pick.selection} @ {pick.odds}")
                    print(f"  EV: {pick.expected_value:.1%}")
                    print(f"  Confidence: {pick.confidence:.1%}")
                    print(f"  Risk: {pick.risk_level}")
                    print(f"  Stake: {pick.kelly_stake_percentage:.1f}% of bankroll")
                    print(f"  Reasoning: {reasoning}")

        elif command == "--settle":
            print("Settling pending picks against actual results...")
            settled_picks = agent.settle_predictions()
            stats = agent.get_stats()
            all_time = stats.get("all_time", {})
            print(f"\nSettled: {len(settled_picks)} picks")
            print(f"All time: {all_time.get('total', 0)} picks, Win rate: {all_time.get('win_rate', 0):.1%}")
            print(f"Pending: {stats.get('pending', 0)} picks")

            # Send settlement report to Telegram
            if settled_picks and agent.telegram.enabled:
                await agent.telegram.send_settlement_report(settled_picks, stats)
                print("Settlement report sent to Telegram!")

        elif command == "--stats":
            agent.settle_predictions()  # Settle any new results first
            stats = agent.get_stats()

            if stats.get("total", 0) == 0 and stats.get("all_time", {}).get("total", 0) == 0:
                print("No prediction history yet. Run --picks first.")
            else:
                print(f"\n{'='*50}")
                print("PREDICTION STATISTICS")
                print(f"{'='*50}")

                for period, label in [
                    ("yesterday", "Yesterday"),
                    ("last_7_days", "Last 7 Days"),
                    ("last_30_days", "Last 30 Days"),
                    ("all_time", "All Time"),
                ]:
                    s = stats.get(period, {})
                    if s.get("total", 0) > 0:
                        print(f"\n{label}:")
                        print(f"  Record: {s['wins']}W - {s['losses']}L ({s['total']} total)")
                        print(f"  Win Rate: {s['win_rate']:.1%}")
                        print(f"  ROI: {s['roi']:.1%}")
                        print(f"  Avg Odds (W): {s['avg_odds_wins']:.2f} | Avg Odds (L): {s['avg_odds_losses']:.2f}")

                print(f"\nPending: {stats.get('pending', 0)} picks")

                by_market = stats.get("by_market", {})
                if by_market:
                    print(f"\nBy Market:")
                    for market, ms in by_market.items():
                        if ms.get("total", 0) > 0:
                            print(f"  {market}: {ms['wins']}W-{ms['losses']}L ({ms['win_rate']:.1%})")

                cov = stats.get("model_coverage", {})
                if cov:
                    print(f"\nModel Coverage:")
                    print(f"  Poisson: {cov.get('poisson_teams', 0)} teams with strength data")
                    print(f"  Elo: {cov.get('elo_teams', 0)} teams with ratings")
                    print(f"  ML Models: {'Fitted' if cov.get('ml_fitted') else 'Not fitted'}")

        elif command == "--tune":
            print("Tuning ensemble weights from recent results...")
            agent.predictor.fit()
            weights = agent.tune_ensemble_weights()
            if weights:
                print(f"\nNew weights: {json.dumps(weights, indent=2)}")
            else:
                print("Not enough data to tune weights.")

        elif command == "--analyze" and len(sys.argv) > 2:
            agent.predictor.fit()
            match_id = int(sys.argv[2])
            analysis = await agent.analyze_fixture(match_id)
            print(f"\nMatch: {analysis.match_name}")
            print(f"Date: {analysis.match_date}")
            print(f"League: {analysis.league}")
            print(f"\nPredictions (Ensemble):")
            ens = analysis.predictions.get("ensemble", {})
            print(f"  Home Win: {ens.get('home_win', 0):.1%}")
            print(f"  Draw: {ens.get('draw', 0):.1%}")
            print(f"  Away Win: {ens.get('away_win', 0):.1%}")
            print(f"  xG: {ens.get('home_xg', 0):.2f} - {ens.get('away_xg', 0):.2f}")
            print(f"\nValue Bets: {len(analysis.recommendations)}")
            for rec in analysis.recommendations:
                print(f"  - {rec.selection} @ {rec.odds} (EV: {rec.expected_value:.1%})")

        elif command == "--telegram-setup":
            print("=" * 50)
            print("Telegram Bot Setup Guide")
            print("=" * 50)
            print()
            print("Step 1: Create a Telegram Bot")
            print("  - Open Telegram and search for @BotFather")
            print("  - Send /newbot and follow the prompts")
            print("  - Copy the bot token (e.g., 123456:ABC-DEF...)")
            print()
            print("Step 2: Get your Chat ID")
            print("  - Start a chat with your new bot (click Start)")
            print("  - Send any message to the bot")
            print("  - Open: https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates")
            print("  - Find 'chat':{'id': YOUR_CHAT_ID} in the response")
            print()
            print("Step 3: Update config/config.yaml")
            print("  notifications:")
            print("    telegram_enabled: true")
            print("    telegram_bot_token: \"YOUR_BOT_TOKEN\"")
            print("    telegram_chat_id: \"YOUR_CHAT_ID\"")
            print()
            print("Step 4: Test with --telegram-test")

        elif command == "--telegram-test":
            if not agent.telegram.enabled:
                print("Telegram is not enabled. Run --telegram-setup first.")
                print("Then set telegram_enabled: true in config/config.yaml")
            else:
                await agent.telegram.send_alert(
                    "Football Betting Agent - Test message.\n"
                    "Telegram notifications are working!"
                )
                print("Test message sent! Check your Telegram.")

        else:
            print(f"Unknown command: {command}")

    finally:
        await agent.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
