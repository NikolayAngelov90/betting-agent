"""Main Football Betting Agent orchestrator."""

import asyncio
import numpy as np
from datetime import datetime, date, timedelta
from dataclasses import dataclass
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
from src.data.models import Match, Team, Odds
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
        return all_recommendations

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
        print("  --train             Train ML models on historical data")
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
                for i, pick in enumerate(picks, 1):
                    print(f"\n{'='*50}")
                    print(f"Pick #{i}: {pick.match}")
                    print(f"  Bet: {pick.selection} @ {pick.odds}")
                    print(f"  EV: {pick.expected_value:.1%}")
                    print(f"  Confidence: {pick.confidence:.1%}")
                    print(f"  Risk: {pick.risk_level}")
                    print(f"  Stake: {pick.kelly_stake_percentage:.1f}% of bankroll")
                    print(f"  Reasoning: {pick.reasoning}")

                # Send to Telegram if enabled
                if agent.telegram.enabled:
                    await agent.telegram.send_daily_picks(picks)
                    print(f"\nPicks sent to Telegram!")

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
