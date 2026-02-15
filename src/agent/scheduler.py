"""Automated scheduling pipeline for the betting agent."""

import asyncio
from datetime import datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from src.agent.betting_agent import FootballBettingAgent
from src.utils.logger import get_logger

logger = get_logger()


class AutomatedPipeline:
    """Scheduled task runner for daily betting agent operations.

    Schedule:
        11:00 - Full daily pipeline: update data, generate picks, send to Telegram
    """

    def __init__(self, agent: FootballBettingAgent = None):
        self.agent = agent or FootballBettingAgent()
        self.scheduler = AsyncIOScheduler()
        self._setup_schedule()

    def _setup_schedule(self):
        """Configure daily 11:00 AM pipeline."""
        self.scheduler.add_job(
            self._daily_pipeline,
            CronTrigger(hour=11, minute=0),
            id="daily_pipeline",
            name="Daily full pipeline (11:00 AM)",
        )

    async def _daily_pipeline(self):
        """Run the full daily pipeline: update data, generate picks, send to Telegram."""
        logger.info("[Scheduler] Starting daily pipeline...")
        try:
            # Step 1: Update all data sources
            logger.info("[Scheduler] Updating results...")
            await self.agent.scraper.update()

            logger.info("[Scheduler] Updating odds...")
            await self.agent.odds_collector.update()

            logger.info("[Scheduler] Updating injuries...")
            await self.agent.injury_tracker.update()

            logger.info("[Scheduler] Updating news...")
            await self.agent.news_aggregator.update()

            # Step 2: Settle yesterday's predictions
            logger.info("[Scheduler] Settling pending picks...")
            self.agent.settle_predictions()

            # Step 3: Retrain models with latest data
            logger.info("[Scheduler] Fitting models...")
            self.agent.predictor.fit()

            # Step 4: Generate predictions
            logger.info("[Scheduler] Generating predictions...")
            picks = await self.agent.get_daily_picks()
            logger.info(f"[Scheduler] Found {len(picks)} value picks for today")

            for pick in picks:
                logger.info(
                    f"[Pick] {pick.match}: {pick.selection} @ {pick.odds} "
                    f"(EV: {pick.expected_value:.1%})"
                )

            # Step 5: Send picks via Telegram with stats
            if self.agent.telegram.enabled and picks:
                stats = self.agent.get_stats()
                await self.agent.telegram.send_daily_picks(picks, stats=stats)
                logger.info("[Scheduler] Picks sent to Telegram")

            logger.info("[Scheduler] Daily pipeline complete")
        except Exception as e:
            logger.error(f"[Scheduler] Daily pipeline failed: {e}")

    def start(self):
        """Start the scheduler."""
        self.scheduler.start()
        logger.info("[Scheduler] Automated pipeline started")
        logger.info(f"[Scheduler] {len(self.scheduler.get_jobs())} jobs scheduled")

    def stop(self):
        """Stop the scheduler."""
        self.scheduler.shutdown()
        logger.info("[Scheduler] Automated pipeline stopped")


async def run_pipeline():
    """Run the automated pipeline."""
    pipeline = AutomatedPipeline()
    pipeline.start()

    try:
        # Keep running
        while True:
            await asyncio.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        pipeline.stop()
        await pipeline.agent.shutdown()


if __name__ == "__main__":
    asyncio.run(run_pipeline())
