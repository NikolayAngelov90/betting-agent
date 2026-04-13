"""Main Football Betting Agent orchestrator."""

import asyncio
import json
import numpy as np
from datetime import datetime, date, timedelta
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict

from src.scrapers.flashscore_scraper import FlashscoreScraper
from src.scrapers.injury_scraper import InjuryScraper
from src.scrapers.historical_loader import HistoricalDataLoader
from src.scrapers.apifootball_scraper import APIFootballScraper
from src.scrapers.footballdataorg_scraper import FootballDataOrgScraper
from src.scrapers.theodds_scraper import TheOddsScraper
from src.features.feature_engineer import FeatureEngineer
from src.models.ensemble import EnsemblePredictor
from src.betting.value_calculator import ValueBettingCalculator, BetRecommendation
from src.reporting.telegram_bot import TelegramNotifier
from src.data.models import Match, Team, Odds, SavedPick
from src.data.database import get_db, init_db
from src.utils.config import get_config
from src.utils.logger import get_logger, setup_logger, utcnow

logger = get_logger()


def _sync_create_features(feature_engineer, match_id, as_of_date):
    """Run create_features synchronously in a thread (for ML training).

    create_features is async but with for_training=True all internal work
    is synchronous DB queries.  Running via run_in_executor lets multiple
    matches query the DB in parallel threads instead of blocking the
    event loop sequentially.
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(
            feature_engineer.create_features(
                match_id, as_of_date=as_of_date, for_training=True
            )
        )
    finally:
        loop.close()


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
        self.historical_loader = HistoricalDataLoader(self.config)
        self.apifootball = APIFootballScraper(self.config)
        self.injury_tracker = InjuryScraper(self.config, apifootball=self.apifootball)
        self.footballdataorg = FootballDataOrgScraper(self.config)
        self.theodds = TheOddsScraper(self.config)
        self.feature_engineer = FeatureEngineer()
        self.predictor = EnsemblePredictor(self.config)
        self.value_calculator = ValueBettingCalculator(self.config)
        self.telegram = TelegramNotifier(self.config)

        # Track leagues scraped within this process lifetime (for dedup
        # between settle and daily_update even when no new results are found).
        self._scraped_leagues: set = set()

        logger.info("Football Betting Agent initialized")

    _SCRAPED_LEAGUES_FILE = Path("data/scraped_leagues.json")

    def _get_recently_scraped_leagues(self, minutes: int = 30) -> set:
        """Return leagues scraped within the last N minutes.

        Checks both a marker file (cross-process, written after each
        successful Flashscore scrape) and DB timestamps as fallback.
        """
        result: set = set()

        # 1. Marker file (reliable cross-process, written by _mark_league_scraped)
        try:
            if self._SCRAPED_LEAGUES_FILE.exists():
                data = json.loads(self._SCRAPED_LEAGUES_FILE.read_text())
                ts = datetime.fromisoformat(data.get("timestamp", ""))
                if (utcnow() - ts).total_seconds() < minutes * 60:
                    result.update(data.get("leagues", []))
        except Exception:
            pass

        # 2. DB timestamp fallback
        cutoff = utcnow() - timedelta(minutes=minutes)
        try:
            with self.db.get_session() as session:
                recent = (
                    session.query(Match.league)
                    .filter(
                        Match.updated_at >= cutoff,
                        Match.home_goals.isnot(None),
                        Match.league.isnot(None),
                    )
                    .distinct()
                    .all()
                )
                result.update(r[0] for r in recent if r[0])
        except Exception:
            pass

        return result

    def _mark_league_scraped(self, league: str):
        """Record a successfully scraped league in a marker file.

        Survives across separate process invocations within the same CI run.
        """
        self._scraped_leagues.add(league)
        try:
            data: dict = {}
            if self._SCRAPED_LEAGUES_FILE.exists():
                data = json.loads(self._SCRAPED_LEAGUES_FILE.read_text())
            existing = set(data.get("leagues", []))
            existing.add(league)
            self._SCRAPED_LEAGUES_FILE.parent.mkdir(parents=True, exist_ok=True)
            self._SCRAPED_LEAGUES_FILE.write_text(json.dumps({
                "timestamp": utcnow().isoformat(),
                "leagues": sorted(existing),
            }))
        except Exception:
            pass

    async def daily_update(self, skip_ml_retrain: bool = False, skip_flashscore_results: bool = False):
        """Run the full daily data collection cycle.

        Args:
            skip_flashscore_results: When True, skip the Flashscore browser results
                scraping block entirely.  Use this in CI so the time-critical
                fixtures/odds/picks path isn't blocked by a 20-min Chrome session.
                Scores for all leagues are already covered by API-Football's
                yesterday-fixtures fetch.  Run --update-results in a separate CI
                step after picks to backfill Flashscore stats/coverage.
        """
        logger.info("Starting daily update cycle")

        # 0. Prune old odds to keep DB size under control (Neon 500MB free tier)
        try:
            self.db.prune_old_odds(keep_days=400)
        except Exception as e:
            logger.warning(f"Odds pruning failed (non-fatal): {e}")

        # 1. Historical data (bootstrap — loads CSV results from football-data.co.uk)
        try:
            await self.historical_loader.update()
        except Exception as e:
            logger.error(f"Historical data loading failed: {e}")

        # 1b. football-data.org: fixtures + results for 9 top leagues (free, no quota).
        # Runs before Flashscore so it can fill gaps when Chrome scraping fails.
        # Covers: PL, BL1, La Liga, Serie A, Ligue 1, CL, Eredivisie, Primeira Liga, ELC.
        try:
            fdo_results = await self.footballdataorg.update_results(days_back=1)
            fdo_fixtures = await self.footballdataorg.sync_fixtures(days_ahead=0)
            logger.info(
                f"football-data.org: {fdo_results} scores updated, "
                f"{fdo_fixtures} new fixtures added"
            )
        except Exception as e:
            logger.warning(f"football-data.org update failed (non-critical): {e}")

        # 2. Flashscore results + fixtures (no API quota — primary fixture source)
        # Results and fixtures are scraped independently so a results timeout does
        # not prevent today's fixtures from being loaded.
        # Each section has a hard time budget to prevent runaway Chrome processes
        # from blowing the CI timeout.
        import time as _timer
        leagues = self.config.get("scraping.flashscore_leagues", [])

        if skip_flashscore_results:
            logger.info(
                "Flashscore results scraping skipped (--skip-flashscore-results). "
                "Scores are covered by API-Football; run --update-results after picks "
                "for full Flashscore coverage."
            )

        _RESULTS_BUDGET_S = 1500  # 25 minutes for results (~60s/league, covers all 27 leagues)
        _FIXTURES_BUDGET_S = 300  # 5 minutes for fixtures (only leagues with today's matches)

        # Skip leagues that were already scraped by settle_predictions() within
        # this CI run. Uses _scraped_leagues (set explicitly during settle) +
        # DB timestamp fallback for cross-process runs.
        _recently_scraped = self._scraped_leagues | self._get_recently_scraped_leagues(minutes=30)
        if _recently_scraped:
            logger.info(
                f"Skipping {len(_recently_scraped)} leagues already scraped by settle: "
                f"{', '.join(sorted(_recently_scraped))}"
            )

        # Reorder leagues: prioritize those with today's fixtures or pending
        # picks (needed for settlement).  Non-priority leagues are scraped only
        # if time budget allows — this prevents 20+ leagues × 60s from blowing
        # the daily-update step timeout.
        _today_leagues: set = set()
        _pending_leagues: set = set()
        try:
            with self.db.get_session() as session:
                _day_start = datetime.combine(date.today(), datetime.min.time())
                _day_end = _day_start + timedelta(days=1)
                _today_rows = (
                    session.query(Match.league)
                    .filter(
                        Match.is_fixture == True,
                        Match.match_date >= _day_start,
                        Match.match_date < _day_end,
                        Match.league.isnot(None),
                    )
                    .distinct()
                    .all()
                )
                _today_leagues = {r[0] for r in _today_rows if r[0]}

                # Leagues with unsettled picks (need fresh results for settlement)
                _pending_rows = (
                    session.query(Match.league)
                    .join(SavedPick, SavedPick.match_id == Match.id)
                    .filter(
                        SavedPick.result.is_(None),
                        Match.league.isnot(None),
                    )
                    .distinct()
                    .all()
                )
                _pending_leagues = {r[0] for r in _pending_rows if r[0]}
        except Exception:
            pass

        _important = _today_leagues | _pending_leagues
        _priority = [l for l in leagues if l in _important]
        _rest = [l for l in leagues if l not in _important]
        _ordered_leagues = _priority + _rest
        if _priority:
            logger.info(
                f"Prioritized {len(_priority)} leagues (fixtures/pending picks): "
                f"{', '.join(_priority)}"
            )

        if not skip_flashscore_results:
            try:
                _results_deadline = _timer.monotonic() + _RESULTS_BUDGET_S
                for league in _ordered_leagues:
                    if _timer.monotonic() > _results_deadline:
                        logger.warning("Flashscore results: time budget exhausted, skipping remaining leagues")
                        break
                    if league in _recently_scraped:
                        continue
                    try:
                        await asyncio.wait_for(
                            self.scraper.scrape_league_results(league, skip_stats=True),
                            timeout=60,
                        )
                        self._mark_league_scraped(league)
                    except asyncio.TimeoutError:
                        logger.warning(f"Flashscore results timeout for {league}, continuing")
                        try:
                            self.scraper.close_driver()  # reset Chrome for next league
                        except Exception:
                            pass
                    except Exception as e:
                        logger.debug(f"Flashscore results error for {league}: {e}")
                logger.info("Flashscore results update complete")
            except Exception as e:
                logger.error(f"Flashscore results update failed: {e}")
            finally:
                try:
                    self.scraper.close_driver()
                except Exception:
                    pass

        # Fixtures (today's upcoming matches) — only scrape leagues that have
        # today's fixtures OR are priority (pending picks).  Skipping leagues
        # with 0 fixtures avoids ~5-7s per wasted page load × 20+ leagues.
        _fixture_leagues = [
            l for l in _ordered_leagues
            if l in _important  # has today's fixtures or pending picks
        ]
        _skipped_fixture_leagues = len(_ordered_leagues) - len(_fixture_leagues)
        if _skipped_fixture_leagues:
            logger.info(
                f"Fixtures: scraping {len(_fixture_leagues)} leagues with fixtures/pending, "
                f"skipping {_skipped_fixture_leagues} without"
            )
        try:
            _fixtures_deadline = _timer.monotonic() + _FIXTURES_BUDGET_S
            for league in _fixture_leagues:
                if _timer.monotonic() > _fixtures_deadline:
                    logger.warning("Flashscore fixtures: time budget exhausted, skipping remaining leagues")
                    break
                try:
                    await asyncio.wait_for(
                        self.scraper.scrape_league_fixtures(league), timeout=90,
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"Flashscore fixture timeout for {league}, continuing")
                    try:
                        self.scraper.close_driver()  # reset Chrome for next league
                    except Exception:
                        pass
                except Exception as e:
                    logger.debug(f"Flashscore fixture error for {league}: {e}")
            logger.info("Flashscore fixtures update complete")
        except Exception as e:
            logger.error(f"Flashscore fixtures update failed: {e}")
        finally:
            try:
                self.scraper.close_driver()
            except Exception:
                pass

        # 2b. API-Football (fixtures, xG, advanced stats, odds).
        try:
            await asyncio.wait_for(self.apifootball.update(), timeout=1080)  # 18 min cap
            logger.info("API-Football update complete")
        except asyncio.TimeoutError:
            logger.warning("API-Football update timed out after 10 minutes")
        except Exception as e:
            logger.error(f"API-Football update failed: {e}")

        # 2c. The Odds API — supplemental odds for leagues with today's fixtures.
        # Free tier: 500 credits/month (~1 credit/league call). Only calls leagues
        # that actually have fixtures today to minimise credit burn.
        try:
            theodds_written = await asyncio.wait_for(self.theodds.update(), timeout=240)
            logger.info(f"The Odds API update complete: {theodds_written} odds rows written")
        except asyncio.TimeoutError:
            logger.warning("The Odds API update timed out after 4 minutes")
        except Exception as e:
            logger.warning(f"The Odds API update failed (non-critical): {e}")

        # 2d. Flashscore per-match stats enrichment — DISABLED.
        # API-Football (BUDGET_XG=30) is the primary stats source now and covers
        # shots/possession/xG via fast API calls (~0.5s/match). Flashscore browser
        # scraping (~12s/match) is unreliable due to Cloudflare blocking in CI.
        self.scraper.close_driver()

        # 4. Injury data (cap at 5 min — API-Football calls can be slow)
        try:
            await asyncio.wait_for(self.injury_tracker.update(), timeout=300)
            logger.info("Injury update complete")
        except asyncio.TimeoutError:
            logger.warning("Injury update timed out after 5 minutes")
        except Exception as e:
            logger.error(f"Injury update failed: {e}")

        # 5. Fit/update prediction models
        try:
            self.predictor.fit()
            self.feature_engineer.elo_ratings = self.predictor.elo.ratings
            logger.info("Models fitted")
        except Exception as e:
            logger.error(f"Model fitting failed: {e}")

        # 6a. Retrain ML models if stale.
        # When --skip-ml-retrain is passed (CI), defer to a dedicated --train step
        # with its own timeout so scraping + training don't compete for time.
        try:
            max_age = self.config.get("models.ml_retrain_days", 3)
            if self._ml_models_stale(max_age_days=max_age):
                stale_info = getattr(self.predictor.ml_models, "trained_at", "never")
                if skip_ml_retrain:
                    logger.info(f"ML models stale (last trained: {stale_info}) — deferred to --train step")
                else:
                    logger.info(f"ML models stale (last trained: {stale_info}) — retraining")
                    await asyncio.wait_for(
                        self.train_ml_models(max_samples=2000),
                        timeout=720,  # 12 min cap
                    )
                    logger.info("ML models retrained")
            else:
                logger.debug("ML models fresh — skipping retrain")
        except asyncio.TimeoutError:
            logger.warning("ML retrain timed out after 12 minutes — skipping")
        except Exception as e:
            logger.warning(f"ML retrain failed: {e}")

        # 6b. Targeted backfill for low-coverage teams in today's fixtures.
        try:
            _today = date.today()
            _start = datetime.combine(_today, datetime.min.time())
            _end = _start + timedelta(days=1)
            with self.db.get_session() as _sess:
                _fixtures = _sess.query(Match).filter(
                    Match.is_fixture == True,
                    Match.match_date >= _start,
                    Match.match_date < _end,
                ).all()
                _low_cov_team_ids = set()
                for _m in _fixtures:
                    cov = self.predictor.check_coverage(_m.home_team_id, _m.away_team_id)
                    if not cov["home_poisson"] or not cov["home_elo"]:
                        _low_cov_team_ids.add(_m.home_team_id)
                    if not cov["away_poisson"] or not cov["away_elo"]:
                        _low_cov_team_ids.add(_m.away_team_id)
            if _low_cov_team_ids:
                _backfill_budget = min(15, self.apifootball.remaining_budget())
                logger.info(
                    f"Found {len(_low_cov_team_ids)} low-coverage teams in today's fixtures — "
                    f"triggering targeted backfill (budget: {_backfill_budget})"
                )
                if _backfill_budget > 0:
                    _budget_before = self.apifootball.remaining_budget()
                    _bf_seasons = self.config.get("models.backfill_seasons", (2022, 2023, 2024))
                    _bf_min = self.config.get("models.backfill_min_matches", 10)
                    await self.apifootball.backfill_team_history(
                        min_matches=_bf_min,
                        seasons=_bf_seasons,
                        max_budget=_backfill_budget,
                        min_remaining_budget=0,
                        target_team_ids=_low_cov_team_ids,
                    )
                    _reqs_used = _budget_before - self.apifootball.remaining_budget()
                    if _reqs_used > 0:
                        self.predictor.fit()
                        self.feature_engineer.elo_ratings = self.predictor.elo.ratings
                        logger.info(f"Models re-fitted after backfill ({_reqs_used} requests used)")
                    else:
                        logger.debug("Backfill made no API calls — skipping re-fit")
        except Exception as e:
            logger.warning(f"Low-coverage backfill failed: {e}")

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

        # Coverage gate — skip fixtures where at least one team has no data
        # from any model (no Poisson strengths AND no Elo rating).
        # score-based check (< 0.50) was an edge case: two teams with only Elo
        # ratings scored exactly 0.50 and slipped through despite 0% historical data.
        coverage = self.predictor.check_coverage(home_id, away_id)
        home_has_data = coverage["home_poisson"] or coverage["home_elo"]
        away_has_data = coverage["away_poisson"] or coverage["away_elo"]
        if not home_has_data or not away_has_data:
            missing = []
            if not home_has_data:
                missing.append("home")
            if not away_has_data:
                missing.append("away")
            logger.info(
                f"Skipping {match_name}: no historical data for {'/'.join(missing)} team "
                f"(coverage {coverage['score']:.0%})"
            )
            return MatchAnalysis(
                match_id=match_id, match_name=match_name, match_date=match_date,
                league=league, features={}, predictions={}, recommendations=[],
                injury_report={},
            )

        # Generate features
        features = await self.feature_engineer.create_features(match_id)
        feature_vector = self.feature_engineer.create_feature_vector(features)
        feature_names = self.feature_engineer.get_feature_names(features)

        # Get predictions
        predictions = self.predictor.predict(home_id, away_id, feature_vector,
                                             feature_names=feature_names,
                                             league=league)

        # Fetch injury data in parallel — local-DB reads
        home_injuries, away_injuries = await asyncio.gather(
            self.injury_tracker.get_injury_summary(home_id),
            self.injury_tracker.get_injury_summary(away_id),
        )
        injury_report = {"home": home_injuries, "away": away_injuries}

        # Build context for reasoning
        context = self._build_context(features, injury_report)

        # Find value bets
        recommendations = self.value_calculator.find_value_bets(
            predictions, odds_data, match_name, context,
            home_team_name=home_team_name, away_team_name=away_team_name,
            match_id=match_id, league=league,
        )
        for rec in recommendations:
            rec.match_date = match_date

        return MatchAnalysis(
            match_id=match_id,
            match_name=match_name,
            match_date=match_date,
            league=league,
            features=features,
            predictions=predictions,
            recommendations=recommendations,
            injury_report=injury_report,
        )

    async def get_daily_picks(self, target_date: date = None,
                              max_picks_per_match: int = 2,
                              leagues: List[str] = None) -> List[BetRecommendation]:
        """Get high-confidence value betting picks for a specific date.

        EV and confidence thresholds are read from config (betting.min_expected_value /
        betting.min_confidence) by the ValueBettingCalculator — no need to pass them here.

        Args:
            target_date: Date to get picks for (defaults to today)
            max_picks_per_match: Maximum picks allowed per single match (default 2)
            leagues: Optional list of league keys to restrict picks to

        Returns:
            List of BetRecommendation sorted by EV × confidence × model-agreement bonus
        """
        target = target_date or date.today()
        league_label = f" (leagues: {', '.join(leagues)})" if leagues else ""
        logger.info(f"Getting daily picks for {target}{league_label}")

        with self.db.get_session() as session:
            day_start = datetime.combine(target, datetime.min.time())
            day_end = day_start + timedelta(days=1)
            # Only include matches that haven't kicked off yet — skip finished
            # matches and those already in progress (match_date is kickoff time).
            # DB stores UTC datetimes so compare against UTC, not local time.
            now = utcnow()
            query = session.query(Match).filter(
                Match.is_fixture == True,
                Match.match_date >= day_start,
                Match.match_date < day_end,
                Match.match_date > now,
            )
            if leagues:
                query = query.filter(Match.league.in_(leagues))
            else:
                # Restrict to configured leagues only — fixtures from non-configured
                # leagues (created by API-Football's LEAGUE_ID_MAP) waste analysis
                # time and API budget without having Flashscore enrichment.
                _configured = self.config.get("scraping.flashscore_leagues", [])
                if _configured:
                    query = query.filter(Match.league.in_(_configured))
            fixtures = query.all()

            # Diagnostic: log how many fixtures per league for tracing missing matches
            league_counts: dict = {}
            for f in fixtures:
                league_counts[f.league] = league_counts.get(f.league, 0) + 1
            logger.info(
                f"Fixture query: {len(fixtures)} matches for {target} "
                f"(now={now.strftime('%H:%M')} UTC), "
                f"by league: {dict(sorted(league_counts.items()))}"
            )

            # Deduplicate fixtures: Flashscore and football-data.org may create
            # separate Match records for the same game (e.g. "Man City" vs
            # "Manchester City"). Keep the record with more odds or with
            # flashscore_id (better enrichment data).
            from src.scrapers.flashscore_scraper import FlashscoreScraper as _FS
            _nm = _FS._team_names_similar
            # seen_list: [(match_id, home_name, league, match_date)]
            seen_list: list = []
            dedup_ids: list = []

            def _hours_apart(dt1, dt2):
                """Absolute time difference in hours (handles midnight wrap)."""
                return abs((dt1 - dt2).total_seconds()) / 3600

            for f in fixtures:
                ht = session.get(Team, f.home_team_id)
                h_name = ht.name if ht else ""
                # Check if we've seen a match in same league within 2h window
                dup_idx = None
                for idx, (existing_id, existing_home, e_league, e_dt) in enumerate(seen_list):
                    if e_league == f.league and _hours_apart(e_dt, f.match_date) <= 2:
                        if _nm(h_name, existing_home):
                            dup_idx = idx
                            break
                if dup_idx is not None:
                    # Keep the one with more odds data
                    existing_id = seen_list[dup_idx][0]
                    new_odds = session.query(Odds).filter_by(match_id=f.id).count()
                    old_odds = session.query(Odds).filter_by(match_id=existing_id).count()
                    if new_odds > old_odds:
                        dedup_ids.remove(existing_id)
                        dedup_ids.append(f.id)
                        seen_list[dup_idx] = (f.id, h_name, f.league, f.match_date)
                        logger.debug(f"Dedup: replaced fixture {existing_id} with {f.id} ({h_name}, more odds)")
                    else:
                        logger.debug(f"Dedup: skipping duplicate fixture {f.id} ({h_name}, fewer odds)")
                else:
                    seen_list.append((f.id, h_name, f.league, f.match_date))
                    dedup_ids.append(f.id)

            fixture_ids = dedup_ids
            if len(dedup_ids) < len(fixtures):
                logger.info(f"Deduplicated {len(fixtures)} → {len(dedup_ids)} fixtures")

        if not fixture_ids:
            logger.info(f"No fixtures found for {target}")
            return []

        # ── API-Football odds fallback ─────────────────────────────────────────
        # For fixtures that have NO real bookmaker odds (zero odds, or only
        # "Flashscore" display odds), try API-Football if they have an
        # apifootball_id and we have budget remaining (free tier = 100/day).
        if self.apifootball.enabled:
            _apifb_budget_remaining = min(self.apifootball.remaining_budget(), 40)
            with self.db.get_session() as session:
                # Tier-1: matches with zero real bookmaker odds (highest priority)
                apifb_fallback = []
                # Tier-2: matches that have TheOddsAPI 1X2/2.5 odds but are
                # missing Over 1.5 (TheOddsAPI never provides that line)
                apifb_missing_over15 = []
                for fid in fixture_ids:
                    m = session.get(Match, fid)
                    if m and m.apifootball_id:
                        ht = m.home_team.name if m.home_team else str(m.home_team_id)
                        at = m.away_team.name if m.away_team else str(m.away_team_id)
                        # Count only REAL bookmaker odds (exclude "Flashscore" display odds)
                        real_odds = session.query(Odds).filter(
                            Odds.match_id == fid,
                            Odds.bookmaker != "Flashscore",
                        ).count()
                        if real_odds == 0:
                            apifb_fallback.append((fid, m.apifootball_id, ht, at))
                        else:
                            # Has some odds but check if Over 1.5 is missing
                            has_over15 = session.query(Odds).filter(
                                Odds.match_id == fid,
                                Odds.market_type == "over_under",
                                Odds.selection == "Over 1.5",
                            ).count()
                            if not has_over15:
                                apifb_missing_over15.append((fid, m.apifootball_id, ht, at))

                # Combine: zero-odds tier first, then missing-Over-1.5 tier
                apifb_fallback = apifb_fallback + apifb_missing_over15
            if apifb_fallback:
                capped = apifb_fallback[:_apifb_budget_remaining]
                logger.info(
                    f"Fetching API-Football odds for {len(capped)}/{len(apifb_fallback)} fixtures "
                    f"({len(apifb_fallback) - len(apifb_missing_over15)} no-odds + "
                    f"{len(apifb_missing_over15)} missing Over 1.5) "
                    f"(budget: {_apifb_budget_remaining} requests left)"
                )
                import time as _timer
                _FALLBACK_ODDS_BUDGET_S = 600  # 10-minute hard cap
                _fallback_deadline = _timer.monotonic() + _FALLBACK_ODDS_BUDGET_S
                for match_id, fixture_id, home, away in capped:
                    if _timer.monotonic() > _fallback_deadline:
                        logger.warning(
                            f"Fallback odds time budget exhausted "
                            f"({_FALLBACK_ODDS_BUDGET_S // 60} min)"
                        )
                        break
                    try:
                        odds_data = await self.apifootball._fetch_fixture_odds(fixture_id)
                        if odds_data:
                            count = self.apifootball._save_fixture_odds(match_id, odds_data)
                            logger.info(f"API-Football fallback: saved {count} odds for {home} vs {away}")
                        else:
                            logger.debug(f"No API-Football odds for fixture {fixture_id} ({home} vs {away})")
                    except Exception as exc:
                        logger.warning(f"API-Football fallback failed for fixture {fixture_id}: {exc}")

        # Data coverage report — flag under-covered fixtures and leagues
        from src.scrapers.historical_loader import LEAGUE_CSV_MAP, EXTRA_LEAGUE_CSV_MAP
        # Flashscore-scraped leagues count as "covered" (results come from live scraping,
        # not from football-data.co.uk CSVs).  CL/EL/ECL are covered this way.
        _flashscore_leagues = set(self.config.get("scraping.flashscore_leagues", []))
        all_hist_leagues = (
            set(LEAGUE_CSV_MAP.keys())
            | set(EXTRA_LEAGUE_CSV_MAP.keys())
            | _flashscore_leagues
        )

        with self.db.get_session() as session:
            low_coverage = []
            uncovered_leagues = set()
            for fid in fixture_ids:
                m = session.get(Match, fid)
                if not m:
                    continue
                cov = self.predictor.check_coverage(m.home_team_id, m.away_team_id)
                if cov["score"] < 1.0:
                    ht = session.get(Team, m.home_team_id)
                    at = session.get(Team, m.away_team_id)
                    name = f"{ht.name if ht else m.home_team_id} vs {at.name if at else m.away_team_id}"
                    low_coverage.append((name, cov["score"], m.league))
                if m.league and m.league not in all_hist_leagues:
                    uncovered_leagues.add(m.league)
            if low_coverage:
                logger.warning(
                    f"Data coverage gaps in {len(low_coverage)}/{len(fixture_ids)} fixtures:"
                )
                for name, score, league in low_coverage:
                    logger.warning(f"  {name}: coverage {score:.0%} [{league}]")
            if uncovered_leagues:
                logger.warning(
                    f"Leagues with fixtures but NO historical data source: "
                    f"{', '.join(sorted(uncovered_leagues))}. "
                    f"Add them to historical_loader.py to improve predictions."
                )

        # Value threshold auto-calibration: adjust min EV based on recent hit rate.
        # Hot streak → tighten (be more selective), cold streak → loosen slightly.
        self._auto_calibrate_ev_threshold()

        # Pre-filter: skip fixtures with no real bookmaker odds.
        # Feature engineering costs ~1s × 146 DB queries on Neon.  Fixtures with
        # zero odds can never produce an EV bet — skip them before analysis to avoid
        # wasting the full 45-minute CI budget on a large Sunday fixture slate.
        with self.db.get_session() as session:
            odds_fixture_ids = []
            no_odds_count = 0
            for fid in fixture_ids:
                has_real_odds = session.query(Odds).filter(
                    Odds.match_id == fid,
                    Odds.bookmaker != "Flashscore",
                ).count() > 0
                if has_real_odds:
                    odds_fixture_ids.append(fid)
                else:
                    no_odds_count += 1
            if no_odds_count:
                logger.info(
                    f"Skipping {no_odds_count} fixtures with no real bookmaker odds "
                    f"({len(odds_fixture_ids)} remaining for analysis)"
                )
            fixture_ids = odds_fixture_ids

        if not fixture_ids:
            logger.info("No fixtures with bookmaker odds for analysis")
            return []

        # Analyze fixtures with bounded concurrency — feature engineering and
        # prediction involve synchronous DB queries that block the event loop,
        # so a semaphore limits how many run at once.  Default 5 keeps weekend
        # 70-fixture runs within CI timeout; override via ANALYSIS_CONCURRENCY.
        import os as _os
        _CONCURRENCY = int(_os.environ.get("ANALYSIS_CONCURRENCY", "5"))
        _SEM = asyncio.Semaphore(_CONCURRENCY)

        async def _analyze_with_sem(mid):
            async with _SEM:
                return await self.analyze_fixture(mid)

        logger.info(f"Analyzing {len(fixture_ids)} fixtures (concurrency={_CONCURRENCY})...")
        analyses = await asyncio.gather(
            *(_analyze_with_sem(mid) for mid in fixture_ids),
            return_exceptions=True,
        )

        all_recommendations = []
        for mid, result in zip(fixture_ids, analyses):
            if isinstance(result, Exception):
                logger.error(f"Error analyzing match {mid}: {result}")
                continue
            all_recommendations.extend(result.recommendations)

        # Sort order: EV × confidence × agreement bonus × contrarian bonus.
        # Contrarian picks (model significantly disagrees with market) get
        # a boost when backed by strong model agreement — these are genuine
        # edges the market is mispricing, not model errors.
        _agreement_bonus = {"unanimous": 1.15, "majority": 1.0, "split": 0.85, "unknown": 0.95}

        def _contrarian_bonus(r):
            """1.0 for normal, up to 1.10 for contrarian + unanimous."""
            cv = getattr(r, "contrarian_value", 0) or 0
            if cv >= 1.3 and r.model_agreement == "unanimous":
                return 1.10
            if cv >= 1.3 and r.model_agreement == "majority":
                return 1.05
            return 1.0

        # Log contrarian picks for visibility
        for r in all_recommendations:
            cv = getattr(r, "contrarian_value", 0) or 0
            if cv >= 1.3:
                logger.info(
                    f"Contrarian pick: {r.match} {r.selection} — model "
                    f"{r.predicted_probability:.0%} vs market "
                    f"{1/r.odds:.0%} ({cv:.1f}x divergence, {r.model_agreement})"
                )

        all_recommendations.sort(
            key=lambda r: r.expected_value * r.confidence
            * _agreement_bonus.get(r.model_agreement, 1.0)
            * _contrarian_bonus(r),
            reverse=True,
        )

        # Limit picks per match: keep only the top N by confidence per match.
        # First deduplicate identical (match, selection) pairs, then group by
        # match and keep the highest-confidence picks from each group.
        seen_pick_keys: set = set()
        deduped = []
        for rec in all_recommendations:
            key = (rec.match, rec.selection)
            if key in seen_pick_keys:
                logger.debug(f"Skipping duplicate pick for {rec.match}: {rec.selection}")
                continue
            seen_pick_keys.add(key)
            deduped.append(rec)

        limited = deduped
        if max_picks_per_match:
            from collections import Counter, defaultdict
            # Pre-populate from already-saved picks for today so re-running --picks
            # doesn't accumulate more than max_picks_per_match across multiple runs.
            existing_counts: dict = Counter()
            with self.db.get_session() as _sess:
                _existing = _sess.query(SavedPick.match_id).filter(
                    SavedPick.pick_date == target
                ).all()
                for (_mid,) in _existing:
                    existing_counts[_mid] += 1

            # Group picks by match_id, sort each group by confidence descending,
            # then keep only the top N (minus already-saved) from each match.
            by_match: dict = defaultdict(list)
            for rec in deduped:
                by_match[rec.match_id].append(rec)

            limited = []
            for match_id, group in by_match.items():
                group.sort(key=lambda r: r.confidence, reverse=True)
                already = existing_counts.get(match_id, 0)
                slots = max(0, max_picks_per_match - already)
                if slots < len(group):
                    skipped = group[slots:]
                    for s in skipped:
                        logger.debug(
                            f"Skipping lower-confidence pick for {s.match}: "
                            f"{s.selection} (conf={s.confidence:.1%}, "
                            f"keeping top {max_picks_per_match} per match)"
                        )
                limited.extend(group[:slots])

            # Re-sort by EV × confidence × agreement for final ordering
            limited.sort(
                key=lambda r: r.expected_value * r.confidence
                * _agreement_bonus.get(r.model_agreement, 1.0),
                reverse=True,
            )
        all_recommendations = limited

        # Market correlation filter: when a match has 2 correlated picks
        # (e.g. Home Win + Over 2.5), keep only the higher-EV one to avoid
        # over-concentrating risk on correlated outcomes.
        all_recommendations = self._filter_correlated_picks(all_recommendations)

        # No portfolio caps — include all value picks to avoid missing edges.
        # Risk is managed at the Kelly fraction and max_stake_percentage level.

        # Drawdown circuit breaker: scale down stakes (or skip picks entirely)
        # when recent performance shows a significant drawdown.
        dd_multiplier = self._get_drawdown_multiplier()
        if dd_multiplier < 1.0:
            if dd_multiplier <= 0.0:
                logger.warning(
                    "Drawdown circuit breaker TRIGGERED — skipping all picks "
                    "(drawdown exceeds pause threshold)"
                )
                return [], []
            # Tighten EV threshold proportionally with drawdown severity so
            # fewer marginal picks pass when we're in a losing run.
            # Formula: +2pp at full pause threshold (multiplier→0), +0pp at no drawdown.
            # Example: multiplier=0.55 → +0.9pp (3%→3.9%)
            dd_ev_boost = round((1.0 - dd_multiplier) * 0.02, 4)
            tightened_ev = round(self.value_calculator.min_ev + dd_ev_boost, 4)
            self.value_calculator.min_ev = tightened_ev
            logger.info(
                f"Drawdown circuit breaker: scaling stakes to "
                f"{dd_multiplier:.0%} of normal; "
                f"tightened EV threshold to {tightened_ev:.1%} (+{dd_ev_boost:.1%})"
            )
            for rec in all_recommendations:
                rec.kelly_stake_percentage = round(
                    rec.kelly_stake_percentage * dd_multiplier, 2
                )

        # Daily exposure limit: cap total Kelly stake across all picks to
        # prevent over-betting even when many value picks are found.
        max_daily_exposure = self.config.get("betting.max_total_kelly_pct", 40.0)
        if max_daily_exposure > 0 and all_recommendations:
            # Already sorted by EV×conf×agreement — trim from the bottom
            total_exposure = sum(r.kelly_stake_percentage for r in all_recommendations)
            if total_exposure > max_daily_exposure:
                capped = []
                running = 0.0
                for rec in all_recommendations:
                    if running + rec.kelly_stake_percentage > max_daily_exposure:
                        # If we haven't added any yet, add this one scaled down
                        if not capped:
                            rec.kelly_stake_percentage = round(max_daily_exposure, 2)
                            capped.append(rec)
                        break
                    running += rec.kelly_stake_percentage
                    capped.append(rec)
                trimmed = len(all_recommendations) - len(capped)
                logger.info(
                    f"Daily exposure cap: {total_exposure:.1f}% → "
                    f"{sum(r.kelly_stake_percentage for r in capped):.1f}% "
                    f"(dropped {trimmed} lowest-ranked picks, "
                    f"cap={max_daily_exposure:.0f}%)"
                )
                all_recommendations = capped

        logger.info(f"Found {len(all_recommendations)} high-confidence picks for {target}")

        # Save picks; get back only the ones new this run (for Telegram deduplication)
        new_picks = self._save_picks(all_recommendations, target)

        return all_recommendations, new_picks

    def _save_picks(self, picks: List[BetRecommendation], pick_date: date) -> List[BetRecommendation]:
        """Save picks to database for result tracking.

        Returns the list of picks that were *newly* saved this run so the caller
        can send only fresh picks to Telegram and avoid re-notifying for picks
        that were already sent in an earlier run on the same day.
        """
        if not picks:
            return []

        new_picks: List[BetRecommendation] = []
        with self.db.get_session() as session:
            for pick in picks:
                # Skip if already saved (same match + selection + date)
                existing = session.query(SavedPick).filter(
                    SavedPick.match_id == pick.match_id,
                    SavedPick.selection == pick.selection,
                    SavedPick.pick_date == pick_date,
                ).first()
                if not existing:
                    # Secondary guard: same match name (catches duplicate match rows)
                    existing = session.query(SavedPick).filter(
                        SavedPick.match_name == pick.match,
                        SavedPick.selection == pick.selection,
                        SavedPick.pick_date == pick_date,
                    ).first()
                if existing:
                    continue

                saved = SavedPick(
                    match_id=pick.match_id,
                    pick_date=pick_date,
                    match_name=pick.match,
                    league=pick.league,
                    market=pick.market,
                    selection=pick.selection,
                    odds=float(pick.odds),
                    predicted_probability=float(pick.predicted_probability),
                    expected_value=float(pick.expected_value),
                    confidence=float(pick.confidence),
                    kelly_stake_percentage=float(pick.kelly_stake_percentage),
                    risk_level=pick.risk_level,
                    used_fallback_odds=pick.used_fallback_odds,
                )
                session.add(saved)
                new_picks.append(pick)

            session.commit()
            logger.info(f"Saved {len(new_picks)} new picks to database (skipped {len(picks) - len(new_picks)} duplicates)")
        return new_picks

    async def scrape_results(self, budget_seconds: int = 2880):
        """Scrape Flashscore results for all configured leagues.

        Intended to run as a separate CI step AFTER picks so the time-critical
        fixtures/odds/picks path is not blocked.  Covers all leagues with a
        48-minute budget, giving the ensemble full historical coverage for
        tomorrow's predictions.

        Ordering:
          1. Leagues with unsettled picks (settlement-critical)
          2. All remaining domestic/mid-size leagues
          3. Large slow tournaments (UCL/UEL/ECL) last — these have the most
             match rows and consistently take 3-5 min each in CI
        """
        import time as _timer

        logger.info("Starting Flashscore results scrape (all leagues)")
        leagues = self.config.get("scraping.flashscore_leagues", [])
        _recently_scraped = self._scraped_leagues | self._get_recently_scraped_leagues(minutes=60)

        # Priority: leagues with pending picks first (important for settlement)
        _pending_leagues: set = set()
        try:
            with self.db.get_session() as session:
                _pending_rows = (
                    session.query(Match.league)
                    .join(SavedPick, SavedPick.match_id == Match.id)
                    .filter(SavedPick.result.is_(None), Match.league.isnot(None))
                    .distinct()
                    .all()
                )
                _pending_leagues = {r[0] for r in _pending_rows if r[0]}
        except Exception:
            pass

        # Deprioritize large tournaments within _rest: they have the most rows
        # and take 3-5 min each, crowding out faster domestic leagues.
        # Only applies when they have NO pending picks (if they do, they're priority).
        _SLOW_LEAGUES = {
            "europe/champions-league",
            "europe/europa-league",
            "europe/europa-conference-league",
        }
        _priority = [l for l in leagues if l in _pending_leagues]
        _rest_normal = [l for l in leagues if l not in _pending_leagues and l not in _SLOW_LEAGUES]
        _rest_slow   = [l for l in leagues if l not in _pending_leagues and l in _SLOW_LEAGUES]
        _ordered = _priority + _rest_normal + _rest_slow

        if _priority:
            logger.info(f"scrape_results priority order: {len(_priority)} pending-pick leagues first, "
                        f"{len(_rest_slow)} slow tournaments last")

        scraped, skipped = 0, 0
        deadline = _timer.monotonic() + budget_seconds
        try:
            for league in _ordered:
                remaining_s = deadline - _timer.monotonic()
                if remaining_s <= 0:
                    logger.warning(
                        f"Flashscore results budget exhausted after {budget_seconds}s, "
                        f"{len(_ordered) - scraped - skipped} leagues not reached"
                    )
                    break
                if league in _recently_scraped:
                    skipped += 1
                    continue
                _league_start = _timer.monotonic()
                try:
                    await asyncio.wait_for(
                        self.scraper.scrape_league_results(league, skip_stats=True),
                        timeout=300,  # 5 min hard cap per league — Chrome is blocking so this
                    )               # is a last-resort guard; normal scrape is 60-150s
                    self._mark_league_scraped(league)
                    scraped += 1
                    _elapsed = _timer.monotonic() - _league_start
                    logger.info(
                        f"[update-results] {league}: {_elapsed:.0f}s "
                        f"({deadline - _timer.monotonic():.0f}s budget remaining)"
                    )
                except asyncio.TimeoutError:
                    _elapsed = _timer.monotonic() - _league_start
                    logger.warning(
                        f"Flashscore results timeout for {league} after {_elapsed:.0f}s, continuing"
                    )
                    try:
                        self.scraper.close_driver()
                    except Exception:
                        pass
                except Exception as e:
                    logger.debug(f"Flashscore results error for {league}: {e}")
        finally:
            try:
                self.scraper.close_driver()
            except Exception:
                pass

        logger.info(f"Flashscore results scrape complete: {scraped} scraped, {skipped} skipped (already done)")

    async def settle_predictions(self):
        """Fetch recent results and settle pending picks.

        Uses Flashscore as the PRIMARY results source (no API quota) so that
        settlement works even when the API-Football daily limit is exhausted.
        API-Football is attempted afterwards as a secondary enrichment only.

        Returns:
            List of dicts with settled pick details for reporting.
        """
        # 1. Find all leagues with pending picks and how far back to look.
        days_back = 2
        leagues_with_pending: list = []
        try:
            with self.db.get_session() as session:
                pending_all = (
                    session.query(SavedPick)
                    .filter(SavedPick.result.is_(None))
                    .all()
                )
                if not pending_all:
                    logger.info("No pending picks to settle")
                    return []

                seen_leagues: set = set()
                for pick in pending_all:
                    match = session.get(Match, pick.match_id)
                    if match and match.league and match.league not in seen_leagues:
                        leagues_with_pending.append(match.league)
                        seen_leagues.add(match.league)

                oldest = min(
                    pending_all,
                    key=lambda p: p.pick_date if p.pick_date else date.today(),
                )
                if oldest.pick_date:
                    oldest_date = (
                        oldest.pick_date
                        if isinstance(oldest.pick_date, date)
                        else date.fromisoformat(str(oldest.pick_date))
                    )
                    delta = (date.today() - oldest_date).days
                    days_back = max(days_back, delta + 1)
                    days_back = min(days_back, 7)
        except Exception:
            pass

        logger.info(
            f"Settling picks: {days_back} days back, "
            f"{len(leagues_with_pending)} leagues with pending picks"
        )

        # 2. Settle uses DB results only — --update runs before --settle in the
        # workflow, so API-Football + football-data.org have already populated
        # yesterday's scores.  Flashscore is scraped post-picks via --update-results.
        logger.info(
            f"Settling against DB results (no inline scraping — run --update-results after picks)"
        )

        settled = []

        with self.db.get_session() as session:
            pending = session.query(SavedPick).filter(
                SavedPick.result.is_(None),
            ).all()

            # Also re-evaluate recently settled picks where the stored score
            # differs from the current match score — catches cases where the
            # scraper first captured a live/partial score, settled the pick,
            # and a later scrape wrote the correct final score.
            from datetime import timedelta as _td2
            correction_window = utcnow() - _td2(days=3)
            already_settled = (
                session.query(SavedPick)
                .filter(
                    SavedPick.result.isnot(None),
                    SavedPick.settled_at >= correction_window,
                    SavedPick.actual_home_goals.isnot(None),
                )
                .all()
            )
            needs_correction = []
            for sp in already_settled:
                m = session.get(Match, sp.match_id)
                if (m and m.home_goals is not None
                        and (m.home_goals != sp.actual_home_goals
                             or m.away_goals != sp.actual_away_goals)):
                    logger.warning(
                        f"Score mismatch on settled pick ID={sp.id} "
                        f"({sp.match_name} {sp.selection}): "
                        f"settled with {sp.actual_home_goals}-{sp.actual_away_goals}, "
                        f"match now shows {m.home_goals}-{m.away_goals} — re-settling"
                    )
                    needs_correction.append(sp)
            # Combine: pending first, then corrections
            all_picks_to_process = list(pending) + needs_correction

            if not all_picks_to_process:
                logger.info("No pending picks to settle")
                return settled

            for pick in all_picks_to_process:
                match = session.get(Match, pick.match_id)

                # Skip picks for matches that haven't started yet — the match record
                # still has is_fixture=True and no goals. Don't fall through to the
                # fuzzy fallback which could find yesterday's completed match.
                if match and match.is_fixture:
                    continue

                # Fallback: if the primary match_id has no result, scan same league+date
                # by team-name similarity (handles cross-source name mismatches)
                if (not match or match.home_goals is None or match.away_goals is None) and pick.match_name:
                    parts = pick.match_name.split(" vs ", 1)
                    if len(parts) == 2 and pick.match_id:
                        ref_match = session.get(Match, pick.match_id)
                        ref_date = ref_match.match_date if ref_match else None
                        if ref_date:
                            from datetime import timedelta as _td
                            # Narrow window: 4h back (UTC offset tolerance) + 12h forward
                            # (score reporting delay).  ±24h was too wide — it reached
                            # yesterday's completed matches and settled today's unplayed picks.
                            window_start = ref_date - _td(hours=4)
                            window_end = ref_date + _td(hours=12)
                            # Filter by league (from pick) to avoid false cross-league matches
                            _league_filter = [Match.league == pick.league] if pick.league else []
                            candidates = (
                                session.query(Match)
                                .filter(
                                    Match.is_fixture == False,
                                    Match.home_goals.isnot(None),
                                    Match.match_date >= window_start,
                                    Match.match_date <= window_end,
                                    *_league_filter,
                                )
                                .all()
                            )
                            from src.scrapers.flashscore_scraper import FlashscoreScraper as _FS
                            h_name, a_name = parts[0].strip(), parts[1].strip()
                            for cand in candidates:
                                ch = session.get(Team, cand.home_team_id)
                                ca = session.get(Team, cand.away_team_id)
                                if ch and ca and _FS._team_names_similar(ch.name, h_name) and _FS._team_names_similar(ca.name, a_name):
                                    match = cand
                                    break
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
                elif sel == "Under 1.5 Goals":
                    won = total < 1.5
                elif sel == "Over 3.5 Goals":
                    won = total > 3.5
                elif sel == "Under 3.5 Goals":
                    won = total < 3.5
                elif sel == "BTTS Yes":
                    won = btts
                elif sel == "BTTS No":
                    won = not btts
                elif sel == "Home Over 1.5":
                    won = hg >= 2
                elif sel == "Away Over 1.5":
                    won = ag >= 2

                pick.result = "win" if won else "loss"
                pick.actual_home_goals = hg
                pick.actual_away_goals = ag
                pick.settled_at = utcnow()

                settled.append({
                    "match_name": pick.match_name,
                    "selection": pick.selection,
                    "odds": pick.odds,
                    "result": pick.result,
                    "score": f"{hg}-{ag}",
                    "stake": pick.kelly_stake_percentage,
                    "pick_date": pick.pick_date,
                    "league": pick.league,
                    "home_xg": match.home_xg,
                    "away_xg": match.away_xg,
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

            # Fallback vs real odds breakdown
            real_odds_settled = [p for p in settled if not getattr(p, "used_fallback_odds", False)]
            fallback_settled = [p for p in settled if getattr(p, "used_fallback_odds", False)]

            # Stale picks health check — pending picks older than 48h
            stale_cutoff = now - timedelta(days=2)
            stale_picks = [p for p in pending if p.pick_date and p.pick_date < stale_cutoff]

            # Per-league breakdown
            leagues = {}
            for p in settled:
                lg = getattr(p, "league", None) or "unknown"
                leagues.setdefault(lg, []).append(p)

            # Calibration: predicted probability vs actual win rate by bucket
            calibration = {}
            for bucket_lo in range(45, 95, 10):  # 45-55, 55-65, ..., 85-95
                bucket_hi = bucket_lo + 10
                label = f"{bucket_lo}-{bucket_hi}%"
                bucket_picks = [
                    p for p in settled
                    if p.predicted_probability is not None
                    and bucket_lo / 100 <= p.predicted_probability < bucket_hi / 100
                ]
                if bucket_picks:
                    wins = sum(1 for p in bucket_picks if p.result == "win")
                    calibration[label] = {
                        "predicted_avg": round(np.mean([p.predicted_probability for p in bucket_picks]), 3),
                        "actual_win_rate": round(wins / len(bucket_picks), 3),
                        "count": len(bucket_picks),
                    }

            # Brier score: mean squared error of predicted probabilities vs outcomes
            brier_samples = [
                (p.predicted_probability, 1.0 if p.result == "win" else 0.0)
                for p in settled
                if p.predicted_probability is not None
            ]
            brier_score = None
            if brier_samples:
                brier_score = round(
                    float(np.mean([(pred - actual) ** 2 for pred, actual in brier_samples])), 4
                )

            # CLV (Closing Line Value): predicted_prob - 1/odds
            # Positive CLV = model found genuine edge vs market
            clv_values = [
                p.predicted_probability - (1.0 / p.odds)
                for p in settled
                if p.predicted_probability is not None and p.odds and p.odds > 1.0
            ]
            avg_clv = round(float(np.mean(clv_values)), 4) if clv_values else None

            stats = {
                "all_time": calc_stats(settled),
                "last_7_days": calc_stats(week_picks),
                "last_30_days": calc_stats(month_picks),
                "yesterday": calc_stats(yesterday_picks),
                "pending": len(pending),
                "by_market": {m: calc_stats(picks) for m, picks in markets.items()},
                "by_league": {lg: calc_stats(picks) for lg, picks in leagues.items()},
                "model_coverage": {
                    "poisson_teams": poisson_teams,
                    "elo_teams": elo_teams,
                    "ml_fitted": ml_fitted,
                },
                "odds_source": {
                    "real_odds": calc_stats(real_odds_settled),
                    "fallback_odds": calc_stats(fallback_settled),
                },
                "stale_picks": len(stale_picks),
                "calibration": calibration,
                "brier_score": brier_score,
                "avg_clv": avg_clv,
            }

            return stats

    async def tune_ensemble_weights(self):
        """Adjust ensemble weights based on recent prediction accuracy.

        Optimized: batch-loads all match data in one query, runs Poisson/Elo
        predictions in-memory (no per-pick DB round-trips or ML feature
        engineering). Completes in seconds instead of tens of minutes.
        """
        with self.db.get_session() as session:
            month_ago = date.today() - timedelta(days=30)
            rows = session.query(SavedPick).filter(
                SavedPick.result.isnot(None),
                SavedPick.pick_date >= month_ago,
            ).all()
            # Extract all needed attributes inside session to avoid detached-instance errors
            settled = [
                {
                    "match_id": p.match_id,
                    "market": p.market,
                    "selection": p.selection,
                    "actual_home_goals": p.actual_home_goals,
                    "actual_away_goals": p.actual_away_goals,
                    "match_date": p.pick_date,
                }
                for p in rows
            ]

        if len(settled) < 20:
            logger.info(f"Not enough settled picks for tuning ({len(settled)}, need 20+)")
            return

        logger.info(f"Tuning on {len(settled)} settled picks")

        # Batch-load all match data in one query to avoid per-pick DB round-trips
        match_ids = list({p["match_id"] for p in settled})
        match_data = {}  # match_id -> (home_team_id, away_team_id, league)
        with self.db.get_session() as session:
            matches = session.query(Match).filter(Match.id.in_(match_ids)).all()
            for m in matches:
                match_data[m.id] = (m.home_team_id, m.away_team_id, m.league or "")

        # Filter to 1X2 picks with valid match data and compute actual results
        outcomes = []  # list of (pick, home_id, away_id, league, actual_result)
        for pick in settled:
            if pick["market"] != "1X2":
                continue
            md = match_data.get(pick["match_id"])
            if not md:
                continue
            home_id, away_id, league = md
            hg, ag = pick["actual_home_goals"], pick["actual_away_goals"]
            if hg is None or ag is None:
                continue
            if hg > ag:
                actual = "home_win"
            elif hg == ag:
                actual = "draw"
            else:
                actual = "away_win"
            outcomes.append((pick, home_id, away_id, league, actual))

        if not outcomes:
            logger.info("No 1X2 outcomes to tune on")
            return

        # Single pass: accuracy + calibration data + Bayesian updates
        model_correct = {"poisson": 0, "elo": 0, "ml": 0}
        model_total = {"poisson": 0, "elo": 0, "ml": 0}
        model_predictions: dict = {"poisson": [], "elo": []}
        ensemble_correct = 0
        ensemble_total = 0
        bayesian = self.predictor.bayesian_weights
        sel_map = {"Home Win": "home_win", "Draw": "draw", "Away Win": "away_win"}
        keys_1x2 = ["home_win", "draw", "away_win"]

        for i, (pick, home_id, away_id, league, actual) in enumerate(outcomes):
            if i % 50 == 0:
                logger.info(f"  Tuning: {i}/{len(outcomes)} 1X2 picks processed")

            poisson_pred = self.predictor.poisson.predict(home_id, away_id, league=league)
            elo_pred = self.predictor.elo.predict(home_id, away_id)

            # --- Accuracy ---
            for model_name, pred in [("poisson", poisson_pred), ("elo", elo_pred)]:
                best = max(keys_1x2, key=lambda k: pred.get(k, 0))
                model_total[model_name] += 1
                if best == actual:
                    model_correct[model_name] += 1

            # ML accuracy: use existing fitted models with simple predict (no feature eng)
            # ML weight is derived from Poisson/Elo accuracy ratio — skip expensive re-eval
            # to keep tuning fast.

            # --- Calibration data ---
            p_best_key = max(keys_1x2, key=lambda k: poisson_pred.get(k, 0))
            model_predictions["poisson"].append(
                (poisson_pred.get(p_best_key, 0.33), 1 if p_best_key == actual else 0)
            )
            e_best_key = max(keys_1x2, key=lambda k: elo_pred.get(k, 0))
            model_predictions["elo"].append(
                (elo_pred.get(e_best_key, 0.33), 1 if e_best_key == actual else 0)
            )

            # --- Ensemble accuracy ---
            if sel_map.get(pick["selection"]) == actual:
                ensemble_correct += 1
            ensemble_total += 1

            # --- Bayesian per-league updates ---
            if league:
                days_ago = (date.today() - pick["match_date"]).days if pick["match_date"] else 0
                poisson_best = max(keys_1x2, key=lambda k: poisson_pred.get(k, 0))
                bayesian.update(league, "poisson", poisson_best == actual, days_ago, market="1X2")
                elo_best = max(keys_1x2, key=lambda k: elo_pred.get(k, 0))
                bayesian.update(league, "elo", elo_best == actual, days_ago, market="1X2")

            # --- Poisson goals accuracy for Bayesian ---
            hg, ag = pick["actual_home_goals"], pick["actual_away_goals"]
            if hg is not None and ag is not None and league:
                total_goals = hg + ag
                actual_over25 = total_goals > 2.5
                poisson_over25 = poisson_pred.get("over_2.5", 0.5) > 0.5
                days_ago = (date.today() - pick["match_date"]).days if pick["match_date"] else 0
                bayesian.update(league, "poisson", poisson_over25 == actual_over25, days_ago, market="goals")

        logger.info(f"  Tuning: {len(outcomes)}/{len(outcomes)} 1X2 picks processed")

        # Calculate accuracy per model
        accuracies = {}
        for model in ["poisson", "elo", "ml"]:
            if model_total[model] > 0:
                accuracies[model] = model_correct[model] / model_total[model]
            else:
                accuracies[model] = 0.0

        if sum(accuracies.values()) == 0:
            logger.info("No accuracy data to tune weights")
            return

        if ensemble_total > 0:
            accuracies["ensemble"] = ensemble_correct / ensemble_total

        # Normalize to get new weights (exclude ensemble from weight calc)
        base_models = {k: v for k, v in accuracies.items() if k != "ensemble"}
        total_acc = sum(base_models.values())
        if total_acc == 0:
            logger.info("No accuracy data to tune weights (all models 0)")
            return

        # ML weight: if ML models are fitted, give them a baseline weight
        # derived from the average of Poisson/Elo accuracy (since we skip
        # expensive ML re-evaluation). If not fitted, ML stays at 0.
        if self.predictor.ml_models.is_fitted and accuracies["ml"] == 0.0:
            avg_pe = (accuracies["poisson"] + accuracies["elo"]) / 2
            accuracies["ml"] = avg_pe * 0.6  # slight discount vs proven models
            base_models["ml"] = accuracies["ml"]
            total_acc = sum(base_models.values())

        new_weights = {
            "poisson": round(base_models["poisson"] / total_acc, 3),
            "elo": round(base_models["elo"] / total_acc, 3),
            "xgboost": round((base_models["ml"] / total_acc) * 0.6, 3),
            "random_forest": round((base_models["ml"] / total_acc) * 0.4, 3),
        }

        # Save tuned weights
        weights_path = Path("data/models/ensemble_weights.json")
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        weights_path.write_text(json.dumps(new_weights, indent=2))

        logger.info(f"Tuned ensemble weights: {new_weights}")
        logger.info(f"Model accuracies: {accuracies}")

        # Apply immediately
        self.predictor.weights = new_weights

        # Model Confidence Calibration from data collected above
        try:
            cal_factors = {}
            for model, data in model_predictions.items():
                if len(data) < 15:
                    cal_factors[model] = 1.0
                    continue
                bins = [(0.33, 0.45), (0.45, 0.55), (0.55, 0.65), (0.65, 0.75), (0.75, 1.01)]
                total_error = 0.0
                n_bins_used = 0
                for lo, hi in bins:
                    in_bin = [(p, h) for p, h in data if lo <= p < hi]
                    if len(in_bin) < 3:
                        continue
                    mean_pred = sum(p for p, _ in in_bin) / len(in_bin)
                    mean_hit = sum(h for _, h in in_bin) / len(in_bin)
                    total_error += abs(mean_pred - mean_hit)
                    n_bins_used += 1
                if n_bins_used > 0:
                    avg_cal_error = total_error / n_bins_used
                    cal_factors[model] = round(max(0.6, 1.0 - avg_cal_error), 3)
                else:
                    cal_factors[model] = 1.0

            cal_path = Path("data/models/calibration.json")
            cal_path.parent.mkdir(parents=True, exist_ok=True)
            cal_path.write_text(json.dumps(cal_factors, indent=2))
            self.predictor.calibration_factors.update(cal_factors)
            logger.info(f"Calibration factors: {cal_factors}")
        except Exception as e:
            logger.warning(f"Calibration computation failed: {e}")

        # Save Bayesian weights (updated in the main loop above)
        bayesian.save()
        bw_summary = bayesian.get_league_summary()
        logger.info(f"Bayesian weights updated: {len(bw_summary) - 1} leagues learned")
        for lg, info in bw_summary.items():
            if lg == "global":
                logger.info(f"  Global weights: {info}")
            elif isinstance(info, dict) and info.get("observations", 0) > 0:
                logger.debug(f"  {lg}: {info['weights']} ({info['observations']} obs)")

        return {"weights": new_weights, "accuracies": accuracies}

    async def train_ml_models(self, max_samples: int = 2000):
        """Train ML models on historical match data.

        Extracts features from completed matches, creates labels from results,
        and trains XGBoost/RandomForest/LogisticRegression models.

        Args:
            max_samples: Maximum matches to use for training (most recent).
                         More = better accuracy but slower. Default 2000.
        """
        logger.info("Starting ML model training...")

        # Fit Poisson/Elo first (needed for feature context) — skip if already fitted
        # (e.g. when called from learn_from_settled which fits before calling us)
        if not self.predictor.poisson._team_strengths:
            self.predictor.fit()
            self.feature_engineer.elo_ratings = self.predictor.elo.ratings

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
                    "match_date": m.match_date,
                }
                for m in reversed(matches)
            ]

        logger.info(f"Using {len(match_data)} most recent matches for training")

        if len(match_data) < 50:
            logger.warning(f"Not enough matches for ML training ({len(match_data)}, need 50+)")
            return

        # Build feature matrix and labels — process in parallel batches
        # Hard time budget prevents ML training from blowing the CI timeout.
        import time as _timer
        _ML_TRAIN_BUDGET_S = 1440  # 24 minutes (fits in 35-min CI step timeout)
        _ml_deadline = _timer.monotonic() + _ML_TRAIN_BUDGET_S

        # Clear standings cache so monthly-coarsened keys start fresh
        self.feature_engineer.team_features.clear_standings_cache()

        X_list = []
        y_list = []        # 1X2: 0=away, 1=draw, 2=home
        y_goals_list = []  # over/under 2.5: 1=over, 0=under/equal
        feature_names = None
        skipped = 0
        # Neon PostgreSQL has ~50ms network latency per query; limit concurrency
        # to avoid connection pool exhaustion. SQLite is local so 50 is fine.
        BATCH_SIZE = 10 if self.db.is_postgres else 50

        for batch_start in range(0, len(match_data), BATCH_SIZE):
            if _timer.monotonic() > _ml_deadline:
                logger.warning(
                    f"ML training time budget exhausted ({_ML_TRAIN_BUDGET_S // 60} min) "
                    f"after {len(X_list)} valid samples — proceeding with partial data"
                )
                break

            batch = match_data[batch_start:batch_start + BATCH_SIZE]

            # Fan out: compute features for all matches in the batch concurrently.
            # for_training=True skips weather/news API calls and coarsens
            # league standings cache to monthly → ~6x fewer DB queries per match.
            # create_features with for_training=True is effectively synchronous
            # (no HTTP calls), so we use run_in_executor to parallelize the
            # synchronous DB queries across threads instead of blocking the
            # event loop with asyncio.gather on sync work.
            import functools
            _loop = asyncio.get_event_loop()

            async def _features_in_thread(md):
                return await _loop.run_in_executor(
                    None,
                    functools.partial(
                        _sync_create_features,
                        self.feature_engineer, md["id"], md["match_date"],
                    ),
                )

            batch_results = await asyncio.gather(
                *(_features_in_thread(md) for md in batch),
                return_exceptions=True,
            )

            for md, result in zip(batch, batch_results):
                if isinstance(result, Exception) or not result:
                    skipped += 1
                    continue

                features = result
                vec = self.feature_engineer.create_feature_vector(features)
                if feature_names is None:
                    feature_names = self.feature_engineer.get_feature_names(features)

                if feature_names and len(vec) == len(feature_names):
                    X_list.append(vec)
                    if md["home_goals"] > md["away_goals"]:
                        y_list.append(2)
                    elif md["home_goals"] == md["away_goals"]:
                        y_list.append(1)
                    else:
                        y_list.append(0)
                    # Over 2.5 label: total goals > 2 (integer goals, so >2 means >=3)
                    y_goals_list.append(
                        1 if (md["home_goals"] + md["away_goals"]) > 2 else 0
                    )
                else:
                    skipped += 1

            processed = min(batch_start + BATCH_SIZE, len(match_data))
            if processed % 25 == 0 or processed == len(match_data):
                logger.info(f"Processed {processed}/{len(match_data)} matches ({len(X_list)} valid)...")

        if len(X_list) < 50:
            logger.warning(f"Only {len(X_list)} valid samples (skipped {skipped}), need 50+")
            return

        X = np.array(X_list)
        y = np.array(y_list)
        y_goals = np.array(y_goals_list)

        logger.info(f"Training on {len(X)} samples, {X.shape[1]} features (skipped {skipped})")

        # Train 1X2 models
        self.predictor.ml_models.fit(X, y, feature_names)
        self.predictor.ml_models.save()

        # Log feature importance
        importance = self.predictor.ml_models.get_feature_importance()
        for model_name, features in importance.items():
            top5 = features[:5]
            logger.info(f"{model_name} top features: {', '.join(f'{n}={v:.3f}' for n, v in top5)}")

        # Train dedicated over/under 2.5 goals model
        over25_rate = float(np.mean(y_goals))
        logger.info(
            f"Training GoalsMLModel: over 2.5 base rate = {over25_rate:.1%} "
            f"({int(y_goals.sum())}/{len(y_goals)} matches)"
        )
        self.predictor.goals_model.fit(X, y_goals, feature_names)
        self.predictor.goals_model.save()

        logger.info("ML model training complete")

    def _ml_models_stale(self, max_age_days: int = 3) -> bool:
        """Check if ML models are older than max_age_days and need retraining.

        Checks both the 1X2 classifier and the GoalsMLModel — if either is
        stale (or never trained), returns True so both get retrained together.
        """
        from datetime import timezone

        for label, model in [
            ("1X2", self.predictor.ml_models),
            ("Goals", self.predictor.goals_model),
        ]:
            trained_at = getattr(model, "trained_at", None)
            if not trained_at:
                logger.debug(f"ML model '{label}' has no trained_at — treating as stale")
                return True
            try:
                trained_dt = datetime.fromisoformat(trained_at)
                if trained_dt.tzinfo is None:
                    trained_dt = trained_dt.replace(tzinfo=timezone.utc)
                age_days = (utcnow() - trained_dt).days
                if age_days >= max_age_days:
                    return True
            except Exception as e:
                logger.warning(f"ML model '{label}' trained_at parse failed ({trained_at!r}): {e}")
                return True
        return False

    async def learn_from_settled(self):
        """Post-settlement learning: update all model feedback loops.

        Called automatically after settle_predictions() to ensure every CI
        run learns from newly settled picks without requiring a separate
        --tune command. This closes the feedback loop so models improve
        workflow-to-workflow.

        Steps (with dependency awareness):
        1. Fit Poisson/Elo on latest match results
        2. Tune ensemble weights (depends on step 1 — uses fresh Poisson strengths)
        3. Auto-calibrate EV threshold (persisted)
        4. Retrain ML models if stale (>3 days since last training)
        """
        logger.info("Post-settlement learning: updating model feedback loops")

        # 1. Refit Poisson/Elo with latest results
        poisson_ok = False
        try:
            self.predictor.fit()
            self.feature_engineer.elo_ratings = self.predictor.elo.ratings
            poisson_ok = True
            logger.info("Poisson/Elo refitted with latest results")
        except Exception as e:
            logger.warning(f"Poisson/Elo refit failed: {e}")

        # 2. Tune ensemble weights + Bayesian weights + calibration
        # Skip if Poisson refit failed — tuning on stale weights would be misleading
        if poisson_ok:
            try:
                logger.info("Starting ensemble weight tuning (may take several minutes)...")
                result = await asyncio.wait_for(
                    self.tune_ensemble_weights(),
                    timeout=600,  # 10 min max — prevent CI hangs
                )
                if result:
                    logger.info(
                        f"Ensemble weights tuned: {result['weights']} "
                        f"(accuracies: {result['accuracies']})"
                    )
            except asyncio.TimeoutError:
                logger.warning("Ensemble weight tuning timed out after 10 min — skipping")
            except Exception as e:
                logger.warning(f"Ensemble weight tuning failed: {e}")
        else:
            logger.warning("Skipping ensemble tuning — Poisson refit failed")

        # 3. EV threshold calibration (persisted to disk)
        try:
            self._auto_calibrate_ev_threshold()
        except Exception as e:
            logger.warning(f"EV calibration failed: {e}")

        # 4. ML retrain is deferred to --train or --update (takes ~11 min,
        # too expensive for --settle which runs under a 25-min CI timeout).
        max_age = self.config.get("models.ml_retrain_days", 3)
        if self._ml_models_stale(max_age_days=max_age):
            stale_info = getattr(self.predictor.ml_models, "trained_at", "never")
            logger.info(
                f"ML models stale (last trained: {stale_info}) — "
                f"skipping retrain during settle, will retrain on next --train/--update"
            )

        logger.info("Post-settlement learning complete")

    def _auto_calibrate_ev_threshold(self):
        """Dynamically adjust min EV threshold based on recent hit rate.

        Looks at the last N settled picks and computes hit rate. Compared
        to the expected baseline (~50-55%), adjusts min_ev:
        - Hit rate > 60%: hot streak → tighten min_ev by up to +2pp
        - Hit rate 45-60%: normal range → keep base min_ev
        - Hit rate < 45%: cold streak → loosen min_ev by up to -1.5pp

        This prevents over-betting during cold streaks (by demanding higher
        edge) and captures more volume during hot streaks.  Adjustments are
        bounded: min_ev never goes below 1% or above 8%.

        The calibrated threshold is persisted to data/models/ev_threshold.json
        so it survives across CI runs.
        """
        from src.models.ml_models import MODELS_DIR as _MODELS_DIR

        lookback = self.config.get("models.ev_calibration_lookback", 40)
        base_ev = self.config.betting.get("min_expected_value", 0.03)
        ev_path = _MODELS_DIR / "ev_threshold.json"

        # Load previously persisted threshold (used as starting point for --picks
        # runs; during learn_from_settled it will be recalculated and re-persisted).
        prev_ev = base_ev
        try:
            if ev_path.exists():
                saved = json.loads(ev_path.read_text())
                saved_ev = saved.get("min_ev")
                if saved_ev and 0.01 <= saved_ev <= 0.08:
                    prev_ev = saved_ev
                    self.value_calculator.min_ev = saved_ev
                    logger.debug(f"Loaded persisted EV threshold: {saved_ev:.1%}")
        except Exception:
            pass

        try:
            with self.db.get_session() as session:
                recent = (
                    session.query(SavedPick)
                    .filter(SavedPick.result.isnot(None))
                    .order_by(SavedPick.pick_date.desc(), SavedPick.id.desc())
                    .limit(lookback)
                    .all()
                )
                if len(recent) < 15:
                    return  # not enough data

                wins = sum(1 for p in recent if p.result == "win")
                hit_rate = wins / len(recent)

                if hit_rate > 0.60:
                    # Hot streak: tighten — be more selective (raise min EV)
                    # Scale: 60% → +0pp, 75%+ → +2pp
                    bonus = min((hit_rate - 0.60) / 0.15, 1.0) * 0.02
                    new_ev = base_ev + bonus
                elif hit_rate < 0.45:
                    # Cold streak: loosen slightly — lower min EV by at most 0.5pp.
                    # Capped tightly because loosening on a bad run admits more poor
                    # bets and accelerates the drawdown. Primary protection is the
                    # drawdown circuit breaker; EV floor is a secondary guard.
                    # Scale: 45% → -0pp, 30%- → -0.5pp
                    penalty = min((0.45 - hit_rate) / 0.15, 1.0) * 0.005
                    new_ev = base_ev - penalty
                else:
                    new_ev = base_ev  # normal range, keep base

                # Clamp to sane bounds.
                # Floor = base_ev - 0.5pp to prevent runaway loosening on cold streaks
                # (previously 0.01 allowed threshold to fall to 1%, far below useful range).
                ev_floor = max(base_ev - 0.005, 0.01)
                new_ev = max(ev_floor, min(0.08, round(new_ev, 4)))

                if abs(new_ev - prev_ev) > 0.001:
                    self.value_calculator.min_ev = new_ev
                    direction = "tightened" if new_ev > base_ev else "loosened"
                    logger.info(
                        f"EV auto-calibration: {direction} min_ev from "
                        f"{base_ev:.1%} → {new_ev:.1%} "
                        f"(hit rate={hit_rate:.0%} over last {len(recent)} picks)"
                    )

                # Persist to disk so next CI run starts from this threshold
                ev_path.parent.mkdir(parents=True, exist_ok=True)
                ev_path.write_text(json.dumps({
                    "min_ev": new_ev,
                    "hit_rate": round(hit_rate, 4),
                    "n_picks": len(recent),
                    "updated_at": utcnow().isoformat(),
                }, indent=2))
        except Exception as e:
            logger.warning(f"EV auto-calibration failed: {e}")

    # Pairs of (selection_A, selection_B) that are positively correlated —
    # if one hits, the other is significantly more likely to hit too.
    _CORRELATED_PAIRS = {
        # Home win correlates with high-scoring outcomes
        ("Home Win", "Over 2.5"),
        ("Home Win", "Home Over 1.5"),
        # Away win correlates with high-scoring outcomes
        ("Away Win", "Over 2.5"),
        ("Away Win", "Away Over 1.5"),
        # Draw correlates with low-scoring outcomes
        ("Draw", "Under 2.5"),
        # BTTS correlates with over goals
        ("Yes", "Over 2.5"),        # BTTS Yes + Over 2.5
        # Under markets are correlated with each other
        ("Under 2.5", "Under 1.5"),
        ("Under 2.5", "Home Under 1.5"),
        ("Under 2.5", "Away Under 1.5"),
        # Over markets correlate with each other
        ("Over 2.5", "Home Over 1.5"),
        ("Over 2.5", "Away Over 1.5"),
        # BTTS No correlates with under goals and team unders
        ("No", "Under 2.5"),
        ("No", "Home Under 1.5"),
        ("No", "Away Under 1.5"),
    }

    def _filter_correlated_picks(
        self, picks: List[BetRecommendation]
    ) -> List[BetRecommendation]:
        """Remove the lower-EV pick from correlated pairs on the same match.

        When two picks on the same match are positively correlated (e.g.
        Home Win + Over 2.5), keeping both over-concentrates risk. This
        method detects such pairs and drops the one with lower
        EV × confidence, logging the removal.
        """
        from collections import defaultdict

        by_match: dict = defaultdict(list)
        for p in picks:
            by_match[p.match_id].append(p)

        to_remove: set = set()  # indices in `picks`
        pick_index = {id(p): i for i, p in enumerate(picks)}

        for match_id, group in by_match.items():
            if len(group) < 2:
                continue
            # Check all pairs within this match
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    a, b = group[i], group[j]
                    pair = (a.selection, b.selection)
                    pair_rev = (b.selection, a.selection)
                    if pair in self._CORRELATED_PAIRS or pair_rev in self._CORRELATED_PAIRS:
                        # Drop the lower-EV one
                        score_a = a.expected_value * a.confidence
                        score_b = b.expected_value * b.confidence
                        loser = b if score_a >= score_b else a
                        winner = a if score_a >= score_b else b
                        idx = pick_index[id(loser)]
                        if idx not in to_remove:
                            to_remove.add(idx)
                            logger.info(
                                f"Correlation filter: dropping '{loser.selection}' "
                                f"(EV={loser.expected_value:.1%}) for {loser.match} — "
                                f"correlated with '{winner.selection}' "
                                f"(EV={winner.expected_value:.1%})"
                            )

        if to_remove:
            picks = [p for i, p in enumerate(picks) if i not in to_remove]
        return picks

    def _get_drawdown_multiplier(self) -> float:
        """Compute a stake multiplier based on recent bankroll drawdown.

        Looks at the last N settled picks (configurable via
        drawdown_lookback_picks, default 30) and computes cumulative P&L
        as a fraction of total staked.  Three zones:

        - ROI >= reduce_threshold (default -0.10):  normal (1.0×)
        - reduce < ROI < pause:  linear scale from 1.0× down to 0.0×
        - ROI <= pause_threshold (default -0.30):  pause all betting (0.0×)

        Returns a multiplier in [0.0, 1.0].
        """
        lookback = self.config.get("models.drawdown_lookback_picks", 30)
        reduce_at = self.config.get("models.drawdown_reduce_threshold", -0.10)
        pause_at = self.config.get("models.drawdown_pause_threshold", -0.30)

        # Guard: if thresholds are equal or inverted, no interpolation possible
        if reduce_at <= pause_at:
            return 1.0

        try:
            with self.db.get_session() as session:
                recent = (
                    session.query(SavedPick)
                    .filter(SavedPick.result.isnot(None))
                    .order_by(SavedPick.pick_date.desc(), SavedPick.id.desc())
                    .limit(lookback)
                    .all()
                )
                if len(recent) < 10:
                    return 1.0  # not enough data to judge

                total_staked = 0.0
                total_profit = 0.0
                for p in recent:
                    if p.result not in ("win", "loss"):
                        continue  # skip void/push picks
                    stake = p.kelly_stake_percentage or 1.0
                    total_staked += stake
                    if p.result == "win":
                        total_profit += stake * (p.odds - 1)
                    else:
                        total_profit -= stake

                if total_staked == 0:
                    return 1.0

                roi = total_profit / total_staked

                if roi >= reduce_at:
                    return 1.0
                if roi <= pause_at:
                    logger.warning(
                        f"Drawdown breaker: ROI={roi:.1%} over last "
                        f"{len(recent)} picks (pause threshold={pause_at:.0%})"
                    )
                    return 0.0

                # Linear interpolation between reduce_at (1.0) and pause_at (0.0)
                multiplier = (roi - pause_at) / (reduce_at - pause_at)
                logger.info(
                    f"Drawdown breaker: ROI={roi:.1%} over last "
                    f"{len(recent)} picks -> stake multiplier={multiplier:.2f}"
                )
                return round(multiplier, 2)

        except Exception as e:
            logger.warning(f"Drawdown circuit breaker check failed: {e}")
            return 1.0

    def _build_context(self, features: Dict, injury_report: Dict) -> Dict:
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

        # xG data for decision support
        context["home_xg_avg"] = features.get("home_xg_avg", 0.0)
        context["away_xg_avg"] = features.get("away_xg_avg", 0.0)
        context["home_xg_overperformance"] = features.get("home_xg_overperformance", 0.0)
        context["away_xg_overperformance"] = features.get("away_xg_overperformance", 0.0)

        # Pass raw features dict for model agreement checks on goals markets
        context["features_dict"] = features

        return context

    async def shutdown(self):
        """Clean up resources."""
        self.scraper.close_driver()
        await self.scraper.close()
        await self.apifootball.close()
        await self.injury_tracker.close()
        await self.historical_loader.close()
        await self.theodds.close()
        logger.info("Agent shutdown complete")



async def main():
    """CLI entry point."""
    import sys
    from pathlib import Path
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    except Exception:
        pass

    agent = FootballBettingAgent()

    if len(sys.argv) < 2:
        print("Usage: python -m src.agent.betting_agent <command>")
        print("\nCommands:")
        print("  --init              Initialize database and collect data")
        print("  --update            Run daily data update, skipping Flashscore results (--skip-ml-retrain to defer ML)")
        print("  --update-results    Scrape Flashscore results for all leagues (run after --picks in CI)")
        print("  --picks             Show today's value picks")
        print("  --settle            Settle pending picks with actual results")
        print("  --report            Send comprehensive performance report to Telegram")
        print("  --stats             Show prediction statistics")
        print("  --train             Train ML models on historical data")
        print("  --tune              Tune ensemble weights from recent results")
        print("  --analyze <id>      Analyze a specific match")
        print("  --backfill-history  Fetch historical data for low-coverage teams")
        print("  --telegram-setup    Setup Telegram bot notifications")
        print("  --telegram-test     Send a test Telegram message")
        print("  --telegram-welcome  Send group welcome/info message to Telegram")
        return

    command = sys.argv[1]

    try:
        if command == "--init":
            print("Initializing database and running first data collection...")
            await agent.daily_update()
            print("Initialization complete.")

        elif command == "--update":
            skip_ml = "--skip-ml-retrain" in sys.argv
            print("Running daily update (Flashscore results deferred to --update-results)...")
            await agent.daily_update(skip_ml_retrain=skip_ml, skip_flashscore_results=True)
            print("Update complete.")

        elif command == "--update-results":
            print("Scraping Flashscore results for all leagues...")
            await agent.scrape_results()
            print("Results scrape complete.")

        elif command == "--train":
            print("Training ML models on historical data...")
            # Use fewer samples when running against remote Neon DB (CI)
            # to avoid 30K+ network round-trips for feature extraction.
            from src.data.database import get_db as _get_db_check
            _db_tmp = _get_db_check()
            _max = 200 if _db_tmp.is_postgres else 2000
            await agent.train_ml_models(max_samples=_max)
            print("ML training complete.")

        elif command == "--picks":
            # Parse optional --leagues filter: --picks --leagues eng1,eng2
            league_filter = None
            for i, arg in enumerate(sys.argv[2:], start=2):
                if arg == "--leagues" and i + 1 < len(sys.argv):
                    league_filter = [l.strip() for l in sys.argv[i + 1].split(",")]
                    break
            agent.predictor.fit()
            agent.feature_engineer.elo_ratings = agent.predictor.elo.ratings
            picks, new_picks = await agent.get_daily_picks(leagues=league_filter)
            if not picks:
                print("No value picks found for today.")
            else:
                # Send to Telegram only the picks that are NEW this run.
                # On a re-run on the same day, previously-sent picks are excluded
                # so users don't receive duplicate notifications.
                if agent.telegram.enabled:
                    if new_picks:
                        stats = agent.get_stats()
                        await agent.telegram.send_daily_picks(new_picks, stats=stats)
                        print(f"\nPicks sent to Telegram! ({len(new_picks)} new)")
                    else:
                        print(f"\nNo new picks — Telegram not re-notified (all {len(picks)} already saved).")

                # Group by league for organized output
                from src.reporting.telegram_bot import LEAGUE_DISPLAY
                picks_by_league = {}
                for pick in picks:
                    lg = pick.league or "Other"
                    picks_by_league.setdefault(lg, []).append(pick)

                pick_num = 0
                for league_key in sorted(picks_by_league.keys()):
                    league_picks = picks_by_league[league_key]
                    league_name = LEAGUE_DISPLAY.get(league_key, league_key)
                    league_display = league_name.encode("ascii", "replace").decode()
                    print(f"\n{'='*60}")
                    print(f"  {league_display}")
                    print(f"{'='*60}")

                    for pick in league_picks:
                        pick_num += 1
                        match_name = pick.match.encode("ascii", "replace").decode()
                        reasoning = pick.reasoning.encode("ascii", "replace").decode()

                        agreement_tag = ""
                        if pick.model_agreement:
                            agreement_tag = f" [{pick.model_agreement.upper()}]"

                        print(f"\n  Pick #{pick_num}: {match_name}")
                        if pick.predicted_xg:
                            print(f"    xG: {pick.predicted_xg}")
                        print(f"    Bet: {pick.selection} @ {pick.odds:.2f}")
                        print(f"    EV: {pick.expected_value:.1%} | Conf: {pick.confidence:.1%} | Risk: {pick.risk_level}")
                        print(f"    Stake: {pick.kelly_stake_percentage:.1f}% of bankroll")
                        if pick.model_agreement:
                            line = f"    Models: {pick.model_agreement}{agreement_tag}"
                            if pick.models_for:
                                line += f" - {pick.models_for} agree"
                            if pick.models_against:
                                line += f" | {pick.models_against} disagree"
                            print(line)
                        if pick.xg_edge:
                            print(f"    xG edge: {pick.xg_edge}")
                        if pick.used_fallback_odds:
                            print(f"    !! Estimated odds (no bookmaker data)")
                        print(f"    Reasoning: {reasoning}")

        elif command == "--settle":
            print("Settling pending picks against actual results...")
            settled_picks = await agent.settle_predictions()
            stats = agent.get_stats()
            all_time = stats.get("all_time", {})
            print(f"\nSettled: {len(settled_picks)} picks")
            print(f"All time: {all_time.get('total', 0)} picks, Win rate: {all_time.get('win_rate', 0):.1%}")
            if stats.get("avg_clv") is not None:
                print(f"Avg CLV: {stats['avg_clv']:+.3f}")
            print(f"Pending: {stats.get('pending', 0)} picks")

            # Send settlement report to Telegram
            if settled_picks and agent.telegram.enabled:
                await agent.telegram.send_settlement_report(settled_picks, stats)
                print("Settlement report sent to Telegram!")

            # Post-settlement learning: update weights, calibration, retrain if stale
            if settled_picks:
                print("Running post-settlement learning...")
                await agent.learn_from_settled()
                print("Learning complete — models updated.")

        elif command == "--stats":
            await agent.settle_predictions()  # Settle any new results first
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

                by_league = stats.get("by_league", {})
                if by_league:
                    print(f"\nBy League:")
                    for lg, ls in sorted(by_league.items(), key=lambda x: x[1].get("total", 0), reverse=True):
                        if ls.get("total", 0) > 0:
                            print(f"  {lg}: {ls['wins']}W-{ls['losses']}L ({ls['win_rate']:.1%}) ROI: {ls['roi']:.1%}")

                cov = stats.get("model_coverage", {})
                if cov:
                    print(f"\nModel Coverage:")
                    print(f"  Poisson: {cov.get('poisson_teams', 0)} teams with strength data")
                    print(f"  Elo: {cov.get('elo_teams', 0)} teams with ratings")
                    print(f"  ML Models: {'Fitted' if cov.get('ml_fitted') else 'Not fitted'}")

                odds_src = stats.get("odds_source", {})
                real = odds_src.get("real_odds", {})
                fb = odds_src.get("fallback_odds", {})
                if real.get("total", 0) or fb.get("total", 0):
                    print(f"\nOdds Source Breakdown:")
                    if real.get("total", 0):
                        print(f"  Real odds: {real['wins']}W-{real['losses']}L ({real['win_rate']:.1%}) ROI: {real['roi']:.1%}")
                    if fb.get("total", 0):
                        print(f"  Fallback odds: {fb['wins']}W-{fb['losses']}L ({fb['win_rate']:.1%}) ROI: {fb['roi']:.1%}")

                cal = stats.get("calibration", {})
                if cal:
                    print(f"\nCalibration (predicted vs actual):")
                    for bucket, data in sorted(cal.items()):
                        pred = data["predicted_avg"]
                        actual = data["actual_win_rate"]
                        n = data["count"]
                        gap = actual - pred
                        indicator = "OK" if abs(gap) < 0.10 else ("OVER" if gap > 0 else "UNDER")
                        print(f"  {bucket}: predicted {pred:.0%} / actual {actual:.0%} (n={n}) [{indicator}]")

                stale = stats.get("stale_picks", 0)
                if stale > 0:
                    print(f"\n!! WARNING: {stale} picks pending >48h — check settlement logic !!")

                brier = stats.get("brier_score")
                avg_clv = stats.get("avg_clv")
                if brier is not None or avg_clv is not None:
                    print(f"\nModel Quality:")
                    if brier is not None:
                        cal_label = "good" if brier < 0.20 else ("fair" if brier < 0.25 else "poor")
                        print(f"  Brier Score: {brier:.4f} ({cal_label})")
                    if avg_clv is not None:
                        print(f"  Avg CLV: {avg_clv:+.4f} ({'edge' if avg_clv > 0 else 'no edge'})")

        elif command == "--report":
            print("Generating performance report...")
            await agent.settle_predictions()
            stats = agent.get_stats()
            if agent.telegram.enabled:
                await agent.telegram.send_performance_report(stats)
                print("Performance report sent to Telegram!")
            else:
                print("Telegram not enabled — printing to console:")
                at = stats.get("all_time", {})
                print(f"  All time: {at.get('total', 0)} picks, "
                      f"{at.get('wins', 0)}W-{at.get('losses', 0)}L "
                      f"({at.get('win_rate', 0):.1%}) ROI: {at.get('roi', 0):.1%}")
                brier = stats.get("brier_score")
                avg_clv = stats.get("avg_clv")
                if brier is not None:
                    print(f"  Brier Score: {brier:.4f}")
                if avg_clv is not None:
                    print(f"  Avg CLV: {avg_clv:+.4f}")

        elif command == "--tune":
            print("Tuning ensemble weights from recent results...")
            agent.predictor.fit()
            agent.feature_engineer.elo_ratings = agent.predictor.elo.ratings
            result = await agent.tune_ensemble_weights()
            if result:
                print(f"\nModel Accuracy (1X2 picks, last 30d):")
                for model, acc in sorted(result["accuracies"].items()):
                    print(f"  {model}: {acc:.1%}")
                print(f"\nNew weights: {json.dumps(result['weights'], indent=2)}")
            else:
                print("Not enough data to tune weights.")

        elif command == "--analyze" and len(sys.argv) > 2:
            agent.predictor.fit()
            agent.feature_engineer.elo_ratings = agent.predictor.elo.ratings
            match_id = int(sys.argv[2])
            analysis = await agent.analyze_fixture(match_id)
            print(f"\nMatch: {analysis.match_name}")
            print(f"Date: {analysis.match_date}")
            print(f"League: {analysis.league}")

            # Model predictions comparison
            print(f"\nPredictions:")
            ens = analysis.predictions.get("ensemble", {})
            poisson = analysis.predictions.get("poisson", {})
            elo = analysis.predictions.get("elo", {})
            print(f"  {'':15s} {'Ensemble':>10s} {'Poisson':>10s} {'Elo':>10s}")
            for outcome in ["home_win", "draw", "away_win"]:
                label = outcome.replace("_", " ").title()
                print(f"  {label:15s} {ens.get(outcome, 0):>9.1%} {poisson.get(outcome, 0):>9.1%} {elo.get(outcome, 0):>9.1%}")

            print(f"\n  xG: {ens.get('home_xg', 0):.2f} - {ens.get('away_xg', 0):.2f}")
            print(f"  Most Likely Score: {ens.get('most_likely_score', 'N/A')}")
            print(f"  Over 2.5: {ens.get('over_2.5', 0):.1%} | BTTS: {ens.get('btts_yes', 0):.1%}")

            # xG features
            f = analysis.features
            if f.get("home_xg_matches", 0) or f.get("away_xg_matches", 0):
                print(f"\n  xG Rolling Averages:")
                print(f"    Home: xG {f.get('home_xg_avg', 0):.2f} for, {f.get('home_xg_against_avg', 0):.2f} against (overperf: {f.get('home_xg_overperformance', 0):+.2f})")
                print(f"    Away: xG {f.get('away_xg_avg', 0):.2f} for, {f.get('away_xg_against_avg', 0):.2f} against (overperf: {f.get('away_xg_overperformance', 0):+.2f})")

            print(f"\nValue Bets: {len(analysis.recommendations)}")
            for rec in analysis.recommendations:
                agreement = f" [{rec.model_agreement}]" if rec.model_agreement else ""
                print(f"  - {rec.selection} @ {rec.odds:.2f} (EV: {rec.expected_value:.1%}, Conf: {rec.confidence:.0%}){agreement}")
                if rec.xg_edge:
                    print(f"    xG: {rec.xg_edge}")
                if rec.models_for:
                    print(f"    For: {rec.models_for}", end="")
                    if rec.models_against:
                        print(f" | Against: {rec.models_against}", end="")
                    print()

        elif command == "--backfill-history":
            print("Fetching historical match data for low-coverage teams...")
            # Step 1: football-data.org bulk fetch (free, no daily limit).
            # Fetches ALL finished matches for 9 competitions × 3 seasons ≈ 27 calls.
            print("  [1/2] football-data.org: bulk-fetching 2023/2024/2025 seasons...")
            fdo_saved = await agent.footballdataorg.backfill_historical_seasons(seasons=[2023, 2024, 2025])
            print(f"  FDO: {fdo_saved} new match records saved")
            # Step 2: API-Football backfill for teams not covered by FDO (uses leftover budget).
            print("  [2/2] API-Football: backfilling remaining low-coverage teams...")
            await agent.apifootball.backfill_team_history(min_matches=20, seasons=[2023, 2024, 2025], max_budget=80)
            print("Historical backfill complete.")

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

        elif command == "--telegram-welcome":
            if not agent.telegram.enabled:
                print("Telegram is not enabled. Run --telegram-setup first.")
            else:
                await agent.telegram.send_welcome_message()
                print("Welcome message sent to Telegram group!")

        else:
            print(f"Unknown command: {command}")

    finally:
        await agent.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
