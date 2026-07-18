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
from src.reporting.telegram_bot import (
    TelegramNotifier,
    _cold_streak_alerted_today,
    _mark_cold_streak_alerted,
)
from src.data.models import Match, Team, Odds, SavedPick, Injury
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
        # Per-run memo for analyze_fixture: get_daily_picks analyzes every fixture,
        # then the Claude pick-review re-analyzes each saved pick — reusing the
        # cached MatchAnalysis avoids running feature engineering + prediction
        # twice per match. Cleared at the start of get_daily_picks so each run is
        # fresh; keyed by match_id.
        self._analysis_cache: dict = {}

        # Reset stale ml=0.0 calibration on a fresh database. The cached
        # calibration.json may carry ml=0.0 from the old proxy-formula bug;
        # when the DB has no settled picks yet there is no basis for zeroing ML.
        self._reset_stale_ml_calibration()

        logger.info("Football Betting Agent initialized")

    def _reset_stale_ml_calibration(self) -> None:
        """Reset ml calibration factor to 1.0 when the DB has no settled picks.

        Handles the "warm cache, cold DB" scenario: ML models are loaded from
        cache but the database is fresh (no pick history). In this case any
        ml=0.0 in calibration.json is stale (left by the old proxy-formula bug)
        and must not suppress ML predictions on a brand-new database.
        """
        if self.predictor.calibration_factors.get("ml", 1.0) != 0.0:
            return
        if not self.predictor.ml_models.is_fitted:
            return
        try:
            with self.db.get_session() as session:
                settled = session.query(SavedPick).filter(
                    SavedPick.result.isnot(None)
                ).count()
            if settled == 0:
                self.predictor.calibration_factors["ml"] = 1.0
                logger.info(
                    "Fresh database detected: reset stale ml=0.0 calibration "
                    "to 1.0 (no settled picks to justify exclusion)"
                )
                cal_path = Path("data/models/calibration.json")
                try:
                    existing = json.loads(cal_path.read_text()) if cal_path.exists() else {}
                    existing["ml"] = 1.0
                    cal_path.write_text(json.dumps(existing, indent=2))
                    logger.info("Saved corrected calibration (ml=1.0) to disk")
                except Exception as _save_err:
                    logger.debug(f"Could not save corrected calibration: {_save_err}")
        except Exception as _e:
            logger.debug(f"Stale calibration reset skipped: {_e}")

    _SCRAPED_LEAGUES_FILE = Path("data/scraped_leagues.json")

    def _get_recently_scraped_leagues(self, minutes: int = 30) -> set:
        """Return leagues scraped within the last N minutes.

        Marker file format (per-league timestamps so a single global timestamp
        no longer treats all leagues as fresh whenever ANY league is scraped):
            {"leagues": {"<league>": "<iso-timestamp>", ...}}
        """
        result: set = set()

        # 1. Marker file (reliable cross-process, written by _mark_league_scraped)
        try:
            if self._SCRAPED_LEAGUES_FILE.exists():
                data = json.loads(self._SCRAPED_LEAGUES_FILE.read_text())
                cutoff = utcnow() - timedelta(minutes=minutes)
                # New format: {"leagues": {league: iso_ts}}
                leagues_field = data.get("leagues", {})
                if isinstance(leagues_field, dict):
                    for league, iso_ts in leagues_field.items():
                        try:
                            ts = datetime.fromisoformat(iso_ts)
                            if ts >= cutoff:
                                result.add(league)
                        except Exception:
                            continue
                else:
                    # Legacy format: {"timestamp": ..., "leagues": [...]} —
                    # one global timestamp.  Honour it once on read so we don't
                    # silently break callers during the upgrade window.
                    ts_str = data.get("timestamp", "")
                    if ts_str:
                        try:
                            ts = datetime.fromisoformat(ts_str)
                            if ts >= cutoff:
                                result.update(leagues_field or [])
                        except Exception:
                            pass
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

        Stores per-league timestamps so each league's freshness is tracked
        independently. Migrates legacy {"timestamp": ..., "leagues": [...]}
        format on first write.
        """
        self._scraped_leagues.add(league)
        try:
            now_iso = utcnow().isoformat()
            data: dict = {}
            if self._SCRAPED_LEAGUES_FILE.exists():
                try:
                    data = json.loads(self._SCRAPED_LEAGUES_FILE.read_text())
                except Exception:
                    data = {}
            leagues_field = data.get("leagues", {})
            if isinstance(leagues_field, list):
                # Legacy: convert list → dict using the legacy global timestamp,
                # then we'll overwrite this league with NOW below.
                legacy_ts = data.get("timestamp") or now_iso
                leagues_field = {l: legacy_ts for l in leagues_field}
            elif not isinstance(leagues_field, dict):
                leagues_field = {}
            leagues_field[league] = now_iso
            self._SCRAPED_LEAGUES_FILE.parent.mkdir(parents=True, exist_ok=True)
            self._SCRAPED_LEAGUES_FILE.write_text(
                json.dumps({"leagues": leagues_field}, indent=2)
            )
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
        # AC1: extend with DB-derived leagues not in static config (e.g. added mid-run by API-Football)
        _ordered_leagues = self._merge_flashscore_targets(_priority + _rest, _today_leagues | _pending_leagues)
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
        _fs_skip = set(self.config.get("scraping.flashscore_skip_leagues", []))
        _fs_skip_fixtures_only = set(
            self.config.get("scraping.flashscore_skip_fixtures_leagues", [])
        )
        _fs_skip_fixtures = _fs_skip | _fs_skip_fixtures_only
        _fixture_leagues = [
            l for l in _ordered_leagues
            if l in _important  # has today's fixtures or pending picks
            and l not in _fs_skip_fixtures
        ]
        _skipped_fixture_leagues = len(_ordered_leagues) - len(_fixture_leagues)
        if _fs_skip_fixtures:
            logger.info(
                f"Flashscore fixtures: skipping {sorted(_fs_skip_fixtures)} "
                f"(flashscore_skip_leagues + flashscore_skip_fixtures_leagues)"
            )
        if _skipped_fixture_leagues:
            logger.info(
                f"Fixtures: scraping {len(_fixture_leagues)} leagues with fixtures/pending, "
                f"skipping {_skipped_fixture_leagues} without"
            )
        _league_fixture_results: dict = {}
        try:
            _fixtures_deadline = _timer.monotonic() + _FIXTURES_BUDGET_S
            for league in _fixture_leagues:
                if _timer.monotonic() > _fixtures_deadline:
                    logger.warning("Flashscore fixtures: time budget exhausted, skipping remaining leagues")
                    break
                try:
                    result = await asyncio.wait_for(
                        self.scraper.scrape_league_fixtures(league), timeout=90,
                    )
                    if result is not None:
                        _league_fixture_results[league] = result
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

        try:
            await self._check_empty_fixture_leagues(_league_fixture_results, _fixture_leagues)
        except Exception as e:
            logger.warning(f"Empty-league Telegram alert failed (non-fatal): {e}")

        # 2b. API-Football (fixtures, xG, advanced stats, odds).
        # Before spending quota: flag today's fixtures that CANNOT be analyzed
        # (non-national league + a team with zero completed matches — the exact
        # class analyze_fixture skips). Odds + injuries skip them, preserving
        # budget for the coverage backfill that would actually fix those teams.
        # (2026-07-09: 23/26 unanalyzable qualifying fixtures ate ~95 requests
        # and left the backfill a budget of 2.) Fixtures created seconds later
        # inside update() aren't flagged — fixtures normally exist a day ahead.
        try:
            self.apifootball.skip_analysis_match_ids = self._unanalyzable_today()
        except Exception as _sk_e:
            logger.debug(f"skip-set computation failed (non-fatal): {_sk_e}")
            self.apifootball.skip_analysis_match_ids = set()
        try:
            await asyncio.wait_for(self.apifootball.update(), timeout=1080)  # 18 min cap
            logger.info("API-Football update complete")
        except asyncio.TimeoutError:
            logger.warning("API-Football update timed out after 10 minutes")
        except Exception as e:
            logger.error(f"API-Football update failed: {e}")

        if getattr(self.apifootball, "_account_suspended", False):
            try:
                await self.telegram.send_alert(
                    "🚨 API-Football account suspended — fixtures, xG, odds, and injury "
                    "data are unavailable. Check account at dashboard.api-football.com"
                )
            except Exception as _tg_err:
                logger.warning(f"Failed to send API suspension Telegram alert: {_tg_err}")
        elif getattr(self.apifootball, "_xg_all_failed", False):
            try:
                await self.telegram.send_alert(
                    "⚠️ xG backfill: all API-Football stats requests failed this run. "
                    "xG features will use zero values until next successful run."
                )
            except Exception as _tg_err:
                logger.warning(f"Failed to send xG failure Telegram alert: {_tg_err}")

        # 2c. The Odds API — supplemental odds for leagues with today's fixtures.
        # Free tier: 500 credits/month (~1 credit/league call). Only calls leagues
        # that actually have fixtures today to minimise credit burn.

        # Warn BEFORE the update if last-known credits are already low — this way
        # operators see the alert even if today's update exhausts the last few credits.
        from src.scrapers.theodds_scraper import _load_persisted_credits, _CREDITS_LOW_THRESHOLD
        _pre_credits = _load_persisted_credits()
        if _pre_credits is not None and _pre_credits < _CREDITS_LOW_THRESHOLD:
            _pre_warn = (
                f"⚠️ TheOddsAPI credits low (from last run): {_pre_credits} remaining. "
                f"Odds coverage will degrade when quota is exhausted."
            )
            logger.warning(_pre_warn)
            try:
                await self.telegram.send_alert(_pre_warn)
            except Exception:
                pass

        try:
            theodds_written = await asyncio.wait_for(self.theodds.update(), timeout=300)
            logger.info(f"The Odds API update complete: {theodds_written} odds rows written")
            _remaining = getattr(self.theodds, "_remaining_requests", None)
            if _remaining is not None and _remaining < _CREDITS_LOW_THRESHOLD:
                _warn_msg = (
                    f"⚠️ TheOddsAPI credits low: {_remaining} remaining this month. "
                    f"Odds coverage will degrade when quota is exhausted."
                )
                logger.warning(_warn_msg)
                # Avoid double-alerting if pre-update warning already fired for same count
                if _pre_credits is None or abs(_remaining - _pre_credits) > 5:
                    try:
                        await self.telegram.send_alert(_warn_msg)
                    except Exception as _tg_err:
                        logger.warning(f"Failed to send TheOddsAPI credit warning: {_tg_err}")
        except asyncio.TimeoutError:
            logger.warning("The Odds API update timed out after 5 minutes")
        except Exception as e:
            logger.warning(f"The Odds API update failed (non-critical): {e}")

        # 2d. Flashscore per-match stats enrichment — DISABLED.
        # API-Football (BUDGET_XG=30) is the primary stats source now and covers
        # shots/possession/xG via fast API calls (~0.5s/match). Flashscore browser
        # scraping (~12s/match) is unreliable due to Cloudflare blocking in CI.
        self.scraper.close_driver()

        # 4. Injury data (cap at 12 min — API-Football calls can be slow).
        # Today's fixtures are fetched first, sorted by league tier so that on high-
        # fixture days (88+ in run #134) the top leagues are covered before timeout.
        # Tier order: top-5 → major European → second tier → rest.
        _INJURY_TIER1 = {
            "england/premier-league", "germany/bundesliga", "spain/laliga",
            "italy/serie-a", "france/ligue-1",
        }
        _INJURY_TIER2 = {
            "netherlands/eredivisie", "portugal/primeira-liga", "turkey/super-lig",
            "belgium/jupiler-pro-league", "scotland/premiership",
            "europe/champions-league", "europe/europa-league",
        }

        def _injury_league_rank(league: str) -> int:
            if league in _INJURY_TIER1:
                return 0
            if league in _INJURY_TIER2:
                return 1
            return 2

        _today_start = datetime.combine(date.today(), datetime.min.time())
        _today_end = _today_start + timedelta(days=1)
        _injury_today_ids: list = []
        _injury_pending_ids: list = []
        try:
            with self.db.get_session() as _isess:
                # Fetch today's fixtures with league info for tier-sorting
                _today_rows = _isess.query(Match.id, Match.league).filter(
                    Match.is_fixture == True,
                    Match.match_date >= _today_start,
                    Match.match_date < _today_end,
                ).all()
                # Sort by tier so tier-1 leagues are always processed first
                _today_rows_sorted = sorted(
                    _today_rows, key=lambda r: _injury_league_rank(r[1] or "")
                )
                _injury_today_ids = [row[0] for row in _today_rows_sorted]
                _injury_pending_ids = [
                    row[0] for row in _isess.query(SavedPick.match_id)
                    .filter(SavedPick.result.is_(None))
                    .distinct()
                    .all()
                ]
        except Exception:
            pass
        _today_set = set(_injury_today_ids)
        _injury_priority_ids = _injury_today_ids + [
            mid for mid in _injury_pending_ids if mid not in _today_set
        ]
        try:
            await asyncio.wait_for(
                self.injury_tracker.update(priority_fixture_ids=_injury_priority_ids),
                timeout=720,  # 12 min (up from 8) — 88 fixtures × ~8s = ~12 min worst case
            )
            logger.info("Injury update complete")
        except asyncio.TimeoutError:
            logger.warning("Injury update timed out after 12 minutes")
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

        # 6b. Targeted backfill for low-coverage teams in today's AND tomorrow's
        # fixtures. Tomorrow's fixtures already exist (created a day ahead), so
        # backfilling their teams today spreads the API quota across two days and
        # the data is ready when the fixture is actually analyzed — qualifying
        # rounds bring many unknown minnows at once (12 on 2026-07-08).
        try:
            _today = date.today()
            _start = datetime.combine(_today, datetime.min.time())
            _end = _start + timedelta(days=2)
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
                # 25-request cap: enough for ~8 team-seasons a day while leaving
                # plenty of the 100/day quota for odds + injuries + settlement.
                _backfill_budget = min(25, self.apifootball.remaining_budget())
                logger.info(
                    f"Found {len(_low_cov_team_ids)} low-coverage teams in today's/"
                    f"tomorrow's fixtures — triggering targeted backfill "
                    f"(budget: {_backfill_budget})"
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

        # Invalidate the league baselines cache so the picks step uses the
        # freshly ingested historical data, not stats from before --update ran.
        try:
            self.feature_engineer.clear_league_cache()
        except Exception as _e:
            logger.debug(f"clear_league_cache failed: {_e}")

        logger.info("Daily update cycle complete")

    def _unanalyzable_today(self) -> set:
        """Match IDs of today's fixtures that analysis will certainly skip.

        A non-national fixture is skipped by analyze_fixture when either team
        has NO historical data at all. Detecting that from the DB (zero
        completed matches for the team) is model-independent and cheap, so the
        odds/injury fetchers can avoid spending quota on those fixtures.
        National-team fixtures are never flagged (WC: every match analyzed).
        """
        from sqlalchemy import or_ as _or
        from src.models.poisson_model import NATIONAL_TEAM_LEAGUES
        _start = datetime.combine(date.today(), datetime.min.time())
        _end = _start + timedelta(days=1)
        skip: set = set()
        with self.db.get_session() as session:
            fixtures = session.query(Match).filter(
                Match.is_fixture == True,  # noqa: E712
                Match.match_date >= _start,
                Match.match_date < _end,
            ).all()
            club = [m for m in fixtures if (m.league or "") not in NATIONAL_TEAM_LEAGUES]
            # One count query per distinct team (bounded: 2×fixtures).
            counts: dict = {}
            for m in club:
                for tid in (m.home_team_id, m.away_team_id):
                    if tid not in counts:
                        counts[tid] = session.query(Match.id).filter(
                            Match.is_fixture == False,  # noqa: E712
                            Match.home_goals.isnot(None),
                            _or(Match.home_team_id == tid, Match.away_team_id == tid),
                        ).count()
            for m in club:
                if counts.get(m.home_team_id, 0) == 0 or counts.get(m.away_team_id, 0) == 0:
                    skip.add(m.id)
        if skip:
            logger.info(
                f"Flagged {len(skip)}/{len(fixtures)} of today's fixtures as "
                f"unanalyzable (zero-history team) — odds/injury quota will skip them"
            )
        return skip

    def _should_force_pick(self, league: str, coverage_score: float) -> bool:
        """Whether a fixture with NO value pick still gets a forced tracked pick.

        National-team competitions (WC): every match, per betting.wc_pick_every_match.
        Club fixtures: only when the model has decent data on both teams —
        coverage_score >= betting.club_pick_min_coverage (0 disables). Keeps
        well-known matchups from producing empty days while minnows the model
        has never seen stay no-bet.
        """
        from src.models.poisson_model import NATIONAL_TEAM_LEAGUES
        if league in NATIONAL_TEAM_LEAGUES:
            return bool(self.config.get("betting.wc_pick_every_match", True))
        try:
            min_cov = float(self.config.get("betting.club_pick_min_coverage", 0.75) or 0)
        except (TypeError, ValueError):
            return False
        return min_cov > 0 and coverage_score >= min_cov

    async def analyze_fixture(self, match_id: int) -> MatchAnalysis:
        """Run complete analysis on a single fixture.

        Args:
            match_id: Match database ID

        Returns:
            MatchAnalysis with predictions and recommendations
        """
        if not hasattr(self, "_analysis_cache"):
            self._analysis_cache = {}
        cached = self._analysis_cache.get(match_id)
        if cached is not None:
            return cached
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
                    "opening_odds": o.opening_odds,
                }
                for o in odds_records
            ]

        # Coverage gate — skip fixtures where at least one team has no data
        # from any model (no Poisson strengths AND no Elo rating).
        # score-based check (< 0.50) was an edge case: two teams with only Elo
        # ratings scored exactly 0.50 and slipped through despite 0% historical data.
        #
        # EXCEPTION — national-team competitions (WC): a pick and a briefing must
        # exist for EVERY match. The Poisson/Elo models degrade gracefully for
        # unknown teams (league-average regression / default rating), so we
        # proceed with bland predictions, flag them as low-coverage, and let the
        # briefing LLM's web research carry the final decision.
        coverage = self.predictor.check_coverage(home_id, away_id)
        home_has_data = coverage["home_poisson"] or coverage["home_elo"]
        away_has_data = coverage["away_poisson"] or coverage["away_elo"]
        low_coverage = not home_has_data or not away_has_data
        from src.models.poisson_model import NATIONAL_TEAM_LEAGUES
        if low_coverage:
            missing = []
            if not home_has_data:
                missing.append("home")
            if not away_has_data:
                missing.append("away")
            if league not in NATIONAL_TEAM_LEAGUES:
                logger.info(
                    f"Skipping {match_name}: no historical data for {'/'.join(missing)} team "
                    f"(coverage {coverage['score']:.0%})"
                )
                return MatchAnalysis(
                    match_id=match_id, match_name=match_name, match_date=match_date,
                    league=league, features={}, predictions={}, recommendations=[],
                    injury_report={},
                )
            logger.warning(
                f"Low coverage for {match_name} ({'/'.join(missing)} team, "
                f"{coverage['score']:.0%}) — proceeding anyway (WC: every match "
                f"gets a pick + briefing; model output flagged as weak prior)"
            )
        elif league not in NATIONAL_TEAM_LEAGUES and coverage["score"] < 0.5:
            # Both teams have SOME history but it's thin/stale (e.g. a club league
            # just resumed after the WC pause) — predictions regress to the mean,
            # so selections rarely clear the confidence floor and the league
            # produces few/no picks. Surface it so the thinness is visible rather
            # than hiding inside per-match value rejections. Self-heals as results
            # accumulate; a one-off --backfill-history speeds it up.
            logger.warning(
                f"Thin model coverage for {match_name} (league {league}, "
                f"score {coverage['score']:.0%}) — predictions will be low-"
                f"confidence until the league accumulates more results"
            )

        # Generate features
        features = await self.feature_engineer.create_features(match_id)
        if low_coverage:
            features["model_low_coverage"] = 1
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
        # Expose pick-level calibration factors so value_calculator can raise the EV
        # floor for markets that systematically overestimate goal probability.
        if hasattr(self.predictor, "pick_calibration") and self.predictor.pick_calibration:
            context["pick_calibration"] = self.predictor.pick_calibration

        # Find value bets
        recommendations = self.value_calculator.find_value_bets(
            predictions, odds_data, match_name, context,
            home_team_name=home_team_name, away_team_name=away_team_name,
            match_id=match_id, league=league,
        )

        # Forced picks: when a fixture produces no value bet, still save the
        # single best bettable selection so the match gets a tracked pick —
        # for two fixture classes:
        #   1. National-team competitions (WC): EVERY match gets a pick
        #      (betting.wc_pick_every_match).
        #   2. Club fixtures where the model actually KNOWS the teams —
        #      coverage >= betting.club_pick_min_coverage (default 0.75; 0
        #      disables). Prevents whole no-pick days (2026-07-08: 7 qualifying
        #      fixtures, 0 picks) without betting blind on minnows the model
        #      has never seen. Claude's review still verifies/switches it.
        if not recommendations:
            from src.models.poisson_model import NATIONAL_TEAM_LEAGUES
            _is_national = league in NATIONAL_TEAM_LEAGUES
            if self._should_force_pick(league, coverage["score"]):
                # On low-coverage WC fixtures the model is essentially blind, so
                # let the MARKET pick the favourite (maximise win rate) instead of
                # chasing the model's noisy EV. The briefing LLM (live web
                # research) can still CHANGE the selection afterward.
                # Club forced picks carry quality floors: blended (model+market)
                # win probability must reach betting.club_pick_min_blend AND the
                # model's own EV must not be worse than betting.club_pick_min_ev
                # (the model actively disputing the market price was the losing
                # cohort: EV<-5% forced picks ran ~-13% ROI). WC pick-every-match
                # stays unfloored (a pick must always exist; the Claude review is
                # its safety net).
                _blend_floor = 0.0 if _is_national else float(
                    self.config.get("betting.club_pick_min_blend", 0.55) or 0
                )
                _ev_cfg = self.config.get("betting.club_pick_min_ev", -0.05)
                _ev_floor = None if (_is_national or _ev_cfg is None) else float(_ev_cfg)
                best = self.value_calculator.find_best_bet(
                    predictions, odds_data, match_name, context,
                    home_team_name=home_team_name, away_team_name=away_team_name,
                    match_id=match_id, league=league,
                    prefer_market=bool(low_coverage),
                    min_blend_prob=_blend_floor,
                    min_forced_ev=_ev_floor,
                )
                if best:
                    best.is_forced = True  # one-per-match guard in _save_picks
                    recommendations = [best]
                    _basis = "market favourite (thin data)" if low_coverage else "model EV"
                    _kind = ("WC pick-every-match" if _is_national
                             else f"Club pick (coverage {coverage['score']:.0%})")
                    logger.info(
                        f"{_kind}: {match_name} → {best.selection} "
                        f"@ {best.odds:.2f} (EV {best.expected_value:+.1%}, "
                        f"conf {best.confidence:.0%}; {_basis})"
                    )

        for rec in recommendations:
            rec.match_date = match_date

        result = MatchAnalysis(
            match_id=match_id,
            match_name=match_name,
            match_date=match_date,
            league=league,
            features=features,
            predictions=predictions,
            recommendations=recommendations,
            injury_report=injury_report,
        )
        self._analysis_cache[match_id] = result
        return result

    def _merge_flashscore_targets(self, static_leagues: list, db_leagues: set) -> list:
        """Return static config leagues extended with any DB-derived leagues not already listed.

        Static list order is preserved (AC4). New leagues are appended alphabetically.
        """
        static_set = set(static_leagues)
        extras = sorted(slug for slug in db_leagues if slug not in static_set)
        for slug in extras:
            logger.info(f"Added dynamic Flashscore target: {slug} (fixtures/pending picks, not in static config)")
        return list(static_leagues) + extras

    async def _check_empty_fixture_leagues(
        self, league_results: dict, attempted_leagues: list = None
    ) -> None:
        """Warn when tier-1 leagues return 0 fixtures or timed out during scraping.

        For major leagues (PL, Bundesliga, La Liga, Serie A, Ligue 1) a zero-fixture
        result is almost always a scraper failure rather than a genuine rest day, so we:
        - Log a WARNING (not INFO)
        - Check the DB for already-stored fixtures for that league today
        - Send a Telegram alert so operators know picks may be incomplete
        """
        _TIER1 = {
            "england/premier-league", "germany/bundesliga", "spain/laliga",
            "italy/serie-a", "france/ligue-1", "netherlands/eredivisie",
            "portugal/primeira-liga", "turkey/super-lig",
        }
        off_season = set(self.config.get("scraping.off_season_leagues", []))
        today_start = datetime.combine(date.today(), datetime.min.time())
        today_end = today_start + timedelta(days=1)

        # Leagues that were attempted but not in results at all (timed out)
        attempted = set(attempted_leagues or [])
        timed_out = attempted - set(league_results.keys())

        # Leagues that returned 0 fixtures (scrape completed but empty)
        zero_result = {
            league for league, fixtures in league_results.items()
            if len(fixtures) == 0 and league not in off_season
        }

        problem_leagues = (timed_out | zero_result) - off_season
        if not problem_leagues:
            return

        tier1_problems = problem_leagues & _TIER1
        non_tier1 = problem_leagues - _TIER1

        if non_tier1:
            logger.info(f"Flashscore: 0 fixtures / timeout for: {', '.join(sorted(non_tier1))}")

        if not tier1_problems:
            return

        # For tier-1 problems: check how many fixtures exist in DB for today
        db_counts = {}
        try:
            with self.db.get_session() as session:
                for league in tier1_problems:
                    n = session.query(Match).filter(
                        Match.league == league,
                        Match.is_fixture == True,
                        Match.match_date >= today_start,
                        Match.match_date < today_end,
                    ).count()
                    db_counts[league] = n
        except Exception:
            pass

        # Separate true scraping failures (DB has fixtures but Flashscore missed them)
        # from no-match days (both Flashscore AND DB return 0 — not a failure).
        # Timeouts are always failures regardless of DB count.
        true_failures = []   # needs Telegram alert
        no_match_days = []   # just log INFO, no alert spam

        for league in sorted(tier1_problems):
            db_n = db_counts.get(league, 0)
            if league in timed_out:
                reason = "timeout"
                true_failures.append((league, reason, db_n))
                logger.warning(
                    f"Flashscore tier-1 timeout for {league} "
                    f"— {db_n} fixtures in DB from API-Football"
                )
            elif db_n > 0:
                reason = "0 fixtures returned"
                true_failures.append((league, reason, db_n))
                logger.warning(
                    f"Flashscore tier-1 failure for {league}: 0 fixtures returned "
                    f"— {db_n} fixtures already in DB from API-Football"
                )
            else:
                no_match_days.append(league)
                logger.info(
                    f"Flashscore: 0 fixtures for tier-1 league {league} "
                    f"— no matches scheduled today (DB also empty)"
                )

        if no_match_days:
            logger.info(
                f"Flashscore: no matches today for {len(no_match_days)} tier-1 league(s): "
                + ", ".join(no_match_days)
            )

        if not true_failures:
            return  # all zeros were genuine no-match days, no alert needed

        reasons = [
            f"  • {lg}: {reason} ({db_n} fixtures in DB for today)"
            for lg, reason, db_n in true_failures
        ]
        alert = (
            f"⚠️ Flashscore scrape failed for {len(true_failures)} tier-1 league(s):\n"
            + "\n".join(reasons)
            + "\nPicks may be missing venue/referee enrichment; API-Football fixtures used as fallback."
        )
        try:
            await self.telegram.send_alert(alert)
        except Exception as _te:
            logger.warning(f"Could not send tier-1 fixture failure alert: {_te}")

    async def get_daily_picks(self, target_date: date = None,
                              max_picks_per_match: int = 2,
                              leagues: List[str] = None,
                              force: bool = False) -> List[BetRecommendation]:
        """Get high-confidence value betting picks for a specific date.

        EV and confidence thresholds are read from config (betting.min_expected_value /
        betting.min_confidence) by the ValueBettingCalculator — no need to pass them here.

        Args:
            target_date: Date to get picks for (defaults to today)
            max_picks_per_match: Maximum picks allowed per single match (default 2)
            leagues: Optional list of league keys to restrict picks to
            force: If True, skip idempotency check and regenerate even if today's picks exist

        Returns:
            Tuple of (picks, new_picks, dropped_picks)
        """
        target = target_date or date.today()
        # Fresh analysis memo for this pick run (reused by the Claude review).
        if not hasattr(self, "_analysis_cache"):
            self._analysis_cache = {}
        self._analysis_cache.clear()

        # Reload calibration from disk if it was updated by a recent --train/--tune.
        # This ensures picks always use the freshest model weights without a restart.
        _cal_path = Path("data/models/calibration.json")
        if _cal_path.exists():
            try:
                _cal = json.loads(_cal_path.read_text())
                if _cal != self.predictor.calibration_factors:
                    self.predictor.calibration_factors.update(_cal)
                    logger.info(f"Reloaded calibration factors from disk: {_cal}")

                # Apply Poisson/Elo floors using last-tuned accuracies so that
                # a failed or skipped tune run can't leave a suppressed floor value
                # persisted across the pick cycle.
                _acc_path = Path("data/models/model_accuracies.json")
                if _acc_path.exists():
                    try:
                        _accs = json.loads(_acc_path.read_text())
                        _ensemble_acc = _accs.get("ensemble", 0.0)
                        for _mdl in ("poisson", "elo"):
                            _mdl_acc = _accs.get(_mdl, 0.0)
                            if (
                                _mdl_acc > _ensemble_acc + 0.05
                                and self.predictor.calibration_factors.get(_mdl, 1.0) < 1.0
                            ):
                                self.predictor.calibration_factors[_mdl] = 1.0
                                logger.info(
                                    f"Load-time {_mdl.capitalize()} floor applied: "
                                    f"{_mdl} {_mdl_acc:.1%} > ensemble {_ensemble_acc:.1%} "
                                    f"(+5pp) — calibration restored to 1.0"
                                )
                    except Exception as _fe:
                        logger.debug(f"Could not apply load-time model floor: {_fe}")
            except Exception as _ce:
                logger.debug(f"Could not reload calibration: {_ce}")

        # Idempotency guard — skip if today's picks already exist (AC1 Story 8.3)
        if not force:
            with self.db.get_session() as _isess:
                _existing = _isess.query(SavedPick).filter(
                    SavedPick.pick_date == target
                ).count()
            if _existing > 0:
                logger.info(
                    f"Today's picks already generated ({_existing} picks). "
                    f"Use --force to regenerate."
                )
                return [], [], []

        league_label = f" (leagues: {', '.join(leagues)})" if leagues else ""
        logger.info(f"Getting daily picks for {target}{league_label}")

        with self.db.get_session() as session:
            day_start = datetime.combine(target, datetime.min.time())
            # WC 2026 is in North America (UTC-4 to UTC-7). Late evening matches
            # (e.g. 9 PM ET = 01:00 UTC) fall on the next UTC calendar day.
            # Extend the pick window to 30 hours so those fixtures are included.
            _wc_leagues = {"world/fifa-world-cup"}
            _configured_leagues = set(self.config.get("scraping.flashscore_leagues", []))
            _wc_only = bool(_wc_leagues & _configured_leagues)
            day_end = day_start + timedelta(hours=30 if _wc_only else 24)
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
            from src.utils.team_names import team_names_similar as _nm
            # seen_list: [(match_id, home_name, away_name, league, match_date, afid)]
            seen_list: list = []
            dedup_ids: list = []

            def _hours_apart(dt1, dt2):
                """Absolute time difference in hours (handles midnight wrap)."""
                return abs((dt1 - dt2).total_seconds()) / 3600

            for f in fixtures:
                ht = session.get(Team, f.home_team_id)
                at = session.get(Team, f.away_team_id)
                h_name = ht.name if ht else ""
                a_name = at.name if at else ""
                # Same fixture if the API-Football id matches (reliable even when
                # the two rows carry different dates — the scraper briefly stored
                # one fixture under two dates), OR same league + kickoff within 2h
                # + both team names match (cross-source rows with no shared afid).
                dup_idx = None
                for idx, (e_id, e_home, e_away, e_league, e_dt, e_afid) in enumerate(seen_list):
                    same_afid = f.apifootball_id is not None and f.apifootball_id == e_afid
                    same_fixture = (
                        e_league == f.league and _hours_apart(e_dt, f.match_date) <= 2
                        and _nm(h_name, e_home) and _nm(a_name, e_away)
                    )
                    if same_afid or same_fixture:
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
                        seen_list[dup_idx] = (f.id, h_name, a_name, f.league, f.match_date, f.apifootball_id)
                        logger.debug(f"Dedup: replaced fixture {existing_id} with {f.id} ({h_name}, more odds)")
                    else:
                        logger.debug(f"Dedup: skipping duplicate fixture {f.id} ({h_name}, fewer odds)")
                else:
                    seen_list.append((f.id, h_name, a_name, f.league, f.match_date, f.apifootball_id))
                    dedup_ids.append(f.id)

            fixture_ids = dedup_ids
            if len(dedup_ids) < len(fixtures):
                logger.info(f"Deduplicated {len(fixtures)} → {len(dedup_ids)} fixtures")

        if not fixture_ids:
            logger.info(f"No fixtures found for {target}")
            return [], [], []

        # ── API-Football odds fallback ─────────────────────────────────────────
        # For fixtures that have NO real bookmaker odds (zero odds, or only
        # "Flashscore" display odds), try API-Football if they have an
        # apifootball_id and we have budget remaining (free tier = 100/day).
        if self.apifootball.enabled:
            _apifb_budget_remaining = min(self.apifootball.remaining_budget(), 40)

            # Over 1.5 is never provided by TheOdds API so we fall back to AF
            # for it — but only for top leagues where market liquidity is high
            # and AF odds quality is reliable.  Skipping lower divisions saves
            # ~25 requests/day (from ~39 to ~12) on busy fixtures days.
            _over15_leagues = set(self.config.get(
                "betting.over15_priority_leagues",
                [
                    "england/premier-league",
                    "england/championship",
                    "spain/laliga",
                    "germany/bundesliga",
                    "italy/serie-a",
                    "france/ligue-1",
                    "netherlands/eredivisie",
                    "portugal/primeira-liga",
                    "belgium/jupiler-pro-league",
                    "turkey/super-lig",
                    "scotland/premiership",
                    "europe/champions-league",
                    "europe/europa-league",
                    "europe/europa-conference-league",
                ],
            ))

            with self.db.get_session() as session:
                # Tier-1: matches with zero real bookmaker odds (highest priority)
                apifb_fallback = []
                # Tier-2: matches that have TheOddsAPI 1X2/2.5 odds but are
                # missing Over 1.5 (TheOddsAPI never provides that line).
                # Limited to priority leagues to preserve the 100/day AF quota.
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
                        elif m.league in _over15_leagues:
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

        # Value threshold auto-calibration: hot streak → relax, cold streak → tighten.
        self._recent_roi = None
        self._recent_roi_n = 0
        self._auto_calibrate_ev_threshold()
        # Alert operators when the model is in a sustained cold streak (ROI < -15%).
        # Guard: only send once per day (same alert fires from --settle AND --picks).
        _cal_roi = getattr(self, "_recent_roi", None)
        if _cal_roi is not None and _cal_roi < -0.15:
            # Build per-market breakdown so operators can diagnose which markets are losing
            _mkt_breakdown = ""
            try:
                lookback = self.config.get("models.ev_calibration_lookback", 40)
                with self.db.get_session() as _cs:
                    _recent_picks = (
                        _cs.query(SavedPick)
                        .filter(SavedPick.result.isnot(None))
                        .order_by(SavedPick.pick_date.desc(), SavedPick.id.desc())
                        .limit(lookback)
                        .all()
                    )
                    _mkt_roi: dict = {}
                    for _p in _recent_picks:
                        if not _p.odds or _p.odds <= 1.0:
                            continue
                        _mkt = _p.market or "Unknown"
                        _profit = (_p.odds - 1) if _p.result == "win" else -1.0
                        _mkt_roi.setdefault(_mkt, []).append(_profit)
                    _parts = []
                    for _mkt, _profits in sorted(_mkt_roi.items()):
                        _r = sum(_profits) / len(_profits)
                        _parts.append(f"{_mkt}: {_r:+.0%} ({len(_profits)})")
                    if _parts:
                        _mkt_breakdown = "\nBy market: " + " | ".join(_parts)
            except Exception:
                pass
            _cold_msg = (
                f"⚠️ Cold streak alert: ROI={_cal_roi:+.1%} over last "
                f"{getattr(self, '_recent_roi_n', '?')} settled picks — "
                f"model is underperforming. EV threshold tightened to "
                f"{self.value_calculator.min_ev:.1%}. Monitor picks closely."
                f"{_mkt_breakdown}"
            )
            if not _cold_streak_alerted_today():
                logger.warning(_cold_msg)
                try:
                    await self.telegram.send_alert(_cold_msg)
                    _mark_cold_streak_alerted()
                except Exception:
                    pass
            else:
                logger.info(
                    f"Cold streak already alerted today (ROI={_cal_roi:+.1%}, "
                    f"EV threshold={self.value_calculator.min_ev:.1%}) — skipping duplicate Telegram"
                )

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
            return [], [], []

        # Bulk-preload all DB data for the fixture loop in a single batch.
        # preload_batch() catches its own exceptions and sets _preload_cache = None
        # on failure, so the per-fixture fallback path activates automatically.
        self.feature_engineer.preload_batch(fixture_ids)

        # Warn operators when ML is excluded from today's ensemble (Story 7.3)
        await self._check_ml_zero_weight()

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
        self.feature_engineer._preload_cache = None  # free memory after analysis

        all_recommendations = []
        _skipped_coverage = 0
        for mid, result in zip(fixture_ids, analyses):
            if isinstance(result, Exception):
                logger.error(f"Error analyzing match {mid}: {result}")
                continue
            if not result.predictions:
                _skipped_coverage += 1  # analyze_fixture bailed on zero coverage
            all_recommendations.extend(result.recommendations)
        if _skipped_coverage:
            logger.info(
                f"Coverage summary: {_skipped_coverage}/{len(fixture_ids)} fixtures "
                f"skipped for missing historical data (no pick possible) — expected "
                f"for qualifying-round minnows; --backfill-history can close gaps"
            )

        # Sort order: EV × confidence × agreement bonus × contrarian bonus.
        # Contrarian picks (model significantly disagrees with market) get
        # a boost when backed by strong model agreement — these are genuine
        # edges the market is mispricing, not model errors.
        _agreement_bonus = {"unanimous": 1.15, "solo": 1.05, "majority": 1.0, "split": 0.85, "unknown": 0.95}

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

        # Matches the AI briefing already ruled on today are FROZEN: a later
        # pipeline run must not resurrect a vetoed bet (it once came back under
        # a different selection, dodging the (match, selection) dedup) or add
        # picks behind the final decision. Markers are written by the briefing
        # service ("final:<match_id>" in data/briefings_sent.json, cached
        # across CI runs).
        try:
            import json as _json
            _bs_path = Path("data/briefings_sent.json")
            if _bs_path.exists():
                _today_keys = _json.loads(_bs_path.read_text()).get(
                    date.today().isoformat(), []
                )
                _final_ids = {
                    int(k.split(":", 1)[1])
                    for k in _today_keys
                    if k.startswith("final:")
                }
                if _final_ids:
                    _before = len(all_recommendations)
                    all_recommendations = [
                        r for r in all_recommendations if r.match_id not in _final_ids
                    ]
                    if len(all_recommendations) < _before:
                        logger.info(
                            f"Skipped {_before - len(all_recommendations)} pick(s) on "
                            f"{len(_final_ids)} match(es) already finalized by the AI "
                            f"briefing today (decisions are binding for the day)"
                        )
        except Exception as _fe:
            logger.debug(f"Briefing-final filter skipped: {_fe}")

        # Daily exposure limit: cap total Kelly stake across all picks to
        # prevent over-betting even when many value picks are found.
        max_daily_exposure = self.config.get("betting.max_total_kelly_pct", 40.0)
        dropped_picks_cap: List[BetRecommendation] = []
        if max_daily_exposure > 0 and all_recommendations:
            # Already sorted by EV×conf×agreement — trim from the bottom
            total_exposure = sum(r.kelly_stake_percentage for r in all_recommendations)
            if total_exposure > max_daily_exposure:
                all_recommendations, dropped_picks_cap = self._apply_exposure_cap(
                    all_recommendations, max_daily_exposure
                )
                logger.info(
                    f"Daily exposure cap: {total_exposure:.1f}% → "
                    f"{sum(r.kelly_stake_percentage for r in all_recommendations):.1f}% "
                    f"(dropped {len(dropped_picks_cap)} lowest-ranked picks, "
                    f"cap={max_daily_exposure:.0f}%)"
                )

        logger.info(
            f"Found {len(all_recommendations)} high-confidence picks for {target} "
            f"(EV threshold: {self.value_calculator.min_ev:.1%})"
        )

        # Save picks; get back only the ones new this run (for Telegram deduplication)
        new_picks = self._save_picks(all_recommendations, target)

        return all_recommendations, new_picks, dropped_picks_cap

    def _apply_exposure_cap(
        self, recommendations: List[BetRecommendation], max_pct: float
    ) -> tuple:
        """Trim recommendations to the daily Kelly exposure cap.

        Returns (capped, dropped) where capped sums to ≤ max_pct.
        Each dropped pick is logged at INFO with match, market, odds, and Kelly stake.
        The first pick is never dropped (scaled down to max_pct if it alone exceeds cap).
        """
        capped: List[BetRecommendation] = []
        running = 0.0
        for rec in recommendations:
            if running + rec.kelly_stake_percentage > max_pct:
                if not capped:
                    rec.kelly_stake_percentage = round(max_pct, 2)
                    capped.append(rec)
                break
            running += rec.kelly_stake_percentage
            capped.append(rec)
        dropped = recommendations[len(capped):]
        for dp in dropped:
            logger.info(
                f"Dropped by exposure cap: {dp.match} "
                f"— {dp.market} @ {dp.odds:.2f} "
                f"(Kelly {dp.kelly_stake_percentage:.2f}%)"
            )
        return capped, dropped

    def _save_picks(self, picks: List[BetRecommendation], pick_date: date) -> List[BetRecommendation]:
        """Save picks to database for result tracking.

        Returns the list of picks that were *newly* saved this run so the caller
        can send only fresh picks to Telegram and avoid re-notifying for picks
        that were already sent in an earlier run on the same day.
        """
        if not picks:
            return []

        from src.models.poisson_model import NATIONAL_TEAM_LEAGUES
        new_picks: List[BetRecommendation] = []
        with self.db.get_session() as session:
            for pick in picks:
                # WC "pick every match" = EXACTLY ONE pick per match per day.
                # A later run (e.g. the backup after the primary briefing failed
                # on the Pro session limit and the match wasn't frozen) would
                # otherwise regenerate a DIFFERENT forced selection — the
                # (match, selection) dedup below lets it through, producing two
                # tracked picks + a double-pick footer (Norway vs Senegal:
                # Home Win + Over 2.5). Enforce one-per-match for national-team
                # leagues regardless of selection.
                # Same guard for coverage-gated club forced picks: a later run
                # would otherwise regenerate a different forced selection (odds
                # move) and slip past the (match, selection) dedup below.
                if (pick.league in NATIONAL_TEAM_LEAGUES
                        or getattr(pick, "is_forced", False)):
                    dup = session.query(SavedPick).filter(
                        SavedPick.match_id == pick.match_id,
                        SavedPick.pick_date == pick_date,
                    ).first()
                    if dup:
                        continue

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
                    model_agreement=getattr(pick, "model_agreement", None),
                )
                session.add(saved)
                new_picks.append(pick)

                # AC4: warn when a new pick has no injury data for its match
                match_row = session.query(Match).filter(Match.id == pick.match_id).first()
                if match_row:
                    injury_count = session.query(Injury).filter(
                        Injury.team_id.in_([match_row.home_team_id, match_row.away_team_id])
                    ).count()
                    if injury_count == 0:
                        logger.warning(
                            f"Pick saved with no injury data for fixture {pick.match_id} "
                            f"({pick.match}) — injury features will be zero"
                        )

            session.commit()
            logger.info(f"Saved {len(new_picks)} new picks to database (skipped {len(picks) - len(new_picks)} duplicates)")
        return new_picks

    async def scrape_results(self, budget_seconds: int = 3300):
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

        # Leagues that consistently render 100-400+ rows on the first page and
        # take 3-7 min per run in CI.  Deprioritized to end of queue (when no
        # pending picks) AND given a higher minimum-budget gate so they are
        # skipped rather than started when remaining budget is too low to finish.
        # Austria/Denmark added after run #93 showed them at 367s/408s.
        _SLOW_LEAGUES = {
            "europe/champions-league",
            "europe/europa-league",
            "europe/europa-conference-league",
            "austria/bundesliga",
            "denmark/superliga",
            # Nordic summer leagues return the full ~90-match season on the
            # results page; re-upserting all of them each run took ~18 min per
            # league on high-latency Postgres. Cap to the most-recent 25 — the
            # season backlog is already stored and new results always land in
            # that slice. (Observed 2026-07-06.)
            "sweden/allsvenskan",
            "norway/eliteserien",
        }
        # Skip leagues excluded from all Flashscore scraping
        _fs_skip_results = set(self.config.get("scraping.flashscore_skip_leagues", []))
        # Also skip leagues excluded specifically from results scraping (e.g. ECL timeout)
        _fs_skip_results_only = set(self.config.get("scraping.flashscore_skip_results_leagues", []))
        _fs_skip_results |= _fs_skip_results_only
        if _fs_skip_results:
            leagues = [l for l in leagues if l not in _fs_skip_results]
            logger.info(f"scrape_results: skipping {sorted(_fs_skip_results)} (flashscore_skip_leagues + flashscore_skip_results_leagues)")

        _priority = [l for l in leagues if l in _pending_leagues]
        _rest_normal = [l for l in leagues if l not in _pending_leagues and l not in _SLOW_LEAGUES]
        _rest_slow   = [l for l in leagues if l not in _pending_leagues and l in _SLOW_LEAGUES]
        _ordered = _priority + _rest_normal + _rest_slow

        if _priority:
            logger.info(f"scrape_results priority order: {len(_priority)} pending-pick leagues first, "
                        f"{len(_rest_slow)} slow tournaments last")

        # Minimum budget required before even attempting a league.
        # Slow leagues (400-500s typical) need more runway or they'll breach
        # the CI step wall.  Normal leagues need ~150s for Chrome spin-up.
        # If remaining budget is below the gate, skip rather than start.
        _MIN_BUDGET = {
            "slow": 550,   # slow leagues: 500s typical + 50s safety margin
            "normal": 150, # normal leagues: ~80s typical + 70s safety margin
        }

        scraped, skipped = 0, 0
        _pending_failed: set = set()  # pending-pick leagues that failed/timed out
        deadline = _timer.monotonic() + budget_seconds
        loop = asyncio.get_event_loop()
        try:
            for league in _ordered:
                remaining_s = deadline - _timer.monotonic()
                min_needed = _MIN_BUDGET["slow"] if league in _SLOW_LEAGUES else _MIN_BUDGET["normal"]
                if remaining_s < min_needed:
                    logger.warning(
                        f"Skipping {league}: only {remaining_s:.0f}s budget left "
                        f"(needs ≥{min_needed}s) — "
                        f"{len(_ordered) - scraped - skipped} leagues not reached"
                    )
                    # Any pending leagues we didn't even start
                    for remaining_league in _ordered[_ordered.index(league):]:
                        if remaining_league in _pending_leagues:
                            _pending_failed.add(remaining_league)
                    break
                if league in _recently_scraped:
                    skipped += 1
                    continue
                # asyncio.wait_for is a best-effort guard; with run_in_executor
                # the Chrome thread may outlive the asyncio timeout.
                # The real protection is the budget gate above.
                _per_league_cap = 450 if league in _SLOW_LEAGUES else 280
                # Slow tournaments (UCL/UEL/ECL/Austria/Denmark) can return 100+
                # results.  The synchronous DB upsert loop has no await points so
                # asyncio.wait_for cannot interrupt it mid-loop.  Limiting to the
                # 25 most-recent matches keeps wall time under ~180s even on high-
                # latency Neon days (25 × ~7s = ~175s).
                _max_results = 25 if league in _SLOW_LEAGUES else None
                _league_start = _timer.monotonic()
                try:
                    await asyncio.wait_for(
                        self.scraper.scrape_league_results(
                            league, skip_stats=True, max_results=_max_results
                        ),
                        timeout=_per_league_cap,
                    )
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
                        f"Flashscore results timeout for {league} after {_elapsed:.0f}s "
                        f"(cap={_per_league_cap}s), continuing"
                    )
                    if league in _pending_leagues:
                        _pending_failed.add(league)
                    # Run close_driver in an executor so it can't block the event loop
                    # (driver.quit() and camoufox teardown can themselves take 30-60s).
                    try:
                        await asyncio.wait_for(
                            loop.run_in_executor(None, self.scraper.close_driver),
                            timeout=30,
                        )
                    except Exception:
                        pass
                except Exception as e:
                    logger.debug(f"Flashscore results error for {league}: {e}")
                    if league in _pending_leagues:
                        _pending_failed.add(league)
        finally:
            try:
                await asyncio.wait_for(
                    loop.run_in_executor(None, self.scraper.close_driver),
                    timeout=30,
                )
            except Exception:
                pass

        logger.info(f"Flashscore results scrape complete: {scraped} scraped, {skipped} skipped (already done)")

        # Alert when leagues with unsettled picks could not be scraped — picks
        # in those leagues will remain stuck in 'pending' until the next run.
        if _pending_failed:
            _alert = (
                f"⚠️ Flashscore results: failed to scrape {len(_pending_failed)} league(s) "
                f"with pending picks — settlement may be delayed.\n"
                f"Affected: {', '.join(sorted(_pending_failed))}"
            )
            logger.warning(_alert)
            try:
                await self.telegram.send_alert(_alert)
            except Exception as _tg_err:
                logger.warning(f"Could not send results-failure Telegram alert: {_tg_err}")

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

        # 2a. For stale pending picks (match date >24h ago, still no result in DB),
        # try a one-off API-Football fetch for the missing date(s).  This recovers
        # results for leagues where Flashscore scraping consistently fails
        # (e.g. portugal/primeira-liga).
        try:
            stale_dates: set = set()
            with self.db.get_session() as _sess:
                for _pick in _sess.query(SavedPick).filter(SavedPick.result.is_(None)):
                    _match = _sess.get(Match, _pick.match_id)
                    if _match and _match.home_goals is None and _match.match_date:
                        _match_naive = (
                            _match.match_date.replace(tzinfo=None)
                            if getattr(_match.match_date, "tzinfo", None)
                            else _match.match_date
                        )
                        if _match_naive < utcnow() - timedelta(hours=24):
                            _d = _match.match_date.date() if hasattr(_match.match_date, "date") else _match.match_date
                            stale_dates.add(_d)
            if stale_dates:
                # Quota guard: this fetch shares the 100/day API-Football budget
                # with --update's odds fetch. Don't drain it for old results —
                # skip when the remaining budget is low, and fetch at most the 2
                # most-recent stale dates (older results rarely arrive late and
                # aren't worth starving today's odds).
                _remaining = self.apifootball.remaining_budget()
                _MIN_BUDGET = 25
                if _remaining < _MIN_BUDGET:
                    logger.warning(
                        f"Settle: {len(stale_dates)} stale date(s) but only "
                        f"{_remaining} API requests left (<{_MIN_BUDGET}) — "
                        f"skipping stale-result fetch to preserve odds budget"
                    )
                else:
                    _to_fetch = sorted(stale_dates, reverse=True)[:2]
                    logger.info(
                        f"Settle: {len(stale_dates)} stale date(s) with no result — "
                        f"fetching {len(_to_fetch)} most-recent from API-Football: "
                        f"{_to_fetch} (budget {_remaining})"
                    )
                    for _d in _to_fetch:
                        await self.apifootball._fetch_fixtures_by_date(_d)
        except Exception as _exc:
            logger.warning(f"Settle: API-Football stale-result pre-fetch failed: {_exc}")

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
                # Exception: if the match is >3h past its scheduled kickoff, is_fixture=True
                # just means the result was never fetched — let those fall through.
                if match and match.is_fixture:
                    _mdt = match.match_date
                    if _mdt is None:
                        continue
                    _mdt_naive = _mdt.replace(tzinfo=None) if getattr(_mdt, "tzinfo", None) else _mdt
                    if _mdt_naive > utcnow() - timedelta(hours=3):
                        continue  # Genuinely upcoming or just kicked off

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
                            from src.utils.team_names import team_names_similar
                            h_name, a_name = parts[0].strip(), parts[1].strip()
                            for cand in candidates:
                                ch = session.get(Team, cand.home_team_id)
                                ca = session.get(Team, cand.away_team_id)
                                if ch and ca and team_names_similar(ch.name, h_name) and team_names_similar(ca.name, a_name):
                                    match = cand
                                    break
                if not match or match.home_goals is None or match.away_goals is None:
                    continue  # Match not completed yet

                # Settle on the 90-MINUTE (regulation) score. Bookmakers settle
                # 1X2 / Over-Under / BTTS / Team-Goals on 90 minutes only — extra
                # time and penalties (WC knockouts from the Round of 32 on) do NOT
                # count. match.home_goals/away_goals include extra-time goals for
                # AET fixtures; regulation_home_goals/away_goals hold the 90' score.
                # Fall back to the full score for group games (no ET, regulation
                # not separately stored).
                if (match.regulation_home_goals is not None
                        and match.regulation_away_goals is not None):
                    hg = match.regulation_home_goals
                    ag = match.regulation_away_goals
                    if hg != match.home_goals or ag != match.away_goals:
                        logger.info(
                            f"Settling {pick.match_name} on 90-min score {hg}-{ag} "
                            f"(final incl. extra time was {match.home_goals}-{match.away_goals})"
                        )
                else:
                    hg = match.home_goals
                    ag = match.away_goals
                total = hg + ag
                btts = hg > 0 and ag > 0

                # Determine actual outcome for the selection.
                # Unknown selections must NOT silently default to "loss" — that
                # masked a real bug in the past where 7 picks were wrongly settled.
                sel = pick.selection
                resolved = True
                won = False

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
                else:
                    resolved = False

                if not resolved:
                    logger.error(
                        f"Cannot settle pick id={pick.id}: unknown selection "
                        f"{sel!r} for {pick.match_name}. Pick left pending — "
                        f"add a settlement branch for this market or correct the "
                        f"selection name."
                    )
                    continue  # leave result = None so we don't poison stats

                pick.result = "win" if won else "loss"
                # Store the FINAL score (incl. ET) in actual_* — this is what the
                # score-mismatch re-settlement check compares against match.home_goals;
                # storing the regulation score here would make every AET match look
                # mismatched and re-settle forever. `won` was decided on regulation.
                pick.actual_home_goals = match.home_goals
                pick.actual_away_goals = match.away_goals
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

            # Stuck-pick sweep: a pick whose match is >10 days old and still has
            # no result will never settle (result was never found) and silently
            # inflates the pending count forever. Void it (result='void' is
            # excluded from win/loss stats) so pending reflects only live picks.
            _STUCK_DAYS = 10
            _stuck_cutoff = date.today() - timedelta(days=_STUCK_DAYS)
            _voided = 0
            for _p in session.query(SavedPick).filter(SavedPick.result.is_(None)).all():
                _pd = _p.pick_date
                if isinstance(_pd, str):
                    try:
                        _pd = date.fromisoformat(_pd)
                    except Exception:
                        _pd = None
                if _pd and _pd < _stuck_cutoff:
                    _p.result = "void"
                    _p.settled_at = utcnow()
                    _voided += 1
                    logger.warning(
                        f"Stuck-pick void: {_p.match_name} | {_p.selection} "
                        f"(pick_date {_p.pick_date}, never settled in {_STUCK_DAYS}d)"
                    )
            if _voided:
                logger.warning(f"Voided {_voided} stuck pick(s) older than {_STUCK_DAYS} days")

            session.commit()

        for s in settled:
            result_icon = "✅" if s["result"] == "win" else "❌"
            logger.info(
                f"Settled {result_icon} {s['match_name']} | {s['selection']} "
                f"@ {s['odds']:.2f} → {s['result'].upper()} [{s['score']}] "
                f"(stake {s['stake']:.1f}%, {s['league']})"
            )
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

            # Model coverage summary (encapsulated — no private-attr access)
            _cov = self.predictor.coverage_summary()
            poisson_teams = _cov["poisson_teams"]
            elo_teams = _cov["elo_teams"]
            ml_fitted = _cov["ml_fitted"]

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

    def rolling_backtest(self, window_days: int = 30) -> None:
        """Print per-window (monthly by default) performance over all settled picks.

        Groups settled picks into rolling windows and reports win rate, ROI,
        and average EV per window so drift or improvement over time is visible.
        """
        with self.db.get_session() as session:
            all_picks = session.query(SavedPick).filter(
                SavedPick.result.isnot(None)
            ).order_by(SavedPick.pick_date).all()
            if not all_picks:
                print("No settled picks found.")
                return
            rows = [
                {
                    "date": p.pick_date,
                    "result": p.result,
                    "odds": p.odds or 0.0,
                    "kelly": p.kelly_stake_percentage or 0.0,
                    "ev": p.expected_value or 0.0,
                    "market": p.market or "",
                }
                for p in all_picks
            ]

        # Group into calendar-month windows
        from collections import defaultdict
        by_month: dict = defaultdict(list)
        for row in rows:
            d = row["date"]
            key = f"{d.year}-{d.month:02d}" if d else "unknown"
            by_month[key].append(row)

        header = f"{'Month':<10} {'Picks':>6} {'W':>5} {'L':>5} {'Win%':>7} {'ROI%':>8} {'AvgOdds':>9} {'AvgEV%':>8}"
        print("\nRolling backtest — per-month performance")
        print("=" * len(header))
        print(header)
        print("-" * len(header))

        cumulative_staked = 0.0
        cumulative_profit = 0.0
        for month in sorted(by_month.keys()):
            picks = by_month[month]
            wins = sum(1 for p in picks if p["result"] == "win")
            losses = sum(1 for p in picks if p["result"] == "loss")
            total = wins + losses
            win_rate = wins / total if total else 0.0
            staked = sum(p["kelly"] for p in picks)
            profit = sum(
                p["kelly"] * (p["odds"] - 1) if p["result"] == "win" else -p["kelly"]
                for p in picks
            )
            roi = profit / staked if staked else 0.0
            avg_odds = sum(p["odds"] for p in picks) / len(picks) if picks else 0.0
            avg_ev = sum(p["ev"] for p in picks) / len(picks) if picks else 0.0
            cumulative_staked += staked
            cumulative_profit += profit
            print(
                f"{month:<10} {total:>6} {wins:>5} {losses:>5} "
                f"{win_rate:>7.1%} {roi:>8.1%} {avg_odds:>9.2f} {avg_ev:>7.1%}"
            )

        print("-" * len(header))
        cum_roi = cumulative_profit / cumulative_staked if cumulative_staked else 0.0
        total_picks = len(rows)
        total_wins = sum(1 for r in rows if r["result"] == "win")
        print(
            f"{'ALL TIME':<10} {total_picks:>6} {total_wins:>5} {total_picks-total_wins:>5} "
            f"{total_wins/total_picks if total_picks else 0:>7.1%} {cum_roi:>8.1%}"
        )

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

        if len(settled) < 10:
            logger.info(f"Not enough settled picks for tuning ({len(settled)}, need 10+)")
            return

        logger.info(f"Tuning on {len(settled)} settled picks")

        # ── Leak-resistant refit ─────────────────────────────────────────────
        # Refit Poisson/Elo using ONLY data older than the oldest settled pick,
        # so the accuracy we measure below isn't inflated by Poisson strengths
        # that already saw those outcomes. This is a one-time cost; after
        # tuning we restore the full-data fit so live prediction sees the
        # latest information.
        oldest_pick_date = min(
            (p["match_date"] for p in settled if p["match_date"]),
            default=None,
        )
        if oldest_pick_date is not None:
            try:
                logger.info(
                    f"Refitting Poisson/Elo on data before {oldest_pick_date} "
                    f"to avoid leakage during tuning"
                )
                self.predictor.fit(as_of_date=oldest_pick_date)
            except Exception as e:
                logger.warning(f"Leak-resistant refit failed, using current fit: {e}")

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

        # Cap total outcomes to the N most recent picks so the full loop stays
        # well inside the 10-minute CI timeout.  Poisson/Elo predictions are
        # in-memory but iterating 70+ outcomes + the leak-resistant refit was
        # already consuming most of the budget.  _ML_EVAL_CAP picks within that
        # window get real ML feature evaluation.
        _TUNE_PICK_CAP = 40
        _ML_EVAL_CAP = 15
        if len(outcomes) > _TUNE_PICK_CAP:
            outcomes = outcomes[-_TUNE_PICK_CAP:]
            logger.info(f"  Tuning: capped to most recent {_TUNE_PICK_CAP} 1X2 outcomes")

        # Preload team history for the ML-evaluated picks so create_features()
        # uses the cache instead of per-pick DB queries.
        ml_eval_start = max(0, len(outcomes) - _ML_EVAL_CAP) if self.predictor.ml_models.is_fitted else len(outcomes)
        if self.predictor.ml_models.is_fitted and ml_eval_start < len(outcomes):
            _ml_match_ids = [outcomes[i][0]["match_id"] for i in range(ml_eval_start, len(outcomes))]
            try:
                self.feature_engineer.preload_batch(
                    _ml_match_ids, cap_per_team=200, cutoff_days=0
                )
                logger.debug(f"  Tuning preload: {len(self.feature_engineer._preload_cache.get('team_history', {}))} teams cached for ML eval")
            except Exception as _pre_err:
                logger.debug(f"  Tuning preload failed (falling back to per-pick queries): {_pre_err}")

        # Single pass: accuracy + calibration data + Bayesian updates
        model_correct = {"poisson": 0, "elo": 0, "ml": 0}
        model_total = {"poisson": 0, "elo": 0, "ml": 0}
        model_predictions: dict = {"poisson": [], "elo": [], "ml": []}
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

            # --- ML accuracy on a capped subset (fitted models only) ---
            ml_pred = None
            if i >= ml_eval_start:
                try:
                    feats = await self.feature_engineer.create_features(
                        pick["match_id"],
                        as_of_date=pick["match_date"],
                        for_training=True,
                    )
                    if feats:
                        vec = self.feature_engineer.create_feature_vector(feats)
                        names = self.feature_engineer.get_feature_names(feats)
                        ml_predictions = self.predictor.ml_models.predict(
                            vec, feature_names=names
                        )
                        ml_pred = ml_predictions.get("ml_average")
                except Exception as _ml_err:
                    logger.debug(f"ML eval failed for pick: {_ml_err}")

            if ml_pred:
                ml_best = max(keys_1x2, key=lambda k: ml_pred.get(k, 0))
                model_total["ml"] += 1
                if ml_best == actual:
                    model_correct["ml"] += 1
                model_predictions["ml"].append(
                    (ml_pred.get(ml_best, 0.33), 1 if ml_best == actual else 0)
                )
                if league:
                    _days_ago = (
                        (date.today() - pick["match_date"]).days
                        if pick["match_date"] else 0
                    )
                    bayesian.update(league, "ml", ml_best == actual, _days_ago, market="1X2")

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

        # Release tuning preload cache — it's only needed for this loop
        self.feature_engineer._preload_cache = None

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

        # ML weight: when ML models are fitted, the loop above evaluated a
        # capped sample (last 80 picks) and accuracies["ml"] holds a real
        # accuracy. If for some reason no samples ran (e.g. all feature builds
        # failed) we fall back to a proxy derived from Poisson/Elo accuracy
        # so ML still contributes a non-zero weight.
        ml_is_proxy = False
        if self.predictor.ml_models.is_fitted and accuracies["ml"] == 0.0:
            avg_pe = (accuracies["poisson"] + accuracies["elo"]) / 2
            accuracies["ml"] = avg_pe * 0.6  # slight discount vs proven models
            base_models["ml"] = accuracies["ml"]
            total_acc = sum(base_models.values())
            ml_is_proxy = True  # proxy value — gate must not apply threshold check

        # Diagnostic snapshot of model accuracies — Bayesian per-league weights
        # are the sole runtime weights, so this dict is for logging/CLI display only.
        new_weights = {
            "poisson": round(base_models["poisson"] / total_acc, 3),
            "elo": round(base_models["elo"] / total_acc, 3),
            "xgboost": round((base_models["ml"] / total_acc) * 0.6, 3),
            "random_forest": round((base_models["ml"] / total_acc) * 0.4, 3),
        }
        logger.info(f"Aggregate model accuracies (1X2): {accuracies}")
        logger.info(f"Diagnostic weight ratios (not used at runtime): {new_weights}")

        # Model Confidence Calibration from data collected above.
        # Floors at 0.85 (max 15% weight reduction) and requires ≥5 samples/bin
        # to avoid noisy estimates from sparse data.  New factors are blended
        # 70/30 with the previous run's factors (EMA) to prevent jumpy adjustments
        # when a single bad streak skews a bin.
        try:
            cal_path = Path("data/models/calibration.json")
            # Load previous factors for EMA smoothing
            prev_cal: dict = {}
            try:
                if cal_path.exists():
                    prev_cal = json.loads(cal_path.read_text())
            except Exception:
                pass

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
                    if len(in_bin) < 5:  # require ≥5 samples for a reliable bin estimate
                        continue
                    mean_pred = sum(p for p, _ in in_bin) / len(in_bin)
                    mean_hit = sum(h for _, h in in_bin) / len(in_bin)
                    total_error += abs(mean_pred - mean_hit)
                    n_bins_used += 1
                if n_bins_used > 0:
                    avg_cal_error = total_error / n_bins_used
                    # Floor at 0.85 — cap max weight reduction at 15% so sparse
                    # data can't drive models to near-zero influence.
                    raw_factor = round(max(0.85, 1.0 - avg_cal_error), 3)
                    # EMA: blend 70% new estimate with 30% previous to dampen
                    # run-to-run volatility from small sample noise.
                    prev = prev_cal.get(model, raw_factor)
                    cal_factors[model] = round(0.7 * raw_factor + 0.3 * prev, 3)
                else:
                    cal_factors[model] = 1.0

            self._apply_ml_calibration_gate(accuracies, cal_factors, prev_cal, ml_is_proxy=ml_is_proxy)

            # Poisson weight floor: when Poisson clearly outperforms the ensemble
            # (>5pp gap), protect it from calibration discount.  Without this floor,
            # a short-run calibration error can inadvertently suppress the best
            # individual model while ML (or other noisy models) dominate.
            poisson_acc = accuracies.get("poisson", 0.0)
            ensemble_acc = accuracies.get("ensemble", 0.0)
            if poisson_acc > ensemble_acc + 0.05 and cal_factors.get("poisson", 1.0) < 1.0:
                cal_factors["poisson"] = 1.0
                logger.info(
                    f"Poisson floor applied: Poisson {poisson_acc:.1%} > "
                    f"ensemble {ensemble_acc:.1%} (+5pp) — calibration restored to 1.0"
                )

            # Elo weight floor: same logic as Poisson — protect Elo when it clearly
            # outperforms the blended ensemble.
            elo_acc = accuracies.get("elo", 0.0)
            if elo_acc > ensemble_acc + 0.05 and cal_factors.get("elo", 1.0) < 1.0:
                cal_factors["elo"] = 1.0
                logger.info(
                    f"Elo floor applied: Elo {elo_acc:.1%} > "
                    f"ensemble {ensemble_acc:.1%} (+5pp) — calibration restored to 1.0"
                )

            # Alert operators via Telegram when ML is excluded (AC1 of Story 4.2)
            if cal_factors.get("ml", 1.0) == 0.0:
                ml_acc = accuracies.get("ml", 0.0)
                try:
                    await self.telegram.send_alert(
                        f"⚠️ ML training produced a below-threshold model "
                        f"(accuracy {ml_acc:.1%}). "
                        f"ML excluded from ensemble — check training data quality."
                    )
                except Exception as _tg_err:
                    logger.warning(f"Failed to send ML training failure Telegram alert: {_tg_err}")

            cal_path.parent.mkdir(parents=True, exist_ok=True)
            cal_path.write_text(json.dumps(cal_factors, indent=2))
            self.predictor.calibration_factors.update(cal_factors)
            logger.info(f"Calibration factors: {cal_factors}")

            # Persist model accuracies so get_daily_picks() can apply Poisson/Elo
            # floors at load time even when tune_ensemble_weights() didn't run this CI cycle.
            _acc_path = Path("data/models/model_accuracies.json")
            try:
                _acc_path.write_text(json.dumps(accuracies, indent=2))
            except Exception as _ae:
                logger.debug(f"Could not persist model accuracies: {_ae}")

            # Reset ML zero-count when ML accuracy recovers (Story 7.3 AC3)
            if cal_factors.get("ml", 0.0) > 0.0:
                self._reset_ml_zero_count()
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

        # Restore the full-data fit so live prediction sees the latest information.
        # (We refit on as_of_date=oldest_pick_date earlier to avoid look-ahead leakage.)
        if oldest_pick_date is not None:
            try:
                self.predictor.fit()
                self.feature_engineer.elo_ratings = self.predictor.elo.ratings
                logger.debug("Restored full-data Poisson/Elo fit after tuning")
            except Exception as e:
                logger.warning(f"Failed to restore full-data fit after tuning: {e}")

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

        # Bulk-preload all team history for all training matches in 3 queries.
        # With no date cutoff (cutoff_days=0) and a larger cap (200/team), the
        # cache covers as_of_date filtering in Python — eliminates per-sample
        # form/H2H/xG/referee DB round-trips.  On Neon (50ms RTT) this turns
        # ~10-13 queries × 500 samples = ~25,000ms into 3 queries + Python loops.
        _all_train_ids = [md["id"] for md in match_data]
        try:
            self.feature_engineer.preload_batch(
                _all_train_ids, cap_per_team=200, cutoff_days=0
            )
            logger.info(
                f"Training preload complete: "
                f"{len(self.feature_engineer._preload_cache.get('team_history', {}))} teams cached"
            )
        except Exception as _pre_err:
            logger.warning(f"Training preload failed (falling back to per-sample queries): {_pre_err}")

        # Build feature matrix and labels — process in parallel batches
        # Hard time budget prevents ML training from blowing the CI timeout.
        import time as _timer
        _ML_TRAIN_BUDGET_S = 2700  # 45 minutes (raised from 30 to allow full 500-sample runs)
        _ml_start = _timer.monotonic()
        _ml_deadline = _ml_start + _ML_TRAIN_BUDGET_S

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

        _budget_exhausted = False
        for batch_start in range(0, len(match_data), BATCH_SIZE):
            if _timer.monotonic() > _ml_deadline:
                _budget_exhausted = True
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

        # Throughput summary (AC2 of Story 4.3)
        _elapsed_min = (_timer.monotonic() - _ml_start) / 60
        _rate = len(X_list) / _elapsed_min if _elapsed_min > 0 else 0
        logger.info(
            f"Training throughput: {len(X_list)} samples in {_elapsed_min:.1f}min "
            f"({_rate:.1f} samples/min)"
        )
        if _budget_exhausted:
            _target = len(match_data)
            _pct = len(X_list) / _target * 100 if _target > 0 else 0
            _alert_msg = (
                f"⚠️ ML training budget exhausted at {len(X_list)}/{_target} samples "
                f"({_pct:.0f}%). Throughput: {_rate:.1f} samples/min."
            )
            logger.warning(_alert_msg)
            try:
                await self.telegram.send_alert(_alert_msg)
            except Exception as _tg_err:
                logger.warning(f"Failed to send training budget alert: {_tg_err}")

        if len(X_list) < 50:
            logger.warning(f"Only {len(X_list)} valid samples (skipped {skipped}), need 50+")
            return

        X = np.array(X_list)
        y = np.array(y_list)
        y_goals = np.array(y_goals_list)

        # Log 1X2 class distribution — helps diagnose if ML predicts only majority class
        _label_names = {0: "Away", 1: "Draw", 2: "Home"}
        _class_counts = {_label_names[c]: int(np.sum(y == c)) for c in [0, 1, 2]}
        _class_pcts = {k: f"{v/len(y):.1%}" for k, v in _class_counts.items()}
        logger.info(
            f"1X2 class distribution ({len(y)} samples): "
            + ", ".join(f"{k}={v} ({_class_pcts[k]})" for k, v in _class_counts.items())
        )

        logger.info(f"Training on {len(X)} samples, {X.shape[1]} features (skipped {skipped})")

        # Train 1X2 models
        self.predictor.ml_models.fit(X, y, feature_names)
        self.predictor.ml_models.save()

        # Per-class precision/recall from cross-validated predictions (leak-free diagnostic)
        try:
            from sklearn.model_selection import cross_val_predict
            from sklearn.metrics import classification_report
            _best_model = self.predictor.ml_models._models.get(
                "xgboost") or next(iter(self.predictor.ml_models._models.values()))
            _cv_pred = cross_val_predict(_best_model, X, y, cv=3)
            _report = classification_report(
                y, _cv_pred,
                target_names=["Away", "Draw", "Home"],
                output_dict=True,
                zero_division=0,
            )
            for cls in ["Away", "Draw", "Home"]:
                r = _report.get(cls, {})
                logger.info(
                    f"  1X2 CV [{cls}]: precision={r.get('precision', 0):.2f} "
                    f"recall={r.get('recall', 0):.2f} f1={r.get('f1-score', 0):.2f} "
                    f"support={int(r.get('support', 0))}"
                )
            logger.info(f"  1X2 CV accuracy={_report.get('accuracy', 0):.3f}")
        except Exception as _cv_err:
            logger.debug(f"1X2 CV classification report failed (non-fatal): {_cv_err}")

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

        # GoalsML accuracy gate: if the model cannot beat the majority-class baseline
        # (predicting the more common outcome every time) it is actively hurting
        # over/under predictions. Disable its blend weight for this session.
        try:
            _goals_val_acc = getattr(self.predictor.goals_model, "last_val_accuracy", None)
            _goals_baseline = getattr(self.predictor.goals_model, "last_majority_baseline", None)
            if _goals_val_acc is not None and _goals_baseline is not None:
                self.predictor.goals_ml_accuracy = _goals_val_acc
                self.predictor.goals_ml_majority_baseline = _goals_baseline
                if _goals_val_acc < _goals_baseline:
                    self.predictor.goals_ml_disabled = True
                    logger.warning(
                        f"GoalsML blend DISABLED: val accuracy {_goals_val_acc:.1%} < "
                        f"majority baseline {_goals_baseline:.1%} — "
                        f"model adds noise; Poisson+bookmaker used for over/under"
                    )
                else:
                    self.predictor.goals_ml_disabled = False
                    logger.info(
                        f"GoalsML blend active: val accuracy {_goals_val_acc:.1%} vs "
                        f"majority baseline {_goals_baseline:.1%}"
                    )
        except Exception as _gate_err:
            logger.debug(f"GoalsML accuracy gate check failed (non-fatal): {_gate_err}")

        # Free the training preload cache — it holds history for all training teams
        # and is no longer needed now that feature extraction is done.
        self.feature_engineer._preload_cache = None

        logger.info("ML model training complete")

        # If a previous tune_ensemble_weights() zeroed-out ML calibration and wrote
        # ml: 0.0 to disk, reset it now so the --picks step (which runs after --train
        # in CI) uses the freshly trained model instead of silently excluding ML.
        # The next tune call will re-evaluate the model's actual accuracy and re-apply
        # the gate if it genuinely underperforms.
        _cal_path = Path("data/models/calibration.json")
        try:
            if _cal_path.exists():
                _cal = json.loads(_cal_path.read_text())
                _prev_ml_cal = _cal.get("ml", 1.0)
                if _prev_ml_cal < 1.0:
                    if _budget_exhausted:
                        logger.warning(
                            f"ML calibration NOT reset (training budget-exhausted at "
                            f"{len(X_list)}/{len(match_data)} samples) — "
                            f"ml={_prev_ml_cal} calibration remains until a full training run"
                        )
                    else:
                        # Only restore ML to full weight when the freshly-trained model
                        # actually beats the naive majority-class baseline.  If it doesn't,
                        # keep the previous gate discount so --picks doesn't accidentally
                        # inherit a below-baseline model.
                        _val_acc = getattr(self.predictor.ml_models, "last_val_accuracy", None)
                        _majority = getattr(self.predictor.ml_models, "last_majority_baseline", None)
                        _beats_baseline = (
                            _val_acc is not None
                            and _majority is not None
                            and _val_acc > _majority
                        )
                        if _beats_baseline:
                            _cal["ml"] = 1.0
                            _cal_path.write_text(json.dumps(_cal, indent=2))
                            self.predictor.calibration_factors["ml"] = 1.0
                            logger.info(
                                f"ML calibration restored to 1.0 after retraining "
                                f"(val_acc={_val_acc:.1%} > majority_baseline={_majority:.1%})"
                            )
                        elif _val_acc is not None:
                            # Model trained but doesn't beat baseline — apply partial discount
                            # so it contributes a little rather than being silently reset to full
                            _new_ml = 0.6 if _prev_ml_cal == 0.0 else _prev_ml_cal
                            _cal["ml"] = _new_ml
                            _cal_path.write_text(json.dumps(_cal, indent=2))
                            self.predictor.calibration_factors["ml"] = _new_ml
                            logger.warning(
                                f"ML val_acc={_val_acc:.1%} does not beat majority "
                                f"baseline={_majority:.1%} — keeping partial discount ml={_new_ml}"
                            )
                        else:
                            # No val accuracy recorded (e.g. training skipped validation step)
                            # — keep the gate's existing decision rather than guessing
                            logger.info(
                                f"ML calibration unchanged (ml={_prev_ml_cal}) — "
                                f"no validation accuracy available from fresh training run"
                            )
        except Exception as _ce:
            logger.warning(f"Could not update ML calibration after training: {_ce}")

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
                if trained_dt.tzinfo is not None:
                    trained_dt = trained_dt.replace(tzinfo=None)
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
            self._recent_roi = None
            self._auto_calibrate_ev_threshold()
            _settle_roi = getattr(self, "_recent_roi", None)
            if _settle_roi is not None and _settle_roi < -0.15:
                _cold_msg = (
                    f"⚠️ Cold streak alert: ROI={_settle_roi:+.1%} over last "
                    f"{getattr(self, '_recent_roi_n', '?')} settled picks — "
                    f"model is underperforming. EV threshold tightened to "
                    f"{self.value_calculator.min_ev:.1%}. Monitor picks closely."
                )
                logger.warning(_cold_msg)
                if not _cold_streak_alerted_today():
                    try:
                        await self.telegram.send_alert(_cold_msg)
                        _mark_cold_streak_alerted()
                    except Exception:
                        pass
        except Exception as e:
            logger.warning(f"EV calibration failed: {e}")

        # 3.5. Per-market calibration from pick outcomes (goals/BTTS markets).
        # Uses SavedPick.predicted_probability vs actual win/loss to measure how
        # well-calibrated the ensemble is per market — supplements the 1X2-only
        # calibration computed inside tune_ensemble_weights().
        try:
            self.calibrate_from_pick_outcomes()
        except Exception as e:
            logger.warning(f"Pick calibration failed: {e}")

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

    def calibrate_from_pick_outcomes(self) -> dict:
        """Compute per-market calibration factors from settled pick history.

        Compares SavedPick.predicted_probability against actual win/loss outcomes
        for goals and BTTS markets. Produces a per-market calibration factor stored
        in data/models/pick_calibration.json and applied by EnsemblePredictor.predict()
        via center-shrink (factor<1 means model is overconfident → shrink toward 0.5).

        1X2 calibration is handled separately by tune_ensemble_weights() (per-model).
        This method fills the gap for goals/BTTS markets which have no per-model loop.

        Returns:
            Dict of {market_key: factor} written to disk.
        """
        lookback = date.today() - timedelta(days=90)
        with self.db.get_session() as session:
            picks = session.query(SavedPick).filter(
                SavedPick.result.isnot(None),
                SavedPick.result != "void",
                SavedPick.predicted_probability.isnot(None),
                SavedPick.pick_date >= lookback,
            ).all()
            pick_data = [
                {
                    "market": p.market,
                    "selection": p.selection,
                    "prob": p.predicted_probability,
                    "win": 1 if p.result == "win" else 0,
                }
                for p in picks
            ]

        # Convert picks to (P(canonical_direction), actual) pairs.
        # Canonical direction: over_2.5 = P(over 2.5 goals), btts = P(BTTS yes),
        #                      over_1.5 = P(over 1.5 goals).
        # "Under" and "No" picks are inverted so all data points share the same axis.
        market_pairs: dict = {}
        for d in pick_data:
            mkt = d["market"]
            sel = d["selection"]
            prob = d["prob"]
            win = d["win"]

            if mkt == "Over 2.5":
                market_pairs.setdefault("over_2.5", []).append((prob, win))
            elif mkt == "Under 2.5":
                market_pairs.setdefault("over_2.5", []).append((1.0 - prob, 1 - win))
            elif mkt == "Over 1.5":
                market_pairs.setdefault("over_1.5", []).append((prob, win))
            elif mkt == "Under 1.5":
                market_pairs.setdefault("over_1.5", []).append((1.0 - prob, 1 - win))
            elif mkt == "BTTS" and sel == "BTTS Yes":
                market_pairs.setdefault("btts", []).append((prob, win))
            elif mkt == "BTTS" and sel == "BTTS No":
                market_pairs.setdefault("btts", []).append((1.0 - prob, 1 - win))

        # Load previous factors for EMA smoothing (prevents jumpy adjustments)
        cal_path = Path("data/models/pick_calibration.json")
        prev_cal: dict = {}
        try:
            if cal_path.exists():
                prev_cal = json.loads(cal_path.read_text())
        except Exception:
            pass

        cal_factors = dict(prev_cal)
        bins = [(0.40, 0.55), (0.55, 0.65), (0.65, 0.75), (0.75, 0.88), (0.88, 1.01)]

        for mkey, pairs in market_pairs.items():
            if len(pairs) < 20:
                logger.info(
                    f"Pick calibration [{mkey}]: not enough data "
                    f"({len(pairs)} picks, need 20+) — keeping previous"
                )
                continue

            total_error = 0.0
            n_bins = 0
            for lo, hi in bins:
                in_bin = [(p, a) for p, a in pairs if lo <= p < hi]
                if len(in_bin) < 5:
                    continue
                mean_pred = sum(p for p, _ in in_bin) / len(in_bin)
                mean_actual = sum(a for _, a in in_bin) / len(in_bin)
                total_error += abs(mean_pred - mean_actual)
                n_bins += 1

            if n_bins == 0:
                continue

            avg_error = total_error / n_bins
            raw_factor = round(max(0.85, 1.0 - avg_error), 3)
            prev = prev_cal.get(mkey, raw_factor)
            cal_factors[mkey] = round(0.7 * raw_factor + 0.3 * prev, 3)
            logger.info(
                f"Pick calibration [{mkey}]: factor={cal_factors[mkey]:.3f} "
                f"(n={len(pairs)}, avg_cal_error={avg_error:.3f}, bins_used={n_bins})"
            )

        cal_path.parent.mkdir(parents=True, exist_ok=True)
        cal_path.write_text(json.dumps(cal_factors, indent=2))
        self.predictor.pick_calibration = cal_factors
        logger.info(f"Pick calibration saved: {cal_factors}")
        return cal_factors

    async def _check_ml_zero_weight(self) -> None:
        """Send Telegram WARNING when ML cal_factor is 0.0; escalate to CRITICAL at 4+ runs.

        Tracks consecutive zero-weight runs in data/models/ml_zero_count.json.
        Increments the counter at most once per calendar day (same-day re-runs are safe).
        Called from get_daily_picks() after the ensemble predictor is ready.
        """
        if self.predictor.calibration_factors.get("ml", 1.0) != 0.0:
            return

        _ml_zero_path = Path("data/models/ml_zero_count.json")
        _today_str = date.today().isoformat()
        try:
            _zdata = (
                json.loads(_ml_zero_path.read_text())
                if _ml_zero_path.exists()
                else {"count": 0, "last_updated": ""}
            )
            # Increment once per calendar day — same-day re-runs don't double-count
            if _zdata.get("last_updated") != _today_str:
                _zdata["count"] = _zdata.get("count", 0) + 1
                _zdata["last_updated"] = _today_str
                _ml_zero_path.parent.mkdir(parents=True, exist_ok=True)
                _ml_zero_path.write_text(json.dumps(_zdata))
        except Exception as _ze:
            logger.warning(f"Failed to update ml_zero_count.json: {_ze}")
            _zdata = {"count": 1}

        try:
            await self.telegram.send_alert(
                "⚠️ ML model contributing 0% to today's picks — calibration gate active. "
                "Check training pipeline."
            )
        except Exception as _tg_err:
            logger.warning(f"ML zero-weight Telegram alert failed: {_tg_err}")

        if _zdata.get("count", 1) >= 4:
            try:
                await self.telegram.send_alert(
                    f"🚨 ML excluded from ensemble for {_zdata['count']} consecutive runs "
                    f"— investigate training pipeline"
                )
            except Exception as _tg_err:
                logger.warning(f"ML CRITICAL escalation Telegram alert failed: {_tg_err}")

    def _reset_ml_zero_count(self) -> None:
        """Reset ml_zero_count.json to 0 when ML calibration factor recovers. (AC3)"""
        _ml_zero_path = Path("data/models/ml_zero_count.json")
        if _ml_zero_path.exists():
            try:
                _ml_zero_path.write_text(
                    json.dumps({"count": 0, "last_updated": date.today().isoformat()})
                )
                logger.debug("ML zero-weight counter reset (ML accuracy recovered)")
            except Exception as _rze:
                logger.warning(f"Failed to reset ml_zero_count.json: {_rze}")

    def _apply_ml_calibration_gate(
        self, accuracies: dict, cal_factors: dict, prev_cal: dict, ml_is_proxy: bool = False
    ) -> None:
        """Zero-out ML calibration when accuracy is below-random; restore when recovered.

        Modifies cal_factors in-place. Called from tune_ensemble_weights() after
        the Poisson/Elo calibration loop and before saving calibration.json.

        When ml_is_proxy=True, the accuracy value was derived from a proxy formula
        (avg_pe * 0.6 ≈ 0.26) that is always below 0.35 regardless of actual model
        quality. In this case the gate threshold must not apply — ML models are fitted
        and should be included at full calibration weight.
        """
        ml_acc = accuracies.get("ml", 0.0)
        if ml_is_proxy:
            # Proxy-derived accuracy cannot be used for gate decisions.
            # ML models are fitted — include at full calibration weight.
            if prev_cal.get("ml", 1.0) == 0.0:
                logger.info("ML re-entering ensemble (proxy mode — threshold gate skipped for fitted model)")
            cal_factors["ml"] = 1.0
            return
        if ml_acc < 0.30:
            cal_factors["ml"] = 0.0
            logger.warning(
                f"ML accuracy {ml_acc:.1%} below random baseline — ML excluded from ensemble"
            )
        elif ml_acc < 0.35:
            # Partial discount: ML is worse than a random 3-class baseline but not
            # catastrophically so — reduce its ensemble contribution by 40% rather
            # than silencing it completely.  Full exclusion resumes below 30%.
            cal_factors["ml"] = 0.6
            logger.warning(
                f"ML accuracy {ml_acc:.1%} below threshold — applying 40% discount (cal=0.6)"
            )
        else:
            if prev_cal.get("ml", 1.0) == 0.0:
                logger.info(
                    f"ML accuracy {ml_acc:.1%} recovered — ML re-entering ensemble"
                )
            cal_factors["ml"] = 1.0

    def _auto_calibrate_ev_threshold(self):
        """Dynamically adjust min EV threshold based on recent ROI.

        Looks at the last N settled picks and computes per-unit ROI:
            roi = sum((odds - 1) if win else -1) / n

        ROI is market-agnostic — unlike hit rate, it correctly accounts
        for odds level (e.g. 40% win rate at 1.75 = -30% ROI is cold;
        40% at 2.80 = +12% ROI is hot).

        - ROI > +10%: hot → relax min_ev by up to -0.5pp (catch more value)
        - ROI -10% to +10%: normal → keep base min_ev
        - ROI < -10%: cold → TIGHTEN min_ev by up to +2pp (only best bets)

        Bounded: min_ev never goes below base or above 10%.
        The calibrated threshold is persisted to data/models/ev_threshold.json.
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

                # ROI = sum of profits per unit staked, market-agnostic
                # win: profit = odds - 1; loss: profit = -1
                roi_values = [
                    (p.odds - 1) if p.result == "win" else -1.0
                    for p in recent
                    if p.odds and p.odds > 1.0
                ]
                if len(roi_values) < 15:
                    return  # not enough picks with valid odds
                roi = sum(roi_values) / len(roi_values)

                if roi > 0.10:
                    # Hot streak: slight relaxation — model is outperforming, allow
                    # marginally lower EV threshold to capture more value bets.
                    # Scale: +10% ROI → -0pp, +30%+ → -0.5pp
                    bonus = min((roi - 0.10) / 0.20, 1.0) * 0.005
                    new_ev = base_ev - bonus
                elif roi < -0.10:
                    # Cold streak: TIGHTEN — only the highest-EV bets during
                    # underperformance. Lower quality picks will make a slump worse.
                    # Scale: -10% ROI → +0pp, -30%- → +2pp
                    tighten = min((-roi - 0.10) / 0.20, 1.0) * 0.02
                    new_ev = base_ev + tighten
                else:
                    new_ev = base_ev  # normal range, keep base

                # Clamp to sane bounds: floor at base, ceiling at 10%
                ev_floor = base_ev
                new_ev = max(ev_floor, min(0.10, round(new_ev, 4)))

                if abs(new_ev - prev_ev) > 0.001:
                    self.value_calculator.min_ev = new_ev
                    direction = "tightened" if new_ev > base_ev else "loosened"
                    logger.info(
                        f"EV auto-calibration: {direction} min_ev from "
                        f"{base_ev:.1%} → {new_ev:.1%} "
                        f"(ROI={roi:+.1%} over last {len(roi_values)} picks)"
                    )

                # Store recent ROI so callers can issue a Telegram cold-streak alert
                self._recent_roi = roi
                self._recent_roi_n = len(roi_values)

                # Persist to disk so next CI run starts from this threshold
                ev_path.parent.mkdir(parents=True, exist_ok=True)
                ev_path.write_text(json.dumps({
                    "min_ev": new_ev,
                    "roi": round(roi, 4),
                    "n_picks": len(roi_values),
                    "updated_at": utcnow().isoformat(),
                }, indent=2))
        except Exception as e:
            logger.warning(f"EV auto-calibration failed: {e}")

    # Pairs of (selection_A, selection_B) that are positively correlated —
    # if one hits, the other is significantly more likely to hit too.
    # Selection names must match BetRecommendation.selection exactly:
    #   "Home Win", "Draw", "Away Win"
    #   "Over 1.5 Goals", "Over 2.5 Goals", "Over 3.5 Goals"
    #   "Under 1.5 Goals", "Under 2.5 Goals", "Under 3.5 Goals"
    #   "BTTS Yes", "BTTS No"
    #   "Home Over 1.5", "Away Over 1.5"
    _CORRELATED_PAIRS = {
        # Home win correlates with high-scoring outcomes
        ("Home Win", "Over 2.5 Goals"),
        ("Home Win", "Over 1.5 Goals"),
        ("Home Win", "Home Over 1.5"),
        # Away win correlates with high-scoring outcomes
        ("Away Win", "Over 2.5 Goals"),
        ("Away Win", "Over 1.5 Goals"),
        ("Away Win", "Away Over 1.5"),
        # Draw correlates with low-scoring outcomes
        ("Draw", "Under 2.5 Goals"),
        ("Draw", "Under 1.5 Goals"),
        # BTTS Yes correlates with over goals (key fix: was "Yes"/"Over 2.5")
        ("BTTS Yes", "Over 2.5 Goals"),
        ("BTTS Yes", "Over 1.5 Goals"),
        # Under markets are correlated with each other
        ("Under 2.5 Goals", "Under 1.5 Goals"),
        # Over markets correlate with team-level overs
        ("Over 2.5 Goals", "Home Over 1.5"),
        ("Over 2.5 Goals", "Away Over 1.5"),
        ("Over 1.5 Goals", "Home Over 1.5"),
        ("Over 1.5 Goals", "Away Over 1.5"),
        # BTTS No correlates with under goals
        ("BTTS No", "Under 2.5 Goals"),
        ("BTTS No", "Under 1.5 Goals"),
        # Team-level overs imply high-scoring match → correlated with Over 3.5
        ("Home Over 1.5", "Over 3.5 Goals"),
        ("Away Over 1.5", "Over 3.5 Goals"),
    }

    # Composite score used by both the main sort and the correlation filter,
    # so a pick promoted by unanimous agreement isn't dropped in favour of a
    # split-models pick that has marginally higher raw EV*confidence.
    _AGREEMENT_BONUS = {"unanimous": 1.15, "solo": 1.05, "majority": 1.0, "split": 0.85, "unknown": 0.95}

    def _composite_score(self, r: BetRecommendation) -> float:
        agr_bonus = self._AGREEMENT_BONUS.get(r.model_agreement, 1.0)
        cv = getattr(r, "contrarian_value", 0) or 0
        if cv >= 1.3 and r.model_agreement == "unanimous":
            cont_bonus = 1.10
        elif cv >= 1.3 and r.model_agreement == "majority":
            cont_bonus = 1.05
        else:
            cont_bonus = 1.0
        return r.expected_value * r.confidence * agr_bonus * cont_bonus

    def _filter_correlated_picks(
        self, picks: List[BetRecommendation]
    ) -> List[BetRecommendation]:
        """Remove the lower-ranked pick from correlated pairs on the same match.

        When two picks on the same match are positively correlated (e.g.
        Home Win + Over 2.5), keeping both over-concentrates risk. This
        method detects such pairs and drops the one with the lower composite
        score (EV × confidence × agreement × contrarian bonus, same metric
        used by the main sort) — so a unanimous pick is never dropped in
        favour of a split-models pick with marginally higher raw EV.
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
                        score_a = self._composite_score(a)
                        score_b = self._composite_score(b)
                        ev_diff = abs(a.expected_value - b.expected_value)
                        if ev_diff < 0.10:
                            # Small EV gap: prefer better model agreement to avoid
                            # keeping a solo-Poisson pick over a majority/unanimous one
                            _agr_rank = {
                                "unanimous": 4, "majority": 3,
                                "solo": 2, "split": 1, "unknown": 0,
                            }
                            agr_a = _agr_rank.get(a.model_agreement, 0)
                            agr_b = _agr_rank.get(b.model_agreement, 0)
                            if agr_a != agr_b:
                                loser = b if agr_a >= agr_b else a
                                winner = a if agr_a >= agr_b else b
                            else:
                                loser = b if score_a >= score_b else a
                                winner = a if score_a >= score_b else b
                        else:
                            loser = b if score_a >= score_b else a
                            winner = a if score_a >= score_b else b
                        idx = pick_index[id(loser)]
                        if idx not in to_remove:
                            to_remove.add(idx)
                            logger.info(
                                f"Correlation filter: dropping '{loser.selection}' "
                                f"(EV={loser.expected_value:.1%}, agreement="
                                f"{loser.model_agreement}) for {loser.match} — "
                                f"correlated with '{winner.selection}' "
                                f"(EV={winner.expected_value:.1%}, agreement="
                                f"{winner.model_agreement})"
                            )

        if to_remove:
            picks = [p for i, p in enumerate(picks) if i not in to_remove]
        return picks

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



def _configure_cli_runtime():
    """One-time CLI runtime setup (UTF-8 streams + .env load).

    Kept in a dedicated function so importing `main` doesn't replace stdout
    / stderr as a side effect of the import — the previous version did this
    inside main() but at module import time anyone touching the symbol would
    inherit it.  Now it only runs when the CLI entry point actually starts.
    """
    import io
    import sys
    from pathlib import Path

    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    except Exception:
        pass


async def main():
    """CLI entry point."""
    import sys

    _configure_cli_runtime()

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
        print("  --backfill-wc       Backfill FIFA World Cup 2018/2022/2026 match history")
        print("  --briefing          Post AI preview briefings for today's WC matches (no lineups)")
        print("  --prematch-briefing [min] Post lineup-aware AI briefings for WC matches ~min before KO (default 45)")
        print("  --backfill-stats [N] Backfill xG/shots/venue/halftime for historical matches (default budget: 80 requests)")
        print("  --telegram-setup    Setup Telegram bot notifications")
        print("  --telegram-test     Send a test Telegram message")
        print("  --telegram-welcome  Send group welcome/info message to Telegram")
        print("  --backtest-rolling  Show per-month rolling performance over all settled picks")
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
            _max = 500 if _db_tmp.is_postgres else 2000
            await agent.train_ml_models(max_samples=_max)
            print("ML training complete.")

        elif command == "--picks":
            # Parse optional --leagues filter and --force flag
            league_filter = None
            force_picks = "--force" in sys.argv
            for i, arg in enumerate(sys.argv[2:], start=2):
                if arg == "--leagues" and i + 1 < len(sys.argv):
                    league_filter = [l.strip() for l in sys.argv[i + 1].split(",")]
                    break
            agent.predictor.fit()
            agent.feature_engineer.elo_ratings = agent.predictor.elo.ratings
            picks, new_picks, dropped_picks = await agent.get_daily_picks(
                leagues=league_filter, force=force_picks
            )
            if not picks:
                print("No value picks found for today.")
            else:
                # Claude reviews EVERY pick (WC every-match + club value picks),
                # making the final KEEP/CHANGE call on each tracked bet. No
                # briefing article is posted (briefings.send_to_telegram=false);
                # the picks summary below is the only Telegram output and it
                # reflects Claude's finalized selection. Runs before the send so
                # switches are reflected; no-ops if briefings are disabled or no
                # Claude auth is present (model picks sent as-is).
                if agent.config.get("briefings.enabled", True) and agent.config.get(
                    "briefings.finalize_picks", True
                ):
                    try:
                        from src.reporting.match_briefing import MatchBriefingService
                        reviewer = MatchBriefingService(agent)
                        await reviewer.finalize_picks_with_claude(new_picks or picks)
                    except Exception as _rev_e:
                        logger.warning(
                            f"Claude pick review failed — sending model picks as-is: {_rev_e}"
                        )
                # WC mode kept the bulk summary suppressed while briefings were
                # authoritative; with briefings off, the summary IS the output.
                _suppress_summary = agent.config.get(
                    "notifications.suppress_picks_summary", False
                )
                if _suppress_summary:
                    print(
                        f"\nBulk picks summary suppressed "
                        f"(notifications.suppress_picks_summary). {len(new_picks)} new pick(s) saved."
                    )
                # Send to Telegram only the picks that are NEW this run.
                # On a re-run on the same day, previously-sent picks are excluded
                # so users don't receive duplicate notifications.
                if agent.telegram.enabled and not _suppress_summary:
                    if new_picks:
                        stats = agent.get_stats()
                        # Detect when all picks lack injury data (e.g. API suspended)
                        no_injury_data = False
                        injury_data_stale = False
                        injury_budget_exhausted = False
                        try:
                            match_ids = [p.match_id for p in new_picks if p.match_id]
                            if match_ids:
                                with agent.db.get_session() as _isess:
                                    _matches = _isess.query(Match).filter(Match.id.in_(match_ids)).all()
                                    _team_ids = [
                                        tid
                                        for m in _matches
                                        for tid in (m.home_team_id, m.away_team_id)
                                        if tid
                                    ]
                                    if _team_ids:
                                        _today_start = datetime.combine(
                                            date.today(), datetime.min.time()
                                        )
                                        # Check total injury records for today across all teams
                                        _any_today = (
                                            _isess.query(Injury)
                                            .filter(Injury.updated_at >= _today_start)
                                            .count()
                                        )
                                        # No injury records at all today → budget exhausted
                                        injury_budget_exhausted = _any_today == 0
                                        _inj_total = (
                                            _isess.query(Injury)
                                            .filter(Injury.team_id.in_(_team_ids))
                                            .count()
                                        )
                                        no_injury_data = _inj_total == 0
                                        if not no_injury_data:
                                            _fresh = (
                                                _isess.query(Injury)
                                                .filter(
                                                    Injury.team_id.in_(_team_ids),
                                                    Injury.updated_at >= _today_start,
                                                )
                                                .count()
                                            )
                                            injury_data_stale = _fresh == 0
                        except Exception as _ierr:
                            logger.debug(f"Injury data check failed: {_ierr}")
                        await agent.telegram.send_daily_picks(
                            new_picks, stats=stats, dropped_picks=dropped_picks,
                            no_injury_data=no_injury_data,
                            injury_data_stale=injury_data_stale,
                            injury_budget_exhausted=injury_budget_exhausted,
                            force=force_picks,
                        )
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
                        op_str = ""
                        op = getattr(pick, "opening_odds", 0) or 0
                        if op > 1.0 and abs(pick.odds - op) / op >= 0.03:
                            direction = "↑" if pick.odds > op else "↓"
                            op_str = f"  (opened {op:.2f} {direction}{abs((pick.odds-op)/op)*100:.1f}%)"
                        print(f"    Bet: {pick.selection} @ {pick.odds:.2f}{op_str}")
                        print(f"    EV: {pick.expected_value:.1%} | Conf: {pick.confidence:.1%} | Risk: {pick.risk_level}")
                        print(f"    Stake: {pick.kelly_stake_percentage:.1f}% of bankroll")
                        if pick.model_agreement:
                            agr = pick.model_agreement
                            if agr == "solo":
                                line = f"    Models: Only {pick.models_for} (single model signal)"
                            else:
                                line = f"    Models: {agr}{agreement_tag}"
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

            # Query picks from yesterday that are still unresolved after this settle run.
            # Passed to the settlement report so operators can see which picks need follow-up.
            pending_from_yesterday: list = []
            try:
                _yesterday = date.today() - timedelta(days=1)
                with agent.db.get_session() as _psess:
                    _pending_rows = _psess.query(SavedPick).filter(
                        SavedPick.result.is_(None),
                        SavedPick.pick_date == _yesterday,
                    ).all()
                    pending_from_yesterday = [
                        {
                            "match_name": p.match_name,
                            "selection": p.selection,
                            "odds": p.odds,
                            "pick_date": p.pick_date,
                            "league": p.league,
                            "stake": p.kelly_stake_percentage or 0,
                        }
                        for p in _pending_rows
                    ]
            except Exception as _pe:
                logger.warning(f"Could not query yesterday's pending picks: {_pe}")

            # Send settlement report to Telegram — fires even when 0 settled but
            # there are pending picks from yesterday, so operators see named stuck picks.
            if (settled_picks or pending_from_yesterday) and agent.telegram.enabled:
                await agent.telegram.send_settlement_report(
                    settled_picks, stats, pending_picks=pending_from_yesterday
                )
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
            no_settle = "--no-settle" in sys.argv
            if not no_settle:
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

        elif command == "--backtest-rolling":
            print("Rolling backtest — per-month performance over all settled picks")
            agent.rolling_backtest()

        elif command in ("--briefing", "--prematch-briefing"):
            from src.reporting.match_briefing import MatchBriefingService
            agent.predictor.fit()
            agent.feature_engineer.elo_ratings = agent.predictor.elo.ratings
            service = MatchBriefingService(agent)
            if command == "--briefing":
                print("Generating WC preview briefings...")
                n = await service.run_preview_briefings()
            else:
                # Optional T-window override: --prematch-briefing 45
                window = 45
                if len(sys.argv) > 2:
                    try:
                        window = int(sys.argv[2])
                    except ValueError:
                        pass
                print(f"Generating WC pre-match (T-{window}) briefings...")
                n = await service.run_prematch_briefings(window_min=window)
            print(f"Briefings sent: {n}")

        elif command == "--backfill-wc":
            # National-team history backfill via API-Football. Coverage for the
            # WC prediction models requires recent matches for EVERY finalist —
            # WC editions alone are not enough (many 2026 teams missed WC 2022),
            # so qualifiers and friendlies are the primary sources. football-
            # data.org is not used: its free tier 403s on historical seasons.
            print("Backfilling national-team match history via API-Football...")
            # IMPORTANT: the free plan only serves seasons 2022-2024 (verified in
            # run 27339814324 — 2025/2026 requests return a plan error). All
            # targets below are within that window. Friendlies are the ONLY
            # source for the WC hosts (Mexico/USA/Canada play no qualifiers).
            # Already-loaded comps (WC 2022, UEFA quals 2024, CAF quals 2023)
            # dedupe to "0 created" at the cost of 1 request each.
            # Ordered MOST RECENT FIRST: the step can hit its CI timeout
            # mid-list (friendlies seasons take ~20 min of Supabase inserts
            # each), and recent matches matter most (Poisson half-life 180d).
            # Friendlies 2022 dropped — too old to carry model weight.
            _targets = [
                ("world/friendlies", (2024,)),
                ("world/euro-championship", (2024,)),
                ("world/copa-america", (2024,)),
                ("world/concacaf-nations-league", (2024, 2023)),
                ("world/uefa-nations-league", (2024,)),
                ("world/africa-cup-of-nations", (2023,)),
                ("world/asian-cup", (2023,)),
                ("world/gold-cup", (2023,)),
                ("world/friendlies", (2023,)),
                ("world/fifa-world-cup", (2022, 2026)),
                ("world/wc-qualification-europe", (2024,)),
                ("world/wc-qualification-africa", (2023,)),
            ]
            total_created = 0
            total_updated = 0
            for _league_key, _seasons in _targets:
                for _season in _seasons:
                    try:
                        c, u = await agent.apifootball.backfill_competition_season(
                            _league_key, _season
                        )
                    except Exception as _bf_err:
                        print(f"  {_league_key} {_season}: FAILED ({_bf_err})")
                        continue
                    total_created += c
                    total_updated += u
                    print(f"  {_league_key} {_season}: {c} created, {u} updated")
            print(
                f"National-team backfill complete: "
                f"{total_created} created, {total_updated} updated."
            )

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

        elif command == "--backfill-stats":
            # Optional positional arg: budget (default 80 requests).
            # Run on a day without a prior daily update to stay under the 100 req/day limit.
            budget = 80
            if len(sys.argv) > 2:
                try:
                    budget = int(sys.argv[2])
                except ValueError:
                    print(f"Invalid budget '{sys.argv[2]}', using default 80.")
            print(f"Backfilling match stats for historical API-Football-linked matches (budget: {budget} requests)...")
            print("Pass 1 (60%): xG, shots, saves, offsides via /fixtures/statistics")
            print("Pass 2 (40%): venue, halftime, regulation, penalty via /fixtures?id=")
            result = await agent.apifootball.backfill_match_stats(budget=budget)
            print(
                f"Backfill complete: {result['stats']} stats + {result['details']} fixture details updated."
            )
            if result["stats"] + result["details"] == 0:
                print("Nothing to backfill — all API-Football-linked matches already have stats.")
            else:
                remaining = budget - result["stats"] - result["details"]
                print(f"Remaining budget unused: ~{remaining} requests.")
                print("Run again tomorrow to continue backfilling (100 req/day free-tier limit).")

        else:
            print(f"Unknown command: {command}")

    finally:
        await agent.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
