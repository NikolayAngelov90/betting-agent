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
from src.scrapers.news_scraper import NewsScraper
from src.scrapers.historical_loader import HistoricalDataLoader
from src.scrapers.apifootball_scraper import APIFootballScraper
from src.scrapers.footballdataorg_scraper import FootballDataOrgScraper
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
        self.injury_tracker = InjuryScraper(self.config)
        self.news_aggregator = NewsScraper(self.config)
        self.historical_loader = HistoricalDataLoader(self.config)
        self.apifootball = APIFootballScraper(self.config)
        self.footballdataorg = FootballDataOrgScraper(self.config)
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

        # 1b. football-data.org: fixtures + results for 9 top leagues (free, no quota).
        # Runs before Flashscore so it can fill gaps when Chrome scraping fails.
        # Covers: PL, BL1, La Liga, Serie A, Ligue 1, CL, Eredivisie, Primeira Liga, ELC.
        try:
            fdo_results = await self.footballdataorg.update_results(days_back=1)
            fdo_fixtures = await self.footballdataorg.sync_fixtures(days_ahead=2)
            # Bulk-import current season matches for all 9 competitions. FDO has no
            # daily limit so this runs for free. Fills coverage gaps for teams that
            # Flashscore missed or that were promoted mid-season.
            fdo_hist = await self.footballdataorg.backfill_historical_seasons(seasons=[2025])
            logger.info(
                f"football-data.org: {fdo_results} scores updated, "
                f"{fdo_fixtures} new fixtures added, {fdo_hist} historical matches saved"
            )
        except Exception as e:
            logger.warning(f"football-data.org update failed (non-critical): {e}")

        # 2. Flashscore results + fixtures (no API quota — primary fixture source)
        # Results and fixtures are scraped independently so a results timeout does
        # not prevent today's fixtures from being loaded.
        leagues = self.config.get("scraping.flashscore_leagues", [])
        try:
            for league in leagues:
                try:
                    await asyncio.wait_for(
                        self.scraper.scrape_league_results(league, skip_stats=True),
                        timeout=60,
                    )
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

        # Fixtures (today's upcoming matches) — always attempted independently
        # Driver is freshly created here (results session was closed above).
        try:
            for league in leagues:
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

        # 2b. Scrape Flashscore odds for upcoming fixtures (1X2 + O/U + BTTS).
        # Runs after fixtures are saved so all flashscore_ids are populated.
        try:
            from datetime import date, timedelta as _td
            from src.data.models import Odds as _Odds
            _today = date.today()
            _start = datetime.combine(_today, datetime.min.time())
            _end = _start + _td(days=1)  # today only
            with self.db.get_session() as _sess:
                _upcoming = _sess.query(Match).filter(
                    Match.is_fixture == True,
                    Match.match_date >= _start,
                    Match.match_date < _end,
                    Match.flashscore_id.isnot(None),
                ).all()
                # Check per market type so O/U + BTTS get scraped even when
                # a previous run already stored 1X2 odds for this fixture.
                _to_scrape = []
                for _m in _upcoming:
                    _missing = []
                    if _sess.query(_Odds).filter_by(
                        match_id=_m.id, bookmaker="Flashscore", market_type="1X2"
                    ).count() == 0:
                        _missing.append("home-draw-away")
                    if _sess.query(_Odds).filter_by(
                        match_id=_m.id, bookmaker="Flashscore", market_type="over_under"
                    ).count() == 0:
                        _missing.append("over-under")
                    if _sess.query(_Odds).filter_by(
                        match_id=_m.id, bookmaker="Flashscore", market_type="btts"
                    ).count() == 0:
                        _missing.append("btts")
                    if _missing:
                        _to_scrape.append((_m.id, _m.flashscore_id, tuple(_missing)))
            if _to_scrape:
                logger.info(
                    f"Pre-caching Flashscore odds for {len(_to_scrape)} upcoming fixtures..."
                )
                _odds_scraper = FlashscoreScraper()
                _cf_failures = 0  # consecutive Cloudflare / no-data failures
                _cf_abort_threshold = 6  # raised from 3: some fixtures legitimately have no odds
                import time as _odds_timer
                _ODDS_BUDGET_S = 1200  # 20-minute hard cap — prevents odds from eating the full job
                _odds_deadline = _odds_timer.monotonic() + _ODDS_BUDGET_S
                try:
                    for _mid, _fsid, _markets in _to_scrape:
                        if _odds_timer.monotonic() > _odds_deadline:
                            logger.warning(
                                f"Odds pre-caching: {_ODDS_BUDGET_S // 60}-minute budget "
                                f"exhausted — stopping early; remaining fixtures will use "
                                f"API-Football odds"
                            )
                            break
                        if _cf_failures >= _cf_abort_threshold:
                            logger.warning(
                                f"Cloudflare blocking detected ({_cf_failures} consecutive "
                                f"failures) — aborting Flashscore odds pre-cache; "
                                f"API-Football will cover remaining fixtures"
                            )
                            break
                        try:
                            from src.data.models import Odds as _OddsCheck
                            _odds_before = 0
                            with self.db.get_session() as _chk:
                                _odds_before = _chk.query(_OddsCheck).filter_by(
                                    match_id=_mid, bookmaker="Flashscore"
                                ).count()
                            await _odds_scraper.scrape_and_save_odds(
                                _mid, _fsid,
                                markets=_markets,
                            )
                            with self.db.get_session() as _chk:
                                _odds_after = _chk.query(_OddsCheck).filter_by(
                                    match_id=_mid, bookmaker="Flashscore"
                                ).count()
                            if _odds_after > _odds_before:
                                _cf_failures = 0  # reset on success
                            else:
                                _cf_failures += 1
                        except Exception as _exc:
                            logger.warning(f"Odds pre-cache failed for {_mid}: {_exc}")
                            _cf_failures += 1
                finally:
                    _odds_scraper.close_driver()
                logger.info("Flashscore odds pre-caching complete.")
        except Exception as e:
            logger.error(f"Flashscore odds pre-cache failed: {e}")

        # 2c. Flashscore per-match stats enrichment (shots, possession, corners, etc.)
        # stats_only=True skips the referee/venue page (~12s/match vs ~22s), so 50
        # matches ≈ 700s — comfortably inside the 15-minute cap.
        # days_back=14 catches matches missed if CI was down for a day or two.
        #
        # Always open a fresh Chrome session here — the previous session may have
        # been flagged by Cloudflare during odds pre-caching, causing all subsequent
        # requests to silently return empty pages (run #29: 0/40 enriched despite
        # match detail pages not being odds-comparison protected).
        self.scraper.close_driver()
        try:
            await asyncio.wait_for(
                self.scraper.enrich_recent_match_stats(days_back=14, max_matches=50),
                timeout=900,  # 15-minute cap for the whole pass
            )
        except asyncio.TimeoutError:
            logger.warning("Flashscore stats enrichment timed out (partial results saved)")
        except Exception as e:
            logger.error(f"Flashscore stats enrichment failed: {e}")
        finally:
            try:
                self.scraper.close_driver()
            except Exception:
                pass

        # 3. API-Football (fixtures, xG, advanced stats)
        try:
            await self.apifootball.update()
            logger.info("API-Football update complete")
        except Exception as e:
            logger.error(f"API-Football update failed: {e}")

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
            self.feature_engineer.elo_ratings = self.predictor.elo.ratings
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
                injury_report={}, news_summary={},
            )

        # Generate features
        features = await self.feature_engineer.create_features(match_id)
        feature_vector = self.feature_engineer.create_feature_vector(features)
        feature_names = self.feature_engineer.get_feature_names(features)

        # Get predictions
        predictions = self.predictor.predict(home_id, away_id, feature_vector,
                                             feature_names=feature_names,
                                             league=league)

        # Fetch injury and news data in parallel — all four are local-DB reads
        (
            home_injuries, away_injuries,
            home_sentiment, away_sentiment,
        ) = await asyncio.gather(
            self.injury_tracker.get_injury_summary(home_id),
            self.injury_tracker.get_injury_summary(away_id),
            self.news_aggregator.get_team_sentiment(home_id),
            self.news_aggregator.get_team_sentiment(away_id),
        )
        injury_report = {"home": home_injuries, "away": away_injuries}
        news_summary = {"home": home_sentiment, "away": away_sentiment}

        # Build context for reasoning
        context = self._build_context(features, injury_report, news_summary)

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
            news_summary=news_summary,
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
            now = datetime.utcnow()
            query = session.query(Match).filter(
                Match.is_fixture == True,
                Match.match_date >= day_start,
                Match.match_date < day_end,
                Match.match_date > now,
            )
            if leagues:
                query = query.filter(Match.league.in_(leagues))
            fixtures = query.all()
            fixture_ids = [f.id for f in fixtures]

        if not fixture_ids:
            logger.info(f"No fixtures found for {target}")
            return []

        # ── Scrape live Flashscore odds for today's fixtures ──────────────────
        # Check per-market which fixtures are still missing odds so that a
        # partial run (e.g. --update aborted after 3 CF failures) doesn't
        # prevent the remaining markets from being scraped here.
        _MARKET_MAP = [
            ("home-draw-away", "1X2"),
            ("over-under",     "over_under"),
            ("btts",           "btts"),
        ]
        with self.db.get_session() as session:
            to_scrape = []
            for fid in fixture_ids:
                m = session.get(Match, fid)
                if not (m and m.flashscore_id):
                    continue
                missing = [
                    slug for slug, mtype in _MARKET_MAP
                    if session.query(Odds).filter_by(
                        match_id=fid, bookmaker="Flashscore", market_type=mtype
                    ).count() == 0
                ]
                if missing:
                    # Track how many markets are missing (fewer = partial, more likely to succeed)
                    to_scrape.append((fid, m.flashscore_id, tuple(missing), len(missing)))

        if to_scrape:
            # Sort: partial-odds fixtures first (1-2 missing markets), then full-missing (3).
            # Partial fixtures are more likely to succeed (1X2 worked, just O/U or BTTS needed).
            # Full-missing fixtures are more likely to be CF-blocked or have no data at all.
            to_scrape.sort(key=lambda x: x[3])
            logger.info(f"Scraping Flashscore odds for {len(to_scrape)} fixtures...")
            import time as _picks_timer
            _PICKS_ODDS_BUDGET_S = 600  # 10-minute cap — update already did the heavy lifting
            _picks_odds_deadline = _picks_timer.monotonic() + _PICKS_ODDS_BUDGET_S
            _picks_cf_failures = 0
            _picks_cf_abort = 5  # higher threshold: some fixtures legitimately have no odds
            scraper = FlashscoreScraper()
            try:
                for match_id, fs_id, markets, _n_missing in to_scrape:
                    if _picks_timer.monotonic() > _picks_odds_deadline:
                        logger.warning(
                            f"Picks odds scraping: {_PICKS_ODDS_BUDGET_S // 60}-min budget "
                            f"exhausted — remaining fixtures will use existing odds"
                        )
                        break
                    if _picks_cf_failures >= _picks_cf_abort:
                        logger.warning(
                            f"Cloudflare blocking detected ({_picks_cf_failures} consecutive "
                            f"failures) — aborting odds scrape in picks"
                        )
                        break
                    try:
                        _odds_before = 0
                        with self.db.get_session() as _chk:
                            _odds_before = _chk.query(Odds).filter_by(
                                match_id=match_id, bookmaker="Flashscore"
                            ).count()
                        await scraper.scrape_and_save_odds(
                            match_id, fs_id, markets=markets,
                        )
                        with self.db.get_session() as _chk:
                            _odds_after = _chk.query(Odds).filter_by(
                                match_id=match_id, bookmaker="Flashscore"
                            ).count()
                        if _odds_after > _odds_before:
                            _picks_cf_failures = 0
                        else:
                            _picks_cf_failures += 1
                    except Exception as exc:
                        logger.warning(f"Odds scrape failed for match {match_id}: {exc}")
                        _picks_cf_failures += 1
            finally:
                scraper.close_driver()
            logger.info("Flashscore odds scraping complete.")
        else:
            logger.info("All fixtures already have odds or no flashscore_id available.")

        # ── API-Football odds fallback ─────────────────────────────────────────
        # For fixtures that still have NO odds from any source, try API-Football
        # if they have an apifootball_id. This catches Europa/Conference League
        # matches that Flashscore missed or that have no flashscore_id.
        if self.apifootball.enabled:
            with self.db.get_session() as session:
                apifb_fallback = []
                for fid in fixture_ids:
                    m = session.get(Match, fid)
                    if m and m.apifootball_id:
                        total_odds = session.query(Odds).filter_by(match_id=fid).count()
                        if total_odds == 0:
                            ht = m.home_team.name if m.home_team else str(m.home_team_id)
                            at = m.away_team.name if m.away_team else str(m.away_team_id)
                            apifb_fallback.append((fid, m.apifootball_id, ht, at))
            if apifb_fallback:
                logger.info(f"Fetching API-Football odds for {len(apifb_fallback)} fixtures with no odds")
                for match_id, fixture_id, home, away in apifb_fallback:
                    try:
                        odds_data = await self.apifootball._fetch_fixture_odds(fixture_id)
                        if odds_data:
                            count = self.apifootball._save_fixture_odds(match_id, odds_data)
                            logger.info(f"API-Football fallback: saved {count} odds for {home} vs {away}")
                        else:
                            logger.warning(f"No odds from any source for fixture {fixture_id} ({home} vs {away})")
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

        # Analyze all fixtures concurrently — injury and news lookups are local-DB
        # queries, so running them in parallel is safe with SQLite WAL mode.
        logger.info(f"Analyzing {len(fixture_ids)} fixtures concurrently...")
        analyses = await asyncio.gather(
            *(self.analyze_fixture(mid) for mid in fixture_ids),
            return_exceptions=True,
        )

        all_recommendations = []
        for mid, result in zip(fixture_ids, analyses):
            if isinstance(result, Exception):
                logger.error(f"Error analyzing match {mid}: {result}")
                continue
            # value_calculator.find_value_bets() already applied EV/confidence
            # filters from config — no secondary filter needed here.
            all_recommendations.extend(result.recommendations)

        # Sort order mirrors value_calculator: EV × confidence × agreement bonus.
        # This ensures the portfolio cap (below) takes the best-conviction picks first.
        _agreement_bonus = {"unanimous": 1.15, "majority": 1.0, "split": 0.85, "unknown": 0.95}
        all_recommendations.sort(
            key=lambda r: r.expected_value * r.confidence
            * _agreement_bonus.get(r.model_agreement, 1.0),
            reverse=True,
        )

        # Limit picks per match and deduplicate identical (match, selection) pairs.
        # Use (match string, selection) as key to also catch the case where the
        # same physical game has two DB rows with different IDs (e.g. scraped twice).
        seen_pick_keys: set = set()
        limited = []
        if max_picks_per_match:
            from collections import Counter
            match_counts: dict = Counter()
            # Pre-populate from already-saved picks for today so re-running --picks
            # doesn't accumulate more than max_picks_per_match across multiple runs.
            with self.db.get_session() as _sess:
                _existing = _sess.query(SavedPick.match_id).filter(
                    SavedPick.pick_date == target
                ).all()
                for (_mid,) in _existing:
                    match_counts[_mid] += 1
        for rec in all_recommendations:
            key = (rec.match, rec.selection)
            if key in seen_pick_keys:
                logger.debug(f"Skipping duplicate pick for {rec.match}: {rec.selection}")
                continue
            if max_picks_per_match and match_counts[rec.match_id] >= max_picks_per_match:
                logger.debug(
                    f"Skipping extra pick for match {rec.match}: "
                    f"{rec.selection} (already {max_picks_per_match} picks)"
                )
                continue
            seen_pick_keys.add(key)
            limited.append(rec)
            if max_picks_per_match:
                match_counts[rec.match_id] += 1
        all_recommendations = limited

        # No portfolio caps — include all value picks to avoid missing edges.
        # Risk is managed at the Kelly fraction and max_stake_percentage level.

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
                    odds=pick.odds,
                    predicted_probability=pick.predicted_probability,
                    expected_value=pick.expected_value,
                    confidence=pick.confidence,
                    kelly_stake_percentage=pick.kelly_stake_percentage,
                    risk_level=pick.risk_level,
                    used_fallback_odds=pick.used_fallback_odds,
                )
                session.add(saved)
                new_picks.append(pick)

            session.commit()
            logger.info(f"Saved {len(new_picks)} new picks to database (skipped {len(picks) - len(new_picks)} duplicates)")
        return new_picks

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

        # 2. PRIMARY: Flashscore — first-page results only, no quota needed.
        # skip_stats=True means no load_more clicks and no detail page scraping,
        # so the first page (~10-20 latest results) loads in ~10-15s per league.
        if leagues_with_pending:
            logger.info(
                f"Flashscore settle: fetching results for "
                f"{len(leagues_with_pending)} leagues"
            )
            for league in leagues_with_pending:
                try:
                    await asyncio.wait_for(
                        self.scraper.scrape_league_results(league, skip_stats=True),
                        timeout=60,
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"Flashscore settle timeout for {league}, skipping")
                    # Close the driver so the still-running background thread's
                    # session is terminated before the next league starts.
                    try:
                        self.scraper.close_driver()
                    except Exception:
                        pass
                except Exception as e:
                    logger.debug(f"Flashscore settle error for {league}: {e}")
            # Final close after the full loop (no-op if already closed above).
            try:
                self.scraper.close_driver()
            except Exception:
                pass

        # Flashscore is the sole results source — API-Football free plan only
        # allows today±1, so calling it here causes plan-restriction errors.

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
                # Fallback: if the primary match_id has no result, scan same league+date
                # by team-name similarity (handles cross-source name mismatches)
                if (not match or match.home_goals is None or match.away_goals is None) and pick.match_name:
                    parts = pick.match_name.split(" vs ", 1)
                    if len(parts) == 2 and pick.match_id:
                        ref_match = session.get(Match, pick.match_id)
                        ref_date = ref_match.match_date if ref_match else None
                        if ref_date:
                            from datetime import timedelta as _td
                            window_start = ref_date - _td(hours=24)
                            window_end = ref_date + _td(hours=24)
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
                elif sel == "Over 3.5 Goals":
                    won = total > 3.5
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
                pick.settled_at = datetime.utcnow()

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
                    match_league = match.league or ""

                # Actual result
                if pick.actual_home_goals > pick.actual_away_goals:
                    actual = "home_win"
                elif pick.actual_home_goals == pick.actual_away_goals:
                    actual = "draw"
                else:
                    actual = "away_win"

                # Get individual model predictions
                poisson_pred = self.predictor.poisson.predict(home_id, away_id, league=match_league)
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

        # Ensemble accuracy (check if the ensemble's top pick matched)
        ensemble_correct = 0
        ensemble_total = 0
        for pick in settled:
            if pick.market != "1X2":
                continue
            if pick.actual_home_goals is None:
                continue
            if pick.actual_home_goals > pick.actual_away_goals:
                actual = "home_win"
            elif pick.actual_home_goals == pick.actual_away_goals:
                actual = "draw"
            else:
                actual = "away_win"
            # Check if the ensemble's top selection matches actual
            sel_map = {"Home Win": "home_win", "Draw": "draw", "Away Win": "away_win"}
            if sel_map.get(pick.selection) == actual:
                ensemble_correct += 1
            ensemble_total += 1

        if ensemble_total > 0:
            accuracies["ensemble"] = ensemble_correct / ensemble_total

        # Normalize to get new weights (exclude ensemble from weight calc)
        base_models = {k: v for k, v in accuracies.items() if k != "ensemble"}
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

        # Fit Poisson/Elo first (needed for feature context)
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

        if len(match_data) < 100:
            logger.warning("Not enough matches for ML training (need 100+)")
            return

        # Build feature matrix and labels — process in parallel batches
        X_list = []
        y_list = []        # 1X2: 0=away, 1=draw, 2=home
        y_goals_list = []  # over/under 2.5: 1=over, 0=under/equal
        feature_names = None
        skipped = 0
        BATCH_SIZE = 50  # concurrent DB sessions; WAL mode supports parallel reads

        for batch_start in range(0, len(match_data), BATCH_SIZE):
            batch = match_data[batch_start:batch_start + BATCH_SIZE]

            # Fan out: compute features for all matches in the batch concurrently
            batch_results = await asyncio.gather(
                *(self.feature_engineer.create_features(md["id"], as_of_date=md["match_date"]) for md in batch),
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
            if processed % 200 == 0 or processed == len(match_data):
                logger.info(f"Processed {processed}/{len(match_data)} matches ({len(X_list)} valid)...")

        if len(X_list) < 100:
            logger.warning(f"Only {len(X_list)} valid samples (skipped {skipped}), need 100+")
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
        await self.news_aggregator.close()
        await self.historical_loader.close()
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
        print("  --update            Run daily data update")
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
            result = agent.tune_ensemble_weights()
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

        else:
            print(f"Unknown command: {command}")

    finally:
        await agent.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
