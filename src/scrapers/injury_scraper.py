"""Injury data scraper using API-Football /injuries endpoint.

Fetches injuries for today's fixtures so the model can factor in missing
players when generating picks.  Each fixture costs 1 API request.
"""

import time as _time
from datetime import datetime, date, timedelta, timezone
from typing import List, Optional

from src.data.models import Player, Injury, Team, Match, Odds
from src.data.database import get_db
from src.utils.logger import get_logger

logger = get_logger()


class InjuryScraper:
    """Fetches injury data from API-Football for today's fixtures."""

    def __init__(self, config=None, apifootball=None):
        """
        Args:
            config: Application config dict.
            apifootball: APIFootballScraper instance (shared to respect quota).
        """
        self.config = config or {}
        self.apifootball = apifootball
        self.db = get_db()

    async def update(self, priority_fixture_ids=None):
        """Fetch injuries for today's fixtures that have an apifootball_id.

        Uses the shared APIFootballScraper instance so all requests count
        against the single daily quota.  Budget is min(fixture_count,
        remaining requests, max_injury_budget) — the cap (default 30)
        leaves room for targeted backfill after injuries complete.

        Args:
            priority_fixture_ids: Optional list of match DB IDs to fetch first.
                                  Used to prioritise fixtures with open picks.
        """
        if not self.apifootball or not self.apifootball.enabled:
            logger.debug("API-Football not available — skipping injury update")
            return

        # Dynamic budget: use remaining requests, capped to fixture count and
        # a configurable max to leave room for targeted backfill after injuries.
        # Keep budget at ≤15 so the sequential API+DB loop finishes within
        # the 5-minute (300s) asyncio timeout.  26 fixtures × ~11s each = ~299s
        # which always times out; 15 × 11s = ~165s which always completes.
        max_injury = 15
        if hasattr(self.config, 'get'):
            max_injury = self.config.get("models.max_injury_budget", 15)
        injury_budget = min(self.apifootball.remaining_budget(), max_injury)
        today_start = datetime.combine(date.today(), datetime.min.time())

        if injury_budget <= 0:
            # No budget — log how much cached data is still available as fallback
            with self.db.get_session() as session:
                cached_total = session.query(Injury).filter(
                    Injury.source == "api-football"
                ).count()
            if cached_total > 0:
                logger.info(
                    f"No API budget for injuries — using {cached_total} cached injuries "
                    f"from previous run as fallback"
                )
            else:
                logger.warning("No API budget for injuries and no cached data available")
            return

        # Get upcoming fixtures that have odds (skip fixtures with no market interest)
        upcoming_cutoff = today_start + timedelta(days=3)

        with self.db.get_session() as session:
            # Only fetch injury data for fixtures with at least one odds record —
            # fixtures without odds have no pick interest and don't need injury context.
            fixtures = (
                session.query(Match)
                .join(Odds, Match.id == Odds.match_id)
                .filter(
                    Match.is_fixture == True,
                    Match.apifootball_id.isnot(None),
                    Match.match_date >= today_start,
                    Match.match_date < upcoming_cutoff,
                )
                .distinct()
                .all()
            )
            fixture_list = [
                (m.id, m.apifootball_id, m.home_team_id, m.away_team_id)
                for m in fixtures
            ]

            # Find teams that already have fresh injury data fetched today
            fresh_team_ids = set(
                row[0]
                for row in session.query(Injury.team_id)
                .filter(
                    Injury.source == "api-football",
                    Injury.updated_at >= today_start,
                )
                .distinct()
                .all()
            )

        if not fixture_list:
            logger.debug("No fixtures for injury update")
            return

        # Skip fixtures where both teams already have today's injury data
        fixtures_to_fetch = [
            (mid, fid, htid, atid)
            for mid, fid, htid, atid in fixture_list
            if htid not in fresh_team_ids or atid not in fresh_team_ids
        ]
        already_fresh_count = len(fixture_list) - len(fixtures_to_fetch)
        if already_fresh_count:
            logger.info(
                f"Injury freshness: {already_fresh_count}/{len(fixture_list)} fixtures "
                f"already have today's data, fetching {len(fixtures_to_fetch)}"
            )
        fixture_list = fixtures_to_fetch

        if not fixture_list:
            logger.info("Injury update: all fixtures already have today's injury data")
            return

        # Prioritise fixtures with open picks so the most important ones are
        # fetched first if the budget or time runs out before the full list.
        if priority_fixture_ids:
            _priority_set = set(priority_fixture_ids)
            fixture_list.sort(key=lambda x: (0 if x[0] in _priority_set else 1))

        # Cap budget to actual fixture count — release unused requests for backfill
        injury_budget = min(injury_budget, len(fixture_list))

        logger.info(
            f"Fetching injuries for {injury_budget}"
            f"/{len(fixture_list)} fixtures (budget: {injury_budget})"
        )

        fetched = 0
        total_saved = 0
        zero_injury_fixtures = []  # track fixtures with no data for team-level fallback
        all_processed_teams: set = set()  # collect for squad stats pass below
        for match_id, fixture_id, home_team_id, away_team_id in fixture_list:
            if fetched >= injury_budget:
                break
            if self.apifootball._plan_restricted:
                logger.debug("Skipping injury fetch — API-Football plan restriction active")
                break
            try:
                _t0 = _time.monotonic()
                count = await self._fetch_fixture_injuries(
                    fixture_id, home_team_id, away_team_id
                )
                _elapsed = _time.monotonic() - _t0
                logger.debug(f"Injury fetch for fixture {fixture_id}: {_elapsed:.1f}s")
                total_saved += count
                fetched += 1
                all_processed_teams.add(home_team_id)
                all_processed_teams.add(away_team_id)
                if count == 0:
                    zero_injury_fixtures.append((home_team_id, away_team_id))
            except Exception as e:
                logger.warning(f"Injury fetch failed for fixture {fixture_id}: {e}")

        # Team-level fallback: fixture endpoint returns nothing for lower leagues.
        # Try /injuries?team={id}&season=YEAR for each team that had 0 injuries.
        # Each team costs 1 request; cap total fallback to leave budget for picks.
        if zero_injury_fixtures and self.apifootball.remaining_budget() > 0 and not self.apifootball._plan_restricted:
            current_year = date.today().year
            team_budget = min(self.apifootball.remaining_budget(), len(zero_injury_fixtures) * 2, 10)
            team_fetched = 0
            seen_teams: set = set()
            for home_id, away_id in zero_injury_fixtures:
                for team_db_id in (home_id, away_id):
                    if team_fetched >= team_budget:
                        break
                    if team_db_id in seen_teams:
                        continue
                    seen_teams.add(team_db_id)
                    try:
                        count = await self._fetch_team_injuries(team_db_id, current_year)
                        total_saved += count
                        team_fetched += 1
                    except Exception as e:
                        logger.debug(f"Team injury fallback failed for team {team_db_id}: {e}")
            if team_fetched:
                logger.debug(f"Team injury fallback: {team_fetched} teams queried, {total_saved} total saved")

        logger.info(f"Injury update: saved {total_saved} injuries from {fetched} fixtures")

        # Squad stats pass: for each team we just processed, update player
        # positions and is_key_player from /players?team&season stats.
        # Only targets teams that still have players with empty position strings.
        # Budget: min(remaining - 5, teams * 2, 20) — reserve 5 calls for backfill.
        if all_processed_teams and not self.apifootball._plan_restricted:
            season_year = date.today().year
            if date.today().month < 7:
                season_year -= 1  # e.g. May 2026 -> 2025 season
            with self.db.get_session() as _s:
                teams_needing_squad = [
                    tid for tid in all_processed_teams
                    if _s.query(Player).filter(
                        Player.team_id == tid, Player.position == ""
                    ).first()
                ]
            squad_budget = min(
                max(0, self.apifootball.remaining_budget() - 5),
                len(teams_needing_squad) * 2,
                20,
            )
            if squad_budget > 0 and teams_needing_squad:
                logger.info(
                    f"Squad stats: updating {len(teams_needing_squad)} teams "
                    f"with empty player positions (budget: {squad_budget})"
                )
                for team_db_id in teams_needing_squad:
                    if self.apifootball.remaining_budget() <= 5:
                        break
                    try:
                        await self._fetch_and_update_squad_stats(team_db_id, season_year)
                    except Exception as e:
                        logger.debug(f"Squad stats failed for team {team_db_id}: {e}")

        # Log fixture IDs skipped due to budget exhaustion (AC3 of Story 5.1)
        skipped_ids = [match_id for match_id, _, _, _ in fixture_list[fetched:]]
        if skipped_ids:
            logger.info(f"Injury data missing for fixture IDs (budget cutoff): {skipped_ids}")
        elif fixture_list:
            logger.info(f"Injury data complete for all {len(fixture_list)} fixtures")

        # Purge injuries older than 48h — keep today's and yesterday's data as fallback.
        # Never bulk-delete before fetching (that destroys the fallback on budget-zero runs).
        cutoff_48h = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=48)
        with self.db.get_session() as session:
            old_count = session.query(Injury).filter(
                Injury.source == "api-football",
                Injury.updated_at < cutoff_48h,
            ).count()
            if old_count:
                session.query(Injury).filter(
                    Injury.source == "api-football",
                    Injury.updated_at < cutoff_48h,
                ).delete()
                session.commit()
                logger.debug(f"Purged {old_count} injury records older than 48h")

    async def _fetch_fixture_injuries(
        self, fixture_id: int, home_team_id: int, away_team_id: int
    ) -> int:
        """Fetch and save injuries for a single fixture from API-Football.

        API endpoint: GET /injuries?fixture={id}
        Response contains player name, type/reason, and team info.

        Returns number of injuries saved.
        """
        data = await self.apifootball._api_get(
            "/injuries", {"fixture": fixture_id}
        )
        if not data:
            return 0

        response = data.get("response", [])
        if not response:
            return 0

        saved = 0
        with self.db.get_session() as session:
            for entry in response:
                try:
                    player_info = entry.get("player", {})
                    team_info = entry.get("team", {})
                    player_name = player_info.get("name", "")
                    reason = player_info.get("reason", "Unknown")
                    player_type = player_info.get("type", "")  # "Missing Fixture"
                    player_photo = player_info.get("photo", "")
                    api_team_id = team_info.get("id")

                    if not player_name:
                        continue

                    # Map API-Football team ID to our internal team_id
                    team_id = None
                    if api_team_id:
                        team = session.query(Team).filter_by(
                            apifootball_team_id=api_team_id
                        ).first()
                        if team:
                            team_id = team.id
                    # Fallback: use home/away from fixture
                    if not team_id:
                        team_id = home_team_id

                    # Determine status from type field
                    status = "out"
                    if player_type and "doubtful" in player_type.lower():
                        status = "doubtful"

                    # Find or create player
                    player = session.query(Player).filter_by(
                        name=player_name,
                        team_id=team_id,
                    ).first()
                    if not player:
                        player = Player(
                            name=player_name,
                            team_id=team_id,
                            position="",
                        )
                        session.add(player)
                        session.flush()

                    # Check for existing injury (same player + type)
                    existing = session.query(Injury).filter_by(
                        player_id=player.id,
                        injury_type=reason,
                    ).first()

                    if existing:
                        existing.status = status
                        existing.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
                    else:
                        injury = Injury(
                            player_id=player.id,
                            team_id=team_id,
                            injury_type=reason,
                            start_date=date.today(),
                            status=status,
                            source="api-football",
                        )
                        session.add(injury)
                        saved += 1

                except Exception as e:
                    logger.debug(f"Error parsing injury entry: {e}")
                    continue

            session.commit()
        return saved

    async def _fetch_team_injuries(self, team_db_id: int, season: int) -> int:
        """Fallback: fetch injuries for a team by season from API-Football.

        Used when the fixture-level endpoint returns nothing (common for lower
        leagues).  Queries /injuries?team={apifootball_id}&season={year}.

        Returns number of new injuries saved.
        """
        # Look up the API-Football team ID
        with self.db.get_session() as session:
            team = session.query(Team).filter_by(id=team_db_id).first()
            if not team or not team.apifootball_team_id:
                return 0
            api_team_id = team.apifootball_team_id

        data = await self.apifootball._api_get(
            "/injuries", {"team": api_team_id, "season": season}
        )
        if not data:
            return 0
        response = data.get("response", [])
        if not response:
            return 0

        saved = 0
        with self.db.get_session() as session:
            for entry in response:
                try:
                    player_info = entry.get("player", {})
                    player_name = player_info.get("name", "")
                    reason = player_info.get("reason", "Unknown")
                    player_type = player_info.get("type", "")

                    if not player_name:
                        continue

                    status = "out"
                    if player_type and "doubtful" in player_type.lower():
                        status = "doubtful"

                    player = session.query(Player).filter_by(
                        name=player_name, team_id=team_db_id
                    ).first()
                    if not player:
                        player = Player(name=player_name, team_id=team_db_id, position="")
                        session.add(player)
                        session.flush()

                    existing = session.query(Injury).filter_by(
                        player_id=player.id, injury_type=reason
                    ).first()
                    if existing:
                        existing.status = status
                        existing.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
                    else:
                        session.add(Injury(
                            player_id=player.id,
                            team_id=team_db_id,
                            injury_type=reason,
                            start_date=date.today(),
                            status=status,
                            source="api-football",
                        ))
                        saved += 1
                except Exception as e:
                    logger.debug(f"Error parsing team injury entry: {e}")
                    continue
            session.commit()
        return saved

    async def _fetch_and_update_squad_stats(self, team_db_id: int, season_year: int) -> int:
        """Fetch player stats for a team and update position + is_key_player.

        Uses /players?team={api_id}&season={year} (paginated, ~20 players/page).
        Position is taken from statistics[0].games.position.
        is_key_player = appearances >= 10 OR minutes >= 900 in the season.

        Only overwrites position if it is currently empty.
        Always overwrites is_key_player so it stays current with playing time.

        Returns number of player records updated.
        """
        with self.db.get_session() as _s:
            team = _s.query(Team).filter_by(id=team_db_id).first()
            if not team or not team.apifootball_team_id:
                return 0
            api_team_id = team.apifootball_team_id

        updated = 0
        page = 1
        while True:
            data = await self.apifootball._api_get(
                "/players",
                {"team": api_team_id, "season": season_year, "page": page},
            )
            if not data:
                break

            response = data.get("response", [])
            if not response:
                break

            total_pages = data.get("paging", {}).get("total", 1)

            with self.db.get_session() as session:
                for entry in response:
                    try:
                        player_info = entry.get("player", {})
                        stats_list = entry.get("statistics", [])
                        if not player_info or not stats_list:
                            continue

                        player_name = player_info.get("name", "")
                        if not player_name:
                            continue

                        games = stats_list[0].get("games", {})
                        position = games.get("position") or ""
                        # API-Football spells it "appearences" (typo in their API)
                        appearances = games.get("appearences") or 0
                        minutes = games.get("minutes") or 0
                        is_key = bool(appearances >= 10 or minutes >= 900)

                        player = session.query(Player).filter_by(
                            name=player_name, team_id=team_db_id
                        ).first()
                        if player:
                            if not player.position:
                                player.position = position
                            player.is_key_player = is_key
                            updated += 1
                        else:
                            session.add(Player(
                                name=player_name,
                                team_id=team_db_id,
                                position=position,
                                is_key_player=is_key,
                            ))
                            updated += 1
                    except Exception as e:
                        logger.debug(f"Squad stats parse error for team {team_db_id}: {e}")
                session.commit()

            if page >= total_pages or self.apifootball.remaining_budget() <= 5:
                break
            page += 1

        logger.debug(
            f"Squad stats updated: team_db_id={team_db_id}, "
            f"season={season_year}, {updated} players"
        )
        return updated

    async def get_team_injuries(self, team_id: int) -> List[Injury]:
        """Get current injuries for a team from the database."""
        with self.db.get_session() as session:
            return session.query(Injury).filter_by(
                team_id=team_id,
                status="out",
            ).all()

    async def get_injury_summary(self, team_id: int) -> dict:
        """Get an injury impact summary for a team."""
        with self.db.get_session() as session:
            injuries = session.query(Injury).join(Player).filter(
                Injury.team_id == team_id,
                Injury.status.in_(["out", "doubtful"]),
            ).all()

            key_players_out = sum(
                1 for i in injuries if i.player and i.player.is_key_player
            )

            return {
                "total_injured": len(injuries),
                "key_players_out": key_players_out,
                "injuries": [
                    {
                        "player": i.player.name if i.player else "Unknown",
                        "type": i.injury_type,
                        "status": i.status,
                        "position": i.player.position if i.player else "",
                        "is_key_player": i.player.is_key_player if i.player else False,
                    }
                    for i in injuries
                ],
            }

    async def close(self):
        """No resources to clean up (API-Football uses shared scraper)."""
        pass
