"""Injury data scraper using API-Football /injuries endpoint.

Fetches injuries for today's fixtures so the model can factor in missing
players when generating picks.  Each fixture costs 1 API request.
"""

from datetime import datetime, date, timedelta
from typing import List, Optional

from src.data.models import Player, Injury, Team, Match
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

    async def update(self):
        """Fetch injuries for today's fixtures that have an apifootball_id.

        Uses the shared APIFootballScraper instance so all requests count
        against the single daily quota.  Budget is dynamic: min(fixture_count,
        remaining requests) so low-fixture days release budget for other uses.
        """
        if not self.apifootball or not self.apifootball.enabled:
            logger.debug("API-Football not available — skipping injury update")
            return

        # Dynamic budget: use whatever remains after static steps, capped to
        # actual fixture count (no point reserving 55 if only 20 fixtures today)
        budget_remaining = max(
            0,
            self.apifootball._daily_limit
            - self.apifootball._requests_today
            - self.apifootball.BUDGET_RESERVE
        )
        injury_budget = budget_remaining  # will be capped to len(fixture_list) below
        if injury_budget <= 0:
            logger.debug("No API budget remaining for injuries")
            return

        # Get today's fixtures with apifootball_id
        today_start = datetime.combine(date.today(), datetime.min.time())
        today_end = today_start + timedelta(days=1)

        with self.db.get_session() as session:
            fixtures = session.query(Match).filter(
                Match.is_fixture == True,
                Match.apifootball_id.isnot(None),
                Match.match_date >= today_start,
                Match.match_date < today_end,
            ).all()
            fixture_list = [
                (m.id, m.apifootball_id, m.home_team_id, m.away_team_id)
                for m in fixtures
            ]

        if not fixture_list:
            logger.debug("No fixtures for injury update")
            return

        # Cap budget to actual fixture count — release unused requests for backfill
        injury_budget = min(injury_budget, len(fixture_list))

        logger.info(
            f"Fetching injuries for {injury_budget}"
            f"/{len(fixture_list)} fixtures (budget: {injury_budget})"
        )

        # Clear stale injuries from previous days — keep only today's data
        with self.db.get_session() as session:
            old_count = session.query(Injury).filter(
                Injury.source == "api-football"
            ).count()
            if old_count:
                session.query(Injury).filter(
                    Injury.source == "api-football"
                ).delete()
                session.commit()
                logger.debug(f"Cleared {old_count} stale API-Football injuries")

        fetched = 0
        total_saved = 0
        for match_id, fixture_id, home_team_id, away_team_id in fixture_list:
            if fetched >= injury_budget:
                break
            try:
                count = await self._fetch_fixture_injuries(
                    fixture_id, home_team_id, away_team_id
                )
                total_saved += count
                fetched += 1
            except Exception as e:
                logger.warning(f"Injury fetch failed for fixture {fixture_id}: {e}")

        logger.info(f"Injury update: saved {total_saved} injuries from {fetched} fixtures")

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
                        existing.updated_at = datetime.utcnow()
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
            total_value = sum(
                (i.player.market_value or 0) for i in injuries if i.player
            )

            return {
                "total_injured": len(injuries),
                "key_players_out": key_players_out,
                "total_market_value_lost": total_value,
                "injuries": [
                    {
                        "player": i.player.name if i.player else "Unknown",
                        "type": i.injury_type,
                        "status": i.status,
                        "position": i.player.position if i.player else "",
                        "expected_return": (
                            str(i.expected_return) if i.expected_return else "Unknown"
                        ),
                    }
                    for i in injuries
                ],
            }

    async def close(self):
        """No resources to clean up (API-Football uses shared scraper)."""
        pass
