"""Injury and squad data scraper."""

from datetime import datetime, date
from typing import List, Optional

from bs4 import BeautifulSoup

from src.scrapers.base_scraper import BaseScraper
from src.data.models import Player, Injury, Team
from src.data.database import get_db
from src.utils.logger import get_logger

logger = get_logger()

TRANSFERMARKT_BASE = "https://www.transfermarkt.com"


class InjuryScraper(BaseScraper):
    """Scrapes injury and squad information from Transfermarkt and other sources."""

    def __init__(self, config=None):
        super().__init__(config)

    async def update(self):
        """Update injury data for all teams in the database."""
        logger.info("Starting injury update cycle")
        db = get_db()

        with db.get_session() as session:
            teams = session.query(Team).filter(Team.transfermarkt_id.isnot(None)).all()
            team_ids = [(t.id, t.transfermarkt_id, t.name) for t in teams]

        for team_id, tm_id, team_name in team_ids:
            try:
                await self.scrape_team_injuries(team_id, tm_id)
                await self._rate_limit()
            except Exception as e:
                logger.error(f"Error scraping injuries for {team_name}: {e}")

        logger.info("Injury update cycle complete")

    async def scrape_team_injuries(self, team_id: int, transfermarkt_id: str) -> List[dict]:
        """Scrape current injuries for a team from Transfermarkt.

        Args:
            team_id: Internal team database ID
            transfermarkt_id: Transfermarkt team identifier

        Returns:
            List of injury data dictionaries
        """
        url = f"{TRANSFERMARKT_BASE}/club/injuries/verein/{transfermarkt_id}"
        logger.info(f"Scraping injuries for team_id={team_id}")

        try:
            html = await self.fetch(url)
        except Exception as e:
            logger.error(f"Failed to fetch injuries page: {e}")
            return []

        injuries = self._parse_injuries_page(html, team_id)

        db = get_db()
        with db.get_session() as session:
            for injury_data in injuries:
                self._save_injury(session, injury_data)

        logger.info(f"Scraped {len(injuries)} injuries for team_id={team_id}")
        return injuries

    def _parse_injuries_page(self, html: str, team_id: int) -> List[dict]:
        """Parse the Transfermarkt injuries page."""
        soup = BeautifulSoup(html, "html.parser")
        injuries = []

        table = soup.find("table", class_="items")
        if not table:
            return injuries

        rows = table.find_all("tr", class_=["odd", "even"])
        for row in rows:
            try:
                cells = row.find_all("td")
                if len(cells) < 4:
                    continue

                # Player name
                player_link = row.find("a", class_="spielprofil_tooltip")
                if not player_link:
                    continue

                player_name = player_link.text.strip()
                tm_player_id = player_link.get("id", "")

                # Position
                position_cell = cells[1] if len(cells) > 1 else None
                position = position_cell.text.strip() if position_cell else ""

                # Injury type
                injury_cell = cells[2] if len(cells) > 2 else None
                injury_type = injury_cell.text.strip() if injury_cell else "Unknown"

                # Date since
                since_cell = cells[3] if len(cells) > 3 else None
                start_date = None
                if since_cell:
                    try:
                        start_date = datetime.strptime(since_cell.text.strip(), "%b %d, %Y").date()
                    except ValueError:
                        start_date = date.today()

                # Expected return
                return_cell = cells[4] if len(cells) > 4 else None
                expected_return = None
                if return_cell:
                    return_text = return_cell.text.strip()
                    if return_text and return_text != "?":
                        try:
                            expected_return = datetime.strptime(return_text, "%b %d, %Y").date()
                        except ValueError:
                            pass

                injuries.append({
                    "team_id": team_id,
                    "player_name": player_name,
                    "transfermarkt_id": tm_player_id,
                    "position": position,
                    "injury_type": injury_type,
                    "start_date": start_date,
                    "expected_return": expected_return,
                    "status": "out",
                    "source": "transfermarkt",
                })

            except Exception as e:
                logger.debug(f"Error parsing injury row: {e}")
                continue

        return injuries

    def _save_injury(self, session, injury_data: dict):
        """Save an injury record, creating player if needed."""
        # Find or create player
        player = session.query(Player).filter_by(
            name=injury_data["player_name"],
            team_id=injury_data["team_id"],
        ).first()

        if not player:
            player = Player(
                name=injury_data["player_name"],
                team_id=injury_data["team_id"],
                position=injury_data.get("position", ""),
                transfermarkt_id=injury_data.get("transfermarkt_id", ""),
            )
            session.add(player)
            session.flush()

        # Check for existing injury
        existing = session.query(Injury).filter_by(
            player_id=player.id,
            injury_type=injury_data["injury_type"],
        ).first()

        if existing:
            existing.status = injury_data["status"]
            existing.expected_return = injury_data.get("expected_return")
            existing.updated_at = datetime.utcnow()
        else:
            injury = Injury(
                player_id=player.id,
                team_id=injury_data["team_id"],
                injury_type=injury_data["injury_type"],
                start_date=injury_data.get("start_date"),
                expected_return=injury_data.get("expected_return"),
                status=injury_data["status"],
                source=injury_data.get("source", "transfermarkt"),
            )
            session.add(injury)

    async def get_team_injuries(self, team_id: int) -> List[Injury]:
        """Get current injuries for a team from the database."""
        db = get_db()
        with db.get_session() as session:
            return session.query(Injury).filter_by(
                team_id=team_id,
                status="out",
            ).all()

    async def get_injury_summary(self, team_id: int) -> dict:
        """Get an injury impact summary for a team."""
        db = get_db()
        with db.get_session() as session:
            injuries = session.query(Injury).join(Player).filter(
                Injury.team_id == team_id,
                Injury.status.in_(["out", "doubtful"]),
            ).all()

            key_players_out = sum(1 for i in injuries if i.player and i.player.is_key_player)
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
                        "expected_return": str(i.expected_return) if i.expected_return else "Unknown",
                    }
                    for i in injuries
                ],
            }
