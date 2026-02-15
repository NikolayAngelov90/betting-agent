"""Injury impact feature calculations."""

from src.data.models import Injury, Player, Team
from src.data.database import get_db
from src.utils.logger import get_logger

logger = get_logger()

# Position groups for impact scoring
POSITION_GROUPS = {
    "Goalkeeper": "goalkeeper",
    "Centre-Back": "defense", "Left-Back": "defense", "Right-Back": "defense",
    "Defender": "defense",
    "Defensive Midfield": "midfield", "Central Midfield": "midfield",
    "Attacking Midfield": "midfield", "Midfielder": "midfield",
    "Left Midfield": "midfield", "Right Midfield": "midfield",
    "Left Winger": "attack", "Right Winger": "attack",
    "Centre-Forward": "attack", "Second Striker": "attack",
    "Forward": "attack", "Striker": "attack",
}


class InjuryFeatures:
    """Calculates injury impact features for a team."""

    def __init__(self):
        self.db = get_db()

    def get_injury_features(self, team_id: int) -> dict:
        """Calculate injury impact features for a team.

        Args:
            team_id: Team database ID

        Returns:
            Dictionary of injury impact features
        """
        with self.db.get_session() as session:
            # Get current injuries with player info
            injuries = session.query(Injury).join(Player).filter(
                Injury.team_id == team_id,
                Injury.status.in_(["out", "doubtful"]),
            ).all()

            # Get total squad info
            squad = session.query(Player).filter_by(team_id=team_id).all()

            if not squad:
                return self._empty_injury_features()

            total_squad_value = sum(p.market_value or 0 for p in squad)
            injured_value = sum(
                (i.player.market_value or 0) for i in injuries if i.player
            )

            key_players_injured = sum(
                1 for i in injuries if i.player and i.player.is_key_player
            )

            # Position-based analysis
            injured_positions = [
                POSITION_GROUPS.get(i.player.position, "unknown")
                for i in injuries if i.player and i.player.position
            ]

            gk_available = not any(p == "goalkeeper" for p in injured_positions)
            defenders_out = sum(1 for p in injured_positions if p == "defense")
            midfielders_out = sum(1 for p in injured_positions if p == "midfield")
            attackers_out = sum(1 for p in injured_positions if p == "attack")

            # Defensive stability: 1.0 = full strength, decreases with injuries
            total_defenders = sum(
                1 for p in squad
                if POSITION_GROUPS.get(p.position, "") == "defense"
            )
            defensive_stability = 1.0
            if total_defenders > 0:
                defensive_stability = max(0.0, 1.0 - (defenders_out / total_defenders) * 0.8)

            # Attacking threat: similar logic
            total_attackers = sum(
                1 for p in squad
                if POSITION_GROUPS.get(p.position, "") == "attack"
            )
            attacking_threat = 1.0
            if total_attackers > 0:
                attacking_threat = max(0.0, 1.0 - (attackers_out / total_attackers) * 0.8)

            return {
                "total_injured": len(injuries),
                "key_players_injured": key_players_injured,
                "injured_market_value": injured_value,
                "injured_value_percentage": round(
                    injured_value / total_squad_value, 3
                ) if total_squad_value else 0,
                "goalkeeper_available": gk_available,
                "defenders_out": defenders_out,
                "midfielders_out": midfielders_out,
                "attackers_out": attackers_out,
                "defensive_stability_score": round(defensive_stability, 3),
                "attacking_threat_score": round(attacking_threat, 3),
            }

    def _empty_injury_features(self) -> dict:
        return {
            "total_injured": 0,
            "key_players_injured": 0,
            "injured_market_value": 0,
            "injured_value_percentage": 0,
            "goalkeeper_available": True,
            "defenders_out": 0,
            "midfielders_out": 0,
            "attackers_out": 0,
            "defensive_stability_score": 1.0,
            "attacking_threat_score": 1.0,
        }
