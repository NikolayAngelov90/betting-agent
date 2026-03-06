"""Bankroll management and performance tracking."""

from dataclasses import dataclass
from datetime import datetime
from src.utils.logger import utcnow
from typing import List, Optional

from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger()


@dataclass
class BetRecord:
    """Record of a placed bet for tracking."""
    match: str
    market: str
    selection: str
    odds: float
    stake: float
    predicted_probability: float
    expected_value: float
    result: Optional[str] = None   # 'win', 'loss', 'void', None (pending)
    profit: float = 0.0
    placed_at: datetime = None

    def __post_init__(self):
        if self.placed_at is None:
            self.placed_at = utcnow()


class BankrollManager:
    """Manages bankroll, tracks bets, and calculates performance metrics."""

    def __init__(self, initial_bankroll: float = 1000.0, config=None):
        self.config = config or get_config()
        self.initial_bankroll = initial_bankroll
        self.current_bankroll = initial_bankroll
        self.bets: List[BetRecord] = []
        self.max_stake_pct = self.config.get("betting.max_stake_percentage", 5.0)

    def calculate_stake(self, kelly_pct: float) -> float:
        """Calculate actual stake amount from Kelly percentage.

        Args:
            kelly_pct: Kelly criterion stake percentage

        Returns:
            Stake amount in currency units
        """
        pct = min(kelly_pct, self.max_stake_pct)
        stake = self.current_bankroll * (pct / 100.0)
        return round(stake, 2)

    def place_bet(self, match: str, market: str, selection: str,
                  odds: float, stake: float, predicted_prob: float,
                  ev: float) -> BetRecord:
        """Record a placed bet."""
        bet = BetRecord(
            match=match, market=market, selection=selection,
            odds=odds, stake=stake,
            predicted_probability=predicted_prob,
            expected_value=ev,
        )
        self.bets.append(bet)
        self.current_bankroll -= stake
        logger.info(f"Bet placed: {selection} @ {odds} — stake {stake:.2f}")
        return bet

    def settle_bet(self, bet_index: int, result: str):
        """Settle a bet with its result.

        Args:
            bet_index: Index of the bet in self.bets
            result: 'win', 'loss', or 'void'
        """
        bet = self.bets[bet_index]
        bet.result = result

        if result == "win":
            profit = bet.stake * (bet.odds - 1)
            bet.profit = profit
            self.current_bankroll += bet.stake + profit
        elif result == "void":
            bet.profit = 0
            self.current_bankroll += bet.stake
        else:
            bet.profit = -bet.stake

        logger.info(f"Bet settled: {bet.selection} — {result}, profit={bet.profit:.2f}")

    def get_performance(self) -> dict:
        """Calculate overall performance metrics."""
        settled = [b for b in self.bets if b.result is not None]
        if not settled:
            return {
                "total_bets": 0, "roi": 0, "yield_pct": 0,
                "hit_rate": 0, "profit": 0, "current_bankroll": self.current_bankroll,
            }

        total_staked = sum(b.stake for b in settled)
        total_profit = sum(b.profit for b in settled)
        wins = sum(1 for b in settled if b.result == "win")

        return {
            "total_bets": len(settled),
            "wins": wins,
            "losses": len(settled) - wins,
            "hit_rate": round(wins / len(settled), 4) if settled else 0,
            "total_staked": round(total_staked, 2),
            "total_profit": round(total_profit, 2),
            "roi": round(total_profit / self.initial_bankroll, 4) if self.initial_bankroll else 0,
            "yield_pct": round(total_profit / total_staked, 4) if total_staked else 0,
            "current_bankroll": round(self.current_bankroll, 2),
            "peak_bankroll": round(max(
                self.initial_bankroll,
                self.current_bankroll,
            ), 2),
        }
