"""Daily summary dashboard generator."""

from datetime import date, datetime
from typing import List

from src.agent.betting_agent import MatchAnalysis
from src.betting.bankroll_manager import BankrollManager
from src.reporting.match_report import MatchReportGenerator
from src.utils.logger import get_logger

logger = get_logger()


class DashboardGenerator:
    """Generates daily summary dashboards with predictions, picks, and performance."""

    def __init__(self):
        self.report_gen = MatchReportGenerator()

    def generate_daily_dashboard(self, analyses: List[MatchAnalysis],
                                  bankroll: BankrollManager,
                                  target_date: date = None) -> str:
        """Generate a complete daily dashboard.

        Args:
            analyses: List of MatchAnalysis for today's fixtures
            bankroll: BankrollManager for performance data
            target_date: Date for the dashboard

        Returns:
            Formatted markdown dashboard string
        """
        target = target_date or date.today()

        sections = [
            self._dashboard_header(target),
            self._fixtures_summary(analyses),
            self._top_picks(analyses),
            self._performance_summary(bankroll),
        ]

        return "\n\n---\n\n".join(sections)

    def _dashboard_header(self, target_date: date) -> str:
        return (
            f"# Daily Betting Dashboard\n"
            f"**Date:** {target_date.strftime('%A, %B %d, %Y')}\n"
            f"**Generated:** {datetime.now().strftime('%H:%M:%S')}"
        )

    def _fixtures_summary(self, analyses: List[MatchAnalysis]) -> str:
        if not analyses:
            return "## Today's Fixtures\nNo fixtures for today."

        lines = [
            "## Today's Fixtures",
            f"**{len(analyses)} matches analyzed**\n",
            "| Match | League | Prediction | xG |",
            "|-------|--------|------------|-----|",
        ]

        for a in analyses:
            ens = a.predictions.get("ensemble", {})
            hw = ens.get("home_win", 0)
            d = ens.get("draw", 0)
            aw = ens.get("away_win", 0)

            # Determine top prediction
            if hw >= d and hw >= aw:
                pred = f"Home ({hw:.0%})"
            elif aw >= hw and aw >= d:
                pred = f"Away ({aw:.0%})"
            else:
                pred = f"Draw ({d:.0%})"

            xg = f"{ens.get('home_xg', 0):.1f}-{ens.get('away_xg', 0):.1f}"
            lines.append(f"| {a.match_name} | {a.league} | {pred} | {xg} |")

        return "\n".join(lines)

    def _top_picks(self, analyses: List[MatchAnalysis]) -> str:
        # Gather all recommendations
        all_recs = []
        for a in analyses:
            all_recs.extend(a.recommendations)

        all_recs.sort(key=lambda r: r.expected_value * r.confidence, reverse=True)
        top = all_recs[:10]

        if not top:
            return "## Top Value Picks\nNo value bets identified today."

        lines = [
            "## Top Value Picks",
            f"**{len(top)} value bets found**\n",
            "| # | Match | Bet | Odds | EV | Confidence | Risk | Stake |",
            "|---|-------|-----|------|-----|------------|------|-------|",
        ]

        for i, rec in enumerate(top, 1):
            lines.append(
                f"| {i} | {rec.match} | {rec.selection} | {rec.odds:.2f} | "
                f"{rec.expected_value:.1%} | {rec.confidence:.0%} | "
                f"{rec.risk_level} | {rec.kelly_stake_percentage:.1f}% |"
            )

        return "\n".join(lines)

    def _performance_summary(self, bankroll: BankrollManager) -> str:
        perf = bankroll.get_performance()

        return (
            f"## Performance Tracker\n"
            f"- **Bankroll:** {perf['current_bankroll']:.2f}\n"
            f"- **Total Bets:** {perf['total_bets']}\n"
            f"- **Hit Rate:** {perf.get('hit_rate', 0):.1%}\n"
            f"- **ROI:** {perf.get('roi', 0):.1%}\n"
            f"- **Yield:** {perf.get('yield_pct', 0):.1%}\n"
            f"- **Total Profit:** {perf.get('total_profit', 0):.2f}"
        )
