"""Match analysis report generator."""

from datetime import datetime
from typing import Dict, List

from src.betting.value_calculator import BetRecommendation
from src.utils.logger import get_logger

logger = get_logger()


class MatchReportGenerator:
    """Generates formatted match analysis reports in markdown."""

    def generate_report(self, analysis) -> str:
        """Generate a full match analysis report.

        Args:
            analysis: MatchAnalysis dataclass instance

        Returns:
            Formatted markdown report string
        """
        sections = [
            self._header(analysis),
            self._team_form(analysis),
            self._h2h_section(analysis),
            self._injury_report(analysis),
            self._news_sentiment(analysis),
            self._predictions(analysis),
            self._value_bets(analysis),
        ]

        return "\n\n".join(sections)

    def _header(self, analysis) -> str:
        match_date = analysis.match_date
        if isinstance(match_date, datetime):
            date_str = match_date.strftime("%Y-%m-%d %H:%M")
        else:
            date_str = str(match_date)

        return (
            f"# Match Analysis: {analysis.match_name}\n"
            f"**Date:** {date_str} | **League:** {analysis.league}"
        )

    def _team_form(self, analysis) -> str:
        f = analysis.features
        home_form = f.get("home_overall_form_string", "N/A")
        away_form = f.get("away_overall_form_string", "N/A")
        home_pts = f.get("home_overall_points", 0)
        away_pts = f.get("away_overall_points", 0)
        home_gs = f.get("home_overall_goals_scored", 0)
        home_gc = f.get("home_overall_goals_conceded", 0)
        away_gs = f.get("away_overall_goals_scored", 0)
        away_gc = f.get("away_overall_goals_conceded", 0)
        home_pos = f.get("home_league_league_position", "?")
        away_pos = f.get("away_league_league_position", "?")

        # Extract team names from match_name
        parts = analysis.match_name.split(" vs ")
        home_name = parts[0] if len(parts) > 0 else "Home"
        away_name = parts[1] if len(parts) > 1 else "Away"

        return (
            f"## Team Form\n"
            f"### {home_name} (Form: {home_form})\n"
            f"- Last 5: {home_pts} pts, {home_gs} scored, {home_gc} conceded\n"
            f"- League position: {home_pos}\n\n"
            f"### {away_name} (Form: {away_form})\n"
            f"- Last 5: {away_pts} pts, {away_gs} scored, {away_gc} conceded\n"
            f"- League position: {away_pos}"
        )

    def _h2h_section(self, analysis) -> str:
        f = analysis.features
        meetings = f.get("h2h_total_meetings", 0)
        if meetings == 0:
            return "## Head-to-Head\nNo previous meetings on record."

        home_w = f.get("h2h_home_wins", 0)
        draws = f.get("h2h_draws", 0)
        away_w = f.get("h2h_away_wins", 0)
        avg_goals = f.get("h2h_avg_total_goals", 0)
        btts = f.get("h2h_btts_percentage", 0)
        over25 = f.get("h2h_over_25_percentage", 0)

        return (
            f"## Head-to-Head (Last {meetings} meetings)\n"
            f"- Home wins: {home_w} | Draws: {draws} | Away wins: {away_w}\n"
            f"- Avg goals: {avg_goals:.1f} per game\n"
            f"- BTTS: {btts:.0%} | Over 2.5: {over25:.0%}"
        )

    def _injury_report(self, analysis) -> str:
        report = analysis.injury_report
        parts = analysis.match_name.split(" vs ")
        home_name = parts[0] if len(parts) > 0 else "Home"
        away_name = parts[1] if len(parts) > 1 else "Away"

        lines = ["## Injury Report"]

        for label, key in [(home_name, "home"), (away_name, "away")]:
            data = report.get(key, {})
            injuries = data.get("injuries", [])
            lines.append(f"### {label}")
            if not injuries:
                lines.append("- No injuries reported")
            else:
                for inj in injuries:
                    status_icon = "X" if inj["status"] == "out" else "?"
                    lines.append(
                        f"- [{status_icon}] {inj['player']} ({inj['type']}) "
                        f"- {inj.get('position', 'N/A')} "
                        f"- Return: {inj.get('expected_return', 'Unknown')}"
                    )

        return "\n".join(lines)

    def _news_sentiment(self, analysis) -> str:
        news = analysis.news_summary
        home_s = news.get("home", {})
        away_s = news.get("away", {})

        return (
            f"## News & Sentiment\n"
            f"- Home team: {home_s.get('trend', 'neutral')} "
            f"(score: {home_s.get('avg_sentiment', 0):.2f}, "
            f"{home_s.get('article_count', 0)} articles)\n"
            f"- Away team: {away_s.get('trend', 'neutral')} "
            f"(score: {away_s.get('avg_sentiment', 0):.2f}, "
            f"{away_s.get('article_count', 0)} articles)"
        )

    def _predictions(self, analysis) -> str:
        ens = analysis.predictions.get("ensemble", {})

        rows = [
            "## Predictions",
            "| Market | Prediction | Probability |",
            "|--------|------------|-------------|",
            f"| 1X2 | Home Win | {ens.get('home_win', 0):.1%} |",
            f"| 1X2 | Draw | {ens.get('draw', 0):.1%} |",
            f"| 1X2 | Away Win | {ens.get('away_win', 0):.1%} |",
            f"| Over/Under | Over 2.5 | {ens.get('over_2.5', 0):.1%} |",
            f"| Over/Under | Under 2.5 | {ens.get('under_2.5', 0):.1%} |",
            f"| BTTS | Yes | {ens.get('btts_yes', 0):.1%} |",
            f"| BTTS | No | {ens.get('btts_no', 0):.1%} |",
            "",
            f"**Expected Goals:** {ens.get('home_xg', 0):.2f} - {ens.get('away_xg', 0):.2f}",
            f"**Most Likely Score:** {ens.get('most_likely_score', 'N/A')}",
        ]

        return "\n".join(rows)

    def _value_bets(self, analysis) -> str:
        recs = analysis.recommendations
        if not recs:
            return "## Value Bet Recommendations\nNo value bets found for this match."

        lines = ["## Value Bet Recommendations"]
        stars_map = {"low": "***", "medium": "**", "high": "*"}

        for i, rec in enumerate(recs, 1):
            stars = stars_map.get(rec.risk_level, "*")
            lines.append(
                f"\n{i}. **{rec.selection} @ {rec.odds}** {stars}\n"
                f"   - EV: {rec.expected_value:.1%}\n"
                f"   - Confidence: {rec.confidence:.1%}\n"
                f"   - Stake: {rec.kelly_stake_percentage:.1f}% of bankroll\n"
                f"   - Risk: {rec.risk_level}\n"
                f"   - {rec.reasoning}"
            )

        return "\n".join(lines)
