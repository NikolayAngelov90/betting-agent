"""Telegram notification bot for betting picks."""

import asyncio
from datetime import timezone, timedelta
from typing import List, Dict
from itertools import groupby

from src.betting.value_calculator import BetRecommendation
from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger()

# League display names for clean formatting
LEAGUE_DISPLAY = {
    "england/premier-league": "🏴 Premier League",
    "england/championship": "🏴 Championship",
    "spain/laliga": "🇪🇸 La Liga",
    "spain/laliga2": "🇪🇸 La Liga 2",
    "germany/bundesliga": "🇩🇪 Bundesliga",
    "germany/2-bundesliga": "🇩🇪 2. Bundesliga",
    "italy/serie-a": "🇮🇹 Serie A",
    "italy/serie-b": "🇮🇹 Serie B",
    "france/ligue-1": "🇫🇷 Ligue 1",
    "france/ligue-2": "🇫🇷 Ligue 2",
    "netherlands/eredivisie": "🇳🇱 Eredivisie",
    "portugal/primeira-liga": "🇵🇹 Primeira Liga",
    "belgium/jupiler-pro-league": "🇧🇪 Jupiler Pro",
    "turkey/super-lig": "🇹🇷 Super Lig",
    "scotland/premiership": "🏴 Scottish Prem",
    "austria/bundesliga": "🇦🇹 Bundesliga",
    "switzerland/super-league": "🇨🇭 Super League",
    "greece/super-league": "🇬🇷 Super League",
    "denmark/superliga": "🇩🇰 Superliga",
    "norway/eliteserien": "🇳🇴 Eliteserien",
    "sweden/allsvenskan": "🇸🇪 Allsvenskan",
    "finland/veikkausliiga": "🇫🇮 Veikkausliiga",
    "poland/ekstraklasa": "🇵🇱 Ekstraklasa",
    "romania/liga-1": "🇷🇴 Liga I",
    "europe/champions-league": "🏆 Champions League",
    "europe/europa-league": "🏆 Europa League",
    "europe/europa-conference-league": "🏆 Conference League",
}


class TelegramNotifier:
    """Sends betting picks and alerts via Telegram."""

    def __init__(self, config=None):
        self.config = config or get_config()
        notifications = self.config.notifications
        self.enabled = notifications.get("telegram_enabled", False)
        self.bot_token = notifications.get("telegram_bot_token", "")
        self.chat_id = notifications.get("telegram_chat_id", "")
        self._bot = None

    def _get_bot(self):
        """Lazy-load the Telegram bot."""
        if self._bot is None and self.enabled and self.bot_token:
            try:
                from telegram import Bot
                self._bot = Bot(token=self.bot_token)
                logger.info("Telegram bot initialized")
            except ImportError:
                logger.warning("python-telegram-bot not installed, run: pip install python-telegram-bot")
            except Exception as e:
                logger.error(f"Failed to initialize Telegram bot: {e}")
        return self._bot

    async def send_daily_picks(self, picks: List[BetRecommendation], stats: dict = None):
        """Send daily picks summary via Telegram with rich formatting."""
        if not self.enabled:
            return

        bot = self._get_bot()
        if not bot:
            return

        if not picks:
            await self._send_message("No value picks found for today.")
            return

        from datetime import date
        header = f"<b>Daily Value Picks - {date.today().strftime('%d %b %Y')}</b>\n"
        header += f"<i>{len(picks)} picks found</i>\n"

        # Add stats summary if available — always show all-time, add shorter periods
        # only when they differ (avoids showing the same number three times when the
        # agent is new and all data fits within a single day).
        if stats:
            parts = []
            all_time = stats.get("all_time", {})
            at_total = all_time.get("total", 0)
            if at_total > 0:
                parts.append(
                    f"All time ({at_total}): {all_time['wins']}W-{all_time['losses']}L "
                    f"({all_time['win_rate']:.0%})"
                )
            for period, label in [("last_7_days", "7d"), ("yesterday", "Yesterday")]:
                s = stats.get(period, {})
                s_total = s.get("total", 0)
                # Only add if this period has fewer picks than all_time (i.e. adds info)
                if s_total > 0 and s_total < at_total:
                    parts.append(f"{label}: {s['wins']}/{s_total} ({s['win_rate']:.0%})")
            if parts:
                header += f"\n📊 <i>{' | '.join(parts)}</i>\n"

        # Group picks by league, then by match
        picks_by_league: Dict[str, List[BetRecommendation]] = {}
        for pick in picks:
            league = pick.league or "Other"
            picks_by_league.setdefault(league, []).append(pick)

        lines = [header]
        pick_num = 0

        for league_key in sorted(picks_by_league.keys()):
            league_picks = picks_by_league[league_key]
            league_name = LEAGUE_DISPLAY.get(league_key, league_key)
            lines.append(f"\n<b>{league_name}</b>")

            # Group by match within league
            picks_by_match: Dict[str, List[BetRecommendation]] = {}
            for pick in league_picks:
                picks_by_match.setdefault(pick.match, []).append(pick)

            for match_name, match_picks in picks_by_match.items():
                for pick in match_picks:
                    pick_num += 1
                    risk_emoji = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(pick.risk_level, "⚪")
                    agreement_icon = _agreement_icon(pick.model_agreement)

                    # Kickoff time in Kyiv local time (UTC+2 winter / UTC+3 summer)
                    kickoff_str = ""
                    if pick.match_date:
                        try:
                            from zoneinfo import ZoneInfo
                            local_tz = ZoneInfo("Europe/Kiev")
                        except Exception:
                            local_tz = timezone(timedelta(hours=2))
                        local_dt = pick.match_date.replace(tzinfo=timezone.utc).astimezone(local_tz)
                        kickoff_str = f" ⏰ {local_dt.strftime('%H:%M')}"

                    # Pick header: number + match name + kickoff
                    line = f"\n  {risk_emoji} <b>Pick #{pick_num}: {match_name}</b>{kickoff_str}"

                    # xG
                    if pick.predicted_xg:
                        line += f"\n      xG: {pick.predicted_xg}"

                    # Bet
                    line += f"\n      Bet: <b>{pick.selection}</b> @ {pick.odds:.2f}"

                    # EV / Conf / Risk / Stake
                    line += (
                        f"\n      EV: {pick.expected_value:.1%} | "
                        f"Conf: {pick.confidence:.0%} | "
                        f"Risk: {pick.risk_level}"
                        f"\n      Stake: {pick.kelly_stake_percentage:.1f}% of bankroll"
                    )

                    # Model agreement
                    if pick.model_agreement:
                        agr_label = pick.model_agreement.upper()
                        line += f"\n      {agreement_icon} Models: {pick.model_agreement} [{agr_label}]"
                        if pick.models_for:
                            line += f" - {pick.models_for} agree"

                    # xG edge
                    if pick.xg_edge:
                        line += f"\n      📈 {pick.xg_edge}"

                    # Fallback odds warning
                    if pick.used_fallback_odds:
                        line += "\n      ⚠️ Estimated odds (no bookmaker data)"

                    # Reasoning (includes form strings, H2H, model rationale)
                    if pick.reasoning:
                        line += f"\n      <i>{pick.reasoning}</i>"

                    lines.append(line)

        # Send (split if needed)
        message = "\n".join(lines)
        await self._send_chunked(message, header)

    async def send_settlement_report(self, settled_picks: list, stats: dict = None):
        """Send settlement results for yesterday's picks via Telegram."""
        if not self.enabled or not settled_picks:
            return

        from datetime import date, timedelta
        yesterday = date.today() - timedelta(days=1)

        # Filter to yesterday's settled picks (or show all if dates vary)
        yesterday_picks = [p for p in settled_picks if p.get("pick_date") == yesterday]
        picks_to_show = yesterday_picks if yesterday_picks else settled_picks

        wins = sum(1 for p in picks_to_show if p["result"] == "win")
        losses = sum(1 for p in picks_to_show if p["result"] == "loss")
        total = wins + losses
        win_rate = wins / total if total > 0 else 0

        # Calculate profit/loss for the settled batch
        profit = sum(
            p["stake"] * (p["odds"] - 1) if p["result"] == "win" else -p["stake"]
            for p in picks_to_show
        )

        label = yesterday.strftime('%d %b %Y') if yesterday_picks else "Recent"
        header = f"<b>📊 Settlement Report - {label}</b>\n"
        # Batch result (just settled)
        profit_emoji = "📈" if profit >= 0 else "📉"
        header += f"Batch: {wins}W-{losses}L ({win_rate:.0%}) {profit_emoji} {profit:+.1f}%\n"
        # Running all-time totals from DB
        if stats:
            at = stats.get("all_time", {})
            at_total = at.get("total", 0)
            if at_total > 0:
                at_roi = at.get("roi", 0)
                at_emoji = "📈" if at_roi >= 0 else "📉"
                header += (
                    f"<b>All time ({at_total}): {at['wins']}W-{at['losses']}L "
                    f"({at['win_rate']:.0%}) {at_emoji} ROI: {at_roi:.1%}</b>\n"
                )

        lines = [header]

        # Group settled picks by league
        by_league: Dict[str, list] = {}
        for pick in picks_to_show:
            lg = pick.get("league", "Other") or "Other"
            by_league.setdefault(lg, []).append(pick)

        for league_key in sorted(by_league.keys()):
            league_name = LEAGUE_DISPLAY.get(league_key, league_key)
            league_picks = by_league[league_key]
            lg_wins = sum(1 for p in league_picks if p["result"] == "win")
            lg_total = len(league_picks)
            lines.append(f"\n<b>{league_name}</b> ({lg_wins}/{lg_total})")

            for pick in league_picks:
                result_emoji = "✅" if pick["result"] == "win" else "❌"
                score = pick.get("score", "?-?")
                xg_str = ""
                if pick.get("home_xg") is not None and pick.get("away_xg") is not None:
                    xg_str = f" | xG: {pick['home_xg']:.1f}-{pick['away_xg']:.1f}"

                lines.append(
                    f"  {result_emoji} <b>{pick['match_name']}</b> ({score}){xg_str}\n"
                    f"      {pick['selection']} @ {pick['odds']:.2f} | Stake: {pick['stake']:.1f}%"
                )

        # Add period breakdown (all-time already shown in header)
        if stats:
            at_total = stats.get("all_time", {}).get("total", 0)
            period_lines = []
            for period, slabel in [("last_30_days", "Last 30 days"), ("last_7_days", "Last 7 days")]:
                s = stats.get(period, {})
                s_total = s.get("total", 0)
                # Only show if period has fewer picks than all-time (adds meaningful info)
                if s_total > 0 and s_total < at_total:
                    roi_emoji = "📈" if s.get("roi", 0) >= 0 else "📉"
                    period_lines.append(
                        f"{slabel}: {s['wins']}W-{s['losses']}L ({s['win_rate']:.0%}) "
                        f"{roi_emoji} ROI: {s['roi']:.1%}"
                    )
            if period_lines:
                lines.append("\n<b>─── Period Breakdown ───</b>")
                lines.extend(period_lines)

            # Odds source breakdown
            odds_src = stats.get("odds_source", {})
            real = odds_src.get("real_odds", {})
            fb = odds_src.get("fallback_odds", {})
            if real.get("total", 0) and fb.get("total", 0):
                lines.append(
                    f"\n<i>Real odds: {real['win_rate']:.0%} win rate | "
                    f"Fallback: {fb['win_rate']:.0%} win rate</i>"
                )

            pending = stats.get("pending", 0)
            if pending > 0:
                lines.append(f"\n⏳ {pending} picks still pending")

        message = "\n".join(lines)
        await self._send_message(message)

    async def send_alert(self, text: str):
        """Send a generic alert message."""
        if not self.enabled:
            return
        await self._send_message(text)

    async def _send_chunked(self, message: str, header: str = ""):
        """Send a message, splitting into chunks if over Telegram's 4096 char limit."""
        if len(message) <= 4000:
            await self._send_message(message)
            return

        # Split by double newline (paragraph breaks)
        paragraphs = message.split("\n\n")
        chunk = ""
        for para in paragraphs:
            if len(chunk) + len(para) + 2 > 3800:
                if chunk.strip():
                    await self._send_message(chunk)
                chunk = para
            else:
                chunk = chunk + "\n\n" + para if chunk else para

        if chunk.strip():
            await self._send_message(chunk)

    async def _send_message(self, text: str):
        """Send a message via Telegram using HTML parse mode."""
        bot = self._get_bot()
        if not bot or not self.chat_id:
            logger.debug("Telegram not configured, skipping message")
            return

        try:
            await bot.send_message(
                chat_id=self.chat_id,
                text=text,
                parse_mode="HTML",
            )
            logger.info("Telegram message sent")
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")


def _agreement_icon(agreement: str) -> str:
    """Map model agreement level to an emoji."""
    return {
        "unanimous": "🎯",
        "majority": "✊",
        "split": "⚠️",
    }.get(agreement, "")
