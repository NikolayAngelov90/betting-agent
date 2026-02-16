"""Telegram notification bot for betting picks."""

import asyncio
from typing import List, Dict
from itertools import groupby

from src.betting.value_calculator import BetRecommendation
from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger()


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

        # Add stats summary if available
        if stats:
            parts = []
            for period, label in [("yesterday", "Yesterday"), ("last_7_days", "7d"), ("all_time", "All time")]:
                s = stats.get(period, {})
                if s.get("total", 0) > 0:
                    parts.append(f"{label}: {s['wins']}/{s['total']} ({s['win_rate']:.0%})")
            if parts:
                header += f"\n📊 <i>{' | '.join(parts)}</i>\n"

        # Group picks by match
        picks_by_match: Dict[str, List[BetRecommendation]] = {}
        for pick in picks:
            picks_by_match.setdefault(pick.match, []).append(pick)

        lines = [header]
        pick_num = 0
        for match_name, match_picks in picks_by_match.items():
            lines.append(f"\n<b>{match_name}</b>")

            for pick in match_picks:
                pick_num += 1
                risk_emoji = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(pick.risk_level, "⚪")

                lines.append(
                    f"  {risk_emoji} <b>{pick.selection}</b> @ {pick.odds:.2f}\n"
                    f"      EV: {pick.expected_value:.1%} | "
                    f"Conf: {pick.confidence:.0%} | "
                    f"Stake: {pick.kelly_stake_percentage:.1f}%"
                )

        # Split into multiple messages if too long (Telegram limit ~4096 chars)
        message = "\n".join(lines)
        if len(message) > 4000:
            # Send in chunks per match group
            chunk = header
            for match_name, match_picks in picks_by_match.items():
                block = f"\n<b>{match_name}</b>\n"
                for pick in match_picks:
                    risk_emoji = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(pick.risk_level, "⚪")
                    block += (
                        f"  {risk_emoji} <b>{pick.selection}</b> @ {pick.odds:.2f}\n"
                        f"      EV: {pick.expected_value:.1%} | "
                        f"Conf: {pick.confidence:.0%} | "
                        f"Stake: {pick.kelly_stake_percentage:.1f}%\n"
                    )

                if len(chunk) + len(block) > 3800:
                    await self._send_message(chunk)
                    chunk = ""
                chunk += block

            if chunk.strip():
                await self._send_message(chunk)
        else:
            await self._send_message(message)

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

        # Calculate profit/loss
        profit = sum(
            p["stake"] * (p["odds"] - 1) if p["result"] == "win" else -p["stake"]
            for p in picks_to_show
        )

        label = yesterday.strftime('%d %b %Y') if yesterday_picks else "Recent"
        header = f"<b>📊 Settlement Report - {label}</b>\n"
        header += f"<b>Record: {wins}W - {losses}L ({win_rate:.0%})</b>\n"
        profit_emoji = "📈" if profit >= 0 else "📉"
        header += f"{profit_emoji} P/L: {profit:+.1f}% of bankroll\n"

        lines = [header]
        for pick in picks_to_show:
            result_emoji = "✅" if pick["result"] == "win" else "❌"
            lines.append(
                f"\n{result_emoji} <b>{pick['match_name']}</b> ({pick['score']})\n"
                f"    {pick['selection']} @ {pick['odds']:.2f} | Stake: {pick['stake']:.1f}%"
            )

        # Add overall stats
        if stats:
            lines.append("\n<b>─── Overall Stats ───</b>")
            for period, label in [("last_7_days", "Last 7 days"), ("last_30_days", "Last 30 days"), ("all_time", "All time")]:
                s = stats.get(period, {})
                if s.get("total", 0) > 0:
                    roi_emoji = "📈" if s.get("roi", 0) >= 0 else "📉"
                    lines.append(
                        f"{label}: {s['wins']}W-{s['losses']}L ({s['win_rate']:.0%}) "
                        f"{roi_emoji} ROI: {s['roi']:.1%}"
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
