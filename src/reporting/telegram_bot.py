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

    async def send_daily_picks(self, picks: List[BetRecommendation]):
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
