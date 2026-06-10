"""LLM-powered match briefings (Claude Opus 4.8 + server-side web search).

Two modes:
  * Preview briefing  — runs the evening before / hours before kickoff.
    Rich pre-match intelligence (storyline, form, team news, betting angle)
    WITHOUT lineups, since confirmed XIs aren't published a day out.
  * Pre-match briefing — runs ~45 min before kickoff with confirmed starting
    XIs fetched from API-Football, plus a verification of the model's pick
    against the actual lineup.

Both fold in the betting-agent's OWN model output (ensemble probabilities, xG,
the value pick) so the briefing ends with how the live picture lines up against
the prediction — not a generic web preview.

The "online research" is done by Claude's server-side web_search tool, so this
runs fully automated inside CI with no scraping of news sites required.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone, date
from pathlib import Path

from src.data.models import Match, Team
from src.utils.logger import get_logger

logger = get_logger()

# Per-day record of (kind, match_id) already briefed, so the every-15-min
# pre-match workflow (and any double daily run) never double-posts. Cached
# across CI runs via actions/cache (see prematch-briefings.yml).
_SENT_PATH = Path("data/briefings_sent.json")

# International tournaments this briefing applies to. Imported lazily elsewhere,
# duplicated here as a module constant to avoid a hard dependency at import time.
_WC_LEAGUES = {"world/fifa-world-cup"}

_BRIEFING_MODEL = "claude-opus-4-8"

# Telegram HTML supports a small tag subset. Instruct the model accordingly.
_HTML_RULES = (
    "Format for Telegram using ONLY these HTML tags: <b>bold</b> and <i>italic</i>. "
    "Do NOT use Markdown (** or ##), tables, or any other HTML tags. "
    "Use emojis as section markers. Keep the whole briefing under ~3500 characters."
)


def _kyiv(dt: datetime) -> str:
    """Format a naive-UTC datetime as Kyiv local HH:MM."""
    if dt is None:
        return "TBD"
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo("Europe/Kiev")
    except Exception:
        tz = timezone(timedelta(hours=3))  # EEST fallback
    return dt.replace(tzinfo=timezone.utc).astimezone(tz).strftime("%H:%M")


class MatchBriefingService:
    """Generates and posts LLM match briefings. Reuses the agent's predictor,
    scrapers, DB and Telegram notifier."""

    def __init__(self, agent):
        self.agent = agent
        self.config = agent.config
        self.db = agent.db
        self.telegram = agent.telegram

    # ---- public entry points -------------------------------------------------

    async def run_preview_briefings(self) -> int:
        """Briefing for every upcoming WC fixture today (no lineups). Returns count sent."""
        if not self._enabled():
            return 0
        fixtures = self._wc_fixtures(window_hours=30, min_minutes_ahead=0)
        if not fixtures:
            logger.info("Briefings: no upcoming WC fixtures for preview")
            return 0
        logger.info(f"Briefings: generating {len(fixtures)} preview briefing(s)")
        sent = 0
        for match_id in fixtures:
            try:
                if await self._briefing_for(match_id, lineup_aware=False):
                    sent += 1
            except Exception as e:
                logger.warning(f"Preview briefing failed for match {match_id}: {e}")
        return sent

    async def run_prematch_briefings(self, window_min: int = 45,
                                     slack_min: int = 12) -> int:
        """Lineup-aware briefing for WC fixtures kicking off in roughly
        [window_min - slack_min, window_min + slack_min] minutes. Designed to be
        fired by a workflow that runs every ~15 min. Returns count sent."""
        if not self._enabled():
            return 0
        lo = max(0, window_min - slack_min)
        hi = window_min + slack_min
        fixtures = self._wc_fixtures(
            window_hours=2, min_minutes_ahead=lo, max_minutes_ahead=hi
        )
        if not fixtures:
            logger.info(
                f"Briefings: no WC fixtures kicking off in {lo}-{hi} min window"
            )
            return 0
        logger.info(f"Briefings: generating {len(fixtures)} pre-match (T-{window_min}) briefing(s)")
        sent = 0
        for match_id in fixtures:
            try:
                if await self._briefing_for(match_id, lineup_aware=True):
                    sent += 1
            except Exception as e:
                logger.warning(f"Pre-match briefing failed for match {match_id}: {e}")
        return sent

    # ---- idempotency guard ---------------------------------------------------

    def _load_sent(self) -> dict:
        """Return {today_iso: [keys]} keeping only today's entries."""
        today = date.today().isoformat()
        try:
            data = json.loads(_SENT_PATH.read_text())
            if isinstance(data, dict) and today in data:
                return {today: list(data[today])}
        except Exception:
            pass
        return {today: []}

    def _already_sent(self, kind: str, match_id: int) -> bool:
        today = date.today().isoformat()
        return f"{kind}:{match_id}" in self._load_sent().get(today, [])

    def _mark_sent(self, kind: str, match_id: int) -> None:
        today = date.today().isoformat()
        state = self._load_sent()
        key = f"{kind}:{match_id}"
        if key not in state[today]:
            state[today].append(key)
        try:
            _SENT_PATH.parent.mkdir(parents=True, exist_ok=True)
            _SENT_PATH.write_text(json.dumps(state, indent=2))
        except Exception as e:
            logger.debug(f"Could not persist briefing-sent marker: {e}")

    # ---- internals -----------------------------------------------------------

    def _enabled(self) -> bool:
        if not self.config.get("briefings.enabled", True):
            logger.info("Briefings disabled via config")
            return False
        if not os.environ.get("ANTHROPIC_API_KEY"):
            logger.warning("Briefings skipped: ANTHROPIC_API_KEY not set")
            return False
        try:
            import anthropic  # noqa: F401
        except ImportError:
            logger.warning("Briefings skipped: anthropic package not installed")
            return False
        return True

    def _wc_fixtures(self, window_hours: int, min_minutes_ahead: int = 0,
                     max_minutes_ahead: int = None) -> list:
        """Return WC fixture match IDs kicking off within the given window."""
        now = datetime.utcnow()
        start = now + timedelta(minutes=min_minutes_ahead)
        end = (now + timedelta(minutes=max_minutes_ahead)
               if max_minutes_ahead is not None
               else now + timedelta(hours=window_hours))
        with self.db.get_session() as session:
            rows = session.query(Match.id).filter(
                Match.is_fixture == True,  # noqa: E712
                Match.league.in_(_WC_LEAGUES),
                Match.match_date >= start,
                Match.match_date <= end,
            ).order_by(Match.match_date.asc()).all()
        return [r[0] for r in rows]

    async def _briefing_for(self, match_id: int, lineup_aware: bool) -> bool:
        """Build the dossier, optionally fetch lineups, ask Claude, post to Telegram."""
        kind = "prematch" if lineup_aware else "preview"
        if self._already_sent(kind, match_id):
            logger.info(f"Briefing already sent today [{kind}] for match {match_id} — skipping")
            return False

        analysis = await self.agent.analyze_fixture(match_id)
        if not analysis or not analysis.predictions:
            logger.info(f"Briefing skipped for match {match_id}: no prediction (low coverage)")
            return False

        # Round/group + API-Football fixture id straight from the row.
        with self.db.get_session() as session:
            m = session.get(Match, match_id)
            round_name = (m.round or "") if m else ""
            apifootball_id = m.apifootball_id if m else None

        lineups = {}
        if lineup_aware and apifootball_id:
            try:
                lineups = await self.agent.apifootball.fetch_lineups(apifootball_id)
            except Exception as e:
                logger.debug(f"Lineup fetch failed for fixture {apifootball_id}: {e}")
        if lineup_aware and not lineups:
            logger.info(
                f"Lineups not yet published for {analysis.match_name} — "
                f"posting briefing with team-news context instead"
            )

        dossier = self._build_dossier(analysis, round_name, lineups)
        briefing = await self._research_and_write(
            analysis.match_name, dossier, lineup_aware=lineup_aware,
            has_lineups=bool(lineups),
        )
        if not briefing:
            return False

        header = self._header(analysis, round_name, lineups)
        await self.telegram._send_chunked(f"{header}\n\n{briefing}")
        self._mark_sent(kind, match_id)
        label = "PRE-MATCH (lineups)" if lineups else (
            "PRE-MATCH" if lineup_aware else "PREVIEW")
        logger.info(f"Briefing sent [{label}]: {analysis.match_name}")
        return True

    def _header(self, analysis, round_name: str, lineups: dict) -> str:
        ko = _kyiv(analysis.match_date)
        tag = "🟢 LINEUPS CONFIRMED" if lineups else "📋 PREVIEW"
        rnd = f" · {round_name}" if round_name else ""
        return (
            f"<b>🏆 World Cup Briefing — {analysis.match_name}</b>\n"
            f"<i>{tag}{rnd} · ⏰ {ko} Kyiv</i>"
        )

    def _build_dossier(self, analysis, round_name: str, lineups: dict) -> str:
        """Deterministic data block from our own model + DB. Fed to Claude as ground truth."""
        ens = analysis.predictions.get("ensemble", {}) or {}
        f = analysis.features or {}
        lines = [f"MATCH: {analysis.match_name}"]
        if round_name:
            lines.append(f"STAGE: {round_name}")
        lines.append(f"KICKOFF (Kyiv): {_kyiv(analysis.match_date)}")

        # Model prediction
        lines.append("")
        lines.append("OUR MODEL (ensemble Poisson+Elo+ML):")
        lines.append(
            f"  1X2: Home {ens.get('home_win', 0):.0%} / "
            f"Draw {ens.get('draw', 0):.0%} / Away {ens.get('away_win', 0):.0%}"
        )
        lines.append(
            f"  xG: {ens.get('home_xg', 0):.2f} - {ens.get('away_xg', 0):.2f} | "
            f"most likely score {ens.get('most_likely_score', 'N/A')}"
        )
        lines.append(
            f"  Over 2.5: {ens.get('over_2.5', 0):.0%} | BTTS: {ens.get('btts_yes', 0):.0%}"
        )

        # WC group standings context (from feature engineer section 15)
        if f.get("wc_is_group_stage"):
            lines.append(
                f"  Group form so far — "
                f"home {int(f.get('wc_home_points', 0))}pts (GD {int(f.get('wc_home_gd', 0)):+d}), "
                f"away {int(f.get('wc_away_points', 0))}pts (GD {int(f.get('wc_away_gd', 0)):+d})"
            )

        # Value picks
        recs = analysis.recommendations or []
        lines.append("")
        if recs:
            lines.append("OUR VALUE PICK(S):")
            for r in recs[:4]:
                agr = f", {r.model_agreement}" if getattr(r, "model_agreement", None) else ""
                lines.append(
                    f"  {r.selection} @ {r.odds:.2f} "
                    f"(EV {r.expected_value:+.0%}, conf {r.confidence:.0%}{agr})"
                )
        else:
            lines.append("OUR VALUE PICK(S): none — no edge vs market on this match.")

        # Injuries (our scraped data). get_injury_summary returns an `injuries`
        # list of dicts; key_players_out is a count, not names.
        inj = analysis.injury_report or {}
        for side, label in (("home", "Home"), ("away", "Away")):
            data = inj.get(side) or {}
            entries = data.get("injuries") or []
            out = [
                e for e in entries
                if isinstance(e, dict) and e.get("status") == "out"
            ]
            if out:
                names = ", ".join(
                    f"{e.get('player', '?')} ({e.get('position') or e.get('status')})"
                    for e in out[:6]
                )
                lines.append(f"INJURIES/OUT ({label}): {names}")

        # Confirmed lineups (pre-match only)
        if lineups:
            lines.append("")
            lines.append("CONFIRMED STARTING XIs:")
            for side in ("home", "away"):
                blk = lineups.get(side)
                if not blk:
                    continue
                xi = blk.get("start_xi") or []
                names = ", ".join(
                    p["name"] for p in xi if p.get("name")
                ) or "(not parsed)"
                form = f" [{blk['formation']}]" if blk.get("formation") else ""
                coach = f" — coach {blk['coach']}" if blk.get("coach") else ""
                lines.append(f"  {blk.get('team', side)}{form}{coach}: {names}")

        return "\n".join(lines)

    async def _research_and_write(self, match_name: str, dossier: str,
                                  lineup_aware: bool, has_lineups: bool) -> str:
        """Call Claude Opus 4.8 with server-side web search; return Telegram-HTML briefing."""
        from anthropic import AsyncAnthropic

        client = AsyncAnthropic()  # reads ANTHROPIC_API_KEY from env

        if has_lineups:
            task = (
                "Confirmed starting XIs are included in the dossier. Search the web for "
                "the latest team news, tactical context, and any last-minute changes, then "
                "write a pre-match briefing. Explicitly assess how the confirmed lineups "
                "affect OUR model's pick: are key attackers/defenders starting or benched? "
                "Does the XI support or undercut the pick? End with a clear VERDICT line."
            )
        elif lineup_aware:
            task = (
                "Lineups are not published yet. Search for the latest probable XIs, team "
                "news and injuries, then write a pre-match briefing and assess how it lines "
                "up against OUR model's pick. End with a clear VERDICT line."
            )
        else:
            task = (
                "Write a rich pre-match PREVIEW (the match is hours away, so no confirmed "
                "lineups exist — do not invent any). Search the web for the storyline, recent "
                "form, head-to-head, team news/injuries, and the betting market angle. Close "
                "by relating the picture to OUR model's prediction and pick."
            )

        system = (
            "You are an expert football analyst writing a concise, high-signal World Cup "
            "match briefing for a Telegram betting channel. You have access to a web search "
            "tool — use it to gather current, factual information (storyline, recent form, "
            "head-to-head, team news, injuries, betting market). Never invent lineups, "
            "injuries, or quotes; if something is uncertain, say so. Be specific and cite "
            "what you found. " + _HTML_RULES
        )

        user = (
            f"{task}\n\n"
            f"Here is our internal model dossier for {match_name} (treat these numbers as "
            f"ground truth from our prediction system — research everything else):\n\n"
            f"{dossier}\n\n"
            "Structure the briefing with emoji section headers, e.g.: 📜 Storyline, "
            "📈 Form, 🩹 Team news, 🎯 Model & market, 🔮 Verdict. End with a one-line "
            "'Sources:' note listing the main outlets you used."
        )

        tools = [{"type": "web_search_20260209", "name": "web_search"}]

        messages = [{"role": "user", "content": user}]
        text_out = ""
        # Server-side web search runs a sampling loop; it may return pause_turn if it
        # hits the iteration cap. Re-send to resume, with a safety bound.
        for _ in range(6):
            try:
                resp = await client.messages.create(
                    model=_BRIEFING_MODEL,
                    max_tokens=4096,
                    thinking={"type": "adaptive"},
                    system=system,
                    tools=tools,
                    messages=messages,
                )
            except Exception as e:
                logger.warning(f"Claude briefing call failed for {match_name}: {e}")
                return ""

            if resp.stop_reason == "pause_turn":
                messages = [
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": resp.content},
                ]
                continue

            text_out = "".join(
                b.text for b in resp.content if getattr(b, "type", "") == "text"
            ).strip()
            break

        return text_out
