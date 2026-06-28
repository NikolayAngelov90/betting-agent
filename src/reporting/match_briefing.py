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
import re
from datetime import datetime, timedelta, timezone, date
from pathlib import Path

from src.data.models import Match, Team, Odds, SavedPick
from src.utils.logger import get_logger

logger = get_logger()

# Machine-readable decision block the LLM appends after the (Bulgarian) briefing.
_DECISION_RE = re.compile(
    r"<<<DECISION>>>(.*?)<<<END>>>", re.DOTALL | re.IGNORECASE
)

# Telegram HTML allows only a small tag set; we permit <b> and <i> per _HTML_RULES.
_ALLOWED_TAG_RE = re.compile(r"</?(b|i)>", re.IGNORECASE)

# Switch-language used to detect prose that implies a pick change while the
# machine decision said KEEP (Bulgarian + English). Heuristic, for an audit warning.
_SWITCH_PROSE_RE = re.compile(
    r"\b(сменям|смяна|променям|преминавам|залагам вместо|вместо това залаг|"
    r"switch(?:ing)? to|chang(?:e|ing) (?:the|our|my) (?:pick|bet|selection)|"
    r"instead I'?d back|I'?ll back .* instead)\b",
    re.IGNORECASE,
)


def _sanitize_telegram_html(text: str) -> str:
    """Make LLM output safe for Telegram's HTML parse mode.

    Escapes &, <, > everywhere EXCEPT the allowed <b>/<i> tags. If the allowed
    tags are unbalanced (which Telegram rejects with a parse error, losing the
    whole message), strips all tags and returns plain escaped text instead.
    """
    def _esc(s: str) -> str:
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # Balance check on allowed tags — unbalanced HTML is rejected by Telegram.
    opens_b = len(re.findall(r"<b>", text, re.IGNORECASE))
    closes_b = len(re.findall(r"</b>", text, re.IGNORECASE))
    opens_i = len(re.findall(r"<i>", text, re.IGNORECASE))
    closes_i = len(re.findall(r"</i>", text, re.IGNORECASE))
    if opens_b != closes_b or opens_i != closes_i:
        return _esc(_ALLOWED_TAG_RE.sub("", text))

    parts = []
    last = 0
    for m in _ALLOWED_TAG_RE.finditer(text):
        parts.append(_esc(text[last:m.start()]))
        parts.append(m.group(0).lower())
        last = m.end()
    parts.append(_esc(text[last:]))
    return "".join(parts)

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
    """Format a naive-UTC datetime as Bulgarian local HH:MM (Europe/Sofia)."""
    if dt is None:
        return "TBD"
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo("Europe/Sofia")
    except Exception:
        tz = timezone(timedelta(hours=3))  # EEST fallback
    return dt.replace(tzinfo=timezone.utc).astimezone(tz).strftime("%H:%M")


# Bulgarian style guide injected into the prompt when briefings.language is
# Bulgarian. The generic "write in Bulgarian" instruction produced stiff,
# translated-sounding prose; this pins register and football terminology.
_BG_STYLE_GUIDE = """
СТИЛ НА БЪЛГАРСКИЯ ЕЗИК (задължително — пишеш за български телеграм канал):
- Пиши като опитен български спортен журналист (регистър на Gong.bg / Sportal.bg /
  Dsport) — жив, уверен, аналитичен тон, кратки и ясни изречения. НЕ превеждай
  дума по дума от английски; мисли на български.
- Никакви англицизми и калки. Примери за ПРАВИЛНО: "двубой/среща/сблъсък" (НЕ
  "мач-ъп"), "стартови състави", "контузени/наказани", "серия/форма", "головете",
  "головна разлика", "коефициент", "залог", "стойностен залог", "букмейкъри",
  "фаворит", "аутсайдер", "равенство", "головаво натоварване", "автобус/нисък блок",
  "пресинг", "владение на топката", "головете и в двете врати".
- Имена на държави/отбори на български: Саудитска Арабия, Уругвай, Мексико, Южна
  Корея, Чехия, Бразилия, Кот д'Ивоар, Хърватия и т.н. Имена на играчи и треньори
  в утвърдена българска транскрипция (Килиан Мбапе, Винисиус Жуниор, Лаутаро
  Мартинес).
- Пазарите — на български, с английския термин в скоби ЕДИНСТВЕНО първия път:
  "Двата отбора да отбележат — Да (BTTS Yes)", "Над 2.5 гола (Over 2.5)",
  "Победа за Уругвай (Away Win)".
- Бъди КОНКРЕТЕН, не общ: цитирай реални факти от търсенето — последни 4-5 резултата,
  точни голове, ключови контузени по име, голова статистика, очна ставка (Н2Н),
  пазарни коефициенти. Избягвай празни клишета ("ще бъде интересен двубой").
- Секции (използвай точно тези заглавия): 📜 Контекст, 📈 Форма и статистика,
  🩹 Контузени и състави, 🎯 Нашата прогноза и пазарът, 🔮 Заключение.
- Дължина: стегнато, ~250-400 думи. Накрая се препрочети и пренапиши всичко, което
  звучи като машинен превод или ученическо съчинение.
"""


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
        """Briefing for every upcoming WC fixture today (no lineups). Returns count sent.

        The window is anchored to the SAME bounds get_daily_picks uses (today
        00:00 UTC + 30h) — not now+30h — so we only brief matches that have a
        saved pick from this run, and tomorrow's matches are briefed by
        tomorrow's run exactly once (the sent-guard is day-keyed).
        """
        if not self._enabled():
            return 0
        now = datetime.utcnow()
        day_start = datetime.combine(date.today(), datetime.min.time())
        fixtures = self._wc_fixtures_between(
            start=max(now, day_start), end=day_start + timedelta(hours=30)
        )
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
        backend = self._resolve_backend()
        if backend is None:
            logger.warning(
                "Briefings skipped: no auth available — set CLAUDE_CODE_OAUTH_TOKEN "
                "(Pro subscription, $0 extra) or ANTHROPIC_API_KEY (per-token credits)"
            )
            return False
        if backend == "anthropic_api":
            try:
                import anthropic  # noqa: F401
            except ImportError:
                logger.warning("Briefings skipped: anthropic package not installed")
                return False
        return True

    def _resolve_backend(self):
        """Pick the LLM backend for briefings.

        claude_code  — headless Claude Code CLI billed to the Claude Pro
                       subscription (CLAUDE_CODE_OAUTH_TOKEN). $0 extra.
        anthropic_api — direct API with ANTHROPIC_API_KEY (per-token credits).

        The configured backend is used when its auth is present; otherwise we
        fall back to whichever auth exists so briefings never silently die.
        Returns None when neither is available.
        """
        preferred = self.config.get("briefings.backend", "claude_code")
        has_token = bool(os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"))
        has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
        if preferred == "claude_code":
            if has_token:
                return "claude_code"
            if has_key:
                logger.info(
                    "Briefings: CLAUDE_CODE_OAUTH_TOKEN not set — falling back to "
                    "the Anthropic API (this consumes paid credits)"
                )
                return "anthropic_api"
        else:
            if has_key:
                return "anthropic_api"
            if has_token:
                logger.info("Briefings: ANTHROPIC_API_KEY not set — using Claude Code subscription")
                return "claude_code"
        return None

    def _wc_fixtures(self, window_hours: int, min_minutes_ahead: int = 0,
                     max_minutes_ahead: int = None) -> list:
        """Return WC fixture match IDs kicking off within a now-relative window."""
        now = datetime.utcnow()
        start = now + timedelta(minutes=min_minutes_ahead)
        end = (now + timedelta(minutes=max_minutes_ahead)
               if max_minutes_ahead is not None
               else now + timedelta(hours=window_hours))
        return self._wc_fixtures_between(start, end)

    def _wc_fixtures_between(self, start: datetime, end: datetime) -> list:
        """Return WC fixture match IDs with kickoff in [start, end] (UTC)."""
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

        # Menu of selections Claude may switch the pick to (real odds only).
        odds_data, home_name, away_name = self._match_odds_and_teams(match_id)
        menu = []
        finalize = self.config.get("briefings.finalize_picks", True)
        if finalize and odds_data:
            try:
                menu = self.agent.value_calculator.available_selections(
                    analysis.predictions, odds_data,
                    home_team_name=home_name, away_team_name=away_name,
                )
            except Exception as e:
                logger.debug(f"available_selections failed for {match_id}: {e}")

        # Anchor the dossier to the ACTUAL saved pick (the tracked bet), not a
        # fresh re-analysis — find_best_bet is non-deterministic across runs
        # (odds move), so re-deriving here gave Claude a different selection than
        # what was saved, and a KEEP then contradicted the footer (Saudi Arabia
        # vs Uruguay: prose kept "BTTS Yes" but tracked bet was Over 2.5).
        saved_pick = self._current_saved_pick(match_id)
        dossier = self._build_dossier(analysis, round_name, lineups, saved_pick)
        briefing, decision = await self._research_and_write(
            analysis.match_name, dossier, lineup_aware=lineup_aware,
            has_lineups=bool(lineups), menu=menu if finalize else [],
        )
        if not briefing:
            return False

        # Apply Claude's final decision to the persisted pick (KEEP/CHANGE).
        if finalize and decision:
            try:
                bound = self._apply_decision(
                    match_id, decision, analysis, odds_data, home_name, away_name
                )
                # Freeze the match for the rest of the day ONLY when the decision
                # bound to a real saved pick: later pipeline runs must not
                # regenerate picks behind the final ruling. When no pick existed,
                # freezing would block the pick from ever being created today
                # (USA vs Paraguay, 2026-06-12).
                if bound:
                    self._mark_sent("final", match_id)
            except Exception as e:
                logger.warning(f"Could not apply briefing decision for {match_id}: {e}")

        header = self._header(analysis, round_name, lineups)
        footer = self._final_bet_footer(match_id) if finalize else ""
        await self.telegram._send_chunked(
            f"{header}\n\n{_sanitize_telegram_html(briefing)}{footer}"
        )
        self._mark_sent(kind, match_id)
        label = "PRE-MATCH (lineups)" if lineups else (
            "PRE-MATCH" if lineup_aware else "PREVIEW")
        logger.info(f"Briefing sent [{label}]: {analysis.match_name}")
        return True

    def _match_odds_and_teams(self, match_id: int):
        """Return (odds_data list, home_name, away_name) for a match — same odds_data
        shape analyze_fixture feeds the value calculator."""
        odds_data, home_name, away_name = [], "", ""
        try:
            with self.db.get_session() as session:
                m = session.get(Match, match_id)
                if m:
                    ht = session.get(Team, m.home_team_id)
                    at = session.get(Team, m.away_team_id)
                    home_name = ht.name if ht else ""
                    away_name = at.name if at else ""
                rows = session.query(Odds).filter(Odds.match_id == match_id).all()
                odds_data = [
                    {
                        "market_type": o.market_type,
                        "selection": o.selection,
                        "odds_value": o.odds_value,
                        "bookmaker": o.bookmaker,
                        "opening_odds": o.opening_odds,
                    }
                    for o in rows
                ]
        except Exception as e:
            logger.debug(f"_match_odds_and_teams failed for {match_id}: {e}")
        return odds_data, home_name, away_name

    def _apply_decision(self, match_id, decision, analysis, odds_data,
                        home_name, away_name) -> bool:
        """Apply the LLM verdict to today's saved pick for this match.

        KEEP  → leave the saved pick unchanged.
        CHANGE→ rebuild the pick for the chosen market_key and overwrite the row.

        Returns True only when the decision actually bound to a saved pick —
        the caller writes the day-freeze marker ONLY then. Freezing a match
        whose decision applied to nothing once blocked the pick from ever
        being created that day (USA vs Paraguay, 2026-06-12).
        """
        action = (decision.get("action") or "").upper()

        with self.db.get_session() as session:
            picks = session.query(SavedPick).filter(
                SavedPick.match_id == match_id,
                SavedPick.pick_date == date.today(),
                SavedPick.result.is_(None),
            ).order_by(SavedPick.expected_value.desc()).all()
            if not picks:
                logger.info(f"Briefing decision {action}: no saved pick for match {match_id}")
                return False
            if action != "CHANGE":
                return True  # KEEP on an existing pick → decision is binding
            primary = picks[0]

            # CHANGE
            market_key = decision.get("market_key")
            new = self.agent.value_calculator.build_selection_pick(
                analysis.predictions, odds_data, market_key,
                match_name=analysis.match_name, home_team_name=home_name,
                away_team_name=away_name, match_id=match_id, league=analysis.league,
            )
            if not new:
                logger.info(
                    f"Briefing CHANGE to '{market_key}' rejected for "
                    f"{analysis.match_name}: no real odds — keeping original pick"
                )
                return True
            # If another pending pick on this match already holds the target
            # selection, switching the primary would duplicate the bet —
            # consolidate by dropping the primary instead.
            for other in picks[1:]:
                if other.selection == new.selection:
                    session.delete(primary)
                    session.commit()
                    logger.info(
                        f"Briefing SWITCH consolidated: {analysis.match_name} already "
                        f"holds {new.selection} — dropped duplicate {primary.selection}"
                    )
                    return True
            old_sel = primary.selection
            # Cast numerics to plain float — build_selection_pick carries numpy
            # scalars (round() on model probs → np.float64), and under numpy 2.x
            # those render as 'np.float64(..)' in SQL, aborting the UPDATE. The
            # global adapter in database.py also covers this; the float() here
            # mirrors _save_picks and keeps the write correct under SQLite too.
            primary.market = new.market
            primary.selection = new.selection
            primary.odds = float(new.odds)
            primary.predicted_probability = float(new.predicted_probability)
            primary.expected_value = float(new.expected_value)
            primary.confidence = float(new.confidence)
            primary.kelly_stake_percentage = float(new.kelly_stake_percentage)
            primary.risk_level = new.risk_level
            session.commit()
            _reason = (decision.get("reason") or "").strip() or "no reason given"
            logger.info(
                f"Briefing SWITCH: {analysis.match_name} {old_sel} → "
                f"{new.selection} @ {new.odds:.2f} (Claude's call — {_reason})"
            )
            return True

    # Bulgarian display names for tracked selections (footer is code-generated).
    _SELECTION_BG = {
        "Home Win": "Победа за домакина",
        "Away Win": "Победа за госта",
        "Draw": "Равенство",
        "Over 1.5 Goals": "Над 1.5 гола",
        "Over 2.5 Goals": "Над 2.5 гола",
        "Over 3.5 Goals": "Над 3.5 гола",
        "Under 1.5 Goals": "Под 1.5 гола",
        "Under 2.5 Goals": "Под 2.5 гола",
        "Under 3.5 Goals": "Под 3.5 гола",
        "BTTS Yes": "Двата отбора да отбележат — Да",
        "BTTS No": "Двата отбора да отбележат — Не",
        "Home Over 1.5": "Домакинът над 1.5 гола",
        "Away Over 1.5": "Гостът над 1.5 гола",
    }

    def _final_bet_footer(self, match_id: int) -> str:
        """Authoritative footer with the FINAL tracked bet(s), read back from the
        DB after the decision was applied. Guarantees the Telegram message can
        never contradict what is actually tracked (e.g. when a CHANGE was
        rejected for missing odds after the prose was already written)."""
        try:
            with self.db.get_session() as session:
                rows = session.query(
                    SavedPick.selection, SavedPick.odds, SavedPick.kelly_stake_percentage
                ).filter(
                    SavedPick.match_id == match_id,
                    SavedPick.pick_date == date.today(),
                    SavedPick.result.is_(None),
                ).all()
        except Exception as e:
            logger.debug(f"Final-bet footer failed for {match_id}: {e}")
            return ""
        if not rows:
            return "\n\n📌 <b>Финален залог:</b> няма записан залог за този двубой"
        lines = []
        for sel, odds, stake in rows:
            bg = self._SELECTION_BG.get(sel, sel)
            stake_txt = f" · залог {stake:.1f}%" if stake else ""
            lines.append(f"📌 <b>Финален залог:</b> {bg} ({sel}) @ {odds:.2f}{stake_txt}")
        return "\n\n" + "\n".join(lines)

    def _header(self, analysis, round_name: str, lineups: dict) -> str:
        ko = _kyiv(analysis.match_date)
        tag = "🟢 СЪСТАВИТЕ СА ОБЯВЕНИ" if lineups else "📋 ПРЕВЮ"
        rnd = f" · {self._round_bg(round_name)}" if round_name else ""
        return (
            f"<b>🏆 Световно първенство — {analysis.match_name}</b>\n"
            f"<i>{tag}{rnd} · ⏰ {ko} ч. българско време</i>"
        )

    @staticmethod
    def _round_bg(round_name: str) -> str:
        """Translate API-Football round names to Bulgarian for the header."""
        rn = round_name or ""
        low = rn.lower()
        if "group stage" in low:
            # "Group Stage - 1" → "Групова фаза, кръг 1"
            num = rn.split("-")[-1].strip() if "-" in rn else ""
            return f"Групова фаза, кръг {num}" if num.isdigit() else "Групова фаза"
        return {
            "round of 32": "1/16-финал",
            "round of 16": "Осминафинал",
            "quarter-finals": "Четвъртфинал",
            "semi-finals": "Полуфинал",
            "third place final": "Мач за 3-то място",
            "3rd place final": "Мач за 3-то място",
            "final": "Финал",
        }.get(low, rn)

    def _current_saved_pick(self, match_id: int):
        """Return (selection, odds, market) of today's tracked pick for this match,
        or None — the dossier anchors to this so the briefing reasons about the
        bet that is actually saved, not a re-derived one."""
        try:
            with self.db.get_session() as session:
                row = session.query(
                    SavedPick.selection, SavedPick.odds, SavedPick.market
                ).filter(
                    SavedPick.match_id == match_id,
                    SavedPick.pick_date == date.today(),
                    SavedPick.result.is_(None),
                ).order_by(SavedPick.expected_value.desc()).first()
                if row:
                    return {"selection": row[0], "odds": row[1], "market": row[2]}
        except Exception as e:
            logger.debug(f"_current_saved_pick failed for {match_id}: {e}")
        return None

    def _build_dossier(self, analysis, round_name: str, lineups: dict,
                       saved_pick: dict = None) -> str:
        """Deterministic data block from our own model + DB. Fed to Claude as ground truth."""
        ens = analysis.predictions.get("ensemble", {}) or {}
        f = analysis.features or {}
        lines = [f"MATCH: {analysis.match_name}"]
        if round_name:
            lines.append(f"STAGE: {round_name}")
        lines.append(f"KICKOFF (Bulgarian time, EEST): {_kyiv(analysis.match_date)}")

        # Model prediction
        lines.append("")
        if f.get("model_low_coverage"):
            lines.append(
                "⚠ MODEL COVERAGE WARNING: our database has little or no match "
                "history for at least one of these teams, so the model numbers "
                "below are WEAK PRIORS (league-average regression), not informed "
                "estimates. Base your decision primarily on your own web research "
                "(recent results, qualifying form, squad strength, market odds)."
            )
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

        # Current tracked pick — anchored to the SAVED bet (what KEEP refers to
        # and what the footer will show), so prose, decision and footer agree.
        lines.append("")
        if saved_pick:
            lines.append(
                f"OUR CURRENT PICK (the bet on the slate right now): "
                f"{saved_pick['selection']} @ {saved_pick['odds']:.2f}. "
                f"KEEP means back exactly this; CHANGE means replace it."
            )
        else:
            lines.append(
                "OUR CURRENT PICK: none yet — choose the best selection from the menu."
            )

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
                                  lineup_aware: bool, has_lineups: bool,
                                  menu: list = None):
        """Call Claude Opus 4.8 with server-side web search.

        Returns (briefing_text, decision) where decision is a dict
        {action: KEEP|PASS|CHANGE, market_key, confidence} or None.
        """
        language = self.config.get("briefings.language", "Bulgarian")
        menu = menu or []

        if has_lineups:
            task = (
                "Confirmed starting XIs are included in the dossier. Search the web for "
                "the latest team news, tactical context, and any last-minute changes, then "
                "write a pre-match briefing. Explicitly assess how the confirmed lineups "
                "affect our pick: are key attackers/defenders starting or benched?"
            )
        elif lineup_aware:
            task = (
                "Lineups are not published yet. Search for the latest probable XIs, team "
                "news and injuries, then write a pre-match briefing and assess how it lines "
                "up against our pick."
            )
        else:
            task = (
                "Write a rich pre-match PREVIEW (the match is hours away, so no confirmed "
                "lineups exist — do not invent any). Run SEVERAL web searches to gather "
                "CONCRETE facts: each team's last 4-5 results with scores, their current "
                "form/streak, the head-to-head record, confirmed injuries/suspensions by "
                "player name, what's at stake (qualification scenario), and the current "
                "bookmaker odds. Build the briefing around those specific facts — no vague "
                "filler. If a search returns nothing for a thin-data team, say so plainly."
            )

        # Decision authority: confirm or switch among real-odds selections.
        # There is NO pass/veto — the channel carries a bet on every WC match,
        # so Claude's job is to pick the SAFEST/BEST selection, not to opt out.
        if menu:
            menu_lines = "\n".join(
                f"  - market_key={m['market_key']} | {m['selection']} @ {m['odds']} "
                f"(model {m['probability']:.0%})"
                for m in menu
            )
            decision_block = (
                "\n\nFINAL CALL ON THE BET — pick the selection MOST LIKELY TO WIN.\n"
                "A bet is always placed on this match. The model has proposed a current pick; "
                "your job is to VERIFY it against your web research and choose the selection "
                "with the highest chance of winning at odds of at least 1.50.\n"
                "  KEEP   — the model's current pick is already the most likely winner.\n"
                "  CHANGE — your research (recent form, goals, injuries, matchup, market) "
                "shows another selection from the menu is MORE LIKELY TO WIN. Switch to it.\n"
                "Prefer safe, high-probability outcomes over coin-flips: a strong favourite "
                "scoring 2+ (Home/Away Over 1.5), a clear winner, or both-teams-to-score when "
                "two attacking sides meet are all good, high-hit-rate choices. Pick purely on "
                "win probability, not on price — but stay at or above 1.50.\n"
                "Choose ONLY from this menu (the selections bookmakers price); each line shows "
                "the model's probability and the odds:\n"
                f"{menu_lines}\n\n"
                "Your verdict section MUST state plainly whether you KEEP or CHANGE, and that "
                "MUST match the machine block below — never say you are changing in the prose "
                "and then KEEP (or vice versa). Do NOT state final odds in the prose; an "
                "authoritative footer with the tracked bet is appended automatically. As the "
                "VERY LAST thing in your reply append this exact block (English keys, nothing "
                "after it):\n"
                "<<<DECISION>>>\n"
                "action: KEEP|CHANGE\n"
                "market_key: <one key from the menu, required only if action is CHANGE>\n"
                "confidence: <your win probability for the final pick, 0.0-1.0>\n"
                "reason: <one short phrase: why this selection is the most likely to win>\n"
                "<<<END>>>"
            )
        else:
            decision_block = ""

        _is_bg = language.strip().lower() in ("bulgarian", "български", "bg")
        style_block = _BG_STYLE_GUIDE if _is_bg else (
            f"Write in fluent, native-sounding {language} — the register of a "
            f"professional sportswriter, never a literal translation. Re-read the "
            f"final text and rewrite anything that sounds machine-translated."
        )

        system = (
            f"You are an expert football analyst writing a concise, high-signal World Cup "
            f"match briefing for a Telegram betting channel. Do all your web research and "
            f"reasoning in English (search in English for the best coverage) — but write the "
            f"FINAL briefing message that gets posted to Telegram ENTIRELY IN {language.upper()}. "
            f"Only the final message is in {language}; everything else (searches, thinking) "
            f"stays in English. You have access to a web search tool — use it to gather "
            f"current, factual information (storyline, recent form, head-to-head, team news, "
            f"injuries, betting market). Never invent lineups, injuries, or quotes; if "
            f"something is uncertain, say so. Be specific and cite what you found. "
            + _HTML_RULES + "\n" + style_block
        )

        user = (
            f"{task}\n\n"
            f"Here is our internal model dossier for {match_name} (treat these numbers as "
            f"ground truth from our prediction system — research everything else):\n\n"
            f"{dossier}\n\n"
            f"Structure the briefing with emoji section headers in {language} (follow the "
            f"style guide's section names if one was given). End the prose with a one-line "
            f"'Източници:' / 'Sources:' note listing the main outlets you used."
            f"{decision_block}"
        )

        backend = self._resolve_backend()
        if backend == "claude_code":
            text_out = await self._call_claude_code(system, user, match_name)
            # Resilience: Claude Code shares the Pro 5-hour quota with the user's
            # own usage and periodically returns "session limit reached", which
            # zeroed out whole briefing runs (Jun 16 & Jun 22). When it comes back
            # empty, fall back to the paid Anthropic API for THIS briefing so the
            # match still gets covered. Only fires on failure, only if an API key
            # is present, and is config-gated.
            if (not text_out
                    and os.environ.get("ANTHROPIC_API_KEY")
                    and self.config.get("briefings.api_fallback", True)):
                logger.warning(
                    f"Claude Code unavailable for {match_name} (likely Pro session "
                    f"limit) — falling back to the Anthropic API (uses paid credits)"
                )
                text_out = await self._call_anthropic_api(system, user, match_name)
        else:
            text_out = await self._call_anthropic_api(system, user, match_name)

        decision = self._parse_decision(text_out) if menu else None
        # Audit log so prose-vs-decision mismatches are visible in CI: did the
        # model emit a parseable decision block, and what did it decide?
        if menu:
            _block_found = bool(_DECISION_RE.search(text_out or ""))
            if decision:
                logger.info(
                    f"Briefing decision [{match_name}]: action={decision['action']} "
                    f"market_key={decision.get('market_key')} "
                    f"reason={decision.get('reason')!r} (block_found={_block_found})"
                )
            else:
                logger.warning(
                    f"Briefing decision [{match_name}]: NO parseable decision block "
                    f"(block_found={_block_found}) — pick left unchanged. "
                    f"Output tail: {(text_out or '')[-200:]!r}"
                )
        # Strip the machine-readable block from the user-facing briefing — both the
        # wrapped form and any unwrapped trailing field lines (action:/market_key:
        # /confidence:/reason:) the model may have emitted without the markers.
        clean = _DECISION_RE.sub("", text_out)
        clean = re.sub(
            r"(?im)^\s*(action|market_key|confidence|reason)\s*[:=].*$", "", clean
        )
        clean = re.sub(r"<<<\s*/?\s*(DECISION|END)\s*>>>", "", clean, flags=re.IGNORECASE)
        clean = clean.strip()

        # M3: flag prose-vs-decision divergence. If the machine decision is KEEP
        # but the user-facing prose strongly implies a switch (BG/EN switch verbs),
        # the message will contradict the tracked bet — log it so it's auditable.
        if menu and decision and decision["action"] == "KEEP":
            if _SWITCH_PROSE_RE.search(clean):
                logger.warning(
                    f"Briefing prose/decision divergence [{match_name}]: decision is "
                    f"KEEP but prose contains switch-language — message may read as a "
                    f"change while the tracked pick is unchanged. Review the verdict wording."
                )
        return clean, decision

    async def _call_anthropic_api(self, system: str, user: str, match_name: str) -> str:
        """Direct Anthropic API path (ANTHROPIC_API_KEY — paid per-token credits)."""
        from anthropic import AsyncAnthropic

        client = AsyncAnthropic()  # reads ANTHROPIC_API_KEY from env
        tools = [{"type": "web_search_20260209", "name": "web_search"}]

        messages = [{"role": "user", "content": user}]
        text_out = ""
        # max_tokens must cover adaptive thinking + several web searches + the
        # full Bulgarian briefing; 4096 was being consumed by thinking/search and
        # the final text came back empty. 16000 gives ample room.
        for _ in range(6):
            try:
                resp = await client.messages.create(
                    model=_BRIEFING_MODEL,
                    max_tokens=16000,
                    thinking={"type": "adaptive"},
                    system=system,
                    tools=tools,
                    messages=messages,
                )
            except Exception as e:
                logger.warning(f"Claude briefing call failed for {match_name}: {e}")
                return ""

            # Server-side web search may return pause_turn at the iteration cap —
            # re-send to resume.
            if resp.stop_reason == "pause_turn":
                messages = [
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": resp.content},
                ]
                continue

            text_out = "".join(
                b.text for b in resp.content if getattr(b, "type", "") == "text"
            ).strip()
            if not text_out and resp.stop_reason == "max_tokens":
                logger.warning(
                    f"Anthropic briefing for {match_name} hit max_tokens with no "
                    f"text (thinking/search consumed the budget)"
                )
            break
        return text_out

    async def _call_claude_code(self, system: str, user: str, match_name: str) -> str:
        """Headless Claude Code path — billed to the Claude Pro subscription
        (CLAUDE_CODE_OAUTH_TOKEN), $0 in API credits.

        Runs `claude -p` with only WebSearch/WebFetch allowed, in a temp cwd so
        no repo CLAUDE.md/project context pollutes the prompt. ANTHROPIC_API_KEY
        is stripped from the child env so the CLI can never silently bill the
        paid API instead of the subscription.
        """
        import asyncio
        import shutil
        import tempfile

        exe = shutil.which("claude")
        if not exe:
            logger.warning(
                "Briefings: `claude` CLI not found on PATH — install with "
                "`npm install -g @anthropic-ai/claude-code`"
            )
            return ""

        env = dict(os.environ)
        env.pop("ANTHROPIC_API_KEY", None)  # subscription billing only — never credits

        prompt = f"{system}\n\n{user}"
        tmpdir = tempfile.mkdtemp(prefix="briefing_")
        cmd = [
            exe, "-p", prompt,
            "--model", "sonnet",  # Pro-plan Claude Code model
            "--allowedTools", "WebSearch,WebFetch",
            "--output-format", "text",
        ]
        # CI runners research slowly — a real briefing took 4.5 min, and one hit
        # the old 480s cap mid-research, costing the match its briefing for the
        # day. 900s ceiling + one retry.
        _TIMEOUT_S = 900
        for _attempt in (1, 2):
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=env,
                    cwd=tmpdir,
                )
                try:
                    out, err = await asyncio.wait_for(
                        proc.communicate(), timeout=_TIMEOUT_S
                    )
                    break
                except asyncio.TimeoutError:
                    proc.kill()
                    logger.warning(
                        f"Claude Code briefing timed out ({_TIMEOUT_S}s) for "
                        f"{match_name} (attempt {_attempt}/2)"
                    )
                    if _attempt == 2:
                        return ""
            except Exception as e:
                logger.warning(f"Claude Code briefing error for {match_name}: {e}")
                return ""
        try:
            if proc.returncode != 0:
                # Claude Code writes most errors (usage limits, auth) to STDOUT
                # in -p mode — log both streams or failures are undiagnosable.
                _err = err.decode(errors="replace").strip()
                _out = out.decode(errors="replace").strip()
                logger.warning(
                    f"Claude Code briefing failed for {match_name} "
                    f"(exit {proc.returncode}): stderr={_err[:300]!r} "
                    f"stdout={_out[-300:]!r}"
                )
                return ""
            return out.decode(errors="replace").strip()
        except Exception as e:
            logger.warning(f"Claude Code briefing error for {match_name}: {e}")
            return ""

    @staticmethod
    def _parse_decision(text: str):
        """Extract the decision into {action, market_key, confidence, reason}.

        Robust to the model dropping or mangling the <<<DECISION>>> wrapper: if the
        strict block is absent, fall back to scanning for `action:`/`market_key:`/
        etc. lines anywhere in the text. This was a real failure mode — the
        headless Claude Code agent sometimes omitted the wrapper, so the decision
        parsed as None and CHANGE was silently never applied while the prose said
        it was changing.
        """
        text = text or ""
        m = _DECISION_RE.search(text)
        if m:
            body = m.group(1)
        else:
            # Loose fallback: only the field lines, no wrapper. Require an
            # explicit action line so we don't pick up the word "action" in prose.
            if not re.search(r"^\s*action\s*[:=]\s*(KEEP|CHANGE|PASS)\b",
                             text, re.IGNORECASE | re.MULTILINE):
                return None
            body = text
        out = {}
        for line in body.splitlines():
            mm = re.match(r"\s*(action|market_key|confidence|reason)\s*[:=]\s*(.+)",
                          line, re.IGNORECASE)
            if mm:
                out[mm.group(1).strip().lower()] = mm.group(2).strip()
        action = (out.get("action") or "").upper()
        if action == "PASS":
            # Veto authority was removed (user decision: a bet on EVERY match).
            # If the model still emits PASS, treat it as KEEP.
            action = "KEEP"
        if action not in ("KEEP", "CHANGE"):
            return None
        decision = {
            "action": action,
            "market_key": out.get("market_key") or None,
            "reason": out.get("reason") or "",
        }
        try:
            decision["confidence"] = float(out.get("confidence", ""))
        except (TypeError, ValueError):
            decision["confidence"] = None
        return decision
