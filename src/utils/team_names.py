"""Team name similarity helper.

Used across scrapers, settlement fallback, and pick deduplication to decide
whether two team-name strings from different data sources refer to the same
team.  Previously this lived as a private static method on FlashscoreScraper
and was reached into from `betting_agent.py` and `apifootball_scraper.py` via
`from src.scrapers.flashscore_scraper import FlashscoreScraper as _FS`,
which created a hidden cross-module dependency.
"""

import unicodedata
from difflib import SequenceMatcher


# Known cross-source name aliases (lowercase canonical form).
# Only add entries that cannot be resolved by the token/prefix logic below
# (e.g. completely different abbreviation styles).
NAME_ALIASES: dict = {
    # Atletico Madrid (API-Football uses "Ath", Flashscore uses "Atl.")
    "ath madrid": "atletico madrid",
    "atl. madrid": "atletico madrid",
    "atl madrid": "atletico madrid",
    # PSG (API-Football: "Paris SG", Flashscore: "PSG")
    "paris sg": "paris saint-germain",
    "psg": "paris saint-germain",
    # Olympiakos / Olympiacos (different Latin transliterations)
    "olympiakos": "olympiacos piraeus",
    "olympiacos": "olympiacos piraeus",
    "olympiakos piraeus": "olympiacos piraeus",
    # Dinamo Bucharest abbreviations ("Din." prefix)
    "din. bucuresti": "dinamo bucuresti",
    "din bucuresti": "dinamo bucuresti",
}


# Common club-type tags that differ between sources and should be stripped.
_STRIP = {"fc", "sc", "sk", "afc", "sfc", "cf", "bk", "fk", "ac", "as", "cd", "ad"}


# Common abbreviations that differ between sources.
_ABBREVS = {
    "utd": "united", "united": "united",
    "cty": "city", "city": "city",
    "ath": "athletic", "athletic": "athletic",
    "weds": "wednesday", "wednesday": "wednesday",
    "wed": "wednesday",
    "nott'm": "nottingham", "nottingham": "nottingham",
}


def _norm(s: str) -> str:
    s = s.lower().strip()
    if s in NAME_ALIASES:
        s = NAME_ALIASES[s]
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return s


def _tokens(n: str) -> list:
    return [t.rstrip(".") for t in n.split() if t.rstrip(".") not in _STRIP]


def _tok_match(t1: str, t2: str) -> bool:
    if t2.startswith(t1) or t1.startswith(t2):
        return True
    if _ABBREVS.get(t1) and _ABBREVS.get(t1) == _ABBREVS.get(t2):
        return True
    return SequenceMatcher(None, t1, t2).ratio() >= 0.75


def team_names_similar(name_a: str, name_b: str) -> bool:
    """Return True if two team names likely refer to the same team.

    Handles common differences between data sources:
    - Abbreviations:    "Man City" vs "Manchester City"
    - Short names:      "Oxford" vs "Oxford Utd" / "Oxford United"
    - Prefixes:         "SK Rapid" vs "Rapid Vienna"
    - Suffixes:         "Bradford" vs "Bradford City"
    - Diacritics:       "Munchen" vs "Munich", "Castellon" vs "Castellon"
    - Known aliases:    "Ath Madrid" vs "Atl. Madrid", "PSG" vs "Paris SG"
    - Spelling variants:"Olympiakos" vs "Olympiacos"
    """
    if name_a == name_b:
        return True

    a, b = _norm(name_a), _norm(name_b)
    if a == b:
        return True

    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return False

    shorter, longer = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    matches = sum(
        1 for tok in shorter
        if any(_tok_match(tok, long_tok) for long_tok in longer)
    )
    return matches / len(shorter) >= 0.7
