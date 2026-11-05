"""ONE market/selection vocabulary, readable from outside the pricing path.

WHY THIS EXISTS, AND WHAT THE SPLIT COST. 2026-11-05.

`saved_picks` and `odds` name the same bet differently on BOTH axes:

    the same bet          market          selection
    ------------------    ------------    ---------------------
    `odds` says           over_under      Under 3.5
    `saved_picks` say     Under 3.5       Under 3.5 Goals

The mapping that reconciles them existed only as **two identical copies inside
two methods of `value_calculator`**, where nothing outside that file could read
it. Measured consequences:

* `h1_analysis.one_price_per_fixture` compared `(market, selection)` with raw
  tuple equality and matched **0 of 1,120** trajectories against 2,587 picks,
  on 11 of 12 fixtures whose identifier spaces overlapped. The step can never
  have matched anything.
* the market-mix join dropped **900 of 1,583** settled picks, and the
  pre-registration had to describe reading the mapping *by parsing
  `value_calculator.py`* -- a source-text dependency on a pricing file.

So the vocabulary moves here. This module imports the standard library ONLY:
nothing on the capture path, so importing it cannot recruit a test into the
capture gate (`scripts/capture_critical.CAPTURE_PATH`), which the
import-pulls-a-file-into-the-gate trap has cost four times.

HOW COMPARISONS ARE MEANT TO WORK. Never raw tuple equality. Put both sides
through `canonical()` and compare the results, or call `same_bet()`.
`canonical()` returns **None** when it cannot tell -- an unknown selection, or
one that is ambiguous without a market -- because a vocabulary that guesses is
worse than the split it replaces.

THE AMBIGUITY IS REAL AND IS WHY THE MARKET IS REQUIRED. `"Home"` is an alias
of `Home Win` (1X2) *and* of `DNB Home` (draw_no_bet); `"Away"` likewise. With
the market given they separate; without it `canonical()` refuses.
"""
from typing import Dict, List, Optional, Tuple

#: Placeholders for the two values that are not constants. `selection_map` in
#: `value_calculator` interpolated the live team names directly into the alias
#: lists, because the Odds API stores team names for 1X2.
HOME_TEAM = "\x00HOME_TEAM\x00"
AWAY_TEAM = "\x00AWAY_TEAM\x00"

#: canonical pick-side `selection` -> the values the `odds` table may carry.
#:
#: MOVED VERBATIM from `value_calculator._find_best_odds`, which held one of
#: two AST-identical copies (22 keys each, zero symmetric difference, measured
#: 2026-11-05). Order is preserved because `_find_best_odds` iterates it and
#: order decides nothing today -- but a reordering would be a silent change to
#: a pricing input, so it is kept.
SELECTION_ALIASES: Dict[str, Tuple[str, ...]] = {
    # "Home Win" / "Away Win" are stored by Flashscore scraper;
    # legacy Odds-API records use "1"/"2" or team names.
    "Home Win": ("1", "Home", "home_win", "Home Win", HOME_TEAM),
    "Draw": ("X", "Draw", "draw", "Draw"),
    "Away Win": ("2", "Away", "away_win", "Away Win", AWAY_TEAM),
    # Flashscore O/U stored without " Goals" suffix ("Over 2.5", etc.)
    "Over 2.5 Goals": ("Over 2.5", "Over", "Over 2.5 Goals"),
    "Under 2.5 Goals": ("Under 2.5", "Under", "Under 2.5 Goals"),
    "Over 1.5 Goals": ("Over 1.5", "Over 1.5 Goals"),
    "Under 1.5 Goals": ("Under 1.5", "Under 1.5 Goals"),
    "Over 3.5 Goals": ("Over 3.5", "Over 3.5 Goals"),
    "Under 3.5 Goals": ("Under 3.5", "Under 3.5 Goals"),
    "Over 4.5 Goals": ("Over 4.5", "Over 4.5 Goals"),
    "Under 4.5 Goals": ("Under 4.5", "Under 4.5 Goals"),
    "BTTS Yes": ("Yes", "BTTS Yes"),
    "BTTS No": ("No", "BTTS No"),
    # Team goal line markets (stored via API-Football team_goals market)
    "Home Over 0.5": ("Home Over 0.5", "Home Team Over 0.5", "Home - Over 0.5"),
    "Away Over 0.5": ("Away Over 0.5", "Away Team Over 0.5", "Away - Over 0.5"),
    "Home Over 1.5": ("Home Over 1.5", "Home Team Over 1.5", "Home - Over 1.5"),
    "Away Over 1.5": ("Away Over 1.5", "Away Team Over 1.5", "Away - Over 1.5"),
    # Double Chance / Draw No Bet (stored via API-Football)
    "Double Chance 1X": ("Double Chance 1X", "1X", "Home/Draw"),
    "Double Chance 12": ("Double Chance 12", "12", "Home/Away"),
    "Double Chance X2": ("Double Chance X2", "X2", "Draw/Away"),
    "DNB Home": ("DNB Home", "Home"),
    "DNB Away": ("DNB Away", "Away"),
}

#: pick-side `market` -> the values the `odds` table's `market_type` may carry.
#: MOVED VERBATIM from the same two copies (13 keys each).
MARKET_ALIASES: Dict[str, Tuple[str, ...]] = {
    "1X2": ("1X2", "h2h"),
    "Over 2.5": ("over_under", "totals"),
    "Under 2.5": ("over_under", "totals"),
    "Over 1.5": ("over_under", "totals"),
    "Under 1.5": ("over_under", "totals"),
    "Over 3.5": ("over_under", "totals"),
    "Under 3.5": ("over_under", "totals"),
    "Over 4.5": ("over_under", "totals"),
    "Under 4.5": ("over_under", "totals"),
    "BTTS": ("btts",),
    "Team Goals": ("team_goals", "team_over_under"),
    "Double Chance": ("double_chance",),
    "Draw No Bet": ("draw_no_bet",),
}

#: canonical `selection` -> its market FAMILY, the first alias of its market.
#:
#: This is the axis that disambiguates: "Home" is an alias of two different
#: canonical selections in two different families. `capture_closing_lines`
#: holds the same selection -> market_type relation inside `SELECTION_SPEC`
#: (with the line, side and leg index it needs besides); the two agree on all
#: 22 keys, measured 2026-11-05, and
#: `tests/test_market_vocabulary.py` asserts they keep agreeing rather than
#: this module silently becoming a fourth table.
SELECTION_FAMILY: Dict[str, str] = {
    "Home Win": "1X2",
    "Draw": "1X2",
    "Away Win": "1X2",
    "Over 1.5 Goals": "over_under",
    "Under 1.5 Goals": "over_under",
    "Over 2.5 Goals": "over_under",
    "Under 2.5 Goals": "over_under",
    "Over 3.5 Goals": "over_under",
    "Under 3.5 Goals": "over_under",
    "Over 4.5 Goals": "over_under",
    "Under 4.5 Goals": "over_under",
    "BTTS Yes": "btts",
    "BTTS No": "btts",
    "Home Over 0.5": "team_goals",
    "Away Over 0.5": "team_goals",
    "Home Over 1.5": "team_goals",
    "Away Over 1.5": "team_goals",
    "DNB Home": "draw_no_bet",
    "DNB Away": "draw_no_bet",
    "Double Chance 1X": "double_chance",
    "Double Chance 12": "double_chance",
    "Double Chance X2": "double_chance",
}

#: every `market_type`/`market` spelling -> its family. Built from
#: MARKET_ALIASES so there is no second list to drift.
_FAMILY_OF_MARKET: Dict[str, str] = {}
for _pick_market, _aliases in MARKET_ALIASES.items():
    _family = _aliases[0]
    _FAMILY_OF_MARKET[_pick_market] = _family
    for _a in _aliases:
        _FAMILY_OF_MARKET[_a] = _family
del _pick_market, _aliases, _family, _a


def market_aliases(market: str) -> List[str]:
    """The `market_type` values that correspond to this pick `market`.

    Identical to the inline `market_map.get(market, [market])` it replaces,
    including the fall-through for an unknown market.
    """
    found = MARKET_ALIASES.get(market)
    return list(found) if found is not None else [market]


def selection_aliases(selection: str, home_team_name: str = "",
                      away_team_name: str = "") -> List[str]:
    """The `odds.selection` values that correspond to this pick `selection`.

    Identical to the inline
    `[s for s in selection_map.get(selection, [selection]) if s]`, including
    the fall-through and the empty-string filter -- an absent team name must
    not become an alias that matches an empty `selection` column.
    """
    found = SELECTION_ALIASES.get(selection)
    raw = list(found) if found is not None else [selection]
    out = []
    for s in raw:
        if s == HOME_TEAM:
            s = home_team_name
        elif s == AWAY_TEAM:
            s = away_team_name
        if s:
            out.append(s)
    return out


def family_of_market(market: Optional[str]) -> Optional[str]:
    """The family for either vocabulary's market spelling, or None."""
    if not market:
        return None
    return _FAMILY_OF_MARKET.get(market)


def canonical(market: Optional[str], selection: Optional[str],
              home_team_name: str = "", away_team_name: str = ""
              ) -> Optional[Tuple[str, str]]:
    """`(family, canonical_selection)` for EITHER vocabulary, or None.

    None means "cannot tell", and it is never a guess:

      * the selection matches no canonical name and no alias;
      * it matches more than one and the market does not separate them
        ("Home" without a market is `Home Win` or `DNB Home`);
      * the market and the selection disagree about the family, which means
        one of the two is wrong and silently preferring either would invent a
        bet nobody placed.
    """
    if not selection:
        return None
    want_family = family_of_market(market)

    if selection in SELECTION_FAMILY:
        canon = selection
    else:
        cands = [
            name for name in SELECTION_ALIASES
            if selection in selection_aliases(
                name, home_team_name, away_team_name)
        ]
        if want_family is not None:
            cands = [c for c in cands
                     if SELECTION_FAMILY.get(c) == want_family]
        if len(cands) != 1:
            return None
        canon = cands[0]

    family = SELECTION_FAMILY.get(canon)
    if family is None:
        return None
    if want_family is not None and want_family != family:
        return None
    return (family, canon)


def same_bet(market_a: Optional[str], selection_a: Optional[str],
             market_b: Optional[str], selection_b: Optional[str],
             home_team_name: str = "", away_team_name: str = "") -> bool:
    """Whether two (market, selection) pairs name the same bet.

    False when either side is uncanonicalisable: "I cannot tell" must never
    read as "they are the same". Callers that need the distinction compare
    `canonical()` results directly and handle None themselves.
    """
    a = canonical(market_a, selection_a, home_team_name, away_team_name)
    if a is None:
        return False
    b = canonical(market_b, selection_b, home_team_name, away_team_name)
    return b is not None and a == b
