"""Selection mode `confident_value`: the safest picks that still carry a
good price. Stage 31, registered in docs/confident-value-preregistration.md.

ONE DEFINITION, TWO CALLERS. The backtest (`scripts/confident_value.py`) and
the pipeline flag (`betting.selection_mode`) both build their leg sets, their
blended probability and their one-pick-per-fixture choice HERE. A rule that
is backtested by one implementation and shipped by another has not been
backtested.

WHAT IT IS. A leg qualifies when the price the mode takes -- the median across
books of the sharp anchor's feed, inside one snapshot -- is at least
`min_price`, and its BLENDED probability clears a floor. One pick per fixture:
the qualifying leg with the highest blended probability. No daily count; a day
on which nothing qualifies produces no picks, and that is not an error.

SWITCHED ON at s5.23 (2026-11-13) in config.example.yaml, with its keys in
TRACKED_KEYS. An ABSENT `betting.selection_mode` still means `standard`, under
which nothing in this module is reached.

Pure: standard library plus `src.evaluation.devig` and `src.data.market_spec`,
no database, no clock, no network.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

MODE_STANDARD = "standard"
MODE_CONFIDENT_VALUE = "confident_value"
MODES = (MODE_STANDARD, MODE_CONFIDENT_VALUE)

#: The operator's definition of a good price.
MIN_PRICE = 1.70
#: Registered grids. Fixed before any outcome was read.
P_MIN_GRID = (0.50, 0.52, 0.54, 0.56, 0.58)
EDGE_GRID = (0.00, 0.02, 0.05)
#: Odds bands: [lo, hi) except the last, which is closed at the top.
BANDS = ((1.70, 2.00), (2.00, 2.50), (2.50, 3.50))
LANE_CONFIDENT = "confident"       # taken in [1.70, 2.00), on probability
LANE_VALUE = "value"               # taken in [2.00, 3.50], on edge

#: Seconds inside which two rows count as one snapshot (Stage 30's figure).
SNAPSHOT_TOLERANCE_S = 120.0
#: The sharp anchor is the Pinnacle of the SAME feed. Across feeds the median
#: gap is 255 s and the 90th percentile 8.1 hours, so a cross-feed pair is
#: not a snapshot.
ANCHORS = {"apifootball": "Pinnacle", "theodds": "TheOddsAPI-pinnacle"}

#: Which sources have an opinion on which family. Elo rates a result, not a
#: goal count, so it has none on the binary families.
FAMILY_SOURCES = {
    "1X2": ("market", "elo", "poisson"),
    "over_under": ("market", "poisson"),
    "btts": ("market", "poisson"),
    "team_goals": ("market", "poisson"),
}
#: The only lines a model probability exists for.
LINES = {"over_under": ("1.5", "2.5", "3.5", "4.5"),
         "team_goals": ("0.5", "1.5")}


#: The legs the pipeline can EXPRESS AND SETTLE: canonical selection ->
#: (market as the pipeline names it, the model's prediction key). This is
#: `ValueBettingCalculator._market_specs` restricted to the four families,
#: and a test holds the two equal. A leg outside it -- a team-goals Under,
#: which the grader does not know -- is never picked, in the backtest or live.
PICKABLE = {
    "Home Win": ("1X2", "home_win"),
    "Draw": ("1X2", "draw"),
    "Away Win": ("1X2", "away_win"),
    "Over 1.5 Goals": ("Over 1.5", "over_1.5"),
    "Over 2.5 Goals": ("Over 2.5", "over_2.5"),
    "Over 3.5 Goals": ("Over 3.5", "over_3.5"),
    "Over 4.5 Goals": ("Over 4.5", "over_4.5"),
    "Under 1.5 Goals": ("Under 1.5", "under_1.5"),
    "Under 2.5 Goals": ("Under 2.5", "under_2.5"),
    "Under 3.5 Goals": ("Under 3.5", "under_3.5"),
    "Under 4.5 Goals": ("Under 4.5", "under_4.5"),
    "BTTS Yes": ("BTTS", "btts_yes"),
    "BTTS No": ("BTTS", "btts_no"),
    "Home Over 0.5": ("Team Goals", "home_over_0.5"),
    "Away Over 0.5": ("Team Goals", "away_over_0.5"),
    "Home Over 1.5": ("Team Goals", "home_over_1.5"),
    "Away Over 1.5": ("Team Goals", "away_over_1.5"),
}


def pipeline_selection(market: str, key: str) -> Optional[str]:
    """The pipeline's canonical name for a leg, or None when the pipeline
    cannot express or settle it."""
    from src.data.market_vocabulary import canonical
    canon = canonical(market, key)
    if canon is None or canon[1] not in PICKABLE:
        return None
    return canon[1]


def feed_of(bookmaker: str) -> str:
    return "theodds" if bookmaker.startswith("TheOddsAPI-") else "apifootball"


def band_of(price: float) -> Optional[Tuple[float, float]]:
    for lo, hi in BANDS:
        if lo <= price < hi:
            return (lo, hi)
    if price == BANDS[-1][1]:
        return BANDS[-1]
    return None


# ═════════════════════════════ leg sets ═════════════════════════════════════

def build_sets(rows: Iterable[Tuple]) -> Tuple[dict, dict]:
    """Complete leg sets for ONE fixture, from its odds rows.

    `rows` is `(bookmaker, market_type, selection, odds_value, timestamp)`;
    `timestamp` may be None when the caller has none, and then the snapshot
    rule cannot be applied and is not pretended to be.

    Returns `(sets, refused)`. `sets` maps `(bookmaker, market, line, side)`
    to `{"keys", "prices", "q", "ts", "ts_min", "overround"}`, with the legs
    in `market_spec`'s own order. A set is refused when a leg is missing, when
    its overround is outside the spec's band (the gate that excludes the
    API-Football "Home/Away" trap), or when its legs span more than one
    snapshot.
    """
    from src.data.market_spec import get_spec

    raw: Dict[Tuple[str, str], Dict[str, Tuple[float, object]]] = {}
    for bm, mt, sel, val, ts in rows:
        if bm is None or val is None or val <= 1.0:
            continue
        raw.setdefault((bm, mt), {})[sel] = (float(val), ts)

    sets: Dict[Tuple, dict] = {}
    refused: Dict[str, int] = {}
    for (bm, market), by_sel in raw.items():
        spec = get_spec(market)
        if spec is None or spec.overlapping:
            continue
        lines = {None}
        if market in ("over_under", "team_goals"):
            lines = {m.group(1) for k in by_sel
                     for m in [re.search(r"(\d+\.\d+)", k)] if m}
        sides = {None} if market != "team_goals" else {"Home", "Away"}
        for line in lines:
            for side in sides:
                legs = []
                for aliases in spec.legs:
                    found = None
                    for alias in aliases:
                        if market == "over_under" and line:
                            key = f"{alias} {line}"
                        elif market == "team_goals" and line and side:
                            key = f"{side} {alias} {line}"
                        else:
                            key = alias
                        if key in by_sel:
                            found = (key,) + by_sel[key]
                            break
                    if found is None:
                        legs = None
                        break
                    legs.append(found)
                if not legs:
                    continue
                prices = [pr for _k, pr, _t in legs]
                orr = sum(1.0 / x for x in prices)
                if not (spec.overround_min <= orr <= spec.overround_max):
                    refused["overround_out_of_band"] = \
                        refused.get("overround_out_of_band", 0) + 1
                    continue
                tss = [t for _k, _p, t in legs]
                timed = all(t is not None for t in tss)
                if timed and (max(tss) - min(tss)).total_seconds() \
                        > SNAPSHOT_TOLERANCE_S:
                    refused["legs_not_one_snapshot"] = \
                        refused.get("legs_not_one_snapshot", 0) + 1
                    continue
                sets[(bm, market, line, side)] = {
                    "keys": [k for k, _p, _t in legs],
                    "prices": prices,
                    "q": [1.0 / x for x in prices],
                    "ts": max(tss) if timed else None,
                    "ts_min": min(tss) if timed else None,
                    "overround": orr,
                }
    return sets, refused


def same_snapshot(a: dict, b: dict) -> Optional[bool]:
    """True/False, or None when either set carries no timestamp."""
    if a["ts"] is None or b["ts"] is None:
        return None
    lo = min(a["ts_min"], b["ts_min"])
    hi = max(a["ts"], b["ts"])
    return (hi - lo).total_seconds() <= SNAPSHOT_TOLERANCE_S


def median(xs: Sequence[float]) -> float:
    """The pipeline's own aggregation (`_find_best_odds`): one book -> its
    price; odd -> the middle; even -> the mean of the middle two."""
    s = sorted(xs)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


def slots(sets: dict, require_snapshot: bool = True,
          exclude: Sequence[str] = (),
          cross_feed: bool = False) -> Dict[Tuple, dict]:
    """One priced slot per `(market, line, side)` of a fixture.

    For each slot and each feed that carries the anchor: the anchor's Shin
    fair probabilities and the TAKEN price of every leg -- the median across
    that feed's books whose complete set shares the anchor's keys and, when
    timestamps exist, its snapshot. The anchor itself is one of the books.
    When both feeds qualify the one with more books is used; a tie goes to
    The Odds API.

    Returns `{(market, line, side): {"keys", "fair", "taken", "books",
    "feed", "ts", "prices_by_book"}}`. A slot whose anchor set Shin refuses
    is absent, never approximated.

    Two SENSITIVITY switches, both off in the registered rule and in the
    flag: `exclude` drops named books from the median (the anchor is never
    dropped), and `cross_feed` takes the median across every book of every
    feed with no snapshot test -- the live pipeline's own `_find_best_odds`
    policy, kept so the difference between the two can be measured.
    """
    from src.evaluation.devig import shin

    by_slot: Dict[Tuple, Dict[str, dict]] = {}
    for (bm, market, line, side), st in sets.items():
        if bm == "Flashscore":
            continue
        by_slot.setdefault((market, line, side), {})[bm] = st

    out: Dict[Tuple, dict] = {}
    for slot, books in by_slot.items():
        best = None
        for feed in ("theodds", "apifootball"):
            a = books.get(ANCHORS[feed])
            if a is None:
                continue
            fair = shin(a["q"])
            if fair is None:
                continue
            members = {}
            for bm, st in books.items():
                if st["keys"] != a["keys"]:
                    continue
                if bm != ANCHORS[feed] and bm in exclude:
                    continue
                if not cross_feed:
                    if feed_of(bm) != feed:
                        continue
                    snap = same_snapshot(a, st)
                    if snap is False or (snap is None and require_snapshot):
                        continue
                members[bm] = st["prices"]
            if not members:
                continue
            cand = {
                "keys": list(a["keys"]), "fair": list(fair),
                "taken": [median([p[i] for p in members.values()])
                          for i in range(len(a["keys"]))],
                "books": len(members), "feed": feed, "ts": a["ts"],
                "prices_by_book": members,
            }
            if best is None or cand["books"] > best["books"]:
                best = cand
        if best is not None:
            out[slot] = best
    return out


def family_of(market: str, line: Optional[str]) -> Optional[str]:
    """The blend family of a slot, or None when no model probability exists
    for it (an unmodelled line, an unmodelled market)."""
    if market not in FAMILY_SOURCES:
        return None
    if market in LINES and line not in LINES[market]:
        return None
    return market


# ═════════════════════════ the blended probability ══════════════════════════

def model_set(market: str, line: Optional[str], side: Optional[str],
              keys: Sequence[str], pred: Optional[dict]) -> Optional[List[float]]:
    """A model's probabilities for one leg set, in the set's own key order.

    `pred` is a model's `predict()` dict (`home_win`, `over_2.5`, `btts_yes`,
    `home_over_0.5`, ...). None when the model has no opinion on this set --
    never a default. An Under is one minus its Over.
    """
    if not pred:
        return None
    out = []
    for key in keys:
        k = key.lower()
        if market == "1X2":
            name = {"home": "home_win", "draw": "draw", "away": "away_win",
                    "home win": "home_win", "away win": "away_win",
                    "1": "home_win", "x": "draw", "2": "away_win"}.get(k)
            v = pred.get(name) if name else None
        elif market == "btts":
            v = pred.get("btts_yes")
            if v is not None and (k.endswith("no") or k == "no"):
                v = 1.0 - v
        elif market == "over_under":
            v = pred.get(f"over_{line}")
            if v is not None and k.startswith("under"):
                v = 1.0 - v
        elif market == "team_goals":
            v = pred.get(f"{(side or '').lower()}_over_{line}")
            if v is not None and "under" in k:
                v = 1.0 - v
        else:
            v = None
        if v is None:
            return None
        out.append(float(v))
    total = sum(out)
    if total <= 0:
        return None
    return [x / total for x in out]


def pool(sources: Dict[str, Optional[Sequence[float]]],
         weights: Dict[str, float]) -> Optional[List[float]]:
    """Linear opinion pool over one leg set, renormalised.

    The market is REQUIRED: with no anchor there is no blend, and None is
    returned rather than a model-only number. Any other source with no
    opinion is dropped and the remaining weights renormalised.
    """
    market = sources.get("market")
    if market is None:
        return None
    present = {k: v for k, v in sources.items()
               if v is not None and weights.get(k, 0.0) > 0.0}
    total = sum(weights[k] for k in present)
    if total <= 0:
        return None
    n = len(market)
    out = [sum(weights[k] * present[k][i] for k in present) / total
           for i in range(n)]
    s = sum(out)
    return [x / s for x in out]


def agrees(sources: Dict[str, Optional[Sequence[float]]], index: int) -> bool:
    """Every source WITH an opinion ranks leg `index` strictly first, and at
    least two sources have one. A tie for first is not agreement."""
    seen = 0
    for v in sources.values():
        if v is None:
            continue
        seen += 1
        top = max(v)
        if v[index] != top or sum(1 for x in v if x == top) != 1:
            return False
    return seen >= 2


@dataclass(frozen=True)
class Leg:
    fixture: int
    family: str
    key: Tuple            # (market, line, side, selection key)
    taken: float
    p: float              # blended probability
    agree: bool
    fair: float = 0.0     # the anchor's Shin probability
    books: int = 0
    feed: str = ""
    selection: str = ""   # the pipeline's canonical name, a key of PICKABLE


def legs_from_sources(fixture: int, market: str, line: Optional[str],
                      side: Optional[str], keys: Sequence[str],
                      sources: Dict[str, Optional[Sequence[float]]],
                      taken: Sequence[float], fair: Sequence[float],
                      books: int, feed: str,
                      weights: Dict[str, Dict[str, float]]) -> List[Leg]:
    """The PICKABLE legs of one priced set, blended. The one place a `Leg`
    is made: the backtest and the flag both come through here."""
    fam = family_of(market, line)
    if fam is None:
        return []
    blended = pool(sources, weights.get(fam, {"market": 1.0}))
    if blended is None:
        return []
    out: List[Leg] = []
    for i, key in enumerate(keys):
        sel = pipeline_selection(market, key)
        if sel is None:
            continue
        out.append(Leg(
            fixture=fixture, family=fam, key=(market, line, side, key),
            taken=taken[i], p=blended[i], agree=agrees(sources, i),
            fair=fair[i], books=books, feed=feed, selection=sel))
    return out


def sources_for(market: str, line: Optional[str], side: Optional[str],
                keys: Sequence[str], fair: Sequence[float],
                model_preds: Dict[str, Optional[dict]]
                ) -> Dict[str, Optional[Sequence[float]]]:
    """The market's and each model's probabilities for one set."""
    fam = family_of(market, line)
    sources: Dict[str, Optional[Sequence[float]]] = {"market": list(fair)}
    for name in FAMILY_SOURCES.get(fam, ()):
        if name != "market":
            sources[name] = model_set(market, line, side, keys,
                                      model_preds.get(name))
    return sources


def legs_for_fixture(fixture: int, priced: Dict[Tuple, dict],
                     model_preds: Dict[str, Optional[dict]],
                     weights: Dict[str, Dict[str, float]]) -> List[Leg]:
    """Every pickable leg of one fixture, with its blended probability.

    `priced` is `slots()`; `model_preds` maps a source name (`elo`,
    `poisson`) to that model's `predict()` dict, or None when the model has
    no opinion on this fixture; `weights` maps a family to its weights.
    """
    out: List[Leg] = []
    for (market, line, side), sl in priced.items():
        if family_of(market, line) is None:
            continue
        sources = sources_for(market, line, side, sl["keys"], sl["fair"],
                              model_preds)
        out.extend(legs_from_sources(
            fixture, market, line, side, sl["keys"], sources, sl["taken"],
            sl["fair"], sl["books"], sl["feed"], weights))
    return out


def weights_from_config(config) -> Dict[str, Dict[str, float]]:
    """`betting.confident_value.weights`, validated. Absent -> the market
    alone. A malformed vector is REFUSED: a weight on a source the family
    does not have, a negative weight, or a sum that is not one would each
    silently change the probability the rule selects on."""
    raw = config.get("betting.confident_value.weights", None)
    if not raw:
        return {fam: {"market": 1.0} for fam in FAMILY_SOURCES}
    out: Dict[str, Dict[str, float]] = {}
    for fam, vec in raw.items():
        if fam not in FAMILY_SOURCES:
            raise ValueError(f"confident_value.weights: unknown family {fam!r}")
        vec = {k: float(v) for k, v in vec.items()}
        extra = set(vec) - set(FAMILY_SOURCES[fam])
        if extra:
            raise ValueError(
                f"confident_value.weights[{fam}]: {sorted(extra)} has no "
                f"opinion on this family")
        if any(v < 0 for v in vec.values()) \
                or abs(sum(vec.values()) - 1.0) > 1e-6:
            raise ValueError(
                f"confident_value.weights[{fam}] must be non-negative and "
                f"sum to 1, got {vec}")
        out[fam] = vec
    for fam in FAMILY_SOURCES:
        out.setdefault(fam, {"market": 1.0})
    return out


# ═══════════════════════════ the selection rule ═════════════════════════════

@dataclass(frozen=True)
class Rule:
    name: str
    lo: float = MIN_PRICE
    hi: Optional[float] = None       # None = unbounded above
    hi_closed: bool = False
    p_min: Optional[float] = None    # qualify on blended probability
    edge: Optional[float] = None     # qualify on blended p x taken >= 1 + e
    require_agreement: bool = False
    lane: str = ""

    def qualifies(self, leg: Leg) -> bool:
        if leg.taken < self.lo:
            return False
        if self.hi is not None and (
                leg.taken > self.hi
                or (leg.taken == self.hi and not self.hi_closed)):
            return False
        if self.require_agreement and not leg.agree:
            return False
        if self.p_min is not None and leg.p < self.p_min:
            return False
        if self.edge is not None and leg.p * leg.taken < 1.0 + self.edge:
            return False
        return True


def _rank(leg: Leg):
    return (-leg.p, -leg.taken, tuple(str(x) for x in leg.key))


def select(legs: Iterable[Leg], rule: Rule) -> List[Leg]:
    """ONE pick per fixture: the qualifying leg with the highest blended
    probability; ties to the higher taken price, then to the key. An empty
    list is a valid answer -- a day on which nothing qualifies."""
    best: Dict[int, Leg] = {}
    for leg in legs:
        if not rule.qualifies(leg):
            continue
        cur = best.get(leg.fixture)
        if cur is None or _rank(leg) < _rank(cur):
            best[leg.fixture] = leg
    return [best[f] for f in sorted(best)]


def registered_grid() -> List[Rule]:
    """The 26 cells of the pre-registration, in its order."""
    rules: List[Rule] = []
    for agree in (False, True):
        tag = "agree" if agree else "all"
        for p in P_MIN_GRID:
            rules.append(Rule(f"C p>={p:.2f} {tag}", lo=MIN_PRICE, p_min=p,
                              require_agreement=agree, lane="C"))
    for agree in (False, True):
        tag = "agree" if agree else "all"
        for p in P_MIN_GRID:
            rules.append(Rule(f"L1 p>={p:.2f} {tag}", lo=1.70, hi=2.00,
                              p_min=p, require_agreement=agree,
                              lane=LANE_CONFIDENT))
    for agree in (False, True):
        tag = "agree" if agree else "all"
        for e in EDGE_GRID:
            rules.append(Rule(f"L2 e>={e:.2f} {tag}", lo=2.00, hi=3.50,
                              hi_closed=True, edge=e,
                              require_agreement=agree, lane=LANE_VALUE))
    return rules


def rule_from_config(config) -> Rule:
    """The rule the flag would run. Every number comes from the config."""
    g = lambda k, d: config.get(f"betting.confident_value.{k}", d)  # noqa: E731
    return Rule(
        name="confident_value (config)",
        lo=float(g("min_price", MIN_PRICE)),
        p_min=float(g("p_min", P_MIN_GRID[0])),
        require_agreement=bool(g("require_agreement", False)),
        lane=LANE_CONFIDENT)


def mode_of(config) -> str:
    """`betting.selection_mode`, defaulting to `standard`. An unknown value
    is REFUSED rather than read as either mode: a typo must not silently
    select a rule."""
    mode = config.get("betting.selection_mode", MODE_STANDARD) or MODE_STANDARD
    if mode not in MODES:
        raise ValueError(
            f"betting.selection_mode={mode!r} is not one of {MODES}")
    return mode
