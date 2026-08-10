"""Dual CLV attribution — the frozen model's selection vs the final persisted one.

Stage 9.

Why two series
--------------
A saved pick can carry two different bets. The frozen Stage 5 model chooses one;
the optional Claude KEEP/CHANGE review may then overwrite ``market`` and
``selection`` on the row. Measuring CLV from the persisted selection alone
answers "how did model + review do?", which is not the question the experiment
was built around:

    Series A (``model``) — did the FROZEN model identify selections whose prices
                           moved in the expected direction before closing?
    Series B (``final``) — did the selection actually taken move that way?

These are different questions and must never be silently merged. Every
observation therefore names its attribution explicitly. There is no default and
no "current"/"original"/"saved" — a reader must not have to guess which bet a
number refers to.

What this module is
-------------------
The pure, schema-independent core: given a pick row, it says what each series is
betting on, at what price, and — when a series cannot be measured — exactly why.
It computes nothing about closing prices itself; the Stage 8 validity rules stay
where they are, and each series is put through them independently.

Unavailable is not a failure
----------------------------
``unavailable`` means "this row cannot tell us what the model picked", not "the
model picked badly". 999 of the 1,070 historical picks predate the
``model_selection`` snapshot entirely. Folding those into a CLV denominator
would silently report the final series as if it were the model's.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

#: The two attribution series. Deliberately verbose values — these strings end
#: up in reports and logs, where "model"/"final" must be self-explanatory.
MODEL = "model"
FINAL = "final"
ATTRIBUTIONS = (MODEL, FINAL)

#: Why a series cannot be measured. Distinct codes, because they call for
#: different responses and must never be aggregated into one "missing" bucket.
#:
#: NO_MODEL_SNAPSHOT     the row predates model_market/model_selection. Nothing
#:                       is wrong with the pick; the record is simply silent.
#: MODEL_PRICE_NOT_KEPT  a Claude CHANGE overwrote SavedPick.odds with the new
#:                       selection's price, and the model selection's own taken
#:                       price was never stored anywhere. Without a taken price
#:                       there is no CLV to compute — CLV is taken/closing - 1.
#: NO_TAKEN_PRICE        the row has no usable decimal price at all.
#: NO_SELECTION          market/selection missing.
NO_MODEL_SNAPSHOT = "no_model_snapshot"
MODEL_PRICE_NOT_KEPT = "model_taken_price_not_recorded"
NO_TAKEN_PRICE = "no_taken_price"
NO_SELECTION = "no_selection"


@dataclass(frozen=True)
class SeriesSpec:
    """What one attribution series is betting on, and at what price.

    ``taken_odds`` is the price at the moment of the pick — the numerator of
    CLV. It is not recoverable after the fact: the odds table holds exactly one
    row per (match, bookmaker, market, selection) and every refresh overwrites
    it, so there is no price history to look back into.
    """

    attribution: str
    market: Optional[str] = None
    selection: Optional[str] = None
    taken_odds: Optional[float] = None
    unavailable_reason: Optional[str] = None

    @property
    def measurable(self) -> bool:
        """Whether this series has everything CLV needs from the pick side.

        A measurable series can still end up ``missing``/``late``/``invalid``
        once the Stage 8 closing rules are applied — that is a separate step and
        a different fact about the world.
        """
        return self.unavailable_reason is None

    def describe(self) -> str:
        if self.measurable:
            return (f"{self.attribution}: {self.market} / {self.selection} "
                    f"@ {self.taken_odds}")
        return f"{self.attribution}: unavailable ({self.unavailable_reason})"


def _value(row, *names):
    """First present attribute among ``names`` (rows come as ORM objects, Row
    tuples or the report's flat _Pick view, which name things differently)."""
    for n in names:
        if hasattr(row, n):
            v = getattr(row, n)
            if v is not None:
                return v
    return None


def resolve(pick) -> Tuple[SeriesSpec, SeriesSpec]:
    """(model_spec, final_spec) for one saved pick.

    Never raises and never guesses. A series that cannot be measured comes back
    with a reason code rather than a plausible-looking substitute — substituting
    the final selection's price for the model's is the single mistake that would
    make the whole experiment answer the wrong question while looking healthy.
    """
    final_market = _value(pick, "market")
    final_selection = _value(pick, "selection")
    final_odds = _value(pick, "odds", "taken_odds")

    if not final_selection or not final_market:
        final = SeriesSpec(FINAL, unavailable_reason=NO_SELECTION)
    elif not final_odds or float(final_odds) <= 1.0:
        final = SeriesSpec(FINAL, final_market, final_selection,
                           unavailable_reason=NO_TAKEN_PRICE)
    else:
        final = SeriesSpec(FINAL, final_market, final_selection,
                           float(final_odds))

    model_market = _value(pick, "model_market")
    model_selection = _value(pick, "model_selection")

    if not model_selection or not model_market:
        # Historical rows written before the snapshot columns existed. The final
        # series is still measurable; the model series is simply unrecorded.
        return SeriesSpec(MODEL, unavailable_reason=NO_MODEL_SNAPSHOT), final

    if model_selection == final_selection and model_market == final_market:
        # The review kept the pick (or never ran), so SavedPick.odds IS the
        # price the model was quoted. One bet, two attributions.
        if final.measurable:
            return SeriesSpec(MODEL, model_market, model_selection,
                              final.taken_odds), final
        return SeriesSpec(MODEL, model_market, model_selection,
                          unavailable_reason=NO_TAKEN_PRICE), final

    # A genuine CHANGE: the review replaced the selection AND the price. The
    # model selection's taken price is gone — `_apply_decision` assigns
    # `primary.odds = float(new.odds)` and nothing preserves the old value.
    # model_probability survives, but a probability is not a price.
    return SeriesSpec(MODEL, model_market, model_selection,
                      unavailable_reason=MODEL_PRICE_NOT_KEPT), final


def shares_one_observation(model_spec: SeriesSpec, final_spec: SeriesSpec) -> bool:
    """Whether both series ride on the SAME underlying market observation.

    True when the review kept the model's pick. The closing price must then be
    captured once and attributed twice: two API-quota-free counters over one
    fact, never two capture operations and never two independent fixtures in the
    statistics (Stage 9, sections 6 and 13).
    """
    return (model_spec.measurable and final_spec.measurable
            and model_spec.market == final_spec.market
            and model_spec.selection == final_spec.selection)


def coverage_class(model_spec: SeriesSpec, final_spec: SeriesSpec) -> str:
    """Cross-tab bucket for section 12's coverage table, from the pick side only.

    Says nothing about whether a closing price was found — only whether the row
    carries what each series would need.
    """
    m, f = model_spec.measurable, final_spec.measurable
    if m and f:
        return "both_measurable" if not shares_one_observation(
            model_spec, final_spec) else "both_measurable_same_selection"
    if m:
        return "model_only_measurable"
    if f:
        return "final_only_measurable"
    return "neither_measurable"


def selection_changed(pick) -> Optional[bool]:
    """True/False when the snapshot allows the comparison, None when it does not.

    Explicitly tri-state: a missing snapshot is not evidence that the selection
    was unchanged, and returning False there would quietly inflate the
    "model == final" population by 999 rows.
    """
    model_selection = _value(pick, "model_selection")
    if not model_selection:
        return None
    return model_selection != _value(pick, "selection")
