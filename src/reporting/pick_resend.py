"""Re-send a daily card Telegram has already received. READ-ONLY BY CONSTRUCTION.

A Telegram message can be deleted by accident, and the picks outlive it: the
card is a VIEW over `saved_picks`, so losing the message does not lose the
information. This module rebuilds that view and hands it back for sending.

WHY THIS IS NOT A ONE-OFF SCRIPT. The first time it was needed the obvious
move was an ad-hoc query and a hand-formatted message. That would have been a
SECOND CARD: the formatter renders fields the store does not hold, so a
hand-built copy silently drops lines and the reader cannot tell a resend from
a correction. The capability therefore carries its own refusal.

    A RESEND THAT DIFFERS IS NOT A COPY. IT IS A SECOND CARD.

So `certify()` runs BEFORE anything is sent and names every field the card
needs that the store cannot supply. The default is REFUSAL. Sending a
reconstruction is possible, but only by explicit opt-in and only labelled as
one, because a reconstruction presented as the original is the worse failure.

WHAT IS RECOVERABLE, AND HOW IT IS KNOWN
----------------------------------------
`contrarian_value` is recovered EXACTLY, not estimated. `value_calculator`
computes `divergence = prob / implied` with `implied = 1 / best_odds`, so the
ratio is `predicted_probability * odds` and BOTH factors are columns. This is
an algebraic identity, not a correlation.

`opening_odds` is NOT recovered. The `odds` table has an `opening_odds`
column, but the pick used ONE bookmaker's price and WHICH BOOK IS NOT STORED.
Taking a max or a mean across books would reproduce the field's TYPE and not
its VALUE, and the movement annotation has a 3% threshold that a wrong book
can cross. A field that is reconstructible to the wrong value is worse than
one left out, because only the second is visible.

`models_for`, `predicted_xg` and `xg_edge` are computed from live features and
the per-model agreement at generation time. Nothing persists them.

WHY `models_for` IS A CERTAIN LOSS AND NOT A GUESS
--------------------------------------------------
Read off the agreement definition in `value_calculator`, not inferred:

    solo      <=> len(models_against) == 0 and len(models_for) == 1
    unanimous <=> len(models_against) == 0 and len(models_for) >= 2
    majority  <=> len(models_for) > len(models_against)      -> >= 1
    split     <=> otherwise                     -> models_for MAY be empty

The first three IMPLY a non-empty `models_for`, and the formatter appends
" - {models_for} agree" whenever both it and `model_agreement` are non-empty.
So for those three the suffix certainly rendered and certainly cannot be
rebuilt. For `split` it is UNDETERMINED - which is reported as undetermined
rather than folded into either side.

THIS MODULE WRITES NOTHING AND SENDS NOTHING. It has no session of its own, no
commit, and does not touch `picks_sent_date.txt`: a resend that moved the
picks-sent marker could unfreeze the review path, whose race has already cost
a day. The census in `row_census()` exists so that claim is MEASURED after the
fact rather than asserted from reading this docstring.
"""

from __future__ import annotations

import datetime as dt
from typing import Dict, List, Tuple

from loguru import logger
from sqlalchemy import text

# Every table, so the no-persistence proof is over the whole schema rather
# than over the tables someone remembered to list.
CENSUS_TABLES = (
    "api_budget", "injuries", "injury_observations", "matches", "odds",
    "odds_snapshots", "pick_observations", "pick_runs", "players",
    "saved_picks", "team_former_names", "teams",
)

# Card fields the store does not hold. Keyed to how the gap is reported.
NOT_STORED = ("models_for", "predicted_xg", "xg_edge", "opening_odds")

# `model_agreement` values that IMPLY a non-empty `models_for` (see module
# docstring; derived from the agreement definition, not sampled).
AGREEMENT_IMPLIES_MODELS_FOR = ("solo", "unanimous", "majority")


def row_census(session, engine=None) -> Dict[str, int]:
    """Row count per table. The before/after pair is the no-persistence proof.

    ENUMERATED FROM THE SCHEMA, not from `CENSUS_TABLES`, when an engine is
    available. A census over a hand-kept list is the vacuous-filter shape one
    level up: it would pass over a write to any table added after the list was
    written, and the list is exactly the kind of thing that goes stale quietly.
    """
    names = tuple(CENSUS_TABLES)
    if engine is not None:
        from sqlalchemy import inspect as _sa_inspect
        found = tuple(_sa_inspect(engine).get_table_names())
        if found:
            names = found
    out = {}
    for t in names:
        out[t] = int(session.execute(text(f'SELECT count(*) FROM "{t}"')).scalar())
    return out


def load_card(db, day: dt.date) -> Tuple[List[object], Dict[str, object]]:
    """Load the saved card for `day`. READS ONLY. Returns `(picks, meta)`.

    `assert db.is_postgres` comes FIRST. An empty SQLite file answers this
    query with a plausible zero, and a resend that sends nothing because it
    silently read the wrong database looks identical to a day with no picks.
    """
    assert db.is_postgres, (
        "REFUSING to build a card from a non-PostgreSQL session. A SQLite "
        "fallback answers the pick query with a plausible empty card, and an "
        "empty resend is indistinguishable from a day that had no picks. "
        "Load the environment first (see scripts/cohort_status.py)."
    )
    from src.betting.value_calculator import BetRecommendation

    picks: List[object] = []
    with db.get_session() as s:
        rows = s.execute(text(
            "SELECT sp.id, sp.match_id, sp.match_name, sp.league, sp.market, "
            "       sp.selection, sp.odds, sp.predicted_probability, "
            "       sp.expected_value, sp.confidence, sp.kelly_stake_percentage, "
            "       sp.risk_level, sp.model_agreement, sp.used_fallback_odds, "
            "       sp.is_paper, sp.review_action, m.match_date "
            "  FROM saved_picks sp LEFT JOIN matches m ON m.id = sp.match_id "
            # EQUALITY ON A `Date` COLUMN, NOT A `::date` CAST AND NOT A
            # DATETIME RANGE. `SavedPick.pick_date` is `Column(Date)`, so the
            # cast is a PostgreSQL-only no-op and a datetime range matches
            # NOTHING on SQLite - "2026-11-02" sorts before
            # "2026-11-02 00:00:00", so `>= midnight` is false for every row.
            # That silent empty result is why this query has to be runnable
            # against the SQLite fixture: the census test caught it, the
            # production cast had hidden it.
            " WHERE sp.pick_date = :d "
            " ORDER BY sp.id"), {"d": day}).all()
        meta = {
            "n": len(rows),
            "paper": bool(rows) and all(bool(r.is_paper) for r in rows),
            "mixed_paper": len({bool(r.is_paper) for r in rows}) > 1,
            "agreements": sorted({(r.model_agreement or "") for r in rows}),
            "matches": len({r.match_name for r in rows}),
            "max_per_match": max(
                [sum(1 for x in rows if x.match_name == r.match_name)
                 for r in rows] or [0]),
        }
        for r in rows:
            # THE DRIVER DECIDES THE TYPE, SO THE TYPE IS PINNED HERE.
            # psycopg2 returns `match_date` as a datetime; the SQLite driver
            # returns the same column as a str, and the card formatter calls
            # `.strftime` on it. Coercing once means the resend does not
            # behave differently on the database the no-write census uses
            # from the one it runs against.
            _ko = r.match_date
            if isinstance(_ko, str):
                try:
                    _ko = dt.datetime.fromisoformat(_ko)
                except ValueError:
                    _ko = None
            picks.append(BetRecommendation(
                match=r.match_name,
                match_id=r.match_id,
                market=r.market,
                selection=r.selection,
                odds=float(r.odds),
                predicted_probability=float(r.predicted_probability),
                expected_value=float(r.expected_value),
                confidence=float(r.confidence),
                kelly_stake_percentage=float(r.kelly_stake_percentage),
                # Not on the card. Set to the stake so no caller reads a zero
                # as "no stake"; the card prints kelly_stake_percentage.
                recommended_stake=float(r.kelly_stake_percentage),
                reasoning="",
                risk_level=r.risk_level or "",
                league=r.league or "",
                model_agreement=r.model_agreement or "",
                used_fallback_odds=bool(r.used_fallback_odds),
                match_date=_ko,
                # EXACT, by the algebraic identity in the module docstring.
                contrarian_value=round(
                    float(r.predicted_probability) * float(r.odds), 2),
                # DELIBERATELY NOT RECONSTRUCTED - see the docstring.
                opening_odds=0.0,
                models_for="",
                predicted_xg="",
                xg_edge="",
            ))
    return picks, meta


def certify(picks: List[object], meta: Dict[str, object]) -> Dict[str, object]:
    """Can this card be reproduced EXACTLY? Returns the verdict and the gaps.

    `reproducible` is only True when no card line can differ. The two kinds of
    gap are kept apart on purpose: `certain` is derived from a definition,
    `undetermined` is honestly unknown, and collapsing them would turn a fact
    into a guess in whichever direction suited the caller.
    """
    certain: Dict[str, int] = {}
    undetermined: Dict[str, int] = {}

    n_implied = sum(1 for p in picks
                    if (p.model_agreement or "") in AGREEMENT_IMPLIES_MODELS_FOR)
    n_split = sum(1 for p in picks
                  if (p.model_agreement or "") not in AGREEMENT_IMPLIES_MODELS_FOR
                  and (p.model_agreement or ""))
    # `> 0` RATHER THAN TRUTHINESS. A bare `if n_implied:` reads as "if any",
    # but it is also the shape that swallows a real zero elsewhere in this
    # codebase, and here zero is a MEANINGFUL answer - no pick rendered the
    # suffix, so nothing is lost. Said explicitly so the two cannot be
    # confused by a later reader.
    if n_implied > 0:
        certain["models_for"] = n_implied
    if n_split > 0:
        undetermined["models_for"] = n_split
    if picks:
        for field in NOT_STORED:
            if field == "models_for":
                continue        # split above into certain / undetermined
            # Never stored. `predicted_xg` and `xg_edge` depend on
            # generation-time xG; `opening_odds` on a bookmaker identity that
            # is not recorded, so the formatter's 3% movement threshold
            # cannot be evaluated either way.
            undetermined[field] = len(picks)

    # NOT_STORED IS THE SINGLE SOURCE, AND THIS IS WHAT MAKES IT ONE. Adding
    # a field to that tuple and forgetting it here would silently drop it
    # from the verdict - a gap the resend then would not report, which is the
    # one failure this module exists to prevent.
    if picks:
        # `models_for` is excluded because it is CONDITIONALLY a gap: the
        # formatter prints no Models line at all when `model_agreement` is
        # blank, so with no agreement recorded nothing is lost and its
        # absence from the verdict is correct. Every OTHER declared field is
        # unconditional and must appear.
        unaccounted = (set(NOT_STORED) - {"models_for"}
                       - set(certain) - set(undetermined))
        assert not unaccounted, (
            f"fields declared NOT_STORED but absent from the verdict: "
            f"{sorted(unaccounted)}. Every declared gap must be reported.")

    return {
        "reproducible": not certain and not undetermined,
        "certain": certain,
        "undetermined": undetermined,
        "recovered_exactly": {"contrarian_value": "predicted_probability * odds"},
        "n": meta.get("n", 0),
    }


def format_verdict(cert: Dict[str, object], day: dt.date) -> str:
    """The refusal, written so the reader can act on it."""
    lines = [
        f"RESEND REFUSED for {day} - the stored card cannot be reproduced "
        f"exactly, and a resend that differs is a second card.",
        f"  picks in store: {cert['n']}",
    ]
    if cert["certain"]:
        lines.append("  CERTAIN to differ (derived from the agreement "
                     "definition, not sampled):")
        for f, n in sorted(cert["certain"].items()):
            lines.append(f"     {f:16s} missing on {n} pick(s) - the card "
                         f"rendered it and the store does not hold it")
    if cert["undetermined"]:
        lines.append("  UNDETERMINED (not stored; cannot be shown either way):")
        for f, n in sorted(cert["undetermined"].items()):
            lines.append(f"     {f:16s} on up to {n} pick(s)")
    lines.append("  recovered EXACTLY: " + ", ".join(
        f"{k} = {v}" for k, v in cert["recovered_exactly"].items()))
    lines.append("  To send the reconstruction anyway, labelled as one: "
                 "--resend <date> --accept-reconstructed")
    return "\n".join(lines)
