"""The h1-collect form must be able to fund what its own fields promise.

ON 2026-11-03 IT COULD NOT. A manual dispatch took the form's defaults —
`max_credits=6`, `rounds=3` — and three rounds over the two pinned leagues
that had fixtures costs 3 x 2 x 2 = 12 credits. The job spent 6, completed
TWO rounds, and left nine Champions League fixtures holding TWO instants at a
123-minute gap. A qualifying triple needs three.

That was the window's only in-band day, and it was spent discovering that the
form's `max_credits` default and its `rounds` default were never checked
against each other.

THE CEILING IS NOW DERIVED, not stated: `rounds x pinned_leagues x
CREDITS_PER_REQUEST`. An empty value derives it; a value BELOW the floor is
refused with the arithmetic, because an under-funded run does not collect
less — it collects NOTHING that qualifies, and spends the day to do it.

A league with no in-band fixture is skipped at zero cost, so the floor is a
CEILING and not a forecast: deriving 24 does not spend 24.
"""

import pathlib
import re

import pytest
import yaml

WF = pathlib.Path(".github/workflows/h1-collect.yml")
SRC = WF.read_text(encoding="utf-8")
DOC = yaml.safe_load(SRC)
INPUTS = (DOC.get("on") or DOC.get(True))["workflow_dispatch"]["inputs"]

#: `CREDITS_PER_REQUEST` in src/data/odds_quota. Pinned here because the
#: workflow's shell arithmetic cannot import it, so the two must be asserted
#: equal rather than assumed.
CPR = 2


def test_the_CREDIT_FACTOR_MATCHES_THE_CODE():
    from src.data.odds_quota import CREDITS_PER_REQUEST
    assert CREDITS_PER_REQUEST == CPR, (
        f"the workflow multiplies by {CPR} and the code charges "
        f"{CREDITS_PER_REQUEST}; the derived ceiling would be wrong")
    assert f"* {CPR} ))" in SRC or f"* {CPR} )" in SRC, (
        "the workflow no longer multiplies by the credit factor")


def test_max_credits_DEFAULT_IS_EMPTY_so_it_can_be_DERIVED():
    """A stated default cannot track `rounds`; an empty one can."""
    assert INPUTS["max_credits"].get("default") == "", (
        f"max_credits defaults to "
        f"{INPUTS['max_credits'].get('default')!r}. A literal default cannot "
        f"follow the rounds field, which is exactly how 6 met 3 on 11-03.")
    assert "LEAVE EMPTY" in INPUTS["max_credits"]["description"]


def test_the_WORKFLOW_DERIVES_THE_FLOOR_AND_REFUSES_BELOW_IT():
    assert "LEAGUE_N=" in SRC and "NEED=$(( ROUNDS * LEAGUE_N" in SRC, (
        "the floor is not derived from rounds and the league count")
    assert "REFUSING TO START UNDER-FUNDED" in SRC, (
        "an explicitly under-funded max_credits is not refused")
    assert "123-minute" in SRC or "TWO instants" in SRC, (
        "the comment no longer records WHY this guard exists, so the next "
        "reader may remove it as belt-and-braces")


def _decide(max_credits, rounds, leagues):
    """The workflow's arithmetic, EVALUATED FROM ITS OWN TEXT.

    NOT shelled out. `subprocess` resolves `bash` to WSL on the development
    machine, where it has no `/bin/bash`, so a shell-based test SKIPS locally
    and RUNS in CI -- the same environment divergence that cost the card this
    stage is fixing. A skipped known-wrong control proves nothing.

    Instead the three lines are located in the workflow source and asserted
    to say what this function does. If the workflow changes, the assertions
    below fail rather than this mirror drifting away from it.
    """
    for frag in ("LEAGUE_N=$(printf", "tr ','", "grep -c"):
        assert frag in SRC, (
            f"the league count no longer derives via {frag!r}; this "
            f"mirror of the shell arithmetic would be wrong")
    assert f"NEED=$(( ROUNDS * LEAGUE_N * {CPR} ))" in SRC, (
        "the floor formula changed; this mirror is now wrong")
    assert 'if [ -z "$MAXC" ]; then' in SRC, "the empty-derives branch is gone"
    assert 'elif [ "$MAXC" -lt "$NEED" ]; then' in SRC, (
        "the below-floor refusal branch is gone")

    league_n = len([x for x in leagues.split(",") if x.strip()])
    need = rounds * league_n * CPR
    if max_credits == "":
        return f"DERIVED {need}"
    if int(max_credits) < need:
        return f"REFUSED {need}"
    return f"ACCEPTED {max_credits} {need}"


PINNED = ("england/championship,europe/champions-league,"
          "europe/europa-league,europe/europa-conference-league")


def test_KNOWN_WRONG_the_11_03_COMBINATION_IS_NOW_REFUSED():
    """The exact pair that truncated the window's only in-band day."""
    assert _decide("6", 3, PINNED).startswith("REFUSED"), (
        "max_credits=6 with rounds=3 is still accepted; this is the "
        "combination that produced two instants and no trajectory")


def test_KNOWN_WRONG_twelve_is_ALSO_refused_against_the_PINNED_set():
    """12 was derived from the TWO leagues that happened to fire, not the four
    that are pinned. The ceiling has to cover the pinned set, because which
    leagues carry an in-band fixture is not known when the job starts."""
    assert _decide("12", 3, PINNED).startswith("REFUSED"), (
        "12 is accepted for a 4-league pinned set needing 24; if a third "
        "league has fixtures the run truncates exactly as 11-03 did")


def test_an_EMPTY_VALUE_DERIVES_the_pinned_set_floor():
    assert _decide("", 3, PINNED) == "DERIVED 24"
    assert _decide("", 1, PINNED) == "DERIVED 8"


def test_CONTROL_a_SUFFICIENT_VALUE_IS_ACCEPTED():
    """Without this, 'refuses' could be 'refuses everything'."""
    out = _decide("24", 3, PINNED)
    assert out.startswith("ACCEPTED"), out
    assert _decide("100", 3, PINNED).startswith("ACCEPTED")


def test_the_FLOOR_SCALES_WITH_BOTH_TERMS():
    """It is derived, so changing either field changes it."""
    assert _decide("", 3, "a/b") == "DERIVED 6"
    assert _decide("", 3, "a/b,c/d") == "DERIVED 12"
    assert _decide("", 2, PINNED) == "DERIVED 16"


def test_the_SCHEDULED_PATH_STILL_GETS_A_CEILING():
    """A cron passes no inputs, so every value arrives empty and derived."""
    assert _decide("", 3, PINNED) == "DERIVED 24", (
        "the scheduled path would run with no ceiling at all")
    # and the old hard-coded 24 must be gone, or it can drift from the
    # derivation. A NEGATIVE scan needs its haystack anchored: "MAXC=24 is
    # absent" is an empty claim if SRC is the wrong file or an empty string.
    from tests.conftest import anchored
    hay = anchored(SRC, "MAXC=", "the h1-collect Collect step")
    assert not re.search(r"MAXC=24\b", hay), (
        "a literal MAXC=24 remains; it would not follow a change to rounds")
