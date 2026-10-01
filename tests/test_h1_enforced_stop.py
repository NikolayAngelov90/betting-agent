"""H1 Requirement 1 — the enforced stop, and proof that each limb fires.

STAGE 26, un-suspended 2026-10-01 for this requirement alone.

`TARGET_N` and `CREDIT_CEILING` lived only in `scripts/h1_collection_check.py`,
which REPORTS that the stop condition is met while the runner keeps spending. On a
monthly budget that is not a stop. H1 is sized at 106-168 credits of a 450-credit
month, and an unbounded collection loop empties it in one run with no retry until
the next reset.

WHY THE CONTROLS MATTER MORE THAN USUAL: **this stop cannot be tested in
production.** Firing it for real costs the experiment's budget, and the budget is
monthly. A stop that has never been observed to fire is a stop that has not been
tested, so both limbs are driven here — `n` to `TARGET_N` with injected
observations, and the credit counter to `CREDIT_CEILING` — without spending a
real credit.

THE THREE TERMINAL STATES ARE NOT INTERCHANGEABLE, and the suite asserts they are
distinguishable in the log, because that is the whole point of registering them:

    COMPLETE     got the data
    CEILING_HIT  ran out of budget first
    NO_DATA      spent the budget and collected nothing
"""

import asyncio
import datetime as dt

import pytest

from scripts.h1_collection_check import (
    COLLECTION_RUNNING,
    COLLECTION_STATES,
    COLLECTION_TERMINAL_STATES,
    CREDIT_CEILING,
    TARGET_N,
    CollectionStop,
    collection_stop,
)


# ── the outcomes are REGISTERED, in the type that emits them ────────────────

def test_the_states_are_registered_and_exhaustive():
    assert COLLECTION_STATES == ("COLLECTING", "COMPLETE", "CEILING_HIT", "NO_DATA")
    assert COLLECTION_RUNNING not in COLLECTION_TERMINAL_STATES
    assert len(COLLECTION_TERMINAL_STATES) == 3


def test_every_reachable_state_is_in_the_registry():
    """No decision may return a state the registry does not name."""
    seen = set()
    for n in (0, TARGET_N - 1, TARGET_N):
        for c in (0, CREDIT_CEILING - 1, CREDIT_CEILING, None):
            for raw in (0, 500):
                seen.add(collection_stop(n_fixtures=n, credits_spent=c,
                                         raw_rows=raw).state)
    assert seen <= set(COLLECTION_STATES), seen - set(COLLECTION_STATES)


# ── POSITIVE CONTROL, LIMB 1: n reaches TARGET_N ───────────────────────────

def test_POSITIVE_CONTROL_limb1_n_reaches_TARGET_N_and_HALTS():
    """Drive n to the target and confirm the halt. The limb fires."""
    below = collection_stop(n_fixtures=TARGET_N - 1, credits_spent=10,
                            raw_rows=500)
    assert below.halt is False and below.state == "COLLECTING"

    at = collection_stop(n_fixtures=TARGET_N, credits_spent=10, raw_rows=500)
    assert at.halt is True, "n reached TARGET_N and collection did NOT halt"
    assert at.state == "COMPLETE"
    assert str(TARGET_N) in at.reason


def test_limb1_is_GREATER_OR_EQUAL_not_equality():
    """`n == TARGET_N + 1` must still halt.

    An `==` test skips the stop whenever a slot adds two trajectories at once,
    and the loop runs on past the target. The off-by-one that does not announce
    itself.
    """
    assert collection_stop(n_fixtures=TARGET_N + 5, credits_spent=0,
                           raw_rows=500).halt is True


# ── POSITIVE CONTROL, LIMB 2: credits reach CREDIT_CEILING ─────────────────

def test_POSITIVE_CONTROL_limb2_credits_reach_the_CEILING_and_HALT():
    """Drive the credit counter to the ceiling. No real credit is spent."""
    below = collection_stop(n_fixtures=1, credits_spent=CREDIT_CEILING - 1,
                            raw_rows=500)
    assert below.halt is False and below.state == "COLLECTING"

    at = collection_stop(n_fixtures=1, credits_spent=CREDIT_CEILING,
                         raw_rows=500)
    assert at.halt is True, "credits reached the ceiling and collection did NOT halt"
    assert at.state == "CEILING_HIT"
    assert "SHORT" in at.reason, (
        "CEILING_HIT does not say the collection stopped short of its target — "
        "a reader would file it as a normal completion")


def test_limb2_FAILS_CLOSED_on_an_unreadable_ledger():
    """`None` credits halt. A collection that cannot see its spend must not spend.

    The alternative is `None >= 200`, which raises in Python 3 — or, worse, a
    guard written as `if credits and credits >= CEILING`, which silently passes.
    """
    s = collection_stop(n_fixtures=1, credits_spent=None, raw_rows=500)
    assert s.halt is True
    assert "UNREADABLE" in s.reason


# ── the third terminal state ────────────────────────────────────────────────

def test_NO_DATA_is_not_COMPLETE_and_not_CEILING_HIT():
    """Spent the budget, collected nothing — a different fact from both."""
    s = collection_stop(n_fixtures=0, credits_spent=CREDIT_CEILING, raw_rows=0)
    assert s.halt is True
    assert s.state == "NO_DATA"
    assert s.state not in ("COMPLETE", "CEILING_HIT")
    assert "not working" in s.reason


def test_NO_DATA_refines_a_halt_and_does_not_TRIGGER_one():
    """Zero observations early in collection is not yet a failure.

    Halting the moment `raw_rows == 0` would stop the collection before its
    first slot had written anything.
    """
    s = collection_stop(n_fixtures=0, credits_spent=0, raw_rows=0)
    assert s.halt is False and s.state == "COLLECTING"


def test_the_three_terminal_states_PRINT_DIFFERENTLY():
    """Distinguishable in the log, not merely in the enum."""
    msgs = {
        collection_stop(n_fixtures=TARGET_N, credits_spent=0, raw_rows=9).state:
            str(collection_stop(n_fixtures=TARGET_N, credits_spent=0, raw_rows=9)),
        collection_stop(n_fixtures=0, credits_spent=CREDIT_CEILING, raw_rows=9).state:
            str(collection_stop(n_fixtures=0, credits_spent=CREDIT_CEILING, raw_rows=9)),
        collection_stop(n_fixtures=0, credits_spent=CREDIT_CEILING, raw_rows=0).state:
            str(collection_stop(n_fixtures=0, credits_spent=CREDIT_CEILING, raw_rows=0)),
    }
    assert set(msgs) == set(COLLECTION_TERMINAL_STATES)
    assert len(set(msgs.values())) == 3, "two terminal states print the same line"
    for state, line in msgs.items():
        assert f"state={state}" in line


# ── ONE DEFINITION: the runner imports, it does not copy ───────────────────

def test_the_runner_IMPORTS_the_constants_and_keeps_no_copy():
    import pathlib
    src = pathlib.Path("src/scrapers/theodds_scraper.py").read_text(encoding="utf-8")
    assert "from scripts.h1_collection_check import collection_state" in src, (
        "the runner does not import the stop — if it re-derived it, the check "
        "and the runner can disagree about when to stop")
    code = "\n".join(l for l in src.splitlines() if not l.strip().startswith("#"))
    for const in ("TARGET_N", "CREDIT_CEILING"):
        assert f"{const} =" not in code, (
            f"{const} is assigned inside the runner — that is a second copy of "
            f"a REGISTERED constant, and changing one changes the experiment")


# ── the flag cannot be forgotten ────────────────────────────────────────────

def _scraper(with_db: bool = False):
    """A scraper built past `__init__`, so no network and no credits.

    `with_db` attaches an in-memory SQLite manager for the paths that query
    before returning. The DB is empty, which is all these tests need — none of
    them asserts a count, only which branch was taken.
    """
    from src.utils.config import Config
    import src.scrapers.theodds_scraper as ts
    sc = ts.TheOddsScraper.__new__(ts.TheOddsScraper)
    sc.config = Config("config/config.example.yaml")
    sc.api_key = None                      # no network, no credits
    if with_db:
        import src.data.database as db_mod
        from src.data.models import Base
        mgr = db_mod.DatabaseManager(
            config=type("C", (), {"database": {"sqlite_path": ":memory:"}})())
        Base.metadata.create_all(mgr.engine)
        sc.db = mgr
    return sc


@pytest.mark.parametrize("window,interval", [(360, 180), (120, 120), (360, 120)])
def test_H1_SHAPED_parameters_without_the_flag_are_REFUSED(window, interval):
    """The stop is collection-scoped, so it needs a flag — and a flag is a single
    point of forgetting unless the parameters themselves demand it."""
    sc = _scraper()
    with pytest.raises(ValueError, match="h1_collection=False"):
        asyncio.run(sc.refresh_imminent(window_minutes=window,
                                        min_interval_minutes=interval))


def test_ORDINARY_parameters_are_UNAFFECTED():
    """Normal pricing must not acquire a 200-credit global ceiling.

    `api_key = None` makes this return before any network call, which is the
    point: the H1 guard did not intercept it.
    """
    sc = _scraper()
    plan = asyncio.run(sc.refresh_imminent(window_minutes=120,
                                           min_interval_minutes=180))
    assert plan["skipped"]["*"] == "ODDS_API_KEY not configured"
    assert "h1_stop" not in plan, (
        "the H1 stop was evaluated on an ordinary refresh — the ceiling is a "
        "bound on the collection, not on the month's normal operation")


def test_the_runner_REFUSES_TO_SPEND_when_the_stop_cannot_be_evaluated():
    """Fail closed: an unevaluable stop is not a stop that passed."""
    import src.scrapers.theodds_scraper as ts
    sc = _scraper()
    sc.api_key = "x"                        # get past the key check

    import scripts.h1_collection_check as chk
    orig = chk.collection_state
    chk.collection_state = lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("ledger down"))
    try:
        plan = asyncio.run(sc.refresh_imminent(
            window_minutes=360, min_interval_minutes=120, h1_collection=True))
    finally:
        chk.collection_state = orig
    assert plan["halted"] is True
    assert plan["h1_stop"] == "UNEVALUATED"
    assert plan["credits_claimed"] == 0 and plan["odds_written"] == 0


def test_a_DRY_RUN_is_exempt_from_the_refusal():
    """A dry run returns above `_fetch_and_persist`, so it cannot spend.

    The first version of the shape check refused `test_dry_run_spends_nothing`
    (window=1440, interval=0) — the existing suite caught a guard that would have
    broken a legitimate caller. Exempting it is correct on the merits, not a
    workaround: a stop that guards spending has nothing to guard here.
    """
    sc = _scraper(with_db=True)
    sc.api_key = "x"
    plan = asyncio.run(sc.refresh_imminent(
        window_minutes=1440, min_interval_minutes=0, dry_run=True))
    assert plan["dry_run"] is True
    assert plan["credits_claimed"] == 0
    assert "h1_stop" not in plan


def test_the_shape_test_stays_BROAD_not_pinned_to_360_slash_120():
    """Fail closed: a caller inventing its own wide window also spends unbounded.

    The registered H1 pair is 360/120, but refusing only that pair would let
    window=1440 through — which is MORE spending, not less.
    """
    sc = _scraper()
    for window, interval in ((1440, 180), (400, 180), (120, 60), (120, 1)):
        with pytest.raises(ValueError):
            asyncio.run(sc.refresh_imminent(window_minutes=window,
                                            min_interval_minutes=interval))
