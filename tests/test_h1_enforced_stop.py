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
    assert COLLECTION_STATES == ("COLLECTING", "COMPLETE", "CEILING_HIT",
                                 "NO_DATA", "RESERVATION_EXHAUSTED")
    assert COLLECTION_RUNNING not in COLLECTION_TERMINAL_STATES
    assert len(COLLECTION_TERMINAL_STATES) == 4


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
    """Drive the credit counter to the REGISTERED ceiling. No real credit spent.

    `reservation` is raised ABOVE the ceiling here to isolate this limb. In
    production the reservation (168) is smaller and binds first — that is
    `RESERVATION_EXHAUSTED`, tested separately. Isolating the limb is what makes
    this a control for the ceiling rather than for whichever bound happens to be
    lower.
    """
    high = CREDIT_CEILING + 100
    below = collection_stop(n_fixtures=1, credits_spent=CREDIT_CEILING - 1,
                            raw_rows=500, reservation=high)
    assert below.halt is False and below.state == "COLLECTING"

    at = collection_stop(n_fixtures=1, credits_spent=CREDIT_CEILING,
                         raw_rows=500, reservation=high)
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


def test_the_terminal_states_PRINT_DIFFERENTLY():
    """Distinguishable in the log, not merely in the enum.

    All four are reachable only when the reservation and the ceiling are
    DIFFERENT numbers, which is the production arrangement (168 < 200).
    """
    res = 168
    cases = [
        dict(n_fixtures=TARGET_N, credits_spent=0, raw_rows=9),
        dict(n_fixtures=0, credits_spent=res, raw_rows=9),
        dict(n_fixtures=0, credits_spent=CREDIT_CEILING, raw_rows=9),
        dict(n_fixtures=0, credits_spent=res, raw_rows=0),
    ]
    msgs = {}
    for kw in cases:
        s = collection_stop(reservation=res, **kw)
        msgs[s.state] = str(s)
    assert set(msgs) == set(COLLECTION_TERMINAL_STATES), set(msgs)
    assert len(set(msgs.values())) == 4, "two terminal states print the same line"
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


# ═══════════════ THE ALLOCATOR (Stage 26, 2026-10-01) ════════════════════════
#
# Two independent ceilings over one exhaustible pool are not a budget. These
# assert there is ONE allocation, that each consumer halts on its own side of it,
# and that neither can borrow from the other.

import datetime as _d

from src.data.odds_quota import (
    DEFAULT_MONTHLY_BUDGET,
    DEFAULT_SAFETY_MARGIN,
    H1_RESERVATION_FROM,
    H1_RESERVATION_UNTIL,
    H1_RESERVED_CREDITS,
    allocate,
    h1_reservation,
)

IN_WINDOW = _d.date(2026, 10, 10)
BEFORE_WINDOW = _d.date(2026, 10, 8)
AFTER_WINDOW = _d.date(2026, 10, 17)


def test_there_is_ONE_allocation_and_the_remainder_is_COMPUTED():
    """`normal_ceiling` must be the remainder, never a second constant."""
    a = allocate(450, 50, today=IN_WINDOW)
    assert a["pool"] == 400
    assert a["h1_reserved"] == H1_RESERVED_CREDITS == 168
    assert a["normal_ceiling"] == 400 - 168 == 232
    assert a["h1_reserved"] + a["normal_ceiling"] == a["pool"], (
        "the two sides do not sum to the pool — then one of them is an "
        "independent ceiling again")


def test_the_reservation_is_BY_CALENDAR_not_by_a_flag():
    """A switch someone must unset would throttle pricing every month after."""
    assert h1_reservation(BEFORE_WINDOW) == 0
    assert h1_reservation(IN_WINDOW) == H1_RESERVED_CREDITS
    assert h1_reservation(AFTER_WINDOW) == 0, (
        "the reservation outlives its window — SUP-1, on the credit budget")
    assert (H1_RESERVATION_UNTIL - H1_RESERVATION_FROM).days == 8


def test_outside_the_window_normal_operation_gets_the_WHOLE_pool():
    a = allocate(450, 50, today=BEFORE_WINDOW)
    assert a["h1_reserved"] == 0 and a["normal_ceiling"] == a["pool"] == 400


def test_the_reservation_is_SMALLER_than_the_registered_ceiling():
    """Which is why RESERVATION_EXHAUSTED exists and binds first."""
    from scripts.h1_collection_check import CREDIT_CEILING
    assert H1_RESERVED_CREDITS < CREDIT_CEILING


# ── POSITIVE CONTROL 1: normal operation halts at the remainder ─────────────

class _Store:
    """Minimal ledger stand-in: `used` is injected, `remaining` honours reserve."""

    def __init__(self, used, budget):
        self._used, self._budget = used, budget

    def available(self):
        return True

    def used(self, day=None):
        return self._used

    def remaining(self, reserve=0, day=None):
        return self._budget - reserve - self._used


def _quota(used, for_h1=False, budget=450, margin=50):
    from src.data.odds_quota import OddsApiQuota
    q = OddsApiQuota.__new__(OddsApiQuota)
    q.for_h1 = for_h1
    q.monthly_budget, q.safety_margin = budget, margin
    q.max_credits_per_run, q.spent_this_run = 0, 0
    q._store = _Store(used, budget)
    return q


def test_POSITIVE_CONTROL_normal_operation_HALTS_at_its_remainder():
    """Drive normal spend to the remainder and confirm it stops with H1's
    reservation untouched. The limb fires."""
    # 231 of 232 spent: one request (2 credits) still fits? 232-231 = 1 < 2.
    below = _quota(used=200).remaining(today=IN_WINDOW)
    assert below == 232 - 200 == 32, below

    at = _quota(used=232).remaining(today=IN_WINDOW)
    assert at == 0, (
        f"normal operation still has {at} spendable credits after reaching its "
        f"remainder ceiling — it is eating H1's reservation")
    assert _quota(used=232).max_requests(today=IN_WINDOW) == 0

    # AND THE RESERVATION IS INTACT: the pool still holds 168 unspent.
    assert 400 - 232 == H1_RESERVED_CREDITS


def test_normal_operation_is_UNTHROTTLED_outside_the_window():
    """The throttle must cost nothing on the other 23 days of the month."""
    assert _quota(used=232).remaining(today=BEFORE_WINDOW) == 400 - 232 == 168


# ── POSITIVE CONTROL 2: H1 halts at its reservation, cannot borrow ──────────

def test_POSITIVE_CONTROL_H1_HALTS_at_its_reservation_without_borrowing():
    """Drive H1's spend to the reservation and confirm the halt — and that the
    state is RESERVATION_EXHAUSTED, not CEILING_HIT."""
    from scripts.h1_collection_check import CREDIT_CEILING, collection_stop

    below = collection_stop(n_fixtures=5, credits_spent=H1_RESERVED_CREDITS - 1,
                            raw_rows=500, reservation=H1_RESERVED_CREDITS)
    assert below.halt is False

    at = collection_stop(n_fixtures=5, credits_spent=H1_RESERVED_CREDITS,
                         raw_rows=500, reservation=H1_RESERVED_CREDITS)
    assert at.halt is True, "H1 reached its reservation and did NOT halt"
    assert at.state == "RESERVATION_EXHAUSTED"
    assert "may NOT borrow" in at.reason
    assert str(CREDIT_CEILING) in at.reason, (
        "the state does not record that the registered ceiling was never "
        "reached — a reader would think the experiment hit its own bound")


def test_H1_sees_its_reservation_rather_than_having_it_withheld():
    """`for_h1=True` must not have the reservation deducted from it — the
    reservation IS H1's to spend; its bound is `collection_stop`, not the ledger."""
    assert _quota(used=0, for_h1=True).remaining(today=IN_WINDOW) == 400
    assert _quota(used=0, for_h1=False).remaining(today=IN_WINDOW) == 232


def test_RESERVATION_EXHAUSTED_is_DISTINGUISHABLE_from_the_other_three():
    from scripts.h1_collection_check import (
        COLLECTION_TERMINAL_STATES, CREDIT_CEILING, collection_stop)
    cases = {
        "COMPLETE": dict(n_fixtures=39, credits_spent=10, raw_rows=9),
        "RESERVATION_EXHAUSTED": dict(n_fixtures=5, credits_spent=168, raw_rows=9),
        "CEILING_HIT": dict(n_fixtures=5, credits_spent=CREDIT_CEILING, raw_rows=9),
        "NO_DATA": dict(n_fixtures=0, credits_spent=168, raw_rows=0),
    }
    lines = {}
    for expected, kw in cases.items():
        s = collection_stop(reservation=H1_RESERVED_CREDITS, **kw)
        assert s.state == expected, f"{kw} -> {s.state}, expected {expected}"
        lines[expected] = str(s)
    assert set(lines) == set(COLLECTION_TERMINAL_STATES)
    assert len(set(lines.values())) == 4, "two terminal states print identically"


def test_the_ONSET_is_pinned_not_only_the_expiry():
    """The expiry is the half everyone pins. The ONSET is the half that can
    silently cost a week.

    An allocator that activates early throttles normal pricing for seven days
    for nothing, and nothing would fail — pricing would simply get less budget
    and go dark sooner, which looks exactly like a busy month.
    """
    assert H1_RESERVATION_FROM == _d.date(2026, 10, 9), (
        "the reservation's start moved; it must match the card's return date, "
        "measured from the scraper as 10-09/10-10 for the covered eight")
    assert h1_reservation(_d.date(2026, 10, 8)) == 0, "activates a day early"
    assert h1_reservation(_d.date(2026, 10, 9)) == H1_RESERVED_CREDITS
    assert h1_reservation(_d.date(2026, 10, 16)) == H1_RESERVED_CREDITS
    assert h1_reservation(_d.date(2026, 10, 17)) == 0, "outlives its window"


def test_the_allocator_is_DORMANT_before_the_onset():
    """What a dormant allocator means, asserted rather than assumed: normal
    operation sees the WHOLE pool and the reservation is not yet deducted."""
    for day in (_d.date(2026, 10, 2), _d.date(2026, 10, 8)):
        a = allocate(450, 50, today=day)
        assert a["h1_reserved"] == 0, f"{day}: reservation deducted early"
        assert a["normal_ceiling"] == a["pool"] == 400, (
            f"{day}: normal operation is already throttled — seven days of "
            f"reduced pricing bought nothing")


def test_only_the_FAIL_CLOSED_branch_reaches_CEILING_HIT_from_the_runner():
    """An honest reachability statement, not a claim that all four fire live.

    The runner calls `collection_state`, which uses the DEFAULT reservation
    (168). Because 168 < CREDIT_CEILING (200), the reservation ALWAYS binds
    first on a readable ledger — so from the runner as shipped, `CEILING_HIT`
    is reachable only through `credits_spent is None`, the fail-closed path.
    That is the safe direction, and it is recorded so nobody reads a missing
    CEILING_HIT as the ceiling being untested.
    """
    from scripts.h1_collection_check import CREDIT_CEILING, collection_stop
    assert H1_RESERVED_CREDITS < CREDIT_CEILING

    # readable ledger, far past both bounds -> the RESERVATION is reported
    s = collection_stop(n_fixtures=0, credits_spent=CREDIT_CEILING + 50,
                        raw_rows=9)
    assert s.state == "CEILING_HIT", s          # past the ceiling too
    mid = collection_stop(n_fixtures=0, credits_spent=H1_RESERVED_CREDITS + 1,
                          raw_rows=9)
    assert mid.state == "RESERVATION_EXHAUSTED", mid

    # unreadable ledger -> CEILING_HIT via fail-closed, with no credit figure
    none = collection_stop(n_fixtures=0, credits_spent=None, raw_rows=9)
    assert none.state == "CEILING_HIT" and "UNREADABLE" in none.reason


def test_the_H1_window_values_are_PASSED_never_defaulted():
    """The revert is the removal of two arguments. That only holds if the
    defaults are the NORMAL values."""
    import inspect
    import src.scrapers.theodds_scraper as ts
    sig = inspect.signature(ts.TheOddsScraper.refresh_imminent)
    assert sig.parameters["window_minutes"].default == 120
    assert sig.parameters["min_interval_minutes"].default == 180
    assert sig.parameters["h1_collection"].default is False, (
        "h1_collection defaults to True — then the revert is an edit, not a "
        "removal, and every ordinary refresh carries the collection stop")
