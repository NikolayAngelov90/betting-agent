"""The reservation ALLOCATES. Proved by a consumer being refused — 2026-10-23.

> Enforcement is proved by a consumer being refused, never by reading a
> printed number.

THE BLOCKER, MEASURED BEFORE IT WAS FIXED. On a throwaway ledger, inside the
reservation window, a `for_h1=True` instance asked for 144 league requests:

    H1 asked 144 req (288 cr) -> GRANTED 144 (288 cr)
    H1 attributed: 288 of a 168 reservation      overdrawn by 120
    normal.remaining AFTER: 0                    a claim of 50 -> 0 granted

So the allocation was ONE-SIDED. Stage 29 bound the normal side —
`reserved_from_me` withholds H1's share and `claim_requests` enforces it — and
the H1 side was deliberately left to `collection_stop`, which is a PRE-FLIGHT
CHECK: it reads cumulative spend before the run starts and cannot bound what
the run then claims. `remaining()` reported 400 to an instance allocated 168.

AND FIXING IT EXPOSED A SECOND DEFECT, in the opposite direction. With H1's
168 spent, normal operation's `remaining()` read **64** rather than its 232
ceiling, because the reservation was withheld from the pool a SECOND time
after H1 had already drawn it: 168 + 64 = 232 spendable of a 400 pool, with
**168 credits unspendable by anyone**. That is Stage 26's "two independent
ceilings over one exhaustible pool" re-entering through the other door, and it
is why `reserved_from_me` now withholds the UNSPENT reservation.

WHY THE HARNESS DID NOT CATCH THE BLOCKER, and it is the same shape as the two
defects Stage 29 found: every existing enforcement test drove the NORMAL
instance. `for_h1=True` was tested for ATTRIBUTION — that credits land in the
right counter — and never for a BOUND. A direction that is never driven is
never wrong.
"""

from datetime import date, datetime

import pytest

import src.data.database as db_mod
import src.data.odds_quota as oq
from src.data.models import Base
from src.data.odds_quota import (
    CREDITS_PER_REQUEST,
    H1_RESERVED_CREDITS,
    OddsApiQuota,
    allocate,
    credits_for,
)
from scripts.h1_collection_check import (
    COLLECTION_RUNNING,
    COLLECTION_TERMINAL_STATES,
    CREDIT_CEILING,
    collection_state,
    collection_stop,
    effective_bound,
)
from tests.h1_synthetic import build

#: Production's numbers, stated so the arithmetic below is checkable.
BUDGET, MARGIN = 450, 50
POOL = BUDGET - MARGIN                      # 400
NORMAL_CEILING = POOL - H1_RESERVED_CREDITS  # 232

#: PINNED to a day inside the reservation window. A test that reads the wall
#: clock states no regime — three tests in the sibling suite broke on
#: 2026-10-09 by inheriting the allocator's first active day.
IN_WINDOW = oq.H1_RESERVATION_FROM


def _mgr(tmp_path, name="h1alloc.db"):
    mgr = db_mod.DatabaseManager(
        config=type("C", (), {"database": {"sqlite_path": str(tmp_path / name)}})())
    Base.metadata.create_all(mgr.engine)
    assert not mgr.is_postgres, "this suite must never reach production"
    return mgr


def _q(mgr, *, for_h1):
    """`max_credits_per_run=0` so the PER-RUN ceiling cannot be mistaken for
    the allocation — the two bound different things and conflating them is how
    a one-sided allocation looked enforced."""
    return OddsApiQuota(mgr, monthly_budget=BUDGET, safety_margin=MARGIN,
                        max_credits_per_run=0, for_h1=for_h1)


# ═══════════════ THE BLOCKER: H1 IS REFUSED PAST ITS SHARE ═══════════════

def test_the_BLOCKER_H1_is_REFUSED_past_its_RESERVATION(tmp_path):
    """Driven, not read. Before the fix this granted all 144 requests."""
    mgr = _mgr(tmp_path)
    h1 = _q(mgr, for_h1=True)

    ask = H1_RESERVED_CREDITS // CREDITS_PER_REQUEST + 60      # 84 + 60
    granted = h1.claim_requests(ask, today=IN_WINDOW)

    assert credits_for(granted) == H1_RESERVED_CREDITS, (
        f"H1 asked for {credits_for(ask)} credits against a "
        f"{H1_RESERVED_CREDITS}-credit reservation and was granted "
        f"{credits_for(granted)} — the allocation is reported but not enforced "
        f"on H1's side")
    assert h1.h1_used(IN_WINDOW) == H1_RESERVED_CREDITS

    # AND THE REFUSAL PERSISTS. A bound that only clips the first claim is not
    # a bound; the second ask must get nothing.
    assert h1.claim_requests(10, today=IN_WINDOW) == 0, (
        "a second claim was granted after the reservation was fully spent")
    assert h1.h1_headroom(IN_WINDOW) == 0


def test_normal_pricing_is_REFUSED_past_its_REMAINDER(tmp_path):
    """The other direction, driven on the same fixture shape."""
    mgr = _mgr(tmp_path)
    normal = _q(mgr, for_h1=False)

    granted = normal.claim_requests(200, today=IN_WINDOW)
    assert credits_for(granted) == NORMAL_CEILING, (
        f"normal pricing was granted {credits_for(granted)} against a "
        f"{NORMAL_CEILING}-credit remainder")
    assert POOL - normal.used(IN_WINDOW) >= H1_RESERVED_CREDITS, (
        "the reservation did not survive normal pricing's exhaustion attempt")


def test_EACH_CONSUMER_EXHAUSTING_FIRST_LEAVES_THE_OTHER_WHOLE(tmp_path):
    """The two directions driven AGAINST EACH OTHER, both orderings.

    This is the test the blocker needed and nothing had: each consumer is run
    to refusal and then the OTHER is asked for its full share.
    """
    for order in (("h1", "normal"), ("normal", "h1")):
        mgr = _mgr(tmp_path, name=f"{'_'.join(order)}.db")
        inst = {"h1": _q(mgr, for_h1=True), "normal": _q(mgr, for_h1=False)}
        expect = {"h1": H1_RESERVED_CREDITS, "normal": NORMAL_CEILING}

        spent = {}
        for who in order:
            granted = inst[who].claim_requests(300, today=IN_WINDOW)
            spent[who] = credits_for(granted)
            assert spent[who] == expect[who], (
                f"{order[0]} first: {who} got {spent[who]} credits, expected "
                f"its full allocation of {expect[who]} — going "
                f"{'first' if who == order[0] else 'second'} must not change "
                f"what a consumer is entitled to")

        assert sum(spent.values()) == POOL, (
            f"{order[0]} first: {sum(spent.values())} of a {POOL} pool was "
            f"spendable, stranding {POOL - sum(spent.values())} credits that "
            f"neither consumer could reach")


def test_NO_CREDIT_IS_STRANDED_once_H1_HAS_SPENT(tmp_path):
    """The second defect, pinned. The reservation must be withheld ONCE.

    With H1's share spent, normal operation's ceiling is unchanged — the
    credits came out of H1's allocation, not out of normal's.
    """
    mgr = _mgr(tmp_path)
    h1, normal = _q(mgr, for_h1=True), _q(mgr, for_h1=False)

    before = normal.remaining(IN_WINDOW)
    assert before == NORMAL_CEILING, before

    h1.claim_requests(H1_RESERVED_CREDITS // CREDITS_PER_REQUEST,
                      today=IN_WINDOW)
    after = normal.remaining(IN_WINDOW)

    assert after == NORMAL_CEILING, (
        f"normal operation's ceiling fell from {before} to {after} when H1 "
        f"spent its own reservation — the reservation is being withheld twice, "
        f"stranding {NORMAL_CEILING - after} credits")


# ═════════════════ ONE DEFINITION, TWO ENFORCEMENT POINTS ════════════════

def test_BOTH_CEILINGS_COME_FROM_allocate_AND_SUM_TO_THE_POOL():
    a = allocate(BUDGET, MARGIN, IN_WINDOW)
    assert a["normal_ceiling"] + a["h1_ceiling"] == a["pool"], (
        f"the two ceilings sum to {a['normal_ceiling'] + a['h1_ceiling']} "
        f"against a pool of {a['pool']} — they are no longer one allocation")
    assert a["h1_ceiling"] == a["h1_reserved"], (
        "h1_ceiling has drifted from h1_reserved; it is meant to be the SAME "
        "number named at the point it is enforced, not a second constant")


def test_THE_REPORTED_CEILING_IS_THE_ENFORCED_ONE_for_both(tmp_path):
    """`remaining()` is what `claim_requests` clamps to, on both sides.

    Its docstring has been false twice — once per side — so it is checked
    against the claim rather than read.
    """
    for for_h1, ceiling in ((False, NORMAL_CEILING),
                            (True, H1_RESERVED_CREDITS)):
        mgr = _mgr(tmp_path, name=f"rep{int(for_h1)}.db")
        q = _q(mgr, for_h1=for_h1)
        reported = q.remaining(IN_WINDOW)
        assert reported == ceiling, (for_h1, reported, ceiling)
        granted = q.claim_requests(500, today=IN_WINDOW)
        assert credits_for(granted) == reported, (
            f"for_h1={for_h1}: reported {reported} spendable, the claim "
            f"granted {credits_for(granted)}")


def test_OUTSIDE_THE_WINDOW_H1_HAS_NO_ALLOCATION_AND_NORMAL_HAS_THE_POOL(tmp_path):
    """The reservation is a CALENDAR fact. Outside the window H1's ceiling is
    zero — and that is a refusal, not a silent full pool."""
    out = date(2026, 10, 20)        # same month, so only the window differs
    assert allocate(BUDGET, MARGIN, out)["h1_ceiling"] == 0
    assert allocate(BUDGET, MARGIN, out)["normal_ceiling"] == POOL

    mgr = _mgr(tmp_path, name="out.db")
    assert _q(mgr, for_h1=True).claim_requests(10, today=out) == 0, (
        "H1 spent credits outside its reservation window")
    assert credits_for(_q(mgr, for_h1=False).claim_requests(500, today=out)) \
        == POOL


# ═══════════ THE TERMINAL STATES, AND ONE REACHED AS WIRED ═══════════════

def test_RESERVATION_EXHAUSTED_is_reachable_FROM_THE_WIRED_PATH(tmp_path):
    """Through `collection_state(session, since)` — the function the runner
    calls — not through `collection_stop` with a hand-supplied number.

    THIS IS THE DISTINCTION STAGE 29 NAMED: every `collection_stop` test
    passed `credits_spent` IN as an argument, so the one place that DERIVES it
    was never tested against the facts. A parameter supplied only by the
    harness was never wrong.
    """
    mgr = _mgr(tmp_path, name="wired.db")

    # real observations, so `raw_rows > 0` and the halt is the BUDGET's, not
    # NO_DATA's — the two are different facts and must not be confused here
    with mgr.get_session() as s:
        build(s, n_fixtures=4, first_day=date(2026, 11, 1), days=2)

    # spend H1's whole allocation through the REAL claim path
    h1 = _q(mgr, for_h1=True)
    h1.claim_requests(H1_RESERVED_CREDITS // CREDITS_PER_REQUEST,
                      today=IN_WINDOW)
    assert h1.h1_used(IN_WINDOW) == H1_RESERVED_CREDITS

    with mgr.get_session() as s:
        stop = collection_state(s, datetime(2026, 10, 1))

    assert stop.halt is True, stop
    assert stop.state == "RESERVATION_EXHAUSTED", (
        f"the wired path reported {stop.state!r}, not RESERVATION_EXHAUSTED, "
        f"with H1's attributed spend at exactly its allocation. {stop.reason}")


def test_THE_RESERVATION_MUST_DIVIDE_BY_CREDITS_PER_REQUEST():
    """Otherwise `RESERVATION_EXHAUSTED` becomes UNREACHABLE — a livelock.

    The claim grants whole requests, so H1's attributed spend only ever takes
    multiples of `CREDITS_PER_REQUEST`. `collection_stop` halts on
    `credits_spent >= bound`. If the bound is not such a multiple, the largest
    reachable spend is strictly below it: nothing can be claimed (headroom <
    one request) and nothing halts (spend < bound), so the state stays
    `COLLECTING` for ever.

    168 / 2 happens to divide. That is luck until a test requires it.
    """
    assert H1_RESERVED_CREDITS % CREDITS_PER_REQUEST == 0, (
        f"the reservation {H1_RESERVED_CREDITS} is not a multiple of "
        f"{CREDITS_PER_REQUEST} credits per request. The largest reachable "
        f"attributed spend is "
        f"{(H1_RESERVED_CREDITS // CREDITS_PER_REQUEST) * CREDITS_PER_REQUEST}, "
        f"which is below the bound, so the claim refuses and the stop never "
        f"fires: LIVELOCK. Round the reservation to a multiple, or make the "
        f"stop fire when headroom drops below one request.")
    # and the bound the stop actually uses, which is the capped one
    assert effective_bound(H1_RESERVED_CREDITS) % CREDITS_PER_REQUEST == 0


def test_THE_FIVE_TERMINAL_STATES_STAY_DISTINGUISHABLE():
    """Each reachable, each distinct, and the non-terminal one is named.

    `BUDGET_UNREADABLE` was split out of `CEILING_HIT` because the latter was
    carrying two facts — a bound reached, and the ledger unreadable — and a
    fail-closed halt reported as a bound event is a lie about why it stopped.
    """
    cases = {
        "BUDGET_UNREADABLE": dict(n_fixtures=5, credits_spent=None,
                                  raw_rows=50, reservation=H1_RESERVED_CREDITS),
        "NO_DATA": dict(n_fixtures=0, credits_spent=H1_RESERVED_CREDITS,
                        raw_rows=0, reservation=H1_RESERVED_CREDITS),
        "COMPLETE": dict(n_fixtures=39, credits_spent=10, raw_rows=500,
                         reservation=H1_RESERVED_CREDITS),
        "CEILING_HIT": dict(n_fixtures=5, credits_spent=CREDIT_CEILING,
                            raw_rows=50, reservation=CREDIT_CEILING + 50),
        "RESERVATION_EXHAUSTED": dict(n_fixtures=5,
                                      credits_spent=H1_RESERVED_CREDITS,
                                      raw_rows=50,
                                      reservation=H1_RESERVED_CREDITS),
    }
    assert set(cases) == set(COLLECTION_TERMINAL_STATES), (
        f"the terminal states have changed: "
        f"{set(COLLECTION_TERMINAL_STATES) ^ set(cases)}")
    seen = {}
    for want, kw in cases.items():
        got = collection_stop(**kw)
        assert got.halt is True and got.state == want, (want, got)
        seen[want] = got.reason
    assert len(set(seen.values())) == len(seen), (
        "two terminal states gave the SAME reason text, so they are not "
        "distinguishable in a log")

    # and the non-terminal state is a state, not an absence
    running = collection_stop(n_fixtures=5, credits_spent=10, raw_rows=50,
                              reservation=H1_RESERVED_CREDITS)
    assert running.halt is False and running.state == COLLECTION_RUNNING


# ══════════════════════ ATTRIBUTION STAYS SEPARABLE ══════════════════════

def test_for_h1_ATTRIBUTION_KEEPS_THE_SPEND_SEPARABLE(tmp_path):
    """`h1_used()` must remain H1's spend and `used()` the pool's total.

    Comparing the pool total to the reservation was Stage 29's second defect:
    on 2026-10-09 it charged H1 with 38 credits spent by normal pricing before
    H1 had run at all.
    """
    mgr = _mgr(tmp_path)
    normal, h1 = _q(mgr, for_h1=False), _q(mgr, for_h1=True)

    normal.claim_requests(10, today=IN_WINDOW)          # 20 credits, not H1's
    assert h1.h1_used(IN_WINDOW) == 0, (
        f"normal pricing's spend was attributed to H1: "
        f"{h1.h1_used(IN_WINDOW)} credits")

    h1.claim_requests(5, today=IN_WINDOW)               # 10 credits, H1's
    assert h1.h1_used(IN_WINDOW) == credits_for(5)
    assert h1.used(IN_WINDOW) == credits_for(15), (
        "the pool total no longer counts both consumers")


def test_AN_UNATTRIBUTABLE_H1_CREDIT_IS_REFUSED(tmp_path, monkeypatch):
    """H1 fails CLOSED when attribution is impossible.

    This refusal is what makes `h1_credits_used`'s "a missing row is zero"
    safe: without it, a bound compared against an unwritten counter never
    binds, and the zero is the silent kind.
    """
    mgr = _mgr(tmp_path)
    h1 = _q(mgr, for_h1=True)
    monkeypatch.setattr(h1._h1_store, "available", lambda: False)

    assert h1.h1_headroom(IN_WINDOW) is None, (
        "an unreadable attribution counter reported a number")
    assert h1.remaining(IN_WINDOW) == 0, (
        "an unreadable attribution counter reported spendable credits — it "
        "must fail closed, because not knowing the spend is not headroom")
    assert h1.claim_requests(10, today=IN_WINDOW) == 0


def test_AN_UNREADABLE_H1_COUNTER_PROTECTS_THE_RESERVATION(tmp_path, monkeypatch):
    """And the fail-closed direction on the NORMAL side is the opposite one.

    When H1's spend cannot be read, normal operation withholds the FULL
    reservation. Not knowing how much H1 has drawn must protect H1's share,
    never release it — releasing it is how a reservation silently becomes a
    suggestion.
    """
    mgr = _mgr(tmp_path)
    normal = _q(mgr, for_h1=False)
    monkeypatch.setattr(normal._h1_store, "available", lambda: False)

    assert normal.reserved_from_me(IN_WINDOW) == MARGIN + H1_RESERVED_CREDITS
    assert normal.remaining(IN_WINDOW) == NORMAL_CEILING
