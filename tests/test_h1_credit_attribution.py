"""H1 credit attribution and the allocator's enforcement — Stage 29, 2026-10-09.

TWO DEFECTS, FOUND BY WIRING THE CALLER AND MEASURED BEFORE ANY CREDIT WAS
SPENT. They pointed in opposite directions and composed into a lie.

    1. THE RESERVATION WAS REPORTED AND NEVER ENFORCED.
       `remaining()` reserved `reserved_from_me()` (margin + H1's 168) and said
       232. `claim_requests` — the ONLY site that spends — claimed against
       `safety_margin` alone. Asked for 200 requests inside the window, it
       granted 200 = 400 credits and consumed all 168 of H1's reservation. The
       `remaining()` docstring asserted that `claim_requests` "walks down from
       this number"; it walked down from what the CALLER asked for.

    2. THE STOP CHARGED NORMAL PRICING'S SPEND TO H1.
       `collection_state` passed `credits_used()` — MONTH-TO-DATE TOTAL for the
       provider — as `credits_spent`, compared against the 168 reservation. On
       2026-10-09, with 38 credits spent by normal pricing and H1 not yet run,
       the stop read `credits=38/168`. H1's usable budget was 130, not 168: 21
       league-days instead of 28, a 25% cut to a sized experiment's capacity
       from an accounting error.

       The arithmetic is not the worst of it. Had normal pricing alone reached
       168, H1 would have halted `RESERVATION_EXHAUSTED` having spent ZERO,
       with a stated reason — "the allocation is spent" — that was false.

WHY THE HARNESS DID NOT CATCH EITHER, and it is the same reason both times:

    * Every `collection_stop` test PASSED `credits_spent` IN as an argument.
      The one place that DERIVES it was never tested against the distinction.
      PNC-1 says a producer with no consumer looks like a working feature; this
      is its mirror — a parameter supplied only by the harness was never wrong.
    * `reserved_from_me` was tested through `remaining()`, which reports. It
      was never tested through `claim_requests`, which spends. A number is only
      enforced where it is enforced.

AND THE REPORT KNEW. `render` carried the exact caveat — "month-to-date is not
collection-to-date" — on the side that only prints, while the enforcing path
compared the numbers anyway. One quantity, two definitions, the warning on the
harmless branch. Worse, that caveat was DISARMED: gated on
`since.date() < month_key()`, and with `--since` defaulting to the 10-01 reset
the flag read False exactly while the window started 10-09 with 38 credits gone.

THE REPAIR IS A MEASUREMENT, NOT AN INFERENCE. `PROVIDER_H1` counts credits
claimed by a `for_h1=True` instance, and `claim_requests` REFUSES to spend what
it cannot attribute — so a counter reading 0 means zero spent rather than broken
attribution. That refusal is what makes a missing row safe to read as zero.

Every date here is PINNED. Three tests in this suite broke on 2026-10-09
because they called a quota method with no explicit `today=` and inherited the
allocator's first active day; 22 more are latent. A test that reads the wall
clock states no regime.
"""

from datetime import date, timedelta

from src.data import h1_window as _h1_window

import pytest

import src.data.database as db_mod
from src.data.models import Base
from src.data.odds_quota import (
    H1_RESERVED_CREDITS,
    PROVIDER,
    PROVIDER_H1,
    OddsApiQuota,
    allocate,
    credits_for,
    h1_reservation,
    month_key,
)
from scripts.h1_collection_check import (
    CREDIT_CEILING,
    collection_stop,
    credits_used,
    h1_credits_used,
)

#: ── THE DATES MOVED WITH THE WINDOW. 2026-10-28, AUTHORISED. ─────────────
#:
#: WAS 2026-10-09..2026-10-17. The re-run is chosen and starts at the credit
#: reset, so the window is 2026-11-01..2026-11-09 exclusive — the SAME
#: pre-registered eight days, repositioned. Only these date pins move; not one
#: assertion about the allocation, the attribution or the stop is weakened.
#:
#: These are the "dated pins" that are expected to fail on a window move, and
#: they are updated in the same commit as the move with the argument beside
#: them, the way invariant 6b is handled.
#:
#: Inside the reservation window. The window is 2026-11-01..2026-11-09.
#: ── DERIVED, NOT RESTATED. 2026-10-28. ───────────────────────────────────
#: These were literals, and literals are how four sites came to be
#: half-moved in one week. `h1_window` is the only place either date
#: appears; a test that needs "a day inside the window" asks for one.
#: The deliberate PINS -- BOUNDARY_DATES, WINDOW_EDGES, the ONSET pin --
#: keep their literals on purpose: a pin that imports its own subject is
#: a tautology. They are exempted by name in
#: `test_every_H1_DATE_SITE_IMPORTS_rather_than_RESTATES`.
IN_WINDOW = _h1_window.FROM
#: Outside it, and deliberately in the SAME month so the month-key is identical
#: and only the reservation differs.
OUT_OF_WINDOW = _h1_window.UNTIL + timedelta(days=11)

POOL = 400                  # 450 budget - 50 margin, production's numbers
NORMAL_CEILING = 232        # POOL - 168


def _mgr(tmp_path, name="h1cred.db"):
    mgr = db_mod.DatabaseManager(
        config=type("C", (), {"database": {"sqlite_path": str(tmp_path / name)}})())
    Base.metadata.create_all(mgr.engine)
    assert not mgr.is_postgres, "this suite must never reach production"
    return mgr


def _quota(tmp_path, *, for_h1=False, name="h1cred.db", mgr=None):
    return OddsApiQuota(mgr or _mgr(tmp_path, name),
                        monthly_budget=450, safety_margin=50,
                        max_credits_per_run=0, for_h1=for_h1)


# ═══════════════════════════ DEFECT 1: enforcement ═══════════════════════════

def test_the_allocation_is_what_the_claim_enforces(tmp_path):
    """The reported ceiling and the enforced one must be the same number."""
    q = _quota(tmp_path)
    assert allocate(450, 50, IN_WINDOW)["normal_ceiling"] == NORMAL_CEILING
    assert q.remaining(IN_WINDOW) == NORMAL_CEILING
    granted = q.claim_requests(200, today=IN_WINDOW)
    assert credits_for(granted) == NORMAL_CEILING, (
        f"the claim granted {credits_for(granted)} credits against a reported "
        f"ceiling of {NORMAL_CEILING} — reported and enforced have diverged")


def test_normal_pricing_cannot_spend_the_H1_RESERVATION(tmp_path):
    """The defect, pinned. Named in `remaining()`'s docstring.

    Before the fix this granted 200 of 200 requests = 400 credits and left H1
    with nothing. A reservation that the spending path does not consult is a
    comment.
    """
    q = _quota(tmp_path)
    q.claim_requests(200, today=IN_WINDOW)
    assert q.used(IN_WINDOW) <= NORMAL_CEILING, (
        f"normal pricing spent {q.used(IN_WINDOW)} of a {POOL} pool, eating "
        f"{q.used(IN_WINDOW) - NORMAL_CEILING} credits of H1's "
        f"{H1_RESERVED_CREDITS}-credit reservation")
    assert POOL - q.used(IN_WINDOW) >= H1_RESERVED_CREDITS, (
        "the reservation did not survive normal pricing's exhaustion attempt")


def test_the_positive_control_a_claim_that_reserves_only_the_margin_breaches(tmp_path):
    """PROVE THE TEST ABOVE CAN FAIL. It is the pre-fix behaviour exactly.

    Without this, `test_normal_pricing_cannot_spend_the_H1_RESERVATION` could
    pass because nothing was ever claimed — the vacuous-filter shape (VAC-1)
    one level up. Here the reservation is withheld from `reserved_from_me` and
    the breach MUST reappear.
    """
    q = _quota(tmp_path)
    q.reserved_from_me = lambda today=None: q.safety_margin   # the old value
    q.claim_requests(200, today=IN_WINDOW)
    assert q.used(IN_WINDOW) == POOL, (
        "reserving only the safety margin no longer breaches the reservation — "
        "then the guard above is not measuring what it claims to")
    assert POOL - q.used(IN_WINDOW) == 0, "H1 would have had nothing left"


def test_outside_the_window_normal_pricing_gets_the_WHOLE_pool(tmp_path):
    """THE OTHER DIRECTION, or the fix is a permanent throttle.

    `h1_reservation` is calendar-gated precisely so it cannot outlive the
    experiment. Enforcing it in the claim must not survive 10-17 either.
    """
    assert h1_reservation(OUT_OF_WINDOW) == 0
    q = _quota(tmp_path)
    assert q.remaining(OUT_OF_WINDOW) == POOL
    q.claim_requests(500, today=OUT_OF_WINDOW)
    assert q.used(OUT_OF_WINDOW) == POOL, (
        f"outside the H1 window normal pricing got {q.used(OUT_OF_WINDOW)} of "
        f"{POOL} — the reservation is throttling a month it does not own")


def test_an_H1_instance_may_spend_the_reservation_itself(tmp_path):
    """The reservation is withheld from normal pricing FOR H1, not from H1."""
    mgr = _mgr(tmp_path)
    normal = _quota(tmp_path, mgr=mgr)
    h1 = _quota(tmp_path, mgr=mgr, for_h1=True)
    normal.claim_requests(200, today=IN_WINDOW)
    assert normal.used(IN_WINDOW) == NORMAL_CEILING
    granted = h1.claim_requests(10, today=IN_WINDOW)
    assert granted == 10, (
        "H1 was refused its own reservation after normal pricing hit the "
        "remainder — the allocation would then protect nobody")
    assert h1.used(IN_WINDOW) == NORMAL_CEILING + 20


# ════════════════════════════ DEFECT 2: attribution ══════════════════════════

def test_the_stop_counts_only_H1s_OWN_spend(tmp_path):
    """Production's 2026-10-09 state, replayed: 38 normal credits, H1 at zero.

    Before the fix the stop read `credits=38/168` before H1 had made a request.
    """
    mgr = _mgr(tmp_path)
    normal = _quota(tmp_path, mgr=mgr)
    normal.claim_requests(19, today=IN_WINDOW)          # 38 credits
    assert normal.used(IN_WINDOW) == 38

    with mgr.get_session() as s:
        attributed = h1_credits_used(s, IN_WINDOW)
        pool_total = credits_used(s, IN_WINDOW)
    assert pool_total == 38, "the pool total should still see normal's spend"
    assert attributed == 0, (
        f"H1 is charged {attributed} credits it did not spend — normal "
        f"pricing's {pool_total} leaked into the experiment's account")

    stop = collection_stop(n_fixtures=0, credits_spent=attributed, raw_rows=0,
                           reservation=h1_reservation(IN_WINDOW))
    assert stop.halt is False
    assert "credits=0/168" in stop.reason, stop.reason


def test_H1s_spend_IS_attributed(tmp_path):
    """The counter must move when H1 spends, or it is a constant zero."""
    mgr = _mgr(tmp_path)
    normal = _quota(tmp_path, mgr=mgr)
    h1 = _quota(tmp_path, mgr=mgr, for_h1=True)
    normal.claim_requests(19, today=IN_WINDOW)          # 38 credits, not H1's
    h1.claim_requests(3, today=IN_WINDOW)               # 6 credits, H1's

    assert h1.used(IN_WINDOW) == 44, "the pool must count BOTH"
    assert h1.h1_used(IN_WINDOW) == 6, "the account must count only H1's"
    with mgr.get_session() as s:
        assert h1_credits_used(s, IN_WINDOW) == 6
        assert credits_used(s, IN_WINDOW) == 44


def test_a_full_reservation_of_H1_spend_exhausts_it_and_nothing_else_does(tmp_path):
    """RESERVATION_EXHAUSTED must mean H1 spent 168, not that the month did."""
    mgr = _mgr(tmp_path)
    normal = _quota(tmp_path, mgr=mgr)
    normal.claim_requests(84, today=IN_WINDOW)          # 168 credits of NORMAL
    with mgr.get_session() as s:
        stop = collection_stop(n_fixtures=5, credits_spent=h1_credits_used(s, IN_WINDOW),
                               raw_rows=100,
                               reservation=h1_reservation(IN_WINDOW))
    assert stop.state != "RESERVATION_EXHAUSTED", (
        "the month spent 168 and H1 spent nothing, yet the stop reported H1's "
        "allocation as exhausted — a terminal state with a false reason")
    assert stop.halt is False

    # And the state IS reachable, by the only thing that should reach it.
    h1 = _quota(tmp_path, mgr=mgr, for_h1=True)
    assert h1.h1_used(IN_WINDOW) == 0
    stop2 = collection_stop(n_fixtures=5, credits_spent=H1_RESERVED_CREDITS,
                            raw_rows=100,
                            reservation=h1_reservation(IN_WINDOW))
    assert stop2.state == "RESERVATION_EXHAUSTED" and stop2.halt is True


# ═════════════════════ the refusal that makes zero readable ══════════════════

def test_H1_refuses_to_spend_what_it_cannot_attribute(tmp_path):
    """THE PROPERTY THE WHOLE REPAIR RESTS ON.

    A missing counter row is read as zero. That is only safe because an
    unattributed H1 credit is impossible — otherwise the zero is the silent
    kind and a bound compared against it never binds.
    """
    mgr = _mgr(tmp_path)
    h1 = _quota(tmp_path, mgr=mgr, for_h1=True)
    h1._h1_store.record_attributed = lambda n, day=None: None

    granted = h1.claim_requests(5, today=IN_WINDOW)
    assert granted == 0, (
        f"H1 was granted {granted} request(s) it could not attribute — those "
        f"credits would be invisible to the stop")
    assert h1.used(IN_WINDOW) == 0, (
        f"the pool shows {h1.used(IN_WINDOW)} credits spent after a refused "
        f"claim — the claim was not released, so the month lost credits to a "
        f"request that never happened")


def test_normal_pricing_still_fails_OPEN_when_the_ledger_is_gone(tmp_path):
    """THE ASYMMETRY, both halves. Stale odds are worse than unbudgeted ones.

    Deliberately opposite to H1 below: the card's prices must not depend on a
    bookkeeping table, while a registered experiment's credits must.
    """
    mgr = _mgr(tmp_path)
    q = _quota(tmp_path, mgr=mgr)
    q._store._available = False
    assert q.claim_requests(4, today=IN_WINDOW) == 4, (
        "normal pricing was refused because the ledger was unavailable — the "
        "card loses its prices to a missing bookkeeping table")


def test_H1_fails_CLOSED_when_the_ledger_is_gone(tmp_path):
    mgr = _mgr(tmp_path)
    h1 = _quota(tmp_path, mgr=mgr, for_h1=True)
    h1._store._available = False
    h1._h1_store._available = False
    assert h1.claim_requests(4, today=IN_WINDOW) == 0, (
        "H1 spent credits it cannot account for; `collection_stop` would read "
        "an attributed zero and never reach its bound")


def test_an_unreadable_ledger_gives_the_stop_BUDGET_UNREADABLE(tmp_path):
    """None, not 0. The third-state family, on the credit axis.

    `h1_credits_used` returns 0 for a MISSING ROW (true: nothing spent) and
    None for an UNREADABLE LEDGER (unknown). Collapsing them would turn
    "cannot see my own spend" into "have not spent", which is how a guard
    becomes a no-op.
    """
    mgr = _mgr(tmp_path)
    with mgr.get_session() as s:
        s.execute(__import__("sqlalchemy").text("DROP TABLE api_budget"))
        s.commit()
        assert h1_credits_used(s, IN_WINDOW) is None, (
            "a dropped table answered with a number — the stop would compare "
            "it to the reservation and keep spending")
    stop = collection_stop(n_fixtures=0, credits_spent=None, raw_rows=0,
                           reservation=H1_RESERVED_CREDITS)
    assert stop.state == "BUDGET_UNREADABLE" and stop.halt is True


def test_the_two_quantities_are_not_the_same_variable(tmp_path):
    """One quantity with two definitions is THE HABIT. These are two
    quantities with two names, and the test exists so they cannot re-merge."""
    mgr = _mgr(tmp_path)
    normal = _quota(tmp_path, mgr=mgr)
    h1 = _quota(tmp_path, mgr=mgr, for_h1=True)
    normal.claim_requests(10, today=IN_WINDOW)      # 20
    h1.claim_requests(2, today=IN_WINDOW)           # 4
    with mgr.get_session() as s:
        pool, mine = credits_used(s, IN_WINDOW), h1_credits_used(s, IN_WINDOW)
    assert (pool, mine) == (24, 4), (pool, mine)
    assert pool != mine, (
        "the pool total and H1's attributed spend returned the same number — "
        "if they are ever equal by construction the distinction is gone")
    assert PROVIDER != PROVIDER_H1
    # THE MONTH KEY FOLLOWS THE WINDOW. It is asserted against the window's
    # own month rather than a second literal, so the two cannot disagree
    # when the window moves again -- which is how these three tests came to
    # depend on the window and the wall clock sharing a month.
    assert month_key(IN_WINDOW) == IN_WINDOW.replace(day=1)


def test_the_bound_is_the_reservation_and_the_ceiling_is_not_reachable_below_it():
    """168 < 200, so the reservation binds first and CEILING_HIT needs a
    misconfiguration to reach. Pinned because collapsing the two ceilings into
    `effective_bound` made this the invariant that keeps five states honest."""
    assert H1_RESERVED_CREDITS < CREDIT_CEILING
    stop = collection_stop(n_fixtures=1, credits_spent=H1_RESERVED_CREDITS,
                           raw_rows=10, reservation=H1_RESERVED_CREDITS)
    assert stop.state == "RESERVATION_EXHAUSTED"


# ══════════════════ the guard on the guard (conftest's regime fixture) ═══════

def test_an_EXPLICIT_date_still_gets_the_REAL_reservation():
    """conftest neutralises `h1_reservation()` — it must not neutralise
    `h1_reservation(d)`, or every test in this file measures nothing.

    This is the control the alert harness taught: a fixture that silences the
    thing under test turns the whole file green and vacuous. VAC-1 applied to a
    fixture rather than a filter.
    """
    assert h1_reservation(IN_WINDOW) == H1_RESERVED_CREDITS, (
        "the conftest fixture is masking an explicitly dated reservation — "
        "every allocator assertion here is now vacuous")
    assert h1_reservation(OUT_OF_WINDOW) == 0
    assert allocate(450, 50, IN_WINDOW)["h1_reserved"] == H1_RESERVED_CREDITS


def test_the_fixture_governs_the_ENFORCEMENT_path_and_only_that():
    """WHAT THE FIXTURE ACTUALLY REACHES, measured rather than assumed.

    conftest rebinds the module attribute `src.data.odds_quota.h1_reservation`.
    `reserved_from_me` looks that global up AT CALL TIME, so the claim path —
    the one that spends, and the one the seven latent tests went through — sees
    the declared-only version. This module did `from ... import h1_reservation`
    at import, so its own name still points at the real function.

    THAT ASYMMETRY IS THE POINT AND IT IS ALSO THE LIMIT. It is recorded here
    rather than smoothed over, because the next person will reasonably expect a
    monkeypatched calendar predicate to be patched everywhere, and it is not.
    The surface was measured on 2026-10-09: no test outside this file calls
    `h1_reservation()` undated, and the three production callers
    (`render`'s `in_window` default, the `--h1` banner, and the report's bound
    line) all import it inside the function, so they resolve the patched one.
    """
    import src.data.odds_quota as oq

    # the ENFORCEMENT path, undated: no regime declared, so no allocation
    q = _quota(__import__("pathlib").Path("."), name=":memory:",
               mgr=_mgr_memory())
    assert q.reserved_from_me() == q.safety_margin, (
        "an undated `reserved_from_me` withheld the reservation — the "
        "conftest fixture is not active and the latent class is live again")

    # the module global IS patched; this module's imported name is NOT
    assert oq.h1_reservation() == 0
    assert h1_reservation(IN_WINDOW) == H1_RESERVED_CREDITS, (
        "an explicitly dated call lost the real answer — then every allocator "
        "assertion in this file is vacuous")


def _mgr_memory():
    import tempfile
    return _mgr(__import__("pathlib").Path(tempfile.mkdtemp()), "regime.db")
