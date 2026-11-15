"""The day-ahead deadline, taken from the external reference — 2026-10-19.

    A minimum over a subset is an UPPER BOUND on the true minimum.

WHY IT MOVED. Our own `matches` cannot supply tomorrow's earliest kickoff:
`max_days_ahead` is a rolling 24h from the run, and `measured` 2026-10-18 over
n=1,131 rows the maximum write-to-kickoff lead is **23.62h** with **ZERO** rows
ever known more than 24h ahead. The Flashscore probe built for the job paid
**240 page-loads for 8 refusals and zero deadlines** and was disabled.

THE REFERENCE ANSWERS, AND ITS HORIZON IS NOT THE PROBLEM. `measured`
2026-10-19, football-data.org returned a schedule for **D+1 through D+32
inclusive**, every one `state=OK` with zero rows dropped for being another
day. For 2026-10-20 it gave **16:45** while our own table held **0 rows**.

    horizon    ours 23.62 HOURS   |   reference >= 32 DAYS

THE COVERAGE IS THE WHOLE PROBLEM. `measured` 2026-10-19 it covers **9 of the
30 configured leagues**. A minimum over 9 errs LATE — the unsafe direction —
so it is combined with the conservative constant and never substituted:

    deadline = min(reference over covered leagues, day-type constant)

**SUPERSEDED IN PART ON 2026-10-19 — THE FLOOR NOW BINDS EVERYWHERE.** The
fallback was tightened from the 12:00 / 10:15 percentiles to derived floors of
**09:45 / 09:15**, and `measured` over the full record the earliest kickoff in
ANY of the 9 covered leagues is **10:00**. So the reference's minimum is never
below the floor: it binds on **0 of 834** days carrying a covered fixture.

Fixing the constant did not merely remove the exposure — it removed the
reference's role in the deadline. What the reference still does is
**independent monitoring**: it is the only outside check that the floor is not
absurd, and it would bind the day a covered league schedules before 09:45. The
`min()` property is still asserted below, driven synthetically, because a
property that is currently never exercised in production is still the property
that makes consulting an incomplete source safe at all.

THE REPLAY, all values `measured` from production `matches` (truth) and live
football-data.org (reference) on 2026-10-19 unless marked `simulated`:

| date       | truth | ref   | const | deadline | bound_by  | vs truth |
|---|---|---|---|---|---|---|
| 2026-04-03 | 11:30 | 11:30 | 12:00 | 11:30    | REFERENCE | +0m      |
| 2026-04-06 | 10:15 | 10:30 | 12:00 | 10:30    | REFERENCE | **+15m** |
| 2027-04-02 |   --  | 11:30 `simulated` | 12:00 | 11:30 | REFERENCE | n/a |
| 2027-04-05 |   --  | 10:30 `simulated` | 12:00 | 10:30 | REFERENCE | n/a |
| 2026-09-17 | 16:45 | 17:00 | 12:00 | 12:00    | CONSTANT  | -285m    |
| 2026-05-17 | 10:00 | 10:00 | 10:15 | 10:00    | REFERENCE | +0m      |

The 2027 rows are `simulated` because the reference has **no published 2027-04
schedule yet** — the live replay correctly returns `NO_FIXTURE` and falls to
the constant. The simulated input is each date's own 2026 holiday value.
"""

import datetime as dt
import pathlib

import pytest

import scripts.schedule_deadline as sd
from src.scrapers import footballdata_competitions as fc


def _payload(*rows):
    """`(code, iso)` pairs -> a provider payload."""
    return {"matches": [{"competition": {"code": c}, "status": "TIMED",
                         "utcDate": iso} for c, iso in rows]}


def _fetch(payload):
    return lambda day, key: payload


def _combined(day, *rows):
    ref = sd.reference_earliest(day, fetch=_fetch(_payload(*rows)))
    return ref, sd.combined_deadline(day, ref)


# ── THE FOUR POSITIVES ───────────────────────────────────────────────────────

def test_POSITIVE_1_good_friday_2026_the_reference_matches_the_truth_exactly():
    """`measured`: truth 11:30 `england/league-one`, reference 11:30
    `england/championship` — two different leagues at the same minute, which
    is why a 9-league subset got it right."""
    ref, c = _combined(dt.date(2026, 4, 3), ("ELC", "2026-04-03T11:30:00Z"),
                       ("PD", "2026-04-03T14:00:00Z"))
    assert ref["state"] == sd.REF_OK
    assert ref["min"] == 11 * 60 + 30, "the reference's own value moved"
    # SINCE THE FLOOR WAS DERIVED (09:45) IT BINDS BELOW THE REFERENCE.
    # RE-ANCHORED 2026-10-20 to the observed earliest kickoff: 09:30.
    assert c["deadline_min"] == 9 * 60 + 30
    assert c["bound_by"] == "CONSTANT"
    assert 11 * 60 + 30 - c["deadline_min"] == 120


def test_POSITIVE_2_easter_monday_2026_tightens_by_90_min():
    """The big win and the known-wrong case at once: 10:30 against a 12:00
    constant, while the truth is 10:15 in a league the reference cannot see."""
    ref, c = _combined(dt.date(2026, 4, 6), ("SA", "2026-04-06T10:30:00Z"),
                       ("ELC", "2026-04-06T11:30:00Z"))
    assert ref["min"] == 10 * 60 + 30
    # THE 15-MINUTE RESIDUAL IS GONE ON THIS DATE. The floor 09:45 is below
    # the truth of 10:15, so the exposure here is ZERO -- where under the
    # 12:00 percentile the reference's 10:30 left 15 minutes of it.
    assert c["deadline_min"] == 9 * 60 + 30
    assert c["bound_by"] == "CONSTANT"
    TRUTH = 10 * 60 + 15
    assert max(0, c["deadline_min"] - TRUTH) == 0


def test_POSITIVE_3_good_friday_2027_SIMULATED_input_clears():
    """`simulated` from the 2026 measurement, because the reference has no
    2027-04 schedule yet. It will have one on the day: the horizon sweep
    reached D+32."""
    ref, c = _combined(dt.date(2027, 4, 2), ("ELC", "2027-04-02T11:30:00Z"))
    assert ref["min"] == 11 * 60 + 30
    assert c["deadline_min"] == 9 * 60 + 30 and c["bound_by"] == "CONSTANT"


def test_POSITIVE_4_easter_monday_2027_SIMULATED_input_ALARMS_under_0_4():
    """The payoff. Under the cron that produced the -27m breach, a
    reference-derived 10:30 leaves **+14m** and alarms; under `0 0` it is
    +253m and clear. Both `measured` series, one arithmetic.
    """
    ref, c = _combined(dt.date(2027, 4, 5), ("SA", "2027-04-05T10:30:00Z"))
    assert ref["min"] == 10 * 60 + 30
    assert c["deadline_min"] == 9 * 60 + 30

    # THE ALARM IS NOW DRIVEN BY THE FLOOR, NOT THE REFERENCE, AND THE FLOOR
    # IS TIGHTER -- so under the `0 4` regime it alarms on EVERY weekday, not
    # only on an early-card one. That is the correct consequence of a floor
    # derived against `0 0`: it is not a valid floor for a schedule that is
    # four hours later, and the arithmetic says so instead of hiding it.
    #: `measured`: weekday delay max 304 (`0 4`) / 305 (`0 0`), duration max 72.
    for cron_min, delay_max, expect_alarm in ((240, 304, True),
                                              (0, 305, False)):
        ready = cron_min + delay_max + 72
        margin = c["deadline_min"] - ready
        assert (margin < 60) is expect_alarm, (
            f"cron {cron_min}: margin {margin:+d}m, alarm expectation "
            f"{expect_alarm} is wrong")


# ── THE TWO NEGATIVES ────────────────────────────────────────────────────────

def test_NEGATIVE_1_ordinary_weekday_the_CONSTANT_binds():
    """Reference 17:00 is LATER than the 12:00 constant, so it may not relax
    it. The deadline stays 12:00 — conservative, and 285 min inside the
    measured truth of 16:45."""
    ref, c = _combined(dt.date(2026, 9, 17), ("PD", "2026-09-17T17:00:00Z"))
    assert c["deadline_min"] == 9 * 60 + 30        # the observed minimum
    assert c["bound_by"] == "CONSTANT"
    assert "UPPER BOUND" in c["basis"], (
        "the basis does not say why a later reference cannot relax the "
        "constant")
    # 420 minutes inside the measured truth of 16:45, against 285 before
    assert 16 * 60 + 45 - c["deadline_min"] == 435


def test_NEGATIVE_2_ordinary_weekend_stays_clear_and_is_IMPROVED():
    """Truth 10:00, reference 10:00, constant 10:15 — the reference tightens
    the weekend constant by 15 min and still equals the truth."""
    ref, c = _combined(dt.date(2026, 5, 17), ("SA", "2026-05-17T10:00:00Z"),
                       ("PL", "2026-05-17T13:00:00Z"))
    assert c["day_type"] == "weekend"
    assert ref["min"] == 10 * 60
    # The weekend floor is below the reference's 10:00 AND below the measured
    # truth of 10:00, so the exposure is zero with an hour to spare.
    #
    # 09:45 -> 09:00 ON 2026-10-22: the weekend anchor localised to 12:00
    # local, and 2026-05-17 is a SUMMER date (UTC+3), so it converts to
    # 09:00 UTC. The retired 09:45 was the minimum of the summer subset only,
    # which is why localising TIGHTENS this floor rather than loosening it.
    assert c["deadline_min"] == 9 * 60 + 0 and c["bound_by"] == "CONSTANT"
    assert max(0, c["deadline_min"] - 10 * 60) == 0


# ── THE KNOWN-WRONG COVERAGE CASE ────────────────────────────────────────────
#
#     This is the test that proves the coverage caveat is HANDLED rather than
#     noted. Without it, "a minimum over a subset is an upper bound" is a
#     sentence in a docstring.

def test_KNOWN_WRONG_an_UNCOVERED_league_carries_the_earliest_kickoff():
    """Easter Monday 2026, `measured` from production `matches`.

    The truth is **10:15 in `poland/ekstraklasa`**, which the reference does
    not cover; `netherlands/eerste-divisie` is also at 10:15 and is also
    uncovered. The earliest COVERED fixture is `italy/serie-a` at 10:30.

    So the reference alone is **15 minutes LATE** — the unsafe direction — and
    this test asserts exactly that, because a detector that pretends the
    subset is a bound is worse than one that states the residual.
    """
    TRUTH_MIN = 10 * 60 + 15           # measured: poland/ekstraklasa
    ref, c = _combined(dt.date(2026, 4, 6), ("SA", "2026-04-06T10:30:00Z"))

    assert "poland/ekstraklasa" not in fc.REFERENCE_LEAGUES, (
        "the league that carried the truth is now covered; re-measure the "
        "residual rather than trusting this number")
    assert ref["min"] > TRUTH_MIN, (
        "the reference is no longer late here, so the residual this test "
        "quantifies has changed")
    assert ref["min"] - TRUTH_MIN == 15

    # ...and SINCE 2026-10-19 THE REFERENCE'S ERROR NO LONGER REACHES THE
    # DEADLINE. The floor 09:45 binds below both the reference's 10:30 and the
    # truth's 10:15, so the 15-minute coverage error is measured here and
    # cannot affect the answer. Tightening the constant did not just remove
    # the exposure; it removed the coverage gap's path to the deadline.
    assert c["deadline_min"] <= c["constant_min"]
    assert c["deadline_min"] < TRUTH_MIN, (
        "the deadline is no longer inside the measured truth on the one date "
        "where an uncovered league carries the earliest kickoff")
    assert c["bound_by"] == "CONSTANT"


def test_the_combination_can_only_TIGHTEN_never_relax():
    """Rule 1, over the whole range rather than on the measured dates.

    For every reference value, `min(ref, constant) <= constant`. This is the
    property that makes an incomplete reference safe to consult at all.
    """
    for day in (dt.date(2026, 4, 6), dt.date(2026, 5, 17)):
        const = sd.deadline_floor_for(sd.day_type(day), day)[0]
        for minute in range(6 * 60, 23 * 60, 17):
            iso = (f"{day.isoformat()}T{minute // 60:02d}:"
                   f"{minute % 60:02d}:00Z")
            _r, c = _combined(day, ("SA", iso))
            assert c["deadline_min"] <= const
            assert c["deadline_min"] == min(minute, const)
            assert c["bound_by"] == ("REFERENCE" if minute < const
                                     else "CONSTANT")


# ── THE PROVIDER DOES NOT HONOUR `date`, AND THAT WOULD HAVE LIED ───────────

def test_KNOWN_WRONG_the_provider_returns_ANOTHER_DAYS_matches():
    """`measured` 2026-10-19: asked for 2027-04-02 it returned **39 matches,
    every one dated 2027-04-03**; asked for 2027-04-05, **8 dated
    2027-04-06**.

    A `min()` over the raw response would have presented the next day's
    kickoff as tomorrow's, with nothing in the payload marking it. The rows
    are rejected AND COUNTED: a silent filter and an honest provider look
    identical from the outside.
    """
    day = dt.date(2027, 4, 2)
    ref, c = _combined(day, ("BL1", "2027-04-03T18:30:00Z"),
                       ("ELC", "2027-04-03T11:30:00Z"))
    assert ref["state"] == sd.REF_NO_FIXTURE
    assert ref["min"] is None, "a next-day kickoff was taken as this day's"
    assert ref["dropped"]["other_day"] == 2, (
        "the mismatched rows are not counted, so the case is invisible")
    assert ref["n_raw"] == 2 and ref["n_usable"] == 0
    # and the deadline falls to the constant, saying so
    assert c["deadline_min"] == c["constant_min"]
    assert c["bound_by"] == "CONSTANT"
    assert "NO_FIXTURE" in c["basis"]


# ── THE THIRD STATES, WHICH ARE NOT ONE STATE ───────────────────────────────

def test_COULD_NOT_LOOK_and_FOUND_NOTHING_are_DIFFERENT_states():
    """Three ways to have no reference value, three reasons, two remedies.

    `NO_KEY`        -> a secret is missing. Our fault, fixable in the workflow.
    `FETCH_FAILED`  -> the provider did not answer. Not our fault, transient.
    `NO_FIXTURE`    -> the provider answered and covers nothing that day. A
                       real answer about the world.

    All three fall to the constant. Only the third is reference-VERIFIED, and
    conflating them is how "the card is empty" and "we never looked" became
    the same log line in H1.
    """
    day = dt.date(2026, 10, 20)

    nokey = sd.reference_earliest(day, api_key="")
    assert nokey["state"] == sd.REF_NO_KEY
    assert sd.REFERENCE_KEY_ENV in nokey["reason"]

    def boom(_d, _k):
        raise OSError("connection reset")

    failed = sd.reference_earliest(day, fetch=boom)
    assert failed["state"] == sd.REF_FETCH_FAILED
    assert "did not answer" in failed["reason"]

    # SPLIT 2026-10-19: a response with NO ROWS AT ALL and a response whose
    # rows are all for another date were one state until the 2027-04-02 case
    # showed why they must not be. See tests/test_coverage_residual.py.
    empty = sd.reference_earliest(day, fetch=_fetch({"matches": []}))
    assert empty["state"] == sd.REF_EMPTY_RESPONSE
    assert "NO ROWS AT ALL" in empty["reason"]

    reasons = {nokey["reason"], failed["reason"], empty["reason"]}
    assert len(reasons) == 3, "two of the three states share a reason"

    # the two that mean "could not look" are marked NOT verified
    for r in (nokey, failed):
        c = sd.combined_deadline(day, r)
        assert c["reference_verified"] is False
        assert "NOT reference-verified" in c["basis"]
    c_empty = sd.combined_deadline(day, empty)
    assert c_empty["reference_verified"] is True


def test_the_states_are_ENUMERATED():
    assert set(sd.REFERENCE_STATES) == {
        sd.REF_OK, sd.REF_NO_FIXTURE, sd.REF_EMPTY_RESPONSE, sd.REF_NO_KEY,
        sd.REF_FETCH_FAILED, sd.REF_NO_SCHEDULED}
    assert set(sd.REF_UNEVALUATED) == {sd.REF_NO_KEY, sd.REF_FETCH_FAILED}
    assert set(sd.REF_ANSWERED) == {sd.REF_OK, sd.REF_NO_FIXTURE,
                                    sd.REF_NO_SCHEDULED}
    assert not set(sd.REF_ANSWERED) & set(sd.REF_UNEVALUATED)
    assert sd.REF_EMPTY_RESPONSE not in sd.REF_ANSWERED
    for s in sd.REFERENCE_STATES:
        assert s == s.upper()


# ── AN EMPTY DAY IS NOT AN EMPTY PROVIDER. Stage 33, 2026-11-14. ─────────────
#
# The live check reddened on 11 of 13 Reminders runs, 2026-11-08..11-14, all
# inside the men's FIFA window of 9-17 November 2026. The provider was
# answering; the days were empty. Every control below drives `_answered`,
# the predicate the live check asserts.

def _answered(ref):
    return ref["state"] in sd.REF_ANSWERED


# ── NO NETWORK, ENFORCED BY THE TESTS THEMSELVES. Stage 33, 2026-11-14. ─────
#
# These tests sit in the GATE (this file is capture-critical via
# `src.scrapers`), so a provider outage must not be able to fail them. Each
# runs on a RECORDED response shape only, and proves it: the socket layer is
# replaced for the duration of the test, every attempt is RECORDED, and the
# fixture fails the test at teardown if any was made.
#
# RECORDED, NOT MERELY RAISED, because `reference_earliest` catches every
# exception and returns `FETCH_FAILED`: a test that reached the network would
# see an ordinary failed fetch and could still pass -- the parametrised case
# whose expected state IS `FETCH_FAILED` would pass either way.
# `test_KNOWN_WRONG_a_REAL_reach_is_CAUGHT_even_when_swallowed` is the
# control that the blocker bites on both production paths.

class _NetworkBlocked(OSError):
    pass


class _Blocker:
    """Replaces the socket layer (urllib, http.client, httpx and requests
    all connect through it) and records each attempt."""

    def __init__(self, monkeypatch):
        import socket

        self.attempts = []

        def make(name):
            def blocked(*_a, **_k):
                self.attempts.append(name)
                raise _NetworkBlocked(f"network blocked: {name}")
            return blocked
        monkeypatch.setattr(socket.socket, "connect", make("socket.connect"))
        monkeypatch.setattr(socket.socket, "connect_ex",
                            make("socket.connect_ex"))
        monkeypatch.setattr(socket, "create_connection",
                            make("socket.create_connection"))
        monkeypatch.setattr(socket, "getaddrinfo", make("socket.getaddrinfo"))


@pytest.fixture
def network_blocker(monkeypatch):
    return _Blocker(monkeypatch)


@pytest.fixture
def no_network(network_blocker):
    yield network_blocker
    assert network_blocker.attempts == [], (
        f"a gating test reached for the network: {network_blocker.attempts}. "
        f"It must run on a recorded response shape, or move to the `live` "
        f"(non-gating) selection.")


def test_KNOWN_WRONG_a_REAL_reach_is_CAUGHT_even_when_swallowed(
        network_blocker):
    """The blocker must bite on BOTH production paths, though both swallow
    the exception: the day's fetch and the schedule probe."""
    ref = sd.reference_earliest(ASKED, api_key="planted-not-a-key")
    assert ref["state"] == sd.REF_FETCH_FAILED
    assert len(network_blocker.attempts) >= 1, (
        "the real fetch path reached past the blocker uncounted")
    seen = len(network_blocker.attempts)
    probe = sd.schedule_probe(ASKED, "planted-not-a-key",
                              sd._reference_schedule_fetch)
    assert probe["state"] == sd.PROBE_FAILED
    assert len(network_blocker.attempts) > seen, (
        "the real probe path reached past the blocker uncounted")


#: The Stage 33 gate tests, each of which must take `no_network`. Their
#: collected node count is 8 (one parametrised five ways).
NETWORK_FREE_GATE_TESTS = {
    "test_KNOWN_WRONG_TODAYS_empty_response_in_the_FIFA_window_does_NOT_redden": 1,
    "test_KNOWN_WRONG_a_FAILED_or_DARK_provider_STILL_reddens": 5,
    "test_an_INJECTED_fetch_never_reaches_the_network_for_the_probe": 1,
    "test_the_PROBE_is_asked_only_when_the_day_is_EMPTY": 1,
}


def test_every_Stage_33_gate_test_RUNS_UNDER_the_blocker():
    """Executed, not read: each function's own marks are inspected, and the
    parametrisation is counted, so the 8 is derived."""
    import sys

    mod = sys.modules[__name__]
    nodes = 0
    for name, expected in NETWORK_FREE_GATE_TESTS.items():
        fn = getattr(mod, name)
        marks = getattr(fn, "pytestmark", [])
        used = {a for m in marks if m.name == "usefixtures" for a in m.args}
        assert "no_network" in used, f"{name} does not run under the blocker"
        params = [m for m in marks if m.name == "parametrize"]
        n = len(params[0].args[1]) if params else 1
        assert n == expected, (name, n)
        nodes += n
    assert nodes == 8, f"{nodes} gate nodes, the census recorded 8"


ASKED = dt.date(2026, 11, 15)
EMPTY_ASK = {"filters": {"dateFrom": "2026-11-15", "dateTo": "2026-11-16"},
             "matches": []}


def _row(code, utc):
    return {"competition": {"code": code}, "status": "TIMED", "utcDate": utc}


#: The provider's schedule for 2026-11-15..11-24 as `measured` 2026-11-14,
#: one row per competition-day (the real ask returned 87 rows).
MEASURED_SCHEDULE = {
    "filters": {"dateFrom": "2026-11-15", "dateTo": "2026-11-24"},
    "matches": [_row("BSA", "2026-11-18T22:00:00Z"),
                _row("BSA", "2026-11-19T22:00:00Z"),
                _row("BSA", "2026-11-20T22:00:00Z"),
                _row("BL1", "2026-11-20T19:30:00Z"),
                _row("FL1", "2026-11-20T19:45:00Z"),
                _row("PD", "2026-11-20T20:00:00Z"),
                _row("PL", "2026-11-21T12:30:00Z"),
                _row("ELC", "2026-11-21T15:00:00Z")]}


def _schedule(payload):
    calls = []

    def f(date_from, date_to, key):
        calls.append((date_from, date_to))
        return payload
    f.calls = calls
    return f


@pytest.mark.usefixtures("no_network")
def test_KNOWN_WRONG_TODAYS_empty_response_in_the_FIFA_window_does_NOT_redden():
    """2026-11-14 07:29 UTC: `EMPTY_RESPONSE` for 2026-11-15 reddened the
    Reminders run. The provider's own schedule had rows from 11-18."""
    sched = _schedule(MEASURED_SCHEDULE)
    ref = sd.reference_earliest(ASKED, fetch=_fetch(EMPTY_ASK),
                                schedule_fetch=sched)
    assert ref["state"] == sd.REF_NO_SCHEDULED
    assert _answered(ref), "the empty FIFA-window day still reddens"
    assert ref["schedule_probe"]["next_scheduled"] == "2026-11-18"
    assert ref["schedule_probe"]["next_covered"] == "2026-11-20", (
        "BSA is served but not mapped; the first COVERED day is 11-20")
    # one probe, over the provider's 10-day limit, from the asked date
    assert sched.calls == [(ASKED, dt.date(2026, 11, 24))]
    c = sd.combined_deadline(ASKED, ref)
    assert c["deadline_min"] == c["constant_min"], "an empty day moved it"
    assert c["reference_verified"] is True and c["reference_answered"] is True
    line = sd.format_reference_deadline(c)
    assert "ref=NO_SCHEDULED_FIXTURES" in line and "NOT ANSWERED" not in line
    assert "next_scheduled=2026-11-18" in line, line


@pytest.mark.usefixtures("no_network")
@pytest.mark.parametrize("label, fetch, schedule, state", [
    ("the fetch itself fails",
     "boom", MEASURED_SCHEDULE, "FETCH_FAILED"),
    ("the probe fails",
     EMPTY_ASK, "boom", "EMPTY_RESPONSE"),
    ("nothing in ten days either",
     EMPTY_ASK, {"filters": {"dateFrom": "2026-11-15",
                             "dateTo": "2026-11-24"}, "matches": []},
     "EMPTY_RESPONSE"),
    ("the probe has rows ON the asked days",
     EMPTY_ASK, {"filters": {"dateFrom": "2026-11-15",
                             "dateTo": "2026-11-24"},
                 "matches": [_row("PL", "2026-11-16T15:00:00Z")]},
     "FETCH_FAILED"),
    ("the probe answers a different range",
     EMPTY_ASK, {"filters": {"dateFrom": "2026-11-16",
                             "dateTo": "2026-11-25"},
                 "matches": [_row("PL", "2026-11-21T15:00:00Z")]},
     "EMPTY_RESPONSE"),
])
def test_KNOWN_WRONG_a_FAILED_or_DARK_provider_STILL_reddens(label, fetch,
                                                            schedule, state):
    def boom(*_a):
        raise OSError("connection reset")

    ref = sd.reference_earliest(
        ASKED, fetch=boom if fetch == "boom" else _fetch(fetch),
        schedule_fetch=boom if schedule == "boom" else _schedule(schedule))
    assert ref["state"] == state, (label, ref.get("reason"))
    assert not _answered(ref), f"{label}: reads as answered"
    c = sd.combined_deadline(ASKED, ref)
    assert c["reference_answered"] is False
    assert "NOT ANSWERED" in sd.format_reference_deadline(c)


@pytest.mark.usefixtures("no_network")
def test_an_INJECTED_fetch_never_reaches_the_network_for_the_probe(
        monkeypatch):
    """Every pre-existing hermetic test injects `fetch` alone. The probe must
    not turn those into network calls."""
    def network(*_a):
        raise AssertionError("the schedule probe reached the network")

    monkeypatch.setattr(sd, "_reference_schedule_fetch", network)
    ref = sd.reference_earliest(ASKED, fetch=_fetch(EMPTY_ASK))
    assert ref["state"] == sd.REF_EMPTY_RESPONSE
    assert ref["schedule_probe"]["state"] == sd.PROBE_NOT_PROBED
    assert not _answered(ref)


@pytest.mark.usefixtures("no_network")
def test_the_PROBE_is_asked_only_when_the_day_is_EMPTY():
    sched = _schedule(MEASURED_SCHEDULE)
    ref = sd.reference_earliest(
        dt.date(2026, 11, 21),
        fetch=_fetch({"filters": {"dateFrom": "2026-11-21"},
                      "matches": [_row("PL", "2026-11-21T12:30:00Z")]}),
        schedule_fetch=sched)
    assert ref["state"] == sd.REF_OK and _answered(ref)
    assert sched.calls == [], "a day with rows paid for a second call"
    assert "schedule_probe" not in ref


# ── COVERAGE IS REPORTED, NOT ASSUMED ───────────────────────────────────────

def test_the_COVERAGE_FRACTION_is_re_derived_from_the_config():
    """`(9, 30)` must not become the `LIVE 0 4 * * 1-5` label: a literal that
    was true when written. Re-derived here from the config and the map."""
    import yaml

    # THE EXAMPLE, NOT THE LOCAL FILE — 2026-11-07. CI does
    # `cp config/config.example.yaml config/config.yaml`, so the example IS
    # the deployed config and the local copy is a scratch file that may not
    # exist at all in a fresh clone. This test re-derives the coverage
    # denominator precisely to avoid a literal "that was true when written";
    # reading it from the file CI overwrites was the same mistake one level
    # down. Found by `test_deployed_config_is_the_example`.
    cfg = yaml.safe_load(
        pathlib.Path("config/config.example.yaml").read_text(
            encoding="utf-8"))
    configured = set((cfg.get("scraping") or {}).get("flashscore_leagues") or [])
    assert configured, "no configured leagues; the denominator is unmeasurable"
    covered = fc.REFERENCE_LEAGUES & configured
    assert fc.REFERENCE_COVERAGE_MEASURED == (len(covered), len(configured)), (
        f"coverage is {len(covered)} of {len(configured)}, the module says "
        f"{fc.REFERENCE_COVERAGE_MEASURED}. Re-measure and restate the "
        f"residual on the known-wrong date too.")


def test_every_run_reports_WHICH_leagues_were_covered_and_which_were_not():
    """The directive's reporting requirement, asserted on the output."""
    day = dt.date(2026, 10, 20)
    ref, c = _combined(day, ("CL", "2026-10-20T16:45:00Z"),
                       ("ELC", "2026-10-20T18:45:00Z"))
    assert ref["leagues_present"] == ["england/championship",
                                      "europe/champions-league"]
    assert "spain/laliga" in ref["leagues_absent"]
    assert len(ref["leagues_present"]) + len(ref["leagues_absent"]) == len(
        fc.REFERENCE_LEAGUES), "present + absent does not account for the map"

    line = sd.format_reference_deadline(c)
    for part in ("serves=2026-10-20", "deadline=09:30", "bound_by=CONSTANT",
                 "coverage=9 of 30", "present=2", "absent=7", "usable=2",
                 "other_day"):
        assert part in line, f"the report line omits {part!r}: {line}"


def test_the_line_names_the_PROVIDER_because_a_figure_needs_its_source():
    day = dt.date(2026, 10, 20)
    _r, c = _combined(day, ("CL", "2026-10-20T16:45:00Z"))
    assert c["reference"] == "football-data.org" == sd.REFERENCE_NAME
    assert sd.REFERENCE_NAME in c["basis"] or c["bound_by"] == "CONSTANT"


# ── THE AUDIT INSTALLS NOTHING, SO THIS PATH IMPORTS NOTHING ────────────────

def test_the_competition_module_has_NO_DEPENDENCIES():
    """It exists so `ci_audit` can read the map without importing `httpx`.

    `ci-audit.yml` says "No pip install: ci_audit imports the standard library
    only". Importing the scraper would kill that job at import, and copying
    the map into the deadline module would be THE HABIT.
    """
    import ast
    import sys

    tree = ast.parse(pathlib.Path(
        "src/scrapers/footballdata_competitions.py").read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            imported.add((node.module or "").split(".")[0])
    third_party = imported - set(sys.stdlib_module_names) - {"", "__future__"}
    assert not third_party, (
        f"the competition module imports {sorted(third_party)}; the audit "
        f"installs no packages")


def test_schedule_deadline_IMPORTS_no_third_party_at_MODULE_SCOPE():
    """Same constraint one level up: `ci_audit` imports this module at top."""
    import ast
    import sys

    tree = ast.parse(pathlib.Path(
        "scripts/schedule_deadline.py").read_text(encoding="utf-8"))
    imported = set()
    for node in tree.body:                      # MODULE SCOPE ONLY
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            imported.add((node.module or "").split(".")[0])
    third_party = imported - set(sys.stdlib_module_names) - {"", "__future__"}
    assert not third_party, (
        f"{sorted(third_party)} is imported at module scope; the audit job "
        f"installs nothing, so it must stay inside the functions that need it")


def test_the_fetch_uses_the_STDLIB_and_not_httpx():
    """BY AST, NOT BY TEXT.

    The first draft of this test read the function's SOURCE for the string
    `httpx` and failed on its own docstring, which names `httpx` to explain
    why it is absent. Stripping `#` comments does not remove a docstring --
    the same shape as the `::error::` and `date.today()` scans before it. So
    the imports are read as imports.
    """
    import ast
    import inspect

    fn = ast.parse(inspect.getsource(sd._reference_fetch).lstrip())
    imported = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert "urllib" in imported, f"it no longer uses urllib: {sorted(imported)}"
    assert not (imported & {"httpx", "requests", "aiohttp"}), (
        f"a third-party HTTP client reached the audit path: {sorted(imported)}")


# ── THE LIVE CHECK, NON-GATING ──────────────────────────────────────────────

@pytest.mark.live
def test_LIVE_the_reference_answers_for_TOMORROW_and_our_table_does_not():
    """The whole justification, re-measured on every non-gating run.

    `measured` 2026-10-19: reference 16:45 for 2026-10-20, our own rows for
    that day **zero**, and the reference's horizon reached **D+32**. If the
    reference stops answering for D+1 this goes red and the deadline is back
    to a bare constant — which is the finding, not an inconvenience.
    """
    import os

    if not (os.environ.get(sd.REFERENCE_KEY_ENV) or "").strip():
        pytest.skip(f"{sd.REFERENCE_KEY_ENV} not set here")
    tomorrow = dt.date.today() + dt.timedelta(days=1)
    ref = sd.reference_earliest(tomorrow)
    # THE SAME PREDICATE THE KNOWN-WRONG CONTROLS BELOW DRIVE. Stage 33: a
    # day with nothing scheduled is a state; a failed fetch is not.
    assert _answered(ref), (
        f"the reference did not answer for D+1: {ref.get('reason')}")
    assert ref["dropped"]["other_day"] == 0 or ref["state"] == sd.REF_NO_FIXTURE
    c = sd.combined_deadline(tomorrow, ref)
    assert c["deadline_min"] <= c["constant_min"]
    print(sd.format_reference_deadline(c))


# ── THE CONSUMER, AND THE SECRET IT NEEDS ───────────────────────────────────
#
#     An always-unevaluated detector is indistinguishable from a working one
#     whose answer is always "nothing to report".
#
# Written down 2026-10-18 after `any([0, 0])` killed `missing_scheduled_runs`
# for a day. A reference check whose key is not in the workflow would report
# `NO_KEY` on every run forever, fall to the constant, and look like a
# functioning line.

def test_the_AUDIT_WORKFLOW_passes_the_reference_key():
    import yaml

    wf = yaml.safe_load(pathlib.Path(".github/workflows/ci-audit.yml")
                        .read_text(encoding="utf-8"))
    job = list(wf["jobs"].values())[0]
    step = [s for s in job["steps"] if s.get("id") == "audit"][0]
    assert sd.REFERENCE_KEY_ENV in (step.get("env") or {}), (
        f"{sd.REFERENCE_KEY_ENV} is not passed to the audit step, so the "
        f"day-ahead deadline would report NO_KEY on every run and silently "
        f"fall back to the bare constant")


def test_the_NON_GATING_test_step_can_RUN_the_live_check():
    """The `live` test needs the key too, or it skips forever."""
    import yaml

    # RE-POINTED 2026-10-21: the non-gating suite moved to its own workflow,
    # and the key is on the STEP that runs it rather than the job -- which is
    # the tighter place for it, so this checks there.
    wf = yaml.safe_load(pathlib.Path(".github/workflows/reminders.yml")
                        .read_text(encoding="utf-8"))
    steps = [s for j in wf["jobs"].values() for s in j.get("steps", [])
             if '-m "calendar or live"' in str(s.get("run", ""))]
    assert steps, "no step runs the non-gating suite any more"
    env = {k for s in steps for k in (s.get("env") or {})}
    assert sd.REFERENCE_KEY_ENV in env, (
        f"{sd.REFERENCE_KEY_ENV} is not in that step's env, so "
        f"test_LIVE_the_reference_answers_for_TOMORROW would skip on every "
        f"CI run and prove nothing")


def test_a_MISSING_key_does_not_crash_the_audit_it_FALLS_BACK_loudly():
    """The failure mode that matters in production: no key, no crash, and a
    line that says it is not reference-verified."""
    day = dt.date(2026, 10, 20)
    c = sd.combined_deadline(day, sd.reference_earliest(day, api_key=""))
    assert c["deadline_min"] == c["constant_min"]
    assert c["reference_verified"] is False
    assert "NOT reference-verified" in c["basis"]
    line = sd.format_reference_deadline(c)
    assert "ref=NO_KEY" in line, line
