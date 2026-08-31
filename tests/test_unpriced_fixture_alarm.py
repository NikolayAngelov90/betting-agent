"""UNPRICED must be distinguishable from UNWANTED — the 2026-08-29 replay, pinned.

THE DEFECT CLASS THIS PINS. A fixture with no odds produces no pick, and so does
a fixture nobody wanted to bet. They are identical in every output the pipeline
emits, which is how the Stage 20 identity-gate regression survived three days:
Leuven vs St. Liege was refused by the gate, carried ZERO odds, and looked
exactly like a fixture the model had passed on.

MEASURED against the live database 2026-08-31, replaying the two days:

    2026-08-29   18 unpriced -> 3 ALARM, 15 INFO
                 ALARM  Avellino vs L.R. Vicenza        italy/serie-b     peers=5
                 ALARM  Leuven vs St. Liege             belgium/jupiler   peers=3
                 ALARM  Radomiak Radom vs Cracovia      poland/ekstraklasa peers=2
    2026-08-30   10 unpriced -> 0 ALARM, 10 INFO

The Belgian shape is reproduced below as the canonical case: Leuven alarms while
its three same-day peers, which carried 77, 79 and 175 odds rows, stay silent.

THREE PROPERTIES ARE PINNED, and each one is a way the check could rot:

  1. IT IS SELF-CALIBRATING. No threshold, no maintained league list. A fixture
     is compared against its OWN same-league same-day peers, so a quiet league,
     an off-season and an uncovered competition are all silent by construction
     rather than by exclusion.
  2. IT SPLITS BY CAUSE. Unsplit it fired 18 times on a day with 3 real
     problems, which is the `fixtures_zero_active` failure again — a check that
     fires every day is a check that gets ignored.
  3. A DEAD CHECK IS NOT A CLEAN CHECK. Run against a bad credential the first
     version printed "0 unpriced" and returned success. Empty means
     measured-and-clean; None means unmeasured.
  4. IT RUNS AFTER EVERY ODDS PATH. Its first live firing (2026-08-31) sat
     right after `API-Football update complete`, alarmed on match 50969 at
     09:22:51, and 86 odds rows landed at 09:23:01 — false eleven seconds
     later, on a fixture that was then picked. Replayed from the end of the
     run, that day's alarms drop 2 -> 1, and the survivor (50976) genuinely
     still carries no odds.
"""

from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.data.coverage_checks import find_unpriced_fixtures, report_unpriced_fixtures
from src.data.models import Base, Match, Odds, Team

DAY = datetime(2026, 8, 29)
NEXT = datetime(2026, 8, 30)
AFTER = datetime(2026, 8, 31)


@pytest.fixture()
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine)()
    yield s
    s.close()


def _team(s, tid, name, af_id=None):
    s.add(Team(id=tid, name=name, apifootball_team_id=af_id))


def _fixture(s, mid, league, when, home, away, n_odds, af_id=None, fs_id=None):
    """One fixture with `n_odds` price rows attached."""
    s.add(Match(id=mid, league=league, match_date=when,
                home_team_id=home, away_team_id=away,
                apifootball_id=af_id, flashscore_id=fs_id))
    for i in range(n_odds):
        s.add(Odds(match_id=mid, bookmaker=f"book{i}", market_type="1X2",
                   selection="Home", odds_value=2.0))


@pytest.fixture()
def belgium(session):
    """2026-08-29 belgium/jupiler-pro-league, as it actually stood."""
    names = ["Leuven", "St. Liege", "Anderlecht", "Genk", "Gent",
             "Club Brugge", "Westerlo", "Mechelen"]
    for i, n in enumerate(names, start=1):
        _team(session, i, n, af_id=100 + i)

    # THE DEFECT: refused by the identity gate, so it never got an
    # API-Football id and was never priced. It survived via Flashscore.
    _fixture(session, 1, "belgium/jupiler-pro-league", DAY, 1, 2,
             n_odds=0, af_id=None, fs_id="fs-leuven")
    # Its three peers, priced normally, on the same day in the same league.
    _fixture(session, 2, "belgium/jupiler-pro-league", DAY, 3, 4,
             n_odds=77, af_id=9002)
    _fixture(session, 3, "belgium/jupiler-pro-league", DAY, 5, 6,
             n_odds=79, af_id=9003)
    _fixture(session, 4, "belgium/jupiler-pro-league", DAY, 7, 8,
             n_odds=175, af_id=9004)
    session.commit()
    return session


def _alarms(s, lo=DAY, hi=NEXT):
    found = find_unpriced_fixtures(s, lo, hi)
    assert found is not None, "the check did not run"
    return [f for f in found if not f[5]]


# ── the canonical replay ────────────────────────────────────────────────────
def test_names_leuven_and_is_silent_on_its_three_peers(belgium):
    alarms = _alarms(belgium)
    assert len(alarms) == 1, f"expected exactly one alarm, got {alarms}"
    _, league, _, home, away, has_af, has_fs, peers, med = alarms[0]
    assert (home, away) == ("Leuven", "St. Liege")
    assert league == "belgium/jupiler-pro-league"
    assert peers == 3, "the three priced peers are what calibrates the alarm"
    assert med == 79, "median of 77/79/175"
    assert has_af is False, "no API-Football id is the reported cause"
    assert has_fs is True, "the row exists only because Flashscore found it"


def test_report_alarms_once_and_returns_the_count(belgium):
    assert report_unpriced_fixtures(belgium, DAY, NEXT) == 1


# ── property 1: self-calibrating, so silence is by construction ─────────────
def test_a_league_where_NOTHING_is_priced_is_silent(session):
    """An uncovered competition, an off-season, a quiet day: no peers, no alarm."""
    for i, n in enumerate(["A", "B", "C", "D"], start=1):
        _team(session, i, n)
    for mid, (h, a) in enumerate([(1, 2), (3, 4)], start=1):
        _fixture(session, mid, "obscure/cup", DAY, h, a, n_odds=0)
    session.commit()
    assert _alarms(session) == [], (
        "with no priced peer there is no evidence of a gap — this is the "
        "property that removes the need for a maintained league list")


def test_the_same_fixture_on_a_DIFFERENT_day_does_not_calibrate_it(belgium):
    """Peers are same-league AND same-day. Yesterday's prices prove nothing."""
    _team(belgium, 9, "Kortrijk")
    _team(belgium, 10, "Charleroi")
    _fixture(belgium, 5, "belgium/jupiler-pro-league", NEXT, 9, 10, n_odds=0)
    belgium.commit()
    assert _alarms(belgium, NEXT, AFTER) == [], (
        "the only 08-30 fixture has no priced peer ON 08-30; the 08-29 prices "
        "must not be borrowed to calibrate it")


# ── property 2: the split by cause ─────────────────────────────────────────
def test_resolved_but_unpriced_is_INFO_not_an_alarm(belgium):
    """15 of 18 on 08-29 were odds-budget coverage, not identity corruption."""
    _team(belgium, 11, "OH Leuven B")
    _team(belgium, 12, "Beerschot")
    _fixture(belgium, 6, "belgium/jupiler-pro-league", DAY, 11, 12,
             n_odds=0, af_id=9006)
    belgium.commit()

    found = find_unpriced_fixtures(belgium, DAY, NEXT)
    assert len(found) == 2, "both unpriced fixtures are found"
    assert report_unpriced_fixtures(belgium, DAY, NEXT) == 1, (
        "only the unresolved one alarms — a fixture that HAS an API-Football id "
        "and no odds is a budget decision, and counting it would drown the "
        "three that are not")


# ── property 3: a dead check must not read as a clean one ──────────────────
def test_a_failed_query_returns_None_not_empty(session):
    class Dead:
        def execute(self, *a, **k):
            raise RuntimeError("password authentication failed")

    assert find_unpriced_fixtures(Dead(), DAY, NEXT) is None, (
        "returning [] here is how a broken check reports 'nothing wrong' — "
        "this exact failure printed '2026-08-29: 0 unpriced' against a dead "
        "credential on 2026-08-31")
    assert report_unpriced_fixtures(Dead(), DAY, NEXT) == 0


# ── property 4: placement, so the eleven-second window cannot come back ─────
def test_the_check_runs_after_every_odds_path():
    """A structural pin, because the timing bug was invisible to every unit test.

    The check reads finished state. Placed mid-pipeline it reports on a
    half-written table and calls a fixture unpriced that is about to be
    priced. Ordering is the whole correctness argument, so ordering is what
    this asserts.
    """
    import inspect
    from src.agent.betting_agent import FootballBettingAgent

    src = inspect.getsource(FootballBettingAgent.daily_update)
    call = src.index("report_unpriced_fixtures")
    for earlier in ("API-Football update complete",
                    "The Odds API update complete",
                    "Low-coverage backfill failed"):
        assert src.index(earlier) < call, (
            f"the unpriced check now runs BEFORE {earlier!r} — it would report "
            "on odds that have not been written yet, which is exactly the "
            "false alarm of 2026-08-31 (50969: alarm at 09:22:51, 86 odds "
            "rows at 09:23:01)")
    assert call < src.index("Daily update cycle complete"), (
        "the check must still be inside daily_update, before it reports done")


# ── property 5: the dead-check detector must not itself be dead ────────────
def test_every_way_the_check_can_fail_reaches_the_audit():
    """It did not, for two days, and nothing failed.

    `ci_audit.extract` int()s each pattern's last match and swallowed
    TypeError/ValueError. `unpriced_check_dead` has no numeric capture group,
    so it matched, failed to convert, and never reached `facts` — the
    assertion written to catch a dead check was itself dead. Both failure
    paths are asserted here because they emit from different call sites: the
    inner one from `report_unpriced_fixtures`, the outer one from
    `daily_update` when the call raises before reporting.
    """
    from scripts.ci_audit import assertions, extract

    inner = ("UNPRICED FIXTURE CHECK DID NOT RUN — the query failed, so this "
             "run has NO evidence either way about unpriced fixtures.")
    outer = ("UNPRICED FIXTURE CHECK DID NOT RUN — it raised before reporting "
             "(ImportError). This run has NO evidence either way.")
    for label, line in (("inner", inner), ("outer", outer)):
        assert assertions(extract(line), []), (
            f"the {label} failure path does not reach the audit — a run where "
            "the check never executed would be reported CLEAN")

    assert not assertions(extract("a healthy run says nothing"), []), (
        "silence on a healthy run, or the assertion is noise")
