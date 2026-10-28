"""The WHOLE analysis path, driven on a synthetic collection — 2026-10-28.

> Everything past three-instant existence has run on real data exactly once.

`tests/test_h1_analysis.py` exercises the analysis FUNCTIONS on hand-built
series: `pearson`, `interpret`, `upper_bound_rho`, the NO_DATA states. What it
never does is run the path a real collection takes —

    load_observations -> attach_fixture_identity -> count_qualifying_fixtures
                      -> build_trajectories -> one_price_per_fixture

— against a DATABASE. That gap is why `load_observations` queried `odds` for
46 days while a qualifying triple needs three instants and `odds` carries a
UNIQUE index that holds exactly one: **n was structurally ZERO on every input,
forever**, and every function-level test passed throughout.

So this file drives the path on `tests/h1_synthetic.py`, which builds the
`rho = 0.42` structure in on purpose: several (book, market, selection) series
per fixture, so a count at KEY grain and a count at FIXTURE grain differ by an
order of magnitude and the test can say which it means.
"""

from datetime import date, datetime, timedelta

import pytest

import src.data.database as db_mod
import scripts.h1_analysis as ha
from scripts.h1_collection_check import (
    H1_WINDOW_TO,
    attach_fixture_identity,
    count_qualifying_fixtures,
    load_observations,
    provider_of,
)
from src.data.models import Base
from src.data.odds_quota import H1_RESERVATION_FROM
from tests.h1_synthetic import GAPS, build

#: Inside the configured window, derived so it cannot be left behind when the
#: window moves again.
DAY_IN = H1_RESERVATION_FROM
#: Strictly after the clamp, for the discrimination control.
DAY_OUT = date.fromisoformat(H1_WINDOW_TO) + timedelta(days=2)


@pytest.fixture
def db(tmp_path):
    mgr = db_mod.DatabaseManager(config=type("C", (), {
        "database": {"sqlite_path": str(tmp_path / "path.db")}})())
    Base.metadata.create_all(mgr.engine)
    assert not mgr.is_postgres, "this suite must never reach production"
    return mgr


def _window():
    lo = datetime.combine(H1_RESERVATION_FROM, datetime.min.time())
    return lo, datetime.fromisoformat(H1_WINDOW_TO)


# ═══════════════════ THE PATH, STEP BY STEP, ON A DATABASE ═══════════════

def test_EVERY_STEP_EXECUTES_and_what_each_RETURNS(db, capsys):
    """Six fixtures over three dates. Each step's output is asserted, so a
    step that silently returns nothing cannot be mistaken for a clean run."""
    with db.get_session() as s:
        info = build(s, n_fixtures=6, first_day=DAY_IN, days=3)
    lo, hi = _window()

    with db.get_session() as s:
        obs, dropped = load_observations(s, lo, until=hi)
        assert obs, "step 1 load_observations returned NOTHING"
        assert len(obs) == info["observations"], (len(obs), info)
        # the drop ledger is reported, not implied
        assert set(dropped) >= {"kept"}, dropped

        attach_fixture_identity(s, obs)
        assert all(o.get("fixture") is not None for o in obs), (
            "step 2 attach_fixture_identity left a fixture unidentified")

        series, by_provider = count_qualifying_fixtures(obs)
        assert by_provider, "step 3 found no qualifying fixtures"
        trajectories = ha.build_trajectories(obs)
        assert trajectories, "step 4 build_trajectories returned NOTHING"

    # every observation is TheOddsAPI's, and the strata stay apart
    assert set(by_provider) == {"TheOddsAPI"}, set(by_provider)
    assert provider_of("Pinnacle") == "API-Football", (
        "the provider split collapsed; merging the strata is what H2 died of")


def test_THE_UNIT_IS_THE_FIXTURE_and_the_KEY_count_is_an_ORDER_LARGER(db):
    """A count at key grain overstated `n` a hundredfold, in the direction
    that licenses an analysis. Both grains are computed here so the gap is
    visible rather than asserted."""
    with db.get_session() as s:
        info = build(s, n_fixtures=6, first_day=DAY_IN, days=3)
    lo, hi = _window()
    with db.get_session() as s:
        obs, _ = load_observations(s, lo, until=hi)
        attach_fixture_identity(s, obs)
        series, by_provider = count_qualifying_fixtures(obs)

    n_fixtures = len(by_provider["TheOddsAPI"])
    n_series = series["TheOddsAPI"]
    assert n_fixtures == 6, n_fixtures
    assert n_series == info["series"], (n_series, info["series"])
    assert n_series >= 10 * n_fixtures, (
        f"the fixture is the unit and the key count must be visibly larger "
        f"for this control to mean anything: {n_series} series on "
        f"{n_fixtures} fixtures")
    # and the fixture count is what the stop is fed
    from scripts.h1_collection_check import collection_stop
    stop = collection_stop(n_fixtures=n_fixtures, credits_spent=10,
                           raw_rows=len(obs))
    assert f"n={n_fixtures}/" in stop.reason, (
        "the stop is quoting a grain other than the fixture")


def test_load_observations_READS_odds_snapshots_AND_NOT_odds(db):
    """Pinned because the other table CANNOT hold the input.

    `odds` carries a UNIQUE index on (match_id, bookmaker, market_type,
    selection) and is overwritten in place, so it holds at most ONE instant
    per series. A qualifying triple needs three. Measured 2026-10-10: 10,066
    pre-kickoff series in `odds`, instants-per-series histogram {1: 10066} —
    not one series ever observed twice, for 46 days, while every
    function-level test passed.
    """
    import inspect

    from tests.conftest import anchored, code_of

    # `code_of` AND NOT A HAND-ROLLED STRIPPER. The one-liner
    # `l.strip().startswith("#")` handles comments and is blind to
    # DOCSTRINGS — the half that bit on 10-22 and 10-23 — and the `code_of`
    # ratchet counted two of these the moment they were written (hand 8 -> 10).
    body = anchored(code_of(inspect.getsource(load_observations)),
                    "SELECT", "load_observations")
    assert "odds_snapshots" in body, (
        "load_observations no longer reads odds_snapshots")
    assert "FROM odds " not in body and "from odds " not in body, (
        "load_observations is reading `odds`, which cannot hold three "
        "instants of one series")
    # and the write really lands where the read looks
    with db.get_session() as s:
        build(s, n_fixtures=2, first_day=DAY_IN, days=1)
    lo, hi = _window()
    with db.get_session() as s:
        from sqlalchemy import text
        snaps = s.execute(text("SELECT count(*) FROM odds_snapshots")).scalar()
        obs, _ = load_observations(s, lo, until=hi)
    assert snaps and obs, (snaps, len(obs))


# ════════════════ THE CLAMP, WITH ITS DISCRIMINATION CONTROL ═════════════

def test_the_H1_WINDOW_TO_CLAMP_EXCLUDES_post_window_rows(db):
    """And the control is that WITHOUT it those rows are counted.

    A scheduled collection run fired THREE DAYS PAST the window and wrote
    rows. `load_observations` had only a lower bound, so the analysis would
    have counted them. Last time the bound excluded 14,753 of 40,134
    observations — a 58% inflation avoided — and post-window data entering a
    closed experiment's count is how a dead result becomes a live one by
    accident.
    """
    with db.get_session() as s:
        inside = build(s, n_fixtures=3, first_day=DAY_IN, days=1)
    with db.get_session() as s:
        outside = build(s, n_fixtures=4, first_day=DAY_OUT, days=1,
                        league="spain/laliga")
    lo, hi = _window()

    with db.get_session() as s:
        clamped, _ = load_observations(s, lo, until=hi)
        unclamped, _ = load_observations(s, lo, until=None)
        attach_fixture_identity(s, clamped)
        attach_fixture_identity(s, unclamped)
        _, cl_by = count_qualifying_fixtures(clamped)
        _, un_by = count_qualifying_fixtures(unclamped)

    n_cl = len(cl_by.get("TheOddsAPI", ()))
    n_un = len(un_by.get("TheOddsAPI", ()))
    assert n_cl == 3, f"the clamp dropped in-window fixtures too: {n_cl}"
    assert n_un == 7, (
        f"THE CONTROL FAILED: without the clamp the post-window fixtures were "
        f"not counted either ({n_un}), so this test cannot show the clamp "
        f"does anything")
    assert n_un > n_cl, "the clamp excluded nothing"
    inflation = (n_un - n_cl) / n_cl
    assert inflation > 0.5, (
        f"the planted inflation is only {inflation:.0%}; the control should "
        f"be comparable to the 58% the real bound avoided")


def test_the_CLAMP_IS_PASSED_BY_THE_CALLER_not_remembered(db):
    """`h1_analysis.main` must pass `until`, and its lower bound must come
    from the reservation rather than a literal.

    THE DEFAULT WAS `datetime(2026, 10, 1)` — October's reset — and with the
    clamp at 2026-11-09 that default population was 2026-10-01..11-09: five
    weeks of NORMAL-PRICING snapshots counted as H1 observations. The third
    half-moved copy of the window found on 2026-10-28.
    """
    import inspect

    from tests.conftest import anchored, code_of

    code = anchored(code_of(inspect.getsource(ha.main)),
                    "load_observations", "h1_analysis.main")
    assert "until=until" in code, (
        "main no longer passes the clamp to load_observations")
    # ── STRONGER THAN "DERIVED FROM THE RESERVATION". 2026-10-28. ─────────
    #
    # This asserted `H1_RESERVATION_FROM in code`, which was satisfied by
    # `datetime.combine(H1_RESERVATION_FROM, datetime.min.time())` — a
    # CONSUMER COMPUTING A DATE FROM THE WINDOW, the thing the window module's
    # own rule forbids. The derivation now lives beside the definition and the
    # assertion is that main imports it rather than performing it.
    assert "as_datetimes" in code, (
        "main computes its own lower bound again; `as_datetimes()` exists so "
        "there is one place for that derivation to be wrong")
    assert "datetime.combine" not in code, (
        "main is back to combining a date and a time itself")
    assert "datetime(2026, 10, 1)" not in code, (
        "October's reset is back as the default lower bound")


# ══════════════════════ NO DATA IS NOT A NULL, ANYWHERE ══════════════════

def test_NO_DATA_is_not_a_NULL_at_EVERY_step(db):
    """An empty database must produce an empty-and-SAID-SO at each step, never
    a zero that reads like a measurement."""
    lo, hi = _window()
    with db.get_session() as s:
        obs, dropped = load_observations(s, lo, until=hi)
        assert obs == [], obs
        assert isinstance(dropped, dict), (
            "the drop ledger vanished, so 'nothing loaded' carries no reason")
        attach_fixture_identity(s, obs)          # must not raise on empty
        series, by_provider = count_qualifying_fixtures(obs)
    assert dict(series) == {} and dict(by_provider) == {}
    assert ha.build_trajectories(obs) == []

    # and the interpretation of n=0 is a STATE, not a correlation of zero
    verdict, notes = ha.interpret(0, None, None)
    assert verdict != "", "n=0 produced an empty verdict"
    assert "NO DATA" in verdict.upper() or any(
        "NO DATA" in n.upper() for n in notes), (verdict, notes)
    assert ha.pearson([], []) is None, (
        "pearson of nothing returned a number")


def test_a_SERIES_TOO_CLOSE_TOGETHER_yields_NOTHING_and_the_gaps_are_real(db):
    """The separation rule, driven through the database rather than asserted
    on a list of datetimes.

    The synthetic builder spaces instants 45 and 60 minutes apart — clearing
    MIN_GAP_MINUTES=30 and the 2.0 ratio with margin — so a failure here is
    the predicate and not a boundary coincidence.
    """
    from scripts.h1_collection_check import MAX_INTERVAL_RATIO, MIN_GAP_MINUTES
    assert GAPS == (45, 60), GAPS
    assert min(GAPS) > MIN_GAP_MINUTES, (GAPS, MIN_GAP_MINUTES)
    assert max(GAPS) / min(GAPS) < MAX_INTERVAL_RATIO, GAPS

    with db.get_session() as s:
        build(s, n_fixtures=2, first_day=DAY_IN, days=1)
    lo, hi = _window()
    with db.get_session() as s:
        obs, _ = load_observations(s, lo, until=hi)
        attach_fixture_identity(s, obs)
        # THE ACTUAL GAPS IN MINUTES, not a boolean
        by_key = {}
        for o in obs:
            by_key.setdefault(
                (o["fixture"], o["bookmaker"], o["market"], o["selection"]),
                []).append(o["ts"])
    gaps = set()
    for times in by_key.values():
        ts = sorted(set(times))
        gaps |= {round((b - a).total_seconds() / 60)
                 for a, b in zip(ts, ts[1:])}
    assert gaps == set(GAPS), (
        f"the gaps reaching the predicate are {sorted(gaps)} minutes, not "
        f"{sorted(GAPS)} — the builder and the database disagree")
