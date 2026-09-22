"""The H1 collection check counts SEPARATION, not rows.

Stage 26 Part B requirement 3 existed only as a paragraph until 2026-09-22.
The failure it guards against is measured, not hypothetical: fourteen fixtures
appeared to carry three pre-kickoff points and were six-minute pairs written by
one run.

    A purchase can burn its ceiling producing duplicates and report success.

These tests are the reason the check is written nine days before collection
rather than on its first morning.
"""

from datetime import datetime, timedelta

import pytest

import scripts.h1_collection_check as h1

T0 = datetime(2026, 10, 2, 12, 0)


def _t(*minutes):
    return [T0 + timedelta(minutes=m) for m in minutes]


# ── SEPARATION ───────────────────────────────────────────────────────────────

def test_six_minute_duplicates_are_NOT_a_trajectory():
    """THE MEASURED FAILURE, replayed. Three rows, no separation."""
    assert h1.qualifying_triple(_t(0, 6, 12)) is None, (
        "three points six minutes apart counted as a trajectory — this is the "
        "exact shape of the fourteen fixtures that inflated the earlier count")


def test_two_separated_points_are_not_enough():
    """H1 needs t0->t1 AND t1->t2. Two points is one interval, not a series."""
    assert h1.qualifying_triple(_t(0, 120)) is None


def test_a_clean_120_minute_cadence_qualifies():
    """The designed cadence: window 360, interval 120."""
    got = h1.qualifying_triple(_t(0, 120, 240))
    assert got == tuple(_t(0, 120, 240))


def test_exactly_thirty_minutes_is_ACCEPTED():
    """The registration says >= 30, so the boundary is in, not out."""
    assert h1.qualifying_triple(_t(0, 30, 60)) is not None


def test_twenty_nine_minutes_is_REFUSED():
    assert h1.qualifying_triple(_t(0, 29, 58)) is None


# ── COMPARABILITY, the within-fixture control ────────────────────────────────

def test_lopsided_intervals_are_REFUSED():
    """30 then 300 is drift against autocorrelation, not a control.

    Both gaps clear 30 minutes; the ratio is 10x. Excluding this is what makes
    t0->t1 a like-for-like control for t1->t2.
    """
    assert h1.qualifying_triple(_t(0, 30, 330)) is None


def test_the_ratio_boundary_is_strict():
    """Exactly 2x is excluded: the rule is 'differ by < 2x'."""
    assert h1.qualifying_triple(_t(0, 30, 90)) is None      # 30 vs 60 == 2.0
    assert h1.qualifying_triple(_t(0, 30, 89)) is not None   # 30 vs 59 < 2.0


def test_the_triple_is_SEARCHED_not_taken_in_order():
    """A dense point early must not destroy a trajectory that exists.

    Greedy left-to-right picks 0 and 35 and then fails the ratio on 155. The
    qualifying triple 35/95/155 is present and must be found.
    """
    assert h1.qualifying_triple(_t(0, 35, 95, 155)) is not None


def test_duplicates_cannot_MANUFACTURE_separation():
    """Noise around two real points is still two real points."""
    assert h1.qualifying_triple(_t(0, 1, 2, 3, 120, 121, 122)) is None


# ── THE UNIT IS THE FIXTURE ──────────────────────────────────────────────────

def _obs(fixture, bookmaker, minutes, market="1X2", selection="Home"):
    return [{"fixture": fixture, "bookmaker": bookmaker, "market": market,
             "selection": selection, "ts": t} for t in _t(*minutes)]


def test_four_qualifying_series_on_ONE_fixture_count_as_ONE():
    """H5's sigma came back fifty-fold wrong on exactly this distinction."""
    obs = []
    for sel in ("Home", "Draw", "Away", "Over 2.5"):
        obs += _obs(7, "TheOddsAPI-pinnacle", (0, 120, 240), selection=sel)
    series, fixtures = h1.count_qualifying_fixtures(obs)
    assert len(fixtures["TheOddsAPI"]) == 1, (
        "one fixture counted more than once — n would be inflated fourfold")
    assert series["TheOddsAPI"] == 4


def test_providers_are_counted_APART_never_pooled():
    """+37.3% against -0.61%: different instruments, not different samples."""
    obs = (_obs(1, "TheOddsAPI-pinnacle", (0, 120, 240))
           + _obs(2, "Pinnacle", (0, 120, 240)))
    _, fixtures = h1.count_qualifying_fixtures(obs)
    assert len(fixtures["TheOddsAPI"]) == 1
    assert len(fixtures["API-Football"]) == 1
    assert set(fixtures) == {"TheOddsAPI", "API-Football"}


def test_provider_is_read_from_the_bookmaker_string():
    assert h1.provider_of("TheOddsAPI-unibet_se") == "TheOddsAPI"
    assert h1.provider_of("Bet365") == "API-Football"


def test_points_from_DIFFERENT_books_do_not_form_a_trajectory():
    """A series is one book's price series; splicing two books invents moves."""
    obs = (_obs(1, "TheOddsAPI-pinnacle", (0,))
           + _obs(1, "TheOddsAPI-betsson", (120,))
           + _obs(1, "TheOddsAPI-matchbook", (240,)))
    _, fixtures = h1.count_qualifying_fixtures(obs)
    assert not fixtures


# ── OVERROUND: the band, and the third state ─────────────────────────────────

def test_an_incomplete_market_is_NOT_COMPUTABLE_not_out_of_band():
    """Two legs of a three-way market give a number that is simply wrong."""
    assert h1.overround({"Home": 2.0, "Away": 4.0}, "1X2") is None
    assert not h1.in_band(None)


def test_the_two_way_trap_is_excluded_by_the_band():
    """H2 read +0.705% and was the draw-excluded market.

    A 1X2 priced as a two-way sums near 1.0 once the draw is gone, which is
    below the floor — so the band catches it without naming the trap.
    """
    assert h1.overround({"Home": 2.0, "Draw": 3.6, "Away": 3.9}, "1X2") > 1.005
    two_way = h1.overround({"Home": 2.02, "Draw": 1000.0, "Away": 1.99}, "1X2")
    assert not h1.in_band(two_way), two_way


def test_a_normal_book_is_in_band():
    assert h1.in_band(h1.overround(
        {"Home": 2.10, "Draw": 3.40, "Away": 3.60}, "1X2"))


def test_a_stale_or_absurd_line_is_out_of_band():
    assert not h1.in_band(h1.overround(
        {"Home": 1.20, "Draw": 2.00, "Away": 2.00}, "1X2"))


# ── THE THREE OUTCOMES, WHICH MUST READ DIFFERENTLY ──────────────────────────

def test_no_observations_reads_as_DID_NOT_RUN():
    out = "\n".join(h1.render(0, {}, {}, {}, raw_rows=0, used=0))
    assert "H1 COLLECTION DID NOT RUN" in out
    assert "PRODUCED NOTHING" not in out


def test_observations_but_no_trajectories_reads_as_PRODUCED_NOTHING():
    """L2 shipped inert and measured as 'implemented, 0 saved'."""
    out = "\n".join(h1.render(0, {}, {}, {}, raw_rows=4200, used=120))
    assert "PRODUCED NOTHING" in out
    assert "DID NOT RUN" not in out
    assert "spending and not collecting" in out


def test_progress_reads_as_n_of_39():
    out = "\n".join(h1.render(11, {"TheOddsAPI": set(range(11))},
                              {"TheOddsAPI": 40}, {}, raw_rows=9000, used=90))
    assert "H1 COLLECTION: 11 of 39 qualifying trajectories" in out
    assert "DID NOT RUN" not in out and "PRODUCED NOTHING" not in out


def test_the_three_lines_are_mutually_exclusive():
    """Two of them in one report would make the ledger unreadable."""
    markers = ("DID NOT RUN", "PRODUCED NOTHING", "qualifying trajectories")
    for n, raw in ((0, 0), (0, 500), (5, 500)):
        out = "\n".join(h1.render(n, {"TheOddsAPI": set(range(n))} if n else {},
                                  {}, {}, raw_rows=raw, used=0))
        assert sum(m in out for m in markers) == 1, out


# ── THE STOP CONDITIONS, WHICH MUST BE VISIBLE ───────────────────────────────

def test_reaching_the_target_says_so():
    out = "\n".join(h1.render(39, {"TheOddsAPI": set(range(39))},
                              {}, {}, raw_rows=9000, used=100))
    assert "TARGET REACHED" in out and "Restore the window" in out


def test_the_credit_ceiling_says_so_even_below_target():
    """Whichever comes first. A ceiling hit at n=3 still stops the run."""
    out = "\n".join(h1.render(3, {"TheOddsAPI": {1, 2, 3}}, {}, {},
                              raw_rows=900, used=h1.CREDIT_CEILING))
    assert "CREDIT CEILING REACHED" in out
    assert "regardless of n" in out


def test_an_unreadable_ledger_says_UNKNOWN_not_zero():
    """Zero credits used and 'cannot tell' must not print the same."""
    out = "\n".join(h1.render(3, {"TheOddsAPI": {1, 2, 3}}, {}, {},
                              raw_rows=900, used=None))
    assert "UNKNOWN" in out
    assert "CREDIT CEILING REACHED" not in out


def test_a_second_provider_is_flagged_loudly():
    out = "\n".join(h1.render(
        2, {"TheOddsAPI": {1, 2}, "API-Football": {9}}, {}, {},
        raw_rows=900, used=10))
    assert "MORE THAN ONE PROVIDER QUALIFIED" in out


# ── THE REGISTERED CONSTANTS ARE NOT TUNEABLE ────────────────────────────────

def test_the_thresholds_match_the_pre_registration():
    """If one of these moves, the experiment moved with it."""
    assert (h1.TARGET_N, h1.CREDIT_CEILING) == (39, 200)
    assert (h1.MIN_GAP_MINUTES, h1.MAX_INTERVAL_RATIO) == (30, 2.0)
    assert (h1.OVERROUND_LO, h1.OVERROUND_HI) == (1.005, 1.25)


# ── MARKET ASSEMBLY: the defect found by RUNNING it, nine days early ─────────

def test_legs_written_seconds_apart_are_ONE_instant():
    """THE DEFECT. Measured on 87,380 production rows 2026-09-22.

    Keying the overround on the exact timestamp gave three 1X2 legs in ~0
    cases and one leg in 32,279 — so every instant scored NOT COMPUTABLE, the
    band dropped everything, and the check printed "PRODUCED NOTHING - the
    apparatus is not working" regardless of what was collected.

    A false alarm indistinguishable from the real one is worse than no alarm.
    """
    keys = [(1, "TheOddsAPI-pinnacle", "1X2", T0 + timedelta(seconds=s))
            for s in (0, 3, 7)]
    got = h1.assemble_instants(keys)
    assert len(set(got.values())) == 1, (
        "three legs of one market landed in three instants — the overround "
        "can never be computed and the band drops the whole series")


def test_two_genuine_observations_stay_APART():
    """The tolerance must not merge real points. 120 minutes is not 120s."""
    keys = [(1, "b", "1X2", T0), (1, "b", "1X2", T0 + timedelta(minutes=120))]
    assert len(set(h1.assemble_instants(keys).values())) == 2


def test_the_tolerance_CANNOT_reach_the_separation_floor():
    """The relationship is pinned, not just the number.

    Chaining means a cluster can extend past one tolerance width, so the
    guarantee has to be argued on the gap between the two constants rather
    than on the constant alone.
    """
    assert h1.MARKET_ASSEMBLY_SECONDS * 2 < h1.MIN_GAP_MINUTES * 60, (
        "market assembly is close enough to the separation floor that "
        "assembling one market could swallow two genuine observations")


def test_different_books_are_never_assembled_together():
    keys = [(1, "book-a", "1X2", T0), (1, "book-b", "1X2", T0)]
    got = h1.assemble_instants(keys)
    assert got[(1, "book-a", "1X2", T0)] == T0
    assert got[(1, "book-b", "1X2", T0)] == T0
    assert len(got) == 2


def test_a_chain_of_close_writes_is_one_instant():
    """40s apart across three writes is one market, not three."""
    keys = [(1, "b", "1X2", T0 + timedelta(seconds=s)) for s in (0, 40, 80)]
    assert len(set(h1.assemble_instants(keys).values())) == 1
