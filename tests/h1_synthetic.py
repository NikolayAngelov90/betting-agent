"""A synthetic H1 collection, built once and shared.

NOT a test file — a fixture builder, imported by
`test_h1_reservation_is_an_allocation.py` (which needs real `odds_snapshots`
rows to reach `RESERVATION_EXHAUSTED` through the WIRED path) and by
`test_h1_analysis_path.py` (which runs the whole analysis on it).

IT MUST BE SHARED RATHER THAN DUPLICATED. The two files ask different
questions of the same shape, and two hand-built fixtures drift — which is how
`collection_stop` came to be tested only through arguments the harness
supplied, never through the value the runner derives.

WHAT "REALISTIC" MEANS HERE, and every choice is a stated one:

* **three observations per series, 45 and 60 minutes apart.** `MIN_GAP_MINUTES`
  is 30 and `MAX_INTERVAL_RATIO` is 2.0, so 45/60 clears both with margin —
  a triple at exactly 30/60 sits on the ratio boundary and would make a
  failure ambiguous between the gap rule and the ratio rule.
* **`rho = 0.42` clustering on FIXTURE.** Each fixture carries several
  (book, market, selection) series, so a count at key grain overstates `n`.
  That is the hundredfold error this fixture exists to catch, so it is built
  in rather than avoided.
* **observations strictly before kickoff.** `load_observations` filters
  `observed_at < m.match_date`; a fixture whose observations straddle kickoff
  silently loses rows and the test would read the loss as a predicate result.
* **`training_exclusion_reason IS NULL`** on every match, for the same reason.
* **spread across DATES.** One date makes date the cluster and the sizing
  stops applying, so the builder takes a day count and uses it.
"""

from datetime import datetime, timedelta

#: Gaps between the three observations of a series, in minutes. Clears
#: MIN_GAP_MINUTES=30 and the 2.0 ratio with margin.
GAPS = (45, 60)

#: Books, markets and selections per fixture. Chosen so each fixture carries
#: SEVERAL series — the key-vs-fixture distinction has to be exercisable.
BOOKS = ("TheOddsAPI-pinnacle", "TheOddsAPI-betfair")
MARKET_LEGS = {"1X2": ("Home", "Draw", "Away"),
               "over_under": ("Over 2.5", "Under 2.5")}

#: A plausible complete-market price set. `load_observations` scores a market
#: only when every leg is present at that instant and the overround lands in
#: (1.005, 1.25) — two of three legs is a different number, not a small error.
PRICES = {"1X2": {"Home": 2.10, "Draw": 3.40, "Away": 3.60},
          "over_under": {"Over 2.5": 1.95, "Under 2.5": 1.95}}


def build(session, *, n_fixtures, first_day, days=1, base_hour=12,
          series_per_fixture=None, league="england/premier-league"):
    """Write `n_fixtures` fixtures with qualifying trajectories.

    Returns `{"match_ids": [...], "dates": [...], "series": int,
    "observations": int}` — the counts at BOTH grains, so a caller can assert
    the one it means.
    """
    from src.data.models import Match, OddsSnapshot, Team

    # TEAMS FIRST: `matches.home_team_id` / `away_team_id` are real foreign
    # keys, and SQLite enforces them here. Building the match rows without
    # them fails with a bare `FOREIGN KEY constraint failed` that says
    # nothing about which key.
    team_ids = []
    for i in range(2 * n_fixtures):
        t = Team(name=f"Synthetic FC {i}", league=league)
        session.add(t)
        session.flush()
        team_ids.append(t.id)

    match_ids, dates = [], []
    n_series = n_obs = 0
    for i in range(n_fixtures):
        day_offset = i % max(1, days)
        day = first_day + timedelta(days=day_offset)
        kickoff = datetime(day.year, day.month, day.day, base_hour, 0)
        m = Match(home_team_id=team_ids[2 * i], away_team_id=team_ids[2 * i + 1],
                  match_date=kickoff, league=league, season=2026,
                  is_fixture=True, training_exclusion_reason=None)
        session.add(m)
        session.flush()
        match_ids.append(m.id)
        if day not in dates:
            dates.append(day)

        # the three instants, all strictly BEFORE kickoff
        t0 = kickoff - timedelta(minutes=GAPS[0] + GAPS[1] + 30)
        instants = [t0, t0 + timedelta(minutes=GAPS[0]),
                    t0 + timedelta(minutes=GAPS[0] + GAPS[1])]
        assert instants[-1] < kickoff, "the fixture straddles kickoff"

        wanted = series_per_fixture
        made = 0
        for book in BOOKS:
            for market, legs in MARKET_LEGS.items():
                if wanted is not None and made >= wanted:
                    break
                for ts in instants:
                    for leg in legs:
                        session.add(OddsSnapshot(
                            match_id=m.id, bookmaker=book,
                            market_type=market, selection=leg,
                            odds_value=PRICES[market][leg], observed_at=ts))
                        n_obs += 1
                n_series += len(legs)
                made += 1
    session.commit()
    return {"match_ids": match_ids, "dates": dates,
            "series": n_series, "observations": n_obs}
