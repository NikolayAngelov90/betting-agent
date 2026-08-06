"""End-to-end check that column-projected reads changed nothing.

The egress work replaced ``session.query(Match)`` / ``query(Odds)`` /
``query(Team)`` / ``query(Player)`` with explicit column lists across the
feature pipeline. Those queries are all built with the ORM, so a mistake shows
up as either invalid SQL or a silently missing attribute — neither of which the
existing MagicMock-based preload tests can see.

This test runs the real pipeline against a real (SQLite) database:

  · ``create_features`` with the preload cache populated,
  · ``create_features`` with no cache at all (the per-fixture query fallback),

and requires the two to agree — which is also the invariant preload_batch was
always supposed to hold.
"""

import asyncio
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from src.data.database import DatabaseManager
from src.data.models import Team, Match, Odds, Player, Injury


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    mgr = DatabaseManager(config=SimpleNamespace(
        database={"sqlite_path": str(tmp_path / "projection_test.db")}
    ))
    assert not mgr.is_postgres, "test DB must be SQLite, not production Postgres"
    mgr.create_tables()
    return mgr


def _seed(db):
    """A small league with full stat coverage, plus one upcoming fixture."""
    base = datetime.now() - timedelta(days=200)
    with db.get_session() as s:
        teams = [Team(name=f"Club {i}", league="test/league", country="Test")
                 for i in range(6)]
        s.add_all(teams)
        s.flush()
        ids = [t.id for t in teams]

        for i in range(80):
            # Round-robin that guarantees Club 0 v Club 1 meetings, so the H2H
            # features under test are non-empty.
            home = ids[i % 6]
            away = ids[(i + 1) % 6]
            s.add(Match(
                home_team_id=home, away_team_id=away,
                match_date=base + timedelta(days=i * 2),
                league="test/league", season="2025",
                home_goals=i % 4, away_goals=(i + 1) % 3,
                home_xg=0.6 + (i % 5) * 0.25, away_xg=0.5 + (i % 4) * 0.3,
                home_shots=8 + i % 7, away_shots=6 + i % 5,
                home_shots_on_target=3 + i % 4, away_shots_on_target=2 + i % 3,
                home_possession=45.0 + (i % 20), away_possession=55.0 - (i % 20),
                home_corners=3 + i % 6, away_corners=2 + i % 5,
                home_fouls=9 + i % 5, away_fouls=11 - i % 5,
                home_yellow_cards=i % 4, away_yellow_cards=(i + 1) % 4,
                home_red_cards=0, away_red_cards=1 if i % 17 == 0 else 0,
                home_dangerous_attacks=40 + i % 15, away_dangerous_attacks=35 + i % 12,
                home_saves=2 + i % 5, away_saves=3 + i % 4,
                home_offsides=i % 4, away_offsides=(i + 2) % 4,
                home_free_kicks=10 + i % 6, away_free_kicks=12 - i % 6,
                regulation_home_goals=i % 4, regulation_away_goals=(i + 1) % 3,
                referee="Ref A" if i % 2 else "Ref B",
                venue="Ground", round="Regular Season - 5",
                is_fixture=False,
            ))

        fixture = Match(
            home_team_id=ids[0], away_team_id=ids[1],
            match_date=datetime.now() + timedelta(days=1),
            league="test/league", season="2025", referee="Ref A",
            venue="Ground", round="Regular Season - 41", is_fixture=True,
        )
        s.add(fixture)
        s.flush()
        fixture_id = fixture.id

        for book in ("Bet365", "Pinnacle"):
            s.add_all([
                Odds(match_id=fixture_id, bookmaker=book, market_type="1X2",
                     selection="Home", odds_value=2.10, opening_odds=2.25),
                Odds(match_id=fixture_id, bookmaker=book, market_type="1X2",
                     selection="Draw", odds_value=3.40, opening_odds=3.30),
                Odds(match_id=fixture_id, bookmaker=book, market_type="1X2",
                     selection="Away", odds_value=3.60, opening_odds=3.50),
                Odds(match_id=fixture_id, bookmaker=book, market_type="over_under",
                     selection="Over 2.5", odds_value=1.95, opening_odds=2.00),
                Odds(match_id=fixture_id, bookmaker=book, market_type="over_under",
                     selection="Under 2.5", odds_value=1.90, opening_odds=1.85),
                Odds(match_id=fixture_id, bookmaker=book, market_type="btts",
                     selection="Yes", odds_value=1.80, opening_odds=1.75),
                Odds(match_id=fixture_id, bookmaker=book, market_type="btts",
                     selection="No", odds_value=2.00, opening_odds=2.05),
            ])

        # Squad + injuries for the injury-feature join.
        for i, pos in enumerate(
            ["Goalkeeper", "Defender", "Defender", "Midfielder", "Attacker"]
        ):
            p = Player(name=f"P{i}", team_id=ids[0], position=pos,
                       is_key_player=(i == 4))
            s.add(p)
            s.flush()
            if i in (1, 4):
                s.add(Injury(player_id=p.id, team_id=ids[0],
                             injury_type="knock", status="out"))

    return fixture_id, ids


def _engine(db):
    from src.features.feature_engineer import FeatureEngineer
    fe = FeatureEngineer()
    fe.db = db
    fe.team_features.db = db
    fe.h2h_features.db = db
    fe.injury_features.db = db
    return fe


def _features(db, fixture_id, use_preload, as_of_date=None, **preload_kwargs):
    fe = _engine(db)
    if use_preload:
        fe.preload_batch([fixture_id], **preload_kwargs)
        assert fe._preload_cache is not None, "preload_batch swallowed an error"
    return asyncio.run(
        fe.create_features(fixture_id, as_of_date=as_of_date, for_training=True)
    )


def _assert_identical(preloaded, live, label=""):
    assert preloaded, f"{label}: create_features returned nothing"
    assert set(preloaded) == set(live), (
        f"{label}: feature keys diverged: {set(preloaded) ^ set(live)}")
    for key in preloaded:
        a, b = preloaded[key], live[key]
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            assert a == pytest.approx(b, rel=1e-9, abs=1e-12), \
                f"{label}: {key}: preload={a} live={b}"
        else:
            assert a == b, f"{label}: {key}: preload={a!r} live={b!r}"


# ── Adversarial fixture ──────────────────────────────────────────────────────
# The friendly seed above cannot exercise the failure modes the cache has to
# survive. This one is built specifically to break it:
#   · history spans ~3 years, so the 365-day preload window truncates
#   · the two fixture teams play 100+ matches, so cap_per_team (60) truncates
#   · 12 clubs in the league, so standings have real spread and the fixture's
#     teams are NOT the top two (the old bug always put them 1st and 2nd)
#   · the referee mostly officiates matches involving OTHER clubs, which the old
#     two-team scan could never see
#   · half the head-to-head meetings are older than the 365-day window
def _seed_adversarial(db):
    # Span ~960 days ending ~5 weeks ago: roughly a third of the history falls
    # inside the live 365-day preload window and the rest outside it, so the
    # date cutoff genuinely truncates while the referee's 365-day window still
    # has matches to find.
    base = datetime.now() - timedelta(days=960)
    n_teams = 12
    with db.get_session() as s:
        teams = [Team(name=f"FC {i:02d}", league="hostile/division", country="Test")
                 for i in range(n_teams)]
        other = [Team(name=f"Cup {i}", league="hostile/cup", country="Test")
                 for i in range(4)]
        s.add_all(teams + other)
        s.flush()
        ids = [t.id for t in teams]
        cup_ids = [t.id for t in other]

        day = 0
        # 7 seasons of a 12-team round robin → 77 league matches per team, so
        # the live 365-day window keeps only ~30 of them.
        for season in range(7):
            for rnd in range(n_teams - 1):
                for k in range(n_teams // 2):
                    home = ids[(rnd + k) % n_teams]
                    away = ids[(rnd - k - 1) % n_teams]
                    if home == away:
                        continue
                    day += 2
                    # Referee assignment deliberately favours matches that do
                    # NOT involve teams 0 and 1 (the fixture's teams).
                    involves_fixture_teams = home in ids[:2] or away in ids[:2]
                    ref = "Whistle B" if involves_fixture_teams else "Whistle A"
                    s.add(Match(
                        home_team_id=home, away_team_id=away,
                        match_date=base + timedelta(days=day),
                        league="hostile/division", season=f"20{18 + season}",
                        home_goals=(home + rnd) % 5, away_goals=(away + k) % 4,
                        home_xg=0.4 + ((home + rnd) % 6) * 0.3,
                        away_xg=0.3 + ((away + k) % 5) * 0.35,
                        home_shots=7 + (rnd % 9), away_shots=5 + (k % 8),
                        home_shots_on_target=2 + (rnd % 5),
                        away_shots_on_target=1 + (k % 4),
                        home_possession=40.0 + (rnd % 25),
                        away_possession=60.0 - (rnd % 25),
                        home_corners=2 + (rnd % 8), away_corners=1 + (k % 7),
                        home_fouls=8 + (rnd % 7), away_fouls=10 - (k % 6),
                        home_yellow_cards=(rnd + k) % 5,
                        away_yellow_cards=(rnd + 1) % 4,
                        home_red_cards=1 if (day % 53 == 0) else 0,
                        away_red_cards=0,
                        home_dangerous_attacks=30 + (rnd % 20),
                        away_dangerous_attacks=25 + (k % 18),
                        home_saves=1 + (rnd % 6), away_saves=2 + (k % 5),
                        home_offsides=rnd % 5, away_offsides=k % 4,
                        home_free_kicks=9 + (rnd % 8), away_free_kicks=11 - (k % 7),
                        regulation_home_goals=(home + rnd) % 5,
                        regulation_away_goals=(away + k) % 4,
                        referee=ref, venue="Stadium", round=f"Round {rnd}",
                        is_fixture=False,
                    ))

        # Cup matches for teams 0 and 1 against clubs outside their division —
        # these must NOT count toward league standings but MUST count toward
        # overall form.
        for j in range(12):
            day += 5
            s.add(Match(
                home_team_id=ids[j % 2], away_team_id=cup_ids[j % 4],
                match_date=base + timedelta(days=day),
                league="hostile/cup", season="2024",
                home_goals=j % 4, away_goals=(j + 1) % 3,
                home_xg=1.1, away_xg=0.7,
                referee="Whistle A", venue="Stadium",
                is_fixture=False,
            ))

        fixture = Match(
            home_team_id=ids[0], away_team_id=ids[1],
            match_date=datetime.now() + timedelta(days=1),
            league="hostile/division", season="2025",
            referee="Whistle A", venue="Stadium", round="Round 40",
            is_fixture=True,
        )
        s.add(fixture)
        s.flush()
        fixture_id = fixture.id
        s.add_all([
            Odds(match_id=fixture_id, bookmaker="Bet365", market_type="1X2",
                 selection=sel, odds_value=v, opening_odds=v + 0.1)
            for sel, v in (("Home", 2.2), ("Draw", 3.3), ("Away", 3.4))
        ])
    return fixture_id, ids


class TestFeatureProjectionEquivalence:
    def test_preloaded_and_live_features_agree(self, db):
        """EVERY feature must match. No exclusions.

        Two families used to be exempted here — league standings and referee
        stats — because preload_batch only cached the fixture's own two teams
        while those features need league-wide and referee-wide context. That was
        a real correctness bug, not a quirk to tolerate; the cache now carries
        dedicated scopes for both and falls back to the live query whenever it
        cannot prove its slice is exact. The exclusion list is deliberately gone.
        """
        fixture_id, _ = _seed(db)

        preloaded = _features(db, fixture_id, use_preload=True)
        live = _features(db, fixture_id, use_preload=False)

        assert preloaded, "create_features returned nothing"
        assert set(preloaded) == set(live), (
            f"feature keys diverged: "
            f"{set(preloaded) ^ set(live)}"
        )
        _assert_identical(preloaded, live, "friendly seed")
        assert len(preloaded) > 200, f"only {len(preloaded)} features generated"

    def test_features_are_actually_populated(self, db):
        """Guard against 'both paths agree because both return zeros'."""
        fixture_id, _ = _seed(db)
        feats = _features(db, fixture_id, use_preload=False)

        numeric = [v for v in feats.values() if isinstance(v, (int, float))]
        assert len(numeric) > 50, f"only {len(numeric)} numeric features"
        assert sum(1 for v in numeric if v != 0) > 25, "nearly everything is zero"

        # Spot-check one feature from each projected query.
        assert feats["home_overall_goals_scored_per_match"] > 0      # team_features
        assert feats["h2h_total_meetings"] > 0                       # h2h_features
        assert feats["home_xg_avg"] > 0                              # _get_xg_features
        assert feats["referee_cards_per_match_avg"] > 0              # referee features
        assert feats["league_avg_goals"] > 0                         # league features
        assert feats["home_implied_prob"] > 0                        # bookmaker odds
        assert feats["max_abs_movement"] > 0                         # odds movement

    def test_injury_features_survive_the_join_projection(self, db):
        """The Injury→Player join now selects only player columns."""
        from src.features.injury_features import InjuryFeatures
        _, ids = _seed(db)

        inj = InjuryFeatures()
        inj.db = db
        feats = inj.get_injury_features(ids[0])

        assert feats["total_injured"] == 2
        assert feats["key_players_injured"] == 1   # only P4 is flagged key
        assert feats["defenders_out"] == 1
        assert feats["attackers_out"] == 1
        assert feats["goalkeeper_available"] is True
        assert feats["defensive_stability_score"] < 1.0

    def test_standings_projection_still_ranks_teams(self, db):
        """_get_league_standings switched Team rows for an (id, name) projection."""
        from src.features.team_features import TeamFeatures
        _, ids = _seed(db)

        tf = TeamFeatures()
        tf.db = db
        standings = tf._get_league_standings("test/league")

        assert len(standings) == 6
        assert all(s["team_name"].startswith("Club ") for s in standings)
        points = [s["points"] for s in standings]
        assert points == sorted(points, reverse=True)
        assert sum(points) > 0


class TestAdversarialEquivalence:
    """The cases that actually broke: truncation, wide scope, training cutoffs."""

    def test_every_feature_matches_under_truncation(self, db):
        """cap_per_team=60 and the 365-day window both bite here.

        Teams 0 and 1 have ~140 matches across ~3 years, so the preload slice is
        a strict prefix of their history. Every feature must still be identical —
        either because the slice provably contains the requested rows, or because
        the accessor declined and the live query ran.
        """
        fixture_id, _ = _seed_adversarial(db)
        _assert_identical(
            _features(db, fixture_id, use_preload=True),
            _features(db, fixture_id, use_preload=False),
            "adversarial / live window",
        )

    def test_every_feature_matches_with_training_settings(self, db):
        """Training preloads with cap 200 and no date cutoff."""
        fixture_id, _ = _seed_adversarial(db)
        _assert_identical(
            _features(db, fixture_id, use_preload=True,
                      cap_per_team=200, cutoff_days=0),
            _features(db, fixture_id, use_preload=False),
            "adversarial / training window",
        )

    def test_every_feature_matches_with_as_of_date(self, db):
        """as_of_date shrinks each cached slice after the fact — the case that
        makes a naive 'I have rows, good enough' cache wrong."""
        fixture_id, _ = _seed_adversarial(db)
        cutoff = datetime.now() - timedelta(days=400)
        _assert_identical(
            _features(db, fixture_id, use_preload=True, as_of_date=cutoff,
                      cap_per_team=200, cutoff_days=0),
            _features(db, fixture_id, use_preload=False, as_of_date=cutoff),
            "adversarial / as_of_date",
        )

    def test_every_feature_matches_when_caps_force_fallback(self, db):
        """Absurdly tight caps make almost every cached slice unprovable.

        This is the safety net: when the cache can answer almost nothing, the
        pipeline must still produce exactly the live answer.
        """
        fixture_id, _ = _seed_adversarial(db)
        _assert_identical(
            _features(db, fixture_id, use_preload=True,
                      cap_per_team=3, league_cap=5, referee_cap=2),
            _features(db, fixture_id, use_preload=False),
            "adversarial / starved cache",
        )

    def test_standings_are_not_collapsed_to_the_fixture_teams(self, db):
        """The original bug, pinned directly.

        preload_batch cached only the two teams playing, so every other club in
        the division scored 0 points and the fixture's teams always came out 1st
        and 2nd. With 12 clubs and three seasons of results that must not happen.
        """
        fixture_id, ids = _seed_adversarial(db)

        fe = _engine(db)
        fe.preload_batch([fixture_id])
        cached = fe.team_features._get_league_standings(
            "hostile/division", preload_cache=fe._preload_cache)

        live_fe = _engine(db)
        live = live_fe.team_features._get_league_standings("hostile/division")

        assert len(cached) == 12 == len(live)
        assert [s["team_id"] for s in cached] == [s["team_id"] for s in live]
        assert [s["points"] for s in cached] == [s["points"] for s in live]

        # Every club has real results — none of the zeroed placeholders.
        assert all(s["matches_played"] > 20 for s in cached)
        # And the fixture's own teams are not automatically on top.
        top_two = {cached[0]["team_id"], cached[1]["team_id"]}
        assert top_two != {ids[0], ids[1]}

    def test_referee_stats_span_the_whole_division(self, db):
        """'Whistle A' mostly officiates matches NOT involving teams 0 and 1.

        The old two-team scan could only ever see the handful it did, so the
        averages were computed from the wrong sample — and it ignored the
        365-day window the live query applies.
        """
        fixture_id, _ = _seed_adversarial(db)

        fe = _engine(db)
        fe.preload_batch([fixture_id])
        cached = fe._get_referee_features("Whistle A")

        live_fe = _engine(db)
        live = live_fe._get_referee_features("Whistle A")

        assert cached == live
        assert live["referee_matches"] > 0

    def test_h2h_survives_the_history_window(self, db):
        """Meetings older than the 365-day team-history window must still count."""
        fixture_id, ids = _seed_adversarial(db)

        fe = _engine(db)
        fe.preload_batch([fixture_id])
        cached = fe.h2h_features.get_h2h_features(
            ids[0], ids[1], preload_cache=fe._preload_cache)

        live = _engine(db).h2h_features.get_h2h_features(ids[0], ids[1])

        assert cached == live
        assert live["h2h_total_meetings"] >= 3


class TestCompletenessRule:
    """Unit tests for the rule the whole design rests on."""

    def _cache(self, rows, complete):
        return {"team_history": {1: rows}, "team_complete": {1} if complete else set()}

    def test_unknown_team_is_a_miss_not_an_empty_answer(self):
        from src.features import preload_cache as pc
        assert pc.team_rows(self._cache([], False), 99, limit=5) is None

    def test_enough_matching_rows_is_exact_even_when_truncated(self):
        from src.features import preload_cache as pc
        rows = [{"n": i} for i in range(10)]
        got = pc.team_rows(self._cache(rows, False), 1, limit=5)
        assert got == rows[:5]

    def test_too_few_rows_in_a_truncated_slice_is_a_miss(self):
        from src.features import preload_cache as pc
        rows = [{"n": i} for i in range(3)]
        assert pc.team_rows(self._cache(rows, False), 1, limit=5) is None

    def test_too_few_rows_in_a_complete_slice_is_the_answer(self):
        from src.features import preload_cache as pc
        rows = [{"n": i} for i in range(3)]
        assert pc.team_rows(self._cache(rows, True), 1, limit=5) == rows

    def test_predicate_is_applied_before_the_count_check(self):
        from src.features import preload_cache as pc
        rows = [{"keep": i % 2 == 0} for i in range(10)]  # 5 keepers
        assert pc.team_rows(self._cache(rows, False), 1, limit=5,
                            predicate=lambda r: r["keep"]) is not None
        assert pc.team_rows(self._cache(rows, False), 1, limit=6,
                            predicate=lambda r: r["keep"]) is None

    def test_empty_complete_slice_answers_empty(self):
        from src.features import preload_cache as pc
        assert pc.team_rows(self._cache([], True), 1, limit=5) == []

    def test_no_cache_is_always_a_miss(self):
        from src.features import preload_cache as pc
        assert pc.team_rows(None, 1, limit=5) is None
        assert pc.h2h_rows(None, 1, 2, limit=5) is None
        assert pc.referee_rows(None, "x", limit=5) is None


class TestAnalyzeFixtureOddsSource:
    """analyze_fixture must read the odds preload_batch already fetched."""

    def test_cached_odds_equal_the_database_rows(self, db):
        """Whichever path runs, analyze_fixture must see the same odds."""
        fixture_id, _ = _seed(db)

        fe = _engine(db)
        fe.preload_batch([fixture_id])
        cached = sorted(
            (o["market_type"], o["selection"], o["odds_value"],
             o["bookmaker"], o["opening_odds"])
            for o in fe._preload_cache["odds"][fixture_id]
        )

        from src.data.models import Odds
        with db.get_session() as s:
            rows = s.query(
                Odds.market_type, Odds.selection, Odds.odds_value,
                Odds.bookmaker, Odds.opening_odds,
            ).filter_by(match_id=fixture_id).all()
        live = sorted(tuple(r) for r in rows)

        assert cached == live
        assert len(cached) == 14   # 2 bookmakers x 7 selections

    def test_uncached_fixture_still_queries(self, db):
        """A fixture outside the preloaded batch must fall back, not go blank."""
        fixture_id, _ = _seed(db)
        fe = _engine(db)
        fe.preload_batch([fixture_id])
        assert 999999 not in fe._preload_cache["odds"]
