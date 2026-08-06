"""Sharded analysis must be indistinguishable from the single-process run.

The pipeline is one runner doing everything sequentially, which does not scale.
Analysis is per-fixture and parallelises cleanly — but the phase *after* it does
not: ranking, the per-match cap, correlation filtering and the daily exposure
cap (`betting.max_total_kelly_pct`) are properties of the whole day's slate.

That is the trap these tests exist for. Four shards each applying a 40% exposure
cap would stake 160% of bankroll. So shards emit candidates and a single collect
step runs the portfolio phase once over the union.
"""

from dataclasses import replace
from datetime import date, datetime, timedelta
from types import SimpleNamespace

import pytest

from src.betting import candidate_io
from src.betting.value_calculator import BetRecommendation


def _rec(match_id, selection="Home Win", ev=0.10, kelly=5.0, **kw):
    base = dict(
        match=f"Team{match_id}A vs Team{match_id}B", match_id=match_id,
        market="1X2", selection=selection, odds=2.0,
        predicted_probability=0.55, expected_value=ev, confidence=0.6,
        kelly_stake_percentage=kelly, recommended_stake=kelly * 10,
        reasoning="test", risk_level="medium", league="test/league",
        model_agreement="unanimous", match_date=datetime(2026, 8, 10, 18, 0),
    )
    base.update(kw)
    return BetRecommendation(**base)


class TestCandidateRoundTrip:
    """The wire format between a shard job and the collect job."""

    def test_round_trip_preserves_every_field(self, tmp_path):
        original = _rec(7, ev=0.1234, kelly=3.5, contrarian_value=1.42,
                        opening_odds=2.15, models_for="Poisson, Elo")
        candidate_io.write_candidates(tmp_path / "candidates-001.json", [original],
                                      shard=1, shards=4)
        [restored] = candidate_io.read_candidates([tmp_path / "candidates-001.json"])
        assert restored == original

    def test_discovery_is_deterministically_ordered(self, tmp_path):
        """Ties in the portfolio sort must not depend on directory order."""
        for shard in (3, 1, 2):
            candidate_io.write_candidates(
                tmp_path / f"candidates-{shard:03d}.json", [_rec(shard)],
                shard=shard, shards=3)
        found = candidate_io.discover(tmp_path)
        assert [p.name for p in found] == [
            "candidates-001.json", "candidates-002.json", "candidates-003.json"]

    def test_a_missing_shard_file_costs_only_that_shard(self, tmp_path):
        """Losing one shard must not lose the day."""
        candidate_io.write_candidates(tmp_path / "candidates-001.json",
                                      [_rec(1), _rec(2)], shard=1, shards=2)
        loaded = candidate_io.read_candidates(
            [tmp_path / "candidates-001.json", tmp_path / "candidates-002.json"])
        assert len(loaded) == 2

    def test_schema_skew_is_refused_not_silently_misread(self, tmp_path):
        import json
        p = tmp_path / "candidates-001.json"
        candidate_io.write_candidates(p, [_rec(1)], shard=1, shards=1)
        payload = json.loads(p.read_text())
        payload["schema_version"] = 99
        p.write_text(json.dumps(payload))
        assert candidate_io.read_candidates([p]) == []

    def test_unknown_fields_are_ignored(self, tmp_path):
        """A collector on a newer revision than a shard must not abort."""
        import json
        p = tmp_path / "candidates-001.json"
        candidate_io.write_candidates(p, [_rec(1)], shard=1, shards=1)
        payload = json.loads(p.read_text())
        payload["recommendations"][0]["a_field_from_the_future"] = 123
        p.write_text(json.dumps(payload))
        [restored] = candidate_io.read_candidates([p])
        assert restored.match_id == 1


class TestPartitionIsDisjointAndComplete:
    """match_id % shards: every fixture analysed exactly once."""

    @pytest.mark.parametrize("shards", [2, 3, 4, 8])
    def test_every_fixture_lands_in_exactly_one_shard(self, shards):
        fixture_ids = list(range(1, 501))
        seen = []
        for shard in range(1, shards + 1):
            seen += [f for f in fixture_ids if f % shards == (shard - 1)]
        assert sorted(seen) == fixture_ids, "partition must be a exact cover"
        assert len(seen) == len(set(seen)), "no fixture may be analysed twice"

    def test_a_match_never_straddles_two_shards(self):
        """Why match_id is the key.

        The per-match pick cap and the correlation filter both reason within a
        single match. Keying on match_id keeps those correct inside a shard.
        """
        shards = 4
        owners = {}
        for shard in range(1, shards + 1):
            for fid in range(1, 200):
                if fid % shards == (shard - 1):
                    owners.setdefault(fid, []).append(shard)
        assert all(len(v) == 1 for v in owners.values())


class _Agent:
    """Minimal stand-in exposing the real finalize_picks."""

    def __init__(self, db, exposure_cap=40.0, max_per_match=2):
        from src.agent.betting_agent import FootballBettingAgent
        self._real = FootballBettingAgent.__new__(FootballBettingAgent)
        self._real.db = db
        self._real.config = SimpleNamespace(get=lambda k, d=None: {
            "betting.max_total_kelly_pct": exposure_cap,
        }.get(k, d))
        self._real.value_calculator = SimpleNamespace(min_ev=0.02)
        self._real._saved = []

        def _save(recs, target):
            self._real._saved = list(recs)
            return list(recs)
        self._real._save_picks = _save
        self.max_per_match = max_per_match

    def finalize(self, recs, target=None):
        return self._real.finalize_picks(
            recs, target or date(2026, 8, 10),
            max_picks_per_match=self.max_per_match)


@pytest.fixture
def db(tmp_path, monkeypatch):
    from src.data.database import DatabaseManager
    monkeypatch.delenv("DATABASE_URL", raising=False)
    mgr = DatabaseManager(config=SimpleNamespace(
        database={"sqlite_path": str(tmp_path / "shard.db")}))
    assert not mgr.is_postgres
    mgr.create_tables()
    return mgr


class TestPortfolioPhaseIsGlobal:
    """The reason sharding cannot simply run the whole pipeline N times."""

    def test_exposure_cap_is_not_multiplied_by_shard_count(self, db, monkeypatch):
        """The headline failure mode.

        Twelve picks at 5% Kelly = 60% exposure against a 40% cap. Split across
        four shards, each shard sees 15% and nothing is trimmed — so a naive
        sharding would stake 60%. Collecting first must trim to 40%.
        """
        monkeypatch.chdir(db.config.database["sqlite_path"].rsplit("\\", 1)[0]
                          if "\\" in db.config.database["sqlite_path"]
                          else "/tmp")
        recs = [_rec(i, ev=0.10 + i * 0.001, kelly=5.0) for i in range(1, 13)]

        # What each shard would have concluded on its own.
        per_shard_total = 0.0
        for shard in range(1, 5):
            mine = [r for r in recs if r.match_id % 4 == (shard - 1)]
            kept, _, _ = _Agent(db).finalize(list(mine))
            per_shard_total += sum(r.kelly_stake_percentage for r in kept)
        assert per_shard_total > 40.0, (
            "test is not exercising the failure: shards did not exceed the cap")

        # What the collect step concludes over the union.
        kept, _, dropped = _Agent(db).finalize(list(recs))
        collected_total = sum(r.kelly_stake_percentage for r in kept)
        assert collected_total <= 40.0, (
            f"collected exposure {collected_total} exceeds the 40% cap")
        assert dropped, "the cap should have trimmed the lowest-ranked picks"

    def test_ranking_is_global_not_per_shard(self, db, monkeypatch):
        """A weak pick in a sparse shard must not outrank a strong one elsewhere."""
        monkeypatch.chdir("/tmp")
        strong = [_rec(i, ev=0.30, kelly=12.0) for i in (4, 8, 12)]   # all shard 1
        weak = [_rec(i, ev=0.03, kelly=12.0) for i in (1, 2, 3)]      # other shards
        kept, _, _ = _Agent(db).finalize(strong + weak)
        kept_ids = [r.match_id for r in kept]
        # With a 40% cap and 12% stakes only ~3 survive; they must be the strong ones.
        assert set(kept_ids) <= {4, 8, 12}, f"weak picks survived: {kept_ids}"


class TestShardThenCollectEqualsUnsharded:
    """The property the whole design rests on."""

    def _finalize(self, db, recs):
        kept, _, _ = _Agent(db).finalize(list(recs))
        return [(r.match_id, r.selection, round(r.expected_value, 6)) for r in kept]

    def test_identical_result_for_any_shard_count(self, db, tmp_path, monkeypatch):
        monkeypatch.chdir("/tmp")
        recs = [
            _rec(i, ev=0.05 + (i % 7) * 0.01, kelly=1.5,
                 selection="Home Win" if i % 2 else "Over 2.5 Goals")
            for i in range(1, 41)
        ]

        reference = self._finalize(db, recs)
        assert reference, "reference run produced no picks"

        for shards in (2, 3, 4, 8):
            out_dir = tmp_path / f"s{shards}"
            for shard in range(1, shards + 1):
                mine = [r for r in recs if r.match_id % shards == (shard - 1)]
                candidate_io.write_candidates(
                    out_dir / f"candidates-{shard:03d}.json", mine,
                    shard=shard, shards=shards)
            collected = candidate_io.read_candidates(candidate_io.discover(out_dir))
            assert len(collected) == len(recs), "sharding lost or duplicated candidates"
            assert self._finalize(db, collected) == reference, (
                f"{shards}-way sharding changed the day's picks")


class TestRankingIsReproducible:
    """The tie-break bug sharding exposed — it predates sharding.

    The ranking score (EV x confidence x agreement x contrarian) is not a total
    order: ties are common. Python's sort is stable, so before this fix the
    survivor of a tie was whichever the analysis happened to emit first. Re-run
    --picks with fixtures in a different order and you got different bets.
    """

    def test_shuffled_input_yields_identical_picks(self, db, monkeypatch):
        import random
        monkeypatch.chdir("/tmp")
        # All tied on the score, distinguished only by identity.
        recs = [_rec(i, ev=0.10, kelly=2.0) for i in range(1, 31)]

        reference = [r.match_id for r in _Agent(db).finalize(list(recs))[0]]
        assert reference, "no picks produced"

        for seed in range(5):
            shuffled = list(recs)
            random.Random(seed).shuffle(shuffled)
            got = [r.match_id for r in _Agent(db).finalize(shuffled)[0]]
            assert got == reference, f"seed {seed} changed the picks"

    def test_ties_within_a_match_resolve_deterministically(self, db, monkeypatch):
        monkeypatch.chdir("/tmp")
        a = _rec(5, selection="Home Win", ev=0.10, confidence=0.6, kelly=1.0)
        b = _rec(5, selection="Over 2.5 Goals", ev=0.10, confidence=0.6, kelly=1.0)
        first = [r.selection for r in _Agent(db, max_per_match=1).finalize([a, b])[0]]
        second = [r.selection for r in _Agent(db, max_per_match=1).finalize([b, a])[0]]
        assert first == second, "per-match cap kept a different pick on reorder"


class TestCliShardParsing:
    """`--shard 1/1` must not require a collect step."""

    def _run(self, argv, agent):
        import asyncio
        from unittest.mock import patch
        from src.agent.betting_agent import main
        with patch("src.agent.betting_agent.FootballBettingAgent", return_value=agent), \
             patch("src.agent.betting_agent._configure_cli_runtime"), \
             patch("sys.argv", argv):
            asyncio.run(main())

    def _agent(self):
        from unittest.mock import AsyncMock, MagicMock
        agent = MagicMock()
        agent.shutdown = AsyncMock()
        agent.get_daily_picks = AsyncMock(return_value=([], [], []))
        return agent

    def test_one_way_shard_takes_the_normal_path(self):
        agent = self._agent()
        self._run(["betting_agent", "--picks", "--shard", "1/1"], agent)
        _, kwargs = agent.get_daily_picks.call_args
        assert kwargs.get("shards") in (None, 1), (
            "a 1-way split must not enter the sharded path")

    def test_real_shard_is_passed_through(self):
        agent = self._agent()
        self._run(["betting_agent", "--picks", "--shard", "2/4",
                   "--out", "/tmp/candidates-002.json"], agent)
        _, kwargs = agent.get_daily_picks.call_args
        assert (kwargs["shard"], kwargs["shards"]) == (2, 4)

    def test_out_of_range_shard_is_rejected(self, capsys):
        agent = self._agent()
        self._run(["betting_agent", "--picks", "--shard", "5/4"], agent)
        assert "Invalid --shard" in capsys.readouterr().out
        agent.get_daily_picks.assert_not_awaited()

    def test_shard_and_collect_are_mutually_exclusive(self, capsys):
        agent = self._agent()
        self._run(["betting_agent", "--picks", "--shard", "1/4",
                   "--collect", "/tmp"], agent)
        assert "mutually exclusive" in capsys.readouterr().out
