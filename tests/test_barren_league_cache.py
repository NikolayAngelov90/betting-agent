"""Stage 15, L2 — the only lever implemented, and the guards it needs.

The lever removes requests that return nothing. That is safe ONLY while three
properties hold, and each is pinned here because each has a plausible edit that
would quietly destroy it:

1. A league must EARN exclusion. A threshold of 1 would have stopped pricing
   four major leagues in the measured window.
2. The exclusion must EXPIRE. Without this the cache is a blocklist that learns
   once and is wrong forever — the failure this design exists to prevent.
3. A refusal must be VISIBLE. A skip that logs nothing is indistinguishable
   from a league that was never a candidate.
"""

import json
import logging
import pathlib
from datetime import timedelta

import pytest

from src.scrapers.barren_leagues import (
    BARREN_CONSECUTIVE_THRESHOLD, BARREN_TTL_DAYS, BarrenLeagueCache)
from src.utils.logger import utcnow


@pytest.fixture
def cache(tmp_path):
    return BarrenLeagueCache(tmp_path / "barren.json")


def test_one_empty_fetch_does_not_exclude(cache):
    """france/ligue-1 returned no_rows exactly once and is plainly priced."""
    cache.record("france/ligue-1", empty=True)
    assert not cache.should_skip("france/ligue-1"), (
        "a single empty response excluded a league — in the measured window "
        "that rule would have stopped pricing Ligue 1, Eredivisie, LaLiga and "
        "Primeira Liga, all of which are priced")


def test_exclusion_requires_the_full_threshold(cache):
    lg = "europe/europa-conference-league"
    for i in range(BARREN_CONSECUTIVE_THRESHOLD - 1):
        cache.record(lg, empty=True)
        assert not cache.should_skip(lg), f"excluded after only {i + 1}"
    cache.record(lg, empty=True)
    assert cache.should_skip(lg), "never excluded despite reaching the threshold"


def test_one_success_clears_the_record(cache):
    """Coverage returns mid-season; the cache must notice on the first success."""
    lg = "europe/europa-league"
    for _ in range(BARREN_CONSECUTIVE_THRESHOLD):
        cache.record(lg, empty=True)
    assert cache.should_skip(lg)
    cache.record(lg, empty=False)
    assert not cache.should_skip(lg), (
        "the provider returned odds and the league is still excluded")


def test_the_exclusion_expires(cache):
    """THE load-bearing test. If this fails, L2 has become a blocklist.

    An exclusion that never lifts encodes a fact that expires — the provider
    does not price Conference League qualifiers in July and does price the
    group stage in September. A cache that cannot learn that is worse than the
    waste it removes, because the waste is 34 credits and this is silent
    permanent blindness.
    """
    lg = "europe/champions-league"
    for _ in range(BARREN_CONSECUTIVE_THRESHOLD):
        cache.record(lg, empty=True)
    assert cache.should_skip(lg)

    later = utcnow() + timedelta(days=BARREN_TTL_DAYS, minutes=1)
    assert not cache.should_skip(lg, now=later), (
        f"a league excluded {BARREN_TTL_DAYS} days ago is STILL excluded — the "
        "cache never expires and will never discover it was wrong")


def test_after_expiry_the_league_must_re_earn_exclusion(cache):
    """Otherwise the probe's single empty response re-excludes immediately."""
    lg = "europe/champions-league"
    for _ in range(BARREN_CONSECUTIVE_THRESHOLD):
        cache.record(lg, empty=True)
    later = utcnow() + timedelta(days=BARREN_TTL_DAYS, minutes=1)
    cache.should_skip(lg, now=later)          # expires it
    cache.record(lg, empty=True, now=later)   # the probe comes back empty
    assert not cache.should_skip(lg, now=later), (
        "one empty probe after expiry re-excluded the league — the TTL is "
        "cosmetic and the cache is still permanent")


def test_the_refusal_is_logged(cache, caplog):
    lg = "europe/europa-conference-league"
    with caplog.at_level(logging.INFO):
        for _ in range(BARREN_CONSECUTIVE_THRESHOLD):
            cache.record(lg, empty=True)
    assert any("EXCLUDING" in r.message and lg in r.message
               for r in caplog.records), (
        "a league was silently dropped from every future refresh")


def test_state_survives_a_restart(tmp_path):
    """CI is a fresh checkout each run; an in-memory cache saves nothing."""
    path = tmp_path / "barren.json"
    c1 = BarrenLeagueCache(path)
    for _ in range(BARREN_CONSECUTIVE_THRESHOLD):
        c1.record("europe/europa-league", empty=True)
    c1.save()
    assert BarrenLeagueCache(path).should_skip("europe/europa-league"), (
        "the cache did not survive a process restart, so in CI it is a no-op")


def test_a_corrupt_file_does_not_take_the_refresh_down(tmp_path):
    path = tmp_path / "barren.json"
    path.write_text("{not json", encoding="utf-8")
    assert not BarrenLeagueCache(path).should_skip("anything"), (
        "a corrupt cache file must degrade to 'refresh everything', not raise")


def test_an_unwritable_path_does_not_raise(tmp_path):
    c = BarrenLeagueCache(tmp_path / "nope" / "x" / "barren.json")
    c.record("a/b", empty=True)
    c.save()  # must not raise


def test_a_fetch_error_is_not_evidence_of_being_unpriced():
    """Pinned at the call site's contract: only `no_rows` counts.

    A timeout says nothing about the provider's catalogue. Counting it would
    let three flaky runs exclude a perfectly well-priced league.
    """
    src = pathlib.Path("src/scrapers/theodds_scraper.py").read_text(encoding="utf-8")
    assert 'if outcome in ("ok", "no_rows"):' in src, (
        "the barren cache is being fed from outcomes that may include "
        "'error' — a network failure would count as the provider not pricing "
        "the league")


def test_the_workflow_persists_the_record_across_runs():
    """WITHOUT THIS THE LEVER IS A NO-OP, and silently so.

    `closing-lines.yml` had no cache step of any kind. Every run starts from a
    fresh checkout, so the record would be written, consulted once within the
    same process, and discarded. The exclusion threshold is three CONSECUTIVE
    runs — unreachable if nothing survives a run. The lever would have measured
    as "implemented, 0 credits saved", which is indistinguishable from the
    provider having started pricing everything.
    """
    wf = pathlib.Path(".github/workflows/closing-lines.yml").read_text(encoding="utf-8")
    assert "actions/cache/restore" in wf and "odds_barren_leagues.json" in wf, (
        "closing-lines.yml no longer restores the barren-league record — the "
        "L2 lever is now a no-op on every run")
    assert "actions/cache/save" in wf, (
        "closing-lines.yml no longer saves the barren-league record — it can "
        "be read but never accumulates, so nothing ever reaches the threshold")
    assert "restore-keys" in wf, (
        "the restore has no restore-keys, so a run-id-scoped key can only ever "
        "hit its own run — which never exists on a first attempt")


def test_the_cache_file_is_not_committed():
    """A committed record ships a frozen exclusion list to every deployment."""
    ignored = pathlib.Path(".gitignore").read_text(encoding="utf-8")
    assert "odds_barren_leagues.json" in ignored, (
        "the barren-league record is not gitignored — committing it would ship "
        "one machine's observations as everyone's permanent exclusions")
