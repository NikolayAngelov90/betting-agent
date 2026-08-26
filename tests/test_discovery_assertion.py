"""Stage 19 Part D — the audit must be able to see a day that discovers nothing.

On 2026-08-26 the pipeline analysed 0 fixtures against a card of six real
matches. `ci_audit.py` flagged the run only for the API-Football suspension. The
zero was invisible to the mechanical pass and surfaced because a human looked at
a football calendar.

Stage 14 specified self-calibrating assertions for exactly this shape — a unit
that produced data recently produces none. Discovery was never given one.

THE POSITIVE CONTROL MATTERS AS MUCH AS THE ASSERTION. Stage 13 established that
two audit disables were no-ops whose regexes matched nothing, so "26 passed"
meant nothing. An assertion that fires on everything is as useless as one that
fires on nothing, so both directions are pinned here.
"""

import re

from scripts.ci_audit import PATTERNS, assertions, extract

ACTIVE_ZERO = ("2026-08-26 10:20:33 | WARNING | src.scrapers.flashscore_scraper:"
               "scrape_league_fixtures:687 - Flashscore returned 0 fixtures for "
               "spain/laliga \u2014 expected \u22651 for active season")
NO_FIXTURES = ("2026-08-26 10:22:25 | INFO | __main__:get_daily_picks:1435 - "
               "No fixtures found for 2026-08-26")
GOOD_DAY = ("2026-05-30 09:31:02 | INFO | src.scrapers.flashscore_scraper:"
            "scrape_league_fixtures:690 - Scraped 11 fixtures from spain/laliga")


def test_zero_fixtures_for_an_active_league_is_a_finding():
    hits = assertions(extract(ACTIVE_ZERO), [])
    assert any("0 fixtures" in h for h in hits), (
        "a league the scraper itself calls in-season returned nothing and the "
        "audit said nothing. This was true on EVERY run from 2026-05-30 to "
        "2026-08-26 and was never reported.")


def test_a_day_that_found_no_fixtures_at_all_is_a_finding():
    hits = assertions(extract(NO_FIXTURES), [])
    assert any("NO FIXTURES FOUND" in h for h in hits)


def test_a_working_day_is_silent():
    """The positive control. An assertion that fires on success is noise."""
    hits = assertions(extract(GOOD_DAY), [])
    assert not any("fixture" in h.lower() for h in hits), (
        f"fired on a day that scraped 11 fixtures: {hits}")


def test_the_off_season_exclusion_is_inherited_not_reimplemented():
    """THE HABIT guard. The scraper already knows which leagues are dormant.

    The pattern keys on the scraper's own 'expected >=1 for active season'
    warning, which is emitted only for leagues absent from
    `scraping.off_season_leagues`. A second off-season list in the audit would
    be a sixth data-layer copy of a definition, and the two would drift.
    """
    assert "active season" in PATTERNS["fixtures_zero_active"], (
        "the discovery assertion no longer keys on the scraper's own "
        "in-season judgement — it now needs its own off-season list, which is "
        "a definition that will drift from the scraper's")


def test_patterns_are_valid_regexes():
    for name, pat in PATTERNS.items():
        re.compile(pat)
