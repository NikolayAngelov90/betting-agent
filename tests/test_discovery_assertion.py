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
import pytest

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


# Stage 19 item 1: PER-SOURCE, not aggregate.
#
# The first assertion shipped in Stage 19 keyed on total discovery. Replayed
# against 2026-05-31 -- the day Flashscore went silent -- it did NOT fire:
# flashscore=0, football-data.org=0, API-Football=13, so the total looked
# healthy. It then stayed silent for 88 days. The aggregate rebuilt the very
# blindness it was written to remove.

FLASHSCORE_DEAD_AF_ALIVE = chr(10).join([
    "Scraped 0 fixtures from spain/laliga",
    "API-Football: creating new fixture Roma vs Fiorentina",
    "API-Football: creating new fixture Torino vs Milan",
])
ALL_HEALTHY = chr(10).join([
    "Scraped 11 fixtures from spain/laliga",
    "football-data.org: 3 scores updated, 4 new fixtures added",
    "API-Football: creating new fixture Roma vs Fiorentina",
])


def _hist(**counts):
    """History in which each named source recently produced."""
    return [dict(counts) for _ in range(3)]


def test_one_dead_source_fires_even_while_others_produce():
    """THE 2026-05-31 CASE. This is the whole point of the per-source design."""
    facts = extract(FLASHSCORE_DEAD_AF_ALIVE)
    facts["is_first_run_of_day"] = True
    hits = assertions(facts, _hist(src_flashscore_fixtures=11,
                                   src_apifootball_fixtures=6))
    assert any("Flashscore fixtures = 0" in h for h in hits), (
        "Flashscore produced nothing while API-Football produced 2, and the "
        "audit stayed silent. That is the 88-day blindness, rebuilt.")


def test_all_sources_healthy_is_silent():
    facts = extract(ALL_HEALTHY)
    facts["is_first_run_of_day"] = True
    hits = assertions(facts, _hist(src_flashscore_fixtures=11,
                                   src_footballdataorg_fixtures=4,
                                   src_apifootball_fixtures=6))
    assert not any("fixtures = 0" in h for h in hits), hits


def test_a_same_day_rerun_does_not_fire():
    """A second run finds no NEW fixtures because the first added them."""
    facts = extract(FLASHSCORE_DEAD_AF_ALIVE)
    facts["is_first_run_of_day"] = False
    hits = assertions(facts, _hist(src_flashscore_fixtures=11))
    assert not any("fixtures = 0" in h for h in hits), (
        "fired on a same-day re-run, where zero new fixtures is correct")


def test_no_history_says_nothing():
    """An empty history means nothing can be said, and saying nothing is right."""
    facts = extract(FLASHSCORE_DEAD_AF_ALIVE)
    facts["is_first_run_of_day"] = True
    assert not any("fixtures = 0" in h for h in assertions(facts, []))


# Stage 19 item 3: with API-Football restored, ALL THREE sources may be alive.
# That is precisely the condition under which the watch must not go back to
# sleep -- it is the 2026-05-30 condition, when two healthy sources hid a third
# that had died. Each source is forced to zero in turn, by injection, rather
# than by waiting for it to happen.

HEALTHY = {
    "src_flashscore_fixtures": 11,
    "src_footballdataorg_fixtures": 4,
    "src_apifootball_fixtures": 6,
}
LINES = {
    "src_flashscore_fixtures": "Scraped {n} fixtures from spain/laliga",
    "src_footballdataorg_fixtures":
        "football-data.org: 3 scores updated, {n} new fixtures added",
    "src_apifootball_fixtures": "API-Football: creating new fixture A vs B",
}
LABELS = {
    "src_flashscore_fixtures": "Flashscore fixtures",
    "src_footballdataorg_fixtures": "football-data.org fixtures",
    "src_apifootball_fixtures": "API-Football fixtures",
}


def _log_with(counts):
    """Build a log in which each source reports the given count."""
    out = []
    for key, n in counts.items():
        if key == "src_apifootball_fixtures":
            out.extend([LINES[key].format(n=n)] * n)
            if n == 0:
                out.append("API-Football update complete (1 requests used)")
        else:
            out.append(LINES[key].format(n=n))
    return chr(10).join(out)


def test_all_three_alive_is_silent():
    """The everyday case. Firing here would train people to ignore it."""
    facts = extract(_log_with(HEALTHY))
    facts["is_first_run_of_day"] = True
    hits = assertions(facts, [dict(HEALTHY) for _ in range(3)])
    assert not any("fixtures = 0" in h for h in hits), hits


@pytest.mark.parametrize("dead", sorted(HEALTHY))
def test_each_source_forced_to_zero_fires_while_the_others_are_healthy(dead):
    """THE 88-DAY FAILURE, injected once per source.

    On 2026-05-30 Flashscore died while API-Football and football-data.org
    carried on. The aggregate stayed healthy and nothing alarmed for 88 days.
    With API-Football restored, that exact configuration is live again.
    """
    counts = dict(HEALTHY)
    counts[dead] = 0
    facts = extract(_log_with(counts))
    facts["is_first_run_of_day"] = True
    hits = assertions(facts, [dict(HEALTHY) for _ in range(3)])

    assert any(LABELS[dead] in h and "= 0" in h for h in hits), (
        f"{LABELS[dead]} produced nothing while the other two produced "
        f"normally, and the audit said nothing. That is the 88-day blindness.\n"
        f"hits={hits}")
    for other in HEALTHY:
        if other != dead:
            assert not any(LABELS[other] in h and "= 0" in h for h in hits), (
                f"fired for {LABELS[other]}, which produced {counts[other]}")
