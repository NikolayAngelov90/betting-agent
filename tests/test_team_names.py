"""Regression tests for national-team name aliases.

Different data sources name the same national side differently
(API-Football "USA" vs football-data.org "United States"). Without an alias,
token/prefix matching can't equate them, so a second Match row is created for
the same fixture and it gets briefed/picked twice (USA vs Bosnia, 2026-07-02).
"""

from src.utils.team_names import team_names_similar


def test_usa_variants_match():
    assert team_names_similar("USA", "United States")
    assert team_names_similar("USA", "United States of America")
    assert team_names_similar("United States", "United States of America")


def test_bosnia_variants_match():
    assert team_names_similar("Bosnia & Herzegovina", "Bosnia-Herzegovina")
    assert team_names_similar("Bosnia and Herzegovina", "Bosnia-Herzegovina")


def test_other_national_variants_match():
    assert team_names_similar("Korea Republic", "South Korea")
    assert team_names_similar("Ivory Coast", "Cote d'Ivoire")
    assert team_names_similar("Czechia", "Czech Republic")


def test_distinct_teams_do_not_match():
    # Aliases must not over-merge unrelated sides.
    assert not team_names_similar("USA", "Australia")
    assert not team_names_similar("South Korea", "North Korea")
    assert not team_names_similar("Bosnia-Herzegovina", "Croatia")
