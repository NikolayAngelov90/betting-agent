"""Regression tests for national-team name aliases.

Different data sources name the same national side differently
(API-Football "USA" vs football-data.org "United States"). Without an alias,
token/prefix matching can't equate them, so a second Match row is created for
the same fixture and it gets briefed/picked twice (USA vs Bosnia, 2026-07-02).
"""

from src.utils.team_names import team_names_similar, same_team_strict


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


class TestSameTeamStrict:
    """same_team_strict is the write-path dedup guard: it must collapse pure
    spelling variants but NEVER merge two genuinely different clubs."""

    def test_collapses_diacritic_variants(self):
        assert same_team_strict("Malmö FF", "Malmo FF")
        assert same_team_strict("IFK Göteborg", "IFK Goteborg")

    def test_collapses_club_tag_variants(self):
        assert same_team_strict("FC Copenhagen", "Copenhagen")
        assert same_team_strict("SK Rapid Wien", "Rapid Wien")

    def test_collapses_curated_aliases(self):
        assert same_team_strict("PSG", "Paris Saint-Germain")
        assert same_team_strict("Ath Madrid", "Atletico Madrid")

    def test_keeps_same_city_rivals_apart(self):
        # The whole reason this is strict (no fuzzy ratio): these share a token
        # but are different clubs and must stay separate.
        assert not same_team_strict("AC Milan", "Inter Milan")
        assert not same_team_strict("Sheffield United", "Sheffield Wednesday")
        assert not same_team_strict("Manchester United", "Manchester City")

    def test_keeps_unrelated_clubs_apart(self):
        assert not same_team_strict("Arsenal", "Arsenal Tula")
        assert not same_team_strict("Legia Warsaw", "Lech Poznan")
