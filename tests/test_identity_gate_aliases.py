"""Stage 20 Part A — the identity gate's first live firings, classified and replayed.

The gate shipped `correct by construction, unverified in production` and ran for
the first time on 2026-08-27, the first day with a working API-Football account.
It refused three fixtures. A fail-closed gate's first live firing is exactly when
its refusals must be INSPECTED rather than trusted.

CLASSIFICATION, verified against the provider (GET /teams?id=) rather than recalled:

  604  Maccabi Tel Aviv  Israel,  Tel Aviv, 1906  vs stored "Telstar" (Netherlands)
       -> CORRECT REFUSAL. Two genuinely different clubs. Must stay refused.
  3502 FC Iberia 1999    Georgia, Tbilisi,  1999  vs stored "Saburtalo"
       -> FALSE POSITIVE. Same Tbilisi club, same founding year, renamed.
  531  Athletic Club     Spain,   Bilbao,   1898  vs stored "Ath Bilbao"
       -> FALSE POSITIVE. Cost Barcelona vs Ath Bilbao, one of the two fixtures
          Stage 19 predicted; it survived only because Flashscore found it too.

THE FIX IS KNOWLEDGE, NOT TOLERANCE. Two curated aliases were added and the
anchor rule is untouched — no ratio, no threshold, no widened country band. The
residual class is a legitimate pair with ZERO shared tokens, which no lexical
test can reach by construction, and that is precisely why a curated table exists.

AN ALIAS THAT ADMITS AN IMPOSTOR IS WORSE THAN THE FALSE POSITIVE IT FIXES, so
the impostor cases are replayed here alongside the fixes.
"""

import pytest

from src.scrapers.apifootball_scraper import (
    is_a_real_country, names_share_an_anchor)


# ── the two false positives must now resolve ────────────────────────────────
@pytest.mark.parametrize("incoming,stored", [
    ("Athletic Club", "Ath Bilbao"),
    ("FC Iberia 1999", "Saburtalo"),
    ("Iberia 1999", "Saburtalo"),
])
def test_false_positives_now_resolve(incoming, stored):
    assert names_share_an_anchor(incoming, stored), (
        f"{incoming!r} still refuses {stored!r} — a legitimate fixture is still "
        "being lost every time this pair appears")


# ── the impostor must STILL be refused ──────────────────────────────────────
def test_the_telstar_impostor_is_still_refused():
    """The defect the gate was built for. API-Football id 604 is Maccabi Tel
    Aviv (Israel); the row holding that id was named Telstar, a Dutch club
    whose Eredivisie history then priced the away side."""
    assert not names_share_an_anchor("Maccabi Tel Aviv", "Telstar"), (
        "an alias now admits the Telstar/Maccabi impostor — this is strictly "
        "worse than the false positives it was meant to fix")


def test_pau_fc_st_pauli_is_still_refused():
    assert not names_share_an_anchor("Pau FC", "St. Pauli")


# ── the country check still closes what the anchor test provably cannot ─────
def test_rapid_vienna_bucuresti_shares_an_anchor_and_needs_the_country_check():
    """Documented limitation, pinned so it is never mistaken for a regression.

    Both anchor on "rapid", so the LEXICAL test passes them — by design, it
    closes only the total-disagreement case. The COUNTRY check is what refuses
    this pair, and that is what the next test verifies.
    """
    assert names_share_an_anchor("Rapid Vienna", "Rapid Bucuresti")


def test_the_country_check_refuses_rapid_vienna_against_a_romanian_row():
    """Replays the gate's country clause exactly as `_get_or_create_team_id`
    applies it: refuse only when BOTH sides name a real country and they
    differ."""
    incoming_country, stored_country = "Austria", "Romania"
    assert is_a_real_country(incoming_country)
    assert is_a_real_country(stored_country)
    refuses = (is_a_real_country(incoming_country)
               and is_a_real_country(stored_country)
               and incoming_country.strip().lower() != stored_country.strip().lower())
    assert refuses, "the country check no longer separates Austria from Romania"


@pytest.mark.parametrize("value", ["Europe", "World", "", None, "Other"])
def test_missing_country_information_falls_through_rather_than_refusing(value):
    """`teams.country` records where a club was first SEEN, not where it plays.

    Saburtalo is stored as "Europe" because a continental tie created it, so a
    country check that refused on non-countries would reject legitimate
    fixtures wholesale — including the very one this stage fixed.
    """
    assert not is_a_real_country(value), (
        f"{value!r} is being treated as a real country; the gate would refuse "
        "fixtures whose stored row simply lacks country information")


def test_the_alias_knowledge_lives_in_exactly_one_table():
    """THE HABIT guard, and it earned its keep immediately.

    Stage 20 found TWO alias tables already in the tree:
    `apifootball_scraper.TEAM_NAME_ALIASES` (177 entries, "API-Football name ->
    historical name") and `team_names.NAME_ALIASES` (22), overlapping on 'psg'
    and 'olympiakos piraeus'.

    The first Stage 20 fix added "athletic club" to the SECOND table — and this
    guard failed, because `TEAM_NAME_ALIASES['Athletic Club'] = 'Ath Bilbao'`
    ALREADY EXISTED. The knowledge was present and simply unreachable: the alias
    table is consulted at step 2 of `_get_or_create_team_id`, while the gate
    refuses at step 0. Only ONE alias was genuinely missing.
    """
    from src.scrapers.apifootball_scraper import TEAM_NAME_ALIASES
    from src.utils.team_names import NAME_ALIASES
    lowered = {k.lower() for k in TEAM_NAME_ALIASES}
    assert "athletic club" in lowered, "the pre-existing Athletic alias vanished"
    assert "fc iberia 1999" in lowered, "the one genuinely new alias is missing"
    assert not ({"athletic club", "fc iberia 1999", "iberia 1999"}
                & set(NAME_ALIASES)), (
        "a Stage 20 alias was duplicated into team_names.NAME_ALIASES — two "
        "tables would then carry the same knowledge and drift. Pick one.")


def test_the_lowercase_index_covers_the_whole_table():
    """The index and the table cannot drift: it is derived, not maintained."""
    from src.scrapers.apifootball_scraper import _ALIAS_LOWER, TEAM_NAME_ALIASES
    assert len(_ALIAS_LOWER) == len({k.strip().lower() for k in TEAM_NAME_ALIASES})


# ── Stage 21 Part A: the regression Stage 20 created, and the rule behind it ──
#
# Stage 20 REPLACED each name with its canonical form before computing anchors.
# TEAM_NAME_ALIASES["Standard Liege"] = "Standard" is a PRE-EXISTING entry, so:
#     "Standard Liege" -> "Standard"   anchors {stan, standard}
#     "St. Liege"      -> (no alias)   anchors {lieg, liege}
#     intersection EMPTY -> refused, though the RAW names share "liege"
#
# THE RULE: a one-directional map (many provider forms -> ONE canonical form)
# applied to BOTH SIDES of a comparison can delete the very token the comparison
# depends on.
#
# The sharper form, from the Stage 21 audit of `_norm`:
#   * canonicalising both sides is SAFE for an EQUALITY test — it can only
#     increase agreement (this is why `same_team_strict` is unaffected);
#   * canonicalising both sides is UNSAFE for an OVERLAP test — it can remove
#     the overlapping token (this gate, and `team_names_similar`'s ratio).
#
# The fix is to UNION raw and aliased anchors, which can only ADD anchors and so
# can never refuse a pair the pre-Stage-20 gate accepted.

PREVIOUSLY_PASSING = [
    ("Standard Liege", "St. Liege"),          # the Stage 20 regression itself
    ("Union St. Gilloise", "St. Gilloise"),
    ("NEC Nijmegen", "Nijmegen"),
    ("Heart Of Midlothian", "Hearts"),
    ("Red Bull Salzburg", "Salzburg"),
    ("CFR 1907 Cluj", "CFR Cluj"),
    ("Universitatea Craiova", "Univ. Craiova"),
    ("Ferencvarosi TC", "Ferencvaros"),
]


@pytest.mark.parametrize("incoming,stored", PREVIOUSLY_PASSING)
def test_every_previously_passing_pair_still_passes(incoming, stored):
    """Union can only ADD anchors, so this set can never shrink."""
    assert names_share_an_anchor(incoming, stored), (
        f"{incoming!r} vs {stored!r} is refused. Stage 20 introduced exactly "
        "this failure by REPLACING names with their canonical form; if it has "
        "returned, canonicalisation is being applied instead of unioned.")


def test_the_standard_liege_regression_specifically():
    """Named because it reached production and skipped a real fixture."""
    assert names_share_an_anchor("Standard Liege", "St. Liege")


@pytest.mark.parametrize("incoming,stored", [
    ("Maccabi Tel Aviv", "Telstar"),
    ("Pau FC", "St. Pauli"),
    ("Cracovia Krakow", "Rakow"),
])
def test_impostors_are_still_refused_after_the_union_fix(incoming, stored):
    """A fix that admits an impostor is worse than the false positive it removes.

    Cracovia/Rakow is the fifth confirmed identity corruption: stored row 411
    ("Rakow") carries API-Football id 350, which is Cracovia's. The gate found
    it by refusing a real fixture. Recorded, NOT repaired.
    """
    assert not names_share_an_anchor(incoming, stored)
