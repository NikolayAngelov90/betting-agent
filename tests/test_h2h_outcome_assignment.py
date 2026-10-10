"""1X2 outcome assignment — the away price was written as Home. 2026-10-10.

FOUND BY LOOKING FOR A DUPLICATING SNAPSHOT WRITER AND FINDING A MISCLASSIFIER.

`_save_game_odds` classified each `h2h` outcome with a chain that tested the
HOME name first:

    if _team_names_similar(oname, home_name):   selection = "Home"
    elif oname.lower() == "draw":               selection = "Draw"
    elif _team_names_similar(oname, away_name): selection = "Away"

`_team_names_similar` is a FUZZY predicate and `_tok_match` accepts substring
containment, so `"ce"` matches `"ceuta"`; with a one-token name the token ratio
is 1/1 = 1.0. For **Ceuta vs Sabadell** the away outcome `"CE Sabadell FC"`
therefore satisfied the FIRST branch and was written as **"Home"**. Two Home
rows, no Away row — and in `odds`, which is UNIQUE on
(match, bookmaker, market, selection), the second write OVERWROTE the first, so
the live Home price was the AWAY team's price.

MEASURED IN PRODUCTION 2026-10-10, TheOddsAPI 1X2 over 2,171 matches:

    44,719 (match, bookmaker) keys    44,210 complete (H+D+A)
       486 missing AWAY, Home present      0 missing HOME
        28 matches affected                12 of them carry a saved pick
                                            4 of those on TODAY's card

`missing_HOME = 0` is the signature: the misassignment is strictly directional,
exactly as testing `home_name` first predicts. A symmetric naming problem would
have produced both.

The affected pairs name the mechanism — shared or short tokens:
Ceuta/Sabadell, Villarreal/Real Madrid, Start/Fredrikstad, Paris FC/Paris SG,
Dundee Utd/Dundee FC, St. Mirren/St Johnstone, Gent/Genk, West Ham/Wolves.

THE FIX IS A SHAPE CHANGE, not a threshold change. Two outcomes and two teams
is an ASSIGNMENT problem: which pairing is better, not whether each name passes
a bar. `_name_affinity` grades it and the better of the two pairings wins.
`_team_names_similar` is left EXACTLY as it was, because fixture matching
applies it as a conjunction over home AND away, where a false positive on one
side is caught by the other — it is right there and wrong here.

Why a graded score and not the boolean: for Ceuta the token ratio returns 1.0
for BOTH pairings, so no boolean or token-argmax can separate them. The
sequence ratio gives 0.770/0.843 direct against 0.376/0.376 swapped.
"""

from datetime import datetime, timedelta

import pytest

import src.data.database as db_mod
from src.data.models import Base, Match, Odds, Team
from src.scrapers.theodds_scraper import (
    _OUTCOME_MIN_AFFINITY,
    TheOddsScraper,
    _name_affinity,
    _team_names_similar,
)

#: The real pair, and the real API naming. `AD Ceuta FC` / `CE Sabadell FC` are
#: the forms that produced the defect.
CEUTA, SABADELL = "AD Ceuta FC", "CE Sabadell FC"


def _mgr(tmp_path, name="h2h.db"):
    mgr = db_mod.DatabaseManager(
        config=type("C", (), {"database": {"sqlite_path": str(tmp_path / name)}})())
    Base.metadata.create_all(mgr.engine)
    assert not mgr.is_postgres
    return mgr


def _scraper(mgr):
    sc = TheOddsScraper.__new__(TheOddsScraper)
    sc.config = type("C", (), {"get": staticmethod(lambda *a, **k: None)})()
    sc.api_key = "test-key"
    sc.db = mgr
    sc._remaining_requests = None
    return sc


def _game(home, away, h_price, d_price, a_price, book="pinnacle"):
    """One Odds-API game payload, outcomes in the API's own order."""
    return {
        "home_team": home, "away_team": away,
        "bookmakers": [{
            "key": book,
            "markets": [{
                "key": "h2h",
                "outcomes": [
                    {"name": home, "price": h_price},
                    {"name": away, "price": a_price},
                    {"name": "Draw", "price": d_price},
                ],
            }],
        }],
    }


def _seed(mgr, mid, home, away):
    with mgr.get_session() as s:
        s.add(Team(id=1, name=home))
        s.add(Team(id=2, name=away))
        s.flush()
        s.add(Match(id=mid, league="spain/laliga2",
                    match_date=datetime(2026, 10, 10, 14, 15),
                    is_fixture=True, home_team_id=1, away_team_id=2))
        s.commit()


def _written(mgr, mid):
    with mgr.get_session() as s:
        return {o.selection: o.odds_value for o in
                s.query(Odds).filter_by(match_id=mid, market_type="1X2").all()}


# ═════════════════════════ the defect, driven end to end ═════════════════════

def test_the_away_outcome_is_NOT_written_as_Home(tmp_path):
    """Ceuta vs Sabadell, the exact case. Home 2.30, Draw 3.45, Away 2.96."""
    mgr = _mgr(tmp_path)
    _seed(mgr, 52743, "Ceuta", "Sabadell")
    sc = _scraper(mgr)
    with mgr.get_session() as s:
        sc._save_game_odds(
            _game(CEUTA, SABADELL, 2.30, 3.45, 2.96), 52743, [], session=s)
        s.commit()

    got = _written(mgr, 52743)
    assert got == {"Home": 2.30, "Draw": 3.45, "Away": 2.96}, (
        f"1X2 was written as {got} — before the fix this produced "
        f"{{'Home': <away price>, 'Draw': ...}} with NO Away row, because "
        f"'CE Sabadell FC' satisfied the home branch first")


def test_the_premise_the_old_chain_relied_on_is_FALSE(tmp_path):
    """The positive control, and it is why a threshold tweak cannot fix this.

    If `_team_names_similar(away_outcome, home_name)` were False, the old chain
    would have been correct. It is True — so the old code was wrong on this
    input by construction, and this test says so rather than asserting it.
    """
    assert _team_names_similar(SABADELL, CEUTA) is True, (
        "the away OUTCOME name no longer fuzzily matches the game header's "
        "home_team — the defect's precondition is gone, so check whether this "
        "file still tests anything")
    assert _team_names_similar(CEUTA, CEUTA) is True
    # The same holds for six more production pairs, compared the way the old
    # code compared them (API outcome name vs API header name):
    for away, home in [("Villarreal", "Real Madrid"), ("Paris SG", "Paris FC"),
                       ("Dundee FC", "Dundee Utd"), ("Genk", "Gent"),
                       ("Wolves", "West Ham"), ("Alanyaspor", "Antalyaspor")]:
        assert _team_names_similar(away, home) is True, (away, home)


def test_the_token_ratio_CANNOT_separate_them_but_affinity_can():
    """Why the fix needed a graded score rather than an argmax over booleans."""
    direct = _name_affinity(CEUTA, "Ceuta") + _name_affinity(SABADELL, "Sabadell")
    swap = _name_affinity(CEUTA, "Sabadell") + _name_affinity(SABADELL, "Ceuta")
    assert direct > swap, (direct, swap)
    assert direct - swap > 0.5, (
        f"the margin is only {direct - swap:.3f} — too thin to rely on; the "
        f"measured values were 1.613 against 0.752")


def test_a_swapped_payload_is_assigned_by_CONTENT_not_by_order(tmp_path):
    """Order must not be the deciding signal, or the API's ordering is a
    silent dependency. Away first, home second."""
    mgr = _mgr(tmp_path)
    _seed(mgr, 900, "Ceuta", "Sabadell")
    sc = _scraper(mgr)
    payload = {
        "home_team": CEUTA, "away_team": SABADELL,
        "bookmakers": [{"key": "pinnacle", "markets": [{"key": "h2h", "outcomes": [
            {"name": SABADELL, "price": 2.96},
            {"name": "Draw", "price": 3.45},
            {"name": CEUTA, "price": 2.30},
        ]}]}],
    }
    with mgr.get_session() as s:
        sc._save_game_odds(payload, 900, [], session=s)
        s.commit()
    assert _written(mgr, 900) == {"Home": 2.30, "Draw": 3.45, "Away": 2.96}


@pytest.mark.parametrize("home,away", [
    ("Ceuta", "Sabadell"),
    ("Real Madrid", "Villarreal"),
    ("Start", "Fredrikstad"),
    ("Paris FC", "Paris SG"),
    ("St. Mirren", "St Johnstone"),
    ("Gent", "Genk"),
    ("West Ham", "Wolves"),
    ("Wisla Krakow", "Wieczysta Krakow"),
    ("Antalyaspor", "Alanyaspor"),
])
def test_every_pair_that_was_CORRUPTED_in_production_now_assigns(tmp_path, home, away):
    """The affected pairs, taken from the production query rather than invented.

    Each is driven through the real parser with the DB names as the API names,
    which is the favourable case; the point is that NONE of them may now lose
    its Away row.
    """
    mgr = _mgr(tmp_path, f"{abs(hash((home, away)))}.db")
    _seed(mgr, 1, home, away)
    sc = _scraper(mgr)
    with mgr.get_session() as s:
        sc._save_game_odds(_game(home, away, 2.10, 3.40, 3.20), 1, [], session=s)
        s.commit()
    got = _written(mgr, 1)
    assert got == {"Home": 2.10, "Draw": 3.40, "Away": 3.20}, (
        f"{home} vs {away} wrote {got} — a missing or duplicated side means "
        f"the live price for one team is the other team's price")


# ═══════════════════════ refusing instead of guessing ════════════════════════

def test_a_market_that_is_not_two_teams_is_SKIPPED_and_COUNTED(tmp_path):
    """One non-draw outcome is a shape we do not understand.

    Partially writing it would put one real price on an unknown side. Counted
    so the skip is a number rather than an absence.
    """
    mgr = _mgr(tmp_path)
    _seed(mgr, 2, "Ceuta", "Sabadell")
    sc = _scraper(mgr)
    payload = {
        "home_team": CEUTA, "away_team": SABADELL,
        "bookmakers": [{"key": "pinnacle", "markets": [{"key": "h2h", "outcomes": [
            {"name": CEUTA, "price": 2.30},
            {"name": "Draw", "price": 3.45},
        ]}]}],
    }
    with mgr.get_session() as s:
        sc._save_game_odds(payload, 2, [], session=s)
        s.commit()
    got = _written(mgr, 2)
    assert "Home" not in got and "Away" not in got, (
        f"a one-sided market wrote a team price anyway: {got}")
    assert got == {"Draw": 3.45}, got
    assert getattr(sc, "_h2h_unassigned", 0) == 1, (
        "the skip was silent — an unparsed market must be counted, or "
        "'nothing to price' and 'could not price it' read identically")


def test_an_unrelated_market_is_REFUSED_rather_than_assigned(tmp_path):
    """Both outcomes naming other clubs entirely: below the affinity floor."""
    mgr = _mgr(tmp_path)
    _seed(mgr, 3, "Ceuta", "Sabadell")
    sc = _scraper(mgr)
    payload = {
        # header says Ceuta vs Sabadell; the outcomes name two other clubs.
        # Only a payload INTERNALLY inconsistent like this can reach the floor:
        # the API normally repeats its own header names in the outcomes, which
        # is why the floor is a narrow guard and not the decision.
        "home_team": CEUTA, "away_team": SABADELL,
        "bookmakers": [{"key": "pinnacle", "markets": [{"key": "h2h", "outcomes": [
            {"name": "Shakhtar Donetsk", "price": 2.3},
            {"name": "Rosenborg", "price": 3.2},
            {"name": "Draw", "price": 3.4},
        ]}]}],
    }
    with mgr.get_session() as s:
        sc._save_game_odds(payload, 3, [], session=s)
        s.commit()
    got = _written(mgr, 3)
    assert "Home" not in got and "Away" not in got, (
        f"a market naming two unrelated clubs was assigned anyway: {got}")
    assert getattr(sc, "_h2h_unassigned", 0) == 1


def test_the_affinity_floor_does_not_reject_a_GENUINE_naming_difference(tmp_path):
    """The other direction. The floor is a safety net, not the decision.

    The game reached this fixture because `_team_names_similar` already matched
    BOTH sides upstream, so rejecting a real market here costs a price for no
    gain. `PSV` against `PSV Eindhoven` is a real alias in the table.
    """
    mgr = _mgr(tmp_path)
    _seed(mgr, 4, "PSV Eindhoven", "Ajax")
    sc = _scraper(mgr)
    payload = {
        # header carries the long form, the outcomes the short one — a real
        # header/outcome difference rather than the same string twice. The
        # first version of this test passed "PSV" as BOTH and therefore scored
        # a trivial 1.0, exercising no alias at all.
        "home_team": "PSV Eindhoven", "away_team": "Ajax",
        "bookmakers": [{"key": "pinnacle", "markets": [{"key": "h2h", "outcomes": [
            {"name": "PSV", "price": 1.80},
            {"name": "Ajax", "price": 4.20},
            {"name": "Draw", "price": 3.60},
        ]}]}],
    }
    with mgr.get_session() as s:
        sc._save_game_odds(payload, 4, [], session=s)
        s.commit()
    assert _written(mgr, 4) == {"Home": 1.80, "Draw": 3.60, "Away": 4.20}, (
        f"a genuine alias was refused — floor {_OUTCOME_MIN_AFFINITY} is too "
        f"high")
    assert getattr(sc, "_h2h_unassigned", 0) == 0


def test_totals_still_carry_their_LINE_in_the_selection(tmp_path):
    """Untouched by this change, and pinned because the same loop writes it.

    Over 2.25 and Over 2.75 are different selections, not duplicates — that
    distinction is what kept `over_under` out of this defect.
    """
    mgr = _mgr(tmp_path)
    _seed(mgr, 5, "Ceuta", "Sabadell")
    sc = _scraper(mgr)
    payload = {
        "home_team": CEUTA, "away_team": SABADELL,
        "bookmakers": [{"key": "pinnacle", "markets": [{"key": "totals", "outcomes": [
            {"name": "Over", "point": 2.25, "price": 1.81},
            {"name": "Under", "point": 2.25, "price": 1.93},
            {"name": "Over", "point": 2.75, "price": 1.92},
            {"name": "Under", "point": 2.75, "price": 1.83},
        ]}]}],
    }
    with mgr.get_session() as s:
        sc._save_game_odds(payload, 5, [], session=s)
        s.commit()
    with mgr.get_session() as s:
        got = {o.selection: o.odds_value for o in
               s.query(Odds).filter_by(match_id=5, market_type="over_under").all()}
    assert got == {"Over 2.25": 1.81, "Under 2.25": 1.93,
                   "Over 2.75": 1.92, "Under 2.75": 1.83}, got


# ═════════════ PART A: the snapshot writer no longer duplicates ══════════════

def test_one_call_writes_ONE_snapshot_row_per_selection(tmp_path):
    """THE PART A ASSERTION. Zero-gap "trajectories" were this defect's shadow.

    `_upsert_odds` calls `record_price` on every invocation with its own
    `utcnow()`, so classifying two outcomes onto one selection produced two
    snapshot rows MILLISECONDS apart. Measured in production 2026-10-10: 585
    (match, bookmaker, market, selection) keys carry more than one row inside
    a single second, 664 excess rows, 577 of them with DIFFERENT values, last
    seen 04:55:38 that morning — live.

    Those keys are also the only ones that reached three instants, so 100% of
    the apparent three-point trajectories were this. A trajectory with zero
    separation is not a trajectory.
    """
    from src.data.models import OddsSnapshot

    mgr = _mgr(tmp_path, "snapdup.db")
    _seed(mgr, 10, "Ceuta", "Sabadell")
    sc = _scraper(mgr)
    with mgr.get_session() as s:
        sc._save_game_odds(
            _game(CEUTA, SABADELL, 2.30, 3.45, 2.96), 10, [], session=s)
        s.commit()

    with mgr.get_session() as s:
        rows = s.query(OddsSnapshot).filter_by(match_id=10,
                                               market_type="1X2").all()
        per_sel = {}
        for r in rows:
            per_sel[r.selection] = per_sel.get(r.selection, 0) + 1

    assert per_sel == {"Home": 1, "Draw": 1, "Away": 1}, (
        f"one API call produced {per_sel} snapshot rows per selection — "
        f"anything above 1 is a same-instant duplicate and manufactures a "
        f"zero-gap trajectory")
    assert len(rows) == 3


def test_a_second_call_appends_rather_than_overwriting(tmp_path):
    """The append-only property H1 depends on, asserted through the writer.

    Two calls at different times must leave TWO rows per selection in
    `odds_snapshots` while `odds` keeps exactly one — that asymmetry is the
    whole reason the snapshot table exists.
    """
    from src.data.models import OddsSnapshot

    mgr = _mgr(tmp_path, "snapappend.db")
    _seed(mgr, 11, "Ceuta", "Sabadell")
    sc = _scraper(mgr)
    for price in (2.30, 2.45):
        with mgr.get_session() as s:
            sc._save_game_odds(
                _game(CEUTA, SABADELL, price, 3.45, 2.96), 11, [], session=s)
            s.commit()

    with mgr.get_session() as s:
        snaps = s.query(OddsSnapshot).filter_by(
            match_id=11, market_type="1X2", selection="Home").count()
        live = s.query(Odds).filter_by(
            match_id=11, market_type="1X2", selection="Home").all()
    assert snaps == 2, f"odds_snapshots kept {snaps} of 2 observations"
    assert len(live) == 1 and live[0].odds_value == 2.45, (
        "`odds` must keep exactly the latest price — if this changes, every "
        "current-price consumer is affected")


# ═══════════ A5: THE REPLAY CONTROL — 26 affected + 60 previously-correct ════

#: Every distinct (home, away) pair among the 28 matches the defect's signature
#: identifies in production, 2026-10-10. Data, not invention.
AFFECTED_PAIRS = [
    ("AEL Larissa", "Kifisia"), ("AIK", "Sirius"),
    ("Albacete", "Real Sociedad B"), ("Antalyaspor", "Alanyaspor"),
    ("Blackpool", "Exeter"), ("Burgos CF", "Real Sociedad B"),
    ("Ceuta", "Sabadell"), ("Dundee Utd", "Dundee FC"),
    ("Fredrikstad", "Start"), ("Gent", "Genk"), ("Kifisia", "AEL Larissa"),
    ("Le Havre", "Marseille"), ("Lille OSC", "Le Havre"),
    ("Malaga", "Dep. La Coruna"), ("Paris FC", "Paris SG"),
    ("Real Madrid", "Villarreal"),
    ("Real Racing Club de Santander", "Villarreal"),
    ("St. Mirren", "St Johnstone"), ("Start", "Fredrikstad"),
    ("Start", "Viking"), ("Sunderland", "Brighton"), ("Viking", "Start"),
    ("Villarreal", "Dep. La Coruna"), ("West Ham", "Wolves"),
    ("Wisla Krakow", "Wieczysta Krakow"), ("Zurich", "Grasshoppers"),
]

#: 60 pairs drawn at random from matches whose 1X2 is COMPLETE in production —
#: the cases the fix must NOT change. An assignment fix that reassigns a
#: working case is worse than the defect it repairs.
CONTROL_PAIRS = [
    ("AEK Athens", "PAOK"), ("Ajax", "Groningen"),
    ("Asteras Tripolis", "Panserraikos"), ("Austria Vienna", "LASK"),
    ("Austria Vienna", "Sturm Graz"), ("Barcelona", "Feyenoord"),
    ("Barnet", "Bromley"), ("Barnsley", "Stockport County"),
    ("Brommapojkarna", "Vasteras SK"), ("Cambuur", "Excelsior"),
    ("Castellon", "Cadiz CF"), ("Charlton", "Ipswich"),
    ("Clermont Foot", "Guingamp"), ("Como", "Napoli"),
    ("Crystal Palace", "Vallecano"), ("Dep. La Coruna", "Leganes"),
    ("Everton", "Liverpool"), ("Exeter", "Bradford City"),
    ("FC Copenhagen", "Odense"), ("FC Metz", "Paris FC"),
    ("FC Porto", "Arouca"), ("FC Porto", "Moreirense"),
    ("FC Porto", "Nott'm Forest"), ("France", "Iraq"), ("Genk", "Kortrijk"),
    ("Gornik Zabrze", "Zaglebie"), ("Grimsby", "Shrewsbury"),
    ("Hartberg", "Ried"), ("Heerenveen", "PEC Zwolle"), ("Malmo FF", "AIK"),
    ("Middlesbrough", "Southampton"), ("Midtjylland", "Viborg"),
    ("Monaco", "Marseille"), ("Nacional", "Alverca"), ("Nantes", "Brest"),
    ("Nijmegen", "Go Ahead Eagles"), ("Odense", "Fredericia"),
    ("Osasuna", "Sevilla"), ("Panetolikos", "Kifisia"),
    ("Paris SG", "Lorient"), ("Paris SG", "Le Mans"),
    ("Parma Calcio 1913", "Monza"), ("Portsmouth", "Birmingham"),
    ("Preston", "Blackburn"), ("Randers FC", "Vejle"),
    ("Rangers", "Hibernian"), ("Rochdale", "Swindon"), ("Servette", "Luzern"),
    ("Sion", "Thun"), ("SK Rapid", "Salzburg"),
    ("St Etienne", "Clermont Foot"), ("Start", "Tromso"),
    ("Swindon", "Chesterfield"), ("Tottenham", "Brighton"),
    ("Tromso", "Kristiansund"), ("Union Berlin", "Elversberg"),
    ("Valerenga", "Sarpsborg 08"), ("Viking", "Aalesund"),
    ("Wieczysta Krakow", "Lech Poznan"), ("Wycombe", "Luton"),
]

#: The one pair the fix CANNOT resolve, named rather than hidden.
UNRESOLVABLE = ("Dundee Utd", "Dundee FC")


def _margin(home, away):
    """direct minus swap. Positive means the correct pairing wins."""
    return ((_name_affinity(home, home) + _name_affinity(away, away))
            - (_name_affinity(home, away) + _name_affinity(away, home)))


def test_the_CONTROL_pairs_are_all_resolved_with_room_to_spare():
    """The cases the fix must not change. Measured min margin 1.0000."""
    margins = sorted((_margin(h, a), h, a) for h, a in CONTROL_PAIRS)
    assert margins[0][0] > 0, (
        f"a previously-correct pair is now a tie or inverted: {margins[0]} — "
        f"an assignment fix that reassigns a working case is worse than the "
        f"defect")
    assert margins[0][0] >= 0.9, (
        f"the tightest control margin fell to {margins[0][0]:.4f} from a "
        f"measured 1.0000 — the headroom on working cases has shrunk")


def test_every_AFFECTED_pair_is_resolved_EXCEPT_the_one_that_cannot_be():
    """25 of 26 resolve by content. The 26th is named and refuses.

    Reported as a residual rather than smoothed over: `Dundee Utd` and
    `Dundee FC` both canonicalise to `"dundee"`, because `_canon_team_name`
    strips " utd" and " fc" alike. Margin is EXACTLY 0.0000, so no graded score
    over these names can separate them — the information is destroyed by
    canonicalisation, before any comparison happens.
    """
    ties = [(h, a) for h, a in AFFECTED_PAIRS if _margin(h, a) <= 0]
    assert ties == [UNRESOLVABLE], (
        f"the set of unresolvable affected pairs changed: {ties} — if it grew, "
        f"the fix resolves less than it did; if it shrank, update this test "
        f"and the ledger rather than leaving a stale claim")
    for h, a in AFFECTED_PAIRS:
        if (h, a) == UNRESOLVABLE:
            continue
        assert _margin(h, a) > 0, (h, a, _margin(h, a))


def test_the_canonicaliser_is_what_destroys_the_Dundee_information():
    """Pinned at the cause, not the symptom.

    Fixing this would mean changing `_canon_team_name`, which
    `_team_names_similar` shares with FIXTURE MATCHING across three scrapers —
    so it is left alone deliberately and the tie refuses instead.
    """
    from src.scrapers.theodds_scraper import _canon_team_name

    assert _canon_team_name("Dundee Utd") == _canon_team_name("Dundee FC") == "dundee"
    assert _margin(*UNRESOLVABLE) == 0.0


@pytest.mark.parametrize("home,away", AFFECTED_PAIRS + CONTROL_PAIRS)
def test_the_full_REPLAY_through_the_real_parser(tmp_path, home, away):
    """Both groups, driven end to end. 86 pairs, one database each.

    The tie refuses rather than writing a side, so it is asserted as a refusal
    — which is the honest expectation, not a weakened one.
    """
    mgr = _mgr(tmp_path, f"r{abs(hash((home, away))) % 10**9}.db")
    _seed(mgr, 1, home, away)
    sc = _scraper(mgr)
    with mgr.get_session() as s:
        sc._save_game_odds(_game(home, away, 2.10, 3.40, 3.20), 1, [], session=s)
        s.commit()
    got = _written(mgr, 1)

    if (home, away) == UNRESOLVABLE:
        assert got == {"Draw": 3.40}, (
            f"the unresolvable pair wrote a team price: {got} — a tie must "
            f"refuse, not break toward payload order")
        assert getattr(sc, "_h2h_unassigned", 0) == 1
        return

    assert got == {"Home": 2.10, "Draw": 3.40, "Away": 3.20}, (
        f"{home} vs {away} wrote {got} — a missing or duplicated side means "
        f"one team's live price is the other team's price")
    assert getattr(sc, "_h2h_unassigned", 0) == 0
