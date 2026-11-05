"""ONE market/selection vocabulary: the copies are zero and the controls hold.

WHAT THIS GUARDS. 2026-11-05.

`src/data/market_vocabulary.py` replaced two AST-identical copies of
`selection_map`/`market_map` that lived inside two methods of
`value_calculator`. Measured cost of the split:

    h1_analysis step 4        0 of 1,120 trajectories matched 2,587 picks
    market-mix join         691 of 1,603 settled picks matched; 912 dropped

Both are now whole. The controls below are the ones that make the extraction
safe rather than merely tidy:

    KNOWN-WRONG   `over_under`/`Under 3.5` must equal `Under 3.5`/`Under 3.5 Goals`
    POSITIVE      genuinely different bets must STILL not match -- an
                  extraction that makes everything equal is worse than the split
    REFUSAL       an unnameable or ambiguous bet is None, never a guess

NOTE ON IMPORTS. This file imports `src.data.market_vocabulary` (standard
library only) and reads `capture_closing_lines` BY AST rather than importing
it, because that module is on the capture path and importing it here would
recruit these tests into the capture gate -- the trap that has cost four
events. `tests/test_gate_is_opt_in.py` holds the node-id check.
"""
import ast
import pathlib

from src.data.market_vocabulary import (
    HOME_TEAM,
    MARKET_ALIASES,
    SELECTION_ALIASES,
    SELECTION_FAMILY,
    canonical,
    family_of_market,
    market_aliases,
    same_bet,
    selection_aliases,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent
PRICER = ROOT / "src" / "betting" / "value_calculator.py"
CLOSING = ROOT / "scripts" / "capture_closing_lines.py"


# ── THE COPIES ARE ZERO, AND THE SCAN PROVES IT COULD HAVE FOUND THEM ───────

def test_the_pricer_holds_ZERO_copies_of_the_maps():
    """A scan finding zero sites is UNEVALUATED unless the anchor is asserted.

    `selection_map = {` is what was there. Asserting its absence proves
    nothing unless this file is demonstrably the pricer, so the anchor is
    checked first -- a renamed or moved file must fail loudly, not silently
    report clean.
    """
    src = PRICER.read_text(encoding="utf-8")
    assert "def _find_best_odds" in src and "def _find_opening_odds" in src, (
        "UNEVALUATED: this is not the pricer (neither method is present), so "
        "an absence claim about its copies verifies nothing")
    tree = ast.parse(src)
    literals = [
        n.lineno for n in ast.walk(tree)
        if isinstance(n, ast.Assign) and len(n.targets) == 1
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id in ("selection_map", "market_map")
        and isinstance(n.value, ast.Dict)
    ]
    assert literals == [], (
        f"the pricer holds {len(literals)} inline vocabulary dict(s) again at "
        f"line(s) {literals}; the extraction is undone")
    assert "market_vocabulary" in src, (
        "the pricer no longer imports the shared vocabulary, so the two "
        "copies did not move -- they were deleted")


def test_the_scan_above_CAN_find_a_copy_known_wrong_control():
    """The known-wrong input for the scan: a planted dict must be found."""
    planted = ast.parse('def f():\n    selection_map = {"a": ["b"]}\n')
    found = [
        n.lineno for n in ast.walk(planted)
        if isinstance(n, ast.Assign) and len(n.targets) == 1
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id in ("selection_map", "market_map")
        and isinstance(n.value, ast.Dict)
    ]
    assert found == [2], (
        f"the predicate used above cannot find a planted copy ({found}), so "
        f"its zero on the real file means nothing")


# ── THE KNOWN-WRONG CONTROL ─────────────────────────────────────────────────

def test_KNOWN_WRONG_the_pair_that_matched_zero_of_1120_now_matches():
    odds_side = canonical("over_under", "Under 3.5")
    pick_side = canonical("Under 3.5", "Under 3.5 Goals")
    assert odds_side is not None and pick_side is not None
    assert odds_side == pick_side == ("over_under", "Under 3.5 Goals")
    assert ("over_under", "Under 3.5") != ("Under 3.5", "Under 3.5 Goals"), (
        "if raw tuple equality already matched, this control is vacuous")
    assert same_bet("over_under", "Under 3.5", "Under 3.5", "Under 3.5 Goals")


def test_KNOWN_WRONG_the_market_axis_differs_even_when_the_selection_matches():
    """`team_goals` vs `Team Goals`: selection-only normalisation misses this."""
    a = canonical("team_goals", "Home Over 1.5")
    b = canonical("Team Goals", "Home Over 1.5")
    assert a == b == ("team_goals", "Home Over 1.5")
    assert "team_goals" != "Team Goals"


def test_every_production_pairing_in_the_window_reconciles():
    """Both vocabularies as they actually appear, measured 2026-11-05."""
    pairs = [
        (("1X2", "Home"), ("1X2", "Home Win")),
        (("1X2", "Away"), ("1X2", "Away Win")),
        (("1X2", "Draw"), ("1X2", "Draw")),
        (("over_under", "Over 2.5"), ("Over 2.5", "Over 2.5 Goals")),
        (("over_under", "Under 2.5"), ("Under 2.5", "Under 2.5 Goals")),
        (("over_under", "Under 3.5"), ("Under 3.5", "Under 3.5 Goals")),
        (("over_under", "Under 4.5"), ("Under 4.5", "Under 4.5 Goals")),
        (("team_goals", "Home Over 1.5"), ("Team Goals", "Home Over 1.5")),
        (("team_goals", "Away Over 1.5"), ("Team Goals", "Away Over 1.5")),
        (("btts", "Yes"), ("BTTS", "BTTS Yes")),
        (("double_chance", "X2"), ("Double Chance", "Double Chance X2")),
        (("draw_no_bet", "DNB Away"), ("Draw No Bet", "DNB Away")),
    ]
    for (om, osel), (pm, psel) in pairs:
        assert same_bet(om, osel, pm, psel), (
            f"the odds row {om}/{osel} does not reconcile with the pick "
            f"{pm}/{psel}, which is the split this module exists to close")


# ── THE POSITIVE CONTROL ────────────────────────────────────────────────────

def test_POSITIVE_CONTROL_genuinely_different_bets_do_NOT_match():
    """An extraction that makes everything equal is worse than the split."""
    different = [
        (("over_under", "Over 2.5"), ("Under 2.5", "Under 2.5 Goals"), "side"),
        (("over_under", "Over 2.5"), ("Over 3.5", "Over 3.5 Goals"), "line"),
        (("1X2", "Home"), ("1X2", "Away Win"), "outcome"),
        (("btts", "Yes"), ("BTTS", "BTTS No"), "btts side"),
        (("team_goals", "Home Over 1.5"),
         ("Team Goals", "Away Over 1.5"), "team"),
        (("double_chance", "1X"),
         ("Double Chance", "Double Chance X2"), "dc leg"),
    ]
    for (am, asel), (bm, bsel), label in different:
        assert not same_bet(am, asel, bm, bsel), (
            f"{label}: {am}/{asel} was equated with {bm}/{bsel}")


def test_POSITIVE_CONTROL_the_HOME_collision_stays_separated():
    """`Home` is an alias in TWO families; the market is what separates them."""
    assert canonical("1X2", "Home") == ("1X2", "Home Win")
    assert canonical("draw_no_bet", "Home") == ("draw_no_bet", "DNB Home")
    assert not same_bet("1X2", "Home", "Draw No Bet", "DNB Home")
    assert canonical(None, "Home") is None, (
        "without a market `Home` is Home Win OR DNB Home, and guessing it is "
        "how a bet nobody placed gets invented")


def test_every_canonical_name_is_its_own_fixed_point():
    for name in SELECTION_ALIASES:
        fam = SELECTION_FAMILY[name]
        assert canonical(fam, name) == (fam, name), name


# ── REFUSAL, NOT A GUESS ────────────────────────────────────────────────────

def test_it_REFUSES_rather_than_guessing():
    assert canonical("1X2", "Under 3.5") is None, (
        "market and selection disagree about the family; preferring either "
        "silently invents a bet")
    assert canonical("over_under", "Gibberish") is None
    assert canonical("over_under", "Over 2.25") is None, (
        "quarter lines are real prices the pipeline never picks; they have no "
        "canonical name and must not acquire one by accident"
    )
    assert canonical(None, None) is None
    assert canonical("over_under", "") is None


def test_same_bet_is_FALSE_when_either_side_is_unnameable():
    """'I cannot tell' must never read as 'they are the same'."""
    assert not same_bet("over_under", "Over 2.25",
                        "over_under", "Over 2.25"), (
        "two unnameable sides compared equal, which would silently pool "
        "prices for bets the vocabulary cannot name")


def test_a_team_name_resolves_only_when_the_name_is_supplied():
    assert canonical("1X2", "Arsenal") is None
    assert canonical("1X2", "Arsenal", home_team_name="Arsenal") == (
        "1X2", "Home Win")
    assert canonical("1X2", "Arsenal", away_team_name="Arsenal") == (
        "1X2", "Away Win")


# ── THE EXPANSION HELPERS ARE BYTE-EQUIVALENT TO WHAT THEY REPLACED ─────────

def test_selection_aliases_preserves_the_fall_through_and_the_empty_filter():
    assert selection_aliases("Nothing Like This") == ["Nothing Like This"]
    got = selection_aliases("Home Win", "", "")
    assert HOME_TEAM not in got and "" not in got, (
        "an absent team name became an alias, which would match an empty "
        "`selection` column")
    assert got == ["1", "Home", "home_win", "Home Win"]
    assert selection_aliases("Home Win", "Arsenal", "Chelsea") == [
        "1", "Home", "home_win", "Home Win", "Arsenal"]


def test_market_aliases_preserves_the_fall_through():
    assert market_aliases("Nothing Like This") == ["Nothing Like This"]
    assert market_aliases("Under 3.5") == ["over_under", "totals"]
    assert market_aliases("1X2") == ["1X2", "h2h"]


def test_family_of_market_accepts_BOTH_vocabularies():
    assert family_of_market("Under 3.5") == "over_under"
    assert family_of_market("over_under") == "over_under"
    assert family_of_market("totals") == "over_under"
    assert family_of_market("h2h") == "1X2"
    assert family_of_market("unknown") is None
    assert family_of_market(None) is None


# ── THE THIRD TABLE MUST KEEP AGREEING ──────────────────────────────────────

def _selection_spec_by_ast():
    """`SELECTION_SPEC` read from source, NOT imported.

    `capture_closing_lines` is on the capture path; importing it here would
    make this file `capture_critical` and let a vocabulary test stop price
    capture.
    """
    tree = ast.parse(CLOSING.read_text(encoding="utf-8"))
    for n in ast.walk(tree):
        tgt = None
        if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            tgt = n.target.id
        elif isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name):
            tgt = n.targets[0].id
        if tgt == "SELECTION_SPEC":
            return {k.value: v.elts[0].value
                    for k, v in zip(n.value.keys, n.value.values)}
    raise AssertionError("UNEVALUATED: SELECTION_SPEC not found in "
                         f"{CLOSING.name}, so no agreement can be checked")


def test_SELECTION_SPEC_agrees_with_this_module_on_every_key():
    """Measured 2026-11-05: 22 keys, identical sets, same six families.

    `capture_closing_lines.SELECTION_SPEC` carries the line, side and leg
    index it needs besides the family, so it is not a duplicate of this table
    -- but the family relation IS shared, and a third spelling of it would be
    the split all over again. Consolidating it is registered, not done here:
    it is on the capture path.
    """
    spec = _selection_spec_by_ast()
    assert len(spec) == 22, len(spec)
    assert set(spec) == set(SELECTION_FAMILY), (
        f"only in SELECTION_SPEC: {sorted(set(spec) - set(SELECTION_FAMILY))}; "
        f"only here: {sorted(set(SELECTION_FAMILY) - set(spec))}")
    for name, market_type in spec.items():
        assert SELECTION_FAMILY[name] == market_type, (
            f"{name}: capture_closing_lines says {market_type!r}, this module "
            f"says {SELECTION_FAMILY[name]!r}")


def test_every_selection_alias_key_has_a_family():
    assert set(SELECTION_ALIASES) == set(SELECTION_FAMILY)
    assert set(SELECTION_FAMILY.values()) <= {
        a[0] for a in MARKET_ALIASES.values()}
