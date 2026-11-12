"""The capture gate is OPT-IN, with its controls — 2026-10-28.

> **Four cards lost. Three of the four were clock-dependent tests that say
> nothing about capture correctness.**

`daily-picks`'s `Run tests` gated EVERY test and excluded two markers by hand
(`not calendar and not live`), so each new test gated price capture by DEFAULT
and had to be opted OUT. That default has been wrong four times:

    2026-10-09  the empty-card suppression's expiry, + three quota tests that
                inherited the H1 allocator's first active day      1 card
    2026-10-13  a rolling-window assertion over a live `gh run list`
                                                                   6 cards
    2026-10-21  a COUNT against `finding_register.evaluate`, whose predicate
                reads a path git does not track                    1 card
    2026-10-27  a frozen `NOW` fixture asserted against `mf.main()`, which
                reads the clock with no seam                       1 card

NINE CARDS, four events, and not one of the four tests says anything about
whether capture writes correct rows.

THE SET IS DERIVED, NOT LISTED. `scripts/capture_critical.py` holds the
criterion — a test is capture-critical if it imports a module on the path that
WRITES a captured row — and `tests/conftest.py` applies the marker at
collection. A hand list is the `LIVE_NAMES` shape: 16 seamless clock readers
in `scripts/` and 0 of them registered, because a list is a remedy someone
must remember.

AND IT DOES NOT CLOSE `UNCOVERABLE`'s CLOCK-READER GAP. "A reader nobody
registered is invisible" is still true and the enumeration stays. This makes
the gap HARMLESS rather than fixed: an unregistered clock reader can still
make a test flap, but unless that test is capture-critical the flap can no
longer reach `--update`. **Routed around, not closed** — and a
capture-critical test with an unregistered clock reader is exactly as
dangerous as it was before.
"""

import ast
import pathlib
import subprocess
import sys

import pytest

from scripts.capture_critical import (
    CAPTURE_PATH,
    GATING_EXPR,
    MARKER,
    NON_GATING_EXPR,
    census,
    classify,
)

#: `measured` 2026-10-28 by `scripts/capture_critical.py`.
#:
#: ── 69 -> 71, AND IT IS THE CRITERION'S KNOWN COST ───────────────────────
#:
#: `test_h1_analysis_path.py` and `test_h1_precondition.py` both build a
#: SYNTHETIC SQLite database, so both import `src.data.models` and
#: `src.data.database` and the criterion marks them capture-critical. The gate
#: grew 1,641 -> 1,668 for 27 tests about H1's input guard and analysis path,
#: neither of which says anything about whether capture writes correct rows.
#:
#: THAT IS OVER-SELECTION AND IT IS ACCEPTED RATHER THAN GAMED. The criterion
#: is import-based at FILE grain with one hop, and any test that stands up a
#: database satisfies it. Splitting files to dodge the import would make the
#: set a hand-curated list again -- the `LIVE_NAMES` shape -- and a list is
#: what this replaced.
#:
#: THE DIRECTION OF THE ERROR IS THE SAFE ONE: over-selection keeps a test in
#: the gate that need not be there, which costs gate time. Under-selection
#: lets a capture defect through, which costs a card. 27 tests of gate time
#: against nine cards is the trade, and it is the same trade stated in
#: `non-gating-tests.yml`.
#:
#: 131 -> 132 files and 62 -> 63 non-critical when THIS FILE was added, and
#: it is correctly non-critical: a test about the gate does not import the
#: write path, so it must not be able to stop capture. `capture_critical`
#: is unchanged at 69.
#: 134 -> 135 files and 72 -> 73 capture_critical on 2026-11-03, when
#: `test_resend_is_read_only.py` arrived. It is correctly CRITICAL: it
#: exercises the saved-pick read and the Telegram send path, and the thing it
#: proves is that `--resend` writes NO ROW TO ANY TABLE. A resend that wrote
#: would corrupt the pick record, so a regression there must be able to stop
#: capture rather than merely redden a later job.
#: 135 -> 137 files and 73 -> 75 capture_critical on 2026-11-03, with
#: `test_two_month_counter.py` and `test_five_mechanisms_fire.py`. Both are
#: correctly CRITICAL: one guards the credit counter that bounds spending, the
#: other proves the five H1 mechanisms actually fire. A regression in either
#: lets the subsystem spend or collect wrongly, so it must be able to stop
#: capture rather than merely redden a later job.
#: 137 -> 138 files and 62 -> 63 NOT-critical on 2026-11-04, with
#: `test_client_channel.py`. Correctly NON-critical: `telegram_bot` reports,
#: it does not write prices or picks, so a notification regression must not be
#: able to stop capture. The gate count is unchanged at 1710.
#: 138 -> 140 files and 75 -> 77 capture_critical on 2026-11-04, with
#: `test_code_of_is_version_independent.py` and
#: `test_dispatch_defaults_are_fundable.py`. Both correctly CRITICAL: the
#: first guards the helper whose version-sensitivity cost the 11-04 card, the
#: second guards the dispatch form that truncated the window's only in-band
#: day. A regression in either loses a card or a collection day.
#: 140 -> 141 files and 63 -> 64 NOT-critical on 2026-11-04, with
#: `test_client_chat_is_sealed.py`. Correctly NON-critical, like
#: `test_client_channel.py`: it guards WHICH CHAT a message reaches, which is
#: a notification concern. A regression there is serious -- operator traffic
#: in front of clients -- but it must not stop price capture.
#: 141 -> 142 files and 77 -> 78 capture_critical on 2026-11-04, with
#: `test_settlement_population_is_one.py`. Correctly CRITICAL: it seeds a
#: database and drives `experiment_record.build`, so it guards whether a
#: REPORTED population is the one that was COUNTED — data correctness.
#:
#: It was SPLIT OUT of `test_client_chat_is_sealed.py`, which stays
#: NON-critical. Adding a DB-backed helper there had pulled the whole file,
#: including the five send drives, into the gate: the
#: import-pulls-a-file-into-the-gate trap, third occurrence. A test about
#: WHICH CHAT a message reaches must not be able to stop price capture.
#: RE-MEASURED 2026-11-05: files 142 -> 143, not_critical 64 -> 65,
#: `capture_critical` UNCHANGED AT 78. The new file is
#: `test_unreadable_is_unevaluated.py`, and it is non-critical BY
#: CONSTRUCTION: its helper lives in `tests/gh_support.py`, which
#: imports the standard library only. Written into `conftest.py`
#: first, it pulled 25 tests into the gate -- its own 11 plus every
#: static test in `test_absent_run_and_unparseable.py`, which is ONE
#: OF THE FOUR LOSSES. `FOUR_LOSSES` caught it. Fourth occurrence of
#: the import-pulls-a-file-into-the-gate trap, and the first that a
#: guard rather than a card reported.
#: RE-MEASURED 2026-11-05 (vocabulary extraction): files 143 -> 144,
#: not_critical 65 -> 66, `capture_critical` UNCHANGED AT 78 for the
#: second time today. `test_market_vocabulary.py` is non-critical BY
#: CONSTRUCTION: `src.data.market_vocabulary` imports the standard
#: library only and is not on CAPTURE_PATH, and the file reads
#: `capture_closing_lines.SELECTION_SPEC` BY AST rather than importing
#: it, because that module IS on the capture path. Declining the fifth
#: opportunity for the import-pulls-a-file-into-the-gate trap.
#: RE-MEASURED 2026-11-06 (II), after the picks-bound substitution:
#: files 144 -> 146, not_critical 66 -> 68, `capture_critical`
#: UNCHANGED AT 78 for the THIRD time in two days. The two new files
#: are non-critical by construction:
#:   test_decision_record_is_not_a_live_constant.py -- the separation
#:     of the live bound from the s5.19 decision record, with the swap
#:     control. Deliberately does NOT import `tests.conftest`, because
#:     `code_of` would drag a decision-record test into the gate.
#:   test_two_consumers_read_one_input.py -- the standing cross-read
#:     check. It measures gate membership, so it must not be IN the
#:     gate it measures.
#: RE-MEASURED 2026-11-06 (III), market mix: files 146 -> 147,
#: not_critical 68 -> 69, `capture_critical` UNCHANGED AT 78 for the
#: FOURTH time in two days. `test_market_mix_part_a.py` is non-critical
#: because `scripts/market_mix.py` keeps every database and
#: `market_spec` import INSIDE its functions, so importing it reaches
#: nothing on the capture path.
CENSUS = {"files": 153, "capture_critical": 79, "not_critical": 74}

#: `measured` 2026-10-28 by pytest collection, node ids enumerated.
#: IMPORTED, NOT REBUILT. These were defined in `capture_critical.py` and
#: re-derived here, which the PNC scan reported as two producers with no
#: consumer -- correctly: two definitions of one selector drift, and the
#: selector is what the workflows run.
GATE_EXPR = GATING_EXPR
MARKERS_EXPR = "calendar or live"

#: THE FOUR LOSSES. `file` is the test module that stopped the gate; `gates`
#: is whether it would STILL be able to under the new direction.
FOUR_LOSSES = (
    ("2026-10-09", 1, "test_empty_card_suppression.py",
     "test_the_suppression_has_EXPIRED"),
    ("2026-10-13", 6, "test_absent_run_and_unparseable.py", None),
    ("2026-10-21", 1, "test_finding_register.py", None),
    ("2026-10-27", 1, "test_model_freshness.py", None),
)


def _ids(expr):
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "--collect-only",
         "-m", expr], capture_output=True, text=True, encoding="utf-8",
        errors="replace")
    return {l.strip() for l in r.stdout.splitlines()
            if "::" in l and l.strip().startswith("tests")}


# ════════════════════════════ THE CRITERION ══════════════════════════════

def test_the_CRITERION_is_the_ONLY_judgement_and_it_is_RECORDED():
    """`CAPTURE_PATH` is the judgement; everything else is derived from it."""
    assert len(CAPTURE_PATH) >= 8, CAPTURE_PATH
    for m in CAPTURE_PATH:
        assert m.startswith("src."), m
        p = pathlib.Path(m.replace(".", "/"))
        assert p.exists() or p.with_suffix(".py").exists(), (
            f"{m} is in CAPTURE_PATH and does not exist — the criterion is "
            f"selecting on a module that is gone")
    # the models, the features and the reporting layer are DELIBERATELY out
    for absent in ("src.models", "src.features", "src.reporting",
                   "src.evaluation"):
        assert absent not in CAPTURE_PATH, (
            f"{absent} is on the capture path now; a defect there produces a "
            f"wrong PICK, which is recoverable, and the gate exists for the "
            f"irrecoverable half")


def test_the_CENSUS_holds(capsys):
    c = census()
    for k, v in CENSUS.items():
        assert c[k] == v, (
            f"the census moved: {k} is {c[k]}, recorded {v}. Say what was "
            f"added and whether it belongs on the capture path.")


def test_the_MARKER_IS_APPLIED_AUTOMATICALLY_not_written_by_hand():
    """No test file may carry the marker in its own source.

    Writing it by hand would make the set a LIST again, and a list drifts from
    its criterion — which is the whole reason this is derived.
    """
    # THE ANCHOR IS ASSERTED INLINE, not through `tests.conftest.anchored`.
    # Importing conftest made THIS FILE capture-critical — conftest reaches
    # the write path through its database fixtures — so the test about the
    # gate landed inside the gate. A guard that can stop capture is exactly
    # what this change exists to avoid.
    offenders = []
    files = sorted(pathlib.Path("tests").glob("test_*.py"))
    assert files, "the tests/ glob found nothing, so this verifies nothing"
    for f in files:
        src = f.read_text(encoding="utf-8")
        assert "def test_" in src, (
            f"UNEVALUATED: {f.name} holds no test, so an absence claim about "
            f"its markers verifies nothing")
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for n in ast.walk(tree):
            if isinstance(n, ast.Attribute) and n.attr == MARKER:
                # this file NAMES the marker in prose and assertions; that is
                # not a `pytest.mark` application
                if f.name != "test_gate_is_opt_in.py":
                    offenders.append(f.name)
                break
    assert not offenders, (
        f"the marker is applied by hand in {offenders}; it is derived from "
        f"the import graph in tests/conftest.py and must stay derived")


def test_ONE_collection_hook_and_a_SECOND_IS_REFUSED():
    """Python keeps only the LAST `pytest_collection_modifyitems` in a module.

    THERE WERE THREE. The first version of the marker application was added as
    its own hook and NEVER RAN — `-m capture_critical` selected nothing and
    reported no error — because two were already defined and the later one
    shadowed both. The middle one had itself been dead since the last one
    absorbed its caplog logic, byte-identical and looking live.

        A shadowed hook is invisible, so it is refused rather than noticed.
    """
    src = pathlib.Path("tests/conftest.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    hooks = [n for n in tree.body
             if isinstance(n, ast.FunctionDef)
             and n.name == "pytest_collection_modifyitems"]
    assert len(hooks) == 1, (
        f"{len(hooks)} definitions of pytest_collection_modifyitems in "
        f"tests/conftest.py, at lines {[h.lineno for h in hooks]}. Python "
        f"keeps the LAST one and the others never run. Fold them into one.")
    body = ast.get_source_segment(src, hooks[0]) or ""
    assert "capture_critical" in body and "caplog" in body, (
        "the single hook has lost one of its two jobs")


# ═══════════════ THE FOUR KNOWN-WRONG CONTROLS, ALL FOUR ═════════════════

@pytest.mark.parametrize("when,cards,module,which", FOUR_LOSSES)
def test_KNOWN_WRONG_each_of_the_FOUR_LOSSES_CANNOT_GATE(when, cards, module,
                                                         which):
    """Each of the four tests that cost cards must fail to gate.

    `test_empty_card_suppression.py` IS capture-critical — it imports
    `src.scrapers.flashscore_scraper`, inside one test, at line 307 — so the
    file-level criterion selects it. The test that actually cost the card is
    `calendar`-marked, and the gate keeps `not calendar and not live` for
    exactly this reason: THE TWO MECHANISMS DO NOT COVER EACH OTHER. Drop the
    markers on the strength of the inversion and that card is lost again.
    """
    c = classify()
    critical = c[module]["critical"]
    gate = _ids(GATE_EXPR)
    in_gate = {i for i in gate if i.split("::")[0].endswith("/" + module)
               or i.startswith(f"tests/{module}::")}
    if which is None:
        assert not critical, (
            f"{module} ({when}, {cards} card(s)) is capture-critical via "
            f"{c[module]['via']} and would still be able to stop capture")
        assert not in_gate, f"{module} is still in the gating selection"
    else:
        # critical file, but the FAILING test must be out of the gate
        assert not any(i.endswith("::" + which) for i in in_gate), (
            f"{which} ({when}, {cards} card(s)) is still in the gating "
            f"selection; the marker exclusion no longer removes it")


# ═══════════════════════════ POSITIVE CONTROL ════════════════════════════

#: Tests that assert CAPTURE CORRECTNESS and must stay inside the gate. Each
#: was confirmed by PLANTING THE REAL DEFECT and running the gating selection
#: on 2026-10-28:
#:
#:   budget claim double-spends  (`allowed = n_requests`, ignoring the ledger)
#:       -> 6 failures inside the gate, including
#:          test_THE_REFUSAL_STILL_HOLDS_INSIDE_THE_NEW_WINDOW
#:   the two-way bet admitted as a 1X2 authority (the corruption that hit
#:   2,486 matches and seven books)
#:       -> 3 failures inside the gate, including
#:          test_1_home_away_must_never_populate_1x2
MUST_GATE = (
    ("test_market_separation.py", "test_1_home_away_must_never_populate_1x2"),
    ("test_market_separation.py",
     "test_every_mapped_bet_is_authoritative_for_its_market"),
    ("test_h1_reservation_is_an_allocation.py",
     "test_the_BLOCKER_H1_is_REFUSED_past_its_RESERVATION"),
)


@pytest.mark.parametrize("module,which", MUST_GATE)
def test_POSITIVE_CONTROL_a_real_CAPTURE_DEFECT_still_STOPS_CAPTURE(module,
                                                                    which):
    """If these leave the gate, the inversion went too far and it stops."""
    assert classify()[module]["critical"], (
        f"{module} is no longer capture-critical; a defect it catches would "
        f"no longer stop capture")
    gate = _ids(GATE_EXPR)
    assert any(i.endswith("::" + which) for i in gate), (
        f"{which} is not in the gating selection — a genuine capture defect "
        f"would no longer stop capture, which is the condition that stops "
        f"this change")


# ════════════════════════ THE PARTITION IS EXACT ═════════════════════════

def test_THE_THREE_SELECTIONS_PARTITION_THE_SUITE():
    """Every test runs exactly once, somewhere.

    A test in none of the three is a test that stopped running, which is the
    quiet way an inversion becomes a deletion. A test in two relays one
    verdict twice.
    """
    gate, nong, marks = _ids(GATE_EXPR), _ids(NON_GATING_EXPR), _ids(MARKERS_EXPR)
    everything = _ids("")
    assert not (gate & nong), sorted(gate & nong)[:5]
    assert not (gate & marks), sorted(gate & marks)[:5]
    assert not (nong & marks), sorted(nong & marks)[:5]
    missing = everything - (gate | nong | marks)
    assert not missing, (
        f"{len(missing)} test(s) are in NO selection and have stopped "
        f"running: {sorted(missing)[:5]}")


def test_THE_GATE_SHRANK_AND_THE_FIGURES_ARE_RECORDED():
    """`measured` 2026-10-28. The gate went 2,547 -> 1,641.

    906 tests left the gate, and 16 more arrived in the non-gating
    selection with this file, for 923.
    """
    # RE-MEASURED: gate 1680 -> 1696 -> 1710 (11-03) -> 1729 (11-04). Non-gating is UNCHANGED at
    # 911, and the first attempt at this update said 909 because a pytest
    # summary line reports tests PASSED while this assertion counts tests
    # COLLECTED -- 909 passed plus the 2 failures that prompted the update.
    # The two numbers coincide only on a green run, which is exactly when
    # nobody is reading them.
    #
    # +15 gating, all of them new tests rather than re-marked ones:
    #   11  test_resend_is_read_only.py -- the no-write row census, its
    #       injected-write positive control, the marker pair (resend does not
    #       move it, the ordinary send does) and the refusal with its
    #       known-wrong input
    #    5  test_h1_precondition.py -- the silent-zero alarm's controls: that
    #       a day WITH real trajectories reports them, that an out-of-window
    #       day is excluded by the window bound, and the streak pair (a zero
    #       day counts whatever its reason; a day that COLLECTED ends it).
    #       Net +1 against the test it replaced, which pinned the exemption
    #       that was truncating a real 3-day streak to 1.
    #   +14  the two-month counter's both-controls pair (7) and the five
    #        mechanisms' firings (7), added the same day
    #   +19 on 2026-11-04: 10 for `code_of`'s version-independence (the
    #       continuation-string known-wrong case, the two-forms equivalence,
    #       the four still-dropped docstring forms, comments, the workflow
    #       python pin and the recorded local-interpreter gap) and 9 for the
    #       dispatch form being mutually fundable.
    gate, nong = _ids(GATE_EXPR), _ids(NON_GATING_EXPR)
    # ── THE GATE GREW, FOR THE FIRST TIME SINCE IT WAS INVERTED. ─────────
    #
    #   +23 on 2026-11-11: test_band_on_legs.py, the leg-level FLB test. It
    #       is capture_critical `via src.data.market_spec`, which one of its
    #       known-wrong controls imports AT FUNCTION LEVEL -- and
    #       `_modules_imported` uses `ast.walk`, so a function-level import
    #       counts exactly as a module-level one does.
    #
    #       ACCEPTED RATHER THAN ENGINEERED AROUND, and the trade is stated:
    #       the control proves `_keys_for` mirrors `extract_legs`' own key
    #       construction for a REAL spec, and it caught a tie that mis-graded
    #       legs and moved the 70-79 gap from +1.66 to +1.43pp. A fabricated
    #       spec would have proved nothing. All 23 tests are STATIC -- they
    #       read constants, source text and the AST, and touch no database
    #       and no clock -- so none can lose the day card, which is the risk
    #       the gate exists to carry.
    #   +5 on 2026-11-12: the de-vigging CORRECTION, in the same file. The
    #       four bands reported PRESENT on 2026-11-11 do not survive
    #       subtraction of the artefact proportional de-vigging produces --
    #       `gap = M*(p-1/n)/(1+M)` is what a PERFECTLY CALIBRATED book shows
    #       — so these pin the formula, that not one band survives, that the
    #       70-80 artefact EXCEEDS the gap once reported as the finding, and
    #       that the superseded verdict is kept rather than quietly replaced.
    assert len(gate) == 1760, (
        f"the gating selection is {len(gate)}, recorded 1760 — say what moved")
    #   +7  test_client_channel.py: the sender census, the operator-channel
    #       routing pair, and the three known-wrong controls (failed run sends
    #       nothing / pick run sends the card / settlement sends the
    #       settlement) plus the population assertion
    #   +11 on 2026-11-04: test_client_chat_is_sealed.py -- the two
    #       destination-resolution checks, the AST seal and its known-wrong
    #       control, the two workflow-env assertions, and the FIVE DRIVES
    #       (ci-audit, reminders as verdict and as failure, a crash with no
    #       finding, a picks run, a settlement).
    #   +6 later the same day: the settlement's client-language controls
    #      (figure+period, the plain money statement, the known-wrong
    #      operator-vocabulary check) and the operator-channel trio (unset
    #      id annotates, the reintroduced fallback is caught, and no code
    #      path can resolve the client id).
    #   +11 on 2026-11-05: test_unreadable_is_unevaluated.py -- an unreadable
    #       run list is now a REPORTED UNEVALUATED rather than a
    #       `pytest.skip`, because a skip and a pass are the same thing in a
    #       green run, which is how a deleted PAT could leave a consumer
    #       broken AND green. The known-wrong control (a dead token must
    #       report, not skip, not pass), the positive control (a readable
    #       list still reaches a verdict), the empty and non-JSON shapes,
    #       three message-content checks, the source-shape check, the
    #       no-skip ratchet over both callers, and the two permissions
    #       assertions. GATE UNCHANGED at 1732 -- see CENSUS.
    #   +16 on 2026-11-05: test_market_vocabulary.py -- the market/selection
    #       vocabulary left `value_calculator`, where it lived in two
    #       AST-identical copies. The zero-copies scan WITH ITS ANCHOR
    #       ASSERTED and its planted-positive control, the known-wrong pair
    #       that matched 0 of 1,120 trajectories, the market-axis case
    #       (`team_goals` vs `Team Goals`), the twelve production pairings,
    #       the positive control over six genuinely-different bets, the
    #       `Home` collision across two families, the fixed-point check, four
    #       refusal cases, the two expansion helpers' fall-through and
    #       empty-filter behaviour, and the SELECTION_SPEC agreement.
    #       GATE UNCHANGED at 1732, verified as NODE IDS -- see CENSUS.
    #   +9 on 2026-11-06: controls added while clearing four reds, and the
    #       GATE and the CENSUS did not move at all (1732, 144/78/66) --
    #       every one landed outside the capture path.
    #         test_no_secrets_in_repo.py  +3  the known-wrong control the
    #           docs/memory coverage was missing (the memory file AS IT WAS,
    #           reconstructed with synthetic values, must be REFUSED), its
    #           synthetic-fixture check, and a clean-text control so a
    #           non-empty result means something
    #         test_two_margins.py         +3  the structural acknowledgement
    #           keyed by day type: the keying itself, the binding term named,
    #           the weekday/weekend conflation as a known-wrong control, and
    #           a missing-day-type refusal
    #         test_margin_acknowledged.py +3  the retirement, the vacuity
    #           guard (an empty register answers {} to everything, so every
    #           refusal would pass for the wrong reason), and the prepared
    #           re-derivation recorded with its blocker
    #   +14 on 2026-11-06 (II): the picks-bound substitution, FORWARD ONLY.
    #       GATE UNCHANGED at 1732 and the CENSUS unchanged at 78 critical --
    #       every one landed outside the capture path, and one was MOVED out
    #       of it (a live per-run-history read I had just written into
    #       `test_contamination_and_margin`, which would have been the FOURTH
    #       live-state-in-the-gate card loss).
    #         test_decision_record_is_not_a_live_constant.py  +6  both
    #           directions of the separation, the swap control, and a refusal
    #           to run at all if the two figures ever agree
    #         test_two_consumers_read_one_input.py            +7  the standing
    #           cross-read: the gate must hold zero live per-run-history
    #           readers, any selection holding one needs fetch-depth 0, and a
    #           stubbed caller must NOT be counted
    #         test_cron_split_by_daytype.py                   +1  the consumer
    #           of HEADROOM_NOT_SPENT, with the n and the projected date
    #   +9 on 2026-11-06 (III): test_market_mix_part_a.py. Part A recomputed
    #       on the whole 1,639-pick population (the 683-pick join could not
    #       reach 1X2 or over_under, 775 picks and the largest market).
    #       H-MIX is REFUTED AT PART A: no market is within 3.8x of the 1.5%
    #       cut, so the restricted population is empty. These pin the cut
    #       AGAINST BEING WIDENED to fit the result, which is the
    #       pre-registration's hardest line, with the widening itself as the
    #       known-wrong control; and they pin which figure moved when the
    #       population tripled (btts +1.48pp) and which held (double_chance
    #       -0.06pp). GATE UNCHANGED at 1732.
    #   +17 on 2026-11-06 (IV): test_cost_by_book.py. The per-book cost
    #       stage. These pin three things that fail silently: the UNIT (a
    #       band-specific margin converted at the population's average odds
    #       inverts the sign of the netting -- I made that error here, so the
    #       wrong conversion is kept as the known-wrong control); that the two
    #       CHEAPEST books are EXCHANGES whose commission is not in the
    #       quoted price, so 0.59pp must never become a planning cost; and
    #       that every cost figure has a COVERAGE figure beside it.
    #       GATE UNCHANGED at 1732: `scripts.cost_by_book` keeps every
    #       database and market_spec import INSIDE its functions, so importing
    #       it reaches nothing on the capture path. Verified through
    #       `classify()`, which returns critical=False via=[] for the file.
    #   +17 on 2026-11-06 (V): test_exchange_terms.py. The three unmeasured
    #       terms behind the exchange result. These pin that SIZE is
    #       UNAVAILABLE and that the consequence is UNESTABLISHED and NOT
    #       REFUTED -- only one of those ends the profitability question --
    #       and they DERIVE from `config/config.yaml` that the anomaly's
    #       70-79% band lies below our own 66.7% ceiling, so 0 of 1,639
    #       picks sit in it and none can while `betting.min_odds` is 1.50.
    #       THAT FLOOR MOVED TO 1.14 on 2026-11-06 (cohort s5.20) to open
    #       the band for measurement, so those assertions now record the
    #       s5.19 state as history and check the NEW floor -- see
    #       docs/flb-band-preregistration.md (fed8338).
    #       The config read is live, so moving the floor fails the test
    #       rather than silently invalidating the finding.
    #       GATE UNCHANGED at 1732, for the same reason as (IV).
    #   +2 on 2026-11-07: test_exchange_terms.py gained two tests when the
    #       floor it had pinned at 1.50 MOVED to 1.14 (cohort s5.20). Its
    #       disjointness assertion was written to fail loudly if that
    #       happened, and it did its job, so it is now SPLIT: one test keeps
    #       the s5.19 arithmetic as a dated historical finding, one reads the
    #       DEPLOYED config and checks the new floor still admits the whole
    #       de-vigged band (and is not the naive 1/0.79 bound). The second
    #       new test closes a stale literal the change exposed --
    #       `gate_registry` quoted the floor in prose and nothing checked it.
    #   +31 on 2026-11-07 (II): test_exposure_acknowledged.py and
    #       test_deployed_config_is_the_example.py -- 28 test FUNCTIONS but
    #       31 NODE IDS, because four of them are parametrised by day type.
    #       This selection counts node ids, and counting functions is how I
    #       first recorded 1070. The structural margin
    #       was RE-SPECIFIED against the EXPOSURE -- a frequency, 0 days over
    #       the registered minimum in every stratum across 1,015 evaluated
    #       days -- because the margin itself is negative by construction
    #       against a three-year-minimum floor and so cannot be acknowledged.
    #       The second file scans for a deployed config value read from the
    #       LOCAL file instead of the example, which is the file CI copies
    #       over it; one real offender was found and fixed.
    #       GATE UNCHANGED at 1732: both keep their database work inside
    #       functions, verified through `classify()` (critical=False, via=[]).
    #   +31 on 2026-11-12: test_sharp_anchor.py, Stage 30. The synthetic
    #       control the band test lacked is DRIVEN here, not read off pinned
    #       numbers: four de-vigging estimators against a perfectly
    #       calibrated book under four margin models, each exact only on the
    #       model it inverts. Also the seal (the outcome-free steps load
    #       WITHOUT the score columns, read from the AST), the registered
    #       verdict, and the residual harness's own controls -- zero features
    #       reproduces the market exactly, a planted feature is found, noise
    #       is not. GATE UNCHANGED at 1760: `src/evaluation/devig.py` is pure
    #       stdlib and both scripts import their database work in functions.
    assert len(nong) == 1104, (
        f"the non-gating selection is {len(nong)}, recorded 1104")
    # and the gate must never grow back to everything
    assert len(gate) < len(gate) + len(nong), "the inversion is undone"


def test_THE_WORKFLOWS_CARRY_THE_THREE_SELECTIONS():
    """The gate, its complement, and the markers — each in exactly one place.

    ONE MEANING PER RUN: `daily-picks` red = capture is at risk;
    `reminders` red = a decision is due; `non-gating-tests` red = a defect
    that does not touch capture. The 916 are NOT routed through `reminders`,
    whose classifier reads a pytest exit of 1 as "A REMINDER IS DUE" and
    prints a marker `ci-audit` treats as a deliberate red.
    """
    dp = pathlib.Path(".github/workflows/daily-picks.yml").read_text(
        encoding="utf-8")
    ng = pathlib.Path(".github/workflows/non-gating-tests.yml").read_text(
        encoding="utf-8")
    rm = pathlib.Path(".github/workflows/reminders.yml").read_text(
        encoding="utf-8")
    assert f'-m "{GATE_EXPR}"' in " ".join(dp.split()), (
        "daily-picks no longer runs the opt-in gate")
    assert f'-m "{NON_GATING_EXPR}"' in " ".join(ng.split()), (
        "non-gating-tests no longer runs the gate's complement")
    assert f'-m "{MARKERS_EXPR}"' in " ".join(rm.split()), (
        "reminders no longer runs the calendar/live selection")
    # and the non-gating selection must not ALSO run in reminders
    assert NON_GATING_EXPR not in " ".join(rm.split()), (
        "the non-gating tests are routed through reminders, whose exit 1 "
        "means A REMINDER IS DUE — every ordinary failure would certify that "
        "a date had arrived")


def test_THE_COST_IS_STATED_WHERE_THE_TRADE_IS_MADE():
    """Capture can now proceed on a tree whose non-critical tests fail.

    The classes that admits must be written down, with the reason the trade is
    the better one, so the first time it costs something that comment is what
    gets revisited rather than re-derived.
    """
    ng = pathlib.Path(".github/workflows/non-gating-tests.yml").read_text(
        encoding="utf-8").upper()
    for must in ("RECOVERABLE", "NEVER BACKFILLS", "WRONG PICK"):
        assert must in ng, f"the stated cost does not mention {must}"
    conf = pathlib.Path("tests/conftest.py").read_text(encoding="utf-8")
    assert "four times" in conf, (
        "the conftest no longer records that the old default was wrong four "
        "times — the measurement that justifies the inversion")


def test_THE_CLOCK_READER_GAP_IS_ROUTED_AROUND_AND_SAYS_SO():
    """Rule 1: say whether a limitation is closed or routed around.

    The 16 unregistered seamless clock readers are STILL unregistered. What
    changed is that a flap in a non-critical test can no longer reach
    `--update`. A capture-critical test with an unregistered clock reader is
    exactly as dangerous as before, and that is the residual.
    """
    doc = pathlib.Path("scripts/capture_critical.py").read_text(
        encoding="utf-8")
    assert "UNCOVERABLE" in doc and "HARMLESS rather than fixed" in doc, (
        "the module does not say whether the clock-reader gap is closed or "
        "routed around")
    import scripts.calendar_marker_scan as scan
    assert scan.SEAMLESS_READERS, (
        "the enumeration was deleted; the gap is routed around, not closed, "
        "so the record of it stays")


# ════════ TWO PINS FROM THIS WEEK — PART D, 2026-10-28 ══════════════════

def test_PIN_1_the_MARKED_SELECTION_IS_NON_EMPTY():
    """A selection returning zero items is `UNEVALUATED`, not an empty one.

    `-m capture_critical` selected NOTHING and reported no error, because
    Python kept only the last `pytest_collection_modifyitems` and the marker
    hook was the third defined. Everything looked fine: pytest printed
    "no tests collected", exit 0 on `--collect-only`, and the gate would have
    run an EMPTY suite and passed.

        A GATE THAT SELECTS NOTHING PASSES. That is the worst possible
        failure mode for a guard, and it is indistinguishable from a clean
        tree unless the count itself is asserted.
    """
    gate = _ids(GATE_EXPR)
    assert gate, (
        "the capture-critical selection is EMPTY. That is UNEVALUATED, not a "
        "clean gate: `Run tests` would run no tests and pass. Check that "
        "tests/conftest.py has exactly ONE "
        "pytest_collection_modifyitems and that it still applies the marker.")
    assert len(gate) > 1000, (
        f"the gating selection fell to {len(gate)}; 1,641 was measured "
        f"2026-10-28 and a collapse to a handful means the marker is being "
        f"applied to almost nothing")


def test_PIN_1_CONTROL_a_SECOND_HOOK_IS_DETECTED_AS_DEAD():
    """The known-wrong control: define two hooks, and the earlier must be
    identified as the dead one.

    This is the detection, run against a planted module rather than against
    `conftest.py` — a guard that can only check one file has not been shown to
    work.
    """
    import ast as _ast
    import textwrap

    def live_and_dead(src):
        """`(live_lineno, [dead_linenos])` for a module's hook definitions."""
        tree = _ast.parse(textwrap.dedent(src))
        hooks = [n for n in tree.body
                 if isinstance(n, _ast.FunctionDef)
                 and n.name == "pytest_collection_modifyitems"]
        if not hooks:
            return None, []
        return hooks[-1].lineno, [h.lineno for h in hooks[:-1]]

    planted = '''
        def pytest_collection_modifyitems(config, items):
            items.append("first")

        def pytest_collection_modifyitems(config, items):
            items.append("second")
    '''
    live, dead = live_and_dead(planted)
    assert dead, "two hooks were defined and neither was reported dead"
    assert len(dead) == 1 and dead[0] < live, (live, dead)

    # THE POSITIVE CONTROL: one hook is not reported dead
    live1, dead1 = live_and_dead('''
        def pytest_collection_modifyitems(config, items):
            items.append("only")
    ''')
    assert live1 and dead1 == [], (live1, dead1)

    # and no hook at all is distinguishable from one hook
    live0, dead0 = live_and_dead("x = 1\n")
    assert live0 is None and dead0 == [], (live0, dead0)

    # the live file has exactly one, by the same function. ANCHORED: an
    # emptiness claim about a haystack that is not the right one verifies
    # nothing, and `live_c is not None` does not count as an anchor because
    # it is checked AFTER the parse rather than establishing the input.
    # Asserted INLINE rather than through `tests.conftest.anchored`:
    # importing conftest reaches the write path through its database fixtures
    # and would make the test ABOUT the gate capture-critical, which is what
    # this change exists to avoid. Same reason as
    # `test_the_MARKER_IS_APPLIED_AUTOMATICALLY_not_written_by_hand`.
    conf = pathlib.Path("tests/conftest.py").read_text(encoding="utf-8")
    assert "def pytest_collection_modifyitems" in conf, (
        "UNEVALUATED: tests/conftest.py holds no collection hook at all, so "
        "an emptiness claim about its dead ones verifies nothing")
    live_c, dead_c = live_and_dead(conf)
    assert live_c is not None, "conftest has no collection hook at all"
    assert dead_c == [], (
        f"conftest.py has dead hook(s) at line(s) {dead_c}; Python keeps only "
        f"the one at line {live_c} and the others never run")


#: ── PIN 2 ────────────────────────────────────────────────────────────────
#:
#: > **Before optimising a duration term, establish that everything in it
#: > belongs there.**
#:
#: TWO INSTANCES, a week apart, and both were the same mistake: a term was
#: taken as given and then minimised, when the first question was whether its
#: contents belonged to it at all.
#:
#:   2026-10-20  THE MARGIN'S DURATION TERM measured run START to run END,
#:               so it charged the card for the POST-PICKS results scrape --
#:               mean 12.8 min, max 49.6 min, n=111. Correcting the term
#:               recovered 14.6 of 19.8 minutes WITH NO STEP GETTING FASTER.
#:               Work had been done to speed up a run that was already fast
#:               enough; the measurement was wrong, not the code.
#:
#:   2026-10-28  THE CAPTURE GATE ran 2,547 tests, of which 906 had nothing
#:               to do with whether capture writes correct rows. Removing
#:               them took the gate from 8m01s to 3m05s -- again with NO TEST
#:               GETTING FASTER. And the same term had already been
#:               "optimised" by parallelising the Claude review, which bought
#:               +30.5 minutes of margin by cutting spread.
#:
#: THE RULE IS CHEAPER THAN EITHER FIX. Both times the question "does
#: everything in this term belong here" was answerable in minutes from data
#: already on disk, and both times it returned a larger saving than the
#: optimisation it would have replaced.
DURATION_TERM_INSTANCES = (
    ("2026-10-20", "the margin's duration term charged the card for the "
                   "post-picks scrape (mean 12.8m, max 49.6m, n=111); "
                   "correcting it recovered 14.6 of 19.8 min with no step "
                   "getting faster"),
    ("2026-10-28", "the capture gate ran 906 tests unrelated to capture; "
                   "removing them took it from 8m01s to 3m05s with no test "
                   "getting faster"),
)


def test_PIN_2_the_DURATION_TERM_RULE_is_recorded_with_BOTH_instances():
    """A rule with one instance is an anecdote. This one has two, a week
    apart, and the second was found by applying it rather than by being
    bitten again."""
    assert len(DURATION_TERM_INSTANCES) == 2, DURATION_TERM_INSTANCES
    dates = [d for d, _ in DURATION_TERM_INSTANCES]
    assert dates == sorted(dates), "the instances are not in date order"
    for when, what in DURATION_TERM_INSTANCES:
        assert len(what) > 80, f"{when} is too vague to be an instance"
        assert "no step getting faster" in what or \
               "no test getting faster" in what, (
            f"{when} does not record the diagnostic property: the saving came "
            f"from the TERM being wrong, not from the work being slow")
    # the gate figure is the one this file can check for itself
    gate, nong = _ids(GATE_EXPR), _ids(NON_GATING_EXPR)
    assert len(nong) >= 900, (
        f"only {len(nong)} tests are outside the gate; the 2026-10-28 "
        f"instance rests on 906 having left it")
