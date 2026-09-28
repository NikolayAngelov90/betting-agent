"""The empty-card suppression, and the test that deletes it.

THE MEASUREMENT IT RESTS ON, 2026-09-27: `openfootball/football.json` 2026-27 —
a reference independent of Flashscore and API-Football — shows ZERO fixtures in
all eight covered leagues from 2026-09-21 through 2026-10-08. Every one of the
eight has its last fixture on 09-20 and its next on 10-09 or 10-10. The card
returns 10-09 with 7 fixtures and 41 on 10-10.

So "zero fixtures" is the EXPECTED state for eleven more days, and the discovery
alarms would otherwise fire daily with a known cause until they were tuned out —
which is how `fixtures_zero_active` became noise three times.

BUT SUPPRESSING THEM IS SUP-1 UNLESS IT EXPIRES. A guard that silences the alarms
about its own subject makes the first real collapse after 10-09 invisible. The
three requirements, each pinned below:

    1. the end date is a LITERAL, not a flag anyone must remember to unset
    2. `test_the_suppression_has_EXPIRED` FAILS from 10-09, so the suppression
       deletes itself rather than becoming permanent
    3. suppressed findings are still PRINTED, tagged with the grid cell that
       justifies each — suppressed is not unobserved

CLR-2: the action that clears the suppressed condition is THE CARD RETURNING,
which arrives by CALENDAR rather than by repair. **This is the first suppression
in this project cleared by time rather than by a fix** — and that is precisely
why it cannot be left to a judgement call.
"""

import datetime as dt
import importlib.util
import pathlib

import pytest

_spec = importlib.util.spec_from_file_location(
    "_ci_audit_empty", pathlib.Path("scripts/ci_audit.py"))
ci = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ci)

IN_WINDOW = dt.date(2026, 9, 27)
BEFORE = dt.date(2026, 9, 20)
AFTER = dt.date(2026, 10, 9)

COVERED = ["england/premier-league", "italy/serie-a"]
UNCOVERED = ["sweden/allsvenskan", "romania/liga-1"]

ZERO_HITS = [
    "NO FIXTURES FOUND for the day — nothing was analysed",
    "30 fixture scrape(s) attempted, 0 fixtures found in total",
    "Flashscore fixtures: 0 created AND 0 matched while other sources produce",
]
REAL_HITS = [
    "2 traceback(s) in the log",
    "1 message(s) LOST — a report part the retry did not recover",
    "core step(s) DID NOT RUN: update, picks (incl. review)",
]


# ── REQUIREMENT 2, FIRST, BECAUSE IT IS THE ONE THAT MATTERS ─────────────────

def test_the_suppression_has_EXPIRED():
    """FAILS FROM 2026-10-09. That is the point, not a bug.

    When this goes red the card has returned and the suppression must be
    DELETED — this test, `EMPTY_CARD_FROM`, `EMPTY_CARD_UNTIL`,
    `EMPTY_CARD_SUPPRESSIBLE`, `empty_card_window_active`,
    `partition_empty_card`, and its call site in `main()`.

    Do not extend the date to make this pass. Extending it is the decision the
    expiry exists to force someone to make deliberately, and it needs a fresh
    measurement from the reference, not a nudge.
    """
    today = dt.date.today()
    assert today < ci.EMPTY_CARD_UNTIL, (
        f"the empty-card suppression expired on {ci.EMPTY_CARD_UNTIL} and today "
        f"is {today}. The card has returned; DELETE the suppression rather than "
        f"moving the date. If the break was genuinely extended, re-measure it "
        f"against openfootball and record the new grid before changing anything.")


def test_the_end_date_is_a_LITERAL_not_a_config_flag():
    """A flag someone must unset is the failure mode, not the fix."""
    src = pathlib.Path("scripts/ci_audit.py").read_text(encoding="utf-8")
    assert "EMPTY_CARD_UNTIL = _dt.date(2026, 10, 9)" in src
    assert "config.get" not in src.split("EMPTY_CARD_FROM")[1][:400], (
        "the window reads from config — then it is a switch, and a switch left "
        "on is how SUP-1 happens")


def test_the_window_closes_by_DATE_and_not_by_anything_else():
    assert ci.empty_card_window_active(dt.date(2026, 10, 8)) is True
    assert ci.empty_card_window_active(dt.date(2026, 10, 9)) is False
    assert ci.empty_card_window_active(dt.date(2027, 1, 1)) is False


# ── REQUIREMENT 3: suppressed is not unobserved ──────────────────────────────

def test_suppressed_findings_are_RETURNED_not_dropped():
    kept, supp, _cen = ci.partition_empty_card(ZERO_HITS, IN_WINDOW, today=IN_WINDOW, zero_leagues=COVERED)
    assert kept == []
    assert len(supp) == len(ZERO_HITS), "a finding vanished instead of being tagged"


def test_each_suppressed_finding_CARRIES_ITS_JUSTIFICATION():
    _, supp, _cen = ci.partition_empty_card(ZERO_HITS, IN_WINDOW, today=IN_WINDOW, zero_leagues=COVERED)
    for s in supp:
        assert "[SUPPRESSED:" in s
        assert "openfootball 2026-27" in s, "no reference named"
        assert "2026-09-27" in s, "the run's own date is not in the justification"
        assert "card returns 2026-10-09" in s, "no expiry in the record"


def test_the_original_finding_text_SURVIVES_inside_the_tag():
    _, supp, _cen = ci.partition_empty_card(
        ["NO FIXTURES FOUND for the day — nothing was analysed"],
        IN_WINDOW, today=IN_WINDOW, zero_leagues=COVERED)
    assert supp[0].startswith("NO FIXTURES FOUND for the day"), (
        "the finding was rewritten rather than annotated — a reader cannot "
        "grep the ledger for the original text")


# ── THE SUPPRESSION IS NARROW, WHICH IS THE WHOLE SAFETY ARGUMENT ────────────

def test_a_REAL_failure_inside_the_window_is_NOT_suppressed():
    """An empty card explains zero fixtures. It explains nothing else.

    A traceback, a lost message or a skipped step is not excused by the
    calendar, and suppressing one would be the SUP-1 this design exists to
    avoid.
    """
    kept, supp, _cen = ci.partition_empty_card(REAL_HITS, IN_WINDOW, today=IN_WINDOW, zero_leagues=COVERED)
    assert kept == REAL_HITS, f"a real failure was suppressed: {supp}"
    assert supp == []


def test_a_mixed_run_keeps_the_real_and_tags_the_expected():
    kept, supp, _cen = ci.partition_empty_card(
        ZERO_HITS + REAL_HITS, IN_WINDOW, today=IN_WINDOW, zero_leagues=COVERED)
    assert kept == REAL_HITS
    assert len(supp) == len(ZERO_HITS)


# ── TWO CONDITIONS, BOTH REQUIRED ────────────────────────────────────────────

def test_a_run_BEFORE_the_window_is_not_suppressed():
    """09-20 was the last normal day, with 31 fixtures. Nothing to excuse."""
    kept, supp, _cen = ci.partition_empty_card(ZERO_HITS, BEFORE, today=IN_WINDOW, zero_leagues=COVERED)
    assert kept == ZERO_HITS and supp == []


def test_a_run_AFTER_the_window_is_not_suppressed():
    kept, supp, _cen = ci.partition_empty_card(ZERO_HITS, AFTER, today=IN_WINDOW, zero_leagues=COVERED)
    assert kept == ZERO_HITS and supp == []


def test_an_in_window_run_audited_AFTER_the_expiry_is_not_re_alarmed():
    """The second condition. Re-auditing 09-27 in November must not resurrect
    eleven days of alarms that were correct to suppress at the time — nor
    suppress them silently, which is why `empty_card_window_active` gates on
    TODAY and the window gates on the RUN's date. Both are required."""
    kept, supp, _cen = ci.partition_empty_card(
        ZERO_HITS, IN_WINDOW, today=dt.date(2026, 11, 1), zero_leagues=COVERED)
    assert kept == ZERO_HITS, "the expired suppression still applied"
    assert supp == []


def test_a_run_with_no_parseable_date_is_NOT_suppressed():
    """Fail closed. An unknown date cannot license a suppression."""
    kept, supp, _cen = ci.partition_empty_card(ZERO_HITS, None, today=IN_WINDOW, zero_leagues=COVERED)
    assert kept == ZERO_HITS and supp == []


# ── CONDITION 3: the suppression may not exceed its reference ────────────────
#
# ADDED THE SAME DAY AS THE OTHER TWO. The suppression's justification is a grid
# covering 8 of the 30 configured leagues, and it was being applied to all 30 —
# so for eleven days a genuine discovery failure in the other 22 would have
# recorded as "expected" when nothing established that it was.
#
# The residual it was hiding is large: on 09-19 the uncovered 22 produced 75 of
# 116 tracked fixtures, on 09-26 all 25, and on 09-27 all 5.

def test_a_zero_from_a_COVERED_league_is_suppressed():
    kept, supp, _cen = ci.partition_empty_card(
        ZERO_HITS, IN_WINDOW, today=IN_WINDOW, zero_leagues=COVERED)
    assert kept == [] and len(supp) == len(ZERO_HITS)


def test_a_zero_from_an_UNCOVERED_league_ALARMS():
    """THE POINT OF THE THIRD CONDITION.

    The reference says nothing about Allsvenskan. A zero there is unexplained,
    and suppressing it would be exactly the blind spot this condition removes.
    """
    kept, supp, _cen = ci.partition_empty_card(
        ZERO_HITS, IN_WINDOW, today=IN_WINDOW, zero_leagues=UNCOVERED)
    assert kept == ZERO_HITS, "an unreferenced zero was suppressed"
    assert supp == []


def test_ONE_uncovered_league_is_enough_to_suppress_NOTHING():
    """Fail closed on the mixture. Seven covered zeros plus one uncovered is
    not seven-eighths explained — the finding is run-level and cannot be split,
    so the whole of it alarms."""
    kept, supp, _cen = ci.partition_empty_card(
        ZERO_HITS, IN_WINDOW, today=IN_WINDOW,
        zero_leagues=COVERED + ["poland/ekstraklasa"])
    assert kept == ZERO_HITS and supp == []


def test_the_covered_set_is_a_LITERAL_not_a_config_read():
    """The same pin `EMPTY_CARD_UNTIL` carries, for the same reason."""
    src = pathlib.Path("scripts/ci_audit.py").read_text(encoding="utf-8")
    block = src.split("EMPTY_CARD_COVERED_LEAGUES")[1][:600]
    assert "frozenset({" in block
    assert "config.get" not in block, (
        "the covered set reads from config — then it is a switch, and the "
        "suppression's scope becomes editable without a new measurement")
    assert ci.EMPTY_CARD_COVERED_LEAGUES == frozenset({
        "england/premier-league", "england/championship", "spain/laliga",
        "germany/bundesliga", "italy/serie-a", "france/ligue-1",
        "netherlands/eredivisie", "portugal/primeira-liga"})
    assert len(ci.EMPTY_CARD_COVERED_LEAGUES) == 8, (
        "the covered set changed size without the grid being re-measured")


def test_no_leagues_determined_suppresses_NOTHING():
    """`None` and `[]` both fail closed: with no league named, nothing is
    attributable to the reference."""
    for z in (None, []):
        kept, supp, _cen = ci.partition_empty_card(
            ZERO_HITS, IN_WINDOW, today=IN_WINDOW, zero_leagues=z)
        assert kept == ZERO_HITS and supp == []


def test_the_extractor_finds_the_zero_leagues_in_a_real_log_shape():
    """Read from the log, because the findings name no league.

    Both phrasings the scraper emits, plus the `Scraped 0` line, which is the
    one that appeared 29 times on 09-27.
    """
    log = ("x\n" * 200
           + "Scraped 0 fixtures from england/premier-league\n"
           + "Flashscore: italy/serie-a has no fixtures within the requested window\n"
           + "Flashscore returned 0 fixtures for sweden/allsvenskan - the page\n"
           + "Scraped 5 fixtures from spain/laliga2\n")
    got = ci.extract(log)["zero_fixture_leagues"]
    assert got == ["england/premier-league", "italy/serie-a",
                   "sweden/allsvenskan"], got
    assert "spain/laliga2" not in got, "a league that DID produce was counted as zero"


# ── THE FINDING IS LEAGUE-SCOPED, AND SO IS CONDITION 3 ──────────────────────
#
# BEFORE 2026-09-27 the discovery-zero findings were run-level, and a run-level
# finding CANNOT BE PARTIALLY CLEARED: one unreferenced league among 29 left the
# whole finding unexplained, so the suppression fired nowhere. That was a
# property of the FINDING, not of the reference.
#
# `FS_DISCOVERY` carries the league, so a finding built from it is cleared on its
# own evidence: a covered league's zero is suppressed, an uncovered league's zero
# alarms, and a mixed day is partially explained.

import src.scrapers.flashscore_scraper as _fs

CUT = dt.datetime(2026, 9, 28, 8, 52)
FAR = dt.datetime(2026, 10, 10, 14, 0)


def _fs_line(league, rows=100, earliest=FAR, kept=0, off=False):
    return _fs.discovery_outcome(league=league, rows=rows, earliest=earliest,
                                 cutoff=CUT, kept=kept, off_season=off)[1]


def _hits_from(*lines):
    log = "x\n" * 200 + "\n".join(lines) + "\n"
    f = ci.extract(log)
    return f, ci.assertions(f, [])


# ── POSITIVE CONTROL FOR THE EMITTER, not just the extractor ─────────────────

def test_the_emitter_produces_a_finding_for_a_league_at_ZERO():
    """A test for absence must first prove it can observe presence.

    The extractor already had this test; the emitter needs its own, because a
    line that is never emitted and a league that is never at zero produce the
    same empty finding list.
    """
    _, hits = _hits_from(_fs_line("austria/bundesliga"))
    disc = [h for h in hits if h.startswith("discovery: ")]
    assert len(disc) == 1, hits
    assert "austria/bundesliga found 0 fixtures" in disc[0]
    assert "state=none-in-range" in disc[0]


def test_the_emitter_produces_NO_finding_for_spain_laliga2_which_produced_5():
    """THE MIRROR. `laliga2` delivered 5 fixtures on 09-27 and must be silent."""
    _, hits = _hits_from(_fs_line("spain/laliga2", kept=5))
    assert not [h for h in hits if h.startswith("discovery: ")], hits


def test_an_OFF_SEASON_league_produces_no_finding():
    """The config calls it dormant; that is expected, not a discovery zero."""
    _, hits = _hits_from(_fs_line("world/fifa-world-cup", rows=0, off=True))
    assert not [h for h in hits if h.startswith("discovery: ")]


def test_NO_ROWS_and_NONE_IN_RANGE_are_different_states_in_the_finding():
    """Identical zeros in the tracked count, nothing else in common.

    `no-rows` means the page yielded nothing at all; `none-in-range` means it
    yielded rows whose kickoffs all lay beyond the cutoff. Collapsing them is
    what made the residual unreadable.
    """
    _, hits = _hits_from(_fs_line("a/b", rows=None, earliest=None),
                         _fs_line("c/d", rows=90, earliest=FAR))
    disc = sorted(h for h in hits if h.startswith("discovery: "))
    assert "state=no-rows" in disc[0] and "earliest_parsed=None" in disc[0]
    assert "state=none-in-range" in disc[1]
    assert "earliest_parsed=2026-10-10" in disc[1]


def test_an_unknown_row_count_FAILS_CLOSED_into_no_rows():
    """`rows is None` cannot claim the league is merely quiet."""
    assert "state=no-rows" in _fs_line("a/b", rows=None, earliest=None)


# ── CONDITION 3 PER FINDING: a mixed day is PARTIALLY explained ──────────────

def test_a_mixed_day_is_PARTIALLY_explained():
    """THE POINT OF THE GRANULARITY CHANGE.

    Under the run-level finding, one uncovered league left everything
    unexplained. Now the covered league's zero clears and the uncovered one
    does not.
    """
    f, hits = _hits_from(_fs_line("italy/serie-a"),
                         _fs_line("sweden/allsvenskan"))
    kept, supp, _cen = ci.partition_empty_card(
        hits, IN_WINDOW, today=IN_WINDOW,
        zero_leagues=f.get("zero_fixture_leagues"))
    assert len(supp) == 1 and "italy/serie-a" in supp[0]
    assert len([h for h in kept if h.startswith("discovery: ")]) == 1
    assert any("sweden/allsvenskan" in h for h in kept)


def test_the_09_27_REPLAY_yields_8_suppressed_and_21_alarmed():
    """The number the directive named, on the real league set of 2026-09-27.

    29 leagues reported zero, 8 of them covered by the reference. If this is
    anything other than 8 and 21, the extractor and the emitter disagree.
    """
    covered = sorted(ci.EMPTY_CARD_COVERED_LEAGUES)
    uncovered = [f"unc{i}/league" for i in range(21)]
    lines = [_fs_line(lg) for lg in covered + uncovered]
    f, hits = _hits_from(*lines)
    kept, supp, _cen = ci.partition_empty_card(
        hits, IN_WINDOW, today=IN_WINDOW,
        zero_leagues=f.get("zero_fixture_leagues"))
    assert len(supp) == 8, f"expected 8 suppressed, got {len(supp)}"
    assert len([h for h in kept if h.startswith("discovery: ")]) == 21


def test_a_league_scoped_finding_is_NOT_cleared_by_the_run_level_rule():
    """Even with every zero-reporting league covered, an uncovered league's own
    finding must alarm. The two rules must not leak into each other."""
    f, hits = _hits_from(_fs_line("sweden/allsvenskan"))
    kept, supp, _cen = ci.partition_empty_card(
        hits, IN_WINDOW, today=IN_WINDOW,
        zero_leagues=sorted(ci.EMPTY_CARD_COVERED_LEAGUES))
    assert supp == []
    assert any("sweden/allsvenskan" in h for h in kept)


def test_ONE_definition_only__no_fourth_phrasing():
    """One definition, and a guard. The three collapsed to one in this stage."""
    src = pathlib.Path("src/scrapers/flashscore_scraper.py").read_text(encoding="utf-8")
    code = "\n".join(l for l in src.splitlines() if not l.strip().startswith("#"))
    assert code.count("FS_DISCOVERY") == 1, "more than one emitter"
    for gone in ("has no fixtures within the requested",
                 "returned 0 fixtures for"):
        assert gone not in code, f"an old phrasing survived: {gone}"


# ── VAC-1: THE CENSUS. A zero from a filter must say WHICH zero it is ────────
#
# The defect this closes, measured 2026-09-28 on production run 36307004104:
# the suppression printed no `~` lines and was reported as "fires nowhere". The
# same output is produced by a filter that evaluated 29 findings and cleared
# none. Only reading `assertions()` and the `disc` field separated them, which
# means the report could not answer its own question.
#
# Every test below asserts the three cases are DISTINGUISHABLE, not that any
# particular one occurred.

def _census(hits, run_date, today=IN_WINDOW, zero_leagues=COVERED):
    return ci.partition_empty_card(
        hits, run_date, today=today, zero_leagues=zero_leagues)[2]


def test_examined_nothing_and_cleared_nothing_are_DIFFERENT_CENSUSES():
    """THE VAC-1 PROPERTY, stated directly.

    Handed nothing vs handed real findings and clearing none: `suppressed` is 0
    in both, so `suppressed` alone cannot be the report.
    """
    handed_nothing = _census([], IN_WINDOW)
    handed_real = _census(REAL_HITS, IN_WINDOW)

    assert handed_nothing.suppressed == handed_real.suppressed == 0
    assert handed_nothing.examined == 0
    assert handed_real.examined == len(REAL_HITS)
    assert str(handed_nothing) != str(handed_real), (
        "the two cases print identically — VAC-1 is not closed")


def test_the_census_counts_CANDIDATES_separately_from_examined():
    """`examined` alone still cannot say the filter had anything to work on.

    The 09-27 production case exactly: two findings examined, NEITHER of them
    suppressible, so the filter's zero says nothing about the calendar.
    """
    c = _census(REAL_HITS, IN_WINDOW)
    assert c.examined == 3
    assert c.candidates == 0, "a traceback is not a suppression candidate"
    assert c.alarmed == 3

    c2 = _census(ZERO_HITS, IN_WINDOW)
    assert c2.candidates == len(ZERO_HITS)
    assert c2.suppressed == len(ZERO_HITS)
    assert c2.alarmed == 0


def test_a_filter_that_NEVER_ENGAGED_names_the_reason():
    """Out of window, expired, and unparseable date are three separate states.

    All three previously returned the same `[]` as an engaged filter.
    """
    assert _census(ZERO_HITS, BEFORE).reason == "window"
    assert _census(ZERO_HITS, None).reason == "run-date-unparseable"
    assert _census(ZERO_HITS, IN_WINDOW, today=AFTER).reason == "expired"
    for r in (BEFORE, None):
        assert _census(ZERO_HITS, r).examined == 0, (
            "a short-circuit reported findings as examined")


def test_an_ENGAGED_filter_carries_no_not_engaged_reason():
    """The positive control for the reason field: it must be empty when the
    filter really did run, or `NOT-ENGAGED` becomes decoration."""
    c = _census(ZERO_HITS, IN_WINDOW)
    assert c.reason == ""
    assert "NOT-ENGAGED" not in str(c)
    assert c.examined == len(ZERO_HITS)


def test_engaged_but_handed_nothing_is_its_OWN_state():
    """`no-findings` is not `window`. The run had nothing to say, which is
    different from the suppression declining to look."""
    c = _census([], IN_WINDOW)
    assert c.reason == "no-findings"
    assert c.examined == 0 and c.candidates == 0


def test_alarmed_plus_suppressed_ACCOUNTS_for_every_examined_finding():
    """No finding may be counted twice or lost between the two buckets."""
    for hits in ([], REAL_HITS, ZERO_HITS, REAL_HITS + ZERO_HITS):
        c = _census(hits, IN_WINDOW)
        assert c.suppressed + c.alarmed == c.examined, (
            f"census does not balance for {hits!r}: {c}")
        assert c.candidates <= c.examined


def test_the_census_is_PRINTED_unconditionally_at_the_call_site():
    """A line that appears only when something was suppressed is a line whose
    absence means nothing — the AF_LEAGUE_FILTER lesson, applied here."""
    src = pathlib.Path("scripts/ci_audit.py").read_text(encoding="utf-8")
    body = src.split("for h in _suppressed:")[1][:400]
    assert "print(f\"{'':<56} {_census}\")" in body, (
        "the census is not printed after the suppressed block")
    assert "if _census" not in body and "if _suppressed" not in body, (
        "the census print is gated — then its absence is ambiguous again")
