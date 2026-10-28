"""H1's input precondition, its skip reasons and its silent zero — 2026-10-18.

    Independence of schedule is not independence of input. Two jobs that never
    collide can still have one starve the other when one writes the population
    the other samples.

THE MEASURED FAILURE. H1's job ran on 10-13, 10-14 and 10-15, **succeeded
every time**, spent **0 TheOddsAPI credits**, wrote **0 rows**, and reported
`state=RUNNING_AND_FINE alarms=False`. The window closed at **n=4 of 39**,
floor 33, power 0.073. Its only trace was one INFO line per run:

    TheOddsAPI imminent refresh: nothing to do (window 360 min) — 4 league(s) skipped

Three defects in that sentence, each fixed and controlled here:

  C1  `nothing to do` conflated an EMPTY CARD with a card carrying nothing
      USABLE. H1 is entirely downstream of `daily-picks --update`, which was
      dead 10-13 → 10-17, and nothing in H1 said so.
  C2  it logged a COUNT. By the time anyone asked why, `is_fixture` had been
      flipped true → false on all three days by the post-picks results
      scrape, and the reason was unrecoverable.
  C3  nobody saw it. H1 had no alarm for a day that collects nothing.

THE BAND, and it is not the window. With 3 rounds at 120-minute gaps the last
request fires at `T+240`, so a usable fixture sits in `(T+240, T+360]`. A
fixture at `T+60` is inside the 360-minute window and useless. Counting the
window overstates what is collectable, which is how `nothing to do` and
`nothing usable` became the same line.
"""

import datetime as dt

import pytest

import scripts.h1_precondition as pre


# ── C1: THE INPUT DEPENDENCY, MADE EXPLICIT ──────────────────────────────────

class _Sess:
    """A session whose four counts are dictated, so each state is reachable."""

    def __init__(self, total, as_fix, band, band_pinned, raise_on=None):
        self._v = [total, as_fix, band, band_pinned]
        self._i = 0
        self._raise = raise_on

    def execute(self, *_a, **_k):
        if self._raise:
            raise self._raise
        v = self._v[self._i]
        self._i += 1
        return type("R", (), {"scalar": lambda _s, _v=v: _v})()


def test_the_BAND_is_derived_from_the_ROUNDS_and_the_GAP_not_chosen():
    """240 is `(3-1) * 120`, not a number someone liked."""
    assert pre.BAND_LO_MIN == (pre.H1_ROUNDS - 1) * pre.H1_GAP_MINUTES == 240
    assert pre.BAND_HI_MIN == 360
    assert pre.BAND_LO_MIN < pre.BAND_HI_MIN


def test_NO_CARD_is_a_DISTINCT_terminal_state_with_the_count():
    """The actual failure of 10-13..10-15, and it must not read as
    `nothing to do`."""
    p = pre.evaluate(_Sess(12, 0, 0, 0), ["a/b"], now=dt.datetime(2026, 10, 13, 16, 36))
    assert p.state == pre.STATE_NO_CARD
    assert not p.can_collect
    assert "NO CARD" in p.reason
    assert "12 row(s)" in p.reason, "the count is missing from the reason"
    assert "daily-picks --update" in p.reason, (
        "the reason does not name what writes the population H1 samples")
    assert "NOT 'nothing to do'" in p.reason


def test_NO_FIXTURE_IN_BAND_is_SEPARATE_from_NO_CARD():
    """A full card with nothing 4–6 hours out is a different fact, and the one
    the live card showed on 2026-10-18: 68 fixtures, 9 in band, 0 pinned."""
    p = pre.evaluate(_Sess(68, 68, 9, 0), ["a/b", "c/d"],
                     now=dt.datetime(2026, 10, 18, 8, 0))
    assert p.state == pre.STATE_NO_FIXTURE_IN_BAND
    assert not p.can_collect
    assert p.card_as_fixture == 68 and p.in_band == 9 and p.in_band_pinned == 0
    assert "overstates" in p.reason, (
        "the reason does not say why the WINDOW count is misleading")


def test_OK_requires_a_PINNED_fixture_in_the_band():
    p = pre.evaluate(_Sess(68, 68, 9, 3), ["a/b"],
                     now=dt.datetime(2026, 10, 18, 8, 0))
    assert p.state == pre.STATE_READY and p.can_collect
    assert "3 pinned fixture(s)" in p.reason


def test_an_UNREADABLE_database_is_UNEVALUATED_not_clean():
    """Third state. `could not look` is not `found nothing` — the sixth time
    this file set has had to separate those."""
    p = pre.evaluate(_Sess(0, 0, 0, 0, raise_on=RuntimeError("boom")),
                     ["a/b"], now=dt.datetime(2026, 10, 18, 8, 0))
    assert p.state == pre.STATE_UNEVALUATED
    assert not p.can_collect
    assert "NOT a clean precondition" in p.reason


def test_the_states_are_ENUMERATED_so_the_set_is_read_not_inferred():
    assert set(pre.PRECONDITION_STATES) == {
        pre.STATE_READY, pre.STATE_NO_CARD, pre.STATE_NO_FIXTURE_IN_BAND,
        pre.STATE_UNEVALUATED}
    for s in pre.PRECONDITION_STATES:
        assert s == s.upper()


def test_the_precondition_SPENDS_NOTHING():
    """Its whole purpose is to distinguish 'the provider had nothing' from 'we
    had nothing to ask about' WITHOUT paying a credit to find out."""
    import inspect

    src = inspect.getsource(pre.evaluate)
    for spender in ("requests", "aiohttp", "fetch", "api_key", "claim_requests"):
        assert spender not in src, (
            f"the precondition touches {spender!r}; it must be read-only and "
            f"free, or it cannot be run before deciding to collect")


# ── C2: SKIPS LOG REASONS, NOT COUNTS. POSITIVE CONTROL PER REASON. ─────────

def test_POSITIVE_CONTROL_every_skip_reason_prints_DISTINGUISHABLY():
    """Force each one and confirm the texts differ.

    The old log printed `4 league(s) skipped` for all three causes, so they
    were indistinguishable by construction. A count with no reasons is
    auditable only until the state changes — and the state changed.
    """
    import pathlib

    src = pathlib.Path("src/scrapers/theodds_scraper.py").read_text(
        encoding="utf-8")
    reasons = ["not mapped to a The Odds API sport key",
               "no pending pick awaiting a closing line",
               "NO FIXTURE IN THE WINDOW"]
    for r in reasons:
        assert r in src, f"the skip reason {r!r} is gone"
    assert len(set(reasons)) == 3, "two reasons are textually identical"

    # and the log must emit the REASONS, not their length
    assert "{lg}={why}" in src, (
        "the skip log no longer joins league=reason pairs")
    assert "skipped: {detail}" in src, (
        "the log prints a count with no detail again")


def test_the_EMPTY_ROW_case_captures_the_state_AT_THE_DECISION():
    """The reason that did not exist, and the one that mattered.

    `is_fixture` is flipped true -> false by the post-picks results scrape, so
    a reason re-derived later reads a different world. These counts are taken
    at the instant of the decision for exactly that reason.
    """
    import pathlib

    src = pathlib.Path("src/scrapers/theodds_scraper.py").read_text(
        encoding="utf-8")
    block = src.split("NO FIXTURE IN THE WINDOW", 1)[1][:700]
    for part in ("across today+tomorrow", "of them", "is_fixture",
                 "future fixture(s) at any horizon"):
        assert part in block, f"the captured state omits {part!r}"
    assert "A zero here is the CARD, not the selection." in block


# ── C3: THE SILENT ZERO, ALARMED ────────────────────────────────────────────

def _day(d, fixtures, traj, state=pre.STATE_NO_CARD, reason="r"):
    return pre.ZeroDay(dt.date(2026, 10, d), fixtures, traj, state, reason)


def test_the_streak_counts_NO_CARD_days_because_not_having_one_IS_the_failure():
    rows = [_day(13, 8, 0), _day(14, 4, 0), _day(15, 1, 0)]
    assert len(pre.zero_collection_streak(rows)) == 3


def test_a_day_that_COLLECTED_ends_the_streak():
    rows = [_day(13, 8, 0), _day(14, 4, 2), _day(15, 1, 0)]
    assert [r.day.day for r in pre.zero_collection_streak(rows)] == [15]


def test_a_day_with_NO_FIXTURES_AT_ALL_and_no_NO_CARD_state_ends_it():
    """A day football does not play is not H1's failure."""
    rows = [_day(13, 0, 0, state=pre.STATE_NO_FIXTURE_IN_BAND),
            _day(14, 4, 0)]
    assert len(pre.zero_collection_streak(rows)) == 1


def test_POSITIVE_CONTROL_the_alarm_ESCALATES_and_N2_N3_DIFFER():
    """Same contract as `pipeline_loss_alert`, and N was fixed before the
    firing rate was known for the same reason."""
    bodies = [
        pre.format_zero_alarm([_day(12 + i, 8, 0) for i in range(n)],
                              39, 33, 4, 6, 168)
        for n in (1, 2, 3, 4)]
    heads = [b.splitlines()[0] for b in bodies]
    assert len(set(heads)) == 4, f"identical messages: {heads}"
    assert "COLLECTED NOTHING TODAY" in heads[0]
    assert "NOT SELF-CORRECTING" in heads[1]
    assert "WINDOW IS AT RISK" in heads[2]
    assert pre.ESCALATE_NOT_SELF_CORRECTING == 2
    assert pre.ESCALATE_WINDOW_AT_RISK == 3


def test_the_alarm_carries_n_the_FLOOR_the_CREDITS_and_the_IRREVERSIBILITY():
    body = pre.format_zero_alarm([_day(13, 8, 0), _day(14, 4, 0)],
                                 39, 33, 4, 6, 168,
                                 run_url="https://example.invalid/r")
    assert "n=4 of 39" in body and "floor 33" in body
    assert "6 spent of 168 reserved (TheOddsAPI)" in body, (
        "the credit figure does not name its provider")
    assert "never asked, not that it was refused" in body, (
        "a zero spend reads as a refusal unless it says otherwise")
    assert "NOT RECOVERABLE" in body.upper()
    assert "CLEARING ACTION" in body and "OPERATOR'S" in body
    assert "dispatch daily-picks first" in body, (
        "the NO_CARD remedy does not name the upstream job")
    assert "BAND density, not by card size" in body
    assert "https://example.invalid/r" in body


def test_the_clearing_action_DIFFERS_by_state():
    """One remedy for two different causes is the fixed-sentence defect."""
    body = pre.format_zero_alarm([_day(13, 8, 0)], 39, 33, 4, 6, 168)
    assert "if state=NO_CARD" in body
    assert "if NO_FIXTURE_IN_BAND" in body
    assert body.index("if state=NO_CARD") < body.index("if NO_FIXTURE_IN_BAND")


# ══════════ INPUT GUARDED — BOTH CONTROLS AND THE WIRING. 2026-10-28 ════════
#
# `h1_precondition.evaluate` existed, had four states and fifteen tests, and
# THE COLLECTION LOOP NEVER CALLED IT. The first attempt spent credits on days
# whose card was 8, 4 and 1 fixtures, skipped even those, and the REASON was
# never recorded. PNC-1: a producer with no consumer looks like a working
# feature, and its own passing tests are what made it look that way.

import datetime as _d2

import scripts.h1_collect as _h1c
import scripts.h1_precondition as _pre2


class _Sess2:
    """A session whose scalar answers are scripted, in the order `evaluate`
    asks: total rows, rows that are fixtures, in-band, in-band-and-pinned."""

    def __init__(self, *answers):
        self._a = list(answers)

    def execute(self, *_a, **_k):
        v = self._a.pop(0)
        return type("R", (), {"scalar": lambda self, v=v: v})()


def test_CONTROL_a_card_WITH_fixtures_in_band_reaches_READY():
    p = _pre2.evaluate(_Sess2(40, 40, 9, 4), ["england/premier-league"])
    assert p.state == _pre2.STATE_READY, p.reason
    assert p.can_collect is True
    assert "4 pinned fixture(s)" in p.reason


def test_CONTROL_a_card_with_NONE_in_band_reports_NO_FIXTURE_IN_BAND():
    """And it must SPEND NOTHING. "there is a card and nothing in it can
    carry three instants" is a different fact from "there is no card", and
    neither is "nothing to do"."""
    p = _pre2.evaluate(_Sess2(40, 40, 0, 0), ["england/premier-league"])
    assert p.state == _pre2.STATE_NO_FIXTURE_IN_BAND, p.reason
    assert p.can_collect is False
    assert "NO FIXTURE IN BAND" in p.reason
    # the counts that justify it are IN the reason, captured now
    assert "40 fixture(s) on the card" in p.reason


def test_the_FOUR_STATES_PRINT_DIFFERENTLY_and_nothing_to_do_is_not_one():
    reasons = {}
    for state, sess, lg in (
            (_pre2.STATE_NO_CARD, _Sess2(5, 0, 0, 0), ["x"]),
            (_pre2.STATE_NO_FIXTURE_IN_BAND, _Sess2(40, 40, 0, 0), ["x"]),
            (_pre2.STATE_READY, _Sess2(40, 40, 9, 4), ["x"])):
        p = _pre2.evaluate(sess, lg)
        assert p.state == state, (state, p.state, p.reason)
        reasons[state] = p.reason
    # UNEVALUATED: the question could not be asked
    class _Boom:
        def execute(self, *a, **k):
            raise RuntimeError("ledger unreachable")
    p = _pre2.evaluate(_Boom(), ["x"])
    assert p.state == _pre2.STATE_UNEVALUATED
    assert "NOT a clean precondition" in p.reason
    reasons[_pre2.STATE_UNEVALUATED] = p.reason

    assert set(reasons) == set(_pre2.PRECONDITION_STATES), set(reasons)
    assert len(set(reasons.values())) == 4, (
        "two states share a reason, so they are not distinguishable in a log")
    # ── THE PHRASE MAY APPEAR ONLY INSIDE ITS OWN DENIAL. ────────────────
    #
    # The first version asserted the phrase was absent, and NO_CARD's reason
    # ends "This is NOT 'nothing to do'." -- so the assertion matched the
    # DENIAL of the thing it was forbidding. Sixteenth instance of the
    # comment-blind class this week, in a test written the same hour as the
    # fourteenth and fifteenth.
    for state, r in reasons.items():
        low = r.lower()
        i = low.find("nothing to do")
        if i < 0:
            continue
        before = low[max(0, i - 24):i]
        assert "not" in before, (
            f"{state} uses the phrase 'nothing to do' as a CLAIM rather than "
            f"denying it: ...{r[max(0, i - 40):i + 20]}...")


def test_NOT_PINNED_MEANS_ALL_not_NONE():
    """`league = ANY(:lg)` with an empty list matches NOTHING.

    An unpinned run scored `band_pinned = 0` and reported NO_FIXTURE_IN_BAND
    on a card full of in-band fixtures, while the loop printed "pinned
    leagues: ALL (not pinned)" for the same case. The two disagreed about what
    an empty list meant. An empty selection is not an empty set.
    """
    p = _pre2.evaluate(_Sess2(40, 40, 9, 0), [])
    assert p.state == _pre2.STATE_READY, p.reason
    assert "ALL leagues (none pinned)" in p.reason
    # and with genuinely nothing in band it still refuses
    q = _pre2.evaluate(_Sess2(40, 40, 0, 0), [])
    assert q.state == _pre2.STATE_NO_FIXTURE_IN_BAND, q.reason
    assert "no league pinned so ALL were considered" in q.reason


def test_the_PRECONDITION_IS_CONSULTED_BEFORE_ANY_SPEND(monkeypatch, capsys,
                                                        tmp_path):
    """The wiring, driven. The loop's own tests stub this, so without this
    test a stub would stand in for a call that never happens — which is
    exactly the state `h1_precondition` was in until today.
    """
    import src.data.database as db_mod
    from src.data.models import Base

    # ── `main()` LOADS `.env`, SO IT IS STUBBED OUT HERE. ─────────────────
    #
    # `h1_collect._load_env` calls `load_dotenv()`, which re-introduces
    # DATABASE_URL after conftest strips it. Driving `main()` from a test
    # therefore poisons the WHOLE SESSION: this test passed while every
    # later DB fixture pointed at PRODUCTION, and the failure surfaced as
    # `assert not mgr.is_postgres` in `test_h1_collect_loop`'s fixture --
    # six errors in a file that had nothing to do with the change.
    #
    # The fixture guard caught it, which is why no production write happened.
    # The record says "an import-time load_dotenv reaches prod"; this is the
    # CALL-TIME version of the same thing, reached from a test rather than
    # from an import.
    monkeypatch.setattr(_h1c, "_load_env", lambda: None)

    mgr = db_mod.DatabaseManager(config=type("C", (), {
        "database": {"sqlite_path": str(tmp_path / "pre.db")}})())
    Base.metadata.create_all(mgr.engine)
    assert not mgr.is_postgres
    monkeypatch.setattr(db_mod, "get_db", lambda: mgr)

    called = {"n": 0}
    real = _pre2.evaluate

    def spy(session, leagues, *a, **k):
        called["n"] += 1
        return _pre2.Precondition(
            _pre2.STATE_NO_FIXTURE_IN_BAND, 9, 9, 0, 0,
            (_pre2.BAND_LO_MIN, _pre2.BAND_HI_MIN), tuple(leagues),
            "NO FIXTURE IN BAND: spied")
    monkeypatch.setattr(_pre2, "evaluate", spy)

    spent = {"claims": 0}
    monkeypatch.setattr(_h1c, "_one_round",
                        lambda *a, **k: spent.__setitem__("claims",
                                                          spent["claims"] + 1))
    rc = _h1c.main(["--max-credits", "6", "--rounds", "3",
                    "--gap-minutes", "120", "--allow-sqlite"])
    out = capsys.readouterr().out

    assert called["n"] == 1, (
        f"the precondition was consulted {called['n']} times; the loop must "
        f"ask it exactly once, BEFORE any round")
    assert spent["claims"] == 0, (
        "a round ran after the precondition refused — credits would have "
        "been claimed on a card with nothing in band")
    assert rc == 0
    assert "NOT COLLECTING" in out and "NO_FIXTURE_IN_BAND" in out
    assert "captured NOW" in out, (
        "the log does not say the reason was captured before the post-picks "
        "scrape can flip `is_fixture`")
    assert real is not _pre2.evaluate      # the spy really replaced it


# ══════ THE FIFA WINDOW AND THE CLEARANCES, PINNED — 2026-10-28 ═════════════
#
# These eight constants were written as a record and had NO CONSUMER, which
# the PNC scan reported (no_consumer 33 -> 41). A recorded measurement with
# nothing reading it is a comment with a type annotation; the natural consumer
# of a measurement is the test that pins it.

def test_the_FIFA_WINDOW_does_not_overlap_the_COLLECTION_WINDOW():
    """`FIFA WINDOW CHECKED`. The proxy's largest stated risk, closed.

    A published external fact is checked when it is KNOWABLE, not when it
    binds — so this was resolved on 2026-10-28 rather than deferred to the
    day's precondition. Source: en.wikipedia.org/wiki/
    FIFA_International_Match_Calendar, fetched 2026-10-28 ("9-17 November
    2026"), corroborated by beinsports.com and the CONCACAF Nations League
    League A quarter-finals.
    """
    from src.data.odds_quota import (H1_RESERVATION_FROM,
                                     H1_RESERVATION_UNTIL)

    w0 = _d2.date.fromisoformat(_pre2.FIFA_WINDOW_NOV_2026[0])
    w1 = _d2.date.fromisoformat(_pre2.FIFA_WINDOW_NOV_2026[1])
    assert w0 < w1, _pre2.FIFA_WINDOW_NOV_2026

    days = [H1_RESERVATION_FROM + _d2.timedelta(days=i)
            for i in range((H1_RESERVATION_UNTIL - H1_RESERVATION_FROM).days)]
    overlap = [d for d in days if w0 <= d <= w1]
    assert overlap == [], (
        f"the collection window now overlaps the FIFA international window on "
        f"{overlap}. The club card thins sharply inside it, so the window "
        f"must be RESCHEDULED before 2026-11-01 — a change of position, not "
        f"of length.")

    # THE MARGIN IS ONE DAY, AND THAT IS LUCK RECORDED AS LUCK. The window was
    # derived from the credit reset plus the pre-registered eight days; nobody
    # consulted the FIFA calendar when choosing it.
    margin = (w0 - days[-1]).days
    assert margin == 1, (
        f"the margin to the FIFA window is {margin} day(s), recorded as 1. If "
        f"it has grown, say why; if it has shrunk, the last collection days "
        f"are in the break.")
    assert "wikipedia" in _pre2.FIFA_WINDOW_SOURCE.lower(), (
        "the external source is no longer named")
    assert "lower confidence" in _pre2.FIFA_WINDOW_SOURCE, (
        "the 2025 dates' weaker provenance is no longer flagged")


def test_the_PROXY_WEEK_SITS_IN_THE_SAME_POSITION_RELATIVE_TO_THE_BREAK():
    """What makes the proxy comparable on this axis, and it is stronger than
    "same calendar position": both weeks are the eight club-football days
    IMMEDIATELY BEFORE a November international window.
    """
    p0 = _d2.date.fromisoformat(_pre2.PROXY_WEEK[0])
    p1 = _d2.date.fromisoformat(_pre2.PROXY_WEEK[1])
    b0 = _d2.date.fromisoformat(_pre2.FIFA_WINDOW_NOV_2025[0])
    proxy_days = [p0 + _d2.timedelta(days=i) for i in range((p1 - p0).days)]
    assert [d for d in proxy_days if d >= b0] == [], (
        "the proxy week overlaps the 2025 international window, so it was "
        "measuring a THINNED card and overstates nothing — it understates")
    assert (b0 - proxy_days[-1]).days == 2, (
        "the proxy's distance to its own break moved; the comparability "
        "argument rests on both weeks sitting just before one")


def test_the_START_TIME_and_BOTH_CLEARANCES_are_pinned():
    """`INPUT GUARDED`'s two numbers, with the binding end named.

    LATE END  the card must exist before T. +358 min (weekday, mean+3sd),
              from the new-cron regime n=6. The duration term is run END
              rather than time-to-picks, so it is conservative.
    EARLY END the band's last round against the earliest in-band kickoff.
              +60 min, from the proxy week.

    THE EARLY END BINDS. A fixture would have to be rescheduled an hour
    earlier to be lost, and moving T later buys nothing because the band moves
    with it.
    """
    assert _pre2.H1_START_MIN_UTC == 14 * 60, _pre2.H1_START_MIN_UTC
    assert _pre2.H1_CLEARANCE_LATE_MIN == 358
    assert _pre2.H1_CLEARANCE_EARLY_MIN == 60
    assert _pre2.H1_CLEARANCE_EARLY_MIN < _pre2.H1_CLEARANCE_LATE_MIN, (
        "the late end now binds; the derivation and the comments say the "
        "early end does, so one of them is stale")
    # the band is consistent with the start: the last round is T + BAND_LO
    assert _pre2.BAND_LO_MIN == (_pre2.H1_ROUNDS - 1) * _pre2.H1_GAP_MINUTES
    for prov in (_pre2.H1_START_PROVENANCE, _pre2.H1_OVERLAP_PROVENANCE):
        assert "measured 2026-10-28" in prov, prov[:60]
    assert "n=6" in _pre2.H1_START_PROVENANCE, (
        "the late-end figure no longer states its n")
    assert "series grain" in _pre2.H1_OVERLAP_PROVENANCE, (
        "the overlap finding no longer states the grain it was settled at — "
        "fixture-day grain said 67-103 instants and series grain said 1")
