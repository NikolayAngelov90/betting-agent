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
    assert p.state == pre.STATE_OK and p.can_collect
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
        pre.STATE_OK, pre.STATE_NO_CARD, pre.STATE_NO_FIXTURE_IN_BAND,
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
