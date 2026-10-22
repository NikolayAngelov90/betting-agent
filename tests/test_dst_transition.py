"""Europe/Sofia goes UTC+3 -> UTC+2 on 2026-10-25 — 2026-10-22.

THE TRANSITION IS AT 01:00 UTC ON SUNDAY 2026-10-25 (the EU rule: last Sunday
of October). Every local time this project computes, displays or compares
shifts by an hour, and this is the sixth "establish the clock" instance in the
programme — the first that arrives on a schedule rather than by surprise.

TWO SITES ASSUMED AN OFFSET, with OPPOSITE literals, so each was correct for
half the year and nothing compared them:

    match_briefing._kyiv   timezone(timedelta(hours=3))   # "EEST fallback"
    telegram_bot           timezone(timedelta(hours=2))

and a third wrote the abbreviation itself into a prompt:

    f"KICKOFF (Bulgarian time, EEST): ..."

From 2026-10-25 that line asserts EEST while printing EET.

EVERY TEST BELOW IS DRIVEN AT 2026-10-24 **AND** 2026-10-26. A one-sided test
cannot tell a derived offset from a literal that happens to match today.
"""

import datetime as dt
import pathlib
import re
from zoneinfo import ZoneInfo

import pytest

TZ = ZoneInfo("Europe/Sofia")
KYIV = ZoneInfo("Europe/Kyiv")

#: The two sides of the transition. Chosen one day either side so the
#: transition hour itself is never what is being measured.
BEFORE = dt.date(2026, 10, 24)      # Saturday, UTC+3 (EEST)
AFTER = dt.date(2026, 10, 26)       # Monday,   UTC+2 (EET)


def _offset(day, zone=TZ):
    return int(dt.datetime.combine(day, dt.time(12), tzinfo=zone)
               .utcoffset().total_seconds() // 3600)


# ── the transition itself, from the zone database rather than from belief ──

def test_the_TRANSITION_DATE_and_DIRECTION_are_what_this_file_assumes():
    """If tzdata ever disagrees, every expectation below is void, so it is
    checked first rather than relied on."""
    assert _offset(BEFORE) == 3, "Europe/Sofia is not UTC+3 on 2026-10-24"
    assert _offset(AFTER) == 2, "Europe/Sofia is not UTC+2 on 2026-10-26"
    # and Kyiv moves with it, which is why the two declared zones in this
    # project have never differed by a minute
    assert _offset(BEFORE, KYIV) == 3 and _offset(AFTER, KYIV) == 2


def test_the_ABBREVIATION_changes_too_which_is_what_a_LITERAL_label_misses():
    before = dt.datetime.combine(BEFORE, dt.time(12), tzinfo=TZ).strftime("%Z")
    after = dt.datetime.combine(AFTER, dt.time(12), tzinfo=TZ).strftime("%Z")
    assert before == "EEST" and after == "EET", (before, after)


# ── the briefing header: derived offset AND derived abbreviation ──────────

def test_the_BRIEFING_local_time_shifts_by_an_HOUR_across_the_transition():
    """The same UTC instant, one hour apart in local terms. A literal +3
    fallback returns the same string on both sides, which is the bug."""
    from src.reporting.match_briefing import _operator_local

    before = _operator_local(dt.datetime.combine(BEFORE, dt.time(9, 30)))
    after = _operator_local(dt.datetime.combine(AFTER, dt.time(9, 30)))
    assert before == "12:30", before
    assert after == "11:30", after
    assert before != after, (
        "09:30 UTC formats identically on both sides of the transition, so "
        "the offset is a literal, not derived")


def test_the_BRIEFING_label_is_EEST_then_EET_and_is_NOT_HARD_CODED():
    from src.reporting.match_briefing import _operator_local

    assert _operator_local(
        dt.datetime.combine(BEFORE, dt.time(9, 30)), with_zone=True) \
        == "12:30 EEST"
    assert _operator_local(
        dt.datetime.combine(AFTER, dt.time(9, 30)), with_zone=True) \
        == "11:30 EET"

    # COMMENT- AND DOCSTRING-BLIND. The first version stripped only lines
    # beginning with `#` and matched this module's own DOCSTRING, which quotes
    # the literal it replaced. `conftest.code_of` tokenises, so a `#` inside a
    # string is not treated as a comment either.
    from tests.conftest import code_of
    code = code_of(pathlib.Path("src/reporting/match_briefing.py")
                   .read_text(encoding="utf-8"))
    assert "EEST" not in code, (
        "the abbreviation is written into the source again; it must come "
        "from %Z, which resolves to EET after 2026-10-25")
    # ANCHORED ON THE CLOSING PAREN. `hours=3` alone matched
    # `timedelta(hours=30)` — the unrelated 30-hour World Cup day window — so
    # the guard failed on code it has no opinion about.
    assert "hours=3)" not in code and "hours=2)" not in code, (
        "an assumed UTC offset is back in the briefing formatter")


def test_the_PROMPT_HEADER_names_the_zone_and_takes_the_abbreviation_from_it():
    from tests.conftest import code_of
    code = code_of(pathlib.Path("src/reporting/match_briefing.py")
                   .read_text(encoding="utf-8"))
    assert 'f"KICKOFF ({OPERATOR_TZ} local): ' in code, (
        "the prompt header no longer names the zone it is quoting")
    assert "with_zone=True" in code, (
        "the header stopped carrying the clock, so the time is a bare hour")
    assert "Bulgarian time, EEST" not in code


def test_the_FALLBACK_REFUSES_to_guess_and_says_UTC(monkeypatch):
    """A wrong clock that looks right is worse than a right clock that looks
    foreign. With the zone database unavailable it must report UTC, not a
    local-looking time that is not local."""
    import src.reporting.match_briefing as mb

    real_import = __import__

    def _no_zoneinfo(name, *a, **k):
        if name == "zoneinfo":
            raise ImportError("simulated: no tzdata")
        return real_import(name, *a, **k)

    monkeypatch.setattr("builtins.__import__", _no_zoneinfo)
    out = mb._operator_local(dt.datetime.combine(AFTER, dt.time(9, 30)))
    assert "09:30" in out, out
    assert "UTC" in out, f"the fallback printed a time with no clock: {out!r}"
    assert "unavailable" in out
    assert "11:30" not in out and "12:30" not in out, (
        "the fallback produced a LOCAL-looking time without a zone database")


# ── the Telegram card ────────────────────────────────────────────────────

def test_the_TELEGRAM_CARD_derives_its_offset_and_names_its_clock():
    from tests.conftest import code_of
    code = code_of(pathlib.Path("src/reporting/telegram_bot.py")
                   .read_text(encoding="utf-8"))
    assert "timedelta(hours=2)" not in code and "timedelta(hours=3)" not in code, (
        "an assumed offset is back on the picks card")
    assert '"Europe/Kyiv"' in code, "the current zone name is gone"
    assert "'%H:%M %Z'" in code, (
        "the card prints a bare hour again; a kickoff read in two countries "
        "must carry its clock")


@pytest.mark.parametrize("day,expect", [(BEFORE, "12:30 EEST"),
                                        (AFTER, "11:30 EET")])
def test_the_CARD_formats_the_same_instant_differently_on_each_side(day, expect):
    """Driven through the real zone rather than through the bot, which needs a
    token: the assertion is that the SAME UTC instant yields two different
    labelled strings, which a literal offset cannot do."""
    aware = dt.datetime.combine(day, dt.time(9, 30),
                                tzinfo=dt.timezone.utc).astimezone(KYIV)
    assert aware.strftime("%H:%M %Z") == expect


# ── the two helpers that were ALREADY correct, pinned so they stay so ────

@pytest.mark.parametrize("module", ["scripts.pipeline_loss_alert",
                                    "scripts.fixture_horizon_probe"])
def test_the_IN_TZ_helpers_were_ALREADY_DERIVING_and_still_are(module):
    """`_in_tz` returns `(HH:MM, label)` from ZoneInfo and `(None, reason)`
    when the zone is unavailable — it refuses rather than assuming. Both
    sides, so the label is proved to move."""
    import importlib

    m = importlib.import_module(module)
    hhmm_b, lab_b = m._in_tz(BEFORE, 9 * 60 + 30)
    hhmm_a, lab_a = m._in_tz(AFTER, 9 * 60 + 30)
    assert (hhmm_b, lab_b) == ("12:30", "EEST+0300"), (hhmm_b, lab_b)
    assert (hhmm_a, lab_a) == ("11:30", "EET+0200"), (hhmm_a, lab_a)
    assert m.OPERATOR_TZ == "Europe/Sofia"


# ── what the crons MEAN to Niki changes, though they do not move ─────────

#: `measured` 2026-10-22 from `.github/workflows/daily-picks.yml`.
CRONS_UTC = {"weekend": 0, "weekday": 0}


def test_the_CRONS_DO_NOT_MOVE_but_their_LOCAL_MEANING_SHIFTS_an_hour():
    """The crons are UTC and GitHub does not reinterpret them. What changes is
    what they mean to a person: a 00:00 UTC start is 03:00 local on 10-24 and
    02:00 local on 10-26. Nothing to fix — stated so the hour is not later
    mistaken for a regression in the scheduler."""
    wf = pathlib.Path(".github/workflows/daily-picks.yml").read_text(
        encoding="utf-8")
    found = sorted(set(re.findall(r"- cron: '(\d+) (\d+) \* \* ([^']+)'", wf)))
    assert found, "daily-picks carries no parseable cron"
    for minute, hour, _dow in found:
        utc_min = int(hour) * 60 + int(minute)
        assert utc_min == CRONS_UTC["weekday"], (
            f"the cron moved to {hour}:{minute} UTC; re-derive the local "
            f"meanings below before trusting them")
    for day, expect in ((BEFORE, "03:00"), (AFTER, "02:00")):
        local = dt.datetime.combine(
            day, dt.time(0, 0), tzinfo=dt.timezone.utc).astimezone(TZ)
        assert local.strftime("%H:%M") == expect, (day, local)


# ── THE DEADLINE CONSTANTS ARE SUMMER FIGURES, AND THAT IS MEASURED ──────

#: THE EARLIEST KICKOFF IS FIXED IN **LOCAL** TIME, so its UTC minute moves
#: with the clocks. `measured` 2026-10-22 over production `matches`, the 30
#: configured leagues, phantoms excluded (`microsecond == 0`, `hour >= 6`):
#:
#:   day type   offset   n     min UTC   min LOCAL
#:   weekday    UTC+3    421   09:30     12:30
#:   weekday    UTC+2    297   10:30     12:30     <- the same local time
#:   weekend    UTC+3    211   09:45     12:45
#:   weekend    UTC+2    164   10:00     12:00
#:
#: The weekday minimum is **12:30 local on both sides**. So
#: `OBSERVED_EARLIEST_MIN["weekday"] = 09:30` is a SUMMER figure: its anchor
#: is 2023-10-26, five days before that year's transition. From 2026-10-25 the
#: weekday earliest kickoff is 10:30 UTC and the constant is **60 minutes
#: pessimistic**; the weekend one is 15 minutes pessimistic.
EARLIEST_KICKOFF_BY_OFFSET = {
    ("weekday", 3): (570, 421), ("weekday", 2): (630, 297),
    ("weekend", 3): (585, 211), ("weekend", 2): (600, 164),
}


def test_the_WEEKDAY_MINIMUM_is_the_SAME_LOCAL_TIME_on_both_sides():
    """The claim the 60 minutes rests on, stated as arithmetic rather than as
    prose: 09:30 UTC at +3 and 10:30 UTC at +2 are both 12:30 local."""
    summer = EARLIEST_KICKOFF_BY_OFFSET[("weekday", 3)][0] + 3 * 60
    winter = EARLIEST_KICKOFF_BY_OFFSET[("weekday", 2)][0] + 2 * 60
    assert summer == winter == 12 * 60 + 30, (summer, winter)


def test_the_DEADLINE_FLOOR_still_holds_the_SUMMER_figure_and_its_ANCHOR():
    """Pinned, NOT changed. Moving the deadline is a Part C decision with a
    cohort bump behind it; this records that the constant is conservative by a
    measured hour so the next reader does not re-derive it from scratch."""
    import scripts.schedule_deadline as sd

    assert sd.OBSERVED_EARLIEST_MIN["weekday"] == 570      # 09:30 UTC
    assert sd.OBSERVED_EARLIEST_MIN["weekend"] == 585      # 09:45 UTC
    floor_min, _what, prov = sd.DEADLINE_FLOOR["weekday"]
    assert floor_min == 570
    assert "2023-10-26" in prov, (
        "the weekday floor's anchor date is no longer recorded; it is five "
        "days before that year's DST transition, which is why the figure is "
        "a summer one")
    winter_min = EARLIEST_KICKOFF_BY_OFFSET[("weekday", 2)][0]
    assert winter_min - floor_min == 60, (
        f"the measured winter minimum is {winter_min} against a floor of "
        f"{floor_min}; the 60-minute statement in the ledger is stale")
