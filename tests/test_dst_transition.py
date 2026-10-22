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

#: `measured` 2026-10-22 from `.github/workflows/daily-picks.yml`, AFTER the
#: `s5.19` move. They were both 0 when this test was written that morning.
CRONS_UTC = {"weekend": 30, "weekday": 60}


def test_the_CRONS_DO_NOT_MOVE_but_their_LOCAL_MEANING_SHIFTS_an_hour():
    """The crons are UTC and GitHub does not reinterpret them. What changes is
    what they mean to a person. Nothing to fix — stated so the hour is not
    later mistaken for a regression in the scheduler.

    RE-POINTED 2026-10-22 when the crons moved; the PROPERTY is unchanged and
    is now checked per day type, since the two no longer share a minute.
    """
    wf = pathlib.Path(".github/workflows/daily-picks.yml").read_text(
        encoding="utf-8")
    found = {}
    for minute, hour, dow in re.findall(
            r"- cron: '(\d+) (\d+) \* \* ([^']+)'", wf):
        key = "weekend" if ("6" in dow or "0" in dow.split(",")) else "weekday"
        found[key] = int(hour) * 60 + int(minute)
    assert found == CRONS_UTC, (
        f"the crons are {found}; re-derive the local meanings below before "
        f"trusting them")
    #: What each cron MEANS locally, either side of the transition. Derived
    #: from ZoneInfo, not written down twice.
    for key, utc_min in CRONS_UTC.items():
        summer = dt.datetime.combine(
            BEFORE, dt.time(utc_min // 60, utc_min % 60),
            tzinfo=dt.timezone.utc).astimezone(TZ)
        winter = dt.datetime.combine(
            AFTER, dt.time(utc_min // 60, utc_min % 60),
            tzinfo=dt.timezone.utc).astimezone(TZ)
        assert (winter.hour * 60 + winter.minute) ==             (summer.hour * 60 + summer.minute) - 60, (
            f"{key}: the local meaning did not shift by exactly an hour "
            f"({summer:%H:%M} -> {winter:%H:%M})")
    # and the concrete pair, so the shift is legible rather than only asserted
    assert dt.datetime.combine(BEFORE, dt.time(1, 0),
                               tzinfo=dt.timezone.utc)        .astimezone(TZ).strftime("%H:%M") == "04:00"
    assert dt.datetime.combine(AFTER, dt.time(1, 0),
                               tzinfo=dt.timezone.utc)        .astimezone(TZ).strftime("%H:%M") == "03:00"


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
#: The weekday minimum is **12:30 local on both sides**, so the retired
#: `09:30 UTC` was a SUMMER figure -- its anchor is 2023-10-26, three days
#: before that year's transition -- and from 2026-10-25 the weekday earliest
#: kickoff is 10:30 UTC.
#:
#: THE WEEKEND ROWS ARE PER-OFFSET SUBSET MINIMA AND THE ANCHOR IS NOT.
#: Corrected 2026-10-22: the localised anchor is the minimum over the WHOLE
#: record, which for the weekend is **12:00 local** (2024-11-02, a winter
#: day) -- not the 12:45 the UTC+3 subset shows. So the retired weekend
#: constant was 45 minutes too LATE in summer, and localising it TIGHTENS the
#: summer floor while loosening the winter one by 15. A correction, both ways.
#: See `test_the_DEADLINE_FLOOR_IS_NOW_LOCALISED_and_the_RETIRED_PAIR_is_kept`.
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


def test_the_DEADLINE_FLOOR_IS_NOW_LOCALISED_and_the_RETIRED_PAIR_is_kept():
    """REPOINTED 2026-10-22, the day after it was written.

    It asserted that the floor "still holds the SUMMER figure", pinned rather
    than changed because moving a deadline needed its own stage. That stage
    happened: the anchors are LOCAL minutes converted per day, and the retired
    UTC pair is kept beside them so the correction stays checkable.
    """
    import scripts.schedule_deadline as sd

    # the anchors are LOCAL now
    assert sd.OBSERVED_EARLIEST_LOCAL_MIN["weekday"] == 12 * 60 + 30
    assert sd.OBSERVED_EARLIEST_LOCAL_MIN["weekend"] == 12 * 60 + 0
    # and the retired pair survives, BOTH of them summer figures -- which IS
    # the defect, so deleting them would delete the evidence
    assert sd.RETIRED_OBSERVED_EARLIEST_UTC_MIN == {"weekday": 570,
                                                    "weekend": 585}

    # BOTH SIDES, through the shipped function
    for day, wd, we in ((BEFORE, 9 * 60 + 30, 9 * 60 + 0),
                        (AFTER, 10 * 60 + 30, 10 * 60 + 0)):
        assert sd.observed_earliest_min("weekday", day) == wd, day
        assert sd.observed_earliest_min("weekend", day) == we, day
        fl, _what, prov = sd.deadline_floor_for("weekday", day)
        assert fl == wd
        assert "2023-10-26" in prov, (
            "the weekday anchor's date is no longer recorded; it is three "
            "days before that year's DST transition, which is why the "
            "retired UTC figure was a summer one")

    # THE WEEKDAY CORRECTION IS +60 IN WINTER AND ZERO IN SUMMER
    retired_wd = sd.RETIRED_OBSERVED_EARLIEST_UTC_MIN["weekday"]
    assert sd.observed_earliest_min("weekday", BEFORE) - retired_wd == 0
    assert sd.observed_earliest_min("weekday", AFTER) - retired_wd == 60

    # AND THE WEEKEND CORRECTION RUNS THE OTHER WAY IN SUMMER: -45. The
    # retired 09:45 was the minimum of the SUMMER SUBSET only; the true local
    # minimum is 12:00 and it comes from a winter day (2024-11-02). A
    # localisation is a correction, not a gift, and this half of it TIGHTENS.
    retired_we = sd.RETIRED_OBSERVED_EARLIEST_UTC_MIN["weekend"]
    assert sd.observed_earliest_min("weekend", BEFORE) - retired_we == -45
    assert sd.observed_earliest_min("weekend", AFTER) - retired_we == 15
    assert "TIGHTENS it by" in pathlib.Path(
        "scripts/schedule_deadline.py").read_text(encoding="utf-8"), (
        "the tightening is no longer recorded where the floor is defined")


def test_the_WEEKEND_floor_is_now_the_EARLIER_one_on_BOTH_sides():
    """The ordering inverted, and permanently rather than seasonally.

    The retired UTC pair had weekday 09:30 < weekend 09:45. In LOCAL time the
    weekend minimum is 12:00 against the weekday's 12:30, so the weekend binds
    whichever season it is. The old ordering was an artefact of two summer
    figures measured on different days.

    The MARGIN still fires on the weekday, because the weekday DELAY is worse.
    Which deadline is earlier and which margin is tighter are different
    questions.
    """
    import scripts.schedule_deadline as sd

    for day in (BEFORE, AFTER):
        assert sd.observed_earliest_min("weekend", day) < \
            sd.observed_earliest_min("weekday", day), day


def test_the_OFFSET_HELPER_RAISES_rather_than_guessing(monkeypatch):
    """A deadline computed from an assumed offset is an hour wrong for half
    the year, silently. `operator_offset_min` has no fallback on purpose."""
    import scripts.schedule_deadline as sd

    real_import = __import__

    def _no_zoneinfo(name, *a, **k):
        if name == "zoneinfo":
            raise ImportError("simulated: no tzdata")
        return real_import(name, *a, **k)

    monkeypatch.setattr("builtins.__import__", _no_zoneinfo)
    with pytest.raises(ImportError):
        sd.operator_offset_min(AFTER)
