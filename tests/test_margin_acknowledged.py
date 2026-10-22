"""The standing weekday margin breach, acknowledged — 2026-10-22.

THE FINDING IT COVERS: the weekday schedule margin is `+21m` against a
30-minute threshold, it was true on 30 of 30 sampled alarming runs, and no
action available to the operator clears it on the day. CLR-2 says that is a
thing to REPORT, not to alarm on.

WHAT THIS FILE REFUSES TO LET THE ACKNOWLEDGEMENT BECOME:

  * a silencer -- the acknowledged path must say everything the alarm said,
    including WHY the reading is conservative, which a truncation at 88
    characters had been cutting off;
  * a figure read off a display -- `+21` is `20.665799601890512`;
  * permanent -- the review date is compared, not merely stored;
  * level-triggered -- it alarms when the margin DETERIORATES past a floor,
    and the floor's distance from the reading is the series' own MEASURED
    single-point jitter, not a guess.
"""

import datetime as dt
import pathlib

import pytest

import scripts.ci_audit as ci

ACK = ci.ACKNOWLEDGED_MARGIN_BREACH["weekday"]
SRC = pathlib.Path("scripts/ci_audit.py").read_text(encoding="utf-8")


# ── the figure ─────────────────────────────────────────────────────────────

def test_the_READING_is_stored_at_FULL_PRECISION_not_as_displayed():
    """`+21` is not the number. An acknowledgement recorded at a rounded
    figure absorbs a breach it was never shown -- demonstrated on 2026-10-21,
    when `19.0` correctly refused `18.89232917906861`."""
    assert ACK["measured_at"] == 20.665799601890512
    assert f"{ACK['measured_at']:+.0f}" == "+21", (
        "the stored reading no longer displays as the figure in the record")
    assert ACK["measured_at"] != round(ACK["measured_at"]), (
        "the reading was rounded on its way into the table")


def test_the_FLOOR_is_BELOW_the_reading_by_the_MEASURED_jitter():
    """The floor is derived, not chosen. 11.32 is the largest ORDINARY
    single-point move of the binding, measured by recomputing it over growing
    prefixes of the weekday series (+5.53, +2.79, -11.32, -2.24, +1.77;
    -97.91 is the cluster boundary and is excluded)."""
    assert ACK["jitter"] == 11.32 == ci.ACKNOWLEDGED_MARGIN_BREACH_JITTER
    assert ACK["binding"] == pytest.approx(ACK["measured_at"] - 11.32)
    assert ACK["binding"] < ACK["measured_at"], (
        "a floor at or above the reading re-alarms on the next scheduled run, "
        "which is how the daily red comes back")
    assert ACK["binding"] < ci.MARGIN_ALARM_MIN, "the floor must be a breach"


def test_the_JITTER_is_DERIVED_IN_THE_SOURCE_with_its_measurements():
    """A band whose width is a bare literal is a threshold hard-coded from a
    small sample. The five moves it comes from have to be written down."""
    i = SRC.index("ACKNOWLEDGED_MARGIN_BREACH_JITTER = ")
    head = SRC[max(0, i - 2600):i]
    for move in ("+5.53", "+2.79", "-11.32", "-2.24", "+1.77", "-97.91"):
        assert move in head, (
            f"the single-point move {move} is no longer recorded beside the "
            f"band derived from it")
    assert "cluster boundary" in head or "cluster" in head, (
        "the excluded move must be excluded FOR A STATED REASON")


# ── deterioration, not level: DRIVEN ──────────────────────────────────────

def test_DRIVEN_it_does_NOT_alarm_at_the_acknowledged_level():
    """The level is a breach and is covered."""
    assert ci.margin_breach_is_acknowledged("weekday", ACK["measured_at"])
    assert ci.margin_breach_is_acknowledged("weekday", ACK["binding"]), (
        "the floor itself is not covered, so the comparison is not >=")


def test_DRIVEN_it_DOES_alarm_on_a_DETERIORATION_past_the_floor():
    worse = ACK["binding"] - 0.01
    assert ci.margin_breach_is_acknowledged("weekday", worse) == {}, (
        f"a binding of {worse} is worse than the acknowledged floor "
        f"{ACK['binding']} and was still absorbed")


def test_DRIVEN_an_IMPROVEMENT_is_covered_too():
    """Covering better-than-acknowledged is the point: the acknowledgement is
    a floor on how bad it may get, not a window."""
    assert ci.margin_breach_is_acknowledged("weekday", 29.0)


def test_DRIVEN_the_EXPIRY_is_COMPARED_and_not_merely_STORED(monkeypatch):
    """A deferral with a date nothing reads is not a deferral. The previous
    entry's review date passed on 2026-10-18 and it kept absorbing."""
    assert ACK["review"] == "2026-11-04"

    real_date = ci._dt.date

    class _After(real_date):
        @classmethod
        def today(cls):
            return real_date(2026, 11, 5)

    monkeypatch.setattr(ci._dt, "date", _After)
    assert ci.margin_breach_is_acknowledged("weekday", ACK["measured_at"]) == {}, (
        "the acknowledgement outlived its own review date")


def test_the_REVIEW_DATE_is_the_one_the_CLEARING_ACTION_lands_on():
    """2026-11-04 is not a round number: it is the projected date the
    parallel-regime series reaches n=10, which is the clearing action."""
    assert dt.date.fromisoformat(ACK["review"]) > dt.date(2026, 10, 22)
    assert "n=10" in ACK["action"] or "n>=10" in ACK["action"] \
        or "reaching n=10" in ACK["action"], ACK["action"]
    assert "FULL_CARD_PICKS_MAX" in ACK["action"], (
        "the durable clearing action no longer names the constant it "
        "re-derives")


# ── the acknowledged path must not report LESS than the alarm ─────────────

def test_the_WHY_carries_the_CONSERVATISM_and_is_NOT_TRUNCATED():
    """It was printed as `_ack['why'][:88]`, which stopped mid-sentence and
    cut the conservatism statement off entirely."""
    assert len(ACK["why"]) > 88, "the why is short enough that this is moot"
    assert "[:88]" not in SRC, "the truncation is back"
    for frag in ("CONSERVATIVE", "SERIAL", "88.1", "35.1", "BIMODAL"):
        assert frag in ACK["why"], f"the why stopped saying {frag!r}"


def test_the_BIMODALITY_is_stated_WITH_BOTH_CLUSTERS_and_the_SAME_CRON():
    """`mean+3sd` over a bimodal series measures between-cluster spread, not
    tail risk. The claim is only checkable if both clusters are written down,
    and it only means anything if the cron did NOT change between them."""
    why = ACK["why"]
    assert "255.9-304.9" in why and "128.4-144.1" in why, why
    assert "same cron" in why, (
        "without this, a reader cannot tell the bimodality from an ordinary "
        "regime change -- and a regime change would justify SPLITTING the "
        "series instead of qualifying it")


def test_the_ACKNOWLEDGED_BRANCH_prints_the_READING_the_FLOOR_and_the_CLOCK():
    """Statically, because driving it needs a live `gh`. Each of the four is
    something the alarm used to carry and the first draft of the
    acknowledgement dropped."""
    i = SRC.index("ACKNOWLEDGED since {_ack['since']}")
    block = SRC[i:i + 1600]
    assert "reading now {_margin['binding']:+.2f}" in block, (
        "the CURRENT reading is not printed, so the headroom is invisible")
    assert "FLOOR of {_ack['binding']:+.2f}" in block, (
        "the floor is printed without being called a floor")
    assert "_ack['jitter']" in block, "the band's width is not shown"
    assert "{_ack['why']}" in block, "the why is not printed in full"
    assert "{_ack['action']}" in block, "the clearing action is not printed"
    assert "MIN_N_TO_REDERIVE_PICKS_BOUND" in block, (
        "the regime clock is not printed beside the acknowledgement, so "
        "'until when' has to be remembered")


# ── KNOWN-WRONG CONTROLS ──────────────────────────────────────────────────

def test_KNOWN_WRONG_the_WEEKEND_margin_is_NOT_acknowledged():
    """The weekend binding was +135.7 on 2026-10-22 and clears the threshold
    outright. Acknowledging a day type that does not breach would be a
    standing absorber waiting for a real breach to arrive under it."""
    assert "weekend" not in ci.ACKNOWLEDGED_MARGIN_BREACH
    assert ci.margin_breach_is_acknowledged("weekend", 5.0) == {}


def test_KNOWN_WRONG_an_UNPARSEABLE_review_date_absorbs_NOTHING(monkeypatch):
    saved = ci.ACKNOWLEDGED_MARGIN_BREACH
    ci.ACKNOWLEDGED_MARGIN_BREACH = {
        "weekday": dict(ACK, review="not-a-date")}
    try:
        assert ci.margin_breach_is_acknowledged(
            "weekday", ACK["measured_at"]) == {}
    finally:
        ci.ACKNOWLEDGED_MARGIN_BREACH = saved


def test_KNOWN_WRONG_a_day_type_with_no_entry_is_not_covered():
    assert ci.margin_breach_is_acknowledged("", -500.0) == {}
    assert ci.margin_breach_is_acknowledged("holiday", -500.0) == {}
