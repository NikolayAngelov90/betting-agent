"""The margin on a DISPATCHED start. Stage 34, 2026-11-21.

The card is published by an external dispatch at 10:30 Europe/Sofia. The
margin is measured from the dispatched run's ACTUAL START: its delay after
10:30 local is the series, the dispatch minute is the start, and the
deadlines are the same local anchors. Hermetic: every GitHub read is the
recorded `gh run list` shape, injected through the reader's `raw` seam.
"""
import datetime as dt
import json

import pytest

import scripts.ci_audit as ci
from scripts import schedule_deadline as sd

UTC = dt.timezone.utc


def _run(rid, iso, event="workflow_dispatch"):
    return {"databaseId": rid, "createdAt": iso, "startedAt": iso,
            "event": event}


# ── the dispatch minute and the delay ────────────────────────────────────────

def test_the_dispatch_minute_follows_DST_because_the_job_runs_in_Sofia():
    assert sd.dispatch_utc_min(dt.date(2026, 12, 15)) == 8 * 60 + 30   # EET
    assert sd.dispatch_utc_min(dt.date(2027, 6, 15)) == 7 * 60 + 30    # EEST


@pytest.mark.parametrize("iso, want", [
    ("2026-11-22T08:32:00Z", 2.0),       # 10:32 local, winter
    ("2027-06-15T07:31:30Z", 1.5),       # 10:31:30 local, summer
    ("2026-11-22T08:29:00Z", None),      # BEFORE 10:30: not the dispatch
    ("2026-11-22T11:31:00Z", None),      # 13:31 local: past the window
])
def test_the_DELAY_is_minutes_after_THAT_DAYS_10_30_local(iso, want):
    t = dt.datetime.fromisoformat(iso.replace("Z", "+00:00"))
    got = sd.dispatch_delay_min(t)
    assert got == (pytest.approx(want) if want is not None else None)


# ── the reader, on the recorded shape ────────────────────────────────────────

def test_the_READER_keeps_THE_dispatch_and_COUNTS_the_manual_ones():
    raw = json.dumps([
        _run(1, "2026-11-22T08:33:00Z"),                   # the dispatch
        _run(2, "2026-11-22T15:00:00Z"),                   # manual, excluded
        _run(3, "2026-11-20T08:31:00Z"),                   # before SINCE
        _run(4, "2026-11-23T08:31:00Z", event="schedule"),  # not a dispatch
        _run(5, "2026-11-23T08:30:45Z"),                   # the dispatch
    ])
    out = ci.collect_dispatch_series(raw=raw)
    assert out["unreadable"] is False
    assert [(str(d), round(x, 2), r) for d, x, r in out["series"]] == [
        ("2026-11-22", 3.0, 1), ("2026-11-23", 0.75, 5)]
    assert [r for _t, r in out["excluded"]] == [2], (
        "a manual dispatch was dropped silently instead of counted")


@pytest.mark.parametrize("raw", ["not json", '{"a": 1}'])
def test_an_UNREADABLE_listing_is_UNREADABLE_not_an_empty_series(raw):
    out = ci.collect_dispatch_series(raw=raw)
    assert out["unreadable"] is True and out["series"] == []


# ── the margin, driven on a dispatched start ─────────────────────────────────

def test_the_MARGIN_is_computed_from_the_DISPATCHED_start():
    """One run 2 min after 10:30: weekday 12:30 - (10:30 + 2 + 64.59) =
    +53.4 against the audit's 30; weekend 12:00 - ... = +23.4."""
    day = dt.date(2026, 11, 23)                    # a Monday, EET
    disp = {"series": [(dt.date(2026, 11, 22), 2.0, 1)], "excluded": [],
            "unreadable": False}
    by_type, crons = ci.schedule_inputs_under_dispatch(disp, day)
    assert crons == {"weekday": 510, "weekend": 510}
    assert by_type["weekday"] == by_type["weekend"] == [
        (dt.date(2026, 11, 22), 2.0)], "the one scheduler's delays are POOLED"
    m = ci.schedule_margins_by_daytype(by_type, crons, day=day)
    assert m["weekday"]["margin_3sd"] == pytest.approx(
        120 - 2.0 - ci.FULL_CARD_PICKS_3SD, abs=0.01)
    assert m["weekend"]["margin_3sd"] == pytest.approx(
        90 - 2.0 - ci.FULL_CARD_PICKS_3SD, abs=0.01)


def test_KNOWN_WRONG_the_START_MINUTE_omission_is_CAUGHT():
    """The weekday-margin function left the start minute out from
    2026-10-22: harmless under `0 0` (minute 0), 60 minutes high under
    `0 1`, ~550 high under the dispatch. Its figure must now move one for one
    with the start."""
    lag = [2.0, 3.0]
    at_dispatch = ci.serial_term_weekday_margin(lag, 510)
    at_midnight = ci.serial_term_weekday_margin(lag, 0)
    assert at_midnight - at_dispatch == pytest.approx(510)
    assert at_dispatch < 120, "the start minute is not being subtracted"


# ── the weekend cost, acknowledged against its EXPOSURE ──────────────────────

SUNDAY, MONDAY = dt.date(2026, 11, 22), dt.date(2026, 11, 23)


def test_the_WEEKEND_is_ACKNOWLEDGED_while_picks_are_ready_before_12_00():
    ack = sd.dispatch_exposure_ack(SUNDAY, 2.0, ci.FULL_CARD_PICKS_3SD, None)
    assert ack["state"] == "ACKNOWLEDGED", ack
    assert "operator's decision" in ack["why"]
    assert "0 of 257 weekend days" in ack["why"]


@pytest.mark.parametrize("delay, duration, ref_min_utc, label", [
    (0.0, 90.0, None, "picks at 12:00 local -- the earliest kickoff ever"),
    (2.0, 64.59, 9 * 60 + 30, "tomorrow's first kickoff 11:30 local"),
])
def test_KNOWN_WRONG_a_kickoff_BEFORE_picks_are_ready_is_EXPOSED(
        delay, duration, ref_min_utc, label):
    ack = sd.dispatch_exposure_ack(SUNDAY, delay, duration, ref_min_utc)
    assert ack["state"] == "EXPOSED", (label, ack)


def test_a_WEEKDAY_and_an_UNMEASURED_dispatch_are_NOT_APPLICABLE():
    assert sd.dispatch_exposure_ack(MONDAY, 2.0, 64.59, None)[
        "state"] == "NOT_APPLICABLE"
    assert sd.dispatch_exposure_ack(SUNDAY, None, 64.59, None)[
        "state"] == "NOT_APPLICABLE"


def test_the_MEASURED_exposure_is_recorded_with_its_population():
    m = sd.DISPATCH_EXPOSURE_MEASURED
    assert m["measured"] == "2026-11-21"
    assert m["days"] == {"weekend": 257, "weekday": 529}
    assert m["earliest_local"] == {"weekend": "12:00", "weekday": "12:30"}
    assert m["exposed_at_review_era_weekend_term"]["weekend"] == 37
