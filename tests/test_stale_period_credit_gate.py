"""A stale credit figure is NO reading, not a low one.

THE SCENARIO, recorded as a near-miss in August and now near-certain.

`_load_persisted_credits()` read `remaining` and never read `updated`, though
`_persist_credits` has always written both. **August closed at 15 remaining and
cleared the <=10 hard skip by five credits.** September is projected to exhaust
around 2026-09-14, so the file will carry a figure at or near zero into
1 October — and the first run of the new month would have skipped the odds fetch
entirely, on a number describing a finished month, with a full 500-credit tier
sitting unused.

    A persisted figure whose `updated` date is not the current period is not a
    low reading. It is NO reading, and the response to no reading is to PROBE,
    not to skip. /v4/sports is free.

This is the same distinction drawn twice already in this file's neighbourhood:
`[]` versus `None` on a league fetch, and 429-with-credits versus
429-with-zero. Three states collapsed into two, three times, in one module.
"""

import asyncio
import json
from datetime import date, timedelta

import pytest

import src.scrapers.theodds_scraper as tos


def _write(tmp_path, monkeypatch, blob):
    p = tmp_path / "theodds_credits.json"
    p.write_text(json.dumps(blob))
    monkeypatch.setattr(tos, "_CREDITS_STATE_PATH", p)
    return p


# ═══════════════════════════════════════════ what counts as a reading

def test_a_current_period_reading_is_used(tmp_path, monkeypatch):
    _write(tmp_path, monkeypatch,
           {"remaining": 42, "updated": date.today().isoformat()})
    assert tos._load_persisted_credits() == 42


def test_last_months_reading_is_NOT_a_reading(tmp_path, monkeypatch):
    """The August-into-September case, and the one that bites on 1 October."""
    last_month = (date.today().replace(day=1) - timedelta(days=1)).isoformat()
    _write(tmp_path, monkeypatch, {"remaining": 3, "updated": last_month})
    assert tos._load_persisted_credits() is None, (
        "a 3 written last month was treated as 3 remaining today. The quota "
        "resets at the period boundary; that figure describes a finished month")


def test_the_exact_august_near_miss_is_still_handled(tmp_path, monkeypatch):
    """August closed at 15 and cleared the <=10 skip by five.

    Under the old code this passed by luck. It must now be rejected for the
    right reason — being from another period — rather than accepted for the
    wrong one.
    """
    _write(tmp_path, monkeypatch, {"remaining": 15, "updated": "2026-08-31"})
    if tos._credits_period("2026-08-31") != tos._credits_period(date.today()):
        assert tos._load_persisted_credits() is None


def test_a_reading_with_no_date_is_not_a_reading(tmp_path, monkeypatch):
    """Undatable is unusable for the same reason stale is."""
    _write(tmp_path, monkeypatch, {"remaining": 5})
    assert tos._load_persisted_credits() is None


def test_a_missing_file_is_not_a_reading(tmp_path, monkeypatch):
    monkeypatch.setattr(tos, "_CREDITS_STATE_PATH", tmp_path / "absent.json")
    assert tos._load_persisted_credits() is None


def test_persist_then_load_roundtrips_within_the_period(tmp_path, monkeypatch):
    """The writer and the reader must agree about the date field."""
    p = tmp_path / "rt.json"
    monkeypatch.setattr(tos, "_CREDITS_STATE_PATH", p)
    tos._persist_credits(77)
    assert tos._load_persisted_credits() == 77
    assert "updated" in json.loads(p.read_text())


# ═══════════════════════════════════════════════════════ the probe

def test_the_probe_reads_headers_and_costs_nothing():
    """/v4/sports is free; it is asked, not assumed."""
    sc = tos.TheOddsScraper.__new__(tos.TheOddsScraper)
    sc.api_key = "k"
    sc._remaining_requests = None
    sc._used_requests = None
    asked = {}

    class _R:
        status = 200
        headers = {"x-requests-remaining": "500", "x-requests-used": "0"}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    class _S:
        def get(self, url, **kw):
            asked["url"] = url
            return _R()

    async def _sess():
        return _S()

    sc._get_session = _sess
    monkeypatched = tos._CREDITS_STATE_PATH
    assert asyncio.run(sc.probe_credits()) == 500
    assert asked["url"].endswith("/sports"), (
        f"the probe must hit the free /v4/sports endpoint, not {asked['url']}")


def test_a_failed_probe_means_proceed_not_skip():
    """A probe that cannot answer is not evidence of an empty tier."""
    sc = tos.TheOddsScraper.__new__(tos.TheOddsScraper)
    sc.api_key = "k"
    sc._remaining_requests = None
    sc._used_requests = None

    async def _sess():
        raise RuntimeError("network down")

    sc._get_session = _sess
    assert asyncio.run(sc.probe_credits()) is None


def test_update_probes_when_the_reading_is_stale(tmp_path, monkeypatch):
    """THE POINT. Stale figure at zero + a full tier => proceed, not skip."""
    last_month = (date.today().replace(day=1) - timedelta(days=1)).isoformat()
    _write(tmp_path, monkeypatch, {"remaining": 0, "updated": last_month})

    sc = tos.TheOddsScraper.__new__(tos.TheOddsScraper)
    sc.api_key = "k"
    sc._remaining_requests = None
    sc._used_requests = None
    probed = {"n": 0}

    async def _probe():
        probed["n"] += 1
        return 500

    sc.probe_credits = _probe
    sc._leagues_with_today_fixtures = lambda: []      # stop right after the gate

    assert asyncio.run(sc.update()) == 0
    assert probed["n"] == 1, (
        "a stale zero did not trigger the free probe — 1 October would have "
        "skipped the odds fetch on September's number")


def test_update_still_skips_on_a_genuine_current_period_zero(tmp_path, monkeypatch):
    """The gate must keep its teeth. A real low reading still skips."""
    _write(tmp_path, monkeypatch,
           {"remaining": 2, "updated": date.today().isoformat()})

    sc = tos.TheOddsScraper.__new__(tos.TheOddsScraper)
    sc.api_key = "k"
    sc._remaining_requests = None
    sc._used_requests = None
    probed = {"n": 0}

    async def _probe():
        probed["n"] += 1
        return 500

    sc.probe_credits = _probe
    called = {"leagues": 0}

    def _leagues():
        called["leagues"] += 1
        return ["england/premier-league"]

    sc._leagues_with_today_fixtures = _leagues

    assert asyncio.run(sc.update()) == 0
    assert probed["n"] == 0, "a current-period reading must not be re-probed"
    assert called["leagues"] == 0, "the hard skip did not fire on a real 2"
