"""The coverage residual, the provider echo, and the empty fold — 2026-10-19.

    A residual is a distribution until its tail is measured. One observation
    is not a bound.

PART A. Easter Monday 2026 measured a 15-minute gap between the earliest
uncovered fixture and the earliest covered one. That was ONE case, and not a
typical one.

THE RESIDUAL, `measured` 2026-10-19 over production `matches`, phantoms and
`world/*` excluded, configured leagues only, defined as
`min(kickoff | 21 uncovered) - min(kickoff | 9 covered)` so that NEGATIVE
means the uncovered card is earlier and the reference is LATE:

| stratum | n | negative | median | p90 | p99 | max | >60m |
|---|---|---|---|---|---|---|---|
| ALL     | 585 | 388 (66.3%) | 60 | 240 | 450 | **585** | 45.6% |
| weekday | 305 | 235 (77.0%) | 120 | 285 | 510 | **585** | 59.0% |
| weekend | 280 | 153 (54.6%) | 30 | 180 | 330 | 360 | 31.1% |
| holiday |  14 |  11 (78.6%) | 60 | 360 | 510 | 510 | 42.9% |

Taken at face value that says the combination is unsafe on half of all days
and the needed minimum is 570 minutes. **Both conclusions are wrong**, and the
reason is the point of this file: the residual is not what the margin bears.

THE CONSTANT BINDS ON 1087 OF 1090 DAYS. The deadline is
`min(reference, constant)` and both constants sit far below the first covered
kickoff, so what the margin bears is the EXPOSURE:

    exposure = max(0, deadline - min(kickoff | all 30 configured leagues))

| stratum | n | exposure>0 | p90 | p99 | max | >60m |
|---|---|---|---|---|---|---|
| ALL              | 1090 | 21 (1.9%) | 0 | 45 | **150** | 8 (0.7%) |
| weekday          |  715 | 14 (2.0%) | 0 | 75 | **150** | 8 (1.1%) |
| weekend          |  375 |  7 (1.9%) | 0 | 15 | 30 | 0 (0.0%) |
| weekday/holiday  |   22 |  3 (13.6%) | 15 | 90 | 90 | 1 (4.5%) |

**THE 60-MINUTE MINIMUM DOES NOT ABSORB IT ON THE WEEKDAY OR HOLIDAY STRATUM**
— max 150m, exceeded on 8 of 715 weekday days. That is the finding. The
weekend stratum IS absorbed (max 30m).

**AND IT IS NOT INTRODUCED BY THE REFERENCE.** Under the constant alone the
exposure is max 150m on 23 of 1090 days; under the combination, max 150m on
21. `min()` can only lower a deadline, so exposure(combination) <=
exposure(constant) always — measured, 3 days improved (median 30m, max 90m)
and **ZERO** days made worse. The tail belongs to the 12:00 weekday constant,
already recorded on 2026-10-10 as the ~12th percentile over an observed
minimum of 09:59.

THE DRIVER IS REAL, NOT A TIMEZONE ARTEFACT. `bulgaria/efbet-league` is the
earliest uncovered league on 7 of the 10 worst days; its own distribution is
min 09:30, p05 10:30, p50 15:30 UTC — i.e. 11:30/12:30 local, ordinary for
Bulgaria — with `finland/veikkausliiga` at 09:59 and `italy/serie-b` at 10:30.
Checked against three other UTC+2/+3 leagues, and against
`england/premier-league`, whose own minimum is 11:30 UTC with 24 fixtures
before 12:00: a COVERED league can go below the constant too, which is why the
reference binds on any day at all.
"""

import datetime as dt
import json
import pathlib

import pytest

import scripts.schedule_deadline as sd


# ── PART A: THE RESIDUAL TRAVELS WITH THE DEADLINE ──────────────────────────

def test_the_measured_exposure_is_RECORDED_per_stratum():
    """The distribution, not the one observation."""
    assert set(sd.COVERAGE_EXPOSURE) == {"weekday", "weekend"}
    wd, we = sd.COVERAGE_EXPOSURE["weekday"], sd.COVERAGE_EXPOSURE["weekend"]
    assert (wd["p99"], wd["max"], wd["n"], wd["over60"]) == (75, 150, 715, 8)
    assert (we["p99"], we["max"], we["n"], we["over60"]) == (15, 30, 375, 0)
    assert we["max"] <= 60 < wd["max"], (
        "the strata no longer fail differently; the weekend was absorbed by "
        "60m and the weekday was not, and that asymmetry is the finding")


def test_the_MINIMUM_is_DERIVED_from_the_tail_not_chosen():
    """60 + p99, and the registered 60 is NOT edited.

    `MIN_ACCEPTABLE_MARGIN_MIN` was stated on 2026-10-10 BEFORE any of this
    was computed. Refitting a pre-registered threshold to a later measurement
    is how a stated bound becomes a fitted one, so the exposure is added as
    its own term instead.
    """
    assert sd.MIN_ACCEPTABLE_MARGIN_MIN == 60, (
        "the pre-registered minimum was edited; it must stay 60 and the "
        "residual must be carried as a separate, derived term")
    for dtype, e in sd.COVERAGE_EXPOSURE.items():
        assert sd.EFFECTIVE_MIN_MARGIN[dtype] == 60 + e["p99"]
    assert sd.EFFECTIVE_MIN_MARGIN == {"weekday": 135, "weekend": 75}


def test_the_derived_minimum_CREATES_NO_ALARM_TODAY():
    """Raising the requirement must be shown not to manufacture a finding.

    `measured`: the prospective margin is +225m binding, which clears 135 and
    the max-case 210 alike.
    """
    MEASURED_BINDING = 225
    for dtype, m in sd.EFFECTIVE_MIN_MARGIN.items():
        assert MEASURED_BINDING >= m, f"{dtype}: {MEASURED_BINDING} < {m}"
    worst = max(60 + e["max"] for e in sd.COVERAGE_EXPOSURE.values())
    assert worst == 210 and MEASURED_BINDING >= worst


def test_the_RESIDUAL_is_printed_BESIDE_the_deadline_with_its_BASIS():
    """An improved deadline must never read as a safe one."""
    day = dt.date(2026, 10, 20)
    ref = sd.reference_earliest(day, fetch=lambda d, k: {
        "filters": {"dateFrom": "2026-10-20"},
        "matches": [{"competition": {"code": "CL"}, "status": "TIMED",
                     "utcDate": "2026-10-20T16:45:00Z"}]})
    c = sd.combined_deadline(day, ref)
    assert c["exposure"]["p99"] == 75
    line = sd.format_reference_deadline(c)
    assert "residual[" in line, "the residual is not printed"
    for part in ("p99=75m", "max=150m", "over60=8/715d",
                 "effective minimum 135m", "60m registered + 75m p99",
                 "basis:", "measured 2026-10-19"):
        assert part in line, f"the residual line omits {part!r}:{chr(10)}{line}"


def test_the_exposure_the_VERDICT_uses_is_the_one_PRINTED():
    """One number, one definition. The `LIVE 0 4 * * 1-5` label is what
    happens when the printed figure and the used figure are two."""
    import inspect

    src = inspect.getsource(sd.coverage_exposure)
    assert "COVERAGE_EXPOSURE[dtype]" in src or "COVERAGE_EXPOSURE[" in src
    import scripts.ci_audit as ci

    audit = inspect.getsource(ci.main)
    assert 'min_margin=_ahead["exposure"]["effective_min"]' in audit, (
        "the audit no longer takes its minimum from the same exposure dict it "
        "prints, so the two can diverge")


def test_the_combination_can_only_LOWER_the_exposure():
    """The property that makes consulting an incomplete reference safe.

    Driven over the range rather than asserted: for every reference value, the
    combined deadline is <= the constant, so the exposure against any true
    kickoff is <= the constant's exposure.
    """
    for day in (dt.date(2026, 4, 6), dt.date(2026, 5, 17)):
        const = sd.DEADLINE_PERCENTILES[sd.day_type(day)][0]
        for minute in range(6 * 60, 23 * 60, 23):
            iso = f"{day.isoformat()}T{minute // 60:02d}:{minute % 60:02d}:00Z"
            ref = sd.reference_earliest(day, fetch=lambda d, k, _i=iso: {
                "filters": {"dateFrom": d.isoformat()},
                "matches": [{"competition": {"code": "SA"},
                             "status": "TIMED", "utcDate": _i}]})
            c = sd.combined_deadline(day, ref)
            for true_ko in range(6 * 60, 23 * 60, 97):
                exp_const = max(0, const - true_ko)
                exp_comb = max(0, c["deadline_min"] - true_ko)
                assert exp_comb <= exp_const, (
                    f"the combination increased the exposure at "
                    f"ref={minute} true={true_ko}")


# ── PART B: THE RESPONSE IS VALIDATED AGAINST THE REQUEST ───────────────────

FIXTURE = pathlib.Path("tests/fixtures/footballdata_2027-04-02_wrong_date.json")


def test_KNOWN_WRONG_the_2027_04_02_response_VERBATIM_gives_NO_FIXTURE():
    """The case that motivated the filter, as the provider actually sent it.

    `measured` 2026-10-19: asked for 2027-04-02, football-data.org returned
    **39 matches, every one dated 2027-04-03**. Saved verbatim, 52,343 bytes,
    so the control is the real payload and not a paraphrase of it.
    """
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert len(payload["matches"]) == 39, "the fixture was edited"
    assert {m["utcDate"][:10] for m in payload["matches"]} == {"2027-04-03"}

    ref = sd.reference_earliest(dt.date(2027, 4, 2),
                                fetch=lambda d, k: payload)
    assert ref["state"] == sd.REF_NO_FIXTURE
    assert ref["min"] is None, (
        "the next day's earliest kickoff was attributed to the day we asked "
        "about -- the exact confident wrong answer this filter exists for")
    assert ref["dropped"]["other_day"] == 39, (
        "the discarded rows are not counted, so silently dropping 39 rows and "
        "silently using them look the same in the output")
    assert "39 of them were for a DIFFERENT DATE" in ref["reason"]

    # ...and the deadline falls to the constant rather than to 2027-04-03
    c = sd.combined_deadline(dt.date(2027, 4, 2), ref)
    assert c["deadline_min"] == c["constant_min"] and c["bound_by"] == "CONSTANT"


def test_the_PROVIDER_ECHOES_THE_REQUEST_and_date_is_a_TWO_DAY_WINDOW():
    """The correction to my own earlier note.

    It said the provider "does not honour `date`". Its own echo disproves
    that: `filters: {dateFrom: 2027-04-02, dateTo: 2027-04-03}`. `date=X`
    expands to `[X, X+1]` INCLUSIVE, so those 39 rows are the honest second
    day of the window and the defect would have been OURS -- attributing a
    two-day window's minimum to one day.
    """
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert payload["filters"]["dateFrom"] == "2027-04-02"
    assert payload["filters"]["dateTo"] == "2027-04-03"
    ref = sd.reference_earliest(dt.date(2027, 4, 2),
                                fetch=lambda d, k: payload)
    assert ref["echo"]["dateFrom"] == "2027-04-02"
    assert ref["echo_matches_request"] is True, (
        "the echo matched and was reported as a mismatch")


def test_an_ECHO_THAT_DISAGREES_is_a_FAILED_FETCH_not_an_answer():
    """If `dateFrom` is not what we asked for, the provider answered a
    different question and NOTHING in the response may be attributed."""
    ref = sd.reference_earliest(dt.date(2026, 10, 20), fetch=lambda d, k: {
        "filters": {"dateFrom": "2026-11-01", "dateTo": "2026-11-02"},
        "matches": [{"competition": {"code": "CL"}, "status": "TIMED",
                     "utcDate": "2026-10-20T09:00:00Z"}]})
    assert ref["state"] == sd.REF_FETCH_FAILED
    assert ref["min"] is None, (
        "a row was read out of a response that answered another question")
    assert ref["echo_matches_request"] is False
    assert "DIFFERENT question" in ref["reason"]


def test_FOUR_STATES_where_there_are_FOUR_FACTS():
    """`NO_FIXTURE` and `EMPTY_RESPONSE` were one state until today.

    An empty response says the provider has nothing anywhere near the query.
    `NO_FIXTURE` says it answered WITH ROWS and not one was for the day asked
    about -- which is what 2027-04-02 did. Conflating them hides exactly the
    case where the provider answered a different question.
    """
    day = dt.date(2026, 10, 20)
    echo = {"dateFrom": day.isoformat(), "dateTo": "2026-10-21"}

    empty = sd.reference_earliest(day, fetch=lambda d, k: {
        "filters": echo, "matches": []})
    wrongday = sd.reference_earliest(day, fetch=lambda d, k: {
        "filters": echo,
        "matches": [{"competition": {"code": "CL"}, "status": "TIMED",
                     "utcDate": "2026-10-21T16:45:00Z"}]})
    nokey = sd.reference_earliest(day, api_key="")

    def boom(_d, _k):
        raise OSError("reset")

    failed = sd.reference_earliest(day, fetch=boom)

    assert empty["state"] == sd.REF_EMPTY_RESPONSE
    assert wrongday["state"] == sd.REF_NO_FIXTURE
    assert nokey["state"] == sd.REF_NO_KEY
    assert failed["state"] == sd.REF_FETCH_FAILED
    reasons = {r["reason"] for r in (empty, wrongday, nokey, failed)}
    assert len(reasons) == 4, f"states share a reason: {len(reasons)} distinct"
    assert "NO ROWS AT ALL" in empty["reason"]
    assert wrongday["dropped"]["other_day"] == 1

    # the two that LOOKED are reference-verified; the two that could not are not
    for r, verified in ((empty, True), (wrongday, True),
                        (nokey, False), (failed, False)):
        assert sd.combined_deadline(day, r)["reference_verified"] is verified


def test_the_states_are_ENUMERATED_and_all_five_are_reachable():
    assert set(sd.REFERENCE_STATES) == {
        sd.REF_OK, sd.REF_NO_FIXTURE, sd.REF_EMPTY_RESPONSE, sd.REF_NO_KEY,
        sd.REF_FETCH_FAILED}
    assert len(set(sd.REFERENCE_STATES)) == 5


# ── PART B, THE SWEEP: THE WEATHER READ HAD THE SAME SHAPE ──────────────────

def test_the_WEATHER_read_validates_the_ECHOED_date():
    """Second instance of the attributing shape, found by the sweep.

    `_fetch_daily_weather` requests `start_date=end_date=date_str` and read
    `daily["temperature_2m_max"][0]`, attributing element 0 to the match's
    date WITHOUT ever reading `daily["time"]` -- which is the provider echoing
    the dates it answered for. `measured` 2026-10-19: the echo exists and is
    honest (asked 2026-10-22, echoed `['2026-10-22']`).
    """
    import ast
    import inspect

    import src.features.weather_service as w

    src = inspect.getsource(w._fetch_daily_weather)
    tree = ast.parse(src.lstrip())
    consts = {n.value for n in ast.walk(tree)
              if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert "time" in consts, (
        "the weather read does not look at `daily['time']`, so it still "
        "attributes element 0 to the requested date on trust")
    assert hasattr(w, "_date_mismatches"), (
        "mismatches are not counted; a silent drop and an honest provider "
        "look the same")


def test_a_WRONG_DAY_forecast_takes_the_EXISTING_unavailable_path():
    """Driven, with the network stubbed. A wrong-day forecast IS 'no weather
    for this match', so it reuses the path a failed fetch already takes rather
    than inventing a third behaviour."""
    import json as _json
    import urllib.request

    import src.features.weather_service as w

    class _Resp:
        def __init__(self, payload):
            self._p = _json.dumps(payload).encode()

        def read(self):
            return self._p

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _fake(payload):
        return lambda *_a, **_k: _Resp(payload)

    day = dt.date(2026, 10, 20)
    w._WEATHER_CACHE.clear()
    w._date_mismatches = 0
    orig = urllib.request.urlopen
    try:
        # WRONG day echoed -> unavailable, counted
        urllib.request.urlopen = _fake({"daily": {
            "time": ["2026-10-21"], "temperature_2m_max": [30.0],
            "precipitation_sum": [0.0], "windspeed_10m_max": [5.0]}})
        r = w._fetch_daily_weather(51.5, -0.12, day)
        assert r["weather_available"] == 0, (
            "a forecast for another day was attributed to this match")
        assert r["weather_temp_c"] == 12.0, "it is not the defaults path"
        assert w._date_mismatches == 1

        # RIGHT day echoed -> used
        w._WEATHER_CACHE.clear()
        urllib.request.urlopen = _fake({"daily": {
            "time": [day.isoformat()], "temperature_2m_max": [21.5],
            "precipitation_sum": [2.0], "windspeed_10m_max": [8.0]}})
        r2 = w._fetch_daily_weather(51.5, -0.12, day)
        assert r2["weather_available"] == 1 and r2["weather_temp_c"] == 21.5
        assert w._date_mismatches == 1, "a matching echo was counted"
    finally:
        urllib.request.urlopen = orig
        w._WEATHER_CACHE.clear()
        w._date_mismatches = 0


def test_the_SWEEP_is_RECORDED_including_what_it_cleared():
    """A sweep that lists only its findings cannot be audited.

    openfootball is NOT a call site -- it appears only in `ci_audit.py`
    comments, as the manual reference used for two verifications. Asserted so
    the claim is checkable rather than remembered.
    """
    import pathlib as _p

    from tests.conftest import code_of

    for f in ("src/scrapers/apifootball_scraper.py",
              "src/scrapers/theodds_scraper.py",
              "src/features/weather_service.py",
              "src/scrapers/footballdataorg_scraper.py"):
        assert _p.Path(f).exists(), f"{f} moved; re-run the sweep"
    # BY CALL SITE, NOT BY WORD. The name appears in `ci_audit.py` inside a
    # suppression MESSAGE -- `f"reference openfootball 2026-27 shows 0
    # fixtures..."` -- which `code_of` keeps, because it strips comments and
    # docstrings and not ordinary string literals. A reference that is quoted
    # in a log line is not a reference that is fetched, and the test must
    # distinguish those or it fails on prose for the fourth time this stage.
    import re as _re

    urls = []
    for f in list(_p.Path("src").rglob("*.py")) + list(
            _p.Path("scripts").rglob("*.py")):
        if "__pycache__" in f.parts:
            continue
        for m in _re.finditer(r"https?://[^\s\"']+", f.read_text(
                encoding="utf-8", errors="replace")):
            if "openfootball" in m.group(0) or "githubusercontent" in m.group(0):
                urls.append(f"{f.as_posix()}: {m.group(0)}")
    assert not urls, (
        f"openfootball has become a LIVE call site and must be swept for the "
        f"request/response shape: {urls}")

    # TheOddsAPI carries NO date parameter: the sport key is in the PATH, and
    # each event carries its own `commence_time`, which is what is read.
    odds = _p.Path("src/scrapers/theodds_scraper.py").read_text(encoding="utf-8")
    assert "/sports/{sport_key}/odds" in odds.replace('f"', '"')
    assert "commence_time" in odds


# ── PART C: DID AN EMPTY FOLD EVER REACH A RECORDED BASELINE? ──────────────
#
#     ANSWER: NO, in either harness, and the evidence differs between them.
#
# `run_baseline.py`  -- `walk_forward(min_train=500)` is its only caller and
#   skips any cutoff below it. MEASURED over the five recorded cutoffs against
#   the population as of 2026-08-07 (39,658 rows): trains were 30,916 /
#   32,611 / 35,001 / 38,247 / 39,183. ZERO empty, ZERO skipped. Under `b1`
#   the filter was LOOSER (584 excluded rows admitted), so the recorded trains
#   were >= these -- monotone, so the b1 figures are covered too.
#
# `run_clean_baseline.py` -- `run_market` skips when `len(train) < 300`, and
#   here that guard WAS LOAD-BEARING. Against the population as of 2026-08-07
#   (2,671 clean rows, against the recorded manifest's 2,700 -- a 29-row
#   agreement that validates the reconstruction) THREE OF FIVE cutoffs had an
#   ENTIRELY EMPTY train: 2025-08-01, 2025-11-01, 2026-02-01. All three were
#   skipped, so `fit_poisson` was never called on them.
#
# EMPTY FOLDS THAT REACHED THE DEFECT, over every recorded baseline: ZERO.
# No recorded figure moves, so there is NO `baseline_revision` event. The
# guard added on 2026-10-18 is PROSPECTIVE, and worth having: in the clean
# harness the `or 1.0` was one `continue` away from being exercised on three
# folds in five.

def test_BOTH_harnesses_GUARD_the_call_site_and_that_is_what_PROTECTED_them():
    """The guards are the evidence, so they must not quietly disappear."""
    import inspect

    from src.evaluation.baseline import walk_forward

    assert inspect.signature(walk_forward).parameters["min_train"].default == 500
    body = inspect.getsource(walk_forward)
    assert "len(train) < min_train" in body and "continue" in body

    clean = pathlib.Path("scripts/run_clean_baseline.py").read_text(
        encoding="utf-8")
    assert "if len(train) < 300 or not test:" in clean, (
        "the clean harness lost its call-site guard -- and MEASURED, three of "
        "its five recorded folds had an entirely empty training set, so that "
        "guard is what kept `w.sum() or 1.0` from ever being exercised")


def test_the_recorded_baselines_are_STAMPED_and_the_revision_did_NOT_move():
    """No recorded figure rests on an empty fold, so this stage is not a
    `baseline_revision` event."""
    from src.evaluation.baseline import BASELINE_REVISION

    assert BASELINE_REVISION == "b2", (
        "the baseline revision moved; Part C measured ZERO empty folds "
        "reaching a recorded figure, so nothing in this stage justifies a bump")
    recs = sorted(pathlib.Path("data/baselines").glob("*.json"))
    assert len(recs) == 3, f"recorded baselines changed: {[p.name for p in recs]}"
    for p in recs:
        d = json.loads(p.read_text(encoding="utf-8"))
        assert d.get("baseline_revision") == "b1.unstamped-pre-exclusion", (
            f"{p.name} carries {d.get('baseline_revision')!r}; the three "
            f"recorded snapshots are all pre-exclusion b1")


def test_the_PROSPECTIVE_guard_still_fires_on_a_DIRECT_call():
    """It never fired through the harness and it is still worth having."""
    import datetime as _dt

    import scripts.run_baseline as rb
    import scripts.run_clean_baseline as rcb

    for mod in (rb, rcb):
        with pytest.raises(ValueError, match="nothing to fit"):
            mod.fit_poisson([], _dt.date(2026, 1, 1))


# ── THE SWEEP, ENUMERATED AND PINNED ───────────────────────────────────────
#
#     A provider that answers a different question than it was asked, without
#     saying so, is indistinguishable from one that answered.
#
# EVERY external host the code calls, `measured` 2026-10-19 by extracting
# every URL literal from `src/` and `scripts/`, with the verdict for each:
#
# | host | request parameter | attributed? | verdict |
# |---|---|---|---|
# | api.football-data.org    | `?date=`             | YES | **FIXED**: echo + per-row |
# | api.open-meteo.com       | `start_date/end_date`| YES | **FIXED**: reads `daily.time` |
# | geocoding-api.open-meteo | `?name=`             | YES | PARTLY VALIDATED, already |
# | v3.football.api-sports.io| `?date=`             | no  | rows keyed by afid, own date |
# | api.the-odds-api.com     | key in PATH, no date | no  | each event's `commence_time` |
# | www.flashscore.com       | page scrape, none    | no  | rows carry parsed dates |
# | www.football-data.co.uk  | season/league in PATH| n/a | CSV, no echo to compare |
# | api.telegram.org         | outbound send        | n/a | nothing read back |
#
# GEOCODING, reported and NOT fixed. It fetches 3 results and prefers one
# whose `country_code` matches the request -- the validation already exists,
# with its reason on file (Dunfermline resolving to Illinois). What remains is
# the fallback to `results[0]` when NO result matches the country, which is a
# silent substitution of the wrong city. Changing it would change
# `weather_*` FEATURE VALUES, which is prediction-affecting and a cohort
# event, so it is recorded here rather than altered in an observability stage.

#: `measured` 2026-10-19. Every external host reachable from `src/` or
#: `scripts/`. Pinned so a NEW external read cannot land without the
#: request/response sweep being redone.
EXTERNAL_HOSTS = {
    "api.football-data.org",
    "api.open-meteo.com",
    "api.telegram.org",
    "api.the-odds-api.com",
    "geocoding-api.open-meteo.com",
    "v3.football.api-sports.io",
    "www.flashscore.com",
    "www.football-data.co.uk",
    "www.football-data.org",      # the human sign-up page, in a docstring
}


def test_the_SET_OF_EXTERNAL_HOSTS_has_not_grown():
    """A new provider is a new chance to answer a different question."""
    import re as _re

    found = set()
    for f in (list(pathlib.Path("src").rglob("*.py"))
              + list(pathlib.Path("scripts").rglob("*.py"))):
        if "__pycache__" in f.parts:
            continue
        for m in _re.finditer(r"https?://[^\s\"']+",
                              f.read_text(encoding="utf-8", errors="replace")):
            host = m.group(0).split("/")[2]
            if not host.endswith("example.invalid"):
                found.add(host)
    assert found == EXTERNAL_HOSTS, (
        f"the external host set changed.{chr(10)}"
        f"  added  : {sorted(found - EXTERNAL_HOSTS)}{chr(10)}"
        f"  removed: {sorted(EXTERNAL_HOSTS - found)}{chr(10)}"
        f"Each addition must be swept: does the request carry a parameter the "
        f"response is attributed to, and is the echo compared?")


def test_the_GEOCODER_already_validates_the_COUNTRY_and_the_residual_is_NAMED():
    """Reported, not fixed, and the reason is in the test rather than only in
    a comment."""
    import inspect

    import src.features.weather_service as w

    src = inspect.getsource(w._geocode)
    assert 'country_code", "").upper() == country_code.upper()' in src, (
        "the geocoder stopped comparing the returned country against the "
        "requested one")
    assert "chosen = results[0]" in src, (
        "the residual changed; re-measure before relying on this note")
