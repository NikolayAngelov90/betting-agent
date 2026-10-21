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
    # ── RECOMPUTED 2026-10-19 UNDER THE DERIVED FLOOR. ────────────────────
    #
    # The figures below were taken under the 12:00 / 10:15 percentiles:
    #   weekday  p99 75m  max 150m  >60m on 8 of 715 days
    #   weekend  p99 15m  max  30m  >60m on 0 of 375 days
    # The floor (09:45 / 09:15) removed almost all of it, because the tail
    # belonged to the constant and not to the reference.
    assert set(sd.COVERAGE_EXPOSURE) == {"weekday", "weekend"}
    wd, we = sd.COVERAGE_EXPOSURE["weekday"], sd.COVERAGE_EXPOSURE["weekend"]
    # RE-ANCHORED 2026-10-20: the deadline is the OBSERVED EARLIEST KICKOFF (weekday 09:30, weekend 09:45), not ceil_15(ready+60). The self-anchored floor was a TAUTOLOGY -- it made margin>=60 true by construction -- and freezing it against a live series breached 60 within a day when mean+3sd drifted +11.4m on one new observation.
    # Exposure is now ZERO BY CONSTRUCTION on both strata: the deadline IS
    # the observed minimum, so max(0, deadline - true) cannot be positive
    # over the measured window. The 15-minute weekday residual existed
    # only because the self-anchored floor sat ABOVE the minimum.
    assert (wd["p99"], wd["max"], wd["n"], wd["over60"]) == (0, 0, 715, 0)
    assert (we["p99"], we["max"], we["n"], we["over60"]) == (0, 0, 375, 0)
    assert wd["over60"] == we["over60"] == 0, (
        "THE TARGET: zero days above 60 minutes, on every stratum")
    assert wd["max"] == 0, (
        "a weekday exposure reappeared; the deadline is the observed "
        "minimum, so this can only happen if the minimum was re-measured "
        "lower without re-anchoring the deadline")


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
    # THE TERM IS NOW ZERO AND THE ARITHMETIC IS UNCHANGED. It read 135 / 75
    # under the old constants; the floor drove the p99 to zero, so the
    # effective minimum collapses back onto the registered 60. The term is
    # KEPT rather than removed -- a term that disappears when it is satisfied
    # cannot be seen to have been satisfied.
    assert sd.EFFECTIVE_MIN_MARGIN == {"weekday": 60, "weekend": 60}


def test_the_deadline_now_FAILS_the_registered_60_AND_THAT_IS_THE_FINDING():
    """RENAMED AND INVERTED 2026-10-20. It was
    `test_the_derived_minimum_CREATES_NO_ALARM_TODAY`, and it passed because
    the deadline was SELF-ANCHORED at `ceil_15(ready + 60)` -- which makes
    "the margin clears 60" true by construction. A check that cannot fail is
    not a check.

    Re-anchored to the OBSERVED EARLIEST KICKOFF, the arithmetic reports the
    truth, `measured` 2026-10-20:

        weekday  deadline 09:30   ready 08:49   margin **+40m**  vs 60
        weekend  deadline 09:45   ready 08:03   margin +101.7m   clears

    **19.8 minutes short on the weekday**, and the self-anchoring was hiding
    15 of them. The registered 60 is NOT edited -- it was stated 2026-10-10
    before any of this -- so the breach stands as a finding.

    NO DEADLINE CHOICE CLEARS IT: the earliest kickoff is a fact and the ready
    time is ours. The levers are an earlier cron or a faster run, and the
    duration profile sized the second at ~7 min (injury loop) + ~7 min (Claude
    review) + up to ~9 min (AF upsert) -- any two of which close 19.8 minutes.
    """
    #: `measured` 2026-10-20: weekday ready_binding 529.8, deadline 570.
    MEASURED_WEEKDAY_MARGIN = 40
    assert sd.MIN_ACCEPTABLE_MARGIN_MIN == 60, (
        "the registered minimum was edited to make the breach go away")
    assert MEASURED_WEEKDAY_MARGIN < sd.EFFECTIVE_MIN_MARGIN["weekday"], (
        "the weekday margin now clears the effective minimum -- better than "
        "the state recorded on 2026-10-20, and it must be re-measured and "
        "re-stated rather than assumed")
    shortfall = sd.EFFECTIVE_MIN_MARGIN["weekday"] - MEASURED_WEEKDAY_MARGIN
    assert shortfall == 20, (
        f"the shortfall moved to {shortfall}m; it was 19.8 rounded to 20")

    # the exposure term is ZERO and the trade moved into the margin
    assert sd.COVERAGE_EXPOSURE["weekday"]["max"] == 0
    assert sd.EFFECTIVE_MIN_MARGIN == {"weekday": 60, "weekend": 60}

    # the rejected design is retained so nobody re-derives it
    assert sd.REJECTED_SELF_ANCHORED_FLOOR["weekday"] == 9 * 60 + 45
    assert "tautology" in sd.REJECTED_SELF_ANCHORED_FLOOR["why_rejected"]


def test_the_RESIDUAL_is_printed_BESIDE_the_deadline_with_its_BASIS():
    """An improved deadline must never read as a safe one."""
    day = dt.date(2026, 10, 20)
    ref = sd.reference_earliest(day, fetch=lambda d, k: {
        "filters": {"dateFrom": "2026-10-20"},
        "matches": [{"competition": {"code": "CL"}, "status": "TIMED",
                     "utcDate": "2026-10-20T16:45:00Z"}]})
    c = sd.combined_deadline(day, ref)
    assert c["exposure"]["p99"] == 0
    line = sd.format_reference_deadline(c)
    assert "residual[" in line, "the residual is not printed"
    # PRINTED AT ZERO, which is the whole point: the figure must be visible
    # as SATISFIED rather than absent.
    for part in ("p99=0m", "max=0m", "over60=0/715d",
                 "effective minimum 60m", "60m registered + 0m p99",
                 "basis:", "measured 2026-10-19"):
        assert part in line, f"the residual line omits {part!r}:{chr(10)}{line}"


def test_the_exposure_the_VERDICT_uses_is_the_one_PRINTED():
    """One number, one definition. The `LIVE 0 4 * * 1-5` label is what
    happens when the printed figure and the used figure are two."""
    import inspect

    src = inspect.getsource(sd.coverage_exposure)
    assert "COVERAGE_EXPOSURE[dtype]" in src or "COVERAGE_EXPOSURE[" in src
    import scripts.ci_audit as ci

    # ── ASSERTED AS AN EQUALITY, NOT A SOURCE STRING. 2026-10-21. ─────────
    #
    # This matched `min_margin=_ahead["exposure"]["effective_min"]` in the
    # audit's source. The two-margin split routes the minimum through
    # `two_margins`, so the literal moved -- but the invariant it protects
    # ("the minimum the verdict uses is the one printed") is a fact about
    # NUMBERS, and comparing them directly cannot be defeated by a rename.
    # Same lesson as the comment-blind text scans: assert the value.
    import datetime as dt

    delays = [200.0, 240.0, 260.0, 280.0, 300.0, 220.0, 230.0, 250.0]
    durations = [30.0, 40.0, 42.0, 45.0, 50.0, 60.0, 35.0, 38.0]
    for day in (dt.date(2026, 10, 22), dt.date(2026, 10, 24)):   # wkday, wkend
        dtype = sd.day_type(day)
        m = sd.two_margins(day, {"min": 16 * 60, "state": "OK"}, 0,
                           delays, durations)
        for half in ("structural", "actionable"):
            assert m[half]["min_margin"] == sd.EFFECTIVE_MIN_MARGIN[dtype], (
                f"{half} verdict uses {m[half]['min_margin']} while the "
                f"printed effective minimum is "
                f"{sd.EFFECTIVE_MIN_MARGIN[dtype]} -- the two have diverged")
        # and the printed effective minimum is still built from the exposure
        # that `coverage_exposure` reports, at whatever value that is
        assert sd.EFFECTIVE_MIN_MARGIN[dtype] == (
            sd.MIN_ACCEPTABLE_MARGIN_MIN
            + sd.coverage_exposure(day)["p99"]), (
            "the effective minimum is no longer the registered minimum plus "
            "the exposure this module measures")


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
              "src/scrapers/footballdataorg_scraper.py"):
        assert _p.Path(f).exists(), f"{f} moved; re-run the sweep"
    # DELETED 2026-10-20 — `src/features/weather_service.py` was the fourth
    # entry here. Its two Open-Meteo hosts left the system with it.
    assert not _p.Path("src/features/weather_service.py").exists()
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
    "api.telegram.org",
    "api.the-odds-api.com",
    "v3.football.api-sports.io",
    "www.flashscore.com",
    "www.football-data.co.uk",
    "www.football-data.org",      # the human sign-up page, in a docstring
}

#: SHRANK 2026-10-20, and that is a measurement rather than housekeeping:
#: deleting the weather producer removed `api.open-meteo.com` AND
#: `geocoding-api.open-meteo.com` from the set of providers this system talks
#: to at all. Nine hosts to seven.
RETIRED_HOSTS = {"api.open-meteo.com", "geocoding-api.open-meteo.com"}


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

# ═══════ RETIRED 2026-10-20 — THE PRODUCER THEY GUARDED WAS DELETED ═══════
#
# `src/features/weather_service.py` and `FeatureEngineer._get_weather_features`
# are gone: the six `weather_*` keys were read by nothing. The checks below
# were correct and are now moot, so they are RETIRED WITH THEIR REASON rather
# than deleted silently -- a check that vanishes without a note is
# indistinguishable from one that was quietly dropped because it failed.
#
#   test_the_WEATHER_read_validates_the_ECHOED_date
#       pinned that `_fetch_daily_weather` compared `daily["time"]` against
#       the requested date. The GENERAL rule it was an instance of survives
#       for football-data.org in `test_the_PROVIDER_ECHOES_THE_REQUEST...`.
#
#   test_a_WRONG_DAY_forecast_takes_the_EXISTING_unavailable_path
#       drove the mismatch branch with the network stubbed. No branch left.
#
#   test_the_GEOCODER_already_validates_the_COUNTRY_and_the_residual_is_NAMED
#       pinned `results[0]` and the country preference so the reported-not-
#       fixed note could not go stale. The note is now HISTORY: the census
#       (3 of 185 venues wrong, 4 fixtures, ZERO realised effect) stands as a
#       record of what was deleted, not of what is exposed.
#
# What replaces all three: `test_the_WEATHER_PRODUCER_IS_GONE`, which fails if
# the producer comes back without its consumer.


def test_the_WEATHER_PRODUCER_IS_GONE():
    """The deletion, pinned. Gated on the aligned vector for 8 real fixtures
    being byte-identical before and after -- it was, with raw features going
    272 -> 266 (exactly the six keys) and 8 distinct hashes of 8 proving the
    instrument discriminates."""
    assert not pathlib.Path("src/features/weather_service.py").exists(), (
        "the weather service is back. It had no consumer: training skipped "
        "it by construction and `_align_features` dropped it at predict "
        "time. If it returns, it needs a reader first.")
    from src.features.feature_engineer import FeatureEngineer

    for gone in ("_get_weather_features", "_league_country_code",
                 "_LEAGUE_COUNTRY_CODES"):
        assert not hasattr(FeatureEngineer, gone), (
            f"`{gone}` is back; it existed only for the weather producer")
    src = pathlib.Path("src/features/feature_engineer.py").read_text(
        encoding="utf-8")
    assert "MARK-NEVER-DELETE APPLIES TO DATA, NOT TO PRODUCERS" in src, (
        "the deletion no longer records the data/producer split it rests on")
    assert "AWAITING_CONSUMER" in src, (
        "the note no longer contrasts this with the substrates that DO have "
        "registered questions and are therefore not deletable")
