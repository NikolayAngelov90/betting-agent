"""The effective fold count, and the unverified first result — 2026-10-19.

    Every figure carries the population it was computed over — including the
    effective fold count.

PART B. `run_clean_baseline` requests five cutoffs and skips any whose
training set is under 300 rows. `measured` 2026-10-19 against the population as
it stood for the recorded 2026-08-07 run (**2,671** clean rows, against that
run's own manifest of **2,700 qualified** — a 29-row agreement that validates
the reconstruction):

| cutoff | train | ran? |
|---|---|---|
| 2025-08-01 | **0** | skipped, empty train |
| 2025-11-01 | **0** | skipped, empty train |
| 2026-02-01 | **0** | skipped, empty train |
| 2026-05-01 | 1,401 | ran |
| 2026-07-01 | 2,258 | ran |

**EFFECTIVE FOLDS: 2 of 5, and they are ADJACENT** — the real span is
2026-05-01 … 2026-08-07, about one quarter, not the twelve months five
chronological cutoffs imply. Clean odds coverage only begins in 2026-02.

`docs/stage4-clean-baseline-2026-08-07.md` recorded "Walk-forward, five
chronological cutoffs" and nothing stated otherwise.

AND THE SAME DEFECT IS IN THE DECISIVE TEST. `docs/predictive-audit-
2026-08-07.md` records "Five cutoffs … 5,771 out-of-sample matches (1,710 with
bookmaker prices)". The five cutoffs are right for the 5,771; they are wrong
for the **1,710 priced** matches that every blend-sweep figure is computed
over, because odds collection began 2026-02:

| cutoff | OOS | carrying a 1X2 price |
|---|---|---|
| 2025-08-01 | 1,035 | **0** |
| 2025-11-01 | 1,514 | **0** |
| 2026-02-01 | 2,099 | 845 |
| 2026-05-01 | 956 | 872 |
| 2026-07-01 | 630 | 414 |

**EFFECTIVE FOLDS FOR EVERY PRICED FIGURE: 3 of 5**, spanning about six
months. (Totals 6,234 / 2,131 here against the recorded 5,771 / 1,710 — the
difference is the exclusion filter and a looser "has a price" definition; the
**zeros** are structural, not a filter artefact.)

WHAT IT DOES AND DOES NOT UNDERMINE. Both bootstraps resample OBSERVATIONS, so
their intervals stand for their populations. What adjacent folds inside a
single odds-collection era cannot support is **temporal generality** — and the
recommendation drawn from the second one (`bookmaker_blend_weight` → 1.0) is a
standing configuration claim, which is the kind that needs it. No figure moves;
`BASELINE_REVISION` stays `b2`. This is a labelling defect.

PART C is REPORT ONLY — fixing either first-result site changes feature values,
which is a cohort event and belongs with the 11-01 queue.
"""

import json
import pathlib

import pytest


# ── PART B ──────────────────────────────────────────────────────────────────

def test_the_clean_harness_RECORDS_its_effective_fold_count():
    """It is returned with the scores, like `_losses`, so it travels."""
    import inspect

    import scripts.run_clean_baseline as rcb

    src = inspect.getsource(rcb.run_market)
    for part in ("folds_ran", "folds_skipped", "effective_folds",
                 "requested_folds", "EMPTY TRAIN"):
        assert part in src, f"run_market no longer records {part!r}"
    assert 's["_folds"] = fold_label' in src, (
        "the fold label no longer travels with the score, so a figure can be "
        "printed without it")


def test_a_clean_baseline_TABLE_CANNOT_PRINT_without_the_fold_count():
    """The mechanical half. A remedy that must be remembered is not one."""
    import scripts.run_clean_baseline as rcb

    unlabelled = {"X": {"log_loss": 1.0, "brier": 0.5, "ece": 0.01,
                        "accuracy": 0.5, "n": 10}}
    with pytest.raises(ValueError, match="WITHOUT its effective fold count"):
        rcb.show(unlabelled, "T")

    labelled = {"X": dict(unlabelled["X"], _folds={
        "effective_folds": 2, "requested_folds": 5,
        "folds_ran": [("2026-05-01", 1401, 854)],
        "folds_skipped": [("2025-08-01", 0, 0, "EMPTY TRAIN")],
        "span": "2026-05-01 .. 2026-06-30"})}
    rcb.show(labelled, "T")      # must not raise


def test_the_SKIP_REASON_distinguishes_an_EMPTY_train_from_a_SMALL_one():
    """`train<300` and `EMPTY TRAIN` are different facts: one says the window
    is thin, the other says the window does not exist. Three of the five
    recorded cutoffs were the second."""
    import inspect

    import scripts.run_clean_baseline as rcb

    src = inspect.getsource(rcb.run_market)
    assert '"EMPTY TRAIN" if not train else' in src
    assert '"train<300"' in src and '"no test row"' in src


def test_the_CLEAN_BASELINE_DOC_carries_the_correction():
    doc = pathlib.Path("docs/stage4-clean-baseline-2026-08-07.md").read_text(
        encoding="utf-8")
    assert "EFFECTIVE FOLDS: 2 of 5" in doc, (
        "the clean-baseline doc no longer states its effective fold count")
    assert "2,671" in doc and "2,700" in doc, (
        "the reconstruction's cross-check against the recorded manifest is "
        "gone, so the measurement cannot be audited")
    # WHITESPACE-NORMALISED: the phrase wraps across a line with a `> `
    # blockquote prefix in between, so a raw substring test fails on the
    # formatting rather than on the content.
    import re as _re

    flat = _re.sub(r"\s+(>\s+)?", " ", doc)
    assert "temporal generality" in flat, (
        "the doc does not say what two adjacent folds cannot support")
    # and the wrong claim must not stand unqualified
    body = doc.split("CORRECTED 2026-10-19", 1)
    assert len(body) == 2, "the correction marker is gone"
    assert "five chronological cutoffs" not in body[0], (
        "the uncorrected claim is still above the correction")


def test_the_DECISIVE_AUDIT_carries_its_own_addendum():
    doc = pathlib.Path("docs/predictive-audit-2026-08-07.md").read_text(
        encoding="utf-8")
    assert "EFFECTIVE FOLDS FOR EVERY PRICED FIGURE: 3 of 5" in doc
    assert "bookmaker_blend_weight" in doc.split("ADDENDUM 2026-10-19", 1)[1][:3000], (
        "the addendum does not name the standing recommendation that rests on "
        "the priced population")


def test_the_RECORDED_snapshots_are_UNTOUCHED():
    """A labelling correction does not edit a figure. `b2` holds."""
    from src.evaluation.baseline import BASELINE_REVISION

    assert BASELINE_REVISION == "b2"
    clean = json.loads(pathlib.Path("data/baselines/clean-20260807.json")
                       .read_text(encoding="utf-8"))
    assert clean["baseline_revision"] == "b1.unstamped-pre-exclusion"
    # the headline figure, unchanged
    assert round(clean["phase4_1x2"]["B de-vigged consensus"]["log_loss"],
                 4) == 1.0012
    assert clean["manifest"].count("matches qualified") == 1
    assert "2700" in clean["manifest"] or "2,700" in clean["manifest"]


# ── PART C: REPORT ONLY ────────────────────────────────────────────────────
#
# TWO SITES TAKE AN UNVERIFIED FIRST RESULT.
#
# 1. `apifootball_scraper` `/teams?search=` -> `teams_resp[0]["team"]["id"]`,
#    written persistently to `Team.apifootball_team_id` with NO name and NO
#    country comparison. `measured` 2026-10-19: 1,375 of 1,467 teams carry an
#    id. **How many came from this path is UNRECORDED** -- there is no
#    provenance column -- and the last 12 daily-picks runs show ZERO
#    "Resolved N team IDs via /teams?search" lines. So the population at risk
#    is bounded above by 1,375 and cannot be enumerated; by this project's own
#    rule it is therefore not cited as a count.
#
# 2. `weather_service._geocode` -> `results[0]`. Two different fallbacks:
#    with a country hint it fetches 3 results and prefers a country match
#    (reason on file: Dunfermline resolving to Illinois); with NO hint it
#    fetches ONE and uses it unconditionally. `measured` 2026-10-19: 27 of the
#    30 configured leagues carry a hint; the 3 that do not are the
#    PAN-EUROPEAN competitions, where a single country hint would be wrong by
#    construction -- **251 (venue, league) pairs, 419 fixtures, 184 distinct
#    venues** resolve with no verification at all.
#
# THE SPOT-CHECKS, each with its draw rule stated BEFORE the result:
#
#   N=12, the most frequent venues      -> 0 WRONG
#   N=12, the least frequent venues in  -> 1 WRONG, 2 unresolvable
#         the 3 pan-European leagues
#
# The first sample answered a different question than it looked like: ranking
# by population saves every major city, and every one of those 12 had US
# homonyms ranked below it. The draw rule I pre-committed to was biased toward
# the easy cases, so a second sample was drawn FROM THE TAIL with its own
# rule stated first, rather than re-drawing the first one.
#
# THE ONE WRONG RESULT: **'Bar' -> Qionghai, CN** (19.242, 110.464), population
# 528,238, while Bar, Montenegro (42.094, 19.098) ranks FOURTH. 94 degrees
# apart. Measured effect on that fixture's features, 2026-10-22:
#
#   weather_temp_c       29.3  against  19.7   ->  +9.6 C
#   weather_precip_mm     7.2  against  17.8   -> -10.6 mm
#   weather_wind_kmh     12.4  against  12.6   ->  -0.2
#   weather_is_raining      1  against      1   ->   0  (both above 1.0mm)
#   weather_is_windy        0  against      0   ->   0
#
# The two continuous features move materially; both binary flags happen to
# agree on that date. 'Biisen' and 'Hajvalia' return NO results and fall to
# `_DEFAULTS` with `weather_available: 0`, which is the honest outcome.

def test_the_geocoder_IS_GONE_so_the_census_is_HISTORY():
    """RETIRED-AND-REPLACED 2026-10-20. It pinned `results[0]` and the
    three-result country preference so the reported-not-fixed note could not
    go stale. The producer was deleted instead: the census (3 of 185 venues
    wrong, 4 fixtures, ZERO realised effect because weather reached no model,
    no prompt and no calculator) is now a record of what was removed.

    The OTHER unverified first result -- API-Football's `/teams?search=` --
    is NOT deleted and keeps its own check below: it writes
    `Team.apifootball_team_id` persistently, so it has a consumer.
    """
    assert not pathlib.Path("src/features/weather_service.py").exists(), (
        "the geocoder is back; re-measure the 185-venue census before "
        "relying on the figures in this file")


def test_the_TEAM_SEARCH_site_still_takes_the_first_result_UNVERIFIED():
    import pathlib as _p

    src = _p.Path("src/scrapers/apifootball_scraper.py").read_text(
        encoding="utf-8")
    assert 'api_team_id_val = teams_resp[0].get("team", {}).get("id")' in src, (
        "the /teams?search= first-result path changed; re-size it")


def test_the_TEAM_SEARCH_site_has_NO_PROVENANCE_and_that_is_the_finding():
    """An entity whose origin is unrecorded cannot be audited.

    `Team` has no column saying HOW `apifootball_team_id` was obtained, so the
    count of ids resting on an unverified search cannot be produced from the
    database at all. That is why this is reported as a bound and not as a
    count.
    """
    from src.data.models import Team

    cols = set(Team.__table__.columns.keys())
    assert "apifootball_team_id" in cols
    assert not any("provenance" in c or "source" in c for c in cols), (
        "a provenance column now exists -- then the count IS producible and "
        "must be measured rather than bounded")


def test_the_LEAGUE_COUNTRY_MAP_went_with_the_geocoder():
    """RETIRED-AND-REPLACED 2026-10-20. It checked that exactly the three
    pan-European competitions lacked a country hint -- the population that
    geocoded on `results[0]` with no verification at all. The map existed
    ONLY to disambiguate geocoding, so it was deleted with the producer; had
    it been left behind it would have been a fresh instance of the very
    defect this stage removed.
    """
    from src.features.feature_engineer import FeatureEngineer as FE

    assert not hasattr(FE, "_league_country_code")
    assert not hasattr(FE, "_LEAGUE_COUNTRY_CODES")
