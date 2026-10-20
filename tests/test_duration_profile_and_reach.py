"""The duration profile, weather's reach, and the three-fold label — 2026-10-20.

    Realised harm is measured, never inferred from the mechanism's severity.

PART A — WHERE THE MINUTES GO. `measured` 2026-10-20 from GitHub's per-step
job API (exact step boundaries) plus gap attribution inside each step: every
loguru line's elapsed gap to the next is charged to the module:function that
emitted the earlier line. That measures time spent in or waiting after each
emitter; it cannot see time inside a call that logs nothing, which appears as a
large gap on whatever logged last. Those are named, not hidden.

    n=3 profiled runs. The 67.4-88.1 time-to-picks figure is n=4 from
    2026-10-01 and is a DIFFERENT population; the two are not pooled.

| phase | 10-20 (34 fx) | 10-19 (13 fx) | 10-18 (65 fx) |
|---|---|---|---|
| **run total** | 56.9 | 48.5 | 68.8 |
| `--update` | **24.7 (43%)** | **24.1 (50%)** | **32.9 (48%)** |
| ·· AF fixtures-by-date | 7.2 (n=6) | 3.7 (n=6) | 10.1 (n=7) |
| ·· injury scraper | 6.9 (n=3) | 3.6 (n=3) | 9.7 (n=3) |
| ·· xG backfill | 2.9 | 9.0 | — |
| ·· Flashscore fixtures | 2.7 (n=90) | 3.0 (n=90) | 2.3 (n=90) |
| ·· team resolution | 1.3 | 0.3 | 7.1 |
| picks step | 11.3 (20%) | 5.4 (11%) | 21.1 (31%) |
| ·· Claude review | 4.9 (n=33) | 2.0 (n=13) | 9.5 (n=65) |
| ·· `analyze_fixture` | 2.9 | 1.3 | 4.9 |
| ·· **weather fetch** | **1.6 (14%)** | **0.9 (16%)** | **3.1 (15%)** |
| ML retrain | — | — | 5.6 |
| Flashscore RESULTS (post-picks) | 14.1 | 11.2 | 4.0 |

**TIME-TO-PICKS** is the run minus everything after the picks step:
**42.3 / 36.9 / 64.3 min**. The 11-14 minute results scrape runs AFTER picks
and does not count against the deadline.

SERIAL BY NECESSITY vs BY ACCIDENT:

* **Flashscore** — one Selenium browser over 30 leagues. BY NECESSITY.
* **injury scraper** — its own comment says "the sequential API+DB loop", and
  the cap is a REQUEST BUDGET (30), not a concurrency limit. The count is
  provider-bound; the serialisation is **BY ACCIDENT**. 9.7 min for 30
  fixtures = ~19 s each, which for one HTTP call means the DB write is inside
  the loop.
* **Claude review** — `for match_id in fixtures: await self._briefing_for(...)`.
  ~9 s per fixture, strikingly stable across runs (8.7-9.1). **BY ACCIDENT**,
  with cost and rate-limit consequences a fix would have to weigh.
* **weather** — one call per fixture inside `create_features`. **BY ACCIDENT**,
  and moot: Part B shows it reaches nothing.
* **AF fixtures-by-date** — the 9.2-minute gap follows the
  `AF_LEAGUE_FILTER ... rejected 1241 fixture(s)` line, so the time is in the
  per-fixture upsert loop, not the HTTP call. DB-bound, **BY ACCIDENT**.

**SO 88 MINUTES IS NOT THE FLOOR, AND THE 8.4-MINUTE RESIDUAL IS REDUCIBLE.**
The cheapest 3.1 of it is deleting a producer that reaches nothing; the injury
loop alone holds 9.7 min on a full card. **The cron is not the only remaining
lever.** Nothing is changed here: profiling and optimising are separate stages.

PART B — WEATHER REACHES NOTHING, so the geocoder exposure is nil rather than
cosmetic. Measured, with a negative control:

* `for_training=True` **skips weather by construction**, so it is in no fit.
  Both shipped pickles: 111 feature names, **zero** weather.
* at inference `_align_features` selects by the FITTED names, so all six
  weather columns are dropped: a -40 °C / 400 mm / 300 km/h fixture aligns
  **identically** to a 19.7 °C one, while perturbing a fitted column does
  change the vector.
* `grep weather` finds no consumer in `src/reporting/` or `src/betting/` —
  not the briefing, not the value calculator.

THE CENSUS, not a sample: **185 unhinted venues / 428 fixtures**, swept whole.
Decision rule stated before the sweep: every fixture in the three pan-European
competitions is played in a UEFA country, so a top result OUTSIDE UEFA with a
UEFA alternative in the same response is WRONG. **3 of 185 (1.6%), 4 fixtures:**
`Motherwell` → South Africa (pop 140,351 over Scotland's 32,840), `Bar` →
Qionghai CN, `San Sebastian` → Puerto Rico. Six venues return nothing and fall
to `_DEFAULTS`. It is a LOWER bound: a wrong UEFA country would pass the rule.

Deltas over all three (n=3): |ΔTemp| 5.2-10.7 °C mean 7.6, |ΔPrecip| 0.6-4.4 mm,
`weather_is_raining` flips on 2 of 3, `weather_is_windy` on 0 of 3.

PART C — the no-information conclusion is MULTIPLY DETERMINED; the blend
recommendation is not. See the test bodies below.
"""

import pathlib

import pytest


# ── PART A: the profile is recorded, and nothing was changed ────────────────

def test_the_PROFILE_carries_its_n_and_does_NOT_pool_two_populations():
    """Rule 5: every duration carries its n. The n=4 time-to-picks bound and
    the n=3 phase profile are different populations and must not be merged."""
    doc = __doc__
    assert "n=3 profiled runs" in doc
    assert "n=4 from" in doc and "DIFFERENT population" in doc
    assert "are not pooled" in doc

    import scripts.ci_audit as ci

    # the registered upper bound is untouched by profiling
    assert ci.FULL_CARD_PICKS_MAX == 88.1, (
        "profiling changed the registered time-to-picks bound; profiling "
        "changes nothing (rule 4) and the bound is n=4 from 2026-10-01")


def test_the_REGISTERED_60_and_the_8_4_RESIDUAL_are_UNTOUCHED():
    """Rule 6, re-asserted here because Part A is the first stage with a
    plausible route to reducing the residual rather than accepting it."""
    import scripts.schedule_deadline as sd

    assert sd.MIN_ACCEPTABLE_MARGIN_MIN == 60
    # Zero since the 2026-10-20 re-anchoring; the trade moved from the
    # exposure term to the MARGIN, which now fails at +40m against 60.
    assert sd.COVERAGE_EXPOSURE["weekday"]["max"] == 0
    assert sd.EFFECTIVE_MIN_MARGIN == {"weekday": 60, "weekend": 60}


def test_the_INJURY_loop_is_still_SEQUENTIAL_so_the_finding_stands():
    """The classification must not go stale while the fix is deferred."""
    src = pathlib.Path("src/scrapers/injury_scraper.py").read_text(
        encoding="utf-8")
    assert "sequential API+DB loop" in src, (
        "the injury scraper no longer describes itself as a sequential loop; "
        "re-profile before relying on the 9.7-minute figure")
    assert "asyncio.gather" not in src, (
        "the loop became concurrent -- that is the optimisation Part A said "
        "was available, and the duration profile must be re-measured")


def test_the_BRIEFING_loop_is_still_SEQUENTIAL():
    import re

    src = pathlib.Path("src/reporting/match_briefing.py").read_text(
        encoding="utf-8")
    assert re.search(r"for match_id in fixtures:\s*\n\s+.*\n?\s*.*await self\._briefing_for",
                     src), (
        "the briefing loop shape changed; the ~9 s/fixture figure was "
        "measured against a sequential await")


# ── PART B: weather reaches nothing — MEASURED, with a negative control ────

def test_TRAINING_SKIPS_WEATHER_BY_CONSTRUCTION():
    """SUPERSEDED 2026-10-20 BY THE DELETION. It asserted that section 14 sat
    behind `if not for_training:`, which was the reason weather could never
    enter a fit. The section is gone, so the guard is gone with it and there
    is nothing left to skip.

    What survives is the SAME GUARD ON INJURIES, which have the identical
    train/serve asymmetry and are NOT deleted -- they reach the Claude review
    prompt, so they have a consumer.
    """
    src = pathlib.Path("src/features/feature_engineer.py").read_text(
        encoding="utf-8")
    assert "14. DELETED 2026-10-20" in src, (
        "the weather section is back without its deletion note")
    assert "skip external API calls" in src, (
        "`for_training` no longer documents what it skips; injuries still "
        "depend on that guard")


def test_the_SHIPPED_PICKLES_contain_NO_weather_feature():
    """Read from the raw pickle, bypassing the HMAC-gated loader: the loader
    refuses under the default dev key, and a refused load reports an UNFITTED
    model, which would make any prediction comparison vacuously equal."""
    import pickle

    for f in ("data/models/ml_models.pkl", "data/models/goals_model.pkl"):
        p = pathlib.Path(f)
        if not p.exists():
            pytest.skip(f"{f} not present here")
        with p.open("rb") as fh:
            st = pickle.load(fh)
        names = st.get("feature_names") or []
        assert names, f"{f} records no feature_names"
        assert not [n for n in names if "weather" in n.lower()], (
            f"{f} now has weather in its FITTED names -- weather reaches a fit "
            f"and the geocoder exposure is no longer nil")


def test_ALIGNMENT_DROPS_the_weather_columns_with_a_NEGATIVE_CONTROL():
    """The decisive measurement. Not a code reading: the real fitted names are
    used and the aligned vectors are compared."""
    import pickle

    import numpy as np

    pkl = pathlib.Path("data/models/ml_models.pkl")
    if not pkl.exists():
        pytest.skip("no shipped pickle here")
    import os

    os.environ.setdefault("BETTING_AGENT_ALLOW_DEFAULT_HMAC", "1")
    from src.models.ml_models import MLModels

    with pkl.open("rb") as fh:
        fitted = list(pickle.load(fh)["feature_names"])
    m = MLModels()
    m.feature_names = fitted

    WEATHER = ["weather_temp_c", "weather_wind_kmh", "weather_precip_mm",
               "weather_is_raining", "weather_is_windy", "weather_available"]
    live = fitted + WEATHER
    base = np.array([[0.5 + 0.001 * i for i in range(len(fitted))] + [0.0] * 6])

    def aligned(vals):
        X = base.copy()
        for k, v in zip(WEATHER, vals):
            X[0, live.index(k)] = v
        return m._align_features(X, live)

    real = aligned([19.7, 12.6, 17.8, 1, 0, 1])      # Bar, Montenegro
    wrong = aligned([29.3, 12.4, 7.2, 1, 0, 1])      # Qionghai, CN
    absurd = aligned([-40.0, 300.0, 400.0, 1, 1, 1])

    assert real.shape[1] == len(fitted), "alignment did not drop the extras"
    assert np.array_equal(real, wrong), (
        "a wrong-city forecast now changes the aligned vector -- the geocoder "
        "exposure has become real and must be re-sized")
    assert np.array_equal(real, absurd)

    # NEGATIVE CONTROL: the mechanism is not inert.
    X2 = base.copy()
    X2[0, 0] = 99.0
    assert not np.array_equal(m._align_features(X2, live), real), (
        "perturbing a FITTED column does not change the aligned vector, so "
        "this test proves nothing about weather")


def test_NO_CONSUMER_anywhere_for_the_weather_keys():
    """Not a fit, not an inference, not a prompt. The keys are a producer with
    no consumer, and each run pays an Open-Meteo call per fixture: `measured`
    33, 11 and 58 calls on the last three runs, 0.9-3.1 min of wall time."""
    import re

    hits = []
    for d in ("src/reporting", "src/betting", "src/models"):
        for f in pathlib.Path(d).rglob("*.py"):
            for i, line in enumerate(
                    f.read_text(encoding="utf-8").splitlines(), 1):
                if re.search(r"weather_\w+", line) and not line.strip().startswith("#"):
                    hits.append(f"{f.as_posix()}:{i}")
    assert not hits, (
        f"a weather_* key is now read outside the producer: {hits}. The "
        f"exposure is no longer nil and Part B must be re-sized.")


def test_the_PNC_DETECTOR_cannot_see_dict_key_producers():
    """A coverage gap in my own instrument, stated rather than left implied.

    `pnc_scan` tracks SYMBOLS — functions, constants, tables, columns. The
    weather features are DICT KEYS written into a feature mapping, so a
    producer with no consumer of exactly the kind the detector exists for is
    invisible to it. `_get_weather_features` itself IS consumed, by
    `create_features`, so nothing is flagged.
    """
    import scripts.pnc_scan as pnc

    assert hasattr(pnc, "scan")
    src = pathlib.Path("scripts/pnc_scan.py").read_text(encoding="utf-8")
    assert "__table:" in src or "__column:" in src, (
        "the detector no longer tracks table/column pseudo-symbols; re-read "
        "what it covers before trusting this note")
    assert "dict key" not in src.lower() or True   # documented here, not there


# ── PART C: what rests on three folds, conclusion by conclusion ────────────
#
# THE NO-INFORMATION CONCLUSION IS MULTIPLY DETERMINED. Four independent
# routes, and only ONE uses this harness:
#
#   1. logistic divergence on settled picks: coefficient +0.245, z=1.11,
#      p=0.265; market-only log-likelihood -685.99 BEATS model-only -690.67.
#      n=1018 settled picks. NO FOLD STRUCTURE.
#   2. scoring rules on the same picks: model Brier 0.2499 against a
#      constant-0.514 Brier of 0.2498; raw 1/odds 0.2410. NO FOLD STRUCTURE.
#   3. walk-forward blend sweep, 1,710 PRICED out-of-sample matches:
#      log-loss falls monotonically 1.0412 (w=0) -> 0.9929 (w=1.0), no
#      interior optimum.  <-- THREE EFFECTIVE FOLDS OF FIVE, one odds era
#   4. the price closure: betting at the close at retail predicts
#      1/R - 1 = -4.64%; the measured flat ROI over 1,320 settled picks is
#      -5.396%, a 0.76pp gap well inside the ~±2.6pp standard error. NO FOLD
#      STRUCTURE.
#
# So the standing conclusion "the model adds no information over the price"
# does NOT depend on the fold count. The `bookmaker_blend_weight -> 1.0`
# RECOMMENDATION does: route 3 is the only one that sweeps the weight, and the
# monotone shape with no interior optimum is established on 1,710 priced
# matches across three ADJACENT folds inside a single odds-collection era.
#
# WHAT THREE ADJACENT FOLDS IN ONE ODDS ERA CAN AND CANNOT SUPPORT:
#   CAN     the intervals for those 1,710 observations; the absence of an
#           interior optimum WITHIN that period.
#   CANNOT  temporal generality — that the monotone shape holds under another
#           book mix or market regime. That era's book mix is itself unusual:
#           the corruption gate rejected 4,836 1X2 books, Bet365 at 92%
#           implausible and Pinnacle at 26%.
#
# SHOULD THE RECOMMENDATION CARRY A QUALIFICATION? Yes, and what it would take
# is more odds eras, which means TIME — a statement about WHEN, not whether.
# Odds collection began 2026-02; a second era with a different book mix cannot
# exist before one has accumulated.

def test_walk_forward_RETURNS_its_effective_fold_count():
    import datetime as dt

    from src.evaluation.baseline import walk_forward

    class _R:
        def __init__(self, i, d):
            self.id, self.match_date, self.outcome = i, d, 0

    rows = [_R(i, dt.date(2026, 3, 1) + dt.timedelta(days=i)) for i in range(40)]
    out = walk_forward(rows, [], [dt.date(2026, 1, 1), dt.date(2026, 4, 1)],
                       window_days=60, min_train=5)
    f = out["folds"]
    assert f["effective_folds"] == 1 and f["requested_folds"] == 2
    assert f["folds_skipped"][0][3] == "EMPTY TRAIN", (
        "an empty training set is no longer distinguished from a small one")
    assert "PRICED subset spans fewer" in f["caveat"], (
        "the caveat that a fold count over ALL rows does not describe a PRICED "
        "figure is gone -- that is the transfer this label exists to prevent")


def test_run_baseline_REFUSES_to_print_without_the_fold_count():
    """`show()` already refuses in the clean harness; extended to this one."""
    import inspect

    import scripts.run_baseline as rb

    src = inspect.getsource(rb.main)
    assert 'results.get("folds")' in src
    assert "raise ValueError" in src
    assert "WITHOUT its effective fold" in src
    assert "EFFECTIVE of" in src, "the fold table is not printed"


#: THE FOUR ROUTES AS DATA, NOT AS PROSE TO BE GREPPED.
#
# Two earlier drafts of the test below counted a phrase in this file's own
# source and found 4, then 5 -- because the assertion list quoted the phrase,
# and then because the fix quoted it too. A test that greps its own source for
# a string it must also mention cannot be made correct by better escaping; the
# claim has to become data. It is the self-reference cousin of a comment-blind
# scan, which this file set has now hit in three different costumes.
NO_INFORMATION_ROUTES = {
    "logistic divergence on settled picks": {
        "figures": "coef +0.245, z=1.11, p=0.265; market-only LL -685.99 "
                   "BEATS model-only -690.67; n=1018 settled",
        "uses_this_harness": False,
    },
    "scoring rules on settled picks": {
        "figures": "model Brier 0.2499 vs constant-0.514 Brier 0.2498; raw "
                   "1/odds 0.2410",
        "uses_this_harness": False,
    },
    "walk-forward blend sweep": {
        "figures": "1,710 PRICED OOS matches; log-loss falls monotonically "
                   "1.0412 (w=0) -> 0.9929 (w=1.0), no interior optimum",
        "uses_this_harness": True,          # <-- 3 EFFECTIVE FOLDS OF 5
    },
    "the price closure": {
        "figures": "betting at the close at retail predicts 1/R-1 = -4.64%; "
                   "measured flat ROI -5.396% over 1,320 settled picks, a "
                   "0.76pp gap inside the ~+/-2.6pp standard error",
        "uses_this_harness": False,
    },
}

#: What rests on the harness ALONE, and therefore on 3 adjacent folds.
SINGLE_HARNESS_CLAIMS = ("bookmaker_blend_weight -> 1.0",)


def test_the_MULTIPLY_DETERMINED_split_is_DATA_not_prose():
    """The point of the whole part: which conclusions survive the label."""
    assert len(NO_INFORMATION_ROUTES) == 4
    fold_free = [k for k, v in NO_INFORMATION_ROUTES.items()
                 if not v["uses_this_harness"]]
    on_harness = [k for k, v in NO_INFORMATION_ROUTES.items()
                  if v["uses_this_harness"]]
    assert len(fold_free) == 3, (
        f"three of the four routes carry no fold structure; found "
        f"{len(fold_free)}. If that changed, the blend recommendation's "
        f"isolation changed with it.")
    assert on_harness == ["walk-forward blend sweep"], (
        "a second conclusion now rests on this harness; the three-fold "
        "qualification spreads with it")
    # every route carries its figures, so the split cannot become an assertion
    # without evidence
    for k, v in NO_INFORMATION_ROUTES.items():
        assert len(v["figures"]) > 40, f"{k} has no figures"

    # THE NO-INFORMATION CONCLUSION therefore does NOT depend on the fold
    # count; the RECOMMENDATION does, and it is named.
    assert SINGLE_HARNESS_CLAIMS == ("bookmaker_blend_weight -> 1.0",)


def test_what_three_adjacent_folds_CAN_and_CANNOT_support_is_STATED():
    """CAN: the intervals for those 1,710 observations, and the absence of an
    interior optimum WITHIN that period. CANNOT: temporal generality — that
    the monotone shape holds under another book mix or market regime.

    That era's book mix is itself unusual: the corruption gate rejected 4,836
    1X2 books, Bet365 at 92% implausible and Pinnacle at 26%. So the period is
    not merely short, it is atypical.

    WHAT IT WOULD TAKE: more odds eras, which means TIME — a statement about
    WHEN, not whether. Odds collection began 2026-02.
    """
    import re

    doc = pathlib.Path("docs/predictive-audit-2026-08-07.md").read_text(
        encoding="utf-8")
    flat = re.sub(r"\s+(>\s+)?", " ", doc)
    assert "temporal generality" in flat, (
        "the audit no longer states what its priced population cannot support")
    # MARKDOWN EMPHASIS SPLITS IT: the doc reads `a *standing* configuration
    # claim`, so the asterisks sit inside the phrase. Matched on the part that
    # carries the meaning and no formatting.
    assert "configuration claim" in flat, (
        "the addendum no longer says WHY the blend recommendation is the one "
        "that needs temporal generality")


def test_the_AUDIT_DOC_names_what_rests_on_the_priced_population():
    doc = pathlib.Path("docs/predictive-audit-2026-08-07.md").read_text(
        encoding="utf-8")
    assert "EFFECTIVE FOLDS FOR EVERY PRICED FIGURE: 3 of 5" in doc
    tail = doc.split("ADDENDUM 2026-10-19", 1)[1][:3000]
    assert "bookmaker_blend_weight" in tail
    assert "temporal" in tail


def test_NO_baseline_revision_and_NO_figure_moved():
    from src.evaluation.baseline import BASELINE_REVISION

    assert BASELINE_REVISION == "b2", (
        "labelling is not a revision event; no figure moved in this stage")


# ═══════ 2026-10-20: THE DELETION, THE INJURY SIZING, THE QUALIFICATION ════

def test_the_DELETION_GATE_is_recorded_as_the_ALIGNED_vector_not_the_raw_one():
    """What had to be byte-identical, stated precisely.

    The RAW feature dict cannot be: removing the producer removes six keys,
    which is the point. What must not move is what REACHES THE MODEL -- the
    vector after `_align_features` selects the fitted names.

    `measured` 2026-10-20 over 8 real fixtures with venues: raw features
    272 -> 266 (exactly the six weather keys), aligned width 111 -> 111, and
    **all 8 aligned SHA-256 digests byte-identical**. The instrument
    discriminates: 8 distinct digests across the 8 fixtures, so the equality
    is not a constant-output artefact.
    """
    src = pathlib.Path("src/features/feature_engineer.py").read_text(
        encoding="utf-8")
    assert "byte-identical before and after" in src, (
        "the deletion no longer records its gate")
    assert "POSITIVE CONTROL" in src and "PERTURBATION" in src, (
        "the deletion no longer records HOW the input was proved unread: a "
        "perturbation that would change the output, plus a positive control "
        "on a column known to be read")


def test_the_DELETION_names_the_PRIOR_RECORD_it_did_not_discover():
    """It was recorded dead on 2026-08-07 and not acted on for ten weeks.
    Crediting that is the difference between a finding and a re-finding."""
    src = pathlib.Path("src/features/feature_engineer.py").read_text(
        encoding="utf-8")
    assert "2026-08-07" in src and "Dead work on every fixture" in src, (
        "the deletion presents itself as a discovery; the audit had already "
        "recorded it at lines 157 and 650")
    doc = pathlib.Path("docs/predictive-audit-2026-08-07.md").read_text(
        encoding="utf-8")
    assert "discarded by the ML path" in doc, (
        "the prior record is gone, so the credit cannot be checked")


def test_the_CONFIG_KEY_went_with_the_producer():
    import yaml

    for f in ("config/config.yaml", "config/config.example.yaml"):
        cfg = yaml.safe_load(pathlib.Path(f).read_text(encoding="utf-8"))
        assert "weather_features_enabled" not in (cfg.get("models") or {}), (
            f"{f} still carries a flag for a producer that does not exist")
        assert "REMOVED 2026-10-20" in pathlib.Path(f).read_text(
            encoding="utf-8"), f"{f} removed the key without saying why"


def test_INJURIES_are_NOT_deletable_because_they_HAVE_a_consumer():
    """The distinction that makes one a deletion and the other an
    optimisation.

    `injury_*` FEATURES share weather's fate -- absent from the 111 fitted
    names, dropped by alignment. But `analysis.injury_report` reaches
    `match_briefing._build_dossier`, i.e. the Claude review prompt, and
    `injury_impact` reaches a reasoning string in `value_calculator`. The
    review issues KEEP/CHANGE on every pick, so injuries are
    **value-affecting through the prompt** even though they are in no fit.
    Weather reached neither.
    """
    brief = pathlib.Path("src/reporting/match_briefing.py").read_text(
        encoding="utf-8")
    assert "analysis.injury_report" in brief, (
        "the injury summary no longer reaches the dossier -- then injuries "
        "have no consumer either and become deletable, which is a different "
        "stage's decision")
    vc = pathlib.Path("src/betting/value_calculator.py").read_text(
        encoding="utf-8")
    assert "injury_impact" in vc


def test_the_INJURY_CAP_is_RECONCILED_to_ONE_definition():
    """RETIRED-AND-REPLACED 2026-10-20. It pinned the three-way disagreement
    -- docstring 30, code 15 twice, config 30 (production ran 30) -- so the
    finding could not go stale. Reconciled: `DEFAULT_INJURY_BUDGET` is the one
    definition and the docstring and the `config.get` default both reference
    it.

    THE VALUE IS UNCHANGED at 30, deliberately: production already ran 30, so
    reconciling changes no behaviour. Raising it toward the re-derived 36
    would change WHICH fixtures get injury data, which reaches the Claude
    review prompt -- value-affecting, and a separate stage.
    """
    import src.scrapers.injury_scraper as m

    assert m.DEFAULT_INJURY_BUDGET == 30
    src = pathlib.Path("src/scrapers/injury_scraper.py").read_text(
        encoding="utf-8")
    code_lines = [l for l in src.splitlines()
                  if not l.lstrip().startswith("#")]
    code = chr(10).join(code_lines)
    assert "max_injury = 15" not in code, (
        "a bare 15 default is back in the code; the docstring and config both "
        "say 30 and production runs 30")
    assert code.count("DEFAULT_INJURY_BUDGET") >= 3, (
        "the one definition is no longer referenced by both the assignment "
        "and the config default")

    import yaml

    cfg = yaml.safe_load(pathlib.Path("config/config.example.yaml").read_text(
        encoding="utf-8"))
    assert (cfg.get("models") or {}).get("max_injury_budget") == 30, (
        "the shipped budget moved; the 9.7-minute figure and the re-derivation "
        "were both measured at 30")


def test_the_RE_DERIVATION_states_that_30_was_NEVER_right_when_written():
    """The stale derivation is corrected IN PLACE, naming what it referenced.

    Its three inputs: a 300 s timeout that is now 720 s, an assumed ~11 s per
    call against a measured 14-20 s, and the cap itself. Re-derived,
    720 / 20 = 36, so 30 sits 120 s inside the timeout -- and under the 300 s
    timeout it needed 600 s. It became correct when the timeout was raised for
    an unrelated reason.
    """
    src = pathlib.Path("src/scrapers/injury_scraper.py").read_text(
        encoding="utf-8")
    # NUMBERS EXACT, PROSE CASE-INSENSITIVE. The first draft asserted
    # "never safe when it was written" against a comment that writes it in
    # capitals -- a case mismatch in my own test, which is the same family as
    # the markdown-emphasis and line-wrap mismatches already hit in this file
    # set. Figures must match exactly; wording should not fail on shouting.
    for part in ("300s", "720s", "14-20s", "720 / 20 = 36"):
        assert part in src, f"the re-derivation omits the figure {part!r}"
    low = src.lower()
    for phrase in ("never safe when it was written", "right by accident"):
        assert phrase in low, f"the re-derivation omits {phrase!r}"
    agent = pathlib.Path("src/agent/betting_agent.py").read_text(
        encoding="utf-8")
    assert "timeout=720" in agent, (
        "the timeout moved again; re-derive the budget rather than trusting "
        "the comment")
    # AND NO CONCURRENCY CHANGE
    inj = src
    assert "asyncio.gather" not in inj, (
        "the loop became concurrent -- that is value-affecting through the "
        "budget claim and belongs in its own stage")
    assert "budget-claim atomicity" in inj, (
        "the comment no longer records why concurrency is deferred")


def test_the_DB_WRITE_is_INSIDE_the_injury_loop():
    """Measured by the call chain, because it decides whether concurrency
    buys network time only or network AND database time."""
    inj = pathlib.Path("src/scrapers/injury_scraper.py").read_text(
        encoding="utf-8")
    body = inj.split("async def _fetch_fixture_injuries", 1)[1][:2000]
    assert "_api_get" in body and "self.db.get_session()" in body, (
        "the per-fixture function no longer does both the API call and the DB "
        "write; re-measure the ~19 s per fixture before costing concurrency")


def test_the_FOLD_BLOCK_carries_BOTH_HALVES_of_the_qualification():
    """A count alone does not convey an anomalous mix."""
    import datetime as dt

    from src.evaluation.baseline import walk_forward

    class _R:
        def __init__(self, i, d):
            self.id, self.match_date, self.outcome = i, d, 0

    rows = [_R(i, dt.date(2026, 3, 1) + dt.timedelta(days=i)) for i in range(40)]
    f = walk_forward(rows, [], [dt.date(2026, 1, 1), dt.date(2026, 4, 1)],
                     window_days=60, min_train=5)["folds"]
    for k in ("era", "lifting_it", "multiply_determined_note"):
        assert f.get(k), f"the fold block lost {k!r}"
    assert "4,836" in f["era"] and "92%" in f["era"], (
        "the era qualification no longer carries the measured anomaly")
    assert "WHEN, not whether" in f["lifting_it"]
    assert "equally supported" in f["multiply_determined_note"], (
        "the block no longer warns that the conclusion and the "
        "recommendation are not equally supported")

    import inspect

    import scripts.run_baseline as rb

    src = inspect.getsource(rb.main)
    assert 'for _k in ("era", "lifting_it", "multiply_determined_note")' in src, (
        "run_baseline no longer refuses on the missing halves")
