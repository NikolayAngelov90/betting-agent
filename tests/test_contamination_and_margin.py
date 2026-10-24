"""The phantom exclusion's consumers, the measured schedule margin, and the
tool that gave retired advice.

STAGE 23, 2026-10-01.

THREE FINDINGS, ALL FROM THE SAME SHAPE — a rule recorded in one place and not
applied in another:

  A  `training_exclusion_reason = phantom_kickoff_now_stamp` marks 510 rows whose
     `match_date` is a `datetime.now()` stamp. `MatchHistory._base_filter()` is
     the single predicate that applies it, and Poisson, Elo and feature_engineer
     inherit it through `get_completed_matches`. `scripts/run_baseline.py` —
     the evidence bar every model parameter change must clear — queries `Match`
     directly, applies `is_fixture`/`home_goals`/`away_goals`/`match_date`, and
     does NOT apply the exclusion. Up to 17.3% of its population on a recent
     window is excluded rows, and `walk_forward` assigns train/test folds BY
     `match_date`, so a fabricated date does not merely add noise — it puts the
     match in the wrong fold.

  B  `2026-11-06` came from a fit with R^2 = 0.451. A fitted date cited bare
     becomes a measurement, so the audit now computes the margin from the live
     series every day and the date survives only as `simulated`.

  C  `cohort_status.py` printed `VERDICT: AMEND` for two weeks after `s5.14`
     retired that rule. An executable giving retired advice is consulted exactly
     when the decision is made.

These tests pin the reconciliation. They do not re-measure the database —
measurement belongs to the ledger entry.
"""

import datetime as dt
import importlib.util
import pathlib
import re

import pytest

_spec = importlib.util.spec_from_file_location(
    "_ci_audit_margin", pathlib.Path("scripts/ci_audit.py"))
ci = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ci)


# ── A: the exclusion's single definition, and the consumer that bypasses it ──

def test_the_single_predicate_EXISTS_and_is_not_duplicated_again():
    """A4: build a new one only if there is more than one consumer. There is one.

    `MatchHistory._base_filter()` already is the single definition. Adding a
    second would be THE HABIT's seventh instance, so this asserts the existing
    one is where the rule lives.
    """
    src = pathlib.Path("src/data/match_history.py").read_text(encoding="utf-8")
    assert "def _base_filter()" in src
    assert "Match.training_exclusion_reason.is_(None)" in src, (
        "the shared accessor stopped applying the exclusion — Poisson, Elo and "
        "feature_engineer all inherit it from here")


def test_run_baseline_APPLIES_the_exclusion_via_the_single_definition():
    """THE GAP, CLOSED 2026-10-01 (Stage 24). Was a pinned known defect.

    The previous version of this test asserted the exclusion was ABSENT and said
    it must be inverted when closed. This is that inversion — the gap could not
    be closed silently, and it was not forgotten.

    The predicate must come from `_HistoryCache._base_filter()`, not be
    re-typed: one rule, one definition. A hand-copied copy is how
    feature_engineer ended up with its own, and how three discovery phrasings
    happened.
    """
    src = pathlib.Path("scripts/run_baseline.py").read_text(encoding="utf-8")
    block = src.split("def load_rows(", 1)[1].split("\ndef ", 1)[0]
    assert "_HistoryCache._base_filter()" in block, (
        "run_baseline.py no longer applies the shared exclusion predicate — the "
        "evidence bar for every model change is admitting phantom rows again")
    assert "Match.training_exclusion_reason" not in block, (
        "the predicate was re-typed inline instead of imported. One definition: "
        "use _HistoryCache._base_filter()")


def test_the_baseline_carries_a_REVISION():
    """Part C: a changed evaluation population is a different experiment.

    Picks carry `model_version`; baselines carried nothing, so every correction
    to the evaluation set was irreversible instead of a bump. That is the only
    reason closing the gap ever looked like a choice.
    """
    from src.evaluation.baseline import BASELINE_REVISION, baseline_fingerprint
    assert BASELINE_REVISION == "b2"
    fp = baseline_fingerprint(exclusion="x", fold_strategy="y", window_days=60,
                              cutoffs=[dt.date(2026, 1, 1)], since="2022-01-01")
    assert fp.startswith("b2.") and len(fp.split(".")[1]) == 6
    src = pathlib.Path("scripts/run_baseline.py").read_text(encoding="utf-8")
    assert 'payload["baseline_revision"]' in src, (
        "the snapshot is written without a revision, so a corrected population "
        "is indistinguishable from a changed result")


def test_the_fingerprint_SEPARATES_the_three_things_that_define_a_population():
    """Exclusion, fold strategy and window must each move the digest.

    A fingerprint that ignores one of them silently pools two experiments — the
    `__code__`-in-TRACKED_KEYS lesson, one level out.
    """
    from src.evaluation.baseline import baseline_fingerprint
    base = dict(exclusion="a", fold_strategy="b", window_days=60,
                cutoffs=[dt.date(2026, 1, 1)], since="2022-01-01")
    ref = baseline_fingerprint(**base)
    for field, other in (("exclusion", "a2"), ("fold_strategy", "b2"),
                         ("window_days", 90), ("since", "2023-01-01"),
                         ("cutoffs", [dt.date(2026, 2, 1)])):
        alt = dict(base); alt[field] = other
        assert baseline_fingerprint(**alt) != ref, (
            f"changing {field} does not change the baseline revision")


def test_pre_fix_snapshots_are_STAMPED_not_deleted():
    """Mark, never delete — third instance, after the phantoms and the 81."""
    import json
    snaps = sorted(pathlib.Path("data/baselines").glob("*.json"))
    assert snaps, "the recorded baseline snapshots are gone"
    for p in snaps:
        d = json.loads(p.read_text(encoding="utf-8"))
        assert "baseline_revision" in d, f"{p.name} carries no revision"
        if d["baseline_revision"].startswith("b1"):
            assert "training_exclusion_reason" in d["baseline_revision_note"], (
                f"{p.name} is stamped b1 but does not say what b1 means")


def test_the_NOT_GATED_marker_convention_is_still_documented():
    """The convention that lets an ungated path assert something specific.

    A bare missing filter is indistinguishable from an oversight; the marker
    forces a copy-paste to claim one of three categories, so a wrong claim is
    visible.
    """
    src = pathlib.Path("src/data/match_history.py").read_text(encoding="utf-8")
    assert "training-exclusion: NOT GATED" in src
    for cat in ("populates", "repairs", "resolves"):
        assert cat in src


def test_every_marked_ungated_site_names_exactly_one_category():
    """A marker claiming two categories claims nothing."""
    cats = ("populates", "repairs", "resolves")
    for p in list(pathlib.Path("src").rglob("*.py")):
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if "training-exclusion: NOT GATED" not in line:
                continue
            if "<populates|repairs|resolves>" in line:
                continue                      # the convention's own definition
            named = [c for c in cats if f"({c})" in line]
            assert len(named) == 1, f"{p}:{i} names {named}: {line.strip()}"


# ── B: the margin is measured daily, and the fitted date never stands alone ──

def test_the_deadline_is_the_WEEKEND_kickoff_and_says_so():
    """The conservative deadline follows the localized weekend kickoff floor.

    The provenance below records why the earlier 10:04 estimate was invalid.
    """
    # RE-ANCHORED 2026-10-20: the deadline is the OBSERVED EARLIEST KICKOFF (weekday 09:30, weekend 09:45), not ceil_15(ready+60). The self-anchored floor was a TAUTOLOGY -- it made margin>=60 true by construction -- and freezing it against a live series breached 60 within a day when mean+3sd drifted +11.4m on one new observation.
    # 570 -> 540 ON 2026-10-22, AND THE TEST'S OWN NAME BECAME TRUE. The
    # deadline anchors were localised, and the weekend's is 12:00 local
    # against the weekday's 12:30 — so the WEEKEND is the earlier floor on
    # both sides of a DST transition. The retired UTC pair said the opposite
    # (weekday 09:30 < weekend 09:45), which was an artefact of two figures
    # measured in the same season on different days.
    assert ci.SCHEDULE_DEADLINE_MIN == ci.DEADLINE_MIN_BY_DAYTYPE["weekend"]
    # THE `[-1400:]` WINDOW IS GONE TOO, for the reason this file already
    # records one test below: a width is not an order. The localisation added
    # two small classes above the constant on 2026-10-22 and pushed both
    # provenance notes out of the window, failing a property that still held.
    # What is under test is that the deadline's provenance is RECORDED, so
    # the scan is over the module.
    src = pathlib.Path("scripts/ci_audit.py").read_text(encoding="utf-8")
    assert "sub-second" in src, (
        "the deadline no longer records that 10:04 came from phantom rows")
    assert "openfootball" in src, (
        "the deadline cites no INDEPENDENT source — verifying a figure against "
        "the data that produced it is not verification")


def test_the_threshold_is_declared_as_a_literal_below_one_sd():
    """Stated on its own terms, not fitted to the current margin."""
    assert ci.MARGIN_ALARM_MIN == 30
    assert ci.MARGIN_ALARM_MIN < 36.7, (
        "the threshold is at or above one sd of the observed delay, so a "
        "one-sigma-late day at the threshold would still make the card and the "
        "alarm fires too late to mean anything")


def test_the_margin_uses_max_AND_mean_plus_3sd_never_the_mean_alone():
    series = [(dt.date(2026, 10, 1), 300.0), (dt.date(2026, 10, 2), 360.0),
              (dt.date(2026, 10, 3), 420.0)]
    m = ci.schedule_margin(series, 0)
    assert "margin_max" in m and "margin_3sd" in m
    assert m["binding"] == min(m["margin_max"], m["margin_3sd"]), (
        "the binding margin is not the worse of the two — that is the mean "
        "creeping back in")
    # and the mean alone must never be the reported margin
    mean_margin = ci.SCHEDULE_DEADLINE_MIN - (0 + m["mean"] + ci.FULL_CARD_PICKS_3SD)
    assert m["binding"] < mean_margin


def test_an_EMPTY_series_is_UNEVALUATED_not_safe():
    """Fail closed: no runs under the current cron is not a healthy margin."""
    m = ci.schedule_margin([], 0)
    assert m["n"] == 0
    assert "alarm" not in m or m.get("alarm") is not True
    assert "UNEVALUATED" in ci.format_schedule_margin(m)


def test_the_margin_line_is_printed_UNCONDITIONALLY():
    """Every day, alarm or not — the AF_LEAGUE_FILTER lesson again."""
    src = pathlib.Path("scripts/ci_audit.py").read_text(encoding="utf-8")
    # 2026-10-10: the single `collect_lag_series()` call became a
    # per-day-type pair when the cron split. The property under test is
    # UNCHANGED - the margin prints before anything can branch on `alarm` -
    # so the anchor moves and the assertion does not.
    # anchor on the CALL SITE in main(), not the definition: the helpers
    # that build these series are defined above it and share the name.
    # The anchor has moved twice: the single `collect_lag_series()` call
    # became a per-day-type pair when the cron split, and the
    # discovered-margin block that briefly followed it was removed on
    # 2026-10-11 (it needed a database this step does not have). The
    # PROPERTY is unchanged throughout - the margin prints before anything
    # can branch on `alarm`.
    # AND THE WINDOW IS GONE. `[:1400]` was a third moving part: on
    # 2026-10-11 a comment added between the anchor and the print pushed
    # "alarm" past 1400 characters and the test failed with
    # `ValueError: substring not found` — a FALSE RED on a property that
    # still held. An ordering assertion needs an order, not a width.
    #
    # AND NOW IT IS COMMENT-BLIND, which ends the class rather than paying
    # for it a third time. On 2026-10-22 a comment inserted between the
    # anchor and the print contained the word "alarm" — so the test reported
    # that the margin only prints when it alarms, on the strength of a
    # sentence saying the opposite. The PROPERTY is about code; the scan now
    # reads only code.
    from tests.conftest import code_of
    src = code_of(src)
    body = src.split("_by_type, _crons = collect_lag_series_by_daytype()",
                     1)[1]
    assert "print(format_schedule_margins(_margins))" in body
    assert body.index("print(format_schedule_margins(_margins))") < \
        body.index("alarm"), "the margin line is printed only when it alarms"


def test_a_breaching_margin_ALARMS_and_names_the_remedy():
    """Positive control: the threshold must actually fire."""
    bad = [(dt.date(2026, 11, 20), 540.0)] * 4        # 9h of delay
    m = ci.schedule_margin(bad, 0)
    assert m["alarm"] is True, m
    assert "DECISION FIRES" in ci.format_schedule_margin(m)
    assert "weekend" in ci.MARGIN_REMEDY, (
        "no remedy on file — the decision fires with no option prepared")


def test_the_fitted_date_NEVER_appears_without_its_R_SQUARED():
    """Rule 3, enforced on the files that cite it.

    `2026-11-06` is a projection from R^2=0.451. Anywhere it is written, the fit
    must be written too, or it reads as a measured deadline.
    """
    for p in (pathlib.Path("scripts/ci_audit.py"),
              pathlib.Path(".github/workflows/daily-picks.yml")):
        text = p.read_text(encoding="utf-8")
        for mo in re.finditer(r"2026-11-06", text):
            window = text[max(0, mo.start() - 400):mo.end() + 400]
            assert "0.451" in window or "R^2" in window, (
                f"{p}: 2026-11-06 is cited without its fit")


# ── C: the tool and the documented rule cannot diverge again ────────────────

def test_cohort_status_has_ONE_verdict_and_it_is_BUMP():
    """s5.14 retired AMEND. The tool must not offer it."""
    src = pathlib.Path("scripts/cohort_status.py").read_text(encoding="utf-8")
    code = "\n".join(l for l in src.splitlines()
                     if not l.strip().startswith("#"))
    code = code.split('"""', 2)[-1]                  # drop the module docstring
    assert 'print("VERDICT: BUMP")' in code
    assert 'VERDICT: AMEND' not in code, (
        "cohort_status still prints AMEND, which s5.14 retired: the count it "
        "keys on is mutable between the check and the next pick")


def test_cohort_status_reads_the_DEPLOYED_config():
    """The local `config/config.yaml` is gitignored and carries no authority."""
    src = pathlib.Path("scripts/cohort_status.py").read_text(encoding="utf-8")
    assert 'Config("config/config.example.yaml")' in src
    assert 'Config("config/config.yaml")' not in src, (
        "the cohort tool fingerprints the gitignored local config — the exact "
        "Stage 10.1 defect, in the tool that gates cohort decisions")


def test_the_tools_vocabulary_matches_the_DOCUMENTED_rule():
    """THE PIN. The two cannot drift apart silently again.

    `experiment_pins.py` carries the retirement in prose; this asserts the
    executable agrees with it.
    """
    pins = pathlib.Path("tests/experiment_pins.py").read_text(encoding="utf-8")
    assert "amend-while-empty rule is RETIRED" in pins, (
        "the retirement is no longer documented — if it was un-retired, "
        "cohort_status must offer AMEND again and this test must change with it")
    tool = pathlib.Path("scripts/cohort_status.py").read_text(encoding="utf-8")
    assert "ALWAYS BUMP" in tool.upper()


# ── D: the two cron series are never pooled ─────────────────────────────────

def test_membership_is_by_the_runs_OWN_sha_not_by_date():
    src = pathlib.Path("scripts/ci_audit.py").read_text(encoding="utf-8")
    body = src.split("def collect_lag_series", 1)[1].split("\ndef ", 1)[0]
    assert "headSha" in body
    assert "390a4be" in body, (
        "the series no longer records WHY membership is by sha — a 09:37 run "
        "read as an 11h30m outlier of a 03:00 series")


def test_a_run_under_a_DIFFERENT_cron_is_excluded_not_rescaled():
    """The `390a4be` rule, as code rather than as a note.

    Caught PROSPECTIVELY this time: the 03:00 population closed at n=32 and the
    00:00 population opened, and the guard was written before the two could be
    pooled rather than after a figure had been published from the mixture.
    """
    src = pathlib.Path("scripts/ci_audit.py").read_text(encoding="utf-8")
    body = src.split("def collect_lag_series", 1)[1].split("\ndef ", 1)[0]
    assert "if by_sha[sha] != cron_now" in body and "continue" in body
    assert "NOT comparable" in body


def test_current_cron_minutes_refuses_a_MULTI_cron_workflow():
    """Two crons mean nearest-preceding attribution, which CENSORS the delay —
    the `closing-lines` defect. Refuse rather than pick one."""
    assert ci.current_cron_minutes() is None, (
        "the single-cron reader resolved a value for a SPLIT workflow - "
        "that is nearest-preceding attribution, which censors the delay")
    # The refusal is now EXERCISED BY PRODUCTION rather than hypothetical,
    # and refusing is no longer sufficient: a day type is exactly the axis
    # on which the two crons do NOT compete, because the week is
    # partitioned and no run can have both as its nearest predecessor.
    by_type = ci.current_crons_by_daytype()
    # THEY DIVERGE AGAIN FROM 2026-10-22 (`s5.19`): weekend 00:30, weekday
    # 01:00. They were both 0 from 2026-10-18 after `0 4` measured a -241m
    # binding margin, and the split survived as two entries even while the
    # minutes were equal -- because the series are keyed by DAY TYPE, not by
    # cron minute, so equal minutes never pooled them. Now each is the latest
    # 15-minute boundary clearing the registered 60 against its OWN localised
    # floor and its OWN slow-cluster spread, and the weekend's is the earlier
    # one because its floor is (12:00 local against 12:30).
    assert by_type == {"weekend": 30, "weekday": 60}, by_type
    assert ci.current_cron_minutes() is None, (
        "the single-cron reader resolved a value even though BOTH crons sit "
        "at minute 0 - equal minutes are still two crons, and the reader must "
        "refuse a multi-cron file on its shape, not on its values")
    assert ci.MARGIN_TRAILING_DAYS == 28


# ── D: the cohort tool reads only TRACKED inputs ────────────────────────────
#
# The tool that gates cohort decisions fingerprinted `config/config.yaml` — a
# gitignored file. Same root as this week's test-skip-count surprise: an
# UNTRACKED FILE PARTICIPATING IN A DECISION THAT MUST BE REPRODUCIBLE.
#
# The retrospective question cannot be answered: an untracked file has no
# history, so what it contained at each of the 15 recorded bumps is
# unrecoverable. Today it is sha256-identical to the example, so the answer is
# PROBABLY none — which is luck, already recorded as luck, and not a check.
# From s5.15 onward the question cannot arise, and that is what these pin.

def _in_committed_tree(path: str) -> bool:
    """Is `path` in HEAD's tree?

    NOT `git ls-files --error-unmatch`, which the suite's own
    `test_no_check_scopes_itself_on_staged_files_alone` rejects: that reads the
    INDEX, so it answers differently before and after `git add` at the same
    commit. Caught by that test while writing this one — second time this class
    has bitten me.

    `HEAD:` is also the right question rather than merely the compliant one. A
    cohort verdict has to be reproducible from a COMMIT, so "is this file in the
    committed tree" is what makes the bump auditable; "is it staged" does not.

    `encoding="utf-8"` because `text=True` alone decodes with the platform codec,
    and on a cp1251 console that leaves `.stdout` as None — which reads as an
    empty result, which reads as a clean finding.
    """
    import subprocess
    r = subprocess.run(["git", "cat-file", "-e", f"HEAD:{path}"],
                       capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    return r.returncode == 0


def test_cohort_status_opens_only_TRACKED_files():
    """Every path the tool names must be in git.

    Read from the source rather than by executing it, so the assertion covers
    the code path taken on any machine rather than the one taken here.
    """
    src = pathlib.Path("scripts/cohort_status.py").read_text(encoding="utf-8")
    paths = set(re.findall(r'["\'](config/[^"\']+|data/[^"\']+)["\']', src))
    assert paths, "no config path found — did the tool stop reading one?"
    for p in sorted(paths):
        assert _in_committed_tree(p), (
            f"cohort_status.py reads {p!r}, which is NOT tracked by git. A "
            f"cohort verdict computed from an untracked file is not "
            f"reproducible, and the bump it authorises is unauditable")


def test_POSITIVE_CONTROL_an_untracked_config_would_FAIL_that_test():
    """The control: the check must be able to fail.

    `config/config.yaml` is the real untracked file the tool used to read, so it
    is the honest negative case — not a fabricated path that git would reject
    for any reason.
    """
    assert pathlib.Path("config/config.yaml").exists() or True
    assert not _in_committed_tree("config/config.yaml"), (
        "config/config.yaml is now TRACKED — then the Stage 10.1 defect has "
        "been reintroduced from the other side: a local convenience file has "
        "become a specification")
    assert _in_committed_tree("config/config.example.yaml"), (
        "the deployed config is not tracked, so nothing the tool reads is")


def test_the_env_loader_reads_dotenv_which_is_NOT_a_fingerprint_input():
    """`.env` is untracked and read by this tool — and that is fine.

    It supplies DATABASE_URL, which selects WHERE the count comes from, not WHAT
    the fingerprint is. The distinction is the whole point of the test above:
    untracked inputs may not feed the FINGERPRINT; they may feed the connection.
    """
    src = pathlib.Path("scripts/cohort_status.py").read_text(encoding="utf-8")
    assert '".env"' in src
    fp_block = src.split("version = model_version(", 1)[1][:120]
    assert "config.example.yaml" in fp_block
    assert "environ" not in fp_block, (
        "an environment value reached the fingerprint call — env is untracked "
        "and must not decide the cohort label")
