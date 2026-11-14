"""The first-run check, driven on lines in the format the run log really has.

`scripts/first_run_check.py` verifies a picks run under `confident_value`.
Its first version filtered log lines on the STEP NAME, and `gh run view
--log` labels the picks step's lines `UNKNOWN STEP` -- so it found no forced
pick on 2026-11-14, a run that made sixteen. A matcher with zero hits
verified nothing. These tests use that exact line shape, and every PASS here
sits beside a case that must FAIL.
"""
import pytest

from scripts import first_run_check as frc

STEP = "daily-picks\tUNKNOWN STEP\t2026-11-14T07:20:43.5374400Z "


def _rt(msg, module="__main__:analyze_fixture:1293", level="INFO"):
    return (STEP + f"2026-11-14 07:20:43 | {level:8s} | {module} - {msg}")


FORCED = _rt("Club pick (coverage 100%): Exeter vs X -> Home Win @ 1.62")
SHADOW_FORCED = _rt("[standard-rule shadow, NOT a pick] Club pick (coverage "
                    "100%): Exeter vs X -> Home Win @ 1.62")
REVIEW = _rt("Claude reviewed 28/28 match(es), 0 failed",
             module="src.reporting.match_briefing:finalize_picks_with_claude")
TEST_ID = ("daily-picks\tRun tests\t2026-11-14T07:00:00Z tests/"
           "test_pick_review.py::test_finalize_empty_picks PASSED")
ENV = "daily-picks\tUNKNOWN STEP\t2026-11-14T07:00:00Z   ANTHROPIC_API_KEY: ***"


def test_a_FORCED_pick_in_the_REAL_line_format_is_FOUND():
    f = frc.log_facts("\n".join([FORCED, FORCED]))
    assert f["forced_count"] == 2


def test_the_SHADOWS_forced_pick_is_NOT_a_forced_pick():
    assert frc.log_facts(SHADOW_FORCED)["forced_count"] == 0


def test_a_REVIEW_line_is_FOUND_and_TEST_IDS_and_ENV_DUMPS_are_not():
    assert frc.log_facts(REVIEW)["review_count"] == 1
    quiet = frc.log_facts("\n".join([TEST_ID, ENV, _rt("ordinary line")]))
    assert quiet["review_count"] == 0 and quiet["runtime_lines"] == 1


def test_a_log_with_NO_RUNTIME_LINE_is_UNEVALUATED_never_a_pass():
    facts = frc.log_facts("\n".join([TEST_ID, ENV]))
    out = frc.evaluate([], [], [], facts)
    assert out["no_forced_picks"]["state"] == frc.UNEVALUATED
    assert out["no_review_call"]["state"] == frc.UNEVALUATED
    assert frc.evaluate([], [], [], frc.log_facts(None))[
        "no_review_call"]["state"] == frc.UNEVALUATED


def _pick(mid, odds, p, cohort=frc.EXPECTED_COHORT, sel="Over 2.5 Goals"):
    return {"match_id": mid, "odds": odds, "p": p, "model_version": cohort,
            "selection": sel, "market": "Over 2.5"}


RUN = [{"outcome": "PICKS_MADE", "picks_saved": 2, "fixtures_on_slate": 9,
        "fixtures_evaluated": 9, "analysis_errors": 0,
        "selection_mode": "confident_value"}]
SHADOW = [{"match_id": 1, "rule": "standard", "selection": "Home Win",
           "odds": 1.55, "p": 0.6},
          {"match_id": 2, "rule": "standard", "selection": "Over 2.5 Goals",
           "odds": 1.80, "p": 0.55}]
SEND = "src.reporting.telegram_bot:_send_chunked:1025"
CARD_SENT = _rt("REPORT_DELIVERY report=daily picks chunks=1 sent=1 "
                "failed=none terminator=yes attempts=1", module=SEND)
LINE_SENT = _rt("REPORT_DELIVERY report=no picks today chunks=1 sent=1 "
                "failed=none terminator=yes attempts=1", module=SEND)
CV = _rt("confident_value: A vs B -> 4 pickable leg(s)")
CLEAN = frc.log_facts("\n".join([CV, CARD_SENT]))
ZERO = frc.log_facts("\n".join([CV, LINE_SENT]))


def test_a_CLEAN_RUN_passes_every_check():
    out = frc.evaluate([_pick(1, 1.74, 0.55), _pick(2, 1.80, 0.56)], RUN,
                       SHADOW, CLEAN)
    assert {k: v["state"] for k, v in out.items()} == {
        k: frc.PASS for k in out}, out


def test_KNOWN_WRONG_the_2026_11_14_old_code_run_FAILS():
    """The real shape of the run that used the old code: cohort 7f9e82,
    prices from 1.50, probabilities from 0.5205, no tables, forced picks."""
    old = [_pick(1, 1.50, 0.5205, cohort="stage5_baseline_20260807.7f9e82"),
           _pick(2, 1.785, 0.70, cohort="stage5_baseline_20260807.7f9e82")]
    out = frc.evaluate(old, [], [], frc.log_facts(FORCED))
    assert out["cohort"]["state"] == frc.FAIL
    assert out["price_floor"]["state"] == frc.FAIL
    assert out["probability_floor"]["state"] == frc.FAIL
    assert out["no_forced_picks"]["state"] == frc.FAIL
    assert out["pick_runs"]["state"] == frc.UNEVALUATED
    assert out["shadow_stored"]["state"] == frc.FAIL


@pytest.mark.parametrize("picks, check", [
    ([_pick(1, 1.69, 0.60)], "price_floor"),
    ([_pick(1, 1.80, 0.539)], "probability_floor"),
    ([_pick(1, 1.80, 0.60), _pick(1, 1.75, 0.58, sel="Home Win")],
     "one_per_fixture"),
    ([_pick(1, 1.80, 0.60, cohort="other.abc123")], "cohort"),
])
def test_each_SELECTION_CHECK_fails_on_its_own_violation(picks, check):
    run = [dict(RUN[0], picks_saved=len(picks))]
    assert frc.evaluate(picks, run, SHADOW, CLEAN)[check]["state"] == frc.FAIL


def test_a_ZERO_PICK_DAY_with_its_RECORD_passes_and_without_is_UNEVALUATED():
    none_q = [{"outcome": "NONE_QUALIFIED", "picks_saved": 0,
               "fixtures_evaluated": 6, "analysis_errors": 0,
               "selection_mode": "confident_value"}]
    out = frc.evaluate([], none_q, SHADOW, CLEAN)
    assert out["price_floor"]["state"] == frc.PASS
    assert out["pick_runs"]["state"] == frc.PASS
    bare = frc.evaluate([], [], [], CLEAN)
    assert bare["price_floor"]["state"] == frc.UNEVALUATED


def test_a_RECORD_that_DISAGREES_with_the_picks_FAILS():
    wrong = [dict(RUN[0], outcome="NONE_QUALIFIED", picks_saved=0)]
    out = frc.evaluate([_pick(1, 1.74, 0.55)], wrong, SHADOW, CLEAN)
    assert out["pick_runs"]["state"] == frc.FAIL


# ── THE ZERO-PICK PATH. Stage 33: 2026-11-15 falls in the FIFA window. ──────
#
# A NONE_QUALIFIED day passes only RECORDED and ERROR-FREE, with the client's
# one plain line delivered. Every PASS below sits beside the FAIL it must not
# become.

def _zero(outcome="NONE_QUALIFIED", evaluated=6, errors=0, slate=6):
    return [{"outcome": outcome, "picks_saved": 0, "fixtures_on_slate": slate,
             "fixtures_evaluated": evaluated, "analysis_errors": errors,
             "selection_mode": "confident_value"}]


def test_the_ZERO_PICK_PATH_passes_recorded_error_free_and_DELIVERED():
    out = frc.evaluate([], _zero(), SHADOW, ZERO)
    assert {k: v["state"] for k, v in out.items()} == {
        k: frc.PASS for k in out}, out
    assert "ZERO-PICK path" in out["pick_runs"]["why"]
    assert "slate 6 fixture(s), 6 evaluated" in out["pick_runs"]["why"]


@pytest.mark.parametrize("runs, check", [
    (_zero(errors=1), "pick_runs"),                 # not error-free
    (_zero(outcome="INCOMPLETE", errors=2), "pick_runs"),
    (_zero(outcome="NO_PRICED_FIXTURES"), "pick_runs"),
    (_zero(outcome="INCOMPLETE", errors=2), "client_delivery"),
])
def test_a_ZERO_PICK_day_that_is_NOT_the_designed_path_FAILS(runs, check):
    assert frc.evaluate([], runs, SHADOW, ZERO)[check]["state"] == frc.FAIL


def test_the_WRONG_receipt_or_a_FAILED_send_is_not_a_delivery():
    # a zero-pick day that sent the CARD receipt, not the line
    assert frc.evaluate([], _zero(), SHADOW, CLEAN)[
        "client_delivery"]["state"] == frc.FAIL
    failed = frc.log_facts("\n".join([CV, LINE_SENT.replace(
        "sent=1 failed=none", "sent=0 failed=1")]))
    assert frc.evaluate([], _zero(), SHADOW, failed)[
        "client_delivery"]["state"] == frc.FAIL
    # the workflow's own text naming the marker is not a runtime receipt
    echo = frc.log_facts("\n".join([CV, "daily-picks\tUNKNOWN STEP\t2026-11-"
                                    "15T07:00:00Z grep REPORT_DELIVERY "
                                    "report=no picks today failed=none"]))
    assert echo["delivery"] == []


def test_NO_SHADOW_rows_pass_only_when_NOTHING_was_evaluated():
    none_evaluated = frc.evaluate([], _zero("NO_FIXTURES", evaluated=0,
                                            slate=0), [], ZERO)
    assert none_evaluated["shadow_stored"]["state"] == frc.PASS
    six_evaluated = frc.evaluate([], _zero(), [], ZERO)
    assert six_evaluated["shadow_stored"]["state"] == frc.FAIL
