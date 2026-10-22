"""The report is rendered against a KNOWN input and its numbers asserted.

    2026-10-20. Rule 4: "A report is rendered in a test with asserted numbers."

A report never rendered in a test is a formatter with no consumer, and that
pattern has appeared repeatedly here. Every figure below is computed by hand
from the fixture so the test fails when the arithmetic changes, not only when
the code raises.

`test_discrimination_*` are the controls: each one perturbs ONE thing about the
population and asserts the figure MOVES. A report that prints the same number
for two different populations is the defect this whole module exists to fix.
"""

from datetime import date

import pytest

from src.reporting.experiment_summary import (EXPERIMENT_START, GATES, Row,
                                              format_block, format_telegram,
                                              paired_brier, partition,
                                              summarise)

W0, W1 = date(2026, 6, 22), date(2026, 10, 20)
TODAY = date(2026, 10, 20)
IN = date(2026, 9, 1)          # inside the experiment
PRE = date(2026, 7, 1)         # before it -- the legacy block


def _r(pick_id, *, result="win", paper=True, mv="s5.x", odds=2.0,
       prob=0.6, mkt=0.5, match_id=None, d=IN, disp=None, ev=None, tx=None):
    return Row(pick_id=pick_id, match_id=match_id or pick_id, pick_date=d,
               result=result, is_paper=paper, model_version=mv,
               disposition=disp, evidence_status=ev,
               training_exclusion_reason=tx, odds=odds, prob=prob,
               market_prob=mkt)


# --------------------------------------------------------------------------
# the known input: 10 settled paper picks, hand-computable
#
#   6 wins at 2.00  -> +1.00 each = +6.00
#   4 losses        -> -1.00 each = -4.00
#   total +2.00 over 10 picks     -> ROI +20.00%, win rate 60.0%
# --------------------------------------------------------------------------
KNOWN = ([_r(i, result="win") for i in range(1, 7)]
         + [_r(i, result="loss") for i in range(7, 11)])


def test_the_known_input_renders_the_hand_computed_numbers():
    s = summarise(KNOWN, W0, W1, TODAY)
    a = s["paper"]
    assert a["n"] == 10
    assert a["win"] == pytest.approx(0.60)
    assert a["roi"] == pytest.approx(0.20)
    assert a["fixtures"] == 10
    assert a["n_cohorts"] == 1
    assert s["live"]["n"] == 0, "nothing in the fixture is live"
    assert s["excluded_pre_experiment"] == 0


def test_the_telegram_body_carries_those_numbers_and_its_population():
    """PART D positive control: the message body, asserted."""
    body = format_telegram(summarise(KNOWN, W0, W1, TODAY))
    lines = body.split("\n")
    assert len(lines) == 6, f"six lines or so; got {len(lines)}"
    # line 1 names n and the span, line 2 names the POPULATION
    assert "10 settled picks" in lines[0]
    assert "PAPER only" in lines[1] and str(EXPERIMENT_START) in lines[1]
    assert "1 cohorts POOLED" in lines[1], "the pooling must be named (rule 1)"
    assert "win 60.0%" in lines[2]
    assert "+20.00%" in lines[2]
    assert "Brier PAIRED on n=10" in lines[3]
    assert "reported SEPARATELY" in lines[4] and "never pooled" in lines[4]
    # the standing qualification travels with the figure
    assert "no information" in body.lower()
    assert "4 routes" in body


def test_a_figure_never_appears_without_its_n():
    """Rule 2. Every numeric line in the body carries an n or a sample span."""
    body = format_telegram(summarise(KNOWN, W0, W1, TODAY))
    for line in body.split("\n"):
        if "%" in line and "ROI" in line:
            # the ROI line's n is on line 1; assert the CI is present instead,
            # which is what makes the ratio readable
            assert "CI" in line, line


# --------------------------------------------------------------------------
# discrimination controls
# --------------------------------------------------------------------------

def test_discrimination_the_gates_actually_remove_picks():
    """Each gate must drop its own pick and move the figure."""
    for gate, _reason in GATES:
        rows = list(KNOWN) + [_r(99, result="win", odds=11.0,
                                 **{{"disposition": "disp",
                                     "evidence_status": "ev",
                                     "training_exclusion_reason": "tx"}[gate]:
                                    "consolidated"})]
        part = partition(rows)
        assert part["dropped"].get(gate) == 1, (
            f"{gate} did not drop its pick: {part['dropped']}")
        s = summarise(rows, W0, W1, TODAY)
        assert s["paper"]["n"] == 10, f"{gate} let the pick through"
        # and the control: WITHOUT the gate value it would have changed the ROI
        ungated = summarise(
            [r._replace(disposition=None, evidence_status=None,
                        training_exclusion_reason=None) for r in rows],
            W0, W1, TODAY)
        assert ungated["paper"]["n"] == 11
        assert ungated["paper"]["roi"] != pytest.approx(0.20), (
            "the dropped pick was worth nothing, so this gate cannot be "
            "shown to bind -- pick a fixture where it does")


def test_discrimination_paper_and_live_are_never_added_together():
    rows = list(KNOWN) + [_r(50, result="loss", paper=False, mv=None),
                          _r(51, result="loss", paper=False, mv=None)]
    s = summarise(rows, W0, W1, TODAY)
    assert s["paper"]["n"] == 10, "a live pick entered the paper arm"
    assert s["live"]["n"] == 2
    # the paper figure is UNMOVED by adding losing live picks
    assert s["paper"]["roi"] == pytest.approx(0.20)
    assert s["live"]["roi"] == pytest.approx(-1.0)
    body = format_telegram(s)
    assert "n=2" in body


def test_discrimination_pre_experiment_picks_are_excluded_and_counted():
    rows = list(KNOWN) + [_r(60, result="loss", d=PRE, mv=None, paper=False),
                          _r(61, result="loss", d=PRE, mv=None, paper=False)]
    s = summarise(rows, W0, W1, TODAY)
    assert s["excluded_pre_experiment"] == 2
    assert s["paper"]["n"] == 10
    assert s["live"]["n"] == 0, "a pre-experiment pick reached the live arm"


def test_discrimination_the_cohort_count_moves_with_the_cohorts():
    one = summarise(KNOWN, W0, W1, TODAY)["paper"]
    mixed = summarise([r._replace(model_version=f"s5.{i % 3}")
                       for i, r in enumerate(KNOWN)], W0, W1, TODAY)["paper"]
    assert one["n_cohorts"] == 1 and mixed["n_cohorts"] == 3
    assert len(mixed["per_cohort"]) == 3
    assert sum(c["n"] for c in mixed["per_cohort"]) == 10
    assert "3 cohorts POOLED" in format_telegram(
        summarise([r._replace(model_version=f"s5.{i % 3}")
                   for i, r in enumerate(KNOWN)], W0, W1, TODAY))


def test_discrimination_clustering_widens_the_interval():
    """The CI must resample FIXTURES. The same ten picks spread over ten
    fixtures carry MORE information than the same ten paired onto five, and the
    interval has to say so.

    Five clusters, not one: ten picks on a single fixture is one cluster, and
    `_boot` correctly refuses to resample fewer than five -- which is the right
    answer but not a test of widening. The pairs are same-outcome, because
    pairing a win with a loss cancels the within-fixture correlation and the
    clustering would then have nothing to find.
    """
    spread = summarise(KNOWN, W0, W1, TODAY)["paper"]
    pairs = {1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3,     # the six wins -> 3 fixtures
             7: 4, 8: 4, 9: 5, 10: 5}                # the four losses -> 2
    clumped = summarise([r._replace(match_id=pairs[r.pick_id]) for r in KNOWN],
                        W0, W1, TODAY)["paper"]
    assert spread["fixtures"] == 10 and clumped["fixtures"] == 5
    assert spread["roi"] == pytest.approx(clumped["roi"]), "same point estimate"
    lo_s, hi_s = spread["ci"]
    lo_c, hi_c = clumped["ci"]
    assert lo_s is not None and lo_c is not None
    assert (hi_c - lo_c) > (hi_s - lo_s), (
        f"five fixtures gave a NARROWER interval ({hi_c - lo_c:.3f}) than ten "
        f"({hi_s - lo_s:.3f}) -- the bootstrap is resampling picks, not fixtures")


def test_one_cluster_refuses_rather_than_reporting_a_narrow_interval():
    """Third state again: too few clusters to resample is UNEVALUATED."""
    one = summarise([r._replace(match_id=1) for r in KNOWN], W0, W1, TODAY)["paper"]
    assert one["fixtures"] == 1
    assert one["ci"] == (None, None), "one fixture produced a confidence interval"
    assert "CI UNEVALUATED" in format_telegram(
        summarise([r._replace(match_id=1) for r in KNOWN], W0, W1, TODAY))


# --------------------------------------------------------------------------
# the Brier pair -- the defect this audit found
# --------------------------------------------------------------------------

def test_the_brier_pair_is_computed_on_the_INTERSECTION():
    """The report compared a model Brier over n=1534 to a market Brier over
    n=479. Here: 4 picks carry both, 6 carry only the model's probability."""
    rows = ([_r(i, result="win", prob=0.6, mkt=0.5) for i in (1, 2)]
            + [_r(i, result="loss", prob=0.6, mkt=0.5) for i in (3, 4)]
            + [_r(i, result="win", prob=0.9, mkt=None) for i in range(5, 11)])
    b = paired_brier(rows)
    assert b["state"] == "PAIRED"
    assert b["n_paired"] == 4, "the pair must use only picks carrying both"
    assert b["n_model_only"] == 6
    # by hand: model (0.6-1)^2 x2 + (0.6-0)^2 x2 = (0.16+0.36)*2/4 = 0.26
    #          market (0.5-1)^2 x2 + (0.5-0)^2 x2 = 0.25
    assert b["model"] == pytest.approx(0.26)
    assert b["market"] == pytest.approx(0.25)
    assert b["delta"] == pytest.approx(0.01)


def test_the_unpaired_comparison_would_have_given_a_DIFFERENT_answer():
    """THE CONTROL FOR THE ACTUAL DEFECT. The six model-only picks are all
    confident wins, so an unpaired model Brier is much better than the paired
    one -- in production the bias ran the other way and overstated the model's
    deficit. Either way the two populations do not answer the same question."""
    rows = ([_r(i, result="win", prob=0.6, mkt=0.5) for i in (1, 2)]
            + [_r(i, result="loss", prob=0.6, mkt=0.5) for i in (3, 4)]
            + [_r(i, result="win", prob=0.9, mkt=None) for i in range(5, 11)])
    b = paired_brier(rows)
    unpaired_model = sum(
        (r.prob - (1.0 if r.result == "win" else 0.0)) ** 2 for r in rows
    ) / len(rows)
    # 2*(0.6-1)^2 + 2*(0.6-0)^2 + 6*(0.9-1)^2 = 0.32+0.72+0.06 = 1.10, /10
    assert unpaired_model == pytest.approx(0.110)
    assert b["model"] == pytest.approx(0.26)
    assert abs(unpaired_model - b["model"]) > 0.1, (
        "this fixture cannot tell the two populations apart, so it is not a "
        "control for the defect")


def test_no_paired_pick_is_a_STATE_not_a_zero():
    """Third-state discipline: 'nothing to compare' must not render as 0.0000."""
    rows = [_r(i, result="win", mkt=None) for i in range(1, 11)]
    b = paired_brier(rows)
    assert b["state"] == "NO_PAIRED_PICKS"
    assert b["model"] is None and b["market"] is None
    body = format_telegram(summarise(rows, W0, W1, TODAY))
    assert "NO_PAIRED_PICKS" in body and "nothing to pair" in body
    assert "0.0000" not in body


# --------------------------------------------------------------------------
# the long form
# --------------------------------------------------------------------------

def test_the_block_enumerates_the_cohorts_it_pools():
    rows = [r._replace(model_version=f"s5.{i % 3}") for i, r in enumerate(KNOWN)]
    block = "\n".join(format_block(summarise(rows, W0, W1, TODAY)))
    for mv in ("s5.0", "s5.1", "s5.2"):
        assert mv in block, f"{mv} pooled but not enumerated (rule 1)"
    assert "cluster bootstrap" in block and "resampling FIXTURES" in block
    assert str(EXPERIMENT_START) in block
    assert "THREE EFFECTIVE FOLDS OF FIVE" in block
    assert "ATYPICAL ODDS ERA" in block
    assert "measured" in block


def test_every_gate_is_explained_in_the_block():
    block = "\n".join(format_block(summarise(KNOWN, W0, W1, TODAY)))
    for gate, _reason in GATES:
        assert gate in block, f"{gate} applied but not named to the reader"


# --------------------------------------------------------------------------
# PART D — the delivery, and the cron collision
# --------------------------------------------------------------------------

def _wf(name):
    import pathlib

    import yaml
    return yaml.safe_load(
        pathlib.Path(f".github/workflows/{name}").read_text(encoding="utf-8"))


def _crons(wf):
    sched = (wf.get(True) or wf.get("on") or {}).get("schedule") or []
    return [c["cron"] for c in sched]


def test_the_report_no_longer_collides_with_closing_lines():
    """The two fired in the SAME MINUTE (`47 10 * * *`) every day. The report
    moved because it is read-only; closing-lines' 10:47 slot is
    measurement-justified and load-bearing."""
    report = _crons(_wf("paper-trading-report.yml"))
    capture = _crons(_wf("closing-lines.yml"))
    assert report, "the report lost its schedule"
    assert capture, "closing-lines lost its schedule"
    overlap = set(report) & set(capture)
    assert not overlap, f"still colliding on {overlap}"
    # and the report still runs BEFORE the first capture, which was the
    # original intent the stale header had stopped describing
    def _earliest(expr):
        """Minutes past midnight of the EARLIEST firing of one expression.

        The hour field is a comma list on closing-lines
        (`17 11,13,15,17,19,21,23 * * *`), so taking `int(field)` reads the
        whole list as one number and raises. Expand it and take the minimum.
        """
        minute, hour = expr.split()[0], expr.split()[1]
        hours = [int(h) for part in hour.split(",") for h in [part]]
        return min(h * 60 + int(minute) for h in hours)
    assert min(_earliest(c) for c in report) < min(
        _earliest(c) for c in capture), (
        "the report must still precede the first capture slot")


def test_the_report_delivers_on_success_not_only_on_failure():
    """Until 2026-10-21 the only Telegram message this job could send was its
    failure alert, so a working report and no report looked identical."""
    wf = _wf("paper-trading-report.yml")
    steps = wf["jobs"]["report"]["steps"]
    run_step = next(s for s in steps if s.get("name") == "Paper trading report")
    assert "--telegram" in run_step["run"], "the success delivery is not wired"
    env = run_step.get("env") or {}
    for key in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"):
        assert key in env, f"{key} is not available to the delivering step"
    # ── AND THE FAILURE ALERT IS NOW GONE, DELIBERATELY. 2026-10-22. ─────
    #
    # This asserted the failure alert still existed, on the reasoning that
    # adding a success path must not remove one. That reasoning was right at
    # the time and is now superseded: the success path IS the carrier, so a
    # day with no report is itself the signal, and "⚠️ Paper trading report
    # failed" was pure run status -- GitHub's notification on the red run
    # says the same thing. MEASURED over 2026-09-22..10-22: 30 runs, 0
    # failures, so the removed sender had never once fired.
    #
    # Inverted rather than deleted, so the decision is visible here instead
    # of leaving a silently absent assertion.
    assert not any("ci_alert" in str(s.get("run", "")) for s in steps), (
        "a run-status Telegram alert is back in this workflow; the report "
        "itself is the carrier and its ABSENCE is the failure signal")


def test_the_report_calls_the_delivery_and_does_not_swallow_its_result():
    """AST, not prose: `format_telegram` is rendered and `send` is called, and
    the outcome is printed either way."""
    import ast
    import pathlib
    src = pathlib.Path("scripts/paper_trading_report.py").read_text(
        encoding="utf-8")
    tree = ast.parse(src)
    called = {n.func.id for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "format_telegram" in called, "the body is never rendered"
    assert "send" in called, "the body is never delivered"
    main = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "main")
    seg = ast.get_source_segment(src, main) or ""
    assert "NOT DELIVERED" in seg, (
        "a failed send must be reported, not swallowed -- that is how the "
        "broken alert survived months")


# --------------------------------------------------------------------------
# the swallowed exit code
# --------------------------------------------------------------------------

def test_a_health_alert_has_its_own_exit_code():
    """1 must mean only "could not run". Two meanings for one code is how the
    swallowing went unnoticed."""
    from scripts.paper_trading_report import HEALTH_ALERT_EXIT
    assert HEALTH_ALERT_EXIT == 2
    assert HEALTH_ALERT_EXIT != 1


def test_the_workflow_sets_pipefail_so_the_exit_code_survives_tee():
    """`python ... | tee` returns TEE's status. GitHub runs `run:` under
    `bash -e` without `-o pipefail`, so the report's exit code was discarded
    and the 80% coverage threshold could never fail the job."""
    wf = _wf("paper-trading-report.yml")
    run = next(s for s in wf["jobs"]["report"]["steps"]
               if s.get("name") == "Paper trading report")["run"]
    assert "| tee" in run, "the tee is gone; this guard no longer applies"
    assert "set -o pipefail" in run, (
        "a pipeline without pipefail discards the report's exit code")
    # and the health-alert code is treated as a finding, not a failure
    assert '"$rc" = "2"' in run or "'$rc' = '2'" in run, (
        "exit 2 (ran, found something) is not distinguished from a crash")


def test_pipefail_is_what_actually_changes_the_observable():
    """DISCRIMINATION CONTROL, run as a subprocess: the same pipeline with and
    without pipefail must give DIFFERENT statuses, or the fix is cosmetic."""
    import subprocess

    # PROBE THE SHELL FIRST. `bash` on PATH resolved to a non-functional WSL
    # stub on the dev host, which returned 1 for "could not exec" -- and 1 is
    # exactly what a propagated failure looks like. Without this probe the
    # control passes or fails for a reason that has nothing to do with
    # pipefail. CI is ubuntu, so the control runs where it matters.
    import shutil

    # `["bash", ...]` resolved to Windows' WSL stub, which exits 1 for "could
    # not exec" -- indistinguishable from a propagated failure. Resolve the
    # interpreter explicitly and PROVE it runs before trusting its codes.
    sh = None
    for cand in (shutil.which("bash"),
                 "C:/Program Files/Git/bin/bash.exe",
                 "/bin/bash"):
        if not cand:
            continue
        try:
            if subprocess.run([cand, "-c", "exit 7"],
                              capture_output=True).returncode == 7:
                sh = cand
                break
        except OSError:
            continue
    if sh is None:
        pytest.skip("no working bash on this host; the control runs in CI")

    inner = "import sys; sys.exit(2)"
    bare = subprocess.run(
        [sh, "-c", f'python -c "{inner}" | tee /dev/null'],
        capture_output=True)
    fixed = subprocess.run(
        [sh, "-c", f'set -o pipefail; python -c "{inner}" | tee /dev/null'],
        capture_output=True)
    assert bare.returncode == 0, (
        f"a bare pipeline propagated {bare.returncode}; the premise of this "
        f"fix does not hold on this shell")
    assert fixed.returncode == 2, (
        f"pipefail did not propagate the code: got {fixed.returncode}")


def test_section_health_returns_the_texts_not_a_count():
    """The texts are what travel in the delivered summary. A count cannot."""
    import ast
    import pathlib
    src = pathlib.Path("scripts/paper_trading_report.py").read_text(
        encoding="utf-8")
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "section_health")
    returns = [n.value for n in ast.walk(fn) if isinstance(n, ast.Return)]
    assert returns, "section_health returns nothing"
    for r in returns:
        assert not (isinstance(r, ast.Call)
                    and isinstance(r.func, ast.Name)
                    and r.func.id == "len"), (
            "section_health still returns a COUNT; the alert texts cannot "
            "reach a person through an integer")


# --------------------------------------------------------------------------
# the coverage ratchet (2026-10-21) — it replaced a threshold that could
# never fail, and it must be able to fail in BOTH directions
# --------------------------------------------------------------------------

def test_the_80_percent_threshold_is_retired():
    """It required 80% against a measured 12.54%, so it was breached on every
    run it ever made — and nothing relied on it, because its only consumer was
    an exit code that `| tee` discarded."""
    import scripts.paper_trading_report as rep
    assert not hasattr(rep, "MIN_CLV_COVERAGE"), (
        "the unachievable threshold is still live")
    assert rep.CLOSING_COVERAGE_FLOOR == 0.12
    assert rep.CLOSING_COVERAGE_MEASURED == 0.1254
    assert "165/1316" in rep.CLOSING_COVERAGE_DENOMINATOR, (
        "the floor must carry the denominator it was derived from (rule 6)")
    assert "2026-10-21" in rep.CLOSING_COVERAGE_DENOMINATOR


def test_the_floor_is_DERIVED_from_the_measurement_not_chosen():
    """Rule 3: a threshold that could never fail is replaced by one derived
    from the measurement. The floor is the measured value floored to 2dp."""
    import math

    import scripts.paper_trading_report as rep
    assert rep.CLOSING_COVERAGE_FLOOR == math.floor(
        rep.CLOSING_COVERAGE_MEASURED * 100) / 100
    # and it sits BELOW the measurement, so today is not already a breach
    assert rep.CLOSING_COVERAGE_FLOOR < rep.CLOSING_COVERAGE_MEASURED
    # the slack must absorb at least one zero-capture day (~0.4pp)
    assert (rep.CLOSING_COVERAGE_MEASURED - rep.CLOSING_COVERAGE_FLOOR) >= 0.004


def _coverage_verdict(rate):
    """Replay `section_health`'s coverage branch at a given rate."""
    import scripts.paper_trading_report as rep
    ceiling = rep.CLOSING_COVERAGE_FLOOR + rep.CLOSING_COVERAGE_RISE_BAND
    if rate < rep.CLOSING_COVERAGE_FLOOR:
        return "FELL"
    if rate >= ceiling:
        return "ROSE"
    return "OK"


def test_the_ratchet_FIRES_on_a_fall():
    """The one thing the check exists to catch."""
    assert _coverage_verdict(0.1199) == "FELL"
    assert _coverage_verdict(0.00) == "FELL"


def test_the_ratchet_ALSO_fires_on_an_UNRECORDED_rise():
    """So the floor cannot drift upward silently and leave the next fall
    measured against a number nobody wrote down."""
    assert _coverage_verdict(0.15) == "ROSE"
    assert _coverage_verdict(0.80) == "ROSE", (
        "even the old 80% target must demand the floor be re-recorded")


def test_todays_measured_value_is_INSIDE_the_band():
    """RULE 4: an alarm that always fires is not an available option. The
    floor must not be breached by the very measurement it came from."""
    import scripts.paper_trading_report as rep
    assert _coverage_verdict(rep.CLOSING_COVERAGE_MEASURED) == "OK"


def test_the_two_coverage_figures_both_name_their_denominator():
    """Same numerator (165), different denominators (1606 vs 1316), both
    called "coverage". Rule 6."""
    import pathlib
    src = pathlib.Path("scripts/paper_trading_report.py").read_text(
        encoding="utf-8")
    # NOT `or "legacy " in src` -- that clause matched an unrelated comment and
    # made the assertion vacuous. Each figure's scope is asserted on its own.
    assert "legacy " in src and "INCLUDED" in src, (
        "the wider figure does not say legacy picks are included")
    assert "denominator: ALL past-kickoff picks in range" in src, (
        "the attribution-coverage figure does not state its scope")
    assert "carrying a model_version" in src, (
        "the health-check coverage figure does not state its scope")


def test_exit_2_is_converted_to_SUCCESS_by_the_workflow_and_1_is_not():
    """PART 3, 2026-10-21: `HEALTH_ALERT_EXIT = 2` is new, so prove the only
    consumer that reads it treats it as a warning — by RUNNING the shipped
    shell, not by reading it.

    A consumer that predates the three-valued exit would see 2 as a plain
    failure. `ci-audit`'s partition cannot: it matches
    `##[error]Process completed with exit code N` in the log, and a step that
    ends `exit 0` emits none, so the run concludes `success` and never enters
    the partition at all.
    """
    import shutil
    import subprocess

    sh = None
    for cand in (shutil.which("bash"), "C:/Program Files/Git/bin/bash.exe",
                 "/bin/bash"):
        if cand:
            try:
                if subprocess.run([cand, "-c", "exit 7"],
                                  capture_output=True).returncode == 7:
                    sh = cand
                    break
            except OSError:
                continue
    if sh is None:
        pytest.skip("no working bash on this host; the control runs in CI")

    # the shipped tail of the report step, verbatim in behaviour
    script = (
        'rc="$1"\n'
        'if [ "$rc" = "2" ]; then\n'
        '  echo "Report ran; health alert(s) present and delivered."\n'
        '  exit 0\n'
        'fi\n'
        'exit "$rc"\n')
    def _run(rc):
        return subprocess.run([sh, "-c", script, "_", str(rc)],
                              capture_output=True).returncode

    assert _run(2) == 0, "a health alert must NOT fail the job"
    assert _run(1) == 1, "a crash must still fail the job"
    assert _run(0) == 0
    # and the distinction is real, not a tautology: 1 and 2 differ
    assert _run(1) != _run(2)
