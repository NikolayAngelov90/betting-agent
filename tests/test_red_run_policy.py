"""Stage 14, DEL-2 — daily-picks goes red only when pick generation failed.

Part A found 27 runs audited, 1 BROKEN and 9 DEGRADED, **all 27 reporting
`conclusion: success`**. Every core step in `daily-picks` carries
`continue-on-error: true`, and the failure-check step detected failures and
exited 0 — so GitHub's own failure notifications never fired for the workflow
that produces picks.

THE POLICY, decided narrowly by the operator:

    pick generation failed   ->  job RED
    anything else failed     ->  Telegram + ::error:: annotation, job green

THE MECHANISM MATTERS MORE THAN THE POLICY. `continue-on-error` stays on every
step. Removing it would halt the job at the failure, so `--update-results`, the
second `--settle` and the cache saves would never run. This changes the run's
COLOUR, not its execution.

`outcome` IS THREE-VALUED, AND THIS FILE TESTED TWO (found 2026-09-20). `OK`
below is all-`"success"` and every case overrode one key with `"failure"`. No
test ever passed `"skipped"` — so nothing here covered run 35498465743, where
`Run tests` (the one core step with no continue-on-error) failed, the job
halted, update/settle/picks were SKIPPED, and the script printed
`All critical steps OK`. The step that exists to catch "the day produced
nothing" reported the day as healthy, and DEL-2 never reached its branch
because the branch names `failure` and the picks step had never run.

    failure   it ran and broke
    skipped   an earlier unguarded step failed and it never started
    success   it ran

FAILED and DID NOT RUN are different events with the same consequence for the
reader: no picks. They are counted apart and alerted together, and BOTH turn
the run red when they land on pick generation.

The axis is the thing to remember: `continue-on-error` on a scraper absorbs a
routine, recoverable failure — Flashscore times out, results arrive tomorrow. On
pick generation it converts "the day produced nothing" into a green run. Those
were never the same policy; they had the same setting because nobody had
separated them.
"""

import os
import pathlib
import re
import subprocess
import sys
import tempfile

import pytest
import yaml

WF = pathlib.Path(".github/workflows/daily-picks.yml")
OK = {"O_UPDATE": "success", "O_SETTLE1": "success", "O_PICKS": "success",
      "O_RESULTS": "success", "O_SETTLE2": "success"}


def _steps():
    d = yaml.safe_load(WF.read_text(encoding="utf-8"))
    return list(d["jobs"].values())[0]["steps"]


#: The step's NAME changed on 2026-10-22 ("Alert on critical step failure" ->
#: "Decide the run colour from the critical steps") when its Telegram send was
#: removed and only its colour decision was left. Every test here broke with
#: `StopIteration` on the old name, which is a bad failure mode for a locator:
#: it says nothing about what moved.
#:
#: SO IT IS LOCATED BY WHAT IT DOES. The step that reads the five step
#: outcomes and decides the run's colour is the step whose script indexes
#: `steps["picks (incl. review)"]` -- that is behaviour, and it survives
#: renaming.
_PICKS_KEY = 'steps["picks (incl. review)"]'


def _check_step():
    found = [s for s in _steps() if _PICKS_KEY in (s.get("run") or "")]
    assert len(found) == 1, (
        f"{len(found)} steps decide the run colour from the picks outcome; "
        f"expected exactly 1. Two would mean two policies.")
    return found[0]


def _run(outcomes):
    """Run the workflow's own failure-check script with simulated outcomes."""
    script = re.search(r"<<'EOF'\n(.*?)\n\s*EOF",
                       _check_step()["run"], re.S).group(1)
    path = tempfile.mktemp(suffix=".py")
    pathlib.Path(path).write_text(script, encoding="utf-8")
    # Built by update, not dict(**OK, **outcomes): a case that overrides a
    # key in OK is the whole point, and dict() rejects the duplicate keyword.
    env = dict(os.environ)
    env.update(RUN_URL="http://example/run",
               TELEGRAM_BOT_TOKEN="", TELEGRAM_CHAT_ID="",
               # Linux CI already sets this; a Windows console does not, and
               # the alert text contains an emoji.
               PYTHONIOENCODING="utf-8")
    env.update(OK)
    env.update(outcomes)
    r = subprocess.run([sys.executable, path], capture_output=True,
                       text=True, encoding="utf-8", errors="replace",
                       env=env)
    return r.returncode, (r.stdout + r.stderr)


def test_pick_generation_failure_makes_the_run_red():
    """The one case the operator chose. A green run here would mean nothing."""
    code, out = _run({"O_PICKS": "failure"})
    assert code == 1, (
        "pick generation failed and the job still reports success — the run "
        "produced no picks and nothing in the Actions list says so")
    assert "::error::" in out
    assert "step states: step(s) FAILED" in out, (
        "the step states are no longer printed. The Telegram send was "
        "removed 2026-10-22 and this LOG LINE is what replaces it, so "
        "losing the line loses the information entirely")


def test_a_scraper_failure_leaves_the_run_green_but_alerts():
    """Routine and recoverable. Red here would train people to ignore red."""
    code, out = _run({"O_RESULTS": "failure"})
    assert code == 0, (
        "a scraper failure turned the run red — that is the noise this policy "
        "was deliberately narrowed to avoid")
    assert "step states: step(s) FAILED" in out, (
        "the run stayed green AND said nothing — the worst of both")
    # ── WHAT THIS USED TO ASSERT, AND WHY IT WAS WRONG. ──────────────────
    #
    # It asserted `"::error::" in out`. That annotation came from
    # `alert_delivery._annotate`, which fires only when a Telegram delivery
    # FAILS -- and this harness sets `TELEGRAM_BOT_TOKEN=""`. So the test was
    # pinning behaviour that existed ONLY because its own fixture had no
    # credentials; in production, with a token, the send succeeded and no
    # annotation was ever printed.
    #
    # The real carrier for a green run with a failed scraper is the AUDIT,
    # which reads this very line. So that is what is pinned now -- and it
    # catches the coupling that nearly went unnoticed: the audit's regex
    # required an em dash, and normalising it here would have emptied
    # `steps_failed` with no test failing.
    import scripts.ci_audit as _ci
    assert _ci.extract(out)["steps_failed"] == ["update-results"], (
        "the audit can no longer read the step names out of this line, so "
        "the only remaining carrier for a green-but-degraded run is gone")


def test_settlement_failure_is_currently_green_and_that_is_a_known_choice():
    """Pinned so the next decision starts from evidence, not from scratch.

    A failed settlement means picks go ungraded, which degrades the record
    silently. The same argument that justified failing on pick generation
    applies — it was considered and NOT chosen, and is the obvious next
    candidate. If this test starts failing, someone extended the policy; update
    the ledger entry rather than deleting the test.
    """
    code, _ = _run({"O_SETTLE2": "failure"})
    assert code == 0


def test_all_success_is_silent():
    code, out = _run({})
    assert code == 0
    assert "step states:" not in out


# ── SKIPPED: the third state, and the one that actually happened ──────────

def test_a_SKIPPED_picks_step_makes_the_run_red():
    """No picks were produced. How the step failed to produce them is detail.

    This is the case the file did not have on 2026-09-20, and its absence is
    why `All critical steps OK` was printed on a day with zero picks.
    """
    code, out = _run({"O_PICKS": "skipped"})
    assert code == 1, (
        "the picks step never ran and the job still reports success — the "
        "run produced no picks and nothing in the Actions list says so")
    assert "DID NOT RUN" in out, out
    assert "::error::" in out


def test_the_REAL_2026_09_20_outcomes_are_not_reported_as_healthy():
    """Run 35498465743, replayed exactly as GitHub reported it.

    `Run tests` failed, so everything above the `always()` barrier was
    skipped and everything below it ran. Pinned with the real values rather
    than a constructed case, because the constructed cases all passed.
    """
    code, out = _run({"O_UPDATE": "skipped", "O_SETTLE1": "skipped",
                      "O_PICKS": "skipped", "O_RESULTS": "success",
                      "O_SETTLE2": "success"})
    assert "All critical steps OK" not in out, (
        "the day produced no picks and the alert called it OK — this is the "
        "exact output of run 35498465743")
    assert code == 1
    assert "update" in out and "picks (incl. review)" in out


def test_a_skipped_scraper_alerts_but_stays_green():
    """Same axis as a failed scraper: recoverable, so loud but not red."""
    code, out = _run({"O_RESULTS": "skipped"})
    assert code == 0
    assert "DID NOT RUN" in out, out


def test_FAILED_and_DID_NOT_RUN_are_reported_apart_in_one_alert():
    """Summing them would lose which steps crashed and which never started."""
    code, out = _run({"O_UPDATE": "failure", "O_PICKS": "skipped"})
    assert "step(s) FAILED — update" in out, out
    assert "DID NOT RUN — picks (incl. review)" in out, out
    assert code == 1


def test_an_outcome_that_is_none_of_the_three_is_reported_as_itself():
    """`cancelled` at the 360-minute cap is not success, failure or skipped.

    Folding it into either would be the same collapse one level down, so it
    is named and surfaced rather than classified.
    """
    code, out = _run({"O_UPDATE": "cancelled"})
    assert "UNEXPECTED state" in out, out
    assert "cancelled" in out
    assert code == 0, "a cancelled update is not pick generation"


def test_continue_on_error_is_still_set_on_every_core_step():
    """The colour changed; the execution must not.

    Removing `continue-on-error` would halt the job at the failing step, so
    `--update-results`, the second `--settle` and the cache saves would never
    run. A red run that skips its database save is strictly worse than a green
    one that does not.
    """
    guarded = [s.get("name", "") for s in _steps()
               if s.get("continue-on-error") is True]
    # 9 -> 10 on 2026-10-09 (Stage 28). The tenth is the CALENDAR step, and it
    # carries the flag for the opposite reason to the other nine: they must not
    # halt the job so that `--update-results`, the second `--settle` and the
    # cache saves still run; this one must not halt the job so that PRICE
    # CAPTURE still runs. Same mechanism, and it is declared here rather than
    # absorbed into the number, because this guard exists so a step cannot LOSE
    # the flag silently — which means a step gaining it must be named.
    # 10 -> 11 on 2026-10-18. THE ELEVENTH IS THE LOSS REPORT, and it carries
    # the flag for a third reason: a failure to REPORT a failure must not
    # change the run's verdict. It reads the card from the database to say how
    # many fixtures were lost, so it can fail on its own (a database hiccup),
    # and if that halted the job it would convert an alert problem into a
    # pipeline problem — on the step whose entire purpose is to describe a
    # pipeline problem. Named here rather than absorbed into the number,
    # because this guard exists so a step cannot LOSE the flag silently.
    # 11 -> 10 on 2026-10-21, AND IT IS A REMOVAL, NOT A LOSS. The calendar
    # step LEFT this workflow for `.github/workflows/reminders.yml`: its flag
    # stopped it gating capture but not REDDENING the run, so on 2026-10-21
    # daily-picks concluded `failure` with all five critical steps green. A
    # separate workflow cannot do either. The flag now lives on the reminder
    # step there, for its own reason (the log must still upload), and
    # `test_calendar_separation` asserts it.
    # 10 -> 11 on 2026-10-22, AND IT IS AN ADDITION, NAMED AS THIS GUARD
    # REQUIRES. `Check the MODEL, not the retrain step` gained the flag for
    # the same reason as the loss report: it is a DETECTOR, and a detector
    # that reddens the run would put a stale model on the same axis as a lost
    # card. DEL-2 reserves this workflow's red for "the day produced no
    # picks", and a stale model still produced picks -- from Poisson + Elo,
    # which is precisely why it needs reporting rather than a colour. It also
    # reads and writes files under `data/models`, so it can fail on its own
    # without that meaning anything about the pipeline.
    assert len(guarded) == 11, (
        f"{len(guarded)} steps carry continue-on-error, expected 11 — if a step "
        "lost it, the job now HALTS there instead of merely reporting red; if "
        "one gained it, name it here and say why")
    assert "Check the MODEL, not the retrain step" in guarded, (
        "the model-freshness detector lost continue-on-error, so a stale "
        "model now reddens a run that produced a full card")
    assert not any("calendar" in n for n in guarded), (
        "a calendar step is back inside the capture workflow; the separation "
        "is across WORKFLOWS now, not across steps")
    import yaml as _yaml
    _rem = _yaml.safe_load(pathlib.Path(
        ".github/workflows/reminders.yml").read_text(encoding="utf-8"))
    _rsteps = [s for j in _rem["jobs"].values() for s in j.get("steps", [])]
    assert any(s.get("id") == "reminders" and s.get("continue-on-error") is True
               for s in _rsteps), (
        "the reminder step lost continue-on-error — its log no longer uploads "
        "and the step that explains the failure never runs")
    assert any("Generate, review, and send picks" in n for n in guarded), (
        "the picks step lost continue-on-error — the job will now halt before "
        "update-results, the second settle, and the cache saves")


def test_every_step_after_the_check_still_runs_when_it_exits_nonzero():
    """The ordering constraint that would silently defeat this.

    A step that exits non-zero causes later steps to be skipped unless they
    carry `if: always()`. Verified rather than assumed: a red run that skips
    `Save ML models cache` would lose the retrained model every time it fired.
    """
    steps = _steps()
    idx = next(i for i, s in enumerate(steps)
               if _PICKS_KEY in (s.get("run") or ""))
    for s in steps[idx + 1:]:
        cond = str(s.get("if", ""))
        assert "always()" in cond or "failure()" in cond, (
            f"step {s.get('name')!r} runs after the failure check with "
            f"if={cond!r} — it will be SKIPPED on a red run. Give it "
            "`if: always()` or move the check after it.")
