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


def _check_step():
    return next(s for s in _steps()
                if "Alert on critical step failure" in s.get("name", ""))


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
    assert "Daily picks: step(s) FAILED" in out, "no alert was sent"


def test_a_scraper_failure_leaves_the_run_green_but_alerts():
    """Routine and recoverable. Red here would train people to ignore red."""
    code, out = _run({"O_RESULTS": "failure"})
    assert code == 0, (
        "a scraper failure turned the run red — that is the noise this policy "
        "was deliberately narrowed to avoid")
    assert "Daily picks: step(s) FAILED" in out, (
        "the run stayed green AND said nothing — the worst of both")
    assert "::error::" in out


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
    assert "Daily picks: step(s) FAILED" not in out


def test_continue_on_error_is_still_set_on_every_core_step():
    """The colour changed; the execution must not.

    Removing `continue-on-error` would halt the job at the failing step, so
    `--update-results`, the second `--settle` and the cache saves would never
    run. A red run that skips its database save is strictly worse than a green
    one that does not.
    """
    guarded = [s.get("name", "") for s in _steps()
               if s.get("continue-on-error") is True]
    assert len(guarded) == 9, (
        f"{len(guarded)} steps carry continue-on-error, expected 9 — if a step "
        "lost it, the job now HALTS there instead of merely reporting red")
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
               if "Alert on critical step failure" in s.get("name", ""))
    for s in steps[idx + 1:]:
        cond = str(s.get("if", ""))
        assert "always()" in cond or "failure()" in cond, (
            f"step {s.get('name')!r} runs after the failure check with "
            f"if={cond!r} — it will be SKIPPED on a red run. Give it "
            "`if: always()` or move the check after it.")
